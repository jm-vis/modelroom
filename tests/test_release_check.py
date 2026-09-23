"""scripts/release-check.py against real, throwaway git repositories (git init, no mocks).

Every leak planted below is assembled at runtime from pieces, so this file itself stays clean
under the public hygiene check that scans it like every other tracked file.
"""

import importlib.util
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "release-check.py"
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the release gate is tested from a clone", allow_module_level=True)
_SPEC = importlib.util.spec_from_file_location("release_check", SCRIPT)
rc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rc)

LISTS = rc.load_negative_list()
LEAK_ADDRESS = "someone" + "@" + "example.org"
LEAK_PATH = "/ho" + "me/someone/notes.md"
ENTRY_POINTS = "[console_scripts]\nmodelroom = modelroom.cli:main\n"


def git(repo: Path, *args: str) -> str:
    base = ["git", "-c", "user.name=tester", "-c", "user.email=tester", "-c", "commit.gpgsign=false",
            "-c", "tag.gpgsign=false", "-c", f"core.hooksPath={repo / '.no-hooks'}"]
    return subprocess.run(base + list(args), cwd=repo, capture_output=True, text=True, check=True).stdout


def commit_all(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def make_repo(tmp_path: Path, version: str = "0.1.0", changelog_heading: str = "## [0.1.0] - 2026-09-23") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    (repo / "pyproject.toml").write_text(f'[project]\nname = "demo"\nversion = "{version}"\n', encoding="utf-8")
    (repo / "CHANGELOG.md").write_text(f"# Changelog\n\n## [Unreleased]\n\n{changelog_heading}\n\n- first\n",
                                       encoding="utf-8")
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    commit_all(repo, "feat: first")
    return repo


def source(repo: Path, tag: str | None = None) -> list[str]:
    return rc.source_findings(repo, "HEAD", tag, LISTS)


def test_clean_repository_has_no_source_findings(tmp_path):
    assert source(make_repo(tmp_path)) == []


def test_dirty_worktree_is_a_finding(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "scratch.txt").write_text("x\n", encoding="utf-8")
    assert any(finding.startswith("worktree:") for finding in source(repo))


def test_version_without_changelog_section_is_a_finding(tmp_path):
    repo = make_repo(tmp_path, changelog_heading="## [0.0.9] - 2026-09-01")
    assert any("CHANGELOG.md" in finding and "0.1.0" in finding for finding in source(repo))


def test_invalid_version_is_a_finding(tmp_path):
    repo = make_repo(tmp_path, version="one.two", changelog_heading="## [one.two] - 2026-09-23")
    assert any("pyproject.toml" in finding and "PEP 440" in finding for finding in source(repo))


def test_leak_in_a_file_of_the_ref_names_place_and_class_but_not_the_text(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text(f"# notes\nwrite to {LEAK_ADDRESS}\n", encoding="utf-8")
    commit_all(repo, "docs: notes")
    findings = source(repo)
    assert "ref:notes.md:2: e-mail address" in findings
    assert not any(LEAK_ADDRESS in finding for finding in findings)


def test_leak_only_in_history_is_found_although_the_file_is_gone(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text(f"see {LEAK_PATH}\n", encoding="utf-8")
    commit_all(repo, "docs: notes")
    (repo / "notes.md").unlink()
    commit_all(repo, "docs: drop notes")
    findings = source(repo)
    assert not any(finding.startswith("ref:") for finding in findings)
    history = [finding for finding in findings if finding.startswith("history:")]
    assert len(history) == 1
    assert ":notes.md:1: absolute user path" in history[0]
    assert LEAK_PATH not in history[0]


def test_leak_only_in_a_commit_message_is_a_meta_finding(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    commit_all(repo, f"docs: more\n\nasked by {LEAK_ADDRESS}")
    findings = source(repo)
    assert [finding for finding in findings if not finding.startswith("meta:")] == []
    meta = [finding for finding in findings if finding.startswith("meta:")]
    assert len(meta) == 1 and meta[0].endswith(":message:3: e-mail address")
    assert LEAK_ADDRESS not in meta[0]


def commit_as(repo: Path, email: str) -> None:
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", f"user.email={email}", "commit", "-q", "-m", "docs: more")


def meta_only(repo: Path) -> list[str]:
    return [finding for finding in source(repo) if finding.startswith("meta:")]


def test_the_configured_maintainer_address_is_allowed_in_metadata(tmp_path, monkeypatch):
    monkeypatch.delenv(rc.MAINTAINER_ENV, raising=False)
    repo = make_repo(tmp_path)
    git(repo, "config", "user.email", LEAK_ADDRESS)
    commit_as(repo, LEAK_ADDRESS)
    assert meta_only(repo) == []


def test_an_address_other_than_the_configured_maintainer_stays_a_finding(tmp_path, monkeypatch):
    monkeypatch.delenv(rc.MAINTAINER_ENV, raising=False)
    repo = make_repo(tmp_path)
    git(repo, "config", "user.email", "other" + "@" + "example.net")
    commit_as(repo, LEAK_ADDRESS)
    assert len(meta_only(repo)) == 2


def test_the_environment_overrides_the_configured_maintainer(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    git(repo, "config", "user.email", "other" + "@" + "example.net")
    commit_as(repo, LEAK_ADDRESS)
    monkeypatch.setenv(rc.MAINTAINER_ENV, LEAK_ADDRESS)
    assert meta_only(repo) == []


def test_an_accepted_meta_finding_is_bound_to_commit_field_line_and_class(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    commit_all(repo, f"docs: more\n\nasked by {LEAK_ADDRESS}\nsee {LEAK_PATH}")
    sha = git(repo, "rev-parse", "HEAD").strip()
    accepted = {(sha, "message", 3, "e-mail address")}
    findings = rc.meta_findings(repo, LISTS, accepted)
    assert findings == [f"meta:{sha[:12]}:message:4: absolute user path"]
    elsewhere = {(sha, "message", 4, "e-mail address"), ("0" * 40, "message", 3, "e-mail address")}
    assert len(rc.meta_findings(repo, LISTS, elsewhere)) == 2


def test_leak_in_an_author_address_is_a_meta_finding(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", f"user.email={LEAK_ADDRESS}", "commit", "-q", "-m", "docs: more")
    findings = source(repo)
    assert any(finding.endswith(":author email: e-mail address") for finding in findings)
    assert any(finding.endswith(":committer email: e-mail address") for finding in findings)


def test_noreply_addresses_in_metadata_are_allowed_but_nothing_else_on_the_line(tmp_path):
    repo = make_repo(tmp_path)
    bot = "noreply" + "@" + "anthropic.com"
    user = "12345+someone" + "@" + "users.noreply.github.com"
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", f"user.email={user}", "commit", "-q", "-m", f"docs: more\n\nCo-Authored-By: Bot <{bot}>")
    assert source(repo) == []
    (repo / "other.md").write_text("other\n", encoding="utf-8")
    commit_all(repo, f"docs: other\n\nCo-Authored-By: Bot <{bot}>, {LEAK_ADDRESS}")
    assert [finding for finding in source(repo) if finding.startswith("meta:")] != []


def test_correct_annotated_tag_on_head_passes(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "tag", "-a", "v0.1.0", "-m", "release 0.1.0")
    assert source(repo, "v0.1.0") == []


@pytest.mark.parametrize("case", ["missing", "wrong name", "not on head", "lightweight"])
def test_wrong_tag_is_a_finding(tmp_path, case):
    repo = make_repo(tmp_path)
    tag = "v0.1.0"
    if case == "wrong name":
        tag = "v0.0.9"
        git(repo, "tag", "-a", tag, "-m", "release")
    elif case == "not on head":
        git(repo, "tag", "-a", tag, "-m", "release")
        (repo / "later.md").write_text("later\n", encoding="utf-8")
        commit_all(repo, "docs: later")
    elif case == "lightweight":
        git(repo, "tag", tag)
    assert any(finding.startswith(f"tag:{tag}:") for finding in source(repo, tag))


def test_leak_in_a_tag_message_is_a_meta_finding(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "tag", "-a", "v0.1.0", "-m", f"release, ask {LEAK_ADDRESS}")
    assert any(finding.startswith("meta:tag v0.1.0:") for finding in source(repo, "v0.1.0"))


# -- codex round 1: ways a leak could slip through, each red before its fix ---------------------


def commit_blob(repo: Path, path: str, content: bytes, message: str) -> None:
    """Commit `content` at `path` through the index only, so any file name git accepts works."""
    blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input=content,
                          capture_output=True, check=True).stdout.decode().strip()
    # core.protectNTFS=false: the name exists only in the index and in history, never on disk.
    git(repo, "-c", "core.protectNTFS=false", "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
    git(repo, "commit", "-q", "-m", message)


def history_only(repo: Path) -> list[str]:
    return [finding for finding in source(repo) if finding.startswith("history:")]


def test_binary_file_in_history_is_still_scanned(tmp_path):
    repo = make_repo(tmp_path)
    commit_blob(repo, "notes.txt", b"\x00\x01 mail " + LEAK_ADDRESS.encode() + b"\n", "docs: notes")
    git(repo, "rm", "-q", "--cached", "notes.txt")
    git(repo, "commit", "-q", "-m", "docs: drop notes")
    assert any(finding.endswith("notes.txt:1: e-mail address") for finding in history_only(repo))


def test_quoted_file_name_in_history_is_still_scanned(tmp_path):
    repo = make_repo(tmp_path)
    commit_blob(repo, 'odd "name".md', f"see {LEAK_PATH}\n".encode(), "docs: odd")
    git(repo, "rm", "-q", "--cached", 'odd "name".md')
    git(repo, "commit", "-q", "-m", "docs: drop odd")
    assert any(finding.endswith(':odd "name".md:1: absolute user path') for finding in history_only(repo))


def test_file_name_with_a_space_in_history_is_reported_without_the_trailing_tab(tmp_path):
    repo = make_repo(tmp_path)
    commit_blob(repo, "my notes.md", f"see {LEAK_PATH}\n".encode(), "docs: notes")
    assert any(finding.endswith(":my notes.md:1: absolute user path") for finding in history_only(repo))


def test_diff_noprefix_in_the_repository_config_does_not_hide_history(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "config", "diff.noprefix", "true")
    commit_blob(repo, "notes.md", f"see {LEAK_PATH}\n".encode(), "docs: notes")
    assert any(finding.endswith(":notes.md:1: absolute user path") for finding in history_only(repo))


def test_pure_rename_from_a_skipped_suffix_is_scanned_under_the_new_name(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "notes.lock").write_text(f"see {LEAK_PATH}\n", encoding="utf-8")
    commit_all(repo, "chore: lock")
    git(repo, "mv", "notes.lock", "notes.txt")
    git(repo, "commit", "-q", "-m", "chore: rename")
    git(repo, "rm", "-q", "notes.txt")
    git(repo, "commit", "-q", "-m", "chore: drop")
    assert any(finding.endswith(":notes.txt:1: absolute user path") for finding in history_only(repo))


def test_control_characters_in_a_commit_message_do_not_cut_the_scan(tmp_path):
    repo = make_repo(tmp_path)
    message = tmp_path / "message.txt"
    message.write_bytes(f"docs: more\x1fcontact {LEAK_ADDRESS}\x1eend\n".encode())
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--cleanup=verbatim", "-F", str(message))
    assert any(finding.endswith(":message:1: e-mail address") for finding in meta_only(repo))


def test_an_allowed_noreply_address_does_not_cover_a_longer_domain(tmp_path, monkeypatch):
    monkeypatch.delenv(rc.MAINTAINER_ENV, raising=False)
    repo = make_repo(tmp_path)
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    commit_all(repo, "docs: more\n\nCo-Authored-By: Bot <" + "noreply" + "@" + "anthropic.com.evil.example>")
    assert any(finding.endswith(":message:3: e-mail address") for finding in meta_only(repo))


def test_an_inner_tag_reachable_only_through_an_outer_tag_is_scanned(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "tag", "-a", "inner", "-m", f"ask {LEAK_ADDRESS}")
    git(repo, "tag", "-a", "outer", "-m", "outer", "inner")
    git(repo, "tag", "-d", "inner")
    assert any(finding.startswith("meta:tag ") and finding.endswith(": e-mail address") for finding in meta_only(repo))


def test_a_leak_in_the_version_is_never_printed(tmp_path):
    version = "0.1.0+" + "sk-" + "ant-demo"
    repo = make_repo(tmp_path, version=version, changelog_heading=f"## [{version}] - 2026-09-23")
    findings = source(repo)
    assert any("pyproject.toml" in finding for finding in findings)
    assert not any("ant-demo" in finding for finding in findings)


def test_a_leak_in_a_file_name_is_withheld(tmp_path):
    repo = make_repo(tmp_path)
    name = LEAK_ADDRESS + ".md"
    commit_blob(repo, name, b"clean\n", "docs: named")
    findings = source(repo)
    assert any(finding.startswith("ref:") and "file name" in finding for finding in findings)
    assert not any(LEAK_ADDRESS in finding for finding in findings)


def test_a_valid_but_not_canonical_version_passes_and_a_local_version_does_not(tmp_path):
    assert source(make_repo(tmp_path / "a", version="0.1.0-1", changelog_heading="## [0.1.0-1] - 2026-09-23")) == []
    local = make_repo(tmp_path / "b", version="0.1.0+local1", changelog_heading="## [0.1.0+local1] - 2026-09-23")
    assert any("local version" in finding for finding in source(local))


def test_a_tag_on_a_tree_is_a_finding_not_an_error(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "tag", "-a", "v0.1.0", "-m", "release", "HEAD^{tree}")
    assert any(finding.startswith("tag:v0.1.0:") for finding in source(repo, "v0.1.0"))


def test_a_failed_build_still_prints_every_finding_found_before(tmp_path, capsys):
    repo = make_repo(tmp_path)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n\n[build-system]\nrequires = ["no-such-backend-zz"]\n'
        'build-backend = "no_such_backend_zz"\n', encoding="utf-8")
    commit_all(repo, "build: broken backend")
    (repo / "scratch.txt").write_text("x\n", encoding="utf-8")
    assert rc.main(["--repo", str(repo)]) == 2
    out = capsys.readouterr().out
    assert "worktree: not clean" in out and "RELEASE-CHECK: error:" in out


# -- codex round 2 --------------------------------------------------------------------------------


def test_a_leak_added_only_in_a_merge_resolution_is_found_with_combined_diff_config(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "config", "log.diffMerges", "combined")
    (repo / "a.md").write_text("base\n", encoding="utf-8")
    commit_all(repo, "docs: base")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "a.md").write_text("side\n", encoding="utf-8")
    commit_all(repo, "docs: side")
    git(repo, "checkout", "-q", "main")
    (repo / "a.md").write_text("main\n", encoding="utf-8")
    commit_all(repo, "docs: main")
    subprocess.run(["git", "merge", "-q", "side"], cwd=repo, capture_output=True)
    (repo / "a.md").write_text(f"resolved {LEAK_PATH}\n", encoding="utf-8")
    commit_all(repo, "merge side")
    (repo / "a.md").write_text("clean\n", encoding="utf-8")
    commit_all(repo, "docs: clean")
    assert any(finding.endswith(":a.md:1: absolute user path") for finding in history_only(repo))


def test_a_replace_ref_does_not_hide_the_original_commit_message(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    commit_all(repo, f"docs: more\n\nasked by {LEAK_ADDRESS}")
    original = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "commit", "-q", "--amend", "-m", "docs: more")
    clean = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "reset", "-q", "--hard", original)
    git(repo, "replace", original, clean)
    assert any(finding.startswith(f"meta:{original[:12]}:message:") for finding in meta_only(repo))


def test_a_shallow_clone_cannot_pass(tmp_path, capsys):
    origin = make_repo(tmp_path / "origin")
    (origin / "more.md").write_text("more\n", encoding="utf-8")
    commit_all(origin, "docs: more")
    subprocess.run(["git", "clone", "-q", "--depth", "1", origin.as_uri(), str(tmp_path / "shallow")], check=True,
                   capture_output=True)
    assert rc.main(["--repo", str(tmp_path / "shallow")]) == 2
    assert "shallow" in capsys.readouterr().out


def test_signature_headers_are_scanned_too(tmp_path):
    fields, _ = rc.object_fields(("tree 0\nauthor a <b> 1 +0000\ncommitter a <b> 1 +0000\ngpgsig -----BEGIN-----\n"
                                  f" Comment: {LEAK_ADDRESS}\n -----END-----\n\nmsg\n").encode())
    assert any(LEAK_ADDRESS in value for _, value in fields)


def test_an_empty_file_with_a_leaky_name_in_history_is_found(tmp_path):
    repo = make_repo(tmp_path)
    commit_blob(repo, LEAK_ADDRESS + ".txt", b"", "docs: empty")
    git(repo, "rm", "-q", "--cached", LEAK_ADDRESS + ".txt")
    git(repo, "commit", "-q", "-m", "docs: drop")
    findings = history_only(repo)
    assert any(finding.endswith("e-mail address in file name") for finding in findings)
    assert not any(LEAK_ADDRESS in finding for finding in findings)


def test_a_leaky_header_name_is_found_and_withheld():
    body = f"tree 0\nx-contact-{LEAK_ADDRESS} {LEAK_PATH}\n\nmsg\n".encode()
    findings = rc.object_findings("meta:abc", body, LISTS, rc.re.compile("(?!)"))
    assert any(finding.endswith("e-mail address in field name") for finding in findings)
    assert not any(LEAK_ADDRESS in finding for finding in findings)


def test_untracked_files_count_even_when_the_config_hides_them(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "scratch.txt").write_text("x\n", encoding="utf-8")
    assert any(finding.startswith("worktree:") for finding in source(repo))


def test_a_version_with_surrounding_whitespace_is_valid(tmp_path):
    repo = make_repo(tmp_path, version=" 0.1.0 ")
    assert [finding for finding in source(repo) if "pyproject" in finding or "CHANGELOG" in finding] == []


def test_a_leaky_archive_name_is_withheld(tmp_path):
    wheel = make_wheel(tmp_path / (LEAK_ADDRESS + "-0.1.0-py3-none-any.whl"), {"modelroom/n.py": f"# {LEAK_PATH}\n"})
    findings = rc.distribution_findings(wheel, LISTS)
    assert findings and not any(LEAK_ADDRESS in finding for finding in findings)


def test_duplicate_archive_members_are_each_scanned_and_reported(tmp_path):
    wheel = tmp_path / "modelroom-0.1.0-py3-none-any.whl"
    make_wheel(wheel)
    with zipfile.ZipFile(wheel, "a") as archive, pytest.warns(UserWarning):
        archive.writestr("modelroom/notes.txt", f"{LEAK_ADDRESS}\n")
        archive.writestr("modelroom/notes.txt", "clean\n")
    findings = rc.distribution_findings(wheel, LISTS)
    assert any(finding.endswith("modelroom/notes.txt:1: e-mail address") for finding in findings)
    assert any("duplicate member" in finding for finding in findings)


def make_wheel(path: Path, extra: dict[str, str] | None = None, entry_points: str = ENTRY_POINTS) -> Path:
    members = {
        "modelroom/__init__.py": '"""demo"""\n',
        "modelroom/cli.py": "def main():\n    return 0\n",
        "modelroom-0.1.0.dist-info/METADATA": "Metadata-Version: 2.4\nName: modelroom\nVersion: 0.1.0\n",
        "modelroom-0.1.0.dist-info/entry_points.txt": entry_points,
    }
    members.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return path


def make_sdist(path: Path, extra: dict[str, str] | None = None) -> Path:
    members = {"PKG-INFO": "Name: modelroom\n", "modelroom/__init__.py": '"""demo"""\n', "README.md": "# demo\n"}
    members.update(extra or {})
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(f"modelroom-0.1.0/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_clean_wheel_and_sdist_pass(tmp_path):
    wheel = make_wheel(tmp_path / "modelroom-0.1.0-py3-none-any.whl")
    sdist = make_sdist(tmp_path / "modelroom-0.1.0.tar.gz")
    assert rc.distribution_findings(wheel, LISTS) == []
    assert rc.distribution_findings(sdist, LISTS) == []


@pytest.mark.parametrize(
    "member",
    [".env", "state/modelroom.lock", "state/run-status.json", "state/hardware/kunde.json", ".trash/old.py",
     "modelroom.toml", "tests/fixtures/state/modelroom.json"],
)
def test_forbidden_file_class_in_the_wheel_is_a_finding(tmp_path, member):
    wheel = make_wheel(tmp_path / "modelroom-0.1.0-py3-none-any.whl", {member: "{}\n"})
    findings = rc.distribution_findings(wheel, LISTS)
    assert any(f":{member}: forbidden file" in finding for finding in findings)


def test_release_scripts_in_the_sdist_are_a_finding(tmp_path):
    sdist = make_sdist(tmp_path / "modelroom-0.1.0.tar.gz", {"scripts/release-check.py": "print()\n"})
    assert any(":scripts/release-check.py: forbidden file" in finding
               for finding in rc.distribution_findings(sdist, LISTS))


def test_leak_inside_a_wheel_member_is_a_finding(tmp_path):
    wheel = make_wheel(tmp_path / "modelroom-0.1.0-py3-none-any.whl", {"modelroom/notes.py": f"# {LEAK_ADDRESS}\n"})
    findings = rc.distribution_findings(wheel, LISTS)
    assert any(finding.endswith(":modelroom/notes.py:1: e-mail address") for finding in findings)
    assert not any(LEAK_ADDRESS in finding for finding in findings)


def test_a_member_with_a_nul_byte_is_still_scanned(tmp_path):
    wheel = make_wheel(tmp_path / "modelroom-0.1.0-py3-none-any.whl", {"modelroom/data.bin": f"\0{LEAK_ADDRESS}\n"})
    assert any(finding.endswith(":modelroom/data.bin:1: e-mail address") for finding in rc.distribution_findings(wheel, LISTS))


def test_wheel_without_console_entry_is_a_finding(tmp_path):
    wheel = make_wheel(tmp_path / "modelroom-0.1.0-py3-none-any.whl", entry_points="[console_scripts]\n")
    assert any("console entry" in finding for finding in rc.distribution_findings(wheel, LISTS))


def test_summary_line():
    assert rc.summary_line([], "0.1.0") == "RELEASE-CHECK: OK 0.1.0"
    assert rc.summary_line(["a", "b"], "0.1.0") == "RELEASE-CHECK: 2 findings"


def test_not_a_git_repository_is_exit_2(tmp_path, capsys):
    assert rc.main(["--repo", str(tmp_path)]) == 2
