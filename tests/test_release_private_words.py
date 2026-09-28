"""The private word list of the release gate (GF decision 2026-09-28): words outside the repository
that must not be published, found by place and class only, never printed.

Every word below is assembled at runtime, like the leaks of `tests/test_release_check.py`.
"""

import importlib.util
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "release-check.py"
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the release gate is tested from a clone", allow_module_level=True)
_SPEC = importlib.util.spec_from_file_location("release_check_words", SCRIPT)
rc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rc)

WORD = "zq" + "workbench" + "7"  # a word no file of this repository holds


def git(repo: Path, *args: str) -> str:
    base = ["git", "-c", "user.name=tester", "-c", "user.email=tester", "-c", "commit.gpgsign=false",
            "-c", f"core.hooksPath={repo / '.no-hooks'}"]
    return subprocess.run(base + list(args), cwd=repo, capture_output=True, text=True, check=True).stdout


def commit_all(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    (repo / "pyproject.toml").write_text('[project]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [0.1.0] - 2026-09-28\n\n- first\n", encoding="utf-8")
    commit_all(repo, "feat: first")
    return repo


@pytest.fixture
def words(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "private-words.txt"
    path.write_text(f"# the maintainer's words\n\n  {WORD.upper()}  \n", encoding="utf-8")
    monkeypatch.setenv(rc.PRIVATE_WORDS_ENV, str(path))
    return path


def lists():
    return rc.with_private_words(rc.load_negative_list(), rc.private_words_pattern())


def test_a_word_in_a_file_of_head_is_found_in_any_case_by_place_and_class(tmp_path, words):
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text(f"# notes\nthe machine {WORD} ran it\n", encoding="utf-8")
    commit_all(repo, "docs: notes")
    findings = rc.source_findings(repo, "HEAD", None, lists())
    assert "ref:notes.md:2: private word" in findings
    assert not any(WORD in finding.lower() for finding in findings)


def test_a_word_in_a_commit_message_is_a_meta_finding(tmp_path, words):
    repo = make_repo(tmp_path)
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    commit_all(repo, f"docs: more\n\nmeasured on {WORD}")
    assert any(finding.startswith("meta:") and finding.endswith(": private word") for finding in rc.meta_findings(repo, lists()))


def test_a_word_inside_a_built_distribution_is_a_finding(tmp_path, words):
    wheel = tmp_path / "modelroom-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("modelroom/notes.txt", f"built on {WORD}\n")
    assert any(finding.endswith(": private word") for finding in rc.distribution_findings(wheel, lists()))


def test_a_word_only_in_the_earlier_contents_of_the_history_is_accepted(tmp_path, words):
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text(f"{WORD}\n", encoding="utf-8")
    commit_all(repo, "docs: notes")
    (repo / "notes.md").write_text("clean\n", encoding="utf-8")
    commit_all(repo, "docs: clean")
    assert not any("private word" in finding for finding in rc.source_findings(repo, "HEAD", None, lists()))


def test_a_list_without_a_hit_finds_nothing(tmp_path, words):
    repo = make_repo(tmp_path)
    assert rc.source_findings(repo, "HEAD", None, lists()) == []


def test_an_unset_variable_is_a_usage_error(monkeypatch):
    monkeypatch.delenv(rc.PRIVATE_WORDS_ENV, raising=False)
    with pytest.raises(rc.UsageError, match="is not set"):
        rc.private_words_pattern()


def test_a_missing_file_is_a_usage_error_that_names_no_path(tmp_path, monkeypatch):
    monkeypatch.setenv(rc.PRIVATE_WORDS_ENV, str(tmp_path / "gone.txt"))
    with pytest.raises(rc.UsageError) as caught:
        rc.private_words_pattern()
    assert "gone.txt" not in str(caught.value)


def test_the_gate_run_never_prints_the_word_and_does_not_pass_without_the_list(tmp_path, words, monkeypatch, capsys):
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text(f"{WORD}\n", encoding="utf-8")
    commit_all(repo, "docs: notes")
    code = rc.main(["--repo", str(repo)])
    out = capsys.readouterr().out
    assert code != 0 and "ref:notes.md:1: private word" in out and WORD not in out.lower()
    monkeypatch.delenv(rc.PRIVATE_WORDS_ENV)
    assert rc.main(["--repo", str(repo)]) == 2
    out = capsys.readouterr().out
    assert f"RELEASE-CHECK: error: {rc.PRIVATE_WORDS_ENV} is not set" in out and "RELEASE-CHECK: OK" not in out
