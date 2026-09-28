"""Every scan path of the release gate against every class and reading (Codex diff review r3 of AP9-GPU):
binary suffixes, metadata in other encodings, the e-mail allowance, the global address, the archive's
own metadata, parser failures and a run without the private word list.

Every leak below is assembled at runtime; nothing of it is in this file as written.
"""

import io
import subprocess
import tarfile
import zipfile

from test_release_private_words import WORD, commit_all, git, lists, make_repo, rc, words  # noqa: F401 -- `words` is a fixture

SUBSYS = "SUBSYS_" + "1A2B3C4D"
ADDRESS = ".".join(["81", "23", "140", "7"])  # a fictitious global address
LEAK_PATH = "/ho" + "me/someone/notes.md"


def wheel_with(tmp_path, members: dict, comment: bytes = b"", folders: tuple = ()) -> "Path":  # noqa: F821
    wheel = tmp_path / "modelroom-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.comment = comment
        for folder in folders:
            archive.writestr(zipfile.ZipInfo(folder), b"")
        for name, data in members.items():
            archive.writestr(name, data)
    return wheel


def classes(findings: list[str]) -> set[str]:
    return {finding.rsplit(": ", 1)[-1] for finding in findings}


# --- r3-6: a skipped suffix gets every class and every reading ---------------------------------------------------


def test_a_subsystem_id_in_a_lock_file_and_a_utf16_word_in_a_png_are_found(tmp_path, words):
    wheel = wheel_with(tmp_path, {"modelroom/uv.lock": f"device {SUBSYS}\n".encode(),
                                  "modelroom/icon.png": b"\x89PNG\r\n\x1a\n" + f"comment {WORD}".encode("utf-16-le") + b"\x00\x00"})
    findings = rc.distribution_findings(wheel, lists())
    assert any(f.endswith("uv.lock:1: hardware subsystem id") for f in findings)
    assert any("icon.png" in f and f.endswith("private word") for f in findings)


def test_a_binary_file_is_read_in_its_printable_runs_not_its_noise(tmp_path, words):
    """Random bytes hold an `@` and a `.` often enough; only a run of printable text is text."""
    noise = b"\x89PNG\x00\x01a" + b"@b.c\x02\x03"
    address = ("some" + "one@" + "example.org").encode()
    quiet = rc.distribution_findings(wheel_with(tmp_path, {"modelroom/a.png": noise}), lists())
    assert "e-mail address" not in classes(quiet)
    loud = rc.distribution_findings(wheel_with(tmp_path, {"modelroom/a.png": noise + address + b"\x00"}), lists())
    assert "e-mail address" in classes(loud)


def _word_list(tmp_path, monkeypatch, *found: str):
    path = tmp_path / "more-words.txt"
    path.write_text("".join(f"{word}\n" for word in found), encoding="utf-8")
    monkeypatch.setenv(rc.PRIVATE_WORDS_ENV, str(path))
    return rc.with_private_words(rc.load_negative_list(), rc.private_words())


def test_a_short_word_and_a_non_ascii_word_are_found_in_a_binary_file_as_bytes(tmp_path, monkeypatch):
    """Codex r4-1: a word shorter than a printable run, and one with an umlaut, in any encoding."""
    short, umlaut = "zq" + "7", "zq" + "m\u00fcller" + "7"
    lists_ = _word_list(tmp_path, monkeypatch, short, umlaut)
    for encoding in rc.WORD_ENCODINGS:
        for word in (short, umlaut):
            data = b"\x89PNG\x00\x01" + word.encode(encoding) + b"\x02\x00"
            wheel = wheel_with(tmp_path, {"modelroom/icon.png": data})
            assert "private word" in classes(rc.distribution_findings(wheel, lists_)), (encoding, word)


def test_a_non_ascii_word_in_mixed_case_is_found_in_every_encoding_and_in_archive_metadata(tmp_path, monkeypatch):
    """Codex r5-1: the list says `m\u00fcller-s\u00fcd`, the file `M\u00fcller-S\u00dcD` -- a case no byte form of the list holds."""
    lists_ = _word_list(tmp_path, monkeypatch, "zqm\u00fcller-s\u00fcd")
    stored = "ZqM\u00fcller-S\u00dcD"
    for encoding in rc.WORD_ENCODINGS:
        for prefix in (b"\x89PNG\x00", b"\x89PNG"):  # an even and an odd offset
            wheel = wheel_with(tmp_path, {"modelroom/icon.png": prefix + stored.encode(encoding) + b"\x00\x00"})
            assert "private word" in classes(rc.distribution_findings(wheel, lists_)), (encoding, prefix)
        wheel = wheel_with(tmp_path, {"modelroom/cli.py": b"x = 1\n"}, comment=b"by " + stored.encode(encoding))
        assert any("<archive comment>" in f and f.endswith("private word") for f in rc.distribution_findings(wheel, lists_)), encoding


def test_a_word_in_a_latin1_text_file_is_found(tmp_path, monkeypatch):
    word = "zq" + "m\u00fcller" + "7"
    lists_ = _word_list(tmp_path, monkeypatch, word)
    wheel = wheel_with(tmp_path, {"modelroom/notes.txt": f"built by {word}\n".encode("latin-1")})
    assert "private word" in classes(rc.distribution_findings(wheel, lists_))


def test_an_address_in_a_utf16_file_deleted_later_is_found_in_the_history(tmp_path, words):
    """Codex r4-2: the history reads a line of a UTF-16 file as HEAD would have read it."""
    repo = make_repo(tmp_path)
    address = "some" + "one@" + "example.org"
    (repo / "notes.txt").write_bytes(f"hello\nwrite to {address}\n".encode("utf-16-le"))
    commit_all(repo, "docs: notes")
    (repo / "notes.txt").unlink()
    commit_all(repo, "docs: drop")
    findings = rc.history_findings(repo, lists())
    assert any(f.endswith(": e-mail address") and ":notes.txt:" in f for f in findings)
    assert not any(address in f for f in findings)


def test_an_empty_file_named_with_a_word_and_deleted_later_is_found_in_the_history(tmp_path, words):
    """Codex r4-3: the exception of the history is for its contents, not its names."""
    repo = make_repo(tmp_path)
    (repo / f"{WORD}.txt").write_bytes(b"")
    commit_all(repo, "docs: empty")
    (repo / f"{WORD}.txt").unlink()
    commit_all(repo, "docs: drop")
    findings = rc.history_findings(repo, lists())
    assert any(f.endswith("private word in file name") for f in findings)
    assert not any(WORD in f.lower() for f in findings)


# --- r3-7: metadata in its declared encoding, with JSON escapes --------------------------------------------------


def test_a_json_escaped_word_in_a_tag_message_is_found(tmp_path, words):
    repo = make_repo(tmp_path)
    git(repo, "tag", "-a", "v0.1.0", "-m", "release " + "".join(f"\\u{ord(c):04x}" for c in WORD))
    assert any(f.startswith("meta:tag v0.1.0:message:") and f.endswith("private word") for f in rc.meta_findings(repo, lists()))


def test_a_word_in_a_latin1_commit_is_found_and_an_unknown_encoding_is_a_finding(tmp_path, monkeypatch):
    word = "zq" + "w\u00fcrk" + "7"  # a real non-ASCII letter (Codex r4-5)
    assert not word.isascii()
    path = tmp_path / "latin-words.txt"
    path.write_text(f"{word}\n", encoding="utf-8")
    monkeypatch.setenv(rc.PRIVATE_WORDS_ENV, str(path))
    repo = make_repo(tmp_path)
    (repo / "message.txt").write_bytes(f"docs: more\n\nmeasured on {word}\n".encode("latin-1"))
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    git(repo, "add", "more.md")
    git(repo, "-c", "i18n.commitEncoding=ISO-8859-1", "commit", "-q", "-F", "message.txt")
    (repo / "message.txt").unlink()
    assert any(f.endswith(": private word") for f in rc.meta_findings(repo, lists()))
    tree = git(repo, "rev-parse", "HEAD^{tree}").strip()
    raw = f"tree {tree}\nauthor a <b> 1 +0000\ncommitter a <b> 1 +0000\nencoding x-no-such-encoding\n\nmsg\n"
    sha = subprocess.run(["git", "hash-object", "-t", "commit", "-w", "--stdin"], cwd=repo, input=raw.encode(),
                         capture_output=True, check=True).stdout.decode().strip()
    git(repo, "update-ref", "refs/heads/odd", sha)
    assert f"meta:{sha[:12]}: unsupported encoding" in rc.meta_findings(repo, lists())


# --- r3-8: the e-mail allowance covers the class `e-mail address` only --------------------------------------------


def test_a_word_as_the_user_of_an_allowed_noreply_address_is_found_for_author_and_tagger(tmp_path, words):
    repo = make_repo(tmp_path)
    address = f"{WORD}" + "@users.noreply.github.com"
    (repo / "more.md").write_text("more\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", f"user.email={address}", "commit", "-q", "-m", "docs: more")
    git(repo, "-c", f"user.email={address}", "tag", "-a", "v0.1.0", "-m", "release")
    findings = rc.meta_findings(repo, lists())
    assert any(":author email: private word" in f for f in findings)
    assert any(f.startswith("meta:tag v0.1.0:tagger email: private word") for f in findings)
    assert not any("e-mail address" in f for f in findings)  # the allowed address itself stays allowed


# --- r3-9 and r3-10: the processor over lines, the global address in every path ------------------------------------


def test_a_processor_on_the_next_line_and_a_global_address_are_found_in_head_and_in_a_wheel(tmp_path, words):
    repo = make_repo(tmp_path)
    (repo / "system.json").write_text('{\n  "cpu_name":\n    "' + "Intel Xeon" + '"\n}\n', encoding="utf-8")
    (repo / "notes.md").write_text(f"ssh to {ADDRESS}\n", encoding="utf-8")
    commit_all(repo, f"docs: notes\n\nrented at {ADDRESS}")
    source = rc.source_findings(repo, "HEAD", None, lists())
    assert "ref:system.json:2: processor model" in source and "ref:notes.md:1: global IPv4 address" in source
    assert any(f.startswith("meta:") and f.endswith("global IPv4 address") for f in source)
    assert not any(f.startswith("history:") and ("IPv4" in f or "processor" in f) for f in source)  # current state only
    wheel = wheel_with(tmp_path, {"modelroom/notes.txt": f"ssh to {ADDRESS}\n".encode()})
    assert "global IPv4 address" in classes(rc.distribution_findings(wheel, lists()))


def test_the_escaped_placeholder_is_no_processor_finding(tmp_path, words):
    wheel = wheel_with(tmp_path, {"modelroom/system.json": b'{"cpu_' + b'name": "x86\\u0020CPU"}\n'})
    assert "processor model" not in classes(rc.distribution_findings(wheel, lists()))


# --- r3-11: the archive's own metadata ------------------------------------------------------------------------------


def test_a_word_in_the_zip_comment_and_in_a_folder_entry_is_found_and_withheld(tmp_path, words):
    wheel = wheel_with(tmp_path, {"modelroom/cli.py": b"x = 1\n"}, comment=f"built on {WORD}".encode(), folders=(f"modelroom/{WORD}/",))
    findings = rc.distribution_findings(wheel, lists())
    assert any("<archive comment>" in f and f.endswith("private word") for f in findings)
    assert any(f.endswith("private word in entry name") for f in findings)
    assert not any(WORD in f.lower() for f in findings)


def test_a_word_as_a_link_target_or_an_owner_of_the_sdist_is_found(tmp_path, words):
    sdist = tmp_path / "modelroom-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo("modelroom-0.1.0/README.md")
        info.size, info.uname = 3, WORD
        archive.addfile(info, io.BytesIO(b"hi\n"))
        link = tarfile.TarInfo("modelroom-0.1.0/link")
        link.type, link.linkname = tarfile.SYMTYPE, f"../{WORD}"
        archive.addfile(link)
    findings = rc.distribution_findings(sdist, lists())
    assert any("<owner>" in f for f in findings) and any("<link target>" in f for f in findings)
    assert not any(WORD in f.lower() for f in findings)


# --- r3-13: a parser's message never reaches the output --------------------------------------------------------------


def test_an_entry_points_file_that_does_not_read_is_a_finding_without_its_text(tmp_path, words):
    wheel = wheel_with(tmp_path, {"modelroom/cli.py": b"x = 1\n", "m-0.1.dist-info/entry_points.txt": f"{WORD} = broken\n".encode()})
    findings = rc.distribution_findings(wheel, lists())
    assert any(f.endswith("entry_points.txt does not read") for f in findings)
    assert not any(WORD in f.lower() and "private word" not in f for f in findings)
    assert not any(WORD in f.lower() for f in findings)


def test_an_unexpected_error_is_reported_by_its_class_only_on_stdout_and_stderr(tmp_path, words, monkeypatch, capsys):
    repo = make_repo(tmp_path)

    def broken(*_args):
        raise ValueError(f"line 1: {WORD}")

    monkeypatch.setattr(rc, "dist_findings", broken)
    assert rc.main(["--repo", str(repo)]) == 2
    captured = capsys.readouterr()
    assert "unexpected ValueError" in captured.out and WORD not in (captured.out + captured.err).lower()


# --- r3-14: without the word list every name from the data is withheld -----------------------------------------------


def test_without_the_word_list_every_name_from_the_data_is_withheld(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(rc.PRIVATE_WORDS_ENV, raising=False)
    repo = make_repo(tmp_path)
    (repo / f"notes-{WORD}.md").write_text(f"see {LEAK_PATH}\n", encoding="utf-8")
    commit_all(repo, "docs: notes")
    assert rc.main(["--repo", str(repo)]) == 2
    out = capsys.readouterr().out
    assert "absolute user path" in out and "<withheld>" in out and WORD not in out.lower()
