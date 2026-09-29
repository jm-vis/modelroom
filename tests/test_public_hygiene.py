"""This repository is public. Nothing operator-specific may be tracked.

The negative list below is deliberately about classes of leakage (machine paths, e-mail
addresses, internal workspace vocabulary), not about particular people. The git hooks in
.githooks/ run this file before every commit and push.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".lock"}
# The placeholders a recording's `cpu_name` may carry instead of the processor it was taken on.
NEUTRAL_CPU_NAMES = {"", "x86 CPU", "arm64 CPU", "Apple Silicon CPU"}

FORBIDDEN = {
    "absolute user path": re.compile(r"(?i)(?:[a-z]:[\\/]+users[\\/]|/home/[a-z]|/Users/[A-Za-z])"),
    "e-mail address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "internal workspace name": re.compile(r"(?i)agenticos|10_visconsult|00_context"),
    "persona name": re.compile(r"(?i)\b(vico|vica|cura)\b"),
    "chronicle vocabulary": re.compile(r"\bGF-|Sitzung #\d"),
    "protected folder token": re.compile(r"(?i)[\\/]_lokal[\\/]|strict_confidential"),
    "api key prefix": re.compile(r"\b(sk-or-|sk-ant-|hf_[A-Za-z0-9]{20,})"),
    # Device and machine identifiers of a recording (AP9-GPU, GF decision 2026-09-28): the subsystem
    # id names the maker's board of one device model, the processor names the machine. A recording
    # keeps what the detection needs (vendor, device id, card name, memory) and neutralizes the rest.
    "hardware subsystem id": re.compile(r"(?i)SUBSYS_(?!00000000)[0-9a-f]{8}"),
}
# The processor is a check of the whole document, not of a line: the value of `cpu_name` may stand
# on the next line, and it is judged decoded (`"x86 CPU"` is the placeholder) -- `processor_model_lines`.
PROCESSOR_MODEL = "processor model"
_CPU_NAME = re.compile(r'"cpu_name"\s*:\s*"((?:[^"\\]|\\.)*)"')


def processor_model_lines(text: str) -> list[int]:
    """The line of every `cpu_name` whose decoded value is no placeholder (`NEUTRAL_CPU_NAMES`)."""
    import json

    lines = []
    for match in _CPU_NAME.finditer(text):
        try:
            value = json.loads(f'"{match.group(1)}"')
        except ValueError:
            value = None  # a value that does not decode is no placeholder either
        if value not in NEUTRAL_CPU_NAMES:
            lines.append(text.count("\n", 0, match.start()) + 1)
    return lines
# A global IPv4 address (AP9-UX): the address of a rented machine must not reach the repository.
# No regex can say "global", so this one is a function next to the list, with a name of its own.
# Loopback, the private ranges and the documentation ranges of RFC 5737 stay quiet -- the code, the
# documents and the tests use those. Hostnames of rental providers have no pattern here on purpose:
# one would first have to write their naming rule into this public repository, and a general one
# (word-word-digits) fires on model and product names; the protection sits at the source instead.
GLOBAL_IPV4 = "global IPv4 address"
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def global_ipv4(line: str) -> bool:
    """Whether `line` holds an IPv4 address that `ipaddress` reads and calls global."""
    import ipaddress

    for match in _IPV4.finditer(line):
        try:
            if ipaddress.ip_address(match.group(0)).is_global:
                return True
        except ValueError:
            continue
    return False


# Places where a pattern is legitimate. Each line is (path, pattern name, reason).
ALLOWED = {
    ("tests/test_public_hygiene.py", GLOBAL_IPV4): "the negative list names what it forbids",
    ("tests/test_config.py", GLOBAL_IPV4): "a four-part version string, not an address",
    ("tests/test_public_hygiene.py", "internal workspace name"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "persona name"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "chronicle vocabulary"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "protected folder token"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "api key prefix"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "absolute user path"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "e-mail address"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "hardware subsystem id"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "processor model"): "the negative list names what it forbids",
}
# Every check in one place, the way `scripts/release-check.py` takes them over: the checks of one line
# (the patterns and the global address) and the checks of a whole document (the processor).
LINE_CHECKS = {**{name: pattern.search for name, pattern in FORBIDDEN.items()}, GLOBAL_IPV4: global_ipv4}
DOCUMENT_CHECKS = {PROCESSOR_MODEL: processor_model_lines}
WITHHELD = "<withheld>"


def classify(text: str, path: str = "") -> list[tuple[int, str]]:
    """`(line, class)` of every hit in `text`, except where `ALLOWED` names the path and the class."""
    hits = []
    for name, found in LINE_CHECKS.items():
        if (path, name) not in ALLOWED:
            hits += [(line_no, name) for line_no, line in enumerate(text.split("\n"), start=1) if found(line)]
    for name, lines_of in DOCUMENT_CHECKS.items():
        if (path, name) not in ALLOWED:
            hits += [(line_no, name) for line_no in lines_of(text)]
    return sorted(set(hits))


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / line for line in out.splitlines() if line]


PRIVATE_WORDS_ENV = "MODELROOM_RELEASE_PRIVATE_WORDS"


def private_words() -> list[str] | None:
    """The release gate's private words (`scripts/release-check.py`), or `None` where the file is not
    named or does not read. Only to decide whether a path may be printed; the words stay outside."""
    import os

    location = os.environ.get(PRIVATE_WORDS_ENV, "").strip()
    try:
        lines = Path(location).read_text(encoding="utf-8-sig").splitlines() if location else None
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return None if lines is None else [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def findings(paths: list[Path] | None = None, root: Path = REPO, words: list[str] | None | bool = True) -> list[str]:
    """Every hit by place and class only, never the line. A path is printed only where the private
    words are known (`words`, by default read like the release gate reads them) and it holds none of
    them and no hit of its own; otherwise it is withheld (Codex r4-4)."""
    known = private_words() if words is True else words
    hits = []
    for path in paths if paths is not None else tracked_files():
        if path.suffix in SKIP_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        private = known is None or any(word.casefold() in rel.casefold() for word in known)
        shown = WITHHELD if private or classify(rel) else rel
        try:  # every touch of the file system inside: its errors name the full path (Codex r5-2, r6)
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # its message names the full path: only the place, redacted
            hits.append(f"{shown}: unreadable file")
            continue
        hits += [f"{shown}:{line_no}: {name}" for line_no, name in classify(text, rel)]
    return hits


def test_tracked_files_are_free_of_operator_specifics():
    hits = findings()
    assert hits == [], "operator-specific content in tracked files:\n" + "\n".join(hits)


def test_negative_list_catches_a_planted_leak(tmp_path):
    """The guard has to go red on a real leak, otherwise its green means nothing."""
    planted = "notes for the maintainer at C:\\Users\\someone\\workspace\\file.md"
    assert any(p.search(planted) for p in FORBIDDEN.values())


def test_the_global_ipv4_check_catches_a_planted_address_and_nothing_else():
    """Its own assert: `any` over the other patterns would stay green on the path alone (AP9-UX)."""
    planted = "ssh root@" + ".".join(["81", "23", "140", "7"]) + " -p 22"  # a fictitious global address, built here
    assert global_ipv4(planted)
    for quiet in ("127.0.0.1", "10.1.2.3", "192.168.178.20", "172.16.0.9", "192.0.2.10", "198.51.100.7", "203.0.113.5"):
        assert not global_ipv4(f"listen on {quiet}:8000"), quiet
    assert global_ipv4("version 1.1.16.2")  # a version reads as an address: `ALLOWED` names the file
    assert ("tests/test_config.py", GLOBAL_IPV4) in ALLOWED


def test_a_subsystem_id_other_than_zeros_is_a_finding_in_either_case():
    """AP9-GPU: the maker's subsystem id of a recorded device; `00000000` is the neutral form."""
    pattern = FORBIDDEN["hardware subsystem id"]
    for leak in ("PCI\\VEN_10DE&DEV_2F38&SUBSYS_" + "1A2B3C4D&REV_A1", "pci\\ven_10de&dev_2f38&subsys_" + "1a2b3c4d"):
        assert pattern.search(leak), leak
    for quiet in ("PCI\\VEN_10DE&DEV_2F38&SUBSYS_00000000&REV_A1", "pci\\ven_10de&subsys_00000000", "SUBSYSTEM_ID"):
        assert not pattern.search(quiet), quiet


def test_a_processor_name_other_than_a_placeholder_is_a_finding():
    leaks = ('"cpu_name": "' + "AMD EPYC 7413" + '",', '"cpu_name":"' + "Intel Xeon" + '"', '"cpu_name": "x86 CPU (' + 'laptop)"',
             '{\n  "cpu_name":\n    "' + "Intel Xeon" + '"\n}')  # the value on the next line (Codex r3-9)
    for leak in leaks:
        assert processor_model_lines(leak), leak
    assert processor_model_lines('{\n  "cpu_name":\n    "' + "Intel Xeon" + '"\n}') == [2]
    quiet = ('"cpu_name": "x86 CPU",', '"cpu_name": "arm64 CPU"', '"cpu_name": ""', '"gpu_name": "NVIDIA L4"',
             '"cpu_name": "x86\\u0020CPU"')  # the placeholder, escaped: no false alarm
    for text in quiet:
        assert not processor_model_lines(text), text


def test_a_finding_names_place_and_class_and_withholds_a_leaking_path(tmp_path):
    """Codex r3-12: the output carries no line and no path that is a hit itself."""
    leak = "SUBSYS_" + "1A2B3C4D"
    named = tmp_path / f"{leak}.txt"
    named.write_text(f"ok\nthe device {leak}\n", encoding="utf-8")
    plain = tmp_path / "plain.json"
    plain.write_text('{"cpu_name": "' + "Intel Xeon" + '"}\n', encoding="utf-8")
    hits = findings([named, plain], tmp_path, words=[])
    assert hits == [f"{WITHHELD}:2: hardware subsystem id", "plain.json:1: processor model"]
    assert not any(leak in hit or "Xeon" in hit for hit in hits)


def test_a_path_is_printed_only_where_the_private_words_are_known_and_absent(tmp_path):
    """Codex r4-4: this test runs without the gate's word list; a data path is withheld then."""
    word = "zq" + "workbench" + "7"
    named = tmp_path / f"notes-{word}.txt"
    named.write_text("device SUBSYS_" + "1A2B3C4D\n", encoding="utf-8")
    plain = tmp_path / "plain.txt"
    plain.write_text("device SUBSYS_" + "1A2B3C4D\n", encoding="utf-8")
    without = findings([named, plain], tmp_path, words=None)
    assert without == [f"{WITHHELD}:1: hardware subsystem id"] * 2
    known = findings([named, plain], tmp_path, words=[word.upper()])
    assert known == [f"{WITHHELD}:1: hardware subsystem id", "plain.txt:1: hardware subsystem id"]
    assert not any(word in hit for hit in without + known)


def test_a_file_that_cannot_be_read_is_a_redacted_finding(tmp_path, capsys):
    """Codex r5-2: a real read error (a folder where a file was listed) names no path and no word."""
    word = "zq" + "workbench" + "7"
    folder = tmp_path / f"notes-{word}.txt"
    folder.mkdir()
    hits = findings([folder], tmp_path, words=[word])
    print("\n".join(hits))
    captured = capsys.readouterr()
    assert hits == [f"{WITHHELD}: unreadable file"]
    assert word not in captured.out + captured.err and str(tmp_path) not in captured.out + captured.err


def test_an_error_of_the_existence_check_is_a_redacted_finding(tmp_path, monkeypatch, capsys):
    """Codex r6: `exists` itself can fail, with the path in its message."""
    word = "zq" + "workbench" + "7"
    path = tmp_path / f"notes-{word}.txt"

    def refused(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "exists", refused)
    hits = findings([path], tmp_path, words=[word])
    print("\n".join(hits))
    captured = capsys.readouterr()
    assert hits == [f"{WITHHELD}: unreadable file"]
    assert word not in captured.out + captured.err and str(tmp_path) not in captured.out + captured.err


if __name__ == "__main__":
    problems = findings()
    for problem in problems:
        print(problem)
    raise SystemExit(1 if problems else 0)
