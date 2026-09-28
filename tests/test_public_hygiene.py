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
    "processor model": re.compile(r'"cpu_name"\s*:\s*"(?!(?:' + "|".join(map(re.escape, sorted(NEUTRAL_CPU_NAMES))) + r')")'),
}
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


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / line for line in out.splitlines() if line]


def findings() -> list[str]:
    hits = []
    for path in tracked_files():
        if path.suffix in SKIP_SUFFIXES or not path.exists():
            continue
        rel = path.relative_to(REPO).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        checks = [(name, pattern.search) for name, pattern in FORBIDDEN.items()] + [(GLOBAL_IPV4, global_ipv4)]
        for name, found in checks:
            if (rel, name) in ALLOWED:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if found(line):
                    hits.append(f"{rel}:{line_no}: {name}: {line.strip()[:120]}")
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
    pattern = FORBIDDEN["processor model"]
    for leak in ('"cpu_name": "' + "AMD EPYC 7413" + '",', '"cpu_name":"' + "Intel Xeon" + '"', '"cpu_name": "x86 CPU (' + 'laptop)"'):
        assert pattern.search(leak), leak
    for quiet in ('"cpu_name": "x86 CPU",', '"cpu_name": "arm64 CPU"', '"cpu_name": ""', '"gpu_name": "NVIDIA L4"'):
        assert not pattern.search(quiet), quiet


if __name__ == "__main__":
    problems = findings()
    for problem in problems:
        print(problem)
    raise SystemExit(1 if problems else 0)
