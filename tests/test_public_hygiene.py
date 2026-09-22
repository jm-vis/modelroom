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

FORBIDDEN = {
    "absolute user path": re.compile(r"(?i)(?:[a-z]:[\\/]+users[\\/]|/home/[a-z]|/Users/[A-Za-z])"),
    "e-mail address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "internal workspace name": re.compile(r"(?i)agenticos|10_visconsult|00_context"),
    "persona name": re.compile(r"(?i)\b(vico|vica|cura)\b"),
    "chronicle vocabulary": re.compile(r"\bGF-|Sitzung #\d"),
    "protected folder token": re.compile(r"(?i)[\\/]_lokal[\\/]|strict_confidential"),
    "api key prefix": re.compile(r"\b(sk-or-|sk-ant-|hf_[A-Za-z0-9]{20,})"),
}

# Places where a pattern is legitimate. Each line is (path, pattern name, reason).
ALLOWED = {
    ("tests/test_public_hygiene.py", "internal workspace name"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "persona name"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "chronicle vocabulary"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "protected folder token"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "api key prefix"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "absolute user path"): "the negative list names what it forbids",
    ("tests/test_public_hygiene.py", "e-mail address"): "the negative list names what it forbids",
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
        for name, pattern in FORBIDDEN.items():
            if (rel, name) in ALLOWED:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{rel}:{line_no}: {name}: {line.strip()[:120]}")
    return hits


def test_tracked_files_are_free_of_operator_specifics():
    hits = findings()
    assert hits == [], "operator-specific content in tracked files:\n" + "\n".join(hits)


def test_negative_list_catches_a_planted_leak(tmp_path):
    """The guard has to go red on a real leak, otherwise its green means nothing."""
    planted = "notes for the maintainer at C:\\Users\\someone\\workspace\\file.md"
    assert any(p.search(planted) for p in FORBIDDEN.values())


if __name__ == "__main__":
    problems = findings()
    for problem in problems:
        print(problem)
    raise SystemExit(1 if problems else 0)
