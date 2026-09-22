"""The version has exactly one source: [project] version in pyproject.toml."""

import re
import subprocess
import tomllib
from pathlib import Path

import modelroom

REPO = Path(__file__).resolve().parent.parent
VERSION_LITERAL = re.compile(r"""(?i)(^|[^\w.])(__version__|version)\s*[:=]\s*["']\d+\.\d+\.\d+["']""")


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / line for line in out.splitlines() if line]


def pyproject_version() -> str:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_pyproject_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pyproject_version())


def test_package_reports_the_pyproject_version():
    assert modelroom.__version__ == pyproject_version()


def test_no_second_version_literal_in_tracked_files():
    offenders = []
    for path in tracked_files():
        if path.name in {"pyproject.toml", "uv.lock", "CHANGELOG.md"} or path.suffix in {".png", ".jpg", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if VERSION_LITERAL.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{line_no}: {line.strip()}")
    assert offenders == [], "version literals outside pyproject.toml:\n" + "\n".join(offenders)
