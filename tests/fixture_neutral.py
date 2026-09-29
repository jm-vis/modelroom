"""The fixed values of the neutral names (FR-N) the suite works with, and the search for a sentinel.

`THIS_MACHINE` is the key `tests/conftest.py` gives this computer in every test. The sentinels stand
for a host name, an address and a build name a real machine could carry; they are made up and hold
only documentation addresses (RFC 5737), so they pass the hygiene test.
"""

from __future__ import annotations

from pathlib import Path

THIS_MACHINE = "computer-5c1e9a07"
SENTINEL_HOST = "sentinel-host"
SENTINEL_MODEL = "sentinel-host:latest"
SENTINEL_VERSIONS = ("0.1.0+203.0.113.77", "0.1.0-buildhost")


def files_with(folder: Path, needle: str) -> list[str]:
    """Every file under `folder` whose bytes carry `needle`, as paths relative to `folder`."""
    if not folder.exists():
        return []
    found = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and needle.encode("utf-8") in path.read_bytes():
            found.append(path.relative_to(folder).as_posix())
    return found
