"""Linux: the display adapters `lspci` lists (moved out of `measure.py`, AP9-GPU G0)."""

from __future__ import annotations

import re

LSPCI_ARGS = ("lspci", "-nn")
UNKNOWN_VENDOR = "unknown"

_LSPCI_CLASS_RE = re.compile(r"^\S+\s+.*?\[([0-9a-fA-F]{4})\]:")
_LSPCI_ID_RE = re.compile(r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]")


def _lspci_vendors(stdout: str) -> list[str]:
    """Every `lspci -nn` line of PCI class 03 (display controllers), by vendor id."""
    vendors = []
    for line in stdout.splitlines():
        device_class = _LSPCI_CLASS_RE.match(line)
        if device_class is None or not device_class.group(1).startswith("03"):
            continue
        ids = _LSPCI_ID_RE.findall(line)
        vendors.append(ids[-1][0].lower() if ids else UNKNOWN_VENDOR)
    return vendors
