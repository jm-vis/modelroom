"""Windows: the display adapters `Win32_VideoController` lists (moved out of `measure.py`, AP9-GPU G0)."""

from __future__ import annotations

import re

from .gpu_linux import UNKNOWN_VENDOR

WINDOWS_ADAPTER_ARGS = (
    "powershell",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID }",
)

_PNP_VENDOR_RE = re.compile(r"VEN_([0-9a-fA-F]{4})")


def _pnp_vendors(stdout: str) -> list[str]:
    """Every `Win32_VideoController.PNPDeviceID`, by vendor id; a line without one is unknown."""
    vendors = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        match = _PNP_VENDOR_RE.search(line)
        vendors.append(match.group(1).lower() if match else UNKNOWN_VENDOR)
    return vendors
