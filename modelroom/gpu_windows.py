"""Windows: the display adapters `Win32_VideoController` lists, and the memory the registry holds for
them (AP9-GPU G3, ADR 0011).

`AdapterRAM` and the 32-bit `MemorySize` are never read: both stop at 4 GiB. The registry's 64-bit
`HardwareInformation.qwMemorySize` is shown next to a card, never counted (D7, D11): the counted
memory of a card comes from its driver's own tool.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .gpu_usable import exclusion, form_of, graphics_adapter, is_display_adapter, own_verdict, vendor_of_pci
from .llmfit import Runner
from .measure_base import _gib, _run
from .profile import GraphicsAdapter

WINDOWS_ADAPTER_ARGS = (
    "powershell",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "Get-CimInstance Win32_VideoController | Select-Object Name, PNPDeviceID | ConvertTo-Json -Compress",
)
WINDOWS_REGISTRY_ARGS = (
    "powershell",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "Get-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
    "-ErrorAction SilentlyContinue | Select-Object DriverDesc, MatchingDeviceId, "
    "'HardwareInformation.qwMemorySize' | ConvertTo-Json -Compress",
)
QW_MEMORY_SIZE = "HardwareInformation.qwMemorySize"
# The least registry value taken as a memory size: 64 MiB. Anything smaller is no graphics memory.
MIN_REGISTRY_BYTES = 67108864
REGISTRY_ENTRIES_REASON = "the registry holds {entries} entries for {adapters} adapters"

_VEN_DEV_RE = re.compile(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})", re.IGNORECASE)


@dataclass(frozen=True)
class Listed:
    """One `Win32_VideoController`: its name, its `PNPDeviceID`, and the vendor and device read from it."""

    name: str | None
    pnp: str | None
    vendor_id: str | None
    device_id: str | None

    @property
    def key(self) -> str | None:
        return f"VEN_{self.vendor_id}&DEV_{self.device_id}".upper() if self.vendor_id and self.device_id else None


def _objects(stdout: str) -> list | None:
    """`ConvertTo-Json` output as a list of objects: one object alone is a list of one; `None` where it does not read."""
    text = stdout.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return None
    items = data if isinstance(data, list) else [data]
    return items if all(isinstance(item, dict) for item in items) else None


def list_video_controllers(runner: Runner) -> tuple[list[Listed] | None, str | None]:
    """Every video controller, or `(None, reason)` where the list could not be read."""
    stdout, reason = _run(runner, WINDOWS_ADAPTER_ARGS)
    if stdout is None:
        return None, reason
    items = _objects(stdout)
    if items is None:
        return None, "the list of video controllers did not read"
    listed = []
    for item in items:
        name = item.get("Name") if isinstance(item.get("Name"), str) and item.get("Name").strip() else None
        pnp = item.get("PNPDeviceID") if isinstance(item.get("PNPDeviceID"), str) else None
        match = _VEN_DEV_RE.search(pnp or "")
        vendor, device = (match.group(1).lower(), match.group(2).lower()) if match else (None, None)
        listed.append(Listed(name.strip() if name else None, pnp, vendor, device))
    return listed, None


def registry_bytes(value: object) -> int | None:
    """`qwMemorySize` as a number of bytes: a number, or exactly 8 bytes little-endian; else `None`."""
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, list) and len(value) == 8 and all(isinstance(b, int) and not isinstance(b, bool) and 0 <= b <= 255 for b in value):
        number = int.from_bytes(bytes(value), "little")
    else:
        return None
    return number if MIN_REGISTRY_BYTES <= number < 2**64 else None  # a 64-bit value, nothing larger


def registry_memory(runner: Runner, listed: list[Listed]) -> dict[str, tuple[float | None, str | None]]:
    """`VEN_x&DEV_y -> (GiB, reason)` for every listed key the registry names; an entry no adapter has is left out."""
    stdout, _reason = _run(runner, WINDOWS_REGISTRY_ARGS)
    items = _objects(stdout) if stdout is not None else None
    if not items:
        return {}
    adapters = {}
    for entry in listed:
        if entry.key is not None:
            adapters[entry.key] = adapters.get(entry.key, 0) + 1
    entries: dict[str, list[int | None]] = {}
    for item in items:
        match = _VEN_DEV_RE.search(str(item.get("MatchingDeviceId") or ""))
        key = f"VEN_{match.group(1)}&DEV_{match.group(2)}".upper() if match else None
        if key in adapters:
            entries.setdefault(key, []).append(registry_bytes(item.get(QW_MEMORY_SIZE)))
    found = {}
    for key, values in entries.items():
        if all(value is None for value in values):
            continue  # no size at all (integrated graphics): nothing to show, nothing to explain
        complete = None not in values and len(values) >= adapters[key] and len(set(values)) == 1
        if complete:
            found[key] = (_gib(values[0]), None)
        else:  # one value for some of the entries only, or different values: unknown, never guessed
            found[key] = (None, REGISTRY_ENTRIES_REASON.format(entries=len(values), adapters=adapters[key]))
    return found


def windows_adapters(listed: list[Listed], memory: dict[str, tuple[float | None, str | None]]) -> list[GraphicsAdapter]:
    """Every listed controller as an adapter (the caller has taken out what `nvidia-smi` lists itself)."""
    adapters = []
    for entry in listed:
        vendor = vendor_of_pci(entry.vendor_id)
        display = is_display_adapter(entry.vendor_id, entry.name, entry.pnp)
        gib, reason = memory.get(entry.key or "", (None, None))
        verdict = exclusion(vendor, "win32", display=display) or own_verdict(vendor, "win32")
        adapters.append(
            graphics_adapter(
                vendor,
                verdict,
                form_of(vendor, entry.name, "win32"),
                "registry" if gib is not None else "windows",
                name=entry.name,
                memory_gib=gib,
                memory_reason=reason,
                pci_id=f"{entry.vendor_id}:{entry.device_id}" if entry.key else None,
            )
        )
    return adapters
