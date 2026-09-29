"""Linux and WSL: the display adapters `lspci -nnD` lists, the memory of an AMD card from sysfs, and
the ROCm compute driver (kfd) of each card (AP9-GPU G4, ADR 0011).

Sources, each checked against the kernel tree before it was built in (D5, D8, D12):

- `mem_info_vram_total` / `mem_info_vram_used`: bytes, `drivers/gpu/drm/amd/amdgpu/amdgpu_vram_mgr.c`
  (documented in `Documentation/gpu/amdgpu/driver-misc.rst`), tag v6.16;
- `/sys/class/kfd/kfd/topology/nodes/*/properties`, lines `location_id` (`pci_dev_id`: bus * 256 +
  device * 8 + function) and `domain` (`pci_domain_nr`), `drivers/gpu/drm/amd/amdkfd/kfd_topology.c`,
  tag v6.16;
- Intel: neither the xe `tile*/physical_vram_size_bytes` nor an i915 `lmem_total_bytes` is in a
  tagged kernel (checked 2026-09-28), so no Intel memory is read (D12: a source that cannot be shown is left out).

A card is tied to its sysfs folder only by `PCI_SLOT_NAME` in `card*/device/uevent`, never by the
order of the `card*` folders; a card without exactly one match has no sysfs value (D8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .gpu_usable import NO_SYSFS_REASON, exclusion, form_of, graphics_adapter, is_display_adapter, own_verdict, vendor_of_pci
from .measure_base import FileText, ListDir, _gib, _whole_number
from .profile import GraphicsAdapter

LSPCI_ARGS = ("lspci", "-nnD")
UNKNOWN_VENDOR = "unknown"
DRM_CLASS = "/sys/class/drm"
KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"
WSL_OSRELEASE_FILE = "/proc/sys/kernel/osrelease"
VRAM_TOTAL_FILE = "mem_info_vram_total"
VRAM_USED_FILE = "mem_info_vram_used"

_LSPCI_CLASS_RE = re.compile(r"^\S+\s+.*?\[([0-9a-fA-F]{4})\]:")
_LSPCI_ID_RE = re.compile(r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]")
_LSPCI_LINE_RE = re.compile(r"^(?:([0-9a-fA-F]{4}):)?([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])\s+.*?\[[0-9a-fA-F]{4}\]:\s*(.*)$")
_CARD_RE = re.compile(r"^card[0-9]+$")


@dataclass(frozen=True)
class LinuxListed:
    """One class-03 line of `lspci -nnD`: its address, its vendor and device, its name."""

    slot: str | None
    vendor_id: str | None
    device_id: str | None
    name: str | None


def _lspci_vendors(stdout: str) -> list[str]:
    """Every `lspci` line of PCI class 03 (display controllers), by vendor id."""
    vendors = []
    for line in stdout.splitlines():
        device_class = _LSPCI_CLASS_RE.match(line)
        if device_class is None or not device_class.group(1).startswith("03"):
            continue
        ids = _LSPCI_ID_RE.findall(line)
        vendors.append(ids[-1][0].lower() if ids else UNKNOWN_VENDOR)
    return vendors


def lspci_adapters(stdout: str) -> list[LinuxListed]:
    """Every class-03 line with its address (`0000:` added where `lspci -nn` left the domain out)."""
    listed = []
    for line in stdout.splitlines():
        device_class = _LSPCI_CLASS_RE.match(line)
        if device_class is None or not device_class.group(1).startswith("03"):
            continue
        ids = _LSPCI_ID_RE.findall(line)
        vendor, device = (ids[-1][0].lower(), ids[-1][1].lower()) if ids else (None, None)
        match = _LSPCI_LINE_RE.match(line)
        slot = name = None
        if match:
            slot = f"{(match.group(1) or '0000').lower()}:{match.group(2).lower()}:{match.group(3).lower()}.{match.group(4)}"
            name = _lspci_name(match.group(5))
        listed.append(LinuxListed(slot, vendor, device, name))
    return listed


def _lspci_name(text: str) -> str | None:
    """The name part of a line: everything before the last `[vendor:device]`; `Device` alone is no name."""
    cut = list(_LSPCI_ID_RE.finditer(text))
    name = (text[: cut[-1].start()] if cut else text).strip()
    return None if name in ("", "Device") else name


def is_wsl(read_text: FileText) -> bool:
    """WSL: the kernel's release names Microsoft (F19)."""
    try:
        return "microsoft" in read_text(WSL_OSRELEASE_FILE).lower()
    except OSError:
        return False


def drm_slots(read_text: FileText, list_dir: ListDir) -> dict[str, list[str]]:
    """`PCI address -> [card folder, ...]` from every `card*/device/uevent`; nothing where sysfs does not list."""
    try:
        names = [name for name in list_dir(DRM_CLASS) if _CARD_RE.match(name)]
    except OSError:
        return {}
    slots: dict[str, list[str]] = {}
    for name in names:
        try:
            text = read_text(f"{DRM_CLASS}/{name}/device/uevent")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                slots.setdefault(line.split("=", 1)[1].strip().lower(), []).append(name)
    return slots


def sysfs_bytes(read_text: FileText, card: str, file: str, minimum: int = 1) -> int | None:
    """One whole number of bytes from `card*/device/<file>`, at least `minimum`; `None` where it does not read."""
    try:
        value = _whole_number(read_text(f"{DRM_CLASS}/{card}/device/{file}").strip())
    except OSError:
        return None
    # `%llu`: a value beyond 64 bits is no byte count the kernel wrote, and no float could hold it anyway.
    return value if value is not None and minimum <= value < 2**64 else None


def kfd_slots(read_text: FileText, list_dir: ListDir) -> set[str]:
    """The PCI address of every kfd node (D5): `domain` and `location_id` of its `properties`."""
    try:
        nodes = list_dir(KFD_NODES)
    except OSError:
        return set()
    slots = set()
    for node in nodes:
        try:
            text = read_text(f"{KFD_NODES}/{node}/properties")
        except OSError:
            continue
        fields = dict(line.split(None, 1) for line in text.splitlines() if len(line.split(None, 1)) == 2)
        domain, location = _whole_number(fields.get("domain", "").strip()), _whole_number(fields.get("location_id", "").strip())
        if domain is None or location is None or location <= 0 or domain < 0:
            continue
        bus, device, function = location >> 8, (location >> 3) & 0x1F, location & 0x7
        slots.add(f"{domain:04x}:{bus:02x}:{device:02x}.{function}")
    return slots


def linux_adapters(
    listed: list[LinuxListed], read_text: FileText, list_dir: ListDir, *, wsl: bool
) -> list[GraphicsAdapter]:
    """Every listed device as an adapter (the caller has taken out what `nvidia-smi` lists itself)."""
    amd = any(vendor_of_pci(entry.vendor_id) == "amd" for entry in listed)
    slots = drm_slots(read_text, list_dir) if amd else {}
    kfd = kfd_slots(read_text, list_dir) if amd else set()
    return [_one_adapter(entry, vendor_of_pci(entry.vendor_id), read_text, slots, kfd, wsl) for entry in listed]


def _one_adapter(entry: LinuxListed, vendor, read_text: FileText, slots: dict[str, list[str]], kfd: set[str], wsl: bool) -> GraphicsAdapter:
    display = is_display_adapter(entry.vendor_id, entry.name, wsl=wsl)
    standin = wsl and (entry.vendor_id or "") == "1414"
    memory = memory_reason = None
    if vendor == "amd":
        cards = slots.get(entry.slot or "", [])
        if len(cards) == 1:
            memory = _gib(sysfs_bytes(read_text, cards[0], VRAM_TOTAL_FILE))
            memory_reason = None if memory is not None else f"{VRAM_TOTAL_FILE} could not be read"
        else:
            memory_reason = NO_SYSFS_REASON
    verdict = exclusion(vendor, "linux", display=display) or own_verdict(
        vendor, "linux", kfd=(entry.slot in kfd), wsl_standin=standin
    )
    return graphics_adapter(
        vendor,
        verdict,
        form_of(vendor, entry.name, "linux", slot=entry.slot),
        "sysfs" if memory is not None else "lspci",
        name=entry.name,
        memory_gib=memory,
        memory_reason=memory_reason,
        pci_id=f"{entry.vendor_id}:{entry.device_id}" if entry.vendor_id and entry.device_id else None,
        pci_slot=entry.slot,
    )
