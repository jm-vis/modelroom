"""The GPU verdict of this machine: every graphics unit its sources name, which of them count, and the
state the fit gates on (AP9-GPU G7, ADR 0011).

The chain: `nvidia-smi` (and its compute capability), the platform's own list of display adapters
(`lspci -nnD` with sysfs and kfd on Linux, `Win32_VideoController` with the registry on Windows),
`sysctl` on macOS; llmfit only where modelroom counts no card itself (mode B, `gpu_llmfit`). Per
backend exactly one own source may count (D7): CUDA from `nvidia-smi`, ROCm from sysfs; the registry
never counts. Of the counted backends, the one with the largest sum counts (F8).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from .gpu_linux import LSPCI_ARGS, is_wsl, linux_adapters, lspci_adapters
from .gpu_llmfit import llmfit_adapters
from .gpu_macos import apple_reading, apple_silicon, intel_mac_reading
from .gpu_nvidia import card_details, nvidia_adapters, read_nvidia
from .gpu_usable import (
    DISPLAY_ONLY_NOTE,
    DISPLAY_ONLY_VENDORS,  # noqa: F401 -- exported back through `measure`
    NO_NVIDIA_SMI_REASON,
    NVIDIA_TOOLS_HINT,
    OTHER_BACKEND_REASON,
    WSL_STANDIN_REASON,
    adapter_label,
    is_display_only,
    recognized,
)
from .gpu_windows import list_video_controllers, registry_memory, windows_adapters
from .llmfit import LlmfitReference, Runner
from .measure_base import FileText, GpuReading, ListDir, _run, no_listing
from .profile import GpuAdapter, GraphicsAdapter

# The platforms whose GPU this measures at all; everywhere else is `unsupported_platform`.
MEASURED_PLATFORMS: frozenset[str] = frozenset({"win32", "linux", "darwin"})
INTEL_VENDOR = "8086"
UNREADABLE_NOTE = "a graphics adapter is present but nothing about it could be read"
NOT_RECOGNIZED = "not recognized: "
WSL_NOTE = f"{WSL_STANDIN_REASON}; nvidia-smi did not answer"
MODE_B_NOTE = "no card could be counted from modelroom's own reading; the counted cards are llmfit's"
NO_COUNTED_NOTE = "no counted graphics card: "
NOT_LISTED_REASON = "nvidia-smi does not list this card"
UNREADABLE_SMI_REASON = "the answer of nvidia-smi could not be read"
_BACKEND_ORDER = ("cuda", "rocm")


def _no_file(path: str) -> str:
    raise FileNotFoundError(f"{path}: no such file")


def measure_gpu(
    platform: str,
    runner: Runner,
    read_text: FileText | None = None,
    list_dir: ListDir | None = None,
    reference: LlmfitReference | None = None,
    ram_gib: float | None = None,
) -> GpuReading:
    """The GPU state, the counted graphics memory and every adapter of this machine (G7).

    Without a file layer no file is read, without `reference` llmfit adds nothing: the real
    layers come from `measure.measure_hardware`. `ram_gib` is the memory of a Mac (unified memory).
    """
    read = read_text if read_text is not None else _no_file
    listing = list_dir if list_dir is not None else no_listing
    llmfit = reference if reference is not None and reference.status == "available" else None
    if platform not in MEASURED_PLATFORMS:
        return GpuReading("unsupported_platform", None, "unknown", None, f"the GPU is not measured on {platform} yet")
    if platform == "darwin":
        evidence, name = apple_silicon(runner, llmfit.gpus if llmfit else ())
        return apple_reading(runner, ram_gib, evidence, name) if evidence else intel_mac_reading(runner)
    nvidia = read_nvidia(runner)
    # An answer of `nvidia-smi` that does not read is one failed source, not the end of the
    # measurement: its diagnosis stays, and every other source and llmfit are still asked.
    problem = nvidia.failure.note if nvidia.failure is not None else None
    capabilities, slots = card_details(runner) if nvidia.cards else ({}, {})
    own = nvidia_adapters(nvidia.cards, capabilities, platform, slots) if nvidia.cards else []
    failed = [] if nvidia.cards else [problem or nvidia.reason or NO_NVIDIA_SMI_REASON]
    listed, reason, wsl = _platform_adapters(platform, runner, read, listing, own)
    listed = [_nvidia_reason(adapter, bool(nvidia.cards), problem) for adapter in listed]
    if reason is not None:
        failed.append(reason)
    failed.append(llmfit_reason(reference))
    reading = decide(own + listed, llmfit, platform, failed=failed, listed=reason is None, wsl=wsl)
    if problem is not None and problem not in (reading.note or ""):
        reading = replace(reading, note="; ".join(part for part in (reading.note, problem) if part))
    return reading


def _platform_adapters(platform: str, runner: Runner, read: FileText, listing: ListDir, cards: list[GraphicsAdapter]):
    """`(adapters, reason the list failed, WSL)` of the platform's own list of display adapters,
    without the entries that are, beyond doubt, a card `nvidia-smi` already lists (`unmatched`)."""
    if platform == "linux":
        stdout, reason = _run(runner, LSPCI_ARGS)
        if stdout is None:
            return [], reason, False
        entries = unmatched(lspci_adapters(stdout), cards)
        wsl = any(entry.vendor_id == "1414" for entry in entries) and is_wsl(read)
        return linux_adapters(entries, read, listing, wsl=wsl), None, wsl
    entries, reason = list_video_controllers(runner)
    if entries is None:
        return [], reason, False
    entries = unmatched(entries, cards)
    memory = registry_memory(runner, entries) if entries else {}
    return windows_adapters(entries, memory), None, False


def unmatched(entries: list, cards: list[GraphicsAdapter]) -> list:
    """The listed entries without the NVIDIA ones that are certainly a card `nvidia-smi` lists.

    Certain means, first: the same PCI address, where both sides carry one (`lspci`); an entry at
    another address is another card, whatever the counts. Then, for the entries that carry no
    address (`Win32_VideoController`) and the cards no address matched: the same name, one entry per
    card. Every other NVIDIA entry stays, shown and not counted: a card `nvidia-smi` does not list.
    """
    if not cards or not any(entry.vendor_id == "10de" for entry in entries):
        return list(entries)
    slots = {getattr(entry, "slot", None) for entry in entries if entry.vendor_id == "10de"} - {None}
    by_address = {card.pci_slot for card in cards if card.pci_slot in slots}
    names = Counter(card.name for card in cards if card.pci_slot not in by_address)
    kept = []
    for entry in entries:
        slot = getattr(entry, "slot", None)
        if entry.vendor_id == "10de" and slot is not None and slot in by_address:
            continue
        if entry.vendor_id == "10de" and slot is None and entry.name and names[entry.name] > 0:
            names[entry.name] -= 1
            continue
        kept.append(entry)
    return kept


def _nvidia_reason(adapter: GraphicsAdapter, answered: bool, problem: str | None) -> GraphicsAdapter:
    """An NVIDIA entry of the list next to an answer of `nvidia-smi`: why its card is not in it."""
    if adapter.vendor != "nvidia" or not adapter.reason or NO_NVIDIA_SMI_REASON not in adapter.reason:
        return adapter
    if answered:
        return adapter.model_copy(update={"reason": adapter.reason.replace(NO_NVIDIA_SMI_REASON, NOT_LISTED_REASON, 1)})
    if problem is not None:
        return adapter.model_copy(update={"reason": adapter.reason.replace(NO_NVIDIA_SMI_REASON, UNREADABLE_SMI_REASON, 1)})
    return adapter


def decide(
    own: list[GraphicsAdapter], llmfit: LlmfitReference | None, platform: str, *, failed: list[str], listed: bool, wsl: bool
) -> GpuReading:
    """The state, the first row of the table in ADR 0011 that applies (rows 3 to 10)."""
    counted = count_adapters(own)
    if any(adapter.counted for adapter in counted):
        return counted_reading(counted, mode_b=False)
    extra = count_adapters(llmfit_adapters(llmfit.gpus, platform, own) if llmfit else [])
    every = own + extra
    if any(adapter.counted for adapter in extra):
        return counted_reading(every, mode_b=True)
    if wsl and own and not extra and all(adapter.reason == WSL_STANDIN_REASON for adapter in own):
        return GpuReading("present_unmeasured", None, "unknown", None, WSL_NOTE, graphics=tuple(every))
    if any(recognized(adapter) for adapter in every):
        return GpuReading("none", 0.0, "none", None, no_counted_note(every), graphics=tuple(every))
    if every and all(is_display_only(adapter) for adapter in every):
        return GpuReading("none", 0.0, "none", None, DISPLAY_ONLY_NOTE, graphics=tuple(every))
    if every:
        return GpuReading("present_unmeasured", None, "unknown", None, UNREADABLE_NOTE, graphics=tuple(every))
    if not listed:
        return GpuReading("present_unmeasured", None, "unknown", None, NOT_RECOGNIZED + "; ".join(failed))
    return GpuReading("none", 0.0, "none")


def llmfit_reason(reference: LlmfitReference | None) -> str:
    """What llmfit said about the graphics units, for the `not recognized` line."""
    if reference is None:
        return "llmfit was not asked"
    if reference.status != "available":
        return reference.reason or f"llmfit is {reference.status}"
    return "llmfit lists no graphics card"


def count_adapters(adapters: list[GraphicsAdapter]) -> list[GraphicsAdapter]:
    """Count the backend with the largest sum (F8, a tie goes to CUDA); every other candidate keeps
    `OTHER_BACKEND_REASON`. A candidate is a card `graphics_adapter` left with that reason."""
    sums: dict[str, float] = {}
    for adapter in adapters:
        if adapter.reason == OTHER_BACKEND_REASON and adapter.memory_gib is not None:
            sums[adapter.backend] = sums.get(adapter.backend, 0.0) + adapter.memory_gib
    if not sums:
        return list(adapters)
    winner = max(sums, key=lambda backend: (sums[backend], -_rank(backend)))
    return [
        adapter.model_copy(update={"counted": True, "reason": None})
        if adapter.reason == OTHER_BACKEND_REASON and adapter.backend == winner
        else adapter
        for adapter in adapters
    ]


def _rank(backend: str) -> int:
    return _BACKEND_ORDER.index(backend) if backend in _BACKEND_ORDER else len(_BACKEND_ORDER)


def counted_reading(adapters: list[GraphicsAdapter], *, mode_b: bool) -> GpuReading:
    """One counted card is `measured`, two or more are `multi_gpu` with their sum; one source."""
    counted = [adapter for adapter in adapters if adapter.counted]
    source = counted[0].source
    cards = tuple(
        GpuAdapter(index=index, name=adapter.name, vram_gib=adapter.memory_gib, vram_source=source)
        for index, adapter in enumerate(counted)
    )
    first = cards[0]
    notes = [MODE_B_NOTE] if mode_b else []
    if len(cards) == 1:
        return GpuReading("measured", first.vram_gib, source, first.name, "; ".join(notes) or None, cards, tuple(adapters))
    names = {card.name for card in cards}
    if len(names) == 1:
        listed = f"{len(cards)} x {first.name or 'graphics card'}"
    else:
        listed = ", ".join(f"{card.name or 'graphics card'} {card.vram_gib:.2f} GiB" for card in cards)
    total = round(sum(card.vram_gib for card in cards), 2)
    notes.insert(0, f"{source} lists {len(cards)} adapters: {listed}")
    return GpuReading("multi_gpu", total, source, first.name if len(names) == 1 else None, "; ".join(notes), cards, tuple(adapters))


def no_counted_note(adapters: list[GraphicsAdapter]) -> str:
    """Why no card counts, card by card; the NVIDIA tools named where their answer was missing."""
    parts = [f"{adapter_label(adapter)}: {adapter.reason}" for adapter in adapters if adapter.reason and not is_display_only(adapter)]
    note = NO_COUNTED_NOTE + "; ".join(parts)
    if any(adapter.vendor == "nvidia" and NO_NVIDIA_SMI_REASON in (adapter.reason or "") for adapter in adapters):
        note += f"; {NVIDIA_TOOLS_HINT}"
    return note
