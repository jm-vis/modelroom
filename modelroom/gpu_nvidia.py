"""`nvidia-smi`: every NVIDIA card with its memory, and its compute capability (AP9-GPU F16).

The card list is the one this module always read (moved out of `measure.py`, G0); a second call
asks for the compute capability, which alone can shut a card out (below 5.0, docs.ollama.com/gpu).
A capability that cannot be read shuts nothing out: the card list itself is the proof of CUDA.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .gpu_usable import exclusion, graphics_adapter, own_verdict
from .llmfit import Runner
from .measure_base import MIB, GpuReading, _gib, _run, _whole_number
from .profile import GpuAdapter, GraphicsAdapter

NVIDIA_SMI_ARGS = ("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits")
NVIDIA_CC_ARGS = ("nvidia-smi", "--query-gpu=index,compute_cap,pci.bus_id", "--format=csv,noheader,nounits")
# `nvidia-smi` writes the PCI domain with eight digits (`00000000:01:00.0`); lspci writes four.
_BUS_ID_RE = re.compile(r"^([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$")


@dataclass(frozen=True)
class NvidiaReading:
    """What `nvidia-smi` said: the cards, a reading that is unmeasured as a whole, or no answer."""

    cards: tuple[GpuAdapter, ...] = ()
    failure: GpuReading | None = None
    reason: str | None = None


def read_nvidia(runner: Runner) -> NvidiaReading:
    """The cards `nvidia-smi` lists; `failure` where its answer does not read, `reason` where there was none."""
    stdout, reason = _run(runner, NVIDIA_SMI_ARGS)
    if stdout is None:
        return NvidiaReading(reason=reason)
    rows = [line for line in stdout.splitlines() if line.strip()]
    if not rows:
        return NvidiaReading(reason="nvidia-smi lists no card")
    adapters = []
    for row in rows:
        adapter = _nvidia_adapter(row)
        if isinstance(adapter, GpuReading):
            return NvidiaReading(failure=adapter)  # a card without a number is no half measurement
        adapters.append(adapter)
    indexes = [adapter.index for adapter in adapters]
    if len(adapters) == 1 and indexes != [0]:
        note = f"nvidia-smi reports this adapter at index {indexes[0]}, not index 0"
        return NvidiaReading(failure=GpuReading("present_unmeasured", None, "unknown", adapters[0].name, note))
    if indexes != list(range(len(adapters))):
        note = f"nvidia-smi reports the adapters at indexes {', '.join(map(str, indexes))}, not 0 to {len(adapters) - 1}"
        return NvidiaReading(failure=GpuReading("present_unmeasured", None, "unknown", None, note))
    return NvidiaReading(cards=tuple(adapters))


def _nvidia_adapter(row: str) -> GpuAdapter | GpuReading:
    """One row of `nvidia-smi` as a card, or the unmeasured reading that says why it is none."""
    fields = [field.strip() for field in row.split(",")]
    index = _whole_number(fields[0]) if fields else None
    mib = _whole_number(fields[2]) if len(fields) == 3 else None
    if len(fields) != 3 or index is None or mib is None:
        return GpuReading("present_unmeasured", None, "unknown", None, f"nvidia-smi output could not be read: {row.strip()[:120]!r}")
    if index < 0:
        return GpuReading("present_unmeasured", None, "unknown", fields[1], f"nvidia-smi reports this adapter at index {index}, not index 0")
    vram_gib = _gib(mib * MIB)
    if vram_gib is None:
        note = f"nvidia-smi reports no usable memory for this adapter: {str(mib)[:40]} MiB"
        return GpuReading("present_unmeasured", None, "unknown", fields[1], note)
    return GpuAdapter(index=index, name=fields[1], vram_gib=vram_gib, vram_source="nvidia-smi")


def card_details(runner: Runner) -> tuple[dict[int, float], dict[int, str]]:
    """`(index -> compute capability, index -> PCI address)` of every card whose line reads; both
    empty where the call does not. The address ties a card to its line of `lspci` (Linux)."""
    stdout, _reason = _run(runner, NVIDIA_CC_ARGS)
    capabilities: dict[int, float] = {}
    slots: dict[int, str] = {}
    for row in (stdout or "").splitlines():
        parts = [part.strip() for part in row.split(",")]
        index = _whole_number(parts[0]) if len(parts) in (2, 3) else None
        if index is None:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            value = float("nan")
        if value == value and 0 < value < 100:  # a real version number, not NaN
            capabilities[index] = value
        match = _BUS_ID_RE.match(parts[2]) if len(parts) == 3 else None
        if match:
            domain = int(match.group(1), 16)
            slots[index] = f"{domain:04x}:{match.group(2).lower()}:{match.group(3).lower()}.{match.group(4)}"
    return capabilities, slots


def nvidia_adapters(
    cards: tuple[GpuAdapter, ...], capabilities: dict[int, float], platform: str, slots: dict[int, str] | None = None
) -> list[GraphicsAdapter]:
    """Every card `nvidia-smi` listed as an adapter: CUDA, unless its compute capability shuts it out."""
    adapters = []
    for card in cards:
        verdict = exclusion("nvidia", platform, compute_capability=capabilities.get(card.index)) or own_verdict(
            "nvidia", platform, nvidia_listed=True
        )
        slot = (slots or {}).get(card.index)
        adapters.append(graphics_adapter("nvidia", verdict, "discrete", "nvidia-smi", name=card.name, memory_gib=card.vram_gib, pci_slot=slot))
    return adapters
