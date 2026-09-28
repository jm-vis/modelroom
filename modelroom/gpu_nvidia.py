"""`nvidia-smi`: every NVIDIA card with its memory (moved out of `measure.py`, AP9-GPU G0)."""

from __future__ import annotations

from .llmfit import Runner
from .measure_base import MIB, GpuReading, _gib, _run, _whole_number
from .profile import GpuAdapter

NVIDIA_SMI_ARGS = ("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits")


def _nvidia_smi_gpu(runner: Runner) -> GpuReading | None:
    """The `nvidia-smi` verdict, or `None` when it did not answer at all (adapter rule then)."""
    stdout, _ = _run(runner, NVIDIA_SMI_ARGS)
    if stdout is None:
        return None
    rows = [line for line in stdout.splitlines() if line.strip()]
    if not rows:
        return None
    adapters = []
    for row in rows:
        adapter = _nvidia_adapter(row)
        if isinstance(adapter, GpuReading):
            return adapter  # a card without a number is no half measurement: the whole reading is unmeasured
        adapters.append(adapter)
    indexes = [adapter.index for adapter in adapters]
    if len(adapters) == 1 and indexes != [0]:
        note = f"nvidia-smi reports this adapter at index {indexes[0]}, not index 0"
        return GpuReading("present_unmeasured", None, "unknown", adapters[0].name, note)
    if indexes != list(range(len(adapters))):
        note = f"nvidia-smi reports the adapters at indexes {', '.join(map(str, indexes))}, not 0 to {len(adapters) - 1}"
        return GpuReading("present_unmeasured", None, "unknown", None, note)
    return _nvidia_cards(tuple(adapters))


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


def _nvidia_cards(adapters: tuple[GpuAdapter, ...]) -> GpuReading:
    """One card is `measured` as before; two or more are `multi_gpu` with their sum and a note."""
    first = adapters[0]
    if len(adapters) == 1:
        return GpuReading("measured", first.vram_gib, "nvidia-smi", first.name, adapters=adapters)
    names = {adapter.name for adapter in adapters}
    if len(names) == 1:
        listed = f"{len(adapters)} x {first.name or 'graphics card'}"
    else:
        listed = ", ".join(f"{adapter.name or 'graphics card'} {adapter.vram_gib:.2f} GiB" for adapter in adapters)
    total = round(sum(adapter.vram_gib for adapter in adapters), 2)
    note = f"nvidia-smi lists {len(adapters)} adapters: {listed}"
    return GpuReading("multi_gpu", total, "nvidia-smi", first.name if len(names) == 1 else None, note, adapters)
