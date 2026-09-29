"""The graphics adapters of a machine in words: one line each for the terminal, one table for the
Markdown view (AP9-GPU G8, ADR 0011). Never `unknown` where a source knows something; a machine whose
units could not be read says `not recognized` with the reason.
"""

from __future__ import annotations

from typing import Sequence

from .gpu_usable import adapter_label
from .profile import GraphicsAdapter, HardwareProfile

LINE_LABEL = "graphics  "
NOT_RECOGNIZED = "not recognized"
MODE_B_HINT = "rows from llmfit may describe a card listed above"
BACKEND_WORDS = {"cuda": "CUDA", "rocm": "ROCm", "metal": "Metal", "vulkan": "Vulkan"}
ADAPTER_COLUMNS = ("Machine", "#", "Vendor", "Model", "Memory GiB", "Memory", "Form", "Source", "Usable", "Counted", "Reason")
ADAPTER_HINT = (
    "Graphics adapters: every unit a source named. Usable: yes = Ollama has a native backend for it "
    "(CUDA, ROCm, Metal); unconfirmed = no proof; no = shut out. Only counted cards enter the fit; "
    "a machine with none counted computes as a CPU machine."
)


def memory_words(adapter: GraphicsAdapter) -> str:
    """`12.0 GB`, `shared memory`, `unified memory 32.0 GB` or `memory not measured`."""
    if adapter.memory_kind == "unified":
        return f"unified memory {_gb(adapter.memory_gib)}" if adapter.memory_gib is not None else "unified memory, size not measured"
    if adapter.memory_kind == "shared":
        return "shared memory"
    return _gb(adapter.memory_gib) if adapter.memory_gib is not None else "memory not measured"


def _gb(gib: float) -> str:
    """The screen's `GB` (binary, one decimal), as `room.gb` writes it."""
    return f"{gib:.1f} GB"


def usable_words(adapter: GraphicsAdapter) -> str:
    """`usable (CUDA)`, `not counted: <reason>`, `may be usable (Vulkan), not counted: <reason>`, `not usable: <reason>`."""
    if adapter.usable == "yes" and adapter.reason is None:
        return f"usable ({BACKEND_WORDS.get(adapter.backend, adapter.backend)})"
    if adapter.usable == "no":
        return f"not usable: {adapter.reason}"
    if adapter.backend == "vulkan" and adapter.usable == "unconfirmed" and adapter.form != "integrated":
        return f"may be usable (Vulkan), not counted: {adapter.reason}"
    return f"not counted: {adapter.reason}"


def adapter_line(adapter: GraphicsAdapter) -> str:
    """`graphics  <name> · <memory> · <source> · <usable>`."""
    return f"{LINE_LABEL}{adapter_label(adapter)} · {memory_words(adapter)} · {adapter.source} · {usable_words(adapter)}"


def adapter_lines(adapters: Sequence[GraphicsAdapter], gpu_state: str, note: str | None = None) -> list[str]:
    """Every adapter a line, the llmfit hint once under its rows; `not recognized` where nothing was read."""
    if not adapters:
        if gpu_state != "present_unmeasured":
            return []
        reason = note or "the graphics adapters could not be read"
        return [LINE_LABEL + (reason if reason.startswith(NOT_RECOGNIZED) else f"{NOT_RECOGNIZED}: {reason}")]
    lines = [adapter_line(adapter) for adapter in adapters]
    if any(adapter.source == "llmfit" for adapter in adapters):
        lines.append(f"{LINE_LABEL}{MODE_B_HINT}")
    return lines


def adapter_table_rows(machine: str, profile: HardwareProfile | None) -> list[list[str]]:
    """The rows of one machine for the Markdown table `Graphics adapters`."""
    if profile is None:
        return []
    return [
        [
            machine,
            str(number),
            adapter.vendor,
            adapter.name or "-",
            f"{adapter.memory_gib:.2f}" if adapter.memory_gib is not None else "-",
            memory_words(adapter),
            adapter.form,
            adapter.source,
            adapter.usable,
            "yes" if adapter.counted else "no",
            adapter.reason or "",
        ]
        for number, adapter in enumerate(profile.adapters, start=1)
    ]
