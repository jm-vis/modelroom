"""llmfit as the fallback and as the cross-check of the graphics memory (AP9-GPU G6, D2, D3, D10).

Two modes, never mixed (D2). Mode A: modelroom counts a card of its own -- then llmfit adds no
adapter and is only held against that reading. Mode B: modelroom counts none -- then llmfit's own
`CUDA`/`ROCm` entries may count, llmfit alone. Every exclusion of modelroom comes first, in mode B
too (D3): a vendor modelroom found unusable does not come back through llmfit (Z1: Vulkan and SYCL
entries included).
"""

from __future__ import annotations

from .gpu_usable import (
    EXCLUDED_VENDOR_REASON,
    LLMFIT_CAP_REASON,
    OTHER_BACKEND_REASON,
    Verdict,
    exclusion,
    form_of,
    graphics_adapter,
    llmfit_verdict,
    vendor_of_name,
)
from .llmfit import LlmfitGpu, LlmfitReference
from .measure_base import GpuReading
from .profile import CrossCheck, GraphicsAdapter, crosscheck

# The backends that become adapters: CUDA and ROCm may count, Vulkan and SYCL are shown (F3).
ADAPTER_BACKENDS = ("CUDA", "ROCm", "Vulkan", "SYCL")
# llmfit on Windows reads `AdapterRAM` first, which stops at 4 GiB (F10, R4): such a value never counts.
WINDOWS_CAP_GB = 4.0
NO_LLMFIT_MEMORY_REASON = "llmfit reads no memory for this card"
BACKEND_NAMES = {"cuda": "CUDA", "rocm": "ROCm"}
BACKEND_VENDORS = {"cuda": "nvidia", "rocm": "amd"}


def llmfit_adapters(gpus: tuple[LlmfitGpu, ...], platform: str, own: list[GraphicsAdapter]) -> list[GraphicsAdapter]:
    """Mode B: `count` adapters per entry of a backend that has one, each through the exclusions first.

    A card that may count carries `OTHER_BACKEND_REASON` until the count takes it (`gpu.count_adapters`);
    a Windows value at or below 4.0 GB (the raw number, F10) and a card without memory (Z2) never do.
    """
    blocked = {adapter.vendor for adapter in own if adapter.usable == "no" and adapter.vendor != "other"}
    adapters = []
    for entry in gpus:
        if entry.backend not in ADAPTER_BACKENDS:
            continue
        vendor = vendor_of_name(entry.name)
        fallback = llmfit_verdict(entry.backend)
        verdict = exclusion(vendor, platform) or (
            Verdict("no", fallback.backend, EXCLUDED_VENDOR_REASON) if vendor in blocked else fallback
        )
        memory = round(entry.vram_gb, 2) if entry.vram_gb > 0 and round(entry.vram_gb, 2) > 0 else None
        adapter = graphics_adapter(
            vendor, verdict, form_of(vendor, entry.name, platform), "llmfit",
            name=entry.name, memory_gib=memory, memory_reason=NO_LLMFIT_MEMORY_REASON,
        )
        if adapter.reason == OTHER_BACKEND_REASON and platform == "win32" and entry.vram_gb <= WINDOWS_CAP_GB:
            adapter = adapter.model_copy(update={"reason": LLMFIT_CAP_REASON})
        adapters += [adapter] * entry.count
    return adapters


def vram_crosscheck(gpu: GpuReading, reference: LlmfitReference, cpu_only: bool) -> CrossCheck:
    """The cross-check of the graphics memory, the first row of the table in ADR 0011 that applies."""
    own = gpu.vram_gib
    available = reference.status == "available"
    if cpu_only:
        return CrossCheck(status="absent", own_gib=own)
    if own is None:
        return _absent_or_error(own, reference)
    counted = [adapter for adapter in gpu.graphics if adapter.counted]
    if gpu.gpu_state in ("measured", "multi_gpu") and counted and counted[0].source == "llmfit":
        return CrossCheck(status="not_comparable", own_gib=own, llmfit_gib=own)
    if gpu.gpu_state == "multi_gpu" and len({card.vram_gib for card in gpu.adapters}) > 1:
        llmfit_gib = reference.cuda_vram_gib if available else None
        return CrossCheck(status="not_comparable", own_gib=own, llmfit_gib=llmfit_gib)
    if gpu.gpu_state in ("measured", "multi_gpu") and counted:
        if not available:
            return _absent_or_error(own, reference)
        return _same_cards(gpu, counted, reference)
    if gpu.gpu_state == "none" and gpu.graphics:
        return CrossCheck(status="not_comparable", own_gib=own)
    return _one_check(own, reference.vram_gib, reference.status)


def _same_cards(gpu: GpuReading, counted: list[GraphicsAdapter], reference: LlmfitReference) -> CrossCheck:
    """Mode A: compared only where llmfit's entries of the counted backend are exactly the counted cards."""
    backend = counted[0].backend
    entries = [entry for entry in reference.gpus if entry.backend == BACKEND_NAMES.get(backend)]
    vendor = BACKEND_VENDORS.get(backend)
    uncounted = any(adapter.vendor == vendor and not adapter.counted for adapter in gpu.graphics)
    if sum(entry.count for entry in entries) == len(counted) and not uncounted:
        return crosscheck(gpu.vram_gib, round(sum(entry.vram_gb * entry.count for entry in entries), 2))
    return CrossCheck(status="not_comparable", own_gib=gpu.vram_gib, llmfit_gib=reference.vram_gib)


def _absent_or_error(own: float | None, reference: LlmfitReference) -> CrossCheck:
    return CrossCheck(status="error" if reference.status == "error" else "absent", own_gib=own)


def _one_check(own_gib: float | None, llmfit_gib: float | None, status: str) -> CrossCheck:
    if status == "available" and own_gib is not None and llmfit_gib is not None:
        return crosscheck(own_gib, llmfit_gib)
    return CrossCheck(status="error" if status == "error" else "absent", own_gib=own_gib)
