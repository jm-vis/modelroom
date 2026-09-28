"""macOS: Apple Silicon as one pool of unified memory, an Intel Mac as a CPU machine (AP9-GPU G5, D9).

Where the memory comes from and what proves Apple Silicon are two readings, kept apart: the memory
is always the RAM `sysctl hw.memsize` read (never llmfit's), the proof is `sysctl
hw.optional.arm64`, else an llmfit entry `Metal` with unified memory. An Intel Mac runs Ollama on the
CPU (F15); its graphics units are listed by `system_profiler` and none of them is usable.
"""

from __future__ import annotations

import json
import re

from .gpu_usable import INTEL_MAC_REASON, Verdict, exclusion, form_of, graphics_adapter, vendor_of_name
from .llmfit import Runner
from .measure_base import GIB, MIB, GpuReading, _run
from .profile import GraphicsAdapter

MACOS_ARM64_ARGS = ("sysctl", "-n", "hw.optional.arm64")
MACOS_CHIP_ARGS = ("sysctl", "-n", "machdep.cpu.brand_string")
SYSTEM_PROFILER_ARGS = ("system_profiler", "SPDisplaysDataType", "-json")
APPLE_GRAPHICS = "Apple graphics"
MACOS_RESERVE_NOTE = "macOS keeps part of the memory for the system"
_VRAM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(MB|GB)\s*$", re.IGNORECASE)


def apple_silicon(runner: Runner, llmfit_gpus: tuple) -> tuple[str | None, str | None]:
    """`(evidence, llmfit name)`: `sysctl` where `hw.optional.arm64` is 1; a valid `0` is an Intel
    Mac, whatever llmfit says; only where the reading failed or does not read, `llmfit` where it
    lists `Metal` with unified memory; else `(None, None)` -- an Intel Mac."""
    stdout, _reason = _run(runner, MACOS_ARM64_ARGS)
    metal = next((gpu for gpu in llmfit_gpus if gpu.backend == "Metal" and gpu.unified_memory), None)
    answer = stdout.strip() if stdout is not None else None
    if answer == "1":
        return "sysctl", metal.name if metal else None
    if answer == "0":
        return None, None
    if metal is not None:
        return "llmfit", metal.name
    return None, None


def apple_reading(runner: Runner, ram_gib: float | None, evidence: str, llmfit_name: str | None) -> GpuReading:
    """Apple Silicon: `unified_memory`, one unified unit holding the RAM from `sysctl`, or both unknown."""
    stdout, _reason = _run(runner, MACOS_CHIP_ARGS)
    chip = stdout.strip() if stdout is not None and stdout.strip() else None
    name = chip or llmfit_name or APPLE_GRAPHICS
    verdict = Verdict("yes", "metal", evidence=evidence)
    adapter = graphics_adapter("apple", verdict, "unified", "sysctl", name=name, memory_gib=ram_gib)
    source = "sysctl" if ram_gib is not None else "unknown"
    return GpuReading("unified_memory", ram_gib, source, name, MACOS_RESERVE_NOTE, graphics=(adapter,))


def intel_mac_reading(runner: Runner) -> GpuReading:
    """An Intel Mac: a CPU machine (F15); every unit `system_profiler` lists is shown, none usable."""
    stdout, _reason = _run(runner, SYSTEM_PROFILER_ARGS)
    adapters = tuple(_profiler_adapters(stdout or ""))
    return GpuReading("none", 0.0, "none", None, INTEL_MAC_REASON, graphics=adapters)


def _profiler_adapters(stdout: str) -> list[GraphicsAdapter]:
    try:
        data = json.loads(stdout)
    except ValueError:
        return []
    entries = data.get("SPDisplaysDataType") if isinstance(data, dict) else None
    adapters = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("sppci_model"), str):
            continue
        name = entry["sppci_model"].strip() or None
        vendor = vendor_of_name(name)
        verdict = exclusion(vendor, "darwin", intel_mac=True)
        memory = _vram_gib(entry.get("spdisplays_vram"))
        adapters.append(graphics_adapter(vendor, verdict, form_of(vendor, name, "darwin"), "system_profiler", name=name, memory_gib=memory))
    return adapters


def _vram_gib(value: object) -> float | None:
    """`spdisplays_vram` (`8 GB`, `1536 MB`) in GiB, or `None`."""
    match = _VRAM_RE.match(value) if isinstance(value, str) else None
    if match is None:
        return None
    number = float(match.group(1)) * (GIB if match.group(2).upper() == "GB" else MIB)
    gib = round(number / GIB, 2)
    return gib if gib > 0 else None
