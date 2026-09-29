"""Shared helpers of the AP9-GPU tests (`tests/test_gpu_*.py`): answers, runners, llmfit readings.

Not a test module itself. Every command a graphics measurement can run has an answer here: `not
installed` unless a test gives one of its own, so a test names only the sources it is about.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from modelroom.llmfit import FixtureRunner, LlmfitGpu, LlmfitReference
from modelroom.measure import (
    LSPCI_ARGS,
    MACOS_ARM64_ARGS,
    MACOS_CHIP_ARGS,
    MACOS_MEMSIZE_ARGS,
    MACOS_PLATFORM_UUID_ARGS,
    NVIDIA_CC_ARGS,
    NVIDIA_SMI_ARGS,
    SYSTEM_PROFILER_ARGS,
    WINDOWS_ADAPTER_ARGS,
    WINDOWS_CHASSIS_ARGS,
    WINDOWS_MACHINE_GUID_ARGS,
    WINDOWS_REGISTRY_ARGS,
)

FIXTURES = Path(__file__).parent / "fixtures"
GPU_ARGS = {
    "smi": NVIDIA_SMI_ARGS,
    "cc": NVIDIA_CC_ARGS,
    "lspci": LSPCI_ARGS,
    "cim": WINDOWS_ADAPTER_ARGS,
    "registry": WINDOWS_REGISTRY_ARGS,
    "arm64": MACOS_ARM64_ARGS,
    "chip": MACOS_CHIP_ARGS,
    "profiler": SYSTEM_PROFILER_ARGS,
    # Not graphics sources, but read by every whole measurement (`measure_hardware`).
    "memsize": MACOS_MEMSIZE_ARGS,
    "uuid": MACOS_PLATFORM_UUID_ARGS,
    "chassis": WINDOWS_CHASSIS_ARGS,
    "guid": WINDOWS_MACHINE_GUID_ARGS,
}


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def done(args, stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr="")


def gpu_runner(**answers) -> FixtureRunner:
    """A runner for the graphics sources by short name (`GPU_ARGS`); a text is a successful answer,
    an exception is raised, and every source not given is not installed."""
    responses = {}
    for name, args in GPU_ARGS.items():
        answer = answers.get(name, FileNotFoundError(f"{args[0]}: not found on PATH"))
        responses[args] = done(args, answer) if isinstance(answer, str) else answer
    return FixtureRunner(responses)


def cim(*entries: tuple[str | None, str | None]) -> str:
    """`Win32_VideoController` as `ConvertTo-Json -Compress` gives it: one object alone, else a list."""
    items = [{"Name": name, "PNPDeviceID": pnp} for name, pnp in entries]
    return json.dumps(items[0] if len(items) == 1 else items)


def registry(*entries: tuple[str, object]) -> str:
    """The registry's display class as `ConvertTo-Json -Compress` gives it."""
    items = [{"DriverDesc": "card", "MatchingDeviceId": key, "HardwareInformation.qwMemorySize": size} for key, size in entries]
    return json.dumps(items[0] if len(items) == 1 else items)


def reference(*gpus: LlmfitGpu, vram_gib: float | None = None, status: str = "available", ram_gib: float = 64.0) -> LlmfitReference:
    """llmfit's reading with its card list; `vram_gib` is `system.gpu_vram_gb`."""
    if status != "available":
        return LlmfitReference(status, None, None, None, f"llmfit is {status}")
    vram = vram_gib if vram_gib is not None else (gpus[0].vram_gb if gpus else 0.0)
    cuda = round(sum(gpu.vram_gb * gpu.count for gpu in gpus if gpu.backend == "CUDA"), 2)
    return LlmfitReference("available", "1.1.16", ram_gib, vram, None, cuda_vram_gib=cuda, gpus=tuple(gpus))


def llmfit_file(name: str) -> tuple[LlmfitGpu, ...]:
    """The card list of an `llmfit system --json` fixture, read the way the package reads it."""
    from modelroom.llmfit import llmfit_gpus

    return llmfit_gpus(json.loads(fixture(name))["system"])
