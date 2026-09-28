"""The GPU verdict of this machine: `nvidia-smi` first, then the adapter rule (moved out of
`measure.py`, AP9-GPU G0)."""

from __future__ import annotations

from .gpu_linux import LSPCI_ARGS, _lspci_vendors
from .gpu_nvidia import _nvidia_smi_gpu
from .gpu_windows import WINDOWS_ADAPTER_ARGS, _pnp_vendors
from .llmfit import Runner
from .measure_base import GpuReading, _run

# The platforms whose GPU this measures at all; everywhere else is `unsupported_platform`.
MEASURED_PLATFORMS: frozenset[str] = frozenset({"win32", "linux"})

# PCI vendors whose class-03 devices are pure display or virtual adapters: they never run a
# model, so a machine that has only these is a CPU machine, not a machine with an unmeasured
# GPU. A positive list on purpose -- an unknown vendor is never silently declared harmless.
DISPLAY_ONLY_VENDORS = {
    "1234": "QEMU/Bochs",
    "1af4": "virtio",
    "15ad": "VMware",
    "1b36": "Red Hat",
    "80ee": "VirtualBox",
    "1414": "Microsoft Hyper-V",
    "5853": "Xen",
    "1d0f": "Amazon",
}
INTEL_VENDOR = "8086"
INTEL_NOTE = "Ollama uses the CPU on Intel graphics; run `hardware --cpu-only` for the CPU fit"
DISPLAY_ONLY_NOTE = "display adapter only"


def measure_gpu(platform: str, runner: Runner) -> GpuReading:
    """The GPU state and the measured VRAM of every NVIDIA adapter: one is `measured`, two or more
    `multi_gpu` with their sum (decided 2026-09-26).

    `nvidia-smi` first; when it is not there or cannot answer, the adapter rule decides between
    "no GPU at all" and "an adapter is present but unmeasured" -- a graphics adapter's own
    reported memory (`AdapterRAM`) is never read, it is not VRAM. macOS and everything else is
    `unsupported_platform`: no unified-memory statement is invented.
    """
    if platform not in MEASURED_PLATFORMS:
        return GpuReading("unsupported_platform", None, "unknown", None, f"the GPU is not measured on {platform} yet")
    measured = _nvidia_smi_gpu(runner)
    if measured is not None:
        return measured
    return _adapter_rule(platform, runner)


def _adapter_rule(platform: str, runner: Runner) -> GpuReading:
    """No nvidia-smi: decide from the PCI vendor ids of the class-03 (display) devices."""
    vendors, reason = _display_adapter_vendors(platform, runner)
    if vendors is None:
        note = f"the graphics adapters could not be listed, so a GPU cannot be ruled out ({reason})"
        return GpuReading("present_unmeasured", None, "unknown", None, note)
    if not vendors:
        return GpuReading("none", 0.0, "none")
    measurable = [vendor for vendor in vendors if vendor not in DISPLAY_ONLY_VENDORS]
    if not measurable:
        return GpuReading("none", 0.0, "none", None, DISPLAY_ONLY_NOTE)
    return GpuReading("present_unmeasured", None, "unknown", None, _unmeasured_adapter_note(measurable))


def _unmeasured_adapter_note(vendors: list[str]) -> str:
    """Name every adapter that could hold memory, not just the first one found.

    The Intel advice is only true for a machine whose *only* measurable adapter is Intel
    graphics: a laptop that also carries a discrete card would otherwise be told to compute a
    CPU fit although it has a GPU whose driver tools were simply not installed.
    """
    listed = ", ".join(dict.fromkeys(vendors))
    note = f"a graphics adapter is present (PCI vendor {listed}), but its memory was not measured"
    if set(vendors) == {INTEL_VENDOR}:
        return f"{note}; {INTEL_NOTE}"
    return f"{note}; run `hardware --cpu-only` for the CPU fit"


def _display_adapter_vendors(platform: str, runner: Runner) -> tuple[list[str] | None, str | None]:
    """The PCI vendor id of every display adapter, or `(None, reason)` when none could be listed."""
    if platform == "linux":
        stdout, reason = _run(runner, LSPCI_ARGS)
        return (None, reason) if stdout is None else (_lspci_vendors(stdout), None)
    stdout, reason = _run(runner, WINDOWS_ADAPTER_ARGS)
    return (None, reason) if stdout is None else (_pnp_vendors(stdout), None)
