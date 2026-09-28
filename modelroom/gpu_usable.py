"""What one graphics unit is: its vendor, its build, whether Ollama has a native backend for it
(AP9-GPU G2, ADR 0011). Pure functions, no source is read here.

The yardstick is Ollama's own documentation of `OLLAMA_GPU_DOC_DATE` (`docs.ollama.com/gpu`):
NVIDIA from compute capability 5.0 through CUDA, AMD on Linux through ROCm, Apple Silicon through
Metal, Vulkan on Windows and Linux on by default. vLLM stays NVIDIA as before. Exclusions come
before any proof (D3), and in doubt a card is not counted: a build is only ever read from a
positive mark (D6), never from something missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .profile import GraphicsAdapter

OLLAMA_GPU_DOC_DATE = "2026-09-28"

Vendor = Literal["nvidia", "amd", "intel", "apple", "other"]
Form = Literal["discrete", "integrated", "unified", "unknown"]

PCI_VENDORS: dict[str, Vendor] = {"10de": "nvidia", "1002": "amd", "8086": "intel", "106b": "apple"}
# PCI vendors whose class-03 devices are pure display or virtual adapters: they never run a model.
# A positive list on purpose -- an unknown vendor is never silently declared harmless.
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
HYPER_V_VENDOR = "1414"
BASIC_DISPLAY_NAME = "Microsoft Basic Display Adapter"
# The least compute capability Ollama runs on (docs.ollama.com/gpu, OLLAMA_GPU_DOC_DATE).
MIN_COMPUTE_CAPABILITY = 5.0

# Every reason a card is not usable or not counted, one constant each.
DISPLAY_ONLY_NOTE = "display adapter only"
INTEL_MAC_REASON = "Ollama runs on the CPU on Intel Macs"
OLD_CUDA_REASON = "compute capability {cc} is below 5.0, the least Ollama supports"
AMD_WINDOWS_REASON = "ROCm on Windows needs a HIP7 driver; not checked"
NO_NVIDIA_SMI_REASON = "nvidia-smi did not answer"
NVIDIA_TOOLS_HINT = "install the NVIDIA driver tools (nvidia-smi)"
NO_KFD_REASON = "no ROCm compute driver for this card; Ollama may use it through Vulkan"
VULKAN_REASON = "Ollama may use it through Vulkan"
OTHER_VENDOR_REASON = "no backend of Ollama is known for this vendor"
WSL_STANDIN_REASON = "WSL shows the Windows card only through its driver"
INTEGRATED_REASON = "integrated graphics shares the system memory"
FORM_UNKNOWN_REASON = "whether it is a card of its own or integrated graphics could not be read"
NO_MEMORY_REASON = "its memory was not measured"
SHOWN_ONLY_REASON = "memory read from {source} is shown, never counted"
OTHER_BACKEND_REASON = "another driver than the counted cards"
EXCLUDED_VENDOR_REASON = "a card of this vendor is not usable by the reading above"
LLMFIT_CAP_REASON = "llmfit reads 4.0 GB or less on Windows, the value its source caps"
NO_SYSFS_REASON = "no unique sysfs entry for this card"

_AMD_DISCRETE = ("RX ", "PRO ", "Radeon W", "Instinct", "Radeon VII")
_AMD_INTEGRATED_RE = re.compile(r"Radeon \d{3}M")
_AMD_INTEGRATED_NAMES = ("Radeon Graphics", "Radeon(TM) Graphics")
_INTEL_DISCRETE_RE = re.compile(r"Arc\(TM\) [AB][0-9]{3}|Arc Pro [AB][0-9]{2}")
_INTEL_INTEGRATED_PARTS = ("UHD Graphics", "Iris", "HD Graphics")
INTEL_IGPU_SLOT = "0000:00:02.0"


VENDOR_WORDS: dict[str, str] = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple", "other": "unknown vendor"}


def adapter_label(adapter: GraphicsAdapter) -> str:
    """What a reader calls a card: its name, else its vendor -- never `unknown` where a source knows something."""
    return adapter.name or f"{VENDOR_WORDS[adapter.vendor]} graphics"


def is_display_only(adapter: GraphicsAdapter) -> bool:
    """A display or virtual adapter (D13): shut out with exactly that reason."""
    return adapter.usable == "no" and adapter.reason == DISPLAY_ONLY_NOTE


def recognized(adapter: GraphicsAdapter) -> bool:
    """No display adapter, and of a known vendor or with a name (Begriffe, ADR 0011)."""
    return not is_display_only(adapter) and (adapter.vendor != "other" or adapter.name is not None)


def vendor_of_pci(vendor_id: str | None) -> Vendor:
    """The vendor of a PCI vendor id (`10de`), `other` for every id this does not name."""
    return PCI_VENDORS.get((vendor_id or "").lower(), "other")


def vendor_of_name(name: str | None) -> Vendor:
    """The vendor of a name as llmfit gives it, from its first word."""
    text = (name or "").strip()
    if text.startswith("NVIDIA"):
        return "nvidia"
    if text.startswith(("AMD", "Radeon")):
        return "amd"
    if text.startswith("Intel"):
        return "intel"
    if text.startswith("Apple"):
        return "apple"
    return "other"


def is_display_adapter(vendor_id: str | None, name: str | None, pnp_device_id: str | None = None, *, wsl: bool = False) -> bool:
    """A pure display or virtual adapter (D13), with or without a name; on WSL the Hyper-V id is
    the stand-in of the Windows card, no display adapter (F19)."""
    vendor = (vendor_id or "").lower()
    if wsl and vendor == HYPER_V_VENDOR:
        return False
    if vendor in DISPLAY_ONLY_VENDORS:
        return True
    return (name or "").strip() == BASIC_DISPLAY_NAME or "VEN_1414" in (pnp_device_id or "").upper()


def form_of(vendor: Vendor, name: str | None, platform: str, *, slot: str | None = None) -> Form:
    """The build of a card, from positive marks only (D6); everything else is `unknown`.

    Intel on Linux: the address of the built-in graphics (`0000:00:02.0`) is integrated; a
    memory of its own is not read (no kernel file for it could be shown, D12), so a card of its
    own stays `unknown` there. Intel elsewhere, and every name llmfit gives, by the name.
    """
    text = name or ""
    if vendor == "nvidia":
        return "discrete"
    if vendor == "apple":
        return "unified"
    if vendor == "amd":
        return _amd_form(text)
    if vendor == "intel":
        if platform == "linux" and slot is not None:
            return "integrated" if slot == INTEL_IGPU_SLOT else "unknown"
        return _intel_name_form(text)
    return "unknown"


def _amd_form(name: str) -> Form:
    if any(mark in name for mark in _AMD_DISCRETE):
        return "discrete"
    if _AMD_INTEGRATED_RE.search(name) or any(mark in name for mark in _AMD_INTEGRATED_NAMES):
        return "integrated"
    return "unknown"


def _intel_name_form(name: str) -> Form:
    if _INTEL_DISCRETE_RE.search(name):
        return "discrete"
    stripped = name.strip()
    if any(part in stripped for part in _INTEL_INTEGRATED_PARTS) or stripped == "Intel(R) Graphics" or stripped.endswith("Arc(TM) Graphics"):
        return "integrated"
    return "unknown"


@dataclass(frozen=True)
class Verdict:
    """Whether Ollama can use a card: `yes` with its backend, `no` or `unconfirmed` with the reason."""

    usable: Literal["yes", "no", "unconfirmed"]
    backend: Literal["cuda", "rocm", "metal", "vulkan", "none", "unknown"]
    reason: str | None = None
    evidence: Literal["nvidia-smi", "kfd", "sysctl", "llmfit"] | None = None


def exclusion(vendor: Vendor, platform: str, *, display: bool = False, intel_mac: bool = False, compute_capability: float | None = None) -> Verdict | None:
    """The rules that shut a card out, before any proof (D3); `None` where none applies."""
    if display:
        return Verdict("no", "none", DISPLAY_ONLY_NOTE)
    if intel_mac:
        return Verdict("no", "none", INTEL_MAC_REASON)
    if vendor == "nvidia" and compute_capability is not None and compute_capability < MIN_COMPUTE_CAPABILITY:
        return Verdict("no", "cuda", OLD_CUDA_REASON.format(cc=f"{compute_capability:g}"))
    if vendor == "amd" and platform == "win32":
        return Verdict("unconfirmed", "unknown", AMD_WINDOWS_REASON)
    return None


def own_verdict(vendor: Vendor, platform: str, *, nvidia_listed: bool = False, kfd: bool = False, wsl_standin: bool = False) -> Verdict:
    """The proof for a card modelroom read itself, once no exclusion applied."""
    if vendor == "nvidia":
        return Verdict("yes", "cuda", evidence="nvidia-smi") if nvidia_listed else Verdict("unconfirmed", "cuda", NO_NVIDIA_SMI_REASON)
    if vendor == "amd":
        return Verdict("yes", "rocm", evidence="kfd") if kfd else Verdict("unconfirmed", "vulkan", NO_KFD_REASON)
    if vendor == "intel":
        return Verdict("unconfirmed", "vulkan", VULKAN_REASON)
    if vendor == "apple":
        return Verdict("yes", "metal", evidence="sysctl")
    return Verdict("unconfirmed", "unknown", WSL_STANDIN_REASON if wsl_standin else OTHER_VENDOR_REASON)


def llmfit_verdict(backend: str) -> Verdict:
    """The proof an llmfit entry brings (mode B): `CUDA` and `ROCm` are one, Vulkan and SYCL are not (F3)."""
    if backend == "CUDA":
        return Verdict("yes", "cuda", evidence="llmfit")
    if backend == "ROCm":
        return Verdict("yes", "rocm", evidence="llmfit")
    return Verdict("unconfirmed", "vulkan", VULKAN_REASON)


def graphics_adapter(
    vendor: Vendor,
    verdict: Verdict,
    form: Form,
    source: str,
    *,
    name: str | None = None,
    memory_gib: float | None = None,
    memory_reason: str | None = None,
    pci_id: str | None = None,
    pci_slot: str | None = None,
) -> GraphicsAdapter:
    """One adapter, not counted yet, with the one reason that matters most (`not_counted_reason`)."""
    kind = "unified" if form == "unified" else "shared" if form == "integrated" else "dedicated" if memory_gib is not None else "unknown"
    return GraphicsAdapter(
        vendor=vendor,
        name=name,
        pci_id=pci_id,
        pci_slot=pci_slot,
        memory_gib=None if form == "integrated" else memory_gib,
        memory_kind=kind,
        form=form,
        source=source,
        evidence=verdict.evidence,
        backend=verdict.backend,
        usable=verdict.usable,
        counted=False,
        reason=not_counted_reason(verdict, form, source, memory_gib, memory_reason),
    )


def not_counted_reason(verdict: Verdict, form: Form, source: str, memory_gib: float | None, memory_reason: str | None) -> str | None:
    """Why a card is not counted, the weightiest first: an exclusion, the build, the proof, the memory.

    A usable card of its own with countable memory gets `OTHER_BACKEND_REASON` for now; the count
    (`gpu.count_adapters`) takes it away from every card it counts.
    """
    if verdict.usable == "no":
        return verdict.reason
    if form == "integrated":
        return INTEGRATED_REASON
    if form == "unified":
        return verdict.reason
    if verdict.usable != "yes":
        return verdict.reason
    if form == "unknown":
        return FORM_UNKNOWN_REASON
    if memory_gib is None:
        return memory_reason or NO_MEMORY_REASON
    if source not in ("nvidia-smi", "sysfs", "llmfit"):
        return SHOWN_ONLY_REASON.format(source=source)
    return OTHER_BACKEND_REASON
