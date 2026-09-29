"""AP9-GPU G2: what one graphics unit is -- vendor, build, whether Ollama can use it (pure functions)."""

from __future__ import annotations

import pytest

from fixture_gpu import fixture
from modelroom.gpu_linux import kfd_slots
from modelroom.gpu_usable import (
    AMD_WINDOWS_REASON,
    DISPLAY_ONLY_NOTE,
    INTEGRATED_REASON,
    INTEL_MAC_REASON,
    OTHER_BACKEND_REASON,
    VULKAN_REASON,
    exclusion,
    form_of,
    graphics_adapter,
    is_display_adapter,
    llmfit_verdict,
    own_verdict,
    vendor_of_name,
    vendor_of_pci,
)
from modelroom.measure_base import FixtureFiles

KFD = "/sys/class/kfd/kfd/topology/nodes"


@pytest.mark.parametrize(
    ("vendor_id", "vendor"), [("10de", "nvidia"), ("1002", "amd"), ("8086", "intel"), ("106b", "apple"), ("1234", "other"), (None, "other"), ("10DE", "nvidia")]
)
def test_the_pci_vendor_names_the_vendor(vendor_id, vendor):
    assert vendor_of_pci(vendor_id) == vendor


@pytest.mark.parametrize(
    ("name", "vendor"),
    [("NVIDIA L4", "nvidia"), ("AMD Radeon RX 7900 XTX", "amd"), ("Radeon 780M", "amd"), ("Intel Arc A770", "intel"), ("Apple M3", "apple"), ("GeForce", "other"), (None, "other")],
)
def test_an_llmfit_name_names_the_vendor_by_its_first_word(name, vendor):
    assert vendor_of_name(name) == vendor


@pytest.mark.parametrize(
    ("vendor", "name", "platform", "slot", "form"),
    [
        ("nvidia", None, "linux", None, "discrete"),
        ("amd", "AMD Radeon RX 7900 XTX", "win32", None, "discrete"),
        ("amd", "AMD Radeon PRO W7900", "linux", None, "discrete"),
        ("amd", "AMD Radeon W6800", "linux", None, "discrete"),
        ("amd", "Aldebaran/MI200 [Instinct MI250X/MI250]", "linux", None, "discrete"),
        ("amd", "AMD Radeon VII", "linux", None, "discrete"),
        ("amd", "Rembrandt [Radeon 680M]", "linux", None, "integrated"),
        ("amd", "AMD Radeon(TM) Graphics", "win32", None, "integrated"),
        ("amd", "AMD Radeon Graphics", "linux", None, "integrated"),
        ("amd", "Advanced Micro Devices, Inc. [AMD/ATI] Raphael", "linux", None, "unknown"),
        ("intel", "Intel(R) Arc(TM) A770 Graphics", "win32", None, "discrete"),
        ("intel", "Intel(R) Arc Pro B60 Graphics", "win32", None, "discrete"),
        ("intel", "Intel(R) UHD Graphics 770", "win32", None, "integrated"),
        ("intel", "Intel(R) Iris(R) Xe Graphics", "win32", None, "integrated"),
        ("intel", "Intel(R) HD Graphics 620", "win32", None, "integrated"),
        ("intel", "Intel(R) Graphics", "win32", None, "integrated"),
        ("intel", "Intel(R) Arc(TM) Graphics", "win32", None, "integrated"),
        ("intel", "Intel(R) Arc(TM) Pro Graphics", "win32", None, "unknown"),
        ("intel", None, "win32", None, "unknown"),
        ("intel", "Intel Corporation Device", "linux", "0000:00:02.0", "integrated"),
        ("intel", "Intel Corporation DG2 [Arc A770]", "linux", "0000:03:00.0", "unknown"),
        ("apple", "Apple M3 Max", "darwin", None, "unified"),
        ("other", "Matrox G200eR2", "linux", None, "unknown"),
    ],
)
def test_the_build_comes_from_positive_marks_only(vendor, name, platform, slot, form):
    assert form_of(vendor, name, platform, slot=slot) == form


def test_integrated_graphics_with_much_memory_is_still_integrated():
    """`Radeon Graphics` with 96 GB: the size is no mark of a card of its own (D6)."""
    adapter = graphics_adapter("amd", own_verdict("amd", "linux", kfd=True), form_of("amd", "AMD Radeon Graphics", "linux"), "sysfs", name="AMD Radeon Graphics", memory_gib=96.0)
    assert (adapter.form, adapter.counted, adapter.reason, adapter.memory_gib) == ("integrated", False, INTEGRATED_REASON, None)


@pytest.mark.parametrize(
    ("vendor_id", "name", "pnp", "wsl", "display"),
    [
        ("1234", None, None, False, True),
        ("1414", None, None, False, True),
        ("1414", None, None, True, False),
        (None, "Microsoft Basic Display Adapter", "ROOT\\BasicDisplay\\0000", False, True),
        (None, None, "PCI\\VEN_1414&DEV_5353\\X", False, True),
        ("10de", "NVIDIA L4", None, False, False),
    ],
)
def test_a_display_adapter_is_known_with_or_without_a_name(vendor_id, name, pnp, wsl, display):
    assert is_display_adapter(vendor_id, name, pnp, wsl=wsl) is display


def test_exclusions_come_before_every_proof():
    assert exclusion("nvidia", "linux", display=True).reason == DISPLAY_ONLY_NOTE
    assert exclusion("intel", "darwin", intel_mac=True).reason == INTEL_MAC_REASON
    assert exclusion("nvidia", "linux", compute_capability=3.5).usable == "no"
    assert exclusion("nvidia", "linux", compute_capability=5.0) is None
    assert exclusion("nvidia", "linux", compute_capability=None) is None
    assert exclusion("amd", "win32") == exclusion("amd", "win32")
    assert (exclusion("amd", "win32").usable, exclusion("amd", "win32").reason) == ("unconfirmed", AMD_WINDOWS_REASON)
    assert exclusion("amd", "linux") is None


def test_the_proofs_of_an_own_reading():
    assert (own_verdict("nvidia", "linux", nvidia_listed=True).usable, own_verdict("nvidia", "linux").usable) == ("yes", "unconfirmed")
    assert (own_verdict("amd", "linux", kfd=True).backend, own_verdict("amd", "linux").backend) == ("rocm", "vulkan")
    assert own_verdict("intel", "linux").reason == VULKAN_REASON
    assert own_verdict("apple", "darwin").backend == "metal"
    assert own_verdict("other", "linux").usable == "unconfirmed"


@pytest.mark.parametrize(("backend", "usable"), [("CUDA", "yes"), ("ROCm", "yes"), ("Vulkan", "unconfirmed"), ("SYCL", "unconfirmed")])
def test_llmfit_proves_cuda_and_rocm_only(backend, usable):
    assert llmfit_verdict(backend).usable == usable


def test_a_usable_card_of_its_own_waits_for_the_count():
    adapter = graphics_adapter("nvidia", own_verdict("nvidia", "linux", nvidia_listed=True), "discrete", "nvidia-smi", memory_gib=24.0)
    assert (adapter.usable, adapter.counted, adapter.reason) == ("yes", False, OTHER_BACKEND_REASON)


def _kfd(*nodes: str) -> FixtureFiles:
    return FixtureFiles({f"{KFD}/{number}/properties": fixture(name) for number, name in enumerate(nodes)})


def test_a_kfd_node_names_the_address_of_its_card():
    """`location_id` 768 = bus 3, device 0, function 0 (`pci_dev_id`, kfd_topology.c v6.16)."""
    files = _kfd("kfd_node_cpu_properties.txt", "kfd_node_gpu_properties.txt")
    assert kfd_slots(files, files.list_dir) == {"0000:03:00.0"}


def test_only_the_global_kfd_folder_is_no_proof():
    files = FixtureFiles({"/sys/class/kfd/kfd/topology/generation_id": "1\n"})
    assert kfd_slots(files, files.list_dir) == set()


def test_a_kfd_node_of_another_card_proves_nothing_for_this_one():
    files = _kfd("kfd_node_gpu_properties.txt")
    assert "0000:0c:00.0" not in kfd_slots(files, files.list_dir)
