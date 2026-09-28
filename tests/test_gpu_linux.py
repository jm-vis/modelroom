"""AP9-GPU G4: Linux and WSL -- `lspci -nnD`, the sysfs memory of an AMD card by its address, kfd."""

from __future__ import annotations

import pytest

from fixture_gpu import GPU_ARGS, fixture, gpu_runner
from modelroom.gpu import measure_gpu
from modelroom.gpu_linux import LSPCI_ARGS, WSL_OSRELEASE_FILE, _lspci_vendors, lspci_adapters
from modelroom.gpu_usable import NO_SYSFS_REASON
from modelroom.measure_base import FixtureFiles

DRM = "/sys/class/drm"
KFD = "/sys/class/kfd/kfd/topology/nodes"
TWO_AMD = fixture("lspci_nnD_two_amd.txt")


def _sysfs(cards: dict[str, str], totals: dict[str, str] | None = None, kfd: bool = True, **extra: str) -> FixtureFiles:
    """`card name -> PCI address` as `uevent` gives it, each with its `mem_info_vram_total`."""
    files = {f"{DRM}/{card}/device/uevent": f"DRIVER=amdgpu\nPCI_SLOT_NAME={slot}\n" for card, slot in cards.items()}
    files.update({f"{DRM}/{card}/device/mem_info_vram_total": total for card, total in (totals or {}).items()})
    files[f"{DRM}/card0-DP-1/status"] = "connected\n"  # a connector folder is no card
    if kfd:
        files[f"{KFD}/1/properties"] = fixture("kfd_node_gpu_properties.txt")
    files.update(extra)
    return FixtureFiles(files)


def _measure(files: FixtureFiles, lspci: str = TWO_AMD, **answers):
    return measure_gpu("linux", gpu_runner(lspci=lspci, **answers), files, files.list_dir)


def test_lspci_is_asked_with_the_domain():
    assert LSPCI_ARGS == ("lspci", "-nnD")


def test_every_display_line_carries_its_address_vendor_device_and_name():
    listed = lspci_adapters(TWO_AMD)
    assert [(entry.slot, entry.vendor_id, entry.device_id) for entry in listed] == [
        ("0000:03:00.0", "1002", "744c"), ("0000:0c:00.0", "1002", "744c"), ("0000:0f:00.0", "1002", "164e")]
    assert listed[0].name == "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]"


@pytest.mark.parametrize("name", ["lspci_nnD_two_a100_server.txt", "lspci_nnD_two_l4_server.txt", "lspci_nnD_cpu_server.txt"])
def test_the_recorded_servers_read_the_same_in_the_nnd_form(name):
    old = name.replace("nnD", "nn")
    assert _lspci_vendors(fixture(name)) == _lspci_vendors(fixture(old))
    assert all(entry.slot and entry.slot.startswith("0000:") for entry in lspci_adapters(fixture(name)))
    assert [entry.slot for entry in lspci_adapters(fixture(name))] == [entry.slot for entry in lspci_adapters(fixture(old))]


def test_a_nameless_qemu_device_has_no_name():
    (qemu,) = [entry for entry in lspci_adapters(fixture("lspci_nnD_two_a100_server.txt")) if entry.vendor_id == "1234"]
    assert qemu.name is None


def test_two_amd_cards_in_swapped_card_folders_each_get_their_own_value():
    """`card0` holds the second card and `card1` the first: only `PCI_SLOT_NAME` ties them (D8)."""
    files = _sysfs({"card0": "0000:0c:00.0", "card1": "0000:03:00.0"}, {"card0": "17163091968\n", "card1": "25753026560\n"})
    reading = _measure(files)
    first, second, igpu = reading.graphics
    assert (first.pci_slot, first.memory_gib, first.source, first.usable) == ("0000:03:00.0", 23.98, "sysfs", "yes")
    assert (second.pci_slot, second.memory_gib, second.usable) == ("0000:0c:00.0", 15.98, "unconfirmed")
    assert igpu.memory_gib is None and igpu.counted is False
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source) == ("measured", 23.98, "sysfs")


def test_an_amd_card_without_a_unique_sysfs_entry_is_not_counted():
    files = _sysfs({"card0": "0000:03:00.0", "card1": "0000:03:00.0"}, {"card0": "25753026560\n", "card1": "25753026560\n"})
    first = _measure(files).graphics[0]
    assert (first.memory_gib, first.counted, first.reason) == (None, False, NO_SYSFS_REASON)


def test_an_amd_card_without_any_sysfs_entry_is_not_counted():
    first = _measure(_sysfs({})).graphics[0]
    assert (first.memory_gib, first.counted, first.reason) == (None, False, NO_SYSFS_REASON)


@pytest.mark.parametrize("total", ["lots\n", "0\n", None], ids=["unreadable", "zero", "missing"])
def test_a_memory_file_that_does_not_read_is_no_value_and_never_an_abort(total):
    files = _sysfs({"card0": "0000:03:00.0"}, {"card0": total} if total else {})
    first = _measure(files).graphics[0]
    assert (first.memory_gib, first.counted) == (None, False)
    assert "mem_info_vram_total could not be read" == first.reason


def test_no_amd_card_asks_for_no_sysfs_at_all():
    files = FixtureFiles({})
    measure_gpu("linux", gpu_runner(lspci=fixture("lspci_nnD_nvidia_laptop.txt")), files, files.list_dir)
    assert files.reads == []


def test_amd_on_linux_never_asks_amd_smi():
    runner = gpu_runner(lspci=TWO_AMD)
    measure_gpu("linux", runner, _sysfs({}), _sysfs({}).list_dir)
    assert all(call[0] != "amd-smi" and call[0] != "rocm-smi" for call in runner.calls)
    assert set(runner.calls) <= set(GPU_ARGS.values())


def test_without_a_kfd_node_an_amd_card_may_use_vulkan_and_is_not_counted():
    files = _sysfs({"card0": "0000:03:00.0"}, {"card0": "25753026560\n"}, kfd=False)
    first = _measure(files).graphics[0]
    assert (first.usable, first.backend, first.counted) == ("unconfirmed", "vulkan", False)
    assert "Ollama may use it through Vulkan" in first.reason


def test_intel_memory_is_not_read_on_linux():
    """D12: neither the xe tile file nor an i915 `lmem` file is in a tagged kernel, so none is read."""
    files = FixtureFiles({})
    reading = measure_gpu("linux", gpu_runner(lspci=fixture("lspci_nnD_intel_laptop.txt")), files, files.list_dir)
    (intel,) = reading.graphics
    assert (intel.form, intel.memory_gib, reading.gpu_state) == ("integrated", None, "none")
    assert files.reads == []


@pytest.mark.parametrize("release", ["5.15-microsoft-standard-WSL2\n", "6.6-Microsoft-standard-WSL2\n"])
def test_the_wsl_stand_in_is_no_display_adapter(release):
    files = FixtureFiles({WSL_OSRELEASE_FILE: release})
    reading = measure_gpu("linux", gpu_runner(lspci=fixture("lspci_nnD_wsl.txt")), files, files.list_dir)
    assert reading.gpu_state == "present_unmeasured"
    assert reading.note == "WSL shows the Windows card only through its driver; nvidia-smi did not answer"


def test_outside_wsl_the_hyper_v_device_is_a_display_adapter():
    files = FixtureFiles({WSL_OSRELEASE_FILE: "6.8.0-45-generic\n"})
    reading = measure_gpu("linux", gpu_runner(lspci=fixture("lspci_nnD_wsl.txt")), files, files.list_dir)
    assert (reading.gpu_state, reading.note) == ("none", "display adapter only")


def test_wsl_with_nvidia_smi_counts_the_card():
    files = FixtureFiles({WSL_OSRELEASE_FILE: "5.15-microsoft-standard-WSL2\n"})
    reading = measure_gpu("linux", gpu_runner(smi="0, NVIDIA L4, 23034\n", lspci=fixture("lspci_nnD_wsl.txt")), files, files.list_dir)
    assert reading.gpu_state == "measured" and reading.vram_source == "nvidia-smi"


def test_the_recorded_l4_server_without_nvidia_smi_is_never_present_unmeasured():
    files = FixtureFiles({})
    reading = measure_gpu("linux", gpu_runner(lspci=fixture("lspci_nnD_two_l4_server.txt")), files, files.list_dir)
    assert reading.gpu_state == "none"
    assert [adapter.usable for adapter in reading.graphics] == ["unconfirmed", "unconfirmed"]
