"""AP9-GPU, Codex diff review r1: duplicates only where certain, a failed nvidia-smi is one failed
source, a valid `arm64 = 0`, diagnoses kept, own exclusions first, limits, the words of a unit."""

from __future__ import annotations

from pathlib import Path

from fixture_gpu import cim, fixture, gpu_runner, reference, registry
from fixture_support import v2_profile
from modelroom.gpu import NOT_LISTED_REASON, measure_gpu
from modelroom.gpu_llmfit import vram_crosscheck
from modelroom.gpu_usable import EXCLUDED_VENDOR_REASON, adapter_label, is_display_only
from modelroom.intro import graphics_words
from modelroom.llmfit import FixtureRunner, LlmfitGpu, llmfit_gpus
from modelroom.measure_base import FixtureFiles
from modelroom.profile import HardwareProfile
from modelroom.room import RoomProbes, free_graphics_mib
from modelroom.views_adapters import adapter_line

DRM = "/sys/class/drm"
KFD = "/sys/class/kfd/kfd/topology/nodes"
PNP_A = "PCI\\VEN_10DE&DEV_2204&SUBSYS_00000000&REV_A1\\X"
PNP_B = "PCI\\VEN_10DE&DEV_1B06&SUBSYS_00000000&REV_A1\\X"
AMD_LINE = "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c] (rev c8)\n"
NVIDIA_LINES = (
    "0000:01:00.0 3D controller [0302]: NVIDIA Corporation GA102 [GeForce RTX 3090] [10de:2204] (rev a1)\n"
    "0000:02:00.0 3D controller [0302]: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] [10de:1b06] (rev a1)\n"
)
AMD_SYSFS = {
    f"{DRM}/card1/device/uevent": "PCI_SLOT_NAME=0000:03:00.0\n",
    f"{DRM}/card1/device/mem_info_vram_total": "25753026560\n",
    f"{KFD}/1/properties": fixture("kfd_node_gpu_properties.txt"),
}


# --- 1: an NVIDIA entry nvidia-smi does not list stays ---------------------------------------------------


def test_a_second_nvidia_card_nvidia_smi_does_not_list_stays_visible_and_blocks_the_comparison():
    runner = gpu_runner(smi="0, NVIDIA GeForce RTX 3090, 24576\n", cim=cim(("NVIDIA GeForce RTX 3090", PNP_A), ("NVIDIA GeForce GTX 1080 Ti", PNP_B)))
    llmfit = reference(LlmfitGpu("NVIDIA GeForce RTX 3090", "CUDA", 24.0, 1))
    reading = measure_gpu("win32", runner, reference=llmfit)
    counted, second = reading.graphics
    assert (counted.source, counted.counted) == ("nvidia-smi", True)
    assert (second.name, second.counted, second.reason) == ("NVIDIA GeForce GTX 1080 Ti", False, NOT_LISTED_REASON)
    assert vram_crosscheck(reading, llmfit, False).status == "not_comparable"


def test_on_linux_the_pci_address_of_nvidia_smi_decides_which_line_is_the_same_card():
    files = FixtureFiles({})
    runner = gpu_runner(smi="0, NVIDIA GeForce RTX 3090, 24576\n", cc="0, 8.6, 00000000:01:00.0\n", lspci=NVIDIA_LINES)
    reading = measure_gpu("linux", runner, files, files.list_dir)
    assert [(adapter.pci_slot, adapter.counted) for adapter in reading.graphics] == [("0000:01:00.0", True), ("0000:02:00.0", False)]
    assert reading.graphics[1].reason == NOT_LISTED_REASON


TWO_L4 = "0, NVIDIA L4, 23034\n1, NVIDIA L4, 23034\n"


def test_the_recorded_l4_server_matches_both_cards_by_their_addresses():
    cc = "0, 8.9, 00000000:01:00.0\n1, 8.9, 00000000:02:00.0\n"
    reading = measure_gpu("linux", gpu_runner(smi=TWO_L4, cc=cc, lspci=fixture("lspci_nnD_two_l4_server.txt")))
    assert [adapter.source for adapter in reading.graphics] == ["nvidia-smi", "nvidia-smi"]


def test_as_many_entries_as_cards_at_other_addresses_are_other_cards():
    """Codex r2-1: the same count proves nothing; a line at an address nvidia-smi does not name stays."""
    lspci = "0000:03:00.0 3D controller [0302]: NVIDIA Corporation AD104GL [L4] [10de:27b8] (rev a1)\n"
    llmfit = reference(LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 1))
    reading = measure_gpu("linux", gpu_runner(smi="0, NVIDIA L4, 23034\n", cc="0, 8.9, 00000000:01:00.0\n", lspci=lspci), reference=llmfit)
    assert [(adapter.pci_slot, adapter.counted) for adapter in reading.graphics] == [("0000:01:00.0", True), ("0000:03:00.0", False)]
    assert vram_crosscheck(reading, llmfit, False).status == "not_comparable"


def test_without_addresses_the_same_count_removes_nothing_on_linux():
    reading = measure_gpu("linux", gpu_runner(smi=TWO_L4, lspci=fixture("lspci_nnD_two_l4_server.txt")))
    assert [adapter.source for adapter in reading.graphics] == ["nvidia-smi", "nvidia-smi", "lspci", "lspci"]


def test_on_windows_an_address_of_nvidia_smi_does_not_stop_the_name_match():
    """Codex r2-9: `Win32_VideoController` carries no address; the name decides there."""
    runner = gpu_runner(smi="0, NVIDIA GeForce RTX 3090, 24576\n", cc="0, 8.6, 00000000:01:00.0\n",
                        cim=cim(("NVIDIA GeForce RTX 3090", PNP_A), ("NVIDIA GeForce GTX 1080 Ti", PNP_B)))
    counted, second = measure_gpu("win32", runner).graphics
    assert (counted.source, second.name, second.reason) == ("nvidia-smi", "NVIDIA GeForce GTX 1080 Ti", NOT_LISTED_REASON)


def test_a_hyper_v_id_without_a_device_stays_a_display_adapter():
    """Codex r3-15: `PCI\\VEN_1414` alone and no name is still a display adapter, `none` and not unmeasured."""
    reading = measure_gpu("win32", gpu_runner(cim=cim((None, "PCI\\VEN_1414"))))
    assert (reading.gpu_state, reading.note) == ("none", "display adapter only")
    assert is_display_only(reading.graphics[0])


def test_a_unit_of_an_intel_mac_is_no_display_adapter():
    reading = measure_gpu("darwin", gpu_runner(profiler=fixture("system_profiler_intel_mac.json.txt")), ram_gib=32.0)
    assert reading.graphics and not any(is_display_only(adapter) for adapter in reading.graphics)


def test_a_basic_display_adapter_with_a_registry_conflict_stays_a_display_adapter():
    """Codex r2-10: the kept diagnosis does not turn a display adapter into a graphics unit."""
    pnp, key = "PCI\\VEN_1414&DEV_008E&SUBSYS_00000000&REV_00\\X", "PCI\\VEN_1414&DEV_008E"
    runner = gpu_runner(cim=cim(("Microsoft Basic Display Adapter", pnp), ("Microsoft Basic Display Adapter", pnp)),
                        registry=registry((key, 268435456), (key, None)))
    reading = measure_gpu("win32", runner)
    assert all(is_display_only(adapter) for adapter in reading.graphics)
    assert "the registry holds 2 entries" in reading.graphics[0].reason
    assert (reading.gpu_state, reading.note) == ("none", "display adapter only")


# --- 2: an answer of nvidia-smi that does not read is one failed source ---------------------------------------


def test_an_unreadable_nvidia_smi_does_not_stop_the_amd_card_from_counting():
    files = FixtureFiles(AMD_SYSFS)
    runner = gpu_runner(smi="0, NVIDIA Nova, lots\n", lspci=AMD_LINE)
    reading = measure_gpu("linux", runner, files, files.list_dir)
    assert (reading.gpu_state, reading.vram_source, reading.vram_gib) == ("measured", "sysfs", 23.98)
    assert "nvidia-smi output could not be read" in reading.note


def test_an_unreadable_nvidia_smi_does_not_stop_the_llmfit_fallback():
    llmfit = reference(LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 1))
    reading = measure_gpu("linux", gpu_runner(smi="1, NVIDIA L4, 23034\n", lspci=""), reference=llmfit)
    assert (reading.gpu_state, reading.vram_source) == ("measured", "llmfit")
    assert "not index 0" in reading.note


# --- 3: a valid arm64 of 0 is an Intel Mac ---------------------------------------------------------------------


def test_a_valid_arm64_of_0_is_an_intel_mac_whatever_llmfit_says():
    llmfit = reference(LlmfitGpu("Apple M3 Max", "Metal", 64.0, 1, unified_memory=True))
    reading = measure_gpu("darwin", gpu_runner(arm64="0\n"), reference=llmfit, ram_gib=64.0)
    assert (reading.gpu_state, reading.note) == ("none", "Ollama runs on the CPU on Intel Macs")


def test_an_unreadable_arm64_leaves_the_proof_to_llmfit():
    llmfit = reference(LlmfitGpu("Apple M3 Max", "Metal", 64.0, 1, unified_memory=True))
    assert measure_gpu("darwin", gpu_runner(arm64="yes\n"), reference=llmfit, ram_gib=64.0).gpu_state == "unified_memory"


# --- 4: an incomplete registry is unknown, and every diagnosis stays -------------------------------------------


def test_one_registry_value_for_two_cards_is_unknown_and_the_line_says_why():
    key = "PCI\\VEN_10DE&DEV_2204&SUBSYS_00000000"
    runner = gpu_runner(cim=cim(("NVIDIA GeForce RTX 3090", PNP_A), ("NVIDIA GeForce RTX 3090", PNP_A)), registry=registry((key, 25769803776), (key, None)))
    first, _second = measure_gpu("win32", runner).graphics
    assert first.memory_gib is None
    assert adapter_line(first) == (
        "graphics  NVIDIA GeForce RTX 3090 · memory not measured · windows · "
        "not counted: nvidia-smi did not answer; the registry holds 2 entries for 2 adapters"
    )


def test_a_sysfs_file_that_did_not_read_stays_beside_the_missing_kfd():
    files = FixtureFiles({f"{DRM}/card1/device/uevent": "PCI_SLOT_NAME=0000:03:00.0\n"})
    (amd,) = measure_gpu("linux", gpu_runner(lspci=AMD_LINE), files, files.list_dir).graphics
    assert amd.reason == "no ROCm compute driver for this card; Ollama may use it through Vulkan; mem_info_vram_total could not be read"


# --- 5: own exclusions first, every vendor --------------------------------------------------------------------


def test_an_own_amd_exclusion_on_windows_comes_before_the_windows_rule_for_llmfits_amd_card():
    runner = gpu_runner(cim=cim(("Microsoft Basic Display Adapter", "PCI\\VEN_1002&DEV_744C&SUBSYS_00000000&REV_C8\\X")))
    llmfit = reference(LlmfitGpu("AMD Radeon RX 7900 XTX", "ROCm", 23.98, 1))
    own, fallback = measure_gpu("win32", runner, reference=llmfit).graphics
    assert own.usable == "no"
    assert (fallback.usable, fallback.reason) == ("no", EXCLUDED_VENDOR_REASON)


def test_an_own_display_adapter_shuts_out_llmfits_units_of_an_unknown_vendor_too():
    llmfit = reference(LlmfitGpu("Matrox G200", "Vulkan", 0.5, 1))
    reading = measure_gpu("linux", gpu_runner(lspci=fixture("lspci_nnD_cpu_server.txt")), reference=llmfit)
    assert (reading.graphics[-1].usable, reading.graphics[-1].reason) == ("no", EXCLUDED_VENDOR_REASON)


# --- 6: limits ---------------------------------------------------------------------------------------------------


def test_an_llmfit_count_no_machine_has_makes_the_list_unread_not_a_crash():
    assert llmfit_gpus({"gpus": [{"backend": "CUDA", "vram_gb": 8.0, "count": 10**30}]}) == ()
    assert len(llmfit_gpus({"gpus": [{"backend": "CUDA", "vram_gb": 8.0, "count": 64}]})) == 1


def test_a_sysfs_value_beyond_64_bits_is_no_value_in_the_measurement_and_in_the_free_memory():
    files = FixtureFiles({**AMD_SYSFS, f"{DRM}/card1/device/mem_info_vram_total": "9" * 400 + "\n"})
    (amd,) = measure_gpu("linux", gpu_runner(lspci=AMD_LINE), files, files.list_dir).graphics
    assert amd.memory_gib is None and not amd.counted
    data = v2_profile().model_dump(mode="json")
    card = {"vendor": "amd", "name": "AMD", "pci_slot": "0000:03:00.0", "memory_gib": 24.0, "memory_kind": "dedicated", "form": "discrete",
            "source": "sysfs", "evidence": "kfd", "backend": "rocm", "usable": "yes", "counted": True, "reason": None}
    data.update(vram_gib=24.0, vram_source="sysfs", gpu_name="AMD", gpus=[{"index": 0, "name": "AMD", "vram_gib": 24.0, "vram_source": "sysfs"}], adapters=[card])
    free_files = FixtureFiles({f"{DRM}/card1/device/uevent": "PCI_SLOT_NAME=0000:03:00.0\n",
                               f"{DRM}/card1/device/mem_info_vram_total": "9" * 400, f"{DRM}/card1/device/mem_info_vram_used": "0"})
    room = RoomProbes(environ={}, home=Path("/srv/ada"), system="linux", runner=FixtureRunner({}), read_text=free_files, list_dir=free_files.list_dir)
    assert free_graphics_mib(room, HardwareProfile.model_validate(data)) is None


# --- 7: the words of a unit ----------------------------------------------------------------------------------------


def test_a_nameless_qemu_device_is_named_by_its_pci_vendor():
    reading = measure_gpu("linux", gpu_runner(lspci="0000:00:02.0 VGA compatible controller [0300]: Device [1234:1111] (rev 02)\n"))
    assert adapter_label(reading.graphics[0]) == "QEMU/Bochs display adapter"


def test_a_nameless_intel_unit_is_a_card_not_counted_not_no_card():
    reading = measure_gpu("win32", gpu_runner(cim=cim((None, "PCI\\VEN_8086&DEV_7D67&SUBSYS_00000000&REV_06\\X"))))
    data = v2_profile().model_dump(mode="json")
    data.update(gpu_state="none", vram_gib=0.0, vram_source="none", gpu_name=None, gpus=[], adapters=[a.model_dump() for a in reading.graphics])
    assert graphics_words(HardwareProfile.model_validate(data)) == "no counted graphics card (Intel graphics)"
