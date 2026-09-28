"""AP9-GPU G7 and G6: the state table row by row, the backend with the largest sum, and the cross-check table."""

from __future__ import annotations

import pytest

from fixture_gpu import cim, fixture, gpu_runner, reference
from modelroom.gpu import measure_gpu
from modelroom.gpu_llmfit import vram_crosscheck
from modelroom.gpu_usable import OTHER_BACKEND_REASON
from modelroom.gpu_linux import WSL_OSRELEASE_FILE
from modelroom.llmfit import LlmfitGpu
from modelroom.measure_base import FixtureFiles

DRM = "/sys/class/drm"
KFD = "/sys/class/kfd/kfd/topology/nodes"
NVIDIA_LINE = "0000:01:00.0 3D controller [0302]: NVIDIA Corporation AD102GL [RTX 6000 Ada] [10de:26b1] (rev a1)\n"
AMD_LINE = "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c] (rev c8)\n"
QEMU_LINE = "0000:00:02.0 VGA compatible controller [0300]: Device [1234:1111] (rev 02)\n"
UNKNOWN_LINE = "0000:00:05.0 Display controller [0380]: Device [1af5:1050] (rev 01)\n"
WSL = FixtureFiles({WSL_OSRELEASE_FILE: "5.15-microsoft-standard-WSL2\n"})
CARD = LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 1)


def _linux(lspci: str | Exception | None = None, files: FixtureFiles | None = None, llmfit=None, **answers):
    files = files or FixtureFiles({})
    if lspci is not None:
        answers["lspci"] = lspci
    return measure_gpu("linux", gpu_runner(**answers), files, files.list_dir, llmfit)


def _amd_sysfs(total: str = "25753026560\n") -> FixtureFiles:
    return FixtureFiles({
        f"{DRM}/card1/device/uevent": "PCI_SLOT_NAME=0000:03:00.0\n",
        f"{DRM}/card1/device/mem_info_vram_total": total,
        f"{KFD}/1/properties": fixture("kfd_node_gpu_properties.txt"),
    })


# --- the state table -----------------------------------------------------------------------------------


def test_row_1_an_unmeasured_platform():
    assert measure_gpu("aix", gpu_runner()).gpu_state == "unsupported_platform"


def test_row_3_one_counted_card_is_measured_and_two_are_multi_gpu():
    one = _linux(smi="0, NVIDIA L4, 23034\n")
    two = _linux(smi="0, NVIDIA L4, 23034\n1, NVIDIA L4, 23034\n")
    assert (one.gpu_state, two.gpu_state, two.vram_gib) == ("measured", "multi_gpu", 44.98)


def test_row_4_apple_silicon_is_unified_memory():
    assert measure_gpu("darwin", gpu_runner(arm64="1\n"), ram_gib=32.0).gpu_state == "unified_memory"


@pytest.mark.parametrize("lspci", ["0000:00:00.0 3D controller [0302]: Microsoft Corporation Basic Render Driver [1414:008e]\n",
                                   "0000:00:00.0 3D controller [0302]: Device [1414:008e]\n"], ids=["named", "nameless"])
def test_row_5_the_wsl_stand_in_alone_is_present_unmeasured(lspci):
    reading = _linux(lspci, WSL, llmfit=reference())
    assert reading.gpu_state == "present_unmeasured" and reading.note.startswith("WSL shows the Windows card only")


def test_row_6_a_recognized_card_none_counted_is_none_with_a_note_per_card():
    reading = _linux(fixture("lspci_nnD_nvidia_laptop.txt"))
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source) == ("none", 0.0, "none")
    assert reading.note.startswith("no counted graphics card: ") and reading.note.count(": ") >= 3


def test_row_7_a_nameless_qemu_device_alone_is_a_display_adapter():
    reading = _linux(QEMU_LINE)
    assert (reading.gpu_state, reading.note) == ("none", "display adapter only")


def test_row_7_the_basic_display_adapter_of_windows_is_a_display_adapter():
    reading = measure_gpu("win32", gpu_runner(cim=cim(("Microsoft Basic Display Adapter", "ROOT\\BasicDisplay\\0000"))))
    assert (reading.gpu_state, reading.note) == ("none", "display adapter only")


def test_row_8_qemu_beside_an_unknown_device_is_present_unmeasured():
    reading = _linux(QEMU_LINE + UNKNOWN_LINE)
    assert (reading.gpu_state, reading.note) == ("present_unmeasured", "a graphics adapter is present but nothing about it could be read")


def test_row_9_nothing_listed_and_no_llmfit_card_is_not_recognized_with_every_source():
    reading = _linux(llmfit=reference(status="absent"))
    assert reading.gpu_state == "present_unmeasured"
    assert reading.note.startswith("not recognized: nvidia-smi is not installed")
    assert "lspci is not installed" in reading.note and "llmfit is absent" in reading.note


def test_row_10_no_display_device_is_none():
    reading = _linux(fixture("lspci_nnD_headless.txt"))
    assert (reading.gpu_state, reading.note, reading.graphics) == ("none", None, ())


# --- the backend with the largest sum (F8, D7) --------------------------------------------------------------


def test_the_backend_with_the_larger_counted_sum_counts_after_the_filter():
    """CUDA 40 GiB against ROCm 24 GiB counted plus 24 GiB only in the registry or llmfit: CUDA."""
    reading = _linux(NVIDIA_LINE + AMD_LINE, _amd_sysfs(), reference(LlmfitGpu("AMD Radeon RX 7900 XTX", "ROCm", 23.98, 1)), smi="0, NVIDIA A40, 40960\n")
    nvidia, amd = reading.graphics
    assert (nvidia.counted, amd.counted, amd.reason) == (True, False, OTHER_BACKEND_REASON)
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source) == ("measured", 40.0, "nvidia-smi")


def test_rocm_counts_where_its_sum_is_larger():
    reading = _linux(NVIDIA_LINE + AMD_LINE, _amd_sysfs(), smi="0, NVIDIA T400, 2048\n")
    assert (reading.vram_source, reading.vram_gib) == ("sysfs", 23.98)


def test_a_tie_goes_to_cuda():
    reading = _linux(NVIDIA_LINE + AMD_LINE, _amd_sysfs("25769803776\n"), smi="0, NVIDIA A10, 24576\n")
    assert reading.vram_source == "nvidia-smi"


# --- the cross-check table (G6) ------------------------------------------------------------------------------


def test_crosscheck_row_2_no_own_memory_is_absent_or_error():
    reading = _linux(llmfit=reference(status="absent"))
    assert vram_crosscheck(reading, reference(status="absent"), False).status == "absent"
    assert vram_crosscheck(reading, reference(status="error"), False).status == "error"


def test_crosscheck_row_3_llmfits_own_count_is_not_comparable():
    llmfit = reference(CARD)
    reading = _linux(llmfit=llmfit)
    check = vram_crosscheck(reading, llmfit, False)
    assert (check.status, check.own_gib, check.llmfit_gib) == ("not_comparable", 22.49, 22.49)


def test_crosscheck_row_4_cards_of_different_sizes_stay_not_comparable_even_when_llmfit_failed():
    reading = _linux(smi="0, NVIDIA A6000, 49152\n1, NVIDIA A4000, 16384\n")
    assert vram_crosscheck(reading, reference(status="error"), False).status == "not_comparable"


@pytest.mark.parametrize("status", ["absent", "error"])
def test_crosscheck_row_4a_one_card_with_llmfit_absent_or_failed(status):
    reading = _linux(smi="0, NVIDIA L4, 23034\n")
    assert vram_crosscheck(reading, reference(status=status), False).status == status


def test_crosscheck_row_5_the_same_set_of_cards_is_compared():
    reading = _linux(smi="0, NVIDIA L4, 23034\n1, NVIDIA L4, 23034\n")
    check = vram_crosscheck(reading, reference(LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 2)), False)
    assert (check.status, check.llmfit_gib) == ("confirmed", 44.98)


def test_crosscheck_row_5_an_intel_card_beside_is_no_obstacle():
    reading = measure_gpu("win32", gpu_runner(smi="0, NVIDIA L4, 23034\n", cim=fixture("win_video_controllers_laptop.json.txt")))
    llmfit = reference(CARD, LlmfitGpu("Intel(R) Graphics", "SYCL", 2.0, 1))
    assert vram_crosscheck(reading, llmfit, False).status == "confirmed"


def test_crosscheck_row_6_two_same_named_cards_one_counted_and_llmfit_count_2_is_not_comparable():
    reading = _linux(smi="0, NVIDIA L4, 23034\n", cc="0, 8.9\n")
    llmfit = reference(LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 2), vram_gib=22.49)
    check = vram_crosscheck(reading, llmfit, False)
    assert (check.status, check.llmfit_gib) == ("not_comparable", 22.49)


def test_crosscheck_row_6_an_uncounted_own_card_of_the_same_vendor_is_not_comparable():
    reading = _linux(smi="0, NVIDIA L4, 23034\n1, NVIDIA Tesla K80, 11441\n", cc="0, 8.9\n1, 3.7\n")
    assert reading.gpu_state == "measured"
    assert vram_crosscheck(reading, reference(CARD), False).status == "not_comparable"


def test_crosscheck_row_7_unified_memory_against_llmfits_value():
    reading = measure_gpu("darwin", gpu_runner(arm64="1\n"), ram_gib=32.0)
    assert vram_crosscheck(reading, reference(vram_gib=31.0), False).status == "confirmed"


def test_crosscheck_row_8_none_with_adapters_is_not_comparable_with_own_zero():
    reading = _linux(fixture("lspci_nnD_intel_laptop.txt"))
    check = vram_crosscheck(reading, reference(vram_gib=2.0), False)
    assert (check.status, check.own_gib) == ("not_comparable", 0.0)


def test_crosscheck_row_9_none_without_adapters_compares_zero():
    reading = _linux(fixture("lspci_nnD_headless.txt"))
    assert vram_crosscheck(reading, reference(vram_gib=0.0), False).status == "confirmed"


def test_crosscheck_row_1_cpu_only_is_absent():
    reading = _linux(smi="0, NVIDIA L4, 23034\n")
    assert vram_crosscheck(reading, reference(CARD), True).status == "absent"
