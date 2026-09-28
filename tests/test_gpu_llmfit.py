"""AP9-GPU G6: llmfit as the fallback (mode B) and never mixed with modelroom's own cards (D2, D3, Z1, Z2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fixture_gpu import cim, done, fixture, gpu_runner, llmfit_file, reference
from modelroom.gpu import MODE_B_NOTE, measure_gpu
from modelroom.gpu_usable import EXCLUDED_VENDOR_REASON, LLMFIT_CAP_REASON, VULKAN_REASON
from modelroom.llmfit import FixtureRunner, LlmfitGpu, llmfit_gpus, read_llmfit_reference
from modelroom.measure import build_profile, measure_hardware
from modelroom.measure_base import FixtureFiles
from modelroom.profile import fit_block_reason

NOW = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)
NVIDIA_PNP = "PCI\\VEN_10DE&DEV_2F38&SUBSYS_0D121028&REV_A1\\X"
INTEL_PNP = "PCI\\VEN_8086&DEV_7D67&SUBSYS_0D121028&REV_06\\X"
L4 = LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 1)
ARC = llmfit_file("llmfit_system_intel_arc.json")


def _linux(llmfit, **answers):
    return measure_gpu("linux", gpu_runner(**answers), reference=llmfit)


def test_mode_a_adds_no_llmfit_adapter_even_where_llmfit_knows_a_second_card():
    reading = _linux(reference(LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 2)), smi="0, NVIDIA L4, 23034\n")
    assert [adapter.source for adapter in reading.graphics] == ["nvidia-smi"]
    assert (reading.gpu_state, reading.vram_gib) == ("measured", 22.49)


def test_mode_b_counts_llmfit_alone():
    reading = _linux(reference(LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 2)), lspci=fixture("lspci_nnD_two_l4_server.txt"))
    counted = [adapter for adapter in reading.graphics if adapter.counted]
    assert [adapter.source for adapter in counted] == ["llmfit", "llmfit"]
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source) == ("multi_gpu", 44.98, "llmfit")
    assert MODE_B_NOTE in reading.note
    assert all(not adapter.counted for adapter in reading.graphics if adapter.source == "lspci")


def test_an_own_nvidia_card_shut_out_by_its_compute_capability_keeps_llmfits_cuda_card_out():
    """D3: the card modelroom found unusable does not come back through llmfit."""
    reading = _linux(reference(LlmfitGpu("NVIDIA Tesla K80", "CUDA", 11.17, 1)), smi="0, NVIDIA Tesla K80, 11441\n", cc="0, 3.7\n")
    own, fallback = reading.graphics
    assert (own.usable, own.counted) == ("no", False) and "3.7" in own.reason
    assert (fallback.source, fallback.usable, fallback.counted, fallback.reason) == ("llmfit", "no", False, EXCLUDED_VENDOR_REASON)
    assert reading.gpu_state == "none"


def test_the_same_shut_out_card_reported_through_vulkan_is_not_usable_either():
    """Z1: Vulkan and SYCL entries go through the exclusions first -- `no`, not "may be usable"."""
    reading = _linux(reference(LlmfitGpu("NVIDIA Tesla K80", "Vulkan", 11.17, 1)), smi="0, NVIDIA Tesla K80, 11441\n", cc="0, 3.5\n")
    fallback = reading.graphics[1]
    assert (fallback.backend, fallback.usable, fallback.reason) == ("vulkan", "no", EXCLUDED_VENDOR_REASON)


def test_mode_b_with_own_intel_graphics_counts_llmfits_nvidia_card_and_shows_the_intel_one():
    runner = gpu_runner(cim=cim(("Intel(R) Graphics", INTEL_PNP)))
    reading = measure_gpu("win32", runner, reference=reference(LlmfitGpu("NVIDIA RTX PRO 3000", "CUDA", 11.94, 1)))
    intel, nvidia = reading.graphics
    assert (intel.source, intel.counted, nvidia.source, nvidia.counted) == ("windows", False, "llmfit", True)
    assert reading.gpu_state == "measured"


@pytest.mark.parametrize(
    ("vram", "platform", "counted"),
    [(3.99, "win32", False), (4.0, "win32", False), (4.000001, "win32", True), (4.0, "linux", True)],
)
def test_llmfit_on_windows_at_or_below_4_gb_never_counts_by_the_raw_value(vram, platform, counted):
    reading = measure_gpu(platform, gpu_runner(), reference=reference(LlmfitGpu("NVIDIA GeForce GTX 1650", "CUDA", vram, 1)))
    (card,) = reading.graphics
    assert card.counted is counted
    assert card.reason == (None if counted else LLMFIT_CAP_REASON)


def test_the_count_of_an_entry_is_unfolded():
    reading = _linux(reference(LlmfitGpu("NVIDIA A100-SXM4-80GB", "CUDA", 79.25, 4)))
    assert len(reading.graphics) == 4 and len(reading.adapters) == 4 and reading.vram_gib == 317.0


def test_own_listing_failed_and_llmfit_knows_an_arc_card():
    """r3-1: a Vulkan card llmfit knows is shown with its name and the hint, and the machine computes on the CPU."""
    measured = measure_hardware("linux", gpu_runner(), FixtureFiles({"/proc/meminfo": "MemTotal: 33554432 kB\n"}), lambda: 0, reference=reference(*ARC))
    (arc,) = measured.gpu.graphics
    assert (arc.name, arc.source, arc.backend, arc.usable, arc.counted, arc.reason) == (
        "Intel Arc A770", "llmfit", "vulkan", "unconfirmed", False, VULKAN_REASON)
    profile = build_profile(measured, reference(*ARC), "0123456789abcdef", "box", NOW)
    assert profile.gpu_state == "none" and fit_block_reason(profile) is None


def test_a_sycl_entry_is_shown_the_same_way():
    reading = _linux(reference(LlmfitGpu("Intel(R) Arc(TM) A770 Graphics", "SYCL", 15.91, 1)))
    (card,) = reading.graphics
    assert (card.backend, card.counted, card.form, reading.gpu_state) == ("vulkan", False, "discrete", "none")


@pytest.mark.parametrize("backend", ["CPU (x86)", "CPU (ARM)", "NPU (Ascend)", "Metal"])
def test_cpu_npu_and_metal_entries_make_no_adapter(backend):
    reading = _linux(reference(LlmfitGpu("unit", backend, 8.0, 1)), lspci="")
    assert reading.graphics == () and reading.gpu_state == "none"


def test_a_card_list_that_does_not_read_is_empty_and_never_an_abort():
    assert llmfit_gpus({"gpus": [{"backend": "CUDA", "vram_gb": "a lot", "count": 1}]}) == ()
    assert llmfit_gpus({"gpus": "cards"}) == ()
    assert llmfit_gpus({"gpus": [{"backend": "CUDA", "vram_gb": 8.0, "count": 1, "unified_memory": "no"}]}) == ()


def test_the_card_list_needs_no_call_of_its_own():
    runner = FixtureRunner({
        ("llmfit", "--version"): done(("llmfit",), "llmfit 1.1.16\n"),
        ("llmfit", "system", "--json"): done(("llmfit",), fixture("llmfit_system_amd.json")),
    })
    read = read_llmfit_reference(runner, "1.1.16")
    assert read.gpus == (LlmfitGpu("AMD Radeon RX 7900 XTX", "ROCm", 23.98, 1),)
    assert runner.calls == [("llmfit", "--version"), ("llmfit", "system", "--json")]


def test_an_amd_card_llmfit_reads_with_rocm_counts_in_mode_b():
    reading = _linux(reference(*llmfit_file("llmfit_system_amd.json")))
    (card,) = reading.graphics
    assert (card.vendor, card.backend, card.counted, reading.vram_source) == ("amd", "rocm", True, "llmfit")


def test_amd_from_llmfit_on_windows_is_not_confirmed_either():
    """F4, every source: ROCm on Windows is not checked, llmfit's word included."""
    reading = measure_gpu("win32", gpu_runner(), reference=reference(*llmfit_file("llmfit_system_amd.json")))
    (card,) = reading.graphics
    assert (card.usable, card.counted, reading.gpu_state) == ("unconfirmed", False, "none")


def test_a_cuda_card_llmfit_reads_without_memory_is_shown_and_not_counted():
    """Z2: `vram_gb` 0 is unknown memory; the card stays uncounted with the reason, the CPU fit computes."""
    gpus = llmfit_file("llmfit_system_zero_memory.json")
    measured = measure_hardware("linux", gpu_runner(), FixtureFiles({"/proc/meminfo": "MemTotal: 33554432 kB\n"}), lambda: 0, reference=reference(*gpus))
    cuda, sycl = measured.gpu.graphics
    assert (cuda.memory_gib, cuda.counted, cuda.reason) == (None, False, "llmfit reads no memory for this card")
    assert (sycl.memory_gib, sycl.counted) == (None, False)
    profile = build_profile(measured, reference(*gpus), "0123456789abcdef", "box", NOW)
    assert profile.gpu_state == "none" and fit_block_reason(profile) is None
    assert json.loads(profile.model_dump_json())["adapters"][0]["memory_gib"] is None
