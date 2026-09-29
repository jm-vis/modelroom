"""AP9-GPU G5: macOS -- Apple Silicon as unified memory from `sysctl`, an Intel Mac as a CPU machine."""

from __future__ import annotations

from datetime import datetime, timezone

from fixture_gpu import fixture, gpu_runner, llmfit_file, reference
from modelroom.gpu import measure_gpu
from modelroom.gpu_macos import APPLE_GRAPHICS, MACOS_RESERVE_NOTE
from modelroom.measure import build_crosscheck, build_profile, measure_hardware
from modelroom.measure_base import FixtureFiles
from modelroom.profile import fit_block_reason

NOW = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)
APPLE = llmfit_file("llmfit_system_apple.json")


def test_apple_silicon_by_sysctl_holds_the_ram_as_unified_memory():
    runner = gpu_runner(arm64=fixture("macos_sysctl_arm64.txt"), chip=fixture("macos_sysctl_brand_string.txt"))
    reading = measure_gpu("darwin", runner, ram_gib=128.0)
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source, reading.note) == ("unified_memory", 128.0, "sysctl", MACOS_RESERVE_NOTE)
    (apple,) = reading.graphics
    assert (apple.name, apple.memory_gib, apple.source, apple.evidence, apple.form, apple.reason) == (
        "Apple M3 Max", 128.0, "sysctl", "sysctl", "unified", None)


def test_a_failed_arm64_reading_with_llmfit_metal_is_proved_by_llmfit_and_still_holds_sysctl_ram():
    reading = measure_gpu("darwin", gpu_runner(chip="Apple M3 Max\n"), reference=reference(*APPLE), ram_gib=64.0)
    (apple,) = reading.graphics
    assert (apple.evidence, apple.source, apple.memory_gib) == ("llmfit", "sysctl", 64.0)


def test_a_failed_chip_name_falls_back_to_llmfit_then_to_apple_graphics():
    by_llmfit = measure_gpu("darwin", gpu_runner(arm64="1\n"), reference=reference(*APPLE), ram_gib=64.0)
    alone = measure_gpu("darwin", gpu_runner(arm64="1\n"), ram_gib=64.0)
    assert (by_llmfit.graphics[0].name, alone.graphics[0].name) == ("Apple M3 Max", APPLE_GRAPHICS)


def _mac(runner, llmfit):
    measured = measure_hardware("darwin", runner, FixtureFiles({}), lambda: 0, reference=llmfit)
    return measured, build_profile(measured, llmfit, "0123456789abcdef", "mac", NOW)


def test_an_unknown_ram_leaves_the_unified_memory_unknown_with_either_proof():
    """RAM never comes from llmfit (D9): `hw.memsize` failed, so VRAM is unknown, the check absent, the fit blocked."""
    for runner, llmfit in ((gpu_runner(arm64="1\n"), reference(status="absent")), (gpu_runner(), reference(*APPLE))):
        measured, profile = _mac(runner, llmfit)
        assert (profile.gpu_state, profile.vram_gib, profile.vram_source) == ("unified_memory", None, "unknown")
        assert build_crosscheck(measured, llmfit).vram.status == "absent"
        assert fit_block_reason(profile) == "physical RAM unknown"


def test_a_mac_with_its_ram_writes_a_unified_memory_profile_cross_checked_against_llmfit():
    llmfit = reference(*APPLE, vram_gib=128.0, ram_gib=128.0)
    measured, profile = _mac(gpu_runner(arm64="1\n", memsize="137438953472\n"), llmfit)
    assert (profile.gpu_state, profile.vram_gib, profile.vram_source, profile.gpus) == ("unified_memory", 128.0, "sysctl", [])
    assert profile.llmfit_crosscheck.vram.status == "confirmed" and fit_block_reason(profile) is None


def test_an_intel_mac_is_a_cpu_machine_and_lists_its_units_as_not_usable():
    reading = measure_gpu("darwin", gpu_runner(profiler=fixture("system_profiler_intel_mac.json.txt")), ram_gib=32.0)
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source) == ("none", 0.0, "none")
    assert [(adapter.vendor, adapter.usable, adapter.memory_gib, adapter.source) for adapter in reading.graphics] == [
        ("intel", "no", None, "system_profiler"), ("amd", "no", 8.0, "system_profiler")]
    assert all(adapter.reason == "Ollama runs on the CPU on Intel Macs" for adapter in reading.graphics)


def test_an_intel_mac_whose_units_cannot_be_listed_is_still_a_cpu_machine():
    reading = measure_gpu("darwin", gpu_runner(profiler="not json"), ram_gib=32.0)
    assert (reading.gpu_state, reading.graphics) == ("none", ())
