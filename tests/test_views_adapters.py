"""AP9-GPU G8: the adapters in words -- the terminal lines, the Markdown table, the card of step 5."""

from __future__ import annotations

from pathlib import Path

import pytest

from fixture_gpu import fixture, gpu_runner, reference
from fixture_support import v2_profile
from modelroom.gpu import measure_gpu
from modelroom.intro import hardware_words
from modelroom.llmfit import LlmfitGpu
from modelroom.profile import HardwareProfile
from modelroom.screen import short_hardware
from modelroom.views_adapters import ADAPTER_COLUMNS, adapter_line, adapter_lines, adapter_table_rows

LAPTOP = {"cim": fixture("win_video_controllers_laptop.json.txt"), "registry": fixture("win_registry_display_laptop.json.txt")}
SMI = "0, NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU, 12227\n"


def _lines(reading) -> list[str]:
    return adapter_lines(reading.graphics, reading.gpu_state, reading.note)


def test_the_laptop_reads_as_two_lines_word_for_word():
    assert _lines(measure_gpu("win32", gpu_runner(smi=SMI, **LAPTOP))) == [
        "graphics  NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU · 11.9 GB · nvidia-smi · usable (CUDA)",
        "graphics  Intel(R) Graphics · shared memory · windows · not counted: integrated graphics shares the system memory",
    ]


def test_an_amd_card_llmfit_counts_and_the_mode_b_hint_once():
    llmfit = reference(LlmfitGpu("AMD Radeon RX 7900 XTX", "ROCm", 24.0, 1), LlmfitGpu("AMD Radeon RX 7900 XTX", "Vulkan", 24.0, 1))
    lines = _lines(measure_gpu("linux", gpu_runner(), reference=llmfit))
    assert lines[0] == "graphics  AMD Radeon RX 7900 XTX · 24.0 GB · llmfit · usable (ROCm)"
    assert lines[1] == "graphics  AMD Radeon RX 7900 XTX · 24.0 GB · llmfit · may be usable (Vulkan), not counted: Ollama may use it through Vulkan"
    assert lines[2:] == ["graphics  rows from llmfit may describe a card listed above"]


def test_nothing_read_says_not_recognized_with_every_failed_source():
    (line,) = _lines(measure_gpu("linux", gpu_runner(), reference=reference(status="absent")))
    assert line == "graphics  not recognized: nvidia-smi is not installed (nvidia-smi: not found on PATH); lspci is not installed (lspci: not found on PATH); llmfit is absent"


def test_an_unusable_card_and_unified_memory():
    mac = measure_gpu("darwin", gpu_runner(arm64="1\n", chip="Apple M3 Max\n"), ram_gib=32.0)
    assert _lines(mac) == ["graphics  Apple M3 Max · unified memory 32.0 GB · sysctl · usable (Metal)"]
    k80 = measure_gpu("linux", gpu_runner(smi="0, NVIDIA Tesla K80, 11441\n", cc="0, 3.7\n"))
    assert adapter_line(k80.graphics[0]).endswith("· not usable: compute capability 3.7 is below 5.0, the least Ollama supports")


def test_a_card_without_memory_says_memory_not_measured_and_never_unknown():
    reading = measure_gpu("linux", gpu_runner(lspci=fixture("lspci_nnD_nvidia_laptop.txt")))
    line = adapter_line(reading.graphics[1])
    assert line == "graphics  NVIDIA Corporation Device · memory not measured · lspci · not counted: nvidia-smi did not answer"
    assert "unknown" not in " ".join(_lines(reading))


def _profile(reading) -> HardwareProfile:
    data = v2_profile(vram_gib=reading.vram_gib or 0.0).model_dump(mode="json")
    data.update(gpu_state=reading.gpu_state, vram_gib=reading.vram_gib, vram_source=reading.vram_source, gpu_name=reading.gpu_name,
                gpus=[card.model_dump() for card in reading.adapters], adapters=[a.model_dump() for a in reading.graphics])
    return HardwareProfile.model_validate(data)


def test_the_markdown_rows_of_a_machine():
    profile = _profile(measure_gpu("win32", gpu_runner(smi=SMI, **LAPTOP)))
    rows = adapter_table_rows("workstation", profile)
    assert len(ADAPTER_COLUMNS) == 11 == len(rows[0])
    assert rows[0][:5] == ["workstation", "1", "nvidia", "NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU", "11.94"]
    assert rows[1][5:] == ["shared memory", "integrated", "windows", "unconfirmed", "no", "integrated graphics shares the system memory"]
    assert adapter_table_rows("x", None) == []


def test_the_card_of_step_5_names_the_first_card_where_none_counts():
    profile = _profile(measure_gpu("win32", gpu_runner(**LAPTOP)))
    assert short_hardware(profile).startswith("no counted graphics card (Intel(R) Graphics)")
    assert hardware_words(profile).startswith("no counted graphics card (Intel(R) Graphics)")


def test_the_card_of_step_5_says_not_measured_only_for_present_unmeasured():
    profile = _profile(measure_gpu("linux", gpu_runner()))
    assert profile.gpu_state == "present_unmeasured" and "graphics memory not measured" in short_hardware(profile)


def test_the_card_of_step_5_of_a_mac():
    data = v2_profile(ram_gib=32.0).model_dump(mode="json")
    mac = measure_gpu("darwin", gpu_runner(arm64="1\n"), ram_gib=32.0)
    data.update(gpu_state="unified_memory", vram_gib=32.0, vram_source="sysctl", gpu_name=mac.gpu_name, gpus=[], adapters=[a.model_dump() for a in mac.graphics])
    assert short_hardware(HardwareProfile.model_validate(data)).startswith("unified memory 32 GB")


def test_the_gpu_state_hint_says_counted():
    from modelroom.views import GPU_STATE_HINT

    assert "measured = a counted card, read by modelroom or llmfit" in GPU_STATE_HINT and "none = no counted card" in GPU_STATE_HINT


def test_the_self_test_checks_the_adapter_list_in_form_only():
    """F18: `scripts/selftest.py` runs on the real machine; here only its rule for the adapter list."""
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "selftest.py"
    if not (script.parent.parent / ".git").exists():
        pytest.skip("scripts/ is not shipped in the sdist; the self-test is tested from a clone")
    spec = importlib.util.spec_from_file_location("modelroom_selftest_adapters", script)
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    base = {"gpu_state": "measured", "origin": "measured", "vram_gib": 12.0, "machine_class": "laptop", "machine_class_source": "chassis",
            "gpus": [{"vram_gib": 12.0}], "llmfit_crosscheck": {"ram_physical": {"status": "confirmed"}, "vram": {"status": "confirmed"}}}
    assert st.measurement_problems({**base, "adapters": [{"counted": True}, {"counted": False}]}) == []
    assert st.measurement_problems({**base, "adapters": []})
    assert st.measurement_problems({**base, "adapters": [{"counted": False}]})
