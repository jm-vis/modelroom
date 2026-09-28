"""AP9-GPU G1: the adapter list of schema 4 -- its rules for a list that is not empty, and every
profile an empty list is valid for (D1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fixture_gpu import fixture, gpu_runner, reference
from fixture_support import FIXTURES, export_of, v2_profile, v3_profile
from modelroom.contracts import HardwareSnapshot
from modelroom.examples import EXAMPLES
from modelroom.llmfit import LlmfitGpu
from modelroom.measure import build_profile, measure_hardware
from modelroom.measure_base import FixtureFiles
from modelroom.profile import GraphicsAdapter, HardwareProfile, normalize_profile_v1, read_profile_document

NOW = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)
CARD = {"vendor": "nvidia", "name": "Nova GPU", "memory_gib": 11.94, "memory_kind": "dedicated", "form": "discrete",
        "source": "nvidia-smi", "evidence": "nvidia-smi", "backend": "cuda", "usable": "yes", "counted": True, "reason": None}
IGPU = {"vendor": "intel", "name": "Intel(R) Graphics", "memory_gib": None, "memory_kind": "shared", "form": "integrated",
        "source": "windows", "backend": "vulkan", "usable": "unconfirmed", "counted": False,
        "reason": "integrated graphics shares the system memory"}


def _measured(**changes) -> dict:
    data = v2_profile(vram_gib=11.94).model_dump(mode="json")
    data["adapters"] = [CARD, IGPU]
    data.update(changes)
    return data


def test_a_measured_profile_with_its_adapters_is_valid_and_round_trips():
    profile = HardwareProfile.model_validate(_measured())
    again = HardwareProfile.model_validate(json.loads(profile.model_dump_json()))
    assert again == profile and len(again.adapters) == 2


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"origin": "entered", "os_fingerprint": "none", "os_fingerprint_source": "none"}, "measured by this version"),
        ({"adapters": [IGPU]}, "cards of gpus"),
        ({"adapters": [{**CARD, "name": "Other"}, IGPU]}, "cards of gpus"),
        ({"adapters": [{**CARD, "memory_gib": 8.0}, IGPU]}, "cards of gpus"),
        ({"adapters": [{**CARD, "source": "sysfs"}, IGPU]}, "vram_source"),
    ],
    ids=["entered", "counted-missing", "name-differs", "size-differs", "other-source"],
)
def test_every_rule_of_a_list_that_is_not_empty(changes, words):
    with pytest.raises(ValidationError, match=words):
        HardwareProfile.model_validate(_measured(**changes))


def test_a_state_without_cards_counts_no_adapter():
    data = _measured(gpu_state="none", vram_gib=0.0, vram_source="none", gpu_name=None, gpus=[])
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(data)
    data["adapters"] = [IGPU]
    assert HardwareProfile.model_validate(data).gpu_state == "none"


@pytest.mark.parametrize(
    ("adapter", "valid"),
    [({**IGPU, "reason": None}, False), ({**CARD, "counted": False, "reason": None}, False), ({**CARD, "usable": "no"}, False),
     ({**CARD, "counted": True, "memory_gib": None}, False), ({**IGPU, "pci_id": "8086:7D67"}, False), ({**IGPU, "pci_slot": "00:02.0"}, False),
     ({**IGPU, "pci_id": "8086:7d67", "pci_slot": "0000:00:02.0"}, True)],
    ids=["reason-missing", "uncounted-without-reason", "not-usable-counted", "counted-without-memory", "upper-id", "short-slot", "ids"],
)
def test_one_adapter_is_checked_on_its_own(adapter, valid):
    if valid:
        GraphicsAdapter.model_validate(adapter)
    else:
        with pytest.raises(ValidationError):
            GraphicsAdapter.model_validate(adapter)


APPLE = {"vendor": "apple", "name": "Apple M3 Max", "memory_gib": 64.0, "memory_kind": "unified", "form": "unified",
         "source": "sysctl", "evidence": "sysctl", "backend": "metal", "usable": "yes", "counted": False, "reason": None}


def _unified(**changes) -> dict:
    data = v2_profile(vram_gib=11.94, ram_gib=64.0).model_dump(mode="json")
    data.update(gpu_state="unified_memory", vram_gib=64.0, vram_source="sysctl", gpu_name="Apple M3 Max", gpus=[], adapters=[APPLE])
    data.update(changes)
    return data


def test_unified_memory_is_one_unified_adapter_with_the_ram_from_sysctl_or_both_unknown():
    assert HardwareProfile.model_validate(_unified()).vram_source == "sysctl"
    unknown = _unified(vram_gib=None, vram_source="unknown", ram_physical_gib=None, ram_physical_source="unknown",
                       adapters=[{**APPLE, "memory_gib": None}])
    assert HardwareProfile.model_validate(unknown).vram_gib is None
    for broken in (_unified(vram_gib=32.0), _unified(adapters=[APPLE, IGPU]), _unified(vram_source="llmfit")):
        with pytest.raises(ValidationError):
            HardwareProfile.model_validate(broken)


# --- an empty list is valid for every profile (D1) -------------------------------------------------------


def test_an_empty_list_is_valid_for_schema_1_including_unified_memory_from_llmfit():
    for unified in (True, False):
        legacy = HardwareSnapshot.model_validate({**EXAMPLES["HardwareSnapshot"], "unified_memory": unified})
        profile = normalize_profile_v1(legacy, "0123456789abcdef")
        assert profile.adapters == [] and profile.gpu_state in ("unified_memory", "legacy_unknown")


@pytest.mark.parametrize("version", [2, 3])
def test_an_empty_list_for_schema_2_and_3(version):
    assert read_profile_document(v3_profile(schema_version=version)).adapters == []


def test_a_schema_4_file_without_the_field_reads_and_is_written_back_without_it():
    data = v2_profile().model_dump(mode="json")
    assert "adapters" not in data
    assert read_profile_document(data).adapters == []


def test_every_recorded_server_profile_of_2026_09_27_reads():
    for path in sorted((FIXTURES / "profiles_v4").glob("*.json")):
        assert read_profile_document(json.loads(path.read_text(encoding="utf-8"))).adapters == []


def test_an_export_reads_without_adapters(tmp_path: Path):
    from modelroom.measurements import load_export

    export = load_export(json.loads(export_of("two_l4_server.json", tmp_path).read_text(encoding="utf-8")))
    assert export.profile.adapters == []


def test_cpu_only_leaves_the_list_empty():
    files = FixtureFiles({"/proc/meminfo": "MemTotal: 33554432 kB\n"})
    measured = measure_hardware("linux", gpu_runner(), files, lambda: 0, cpu_only=True, reference=reference(LlmfitGpu("NVIDIA L4", "CUDA", 22.49, 1)))
    profile = build_profile(measured, reference(), "0123456789abcdef", "box", NOW)
    assert profile.adapters == [] and profile.origin == "entered"


@pytest.mark.parametrize("shape", ["card", "cards", "none", "unified"])
def test_the_four_hand_entered_shapes_keep_an_empty_list(shape):
    from modelroom.guided_entered import entered_profile

    profile = entered_profile("studio", 64.0, shape, 24.0, "0123456789abcdef", NOW, cards=2)
    assert profile.adapters == [] and "adapters" not in profile.model_dump(mode="json")


# --- refused -------------------------------------------------------------------------------------------------


def test_a_hand_entered_profile_with_adapters_is_refused():
    data = v2_profile().model_dump(mode="json")
    data.update(origin="entered", os_fingerprint="none", os_fingerprint_source="none", adapters=[IGPU])
    with pytest.raises(ValidationError, match="measured by this version"):
        HardwareProfile.model_validate(data)


def test_schema_3_with_adapters_is_refused():
    with pytest.raises(ValueError, match="no field 'adapters'"):
        read_profile_document(v3_profile(adapters=[]))


@pytest.mark.parametrize("source", ["sysfs", "sysctl"])
def test_schema_3_with_a_new_memory_source_is_refused(source):
    with pytest.raises(ValueError, match=f"no vram_source '{source}'"):
        read_profile_document(v3_profile(vram_source=source))


def test_a_whole_laptop_measurement_writes_a_valid_profile_with_both_units():
    from fixture_support import windows_runner

    measured = measure_hardware("win32", windows_runner(), FixtureFiles({}), lambda: 137_112_547_328, reference=reference(LlmfitGpu("NVIDIA RTX PRO 3000", "CUDA", 11.94, 1)))
    profile = build_profile(measured, reference(LlmfitGpu("NVIDIA RTX PRO 3000", "CUDA", 11.94, 1)), "0123456789abcdef", "laptop", NOW)
    assert [(adapter.vendor, adapter.counted) for adapter in profile.adapters] == [("nvidia", True), ("intel", False)]
    assert profile.llmfit_crosscheck.vram.status == "confirmed"
    assert fixture("win_video_controllers_laptop.json.txt").count("\\\\X") == 2
