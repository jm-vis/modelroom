"""Hardware profile schema 4: every graphics card, and the machine class (decided 2026-09-26).

Schema 4 adds the card list `gpus`, the state `multi_gpu` for two cards or more, and the machine
class with its source (CONTRACTS.md, "HardwareProfile"). A schema-2 or schema-3 file reads as
schema 4 in two steps with a fixed target each (`normalize_profile_v2`, `normalize_profile_v3`),
in both readers of a stored profile (`read_profile_document`, `load_export`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.contracts import HardwareSnapshot, SchemaVersionError
from modelroom.examples import EXAMPLES
from modelroom.measurements import load_export
from modelroom.profile import (
    ENTERED_GPU_STATES,
    FIT_GPU_STATES,
    PROFILE_SCHEMA_RANGE,
    PROFILE_SCHEMA_VERSION,
    HardwareProfile,
    fit_block_reason,
    normalize_profile_v1,
    normalize_profile_v2,
    normalize_profile_v3,
    read_profile_document,
)

from fixture_support import v3_profile

FIXTURES = Path(__file__).parent / "fixtures"
ABSENT = {"status": "absent"}
NO_CHECK = {"ram_physical": ABSENT, "vram": ABSENT}


def test_the_profile_is_schema_4_and_readers_accept_1_to_4():
    assert PROFILE_SCHEMA_VERSION == 4
    assert PROFILE_SCHEMA_RANGE == (1, 5)


def test_several_graphics_cards_compute_measured_or_entered():
    assert "multi_gpu" in FIT_GPU_STATES and "multi_gpu" in ENTERED_GPU_STATES
    assert "multi_gpu_not_covered" not in FIT_GPU_STATES


# --- every profile 0.1.0 writes is valid after the step to schema 4 --------------------------------


def _cpu_only(**changes) -> dict:
    base = v3_profile(
        origin="entered",
        os_fingerprint="none",
        os_fingerprint_source="none",
        vram_gib=0.0,
        vram_source="none",
        gpu_state="none",
        gpu_name=None,
        llmfit_crosscheck={"ram_physical": ABSENT, "vram": {"status": "absent", "own_gib": 0.0}},
    )
    return {**base, **changes}


RAM_CHECK = {"status": "confirmed", "own_gib": 31.7, "llmfit_gib": 31.9}


def _unmeasured(state: str, **changes) -> dict:
    base = v3_profile(
        vram_gib=None,
        vram_source="unknown",
        gpu_state=state,
        gpu_name=None,
        llmfit_crosscheck={"ram_physical": RAM_CHECK, "vram": ABSENT},
    )
    return {**base, **changes}


def _hand(state: str, vram_gib: float, vram_source: str) -> dict:
    return v3_profile(
        display_name="office box",
        os_fingerprint="none",
        os_fingerprint_source="none",
        origin="entered",
        ram_physical_gib=32.0,
        ram_physical_source="entered",
        vram_gib=vram_gib,
        vram_source=vram_source,
        gpu_state=state,
        gpu_name=None,
        llmfit_crosscheck=NO_CHECK,
        llmfit_version=None,
    )


WRITTEN_BY_0_1_0 = {
    "one-card": v3_profile(),
    "cpu-only": _cpu_only(),
    "cpu-only-darwin": _cpu_only(vram_gib=None, vram_source="unknown", gpu_state="unsupported_platform",
                                 llmfit_crosscheck={"ram_physical": ABSENT, "vram": ABSENT}),
    "no-card": v3_profile(vram_gib=0.0, vram_source="none", gpu_state="none", gpu_name=None,
                          llmfit_crosscheck={"ram_physical": RAM_CHECK, "vram": {"status": "confirmed", "own_gib": 0.0, "llmfit_gib": 0.0}}),
    "present-unmeasured": _unmeasured("present_unmeasured", gpu_name="Nova GPU"),
    "multi-gpu-not-covered": _unmeasured("multi_gpu_not_covered"),
    "unsupported-platform": _unmeasured("unsupported_platform"),
    "hand-card": _hand("entered", 12.0, "entered"),
    "hand-no-card": _hand("none", 0.0, "none"),
    "hand-unified": _hand("unified_memory", 0.0, "none"),
    "gpu-name-empty": v3_profile(gpu_name=""),
    "gpu-name-long": v3_profile(gpu_name="N" * 300),
    "llmfit-vram-source": v3_profile(vram_source="llmfit"),
}


@pytest.mark.parametrize("name", sorted(WRITTEN_BY_0_1_0))
def test_every_profile_0_1_0_writes_is_valid_as_schema_4(name):
    data = WRITTEN_BY_0_1_0[name]
    assert HardwareProfile.model_validate(normalize_profile_v3(data)) == read_profile_document(data)
    profile = read_profile_document(data)
    assert profile.schema_version == 4
    assert (profile.machine_class, profile.machine_class_source) == ("unknown", "unknown")
    cards = [(gpu.index, gpu.name, gpu.vram_gib, gpu.vram_source) for gpu in profile.gpus]
    if data["gpu_state"] in ("measured", "entered"):
        assert cards == [(0, data["gpu_name"], data["vram_gib"], data["vram_source"])]
    else:
        assert cards == []


def test_a_schema_1_profile_normalizes_with_an_empty_card_list_and_its_vram_kept():
    """An empty list says nothing about `vram_gib`: schema 1 keeps llmfit's VRAM above 0."""
    legacy = HardwareSnapshot.model_validate(json.loads((FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json").read_text(encoding="utf-8")))
    profile = normalize_profile_v1(legacy, "0123456789abcdef")
    assert profile.gpu_state == "legacy_unknown" and profile.vram_gib and profile.vram_gib > 0
    assert profile.gpus == [] and (profile.machine_class, profile.machine_class_source) == ("unknown", "unknown")


# --- the four shapes of a machine entered by hand, and the machine class ----------------------------


def _v4(**changes) -> dict:
    data = copy.deepcopy(EXAMPLES["HardwareProfile"])
    data.update(changes)
    return data


def _hand_v4(state: str = "entered", cards: int = 1, size: float = 12.0, **changes) -> dict:
    card_states = ("entered", "multi_gpu")
    data = _v4(
        display_name="office box",
        os_fingerprint="none",
        os_fingerprint_source="none",
        origin="entered",
        ram_physical_gib=128.0,
        ram_physical_source="entered",
        vram_gib=cards * size if state in card_states else 0.0,
        vram_source="entered" if state in card_states else "none",
        gpu_state=state,
        gpu_name=None,
        gpus=[{"index": i, "name": None, "vram_gib": size, "vram_source": "entered"} for i in range(cards)]
        if state in card_states else [],
        machine_class="server",
        machine_class_source="entered",
        llmfit_crosscheck=NO_CHECK,
        llmfit_version=None,
    )
    data.update(changes)
    return data


def _measured_cards(sizes: tuple[float, ...] = (40.0, 40.0), **changes) -> dict:
    data = _v4(
        vram_gib=round(sum(sizes), 2),
        vram_source="nvidia-smi",
        gpu_state="multi_gpu",
        gpu_name=None,
        gpus=[{"index": i, "name": "NVIDIA A100-SXM4-40GB", "vram_gib": size, "vram_source": "nvidia-smi"} for i, size in enumerate(sizes)],
        machine_class="server",
        machine_class_source="chassis",
        llmfit_crosscheck={"ram_physical": EXAMPLES["HardwareProfile"]["llmfit_crosscheck"]["ram_physical"], "vram": {"status": "absent", "own_gib": round(sum(sizes), 2)}},
    )
    data.update(changes)
    return data


@pytest.mark.parametrize(
    "payload",
    [
        _hand_v4("entered"),
        _hand_v4("multi_gpu", cards=2, size=24.0),
        _hand_v4("none"),
        _hand_v4("unified_memory"),
        _hand_v4("multi_gpu", cards=2, size=24.0, machine_class="unknown", machine_class_source="unknown"),
        _measured_cards(),
        _measured_cards((48.0, 16.0)),
        _measured_cards((40.0, 40.0, 40.0)),
    ],
    ids=["hand-card", "hand-cards", "hand-no-card", "hand-unified", "hand-cards-no-class", "measured-two",
         "measured-mixed", "measured-three"],
)
def test_the_shapes_of_schema_4_are_valid_and_compute(payload):
    profile = HardwareProfile.model_validate(payload)
    assert fit_block_reason(profile) is None


def test_a_cpu_only_profile_may_carry_a_class_read_from_the_chassis():
    """`--cpu-only` is `origin: entered` with a memory the OS read, so the chassis counts there."""
    data = _v4(origin="entered", os_fingerprint="none", os_fingerprint_source="none", vram_gib=0.0,
               vram_source="none", gpu_state="none", gpu_name=None, gpus=[], machine_class="laptop",
               machine_class_source="chassis",
               llmfit_crosscheck={"ram_physical": ABSENT, "vram": {"status": "absent", "own_gib": 0.0}})
    assert HardwareProfile.model_validate(data).machine_class == "laptop"


@pytest.mark.parametrize(
    "payload",
    [
        _hand_v4("entered", machine_class_source="chassis"),
        _v4(gpus=[]),
        _hand_v4("entered", gpus=[{"index": 0, "name": None, "vram_gib": 6.0, "vram_source": "entered"},
                                  {"index": 1, "name": None, "vram_gib": 6.0, "vram_source": "entered"}]),
        _measured_cards(vram_gib=70.0, llmfit_crosscheck={"ram_physical": ABSENT, "vram": {"status": "absent", "own_gib": 70.0}}),
        _hand_v4("multi_gpu", cards=2, size=24.0, vram_gib=40.0,
                 gpus=[{"index": 0, "name": None, "vram_gib": 24.0, "vram_source": "entered"},
                       {"index": 1, "name": None, "vram_gib": 16.0, "vram_source": "entered"}]),
        _hand_v4("multi_gpu", cards=2, size=24.0,
                 llmfit_crosscheck={"ram_physical": {"status": "confirmed", "own_gib": 128.0, "llmfit_gib": 128.0}, "vram": ABSENT}),
        _v4(machine_class="laptop", machine_class_source="unknown"),
        _v4(machine_class="unknown", machine_class_source="chassis"),
        _v4(vram_gib=0.0, vram_source="none", gpu_state="none", gpu_name=None),
        _measured_cards(gpus=[{"index": 0, "name": "A", "vram_gib": 40.0, "vram_source": "nvidia-smi"},
                              {"index": 2, "name": "A", "vram_gib": 40.0, "vram_source": "nvidia-smi"}]),
        _measured_cards(gpus=[{"index": 0, "name": "A", "vram_gib": 40.0, "vram_source": "nvidia-smi"},
                              {"index": 1, "name": "A", "vram_gib": 40.0, "vram_source": "llmfit"}]),
        _v4(gpus=[{"index": 0, "name": "Other GPU", "vram_gib": 8.0, "vram_source": "nvidia-smi"}]),
        _measured_cards(gpus=[{"index": 0, "name": "A", "vram_gib": 80.0, "vram_source": "nvidia-smi"}], vram_gib=80.0,
                        llmfit_crosscheck={"ram_physical": ABSENT, "vram": {"status": "absent", "own_gib": 80.0}}),
        _measured_cards(vram_source="entered"),
        _hand_v4("multi_gpu", cards=2, size=24.0, vram_source="nvidia-smi",
                 gpus=[{"index": i, "name": None, "vram_gib": 24.0, "vram_source": "nvidia-smi"} for i in range(2)]),
    ],
    ids=[
        "chassis-with-hand-memory",
        "measured-without-cards",
        "entered-with-two-cards",
        "sum-differs-from-vram",
        "hand-cards-of-two-sizes",
        "hand-cards-crosschecked",
        "class-without-source",
        "chassis-without-class",
        "no-card-with-a-card",
        "indexes-not-in-order",
        "card-source-differs",
        "card-name-differs",
        "multi-with-one-card",
        "measured-cards-entered-source",
        "hand-cards-measured-source",
    ],
)
def test_every_forbidden_coupling_of_schema_4_is_invalid(payload):
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(payload)


# --- reading schema 2 and 3 in two steps -------------------------------------------------------------


@pytest.mark.parametrize("gpu_name", [None, "Nova GPU"])
def test_schema_2_reads_as_3_then_4_with_the_name_kept_and_the_dict_unchanged(gpu_name):
    data = v3_profile(schema_version=2, gpu_name=gpu_name)
    before = copy.deepcopy(data)

    step_one = normalize_profile_v2(data)
    step_two = normalize_profile_v3(step_one)

    assert step_one["schema_version"] == 3 and "gpus" not in step_one
    assert step_two["schema_version"] == 4
    assert step_two["gpus"] == [{"index": 0, "name": gpu_name, "vram_gib": 8.0, "vram_source": "nvidia-smi"}]
    assert (step_two["machine_class"], step_two["machine_class_source"]) == ("unknown", "unknown")
    assert data == before and step_one == {**before, "schema_version": 3}
    assert read_profile_document(data).model_dump(mode="json") == step_two


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize(
    "change",
    [{"gpus": []}, {"machine_class": "laptop"}, {"machine_class_source": "chassis"}, {"gpu_state": "multi_gpu"}],
    ids=["gpus", "machine_class", "machine_class_source", "multi_gpu"],
)
def test_schema_2_and_3_cannot_carry_what_schema_4_added(version, change):
    with pytest.raises(ValueError, match=f"schema {version}"):
        read_profile_document(v3_profile(schema_version=version, **change))


@pytest.mark.parametrize(
    ("name", "value"),
    [("ram_physical_source", "entered"), ("vram_source", "entered"), ("gpu_state", "entered"), ("gpu_state", "multi_gpu")],
)
def test_the_step_from_schema_2_refuses_every_value_a_later_schema_added_on_its_own(name, value):
    with pytest.raises(ValueError, match=f"schema 2 has no {name} '{value}'"):
        normalize_profile_v2(v3_profile(schema_version=2, **{name: value}))


@pytest.mark.parametrize(("vram_gib", "valid"), [(79.99, True), (80.01, True), (79.98, False), (80.02, False)])
def test_the_sum_of_the_cards_holds_within_its_tolerance_on_both_sides(vram_gib, valid):
    payload = _measured_cards((40.0, 40.0), vram_gib=vram_gib,
                              llmfit_crosscheck={"ram_physical": ABSENT, "vram": {"status": "absent", "own_gib": vram_gib}})
    if valid:
        assert HardwareProfile.model_validate(payload).vram_gib == vram_gib
    else:
        with pytest.raises(ValidationError):
            HardwareProfile.model_validate(payload)


@pytest.mark.parametrize("version", [5, 0, None, "4", True])
def test_read_profile_document_refuses_versions_outside_1_to_4(version):
    with pytest.raises(SchemaVersionError):
        read_profile_document({"schema_version": version, "garbage": True})


@pytest.mark.parametrize("name", ["export_v1.json", "export_v1_profile_v3.json"])
def test_an_export_with_an_embedded_schema_2_or_3_profile_loads_as_schema_4(name):
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    stored = data["profile"]["schema_version"]
    export = load_export(data)
    assert export.profile.schema_version == 4
    assert [gpu.name for gpu in export.profile.gpus] == ["Nova GPU"]
    assert data["profile"]["schema_version"] == stored and "gpus" not in data["profile"]


def test_an_export_whose_schema_3_profile_carries_a_card_list_is_refused():
    data = json.loads((FIXTURES / "export_v1_profile_v3.json").read_text(encoding="utf-8"))
    data["profile"]["gpus"] = []
    with pytest.raises(ValueError, match="schema 3"):
        load_export(data)
