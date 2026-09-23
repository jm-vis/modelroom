"""Tests for modelroom.profile: hardware profile v2, the llmfit cross-check, reading schema 1."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.contracts import HardwareSnapshot, SchemaVersionError
from modelroom.examples import EXAMPLES
from modelroom.profile import (
    HardwareProfile,
    crosscheck,
    fit_block_reason,
    new_profile_id,
    normalize_profile_v1,
    os_fingerprint,
    read_profile_document,
)

FIXTURES = Path(__file__).parent / "fixtures" / "profiles_v1"


def _profile(**changes) -> dict:
    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    payload.update(changes)
    return payload


def _cpu_only(**changes) -> dict:
    none_check = {"status": "absent", "own_gib": 0.0}
    base = _profile(
        vram_gib=0.0,
        vram_source="none",
        gpu_state="none",
        gpu_name=None,
        llmfit_crosscheck={"ram_physical": EXAMPLES["HardwareProfile"]["llmfit_crosscheck"]["ram_physical"], "vram": none_check},
    )
    base.update(changes)
    return base


# --- identity -------------------------------------------------------------------------------


def test_new_profile_id_is_16_lowercase_hex_and_random():
    ids = {new_profile_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 16 and all(c in "0123456789abcdef" for c in i) for i in ids)


def test_os_fingerprint_is_salted_sha256_prefix_and_ignores_surrounding_whitespace():
    import hashlib

    raw = "4c4c4544-0000-1000-8000-000000000000"
    expected = hashlib.sha256(f"{raw}modelroom".encode()).hexdigest()[:16]
    assert os_fingerprint(raw) == expected
    assert os_fingerprint(f"  {raw}\n") == expected
    assert os_fingerprint(raw) != hashlib.sha256(raw.encode()).hexdigest()[:16]


def test_os_fingerprint_refuses_an_empty_identifier():
    with pytest.raises(ValueError):
        os_fingerprint("  ")


@pytest.mark.parametrize("bad", ["3F9A0C21D4E6B870", "3f9a0c21d4e6b87", "workstation"])
def test_profile_id_must_be_16_lowercase_hex(bad):
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(profile_id=bad))


def test_profile_rejects_schema_version_other_than_2():
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(schema_version=1))


def test_profile_rejects_a_naive_recorded_at():
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(recorded_at="2026-09-23T08:00:00"))


# --- coupled fields ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["none", "legacy"])
def test_fingerprint_sources_without_a_fingerprint_store_none(source):
    extra = {"gpu_state": "legacy_unknown", "vram_source": "llmfit"} if source == "legacy" else {}
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(os_fingerprint_source=source, **extra))
    HardwareProfile.model_validate(_profile(os_fingerprint_source=source, os_fingerprint="none", **extra))


def test_a_measured_fingerprint_must_be_16_hex():
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(os_fingerprint="none"))


def test_an_entered_profile_has_no_fingerprint():
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(origin="entered"))
    HardwareProfile.model_validate(_profile(origin="entered", os_fingerprint="none", os_fingerprint_source="none"))


def test_unknown_ram_is_none_exactly_when_the_source_is_unknown():
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(ram_physical_source="unknown"))
    unknown_ram_check = {"status": "absent", "own_gib": None}
    crosscheck_block = {**EXAMPLES["HardwareProfile"]["llmfit_crosscheck"], "ram_physical": unknown_ram_check}
    HardwareProfile.model_validate(
        _profile(ram_physical_source="unknown", ram_physical_gib=None, llmfit_crosscheck=crosscheck_block)
    )


def test_ram_limit_is_set_exactly_when_it_has_a_scope():
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(ram_limit_scope="cgroup"))
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(ram_limit_gib=16.0))
    HardwareProfile.model_validate(_profile(ram_limit_scope="cgroup", ram_limit_gib=16.0))


def test_gpu_state_none_requires_vram_source_none_and_zero():
    HardwareProfile.model_validate(_cpu_only())
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(gpu_state="none"))
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_cpu_only(vram_gib=2.0))


def test_gpu_state_measured_requires_vram_above_zero():
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_cpu_only(gpu_state="measured"))


def test_legacy_unknown_only_comes_from_a_migrated_profile():
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile(gpu_state="legacy_unknown"))


def test_crosscheck_own_value_must_equal_the_profile_value():
    block = copy.deepcopy(EXAMPLES["HardwareProfile"]["llmfit_crosscheck"])
    block["vram"].update(own_gib=6.0, llmfit_gib=6.0)  # consistent in itself, not with vram_gib 8.0
    with pytest.raises(ValidationError, match="own_gib"):
        HardwareProfile.model_validate(_profile(llmfit_crosscheck=block))


# --- cross-check ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "own, llmfit, status",
    [
        (31.7, 31.9, "confirmed"),
        (100.0, 95.0, "confirmed"),  # exactly 5 %
        (8.0, 7.6, "confirmed"),  # exactly 5 %, not exact in binary floating point
        (7.6, 8.0, "confirmed"),
        (100.0, 94.9, "deviation"),
        (8.0, 12.0, "deviation"),
        (0.0, 0.0, "confirmed"),
        (8.0, None, "absent"),
    ],
)
def test_crosscheck_tolerance_is_five_percent_of_the_larger_value(own, llmfit, status):
    assert crosscheck(own, llmfit).status == status


@pytest.mark.parametrize(
    "stored",
    [
        {"status": "confirmed", "own_gib": 8.0, "llmfit_gib": 16.0},
        {"status": "deviation", "own_gib": 8.0, "llmfit_gib": 8.1},
    ],
)
def test_a_stored_status_must_follow_the_five_percent_rule(stored):
    from modelroom.profile import CrossCheck

    with pytest.raises(ValidationError, match="5 %"):
        CrossCheck.model_validate(stored)


def test_a_stored_confirmed_at_exactly_five_percent_is_accepted():
    from modelroom.profile import CrossCheck

    CrossCheck.model_validate({"status": "confirmed", "own_gib": 8.0, "llmfit_gib": 7.6})


def test_crosscheck_absent_or_error_carries_no_llmfit_value():
    from modelroom.profile import CrossCheck

    with pytest.raises(ValidationError):
        CrossCheck.model_validate({"status": "error", "own_gib": 8.0, "llmfit_gib": 8.0})
    with pytest.raises(ValidationError):
        CrossCheck.model_validate({"status": "confirmed", "own_gib": 8.0})


# --- fit gate -----------------------------------------------------------------------------------


def test_fit_gate_open_for_measured_and_cpu_profiles():
    assert fit_block_reason(HardwareProfile.model_validate(EXAMPLES["HardwareProfile"])) is None
    assert fit_block_reason(HardwareProfile.model_validate(_cpu_only())) is None


@pytest.mark.parametrize("state", ["present_unmeasured", "multi_gpu_not_covered", "unified_memory", "unsupported_platform"])
def test_fit_gate_closed_for_every_other_gpu_state(state):
    profile = HardwareProfile.model_validate(_profile(gpu_state=state))
    assert state in fit_block_reason(profile)


def test_fit_gate_closed_on_a_llmfit_deviation():
    block = copy.deepcopy(EXAMPLES["HardwareProfile"]["llmfit_crosscheck"])
    block["ram_physical"] = {"status": "deviation", "own_gib": 31.7, "llmfit_gib": 24.0}
    profile = HardwareProfile.model_validate(_profile(llmfit_crosscheck=block))
    assert "RAM differs" in fit_block_reason(profile)


# --- reading schema 1 and 3 -------------------------------------------------------------------


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_read_profile_document_reads_schema_1_as_hardware_snapshot():
    assert isinstance(read_profile_document(_fixture("windows-nvidia-laptop.json")), HardwareSnapshot)


def test_read_profile_document_reads_schema_2_as_hardware_profile():
    assert isinstance(read_profile_document(copy.deepcopy(EXAMPLES["HardwareProfile"])), HardwareProfile)


@pytest.mark.parametrize("version", [3, 0, None, "2", True])
def test_read_profile_document_refuses_other_versions_before_field_validation(version):
    with pytest.raises(SchemaVersionError):
        read_profile_document({"schema_version": version, "garbage": True})


@pytest.mark.parametrize(
    "name, gpu_state, vram",
    [
        ("windows-nvidia-laptop.json", "legacy_unknown", 7.6),
        ("linux-cpu-server.json", "legacy_unknown", 0.0),
        ("unified-memory.json", "unified_memory", 48.0),
        ("vram-zero.json", "legacy_unknown", 0.0),
    ],
)
def test_normalize_profile_v1_never_guesses_the_gpu_layout(name, gpu_state, vram):
    legacy = HardwareSnapshot.model_validate(_fixture(name))
    profile = normalize_profile_v1(legacy, "0123456789abcdef")
    assert profile.gpu_state == gpu_state
    assert profile.vram_gib == vram
    assert profile.vram_source == "llmfit"
    assert profile.ram_physical_source == "llmfit"
    assert profile.ram_physical_gib == legacy.ram_gib
    assert profile.display_name == legacy.machine
    assert profile.os_fingerprint == "none" and profile.os_fingerprint_source == "legacy"
    assert profile.recorded_at == legacy.measured_at
    assert profile.llmfit_crosscheck.vram.status == "absent"
    assert fit_block_reason(profile) is not None
