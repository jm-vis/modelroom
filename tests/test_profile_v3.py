"""Hardware profile schema 3: a machine entered by hand, and reading schema 2 everywhere.

Schema 3 adds the value `entered` to `ram_physical_source`, `vram_source` and `gpu_state`, and
makes `unified_memory` a state the fit computes for. Exactly three hand-entered shapes are valid
(CONTRACTS.md, "HardwareProfile"; decided 2026-09-25). A schema-2 file reads as schema 3 through
`read_profile_document`, inside an export file (`load_export`) and in `render`'s own probe.
"""

from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import modelroom.render as render_module
from modelroom.contracts import HardwareSnapshot, SchemaVersionError
from modelroom.examples import EXAMPLES
from modelroom.measurements import load_export
from modelroom.profile import (
    FIT_GPU_STATES,
    PROFILE_SCHEMA_RANGE,
    PROFILE_SCHEMA_VERSION,
    HardwareProfile,
    fit_block_reason,
    normalize_profile_v1,
    read_profile_document,
)

FIXTURES = Path(__file__).parent / "fixtures"
ABSENT = {"status": "absent"}


def _entered(**changes) -> dict:
    """A machine entered by hand with its own graphics card of 12 GiB."""
    data = copy.deepcopy(EXAMPLES["HardwareProfile"])
    data.update(
        schema_version=PROFILE_SCHEMA_VERSION,
        display_name="office box",
        os_fingerprint="none",
        os_fingerprint_source="none",
        origin="entered",
        ram_physical_gib=32.0,
        ram_physical_source="entered",
        vram_gib=12.0,
        vram_source="entered",
        gpu_state="entered",
        gpu_name=None,
        llmfit_crosscheck={"ram_physical": ABSENT, "vram": ABSENT},
        llmfit_version=None,
    )
    data.update(changes)
    return data


def _no_card(**changes) -> dict:
    return _entered(**{"vram_gib": 0.0, "vram_source": "none", "gpu_state": "none", **changes})


def _unified(**changes) -> dict:
    return _entered(**{"vram_gib": 0.0, "vram_source": "none", "gpu_state": "unified_memory", **changes})


def test_the_profile_is_schema_3_and_readers_accept_1_to_3():
    assert PROFILE_SCHEMA_VERSION == 3
    assert PROFILE_SCHEMA_RANGE == (1, 4)


# --- the three hand-entered shapes ----------------------------------------------------------------


@pytest.mark.parametrize("build", [_entered, _no_card, _unified], ids=["graphics-card", "no-card", "unified"])
def test_each_of_the_three_entered_shapes_is_valid_and_computes(build):
    profile = HardwareProfile.model_validate(build())
    assert profile.origin == "entered" and profile.ram_physical_source == "entered"
    assert fit_block_reason(profile) is None


@pytest.mark.parametrize(
    "payload",
    [
        _entered(origin="measured", os_fingerprint="9d2f4b6a8c0e1357", os_fingerprint_source="windows_machineguid"),
        _no_card(origin="measured", os_fingerprint="9d2f4b6a8c0e1357", os_fingerprint_source="windows_machineguid"),
        # a measured GPU never carries an entered size
        _entered(gpu_state="measured"),
        # an entered GPU state needs an entered size above zero
        _entered(vram_source="nvidia-smi"),
        _entered(vram_gib=0.0),
        # an entered size belongs to the entered GPU state and to nothing else
        _entered(gpu_state="present_unmeasured"),
        _entered(gpu_state="unified_memory"),
        # a hand-entered memory is one of the three shapes
        _entered(vram_gib=0.0, vram_source="none", gpu_state="present_unmeasured"),
        _unified(vram_gib=16.0, vram_source="llmfit"),
        # an entered GPU needs an entered memory too
        _entered(ram_physical_source="os"),
        # nothing was cross-checked for a machine entered by hand
        _entered(llmfit_crosscheck={"ram_physical": {"status": "confirmed", "own_gib": 32.0, "llmfit_gib": 32.0}, "vram": ABSENT}),
    ],
    ids=[
        "ram-entered-measured-origin",
        "no-card-measured-origin",
        "measured-state-entered-size",
        "entered-state-measured-size",
        "entered-state-zero",
        "entered-size-other-state",
        "entered-size-unified",
        "hand-memory-present-unmeasured",
        "hand-unified-with-vram",
        "entered-gpu-measured-ram",
        "hand-crosschecked",
    ],
)
def test_every_forbidden_coupling_is_invalid(payload):
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(payload)


def test_a_cpu_only_profile_keeps_its_measured_sources():
    """`hardware --cpu-only` writes `origin: entered` with a RAM the OS measured; that stays valid."""
    cpu_only = _no_card(ram_physical_source="os", ram_physical_gib=31.7)
    profile = HardwareProfile.model_validate(cpu_only)
    assert fit_block_reason(profile) is None


def test_the_fit_computes_for_entered_and_unified_memory():
    assert FIT_GPU_STATES == frozenset({"none", "measured", "entered", "unified_memory"})


def test_a_unified_memory_profile_from_schema_1_is_still_measured_again_first():
    legacy = HardwareSnapshot.model_validate(json.loads((FIXTURES / "profiles_v1" / "unified-memory.json").read_text(encoding="utf-8")))
    profile = normalize_profile_v1(legacy, "0123456789abcdef")
    assert profile.gpu_state == "unified_memory" and profile.schema_version == 3
    reason = fit_block_reason(profile)
    assert reason is not None and "measure again" in reason and "unified_memory" in reason


# --- reading schema 2 -----------------------------------------------------------------------------


def _v2() -> dict:
    return {**copy.deepcopy(EXAMPLES["HardwareProfile"]), "schema_version": 2}


def test_a_schema_2_profile_file_reads_as_schema_3_with_every_value_kept():
    data = _v2()
    profile = read_profile_document(data)
    assert isinstance(profile, HardwareProfile) and profile.schema_version == 3
    assert profile.model_dump(mode="json", exclude={"schema_version"}) == {k: v for k, v in _v2().items() if k != "schema_version"}
    assert data["schema_version"] == 2  # the caller's dict is left as it was


def test_a_schema_2_profile_cannot_carry_a_value_of_schema_3():
    with pytest.raises(ValueError, match="schema 2"):
        read_profile_document({**_entered(), "schema_version": 2})


@pytest.mark.parametrize("version", [4, 0, None, "3", True])
def test_read_profile_document_refuses_versions_outside_1_to_3(version):
    with pytest.raises(SchemaVersionError):
        read_profile_document({"schema_version": version, "garbage": True})


def test_an_export_file_with_an_embedded_schema_2_profile_still_loads():
    data = json.loads((FIXTURES / "export_v1.json").read_text(encoding="utf-8"))
    assert data["profile"]["schema_version"] == 2
    export = load_export(data)
    assert export.schema_version == 1
    assert export.profile.schema_version == 3
    assert data["profile"]["schema_version"] == 2


def test_the_render_module_imports_with_its_probe_profile_on_the_current_schema():
    """`render` builds a probe profile at import time; a fixed old schema number there fails the import.

    A fresh interpreter, so the module is really imported again and no other test sees a reloaded one.
    """
    code = "import modelroom.render as r; print(r.LEGACY_FIT_REASON)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "measure again" in done.stdout
    assert "schema_version=PROFILE_SCHEMA_VERSION" in inspect.getsource(render_module._legacy_fit_reason)
