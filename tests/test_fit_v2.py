"""Tests for modelroom.fit.compute_fit_v2: fit contract v1 behind the schema-2 GPU gate.

The example architecture (32 layers, 8 KV heads, head_dim 128) needs exactly 1 GiB of 16-bit
KV cache at context 8192 and 2 GiB at 16384, so every number below is checkable by hand:
need = weights x 1.10 + kv + 0.50.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import get_args

import pytest

from modelroom.config import MachineConfig
from modelroom.contracts import BaseModelSpec, HardwareSnapshot, Package
from modelroom.examples import EXAMPLES
from modelroom.fit import compute_fit, compute_fit_v2
from modelroom.measurements import Scenario, default_scenario
from modelroom.profile import FIT_GPU_STATES, GpuState, HardwareProfile

GIB = 1024**3
NOW = datetime(2026, 9, 23, 8, 0, 0, tzinfo=timezone.utc)
MACHINE = MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)


def _base_model() -> BaseModelSpec:
    return BaseModelSpec.model_validate(EXAMPLES["BaseModelSpec"])


def _package(weights_gib: float = 4.0, default_context: int | None = 4096) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["files"][0]["size_bytes"] = int(weights_gib * GIB)
    payload["default_context"] = default_context
    return Package.model_validate(payload)


def _profile(**changes) -> HardwareProfile:
    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    payload.update(changes)
    return HardwareProfile.model_validate(payload)


def _cpu_profile(ram: float = 31.7) -> HardwareProfile:
    return _profile(
        ram_physical_gib=ram,
        vram_gib=0.0,
        vram_source="none",
        gpu_state="none",
        gpu_name=None,
        llmfit_crosscheck={
            "ram_physical": {"status": "confirmed", "own_gib": ram, "llmfit_gib": ram},
            "vram": {"status": "absent", "own_gib": 0.0},
        },
    )


def _scenario(context: int = 8192, requests: int = 1) -> Scenario:
    return Scenario(context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=requests)


def test_measured_gpu_profile_computes_on_the_gpu_with_the_scenario_context():
    fit = compute_fit_v2(_profile(), _package(), _base_model(), default_scenario(), MACHINE)
    assert fit.mode == "gpu"
    assert fit.kv_gib == pytest.approx(1.0)
    assert fit.need_gib == pytest.approx(4.0 * 1.10 + 1.0 + 0.5)
    assert fit.pool_gib == pytest.approx(7.0)
    assert fit.fit_class == "good"  # 5.9 / 7.0 = 0.84
    assert fit.context == 8192 and fit.context_assumed is False


def test_package_default_context_is_never_used():
    fit = compute_fit_v2(_profile(), _package(default_context=4096), _base_model(), _scenario(16384), MACHINE)
    assert fit.context == 16384
    assert fit.kv_gib == pytest.approx(2.0)


def test_cpu_profile_uses_physical_ram_and_caps_at_good():
    fit = compute_fit_v2(_cpu_profile(), _package(weights_gib=2.0), _base_model(), default_scenario(), MACHINE)
    assert fit.mode == "cpu"
    assert fit.pool_gib == pytest.approx(31.7 - 8.0)
    assert fit.fit_class == "good"  # ratio 0.16 would be perfect on a GPU


def test_ram_limit_is_a_note_and_never_the_pool():
    profile = _profile(ram_limit_gib=12.0, ram_limit_scope="cgroup")
    fit = compute_fit_v2(profile, _package(weights_gib=10.0), _base_model(), default_scenario(), MACHINE)
    assert fit.mode == "cpu_gpu"
    assert fit.pool_gib == pytest.approx(31.7 - 8.0)


def test_same_numbers_as_fit_v1_for_the_same_memory_and_context():
    snapshot = HardwareSnapshot.model_validate(
        {**EXAMPLES["HardwareSnapshot"], "vram_gib": 8.0, "ram_gib": 31.7, "installed": [], "measured_at": "2026-09-23T08:00:00Z"}
    )
    package = _package(default_context=8192)
    v1 = compute_fit(package, _base_model(), snapshot, MACHINE)
    v2 = compute_fit_v2(_profile(), package, _base_model(), default_scenario(), MACHINE)
    assert (v1.fit_class, v1.mode, v1.need_gib, v1.pool_gib) == (v2.fit_class, v2.mode, v2.need_gib, v2.pool_gib)


@pytest.mark.parametrize("state", ["present_unmeasured", "multi_gpu_not_covered", "unsupported_platform"])
def test_every_uncovered_gpu_state_is_unknown_with_its_reason(state):
    fit = compute_fit_v2(_profile(gpu_state=state), _package(), _base_model(), default_scenario(), MACHINE)
    assert fit.fit_class == "unknown"
    assert state in fit.reason


def test_a_migrated_profile_is_unknown_until_measured_again():
    profile = _profile(
        os_fingerprint="none",
        os_fingerprint_source="legacy",
        gpu_state="legacy_unknown",
        vram_source="llmfit",
        ram_physical_source="llmfit",
    )
    fit = compute_fit_v2(profile, _package(), _base_model(), default_scenario(), MACHINE)
    assert fit.fit_class == "unknown" and "measure again" in fit.reason


def test_a_llmfit_deviation_blocks_the_fit():
    block = copy.deepcopy(EXAMPLES["HardwareProfile"]["llmfit_crosscheck"])
    block["vram"] = {"status": "deviation", "own_gib": 8.0, "llmfit_gib": 6.0}
    fit = compute_fit_v2(_profile(llmfit_crosscheck=block), _package(), _base_model(), default_scenario(), MACHINE)
    assert fit.fit_class == "unknown" and "VRAM differs" in fit.reason


def test_more_than_one_request_is_computed_with_a_kv_cache_per_request():
    """Until 2026-09-25 `unknown`; since then the KV cache counts once per request (test_fit_v3.py)."""
    fit = compute_fit_v2(_profile(), _package(), _base_model(), _scenario(requests=2), MACHINE)
    assert fit.fit_class != "unknown" and fit.reason is None and fit.requests == 2


def _profile_in_state(state: str) -> HardwareProfile:
    """A valid profile for each `gpu_state`; the coupled fields differ per state by contract."""
    if state == "none":
        return _cpu_profile()
    if state == "measured":
        return _profile()
    if state in ("entered", "unified_memory"):
        return _profile(
            os_fingerprint="none",
            os_fingerprint_source="none",
            origin="entered",
            ram_physical_source="entered",
            vram_gib=12.0 if state == "entered" else 0.0,
            vram_source="entered" if state == "entered" else "none",
            gpu_state=state,
            llmfit_crosscheck={"ram_physical": {"status": "absent"}, "vram": {"status": "absent"}},
        )
    if state == "legacy_unknown":
        return _profile(
            os_fingerprint="none",
            os_fingerprint_source="legacy",
            gpu_state=state,
            vram_source="llmfit",
            ram_physical_source="llmfit",
        )
    return _profile(gpu_state=state)


@pytest.mark.parametrize("state", sorted(get_args(GpuState)))
def test_every_gpu_state_either_computes_or_is_unknown_with_a_reason_naming_it(state):
    """One case per `gpu_state` the contract lists, with the reason text a renderer shows.

    A state added to `GpuState` later fails here until the gate has a decision for it, instead
    of silently falling into `unknown` with no wording anybody checked.
    """
    fit = compute_fit_v2(_profile_in_state(state), _package(), _base_model(), default_scenario(), MACHINE)
    if state in FIT_GPU_STATES:
        assert fit.fit_class != "unknown" and fit.reason is None
    else:
        assert fit.fit_class == "unknown"
        assert fit.reason is not None and state in fit.reason


def test_an_architecture_v1_cannot_read_is_computed_from_the_size_instead():
    """Since 2026-09-24 such a package is judged from its size, not set aside (tests/test_fit_size.py)."""
    base = BaseModelSpec.model_validate(
        {**EXAMPLES["BaseModelSpec"], "architecture": {"source_repo": "acme/Nova-7B", "source_revision": None, "kind": "unknown"}}
    )
    fit = compute_fit_v2(_profile(), _package(), base, default_scenario(), MACHINE)
    assert (fit.basis, fit.reason) == ("size", None)
    assert fit.fit_class in ("good", "marginal", "too_tight")


def test_the_v1_package_rules_still_apply():
    """A package that is not a complete gguf build is `unknown`, architecture or no architecture."""
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload.update({"files": [], "complete": False, "quantization": None})
    incomplete = Package.model_validate(payload)
    fit = compute_fit_v2(_profile(), incomplete, _base_model(), default_scenario(), MACHINE)
    assert fit.reason == "only a complete gguf package is judged by fit contract v1"
