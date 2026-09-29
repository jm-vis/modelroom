"""The fit with the reserve of the system memory computed from the machine (AP9-RM RM-1, ADR 0012).

Every entry point reads the reserve through `reserve.ram_reserve` inside `fit._choose_pool`, so the
fit of a profile, the fit from the size or from a parameter count, and the size scale of the
guided mode all compute with the same number. The needs below are built through
`fit_from_parameters` at 1024 tokens: `need = weights x 1.10 + 0.140625 + 0.50`, so a parameter
count is chosen for the need a test names (`_parameters_for`).
"""

from __future__ import annotations

import copy

import pytest

from modelroom.config import MachineConfig
from modelroom.contracts import BaseModelSpec, Package
from modelroom.examples import EXAMPLES
from modelroom.fit import (
    BYTES_PER_PARAMETER_Q4,
    FIXED_OVERHEAD_GIB,
    GIB,
    KV_PER_TOKEN_SIZE_BASIS,
    WEIGHTS_OVERHEAD_RATIO,
    GpuPool,
    compute_fit_v2,
    count_fitting,
    fit_from_parameters,
    fit_from_size,
)
from modelroom.measurements import Scenario
from modelroom.profile import HardwareProfile
from modelroom.ranking import rank_packages

LAPTOP_RAM = 127.70
CONTEXT = 1024
KV_GIB = CONTEXT * KV_PER_TOKEN_SIZE_BASIS / GIB  # 0.140625


def _machine(reserve: float | None = None, vram_reserve: float = 1.0) -> MachineConfig:
    return MachineConfig(reserve_ram_gib=reserve, reserve_vram_gib=vram_reserve, writer=True)


def _weights_for(need_gib: float) -> float:
    return (need_gib - KV_GIB - FIXED_OVERHEAD_GIB) / WEIGHTS_OVERHEAD_RATIO


def _parameters_for(need_gib: float) -> float:
    """The parameter count (billions) whose `Q4_K_M` build needs `need_gib` at 1024 tokens."""
    return _weights_for(need_gib) * GIB / (BYTES_PER_PARAMETER_Q4 * 1e9)


def _profile(ram_gib: float | None = LAPTOP_RAM, vram_gib: float = 12.0, state: str = "measured") -> HardwareProfile:
    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    measured = state == "measured"
    payload.update(
        ram_physical_gib=ram_gib,
        ram_physical_source="os" if ram_gib is not None else "unknown",
        vram_gib=vram_gib if measured else 0.0,
        vram_source="nvidia-smi" if measured else "none",
        gpu_state=state,
        gpu_name="Nova GPU" if measured else None,
        gpus=[{"index": 0, "name": "Nova GPU", "vram_gib": vram_gib, "vram_source": "nvidia-smi"}] if measured else [],
        llmfit_crosscheck={"ram_physical": {"status": "absent"}, "vram": {"status": "absent"}},
    )
    return HardwareProfile.model_validate(payload)


def _package(need_gib: float, name: str) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["files"][0]["size_bytes"] = int(_weights_for(need_gib) * GIB)
    payload["files"][0]["name"] = f"Nova-7B-{name}.gguf"
    payload["quantization"] = name
    return Package.model_validate(payload)


def _unknown_base() -> BaseModelSpec:
    payload = copy.deepcopy(EXAMPLES["BaseModelSpec"])
    payload["architecture"] = {"source_repo": "acme/Nova-7B", "source_revision": None, "kind": "unknown"}
    return BaseModelSpec.model_validate(payload)


def _scenario() -> Scenario:
    return Scenario(context_requested=CONTEXT, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1)


# --- new: the computed reserve ------------------------------------------------------------------


def test_the_laptop_pool_is_memory_minus_the_computed_reserve():
    """127.70 - 16.0 = 111.70, off the card of 12 GiB (`cpu_gpu`)."""
    fit = compute_fit_v2(_profile(), _package(100.0, "Q8_0"), _unknown_base(), _scenario(), _machine())
    assert fit.mode == "cpu_gpu"
    assert (fit.pool_gib, fit.reserve_gib) == (pytest.approx(111.70), 16.0)


def test_a_need_of_112_7_on_the_laptop_is_no_fit():
    """112.7 / 111.70 = 1.009: `no_fit` (with the fixed 8.0 of 0.1.0 it was 0.942, `tight`)."""
    fit = fit_from_parameters(_parameters_for(112.7), GpuPool((12.0,)), LAPTOP_RAM, _machine(), CONTEXT)
    assert fit.need_gib == pytest.approx(112.7)
    assert (fit.fit_class, fit.mode) == ("no_fit", "cpu_gpu")


def test_a_small_machine_keeps_a_positive_pool():
    """3.81 GiB keeps 2.0: pool 1.81, a need of 1.5 is 0.83 of it -- `fits (RAM)`."""
    fit = fit_from_parameters(_parameters_for(1.5), GpuPool(()), 3.81, _machine(), CONTEXT)
    assert fit.pool_gib == pytest.approx(1.81)
    assert (fit.fit_class, fit.mode) == ("fits", "cpu")


def test_unified_memory_takes_the_share_and_the_card_reserve():
    """128 - 32 - 1 = 95.0, mode `gpu`: no upper bound on the share (GF question 4)."""
    fit = compute_fit_v2(_profile(128.0, state="unified_memory"), _package(50.0, "Q4_K_M"), _unknown_base(), _scenario(), _machine())
    assert (fit.mode, fit.pool_gib, fit.reserve_gib) == ("gpu", pytest.approx(95.0), 33.0)


def test_the_size_scale_and_the_ranking_use_the_same_reserve():
    """105 fits the pool of 111.70 (0.94, `tight`); 112.7 and 115 do not. Scale and ranking agree."""
    packages = [_package(105.0, "Q6_K"), _package(112.7, "Q8_0"), _package(115.0, "BF16")]
    base = _unknown_base()
    bases = {package.base_model_hf_repo: base for package in packages}
    profile, machine = _profile(), _machine()
    counted = count_fitting(profile, machine, packages, bases, CONTEXT)
    entries = [(package, compute_fit_v2(profile, package, base, _scenario(), machine)) for package in packages]
    assert counted == len(rank_packages(entries, [], _scenario()).ranked) == 1


def test_an_entered_reserve_of_zero_gives_the_whole_memory():
    fit = fit_from_parameters(_parameters_for(100.0), GpuPool((12.0,)), LAPTOP_RAM, _machine(0.0), CONTEXT)
    assert (fit.pool_gib, fit.reserve_gib) == (pytest.approx(LAPTOP_RAM), 0.0)


def test_a_size_fit_with_unknown_memory_still_fits_a_card_that_holds_it():
    fit = fit_from_parameters(_parameters_for(10.0), GpuPool((24.0,)), None, _machine(), CONTEXT)
    assert (fit.fit_class, fit.mode, fit.pool_gib) == ("fits", "gpu", pytest.approx(23.0))


def test_a_size_fit_with_unknown_memory_and_too_small_a_card_is_no_fit():
    """The card does not hold it; the system memory is unknown, so its pool is 0 - 0: `no_fit`."""
    fit = fit_from_parameters(_parameters_for(30.0), GpuPool((24.0,)), None, _machine(), CONTEXT)
    assert (fit.fit_class, fit.mode, fit.pool_gib) == ("no_fit", "cpu_gpu", 0.0)


def test_a_shared_memory_size_fit_with_unknown_memory_is_no_fit():
    """Unified memory with an unknown size: the pool is 0 minus both reserves, never a fit."""
    package = _package(5.0, "Q4_K_M")
    for machine, pool in ((_machine(), -1.0), (_machine(0.0, vram_reserve=0.0), 0.0)):
        by_parameters = fit_from_parameters(_parameters_for(5.0), 0.0, None, machine, CONTEXT, shared_memory=True)
        by_size = fit_from_size(package, 0.0, None, machine, CONTEXT, shared_memory=True)
        for fit in (by_parameters, by_size):
            assert (fit.fit_class, fit.mode, fit.pool_gib) == ("no_fit", "gpu", pytest.approx(pool))


# --- as before: an entered number computes as it did ----------------------------------------------


def test_an_explicit_reserve_of_8_computes_as_before():
    """Bestand: 127.70 - 8.0 = 119.70, the pool of every folder 0.1.0 wrote."""
    fit = fit_from_parameters(_parameters_for(112.7), GpuPool((12.0,)), LAPTOP_RAM, _machine(8.0), CONTEXT)
    assert fit.pool_gib == pytest.approx(119.70)
    assert fit.fit_class == "tight"


def test_a_profile_without_memory_stays_unknown_behind_the_gate():
    """Bestand: the gate of the profile comes first; the reserve is never reached."""
    fit = compute_fit_v2(_profile(None), _package(10.0, "Q4_K_M"), _unknown_base(), _scenario(), _machine(8.0))
    assert (fit.fit_class, fit.reason) == ("unknown", "physical RAM unknown")
