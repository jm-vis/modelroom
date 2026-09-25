"""Fit with parallel requests and with a machine entered by hand (decided 2026-09-25).

`requests` are the parallel slots the daemon is assumed to keep (`OLLAMA_NUM_PARALLEL`): the
weights are loaded once, the KV cache once per slot, so `kv_gib = context x kv_per_token x
requests` on both bases. A graphics card entered by hand counts like a measured one; unified
memory is one pool, the RAM minus both reserves, in mode `gpu`, with no fallback onto the same
RAM (CONTRACTS.md, "Fit contract v1", "Fit with profile v2").

The example architecture needs exactly 1 GiB of KV cache per request at context 8192, so every
number here can be checked by hand: need = weights x 1.10 + kv + 0.50.
"""

from __future__ import annotations

import copy

import pytest

import modelroom.fit as fit_module
import modelroom.guided_search as guided_search_module
from modelroom.config import MachineConfig
from modelroom.contracts import BaseModelSpec, Fit, Package
from modelroom.examples import EXAMPLES
from modelroom.fit import (
    GIB,
    KV_PER_TOKEN_SIZE_BASIS,
    compute_fit_v2,
    count_fitting,
    fit_from_parameters,
    fit_from_size,
    unknown_fit,
)
from modelroom.guided_context import Checked
from modelroom.guided_contracts import SearchHit
from modelroom.guided_models import model_choices, model_fit
from modelroom.measurements import Scenario
from modelroom.profile import HardwareProfile

from test_guided import _run as guided_run

MACHINE =MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)
ABSENT = {"status": "absent"}
KV_AT_8192_SIZE = 8192 * KV_PER_TOKEN_SIZE_BASIS / GIB


def _package(weights_gib: float = 4.0) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["files"][0]["size_bytes"] = int(weights_gib * GIB)
    return Package.model_validate(payload)


def _base_model(kind: str = "dense_classic") -> BaseModelSpec:
    payload = copy.deepcopy(EXAMPLES["BaseModelSpec"])
    if kind == "unknown":
        payload["architecture"] = {"source_repo": "acme/Nova-7B", "source_revision": None, "kind": "unknown"}
    return BaseModelSpec.model_validate(payload)


def _scenario(requests: int = 1, context: int = 8192) -> Scenario:
    return Scenario(
        context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=requests
    )


def _measured() -> HardwareProfile:
    return HardwareProfile.model_validate(EXAMPLES["HardwareProfile"])


def _hand(ram: float = 32.0, vram: float = 0.0, state: str = "unified_memory") -> HardwareProfile:
    data = copy.deepcopy(EXAMPLES["HardwareProfile"])
    data.update(
        display_name="office box",
        os_fingerprint="none",
        os_fingerprint_source="none",
        origin="entered",
        ram_physical_gib=ram,
        ram_physical_source="entered",
        vram_gib=vram,
        vram_source="entered" if state == "entered" else "none",
        gpu_state=state,
        gpu_name=None,
        llmfit_crosscheck={"ram_physical": ABSENT, "vram": ABSENT},
        llmfit_version=None,
    )
    return HardwareProfile.model_validate(data)


# --- the field -----------------------------------------------------------------------------------


def test_fit_requests_defaults_to_one():
    payload = {key: value for key, value in EXAMPLES["Fit"].items() if key != "requests"}
    assert Fit.model_validate(payload).requests == 1
    assert unknown_fit("no reason").requests == 1


@pytest.mark.parametrize("requests", [0, 1025])
def test_fit_requests_stay_between_1_and_1024(requests):
    with pytest.raises(ValueError):
        Fit.model_validate({**EXAMPLES["Fit"], "requests": requests})


def test_the_ollama_hint_is_a_rule_of_thumb_at_8():
    assert fit_module.OLLAMA_REQUEST_HINT == 8
    assert fit_module.OLLAMA_REQUEST_HINT_TEXT == (
        "beyond 8 requests at once this fit says nothing about throughput: measure under load, "
        "or look at a serving stack (rule of thumb, not measured)"
    )


# --- the KV cache per request ----------------------------------------------------------------------


def test_more_requests_multiply_the_kv_cache_on_the_architecture_basis():
    one = compute_fit_v2(_measured(), _package(), _base_model(), _scenario(1), MACHINE)
    three = compute_fit_v2(_measured(), _package(), _base_model(), _scenario(3), MACHINE)

    assert (one.kv_gib, three.kv_gib) == (pytest.approx(1.0), pytest.approx(3.0))
    assert three.need_gib == pytest.approx(4.0 * 1.10 + 3.0 + 0.50)
    assert (one.requests, three.requests) == (1, 3)
    assert three.fit_class != "unknown" and three.reason is None


def test_more_requests_multiply_the_kv_cache_on_the_size_basis():
    fit = compute_fit_v2(_measured(), _package(), _base_model("unknown"), _scenario(2), MACHINE)

    assert fit.basis == "size" and fit.requests == 2
    assert fit.kv_gib == pytest.approx(2 * KV_AT_8192_SIZE)
    assert fit.fit_class in ("good", "marginal", "too_tight")


def test_the_size_basis_never_reaches_perfect_with_one_request_either():
    fit = fit_from_size(_package(1.0), 0.0, 128.0, MACHINE, 8192)
    assert (fit.fit_class, fit.requests) == ("good", 1)


def test_fit_from_parameters_takes_the_requests():
    one = fit_from_parameters(7.0, 8.0, 31.7, MACHINE, 8192)
    four = fit_from_parameters(7.0, 8.0, 31.7, MACHINE, 8192, requests=4)
    assert four.requests == 4
    assert four.kv_gib == pytest.approx(4 * one.kv_gib)


def test_count_fitting_counts_for_the_requests_it_is_given():
    package = _package(16.0)  # 17.6 + 0.5 + 1 GiB per request against 23.7 GiB of RAM
    base_models = {package.base_model_hf_repo: _base_model()}
    assert count_fitting(_measured(), MACHINE, [package], base_models, 8192) == 1
    assert count_fitting(_measured(), MACHINE, [package], base_models, 8192, requests=4) == 1
    assert count_fitting(_measured(), MACHINE, [package], base_models, 8192, requests=8) == 0


# --- a graphics card entered by hand ---------------------------------------------------------------


def test_an_entered_graphics_card_counts_like_a_measured_one():
    fit = compute_fit_v2(_hand(vram=12.0, state="entered"), _package(), _base_model(), _scenario(), MACHINE)
    assert (fit.mode, fit.pool_gib, fit.reserve_gib) == ("gpu", pytest.approx(11.0), 1.0)


def test_no_graphics_card_entered_by_hand_computes_in_system_memory():
    fit = compute_fit_v2(_hand(state="none"), _package(), _base_model(), _scenario(), MACHINE)
    assert (fit.mode, fit.pool_gib) == ("cpu", pytest.approx(24.0))


# --- unified memory ----------------------------------------------------------------------------------


def test_unified_memory_is_one_pool_of_ram_minus_both_reserves_in_mode_gpu():
    fit = compute_fit_v2(_hand(ram=32.0), _package(), _base_model(), _scenario(), MACHINE)

    assert fit.mode == "gpu"
    assert fit.pool_gib == pytest.approx(23.0)
    assert fit.reserve_gib == pytest.approx(9.0)
    assert fit.fit_class == "perfect"  # 5.9 of 23 GiB, read from the architecture


def test_unified_memory_on_the_size_basis_stays_at_good():
    fit = compute_fit_v2(_hand(ram=32.0), _package(), _base_model("unknown"), _scenario(), MACHINE)
    assert (fit.mode, fit.fit_class, fit.basis) == ("gpu", "good", "size")


def test_unified_memory_has_no_fallback_onto_the_same_ram():
    fit = compute_fit_v2(_hand(ram=32.0), _package(20.0), _base_model(), _scenario(), MACHINE)
    assert (fit.mode, fit.fit_class, fit.pool_gib) == ("gpu", "too_tight", pytest.approx(23.0))


def test_unified_memory_below_both_reserves_is_too_tight():
    fit = compute_fit_v2(_hand(ram=8.5), _package(0.5), _base_model(), _scenario(), MACHINE)
    assert fit.pool_gib == pytest.approx(-0.5)
    assert fit.fit_class == "too_tight"


def test_unified_memory_multiplies_the_kv_cache_too():
    fit = compute_fit_v2(_hand(ram=32.0), _package(), _base_model(), _scenario(8), MACHINE)
    assert fit.kv_gib == pytest.approx(8.0) and fit.requests == 8


# --- the preview of the guided list ------------------------------------------------------------------


def _checked(profile: HardwareProfile) -> Checked:
    return Checked(name="office-box", profile=profile, machine_config=MACHINE)


def test_the_preview_computes_unified_memory_as_one_pool_in_mode_gpu():
    fit = model_fit(7.0, _checked(_hand(ram=32.0)), 8192)
    assert fit.mode == "gpu"
    assert fit.pool_gib == pytest.approx(23.0)
    assert fit.reserve_gib == pytest.approx(9.0)


def test_the_preview_counts_an_entered_graphics_card():
    fit = model_fit(7.0, _checked(_hand(vram=12.0, state="entered")), 8192)
    assert (fit.mode, fit.pool_gib) == ("gpu", pytest.approx(11.0))


def test_the_preview_takes_the_requests_and_defaults_to_one():
    checked = _checked(_measured())
    assert model_fit(7.0, checked, 8192).requests == 1
    assert model_fit(7.0, checked, 8192, requests=3).requests == 3


def test_the_search_step_hands_the_configured_requests_to_the_list(tmp_path, monkeypatch):
    """`guided_search` passes `[guided].requests` -- one request by default -- to `model_choices`."""
    seen: list[dict] = []
    real = guided_search_module.model_choices

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(guided_search_module, "model_choices", spy)
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()

    assert guided_run(tmp_path)[0] == 0

    assert seen and all(kwargs.get("requests") == 1 for kwargs in seen)


def test_model_choices_hand_the_requests_to_every_row():
    hits = [SearchHit.model_validate(EXAMPLES["ModelChoice"]["repos"][0])]
    rows = model_choices(hits, filtered=False, checked=_checked(_measured()), context=8192, requests=2)
    assert rows and all(row.fit.requests == 2 for row in rows)
