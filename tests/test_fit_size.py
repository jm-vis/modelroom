"""Tests for the fit from the size (basis `size`), in `modelroom/fit.py`.

Two entry points, both pure and both computing fit contract v1's formula with a KV cache read
from a constant instead of from an architecture: `compute_fit_v2` falls back to it for a package
whose architecture fit v1 cannot read, and `fit_from_parameters` answers before any package has
been fetched, from a parameter count alone. Every number below is written out so it can be
recomputed by hand (CONTRACTS.md, "Fit from size (basis `size`)").
"""

from __future__ import annotations

import copy

import pytest

from modelroom.config import MachineConfig
from modelroom.contracts import BaseModelSpec, Package
from modelroom.examples import EXAMPLES
from modelroom.fit import (
    BYTES_PER_PARAMETER_Q4,
    GIB,
    KV_PER_TOKEN_SIZE_BASIS,
    compute_fit_v2,
    fit_from_parameters,
)
from modelroom.measurements import Scenario, default_scenario
from modelroom.profile import HardwareProfile

MACHINE = MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)
# 8192 tokens x 147,456 bytes per token = 1.125 GiB of KV cache on the size basis.
KV_AT_8192 = 8192 * KV_PER_TOKEN_SIZE_BASIS / GIB


def _profile(**changes) -> HardwareProfile:
    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    payload.update(changes)
    return HardwareProfile.model_validate(payload)


def _package(weights_gib: float = 4.0) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["files"][0]["size_bytes"] = int(weights_gib * GIB)
    return Package.model_validate(payload)


def _base_model(kind: str = "dense_classic", **architecture) -> BaseModelSpec:
    if kind == "unknown":
        spec = {"source_repo": "acme/Nova-7B", "source_revision": None, "kind": "unknown"}
    else:
        spec = dict(copy.deepcopy(EXAMPLES["Architecture"]), **architecture)
    payload = copy.deepcopy(EXAMPLES["BaseModelSpec"])
    payload["architecture"] = spec
    return BaseModelSpec.model_validate(payload)


def _scenario(context: int = 8192, requests: int = 1) -> Scenario:
    return Scenario(
        context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=requests
    )


# --- after the fetch: a package whose architecture fit v1 cannot read ---------------------------


def test_an_architecture_v1_cannot_read_is_computed_from_the_size():
    """Weights 3.0 GiB x 1.10 + 1.125 KV + 0.50 = 4.925 GiB against a pool of 7.0 (ratio 0.70)."""
    fit = compute_fit_v2(_profile(), _package(3.0), _base_model("unknown"), default_scenario(), MACHINE)

    assert (fit.basis, fit.reason) == ("size", None)
    assert fit.kv_gib == pytest.approx(KV_AT_8192)
    assert fit.kv_gib == pytest.approx(1.125)
    assert fit.need_gib == pytest.approx(3.0 * 1.10 + 1.125 + 0.50)
    assert fit.pool_gib == pytest.approx(7.0)
    assert (fit.fit_class, fit.mode) == ("good", "gpu")


def test_the_size_basis_never_reaches_perfect():
    """`perfect` stays reserved for a package a read architecture proves comfortable."""
    fit = compute_fit_v2(_profile(), _package(1.0), _base_model("unknown"), default_scenario(), MACHINE)

    assert fit.need_gib / fit.pool_gib < 0.60  # this ratio is `perfect` on the architecture basis
    assert (fit.fit_class, fit.basis) == ("good", "size")


def test_the_architecture_basis_is_the_default_and_says_so():
    fit = compute_fit_v2(_profile(), _package(), _base_model(), default_scenario(), MACHINE)

    assert fit.basis == "architecture"


def test_the_size_basis_is_one_assumption_and_no_bound_in_either_direction():
    """36 layers x 8 KV heads x 128 wide: more than a 32-layer model needs, less than a 42-layer one.

    The claim that it is an upper bound was wrong and is corrected here (second-model round,
    2026-09-24): it is one fixed number, which is why the class is capped at `good`.
    """
    package = _package()
    smaller = _base_model(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)
    larger = _base_model(num_hidden_layers=42, num_key_value_heads=8, head_dim=128)

    from_size = compute_fit_v2(_profile(), package, _base_model("unknown"), default_scenario(), MACHINE)
    over = compute_fit_v2(_profile(), package, smaller, default_scenario(), MACHINE)
    under = compute_fit_v2(_profile(), package, larger, default_scenario(), MACHINE)

    assert from_size.kv_gib > over.kv_gib
    assert from_size.kv_gib < under.kv_gib


def test_the_size_basis_stays_within_a_quarter_of_a_real_9b_architecture():
    """Plausibility, with the numbers written out.

    A dense 9B is 42 layers, 8 KV heads, 128 wide: 2 x 42 x 8 x 128 x 2 bytes = 172,032 bytes
    of KV cache per token, against the 147,456 of the size basis -- 14.3 % below it. For the
    5.29 GiB `Q4_K_M` package of the fixtures (5,680,522,464 bytes) the need differs by 2.4 %:
    7.44 GiB on the size basis against 7.63 GiB on the architecture basis. At the largest context
    of the scale (131072) the same pair is 24.32 against 27.32 GiB, 11.0 % apart.
    """
    weights_gib = 5_680_522_464 / GIB  # tests/fixtures, the 9B `Q4_K_M` build
    package = _package(weights_gib)
    dense_9b = _base_model(num_hidden_layers=42, num_key_value_heads=8, head_dim=128)

    for context in (8192, 131072):
        from_size = compute_fit_v2(_profile(), package, _base_model("unknown"), _scenario(context), MACHINE)
        from_architecture = compute_fit_v2(_profile(), package, dense_9b, _scenario(context), MACHINE)

        assert from_architecture.basis == "architecture"
        deviation = abs(from_size.need_gib - from_architecture.need_gib) / from_architecture.need_gib
        assert deviation < 0.25, f"context {context}: {from_size.need_gib} against {from_architecture.need_gib}"
        assert from_size.need_gib < from_architecture.need_gib  # below, not above: no upper bound


def test_the_other_package_rules_stay_unknown_without_an_architecture():
    """No complete package and a weight file without a size are no size to compute with."""
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["files"][0]["size_bytes"] = 0
    no_size = Package.model_validate(payload)
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["files"] = []
    payload["complete"] = False
    payload["quantization"] = None
    incomplete = Package.model_validate(payload)

    for package in (no_size, incomplete):
        fit = compute_fit_v2(_profile(), package, _base_model("unknown"), default_scenario(), MACHINE)
        assert fit.fit_class == "unknown"
        assert fit.basis == "architecture"
        assert fit.reason is not None


def test_a_blocked_profile_is_unknown_before_any_size_is_read():
    fit = compute_fit_v2(
        _profile(gpu_state="present_unmeasured"), _package(), _base_model("unknown"), default_scenario(), MACHINE
    )

    assert (fit.fit_class, fit.basis) == ("unknown", "architecture")


def test_more_than_one_request_is_unknown_on_the_size_basis_as_well():
    fit = compute_fit_v2(_profile(), _package(), _base_model("unknown"), _scenario(requests=2), MACHINE)

    assert fit.fit_class == "unknown"
    assert "one request" in str(fit.reason)


def test_the_context_of_the_scenario_is_the_one_the_kv_cache_is_computed_for():
    fit = compute_fit_v2(_profile(), _package(2.0), _base_model("unknown"), _scenario(32768), MACHINE)

    assert fit.context == 32768
    assert fit.kv_gib == pytest.approx(32768 * KV_PER_TOKEN_SIZE_BASIS / GIB)


# --- before the fetch: a parameter count alone ---------------------------------------------------


def test_nine_billion_parameters_are_about_five_gibibytes_of_weights():
    """`BYTES_PER_PARAMETER_Q4` is 0.6: 9e9 x 0.6 / 1024**3 = 5.03 GiB of weights."""
    fit = fit_from_parameters(9.0, 8.0, 31.7, MACHINE, 8192)

    assert fit.weights_gib == pytest.approx(9e9 * BYTES_PER_PARAMETER_Q4 / GIB)
    assert fit.weights_gib == pytest.approx(5.03, abs=0.01)
    assert fit.need_gib == pytest.approx(fit.weights_gib * 1.10 + 1.125 + 0.50)
    assert (fit.basis, fit.reason, fit.context) == ("size", None, 8192)
    assert fit.fit_class != "perfect"


def test_a_model_too_large_for_both_pools_is_too_tight():
    fit = fit_from_parameters(400.0, 8.0, 31.7, MACHINE, 8192)

    assert fit.fit_class == "too_tight"
    assert fit.mode == "cpu_gpu"


def test_a_machine_without_a_graphics_card_computes_against_its_memory():
    fit = fit_from_parameters(9.0, 0.0, 31.7, MACHINE, 32768)

    assert fit.mode == "cpu"
    assert fit.kv_gib == pytest.approx(32768 * KV_PER_TOKEN_SIZE_BASIS / GIB)
    assert fit.fit_class == "good"
