"""Tests for modelroom.fit_vllm: how many requests of one context a vLLM server holds on one card.

The two constants come from recordings: 0.92 is vLLM 0.30.0's default share of the card, 1.50 GiB
the runtime of the A100 recording (`tests/fixtures/vllm_server_log_a100.txt`, decided 2026-09-27),
with the L4 recording as the second measured value. Nothing here reads a file of the package.
"""

from __future__ import annotations

import json
import re

import pytest

from modelroom.config import MachineConfig
from modelroom.contracts import Architecture, BaseModelSpec, Package, PackageFile
from modelroom.fit_vllm import (
    INCOMPLETE_PACKAGE_REASON,
    NO_CARD_REASON,
    NO_HEAD_COUNT_REASON,
    SHARED_MEMORY_REASON,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_RUNTIME_OVERHEAD_GIB,
    ZERO_WEIGHT_REASON,
    equal_size_group,
    kv_per_token_bytes,
    largest_card,
    tensor_parallel_size,
    vllm_fit,
    vllm_fit_as_fit,
)
from modelroom.measurements import Scenario
from modelroom.profile import HardwareProfile

from fixture_support import FIXTURES, v2_profile

GIB = 1024**3
QWEN25_WEIGHTS_BYTES = 988_097_824


def _log(name: str) -> dict[str, float]:
    """The four numbers of a recorded vLLM start the runtime is computed from."""
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return {
        "utilization": float(re.search(r"'gpu_memory_utilization': ([\d.]+)", text).group(1)),
        "checkpoint": float(re.search(r"Checkpoint size: ([\d.]+) GiB", text).group(1)),
        "kv_gib": float(re.search(r"Available KV cache memory: ([\d.]+) GiB", text).group(1)),
        "kv_tokens": float(re.search(r"GPU KV cache size: ([\d,]+) tokens", text).group(1).replace(",", "")),
    }


def _card_gib(name: str) -> float:
    """The first card of a recorded `nvidia-smi` answer, in GiB."""
    first = (FIXTURES / name).read_text(encoding="utf-8").splitlines()[0]
    return int(first.rsplit(",", 1)[1]) / 1024


def _runtime(log: str, csv: str) -> float:
    numbers = _log(log)
    return numbers["utilization"] * _card_gib(csv) - numbers["checkpoint"] - numbers["kv_gib"]


def _qwen25_architecture() -> Architecture:
    """24 layers, 2 KV heads, and a head width of 64 = 896 / 14 -- the test's own number only."""
    return Architecture(
        source_repo="Qwen/Qwen2.5-0.5B-Instruct",
        source_revision=None,
        kind="dense_classic",
        num_hidden_layers=24,
        num_key_value_heads=2,
        head_dim=896 // 14,
    )


def test_the_runtime_is_the_a100_recording():
    """`0.5 x 80 - 0.920 - 37.58 = 1.50`, one card of two used, utilization 0.5, the same model."""
    assert round(_runtime("vllm_server_log_a100.txt", "nvidia_smi_two_gpus.csv"), 2) == VLLM_RUNTIME_OVERHEAD_GIB == 1.50


def test_the_l4_recording_is_the_second_measured_value():
    """`0.5 x 22.494 - 0.920 - 9.02 = 1.31`: the constant is one recording, not a bound."""
    assert round(_runtime("vllm_server_log_l4.txt", "nvidia_smi_two_l4.csv"), 2) == 1.31


def test_the_share_of_the_card_is_the_vllm_default():
    assert VLLM_GPU_MEMORY_UTILIZATION == 0.92


@pytest.mark.parametrize("log", ["vllm_server_log_l4.txt", "vllm_server_log_a100.txt"])
def test_the_kv_cache_per_token_of_the_architecture_meets_the_logged_token_count(log):
    """9.02 GiB at 24 x 2 x 64 is within 0.1 % of the 787,840 tokens the L4 log names."""
    numbers = _log(log)

    per_token, basis = kv_per_token_bytes(_qwen25_architecture())
    tokens = numbers["kv_gib"] * GIB / per_token

    assert (per_token, basis) == (2 * 24 * 2 * 64 * 2, "architecture")
    assert tokens == pytest.approx(numbers["kv_tokens"], rel=0.001)


def test_one_l4_holds_thirty_two_requests_of_4096_from_size():
    fit = vllm_fit(_card_gib("nvidia_smi_two_l4.csv"), QWEN25_WEIGHTS_BYTES, context=4096, requests=20)

    assert fit.weights_gib == pytest.approx(0.92023, abs=1e-5)
    assert round(fit.kv_gib, 2) == 18.27
    assert fit.kv_tokens == pytest.approx(133_070, abs=1)  # 133,069.6, floored
    assert (fit.at_once, fit.basis, fit.fit_class) == (32, "size", "fits")


def test_an_unknown_architecture_takes_the_size_basis():
    unknown = Architecture(source_repo="a/b", source_revision=None, kind="unknown")

    assert kv_per_token_bytes(unknown) == (147_456, "size")
    assert kv_per_token_bytes(None) == (147_456, "size")


@pytest.mark.parametrize(
    ("card_gib", "requests", "architecture", "expected"),
    [
        (3.0, 1, None, "no_fit"),
        (22.494, 40, None, "tight"),
        (22.494, 16, None, "fits"),
        (22.494, 20, "qwen25", "ample"),
        (22.494, 250, "qwen25", "fits"),
        (22.494, 2000, "qwen25", "tight"),
    ],
)
def test_the_class_is_the_share_of_the_requests_the_card_holds(card_gib, requests, architecture, expected):
    """`no_fit` below one, `tight` below the requests, `ample` at twice on the architecture basis only."""
    arch = _qwen25_architecture() if architecture else None

    fit = vllm_fit(card_gib, QWEN25_WEIGHTS_BYTES, context=4096, requests=requests, architecture=arch)

    assert fit.fit_class == expected


def test_the_largest_card_of_a_profile_is_the_one_computed():
    profile = v2_profile()

    assert largest_card(profile) == max(gpu.vram_gib for gpu in profile.gpus)


def test_a_profile_without_a_card_has_none():
    assert largest_card(v2_profile(vram_gib=0)) is None


def _qwen35_architecture() -> Architecture:
    import json

    from modelroom.architecture import architecture_from_hf_config

    config = json.loads((FIXTURES / "hf_qwen_qwen35_9b_config.json").read_text(encoding="utf-8"))
    return architecture_from_hf_config("Qwen/Qwen3.5-4B", None, config)


def test_at_once_counts_the_state_once_per_request():
    """AP9-UX2 P1: `at_once = floor(kv_bytes / (32_768 x context + 52_690_944))`, exact at the edge."""
    from modelroom.fit_vllm import requests_at_once

    per_request = 32_768 * 32_768 + 52_690_944
    for n in (1, 7, 40):
        assert requests_at_once(n * per_request, 32_768, 32_768, 52_690_944) == n
        assert requests_at_once(n * per_request - 1, 32_768, 32_768, 52_690_944) == n - 1

    architecture = _qwen35_architecture()
    assert kv_per_token_bytes(architecture) == (32_768, "architecture")
    fit = vllm_fit(22.494, 9_319_828_096, context=32_768, requests=1, architecture=architecture)
    kv_bytes = int(fit.kv_gib * GIB)
    assert fit.basis == "architecture"
    assert fit.at_once == kv_bytes // per_request
    assert fit.kv_tokens == kv_bytes // 32_768


# --- AP9-SV2-F: tensor parallel over several cards, as a `Fit` (`vllm_fit_as_fit`) ----------------


def _profile_from(name: str) -> HardwareProfile:
    data = json.loads((FIXTURES / "profiles_v4" / name).read_text(encoding="utf-8"))
    return HardwareProfile.model_validate(data)


def _architecture(num_attention_heads: int | None, **overrides) -> Architecture:
    fields = dict(
        source_repo="Qwen/Qwen3-8B",
        source_revision=None,
        kind="dense_classic",
        num_hidden_layers=36,
        num_key_value_heads=8,
        head_dim=128,
        num_attention_heads=num_attention_heads,
        max_context=131072,
    )
    fields.update(overrides)
    return Architecture(**fields)


def _base_model(architecture: Architecture) -> BaseModelSpec:
    return BaseModelSpec(
        hf_repo="Qwen/Qwen3-8B", repo_aliases=[], ollama_base=None, ollama_tag=None,
        publisher="Qwen", parameters_b=8.0, architecture=architecture,
    )


def _tensor_package(*, weights_bytes: int = 6_098_581_864, complete: bool = True, zero_weight: bool = False) -> Package:
    size = 0 if zero_weight else weights_bytes
    return Package(
        source="huggingface", repo="Qwen/Qwen3-8B-AWQ", revision="a" * 40,
        base_model_hf_repo="Qwen/Qwen3-8B", format="tensor",
        files=[PackageFile(name="model.safetensors", role="weights", size_bytes=size, digest=None)] if complete or zero_weight else [],
        complete=complete, quantization="AWQ", quantization_source="config", default_context=None,
        provenance="from_metadata", unresolved_reason=None, approval=None,
        observed_at="2026-09-22T09:00:00Z", last_seen="2026-09-22T09:00:00Z", active=True,
    )


_MACHINE = MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)
_SCENARIO_10 = Scenario(context_requested=32768, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=10)


def test_t_is_the_largest_power_of_two_dividing_the_heads():
    """4 cards, 14 heads: 4 does not divide 14, 2 does -- `t = 2` (the Qwen2.5-0.5B example of F1)."""
    assert tensor_parallel_size(4, 14) == (2, None)


def test_the_largest_power_of_two_up_to_eight_wins_when_it_divides():
    """4 cards, 32 heads: `t = 4` (the Qwen3-8B example of F1), never above the card count."""
    assert tensor_parallel_size(4, 32) == (4, None)
    assert tensor_parallel_size(16, 32) == (8, None)  # the ceiling of `TENSOR_PARALLEL_SIZES` is 8


def test_no_head_count_computes_on_one_card():
    assert tensor_parallel_size(4, None) == (1, NO_HEAD_COUNT_REASON)


def test_three_equal_cards_and_12_heads_give_t_2():
    """3 is no power of two; the largest one at or below it is 2, and 12 divides by 2."""
    assert tensor_parallel_size(3, 12) == (2, None)


def test_unequal_cards_use_the_group_of_the_largest():
    """`mixed_cards_server.json`: three equal 22.49 GiB L4 cards and one smaller 11 GiB T4 -- the
    group is the three equal ones, in profile order; the smaller card is left out, not mixed in."""
    profile = _profile_from("mixed_cards_server.json")

    group = equal_size_group(profile)

    assert [card.vram_gib for card in group.cards] == [22.49, 22.49, 22.49]
    assert group.left_out == 1


def test_no_cards_at_all_has_an_empty_group():
    group = equal_size_group(v2_profile(vram_gib=0))
    assert group.cards == () and group.left_out == 0


def test_the_tensor_parallel_worked_example_t4_and_t1():
    """The brief's own rechenbeispiel (F2): 4 x 22.49 GiB L4, `Qwen/Qwen3-8B-AWQ` (36 layers, 8 KV
    heads, head_dim 128), context 32768, 10 requests. `t = 4` (32 attention heads): 15 requests at
    once, `fits`. For comparison, forcing `t = 1` (no head count): 3 at once, `tight`.

    `kv_gib`/`need_gib` (Codex r1-1): the KV need is the whole model's, over every one of the `t`
    cards it is spread over -- 36864 B/token (147456 / 4) x 32768 x 10 requests, **times the 4
    cards**: (36864 x 32768 x 10 x 4) / GiB = 45.00 GiB, not the 11.25 GiB of one card alone.
    `need_gib` = weights (5.68) + 4 x 1.50 (6.00) + 45.00 = 56.68 GiB.
    """
    profile = _profile_from("four_l4_server_vllm.json")
    package = _tensor_package()

    with_t4 = vllm_fit_as_fit(profile, package, _base_model(_architecture(32)), _SCENARIO_10, _MACHINE)
    assert (with_t4.cards, with_t4.at_once, with_t4.fit_class, with_t4.mode) == (4, 15, "fits", "gpu_split")
    assert with_t4.weights_gib == pytest.approx(5.679746963083744, abs=1e-9)
    assert with_t4.kv_gib == pytest.approx(45.00, abs=5e-3)
    assert with_t4.need_gib == pytest.approx(56.68, abs=5e-3)
    assert with_t4.reason is None  # a real head count divides the card count: no excuse to give

    with_t1 = vllm_fit_as_fit(profile, package, _base_model(_architecture(None)), _SCENARIO_10, _MACHINE)
    assert (with_t1.cards, with_t1.at_once, with_t1.fit_class, with_t1.mode) == (1, 3, "tight", "gpu")
    # Codex r1-2: the missing head count is not a guard failure (the fit still computes), but a
    # reader must still be told why `t` stopped at one card.
    assert with_t1.reason == NO_HEAD_COUNT_REASON


def test_fewer_kv_heads_than_cards_replicates():
    """`t` above the KV head count does not split the KV cache any further than the heads allow
    (Annahme A2): `t = 4` against 2 KV heads reads the same per-token cost as `t = 2` would."""
    profile = _profile_from("four_l4_server_vllm.json")
    package = _tensor_package()
    few_kv_heads = _architecture(4, num_key_value_heads=2)  # 4 divides 4 attention heads: t = 4

    fit = vllm_fit_as_fit(profile, package, _base_model(few_kv_heads), _SCENARIO_10, _MACHINE)

    assert fit.cards == 4
    two_card_equivalent = _architecture(2, num_key_value_heads=2)
    as_if_t2 = vllm_fit_as_fit(profile, package, _base_model(two_card_equivalent), _SCENARIO_10, _MACHINE)
    # The per-card KV cost is the same at `t = 4` as at `t = 2`: the divisor is `min(t,
    # num_key_value_heads)`, capped at the head count, so a `t` above it replicates rather than
    # splitting any further (Assumption A2). `Fit.kv_gib` is the whole model's KV need, that
    # per-card cost times every card the package is spread over (Codex r1-1) -- twice as much at
    # `t = 4` as at `t = 2`, even though each card alone holds the same share.
    assert fit.kv_gib == as_if_t2.kv_gib * 2
    # `t = 4` still spreads the weights and the runtime overhead over all 4 cards, so its pool and
    # its `at_once` are not the same as `t = 2`'s.
    assert fit.pool_gib == as_if_t2.pool_gib * 2
    assert fit.at_once > as_if_t2.at_once


def test_no_card_is_not_covered():
    """F1: a machine with no graphics card at all gets no vLLM fit -- `unknown`, never `no_fit`."""
    profile = v2_profile(vram_gib=0)
    fit = vllm_fit_as_fit(profile, _tensor_package(), _base_model(_architecture(32)), _SCENARIO_10, _MACHINE)

    assert (fit.fit_class, fit.reason) == ("unknown", NO_CARD_REASON)


def test_shared_memory_is_not_covered():
    """F1: unified memory (Apple) never runs vLLM."""
    from modelroom.profile import CrossCheck, GpuAdapter, LlmfitCrosscheck, PROFILE_SCHEMA_VERSION

    absent = CrossCheck(status="absent")
    profile = HardwareProfile(
        schema_version=PROFILE_SCHEMA_VERSION, profile_id="1" * 16, display_name="mac",
        os_fingerprint="2" * 16, os_fingerprint_source="macos_platform_uuid", origin="measured",
        recorded_at="2026-09-29T09:00:00Z", ram_physical_gib=64.0, ram_physical_source="os",
        ram_limit_gib=None, ram_limit_scope="none", vram_gib=64.0, vram_source="sysctl",
        gpu_state="unified_memory", gpu_name=None, gpus=[], machine_class="unknown",
        machine_class_source="unknown", llmfit_crosscheck=LlmfitCrosscheck(ram_physical=absent, vram=absent),
        llmfit_version=None,
    )

    fit = vllm_fit_as_fit(profile, _tensor_package(), _base_model(_architecture(32)), _SCENARIO_10, _MACHINE)

    assert (fit.fit_class, fit.reason) == ("unknown", SHARED_MEMORY_REASON)


def test_a_card_too_small_is_no_fit():
    """A package whose weights alone exceed what the budget leaves after the runtime overhead is
    `no_fit`, in the no-fit list like any other -- never `unknown`."""
    profile = _profile_from("four_l4_server_vllm.json")
    huge = _tensor_package(weights_bytes=90 * GIB)

    fit = vllm_fit_as_fit(profile, huge, _base_model(_architecture(32)), _SCENARIO_10, _MACHINE)

    assert fit.fit_class == "no_fit"


def test_an_incomplete_tensor_package_is_unknown():
    package = _tensor_package(complete=False)
    profile = _profile_from("four_l4_server_vllm.json")

    fit = vllm_fit_as_fit(profile, package, _base_model(_architecture(32)), _SCENARIO_10, _MACHINE)

    assert (fit.fit_class, fit.reason) == ("unknown", INCOMPLETE_PACKAGE_REASON)


def test_a_zero_weight_file_is_unknown_not_fit():
    package = _tensor_package(zero_weight=True)
    profile = _profile_from("four_l4_server_vllm.json")

    fit = vllm_fit_as_fit(profile, package, _base_model(_architecture(32)), _SCENARIO_10, _MACHINE)

    assert (fit.fit_class, fit.reason) == ("unknown", ZERO_WEIGHT_REASON)


def test_a_context_beyond_the_known_window_is_unknown():
    profile = _profile_from("four_l4_server_vllm.json")
    narrow = _base_model(_architecture(32, max_context=16384))
    scenario = Scenario(context_requested=32768, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=10)

    fit = vllm_fit_as_fit(profile, _tensor_package(), narrow, scenario, _MACHINE)

    assert fit.fit_class == "unknown" and fit.reason == "context beyond the window of this model (16k)"


def test_a_blocked_profile_is_unknown_the_same_way_ollamas_fit_reads_it():
    """vLLM's own guard reads `profile.fit_block_reason` first, the same check `compute_fit_v2`
    makes -- a profile this fit may not compute for at all reads the same for both runtimes."""
    from modelroom.profile import fit_block_reason

    profile = v2_profile(vram_gib=0).model_copy(update={"gpu_state": "present_unmeasured"})
    fit = vllm_fit_as_fit(profile, _tensor_package(), _base_model(_architecture(32)), _SCENARIO_10, _MACHINE)

    assert fit.fit_class == "unknown" and fit.reason == fit_block_reason(profile)
