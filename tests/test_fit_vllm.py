"""Tests for modelroom.fit_vllm: how many requests of one context a vLLM server holds on one card.

The two constants come from recordings: 0.92 is vLLM 0.30.0's default share of the card, 1.50 GiB
the runtime of the A100 recording (`tests/fixtures/vllm_server_log_a100.txt`, decided 2026-09-27),
with the L4 recording as the second measured value. Nothing here reads a file of the package.
"""

from __future__ import annotations

import re

import pytest

from modelroom.contracts import Architecture
from modelroom.fit_vllm import (
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_RUNTIME_OVERHEAD_GIB,
    kv_per_token_bytes,
    largest_card,
    vllm_fit,
)

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
