"""How many requests of one context a vLLM server holds on one card: computed, never a `Fit`.

vLLM takes a share of the card (`--gpu-memory-utilization`, 0.92 by default in 0.30.0,
docs.vllm.ai/en/v0.30.0/configuration/engine_args), loads the weights, keeps some memory for its
runtime, and gives the rest to the KV cache. So:

    budget  = 0.92 x card_gib
    kv_gib  = budget - weights - 1.50
    at_once = floor(kv_bytes / (per_token x context + state))

`weights` is the sum of the safetensors files, with no overhead factor. 1.50 GiB is the runtime of
the A100 recording (`0.5 x 80 - 0.920 - 37.58`, decided 2026-09-27); the L4 recording gave 1.31, the
second measured value. `per_token` is the base model's architecture where it can be read (only the
full-attention layers of a hybrid model), else the size basis of `fit.py` -- and then the class is
capped at `fits`, as there. `state` is a hybrid model's linear-layer state per request, `0`
otherwise (`architecture.state_bytes`, AP9-UX2 P1).

The class says how the requests the ranking assumes stand against what the card holds: `no_fit`
below one request, `tight` below the requests, `ample` at twice the requests or more on the
architecture basis, `fits` otherwise. The largest card of the machine is computed; tensor parallel
over several cards is not (CONTRACTS.md, "Local servers and the vLLM load test", ADR 0008).
`fit.py` is unchanged, and nothing here is persisted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .architecture import kv_token_bytes, state_bytes
from .contracts import Architecture
from .fit import GIB, KV_PER_TOKEN_SIZE_BASIS
from .profile import HardwareProfile

VLLM_GPU_MEMORY_UTILIZATION = 0.92
VLLM_RUNTIME_OVERHEAD_GIB = 1.50


@dataclass(frozen=True)
class VllmFit:
    """One computed vLLM fit: the card, its budget, the three parts of it, and the requests it holds."""

    card_gib: float
    budget_gib: float
    weights_gib: float
    runtime_gib: float
    kv_gib: float
    kv_tokens: int
    at_once: int
    context: int
    requests: int
    basis: str
    fit_class: str


def kv_per_token_bytes(architecture: Architecture | None) -> tuple[int, str]:
    """The 16-bit KV cache of one token in bytes, and its basis: `architecture`, else `size`.

    A `hybrid_linear` model counts only its layers of full attention (`architecture.kv_token_bytes`).
    """
    per_token = kv_token_bytes(architecture) if architecture is not None else None
    if per_token is not None:
        return per_token, "architecture"
    return KV_PER_TOKEN_SIZE_BASIS, "size"


def requests_at_once(kv_bytes: int, per_token: int, context: int, state: int) -> int:
    """How many requests of `context` tokens `kv_bytes` hold: each needs its tokens and its state once."""
    return kv_bytes // (per_token * context + state)


def vllm_fit(
    card_gib: float, weights_bytes: int, *, context: int, requests: int, architecture: Architecture | None = None
) -> VllmFit:
    """The fit of one served model on one card of `card_gib`, for `requests` of `context` tokens.

    A hybrid model's linear-layer state (`architecture.state_bytes`) is counted once per request.
    """
    per_token, basis = kv_per_token_bytes(architecture)
    state = state_bytes(architecture) if basis == "architecture" else 0
    budget = VLLM_GPU_MEMORY_UTILIZATION * card_gib
    weights = weights_bytes / GIB
    kv_gib = budget - weights - VLLM_RUNTIME_OVERHEAD_GIB
    kv_bytes = math.floor(max(kv_gib, 0.0) * GIB)
    at_once = requests_at_once(kv_bytes, per_token, context, state)
    return VllmFit(
        card_gib=card_gib,
        budget_gib=budget,
        weights_gib=weights,
        runtime_gib=VLLM_RUNTIME_OVERHEAD_GIB,
        kv_gib=kv_gib,
        kv_tokens=kv_bytes // per_token,
        at_once=at_once,
        context=context,
        requests=requests,
        basis=basis,
        fit_class=vllm_fit_class(at_once, requests, basis),
    )


def vllm_fit_class(at_once: int, requests: int, basis: str) -> str:
    """`no_fit` below one, `tight` below the requests, `ample` at twice them (architecture only), else `fits`."""
    if at_once < 1:
        return "no_fit"
    if at_once < requests:
        return "tight"
    if at_once >= 2 * requests and basis == "architecture":
        return "ample"
    return "fits"


def largest_card(profile: HardwareProfile) -> float | None:
    """The graphics memory of the largest card of a profile, or `None` for a machine without one."""
    sizes = [gpu.vram_gib for gpu in profile.gpus]
    return max(sizes) if sizes else None


__all__ = [
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_RUNTIME_OVERHEAD_GIB",
    "VllmFit",
    "kv_per_token_bytes",
    "largest_card",
    "requests_at_once",
    "vllm_fit",
    "vllm_fit_class",
]
