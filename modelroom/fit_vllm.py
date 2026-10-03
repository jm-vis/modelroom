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
architecture basis, `fits` otherwise. Since AP9-SV2-F `vllm_fit_as_fit` computes this per card
over the `t` equal cards tensor parallel chooses (F1, F2, rule (b)) and returns fit contract v1's own `Fit`;
what a row then says about itself is `vllm_words`'s. Tensor parallel is computed, never measured
(ADR 0008). `fit.py` is unchanged, and nothing here is persisted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .architecture import kv_token_bytes, state_bytes
from .contracts import Architecture, BaseModelSpec, Fit, Package
from .fit import GIB, KV_PER_TOKEN_SIZE_BASIS
from .fit import unknown_fit as _unknown_fit
from .measurements import Scenario
from .profile import GpuAdapter, HardwareProfile, fit_block_reason

VLLM_GPU_MEMORY_UTILIZATION = 0.92
VLLM_RUNTIME_OVERHEAD_GIB = 1.50

# F1: the powers of two of 1, 2, 4, 8 modelroom will ever ask vLLM to split a model's weights over --
# a simplification of this package, not a limit vLLM itself has. Rule (b) (GF 2026-10-03) takes the
# smallest of them at which the package fits; the largest allowed one is the fallback.
TENSOR_PARALLEL_SIZES: tuple[int, ...] = (8, 4, 2, 1)
# Cards within 1 % of the largest are "the same size" for the grouping that chooses `t` (F1).
EQUAL_CARD_TOLERANCE = 0.01

NO_HEAD_COUNT_REASON = "tensor parallel needs the model's head count; computed on one card"
NO_CARD_REASON = "vLLM needs a graphics card"
SHARED_MEMORY_REASON = "vLLM does not run on shared memory"
INCOMPLETE_PACKAGE_REASON = "only a complete package is judged"
ZERO_WEIGHT_REASON = "a weight file has no size"
_WEIGHT_ROLES = ("weights", "weights_shard")
# Rule (b): the classes of `vllm_fit_class` that count as "the package fits" for choosing `t`.
_FITTING_CLASSES = ("fits", "ample")


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


# --- AP9-SV2-F: tensor parallel over several cards, as a `Fit` (`vllm_fit_as_fit`) -----------------


@dataclass(frozen=True)
class CardGroup:
    """The cards tensor parallel may split a model over (F1): the equal-size group, in profile
    order, and how many smaller cards of the same profile were left out of it."""

    cards: tuple[GpuAdapter, ...]
    left_out: int


def equal_size_group(profile: HardwareProfile) -> CardGroup:
    """The cards of `profile` within 1 % of the largest (F1): the group tensor parallel splits a
    model over. Unequal cards outside it are left out, not mixed in."""
    sizes = [gpu.vram_gib for gpu in profile.gpus]
    if not sizes:
        return CardGroup((), 0)
    largest = max(sizes)
    group = tuple(gpu for gpu in profile.gpus if largest - gpu.vram_gib <= largest * EQUAL_CARD_TOLERANCE)
    return CardGroup(group, len(profile.gpus) - len(group))


def allowed_sizes(group_size: int, num_attention_heads: int) -> tuple[int, ...]:
    """The `t` F1 allows, smallest first: each of `TENSOR_PARALLEL_SIZES` that is at most `group_size`
    cards and divides `num_attention_heads` evenly. `1` is always in it (`group_size` is at least 1
    here: a caller with no card never reaches this)."""
    return tuple(size for size in reversed(TENSOR_PARALLEL_SIZES) if size <= group_size and num_attention_heads % size == 0)


def tensor_parallel_size(group_size: int, num_attention_heads: int | None) -> tuple[int, str | None]:
    """The largest `t` F1 allows (`allowed_sizes`): the one a package that fits at no `t` is shown with.
    Rule (b) (`_vllm_formula`) takes the smallest `t` at which the package fits instead. `1` with
    `NO_HEAD_COUNT_REASON` where the head count is unknown -- the one case `t` itself carries a
    reason; a computed `t` above one needs none."""
    if num_attention_heads is None:
        return 1, NO_HEAD_COUNT_REASON
    return allowed_sizes(group_size, num_attention_heads)[-1], None


def largest_parallel_size(group_size: int) -> int:
    """The most cards F1 can ever choose from an equal group of `group_size`, whatever the model:
    the largest of `TENSOR_PARALLEL_SIZES` at most `group_size` (three cards give two)."""
    return next((size for size in TENSOR_PARALLEL_SIZES if size <= group_size), 1)


def chosen_cards(group: CardGroup, cards: int) -> tuple[GpuAdapter, ...]:
    """The cards a tensor-parallel size of `cards` takes (F1): the first of the group by `index`."""
    return tuple(sorted(group.cards, key=lambda gpu: gpu.index)[:cards])


@dataclass(frozen=True)
class VllmPool:
    """The most vLLM can take of one machine (Codex r6-2): `cards` of its equal `group`, `card_gib`
    the smallest of them, `pool_gib` 0.92 of each -- an upper bound, as `t` depends on the model."""

    cards: int
    group: int
    card_gib: float
    pool_gib: float


def largest_pool(profile: HardwareProfile) -> VllmPool | None:
    """`VllmPool` of `profile` through F1's own choice (`equal_size_group`, `largest_parallel_size`,
    `chosen_cards`) -- what the machine's head says, never a sum of its own; `None` without a card."""
    group = equal_size_group(profile)
    if not group.cards:
        return None
    cards = largest_parallel_size(len(group.cards))
    card_gib = min(gpu.vram_gib for gpu in chosen_cards(group, cards))
    return VllmPool(cards, len(group.cards), card_gib, cards * VLLM_GPU_MEMORY_UTILIZATION * card_gib)


def _per_token_bytes(architecture: Architecture | None, cards: int) -> tuple[float, str]:
    """The 16-bit KV cache of one token in bytes, split over `cards` (F2, Assumption A2): the whole
    model's own `kv_token_bytes` divided by `min(cards, num_key_value_heads)` -- fewer KV heads
    than cards means the heads replicate rather than split any further. The size basis carries no
    head count to split by, so it stays `cards`-independent, as SV1 already had it."""
    per_token = kv_token_bytes(architecture) if architecture is not None else None
    if per_token is None:
        return float(KV_PER_TOKEN_SIZE_BASIS), "size"
    divisor = min(cards, architecture.num_key_value_heads)
    return per_token / divisor, "architecture"


def _package_guard(package: Package) -> str | None:
    """Fit v1's own two guards before any formula runs (`fit.py:141-145`/`:226-231`), in the same
    wording: no complete set of weight files is no size to compute with, whatever the runtime."""
    if not package.complete:
        return INCOMPLETE_PACKAGE_REASON
    if any(f.size_bytes == 0 for f in package.files if f.role in _WEIGHT_ROLES):
        return ZERO_WEIGHT_REASON
    return None


def _tokens_text(context: int) -> str:
    """A window as a reader keeps it: `32k` in whole thousands, else the number (`fit._tokens_text`)."""
    return f"{context // 1024}k" if context % 1024 == 0 else f"{context} tokens"


def vllm_fit_as_fit(
    profile: HardwareProfile,
    package: Package,
    base_model: BaseModelSpec,
    scenario: Scenario,
    machine_config: object,
) -> Fit:
    """vLLM's fit (F1, F2) in fit contract v1's own `Fit` shape, so the rank, the document and the
    table read it exactly as they read an Ollama row -- this is `runtimes.VLLM.fit`, the one place
    this is called from (`runtimes.fit_for`, in turn read by `render._machine_block` and, through
    it, `guided_vllm._fit`: one calculation, one truth).

    Guarded the same way fit v1 itself is guarded, in the same order and the same words: a blocked
    profile (`profile.fit_block_reason`, mirroring `fit.compute_fit_v2`'s own first check), an
    incomplete package or a weight file of size 0 (`fit.py:141-145`), a known context window
    smaller than the scenario's (`fit.py:226-231`) -- `unknown_fit` either way, so a reader of the
    ranked document cannot tell a vLLM package's guard from a GGUF one's. Then the graphics gate
    (F1): unified memory never runs vLLM, and a machine with no card at all gets no row -- both
    `not_covered` (fit class `unknown`), never `no_fit`. `machine_config` is read by every other
    runtime's `fit` (the configured VRAM reserve); vLLM's own 0.92 share already is that reserve,
    so this one carries the parameter only to keep `runtimes.Runtime.fit`'s one shape.
    """
    group = equal_size_group(profile)
    guarded = _vllm_guard_reason(profile, package, base_model, scenario, group)
    if guarded is not None:
        return _vllm_unknown_fit(guarded)
    return _vllm_formula(group, package, base_model, scenario)


def _vllm_unknown_fit(reason: str) -> Fit:
    """`fit.unknown_fit`, but carrying vLLM's own `Fit.runtime` (Codex r1-3): every guard path of
    this module judges a package no runtime but vLLM would ever have been asked about, and the
    default `runtime == "ollama"` of a plain `unknown_fit` would mislabel it -- a `not_covered`
    row's JSON and packager cell must say `vllm`, the same as a ranked one's."""
    return _unknown_fit(reason).model_copy(update={"runtime": "vllm"})


def _vllm_guard_reason(
    profile: HardwareProfile, package: Package, base_model: BaseModelSpec, scenario: Scenario, group: CardGroup
) -> str | None:
    """The reason none of F2's formula runs, or `None` where it may (the guard chain of
    `vllm_fit_as_fit`'s own docstring, split out to keep the one function short)."""
    blocked = fit_block_reason(profile)
    if blocked is not None:
        return blocked
    guard = _package_guard(package)
    if guard is not None:
        return guard
    window = base_model.architecture.max_context
    if window is not None and window < scenario.context_requested:
        return f"context beyond the window of this model ({_tokens_text(window)})"
    if profile.gpu_state == "unified_memory":
        return SHARED_MEMORY_REASON
    if not group.cards:
        return NO_CARD_REASON
    return None


def _vllm_formula(group: CardGroup, package: Package, base_model: BaseModelSpec, scenario: Scenario) -> Fit:
    """F2's formula, over the cards `_vllm_guard_reason` already cleared (`group.cards` is not empty).

    Rule (b) (GF 2026-10-03): `t` is the smallest allowed size (`allowed_sizes`) at which the package
    fits -- shorter latency, the other cards stay free. "Fits" is F2's own class over the first `t`
    cards: `fits` or `ample`, so neither `no_fit` nor `tight` (`unknown` cannot reach the formula).
    At no size it fits: the largest allowed `t`, the best try. No head count: `t = 1` with its reason."""
    heads = base_model.architecture.num_attention_heads
    if heads is None:
        return _vllm_fit_at(group, 1, NO_HEAD_COUNT_REASON, package, base_model, scenario)
    sizes = allowed_sizes(len(group.cards), heads)
    for cards in sizes:
        fit = _vllm_fit_at(group, cards, None, package, base_model, scenario)
        if fit.fit_class in _FITTING_CLASSES:
            return fit
    return _vllm_fit_at(group, sizes[-1], None, package, base_model, scenario)


def _vllm_fit_at(
    group: CardGroup, cards: int, t_reason: str | None, package: Package, base_model: BaseModelSpec, scenario: Scenario
) -> Fit:
    """F2's formula for exactly `cards` cards of `group`, as a `Fit`."""
    card_gib = min(gpu.vram_gib for gpu in chosen_cards(group, cards))
    weights_gib = sum(f.size_bytes for f in package.files if f.role in _WEIGHT_ROLES) / GIB
    budget, at_once, basis, kv_gib = _vllm_budget(card_gib, weights_gib, cards, base_model.architecture, scenario)
    return Fit(
        fit_class=vllm_fit_class(at_once, scenario.requests, basis),
        mode="gpu" if cards == 1 else "gpu_split",
        need_gib=weights_gib + cards * VLLM_RUNTIME_OVERHEAD_GIB + kv_gib,
        weights_gib=weights_gib,
        kv_gib=kv_gib,
        pool_gib=cards * budget,
        reserve_gib=cards * (card_gib - budget),
        context=scenario.context_requested,
        context_assumed=False,
        basis=basis,
        # Codex r1-2: the reason `t` stopped at one card (a missing head count) is not a guard
        # failure -- the fit still computes, F1's own simplification -- but a reader must still be
        # told: carried on every fit_class this formula reaches, ranked or not, `None` where `t`
        # needed no excuse (a real head count, or more than one card was never on offer).
        reason=t_reason,
        requests=scenario.requests,
        runtime="vllm",
        at_once=at_once,
        cards=cards,
    )


def _vllm_budget(
    card_gib: float, weights_gib: float, cards: int, architecture: Architecture, scenario: Scenario
) -> tuple[float, int, str, float]:
    """The budget, `at_once` and `kv_gib` of `_vllm_formula`'s own cards, split out to keep that
    one function short. `at_once` is one card's own check (ADR 0008: tensor parallel is not
    measured, only computed here) -- the available budget after this card's share of the weights
    and the runtime overhead, against the per-card KV cost. `kv_gib` is `Fit`'s own field instead
    (Codex r1-1): the whole model's KV need, that per-card cost times every card `cards` spreads
    the package over -- never the leftover budget alone, which can be negative.
    """
    budget = VLLM_GPU_MEMORY_UTILIZATION * card_gib
    available_kv_gib = budget - weights_gib / cards - VLLM_RUNTIME_OVERHEAD_GIB
    available_kv_bytes = math.floor(max(available_kv_gib, 0.0) * GIB)
    per_token, basis = _per_token_bytes(architecture, cards)
    state = state_bytes(architecture) / cards if basis == "architecture" else 0.0
    at_once = int(requests_at_once(available_kv_bytes, per_token, scenario.context_requested, state))
    kv_needed_per_card_gib = (per_token * scenario.context_requested + state) * scenario.requests / GIB
    return budget, at_once, basis, kv_needed_per_card_gib * cards


__all__ = [
    "EQUAL_CARD_TOLERANCE",
    "INCOMPLETE_PACKAGE_REASON",
    "NO_CARD_REASON",
    "NO_HEAD_COUNT_REASON",
    "SHARED_MEMORY_REASON",
    "TENSOR_PARALLEL_SIZES",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_RUNTIME_OVERHEAD_GIB",
    "ZERO_WEIGHT_REASON",
    "CardGroup",
    "VllmFit",
    "VllmPool",
    "allowed_sizes",
    "chosen_cards",
    "equal_size_group",
    "kv_per_token_bytes",
    "largest_card",
    "largest_parallel_size",
    "largest_pool",
    "requests_at_once",
    "tensor_parallel_size",
    "vllm_fit",
    "vllm_fit_as_fit",
    "vllm_fit_class",
]
