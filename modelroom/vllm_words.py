"""What a vLLM row says about itself, wherever a fit row is explained (AP9-SV2-F).

Ollama's words for a fit -- "Fits into graphics memory", a share of the pool, "all cards together",
"left after the reserve" -- describe fit v1's pool rule (`fit.compute_fit_v2`). A vLLM row is judged
by another rule: how many requests of the ranking's context its cards hold at once
(`fit_vllm.vllm_fit_class`), on the cards tensor parallel chose for it (`Fit.cards`, F1). Every text
that explains a fit row therefore decides on `Fit.runtime` first and, for vLLM, reads its own
quantities -- `at_once`, `requests`, `cards`, `reason` -- through this one module: the row note of
the document (`render`), the key under a table of vLLM rows (`views_runtime`, F5) and the lines of
step 4 (`guided_vllm`). An Ollama row never reaches it, so every Ollama output stays byte-identical.

Imports neither `runtimes` nor `render`: `runtimes` imports `guided_vllm`, which reads this module.
"""

from __future__ import annotations

from .contracts import Fit
from .fit_vllm import VLLM_GPU_MEMORY_UTILIZATION, VLLM_RUNTIME_OVERHEAD_GIB
from .screen import BEST_HINT

VLLM_RUNTIME = "vllm"  # `runtimes.VLLM.key`, spelled here to keep this module free of that import

# The key under a table of vLLM rows alone (F5: a machine with vLLM only, or the vLLM section of a
# machine with both runtimes) -- vLLM's own classes and pool, in place of Ollama's `screen.FIT_HINT`,
# never beside it. One short piece each, so the terminal wraps between them in its 100 columns.
FIT_HINT = (
    "Fit is memory only: how many requests of this context the cards hold at once",
    "ample: twice the requests this ranking assumes or more",
    "fits: that many, below twice them",
    "tight: at least one, fewer than them",
    "no fit: not one",
    f"the pool is {VLLM_GPU_MEMORY_UTILIZATION} of each card vLLM uses, {VLLM_RUNTIME_OVERHEAD_GIB:.2f} GB of it its runtime",
    "gpu: one card",
    "gpu_split: 2, 4 or 8 equal cards of the largest size, as the head count allows",
    "for a fit from size the highest class is fits",
    *BEST_HINT,
)
_SHARE = f"({VLLM_GPU_MEMORY_UTILIZATION} of each card)"
_FACTS = ["fit.runtime", "fit.at_once", "fit.requests", "fit.cards", "fit.need_gib", "fit.pool_gib"]


def is_vllm(fit: Fit) -> bool:
    """Whether `fit` is a vLLM row's -- the one question every explaining text asks first."""
    return fit.runtime == VLLM_RUNTIME


def requests_count(count: int) -> str:
    """`1 request`, `10 requests`."""
    return f"{count} request{'' if count == 1 else 's'}"


def cards_words(cards: int) -> str:
    """`one card`, `4 cards`."""
    return "one card" if cards == 1 else f"{cards} cards"


def card_gib(fit: Fit) -> float:
    """The size of each card the row uses: `pool_gib + reserve_gib` is `t` whole cards (F2)."""
    return (fit.pool_gib + fit.reserve_gib) / fit.cards


def _where(fit: Fit) -> str:
    return f"on {cards_words(fit.cards)} of {card_gib(fit):.1f} GiB"


def _need(fit: Fit) -> str:
    return f"{requests_count(fit.requests)} {'needs' if fit.requests == 1 else 'need'} about {fit.need_gib:.1f} GiB"


def row_note(fit: Fit) -> tuple[str, str, list[str]]:
    """The code, text and facts of a vLLM row's note, by its class (`ample`/`fits`/`tight`/`no_fit`):
    the requests its cards hold at once against the requests the ranking assumes, then what all of
    those need against what vLLM takes of the cards. Never "fits into graphics memory": a `tight`
    vLLM row holds only some of the requests, and a reader must see how many (Codex r2-1)."""
    if fit.fit_class == "no_fit":
        text = f"vLLM holds not even one request at once {_where(fit)}; {_need(fit)}, it takes {fit.pool_gib:.1f} GiB {_SHARE}."
        return "no_fit", text, _FACTS
    if fit.fit_class == "tight":
        text = (
            f"vLLM holds only {fit.at_once} of {requests_count(fit.requests)} at once {_where(fit)}; "
            f"all of them need about {fit.need_gib:.1f} GiB, it takes {fit.pool_gib:.1f} GiB {_SHARE}."
        )
        return "vllm_tight", text, _FACTS
    text = (
        f"vLLM holds {requests_count(fit.at_once)} at once {_where(fit)}; the {_need(fit)} of the "
        f"{fit.pool_gib:.1f} GiB it takes {_SHARE}."
    )
    return "vllm_requests_at_once", text, _FACTS


def reason_sentence(fit: Fit) -> tuple[str, list[str]]:
    """The sentence a vLLM row's own `Fit.reason` adds, or `("", [])` where there is none (Codex
    r1-2): today only F1's "no head count, computed on one card" -- a fit that still computed, not
    a guard failure (those end before the formula and stand under `Not covered` with that reason)."""
    if not is_vllm(fit) or not fit.reason:
        return "", []
    return f" {fit.reason[:1].upper()}{fit.reason[1:]}.", ["fit.reason"]


__all__ = [
    "FIT_HINT",
    "VLLM_RUNTIME",
    "card_gib",
    "cards_words",
    "is_vllm",
    "reason_sentence",
    "requests_count",
    "row_note",
]
