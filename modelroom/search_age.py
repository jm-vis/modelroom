"""`latest` or `legacy` for one publisher model, from positive evidence only.

Split out of `modelroom/search.py` on 2026-09-25, unchanged: the search grew the three forms of
input it reads and the file passed the house code-mass threshold. `search.py` re-exports
`AgeVerdict` and `decide_age`, so every caller still reads them there.

An age statement is always positive evidence about one model -- a successor named at the
publisher's own repository, or the shipped catalog -- and never a comparison of two version
numbers (`AGENTS.md`, the language standard).
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import Age, Catalog
from .contracts import validate_hf_repo
from .http import BudgetExhaustedError, Transport

HF_API = "https://huggingface.co/api"
# At most two `new_version` edges are followed; the second one only names a younger successor
# and never changes the verdict, which is already `legacy` after the first.
MAX_SUCCESSOR_EDGES = 2


@dataclass(frozen=True)
class AgeVerdict:
    """`latest`/`legacy`/`unknown` for one publisher model, with the evidence that decided it."""

    age: Age
    successor: str | None
    evidence: str


def decide_age(transport: Transport, catalog: Catalog, hf_repo: str) -> AgeVerdict:
    """Whether `hf_repo` is the publisher's current model, a superseded one, or neither.

    Positive evidence only, in this order:

    1. a `new_version` on the publisher repo's own card whose target is a repository that
       really exists under the same account: `legacy`, with that target as `successor`. At most
       `MAX_SUCCESSOR_EDGES` edges are followed, and only to name a younger successor -- the
       verdict stays `legacy` even when the second edge fails, is a cycle or runs out of budget;
    2. no `new_version`: the catalog's own statement -- `latest` (with its evidence), `legacy`
       with its successor, or `unknown` when the catalog says neither;
    3. a `new_version` whose target is invalid, the repository itself, or unreachable, and a
       budget that ends before the first edge could be checked: `unknown`. A broken pointer is
       not evidence for the catalog's statement either, so it never falls back to it.

    `transport` is the caller's already-budgeted transport (`run_search` composes
    `RedirectingTransport(BudgetedTransport(...))` once for the whole search), so every request
    here is booked against the run's shared budget like any other.
    """
    try:
        card = _card_data(transport, hf_repo)
    except BudgetExhaustedError:
        return AgeVerdict("unknown", None, "budget exhausted before the first edge")

    new_version = card.get("new_version") if card is not None else None
    if not isinstance(new_version, str):
        age, successor = catalog.age_of(hf_repo)
        return AgeVerdict(age, successor, "catalog")

    owner = hf_repo.split("/", 1)[0]
    seen = {hf_repo}
    successor: str | None = None
    current_target = new_version
    for edge in range(MAX_SUCCESSOR_EDGES):
        if not _is_candidate_successor(current_target, owner, seen):
            break
        try:
            target_card = _card_data(transport, current_target)
        except BudgetExhaustedError:
            break
        if target_card is None:
            break
        seen.add(current_target)
        successor = current_target
        next_target = target_card.get("new_version")
        if not isinstance(next_target, str) or edge + 1 >= MAX_SUCCESSOR_EDGES:
            break
        current_target = next_target

    if successor is None:
        return AgeVerdict("unknown", None, f"new_version {new_version!r} is not a reachable repo of {owner}")
    return AgeVerdict("legacy", successor, f"new_version at {hf_repo}")


def _is_candidate_successor(target: str, owner: str, seen: set[str]) -> bool:
    """A `new_version` target worth one request: a well-formed repo of `owner`, not seen yet."""
    try:
        validate_hf_repo(target)
    except ValueError:
        return False
    return target.split("/", 1)[0] == owner and target not in seen


def _card_data(transport: Transport, hf_repo: str) -> dict | None:
    """One repository's `cardData`, or `None` when *this* repository cannot be read at all.

    The answer has to name `hf_repo` itself (`id`, or `modelId`): the transport follows
    redirects, and the Hub answers a moved repository's path with the repository it moved to, so
    without that check a request for `acme/B` could be satisfied by `other/B` -- and then the
    same-account rule and the cycle check in `decide_age` would both be reading about a
    repository nobody asked for. An answer that names nothing at all is no proof either.

    `BudgetExhaustedError` is the one exception that passes through: it says nothing about the
    repository and the caller has to tell it apart from "read, and it says nothing".
    """
    try:
        response = transport("GET", f"{HF_API}/models/{hf_repo}")
    except BudgetExhaustedError:
        raise
    except Exception:
        return None
    if response.status != 200:
        return None
    try:
        info = response.json()
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    if info.get("id") != hf_repo and info.get("modelId") != hf_repo:
        return None
    card = info.get("cardData")
    return card if isinstance(card, dict) else {}


__all__ = ["HF_API", "MAX_SUCCESSOR_EDGES", "AgeVerdict", "decide_age"]
