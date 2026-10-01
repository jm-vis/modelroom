"""Finding vLLM tensor repositories of an already-resolved base model (AP9-SV2-P, P1).

One request per packager account of `TENSOR_PACKAGERS` (the base model's own owner plus the
listed vLLM packagers), asked only when the machine this search is for has vLLM -- a machine
without it costs no extra request at all, so an ordinary Ollama run and its goldens are
unchanged (CONTRACTS.md, "Local servers and the vLLM load test", `runtimes`).

This module only *finds* repository names and counts the requests it spent; it does not fetch
a repository's own tree or `config.json` (`hf.py`'s area fetch does that afterwards, once the
repository is written into a base model's `repos`, `search_apply.apply_tensor_repos`) and it
never writes anything to disk itself.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Sequence

from .http import BudgetedTransport, BudgetExhaustedError, RedirectingTransport, RequestBudget, Transport
from .relation import check_relation
from .search_pages import HF_API, read_downloads

# GF question F3 (2026-09-29): the base model's own owner (always included, never a request of
# its own -- its repository is already a target, AP9-UX2 P3) plus RedHatAI, the one vLLM
# packager with real inventory at the time this was decided. A fixed constant, not a
# configuration field (P1): the search for Hugging Face tensor packagers is new and narrow
# enough that a second packager is a code change, the same way `DEFAULT_PACKAGERS` once was
# before the GGUF search grew a configuration field -- unlike that one, this list is not (yet)
# user-configurable.
TENSOR_PACKAGERS: tuple[str, ...] = ("RedHatAI",)

# P1: at most this many tensor repositories are kept per base model, by downloads -- the rest
# are named as a count in a note, never silently dropped and never fetched.
MAX_TENSOR_REPOS_PER_MODEL = 4

# One account's tensor page: narrower than the GGUF account page (one sort order, not two) --
# this search only has to tell "a quantized tensor build exists here", not rank a packager's
# whole catalog, so the most-downloaded order alone is enough.
_TENSOR_ACCOUNT_LIMIT = 20
_EXPAND = ("cardData", "downloads", "tags")

TENSOR_NOT_FETCHED_NOTE = "and {count} more vLLM builds not fetched"


def tensor_search_url(base_name: str, account: str) -> str:
    """The pinned request for one account's tensor (safetensors) repositories of `base_name`."""
    query = urllib.parse.urlencode(
        {"author": account, "search": base_name, "filter": "safetensors", "sort": "downloads", "direction": -1}
    )
    url = f"{HF_API}/models?{query}&limit={_TENSOR_ACCOUNT_LIMIT}"
    return url + "".join(f"&expand={field_name}" for field_name in _EXPAND)


def tensor_packagers(base_model_hf_repo: str, packagers: Sequence[str] = TENSOR_PACKAGERS) -> list[str]:
    """The accounts P1 asks for one base model: its own owner, then `packagers`, each once."""
    owner = base_model_hf_repo.split("/", 1)[0]
    accounts = [owner]
    for packager in packagers:
        if packager not in accounts:
            accounts.append(packager)
    return accounts


@dataclass(frozen=True)
class _Candidate:
    repo: str
    downloads: int | None


@dataclass(frozen=True)
class TensorSearchOutcome:
    """What one base model's tensor search found: the repositories to add, and what was left out.

    `repos` is at most `MAX_TENSOR_REPOS_PER_MODEL`, by downloads (highest first), ties broken by
    repository id for a deterministic result; `skipped` is how many more were found and were not;
    `requests` is how many account pages this search asked (for the shared request budget's note).
    """

    repos: tuple[str, ...] = ()
    skipped: int = 0
    requests: int = 0

    def note(self) -> str | None:
        return TENSOR_NOT_FETCHED_NOTE.format(count=self.skipped) if self.skipped else None


def search_tensor_repos(
    transport: Transport,
    budget: RequestBudget,
    base_model_hf_repo: str,
    *,
    packagers: Sequence[str] = TENSOR_PACKAGERS,
) -> TensorSearchOutcome:
    """Ask every account of `tensor_packagers` for `base_model_hf_repo`'s tensor builds.

    One request per account, in order; a hit is accepted when `relation.check_relation` proves
    it `quantized` from this exact base model (the same check the GGUF search and fetch share).
    `BudgetExhaustedError` ends the search where it stands -- the accounts already asked keep
    their candidates, the budget's own exhaustion note covers the rest (CONTRACTS.md, "Request
    budget"); every other response shape is a plain pure read (JSON, a list of objects), and an
    account that does not answer `200` with a list contributes nothing, never a crash.
    """
    budgeted = RedirectingTransport(BudgetedTransport(transport, budget))
    candidates: list[_Candidate] = []
    requests = 0
    for account in tensor_packagers(base_model_hf_repo, packagers):
        url = tensor_search_url(base_model_hf_repo.split("/", 1)[1], account)
        try:
            response = budgeted("GET", url)
        except BudgetExhaustedError:
            break
        requests += 1
        candidates.extend(_accepted(response, base_model_hf_repo))
    candidates.sort(key=lambda candidate: (-(candidate.downloads or 0), candidate.repo))
    kept = candidates[:MAX_TENSOR_REPOS_PER_MODEL]
    return TensorSearchOutcome(
        repos=tuple(candidate.repo for candidate in kept),
        skipped=max(0, len(candidates) - len(kept)),
        requests=requests,
    )


def _accepted(response, base_model_hf_repo: str) -> list[_Candidate]:
    if response.status != 200:
        return []
    try:
        hits = response.json()
    except Exception:
        return []
    if not isinstance(hits, list):
        return []
    accepted: list[_Candidate] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        repo = hit.get("id")
        if not isinstance(repo, str):
            continue
        status, _reason = check_relation(hit.get("tags"), hit.get("cardData"), base_model_hf_repo)
        if status != "quantized":
            continue
        accepted.append(_Candidate(repo=repo, downloads=read_downloads(hit.get("downloads"))))
    return accepted


__all__ = [
    "MAX_TENSOR_REPOS_PER_MODEL",
    "TENSOR_NOT_FETCHED_NOTE",
    "TENSOR_PACKAGERS",
    "TensorSearchOutcome",
    "search_tensor_repos",
    "tensor_packagers",
    "tensor_search_url",
]
