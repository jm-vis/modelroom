"""The catalog rows an input names: the direct find (V8) and the knowledge hint (V9) of AP9-KO.

A search by GGUF finds a publisher's model only through a repository that packages it. A model the
publisher ships as tensors alone, or one only an account nobody listed packages, never shows up
there, and with the owner filter on the list stays empty. The catalog knows the publisher's own
repository, so with a vLLM machine among the chosen ones the search asks the Hub for that repository
directly (`find_direct`), and without one it says what the catalog knows (`knowledge_hints`).

Both start from the same rows (`catalog_rows`): for a typed repository id the row with that id, for
words the rows whose id the words name (`SearchWord.matches`), for a search with no word none.
Nothing here writes anything, and nothing here knows a model, a publisher or a packager by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection

from .catalog import DISPLAYED_KINDS, Catalog, CatalogModel
from .contracts import hf_id_key
from .catalog_findings import KNOWLEDGE_HINT_MAX, knowledge_hint_text
from .guided_contracts import DIRECT_FIND_MAX, UNKNOWN, DirectLogEntry, SearchHit
from .http import BudgetedTransport, BudgetExhaustedError, RequestBudget, Transport
from .search import _catalog_ollama, _created_at, _license, _parameters_b
from .search_pages import ANSWER_ERRORS, TRANSPORT_ERRORS, model_url, read_downloads
from .search_word import SearchWord

BUDGET_ENDED = "the request budget ended"


@dataclass(frozen=True)
class DirectOutcome:
    """What the direct find made of an input: the hits, and one log entry for every model it asked for."""

    hits: list[SearchHit] = field(default_factory=list)
    log: list[DirectLogEntry] = field(default_factory=list)


def catalog_rows(catalog: Catalog, word: SearchWord) -> list[CatalogModel]:
    """The catalog models this input names, `latest` ones first, then in catalog order; text and vision only.

    A typed id names the one row with that id (the Hub's ids are case-insensitive); words name every row
    whose id they match; an input with no word names none.
    """
    models = [model for family in catalog.families for model in family.models if model.kind in DISPLAYED_KINDS]
    if word.form == "id" and word.repo_id is not None:
        named = [model for model in models if hf_id_key(model.hf_repo) == hf_id_key(word.repo_id)]
    elif word.form == "words":
        named = [model for model in models if word.matches(model.hf_repo)]
    else:
        named = []
    return sorted(named, key=lambda model: not model.latest)


def find_direct(
    transport: Transport,
    budget: RequestBudget,
    catalog: Catalog,
    word: SearchWord,
    *,
    skip: Collection[str] = frozenset(),
) -> DirectOutcome:
    """Ask the Hub for the publisher's own repository of every row the input names, at most `DIRECT_FIND_MAX`.

    `skip` holds the `hf_id_key`s of base models the search already offers, so they are not asked for twice.
    One request per model, booked in the shared `budget`; an answer has to name that very repository. A model
    that could not be read, or that the budget cut off, is named in the log and gives no hit -- it is never left
    out of the log.
    """
    source = "id" if word.form == "id" else "word"
    rows = [model for model in catalog_rows(catalog, word) if hf_id_key(model.hf_repo) not in skip][:DIRECT_FIND_MAX]
    # No redirect is followed: a model costs one request, a moved one is logged as not read (a status other than 200).
    asking = BudgetedTransport(transport, budget)
    hits: list[SearchHit] = []
    log: list[DirectLogEntry] = []
    for position, row in enumerate(rows):
        used = budget.used
        try:
            entry, reason = _read(asking, row.hf_repo)
        except BudgetExhaustedError:
            log += _cut_off(rows[position:], source)
            break
        requests = budget.used - used
        if entry is None:
            log.append(DirectLogEntry(repo=row.hf_repo, source=source, status="not_read", requests=requests, reason=reason))
            continue
        hits.append(_direct_hit(catalog, row, entry))
        log.append(DirectLogEntry(repo=row.hf_repo, source=source, status="found", requests=requests))
    return DirectOutcome(hits=hits, log=log)


def _cut_off(rows: list[CatalogModel], source: str) -> list[DirectLogEntry]:
    """The log lines of the models a budget that ended did not let be asked: no request was made for any of them."""
    return [DirectLogEntry(repo=row.hf_repo, source=source, status="budget", requests=0, reason=BUDGET_ENDED) for row in rows]


def _read(asking: Transport, repo: str) -> tuple[dict | None, str | None]:
    """The Hub's own answer for `repo`, or `None` and why not. A budget that ends is the caller's to see."""
    try:
        response = asking("GET", model_url(repo))
    except TRANSPORT_ERRORS as exc:  # a request that did not go through is a model that was not read; a bug goes up
        return None, f"the request failed: {type(exc).__name__}"
    if response.status != 200:
        return None, f"HTTP {response.status}"
    try:
        payload = response.json()
    except ANSWER_ERRORS:
        return None, "the answer is not JSON"
    if not isinstance(payload, dict):
        return None, "the answer is not a repository"
    named = (payload.get("id"), payload.get("modelId"))
    if not any(isinstance(name, str) and hf_id_key(name) == hf_id_key(repo) for name in named):
        return None, "the answer names another repository"
    return payload, None


def _direct_hit(catalog: Catalog, row: CatalogModel, entry: dict) -> SearchHit:
    """The publisher's own model as a hit: resolved as itself, no base model, no relation, age from the catalog."""
    age, successor = catalog.age_of(row.hf_repo)
    card = entry.get("cardData")
    return SearchHit(
        repo=row.hf_repo,
        publisher_status="publisher",
        resolved=True,
        resolved_base_model=row.hf_repo,
        repo_created_at=_created_at(entry.get("createdAt")),
        downloads=read_downloads(entry.get("downloads")),
        parameters_b=_parameters_b(entry.get("safetensors")),
        license=_license(card),
        age=age,
        successor=successor,
        release_basis="stated" if age != UNKNOWN else None,
        ollama=_catalog_ollama(catalog, row.hf_repo),
        found_as="direct",
    )


def knowledge_hints(
    catalog: Catalog, word: SearchWord, found: Collection[str], tensor_proven: Collection[str]
) -> tuple[list[str], int]:
    """The gray lines of a search without a vLLM machine, and how many more the limit left out (V9).

    A row the input names whose list of GGUF repositories is empty, and whose base model no GGUF hit of this
    search resolved to (`found`, by `hf_id_key`). The sentence on tensor packages follows only for a model in
    `tensor_proven`, the snapshot's positive proof; the catalog holds none. No request is made.
    """
    rows = [model for model in catalog_rows(catalog, word) if not model.gguf_repos and hf_id_key(model.hf_repo) not in found]
    lines = []
    for model in rows[:KNOWLEDGE_HINT_MAX]:
        publisher, _slash, name = model.hf_repo.partition("/")
        lines.append(knowledge_hint_text(name, publisher, tensor_offer=hf_id_key(model.hf_repo) in tensor_proven))
    return lines, max(0, len(rows) - KNOWLEDGE_HINT_MAX)


__all__ = ["BUDGET_ENDED", "DirectOutcome", "catalog_rows", "find_direct", "knowledge_hints"]
