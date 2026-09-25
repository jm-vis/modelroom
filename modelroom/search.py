"""Search Hugging Face for a model name, resolve the hits, and write the result into a config.

Four steps, all over the transport this package already has -- no SDK, no second HTTP stack:

1. `run_search` reads the input first (`search_word.parse_search`) and asks the pages that input
   leads to (`search_pages`): one for a typed repository id, two per account for words, one per
   current catalog model for no word at all, plus the two open lists with the owner filter off.
   Everything the resolution needs (`tags`, `cardData`, `createdAt`, `downloads`, `safetensors`)
   is asked for in those requests, so a hit costs no further request.
2. Resolution: a hit is `resolved` only when `relation.check_relation` proves it a `quantized`
   build of exactly one base model, and -- with the owner filter on -- the catalog knows that base
   model's account as a publisher. Everything else is shown with its reason, gets no Ollama name,
   no age and no fit.
3. `decide_age` states `latest`/`legacy` from positive evidence only: a valid `new_version` at
   the publisher repo, else the catalog's own statement, else `unknown`.
4. `apply_hits` turns the resolved hits into families and owner-bound `repos` entries of a
   configuration (pure), and `write_configuration` writes that configuration out under the
   state lock.

Search, resolution, successor lookups and the following `fetch` share **one** request budget
(`DEFAULT_GUIDED_BUDGET`): the caller keeps the `RequestBudget` object and hands what is left
to `fetch.run_fetch` (CONTRACTS.md, "Shared request budget").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Mapping, Sequence

from .catalog import Catalog
from .contracts import validate_hf_repo, validate_ollama_name
from .guided_contracts import UNKNOWN, SearchHit
from .http import BudgetExhaustedError, BudgetedTransport, RedirectingTransport, RequestBudget, Transport
from .relation import check_relation
from .search_age import HF_API, MAX_SUCCESSOR_EDGES, AgeVerdict, decide_age
from .search_apply import apply_hits, family_name_for, write_configuration
from .search_pages import (
    TYPED_LABEL,
    GroupPlan,
    SearchGroup,
    SearchMode,
    SearchPlan,
    SearchRequest,
    accounts_for,
    catalog_plan,
    model_url,
    publishers_for,
    read_downloads,
    typed_plan,
    word_plan,
)
from .search_word import SearchWord, catalog_models, close_matches, latest_model_names, parse_search

# The whole guided run -- search, resolution, successor lookups, tree pages and the fetch --
# shares this many requests unless the caller passes its own budget. 150 since the search asks
# one request per account (decided 2026-09-24): measured live, `qwen` with the owner filter on
# spends 47 requests before the selection list is shown, and 60 left the fetch of two chosen
# repositories `incomplete (budget exhausted)`. A plain `modelroom fetch` has 400 on its own.
DEFAULT_GUIDED_BUDGET = 150
# The shipped positive list of packager accounts, the same one `modelroom.example.toml` ships
# (`test_search.py` keeps the two in step). A guided run that has no configuration yet uses it
# to tell a listed packager from any other account.
DEFAULT_PACKAGERS: tuple[str, ...] = ("unsloth", "bartowski", "mradermacher", "lmstudio-community", "ggml-org")
NO_OLLAMA_LABEL = "none known"
# `search.json`, the file one search leaves behind ("Search log"). 2 since 2026-09-25: the file
# names the `mode` of the search it records, so a reader can tell a word search from a typed
# repository id and from the catalog pages of a search with no word at all.
SEARCH_LOG_SCHEMA_VERSION = 2
_OLLAMA_ENTRY_PREFIX = "ollama:"
# A base id no repository can carry (a space is not allowed in a repo id), so `check_relation`
# reports `base_model_tag` for a hit that does not declare exactly one base.
_NO_SINGLE_BASE = "(no single declared base)"
# Each half of a repository id, as the Hub itself requires it: starts with an alphanumeric
# character, then letters, digits, `.`, `_` or `-`. Stricter than the shared `hf_repo` pattern
# on purpose -- see `_is_repo_id`.
_REPO_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The tag the Hub sets on a repository that holds at least one GGUF file, and the one the word
# search filters by (`filter=gguf`). A typed repository id is read against the same tag, so both
# ways into the search agree on what a GGUF repository is.
GGUF_TAG = "gguf"
# A typed id whose repository holds no GGUF file, and one the Hub does not answer for: both fall
# back to the word search over the name half, so the input is never simply refused.
NO_GGUF_NOTE = "{repo} holds no GGUF file"
UNREADABLE_ID_NOTE = "{repo} is no repository this search could read"
# What a search with no word at all says it did instead.
NO_WORD_NOTE = "no search word: the catalog's current models, one page each"
# With this many requests left or fewer, a search with no word asks no further catalog page. A stop
# rule and not a fence: the page it is already allowed to ask resolves its own repositories, and
# those age lookups spend from the reserve as well (second-model round, 2026-09-25). The number
# exists so that the fetch of a chosen model, which shares this budget, still has requests -- the
# catalog has more current models than one page each plus one age lookup per resolved base model
# fits into 150 in the worst case.
CATALOG_BUDGET_RESERVE = 20
CATALOG_BUDGET_NOTE = "the request budget ended after {asked} of {planned} catalog pages"


class SearchError(Exception):
    """The search request itself failed: a status, a body or a budget that ends the search."""


# What the summary says when the word matches no catalog family: the two shapes differ because
# what *was* asked differs. Claiming "only the listed packagers" while both open lists were asked
# would be untrue (found in the second-model round, 2026-09-24).
NO_PUBLISHER_NOTE = "no publisher in the catalog matches {word!r}; only the listed packagers were asked"
NO_PUBLISHER_NOTE_OPEN = (
    "no publisher in the catalog matches {word!r}; the listed packagers and the two open lists were asked"
)


@dataclass(frozen=True)
class SearchOutcome:
    """One search: its groups, its flat hit list, how many resolved, and what the budget spent.

    `groups` are the answered pages in the order they were asked -- the matching publisher
    accounts, then the positive list, and with the owner filter off the two open pages after
    them. `hits` is the same repositories flat, in exactly that order, so every caller that
    already worked on a flat list (`apply_hits`, the selection list, the answer file's `select`)
    is unchanged. `notes` carries what the search has to say about the *request* rather than
    about a repository, such as a word no catalog family matches.
    """

    query: str
    hits: list[SearchHit]
    budget_used: int
    budget_limit: int
    groups: list[SearchGroup] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # What class the search asked an account by: what it knew then, not what a reader can work out.
    publishers: list[str] = field(default_factory=list)
    listed_packagers: list[str] = field(default_factory=list)
    # Which form the input had (`search_word.parse_search`), so a reader of `search.json` can tell
    # a word search from a typed id and from the catalog pages of a search with no word.
    mode: SearchMode = "word"
    # The repository id that was typed, when one was: the one hit the owner filter never hides.
    typed: str | None = None
    # What the input could have meant, when the catalog names nothing that it does: the words the
    # guided mode offers instead of an empty list (`guided_search.candidate_choices`).
    candidates: list[str] = field(default_factory=list)
    # Pages that were asked without leaving a group behind: the one page of a typed repository id
    # that held no GGUF file, or that the Hub did not answer for. It spent a request and the word
    # search followed, so `requests` has to count it -- otherwise the log says two requests for a
    # run that made three (second-model round, 2026-09-25).
    extra_requests: int = 0

    @property
    def resolved(self) -> int:
        return sum(1 for hit in self.hits if hit.resolved)

    @property
    def unresolved(self) -> int:
        return len(self.hits) - self.resolved

    @property
    def requests(self) -> int:
        """How many pages were asked -- two per account since 2026-09-25, plus the open ones."""
        return sum(group.pages for group in self.groups) + self.extra_requests

    def group_lines(self) -> list[str]:
        """One line per group, above the summary: the account (or the list) and its number.

        An account that answered with nothing keeps its line: "was my account asked at all" is
        exactly the question this change exists to answer, and a silent absence answers it wrong.
        """
        return [*self.notes, *(group.line() for group in self.groups)]

    def summary_line(self) -> str:
        """The one line under the groups, and what `search.json` is read against."""
        return (
            f"{len(self.hits)} repositories, {self.resolved} resolved, "
            f"{self.unresolved} unresolved, {self.requests} requests, "
            f"budget {self.budget_used}/{self.budget_limit}"
        )

    def _group_class(self, label: str, classes: Mapping[str, str]) -> str:
        """What the search asked this group by: its account class, else the plan it belongs to.

        A typed repository id and a catalog page are asked without an account, and calling either
        one `open` would say the two open lists were asked (found in the live probe of 2026-09-25,
        where the screen then read "and the two open lists" for a run that asked neither).
        """
        if self.mode == "id":
            return "typed"
        if self.mode == "catalog":
            return "catalog"
        return classes.get(label, "open")

    def search_log(self, *, filtered: bool, run_at: datetime, unresolved: dict[str, int]) -> dict:
        """What one search did, as the data `search.json` holds (CONTRACTS.md, "Search log").

        The account lines, the request count and the budget left the screen in the test round of
        2026-09-24: they answer "why did this account answer nothing", a question about a file and
        not about the line a reader is looking at. The file and the screen's two notes both come
        from this one reading of the search.
        """
        classes = {n: "packager" for n in self.listed_packagers} | {n: "publisher" for n in self.publishers}
        return {
            "schema_version": SEARCH_LOG_SCHEMA_VERSION,
            "word": self.query,
            "mode": self.mode,
            "filter_owners": filtered,
            "run_at": run_at.isoformat(),
            "accounts": [
                {
                    "account": g.label,
                    "class": self._group_class(g.label, classes),
                    "hits": g.page_size,
                    "page_full": g.page_full,
                }
                for g in self.groups
            ],
            "requests": self.requests,
            "budget": {"used": self.budget_used, "limit": self.budget_limit},
            "resolved": self.resolved,
            "unresolved": [{"reason": reason, "count": count} for reason, count in sorted(unresolved.items())],
        }


@dataclass
class _Resolution:
    """What the resolution of one search carries from page to page.

    `seen` holds every repository id an earlier page already answered with, `ages` one verdict per
    distinct base model, so neither costs a second request. `publisher_required` is the owner
    filter's half of the resolution: with the filter on, only a base model of a catalog publisher
    resolves; with it off, any account's base model does. `packagers` keeps the order of the
    positive list, because the order the accounts are asked in is the order the groups are shown in.
    """

    catalog: Catalog
    packagers: tuple[str, ...]
    publisher_required: bool
    ages: dict[str, AgeVerdict] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)


def run_search(
    transport: Transport,
    name: str,
    *,
    catalog: Catalog,
    listed_packagers: Sequence[str] = DEFAULT_PACKAGERS,
    budget: RequestBudget | None = None,
    ollama_entries: Mapping[str, str] | None = None,
    open_pages: bool = False,
    publisher_required: bool = True,
) -> SearchOutcome:
    """Search for whatever was typed and return every repository answered, resolved or not.

    The input decides the pages (`search_word.parse_search`, `search_pages`):

    - a **repository id** is one request for that repository, shown under the label `typed` and
      resolved without the publisher gate -- the person named it, so it is always selectable. A
      repository with no GGUF file, and one the Hub does not answer for, leave a note and fall
      through to the word search over the id's name half;
    - **words** ask two pages per account -- the catalog's matching publisher accounts first, then
      the positive list -- and, when `open_pages` is set, two more without an account. The guided
      mode passes `open_pages=not filtered`, so the open lists are what switching the owner filter
      *off* adds; they never replace the account groups. More than one word is sent to the Hub as
      one (`SearchWord.hub_word`) and the answer is held against all of them locally;
    - **no word at all** asks one page per current model of the catalog.

    On top of that one age lookup per *distinct* resolved base model (`decide_age`); a hit costs no
    request of its own. A repository more than one page answers with is listed once, in the first
    group that had it. A page that fails ends the search with `SearchError` naming the account --
    an account is never quietly left out, because "unsloth has nothing for this word" and "unsloth
    was not asked" are different answers.
    """
    shared = budget if budget is not None else RequestBudget(DEFAULT_GUIDED_BUDGET)
    redirecting = RedirectingTransport(BudgetedTransport(transport, shared))
    word = parse_search(name)
    state = _Resolution(catalog, tuple(listed_packagers), publisher_required)
    notes: list[str] = []
    typed_group = _typed_group(redirecting, word, state, notes)
    # The typed page is asked whenever the input is an id; without a group it still cost a request.
    asked_typed = word.form == "id" and word.repo_id is not None
    plan = SearchPlan("id", ()) if typed_group is not None else _plan_for(word, state, open_pages=open_pages)
    publishers = publishers_for(catalog, word) if plan.mode == "word" else []
    notes += _plan_notes(plan, word, publishers, open_pages=open_pages)
    groups = [typed_group] if typed_group is not None else []
    groups += _answer_plan(redirecting, plan, word, state, shared, notes)
    hits = _with_ollama_entries([hit for group in groups for hit in group.hits], ollama_entries or {})
    return SearchOutcome(
        query=word.text,
        hits=hits,
        budget_used=shared.used,
        budget_limit=shared.limit,
        groups=_regrouped(groups, hits),
        notes=notes,
        publishers=list(publishers), listed_packagers=list(listed_packagers),
        mode=plan.mode,
        typed=word.repo_id if typed_group is not None else None,
        candidates=_candidates(word, state, plan),
        extra_requests=1 if asked_typed and typed_group is None else 0,
    )


def _plan_for(word: SearchWord, state: _Resolution, *, open_pages: bool) -> SearchPlan:
    """The pages an input that is not a reachable repository id leads to."""
    if word.form == "empty":
        return catalog_plan(latest_model_names(state.catalog))
    return word_plan(word, accounts_for(state.catalog, word, state.packagers), open_pages=open_pages)


def _plan_notes(plan: SearchPlan, word: SearchWord, publishers: Sequence[str], *, open_pages: bool) -> list[str]:
    """What this search has to say about the *request* rather than about a repository."""
    if plan.mode == "catalog":
        return [NO_WORD_NOTE]
    if plan.mode == "word" and not publishers:
        note = NO_PUBLISHER_NOTE_OPEN if open_pages else NO_PUBLISHER_NOTE
        return [note.format(word=word.text)]
    return []


def _candidates(word: SearchWord, state: _Resolution, plan: SearchPlan) -> list[str]:
    """What the input could have meant, when the catalog names nothing that it does.

    Only for a word search: a typed id named its repository, and a search with no word is not a
    guess at all. A word the catalog already knows gets no candidates either -- a correct word
    with no answer today is a fact about the Hub, not a spelling mistake.
    """
    if plan.mode != "word" or catalog_models(state.catalog, word) or publishers_for(state.catalog, word):
        return []
    return close_matches(state.catalog, word)


def _answer_plan(
    transport: Transport,
    plan: SearchPlan,
    word: SearchWord,
    state: _Resolution,
    shared: RequestBudget,
    notes: list[str],
) -> list[SearchGroup]:
    """Every group of the plan, asked in order; the catalog pages stop before the budget is gone.

    The catalog pages are the one plan whose length the input does not bound -- one per current
    model of the catalog -- and the fetch of a chosen model shares this budget. So that plan stops
    asking once `CATALOG_BUDGET_RESERVE` requests or fewer are left and says how far it got,
    rather than ending a run that has a choice to offer with `SearchError`. It is a stop rule before
    a page, not a fence around the reserve: the last page it asks resolves its own repositories, and
    those age lookups spend from the reserve too.
    """
    requests = plan.all_requests()
    groups: list[SearchGroup] = []
    asked = 0
    for index, group in enumerate(plan.groups):
        if plan.mode == "catalog" and shared.remaining <= CATALOG_BUDGET_RESERVE:
            notes.append(CATALOG_BUDGET_NOTE.format(asked=index, planned=len(plan.groups)))
            break
        asked += len(group.requests)
        groups.append(_fetch_group(transport, state, word, group, requests[asked:]))
    return groups


def _typed_group(
    transport: Transport, word: SearchWord, state: _Resolution, notes: list[str]
) -> SearchGroup | None:
    """The one group a typed repository id leads to, or `None` with a note saying why not.

    Resolved without the publisher gate (`publisher_required=False`): the person named this
    repository, so `publisher_unknown` is no reason to hide it, and the owner class stays `other`
    where neither list holds the account. The GGUF file is read off the repository's own `gguf`
    tag -- the same fact the word search filters by -- so this stays one request.
    """
    if word.form != "id" or word.repo_id is None:
        return None
    entry = _typed_entry(transport, word.repo_id)
    if entry is None:
        notes.append(UNREADABLE_ID_NOTE.format(repo=word.repo_id))
        return None
    tags = entry.get("tags")
    if not isinstance(tags, list) or GGUF_TAG not in tags:
        notes.append(NO_GGUF_NOTE.format(repo=word.repo_id))
        return None
    state.seen.add(word.repo_id)
    hit = _build_hit(transport, state, word.repo_id, entry, publisher_required=False)
    return SearchGroup(label=TYPED_LABEL, hits=[hit], page_size=1, page_full=False, already_listed=0)


def _typed_entry(transport: Transport, repo: str) -> dict | None:
    """One typed repository as an answer entry, or `None` when it is not this repository's answer.

    The answer has to name `repo` itself (`id`, or `modelId`): the transport follows redirects and
    the Hub answers a moved repository's path with the repository it moved to, so without that
    check a request for `acme/B` could be satisfied by `other/B`. A budget that ends here is the
    one failure that ends the search -- it says nothing about the repository.
    """
    where = f"search for {repo!r} ({TYPED_LABEL})"
    try:
        response = transport("GET", model_url(repo))
    except BudgetExhaustedError as exc:
        raise SearchError(f"{where} not made: {exc}") from exc
    except Exception:  # noqa: BLE001 -- an unreadable id falls back to the word search
        return None
    if response.status != 200:
        return None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 -- so does an answer that is no JSON
        return None
    if not isinstance(payload, dict) or repo not in (payload.get("id"), payload.get("modelId")):
        return None
    return {**payload, "id": repo}


def _fetch_group(
    transport: Transport,
    state: _Resolution,
    word: SearchWord,
    group: GroupPlan,
    pending: Sequence[SearchRequest],
) -> SearchGroup:
    """One group's pages, asked and resolved; what an earlier page already had is counted only.

    A repository is held against `state.seen` and added to it **before** it is resolved, entry by
    entry. Resolving it first and filtering afterwards would list a repository twice when one page
    carries it twice -- the answer is registry-controlled text, and nothing promises it holds each
    id once -- and would spend an age lookup on a duplicate whose declared base model differs from
    the one the first page declared (found in the second-model round, 2026-09-24).

    A repository the words of a multi-word input do not name is counted and dropped here, before
    it costs an age lookup: the Hub was asked with one of those words and answers with everything
    that word matches.
    """
    entries, page_full = _group_pages(transport, word, group, pending)
    fresh: list[SearchHit] = []
    already_listed = 0
    filtered_out = 0
    for entry in entries:
        repo = _entry_repo(word.text, group.label, entry)
        if repo in state.seen:
            already_listed += 1
            continue
        state.seen.add(repo)
        if word.filters_locally and not word.matches(repo):
            filtered_out += 1
            continue
        fresh.append(_build_hit(transport, state, repo, entry, publisher_required=state.publisher_required))
    return SearchGroup(
        label=group.label,
        hits=fresh,
        page_size=len(entries),
        page_full=page_full,
        already_listed=already_listed,
        pages=len(group.requests),
        filtered_out=filtered_out,
    )


def _group_pages(
    transport: Transport, word: SearchWord, group: GroupPlan, pending: Sequence[SearchRequest]
) -> tuple[list[dict], bool]:
    """Every entry this group's pages answered with, and whether one of them was as long as it may be.

    Only an **account** group can be "full" in the sense the note is about: there may be more
    under this account than one page shows. An open list and a catalog page are as long as they
    are by definition.
    """
    entries: list[dict] = []
    page_full = False
    for position, request in enumerate(group.requests):
        rest = [*group.requests[position + 1 :], *pending]
        page = _fetch_search_page(transport, word.text, request, rest)
        entries += page
        page_full = page_full or (request.account is not None and len(page) >= request.limit)
    return entries, page_full


def _entry_repo(word: str, label: str, entry: dict) -> str:
    """One answer entry's repository id, or `SearchError` naming the page it came from.

    The page belongs in the message: an answer of twenty entries under one of seven accounts is
    otherwise a malformed id with no address, and a reader cannot tell which request to look at.
    """
    repo = entry.get("id")
    where = f"search for {word!r} ({label})"
    if not isinstance(repo, str):
        raise SearchError(f"{where}: a result has no repository id: {entry!r}")
    try:
        validate_hf_repo(repo)
    except ValueError as exc:
        raise SearchError(f"{where}: a result is not a repository id: {exc}") from exc
    return repo


def _regrouped(groups: list[SearchGroup], hits: list[SearchHit]) -> list[SearchGroup]:
    """The same groups carrying the final hit objects, after a typed Ollama name was applied.

    `_with_ollama_entries` builds new `SearchHit` values, so the groups would otherwise hold the
    ones from before that step and show a different Ollama name than the flat list does. It keeps
    the order and the length of the list it is given, so each group takes its own slice back --
    no lookup by repository id, and therefore nothing that could miss one.
    """
    regrouped: list[SearchGroup] = []
    start = 0
    for group in groups:
        end = start + len(group.hits)
        regrouped.append(replace(group, hits=hits[start:end]))
        start = end
    return regrouped


def _with_ollama_entries(hits: list[SearchHit], ollama_entries: Mapping[str, str]) -> list[SearchHit]:
    """Let a typed Ollama name win over the catalog's, for every hit of the same base model.

    A typed name answers "which Ollama package is this model", not "which package is this one
    repository", so it applies to every resolved hit of that base model -- otherwise the list
    would show two different Ollama packages for one model, and which one reached the
    configuration would depend on the order of the hits. Two typed names for the same base model
    that differ are contradictory input and end the search, rather than one of them quietly
    winning.
    """
    typed: dict[str, str] = {}
    for hit in hits:
        name = ollama_entries.get(hit.repo)
        if name is None or not hit.resolved or hit.resolved_base_model is None:
            continue
        validate_ollama_name(name)
        previous = typed.setdefault(hit.resolved_base_model, name)
        if previous != name:
            raise SearchError(
                f"{hit.resolved_base_model}: two different Ollama names were entered "
                f"({previous}, {name}); a base model has one"
            )
    if not typed:
        return hits
    return [
        hit.model_copy(update={"ollama": typed[hit.resolved_base_model]})
        if hit.resolved and hit.resolved_base_model in typed
        else hit
        for hit in hits
    ]


def _fetch_search_page(
    transport: Transport, name: str, request: SearchRequest, pending: Sequence[SearchRequest]
) -> list[dict]:
    """One page as a list of repository objects, or `SearchError` naming the page that failed.

    `pending` are the pages after this one, so a budget that ends here can say how many accounts
    were never asked -- the number a user needs to know that the list in front of them is short
    for a reason that has nothing to do with the Hub.
    """
    where = f"search for {name!r} ({request.label})"
    try:
        response = transport("GET", request.url)
    except BudgetExhaustedError as exc:
        raise SearchError(f"{where} not made: {exc}; {_pending_note(request, pending)}") from exc
    except Exception as exc:  # a transport failure is the user's to see, never swallowed
        raise SearchError(f"{where} failed: {exc}") from exc
    if response.status != 200:
        raise SearchError(f"{where}: unexpected status {response.status}")
    try:
        payload = response.json()
    except Exception as exc:
        raise SearchError(f"{where}: invalid JSON: {exc}") from exc
    if not isinstance(payload, list) or not all(isinstance(entry, dict) for entry in payload):
        raise SearchError(f"{where}: the answer is not a list of repositories")
    return payload


def _pending_note(request: SearchRequest, pending: Sequence[SearchRequest]) -> str:
    """How many pages are still unasked, this one included, so a short list is explained.

    Each account **once**, however many of its pages are left: a reader wants to know which
    accounts have no answer, and an account asked twice would otherwise be named twice (found
    while the second page per account was added, 2026-09-25).
    """
    accounts: list[str] = []
    for page in (request, *pending):
        if page.account is not None and page.account not in accounts:
            accounts.append(page.account)
    if not accounts:
        return f"{1 + len(pending)} pages were not asked"
    return f"{len(accounts)} accounts were not asked: {', '.join(accounts)}"


def _build_hit(
    transport: Transport,
    state: _Resolution,
    repo: str,
    entry: dict,
    *,
    publisher_required: bool,
) -> SearchHit:
    """One answer entry as a `SearchHit`; `repo` is its id, already validated by `_entry_repo`.

    `publisher_required` is the owner filter's half of the resolution (decided 2026-09-25): with
    the filter on, a base model whose account the catalog does not name as a publisher stays
    unresolved with `publisher_unknown`; with the filter off, and for a typed repository id, it
    resolves and its owner class says `other`. The class alone places the account -- the catalog is
    a preference, never a verdict.
    """
    catalog = state.catalog
    ages = state.ages
    tags = entry.get("tags")
    card_data = entry.get("cardData")
    bases, relation = _declared(tags, card_data)
    # The declared base is registry-controlled text. Only exactly one declared base that is a
    # well-formed repository id can carry a relation at all; anything else is asked against a
    # marker no repository can be, so `check_relation` still reports a conflict when it sees one
    # and otherwise says `base_model_tag`.
    usable_base = bases[0] if len(bases) == 1 and _is_repo_id(bases[0]) else None
    status, _reason = check_relation(tags, card_data, usable_base or _NO_SINGLE_BASE)

    common = {
        "repo": repo,
        "publisher_status": owner_class(
            repo.split("/", 1)[0], catalog=catalog, listed_packagers=state.packagers
        ),
        "base_model": bases or None,
        "base_model_relation": relation,
        "repo_created_at": _created_at(entry.get("createdAt")),
        "downloads": read_downloads(entry.get("downloads")),
        "parameters_b": _parameters_b(entry.get("safetensors")),
        "license": _license(card_data),
    }
    if usable_base is None:
        reason = status if status == "metadata_conflict" else "base_model_tag"
        return SearchHit(resolved=False, unresolved_reason=reason, **common)
    if status != "quantized":
        return SearchHit(resolved=False, unresolved_reason=status, **common)

    base = usable_base
    if publisher_required and not catalog.is_publisher(base.split("/", 1)[0]):
        return SearchHit(resolved=False, unresolved_reason="publisher_unknown", **common)

    if base not in ages:
        ages[base] = decide_age(transport, catalog, base)
    verdict = ages[base]
    return SearchHit(
        resolved=True,
        unresolved_reason=None,
        resolved_base_model=base,
        age=verdict.age,
        successor=verdict.successor,
        ollama=_catalog_ollama(catalog, base),
        **common,
    )


def _is_repo_id(value: str) -> bool:
    """Whether `value` is a repository id this module is willing to put in a URL.

    The shared `hf_repo` rule (`contracts.validate_hf_repo`) is the first gate, but it is a
    character-class rule and therefore also accepts a pure path segment: `Qwen/..`, `Qwen/.`,
    `../x` all pass it (measured). Here the value is registry-controlled *and* becomes a path
    segment of the age lookup's URL, so both halves additionally have to start with an
    alphanumeric character and be free of `.`/`..` on their own -- which is the Hub's own rule for
    an account and a repository name.
    """
    try:
        validate_hf_repo(value)
    except ValueError:
        return False
    owner, _, name = value.partition("/")
    return all(_REPO_SEGMENT_RE.fullmatch(part) for part in (owner, name))


def _declared(tags: object, card_data: object) -> tuple[list[str], str]:
    """The base models and the relation a repository *claims*, read leniently for display.

    `check_relation` stays the authority on whether those claims hold together; this only
    gathers them, so a malformed field yields nothing here and the verdict there.
    """
    bases: set[str] = set()
    relations: set[str] = set()
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str) or not tag.startswith("base_model:"):
                continue
            rest = tag[len("base_model:") :]
            relation, separator, _target = rest.partition(":")
            if separator:
                relations.add(relation)
            else:
                bases.add(rest)
    if isinstance(card_data, dict):
        card_base = card_data.get("base_model")
        if isinstance(card_base, str):
            bases.add(card_base)
        elif isinstance(card_base, list):
            bases.update(entry for entry in card_base if isinstance(entry, str))
        card_relation = card_data.get("base_model_relation")
        if isinstance(card_relation, str):
            relations.add(card_relation)
    relation_text = relations.pop() if len(relations) == 1 else UNKNOWN
    return sorted(bases), relation_text


def owner_class(
    owner: str, *, catalog: Catalog, listed_packagers: Sequence[str] = DEFAULT_PACKAGERS
) -> str:
    """Where an account stands: `publisher`, `listed packager` or `other`.

    Three neutral classes, never a judgement of the account and never a statement about the
    quality of what it packages: `publisher` means the catalog names it as the publisher of a
    family, `listed packager` that it is on the positive list of packager accounts, `other`
    that neither is true -- which says nothing more than that. The same class labels a search
    hit's owner and a fetch target's owner, so a reader sees one vocabulary.
    """
    if catalog.is_publisher(owner):
        return "publisher"
    return "listed packager" if owner in listed_packagers else "other"


def _created_at(value: object) -> datetime | None:
    """`createdAt` as an aware UTC instant; anything that is not one reads as `None`.

    The conversion is inside the guard as well, not only the parsing: a timestamp at either end
    of the calendar (`0001-01-01T00:00:00+01:00`) parses and then overflows when it is shifted to
    UTC, which must be "not a date I can carry", never an exception out of the search.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def _parameters_b(safetensors: object) -> float | None:
    """`safetensors.total` in billions, only when it is a real, finite, positive number."""
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or total <= 0:
        return None
    try:
        result = total / 1e9
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _license(card_data: object) -> str:
    if isinstance(card_data, dict) and isinstance(card_data.get("license"), str):
        return card_data["license"]
    return UNKNOWN


def _catalog_ollama(catalog: Catalog, hf_repo: str) -> str | None:
    model = catalog.model_for(hf_repo)
    if model is None or model.ollama_base is None or model.ollama_tag is None:
        return None
    return f"{model.ollama_base}:{model.ollama_tag}"


def parse_ollama_entry(line: str) -> str:
    """Read the Ollama name out of a typed `ollama: <name:tag>` line; raise on anything else."""
    text = line.strip()
    if not text.lower().startswith(_OLLAMA_ENTRY_PREFIX):
        raise ValueError(f"an Ollama entry starts with 'ollama:': {line!r}")
    return validate_ollama_name(text[len(_OLLAMA_ENTRY_PREFIX) :].strip())


def ollama_label(hit: SearchHit) -> str:
    """What the selection list shows in the Ollama column: the name, or `none known`."""
    return hit.ollama if hit.ollama is not None else NO_OLLAMA_LABEL


__all__ = [
    "CATALOG_BUDGET_NOTE",
    "CATALOG_BUDGET_RESERVE",
    "DEFAULT_GUIDED_BUDGET",
    "DEFAULT_PACKAGERS",
    "GGUF_TAG",
    "MAX_SUCCESSOR_EDGES",
    "NO_GGUF_NOTE",
    "NO_OLLAMA_LABEL",
    "NO_PUBLISHER_NOTE",
    "NO_PUBLISHER_NOTE_OPEN",
    "NO_WORD_NOTE",
    "SEARCH_LOG_SCHEMA_VERSION",
    "UNREADABLE_ID_NOTE",
    "AgeVerdict",
    "SearchError",
    "SearchGroup",
    "SearchOutcome",
    "apply_hits",
    "decide_age",
    "family_name_for",
    "ollama_label",
    "owner_class",
    "parse_ollama_entry",
    "run_search",
    "write_configuration",
]
