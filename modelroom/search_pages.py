"""The pages one search asks for, and which of them a typed input leads to.

Decided 2026-09-24, extended 2026-09-25. Until 2026-09-24 the search asked the Hub once, for the
50 most recently created GGUF repositories matching the word (`sort=createdAt`), and the positive
list of packager accounts only *classified* what that one answer happened to contain. Measured in
an empty folder with `qwen` and with `deepseek`: the answer was 50 third-party uploads of the last
few days, and not one repository of `unsloth`, `bartowski` or `Qwen` -- those accounts are older
and fall out of the window. Nothing was selectable, twice. So the account steers the request.

**Two pages per account since 2026-09-25**, the newest twenty and the twenty most downloaded.
Measured live that day: `author=unsloth&search=qwen&sort=createdAt&limit=20` answers with twenty
repositories created after 2026-05, and `unsloth/Qwen3.5-9B-GGUF` (2026-02-28) is not among them;
the same account's `sort=downloads` page carries it in fourth place. One sort order alone therefore
cannot find the plain build of a model whose account publishes often, whatever word is typed.

Three kinds of plan, one per form of the input (`search_word.parse_search`):

1. **words** -- two pages per account: every publisher account of a catalog family the input
   matches and every account of the positive list; and only with the owner filter off two more
   without an account, the twenty most downloaded and the ten newest for that word.
2. **an id** -- one page: that repository itself.
3. **no word at all** -- one page per current model of the catalog, its ten most downloaded GGUF
   repositories.

The request forms are pinned literals (`tests/test_search_pages.py`), because the fixtures of the
test suite are recorded against exactly them.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Literal, Sequence

from .catalog import Catalog
from .guided_contracts import UNKNOWN, SearchHit
from .search_word import SearchWord, family_matches, parse_search

HF_API = "https://huggingface.co/api"
# One account's page. 20 is a compromise measured against the live Hub on 2026-09-24: `unsloth`
# answers `qwen` with far more than 20 repositories, and a longer page makes the selection list
# unreadable rather than more useful -- the summary says when a page was full, so a full page is
# an invitation to use a more specific word, not a silent truncation.
HF_ACCOUNT_LIMIT = 20
# The open `sort=downloads` page. 20 since 2026-09-25: with the owner filter off, the most
# downloaded repositories of a word are the only place a model of an account nobody listed can
# show up at all, and ten of them are mostly one family's siblings.
HF_MOST_DOWNLOADED_LIMIT = 20
# The open `sort=createdAt` page. Ten is what a person reads before deciding, and "the newest" is
# a look at what appeared this week, not a list to choose from.
HF_OPEN_LIMIT = 10
# One page of a search with no word at all: the current models of the catalog are asked one by
# one, so ten per model keeps the whole run inside the shared request budget.
HF_CATALOG_LIMIT = 10
# Every field the resolution reads, repeated once each -- which is how the Hub's model API takes
# `expand`. `downloads` is new since 2026-09-24 and is only answered when it is asked for
# explicitly, the moment `expand` is set at all.
_EXPAND = ("cardData", "createdAt", "downloads", "safetensors", "tags")
MOST_DOWNLOADED_LABEL = "most downloaded"
NEWEST_LABEL = "newest"
# What the one page of a typed repository id is shown under. It is no account page: the person
# named the repository, so the answer is about that one repository.
TYPED_LABEL = "typed"
SearchMode = Literal["word", "id", "catalog"]
# Below this many downloads the plain number is shown, above it a shortened `k`/`M` form: `3551`,
# `68k`, `12.0M` (the three shapes measured live on 2026-09-24). A number is a statement about
# what many people take, never a rank -- the ranking rule alone orders the result.
_DOWNLOADS_PLAIN_BELOW = 10_000


@dataclass(frozen=True)
class SearchRequest:
    """One page of a search: the label its group carries, the URL, and how many it may answer.

    `single` marks the one form whose answer is a repository **object** and not a list of them
    (the typed id's own page), so the reader knows what shape to expect before it looks.
    """

    label: str
    url: str
    limit: int
    account: str | None
    single: bool = False


@dataclass(frozen=True)
class GroupPlan:
    """One label of the answer and the pages that fill it -- two for an account since 2026-09-25.

    The label is what the summary line names, so the two pages of one account stay one group: a
    reader asks "was my account asked", once, and the two sort orders are this package's business.
    """

    label: str
    account: str | None
    requests: tuple[SearchRequest, ...]


@dataclass(frozen=True)
class SearchPlan:
    """Every page one search asks, grouped, and which form of input it came from."""

    mode: SearchMode
    groups: tuple[GroupPlan, ...]

    @property
    def requests(self) -> int:
        return sum(len(group.requests) for group in self.groups)

    def all_requests(self) -> list[SearchRequest]:
        """Every page in the order it is asked, flat -- for the note a spent budget leaves."""
        return [request for group in self.groups for request in group.requests]


@dataclass(frozen=True)
class SearchGroup:
    """One answered label: the hits it contributed, and what its pages looked like.

    `hits` holds only the repositories this group contributed to the flat list -- a repository an
    earlier group already listed stays in that earlier group and is counted in `already_listed`
    here. `page_size` is what its pages really answered with together, so `page_full` can say that
    a more specific word would shorten the list rather than pretending the page was everything.
    `pages` is how many requests this one label cost.

    `page_full` is set for an **account** group only, and only when one of its pages was as long
    as it may be. An open list is as long as it is by definition, so "this page was full" says
    nothing about the word there and the note would be noise on every single search.
    """

    label: str
    hits: list[SearchHit]
    page_size: int
    page_full: bool
    already_listed: int
    pages: int = 1
    # How many of this group's repositories the words of a multi-word input do not name. The Hub
    # takes one string as `search=`, so the pages are asked with one word and the rest of the
    # words are held against the answer here -- and a reader is told how much that removed.
    filtered_out: int = 0

    def line(self) -> str:
        """The one line the summary prints for this group: the label and its numbers."""
        parts = [f"{self.label} {self.page_size}"]
        if self.already_listed:
            parts.append(f"{self.already_listed} of them already listed")
        if self.filtered_out:
            parts.append(f"{self.filtered_out} of them other models")
        text = ", ".join(parts)
        if self.page_full:
            return f"{text} (page full: a more specific word shortens the list)"
        return text


def _expanded(query: str, limit: int) -> str:
    return f"{HF_API}/models?{query}&limit={limit}" + "".join(f"&expand={field}" for field in _EXPAND)


def account_search_url(name: str, account: str) -> str:
    """The pinned request for one account: its newest GGUF repositories matching `name`.

    `author=` is an exact account, not a word search, so this is what "show me what unsloth has"
    means. `sort=createdAt` keeps the newest of that account first.
    """
    query = urllib.parse.urlencode(
        {"author": account, "search": name, "filter": "gguf", "sort": "createdAt", "direction": -1}
    )
    return _expanded(query, HF_ACCOUNT_LIMIT)


def account_downloads_url(name: str, account: str) -> str:
    """The pinned second request for one account: its most downloaded GGUF repositories.

    The page the newest one cannot replace: measured live 2026-09-25, `unsloth` has published more
    than twenty `qwen` repositories since `Qwen3.5-9B-GGUF` was created, so the plain build of a
    listed model is reachable only by what many people take.
    """
    query = urllib.parse.urlencode(
        {"author": account, "search": name, "filter": "gguf", "sort": "downloads", "direction": -1}
    )
    return _expanded(query, HF_ACCOUNT_LIMIT)


def most_downloaded_url(name: str) -> str:
    """The pinned open request: the most downloaded GGUF repositories matching `name`."""
    query = urllib.parse.urlencode({"search": name, "filter": "gguf", "sort": "downloads", "direction": -1})
    return _expanded(query, HF_MOST_DOWNLOADED_LIMIT)


def newest_url(name: str) -> str:
    """The pinned open request: the newest GGUF repositories matching `name`."""
    query = urllib.parse.urlencode({"search": name, "filter": "gguf", "sort": "createdAt", "direction": -1})
    return _expanded(query, HF_OPEN_LIMIT)


def catalog_page_url(model_name: str) -> str:
    """The pinned request for one current model of the catalog: its most downloaded GGUF builds."""
    query = urllib.parse.urlencode(
        {"search": model_name, "filter": "gguf", "sort": "downloads", "direction": -1}
    )
    return _expanded(query, HF_CATALOG_LIMIT)


def model_url(repo: str) -> str:
    """The pinned request for one typed repository id: that repository, with the same five fields.

    One request, and the answer is the repository object itself rather than a list -- so a person
    who knows the id gets that repository whether or not any word search would have found it.
    """
    return f"{HF_API}/models/{repo}?" + "&".join(f"expand={field}" for field in _EXPAND)


def publisher_accounts(catalog: Catalog, name: str) -> list[str]:
    """The publisher accounts of every catalog family the input matches, catalog order, once each.

    A family matches when the input names its name, its publisher account or one of its model ids
    (`search_word.family_matches`). So `qwen` asks `Qwen`, `deepseek` asks `deepseek-ai`, `r1` asks
    the account of the family that carries an `...R1...` model, and `qwen 3.5 9b` asks `Qwen`
    because one of its models is `Qwen/Qwen3.5-9B`.
    """
    return publishers_for(catalog, parse_search(name))


def publishers_for(catalog: Catalog, word: SearchWord) -> list[str]:
    """The same, for an input that is already parsed."""
    accounts: list[str] = []
    for family in catalog.families:
        if family_matches(family, word) and family.publisher not in accounts:
            accounts.append(family.publisher)
    return accounts


def search_accounts(catalog: Catalog, name: str, listed_packagers: Sequence[str]) -> list[str]:
    """Every account this search asks: the matching publishers first, then the positive list.

    An account on both lists is asked once. The order is the order the groups are shown in, so
    the model a person searched for stands above the packagers that repackage it.
    """
    return accounts_for(catalog, parse_search(name), listed_packagers)


def accounts_for(catalog: Catalog, word: SearchWord, listed_packagers: Sequence[str]) -> list[str]:
    """The same, for an input that is already parsed."""
    accounts = publishers_for(catalog, word)
    for packager in listed_packagers:
        if packager not in accounts:
            accounts.append(packager)
    return accounts


def word_plan(word: SearchWord, accounts: Sequence[str], *, open_pages: bool) -> SearchPlan:
    """The pages of a word search: two per account, then the two open ones with the filter off."""
    name = word.hub_word
    groups = [
        GroupPlan(
            label=account,
            account=account,
            requests=(
                SearchRequest(account, account_search_url(name, account), HF_ACCOUNT_LIMIT, account),
                SearchRequest(account, account_downloads_url(name, account), HF_ACCOUNT_LIMIT, account),
            ),
        )
        for account in accounts
    ]
    if open_pages:
        groups.append(_open_group(MOST_DOWNLOADED_LABEL, most_downloaded_url(name), HF_MOST_DOWNLOADED_LIMIT))
        groups.append(_open_group(NEWEST_LABEL, newest_url(name), HF_OPEN_LIMIT))
    return SearchPlan(mode="word", groups=tuple(groups))


def _open_group(label: str, url: str, limit: int) -> GroupPlan:
    return GroupPlan(label=label, account=None, requests=(SearchRequest(label, url, limit, None),))


def typed_plan(repo: str) -> SearchPlan:
    """The one page of a typed repository id."""
    request = SearchRequest(TYPED_LABEL, model_url(repo), 1, None, single=True)
    return SearchPlan(mode="id", groups=(GroupPlan(label=TYPED_LABEL, account=None, requests=(request,)),))


def catalog_plan(model_names: Sequence[str]) -> SearchPlan:
    """One page per current model of the catalog, each under the model's own name as its label."""
    groups = [
        GroupPlan(
            label=name,
            account=None,
            requests=(SearchRequest(name, catalog_page_url(name), HF_CATALOG_LIMIT, None),),
        )
        for name in model_names
    ]
    return SearchPlan(mode="catalog", groups=tuple(groups))


def read_downloads(value: object) -> int | None:
    """`downloads` as a count, or `None` when the field is no count at all.

    Registry-controlled text: a `bool` is not a count (and `True` would otherwise read as `1`), a
    float is not one either, and a negative number is not a number of downloads. Anything that is
    not a whole number at or above zero is `unknown`, never a guess.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def format_downloads(value: int | None) -> str:
    """The downloads column: `3551`, `68k`, `12.0M`, or `unknown` when there is no number.

    The thousands are floored, never rounded: rounding turns `999999` into `1000k`, a shape this
    column does not have. Flooring keeps every value in exactly one of the three shapes.

    Whole-number arithmetic throughout, no float anywhere: the count is registry-controlled text
    and a Python integer has no upper bound, so `value / 1_000_000` raises `OverflowError` for a
    number the Hub could answer with, and one column of one line would end the whole dialog.
    """
    if value is None:
        return UNKNOWN
    if value < _DOWNLOADS_PLAIN_BELOW:
        return str(value)
    if value < 1_000_000:
        return f"{value // 1000}k"
    millions, tenths = divmod(value // 100_000, 10)
    return f"{millions}.{tenths}M"


__all__ = [
    "HF_ACCOUNT_LIMIT",
    "HF_CATALOG_LIMIT",
    "HF_MOST_DOWNLOADED_LIMIT",
    "HF_OPEN_LIMIT",
    "MOST_DOWNLOADED_LABEL",
    "NEWEST_LABEL",
    "TYPED_LABEL",
    "GroupPlan",
    "SearchGroup",
    "SearchMode",
    "SearchPlan",
    "SearchRequest",
    "account_downloads_url",
    "account_search_url",
    "accounts_for",
    "catalog_page_url",
    "catalog_plan",
    "format_downloads",
    "model_url",
    "most_downloaded_url",
    "newest_url",
    "publisher_accounts",
    "publishers_for",
    "read_downloads",
    "search_accounts",
    "typed_plan",
    "word_plan",
]
