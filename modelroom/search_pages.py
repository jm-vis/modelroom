"""The pages one search asks for: one request per account, plus two open lists without one.

Decided 2026-09-24. Until then the search asked the Hub once, for the 50 most recently created
GGUF repositories matching the word (`sort=createdAt`), and the positive list of packager
accounts only *classified* what that one answer happened to contain. Measured in an empty folder
with `qwen` and with `deepseek`: the answer was 50 third-party uploads of the last few days, and
not one repository of `unsloth`, `bartowski` or `Qwen` -- those accounts are older and fall out of
the window. Nothing was selectable, twice.

So the account steers the request now:

1. one request per account, `author=<account>`, for every publisher account of the catalog whose
   family matches the word and for every account of the positive list. Age plays no part: each
   account answers with its own repositories, however old they are.
2. only when the owner filter is off, two more requests without an account -- the ten most
   downloaded and the ten newest GGUF repositories for that word -- so that a fresh fine-tune
   under an account nobody listed is visible at all.

The three request forms are pinned literals (`tests/test_search_pages.py`), because the fixtures
of the test suite are recorded against exactly them.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Sequence

from .catalog import Catalog
from .guided_contracts import UNKNOWN, SearchHit

HF_API = "https://huggingface.co/api"
# One account's page. 20 is a compromise measured against the live Hub on 2026-09-24: `unsloth`
# answers `qwen` with far more than 20 repositories, and a longer page makes the selection list
# unreadable rather than more useful -- the summary says when a page was full, so a full page is
# an invitation to use a more specific word, not a silent truncation.
HF_ACCOUNT_LIMIT = 20
# One open page. Ten is what a person reads before deciding; two such pages (most downloaded,
# newest) cost two requests of the shared budget, which is why they are asked only when the owner
# filter is off.
HF_OPEN_LIMIT = 10
# Every field the resolution reads, repeated once each -- which is how the Hub's model API takes
# `expand`. `downloads` is new since 2026-09-24 and is only answered when it is asked for
# explicitly, the moment `expand` is set at all.
_EXPAND = ("cardData", "createdAt", "downloads", "safetensors", "tags")
MOST_DOWNLOADED_LABEL = "most downloaded"
NEWEST_LABEL = "newest"
# Below this many downloads the plain number is shown, above it a shortened `k`/`M` form: `3551`,
# `68k`, `12.0M` (the three shapes measured live on 2026-09-24). A number is a statement about
# what many people take, never a rank -- the ranking rule alone orders the result.
_DOWNLOADS_PLAIN_BELOW = 10_000


@dataclass(frozen=True)
class SearchRequest:
    """One page of a search: the label its group carries, the URL, and how many it may answer."""

    label: str
    url: str
    limit: int
    account: str | None


@dataclass(frozen=True)
class SearchGroup:
    """One answered page: the hits it contributed, and what the page itself looked like.

    `hits` holds only the repositories this page contributed to the flat list -- a repository an
    earlier page already listed stays in that earlier group and is counted in `already_listed`
    here. `page_size` is what the page really answered with, so `page_full` can say that a more
    specific word would shorten the list rather than pretending the page was everything.

    `page_full` is set for an **account** page only. An open list is ten by definition, so "this
    page was as long as it may be" says nothing about the word there and the note would be noise
    on every single search.
    """

    label: str
    hits: list[SearchHit]
    page_size: int
    page_full: bool
    already_listed: int

    def line(self) -> str:
        """The one line the summary prints for this group: the label and the number."""
        if self.already_listed:
            text = f"{self.label} {self.page_size}, {self.already_listed} of them already listed"
        else:
            text = f"{self.label} {self.page_size}"
        if self.page_full:
            return f"{text} (page full: a more specific word shortens the list)"
        return text


def _expanded(query: str, limit: int) -> str:
    return f"{HF_API}/models?{query}&limit={limit}" + "".join(f"&expand={field}" for field in _EXPAND)


def account_search_url(name: str, account: str) -> str:
    """The pinned request for one account: its newest GGUF repositories matching `name`.

    `author=` is an exact account, not a word search, so this is what "show me what unsloth has"
    means. `sort=createdAt` keeps the newest of that account first; the account's own age is
    irrelevant, which is the whole point of asking per account.
    """
    query = urllib.parse.urlencode(
        {"author": account, "search": name, "filter": "gguf", "sort": "createdAt", "direction": -1}
    )
    return _expanded(query, HF_ACCOUNT_LIMIT)


def most_downloaded_url(name: str) -> str:
    """The pinned open request: the most downloaded GGUF repositories matching `name`."""
    query = urllib.parse.urlencode({"search": name, "filter": "gguf", "sort": "downloads", "direction": -1})
    return _expanded(query, HF_OPEN_LIMIT)


def newest_url(name: str) -> str:
    """The pinned open request: the newest GGUF repositories matching `name`."""
    query = urllib.parse.urlencode({"search": name, "filter": "gguf", "sort": "createdAt", "direction": -1})
    return _expanded(query, HF_OPEN_LIMIT)


def publisher_accounts(catalog: Catalog, name: str) -> list[str]:
    """The publisher accounts of every catalog family the word matches, catalog order, once each.

    A family matches when the word is part of its name, of its publisher account, or of one of its
    model ids -- compared without regard to case. So `qwen` asks `Qwen`, `deepseek` asks
    `deepseek-ai`, and `r1` asks the account of the family that carries an `...R1...` model.
    """
    word = name.strip().casefold()
    accounts: list[str] = []
    for family in catalog.families:
        haystack = [family.name, family.publisher, *(model.hf_repo for model in family.models)]
        if word and any(word in text.casefold() for text in haystack):
            if family.publisher not in accounts:
                accounts.append(family.publisher)
    return accounts


def search_accounts(catalog: Catalog, name: str, listed_packagers: Sequence[str]) -> list[str]:
    """Every account this search asks: the matching publishers first, then the positive list.

    An account on both lists is asked once. The order is the order the groups are shown in, so
    the model a person searched for stands above the packagers that repackage it.
    """
    accounts = publisher_accounts(catalog, name)
    for packager in listed_packagers:
        if packager not in accounts:
            accounts.append(packager)
    return accounts


def search_requests(name: str, accounts: Sequence[str], *, open_pages: bool) -> list[SearchRequest]:
    """The pages this search asks for, in order: one per account, then the two open ones."""
    requests = [
        SearchRequest(label=account, url=account_search_url(name, account), limit=HF_ACCOUNT_LIMIT, account=account)
        for account in accounts
    ]
    if open_pages:
        requests.append(
            SearchRequest(
                label=MOST_DOWNLOADED_LABEL, url=most_downloaded_url(name), limit=HF_OPEN_LIMIT, account=None
            )
        )
        requests.append(SearchRequest(label=NEWEST_LABEL, url=newest_url(name), limit=HF_OPEN_LIMIT, account=None))
    return requests


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
    "HF_OPEN_LIMIT",
    "MOST_DOWNLOADED_LABEL",
    "NEWEST_LABEL",
    "SearchGroup",
    "SearchRequest",
    "account_search_url",
    "format_downloads",
    "most_downloaded_url",
    "newest_url",
    "publisher_accounts",
    "read_downloads",
    "search_accounts",
    "search_requests",
]
