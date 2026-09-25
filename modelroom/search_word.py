"""What someone typed into the search, read as one of three things (decided 2026-09-25).

Until now every input went to the Hub as `search=<text>` word for word. A repository id
(`unsloth/Qwen3.5-9B-GGUF`) found nothing, a name with blanks (`qwen 3.5 9b`) found nothing, and
an empty input was an error. Measured live on 2026-09-25: the one request the search made per
account, the account's newest twenty, does not hold `unsloth/Qwen3.5-9B-GGUF` (created
2026-02-28) at all, so even the exact name of a listed model was unreachable.

So the input is parsed first, into one of three forms:

- an **id**: exactly one `/`, both halves a Hub-shaped name -- the repository the person means;
- **words**: the text cut at blanks, `-`, `_` and `:`, lower case, without the filler words a
  person writes around a name (`gguf`, `model`, `models`, `the`);
- **empty**: nothing was typed, or nothing but filler -- a valid answer that means "show me what
  fits this machine" (`search.py`, the catalog pages).

Everything here is pure: no request, no catalog file read, no clock. The Hub gets **one** word of
a multi-word input (`hub_word`) and the answer is filtered against **all** of them locally
(`SearchWord.matches`), because the Hub's `search=` is one string and a longer one finds less.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Literal

from .catalog import Catalog, CatalogFamily, CatalogModel

SearchForm = Literal["id", "words", "empty"]

# What a person writes around a name without meaning it as part of the name.
FILLER_WORDS: frozenset[str] = frozenset({"gguf", "model", "models", "the"})
# `difflib`'s own scale: 1.0 is equal, 0 is nothing in common. 0.75 is what turns `qwn` into
# `qwen` (0.86) and leaves `q` and `llama` apart (decided 2026-09-25).
CLOSE_MATCH_CUTOFF = 0.75
# How many "did you mean" candidates are offered at most: a list a person reads in one look.
MAX_CANDIDATES = 5
# A letter run shorter than this is no word to search for (`r` of `R1`, `b` of `9B`).
MIN_STEM = 3

_SEPARATORS = re.compile(r"[\s\-_:]+")
_ALL_SEPARATORS = re.compile(r"[\s\-_:/]+")
_LETTER_RUNS = re.compile(r"[a-z]+")
# Each half of a repository id as the Hub itself requires it: an alphanumeric character first,
# then letters, digits, `.`, `_` or `-`. The same rule `search.py::_is_repo_id` applies before it
# puts a half into a URL.
_HUB_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def without_separators(value: str) -> str:
    """`value` lower case with every separator gone, so a typed name can be held against an id.

    `Qwen/Qwen3.5-9B` reads as `qwenqwen3.59b`. The decimal point stays: it belongs to `3.5`, and
    a person who types `qwen 3.5 9b` means that number.
    """
    return _ALL_SEPARATORS.sub("", value).casefold()


@dataclass(frozen=True)
class SearchWord:
    """One parsed input: which of the three forms it is, and the words it is made of.

    `text` is what was typed, stripped of surrounding blanks -- what the search log and the
    screen call the word. `words` is empty exactly for the `empty` form. `repo_id` is set exactly
    for the `id` form and keeps the typed spelling, because a repository id is case sensitive;
    `words` then holds the words of its **name** half, which is what the search falls back to
    when that repository holds no GGUF file.
    """

    text: str
    form: SearchForm
    words: tuple[str, ...] = ()
    repo_id: str | None = None

    @property
    def hub_word(self) -> str:
        """The one word that goes to the Hub as `search=`.

        One word alone is sent as it is -- that is what every recorded answer of this suite was
        taken with, and a single word is already what the person chose. For more than one word the
        longest **letter run** is sent (`qwen 3.5 9b` and `Qwen3.5-9B` both ask `qwen`): the Hub
        answers a longer string with less, and the digits of a size are what differs between a
        model and its siblings, so they are what the local filter is for. Without a letter run of
        its own the longest word is sent unchanged.
        """
        if len(self.words) == 1:
            return self.words[0]
        runs = [run for word in self.words for run in _LETTER_RUNS.findall(word) if len(run) >= 2]
        if runs:
            return max(runs, key=len)
        return max(self.words, key=len) if self.words else ""

    @property
    def filters_locally(self) -> bool:
        """Whether the answer is held against the words at all: only for more than one word.

        One word is the search this package has always made, and its recorded answers carry
        repositories whose id does not hold that word anywhere (a `mistralai/Ministral-*` build
        under the word `mistral`). Filtering a one-word search would drop them.
        """
        return len(self.words) > 1

    def matches(self, repo: str) -> bool:
        """Whether this input names `repo`: every word part of its id.

        One word is held against the id as it stands (today's behavior, so `r1` still finds
        `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`); more than one word is held against the id with
        its separators removed, so `qwen 3.5 9b` finds `unsloth/Qwen3.5-9B-GGUF`.
        """
        if not self.words:
            return False
        if len(self.words) == 1:
            return self.words[0] in repo.casefold()
        haystack = without_separators(repo)
        return all(word in haystack for word in self.words)


def parse_search(text: str) -> SearchWord:
    """Read one typed input as an id, as words, or as empty."""
    stripped = text.strip()
    repo_id = stripped if _is_hub_id(stripped) else None
    source = repo_id.partition("/")[2] if repo_id is not None else stripped
    words = _words_of(source)
    if repo_id is not None:
        return SearchWord(text=stripped, form="id", words=words, repo_id=repo_id)
    if not words:
        return SearchWord(text=stripped, form="empty")
    return SearchWord(text=stripped, form="words", words=words)


def _words_of(text: str) -> tuple[str, ...]:
    """The words of one input: lower case, cut at the separators, without the filler words."""
    parts = [part for part in _SEPARATORS.split(text.casefold()) if part]
    return tuple(part for part in parts if part not in FILLER_WORDS)


def _is_hub_id(text: str) -> bool:
    """Whether `text` is a repository id: exactly one `/`, both halves Hub-shaped."""
    owner, separator, name = text.partition("/")
    if not separator or "/" in name:
        return False
    return all(_HUB_SEGMENT.fullmatch(half) for half in (owner, name))


# --- what the shipped catalog has to say about the input -----------------------------------------


def catalog_models(catalog: Catalog, word: SearchWord) -> list[CatalogModel]:
    """Every catalog model this input names, in catalog order."""
    return [model for family in catalog.families for model in family.models if word.matches(model.hf_repo)]


def family_matches(family: CatalogFamily, word: SearchWord) -> bool:
    """Whether one family is what this input is about: its name, its account, or one of its models.

    The name and the account are held against **all** the words with their separators removed, so
    `qwen 3.5` matches the family `qwen3.5`; a model is held against the same rule
    `SearchWord.matches` uses everywhere else.
    """
    if not word.words:
        return False
    for text in (family.name, family.publisher):
        haystack = without_separators(text)
        if all(part in haystack for part in word.words):
            return True
    return any(word.matches(model.hf_repo) for model in family.models)


def latest_model_names(catalog: Catalog) -> list[str]:
    """The name of every model the catalog calls `latest`, in catalog order, each once.

    The pages of a search with no word at all: one per current model of the catalog
    (`search.py`, "no search word"). The name and not the id, because that is what the Hub's
    `search=` takes.
    """
    names: list[str] = []
    for family in catalog.families:
        for model in family.models:
            if not model.latest:
                continue
            name = model.hf_repo.partition("/")[2]
            if name not in names:
                names.append(name)
    return names


def close_matches(catalog: Catalog, word: SearchWord) -> list[str]:
    """What the input could have meant, when the catalog knows nothing that it names.

    `difflib.get_close_matches` at `CLOSE_MATCH_CUTOFF` over the catalog's family names, its
    publisher accounts, its model names and the letter runs of those names -- the runs are what
    makes a typo in the family's own word reachable (`qwn` -> `qwen`, from `Qwen3.5-9B`). The
    order is the order the words were typed in, each candidate once.
    """
    pool = candidate_pool(catalog)
    found: list[str] = []
    for part in word.words:
        for candidate in difflib.get_close_matches(part, pool, n=MAX_CANDIDATES, cutoff=CLOSE_MATCH_CUTOFF):
            if candidate not in found:
                found.append(candidate)
    return found[:MAX_CANDIDATES]


def candidate_pool(catalog: Catalog) -> list[str]:
    """Every word of the catalog a typo could have meant, lower case, each once."""
    pool: list[str] = []
    for family in catalog.families:
        names = [family.name, family.publisher, *(m.hf_repo.partition("/")[2] for m in family.models)]
        for name in names:
            _add(pool, name.casefold())
            for run in _LETTER_RUNS.findall(name.casefold()):
                if len(run) >= MIN_STEM:
                    _add(pool, run)
    return pool


def _add(pool: list[str], value: str) -> None:
    if value and value not in pool:
        pool.append(value)


__all__ = [
    "CLOSE_MATCH_CUTOFF",
    "FILLER_WORDS",
    "MAX_CANDIDATES",
    "MIN_STEM",
    "SearchForm",
    "SearchWord",
    "candidate_pool",
    "catalog_models",
    "close_matches",
    "family_matches",
    "latest_model_names",
    "parse_search",
    "without_separators",
]
