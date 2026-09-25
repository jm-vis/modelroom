"""Where a model stands in its family when no statement says so: computed from its version numbers.

`decide_age` (`search_age.py`) states `latest` and `legacy` from positive evidence only -- a
`new_version` at the publisher's repository, or the shipped catalog -- and says `unknown` where
neither speaks. For those rows the list reads one more thing off the names (decided 2026-09-25):
within one family and variant of one publisher, the highest version is `latest` and every lower one
`legacy`, with a highest one as its successor. Such a status is `computed`, the list marks it with
`*`, and it reaches neither the catalog, the configuration nor the document.

Two pure steps and no request of their own:

- `line_key` reads one repository name into publisher, family, variant, size, version and date --
  or into `None` when a token of the name has no reading in the grammar below;
- `compute_release` compares the keys of one search's population -- its resolved base models, every
  model of the catalog and every successor the catalog names -- and gives a status to each resolved
  base model whose `unknown` came from a silent catalog (`AgeVerdict.evidence == "catalog"`).

What the search does not see and the catalog does not know is not compared: each account answers
two pages of twenty repositories, filtered by the search word. That is why the status says
`computed` and never stands in for a statement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from .catalog import Catalog
from .guided_contracts import SearchHit
from .guided_models import parameters_from_name
from .search_age import AgeVerdict

# The token classes, in the order they are tried; the first that matches a lowercase token wins.
_PRECISION = re.compile(r"(?:bf|fp|int|q|iq)\d+")
_EXPERTS = re.compile(r"\d+e")
_ACTIVE = re.compile(r"a\d+(?:\.\d+)?[bm]")
_SIZE = re.compile(r"e?\d+(?:\.\d+)?[bmt]")
_FOUR_DIGITS = re.compile(r"\d{4}")
_MONTH = re.compile(r"0[1-9]|1[0-2]")
_YEAR = re.compile(r"20\d{2}")
# A plain number counts as a version only with one or two digits: `735` or `2025` alone say nothing.
_VERSION = re.compile(r"v(\d+(?:\.\d+)*)|(\d{1,2}(?:\.\d+)*)")
_WORD_VERSION = re.compile(r"([a-z]{2,})(\d+(?:\.\d+)*)")
_WORD = re.compile(r"[a-z][a-z0-9]*")
_TOKEN_SPLIT = re.compile(r"[-_]")
# The instruction-tuned model is the ordinary one of a family, so these words make no variant --
# and no family either, before the boundary (`Mistral-Small-Instruct-2409` is a Mistral Small).
_PLAIN_WORDS = frozenset({"instruct", "it", "chat"})

# What a token is. Every class but `_IGNORED` and `_PLAIN` can mark the boundary between the family
# and the rest of the name: the first such token is it.
_IGNORED = "ignored"  # precision, number of experts, active size of a mixture
_BOUNDARY_SIZE = "size"
_DATE = "date"
_NUMBER = "version"
_WORD_NUMBER = "word with version"
_PLAIN = "word"


@dataclass(frozen=True)
class LineDate:
    """A date in a model name, in the form it was written: `2512` (YYMM), `0528` (MMDD), `08-2025`."""

    form: Literal["yymm", "mmdd", "mm-yyyy"]
    year: int | None
    month: int
    day: int | None

    def value(self) -> tuple[int, int, int]:
        return (self.year or 0, self.month, self.day or 0)


@dataclass(frozen=True)
class LineKey:
    """What a name says about where its model stands: its group, its size, its version and date."""

    publisher: str
    family: str
    variant: frozenset[str]
    size: float | None
    version: tuple[int, ...]
    date: LineDate | None

    @property
    def group(self) -> tuple[str, str, frozenset[str]]:
        return (self.publisher, self.family, self.variant)

    def stands_over(self, other: "LineKey") -> bool:
        """A strict partial order: a higher version, else a date against none, else a later date.

        No version is below every version; with equal versions an undated name is the first release
        and a dated one its revision. Two dates of different forms are not comparable, so neither
        name stands over the other. The size plays no part (decided 2026-09-25).
        """
        if self.version != other.version:
            return self.version > other.version
        if self.date is None:
            return False
        if other.date is None:
            return True
        return self.date.form == other.date.form and self.date.value() > other.date.value()


@dataclass(frozen=True)
class Release:
    """A computed status: `latest`, or `legacy` with the successor it was computed against."""

    age: Literal["latest", "legacy"]
    successor: str | None


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    number: tuple[int, ...] = ()
    word: str = ""
    date: LineDate | None = None


def line_key(hf_repo: str, parameters_b: float | None) -> LineKey | None:
    """The key of one repository name, or `None` when the grammar has no reading for it.

    `parameters_b` is the size the search read for this very repository; the name's own size
    (`guided_models.parameters_from_name`, the one size rule of the package) comes first.
    """
    owner, _, name = hf_repo.partition("/")
    tokens = _classified(_TOKEN_SPLIT.split(name.lower()))
    if tokens is None:
        return None
    boundary = next((i for i, token in enumerate(tokens) if token.kind not in (_IGNORED, _PLAIN)), len(tokens))
    family = [token.text for token in tokens[:boundary] if token.kind == _PLAIN and token.text not in _PLAIN_WORDS]
    if boundary < len(tokens) and tokens[boundary].kind == _WORD_NUMBER:
        family.append(tokens[boundary].word)
    parts = _after_boundary(tokens, boundary)
    if not family or parts is None:
        return None
    variant, version, dated = parts
    size = parameters_from_name(name)
    return LineKey(owner.lower(), "-".join(family), variant, size if size is not None else parameters_b, version, dated)


def _after_boundary(
    tokens: list[_Token], boundary: int
) -> tuple[frozenset[str], tuple[int, ...], LineDate | None] | None:
    """Variant, version and date from the boundary token on; `None` for a second version or date."""
    versions = [token.number for token in tokens[boundary:boundary + 1] if token.kind == _WORD_NUMBER]
    versions += [token.number for token in tokens[boundary:] if token.kind == _NUMBER]
    dates = [token.date for token in tokens[boundary:] if token.kind == _DATE]
    if len(versions) > 1 or len(dates) > 1:
        return None
    words = {token.text for token in tokens[boundary + 1:] if token.kind in (_PLAIN, _WORD_NUMBER)}
    return frozenset(words - _PLAIN_WORDS), (versions[0] if versions else ()), (dates[0] if dates else None)


def _classified(texts: list[str]) -> list[_Token] | None:
    """Every token with its class; a month and a year next to each other are one date token."""
    tokens: list[_Token] = []
    position = 0
    while position < len(texts):
        text = texts[position]
        following = texts[position + 1] if position + 1 < len(texts) else ""
        if _MONTH.fullmatch(text) and _YEAR.fullmatch(following) and _early_class(text) is None:
            tokens.append(_Token(_DATE, f"{text}-{following}", date=LineDate("mm-yyyy", int(following), int(text), None)))
            position += 2
            continue
        token = _early_class(text) or _late_class(text)
        if token is None:
            return None
        tokens.append(token)
        position += 1
    return tokens


def _early_class(text: str) -> _Token | None:
    """Classes 1 to 6: precision, experts and active size (all ignored), size, the two date forms."""
    if _PRECISION.fullmatch(text) or _EXPERTS.fullmatch(text) or _ACTIVE.fullmatch(text):
        return _Token(_IGNORED, text)
    if _SIZE.fullmatch(text):
        return _Token(_BOUNDARY_SIZE, text)
    if _FOUR_DIGITS.fullmatch(text):
        first, second = int(text[:2]), int(text[2:])
        if 20 <= first <= 39 and 1 <= second <= 12:
            return _Token(_DATE, text, date=LineDate("yymm", first, second, None))
        if 1 <= first <= 12 and 1 <= second <= 31:
            return _Token(_DATE, text, date=LineDate("mmdd", None, first, second))
    return None


def _late_class(text: str) -> _Token | None:
    """Classes 8 to 10: a version, a word with its version, a word -- else no class at all."""
    version = _VERSION.fullmatch(text)
    if version:
        number = _numbers(version.group(1) or version.group(2))
        return _Token(_NUMBER, text, number=number) if number is not None else None
    word_version = _WORD_VERSION.fullmatch(text)
    if word_version:
        number = _numbers(word_version.group(2))
        return _Token(_WORD_NUMBER, text, number=number, word=word_version.group(1)) if number is not None else None
    if _WORD.fullmatch(text):
        return _Token(_PLAIN, text)
    return None


def _numbers(text: str) -> tuple[int, ...] | None:
    """The parts of a version, or `None` for a part Python refuses to read (thousands of digits).

    A name is registry-controlled text with no length limit, so this is no reading and never an
    exception out of the search.
    """
    try:
        return tuple(int(part) for part in text.split("."))
    except ValueError:
        return None


def compute_release(
    hits: Sequence[SearchHit], catalog: Catalog, ages: Mapping[str, AgeVerdict]
) -> dict[str, Release]:
    """A computed status for every resolved base model whose `unknown` a silent catalog decided.

    `ages` are the verdicts of `decide_age`, one per resolved base model. Only an `unknown` whose
    evidence is the catalog is computed: an `unknown` from a broken `new_version` or from a budget
    that ended is a fault, not silence, and stays as it is. A stated `latest` or `legacy` keeps its
    status and still counts as a member of its group -- a maximal one is a candidate successor.
    A base model this returns nothing for keeps what `decide_age` said.
    """
    keys = _population(hits, catalog)
    maximal = _maximal_members(keys)
    proposed: dict[str, Release] = {}
    for base in _resolved_bases(hits):
        verdict = ages.get(base)
        key = keys.get(base)
        if verdict is None or key is None or (verdict.age, verdict.evidence) != ("unknown", "catalog"):
            continue
        tops = maximal[key.group]
        if base in tops:
            proposed[base] = Release("latest", None)
            continue
        successor = _successor(key, [(top, keys[top]) for top in tops])
        if successor is not None:
            proposed[base] = Release("legacy", successor)
    stated = _stated_edges(hits, catalog, ages)
    cycled = _on_a_cycle(proposed, stated)
    return {base: release for base, release in proposed.items() if base not in cycled}


def with_release(hits: Sequence[SearchHit], catalog: Catalog, ages: Mapping[str, AgeVerdict]) -> list[SearchHit]:
    """The hits with every computed status filled in, each one validated as it is rebuilt.

    A rebuilt hit goes through `SearchHit.model_validate`, not `model_copy(update=...)`: only the
    first runs the coupling checks (`legacy` with its `successor`, `computed` with a known age).
    """
    computed = compute_release(hits, catalog, ages)
    rebuilt: list[SearchHit] = []
    for hit in hits:
        release = computed.get(str(hit.resolved_base_model)) if hit.resolved else None
        if release is None:
            rebuilt.append(hit)
            continue
        fields = {"age": release.age, "successor": release.successor, "release_basis": "computed"}
        rebuilt.append(SearchHit.model_validate({**hit.model_dump(), **fields}))
    return rebuilt


def release_log(hits: Iterable[SearchHit]) -> list[dict]:
    """One entry per resolved base model, in the order of the hits: the `models` of `search.json`."""
    entries: dict[str, dict] = {}
    for hit in hits:
        base = hit.resolved_base_model
        if hit.resolved and base is not None and base not in entries:
            entries[base] = {
                "base_model": base,
                "age": hit.age,
                "successor": hit.successor,
                "release_basis": hit.release_basis,
            }
    return list(entries.values())


def _resolved_bases(hits: Iterable[SearchHit]) -> list[str]:
    bases: list[str] = []
    for hit in hits:
        if hit.resolved and hit.resolved_base_model is not None and hit.resolved_base_model not in bases:
            bases.append(hit.resolved_base_model)
    return bases


def _population(hits: Sequence[SearchHit], catalog: Catalog) -> dict[str, LineKey]:
    """Every interpretable name of this search: its resolved base models and the whole catalog.

    The size of a base model the name does not state is the parameter count of a hit that is that
    very repository -- never the count of a packager's repository, which may be another build.
    """
    counts = {hit.repo: hit.parameters_b for hit in hits if hit.parameters_b is not None}
    names = _resolved_bases(hits)
    for family in catalog.families:
        for model in family.models:
            names += [repo for repo in (model.hf_repo, model.successor) if repo is not None]
    keys: dict[str, LineKey] = {}
    for name in names:
        key = line_key(name, counts.get(name))
        if key is not None:
            keys[name] = key
    return keys


def _maximal_members(keys: Mapping[str, LineKey]) -> dict[tuple, list[str]]:
    """Per group, the members no other member stands over -- all of them, whatever their size."""
    groups: dict[tuple, list[str]] = {}
    for name, key in keys.items():
        groups.setdefault(key.group, []).append(name)
    return {
        group: [name for name in names if not any(keys[other].stands_over(keys[name]) for other in names)]
        for group, names in groups.items()
    }


def _successor(key: LineKey, candidates: list[tuple[str, LineKey]]) -> str | None:
    """The one candidate of the nearest size: the same, else the next larger, else the next smaller.

    A candidate without a size drops out as soon as one with a size fits. More than one left, or an
    unknown size of the `legacy` row with more than one candidate: no successor -- the rule does not
    pick one at random.
    """
    if len(candidates) == 1:
        return candidates[0][0]
    sized = [(name, candidate.size) for name, candidate in candidates if candidate.size is not None]
    if key.size is None or not sized:
        return None
    # The same size is the same number: two sizes that differ are two sizes, however close.
    same = [name for name, size in sized if size == key.size]
    larger = [size for _name, size in sized if size > key.size]
    smaller = [size for _name, size in sized if size < key.size]
    if same:
        chosen = same
    else:
        nearest = min(larger) if larger else max(smaller)
        chosen = [name for name, size in sized if size == nearest]
    return chosen[0] if len(chosen) == 1 else None


def _stated_edges(hits: Sequence[SearchHit], catalog: Catalog, ages: Mapping[str, AgeVerdict]) -> set[tuple[str, str]]:
    """Every successor a statement names: the catalog's, and the verdicts' of this search."""
    edges = {(m.hf_repo, m.successor) for f in catalog.families for m in f.models if m.successor is not None}
    edges |= {(base, verdict.successor) for base, verdict in ages.items() if verdict.successor is not None}
    edges |= {
        (str(hit.resolved_base_model), hit.successor) for hit in hits if hit.resolved and hit.successor is not None
    }
    return edges


def _on_a_cycle(proposed: Mapping[str, Release], stated: set[tuple[str, str]]) -> set[str]:
    """The sources of every computed edge that lies on a cycle of all edges together.

    All edges are drawn first -- the stated ones and every computed one -- and each computed edge is
    checked against that whole graph; the ones on a cycle fall together, so the result does not
    depend on an order. A stated edge never falls.
    """
    computed = {(base, release.successor) for base, release in proposed.items() if release.successor is not None}
    following: dict[str, set[str]] = {}
    for source, target in stated | computed:
        following.setdefault(source, set()).add(target)
    return {source for source, target in computed if _reaches(following, target, source)}


def _reaches(following: Mapping[str, set[str]], start: str, goal: str) -> bool:
    seen = {start}
    pending = [start]
    while pending:
        current = pending.pop()
        if current == goal:
            return True
        for nxt in following.get(current, ()):
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return False


__all__ = ["LineDate", "LineKey", "Release", "compute_release", "line_key", "release_log", "with_release"]
