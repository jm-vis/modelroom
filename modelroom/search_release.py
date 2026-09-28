"""Where a model stands in its family: computed from the version numbers of its line (GF F3, ADR 0010).

`decide_age` (`search_age.py`) states `latest` and `legacy` from positive evidence only -- a
`new_version` at the publisher's repository, or the shipped catalog -- and says `unknown` where
neither speaks. The list reads one more thing off the names (decided 2026-09-25, reworked by
AP9-UX2 U5): within one **line** of one publisher's family, the highest version is `latest` and
every member some other member stands over is `legacy`, with one highest member as its successor.
Such a status is `computed`, the list marks it with `*`, and it reaches neither the catalog, the
configuration nor the document.

A name has two kinds of words (GF F3):

- a **line** is a size step with its own name and its own version series, like Opus/Sonnet/Haiku:
  the documented vocabulary `LINE_WORDS`, plus a word of the letter `r` and a number (`LINE_COUNTERS`,
  `R1`), which is a line with its own counter. New line words come only with a GF decision and a test;
- every other word is a **variation** (Base, Instruct, Mini, Flash, VL, Coder, Next ...). A
  variation belongs to its version: a higher version, whatever its variation, makes every variation
  of the lower one `legacy`.

The group is `(publisher, family, line)`, the family the first word of the name. Two pure steps and
no request of their own:

- `line_key` reads one repository name into publisher, family, line, variation, size, version and
  date -- or into `None` when a token of the name has no reading in the grammar below;
- `compute_release` compares the keys of one search's population -- its resolved base models, every
  model of the catalog and every successor the catalog names, and the unresolved repositories of a
  listed packager whose family is the family of exactly one publisher of the population -- and gives
  a status to each resolved base model: a silent catalog's `unknown` is computed; a catalog's
  `latest` gives way to a higher version; a fault's `unknown` becomes `legacy` where a higher version
  is seen, and says its fault as a note (`release_notes`); a stated `legacy` stays.

What the search does not see and the catalog does not know is not compared: each account answers
two pages of twenty repositories, filtered by the search word. That is why the status says
`computed` and never stands in for a statement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Mapping, Sequence

from .catalog import Catalog
from .guided_contracts import SearchHit
from .parameters import _GGUF_SUFFIX, parameters_from_name
from .search_age import AgeVerdict

# The line vocabulary (GF F3, ADR 0010): size steps with their own name and version series. A word
# of exactly the letter of `LINE_COUNTERS` and a number (`r1`) is a line with its own counter.
LINE_WORDS = frozenset({"small", "medium", "large"})
LINE_COUNTERS = frozenset({"r"})
_LINE_COUNTER = re.compile(r"([a-z])(\d+(?:\.\d+)*)")
# A note of step 2 is cut to this many characters (the 100 columns of the table, less the indent).
NOTE_WIDTH = 97

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
    """What a name says about where its model stands: its group, its size, its version and date.

    The group is `(publisher, family, line)` (GF F3); `variant` -- the variations -- only helps to
    choose a successor.
    """

    publisher: str
    family: str
    line: str
    variant: frozenset[str]
    size: float | None
    version: tuple[int, ...]
    date: LineDate | None

    @property
    def group(self) -> tuple[str, str, str]:
        return (self.publisher, self.family, self.line)

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
    (`parameters.parameters_from_name`, the one size rule of the package) comes first.
    """
    owner, _, name = hf_repo.partition("/")
    tokens = _classified(_TOKEN_SPLIT.split(name.lower()))
    if not tokens or tokens[0].kind not in (_PLAIN, _WORD_NUMBER):
        # A name that starts with a size, a version, a date or a precision has no family word.
        return None
    boundary = next((i for i, token in enumerate(tokens) if token.kind not in (_IGNORED, _PLAIN)), len(tokens))
    parts = _after_boundary(tokens, boundary)
    if parts is None:
        return None
    version, dated = parts
    family = tokens[0].word if tokens[0].kind == _WORD_NUMBER else tokens[0].text
    line, counter, variant = _line_and_variant(tokens, boundary)
    if not version and counter:
        version = counter
    size = parameters_from_name(name)
    return LineKey(
        owner.lower(), family, line, variant, size if size is not None else parameters_b, version, dated
    )


def _after_boundary(tokens: list[_Token], boundary: int) -> tuple[tuple[int, ...], LineDate | None] | None:
    """Version and date from the boundary token on; `None` for a second version or date."""
    versions = [token.number for token in tokens[boundary:boundary + 1] if token.kind == _WORD_NUMBER]
    versions += [token.number for token in tokens[boundary:] if token.kind == _NUMBER]
    dates = [token.date for token in tokens[boundary:] if token.kind == _DATE]
    if len(versions) > 1 or len(dates) > 1:
        return None
    return (versions[0] if versions else ()), (dates[0] if dates else None)


def _line_and_variant(tokens: list[_Token], boundary: int) -> tuple[str, tuple[int, ...], frozenset[str]]:
    """The first line word of the name (with a counter's version), and every other word as a variation.

    The family token is neither. A word with a version at the boundary gives its word (`mt2` ->
    `mt`); one behind the boundary stays whole (`qwen3`), as before. `instruct`, `it` and `chat`
    say nothing (`_PLAIN_WORDS`).
    """
    line, counter = "", ()
    words: list[str] = []
    for index, token in enumerate(tokens[1:], start=1):
        if token.kind == _WORD_NUMBER:
            words.append(token.word if index == boundary else token.text)
        elif token.kind == _PLAIN:
            words.append(token.text)
    variant: set[str] = set()
    for word in words:
        counted = _LINE_COUNTER.fullmatch(word)
        if not line and word in LINE_WORDS:
            line = word
        elif not line and counted and counted.group(1) in LINE_COUNTERS:
            numbers = _numbers(counted.group(2))
            line, counter = counted.group(1), numbers or ()
        elif word not in _PLAIN_WORDS:
            variant.add(word)
    return line, counter, frozenset(variant)


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
    """A computed status for every resolved base model the version rule decides (GF F3, ADR 0010).

    `ages` are the verdicts of `decide_age`, one per resolved base model. By the evidence it came
    from (U5 rule 4):

    - a stated `legacy` stays as it is, always (it agrees with the rule);
    - a silent catalog's `unknown` is computed: `latest` for a highest member, else `legacy`;
    - a catalog's `latest` gives way to a computed `legacy` where another member stands over it;
    - a fault's `unknown` (a broken `new_version`, a budget) becomes `legacy` where another member
      stands over it, and stays `unknown` otherwise; its fault is a note either way (`release_notes`).

    Every member of the population counts -- a stated one too; a highest one is a candidate
    successor. A base model this returns nothing for keeps what `decide_age` said.
    """
    return _releases(hits, catalog, ages)[0]


def release_notes(hits: Sequence[SearchHit], catalog: Catalog, ages: Mapping[str, AgeVerdict]) -> list[str]:
    """The notes of step 2 the rule leaves: a statement that did not read, successors that contradict."""
    return _releases(hits, catalog, ages)[1]


def _releases(
    hits: Sequence[SearchHit], catalog: Catalog, ages: Mapping[str, AgeVerdict]
) -> tuple[dict[str, Release], list[str]]:
    keys = _population(hits, catalog)
    maximal = _maximal_members(keys)
    stated = _stated_edges(hits, catalog, ages)
    releases: dict[str, Release] = {}
    notes: list[str] = []
    for base in _resolved_bases(hits):
        verdict = ages.get(base)
        key = keys.get(base)
        if verdict is None or verdict.age == "legacy":
            continue
        fault = verdict.age == "unknown" and verdict.evidence != "catalog"
        if fault:
            notes.append(_note(f"{_short(base)}: its own statement did not read ({verdict.evidence})"))
        if key is None:
            continue
        tops = maximal[key.group]
        if base in tops:
            if (verdict.age, verdict.evidence) == ("unknown", "catalog"):
                releases[base] = Release("latest", None)
            continue
        if verdict.age == "latest" and (verdict.evidence != "catalog" or not _higher_version(key, tops, keys)):
            continue  # a catalog `latest` gives way to a strictly higher version only, never to a date
        candidates = _ordered_successors(key, [(top, keys[top]) for top in tops])
        free = [name for name in candidates if not _reaches(_following(stated), name, base)]
        if not free:
            notes.append(_note(f"{_short(base)}: the successor statements around it contradict each other"))
        releases[base] = Release("legacy", (free or candidates)[0])
    return releases, notes


def _higher_version(key: LineKey, tops: Sequence[str], keys: Mapping[str, LineKey]) -> bool:
    """Whether a highest member of the group carries a strictly higher version number (U5 rule 4)."""
    return any(keys[top].version > key.version for top in tops)


def _short(repo: str) -> str:
    """What a reader calls a model in a note: the part behind the account."""
    return repo.partition("/")[2] or repo


def _note(text: str) -> str:
    """A note of step 2, cut to `NOTE_WIDTH` characters."""
    return text if len(text) <= NOTE_WIDTH else text[: NOTE_WIDTH - 3] + "..."


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
    """Every interpretable name of this search: its resolved base models, the whole catalog, and
    the unresolved repositories of a listed packager that belong to exactly one publisher's family.

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
    keys.update(_packager_members(hits, keys))
    return keys


def _packager_members(hits: Sequence[SearchHit], keys: Mapping[str, LineKey]) -> dict[str, LineKey]:
    """The unresolved repositories of a listed packager, each under the one publisher of its family.

    Read by the name without `-GGUF` (U5 rule 2): they prove "there is a higher version" and can be
    the successor -- the packager's repository then. A family two publishers of the population share
    counts for none of them.
    """
    publishers: dict[str, set[str]] = {}
    for key in keys.values():
        publishers.setdefault(key.family, set()).add(key.publisher)
    members: dict[str, LineKey] = {}
    for hit in hits:
        if hit.resolved or hit.publisher_status != "listed packager" or hit.repo in keys:
            continue
        owner, _, name = hit.repo.partition("/")
        stem = name[: -len(_GGUF_SUFFIX)] if name.upper().endswith(_GGUF_SUFFIX) else name
        key = line_key(f"{owner}/{stem}", hit.parameters_b)
        owners = publishers.get(key.family, set()) if key is not None else set()
        if len(owners) == 1:
            members[hit.repo] = replace(key, publisher=next(iter(owners)))
    return members


def _maximal_members(keys: Mapping[str, LineKey]) -> dict[tuple, list[str]]:
    """Per group, the members no other member stands over -- all of them, whatever their size."""
    groups: dict[tuple, list[str]] = {}
    for name, key in keys.items():
        groups.setdefault(key.group, []).append(name)
    return {
        group: [name for name in names if not any(keys[other].stands_over(keys[name]) for other in names)]
        for group, names in groups.items()
    }


def _ordered_successors(key: LineKey, candidates: list[tuple[str, LineKey]]) -> list[str]:
    """Every candidate, the one to take first (U5 rule 3): never none where there is one.

    The nearest size first -- the same, else the next larger, else the next smaller, a candidate
    without a size after every one with a size; for a row of unknown size the largest known one --
    then the same variation before another, then the fewest variations, then the name. The same size
    is the same number: two sizes that differ are two sizes, however close.
    """

    def order(item: tuple[str, LineKey]) -> tuple:
        name, candidate = item
        if candidate.size is None:
            nearness: tuple = (3, 0.0)
        elif key.size is None:
            nearness = (0, -candidate.size)
        elif candidate.size == key.size:
            nearness = (0, 0.0)
        elif candidate.size > key.size:
            nearness = (1, candidate.size - key.size)
        else:
            nearness = (2, key.size - candidate.size)
        return (*nearness, candidate.variant != key.variant, len(candidate.variant), name)

    return [name for name, _candidate in sorted(candidates, key=order)]


def _stated_edges(hits: Sequence[SearchHit], catalog: Catalog, ages: Mapping[str, AgeVerdict]) -> set[tuple[str, str]]:
    """Every successor a statement names: the catalog's, and the verdicts' of this search."""
    edges = {(m.hf_repo, m.successor) for f in catalog.families for m in f.models if m.successor is not None}
    edges |= {(base, verdict.successor) for base, verdict in ages.items() if verdict.successor is not None}
    edges |= {
        (str(hit.resolved_base_model), hit.successor) for hit in hits if hit.resolved and hit.successor is not None
    }
    return edges


def _following(stated: set[tuple[str, str]]) -> dict[str, set[str]]:
    """The stated successor edges as a graph. A computed edge `base -> candidate` closes a cycle when
    the candidate reaches the base over stated edges alone (U5 rule 4): the check chooses only the
    successor, never the status, and it depends on no order because no computed edge takes part."""
    following: dict[str, set[str]] = {}
    for source, target in stated:
        following.setdefault(source, set()).add(target)
    return following


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


__all__ = [
    "LINE_COUNTERS",
    "LINE_WORDS",
    "LineDate",
    "LineKey",
    "Release",
    "compute_release",
    "line_key",
    "release_log",
    "release_notes",
    "with_release",
]
