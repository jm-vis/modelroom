"""Search Hugging Face for a model name, resolve the hits, and write the result into a config.

Four steps, all over the transport this package already has -- no SDK, no second HTTP stack:

1. `run_search` asks the Hub's model API once (`search_url`, exactly one request) and turns the
   answer into `SearchHit`s. Everything the resolution needs (`tags`, `cardData`, `createdAt`,
   `safetensors`) is asked for in that one request, so a hit costs no further request.
2. Resolution: a hit is `resolved` only when `relation.check_relation` proves it a `quantized`
   build of exactly one base model *and* the catalog knows that base model's account as a
   publisher. Everything else is shown with its reason, gets no Ollama name, no age and no fit.
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
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .catalog import Age, Catalog
from .config import BaseModelConfig, ConfigError, Configuration, config_from_text
from .contracts import validate_hf_repo, validate_ollama_name
from .guided_contracts import UNKNOWN, SearchHit
from .http import BudgetExhaustedError, BudgetedTransport, RedirectingTransport, RequestBudget, Transport
from .relation import check_relation
from .state import acquire_lock, atomic_write_text, release_lock
from .toml_writer import dump_toml

HF_API = "https://huggingface.co/api"
# The Hub answers at most this many repositories per search; the count is shown to the user so
# a short list is never mistaken for "this is everything that exists".
HF_SEARCH_LIMIT = 50
# The whole guided run -- search, resolution, successor lookups, tree pages and the fetch --
# shares this many requests unless the caller passes its own budget.
DEFAULT_GUIDED_BUDGET = 60
# At most two `new_version` edges are followed; the second one only names a younger successor
# and never changes the verdict, which is already `legacy` after the first.
MAX_SUCCESSOR_EDGES = 2
# The shipped positive list of packager accounts, the same one `modelroom.example.toml` ships
# (`test_search.py` keeps the two in step). A guided run that has no configuration yet uses it
# to tell a listed packager from any other account.
DEFAULT_PACKAGERS: tuple[str, ...] = ("unsloth", "bartowski", "mradermacher", "lmstudio-community", "ggml-org")
NO_OLLAMA_LABEL = "none known"
_OLLAMA_ENTRY_PREFIX = "ollama:"
# A base id no repository can carry (a space is not allowed in a repo id), so `check_relation`
# reports `base_model_tag` for a hit that does not declare exactly one base.
_NO_SINGLE_BASE = "(no single declared base)"
# Each half of a repository id, as the Hub itself requires it: starts with an alphanumeric
# character, then letters, digits, `.`, `_` or `-`. Stricter than the shared `hf_repo` pattern
# on purpose -- see `_is_repo_id`.
_REPO_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class SearchError(Exception):
    """The search request itself failed: a status, a body or a budget that ends the search."""


@dataclass(frozen=True)
class AgeVerdict:
    """`latest`/`legacy`/`unknown` for one publisher model, with the evidence that decided it."""

    age: Age
    successor: str | None
    evidence: str


@dataclass(frozen=True)
class SearchOutcome:
    """One search: its hits, how many resolved, and what the shared budget has spent."""

    query: str
    hits: list[SearchHit]
    budget_used: int
    budget_limit: int

    @property
    def resolved(self) -> int:
        return sum(1 for hit in self.hits if hit.resolved)

    @property
    def unresolved(self) -> int:
        return len(self.hits) - self.resolved

    def summary_line(self) -> str:
        """The one line the guided mode prints under the list."""
        return (
            f"{len(self.hits)} repositories, {self.resolved} resolved, "
            f"{self.unresolved} unresolved, budget {self.budget_used}/{self.budget_limit}"
        )


def search_url(name: str) -> str:
    """The one pinned search request: newest GGUF repositories matching `name`.

    `expand` is repeated once per field rather than sent as a list, which is how the Hub's
    model API takes it. Everything the resolution reads is in this one answer: `tags` carries
    the `base_model:`/`base_model:quantized:` relation (real repos state it there and often
    nowhere else), `cardData` the declared base and the license, `createdAt` the day the
    repository was created ("repo created", never the model's release date), `safetensors` the
    parameter count when the repository has a tensor index at all -- measured 2026-09-23: of 50
    real GGUF hits, 3 carried `safetensors` and 5 no `cardData`, so both are optional.
    """
    query = urllib.parse.urlencode({"search": name, "filter": "gguf", "sort": "createdAt", "direction": -1})
    return f"{HF_API}/models?{query}&limit={HF_SEARCH_LIMIT}&expand=cardData&expand=createdAt&expand=safetensors&expand=tags"


def run_search(
    transport: Transport,
    name: str,
    *,
    catalog: Catalog,
    listed_packagers: Sequence[str] = DEFAULT_PACKAGERS,
    budget: RequestBudget | None = None,
    ollama_entries: Mapping[str, str] | None = None,
) -> SearchOutcome:
    """Search for `name` and return every repository the Hub answers with, resolved or not.

    One request for the search itself, plus one age lookup per *distinct* resolved base model
    (`decide_age`); a hit costs no request of its own. `budget` is the run's shared
    `RequestBudget`; `listed_packagers` is the positive list that makes an owner a `listed
    packager` rather than `other`; `ollama_entries` maps a hit's repo to an Ollama name the
    user typed (`parse_ollama_entry`), which wins over the catalog's.
    """
    shared = budget if budget is not None else RequestBudget(DEFAULT_GUIDED_BUDGET)
    redirecting = RedirectingTransport(BudgetedTransport(transport, shared))
    entries = _fetch_search_page(redirecting, name)

    packagers = frozenset(listed_packagers)
    ages: dict[str, AgeVerdict] = {}
    hits = [_build_hit(redirecting, catalog, packagers, ages, entry) for entry in entries]
    hits = _with_ollama_entries(hits, ollama_entries or {})
    return SearchOutcome(query=name, hits=hits, budget_used=shared.used, budget_limit=shared.limit)


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


def _fetch_search_page(transport: Transport, name: str) -> list[dict]:
    """The search answer as a list of repository objects, or `SearchError` with the reason."""
    url = search_url(name)
    try:
        response = transport("GET", url)
    except BudgetExhaustedError as exc:
        raise SearchError(f"search for {name!r} not made: {exc}") from exc
    except Exception as exc:  # a transport failure is the user's to see, never swallowed
        raise SearchError(f"search for {name!r} failed: {exc}") from exc
    if response.status != 200:
        raise SearchError(f"search for {name!r}: unexpected status {response.status}")
    try:
        payload = response.json()
    except Exception as exc:
        raise SearchError(f"search for {name!r}: invalid JSON: {exc}") from exc
    if not isinstance(payload, list) or not all(isinstance(entry, dict) for entry in payload):
        raise SearchError(f"search for {name!r}: the answer is not a list of repositories")
    return payload


def _build_hit(
    transport: Transport,
    catalog: Catalog,
    listed_packagers: frozenset[str],
    ages: dict[str, AgeVerdict],
    entry: dict,
) -> SearchHit:
    repo = entry.get("id")
    if not isinstance(repo, str):
        raise SearchError(f"a search result has no repository id: {entry!r}")
    try:
        validate_hf_repo(repo)
    except ValueError as exc:
        raise SearchError(f"a search result is not a repository id: {exc}") from exc

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
            repo.split("/", 1)[0], catalog=catalog, listed_packagers=listed_packagers
        ),
        "base_model": bases or None,
        "base_model_relation": relation,
        "repo_created_at": _created_at(entry.get("createdAt")),
        "parameters_b": _parameters_b(entry.get("safetensors")),
        "license": _license(card_data),
    }
    if usable_base is None:
        reason = status if status == "metadata_conflict" else "base_model_tag"
        return SearchHit(resolved=False, unresolved_reason=reason, **common)
    if status != "quantized":
        return SearchHit(resolved=False, unresolved_reason=status, **common)

    base = usable_base
    if not catalog.is_publisher(base.split("/", 1)[0]):
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


# --- latest / legacy from positive evidence ---------------------------------------------------


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


# --- the resolution, written back into the configuration --------------------------------------


def apply_hits(config: Configuration, hits: Iterable[SearchHit], *, catalog: Catalog) -> Configuration:
    """`configuration + resolution -> new configuration`, pure: `config` is never changed.

    Every resolved hit adds (or reuses) its base model's family, lists the base model's account
    under `publishers`, and adds the hit's own repository as an owner-bound `repos` target of
    that base model -- the search finds repositories under accounts the packager list does not
    name, and `repos` is where such a target belongs (CONTRACTS.md, "BaseModelConfig").
    Unresolved hits are ignored: nothing unproven ever reaches the configuration. Repeating the
    same hits changes nothing. The result is validated like any other configuration, so a
    target two base models would claim raises `ConfigError` here rather than at fetch time.

    An Ollama name belongs to the **base model**, not to the repository that led to it, so it is
    collected per base model before anything is written: the name the hits carry wins over one
    the configuration already holds (the user has just decided it), and two hits of the same base
    model naming *different* Ollama packages are contradictory input, refused with `ConfigError`
    rather than settled by whichever hit came first.
    """
    resolved = [hit for hit in hits if hit.resolved and hit.resolved_base_model is not None]
    ollama_by_base = _ollama_by_base(resolved)

    data = config.model_dump(mode="json")
    for hit in resolved:
        base = str(hit.resolved_base_model)
        publisher = base.split("/", 1)[0]
        if publisher not in data["publishers"]:
            data["publishers"] = [*data["publishers"], publisher]
        family = _family_entry(data, base, catalog)
        base_entry = _base_model_entry(family, base)
        if hit.repo not in base_entry["repos"]:
            base_entry["repos"] = [*base_entry["repos"], hit.repo]
        name = ollama_by_base.get(base)
        if name is not None:
            base_entry["ollama_base"], _, base_entry["ollama_tag"] = name.partition(":")
    return Configuration.from_dict(data)


def _ollama_by_base(resolved: Sequence[SearchHit]) -> dict[str, str]:
    """The one Ollama name each base model's hits agree on; `ConfigError` when they disagree."""
    names: dict[str, set[str]] = {}
    for hit in resolved:
        if hit.ollama is not None:
            names.setdefault(str(hit.resolved_base_model), set()).add(hit.ollama)
    agreed: dict[str, str] = {}
    for base, candidates in names.items():
        if len(candidates) > 1:
            raise ConfigError(
                f"{base}: the selected repositories name different Ollama packages "
                f"({', '.join(sorted(candidates))}); a base model has one"
            )
        agreed[base] = candidates.pop()
    return agreed


def _family_entry(data: dict, base: str, catalog: Catalog) -> dict:
    """The family dict that holds `base`, created when no family lists it yet."""
    for family in data["families"]:
        if any(entry["hf_repo"] == base for entry in family["base_models"]):
            return family
    name = family_name_for(catalog, base)
    for family in data["families"]:
        if family["name"] == name:
            return family
    family: dict = {"name": name, "base_models": []}
    data["families"] = [*data["families"], family]
    return family


def _base_model_entry(family: dict, base: str) -> dict:
    """The base model dict inside `family`, created without an Ollama name when it is new.

    The Ollama name is set by `apply_hits` afterwards, once per base model, so it does not
    depend on which hit created the entry.
    """
    for entry in family["base_models"]:
        if entry["hf_repo"] == base:
            return entry
    entry = BaseModelConfig(hf_repo=base).model_dump(mode="json")
    family["base_models"].append(entry)
    return entry


def family_name_for(catalog: Catalog, hf_repo: str) -> str:
    """The family a base model belongs to: the catalog's name, else one made from the repo name.

    A made-up name is the repository's own name in the shape a family name has to have
    (`[a-z0-9][a-z0-9.-]*`): lower case, every other character a hyphen.
    """
    for family in catalog.families:
        if any(model.hf_repo == hf_repo for model in family.models):
            return family.name
    name = hf_repo.split("/", 1)[1].lower()
    slug = "".join(character if character.isalnum() or character in ".-" else "-" for character in name)
    slug = slug.lstrip(".-")
    if not slug:
        raise ValueError(f"cannot make a family name from {hf_repo!r}")
    return slug


_CONFIG_HEADER = (
    "modelroom configuration, written by the guided mode.\n"
    "Families and their owner-bound `repos` targets come from the search; everything else is\n"
    "kept as it was. Edit it by hand at any time -- the guided mode reads it back."
)


def write_configuration(path: Path, config: Configuration, *, now: datetime) -> None:
    """Write `config` to `path` as TOML, under the state lock, once it reads back unchanged.

    `path` is resolved once, and the resolved path is used for both the check and the write --
    the same thing `load_config` does when it reads the file back, so the directory that confines
    `paths.state` is the same one in all three places. A path that is a symlink therefore lands in
    the file it points at, not as a new file over the link.

    The text is produced and read back through `config.config_from_text`, the very reader
    `load_config` uses -- so the schema gate, the path resolution and both path confinement
    checks are the ones the next `modelroom` command will apply -- and the result is compared
    with `config`. All of that happens *before* the lock is taken and anything is written, so a
    configuration that would not read back never reaches the disk (the same order `modelroom
    migrate` uses). The lock is the same `modelroom.lock` every other writer takes: a `fetch`
    running in parallel makes this stop with `LockHeldError` instead of writing over the file it
    is reading, and it is released whether the write succeeds or fails. No backup is kept: the
    guided mode has the user confirm the change before it calls this.
    """
    target = path.resolve()
    text = dump_toml(config.model_dump(mode="json"), _CONFIG_HEADER)
    reread = config_from_text(text, target)
    if reread.model_dump(mode="json") != config.model_dump(mode="json"):
        raise ConfigError(f"{target}: the written configuration does not read back unchanged")

    config.paths.state.mkdir(parents=True, exist_ok=True)
    handle = acquire_lock(config.paths.lock_file, "search", now)
    try:
        atomic_write_text(target, text)
    finally:
        release_lock(handle)


__all__ = [
    "DEFAULT_GUIDED_BUDGET",
    "DEFAULT_PACKAGERS",
    "HF_SEARCH_LIMIT",
    "MAX_SUCCESSOR_EDGES",
    "NO_OLLAMA_LABEL",
    "AgeVerdict",
    "SearchError",
    "SearchOutcome",
    "apply_hits",
    "decide_age",
    "family_name_for",
    "ollama_label",
    "owner_class",
    "parse_ollama_entry",
    "run_search",
    "search_url",
    "write_configuration",
]
