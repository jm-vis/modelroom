"""The rules of the catalog scanner (`catalog_scan.py`), as pure functions: no request, no file, no clock.

Everything the scanner decides from what the Hub answered lives here, so a test hands a rule a value and
reads the value back: what a quantization is, which Hub tasks are never a model of this table, how the kind
of a repo follows from its task, its tags, its base model and its `config.json` (stages 1 to 4 of the brief,
"Art bestimmen, in Stufen"), the request forms, the group a model belongs to, the row a proposal would add,
the order and numbers of a proposal, and its Markdown rendering (`render_proposal`, also what `--show`
of `catalog_publish.py` prints). Family, line, relation, owner class and the kind table are the package's
own (`search_release`, `relation`, `search`, `catalog`); this module adds no second rule for them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from modelroom.catalog import CATALOG_MAX_GGUF_REPOS, CATALOG_MAX_TASK_LEN, CATALOG_MAX_TEXT_LEN, PIPELINE_TAG_TO_KIND, CatalogError, CatalogModel
from modelroom.catalog_schema import build_catalog_v2
from modelroom.contracts import SchemaVersionError, validate_hf_repo
from modelroom.parameters import parameters_from_name
from modelroom.relation import _tag_facts
from modelroom.search import _created_at
from modelroom.search_pages import read_downloads
from modelroom.search_release import line_key

HUB = "https://huggingface.co"
SCHEMA = 1

# A repo is a quantization or a package (a) with a tag `base_model:quantized:...`, (b) with one of these tags,
# or (c) when its name ends on one of these suffixes in any case (brief, "Quantisierungsregel, endlich").
QUANTIZATION_TAGS = ("gguf", "mlx", "awq", "gptq", "exl2", "onnx")
QUANTIZATION_SUFFIXES = (
    "-gguf", "-awq", "-gptq", "-mlx", "-exl2", "-onnx", "-fp8", "-fp4", "-nvfp4", "-mxfp4", "-int4", "-int8",
    "-w4a16", "-w8a8", "-4bit", "-8bit", "-bnb-4bit",
)
# Hub tasks that can be none of the eight kinds: a repo with one of them is never a row and never a candidate
# (case (b)), whatever a side tag says. `feature-extraction` and `sentence-similarity` are in the kind table.
EXCLUDED_PIPELINE_TAGS = (
    "image-classification", "zero-shot-image-classification", "object-detection", "zero-shot-object-detection",
    "image-segmentation", "mask-generation", "keypoint-detection", "depth-estimation", "video-classification",
    "audio-classification", "voice-activity-detection", "text-classification", "token-classification",
    "zero-shot-classification", "fill-mask", "question-answering", "table-question-answering",
    "tabular-classification", "tabular-regression", "time-series-forecasting", "reinforcement-learning",
    "robotics", "graph-ml", "text-to-3d", "image-to-3d",
)
# Stage 4: only top-level keys of `config.json`, and a vision or audio key counts only with a non-empty object.
VISION_CONFIG_KEYS = ("vision_config", "vision_encoder")
AUDIO_CONFIG_KEYS = ("audio_config",)
TEXT_ARCHITECTURE_SUFFIX = "ForCausalLM"

B1_MIN_LIKES, B1_MIN_DOWNLOADS, B1_MAX_AGE = 100, 1000, timedelta(days=30)
B2_MIN_ROWS = 3
MAX_CANDIDATES = 10  # for B1 and B2 together; B3 has no limit
MAX_PAGES = 10  # pages per account and run, all steps together
MAX_GGUF_REPOS = CATALOG_MAX_GGUF_REPOS  # a row's list of GGUF repos is cut here; what the cut takes is no change
OVERLAP = timedelta(days=2)
FIRST_RUN_BACK = timedelta(days=30)
CONFIG_MAX_BYTES, API_MAX_BYTES = 64 * 1024, 1024 * 1024

_TASK_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_LIST_EXPAND = ("cardData", "createdAt", "downloads", "likes", "pipeline_tag", "safetensors", "tags")
_B1_EXPAND = ("createdAt", "likes", "downloads", "tags")
_B2_EXPAND = ("createdAt", "downloads", "likes", "tags")
ROW_FIELDS = tuple(sorted(CatalogModel.model_fields))


# --- request forms ---------------------------------------------------------------------------------------------------


def _models(query: dict, expand: tuple[str, ...]) -> str:
    return f"{HUB}/api/models?{urllib.parse.urlencode(query)}" + "".join(f"&expand={name}" for name in expand)


def list_url(account: str) -> str:
    """The first page of one account: its newest repos, 100 to a page, with the fields the rules read."""
    return _models({"author": account, "sort": "createdAt", "direction": -1, "limit": 100}, _LIST_EXPAND)


def b1_url(task: str) -> str:
    return _models({"pipeline_tag": task, "sort": "trendingScore", "direction": -1, "limit": 100}, _B1_EXPAND)


def b2_url(hf_repo: str) -> str:
    return _models({"filter": f"base_model:quantized:{hf_repo}", "sort": "downloads", "direction": -1, "limit": 10}, _B2_EXPAND)


def base_url(hf_repo: str) -> str:
    return f"{HUB}/api/models/{hf_repo}"


def config_url(hf_repo: str) -> str:
    return f"{HUB}/{hf_repo}/resolve/main/config.json"


def collections_url(owner: str) -> str:
    return f"{HUB}/api/collections?owner={urllib.parse.quote(owner)}"


def next_url_ok(url: object, account: str) -> bool:
    """A stored or announced next page is used only on `huggingface.co/api/models` with `author=<account>`."""
    if not isinstance(url, str):
        return False
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme == "https" and parts.hostname == "huggingface.co" and port in (None, 443)
        and parts.path == "/api/models" and urllib.parse.parse_qs(parts.query).get("author") == [account]
    )


# --- one entry of a Hub list ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """What the rules read of one repo of a list answer; every field is checked, nothing is taken on trust."""

    repo: str
    created: datetime | None
    likes: int
    downloads: int
    tags: tuple[str, ...]
    card: dict
    pipeline: str | None  # the `pipeline_tag` of the top level, verbatim, `None` when absent or not a task name
    parameters_b: float | None

    @property
    def owner(self) -> str:
        return self.repo.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.repo.split("/", 1)[1]


def parse_entry(raw: object) -> Entry | None:
    """One list entry as an `Entry`, or `None` when it names no valid repo."""
    if not isinstance(raw, dict):
        return None
    try:
        repo = validate_hf_repo(raw.get("id") if isinstance(raw.get("id"), str) else raw.get("modelId"))
    except (ValueError, TypeError):
        return None
    tags = raw.get("tags")
    task = raw.get("pipeline_tag")
    total = (raw.get("safetensors") or {}).get("total") if isinstance(raw.get("safetensors"), dict) else None
    sized = isinstance(total, int) and not isinstance(total, bool) and total > 0
    return Entry(
        repo=repo, created=_created_at(raw.get("createdAt")), likes=read_downloads(raw.get("likes")) or 0,
        downloads=read_downloads(raw.get("downloads")) or 0,
        tags=tuple(t for t in tags if isinstance(t, str)) if isinstance(tags, list) else (),
        card=raw["cardData"] if isinstance(raw.get("cardData"), dict) else {},
        pipeline=task if isinstance(task, str) and len(task) <= CATALOG_MAX_TASK_LEN and _TASK_RE.fullmatch(task) else None,
        parameters_b=total / 1_000_000_000 if sized else None,
    )


def is_quantization(tags, repo: str) -> bool:
    """The finite quantization rule of the brief: tag `base_model:quantized:`, a listed tag, or a listed suffix."""
    tags = list(tags)
    name = repo.rsplit("/", 1)[-1].lower()
    return (
        any(tag.startswith("base_model:quantized:") for tag in tags) or any(tag in QUANTIZATION_TAGS for tag in tags)
        or name.endswith(QUANTIZATION_SUFFIXES)
    )


def base_repos(tags) -> list[str]:
    """Every repo the `base_model:` tags name (both forms), sorted and once each."""
    try:
        bases, relations = _tag_facts(list(tags))
    except ValueError:
        return []
    return sorted(bases | {repo for _, repo in relations})


# --- the kind, in stages ----------------------------------------------------------------------------------------------


class KindResult(NamedTuple):
    """`kind` and `kind_source` (the stage name, `None` for the task itself) or `excluded`; `evidence` goes in the proposal."""

    kind: str | None
    source: str | None
    excluded: bool
    evidence: dict


def cheap_kind(entry: Entry) -> KindResult | None:
    """Stage 1 (the task is in the table), case (b) (the task is excluded) and stage 2 (the tags); `None` needs a request."""
    task = entry.pipeline
    if task in PIPELINE_TAG_TO_KIND:
        return KindResult(PIPELINE_TAG_TO_KIND[task], None, False, {"stage": "pipeline_tag", "value": task})
    if task in EXCLUDED_PIPELINE_TAGS:
        return KindResult(None, None, True, {"stage": "excluded", "value": task})
    named = sorted({tag for tag in entry.tags if tag in PIPELINE_TAG_TO_KIND})
    if len({PIPELINE_TAG_TO_KIND[tag] for tag in named}) == 1:
        return KindResult(PIPELINE_TAG_TO_KIND[named[0]], "tags", False, {"stage": "tags", "value": named})
    return None


def single_base(entry: Entry) -> str | None:
    """Stage 3 asks only when the tags name exactly one other, valid repo."""
    bases = base_repos(entry.tags)
    if len(bases) != 1 or bases[0] == entry.repo:
        return None
    try:
        return validate_hf_repo(bases[0])
    except ValueError:
        return None


def kind_from_base_answer(answer: object) -> str | None:
    """Stage 3: the task at the top of the base model's answer, when it is in the table."""
    task = answer.get("pipeline_tag") if isinstance(answer, dict) else None
    return PIPELINE_TAG_TO_KIND.get(task) if isinstance(task, str) else None


def kind_from_config(config: object) -> tuple[str, str] | None:
    """Stage 4: `(kind, deciding key)` from the top-level keys of `config.json`, or `None` when it decides nothing."""
    if not isinstance(config, dict):
        return None
    architectures = config.get("architectures")
    if "architectures" in config and not (isinstance(architectures, list) and all(isinstance(a, str) for a in architectures)):
        return None
    # a key counts only with a non-empty object; any other value is no proof of a modality and spoils nothing
    marked = {key: config[key] for key in VISION_CONFIG_KEYS + AUDIO_CONFIG_KEYS if isinstance(config.get(key), dict) and config[key]}
    vision = [key for key in VISION_CONFIG_KEYS if key in marked]
    audio = [key for key in AUDIO_CONFIG_KEYS if key in marked]
    if vision and audio:
        return None
    if vision:
        return "vision", vision[0]
    if audio:
        return "audio", audio[0]
    if architectures and all(a.endswith(TEXT_ARCHITECTURE_SUFFIX) for a in architectures):
        return "text", "architectures"
    return None


# --- groups, family, latest -----------------------------------------------------------------------------------------


def _key(repo: str, parameters_b: float | None):
    return line_key(repo, parameters_b)


def stands_over(repo: str, kind: str, other: CatalogModel | tuple, size: float | None = None) -> bool:
    """`repo` of `kind` and `size` stands over `other` (a row, or `(repo, kind, parameters_b)`) inside one group of
    (publisher, kind, family, line)."""
    o_repo, o_kind, o_size = (other.hf_repo, other.kind, other.parameters_b) if isinstance(other, CatalogModel) else other
    mine, theirs = _key(repo, size), _key(o_repo, o_size)
    return bool(mine and theirs and kind == o_kind and mine.group == theirs.group and mine.stands_over(theirs))


def family_for(repo: str, kind: str, rows: list[CatalogModel], size: float | None = None) -> str | None:
    """The family of a new row: the one its group's rows share, else the family rule of `search_release`; `None`
    when the name has no family word (no row is proposed then)."""
    key = _key(repo, size)
    if key is None:
        return None
    shared = {r.family for r in rows if r.kind == kind and (rk := _key(r.hf_repo, r.parameters_b)) and rk.group == key.group}
    if len(shared) == 1:
        return shared.pop()
    own, owner = key.family, repo.split("/", 1)[0]
    if any(r.family == own and r.publisher != owner for r in rows):
        own = f"{own}-{re.sub(r'[^a-z0-9.-]+', '-', owner.lower()).strip('-.')}"  # a family belongs to one publisher
    return own[:CATALOG_MAX_TEXT_LEN]


def latest_among(fresh: list[tuple[str, str, float | None]], rows: list[CatalogModel]) -> set[str]:
    """The new repos that stand over a `latest` row of their group and under no other new repo and no row of it
    (`rows` holds the table and the carried new rows: a carried row stands over a fresh one as a fresh one does)."""
    out = set()
    for repo, kind, size in fresh:
        over_latest = any(r.latest and stands_over(repo, kind, r, size) for r in rows)
        others = [(o, k, s) for o, k, s in fresh if o != repo] + [(r.hf_repo, r.kind, r.parameters_b) for r in rows if r.hf_repo != repo]
        if over_latest and not any(stands_over(o, k, (repo, kind, size), s) for o, k, s in others):
            out.add(repo)
    return out


def build_row(entry: Entry, result: KindResult, *, family: str, today: str, latest: bool, latest_source: str | None) -> dict:
    """A complete row of schema 2, as the JSON values `catalog.json` carries."""
    license_name = entry.card.get("license")
    size = entry.parameters_b if entry.parameters_b is not None else parameters_from_name(entry.name)
    return {
        "added": today, "family": family, "gguf_repos": [], "hf_repo": entry.repo,
        "hub_created": entry.created.date().isoformat() if entry.created else None, "hub_pipeline_tag": entry.pipeline,
        "kind": result.kind, "kind_source": result.source, "latest": latest, "latest_checked": today if latest else None,
        "latest_source": latest_source if latest else None,
        "license": license_name if isinstance(license_name, str) and 0 < len(license_name) <= CATALOG_MAX_TEXT_LEN else None,
        "note": None, "ollama_base": None, "ollama_tag": None, "parameters_b": size, "publisher": entry.owner,
        "source": f"{HUB}/{entry.repo}", "successor": None, "successor_checked": None, "successor_source": None,
    }


def new_row_change(row: dict, result: KindResult, open_since: str) -> dict:
    return {
        "change": "new_row", "stage": "A1", "hf_repo": row["hf_repo"], "kind": row["kind"], "open_since": open_since,
        "evidence": [row["source"]], "kind_evidence": result.evidence, "fields": {k: {"old": None, "new": row[k]} for k in ROW_FIELDS},
    }


def field_change(row: CatalogModel, stage: str, pairs: dict, evidence: list[str], open_since: str, field: str) -> dict:
    return {
        "change": "field", "field": field, "stage": stage, "hf_repo": row.hf_repo, "kind": row.kind, "open_since": open_since,
        "evidence": evidence, "fields": {name: {"old": old, "new": new} for name, (old, new) in pairs.items()},
    }


def parse_time(value: object) -> datetime | None:
    """A Hub time as an aware UTC instant, or `None` when it is no time."""
    return _created_at(value)


def b3_candidate(entry: Entry, result: KindResult, open_since: str) -> dict:
    """A repo whose kind stays open: what the Hub said (task, tags), the last check, and what a row would need."""
    license_name = entry.card.get("license")
    return {
        "candidate": "B3", "hf_repo": entry.repo, "account": entry.owner, "likes": entry.likes, "downloads": entry.downloads,
        "created_at": stamp(entry.created) if entry.created else None, "hub_pipeline_tag": entry.pipeline, "tags": list(entry.tags),
        "license": license_name if isinstance(license_name, str) else None, "parameters_b": entry.parameters_b,
        "kind_evidence": result.evidence, "evidence": [f"{HUB}/{entry.repo}"], "open_since": open_since,
    }


def open_candidate(kind: str, account: str, entries: list[Entry], extra: dict, open_since: str) -> dict:
    """A B1 or B2 candidate: the account, its best repo and the repos that make the case (at most ten)."""
    top = entries[0]
    return {
        "candidate": kind, "account": account, "hf_repo": top.repo, "likes": sum(e.likes for e in entries) if kind == "B2" else top.likes,
        "downloads": top.downloads, "created_at": stamp(top.created) if top.created else None, "open_since": open_since,
        "repos": [{"hf_repo": e.repo, "likes": e.likes, "downloads": e.downloads} for e in entries[:10]],
        "evidence": [f"{HUB}/{e.repo}" for e in entries[:10]], **extra,
    }


def entry_of_candidate(candidate: dict) -> Entry:
    """The `Entry` a B3 candidate was made from, for the second check of a candidate that waited for budget."""
    license_name = candidate.get("license")
    return Entry(
        repo=candidate["hf_repo"], created=parse_time(candidate.get("created_at")), likes=candidate.get("likes", 0),
        downloads=candidate.get("downloads", 0), tags=tuple(candidate.get("tags") or ()),
        card={"license": license_name} if isinstance(license_name, str) else {},
        pipeline=candidate.get("hub_pipeline_tag"), parameters_b=candidate.get("parameters_b"),
    )


# --- what carries over from the last proposal ----------------------------------------------------------------------

# The states of a `decision.json` that count as decided: everything from `approved` on.
DECISION_STATES = ("approved", "applied", "committed", "pushed", "uploaded", "published")


def parse_rejected(raw: bytes) -> tuple[set[str], set[str]]:
    """`(repos, accounts)` of the rejection list: names, or objects with a name under `hf_repo` (else `account`); a `/`
    makes it a repo. Any other entry is a `ValueError`, never skipped: a damaged entry must not lift a rejection."""
    entries = json.loads(raw.decode("utf-8"))
    if not isinstance(entries, list):
        raise ValueError("not a list")
    names = [e if isinstance(e, str) else e.get("hf_repo", e.get("account")) if isinstance(e, dict) else None for e in entries]
    for number, name in enumerate(names, start=1):
        if not (isinstance(name, str) and name):
            raise ValueError(f"entry {number} is neither a name nor an object with a name under hf_repo or account")
    return {n for n in names if "/" in n}, {n for n in names if "/" not in n}


def touched(change: dict) -> set[tuple]:
    """Every `(repo, field)` pair a change touches: its main field and each field it carries."""
    repo, fields, field = change["hf_repo"], change.get("fields"), change.get("field")
    return {(repo, name) for name in (fields if isinstance(fields, dict) else ())} | ({(repo, field)} if isinstance(field, str) else set())


def _without_rejected(names: list, rejected) -> list:
    """A list of GGUF repos without those the rejection list now holds (the repo or its account)."""
    return [name for name in names if not (isinstance(name, str) and rejected(name))]


def refresh_field(change: dict, row: CatalogModel, rejected=lambda repo: False) -> dict | None:
    """A carried field change against the table and the rejection list as they are now: new `old` values, `None`
    when the table has the target or nothing new is left."""
    now = row.model_dump(mode="json")
    if change["field"] == "gguf_repos":
        pair = change["fields"]["gguf_repos"]
        if not (isinstance(pair["new"], list) and isinstance(pair["old"], list)):
            return None
        added = [r for r in _without_rejected(pair["new"], rejected) if isinstance(r, str) and r not in pair["old"] and r not in now["gguf_repos"]]
        target = (now["gguf_repos"] + added)[:MAX_GGUF_REPOS]
        change["fields"]["gguf_repos"] = {"old": now["gguf_repos"], "new": target}
        return change if target != now["gguf_repos"] else None  # a full list takes nothing more
    if all(now.get(name) == pair["new"] for name, pair in change["fields"].items()):
        return None
    for name, pair in change["fields"].items():
        pair["old"] = now.get(name)
    return change


def _entries(proposal: dict, name: str) -> list[dict]:
    found = proposal.get(name)
    return [e for e in found if isinstance(e, dict)] if isinstance(found, list) else []


def _ids(decision: dict, key: str) -> set[str]:
    """The numbers and `Kk` of one list of a decision; a value that is no list counts as empty, an entry that is
    neither a number nor a string is skipped."""
    found = decision.get(key)
    return {str(x) for x in found if isinstance(x, (int, str)) and not isinstance(x, bool)} if isinstance(found, list) else set()


def _repo(value: object) -> bool:
    try:
        return bool(validate_hf_repo(value))
    except (ValueError, TypeError):
        return False


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _sound_change(change: dict) -> bool:
    """A carried change has every key the rest of the run reads (also the rendering: `evidence` is a list of strings,
    as every writer of a change writes it), so a damaged file costs the entry and not the run."""
    fields = change["fields"]
    return (
        _repo(change["hf_repo"]) and _strings(change["evidence"]) and isinstance(fields, dict)
        and all(isinstance(p, dict) and {"old", "new"} <= set(p) for p in fields.values())
        and isinstance(change["stage"], str) and isinstance(change["open_since"], str) and isinstance(change["kind"], str)
        and (change["change"] != "field" or isinstance(change["field"], str))
    )


def _sound_candidate(cand: dict) -> bool:
    """A carried candidate has the keys and types the run reads (sorting, the second check of a B3, the rendering)."""
    if not (isinstance(cand["account"], str) and _repo(cand["hf_repo"]) and isinstance(cand["open_since"], str) and _strings(cand["evidence"])):
        return False
    if cand["candidate"] != "B3":
        return isinstance(cand["likes"], int) and not isinstance(cand["likes"], bool)
    tags, task, size = cand.get("tags"), cand.get("hub_pipeline_tag"), cand.get("parameters_b")
    return (
        isinstance(cand.get("kind_evidence"), dict) and (tags is None or isinstance(tags, list) and all(isinstance(t, str) for t in tags))
        and (task is None or isinstance(task, str)) and (size is None or isinstance(size, (int, float)) and not isinstance(size, bool))
    )


def _named(entry: dict) -> list:
    """Every other repo a carried entry takes as evidence: the successor a change names, the base model that gave a
    new row its kind, and the models of a B1 or B2 candidate (its best repo, its list, the rows of a B2)."""
    if "change" in entry:
        base = entry.get("kind_evidence")
        return [entry["fields"].get("successor", {}).get("new")] + ([base.get("value")] if isinstance(base, dict) and base.get("stage") == "base_model" else [])
    return [entry["hf_repo"], *(r["hf_repo"] for r in entry.get("repos", [])), *entry.get("bases", [])]


def carry_over(proposal: dict, decision: object, rows: dict, listed: set[str], rejected) -> tuple[dict, dict, set[str], set[tuple], set[str]]:
    """What of the newest earlier proposal is still open: `(changes, candidates, known repos, decided field pairs,
    decided accounts)`.

    Dropped: what the decision of that folder took (a number or a `Kk` in `approved`, `rejected` or `admitted`, from
    state `approved` on), what the rejection list now holds (also inside a list of GGUF repos, as a successor, as the
    base model of a kind, as a model of a B1 or B2), a new row the table now has, a field change the table already
    carries, a change that touches a field the decision took, an account that is listed now. What the decision took is
    returned too, for the whole run: the repos of its new rows and B3 candidates (in `known`), every `(repo, field)`
    pair its field changes touch and the accounts of its B1 and B2 candidates. Unreadable entries are skipped.
    """
    taken = decision if isinstance(decision, dict) and decision.get("state") in DECISION_STATES else {}
    decided = set().union(*(_ids(taken, key) for key in ("approved", "rejected", "admitted")))
    done_repos: set[str] = set()
    done_fields: set[tuple] = set()
    done_accounts: set[str] = set()
    changes: dict[tuple, dict] = {}
    candidates: dict[tuple, dict] = {}
    for change in _entries(proposal, "changes"):
        try:
            repo, row = change["hf_repo"], rows.get(change["hf_repo"])
            if str(change.get("n")) in decided:
                done_repos.add(repo) if change["change"] == "new_row" else done_fields.update(touched(change))
                continue
            if not _sound_change(change) or rejected(repo) or any(isinstance(r, str) and rejected(r) for r in _named(change)):
                continue
            if change["change"] == "new_row" and row is None:
                if isinstance((gguf := change["fields"].get("gguf_repos", {})).get("new"), list):
                    gguf["new"] = _without_rejected(gguf["new"], rejected)
                changes[("new_row", repo)] = change
            elif change["change"] == "field" and row is not None and (fresh := refresh_field(change, row, rejected)):
                changes[("field", repo, change["field"])] = fresh
        except (KeyError, TypeError, AttributeError):
            continue
    changes = {key: change for key, change in changes.items() if not touched(change) & done_fields}
    for cand in _entries(proposal, "candidates"):
        try:
            kind = cand["candidate"]
            key = cand["hf_repo"] if kind == "B3" else cand["account"]
            if kind in ("B1", "B2", "B3") and cand.get("k") in decided:
                done_repos.add(key) if kind == "B3" else done_accounts.add(key)
                continue
            if kind not in ("B1", "B2", "B3") or not _sound_candidate(cand) or rejected(f"{cand['account']}/x") or rejected(key):
                continue
            if (kind == "B3" and key in rows) or (kind != "B3" and (key in listed or any(not isinstance(r, str) or rejected(r) for r in _named(cand)))):
                continue  # a rejected model is no evidence: the candidate comes again from a run that finds it without one
            candidates[(kind, key)] = cand
        except (KeyError, TypeError, AttributeError):
            continue
    known = {key[1] for key in changes if key[0] == "new_row"} | {key for kind, key in candidates if kind == "B3"} | done_repos
    return changes, candidates, known, done_fields, done_accounts


def collection_members(data: list, owner: str) -> dict[str, str]:
    """Which collection of `owner` holds which repo: the first collection (in the Hub's order) wins."""
    members: dict[str, str] = {}
    for collection in data:
        slug = collection.get("slug") if isinstance(collection, dict) else None
        if not (isinstance(slug, str) and slug.startswith(f"{owner}/") and isinstance(collection.get("items"), list)):
            continue
        for item in collection["items"]:
            if isinstance(item, dict) and item.get("type") == "model" and isinstance(item.get("id"), str):
                members.setdefault(item["id"], f"{HUB}/collections/{slug}")
    return members


def card_texts(card: dict) -> list[str]:
    """Every string of a card except the base-model fields, which name other repos on purpose."""
    texts, stack = [], [v for k, v in card.items() if k not in ("base_model", "base_model_relation")]
    while stack:
        item = stack.pop()
        texts += [item] if isinstance(item, str) else []
        stack += list(item.values()) if isinstance(item, dict) else item if isinstance(item, list) else []
    return texts


def b1_ok(entry: Entry, now: datetime) -> bool:
    """The thresholds of B1 on one repo: young enough, liked and downloaded enough, own base model, no quantization."""
    return (
        entry.created is not None and now - entry.created <= B1_MAX_AGE and entry.likes >= B1_MIN_LIKES
        and entry.downloads >= B1_MIN_DOWNLOADS and not any(t.startswith("base_model:") for t in entry.tags)
        and not is_quantization(entry.tags, entry.repo)
    )


def split_candidates(candidates: list[dict]) -> tuple[list[dict], int]:
    """B1 and B2 by likes, at most ten (and how many were left out); every B3 stays."""
    open_ones = sorted((c for c in candidates if c["candidate"] != "B3"), key=lambda c: (-c["likes"], c["account"], c["candidate"]))
    return open_ones[:MAX_CANDIDATES] + [c for c in candidates if c["candidate"] == "B3"], max(0, len(open_ones) - MAX_CANDIDATES)


# --- order, application, rendering -----------------------------------------------------------------------------------


def order_proposal(changes: list[dict], candidates: list[dict]) -> None:
    """Fixed order, then the numbers: changes by (stage, repo, field), B1 and B2 by likes then account, B3 by age."""
    changes.sort(key=lambda c: (c["stage"], c["hf_repo"], c.get("field") or ""))
    for number, change in enumerate(changes, start=1):
        change["n"] = number
    open_ones = sorted((c for c in candidates if c["candidate"] != "B3"), key=lambda c: (-c["likes"], c["account"], c["candidate"]))
    third = sorted((c for c in candidates if c["candidate"] == "B3"), key=lambda c: (c["open_since"], c["hf_repo"]))
    candidates[:] = open_ones + third
    for number, candidate in enumerate(candidates, start=1):
        candidate["k"] = f"K{number}"


def apply_changes(document: dict, changes: list[dict]) -> dict:
    """A copy of the catalog document (as read from JSON) with `changes` applied; the original is untouched."""
    result = copy.deepcopy(document)
    for change in changes:
        if change["change"] == "new_row":
            result["models"].append({name: pair["new"] for name, pair in change["fields"].items()})
            continue
        row = next(m for m in result["models"] if m["hf_repo"] == change["hf_repo"])
        row.update({name: pair["new"] for name, pair in change["fields"].items()})
    return result


def check_changes(document: dict, changes: list[dict]) -> tuple[list[dict], list[str]]:
    """Each change as its own proposal against the table, then all kept ones together (schema 2 with its rules).

    A change that would not validate is left out and named in the notes; `ValueError` when the rest does not
    validate together, in which case nothing may be written.
    """
    kept, notes = [], []
    for change in changes:
        try:
            build_catalog_v2(json.dumps(apply_changes(document, [change])), Path("catalog.json"))
        except (CatalogError, SchemaVersionError) as exc:
            notes.append(f"{change['hf_repo']}: left out, it would not validate ({' '.join(str(exc).split())[:120]})")
        else:
            kept.append(change)
    try:
        build_catalog_v2(json.dumps(apply_changes(document, kept)), Path("catalog.json"))
    except (CatalogError, SchemaVersionError) as exc:
        raise ValueError(f"the proposal does not validate against schema 2: {' '.join(str(exc).split())[:200]}") from exc
    return kept, notes


def proposal_document(day: str, base_sha: str, notes: list[str], changes: list[dict], candidates: list[dict], incomplete: int, pending) -> dict:
    return {
        "schema": SCHEMA, "date": day, "base_catalog_sha256": base_sha, "notes": notes, "changes": changes, "candidates": candidates,
        "summary": {
            "new": sum(c["change"] == "new_row" for c in changes), "changed": sum(c["change"] == "field" for c in changes),
            "candidates": len(candidates), "incomplete": incomplete, "pending_publication": pending,
        },
    }


def dump_json(data: object) -> bytes:
    """The bytes of every JSON file of the scanner: sorted keys, two spaces, LF, a final newline, no BOM."""
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def stamp(moment: datetime) -> str:
    """A time as the Hub writes it (`2026-10-03T08:15:30.000Z`)."""
    moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def render_proposal(data: dict, sha256: str | None = None) -> str:
    """`proposal.md`: one header line per number, under it every field as `field: old -> new`, whole and unabridged."""
    summary = data.get("summary", {})
    lines = [
        f"# Catalog proposal {data['date']}", f"base catalog sha256: {data['base_catalog_sha256']}",
        f"{summary.get('new', 0)} new, {summary.get('changed', 0)} changed, {summary.get('candidates', 0)} candidates, "
        f"{summary.get('incomplete', 0)} incomplete",
    ]
    if summary.get("pending_publication"):
        lines.append("publication pending: " + json.dumps(summary["pending_publication"], sort_keys=True))
    lines += ["", "## Changes", ""]
    for change in data["changes"]:
        lines.append(f"{change['n']} | {change['stage']} | {change['hf_repo']} | kind {change['kind']} | "
                     f"evidence {' '.join(change.get('evidence', []))} | open since {change['open_since']}")
        if change.get("kind_evidence"):
            lines.append("  kind evidence: " + json.dumps(change["kind_evidence"], sort_keys=True))
        for name in sorted(change["fields"]):
            pair = change["fields"][name]
            lines.append(f"  {name}: {json.dumps(pair['old'])} -> {json.dumps(pair['new'])}")
    lines += ["", "## Candidates", ""]
    for cand in data["candidates"]:
        lines.append(f"{cand['k']} | {cand['candidate']} | {cand.get('account') or cand.get('hf_repo')} | kind {cand.get('kind') or '-'} | "
                     f"evidence {' '.join(cand.get('evidence', []))} | open since {cand['open_since']}")
        shown = {"k", "candidate", "evidence", "open_since", "kind"}
        lines += [f"  {name}: {json.dumps(cand[name], sort_keys=True)}" for name in sorted(cand) if name not in shown]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in data["notes"]]
    lines += ["", f"proposal.json sha256: {sha256 or hashlib.sha256(dump_json(data)).hexdigest()}"]
    return "\n".join(lines) + "\n"

