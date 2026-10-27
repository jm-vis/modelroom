"""The rules of the catalog publisher (`catalog_publish.py`), as pure functions: no file, no process, no request, no clock.

What the maintainer named (`--approve`, `--reject`, `--admit`), what that does to the table and the rejection list, the
shape of `decision.json`, how a candidate of open kind becomes a row (`--complete`), what `--show` prints and the exact
argument list of the one upload process live here, so a test hands a rule a value and reads the value back. The
proposal, its numbers, `apply_changes`, `check_changes` and the rendering are the scanner's own (`catalog_scan_rules`);
this module adds no second rule for them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the scanner's rules sit beside this module
import catalog_scan_rules as R  # noqa: E402

from modelroom.catalog import PIPELINE_TAG_TO_KIND, CatalogError  # noqa: E402
from modelroom.catalog_remote import CHANNEL_V2, CHANNEL_V3, CHANNEL_V4, REMOTE_FILE_NAME  # noqa: E402
from modelroom.catalog_schema import (  # noqa: E402
    SCHEMA3_ROW_DEFAULTS, build_catalog, build_catalog_v2, build_catalog_v3, build_catalog_v4, project_to_schema2, project_to_schema3,
)
from modelroom.contracts import SchemaVersionError  # noqa: E402

KINDS = tuple(sorted(set(PIPELINE_TAG_TO_KIND.values())))  # the eight kinds of the table
# The note `--complete` writes for a kind the maintainer set (the line in `complete_proposal`); the suite hides exactly this form.
MAINTAINER_KIND_NOTE = re.compile(r"Kind (?:" + "|".join(map(re.escape, KINDS)) + r") set by the maintainer on [0-9]{4}-[0-9]{2}-[0-9]{2}\.")
STATES = R.DECISION_STATES  # approved, applied, committed, pushed, uploaded, published: the order of a publication
CATALOG_IN_REPO, REJECTED_IN_REPO = "modelroom/catalog.json", "modelroom/catalog-rejected.json"
# The version of `huggingface_hub` the upload runs with, through `uvx`: no dependency of the package. `uv.lock` names none,
# so this is the newest release the build machine knew; the setup test of the runbook is what shows that this call form works.
HF_HUB_VERSION = "1.28.0"
_NUMBER, _CANDIDATE = re.compile(r"[1-9][0-9]{0,5}"), re.compile(r"K[1-9][0-9]{0,5}")
_SHA256, _COMMIT = re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{40}")
_REMOTE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class PublishError(Exception):
    """The call or one of the inputs is wrong: exit 2, nothing written."""


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_sha(given: str | None, data: bytes) -> str:
    """The checksum the maintainer was shown, against the bytes that are read now."""
    if given is None or not _SHA256.fullmatch(given):
        raise PublishError("--proposal-sha is missing or is not 64 lower-case hex digits")
    if given != sha256_of(data):
        raise PublishError("the proposal has changed since it was shown: --proposal-sha does not match its bytes")
    return given


def check_commit(commit: str) -> str:
    """A full commit hash: nothing else reaches `git`, so a name can never be read as an option."""
    if not _COMMIT.fullmatch(commit or ""):
        raise PublishError("the commit must be a full hash of 40 lower-case hex digits")
    return commit


def check_remote(remote: str) -> str:
    if not _REMOTE.fullmatch(remote or ""):
        raise PublishError("the remote is not a plain name")
    return remote


def upload_args(dataset: str, path: object, message: str) -> list[str]:
    """The one upload process: a fixed argument list, the pinned tool, the file, the target name in the dataset."""
    return ["uvx", "--from", f"huggingface_hub=={HF_HUB_VERSION}", "hf", "upload", dataset, str(path), REMOTE_FILE_NAME,
            "--repo-type", "dataset", "--commit-message", message]


def channel_files(catalog_bytes: bytes) -> dict[str, bytes]:
    """The three files of the dataset that a working catalog (schema 2, 3 or 4) makes, `{file name: bytes}`, newest channel first (AP9-KO, V3).

    `catalog-v4.json` is the schema-4 document: the working catalog itself where it is schema 4, the same rows with
    `schema_version` 4 where it is older (a document of schema 3 reads as 4: its note holds no `vllm_args`).
    `catalog-v3.json` is what release 1.1.0 reads: the working catalog byte for byte where it is schema 3, the same rows
    with `schema_version` 3 where it is schema 2, and else `project_to_schema3` of the document (the switches of a note
    are not carried to it). `catalog-v2.json` is what release 1.0.0 reads: the working catalog byte for byte where it is
    schema 2, and else `project_to_schema2` of the schema-3 form -- made for the old channel by setting the two findings
    of every row to their default first (the old file carries none, and the projection refuses a finding it would
    otherwise drop). Every file is read back with the reader of its channel."""
    try:
        document = json.loads(catalog_bytes.decode("utf-8"))
        version = document.get("schema_version") if isinstance(document, dict) else None
        if version not in (2, 3, 4):
            raise PublishError("the working catalog is none of schema 2, 3 or 4")
        build_catalog(catalog_bytes.decode("utf-8"), Path("catalog.json"))
        v4 = catalog_bytes if version == 4 else R.dump_json({**document, "schema_version": 4})
        if version == 4:
            v3 = R.dump_json(project_to_schema3(document))
        else:
            v3 = catalog_bytes if version == 3 else R.dump_json({**document, "schema_version": 3})
        if version == 2:
            v2 = catalog_bytes
        else:
            plain = json.loads(v3.decode("utf-8"))
            for row in plain.get("models", []):
                row.update(copy.deepcopy(SCHEMA3_ROW_DEFAULTS))
            v2 = R.dump_json(project_to_schema2(plain))
        build_catalog_v4(v4.decode("utf-8"), Path(CHANNEL_V4.file_name))
        build_catalog_v3(v3.decode("utf-8"), Path(CHANNEL_V3.file_name))
        build_catalog_v2(v2.decode("utf-8"), Path(CHANNEL_V2.file_name))
    except PublishError:
        raise
    except (ValueError, AttributeError, TypeError, CatalogError, SchemaVersionError) as exc:
        raise PublishError(f"the working catalog makes no channel files: {' '.join(str(exc).split())[:160]}") from exc
    return {CHANNEL_V4.file_name: v4, CHANNEL_V3.file_name: v3, CHANNEL_V2.file_name: v2}


def upload_folder_args(dataset: str, folder: object, message: str) -> list[str]:
    """The upload process of a folder: a fixed argument list, the pinned tool, a folder of its own sent to the root of the
    dataset (the card, and the three channel files of the catalog in one commit)."""
    return ["uvx", "--from", f"huggingface_hub=={HF_HUB_VERSION}", "hf", "upload", dataset, str(folder), ".",
            "--repo-type", "dataset", "--commit-message", message]


# --- what the maintainer named ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """The three lists as the maintainer gave them: numbers (`"3"`) and candidates (`"K1"`), as text, in the given order."""

    approve: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()
    admit: tuple[str, ...] = ()
    approve_unread: tuple[str, ...] = ()  # numbers of `approve` for which the maintainer confirms a row without a reading

    @property
    def named(self) -> tuple[str, ...]:
        return self.approve + self.reject + self.admit


def _tokens(text: str | None, option: str, pattern: re.Pattern[str], also: re.Pattern[str] | None = None) -> tuple[str, ...]:
    if text is None:
        return ()
    found = tuple(part.strip() for part in text.split(","))
    for part in found:
        if not (pattern.fullmatch(part) or (also is not None and also.fullmatch(part))):
            raise PublishError(f"{option}: {part!r} is not a name of the proposal's list")
    return found


def parse_request(approve: str | None, reject: str | None, admit: str | None, approve_unread: str | None = None) -> Request:
    """`--approve` takes numbers (a candidate is refused later, with its own sentence), `--reject` numbers and candidates,
    `--admit` candidates only, `--approve-unread` numbers only."""
    return Request(_tokens(approve, "--approve", _NUMBER, _CANDIDATE), _tokens(reject, "--reject", _NUMBER, _CANDIDATE),
                   _tokens(admit, "--admit", _CANDIDATE), _tokens(approve_unread, "--approve-unread", _NUMBER))


def _proposal_parts(proposal: dict) -> tuple[dict, dict]:
    try:
        changes = {str(c["n"]): c for c in proposal["changes"]}
        candidates = {c["k"]: c for c in proposal["candidates"]}
    except (KeyError, TypeError) as exc:
        raise PublishError("the proposal has no numbered changes and candidates") from exc
    return changes, candidates


def check_request(proposal: dict, request: Request) -> None:
    """The refusals of `--apply`, in the brief's order: nothing named, a name that is not in the proposal, a name twice, a
    candidate to approve, a candidate of open kind to admit."""
    changes, candidates = _proposal_parts(proposal)
    if not request.named:
        raise PublishError("nothing is named: give at least one of --approve, --reject, --admit")
    for name in request.named:
        if name not in changes and name not in candidates:
            raise PublishError(f"{name} is not in the proposal")
    twice = sorted({name for name in request.named if request.named.count(name) > 1})
    if twice:
        raise PublishError(f"named twice: {', '.join(twice)}")
    for name in request.approve:
        if name.startswith("K"):
            raise PublishError(f"{name} is a candidate: --approve takes numbers; --admit takes a candidate of a new account")
    for name in request.admit:
        if candidates[name]["candidate"] == "B3":
            raise PublishError(f"{name} is a repo of open kind: only --complete can make it a row")
    _check_unread(changes, request)


def _check_unread(changes: dict, request: Request) -> None:
    """`--approve-unread` confirms rows of `--approve`; it is no second decision: each number is in the proposal, named
    once, and also in `--approve`."""
    for name in request.approve_unread:
        if name not in changes:
            raise PublishError(f"{name} is not in the proposal")
    twice = sorted({name for name in request.approve_unread if request.approve_unread.count(name) > 1})
    if twice:
        raise PublishError(f"named twice in --approve-unread: {', '.join(twice)}")
    for name in request.approve_unread:
        if name not in request.approve:
            raise PublishError(f"--approve-unread {name} needs --approve {name}")


def _order(names) -> list:
    """Numbers first (as numbers), then the candidates (as `Kk`), each in numeric order."""
    return [int(n) for n in sorted((n for n in names if not n.startswith("K")), key=int)] + sorted((n for n in names if n.startswith("K")), key=lambda k: int(k[1:]))


# --- the result of an --apply ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """What `--apply` fixes before it touches a file: the target text of both files and the lists as numbers and `Kk`."""

    catalog_text: str
    rejected_text: str
    approved: list
    rejected: list
    admitted: list
    approved_unread: list = field(default_factory=list)  # the approved rows without a reading the maintainer confirmed


def _admission(document: dict, candidate: dict) -> None:
    account = candidate["account"]
    if candidate["candidate"] == "B1":
        document.setdefault("publishers", []).append({"account": account, "name": account, "region": None})
    else:
        document.setdefault("packagers", []).append(account)


def _rejection(proposal_parts: tuple[dict, dict], name: str) -> tuple[str, str]:
    """`(key, value)` of the entry a rejected number or candidate makes: the repo of the change, the account of a B1 or B2
    candidate, the repo of a candidate of open kind."""
    changes, candidates = proposal_parts
    if name in changes:
        return "hf_repo", changes[name]["hf_repo"]
    candidate = candidates[name]
    return ("hf_repo", candidate["hf_repo"]) if candidate["candidate"] == "B3" else ("account", candidate["account"])


def _name_pair(proposal_parts: tuple[dict, dict], name: str) -> tuple[str, str] | None:
    """`(repo, successor)` of a rejected number that is a successor of the name rule (stage `A4`), else `None`: the entry a
    rejection makes for it names the pair, so the row is proposed other successors and other fields still."""
    change = proposal_parts[0].get(name)
    if change is None or change.get("stage") != "A4" or change.get("field") != "successor":
        return None
    return change["hf_repo"], change["fields"]["successor"]["new"]


def _check_reading(change: dict, name: str, request: Request) -> None:
    """A new row whose name the grammar cannot read is taken only with `--approve-unread` for its number. The reading is
    worked out here from the repo and the size of the row; the `reading` mark of the proposal is for the reader only, so a
    missing, old or wrong mark passes nothing."""
    if name not in request.approve_unread and R.unread_of(change):
        raise PublishError(f"row {name} has no reading; approve it with --approve-unread {name} after adding it to SHIPPED_WITHOUT_READING")


def plan_apply(document: dict, catalog_bytes: bytes, rejected_bytes: bytes, proposal: dict, request: Request, today: str) -> Plan:
    """The target of an `--apply`, in memory and validated: the table with exactly the approved changes and the admitted
    accounts (`published` is the day of the call when the table changes at all), the rejection list with the named repos and
    accounts added once. A file nobody changes keeps its bytes."""
    parts = _proposal_parts(proposal)
    taken = [parts[0][name] for name in sorted(request.approve, key=int)]
    for name, change in zip(sorted(request.approve, key=int), taken):
        _check_reading(change, name, request)
    try:
        target = R.apply_changes(document, taken) if taken else copy.deepcopy(document)
        for name in sorted(request.admit, key=lambda k: int(k[1:])):
            _admission(target, parts[1][name])
        if taken or request.admit:
            target["published"] = today
            build_catalog(json.dumps(target), Path("catalog.json"))
    except (CatalogError, SchemaVersionError) as exc:
        raise PublishError(f"the table would not validate: {' '.join(str(exc).split())[:200]}") from exc
    except (StopIteration, KeyError, TypeError, ValueError) as exc:
        raise PublishError(f"a named change cannot be applied: {type(exc).__name__}") from exc
    catalog_text = R.dump_json(target).decode("utf-8") if taken or request.admit else catalog_bytes.decode("utf-8")
    try:
        repos, accounts = R.parse_rejected(rejected_bytes)
        pairs = R.parse_rejected_pairs(rejected_bytes)
        entries = list(json.loads(rejected_bytes.decode("utf-8")))
        before = len(entries)
        for name in _order(request.reject):
            pair = _name_pair(parts, str(name))
            if pair is not None:  # a name-rule proposal: the pair is turned down, never the row
                if pair not in pairs:
                    pairs.add(pair)
                    entries.append({"hf_repo": pair[0], "successor": pair[1], "date": today})
                continue
            key, value = _rejection(parts, str(name))
            if value in (repos if key == "hf_repo" else accounts):
                continue  # on the list already: no duplicate
            (repos if key == "hf_repo" else accounts).add(value)
            entries.append({key: value, "date": today})
        rejected_text = R.dump_json(entries).decode("utf-8") if len(entries) != before else rejected_bytes.decode("utf-8")
        R.parse_rejected(rejected_text.encode("utf-8"))
    except ValueError as exc:
        raise PublishError(f"the rejection list is not valid: {' '.join(str(exc).split())[:200]}") from exc
    return Plan(catalog_text, rejected_text, _order(request.approve), _order(request.reject), _order(request.admit), _order(request.approve_unread))


def decision_document(plan: Plan, proposal_sha: str, today: str, base_catalog_sha: str, base_rejected_sha: str) -> dict:
    """`decision.json` as step 1 writes it: the lists, both bases, both targets as text with their checksums, `approved`."""
    return {
        "proposal_sha": proposal_sha, "approved": plan.approved, "approved_unread": plan.approved_unread, "rejected": plan.rejected,
        "admitted": plan.admitted, "date": today,
        "base_catalog_sha256": base_catalog_sha, "base_rejected_sha256": base_rejected_sha,
        "result_catalog": plan.catalog_text, "result_rejected": plan.rejected_text,
        "result_sha256": sha256_of(plan.catalog_text.encode("utf-8")), "result_rejected_sha256": sha256_of(plan.rejected_text.encode("utf-8")),
        "state": "approved", "commit": None,
    }


def _number(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _candidate(value: object) -> bool:
    return isinstance(value, str) and _CANDIDATE.fullmatch(value) is not None


def _unread_sound(decision: dict) -> bool:
    """`approved_unread` (AP9-KA-H): absent in a decision of the old form, which counts as an empty list; else a list of
    different numbers that are all in `approved`."""
    if "approved_unread" not in decision:
        return True
    found = decision["approved_unread"]
    return isinstance(found, list) and all(_number(v) for v in found) and len(set(found)) == len(found) and set(found) <= set(decision["approved"])


def check_decision(raw: bytes) -> dict:
    """A `decision.json` as this tool and the scanner read it: numbers as `int` (a `bool` is none), candidates as `Kk`;
    a damaged one is an error, never guessed at."""
    try:
        decision = json.loads(raw.decode("utf-8"))
        texts = {"result_catalog": "result_sha256", "result_rejected": "result_rejected_sha256"}
        lists = {"approved": _number, "rejected": lambda v: _number(v) or _candidate(v), "admitted": _candidate}
        sound = (
            isinstance(decision, dict) and decision.get("state") in STATES
            and all(isinstance(decision.get(key), list) and all(ok(v) for v in decision[key]) for key, ok in lists.items())
            and all(isinstance(decision.get(key), str) and _SHA256.fullmatch(decision[key]) for key in ("base_catalog_sha256", "base_rejected_sha256", "proposal_sha", *texts.values()))
            and all(isinstance(decision.get(text), str) and sha256_of(decision[text].encode("utf-8")) == decision[digest] for text, digest in texts.items())
            and (decision.get("commit") is None or isinstance(decision["commit"], str))
            and _unread_sound(decision)
        )
    except (ValueError, AttributeError):
        sound = False
    if not sound:
        raise PublishError("decision.json is damaged: its state, lists or result checksums do not read")
    return decision


# --- --complete --------------------------------------------------------------------------------------------------------


def complete_proposal(document: dict, proposal: dict, request: Request, kind: str | None, today: str) -> tuple[dict, int]:
    """The proposal with one candidate of open kind made a row: `(new proposal, its number)`.

    The row is complete as the scanner would build it, with the kind the maintainer named and `kind_source` `decision`;
    the family comes from the table as it is, `latest` from the group, the source of `latest` is the publisher's page (no
    request is made here). The row must pass the schema alone and together with the other changes; the numbers and the
    order are made again, so a checksum of the old file never fits.
    """
    if len(request.admit) != 1 or request.approve or request.reject:
        raise PublishError("--complete takes exactly one candidate in --admit and nothing else")
    if kind not in KINDS:
        raise PublishError(f"--kind must be one of the eight kinds: {', '.join(KINDS)}")
    _, candidates = _proposal_parts(proposal)
    candidate = candidates.get(request.admit[0])
    if candidate is None or candidate["candidate"] != "B3":
        raise PublishError(f"{request.admit[0]} is not a repo of open kind: --complete is for B3 only")
    try:
        rows = [m for family in build_catalog(json.dumps(document), Path("catalog.json")).families for m in family.models]
        entry = R.entry_of_candidate(candidate)
        if entry.pipeline in PIPELINE_TAG_TO_KIND:
            # The other direction of the rule: a task of the table gives the kind on its own, so no decision may stand
            # next to it (the schema would refuse `kind_source` on such a row). The scanner does not make a B3 of such a
            # candidate; this stops a hand-changed `proposal.json`.
            raise PublishError(
                f"{entry.repo} carries the task {entry.pipeline}, which gives kind {PIPELINE_TAG_TO_KIND[entry.pipeline]} on its own: "
                "--complete is for a candidate without a table task"
            )
        family = R.family_for(entry.repo, kind, rows, entry.parameters_b)
    except (CatalogError, SchemaVersionError, KeyError, TypeError, ValueError) as exc:
        raise PublishError(f"the candidate cannot be read: {type(exc).__name__}") from exc
    if family is None:
        raise PublishError(f"{entry.repo}: no family word in the name, no row can be made")
    result = R.KindResult(kind, "decision", False, {"stage": "decision", "value": kind})
    latest = R.latest_among([(entry.repo, kind, entry.parameters_b)], rows).get(entry.repo) == "over latest"  # as before AP9-KA-H: no `family top` here
    row = R.build_row(entry, result, family=family, today=today, latest=latest, latest_source=f"{R.HUB}/{entry.owner}")
    row["note"] = f"Kind {kind} set by the maintainer on {today}."
    if MAINTAINER_KIND_NOTE.fullmatch(row["note"]) is None:
        raise PublishError(f"the note for the kind does not have the form of MAINTAINER_KIND_NOTE (date {today!r})")
    change = R.mark_reading(R.new_row_change(row, result, candidate["open_since"]))
    changes = [R.mark_reading(c) for c in copy.deepcopy(proposal["changes"])] + [change]
    try:
        kept, notes = R.check_changes(document, changes)
    except ValueError as exc:
        raise PublishError(str(exc)) from exc
    if len(kept) != len(changes):
        raise PublishError(f"the row would not pass the schema together with the proposal: {'; '.join(notes) or 'a change is left out'}")
    return _completed(proposal, changes, request.admit[0]), next(c["n"] for c in changes if c["hf_repo"] == entry.repo and c["change"] == "new_row")


def _completed(proposal: dict, changes: list[dict], admitted: str) -> dict:
    """The proposal with `changes` (the new row among them) and without the candidate `admitted`; numbers made again."""
    data = copy.deepcopy(proposal)
    candidates_left = [c for c in data["candidates"] if c["k"] != admitted]
    R.order_proposal(changes, candidates_left)
    data["changes"], data["candidates"] = changes, candidates_left
    data["summary"].update(new=sum(c["change"] == "new_row" for c in changes), changed=sum(c["change"] == "field" for c in changes),
                           candidates=len(candidates_left))
    return data


# --- --show ------------------------------------------------------------------------------------------------------------


def pending_line(day: str, decision: dict) -> str:
    return f"publication pending: {day}, state {decision['state']}, commit {decision.get('commit') or 'none'}"


def render_open(proposal_bytes: bytes) -> str:
    """The text of `proposal.md`, made from exactly these bytes and their checksum."""
    try:
        return R.render_proposal(json.loads(proposal_bytes.decode("utf-8")), sha256_of(proposal_bytes))
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PublishError(f"the open proposal cannot be shown: {type(exc).__name__}") from exc


def base_line(proposal_bytes: bytes, catalog_sha: str) -> tuple[bool, str]:
    """`(current, line)`: whether the proposal was made against the table as it is."""
    try:
        base = json.loads(proposal_bytes.decode("utf-8"))["base_catalog_sha256"]
    except (ValueError, KeyError, TypeError):
        base = None
    if base == catalog_sha:
        return True, "base: current"
    return False, "base: stale, the proposal was made against another catalog; run the scanner again"
