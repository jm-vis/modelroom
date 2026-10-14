"""Stage 1 of the catalog run: the package findings and the setup notes (AP9-KO, V1, V5, V6, V7).

For each catalog row of kind `text` or `vision` that a run takes (`catalog_findings.package_rows_for_run`), in this order:

1. the card of the publisher's repository is read at a revision fixed first, and the rule of `catalog_scan_card` decides
   whether it holds setup instructions for vLLM (`SetupNote`, V7); one range request for the window `CARD_MAX_BYTES`,
   and of a longer card only the start counts, up to its last whole paragraph and for a positive finding only (no
   finding there: `not_read`); whether a `206` is the whole card or a start its `Content-Range` must prove;
2. the GGUF repositories that quantize the row are listed through the relation (`filter=base_model:quantized:<row>`,
   `gguf` and `sha` requested), whoever owns them, and every weight package of each is checked: the architecture it
   states is compared with the block of the llama.cpp source of one commit (`PackageCheck`, V1, V5).

The architecture of a package comes from the Hub's `gguf` field only for the one file `hub_metadata_file` finds by its
size (`architecture_source` `hub_field`); every other package is read from the first bytes of its first shard
(`file_header`, a range request on the Hub), and a package that cannot be read stays without a check (`unknown`).

The stage only reads. It returns the lists and notes it wants the proposal to carry; the scanner writes them as
proposals, and setting or removing goes through the existing approval. A fault of the source, the network, the budget or a
parser adds no `absent`, removes nothing, and leaves the day of every stored check as it was. The measure of a fault is a
realistic fault of the network or of an answer, and it fails only its unit (a card, a package, a repository, a row),
never the run.

**Every unit is reached in a finite number of runs, whatever the others do (V6).** Rows go by `package_attempts`
(`package_rows_for_run`). Inside a row the same rule holds one level down, in the scanner's state `package_unit_attempts`
(`record_units`: the form of `record_attempts`, the repository compared without case, the identity as written): a GGUF
repository is recorded when its read ended, with a check or with a fault, and a package when the read of its head failed; the repositories of a row are read never-tried first, then the one tried longest
ago (the order of downloads among equals), the packages of a repository likewise (a package with a check costs no request).
So a repository or a package that fails every day goes behind the others. The budget is no fault of a unit: the one it cuts
is not recorded and comes first next time, and what was read before it is kept. A row the budget cuts stays untried, so the
next run takes it first and goes on behind what it recorded -- except the first row of a run that sent a request for it:
it had the whole budget, waiting gives it no more, so it is tried and the rows behind it are not held back. A row for
which no request was sent (the budget was spent before it) is never tried.

**A fixed share for every row (NB).** One row with many repositories must not use the run. What is left of the budget
after the one-time reads (the commit and the file of llama.cpp) is divided by the rows the run takes, and each row may
send `floor(rest / rows)` requests (card, list and repositories together); a request over the share is not sent and the
row ends. The least share is `MIN_ROW_SHARE`, 4: the model answer, the card, one page of the list and the file list of
one repository, so that a row with a quiet card and a short list reads at least one repository; when the budget is too
small for that many rows, the run takes fewer (`budget ended`), as it did before. A row that got its share is **tried**,
though units stay open: a row that waited for all its units would be taken first again and hold the rotation as before.
What the share cuts is no fault: the repository or package it cut is not recorded (see above) and comes first next
time, which keeps V6. Round 1 gives every row its share; round 2 gives what is left of the budget to the rows that were
cut, the one that got it longest ago first (`package_reserve_days` in the scanner's state, ordered by day, then by the order the
rows were taken), so that every cut row comes round in a finite number of runs and none keeps it for ever. The reserve kept back
from round 1 for this is never nil: `max(RESERVE_MIN, rest // 10)`, `RESERVE_MIN` 8, that is the progress of one open unit with what it
needs first. V6 holds as long as the reserve carries the rest of a row's list pages, one file list and one head (two requests when the Hub redirects the read to its CDN): pages <= (share - 2) + (round 2 budget - 3), where round 1 reads share - 2 pages after the model answer and the card and round 2 has what round 1 left, at least the reserve. At the default (198 requests for the stage, 40 rows) the reserve is 19, the share 4 (two list pages in round 1) and round 2 has 38 requests: a list of up to 37 pages, some 1850 repositories at `LIST_PAGE`, is reached; with 16 requests and two rows, 8 pages (tested). A longer list is of the class the management put aside as unrealistic on 2026-10-14 (a list that takes the budget): it is not reached, the run ends without a fault, the row is tried, and the stand still is the same the next day; nothing is deleted. When the budget cannot pay it beside
`MIN_ROW_SHARE` for every row, round 1 takes fewer rows and the others rotate on as before; a budget below
`RESERVE_MIN + MIN_ROW_SHARE` (12) gives its one row all of it and keeps no reserve (taken as unrealistic, like the other
bounds). Known bound of V6: it presumes that a row runs through its units once inside `UNIT_ATTEMPTS_KEPT` (365 days), after which a unit counts as never tried again. At the default budget and with realistic catalogs that holds; extreme catalogs (for example 40 rows with some ninety repositories each) lie beyond it and are taken as unrealistic, like the other bounds (decision of the management, 2026-10-14).
A row is recorded as having got the reserve only when it made progress in round 2 (a unit read to its end, a head read or
failed); a leftover without progress books nothing and the row keeps its priority. a cut row may take it again though all its repositories were read, because a
new package or revision of a visited repository needs it too, and the rotation keeps the others from starving (their card is not asked again, a list the share cut is continued at its next address, repositories read in
round 1 are skipped). Inside a row the order is: never tried first, then the one tried longest ago; among equals the repositories
of the publisher itself, then of the catalog's listed packagers, then all others (the class decides only among equals, or
a class with many repositories would keep the next one unread for ever); the one without a check, then downloads, last.
A list of more than one page (card and list over the share) is read to its repositories in round 2 only; inside a run it is
continued where the share ended, over runs it is not continued page by page (decision of the management, 2026-10-14). Bound: a row whose list alone takes the
whole budget of a run (some ten thousand repositories at `LIST_PAGE`) gets no repository read, and a file list over
more than one page (some thousand entries) is a fault of its repository; both are taken as unrealistic (decision of the
management, 2026-10-14), neither is continued page by page.
"""

from __future__ import annotations

import http.client
import re
import struct
import urllib.parse
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Mapping, NamedTuple, Sequence

import catalog_scan_card as C
import catalog_scan_rules as R

from modelroom.catalog import CatalogModel
from modelroom.catalog_findings import (
    CARD_MAX_BYTES,
    LLAMACPP_COMMIT_ACCEPT,
    LLAMACPP_COMMIT_MAX_BYTES,
    LLAMACPP_FILE_ACCEPT,
    LLAMACPP_FILE_MAX_BYTES,
    LLAMACPP_HOST,
    LLAMACPP_MAX_REDIRECTS,
    ArchBlockError,
    PackageCheck,
    SetupNote,
    compare_architecture,
    hub_metadata_file,
    llamacpp_block_url,
    llamacpp_commit_url,
    llamacpp_page_url,
    merge_package_checks,
    package_rows_for_run,
    parse_arch_names,
    read_commit,
)
from modelroom.catalog_remote import _clean
from modelroom.contracts import hf_id_key
from modelroom.http import (
    _ALLOWED_HTTPS_HOSTS,
    BudgetExhaustedError,
    RedirectingTransport,
    ResponseTooLargeError,
    Transport,
    TransportSecurityError,
)
from modelroom.hf import _LINK_NEXT_RE
from modelroom.quantization import identity_stem, non_weight_role, parse_hf_quant
from modelroom.relation import check_relation

HUB = "https://huggingface.co"
HEADER_BYTES = 64 * 1024  # the range asked of a file, and the most the answer may hold
LIST_PAGE = 50  # the size of a page of the list query; the list is followed to its end, the budget alone limits a run
CHECKS_PER_ROW = 200  # the limit of the schema (`CatalogModel.gguf_checks`)
STAGE = "A9"  # the stage mark of both proposals
UNIT_ATTEMPTS_KEPT = 365  # days a unit's last try is kept in the state; one older counts as never tried, which is as early
RESERVE_MIN = 8  # the least kept back for round 2 (the bound of the list pages is in the module text)
MIN_ROW_SHARE = 4  # the least a row is given: the model answer, the card, one page of the list and the file list of one repository
_FAILURES = (OSError, http.client.HTTPException, RuntimeError, ResponseTooLargeError, TransportSecurityError)
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_ARCH_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_SHARD_RE = re.compile(r"-(?:split-)?(\d{5})-of-\d{5}\.gguf$", re.IGNORECASE)
_NEXT_RE = re.compile(r'rel="?next"?', re.IGNORECASE)
_CONTENT_RANGE_RE = re.compile(r"bytes (\d{1,12})-(\d{1,12})/(\d{1,12})")
_RELATION_RE = re.compile(r"^(base_model:(?:[a-z]+:)?)(.+)$")


# --- the hosts of the scanner (V5): the runtime's list plus the source and the Hub's CDN, and nowhere else -------------------


class ScannerHosts:
    """The HTTPS hosts the scanner's transport may open: the runtime's three, the host of the llama.cpp source, and the
    Hub's own CDN (`*.hf.co`), to which a read of a file's first bytes is redirected. `http._ALLOWED_HTTPS_HOSTS` is not
    touched; this object is handed to the scanner's `UrllibTransport` only."""

    def __contains__(self, host: object) -> bool:
        return isinstance(host, str) and (host in _ALLOWED_HTTPS_HOSTS or host == LLAMACPP_HOST or (host.endswith(".hf.co") and len(host) > len(".hf.co")))


class _Gate:
    """Lets a request through only to the hosts `allowed` names: the layer of the source sees `api.github.com` alone, the
    API layers `huggingface.co` alone, the file layer the Hub and its CDN. A hop that leads elsewhere is refused unsent."""

    def __init__(self, inner: Transport, allowed: Callable[[str], bool]) -> None:
        self._inner, self._allowed = inner, allowed

    def __call__(self, method, url, headers=None, *, max_bytes=None):
        try:
            parts = urllib.parse.urlsplit(url)
            port = parts.port
        except ValueError as exc:  # an address of no URL form (a port that is no number, a broken IPv6 host): refused unsent
            raise TransportSecurityError("the answer leads to an address of no URL form") from exc
        if parts.scheme != "https" or port not in (None, 443) or not self._allowed(parts.hostname or ""):
            raise TransportSecurityError("the answer leads to another host")
        return self._inner(method, url, headers, max_bytes=max_bytes)


# --- the header of a GGUF file --------------------------------------------------------------------------------------------


class HeaderError(ValueError):
    """The first bytes of a file do not show its architecture whole."""


_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9
_MAX_PAIRS = 1 << 16
_MAX_DEPTH = 4  # arrays of arrays of arrays...; a real header has one or two levels


class _Reader:
    def __init__(self, data: bytes, position: int) -> None:
        self.data, self.position = data, position

    def take(self, size: int) -> bytes:
        end = self.position + size
        if size < 0 or end > len(self.data):
            raise HeaderError("the header is cut off before the architecture")
        chunk, self.position = self.data[self.position : end], end
        return chunk

    def number(self, fmt: str) -> int:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> bytes:
        return self.take(self.number("<Q"))

    def skip(self, kind: int, depth: int = 0) -> None:
        if kind in _SCALAR_SIZE:
            self.take(_SCALAR_SIZE[kind])
        elif kind == _STRING:
            self.string()
        elif kind == _ARRAY:
            if depth >= _MAX_DEPTH:
                raise HeaderError("arrays are nested deeper than a header has them")
            inner, count = self.number("<I"), self.number("<Q")
            if inner in _SCALAR_SIZE:
                self.take(_SCALAR_SIZE[inner] * count)
            else:
                for _ in range(count):
                    self.skip(inner, depth + 1)
        else:
            raise HeaderError(f"a value of type {kind} is not known")


def gguf_architecture(data: bytes) -> str:
    """`general.architecture` from the first bytes of a GGUF file, or `HeaderError`.

    Magic `GGUF`, version 2 or 3, then the key-value pairs in order; the value of the pair is read when it is a string, the
    others are skipped by their type. A header that is cut off before the pair, a type nobody here knows, a count that
    cannot be met and an identifier that is no GGUF architecture are faults -- never a guess."""
    reader = _Reader(data, 0)
    if reader.take(4) != b"GGUF":
        raise HeaderError("not a GGUF file")
    if reader.number("<I") not in (2, 3):
        raise HeaderError("a GGUF version nobody here has read")
    reader.number("<Q")  # the tensor count
    pairs = reader.number("<Q")
    if pairs == 0 or pairs > _MAX_PAIRS:
        raise HeaderError("the number of pairs cannot be met in the first bytes")
    for _ in range(pairs):
        key, kind = reader.string(), reader.number("<I")
        if key == b"general.architecture":
            if kind != _STRING:
                raise HeaderError("general.architecture is not a string")
            try:
                value = reader.string().decode("ascii")
            except UnicodeDecodeError as exc:
                raise HeaderError("general.architecture is not ASCII") from exc
            if _ARCH_RE.fullmatch(value) is None:
                raise HeaderError("general.architecture is no GGUF identifier")
            return value
        reader.skip(kind)
    raise HeaderError("general.architecture is not in the file")


# --- the packages of a repository ------------------------------------------------------------------------------------------


class WeightPackage(NamedTuple):
    identity: str  # `identity_stem(file)`, the shard suffix removed
    file: str  # the weight file, or the first shard of the set
    quantization: str | None


def weight_packages(files: Sequence[tuple[str, int]]) -> list[WeightPackage]:
    """The weight packages of a file list `(path, size)`: every GGUF file that is no projector or importance matrix and
    is not a later shard, in the order of the paths. A set whose first shard is missing is no package."""
    found = []
    for path, _size in sorted(files):
        if not path.lower().endswith(".gguf") or non_weight_role(path.rsplit("/", 1)[-1]) is not None:
            continue
        shard = _SHARD_RE.search(path)
        if shard is not None and int(shard.group(1)) != 1:
            continue
        found.append(WeightPackage(identity_stem(path), path, parse_hf_quant(path)))
    return found


def same_quantization(packages: Sequence[WeightPackage]) -> dict[str, int | None]:
    """For each package file the number of packages of its quantization in the repository (`None`: it has none)."""
    counts: dict[str, int] = {}
    for package in packages:
        if package.quantization is not None:
            counts[package.quantization] = counts.get(package.quantization, 0) + 1
    return {package.file: (counts[package.quantization] if package.quantization is not None else None) for package in packages}


# --- the addresses ---------------------------------------------------------------------------------------------------------


def list_url(row_repo: str) -> str:
    """The GGUF repositories of a row: its quantization relation, GGUF only, the most downloaded first, with `gguf`, `sha` and `tags`."""
    query = urllib.parse.urlencode([
        ("filter", f"base_model:quantized:{row_repo}"), ("filter", "gguf"), ("sort", "downloads"), ("direction", -1), ("limit", LIST_PAGE),
        ("expand", "gguf"), ("expand", "sha"), ("expand", "tags"),
    ])
    return f"{HUB}/api/models?{query}"


def tree_url(repo: str, revision: str) -> str:
    return f"{HUB}/api/models/{repo}/tree/{revision}?recursive=true"


def header_url(repo: str, revision: str, file: str) -> str:
    return f"{HUB}/{repo}/resolve/{revision}/{urllib.parse.quote(file, safe='/')}"


def card_url(repo: str, revision: str) -> str:
    return f"{HUB}/{repo}/resolve/{revision}/README.md"


def model_url(repo: str) -> str:
    return f"{HUB}/api/models/{repo}"


# --- a stored check against a newer source ------------------------------------------------------------------------------------


def recompare(check: PackageCheck, names: frozenset[str], commit: str, today: date) -> PackageCheck | None:
    """The check again against the source `names` of `commit`, as a new check -- or `None` when the result is the same, so
    that a source that moves every day moves no check (V6). Repository, revision, file and architecture stay."""
    result = compare_architecture(check.architecture, names)
    if result == check.result:
        return None
    return PackageCheck.model_validate({**check.model_dump(mode="json"), "result": result, "llamacpp_commit": commit,
                                        "source_url": llamacpp_page_url(commit), "checked": today.isoformat()})


# --- the stage ----------------------------------------------------------------------------------------------------------------


class _Fail(Exception):
    """One answer that cannot be used; its text is the reason in a note."""


class _ShareEnd(Exception):
    """The share of the row is spent: the request is not sent, and the row ends -- no fault of anything."""


@dataclass
class StageResult:
    checks: dict[str, list[dict]] = field(default_factory=dict)  # row -> the list the row should hold, for every row read
    setup_notes: dict[str, dict] = field(default_factory=dict)  # row -> the note to propose
    attempted: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)  # the repositories and failed packages tried (`unit`)
    notes: list[str] = field(default_factory=list)
    problem: bool = False
    budget_end: bool = False
    commit: str | None = None
    page_urls: dict[str, list[str]] = field(default_factory=dict)  # row -> pages of the repositories whose checks changed
    cut_rows: list[str] = field(default_factory=list)  # the rows that were cut at their share, in the order they were taken
    reserved: list[str] = field(default_factory=list)  # the rows that got a part of the reserve in round 2
    open_repos: int = 0  # the repositories those rows had listed and not read when the run ended


def unit(repo: str, identity: str | None = None) -> str:
    """The name of a repository, or of one package of it, in `package_unit_attempts`."""
    return repo if identity is None else f"{repo}:{identity}"


def unit_key(name: str) -> str:
    """The form in which two unit names are compared: the repository without case, the identity as written (a file name
    is never folded)."""
    repo, colon, identity = name.partition(":")
    return hf_id_key(repo) + colon + identity


def record_units(attempts: Mapping[str, date], taken: Sequence[str], today: date) -> dict[str, date]:
    """`record_attempts` for unit names: compared by `unit_key`, the youngest day wins and keeps its spelling."""
    merged: dict[str, tuple[str, date]] = {}
    for name, day in [*attempts.items(), *((name, today) for name in taken)]:
        key = unit_key(name)
        if key not in merged or day >= merged[key][1]:
            merged[key] = (name, day)
    return dict(merged.values())


class PackageStage:
    """One run of the stage over the rows it may take. `transport` books every request (the scanner's two budgets)."""

    def __init__(self, transport: Transport, rows: Mapping[str, CatalogModel], today: date, attempts: Mapping[str, date], *,
                 carried_checks: Mapping[str, Sequence[dict]], carried_notes: Mapping[str, dict | None], is_rejected: Callable[[str], bool],
                 limit: int, remaining: Callable[[], int], packagers: Sequence[str] = (),
                 unit_attempts: Mapping[str, date] | None = None, reserve_days: Mapping[str, date] | None = None) -> None:
        self.rows, self.today, self.attempts, self.limit = rows, today, attempts, limit
        self.reserve_days = {hf_id_key(name): day for name, day in (reserve_days or {}).items()}  # row -> the day it last got the reserve
        self.remaining = remaining  # what the budgets still allow the stage, asked when the run takes its rows
        self.packagers = {hf_id_key(account) for account in packagers}  # the listed packagers of the catalog
        self.allowance: int | None = None  # the requests the row in hand may still send, or no limit
        self.row_start = 0
        self.second_round = False
        self.progress = 0  # units read to their end, heads read or failed: what counts as progress of a row in round 2
        self.card_done: set[str] = set()  # rows whose card was read or failed; a second round does not ask it again
        self.candidates: dict[str, list] = {}  # row -> its repositories once its list was read to the end
        self.partial: dict[str, tuple[list, set, list]] = {}  # row -> list pages read, their addresses, the address still ahead
        self.carried_checks = {hf_id_key(row): value for row, value in carried_checks.items()}  # ids are compared without case
        self.carried_notes = {hf_id_key(row): value for row, value in carried_notes.items()}
        self.is_rejected = is_rejected
        self.unit_days = {unit_key(name): day for name, day in (unit_attempts or {}).items()}
        self.sent = 0  # requests this run actually sent (a budget that refuses one sends nothing)
        transport = self._counted(transport)
        self.github = RedirectingTransport(_Gate(transport, lambda host: host == LLAMACPP_HOST), max_hops=LLAMACPP_MAX_REDIRECTS + 1)
        self.api = RedirectingTransport(_Gate(transport, lambda host: host == "huggingface.co"), max_hops=1)
        self.files = RedirectingTransport(_Gate(transport, lambda host: host == "huggingface.co" or host.endswith(".hf.co")), max_hops=2)
        self.result = StageResult()
        self.names: frozenset[str] = frozenset()
        self.commit = ""

    # --- asking

    def _counted(self, transport: Transport) -> Transport:
        def counted(method, url, headers=None, *, max_bytes=None):
            if self.allowance is not None and self.sent - self.row_start >= self.allowance:
                raise _ShareEnd()  # the share of the row is spent: nothing is sent
            try:
                response = transport(method, url, headers, max_bytes=max_bytes)
            except BudgetExhaustedError:
                raise  # refused before it was sent
            except BaseException:
                self.sent += 1
                raise
            self.sent += 1
            return response
        return counted

    def _get(self, layer, url: str, limit: int, headers: dict | None = None, ok: tuple[int, ...] = (200,)):
        try:
            response = layer("GET", url, headers, max_bytes=limit)
        except ValueError as exc:  # a `Location` the redirect cannot even join: this request fails, never the run
            raise _Fail("the answer leads to an address of no URL form") from exc
        except ResponseTooLargeError as exc:
            raise _Fail(_clean(exc)) from exc
        except _FAILURES as exc:
            raise _Fail(_clean(exc)) from exc
        if response.status not in ok:
            raise _Fail(f"HTTP {response.status}")
        return response

    def _json(self, response):
        try:
            return response.json()
        except ValueError as exc:
            raise _Fail("the answer is no JSON") from exc

    # --- the run

    def run(self) -> StageResult:
        try:
            self._source()
        except BudgetExhaustedError:
            self.result.budget_end = True
            self.result.notes.append("budget ended: package checks not started, the llama.cpp source was not read")
            return self.result
        except (_Fail, ArchBlockError, ValueError) as exc:
            self.result.problem = True
            self.result.notes.append(f"package checks: the llama.cpp source could not be read whole ({_clean(exc)}); no check was added, changed or removed")
            return self.result
        eligible = [name for name, row in self.rows.items() if row.kind in ("text", "vision") and not self.is_rejected(name)]
        due = package_rows_for_run(eligible, self.attempts, self.today, self.limit)
        rest = self.remaining()
        # The reserve is never nil: it pays the progress of one open unit with what it needs first (V6). When the budget cannot give
        # it next to the minimum of every row, round 1 takes fewer rows, and the others rotate on as before; one row is always taken
        # when its minimum is there, with what is left of the reserve.
        reserve = max(RESERVE_MIN, rest // 10)
        count = min(len(due), max(0, (rest - reserve) // MIN_ROW_SHARE))
        if count == 0 and due and rest >= MIN_ROW_SHARE:
            count = 1  # a budget below RESERVE_MIN + MIN_ROW_SHARE pays no progress beside a row: the one row gets all of it
            reserve = 0
        taken = due[:count]
        reserve = min(reserve, rest - count * MIN_ROW_SHARE)
        share = (rest - reserve) // len(taken) if taken else 0
        done, current, sent = 0, None, 0
        try:
            for name in taken:  # round 1: every row gets the same share
                current, sent = name, self.sent
                if self._row(self.rows[name], share):
                    self.result.cut_rows.append(name)
                self.result.attempted.append(name)  # tried once it had its share, whatever stays open (module text)
                done += 1
            self.second_round = True
            # round 2: what is left goes to the cut rows, the one that got it longest ago first (`package_reserve_days`), so that
            # every cut row comes round in a finite number of runs and none keeps it for ever
            order = sorted(enumerate(self.result.cut_rows), key=lambda item: (self.reserve_days.get(hf_id_key(item[1]), date.min), item[0]))
            for _, name in order:
                if self.remaining() <= 0:
                    break
                before = self.progress
                self._row(self.rows[name], self.remaining())
                if self.progress > before:  # a leftover without progress books nothing: the row keeps its priority
                    self.result.reserved.append(name)
        except BudgetExhaustedError:
            if current is not None and done == 0 and self.sent > sent:  # it had the whole budget: tried, so it holds back no other row (module text)
                self.result.attempted.append(current)
            self.result.budget_end = True
            self.result.notes.append(f"budget ended: package checks stopped after {done} of {len(due)} rows, the others are taken next run")
        else:
            if len(taken) < len(due):  # the budget did not reach the minimum for every row
                self.result.budget_end = True
                self.result.notes.append(f"budget ended: package checks stopped after {done} of {len(due)} rows, the others are taken next run")
        self.result.open_repos = sum(self._never_tried(name) for name in self.result.cut_rows)
        self.result.notes.append(f"package checks: llama.cpp {self.commit[:12]} read ({len(self.names)} architectures); {done} of {len(due)} rows tried")
        if self.result.cut_rows:
            self.result.notes.append(f"package checks: {len(self.result.cut_rows)} rows ended at their share of {share} requests, "
                                     f"{self.result.open_repos} listed repositories are read next time")
        return self.result

    def _source(self) -> None:
        """The commit, then the block at that commit; whole or not at all (`parse_arch_names`)."""
        answer = self._get(self.github, llamacpp_commit_url(), LLAMACPP_COMMIT_MAX_BYTES, {"Accept": LLAMACPP_COMMIT_ACCEPT})
        commit = read_commit(answer.body.decode("ascii"))
        text = self._get(self.github, llamacpp_block_url(commit), LLAMACPP_FILE_MAX_BYTES, {"Accept": LLAMACPP_FILE_ACCEPT}).body.decode("utf-8")
        self.names, self.commit = parse_arch_names(text), commit
        self.result.commit = commit

    def _row(self, row: CatalogModel, allowance: int) -> bool:
        """The card and the packages of one row within `allowance` requests; `True` when the share ended before the row did."""
        self.allowance, self.row_start = allowance, self.sent
        try:
            if row.hf_repo not in self.card_done:
                try:
                    self._card(row)
                except _Fail as fail:  # not reached: reported, a stored note stays
                    self._unread(row, str(fail))
                self.card_done.add(row.hf_repo)
            try:
                self._packages(row)
            except _Fail as fail:
                self.result.problem = True
                self.result.notes.append(f"{row.hf_repo}: package checks: {fail}")
        except _ShareEnd:
            return True
        finally:
            self.allowance = None
        return False

    # --- the setup note (V7)

    def _unread(self, row: CatalogModel, reason: str) -> None:
        self.result.problem = True
        note = C.not_read(row.hf_repo, reason, self.today)
        self.result.notes.append(f"{row.hf_repo}: model card not read ({note.reason}); a stored setup note is unchanged")

    def _base_note(self, row: CatalogModel) -> SetupNote | None:
        carried = self.carried_notes.get(hf_id_key(row.hf_repo))
        try:
            return SetupNote.model_validate(carried) if carried else row.setup_note
        except ValueError:
            return row.setup_note

    def _card(self, row: CatalogModel) -> None:
        model = row.hf_repo
        try:
            revision = self._json(self._get(self.api, model_url(model), R.API_MAX_BYTES)).get("sha")
        except AttributeError as exc:
            raise _Fail("the model answer is no object") from exc
        if not isinstance(revision, str) or _SHA_RE.fullmatch(revision) is None:
            raise _Fail("the model answer names no revision")
        base = self._base_note(row)
        if base is not None and base.card_revision == revision:
            return  # the card of the stored revision is not read again
        rng = {"Range": f"bytes=0-{CARD_MAX_BYTES - 1}"}
        response = self._get(self.files, card_url(model, revision), CARD_MAX_BYTES, rng, ok=(200, 206))
        if response.header("X-Repo-Commit") not in (None, revision):
            raise _Fail("the card is not from the revision that was asked for")
        body, start = response.body, False
        if response.status == 206:
            start = self._card_range(response.header("Content-Range"), len(body))
        if start:
            body = body[: body.rfind(b"\n") + 1]  # the last line may be cut, inside a character too: it goes
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _Fail("the card is no UTF-8 text") from exc
        if start:
            text = C.whole_paragraphs(text)  # a statement still open at the cut may go on with what limits it: it goes
        found = C.detect(text, model)
        if start and found.status == "none_recognized":
            raise _Fail(f"the card is longer than {CARD_MAX_BYTES} bytes and its start holds no instructions")
        if found.status == "none_recognized" and base is None:
            return
        self.result.setup_notes[model] = C.note_of(model, found, revision, self.today).model_dump(mode="json")

    @staticmethod
    def _card_range(content_range: str | None, size: int) -> bool:
        """What a `206` of the card is, by its `Content-Range`: `False` the whole card (`bytes 0-<size-1>/<size>`), `True`
        the start of a longer one (the window full, `bytes 0-<window-1>/<more>`); anything else -- missing, another range,
        a total that does not fit -- proves neither and is a fault."""
        match = _CONTENT_RANGE_RE.fullmatch((content_range or "").strip())
        if match is None or int(match.group(1)) != 0 or int(match.group(2)) != size - 1:
            raise _Fail("the range of the card answer does not show what part of the card it is")
        total = int(match.group(3))
        if total == size:
            return False
        if size == CARD_MAX_BYTES and total > size:
            return True
        raise _Fail("the range of the card answer does not show what part of the card it is")

    # --- the package checks (V1, V5, V6)

    def _base_checks(self, row: CatalogModel) -> list[PackageCheck]:
        carried = self.result.checks.get(row.hf_repo)  # what this run read of the row in an earlier round
        if carried is None:
            carried = self.carried_checks.get(hf_id_key(row.hf_repo))
        try:
            return [PackageCheck.model_validate(item) for item in carried] if carried is not None else list(row.gguf_checks)
        except ValueError:
            return list(row.gguf_checks)

    def _packages(self, row: CatalogModel) -> None:
        base = self._base_checks(row)
        state = _RowState()
        try:
            self._repos(row, base, state)
        finally:
            merged = merge_package_checks([c for c in base if (hf_id_key(c.repo), c.identity) not in state.dropped], state.fresh)
            kept_old = [c for c in merged if c in base]
            added = [c for c in merged if c not in base]
            merged = kept_old + added[: max(0, CHECKS_PER_ROW - len(kept_old))]
            merged.sort(key=lambda c: (hf_id_key(c.repo), c.identity))
            self.result.checks[row.hf_repo] = [c.model_dump(mode="json") for c in merged]
            earlier = self.result.page_urls.get(row.hf_repo, [])  # the evidence of a first round stays with the checks it belongs to
            self.result.page_urls[row.hf_repo] = sorted({*earlier, *state.changed_repos.values()})

    def _listed(self, row: CatalogModel) -> list:
        """Every page of the list query of a row, to its end; a next page must stay on the Hub's list and may not repeat."""
        # the pages read so far and the next address are kept for this run: a share that ends inside the list is continued there
        progress = self.partial.setdefault(row.hf_repo, ([], set(), [list_url(row.hf_repo)]))
        found, visited, ahead = progress
        while ahead:
            url = ahead[0]
            visited.add(url)
            response = self._get(self.api, url, R.API_MAX_BYTES)
            data = self._json(response)
            if not isinstance(data, list):
                raise _Fail("the list answer is no list")
            link = response.header("Link")
            match = _LINK_NEXT_RE.search(link) if link else None
            try:
                following = urllib.parse.urljoin(url, match.group(1)) if match else None
            except ValueError as exc:
                raise _Fail("the next page of the list has an address of no URL form") from exc
            if following is not None and (not following.startswith(f"{HUB}/api/models?") or following in visited):
                raise _Fail("the next page of the list leads elsewhere or repeats")
            found += data
            ahead[:] = [following] if following is not None else []
        del self.partial[row.hf_repo]
        return found

    @staticmethod
    def _canonical(tags: Sequence[str], row: str) -> list[str]:
        """The tags with every relation to the row spelled as the row is (the Hub compares ids without case; `base_repos`
        and `check_relation` do not)."""
        out = []
        for tag in tags:
            match = _RELATION_RE.match(tag)
            out.append(match.group(1) + row if match and hf_id_key(match.group(2)) == hf_id_key(row) else tag)
        return out

    def _candidates(self, row: CatalogModel) -> list[tuple[str, str, object]]:
        """The repositories of a row that quantize it and nothing else, `(repository, revision, hub gguf field)`, in the list's order."""
        if row.hf_repo in self.candidates:  # the list was read to its end in an earlier round
            return self.candidates[row.hf_repo]
        candidates, seen = [], set()
        for raw in self._listed(row):
            entry = R.parse_entry(raw)
            sha = raw.get("sha") if isinstance(raw, dict) else None
            if entry is None or "gguf" not in entry.tags or not isinstance(sha, str) or _SHA_RE.fullmatch(sha) is None or self.is_rejected(entry.repo):
                continue
            tags = self._canonical(entry.tags, row.hf_repo)
            if R.base_repos(tags) != [row.hf_repo] or check_relation(tags, None, row.hf_repo)[0] != "quantized" or hf_id_key(entry.repo) in seen:
                continue
            seen.add(hf_id_key(entry.repo))
            candidates.append((entry.repo, sha, raw.get("gguf")))
        self.candidates[row.hf_repo] = candidates
        return candidates

    def _owner_class(self, row: CatalogModel, repo: str) -> int:
        """0 the publisher of the row itself, 1 a listed packager of the catalog, 2 any other account."""
        owner = hf_id_key(repo.split("/", 1)[0])
        if owner == hf_id_key(row.hf_repo.split("/", 1)[0]):
            return 0
        return 1 if owner in self.packagers else 2

    def _repos(self, row: CatalogModel, base: list[PackageCheck], state: "_RowState") -> None:
        candidates = self._candidates(row)
        checked = {hf_id_key(c.repo) for c in base}
        # module text: never tried first, then the one tried longest ago; among equals the publisher's, the listed packagers', the
        # rest; then the one without a check; downloads (the list's order) last
        candidates = sorted(candidates, key=lambda item: (self._day(unit(item[0])), self._owner_class(row, item[0]), hf_id_key(item[0]) in checked))
        for repo, sha, hub_gguf in candidates:
            if self.second_round and self._day(unit(repo)) == self.today:
                continue  # read to its end in the first round
            known = [c for c in base if hf_id_key(c.repo) == hf_id_key(repo)]
            try:
                self._repo(repo, sha, hub_gguf, known, state)
            except _Fail as fail:
                self.result.problem = True
                self.result.notes.append(f"{repo}: package checks: {fail}")
            self.result.units.append(unit(repo))  # read to its end, with or without a fault; a budget that ends skips this
            self.progress += 1
            self.unit_days[unit_key(unit(repo))] = self.today  # a second round reads the others first

    def _never_tried(self, name: str) -> int:
        """How many repositories of a cut row nobody read yet; a row whose list the share ended inside, or before, counts as one more
        than it can tell."""
        candidates = self.candidates.get(name)
        if candidates is None:
            return 1
        return sum(1 for repo, _sha, _gguf in candidates if unit_key(unit(repo)) not in self.unit_days)

    def _day(self, name: str) -> date:
        return self.unit_days.get(unit_key(name), date.min)

    def _repo(self, repo: str, sha: str, hub_gguf: object, known: list[PackageCheck], state: "_RowState") -> None:
        response = self._get(self.api, tree_url(repo, sha), R.API_MAX_BYTES)
        if _NEXT_RE.search(response.header("Link") or ""):
            raise _Fail("the file list goes on over a second page")
        listing = self._json(response)
        if not isinstance(listing, list):
            raise _Fail("the file list is no list")
        files = []
        for item in listing:  # whole or not at all: an entry nobody understands makes the list unread, never a shorter list
            kind = item.get("type") if isinstance(item, dict) else None
            if kind == "directory":
                continue
            path, size = (item.get("path"), item.get("size")) if kind == "file" else (None, None)
            if not isinstance(path, str) or not path or not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise _Fail("the file list holds an entry nobody understands")
            files.append((path, size))
        packages = weight_packages(files)
        identities = [package.identity for package in packages]
        counts = same_quantization(packages)
        hub = hub_gguf if isinstance(hub_gguf, dict) else {}
        hub_file = hub_metadata_file([(path, size) for path, size in files if path.lower().endswith(".gguf")], hub.get("totalFileSize"))
        hub_arch = hub.get("architecture")
        state.dropped.update((hf_id_key(c.repo), c.identity) for c in known if c.revision != sha and c.identity not in identities)
        reusable = {(c.identity, c.file): c for c in known if c.revision == sha}  # the same file at the same revision
        for package in sorted([p for p in packages if identities.count(p.identity) == 1], key=lambda p: self._day(unit(repo, p.identity))):
            old = reusable.get((package.identity, package.file))
            if old is not None:  # compared with this source again, no request; the others of the repository are still read
                again = recompare(old, self.names, self.commit, self.today)
                if again is not None:
                    state.fresh.append(again)
                    state.changed_repos[hf_id_key(repo)] = repo
                continue
            if package.file == hub_file and isinstance(hub_arch, str):
                architecture, source = hub_arch, "hub_field"
            else:
                try:
                    architecture, source = self._head(repo, sha, package.file), "file_header"
                except _Fail as fail:
                    self.result.problem = True
                    self.result.notes.append(f"{repo}: package checks: {package.file}: {fail}")
                    self.result.units.append(unit(repo, package.identity))
                    self.progress += 1
                    continue
            try:
                check = PackageCheck(
                    repo=repo, revision=sha, identity=package.identity, file=package.file, quantization=package.quantization,
                    architecture=architecture, architecture_source=source, result=compare_architecture(architecture, self.names),
                    llamacpp_commit=self.commit, source_url=llamacpp_page_url(self.commit), checked=self.today,
                    same_quantization=counts[package.file],
                )
            except ValueError:
                continue  # a name or an identifier the schema refuses: no check, the package stays unknown
            state.fresh.append(check)
            self.progress += 1
            state.changed_repos[hf_id_key(repo)] = repo

    def _head(self, repo: str, sha: str, file: str) -> str:
        rng = {"Range": f"bytes=0-{HEADER_BYTES - 1}"}
        response = self._get(self.files, header_url(repo, sha, file), HEADER_BYTES, rng, ok=(200, 206))
        if response.header("X-Repo-Commit") not in (None, sha):
            raise _Fail("the head is not from the revision that was asked for")
        try:
            return gguf_architecture(response.body)
        except HeaderError as exc:
            raise _Fail(f"the header is not readable ({exc})") from exc


@dataclass
class _RowState:
    fresh: list[PackageCheck] = field(default_factory=list)
    dropped: set[tuple[str, str]] = field(default_factory=set)
    changed_repos: dict[str, str] = field(default_factory=dict)  # hf_id_key -> the repository as written
