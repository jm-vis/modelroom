"""The catalog scanner: reads the Hub, writes a proposal, never changes the table (AP9-KA-T).

    python scripts/catalog_scan.py --catalog modelroom/catalog.json --rejected modelroom/catalog-rejected.json \\
        --out <folder> [--today YYYY-MM-DD] [--budget 300]

It reads the two files and the output folder, asks the Hub with `GET` only, and writes `<out>/<day>/proposal.json`,
`<out>/<day>/proposal.md` and `<out>/scan-state.json`, in that order, each atomically, the state last: a run that stops
in between leaves the old state and the next run reads the same pages again. It cannot write `catalog.json` or
`catalog-rejected.json`; taking a proposal into the table is `catalog_publish.py`, after the maintainer's word. Exit 0;
exit 1 when the budget ended, a page failed or a stored position was lost (the proposal is written all the same); exit 2
when the call or a file is wrong or the proposal cannot be made (nothing is written). The rules are in `catalog_scan_rules.py`.

Every run also proposes successors from the names (stage `A4`, `catalog_scan_latest.name_successors`, NF): the name rule of the
search over the rows of the table, one numbered change each, approved one by one by the maintainer. `--offline` makes only
these proposals, from the table alone, with no request and no mark moved.

With `--packages` (AP9-KO, `catalog_scan_packages.py`) the run also checks the GGUF packages of the catalog rows against the
llama.cpp source and reads the model cards for setup notes, for the rows it takes this day (`PACKAGE_ROWS_PER_RUN`, at most
`PACKAGE_STAGE_BUDGET` requests); that needs a working catalog of schema 3 and opens `api.github.com` and the Hub's CDN for
the scanner alone. Its proposals are field changes `gguf_checks` and `setup_note` (stage A9); a fault changes nothing.

The watermark holds per entry: one created before the account's threshold (`covered_until` minus two days) is seen for the
rows of the table only (their kind and card, and the GGUF repos of one, all from the list data); it is no new row, no
candidate and no card request. An entry with no creation time counts as inside.

A listed publisher with no row in the table gets a seed all the same (`_seed`): five places for open proposals over the
whole account; while one is free, its most downloaded repos are read page after page (a page too long is asked again with
half its `limit`) and the first that count are proposed whatever their age. They stay proposed until the maintainer decides.
The family taken from the first word of a name the grammar cannot read (`FLUX.1-dev`) and the rule that `onnx` next to
`safetensors` is no package hold in the daily walk too, not only in the seed.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the rules module sits beside this script
import catalog_scan_latest as L  # noqa: E402
import catalog_scan_notes as N  # noqa: E402
import catalog_scan_packages as K  # noqa: E402
import catalog_scan_rules as R  # noqa: E402

from modelroom.catalog import CatalogError, CatalogModel  # noqa: E402
from modelroom.catalog_remote import _clean  # noqa: E402
from modelroom.catalog_findings import PACKAGE_ROWS_PER_RUN, PACKAGE_STAGE_BUDGET, llamacpp_page_url, record_attempts  # noqa: E402
from modelroom.catalog_schema import build_catalog  # noqa: E402
from modelroom.contracts import SchemaVersionError, hf_id_key, validate_hf_repo  # noqa: E402
from modelroom.hf import _LINK_NEXT_RE  # noqa: E402
from modelroom.http import (  # noqa: E402
    BudgetedTransport, BudgetExhaustedError, RedirectingTransport, RequestBudget, ResponseTooLargeError,
    Transport, TransportSecurityError, UrllibTransport,
)
from modelroom.relation import check_relation  # noqa: E402
from modelroom.search import owner_class  # noqa: E402

STATE_SCHEMA = 1
DEFAULT_BUDGET, STAGE_BUDGET = 300, 30
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_PACKAGE_FIELDS = ("gguf_checks", "setup_note")  # the fields of the package stage (A9)
_FAILURES = (OSError, http.client.HTTPException, RuntimeError, ResponseTooLargeError, TransportSecurityError)


class UsageError(Exception):
    """The call or one of the two input files is wrong: exit 2, nothing written."""


class _Fail(Exception):
    """One answer that cannot be used; its text is the reason in the proposal, `status` the HTTP status if there was one."""

    status: int | None = None


class _TooLong(_Fail):
    """The answer is longer than the limit the caller set: a list page is asked again with a smaller `limit`."""


class _Lost(Exception):
    """A stored next page that does not answer: the position of a catch-up is lost."""


class DiskFiles:
    """The only way the scanner touches a disk; a test hands in a subclass to see every path."""

    def read(self, path) -> bytes:
        return Path(path).read_bytes()

    def write(self, path, data: bytes) -> None:
        """Atomic: a temporary file of its own in the target folder (created exclusively under a unique name, so it
        never meets another file), then `os.replace`; on any failure only that temporary file is removed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False)
        try:
            with handle:
                handle.write(data)
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def folders(self, path) -> list[str]:
        try:
            return sorted(entry.name for entry in Path(path).iterdir() if entry.is_dir())
        except OSError:
            return []

    def exists(self, path) -> bool:
        return Path(path).exists()


@dataclass
class ScanResult:
    exit_code: int = 0
    error: str | None = None
    proposal: dict | None = None
    main_used: int = 0
    sub_used: int = 0
    kind_check: dict = field(default_factory=dict)


class _Both:
    """The stages 3 and 4 book every try in the run's budget and in their own; a try neither could pay is not made."""

    def __init__(self, transport: Transport, main: RequestBudget, sub: RequestBudget) -> None:
        self._main, self._sub = main, sub
        self._inner = BudgetedTransport(BudgetedTransport(transport, main), sub)

    def remaining(self) -> int:
        return max(0, min(self._main.remaining, self._sub.remaining))

    def __call__(self, method, url, headers=None, *, max_bytes=None):
        if self._main.remaining <= 0 or self._sub.remaining <= 0:
            raise BudgetExhaustedError()
        return self._inner(method, url, headers, max_bytes=max_bytes)


class _Run:
    def __init__(self, catalog_path, rejected_path, out_dir, transport, now, budget, sub_budget, files, packages=False,
                 package_budget=PACKAGE_STAGE_BUDGET, package_rows=PACKAGE_ROWS_PER_RUN, offline=False) -> None:
        self.packages, self.package_budget, self.package_rows, self.package_attempts = packages, package_budget, package_rows, {}
        self.offline = offline
        self.unit_attempts: dict = {}
        self.reserve_days: dict = {}
        self.files, self.out, self.now = files, Path(out_dir), now.astimezone(timezone.utc)
        self.today = self.now.date().isoformat()
        outputs = {(self.out / self.today / name).resolve() for name in ("proposal.json", "proposal.md")} | {(self.out / "scan-state.json").resolve()}
        for given, what in ((catalog_path, "catalog"), (rejected_path, "rejection list")):
            if Path(given).resolve() in outputs:  # the run would overwrite what it reads
                raise UsageError(f"{given}: the {what} is also a file this run writes; choose another --out")
        for folder in (self.out, self.out / self.today):
            if folder.exists() and not folder.is_dir():  # the run could not make its folders; better to know before a request
                raise UsageError(f"{folder}: --out must name a folder, this is a file")
        for path in sorted(outputs):
            if path.exists() and not path.is_file():  # `os.replace` would fail on it, after the first file is written
                raise UsageError(f"{path}: this run writes it, but it is a folder")
        catalog_bytes = self._read(Path(catalog_path), "catalog")
        self.catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
        try:
            self.document = json.loads(catalog_bytes.decode("utf-8"))
            self.catalog = build_catalog(catalog_bytes.decode("utf-8"), Path(catalog_path))
        except (ValueError, CatalogError, SchemaVersionError) as exc:
            raise UsageError(f"{catalog_path}: not a valid catalog: {_clean(exc)}") from exc
        self.rows = {m.hf_repo: m for f in self.catalog.families for m in f.models}
        self.publishers = [p.account for p in self.catalog.publisher_info]
        self.packagers = list(self.catalog.packagers)
        self.accounts = self.publishers + [a for a in self.packagers if a not in self.publishers]
        try:
            rejection_list = self._read(Path(rejected_path), "rejection list")
            self.rejected_repos, self.rejected_accounts = R.parse_rejected(rejection_list)
            self.rejected_pairs = R.parse_rejected_pairs(rejection_list)  # (row, successor) pairs turned down, not the rows
        except ValueError as exc:
            raise UsageError(f"{rejected_path}: not a valid rejection list: {_clean(exc)}") from exc
        self.main, self.sub = RequestBudget(budget), RequestBudget(sub_budget)
        self.pkg = RequestBudget(package_budget)
        self.transport = transport
        both = _Both(transport, self.main, self.sub)
        self.t_list = RedirectingTransport(BudgetedTransport(transport, self.main), max_hops=1)
        self.t_base, self.t_config = RedirectingTransport(both, max_hops=1), RedirectingTransport(both, max_hops=2)
        self.notes: list[str] = []
        self.marks, self.new_marks, self.oldest = {}, {}, {}  # per account: (covered_until, pending), the same after the run, oldest time read
        self.incomplete: set[str] = set()
        self.unread: dict[str, list] = {"accounts": [], "seed": [], "B1": [], "B2": [], "collections": []}
        self.problem = self.budget_end = False
        self.seen: set[str] = set()
        self.known: set[str] = set()  # repos already decided on: carried over, or checked again in this run
        self.done_fields, self.done_accounts = set(), set()  # (repo, field) pairs and accounts a decision took, for the whole run
        self.fresh, self.changes, self.cands, self.gguf = {}, {}, {}, {}
        self.successors: list[tuple[CatalogModel, str]] = []
        self.kind_check = {repo: "not checked" for repo in self.rows}
        self.pending_publication = self.carry_from = self.decision = None

    # --- inputs -----------------------------------------------------------------------------------------------

    def _read(self, path: Path, what: str) -> bytes:
        try:
            return self.files.read(path)
        except OSError as exc:
            raise UsageError(f"{path}: cannot read the {what}: {_clean(exc)}") from exc

    def _json(self, path: Path):
        try:
            return json.loads(self.files.read(path).decode("utf-8"))
        except (OSError, ValueError):
            return None

    def _is_rejected(self, repo: str) -> bool:
        return repo in self.rejected_repos or repo.split("/", 1)[0] in self.rejected_accounts

    def _history(self) -> None:
        """The marks, the newest earlier proposal (to carry over) and any decision that is not yet published."""
        if self.files.exists(self.out / self.today / "decision.json"):
            raise UsageError(f"{self.out / self.today}: a decision exists for this day, the proposal would no longer match it")
        # an offline run replaces today's proposal like a normal one does, so it takes today's over as an earlier day's
        days = sorted(n for n in self.files.folders(self.out) if _DAY.fullmatch(n) and (n <= self.today if self.offline else n < self.today))
        for name in reversed(days):
            decision = self._json(self.out / name / "decision.json")
            if isinstance(decision, dict) and isinstance(decision.get("state"), str) and decision["state"] != "published":
                self.pending_publication = {"date": name, "state": decision["state"]}
                break
        for name in reversed(days):
            proposal = self._json(self.out / name / "proposal.json")
            if isinstance(proposal, dict):
                self.carry_from, self.decision = proposal, self._json(self.out / name / "decision.json")
                break
        published = self.catalog.published
        start = datetime.combine(published, time.min, tzinfo=timezone.utc) if published else self.now.replace(hour=0, minute=0, second=0, microsecond=0)
        raw_state = self._json(self.out / "scan-state.json")
        self.package_attempts = _attempts_of(raw_state)
        self.unit_attempts = _attempts_of(raw_state, "package_unit_attempts")
        self.reserve_days = _attempts_of(raw_state, "package_reserve_days")
        stored = raw_state.get("accounts") if isinstance(raw_state, dict) else None
        for account in self.accounts:
            mark = stored.get(account) if isinstance(stored, dict) else None
            covered = R.parse_time(mark.get("covered_until")) if isinstance(mark, dict) else None
            pending = mark.get("pending") if isinstance(mark, dict) and isinstance(mark.get("pending"), dict) else None
            self.marks[account] = (covered or start - R.FIRST_RUN_BACK, pending if covered else None)

    def _carry(self) -> None:
        """Every change and candidate of the newest earlier proposal that is still open comes back with its first date."""
        if self.carry_from is not None:
            self.changes, self.cands, self.known, self.done_fields, self.done_accounts = R.carry_over(self.carry_from, self.decision, self.rows, set(self.accounts), self._is_rejected, self.today)

    # --- asking -----------------------------------------------------------------------------------------------

    def _get(self, layer, url: str, limit: int):
        try:
            response = layer("GET", url, None, max_bytes=limit)
        except ResponseTooLargeError as exc:
            raise _TooLong(_clean(exc)) from exc
        except _FAILURES as exc:
            raise _Fail(_clean(exc)) from exc
        if response.status != 200:
            fail = _Fail(f"HTTP {response.status}")
            fail.status = response.status
            raise fail
        return response

    @staticmethod
    def _body(response):
        try:
            return response.json()
        except ValueError as exc:  # includes a body that is no UTF-8
            raise _Fail("the answer is no JSON") from exc

    def _page(self, url: str, what: str | None):
        """One list answer; one longer than `API_MAX_BYTES` is asked again with half its `limit` (down to `SMALLEST_PAGE`),
        every try a request of the budget; the page that was read leaves a note."""
        tried = url
        while True:
            try:
                response = self._get(self.t_list, tried, R.API_MAX_BYTES)
                break
            except _TooLong:
                limit = R.limit_of(tried)
                if limit is None or limit // 2 < R.SMALLEST_PAGE:
                    raise
                tried = R.with_limit(url, limit // 2)
                failed = limit
        if tried != url:
            self.notes.append(f"{what or url}: page read with limit {R.limit_of(tried)}, the answer with {failed} was longer than {R.API_MAX_BYTES} bytes")
        return response

    def _list(self, url: str, account: str | None = None, what: str | None = None):
        """One list answer and the next page it announces (`None` at the end); a next page elsewhere is a fault."""
        response = self._page(url, what or account)
        data = self._body(response)
        if not isinstance(data, list):
            raise _Fail("the answer is no list")
        link = response.header("Link")
        match = _LINK_NEXT_RE.search(link) if link else None
        following = urllib.parse.urljoin(url, match.group(1)) if match else None
        if following is not None and account is not None and not R.next_url_ok(following, account):
            raise _Fail("the next page leads elsewhere")
        return data, following

    def _determine(self, entry: R.Entry) -> R.KindResult:
        """The kind in stages: the cheap ones, then one request for the base model, then one for `config.json`."""
        cheap = R.cheap_kind(entry)
        if cheap is not None:
            return cheap
        budget_hit, last, base = False, {"stage": "tags", "value": "no table task in the tags"}, R.single_base(entry)
        if base and not self._is_rejected(base):  # a rejected repo gives no evidence, not even of a kind
            try:
                kind = R.kind_from_base_answer(self._body(self._get(self.t_base, R.base_url(base), R.API_MAX_BYTES)))
                last = {"stage": "base_model", "value": "no task of the table"}
            except BudgetExhaustedError:
                budget_hit, kind = True, None
            except _Fail as fail:
                last, kind = {"stage": "base_model", "value": str(fail)}, None
            if kind:
                return R.KindResult(kind, "base_model", False, {"stage": "base_model", "value": base})
        found = None
        try:
            found = R.kind_from_config(self._body(self._get(self.t_config, R.config_url(entry.repo), R.CONFIG_MAX_BYTES)))
            last = {"stage": "config", "value": "no decisive evidence"}
        except BudgetExhaustedError:
            budget_hit = True
        except _Fail as fail:
            last = {"stage": "config", "value": str(fail)}
        if found:
            return R.KindResult(found[0], "config", False, {"stage": "config", "value": found[1]})
        if budget_hit:
            self.problem = True
            return R.KindResult(None, None, False, {"stage": "budget", "value": "request budget ended"})
        return R.KindResult(None, None, False, last)

    def _classify(self, entry: R.Entry, result: R.KindResult, open_since: str) -> None:
        self.known.add(entry.repo)
        if result.excluded:
            return
        if result.kind is None:
            self.cands[("B3", entry.repo)] = R.b3_candidate(entry, result, open_since)
        else:
            self.fresh[entry.repo] = (entry, result, open_since)

    def _recheck(self) -> None:
        """Candidates that waited for budget are checked again before any new repo: oldest first, then by name."""
        waiting = [c for (kind, repo), c in self.cands.items()
                   if kind == "B3" and c["kind_evidence"].get("stage") == "budget" and not self._is_rejected(repo)]
        for cand in sorted(waiting, key=lambda c: (c["open_since"], c["hf_repo"])):
            entry = R.entry_of_candidate(cand)
            del self.cands[("B3", cand["hf_repo"])]
            self._classify(entry, self._determine(entry), cand["open_since"])

    # --- reading the lists ------------------------------------------------------------------------------------

    def _see(self, entry: R.Entry, before_mark: bool = False) -> None:
        """`before_mark`: the entry was created before the account's threshold; it still checks a row of the table
        and a GGUF repo of one (both from the list data, no request), and starts nothing else."""
        if entry.repo in self.seen:
            return
        self.seen.add(entry.repo)
        if self._is_rejected(entry.repo):
            return  # a rejected repo or account is read for nothing: no check of a row, no card, no evidence
        row = self.rows.get(entry.repo)
        if row is not None:
            self._existing(entry, row)
            return
        if before_mark:
            bases = R.base_repos(entry.tags)
            if len(bases) == 1 and bases[0] in self.rows:
                self._gguf(entry)
            return
        self._mentions(entry)
        self._gguf(entry)
        if (
            entry.repo in self.known or R.packaging_of_another(entry)
            or owner_class(entry.owner, catalog=self.catalog, listed_packagers=self.packagers) != "publisher"
            or not R.own_bases(entry)
        ):
            return
        self._classify(entry, self._determine(entry), self.today)

    def _existing(self, entry: R.Entry, row: CatalogModel) -> None:
        """A row of the table in a list of this run: its kind is checked against the Hub (stages 1 and 2 only), its card is read."""
        found = R.cheap_kind(entry)
        verdict = "not checked" if found is None else "differs" if found.excluded or found.kind != row.kind else "confirmed"
        self.kind_check[row.hf_repo] = verdict
        if verdict == "differs":
            value = found.evidence["value"]
            self.notes.append(f"{row.hf_repo}: kind differs from the table (table {row.kind}, hub {value if isinstance(value, str) else ' '.join(value)}); the row is not changed")
        target = entry.card.get("new_version")
        if isinstance(target, str):
            self.successors.append((row, target))
        else:
            self._mentions(entry)

    def _mentions(self, entry: R.Entry) -> None:
        """A card of an account that names one of its own rows in its text, without `new_version`: a note, never a successor."""
        if "new_version" in entry.card:
            return
        texts = R.card_texts(entry.card)
        named = (r for r in self.rows if r != entry.repo and r.split("/", 1)[0] == entry.owner and not self._is_rejected(r))
        for repo in sorted(r for r in named if any(r in t for t in texts)):
            note = f"card mentions {repo}: {entry.repo} names it in its card, no successor proposed"
            if note not in self.notes:
                self.notes.append(note)

    def _gguf(self, entry: R.Entry) -> None:
        """A GGUF repo of a listed account that quantizes exactly one base model (the relation check of the package)."""
        bases = R.base_repos(entry.tags)
        if "gguf" in entry.tags and len(bases) == 1 and check_relation(list(entry.tags), entry.card or None, bases[0])[0] == "quantized":
            if entry.owner in self.packagers or entry.owner == bases[0].split("/", 1)[0]:
                self.gguf.setdefault(bases[0], set()).add(entry.repo)

    def _walk(self, account: str, url: str, until, pages: int, threshold, resume: bool = False):
        """Pages from `url` until an entry satisfies `until` or the list ends; an entry created before `threshold` is
        seen for the rows of the table only: `(done, next url, pages used, newest time, entries seen)`."""
        newest, used, entries = None, 0, 0
        while True:
            if used >= pages:
                return False, url, used, newest, entries
            try:
                data, following = self._list(url, account)
            except _Fail as fail:
                if resume and used == 0 and fail.status is not None and 400 <= fail.status < 500 and fail.status != 429:
                    raise _Lost() from None  # only a refused cursor is a lost one; any other failure keeps the position
                raise
            used += 1
            hit = False
            for raw in data:
                entry = R.parse_entry(raw)
                if entry is None:
                    continue
                entries += 1
                if entry.created is not None:
                    newest = entry.created if newest is None else max(newest, entry.created)
                    self.oldest[account] = min(self.oldest.get(account, entry.created), entry.created)
                    hit = hit or until(entry.created)
                self._see(entry, entry.created is not None and entry.created < threshold)
            if hit or following is None:
                return True, None, used, newest, entries
            url = following

    def _flag(self, account: str, note: str) -> None:
        self.notes.append(note)
        self.problem = True
        self.incomplete.add(account)

    def _account(self, account: str) -> None:
        """One account, one run: the catch-up rules of the brief ("Ablauf je Konto und Lauf"), ten pages at most."""
        covered, pending = self.marks[account]
        threshold, limit, top = covered - R.OVERLAP, R.MAX_PAGES, R.list_url(account)
        try:
            if pending is not None and not R.next_url_ok(pending.get("next_url"), account):
                self._flag(account, f"{account}: resume position lost, catching up again")
                pending = None
            if pending is None:
                done, following, _, newest, entries = self._walk(account, top, lambda t: t < threshold, limit, threshold)
                state = (max(covered, newest) if newest else covered, None) if done else (covered, {"next_url": following, "newest_seen": R.stamp(newest or covered)})
            else:
                seen = R.parse_time(pending.get("newest_seen")) or covered
                done, following, used, newest, entries = self._walk(account, top, lambda t: t <= seen, limit, threshold)
                latest = max(seen, newest) if newest else seen
                if not done:
                    state = (covered, {"next_url": following, "newest_seen": R.stamp(latest)})
                elif limit - used <= 0:
                    state = (covered, {"next_url": pending["next_url"], "newest_seen": R.stamp(latest)})
                else:
                    try:
                        done, following, _, older, more = self._walk(account, pending["next_url"], lambda t: t < threshold, limit - used, threshold, resume=True)
                        entries += more
                        state = (max(covered, latest, older or covered), None) if done else (covered, {"next_url": following, "newest_seen": R.stamp(latest)})
                    except _Lost:
                        self._flag(account, f"{account}: resume position lost, catching up again")
                        state = (covered, None)
        except BudgetExhaustedError:
            self.budget_end = True
            self.unread["accounts"].append(account)
            return
        except _Fail as fail:
            self._flag(account, f"{account}: page error ({fail}), not advanced")
            return
        if account not in self.oldest and (entries or state[1] is not None):  # entries seen, none with a time: however the list ended
            self._flag(account, f"{account}: no usable time on the pages read, not advanced")
            return
        self.new_marks[account] = state
        if state[1] is not None:
            self.incomplete.add(account)
            self.notes.append(f"{account}: read back to {self.oldest[account].date().isoformat()} only, continues next run")

    # --- the open pages and the evidence -----------------------------------------------------------------------

    def _listed(self, entry: R.Entry) -> bool:
        """Listed already, or rejected (the account or this very repo): never a candidate."""
        return entry.owner in self.publishers or entry.owner in self.packagers or self._is_rejected(entry.repo)

    def _open_pages(self) -> None:
        """B1 (the fifteen task pages) and B2 (one page per latest text or vision row): new accounts that stand out."""
        b1: dict[str, list[R.Entry]] = {}
        for task in R.PIPELINE_TAG_TO_KIND:
            for raw in self._step("B1", R.b1_url(task), f"B1 {task}"):
                entry = R.parse_entry(raw)
                if entry and not self._listed(entry) and R.b1_ok(entry, self.now):
                    b1.setdefault(entry.owner, []).append(entry)
        for account, entries in b1.items():
            self._candidate("B1", account, sorted(entries, key=lambda e: (-e.likes, e.repo)), {})
        b2: dict[str, dict[str, list[R.Entry]]] = {}
        for row in (r for r in self.rows.values() if r.latest and r.kind in ("text", "vision") and not self._is_rejected(r.hf_repo)):
            for raw in self._step("B2", R.b2_url(row.hf_repo), f"B2 {row.hf_repo}"):
                entry = R.parse_entry(raw)
                if entry and "gguf" in entry.tags and not self._listed(entry):
                    b2.setdefault(entry.owner, {}).setdefault(row.hf_repo, []).append(entry)
        for account, per_row in b2.items():
            if len(per_row) >= R.B2_MIN_ROWS:
                everything = {e.repo: e for es in per_row.values() for e in es}  # a GGUF repo that answers for three rows counts once
                self._candidate("B2", account, sorted(everything.values(), key=lambda e: (-e.likes, e.repo)), {"rows": len(per_row), "bases": sorted(per_row)})

    def _seed(self) -> None:
        """A listed publisher with no row in the table, not rejected: the seed adds proposals until the account has
        `SEED_ROWS` open ones in all (fresh rows, carried new rows and candidates of open kind); the daily walk (`_see`)
        may have made more before. With no place free no page is asked; else its most downloaded repos are read, page
        after page (a page too long is asked again with a smaller `limit`, the next-page link must stay on the account,
        a repeated link, a page without an entry and a fifth page end it with a note), until `SEED_PAGE` entries are
        looked at (an unreadable one counts), the places are full or the list ends, every request in the budget. A repo that counts is judged like in `_see`, without the watermark
        and without `self.seen`; one proposed, decided on, judged or rejected already uses no place."""
        for account in self.publishers:
            if any(m.publisher == account for m in self.rows.values()) or account in self.rejected_accounts:
                continue
            before = len(self._open_of(account))
            slots = R.SEED_ROWS - before
            if slots <= 0:  # the five places are taken by proposals nobody decided: no request
                self.notes.append(f"{account}: seeded earlier, {before} proposals open, nothing read")
                continue
            what, url, read, added = f"seed {account}", R.seed_url(account), 0, 0
            visited, pages = {url}, 0
            while url is not None and added < slots and read < R.SEED_PAGE:
                if pages >= R.SEED_PAGES:
                    self.notes.append(f"{account}: seed stopped after {pages} pages (page limit)")
                    break
                try:
                    data, url = self._list(url, account, what)
                except BudgetExhaustedError:
                    self.budget_end = True
                    self.unread["seed"].append(what)
                    break
                except _Fail as fail:
                    self.notes.append(f"{what}: page error ({fail})")
                    self.problem = True
                    break
                pages += 1
                parsed = 0
                for raw in data:
                    if read >= R.SEED_PAGE:  # at most twenty entries are looked at per account, over every page
                        break
                    read += 1  # an entry that cannot be read was read all the same
                    entry = R.parse_entry(raw)
                    if entry is None:
                        continue
                    parsed += 1
                    if added >= slots:
                        continue
                    if (
                        self._is_rejected(entry.repo) or entry.owner.lower() != account.lower() or entry.repo in self.rows
                        or entry.repo in self.known or entry.repo in self.fresh  # proposed, decided on or judged already: no place
                        or R.packaging_of_another(entry) or not R.own_bases(entry)
                    ):
                        continue
                    self._mentions(entry)
                    self._gguf(entry)
                    result = self._determine(entry)
                    self._classify(entry, result, self.today)
                    added += not result.excluded
                if url is not None and added < slots and read < R.SEED_PAGE:  # it would go on: guard the way
                    reason = "page without entries" if parsed == 0 else "repeated link" if url in visited else None
                    if reason:
                        self.notes.append(f"{account}: seed stopped after {pages} pages ({reason})")
                        break
                    visited.add(url)
            if read:
                self.notes.append(f"{account}: seeded, no row in the table: {before} open before, {added} added from {read} repos read")

    def _open_of(self, account: str) -> set[str]:
        """The proposals of one account nobody decided, whoever made them (the walk or the seed): fresh rows, carried new
        rows and B3 candidates (any case). The seed fills the account up to `SEED_ROWS` of them, no more."""
        repos = set(self.fresh) | {key[1] for key in self.changes if key[0] == "new_row"} | {key[1] for key in self.cands if key[0] == "B3"}
        return {repo for repo in repos if repo.split("/", 1)[0].lower() == account.lower() and not self._is_rejected(repo)}

    def _step(self, name: str, url: str, what: str) -> list:
        """One request of an open step: the list, or nothing (a note for a fault, a count for the end of the budget)."""
        try:
            return self._list(url, what=what)[0]
        except BudgetExhaustedError:
            self.budget_end = True
            self.unread[name].append(what)
        except _Fail as fail:
            self.notes.append(f"{what}: page error ({fail})")
            self.problem = True
        return []

    def _candidate(self, kind: str, account: str, entries: list[R.Entry], extra: dict) -> None:
        if account in self.done_accounts:
            return
        before = self.cands.get((kind, account))
        self.cands[(kind, account)] = R.open_candidate(kind, account, entries, extra, before["open_since"] if before else self.today)

    def _collections(self, owners: list[str]) -> dict[str, dict[str, str]]:
        """`GET /api/collections?owner=<account>`, once per account: which collection of the publisher holds which repo."""
        return {o: R.collection_members(self._step("collections", R.collections_url(o), f"collections {o}"), o) for o in owners}

    # --- the proposal -------------------------------------------------------------------------------------------

    def _put(self, key: tuple, change: dict) -> None:
        """A change of this run, unless it touches a field a decision took; one already carried keeps its first date."""
        if R.touched(change) & self.done_fields:
            return
        before = self.changes.get(key)
        if before is not None:
            change["open_since"] = before["open_since"]
            if key[2:] == ("gguf_repos",):
                joined = list(before["fields"]["gguf_repos"]["new"])
                new = (joined + [r for r in change["fields"]["gguf_repos"]["new"] if r not in joined])[:R.MAX_GGUF_REPOS]
                change["fields"]["gguf_repos"]["new"] = new
                kept = {f"{R.HUB}/{r}" for r in new if r not in change["fields"]["gguf_repos"]["old"]}  # evidence names what is added, after the cap
                change["evidence"] = [e for e in before.get("evidence", []) + [e for e in change["evidence"] if e not in before.get("evidence", [])] if e in kept]
        self.changes[key] = change

    def _carried_rows(self) -> list[CatalogModel]:
        """The new rows carried from the last proposal that the table would take, as rows; one it would not take (alone,
        schema 2 with its rules) counts for nothing here and is left to `check_changes` and its note."""
        rows = []
        for key, change in sorted(self.changes.items()):
            if key[0] == "new_row":
                try:
                    build_catalog(json.dumps(R.apply_changes(self.document, [change])), Path("catalog.json"))
                    rows.append(CatalogModel.model_validate({name: pair["new"] for name, pair in change["fields"].items()}))
                except (CatalogError, SchemaVersionError, ValueError, TypeError):
                    continue
        return rows

    def _takes(self, repo: str, change: dict) -> bool:
        """Whether the table would take `change` alone (schema 2 with its rules); a row it would not take is left out
        with a note, so it counts for nothing: not for family, not for latest, not as a successor target."""
        try:
            build_catalog(json.dumps(R.apply_changes(self.document, [change])), Path("catalog.json"))
        except (CatalogError, SchemaVersionError, ValueError) as exc:  # a strange Hub answer costs this row, never the run
            self.notes.append(f"{repo}: left out, it would not validate ({' '.join(str(exc).split())[:120]})")
            self.fresh.pop(repo, None)
            return False
        return True

    def _build(self) -> None:
        carried_fields = N.field_values(self.changes)  # all there is before this run adds its own
        self.notes += N.drift_notes(self.rows.values(), N.last_drift(self.carry_from))  # every shipped row, a rejected one too; only what changed
        carried = self._carried_rows()
        self.carried = {m.hf_repo for m in carried}  # the carried rows that count: family, latest, successor target
        rows = list(self.rows.values()) + carried
        taken = self._taken_rows(rows)
        plain = [m for m in self.rows.values() if not self._is_rejected(m.hf_repo) and m.latest and not (m.latest_source or "").startswith(f"{R.HUB}/collections/{m.publisher}/")]
        collections = self._new_rows(taken, rows, plain)
        self._gguf_changes()
        self._successors()
        self.notes += L.group_tops([(e.repo, r.kind, e.parameters_b) for e, r, *_ in taken], rows, N.new_field_repos(carried_fields, self.changes))[1]
        for row in plain:
            link = collections.get(row.publisher, {}).get(row.hf_repo)
            if link and ("field", row.hf_repo, "successor") not in self.changes:
                self._put(("field", row.hf_repo, "latest_source"), R.field_change(
                    row, "A3", {"latest_source": (row.latest_source, link)}, [link], self.today, "latest_source"))

    def _taken_rows(self, rows: list[CatalogModel]) -> list:
        """The fresh rows the table would take, each with its family and GGUF list: only these count for latest."""
        taken = []
        for entry, result, open_since in sorted(self.fresh.values(), key=lambda item: item[0].repo):
            family = R.family_for(entry.repo, result.kind, rows, entry.parameters_b)
            if family is None:
                self.notes.append(f"{entry.repo}: no family word in the name, not proposed")
                self.fresh.pop(entry.repo)
                continue
            self.notes += N.unread_notes(entry.repo, entry.parameters_b, family)
            trial = R.build_row(entry, result, family=family, today=self.today, latest=False, latest_source=None)
            trial["gguf_repos"] = sorted(self.gguf.get(entry.repo, ()))[:R.MAX_GGUF_REPOS]
            if self._takes(entry.repo, R.new_row_change(trial, result, open_since)):
                rows.append(CatalogModel.model_validate(trial))  # the next new row sees this one: one family, one publisher
                taken.append((entry, result, open_since, family, trial["gguf_repos"]))
        return taken

    def _new_rows(self, taken: list, rows: list[CatalogModel], plain: list[CatalogModel]) -> dict:  # a `family top` row cites the publisher's page
        fresh = [(e.repo, r.kind, e.parameters_b) for e, r, _, _, _ in taken]
        latest = R.latest_among(fresh, rows)  # repo -> "over latest" | "family top"
        owners = [a for a in self.publishers if any(e.owner == a and latest.get(e.repo) == "over latest" for e, _, _, _, _ in taken) or any(m.publisher == a for m in plain)]
        collections = self._collections(owners)
        for entry, result, open_since, family, gguf in taken:
            source = f"{R.HUB}/{entry.owner}" if latest.get(entry.repo) == "family top" else collections.get(entry.owner, {}).get(entry.repo) or f"{R.HUB}/{entry.owner}"
            row = R.build_row(entry, result, family=family, today=self.today, latest=entry.repo in latest, latest_source=source)
            row["gguf_repos"] = gguf
            change = R.new_row_change(row, result, open_since)
            if self._takes(entry.repo, change):  # once more as it stands: a latest row carries more fields
                self._put(("new_row", entry.repo), change)
            else: self.left_out[entry.repo] = result.kind  # refused at the last check: a card naming it names a known row that is no `latest` row
        return collections

    def _gguf_changes(self) -> None:
        for base, found in sorted(self.gguf.items()):
            row = None if self._is_rejected(base) else self.rows.get(base)
            added = sorted(r for r in found if r not in row.gguf_repos) if row is not None else []
            target = (list(row.gguf_repos) + added)[:R.MAX_GGUF_REPOS] if row is not None else []
            if row is not None and target != list(row.gguf_repos):  # a full list takes nothing more: no change
                self._put(("field", base, "gguf_repos"), R.field_change(
                    row, "A2", {"gguf_repos": (list(row.gguf_repos), target)}, [f"{R.HUB}/{r}" for r in added if r in target], self.today, "gguf_repos"))

    def _successors(self) -> None:
        """The successors the cards name, against the table at the end of the run, resolved among themselves (AP9-KA-H)."""
        named = self._take_name_links()
        state = self._recheck_links()
        pointed = {target: repo for repo, (_, target) in sorted(state.items()) if target}
        candidates = {row.hf_repo: target for row, target in sorted(self.successors, key=lambda pair: pair[0].hf_repo)
                      if self._successor_ok(row, target, state)}
        chosen, notes = L.resolve_successors(candidates, pointed)
        self.notes += notes
        for repo, target in sorted(chosen.items()):
            row, card = self.rows[repo], f"{R.HUB}/{repo}"
            self._put(("field", repo, "successor"), R.field_change(row, "A3", L.successor_pairs(row, target, card, self.today), [card], self.today, "successor"))
        self._name_successors(named)

    def _take_name_links(self) -> dict[str, str]:
        """The carried successor changes of the name rule leave `self.changes`, `repo -> open_since`: the rule makes them
        again from the table in this run (`_name_successors`), so one it no longer makes is gone, and the check of
        `_recheck_links` -- a target that is a `latest` row -- is the card's rule and never theirs. A change of the name
        rule is told by its proof: the page of the successor, where a card's is the page of the row."""
        taken = {key[1]: self.changes[key]["open_since"] for key in sorted(self.changes) if key[0] == "field" and key[2] == "successor"
                 and self.changes[key]["fields"]["successor_source"]["new"] == f"{R.HUB}/{self.changes[key]['fields']['successor']['new']}"}
        for repo in taken:
            del self.changes[("field", repo, "successor")]
        return taken

    def _name_successors(self, open_since: dict[str, str]) -> None:
        """The successors the name rule proposes (`L.name_successors`), against the end of the run: after the cards', so a
        row a card gave a successor is not asked again. Each is a change of its own with the page of the successor as proof."""
        found, notes = L.name_successors(list(self.rows.values()), L.end_state(self.rows, self.changes), self._is_rejected, self.rejected_pairs)
        self.notes += notes
        for repo, named in sorted(found.items()):
            row, page = self.rows[repo], f"{R.HUB}/{named.target}"
            change = R.field_change(row, "A4", L.successor_pairs(row, named.target, page, self.today), [page], open_since.get(repo, self.today), "successor")
            self._put(("field", repo, "successor"), {**change, "basis": named.basis})

    def _recheck_links(self) -> dict:
        """The end state, after every carried successor change is checked again: one whose target is no `latest` row
        without a successor at the end of the run is dropped with a note (an older scan did not ask, and a carried change
        was never asked again). A link is decided when the link of its target is decided or the target has none, so a
        chain is cleaned from its end and the names play no part; each round decides at least one link. When no chain
        can be decided, the links of the cycles are dropped and the links that lead into them are decided normally.
        A shipped link to such a row is only noted, in one line."""
        links = {key[1]: change["fields"]["successor"]["new"] for key, change in self.changes.items() if key[0] == "field" and key[2] == "successor"}
        undecided = set(links)
        while undecided:
            ready = {repo for repo in undecided if links[repo] not in undecided}
            if ready:
                state = L.end_state(self.rows, self.changes)
                drop = {repo for repo in ready if state.get(links[repo]) != (True, None)}
            else:  # only cycles (and links into them) are left: the cycle links go
                for start in undecided:
                    walk, node = [], start
                    while node in undecided and node not in walk:
                        walk.append(node)
                        node = links[node]
                    if node in walk:
                        ready.update(walk[walk.index(node):])
                drop = ready
            for repo in sorted(drop):
                del self.changes[("field", repo, "successor")]
                self.notes.append(f"{repo}: carried successor {links[repo]} dropped, it is no latest row without a successor at the end of this run")
            undecided -= ready
        state = L.end_state(self.rows, self.changes)
        old = [f"{repo} -> {row.successor}" for repo, row in sorted(self.rows.items())
               if row.successor and not self._is_rejected(repo) and state.get(row.successor) != (True, None)]
        if old:
            self.notes.append(f"successors that are no latest row (unchanged): {', '.join(old)}")
        return state

    def _successor_ok(self, row: CatalogModel, target: str, state: dict) -> bool:
        """`new_version` on the card names a same-account, same-kind `latest` row without a successor at the end of this run."""
        if state.get(row.hf_repo, (False, None))[1] == target: return False  # the row's carried change names it already
        found, proposed = self.rows.get(target), self.changes.get(("new_row", target)) if target in self.carried or target in self.fresh else None
        kind = found.kind if found is not None else proposed["kind"] if proposed is not None else self.left_out.get(target)
        if self._is_rejected(target):
            kind = None
        try:
            validate_hf_repo(target)
        except ValueError:
            kind = None
        refusal = L.successor_refusal(row, target, kind, state)
        if refusal:
            self.notes.append(refusal)
        return refusal is None

    def _is_rejected_any_case(self, repo: str) -> bool:
        """The rejection list against a Hub id: ids are compared without case (the rows' own check stays as it was)."""
        key, owner = hf_id_key(repo), hf_id_key(repo.split("/", 1)[0])
        return any(hf_id_key(r) == key for r in self.rejected_repos) or any(hf_id_key(a) == owner for a in self.rejected_accounts)

    def _take_over_todays_proposals(self) -> None:
        """A second run on one day does not take the rows the first one tried (`package_attempts`), so what the first one
        proposed for them comes from its file, which this run is about to replace. Its package changes are carried exactly
        as those of an earlier day (`R.carry_over`: against the table as it is now, ids without case), and they stand over
        those, being built on them."""
        earlier = self._json(self.out / self.today / "proposal.json")
        if not isinstance(earlier, dict):
            return
        package_changes = [c for c in R._entries(earlier, "changes") if c.get("change") == "field" and c.get("field") in _PACKAGE_FIELDS]
        carried = R.carry_over({"changes": package_changes}, None, self.rows, set(self.accounts), self._is_rejected_any_case, self.today)[0]
        self.changes.update({key: change for key, change in carried.items() if not R.touched(change) & self.done_fields})

    def _stage_packages(self) -> None:
        """Stage 1 and the setup notes (`catalog_scan_packages`): the rows of this day, their findings as field changes."""
        if not self.packages:
            return
        if self.document.get("schema_version") not in (3, 4):
            self.notes.append("package checks: the working catalog is schema 2, which has no gguf_checks or setup_note; the stage did not run")
            return
        today = date.fromisoformat(self.today)
        self._take_over_todays_proposals()
        for key in [k for k in self.changes if k[0] == "field" and k[2] in _PACKAGE_FIELDS]:
            if self._is_rejected_any_case(key[1]):
                del self.changes[key]  # a row rejected since, in any spelling: its package proposals go, whichever day they come from
            elif key[2] == "gguf_checks" and R.refresh_field(self.changes[key], self.rows[key[1]], self._is_rejected_any_case) is None:
                del self.changes[key]  # carried with the case-sensitive list: a repository or account rejected in any spelling adds nothing
        carried_checks = {key[1]: change["fields"]["gguf_checks"]["new"] for key, change in self.changes.items() if key[0] == "field" and key[2] == "gguf_checks"}
        carried_notes = {key[1]: change["fields"]["setup_note"]["new"] for key, change in self.changes.items() if key[0] == "field" and key[2] == "setup_note"}
        both = _Both(self.transport, self.main, self.pkg)
        stage = K.PackageStage(both, self.rows, today, self.package_attempts, carried_checks=carried_checks,
                               carried_notes=carried_notes, is_rejected=self._is_rejected_any_case, limit=self.package_rows,
                               remaining=both.remaining, packagers=self.packagers, unit_attempts=self.unit_attempts, reserve_days=self.reserve_days)
        result = stage.run()
        self.notes += result.notes
        self.problem = self.problem or result.problem
        self.budget_end = self.budget_end or result.budget_end
        self.package_attempts = record_attempts(self.package_attempts, result.attempted, today)
        units = K.record_units(self.unit_attempts, result.units, today)
        self.unit_attempts = {name: day for name, day in units.items() if (today - day).days <= K.UNIT_ATTEMPTS_KEPT}
        self.reserve_days = record_attempts(self.reserve_days, result.reserved, today)
        for repo, checks in sorted(result.checks.items()):
            row, key = self.rows[repo], ("field", repo, "gguf_checks")
            table = [c.model_dump(mode="json") for c in row.gguf_checks]
            if checks == table:
                self.changes.pop(key, None)  # what was proposed before is what the table holds now, or no longer follows from the source
                continue
            evidence = [llamacpp_page_url(result.commit)] + [f"{R.HUB}/{r}" for r in result.page_urls.get(repo, [])]
            self._put(key, R.field_change(row, K.STAGE, {"gguf_checks": (table, checks)}, evidence, self.today, "gguf_checks"))
        for repo, note in sorted(result.setup_notes.items()):
            row, key = self.rows[repo], ("field", repo, "setup_note")
            if note.get("vllm_args") and self.document.get("schema_version") == 3:  # schema 3 has no switches: the note stands without them
                note = {name: value for name, value in note.items() if name != "vllm_args"}
                self.notes.append(f"{repo}: the card names vLLM switches; the working catalog is schema 3, which has no vllm_args: left out")
            table = row.setup_note.model_dump(mode="json") if row.setup_note else None
            if note == table:
                self.changes.pop(key, None)
                continue
            self._put(key, R.field_change(row, K.STAGE, {"setup_note": (table, note)}, [note["source_url"]], self.today, "setup_note"))

    def _capped(self) -> list[dict]:
        overdue, rest = N.split_overdue(list(self.cands.values()), self.today)  # before the limit of ten
        kept, dropped = R.split_candidates(rest)
        if dropped:
            self.notes.append(f"candidates B1 and B2 over the limit of {R.MAX_CANDIDATES}: {dropped} not listed")
        return overdue + kept

    def _unread_notes(self) -> None:
        if self.unread["accounts"]:
            self.notes.append(f"budget ended: {len(self.unread['accounts'])} accounts not read: {', '.join(self.unread['accounts'])}")
        for name in ("seed", "B1", "B2", "collections"):
            if self.unread[name]:
                self.notes.append(f"budget ended: {len(self.unread[name])} {name} pages not read")
        self.incomplete.update(self.unread["accounts"])

    def execute(self) -> ScanResult:
        self.left_out: dict[str, str] = {}  # repo -> kind of a new row the last `_takes` refused
        self._history()
        self._carry()
        if self.offline:
            self.carried = set()
            self._successors()  # the name rule over the table: no request, no mark moves
        else:
            self._recheck()
            for account in self.accounts:
                if self.budget_end:
                    self.unread["accounts"].append(account)
                else:
                    self._account(account)
            self._seed()
            self._open_pages()
            self._build()
            self._stage_packages()
            self._unread_notes()
            oldest = min(covered for covered, _ in self.marks.values()) - R.OVERLAP
            self.notes.append(f"watermark: entries created before {oldest.date().isoformat()} are read for existing rows only")
        try:
            changes, left_out = R.check_changes(self.document, list(self.changes.values()))
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        self.notes += left_out
        changes = [R.mark_reading(c) for c in changes]  # the `reading` mark of every new row, carried ones too
        candidates = self._capped()
        R.order_proposal(changes, candidates)
        proposal = R.proposal_document(self.today, self.catalog_sha, list(dict.fromkeys(self.notes)), changes, candidates,
                                       len(self.incomplete), self.pending_publication, N.drift_triples(self.rows.values()))
        marks = {a: self.new_marks.get(a, self.marks[a]) for a in self.accounts}
        state = {"schema": STATE_SCHEMA, "accounts": {a: {"covered_until": R.stamp(c), "pending": p} for a, (c, p) in marks.items()}}
        for name, attempts in (("package_attempts", self.package_attempts), ("package_unit_attempts", self.unit_attempts), ("package_reserve_days", self.reserve_days)):
            if attempts:
                state[name] = {unit: day.isoformat() for unit, day in sorted(attempts.items())}
        try:  # all three files exist in memory before the first is written
            data = R.dump_json(proposal)
            text, state_bytes = R.render_proposal(proposal, hashlib.sha256(data).hexdigest()).encode("utf-8"), R.dump_json(state)
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            raise UsageError(f"the proposal cannot be written: {_clean(exc)}") from exc
        folder = self.out / self.today
        self.files.write(folder / "proposal.json", data)
        self.files.write(folder / "proposal.md", text)
        self.files.write(self.out / "scan-state.json", state_bytes)
        return ScanResult(int(self.budget_end or self.problem), None, proposal, self.main.used, self.sub.used, self.kind_check)


def _attempts_of(raw_state: object, name: str = "package_attempts") -> dict[str, date]:
    """The day each row (`package_attempts`) or each repository and package (`package_unit_attempts`, `K.unit`) was last
    tried by the package stage, from the state file; what does not read is never tried. One id in two spellings is one
    entry with the younger day (`record_attempts`)."""
    found = raw_state.get(name) if isinstance(raw_state, dict) else None
    attempts: dict[str, date] = {}
    for unit, day in (found.items() if isinstance(found, dict) else ()):
        try:
            validate_hf_repo(unit.split(":", 1)[0] if name == "package_unit_attempts" else unit)
            attempts[unit] = date.fromisoformat(day)
        except (ValueError, TypeError):
            continue
    return K.record_units(attempts, [], date.min) if name == "package_unit_attempts" else record_attempts(attempts, [], date.min)


def scan(catalog_path, rejected_path, out_dir, *, transport: Transport, now: datetime, budget: int = DEFAULT_BUDGET,
         sub_budget: int = STAGE_BUDGET, files: DiskFiles | None = None, packages: bool = False,
         package_budget: int = PACKAGE_STAGE_BUDGET, package_rows: int = PACKAGE_ROWS_PER_RUN, offline: bool = False) -> ScanResult:
    """One run. `transport` is the only way out to the network, `now` the only clock, `files` the only disk. `packages`
    switches on the package stage (off, the run asks what it asked before AP9-KO). `offline` asks nobody: the run makes the
    proposals of the name rule from the table and carries the open ones, and moves no mark."""
    try:
        run = _Run(catalog_path, rejected_path, out_dir, transport, now, budget, sub_budget, files or DiskFiles(), packages, package_budget, package_rows, offline)
        return run.execute()
    except UsageError as exc:
        return ScanResult(exit_code=2, error=str(exc))


def main(argv: list[str] | None = None, *, transport: Transport | None = None, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propose changes to the catalog from the Hub; never change it.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--today", help="the day of the run, YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--packages", action="store_true", help="also check the GGUF packages and the model cards (needs a catalog of schema 3)")
    parser.add_argument("--offline", action="store_true", help="no request: only the successors of the name rule, made from the table")
    args = parser.parse_args(argv)
    try:
        moment = datetime.combine(datetime.fromisoformat(args.today).date(), time.min, tzinfo=timezone.utc) if args.today else now or datetime.now(timezone.utc)
    except ValueError:
        print(f"--today is no date: {args.today}", file=sys.stderr)
        return 2
    # the hosts of the source and the Hub's CDN are the scanner's alone: `http._ALLOWED_HTTPS_HOSTS` stays what it is
    transport = transport or (UrllibTransport(allowed_https_hosts=K.ScannerHosts()) if args.packages else UrllibTransport())
    result = scan(args.catalog, args.rejected, args.out, transport=transport, now=moment, budget=args.budget, packages=args.packages, offline=args.offline)
    if result.error:
        print(result.error, file=sys.stderr)
    elif result.proposal:
        print(f"proposal {result.proposal['date']}: {result.proposal['summary']}; {result.main_used} requests")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
