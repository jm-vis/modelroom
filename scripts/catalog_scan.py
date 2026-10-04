"""The catalog scanner: reads the Hub, writes a proposal, never changes the table (AP9-KA-T).

    python scripts/catalog_scan.py --catalog modelroom/catalog.json --rejected modelroom/catalog-rejected.json \\
        --out <folder> [--today YYYY-MM-DD] [--budget 300]

It reads the two files and the output folder, asks the Hub with `GET` only, and writes `<out>/<day>/proposal.json`,
`<out>/<day>/proposal.md` and `<out>/scan-state.json`, in that order, each atomically, the state last: a run that stops
in between leaves the old state and the next run reads the same pages again. It cannot write `catalog.json` or
`catalog-rejected.json`; taking a proposal into the table is `catalog_publish.py`, after the maintainer's word. Exit 0;
exit 1 when the budget ended, a page failed or a stored position was lost (the proposal is written all the same); exit 2
when the call or a file is wrong (nothing is written). The rules are in `catalog_scan_rules.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the rules module sits beside this script
import catalog_scan_rules as R  # noqa: E402

from modelroom.catalog import CatalogError, CatalogModel  # noqa: E402
from modelroom.catalog_remote import _clean  # noqa: E402
from modelroom.catalog_schema import build_catalog_v2  # noqa: E402
from modelroom.contracts import SchemaVersionError, validate_hf_repo  # noqa: E402
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
_FAILURES = (OSError, http.client.HTTPException, RuntimeError, ResponseTooLargeError, TransportSecurityError)


class UsageError(Exception):
    """The call or one of the two input files is wrong: exit 2, nothing written."""


class _Fail(Exception):
    """One answer that cannot be used; its text is the reason in the proposal, `status` the HTTP status if there was one."""

    status: int | None = None


class _Lost(Exception):
    """A stored next page that does not answer: the position of a catch-up is lost."""


class DiskFiles:
    """The only way the scanner touches a disk; a test hands in a subclass to see every path."""

    def read(self, path) -> bytes:
        return Path(path).read_bytes()

    def write(self, path, data: bytes) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "wb") as handle:
                handle.write(data)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
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

    def __call__(self, method, url, headers=None, *, max_bytes=None):
        if self._main.remaining <= 0 or self._sub.remaining <= 0:
            raise BudgetExhaustedError()
        return self._inner(method, url, headers, max_bytes=max_bytes)


class _Run:
    def __init__(self, catalog_path, rejected_path, out_dir, transport, now, budget, sub_budget, files) -> None:
        self.files, self.out, self.now = files, Path(out_dir), now.astimezone(timezone.utc)
        self.today = self.now.date().isoformat()
        outputs = {(self.out / self.today / name).resolve() for name in ("proposal.json", "proposal.md")} | {(self.out / "scan-state.json").resolve()}
        for given, what in ((catalog_path, "catalog"), (rejected_path, "rejection list")):
            if Path(given).resolve() in outputs:  # the run would overwrite what it reads
                raise UsageError(f"{given}: the {what} is also a file this run writes; choose another --out")
        catalog_bytes = self._read(Path(catalog_path), "catalog")
        self.catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
        try:
            self.document = json.loads(catalog_bytes.decode("utf-8"))
            self.catalog = build_catalog_v2(catalog_bytes.decode("utf-8"), Path(catalog_path))
        except (ValueError, CatalogError, SchemaVersionError) as exc:
            raise UsageError(f"{catalog_path}: not a valid catalog: {_clean(exc)}") from exc
        self.rows = {m.hf_repo: m for f in self.catalog.families for m in f.models}
        self.publishers = [p.account for p in self.catalog.publisher_info]
        self.packagers = list(self.catalog.packagers)
        self.accounts = self.publishers + [a for a in self.packagers if a not in self.publishers]
        try:
            self.rejected_repos, self.rejected_accounts = R.parse_rejected(self._read(Path(rejected_path), "rejection list"))
        except ValueError as exc:
            raise UsageError(f"{rejected_path}: not a valid rejection list: {_clean(exc)}") from exc
        self.main, self.sub = RequestBudget(budget), RequestBudget(sub_budget)
        both = _Both(transport, self.main, self.sub)
        self.t_list = RedirectingTransport(BudgetedTransport(transport, self.main), max_hops=1)
        self.t_base, self.t_config = RedirectingTransport(both, max_hops=1), RedirectingTransport(both, max_hops=2)
        self.notes: list[str] = []
        self.marks, self.new_marks, self.oldest = {}, {}, {}  # per account: (covered_until, pending), the same after the run, oldest time read
        self.incomplete: set[str] = set()
        self.unread: dict[str, list] = {"accounts": [], "B1": [], "B2": [], "collections": []}
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
        days = sorted(n for n in self.files.folders(self.out) if _DAY.fullmatch(n) and n < self.today)
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
        stored = raw_state.get("accounts") if isinstance(raw_state, dict) else None
        for account in self.accounts:
            mark = stored.get(account) if isinstance(stored, dict) else None
            covered = R.parse_time(mark.get("covered_until")) if isinstance(mark, dict) else None
            pending = mark.get("pending") if isinstance(mark, dict) and isinstance(mark.get("pending"), dict) else None
            self.marks[account] = (covered or start - R.FIRST_RUN_BACK, pending if covered else None)

    def _carry(self) -> None:
        """Every change and candidate of the newest earlier proposal that is still open comes back with its first date."""
        if self.carry_from is not None:
            self.changes, self.cands, self.known, self.done_fields, self.done_accounts = R.carry_over(self.carry_from, self.decision, self.rows, set(self.accounts), self._is_rejected)

    # --- asking -----------------------------------------------------------------------------------------------

    def _get(self, layer, url: str, limit: int):
        try:
            response = layer("GET", url, None, max_bytes=limit)
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

    def _list(self, url: str, account: str | None = None):
        """One list answer and the next page it announces (`None` at the end); a next page elsewhere is a fault."""
        response = self._get(self.t_list, url, R.API_MAX_BYTES)
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
        if base:
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

    def _see(self, entry: R.Entry) -> None:
        if entry.repo in self.seen:
            return
        self.seen.add(entry.repo)
        if self._is_rejected(entry.repo):
            return  # a rejected repo or account is read for nothing: no check of a row, no card, no evidence
        row = self.rows.get(entry.repo)
        if row is not None:
            self._existing(entry, row)
            return
        self._mentions(entry)
        self._gguf(entry)
        if (
            entry.repo in self.known or R.is_quantization(entry.tags, entry.repo)
            or owner_class(entry.owner, catalog=self.catalog, listed_packagers=self.packagers) != "publisher"
            or any(base.split("/", 1)[0] != entry.owner for base in R.base_repos(entry.tags))
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

    def _walk(self, account: str, url: str, until, pages: int, resume: bool = False):
        """Pages from `url` until an entry satisfies `until` or the list ends:
        `(done, next url, pages used, newest time, entries seen)`."""
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
                self._see(entry)
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
                done, following, _, newest, entries = self._walk(account, top, lambda t: t < threshold, limit)
                state = (max(covered, newest) if newest else covered, None) if done else (covered, {"next_url": following, "newest_seen": R.stamp(newest or covered)})
            else:
                seen = R.parse_time(pending.get("newest_seen")) or covered
                done, following, used, newest, entries = self._walk(account, top, lambda t: t <= seen, limit)
                latest = max(seen, newest) if newest else seen
                if not done:
                    state = (covered, {"next_url": following, "newest_seen": R.stamp(latest)})
                elif limit - used <= 0:
                    state = (covered, {"next_url": pending["next_url"], "newest_seen": R.stamp(latest)})
                else:
                    try:
                        done, following, _, older, more = self._walk(account, pending["next_url"], lambda t: t < threshold, limit - used, resume=True)
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
                everything = sorted((e for es in per_row.values() for e in es), key=lambda e: (-e.likes, e.repo))
                self._candidate("B2", account, everything, {"rows": len(per_row), "bases": sorted(per_row)})

    def _step(self, name: str, url: str, what: str) -> list:
        """One request of an open step: the list, or nothing (a note for a fault, a count for the end of the budget)."""
        try:
            return self._list(url)[0]
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
                change["fields"]["gguf_repos"]["new"] = (joined + [r for r in change["fields"]["gguf_repos"]["new"] if r not in joined])[:50]
                change["evidence"] = before.get("evidence", []) + [e for e in change["evidence"] if e not in before.get("evidence", [])]
        self.changes[key] = change

    def _carried_rows(self) -> list[CatalogModel]:
        """The new rows carried from the last proposal, as rows; one that does not validate is left to `check_changes`."""
        rows = []
        for key, change in sorted(self.changes.items()):
            if key[0] == "new_row":
                try:
                    rows.append(CatalogModel.model_validate({name: pair["new"] for name, pair in change["fields"].items()}))
                except (ValueError, TypeError):
                    continue
        return rows

    def _build(self) -> None:
        table = [m for m in self.rows.values() if not self._is_rejected(m.hf_repo)]
        rows = list(self.rows.values()) + self._carried_rows()  # family and latest see the table and the carried new rows
        fresh = sorted(self.fresh.values(), key=lambda item: item[0].repo)
        latest = R.latest_among([(e.repo, r.kind, e.parameters_b) for e, r, _ in fresh], rows)
        plain = [m for m in table if m.latest and not (m.latest_source or "").startswith(f"{R.HUB}/collections/{m.publisher}/")]
        owners = [a for a in self.publishers if any(e.owner == a and e.repo in latest for e, _, _ in fresh) or any(m.publisher == a for m in plain)]
        collections = self._collections(owners)
        for entry, result, open_since in fresh:
            family = R.family_for(entry.repo, result.kind, rows, entry.parameters_b)
            if family is None:
                self.notes.append(f"{entry.repo}: no family word in the name, not proposed")
                self.fresh.pop(entry.repo)
                continue
            source = collections.get(entry.owner, {}).get(entry.repo) or f"{R.HUB}/{entry.owner}"
            row = R.build_row(entry, result, family=family, today=self.today, latest=entry.repo in latest, latest_source=source)
            row["gguf_repos"] = sorted(self.gguf.get(entry.repo, ()))[:50]
            try:
                rows.append(CatalogModel.model_validate(row))  # the next new row sees this one: one family, one publisher
            except ValueError as exc:  # a strange Hub answer costs this row, never the run
                self.notes.append(f"{entry.repo}: left out, it would not validate ({' '.join(str(exc).split())[:120]})")
                self.fresh.pop(entry.repo)
                continue
            self._put(("new_row", entry.repo), R.new_row_change(row, result, open_since))
        for base, found in sorted(self.gguf.items()):
            row = None if self._is_rejected(base) else self.rows.get(base)
            added = sorted(r for r in found if r not in row.gguf_repos) if row is not None else []
            if added:
                self._put(("field", base, "gguf_repos"), R.field_change(
                    row, "A2", {"gguf_repos": (list(row.gguf_repos), (list(row.gguf_repos) + added)[:50])}, [f"{R.HUB}/{r}" for r in added], self.today, "gguf_repos"))
        for row, target in sorted(self.successors, key=lambda pair: pair[0].hf_repo):
            self._successor(row, target)
        for row in plain:
            link = collections.get(row.publisher, {}).get(row.hf_repo)
            if link and ("field", row.hf_repo, "successor") not in self.changes:
                self._put(("field", row.hf_repo, "latest_source"), R.field_change(
                    row, "A3", {"latest_source": (row.latest_source, link)}, [link], self.today, "latest_source"))

    def _successor(self, row: CatalogModel, target: str) -> None:
        """`new_version` on the card of a row names a model of the same account and kind that the table or this proposal
        has (a new row of this run or a carried one, as it stands; a carried row is never changed)."""
        found, proposed = self.rows.get(target), self.changes.get(("new_row", target))
        kind = found.kind if found is not None else proposed["kind"] if proposed is not None else None
        if self._is_rejected(target):
            kind = None
        try:
            validate_hf_repo(target)
        except ValueError:
            kind = None
        if kind is None or target.split("/", 1)[0] != row.publisher or target == row.hf_repo:
            self.notes.append(f"{row.hf_repo}: card names new_version {target}, which is not a model of the table or of this proposal; no successor proposed")
            return
        if kind != row.kind or row.successor is not None:
            self.notes.append(f"{row.hf_repo}: card names new_version {target} of another kind or the row has a successor; no successor proposed")
            return
        card = f"{R.HUB}/{row.hf_repo}"
        pairs = {"successor": (None, target), "successor_source": (None, card), "successor_checked": (None, self.today)}
        if row.latest:
            pairs.update(latest=(True, False), latest_source=(row.latest_source, None),
                         latest_checked=(row.latest_checked.isoformat() if row.latest_checked else None, None))
        self._put(("field", row.hf_repo, "successor"), R.field_change(row, "A3", pairs, [card], self.today, "successor"))

    def _capped(self) -> list[dict]:
        kept, dropped = R.split_candidates(list(self.cands.values()))
        if dropped:
            self.notes.append(f"candidates B1 and B2 over the limit of {R.MAX_CANDIDATES}: {dropped} not listed")
        return kept

    def _unread_notes(self) -> None:
        if self.unread["accounts"]:
            self.notes.append(f"budget ended: {len(self.unread['accounts'])} accounts not read: {', '.join(self.unread['accounts'])}")
        for name in ("B1", "B2", "collections"):
            if self.unread[name]:
                self.notes.append(f"budget ended: {len(self.unread[name])} {name} pages not read")
        self.incomplete.update(self.unread["accounts"])

    def execute(self) -> ScanResult:
        self._history()
        self._carry()
        self._recheck()
        for account in self.accounts:
            if self.budget_end:
                self.unread["accounts"].append(account)
            else:
                self._account(account)
        self._open_pages()
        self._build()
        self._unread_notes()
        try:
            changes, left_out = R.check_changes(self.document, list(self.changes.values()))
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        self.notes += left_out
        candidates = self._capped()
        R.order_proposal(changes, candidates)
        proposal = R.proposal_document(self.today, self.catalog_sha, list(dict.fromkeys(self.notes)), changes, candidates,
                                       len(self.incomplete), self.pending_publication)
        data = R.dump_json(proposal)
        marks = {a: self.new_marks.get(a, self.marks[a]) for a in self.accounts}
        state = {"schema": STATE_SCHEMA, "accounts": {a: {"covered_until": R.stamp(c), "pending": p} for a, (c, p) in marks.items()}}
        folder = self.out / self.today
        self.files.write(folder / "proposal.json", data)
        self.files.write(folder / "proposal.md", R.render_proposal(proposal).encode("utf-8"))
        self.files.write(self.out / "scan-state.json", R.dump_json(state))
        return ScanResult(int(self.budget_end or self.problem), None, proposal, self.main.used, self.sub.used, self.kind_check)


def scan(catalog_path, rejected_path, out_dir, *, transport: Transport, now: datetime, budget: int = DEFAULT_BUDGET,
         sub_budget: int = STAGE_BUDGET, files: DiskFiles | None = None) -> ScanResult:
    """One run. `transport` is the only way out to the network, `now` the only clock, `files` the only disk."""
    try:
        run = _Run(catalog_path, rejected_path, out_dir, transport, now, budget, sub_budget, files or DiskFiles())
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
    args = parser.parse_args(argv)
    try:
        moment = datetime.combine(datetime.fromisoformat(args.today).date(), time.min, tzinfo=timezone.utc) if args.today else now or datetime.now(timezone.utc)
    except ValueError:
        print(f"--today is no date: {args.today}", file=sys.stderr)
        return 2
    result = scan(args.catalog, args.rejected, args.out, transport=transport or UrllibTransport(), now=moment, budget=args.budget)
    if result.error:
        print(result.error, file=sys.stderr)
    elif result.proposal:
        print(f"proposal {result.proposal['date']}: {result.proposal['summary']}; {result.main_used} requests")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
