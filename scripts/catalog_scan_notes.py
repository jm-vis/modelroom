"""Notes and small rules of the catalog scanner that need no request and no clock (AP9-KA-H).

Pure functions over data and the day of the run; no function changes what it is given. This module imports
`search_release`, `catalog_scan_rules` and `catalog_scan_latest`, never `catalog_scan`: the scanner calls it and stays the
owner of the proposal. `catalog_scan_rules` does not import this module.

- K1: why a name has no reading (`unread_notes`);
- K3: a shipped row whose family the grammar no longer forms from its name (`drift_notes`);
- K4: a candidate open for more than `CANDIDATE_DECISION_DAYS` days (`split_overdue`).

The K2 rules (the top of a group, the successors of one run) are in `catalog_scan_latest`.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the rules module sits beside this one
import catalog_scan_latest  # noqa: E402,F401  (the K2 rules sit beside this module; the scanner calls them from there)
import catalog_scan_rules as R  # noqa: E402

from modelroom.search_release import unread_reason  # noqa: E402


def unread_notes(repo: str, size: float | None, family: str) -> list[str]:
    """The note of a new row whose name the grammar does not read, with the reason; no note for a name it reads."""
    why = unread_reason(repo, size)
    return [f"{repo}: no reading ({why}), family {family} taken from its first word"] if why else []


def field_values(changes: dict) -> dict:
    """`{key: {field: new value}}` of the field changes in `changes` (keyed `("field", repo, name)`), for a later comparison."""
    return {key: {name: pair.get("new") for name, pair in change.get("fields", {}).items()} for key, change in changes.items() if key[0] == "field"}


def new_field_repos(before: dict, changes: dict) -> set[str]:
    """The repos with a field change that is new in this run or differs in its new values from `before` (`field_values` taken
    when the carried changes were all there was); a change taken over unchanged is no new member of its group (Rev 7)."""
    return {key[1] for key, values in field_values(changes).items() if before.get(key) != values}


def drift_triples(rows) -> list[list]:
    """`[repo, family in the catalog, forms today]` of every row it is given whose stored family is none of the forms the
    grammar makes from its name today (`family_forms`). This is what the scanner keeps in the proposal (`drift`) for the
    next run. A lost or gained reading is not drift."""
    triples = []
    for row in sorted(rows, key=lambda r: r.hf_repo):
        forms = R.family_forms(row.hf_repo, row.parameters_b)
        if row.family not in forms:
            triples.append([row.hf_repo, row.family, list(forms)])
    return triples


def last_drift(proposal: object) -> list[list] | None:
    """The drift triples the earlier proposal kept, `None` when there is none (the first run, or a proposal of the old form)."""
    found = proposal.get("drift") if isinstance(proposal, dict) else None
    return [t for t in found if isinstance(t, list) and len(t) == 3] if isinstance(found, list) else None


def drift_notes(rows, last: list[list] | None = None) -> list[str]:
    """A note for every drift triple of `rows` that `last` does not hold unchanged (new drift, another family in the
    catalog, other forms today); with no `last` (the first run) every drift is named. A note only, the table is not changed."""
    return [
        f"{repo}: reading drift, family {family} in the catalog, {forms[0] + ' today' if forms else 'no family form today'}"
        for repo, family, forms in drift_triples(rows) if last is None or [repo, family, forms] not in last
    ]


def days_open(open_since: object, today: str) -> int | None:
    """The whole days from `open_since` to `today` (both `YYYY-MM-DD`, UTC), `None` for anything else."""
    try:
        return (date.fromisoformat(today) - date.fromisoformat(open_since)).days  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def split_overdue(candidates: list[dict], today: str) -> tuple[list[dict], list[dict]]:
    """`(overdue, the rest)`, copies of the candidates: the ones given stay as they are. A candidate is overdue from day 31
    on (`days_open > CANDIDATE_DECISION_DAYS`; day 30 is still in time) and its copy carries `overdue_days`; the copy of
    every other one has no such key, so a value carried from an earlier run never outlives the day it was true."""
    overdue, rest = [], []
    for original in candidates:
        cand = {key: value for key, value in original.items() if key != "overdue_days"}
        days = days_open(cand.get("open_since"), today)
        if days is not None and days > R.CANDIDATE_DECISION_DAYS:
            cand["overdue_days"] = days
            overdue.append(cand)
        else:
            rest.append(cand)
    return overdue, rest
