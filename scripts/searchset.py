"""The search test set (`tests/search_testset.toml`): read it, ask it live, report it line by line.

Beside `scripts/selftest.py` rather than inside it (2026-09-25): the acceptance script stands at the
house code-mass threshold, and reading a data file is a job of its own. The set is what answers "does
a person who types this get that model", and the only place that question *can* be answered is the
live Hub -- so criterion 8 is **gate-free**: a model that is missing today is a `WARN` and never ends
the run. What is gated is the shape of the report, which `tests/test_selftest.py` checks against
recorded fixtures.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTSET_FILE = REPO / "tests" / "search_testset.toml"
TESTSET_SCHEMA_VERSION = 1


class TestsetError(Exception):
    """The test set cannot be read, or one of its cases is not a case."""


def load_testset(path: Path = TESTSET_FILE) -> list[dict]:
    """Every case of the test set, in the order the file lists them.

    Read and not hard-coded, so that a case can be added without touching a script. A file that does
    not read, or a case without the three fields a case has, raises: the set is the criterion, and a
    criterion nobody can read says nothing.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise TestsetError(f"{path}: cannot read the search test set: {exc}") from exc
    if raw.get("schema_version") != TESTSET_SCHEMA_VERSION:
        raise TestsetError(
            f"{path}: schema_version is {raw.get('schema_version')!r}, expected {TESTSET_SCHEMA_VERSION}"
        )
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise TestsetError(f"{path}: the test set holds no case")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("input"), str):
            raise TestsetError(f"{path}: a case has no input: {case!r}")
        if not isinstance(case.get("filter"), bool) or not isinstance(case.get("models"), list):
            raise TestsetError(f"{path}: the case {case.get('input')!r} has no filter or no models")
    return cases


def testset_line(case: dict, found: list[str], candidates: list[str], error: str | None = None) -> str:
    """One line of the report: what was typed, and what came back for it.

    `OK` when every model the case names is there, `WARN` when one is missing -- and the words the
    candidates offered instead, because a word with no answer and a word with a suggestion are two
    different results (decided 2026-09-25).
    """
    shown = case["input"] or "(no word)"
    if error is not None:
        return f"WARN {shown}: {error}"
    missing = [model for model in case["models"] if model not in found]
    mark = "OK  " if not missing else "WARN"
    tail = f", missing {', '.join(missing)}" if missing else ""
    hint = f", did you mean {', '.join(candidates)}" if candidates else ""
    return f"{mark} {shown}: {len(found)} models{tail}{hint}"


def testset_problems(cases: list[dict], lines: list[str]) -> list[str]:
    """What is wrong with the *report* rather than with the Hub: one line per case, each a verdict."""
    problems = []
    if len(lines) != len(cases):
        problems.append(f"{len(lines)} report lines for {len(cases)} cases")
    if any(not line.startswith(("OK  ", "WARN ")) for line in lines):
        problems.append(f"a report line carries no verdict: {lines}")
    return problems


def ask_testset(cases: list[dict]) -> list[str]:
    """Ask every case against the live Hub and return one report line each.

    One request budget per case, so a long answer for one word cannot starve the next. Any failure
    is that case's own line -- the report is a record, never a gate.
    """
    from modelroom.catalog import load_catalog
    from modelroom.http import RequestBudget, UrllibTransport
    from modelroom.search import DEFAULT_GUIDED_BUDGET, run_search

    catalog = load_catalog()
    lines = []
    for case in cases:
        try:
            outcome = run_search(
                UrllibTransport(),
                case["input"],
                catalog=catalog,
                budget=RequestBudget(DEFAULT_GUIDED_BUDGET),
                open_pages=not case["filter"],
                publisher_required=case["filter"],
            )
        except Exception as exc:  # noqa: BLE001 -- a live answer never fails the run
            lines.append(testset_line(case, [], [], f"{type(exc).__name__}: {exc}"))
            continue
        found = sorted({str(hit.resolved_base_model) for hit in outcome.hits if hit.resolved})
        lines.append(testset_line(case, found, outcome.candidates))
    return lines


__all__ = [
    "TESTSET_FILE",
    "TESTSET_SCHEMA_VERSION",
    "TestsetError",
    "ask_testset",
    "load_testset",
    "testset_line",
    "testset_problems",
]
