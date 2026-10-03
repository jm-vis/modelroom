"""Fill `hub_pipeline_tag` and `kind_source` of the shipped catalog from the Hub, once.

One `GET https://huggingface.co/api/models/<hf_repo>` per row (no token, one after the other, at
most one redirect, at most one mebibyte), so 62 requests for 62 rows. The task is the answer's top
field `pipeline_tag`, verbatim (`cardData` is never read). Evidence per row, in this order:

1. the task is in the kind table: `kind_source` stays `null`; a task that contradicts `kind` is an
   error, and `kind` itself is never changed;
2. otherwise every table tag of the answer gives one kind, and it is `kind`: `kind_source` is `tags`;
3. otherwise only the repos of `DECISION_REPOS`, whose model cards were read on 2026-10-01, get
   `decision` and a fixed sentence in `note`; any other row is an error.

Nothing is written unless every row passes and the result loads as a catalog; the write is atomic.
A second run over a filled file changes no byte. A 429 stops the run at once. This script and its
test are removed after the review (CHANGELOG names the commit that holds them).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from modelroom.catalog import CATALOG_FILE, CATALOG_MAX_TEXT_LEN, PIPELINE_TAG_TO_KIND, CatalogError, load_catalog
from modelroom.http import (
    BudgetedTransport, BudgetExhaustedError, RedirectingTransport, RedirectLimitError, RequestBudget,
    ResponseTooLargeError, Transport, UrllibTransport,
)

API = "https://huggingface.co/api/models/"
MAX_BYTES = 1024 * 1024
DECISION_DATE = "2026-10-01"
DECISION_REPOS = (
    "mistralai/Ministral-3-14B-Instruct-2512",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Mistral-Small-4-119B-2603",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "mistralai/Magistral-Small-2509",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
)
_TASK_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class RowError(Exception):
    """One row cannot be filled; the message says why (the caller adds the repo)."""


class StopRun(Exception):
    """The run ends at once: a 429 or the end of the budget."""


def task_of(body: object) -> str | None:
    """The answer's top field `pipeline_tag`: `None` if missing or null, the text if it fits the pattern."""
    if not isinstance(body, dict):
        raise RowError("the answer is not a JSON object")
    task = body.get("pipeline_tag")
    if task is None:
        return None
    if not isinstance(task, str) or len(task) > 64 or not _TASK_RE.fullmatch(task):
        raise RowError(f"pipeline_tag does not fit the pattern: {task!r}")
    return task


def _table_tag_kinds(body: dict) -> set[str]:
    tags = body.get("tags")
    return {PIPELINE_TAG_TO_KIND[tag] for tag in tags if tag in PIPELINE_TAG_TO_KIND} if isinstance(tags, list) else set()


def _evidence(row: dict, task: str | None, body: dict) -> str | None:
    """`kind_source` for `row`; `RowError` where neither the task nor the tags decide it."""
    if task in PIPELINE_TAG_TO_KIND:
        if PIPELINE_TAG_TO_KIND[task] != row["kind"]:
            raise RowError(f"task {task!r} gives kind {PIPELINE_TAG_TO_KIND[task]!r}, the row has {row['kind']!r}")
        return None
    kinds = _table_tag_kinds(body)
    if len(kinds) == 1:
        if kinds != {row["kind"]}:
            raise RowError(f"the table tags give kind {sorted(kinds)}, the row has {row['kind']!r}")
        return "tags"
    if row["hf_repo"] in DECISION_REPOS:
        return "decision"
    raise RowError(f"task {task!r} and the table tags ({sorted(kinds)}) do not decide kind {row['kind']!r}")


def _fetch(transport: Transport, repo: str) -> dict:
    try:
        response = transport("GET", API + repo, None, max_bytes=MAX_BYTES)
    except BudgetExhaustedError as exc:
        raise StopRun("the request budget ended") from exc
    except (RedirectLimitError, ResponseTooLargeError) as exc:
        raise RowError(str(exc)) from exc
    if response.status == 429:
        wait = response.header("Retry-After")
        raise StopRun("429, stop" + (f"; Retry-After: {wait}" if wait else ""))
    if response.status != 200:
        raise RowError(f"status {response.status}")
    try:
        body = response.json()
    except ValueError as exc:
        raise RowError(f"the answer is no JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise RowError("the answer is not a JSON object")
    return body


def fill(path: Path, transport: Transport | None = None, budget: int = 62) -> list[str]:
    """Fill the catalog at `path`; the problems found, one per row (empty: the file was written)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    guarded = RedirectingTransport(
        BudgetedTransport(transport or UrllibTransport(), RequestBudget(budget)), max_hops=1
    )
    problems: list[str] = []
    for row in data["models"]:
        repo = row["hf_repo"]
        try:
            body = _fetch(guarded, repo)
            task = task_of(body)
            source = _evidence(row, task, body)
            sentence = f"Kind {row['kind']} from the model card, checked {DECISION_DATE}."
            note = row["note"]
            if source == "decision" and (note is None or sentence not in note):
                note = sentence if note is None else f"{note} {sentence}"
            if note is not None and len(note) > CATALOG_MAX_TEXT_LEN:
                raise RowError(f"the note would be {len(note)} characters, the limit is {CATALOG_MAX_TEXT_LEN}")
        except RowError as exc:
            problems.append(f"{repo}: {exc}")
            continue
        except StopRun as exc:
            return problems + [f"{repo}: {exc}"]
        row["hub_pipeline_tag"], row["kind_source"], row["note"] = task, source, note
    return problems or _write(path, data)


def _write(path: Path, data: dict) -> list[str]:
    """Load the result as a catalog, then replace the file atomically; the problem if it does not load."""
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=".fill-", suffix=".json")
    temp = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(text.encode("utf-8"))
        try:
            load_catalog(temp)
        except CatalogError as exc:
            return [f"the filled catalog does not load: {exc}"]
        os.replace(temp, path)
        return []
    finally:
        temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None, transport: Transport | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=CATALOG_FILE)
    args = parser.parse_args(argv)
    problems = fill(args.catalog, transport)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    print("nothing written" if problems else f"filled {args.catalog}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
