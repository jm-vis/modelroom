"""Criterion 11 of `scripts/selftest.py`: the head count of step 3, as the first run kept it.

Both runs answer `users = 1`: one person, one request -- the request a measurement runs, so the
measurement of criterion 5 stays in measured group 0 in both runs. After the first run the
configuration and the document both hold `users = 1`, `requests_origin = "from_users"` and one
request; any other value fails the self-test. The Markdown sentence `1 request (from 1 user)` is
read from the document alone, so the document is checked as well as the configuration.

Imported by `scripts/selftest.py`, which keeps the run and the report.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

NAME = "(11) the head count of run 1 is kept: one person, from_users, one request"


def head_count_problems(guided: dict, document: dict) -> list[str]:
    """`guided` is the `[guided]` table the first run left, `document` the `models.json` it wrote."""
    found = [
        ("[guided].users", guided.get("users"), 1),
        ("[guided].requests_origin", guided.get("requests_origin"), "from_users"),
        ("[guided].requests", guided.get("requests"), 1),
        ("the document's users", document.get("users"), 1),
        ("the document's requests_origin", document.get("requests_origin"), "from_users"),
        ("the document's scenario.requests", document.get("scenario", {}).get("requests"), 1),
    ]
    return [f"{name} is {value!r}, expected {wanted!r}" for name, value, wanted in found if value != wanted]


def head_count_step(results: Path) -> tuple[str, bool, str]:
    """Criterion 11, read from the files the first run left: the configuration and the document."""
    guided = tomllib.loads((results / "modelroom.toml").read_text(encoding="utf-8")).get("guided", {})
    document = json.loads((results / "docs" / "models.json").read_text(encoding="utf-8"))
    problems = head_count_problems(guided, document)
    detail = "\n".join(problems) or "users 1, requests_origin from_users, 1 request in the configuration and the document"
    return NAME, not problems, detail
