"""Tests for `modelroom.pull_other`: a package for another chosen machine (A3, AP9-FR).

Synthetic, in memory (`tests/fixture_rows.document_multi`): one document with several machine
blocks, no real daemon or disk involved -- `pull_other_step` never reads either.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.dialog import AnswerInvalidError, FileAsker
from modelroom.guided_install import PullContext
from modelroom.pull_choice import NO_PULL_NAME, NONE_VALUE, ambiguous_names, weights_bytes_of
from modelroom.pull_other import (
    KEY,
    NONE_WORDS,
    eligible_other_machines,
    other_rows,
    pull_other_choices,
    pull_other_step,
)

from fixture_rows import document_multi, package

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
UNSLOTH = "unsloth/Nova-7B-GGUF"
Q4 = package(UNSLOTH, "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
Q8 = package(UNSLOTH, "Nova-7B-Q8_0.gguf", "Q8_0")
UNNAMED = package(UNSLOTH, "Nova-7B-mystery.gguf", None)
NAME_Q4 = "hf.co/unsloth/Nova-7B-GGUF:Q4_K_M"
NAME_Q8 = "hf.co/unsloth/Nova-7B-GGUF:Q8_0"


class _Recording(FileAsker):
    """An answer file that keeps the choices of every list it was asked (`tests/test_pull_choice.py`)."""

    def __init__(self, answers: dict) -> None:
        super().__init__(answers)
        self.lists: dict[str, list] = {}

    def select(self, key, question, choices, instruction=None):
        self.lists[key] = list(choices)
        return super().select(key, question, choices, instruction)


def _run(tmp_path: Path, asker):
    from fixture_room import fixed_room
    from fixture_support import offline_servers, windows_probes
    from modelroom.catalog import load_catalog
    from modelroom.daemon import FixtureDaemon
    from modelroom.guided import GuidedRun
    from modelroom.http import FixtureTransport
    from modelroom.local_servers import ServerRun
    from modelroom.screen import Screen

    lines: list[str] = []
    run = GuidedRun(
        asker=asker,
        here=tmp_path,
        pointer_path=tmp_path / "guided.json",
        transport=FixtureTransport({}),
        probes=windows_probes(),
        now=NOW,
        catalog=load_catalog(),
        daemon=FixtureDaemon({}),
        servers=ServerRun(offline_servers()),
        out=lines.append,
        screen=Screen(lines.append, colored=False),
        room=fixed_room(),
    )
    return run, lines


def _context(per_machine: dict, machine: str = "workstation") -> PullContext:
    doc = document_multi(per_machine)
    packages = [package for packages in per_machine.values() for package in packages]
    return PullContext(doc, machine, packages, ambiguous_names(packages))


# --- the list -----------------------------------------------------------------------------------


def test_the_list_shows_every_other_machine_under_its_heading():
    doc = document_multi({"workstation": [Q4], "server-a": [Q8], "server-b": [UNNAMED]})

    choices = pull_other_choices(other_rows(doc, "workstation"))

    headings = [choice.label for choice in choices if choice.heading]
    assert headings == ["server-a", "server-b"]
    assert any(choice.label == f"#1  unsloth · Q8_0  5.0 GB  {NAME_Q8}" for choice in choices)


def test_this_machine_is_not_in_the_other_list():
    doc = document_multi({"workstation": [Q4], "server-a": [Q8]})

    rows = other_rows(doc, "workstation")

    assert rows and all(row.machine != "workstation" for row in rows)
    choices = pull_other_choices(rows)
    assert not any(choice.heading and choice.label == "workstation" for choice in choices)


def test_locked_rows_stay_locked():
    doc = document_multi({"workstation": [Q4], "server-a": [UNNAMED, Q8]})

    choices = pull_other_choices(other_rows(doc, "workstation"))

    locked = [choice for choice in choices if not choice.heading and choice.value != NONE_VALUE and choice.disabled]
    assert [choice.disabled for choice in locked] == [NO_PULL_NAME]


def test_the_pointer_starts_on_none():
    doc = document_multi({"workstation": [Q4], "server-a": [Q8]})

    choices = pull_other_choices(other_rows(doc, "workstation"))

    assert [choice.value for choice in choices if choice.checked] == [NONE_VALUE]
    assert choices[-1].value == NONE_VALUE and choices[-1].label == NONE_WORDS


# --- eligibility and the step --------------------------------------------------------------------


def test_no_other_machine_with_a_pull_name_asks_nothing(tmp_path: Path):
    context = _context({"workstation": [Q4], "server-a": [UNNAMED]})
    assert eligible_other_machines(context.document, "workstation") == frozenset()
    asker = _Recording({})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, context)

    assert KEY not in asker.asked
    assert lines == []


def test_a_missing_answer_chooses_nothing(tmp_path: Path):
    context = _context({"workstation": [Q4], "server-a": [Q8]})
    asker = _Recording({})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, context)

    assert KEY in asker.asked
    assert lines == []  # optional; missing -> nothing, silently (GF question 6)


def test_an_answer_outside_the_list_is_exit_2(tmp_path: Path):
    context = _context({"workstation": [Q4], "server-a": [Q8]})
    asker = _Recording({KEY: "server-a nonsense"})
    run, _lines = _run(tmp_path, asker)

    with pytest.raises(AnswerInvalidError):
        pull_other_step(run, context)


def test_an_import_only_run_still_offers_another_machines_package(tmp_path: Path):
    """This machine (`context.machine`) has no row of its own in the document at all -- an
    import-only run, `this-machine` not chosen -- but another chosen machine still has a package:
    `pull_other_step` asks and hands it over all the same (Codex r1-1; the extra condition the
    first round added -- "this machine has a row here too" -- is gone)."""
    doc = document_multi({"server-a": [Q8]})  # no block for "workstation" at all
    context = PullContext(doc, "workstation", [Q8], ambiguous_names([Q8]))
    asker = _Recording({KEY: f"server-a {NAME_Q8}"})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, context)

    assert any("modelroom pull" in line and NAME_Q8 in line for line in lines)


# --- the handover ---------------------------------------------------------------------------------


def test_the_handover_line_carries_name_and_weight_bytes(tmp_path: Path):
    context = _context({"workstation": [Q4], "server-a": [Q8]})
    asker = _Recording({KEY: f"server-a {NAME_Q8}"})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, context)

    expected_bytes = round(weights_bytes_of(Q8))
    handover = [line for line in lines if "modelroom pull" in line]
    assert handover and NAME_Q8 in handover[0] and f"--weights-bytes {expected_bytes}" in handover[0]
    assert any(f"ollama pull {NAME_Q8}" in line for line in lines)  # the line for a computer without modelroom


def test_the_handover_names_the_neutral_machine_only(tmp_path: Path):
    context = _context({"workstation": [Q4], "server-3f9a0c21": [Q8]})
    asker = _Recording({KEY: f"server-3f9a0c21 {NAME_Q8}"})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, context)

    handover = next(line for line in lines if "modelroom pull" in line)
    assert "on server-3f9a0c21" in handover
    assert "workstation" not in handover  # this machine's own name never stands in the handover
