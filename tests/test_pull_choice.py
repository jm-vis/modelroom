"""Tests for modelroom.pull_choice and the list of step 5 (AP9-UX2 U4, U6; GF F4, F7).

Synthetic, in memory (`tests/fixture_rows.py`): packages of `acme/Nova-7B` and documents whose rows
name them. The ranking rule is not touched -- these rows stand in the order the document gives them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from modelroom.daemon import PULL_PATH, FixtureDaemon
from modelroom.dialog import AnswerInvalidError, FileAsker, TerminalAsker
from modelroom.guided_install import PullContext, pull_step
from modelroom.pull_choice import (
    NO_PULL_NAME,
    NOT_UNIQUE,
    ambiguous_names,
    default_row,
    pull_collisions,
    pull_rows,
    repo_lines,
)

from fixture_rows import GIB, document, fillers, package
from fixture_support import json_response

NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
UNSLOTH = "unsloth/Nova-7B-GGUF"
Q4 = package(UNSLOTH, "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
Q5 = package(UNSLOTH, "Nova-7B-Q5_K_M.gguf", "Q5_K_M")
Q8 = package(UNSLOTH, "Nova-7B-Q8_0.gguf", "Q8_0")
# The same quantization as a single file and in a folder: two identities, one pull name.
BF16_FILE = package(UNSLOTH, "Nova-7B-BF16.gguf", "BF16", 14 * GIB)
BF16_FOLDER = package(UNSLOTH, "BF16/Nova-7B-BF16.gguf", "BF16", 7 * GIB)
NAME = "hf.co/unsloth/Nova-7B-GGUF"


class _Recording(FileAsker):
    """An answer file that keeps the choices of every list it was asked."""

    def __init__(self, answers: dict) -> None:
        super().__init__(answers)
        self.lists: dict[str, list] = {}

    def select(self, key, question, choices, instruction=None):
        self.lists[key] = list(choices)
        return super().select(key, question, choices, instruction)


def _run(tmp_path: Path, asker, daemon=None):
    from fixture_room import fixed_room
    from fixture_support import offline_servers, windows_probes
    from modelroom.catalog import load_catalog
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
        daemon=daemon if daemon is not None else _daemon(),
        servers=ServerRun(offline_servers()),
        out=lines.append,
        screen=Screen(lines.append, colored=False),
        room=fixed_room(),
    )
    return run, lines


def _daemon():
    from modelroom.daemon import TAGS_PATH

    return FixtureDaemon({("GET", TAGS_PATH): json_response("ollama_tags_local_loadtest.json")})


def _context(packages, shown=None) -> PullContext:
    shown = packages if shown is None else shown
    return PullContext(document(shown), "workstation", packages, ambiguous_names(packages))


# --- collisions (U4) --------------------------------------------------------------------------------


def test_pull_collisions_cover_every_active_complete_package_before_the_eligibility_filter():
    hidden = package(UNSLOTH, "BF16/Nova-7B-BF16.gguf", "BF16", provenance="unresolved")
    inactive = package(UNSLOTH, "sub/Nova-7B-Q4_K_M.gguf", "Q4_K_M", active=False)

    collisions = pull_collisions([Q4, BF16_FILE, hidden, inactive])

    assert list(collisions) == [f"{NAME}:BF16"]  # the unresolved one counts, the inactive one does not
    assert len(collisions[f"{NAME}:BF16"]) == 2


def test_rows_without_duplicates_add_no_repo_line():
    assert repo_lines(document([Q8, Q5, Q4]).machines[0]) == []
    assert ambiguous_names([Q8, Q5, Q4]) == frozenset()


def test_a_duplicate_at_place_11_is_not_named():
    """Only the rows shown count: the 11th package of the same `packager · quant` is not in the table."""
    other = package("unsloth/Nova-7B-Instruct-GGUF", "Nova-7B-Instruct-Q4_K_M.gguf", "Q4_K_M")
    shown = [Q4, *fillers(9, avoid=("Q4_K_M",))]

    assert repo_lines(document(shown, total=11).machines[0]) == []
    assert repo_lines(document([*shown[:9], other]).machines[0]) == [
        f"#1  {UNSLOTH}",
        "#10  unsloth/Nova-7B-Instruct-GGUF",
    ]


def test_one_row_per_model_names_duplicates_across_models_only_when_shown():
    luna = package("unsloth/Luna-7B-GGUF", "Luna-7B-Q4_K_M.gguf", "Q4_K_M", base_model_hf_repo="acme/Nova-7B")

    assert repo_lines(document([Q4, luna]).machines[0]) == [f"#1  {UNSLOTH}", "#2  unsloth/Luna-7B-GGUF"]
    assert repo_lines(document([Q4, Q5]).machines[0]) == []


def test_an_appended_measured_row_takes_part_in_the_naming():
    """A measured row the rule appends behind the ten stands in the table, so it is named too."""
    other = package("unsloth/Nova-7B-Instruct-GGUF", "Nova-7B-Instruct-Q8_0.gguf", "Q8_0")
    doc = document([Q8, *fillers(9, avoid=("Q8_0",)), other], total=14)

    assert repo_lines(doc.machines[0]) == [f"#1  {UNSLOTH}", "#11  unsloth/Nova-7B-Instruct-GGUF"]


# --- the list (U6) ------------------------------------------------------------------------------------


def test_every_shown_row_is_a_row_of_the_list():
    rows = pull_rows(document([Q8, Q5, Q4]), "workstation")

    assert [(row.rank, row.name) for row in rows] == [(1, f"{NAME}:Q8_0"), (2, f"{NAME}:Q5_K_M"), (3, f"{NAME}:Q4_K_M")]
    assert pull_rows(document([Q8]), "inference-server") == []


def test_rows_without_a_pull_name_are_locked():
    unnamed = package(UNSLOTH, "Nova-7B-mystery.gguf", None)
    rows = pull_rows(document([unnamed, Q4]), "workstation")

    assert [row.locked for row in rows] == [NO_PULL_NAME, None]


def test_every_row_of_an_ambiguous_pull_name_is_locked():
    doc = document([BF16_FILE, Q4, BF16_FOLDER])
    rows = pull_rows(doc, "workstation", ambiguous_names([BF16_FILE, Q4, BF16_FOLDER]))

    assert [row.locked for row in rows] == [NOT_UNIQUE, None, NOT_UNIQUE]


def test_a_missing_answer_takes_row_one_only_when_it_has_a_pull_name():
    unnamed = package(UNSLOTH, "Nova-7B-mystery.gguf", None)

    assert default_row(pull_rows(document([Q8, Q4]), "workstation")).rank == 1
    assert default_row(pull_rows(document([unnamed, Q4]), "workstation")) is None


def test_a_missing_answer_with_an_ambiguous_row_one_pulls_nothing(tmp_path: Path):
    packages = [BF16_FILE, BF16_FOLDER, Q4]
    asker = _Recording({"pull": True})
    daemon = _daemon()
    run, lines = _run(tmp_path, asker, daemon)

    pull_step(run, _context(packages), True)

    assert "pull" not in asker.asked
    assert not any(call[1] == PULL_PATH for call in daemon.calls)


def test_the_pointer_starts_on_the_first_choosable_row_and_none_is_last(tmp_path: Path):
    unnamed = package(UNSLOTH, "Nova-7B-mystery.gguf", None)
    packages = [unnamed, Q8, Q4]
    asker = _Recording({"pull_package": "none"})
    run, _lines = _run(tmp_path, asker)

    pull_step(run, _context(packages), True)

    choices = asker.lists["pull_package"]
    assert not any(choice.checked for choice in choices)  # nothing marked
    assert choices[0].disabled == NO_PULL_NAME and choices[1].disabled is None
    assert (choices[-1].value, choices[-1].label) == ("none", "none, do not pull")
    assert choices[1].label == f"#2  unsloth · Q8_0  5.0 GB  {NAME}:Q8_0"
    # At a terminal, Enter alone takes the first row that can be chosen, never a locked one.
    with create_pipe_input() as pipe:
        pipe.send_text("\r")
        terminal = TerminalAsker(input=pipe, output=DummyOutput())
        assert terminal.select("pull_package", "Which package should be pulled into Ollama?", choices) == f"{NAME}:Q8_0"


def test_an_empty_list_asks_nothing(tmp_path: Path):
    unnamed = package(UNSLOTH, "Nova-7B-mystery.gguf", None)
    asker = _Recording({})
    run, lines = _run(tmp_path, asker)

    pull_step(run, _context([unnamed]), True)

    assert asker.asked == [] and lines == []


def test_an_approved_row_with_a_hidden_unresolved_counterpart_is_not_installable(tmp_path: Path):
    """The approved single file is shown; its folder form, `unresolved`, is hidden -- one pull name."""
    from modelroom.views import terminal_lines

    approved = package(UNSLOTH, "Nova-7B-BF16.gguf", "BF16", 14 * GIB)
    hidden = package(UNSLOTH, "BF16/Nova-7B-BF16.gguf", "BF16", 7 * GIB, provenance="unresolved")
    shown = [approved, Q4]
    context = PullContext(document(shown), "workstation", [approved, hidden, Q4], ambiguous_names([approved, hidden, Q4]))
    asker = _Recording({"pull_package": f"{NAME}:BF16", "pull": True})
    daemon = _daemon()
    run, _lines = _run(tmp_path, asker, daemon)

    lines = terminal_lines(context.document, "workstation", ambiguous=context.ambiguous)
    assert not any(line.strip().startswith("install") and "ollama pull" in line for line in lines)
    assert any("#1 has no unique pull name: 2 files answer to it" in line for line in lines)
    with pytest.raises(AnswerInvalidError):  # the locked row is no answer an answer file can give
        pull_step(run, context, True)
    assert not any(call[1] == PULL_PATH for call in daemon.calls)
