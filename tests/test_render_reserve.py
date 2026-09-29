"""The reserve of the system memory in the render document and its views (AP9-RM RM-2, ADR 0012).

The document carries the reserve the fit took and where it came from; the Markdown cell says
`(computed)`, the `memory` line names the budget of system memory rows too, the key names the
rule, and a pool of 0 or less is said in words -- never a memory below zero. The fixtures are the
ones of `tests/test_render.py`.
"""

from __future__ import annotations

import json
import re

from modelroom.config import MachineConfig
from modelroom.render import LEGACY_FIT_REASON, NO_PROFILE_REASON, MachineProfile, build_render_document, document_json
from modelroom.measurements import default_scenario
from modelroom.ranking_words import EACH_MACHINE_PIECE, EACH_MACHINE_SENTENCE
from modelroom.room_head import NOTE_WIDTH
from modelroom.screen import FIT_HINT
from modelroom.views import document_markdown, document_terminal

from test_render import GIB, PROFILE_ID, RENDERED_AT, _config, _hf_package, _profile, _ready, _snapshot

# A memory value below zero: a minus that follows no letter or digit (a date's `2026-09` does).
_NEGATIVE = re.compile(r"(?<![0-9A-Za-z])-\d")
HINT = "memory reserve 8.0 GB is set in modelroom.toml; computed from this machine it would be 16.0 GB"


def _machine(reserve: float | None, profile: str | None = PROFILE_ID) -> dict:
    entry = {"reserve_vram_gib": 1.0, "writer": True, "profile": profile}
    return entry if reserve is None else {"reserve_ram_gib": reserve, **entry}


def _document(reserve: float | None, *, ram_gib: float = 127.46, vram_gib: float = 11.94, weights_gib: float = 5.0, found=None):
    config = _config({"workstation": _machine(reserve)})
    snapshot = _snapshot([_hf_package(weights_bytes=int(weights_gib * GIB))])
    found = found or _ready(_profile(vram_gib=vram_gib, ram_gib=ram_gib))
    return build_render_document(config, snapshot, {"workstation": found}, {}, default_scenario(), RENDERED_AT)


def _no_fit_note(document) -> str:
    [entry] = document.machines[0].no_fit
    return entry.note.text


def _assert_no_negative(document) -> None:
    for text in (_no_fit_note(document), document_terminal(document), document_markdown(document)):
        assert not _NEGATIVE.search(text), _NEGATIVE.search(text)


# --- the document ------------------------------------------------------------------------------


def test_the_document_carries_the_reserve_and_its_origin():
    block = _document(None).machines[0]
    assert (block.reserve_ram_gib, block.reserve_ram_origin) == (16.0, "computed")
    assert json.loads(json.dumps(document_json(_document(None))))["machines"][0]["reserve_ram_origin"] == "computed"
    assert _document(8.0).machines[0].reserve_ram_origin == "entered"


def test_the_markdown_cell_says_computed():
    assert "| 16.00 (computed) | 1.00 |" in document_markdown(_document(None))
    assert "| 8.00 | 1.00 |" in document_markdown(_document(8.0))


def test_an_entered_zero_reserve_shows_0_00_entered():
    document = _document(0.0)
    assert (document.machines[0].reserve_ram_gib, document.machines[0].reserve_ram_origin) == (0.0, "entered")
    assert "| 0.00 | 1.00 |" in document_markdown(document)


def test_a_block_without_profile_keeps_its_entered_reserve():
    for status, reason in (("no_profile", NO_PROFILE_REASON), ("legacy", LEGACY_FIT_REASON)):
        found = MachineProfile(status=status, label="workstation", profile=None, reason=reason)
        block = _document(1.5, found=found).machines[0]
        assert (block.status, block.reserve_ram_gib, block.reserve_ram_origin) == (status, 1.5, "entered")


def test_a_block_without_profile_and_without_number_is_unknown():
    found = MachineProfile(status="no_profile", label="workstation", profile=None, reason=NO_PROFILE_REASON)
    document = _document(None, found=found)
    assert (document.machines[0].reserve_ram_gib, document.machines[0].reserve_ram_origin) == (0.0, "unknown")
    row = next(line for line in document_markdown(document).splitlines() if line.startswith("| workstation |"))
    assert "| – | 1.00 |" in row


# --- a pool of 0 or less, in words ---------------------------------------------------------------


def test_a_negative_pool_is_said_in_words():
    """The tester's machine with the entered 8.0 of 0.1.0: 3.81 - 8.0 is no memory at all."""
    document = _document(8.0, ram_gib=3.81, vram_gib=0.0)
    assert "nothing is left after the reserve of 8.0 GiB (3.8 GiB in all)" in _no_fit_note(document)
    _assert_no_negative(document)


def test_a_pool_of_exactly_zero_is_said_in_words():
    document = _document(8.0, ram_gib=8.0, vram_gib=0.0)
    assert "nothing is left after the reserve of 8.0 GiB (8.0 GiB in all)" in _no_fit_note(document)
    _assert_no_negative(document)


def test_memory_below_the_minimum_reserve_is_said_in_words():
    """1.5 GiB keeps the minimum of 2.0: nothing is left."""
    document = _document(None, ram_gib=1.5, vram_gib=0.0)
    assert "nothing is left after the reserve of 2.0 GiB (1.5 GiB in all)" in _no_fit_note(document)
    _assert_no_negative(document)


# --- the views say the reserve -------------------------------------------------------------------


def test_system_memory_rows_name_their_budget():
    """No card: the rows run in system memory, 127.46 - 16.0 = 111.46 GB."""
    text = " ".join(line.strip() for line in document_terminal(_document(None, vram_gib=0.0)).splitlines())
    assert "#1 needs system memory, no graphics card, budget 111.5 GB" in text
    assert "(memory minus reserve, from the profile)" in text


def test_the_key_names_the_computed_reserve():
    piece = "system memory keeps a quarter as reserve, 2 to 16 GB, unless modelroom.toml sets it"
    shared = "one memory shared by system and card counts as gpu, a quarter kept, no upper limit"
    assert piece in FIT_HINT and shared in FIT_HINT
    assert all(len(text) <= NOTE_WIDTH for text in FIT_HINT)
    assert piece in document_markdown(_document(None))


def test_an_old_default_reserve_gets_a_hint_under_its_table():
    """GF question 3 (a): an entered 8.0 on a machine that would compute 16.0 is said, not changed."""
    document = _document(8.0)
    assert document.machines[0].reserve_ram_gib == 8.0
    terminal = document_terminal(document)
    assert "memory reserve 8.0 GB is set in modelroom.toml;" in terminal
    assert "computed from this machine it would be 16.0 GB" in terminal
    assert f"{HINT[0].upper()}{HINT[1:]}." in document_markdown(document)
    for quiet in (_document(8.0, ram_gib=32.0), _document(None), _document(1.5)):
        assert "is set in modelroom.toml" not in document_terminal(quiet) + document_markdown(quiet)


# --- GF question 5 (a): each table is ranked for its own machine -----------------------------------


def _two_machines():
    config = _config({"workstation": _machine(8.0), "server": _machine(8.0, "b1b2b3b4b5b6b7b8")})
    server = _profile(vram_gib=22.5, ram_gib=512.0, profile_id="b1b2b3b4b5b6b7b8")
    found = {"workstation": _ready(), "server": _ready(server)}
    return build_render_document(config, _snapshot([_hf_package()]), found, {}, default_scenario(), RENDERED_AT)


def test_two_machines_say_each_table_is_ranked_for_its_machine():
    document = _two_machines()
    shown = [line for line in document_terminal(document).splitlines() if line.strip().startswith("shown")]
    assert len(shown) == 2 and all(EACH_MACHINE_PIECE in line for line in shown)
    assert document_markdown(document).count(EACH_MACHINE_SENTENCE) == 1


def test_one_machine_says_nothing_more():
    document = _document(None)
    assert EACH_MACHINE_PIECE not in document_terminal(document)
    assert EACH_MACHINE_SENTENCE not in document_markdown(document)
