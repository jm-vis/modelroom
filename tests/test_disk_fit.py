"""A download larger than the free disk is marked, the rank stays (AP9-RM U11, GF 2026-09-29, ADR 0012).

The fixtures are the ones of `tests/test_render.py`: a workstation of 11.94 GB graphics memory and two
builds of one model, `Q8_0` of 6 GiB and `Q4_K_M` of 5 GiB, both on the card -- `Q8_0` first (ADR
0009: the least compression that fits). A row needs its download plus the margin of 1 GiB of the
check before a pull (`room.pull_need_bytes`, AP9-RM U11b): 7 and 6 GiB. The free disk of the machine comes from a `LiveRoom`, as a
guided run reads it. Every test imports what AP9-RM U11 added inside itself, so the red run against
`1435359` fails test by test and not as one module that cannot be collected.
"""

from __future__ import annotations

from contextlib import contextmanager

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from modelroom.measurements import default_scenario
from modelroom.render import build_render_document
from modelroom.room import GIB, Disk, LiveRoom
from modelroom.screen import plain
from modelroom.views import RunView, document_markdown, terminal_blocks

from test_render import RENDERED_AT, _config, _hf_package, _profile, _ready, _snapshot

MACHINE = "workstation"


def _document():
    packages = [_hf_package(weights_bytes=5 * GIB, quantization="Q4_K_M"), _hf_package(weights_bytes=6 * GIB, quantization="Q8_0")]
    return build_render_document(_config(), _snapshot(packages), {MACHINE: _ready()}, {}, default_scenario(), RENDERED_AT)


def _disk(free_gib: float | None) -> Disk:
    return Disk(None, None, "two places Ollama may use") if free_gib is None else Disk(int(free_gib * GIB), "C:")


def _view(free_gib: float | None, machine: str = MACHINE) -> RunView:
    return RunView(install_for=MACHINE, live={machine: LiveRoom(disk=_disk(free_gib))})


def _lines(free_gib: float | None, machine: str = MACHINE) -> list:
    return terminal_blocks(_document(), MACHINE, view=_view(free_gib, machine))


def _text(free_gib: float | None, machine: str = MACHINE) -> list[str]:
    return [plain(line).strip() for line in _lines(free_gib, machine)]


def _rows(text: list[str]) -> list[str]:
    return [line for line in text if line[:2] in ("1 ", "2 ")]


# --- the table, the `disk` line and the install line ------------------------------------------------


def test_no_row_fits_the_disk():
    """4 GB free: both rows marked, the line says what to do, and no command is offered."""
    text = _text(4.0)
    rows = _rows(text)
    assert [row.split()[0] for row in rows] == ["1", "2"]  # the rank stays
    assert rows[0].endswith("6.0 GB ×") and rows[1].endswith("5.0 GB ×")
    assert any(line.startswith("disk      × #1–2 need more than the 4.0 GB free on C:") for line in text)
    assert "free space first, or pick a smaller model" in " ".join(text)
    assert "install   no package fits the 4.0 GB free on C:; free space first" in text
    assert not any("ollama pull" in line for line in text)


def test_some_rows_fit_the_disk():
    """6.5 GB free: `Q8_0` (#1, 6 + 1 GiB) is marked, the install line takes #2 (5 + 1) and says why."""
    text = _text(6.5)
    rows = _rows(text)
    assert rows[0].endswith("6.0 GB ×") and rows[1].endswith("5.0 GB")
    assert "disk      × #1 needs more than the 6.5 GB free on C:" in text
    assert "free space first" not in " ".join(line for line in text if line.startswith("disk"))
    install = next(index for index, line in enumerate(text) if line.startswith("install "))
    assert text[install] == "install   #2  ollama pull hf.co/packager/Nova-8B-GGUF:Q4_K_M"
    assert text[install + 1] == "the best row that also fits the disk; #1 needs more space"


def test_every_row_fits_the_disk():
    text = _text(100.0)
    assert not any("×" in line for line in text)
    assert not any(line.startswith("disk      ") for line in text)  # the `room` block's `disk` piece has no label
    assert "install   #1  ollama pull hf.co/packager/Nova-8B-GGUF:Q8_0" in text


def test_an_unknown_disk_marks_nothing():
    text = _text(None)
    assert not any("×" in line for line in text)
    assert "install   #1  ollama pull hf.co/packager/Nova-8B-GGUF:Q8_0" in text


def test_another_computer_changes_nothing():
    """The live disk belongs to another machine: this machine's table stays as it was."""
    text = _text(4.0, machine="server")
    assert not any("×" in line for line in text)
    assert "install   #1  ollama pull hf.co/packager/Nova-8B-GGUF:Q8_0" in text


def test_the_marked_row_carries_the_band_and_the_lines_their_red():
    """In color the whole row lies on the house red; without color the ` ×` says the same (NO_COLOR)."""
    lines = _lines(6.5)
    marked = [line for line in lines if plain(line).strip().startswith("1 ")]
    other = [line for line in lines if plain(line).strip().startswith("2 ")]
    assert [cls for cls, _text in marked[0]] == ["class:over-disk"]
    assert [cls for cls, _text in other[0]] == [""]
    disk = next(line for line in lines if plain(line).strip().startswith("disk      ×"))
    assert disk[-1][0] == "class:warn"
    none_fits = next(line for line in _lines(4.0) if "no package fits" in plain(line))
    assert none_fits[-1][0] == "class:warn"


def test_the_colors_are_the_house_red():
    from modelroom.intro import RED_BAND, RED_TEXT, style_rules

    rules = dict(style_rules())
    assert RED_TEXT == "#f07a72"
    assert rules["warn"] == f"fg:{RED_TEXT}"
    assert rules["over-disk"] == f"bg:{RED_BAND} fg:#ffffff"


def test_a_marked_table_stays_within_100_columns():
    for free in (4.0, 6.5):
        for line in _lines(free):
            text = plain(line)
            if "ollama pull" not in text:
                assert len(text) <= 100, text


def test_an_ascii_console_marks_with_an_x():
    from modelroom.intro import ASCII_GLYPHS, UNICODE_GLYPHS

    assert (UNICODE_GLYPHS.cross, ASCII_GLYPHS.cross) == ("×", "x")


# --- the Markdown file --------------------------------------------------------------------------


def test_the_markdown_file_marks_the_download_and_says_the_disk_line():
    markdown = document_markdown(_document(), disks={MACHINE: _disk(6.5)})
    row = next(line for line in markdown.splitlines() if line.startswith("| 1 |"))
    assert "| 6.00 × |" in row
    assert "× #1 needs more than the 6.5 GB free on C:" in markdown
    plain_markdown = document_markdown(_document())
    assert "×" not in plain_markdown


# --- the pull list -------------------------------------------------------------------------------


def _pull_rows(free_gib: float | None):
    from modelroom.pull_choice import pull_rows

    return pull_rows(_document(), MACHINE, frozenset(), "·", _disk(free_gib))


def test_a_row_too_large_for_the_disk_is_locked_with_its_reason():
    rows = _pull_rows(6.5)
    assert (rows[0].locked, rows[0].over_disk) == ("needs 6.0 GB plus 1 GB room, 6.5 GB free", True)
    assert (rows[1].locked, rows[1].over_disk) == (None, False)
    assert all(row.locked is None for row in _pull_rows(None))


def test_the_pull_list_draws_a_too_large_row_red_and_not_selectable():
    from modelroom.guided_install import pull_choices

    choices = pull_choices(_pull_rows(6.5), set())
    assert choices[0].label == "#1  packager · Q8_0  6.0 GB  needs 6.0 GB plus 1 GB room, 6.5 GB free"
    assert (choices[0].disabled, choices[0].warn) == ("needs 6.0 GB plus 1 GB room, 6.5 GB free", True)
    assert choices[1].disabled is None and not choices[1].warn


@contextmanager
def _enter():
    from modelroom.dialog import TerminalAsker

    with create_pipe_input() as pipe:
        pipe.send_text("\r")
        yield TerminalAsker(input=pipe, output=DummyOutput())


def test_the_pointer_skips_the_rows_too_large_for_the_disk():
    from modelroom.guided_install import pull_choices

    with _enter() as asker:
        taken = asker.select("pull_package", "Which package should be pulled into Ollama?", pull_choices(_pull_rows(6.5), set()))
    assert taken == "hf.co/packager/Nova-8B-GGUF:Q4_K_M"


def test_the_pointer_rests_on_none_where_no_row_fits():
    from modelroom.guided_install import pull_choices

    with _enter() as asker:
        taken = asker.select("pull_package", "Which package should be pulled into Ollama?", pull_choices(_pull_rows(4.0), set()))
    assert taken == "none"


def test_a_download_half_a_gib_under_the_free_disk_is_marked():
    """The margin of the check before a pull counts: 6 GiB with 6.5 GiB free is no fit for the disk."""
    from modelroom.disk_fit import over_disk
    from modelroom.room import pull_need_bytes

    assert pull_need_bytes(6 * GIB) == 7 * GIB
    assert over_disk(6.0, _disk(6.5)) and not over_disk(5.0, _disk(6.5)) and not over_disk(5.5, _disk(6.5))
    assert _rows(_text(6.5))[0].endswith("6.0 GB ×")
