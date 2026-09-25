"""Tests for modelroom.intro: the start screen, the step heads, the glyphs and the color rule.

The two decisions of this module are about the real stream a run writes to, so they are made
against real streams here -- an `io.TextIOWrapper` with an encoding of its own, and a file that
is not a terminal. Nothing is replaced.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import modelroom
from modelroom.daemon import TAGS_PATH, VERSION_PATH, DaemonError, FixtureDaemon
from modelroom.http import Response
from modelroom.intro import (
    ASCII_GLYPHS,
    STEP_COUNT,
    UNICODE_GLYPHS,
    Fact,
    Intro,
    collect_intro,
    glyphs,
    hardware_words,
    intro_lines,
    print_intro,
    source_url,
    style_rules,
    use_color,
)

from fixture_support import v2_profile, windows_probes


def _stream(encoding: str, *, tty: bool = False):
    """A real text stream with an encoding of its own, as a console would have."""

    class _Buffer(io.BytesIO):
        def isatty(self) -> bool:
            return tty

    wrapper = io.TextIOWrapper(_Buffer(), encoding=encoding, errors="strict")
    return wrapper


def _daemon(version: str = "0.34.2", models: int = 19, names: list[str] | None = None) -> FixtureDaemon:
    listed = names if names is not None else [f"model-{number}:9b" for number in range(models)]
    tags = {"models": [{"name": name, "digest": "0" * 64, "size": 1} for name in listed]}
    return FixtureDaemon(
        {
            ("GET", VERSION_PATH): Response(status=200, body=json.dumps({"version": version}).encode("utf-8")),
            ("GET", TAGS_PATH): Response(status=200, body=json.dumps(tags).encode("utf-8")),
        }
    )


def _intro(daemon=None, config_file: Path | None = None, pointer: Path | None = None) -> Intro:
    return collect_intro(
        daemon if daemon is not None else _daemon(),
        windows_probes(),
        pointer if pointer is not None else Path("nowhere") / "guided.json",
        config_file,
    )


def _text(intro: Intro, stream=None) -> str:
    lines: list[str] = []
    print_intro(intro, lines.append, stream)
    return "\n".join(lines)


# --- the glyph set ------------------------------------------------------------------------------


def test_a_utf_8_stream_gets_the_drawn_glyphs():
    assert glyphs(_stream("utf-8")) is UNICODE_GLYPHS


def test_a_cp1252_console_gets_ascii_instead():
    """A block character would end the run with a UnicodeEncodeError instead of a dialog."""
    assert glyphs(_stream("cp1252")) is ASCII_GLYPHS


def test_a_stream_without_an_encoding_gets_ascii():
    assert glyphs(object()) is ASCII_GLYPHS


def test_every_ascii_glyph_really_is_ascii():
    for value in vars(ASCII_GLYPHS).values():
        assert value.isascii()


# --- the color rule -----------------------------------------------------------------------------


def test_color_only_at_a_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert use_color(_stream("utf-8", tty=True)) is True
    assert use_color(_stream("utf-8", tty=False)) is False


def test_no_color_in_the_environment_turns_color_off(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert use_color(_stream("utf-8", tty=True)) is False


def test_without_color_the_same_lines_go_through_out(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    intro = _intro()

    printed = _text(intro, _stream("utf-8", tty=True))

    assert "ModelRoom" in printed
    assert printed.count("\n") + 1 == len(intro_lines(intro, _stream("utf-8")))


def test_a_caller_that_says_no_color_gets_none_even_at_a_terminal(monkeypatch):
    """`modelroom --answers <file>` is read from a log; its output must not depend on the window."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    lines: list[str] = []

    print_intro(_intro(), lines.append, _stream("utf-8", tty=True), colored=False)

    assert any("ModelRoom" in line for line in lines)
    assert not any("\x1b" in line for line in lines)


def test_at_a_terminal_the_start_screen_is_on_screen_either_way(monkeypatch, capsys):
    """At a terminal prompt_toolkit prints it in color; where it cannot, the same lines still go out.

    Which of the two happens depends on the console this test itself runs in, and that is the
    point: the start screen is the first thing a run prints, and it is never a traceback.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    lines: list[str] = []

    print_intro(_intro(), lines.append, _stream("utf-8", tty=True))

    assert "ModelRoom" in "\n".join(lines) + capsys.readouterr().out


# --- what the start screen says -------------------------------------------------------------------


def test_the_mark_the_tagline_and_the_version_stand_next_to_the_pictogram():
    printed = _text(_intro(), _stream("utf-8"))
    first_three = printed.splitlines()[:3]

    assert UNICODE_GLYPHS.full in first_three[0]
    assert first_three[0].endswith("ModelRoom")
    assert first_three[1].endswith("Which local model packages fit your machine.")
    assert "modelroom " in first_three[2]


def test_the_version_line_names_the_repository_of_the_package_metadata():
    assert source_url() == "github.com/jm-vis/modelroom"
    assert source_url() in _text(_intro(), _stream("utf-8"))


def test_the_three_facts_are_folder_daemon_and_machine():
    assert [fact.label for fact in _intro().facts] == ["folder", "daemon", "machine"]


def test_a_run_with_no_folder_yet_says_the_first_question_asks():
    folder = _intro().facts[0]

    assert folder.value == "not chosen yet"
    assert "first question" in folder.note


def test_a_reachable_daemon_is_named_with_its_version_and_its_number_of_models():
    daemon = _intro().facts[1]

    assert daemon.value == "Ollama 0.34.2"
    assert daemon.note == "reachable, 19 models"


def test_a_daemon_with_cloud_entries_says_how_many_of_them_are_local():
    """`19 models installed` counted twelve cloud entries of the Ollama app as installed packages
    (test round, 2026-09-25). A cloud entry is the daemon's, not this machine's."""
    names = ["granite4.2:8b", "qwen3.5:9b", "gpt-oss:120b-cloud", "glm-5.3-flash:cloud"]

    daemon = _intro(_daemon(names=names)).facts[1]

    assert daemon.note == "reachable, 4 models, 2 of them local"


def test_a_daemon_without_a_cloud_entry_names_one_number_only():
    daemon = _intro(_daemon(names=["granite4.2:8b", "qwen3.5:9b"])).facts[1]

    assert daemon.note == "reachable, 2 models"


def test_a_cloud_entry_is_recognized_by_its_tag_alone():
    from modelroom.intro import is_cloud_name

    assert is_cloud_name("glm-5.3-flash:cloud") is True
    assert is_cloud_name("gpt-oss:120b-cloud") is True
    assert is_cloud_name("granite4.2:8b") is False
    assert is_cloud_name("hf.co/unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL") is False
    assert is_cloud_name("cloud") is False


def test_a_daemon_that_does_not_answer_is_not_reachable():
    unreachable = FixtureDaemon({}, unreachable="connection refused")

    daemon = _intro(unreachable).facts[1]

    assert daemon.value == "Ollama"
    assert daemon.note == "not reachable"


def test_a_daemon_whose_model_list_does_not_arrive_still_shows_its_version():
    half = FixtureDaemon(
        {
            ("GET", VERSION_PATH): Response(status=200, body=b'{"version": "0.34.2"}'),
            ("GET", TAGS_PATH): DaemonError("the daemon went away"),
        }
    )

    daemon = _intro(half).facts[1]

    assert daemon.value == "Ollama 0.34.2"
    assert "did not arrive" in daemon.note


def test_a_machine_with_no_profile_in_this_folder_says_so():
    machine = _intro().facts[2]

    assert machine.value == "workstation"
    assert machine.note == "not measured in this results folder yet"


def test_the_hardware_of_a_measured_machine_is_said_in_plain_words():
    assert hardware_words(v2_profile()) == "one graphics card, 12 GB, 127 GB memory"


def test_a_machine_without_a_graphics_card_says_that_instead():
    profile = v2_profile().model_copy(update={"gpu_state": "none", "vram_gib": 0.0})

    assert hardware_words(profile).startswith("no graphics card")


def test_a_graphics_card_that_was_not_measured_is_not_called_a_number():
    profile = v2_profile().model_copy(update={"gpu_state": "present_unmeasured", "vram_gib": None})

    assert "not measured" in hardware_words(profile)


def test_the_closing_lines_name_the_five_steps_and_the_way_out():
    printed = _text(_intro(), _stream("utf-8"))

    assert "Five steps: configuration, packages, context, measurement, results." in printed
    assert "Files are written as the run goes on." in printed
    assert "Esc leaves a list, Ctrl-C leaves at any point." in printed


def test_the_start_screen_promises_nothing_it_does_not_keep():
    """Step 1 writes a configuration once the folder is known, and a schema-1 file is migrated on
    the way in -- so the line may not say that nothing is written before the end."""
    printed = _text(_intro(), _stream("utf-8"))

    assert "before you say yes" not in printed


# --- the pictogram: shading without color, mark E with it -------------------------------------------


def _mark(colored: bool, encoding: str = "utf-8"):
    """The lines of the mark alone: everything above the blank line that follows it."""
    lines = intro_lines(_intro(), _stream(encoding), colored=colored)
    return lines[: lines.index([])]


def test_with_color_the_mark_is_mark_e_two_lines_per_row_of_cells():
    """Squares of 24 pixels with a gap of 8 in both directions (E, decided 2026-09-24)."""
    lines = _mark(colored=True)

    assert len(lines) == 6
    drawn = ["".join(text for _class, text in line) for line in lines]
    assert drawn[0].startswith(f"{UNICODE_GLYPHS.lower} {UNICODE_GLYPHS.lower}")
    assert drawn[1].startswith(f"{UNICODE_GLYPHS.block} {UNICODE_GLYPHS.block}")
    assert UNICODE_GLYPHS.muted not in "".join(drawn)
    assert UNICODE_GLYPHS.empty not in "".join(drawn)


def test_with_color_the_name_the_tagline_and_the_version_stand_on_lines_two_to_four():
    lines = _mark(colored=True)

    texts = ["".join(text for _class, text in line) for line in lines]
    assert texts[1].endswith("ModelRoom")
    assert texts[2].endswith("Which local model packages fit your machine.")
    assert "modelroom " in texts[3]
    assert [index for index, text in enumerate(texts) if "   " in text.rstrip()] == [1, 2, 3]


def test_the_three_cell_colors_are_white_and_two_blues():
    classes = {name: style for name, style in style_rules()}

    assert classes["mark"].startswith("fg:#ffffff")
    assert classes["mark-muted"] == "fg:#8cacc9"
    assert classes["mark-empty"] == "fg:#3c5470"


def test_with_color_the_three_cell_kinds_keep_their_own_class():
    lines = _mark(colored=True)

    for line in lines[:2]:
        # Only the cells: the text beside the mark begins after the three-space gap.
        cells = line[: line.index(("", "   "))] if ("", "   ") in line else line
        assert [name for name, _text in cells if name] == [
            "class:mark",
            "class:mark",
            "class:mark-muted",
            "class:mark",
        ]


def test_without_color_the_shading_glyphs_stay_as_they_are():
    printed = _text(_intro(), _stream("utf-8"))

    assert len(_mark(colored=False)) == 3
    assert UNICODE_GLYPHS.muted in printed
    assert UNICODE_GLYPHS.empty in printed
    assert UNICODE_GLYPHS.lower not in printed


def test_a_console_without_the_block_glyphs_keeps_its_ascii_mark_in_color():
    drawn = "".join(text for line in _mark(colored=True, encoding="cp1252") for _class, text in line)

    assert ASCII_GLYPHS.block in drawn
    assert drawn.count(ASCII_GLYPHS.block) == 12
    assert drawn.count(ASCII_GLYPHS.lower) == 12


def test_the_skip_glyph_is_an_en_dash_and_a_hyphen_in_ascii():
    """The mark of a step that did nothing; a check mark cannot say "there was nothing to do"."""
    assert UNICODE_GLYPHS.skip == "–"
    assert ASCII_GLYPHS.skip == "-"


# --- the steps ------------------------------------------------------------------------------------


def test_there_are_five_steps():
    assert STEP_COUNT == 5


def test_a_fact_is_shown_as_label_value_and_note():
    intro = Intro(version=modelroom.__version__, source=None, facts=(Fact("folder", "C:/results", "empty"),))

    line = _text(intro, _stream("utf-8")).splitlines()[4]

    assert line.split() == ["folder", "C:/results", "empty"]


def test_a_value_wider_than_its_column_does_not_grow_into_the_note():
    """A results folder of 120 characters left `...resultsa configuration` on one line."""
    long_path = "C:/" + "folder/" * 20 + "results"
    intro = Intro(version=modelroom.__version__, source=None, facts=(Fact("folder", long_path, "a configuration"),))

    line = _text(intro, _stream("utf-8")).splitlines()[4]

    assert f"{long_path}  a configuration" in line
