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

from modelroom.daemon import TAGS_PATH, VERSION_PATH, DaemonError, FixtureDaemon
from modelroom.http import Response
from modelroom.intro import (
    ASCII_GLYPHS,
    STEP_COUNT,
    UNICODE_GLYPHS,
    Fact,
    Intro,
    collect_intro,
    done_line,
    glyphs,
    hardware_words,
    intro_lines,
    print_intro,
    source_url,
    step_head,
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


def _daemon(version: str = "0.34.2", models: int = 19) -> FixtureDaemon:
    tags = {"models": [{"name": f"model-{number}", "digest": "0" * 64, "size": 1} for number in range(models)]}
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
    assert daemon.note == "reachable, 19 models installed"


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


# --- the step heads --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [(1, "Step 1 of 5  Configuration"), (3, "Step 3 of 5  Context"), (5, "Step 5 of 5  Results")],
)
def test_every_step_has_a_head_with_its_number_and_its_name(number, expected):
    assert step_head(number) == expected


def test_there_are_five_steps():
    assert STEP_COUNT == 5


def test_a_finished_step_leaves_one_line_with_a_check_mark(monkeypatch):
    monkeypatch.setattr("sys.stdout", _stream("utf-8"))

    assert done_line(2, "7 packages fetched").startswith(f"{UNICODE_GLYPHS.check} Packages")
    assert done_line(2, "7 packages fetched").endswith("7 packages fetched")


def test_a_finished_step_says_ok_where_a_check_mark_cannot_be_printed(monkeypatch):
    monkeypatch.setattr("sys.stdout", _stream("cp1252"))

    assert done_line(2, "done").startswith("ok Packages")


def test_a_fact_is_shown_as_label_value_and_note():
    intro = Intro(version="0.1.0", source=None, facts=(Fact("folder", "C:/results", "empty"),))

    line = _text(intro, _stream("utf-8")).splitlines()[4]

    assert line.split() == ["folder", "C:/results", "empty"]


def test_a_value_wider_than_its_column_does_not_grow_into_the_note():
    """A results folder of 120 characters left `...resultsa configuration` on one line."""
    long_path = "C:/" + "folder/" * 20 + "results"
    intro = Intro(version="0.1.0", source=None, facts=(Fact("folder", long_path, "a configuration"),))

    line = _text(intro, _stream("utf-8")).splitlines()[4]

    assert f"{long_path}  a configuration" in line
