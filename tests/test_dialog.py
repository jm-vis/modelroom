"""Tests for modelroom.dialog and modelroom.answers: asking, and answering from a file.

The terminal asker is tested against a real `prompt_toolkit` pipe input -- the keys a user
would press travel through prompt_toolkit and questionary unchanged; nothing is replaced.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from modelroom.answers import ANSWERS_SCHEMA_VERSION, AnswerFileError, read_answers
from modelroom.contracts import SchemaVersionError
from modelroom.dialog import (
    AnswerInvalidError,
    AnswerMissingError,
    Canceled,
    Choice,
    FileAsker,
    TerminalAsker,
    is_interactive,
    selectable,
)

DOWN = "\x1b[B"
ENTER = "\r"
CTRL_C = "\x03"
CTRL_D = "\x04"


@contextmanager
def _asker(keys: str, *, closed: bool = False):
    """A real terminal asker whose keys come from a prompt_toolkit pipe, not from a keyboard.

    `closed` closes the pipe instead of sending keys: that is what an end of input looks like to
    prompt_toolkit, and it is how a run without anything left to read really ends. (`Ctrl-D` as a
    character only ends a text prompt; a selection list does not bind it.)
    """
    with create_pipe_input() as pipe:
        if closed:
            pipe.close()
        else:
            pipe.send_text(keys)
        yield TerminalAsker(input=pipe, output=DummyOutput())


CHOICES = [
    Choice("this-machine", "this machine (measure now)", checked=True),
    Choice("import", "import a profile file"),
    Choice("enter", "enter a machine by hand", disabled="stage 2"),
]


# --- the terminal asker, over a real prompt_toolkit input --------------------------------------


def test_text_reads_what_was_typed():
    with _asker("qwen" + ENTER) as asker:
        assert asker.text("search", "What are you looking for?") == "qwen"


def test_text_keeps_the_default_when_nothing_is_typed():
    with _asker(ENTER) as asker:
        assert asker.text("context", "Context?", default="8192") == "8192"


def test_select_returns_the_value_of_the_chosen_entry():
    with _asker(DOWN + ENTER) as asker:
        choices = [Choice("here", "this folder"), Choice("path", "a path")]
        assert asker.select("results", "Where?", choices) == "path"


def test_checkbox_returns_every_checked_value():
    with _asker(" " + DOWN + " " + ENTER) as asker:
        assert asker.checkbox("machines", "Machines?", CHOICES) == ["import"]


def test_a_checked_choice_starts_selected():
    with _asker(ENTER) as asker:
        assert asker.checkbox("machines", "Machines?", CHOICES) == ["this-machine"]


def test_confirm_reads_yes_and_no():
    with _asker("y" + ENTER) as asker:
        assert asker.confirm("filter_owners", "Filter?") is True
    with _asker("n" + ENTER) as asker:
        assert asker.confirm("filter_owners", "Filter?") is False


def test_end_of_input_is_a_clean_cancel():
    with _asker(CTRL_D) as asker, pytest.raises(Canceled):
        asker.text("search", "What are you looking for?")


def test_control_c_stays_a_keyboard_interrupt():
    with _asker(CTRL_C) as asker, pytest.raises(KeyboardInterrupt):
        asker.text("search", "What are you looking for?")


def test_end_of_input_cancels_a_selection_list_too():
    with _asker("", closed=True) as asker, pytest.raises(Canceled):
        asker.checkbox("machines", "Machines?", CHOICES)


def test_a_closed_input_cancels_a_text_question_as_well():
    with _asker("", closed=True) as asker, pytest.raises(Canceled):
        asker.text("search", "What are you looking for?")


def test_is_interactive_says_no_without_a_terminal(monkeypatch, tmp_path):
    with (tmp_path / "out").open("w", encoding="utf-8") as handle:
        monkeypatch.setattr("sys.stdout", handle)
        assert is_interactive() is False


# --- the file asker ----------------------------------------------------------------------------


def test_selectable_leaves_out_every_grayed_out_entry():
    assert selectable(CHOICES) == ["this-machine", "import"]


def test_the_file_asker_answers_every_kind_of_question():
    asker = FileAsker(
        {"search": "qwen", "context": 4096, "results": "here", "machines": ["this-machine"], "filter_owners": False}
    )
    assert asker.text("search", "What?") == "qwen"
    assert asker.text("context", "Context?") == "4096"
    assert asker.select("results", "Where?", [Choice("here", "this folder")]) == "here"
    assert asker.checkbox("machines", "Machines?", CHOICES) == ["this-machine"]
    assert asker.confirm("filter_owners", "Filter?") is False
    assert asker.asked == ["search", "context", "results", "machines", "filter_owners"]


def test_a_missing_answer_names_the_question():
    with pytest.raises(AnswerMissingError) as excinfo:
        FileAsker({}).text("search", "What are you looking for?")
    assert excinfo.value.key == "search"
    assert "What are you looking for?" in str(excinfo.value)


def test_a_grayed_out_entry_cannot_be_answered():
    with pytest.raises(AnswerInvalidError):
        FileAsker({"machines": ["enter"]}).checkbox("machines", "Machines?", CHOICES)


def test_an_answer_outside_the_choices_is_refused():
    with pytest.raises(AnswerInvalidError):
        FileAsker({"results": "elsewhere"}).select("results", "Where?", [Choice("here", "this folder")])


def test_a_switch_has_to_be_true_or_false():
    with pytest.raises(AnswerInvalidError):
        FileAsker({"filter_owners": "yes"}).confirm("filter_owners", "Filter?")


def test_a_list_is_not_text():
    with pytest.raises(AnswerInvalidError):
        FileAsker({"search": ["qwen"]}).text("search", "What?")


# --- the answer file ----------------------------------------------------------------------------


def _write(tmp_path, text: str):
    path = tmp_path / "answers.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_answers_returns_every_answer_but_the_schema_version(tmp_path):
    path = _write(tmp_path, f'schema_version = {ANSWERS_SCHEMA_VERSION}\nsearch = "qwen"\ncontext = 8192\n')
    assert read_answers(path) == {"search": "qwen", "context": 8192}


def test_read_answers_takes_lists_and_switches(tmp_path):
    path = _write(tmp_path, 'schema_version = 1\nmachines = ["this-machine"]\nfilter_owners = true\n')
    assert read_answers(path) == {"machines": ["this-machine"], "filter_owners": True}


def test_a_missing_schema_version_is_a_schema_error(tmp_path):
    with pytest.raises(SchemaVersionError):
        read_answers(_write(tmp_path, 'search = "qwen"\n'))


def test_an_unsupported_schema_version_is_a_schema_error(tmp_path):
    with pytest.raises(SchemaVersionError):
        read_answers(_write(tmp_path, 'schema_version = 9\nsearch = "qwen"\n'))


def test_a_file_that_is_not_toml_names_itself(tmp_path):
    with pytest.raises(AnswerFileError) as excinfo:
        read_answers(_write(tmp_path, "search = \n"))
    assert "answers.toml" in str(excinfo.value)


def test_a_missing_file_names_itself(tmp_path):
    with pytest.raises(AnswerFileError):
        read_answers(tmp_path / "nowhere.toml")


def test_an_answer_of_another_shape_is_refused(tmp_path):
    with pytest.raises(AnswerFileError, match="context"):
        read_answers(_write(tmp_path, "schema_version = 1\ncontext = 1.5\n"))
