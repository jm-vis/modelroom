"""Tests for modelroom.dialog and modelroom.answers: asking, and answering from a file.

The terminal asker is tested against a real `prompt_toolkit` pipe input -- the keys a user
would press travel through prompt_toolkit and questionary unchanged; nothing is replaced.
"""

from __future__ import annotations

import io
from contextlib import contextmanager

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.plain_text import PlainTextOutput

from modelroom.answers import ANSWERS_SCHEMA_VERSION, AnswerFileError, read_answers
from modelroom.contracts import SchemaVersionError
from modelroom.dialog import (
    AnswerInvalidError,
    AnswerMissingError,
    Canceled,
    Choice,
    FileAsker,
    TerminalAsker,
    columns,
    is_interactive,
    selectable,
)
from modelroom.guided_context import LEVELS, NUMBER_VALUE, scale_choices, tokens_of

DOWN = "\x1b[B"
UP = "\x1b[A"
ENTER = "\r"
CTRL_C = "\x03"
CTRL_D = "\x04"
ESC = "\x1b"


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


def test_confirm_is_a_list_with_the_default_under_the_pointer():
    """Yes and no is a list like every other question -- there is no `(Y/n)` to read (decided 2026-09-24)."""
    with _asker(ENTER) as asker:
        assert asker.confirm("filter_owners", "Filter?") is True
    with _asker(DOWN + ENTER) as asker:
        assert asker.confirm("filter_owners", "Filter?") is False


def test_confirm_starts_on_no_when_that_is_the_default():
    with _asker(ENTER) as asker:
        assert asker.confirm("load_test", "Measure?", default=False) is False
    with _asker(UP + ENTER) as asker:
        assert asker.confirm("load_test", "Measure?", default=False) is True


def test_escape_leaves_a_list_the_way_an_end_of_input_does():
    """The instruction line under every list promises it, so it has to be bound."""
    with _asker(ESC) as asker, pytest.raises(Canceled):
        asker.select("results", "Where?", [Choice("here", "this folder"), Choice("path", "a path")])


def test_escape_leaves_a_checkbox_too():
    with _asker(ESC) as asker, pytest.raises(Canceled):
        asker.checkbox("machines", "Machines?", CHOICES)


def test_the_pointer_of_a_select_starts_on_the_checked_entry():
    with _asker(ENTER) as asker:
        choices = [Choice("here", "this folder"), Choice("path", "a path", checked=True)]
        assert asker.select("results", "Where?", choices) == "path"


# --- the answer line is the run's, not the library's --------------------------------------------


def _transcript(keys: str, ask) -> str:
    """Everything the library really wrote for one question, as a terminal would receive it."""
    sink = io.StringIO()
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        ask(TerminalAsker(input=pipe, output=PlainTextOutput(sink)))
    return sink.getvalue()


def test_a_list_writes_no_answer_of_its_own_once_it_is_answered():
    """`done (2 selections)` under a question that is still on screen was the "chaining" of the
    test round of 2026-09-24. The question erases itself and the run writes the answer in words."""
    transcript = _transcript(" " + DOWN + " " + ENTER, lambda asker: asker.checkbox("machines", "Machines?", CHOICES))

    assert "Machines?" in transcript
    assert "done (" not in transcript
    assert "selections" not in transcript


def test_a_text_question_erases_itself_too():
    """`? What are you looking for? qwen` stayed on screen and the run wrote the same line under
    it -- the same question twice (measured 2026-09-24, second-model round)."""
    transcript = _transcript("qwen\r", lambda asker: asker.text("search", "What are you looking for?"))

    assert "What are you looking for? qwen" not in transcript


def test_a_select_writes_no_answer_of_its_own_either():
    choices = [Choice("here", "this folder (C:/results)"), Choice("path", "another path")]

    transcript = _transcript(ENTER, lambda asker: asker.select("results", "Where?", choices))

    assert "Where?" in transcript
    # The library's own answer line would repeat the whole label of the marked entry.
    assert transcript.count("this folder (C:/results)") == 1


def test_a_list_says_what_it_has_to_say_about_itself_in_its_instruction_line():
    """A sentence that explains a list has nothing to say once the list is gone (2026-09-24)."""
    extra = "nothing marked keeps the folder as it is."

    transcript = _transcript(
        ENTER, lambda asker: asker.checkbox("machines", "Machines?", CHOICES, extra)
    )

    # The line wraps at the width of the window, so the text is held against its start.
    assert "Esc leave · nothing marked" in transcript
    assert "Space marks" in transcript


def test_the_instruction_line_of_a_list_without_one_is_the_keys_alone():
    transcript = _transcript(ENTER, lambda asker: asker.checkbox("machines", "Machines?", CHOICES))

    assert "Space marks   Enter confirms   Esc leave" in transcript


# --- the size scale at the terminal (`select_or_text`) -------------------------------------------


def _scale(default_context: int, cap=None):
    return scale_choices(LEVELS, default_context, {}, cap)


def _ask_scale(keys: str, default_context: int, cap=None) -> str:
    with _asker(keys) as asker:
        scale = _scale(default_context, cap)
        return asker.select_or_text("context", "How much?", scale, NUMBER_VALUE, "How many tokens?")


def test_the_scale_answers_with_the_level_the_pointer_starts_on():
    """A folder that kept no context starts on `L`, so Enter alone is 32768 (decided 2026-09-24)."""
    assert tokens_of(_ask_scale(ENTER, 32768)) == 32768


def test_the_scale_starts_on_the_kept_level():
    assert tokens_of(_ask_scale(ENTER, 16384)) == 16384


def test_a_kept_context_that_is_no_level_starts_on_custom():
    assert tokens_of(_ask_scale(ENTER, 5000)) == 5000


def test_the_last_entry_of_the_scale_leads_to_the_number_question():
    """`enter a number` is the last entry, three lines under the pointer's start on `L`."""
    down_to_the_number = DOWN * (len(LEVELS) - [level.name for level in LEVELS].index("L"))
    keys = down_to_the_number + ENTER + "4096" + ENTER

    assert tokens_of(_ask_scale(keys, 32768)) == 4096


def test_a_level_beyond_the_window_is_grayed_out_with_its_reason():
    choices = _scale(32768, (32768, "Qwen/Qwen3.5-9B"))

    assert [choice.value for choice in choices if choice.disabled] == ["XL", "XXL"]
    assert selectable(choices) == ["XS", "S", "M", "L", NUMBER_VALUE]
    assert all(choice.disabled == "beyond the window of Qwen/Qwen3.5-9B" for choice in choices if choice.disabled)


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


# --- the size scale from an answer file -----------------------------------------------------------


def _answered_context(value, cap=None) -> str:
    asker = FileAsker({"context": value})
    return asker.select_or_text("context", "How much?", _scale(32768, cap), NUMBER_VALUE, "How many?")


@pytest.mark.parametrize("value, expected", [("L", 32768), ("32768", 32768), (32768, 32768), ("XS", 4096)])
def test_a_file_answers_the_scale_with_a_level_or_a_number(value, expected):
    assert tokens_of(_answered_context(value)) == expected


def test_a_level_beyond_the_window_cannot_be_answered_from_a_file():
    with pytest.raises(AnswerInvalidError) as excinfo:
        _answered_context("XXL", cap=(32768, "Qwen/Qwen3.5-9B"))

    assert "beyond the window of Qwen/Qwen3.5-9B" in str(excinfo.value)


def test_a_level_beyond_the_window_cannot_be_smuggled_in_with_a_space():
    """`" XXL "` matched no entry, so it passed the check and was translated a step later."""
    with pytest.raises(AnswerInvalidError, match="beyond the window"):
        _answered_context(" XXL ", cap=(32768, "Qwen/Qwen3.5-9B"))


def test_a_level_with_spaces_around_it_is_still_that_level():
    assert tokens_of(_answered_context("  L  ")) == 32768


def test_a_context_that_is_neither_a_level_nor_a_number_is_the_callers_error():
    """The answer itself is text, so the list takes it; the step turns it into its own message."""
    from modelroom.guided import GuidedError

    with pytest.raises(GuidedError, match="neither a level"):
        tokens_of(_answered_context("huge"))


def test_a_context_no_ranking_can_be_computed_for_names_itself():
    from modelroom.guided import GuidedError

    with pytest.raises(GuidedError, match="not a context"):
        tokens_of(_answered_context("0"))


# --- the columns of a list ------------------------------------------------------------------------


def test_columns_pad_every_cell_a_width_is_given_for():
    rows = [["XS", "4k", "a"], ["XXL", "128k", "b"]]

    assert columns(rows, (4, 6)) == ["XS    4k      a", "XXL   128k    b"]


def test_a_cell_wider_than_its_column_pushes_its_own_row_and_is_never_cut():
    assert columns([["overlong", "x"]], (4,)) == ["overlong  x"]


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
