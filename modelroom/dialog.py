"""Asking one question, two ways: at the terminal, or out of an answer file.

`Asker` is the whole interface the guided mode uses -- five kinds of question, each with the
key it is stored under in an answer file. `TerminalAsker` runs `questionary` (MIT, on
`prompt_toolkit`, BSD): the only place in this package that imports it, so every other module
and the three automation commands run without it. `FileAsker` answers from a `dict` read by
`modelroom/answers.py` and never touches a terminal.

Every question looks the same: one style and one set of glyphs, both from `modelroom/intro.py`,
a green pointer on the line the keyboard is on, the grayed-out entries with their reason, the keys
behind the question and what the list has to say about itself as a gray line under it. A yes or no
question is a list of `Yes` and `No` as well -- there is no `(Y/n)` to read anywhere.

A list that marks answers with what is marked, and with the row under the pointer where nothing
is marked at all: `Enter` alone took nothing and a reader read the pointer as the selection ("I
thought the bar was the selection", test round 2026-09-25).

Ending the dialog: Ctrl-C raises `KeyboardInterrupt` (the caller ends with exit `130`), an end
of input and `Esc` in a list raise `Canceled` (exit `2`). All of them leave the run without a
half-written file, because every write the guided mode does is atomic and happens after the last
question of its step.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, Sequence

from .intro import GRAY, GREEN, WHITE, glyphs, style_rules

YES = "yes"
NO = "no"


@dataclass(frozen=True)
class Choice:
    """One entry of a selection list: what is stored, what is shown, and why it is not selectable.

    `disabled` is the grayed-out case -- an entry that is shown for what it says (a machine that
    is already in the results folder, a step that belongs to a later stage) and cannot be picked.
    `checked` marks an entry of a `checkbox` as already marked, and in a `select` the entry the
    pointer starts on; at most one entry of a `select` carries it.

    `heading` is a line of the list that is no entry at all -- the column head of a table-shaped
    list (`guided_models.list_header`). It carries no value, the pointer never rests on it and no
    answer may name it. `dim` draws the whole row gray, for a row a reader should read as a side
    note rather than a recommendation (a `legacy` model, decided 2026-09-25).
    """

    value: str
    label: str
    checked: bool = False
    disabled: str | None = None
    heading: bool = False
    dim: bool = False


class Canceled(Exception):
    """The dialog ended without an answer (end of input)."""


class AnswerMissingError(Exception):
    """The answer file has no answer for a question the dialog asked."""

    def __init__(self, key: str, question: str) -> None:
        super().__init__(f"the answer file has no answer for {key!r} ({question})")
        self.key = key
        self.question = question


class AnswerInvalidError(Exception):
    """The answer file answers a question with something that question cannot take."""


class Asker(Protocol):
    """Five kinds of question. `key` is the answer file's key for the same question.

    `instruction` is what a list says about itself beyond how to move in it -- what a column
    means, what a word in it stands for. It stands in a gray line of its own under the list and
    disappears with the question rather than staying as a line of the run: a sentence that explains
    a list has nothing to say once the list is gone (decided 2026-09-24).
    """

    def text(self, key: str, question: str, default: str = "") -> str: ...

    def select(self, key: str, question: str, choices: Sequence[Choice], instruction: str | None = None) -> str: ...

    def checkbox(
        self, key: str, question: str, choices: Sequence[Choice], instruction: str | None = None
    ) -> list[str]: ...

    def confirm(self, key: str, question: str, default: bool = True) -> bool: ...

    def select_or_text(
        self,
        key: str,
        question: str,
        choices: Sequence[Choice],
        text_value: str,
        text_question: str,
        instruction: str | None = None,
    ) -> str: ...


def columns(rows: Sequence[Sequence[str]], widths: Sequence[int]) -> list[str]:
    """The labels of one list, aligned in columns, so a scale, a hit list and a load test read as
    a table. A cell wider than its column pushes the rest of its own row and is never cut; a
    column `widths` does not name (the last one, usually) is not padded at all."""
    lines = []
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths)]
        lines.append("  ".join([*cells, *row[len(widths) :]]).rstrip())
    return lines


def is_interactive() -> bool:
    """Whether this process has a terminal on both ends -- the condition for asking at all."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):  # a closed or replaced stream
        return False


def selectable(choices: Sequence[Choice]) -> list[str]:
    """The values a caller may answer with: every choice that is neither grayed out nor a heading."""
    return [choice.value for choice in choices if choice.disabled is None and not choice.heading]


class TerminalAsker:
    """Asks at the terminal through `questionary`, in the one style of the guided mode.

    `input`/`output` are `prompt_toolkit`'s own; the real CLI passes neither and questionary
    uses the console. A test passes a `create_pipe_input()` pipe and a `DummyOutput`, so the
    keys a user would press really travel through prompt_toolkit -- no function is replaced.
    """

    def __init__(self, input=None, output=None) -> None:  # noqa: A002 -- prompt_toolkit's own names
        self._streams = {}
        if input is not None:
            self._streams["input"] = input
        if output is not None:
            self._streams["output"] = output
        # The glyph set is a question about the console this process writes to, which is
        # `sys.stdout` -- prompt_toolkit's `Output` is not a text stream and has none of its own.
        self._glyphs = glyphs()

    def _ask(self, question):
        try:
            return question.unsafe_ask()
        except EOFError as exc:
            raise Canceled("the dialog ended without an answer") from exc

    def _style(self):
        import questionary

        return questionary.Style(style_rules() + QUESTION_RULES)

    def _list_question(self, factory, question: str, choices: Sequence[Choice], keys: str, hint: str | None, bind=None, **extra):
        """One selection list, with the pointer, the style and the instruction lines of this dialog.

        `Esc` is bound after the question is built: `questionary` binds Ctrl-C and Enter and then
        swallows every other key, and it takes no key bindings of its own. The binding ends the
        application the way an end of input does, so `Esc` is the documented exit `2` and not a
        second way out. It is added to lists only -- a text question keeps prompt_toolkit's own
        Esc, which is the prefix of its editing keys.

        **The keys stand behind the question, what the list says about itself under the list.** Two
        places, because the two lines are not worth the same: the library gives the list window the
        rows the terminal has left and scrolls it around the pointer, so a line at the end of the
        list is out of sight while the pointer is at its start (measured 2026-09-25 against
        `prompt_toolkit`'s renderer, with 30 models in 24 rows). The keys are what a reader needs at
        the moment they cannot go on -- `Enter` alone took nothing in the test round of 2026-09-25 --
        so they stay next to the question, which has a window of its own and wraps. The glossary of a
        column is a line a reader looks up once; it stands where the mockup of 2026-09-25 draws it.

        The built application erases itself once it is answered (`erase_when_done`), so the
        question and its list leave the screen and the run writes the one answer line in their
        place (`screen.answer_line`). Without it the library writes its own idea of the answer --
        `done (2 selections)`, `[this machine (measure now)]`, the whole marked line of the scale
        -- under a question that is still on screen, which was the "one thing chained to the next"
        of the test round of 2026-09-24.
        """
        import questionary

        rows = [] if not hint else [questionary.Separator(hint)]
        built = factory(
            question,
            choices=[*_questionary_choices(choices), *rows],
            pointer=self._glyphs.pointer,
            instruction=keys,
            style=self._style(),
            **extra,
            **self._streams,
        )
        _bind_escape(built)
        _wrap_list_lines(built)
        if bind is not None:
            bind(built)
        built.application.erase_when_done = True
        return self._ask(built)

    def text(self, key: str, question: str, default: str = "") -> str:
        """A typed answer. It erases itself too, for the same reason a list does.

        Without `erase_when_done` the library leaves `? What are you looking for? qwen` on screen
        and the run writes its own line under it -- the same question twice (measured 2026-09-24,
        second-model round).
        """
        import questionary

        built = questionary.text(
            question, default=default, style=self._style(), erase_when_done=True, **self._streams
        )
        return self._ask(built)

    def select(self, key: str, question: str, choices: Sequence[Choice], instruction: str | None = None) -> str:
        import questionary

        pointer_at = _pointer_at(choices)
        return self._list_question(
            questionary.select, question, choices, self.select_keys(), instruction, default=pointer_at
        )

    def select_keys(self) -> str:
        return f"{self._glyphs.move} move   Enter select   Esc leave"

    def checkbox(
        self, key: str, question: str, choices: Sequence[Choice], instruction: str | None = None
    ) -> list[str]:
        """A list that marks. `Enter` takes the marked rows, and the row under the pointer where
        nothing is marked at all (`_bind_enter_takes_pointed`, decided 2026-09-25)."""
        import questionary

        return self._list_question(
            questionary.checkbox,
            question,
            choices,
            self.checkbox_keys(),
            instruction,
            bind=_bind_enter_takes_pointed,
        )

    def confirm(self, key: str, question: str, default: bool = True) -> bool:
        """A yes or no question as a list of two entries, the default one under the pointer."""
        choices = [Choice(YES, "Yes", checked=default), Choice(NO, "No", checked=not default)]
        return self.select(key, question, choices) == YES

    def select_or_text(
        self,
        key: str,
        question: str,
        choices: Sequence[Choice],
        text_value: str,
        text_question: str,
        instruction: str | None = None,
    ) -> str:
        """A selection list with one entry that leads to a text question (the scale's own number).

        The text question starts empty on purpose: what this folder kept is already the line
        the pointer starts on, and a prefilled field would grow into `14096` the moment
        someone types `4096` in front of it -- prompt_toolkit puts the cursor behind a
        default value, never over it.
        """
        chosen = self.select(key, question, choices, instruction)
        if chosen != text_value:
            return chosen
        return self.text(key, text_question)

    def checkbox_keys(self) -> str:
        """`Enter confirms` said nothing about what it would confirm: a run that pressed it without
        the space bar took nothing at all ("I thought the bar was the selection", test round
        2026-09-25). The line now says both cases, in the order they happen."""
        return f"{self._glyphs.move} move   Space marks   Enter takes the marked rows, or this one   Esc leave"


# The question's own classes, on top of `intro.style_rules`: questionary names them, this package
# only says which color each one is. The pointer and the answer are green, the line the keyboard
# is on is white and bold, a grayed-out entry and the instruction line are gray. `screen.py` hands
# the same list to `prompt_toolkit`, so the answer line the run writes is the green the question's
# own answer was.
QUESTION_RULES = [
    ("qmark", f"fg:{GREEN} bold"),
    ("question", "bold"),
    ("answer", f"fg:{GREEN} bold"),
    ("pointer", f"fg:{GREEN} bold"),
    ("highlighted", f"fg:{WHITE} bold"),
    ("selected", f"fg:{GREEN}"),
    ("instruction", f"fg:{GRAY}"),
    ("disabled", f"fg:{GRAY}"),
    # A line of a list that is no entry: the column head of a table-shaped list, and the
    # instruction lines under it.
    ("separator", f"fg:{GRAY}"),
]


def _pointer_at(choices: Sequence[Choice]) -> str | None:
    """The value the pointer of a `select` starts on: the first checked entry that can be picked."""
    return next((choice.value for choice in choices if choice.checked and choice.disabled is None), None)


def _bind_escape(question) -> None:
    """Make `Esc` end this list the way an end of input does (`Canceled`, exit `2`)."""
    from prompt_toolkit.keys import Keys

    @question.application.key_bindings.add(Keys.Escape, eager=True)
    def _leave(event) -> None:
        event.app.exit(exception=EOFError, style="class:aborting")


def _list_window(question):
    """The window `questionary` draws the choices of a built question in.

    The library builds it inside `checkbox()`/`select()` and keeps no reference a caller could ask
    for, so it is found where it really is: the one window whose content is an `InquirerControl`.
    Read only -- the control is the library's, and nothing here replaces one of its functions. A
    `questionary` that laid its list out differently would raise here rather than silently leave
    `Enter` and the instruction lines to chance; the version is pinned in `uv.lock`, and the tests
    of this module answer the same question every time they run.
    """
    from prompt_toolkit.layout.containers import Window
    from questionary.prompts.common import InquirerControl

    for container in question.application.layout.walk():
        if isinstance(container, Window) and isinstance(container.content, InquirerControl):
            return container
    raise RuntimeError("this questionary version builds its list without an InquirerControl in a window")


def _wrap_list_lines(question) -> None:
    """Let a line of the list wrap instead of being cut at the edge of the window.

    Measured 2026-09-25 in a window of 100 columns: the instruction line of step 2 is 153
    characters, and the library's own window does not wrap -- `fit from the size at 32k context,
    exact after the fetch` was simply gone. Behind the question, where that line used to stand, it
    wrapped, so this restores what it had there.
    """
    from prompt_toolkit.filters import to_filter

    _list_window(question).wrap_lines = to_filter(True)


def _bind_enter_takes_pointed(question) -> None:
    """Make `Enter` take the row under the pointer where nothing is marked (decided 2026-09-25).

    A binding added after the question is built wins over the library's own for the same key
    (prompt_toolkit calls the last matching handler), the same way `_bind_escape` does, so
    `questionary` itself is unchanged. With something marked the answer is exactly what is marked:
    the row under the pointer is then the row a reader moved past, not a choice.

    The library's own handler validates before it exits; this package passes no validator to
    `questionary.checkbox`, so every answer is valid and there is nothing to run here.
    """
    from prompt_toolkit.keys import Keys

    control = _list_window(question).content

    @question.application.key_bindings.add(Keys.ControlM, eager=True)
    def _take(event) -> None:
        if not control.selected_options:
            pointed = control.get_pointed_at()
            if pointed.value is not None and not pointed.disabled:
                control.selected_options.append(pointed.value)
        control.submission_attempted = True
        control.is_answered = True
        event.app.exit(result=[choice.value for choice in control.get_selected_values()])


def _questionary_choices(choices: Sequence[Choice]) -> list:
    """This package's choices as the library's own, with a heading as its `Separator`.

    A `Separator` is what `questionary` draws as a line of the list nobody can point at, which is
    exactly what a column head is. A `dim` row is handed over as a formatted title -- a list of
    `(class, text)` fragments the library prints as they are -- so the row is gray as a whole; its
    characters are the same ones a console without color prints.
    """
    import questionary

    built = []
    for choice in choices:
        if choice.heading:
            built.append(questionary.Separator(choice.label))
            continue
        title = [("class:note", choice.label)] if choice.dim else choice.label
        built.append(
            questionary.Choice(title=title, value=choice.value, checked=choice.checked, disabled=choice.disabled)
        )
    return built


class FileAsker:
    """Answers from the answer file (`modelroom --answers <file>`), with the same four questions.

    Every answer is checked against what the question offers: a `select` has to name one
    selectable choice, a `checkbox` a list of them, a `confirm` a boolean, a `text` a string
    (a whole number is accepted for a question whose answer is a number, e.g. the context).
    A question with no answer raises `AnswerMissingError` naming it.
    """

    def __init__(self, answers: dict[str, object]) -> None:
        self._answers = dict(answers)
        self.asked: list[str] = []

    def _answer(self, key: str, question: str) -> object:
        self.asked.append(key)
        if key not in self._answers:
            raise AnswerMissingError(key, question)
        return self._answers[key]

    def text(self, key: str, question: str, default: str = "") -> str:
        value = self._answer(key, question)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise AnswerInvalidError(f"the answer for {key!r} is not text: {value!r}")
        return str(value)

    def select(self, key: str, question: str, choices: Sequence[Choice], instruction: str | None = None) -> str:
        value = self._answer(key, question)
        allowed = selectable(choices)
        if value not in allowed:
            raise AnswerInvalidError(f"the answer for {key!r} is {value!r}, not one of {allowed}")
        return str(value)

    def checkbox(
        self, key: str, question: str, choices: Sequence[Choice], instruction: str | None = None
    ) -> list[str]:
        value = self._answer(key, question)
        allowed = selectable(choices)
        if not isinstance(value, list) or any(entry not in allowed for entry in value):
            raise AnswerInvalidError(f"the answer for {key!r} is {value!r}, not a list out of {allowed}")
        return list(value)

    def confirm(self, key: str, question: str, default: bool = True) -> bool:
        value = self._answer(key, question)
        if not isinstance(value, bool):
            raise AnswerInvalidError(f"the answer for {key!r} is {value!r}, not true or false")
        return value

    def select_or_text(
        self,
        key: str,
        question: str,
        choices: Sequence[Choice],
        text_value: str,
        text_question: str,
        instruction: str | None = None,
    ) -> str:
        """The answer as text, whether it names an entry of the list or is the free text itself.

        The list of the size scale offers its levels **and** a number of your own, so a file may
        answer `"L"` or `"32768"` and both are right. An entry that is grayed out in this run is
        refused with the reason it is grayed out for, exactly as a `select` refuses one; the
        caller checks the rest (a number this ranking cannot be computed for is its message).
        """
        value = self._answer(key, question)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise AnswerInvalidError(f"the answer for {key!r} is not text or a whole number: {value!r}")
        # Stripped **before** the grayed-out entries are checked, and returned stripped: otherwise
        # `" XXL "` would slip past a level the window forbids and be translated a step later.
        answer = str(value).strip()
        blocked = next((choice for choice in choices if choice.value == answer and choice.disabled is not None), None)
        if blocked is not None:
            raise AnswerInvalidError(f"the answer for {key!r} is {answer!r}, which this run cannot take: {blocked.disabled}")
        return answer
