"""Asking one question, two ways: at the terminal, or out of an answer file.

`Asker` is the whole interface the guided mode uses -- five kinds of question, each with the
key it is stored under in an answer file. `TerminalAsker` runs `questionary` (MIT, on
`prompt_toolkit`, BSD): the only place in this package that imports it, so every other module
and the three automation commands run without it. `FileAsker` answers from a `dict` read by
`modelroom/answers.py` and never touches a terminal.

Every question looks the same: one style and one set of glyphs, both from `modelroom/intro.py`,
a green pointer on the line the keyboard is on, the grayed-out entries with their reason, and
one instruction line under the list. A yes or no question is a list of `Yes` and `No` as well --
there is no `(Y/n)` to read anywhere.

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
    """

    value: str
    label: str
    checked: bool = False
    disabled: str | None = None


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
    means, what nothing marked would do. It stands in the instruction line under the list, which
    disappears with the question, and not as a line of the run: a sentence that explains a list
    has nothing to say once the list is gone (decided 2026-09-24).
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
    """The values a caller may answer with: every choice that is not grayed out."""
    return [choice.value for choice in choices if choice.disabled is None]


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

    def _list_question(self, factory, question: str, choices: Sequence[Choice], instruction: str, **extra):
        """One selection list, with the pointer, the style and the instruction line of this dialog.

        `Esc` is bound after the question is built: `questionary` binds Ctrl-C and Enter and then
        swallows every other key, and it takes no key bindings of its own. The binding ends the
        application the way an end of input does, so `Esc` is the documented exit `2` and not a
        second way out. It is added to lists only -- a text question keeps prompt_toolkit's own
        Esc, which is the prefix of its editing keys.

        The built application erases itself once it is answered (`erase_when_done`), so the
        question and its list leave the screen and the run writes the one answer line in their
        place (`screen.answer_line`). Without it the library writes its own idea of the answer --
        `done (2 selections)`, `[this machine (measure now)]`, the whole marked line of the scale
        -- under a question that is still on screen, which was the "one thing chained to the next"
        of the test round of 2026-09-24.
        """
        import questionary

        built = factory(
            question,
            choices=_questionary_choices(choices),
            pointer=self._glyphs.pointer,
            instruction=instruction,
            style=self._style(),
            **extra,
            **self._streams,
        )
        _bind_escape(built)
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
            questionary.select, question, choices, self.select_instruction(instruction), default=pointer_at
        )

    def checkbox(
        self, key: str, question: str, choices: Sequence[Choice], instruction: str | None = None
    ) -> list[str]:
        import questionary

        return self._list_question(questionary.checkbox, question, choices, self.checkbox_instruction(instruction))

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

    def select_instruction(self, extra: str | None = None) -> str:
        return self._instruction(f"{self._glyphs.move} move   Enter select   Esc leave", extra)

    def checkbox_instruction(self, extra: str | None = None) -> str:
        return self._instruction(f"{self._glyphs.move} move   Space marks   Enter confirms   Esc leave", extra)

    def _instruction(self, keys: str, extra: str | None) -> str:
        """How to move in this list, and what the list itself has to say, on one line."""
        return keys if not extra else f"{keys} {self._glyphs.dot} {extra}"


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


def _questionary_choices(choices: Sequence[Choice]) -> list:
    import questionary

    return [
        questionary.Choice(title=choice.label, value=choice.value, checked=choice.checked, disabled=choice.disabled)
        for choice in choices
    ]


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
