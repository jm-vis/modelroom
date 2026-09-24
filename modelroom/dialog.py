"""Asking one question, two ways: at the terminal, or out of an answer file.

`Asker` is the whole interface the guided mode uses -- four kinds of question, each with the
key it is stored under in an answer file. `TerminalAsker` runs `questionary` (MIT, on
`prompt_toolkit`, BSD): the only place in this package that imports it, so every other module
and the three automation commands run without it. `FileAsker` answers from a `dict` read by
`modelroom/answers.py` and never touches a terminal.

Ending the dialog: Ctrl-C raises `KeyboardInterrupt` (the caller ends with exit `130`), an end
of input raises `Canceled` (exit `2`). Both leave the run without a half-written file, because
every write the guided mode does is atomic and happens after the last question of its step.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Choice:
    """One entry of a selection list: what is stored, what is shown, and why it is not selectable.

    `disabled` is the grayed-out case -- an entry that is shown for what it says (a machine that
    is already in the results folder, a step that belongs to a later stage) and cannot be picked.
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
    """Four kinds of question. `key` is the answer file's key for the same question."""

    def text(self, key: str, question: str, default: str = "") -> str: ...

    def select(self, key: str, question: str, choices: Sequence[Choice]) -> str: ...

    def checkbox(self, key: str, question: str, choices: Sequence[Choice]) -> list[str]: ...

    def confirm(self, key: str, question: str, default: bool = True) -> bool: ...


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
    """Asks at the terminal through `questionary`.

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

    def _ask(self, question):
        try:
            return question.unsafe_ask()
        except EOFError as exc:
            raise Canceled("the dialog ended without an answer") from exc

    def text(self, key: str, question: str, default: str = "") -> str:
        import questionary

        return self._ask(questionary.text(question, default=default, **self._streams))

    def select(self, key: str, question: str, choices: Sequence[Choice]) -> str:
        import questionary

        return self._ask(questionary.select(question, choices=_questionary_choices(choices), **self._streams))

    def checkbox(self, key: str, question: str, choices: Sequence[Choice]) -> list[str]:
        import questionary

        return self._ask(questionary.checkbox(question, choices=_questionary_choices(choices), **self._streams))

    def confirm(self, key: str, question: str, default: bool = True) -> bool:
        import questionary

        return self._ask(questionary.confirm(question, default=default, **self._streams))


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

    def select(self, key: str, question: str, choices: Sequence[Choice]) -> str:
        value = self._answer(key, question)
        allowed = selectable(choices)
        if value not in allowed:
            raise AnswerInvalidError(f"the answer for {key!r} is {value!r}, not one of {allowed}")
        return str(value)

    def checkbox(self, key: str, question: str, choices: Sequence[Choice]) -> list[str]:
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
