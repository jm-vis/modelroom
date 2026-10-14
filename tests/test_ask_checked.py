"""Tests for AP9-RE: a required input asks again in the terminal, an answer file keeps its exit `2`.

The asker is a scripted double for the logic of each question (what is noted, what is written when),
and the real `TerminalAsker` over a `prompt_toolkit` pipe for what the keys do and what the terminal
really shows. Nothing of `prompt_toolkit` is replaced.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from modelroom import guided, guided_context, guided_entered
from modelroom.config import ConfigError, GuidedConfig
from modelroom.dialog import (
    REQUIRED_TEXT_HINT,
    AnswerMissingError,
    Canceled,
    Choice,
    FileAsker,
    TerminalAsker,
)
from modelroom.guided import REQUIRED_AGAIN, GuidedError, ask_checked
from modelroom.guided_context import (
    AT_ONCE_BOTH_KEYS,
    AT_ONCE_KEY,
    NUMBER_VALUE,
    Checked,
    RequestsAnswer,
    at_once_answer,
    tokens_of,
)
from modelroom.guided_entered import DISPLAY_NAME_MAX, _name_of, cards_of, gib_of
from modelroom.selection import RunSelection

DOWN = "\x1b[B"
UP = "\x1b[A"
ENTER = "\r"
CTRL_C = "\x03"
CTRL_D = "\x04"
ESC = "\x1b"


class _Screen:
    """The lines a step writes: notes and answers, in one list of events as well."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.notes: list[str] = []
        self.answers: list[tuple[str, str]] = []
        self.events = events if events is not None else []

    def blank(self) -> None: ...

    def head(self, *_args) -> None: ...

    def note(self, text: str) -> None:
        self.notes.append(text)
        self.events.append(f"note {text}")

    def answer(self, question: str, words: str, **_kwargs) -> None:
        self.answers.append((question, words))
        self.events.append("answer")


class _Terminal:
    """A terminal-like asker that is no `FileAsker`: it answers from a script, key by key, and keeps what it was asked."""

    def __init__(self, script: dict[str, list[str]], screen: _Screen) -> None:
        self.script = {key: list(answers) for key, answers in script.items()}
        self.screen = screen
        self.asked: list[dict] = []

    def holds(self, key: str) -> bool:
        return False

    def _next(self, key: str, **seen) -> str:
        self.asked.append({"key": key, "answers_so_far": len(self.screen.answers), **seen})
        self.screen.events.append(f"ask {key}")
        return self.script[key].pop(0)

    def text(self, key, question, default="", *, required=False):
        return self._next(key, required=required, choices=None)

    def select_or_text(self, key, question, choices, text_value, text_question, instruction=None, *, required=False):
        return self._next(key, required=required, choices=list(choices))


def _run(script: dict[str, list[str]], tmp_path: Path | None = None, **extra) -> SimpleNamespace:
    screen = _Screen()
    asker = _Terminal(script, screen)
    problems: list[str] = []
    return SimpleNamespace(
        asker=asker,
        screen=screen,
        here=tmp_path or Path("."),
        selection=RunSelection(machines=(), models=()),
        servers=SimpleNamespace(found={}),
        catalog=None,
        pointer_path=(tmp_path or Path(".")) / "pointer.json",
        problems=problems,
        failed=problems.append,
        now=None,
        probes=SimpleNamespace(hostname=lambda: "host"),
        **extra,
    )


def _reader_error(read, *args) -> str:
    with pytest.raises(GuidedError) as caught:
        read(*args)
    return str(caught.value)


GUIDED = GuidedConfig()
TOO_LONG = "x" * (DISPLAY_NAME_MAX + 1)


def _context_call(run):
    return guided_context._answered_context(run, SimpleNamespace(guided=GUIDED), Checked(reason="none"))


# One row per question of the inventory that asks again: how to ask it, the key it is asked under, an
# answer its reader refuses, the note that refusal gives, an answer it takes, and what that gives.
SITES = {
    "at_once": dict(
        ask=lambda run: guided_context._answered_requests(run, SimpleNamespace(guided=GUIDED)),
        key="at_once",
        bad="abc",
        note=_reader_error(at_once_answer, "abc", GUIDED),
        good="3",
        result=RequestsAnswer(None, 3, "entered"),
        list_question=True,
        answer_line=False,
    ),
    "context": dict(
        ask=_context_call,
        key="context",
        bad="xyz",
        note=_reader_error(tokens_of, "xyz"),
        good="8192",
        result=8192,
        list_question=True,
        answer_line=False,
    ),
    "entered_name": dict(
        ask=guided_entered._ask_name,
        key="entered_name",
        bad=TOO_LONG,
        note=_reader_error(_name_of, TOO_LONG),
        good="Office",
        result="Office",
        list_question=False,
        answer_line=True,
    ),
    "entered_ram": dict(
        ask=lambda run: guided_entered._ask_gib(run, "entered_ram"),
        key="entered_ram",
        bad="lots",
        note=_reader_error(gib_of, "entered_ram", "lots"),
        good="32",
        result=32.0,
        list_question=True,
        answer_line=True,
    ),
    "entered_vram": dict(
        ask=lambda run: guided_entered._ask_gib(run, "entered_vram"),
        key="entered_vram",
        bad="0",
        note=_reader_error(gib_of, "entered_vram", "0"),
        good="24",
        result=24.0,
        list_question=True,
        answer_line=True,
    ),
    "entered_cards": dict(
        ask=guided_entered._ask_cards,
        key="entered_cards",
        bad="1",
        note=_reader_error(cards_of, "1"),
        good="4",
        result=4,
        list_question=True,
        answer_line=True,
    ),
}


@pytest.fixture(autouse=True)
def _no_snapshot(monkeypatch):
    monkeypatch.setattr(guided_context, "snapshot_packages", lambda *args, **kwargs: ([], []))


@pytest.fixture(params=sorted(SITES))
def site(request):
    return SITES[request.param]


def test_an_invalid_answer_asks_again_with_the_note_of_the_reader(site):
    run = _run({site["key"]: [site["bad"], site["good"]]})

    assert site["ask"](run) == site["result"]
    assert run.screen.notes == [site["note"]]
    # Nothing was written to the screen as an answer before the check took one; the answer line, where this
    # place writes it, is one line and comes after.
    assert [asked["answers_so_far"] for asked in run.asker.asked] == [0, 0]
    assert len(run.screen.answers) == (1 if site["answer_line"] else 0)


@pytest.mark.parametrize("empty", ["", "   ", "\t "])
def test_an_empty_answer_says_an_answer_is_needed(site, empty):
    run = _run({site["key"]: [empty, site["good"]]})

    assert site["ask"](run) == site["result"]
    assert run.screen.notes == [REQUIRED_AGAIN] == ["an answer is needed here"]


def test_the_terminal_keeps_asking_after_many_wrong_answers(site):
    run = _run({site["key"]: [site["bad"]] * 4 + [site["good"]]})

    assert site["ask"](run) == site["result"]
    assert run.screen.notes == [site["note"]] * 4
    assert len(run.asker.asked) == 5


def test_every_required_question_is_asked_as_required_and_a_list_starts_on_the_number_when_asked_again(site):
    run = _run({site["key"]: [site["bad"], site["good"]]})

    site["ask"](run)

    first, second = run.asker.asked
    assert first["required"] is True and second["required"] is True
    if site["list_question"]:
        assert [choice.checked for choice in second["choices"]] == [choice.value == NUMBER_VALUE for choice in second["choices"]]
        # The same rows, the same labels; only the start of the pointer moved.
        assert [(choice.value, choice.label) for choice in second["choices"]] == [
            (choice.value, choice.label) for choice in first["choices"]
        ]
        assert first["choices"] is not second["choices"]


# --- the answer file never asks again ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "ask", "bad"),
    [
        ("at_once", SITES["at_once"]["ask"], "abc"),
        ("at_once", SITES["at_once"]["ask"], ""),
        ("context", _context_call, "xyz"),
        ("context", _context_call, ""),
        ("entered_name", guided_entered._ask_name, ""),
        ("entered_ram", SITES["entered_ram"]["ask"], "0"),
        ("entered_ram", SITES["entered_ram"]["ask"], ""),
        ("entered_vram", SITES["entered_vram"]["ask"], "-3"),
        ("entered_cards", guided_entered._ask_cards, "1"),
        ("entered_cards", guided_entered._ask_cards, ""),
    ],
)
def test_an_answer_file_with_an_invalid_value_ends_the_run_without_asking_again(key, ask, bad):
    run = _run({})
    run.asker = FileAsker({key: bad})

    with pytest.raises(GuidedError):
        ask(run)

    assert run.asker.asked == [key]
    assert run.screen.notes == []


def test_an_answer_file_without_the_key_keeps_its_own_error():
    run = _run({})
    run.asker = FileAsker({})

    with pytest.raises(AnswerMissingError):
        guided_context._answered_requests(run, SimpleNamespace(guided=GUIDED))


def test_an_answer_file_with_users_only_takes_the_old_key_and_does_not_ask_at_once():
    run = _run({})
    run.asker = FileAsker({"users": "5"})

    answer = guided_context._answered_requests(run, SimpleNamespace(guided=GUIDED))

    assert answer.origin == "from_users" and answer.users == 5
    assert run.asker.asked == ["users"]


@pytest.mark.parametrize("bad", ["zzz", "", "0"])
def test_an_answer_file_with_an_invalid_users_value_ends_the_run(bad):
    run = _run({})
    run.asker = FileAsker({"users": bad})

    with pytest.raises(GuidedError):
        guided_context._answered_requests(run, SimpleNamespace(guided=GUIDED))

    assert run.asker.asked == ["users"]


def test_an_answer_file_with_both_keys_is_refused_before_any_question():
    run = _run({})
    run.asker = FileAsker({"users": "5", AT_ONCE_KEY: "3"})

    with pytest.raises(GuidedError) as caught:
        guided_context._answered_requests(run, SimpleNamespace(guided=GUIDED))

    assert str(caught.value) == AT_ONCE_BOTH_KEYS
    assert run.asker.asked == []


# --- ask_checked itself ------------------------------------------------------------------------------


def test_only_the_error_of_the_check_is_caught_and_every_error_of_the_question_passes():
    run = _run({})

    def question(_again: bool) -> str:
        raise AnswerMissingError("key", "question?")

    with pytest.raises(AnswerMissingError):
        ask_checked(run, question, lambda answer: answer)

    def check(_answer: str) -> str:
        raise ValueError("not a guided error")

    with pytest.raises(ValueError):
        ask_checked(run, lambda _again: "x", check)
    assert run.screen.notes == []


def test_the_question_learns_that_it_is_asked_again():
    run = _run({})
    seen: list[bool] = []

    def question(again: bool) -> str:
        seen.append(again)
        return "ok" if len(seen) == 3 else "no"

    def check(answer: str) -> str:
        if answer != "ok":
            raise GuidedError("not yet")
        return answer

    assert ask_checked(run, question, check) == "ok"
    assert seen == [False, True, True]


# --- when the step writes ------------------------------------------------------------------------------


def test_step_3_writes_after_the_answer_was_taken_and_not_before(tmp_path: Path, monkeypatch):
    """A refused answer leaves no answer line and no write; each answer is written once, at the end of its question."""
    events: list[str] = []
    screen = _Screen(events)
    run = _run({"at_once": ["abc", "3"], "context": ["xyz", "L"]}, tmp_path)
    run.screen = screen
    run.asker.screen = screen
    config = SimpleNamespace(guided=GuidedConfig(users=None, requests=3, requests_origin="entered"))

    def write(_run, _file, _change):
        events.append("write")
        return config

    monkeypatch.setattr(guided, "_write_config", write)
    monkeypatch.setattr(guided_context, "machine_checked", lambda *args, **kwargs: Checked(name="m", shown="m"))

    scenario, _config = guided_context.context_step(run, tmp_path / "modelroom.toml", config)

    assert [event for event in events if not event.startswith("note")] == [
        "ask at_once",
        "ask at_once",
        "answer",
        "write",
        "ask context",
        "ask context",
        "answer",
        "write",
    ]
    assert len(screen.notes) == 2
    assert scenario.context_requested == 32768 and scenario.requests == 3


# --- the real terminal asker ---------------------------------------------------------------------------


class _Recording(TerminalAsker):
    """The real asker; it also keeps the lists it was given."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lists: list[list[Choice]] = []

    def select_or_text(self, key, question, choices, text_value, text_question, instruction=None, *, required=False):
        self.lists.append(list(choices))
        return super().select_or_text(key, question, choices, text_value, text_question, instruction, required=required)


def _terminal_run(keys: str):
    kept = GuidedConfig(users=None, requests=7, requests_origin="entered")
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        asker = _Recording(input=pipe, output=DummyOutput())
        run = _run({})
        run.asker = asker
        try:
            answer = guided_context._answered_requests(run, SimpleNamespace(guided=kept))
        finally:
            pass
    return answer, run, asker


def test_the_first_attempt_starts_on_the_line_this_folder_kept():
    answer, run, asker = _terminal_run(ENTER)

    assert answer == RequestsAnswer(None, 7, "entered")
    assert run.screen.notes == []
    assert [choice.value for choice in asker.lists[0] if choice.checked] == [guided_context.KEPT_VALUE]


def test_after_a_wrong_number_the_list_comes_back_on_the_number_and_marks_nothing():
    answer, run, asker = _terminal_run(DOWN + ENTER + "abc" + ENTER + ENTER + "12" + ENTER)

    assert answer == RequestsAnswer(None, 12, "entered")
    assert len(run.screen.notes) == 1
    first, second = asker.lists
    assert [choice.value for choice in first if choice.checked] == [guided_context.KEPT_VALUE]  # untouched by the retry
    assert [choice.value for choice in second if choice.checked] == [NUMBER_VALUE]


def test_after_a_wrong_number_a_row_of_the_list_can_still_be_taken():
    answer, run, _asker = _terminal_run(DOWN + ENTER + "abc" + ENTER + UP + ENTER)

    assert answer == RequestsAnswer(None, 7, "entered")  # the kept line, one row above the number
    assert len(run.screen.notes) == 1


def test_an_empty_number_field_asks_again_with_the_reminder():
    answer, run, _asker = _terminal_run(DOWN + ENTER + ENTER + ENTER + "9" + ENTER)

    assert answer == RequestsAnswer(None, 9, "entered")
    assert run.screen.notes == [REQUIRED_AGAIN]


class _Live(TerminalAsker):
    """The real asker with a hand in every frame of a list: `on_list(number, control)` is called once per
    render of the `number`-th list of this asker, with the library's own control (pointer, marks, rows).
    Keys can be sent from there, so a key arrives only after the list it is meant for is really open."""

    def __init__(self, pipe, on_list) -> None:
        super().__init__(input=pipe, output=DummyOutput())
        self.on_list = on_list
        self.lists = 0

    def _ask(self, question):
        from modelroom.dialog import _list_window

        try:
            control = _list_window(question).content
        except Exception:  # a text field has no list window
            return super()._ask(question)
        self.lists += 1
        number = self.lists
        question.application.after_render.add_handler(lambda _app: self.on_list(number, control))
        return super()._ask(question)


def _live_run(keys: str, on_list, *, limit: float = 20.0):
    """The step-3 question over a pipe with `keys`, `on_list` driving the rest; returns the answer or raises.
    A watchdog closes the pipe after `limit` seconds and the test fails then: a hang is no pass."""
    import threading

    kept = GuidedConfig(users=None, requests=7, requests_origin="entered")
    timed_out: list[bool] = []
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        asker = _Live(pipe, lambda number, control: on_list(pipe, number, control))
        run = _run({})
        run.asker = asker

        def expire() -> None:
            timed_out.append(True)
            pipe.close()

        watchdog = threading.Timer(limit, expire)
        watchdog.start()
        try:
            return guided_context._answered_requests(run, SimpleNamespace(guided=kept)), run
        finally:
            watchdog.cancel()
            assert not timed_out, "the dialog did not end within the time allowed"


def test_the_list_opened_again_really_has_the_pointer_on_the_number_and_nothing_marked():
    states: dict[int, tuple] = {}

    def watch(pipe, number, control):
        states.setdefault(number, (control.get_pointed_at().value, list(control.selected_options)))

    answer, run = _live_run(DOWN + ENTER + "abc" + ENTER + ENTER + "12" + ENTER, watch)

    assert answer == RequestsAnswer(None, 12, "entered")
    assert states[1] == (guided_context.KEPT_VALUE, [])  # the first attempt starts on the kept line, nothing marked
    assert states[2] == (NUMBER_VALUE, [])  # the second one on the number, nothing marked in the control itself


@pytest.mark.parametrize("which", [1, 2])
def test_a_single_esc_leaves_the_list_in_the_first_attempt_and_in_the_one_asked_again(which):
    """Esc is sent by the same means in both: from the first render of that very list, so it is proven open."""
    opened: list[int] = []

    def esc(pipe, number, control):
        if number == which and number not in opened:
            opened.append(number)
            pipe.send_text(ESC)

    keys = "" if which == 1 else DOWN + ENTER + "abc" + ENTER
    with pytest.raises(Canceled):
        _live_run(keys, esc)
    assert opened == [which]
    if which == 2:
        assert len(opened) == 1  # the first list was left by Enter, not by Esc


def test_ctrl_c_and_end_of_input_in_the_question_asked_again_go_through():
    with pytest.raises(KeyboardInterrupt):
        _terminal_run(DOWN + ENTER + "abc" + ENTER + ENTER + CTRL_C)
    with pytest.raises(Canceled):
        _terminal_run(DOWN + ENTER + "abc" + ENTER + ENTER + CTRL_D)


# --- the reminder under a required field ---------------------------------------------------------------


class _Frames(TerminalAsker):
    """The real asker; every frame the renderer draws is kept as the rows of the screen, `None` once it is erased."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frames: list[list[str] | None] = []

    def _ask(self, question):
        def keep(app) -> None:
            screen = app.renderer._last_screen
            if screen is None:
                self.frames.append(None)
                return
            rows = {y: "".join(row[x].char for x in sorted(row)) for y, row in screen.data_buffer.items()}
            self.frames.append([rows[y].rstrip() for y in sorted(rows)])

        question.application.after_render.add_handler(keep)
        return super()._ask(question)


def _frames_of(keys: str, ask) -> list[list[str] | None]:
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        asker = _Frames(input=pipe, output=DummyOutput())
        ask(asker)
    return asker.frames


def test_a_required_text_field_shows_the_reminder_while_it_is_asked_and_not_after():
    frames = _frames_of("5" + ENTER, lambda asker: asker.text("k", "How many?", required=True))

    during = [frame for frame in frames if frame is not None]
    assert during and all(REQUIRED_TEXT_HINT in frame for frame in during)
    assert REQUIRED_TEXT_HINT == "an answer is needed (Ctrl-C leaves)"
    assert during[0][0].startswith("? How many?")
    assert frames[-1] is None  # erased with the question: the reminder does not stay on screen


def test_a_text_field_that_is_not_required_carries_no_reminder():
    frames = _frames_of("5" + ENTER, lambda asker: asker.text("k", "How many?"))

    assert all(frame is None or REQUIRED_TEXT_HINT not in frame for frame in frames)


def test_the_number_field_of_a_list_is_required_when_the_list_is():
    choices = [Choice("1", "one", checked=True), Choice(NUMBER_VALUE, "enter a number")]

    required = _frames_of(
        DOWN + ENTER + "5" + ENTER, lambda asker: asker.select_or_text("k", "Which?", choices, NUMBER_VALUE, "Number", required=True)
    )
    plain = _frames_of(
        DOWN + ENTER + "5" + ENTER, lambda asker: asker.select_or_text("k", "Which?", choices, NUMBER_VALUE, "Number")
    )

    assert any(frame is not None and REQUIRED_TEXT_HINT in frame for frame in required)
    assert all(frame is None or REQUIRED_TEXT_HINT not in frame for frame in plain)
    # The list itself never carries it: only the field behind the number row does.
    assert not any(frame is not None and REQUIRED_TEXT_HINT in frame and "enter a number" in " ".join(frame) for frame in required)


# --- the import path -----------------------------------------------------------------------------------


def _import_run(monkeypatch, tmp_path: Path, script: list[str] | None, answers: dict | None = None):
    """`_import_a_profile` with an import that always fails on its content, so the path is all that matters."""
    calls: list[Path] = []

    def fake_import(config_file, path, now, **_kwargs):
        calls.append(path)
        raise ConfigError("bad content")

    monkeypatch.setattr(guided, "import_profile", fake_import)
    monkeypatch.setattr(guided, "scan_profiles", lambda _directory: SimpleNamespace(profiles={}))
    run = _run({"import_file": script or []}, tmp_path)
    if answers is not None:
        run.asker = FileAsker(answers)
    config = SimpleNamespace(paths=SimpleNamespace(hardware_dir=tmp_path))
    result = guided._import_a_profile(run, tmp_path / "modelroom.toml", config)
    return run, calls, result, config


def test_an_empty_import_path_asks_again_and_nothing_is_imported_meanwhile(monkeypatch, tmp_path: Path):
    good = tmp_path / "profile.json"
    good.write_text("{}", encoding="utf-8")

    for empty in ("", "   "):
        run, calls, _result, _config = _import_run(monkeypatch, tmp_path, [empty, str(good)])
        assert run.screen.notes == [REQUIRED_AGAIN]
        assert calls == [good]
        assert [asked["answers_so_far"] for asked in run.asker.asked] == [0, 0]
        assert len(run.screen.answers) == 1


def test_a_path_with_no_file_asks_again_and_names_it(monkeypatch, tmp_path: Path):
    good = tmp_path / "profile.json"
    good.write_text("{}", encoding="utf-8")
    folder = tmp_path / "a folder"
    folder.mkdir()
    missing = tmp_path / "nope.json"

    run, calls, _result, _config = _import_run(monkeypatch, tmp_path, [str(missing), str(folder), "profile.json"])

    assert run.screen.notes == [f"no file at {missing}", f"no file at {folder}"]
    assert calls == [tmp_path / "profile.json"]  # a relative path is read from this folder


def test_a_file_with_content_the_import_refuses_is_imported_once_and_reported_as_before(monkeypatch, tmp_path: Path):
    bad = tmp_path / "broken.json"
    bad.write_text("not a profile", encoding="utf-8")

    run, calls, result, config = _import_run(monkeypatch, tmp_path, [str(bad)])

    assert calls == [bad]
    assert run.screen.notes == []
    assert run.problems == ["nothing imported: bad content"]
    assert result == (config, set())


@pytest.mark.parametrize("path", ["", "   ", "does-not-exist.json"])
def test_an_answer_file_goes_on_with_whatever_path_it_names(monkeypatch, tmp_path: Path, path):
    run, calls, result, config = _import_run(monkeypatch, tmp_path, None, {"import_file": path})

    assert len(calls) == 1  # no pre-check: the import itself reports what is wrong
    assert run.screen.notes == []
    assert run.problems == ["nothing imported: bad content"]
    assert result == (config, set())


# --- the fit line on top of a long model list ----------------------------------------------------------


def _list_frames(columns: int, rows: int, count: int, fit: str, legend: str, header: str):
    """Every frame of a `count`-model checkbox with `notice=fit` in a window of `columns` x `rows`: the pointer is
    sent to the last model once the list is open, and the list is answered once that row is on screen."""
    import threading

    from prompt_toolkit.data_structures import Size

    class _Out(DummyOutput):
        def get_size(self):
            return Size(rows=rows, columns=columns)

    class _Sized(TerminalAsker):
        def _ask(self, question):
            question.application.after_render.add_handler(lambda app: seen(app))
            return super()._ask(question)

    frames: list[list[str]] = []
    timed_out: list[bool] = []
    choices = [Choice(f"m{number}", f"model-{number:02d}  fits") for number in range(count)]
    with create_pipe_input() as pipe:

        def seen(app) -> None:
            screen = app.renderer._last_screen
            if screen is None:
                return
            buffer = {y: "".join(row[x].char for x in sorted(row)) for y, row in screen.data_buffer.items()}
            frames.append([buffer[y].rstrip() for y in sorted(buffer)])
            if len(frames) == 1:
                pipe.send_text(DOWN * (count - 1))
            elif any(f"model-{count - 1:02d}" in line for line in frames[-1]):
                pipe.send_text(ENTER)

        def expire() -> None:
            timed_out.append(True)
            pipe.close()

        watchdog = threading.Timer(20.0, expire)
        watchdog.start()
        try:
            _Sized(input=pipe, output=_Out()).checkbox("select", "Which?", choices, legend, header=header, notice=fit)
        finally:
            watchdog.cancel()
        assert not timed_out
    return frames


@pytest.mark.parametrize("columns", [80, 100])
def test_the_fit_line_is_in_sight_at_the_start_and_at_the_end_of_a_list_longer_than_the_window(columns):
    from test_guided_models import _measured

    from modelroom.guided_models import fit_line, hint_line, list_header

    fit = fit_line(_measured(), 131072, 250, context_kept=True, requests_kept=True)
    frames = _list_frames(columns, 30, 60, fit, hint_line(_measured()), list_header())

    flat = "".join(fit.split())
    start, end = frames[0], frames[-1]
    assert "".join("".join(line.split()) for line in start).count(flat) == 1  # whole, once
    assert "".join("".join(line.split()) for line in end).count(flat) == 1
    assert any("model-59" in line for line in end) and not any("model-00" in line for line in end)  # it did scroll
    assert all(len(line) <= columns for frame in (start, end) for line in frame)
    # The fit line stands between the keys and the column head, which is the row above the list.
    rows = [line for line in end if line.strip()]
    keys_at = next(index for index, line in enumerate(rows) if line.startswith("? Which?"))
    fit_at = next(index for index, line in enumerate(rows) if line.strip().startswith("fit from size"))
    head_at = next(index for index, line in enumerate(rows) if "Model" in line and "Fit" in line)
    assert keys_at < fit_at < head_at
    assert rows[fit_at].startswith(" " * 5 + "fit from size")  # the indent of the column head
