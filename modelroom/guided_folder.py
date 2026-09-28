"""Step 1 of the guided mode, first question: which results folder, asked where it is not plain.

Moved out of `modelroom/guided.py` (AP9-UX2, size guard): `_known_config`, `_ask_config_file`,
`_config_file`, `_looks_like_results_folder` and `_remember_folder`, with the folder constants.
This module never imports `guided`; `guided.py` exports back what tests read from there.

Two corrections (AP9-UX2 U1, U7; GF F1, F2):

- **U1**: a folder the pointer file remembers is no longer taken without a word when it is not the
  folder this run was started in. The question offers it first, with the pointer on it (F1), then
  this folder, then another path. Every comparison is on resolved paths; the same folder asks
  nothing, as before. `--config` with a file that exists asks nothing either.
- **U7**: before anything is written or migrated, a home folder or the top of a drive is a
  question: `Write the results into this folder anyway?` (No by default). No asks for a path, with
  `<home>/modelroom` offered (F2); at a terminal the question comes again as long as the answer is a
  home folder or a drive; an answer file that names one again ends the run (exit 2), since its
  answers never change (Codex r2-4).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .binding import read_pointer, write_pointer
from .dialog import AnswerInvalidError, Choice, FileAsker
from .screen import THIS_FOLDER, yes_no

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

CONFIG_NAME = "modelroom.toml"
STATE_FOLDER = "state"
MARKDOWN_PATH = ("docs", "models.md")
LAST_FOLDER = "the folder of the last run"
HOME_NOTE = "this is your home folder: modelroom.toml, docs and state would land in it"
ROOT_NOTE = "this is the top of a drive: modelroom.toml, docs and state would land in it"
HOME_AGAIN = "results_path names the home folder or the top of a drive again"
DEFAULT_SUBFOLDER = "modelroom"

QUESTIONS: dict[str, str] = {
    "results": "Where should results live?",
    "results_path": "Path to the results folder",
    "write_home": "Write the results into this folder anyway?",
}


def known_config(run: "GuidedRun", config_arg: Path | None) -> Path | None:
    """The configuration this run can already read before its first question, for the start screen.

    `--config` when it names a file, else the folder the pointer file remembers when it still
    holds a configuration. Neither is decided here -- `config_file` asks and says what it found;
    this is only what the start screen may already show.
    """
    if config_arg is not None and config_arg.is_file():
        return config_arg.resolve()
    current = read_pointer(run.pointer_path).current
    if current is None:
        return None
    candidate = Path(current) / CONFIG_NAME
    return candidate.resolve() if candidate.is_file() else None


def last_elsewhere(run: "GuidedRun", config_arg: Path | None, known: Path | None) -> bool:
    """Whether the start screen names a remembered folder that is not the one this run is in (U1)."""
    return config_arg is None and known is not None and known.parent != run.here.resolve()


def config_file(run: "GuidedRun", config_arg: Path | None) -> Path:
    """`--config` wins over the pointer file; a remembered folder that is not this one is a question."""
    if config_arg is not None:
        if config_arg.is_file():
            return config_arg.resolve()
        run.screen.note(f"{config_arg}: there is no configuration at this path")
        return ask_config_file(run)
    pointer = read_pointer(run.pointer_path)
    if pointer.current is not None:
        last = Path(pointer.current).resolve()
        if (last / CONFIG_NAME).is_file():
            if last == run.here.resolve():
                return (last / CONFIG_NAME).resolve()
            return ask_config_file(run, last)
        # Without the path in front of it: the answer to the question that follows says which
        # folder this run works in, and it said the same path twice (test round, 2026-09-24).
        run.screen.note(f"the results folder of the last run no longer holds a {CONFIG_NAME}")
    return ask_config_file(run)


def ask_config_file(run: "GuidedRun", last: Path | None = None) -> Path:
    """The configuration file the run works on, from the folder question; a home folder or a drive
    is a question of its own before anything is written (U7)."""
    here = run.here.resolve()
    choices = [Choice("here", f"{THIS_FOLDER} ({here})"), Choice("path", "another path")]
    if last is not None:
        # The pointer starts on the folder of the last run (F1): `checked` marks it in a `select`.
        choices.insert(0, Choice("last", f"{LAST_FOLDER} ({last})", checked=True))
    choice = run.asker.select("results", QUESTIONS["results"], choices)
    if choice == "last":
        folder = last
        run.screen.answer(QUESTIONS["results"], LAST_FOLDER)
    elif choice == "here":
        folder = here
        run.screen.answer(QUESTIONS["results"], THIS_FOLDER)
    else:
        folder = _asked_path(run)
    return _allowed_folder(run, folder) / CONFIG_NAME


def _asked_path(run: "GuidedRun", default: str = "") -> Path:
    answered = Path(run.asker.text("results_path", QUESTIONS["results_path"], default=default).strip()).expanduser()
    folder = (answered if answered.is_absolute() else run.here / answered).resolve()
    run.screen.answer(QUESTIONS["results"], str(folder), path=True)
    return folder


def _allowed_folder(run: "GuidedRun", folder: Path) -> Path:
    """The folder, once it is neither a home folder nor the top of a drive, or once that was confirmed."""
    home = Path.home().resolve()
    while home_or_root(folder, home):
        run.screen.note(HOME_NOTE if folder == home else ROOT_NOTE)
        anyway = run.asker.confirm("write_home", QUESTIONS["write_home"], default=False)
        run.screen.answer(QUESTIONS["write_home"], yes_no(anyway))
        if anyway:
            return folder
        folder = _asked_path(run, str(home / DEFAULT_SUBFOLDER))
        if isinstance(run.asker, FileAsker) and home_or_root(folder, home):
            # An answer file gives the same answer every time it is asked: asking again would not end.
            raise AnswerInvalidError(HOME_AGAIN)
    return folder


def home_or_root(folder: Path, home: Path) -> bool:
    """Whether `folder` is the home folder or the top of a drive (`folder.parent == folder`)."""
    resolved = folder.resolve()
    return resolved == home or resolved.parent == resolved


def looks_like_results_folder(folder: Path) -> bool:
    """Whether a folder already holds results, so writing a configuration into it is a question."""
    state = folder / STATE_FOLDER
    return state.is_dir() or (folder / MARKDOWN_PATH[0]).is_dir()


def remember_folder(run: "GuidedRun", folder: Path) -> None:
    """Record the results folder in the pointer file, so the next run finds it without asking.

    Written only when it really changes: the file is one per user for every results folder and is
    replaced as a whole, without a lock (the trade-off `cli._bind_this_machine` documents), so even
    a write that changes nothing could drop a binding a concurrent `hardware` run added in between.
    """
    pointer = read_pointer(run.pointer_path)
    if pointer.current == str(folder):
        return
    try:
        write_pointer(run.pointer_path, pointer.with_current(folder))
    except (OSError, ValueError) as exc:
        run.out(f"{run.pointer_path}: this folder could not be remembered ({exc}); the next run asks again")


__all__ = [
    "CONFIG_NAME",
    "LAST_FOLDER",
    "MARKDOWN_PATH",
    "QUESTIONS",
    "STATE_FOLDER",
    "ask_config_file",
    "config_file",
    "home_or_root",
    "known_config",
    "last_elsewhere",
    "looks_like_results_folder",
    "remember_folder",
]
