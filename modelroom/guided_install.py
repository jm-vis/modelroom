"""Step 5, its last question: pull the first row of this machine into the local Ollama daemon.

The README promised it for the release after 0.1.0 (decided 2026-09-25): the run offers to pull
the package its install line names, instead of only printing the command -- and the command stays
for anyone who prefers to paste it. `guided.py` calls `pull_step` once the render has drawn its
card; the choice of the row is `views.install_target`'s, the one the install line stands on.

Three conditions, all of them: a document was written, this machine's first row has an Ollama
name, and the start screen knew the daemon as reachable. A first row the daemon provably has
already -- by the digest rule of the load test, `loadtest.installed_candidates` -- is a note, not
a question. The pull changes the daemon and nothing else: no configuration, no state, no document.
What it does not check -- disk space, whose daemon answers, another process pulling the same name
-- is CONTRACTS.md, "Guided mode", "The pull".
"""

from __future__ import annotations

import json
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, TextIO

from .contracts import InstalledModel, Package
from .daemon import PULL_PATH, DaemonError
from .dialog import AnswerMissingError, FileAsker
from .document import RenderDocument
from .http import Response
from .intro import label_line
from .loadtest import installed_candidates, installed_models, weight_digest
from .quantization import package_identity_key
from .screen import PULL_CANCELED as CANCELED_LINE
from .screen import PULL_LABEL, install_local_note, plain, pull_progress_text, pull_question, pulled_line, yes_no
from .views import INSTALL_LABEL, install_target

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

KEY = "pull"
# The answer file's table of keys; the question on the screen carries the size of the row as well.
QUESTIONS: dict[str, str] = {KEY: "Pull #1 into Ollama now?"}
NOT_LISTED = "pulled, but the daemon does not list it"
_BEFORE_SUCCESS = "the daemon ended its answer before success"
_MAX_COUNT = 2**63


@dataclass(frozen=True)
class FirstRow:
    """The row a pull is about: its rank, its local Ollama name, its weight, and its package."""

    rank: int
    name: str
    weights_gib: float
    package: Package


def first_row(document: RenderDocument, machine: str, packages: Sequence[Package]) -> FirstRow | None:
    """This machine's first row with its Ollama name and its package from the snapshot, or `None`.

    A `RankedEntry` carries the package's identity and no digest, so the package is found by that
    identity (`package_identity_key`, the key the render builds the row with).
    """
    for block in document.machines:
        target = install_target(block, machine)
        if target is None:
            continue
        entry, name = target
        wanted = tuple(entry.package_identity)
        package = next((package for package in packages if package_identity_key(package) == wanted), None)
        return None if package is None else FirstRow(entry.rank, name, entry.weights_gib, package)
    return None


def pull_step(run: "GuidedRun", first: FirstRow | None, reachable: bool) -> None:
    """Ask for the pull of the first row, and pull it on a yes; a fault is `run.failed`, exit `1`."""
    if first is None or not reachable:
        return
    run.screen.blank()
    if _is_local(run, first):
        run.screen.write(label_line(INSTALL_LABEL, install_local_note(first.rank, run.screen.glyphs.dot)))
        return
    question = pull_question(first.rank, first.weights_gib)
    wants = _wants_pull(run, question)
    run.screen.answer(question, yes_no(wants))
    if wants:
        _pull(run, first, live_line(run))


def _is_local(run: "GuidedRun", first: FirstRow) -> bool:
    """Whether the daemon provably holds this package: the load test's rule, digest and not name.

    A registry package by its manifest digest, a `hf.co/…` name by its name **and** the digest of
    its weight file (`/api/show`). A name the daemon lists without that proof is asked like a
    missing one: the pull brings the registry's build of that name, which for Ollama is an update.
    """
    installed, _reason = installed_models(run.daemon, run.now)
    if installed is None:
        return False
    inventory = installed_candidates([first.package], installed, lambda name: weight_digest(run.daemon, name))
    return bool(inventory.candidates)


def _wants_pull(run: "GuidedRun", question: str) -> bool:
    """The `pull` answer, with `no` for an answer file that does not mention it.

    Optional like `load_test`, for the same reason: whether this question is reached depends on
    the machine -- its daemon, and what the daemon already holds -- not on the answer file.
    """
    try:
        return run.asker.confirm(KEY, question, default=False)
    except AnswerMissingError:
        return False


def _pull(run: "GuidedRun", first: FirstRow, live: "LiveLine | None") -> None:
    """Pull, then read `/api/tags` once to see the name there; every fault is one line."""
    run.screen.write(label_line(PULL_LABEL, [("", first.name)]))
    try:
        answer = _send(run, first, live)
    except DaemonError as exc:
        run.failed(_unfinished(first, str(exc)))
        return
    except KeyboardInterrupt:
        # The central handler says that the run was stopped and ends with 130; this line says what
        # happens to the part of the package that already arrived.
        run.screen.note(CANCELED_LINE)
        raise
    reason = _end_reason(answer)
    if reason is not None:
        run.failed(_unfinished(first, reason))
        return
    installed, why = installed_models(run.daemon, run.now)
    problem = f"{first.name}: pulled, but its list did not arrive ({one_line(str(why))})" if installed is None else None
    problem = problem or pulled_problem(installed or [], first)
    if problem is not None:
        run.failed(problem)
        return
    run.screen.write(pulled_line(first.name, first.weights_gib, run.screen.glyphs))


def _send(run: "GuidedRun", first: FirstRow, live: "LiveLine | None") -> Response:
    """The pull itself, with the live line erased however it ends -- before anything else is said."""

    def follow(line: dict) -> None:
        _check_line(line)
        if live is not None:
            live.show(plain(label_line(PULL_LABEL, [("", one_line(pull_progress_text(line)))])))

    try:
        return run.daemon("POST", PULL_PATH, {"model": first.name, "stream": True}, progress=follow)
    finally:
        if live is not None:
            live.clear()


def _check_line(line: dict) -> None:
    """A line of a pull is a status with optional counts, or an error; anything else is a fault.

    The counts are checked on every line, an error line included: the live line formats whatever
    a line carries, and a count that is no number must not reach it (second-model round, 2026-09-25).
    """
    counts = all(_is_count(line[key]) for key in ("total", "completed") if key in line)
    said = isinstance(line["error"], str) if "error" in line else isinstance(line.get("status"), str)
    if not (counts and said):
        raise DaemonError(f"unexpected line in the daemon's answer: {json.dumps(line)[:200]}")


def _is_count(value: object) -> bool:
    """A byte count a daemon can mean: a whole number from 0 up to what a 64-bit counter holds."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < _MAX_COUNT


def one_line(text: str) -> str:
    """A text of the daemon's as one line: every control character, line breaks included, a space."""
    return "".join(" " if not character.isprintable() else character for character in text)


def _end_reason(answer: Response) -> str | None:
    """Why the pull did not end in `success`, from its status and its last line, or `None`."""
    try:
        payload = json.loads(answer.body) if answer.body else {}
    except (ValueError, RecursionError):
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    if answer.status != 200:
        return f"status {answer.status}" + (f" ({error})" if isinstance(error, str) else "")
    if isinstance(error, str):
        return error
    return None if payload.get("status") == "success" else _BEFORE_SUCCESS


def _unfinished(first: FirstRow, reason: str) -> str:
    return f"the pull of {first.name} did not finish: {one_line(reason)}"


def pulled_problem(installed: Sequence[InstalledModel], first: FirstRow) -> str | None:
    """Whether the daemon now lists what was pulled: by name, and a registry package by its digest.

    The manifest digest `/api/tags` shows for a `hf.co/…` name is Ollama's own manifest, not the
    digest of the GGUF file, so for those the name is what can be checked here.
    """
    entry = next((model for model in installed if model.name == first.name), None)
    if entry is None:
        return f"{first.name}: {NOT_LISTED}"
    if first.package.source == "ollama" and entry.digest != first.package.manifest_digest:
        return f"{first.name}: pulled, but the daemon lists it with another manifest than the fetch recorded"
    return None


class LiveLine:
    """One line at a terminal, written again in place while a pull runs, and erased when it ends.

    Cut to one column less than the terminal is wide: a line that wraps cannot be written again
    in place, and `\\r` would then erase only its last part.
    """

    def __init__(self, stream: TextIO, columns: int | None = None) -> None:
        self._stream = stream
        self._columns = columns if columns is not None else shutil.get_terminal_size().columns
        self._width = 0

    def show(self, text: str) -> None:
        text, width = _cut(text, max(self._columns - 1, 1))
        self._stream.write("\r" + text + " " * max(self._width - width, 0))
        self._width = max(self._width, width)
        self._stream.flush()

    def clear(self) -> None:
        if self._width:
            self._stream.write("\r" + " " * self._width + "\r")
            self._stream.flush()
            self._width = 0


def _cut(text: str, columns: int) -> tuple[str, int]:
    """`text` cut to `columns` terminal columns, and how many it takes: a wide character takes two."""
    taken = 0
    for index, character in enumerate(text):
        width = 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1
        if taken + width > columns:
            return text[:index], taken
        taken += width
    return text, taken


def live_line(run: "GuidedRun", stream: TextIO | None = None) -> LiveLine | None:
    """A live line at a terminal; none under `--answers`, whose log keeps the first line and the last."""
    target = sys.stdout if stream is None else stream
    if isinstance(run.asker, FileAsker):
        return None
    try:
        return LiveLine(target) if target.isatty() else None
    except (AttributeError, ValueError):  # a closed or replaced stream
        return None
