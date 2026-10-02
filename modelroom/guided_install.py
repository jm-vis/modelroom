"""Step 5, its last question: pull a package of this machine's table into the local Ollama daemon.

The README promised it for the release after 0.1.0 (decided 2026-09-25): the run offers to pull a
package instead of only printing the command -- and the command stays for anyone who prefers to
paste it. `guided.py` calls `pull_step` once the render has drawn its card.

Since AP9-UX2 (U6, GF F4) the user chooses the package: a list of the rows shown for this machine
(`pull_choice.pull_rows`), the command of the choice, a check of the disk (U8), then the question.
An answer file without `pull_package` takes `#1` where it has a pull name of its own, as before.
Three conditions, all of them: a document was written, a shown row of this machine has an Ollama
name, and the start screen knew the daemon as reachable. A row the daemon provably has already --
by the digest rule of the load test, `loadtest.installed_candidates` -- is a note, not a question.
The pull changes the daemon and nothing else: no configuration, no state, no document. What it
does not check -- whose daemon answers, another process pulling the same name -- is CONTRACTS.md,
"Guided mode", "The pull".
"""

from __future__ import annotations

import json
import shutil
import sys
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Sequence, TextIO

from .contracts import InstalledModel, Package
from .daemon import PULL_PATH, Daemon, DaemonError
from .dialog import AnswerMissingError, Asker, Choice, FileAsker
from .document import RenderDocument
from .http import Response
from .intro import label_line
from .loadtest import installed_candidates, installed_models, weight_digest
from .pull_choice import KEY as CHOICE_KEY
from .pull_choice import LOCAL_ALREADY, NONE_VALUE, QUESTION as CHOICE_QUESTION
from .pull_choice import PullRow, default_row, package_of, pull_rows, weights_bytes_of
from .quantization import package_identity_key
from .room import Disk, free_disk, gb, pull_need_bytes
from .install_vllm import install_block_lines, vllm_install
from .runtimes import OLLAMA, VLLM, runtime_of
from .screen import PULL_CANCELED as CANCELED_LINE
from .screen import Screen
from .screen import PULL_LABEL, install_local_note, install_pull_line, plain, pull_progress_text, pull_question
from .screen import DEFAULT_PULLED_HINT, pulled_line, yes_no
from .views import INSTALL_LABEL, install_target

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

KEY = "pull"
# The answer file's table of keys; the question on the screen carries the row and its size as well.
QUESTIONS: dict[str, str] = {CHOICE_KEY: CHOICE_QUESTION, KEY: "Pull #1 into Ollama now?"}
NONE_WORDS = "none, do not pull"
DISK_UNKNOWN = "disk space where Ollama keeps its models is not known"
NOT_LISTED = "pulled, but the daemon does not list it"
_BEFORE_SUCCESS = "the daemon ended its answer before success"
_MAX_COUNT = 2**63


@dataclass(frozen=True)
class FirstRow:
    """The row a pull is about: its rank, its local Ollama name, its weight, and its package.

    `rank` and `weights_gib` are `None` outside a table row (`modelroom pull`, A4: no card, no
    table, and a size only where `--weights-bytes` gave one). `package` is `None` there too -- A4
    knows no snapshot, so `pulled_problem` checks no manifest digest for it (Codex, "ohne
    Digestteil"). `weights_bytes` is what the disk check (`_disk_allows`) measures against: the
    guided run computes it from `package.files` (`first_row`, `pull_step`), A4 takes it from
    `--weights-bytes` directly, with no package to compute it from.
    """

    rank: int | None
    name: str
    weights_gib: float | None
    package: Package | None = None
    weights_bytes: float | None = None
    # The install line of the row's runtime (`Runtime.command`); `None` is Ollama's, `ollama pull <name>`.
    command: str | None = None

    @property
    def needed_bytes(self) -> float | None:
        """`weights_bytes` where it was given; else computed from `package.files` where there is
        one (Codex r1-4: an older `FirstRow(rank, name, weights_gib, package)` call, with no fifth
        argument, still gets a real disk check instead of silently checking against `0`); `None`
        with neither -- A4 without `--weights-bytes` and without a package."""
        if self.weights_bytes is not None:
            return self.weights_bytes
        return weights_bytes_of(self.package) if self.package is not None else None


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
        return None if package is None else FirstRow(entry.rank, name, entry.weights_gib, package, weights_bytes_of(package))
    return None


@dataclass(frozen=True)
class PullTarget:
    """What a pull draws on, whether it is the last question of a guided run or `modelroom pull`
    (A4, no results folder, no configuration) -- the seven fields `pull_chosen`, `_disk_allows`,
    `_is_local`, `_holds`, `_pull` and `_send` read; no helper reads a `GuidedRun` any more.

    `disk` is measured again right when it is checked (U8): a callable, not a value, so a target
    built once still reads the free disk of the moment it is asked. `where` names the machine a
    message is not about this one (the bare name, `"gpu1"`, FR-B1) -- `pull_chosen`, `_disk_allows`,
    `pull_question`, `install_local_note` and the success line all read it to say "on gpu1"/"onto
    gpu1" instead of "into Ollama"/"this machine"; the guided run's own row and A4 both say nothing
    extra (`""`, the default). `key` is the answer file's key for the yes/no question (`_wants_pull`):
    `"pull"` (`KEY`, the default) for this machine's own row and A4, `"pull_remote"` for FR-B1's --
    two distinct keys, since a run can ask both in the same answer file.
    """

    daemon: Daemon
    disk: Callable[[], Disk]
    now: datetime
    screen: Screen
    asker: Asker | None
    failed: Callable[[str], None]
    where: str = ""
    key: str = KEY


def pull_target_from_run(run: "GuidedRun") -> PullTarget:
    """A `PullTarget` out of a guided run's own fields -- `pull_chosen` builds one where it is
    handed a `GuidedRun` instead (`test_pull_target_carries_every_field_the_helpers_read`)."""
    return PullTarget(
        daemon=run.daemon, disk=lambda: free_disk(run.room), now=run.now, screen=run.screen, asker=run.asker, failed=run.failed
    )


@dataclass(frozen=True)
class PullContext:
    """What the pull of step 5 chooses from: the document, this machine, the run's packages (before any
    filter) and the pull names no row may claim (`pull_choice.pull_collisions`)."""

    document: RenderDocument
    machine: str
    packages: Sequence[Package]
    ambiguous: frozenset[str] = frozenset()
    # The free disk of this machine as the head of step 5 read it (AP9-RM U11): a row whose download
    # does not fit it cannot be chosen. The check before the pull reads the disk again all the same.
    disk: Disk | None = None


def pull_step(run: "GuidedRun", context: PullContext | None, reachable: bool) -> None:
    """The list of the rows shown, the command of the choice, the disk, the question; a fault is `run.failed`."""
    if context is None:
        return
    rows = _offerable(pull_rows(context.document, context.machine, context.ambiguous, run.screen.glyphs.dot, packages=context.packages), reachable)
    if not any(row.name is not None for row in rows):
        return
    run.screen.blank()
    # What each runtime holds is read before the lock of the disk (U11c): a package held
    # already needs no disk, so it stays a choice and its note -- in the table as in this list.
    held = held_by_runtime(run, rows, context.packages)
    # The disk carries what the Ollama daemon holds and counts for Ollama's rows alone; each other
    # runtime's rows count only what that runtime holds (`pull_choice.runtime_disk`).
    ollama = held.get(OLLAMA.key, frozenset())
    disk = None if context.disk is None else replace(context.disk, held=context.disk.held | ollama)
    rows = _offerable(pull_rows(context.document, context.machine, context.ambiguous, run.screen.glyphs.dot, disk, context.packages, held), reachable)
    locals_ = {(row.runtime, row.identity) for row in rows if row.locked is None and row.identity in held[row.runtime]}
    row = _choose(run, rows, locals_)
    if row is None:
        return
    package = package_of(row, context.packages)
    if package is None:
        return
    if row.runtime == VLLM.key:
        _show_vllm(run, context, row, package, disk, (row.runtime, row.identity) in locals_)
        return
    command = runtime_of(row.runtime).command(str(row.name))
    first = FirstRow(row.rank, str(row.name), row.weights_gib, package, weights_bytes_of(package), command)
    pull_chosen(run, first, local=(row.runtime, row.identity) in locals_)


def _offerable(rows: Sequence[PullRow], reachable: bool) -> list[PullRow]:
    """The rows the list may offer: Ollama's only where the start screen reached its daemon, vLLM's
    always -- modelroom talks to no vLLM to print its command, so a machine with vLLM alone is asked too."""
    return [row for row in rows if reachable or row.runtime != OLLAMA.key]


def _show_vllm(run: "GuidedRun", context: PullContext, row: PullRow, package: Package, disk: Disk | None, local: bool) -> None:
    """A chosen vLLM row: its install lines, nothing else -- no yes/no question and no pull, modelroom
    starts no server (GF 2026-09-29, F1 a). The same block the table's install line draws."""
    block = next(block for block in context.document.machines if block.machine == context.machine)
    entry = next(entry for entry in block.ranked if entry.rank == row.rank)
    relative = disk is not None and disk.vllm is not None and disk.vllm.relative
    install = vllm_install(package, entry.fit, block.profile, relative_cache=relative)
    served = run.servers.found.get("vllm")
    for line in install_block_lines(INSTALL_LABEL, row.rank, install, repo=row.name, served=served, on_disk=local):
        run.screen.write(line)


def held_by_runtime(
    run: "GuidedRun", rows: Sequence[PullRow], packages: Sequence[Package]
) -> dict[str, frozenset[tuple[str, str, str]]]:
    """By runtime key, the identities of the rows' packages that runtime provably holds: each read
    by that runtime's own `on_disk`, over its own rows only -- one runtime's inventory never makes
    another runtime's row "local already" or frees it from the disk check."""
    held: dict[str, frozenset[tuple[str, str, str]]] = {}
    for key in dict.fromkeys(row.runtime for row in rows):
        own = [package for package in (package_of(row, packages) for row in rows if row.runtime == key) if package]
        held[key] = runtime_of(key).on_disk(run, own)
    return held


def other_runtimes_held(run: "GuidedRun", context: PullContext) -> dict[str, frozenset[tuple[str, str, str]]]:
    """What each runtime but Ollama holds of this machine's rows, by its own `on_disk` -- for the
    install line under the table (`views.RunView.held`). Ollama's own holding is the disk read
    before the render (`Disk.held`) already, so its daemon is not asked again here."""
    rows = pull_rows(context.document, context.machine, packages=context.packages)
    return held_by_runtime(run, [row for row in rows if row.runtime != OLLAMA.key], context.packages)


def pull_chosen(target: "PullTarget | GuidedRun", first: FirstRow, *, local: bool | None = None) -> None:
    """The chosen row: a note where it is local, else its command, the disk check (U8), the question.

    `target` is a `PullTarget`, or a `GuidedRun` (the guided run's own call, and every test that
    still builds one directly): a `GuidedRun` carries the same seven fields under the same names
    except `disk`, which it has as `room` instead -- `pull_target_from_run` bridges that one name.
    """
    target = target if isinstance(target, PullTarget) else pull_target_from_run(target)
    if _is_local(target, first) if local is None else local:
        target.screen.write(label_line(INSTALL_LABEL, install_local_note(first.rank, target.screen.glyphs.dot, target.where)))
        return
    target.screen.write(label_line(INSTALL_LABEL, install_pull_line(first.rank, first.name, target.where, first.command)))
    if not _disk_allows(target, first):
        return
    question = pull_question(first.rank, first.weights_gib, first.name, target.where)
    wants = _wants_pull(target, question)
    target.screen.answer(question, yes_no(wants))
    if wants:
        _pull(target, first, live_line(target))


def _disk_allows(target: PullTarget, first: FirstRow) -> bool:
    """Measured again now (U8): enough disk asks; too little says so and asks nothing; unknown asks
    with a note. `first.needed_bytes` is `None` only where A4 was given no `--weights-bytes` at
    all and no package (`cli_pull.py` skips this call then; its own note covers that case).
    `target.where` (FR-B1): `not enough disk on gpu1 for #3: ... free`, the machine named once, up
    front, never repeated as `disk.drive` at the end the way this machine's own row says it."""
    disk = target.disk()
    if not disk.known:
        target.screen.note(DISK_UNKNOWN)
        return True
    need = pull_need_bytes(first.needed_bytes or 0.0)
    if disk.free_bytes >= need:
        return True
    which = f" for #{first.rank}" if first.rank is not None else ""
    if target.where:
        said = f"not enough disk on {target.where}{which}: needs {gb(need)}, {gb(disk.free_bytes)} free"
    else:
        said = f"not enough disk{which}: needs {gb(need)}, {gb(disk.free_bytes)} free on {disk.drive}"
    target.screen.write(label_line(INSTALL_LABEL, [("class:note", said)]))
    return False


def _choose(run: "GuidedRun", rows: Sequence[PullRow], locals_: set[tuple[str, tuple[str, str, str]]]) -> PullRow | None:
    """The row the user chose from the list, or `None` for `none`. An answer file without the key takes
    `#1` where it has a pull name of its own, else nothing (the behavior before U6)."""
    # The one question of the list, asked "installed" once a vLLM row stands in it (I4); Ollama's own
    # words, unchanged, where only its rows do.
    question = VLLM.words.question if any(row.runtime != OLLAMA.key for row in rows) else CHOICE_QUESTION
    try:
        value = run.asker.select(CHOICE_KEY, question, pull_choices(rows, locals_))
    except AnswerMissingError:
        return default_row(rows)
    chosen = next((row for row in rows if row.locked is None and row.value == value), None)
    run.screen.answer(question, NONE_WORDS if chosen is None else f"#{chosen.rank}  {chosen.name}")
    return chosen


def pull_choices(rows: Sequence[PullRow], locals_: set[tuple[str, tuple[str, str, str]]]) -> list[Choice]:
    """The entries of the pull list: one per row, then `none`. A row too large for the free disk is red,
    says what it needs in place of its name and cannot be taken; the pointer starts on the first row
    that can, else on `none` (AP9-RM U11). `locals_` holds `(runtime, identity)` of every row its own
    runtime holds already: "local already" is a fact of one runtime's inventory, not of the package."""
    choices = [
        Choice(row.value, _row_words(row, (row.runtime, row.identity) in locals_), disabled=row.locked, warn=row.over_disk)
        for row in rows
    ]
    return choices + [Choice(NONE_VALUE, NONE_WORDS)]


def _row_words(row: PullRow, local: bool) -> str:
    """`#1  unsloth · Q4_K_S  5.0 GB  hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_S`, the name last and whole;
    a row too large for the disk says `needs 94.1 GB plus 1 GB room, 26.4 GB free` in its place."""
    words = f"#{row.rank}  {row.cell}  {row.weights_gib:.1f} GB  {row.locked if row.over_disk else row.name or ''}".rstrip()
    return f"{words}  {LOCAL_ALREADY}" if local else words


def _holds(target: "PullTarget", installed: Sequence[InstalledModel] | None, package: Package | None, show=None) -> bool:
    """Whether the daemon's inventory provably holds `package` (the load test's digest rule).

    `ollama_held` always passes its own `show`, so this never reads `target.daemon` there -- a
    plain `GuidedRun` satisfies it all the same, since the attribute is named the same on both.
    """
    if installed is None or package is None:
        return False
    look = show if show is not None else (lambda name: weight_digest(target.daemon, name))
    return bool(installed_candidates([package], list(installed), look).candidates)


def ollama_held(target: "PullTarget | GuidedRun", packages: Sequence[Package]) -> frozenset[tuple[str, str, str]]:
    """`runtimes.OLLAMA.on_disk`: the identities of those `packages` the daemon provably holds (one
    look at `/api/tags` and at most one `/api/show` per name); empty where the inventory cannot be
    read (U11c). `target` is anything with the daemon and the clock (`daemon`, `now`)."""
    installed, _reason = installed_models(target.daemon, target.now)
    if installed is None:
        return frozenset()
    seen: dict[str, tuple[str | None, str | None]] = {}

    def show(name: str) -> tuple[str | None, str | None]:
        if name not in seen:
            seen[name] = weight_digest(target.daemon, name)
        return seen[name]

    return frozenset(package_identity_key(package) for package in packages if _holds(target, installed, package, show))


def held_identities(run: "GuidedRun", packages: Sequence[Package]) -> frozenset[tuple[str, str, str]]:
    """What the Ollama daemon provably holds of `packages`, read through the table (`OLLAMA.on_disk`)."""
    return OLLAMA.on_disk(run, packages)


def _is_local(target: "PullTarget", first: FirstRow) -> bool:
    """Whether the daemon provably holds this package: the load test's rule, digest and not name.

    A registry package by its manifest digest, a `hf.co/…` name by its name **and** the digest of
    its weight file (`/api/show`). A name the daemon lists without that proof is asked like a
    missing one: the pull brings the registry's build of that name, which for Ollama is an update.
    Read through the table (`OLLAMA.on_disk`); the inventory is read even without a package.
    """
    packages = [first.package] if first.package is not None else []
    return first.package is not None and package_identity_key(first.package) in OLLAMA.on_disk(target, packages)


def _wants_pull(target: "PullTarget", question: str) -> bool:
    """The `target.key` answer (`"pull"` here, `"pull_remote"` for FR-B1), with `no` for an answer
    file that does not mention it.

    Optional like `load_test`, for the same reason: whether this question is reached depends on
    the machine -- its daemon, and what the daemon already holds -- not on the answer file.
    """
    try:
        return target.asker.confirm(target.key, question, default=False)
    except AnswerMissingError:
        return False


def _pull(target: "PullTarget", first: FirstRow, live: "LiveLine | None") -> None:
    """Pull, then read `/api/tags` once to see the name there; every fault is one line."""
    target.screen.write(label_line(PULL_LABEL, [("", first.name)]))
    try:
        answer = _send(target, first, live)
    except DaemonError as exc:
        target.failed(_unfinished(first, str(exc)))
        return
    except KeyboardInterrupt:
        # The central handler says that the run was stopped and ends with 130; this line says what
        # happens to the part of the package that already arrived.
        target.screen.note(CANCELED_LINE)
        raise
    reason = _end_reason(answer)
    if reason is not None:
        target.failed(_unfinished(first, reason))
        return
    installed, why = installed_models(target.daemon, target.now)
    problem = f"{first.name}: pulled, but its list did not arrive ({one_line(str(why))})" if installed is None else None
    problem = problem or pulled_problem(installed or [], first)
    if problem is not None:
        target.failed(problem)
        return
    hint = f"on {target.where}" if target.where else DEFAULT_PULLED_HINT
    target.screen.write(pulled_line(first.name, first.weights_gib, target.screen.glyphs, hint=hint))


def _send(target: "PullTarget", first: FirstRow, live: "LiveLine | None") -> Response:
    """The pull itself, with the live line erased however it ends -- before anything else is said."""

    def follow(line: dict) -> None:
        _check_line(line)
        if live is not None:
            live.show(plain(label_line(PULL_LABEL, [("", one_line(pull_progress_text(line)))])))

    try:
        return target.daemon("POST", PULL_PATH, {"model": first.name, "stream": True}, progress=follow)
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
    digest of the GGUF file, so for those the name is what can be checked here. Without a package
    (`modelroom pull`, A4) there is no manifest digest to compare against, so only the name is
    checked ("ohne Digestteil", A4).
    """
    entry = next((model for model in installed if model.name == first.name), None)
    if entry is None:
        return f"{first.name}: {NOT_LISTED}"
    if first.package is not None and first.package.source == "ollama" and entry.digest != first.package.manifest_digest:
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


def live_line(target: "PullTarget", stream: TextIO | None = None) -> LiveLine | None:
    """A live line at a terminal; none under `--answers`, whose log keeps the first line and the last."""
    out = sys.stdout if stream is None else stream
    if isinstance(target.asker, FileAsker):
        return None
    try:
        return LiveLine(out) if out.isatty() else None
    except (AttributeError, ValueError):  # a closed or replaced stream
        return None
