"""`modelroom pull <name>`: the command the handover line of `pull_other` names (A3), run on the
machine that should get the package -- no results folder, no configuration, just this machine's
Ollama daemon and disk (A4, AP9-FR).

Six steps, in order, each ending the command at the first one that fails: the name against
`pull_choice.PULL_NAME_RE` (A2, checked again here -- this line came from a paste, not from a search
this package ran itself), the local daemon reachable, a note where it already lists the name (no
digest to prove it, unlike the guided run's own check), the free disk against `--weights-bytes`, the
question (a terminal or `--yes`), the pull itself -- reusing the guided run's own
`_disk_allows`/`_pull` through a `PullTarget` built from this machine alone (`guided_install.py`;
`test_pull_target_carries_every_field_the_helpers_read`).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Callable

from .daemon import VERSION_PATH, Daemon, DaemonError, LocalDaemon
from .dialog import Asker, Canceled, TerminalAsker, is_interactive
from .guided_install import FirstRow, PullTarget, _disk_allows, _pull, live_line
from .loadtest import installed_models
from .pull_choice import PULL_NAME_RE
from .room import RoomProbes, default_room_probes, free_disk
from .screen import Screen, pull_question, yes_no

KEY = "pull"
NOT_SAFE = "is not a name this command pulls (letters, digits, `.`, `_`, `-` only, no `..`)"
NOT_REACHABLE = "Ollama does not answer on this computer at 127.0.0.1:11434"
ALREADY_LISTED = "Ollama lists {name} already; pulling it again updates it"
UNKNOWN_SIZE = "the size of this download is not known; the disk is not checked before it"
NO_TERMINAL = "no terminal to ask; add --yes"
_GIB = 1024**3


class _NoTerminal(Exception):
    """No terminal to ask the question at, and no `--yes` either."""


def run_pull(
    name: str,
    weights_bytes: float | None = None,
    yes: bool = False,
    *,
    daemon: Daemon | None = None,
    room: RoomProbes | None = None,
    now: datetime | None = None,
    out: Callable[[str], None] | None = None,
    asker: Asker | None = None,
) -> int:
    """`modelroom pull`: `0` pulled or the user said no, `1` a daemon/disk/pull fault, `2` an
    invalid name or no way to ask, `130` Ctrl-C. `asker` is dependency injection for a test
    (`tests/test_cli_pull.py`, a `TerminalAsker` over a pipe): the real CLI never passes it, and
    without one a terminal asks, or `--yes` skips the question, or there is no way to ask at all.

    Ctrl-C covers the whole of it (Codex r1-3): version check, the tags note, the disk check, the
    question and the pull itself all run under the one `try` below, so a `KeyboardInterrupt` in any
    of them -- not only the dialog and the pull -- ends the command the same way, exit `130`.
    """
    try:
        return _run_checked(name, weights_bytes, yes, daemon, room, now, out, asker)
    except KeyboardInterrupt:
        print("stopped at your request; nothing was pulled", file=sys.stderr)
        return 130


def _run_checked(
    name: str,
    weights_bytes: float | None,
    yes: bool,
    daemon: Daemon | None,
    room: RoomProbes | None,
    now: datetime | None,
    out: Callable[[str], None] | None,
    asker: Asker | None,
) -> int:
    """Steps 1-2 of `run_pull`, under its one Ctrl-C boundary."""
    if PULL_NAME_RE.fullmatch(name) is None:
        print(f"{name!r} {NOT_SAFE}", file=sys.stderr)
        return 2
    active_daemon = daemon if daemon is not None else LocalDaemon()
    try:
        active_daemon("GET", VERSION_PATH)
    except DaemonError:
        print(NOT_REACHABLE, file=sys.stderr)
        return 1
    return _pull_checked(name, weights_bytes, yes, active_daemon, room, now, out, asker)


def _pull_checked(
    name: str,
    weights_bytes: float | None,
    yes: bool,
    daemon: Daemon,
    room: RoomProbes | None,
    now: datetime | None,
    out: Callable[[str], None] | None,
    asker: Asker | None,
) -> int:
    """Steps 3-6 of `run_pull`: the daemon is already known to answer."""
    active_now = now or datetime.now(timezone.utc)
    screen = Screen(out if out is not None else print)
    installed, _reason = installed_models(daemon, active_now)
    if installed is not None and any(model.name == name for model in installed):
        screen.note(ALREADY_LISTED.format(name=name))
    active_room = room if room is not None else default_room_probes()
    weights_gib = weights_bytes / _GIB if weights_bytes is not None else None
    first = FirstRow(rank=None, name=name, weights_gib=weights_gib, weights_bytes=weights_bytes)
    problems: list[str] = []

    def failed(message: str) -> None:
        problems.append(message)
        print(message, file=sys.stderr)

    target = PullTarget(daemon=daemon, disk=lambda: free_disk(active_room), now=active_now, screen=screen, asker=asker, failed=failed)
    if weights_bytes is None:
        screen.note(UNKNOWN_SIZE)
    elif not _disk_allows(target, first):
        return 1
    return _ask_and_pull(target, first, weights_gib, name, yes, asker, problems)


def _ask_and_pull(
    target: PullTarget, first: FirstRow, weights_gib: float | None, name: str, yes: bool, asker: Asker | None, problems: list[str]
) -> int:
    """Step 5 (the question) and step 6 (the pull itself) of `run_pull`. A `KeyboardInterrupt` from
    either -- the dialog or `_pull` -- is not caught here: `run_pull`'s one `try` covers it, along
    with every step before this one (Codex r1-3)."""
    question = pull_question(None, weights_gib, name)
    try:
        wants = _confirm(question, yes, asker)
    except _NoTerminal:
        print(NO_TERMINAL, file=sys.stderr)
        return 2
    except Canceled as exc:
        print(str(exc), file=sys.stderr)
        return 2
    target.screen.answer(question, yes_no(wants))
    if not wants:
        return 0
    _pull(target, first, live_line(target))
    return 1 if problems else 0


def _confirm(question: str, yes: bool, asker: Asker | None) -> bool:
    """`--yes` skips the question outright; else the given `asker` asks it (a test's own), or a
    terminal (default No), or there is no way to: `_NoTerminal`."""
    if yes:
        return True
    active = asker if asker is not None else (TerminalAsker() if is_interactive() else None)
    if active is None:
        raise _NoTerminal()
    return active.confirm(KEY, question, default=False)


__all__ = ["ALREADY_LISTED", "NOT_REACHABLE", "NOT_SAFE", "NO_TERMINAL", "UNKNOWN_SIZE", "run_pull"]
