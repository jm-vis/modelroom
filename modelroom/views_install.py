"""The `install` line under a machine's table: the one line a reader copies to get a package.

Moved out of `views.py` (size guard, AP9-RM U11) and extended there: where the free disk of the
machine was read, the line takes the best row that also fits it (`disk_fit.best_on_disk`) and says
under it which rows were passed over, and where no row fits it says so in red instead of offering a
command. `views.py` re-exports `INSTALL_LABEL` and `install_target`.

`install_target` reads its name off `runtimes.accepting_on_machine`/`runtime.command` where a
caller passes `packages` (AP9-SV2-L, Codex r1 fix 4) -- `views.py` does not yet (it is unchanged,
outside this sub-package's file ownership), so every call from the shipped run keeps taking the
`entry_pull_name` fallback below, byte for byte; only a caller built for this (`tests/test_runtimes.py`,
the third-entry probe) exercises the table-dispatched branch. The command's own words ("ollama
pull …") stay Ollama's: AP9-SV2-I gives a non-Ollama runtime's install line its own verb.
"""

from __future__ import annotations

from typing import Sequence

from . import runtimes
from .contracts import Package
from .disk_fit import best_on_disk, known, none_fits_words, skipped_words
from .document import MachineRanking, RankedEntry
from .intro import label_line
from .pull_choice import entry_pull_name
from .quantization import package_identity_key
from .room import Disk
from .screen import Line, install_pull_line

INSTALL_LABEL = "install"
OTHER_INSTALL_NOTE = "run it on that computer; this run pulls only onto this one"
# A3 (AP9-FR): said only for a block in `pull_other_machines` -- the guided run passes exactly the
# machines `pull_other_step` will list a row for (`pull_other.eligible_other_machines`), so this
# note never promises a list that does not follow.
OTHER_INSTALL_NOTE_WITH_LIST = "run it on that computer, or choose a package for it below"
NOT_UNIQUE_INSTALL = "has no command: its pull name is not unique"


def _entry_name(entry: RankedEntry, block: MachineRanking, packages: Sequence[Package]) -> str | None:
    """`entry`'s pull name, dispatched through the first runtime this machine has that accepts its
    package where `packages` names it (`runtimes.accepting_on_machine`); `entry_pull_name`
    unchanged otherwise -- the two agree for Ollama always (`runtime.command` **is**
    `pull_choice.package_pull_name` there)."""
    if packages:
        by_identity = {package_identity_key(package): package for package in packages}
        package = by_identity.get(tuple(entry.package_identity))
        if package is not None:
            accepting = runtimes.accepting_on_machine(package, block.profile)
            if accepting:
                return accepting[0].command(package)
    return entry_pull_name(entry)


def install_target(
    block: MachineRanking, install_for: str | None, disk: Disk | None = None, packages: Sequence[Package] = ()
) -> tuple[RankedEntry, str] | None:
    """The first row of the machine this run is on that also fits its disk, and its local Ollama name.

    Only for that machine: the local Ollama name of a package is about the machine that would
    install it. A package with no such name has none. Without a free disk that was read the first
    row is the one, as before AP9-RM U11. The pull of step 5 lets the user choose another row (U6).
    """
    if install_for is None or block.machine != install_for or not block.ranked:
        return None
    entry, _skipped = best_on_disk(block.ranked, disk)
    if entry is None:
        return None
    name = _entry_name(entry, block, packages)
    return None if name is None else (entry, name)


def install_lines(
    block: MachineRanking,
    install_for: str | None,
    ambiguous: frozenset[str],
    disk: Disk | None,
    dash: str,
    pull_other_machines: frozenset[str] = frozenset(),
    packages: Sequence[Package] = (),
) -> list[Line]:
    """The install line of a block, behind a blank line; nothing where a row has no local name.

    Nothing is called and nothing is looked up for it (CONTRACTS.md, "Guided mode", "The screen").
    It stands behind a blank line and the command itself is drawn as a command: in running text
    under the table it went under (test round, 2026-09-25). A pull name more than one package
    answers to has a note instead (U4); another machine of a guided run gets its command with the
    machine named, and the reminder that this run pulls only here (U3). The disk of this machine
    moves the line to the best row that fits it, or turns it into the red `no package fits` (U11).
    """
    own = install_for is not None and block.machine == install_for
    read = known(disk) if own else None
    entry, skipped = best_on_disk(block.ranked, read)
    if own and block.ranked and entry is None:
        return [[], label_line(INSTALL_LABEL, [("class:warn", none_fits_words(read))])]
    target = install_target(block, block.machine if install_for is not None else None, read, packages)
    if target is None:
        return []
    entry, name = target
    if name in ambiguous:
        return [[], label_line(INSTALL_LABEL, [("class:note", f"#{entry.rank} {NOT_UNIQUE_INSTALL}")])]
    if own:
        # The one line of this view that may pass 100 characters: it is a command to copy, and a
        # command that was cut is worse than a line that wraps (second-model round, 2026-09-24).
        lines = [[], label_line(INSTALL_LABEL, install_pull_line(entry.rank, name))]
        return lines + ([label_line("", [("class:note", skipped_words(skipped, dash))])] if skipped else [])
    other = [("class:note", f"#{entry.rank} on {block.label}"), ("", "  "), ("class:command", f"ollama pull {name}")]
    note = OTHER_INSTALL_NOTE_WITH_LIST if block.machine in pull_other_machines else OTHER_INSTALL_NOTE
    return [[], label_line(INSTALL_LABEL, other), label_line("", [("class:note", note)])]


__all__ = [
    "INSTALL_LABEL",
    "NOT_UNIQUE_INSTALL",
    "OTHER_INSTALL_NOTE",
    "OTHER_INSTALL_NOTE_WITH_LIST",
    "install_lines",
    "install_target",
]
