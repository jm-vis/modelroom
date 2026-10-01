"""The `install` line under a machine's table: the one line a reader copies to get a package.

Moved out of `views.py` (size guard, AP9-RM U11) and extended there: where the free disk of the
machine was read, the line takes the best row that also fits it (`best_row`) and says
under it which rows were passed over, and where no row fits it says so in red instead of offering a
command. `views.py` re-exports `INSTALL_LABEL` and `install_target`.

The line belongs to a runtime of the table (`install_choice`): where the run's `packages` name the
row's package, the first runtime of this machine that accepts it gives the name
(`Runtime.install_name`) and the command (`Runtime.command`), and a package no runtime of the
machine accepts gets no line at all; where its package is not known, the line is Ollama's, from the
row itself (`entry_pull_name`). For Ollama the command is `ollama pull <name>`, as before.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import runtimes
from .contracts import Package
from .disk_fit import entry_over_disk, known, none_fits_words, skipped_words
from .document import MachineRanking, RankedEntry
from .intro import label_line
from .pull_choice import row_names, runtime_disk, runtime_value
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


Held = Mapping[str, frozenset] | None


def fitting_pair(
    entry: RankedEntry, block: MachineRanking, disk: Disk | None, packages: Sequence[Package] = (), held: Held = None
) -> tuple[str, str | None] | None:
    """The first `(runtime key, install name)` of `entry` whose download fits the disk as that
    runtime sees it, or `None` where no pair fits. Every runtime of this machine that accepts the
    row's package is tried, in table order (`row_names`), each with its own holding
    (`pull_choice.runtime_disk`: what the Ollama daemon holds frees only an Ollama pair, what
    another runtime's own `on_disk` found -- `held`, by runtime key -- only that runtime's). A row
    whose package no runtime of the machine accepts, or is not known, is judged as Ollama's, as
    before -- so a machine with Ollama alone gets exactly `entry_over_disk`'s answer."""
    package = next((package for package in packages if package_identity_key(package) == tuple(entry.package_identity)), None)
    pairs = row_names(entry, package, block.profile) or [(runtimes.OLLAMA.key, None)]
    return next((pair for pair in pairs if not entry_over_disk(entry, runtime_disk(pair[0], disk, held))), None)


def over_ranks_on_machine(block: MachineRanking, disk: Disk | None, packages: Sequence[Package] = (), held: Held = None) -> list[int]:
    """The ranks of the shown rows no pair of which fits the disk (`fitting_pair`): the rows the
    table marks and the `disk` line names; `disk_fit.over_ranks` on a machine with Ollama alone."""
    return [entry.rank for entry in block.ranked if fitting_pair(entry, block, disk, packages, held) is None]


def best_row(
    block: MachineRanking, disk: Disk | None, packages: Sequence[Package] = (), held: Held = None
) -> tuple[RankedEntry | None, tuple[str, str | None] | None, list[int]]:
    """`disk_fit.best_on_disk` over the pairs of each row: the first shown row with a pair that fits
    the disk (`fitting_pair`), that pair, and the ranks skipped for it."""
    skipped = []
    for entry in block.ranked:
        pair = fitting_pair(entry, block, disk, packages, held)
        if pair is not None:
            return entry, pair, skipped
        skipped.append(entry.rank)
    return None, None, skipped


def install_choice(
    block: MachineRanking,
    install_for: str | None,
    disk: Disk | None = None,
    packages: Sequence[Package] = (),
    held: Mapping[str, frozenset] | None = None,
) -> tuple[RankedEntry, "runtimes.Runtime", str] | None:
    """The first row of the machine this run is on that also fits its disk, the runtime it is
    installed with and its install name there.

    Only for that machine: the install name of a package is about the machine that would install
    it. The runtime is the first of this machine, in table order, that accepts the row's package
    and whose own holding lets its download fit the disk (`best_row`); a package no runtime of the
    machine accepts, or one without a name, has none. Without a free disk that was read the first
    row is the one, as before AP9-RM U11. The pull of step 5 lets the user choose another row (U6).
    """
    if install_for is None or block.machine != install_for or not block.ranked:
        return None
    entry, pair, _skipped = best_row(block, disk, packages, held)
    if entry is None or pair is None or pair[1] is None:
        return None
    key, name = pair
    return entry, runtimes.runtime_of(key), name


def install_target(
    block: MachineRanking, install_for: str | None, disk: Disk | None = None, packages: Sequence[Package] = ()
) -> tuple[RankedEntry, str] | None:
    """`install_choice` without the runtime: the row and its install name."""
    chosen = install_choice(block, install_for, disk, packages)
    return None if chosen is None else (chosen[0], chosen[2])


def install_lines(
    block: MachineRanking,
    install_for: str | None,
    ambiguous: frozenset[str],
    disk: Disk | None,
    dash: str,
    pull_other_machines: frozenset[str] = frozenset(),
    packages: Sequence[Package] = (),
    held: Mapping[str, frozenset] | None = None,
) -> list[Line]:
    """The install line of a block, behind a blank line; nothing where a row has no local name.

    Nothing is called and nothing is looked up for it (CONTRACTS.md, "Guided mode", "The screen").
    It stands behind a blank line and the command itself is drawn as a command: in running text
    under the table it went under (test round, 2026-09-25). A pull name more than one package
    answers to has a note instead (U4); another machine of a guided run gets its command with the
    machine named, and the reminder that this run pulls only here (U3). The disk of this machine
    moves the line to the best row that fits it, or turns it into the red `no package fits` (U11);
    `held` is what each runtime but Ollama holds itself (`guided_install.other_runtimes_held`).
    """
    own = install_for is not None and block.machine == install_for
    read = known(disk) if own else None
    entry, _pair, skipped = best_row(block, read, packages, held)
    if own and block.ranked and entry is None:
        return [[], label_line(INSTALL_LABEL, [("class:warn", none_fits_words(read))])]
    target = install_choice(block, block.machine if install_for is not None else None, read, packages, held)
    if target is None:
        return []
    entry, runtime, name = target
    if runtime_value(name, runtime.key) in ambiguous:
        return [[], label_line(INSTALL_LABEL, [("class:note", f"#{entry.rank} {NOT_UNIQUE_INSTALL}")])]
    if own:
        # The one line of this view that may pass 100 characters: it is a command to copy, and a
        # command that was cut is worse than a line that wraps (second-model round, 2026-09-24).
        lines = [[], label_line(INSTALL_LABEL, install_pull_line(entry.rank, name, command=runtime.command(name)))]
        return lines + ([label_line("", [("class:note", skipped_words(skipped, dash))])] if skipped else [])
    other = [("class:note", f"#{entry.rank} on {block.label}"), ("", "  "), ("class:command", runtime.command(name))]
    note = OTHER_INSTALL_NOTE_WITH_LIST if block.machine in pull_other_machines else OTHER_INSTALL_NOTE
    return [[], label_line(INSTALL_LABEL, other), label_line("", [("class:note", note)])]


__all__ = [
    "INSTALL_LABEL",
    "NOT_UNIQUE_INSTALL",
    "OTHER_INSTALL_NOTE",
    "OTHER_INSTALL_NOTE_WITH_LIST",
    "best_row",
    "fitting_pair",
    "over_ranks_on_machine",
    "install_choice",
    "install_lines",
    "install_target",
]
