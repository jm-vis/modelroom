"""The `install` line under a machine's table: the one line a reader copies to get a package.

Moved out of `views.py` (size guard, AP9-RM U11) and extended there: where the free disk of the
machine was read, the line takes the best row that also fits it and says under it which rows were
passed over, and where no row fits it says so in red instead of offering a command. `views.py`
re-exports `INSTALL_LABEL` and `install_target`.

The line offers the very row the pull of step 5 would: it is built from the rows of the pull list
(`pull_choice.block_rows`, one per runtime of the machine that accepts a row's package, each with
its own holding and its own name) by the one rule both read (`pull_choice.offered`). Its command is
that row's runtime's (`Runtime.command`); where no row is offered, the lock of the row that ended
the search says why (`has no command`, `no package fits`). For Ollama the command is
`ollama pull <name>`, as before.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import runtimes
from .contracts import Package
from .disk_fit import block_place, known, none_fits_words, skipped_words
from .document import MachineRanking, RankedEntry
from .install_vllm import COMPUTED_ONLY, NO_SAFE_COMMAND, install_block_lines, vllm_install
from .intro import label_line
from .pull_choice import NOT_UNIQUE, Offered, PullRow, block_rows, offered, package_of, rank_over
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


def install_offer(
    block: MachineRanking,
    disk: Disk | None = None,
    packages: Sequence[Package] = (),
    held: Held = None,
    ambiguous: frozenset[str] = frozenset(),
) -> Offered:
    """`pull_choice.offered` over this block's pull rows, every rank, past the rows that do not fit
    the disk: the row the install line offers, or why there is none."""
    rows = block_rows(block, ambiguous, disk=disk, packages=packages, held=held)
    return offered(rows, [entry.rank for entry in block.ranked], rank_over(block, rows, disk))


def over_ranks_on_machine(block: MachineRanking, disk: Disk | None, packages: Sequence[Package] = (), held: Held = None) -> list[int]:
    """The ranks of the shown rows whose download fits the disk for no runtime of the machine, each
    with its own holding (`pull_choice.rank_over`): the rows the table marks and the `disk` line
    names; `disk_fit.over_ranks` on a machine with Ollama alone."""
    return rank_over(block, block_rows(block, disk=disk, packages=packages, held=held), disk)


def install_choice(
    block: MachineRanking,
    install_for: str | None,
    disk: Disk | None = None,
    packages: Sequence[Package] = (),
    held: Held = None,
) -> tuple[RankedEntry, "runtimes.Runtime", str] | None:
    """The row of the machine this run is on that the install line offers (`install_offer`, no
    name counted as shared), its runtime and its install name; `None` where none is offered.

    Only for that machine: the install name of a package is about the machine that would install
    it. Without a free disk that was read the first row is the one, as before AP9-RM U11. The pull
    of step 5 lets the user choose another row (U6).
    """
    if install_for is None or block.machine != install_for or not block.ranked:
        return None
    row = install_offer(block, disk, packages, held).row
    if row is None or row.name is None:
        return None
    entry = next(entry for entry in block.ranked if entry.rank == row.rank)
    return entry, runtimes.runtime_of(row.runtime), row.name


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
    held: Held = None,
    live_vllm: object = None,
) -> list[Line]:
    """The install line of a block, behind a blank line; nothing where a row has no local name.

    Nothing is called and nothing is looked up for it (CONTRACTS.md, "Guided mode", "The screen").
    It stands behind a blank line and the command itself is drawn as a command: in running text
    under the table it went under (test round, 2026-09-25). A pull name more than one package
    answers to has a note instead (U4); another machine of a guided run gets its command with the
    machine named, and the reminder that this run pulls only here (U3). The disk of this machine
    moves the line to the best row that fits it, or turns it into the red `no package fits` (U11);
    `held` is what each runtime but Ollama holds itself (`guided_install.other_runtimes_held`).

    A vLLM row (AP9-SV2-I) is two commands and what to do with them (`install_vllm.install_block_lines`),
    nothing to pull; `live_vllm` is what this machine's local server states it serves, so another
    model there adds the stop line.
    """
    if install_for is None or not block.ranked:
        return []
    own = block.machine == install_for
    read = known(block_place(block, disk)) if own else None
    offer = install_offer(block, disk if own else None, packages, held, ambiguous)
    if own and offer.all_over:
        return [[], label_line(INSTALL_LABEL, [("class:warn", none_fits_words(read))])]
    if offer.row is None:
        stop = offer.stop
        note = f"#{stop.rank} {NOT_UNIQUE_INSTALL}" if stop is not None and stop.locked == NOT_UNIQUE else None
        note = f"#{stop.rank} {COMPUTED_ONLY}" if stop is not None and stop.locked == COMPUTED_ONLY else note
        note = f"#{stop.rank} {NO_SAFE_COMMAND}" if stop is not None and stop.locked == NO_SAFE_COMMAND else note
        return [] if note is None else [[], label_line(INSTALL_LABEL, [("class:note", note)])]
    row, runtime = offer.row, runtimes.runtime_of(offer.row.runtime)
    if runtime.key == runtimes.VLLM.key:
        return _vllm_lines(block, row, own, disk, packages, held, live_vllm, offer.skipped, dash)
    command = runtime.command(str(row.name))
    if own:
        # The one line of this view that may pass 100 characters: it is a command to copy, and a
        # command that was cut is worse than a line that wraps (second-model round, 2026-09-24).
        lines = [[], label_line(INSTALL_LABEL, install_pull_line(row.rank, str(row.name), command=command))]
        skipped = list(offer.skipped)
        return lines + ([label_line("", [("class:note", skipped_words(skipped, dash))])] if skipped else [])
    other = [("class:note", f"#{row.rank} on {block.label}"), ("", "  "), ("class:command", command)]
    note = OTHER_INSTALL_NOTE_WITH_LIST if block.machine in pull_other_machines else OTHER_INSTALL_NOTE
    return [[], label_line(INSTALL_LABEL, other), label_line("", [("class:note", note)])]


def _vllm_lines(
    block: MachineRanking, row: PullRow, own: bool, disk: Disk | None, packages: Sequence[Package], held: Held,
    live_vllm: object, skipped: Sequence[int], dash: str,
) -> list[Line]:
    """The install block of a vLLM row under a table: this machine's with its notes (on disk already,
    another model served), another machine's with the machine named and no promise about this run."""
    entry = next(entry for entry in block.ranked if entry.rank == row.rank)
    vllm = disk.vllm if own and disk is not None else None
    relative = vllm is not None and vllm.relative
    install = vllm_install(package_of(row, packages), entry.fit, block.profile, relative_cache=relative)
    if not own:
        return [[], *install_block_lines(INSTALL_LABEL, row.rank, install, repo=row.name, where=block.label)]
    there = row.identity in (held or {}).get(runtimes.VLLM.key, frozenset()) or (vllm is not None and row.identity in vllm.held)
    lines = install_block_lines(INSTALL_LABEL, row.rank, install, repo=row.name, served=live_vllm, on_disk=there)
    if skipped:
        lines.append(label_line("", [("class:note", skipped_words(list(skipped), dash))]))
    return [[], *lines]


__all__ = [
    "INSTALL_LABEL",
    "NOT_UNIQUE_INSTALL",
    "OTHER_INSTALL_NOTE",
    "OTHER_INSTALL_NOTE_WITH_LIST",
    "install_choice",
    "install_lines",
    "install_offer",
    "install_target",
    "over_ranks_on_machine",
]
