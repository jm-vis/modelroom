"""Which package step 5 pulls: the list of the rows shown, and the pull names no row may claim.

AP9-UX2 U4 and U6 (GF F4, F7, ADR 0010). The ranking rule stays as ADR 0009 has it; this module
only reads what the render wrote.

- `pull_collisions` finds the pull names (`hf.co/<repo>:<quant>`, or an Ollama name) that more than
  one package answers to -- over every active, complete GGUF or Ollama package of the run's base
  models, **before** the eligibility filter and whatever the provenance (Codex r2-3, r3-1): an
  approved row can have a hidden counterpart of another file form with the same name, and the pull
  name does not tell file forms apart. Such a name belongs to no row: every row with it is locked,
  has no install line and no disk check.
- `pull_rows` is one row per shown table row of this machine, in table order, each with its pull
  name or the reason it cannot be pulled.
- `repo_lines` names the repository of every shown row whose `packager · quant` another shown row
  of the same block has too (F7), with the file form where one repository has both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import Package
from .disk_fit import over_disk, too_big_words
from .document import MachineRanking, RankedEntry, RenderDocument
from .quantization import package_identity_key
from .room import Disk
from .screen import install_name

NONE_VALUE = "none"
NO_PULL_NAME = "no pull name"
NOT_UNIQUE = "pull name not unique"
LOCAL_ALREADY = "local already"
QUESTION = "Which package should be pulled into Ollama?"
KEY = "pull_package"


def package_pull_name(package: Package) -> str | None:
    """The name Ollama would pull this package by, as the install line spells it (`screen.install_name`)."""
    identity = package_identity_key(package)
    return install_name(identity, package.quantization or "unknown", package.format)


def pull_collisions(packages: Sequence[Package]) -> dict[str, list[tuple[str, str, str]]]:
    """Every pull name more than one package identity answers to, with those identities.

    Over every package that is `active` and `complete` and a GGUF build or an Ollama package --
    independent of its provenance, so an `unresolved` counterpart the render hides counts too.
    """
    by_name: dict[str, set[tuple[str, str, str]]] = {}
    for package in packages:
        if not (package.active and package.complete and (package.format == "gguf" or package.source == "ollama")):
            continue
        name = package_pull_name(package)
        if name is not None:
            by_name.setdefault(name, set()).add(package_identity_key(package))
    return {name: sorted(identities) for name, identities in by_name.items() if len(identities) > 1}


class AmbiguousNames(frozenset):
    """The pull names of `pull_collisions` as the set the views and the list take, with how many
    package identities answer to each (`counts`) -- the number the `shown` line says."""

    counts: dict[str, int]

    def __new__(cls, counts: dict[str, int] | None = None):
        made = super().__new__(cls, counts or {})
        made.counts = dict(counts or {})
        return made


def ambiguous_names(packages: Sequence[Package]) -> AmbiguousNames:
    """The pull names of `pull_collisions`, as the set the views and the list take."""
    return AmbiguousNames({name: len(identities) for name, identities in pull_collisions(packages).items()})


def run_packages(config, selection=None) -> list[Package]:
    """Every package of this folder's snapshot, narrowed to the run's base models; before any filter."""
    from .contracts import SchemaVersionError
    from .selection import narrow_snapshot
    from .state import UnreadableStateFileError, read_snapshot

    try:
        snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError):
        return []
    if snapshot is None:
        return []
    if selection is not None:
        snapshot = narrow_snapshot(snapshot, selection)
    return list(snapshot.packages)


def entry_pull_name(entry: RankedEntry) -> str | None:
    """The pull name of a shown row, or `None` where it has none."""
    return install_name(entry.package_identity, entry.quantization, entry.format)


def unique_pull_name(entry: RankedEntry, ambiguous: frozenset[str]) -> str | None:
    """A row's pull name where it is its own; `None` where it has none or shares it (U4)."""
    name = entry_pull_name(entry)
    return None if name is None or name in ambiguous else name


def collision_pieces(block: MachineRanking, ambiguous: frozenset[str]) -> list[str]:
    """One piece of the `shown` line per shown row whose pull name is not unique."""
    counts = getattr(ambiguous, "counts", {})
    pieces = []
    for entry in block.ranked:
        name = entry_pull_name(entry)
        if name is not None and name in ambiguous:
            files = counts.get(name, 2)
            pieces.append(f"#{entry.rank} has no unique pull name: {files} files answer to it")
    return pieces


@dataclass(frozen=True)
class PullRow:
    """One row of the pull list: the table row it stands for, its name, and why it is locked."""

    rank: int
    cell: str
    weights_gib: float
    name: str | None
    locked: str | None
    identity: tuple[str, str, str]
    # Locked because its download is larger than the free disk (AP9-RM U11): `locked` says how much.
    over_disk: bool = False

    @property
    def value(self) -> str:
        """What an answer file names: the pull name; a locked row a value no answer can take."""
        return self.name if self.locked is None and self.name is not None else f"#{self.rank} locked"


def pull_rows(
    document: RenderDocument, machine: str, ambiguous: frozenset[str] = frozenset(), dot: str = "·", disk: Disk | None = None
) -> list[PullRow]:
    """One row per shown table row of `machine`, in table order (regular and appended measured rows).

    With the free disk of this machine a row whose download does not fit it is locked with how much
    it needs (AP9-RM U11); a lock of its name comes first.
    """
    block = next((block for block in document.machines if block.machine == machine), None)
    if block is None:
        return []
    rows = []
    for entry in block.ranked:
        name = entry_pull_name(entry)
        locked = NO_PULL_NAME if name is None else NOT_UNIQUE if name in ambiguous else None
        over = locked is None and over_disk(entry.weights_gib, disk)
        locked = too_big_words(entry.weights_gib, disk) if over else locked
        cell = f"{entry.packager} {dot} {entry.quantization}"
        rows.append(PullRow(entry.rank, cell, entry.weights_gib, name, locked, tuple(entry.package_identity), over))
    return rows


def default_row(rows: Sequence[PullRow]) -> PullRow | None:
    """The row an answer file without `pull_package` takes: `#1`, where it has a unique name."""
    if rows and rows[0].rank == 1 and rows[0].locked is None:
        return rows[0]
    return None


def package_of(row: PullRow, packages: Sequence[Package]) -> Package | None:
    """The snapshot's package of a row, by the identity the render built the row with."""
    return next((package for package in packages if package_identity_key(package) == row.identity), None)


def repo_lines(block: MachineRanking) -> list[str]:
    """`#2  unsloth/Qwen3.5-9B-GGUF` for every shown row whose `packager · quant` is not its own (F7).

    Only the shown rows of the block count; where one repository has two such rows, the file form
    follows (`BF16/ (folder)`, `(single file)`) -- the same quantization as one file and in a folder.
    """
    groups: dict[tuple[str, str], list[RankedEntry]] = {}
    for entry in block.ranked:
        groups.setdefault((entry.packager, entry.quantization), []).append(entry)
    doubled = [entry for entries in groups.values() if len(entries) > 1 for entry in entries]
    lines = []
    for entry in sorted(doubled, key=lambda item: item.rank):
        source, repo, stem = entry.package_identity
        same_repo = [other for other in groups[(entry.packager, entry.quantization)] if other.package_identity[1] == repo]
        form = f"  {_file_form(stem)}" if source == "huggingface" and len(same_repo) > 1 else ""
        lines.append(f"#{entry.rank}  {repo}{form}")
    return lines


def _file_form(stem: str) -> str:
    """`BF16/ (folder)` for weights in a folder of the repository, `(single file)` otherwise."""
    folders = sorted({part.rsplit("/", 1)[0] for part in stem.split("|") if "/" in part})
    return f"{folders[0]}/ (folder)" if folders else "(single file)"


__all__ = [
    "AmbiguousNames",
    "KEY",
    "LOCAL_ALREADY",
    "NONE_VALUE",
    "NOT_UNIQUE",
    "NO_PULL_NAME",
    "QUESTION",
    "PullRow",
    "ambiguous_names",
    "collision_pieces",
    "default_row",
    "entry_pull_name",
    "package_of",
    "package_pull_name",
    "pull_collisions",
    "pull_rows",
    "repo_lines",
    "run_packages",
    "unique_pull_name",
]
