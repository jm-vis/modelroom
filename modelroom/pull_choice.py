"""Which package step 5 pulls: the list of the rows shown, and the pull names no row may claim.

AP9-UX2 U4 and U6 (GF F4, F7, ADR 0010). The ranking rule stays as ADR 0009 has it; this module
only reads what the render wrote.

- `pull_collisions` finds the pull names (`hf.co/<repo>:<quant>`, or an Ollama name) that more than
  one package answers to -- over every active, complete package a runtime of `runtimes.RUNTIMES`
  accepts, **before** the eligibility filter and whatever the provenance: an
  approved row can have a hidden counterpart of another file form with the same name, and the pull
  name does not tell file forms apart. Such a name belongs to no row: every row with it is locked,
  has no install line and no disk check.
- `pull_rows` is one row per shown table row of this machine, in table order, each with its pull
  name (`runtime.install_name`, read off the table) or the reason it cannot be pulled -- where the
  caller passes the run's `packages`, one row per runtime the machine has (`HardwareProfile.runtimes`;
  Ollama where it names none) that also accepts the row's package: two runtimes accepting the same
  GGUF package on the same machine give it two rows, each with a list value no answer file can
  confuse with the other's (`<runtime.key> <name>`, unprefixed for Ollama, so a machine with Ollama
  alone keeps exactly its values), and a package no runtime of the machine accepts gets no row.
  Without `packages` (`pull_other.py`), or for a row whose package is not among them, each shown
  row stays the one Ollama row it always was.
- `repo_lines` names the repository of every shown row whose `packager · quant` another shown row
  of the same block has too (F7), with the file form where one repository has both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from . import runtimes
from .contracts import Package
from .disk_fit import entry_over_disk, too_big_words
from .document import MachineRanking, RankedEntry, RenderDocument
from .profile import HardwareProfile
from .quantization import package_identity_key
from .room import Disk
from .screen import install_name

NONE_VALUE = "none"
# A2 (AP9-FR, Codex r1-5): "no safe pull name", not "no pull name" -- the one lock text a row
# without a name gets, whether it genuinely has none or `install_name` rejected the one it built.
NO_PULL_NAME = "no safe pull name"
NOT_UNIQUE = "pull name not unique"
LOCAL_ALREADY = "local already"
# Ollama's own question, read off the table (`runtimes.OLLAMA.words.question`, unchanged).
QUESTION = runtimes.OLLAMA.words.question
KEY = "pull_package"

# A2 (AP9-FR, Codex r1-1): the two shapes `screen.install_name` may build, and no other -- checked
# **before** any of the two is ever printed, put in an answer, or sent as `ollama pull`'s argument.
# Each part starts with a letter or digit, then any run of letters, digits, `_`/`-` and single dots
# (`\.(?!\.)`: a dot never right before another one, so `..` -- a path segment, `hf.co/../x` -- never
# passes); nothing outside `[A-Za-z0-9._-]` at all, so a semicolon, a `$(...)`, a backtick, a space,
# a newline, a leading `-` (an option to whatever reads the pasted line) and every Unicode
# look-alike character (the parts are ASCII only) fail on the first character that is not one of
# these. Compiled without `^`/`$` and matched with `fullmatch` on purpose: `$` alone would still let
# a name with a trailing `\n` through (Python's `$` matches before a final newline too).
_PART = r"[A-Za-z0-9](?:[A-Za-z0-9_-]|\.(?!\.))*"
PULL_NAME_RE = re.compile(rf"(?:hf\.co/{_PART}/{_PART}:{_PART}|(?:{_PART}/)?{_PART}:{_PART})")


def package_pull_name(package: Package) -> str | None:
    """The name Ollama would pull this package by, as the install line spells it (`screen.install_name`)."""
    identity = package_identity_key(package)
    return install_name(identity, package.quantization or "unknown", package.format)


def weights_bytes_of(package: Package) -> float:
    """The bytes a pull of `package`'s weight files needs (U8): the sum `--weights-bytes` (A3, A4)
    and the disk check before a pull both start from."""
    return float(sum(file.size_bytes for file in package.files if file.role in ("weights", "weights_shard")))


def runtime_value(name: str | None, runtime: str) -> str | None:
    """The one value a pull-list row or a collision key takes for the install `name` of the runtime
    with key `runtime`: unchanged for Ollama, `<runtime> <name>` for any other -- the runtime's own
    stable key, never a display word two runtimes could happen to share. Shared by
    `pull_collisions`, `PullRow.value` and the install line, so they never disagree."""
    if name is None:
        return None
    return name if runtime == runtimes.OLLAMA.key else f"{runtime} {name}"


def pull_collisions(packages: Sequence[Package]) -> dict[str, list[tuple[str, str, str]]]:
    """Every pull name more than one package identity answers to, with those identities.

    Over every package that is `active` and `complete` and that a runtime of `runtimes.RUNTIMES`
    accepts -- independent of its provenance, so an `unresolved` counterpart the render hides
    counts too, and independent of which machine will pull it (a collision is a property of the
    name, not of one machine's runtimes). A name two different runtimes would give the very same
    package never collides with itself (`runtime_value`'s own key tells them apart); today only
    Ollama ever accepts a package, so this is `package.format == "gguf" or package.source ==
    "ollama"`, unchanged, and `runtime.install_name` is `package_pull_name`, unchanged.
    """
    by_name: dict[str, set[tuple[str, str, str]]] = {}
    for package in packages:
        if not (package.active and package.complete):
            continue
        for runtime in runtimes.accepting_runtimes(package):
            name = runtime_value(runtime.install_name(package), runtime.key)
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


def has_named_row(block: MachineRanking) -> bool:
    """Whether any shown row of `block` has a pull name (A3): the condition `pull_other_step` gates
    on, and the one `views_install.install_lines` reads to know its list follows below a table."""
    return any(entry_pull_name(entry) is not None for entry in block.ranked)


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
    """One row of the pull list: the table row it stands for, its name, and why it is locked.

    `name` is the install name of the runtime the row belongs to (`Runtime.install_name`), what
    its command is built from; `runtime` is that runtime's key -- `"ollama"` for every row without
    `packages` -- the runtime a row is chosen with (`value`), marked "local already" by (its own
    `on_disk`) and shown the command of (`Runtime.command`) is read from here, not guessed again.
    """

    rank: int
    cell: str
    weights_gib: float
    name: str | None
    locked: str | None
    identity: tuple[str, str, str]
    # Locked because its download is larger than the free disk (AP9-RM U11): `locked` says how much.
    over_disk: bool = False
    runtime: str = "ollama"

    @property
    def value(self) -> str:
        """What an answer file names: `runtime_value` of the name; a locked row a value no answer can take."""
        value = runtime_value(self.name, self.runtime)
        return value if self.locked is None and value is not None else f"#{self.rank} locked"


def row_names(entry: RankedEntry, package: Package | None, profile: HardwareProfile | None) -> list[tuple[str, str | None]]:
    """`(runtime key, install name)` for every runtime of this machine that accepts `entry`'s
    package, in table order; none where the package is known and no runtime of the machine accepts
    it; Ollama's name of the row itself where the package is not known (`package` `None`)."""
    if package is None:
        return [(runtimes.OLLAMA.key, entry_pull_name(entry))]
    return [(runtime.key, runtime.install_name(package)) for runtime in runtimes.accepting_on_machine(package, profile)]


def pull_rows(
    document: RenderDocument,
    machine: str,
    ambiguous: frozenset[str] = frozenset(),
    dot: str = "·",
    disk: Disk | None = None,
    packages: Sequence[Package] = (),
    held: Mapping[str, frozenset] | None = None,
) -> list[PullRow]:
    """One row per shown table row of `machine`, in table order (regular and appended measured rows),
    and, where `packages` names its package, one per runtime of this machine that accepts it
    (`row_names`) -- a known package no runtime of the machine accepts gets no row. Without
    `packages` (`pull_other.py`) each row stays the one Ollama row it always was.

    With the free disk of this machine a row whose download does not fit it is locked with how much
    it needs (AP9-RM U11); a lock of its name comes first. A package the row's runtime holds needs
    no disk and is never locked for it (U11c): `held` names, by runtime key, what each runtime
    holds, and replaces `disk.held` for that runtime's rows; without it `disk.held` counts for all.
    """
    block = next((block for block in document.machines if block.machine == machine), None)
    if block is None:
        return []
    by_identity = {package_identity_key(package): package for package in packages}
    rows = []
    for entry in block.ranked:
        package = by_identity.get(tuple(entry.package_identity))
        cell = f"{entry.packager} {dot} {entry.quantization}"
        for key, name in row_names(entry, package, block.profile):
            own = disk if held is None or disk is None else replace(disk, held=held.get(key, frozenset()))
            locked = NO_PULL_NAME if name is None else NOT_UNIQUE if runtime_value(name, key) in ambiguous else None
            over = locked is None and entry_over_disk(entry, own)  # a package held already is never over (U11c)
            locked = too_big_words(entry.weights_gib, own) if over else locked
            rows.append(PullRow(entry.rank, cell, entry.weights_gib, name, locked, tuple(entry.package_identity), over, key))
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
    "PULL_NAME_RE",
    "QUESTION",
    "PullRow",
    "ambiguous_names",
    "collision_pieces",
    "default_row",
    "entry_pull_name",
    "has_named_row",
    "package_of",
    "package_pull_name",
    "pull_collisions",
    "pull_rows",
    "repo_lines",
    "row_names",
    "run_packages",
    "runtime_value",
    "unique_pull_name",
    "weights_bytes_of",
]
