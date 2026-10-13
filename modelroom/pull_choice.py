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
  Without `packages`, or for a row whose package is not among them, each shown Ollama row stays
  the one Ollama row it always was. Since AP9-SV2-I a vLLM row is a row of the list as well
  (`Fit.runtime` says it is one): named by its repo (`vllm <repo>`), judged against vLLM's own
  disk place (`disk_fit.place`), locked `no safe command` or `computed only: ...` where
  `install_vllm.vllm_install` has no line for it. Since AP9-LS-E a Hugging Face GGUF package of a machine
  that names LM Studio has a second entry, `lmstudio <owner>/<repo>@<quant>`, of the same rank: judged against
  LM Studio's own place (`Disk.lmstudio`), locked `no safe command` or `computed only: lms get for ...`
  where `lmstudio_install.lms_reason` says so. Each entry reads the disk of its own runtime
  (`disk_fit.entry_over_disk(..., runtime=key)`); the table row stays one.
- `repo_lines` names the repository of every shown row whose `packager · quant` another shown row
  of the same block has too (F7), with the file form where one repository has both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from . import runtimes
from .catalog_findings import NO_RECOMMENDATION, community_conversion_text, load_blocked_text
from .contracts import Package
from .disk_fit import entry_download_gib, entry_over_disk, entry_place, too_big_words
from .document import MachineRanking, RankedEntry, RenderDocument
from .install_vllm import NO_SAFE_COMMAND, safe_repo, vllm_install
from .lmstudio_install import lms_reason
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
    """The pull name of a shown row, or `None` where it has none: for a vLLM row `vllm <repo>`
    (`runtime_value`, the one value form a collision key shares with the list), for every other
    row Ollama's name, unchanged."""
    if entry.fit.runtime == runtimes.VLLM.key:
        repo = entry.package_identity[1]
        return runtime_value(repo, runtimes.VLLM.key) if safe_repo(repo) else None
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
    # Whether its download needs more than the free disk as its runtime sees it, whatever lock its
    # name has: the disk decides before the name which row is offered (`offered`).
    too_big: bool = False
    # AP9-KO. `blocked`: the red line of an `absent` check (`load_blocked_text`); the row is locked with it, so
    # not even an explicit choice takes it. `community`: the packager of a GGUF package of an account that is
    # neither the publisher of the base model nor a listed packager -- the row stays a choice by name but is
    # never the automatic one (V4). Both leave a row out of the recommendation (`recommendable`).
    blocked: str | None = None
    community: str | None = None

    @property
    def recommendable(self) -> bool:
        """Whether this row may be the automatic recommendation: not blocked and not a community conversion."""
        return self.blocked is None and self.community is None

    @property
    def value(self) -> str:
        """What an answer file names: `runtime_value` of the name; a locked row a value no answer can take."""
        value = runtime_value(self.name, self.runtime)
        return value if self.locked is None and value is not None else f"#{self.rank} locked"


def row_names(entry: RankedEntry, package: Package | None, profile: HardwareProfile | None) -> list[tuple[str, str | None]]:
    """`(runtime key, install name)` for every runtime of this machine that accepts `entry`'s
    package, in table order; none where the package is known and no runtime of the machine accepts
    it; Ollama's name of the row itself where the package is not known (`package` `None`)."""
    if entry.fit.runtime == runtimes.VLLM.key:
        # The render showed this row because the machine's runtime set holds vLLM (the profile, or a
        # server found live on an old one): that decides it, not `profile.runtimes` alone.
        if package is None or not runtimes.VLLM.accepts(package):
            return [(runtimes.VLLM.key, None)]
        return [(runtimes.VLLM.key, runtimes.VLLM.install_name(package))]
    if package is None:
        return [(runtimes.OLLAMA.key, entry_pull_name(entry))]
    return [(runtime.key, runtime.install_name(package)) for runtime in runtimes.accepting_on_machine(package, profile)]


def runtime_disk(runtime: str, disk: Disk | None, held: Mapping[str, frozenset] | None = None) -> Disk | None:
    """`disk` as the rows of the runtime with key `runtime` see it: `disk.held` is what the Ollama
    daemon holds (the guided run reads it there), so it counts for Ollama's rows alone. vLLM's rows
    read the disk nested in it (`Disk.vllm`, the Hugging Face cache; `None` where it was not read),
    holding what it found itself and what its own `on_disk` found (`held`, by runtime key); LM Studio's read
    the disk nested in it too (`Disk.lmstudio`, AP9-LS-E); any other runtime sees the same free disk holding
    what its own `on_disk` found."""
    if disk is None or runtime == runtimes.OLLAMA.key:
        return disk
    own = frozenset((held or {}).get(runtime, frozenset()))
    if runtime == runtimes.VLLM.key:
        return None if disk.vllm is None else replace(disk, vllm=replace(disk.vllm, held=disk.vllm.held | own))
    if runtime == runtimes.LMSTUDIO.key:
        return None if disk.lmstudio is None else replace(disk, lmstudio=replace(disk.lmstudio, held=disk.lmstudio.held | own))
    return replace(disk, held=own)


def pull_rows(
    document: RenderDocument,
    machine: str,
    ambiguous: frozenset[str] = frozenset(),
    dot: str = "·",
    disk: Disk | None = None,
    packages: Sequence[Package] = (),
    held: Mapping[str, frozenset] | None = None,
) -> list[PullRow]:
    """`block_rows` of `machine`'s block of `document`; none where it has no block."""
    block = next((block for block in document.machines if block.machine == machine), None)
    return [] if block is None else block_rows(block, ambiguous, dot, disk, packages, held)


def block_rows(
    block: MachineRanking,
    ambiguous: frozenset[str] = frozenset(),
    dot: str = "·",
    disk: Disk | None = None,
    packages: Sequence[Package] = (),
    held: Mapping[str, frozenset] | None = None,
) -> list[PullRow]:
    """One row per shown table row of the block, in table order (regular and appended measured rows),
    and, where `packages` names its package, one per runtime of this machine that accepts it
    (`row_names`) -- a known package no runtime of the machine accepts gets no row. Without
    `packages` (`pull_other.py`) each row stays the one Ollama row it always was.

    With the free disk of this machine a row whose download does not fit it is locked with how much
    it needs (AP9-RM U11); a lock of its name comes first. A package the row's runtime holds needs
    no disk and is never locked for it (U11c): `disk.held` is what the Ollama daemon holds and
    counts for Ollama's rows alone, `held` names by runtime key what each other runtime holds
    itself (`runtime_disk`).
    """
    by_identity = {package_identity_key(package): package for package in packages}
    rows = []
    for entry in block.ranked:
        package = by_identity.get(tuple(entry.package_identity))
        cell = f"{entry.packager} {dot} {entry.quantization}"
        for key, name in row_names(entry, package, block.profile):
            own = runtime_disk(key, disk, held)
            too_big = entry_over_disk(entry, own, runtime=key)  # a package held already is never over (U11c)
            blocked = _load_block(key, entry)
            locked = blocked or _name_lock(key, name, entry, package, block.profile, ambiguous)
            over = locked is None and too_big
            locked = too_big_words(entry_download_gib(entry, own, runtime=key), entry_place(entry, own, key)) if over else locked
            identity = tuple(entry.package_identity)
            shown = cell if key == runtimes.OLLAMA.key else f"{runtimes.runtime_of(key).name} {dot} {cell}"
            community = entry.packager if _is_community(key, entry) else None
            rows.append(PullRow(entry.rank, shown, entry.weights_gib, name, locked, identity, over, key, too_big, blocked, community))
    return rows


def _gguf_runtime(key: str) -> bool:
    """Whether the runtime with this key loads a GGUF file through llama.cpp: the two a check of the llama.cpp source is about."""
    return key in (runtimes.OLLAMA.key, runtimes.LMSTUDIO.key)


def _load_block(key: str, entry: RankedEntry) -> str | None:
    """The red line that locks a row for its runtime, or `None`: only a check that says `absent` blocks (V2) --
    the installation over Ollama and LM Studio, not a vLLM row (which loads a tensor package, never this file)."""
    check = entry.load_check
    if check is None or check.result != "absent" or not _gguf_runtime(key) or entry.fit.runtime == runtimes.VLLM.key:
        return None
    return load_blocked_text(runtimes.runtime_of(key).name, entry.base_model_hf_repo, check)


def _is_community(key: str, entry: RankedEntry) -> bool:
    """A GGUF row of an unlisted account (`origin` `other`, V4): selectable by name, never the automatic choice."""
    return entry.origin == "other" and _gguf_runtime(key)


def blocked_runtimes(profile: HardwareProfile | None) -> str:
    """The runtimes of a machine a blocked GGUF package would be installed over, as the red line names them:
    `Ollama`, or `Ollama and LM Studio` -- the table shows the package once, whatever it is installed with."""
    present = runtimes.machine_runtime_keys(profile)
    names = [runtime.name for runtime in runtimes.RUNTIMES if runtime.key in present and _gguf_runtime(runtime.key)]
    return " and ".join(names) or runtimes.OLLAMA.name


def blocked_lines_of(block: MachineRanking) -> list[tuple[str, str]]:
    """`(label, red line)` of every package of the block an `absent` check blocks -- what the table says about it;
    the same words the pull list locks a ranked row with (`PullRow.blocked`). A ranked row is named by its rank
    (`#2`); a package set aside (`not_covered`, `no_fit`) has no rank and is named by `packager · quantization`, so
    the block is said whatever reason the package was set aside for (the `shown` line leaves a lone reason out)."""
    names = blocked_runtimes(block.profile)
    blocked = [
        entry
        for entry in [*block.ranked, *block.not_covered, *block.no_fit]
        if entry.load_check is not None and entry.load_check.result == "absent" and entry.fit.runtime != runtimes.VLLM.key
    ]
    short = {id(e): f"#{e.rank}" if isinstance(e, RankedEntry) else f"{e.packager} · {e.quantization}" for e in blocked}
    lines = []
    for entry in blocked:
        label = short[id(entry)]
        if sum(1 for other in blocked if short[id(other)] == label) > 1:  # two alike: name the repository, then the file
            label = f"{label} {entry.package_identity[1]}"
            if sum(1 for other in blocked if other.package_identity[1] == entry.package_identity[1] and short[id(other)] == short[id(entry)]) > 1:
                label = f"{label} {entry.package_identity[2]}"
        lines.append((label, load_blocked_text(names, entry.base_model_hf_repo, entry.load_check)))
    return lines


def _name_lock(
    key: str, name: str | None, entry: RankedEntry, package: Package | None, profile: HardwareProfile | None,
    ambiguous: frozenset[str],
) -> str | None:
    """Why a row cannot be chosen for its name, or `None`. Ollama's: no safe name, then a name more
    than one package answers to. vLLM's: no safe command (name, revision, cards), then `computed
    only` where its cards are not `nvidia-smi`'s (`install_vllm.vllm_install`), then the same collision.
    LM Studio's: `no safe command`, or the `computed only: lms get for ...` of its package
    (`lmstudio_install.lms_reason`), then the same collision."""
    if key == runtimes.VLLM.key:
        reason = NO_SAFE_COMMAND if name is None else vllm_install(package, entry.fit, profile).reason
        if reason is not None:
            return reason
    elif key == runtimes.LMSTUDIO.key:
        reason = NO_SAFE_COMMAND if package is None else lms_reason(package)  # `computed only: ...` or `no safe command`
        if reason is not None:
            return reason
    elif name is None:
        return NO_PULL_NAME
    return NOT_UNIQUE if runtime_value(name, key) in ambiguous else None


@dataclass(frozen=True)
class Offered:
    """Which row of a pull list is offered (`offered`): `row`; else `stop`, the first row of the
    rank that ended the search whose name locks it (`NO_PULL_NAME`, `NOT_UNIQUE`); `skipped` the
    ranks passed over because no row of theirs fits the disk; `all_over` where that is every rank."""

    row: PullRow | None
    skipped: tuple[int, ...] = ()
    stop: PullRow | None = None
    all_over: bool = False
    # AP9-KO (V4): the first row of each rank passed over for the automatic recommendation -- blocked by an
    # `absent` check, or a community conversion. With no `row` and some `passed`, nothing is left to recommend.
    passed: tuple[PullRow, ...] = ()


def recommendation_lines(offer: Offered, nothing_left: bool | None = None) -> list[tuple[str, str]]:
    """`(style, text)` of what a skipped recommendation says, in the order of the ranks: `COMMUNITY_CONVERSION`
    for a community conversion (once per packager), then `NO_RECOMMENDATION` where no row is offered at all. A
    blocked row says why under its table (`blocked_lines_of`), so it adds no line of its own here. Nothing where no
    row was passed over. `nothing_left` overrides "no row is offered at all" for a search that looked at rank 1
    alone (an answer file without `pull_package`): the line is true only if no shown row at all is recommendable."""
    lines: list[tuple[str, str]] = []
    for row in offer.passed:
        if row.blocked is None:
            line = ("class:warn", community_conversion_text(str(row.community)))
            if line not in lines:
                lines.append(line)
    none_left = offer.row is None and offer.stop is None if nothing_left is None else nothing_left
    if offer.passed and none_left:
        lines.append(("class:note", NO_RECOMMENDATION))
    return lines


def offered(
    rows: Sequence[PullRow], ranks: Sequence[int] = (), over: Sequence[int] = (), *, past_disk: bool = True
) -> Offered:
    """The one rule both the install line under the table and the pull of step 5 offer a row by.

    Rank by rank (`ranks`, the shown rows in table order; the ranks of `rows` where not given): a
    rank none of whose rows fits the disk is passed over (`past_disk`, the install line) or ends
    the search (`#1` alone, an answer file without `pull_package`); otherwise the first of its rows
    that fits the disk and whose name is its own (`locked is None`) is offered -- every runtime of
    the machine, table order, each with its own holding (`block_rows`). Where every row of the rank
    that fits is locked by its name, nothing is offered and that row says why (`stop`). A rank with
    no row at all (a package no runtime of the machine accepts) ends the search with nothing,
    unless it is in `over` (its download is over the disk, judged as Ollama's: `rank_over`).

    AP9-KO (V4): only the shown rows are looked at, and a rank none of whose fitting rows is `recommendable`
    (blocked by an `absent` check, or a community conversion) is passed over for the recommendation, never
    pulled: with `past_disk` the search goes on to the next rank (`passed` keeps what was skipped), without it
    (`#1` alone, an answer file without `pull_package`) it ends -- it never moves to another rank.
    """
    ranks = list(ranks) or list(dict.fromkeys(row.rank for row in rows))
    skipped: list[int] = []
    passed: list[PullRow] = []
    for rank in ranks:
        group = [row for row in rows if row.rank == rank]
        fitting = [row for row in group if not row.too_big]
        if not fitting and (group or rank in over):
            if not past_disk:
                return Offered(None, tuple(skipped), passed=tuple(passed))
            skipped.append(rank)
            continue
        eligible = [row for row in fitting if row.recommendable]
        if fitting and not eligible:
            passed.append(fitting[0])
            if not past_disk:
                return Offered(None, tuple(skipped), passed=tuple(passed))
            continue
        free = next((row for row in eligible if row.locked is None), None)
        return Offered(free, tuple(skipped), None if free is not None or not eligible else eligible[0], passed=tuple(passed))
    return Offered(None, tuple(skipped), all_over=bool(ranks) and len(skipped) == len(ranks), passed=tuple(passed))


def rank_over(block: MachineRanking, rows: Sequence[PullRow], disk: Disk | None) -> list[int]:
    """The ranks of the shown rows whose download fits the disk for no runtime of the machine: every
    row of theirs is `too_big`, or -- a row with no runtime at all -- judged as Ollama's."""
    over = []
    for entry in block.ranked:
        group = [row for row in rows if row.rank == entry.rank]
        if all(row.too_big for row in group) if group else entry_over_disk(entry, runtime_disk(entry.fit.runtime, disk), runtime=entry.fit.runtime):
            over.append(entry.rank)
    return over


def default_row(rows: Sequence[PullRow]) -> PullRow | None:
    """The row an answer file without `pull_package` takes: `#1`, where it has a unique name and fits
    the disk -- `offered` over `#1` alone, so of its rows (one per runtime of the machine) the first
    the install line would offer too."""
    return offered(rows, [1], past_disk=False).row


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
    "Offered",
    "ambiguous_names",
    "block_rows",
    "blocked_lines_of",
    "blocked_runtimes",
    "collision_pieces",
    "default_row",
    "entry_pull_name",
    "has_named_row",
    "package_of",
    "offered",
    "package_pull_name",
    "pull_collisions",
    "pull_rows",
    "rank_over",
    "recommendation_lines",
    "repo_lines",
    "row_names",
    "runtime_disk",
    "run_packages",
    "runtime_value",
    "unique_pull_name",
    "weights_bytes_of",
]
