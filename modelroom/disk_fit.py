"""Whether a shown package fits the free disk of the machine it is shown for (AP9-RM U11, ADR 0012).

GF 2026-09-29, question U11 (a): the fit and the rank stay what they are -- memory only, ADR 0009 --
but a row whose download is larger than the free disk where Ollama keeps its models is marked: the
whole row on the house red, ` ×` behind its download (also without color), a line `disk` under
`memory`, an `install` line that takes the best row that also fits the disk, and a pull list whose
too large rows cannot be picked. Only where the free disk of that machine was read (`room.Disk`,
`known`); another computer, and a disk nobody could read, change nothing.

Pure: the rows come from the document, the disk from a `LiveRoom`. The rule is the one of the check
before a pull (`guided_install._disk_allows`), read from the same function, `room.pull_need_bytes`:
the download plus the margin of 1 GB against the free disk (decided 2026-09-29, AP9-RM U11b), so no
row is offered that the check then refuses. The check itself stays, as the second safety. A package
the daemon provably holds already (`Disk.held`) needs no disk and is never marked (U11c, Codex r3-1):
it stays a `local already` row of the pull list, as before U11.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .document import MachineRanking, RankedEntry
from .room import GIB, PULL_DISK_MARGIN_BYTES, Disk, gb, pull_need_bytes
from .screen import ranks_text

DISK_LABEL = "disk"
# What a row too large for the disk carries behind its download, and in front of the `disk` line;
# `glyphs().cross` gives an `x` where the console cannot encode it.
MARK = "×"
_ALL_OVER_TAIL = " -- free space first, or pick a smaller model"


def known(disk: Disk | None) -> Disk | None:
    """The disk where its free space was read, else `None` (nothing is marked then)."""
    return disk if disk is not None and disk.known else None


def place(runtime: str, disk: Disk | None) -> Disk | None:
    """The disk where `runtime` keeps its models, out of the one `Disk` a machine carries: Ollama's is
    that disk itself, vLLM's the one nested in it (`Disk.vllm`, AP9-SV2-I); another runtime has none."""
    if disk is None:
        return None
    return disk if runtime == "ollama" else disk.vllm if runtime == "vllm" else None


def vllm_only(disk: Disk | None) -> Disk | None:
    """`disk` as a computer over SSH lends it to the rows of the disk rule (AP9-SV2-R, R1-1): vLLM's own place
    (`Disk.vllm`, from `room.model_disk`) and nothing of Ollama's -- its free space is unknown here, so no
    Ollama row of that computer is marked, locked or passed over, exactly as before SV2-R (FR B5)."""
    if disk is None:
        return None
    return replace(disk, free_bytes=None, drive=None, held=frozenset(), needs={})


def over_disk(weights_gib: float, disk: Disk | None, identity: Sequence[str] | None = None) -> bool:
    """Whether a download of `weights_gib` and its margin need more than the free disk; never where it
    is unknown, and never for a package its runtime holds already (`identity` in `Disk.held`, U11c).
    `disk` is the place of the row's own runtime (`place`); where it names the bytes still to fetch of
    this package (`Disk.needs`, vLLM's missing start files) those count instead of `weights_gib`.
    The rule of the check before a pull (`room.pull_need_bytes`)."""
    read = known(disk)
    if read is None:
        return False
    held = identity is not None and tuple(identity) in read.held
    return pull_need_bytes(_download_bytes(weights_gib, read, identity), held) > read.free_bytes


def _download_bytes(weights_gib: float, disk: Disk, identity: Sequence[str] | None) -> float:
    still = disk.needs.get(tuple(identity)) if identity is not None else None
    return weights_gib * GIB if still is None else float(still)


def entry_download_gib(entry: RankedEntry, disk: Disk | None) -> float:
    """What the disk rule counts as the download of a shown row, in GiB: the bytes still to fetch
    where its runtime's place names them, else its weights."""
    read = known(place(entry.fit.runtime, disk))
    return entry.weights_gib if read is None else _download_bytes(entry.weights_gib, read, entry.package_identity) / GIB


def entry_over_disk(entry: RankedEntry, disk: Disk | None) -> bool:
    """`over_disk` of a shown row, by its download and its package: the one test of the table, the
    `disk` line, the install line and the pull list. The disk read is the one where the row's own
    runtime keeps its models (`place`): Ollama's for an Ollama row, vLLM's (the Hugging Face cache)
    for a vLLM row -- never the one for the other, and nothing marked where that one was not read."""
    return over_disk(entry.weights_gib, place(entry.fit.runtime, disk), entry.package_identity)


def block_place(block: MachineRanking, disk: Disk | None) -> Disk | None:
    """The disk the words about `block` name: vLLM's for a block of vLLM rows alone (a runtime section,
    or a machine with vLLM only), else Ollama's -- as before for every machine without vLLM."""
    runtimes = {entry.fit.runtime for entry in block.ranked}
    return place("vllm", disk) if runtimes == {"vllm"} else disk


def over_ranks(block: MachineRanking, disk: Disk | None) -> list[int]:
    """The ranks of the shown rows whose download does not fit the free disk."""
    return [entry.rank for entry in block.ranked if entry_over_disk(entry, disk)]


def disk_words(
    block: MachineRanking, disk: Disk | None, dash: str = "–", mark: str = MARK, ranks: Sequence[int] | None = None
) -> str | None:
    """`× #1–10 need more than the 26.4 GB free on C:` (the download and its margin), with what to do
    where no row fits; else `None`. One row `needs`. `ranks` are the rows that do not fit where the
    caller judged them per runtime (`views_install.over_ranks_on_machine`); else `over_ranks`."""
    ranks = over_ranks(block, disk) if ranks is None else list(ranks)
    if not ranks:
        return None
    read = known(block_place(block, disk))
    need = "needs" if len(ranks) == 1 else "need"
    text = f"{mark} {ranks_text(ranks, dash)} {need} more than the {gb(read.free_bytes)} free on {read.drive}"
    return text + (_ALL_OVER_TAIL if len(ranks) == len(block.ranked) else "")


def best_on_disk(entries: Sequence[RankedEntry], disk: Disk | None) -> tuple[RankedEntry | None, list[int]]:
    """The first row that also fits the disk, and the ranks skipped for it; `(None, all)` where none fits."""
    skipped = []
    for entry in entries:
        if not entry_over_disk(entry, disk):
            return entry, skipped
        skipped.append(entry.rank)
    return None, skipped


def skipped_words(skipped: list[int], dash: str = "–") -> str:
    """The gray line under an install line that is not `#1`: why the rows above it were passed over."""
    need = "needs" if len(skipped) == 1 else "need"
    return f"the best row that also fits the disk; {ranks_text(skipped, dash)} {need} more space"


def none_fits_words(disk: Disk) -> str:
    """The red install line where no shown row fits the disk."""
    return f"no package fits the {gb(disk.free_bytes)} free on {disk.drive}; free space first"


def too_big_words(weights_gib: float, disk: Disk) -> str:
    """Why a row of the pull list cannot be picked: `needs 94.1 GB plus 1 GB room, 26.4 GB free`."""
    return f"needs {weights_gib:.1f} GB plus {PULL_DISK_MARGIN_BYTES / GIB:g} GB room, {gb(disk.free_bytes)} free"


__all__ = [
    "DISK_LABEL",
    "MARK",
    "best_on_disk",
    "block_place",
    "disk_words",
    "entry_download_gib",
    "entry_over_disk",
    "known",
    "none_fits_words",
    "over_disk",
    "over_ranks",
    "place",
    "skipped_words",
    "too_big_words",
    "vllm_only",
]
