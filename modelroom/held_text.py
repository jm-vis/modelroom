"""The `local` line under a table: which shown rows a computer's Ollama already holds (AP9-FR-N2, a).

The rows come from the one set the disk rule reads, `Disk.held` -- the identities the daemon provably holds by
the load test's digest rule (U11c): for this computer step 5 read it from its own daemon, for the active
computer over SSH from one `candidates` call. The line says the ranks and the place, never a name of the
computer's own (`CONTRACTS.md`, "The screen", `local`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .runtimes import OLLAMA
from .screen import ranks_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import MachineRanking
    from .room import Disk

LOCAL_LABEL = "local"


def local_pieces(block: "MachineRanking", disk: "Disk | None", where: str, dash: str = "–") -> list[str]:
    """`['#2–5 already in Ollama on this computer']` for the shown Ollama rows `disk` holds, else `[]`.

    `where` is `this computer` or the alias of the active computer. A row of another runtime is never
    called held here: `Disk.held` is what Ollama holds (`pull_choice.runtime_disk`)."""
    if disk is None or not disk.held:
        return []
    ranks = [
        entry.rank for entry in block.ranked if entry.fit.runtime == OLLAMA.key and tuple(entry.package_identity) in disk.held
    ]
    return [f"{ranks_text(ranks, dash)} already in Ollama on {where}"] if ranks else []


__all__ = ["LOCAL_LABEL", "local_pieces"]
