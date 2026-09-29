"""What a result says about the rows it shows: one row per model, and the measured rows appended.

The rows themselves come from `ranking.shown_rows`; which case a machine's block is, and which of its
rows were appended for their measurement, is read back from the facts their notes name
(`ranking.BEST_FACT`, `ranking.SHOWN_FACT`), so the words need nothing but the block. The terminal
view and the Markdown file both take their words from here (ADR 0009), so `views.py` only calls.
"""

from __future__ import annotations

from .ranking import BEST_FACT, SHOWN_FACT, TOP_LIMIT
from .screen import FIT_HINT, ranks_text

# The two pieces of the key where the result shows one row per model -- one wording for the guided
# mode and `modelroom render` alike (Review r1-N5). One model shows its first ten, not all of them
# (Codex F5): "all its packages" promised rows the table does not have.
PER_MODEL_PIECES = (
    "one row per model, its least compression that fits",
    f"a run with one model shows its first {TOP_LIMIT} packages, then the measured ones",
)


# Each table ranks for its own machine, so two machines show different rows (GF question 5 (a),
# AP9-RM): said once per block in the terminal, once under the first table in the Markdown file,
# and only where a result ranks two machines or more.
EACH_MACHINE_PIECE = "ranked for this machine; another machine may show other rows"
EACH_MACHINE_SENTENCE = "Each table is ranked for its own machine; another machine may show other rows."


def several_ranked(document) -> bool:
    """Whether the result ranks two machines or more."""
    return sum(block.status == "ranked" for block in document.machines) >= 2


def per_model(block) -> bool:
    """Whether this block shows one row per model: its rule rows name `ranking.best_per_model`."""
    return any(BEST_FACT in entry.note.facts for entry in block.ranked)


def appended_ranks(block) -> list[int]:
    """The row numbers of the measured rows appended behind the rule's own."""
    return [entry.rank for entry in block.ranked if SHOWN_FACT in entry.note.facts]


def _rows(count: int) -> str:
    return "1 row" if count == 1 else f"{count} rows"


def shown_pieces(block, dash: str, several: bool = False) -> list[str]:
    """The `shown` line's own pieces: how many rows of how many packages, and the appended rows;
    with `several` ranked machines, that the table is this machine's own."""
    shown = len(block.ranked)
    if per_model(block):
        pieces = [f"{_rows(shown)} of {block.ranked_total} packages, one row per model"]
    else:
        pieces = [f"{shown} of {block.ranked_total} packages"]
    ranks = appended_ranks(block)
    if len(ranks) == 1:
        pieces.append(f"{ranks_text(ranks, dash)} is shown for its measurement, not by the rule")
    elif ranks:
        pieces.append(f"{ranks_text(ranks, dash)} are shown for their measurements, not by the rule")
    if several and block.ranked:
        pieces.append(EACH_MACHINE_PIECE)
    return pieces


def key_pieces(block) -> list[str]:
    """The key under a table: the fit words and what `best` means, and one row per model where so."""
    return [*FIT_HINT, *(PER_MODEL_PIECES if per_model(block) else ())]


def showing_line(block) -> str:
    """The Markdown sentence above a machine's table; its count includes the appended rows."""
    shown, measured = len(block.ranked), len(appended_ranks(block))
    if per_model(block):
        tail = f", then {measured} measured" if measured else ""
        return f"Showing {_rows(shown)} of {block.ranked_total} ranked packages: one row per model{tail}."
    if measured:
        return (
            f"Showing {shown} of {block.ranked_total} ranked packages: the first {shown - measured} by the rule, "
            f"then {measured} measured."
        )
    return f"Showing {shown} of {block.ranked_total} ranked packages."


__all__ = [
    "EACH_MACHINE_PIECE",
    "EACH_MACHINE_SENTENCE",
    "PER_MODEL_PIECES",
    "appended_ranks",
    "key_pieces",
    "per_model",
    "several_ranked",
    "showing_line",
    "shown_pieces",
]
