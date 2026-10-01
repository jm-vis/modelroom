"""One section per runtime, where a machine shows rows of two (AP9-SV2-F, F5).

Every row computes as if its runtime had the cards alone, so the rows of Ollama and of vLLM are
alternatives on the same cards, never one list. The document keeps one block per machine with
continuous numbers (`render._rankings_by_runtime`, Ollama's rows first); `runtime_sections` splits
it for the two views, which then draw each part exactly as they draw a machine of one runtime --
its heading, its table, its `memory` line and its key. A block of one runtime is one part, the
block itself, so every Ollama-only output stays byte-identical.

Also the `shown` line's pieces for the rows set aside (`set_aside_pieces`), moved out of `views.py`
(size guard) and read by it unchanged.
"""

from __future__ import annotations

from .document import MachineRanking, SetAsideEntry
from .intro import LABEL_COLUMN
from .runtimes import RUNTIMES, runtime_of
from .screen import JOIN_WIDTH
from .views_runtime import block_runtimes, packager_cell

ALTERNATIVES = "One section per runtime: alternatives on the same cards, not both at once."
# What is left of the 100 characters for a labeled line's text (`views._NOTE_WIDTH`): the label's
# own column and the one space every line of the terminal view carries.
NOTE_WIDTH = JOIN_WIDTH - LABEL_COLUMN - 1
# How many of the packages behind one reason are named before the rest is a number.
_NAMED_PER_REASON = 3


def section_title(runtime_key: str) -> str:
    """`with Ollama`, `with vLLM`: the heading of one runtime's section."""
    return f"with {runtime_of(runtime_key).name}"


def runtime_sections(block: MachineRanking) -> list[tuple[str, MachineRanking]]:
    """`[(runtime key, part)]`: one part per runtime whose rows the block shows, in table order --
    or the block alone, where it shows rows of one runtime or none (no heading then, F5).

    Each part is the block with that runtime's rows only (ranked, `Not covered`, `No fit`), their
    numbers kept. Its `ranked_total` is exact: every part after the first lists all of its ranked
    packages (`render._ranked_entries`, `whole`), so the first part's total is the rest.
    """
    keys = block_runtimes(block)  # the one predicate (Codex r6-1), as the ranking and the rule read it
    if len(keys) < 2:
        return [(keys[0] if keys else RUNTIMES[0].key, block)]
    parts = []
    for key in keys:
        ranked = [entry for entry in block.ranked if entry.fit.runtime == key]
        update = {
            "ranked": ranked,
            "ranked_total": len(ranked),
            "not_covered": [entry for entry in block.not_covered if entry.fit.runtime == key],
            "no_fit": [entry for entry in block.no_fit if entry.fit.runtime == key],
        }
        parts.append((key, block.model_copy(update=update)))
    later = sum(part.ranked_total for _key, part in parts[1:])
    first_key, first = parts[0]
    parts[0] = (first_key, first.model_copy(update={"ranked_total": block.ranked_total - later}))
    return parts


def set_aside_pieces(word: str, entries: list[SetAsideEntry]) -> list[str]:
    """One piece per reason: how many packages it covers, and up to three of their names.

    41 lines of `not covered`, one per package, said the same thing 41 times and pushed the
    ranking itself off the screen (hand test, 2026-09-24); since 2026-09-24 they stand behind the
    `shown` count as pieces of one line. The reason is named only where the list has more than
    one -- with a single reason the word (`no fit`, `not covered`) already says it, and the
    reason itself is in the Markdown file, which has room for it. The word `package` is the
    `shown` piece's own (`10 of 23 packages`), so a piece behind it is the count and the reason.
    """
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        grouped.setdefault(entry.reason, []).append(f"{packager_cell(entry)} {entry.quantization}")
    pieces = []
    for reason, names in sorted(grouped.items()):
        count = str(len(names))
        label = f"{word}, {reason}" if len(grouped) > 1 else word
        pieces.append(f"{count} {label}{_named(names, NOTE_WIDTH - len(count) - len(label) - 1)}")
    return pieces


def _named(names: list[str], room: int) -> str:
    """Up to three of `names` in brackets, and fewer where the room is not there.

    The names are the evidence behind a count, so they are what gives way: three packages of an
    account with a long name made a piece of 137 characters, and `joined` leaves an over-long piece
    whole (measured in the second-model round of 2026-09-24). The count and the reason never give
    way, and nothing is cut mid-name -- a name ends where the next one would have begun.

    `room` is measured against everything this adds: the space in front, both brackets -- three
    characters, not two. With two the line came to 101 (second-model round, 2026-09-25).
    """
    for shown in range(min(_NAMED_PER_REASON, len(names)), 0, -1):
        rest = len(names) - shown
        listed = ", ".join(names[:shown]) + (f" and {rest} more" if rest > 0 else "")
        if len(listed) + 3 <= room:
            return f" ({listed})"
    return ""


__all__ = ["ALTERNATIVES", "NOTE_WIDTH", "runtime_sections", "section_title", "set_aside_pieces"]
