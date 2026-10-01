"""The words a table with vLLM rows carries beyond an Ollama-only one (AP9-SV2-F).

`views.py` reads the ranked document the way it always did and adds nothing of its own: this
module is the one place that turns `Fit.runtime` and a machine's profile into the extra words a
row, a legend or a machine's head carries once vLLM is in the table -- the package cell's prefix
(F4), the legend's extra line, the `runtime` row under a machine's head, and the `memory` line's
own vLLM piece. An Ollama-only table calls none of these to anything but `""`/`None`/`[]`, so its
output stays byte-identical to before AP9-SV2 (CONTRACTS.md, "Fit for vLLM").
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import runtimes
from .document import MachineRanking, RankedEntry, SetAsideEntry
from .fit_vllm import VLLM_GPU_MEMORY_UTILIZATION, VLLM_RUNTIME_OVERHEAD_GIB, equal_size_group, largest_pool
from .profile import HardwareProfile
from .ranking_words import PER_MODEL_PIECES, key_pieces, per_model
from .screen import ranks_text
from .vllm_server import ServedModel
from .vllm_words import FIT_HINT as VLLM_FIT_HINT, is_vllm

RUNTIME_LABEL = "runtime"
_PROFILE_ONLY = "(from its profile)"
_INSTALLED_NOT_SERVING = "installed, not serving"
_ANOTHER_MODEL = "another model"


def package_prefix(fit_runtime: str) -> str:
    """`words.package_prefix` of the runtime a row's `Fit.runtime` names (F4): `""` for Ollama,
    `"vLLM · "` for vLLM -- read off the one table, never a format check of its own."""
    return runtimes.runtime_of(fit_runtime).words.package_prefix


def packager_cell(entry: RankedEntry | SetAsideEntry) -> str:
    """`words.package_prefix + packager` (F4): the text every packager cell of a row carries,
    Markdown and the terminal view alike."""
    return f"{package_prefix(entry.fit.runtime)}{entry.packager}"


def shown_entries(block: MachineRanking) -> list[RankedEntry | SetAsideEntry]:
    """Every row the table of `block` shows: ranked, `Not covered` and `No fit` alike (Codex r2-4).
    A word the key owes a row is owed wherever that row stands, not only among the ranked ones."""
    return [*block.ranked, *block.not_covered, *block.no_fit]


def block_runtimes(block: MachineRanking) -> tuple[str, ...]:
    """`runtimes.runtimes_among` over every row `block` shows (Codex r6-1): two or more is a block
    drawn in sections (`views_sections`) whose rule names that (`render.build_render_document`)."""
    return runtimes.runtimes_among(entry.fit for entry in shown_entries(block))


def vllm_shown(block: MachineRanking) -> bool:
    """Whether any row the table of `block` shows is vLLM's, ranked or set aside (Codex r3-1): such
    a block owes its key even where nothing is ranked. An Ollama-only block answers `False`, so one
    with nothing ranked keeps drawing no key at all, as before AP9-SV2."""
    return any(is_vllm(entry.fit) for entry in shown_entries(block))


def machine_runtimes(block: MachineRanking, live_vllm: Mapping[str, object] | None = None) -> frozenset[str]:
    """The runtimes of `block`'s machine in this view: `runtimes.present_runtimes` of its profile and
    this run's own live vLLM reading for it (`live_vllm`, by machine; none in `modelroom render`) --
    the very set its packages were chosen by (`render.build_render_document`, Codex r7)."""
    return runtimes.present_runtimes(block.profile, (live_vllm or {}).get(block.machine))


def vllm_table(block: MachineRanking, live_vllm: Mapping[str, object] | None = None) -> bool:
    """Whether `block` is drawn as vLLM's (F5, Codex r5-2): a vLLM row is shown, or the machine has
    no Ollama at all (`machine_runtimes`) -- so an empty block of a vLLM-only machine carries no
    Ollama hint either. Never true for a machine without the `runtimes` field."""
    return vllm_shown(block) or runtimes.OLLAMA.key not in machine_runtimes(block, live_vllm)


def key_text(block: MachineRanking) -> list[str]:
    """The whole key under a table, for the Markdown file and the terminal alike: one runtime's
    only, never two side by side (F5, the views split a block of two runtimes into sections first,
    `views_sections.runtime_sections`). Ollama's own (`ranking_words.key_pieces`) for an Ollama
    table -- byte-identical to before AP9-SV2; vLLM's own (`vllm_words.FIT_HINT`) for a table of
    vLLM rows, wherever they stand (`shown_entries`, Codex r2-4/r3-1)."""
    if not vllm_shown(block):
        return key_pieces(block)
    return [*VLLM_FIT_HINT, *(PER_MODEL_PIECES if per_model(block) else ())]


def runtime_names(profile: HardwareProfile | None, live: object = None) -> str | None:
    """`Ollama · vLLM` or `vLLM`: the runtimes of a machine in table order, where vLLM is among
    them (`runtimes.present_runtimes`, the one source); `None` for a machine without vLLM, which
    says nothing new (F5: step 1 and the `runtime` row alike)."""
    present = runtimes.present_runtimes(profile, live)
    if runtimes.VLLM.key not in present:
        return None
    return " · ".join(runtime.name for runtime in runtimes.RUNTIMES if runtime.key in present)


def runtime_set_line(profile: HardwareProfile | None, live: object = None) -> str | None:
    """The `runtime` row as `views` draws it today (Codex r8): the runtimes of the machine
    (`runtime_names`, profile plus live detection -- the set alone) and the profile's own words.
    The live states (`runtime_line`, `_vllm_state`) wait for AP9-SV2-I/R and are not called."""
    names = runtime_names(profile, live)
    if names is None:
        return None
    # `(from its profile)` only where the profile alone names every runtime of the set: one known
    # only from this run's live reading drops it -- the set stands alone, no state word (Fix r8).
    from_profile = runtimes.present_runtimes(profile, live) == runtimes.present_runtimes(profile, None)
    return f"{names} {_PROFILE_ONLY}" if from_profile else names


def runtime_line(profile: HardwareProfile | None, live: object = None, served_matches_run: bool = False) -> str | None:
    """The `runtime` row under a machine's head (F4, F5): `None` where the machine has no vLLM at
    all, else its runtimes (`runtime_names`) and vLLM's state.

    `live` is this run's own step-1 reading for this machine (`runtimes.present_runtimes`'s own
    parameter) -- a `ServedModel` says `serving <name>` (`served_matches_run` decides whether that
    name is one of this run's own models); anything else truthy (a server that answered but not
    with a served model: a key prompt, an error) says `installed, not serving`; `None` (`modelroom
    render`, or a machine this run never talked to) falls back to the profile alone, which is the
    only source `modelroom render` ever has.
    """
    names = runtime_names(profile, live)
    return None if names is None else f"{names} {_vllm_state(live, served_matches_run)}"


def _vllm_state(live: object, served_matches_run: bool = False) -> str:
    if isinstance(live, ServedModel):
        model = _short_name(live.root) if served_matches_run else _ANOTHER_MODEL
        return f"serving {model}"
    if live is not None:
        return _INSTALLED_NOT_SERVING
    return _PROFILE_ONLY


def _short_name(repo: str) -> str:
    return repo.partition("/")[2] or repo


def unequal_cards_note(profile: HardwareProfile | None) -> str | None:
    """The `runtime` row's own continuation line (F1, F4): named only where a card smaller than the
    equal-size group was left out of it -- `None` for a machine of one size, or none at all.

    Says what the machine *offers* (the equal-size group `equal_size_group` finds), never what one
    package's row actually used: `t` is chosen per package (F1, a head count that does not divide
    the group size stops short of it), so a row's own card count is `Fit.cards`, read from the
    `memory` line's own vLLM piece (`vllm_pool_pieces`), not asserted here (Codex r1-4).
    """
    if profile is None:
        return None
    group = equal_size_group(profile)
    pool = largest_pool(profile)
    if pool is None or group.left_out == 0:
        return None
    size = group.cards[0].vram_gib
    cards_word = "card" if len(group.cards) == 1 else "cards"
    left_word = "card is" if group.left_out == 1 else "cards are"
    # Codex r6-2: tensor parallel takes 1, 2, 4 or 8 cards (`fit_vllm.largest_pool`, F1's own choice).
    usable = f"the {pool.group}" if pool.cards == pool.group else f"at most {pool.cards} of the {pool.group}"
    return f"vLLM can split over {usable} {cards_word} of {size:.2f} GB; {group.left_out} smaller {left_word} left out"


def vllm_pool_pieces(entries: Sequence[RankedEntry], dash: str) -> list[str]:
    """The `memory` line's own vLLM piece (F4), one per distinct card count among vLLM rows: the
    ranks, how many cards, and the runtime overhead -- vLLM's rows never join an Ollama `gpu`/
    `gpu_split` piece, even though they share `Fit.mode`'s own words, because the two accounting
    schemes (a reserved share of each card against Ollama's single VRAM reserve) are not the same
    number (`views._pool_pieces`)."""
    by_cards: dict[int, list[int]] = {}
    for entry in entries:
        by_cards.setdefault(entry.fit.cards, []).append(entry.rank)
    pieces = []
    for cards in sorted(by_cards):
        plural = "s" if cards != 1 else ""
        pieces.append(
            f"{ranks_text(by_cards[cards], dash)} vLLM on {cards} card{plural}: {VLLM_GPU_MEMORY_UTILIZATION} of "
            f"each, runtime {VLLM_RUNTIME_OVERHEAD_GIB:.2f} GB per card (one recording)"
        )
    return pieces


__all__ = [
    "RUNTIME_LABEL",
    "block_runtimes",
    "key_text",
    "machine_runtimes",
    "package_prefix",
    "packager_cell",
    "runtime_line",
    "runtime_names",
    "runtime_set_line",
    "shown_entries",
    "unequal_cards_note",
    "vllm_shown",
    "vllm_table",
    "vllm_pool_pieces",
]
