"""The words a table with vLLM rows carries beyond an Ollama-only one (AP9-SV2-F).

`views.py` reads the ranked document the way it always did and adds nothing of its own: this
module is the one place that turns `Fit.runtime` and a machine's profile into the extra words a
row, a legend or a machine's head carries once vLLM is in the table -- the package cell's prefix
(F4), the legend's extra line, the `runtime` row under a machine's head, and the `memory` line's
own vLLM piece. An Ollama-only table calls none of these to anything but `""`/`None`/`[]`, so its
output stays byte-identical to before AP9-SV2 (CONTRACTS.md, "Fit for vLLM").
"""

from __future__ import annotations

from typing import Sequence

from . import runtimes
from .document import RankedEntry, SetAsideEntry
from .fit_vllm import VLLM_GPU_MEMORY_UTILIZATION, VLLM_RUNTIME_OVERHEAD_GIB, equal_size_group
from .profile import HardwareProfile
from .screen import ranks_text
from .vllm_server import ServedModel

RUNTIME_LABEL = "runtime"
_PROFILE_ONLY = "(from its profile)"
_INSTALLED_NOT_SERVING = "installed, not serving"
_ANOTHER_MODEL = "another model"
# F4 (Codex r1-5): `screen.FIT_SCALE_HINT`'s percentage rule ("ample: up to 60 % of the pool", ...)
# is Ollama's own -- `fit_vllm.vllm_fit_class` never reads a percentage of a pool at all, only how
# many requests fit at once. Named only once a vLLM row is shown, so an Ollama-only table's legend
# stays byte-identical to before AP9-SV2 (`legend_lines` below).
_OLLAMA_PERCENT_SCOPE = "The percentage rule above is Ollama's own."
_VLLM_CLASS_RULE = (
    "vLLM's classes count requests at once instead: no fit below one, tight from one below the "
    "requests this ranking assumes, fits from that many below twice them, ample at twice them or "
    "more (architecture basis only)."
)


def package_prefix(fit_runtime: str) -> str:
    """`words.package_prefix` of the runtime a row's `Fit.runtime` names (F4): `""` for Ollama,
    `"vLLM · "` for vLLM -- read off the one table, never a format check of its own."""
    return runtimes.runtime_of(fit_runtime).words.package_prefix


def packager_cell(entry: RankedEntry | SetAsideEntry) -> str:
    """`words.package_prefix + packager` (F4): the text every packager cell of a row carries,
    Markdown and the terminal view alike."""
    return f"{package_prefix(entry.fit.runtime)}{entry.packager}"


def legend_lines(entries: Sequence[RankedEntry]) -> list[str]:
    """Everything the legend carries beyond Ollama's own `screen.FIT_HINT`, once a vLLM row is
    shown: first that the percentage rule above is Ollama's and what vLLM's classes count instead
    (`_OLLAMA_PERCENT_SCOPE`, `_VLLM_CLASS_RULE`, Codex r1-5), then `words.key_text` of every
    runtime a shown row's `Fit.runtime` names, each at most once, in table order. Empty where no
    row names a runtime that carries one (every Ollama-only table) -- `key_pieces(block)` alone is
    then exactly `screen.FIT_HINT` plus `PER_MODEL_PIECES`, byte-identical to before AP9-SV2."""
    seen: list[str] = []
    key_texts: list[str] = []
    has_vllm = False
    for entry in entries:
        runtime = runtimes.runtime_of(entry.fit.runtime)
        has_vllm = has_vllm or runtime.key == runtimes.VLLM.key
        text = runtime.words.key_text
        if text is not None and runtime.key not in seen:
            seen.append(runtime.key)
            key_texts.append(text)
    if not has_vllm:
        return key_texts
    return [_OLLAMA_PERCENT_SCOPE, _VLLM_CLASS_RULE, *key_texts]


def runtime_line(profile: HardwareProfile | None, live: object = None, served_matches_run: bool = False) -> str | None:
    """The `runtime` row under a machine's head (F4): `None` where the machine has no vLLM at all
    (`runtimes.present_runtimes`), else `Ollama · vLLM <state>`.

    `live` is this run's own step-1 reading for this machine (`runtimes.present_runtimes`'s own
    parameter) -- a `ServedModel` says `serving <name>` (`served_matches_run` decides whether that
    name is one of this run's own models); anything else truthy (a server that answered but not
    with a served model: a key prompt, an error) says `installed, not serving`; `None` (`modelroom
    render`, or a machine this run never talked to) falls back to the profile alone, which is the
    only source `modelroom render` ever has.
    """
    if runtimes.VLLM.key not in runtimes.present_runtimes(profile, live):
        return None
    return f"{runtimes.OLLAMA.name} · {runtimes.VLLM.name} {_vllm_state(live, served_matches_run)}"


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
    if not group.cards or group.left_out == 0:
        return None
    size = group.cards[0].vram_gib
    cards_word = "card" if len(group.cards) == 1 else "cards"
    left_word = "card is" if group.left_out == 1 else "cards are"
    return (
        f"vLLM can split over the {len(group.cards)} {cards_word} of {size:.2f} GB; "
        f"{group.left_out} smaller {left_word} left out"
    )


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
    "legend_lines",
    "package_prefix",
    "packager_cell",
    "runtime_line",
    "unequal_cards_note",
    "vllm_pool_pieces",
]
