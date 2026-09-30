"""The head under a machine's table: the budget the fit computes with, and what was read live.

AP9-RM U10 (GF 2026-09-29, ADR 0012). Until then the line under a machine's head was `free` and
said the free graphics memory of the moment -- a number the fit never used. Now it is `room`, and
its pieces say the **budget**: the graphics memory of the profile minus its reserve, the system
memory minus its reserve, each with the size it came from; then the free disk where Ollama keeps
its models, the one live value a pull is checked against. What the daemon holds right now stands
on a line of its own, `loaded`, and only when it holds something: a hint, never a budget.

Pure: the budgets come from the profile and the reserves of the `MachineRanking` (never from
`LiveRoom`), the live values from a `LiveRoom` read elsewhere. Live values belong to the machine
they were read on, whichever it is; `where` only chooses the words for a machine nobody read
(`this`, `other`, or `render`, which reads no machine live).
"""

from __future__ import annotations

from typing import Literal

from .document import MachineRanking
from .intro import LABEL_COLUMN
from .reserve import old_default_hint
from .room import UNCLEAR_REASON, LiveRoom, LoadedModel, disk_words, gb, graphics_words
from .screen import JOIN_WIDTH

Where = Literal["this", "other", "render", "ssh", "ssh_inactive"]

ROOM_LABEL = "room"
LOADED_LABEL = "loaded"
# What is left of the 100 characters for one piece: the label's column and the one space in front.
NOTE_WIDTH = JOIN_WIDTH - LABEL_COLUMN - 1
DISK_UNKNOWN = "disk unknown"
ANOTHER_COMPUTER_DISK = "disk unknown on another computer"
RENDER_DISK = "disk unknown (modelroom render reads no machine live)"
# The graphics states whose cards the fit counts with a memory of their own (`fit._OWN_VRAM_STATES`).
_OWN_CARD_STATES = ("measured", "entered", "multi_gpu")


def head_pieces(block: MachineRanking, live: LiveRoom | None, where: Where) -> list[str]:
    """The pieces of the `room` line of one machine: budgets from the profile, then the disk."""
    profile = block.profile
    if profile is None:
        pieces = ["graphics unknown", "memory unknown"]
    elif profile.gpu_state == "unified_memory":
        pieces = [_shared_piece(block)]
    else:
        pieces = [_graphics_piece(block), _memory_piece(block)]
    return [*pieces, _disk_piece(live, where, block.machine)]


def loaded_pieces(live: LiveRoom | None) -> list[str]:
    """The pieces of the `loaded` line: each model the daemon holds, then the free graphics memory
    of now; nothing where the daemon holds nothing or was not read (G2, W4)."""
    if live is None or not live.loaded:
        return []
    pieces = [_loaded_piece(model) for model in live.loaded]
    if live.graphics_free_mib:
        pieces.append(graphics_words(list(live.graphics_free_mib)))
    return pieces


def reserve_hint_rows(block: MachineRanking) -> list[str]:
    """The hint for a machine that still carries the fixed reserve of 0.1.0, as rows that fit the
    width (GF question 3 (a)); nothing for every other machine."""
    profile = block.profile
    if profile is None:
        return []
    hint = old_default_hint(
        block.reserve_ram_gib, block.reserve_ram_origin, profile.ram_physical_gib, profile.gpu_state == "unified_memory"
    )
    if hint is None:
        return []
    head, _sep, tail = hint.partition("; ")
    return [hint] if len(hint) <= NOTE_WIDTH else [f"{head};", tail]


def _budget(value_gib: float) -> str:
    """A budget as the head says it; one of 0 or less is said in words, never below zero (RM-2)."""
    return f"{value_gib:.1f} GB for models" if value_gib > 0 else "nothing left for models"


def _graphics_piece(block: MachineRanking) -> str:
    """`graphics 11.0 GB for models (12.0 GB card minus reserve)`, the cards counted or listed."""
    profile = block.profile
    if profile.gpu_state == "none" or (profile.gpu_state in _OWN_CARD_STATES and not profile.gpus):
        return "graphics none counted"
    if profile.gpu_state not in _OWN_CARD_STATES:
        return "graphics unknown"  # a state without a number: never `none`
    cards = [gpu.vram_gib for gpu in profile.gpus]
    budget = _budget(sum(card - block.reserve_vram_gib for card in cards))
    sizes = [f"{card:.1f}" for card in cards]
    if len(cards) == 1:
        named = f"{sizes[0]} GB card"
    elif len(set(sizes)) == 1:
        named = f"{len(cards)} x {sizes[0]} GB cards"
    else:
        named = " + ".join(sizes) + " GB cards"
    piece = f"graphics {budget} ({named} minus reserve)"
    if len(piece) > NOTE_WIDTH:
        # One piece is never broken inside (`joined`), so a list too long for the line is a count.
        piece = f"graphics {budget} ({len(cards)} cards, {sum(cards):.1f} GB in all minus reserve)"
    return piece


def _memory_piece(block: MachineRanking) -> str:
    """`memory 111.7 GB for models (127.7 GB minus reserve)`, or `memory unknown` (G1)."""
    ram = block.profile.ram_physical_gib
    if ram is None:
        return "memory unknown"
    return f"memory {_budget(ram - block.reserve_ram_gib)} ({ram:.1f} GB minus reserve)"


def _shared_piece(block: MachineRanking) -> str:
    """Unified memory is one piece: `shared memory 95.0 GB for models (128.0 GB minus both reserves)`."""
    ram = block.profile.ram_physical_gib
    if ram is None:
        return "shared memory unknown"
    budget = _budget(ram - block.reserve_ram_gib - block.reserve_vram_gib)
    return f"shared memory {budget} ({ram:.1f} GB minus both reserves)"


def _disk_piece(live: LiveRoom | None, where: Where, machine: str = "") -> str:
    """The free disk read for this machine, whichever it is; else the words of `where`. A stored
    SSH machine that is not the active one this run (B4, Codex r1 finding 10) is never read live
    at all -- its own line names it, not "another computer"."""
    disk = live.disk if live is not None else None
    if where == "ssh":
        return _ssh_disk_piece(disk)
    if where == "ssh_inactive":
        return f"disk unknown ({machine} not connected in this run)"
    if disk is not None and disk.known:
        return disk_words(disk)
    if where == "this":
        return disk_words(disk) if disk is not None else DISK_UNKNOWN
    return ANOTHER_COMPUTER_DISK if where == "other" else RENDER_DISK


def _ssh_disk_piece(disk) -> str:
    """The active SSH machine's own words (FR-B1, B5), named by its alias, never a path: known,
    `unclear` (two candidate places), or the connection itself failed (`disk.reason` then already
    names it, `"<alias> not reached"`, set by `guided_remote.read_room`)."""
    if disk is not None and disk.known:
        return f"disk {gb(disk.free_bytes)} free on {disk.drive} (where Ollama keeps its models)"
    if disk is not None and disk.reason == UNCLEAR_REASON:
        return f"disk unknown on {disk.drive} (two places Ollama may use)"
    if disk is not None and disk.reason:
        return f"disk unknown ({disk.reason})"
    return f"disk unknown on {disk.drive}" if disk is not None and disk.drive else DISK_UNKNOWN


def _loaded_piece(model: LoadedModel) -> str:
    """`Ollama holds qwen3.5:9b (7.4 GB, 6.9 GB of it on the card)`, the name cut with `…` to fit."""
    sizes = f" ({gb(model.size_bytes)}, {gb(model.vram_bytes)} of it on the card)"
    room = NOTE_WIDTH - len("Ollama holds ") - len(sizes)
    name = model.name if len(model.name) <= room else model.name[: room - 1] + "…"
    return f"Ollama holds {name}{sizes}"


__all__ = [
    "ANOTHER_COMPUTER_DISK",
    "LOADED_LABEL",
    "NOTE_WIDTH",
    "RENDER_DISK",
    "ROOM_LABEL",
    "Where",
    "head_pieces",
    "loaded_pieces",
    "reserve_hint_rows",
]
