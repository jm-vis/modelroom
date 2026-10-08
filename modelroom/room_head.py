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
from .fit_vllm import NO_CARD_REASON, SHARED_MEMORY_REASON, VLLM_GPU_MEMORY_UTILIZATION, largest_pool
from .hf_cache import lmstudio_disk_words, shared_disk_words, vllm_disk_words
from .intro import LABEL_COLUMN
from .reserve import old_default_hint
from .room import UNCLEAR_REASON, Disk, LiveRoom, LoadedModel, disk_words, gb, graphics_words
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
# A machine whose profile names no runtime reads as Ollama's alone (`runtimes.machine_runtime_keys`).
_OLLAMA_ONLY = frozenset({"ollama"})


def head_pieces(
    block: MachineRanking, live: LiveRoom | None, where: Where, runtime_keys: frozenset[str] = _OLLAMA_ONLY
) -> list[str]:
    """The pieces of the `room` line of one machine: budgets from the profile, then the disk.

    `runtime_keys` are the machine's runtimes (`runtimes.machine_runtime_keys`, F5, Codex r5-2):
    Ollama's budgets and its disk only where Ollama is one of them -- named `for Ollama` beside vLLM,
    `for models` alone, word for word as before -- and vLLM's pool where vLLM is. vLLM has no
    system-memory fallback, so it adds no memory piece; its disk is `_disk_pieces`'s. LM Studio runs the
    same llama.cpp under the same fit, so its machine has the same budgets (AP9-LS-E): beside vLLM they are
    named for the runtimes of the GGUF rows, `for LM Studio` or `for Ollama and LM Studio`.
    """
    profile = block.profile
    ollama, vllm = "ollama" in runtime_keys, "vllm" in runtime_keys and profile is not None
    lmstudio = "lmstudio" in runtime_keys
    whom = _gguf_names(ollama, lmstudio) if vllm else "models"
    if profile is None:
        pieces = ["graphics unknown", "memory unknown"]
    elif profile.gpu_state == "unified_memory":
        pieces = [_shared_piece(block, whom)]
    else:
        pieces = [_graphics_piece(block, whom), _memory_piece(block, whom)]
    pieces = (pieces if ollama or lmstudio else []) + ([_vllm_piece(block)] if vllm else [])
    if lmstudio:
        return [*pieces, *_disk_pieces_with_lmstudio(live, where, block.machine, ollama, vllm)]
    return [*pieces, *_disk_pieces(live, where, block.machine, ollama, vllm)]


def _gguf_names(ollama: bool, lmstudio: bool) -> str:
    """`Ollama`, `LM Studio` or `Ollama and LM Studio`: whom the budgets of the GGUF rows are for beside vLLM."""
    return "Ollama and LM Studio" if ollama and lmstudio else "LM Studio" if lmstudio else "Ollama"


def _disk_pieces_with_lmstudio(live: LiveRoom | None, where: Where, machine: str, ollama: bool, vllm: bool) -> list[str]:
    """The disk piece(s) of a machine that names LM Studio (AP9-LS-E, S3): one place per runtime that has one
    read (`Disk.lmstudio`, `Disk.vllm`; this machine only), in the order Ollama, LM Studio, vLLM, the places on
    one drive together in one piece named for all of them, `disk 41.2 GB free on C: (Ollama and LM Studio
    models)`. Ollama's place keeps its own words where it is alone on its drive (`_disk_piece`, also when it
    was not read). A place that was not read says nothing beside Ollama's; a machine with LM Studio and no
    Ollama whose place was not read says that the disk is not known, never Ollama's."""
    disk = live.disk if live is not None else None
    read = disk is not None and disk.known
    places: list[tuple[str, Disk]] = []
    if ollama and read:
        places.append(("Ollama", disk))
    if disk is not None and disk.lmstudio is not None and disk.lmstudio.known:
        places.append(("LM Studio", disk.lmstudio))
    if vllm and disk is not None and disk.vllm is not None and disk.vllm.known:
        places.append(("vLLM", disk.vllm))
    pieces: list[str] = [_disk_piece(live, where, machine)] if ollama and not read else []
    drives: dict[str | None, list[tuple[str, Disk]]] = {}
    for name, place in places:
        drives.setdefault(place.drive, []).append((name, place))
    for members in drives.values():
        names = [name for name, _place in members]
        if len(members) > 1:
            pieces.append(shared_disk_words(members[0][1], names))
        elif names == ["Ollama"]:
            pieces.append(_disk_piece(live, where, machine))
        else:
            words = lmstudio_disk_words if names == ["LM Studio"] else vllm_disk_words
            pieces.append(words(members[0][1]))
    if not ollama and not places:
        pieces.append(_unread_disk_piece(where, machine))
    return pieces


def _unread_disk_piece(where: Where, machine: str) -> str:
    """The words for a place nobody read on a machine without Ollama: what `_disk_piece` says when no disk was
    read, and never Ollama's."""
    if where == "ssh_inactive":
        return f"disk unknown ({machine} not connected in this run)"
    return RENDER_DISK if where == "render" else ANOTHER_COMPUTER_DISK if where == "other" else DISK_UNKNOWN


def _disk_pieces(live: LiveRoom | None, where: Where, machine: str, ollama: bool, vllm: bool) -> list[str]:
    """The disk piece(s) of the `room` line (AP9-SV2-I, I2): Ollama's, as before, where Ollama is one of
    the machine's runtimes; vLLM's own place where vLLM is and it was read (`Disk.vllm`, this machine only).
    Both places on one drive are one piece, `disk 41.2 GB free on C: (Ollama and vLLM models)`; two
    drives, two pieces. A vLLM machine whose place was not read says nothing about it (as before)."""
    disk = live.disk if live is not None else None
    own = disk.vllm if vllm and disk is not None else None
    if own is None or not own.known:
        pieces = [_disk_piece(live, where, machine)] if ollama else []
        if vllm and where == "ssh":  # a server that did not say where vLLM keeps its models (AP9-SV2-R, R1)
            pieces.append(f"disk unknown on {machine}")
        return pieces
    if not ollama:
        return [vllm_disk_words(own)]
    # Over SSH the server says itself whether both places are on one drive (`Disk.same_place`).
    same = own.same_place if own.same_place is not None else disk.drive == own.drive
    if disk.known and same:
        return [shared_disk_words(disk)]
    return [_disk_piece(live, where, machine), vllm_disk_words(own)]


def _vllm_piece(block: MachineRanking) -> str:
    """`graphics up to 41.4 GB for vLLM (2 of 3 equal 22.5 GB cards, 0.92 of each)`: the most vLLM
    can take (`fit_vllm.largest_pool`, F1's own card choice, Codex r6-2) -- an upper bound, since
    each model's head count decides its own cards. Its own reason where vLLM cannot run at all."""
    profile = block.profile
    if profile.gpu_state == "unified_memory":
        return f"graphics: {SHARED_MEMORY_REASON}"
    pool = largest_pool(profile)
    if pool is None:
        return f"graphics: {NO_CARD_REASON}"
    share, size = VLLM_GPU_MEMORY_UTILIZATION, f"{pool.card_gib:.1f} GB"
    if pool.cards == 1:
        named = f"{size} card, {share} of it" if pool.group == 1 else f"1 of {pool.group} equal {size} cards, {share} of it"
    elif pool.cards == pool.group:
        named = f"{pool.cards} x {size} cards, {share} of each"
    else:
        named = f"{pool.cards} of {pool.group} equal {size} cards, {share} of each"
    return f"graphics up to {pool.pool_gib:.1f} GB for vLLM ({named})"


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


def _budget(value_gib: float, whom: str = "models") -> str:
    """A budget as the head says it; one of 0 or less is said in words, never below zero (RM-2).
    `whom` is `models`, or `Ollama` where vLLM stands beside it (F5)."""
    return f"{value_gib:.1f} GB for {whom}" if value_gib > 0 else f"nothing left for {whom}"


def _graphics_piece(block: MachineRanking, whom: str = "models") -> str:
    """`graphics 11.0 GB for models (12.0 GB card minus reserve)`, the cards counted or listed."""
    profile = block.profile
    if profile.gpu_state == "none" or (profile.gpu_state in _OWN_CARD_STATES and not profile.gpus):
        return "graphics none counted"
    if profile.gpu_state not in _OWN_CARD_STATES:
        return "graphics unknown"  # a state without a number: never `none`
    cards = [gpu.vram_gib for gpu in profile.gpus]
    budget = _budget(sum(card - block.reserve_vram_gib for card in cards), whom)
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


def _memory_piece(block: MachineRanking, whom: str = "models") -> str:
    """`memory 111.7 GB for models (127.7 GB minus reserve)`, or `memory unknown` (G1)."""
    ram = block.profile.ram_physical_gib
    if ram is None:
        return "memory unknown"
    return f"memory {_budget(ram - block.reserve_ram_gib, whom)} ({ram:.1f} GB minus reserve)"


def _shared_piece(block: MachineRanking, whom: str = "models") -> str:
    """Unified memory is one piece: `shared memory 95.0 GB for models (128.0 GB minus both reserves)`."""
    ram = block.profile.ram_physical_gib
    if ram is None:
        return "shared memory unknown"
    budget = _budget(ram - block.reserve_ram_gib - block.reserve_vram_gib, whom)
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
