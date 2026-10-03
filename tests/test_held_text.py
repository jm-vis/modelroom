"""`modelroom/held_text.py` (AP9-FR-N2, a): the words of the `local` line under a table.

The line itself is proven through the guided run (`tests/test_fr_nachlese.py`, `tests/test_guided_install.py`);
this file pins the rank text and the cases where the line stays out, on rows built the way the document builds them.
"""

from __future__ import annotations

from types import SimpleNamespace

from modelroom.held_text import local_pieces
from modelroom.room import Disk


def _entry(rank: int, identity: tuple[str, str, str], runtime: str = "ollama"):
    return SimpleNamespace(rank=rank, package_identity=list(identity), fit=SimpleNamespace(runtime=runtime))


def _identity(number: int) -> tuple[str, str, str]:
    return (f"base-{number}", f"packager/model-{number}-GGUF", "a" * 40)


def _block(*entries):
    return SimpleNamespace(ranked=list(entries))


def test_a_run_of_held_ranks_is_one_range_and_names_the_place():
    block = _block(*(_entry(rank, _identity(rank)) for rank in range(1, 7)))
    disk = Disk(10**12, "C:", held=frozenset(_identity(rank) for rank in (2, 3, 4, 5)))

    assert local_pieces(block, disk, "this computer") == ["#2–5 already in Ollama on this computer"]


def test_scattered_ranks_stay_apart_and_the_alias_names_the_computer():
    block = _block(*(_entry(rank, _identity(rank)) for rank in range(1, 7)))
    disk = Disk(10**12, "gpu1", held=frozenset(_identity(rank) for rank in (1, 4)))

    assert local_pieces(block, disk, "gpu1") == ["#1, #4 already in Ollama on gpu1"]


def test_the_ascii_dash_is_the_callers_to_choose():
    block = _block(*(_entry(rank, _identity(rank)) for rank in range(1, 4)))
    disk = Disk(10**12, "gpu1", held=frozenset(_identity(rank) for rank in (1, 2, 3)))

    assert local_pieces(block, disk, "gpu1", "-") == ["#1-3 already in Ollama on gpu1"]


def test_nothing_held_nothing_said():
    block = _block(_entry(1, _identity(1)))

    assert local_pieces(block, Disk(10**12, "C:"), "this computer") == []
    assert local_pieces(block, None, "this computer") == []


def test_a_vllm_row_is_never_called_held_in_ollama():
    block = _block(_entry(1, _identity(1), runtime="vllm"), _entry(2, _identity(2)))
    disk = Disk(10**12, "C:", held=frozenset({_identity(1), _identity(2)}))

    assert local_pieces(block, disk, "this computer") == ["#2 already in Ollama on this computer"]
