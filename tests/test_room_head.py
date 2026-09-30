"""The head under a machine's table: budget, not the free memory of the moment (AP9-RM U10, ADR 0012).

`room_head` is pure: the budgets come from the profile and the reserves of the block, the live
values from a `LiveRoom`. Live values belong to the machine they were read on, whichever it is;
`where` only chooses the words for a machine nobody read.
"""

from __future__ import annotations

from modelroom.document import MachineRanking
from modelroom.profile import GpuAdapter
from modelroom.render import build_render_document
from modelroom.measurements import default_scenario
from modelroom.room import GIB, Disk, LiveRoom, LoadedModel
from modelroom.room_head import NOTE_WIDTH, head_pieces, loaded_pieces
from modelroom.screen import plain
from modelroom.views import RunView, _room_lines, terminal_blocks

from test_render import RENDERED_AT, _config, _hf_package, _profile, _ready, _snapshot

DISK = Disk(int(41.2 * GIB), "C:")
QWEN = LoadedModel("qwen3.5:9b", int(7.4 * GIB), int(6.9 * GIB))


def _cards(*sizes: float) -> list[GpuAdapter]:
    return [GpuAdapter(index=index, name="Nova GPU", vram_gib=size, vram_source="nvidia-smi") for index, size in enumerate(sizes)]


def _block(*, ram_gib: float | None = 127.7, cards=(12.0,), state: str | None = None, reserve: float = 16.0) -> MachineRanking:
    """A ranked block, its profile changed without validation: the head reads only these fields."""
    state = state or ("multi_gpu" if len(cards) > 1 else "measured" if cards else "none")
    profile = _profile(vram_gib=12.0).model_copy(
        update={
            "ram_physical_gib": ram_gib,
            "ram_physical_source": "os" if ram_gib is not None else "unknown",
            "gpu_state": state,
            "gpus": _cards(*cards),
            "vram_gib": float(sum(cards)),
            **({} if cards else {"vram_source": "none", "gpu_name": None}),
        }
    )
    return MachineRanking(
        machine="laptop", status="ranked", label="laptop", profile=profile,
        reserve_ram_gib=reserve, reserve_vram_gib=1.0, reserve_ram_origin="computed",
    )


def _room_text(block: MachineRanking, view: RunView) -> list[str]:
    return [plain(line) for line in _room_lines(block, view, "·")]


# --- the budget ----------------------------------------------------------------------------------


def test_one_card_names_budget_and_card():
    assert head_pieces(_block(), None, "render")[0] == "graphics 11.0 GB for models (12.0 GB card minus reserve)"


def test_four_equal_cards_are_counted():
    piece = head_pieces(_block(cards=(22.5,) * 4, reserve=16.0), None, "other")[0]
    assert piece == "graphics 86.0 GB for models (4 x 22.5 GB cards minus reserve)"


def test_unequal_cards_are_listed():
    piece = head_pieces(_block(cards=(22.5, 22.5, 11.9)), None, "other")[0]
    assert piece == "graphics 53.9 GB for models (22.5 + 22.5 + 11.9 GB cards minus reserve)"


def test_no_counted_card():
    assert head_pieces(_block(cards=()), None, "this")[0] == "graphics none counted"


def test_a_state_without_numbers_is_unknown_never_none():
    for state in ("present_unmeasured", "unsupported_platform", "multi_gpu_not_covered"):
        piece = head_pieces(_block(cards=(), state=state), None, "this")[0]
        assert piece == "graphics unknown"


def test_unified_memory_is_one_piece():
    pieces = head_pieces(_block(ram_gib=128.0, cards=(), state="unified_memory", reserve=32.0), None, "render")
    assert pieces[0] == "shared memory 95.0 GB for models (128.0 GB minus both reserves)"
    assert not any(piece.startswith(("graphics", "memory")) for piece in pieces)


def test_memory_names_its_budget_and_size():
    assert head_pieces(_block(), None, "render")[1] == "memory 111.7 GB for models (127.7 GB minus reserve)"
    assert head_pieces(_block(ram_gib=None), None, "render")[1] == "memory unknown"
    assert head_pieces(_block(ram_gib=3.81, reserve=8.0), None, "render")[1] == "memory nothing left for models (3.8 GB minus reserve)"


def test_another_computer_has_budget_and_disk_unknown():
    pieces = head_pieces(_block(cards=(22.5,) * 4), None, "other")
    assert pieces == [
        "graphics 86.0 GB for models (4 x 22.5 GB cards minus reserve)",
        "memory 111.7 GB for models (127.7 GB minus reserve)",
        "disk unknown on another computer",
    ]


def test_render_says_it_reads_no_machine_live():
    assert head_pieces(_block(), None, "render")[-1] == "disk unknown (modelroom render reads no machine live)"


def test_this_machine_without_live_data_says_disk_unknown():
    assert head_pieces(_block(), None, "this")[-1] == "disk unknown"
    assert head_pieces(_block(), LiveRoom(disk=Disk(None, None, "two places Ollama may use")), "this")[-1] == (
        "disk unknown (two places Ollama may use)"
    )


def test_the_active_ssh_machine_names_its_disk_by_alias_where_ollama_keeps_its_models():
    """FR-B1, B5, the brief's own wording: known, unclear, and a failed connection (already named
    by `guided_remote.read_room` in `disk.reason`) -- and never the local wording ("Ollama models")."""
    known = LiveRoom(disk=Disk(int(812.4 * GIB), "gpu1"))
    assert head_pieces(_block(), known, "ssh")[-1] == "disk 812.4 GB free on gpu1 (where Ollama keeps its models)"

    unclear = LiveRoom(disk=Disk(None, "gpu1", "two places Ollama may use"))
    assert head_pieces(_block(), unclear, "ssh")[-1] == "disk unknown on gpu1 (two places Ollama may use)"

    unreachable = LiveRoom(disk=Disk(None, "gpu1", "gpu1 not reached"))
    assert head_pieces(_block(), unreachable, "ssh")[-1] == "disk unknown (gpu1 not reached)"


def test_the_local_machines_own_disk_wording_is_unchanged_by_the_ssh_case():
    """The brief's SSH wording ("where Ollama keeps its models") must not leak onto "this"/"other"."""
    assert head_pieces(_block(), LiveRoom(disk=DISK), "this")[-1] == "disk 41.2 GB free on C: (Ollama models)"
    assert "where Ollama keeps its models" not in head_pieces(_block(), LiveRoom(disk=DISK), "this")[-1]


def test_the_head_never_says_free_now():
    live = LiveRoom(disk=DISK, loaded=(), graphics_free_mib=(512.0,))
    lines = _room_text(_block(), RunView(install_for="laptop", live={"laptop": live}))
    assert lines[0].split()[0] == "room"
    assert not any("free now" in line for line in lines)
    assert "disk 41.2 GB free on C: (Ollama models)" in " ".join(lines)


def test_live_data_never_change_the_budget():
    live = LiveRoom(disk=DISK, loaded=(QWEN,), graphics_free_mib=(512.0,))
    assert head_pieces(_block(), live, "this")[:2] == head_pieces(_block(), None, "render")[:2]


# --- the `loaded` line --------------------------------------------------------------------------


def test_the_loaded_line_names_each_model_and_the_graphics_of_now():
    live = LiveRoom(disk=DISK, loaded=(QWEN,), graphics_free_mib=(512.0,))
    assert loaded_pieces(live) == ["Ollama holds qwen3.5:9b (7.4 GB, 6.9 GB of it on the card)", "graphics 0.5 GB free now"]
    assert loaded_pieces(LiveRoom(loaded=(QWEN,))) == ["Ollama holds qwen3.5:9b (7.4 GB, 6.9 GB of it on the card)"]


def test_the_loaded_line_appears_only_with_a_loaded_model():
    view = RunView(install_for="laptop", live={"laptop": LiveRoom(disk=DISK, loaded=(), graphics_free_mib=(512.0,))})
    assert not any(line.split()[0] == "loaded" for line in _room_text(_block(), view))
    for live in (LiveRoom(disk=DISK), None):
        assert loaded_pieces(live) == []
    view = RunView(install_for="laptop", live={"laptop": LiveRoom(disk=DISK, loaded=(QWEN,))})
    assert [line.split()[0] for line in _room_text(_block(), view)][-1] == "loaded"


def test_a_long_loaded_model_name_is_cut_to_fit():
    long = LoadedModel("hf.co/an-account-with-a-long-name/" + "Very-Long-Model-Name-" * 4 + "GGUF:Q4_K_M", 7 * GIB, 7 * GIB)
    [piece] = loaded_pieces(LiveRoom(loaded=(long,)))
    assert len(piece) == NOTE_WIDTH and "…" in piece and piece.endswith("of it on the card)")


def test_a_card_list_too_long_becomes_a_count():
    piece = head_pieces(_block(cards=(24.0, 23.9, 23.8, 23.7, 23.6, 23.5, 23.4, 23.3)), None, "other")[0]
    assert piece == "graphics 181.2 GB for models (8 cards, 189.2 GB in all minus reserve)"


def test_every_room_line_stays_within_100_columns():
    """Four cards, eight different cards, a long model name and several models: no line over 100."""
    long = LoadedModel("hf.co/an-account-with-a-long-name/" + "Very-Long-Model-Name-" * 4 + "GGUF:Q4_K_M", 7 * GIB, 7 * GIB)
    live = LiveRoom(disk=DISK, loaded=(QWEN, long, QWEN), graphics_free_mib=(512.0,) * 8)
    for cards in ((22.5,) * 4, (24.0, 23.9, 23.8, 23.7, 23.6, 23.5, 23.4, 23.3)):
        lines = _room_text(_block(cards=cards), RunView(install_for="laptop", live={"laptop": live}))
        assert all(len(line) <= 100 for line in lines), lines


# --- live data belong to their machine ----------------------------------------------------------


def test_live_data_of_another_machine_are_shown_for_it():
    """AP9-FR fills more entries of `RunView.live`: they show under that machine's block, not this one's."""
    server = _profile(vram_gib=22.5, ram_gib=512.0, profile_id="b1b2b3b4b5b6b7b8")
    config = _config(
        {
            "workstation": {"reserve_vram_gib": 1.0, "writer": True, "profile": "3f9a0c21d4e6b870"},
            "server": {"reserve_vram_gib": 1.0, "writer": False, "profile": "b1b2b3b4b5b6b7b8"},
        }
    )
    document = build_render_document(
        config, _snapshot([_hf_package()]), {"workstation": _ready(), "server": _ready(server)}, {}, default_scenario(), RENDERED_AT
    )
    remote = LiveRoom(disk=Disk(int(900 * GIB), "/srv"), loaded=(QWEN,))
    view = RunView(install_for="workstation", live={"server": remote})
    blocks = {block.machine: block for block in document.machines}
    own, other = ([plain(line) for line in _room_lines(blocks[name], view, "·")] for name in ("workstation", "server"))
    assert len(terminal_blocks(document, "workstation", view=view)) > len(own) + len(other)
    assert any("disk 900.0 GB free on /srv (Ollama models)" in line for line in other)
    assert any(line.split()[0] == "loaded" for line in other if line.strip())
    assert not any("900.0 GB" in line or line.strip().startswith("loaded") for line in own)
    assert any("disk unknown" in line and "another computer" not in line for line in own)
