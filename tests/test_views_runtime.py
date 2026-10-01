"""Tests for modelroom.views_runtime: the extra words a table carries once vLLM is in it (F4).

Each function here is read directly, not through a whole rendered document: `views.py`'s own
golden files prove the wiring stays byte-identical for an Ollama-only table (CONTRACTS.md, "Fit
for vLLM"); these tests prove what each piece says once a row's `Fit.runtime` is `"vllm"`.
"""

from __future__ import annotations

import json

import pytest

from modelroom.contracts import Fit
from modelroom.document import RankedEntry
from modelroom.guided_contracts import Note
from modelroom.profile import HardwareProfile
from modelroom.views_runtime import (
    RUNTIME_LABEL,
    legend_lines,
    package_prefix,
    packager_cell,
    runtime_line,
    unequal_cards_note,
    vllm_pool_pieces,
)
from modelroom.vllm_server import ServedModel

from fixture_support import FIXTURES, v2_profile

GIB = 1024**3


def _note(text: str = "fits into graphics memory") -> Note:
    return Note(code="fits_in_graphics_memory", subject="package", origin="computed", text=text, facts=["fit.mode"])


def _fit(*, runtime: str = "ollama", cards: int = 1, at_once: int = 0, mode: str = "gpu") -> Fit:
    return Fit(
        fit_class="fits", mode=mode, need_gib=5.0, weights_gib=5.0, kv_gib=1.0, pool_gib=10.0,
        reserve_gib=1.0, context=8192, context_assumed=False, basis="architecture", reason=None,
        requests=1, runtime=runtime, at_once=at_once, cards=cards,
    )


def _entry(rank: int = 1, *, packager: str = "unsloth", quantization: str = "Q4_K_M", fit: Fit | None = None) -> RankedEntry:
    return RankedEntry(
        rank=rank,
        package_identity=("huggingface", "acme/Nova-7B", "Nova-7B-Q4_K_M.gguf"),
        base_model_hf_repo="acme/Nova-7B",
        packager=packager,
        quantization=quantization,
        format="gguf",
        weights_gib=5.0,
        package_context=None,
        provenance="from_metadata",
        note=_note(),
        fit=fit or _fit(),
        measurement_group=1,
    )


# --- the packager cell: the runtime's own prefix --------------------------------------------------


def test_ollama_carries_no_prefix():
    assert package_prefix("ollama") == ""


def test_vllm_carries_its_own_prefix():
    assert package_prefix("vllm") == "vLLM · "


def test_packager_cell_is_the_prefix_and_the_packager():
    entry = _entry(packager="RedHatAI", fit=_fit(runtime="vllm"))
    assert packager_cell(entry) == "vLLM · RedHatAI"
    assert packager_cell(_entry(packager="unsloth")) == "unsloth"


# --- the legend: one extra line per runtime that carries one, each at most once -------------------


def test_legend_is_empty_for_an_ollama_only_table():
    assert legend_lines([_entry()]) == []


def test_legend_names_vllm_once_even_with_several_of_its_rows():
    """Codex r1-5: once a vLLM row is shown, the legend also says the percentage rule above is
    Ollama's own and explains vLLM's classes by concurrent requests instead, before the one line
    per runtime that carries a `key_text`."""
    entries = [_entry(1, fit=_fit(runtime="vllm")), _entry(2, fit=_fit(runtime="vllm")), _entry(3)]

    lines = legend_lines(entries)

    assert lines == [
        "The percentage rule above is Ollama's own.",
        "vLLM's classes count requests at once instead: no fit below one, tight from one below the "
        "requests this ranking assumes, fits from that many below twice them, ample at twice them or "
        "more (architecture basis only).",
        "vLLM · rows start with vllm serve; the others are pulled into Ollama",
    ]


# --- the runtime row under a machine's head: only where vLLM is there ------------------------------


def test_no_row_for_a_machine_without_vllm():
    assert runtime_line(v2_profile()) is None
    assert runtime_line(None) is None


def _with_vllm(profile: HardwareProfile) -> HardwareProfile:
    return profile.model_copy(update={"runtimes": ["ollama", "vllm"]})


def test_profile_only_wording_for_modelroom_render():
    """`modelroom render` reads no machine live: the row says only what the profile remembers."""
    assert runtime_line(_with_vllm(v2_profile())) == "Ollama · vLLM (from its profile)"


def test_serving_names_the_model_when_it_is_one_of_this_runs_own():
    served = ServedModel(id="acme/Nova-7B", root="acme/Nova-7B", max_model_len=4096)
    assert runtime_line(_with_vllm(v2_profile()), served, served_matches_run=True) == "Ollama · vLLM serving Nova-7B"


def test_serving_says_another_model_when_it_is_not_this_runs_own():
    served = ServedModel(id="acme/Other-1B", root="acme/Other-1B", max_model_len=4096)
    assert runtime_line(_with_vllm(v2_profile()), served) == "Ollama · vLLM serving another model"


def test_a_server_that_answered_without_serving_a_model_reads_installed_not_serving():
    """E5: a key prompt or another non-`ServedModel` answer does not count as detected live, but it
    still proves vLLM is there when the profile already remembered it -- `installed, not serving`."""
    assert runtime_line(_with_vllm(v2_profile()), "asks for an API key") == "Ollama · vLLM installed, not serving"


def test_a_truthy_live_value_with_no_remembered_profile_is_still_not_shown():
    """E5 again, the other side: a key prompt alone does not count as detected, and with no stored
    `runtimes` either there is nothing to show at all."""
    assert runtime_line(v2_profile(), "asks for an API key") is None


# --- the unequal-cards continuation line ------------------------------------------------------------


def _profile_from(name: str) -> HardwareProfile:
    data = json.loads((FIXTURES / "profiles_v4" / name).read_text(encoding="utf-8"))
    return HardwareProfile.model_validate(data)


def test_no_note_for_a_machine_of_one_card_size():
    assert unequal_cards_note(_with_vllm(_profile_from("four_l4_server_vllm.json"))) is None
    assert unequal_cards_note(None) is None


def test_the_unequal_cards_note_names_the_group_and_what_was_left_out():
    """Codex r1-4: this says what the machine *offers* (the equal-size group), never what one
    package's row actually used -- `t` is chosen per package and can fall short of the group
    (e.g. a head count that does not divide 3), so the wording says "can split over", not "uses"."""
    profile = _profile_from("mixed_cards_server.json")

    note = unequal_cards_note(profile)

    assert note == "vLLM can split over the 3 cards of 22.49 GB; 1 smaller card is left out"


# --- the memory line's own vLLM piece: never an Ollama gpu/gpu_split piece -------------------------


def test_vllm_pool_pieces_is_empty_without_vllm_rows():
    assert vllm_pool_pieces([], "–") == []


def test_one_piece_per_distinct_card_count_with_its_own_ranks():
    entries = [
        _entry(1, fit=_fit(runtime="vllm", cards=4, mode="gpu_split")),
        _entry(2, fit=_fit(runtime="vllm", cards=4, mode="gpu_split")),
        _entry(3, fit=_fit(runtime="vllm", cards=1, mode="gpu")),
    ]

    pieces = vllm_pool_pieces(entries, "–")

    assert pieces == [
        "#3 vLLM on 1 card: 0.92 of each, runtime 1.50 GB per card (one recording)",
        "#1–2 vLLM on 4 cards: 0.92 of each, runtime 1.50 GB per card (one recording)",
    ]


def test_runtime_label_is_eight_wide_like_every_other_label():
    from modelroom.intro import LABEL_WIDTH

    assert RUNTIME_LABEL == "runtime" and len(RUNTIME_LABEL) <= LABEL_WIDTH


# --- the terminal table's own package cell: the prefix counts toward the cut ------------------------


def test_the_package_cell_carries_the_prefix_and_cuts_with_ellipsis():
    """`views._package_cell`: the prefix is part of the 21-wide `Package` column, not added after
    the cut -- a long packager name of a vLLM row is cut shorter than the same name would be on an
    Ollama row, because `"vLLM · "` already took some of the room."""
    from modelroom.views import _package_cell

    long_name = "RedHatAI-very-long-account-name"
    ollama_row = _entry(packager=long_name, quantization="Q4_K_M")
    vllm_row = _entry(packager=long_name, quantization="Q4_K_M", fit=_fit(runtime="vllm"))

    ollama_cell = _package_cell(ollama_row)
    vllm_cell = _package_cell(vllm_row)

    assert len(ollama_cell) == 21 and ollama_cell.endswith("…")
    assert len(vllm_cell) == 21 and vllm_cell.endswith("…")
    assert vllm_cell.startswith("vLLM · ")
    # The same column width, but the prefix eats into it: less of `long_name` survives uncut.
    ollama_name_shown = ollama_cell.rstrip("…")
    vllm_name_shown = vllm_cell.removeprefix("vLLM · ").rstrip("…")
    assert len(vllm_name_shown) < len(ollama_name_shown)
    assert long_name.startswith(vllm_name_shown) and long_name.startswith(ollama_name_shown)
