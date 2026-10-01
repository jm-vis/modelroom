"""F5 (AP9-SV2-F, decided 2026-10-01 after the hand test): one display per runtime.

A machine with vLLM alone shows vLLM's rows and key only; a machine with both shows one section per
runtime -- its heading, its table, its `memory` line, its key -- and a sentence that the sections are
alternatives; its numbers are unique across both, and the pull offers Ollama's rows alone. A profile
without the `runtimes` field (0.1.0) and an Ollama-only one draw exactly what they drew before.
Every test goes through a whole render document (`build_render_document`) and its views.
"""

from __future__ import annotations

import json

import pytest

from modelroom.profile import HardwareProfile
from modelroom.render import document_json
from modelroom.screen import FIT_HINT
from modelroom.views import document_markdown, document_terminal, terminal_lines

from test_golden_vllm import FIXTURES, GOLDEN, _base_model, _compare, _gguf_package, _profile, _render, _tensor_package

ALTERNATIVES = "One section per runtime: alternatives on the same cards, not both at once."
VLLM_KEY_START = "Fit is memory only: how many requests of this context the cards hold at once"


def _vllm_only_profile() -> HardwareProfile:
    """Four L4 cards and vLLM alone (`runtimes: ["vllm"]`), synthetic from `four_l4_server_vllm.json`."""
    data = json.loads((FIXTURES / "profiles_v4" / "four_l4_server_vllm_only.json").read_text(encoding="utf-8"))
    return HardwareProfile.model_validate(data)


def _profile_without_field() -> HardwareProfile:
    """The same four L4 cards as a 0.1.0 profile would read: no `runtimes` field at all."""
    return _profile().model_copy(update={"runtimes": []})


def _both():
    return _render(_profile(), [_base_model()], [_gguf_package(), _tensor_package()])


def _vllm_only():
    return _render(_vllm_only_profile(), [_base_model()], [_gguf_package(), _tensor_package()])


# --- vLLM alone -------------------------------------------------------------------------------------


def test_a_vllm_only_machine_shows_vllm_rows_and_vllm_key_alone():
    document = _vllm_only()

    block = document.machines[0]
    assert [entry.fit.runtime for entry in [*block.ranked, *block.not_covered, *block.no_fit]] == ["vllm"]
    markdown, terminal = document_markdown(document), document_terminal(document)
    for text in (markdown, terminal):
        assert VLLM_KEY_START in text
        assert FIT_HINT[0] not in text  # no Ollama key
        assert "unsloth" not in text and "Q4_K_M" not in text  # no Ollama row
        assert "vLLM ·" not in text and "vllm serve" not in text
        assert ALTERNATIVES not in text
    assert "runtime   vLLM (from its profile)" in terminal
    assert "Ollama" not in terminal


def test_the_markdown_of_a_vllm_only_machine_reads_as_the_golden_file():
    _compare("golden_vllm_only_four_l4.md", document_markdown(_vllm_only()))


def test_the_terminal_view_of_a_vllm_only_machine_reads_as_the_golden_file():
    _compare("golden_vllm_only_four_l4.txt", document_terminal(_vllm_only()) + "\n")


def test_the_json_of_a_vllm_only_machine_reads_as_the_golden_file():
    _compare("golden_vllm_only_four_l4.json", json.dumps(document_json(_vllm_only()), indent=2, sort_keys=True) + "\n")


# --- both runtimes ----------------------------------------------------------------------------------


def test_both_runtimes_show_two_sections_each_with_its_own_table_memory_and_key():
    document = _both()
    markdown, terminal = document_markdown(document), document_terminal(document)

    for text, heading in ((markdown, "### "), (terminal, " ")):
        assert ALTERNATIVES in text
        ollama, vllm = text.index(f"{heading}with Ollama"), text.index(f"{heading}with vLLM")
        assert ollama < vllm
        assert " · ".join(FIT_HINT) in markdown or FIT_HINT[0] in text[ollama:vllm]
        assert VLLM_KEY_START in text[vllm:] and VLLM_KEY_START not in text[ollama:vllm]
        assert FIT_HINT[0] not in text[vllm:]
        assert "vLLM ·" not in text and "vllm serve" not in text and "(Ollama)" not in text
    sections = terminal.split(" with vLLM")
    assert sections[0].count(" memory    ") == 1 and sections[1].count(" memory    ") == 1


def test_the_numbers_are_unique_over_both_sections_ollama_first():
    block = _both().machines[0]

    assert [(entry.rank, entry.fit.runtime) for entry in block.ranked] == [(1, "ollama"), (2, "vllm")]
    assert block.ranked_total == 2


def test_the_pull_offers_ollamas_rows_alone():
    from modelroom.pull_choice import pull_rows

    document = _both()
    rows = pull_rows(document, "server", packages=[_gguf_package(), _tensor_package()])

    assert [(row.rank, row.runtime) for row in rows] == [(1, "ollama")]
    install = [line for line in terminal_lines(document, install_for="server") if line.startswith(" install")]
    assert len(install) == 1 and "#1" in install[0]


# --- what stays as it was ----------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [_profile_without_field, lambda: _profile().model_copy(update={"runtimes": ["ollama"]})], ids=["0.1.0", "ollama-only"])
def test_a_profile_without_vllm_draws_what_it_drew_before(profile):
    """No vLLM row, no section, no runtime line: the tensor package stays invisible, and the views
    equal those of a snapshot that never had it."""
    with_tensor = _render(profile(), [_base_model()], [_gguf_package(), _tensor_package()])
    without = _render(profile(), [_base_model()], [_gguf_package()])

    assert document_terminal(with_tensor) == document_terminal(without)
    assert "runtime" not in document_terminal(with_tensor) and "with Ollama" not in document_markdown(with_tensor)


def test_one_visibility_rule_for_the_table_and_the_step_3_count():
    """`runtimes.visible_packages` alone decides what a machine shows and counts: vLLM alone hides
    Ollama's packages; no machine at all reads as Ollama's (`machine_runtime_keys`)."""
    from modelroom.runtimes import visible_packages

    gguf, tensor = _gguf_package(), _tensor_package()
    assert visible_packages([gguf, tensor], [(_vllm_only_profile(), None)]) == [tensor]
    assert visible_packages([gguf, tensor], [(_profile(), None)]) == [gguf, tensor]
    assert visible_packages([gguf, tensor], [(_profile_without_field(), None)]) == [gguf]
    assert visible_packages([gguf, tensor], []) == [gguf]


# --- Fix r5 (Codex r5 on 3474641): the rule says what each section shows; the head is per runtime -----


def _awq_of(index: int, base_repo: str = "Qwen/Qwen3-8B"):
    return _tensor_package().model_copy(update={"repo": f"acme{index:02d}/Qwen3-8B-AWQ", "base_model_hf_repo": base_repo})


def _other_model():
    from modelroom.contracts import Architecture

    arch = Architecture(
        source_repo="acme/Other-7B", source_revision="a" * 40, kind="dense_classic",
        num_hidden_layers=28, num_key_value_heads=4, head_dim=128, num_attention_heads=28, max_context=131072,
    )
    return _base_model().model_copy(update={"hf_repo": "acme/Other-7B", "publisher": "acme", "parameters_b": 7.0, "architecture": arch})


def _other_gguf():
    return _gguf_package().model_copy(update={"repo": "unsloth/Other-7B-GGUF", "base_model_hf_repo": "acme/Other-7B"})


def test_eleven_vllm_packages_beside_ollama_are_all_shown_and_the_rule_says_so():
    """Codex r5-1: the vLLM section of a machine with both runtimes lists every ranked vLLM
    package -- eleven here, past the first ten -- and the document's `ranking_rule` says exactly
    that; Ollama's own section keeps the rule's first ten."""
    from modelroom.ranking import RANKING_RULE

    document = _render(_profile(), [_base_model()], [_gguf_package(), *(_awq_of(index) for index in range(11))])

    block = document.machines[0]
    assert [entry.fit.runtime for entry in block.ranked] == ["ollama"] + ["vllm"] * 11
    assert document.ranking_rule.startswith(RANKING_RULE) and document.ranking_rule != RANKING_RULE
    assert "every ranked vLLM package" in document.ranking_rule
    assert f"Ranking rule: {document.ranking_rule}" in document_markdown(document)
    assert "Showing 11 of 11 ranked packages." in document_markdown(document)


def test_one_row_per_model_for_ollama_and_every_vllm_row_and_the_rule_says_so():
    """Codex r5-1 with the selection per model on (two models): Ollama's section shows one row per
    model, the vLLM section every vLLM package, and the per-model rule names both."""
    from modelroom.ranking import RANKING_RULE_PER_MODEL

    packages = [_gguf_package(), _other_gguf(), _awq_of(1), _awq_of(2)]
    document = _render(_profile(), [_base_model(), _other_model()], packages)

    block = document.machines[0]
    assert [entry.fit.runtime for entry in block.ranked] == ["ollama", "ollama", "vllm", "vllm"]
    assert document.ranking_rule.startswith(RANKING_RULE_PER_MODEL) and "every ranked vLLM package" in document.ranking_rule


@pytest.mark.parametrize("ollama_size", [300_000_000_000, 0], ids=["ollama_no_fit", "ollama_not_covered"])
def test_ollama_set_aside_and_eleven_vllm_rows_the_rule_says_what_is_shown(ollama_size):
    """Codex r6-1: Ollama's only package stands under `No fit` / `Not covered`, eleven vLLM packages
    rank. The block still has rows of both runtimes, so it is drawn in sections, the vLLM section
    is whole -- and the rule says so: one predicate decides all three (`runtimes.runtimes_among`)."""
    from modelroom.ranking import BOTH_RUNTIMES_RULE

    gguf = _gguf_package().model_copy(update={"files": [_gguf_package().files[0].model_copy(update={"size_bytes": ollama_size})]})
    document = _render(_profile(), [_base_model()], [gguf, *(_awq_of(index) for index in range(11))])

    block = document.machines[0]
    assert [entry.fit.runtime for entry in block.ranked] == ["vllm"] * 11
    assert [entry.fit.runtime for entry in [*block.no_fit, *block.not_covered]] == ["ollama"]
    assert document.ranking_rule.endswith(BOTH_RUNTIMES_RULE)
    assert ALTERNATIVES in document_markdown(document) and "Showing 11 of 11 ranked packages." in document_markdown(document)


def test_a_vllm_only_machine_with_eleven_packages_follows_the_plain_rule():
    """The counter-proof: one runtime, no sections -- the first ten rows, and the rule unchanged."""
    from modelroom.ranking import RANKING_RULE

    document = _render(_vllm_only_profile(), [_base_model()], [*(_awq_of(index) for index in range(11))])

    assert len(document.machines[0].ranked) == 10 and document.machines[0].ranked_total == 11
    assert document.ranking_rule == RANKING_RULE


def _cards(count: int, sizes: tuple[float, ...] | None = None, runtimes_: tuple[str, ...] = ("vllm",)) -> HardwareProfile:
    """The four-L4 vLLM-only profile cut to `count` equal cards (or the given `sizes`), synthetic."""
    data = json.loads((FIXTURES / "profiles_v4" / "four_l4_server_vllm_only.json").read_text(encoding="utf-8"))
    sizes = sizes or (22.49,) * count
    data["gpus"] = [{**data["gpus"][0], "index": index, "vram_gib": size} for index, size in enumerate(sizes)]
    data["vram_gib"] = round(sum(sizes), 2)
    data["gpu_state"] = "multi_gpu" if len(sizes) > 1 else "measured"
    data["llmfit_crosscheck"]["vram"] = {**data["llmfit_crosscheck"]["vram"], "own_gib": data["vram_gib"], "llmfit_gib": data["vram_gib"]}
    data["runtimes"] = list(runtimes_)
    return HardwareProfile.model_validate(data)


@pytest.mark.parametrize(
    "profile, said",
    [
        (lambda: _cards(4), "graphics up to 82.8 GB for vLLM (4 x 22.5 GB cards, 0.92 of each)"),
        (lambda: _cards(3), "graphics up to 41.4 GB for vLLM (2 of 3 equal 22.5 GB cards, 0.92 of each)"),
        (lambda: _cards(1), "graphics up to 20.7 GB for vLLM (22.5 GB card, 0.92 of it)"),
        (lambda: _cards(4, (22.49, 22.49, 22.49, 11.0)), "graphics up to 41.4 GB for vLLM (2 of 3 equal 22.5 GB cards, 0.92 of each)"),
    ],
    ids=["four", "three", "one", "mixed"],
)
def test_the_vllm_head_says_the_largest_pool_tensor_parallel_can_use(profile, said):
    """Codex r6-2: the head's vLLM pool is the most F1 can choose -- the largest power of two of
    1, 2, 4, 8 up to the equal group, 0.92 of the smallest chosen card -- through `fit_vllm`'s own
    choice, never the whole group summed (three cards gave 62.1 GB, two of them are usable)."""
    document = _render(profile(), [_base_model()], [_tensor_package()])

    room = next(line for line in document_terminal(document).splitlines() if line.startswith(" room"))
    assert said in room


def test_the_unequal_cards_line_names_the_usable_cards():
    """Codex r6-2: three equal cards and a smaller one -- vLLM splits over at most two of the three."""
    document = _render(_cards(4, (22.49, 22.49, 22.49, 11.0)), [_base_model()], [_tensor_package()])

    assert "vLLM can split over at most 2 of the 3 cards of 22.49 GB; 1 smaller card is left out" in document_terminal(document)


def test_an_ollama_only_document_keeps_its_rule_word_for_word():
    from modelroom.ranking import RANKING_RULE

    document = _render(_profile_without_field(), [_base_model()], [_gguf_package(), _tensor_package()])

    assert document.ranking_rule == RANKING_RULE


def _live(document, gib: float = 50.0):
    from modelroom.room import Disk, LiveRoom
    from modelroom.views import RunView, terminal_blocks
    from modelroom.screen import plain

    view = RunView(install_for="server", live={"server": LiveRoom(disk=Disk(free_bytes=int(gib * 1024**3), drive="C:"))})
    return "\n".join(plain(line) for line in terminal_blocks(document, view=view))


def test_a_vllm_only_head_names_the_vllm_pool_and_nothing_of_ollama():
    """Codex r5-2: the `room` line of a machine with vLLM alone says vLLM's pool (0.92 of each card
    of the largest equal group: 4 x 22.49 -> 82.8 GB), no system-memory budget (vLLM has no RAM
    fallback) and no disk said to be Ollama's -- with a disk read live, too."""
    terminal = _live(_vllm_only())

    room = next(line for line in terminal.splitlines() if line.startswith(" room"))
    assert "graphics up to 82.8 GB for vLLM" in room
    assert "memory" not in room and "Ollama" not in terminal and "for models" not in terminal


def test_an_empty_vllm_block_with_a_disk_read_says_nothing_of_ollama():
    """Codex r5-2: a vLLM-only machine with no visible row at all (only GGUF packages exist) -- its
    empty state carries no Ollama reserve hint and no Ollama disk, decided by the machine's runtimes."""
    document = _render(_vllm_only_profile(), [_base_model()], [_gguf_package()])
    assert document.machines[0].ranked == [] and document.machines[0].no_fit == []

    terminal, markdown = _live(document), document_markdown(document)
    for text in (terminal, markdown):
        assert "Ollama" not in text and "modelroom.toml" not in text and "Memory reserve" not in text


def test_both_runtimes_name_each_budget_by_its_runtime():
    """Codex r5-2: with both runtimes the head names Ollama's budgets and vLLM's pool apart."""
    terminal = _live(_both())

    room = [line for line in terminal.splitlines()[:8]]
    head = " ".join(room)
    assert "graphics 86.0 GB for Ollama" in head and "memory 86.3 GB for Ollama" in head
    assert "graphics up to 82.8 GB for vLLM" in head and "for models" not in head


def test_the_step_3_scale_counts_what_the_vllm_only_table_ranks():
    from modelroom.config import MachineConfig
    from modelroom.guided_context import Checked, fit_texts

    from modelroom.runtimes import visible_packages

    config = MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)
    checked = Checked(profile=_vllm_only_profile(), machine_config=config)

    packages = visible_packages([_gguf_package(), _tensor_package()], [(_vllm_only_profile(), None)])
    texts = fit_texts(packages, {_base_model().hf_repo: _base_model()}, checked, [32768], requests=10)

    assert texts[32768].startswith("all 1 package")
    assert len(_vllm_only().machines[0].ranked) == 1
