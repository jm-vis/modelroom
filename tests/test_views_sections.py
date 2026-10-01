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
