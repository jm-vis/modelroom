"""Tests for modelroom.guided_models: one line per model in step 2 of the guided mode.

Every hit here is built in memory; the fit comes from the size of the model, computed against a
schema-2 profile built by `fixture_support.v2_profile` (CONTRACTS.md, "Guided mode", step 2).
"""

from __future__ import annotations

import pytest

from modelroom.config import MachineConfig
from modelroom.guided_context import Checked
from modelroom.guided_contracts import SearchHit
from modelroom.guided_models import (
    LINE_LIMIT,
    MACHINE_NOT_MEASURED,
    PARAMETER_COUNT_UNKNOWN,
    ModelChoice,
    chosen_hits,
    count_line,
    fit_line,
    hint_line,
    list_choices,
    list_context,
    model_choices,
    parameters_from_name,
    unusable_reasons,
)
from modelroom.search import DEFAULT_PACKAGERS

from fixture_support import v2_profile

QWEN = "Qwen/Qwen3.5-9B"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MACHINE = MachineConfig(reserve_ram_gib=16.0, reserve_vram_gib=1.0, writer=True)


def _measured(vram_gib: float = 11.94, ram_gib: float = 127.46) -> Checked:
    return Checked(name="workstation", profile=v2_profile(vram_gib=vram_gib, ram_gib=ram_gib), machine_config=MACHINE)


def _unmeasured() -> Checked:
    return Checked(reason="no measured machine in this folder yet, so the scale shows no fit")


def _hit(repo: str, base: str = QWEN, owner: str = "listed packager", **overrides) -> SearchHit:
    return SearchHit(
        repo=repo,
        publisher_status=owner,
        base_model=[base],
        base_model_relation="quantized",
        resolved=True,
        resolved_base_model=base,
        **overrides,
    )


def _unresolved(repo: str, reason: str = "relation_unknown") -> SearchHit:
    return SearchHit(repo=repo, publisher_status="other", resolved=False, unresolved_reason=reason)


def _models(hits, *, filtered: bool = True, checked: Checked | None = None, context: int = 8192) -> list[ModelChoice]:
    return model_choices(hits, filtered=filtered, checked=checked if checked is not None else _measured(), context=context)


# --- one line per model, not per repository -------------------------------------------------------


def test_every_repository_of_one_base_model_becomes_one_line():
    hits = [
        _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher", downloads=12_031_627),
        _hit("unsloth/Qwen3.5-9B-GGUF", downloads=1_626_475),
        _hit("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", base=DEEPSEEK, downloads=68_445),
    ]

    models = _models(hits)

    assert [model.base_model for model in models] == [QWEN, DEEPSEEK]
    qwen = models[0]
    assert (qwen.name, qwen.publisher) == ("Qwen3.5-9B", "Qwen")
    assert [hit.repo for hit in qwen.repos] == ["Qwen/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-9B-GGUF"]
    assert qwen.packagers == ["Qwen", "unsloth"]
    assert qwen.downloads == 12_031_627 + 1_626_475


def test_the_group_order_of_the_search_is_the_order_of_the_repositories():
    """The publisher page is asked first, so its repository stands first (decided 2026-09-24)."""
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF"), _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher")]

    assert _models(hits)[0].packagers == ["unsloth", "Qwen"]


def test_downloads_are_summed_over_the_repositories_that_have_a_count():
    hits = [_hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher", downloads=1_000), _hit("unsloth/Qwen3.5-9B-GGUF")]

    assert _models(hits)[0].downloads == 1_000


def test_downloads_stay_unknown_when_no_repository_has_a_count():
    assert _models([_hit("unsloth/Qwen3.5-9B-GGUF")])[0].downloads is None


def test_the_ollama_name_and_the_age_come_from_the_repositories():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF", ollama="qwen3.5:9b", age="latest")]

    model = _models(hits)[0]

    assert (model.ollama, model.age) == ("qwen3.5:9b", "latest")


def test_an_unresolved_repository_is_no_model_and_counts_behind_its_reason():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF"), _unresolved("someone/Qwen3.5-9B-GGUF")]

    assert len(_models(hits)) == 1
    assert unusable_reasons(hits, True) == [
        "1 repository cannot be picked: the repository does not say it packages a base model"
    ]


def test_with_the_filter_on_a_model_only_other_accounts_hold_is_not_offered():
    hits = [_hit("community-user/Qwen3.5-9B-GGUF", owner="other")]

    assert _models(hits, filtered=True) == []
    assert _models(hits, filtered=False)[0].packagers == ["community-user"]
    assert unusable_reasons(hits, True) == ["1 repository cannot be picked: not a publisher or a listed packager"]
    assert unusable_reasons(hits, False) == []


def test_with_the_filter_on_an_other_account_is_no_target_of_a_model_that_is_offered():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF"), _hit("community-user/Qwen3.5-9B-GGUF", owner="other")]

    assert _models(hits, filtered=True)[0].packagers == ["unsloth"]
    assert _models(hits, filtered=False)[0].packagers == ["unsloth", "community-user"]


# --- the parameter count from the name ------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Qwen3.5-9B", 9.0),
        ("Qwen3.5-9B-GGUF", 9.0),
        ("Qwen3.5-35B-A3B-MTP", 35.0),
        ("Qwen3.8-2.4T-A95B", 2400.0),
        ("DeepSeek-R1-0528-Qwen3-8B", 8.0),
        ("Qwen3-Embedding-0.6B", 0.6),
        ("Qwen-Image-2.1", None),
        ("EuroLLM-9B-Instruct-2512", 9.0),
        # `8x7B` is no total: the count stands behind a letter, so the name says nothing.
        ("Mixtral-8x7B-Instruct", None),
        # The active parameters of a mixture keep their letter in front of them and never count.
        ("Nova-35B-A3B", 35.0),
        ("Nova-A17B", None),
        # Nor does the tail of a decimal one (second-model round, 2026-09-24: `A0.6B` read as 6).
        ("Nova-A0.6B", None),
        ("Nova-8x0.5B", None),
        # A count of zero is no count: it would otherwise be a model of no size at all.
        ("Nova-0B", None),
        # A number no float can hold is no count either.
        ("Nova-" + "9" * 400 + "B", None),
        ("Nova-8b", 8.0),
        ("Nova", None),
    ],
)
def test_the_parameter_count_is_read_from_the_name(name, expected):
    assert parameters_from_name(name) == expected


def test_a_parameter_count_of_a_repository_wins_over_the_name():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF", parameters_b=9.65)]

    assert _models(hits)[0].parameters_b == pytest.approx(9.65)


def test_without_a_parameter_count_anywhere_the_name_answers():
    assert _models([_hit("unsloth/Qwen3.5-9B-GGUF")])[0].parameters_b == 9.0


# --- the fit of the list --------------------------------------------------------------------------


def test_a_measured_machine_gets_a_fit_from_the_size():
    model = _models([_hit("unsloth/Qwen3.5-9B-GGUF")])[0]

    assert model.fit.basis == "size"
    assert model.fit.fit_class == "good"
    assert model.fit.reason is None
    assert model.fit_word == "good"


def test_a_model_too_large_for_this_machine_is_too_tight():
    hits = [_hit("unsloth/Nova-400B-GGUF", base="acme/Nova-400B")]
    model = _models(hits, checked=_measured(ram_gib=31.7))[0]

    assert model.fit.fit_class == "too_tight"
    assert model.fit_word == "too tight"


def test_without_a_measured_machine_the_fit_is_unknown_with_that_reason():
    model = _models([_hit("unsloth/Qwen3.5-9B-GGUF")], checked=_unmeasured())[0]

    assert (model.fit.fit_class, model.fit.reason) == ("unknown", MACHINE_NOT_MEASURED)
    assert model.fit_word == "unknown"


def test_without_a_parameter_count_the_fit_is_unknown_with_that_reason():
    model = _models([_hit("unsloth/Nova-GGUF", base="acme/Nova")])[0]

    assert (model.fit.fit_class, model.fit.reason) == ("unknown", PARAMETER_COUNT_UNKNOWN)


def test_the_context_of_the_folder_is_the_context_of_the_fit():
    small = _models([_hit("unsloth/Qwen3.5-9B-GGUF")], context=8192)[0]
    large = _models([_hit("unsloth/Qwen3.5-9B-GGUF")], context=131072)[0]

    assert large.fit.kv_gib > small.fit.kv_gib
    assert large.fit.context == 131072


def test_a_model_the_graphics_memory_cannot_hold_says_ram_behind_its_fit():
    """Live on 2026-09-24 a 122B model stood as `good` above a 9B `marginal`: fit v1 caps the
    system-memory pool at `good`, and the list said nothing about the pool. Now it does."""
    hits = [_hit("unsloth/Nova-27B-GGUF", base="acme/Nova-27B"), _hit("unsloth/Nova-9B-GGUF", base="acme/Nova-9B")]

    models = {model.name: model for model in _models(hits, context=32768)}

    assert models["Nova-27B"].fit.mode == "cpu_gpu"
    assert models["Nova-27B"].fit_word == "good (RAM)"
    assert models["Nova-9B"].fit.mode == "gpu"
    assert models["Nova-9B"].fit_word == "marginal"


def test_too_tight_and_unknown_carry_no_pool():
    too_large = _models([_hit("unsloth/Nova-400B-GGUF", base="acme/Nova-400B")], checked=_measured(ram_gib=31.7))[0]
    unknown = _models([_hit("unsloth/Nova-GGUF", base="acme/Nova")])[0]

    assert (too_large.fit_word, unknown.fit_word) == ("too tight", "unknown")


def test_a_fit_in_graphics_memory_stands_above_a_fit_in_ram_whatever_the_class():
    hits = [
        _hit("unsloth/Nova-27B-GGUF", base="acme/Nova-27B", downloads=9_000_000),
        _hit("unsloth/Nova-9B-GGUF", base="acme/Nova-9B", downloads=10),
        _hit("unsloth/Nova-2B-GGUF", base="acme/Nova-2B", downloads=5),
    ]

    models = _models(hits, context=32768)

    assert [model.fit_word for model in models] == ["good", "marginal", "good (RAM)"]


# --- the list itself ------------------------------------------------------------------------------


def _labels(models) -> list[str]:
    return [choice.label for choice in list_choices(models)]


def test_the_list_shows_name_fit_size_packagers_downloads_and_ollama():
    hits = [
        _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher", downloads=12_031_627, ollama="qwen3.5:9b"),
        _hit("unsloth/Qwen3.5-9B-GGUF", downloads=1_626_475, ollama="qwen3.5:9b"),
    ]

    label = _labels(_models(hits))[0]

    assert label.startswith("Qwen3.5-9B")
    assert "good" in label
    assert "9B" in label
    assert "Qwen, unsloth" in label
    assert "13.6M" in label
    assert label.rstrip().endswith("qwen3.5:9b")


def test_a_model_without_an_ollama_name_says_none_known():
    assert _labels(_models([_hit("unsloth/Qwen3.5-9B-GGUF")]))[0].rstrip().endswith("none known")


def test_an_account_with_two_builds_of_one_model_stands_once():
    """`unsloth, unsloth +4` was the live list of 2026-09-24; the column names accounts."""
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF"), _hit("unsloth/Qwen3.5-9B-UD-GGUF"), _hit("bartowski/Qwen3.5-9B-GGUF")]

    model = _models(hits)[0]

    assert model.packagers == ["unsloth", "bartowski"]
    assert len(model.repos) == 3


def test_more_than_three_packagers_are_shortened_with_a_count():
    hits = [_hit(f"{owner}/Qwen3.5-9B-GGUF") for owner in ("unsloth", "bartowski", "mradermacher", "ggml-org", "lmstudio-community")]

    label = _labels(_models(hits))[0]

    assert "unsloth, bartowski +3" in label


def test_no_line_of_the_list_is_wider_than_a_hundred_characters():
    hits = [
        _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher", downloads=12_031_627, ollama="qwen3.5:9b"),
        _hit("unsloth/Qwen3.5-9B-GGUF", downloads=1_626_475, ollama="qwen3.5:9b"),
        _hit("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", base=DEEPSEEK, downloads=68_445),
    ]

    for label in _labels(_models(hits)):
        assert len(label) <= LINE_LIMIT, label


def test_the_hundred_characters_hold_for_the_longest_names_there_are():
    """104 characters with `deepseek-r1:8b` in the second-model round of 2026-09-24.

    The three cells that can grow are the model's name, the packager accounts and the Ollama
    name, so each of them is cut to its column and the last one to what is left of the line.
    """
    hits = [
        _hit(
            "unsloth/Qwen3-Omni-30B-A3B-Instruct-GGUF",
            base="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            downloads=12_031_627,
            ollama="qwen3-omni-30b-a3b-instruct:q4_k_m",
        ),
        _hit("lmstudio-community/Nova-9B-GGUF", base="acme/Nova-9B", downloads=9, ollama="deepseek-r1:8b"),
        _hit("mradermacher/Nova-9B-GGUF", base="acme/Nova-9B", downloads=9),
    ]

    labels = _labels(_models(hits))

    assert len(labels) == 2
    for label in labels:
        assert len(label) <= LINE_LIMIT, f"{len(label)}: {label}"
    omni = next(label for label in labels if label.startswith("Qwen3-Omni"))
    nova = next(label for label in labels if label.startswith("Nova-9B"))
    assert "…" in omni, "a cell that does not fit its column says so"
    # The 14 characters left of the line are exactly `deepseek-r1:8b` (acceptance of 2026-09-24),
    # so an Ollama name of an ordinary length is shown whole.
    assert nova.rstrip().endswith("deepseek-r1:8b")


def test_a_size_and_a_download_count_of_any_length_keep_their_columns():
    """Both come from a registry: a count of 1,234,567 B parameters and a huge download number."""
    hits = [
        _hit("unsloth/Nova-1234567B-GGUF", base="acme/Nova-1234567B", downloads=999_999_999_999),
    ]

    label = _labels(_models(hits))[0]

    assert len(label) <= LINE_LIMIT, f"{len(label)}: {label}"


def test_the_order_is_the_fit_class_then_the_downloads_then_the_name():
    hits = [
        _hit("unsloth/Nova-400B-GGUF", base="acme/Nova-400B", downloads=9_000_000),  # too tight
        _hit("unsloth/Nova-9B-GGUF", base="acme/Nova-9B", downloads=1_000),
        _hit("unsloth/Nova-8B-GGUF", base="acme/Nova-8B", downloads=2_000),
        _hit("unsloth/Nova-One-GGUF", base="acme/Nova-One"),  # no parameter count: unknown
    ]

    models = _models(hits, checked=_measured(ram_gib=31.7))

    assert [model.name for model in models] == ["Nova-8B", "Nova-9B", "Nova-400B", "Nova-One"]


def test_the_values_of_the_list_are_the_base_models():
    assert [choice.value for choice in list_choices(_models([_hit("unsloth/Qwen3.5-9B-GGUF")]))] == [QWEN]


def test_nothing_of_the_list_is_grayed_out():
    assert [choice.disabled for choice in list_choices(_models([_hit("unsloth/Qwen3.5-9B-GGUF")]))] == [None]


# --- the lines around the list ---------------------------------------------------------------------


def test_the_count_line_names_the_models_and_the_repositories_that_cannot_be_used():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF"), _unresolved("someone/Qwen3.5-9B-GGUF")]

    assert count_line(_models(hits), hits, True) == "1 model can be picked; 1 repository cannot:"


def test_the_count_line_counts_in_the_plural_as_well():
    hits = [
        _hit("unsloth/Qwen3.5-9B-GGUF"),
        _hit("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", base=DEEPSEEK),
        _unresolved("someone/Qwen3.5-9B-GGUF"),
        _unresolved("another/Qwen3.5-9B-GGUF", "derivative"),
    ]

    assert count_line(_models(hits), hits, True) == "2 models can be picked; 2 repositories cannot:"


def test_the_context_of_the_list_is_the_one_this_folder_kept():
    """Step 3 asks for the context; step 2 stands on what the folder holds, else fit v1's 8192."""
    assert list_context(None) == 8192
    assert list_context(32768) == 32768


def test_the_fit_line_names_the_context_the_fit_was_computed_for():
    assert fit_line(8192, _measured()) == "fit at 8k context, computed from the size of the model"
    assert fit_line(32768, _measured()).startswith("fit at 32k context")


def test_a_context_of_its_own_is_not_floored_into_a_k_it_is_not():
    assert fit_line(5000, _measured()).startswith("fit at 5000 tokens context")


def test_without_a_measured_machine_there_is_no_fit_line():
    assert fit_line(8192, _unmeasured()) is None


def test_the_hint_line_says_how_to_mark_and_where_the_exact_fit_comes_from():
    line = hint_line(_measured())

    assert line.startswith("Space marks a model, Enter confirms; nothing marked keeps the folder as it is.")
    assert line.endswith(
        "Fit is from the size of the model, the exact fit comes after the fetch; "
        "(RAM) means the graphics memory is too small for it."
    )


def test_the_hint_line_of_an_unmeasured_machine_says_the_fit_is_unknown():
    line = hint_line(_unmeasured())

    assert line.startswith("Space marks a model, Enter confirms; nothing marked keeps the folder as it is.")
    assert line.endswith("Fit is unknown until this machine is measured.")


# --- from the answer to the repositories that are fetched -------------------------------------------


def test_a_chosen_model_fetches_the_publisher_and_the_first_listed_packager():
    """Two repositories per model: every one costs the fetch about ten requests of the budget."""
    hits = [
        _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher"),
        _hit("bartowski/Qwen3.5-9B-GGUF"),
        _hit("unsloth/Qwen3.5-9B-GGUF"),
    ]
    models = _models(hits)

    chosen = chosen_hits(models, [QWEN], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["Qwen/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-9B-GGUF"]


def test_a_model_without_a_publisher_repository_fetches_the_listed_packager_alone():
    hits = [_hit("bartowski/Qwen3.5-9B-GGUF")]

    chosen = chosen_hits(_models(hits), [QWEN], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["bartowski/Qwen3.5-9B-GGUF"]


def test_a_model_only_open_accounts_hold_fetches_the_one_with_the_most_downloads():
    hits = [
        _hit("community-user/Qwen3.5-9B-GGUF", owner="other", downloads=10),
        _hit("someone/Qwen3.5-9B-GGUF", owner="other", downloads=2_000),
    ]

    chosen = chosen_hits(_models(hits, filtered=False), [QWEN], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["someone/Qwen3.5-9B-GGUF"]


def test_an_open_account_is_left_out_where_a_listed_packager_has_the_model_too():
    hits = [
        _hit("community-user/Qwen3.5-9B-GGUF", owner="other", downloads=9_000_000),
        _hit("unsloth/Qwen3.5-9B-GGUF", downloads=10),
    ]

    chosen = chosen_hits(_models(hits, filtered=False), [QWEN], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["unsloth/Qwen3.5-9B-GGUF"]


def test_a_repository_id_picks_exactly_that_repository():
    """The older shape of an answer file: `select` named repositories, and still may."""
    hits = [_hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher"), _hit("unsloth/Qwen3.5-9B-GGUF")]

    chosen = chosen_hits(_models(hits), ["unsloth/Qwen3.5-9B-GGUF"], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["unsloth/Qwen3.5-9B-GGUF"]


def test_a_model_and_a_repository_of_another_model_can_be_answered_together():
    hits = [
        _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher"),
        _hit("unsloth/Qwen3.5-9B-GGUF"),
        _hit("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", base=DEEPSEEK),
    ]

    chosen = chosen_hits(_models(hits), [QWEN, "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == [
        "Qwen/Qwen3.5-9B-GGUF",
        "unsloth/Qwen3.5-9B-GGUF",
        "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
    ]


def test_the_same_repository_is_never_fetched_twice():
    hits = [_hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher"), _hit("unsloth/Qwen3.5-9B-GGUF")]

    chosen = chosen_hits(_models(hits), [QWEN, "unsloth/Qwen3.5-9B-GGUF"], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["Qwen/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-9B-GGUF"]


def test_an_id_that_is_a_model_and_a_repository_at_once_is_read_as_the_model():
    """The list offers base models, so a value that is both means the model (second-model round).

    Reachable: `acme/A` may package `acme/B` and be a base model of `packager/A-GGUF` at the same
    time -- nothing in the relation check forbids it. The precedence is a decision, so it is
    written down here and in CONTRACTS.md rather than left to whichever branch runs first.
    """
    hits = [
        _hit("acme/A", base="acme/B", owner="publisher"),
        _hit("packager/A-GGUF", base="acme/A"),
    ]
    models = _models(hits)

    assert sorted(model.base_model for model in models) == ["acme/A", "acme/B"]
    chosen = chosen_hits(models, ["acme/A"], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["packager/A-GGUF"]


def test_an_answer_that_is_neither_a_model_nor_a_repository_is_left_out():
    """The asker refuses such an answer first; this is the second line of defense, not a guess."""
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF")]

    assert chosen_hits(_models(hits), ["acme/Nothing"], DEFAULT_PACKAGERS) == []


def test_of_an_accounts_builds_the_plain_gguf_repository_is_fetched():
    """`unsloth` answered `-MTP-GGUF`, `-GGUF` and `-UD-GGUF` for one model, newest first, and the
    newest was the special build (measured live 2026-09-24). The plain build is the one a person
    who picked the model expects (decided 2026-09-24)."""
    hits = [
        _hit("unsloth/Qwen3.5-9B-MTP-GGUF"),
        _hit("unsloth/Qwen3.5-9B-GGUF"),
        _hit("unsloth/Qwen3.5-9B-UD-GGUF"),
    ]

    chosen = chosen_hits(_models(hits), [QWEN], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["unsloth/Qwen3.5-9B-GGUF"]


def test_without_a_plain_gguf_build_the_first_of_the_account_is_fetched():
    hits = [_hit("unsloth/Qwen3.5-9B-MTP-GGUF"), _hit("unsloth/Qwen3.5-9B-UD-GGUF")]

    chosen = chosen_hits(_models(hits), [QWEN], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["unsloth/Qwen3.5-9B-MTP-GGUF"]


def test_the_publishers_plain_gguf_build_is_preferred_as_well():
    hits = [_hit("Qwen/Qwen3.5-9B-MTP-GGUF", owner="publisher"), _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher")]

    chosen = chosen_hits(_models(hits), [QWEN], DEFAULT_PACKAGERS)

    assert [hit.repo for hit in chosen] == ["Qwen/Qwen3.5-9B-GGUF"]


# --- the size column ------------------------------------------------------------------------------


def test_a_trillion_count_is_shown_as_t_and_not_as_thousands_of_b():
    """`Qwen3.8-2.4T-A95B` is 2400 in the count and `2.4T` in the column (decided 2026-09-24)."""
    model = _models([_hit("unsloth/Qwen3.8-2.4T-A95B-GGUF", base="Qwen/Qwen3.8-2.4T-A95B")])[0]

    assert (model.parameters_b, model.size_text) == (2400.0, "2.4T")


def test_a_round_trillion_count_is_shown_without_a_decimal():
    assert _models([_hit("unsloth/Nova-1T-GGUF", base="acme/Nova-1T")])[0].size_text == "1T"


def test_a_count_below_a_trillion_stays_in_b():
    assert _models([_hit("unsloth/Nova-999B-GGUF", base="acme/Nova-999B")])[0].size_text == "999B"
