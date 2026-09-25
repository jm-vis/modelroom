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
    COLUMN_NAMES,
    DEFAULT_CONTEXT,
    LABEL_LIMIT,
    LINE_LIMIT,
    MACHINE_NOT_MEASURED,
    PARAMETER_COUNT_UNKNOWN,
    POINTER_WIDTH,
    RELEASE_COLUMN,
    ModelChoice,
    chosen_hits,
    hint_line,
    list_choices,
    list_context,
    list_header,
    model_choices,
    parameters_from_name,
    picked_names,
    unusable_counts,
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
        # Millions count as well, and an effective size keeps its `E` (decided 2026-09-25): one
        # size rule for the list's fit and for the successor of a computed release.
        ("SmolLM2-360M-Instruct", 0.36),
        ("gemma-4-E4B-it", 4.0),
        ("Nova-e2b", 2.0),
        # The `E` is a prefix of a size of its own, never the end of a word in front of it.
        ("Nova-Large4B", None),
        ("Qwen3-30B-A3B", 30.0),
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


def test_the_list_shows_name_fit_size_release_packagers_downloads_and_ollama():
    hits = [
        _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher", downloads=12_031_627, ollama="qwen3.5:9b", age="legacy", successor="Qwen/Qwen3.8-9B"),
        _hit("unsloth/Qwen3.5-9B-GGUF", downloads=1_626_475, ollama="qwen3.5:9b"),
    ]

    label = _labels(_models(hits))[0]

    assert label.startswith("Qwen3.5-9B")
    assert "good" in label
    assert "9B" in label
    assert "legacy" in label
    # `Qwen, unsloth` is 13 characters and the column 12: the first account and a count.
    assert "Qwen +1" in label
    assert "13.6M" in label
    assert label.rstrip().endswith("qwen3.5:9b")


def test_the_head_of_the_list_names_every_column_and_cannot_be_picked():
    """The list had no column head at all, so nobody could tell what a cell meant (2026-09-25)."""
    head = list_header()

    assert head.heading is True
    assert head.label.split() == list(COLUMN_NAMES)
    assert RELEASE_COLUMN in COLUMN_NAMES


def test_the_head_is_indented_so_its_columns_stand_over_the_cells():
    """A heading is drawn without the marker of a row, so its own line carries that width."""
    head = list_header()
    row = _labels(_models([_hit("unsloth/Qwen3.5-9B-GGUF")]))[0]

    assert head.label.index("Model") == 2
    assert head.label.index("Fit") - 2 == row.index("good")


def test_a_model_without_an_ollama_name_says_a_dash():
    """`none known` in a column of its own irritated the test round of 2026-09-25."""
    label = _labels(_models([_hit("unsloth/Qwen3.5-9B-GGUF")]))[0]

    assert label.rstrip().endswith("–")
    assert "none known" not in label


def test_an_unknown_release_is_a_dash_and_never_the_word():
    model = _models([_hit("unsloth/Qwen3.5-9B-GGUF")])[0]

    assert model.release_text == "–"
    assert _labels([model])[0].count("unknown") == 0


def test_the_release_column_carries_the_word_of_the_age():
    latest = _models([_hit("unsloth/Qwen3.5-9B-GGUF", age="latest")])[0]
    legacy = _models([_hit("unsloth/Qwen3.5-9B-GGUF", age="legacy", successor="Qwen/Qwen3.8-9B")])[0]

    assert (latest.release_text, legacy.release_text) == ("latest", "legacy")


@pytest.mark.parametrize("basis", ["stated", None])
def test_a_stated_release_carries_no_star(basis):
    """`None` is a hit built before the basis existed, and it reads as stated."""
    model = _models([_hit("unsloth/Qwen3.5-9B-GGUF", age="latest", release_basis=basis)])[0]

    assert model.release_text == "latest"


def test_a_computed_release_carries_a_star_in_the_width_of_its_column():
    latest = _models([_hit("unsloth/Nova-9B-GGUF", base="acme/Nova-9B", age="latest", release_basis="computed")])[0]
    legacy = _models(
        [_hit("unsloth/Qwen3.5-9B-GGUF", age="legacy", successor="Qwen/Qwen3.8-27B", release_basis="computed")]
    )[0]

    assert (latest.release_text, legacy.release_text) == ("latest*", "legacy*")
    assert (latest.release_basis, legacy.release_basis) == ("computed", "computed")
    assert all(len(text) <= 7 for text in (latest.release_text, legacy.release_text))
    assert all(len(label) <= LABEL_LIMIT for label in _labels([latest, legacy]))
    assert "latest* " in _labels([latest])[0]


def test_the_age_and_its_basis_come_from_the_same_repository():
    """The first repository with a known age decides both; a second one never lends its basis."""
    hits = [
        _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher"),
        _hit("unsloth/Qwen3.5-9B-GGUF", age="legacy", successor="Qwen/Qwen3.8-27B", release_basis="computed"),
        _hit("bartowski/Qwen3.5-9B-GGUF", age="latest", release_basis="stated"),
    ]

    model = _models(hits)[0]

    assert (model.age, model.release_basis, model.release_text) == ("legacy", "computed", "legacy*")


def test_a_computed_basis_without_a_known_age_is_no_model_choice():
    model = _models([_hit("unsloth/Qwen3.5-9B-GGUF")])[0]

    with pytest.raises(ValueError, match="computed"):
        ModelChoice.model_validate({**model.model_dump(), "release_basis": "computed"})


def test_a_computed_legacy_row_is_drawn_gray_like_a_stated_one():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF", age="legacy", successor="Qwen/Qwen3.8-27B", release_basis="computed")]

    assert [choice.dim for choice in list_choices(_models(hits))] == [True]


def test_a_legacy_row_is_drawn_gray_as_a_whole():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF", age="legacy", successor="Qwen/Qwen3.8-9B"), _hit("unsloth/Nova-9B-GGUF", base="acme/Nova-9B", age="latest")]

    dimmed = {choice.label.split()[0]: choice.dim for choice in list_choices(_models(hits))}

    assert dimmed["Qwen3.5-9B"] is True
    assert dimmed["Nova-9B"] is False


def test_an_account_with_two_builds_of_one_model_stands_once():
    """`unsloth, unsloth +4` was the live list of 2026-09-24; the column names accounts."""
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF"), _hit("unsloth/Qwen3.5-9B-UD-GGUF"), _hit("bartowski/Qwen3.5-9B-GGUF")]

    model = _models(hits)[0]

    assert model.packagers == ["unsloth", "bartowski"]
    assert len(model.repos) == 3


def test_packagers_that_do_not_fit_their_column_are_the_first_account_and_a_count():
    """Twelve characters since 2026-09-25: the first account and how many more (`unsloth +4`)."""
    hits = [_hit(f"{owner}/Qwen3.5-9B-GGUF") for owner in ("unsloth", "bartowski", "mradermacher", "ggml-org", "lmstudio-community")]

    label = _labels(_models(hits))[0]

    assert "unsloth +4" in label
    assert "bartowski" not in label


def test_packagers_that_fit_their_column_stand_whole_whatever_their_number():
    """Up to 2026-09-25 a fourth account always became a count; the rule is the width now."""
    hits = [_hit(f"{owner}/Qwen3.5-9B-GGUF") for owner in ("a", "b", "c", "d")]

    assert _models(hits)[0].packagers_text == "a, b, c, d"


@pytest.mark.parametrize(
    "owners, shown",
    [
        (("unsloth",), "unsloth"),
        (("unsloth", "Qwen"), "unsloth +1"),
        (("unsloth", "a", "b", "c"), "unsloth +3"),
        (("Qwen", "unsloth"), "Qwen +1"),
        (("mistralai", "unsloth"), "mistralai +1"),
        (("lmstudio-community", "unsloth"), "lmstudio-co…"),
    ],
)
def test_the_packagers_rule_is_whole_else_first_and_count_else_cut(owners, shown):
    hits = [_hit(f"{owner}/Qwen3.5-9B-GGUF") for owner in owners]

    text = _models(hits)[0].packagers_text

    assert text == shown
    assert len(text) <= 12


def test_the_packagers_column_is_twelve_and_the_ollama_column_fourteen_characters():
    """Measured at the labels, not at constants: a line with every cell at its widest."""
    hits = [_hit(f"{owner}/Nova-9B-GGUF", base="acme/Nova-9B", ollama="abcdefghijklmnopqrst:9b") for owner in ("unsloth", "a", "b", "c")]

    label = _labels(_models(hits))[0]
    head = list_header().label

    packagers_at = head.index("Packagers") - 2
    downloads_at = head.index("Downl.") - 2
    ollama_at = head.index("Ollama") - 2
    assert downloads_at - packagers_at == 12 + 2
    assert label[packagers_at:downloads_at].rstrip() == "unsloth +3"
    assert len(label) - ollama_at == 14
    assert label[ollama_at:] == "abcdefghijklm…"
    assert len(label) <= LABEL_LIMIT


def test_no_line_of_the_list_is_wider_than_a_hundred_characters():
    """The whole line, pointer and marker included: `❯ ○ ` costs four characters of the hundred,
    and the one space every line of the screen carries costs the fifth (decided 2026-09-25)."""
    hits = [
        _hit("Qwen/Qwen3.5-9B-GGUF", owner="publisher", downloads=12_031_627, ollama="qwen3.5:9b"),
        _hit("unsloth/Qwen3.5-9B-GGUF", downloads=1_626_475, ollama="qwen3.5:9b"),
        _hit("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", base=DEEPSEEK, downloads=68_445),
    ]

    assert LABEL_LIMIT == LINE_LIMIT - POINTER_WIDTH
    for label in [list_header().label, *_labels(_models(hits))]:
        assert len(label) <= LABEL_LIMIT, label


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
        assert len(label) <= LABEL_LIMIT, f"{len(label)}: {label}"
    omni = next(label for label in labels if label.startswith("Qwen3-Omni"))
    nova = next(label for label in labels if label.startswith("Nova-9B"))
    assert "…" in omni, "a cell that does not fit its column says so"
    # Fourteen characters since 2026-09-25: `deepseek-r1:8b` stands whole, a longer registry name is
    # still cut here; the install line of step 5 carries the whole one.
    assert nova.rstrip().endswith("deepseek-r1:8b")
    assert omni.rstrip().endswith("…")


def test_an_ollama_name_of_fourteen_characters_is_shown_whole():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF", ollama="granite4.2:30b")]

    assert _labels(_models(hits))[0].rstrip().endswith("granite4.2:30b")


def test_the_catalog_names_that_stand_whole_and_the_five_that_are_cut():
    """27 of the 32 Ollama names of the shipped catalog fit fourteen characters (2026-09-25)."""
    from modelroom.catalog import load_catalog

    names = {
        f"{model.ollama_base}:{model.ollama_tag}"
        for family in load_catalog().families
        for model in family.models
        if model.ollama_base is not None
    }
    cut = sorted(name for name in names if len(name) > 14)

    assert len(names) == 32
    assert cut == sorted(
        ["ministral-3:14b", "qwen3-coder:30b", "qwen3-embedding:8b", "mistral-small3.2:24b", "qwen3-embedding:0.6b"]
    )
    for name in names:
        label = _labels(_models([_hit("unsloth/Qwen3.5-9B-GGUF", ollama=name)]))[0]
        assert label.rstrip().endswith(name if name not in cut else "…"), label


def test_an_ollama_pair_of_the_configuration_is_shown_where_no_hit_names_one():
    """A pair the catalog does not know was a dash in the list, though the fetch uses it."""
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF")]

    models = model_choices(
        hits, filtered=True, checked=_measured(), context=8192, configured_ollama={QWEN: "my-qwen:9b"}
    )

    assert models[0].ollama == "my-qwen:9b"
    assert _labels(models)[0].rstrip().endswith("my-qwen:9b")


def test_a_hit_that_names_an_ollama_name_still_wins_in_the_list():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF", ollama="qwen3.5:9b")]

    models = model_choices(
        hits, filtered=True, checked=_measured(), context=8192, configured_ollama={QWEN: "my-qwen:9b"}
    )

    assert models[0].ollama == "qwen3.5:9b"


def test_a_size_and_a_download_count_of_any_length_keep_their_columns():
    """Both come from a registry: a count of 1,234,567 B parameters and a huge download number."""
    hits = [
        _hit("unsloth/Nova-1234567B-GGUF", base="acme/Nova-1234567B", downloads=999_999_999_999),
    ]

    label = _labels(_models(hits))[0]

    assert len(label) <= LABEL_LIMIT, f"{len(label)}: {label}"


def test_the_order_is_the_fit_class_then_the_downloads_then_the_name():
    hits = [
        _hit("unsloth/Nova-400B-GGUF", base="acme/Nova-400B", downloads=9_000_000),  # too tight
        _hit("unsloth/Nova-9B-GGUF", base="acme/Nova-9B", downloads=1_000),
        _hit("unsloth/Nova-8B-GGUF", base="acme/Nova-8B", downloads=2_000),
        _hit("unsloth/Nova-One-GGUF", base="acme/Nova-One"),  # no parameter count: unknown
    ]

    models = _models(hits, checked=_measured(ram_gib=31.7))

    assert [model.name for model in models] == ["Nova-8B", "Nova-9B", "Nova-400B", "Nova-One"]


def test_the_release_stands_between_the_fit_class_and_the_downloads():
    """`latest` first, then a release nobody knows, then `legacy` (decided 2026-09-25)."""
    hits = [
        _hit("unsloth/Nova-8B-GGUF", base="acme/Nova-8B", downloads=9_000_000, age="legacy", successor="acme/Nova-8B-2512"),
        _hit("unsloth/Nova-9B-GGUF", base="acme/Nova-9B", downloads=1_000),
        _hit("unsloth/Nova-7B-GGUF", base="acme/Nova-7B", downloads=10, age="latest"),
    ]

    models = _models(hits)

    assert [model.name for model in models] == ["Nova-7B", "Nova-9B", "Nova-8B"]


def test_a_computed_release_sorts_like_a_stated_one():
    """Decided 2026-09-25: the star says where the status comes from, not how much it weighs."""
    hits = [
        _hit("unsloth/Nova-8B-GGUF", base="acme/Nova-8B", downloads=9_000_000, age="legacy",
             successor="acme/Nova-8B-2512", release_basis="computed"),
        _hit("unsloth/Nova-9B-GGUF", base="acme/Nova-9B", downloads=1_000),
        _hit("unsloth/Nova-7B-GGUF", base="acme/Nova-7B", downloads=10, age="latest", release_basis="computed"),
        _hit("unsloth/Luna-7B-GGUF", base="acme/Luna-7B", downloads=5, age="latest", release_basis="stated"),
    ]

    models = _models(hits)

    assert [model.name for model in models] == ["Nova-7B", "Luna-7B", "Nova-9B", "Nova-8B"]


def test_the_values_of_the_list_are_the_base_models():
    assert [choice.value for choice in list_choices(_models([_hit("unsloth/Qwen3.5-9B-GGUF")]))] == [QWEN]


def test_nothing_of_the_list_is_grayed_out():
    assert [choice.disabled for choice in list_choices(_models([_hit("unsloth/Qwen3.5-9B-GGUF")]))] == [None]


# --- the lines around the list ---------------------------------------------------------------------


def test_the_reasons_of_the_repositories_that_cannot_be_used_are_counted_once_each():
    """The counts `search.json` holds and the lines below the list are one reading of the hits."""
    hits = [
        _hit("unsloth/Qwen3.5-9B-GGUF"),
        _unresolved("someone/Qwen3.5-9B-GGUF"),
        _unresolved("another/Qwen3.5-9B-GGUF", "derivative"),
    ]

    counts = unusable_counts(hits, True)

    assert sum(counts.values()) == 2
    assert sorted(counts) == [
        "a fine-tune or a merge, not a quantization of one base model",
        "the repository does not say it packages a base model",
    ]
    assert [line.split(" ", 1)[0] for line in unusable_reasons(hits, True)] == ["1", "1"]


def test_the_context_of_the_list_is_the_one_this_folder_kept():
    """Step 3 asks for the context; step 2 stands on what the folder holds, else the scale's level."""
    assert list_context(None) == 32768
    assert list_context(8192) == 8192


def test_the_default_context_of_the_list_is_the_one_the_scale_starts_on():
    """The list said `fit at 8k` while the next question started on `L 32k` (test round 2026-09-24)."""
    from modelroom.guided import DEFAULT_CONTEXT as GUIDED_DEFAULT

    assert DEFAULT_CONTEXT == 32768
    assert DEFAULT_CONTEXT is GUIDED_DEFAULT


def test_the_hint_line_explains_the_release_words_and_names_the_context():
    """`nothing marked keeps the folder as it is` went with the Enter rule (decided 2026-09-25)."""
    line = hint_line(_measured(), 32768)

    assert line == (
        "latest: the publisher's current release of its family · legacy: a successor is named · "
        "*: computed from the version numbers of the family, not stated by the publisher · "
        "fit from the size at 32k context, exact after the fetch"
    )
    assert "nothing marked" not in line


def test_a_context_of_its_own_is_not_floored_into_a_k_it_is_not():
    assert "at 5000 tokens context" in hint_line(_measured(), 5000)


def test_the_hint_line_of_an_unmeasured_machine_says_the_fit_is_unknown():
    line = hint_line(_unmeasured(), 32768)

    assert line.startswith("latest: the publisher's current release of its family")
    assert line.endswith("fit unknown until this machine is measured")


def test_the_answer_of_the_list_is_read_back_in_the_names_the_list_showed():
    hits = [_hit("unsloth/Qwen3.5-9B-GGUF")]
    models = _models(hits)

    assert picked_names(models, ["Qwen/Qwen3.5-9B"]) == ["Qwen3.5-9B"]
    assert picked_names(models, ["unsloth/Qwen3.5-9B-GGUF"]) == ["unsloth/Qwen3.5-9B-GGUF"]


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
