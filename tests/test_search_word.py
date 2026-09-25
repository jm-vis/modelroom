"""Tests for modelroom.search_word: the three forms of input, and what the catalog says about them.

Everything here is pure -- no transport, no fixture, no clock. The cases are the ones the search
of 2026-09-24 could not answer: a repository id, a name with blanks in it, a name with a typo, and
nothing at all (decided 2026-09-25).
"""

from __future__ import annotations

import pytest

from modelroom.catalog import Catalog, load_catalog
from modelroom.search_word import (
    CLOSE_MATCH_CUTOFF,
    FILLER_WORDS,
    candidate_pool,
    catalog_models,
    close_matches,
    family_matches,
    latest_model_names,
    parse_search,
    without_separators,
)

QWEN_BASE = "Qwen/Qwen3.5-9B"
UNSLOTH_GGUF = "unsloth/Qwen3.5-9B-GGUF"


def _catalog() -> Catalog:
    return load_catalog()


def _small_catalog() -> Catalog:
    return Catalog.model_validate(
        {
            "schema_version": 1,
            "families": [
                {
                    "name": "nova",
                    "publisher": "acme",
                    "models": [
                        {
                            "hf_repo": "acme/Nova-7B",
                            "latest": True,
                            "latest_source": "https://huggingface.co/acme",
                            "latest_checked": "2026-09-25",
                            "source": "https://huggingface.co/acme/Nova-7B",
                        },
                        {"hf_repo": "acme/Nova-3B", "source": "https://huggingface.co/acme/Nova-3B"},
                    ],
                },
                {
                    "name": "flux",
                    "publisher": "other-lab",
                    "models": [
                        {
                            "hf_repo": "other-lab/Flux-1B",
                            "latest": True,
                            "latest_source": "https://huggingface.co/other-lab",
                            "latest_checked": "2026-09-25",
                            "source": "https://huggingface.co/other-lab/Flux-1B",
                        },
                        # The same model **name** under another account: two current models whose
                        # name is one word, which is what `latest_model_names` may list only once.
                        {
                            "hf_repo": "other-lab/Nova-7B",
                            "latest": True,
                            "latest_source": "https://huggingface.co/other-lab",
                            "latest_checked": "2026-09-25",
                            "source": "https://huggingface.co/other-lab/Nova-7B",
                        },
                    ],
                },
            ],
        }
    )


# --- the three forms ----------------------------------------------------------------------------


def test_one_word_is_the_words_form_with_that_one_word():
    word = parse_search("qwen")

    assert (word.form, word.words, word.repo_id) == ("words", ("qwen",), None)
    assert word.text == "qwen"


def test_blanks_around_the_input_are_not_part_of_it():
    assert parse_search("  qwen  ").text == "qwen"


@pytest.mark.parametrize(
    "text, words",
    [
        ("qwen 3.5 9b", ("qwen", "3.5", "9b")),
        ("Qwen3.5-9B", ("qwen3.5", "9b")),
        ("qwen3.5_9b", ("qwen3.5", "9b")),
        ("qwen3.5:9b", ("qwen3.5", "9b")),
        ("QWEN   3.5", ("qwen", "3.5")),
    ],
)
def test_the_input_is_cut_at_blanks_and_at_the_three_separators(text, words):
    assert parse_search(text).words == words


def test_the_filler_words_are_not_part_of_the_input():
    assert parse_search("the qwen 3.5 9b gguf model").words == ("qwen", "3.5", "9b")
    assert FILLER_WORDS == frozenset({"gguf", "model", "models", "the"})


def test_nothing_typed_is_the_empty_form():
    for text in ("", "   "):
        word = parse_search(text)
        assert (word.form, word.words) == ("empty", ())


def test_filler_alone_is_the_empty_form_too():
    # `gguf` says nothing about which model, so there is nothing to search for.
    assert parse_search("gguf").form == "empty"


def test_an_owner_slash_name_is_the_id_form_and_keeps_its_spelling():
    word = parse_search("unsloth/Qwen3.5-9B-GGUF")

    assert word.form == "id"
    assert word.repo_id == UNSLOTH_GGUF  # a repository id is case sensitive
    # The words of the **name** half: what the search falls back to without a GGUF file there.
    assert word.words == ("qwen3.5", "9b")


@pytest.mark.parametrize(
    "text", ["a/b/c", "/name", "owner/", "-owner/name", "owner/-name", "owner name", "qwen"]
)
def test_what_is_no_repository_id_is_read_as_words(text):
    assert parse_search(text).form != "id"


# --- the one word that goes to the Hub, and the filter that follows ------------------------------


def test_one_word_goes_to_the_hub_as_it_is():
    assert parse_search("qwen").hub_word == "qwen"
    assert parse_search("mistral").hub_word == "mistral"


def test_more_than_one_word_sends_the_longest_letter_run():
    assert parse_search("qwen 3.5 9b").hub_word == "qwen"
    assert parse_search("Qwen3.5-9B").hub_word == "qwen"
    assert parse_search("deepseek r1").hub_word == "deepseek"


def test_without_a_letter_run_the_longest_word_is_sent():
    assert parse_search("3.5 9b").hub_word == "3.5"


def test_one_word_is_never_held_against_the_answer():
    # Today's behavior, and the reason the recorded answers of this suite stay valid: a
    # `mistralai/Ministral-*` build carries `mistral` only in its owner half.
    assert parse_search("qwen").filters_locally is False
    assert parse_search("qwen 3.5 9b").filters_locally is True


def test_more_than_one_word_matches_an_id_with_its_separators_removed():
    word = parse_search("qwen 3.5 9b")

    assert word.matches(UNSLOTH_GGUF) is True
    assert word.matches(QWEN_BASE) is True
    assert word.matches("unsloth/Qwen3.5-4B-GGUF") is False


def test_one_word_matches_an_id_as_it_stands():
    assert parse_search("r1").matches("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B") is True
    assert parse_search("qwen").matches("unsloth/Qwen3.5-9B-GGUF") is True


def test_the_empty_form_matches_nothing():
    assert parse_search("").matches(UNSLOTH_GGUF) is False


def test_the_separators_are_removed_and_the_decimal_point_is_kept():
    assert without_separators("Qwen/Qwen3.5-9B") == "qwenqwen3.59b"


# --- what the catalog says about the input -------------------------------------------------------


def test_a_name_with_blanks_names_the_catalog_model_it_spells():
    found = [model.hf_repo for model in catalog_models(_catalog(), parse_search("qwen 3.5 9b"))]

    assert found == [QWEN_BASE]


def test_a_family_matches_over_its_name_its_account_or_one_of_its_models():
    catalog = _small_catalog()
    nova, flux = catalog.families

    assert family_matches(nova, parse_search("nova")) is True
    assert family_matches(flux, parse_search("other lab")) is True
    assert family_matches(nova, parse_search("nova 7b")) is True
    assert family_matches(nova, parse_search("flux")) is False
    # `flux` does hold an `other-lab/Nova-7B`, so `nova` matches it as well -- a word is held
    # against every model id, not only against the family's own name.
    assert family_matches(flux, parse_search("nova")) is True
    assert family_matches(flux, parse_search("gamma")) is False


def test_the_empty_form_matches_no_family():
    assert family_matches(_small_catalog().families[0], parse_search("")) is False


def test_the_catalog_names_its_current_models_once_each():
    # `acme/Nova-7B` and `other-lab/Nova-7B` are two current models of one name: one page, not two.
    assert latest_model_names(_small_catalog()) == ["Nova-7B", "Flux-1B"]


def test_the_shipped_catalog_names_more_than_one_current_model():
    names = latest_model_names(_catalog())

    assert len(names) == len(set(names))
    assert len(names) > 1
    assert all("/" not in name for name in names)


# --- did you mean --------------------------------------------------------------------------------


def test_a_typo_in_a_family_word_leads_to_that_word():
    assert "qwen" in close_matches(_catalog(), parse_search("qwn"))


def test_the_pool_holds_the_letter_runs_of_the_names_and_nothing_shorter_than_three():
    pool = candidate_pool(_catalog())

    assert "qwen" in pool
    assert "deepseek" in pool
    assert "r" not in pool
    assert pool == [word.casefold() for word in pool]


def test_a_word_far_from_everything_leads_nowhere():
    assert close_matches(_catalog(), parse_search("zzzzzzzz")) == []


def test_the_threshold_is_the_one_the_decision_named():
    assert CLOSE_MATCH_CUTOFF == 0.75


def test_a_candidate_is_offered_once_however_many_words_reach_it():
    found = close_matches(_catalog(), parse_search("qwn qwn"))

    assert found.count("qwen") == 1
