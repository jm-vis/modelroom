"""Tests for modelroom.search_pages: the three pinned request forms and what they are built from.

The URLs are asserted as literals, not interpolated from the module's own constants: they are the
requests every fixture in this suite was recorded against, and a literal is the only assertion
that notices when one of them changes.
"""

from __future__ import annotations

import pytest

from modelroom.catalog import Catalog, load_catalog
from modelroom.guided_contracts import SearchHit
from modelroom.search_pages import (
    HF_ACCOUNT_LIMIT,
    HF_OPEN_LIMIT,
    MOST_DOWNLOADED_LABEL,
    NEWEST_LABEL,
    SearchGroup,
    account_search_url,
    format_downloads,
    most_downloaded_url,
    newest_url,
    publisher_accounts,
    read_downloads,
    search_accounts,
    search_requests,
)

_EXPAND = "&expand=cardData&expand=createdAt&expand=downloads&expand=safetensors&expand=tags"


def _catalog() -> Catalog:
    return load_catalog()


# --- the three pinned request forms ------------------------------------------------------------


def test_the_account_request_is_the_pinned_literal():
    assert account_search_url("qwen", "unsloth") == (
        "https://huggingface.co/api/models?author=unsloth&search=qwen&filter=gguf"
        "&sort=createdAt&direction=-1"
        "&limit=20" + _EXPAND
    )


def test_the_most_downloaded_request_is_the_pinned_literal():
    assert most_downloaded_url("qwen") == (
        "https://huggingface.co/api/models?search=qwen&filter=gguf&sort=downloads&direction=-1"
        "&limit=10" + _EXPAND
    )


def test_the_newest_request_is_the_pinned_literal():
    assert newest_url("qwen") == (
        "https://huggingface.co/api/models?search=qwen&filter=gguf&sort=createdAt&direction=-1"
        "&limit=10" + _EXPAND
    )


def test_the_two_page_limits_are_pinned():
    assert (HF_ACCOUNT_LIMIT, HF_OPEN_LIMIT) == (20, 10)


def test_every_request_form_quotes_the_word_and_the_account():
    assert "search=qwen+3.5&filter=gguf" in account_search_url("qwen 3.5", "unsloth")
    assert "author=a%26b&search=qwen" in account_search_url("qwen", "a&b")
    assert "search=a%26b&filter=gguf" in most_downloaded_url("a&b")
    assert "search=a%26b&filter=gguf" in newest_url("a&b")


def test_expand_is_repeated_once_per_field_and_names_downloads():
    # `downloads` is only answered when it is asked for, and only the moment `expand` is set at
    # all -- measured against the live API 2026-09-24.
    for url in (account_search_url("qwen", "unsloth"), most_downloaded_url("qwen"), newest_url("qwen")):
        assert url.count("&expand=") == 5
        assert "&expand=downloads" in url


# --- which accounts are asked -------------------------------------------------------------------


def test_a_word_in_a_family_name_asks_that_familys_publisher():
    assert publisher_accounts(_catalog(), "qwen")[0] == "Qwen"


def test_a_word_in_a_publisher_account_asks_that_publisher():
    assert publisher_accounts(_catalog(), "deepseek") == ["deepseek-ai"]


def test_a_word_only_a_model_id_carries_asks_that_familys_publisher():
    # `r1` is in no family name and in no publisher account; only
    # `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` carries it.
    assert publisher_accounts(_catalog(), "r1") == ["deepseek-ai"]


def test_the_comparison_ignores_case_and_surrounding_blanks():
    assert publisher_accounts(_catalog(), "  QWEN  ")[0] == "Qwen"


def test_a_word_no_family_matches_asks_no_publisher():
    assert publisher_accounts(_catalog(), "foo") == []


def test_an_empty_word_asks_no_publisher():
    assert publisher_accounts(_catalog(), "   ") == []


def test_every_publisher_is_named_once_however_many_of_its_models_match():
    catalog = Catalog.model_validate(
        {
            "schema_version": 1,
            "families": [
                {
                    "name": "nova",
                    "publisher": "acme",
                    "models": [
                        {"hf_repo": "acme/Nova-7B", "source": "https://huggingface.co/acme/Nova-7B"},
                        {"hf_repo": "acme/Nova-9B", "source": "https://huggingface.co/acme/Nova-9B"},
                    ],
                },
                {
                    "name": "nova-mini",
                    "publisher": "acme",
                    "models": [{"hf_repo": "acme/NovaMini-1B", "source": "https://huggingface.co/acme/NovaMini-1B"}],
                },
            ],
        }
    )

    assert publisher_accounts(catalog, "nova") == ["acme"]


def test_the_matching_publishers_come_before_the_positive_list():
    accounts = search_accounts(_catalog(), "qwen", ("unsloth", "bartowski"))

    assert accounts[0] == "Qwen"
    assert accounts[-2:] == ["unsloth", "bartowski"]


def test_an_account_on_both_lists_is_asked_once():
    accounts = search_accounts(_catalog(), "qwen", ("unsloth", "Qwen", "bartowski"))

    assert accounts.count("Qwen") == 1
    assert accounts == ["Qwen", "deepseek-ai", "unsloth", "bartowski"]


def test_a_word_without_a_family_asks_the_packagers_alone():
    assert search_accounts(_catalog(), "foo", ("unsloth", "bartowski")) == ["unsloth", "bartowski"]


# --- the pages of one search --------------------------------------------------------------------


def test_with_the_filter_on_there_is_one_request_per_account_and_no_open_page():
    requests = search_requests("qwen", ["Qwen", "unsloth"], open_pages=False)

    assert [request.label for request in requests] == ["Qwen", "unsloth"]
    assert [request.account for request in requests] == ["Qwen", "unsloth"]
    assert [request.limit for request in requests] == [20, 20]
    assert requests[0].url == account_search_url("qwen", "Qwen")


def test_with_the_filter_off_the_two_open_pages_come_after_the_accounts():
    requests = search_requests("qwen", ["Qwen", "unsloth"], open_pages=True)

    assert [request.label for request in requests] == ["Qwen", "unsloth", MOST_DOWNLOADED_LABEL, NEWEST_LABEL]
    assert [request.account for request in requests][-2:] == [None, None]
    assert [request.limit for request in requests][-2:] == [10, 10]
    assert requests[-2].url == most_downloaded_url("qwen")
    assert requests[-1].url == newest_url("qwen")


def test_the_labels_of_the_open_pages_are_the_words_of_the_language_standard():
    assert (MOST_DOWNLOADED_LABEL, NEWEST_LABEL) == ("most downloaded", "newest")


# --- the line a group prints --------------------------------------------------------------------


def _group(**overrides) -> SearchGroup:
    fields = {"label": "unsloth", "hits": [], "page_size": 2, "page_full": False, "already_listed": 0}
    return SearchGroup(**{**fields, **overrides})


def test_a_group_line_names_the_label_and_the_number():
    assert _group().line() == "unsloth 2"


def test_a_full_page_says_that_a_more_specific_word_shortens_the_list():
    assert _group(page_size=20, page_full=True).line() == (
        "unsloth 20 (page full: a more specific word shortens the list)"
    )


def test_an_open_page_counts_the_repositories_an_account_page_already_listed():
    assert _group(label=MOST_DOWNLOADED_LABEL, page_size=10, already_listed=3).line() == (
        "most downloaded 10, 3 of them already listed"
    )


def test_an_account_that_answered_with_nothing_still_gets_its_line():
    assert _group(label="bartowski", page_size=0).line() == "bartowski 0"


# --- downloads ----------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False, 1.5, "12", None, -1, [3]])
def test_a_downloads_field_that_is_no_count_reads_as_unknown(value):
    assert read_downloads(value) is None


@pytest.mark.parametrize("value", [0, 1, 3551, 68445, 12_031_627])
def test_a_whole_count_at_or_above_zero_is_read_as_it_is(value):
    assert read_downloads(value) == value


@pytest.mark.parametrize(
    "value, shown",
    [
        (None, "unknown"),
        (0, "0"),
        (999, "999"),
        (3551, "3551"),
        (9999, "9999"),
        (10_000, "10k"),
        (68_445, "68k"),
        (999_499, "999k"),
        # the thousands are floored: rounding would make this `1000k`, a shape this column has not
        (999_999, "999k"),
        (1_000_000, "1.0M"),
        (1_950_000, "1.9M"),
        (12_031_627, "12.0M"),
    ],
)
def test_the_downloads_column_shows_a_number_a_k_or_an_m(value, shown):
    assert format_downloads(value) == shown


def test_a_count_no_float_could_carry_is_still_formatted():
    # `downloads` is registry-controlled text and a Python integer has no upper bound, so a float
    # division raises `OverflowError` -- one column of one line would end the whole dialog.
    huge = 10**400
    value = read_downloads(huge)
    hit = SearchHit(repo="community-user/Odd-GGUF", unresolved_reason="relation_unknown", downloads=value)

    assert hit.downloads == huge
    assert format_downloads(value) == f"{huge // 1_000_000}.0M"
