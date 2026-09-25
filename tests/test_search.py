"""Tests for modelroom.search: the Hugging Face search, resolution, age and the written config.

Network-facing code is tested against recorded fixtures only, never the live API. Since
2026-09-24 one search asks one page per account and, with the owner filter off, two open pages on
top; the pinned answers are `tests/fixtures/hf_search_qwen_*.json`, bound by account in
`fixture_support.search_transport_mapping` (see `tests/fixtures/README.md` for how they were
recorded and why their entries are curated). The request forms themselves are pinned in
`tests/test_search_pages.py`.
"""

from __future__ import annotations

import json
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from modelroom.catalog import Catalog, load_catalog
from modelroom.config import ConfigError, Configuration, load_config, package_targets
from modelroom.http import BudgetedTransport, RequestBudget, Response
from modelroom.search import (
    DEFAULT_PACKAGERS,
    DEFAULT_GUIDED_BUDGET,
    NO_OLLAMA_LABEL,
    SearchError,
    apply_hits,
    decide_age,
    ollama_label,
    owner_class,
    parse_ollama_entry,
    run_search,
    write_configuration,
)
from modelroom.search_pages import account_search_url, most_downloaded_url, newest_url
from modelroom.state import LockHeldError, acquire_lock, release_lock

from fixture_support import (
    SEARCH_ACCOUNT_PAGES,
    build_transport,
    json_response,
    mistral_search_mapping,
    qwen35_example_config_dict,
    search_transport_mapping,
)

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).resolve().parent.parent

QWEN_INFO = "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"
DEEPSEEK_REPO = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
DEEPSEEK_INFO = f"https://huggingface.co/api/models/{DEEPSEEK_REPO}"
# The accounts one search for `qwen` asks, in the order it asks them: the catalog's two matching
# publishers first (`Qwen` by family name, `deepseek-ai` because one of its model ids carries the
# word), then the shipped positive list.
ASKED_ACCOUNTS = ["Qwen", "deepseek-ai", *DEFAULT_PACKAGERS]


def _catalog() -> Catalog:
    return load_catalog()


def _search_mapping(extra: dict | None = None) -> dict:
    mapping = search_transport_mapping()
    mapping.update(extra or {})
    return mapping


def _one_page_mapping(entries: list[dict]) -> dict:
    """Every account answers with nothing but `unsloth`, which answers with `entries`.

    For the cases about one entry (a malformed id, a `createdAt` at the edge of the calendar): the
    other accounts are still asked -- a page is never left out -- and answer with an empty list.
    """
    mapping = {
        ("GET", account_search_url("qwen", account)): json_response("hf_search_none.json")
        for account in ASKED_ACCOUNTS
    }
    mapping[("GET", account_search_url("qwen", "unsloth"))] = Response(
        status=200, body=json.dumps(entries).encode("utf-8")
    )
    return mapping


def _hit(outcome, repo: str):
    return next(hit for hit in outcome.hits if hit.repo == repo)


def _base_model_of(config: Configuration, hf_repo: str):
    """The configured base model with this `hf_repo`, whichever family holds it."""
    return next(
        base_model
        for family in config.families
        for base_model in family.base_models
        if base_model.hf_repo == hf_repo
    )


# --- the request itself --------------------------------------------------------------------


def test_the_filter_asks_one_page_per_account_in_the_order_of_the_groups():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    pages = [call for call in transport.calls if "filter=gguf" in call[1]]
    assert pages == [("GET", account_search_url("qwen", account)) for account in ASKED_ACCOUNTS]
    assert [group.label for group in outcome.groups] == ASKED_ACCOUNTS
    assert outcome.requests == len(ASKED_ACCOUNTS) == 7


def test_the_filter_asks_no_open_page():
    transport = build_transport(_search_mapping())

    run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert ("GET", most_downloaded_url("qwen")) not in transport.calls
    assert ("GET", newest_url("qwen")) not in transport.calls


def test_the_budget_is_one_request_per_account_plus_one_age_lookup_per_base_model():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    # seven accounts, plus one age lookup per *distinct* resolved base model (two of them for
    # three resolved hits): a hit itself costs no request
    assert outcome.budget_used == 9
    assert outcome.budget_limit == 60


def test_without_the_filter_the_two_open_pages_follow_the_accounts():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True)

    pages = [call for call in transport.calls if "filter=gguf" in call[1]]
    assert pages[-2:] == [("GET", most_downloaded_url("qwen")), ("GET", newest_url("qwen"))]
    assert [group.label for group in outcome.groups][-2:] == ["most downloaded", "newest"]
    assert outcome.budget_used == 11


def test_without_the_filter_the_account_groups_stay_and_the_open_lists_come_on_top():
    filtered = run_search(build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60))
    open_run = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    assert [hit.repo for hit in open_run.hits][: len(filtered.hits)] == [hit.repo for hit in filtered.hits]
    assert len(open_run.hits) > len(filtered.hits)


def test_a_repository_two_pages_answer_with_is_listed_once_in_its_account_group():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    repos = [hit.repo for hit in outcome.hits]
    assert repos.count("unsloth/Qwen3.5-9B-GGUF") == 1
    unsloth = next(group for group in outcome.groups if group.label == "unsloth")
    assert "unsloth/Qwen3.5-9B-GGUF" in [hit.repo for hit in unsloth.hits]
    most_downloaded = next(group for group in outcome.groups if group.label == "most downloaded")
    assert most_downloaded.already_listed == 1
    assert most_downloaded.line() == "most downloaded 3, 1 of them already listed"


EUROLLM = "utter-project/EuroLLM-9B-Instruct"
EUROLLM_INFO = f"https://huggingface.co/api/models/{EUROLLM}"


def _quantization_of(repo: str, base: str, downloads: int) -> dict:
    """One answer entry that declares itself a quantization of `base`, as the Hub words it."""
    return {
        "id": repo,
        "tags": ["gguf", f"base_model:{base}", f"base_model:quantized:{base}"],
        "cardData": {"base_model": [base], "license": "apache-2.0"},
        "downloads": downloads,
    }


@pytest.mark.parametrize("page", ["unsloth", "most downloaded"])
def test_a_repository_answered_twice_is_listed_once_and_costs_no_second_age_lookup(page):
    # The answer is registry-controlled text and nothing promises each id appears once, on one page
    # or across two. The duplicate declares a base model *no other hit declares*, so a version that
    # resolved first and deduplicated afterwards would spend an age lookup for it -- which this test
    # would then see in `transport.calls` and in the budget.
    repo = "unsloth/Qwen3.5-9B-GGUF"
    first = _quantization_of(repo, "Qwen/Qwen3.5-9B", 111)
    duplicate = _quantization_of(repo, EUROLLM, 222)
    unsloth_page = [first] if page != "unsloth" else [first, duplicate]
    extra: dict = {
        ("GET", account_search_url("qwen", "unsloth")): Response(
            status=200, body=json.dumps(unsloth_page).encode("utf-8")
        )
    }
    if page == "most downloaded":
        extra[("GET", most_downloaded_url("qwen"))] = Response(
            status=200, body=json.dumps([duplicate]).encode("utf-8")
        )
        extra[("GET", newest_url("qwen"))] = Response(status=200, body=b"[]")
    transport = build_transport(_search_mapping(extra))

    outcome = run_search(
        transport,
        "qwen",
        catalog=_catalog(),
        budget=RequestBudget(60),
        open_pages=page == "most downloaded",
    )

    # listed once, in the group that had it first, with the metadata of that first entry
    unsloth = next(group for group in outcome.groups if group.label == "unsloth")
    assert [(hit.repo, hit.downloads) for hit in unsloth.hits] == [(repo, 111)]
    assert [hit.repo for hit in outcome.hits].count(repo) == 1
    assert _hit(outcome, repo).resolved_base_model == "Qwen/Qwen3.5-9B"
    # the duplicate was never resolved, so its base model was never asked about
    assert ("GET", EUROLLM_INFO) not in transport.calls
    # seven accounts (plus the two open pages in that case) and the one age lookup the only
    # resolved base model of these pages costs -- nothing more
    assert outcome.budget_used == (10 if page == "most downloaded" else 8)
    counted = next(group for group in outcome.groups if group.label == page)
    assert counted.already_listed == 1
    assert counted.line() == ("unsloth 2, 1 of them already listed" if page == "unsloth" else "most downloaded 1, 1 of them already listed")


def test_a_full_open_page_carries_no_page_full_note():
    # Ten is what an open list always answers with, so the note would stand on every search and
    # say nothing about the word (measured live 2026-09-24, both open pages answered ten).
    full_ten = [
        {"id": f"community-user/Qwen-Part{index:02d}-GGUF", "tags": ["gguf"], "cardData": {}} for index in range(10)
    ]
    transport = build_transport(
        _search_mapping(
            {("GET", most_downloaded_url("qwen")): Response(status=200, body=json.dumps(full_ten).encode("utf-8"))}
        )
    )

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True)

    most_downloaded = next(group for group in outcome.groups if group.label == "most downloaded")
    assert most_downloaded.page_size == 10
    assert most_downloaded.page_full is False
    assert most_downloaded.line() == "most downloaded 10"


def test_a_group_line_per_account_names_the_account_and_its_number():
    outcome = run_search(build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert outcome.group_lines() == [
        "Qwen 1",
        "deepseek-ai 0",
        "unsloth 3",
        "bartowski 0",
        "mradermacher 0",
        "lmstudio-community 0",
        "ggml-org 0",
    ]


def test_the_search_log_carries_one_entry_per_asked_page_with_its_class():
    """The numbers of the group lines, as data: `search.json` (CONTRACTS.md, "Search log")."""
    from datetime import datetime, timezone

    outcome = run_search(build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60))

    log = outcome.search_log(
        filtered=True, run_at=datetime(2026, 9, 25, 8, 0, 0, tzinfo=timezone.utc), unresolved={"no base model": 1}
    )

    assert log["schema_version"] == 1
    assert (log["word"], log["filter_owners"], log["run_at"]) == ("qwen", True, "2026-09-25T08:00:00+00:00")
    assert [entry["account"] for entry in log["accounts"]] == [group.label for group in outcome.groups]
    assert [entry["hits"] for entry in log["accounts"]] == [group.page_size for group in outcome.groups]
    assert {entry["account"]: entry["class"] for entry in log["accounts"]}["Qwen"] == "publisher"
    assert {entry["account"]: entry["class"] for entry in log["accounts"]}["unsloth"] == "packager"
    assert (log["requests"], log["resolved"]) == (outcome.requests, outcome.resolved)
    assert log["budget"] == {"used": outcome.budget_used, "limit": 60}
    assert log["unresolved"] == [{"reason": "no base model", "count": 1}]


def test_the_two_open_lists_are_a_class_of_their_own_in_the_search_log():
    from datetime import datetime, timezone

    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True)
    log = outcome.search_log(
        filtered=False, run_at=datetime(2026, 9, 25, tzinfo=timezone.utc), unresolved={}
    )

    assert [entry["account"] for entry in log["accounts"] if entry["class"] == "open"] == [
        "most downloaded",
        "newest",
    ]


def test_the_two_notes_of_the_screen_come_from_the_same_reading_as_the_file():
    """One search, one set of numbers: the file and the screen may not disagree (2026-09-24)."""
    from datetime import datetime, timezone

    from modelroom.screen import search_notes

    outcome = run_search(build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60))
    log = outcome.search_log(filtered=True, run_at=datetime(2026, 9, 25, tzinfo=timezone.utc), unresolved={})

    notes = search_notes(log, len(outcome.hits), 2)

    # `deepseek-ai` is a publisher of the catalog and was asked; it answered with nothing, so the
    # note does not name it.
    assert "deepseek-ai" in outcome.publishers
    assert notes[0] == "searched Hugging Face at the publisher Qwen and the five listed packagers"
    assert notes[1] == f"{len(outcome.hits)} repositories, 2 of them models you can pick from"


def test_a_full_account_page_says_that_a_more_specific_word_shortens_the_list():
    transport = build_transport(
        _search_mapping(
            {("GET", account_search_url("qwen", "unsloth")): json_response("hf_search_qwen_page_full.json")}
        )
    )

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    unsloth = next(group for group in outcome.groups if group.label == "unsloth")
    assert unsloth.page_full is True
    assert unsloth.line() == "unsloth 20 (page full: a more specific word shortens the list)"


def test_a_page_one_entry_short_of_the_limit_is_not_called_full():
    full = json.loads((Path(__file__).parent / "fixtures" / "hf_search_qwen_page_full.json").read_text("utf-8"))
    transport = build_transport(
        _search_mapping(
            {
                ("GET", account_search_url("qwen", "unsloth")): Response(
                    status=200, body=json.dumps(full[:19]).encode("utf-8")
                )
            }
        )
    )

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    unsloth = next(group for group in outcome.groups if group.label == "unsloth")
    assert unsloth.page_full is False
    assert unsloth.line() == "unsloth 19"


def _no_family_mapping(open_pages: bool = False) -> dict:
    mapping = {
        ("GET", account_search_url("nebula", account)): json_response("hf_search_none.json")
        for account in DEFAULT_PACKAGERS
    }
    if open_pages:
        mapping[("GET", most_downloaded_url("nebula"))] = json_response("hf_search_none.json")
        mapping[("GET", newest_url("nebula"))] = json_response("hf_search_none.json")
    return mapping


def test_a_word_no_catalog_family_matches_asks_the_packagers_and_says_so():
    outcome = run_search(
        build_transport(_no_family_mapping()), "nebula", catalog=_catalog(), budget=RequestBudget(60)
    )

    assert [group.label for group in outcome.groups] == list(DEFAULT_PACKAGERS)
    assert outcome.notes == ["no publisher in the catalog matches 'nebula'; only the listed packagers were asked"]
    assert outcome.group_lines()[0] == outcome.notes[0]


def test_without_the_filter_that_note_names_the_open_lists_as_well():
    # "only the listed packagers were asked" would be untrue here: both open lists were asked too.
    outcome = run_search(
        build_transport(_no_family_mapping(open_pages=True)),
        "nebula",
        catalog=_catalog(),
        budget=RequestBudget(60),
        open_pages=True,
    )

    assert outcome.notes == [
        "no publisher in the catalog matches 'nebula'; the listed packagers and the two open lists were asked"
    ]


def test_a_publisher_that_is_also_a_listed_packager_is_asked_once():
    mapping = {
        ("GET", account_search_url("qwen", account)): json_response(name)
        for account, name in SEARCH_ACCOUNT_PAGES.items()
    }
    mapping[("GET", QWEN_INFO)] = json_response("hf_qwen_qwen35_9b_model.json")
    mapping[("GET", DEEPSEEK_INFO)] = Response(
        status=200, body=json.dumps({"id": DEEPSEEK_REPO, "sha": "b" * 40}).encode("utf-8")
    )
    transport = build_transport(mapping)

    outcome = run_search(
        transport,
        "qwen",
        catalog=_catalog(),
        listed_packagers=("unsloth", "Qwen"),
        budget=RequestBudget(60),
    )

    assert [group.label for group in outcome.groups] == ["Qwen", "deepseek-ai", "unsloth"]
    assert transport.calls.count(("GET", account_search_url("qwen", "Qwen"))) == 1


def test_run_search_defaults_to_the_shared_guided_budget():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog())

    # 150, not 60 (decided 2026-09-24): measured live, `qwen` with the owner filter on spends 47
    # requests before the first question is answered -- 7 account pages and one age lookup per
    # distinct resolved base model -- and a budget of 60 left the fetch of two chosen repositories
    # `incomplete (budget exhausted)`. A plain `modelroom fetch` has 400 on its own.
    assert outcome.budget_limit == DEFAULT_GUIDED_BUDGET == 150


def test_run_search_reports_a_failing_status_and_names_the_account():
    transport = build_transport(
        _search_mapping({("GET", account_search_url("qwen", "mradermacher")): Response(status=503, body=b"")})
    )

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert "503" in str(exc.value)
    assert "mradermacher" in str(exc.value)


@pytest.mark.parametrize(
    "body, message",
    [
        (b"{}", "the answer is not a list of repositories"),
        (b"[1, 2]", "the answer is not a list of repositories"),
        (b"not json at all", "invalid JSON"),
    ],
)
def test_run_search_reports_a_body_it_cannot_read_and_names_the_page(body, message):
    transport = build_transport(
        _search_mapping({("GET", account_search_url("qwen", "Qwen")): Response(status=200, body=body)})
    )

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert str(exc.value).startswith("search for 'qwen' (Qwen): ")
    assert message in str(exc.value)
    # the page that failed is the first one, so no later account was asked
    assert ("GET", account_search_url("qwen", "unsloth")) not in transport.calls


def test_a_transport_that_raises_ends_the_search_with_the_page_in_the_message():
    def failing(method: str, url: str, headers=None):
        raise OSError("the name does not resolve")

    with pytest.raises(SearchError) as exc:
        run_search(failing, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert str(exc.value) == "search for 'qwen' (Qwen) failed: the name does not resolve"


def test_a_budget_that_ends_before_the_open_pages_says_no_account_is_missing():
    # Every account was asked; what is left over is the two open pages, so the message counts
    # pages and not accounts.
    transport = build_transport(_search_mapping())

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(9), open_pages=True)

    assert str(exc.value) == (
        "search for 'qwen' (most downloaded) not made: budget exhausted; 2 pages were not asked"
    )
    assert ("GET", newest_url("qwen")) not in transport.calls


def test_run_search_reports_an_exhausted_budget_instead_of_an_empty_result():
    transport = build_transport(_search_mapping())

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(0))

    assert "budget" in str(exc.value)
    assert "7 accounts were not asked" in str(exc.value)


def test_a_budget_that_ends_in_the_middle_names_the_accounts_still_unasked():
    transport = build_transport(_search_mapping())

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(2))

    # the first page and the age lookup its resolved hit costs used both requests, so the second
    # account is where the budget ends -- and six of the seven accounts have no answer
    assert str(exc.value) == (
        "search for 'qwen' (deepseek-ai) not made: budget exhausted; 6 accounts were not asked: "
        "deepseek-ai, unsloth, bartowski, mradermacher, lmstudio-community, ggml-org"
    )


# --- resolution ------------------------------------------------------------------------------


def test_the_unsloth_hit_resolves_to_its_publisher_base_model():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, "unsloth/Qwen3.5-9B-GGUF")
    assert hit.resolved is True
    assert hit.resolved_base_model == "Qwen/Qwen3.5-9B"
    assert hit.base_model == ["Qwen/Qwen3.5-9B"]
    assert hit.base_model_relation == "quantized"
    assert hit.unresolved_reason is None
    assert hit.publisher_status == "listed packager"
    assert hit.license == "apache-2.0"
    assert hit.repo_created_at == datetime(2026, 2, 28, 14, 4, 29, tzinfo=timezone.utc)
    assert hit.parameters_b is None
    assert hit.downloads == 1_626_475


def test_a_search_for_mistral_resolves_the_publishers_own_build_and_a_packagers():
    """Test round of 2026-09-24: `mistral` found repositories and not one of them was selectable,
    because `mistralai` was no publisher of the catalog. It is one since 2026-09-25."""
    outcome = run_search(
        build_transport(mistral_search_mapping()), "mistral", catalog=_catalog(), budget=RequestBudget(60)
    )

    assert outcome.notes == []
    assert [hit.unresolved_reason for hit in outcome.hits] == [None, None, None, None]
    publisher_hit = _hit(outcome, "mistralai/Ministral-3-14B-Instruct-2512-GGUF")
    assert publisher_hit.resolved is True
    assert publisher_hit.publisher_status == "publisher"
    assert publisher_hit.resolved_base_model == "mistralai/Ministral-3-14B-Instruct-2512"
    assert publisher_hit.ollama == "ministral-3:14b"
    assert publisher_hit.age == "latest"
    packager_hit = _hit(outcome, "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF")
    assert packager_hit.resolved is True
    assert packager_hit.publisher_status == "listed packager"
    assert packager_hit.resolved_base_model == "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    assert packager_hit.ollama == "mistral-small3.2:24b"


def test_the_publishers_own_gguf_repo_is_marked_publisher_and_carries_its_parameter_count():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, "Qwen/Qwen3.5-9B-GGUF")
    assert hit.resolved is True
    assert hit.publisher_status == "publisher"
    assert hit.parameters_b == pytest.approx(9.653104368)


def test_a_finetune_hit_is_unresolved_as_a_derivative():
    # A fine-tune of a foreign account is what an open list brings in, so it is only there.
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    hit = _hit(outcome, "community-user/Qwen3.5-9B-Roleplay-GGUF")
    assert hit.resolved is False
    assert hit.unresolved_reason == "derivative"
    assert hit.publisher_status == "other"
    assert hit.ollama is None


@pytest.mark.parametrize(
    "repo, reason",
    [
        ("unsloth/Qwen3.5-9B-Draft-GGUF", "relation_unknown"),
        ("community-user/Qwen3.5-9B-Mystery-GGUF", "base_model_tag"),
        ("community-user/Qwen3.5-9B-Plain-GGUF", "relation_unknown"),
        ("community-user/Nebula-9B-GGUF", "publisher_unknown"),
        ("community-user/Qwen3.5-9B-Conflict-GGUF", "metadata_conflict"),
    ],
)
def test_every_unresolved_reason_is_reported_from_the_pinned_answers(repo, reason):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    hit = _hit(outcome, repo)
    assert hit.resolved is False
    assert hit.unresolved_reason == reason


def test_an_account_of_the_positive_list_can_answer_with_an_unresolved_repository_too():
    # The owner filter narrows the *request* to the accounts, not the resolution: a repository of
    # a listed packager that says nothing about a base model is still unresolved.
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, "unsloth/Qwen3.5-9B-Draft-GGUF")
    assert (hit.resolved, hit.unresolved_reason, hit.publisher_status) == (
        False,
        "relation_unknown",
        "listed packager",
    )


def test_an_unresolved_hit_inherits_neither_an_ollama_name_nor_an_age():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    for hit in outcome.hits:
        if not hit.resolved:
            assert hit.ollama is None
            assert hit.age == "unknown"
            assert hit.successor is None


def test_the_summary_line_counts_repositories_resolved_unresolved_requests_and_the_budget():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    assert outcome.summary_line() == "4 repositories, 3 resolved, 1 unresolved, 7 requests, budget 9/60"


def test_the_summary_line_of_a_run_without_the_filter_counts_the_open_pages_too():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    assert outcome.summary_line() == "9 repositories, 3 resolved, 6 unresolved, 9 requests, budget 11/60"


def test_the_flat_hit_list_is_the_groups_one_after_the_other():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    assert [hit.repo for hit in outcome.hits] == [hit.repo for group in outcome.groups for hit in group.hits]
    assert [hit.repo for hit in outcome.hits] == [
        "Qwen/Qwen3.5-9B-GGUF",
        "unsloth/Qwen3.5-9B-Draft-GGUF",
        "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        "unsloth/Qwen3.5-9B-GGUF",
        "community-user/Qwen3.5-9B-Roleplay-GGUF",
        "community-user/Nebula-9B-GGUF",
        "community-user/Qwen3.5-9B-Mystery-GGUF",
        "community-user/Qwen3.5-9B-Conflict-GGUF",
        "community-user/Qwen3.5-9B-Plain-GGUF",
    ]


@pytest.mark.parametrize(
    "repo, downloads",
    [
        ("Qwen/Qwen3.5-9B-GGUF", 12_031_627),
        ("unsloth/Qwen3.5-9B-Draft-GGUF", 3551),
        ("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", 68_445),
        ("community-user/Qwen3.5-9B-Plain-GGUF", 0),
        # the one entry of the pinned answers that carries no `downloads` field at all
        ("community-user/Qwen3.5-9B-Mystery-GGUF", None),
    ],
)
def test_the_downloads_of_a_hit_come_from_the_answer(repo, downloads):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    assert _hit(outcome, repo).downloads == downloads


@pytest.mark.parametrize(
    "declared",
    [
        "not-a-repo-id",
        "(no single declared base)",
        "a/b/c",
        # A path segment, not a repository: the shared `hf_repo` pattern still lets these
        # through, and the declared base becomes a URL path segment of the age lookup.
        "Qwen/..",
        "Qwen/.",
        "../x",
        "Qwen/-leading-hyphen",
    ],
)
def test_a_declared_base_that_is_not_a_repo_id_is_unresolved_not_a_crash(declared):
    # The declared base comes from the registry; a value that is not a repository id (including
    # one that happens to equal the module's own "no single base" marker) must end as a reason,
    # never as a `SearchHit` validation error out of `run_search`.
    entries = [
        {
            "id": "community-user/Odd-GGUF",
            "tags": [f"base_model:{declared}", f"base_model:quantized:{declared}"],
            "cardData": {},
        }
    ]
    transport = build_transport(_one_page_mapping(entries))

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    hit = _hit(outcome, "community-user/Odd-GGUF")
    assert hit.resolved is False
    assert hit.unresolved_reason == "base_model_tag"


@pytest.mark.parametrize(
    "created", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-05:00", "not a date", 17, None]
)
def test_a_created_at_the_conversion_cannot_carry_is_unknown_not_a_crash(created):
    # `createdAt` is registry text; a value at the edge of the calendar makes the conversion to
    # UTC overflow, which must end as "unknown", never as an exception out of the search.
    entries = [{"id": "community-user/Odd-GGUF", "tags": [], "cardData": {}, "createdAt": created}]
    transport = build_transport(_one_page_mapping(entries))

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert _hit(outcome, "community-user/Odd-GGUF").repo_created_at is None


@pytest.mark.parametrize(
    "entry, message",
    [
        ({}, "a result has no repository id"),
        ({"id": 17}, "a result has no repository id"),
        ({"id": "no-owner", "tags": [], "cardData": {}}, "a result is not a repository id"),
    ],
)
def test_an_entry_without_a_usable_repo_id_ends_the_search_and_names_the_page(entry, message):
    # An answer of twenty entries under one of seven accounts: without the page in the message a
    # reader cannot tell which request to look at.
    transport = build_transport(
        _search_mapping({("GET", account_search_url("qwen", "bartowski")): Response(
            status=200, body=json.dumps([entry]).encode("utf-8")
        )})
    )

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert str(exc.value).startswith("search for 'qwen' (bartowski): ")
    assert message in str(exc.value)


def test_the_shipped_packager_allow_list_matches_the_example_configuration():
    example = tomllib.loads((REPO_ROOT / "modelroom.example.toml").read_text(encoding="utf-8"))

    assert list(DEFAULT_PACKAGERS) == example["packagers"]


def test_owner_class_labels_a_fetch_target_the_same_way_as_a_search_hit():
    catalog = _catalog()

    assert owner_class("Qwen", catalog=catalog) == "publisher"
    assert owner_class("unsloth", catalog=catalog) == "listed packager"
    assert owner_class("community-user", catalog=catalog) == "other"
    assert owner_class("unsloth", catalog=catalog, listed_packagers=()) == "other"


def test_an_owner_outside_both_lists_is_other_not_a_packager():
    # With an empty positive list `unsloth` is asked no page of its own, so the repository can only
    # come in through an open list -- and there it is labeled `other`, which is what this is about.
    outcome = run_search(
        build_transport(_search_mapping()),
        "qwen",
        catalog=_catalog(),
        listed_packagers=(),
        budget=RequestBudget(60),
        open_pages=True,
    )

    assert [group.label for group in outcome.groups] == ["Qwen", "deepseek-ai", "most downloaded", "newest"]
    assert _hit(outcome, "unsloth/Qwen3.5-9B-GGUF").publisher_status == "other"


# --- Ollama assignment -----------------------------------------------------------------------


def test_a_resolved_hit_takes_its_ollama_name_from_the_catalog():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    assert _hit(outcome, "unsloth/Qwen3.5-9B-GGUF").ollama == "qwen3.5:9b"


def test_a_user_entry_overrides_the_catalogs_ollama_name():
    outcome = run_search(
        build_transport(_search_mapping()),
        "qwen",
        catalog=_catalog(),
        budget=RequestBudget(60),
        ollama_entries={"unsloth/Qwen3.5-9B-GGUF": "qwen3.5:9b-q4_K_M"},
    )

    assert _hit(outcome, "unsloth/Qwen3.5-9B-GGUF").ollama == "qwen3.5:9b-q4_K_M"


def test_the_groups_carry_the_same_hit_objects_as_the_flat_list_after_such_an_entry():
    # A typed Ollama name builds new `SearchHit` values, so groups that kept the ones from before
    # would show a different Ollama package than the flat list does.
    outcome = run_search(
        build_transport(_search_mapping()),
        "qwen",
        catalog=_catalog(),
        budget=RequestBudget(60),
        ollama_entries={"unsloth/Qwen3.5-9B-GGUF": "qwen3.5:9b-q4_K_M"},
    )

    grouped = [hit for group in outcome.groups for hit in group.hits]
    assert grouped == outcome.hits
    assert all(grouped[index] is outcome.hits[index] for index in range(len(grouped)))
    in_group = next(hit for hit in grouped if hit.repo == "unsloth/Qwen3.5-9B-GGUF")
    assert in_group.ollama == "qwen3.5:9b-q4_K_M"
    # the name belongs to the base model, so the publisher's own repository carries it as well
    publisher_hit = next(hit for hit in grouped if hit.repo == "Qwen/Qwen3.5-9B-GGUF")
    assert publisher_hit.ollama == "qwen3.5:9b-q4_K_M"


def test_parse_ollama_entry_reads_the_typed_line():
    assert parse_ollama_entry("ollama: qwen3.5:9b") == "qwen3.5:9b"
    assert parse_ollama_entry("  OLLAMA:qwen3.5:9b  ") == "qwen3.5:9b"


@pytest.mark.parametrize("line", ["qwen3.5:9b", "ollama: qwen3.5", "ollama:", "ollama: a b:c"])
def test_parse_ollama_entry_rejects_anything_else(line):
    with pytest.raises(ValueError):
        parse_ollama_entry(line)


def test_a_resolved_model_the_catalog_gives_no_ollama_name_is_labeled_none_known():
    # The decisive case: the hit *resolved*, its publisher is known, and the catalog simply has no
    # Ollama assignment for that model -- nothing may be invented from the name.
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF")
    assert hit.resolved is True
    assert hit.resolved_base_model == DEEPSEEK_REPO
    assert hit.ollama is None
    assert ollama_label(hit) == NO_OLLAMA_LABEL


def test_a_model_without_an_assignment_is_labeled_none_known():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    assert ollama_label(_hit(outcome, "unsloth/Qwen3.5-9B-GGUF")) == "qwen3.5:9b"
    assert ollama_label(_hit(outcome, "community-user/Nebula-9B-GGUF")) == NO_OLLAMA_LABEL
    assert NO_OLLAMA_LABEL == "none known"


# --- latest / legacy from positive evidence ---------------------------------------------------


def _catalog_with(models: list[dict]) -> Catalog:
    return Catalog.model_validate({"schema_version": 1, "families": [{"name": "nova", "publisher": "acme", "models": models}]})


def _plain(hf_repo: str) -> dict:
    return {"hf_repo": hf_repo, "source": f"https://huggingface.co/{hf_repo}"}


def _catalog_calling_nova_latest() -> Catalog:
    """A catalog that states `latest` for `acme/Nova-7B`, with its evidence.

    Used wherever the rule is "this is `unknown` and must *not* fall back to the catalog": with a
    silent catalog those tests would pass even if the fallback happened.
    """
    return _catalog_with(
        [
            {
                "hf_repo": "acme/Nova-7B",
                "latest": True,
                "latest_source": "https://huggingface.co/acme",
                "latest_checked": date(2026, 9, 23).isoformat(),
                "source": "https://huggingface.co/acme/Nova-7B",
            }
        ]
    )


def _info(card: dict | None, repo: str = "acme/Nova-7B") -> Response:
    """A model-info answer for `repo`, in the shape the Hub sends it (`id` included)."""
    payload: dict = {"id": repo, "sha": "a" * 40}
    if card is not None:
        payload["cardData"] = card
    return Response(status=200, body=json.dumps(payload).encode("utf-8"))


def _url(repo: str) -> str:
    return f"https://huggingface.co/api/models/{repo}"


def test_a_valid_new_version_makes_the_model_legacy_with_its_successor():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({}, "acme/Nova-7B-2512"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")


def test_a_second_valid_edge_names_the_youngest_successor_and_stays_legacy():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({"new_version": "acme/Nova-7B-2606"}, "acme/Nova-7B-2512"),
            ("GET", _url("acme/Nova-7B-2606")): _info({"new_version": "acme/Nova-7B-2701"}, "acme/Nova-7B-2606"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    # at most two edges are followed; the third jump is never made
    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2606")
    assert ("GET", _url("acme/Nova-7B-2701")) not in transport.calls


def test_a_failing_second_edge_leaves_the_model_legacy_with_the_first_successor():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({"new_version": "acme/Nova-7B-gone"}, "acme/Nova-7B-2512"),
            ("GET", _url("acme/Nova-7B-gone")): Response(status=401, body=b"{}"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")


def test_a_new_version_target_that_answers_for_another_repository_is_unknown():
    # The transport follows redirects, so a request for `acme/Nova-7B-moved` can be answered by
    # `other/Nova-7B` -- which would slip past the same-account rule. The answer has to name the
    # repository that was asked for before it counts as that repository existing.
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-moved"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-moved")): _info({}, "other/Nova-7B"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_target_that_redirects_back_to_the_model_itself_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-moved"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-moved")): _info({}, "acme/Nova-7B"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_target_whose_answer_names_no_repository_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): Response(status=200, body=b"{}"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_target_that_does_not_exist_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-gone"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-gone")): Response(status=401, body=b"{}"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_under_another_account_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({"new_version": "other/Nova-7B-2512"}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert verdict.age == "unknown"
    assert ("GET", _url("other/Nova-7B-2512")) not in transport.calls


def test_a_self_referencing_new_version_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B"}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_cycle_on_the_second_edge_keeps_the_first_successor():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({"new_version": "acme/Nova-7B"}, "acme/Nova-7B-2512"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")


def test_a_budget_that_ends_before_the_first_edge_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    budget = RequestBudget(1)
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({}, "acme/Nova-7B-2512"),
        }
    )

    verdict = decide_age(BudgetedTransport(transport, budget), catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)
    assert budget.remaining == 0


def test_a_budget_that_ends_after_the_first_valid_edge_stays_legacy():
    # The second edge only names a younger successor; running out of budget for it must not undo
    # the `legacy` the first edge already proved.
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    budget = RequestBudget(2)
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info(
                {"new_version": "acme/Nova-7B-2606"}, "acme/Nova-7B-2512"
            ),
            ("GET", _url("acme/Nova-7B-2606")): _info({}, "acme/Nova-7B-2606"),
        }
    )

    verdict = decide_age(BudgetedTransport(transport, budget), catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")
    assert budget.remaining == 0
    assert ("GET", _url("acme/Nova-7B-2606")) not in transport.calls


def test_a_budget_that_ends_before_the_repo_is_read_at_all_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport({})

    verdict = decide_age(BudgetedTransport(transport, RequestBudget(0)), catalog, "acme/Nova-7B")

    assert verdict.age == "unknown"
    assert transport.calls == []


def test_without_a_new_version_the_catalogs_latest_statement_decides():
    catalog = _catalog_with(
        [
            {
                "hf_repo": "acme/Nova-7B",
                "latest": True,
                "latest_source": "https://huggingface.co/acme",
                "latest_checked": date(2026, 9, 23).isoformat(),
                "source": "https://huggingface.co/acme/Nova-7B",
            }
        ]
    )
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("latest", None)


def test_without_a_new_version_a_catalog_successor_makes_the_model_legacy():
    catalog = _catalog_with(
        [
            {
                "hf_repo": "acme/Nova-7B",
                "successor": "acme/Nova-7B-2512",
                "source": "https://huggingface.co/acme/Nova-7B-2512",
            },
            _plain("acme/Nova-7B-2512"),
        ]
    )
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")


def test_a_model_the_catalog_says_nothing_about_is_unknown():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_that_is_not_a_string_is_no_evidence_and_the_catalog_decides():
    catalog = _catalog_with(
        [
            {
                "hf_repo": "acme/Nova-7B",
                "latest": True,
                "latest_source": "https://huggingface.co/acme",
                "latest_checked": date(2026, 9, 23).isoformat(),
                "source": "https://huggingface.co/acme/Nova-7B",
            }
        ]
    )
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({"new_version": ["acme/Nova-7B-2512"]}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert verdict.age == "latest"


# --- the resolution written back into the configuration ---------------------------------------


def _config(tmp_path) -> Configuration:
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data["schema_version"] = 2
    data["families"] = []
    data["publishers"] = []
    return Configuration.from_dict(data)


def _resolved_hits(tmp_path=None):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )
    return [hit for hit in outcome.hits if hit.resolved]


def test_apply_hits_writes_the_base_model_as_a_family_and_the_repo_as_an_owner_bound_target(tmp_path):
    config = _config(tmp_path)

    updated = apply_hits(config, _resolved_hits(), catalog=_catalog())

    # one family per resolved base model, named after the catalog's own family
    assert [family.name for family in updated.families] == ["qwen3.5", "deepseek-r1"]
    base_model = _base_model_of(updated, "Qwen/Qwen3.5-9B")
    assert base_model.hf_repo == "Qwen/Qwen3.5-9B"
    # the order of the groups: the publisher's own repository first, then the packager's
    assert base_model.repos == ["Qwen/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-9B-GGUF"]
    assert updated.publishers == ["Qwen", "deepseek-ai"]
    assert (base_model.ollama_base, base_model.ollama_tag) == ("qwen3.5", "9b")
    assert "unsloth/Qwen3.5-9B-GGUF" in package_targets(updated, base_model)


def _hits_with_entry(entry: str | None):
    outcome = run_search(
        build_transport(_search_mapping()),
        "qwen",
        catalog=_catalog(),
        budget=RequestBudget(60),
        ollama_entries={"Qwen/Qwen3.5-9B-GGUF": entry} if entry else None,
    )
    return [hit for hit in outcome.hits if hit.resolved]


def test_a_typed_ollama_name_reaches_the_configuration_whatever_the_hit_order(tmp_path):
    # The user's own assignment belongs to the base model, so it must not depend on which hit of
    # that base model happens to be processed first, nor on the base model already being there.
    hits = _hits_with_entry("qwen3.5:9b-q4_K_M")

    forwards = apply_hits(_config(tmp_path), hits, catalog=_catalog())
    backwards = apply_hits(_config(tmp_path), list(reversed(hits)), catalog=_catalog())

    for updated in (forwards, backwards):
        base_model = _base_model_of(updated, "Qwen/Qwen3.5-9B")
        assert (base_model.ollama_base, base_model.ollama_tag) == ("qwen3.5", "9b-q4_K_M")


def test_a_typed_ollama_name_also_replaces_one_the_configuration_already_carries(tmp_path):
    first = apply_hits(_config(tmp_path), _hits_with_entry(None), catalog=_catalog())
    assert _base_model_of(first, "Qwen/Qwen3.5-9B").ollama_tag == "9b"

    second = apply_hits(first, _hits_with_entry("qwen3.5:9b-q4_K_M"), catalog=_catalog())

    assert _base_model_of(second, "Qwen/Qwen3.5-9B").ollama_tag == "9b-q4_K_M"


def test_two_typed_ollama_names_for_one_base_model_end_the_search():
    with pytest.raises(SearchError) as exc:
        run_search(
            build_transport(_search_mapping()),
            "qwen",
            catalog=_catalog(),
            budget=RequestBudget(60),
            ollama_entries={
                "unsloth/Qwen3.5-9B-GGUF": "qwen3.5:9b",
                "Qwen/Qwen3.5-9B-GGUF": "qwen3.5:9b-q4_K_M",
            },
        )

    assert "Qwen/Qwen3.5-9B" in str(exc.value)


def test_apply_hits_refuses_hits_of_one_base_model_that_name_different_ollama_packages(tmp_path):
    # `run_search` already refuses contradictory input; this is the same rule at the writing end,
    # for a caller that assembles hits itself.
    hits = _hits_with_entry(None)
    contradictory = [hits[0], hits[1].model_copy(update={"ollama": "qwen3.5:9b-q4_K_M"})]

    with pytest.raises(ConfigError) as exc:
        apply_hits(_config(tmp_path), contradictory, catalog=_catalog())

    assert "Qwen/Qwen3.5-9B" in str(exc.value)


def test_apply_hits_leaves_the_original_configuration_untouched(tmp_path):
    config = _config(tmp_path)

    apply_hits(config, _resolved_hits(), catalog=_catalog())

    assert config.families == []
    assert config.publishers == []


def test_apply_hits_is_idempotent(tmp_path):
    config = _config(tmp_path)
    hits = _resolved_hits()

    once = apply_hits(config, hits, catalog=_catalog())
    twice = apply_hits(once, hits, catalog=_catalog())

    assert twice.model_dump(mode="json") == once.model_dump(mode="json")


def test_apply_hits_ignores_unresolved_hits(tmp_path):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )
    unresolved = [hit for hit in outcome.hits if not hit.resolved]

    updated = apply_hits(_config(tmp_path), unresolved, catalog=_catalog())

    assert updated.families == []


def test_apply_hits_names_a_family_after_the_base_model_when_the_catalog_has_none(tmp_path):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )
    hit = _hit(outcome, "unsloth/Qwen3.5-9B-GGUF")
    empty_catalog = Catalog.model_validate({"schema_version": 1, "families": []})

    updated = apply_hits(_config(tmp_path), [hit], catalog=empty_catalog)

    assert [family.name for family in updated.families] == ["qwen3.5-9b"]
    assert _base_model_of(updated, "Qwen/Qwen3.5-9B").ollama_base == "qwen3.5"


def test_write_configuration_writes_a_file_that_reads_back_as_the_same_configuration(tmp_path):
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW)

    reread = load_config(path)
    assert reread.model_dump(mode="json") == config.model_dump(mode="json")
    assert tomllib.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2


def test_write_configuration_leaves_out_a_context_no_guided_run_chose(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW)

    assert "context" not in tomllib.loads(path.read_text(encoding="utf-8")).get("guided", {})
    assert load_config(path).guided.context is None


def test_write_configuration_keeps_the_context_a_guided_run_chose(tmp_path):
    config = _config(tmp_path)
    config = config.model_copy(update={"guided": config.guided.model_copy(update={"context": 4096})})
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW)

    assert tomllib.loads(path.read_text(encoding="utf-8"))["guided"]["context"] == 4096
    assert load_config(path).guided.context == 4096


def test_write_configuration_writes_where_it_checked_even_through_a_symlink(tmp_path):
    # The check resolves the path (it has to, to know which directory confines `paths.state`), so
    # the write has to use the same resolved path: writing to the link itself would replace the
    # link with a file next to it, and the next `load_config` of the real file would then find a
    # state directory outside its own tree.
    real_dir = tmp_path / "real"
    work_dir = tmp_path / "work"
    real_dir.mkdir()
    work_dir.mkdir()
    real_path = real_dir / "modelroom.toml"
    link_path = work_dir / "modelroom.toml"
    try:
        link_path.symlink_to(Path("..") / "real" / "modelroom.toml")
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"this platform does not allow creating a symlink here: {exc}")
    real_path.write_text("", encoding="utf-8")
    data = qwen35_example_config_dict(str(real_dir / "state"), str(real_dir / "models.md"))
    data["schema_version"] = 2
    data["families"] = []
    data["publishers"] = []
    config = Configuration.from_dict(data)

    write_configuration(link_path, config, now=NOW)

    assert link_path.is_symlink(), "the link must still be a link, not a file of its own"
    assert load_config(real_path).model_dump(mode="json") == config.model_dump(mode="json")


def test_write_configuration_releases_the_lock_when_the_write_fails(tmp_path):
    # A guard that is only ever seen succeeding proves nothing about the `finally`.
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    path.mkdir()  # a directory where the file should go: os.replace cannot overwrite it

    with pytest.raises(OSError):
        write_configuration(path, config, now=NOW)

    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    release_lock(handle)


def test_write_configuration_stops_at_another_processs_lock(tmp_path):
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()
    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    try:
        with pytest.raises(LockHeldError):
            write_configuration(path, config, now=NOW)
    finally:
        release_lock(handle)

    assert not path.exists()


def test_write_configuration_refuses_a_state_directory_the_reader_would_reject(tmp_path):
    # `load_config` confines `paths.state` to the config file's own directory (F7); the writer
    # has to fail on the same rule rather than leave a file no command can read.
    data = qwen35_example_config_dict(str(tmp_path / "outside" / "state"), str(tmp_path / "models.md"))
    data["schema_version"] = 2
    data["families"] = []
    data["publishers"] = []
    config = Configuration.from_dict(data)
    path = tmp_path / "here" / "modelroom.toml"
    path.parent.mkdir()

    with pytest.raises(ConfigError):
        write_configuration(path, config, now=NOW)

    assert not path.exists()


def test_write_configuration_releases_the_lock_afterwards(tmp_path):
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW)

    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    release_lock(handle)
