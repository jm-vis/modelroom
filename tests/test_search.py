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
    CATALOG_BUDGET_RESERVE,
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
from modelroom.search_pages import (
    account_downloads_url,
    account_search_url,
    catalog_page_url,
    model_url,
    most_downloaded_url,
    newest_url,
    search_accounts,
)
from modelroom.search_word import latest_model_names
from modelroom.state import LockHeldError, acquire_lock, release_lock

from fixture_support import (
    CATALOG_PAGE_FOREIGN_REPO,
    CATALOG_PAGE_MODEL,
    CATALOG_PAGE_REPO,
    MISTRAL_BASE_MODELS,
    SEARCH_ACCOUNT_PAGES,
    TYPED_GGUF_REPO,
    TYPED_NO_GGUF_REPO,
    build_transport,
    catalog_transport_mapping,
    json_response,
    mistral_search_mapping,
    qwen35_example_config_dict,
    search_transport_mapping,
    typed_transport_mapping,
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


def _account_pages(
    account: str, newest: Response, downloads: Response | None = None, word: str = "qwen"
) -> dict:
    """The two pages one account is asked with since 2026-09-25: its newest, and its most downloaded.

    Without a second answer the same page stands for both sort orders, which is what an account
    with few repositories really answers -- the duplicate is then counted as `already listed`.
    """
    return {
        ("GET", account_search_url(word, account)): newest,
        ("GET", account_downloads_url(word, account)): newest if downloads is None else downloads,
    }


def _one_page_mapping(entries: list[dict]) -> dict:
    """Every account answers with nothing but `unsloth`, whose newest page answers with `entries`.

    For the cases about one entry (a malformed id, a `createdAt` at the edge of the calendar): the
    other accounts are still asked -- a page is never left out -- and answer with an empty list.
    """
    empty = json_response("hf_search_none.json")
    mapping: dict = {}
    for account in ASKED_ACCOUNTS:
        mapping.update(_account_pages(account, empty))
    mapping.update(
        _account_pages("unsloth", Response(status=200, body=json.dumps(entries).encode("utf-8")), empty)
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


def test_the_filter_asks_two_pages_per_account_in_the_order_of_the_groups():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    pages = [call for call in transport.calls if "filter=gguf" in call[1]]
    expected = [
        ("GET", url(account))
        for account in ASKED_ACCOUNTS
        for url in (lambda a: account_search_url("qwen", a), lambda a: account_downloads_url("qwen", a))
    ]
    assert pages == expected
    # One group per account all the same: a reader asks "was my account asked" once, and the two
    # sort orders are the package's business (decided 2026-09-25).
    assert [group.label for group in outcome.groups] == ASKED_ACCOUNTS
    assert [group.pages for group in outcome.groups] == [2] * 7
    assert outcome.requests == 2 * len(ASKED_ACCOUNTS) == 14


def test_the_filter_asks_no_open_page():
    transport = build_transport(_search_mapping())

    run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert ("GET", most_downloaded_url("qwen")) not in transport.calls
    assert ("GET", newest_url("qwen")) not in transport.calls


def test_the_budget_is_two_requests_per_account_plus_one_age_lookup_per_base_model():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    # seven accounts at two pages each, plus one age lookup per *distinct* resolved base model
    # (two of them for three resolved hits): a hit itself costs no request
    assert outcome.budget_used == 16
    assert outcome.budget_limit == 60


def test_without_the_filter_the_two_open_pages_follow_the_accounts():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True)

    pages = [call for call in transport.calls if "filter=gguf" in call[1]]
    assert pages[-2:] == [("GET", most_downloaded_url("qwen")), ("GET", newest_url("qwen"))]
    assert [group.label for group in outcome.groups][-2:] == ["most downloaded", "newest"]
    assert outcome.requests == 16
    assert outcome.budget_used == 18


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
    extra: dict = _account_pages(
        "unsloth",
        Response(status=200, body=json.dumps(unsloth_page).encode("utf-8")),
        json_response("hf_search_none.json"),
    )
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
    # seven accounts at two pages each (plus the two open pages in that case) and the one age
    # lookup the only resolved base model of these pages costs -- nothing more
    assert outcome.budget_used == (17 if page == "most downloaded" else 15)
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
    # One line per account, whatever it cost: both sort orders of the pinned answers carry the same
    # repositories, so the second page of an account is entirely `already listed`.
    outcome = run_search(build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert outcome.group_lines() == [
        "Qwen 2, 1 of them already listed",
        "deepseek-ai 0",
        "unsloth 6, 3 of them already listed",
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

    assert log["schema_version"] == 3
    assert log["mode"] == "word"
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
            _account_pages(
                "unsloth",
                json_response("hf_search_qwen_page_full.json"),
                json_response("hf_search_none.json"),
            )
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
            _account_pages(
                "unsloth",
                Response(status=200, body=json.dumps(full[:19]).encode("utf-8")),
                json_response("hf_search_none.json"),
            )
        )
    )

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    unsloth = next(group for group in outcome.groups if group.label == "unsloth")
    assert unsloth.page_full is False
    assert unsloth.line() == "unsloth 19"


def _no_family_mapping(open_pages: bool = False) -> dict:
    empty = json_response("hf_search_none.json")
    mapping: dict = {}
    for account in DEFAULT_PACKAGERS:
        mapping.update(_account_pages(account, empty, word="nebula"))
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
    mapping: dict = {}
    for account, name in SEARCH_ACCOUNT_PAGES.items():
        mapping.update(_account_pages(account, json_response(name)))
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
        _search_mapping(_account_pages("mradermacher", Response(status=503, body=b"")))
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
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(16), open_pages=True)

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
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(3))

    # `Qwen`'s two pages and the age lookup its resolved hit costs used all three requests, so the
    # second account is where the budget ends -- and six of the seven accounts have no answer
    assert str(exc.value) == (
        "search for 'qwen' (deepseek-ai) not made: budget exhausted; 6 accounts were not asked: "
        "deepseek-ai, unsloth, bartowski, mradermacher, lmstudio-community, ggml-org"
    )


# --- a typed repository id (B, decided 2026-09-25) ----------------------------------------------


def test_a_typed_repository_id_is_one_request_and_one_group_labeled_typed():
    transport = build_transport(typed_transport_mapping())

    outcome = run_search(transport, TYPED_GGUF_REPO, catalog=_catalog(), budget=RequestBudget(60))

    assert outcome.mode == "id"
    assert outcome.typed == TYPED_GGUF_REPO
    assert [group.label for group in outcome.groups] == ["typed"]
    assert [hit.repo for hit in outcome.hits] == [TYPED_GGUF_REPO]
    # one page, plus the one age lookup its resolved base model costs
    assert outcome.requests == 1
    assert outcome.budget_used == 2
    assert transport.calls[0] == ("GET", model_url(TYPED_GGUF_REPO))


def test_a_typed_repository_id_resolves_and_says_which_model_it_packages():
    outcome = run_search(
        build_transport(typed_transport_mapping()), TYPED_GGUF_REPO, catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, TYPED_GGUF_REPO)
    assert hit.resolved is True
    assert hit.resolved_base_model == "Qwen/Qwen3.5-9B"
    assert hit.publisher_status == "listed packager"
    assert hit.downloads == 1_443_818


def test_a_typed_id_resolves_although_no_catalog_publisher_owns_its_base_model():
    # `publisher_unknown` does not apply to a repository somebody typed: they said which one they
    # want, so the catalog has nothing left to prefer (decided 2026-09-25).
    repo = "packager/Nebula-9B-GGUF"
    answer = Response(
        status=200,
        body=json.dumps(
            {
                "id": repo,
                "tags": ["gguf", "base_model:nebula-lab/Nebula-9B", "base_model:quantized:nebula-lab/Nebula-9B"],
                "cardData": {"base_model": ["nebula-lab/Nebula-9B"], "license": "apache-2.0"},
                "downloads": 42,
            }
        ).encode("utf-8"),
    )
    mapping = {
        ("GET", model_url(repo)): answer,
        ("GET", "https://huggingface.co/api/models/nebula-lab/Nebula-9B"): Response(
            status=200, body=json.dumps({"id": "nebula-lab/Nebula-9B", "cardData": {}}).encode("utf-8")
        ),
    }

    outcome = run_search(build_transport(mapping), repo, catalog=_catalog(), budget=RequestBudget(60))

    hit = _hit(outcome, repo)
    assert (hit.resolved, hit.unresolved_reason) == (True, None)
    assert hit.resolved_base_model == "nebula-lab/Nebula-9B"
    assert hit.publisher_status == "other"
    # No catalog statement and no `new_version`: nothing is stated. Alone in its family, the model is
    # the highest version of it this search knows, so the list computes `latest` (decided 2026-09-25).
    assert (hit.age, hit.successor, hit.release_basis) == ("latest", None, "computed")


def test_a_typed_id_without_a_gguf_file_says_so_and_searches_for_the_name():
    mapping = {**search_transport_mapping(), **typed_transport_mapping()}
    transport = build_transport(mapping)

    outcome = run_search(transport, TYPED_NO_GGUF_REPO, catalog=_catalog(), budget=RequestBudget(60))

    assert f"{TYPED_NO_GGUF_REPO} holds no GGUF file" in outcome.notes
    assert outcome.mode == "word"
    assert outcome.typed is None
    # the word search over the name half followed: the accounts were asked with `qwen`
    assert ("GET", account_search_url("qwen", "unsloth")) in transport.calls
    assert TYPED_GGUF_REPO in [hit.repo for hit in outcome.hits]


def test_a_typed_id_that_fell_back_still_counts_the_page_it_asked():
    # It spent a request and left no group behind, so a `requests` built from the groups alone said
    # two for a run that made three (second-model round, 2026-09-25).
    mapping = {**search_transport_mapping(), **typed_transport_mapping()}

    outcome = run_search(
        build_transport(mapping), TYPED_NO_GGUF_REPO, catalog=_catalog(), budget=RequestBudget(60)
    )

    assert outcome.requests == sum(group.pages for group in outcome.groups) + 1
    # The typed page, then two pages for each of the six accounts the name half asks: `Qwen` as the
    # one matching publisher plus the five listed packagers.
    assert outcome.requests == 13


def test_a_typed_id_the_hub_does_not_answer_for_says_so_and_searches_for_the_name():
    repo = "unsloth/Qwen3.5-9B-GGUF"
    mapping = {**search_transport_mapping(), ("GET", model_url(repo)): Response(status=401, body=b"")}
    transport = build_transport(mapping)

    outcome = run_search(transport, repo, catalog=_catalog(), budget=RequestBudget(60))

    assert f"{repo} is no repository this search could read" in outcome.notes
    assert outcome.mode == "word"
    assert ("GET", account_search_url("qwen", "unsloth")) in transport.calls


def test_a_typed_id_answered_by_another_repository_is_not_taken_for_it():
    # The transport follows redirects and the Hub answers a moved path with what it moved to.
    repo = "unsloth/Qwen3.5-9B-GGUF"
    moved = Response(status=200, body=json.dumps({"id": "other/Qwen3.5-9B-GGUF", "tags": ["gguf"]}).encode("utf-8"))
    mapping = {**search_transport_mapping(), ("GET", model_url(repo)): moved}

    outcome = run_search(build_transport(mapping), repo, catalog=_catalog(), budget=RequestBudget(60))

    assert f"{repo} is no repository this search could read" in outcome.notes


# --- no search word at all (C, decided 2026-09-25) ----------------------------------------------


def test_no_search_word_asks_one_page_per_current_model_of_the_catalog():
    transport = build_transport(catalog_transport_mapping())

    outcome = run_search(transport, "", catalog=_catalog(), budget=RequestBudget(150))

    planned = latest_model_names(_catalog())
    assert outcome.mode == "catalog"
    assert [group.label for group in outcome.groups] == planned
    assert outcome.requests == len(planned)
    assert [call for call in transport.calls][0] == ("GET", catalog_page_url(planned[0]))
    assert outcome.notes == ["no search word: the catalog's current models, one page each"]


def test_no_search_word_resolves_what_those_pages_answer_with():
    outcome = run_search(
        build_transport(catalog_transport_mapping()), "  ", catalog=_catalog(), budget=RequestBudget(150)
    )

    hit = _hit(outcome, CATALOG_PAGE_REPO)
    assert hit.resolved is True
    assert hit.resolved_base_model == "Qwen/Qwen3.8-27B"
    assert hit.publisher_status == "listed packager"
    group = next(group for group in outcome.groups if group.label == CATALOG_PAGE_MODEL)
    assert group.page_size == 3


@pytest.mark.parametrize("spare, asked", [(1, 1), (3, 3)])
def test_the_catalog_pages_stop_at_the_reserve_instead_of_ending_the_search(spare, asked):
    """The fetch of a chosen model shares this budget, so the catalog plan stops and says how far it
    got -- a run that has a choice to offer is not ended by a `SearchError`. The threshold itself is
    the assertion: with `reserve + n` requests, `n` pages are asked and no more."""
    planned = len(latest_model_names(_catalog()))
    transport = build_transport(catalog_transport_mapping())

    outcome = run_search(
        transport, "", catalog=_catalog(), budget=RequestBudget(CATALOG_BUDGET_RESERVE + spare)
    )

    # The first pages of the catalog answer with nothing, so no age lookup blurs the count.
    assert len(outcome.groups) == asked
    assert outcome.requests == asked
    assert outcome.budget_used == asked
    assert f"the request budget ended after {asked} of {planned} catalog pages" in outcome.notes
    assert asked < planned


def test_the_search_log_of_a_run_without_a_word_names_the_catalog_mode():
    outcome = run_search(
        build_transport(catalog_transport_mapping()), "", catalog=_catalog(), budget=RequestBudget(150)
    )

    log = outcome.search_log(filtered=True, run_at=NOW, unresolved={})

    assert (log["mode"], log["word"]) == ("catalog", "")
    # Not `open`: a catalog page is no open list, and a sentence built on that class said the two
    # open lists were asked (live probe, 2026-09-25).
    assert {entry["class"] for entry in log["accounts"]} == {"catalog"}


def test_the_search_log_of_a_typed_id_names_that_page_as_typed():
    outcome = run_search(
        build_transport(typed_transport_mapping()), TYPED_GGUF_REPO, catalog=_catalog(), budget=RequestBudget(60)
    )

    log = outcome.search_log(filtered=True, run_at=NOW, unresolved={})

    assert (log["mode"], log["word"]) == ("id", TYPED_GGUF_REPO)
    assert log["accounts"] == [{"account": "typed", "class": "typed", "hits": 1, "page_full": False}]


# --- the owner filter as a preference (D, decided 2026-09-25) ------------------------------------


def test_with_the_filter_off_a_model_of_an_account_outside_the_catalog_resolves():
    outcome = run_search(
        build_transport(catalog_transport_mapping()),
        "",
        catalog=_catalog(),
        budget=RequestBudget(150),
        publisher_required=False,
    )

    hit = _hit(outcome, CATALOG_PAGE_FOREIGN_REPO)
    assert hit.resolved is True
    assert str(hit.resolved_base_model).startswith("DavidAU/")  # no publisher of this catalog
    assert hit.publisher_status == "other"
    assert hit.age == "unknown"
    assert "publisher_unknown" not in [hit.unresolved_reason for hit in outcome.hits]


def test_with_the_filter_on_that_same_model_stays_unresolved_with_its_reason():
    outcome = run_search(
        build_transport(catalog_transport_mapping()), "", catalog=_catalog(), budget=RequestBudget(150)
    )

    hit = _hit(outcome, CATALOG_PAGE_FOREIGN_REPO)
    assert (hit.resolved, hit.unresolved_reason) == (False, "publisher_unknown")


def test_the_most_downloaded_page_of_an_account_finds_what_its_newest_page_does_not():
    """Measured live 2026-09-25: `unsloth` has more than twenty `qwen` repositories newer than
    `Qwen3.5-9B-GGUF`, so one sort order alone cannot find the plain build of a listed model."""
    newest = json_response("hf_search_none.json")
    downloads = json_response("hf_search_qwen_unsloth_downloads.json")
    mapping = _search_mapping(_account_pages("unsloth", newest, downloads))
    mapping[("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-4B")] = Response(
        status=200, body=json.dumps({"id": "Qwen/Qwen3.5-4B", "cardData": {}}).encode("utf-8")
    )

    outcome = run_search(build_transport(mapping), "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert TYPED_GGUF_REPO in [hit.repo for hit in outcome.hits]
    unsloth = next(group for group in outcome.groups if group.label == "unsloth")
    assert unsloth.page_size == 2
    assert unsloth.pages == 2


# --- loose input, held against the answer locally (A, decided 2026-09-25) ------------------------


def test_a_name_with_blanks_asks_one_word_and_drops_what_the_others_do_not_name():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen 3.5 9b", catalog=_catalog(), budget=RequestBudget(60))

    # asked with `qwen`, the one word the Hub gets
    assert ("GET", account_search_url("qwen", "unsloth")) in transport.calls
    assert outcome.query == "qwen 3.5 9b"
    repos = [hit.repo for hit in outcome.hits]
    assert TYPED_GGUF_REPO in repos
    # `unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF` carries `qwen` but neither `3.5` nor `9b`
    assert "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF" not in repos
    unsloth = next(group for group in outcome.groups if group.label == "unsloth")
    assert unsloth.filtered_out > 0


def test_one_word_is_never_held_against_the_answer_so_the_pinned_run_is_unchanged():
    # `community-user/Nebula-9B-GGUF` carries the word `qwen` in neither half of its id, so a filter
    # wrongly applied to a one-word search would drop it -- which is the case this test is about.
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    assert all(group.filtered_out == 0 for group in outcome.groups)
    repos = [hit.repo for hit in outcome.hits]
    assert "community-user/Nebula-9B-GGUF" in repos
    assert "qwen" not in "community-user/Nebula-9B-GGUF".casefold()
    assert "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF" in repos


def test_a_typo_leads_to_the_candidates_the_catalog_knows():
    empty = json_response("hf_search_none.json")
    mapping: dict = {}
    for account in DEFAULT_PACKAGERS:
        mapping.update(_account_pages(account, empty, word="qwn"))

    outcome = run_search(build_transport(mapping), "qwn", catalog=_catalog(), budget=RequestBudget(60))

    assert "qwen" in outcome.candidates


def test_a_word_the_catalog_knows_gets_no_candidates():
    # A correct word with no answer today is a fact about the Hub, not a spelling mistake.
    outcome = run_search(build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert outcome.candidates == []


def test_a_typed_id_that_fell_back_can_still_lead_to_candidates():
    # From the fallback on it is a word search, so its words are read like any others.
    empty = json_response("hf_search_none.json")
    mapping: dict = {("GET", model_url("acme/qwn")): Response(status=404, body=b"")}
    for account in DEFAULT_PACKAGERS:
        mapping.update(_account_pages(account, empty, word="qwn"))

    outcome = run_search(build_transport(mapping), "acme/qwn", catalog=_catalog(), budget=RequestBudget(60))

    assert outcome.mode == "word"
    assert "qwen" in outcome.candidates


def test_a_typed_id_and_a_run_without_a_word_get_no_candidates():
    typed = run_search(
        build_transport(typed_transport_mapping()), TYPED_GGUF_REPO, catalog=_catalog(), budget=RequestBudget(60)
    )
    catalog_run = run_search(
        build_transport(catalog_transport_mapping()), "", catalog=_catalog(), budget=RequestBudget(150)
    )

    assert (typed.candidates, catalog_run.candidates) == ([], [])


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

    assert outcome.summary_line() == "4 repositories, 3 resolved, 1 unresolved, 14 requests, budget 16/60"


def test_the_summary_line_of_a_run_without_the_filter_counts_the_open_pages_too():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    assert outcome.summary_line() == "9 repositories, 3 resolved, 6 unresolved, 16 requests, budget 18/60"


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


# --- the computed release, at the recorded answers ------------------------------------------------

MISTRAL_SMALL_3 = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
MISTRAL_SMALL_4 = "mistralai/Mistral-Small-4-119B-2603"


def _mistral_outcome():
    transport = build_transport(mistral_search_mapping())
    return transport, run_search(transport, "mistral", catalog=_catalog(), budget=RequestBudget(60))


def _qwen_outcome():
    return run_search(build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60))


def test_mistral_small_3_2_is_computed_legacy_with_small_4_as_its_successor():
    """Decided 2026-09-25: the size is no rank, so the 24B model has the 119B one as successor."""
    _transport, outcome = _mistral_outcome()

    small_3 = _hit(outcome, "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF")
    small_4 = _hit(outcome, "unsloth/Mistral-Small-4-119B-2603-GGUF")
    # Both are only packaged by `unsloth` here: the release is the one of the base model the hit
    # resolved to, not of the repository it came from.
    assert (small_3.resolved_base_model, small_4.resolved_base_model) == (MISTRAL_SMALL_3, MISTRAL_SMALL_4)
    assert (small_3.age, small_3.successor, small_3.release_basis) == ("legacy", MISTRAL_SMALL_4, "computed")
    assert (small_4.age, small_4.successor, small_4.release_basis) == ("latest", None, "stated")


def test_every_release_a_statement_decided_says_stated():
    _transport, outcome = _mistral_outcome()

    ministral = _hit(outcome, "mistralai/Ministral-3-14B-Instruct-2512-GGUF")
    magistral = _hit(outcome, "mistralai/Magistral-Small-2509-GGUF")
    assert (ministral.age, ministral.release_basis) == ("latest", "stated")
    assert (magistral.age, magistral.successor, magistral.release_basis) == ("legacy", MISTRAL_SMALL_4, "stated")


def test_the_computed_release_asks_no_request_of_its_own():
    transport, outcome = _mistral_outcome()

    accounts = search_accounts(_catalog(), "mistral", DEFAULT_PACKAGERS)
    assert len(transport.calls) == outcome.budget_used == 2 * len(accounts) + len(MISTRAL_BASE_MODELS)


def test_qwen3_5_9b_is_computed_legacy_with_qwen3_8_27b_as_its_successor():
    """The nearest larger size among the highest versions of the family: the catalog's `Qwen3.8-27B`."""
    outcome = _qwen_outcome()

    for repo in ("unsloth/Qwen3.5-9B-GGUF", "Qwen/Qwen3.5-9B-GGUF"):
        hit = _hit(outcome, repo)
        assert (hit.age, hit.successor, hit.release_basis) == ("legacy", "Qwen/Qwen3.8-27B", "computed")


def test_the_deepseek_distill_alone_in_its_family_is_computed_latest():
    hit = _hit(_qwen_outcome(), "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF")

    assert (hit.age, hit.successor, hit.release_basis) == ("latest", None, "computed")


def test_the_list_marks_a_computed_release_with_a_star_at_the_recorded_answers():
    from modelroom.guided_context import Checked
    from modelroom.guided_models import model_choices

    _transport, mistral = _mistral_outcome()
    words = {
        model.name: model.release_text
        for outcome in (mistral, _qwen_outcome())
        for model in model_choices(outcome.hits, filtered=True, checked=Checked(reason="none"), context=8192)
    }

    assert words["Mistral-Small-3.2-24B-Instruct-2506"] == "legacy*"
    assert words["Mistral-Small-4-119B-2603"] == "latest"
    assert words["Magistral-Small-2509"] == "legacy"
    assert words["Qwen3.5-9B"] == "legacy*"
    assert words["DeepSeek-R1-0528-Qwen3-8B"] == "latest*"


def test_the_search_log_names_every_resolved_base_model_with_its_release():
    _transport, outcome = _mistral_outcome()

    log = outcome.search_log(filtered=True, run_at=NOW, unresolved={})

    assert log["schema_version"] == 3
    assert log["models"] == [
        {"base_model": "mistralai/Ministral-3-14B-Instruct-2512", "age": "latest", "successor": None,
         "release_basis": "stated"},
        {"base_model": "mistralai/Magistral-Small-2509", "age": "legacy", "successor": MISTRAL_SMALL_4,
         "release_basis": "stated"},
        {"base_model": MISTRAL_SMALL_4, "age": "latest", "successor": None, "release_basis": "stated"},
        {"base_model": MISTRAL_SMALL_3, "age": "legacy", "successor": MISTRAL_SMALL_4, "release_basis": "computed"},
    ]
    json.dumps(log)  # the file is JSON: every value of it has to be one


def test_the_search_log_names_no_model_of_a_repository_that_did_not_resolve():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60), open_pages=True
    )

    log = outcome.search_log(filtered=True, run_at=NOW, unresolved={})

    assert [entry["base_model"] for entry in log["models"]] == [
        "Qwen/Qwen3.5-9B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    ]


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

    write_configuration(path, config, now=NOW, expected_text=None)

    reread = load_config(path)
    assert reread.model_dump(mode="json") == config.model_dump(mode="json")
    assert tomllib.loads(path.read_text(encoding="utf-8"))["schema_version"] == 3


def test_write_configuration_leaves_out_a_context_no_guided_run_chose(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW, expected_text=None)

    assert "context" not in tomllib.loads(path.read_text(encoding="utf-8")).get("guided", {})
    assert load_config(path).guided.context is None


def test_write_configuration_keeps_the_context_a_guided_run_chose(tmp_path):
    config = _config(tmp_path)
    config = config.model_copy(update={"guided": config.guided.model_copy(update={"context": 4096})})
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW, expected_text=None)

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
    data = qwen35_example_config_dict(str(real_dir / "state"), str(real_dir / "models.md"))
    data["schema_version"] = 2
    data["families"] = []
    data["publishers"] = []
    config = Configuration.from_dict(data)
    write_configuration(real_path, config, now=NOW, expected_text=None)
    config = config.model_copy(update={"guided": config.guided.model_copy(update={"context": 4096})})

    write_configuration(link_path, config, now=NOW, expected_text=real_path.read_text(encoding="utf-8"))

    assert link_path.is_symlink(), "the link must still be a link, not a file of its own"
    assert load_config(real_path).model_dump(mode="json") == config.model_dump(mode="json")


def test_write_configuration_releases_the_lock_when_the_write_fails(tmp_path):
    # A guard that is only ever seen succeeding proves nothing about the `finally`.
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    path.mkdir()  # a directory where the file should go: os.replace cannot overwrite it

    with pytest.raises(OSError):
        write_configuration(path, config, now=NOW, expected_text=None)

    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    release_lock(handle)


def test_write_configuration_stops_at_another_processs_lock(tmp_path):
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()
    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    try:
        with pytest.raises(LockHeldError):
            write_configuration(path, config, now=NOW, expected_text=None)
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
        write_configuration(path, config, now=NOW, expected_text=None)

    assert not path.exists()


def test_write_configuration_releases_the_lock_afterwards(tmp_path):
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW, expected_text=None)

    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    release_lock(handle)
