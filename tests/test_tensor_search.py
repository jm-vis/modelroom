"""Tests for modelroom.tensor_search: finding vLLM tensor repositories of a resolved base model
(AP9-SV2-P, P1).

Synthetic transports throughout: this module only asks a narrow, newly-invented page shape
(`filter=safetensors`), so there is no real recorded answer to hold it against yet -- the same
reasoning `test_hf.py`'s synthetic defects use, stated once here (AGENTS.md, "Test first").
"""

from __future__ import annotations

import json

from modelroom.http import BudgetedTransport, FixtureTransport, RequestBudget, Response
from modelroom.tensor_search import (
    MAX_TENSOR_REPOS_PER_MODEL,
    TENSOR_PACKAGERS,
    search_tensor_repos,
    tensor_packagers,
    tensor_search_url,
)

BASE = "acme/Nova-7B"


def _hit(repo: str, *, quantized: bool = True, downloads: int = 0) -> dict:
    tags = [f"base_model:{BASE}"]
    if quantized:
        tags.append(f"base_model:quantized:{BASE}")
    return {"id": repo, "tags": tags, "cardData": None, "downloads": downloads}


def _page(hits: list[dict]) -> Response:
    return Response(status=200, headers={}, body=json.dumps(hits).encode("utf-8"))


def test_tensor_packagers_is_the_owner_then_the_listed_packagers_each_once():
    assert tensor_packagers(BASE, ("RedHatAI", "acme")) == ["acme", "RedHatAI"]


def test_tensor_search_url_is_pinned():
    url = tensor_search_url("Nova-7B", "RedHatAI")
    assert "author=RedHatAI" in url
    assert "search=Nova-7B" in url
    assert "filter=safetensors" in url
    assert "sort=downloads" in url
    assert "direction=-1" in url


def test_an_account_with_one_quantized_hit_is_accepted():
    transport = FixtureTransport({("GET", tensor_search_url("Nova-7B", "acme")): _page([_hit("acme/Nova-7B-FP8")])})

    outcome = search_tensor_repos(transport, RequestBudget(10), BASE, packagers=())

    assert outcome.repos == ("acme/Nova-7B-FP8",)
    assert outcome.skipped == 0
    assert outcome.requests == 1
    assert outcome.note() is None


def test_an_unrelated_hit_is_not_accepted():
    transport = FixtureTransport(
        {("GET", tensor_search_url("Nova-7B", "acme")): _page([_hit("acme/Nova-7B-FP8", quantized=False)])}
    )

    outcome = search_tensor_repos(transport, RequestBudget(10), BASE, packagers=())

    assert outcome.repos == ()


def test_tensor_search_takes_at_most_four_repos_per_model():
    hits = [_hit(f"RedHatAI/Nova-7B-Q{i}", downloads=100 - i) for i in range(6)]
    transport = FixtureTransport(
        {
            ("GET", tensor_search_url("Nova-7B", BASE.split("/", 1)[0])): _page([]),
            ("GET", tensor_search_url("Nova-7B", "RedHatAI")): _page(hits),
        }
    )

    outcome = search_tensor_repos(transport, RequestBudget(10), BASE)

    assert len(outcome.repos) == MAX_TENSOR_REPOS_PER_MODEL == 4
    # Highest downloads first (RedHatAI/Nova-7B-Q0 has 100, the most).
    assert outcome.repos[0] == "RedHatAI/Nova-7B-Q0"
    assert outcome.skipped == 2
    assert outcome.note() == "and 2 more vLLM builds not fetched"


def test_the_budget_counts_every_tensor_request():
    transport = FixtureTransport(
        {
            ("GET", tensor_search_url("Nova-7B", "acme")): _page([]),
            ("GET", tensor_search_url("Nova-7B", "RedHatAI")): _page([]),
        }
    )
    budget = RequestBudget(10)

    outcome = search_tensor_repos(transport, budget, BASE)

    assert outcome.requests == 2
    assert budget.used == 2


def test_a_budget_exhausted_mid_search_ends_it_with_what_was_already_found():
    transport = FixtureTransport(
        {
            ("GET", tensor_search_url("Nova-7B", "acme")): _page([_hit("acme/Nova-7B-FP8")]),
            ("GET", tensor_search_url("Nova-7B", "RedHatAI")): _page([_hit("RedHatAI/Nova-7B-AWQ")]),
        }
    )
    budget = RequestBudget(1)

    outcome = search_tensor_repos(transport, budget, BASE)

    assert outcome.repos == ("acme/Nova-7B-FP8",)
    assert outcome.requests == 1


def test_an_account_that_does_not_answer_200_contributes_nothing():
    transport = FixtureTransport(
        {
            ("GET", tensor_search_url("Nova-7B", "acme")): Response(status=500, headers={}, body=b"error"),
        }
    )

    outcome = search_tensor_repos(transport, RequestBudget(10), BASE, packagers=())

    assert outcome.repos == ()
    assert outcome.requests == 1


def test_default_tensor_packagers_is_redhatai():
    assert TENSOR_PACKAGERS == ("RedHatAI",)
