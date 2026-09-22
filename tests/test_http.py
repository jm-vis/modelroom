"""Tests for modelroom.http: the Transport protocol, FixtureTransport and the request budget.

No test here ever touches the network: UrllibTransport's use of urllib.request is exercised
indirectly (its request-shaping logic) without opening a real socket, by constructing it and
checking the request object it would send is never enough on its own -- so the real transport
is covered only by construction/attribute tests here, while every fetcher test in
test_hf.py/test_ollama.py drives the same Transport protocol through FixtureTransport.
"""

from __future__ import annotations

import json

import pytest

import modelroom
from modelroom.http import (
    USER_AGENT,
    BudgetedTransport,
    BudgetExhaustedError,
    FixtureTransport,
    Response,
    UrllibTransport,
)


def test_user_agent_carries_the_package_version():
    assert USER_AGENT == f"modelroom/{modelroom.__version__}"


def test_response_json_decodes_the_body():
    response = Response(status=200, headers={}, body=json.dumps({"a": 1}).encode("utf-8"))
    assert response.json() == {"a": 1}


def test_response_header_lookup_is_case_insensitive():
    response = Response(status=200, headers={"Content-Type": "application/json"}, body=b"{}")
    assert response.header("content-type") == "application/json"
    assert response.header("Content-Type") == "application/json"
    assert response.header("missing") is None


def test_urllib_transport_constructs_with_a_default_timeout():
    transport = UrllibTransport()
    assert transport._timeout > 0


# --- FixtureTransport -----------------------------------------------------------------------


def test_fixture_transport_returns_the_recorded_response_and_records_the_call():
    response = Response(status=200, headers={}, body=b"{}")
    transport = FixtureTransport({("GET", "https://example.test/a"): response})

    result = transport("GET", "https://example.test/a")

    assert result is response
    assert transport.calls == [("GET", "https://example.test/a")]


def test_fixture_transport_raises_on_an_unrecorded_call():
    transport = FixtureTransport({})
    with pytest.raises(KeyError):
        transport("GET", "https://example.test/missing")


def test_fixture_transport_records_every_call_including_repeats():
    response = Response(status=200, headers={}, body=b"{}")
    transport = FixtureTransport({("GET", "https://example.test/a"): response})

    transport("GET", "https://example.test/a")
    transport("GET", "https://example.test/a")

    assert transport.calls == [
        ("GET", "https://example.test/a"),
        ("GET", "https://example.test/a"),
    ]


# --- BudgetedTransport ------------------------------------------------------------------------


def test_budgeted_transport_passes_calls_through_under_budget():
    response = Response(status=200, headers={}, body=b"{}")
    inner = FixtureTransport({("GET", "https://example.test/a"): response})
    budgeted = BudgetedTransport(inner, budget=5)

    result = budgeted("GET", "https://example.test/a")

    assert result is response
    assert budgeted.used == 1
    assert budgeted.remaining == 4


def test_budgeted_transport_raises_once_the_budget_is_exhausted():
    response = Response(status=200, headers={}, body=b"{}")
    inner = FixtureTransport({("GET", "https://example.test/a"): response})
    budgeted = BudgetedTransport(inner, budget=1)

    budgeted("GET", "https://example.test/a")
    with pytest.raises(BudgetExhaustedError, match="budget exhausted"):
        budgeted("GET", "https://example.test/a")

    # the exhausted call never reached the inner transport
    assert inner.calls == [("GET", "https://example.test/a")]


def test_budgeted_transport_default_budget_is_400():
    inner = FixtureTransport({})
    budgeted = BudgetedTransport(inner)
    assert budgeted.remaining == 400


# --- F10: a response that followed redirects counts every hop against the budget -----------


def test_budgeted_transport_counts_every_redirect_hop_reported_by_the_response():
    response = Response(status=200, headers={}, body=b"{}", requests_made=3)
    inner = FixtureTransport({("GET", "https://example.test/a"): response})
    budgeted = BudgetedTransport(inner, budget=10)

    result = budgeted("GET", "https://example.test/a")

    assert result is response
    assert budgeted.used == 3
    assert budgeted.remaining == 7


def test_response_requests_made_defaults_to_one():
    response = Response(status=200, headers={}, body=b"{}")
    assert response.requests_made == 1
