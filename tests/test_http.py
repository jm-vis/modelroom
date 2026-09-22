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
    RedirectingTransport,
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


# --- R5: RedirectingTransport follows a chain by calling `inner` once per hop --------------
#
# R5 inverts F10's composition: UrllibTransport makes exactly one request per call and returns
# a 3xx as an ordinary Response; RedirectingTransport(inner) is what follows the chain, calling
# `inner` once per hop, so the run's request budget (BudgetedTransport) can be composed as the
# *inner* of a RedirectingTransport and see -- and book -- every hop before it is made, failure
# paths included. Response.requests_made and BudgetedTransport's post-call accounting are gone.


def test_redirecting_transport_follows_a_single_redirect_and_costs_two_requests():
    inner = FixtureTransport(
        {
            ("GET", "https://example.test/a"): Response(status=302, headers={"Location": "https://example.test/b"}, body=b""),
            ("GET", "https://example.test/b"): Response(status=200, headers={}, body=b'{"ok": true}'),
        }
    )
    budgeted = BudgetedTransport(inner, budget=10)
    redirecting = RedirectingTransport(budgeted)

    result = redirecting("GET", "https://example.test/a")

    assert result.status == 200
    assert result.json() == {"ok": True}
    assert budgeted.used == 2
    assert budgeted.remaining == 8


def test_redirecting_transport_a_six_hop_chain_raises_and_books_only_five_requests():
    # R5: a chain needing 6 requests to resolve exceeds MAX_REDIRECTS (5) -- exactly 5 requests
    # are made (booked against the budget) and the would-be sixth is never made.
    mapping = {}
    for i in range(1, 6):
        mapping[("GET", f"https://example.test/{i}")] = Response(
            status=302, headers={"Location": f"https://example.test/{i + 1}"}, body=b""
        )
    # deliberately no entry for https://example.test/6 -- it must never be requested
    inner = FixtureTransport(mapping)
    budgeted = BudgetedTransport(inner, budget=10)
    redirecting = RedirectingTransport(budgeted)

    with pytest.raises(RuntimeError, match="redirect"):
        redirecting("GET", "https://example.test/1")

    assert budgeted.used == 5
    assert ("GET", "https://example.test/6") not in inner.calls


def test_redirecting_transport_budget_of_one_raises_budget_exhausted_on_the_second_hop():
    inner = FixtureTransport(
        {
            ("GET", "https://example.test/a"): Response(status=302, headers={"Location": "https://example.test/b"}, body=b""),
            ("GET", "https://example.test/b"): Response(status=200, headers={}, body=b"{}"),
        }
    )
    budgeted = BudgetedTransport(inner, budget=1)
    redirecting = RedirectingTransport(budgeted)

    with pytest.raises(BudgetExhaustedError):
        redirecting("GET", "https://example.test/a")

    assert inner.calls == [("GET", "https://example.test/a")]  # the second hop was never made


def test_redirecting_transport_never_follows_a_redirect_for_a_non_get_head_method():
    inner = FixtureTransport(
        {("POST", "https://example.test/a"): Response(status=302, headers={"Location": "https://example.test/b"}, body=b"")}
    )
    redirecting = RedirectingTransport(inner)

    result = redirecting("POST", "https://example.test/a")

    assert result.status == 302
    assert inner.calls == [("POST", "https://example.test/a")]


def test_redirecting_transport_a_redirect_status_without_a_location_header_is_returned_as_is():
    inner = FixtureTransport({("GET", "https://example.test/a"): Response(status=302, headers={}, body=b"")})
    redirecting = RedirectingTransport(inner)

    result = redirecting("GET", "https://example.test/a")

    assert result.status == 302
