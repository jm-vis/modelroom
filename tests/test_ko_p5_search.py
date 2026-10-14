"""AP9-KO P5, part 3: two leftovers of the search (the notes of P0 and P2).

1. `search.py` caught `except Exception` around its requests and its JSON reading; it now catches what the transport
   layer and a garbled answer can really end in (`search_pages.TRANSPORT_ERRORS`, `ANSWER_ERRORS`, the classes of P2)
   and lets a bug go up. The user-facing result of every expected failure stays what it was.
2. The requests of the vLLM tensor search are counted in the `budget.used` of `search.json`, as the requests of the
   direct find are: the log says what is left for the fetch.
"""

from __future__ import annotations

import http.client

import pytest

from modelroom.http import BudgetedTransport, RequestBudget, ResponseTooLargeError, TransportSecurityError
from modelroom.search import SearchError, _fetch_search_page, _typed_entry
from modelroom.search_pages import SearchRequest, account_search_url

from test_ko_p2_search import TENSOR_ONLY, FakeHub, _answer, _gguf_entry, _page, _search_log, _step

_REQUEST = SearchRequest(label="listed packager unsloth", url="https://huggingface.co/api/models?search=nova", limit=20, account="unsloth")

_EXPECTED = [
    OSError("connection reset"),
    http.client.IncompleteRead(b"[{", 10),
    http.client.BadStatusLine("garbage"),
    TransportSecurityError("a host that is not allowed"),
    ResponseTooLargeError(100),
    ValueError("a Location that cannot be parsed"),
]


def _failing(error: BaseException):
    def transport(method, url, headers=None, *, max_bytes=None):
        raise error

    return transport


class _Body:
    """A response whose body is no JSON."""

    status = 200

    def json(self):
        raise ValueError("not json")


# --- the expected failures stay what they were ----------------------------------------------------------------------


@pytest.mark.parametrize("error", _EXPECTED, ids=lambda error: type(error).__name__)
def test_a_request_that_did_not_go_through_is_a_search_error_naming_the_page(error):
    with pytest.raises(SearchError, match="failed"):
        _fetch_search_page(_failing(error), "nova", _REQUEST, [])


@pytest.mark.parametrize("error", _EXPECTED, ids=lambda error: type(error).__name__)
def test_a_typed_id_whose_request_did_not_go_through_falls_back_to_the_word_search(error):
    assert _typed_entry(_failing(error), "acme/Nova-9B") is None


def test_an_answer_that_is_no_json_is_a_search_error_for_a_page_and_a_fallback_for_a_typed_id():
    def transport(method, url, headers=None, *, max_bytes=None):
        return _Body()

    with pytest.raises(SearchError, match="invalid JSON"):
        _fetch_search_page(transport, "nova", _REQUEST, [])
    assert _typed_entry(transport, "acme/Nova-9B") is None


# --- a bug is not an expected failure -------------------------------------------------------------------------------


@pytest.mark.parametrize("bug", [AttributeError("a bug"), KeyError("a bug"), TypeError("a bug")], ids=lambda bug: type(bug).__name__)
def test_a_programming_error_in_a_request_goes_up_from_a_page_and_from_a_typed_id(bug):
    with pytest.raises(type(bug)):
        _fetch_search_page(_failing(bug), "nova", _REQUEST, [])
    with pytest.raises(type(bug)):
        _typed_entry(_failing(bug), "acme/Nova-9B")


def test_a_programming_error_in_reading_an_answer_goes_up():
    class Broken:
        status = 200

        def json(self):
            raise AttributeError("a bug in the reader")

    def transport(method, url, headers=None, *, max_bytes=None):
        return Broken()

    with pytest.raises(AttributeError):
        _fetch_search_page(transport, "nova", _REQUEST, [])
    with pytest.raises(AttributeError):
        _typed_entry(transport, "acme/Nova-9B")


def test_an_ended_budget_still_ends_the_search_with_its_own_words():
    def never(method, url, headers=None, *, max_bytes=None):
        raise AssertionError("not reached")

    spent = BudgetedTransport(never, RequestBudget(0))

    with pytest.raises(SearchError, match="not made"):
        _fetch_search_page(spent, "nova", _REQUEST, [])
    with pytest.raises(SearchError, match="not made"):
        _typed_entry(spent, "acme/Nova-9B")


# --- the tensor requests are in the log's budget --------------------------------------------------------------------


def test_the_requests_of_the_tensor_search_are_counted_in_the_budget_the_search_log_states(tmp_path, monkeypatch):
    pages = {account_search_url("nova", "acme"): _page([_gguf_entry("acme/Nova-9B-GGUF", TENSOR_ONLY)])}
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)}, pages)

    step, run, config, _lines = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=True)

    asked_for_tensors = [url for url in hub.calls if "filter=safetensors" in url]
    assert asked_for_tensors  # the search did ask the tensor accounts about the model the GGUF hit resolved to
    assert _search_log(config)["budget"]["used"] == run.budget.used == len(hub.calls)


def test_a_run_without_a_vllm_machine_books_no_tensor_request(tmp_path, monkeypatch):
    pages = {account_search_url("nova", "acme"): _page([_gguf_entry("acme/Nova-9B-GGUF", TENSOR_ONLY)])}
    hub = FakeHub({TENSOR_ONLY: _answer(TENSOR_ONLY)}, pages)

    _step_, run, config, _lines = _step(tmp_path, monkeypatch, hub, {"search": "nova-9b"}, vllm=False)

    assert not [url for url in hub.calls if "filter=safetensors" in url]
    assert _search_log(config)["budget"]["used"] == run.budget.used == len(hub.calls)
