"""AP9-KO P5 (sweep p5-b1): one search page that fails costs that page, never the search.

Measured live on 2026-10-14: the Hub answered single search pages with `500` and `504`, and two guided runs ended
at one of them with exit 2 -- the hits every other page had already brought were gone. The rule of P2 and P4 holds
here too: a realistic network or answer error (`search_pages.TRANSPORT_ERRORS`, a status other than `200`, an answer
that is no JSON or no list) hits only its own page. The page is named on the screen and in `search.json`; the other
pages are asked and listed as always. Only a search in which **no** page answered ends as before, with the
`SearchError` of the first page -- a search with no answer at all is no result. A budget that ends and a bug keep
their old ways.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelroom.cli import main
from modelroom.http import FixtureTransport, RequestBudget, Response
from modelroom.search import SearchError, run_search
from modelroom.search_pages import account_downloads_url, account_search_url

from fixture_support import build_transport, guided_transport_mapping, offline_daemon, offline_servers, search_transport_mapping, windows_probes
from test_guided import FULL_ANSWERS, QWEN_GGUF, RUN1, UNSLOTH, _pointer, _results, _run, _search_log
from test_search import ASKED_ACCOUNTS, _catalog

# Asked in this order: `Qwen` newest, `Qwen` most downloaded, `deepseek-ai` newest, ... -- so a failing
# `deepseek-ai` newest page sits between two pages that answer, the case of the live run.
FAILING = account_search_url("qwen", "deepseek-ai")
UNSLOTH_PAGE = account_search_url("qwen", "unsloth")


class _Failing(FixtureTransport):
    """The recorded answers, except that the pages in `errors` raise their exception instead of answering."""

    def __init__(self, mapping: dict, errors: dict[str, BaseException]) -> None:
        super().__init__(mapping)
        self._errors = errors

    def __call__(self, method, url, headers=None, *, max_bytes=None):
        if url in self._errors:
            self.calls.append((method, url))
            raise self._errors[url]
        return super().__call__(method, url, headers, max_bytes=max_bytes)


def _with(mapping: dict, **pages: Response) -> dict:
    return {**mapping, **{("GET", url): response for url, response in pages.items()}}


def _guided(tmp_path: Path, transport, answers: dict | None = None) -> tuple[int, list[str]]:
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "home").mkdir(exist_ok=True)
    return _run(tmp_path, answers, transport=transport)


def _notes(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines]


# --- the guided run: success, 503, success ---------------------------------------------------------------------------


def test_a_503_between_two_pages_that_answer_costs_only_that_page_and_the_run_ends_with_exit_0(tmp_path):
    transport = build_transport(_with(guided_transport_mapping(), **{FAILING: Response(status=503, body=b"")}))

    code, lines = _guided(tmp_path, transport)

    assert code == 0
    # Both pages around the failing one answered, and their hits reached the list and the configuration.
    assert ("GET", FAILING) in transport.calls
    assert ("GET", UNSLOTH_PAGE) in transport.calls
    written = (_results(tmp_path) / "modelroom.toml").read_text(encoding="utf-8")
    assert UNSLOTH in written and QWEN_GGUF in written
    # The page is named on the screen, in the words of the old error, and the search said it went on.
    assert "search for 'qwen' (deepseek-ai): unexpected status 503; the search went on without this page" in _notes(lines)
    log = _search_log(tmp_path)
    assert log["schema_version"] == 3
    assert log["failed_pages"] == [{"account": "deepseek-ai", "reason": "unexpected status 503"}]
    # The page still cost its request, and the account keeps its line: it was asked.
    assert log["requests"] == 14
    assert "deepseek-ai" in [entry["account"] for entry in log["accounts"]]


def test_a_search_without_a_failing_page_writes_no_failed_pages_key(tmp_path):
    code, _lines = _guided(tmp_path, build_transport(guided_transport_mapping()))

    assert code == 0
    assert "failed_pages" not in _search_log(tmp_path)


def test_a_publisher_whose_pages_failed_is_not_said_to_have_no_gguf(tmp_path):
    # `Qwen` answers neither page; "the publisher Qwen has no GGUF repository for this search" would be untrue.
    failing = {
        account_search_url("qwen", "Qwen"): Response(status=500, body=b""),
        account_downloads_url("qwen", "Qwen"): Response(status=504, body=b""),
    }
    transport = build_transport(_with(guided_transport_mapping(), **failing))

    # `Qwen`'s own GGUF repository came only from its pages, so only the `unsloth` one is offered.
    code, lines = _guided(tmp_path, transport, {**FULL_ANSWERS, "select": [UNSLOTH]})

    assert code == 0
    # `deepseek-ai` answered, with nothing: that one is a fact and stays in the sentence; `Qwen` is not in it.
    assert "searched Hugging Face at the five listed packagers; the publisher deepseek-ai has no GGUF repository for this search" in _notes(lines)
    assert [entry["account"] for entry in _search_log(tmp_path)["failed_pages"]] == ["Qwen", "Qwen"]


# --- the search itself: timeout, invalid JSON, every page failing ----------------------------------------------------


def test_a_page_that_times_out_is_left_out_and_the_next_pages_are_asked():
    transport = _Failing(search_transport_mapping(), {FAILING: TimeoutError("timed out")})

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert [group.label for group in outcome.groups] == ASKED_ACCOUNTS
    assert {"unsloth/Qwen3.5-9B-GGUF", "Qwen/Qwen3.5-9B-GGUF"} <= {hit.repo for hit in outcome.hits}
    assert "search for 'qwen' (deepseek-ai) failed: timed out; the search went on without this page" in outcome.notes
    assert outcome.failed_pages == [("deepseek-ai", "failed: timed out")]
    assert outcome.groups[1].failed_pages == 1
    assert outcome.requests == 14


@pytest.mark.parametrize(
    "body, reason",
    [
        (b"not json at all", "invalid JSON"),
        (b"{}", "the answer is not a list of repositories"),
        (b"[1, 2]", "the answer is not a list of repositories"),
    ],
)
def test_an_answer_that_does_not_read_is_left_out_and_the_rest_of_the_search_stands(body, reason):
    transport = build_transport(_with(search_transport_mapping(), **{account_search_url("qwen", "Qwen"): Response(status=200, body=body)}))

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    # `Qwen`'s second page carries the same repository, so its hit is still there; every later account was asked.
    assert "Qwen/Qwen3.5-9B-GGUF" in {hit.repo for hit in outcome.hits}
    assert ("GET", UNSLOTH_PAGE) in transport.calls
    assert [label for label, _ in outcome.failed_pages] == ["Qwen"]
    assert reason in outcome.failed_pages[0][1]


def test_when_every_page_fails_the_search_ends_with_the_error_of_the_first_page():
    def failing(method: str, url: str, headers=None, *, max_bytes=None):
        raise OSError("the name does not resolve")

    with pytest.raises(SearchError) as exc:
        run_search(failing, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert str(exc.value) == "search for 'qwen' (Qwen) failed: the name does not resolve"


def test_when_every_page_answers_503_the_guided_run_ends_with_exit_2(tmp_path, capsys):
    mapping = guided_transport_mapping()
    failing = {url: Response(status=503, body=b"") for (_method, url) in search_transport_mapping() if "filter=gguf" in url}
    answers = tmp_path / "answers.toml"
    body = [f"{key} = {json.dumps(value)}" for key, value in FULL_ANSWERS.items()]
    answers.write_text("schema_version = 1\n" + "\n".join(body) + "\n", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()

    code = main(
        ["--answers", str(answers)],
        transport=build_transport(_with(mapping, **failing)),
        probes=windows_probes(),
        pointer_path=_pointer(tmp_path),
        daemon=offline_daemon(),
        servers=offline_servers(),
        now=RUN1,
        here=_results(tmp_path),
        out=[].append,
    )

    assert code == 2
    assert "search for 'qwen' (Qwen): unexpected status 503" in capsys.readouterr().err


def test_a_budget_that_ends_after_a_failed_page_still_ends_the_search_with_its_own_words():
    transport = build_transport(_with(search_transport_mapping(), **{account_search_url("qwen", "Qwen"): Response(status=503, body=b"")}))

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(3))

    assert "not made: budget exhausted" in str(exc.value)


def test_a_bug_in_one_page_still_goes_up():
    transport = _Failing(search_transport_mapping(), {FAILING: AttributeError("a bug")})

    with pytest.raises(AttributeError):
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))
