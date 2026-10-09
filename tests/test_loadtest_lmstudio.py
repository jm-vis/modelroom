"""Tests for modelroom.loadtest_lmstudio: the measurement of an LM Studio server (AP9-LS-M).

One request after the other, timed on the wall: a read of the list, the warm-up, a read, three runs each
followed by a read. The tests run against `FixtureLmStudioServer` with the recordings of
`tests/fixtures/lmstudio_*.json` and an injected clock; nothing touches a network and nothing is written. They
follow the "Vertragstests LS-M" of the measurement protocol v2 contract (section 4), numbered as it numbers them.
"""

from __future__ import annotations

import ast
import copy
import json
import urllib.error
from pathlib import Path

import pytest

from modelroom.http import Response
from modelroom.lmstudio_server import (
    BODY_HEADER,
    BODY_OVERSIZED,
    BODY_UNREAD,
    CHAT_PATH,
    MODELS_PATH,
    LmStudioServerError,
    LmStudioTimeoutError,
    LoadedModel,
)
from modelroom.loadtest_lmstudio import MAX_TOKENS, PROBLEM_WORDS, chat_body, run_lmstudio_load_test
from modelroom.measurements import Scenario

from fixture_http import FIXTURES
from fixture_lmstudio import FakeAnswer, chat_calls, chat_to, http_error, list_calls, lmstudio_servers, scripted_server

PACKAGE = Path(__file__).resolve().parent.parent / "modelroom"
SCENARIO = Scenario(context_requested=8192, context_origin="default", kv_type="f16", kv_type_assumed=True, requests=1)
PROMPT = "Describe how a public library shelves a new book."
DISTINCT = "lmstudio_v1_models_loaded_distinct_key.json"
INSTANCE = LoadedModel("qwen3.5-4b:2", "qwen3.5-4b", 8192, 4)  # the id is not the key
SECOND = 1_000_000_000
LINES = {  # the table of the contract, the wording of the outputs
    1: "stopped answering during the measurement",
    2: "the loaded model changed during the measurement",
    3: "stopped answering during the measurement",
    4: "a request did not answer within 600 s",
    5: "failed to load the model for the request; nothing measured",
    6: "a request did not answer with status 200",
    7: "a request carried no readable finish_reason and usage",
    8: "a request did not end with finish_reason length",
    9: "a request did not carry 128 tokens",
    10: "a request did not fit the context 8192",
    11: "a request took no measurable time",
}


def _recorded(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _clock(*walls: int, log: list | None = None):
    """A clock whose k-th pair of calls (before and after a chat) is `walls[k]` nanoseconds apart, with a long
    stretch between two pairs, so a time that spanned a read or the warm-up would show in the number."""
    state = {"now": 1_000, "open": False, "index": 0}

    def clock() -> int:
        if log is not None:
            log.append("tick")
        if not state["open"]:
            state["open"] = True
            return state["now"]
        state["open"] = False
        state["now"] += walls[min(state["index"], len(walls) - 1)]
        state["index"] += 1
        stop = state["now"]
        state["now"] += 90 * SECOND
        return stop

    return clock


def _measure(server, *walls: int, instance: LoadedModel = INSTANCE, log: list | None = None):
    return run_lmstudio_load_test(server, instance, SCENARIO, PROMPT, _clock(*(walls or (SECOND,)), log=log))


def _server(chats: list, lists: list | None = None):
    return lmstudio_servers(lists or [DISTINCT], chats=chats)


def _good(count: int = 4) -> list:
    return [chat_to(INSTANCE.instance_id)] * count


def _chat_with(**usage) -> dict:
    """The recorded chat with fields of `usage` replaced (a key set to `None` is removed)."""
    payload = _recorded("lmstudio_chat_one.json")
    for key, value in usage.items():
        if value is None:
            payload["usage"].pop(key, None)
        else:
            payload["usage"][key] = value
    return payload


# --- (1) the success: four chats, five reads, in order, the number from the three runs ---------------------


def test_the_success_is_four_chats_and_five_reads_in_order_and_the_number_comes_from_the_three_runs():
    server = _server(_good())

    measurement = _measure(server, 100 * SECOND, 2 * SECOND, 4 * SECOND, SECOND)

    assert measurement.problem is None and measurement.calls == 9
    assert [call[:2] for call in server.calls] == [("GET", MODELS_PATH), ("POST", CHAT_PATH)] * 4 + [("GET", MODELS_PATH)]
    # the warm-up (100 s) is not in it: 128/2, 128/4, 128/1 are 64, 32 and 128 tok/s
    assert (measurement.request_min, measurement.request_max) == (32.0, 128.0)
    assert measurement.request_tps == pytest.approx((64 + 32 + 128) / 3)
    assert (measurement.instance_id, measurement.context) == ("qwen3.5-4b:2", 8192)


def test_the_chat_names_the_instance_id_and_not_the_key_with_the_agreed_body():
    server = _server(_good())

    _measure(server)

    bodies = [call[2] for call in server.calls if call[0] == "POST"]
    assert len(bodies) == 4 and all(body == chat_body("qwen3.5-4b:2", PROMPT) for body in bodies)
    assert bodies[0] == {
        "model": "qwen3.5-4b:2", "max_tokens": 128, "temperature": 0.0, "seed": 42, "stream": False,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    assert "qwen3.5-4b" != bodies[0]["model"]  # the key of the model would have been refused by the fixture


def test_a_chat_that_names_the_key_is_refused_by_the_fixture_server():
    """The check of the fixture server is no empty one: it knows the instance id, and a chat that names the key
    of the model gets the 404 of a model that is not loaded."""
    server = lmstudio_servers([DISTINCT], chats=[chat_to(INSTANCE.instance_id)] * 2)

    right = server("POST", CHAT_PATH, chat_body(INSTANCE.instance_id, PROMPT))
    wrong = server("POST", CHAT_PATH, chat_body(INSTANCE.key, PROMPT))

    assert (right.status, wrong.status) == (200, 404)


# --- (2) each of the five reads: instance missing, other id, other context, read fails ------------------------


def _list_variant(kind: str):
    if kind == "missing":
        return "lmstudio_v1_models_empty.json"
    if kind == "other id":
        return "lmstudio_v1_models_loaded.json"  # the instance there is `qwen3.5-4b`, not `qwen3.5-4b:2`
    if kind == "other context":
        payload = _recorded(DISTINCT)
        payload["models"][0]["loaded_instances"][0]["config"]["context_length"] = 4096
        return payload
    return LmStudioServerError("refused")


@pytest.mark.parametrize("position", range(5))
@pytest.mark.parametrize(
    ("kind", "row"), [("missing", 2), ("other id", 2), ("other context", 2), ("read fails", 1)]
)
def test_a_read_that_differs_or_fails_ends_the_measurement_without_a_further_chat(position, kind, row):
    lists = [DISTINCT] * 5
    lists[position] = _list_variant(kind)
    server = _server(_good(), lists)

    measurement = _measure(server)

    assert measurement.problem == LINES[row]
    assert (measurement.request_tps, measurement.request_min, measurement.request_max) == (None, None, None)
    assert len(chat_calls(server)) == position  # the reads stand after 0, 1, 2, 3 and 4 chats
    assert len(list_calls(server)) == position + 1 and measurement.calls == 2 * position + 1


# --- (3) an invalid warm-up or run: no repeat, no part of a number ---------------------------------------------


@pytest.mark.parametrize("index", range(4), ids=["warm-up", "run 1", "run 2", "run 3"])
def test_an_invalid_warm_up_or_run_ends_it_without_a_repeat_and_without_a_part_of_a_number(index):
    chats = _good()
    chats[index] = "lmstudio_chat_finish_stop.json"
    server = _server(chats)

    measurement = _measure(server)

    assert measurement.problem == LINES[8]
    assert (measurement.request_tps, measurement.request_min, measurement.request_max) == (None, None, None)
    assert len(chat_calls(server)) == index + 1 and measurement.calls == 2 * index + 2


# --- (4) the table, rows 3 to 11, in the words of the contract ------------------------------------------------


def _ok_reply() -> str:
    return "lmstudio_chat_one.json"


def _single(reply, *walls: int):
    """A measurement whose warm-up gets `reply` and ends on it, so the row is the row of that one answer."""
    server = _server([reply])
    return server, _measure(server, *walls)


def test_the_table_has_eleven_rows_in_the_words_of_the_contract():
    assert {row: text.format(context=8192) for row, text in PROBLEM_WORDS.items()} == LINES


def test_row_3_a_chat_without_an_answer():
    server, measurement = _single(LmStudioServerError("gone"))

    assert measurement.problem == LINES[3] and len(chat_calls(server)) == 1


def test_row_4_a_chat_that_reaches_its_limit_of_time():
    server, measurement = _single(LmStudioTimeoutError("limit"))

    assert measurement.problem == LINES[4] and len(chat_calls(server)) == 1


def test_row_4_a_valid_answer_after_more_than_600_s_is_no_number():
    """The clock reads 601 s for the chat that came back whole: the limit hangs on the wall time too."""
    server = _server(_good())

    measurement = _measure(server, 601 * SECOND)

    assert measurement.problem == LINES[4] and len(chat_calls(server)) == 1


def test_a_valid_answer_after_exactly_600_s_counts():
    measurement = _measure(_server(_good()), 600 * SECOND)

    assert measurement.problem is None and measurement.request_tps == pytest.approx(128 / 600)


def test_row_5_a_failed_load_of_the_model():
    _server_, measurement = _single(Response(status=400, body=(FIXTURES / "lmstudio_jit_on_unloaded_chat.json").read_bytes()))

    assert measurement.problem == LINES[5]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_row_6_every_other_status(status):
    _server_, measurement = _single(Response(status=status, body=b'{"error": {"message": "no"}}'))

    assert measurement.problem == LINES[6]


@pytest.mark.parametrize("marker", [BODY_UNREAD, BODY_OVERSIZED])
def test_row_6_a_wrong_status_with_a_body_that_is_not_there_is_row_6_not_row_7(marker):
    _server_, measurement = _single(Response(status=500, headers={BODY_HEADER: marker}))

    assert measurement.problem == LINES[6]


@pytest.mark.parametrize("marker", [BODY_UNREAD, BODY_OVERSIZED])
def test_row_7_a_status_200_with_a_body_that_is_not_there(marker):
    _server_, measurement = _single(Response(status=200, headers={BODY_HEADER: marker}))

    assert measurement.problem == LINES[7]


@pytest.mark.parametrize(
    "reply",
    [
        Response(status=200, body=b"not json"),
        Response(status=200, body=b"[]"),
        Response(status=200, body=b'{"choices": []}'),
        "lmstudio_chat_no_usage.json",
        "lmstudio_chat_prompt_negative.json",
        _chat_with(completion_tokens=-1),
        _chat_with(prompt_tokens=True),
        _chat_with(prompt_tokens=12.0),
        _chat_with(prompt_tokens="12"),
        _chat_with(prompt_tokens=None),
        _chat_with(completion_tokens=True),
        _chat_with(completion_tokens=128.0),
        _chat_with(completion_tokens="128"),
        _chat_with(completion_tokens=None),
    ],
    ids=[
        "no json", "a list", "no choice", "no usage", "prompt -1", "completion -1", "prompt true", "prompt 12.0",
        "prompt '12'", "no prompt", "completion true", "completion 128.0", "completion '128'", "no completion",
    ],
)
def test_row_7_a_body_without_readable_fields_and_counters_that_are_no_whole_numbers(reply):
    _server_, measurement = _single(reply)

    assert measurement.problem == LINES[7]


def test_row_7_finish_reason_must_be_a_string():
    payload = _recorded("lmstudio_chat_one.json")
    payload["choices"][0]["finish_reason"] = 7

    _server_, measurement = _single(payload)

    assert measurement.problem == LINES[7]


def test_row_8_a_run_that_did_not_end_on_length():
    _server_, measurement = _single("lmstudio_chat_finish_stop.json")

    assert measurement.problem == LINES[8]


def test_row_9_a_run_with_other_than_128_tokens():
    _server_, measurement = _single("lmstudio_chat_tokens_127.json")

    assert measurement.problem == LINES[9]


def test_row_10_a_run_that_does_not_fit_the_context():
    _server_, measurement = _single("lmstudio_chat_prompt_large.json")

    assert measurement.problem == LINES[10]


def test_row_11_a_run_without_measurable_time():
    server = _server(_good())

    measurement = _measure(server, 0)

    assert measurement.problem == LINES[11] and len(chat_calls(server)) == 1


def test_a_prompt_of_0_tokens_is_valid_and_the_edge_of_the_context_fits():
    prompt_zero = _server([chat_to(INSTANCE.instance_id, "lmstudio_chat_prompt_zero.json")] * 4)
    edge = _server([chat_to(INSTANCE.instance_id, Response(status=200, body=json.dumps(_chat_with(prompt_tokens=8064)).encode()))] * 4)

    assert _measure(prompt_zero).problem is None
    assert _measure(edge).problem is None  # 8064 + 128 is exactly 8192


# --- the order of the checks: the first row that applies is the one --------------------------------------------


def test_a_wrong_status_with_an_unreadable_body_is_row_6():
    _server_, measurement = _single(Response(status=500, body=b"not json"))

    assert measurement.problem == LINES[6]


def test_a_negative_counter_beside_a_wrong_finish_reason_is_row_7():
    payload = _chat_with(prompt_tokens=-1)
    payload["choices"][0]["finish_reason"] = "stop"

    _server_, measurement = _single(payload)

    assert measurement.problem == LINES[7]


def test_a_wrong_finish_reason_beside_a_wrong_token_count_is_row_8():
    payload = _chat_with(completion_tokens=127)
    payload["choices"][0]["finish_reason"] = "stop"

    _server_, measurement = _single(payload)

    assert measurement.problem == LINES[8]


def test_a_wrong_token_count_beside_a_prompt_beyond_the_context_is_row_9():
    _server_, measurement = _single(_chat_with(completion_tokens=127, prompt_tokens=8100))

    assert measurement.problem == LINES[9]


def test_a_prompt_beyond_the_context_beside_no_time_is_row_10():
    server = _server([chat_to(INSTANCE.instance_id, "lmstudio_chat_prompt_large.json")] * 4)

    assert _measure(server, 0).problem == LINES[10]


def test_the_limit_of_time_comes_before_a_wrong_status():
    server = _server([Response(status=500, body=b"{}")] * 4)

    measurement = _measure(server, 700 * SECOND)

    assert measurement.problem == LINES[4]


# --- the real transport with an injected opener: the limits of 10 s and 600 s, the status and the body ---------------


def _real(chat, *, lists: object = None):
    """A `LocalLmStudioServer` whose list answers with the recording and whose chat is `chat`."""
    listing = lists if lists is not None else FakeAnswer(body=(FIXTURES / DISTINCT).read_bytes())
    return scripted_server({MODELS_PATH: listing, CHAT_PATH: chat})


def test_row_1_a_list_that_reaches_its_limit_of_time_before_any_chat():
    server = _real(FakeAnswer(), lists=TimeoutError("timed out"))

    measurement = _measure(server)

    assert measurement.problem == LINES[1] and measurement.calls == 1
    assert server._opener.opened == [("GET", MODELS_PATH, 10.0)]


def test_row_4_a_chat_that_reaches_its_limit_of_time_after_600_s_is_asked_for():
    server = _real(urllib.error.URLError(TimeoutError("timed out")))

    measurement = _measure(server)

    assert measurement.problem == LINES[4]
    assert server._opener.opened == [("GET", MODELS_PATH, 10.0), ("POST", CHAT_PATH, 600.0)]


def _wrapped_limit():
    return urllib.error.URLError(TimeoutError("timed out"))


@pytest.mark.parametrize(
    "make",
    [lambda: FakeAnswer(read_error=_wrapped_limit()), lambda: http_error(500, read_error=_wrapped_limit())],
    ids=["body read", "error body read"],
)
def test_a_wrapped_limit_of_time_is_row_1_at_the_list_and_row_4_at_the_chat(make):
    at_list = _measure(_real(FakeAnswer(), lists=make()))
    at_chat = _measure(_real(make()))

    assert (at_list.problem, at_chat.problem) == (LINES[1], LINES[4])


@pytest.mark.parametrize(("status", "row"), [(200, 7), (500, 6)])
def test_a_value_error_at_the_read_of_a_chat_keeps_the_status_for_the_row(status, row):
    reply = FakeAnswer(200, read_error=ValueError("closed")) if status == 200 else http_error(status, read_error=ValueError("closed"))

    assert _measure(_real(reply)).problem == LINES[row]


def test_row_3_a_chat_whose_connection_is_gone():
    server = _real(urllib.error.URLError(ConnectionResetError("reset")))

    assert _measure(server).problem == LINES[3]


def test_row_4_comes_before_row_6_for_a_401_whose_body_hits_the_limit_of_time():
    assert _measure(_real(http_error(401, read_error=TimeoutError("timed out")))).problem == LINES[4]


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("body", [b"{}", None], ids=["readable body", "unreadable body"])
def test_row_6_a_401_or_403_of_the_chat_without_a_limit_of_time(status, body):
    reply = http_error(status, b"{}") if body is not None else http_error(status, read_error=OSError("reset"))

    assert _measure(_real(reply)).problem == LINES[6]


def test_row_5_and_row_6_for_a_400_with_and_without_the_recorded_pattern():
    jit = (FIXTURES / "lmstudio_jit_on_unloaded_chat.json").read_bytes()

    assert _measure(_real(http_error(400, jit))).problem == LINES[5]
    assert _measure(_real(http_error(400, b'{"error": {"message": "bad"}}'))).problem == LINES[6]
    assert _measure(_real(http_error(400, read_error=OSError("reset")))).problem == LINES[6]


@pytest.mark.parametrize("why", ["unreadable", "oversized"])
def test_row_7_a_200_whose_body_cannot_be_read_or_is_too_large_on_both_transports(why):
    from modelroom.lmstudio_server import MAX_ANSWER_BYTES

    big = b"x" * (MAX_ANSWER_BYTES + 1)
    real = _real(FakeAnswer(200, read_error=OSError("reset")) if why == "unreadable" else FakeAnswer(200, big))
    fixture = _server([Response(status=200, body=big)]) if why == "oversized" else None

    assert _measure(real).problem == LINES[7]
    if fixture is not None:  # a malformed JSON alone would not reach this branch: the size check does
        assert _measure(fixture).problem == LINES[7]


# --- (5) a thinking model answers with an empty content ----------------------------------------------------------


def test_an_empty_content_with_128_reasoning_tokens_and_draft_counters_gives_the_same_number():
    recorded = _recorded("lmstudio_chat_one.json")
    plain = copy.deepcopy(recorded)
    plain["choices"][0]["message"].pop("reasoning_content")
    plain["usage"].pop("completion_tokens_details")
    plain.pop("stats")
    assert recorded["choices"][0]["message"]["content"] == "" and recorded["usage"]["completion_tokens"] == 128

    with_thinking = _measure(_server([recorded] * 4), 2 * SECOND, 2 * SECOND, 4 * SECOND, SECOND)
    without = _measure(_server([plain] * 4), 2 * SECOND, 2 * SECOND, 4 * SECOND, SECOND)

    assert with_thinking.problem is None and without.problem is None
    assert (with_thinking.request_tps, with_thinking.request_min, with_thinking.request_max) == (
        without.request_tps, without.request_min, without.request_max
    )


# --- (6) the wall time leaves out the reads and the warm-up -------------------------------------------------------


def test_the_clock_is_read_right_before_and_right_after_each_chat_and_nowhere_else():
    log: list[str] = []
    server = _server(_good())
    traced = _Traced(server, log)

    measurement = _measure(traced, SECOND, SECOND, SECOND, SECOND, log=log)

    assert measurement.problem is None
    assert log == _expected_log()


class _Traced:
    """The fixture server, with every call written into the log the clock writes its ticks into."""

    def __init__(self, server, log: list[str]) -> None:
        self._server, self._log, self.base_url = server, log, server.base_url

    def __call__(self, method, path, body=None):
        self._log.append(method if method == "POST" else "GET")
        return self._server(method, path, body)


def _expected_log() -> list[str]:
    chat = ["tick", "POST", "tick"]
    return ["GET"] + chat + ["GET"] + chat + ["GET"] + chat + ["GET"] + chat + ["GET"]


# --- (7) and (8) nothing is written, nothing is ranked -------------------------------------------------------------


def _tree(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()}


def test_a_measurement_leaves_the_results_folder_as_it_was(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (results / "kept.json").write_bytes(b"{}")
    monkeypatch.chdir(tmp_path)
    before = _tree(tmp_path)

    measurement = _measure(_server(_good()))

    assert measurement.problem is None and _tree(tmp_path) == before


def _runtime_imports(path: Path) -> dict[str, set[str]]:
    """The names each `from .x import ...` brings in at run time (the `TYPE_CHECKING` block is left out)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and getattr(child.test, "id", None) == "TYPE_CHECKING":
                continue
            yield child
            yield from walk(child)

    for node in walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").removeprefix("modelroom.") if node.level == 0 else (node.module or "")
            if node.level == 0 and module not in ("modelroom",) and not (node.module or "").startswith("modelroom."):
                continue  # a library, not a module of the package
            if module in ("", "modelroom"):  # `from . import x`, `from modelroom import x`: x is the module
                for alias in node.names:
                    found.setdefault(alias.name, set()).add(alias.name)
            else:
                found.setdefault(module, set()).update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("modelroom."):
                    short = alias.name.removeprefix("modelroom.")
                    found.setdefault(short, set()).add(short)
    return found


def test_the_import_guard_sees_relative_and_absolute_imports_alike(tmp_path):
    source = tmp_path / "probe.py"
    source.write_text(
        "from .measurements import a\nfrom modelroom.ranking import b\nimport modelroom.neutral\n"
        "from modelroom import views\nfrom . import screen\nimport json\nfrom typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from modelroom.state import c\n",
        encoding="utf-8",
    )

    assert _runtime_imports(source) == {
        "measurements": {"a"}, "ranking": {"b"}, "neutral": {"neutral"}, "views": {"views"}, "screen": {"screen"},
    }


def _type_checking_imports(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for inner in node.body:
                if isinstance(inner, ast.ImportFrom):
                    found.setdefault(inner.module or "", set()).update(alias.name for alias in inner.names)
    return found


@pytest.mark.parametrize("name", ["loadtest_lmstudio", "guided_lmstudio", "lmstudio_server"])
def test_no_import_at_run_time_from_ranking_neutral_or_the_records(name):
    runtime = _runtime_imports(PACKAGE / f"{name}.py")

    assert not {"ranking", "neutral", "views", "screen", "state"} & set(runtime), runtime
    allowed_measurements = {"guided_lmstudio": {"shipped_protocol"}}.get(name, set())
    assert runtime.get("measurements", set()) <= allowed_measurements
    allowed_loadtest = {"loadtest_lmstudio": {"_mean_within"}, "guided_lmstudio": {"LoadTestError"}}.get(name, set())
    assert runtime.get("loadtest", set()) <= allowed_loadtest


def test_the_type_only_import_of_the_scenario_stays_outside_the_run_time_imports():
    guided = PACKAGE / "guided_lmstudio.py"

    assert "Scenario" in _type_checking_imports(guided).get("measurements", set())
    assert "Scenario" not in _runtime_imports(guided).get("measurements", set())
    assert "Scenario" not in _runtime_imports(PACKAGE / "loadtest_lmstudio.py").get("measurements", set())


def test_the_measuring_modules_send_the_chat_and_nothing_else():
    """The calls a whole measurement makes are the two the transport allows: the list and the chat."""
    server = _server(_good())

    _measure(server)

    assert {(method, path) for method, path, _body in server.calls} == {("GET", MODELS_PATH), ("POST", CHAT_PATH)}
    assert MAX_TOKENS == 128
