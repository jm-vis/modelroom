"""Tests for several requests at once at an LM Studio server (AP9-PM-1, part B) and the time of a batch (part C).

The instance states a `parallel` and the scenario names more than one request: a run is then a batch of n
requests released together (`loadtest_batch`). The fixture server answers as a function of the request body,
and its answer function moves the clock of its own thread by the wall time the test gives that request, so the
numbers can be worked out by hand. The clock is read `1 + 2n` times per batch (the release in the barrier's
action, then the start and the end of every request) and twice for the warm-up, which runs in the main thread.
"""

from __future__ import annotations

import re
import threading

import pytest

from modelroom.guided_lmstudio import NOT_RANKED_NOTE, card_facts, measurement_lines
from modelroom.http import Response
from modelroom.intro import UNICODE_GLYPHS
from modelroom.lmstudio_server import CHAT_PATH, MODELS_PATH, LmStudioServerError, LmStudioTimeoutError, LoadedModel
from modelroom.loadtest_lmstudio import LmStudioMeasurement, batch_size, run_lmstudio_load_test
from modelroom.loadtest_vllm import MAX_REQUESTS
from modelroom.measurements import Scenario

from fixture_http import FIXTURES, json_response
from fixture_lmstudio import chat_calls, list_calls, lmstudio_servers, run_of

PROMPT = "Describe how a public library shelves a new book."
SECOND = 1_000_000_000
DASH = UNICODE_GLYPHS.skip
DISTINCT = "lmstudio_v1_models_loaded_distinct_key.json"
ID = "qwen3.5-4b:2"
LINES = {
    2: "the loaded model changed during the measurement",
    3: "stopped answering during the measurement",
    4: "a request did not answer within 600 s",
    7: "a request carried no readable finish_reason and usage",
    8: "a request did not end with finish_reason length",
    9: "a request did not carry 128 tokens",
    11: "a request took no measurable time",
}


def _instance(parallel: int | None) -> LoadedModel:
    return LoadedModel(ID, "qwen3.5-4b", 8192, parallel)


def _scenario(requests: int) -> Scenario:
    return Scenario(context_requested=8192, context_origin="default", kv_type="f16", kv_type_assumed=True, requests=requests)


class _Bench:
    """A fixture server with a body-driven chat, and the clock that goes with it.

    `walls` maps the number k of a request (`<prompt> Request <k>.`) to its wall time; the others take one
    second. `replies` maps k to another answer (a `Response`, or an error to raise). `gate` is a barrier every
    request of a batch (k of 1 or more) waits at, so an answer is only given once all n calls have arrived.
    """

    def __init__(self, n: int, walls=None, replies=None, lists=None, gate: bool = False, release_late: int = 0) -> None:
        self.n = n
        self.walls = walls or {}
        self.replies = replies or {}
        self.release_late = release_late  # the release is read this much later than the workers' common start
        self.local = threading.local()
        self.lock = threading.Lock()
        self.reads = 0
        self.prompts: list[str] = []
        self.order: list[int] = []  # the k of every request, in the order its end time was taken
        self.gate = threading.Barrier(n, timeout=3.0) if gate else None
        self.ended: dict[int, threading.Event] = {}
        self.waits: dict[int, int] = {}  # a request k is answered only after the end time of request `waits[k]`
        good = json_response("lmstudio_chat_one.json")
        self._good = good
        self.server = lmstudio_servers(lists or [DISTINCT], chats=[])
        self.server._answers[("POST", CHAT_PATH)] = self._chat

    def event(self, k: int) -> threading.Event:
        with self.lock:
            return self.ended.setdefault(k, threading.Event())

    def clock(self) -> int:
        with self.lock:
            read = self.reads
            self.reads += 1
        if read < 2:  # the warm-up, in the main thread
            return read * SECOND
        _batch, position = divmod(read - 2, 1 + 2 * self.n)
        if position == 0:
            return 1_000 + self.release_late
        k = getattr(self.local, "returned", None)
        if k is not None:  # the end time of request k is being taken: its call has just returned
            self.local.returned = None
            with self.lock:
                self.order.append(k)
            self.event(k).set()
        return getattr(self.local, "now", 1_000)

    def _chat(self, body: dict) -> Response:
        prompt = body["messages"][0]["content"]
        k = int(re.search(r" Request (\d+)\.$", prompt).group(1)) if " Request " in prompt else -1
        with self.lock:
            self.prompts.append(prompt)
        if self.gate is not None and k >= 1:
            self.gate.wait()
        if k in self.waits:
            assert self.event(self.waits[k]).wait(3.0)
        self.local.now = 1_000 + self.walls.get(k, SECOND)
        self.local.returned = k
        return self.replies.get(k, self._good)

    def run(self, instance: LoadedModel, scenario: Scenario) -> LmStudioMeasurement:
        return run_lmstudio_load_test(self.server, instance, scenario, PROMPT, self.clock)


def _k_of(prompt: str) -> int:
    return int(re.search(r" Request (\d+)\.$", prompt).group(1))


# --- B.1 how many at once -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parallel", "requests", "n"),
    [(None, 10, 1), (1, 10, 1), (0, 10, 1), (4, 1, 1), (4, 10, 4), (2, 10, 2), (8, 3, 3), (300, 300, 256), (256, 300, 256), (300, 200, 200)],
)
def test_the_batch_is_the_smallest_of_requests_parallel_and_the_limit(parallel, requests, n):
    assert batch_size(_instance(parallel), _scenario(requests)) == n


# --- (7) n = 1 stays the serial course, byte for byte ---------------------------------------------------------


@pytest.mark.parametrize(("parallel", "requests"), [(None, 10), (1, 10), (4, 1)], ids=["no parallel", "parallel 1", "one request"])
def test_n_of_one_is_the_serial_course_with_the_same_bodies_calls_lines_and_card(parallel, requests):
    chats = [json_response("lmstudio_chat_one.json")] * 4
    server = lmstudio_servers([DISTINCT], chats=chats)
    state = {"i": 0}

    def clock() -> int:
        n, state["i"] = state["i"], state["i"] + 1
        return (n // 2) * 100 * SECOND + (n % 2) * 2 * SECOND

    measurement = run_lmstudio_load_test(server, _instance(parallel), _scenario(requests), PROMPT, clock)

    assert measurement.problem is None and measurement.calls == 9 and measurement.requests == 1
    assert [call[:2] for call in server.calls] == [("GET", MODELS_PATH), ("POST", CHAT_PATH)] * 4 + [("GET", MODELS_PATH)]
    assert all(call[2]["messages"] == [{"role": "user", "content": PROMPT}] for call in chat_calls(server))
    assert (measurement.total_tps, measurement.total_min, measurement.total_max) == (None, None, None)
    assert (measurement.ranking_requests, measurement.parallel) == (requests, parallel)
    assert measurement_lines(measurement, DASH) == [
        f"measured LM Studio {ID}: 64.0 tok/s (64.0{DASH}64.0), one request, wall time, context 8192",
        NOT_RANKED_NOTE,
    ]
    run = run_of(None)
    run.servers.measured["lmstudio"] = measurement
    assert [fact.note for fact in card_facts(run)][0] == "64.0 tok/s, one request, wall time"


# --- (8) parallel 4, requests 10 -----------------------------------------------------------------------------------


# Batch 1 (k 1..4) takes 1, 2, 3 and 4 s, batch 2 (k 5..8) 2 s each, batch 3 (k 9..12) 1 s each.
WALLS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 2, 6: 2, 7: 2, 8: 2, 9: 1, 10: 1, 11: 1, 12: 1}


def _measure_four():
    bench = _Bench(4, walls={k: seconds * SECOND for k, seconds in WALLS.items()})
    return bench, bench.run(_instance(4), _scenario(10))


def test_parallel_4_of_10_requests_is_three_batches_of_4_with_five_reads_and_18_calls():
    bench, measurement = _measure_four()

    assert measurement.problem is None and measurement.requests == 4 and measurement.calls == 18 == 6 + 3 * 4
    methods = [call[0] for call in bench.server.calls]
    assert methods == ["GET", "POST", "GET"] + ["POST"] * 4 + ["GET"] + ["POST"] * 4 + ["GET"] + ["POST"] * 4 + ["GET"]
    assert len(list_calls(bench.server)) == 5 and len(chat_calls(bench.server)) == 13


def test_the_prompts_run_from_0_to_12_without_a_repeat_and_a_batch_holds_its_own_numbers():
    bench, _measurement = _measure_four()

    chats = [call[2]["messages"][0]["content"] for call in chat_calls(bench.server)]
    assert chats[0] == f"{PROMPT} Request 0."
    assert sorted(_k_of(prompt) for prompt in chats) == list(range(13))
    assert all(prompt == f"{PROMPT} Request {_k_of(prompt)}." for prompt in chats)
    assert [sorted(_k_of(prompt) for prompt in chats[start : start + 4]) for start in (1, 5, 9)] == [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
    ]
    assert all(call[2]["model"] == ID and call[2]["max_tokens"] == 128 and call[2]["stream"] is False for call in chat_calls(bench.server))


def test_the_numbers_are_worked_out_from_the_batch_times_and_the_request_times():
    """Batches of 4 x 128 tokens over 4, 2 and 1 s: 128, 256 and 512 tok/s. A request: 128 / its seconds."""
    _bench, measurement = _measure_four()

    assert (measurement.total_min, measurement.total_max) == (128.0, 512.0)
    assert measurement.total_tps == pytest.approx((128 + 256 + 512) / 3)
    assert (measurement.request_min, measurement.request_max) == (32.0, 128.0)
    assert measurement.request_tps == pytest.approx((128 + 64 + 128 / 3 + 32 + 4 * 64 + 4 * 128) / 12)
    assert (measurement.ranking_requests, measurement.parallel) == (10, 4)


# --- (9), (10) the limits of n -------------------------------------------------------------------------------------


def test_parallel_8_of_3_requests_is_3_at_once_and_says_nothing_of_the_limit():
    bench = _Bench(3)

    measurement = bench.run(_instance(8), _scenario(3))
    lines = measurement_lines(measurement, DASH)

    assert measurement.problem is None and measurement.requests == 3 and measurement.calls == 6 + 3 * 3
    assert len(lines) == 3 and lines[0].startswith("measured LM Studio") and ", 3 at once, wall time," in lines[0]
    assert lines[1].startswith("per request: ") and lines[2] == NOT_RANKED_NOTE  # no "of N" line, no limit line


def test_parallel_300_of_300_requests_is_256_at_once():
    bench = _Bench(MAX_REQUESTS)

    measurement = bench.run(_instance(300), _scenario(300))

    assert measurement.problem is None and measurement.requests == 256 and measurement.calls == 6 + 3 * 256
    assert len(chat_calls(bench.server)) == 1 + 3 * 256
    lines = measurement_lines(measurement, DASH)
    assert lines[0] == "256 of 300 requests measured"
    assert not any("allows" in line for line in lines)  # 256 is the limit of modelroom, not the instance's parallel


# --- (11) a failing batch ------------------------------------------------------------------------------------------


def _chat(name: str) -> Response:
    return json_response(name)


def _failing(replies: dict, lists=None, **kwargs):
    before = threading.active_count()
    bench = _Bench(4, replies=replies, lists=lists, **kwargs)
    measurement = bench.run(_instance(4), _scenario(10))
    assert threading.active_count() == before  # every thread of the batch has ended
    assert measurement.request_tps is None and measurement.total_tps is None
    assert measurement.calls == len(bench.server.calls)
    return bench, measurement


def test_two_requests_failing_with_different_rows_in_reverse_order_give_the_row_of_the_lower_index():
    """Index 1 (k 2) has row 9, index 3 (k 4) row 8; index 3 is answered first, row 9 is the one."""
    bench = _Bench(4, replies={2: _chat("lmstudio_chat_tokens_127.json"), 4: _chat("lmstudio_chat_finish_stop.json")})
    bench.waits[2] = 4  # request 2 returns only after the end time of request 4 was taken
    before = threading.active_count()

    measurement = bench.run(_instance(4), _scenario(10))

    assert bench.order.index(4) < bench.order.index(2)  # the reverse order of completion, proven by the clock reads
    assert measurement.problem == LINES[9] != LINES[8]
    assert threading.active_count() == before
    assert (measurement.request_tps, measurement.total_tps) == (None, None)
    assert measurement.calls == len(bench.server.calls) == 2 + 1 + 4
    assert len(list_calls(bench.server)) == 2  # no read after the failed batch


@pytest.mark.parametrize(
    ("reply", "row"),
    [
        (LmStudioServerError("gone"), 3),
        (LmStudioTimeoutError("limit"), 4),
        (Response(status=200, body=b"not json"), 7),
        (_chat("lmstudio_chat_finish_stop.json"), 8),
    ],
    ids=["connection gone", "timeout", "unreadable", "finish stop"],
)
def test_a_request_of_a_batch_that_fails_ends_it_with_the_row_of_the_table(reply, row):
    bench, measurement = _failing({3: reply})

    assert measurement.problem == LINES[row]
    assert len(chat_calls(bench.server)) == 1 + 4 and len(list_calls(bench.server)) == 2


def test_a_change_of_the_instance_after_batch_2_is_row_2():
    lists = [DISTINCT, DISTINCT, DISTINCT, "lmstudio_v1_models_empty.json"]
    bench, measurement = _failing({}, lists=lists)

    assert measurement.problem == LINES[2]
    assert len(list_calls(bench.server)) == 4 and len(chat_calls(bench.server)) == 1 + 4 + 4 and measurement.calls == 13


def test_a_failing_request_and_a_failing_read_in_one_batch_give_the_request_and_the_read_does_not_happen():
    lists = [DISTINCT, DISTINCT, LmStudioServerError("refused")]
    bench, measurement = _failing({2: _chat("lmstudio_chat_tokens_127.json")}, lists=lists)

    assert measurement.problem == LINES[9]
    assert len(list_calls(bench.server)) == 2


@pytest.mark.parametrize("late", [SECOND, 2 * SECOND], ids=["wall time 0", "wall time below 0"])
def test_a_batch_of_no_wall_time_is_row_11_although_every_request_took_time(late):
    """Every request takes 1 s; the release is read 1 s (or 2 s) after the common start, so the batch's own time
    is 0 (or below) while the requests' times are positive: only the check of the batch's time can give row 11."""
    before = threading.active_count()
    bench = _Bench(4, release_late=late)

    measurement = bench.run(_instance(4), _scenario(10))

    assert measurement.problem == LINES[11]
    assert (measurement.request_tps, measurement.total_tps) == (None, None)
    assert measurement.calls == len(bench.server.calls) == 3 + 1 + 4  # the read after the batch was made
    assert len(list_calls(bench.server)) == 3 and threading.active_count() == before


def test_a_read_that_fails_after_a_batch_of_no_wall_time_gives_the_row_of_the_read():
    """The contested case (decision of 2026-10-10): positive request times, a batch time of 0 and a read after
    the batch that shows another model give row 2, not row 11."""
    lists = [DISTINCT, DISTINCT, "lmstudio_v1_models_empty.json"]
    bench = _Bench(4, release_late=SECOND, lists=lists)

    measurement = bench.run(_instance(4), _scenario(10))

    assert measurement.problem == LINES[2]
    assert measurement.calls == len(bench.server.calls) == 3 + 1 + 4 and measurement.request_tps is None


def test_a_request_beyond_600_s_is_row_4():
    bench = _Bench(4, walls={2: 601 * SECOND})

    assert bench.run(_instance(4), _scenario(10)).problem == LINES[4]


# --- (12) the words ------------------------------------------------------------------------------------------------


def test_the_notes_of_a_number_at_once_are_these_lines_exactly():
    _bench, measurement = _measure_four()

    assert measurement_lines(measurement, DASH) == [
        "4 of 10 requests measured",
        f"measured LM Studio {ID}: 298.7 tok/s total (128.0{DASH}512.0), 4 at once, wall time, context 8192",
        f"per request: 86.2 tok/s (32.0{DASH}128.0)",
        "LM Studio allows 4 at once for this model (Max Concurrent Predictions in LM Studio)",
        NOT_RANKED_NOTE,
    ]


def test_the_card_note_of_a_number_at_once_and_its_longest_form_fits_the_column():
    _bench, measurement = _measure_four()
    run = run_of(None)
    run.servers.measured["lmstudio"] = measurement

    assert card_facts(run)[0].note == "298.7 tok/s total, 4 at once, wall time"
    largest = 256 * 128 / 1e-9  # 256 x 128 tokens in one nanosecond: the largest a total can be
    assert f"{largest:.1f}" == "32768000000000.0"
    huge = LmStudioMeasurement(ID, 8192, 1.0, 1.0, 1.0, None, 774, 256, 300, 300, largest, largest, largest)
    run.servers.measured["lmstudio"] = huge
    note = card_facts(run)[0].note
    assert note == "32768000000000.0 tok/s total, 256 at once, wall time" and len(note) == 52 <= 65


# --- (13) the requests really are at the server at once ------------------------------------------------------------


def test_the_server_gives_no_answer_before_all_n_calls_have_arrived():
    """A serial course would fail at the gate (its timeout is the watchdog), not hang."""
    bench = _Bench(4, gate=True)

    measurement = bench.run(_instance(4), _scenario(10))

    assert measurement.problem is None and measurement.requests == 4
    assert bench.gate.n_waiting == 0 and not bench.gate.broken


def test_the_gate_of_the_fixture_stops_a_serial_course():
    gate = _Bench(2, gate=True)
    gate.gate = threading.Barrier(2, timeout=0.2)

    with pytest.raises(threading.BrokenBarrierError):
        gate.server("POST", CHAT_PATH, {"messages": [{"role": "user", "content": f"{PROMPT} Request 1."}]})
    gate.gate.abort()
