"""Tests for modelroom.loadtest_vllm: one warm-up, three batches of n requests at once, nothing written.

Every answer is the one recorded from vLLM 0.30.0 on one L4 (`tests/fixtures/vllm_*_l4*`), and the
clock is injected: each request takes the wall time recorded for it, each batch one of three
synthetic batch times. The requests really run in threads behind one barrier, against the fixture.
"""

from __future__ import annotations

import copy
import json
import re
import threading

import pytest

from modelroom.guided_context import context_scenario
from modelroom.http import Response
from modelroom.loadtest import LoadTestError
from modelroom.loadtest_vllm import BATCHES, MAX_REQUESTS, chat_body, run_vllm_load_test
from modelroom.vllm_server import CHAT_PATH, METRICS_PATH, FixtureVllmServer, ServedModel, VllmServerError

from fixture_support import FIXTURES

SERVED = ServedModel(id="Qwen/Qwen2.5-0.5B-Instruct", root="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=4096)
CONCURRENT = json.loads((FIXTURES / "vllm_chat_concurrent_l4.json").read_text(encoding="utf-8"))
WARM_UP = json.loads((FIXTURES / "vllm_chat_one_l4.json").read_text(encoding="utf-8"))
RECORDED = {record["index"]: record for record in CONCURRENT["requests"]}
METRICS = (FIXTURES / "vllm_metrics_before_l4.txt").read_text(encoding="utf-8")
# Three batch times around the recorded 0.7203 s: 2560 tokens each give 3554.2, 3604.3 and 3505.6.
BATCH_NS = (720_266_104, 710_266_104, 730_266_104)


def _metrics(generated: float, running: float = 0.0, waiting: float = 0.0) -> Response:
    """The recorded metrics page with the three numbers the load test reads set to these."""
    text = re.sub(r"^(vllm:generation_tokens_total\{[^}]*\}) \S+$", rf"\g<1> {generated}", METRICS, flags=re.M)
    text = re.sub(r"^(vllm:num_requests_running\{[^}]*\}) \S+$", rf"\g<1> {running}", text, flags=re.M)
    text = re.sub(r"^(vllm:num_requests_waiting\{[^}]*\}) \S+$", rf"\g<1> {waiting}", text, flags=re.M)
    return Response(status=200, body=text.encode("utf-8"))


def _pages(n: int, **changed) -> list[Response]:
    """Six metrics pages, before and after each batch, each batch adding n x 128 generated tokens."""
    pages = []
    for batch in range(BATCHES):
        before = 128.0 + batch * n * 128
        pages += [_metrics(before), _metrics(before + n * 128)]
    for index, page in changed.items():
        pages[int(index.removeprefix("page"))] = page
    return pages


class _Recorded:
    """The fixture server plus the clock: a request advances its own thread's clock by its wall.

    The main thread reads the clock twice per batch -- when the barrier lets the batch go and when
    the last request is back -- and gets the three synthetic batch times.
    """

    def __init__(self, n: int, pages: list[Response], finish=None, gone_at: int | None = None) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._chats = 0
        self._n = n
        self._finish = finish or {}
        self._gone_at = gone_at
        main = []
        for wall in BATCH_NS:
            main += [0, wall]
        self._main = iter(main)
        self.server = FixtureVllmServer({("POST", CHAT_PATH): self._chat, ("GET", METRICS_PATH): pages})

    def clock(self) -> int:
        if threading.current_thread() is threading.main_thread():
            return next(self._main)
        return getattr(self._local, "now", 0)

    def _chat(self, body: dict) -> Response:
        with self._lock:
            number = self._chats
            self._chats += 1
        if self._gone_at is not None and number >= self._gone_at:
            raise VllmServerError("http://127.0.0.1:8000/v1/chat/completions: the vLLM server did not answer")
        index = int(re.search(r"Request (\d+)\.", body["messages"][0]["content"]).group(1))
        record = WARM_UP if index == 0 else RECORDED[(index - 1) % 20 + 1]
        response = copy.deepcopy(record["response"])
        batch = 0 if number == 0 else (number - 1) // self._n + 1
        if batch in self._finish:
            response["choices"][0]["finish_reason"] = self._finish[batch]
        self._local.now = getattr(self._local, "now", 0) + round(record["wall_seconds"] * 1e9)
        return Response(status=200, body=json.dumps(response).encode("utf-8"))


def _measure(requests: int = 20, context: int = 4096, **kwargs):
    n = min(requests, MAX_REQUESTS)
    recorded = _Recorded(n, kwargs.pop("pages", None) or _pages(n), **kwargs)
    scenario = context_scenario(context, requests)
    measurement = run_vllm_load_test(recorded.server, SERVED, scenario, clock=recorded.clock)
    return measurement, recorded.server, scenario


def test_one_warm_up_three_batches_of_n_and_six_metrics_calls():
    _measurement, server, _scenario = _measure()

    chats = [call for call in server.calls if call[1] == CHAT_PATH]
    metrics = [call for call in server.calls if call[1] == METRICS_PATH]
    assert (len(chats), len(metrics)) == (1 + 3 * 20, 6)
    assert {call[1] for call in server.calls} == {CHAT_PATH, METRICS_PATH}


def test_every_body_is_the_recorded_request():
    _measurement, server, _scenario = _measure()

    bodies = [call[2] for call in server.calls if call[1] == CHAT_PATH]
    assert bodies[0] == WARM_UP["request"] == chat_body(SERVED.id, 0)
    assert sorted(bodies[1:21], key=lambda body: body["messages"][0]["content"]) == sorted(
        (RECORDED[index]["request"] for index in range(1, 21)), key=lambda body: body["messages"][0]["content"]
    )
    assert "stream" not in bodies[0]


def test_the_recorded_run_is_valid_and_comparable_at_its_own_speeds():
    """2560 tokens in 0.7203 s is 3554 tok/s for all twenty; one request ran 179.4 to 185.2."""
    measurement, _server, _scenario = _measure()

    assert (measurement.validity, measurement.comparable) == ("valid", True)
    assert measurement.total_tps == pytest.approx(3554.0, abs=1.0)
    assert (round(measurement.total_min, 1), round(measurement.total_max, 1)) == (3505.6, 3604.3)
    assert (round(measurement.request_min, 1), round(measurement.request_max, 1)) == (179.4, 185.2)
    assert measurement.request_min <= measurement.request_tps <= measurement.request_max
    assert measurement.problem is None


def test_the_warm_up_is_in_no_number():
    measurement, _server, _scenario = _measure()

    assert measurement.requests == 20
    assert round(128 / WARM_UP["wall_seconds"], 1) == 141.8  # would be the slowest request if it counted
    assert round(measurement.request_min, 1) == 179.4


def test_a_ranking_context_other_than_the_servers_is_not_comparable():
    measurement, _server, _scenario = _measure(context=32768)

    assert measurement.validity == "valid"
    assert measurement.problem == "not comparable (context 4096, the ranking's 32768)"


def test_sixty_four_requests_are_comparable_and_sixty_five_are_not():
    at_64, server, _scenario = _measure(requests=64)
    at_65, server_65, scenario = _measure(requests=65)

    assert at_64.comparable is True
    assert len([call for call in server.calls if call[1] == CHAT_PATH]) == 1 + 3 * 64
    assert at_65.requests == 64
    assert at_65.problem == "not comparable (64 requests measured, the ranking assumes 65)"
    assert len([call for call in server_65.calls if call[1] == CHAT_PATH]) == 1 + 3 * 64
    assert scenario.requests == 65  # the scenario never changes


def test_a_request_running_before_a_batch_makes_it_not_comparable():
    pages = _pages(20, page2=_metrics(128.0 + 2560, running=1.0))

    measurement, _server, _scenario = _measure(pages=pages)

    assert measurement.comparable is False
    assert measurement.problem.startswith("not comparable (batch 2:")


def test_another_delta_of_generated_tokens_makes_it_not_comparable():
    pages = _pages(20, page5=_metrics(128.0 + 2 * 2560 + 2560 + 128))

    measurement, _server, _scenario = _measure(pages=pages)

    assert measurement.comparable is False
    assert "batch 3" in measurement.problem and "2688" in measurement.problem


def test_a_stop_in_batch_three_only_is_invalid_and_has_no_speeds():
    measurement, _server, _scenario = _measure(finish={3: "stop"})

    assert measurement.validity == "invalid"
    assert measurement.problem == "invalid (batch 3: finish_reason stop)"
    assert (measurement.total_tps, measurement.request_tps, measurement.total_min, measurement.request_max) == (
        None,
        None,
        None,
        None,
    )


def test_a_server_that_goes_away_is_a_load_test_error():
    with pytest.raises(LoadTestError, match="did not answer"):
        _measure(gone_at=25)
