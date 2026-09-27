"""The load test of a local vLLM server: n requests at once, three times, and nothing written.

One warm-up request, then three batches. Each batch reads `/metrics`, releases n requests
behind one `threading.Barrier`, times each request and the whole batch with the injected clock
(`time.perf_counter_ns`), and reads `/metrics` again. The request is the one recorded on
2026-09-27 against vLLM 0.30.0: a chat of 128 tokens at temperature 0.0 and seed 42, not
streamed, `Describe how a public library shelves a new book. Request <i>.` -- `0` for the
warm-up, `1..n` in every batch.

n is `min(scenario.requests, 64)`; the scenario itself never changes. A wall time includes the
prefill and the queue of the server, so such a speed is never mixed with the one-request speed of
an Ollama measurement (protocol v1). The result is a `VllmMeasurement` in memory: **nothing is
persisted** -- no measurement file, no profile, no snapshot (CONTRACTS.md, "Local servers and
the vLLM load test", ADR 0008).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .loadtest import LoadTestError, _mean_within
from .measurements import Scenario
from .vllm_server import CHAT_PATH, METRICS_PATH, ServedModel, VllmServer, VllmServerError

PROMPT = "Describe how a public library shelves a new book. Request {index}."
MAX_TOKENS = 128
TEMPERATURE = 0.0
SEED = 42
BATCHES = 3
MAX_REQUESTS = 64
FINISH_REASON = "length"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
GENERATED = "vllm:generation_tokens_total"
_METRIC_LINE = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})?\s+(\S+)$")

Clock = Callable[[], int]


def chat_body(model_id: str, index: int) -> dict:
    """The request body, exactly as recorded: `0` is the warm-up, `1..n` the requests of a batch."""
    return {
        "model": model_id,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "messages": [{"role": "user", "content": PROMPT.format(index=index)}],
    }


@dataclass(frozen=True)
class Answer:
    """One request of a batch: its status, how it ended, its tokens and its wall time."""

    status: int
    finish_reason: object = None
    prompt_tokens: object = None
    completion_tokens: object = None
    ns: int = 0


@dataclass(frozen=True)
class Batch:
    """One batch: its answers, its wall time, and the server's metrics before and after it."""

    answers: tuple[Answer, ...]
    wall_ns: int
    before: dict[str, float]
    after: dict[str, float]


@dataclass(frozen=True)
class VllmMeasurement:
    """What one load test of a vLLM server found, in memory only.

    `requests` is n, the requests of each batch; `ranking_requests` what the ranking assumes. The
    speeds are set for a valid result only: `total_*` the tokens of a batch over its wall time
    (mean and range of the three batches), `request_*` the speed of one request (mean and range of
    all 3 x n).
    """

    requests: int
    ranking_requests: int
    context: int
    validity: str
    validity_reason: str | None
    comparable: bool
    comparable_reason: str | None
    total_tps: float | None = None
    total_min: float | None = None
    total_max: float | None = None
    request_tps: float | None = None
    request_min: float | None = None
    request_max: float | None = None

    @property
    def problem(self) -> str | None:
        """`invalid (<reason>)` or `not comparable (<reason>)`, else `None`."""
        if self.validity != "valid":
            return f"invalid ({self.validity_reason})"
        if not self.comparable:
            return f"not comparable ({self.comparable_reason})"
        return None


def run_vllm_load_test(
    server: VllmServer, served: ServedModel, scenario: Scenario, clock: Clock = time.perf_counter_ns
) -> VllmMeasurement:
    """One warm-up, then three batches of n requests at once; `LoadTestError` when the server goes."""
    n = min(scenario.requests, MAX_REQUESTS)
    _call(server, "POST", CHAT_PATH, chat_body(served.id, 0))
    batches = [_batch(server, served.id, n, clock) for _ in range(BATCHES)]
    return _measurement(batches, served, scenario, n)


def _call(server: VllmServer, method: str, path: str, body: dict | None = None):
    try:
        return server(method, path, body)
    except VllmServerError as exc:
        raise LoadTestError(str(exc)) from exc


def _batch(server: VllmServer, model_id: str, n: int, clock: Clock) -> Batch:
    """n requests released together; the batch's time runs from the release to the last answer."""
    before = read_metrics(server)
    barrier = threading.Barrier(n + 1)
    answers: list = [None] * n
    threads = [
        threading.Thread(target=_request, args=(server, model_id, index, barrier, clock, answers), daemon=True)
        for index in range(n)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    started = clock()
    for thread in threads:
        thread.join()
    wall = clock() - started
    for answer in answers:
        if not isinstance(answer, Answer):
            raise answer if isinstance(answer, LoadTestError) else LoadTestError("a request ended without an answer")
    return Batch(tuple(answers), wall, before, read_metrics(server))


def _request(server: VllmServer, model_id: str, index: int, barrier: threading.Barrier, clock: Clock, answers: list) -> None:
    """One request of a batch, in its own thread: wait for the others, then time the call."""
    barrier.wait()
    started = clock()
    try:
        response = _call(server, "POST", CHAT_PATH, chat_body(model_id, index + 1))
        answers[index] = _answer(response.status, response, clock() - started)
    except LoadTestError as exc:
        answers[index] = exc


def _answer(status: int, response, ns: int) -> Answer:
    """What one chat answer says; an answer of 200 without choices or usage is no answer at all."""
    if status != 200:
        return Answer(status=status, ns=ns)
    try:
        payload = response.json()
        choice = payload["choices"][0]
        usage = payload["usage"]
        return Answer(status, choice.get("finish_reason"), usage.get("prompt_tokens"), usage.get("completion_tokens"), ns)
    except (ValueError, RecursionError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LoadTestError(f"a chat answer carries no choices or no usage ({exc})") from exc


def read_metrics(server: VllmServer) -> dict[str, float]:
    """The three numbers the load test reads from `/metrics`, each summed over its label sets."""
    response = _call(server, "GET", METRICS_PATH)
    if response.status != 200:
        raise LoadTestError(f"{server.base_url}{METRICS_PATH}: status {response.status}")
    values: dict[str, float] = {}
    for line in response.body.decode("utf-8", errors="replace").splitlines():
        match = _METRIC_LINE.match(line.strip())
        if match and match.group(1) in (RUNNING, WAITING, GENERATED):
            try:
                values[match.group(1)] = values.get(match.group(1), 0.0) + float(match.group(2))
            except ValueError:
                continue
    return values


def _measurement(batches: list[Batch], served: ServedModel, scenario: Scenario, n: int) -> VllmMeasurement:
    invalid = invalid_reason(batches, served.max_model_len)
    not_comparable = not_comparable_reason(batches, served, scenario, n)
    speeds = {} if invalid is not None else _speeds(batches)
    return VllmMeasurement(
        requests=n,
        ranking_requests=scenario.requests,
        context=served.max_model_len,
        validity="invalid" if invalid else "valid",
        validity_reason=invalid,
        comparable=not_comparable is None,
        comparable_reason=not_comparable,
        **speeds,
    )


def invalid_reason(batches: list[Batch], max_model_len: int) -> str | None:
    """The first rule a request of a batch breaks: 200, `length`, 128 tokens, room in the context."""
    for number, batch in enumerate(batches, start=1):
        for answer in batch.answers:
            reason = _answer_problem(answer, max_model_len)
            if reason is not None:
                return f"batch {number}: {reason}"
        if batch.wall_ns <= 0:
            return f"batch {number}: no measurable wall time"
    return None


def _answer_problem(answer: Answer, max_model_len: int) -> str | None:
    if answer.status != 200:
        return f"status {answer.status}"
    if answer.finish_reason != FINISH_REASON:
        return f"finish_reason {answer.finish_reason}"
    if answer.completion_tokens != MAX_TOKENS:
        return f"{answer.completion_tokens} tokens, not {MAX_TOKENS}"
    if not isinstance(answer.prompt_tokens, int) or answer.prompt_tokens + MAX_TOKENS > max_model_len:
        return f"prompt {answer.prompt_tokens} + {MAX_TOKENS} tokens beyond context {max_model_len}"
    if answer.ns <= 0:
        return "a request with no measurable time"
    return None


def not_comparable_reason(batches: list[Batch], served: ServedModel, scenario: Scenario, n: int) -> str | None:
    """The first rule that fails: n requests, an idle server, n x 128 generated, the context."""
    if n != scenario.requests:
        return f"{n} requests measured, the ranking assumes {scenario.requests}"
    for number, batch in enumerate(batches, start=1):
        reason = _batch_state(batch, n)
        if reason is not None:
            return f"batch {number}: {reason}"
    if served.max_model_len != scenario.context_requested:
        return f"context {served.max_model_len}, the ranking's {scenario.context_requested}"
    return None


def _batch_state(batch: Batch, n: int) -> str | None:
    for when, page in (("before", batch.before), ("after", batch.after)):
        missing = [name for name in (RUNNING, WAITING, GENERATED) if name not in page]
        if missing:
            return f"the metrics {when} it name no {missing[0]}"
        if page[RUNNING] or page[WAITING]:
            return f"{page[RUNNING]:.0f} running, {page[WAITING]:.0f} waiting {when} it"
    delta = batch.after[GENERATED] - batch.before[GENERATED]
    if delta != n * MAX_TOKENS:
        return f"{delta:.0f} tokens generated, {n * MAX_TOKENS} expected"
    return None


def _speeds(batches: list[Batch]) -> dict[str, float | None]:
    """Total per batch and per request, each a mean held inside its own range (`loadtest._mean_within`)."""
    totals = sorted(sum(a.completion_tokens for a in b.answers) / (b.wall_ns / 1e9) for b in batches)
    requests = sorted(a.completion_tokens / (a.ns / 1e9) for b in batches for a in b.answers)
    return {
        "total_tps": _mean_within(totals),
        "total_min": totals[0],
        "total_max": totals[-1],
        "request_tps": _mean_within(requests),
        "request_min": requests[0],
        "request_max": requests[-1],
    }


__all__ = [
    "BATCHES",
    "MAX_REQUESTS",
    "MAX_TOKENS",
    "PROMPT",
    "VllmMeasurement",
    "chat_body",
    "invalid_reason",
    "not_comparable_reason",
    "read_metrics",
    "run_vllm_load_test",
]
