"""The measurement of a local LM Studio server: one request after the other, timed on the wall, nothing written.

The rule is the contract's (CONTRACTS.md, "Local servers and the vLLM load test", LM Studio). One chat of 128
tokens at temperature 0.0 and seed 42, not streamed, to the **instance id** of the one loaded model, as one
warm-up and three runs, one request at a time. The model list is read before the warm-up and after every
request: five reads, each of which has to show the same instance with the same context, or the measurement
ends and gives no number. A run counts with status 200, `finish_reason` `length`, 128 `completion_tokens`
(thinking tokens are part of them and `reasoning_tokens` is not read), room in the context and a wall time
above 0. The time is `clock()` before the call and after the whole body is in; the reads and the warm-up lie
outside it. The number is the mean of the three `128 / seconds` held inside its range.

**modelroom never asks LM Studio to load, unload or download a model. It reads the model list before and
after every request and drops the measurement when the loaded model changed; a load the server does on its
own inside a request can stay unseen and then sits in the measured time.**

The result is an `LmStudioMeasurement` in memory: no record, no profile, no ranking, nothing on the disk.
Every row of the table of the contract ends the measurement as a `problem` of the result, never as an
exception. This module imports nothing from `measurements.py` and nothing from `loadtest.py` but `_mean_within`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .lmstudio_server import CHAT_PATH, LmStudioServer, LmStudioServerError, LmStudioTimeoutError, LoadedModel, read_lmstudio_models
from .loadtest import _mean_within
from .vllm_server import CHAT_TIMEOUT_SECONDS

if TYPE_CHECKING:  # pragma: no cover - a type only, never imported at run time
    from .http import Response
    from .measurements import Scenario

MAX_TOKENS = 128
TEMPERATURE = 0.0
SEED = 42
RUNS = 3
FINISH_REASON = "length"
LOAD_FAILURE = "Failed to load model"

Clock = Callable[[], int]

# The table of the contract by its row number. Row 1 and row 3 say the same, as the contract has it.
PROBLEM_WORDS = {
    1: "stopped answering during the measurement",
    2: "the loaded model changed during the measurement",
    3: "stopped answering during the measurement",
    4: f"a request did not answer within {CHAT_TIMEOUT_SECONDS:.0f} s",
    5: "failed to load the model for the request; nothing measured",
    6: "a request did not answer with status 200",
    7: "a request carried no readable finish_reason and usage",
    8: f"a request did not end with finish_reason {FINISH_REASON}",
    9: f"a request did not carry {MAX_TOKENS} tokens",
    10: "a request did not fit the context {context}",
    11: "a request took no measurable time",
}


@dataclass(frozen=True)
class LmStudioMeasurement:
    """What one measurement of an LM Studio server found, in memory only.

    The speeds are set when `problem` is `None`; `problem` is the words of a row of the table (without the
    prefix `LM Studio: `) and then no speed exists. `calls` counts the calls made to the server (at most nine).
    """

    instance_id: str
    context: int
    request_tps: float | None
    request_min: float | None
    request_max: float | None
    problem: str | None
    calls: int


class _Stop(Exception):
    """A row of the table ends the measurement. Internal: it never leaves this module."""

    def __init__(self, row: int) -> None:
        super().__init__(row)
        self.row = row


def chat_body(instance_id: str, prompt: str) -> dict:
    """The request: the instance id as the model, 128 tokens, temperature 0.0, seed 42, not streamed."""
    return {
        "model": instance_id,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }


def run_lmstudio_load_test(
    server: LmStudioServer, instance: LoadedModel, scenario: "Scenario", prompt: str, clock: Clock = time.perf_counter_ns
) -> LmStudioMeasurement:
    """Read, warm up, read, then three runs each followed by a read; a row of the table ends it with a `problem`.

    `scenario` is the run's: that the context of the instance is the one it asks for is judged before the
    call (`guided_lmstudio`), a request does not change.
    """
    session = _Session(server, instance, prompt, clock)
    seconds: list[float] = []
    try:
        session.read()
        session.chat()
        session.read()
        for _ in range(RUNS):
            seconds.append(session.chat() / 1e9)
            session.read()
    except _Stop as stop:
        words = PROBLEM_WORDS[stop.row].format(context=instance.context_length)
        return LmStudioMeasurement(instance.instance_id, instance.context_length, None, None, None, words, session.calls)
    speeds = sorted(MAX_TOKENS / each for each in seconds)
    return LmStudioMeasurement(
        instance.instance_id, instance.context_length, _mean_within(speeds), speeds[0], speeds[-1], None, session.calls
    )


class _Session:
    """The calls of one measurement: each read and each chat counted, each failing as a row of the table."""

    def __init__(self, server: LmStudioServer, instance: LoadedModel, prompt: str, clock: Clock) -> None:
        self._server = server
        self._instance = instance
        self._prompt = prompt
        self._clock = clock
        self.calls = 0

    def read(self) -> None:
        """One read of the list: the same instance with the same context, else row 1 (no read) or row 2."""
        self.calls += 1
        try:
            models = read_lmstudio_models(self._server)
        except LmStudioServerError:
            raise _Stop(1) from None
        same = (m for m in models.loaded if m.instance_id == self._instance.instance_id)
        if not any(m.context_length == self._instance.context_length for m in same):
            raise _Stop(2)

    def chat(self) -> int:
        """One chat: its wall time in nanoseconds, or the row of the table it ended on."""
        self.calls += 1
        body = chat_body(self._instance.instance_id, self._prompt)
        started = self._clock()
        try:
            response = self._server("POST", CHAT_PATH, body)
        except LmStudioTimeoutError:
            raise _Stop(4) from None
        except LmStudioServerError:
            raise _Stop(3) from None
        wall = self._clock() - started
        if wall > CHAT_TIMEOUT_SECONDS * 1e9:
            raise _Stop(4)
        _judge(response, wall, self._instance.context_length)
        return wall


def _judge(response: "Response", wall: int, context: int) -> None:
    """Rows 5 to 11, in the order of the table: the first that applies ends the measurement."""
    if response.status != 200:
        raise _Stop(5 if response.status == 400 and LOAD_FAILURE in response.body.decode("utf-8", "replace") else 6)
    fields = _fields(response)
    if fields is None:
        raise _Stop(7)
    finish, prompt_tokens, completion_tokens = fields
    if finish != FINISH_REASON:
        raise _Stop(8)
    if completion_tokens != MAX_TOKENS:
        raise _Stop(9)
    if prompt_tokens + MAX_TOKENS > context:
        raise _Stop(10)
    if wall <= 0:
        raise _Stop(11)


def _fields(response: "Response") -> tuple[str, int, int] | None:
    """`finish_reason`, `prompt_tokens`, `completion_tokens`, or `None` where one is missing or has the wrong form."""
    try:
        payload = response.json()
        finish = payload["choices"][0]["finish_reason"]
        usage = payload["usage"]
        prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
    except (ValueError, RecursionError, KeyError, IndexError, TypeError, AttributeError):
        return None
    if isinstance(finish, str) and _count(prompt_tokens) and _count(completion_tokens):
        return finish, prompt_tokens, completion_tokens
    return None


def _count(value: Any) -> bool:
    """A whole number of tokens: `int` and not `bool`, and not below 0 (0 is fine)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


__all__ = [
    "MAX_TOKENS",
    "PROBLEM_WORDS",
    "LmStudioMeasurement",
    "chat_body",
    "run_lmstudio_load_test",
]
