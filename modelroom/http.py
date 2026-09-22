"""Minimal HTTP transport for the fetch work package.

`Transport` is the shape every fetcher (`modelroom/hf.py`, `modelroom/ollama.py`) depends on:
a plain callable `(method, url, headers) -> Response`, passed in by the caller. There is no
monkeypatching and no mocking library anywhere in this project: a test that needs a fetcher to
see a particular response constructs a `FixtureTransport` and passes it in like any other
value. `UrllibTransport` is the only implementation that touches the network, and it is used
nowhere in the test suite.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

import modelroom

USER_AGENT = f"modelroom/{modelroom.__version__}"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_REQUEST_BUDGET = 400


class BudgetExhaustedError(Exception):
    """Raised by `BudgetedTransport` once a run's request budget is used up.

    Callers catch this exactly like any other fetch failure and end the area that was in
    progress as `incomplete` with this message; no further requests are made afterwards.
    """

    def __init__(self) -> None:
        super().__init__("budget exhausted")


@dataclass(frozen=True)
class Response:
    """One HTTP response: status, headers (as given, not lower-cased) and the raw body."""

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> object:
        return json.loads(self.body.decode("utf-8"))

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup; HTTP header names are not case sensitive."""
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None


class Transport(Protocol):
    """`(method, url, headers) -> Response`. Every fetcher takes one of these as a parameter."""

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None) -> Response: ...


class UrllibTransport:
    """The real transport: stdlib `urllib.request`, a fixed User-Agent, one timeout, no retries.

    `urllib.request` already follows a redirect for a plain GET/HEAD by default and applies no
    retry policy of its own, which matches "no redirects beyond what urllib does by default, no
    retries beyond one".
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None) -> Response:
        request_headers = {"User-Agent": USER_AGENT}
        request_headers.update(headers or {})
        request = urllib.request.Request(url, method=method, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return Response(status=response.status, headers=dict(response.headers), body=response.read())
        except urllib.error.HTTPError as exc:
            return Response(status=exc.code, headers=dict(exc.headers or {}), body=exc.read())


class FixtureTransport:
    """Test transport: maps `(method, url)` to a recorded `Response`, records every call made.

    Never touches the network. A call for a `(method, url)` pair with no recorded response is
    a test-authoring error, surfaced immediately as a `KeyError` rather than a silent failure.
    """

    def __init__(self, responses: dict[tuple[str, str], Response]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None) -> Response:
        self.calls.append((method, url))
        return self._responses[(method, url)]


class BudgetedTransport:
    """Wraps a `Transport` with a per-run request budget.

    Every fetch run wraps its transport in one of these so the request count is enforced in
    exactly one place, regardless of which fetcher (Hugging Face, Ollama, or both) is making
    the calls.
    """

    def __init__(self, inner: Transport, budget: int = DEFAULT_REQUEST_BUDGET) -> None:
        self._inner = inner
        self._budget = budget
        self.used = 0

    @property
    def remaining(self) -> int:
        return self._budget - self.used

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None) -> Response:
        if self.used >= self._budget:
            raise BudgetExhaustedError()
        self.used += 1
        return self._inner(method, url, headers)
