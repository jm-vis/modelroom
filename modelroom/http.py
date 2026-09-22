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
import urllib.parse
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
    """One HTTP response: status, headers (as given, not lower-cased) and the raw body.

    `requests_made` (F10, default `1`) is how many actual HTTP requests this response cost --
    more than one when `UrllibTransport` followed redirects to get here. `BudgetedTransport`
    reads it to charge every hop against the run's request budget, not just the one call it
    made into the transport.
    """

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    requests_made: int = 1

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


_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
MAX_REDIRECTS = 5


class _NoAutoRedirect(urllib.request.HTTPRedirectHandler):
    """Disables `urllib.request`'s own, uncounted redirect following.

    Returning `None` from `redirect_request` makes a 3xx surface as an `HTTPError` instead of
    being silently followed -- `UrllibTransport` then follows it itself, one hop at a time, so
    every hop can be counted into `Response.requests_made` (F10).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - stdlib override
        return None


class UrllibTransport:
    """The real transport: stdlib `urllib.request`, a fixed User-Agent, one timeout, no retries.

    Redirects are followed explicitly, up to `MAX_REDIRECTS` hops, only for `GET`/`HEAD` (never
    a method that carries a body); each hop increments `Response.requests_made` so
    `BudgetedTransport` can charge the run's request budget for the real number of HTTP
    requests made, not just the one call made into this transport (F10). More than
    `MAX_REDIRECTS` hops raises `RuntimeError` rather than following forever.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoAutoRedirect)

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None) -> Response:
        request_headers = {"User-Agent": USER_AGENT}
        request_headers.update(headers or {})
        hops = 0
        while True:
            request = urllib.request.Request(url, method=method, headers=request_headers)
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    return Response(
                        status=response.status,
                        headers=dict(response.headers),
                        body=response.read(),
                        requests_made=hops + 1,
                    )
            except urllib.error.HTTPError as exc:
                location = exc.headers.get("Location") if exc.headers else None
                if exc.code in _REDIRECT_STATUSES and method in ("GET", "HEAD") and location:
                    hops += 1
                    if hops > MAX_REDIRECTS:
                        raise RuntimeError(f"{url}: more than {MAX_REDIRECTS} redirects") from exc
                    url = urllib.parse.urljoin(url, location)
                    continue
                return Response(
                    status=exc.code, headers=dict(exc.headers or {}), body=exc.read(), requests_made=hops + 1
                )


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
    the calls. F10: a response's `requests_made` beyond `1` (a redirect chain `UrllibTransport`
    followed) is charged too, added to `used` *after* the call -- the pre-call check below still
    only ever sees one call coming, so a redirect chain can overshoot the budget by at most the
    length of that one chain (documented, not fixed further: knowing a call will redirect, and
    by how much, before making it is not possible).
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
        response = self._inner(method, url, headers)
        self.used += response.requests_made - 1
        return response
