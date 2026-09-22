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

    A 3xx response is returned exactly like any other status (R5): following it is
    `RedirectingTransport`'s job, never something a `Response` itself accounts for.
    """

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


_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
MAX_REDIRECTS = 5


class _NoAutoRedirect(urllib.request.HTTPRedirectHandler):
    """Disables `urllib.request`'s own, uncounted redirect following.

    Returning `None` from `redirect_request` makes a 3xx surface as an ordinary response
    (`status`/`headers` read off the `HTTPError`) instead of being silently followed --
    `UrllibTransport` makes exactly one request per call either way (R5); following a chain is
    `RedirectingTransport`'s job, composed around this transport by the caller (`run_fetch`).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - stdlib override
        return None


class UrllibTransport:
    """The real transport: stdlib `urllib.request`, a fixed User-Agent, one timeout, no retries.

    Makes exactly one HTTP request per call (R5) and returns a 3xx response exactly like any
    other status -- `urllib.request`'s own automatic redirect following is disabled
    (`_NoAutoRedirect`) so a 3xx never gets silently resolved before this transport can even see
    it. Following a redirect chain is `RedirectingTransport`'s job, not this transport's: that
    inversion is what lets `run_fetch` compose `RedirectingTransport(BudgetedTransport(...))` so
    every hop of a chain is booked against the run's request budget *before* it is made, failure
    paths included, rather than only being counted after the whole chain already resolved.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoAutoRedirect)

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None) -> Response:
        request_headers = {"User-Agent": USER_AGENT}
        request_headers.update(headers or {})
        request = urllib.request.Request(url, method=method, headers=request_headers)
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return Response(status=response.status, headers=dict(response.headers), body=response.read())
        except urllib.error.HTTPError as exc:
            return Response(status=exc.code, headers=dict(exc.headers or {}), body=exc.read())


class RedirectingTransport:
    """Follows a redirect chain by calling `inner` once per hop (R5).

    Only a `GET`/`HEAD` response with a 3xx status in `_REDIRECT_STATUSES` and a `Location`
    header is followed -- never a method that carries a body, and never a 3xx with no `Location`
    to follow (both are returned to the caller exactly as received). `urllib.parse.urljoin`
    resolves a relative `Location` against the hop it came from. A chain that has not resolved
    within `MAX_REDIRECTS` hops (the very first request counts as hop 1) raises `RuntimeError`
    rather than following forever; the hop that would exceed the limit is never made, since it
    could only be discovered to be one *more* redirect by making it. Composed around a
    `BudgetedTransport` in `run_fetch` (this class's `inner`), so every hop this class makes --
    including the one whose response turns out to matter for `_REDIRECT_STATUSES` -- passes
    through the budget check first, charging the run's request budget one hop at a time and
    raising `BudgetExhaustedError` mid-chain exactly as it would for any other call.
    """

    def __init__(self, inner: Transport) -> None:
        self._inner = inner

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None) -> Response:
        current_url = url
        for _ in range(MAX_REDIRECTS):
            response = self._inner(method, current_url, headers)
            if response.status not in _REDIRECT_STATUSES or method not in ("GET", "HEAD"):
                return response
            location = response.header("Location")
            if not location:
                return response
            current_url = urllib.parse.urljoin(current_url, location)
        raise RuntimeError(f"{url}: more than {MAX_REDIRECTS} redirects")


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
    the calls. R5: each call is exactly one request, charged before it is made -- a redirect
    chain is `RedirectingTransport`'s job (composed as this transport's *outer* layer in
    `run_fetch`: `RedirectingTransport(BudgetedTransport(...))`), so every hop of a chain already
    arrives here as its own separate call and is booked (and can exhaust the budget) one hop at a
    time, never charged only after the fact.
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
