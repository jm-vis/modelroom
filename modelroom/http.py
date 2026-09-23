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

# P2-1 (fix-round 5): the only HTTPS hosts this package ever talks to (read from the modules
# that build the URLs, not guessed) -- `hf.py::HF_API`, `ollama.py::TAGS_URL`,
# `ollama.py::MANIFEST_URL`. AGENTS.md's own "Security, definition of done" names the local
# Ollama daemon as the one legitimate HTTP (not HTTPS) destination, at
# `ollama_local.py::DEFAULT_BASE_URL`'s host.
_ALLOWED_HTTPS_HOSTS: frozenset[str] = frozenset({"huggingface.co", "ollama.com", "registry.ollama.ai"})
# R7-4 (fix-round 6): an origin (host, port), not just a host -- `_check_allowed` used to accept
# any port on `127.0.0.1` (probe: `http://127.0.0.1:59999/x` was accepted), so a poisoned
# `Location`/`Link` header could reach any local port a caller happened to have something
# listening on, not only the real Ollama daemon. `ollama_local.py::DEFAULT_BASE_URL` names the
# same origin (`test_ollama_local_default_base_url_is_in_the_production_http_allow_list`).
_ALLOWED_HTTP_ORIGINS: frozenset[tuple[str | None, int | None]] = frozenset({("127.0.0.1", 11434)})


class TransportSecurityError(Exception):
    """`UrllibTransport` refuses to open a URL whose scheme/host is not on its allow-list.

    Every URL this package ever hands a `Transport` that did not originate in its own
    configuration or source code -- the `Location` header a 302 answer carries
    (`RedirectingTransport` calls `inner` again with it, so it passes back through this same
    check on the next hop) and the `Link: rel="next"` header `hf.py::_fetch_tree` follows
    verbatim -- eventually reaches `UrllibTransport.__call__`, since that is the only
    implementation that ever opens a real connection (composed as the innermost layer in
    production: `RedirectingTransport(BudgetedTransport(UrllibTransport()))`). This is the one
    choke point that makes AGENTS.md's "HTTPS to huggingface.co/ollama.com/registry.ollama.ai,
    HTTP to a local Ollama daemon" attack-surface promise true regardless of what an upstream
    caller forgot to check: a `file://`/`ftp://`/`data:` URL, or an HTTPS URL to any other host,
    is refused here before anything is opened.
    """


def _check_allowed(
    url: str, allowed_https_hosts: frozenset[str], allowed_http_origins: frozenset[tuple[str | None, int | None]]
) -> None:
    """Raise `TransportSecurityError` unless `url` is `https://<allowed host>` or `http://<allowed
    loopback origin>` -- checked before `UrllibTransport` ever opens a connection.

    R7-4 (fix-round 6): the local allow-list is matched as an origin, `(hostname, port)`, not by
    hostname alone -- a URL with no explicit port parses to `parsed.port is None`, which is not a
    member of `allowed_http_origins` (every entry names an explicit port), so it is refused
    rather than silently matching "any port".
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme == "https" and parsed.hostname in allowed_https_hosts:
        return
    if parsed.scheme == "http" and (parsed.hostname, parsed.port) in allowed_http_origins:
        return
    raise TransportSecurityError(
        f"refusing to open {url!r}: scheme {parsed.scheme!r} / host {parsed.hostname!r} is not on the allow-list"
    )


def _build_opener() -> urllib.request.OpenerDirector:
    """An `OpenerDirector` carrying exactly the handlers this transport needs.

    `urllib.request.build_opener(_NoAutoRedirect)` (the previous implementation) looks like an
    explicit handler list but is not one: `build_opener` auto-fills every *default* handler
    class it does not see represented (directly or by subclass) among its arguments, so leaving
    `FileHandler`/`FTPHandler`/`DataHandler` out of the call still adds them back in -- measured
    directly (fix-round 5 P2-1 probe): `UrllibTransport()("GET", "file:///.../pyvenv.cfg")`
    returned status `None` and 178 bytes of a local file. Building an `OpenerDirector` directly
    and registering only `HTTPHandler`/`HTTPSHandler`/`_NoAutoRedirect`/`HTTPErrorProcessor` has
    no such auto-fill: a `file://`/`ftp://`/`data:` URL is structurally impossible here even if
    `_check_allowed` were ever bypassed, never merely rejected by a check that a bug could skip.

    R7-5 (fix-round 6): the old `build_opener(_NoAutoRedirect)` call also added a `ProxyHandler`
    for free (one of `build_opener`'s auto-filled defaults); this explicit handler list dropped
    that too, silently losing proxy support. `urllib.request.ProxyHandler()` (constructed with no
    arguments, its stdlib default: reads `HTTP_PROXY`/`HTTPS_PROXY` from the environment, honors
    `NO_PROXY`) is added back explicitly. `_check_allowed` in `__call__` below still runs against
    the request's own TARGET url, before `self._opener.open` ever consults a proxy, so a proxy
    can never widen this transport's allow-list -- it can only change how an already-allowed
    request reaches its target. An operator who wants the local Ollama daemon reached directly
    even behind a proxy sets `NO_PROXY=127.0.0.1`; that is a deployment decision, never one this
    package makes on its own.
    """
    opener = urllib.request.OpenerDirector()
    for handler_class in (
        urllib.request.ProxyHandler,
        urllib.request.HTTPHandler,
        urllib.request.HTTPSHandler,
        _NoAutoRedirect,
        urllib.request.HTTPErrorProcessor,
        # `_NoAutoRedirect.redirect_request` returning `None` makes `http_error_302` itself
        # return `None` (stdlib source: `if new is None: return`) -- with no
        # `HTTPDefaultErrorHandler` to fall back to, `OpenerDirector.error` would then return
        # `None` all the way out of `opener.open`, not the 3xx response. This handler's
        # `http_error_default` raises `HTTPError` instead, which `__call__` below already
        # catches and turns into an ordinary `Response` -- the same path a genuine 4xx/5xx
        # takes.
        urllib.request.HTTPDefaultErrorHandler,
    ):
        opener.add_handler(handler_class())
    return opener


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
    paths included, rather than only being counted after the whole chain already resolved --
    and, since every hop calls back into this same transport (P2-1), every hop is also checked
    against the allow-list again, so a poisoned redirect target is refused on the hop that would
    have followed it, never only on the first.

    `allowed_https_hosts`/`allowed_http_origins` default to the production allow-list
    (`_ALLOWED_HTTPS_HOSTS`/`_ALLOWED_HTTP_ORIGINS`); a caller overrides them only to relax the
    check for a test (e.g. a loopback server on a non-default port), always as an explicit
    constructor argument here, never through an environment variable or a module global.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        allowed_https_hosts: frozenset[str] = _ALLOWED_HTTPS_HOSTS,
        allowed_http_origins: frozenset[tuple[str | None, int | None]] = _ALLOWED_HTTP_ORIGINS,
    ) -> None:
        self._timeout = timeout
        self._allowed_https_hosts = allowed_https_hosts
        self._allowed_http_origins = allowed_http_origins
        self._opener = _build_opener()

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None) -> Response:
        _check_allowed(url, self._allowed_https_hosts, self._allowed_http_origins)
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
