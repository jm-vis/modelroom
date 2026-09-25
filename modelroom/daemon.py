"""The local Ollama daemon of this machine: the only transport here that sends a request body.

`modelroom/http.py::Transport` is `(method, url, headers)` and reaches the two registries over
HTTPS; it cannot carry a JSON body, which `POST /api/show` and `POST /api/generate` need. This
module is the second, much smaller transport: `(method, path, body) -> Response` against one
origin, `http://127.0.0.1:11434`, with no proxy handler, no redirect following and no path
other than the five the load test uses and the one pull of step 5 (decided 2026-09-25). Whatever
answers on that port is taken to be this machine's daemon -- a documented limit, not a check this
module can make (CONTRACTS.md, "Load test (stage 1)").

A pull answers with one JSON line per step of it, for as long as the pull takes. That answer is
read one line at a time and handed to a callback, never collected; the last line comes back as
the `Response`.

`LocalDaemon` is the only implementation that opens a connection. A test constructs a
`FixtureDaemon` and passes it in, exactly like `FixtureTransport`/`FixtureRunner` elsewhere.
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Protocol

import modelroom

from .http import Response
from .ollama_local import DEFAULT_BASE_URL

USER_AGENT = f"modelroom/{modelroom.__version__}"

# A short call answers at once; one `generate` of 128 tokens takes minutes on a CPU machine, so
# it gets a limit of its own. A limit that is reached is a fault with a reason, never a traceback.
SHORT_TIMEOUT_SECONDS = 10.0
GENERATE_TIMEOUT_SECONDS = 600.0
# A pull has no limit for the whole of it -- gigabytes take as long as the line takes. What `urllib`
# can limit is each read on the socket: 60 s without a byte is a fault. A daemon that keeps sending
# status lines while no byte of the package arrives is not recognized by this (CONTRACTS.md).
PULL_READ_TIMEOUT_SECONDS = 60.0
# One line of a pull is a status of a few hundred bytes; a line without an end is no answer.
MAX_STREAM_LINE_BYTES = 64 * 1024

VERSION_PATH = "/api/version"
TAGS_PATH = "/api/tags"
PS_PATH = "/api/ps"
SHOW_PATH = "/api/show"
GENERATE_PATH = "/api/generate"
PULL_PATH = "/api/pull"

# Five read and generate calls, and the pull step 5 offers for the first row (decided 2026-09-25).
# Nothing removes a package: there is no delete call here, and a test reads every module for one.
ALLOWED_CALLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", VERSION_PATH),
        ("GET", TAGS_PATH),
        ("GET", PS_PATH),
        ("POST", SHOW_PATH),
        ("POST", GENERATE_PATH),
        ("POST", PULL_PATH),
    }
)

Progress = Callable[[dict], None]

# The same origin `modelroom/http.py::_ALLOWED_HTTP_ORIGINS` names, for the same reason: a port
# other than the daemon's is not the daemon, whatever happens to listen there.
_ALLOWED_ORIGINS: frozenset[tuple[str | None, int | None]] = frozenset({("127.0.0.1", 11434)})


class DaemonError(Exception):
    """The daemon could not be reached, or was asked for something this module never asks for.

    Every failure of a real call ends here with a message a user can act on -- a refused
    connection, a limit that was reached, a body that is not JSON -- so no caller ever sees a
    `urllib` traceback.
    """


class Daemon(Protocol):
    """`(method, path, body) -> Response`, plus the base URL the messages name.

    `progress` is read for a pull only: every line of its answer goes there, and the last one is
    the `Response`.
    """

    base_url: str

    def __call__(
        self, method: str, path: str, body: dict | None = None, progress: Progress | None = None
    ) -> Response: ...


def check_call(method: str, path: str) -> None:
    """Raise `DaemonError` for anything outside `ALLOWED_CALLS` -- checked before a connection."""
    if (method, path) not in ALLOWED_CALLS:
        raise DaemonError(f"refusing to call {method} {path}: this package only calls {_call_list()}")


def _call_list() -> str:
    return ", ".join(f"{method} {path}" for method, path in sorted(ALLOWED_CALLS))


def read_stream(
    readline: Callable[[int], bytes], progress: Progress, where: str, status: int = 200
) -> Response:
    """Read a pull's answer one JSON line at a time; hand each to `progress`, return the last.

    Nothing is collected: a pull of gigabytes is hundreds of lines, and only the last one says how
    it ended. A line that is no UTF-8, no JSON, no JSON object or has no end is a `DaemonError`.
    Reading stops at the first line that ends the pull -- `{"status": "success"}` or one with an
    `error` -- so a daemon that keeps the answer open after it is not waited for, and nothing after
    an error can turn it into a success. What such a line means is the caller's to read.
    """
    last = b""
    while True:
        raw = readline(MAX_STREAM_LINE_BYTES + 1)
        if not raw:
            return Response(status=status, headers={}, body=last)
        if len(raw) > MAX_STREAM_LINE_BYTES:
            raise DaemonError(f"{where}: a line of the answer is longer than {MAX_STREAM_LINE_BYTES} bytes")
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line.decode("utf-8"))
        except (ValueError, RecursionError) as exc:  # `UnicodeDecodeError` is a `ValueError`
            raise DaemonError(f"{where}: a line of the answer is not JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise DaemonError(f"{where}: a line of the answer is not a JSON object")
        progress(payload)
        last = line
        if "error" in payload or payload.get("status") == "success":
            return Response(status=status, headers={}, body=last)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Returns a 3xx to the caller instead of following it (the same trick `http.py` uses)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - stdlib override
        return None


def _build_opener() -> urllib.request.OpenerDirector:
    """An opener with plain HTTP only: no proxy, no HTTPS, no file/ftp/data, no redirects.

    Built by hand rather than with `urllib.request.build_opener`, which auto-fills every default
    handler class it does not see -- including `ProxyHandler`, which would send this machine's
    own daemon traffic to whatever `HTTP_PROXY` names. There is no route off this machine here,
    so there is nothing for a proxy to do.
    """
    opener = urllib.request.OpenerDirector()
    for handler_class in (
        urllib.request.HTTPHandler,
        _NoRedirect,
        urllib.request.HTTPErrorProcessor,
        # `_NoRedirect.redirect_request` returning `None` makes the 3xx handler return `None`
        # too; without a default error handler `opener.open` would hand back `None` instead of
        # the answer. This one raises `HTTPError`, which `__call__` turns into a `Response`.
        urllib.request.HTTPDefaultErrorHandler,
    ):
        opener.add_handler(handler_class())
    return opener


class LocalDaemon:
    """The real daemon transport: one origin, one request per call, no retry.

    `base_url`/`allowed_origins` default to the production pair; a caller overrides them only
    for a test that needs a loopback server on another port, always as an explicit constructor
    argument, never through an environment variable.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        allowed_origins: frozenset[tuple[str | None, int | None]] = _ALLOWED_ORIGINS,
        timeout: float = SHORT_TIMEOUT_SECONDS,
        generate_timeout: float = GENERATE_TIMEOUT_SECONDS,
        pull_timeout: float = PULL_READ_TIMEOUT_SECONDS,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or parsed.path or (parsed.hostname, parsed.port) not in allowed_origins:
            raise DaemonError(f"refusing to talk to {base_url!r}: not a local Ollama daemon origin")
        self.base_url = base_url
        self._timeout = timeout
        self._generate_timeout = generate_timeout
        self._pull_timeout = pull_timeout
        self._opener = _build_opener()

    def timeout_for(self, path: str) -> float:
        """One `generate` run of 128 tokens can take minutes; a pull's limit is per read on the
        socket and not for the whole pull; every other call answers at once."""
        if path == GENERATE_PATH:
            return self._generate_timeout
        return self._pull_timeout if path == PULL_PATH else self._timeout

    def __call__(
        self, method: str, path: str, body: dict | None = None, progress: Progress | None = None
    ) -> Response:
        check_call(method, path)
        request = self._request(method, path, body)
        try:
            with self._opener.open(request, timeout=self.timeout_for(path)) as answer:
                if path == PULL_PATH and answer.status != 200:
                    # `urllib` raises for a 4xx or 5xx only; a 2xx other than 200 is no pull answer
                    # either, and is read like an error's: one line, the rest not waited for
                    # (third-model round, 2026-09-25).
                    body = answer.readline(MAX_STREAM_LINE_BYTES)
                    return Response(status=answer.status, headers=dict(answer.headers), body=body)
                if path == PULL_PATH:
                    # Read inside the `with`: an abort in the callback (Ctrl-C) closes the connection.
                    return read_stream(answer.readline, progress or _ignore, f"{self.base_url}{path}")
                return Response(status=answer.status, headers=dict(answer.headers), body=answer.read())
        except urllib.error.HTTPError as exc:
            # Reading the error body can fail the same ways reading a 200 body can, and this
            # `except` clause is outside the one below -- so it is guarded and closed here.
            try:
                # A pull's error is one line; the rest of an answer that stays open is not waited for.
                body = exc.readline(MAX_STREAM_LINE_BYTES) if path == PULL_PATH else exc.read()
            except (OSError, http.client.HTTPException) as read_exc:
                raise DaemonError(
                    f"{self.base_url}{path}: status {exc.code}, and its body did not arrive ({read_exc})"
                ) from read_exc
            finally:
                exc.close()
            return Response(status=exc.code, headers=dict(exc.headers or {}), body=body)
        # `OSError` covers `URLError` and every socket failure, including the timeout; `ValueError`
        # a URL `urllib` will not build; `HTTPException` a protocol failure that is not an `OSError`
        # at all (`BadStatusLine`, `IncompleteRead` -- a daemon that closes mid-answer).
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise DaemonError(f"{self.base_url}{path}: the Ollama daemon did not answer ({exc})") from exc

    def _request(self, method: str, path: str, body: dict | None) -> urllib.request.Request:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(f"{self.base_url}{path}", data=data, method=method, headers=headers)


class FixtureDaemon:
    """Test daemon: per `(method, path)` one pinned answer, or a sequence of them.

    A single `Response` answers every call for that pair; a list is consumed one call at a
    time, which is what `/api/ps` needs -- its answer changes from observation to observation.
    A `DaemonError` in place of a `Response` is raised instead of returned, so a test can plant
    a daemon that goes away mid-measurement. A pair no test pinned is a test-authoring error and
    surfaces as a `KeyError`, exactly like `FixtureTransport`. `unreachable` makes every call
    fail with that reason -- a machine with no daemon at all.
    """

    def __init__(
        self,
        answers: dict[tuple[str, str], Response | DaemonError | list[Response | DaemonError]],
        unreachable: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._answers = answers
        self._unreachable = unreachable
        self.base_url = base_url
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(
        self, method: str, path: str, body: dict | None = None, progress: Progress | None = None
    ) -> Response:
        self.calls.append((method, path, body))
        if self._unreachable is not None:
            raise DaemonError(self._unreachable)
        check_call(method, path)
        pinned = self._answers[(method, path)]
        answer = pinned.pop(0) if isinstance(pinned, list) else pinned
        if isinstance(answer, DaemonError):
            raise answer
        if path == PULL_PATH and answer.status == 200:
            # A pinned pull is played back line by line, the way `LocalDaemon` reads a real one.
            return read_stream(io.BytesIO(answer.body).readline, progress or _ignore, f"{self.base_url}{path}")
        return answer


def _ignore(_line: dict) -> None:
    """A pull whose caller does not follow its progress still reads it one line at a time."""
