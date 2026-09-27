"""A local vLLM server of this machine: the third transport, and the reader of what it serves.

`modelroom/daemon.py` reaches the Ollama daemon; this module reaches a vLLM server on
`http://127.0.0.1:8000` in exactly the same way -- one origin, no proxy handler, no redirect
following (the opener is `daemon.py`'s own) -- and makes three calls only: `GET /v1/models`,
`GET /metrics` and `POST /v1/chat/completions`. Nothing else is ever sent: no key, no start,
no `/version` or `/health` (CONTRACTS.md, "Local servers and the vLLM load test").

Whatever answers on that port with a model list whose entries are owned by `vllm` is taken to be
this machine's vLLM server -- a documented limit, not a check this module can make. A server
that asks for an API key is said to do so and never read further: modelroom reads an open local
server only (decided 2026-09-27).

`LocalVllmServer` is the only implementation that opens a connection. A test constructs a
`FixtureVllmServer` and passes it in, exactly like `FixtureDaemon`.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

import modelroom

from .daemon import _build_opener
from .http import Response

VLLM_BASE_URL = "http://127.0.0.1:8000"
USER_AGENT = f"modelroom/{modelroom.__version__}"

MODELS_PATH = "/v1/models"
METRICS_PATH = "/metrics"
CHAT_PATH = "/v1/chat/completions"
ALLOWED_CALLS: frozenset[tuple[str, str]] = frozenset(
    {("GET", MODELS_PATH), ("GET", METRICS_PATH), ("POST", CHAT_PATH)}
)

# A short call answers at once; one chat of 128 tokens among 64 at once may take minutes.
SHORT_TIMEOUT_SECONDS = 10.0
CHAT_TIMEOUT_SECONDS = 600.0
# The metrics page of vLLM 0.30.0 is 57 KB; an answer above 4 MiB is no answer of this server.
MAX_ANSWER_BYTES = 4 * 1024 * 1024
KEY_REASON = "the vLLM server asks for an API key; modelroom reads an open local server only"

_ALLOWED_ORIGINS: frozenset[tuple[str | None, int | None]] = frozenset({("127.0.0.1", 8000)})


class VllmServerError(Exception):
    """The server could not be reached, answered too much, or was asked for something never asked."""


class VllmKeyError(VllmServerError):
    """The server answered `401`: it asks for an API key, which modelroom never sends."""


class ServedModelsError(VllmServerError):
    """A vLLM server answered, but its model list names no single model to read."""


@dataclass(frozen=True)
class ServedModel:
    """What the server states it serves: its id, the repository behind it, and its context.

    `stated`: the server's own statement, never measured, never ranked, never provenance.
    """

    id: str
    root: str
    max_model_len: int


class VllmServer(Protocol):
    """`(method, path, body) -> Response`, plus the base URL the messages name."""

    base_url: str

    def __call__(self, method: str, path: str, body: dict | None = None) -> Response: ...


def check_call(method: str, path: str) -> None:
    """Raise `VllmServerError` for anything outside `ALLOWED_CALLS` -- checked before a connection."""
    if (method, path) not in ALLOWED_CALLS:
        calls = ", ".join(f"{m} {p}" for m, p in sorted(ALLOWED_CALLS))
        raise VllmServerError(f"refusing to call {method} {path}: this package only calls {calls}")


def check_origin(base_url: str, allowed: frozenset[tuple[str | None, int | None]] = _ALLOWED_ORIGINS) -> None:
    """Refuse a base URL with another scheme, host or port, or with a path, query, fragment or user."""
    parsed = urllib.parse.urlsplit(base_url)
    extras = parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password
    if parsed.scheme != "http" or extras or (parsed.hostname, parsed.port) not in allowed:
        raise VllmServerError(f"refusing to talk to {base_url!r}: not a local vLLM server origin")


def checked_answer(answer: Response) -> Response:
    """A `401` is the key reason and an answer above the limit a fault; anything else as it came."""
    if answer.status == 401:
        raise VllmKeyError(KEY_REASON)
    if len(answer.body) > MAX_ANSWER_BYTES:
        raise VllmServerError(f"the vLLM server answered more than {MAX_ANSWER_BYTES} bytes")
    return answer


class LocalVllmServer:
    """The real vLLM transport: one origin, one request per call, no retry."""

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        allowed_origins: frozenset[tuple[str | None, int | None]] = _ALLOWED_ORIGINS,
        timeout: float = SHORT_TIMEOUT_SECONDS,
        chat_timeout: float = CHAT_TIMEOUT_SECONDS,
    ) -> None:
        check_origin(base_url, allowed_origins)
        self.base_url = base_url
        self._timeout = timeout
        self._chat_timeout = chat_timeout
        self._opener = _build_opener()

    def timeout_for(self, path: str) -> float:
        """A chat of 128 tokens can take minutes under load; every other call answers at once."""
        return self._chat_timeout if path == CHAT_PATH else self._timeout

    def __call__(self, method: str, path: str, body: dict | None = None) -> Response:
        check_call(method, path)
        request = self._request(method, path, body)
        try:
            with self._opener.open(request, timeout=self.timeout_for(path)) as answer:
                data = answer.read(MAX_ANSWER_BYTES + 1)
                return checked_answer(Response(status=answer.status, headers=dict(answer.headers), body=data))
        except urllib.error.HTTPError as exc:
            try:
                data = exc.read(MAX_ANSWER_BYTES + 1)
            except (OSError, http.client.HTTPException) as read_exc:
                raise VllmServerError(f"{self.base_url}{path}: status {exc.code}, no body ({read_exc})") from read_exc
            finally:
                exc.close()
            return checked_answer(Response(status=exc.code, headers=dict(exc.headers or {}), body=data))
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise VllmServerError(f"{self.base_url}{path}: the vLLM server did not answer ({exc})") from exc

    def _request(self, method: str, path: str, body: dict | None) -> urllib.request.Request:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(f"{self.base_url}{path}", data=data, method=method, headers=headers)


Pinned = Response | VllmServerError | list | Callable[[dict | None], Response]


class FixtureVllmServer:
    """Test server: per `(method, path)` one pinned answer, a sequence of them, or a function of the body.

    A list is consumed one call at a time (`/metrics` changes from batch to batch); a function
    answers a chat by its body, so twenty requests at once each get their own recorded answer. A
    `VllmServerError` in place of an answer is raised. `unreachable` makes every call fail with
    that reason -- a machine with no vLLM server at all.
    """

    def __init__(
        self, answers: dict[tuple[str, str], Pinned], unreachable: str | None = None, base_url: str = VLLM_BASE_URL
    ) -> None:
        self._answers = answers
        self._unreachable = unreachable
        self.base_url = base_url
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method: str, path: str, body: dict | None = None) -> Response:
        self.calls.append((method, path, body))
        if self._unreachable is not None:
            raise VllmServerError(self._unreachable)
        check_call(method, path)
        pinned = self._answers[(method, path)]
        answer = pinned.pop(0) if isinstance(pinned, list) else pinned
        if callable(answer) and not isinstance(answer, Response):
            answer = answer(body)
        if isinstance(answer, VllmServerError):
            raise answer
        return checked_answer(answer)


def read_vllm_models(server: VllmServer) -> ServedModel:
    """The one model a vLLM server states it serves.

    `VllmServerError` when nothing answers, or when what answers is no vLLM model list;
    `VllmKeyError` for a server that asks for a key; `ServedModelsError` when the list names none
    or several models -- then there is no single model a measurement or a fit could be about.
    """
    answer = server("GET", MODELS_PATH)
    if answer.status != 200:
        raise VllmServerError(f"{server.base_url}{MODELS_PATH}: status {answer.status}")
    try:
        payload = answer.json()
    except (ValueError, RecursionError) as exc:
        raise VllmServerError(f"{server.base_url}{MODELS_PATH}: the answer is not JSON") from exc
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise VllmServerError(f"{server.base_url}{MODELS_PATH}: no model list")
    served = [entry for entry in entries if isinstance(entry, dict) and entry.get("owned_by") == "vllm"]
    if not served:
        raise VllmServerError(f"{server.base_url}{MODELS_PATH}: no model owned by vllm")
    if len(served) > 1:
        raise ServedModelsError(f"serves {len(served)} models, modelroom reads one")
    return _served_model(served[0])


def _served_model(entry: dict) -> ServedModel:
    model_id, root, length = entry.get("id"), entry.get("root"), entry.get("max_model_len")
    if not isinstance(model_id, str) or not model_id or not isinstance(root, str) or not root:
        raise ServedModelsError("states no model id or no root")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ServedModelsError(f"states no context for {model_id}")
    return ServedModel(id=model_id, root=root, max_model_len=length)


__all__ = [
    "ALLOWED_CALLS",
    "CHAT_PATH",
    "KEY_REASON",
    "MAX_ANSWER_BYTES",
    "METRICS_PATH",
    "MODELS_PATH",
    "VLLM_BASE_URL",
    "FixtureVllmServer",
    "LocalVllmServer",
    "ServedModel",
    "ServedModelsError",
    "VllmKeyError",
    "VllmServer",
    "VllmServerError",
    "check_call",
    "read_vllm_models",
]
