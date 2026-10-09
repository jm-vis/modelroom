"""A local LM Studio server of this machine: its transport, the reader of its model list, its detection.

`modelroom/vllm_server.py` reaches a vLLM server; this module reaches LM Studio on `http://127.0.0.1:1234`
in the same way -- one origin (judged by that module's own `check_origin`), no proxy handler, no redirect
following (the opener is `daemon.py`'s). Two calls are allowed: `GET /api/v1/models` (10 s) and
`POST /v1/chat/completions` (600 s), the one request of the measurement (`loadtest_lmstudio`). Nothing else
is ever sent: no key, no load, unload or download, no `/v1/models`, no `/metrics` (CONTRACTS.md, "Local
servers and the vLLM load test", LM Studio).

A chat that names a model unloaded a moment ago may make LM Studio load it again; modelroom therefore
reads the list before and after every request and never asks for a load itself. At the chat a `401`, a
`403`, and a body that cannot be read or is too large are not raised but handed on as a `Response` that
carries the status (`x-modelroom-body: unread` or `oversized` for the body), so the measurement can name
the row of its table; only a limit of time is raised as `LmStudioTimeoutError`. **modelroom runs `lms` at
one place only**, the download after a yes in `lmstudio_get.py` (AP9-LS-G): this module and detection look
for the program in the `PATH` and in the home folder and start nothing;
`tests/test_lmstudio_no_process.py` guards it.

The reader gives no package identity: the REST list names a publisher and a key, never a repository or a
file. `LocalLmStudioServer` is the only implementation that opens a connection; a test constructs a
`FixtureLmStudioServer`, like `FixtureVllmServer`.
"""

from __future__ import annotations

import http.client
import json
import re
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .daemon import _build_opener
from .http import Response
from .vllm_server import (
    CHAT_PATH,
    CHAT_TIMEOUT_SECONDS,
    MAX_ANSWER_BYTES,
    SHORT_TIMEOUT_SECONDS,
    USER_AGENT,
    VllmServerError,
    check_origin,
)

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234"
MODELS_PATH = "/api/v1/models"
ALLOWED_CALLS: frozenset[tuple[str, str]] = frozenset({("GET", MODELS_PATH), ("POST", CHAT_PATH)})
# The header of a chat answer whose body is not there: `unread` (it failed to read) or `oversized`.
BODY_HEADER = "x-modelroom-body"
BODY_UNREAD = "unread"
BODY_OVERSIZED = "oversized"
KEY_REASON = "the LM Studio server asks for an API token; modelroom reads an open local server only"

_ALLOWED_ORIGINS: frozenset[tuple[str | None, int | None]] = frozenset({("127.0.0.1", 1234)})
# An instance id and a key are shown on the screen: plain ASCII only.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@/:-]{0,127}")


class LmStudioServerError(Exception):
    """The server could not be reached, answered too much, was asked for something never asked, or is no LM Studio."""


class LmStudioKeyError(LmStudioServerError):
    """The list answered `401` or `403`: the server asks for an API token, which modelroom never sends."""


class LmStudioTimeoutError(LmStudioServerError):
    """A call reached its limit of time (10 s for the list, 600 s for the chat) at any point of the call."""


@dataclass(frozen=True)
class LoadedModel:
    """One loaded instance of a measurable (`llm`, `gguf`) model: its id, its model key and what it runs with.

    `parallel` is `None` where the server states none. No repository, no file, no size: REST names none.
    """

    instance_id: str
    key: str
    context_length: int
    parallel: int | None


@dataclass(frozen=True)
class LmStudioModels:
    """What one read of the list found: the loaded instances, the measurable models, every key.

    `keys` holds the `key` of every entry that is an object, before any filter, so an identifier equal to
    the key of a model that was left out is still known as a model's.
    """

    loaded: tuple[LoadedModel, ...]
    downloaded: int
    keys: frozenset[str]


class LmStudioServer(Protocol):
    """`(method, path, body) -> Response`, plus the base URL the messages name."""

    base_url: str

    def __call__(self, method: str, path: str, body: dict | None = None) -> Response: ...


def check_call(method: str, path: str) -> None:
    """Raise `LmStudioServerError` for anything outside `ALLOWED_CALLS` -- checked before a connection."""
    if (method, path) not in ALLOWED_CALLS:
        calls = ", ".join(f"{m} {p}" for m, p in sorted(ALLOWED_CALLS))
        raise LmStudioServerError(f"refusing to call {method} {path}: this package only calls {calls}")


def _check_origin(base_url: str, allowed: frozenset[tuple[str | None, int | None]]) -> None:
    """The vLLM transport's own rule, with this server's name in the message."""
    try:
        check_origin(base_url, allowed)
    except VllmServerError as exc:
        raise LmStudioServerError(f"refusing to talk to {base_url!r}: not a local LM Studio server origin") from exc


def _marked(answer: Response, marker: str) -> Response:
    """The status and the headers of `answer`, an empty body and the marker for why the body is gone."""
    return Response(status=answer.status, headers={**answer.headers, BODY_HEADER: marker}, body=b"")


def checked_answer(answer: Response, path: str) -> Response:
    """The list: a `401` or `403` is the key reason and an answer above the limit a fault. The chat: its status
    always stays (no key reason there), and an answer above the limit loses its body, marked `oversized`."""
    if path == MODELS_PATH and answer.status in (401, 403):
        raise LmStudioKeyError(KEY_REASON)
    if len(answer.body) > MAX_ANSWER_BYTES:
        if path == CHAT_PATH:
            return _marked(answer, BODY_OVERSIZED)
        raise LmStudioServerError(f"the LM Studio server answered more than {MAX_ANSWER_BYTES} bytes")
    return answer


def _is_timeout(exc: BaseException) -> bool:
    """urllib raises the limit of time as a `TimeoutError` at the read, and as a `URLError` whose reason is one
    at the connection: both are the limit."""
    return isinstance(exc, TimeoutError) or (isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError))


class LocalLmStudioServer:
    """The real LM Studio transport: one origin, one request per call, no retry."""

    def __init__(
        self,
        base_url: str = LMSTUDIO_BASE_URL,
        timeout: float = SHORT_TIMEOUT_SECONDS,
        chat_timeout: float = CHAT_TIMEOUT_SECONDS,
    ) -> None:
        _check_origin(base_url, _ALLOWED_ORIGINS)  # the module's list, read at the call; never an argument
        self.base_url = base_url
        self._timeout = timeout
        self._chat_timeout = chat_timeout
        self._opener = _build_opener()

    def timeout_for(self, path: str) -> float:
        """A chat of 128 tokens can take minutes; the list answers at once."""
        return self._chat_timeout if path == CHAT_PATH else self._timeout

    def __call__(self, method: str, path: str, body: dict | None = None) -> Response:
        check_call(method, path)
        _check_origin(self.base_url, _ALLOWED_ORIGINS)  # `base_url` is public: judged again at every call
        request = self._request(method, path, body)
        try:
            with self._opener.open(request, timeout=self.timeout_for(path)) as answer:
                return self._read_answer(answer, path)
        except urllib.error.HTTPError as exc:
            return self._read_error(exc, path)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise self._failure(exc, path, "the LM Studio server did not answer") from exc

    def _failure(self, exc: BaseException, path: str, what: str) -> LmStudioServerError:
        """A limit of time is `LmStudioTimeoutError` wherever it was reached, anything else a server error."""
        kind = LmStudioTimeoutError if _is_timeout(exc) else LmStudioServerError
        return kind(f"{self.base_url}{path}: {what} ({exc})")

    def _read_answer(self, answer: Any, path: str) -> Response:
        """The body of an answer. A `401` or `403` of the list is decided before the body is read. At the chat a
        body that fails to read leaves the status and a marker."""
        headers = dict(answer.headers)
        if path == MODELS_PATH and answer.status in (401, 403):
            raise LmStudioKeyError(KEY_REASON)
        try:
            data = answer.read(MAX_ANSWER_BYTES + 1)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            if path != CHAT_PATH or _is_timeout(exc):
                raise self._failure(exc, path, "the answer could not be read") from exc
            return _marked(Response(status=answer.status, headers=headers), BODY_UNREAD)
        return checked_answer(Response(status=answer.status, headers=headers, body=data), path)

    def _read_error(self, exc: urllib.error.HTTPError, path: str) -> Response:
        """An error status. A `401` or `403` of the list says it all: its body is not read (it may not be
        readable). Elsewhere the body is read; at the chat one that fails to read leaves the status."""
        headers = dict(exc.headers or {})
        try:
            if exc.code in (401, 403) and path == MODELS_PATH:
                raise LmStudioKeyError(KEY_REASON) from exc
            try:
                data = exc.read(MAX_ANSWER_BYTES + 1)
            except (OSError, ValueError, http.client.HTTPException) as read_exc:
                if path != CHAT_PATH or _is_timeout(read_exc):
                    raise self._failure(read_exc, path, f"status {exc.code}, no body") from read_exc
                return _marked(Response(status=exc.code, headers=headers), BODY_UNREAD)
            return checked_answer(Response(status=exc.code, headers=headers, body=data), path)
        finally:
            exc.close()

    def _request(self, method: str, path: str, body: dict | None) -> urllib.request.Request:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(f"{self.base_url}{path}", data=data, method=method, headers=headers)


Pinned = Response | LmStudioServerError | list | Callable[[dict | None], Response]


class FixtureLmStudioServer:
    """Test server: per `(method, path)` one pinned answer, a sequence of them, or a function of the body.

    A list is consumed one call at a time; a function answers by the body of the call. An
    `LmStudioServerError` in place of an answer is raised. `unreachable` makes every call fail with that
    reason -- a machine with no LM Studio server at all. Every call is kept in `calls`.
    """

    def __init__(
        self, answers: dict[tuple[str, str], Pinned], unreachable: str | None = None, base_url: str = LMSTUDIO_BASE_URL
    ) -> None:
        self._answers = answers
        self._unreachable = unreachable
        self.base_url = base_url
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method: str, path: str, body: dict | None = None) -> Response:
        self.calls.append((method, path, body))
        if self._unreachable is not None:
            raise LmStudioServerError(self._unreachable)
        check_call(method, path)
        pinned = self._answers[(method, path)]
        answer = pinned.pop(0) if isinstance(pinned, list) else pinned
        if callable(answer) and not isinstance(answer, Response):
            answer = answer(body)
        if isinstance(answer, LmStudioServerError):
            raise answer
        return checked_answer(answer, path)


# --- the model list -----------------------------------------------------------------------------------


def read_lmstudio_models(server: LmStudioServer) -> LmStudioModels:
    """`GET /api/v1/models`, read in three steps; `LmStudioServerError` where it is no LM Studio list.

    1. `keys`: the key of every entry that is an object, before any filter.
    2. The structure, which is the version gate (no version is ever read): an entry is valid with a `key`
       string, a `type` string, a `size_bytes` whole number and a `loaded_instances` list. `models` that is
       no list, or a list with entries but none valid, is a fault; an empty list is a server with nothing
       to measure, and so is a list of embedding or MLX entries only.
    3. The measurable ones: `type` `llm`, `format` `gguf`, a safe key; their loaded instances with a safe
       id and a `config.context_length` above 0.
    """
    answer = server("GET", MODELS_PATH)
    where = f"{server.base_url}{MODELS_PATH}"
    if answer.status != 200:
        raise LmStudioServerError(f"{where}: status {answer.status}")
    try:
        payload = answer.json()
    except (ValueError, RecursionError) as exc:
        raise LmStudioServerError(f"{where}: the answer is not JSON") from exc
    entries = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise LmStudioServerError(f"{where}: no LM Studio model list")
    objects = [entry for entry in entries if isinstance(entry, dict)]
    keys = frozenset(entry["key"] for entry in objects if isinstance(entry.get("key"), str))
    valid = [entry for entry in objects if _has_structure(entry)]
    if entries and not valid:
        raise LmStudioServerError(f"{where}: no entry with the shape of an LM Studio model")
    measurable = [
        entry for entry in valid if entry["type"] == "llm" and entry.get("format") == "gguf" and _safe(entry["key"])
    ]
    loaded = tuple(model for entry in measurable for model in _instances(entry))
    return LmStudioModels(loaded=loaded, downloaded=len(measurable), keys=keys)


def _has_structure(entry: dict) -> bool:
    size = entry.get("size_bytes")
    return (
        isinstance(entry.get("key"), str)
        and isinstance(entry.get("type"), str)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and isinstance(entry.get("loaded_instances"), list)
    )


def _safe(name: Any) -> bool:
    return isinstance(name, str) and _SAFE_NAME.fullmatch(name) is not None


def _whole(value: Any, least: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


def _instances(entry: dict) -> list[LoadedModel]:
    found = []
    for item in entry["loaded_instances"]:
        config = item.get("config") if isinstance(item, dict) else None
        if not isinstance(config, dict) or not _safe(item.get("id")) or not _whole(config.get("context_length"), 1):
            continue
        parallel = config.get("parallel") if "parallel" in config else None
        if "parallel" in config and not _whole(parallel, 1):  # absent is allowed, present has to be a number
            continue
        found.append(LoadedModel(item["id"], entry["key"], config["context_length"], parallel))
    return found


# --- detection ----------------------------------------------------------------------------------------


def _program_name(system: str) -> str:
    return "lms.exe" if system.startswith("win") else "lms"


def lms_call(
    *,
    system: str = sys.platform,
    which: Callable[[str], str | None] = shutil.which,
    home: Path | None = None,
    is_file: Callable[[Path], bool] = lambda path: path.is_file(),
) -> str | None:
    """How a later command could name the program: `path`, `home` (`~/.lmstudio/bin` only) or `None`.

    A path search and a file test, nothing else: the program is never started.
    """
    if which("lms") is not None:
        return "path"
    folder = (home if home is not None else Path.home()) / ".lmstudio" / "bin"
    return "home" if is_file(folder / _program_name(system)) else None


def detect_lmstudio(
    server: Any = None,
    *,
    system: str = sys.platform,
    which: Callable[[str], str | None] = shutil.which,
    home: Path | None = None,
    is_file: Callable[[Path], bool] = lambda path: path.is_file(),
) -> str:
    """`LMSTUDIO.detect`: `serving`, `asks_key`, `installed` or `absent` for the machine this runs on.

    A server whose list reads is `serving`, one that asks for a token `asks_key` (never read further);
    otherwise the program in the `PATH` or in `~/.lmstudio/bin` makes it `installed`, else `absent`. No
    platform is excluded. `server` is the LM Studio transport (`None`: not asked); `which`, `home` and
    `is_file` are the machine, passed in by a test. Never a process.
    """
    if server is not None:
        try:
            read_lmstudio_models(server)
            return "serving"
        except LmStudioKeyError:
            return "asks_key"
        except LmStudioServerError:
            pass
    return "installed" if lms_call(system=system, which=which, home=home, is_file=is_file) is not None else "absent"


__all__ = [
    "ALLOWED_CALLS",
    "BODY_HEADER",
    "BODY_OVERSIZED",
    "BODY_UNREAD",
    "CHAT_PATH",
    "KEY_REASON",
    "LMSTUDIO_BASE_URL",
    "MODELS_PATH",
    "FixtureLmStudioServer",
    "LmStudioKeyError",
    "LmStudioModels",
    "LmStudioServer",
    "LmStudioServerError",
    "LmStudioTimeoutError",
    "LoadedModel",
    "LocalLmStudioServer",
    "check_call",
    "detect_lmstudio",
    "lms_call",
    "read_lmstudio_models",
]
