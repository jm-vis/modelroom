"""A local LM Studio server of this machine: its transport, the reader of its model list, its detection.

`modelroom/vllm_server.py` reaches a vLLM server; this module reaches LM Studio on `http://127.0.0.1:1234`
in the same way -- one origin (judged by that module's own `check_origin`), no proxy handler, no redirect
following (the opener is `daemon.py`'s). One call is allowed: `GET /api/v1/models`. Nothing else is ever
sent: no key, no chat, no load, unload or download, no `/v1/models`, no `/metrics` (CONTRACTS.md, "Local
servers and the vLLM load test", LM Studio).

**In 0.2.0 the transport cannot send a chat**: a request that names a model unloaded a moment ago may make
LM Studio load it again, and the recording of that case proves neither a refusal nor a load, so LM Studio
is not measured. The call returns with a later sub-package that proves a measurement safe. **modelroom never runs `lms`**: detection looks for the program in the `PATH` and in the home
folder and starts nothing; `tests/test_lmstudio_no_process.py` guards it.

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
    MAX_ANSWER_BYTES,
    SHORT_TIMEOUT_SECONDS,
    USER_AGENT,
    VllmServerError,
    check_origin,
)

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234"
MODELS_PATH = "/api/v1/models"
ALLOWED_CALLS: frozenset[tuple[str, str]] = frozenset({("GET", MODELS_PATH)})
KEY_REASON = "the LM Studio server asks for an API token; modelroom reads an open local server only"

_ALLOWED_ORIGINS: frozenset[tuple[str | None, int | None]] = frozenset({("127.0.0.1", 1234)})
# An instance id and a key are shown on the screen: plain ASCII only.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@/:-]{0,127}")


class LmStudioServerError(Exception):
    """The server could not be reached, answered too much, was asked for something never asked, or is no LM Studio."""


class LmStudioKeyError(LmStudioServerError):
    """The server answered `401` or `403`: it asks for an API token, which modelroom never sends."""


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


def checked_answer(answer: Response) -> Response:
    """A `401` or `403` is the key reason and an answer above the limit a fault; anything else as it came."""
    if answer.status in (401, 403):
        raise LmStudioKeyError(KEY_REASON)
    if len(answer.body) > MAX_ANSWER_BYTES:
        raise LmStudioServerError(f"the LM Studio server answered more than {MAX_ANSWER_BYTES} bytes")
    return answer


class LocalLmStudioServer:
    """The real LM Studio transport: one origin, one request per call, no retry."""

    def __init__(
        self,
        base_url: str = LMSTUDIO_BASE_URL,
        timeout: float = SHORT_TIMEOUT_SECONDS,
    ) -> None:
        _check_origin(base_url, _ALLOWED_ORIGINS)  # the module's list, read at the call; never an argument
        self.base_url = base_url
        self._timeout = timeout
        self._opener = _build_opener()

    def __call__(self, method: str, path: str, body: dict | None = None) -> Response:
        check_call(method, path)
        _check_origin(self.base_url, _ALLOWED_ORIGINS)  # `base_url` is public: judged again at every call
        request = self._request(method, path, body)
        try:
            with self._opener.open(request, timeout=self._timeout) as answer:
                data = answer.read(MAX_ANSWER_BYTES + 1)
                return checked_answer(Response(status=answer.status, headers=dict(answer.headers), body=data))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):  # the status says it all: the body is not read (it may not be readable)
                exc.close()
                raise LmStudioKeyError(KEY_REASON) from exc
            try:
                data = exc.read(MAX_ANSWER_BYTES + 1)
            except (OSError, http.client.HTTPException) as read_exc:
                raise LmStudioServerError(f"{self.base_url}{path}: status {exc.code}, no body ({read_exc})") from read_exc
            finally:
                exc.close()
            return checked_answer(Response(status=exc.code, headers=dict(exc.headers or {}), body=data))
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise LmStudioServerError(f"{self.base_url}{path}: the LM Studio server did not answer ({exc})") from exc

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
        return checked_answer(answer)


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
    "KEY_REASON",
    "LMSTUDIO_BASE_URL",
    "MODELS_PATH",
    "FixtureLmStudioServer",
    "LmStudioKeyError",
    "LmStudioModels",
    "LmStudioServer",
    "LmStudioServerError",
    "LoadedModel",
    "LocalLmStudioServer",
    "check_call",
    "detect_lmstudio",
    "lms_call",
    "read_lmstudio_models",
]
