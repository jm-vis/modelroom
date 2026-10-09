"""Fixture helpers for the LM Studio tests (AP9-LS): the recorded answers as servers, and the process trap.

`lmstudio_servers` builds a `FixtureLmStudioServer` from the recordings of `tests/fixtures/lmstudio_*.json`
(the model lists, and the chats where a test gives some) and counts the chats it gets, so a test can say
how many were sent. `trap_processes` fails the test on calls
through the patched `subprocess.Popen` and `os.system` functions, and `fake_lms_on_path` puts a program
called `lms` in the `PATH` that leaves a mark when it runs. Not a test module (no `test_` prefix).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

from fixture_http import json_response
from modelroom.http import Response
from modelroom.intro import UNICODE_GLYPHS
from modelroom.lmstudio_server import CHAT_PATH, MODELS_PATH, FixtureLmStudioServer


def _answer(item):
    """A recording by its file name, a body built by the test as a dict, or an answer as it is (a `Response`,
    an `LmStudioServerError` to raise, a function of the request body)."""
    if isinstance(item, str):
        return json_response(item)
    if isinstance(item, dict):
        return Response(status=200, body=json.dumps(item).encode("utf-8"))
    return item


def lmstudio_servers(lists: list, chats: list | None = None) -> FixtureLmStudioServer:
    """A server whose model list answers `lists` one after the other (the last one stays), and whose chat
    answers `chats`, one entry per chat (a chat beyond the entries fails the test with an `IndexError`).

    An entry is a recording by its file name, a dict, or an answer as it is (see `_answer`). Without `chats`
    the server has no answer to a chat and a chat fails the test with a `KeyError`. Every call is kept in
    `calls`; `chat_calls` and `list_calls` count what it was sent."""
    answers = [_answer(item) for item in lists]

    def next_list(_body):
        return answers.pop(0) if len(answers) > 1 else answers[0]

    pinned: dict = {("GET", MODELS_PATH): next_list}
    if chats is not None:
        pinned[("POST", CHAT_PATH)] = [_answer(item) for item in chats]
    return FixtureLmStudioServer(pinned)


def chat_to(instance_id: str, reply: str | Response = "lmstudio_chat_one.json"):
    """An answer to a chat that checks the request: only the model `instance_id` is loaded, so a body that
    names another model (the `key` of the model, say) gets the 404 a server gives for a model it does not know."""

    response = json_response(reply) if isinstance(reply, str) else reply  # read now, not at the request

    def answer(body):
        named = (body or {}).get("model")
        if named != instance_id:
            text = json.dumps({"error": {"message": f"model {named!r} is not loaded"}})
            return Response(status=404, body=text.encode("utf-8"))
        return response

    return answer


def run_of(server: FixtureLmStudioServer | None):
    """A stand-in for the guided run, as far as LM Studio reads it: `servers` (its transport, what the start
    screen found, what step 4 measured), `screen.note`, which keeps every note in `notes`, `screen.glyphs`
    (the drawn set) and `failed`, which keeps every message in `failures`."""
    from modelroom.runtimes import ServerRun

    notes: list[str] = []
    failures: list[str] = []
    transports = {} if server is None else {"lmstudio": server}
    screen = types.SimpleNamespace(note=notes.append, glyphs=UNICODE_GLYPHS)
    return types.SimpleNamespace(
        servers=ServerRun(transports=transports), screen=screen, notes=notes, failed=failures.append, failures=failures
    )


class FakeAnswer:
    """What an opener's `open` hands back: a context manager with a status, headers and a body to read, or
    an error that the read raises (a limit of time, say)."""

    def __init__(self, status: int = 200, body: bytes = b"", headers: dict | None = None, read_error=None) -> None:
        self.status, self.headers, self._body, self._error = status, headers or {}, body, read_error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        self.closed = True
        return False

    def read(self, size: int = -1) -> bytes:
        if self._error is not None:
            raise self._error
        return self._body if size < 0 else self._body[:size]


class _ErrorBody:
    """The body of an HTTP error: its bytes, or an error on the read; `reads` counts the reads."""

    def __init__(self, body: bytes, read_error) -> None:
        self._body, self._error, self.reads, self.closed = body, read_error, 0, False

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self._error is not None:
            raise self._error
        return self._body if size < 0 else self._body[:size]

    def close(self) -> None:
        self.closed = True


def http_error(status: int, body: bytes = b"", read_error=None):
    """An `HTTPError` of `status`, the way `opener.open` raises one, with its body reader (`.fp.reads`)."""
    import urllib.error

    return urllib.error.HTTPError("http://127.0.0.1:1234/x", status, "error", {}, _ErrorBody(body, read_error))


class ScriptedOpener:
    """An opener for `LocalLmStudioServer._opener`: per path an answer (a `FakeAnswer`), an exception to raise,
    or a function of the request. `opened` keeps `(method, path, timeout)` of every call."""

    def __init__(self, replies: dict) -> None:
        self.replies, self.opened = replies, []

    def open(self, request, timeout=None):
        from urllib.parse import urlsplit

        path = urlsplit(request.full_url).path
        self.opened.append((request.get_method(), path, timeout))
        reply = self.replies[path]
        if callable(reply):
            reply = reply(request)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def scripted_server(replies: dict):
    """A `LocalLmStudioServer` whose connection is a `ScriptedOpener`: nothing is opened."""
    from modelroom.lmstudio_server import LocalLmStudioServer

    server = LocalLmStudioServer()
    server._opener = ScriptedOpener(replies)
    return server


def chat_calls(server: FixtureLmStudioServer) -> list:
    """The chats a fixture server was sent."""
    return [call for call in server.calls if call[0] == "POST"]


def list_calls(server: FixtureLmStudioServer) -> list:
    return [call for call in server.calls if call[0] == "GET"]


def trap_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every `subprocess.Popen` (so `subprocess.run` and `os.popen`) and every `os.system` fails the test,
    whatever its arguments: a call by `lms` in the first argument, by an absolute path or by a shell alike."""

    def refuse(*args, **kwargs):
        pytest.fail(f"a process was started: {args!r}", pytrace=False)

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)


def trap_processes_except(monkeypatch: pytest.MonkeyPatch, allowed: list[str]) -> list[list[str]]:
    """Like `trap_processes`, but one call goes through to the real `subprocess.Popen`: the one whose first
    argument is exactly the list `allowed` (AP9-LS-G, the one `lms get`), and only the first such call. Any other call, and any call by
    `os.system`, fails the test. The returned list holds the argument lists that were let through."""
    real_popen = subprocess.Popen
    let_through: list[list[str]] = []

    def guarded(*args, **kwargs):
        first = args[0] if args else kwargs.get("args")
        if not let_through and isinstance(first, list) and first == allowed:
            let_through.append(list(first))
            return real_popen(*args, **kwargs)
        pytest.fail(f"a process was started: {args!r}", pytrace=False)

    def refuse(*args, **kwargs):
        pytest.fail(f"a process was started: {args!r}", pytrace=False)

    monkeypatch.setattr(subprocess, "Popen", guarded)
    monkeypatch.setattr(os, "system", refuse)
    return let_through


def fake_lms_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A program `lms` first in the `PATH` that appends its arguments, one line per run, to the returned mark file."""
    folder = tmp_path / "fake-path"
    folder.mkdir()
    mark = tmp_path / "lms-was-run.txt"
    if sys.platform == "win32":
        (folder / "lms.cmd").write_text(f'@echo off\r\necho %*>> "{mark}"\r\n', encoding="ascii")
    else:
        program = folder / "lms"
        program.write_text(f'#!/bin/sh\necho "$@" >> "{mark}"\n', encoding="ascii")
        program.chmod(program.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")
    return mark


def lms_in_the_home_folder(home: Path, mark: Path | None = None) -> None:
    """`~/.lmstudio/bin/lms` and `lms.exe`: the places Windows and the others keep the program. With `mark`
    the files are scripts that append their arguments to it when run (on a platform that can run them);
    without, empty files."""
    folder = home / ".lmstudio" / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    text = b"" if mark is None else f'#!/bin/sh\necho "$@" >> "{mark}"\n'.encode("ascii")
    for name in ("lms", "lms.exe"):
        (folder / name).write_bytes(text)
        (folder / name).chmod((folder / name).stat().st_mode | stat.S_IXUSR)


def local_server_at(port: int, monkeypatch: pytest.MonkeyPatch):
    """A `LocalLmStudioServer` on a loopback test port. The constructor takes no origin list, so the test
    setup alone widens the module's fixed one, for the length of the test."""
    from modelroom import lmstudio_server

    monkeypatch.setattr(lmstudio_server, "_ALLOWED_ORIGINS", frozenset({("127.0.0.1", port)}))
    return lmstudio_server.LocalLmStudioServer(base_url=f"http://127.0.0.1:{port}")
