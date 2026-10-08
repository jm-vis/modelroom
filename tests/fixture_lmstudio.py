"""Fixture helpers for the LM Studio tests (AP9-LS): the recorded answers as servers, and the process trap.

`lmstudio_servers` builds a `FixtureLmStudioServer` from the recordings of `tests/fixtures/lmstudio_*.json`
and counts the chats it gets, so a test can say that none was sent. `trap_processes` fails the test on calls
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
from modelroom.lmstudio_server import MODELS_PATH, FixtureLmStudioServer


def lmstudio_servers(lists: list) -> FixtureLmStudioServer:
    """A server whose model list answers `lists` one after the other (the last one stays): a recording by
    its file name, or a list built by the test as a dict. It answers no chat -- the transport refuses the
    call -- but keeps every attempt in `calls`; `chat_calls` and `list_calls` count what it was sent."""
    answers = [
        json_response(item) if isinstance(item, str) else Response(status=200, body=json.dumps(item).encode("utf-8"))
        for item in lists
    ]

    def next_list(_body):
        return answers.pop(0) if len(answers) > 1 else answers[0]

    return FixtureLmStudioServer({("GET", MODELS_PATH): next_list})


def run_of(server: FixtureLmStudioServer | None):
    """A stand-in for the guided run, as far as LM Studio reads it: `servers` (its transport, what the start
    screen found, what step 4 measured) and `screen.note`, which keeps every note in `notes`."""
    from modelroom.runtimes import ServerRun

    notes: list[str] = []
    transports = {} if server is None else {"lmstudio": server}
    screen = types.SimpleNamespace(note=notes.append)
    return types.SimpleNamespace(servers=ServerRun(transports=transports), screen=screen, notes=notes)


def chat_calls(server: FixtureLmStudioServer) -> list:
    """The chats a fixture server was sent: none, in every test of LM Studio in 0.2.0."""
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


def fake_lms_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A program `lms` first in the `PATH` that writes the returned mark file when it is run."""
    folder = tmp_path / "fake-path"
    folder.mkdir()
    mark = tmp_path / "lms-was-run.txt"
    if sys.platform == "win32":
        (folder / "lms.cmd").write_text(f'@echo off\r\necho run> "{mark}"\r\n', encoding="ascii")
    else:
        program = folder / "lms"
        program.write_text(f'#!/bin/sh\necho run > "{mark}"\n', encoding="ascii")
        program.chmod(program.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")
    return mark


def lms_in_the_home_folder(home: Path, mark: Path | None = None) -> None:
    """`~/.lmstudio/bin/lms` and `lms.exe`: the places Windows and the others keep the program. With `mark`
    the files are scripts that write it when run (on a platform that can run them); without, empty files."""
    folder = home / ".lmstudio" / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    text = b"" if mark is None else f'#!/bin/sh\necho run > "{mark}"\n'.encode("ascii")
    for name in ("lms", "lms.exe"):
        (folder / name).write_bytes(text)
        (folder / name).chmod((folder / name).stat().st_mode | stat.S_IXUSR)


def local_server_at(port: int, monkeypatch: pytest.MonkeyPatch):
    """A `LocalLmStudioServer` on a loopback test port. The constructor takes no origin list, so the test
    setup alone widens the module's fixed one, for the length of the test."""
    from modelroom import lmstudio_server

    monkeypatch.setattr(lmstudio_server, "_ALLOWED_ORIGINS", frozenset({("127.0.0.1", port)}))
    return lmstudio_server.LocalLmStudioServer(base_url=f"http://127.0.0.1:{port}")
