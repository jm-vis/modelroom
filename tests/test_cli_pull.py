"""Tests for `modelroom pull` (A4, AP9-FR): the command a `pull_other` handover line names, run
with no results folder and no configuration -- only this machine's daemon and disk.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from modelroom.cli_pull import ALREADY_LISTED, NOT_REACHABLE, NO_TERMINAL, UNKNOWN_SIZE, run_pull
from modelroom.daemon import PULL_PATH, TAGS_PATH, VERSION_PATH, FixtureDaemon
from modelroom.dialog import TerminalAsker
from modelroom.room import PULL_DISK_MARGIN_BYTES

from fixture_room import FREE_DISK_BYTES, fixed_room
from fixture_servers import pull_daemon, tags_with
from fixture_support import json_response

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
NAME = "hf.co/acme/Test-7B-GGUF:Q4_K_M"
ALREADY_LISTED_NAME = "granite4.2:8b"  # already in the default `/api/tags` fixture


def _daemon(pull=None) -> FixtureDaemon:
    answers = {
        ("GET", VERSION_PATH): json_response("ollama_version_local.json"),
        ("GET", TAGS_PATH): json_response("ollama_tags_local_loadtest.json"),
    }
    if pull is not None:
        answers[("POST", PULL_PATH)] = pull
    return FixtureDaemon(answers)


def _offline() -> FixtureDaemon:
    from modelroom.ollama_local import DEFAULT_BASE_URL

    return FixtureDaemon({}, unreachable=f"{DEFAULT_BASE_URL}: the Ollama daemon did not answer (refused)")


def test_an_invalid_name_is_exit_2(capsys):
    code = run_pull("a;b", daemon=_daemon(), now=NOW)

    assert code == 2
    assert "a;b" in capsys.readouterr().err


def test_an_unreachable_daemon_is_exit_1(capsys):
    code = run_pull(NAME, daemon=_offline(), now=NOW)

    assert code == 1
    assert NOT_REACHABLE in capsys.readouterr().err


def test_too_little_disk_pulls_nothing(capsys):
    daemon = _daemon()
    room = fixed_room(free_disk_bytes=FREE_DISK_BYTES)
    over_the_free_disk = FREE_DISK_BYTES

    code = run_pull(NAME, weights_bytes=over_the_free_disk, daemon=daemon, room=room, now=NOW)

    assert code == 1
    out = capsys.readouterr()
    assert "not enough disk" in out.err or "not enough disk" in out.out
    assert not any(call[1] == PULL_PATH for call in daemon.calls)
    assert NO_TERMINAL not in out.err  # it never reached the question


def test_exactly_enough_disk_asks(capsys):
    daemon = _daemon()
    room = fixed_room(free_disk_bytes=FREE_DISK_BYTES)
    exactly_enough = FREE_DISK_BYTES - PULL_DISK_MARGIN_BYTES

    code = run_pull(NAME, weights_bytes=exactly_enough, daemon=daemon, room=room, now=NOW)

    # No terminal here (pytest): reaching the question shows as exit 2 with `NO_TERMINAL`, not
    # exit 1 with a disk message -- proof that the check let it through.
    assert code == 2
    assert NO_TERMINAL in capsys.readouterr().err


def test_without_size_asks_with_a_note(capsys):
    daemon = _daemon()

    code = run_pull(NAME, weights_bytes=None, daemon=daemon, room=fixed_room(), now=NOW)

    out = capsys.readouterr()
    assert UNKNOWN_SIZE in out.out
    assert code == 2 and NO_TERMINAL in out.err


def test_a_listed_name_is_a_note_and_still_asked(capsys):
    daemon = _daemon()

    code = run_pull(ALREADY_LISTED_NAME, weights_bytes=1024, daemon=daemon, room=fixed_room(), now=NOW)

    out = capsys.readouterr()
    assert ALREADY_LISTED.format(name=ALREADY_LISTED_NAME) in out.out
    assert code == 2 and NO_TERMINAL in out.err  # a note, not a skip: it still reaches the question


def test_no_terminal_without_yes_is_exit_2(capsys):
    code = run_pull(NAME, weights_bytes=1024, yes=False, daemon=_daemon(), room=fixed_room(), now=NOW)

    assert code == 2
    assert NO_TERMINAL in capsys.readouterr().err


def test_yes_pulls_and_says_pulled(capsys):
    daemon = pull_daemon(after_pull=tags_with(NAME))

    code = run_pull(NAME, weights_bytes=1024, yes=True, daemon=daemon, room=fixed_room(), now=NOW)

    out = capsys.readouterr()
    assert code == 0
    assert any(line.strip().startswith("✓ Pulled") for line in out.out.splitlines())
    assert any(call[1] == PULL_PATH for call in daemon.calls)


def test_no_is_exit_0_and_pulls_nothing(capsys):
    daemon = _daemon()

    with create_pipe_input() as pipe:
        pipe.send_text("\r")  # Enter alone: the pointer starts on No (`default=False`)
        asker = TerminalAsker(input=pipe, output=DummyOutput())
        code = run_pull(NAME, weights_bytes=1024, daemon=daemon, room=fixed_room(), now=NOW, asker=asker)

    assert code == 0
    assert not any(call[1] == PULL_PATH for call in daemon.calls)


def test_a_pull_error_is_exit_1_one_line(capsys):
    from modelroom.daemon import DaemonError

    daemon = _daemon(pull=DaemonError("http://127.0.0.1:11434/api/pull: the Ollama daemon did not answer (timed out)"))

    code = run_pull(NAME, weights_bytes=1024, yes=True, daemon=daemon, room=fixed_room(), now=NOW)

    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert code == 1
    assert len(err_lines) == 1
    assert "timed out" in err_lines[0]
