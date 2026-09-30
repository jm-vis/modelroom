"""`modelroom/remote_client.py` (FR-B1, B2): `hello`/`profile`/`room`, and `SshDaemon`, which lets
`guided_install.py`'s pull machinery run unchanged against a computer read over SSH.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fixture_ssh import FakeSsh
from modelroom.daemon import DaemonError, FixtureDaemon
from modelroom.http import Response
from modelroom.remote_client import SshDaemon, hello, profile, room
from modelroom.remote_server import ServerContext
from modelroom.room import RoomProbes
from modelroom.ssh import SshError

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


def _room_probes(**kw) -> RoomProbes:
    return RoomProbes(environ={}, exists=lambda p: False, **kw)


def _context(**kw) -> ServerContext:
    from modelroom.measure import Probes

    base = dict(
        probes=Probes(platform="linux", runner=lambda a: (_ for _ in ()).throw(FileNotFoundError()), read_text=lambda p: (_ for _ in ()).throw(OSError()), memory_bytes=lambda: 0),
        room=_room_probes(),
        daemon=FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}")}),
        now=lambda: NOW,
        environ={},
    )
    base.update(kw)
    return ServerContext(**base)


def test_hello_reads_version_system_ollama():
    runner = FakeSsh(context=_context())
    result = hello("gpu1", runner)
    assert result["version"] == "0.2.0"
    assert result["ollama"] == "reachable"


def test_a_forwarded_agent_stops_the_run():
    runner = FakeSsh(context=_context(environ={"SSH_AUTH_SOCK": "/tmp/a"}))
    with pytest.raises(SshError, match="switches forwarding off"):
        hello("gpu1", runner)


def test_hello_says_ollama_does_not_answer():
    runner = FakeSsh(context=_context(daemon=FixtureDaemon({}, unreachable="down")))
    with pytest.raises(SshError, match="Ollama does not answer on gpu1"):
        hello("gpu1", runner)


def test_profile_uses_the_given_id_and_the_placeholder_name():
    runner = FakeSsh(context=_context())
    result = profile("gpu1", "3f9a0c21d4e6b870", None, runner)
    assert result["export"]["profile"]["profile_id"] == "3f9a0c21d4e6b870"
    assert result["export"]["profile"]["display_name"] == "remote"


def test_room_carries_no_path():
    runner = FakeSsh(context=_context())
    result = room("gpu1", [], runner)
    assert set(result["disk"].keys()) == {"known", "free_bytes", "unclear"}


def test_ssh_daemon_turns_error_true_into_the_fixed_string_check_line_accepts():
    from modelroom.guided_install import _check_line

    pull = Response(200, {}, json.dumps({"status": "error", "error": "sentinel text"}).encode())
    daemon = FixtureDaemon({("POST", "/api/pull"): pull})
    runner = FakeSsh(context=_context(daemon=daemon))
    seen = []
    answer = SshDaemon("gpu1", runner)("POST", "/api/pull", {"model": "qwen3.5:9b"}, progress=seen.append)
    for line in seen:
        _check_line(line)  # would raise DaemonError if the shape were wrong
    assert seen == [{"error": "Ollama on gpu1 reported an error"}]
    assert answer.status == 200
    assert answer.body == b""


def test_ssh_daemon_turns_done_into_a_success_body():
    lines = [
        Response(200, {}, json.dumps({"status": "pulling manifest"}).encode()),
        Response(200, {}, json.dumps({"status": "success"}).encode()),
    ]
    body = b"\n".join(json.dumps(json.loads(line.body)).encode() for line in lines)
    daemon = FixtureDaemon({("POST", "/api/pull"): Response(200, {}, body)})
    runner = FakeSsh(context=_context(daemon=daemon))
    answer = SshDaemon("gpu1", runner)("POST", "/api/pull", {"model": "qwen3.5:9b"})
    assert answer.status == 200
    assert json.loads(answer.body)["status"] == "success"


def test_ssh_daemon_short_call_carries_status_and_body():
    daemon = FixtureDaemon({("GET", "/api/tags"): Response(200, {}, b'{"models": []}')})
    runner = FakeSsh(context=_context(daemon=daemon))
    answer = SshDaemon("gpu1", runner)("GET", "/api/tags")
    assert answer.status == 200
    assert json.loads(answer.body) == {"models": []}


def test_names_from_the_servers_tags_are_never_shown():
    """`SshDaemon` itself never reads or filters `/api/tags` names -- the unchanged `installed_models`
    (`loadtest.py`) does that, exactly as it does for `LocalDaemon`; this only proves the body
    travels unfiltered, so the existing digest rule still applies unchanged."""
    daemon = FixtureDaemon({("GET", "/api/tags"): Response(200, {}, b'{"models": [{"name": "sentinel-host:latest"}]}')})
    runner = FakeSsh(context=_context(daemon=daemon))
    answer = SshDaemon("gpu1", runner)("GET", "/api/tags")
    assert json.loads(answer.body)["models"][0]["name"] == "sentinel-host:latest"  # travels; never displayed by this module
