"""`modelroom/remote_client.py` (FR-B1, B2): `hello`/`profile`/`room`, and `SshDaemon`, which lets
`guided_install.py`'s pull machinery run unchanged against a computer read over SSH.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fixture_ssh import FakeSsh, FaultSsh
from modelroom.daemon import DaemonError, FixtureDaemon
from modelroom.http import Response
from modelroom.remote_client import SshDaemon, hello, profile, room
from modelroom.remote_server import ServerContext
from modelroom.room import RoomProbes
from modelroom.ssh import ProcessOutcome, SshError

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
    with pytest.raises(DaemonError) as excinfo:
        SshDaemon("gpu1", runner)("POST", "/api/pull", {"model": "qwen3.5:9b"}, progress=seen.append)
    for line in seen:
        _check_line(line)  # would raise DaemonError if the shape were wrong
    assert seen == [{"error": "Ollama on gpu1 reported an error"}]
    assert str(excinfo.value) == "Ollama on gpu1 reported an error; run ollama pull qwen3.5:9b there to see it"
    assert "sentinel" not in str(excinfo.value)


class _ErrorFlood:
    """A server whose pull sends `count` error lines, one after another, then `done: false` -- the
    same bytes object each time, so the runner itself holds nothing that grows with `count`."""

    LINE = b'{"protocol":1,"progress":{"error":true}}'

    def __init__(self, count: int) -> None:
        self.count = count

    def run(self, argv, stdin, total, per_line, limits, on_line):
        for _ in range(self.count):
            on_line(self.LINE)
        on_line(b'{"protocol":1,"result":{"status":200,"done":false}}')
        return ProcessOutcome(0)


def _peak_bytes(count: int) -> int:
    import tracemalloc

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        with pytest.raises(DaemonError, match="reported an error"):
            SshDaemon("gpu1", _ErrorFlood(count), ssh_path="ssh")("POST", "/api/pull", {"model": "qwen3.5:9b"})
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


def test_a_long_stream_of_error_lines_keeps_the_memory_of_the_client_flat():
    """Codex FR-B1 r3 finding 1: a pull has no total limit, so the client keeps one flag for "an
    error line came", never one entry per line -- 60,000 error lines need no more memory than 1,000."""
    small, large = _peak_bytes(1_000), _peak_bytes(60_000)
    assert large - small < 64 * 1024, (small, large)


def test_a_pull_status_other_than_200_is_its_own_sentence():
    daemon = FixtureDaemon({("POST", "/api/pull"): Response(500, {}, b'{"error": "sentinel text"}')})
    runner = FakeSsh(context=_context(daemon=daemon))
    with pytest.raises(DaemonError) as excinfo:
        SshDaemon("gpu1", runner)("POST", "/api/pull", {"model": "qwen3.5:9b"})
    assert str(excinfo.value) == "Ollama on gpu1 answered with status 500"


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


def test_ssh_daemon_and_the_read_only_calls_default_to_no_hardcoded_ssh_path():
    """Codex r1 finding 5: a literal `ssh_path="ssh"` default would bypass `ssh.call`'s own
    `which` resolution (its `ssh_path` parameter short-circuits it whenever it is not `None`), so
    a missing `ssh` binary would raise an uncaught process-start error instead of the fixed `NO_SSH`
    sentence. Every one of this module's own entry points must default to `None`, letting `call`
    resolve it."""
    import inspect

    for func in (hello, profile, room):
        assert inspect.signature(func).parameters["ssh_path"].default is None, func.__name__
    assert SshDaemon("gpu1", FakeSsh(context=_context())).ssh_path is None


def test_an_sshserror_from_a_pull_is_translated_to_daemonerror_not_left_uncaught():
    """Codex r1 finding 5: the pull path in `guided_install.py` only ever catches `DaemonError`."""
    runner = FaultSsh(ProcessOutcome(None, timed_out=True))
    daemon = SshDaemon("gpu1", runner)
    with pytest.raises(DaemonError):
        daemon("POST", "/api/pull", {"model": "qwen3.5:9b"})


def test_a_connection_lost_mid_pull_keeps_the_resumable_hint_through_daemonerror():
    """Codex r1 finding 5: exit 255 after progress already arrived must carry the resumable
    sentence, not the generic connection sentence, even once wrapped as a `DaemonError`."""
    progress_line = json.dumps({"protocol": 1, "progress": {"status": "pulling manifest"}}).encode()
    runner = FaultSsh(ProcessOutcome(255, stderr=b"kex_exchange_identification: read: reset"), lines=(progress_line,))
    daemon = SshDaemon("gpu1", runner)
    with pytest.raises(DaemonError, match="the same pull resumes"):
        daemon("POST", "/api/pull", {"model": "qwen3.5:9b"})


def test_ssh_daemon_pull_progress_streams_through_not_collected():
    """Codex r1 finding 1/2: the progress callback is called as each `FakeSsh`-produced line
    arrives -- proven by asserting the count inside the callback itself, before the call returns."""
    lines = [
        Response(200, {}, json.dumps({"status": "pulling manifest"}).encode()),
        Response(200, {}, json.dumps({"status": "downloading", "total": 10, "completed": 3}).encode()),
        Response(200, {}, json.dumps({"status": "success"}).encode()),
    ]
    body = b"\n".join(json.dumps(json.loads(line.body)).encode() for line in lines)
    daemon_fixture = FixtureDaemon({("POST", "/api/pull"): Response(200, {}, body)})
    runner = FakeSsh(context=_context(daemon=daemon_fixture))
    seen: list[dict] = []

    def progress(entry):
        seen.append(entry)

    SshDaemon("gpu1", runner)("POST", "/api/pull", {"model": "qwen3.5:9b"}, progress=progress)
    assert len(seen) == 3
    assert seen[0]["status"] == "pulling manifest"
    assert seen[-1]["status"] == "success"


def test_names_from_the_servers_tags_are_never_shown():
    """`SshDaemon` itself never reads or filters `/api/tags` names -- the unchanged `installed_models`
    (`loadtest.py`) does that, exactly as it does for `LocalDaemon`; this only proves the body
    travels unfiltered, so the existing digest rule still applies unchanged."""
    daemon = FixtureDaemon({("GET", "/api/tags"): Response(200, {}, b'{"models": [{"name": "sentinel-host:latest"}]}')})
    runner = FakeSsh(context=_context(daemon=daemon))
    answer = SshDaemon("gpu1", runner)("GET", "/api/tags")
    assert json.loads(answer.body)["models"][0]["name"] == "sentinel-host:latest"  # travels; never displayed by this module
