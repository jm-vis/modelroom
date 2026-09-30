"""`modelroom/remote_server.py` (FR-B1, B2): the gegenstelle, in-process -- no ssh, no real Ollama,
no real machine. Every op answers the fixed envelope; nothing here writes to disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fixture_room import fixed_room
from modelroom.daemon import DaemonError, FixtureDaemon
from modelroom.http import Response
from modelroom.measure import Probes
from modelroom.remote_server import (
    ERROR_BAD_REQUEST,
    ERROR_CALL_REFUSED,
    ERROR_OLLAMA_UNREACHABLE,
    ERROR_TOO_LARGE,
    MAX_INPUT_BYTES,
    ServerContext,
    main,
)

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


class _NothingRunner:
    """Every external command answers "not found": `measure_hardware` reads this as a machine with
    no GPU tool and no llmfit, never as a fault -- the profile op still returns a valid profile."""

    def __call__(self, args):
        raise FileNotFoundError(args[0])


def _probes(runner=None) -> Probes:
    return Probes(
        platform="linux",
        runner=runner or _NothingRunner(),
        read_text=lambda path: (_ for _ in ()).throw(OSError("no such file")),
        memory_bytes=lambda: 0,
    )


def _context(**kw) -> ServerContext:
    daemon = FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}")})
    base = dict(probes=_probes(), room=fixed_room(), daemon=daemon, now=lambda: NOW, environ={})
    base.update(kw)
    return ServerContext(**base)


def _one(lines: list[bytes]) -> dict:
    assert len(lines) == 1
    return json.loads(lines[0])


def test_every_line_has_protocol_1_and_exactly_one_field():
    lines, code = main("hello", b"{}", _context())
    assert code == 0
    payload = _one(lines)
    assert payload["protocol"] == 1
    assert len([k for k in ("progress", "result", "error") if k in payload]) == 1


def test_hello_names_protocol_version_system_ollama():
    daemon = FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}")})
    result = _one(main("hello", b"{}", _context(daemon=daemon))[0])["result"]
    assert result["version"] == "0.2.0"
    assert result["system"] == "linux"
    assert result["ollama"] == "reachable"
    assert result["agent_forwarded"] is False


def test_hello_says_unreachable_when_ollama_does_not_answer():
    daemon = FixtureDaemon({}, unreachable="connection refused")
    result = _one(main("hello", b"{}", _context(daemon=daemon))[0])["result"]
    assert result["ollama"] == "unreachable"


def test_hello_names_a_forwarded_agent():
    result = _one(main("hello", b"{}", _context(environ={"SSH_AUTH_SOCK": "/tmp/agent.1"}))[0])["result"]
    assert result["agent_forwarded"] is True


def test_profile_never_calls_the_host_name():
    """`Probes.hostname` is never consulted for the `profile` op (B2): a probe that would raise if
    called proves it, since the profile is built without it."""
    probes = _probes()
    probes = probes.__class__(**{**probes.__dict__, "hostname": lambda: (_ for _ in ()).throw(AssertionError("hostname called"))})
    context = _context(probes=probes)
    payload = {"profile_id": "3f9a0c21d4e6b870", "machine_class": None}
    result = _one(main("profile", json.dumps(payload).encode(), context)[0])["result"]
    assert result["export"]["profile"]["profile_id"] == "3f9a0c21d4e6b870"


def test_profile_uses_the_given_id_and_the_placeholder_name():
    payload = {"profile_id": "3f9a0c21d4e6b870", "machine_class": "server"}
    result = _one(main("profile", json.dumps(payload).encode(), _context())[0])["result"]
    export = result["export"]
    assert export["profile"]["profile_id"] == "3f9a0c21d4e6b870"
    assert export["profile"]["display_name"] == "remote"
    assert export["measurements"] == []


def test_a_profile_request_with_a_bad_id_is_bad_request():
    payload = {"profile_id": "not-an-id", "machine_class": None}
    error = _one(main("profile", json.dumps(payload).encode(), _context())[0])
    assert error["error"] == ERROR_BAD_REQUEST


def test_room_carries_no_drive_no_path():
    result = _one(main("room", b'{"cards": []}', _context())[0])["result"]
    assert set(result.keys()) == {"disk", "graphics_free_mib"}
    assert set(result["disk"].keys()) == {"known", "free_bytes", "unclear"}
    assert "drive" not in result["disk"] and "reason" not in result["disk"]


def test_room_reads_the_free_disk_of_this_machine():
    result = _one(main("room", b'{"cards": []}', _context())[0])["result"]
    assert result["disk"]["known"] is True
    assert result["disk"]["free_bytes"] > 0


def test_daemon_refuses_a_call_outside_the_allowlist_on_the_server():
    body = {"method": "DELETE", "path": "/api/delete", "body": None}
    error = _one(main("daemon", json.dumps(body).encode(), _context())[0])
    assert error["error"] == ERROR_CALL_REFUSED


def test_daemon_refuses_an_unsafe_pull_name():
    body = {"method": "POST", "path": "/api/pull", "body": {"model": "x; rm -rf /"}}
    error = _one(main("daemon", json.dumps(body).encode(), _context())[0])
    assert error["error"] == ERROR_CALL_REFUSED


def test_a_short_daemon_call_carries_status_and_body():
    daemon = FixtureDaemon({("GET", "/api/tags"): Response(200, {}, b'{"models": []}')})
    body = {"method": "GET", "path": "/api/tags", "body": None}
    result = _one(main("daemon", json.dumps(body).encode(), _context(daemon=daemon))[0])["result"]
    assert result == {"status": 200, "body": {"models": []}}


def test_pull_lines_carry_only_known_statuses_and_counts():
    lines = [
        Response(200, {}, json.dumps({"status": "pulling manifest"}).encode()),
        Response(200, {}, json.dumps({"status": "downloading", "total": 10, "completed": 4}).encode()),
        Response(200, {}, json.dumps({"status": "success"}).encode()),
    ]
    daemon = FixtureDaemon({("POST", "/api/pull"): _sequenced_pull(lines)})
    body = {"method": "POST", "path": "/api/pull", "body": {"model": "qwen3.5:9b"}}
    out = [json.loads(line) for line in main("daemon", json.dumps(body).encode(), _context(daemon=daemon))[0]]
    progress = [line["progress"] for line in out if "progress" in line]
    assert progress[0]["status"] == "pulling manifest"
    assert progress[1] == {"status": "working", "total": 10, "completed": 4}
    assert progress[2]["status"] == "success"
    assert out[-1]["result"] == {"status": 200, "done": True}


def _sequenced_pull(lines: list[Response]) -> Response:
    body = b"\n".join(json.dumps(json.loads(line.body)).encode() for line in lines)
    return Response(200, {}, body)


def test_an_ollama_error_line_loses_its_text():
    line = Response(200, {}, json.dumps({"status": "error", "error": "sentinel-host says no"}).encode())
    daemon = FixtureDaemon({("POST", "/api/pull"): line})
    body = {"method": "POST", "path": "/api/pull", "body": {"model": "qwen3.5:9b"}}
    out = [json.loads(line) for line in main("daemon", json.dumps(body).encode(), _context(daemon=daemon))[0]]
    progress = [entry["progress"] for entry in out if "progress" in entry]
    assert progress == [{"error": True}]
    assert "sentinel-host" not in json.dumps(out)


def test_a_bad_request_is_an_error_code_without_text():
    error = _one(main("hello", b"not json", _context())[0])
    assert error == {"protocol": 1, "error": ERROR_BAD_REQUEST}


def test_input_over_the_limit_is_too_large():
    huge = b"{" + b" " * (MAX_INPUT_BYTES + 1) + b"}"
    error = _one(main("hello", huge, _context())[0])
    assert error["error"] == ERROR_TOO_LARGE


def test_stdin_is_read_to_eof():
    """A well-formed request that ends the byte string exactly at its closing brace still reads."""
    payload = json.dumps({}).encode()
    assert payload == b"{}"
    lines, code = main("hello", payload, _context())
    assert code == 0


def test_the_server_writes_nothing(tmp_path):
    daemon = FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}")})
    before = sorted(tmp_path.rglob("*"))
    main("hello", b"{}", _context(daemon=daemon))
    payload = {"profile_id": "3f9a0c21d4e6b870", "machine_class": None}
    main("profile", json.dumps(payload).encode(), _context())
    after = sorted(tmp_path.rglob("*"))
    assert before == after == []


def test_an_unknown_op_is_bad_request():
    error = _one(main("candidates", b"{}", _context())[0])
    assert error["error"] == ERROR_BAD_REQUEST


def test_ollama_unreachable_is_its_own_code():
    daemon = FixtureDaemon({}, unreachable="down")
    body = {"method": "GET", "path": "/api/tags", "body": None}
    error = _one(main("daemon", json.dumps(body).encode(), _context(daemon=daemon))[0])
    assert error["error"] == ERROR_OLLAMA_UNREACHABLE
