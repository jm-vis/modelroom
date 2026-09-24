"""Tests for modelroom.daemon: the local Ollama daemon's own transport.

`LocalDaemon` is the only class here that opens a connection. The tests that need a real
socket bind one on the loopback interface themselves and hand its port to the daemon through
its constructor -- the same way `modelroom/http.py` lets a test relax its allow-list, never
through an environment variable and never by replacing a function.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from modelroom.daemon import (
    ALLOWED_CALLS,
    GENERATE_PATH,
    GENERATE_TIMEOUT_SECONDS,
    PS_PATH,
    SHORT_TIMEOUT_SECONDS,
    SHOW_PATH,
    TAGS_PATH,
    VERSION_PATH,
    DaemonError,
    FixtureDaemon,
    LocalDaemon,
)
from modelroom.http import Response
from modelroom.ollama_local import DEFAULT_BASE_URL

from fixture_support import json_response


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Recording(BaseHTTPRequestHandler):
    """A loopback Ollama stand-in: records what it was sent and answers from `answers`."""

    answers: dict = {}
    seen: list = []

    def _reply(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        _Recording.seen.append((self.command, self.path, body, dict(self.headers)))
        answer = _Recording.answers.get(self.path, (404, {"error": "not found"}))
        status, payload = answer[0], answer[1]
        extra = answer[2] if len(answer) > 2 else {}
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        # A promised length larger than what is sent makes the client read a truncated body.
        self.send_header("Content-Length", str(extra.pop("_promise_length", len(encoded))))
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    do_GET = _reply
    do_POST = _reply

    def log_message(self, *args):  # noqa: D102 - silence the stdlib access log
        return


@pytest.fixture
def loopback():
    """A real HTTP server on a free loopback port, plus a `LocalDaemon` pointed at it."""
    _Recording.answers = {}
    _Recording.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Recording)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    daemon = LocalDaemon(base_url=base_url, allowed_origins=frozenset({("127.0.0.1", port)}))
    try:
        yield daemon, _Recording
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- what the daemon may reach at all ------------------------------------------------------


def test_the_production_daemon_talks_to_the_local_ollama_origin_only():
    assert LocalDaemon().base_url == DEFAULT_BASE_URL
    for other in ("http://127.0.0.1:59999", "http://10.0.0.5:11434", "https://ollama.com"):
        with pytest.raises(DaemonError):
            LocalDaemon(base_url=other)


def test_a_base_url_with_a_path_or_no_port_is_refused():
    for other in ("http://127.0.0.1", "http://127.0.0.1:11434/api", "ftp://127.0.0.1:11434"):
        with pytest.raises(DaemonError):
            LocalDaemon(base_url=other)


def test_only_the_five_documented_calls_are_made(loopback):
    daemon, _ = loopback
    assert ALLOWED_CALLS == frozenset(
        {
            ("GET", VERSION_PATH),
            ("GET", TAGS_PATH),
            ("GET", PS_PATH),
            ("POST", SHOW_PATH),
            ("POST", GENERATE_PATH),
        }
    )
    with pytest.raises(DaemonError) as caught:
        daemon("POST", "/api/pull", {"model": "nova:7b"})
    assert "/api/pull" in str(caught.value)


def test_the_source_never_names_pull_or_delete():
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "modelroom"
    for module in sorted(source.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        assert "/api/pull" not in text, f"{module.name} names /api/pull"
        assert "/api/delete" not in text, f"{module.name} names /api/delete"


# --- the real transport --------------------------------------------------------------------


def test_a_get_carries_no_body_and_returns_the_answer(loopback):
    daemon, recorder = loopback
    recorder.answers[VERSION_PATH] = (200, {"version": "0.34.2"})

    response = daemon("GET", VERSION_PATH)

    assert response.status == 200
    assert response.json() == {"version": "0.34.2"}
    method, path, body, headers = recorder.seen[0]
    assert (method, path, body) == ("GET", VERSION_PATH, b"")
    assert headers["User-Agent"].startswith("modelroom/")


def test_a_post_sends_its_body_as_json(loopback):
    daemon, recorder = loopback
    recorder.answers[SHOW_PATH] = (200, {"details": {}})

    daemon("POST", SHOW_PATH, {"model": "nova:7b"})

    method, path, body, headers = recorder.seen[0]
    assert (method, path) == ("POST", SHOW_PATH)
    assert json.loads(body) == {"model": "nova:7b"}
    assert headers["Content-Type"] == "application/json"


def test_a_status_other_than_200_comes_back_as_a_response(loopback):
    daemon, recorder = loopback
    recorder.answers[PS_PATH] = (500, {"error": "boom"})

    assert daemon("GET", PS_PATH).status == 500


def test_a_daemon_that_is_not_there_is_an_error_with_a_reason():
    port = _free_port()
    daemon = LocalDaemon(
        base_url=f"http://127.0.0.1:{port}", allowed_origins=frozenset({("127.0.0.1", port)})
    )

    with pytest.raises(DaemonError) as caught:
        daemon("GET", VERSION_PATH)

    assert str(port) in str(caught.value)


def test_generate_gets_the_long_timeout_and_everything_else_the_short_one():
    daemon = LocalDaemon()

    assert daemon.timeout_for(GENERATE_PATH) == GENERATE_TIMEOUT_SECONDS
    assert daemon.timeout_for(VERSION_PATH) == SHORT_TIMEOUT_SECONDS
    assert GENERATE_TIMEOUT_SECONDS > SHORT_TIMEOUT_SECONDS


def test_the_opener_carries_no_proxy_and_no_redirect_handler():
    """A proxy would move the local daemon's traffic off this machine; a redirect off its origin."""
    import urllib.request

    handlers = LocalDaemon()._opener.handlers
    assert not any(isinstance(handler, urllib.request.ProxyHandler) for handler in handlers)
    assert not any(type(handler) is urllib.request.HTTPRedirectHandler for handler in handlers)
    assert not any(isinstance(handler, urllib.request.HTTPSHandler) for handler in handlers)


def test_a_redirect_with_a_destination_is_returned_as_it_came_and_never_followed(loopback):
    """With a `Location` a following transport would make a second request; this one must not."""
    daemon, recorder = loopback
    recorder.answers[TAGS_PATH] = (302, {"moved": True}, {"Location": VERSION_PATH})
    recorder.answers[VERSION_PATH] = (200, {"version": "0.34.2"})

    assert daemon("GET", TAGS_PATH).status == 302
    assert [(method, path) for method, path, _body, _headers in recorder.seen] == [("GET", TAGS_PATH)]


def test_a_body_that_does_not_arrive_in_full_is_an_error_with_a_reason(loopback):
    """A daemon that promises more than it sends; `IncompleteRead` is no `OSError`."""
    daemon, recorder = loopback
    recorder.answers[TAGS_PATH] = (200, {"models": []}, {"_promise_length": 4096})

    with pytest.raises(DaemonError) as caught:
        daemon("GET", TAGS_PATH)

    assert "did not answer" in str(caught.value)


def test_an_error_body_that_does_not_arrive_in_full_is_an_error_with_a_reason(loopback):
    """The same read, in the `HTTPError` branch: it is guarded there too, not only for a 200."""
    daemon, recorder = loopback
    recorder.answers[PS_PATH] = (500, {"error": "boom"}, {"_promise_length": 4096})

    with pytest.raises(DaemonError) as caught:
        daemon("GET", PS_PATH)

    assert "500" in str(caught.value)


# --- the fixture daemon ----------------------------------------------------------------------


def test_the_fixture_daemon_answers_in_sequence_and_records_every_call():
    first = json_response("ollama_ps_deepseek_loaded.json")
    second = Response(status=200, body=b'{"models": []}')
    daemon = FixtureDaemon({("GET", PS_PATH): [first, second]})

    assert daemon("GET", PS_PATH).body == first.body
    assert daemon("GET", PS_PATH).json() == {"models": []}
    assert daemon.calls == [("GET", PS_PATH, None), ("GET", PS_PATH, None)]


def test_the_fixture_daemon_repeats_a_single_answer_and_raises_a_planted_error():
    daemon = FixtureDaemon(
        {
            ("GET", VERSION_PATH): json_response("ollama_version_local.json"),
            ("POST", GENERATE_PATH): DaemonError("no answer within 600.0 s"),
        }
    )

    assert daemon("GET", VERSION_PATH).json() == {"version": "0.34.2"}
    assert daemon("GET", VERSION_PATH).json() == {"version": "0.34.2"}
    with pytest.raises(DaemonError):
        daemon("POST", GENERATE_PATH, {"model": "nova:7b"})


def test_the_fixture_daemon_says_so_when_a_test_pinned_no_answer():
    with pytest.raises(KeyError):
        FixtureDaemon({})("GET", TAGS_PATH)


def test_an_unreachable_fixture_daemon_fails_every_call():
    daemon = FixtureDaemon({}, unreachable="cannot reach the Ollama daemon")

    with pytest.raises(DaemonError):
        daemon("GET", TAGS_PATH)
    assert daemon.calls == [("GET", TAGS_PATH, None)]
