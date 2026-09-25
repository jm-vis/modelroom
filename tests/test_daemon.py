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
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

import pytest

from modelroom.daemon import (
    ALLOWED_CALLS,
    GENERATE_PATH,
    GENERATE_TIMEOUT_SECONDS,
    MAX_STREAM_LINE_BYTES,
    PS_PATH,
    PULL_PATH,
    PULL_READ_TIMEOUT_SECONDS,
    SHORT_TIMEOUT_SECONDS,
    SHOW_PATH,
    TAGS_PATH,
    VERSION_PATH,
    DaemonError,
    FixtureDaemon,
    LocalDaemon,
    check_call,
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


def test_only_the_six_documented_calls_are_made(loopback):
    """Five reads and generates, and since 2026-09-25 the one pull of the install step."""
    daemon, _ = loopback
    assert ALLOWED_CALLS == frozenset(
        {
            ("GET", VERSION_PATH),
            ("GET", TAGS_PATH),
            ("GET", PS_PATH),
            ("POST", SHOW_PATH),
            ("POST", GENERATE_PATH),
            ("POST", PULL_PATH),
        }
    )
    for method, path in (("POST", "/api/delete"), ("DELETE", "/api/delete"), ("POST", "/api/copy"), ("POST", "/api/push")):
        with pytest.raises(DaemonError) as caught:
            daemon(method, path, {"model": "nova:7b"})
        assert path in str(caught.value)


def test_a_pull_is_a_post_and_nothing_else(loopback):
    daemon, recorder = loopback
    for method in ("GET", "PUT", "DELETE"):
        with pytest.raises(DaemonError):
            daemon(method, PULL_PATH, {"model": "nova:7b"})
    assert recorder.seen == []  # refused before a connection was opened
    with pytest.raises(DaemonError):
        check_call("GET", PULL_PATH)
    check_call("POST", PULL_PATH)


def test_the_source_never_names_delete_and_names_pull_in_the_transport_only():
    """`/api/delete` stays out of the package; `/api/pull` is one constant, used by one step."""
    source = Path(__file__).resolve().parent.parent / "modelroom"
    for module in sorted(source.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        assert "/api/delete" not in text, f"{module.name} names /api/delete"
        if module.name != "daemon.py":
            assert "/api/pull" not in text, f"{module.name} names /api/pull"
        if module.name not in ("daemon.py", "guided_install.py"):
            assert "PULL_PATH" not in text, f"{module.name} uses PULL_PATH"


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


# --- the pull: a stream of lines, read one at a time --------------------------------------------

PULL_FIXTURE = Path(__file__).parent / "fixtures" / "ollama_pull_smollm2_stream.ndjson"


class _Streaming(BaseHTTPRequestHandler):
    """A loopback daemon that answers a pull as Ollama does: one JSON line after another.

    `script` is a list of `(seconds to wait first, bytes to send)`; the answer ends when the
    script does, with the connection closed (HTTP/1.0, no length). `status` other than 200 sends
    one JSON error body instead, the way the daemon answers a name it does not know.
    """

    script: list = []
    status: int = 200
    seen: list = []

    def do_POST(self):  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length") or 0)
        _Streaming.seen.append((self.command, self.path, self.rfile.read(length)))
        if _Streaming.status != 200 and _Streaming.script:
            # An error status whose answer stays open after its first line.
            self.send_response(_Streaming.status)
            self.end_headers()
            try:
                for wait, chunk in _Streaming.script:
                    time.sleep(wait)
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except OSError:
                return
            return
        if _Streaming.status != 200:
            encoded = b'{"error":"pull model manifest: file does not exist"}'
            self.send_response(_Streaming.status)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            for wait, chunk in _Streaming.script:
                time.sleep(wait)
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError:
            return  # the client went away, which is what an abort test wants

    def log_message(self, *args):  # noqa: D102 - silence the stdlib access log
        return


@pytest.fixture
def streaming():
    """A loopback server that streams a pull, and a factory for a daemon with a short read limit."""
    _Streaming.script = []
    _Streaming.status = 200
    _Streaming.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Streaming)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def daemon(pull_timeout: float = 5.0) -> LocalDaemon:
        return LocalDaemon(
            base_url=f"http://127.0.0.1:{port}",
            allowed_origins=frozenset({("127.0.0.1", port)}),
            pull_timeout=pull_timeout,
        )

    try:
        yield daemon, _Streaming
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _lines(*payloads: dict) -> list[tuple[float, bytes]]:
    return [(0.0, json.dumps(payload).encode("utf-8") + b"\n") for payload in payloads]


def test_a_pull_hands_every_line_to_the_callback_and_returns_the_last_one(streaming):
    make, server = streaming
    recorded = PULL_FIXTURE.read_bytes().splitlines(keepends=True)
    server.script = [(0.0, line) for line in recorded]
    seen: list[dict] = []

    answer = make()("POST", PULL_PATH, {"model": "smollm2:135m", "stream": True}, progress=seen.append)

    assert answer.status == 200
    assert answer.json() == {"status": "success"}
    assert len(seen) == len(recorded)
    assert seen[0] == {"status": "pulling manifest"}
    assert any(line.get("completed") == line.get("total") for line in seen if "total" in line)
    method, path, body = server.seen[0]
    assert (method, path) == ("POST", PULL_PATH)
    assert json.loads(body) == {"model": "smollm2:135m", "stream": True}


def test_a_pull_that_stops_sending_is_an_error_after_the_read_limit(streaming):
    """Nothing arrives for longer than the socket limit: that is a fault, not a wait forever."""
    make, server = streaming
    server.script = _lines({"status": "pulling manifest"}) + [(1.5, b'{"status":"success"}\n')]

    started = time.monotonic()
    with pytest.raises(DaemonError) as caught:
        make(pull_timeout=0.3)("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=lambda _line: None)

    assert time.monotonic() - started < 1.4
    assert "did not answer" in str(caught.value)


def test_a_slow_pull_that_keeps_sending_is_no_error(streaming):
    """The limit is per read, not for the whole pull: lines over four limits in a row are fine."""
    make, server = streaming
    server.script = [(0.2, b'{"status":"pulling abc","total":10,"completed":%d}\n' % step) for step in range(6)]
    server.script.append((0.2, b'{"status":"success"}\n'))
    seen: list[dict] = []

    answer = make(pull_timeout=0.35)("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=seen.append)

    assert answer.json() == {"status": "success"}
    assert len(seen) == 7


@pytest.mark.parametrize(
    "line, words",
    [
        (b"\xff\xfe not text\n", "not JSON"),
        (b"{not json\n", "not JSON"),
        (b"[1, 2]\n", "not a JSON object"),
        (b"{" + b'"a":"' + b"x" * (MAX_STREAM_LINE_BYTES + 10) + b'"}\n', "longer than"),
    ],
    ids=["not-utf-8", "not-json", "not-an-object", "no-end"],
)
def test_a_line_of_a_pull_that_does_not_read_is_an_error_with_a_reason(streaming, line, words):
    make, server = streaming
    server.script = _lines({"status": "pulling manifest"}) + [(0.0, line)]

    with pytest.raises(DaemonError) as caught:
        make()("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=lambda _line: None)

    assert words in str(caught.value)


def test_a_pull_ends_at_its_first_error_line_whatever_follows(streaming):
    """Nothing after an error can turn it into a success (second-model round, 2026-09-25)."""
    make, server = streaming
    server.script = _lines({"status": "pulling manifest"}, {"error": "failed"}, {"status": "success"})
    seen: list[dict] = []

    answer = make()("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=seen.append)

    assert answer.json() == {"error": "failed"}
    assert len(seen) == 2


def test_a_pull_ends_at_success_even_when_the_daemon_keeps_the_answer_open(streaming):
    make, server = streaming
    server.script = _lines({"status": "success"}) + [(1.5, b'{"status":"late"}\n')]

    answer = make(pull_timeout=0.3)("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=lambda _l: None)

    assert answer.json() == {"status": "success"}


def test_a_pull_the_callback_stops_raises_what_the_callback_raised(streaming):
    """Ctrl-C lands in the callback while the stream is open; it goes up as it came."""
    make, server = streaming
    server.script = _lines({"status": "pulling manifest"}, {"status": "pulling abc", "total": 10})

    def stop(_line: dict) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        make()("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=stop)


def test_a_pull_of_a_name_the_daemon_does_not_know_comes_back_as_its_status(streaming):
    make, server = streaming
    server.status = 404
    seen: list[dict] = []

    answer = make()("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=seen.append)

    assert answer.status == 404
    assert "does not exist" in answer.json()["error"]
    assert seen == []


def test_a_pull_error_that_stays_open_is_read_as_its_first_line(streaming):
    """The error body of a pull is one line; the rest of an answer kept open is not waited for."""
    make, server = streaming
    server.status = 500
    server.script = _lines({"error": "boom"}) + [(1.5, b"x" * 1024)]

    started = time.monotonic()
    answer = make(pull_timeout=5.0)("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=lambda _l: None)

    assert answer.status == 500
    assert answer.json() == {"error": "boom"}
    assert time.monotonic() - started < 1.4


def test_a_pull_gets_a_read_limit_of_its_own_and_no_limit_for_the_whole():
    daemon = LocalDaemon()

    assert daemon.timeout_for(PULL_PATH) == PULL_READ_TIMEOUT_SECONDS == 60.0
    assert daemon.timeout_for(TAGS_PATH) == SHORT_TIMEOUT_SECONDS


def test_the_fixture_daemon_plays_a_pinned_pull_line_by_line():
    body = PULL_FIXTURE.read_bytes()
    daemon = FixtureDaemon({("POST", PULL_PATH): Response(status=200, body=body)})
    seen: list[dict] = []

    answer = daemon("POST", PULL_PATH, {"model": "smollm2:135m", "stream": True}, progress=seen.append)

    assert answer.json() == {"status": "success"}
    assert len(seen) == len(body.splitlines())
    assert daemon.calls == [("POST", PULL_PATH, {"model": "smollm2:135m", "stream": True})]


def test_the_fixture_daemon_returns_a_pinned_pull_error_as_it_is():
    error = Response(status=404, body=b'{"error": "file does not exist"}')
    daemon = FixtureDaemon({("POST", PULL_PATH): error})
    seen: list[dict] = []

    assert daemon("POST", PULL_PATH, {"model": "nova:7b", "stream": True}, progress=seen.append).status == 404
    assert seen == []


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
