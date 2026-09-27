"""Tests for modelroom.vllm_server: the transport to a local vLLM server and the model it states.

The transport is exercised against a real HTTP server on a loopback port, the reader against the
answer recorded from vLLM 0.30.0 (`tests/fixtures/vllm_v1_models_l4.json`). No network is
touched and nothing is replaced.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from modelroom.http import Response
from modelroom.vllm_server import (
    CHAT_PATH,
    KEY_REASON,
    MAX_ANSWER_BYTES,
    METRICS_PATH,
    MODELS_PATH,
    VLLM_BASE_URL,
    FixtureVllmServer,
    LocalVllmServer,
    ServedModel,
    ServedModelsError,
    VllmKeyError,
    VllmServerError,
    read_vllm_models,
)

from fixture_support import json_response


class _Recording(BaseHTTPRequestHandler):
    """Answers each path with what the test pinned, and records every request it saw."""

    answers: dict[str, tuple[int, bytes, dict]] = {}
    seen: list[tuple[str, str]] = []

    def _reply(self):
        type(self).seen.append((self.command, self.path))
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        status, body, extra = type(self).answers.get(self.path, (404, b"{}", {}))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    do_GET = _reply
    do_POST = _reply

    def log_message(self, *args):  # noqa: D102 - silence the stdlib access log
        return


@pytest.fixture
def loopback():
    """A real HTTP server on a free loopback port, plus a `LocalVllmServer` pointed at it."""
    _Recording.answers = {}
    _Recording.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Recording)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    vllm = LocalVllmServer(base_url=f"http://127.0.0.1:{port}", allowed_origins=frozenset({("127.0.0.1", port)}))
    try:
        yield vllm, _Recording
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- what the transport may reach at all ------------------------------------------------------


def test_the_production_server_talks_to_the_local_vllm_origin_only():
    assert LocalVllmServer().base_url == VLLM_BASE_URL == "http://127.0.0.1:8000"
    for other in ("http://127.0.0.1:11434", "http://10.0.0.5:8000", "https://127.0.0.1:8000", "http://localhost:8000"):
        with pytest.raises(VllmServerError):
            LocalVllmServer(base_url=other)


@pytest.mark.parametrize(
    "other",
    [
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8000?key=1",
        "http://127.0.0.1:8000#part",
        "http://someone" + "@127.0.0.1:8000",  # user info, written in two parts for the hygiene test
        "http://someone:secret" + "@127.0.0.1:8000",
    ],
    ids=["path", "query", "fragment", "user", "user-and-password"],
)
def test_a_path_a_query_a_fragment_or_user_info_is_refused_before_connecting(other):
    """`daemon.py` checks none of the last three and appends the path; this origin check does not."""
    with pytest.raises(VllmServerError, match="not a local vLLM server origin"):
        LocalVllmServer(base_url=other)


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/version"), ("GET", "/health"), ("POST", "/v1/completions"), ("POST", MODELS_PATH), ("GET", CHAT_PATH)],
)
def test_every_call_but_the_three_is_refused_before_a_connection(loopback, method, path):
    vllm, recorder = loopback

    with pytest.raises(VllmServerError, match="refusing to call"):
        vllm(method, path)

    assert recorder.seen == []


def test_the_three_calls_reach_the_server_as_they_were_named(loopback):
    vllm, recorder = loopback
    recorder.answers = {path: (200, b"{}", {}) for path in (MODELS_PATH, METRICS_PATH, CHAT_PATH)}

    vllm("GET", MODELS_PATH)
    vllm("GET", METRICS_PATH)
    vllm("POST", CHAT_PATH, {"model": "m"})

    assert recorder.seen == [("GET", MODELS_PATH), ("GET", METRICS_PATH), ("POST", CHAT_PATH)]


def test_the_opener_carries_no_proxy_and_no_redirect_handler():
    import urllib.request

    handlers = LocalVllmServer()._opener.handlers
    assert not any(isinstance(handler, urllib.request.ProxyHandler) for handler in handlers)
    assert not any(type(handler) is urllib.request.HTTPRedirectHandler for handler in handlers)
    assert not any(isinstance(handler, urllib.request.HTTPSHandler) for handler in handlers)


def test_a_redirect_is_returned_as_it_came_and_never_followed(loopback):
    vllm, recorder = loopback
    recorder.answers = {MODELS_PATH: (302, b"{}", {"Location": METRICS_PATH}), METRICS_PATH: (200, b"", {})}

    assert vllm("GET", MODELS_PATH).status == 302
    assert recorder.seen == [("GET", MODELS_PATH)]


def test_a_server_that_asks_for_a_key_is_said_to_do_so(loopback):
    vllm, recorder = loopback
    recorder.answers = {MODELS_PATH: (401, b'{"error": "Unauthorized"}', {})}

    with pytest.raises(VllmKeyError) as caught:
        vllm("GET", MODELS_PATH)

    assert str(caught.value) == KEY_REASON
    assert KEY_REASON == "the vLLM server asks for an API key; modelroom reads an open local server only"


def test_an_answer_above_four_mib_is_refused(loopback):
    vllm, recorder = loopback
    recorder.answers = {METRICS_PATH: (200, b"x" * (MAX_ANSWER_BYTES + 1), {})}

    with pytest.raises(VllmServerError, match="more than"):
        vllm("GET", METRICS_PATH)
    assert MAX_ANSWER_BYTES == 4 * 1024 * 1024


def test_a_server_that_is_not_there_is_an_error_with_a_reason():
    with pytest.raises(VllmServerError, match="did not answer"):
        LocalVllmServer(base_url="http://127.0.0.1:9", allowed_origins=frozenset({("127.0.0.1", 9)}))("GET", MODELS_PATH)


def test_a_chat_gets_six_hundred_seconds_and_everything_else_ten():
    vllm = LocalVllmServer()

    assert vllm.timeout_for(CHAT_PATH) == 600.0
    assert vllm.timeout_for(MODELS_PATH) == vllm.timeout_for(METRICS_PATH) == 10.0


# --- the fixture server ---------------------------------------------------------------------------


def test_the_fixture_server_checks_the_calls_and_can_be_unreachable():
    unreachable = FixtureVllmServer({}, unreachable="refused")

    with pytest.raises(VllmServerError, match="refused"):
        unreachable("GET", MODELS_PATH)
    with pytest.raises(VllmServerError, match="refusing to call"):
        FixtureVllmServer({})("GET", "/health")


def test_the_fixture_server_answers_a_chat_by_its_body():
    fixture = FixtureVllmServer({("POST", CHAT_PATH): lambda body: Response(status=200, body=json.dumps(body).encode())})

    assert fixture("POST", CHAT_PATH, {"index": 3}).json() == {"index": 3}
    assert fixture.calls == [("POST", CHAT_PATH, {"index": 3})]


# --- the model the server states it serves -----------------------------------------------------


def test_the_recorded_model_list_states_one_model_its_root_and_its_context():
    fixture = FixtureVllmServer({("GET", MODELS_PATH): json_response("vllm_v1_models_l4.json")})

    served = read_vllm_models(fixture)

    assert served == ServedModel(id="Qwen/Qwen2.5-0.5B-Instruct", root="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=4096)
    assert fixture.calls == [("GET", MODELS_PATH, None)]


def _models(*entries: dict) -> FixtureVllmServer:
    body = json.dumps({"object": "list", "data": list(entries)}).encode("utf-8")
    return FixtureVllmServer({("GET", MODELS_PATH): Response(status=200, body=body)})


def _entry(model_id: str, owner: str = "vllm") -> dict:
    return {"id": model_id, "owned_by": owner, "root": model_id, "max_model_len": 4096}


def test_several_served_models_are_a_reason_and_no_model():
    with pytest.raises(ServedModelsError, match="serves 2 models"):
        read_vllm_models(_models(_entry("a/one"), _entry("a/two")))


def test_a_list_with_no_model_owned_by_vllm_is_no_vllm_server():
    with pytest.raises(VllmServerError, match="owned by vllm"):
        read_vllm_models(_models(_entry("a/one", owner="library")))


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"data": "none"}'])
def test_an_answer_that_is_no_model_list_is_no_vllm_server(body):
    with pytest.raises(VllmServerError):
        read_vllm_models(FixtureVllmServer({("GET", MODELS_PATH): Response(status=200, body=body)}))
