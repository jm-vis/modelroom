"""Tests for modelroom.lmstudio_server: the transport to a local LM Studio server, its reader, its detection.

The transport is exercised against a real HTTP server on a loopback port, the reader against the answers
recorded from LM Studio on 2026-10-08 (`tests/fixtures/lmstudio_*.json`, `tests/fixtures/README.md`).
No network is touched and nothing is replaced. The chat is the second call the transport allows (AP9-LS-M): a
chat goes to a loopback double or an injected opener, never to a real server.
"""

from __future__ import annotations

import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from modelroom.http import Response
from modelroom.lmstudio_server import (
    ALLOWED_CALLS,
    BODY_HEADER,
    BODY_OVERSIZED,
    BODY_UNREAD,
    CHAT_PATH,
    KEY_REASON,
    LMSTUDIO_BASE_URL,
    MAX_ANSWER_BYTES,
    MODELS_PATH,
    FixtureLmStudioServer,
    LmStudioKeyError,
    LmStudioModels,
    LmStudioServerError,
    LmStudioTimeoutError,
    LoadedModel,
    LocalLmStudioServer,
    detect_lmstudio,
    lms_call,
    read_lmstudio_models,
)

from fixture_http import FIXTURES, json_response
from fixture_lmstudio import FakeAnswer, http_error, local_server_at, scripted_server

LLM_KEY = "qwen3.5-4b"
EMBEDDING_KEY = "text-embedding-nomic-embed-text-v1.5"


class _Recording(BaseHTTPRequestHandler):
    """Answers each path with what the test pinned, and records every request it saw."""

    answers: dict[str, tuple[int, bytes, dict]] = {}
    seen: list[tuple[str, str]] = []
    headers_seen: list[dict[str, str]] = []
    bodies_seen: list[bytes] = []

    def _reply(self):
        type(self).seen.append((self.command, self.path))
        type(self).headers_seen.append({name.lower(): value for name, value in self.headers.items()})
        length = int(self.headers.get("Content-Length") or 0)
        type(self).bodies_seen.append(self.rfile.read(length) if length else b"")
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
def loopback(monkeypatch):
    """A real HTTP server on a free loopback port, plus a `LocalLmStudioServer` pointed at it."""
    _Recording.answers = {}
    _Recording.seen = []
    _Recording.headers_seen = []
    _Recording.bodies_seen = []
    server = HTTPServer(("127.0.0.1", 0), _Recording)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    lmstudio = local_server_at(port, monkeypatch)
    try:
        yield lmstudio, _Recording
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _list(name: str = "lmstudio_v1_models_loaded.json") -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _server(payload) -> FixtureLmStudioServer:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return FixtureLmStudioServer({("GET", MODELS_PATH): Response(status=200, body=body)})


def _llm(payload: dict) -> dict:
    return next(entry for entry in payload["models"] if entry["type"] == "llm")


# --- what the transport may reach at all ------------------------------------------------------


def test_the_production_server_talks_to_the_local_lmstudio_origin_only():
    assert LocalLmStudioServer().base_url == LMSTUDIO_BASE_URL == "http://127.0.0.1:1234"
    for other in ("http://127.0.0.1:1235", "http://localhost:1234", "https://127.0.0.1:1234", "http://10.0.0.5:1234"):
        with pytest.raises(LmStudioServerError):
            LocalLmStudioServer(base_url=other)


@pytest.mark.parametrize(
    "other",
    [
        "http://127.0.0.1:1235",
        "http://localhost:1234",
        "https://127.0.0.1:1234",
        "http://127.0.0.1:1234/api/v1",
        "http://127.0.0.1:1234?key=1",
        "http://127.0.0.1:1234#part",
        "http://someone" + "@127.0.0.1:1234",  # user info, written in two parts for the hygiene test
        "http://someone:secret" + "@127.0.0.1:1234",
    ],
    ids=["port", "localhost", "https", "path", "query", "fragment", "user", "user-and-password"],
)
def test_the_transport_refuses_every_other_origin(other):
    with pytest.raises(LmStudioServerError, match="not a local LM Studio server origin"):
        LocalLmStudioServer(base_url=other)


def test_the_whole_surface_is_two_calls():
    assert ALLOWED_CALLS == {("GET", "/api/v1/models"), ("POST", "/v1/chat/completions")}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/v1/models/load"),
        ("POST", "/api/v1/models/unload"),
        ("POST", "/api/v1/models/download"),
        ("POST", "/api/v1/chat"),
        ("GET", "/metrics"),
        ("GET", "/api/v0/models"),
        ("POST", "/api/v0/chat/completions"),
        ("GET", "/v1/models"),
        ("POST", MODELS_PATH),
        ("GET", CHAT_PATH),
        ("PUT", CHAT_PATH),
    ],
)
def test_only_the_two_calls_are_allowed(loopback, method, path):
    lmstudio, recorder = loopback

    with pytest.raises(LmStudioServerError, match="refusing to call"):
        lmstudio(method, path)

    assert recorder.seen == []


def test_the_two_calls_reach_the_server_as_they_were_named(loopback):
    lmstudio, recorder = loopback
    recorder.answers = {MODELS_PATH: (200, b"{}", {}), CHAT_PATH: (200, b'{"choices": []}', {})}

    lmstudio("GET", MODELS_PATH)
    chat = lmstudio("POST", CHAT_PATH, {"model": "m"})

    assert recorder.seen == [("GET", MODELS_PATH), ("POST", CHAT_PATH)]
    assert recorder.bodies_seen == [b"", b'{"model": "m"}'] and chat.status == 200


def test_a_chat_reaches_the_fixture_server_and_is_kept_in_its_calls():
    fixture = FixtureLmStudioServer({("POST", CHAT_PATH): Response(status=200, body=b"{}")})

    fixture("POST", CHAT_PATH, {"model": "m"})

    assert fixture.calls == [("POST", CHAT_PATH, {"model": "m"})]
    with pytest.raises(LmStudioServerError, match="refusing to call GET"):
        fixture("GET", CHAT_PATH)


def test_the_list_gets_10_s_and_the_chat_600_s():
    answer = FakeAnswer(body=b"{}")
    lmstudio = scripted_server({MODELS_PATH: answer, CHAT_PATH: answer})

    lmstudio("GET", MODELS_PATH)
    lmstudio("POST", CHAT_PATH, {"model": "m"})

    assert lmstudio._opener.opened == [("GET", MODELS_PATH, 10.0), ("POST", CHAT_PATH, 600.0)]
    assert (lmstudio.timeout_for(MODELS_PATH), lmstudio.timeout_for(CHAT_PATH)) == (10.0, 600.0)


# --- the limit of time: at the connection, at the body, at the body of an error --------------------------------

_STAGES = {
    "connect, as a TimeoutError": lambda: TimeoutError("timed out"),
    "connect, as a URLError": lambda: urllib.error.URLError(TimeoutError("timed out")),
    "read of the body": lambda: FakeAnswer(read_error=TimeoutError("timed out")),
    "read of the body of an error": lambda: http_error(500, read_error=TimeoutError("timed out")),
    "read of the body, as a URLError": lambda: FakeAnswer(read_error=urllib.error.URLError(TimeoutError("timed out"))),
    "read of the body of an error, as a URLError": lambda: http_error(
        500, read_error=urllib.error.URLError(TimeoutError("timed out"))
    ),
}


@pytest.mark.parametrize("path", [MODELS_PATH, CHAT_PATH], ids=["list", "chat"])
@pytest.mark.parametrize("stage", list(_STAGES))
def test_a_limit_of_time_is_told_apart_at_all_three_places_for_the_list_and_the_chat(stage, path):
    lmstudio = scripted_server({path: _STAGES[stage]()})

    with pytest.raises(LmStudioTimeoutError) as caught:
        lmstudio("GET" if path == MODELS_PATH else "POST", path, None if path == MODELS_PATH else {"model": "m"})

    assert isinstance(caught.value, LmStudioServerError) and path in str(caught.value)


def test_another_failure_to_connect_is_no_limit_of_time():
    lmstudio = scripted_server({CHAT_PATH: urllib.error.URLError(ConnectionRefusedError("refused"))})

    with pytest.raises(LmStudioServerError) as caught:
        lmstudio("POST", CHAT_PATH, {"model": "m"})

    assert not isinstance(caught.value, LmStudioTimeoutError)


# --- a body that is not there: the list raises, the chat keeps the status --------------------------------------------


def test_the_list_raises_for_a_body_it_cannot_read_and_for_a_body_above_the_limit():
    unreadable = scripted_server({MODELS_PATH: FakeAnswer(read_error=OSError("reset"))})
    oversized = scripted_server({MODELS_PATH: FakeAnswer(body=b"x" * (MAX_ANSWER_BYTES + 1))})
    error_unreadable = scripted_server({MODELS_PATH: http_error(500, read_error=OSError("reset"))})

    for lmstudio in (unreadable, oversized, error_unreadable):
        with pytest.raises(LmStudioServerError) as caught:
            lmstudio("GET", MODELS_PATH)
        assert not isinstance(caught.value, LmStudioTimeoutError)


@pytest.mark.parametrize("status", [200, 400, 401, 403, 429, 500])
@pytest.mark.parametrize("why", [BODY_UNREAD, BODY_OVERSIZED])
def test_the_chat_keeps_the_known_status_when_its_body_cannot_be_read_or_is_too_large(status, why):
    if why == BODY_UNREAD:
        reply = FakeAnswer(status, read_error=OSError("reset")) if status == 200 else http_error(status, read_error=OSError("reset"))
    else:
        big = b"x" * (MAX_ANSWER_BYTES + 1)
        reply = FakeAnswer(status, big) if status == 200 else http_error(status, big)
    lmstudio = scripted_server({CHAT_PATH: reply})

    answer = lmstudio("POST", CHAT_PATH, {"model": "m"})

    assert (answer.status, answer.body, answer.header(BODY_HEADER)) == (status, b"", why)


@pytest.mark.parametrize("status", [200, 500])
def test_a_value_error_at_the_read_leaves_the_status_at_the_chat_and_raises_at_the_list(status):
    """A closed stream raises `ValueError` at the read: the chat keeps the status with `unread`, the list raises."""
    def reply():
        return FakeAnswer(status, read_error=ValueError("closed")) if status == 200 else http_error(status, read_error=ValueError("closed"))

    answer = scripted_server({CHAT_PATH: reply()})("POST", CHAT_PATH, {"model": "m"})
    assert (answer.status, answer.body, answer.header(BODY_HEADER)) == (status, b"", BODY_UNREAD)
    with pytest.raises(LmStudioServerError) as caught:
        scripted_server({MODELS_PATH: reply()})("GET", MODELS_PATH)
    assert not isinstance(caught.value, LmStudioTimeoutError)


@pytest.mark.parametrize("status", [200, 400, 401, 403, 429, 500])
def test_a_chat_with_a_body_that_can_be_read_has_no_marker(status):
    reply = FakeAnswer(status, b'{"a": 1}') if status == 200 else http_error(status, b'{"a": 1}')

    answer = scripted_server({CHAT_PATH: reply})("POST", CHAT_PATH, {"model": "m"})

    assert (answer.status, answer.body, answer.header(BODY_HEADER)) == (status, b'{"a": 1}', None)


def test_the_fixture_server_keeps_the_status_of_a_chat_above_the_limit_and_raises_for_the_list():
    big = b"x" * (MAX_ANSWER_BYTES + 1)
    fixture = FixtureLmStudioServer(
        {("POST", CHAT_PATH): [Response(status=200, body=big), Response(status=500, body=big)],
         ("GET", MODELS_PATH): Response(status=200, body=big)}
    )

    first, second = fixture("POST", CHAT_PATH, {}), fixture("POST", CHAT_PATH, {})

    assert [(r.status, r.body, r.header(BODY_HEADER)) for r in (first, second)] == [
        (200, b"", BODY_OVERSIZED), (500, b"", BODY_OVERSIZED)
    ]
    with pytest.raises(LmStudioServerError, match="more than"):
        fixture("GET", MODELS_PATH)


def test_no_proxy_handler_no_redirect():
    handlers = LocalLmStudioServer()._opener.handlers
    assert not any(isinstance(handler, urllib.request.ProxyHandler) for handler in handlers)
    assert not any(type(handler) is urllib.request.HTTPRedirectHandler for handler in handlers)
    assert not any(isinstance(handler, urllib.request.HTTPSHandler) for handler in handlers)


def test_a_redirect_is_returned_as_it_came_and_never_followed(loopback):
    lmstudio, recorder = loopback
    recorder.answers = {MODELS_PATH: (302, b"{}", {"Location": "/metrics"}), "/metrics": (200, b"", {})}

    assert lmstudio("GET", MODELS_PATH).status == 302
    assert recorder.seen == [("GET", MODELS_PATH)]


@pytest.mark.parametrize("status", [401, 403])
def test_401_and_403_are_the_key_reason_and_no_authorization_header_is_sent(loopback, status):
    lmstudio, recorder = loopback
    recorder.answers = {MODELS_PATH: (status, b'{"error": "Unauthorized"}', {})}

    with pytest.raises(LmStudioKeyError) as caught:
        lmstudio("GET", MODELS_PATH)

    assert str(caught.value) == KEY_REASON
    assert KEY_REASON == "the LM Studio server asks for an API token; modelroom reads an open local server only"
    assert all("authorization" not in headers for headers in recorder.headers_seen)


def test_a_request_never_carries_an_authorization_header(loopback):
    lmstudio, recorder = loopback
    recorder.answers = {MODELS_PATH: (200, b"{}", {})}

    lmstudio("GET", MODELS_PATH)

    assert recorder.headers_seen and all("authorization" not in headers for headers in recorder.headers_seen)


def test_an_answer_above_4_mib_is_a_fault(loopback):
    lmstudio, recorder = loopback
    recorder.answers = {MODELS_PATH: (200, b"x" * (MAX_ANSWER_BYTES + 1), {})}

    with pytest.raises(LmStudioServerError, match="more than"):
        lmstudio("GET", MODELS_PATH)
    assert MAX_ANSWER_BYTES == 4 * 1024 * 1024


def test_a_server_that_is_not_there_is_an_error_with_a_reason(monkeypatch):
    with pytest.raises(LmStudioServerError, match="did not answer"):
        local_server_at(9, monkeypatch)("GET", MODELS_PATH)


def test_no_constructor_way_admits_another_origin():
    """The origin list is the module's, never an argument (Codex b1-1): another port is refused, and the
    constructor has no parameter that could widen the list."""
    import inspect

    with pytest.raises(LmStudioServerError, match="not a local LM Studio server origin"):
        LocalLmStudioServer("http://127.0.0.1:4321")
    with pytest.raises(TypeError):
        LocalLmStudioServer("http://127.0.0.1:4321", allowed_origins=frozenset({("127.0.0.1", 4321)}))
    assert list(inspect.signature(LocalLmStudioServer).parameters) == ["base_url", "timeout", "chat_timeout"]


def test_an_origin_assigned_after_the_construction_is_refused_before_a_connection():
    """Codex b2-1: `base_url` is a public attribute, so the origin is judged again at every call."""
    lmstudio = LocalLmStudioServer()

    class _Trap:
        @staticmethod
        def open(*args, **kwargs):
            pytest.fail("a connection was opened", pytrace=False)

    lmstudio._opener = _Trap()
    for other in ("http://127.0.0.1:4321", "http://localhost:1234", "http://127.0.0.1:1234/api"):
        lmstudio.base_url = other
        with pytest.raises(LmStudioServerError, match="not a local LM Studio server origin"):
            lmstudio("GET", MODELS_PATH)


def _refusing_opener(status: int):
    """An opener whose answer is an HTTP error of `status` with a body that cannot be read."""
    import io
    import urllib.error

    class _Unreadable(io.RawIOBase):
        def read(self, *args):
            raise OSError("timed out while reading the body")

    def open_(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "refused", {}, _Unreadable())

    return type("Opener", (), {"open": staticmethod(open_)})()


@pytest.mark.parametrize("status", [401, 403])
def test_a_401_or_403_is_the_key_reason_even_when_its_body_cannot_be_read(status):
    """Codex b1-2: the status decides, the body is never needed."""
    from modelroom.guided_lmstudio import start_fact
    from modelroom.runtimes import Origins

    from fixture_lmstudio import run_of

    lmstudio = LocalLmStudioServer()
    lmstudio._opener = _refusing_opener(status)

    with pytest.raises(LmStudioKeyError) as caught:
        lmstudio("GET", MODELS_PATH)
    assert str(caught.value) == KEY_REASON
    assert detect_lmstudio(lmstudio, system="linux", which=_nothing_on_path, home=Path("/nowhere")) == "asks_key"
    run = run_of(lmstudio)
    fact = start_fact(Origins(daemon=None, servers=run.servers))
    assert (fact.value, fact.note) == ("127.0.0.1:1234", "asks for an API token")


def test_another_status_with_an_unreadable_body_is_still_a_fault_with_the_status():
    lmstudio = LocalLmStudioServer()
    lmstudio._opener = _refusing_opener(500)

    with pytest.raises(LmStudioServerError, match="status 500, no body"):
        lmstudio("GET", MODELS_PATH)


# --- 401 and 403 by call: the list asks for a token, the chat has a status like any other -----------------------


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("form", ["error status", "returned status"])
def test_a_401_or_403_of_the_list_is_the_key_reason_on_the_real_transport_in_both_forms(status, form):
    if form == "error status":
        reply = http_error(status, read_error=ValueError("the body was read"))
    else:
        reply = FakeAnswer(status, read_error=ValueError("the body was read"))
    lmstudio = scripted_server({MODELS_PATH: reply})

    with pytest.raises(LmStudioKeyError):
        lmstudio("GET", MODELS_PATH)

    assert (reply.fp.reads, reply.fp.closed) == (0, True) if form == "error status" else reply.closed  # closed, not read


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("form", ["error status", "returned status"])
def test_a_401_or_403_of_the_chat_is_handed_on_with_its_status_on_the_real_transport(status, form):
    reply = http_error(status, b'{"error": "no"}') if form == "error status" else FakeAnswer(status, b'{"error": "no"}')

    answer = scripted_server({CHAT_PATH: reply})("POST", CHAT_PATH, {"model": "m"})

    assert (answer.status, answer.body) == (status, b'{"error": "no"}')


@pytest.mark.parametrize("status", [401, 403])
def test_a_401_or_403_by_call_on_the_fixture_server(status):
    fixture = FixtureLmStudioServer(
        {("GET", MODELS_PATH): Response(status=status, body=b"{}"), ("POST", CHAT_PATH): Response(status=status, body=b"{}")}
    )

    with pytest.raises(LmStudioKeyError):
        fixture("GET", MODELS_PATH)
    assert fixture("POST", CHAT_PATH, {"model": "m"}).status == status


def test_a_401_of_the_chat_whose_body_hits_the_limit_of_time_is_the_limit_not_the_status():
    lmstudio = scripted_server({CHAT_PATH: http_error(401, read_error=TimeoutError("timed out"))})

    with pytest.raises(LmStudioTimeoutError):
        lmstudio("POST", CHAT_PATH, {"model": "m"})


# --- the fixture server ---------------------------------------------------------------------------


def test_the_fixture_server_checks_the_calls_and_can_be_unreachable():
    with pytest.raises(LmStudioServerError, match="refused"):
        FixtureLmStudioServer({}, unreachable="refused")("GET", MODELS_PATH)
    with pytest.raises(LmStudioServerError, match="refusing to call"):
        FixtureLmStudioServer({})("GET", "/health")


def test_the_fixture_server_records_its_calls_and_gives_the_key_reason_for_a_401():
    fixture = FixtureLmStudioServer({("GET", MODELS_PATH): Response(status=401, body=b"{}")})

    with pytest.raises(LmStudioKeyError):
        fixture("GET", MODELS_PATH)
    assert fixture.calls == [("GET", MODELS_PATH, None)]


# --- the model list ---------------------------------------------------------------------------------


def test_the_recorded_list_reads_one_loaded_gguf_model():
    fixture = FixtureLmStudioServer({("GET", MODELS_PATH): json_response("lmstudio_v1_models_loaded.json")})

    models = read_lmstudio_models(fixture)

    assert models == LmStudioModels(
        loaded=(LoadedModel(instance_id=LLM_KEY, key=LLM_KEY, context_length=8192, parallel=4),),
        downloaded=1,
        keys=frozenset({LLM_KEY, EMBEDDING_KEY}),
    )
    assert fixture.calls == [("GET", MODELS_PATH, None)]


def test_the_recorded_list_without_loaded_model_reads_none_loaded():
    for name in ("lmstudio_jit_on_after.json", "lmstudio_jit_on_identifier_after.json"):
        models = read_lmstudio_models(_server((FIXTURES / name).read_bytes()))
        assert models.loaded == () and models.downloaded == 1


def test_mlx_and_embedding_entries_are_left_out():
    payload = _list()
    mlx = copy.deepcopy(_llm(payload))
    mlx.update(key="mlx-model", format="mlx")
    mlx["loaded_instances"][0]["id"] = "mlx-model"
    unformatted = copy.deepcopy(mlx)
    unformatted.update(key="no-format", format=None)
    unformatted["loaded_instances"][0]["id"] = "no-format"
    embedding = payload["models"][1]
    embedding["loaded_instances"] = [{"id": EMBEDDING_KEY, "config": {"context_length": 2048}}]
    payload["models"] += [mlx, unformatted]

    models = read_lmstudio_models(_server(payload))

    assert [loaded.instance_id for loaded in models.loaded] == [LLM_KEY]
    assert models.downloaded == 1


def test_a_model_without_loaded_instances_is_downloaded_not_loaded():
    payload = _list()
    _llm(payload)["loaded_instances"] = []

    models = read_lmstudio_models(_server(payload))

    assert models.loaded == () and models.downloaded == 1


@pytest.mark.parametrize(
    "unsafe",
    ["a\x1b[31mb", "a\nb", "a b", "a" * 200, "modèl", "-leading", "", "a\u202eb", "a\x00b"],
    ids=["escape", "newline", "space", "200-characters", "non-ascii", "leading-dash", "empty", "bidi", "nul"],
)
def test_an_unsafe_instance_id_is_left_out(unsafe):
    payload = _list()
    _llm(payload)["loaded_instances"][0]["id"] = unsafe

    models = read_lmstudio_models(_server(payload))

    assert models.loaded == ()
    assert models.downloaded == 1


def test_an_unsafe_key_leaves_the_entry_out_but_stays_in_the_keys():
    payload = _list()
    _llm(payload)["key"] = "bad key"
    _llm(payload)["loaded_instances"][0]["id"] = "fine"

    models = read_lmstudio_models(_server(payload))

    assert models.loaded == () and models.downloaded == 0
    assert "bad key" in models.keys


def test_a_foreign_json_on_port_1234_is_not_serving():
    for body in (b'{"data": []}', (FIXTURES / "lmstudio_openai_models_jit_on.json").read_bytes(), b"[]", b"not json"):
        with pytest.raises(LmStudioServerError):
            read_lmstudio_models(_server(body))


def test_a_status_other_than_200_is_an_error_with_the_status():
    with pytest.raises(LmStudioServerError, match="status 500"):
        read_lmstudio_models(FixtureLmStudioServer({("GET", MODELS_PATH): Response(status=500, body=b"{}")}))


def test_an_empty_model_list_is_serving_with_nothing_measurable():
    models = read_lmstudio_models(_server((FIXTURES / "lmstudio_v1_models_empty.json").read_bytes()))

    assert models == LmStudioModels(loaded=(), downloaded=0, keys=frozenset())


def test_a_list_of_embedding_or_mlx_entries_only_is_serving_with_nothing_measurable():
    embedding_only = read_lmstudio_models(_server((FIXTURES / "lmstudio_v1_models_embedding_only.json").read_bytes()))
    payload = _list()
    mlx = _llm(payload)
    mlx.update(format="mlx")
    payload["models"] = [mlx]
    mlx_only = read_lmstudio_models(_server(payload))

    assert embedding_only.loaded == () and embedding_only.downloaded == 0 and embedding_only.keys == {EMBEDDING_KEY}
    assert mlx_only.loaded == () and mlx_only.downloaded == 0


@pytest.mark.parametrize(
    "body",
    [b'{"models": [{}]}', b'{"models": [1, "x"]}', b'{"models": {}}', b'{"models": "none"}', b"{}"],
    ids=["empty-object", "no-objects", "models-dict", "models-text", "no-models"],
)
def test_a_list_without_any_structurally_valid_entry_is_not_serving(body):
    with pytest.raises(LmStudioServerError):
        read_lmstudio_models(_server(body))
    # the recorded stand-in file of the same shape
    with pytest.raises(LmStudioServerError):
        read_lmstudio_models(_server((FIXTURES / "lmstudio_v1_models_no_valid_entry.json").read_bytes()))


def test_an_invalid_entry_next_to_a_valid_one_is_left_out():
    payload = _list()
    payload["models"] += [{}, 3, {"key": "x", "type": "llm"}, {"key": "y", "type": "llm", "size_bytes": True, "loaded_instances": []}]

    models = read_lmstudio_models(_server(payload))

    assert [loaded.instance_id for loaded in models.loaded] == [LLM_KEY]
    assert models.downloaded == 1


def test_keys_hold_every_key_before_any_filter():
    payload = _list()
    payload["models"] += [
        {"key": "mlx-key", "type": "llm", "format": "mlx", "size_bytes": 1, "loaded_instances": []},
        {"key": "embedding-key", "type": "embedding", "format": "gguf", "size_bytes": 1, "loaded_instances": []},
        {"key": "broken-key", "type": "llm"},  # structurally invalid, still a key
        {"key": 7, "type": "llm"},  # no string, no key
    ]

    models = read_lmstudio_models(_server(payload))

    assert models.keys == frozenset({LLM_KEY, EMBEDDING_KEY, "mlx-key", "embedding-key", "broken-key"})


def test_an_entry_outside_the_models_folder_is_no_error():
    """The embedding entry ships with the application (A-LS3): it falls out of the measurable list, no fault."""
    models = read_lmstudio_models(FixtureLmStudioServer({("GET", MODELS_PATH): json_response("lmstudio_v1_models_loaded.json")}))

    assert EMBEDDING_KEY in models.keys and [loaded.key for loaded in models.loaded] == [LLM_KEY]


def test_numbers_are_whole_numbers_and_a_bool_is_none():
    payload = _list()
    instance = _llm(payload)["loaded_instances"][0]
    instance["config"]["parallel"] = True
    assert read_lmstudio_models(_server(payload)).loaded == ()
    instance["config"]["parallel"] = None  # present and null: not a number (Codex b1-4)
    assert read_lmstudio_models(_server(payload)).loaded == ()
    del instance["config"]["parallel"]  # absent: allowed
    assert read_lmstudio_models(_server(payload)).loaded[0].parallel is None
    instance["config"]["context_length"] = True
    assert read_lmstudio_models(_server(payload)).loaded == ()
    instance["config"]["context_length"] = 0
    assert read_lmstudio_models(_server(payload)).loaded == ()
    del instance["config"]
    assert read_lmstudio_models(_server(payload)).loaded == ()


def test_the_reader_makes_one_call_and_it_is_the_list():
    fixture = FixtureLmStudioServer({("GET", MODELS_PATH): json_response("lmstudio_v1_models_loaded.json")})

    read_lmstudio_models(fixture)

    assert [(method, path) for method, path, _body in fixture.calls] == [("GET", "/api/v1/models")]


# --- detection --------------------------------------------------------------------------------------


def _home_with(tmp_path: Path, name: str | None) -> Path:
    home = tmp_path / "home"
    (home / ".lmstudio" / "bin").mkdir(parents=True, exist_ok=True)
    if name is not None:
        (home / ".lmstudio" / "bin" / name).write_bytes(b"")
    return home


def _serving() -> FixtureLmStudioServer:
    return FixtureLmStudioServer({("GET", MODELS_PATH): json_response("lmstudio_v1_models_loaded.json")})


def _nothing_on_path(_name: str) -> str | None:
    return None


def test_detect_serving_asks_key_installed_absent(tmp_path):
    bare = _home_with(tmp_path, None)
    asks = FixtureLmStudioServer({("GET", MODELS_PATH): Response(status=403, body=b"{}")})
    foreign = _server(b'{"data": []}')
    kwargs = {"system": "linux", "home": bare}

    assert detect_lmstudio(_serving(), which=_nothing_on_path, **kwargs) == "serving"
    assert detect_lmstudio(asks, which=lambda name: "/bin/lms", **kwargs) == "asks_key"
    assert detect_lmstudio(foreign, which=lambda name: "/bin/lms", **kwargs) == "installed"
    assert detect_lmstudio(None, which=lambda name: "/bin/lms", **kwargs) == "installed"
    assert detect_lmstudio(FixtureLmStudioServer({}, unreachable="refused"), which=_nothing_on_path, **kwargs) == "absent"
    assert detect_lmstudio(None, which=_nothing_on_path, **kwargs) == "absent"


def test_detect_finds_the_program_under_the_home_folder_by_the_name_of_the_platform(tmp_path):
    windows = _home_with(tmp_path / "w", "lms.exe")
    unix = _home_with(tmp_path / "u", "lms")

    assert detect_lmstudio(None, system="win32", which=_nothing_on_path, home=windows) == "installed"
    assert detect_lmstudio(None, system="linux", which=_nothing_on_path, home=unix) == "installed"
    assert detect_lmstudio(None, system="darwin", which=_nothing_on_path, home=unix) == "installed"
    assert detect_lmstudio(None, system="linux", which=_nothing_on_path, home=windows) == "absent"
    assert detect_lmstudio(None, system="win32", which=_nothing_on_path, home=unix) == "absent"


def test_detect_never_runs_lms(tmp_path, monkeypatch):
    """Detection is a path search and a file test: a process started for it fails the test."""
    import os
    import subprocess

    def refuse(*args, **kwargs):
        raise AssertionError("detection started a process")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)
    home = _home_with(tmp_path, "lms")

    assert detect_lmstudio(_serving(), system="linux", which=lambda name: "/bin/lms", home=home) == "serving"
    assert detect_lmstudio(None, system="linux", which=lambda name: "/bin/lms", home=home) == "installed"
    assert lms_call(system="linux", which=lambda name: "/bin/lms", home=home) == "path"


def test_lms_call_is_path_home_or_none(tmp_path):
    windows = _home_with(tmp_path / "w", "lms.exe")
    unix = _home_with(tmp_path / "u", "lms")
    bare = _home_with(tmp_path / "b", None)

    assert lms_call(system="linux", which=lambda name: "/bin/lms", home=bare) == "path"
    assert lms_call(system="win32", which=lambda name: "C:/lms.exe", home=windows) == "path"
    assert lms_call(system="win32", which=_nothing_on_path, home=windows) == "home"
    assert lms_call(system="linux", which=_nothing_on_path, home=unix) == "home"
    assert lms_call(system="linux", which=_nothing_on_path, home=bare) is None
    assert lms_call(system="win32", which=_nothing_on_path, home=unix) is None


def test_installed_without_path_is_still_installed(tmp_path):
    home = _home_with(tmp_path, "lms.exe")

    assert detect_lmstudio(None, system="win32", which=_nothing_on_path, home=home) == "installed"
    assert lms_call(system="win32", which=_nothing_on_path, home=home) == "home"
