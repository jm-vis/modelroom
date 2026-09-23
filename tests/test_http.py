"""Tests for modelroom.http: the Transport protocol, FixtureTransport and the request budget.

Most tests here never touch the network: UrllibTransport's use of urllib.request is exercised
indirectly (its request-shaping logic) without opening a real socket, by constructing it and
checking the request object it would send is never enough on its own -- so the real transport
is covered mostly by construction/attribute tests here, while every fetcher test in
test_hf.py/test_ollama.py drives the same Transport protocol through FixtureTransport.

Fix-round 5 (P2-2) adds one exception: a real loopback `http.server` proves `_NoAutoRedirect`
actually stops `urllib.request`'s own redirect following (nothing else can prove that without a
real HTTP response carrying a real `Location` header), and doubles as the test bed for P2-1's
allow-list refusal of a redirect target outside it.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.parse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import modelroom
from modelroom.http import (
    USER_AGENT,
    BudgetedTransport,
    BudgetExhaustedError,
    FixtureTransport,
    RedirectingTransport,
    Response,
    TransportSecurityError,
    UrllibTransport,
    _ALLOWED_HTTP_ORIGINS,
    _ALLOWED_HTTPS_HOSTS,
    _check_allowed,
)
from modelroom.ollama_local import DEFAULT_BASE_URL


@pytest.fixture(autouse=True)
def _no_proxy_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """R8-2 (fix-round 7): every test in this module starts without a proxy environment.

    `urllib.request.getproxies_environment` reads every `*_proxy` variable case-insensitively, and
    on Unix a lowercase `http_proxy`/`no_proxy` wins over the uppercase spelling a test sets --
    so a developer machine or CI runner with a proxy configured would route the loopback tests
    through a foreign proxy, or make the proxy tests below pass or fail for the wrong reason.
    Each proxy test sets exactly the variables it needs on top of this clean slate. (On Windows,
    with no environment proxies at all, urllib falls back to the registry proxy settings; those
    are outside this fixture's reach and are the operator's `NO_PROXY` concern, see CONTRACTS.md
    "Transport security and proxying".)
    """
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)


class _RedirectHandler(BaseHTTPRequestHandler):
    """`/a` answers 302 to whatever `redirect_target` currently holds, `/b` answers 200."""

    redirect_target = "/b"

    def do_GET(self) -> None:  # noqa: N802 -- stdlib override
        self.server.requests_seen.append(self.path)  # type: ignore[attr-defined]
        if self.path == "/a":
            self.send_response(302)
            self.send_header("Location", self.redirect_target)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- stdlib override
        pass  # silence the default stderr access log


@contextmanager
def _loopback_server(redirect_target: str = "/b"):
    """A real `ThreadingHTTPServer` on `127.0.0.1:0`; yields its base URL and its request log."""
    handler = type("_Handler", (_RedirectHandler,), {"redirect_target": redirect_target})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.requests_seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server.requests_seen
    finally:
        server.shutdown()
        thread.join()


def test_user_agent_carries_the_package_version():
    assert USER_AGENT == f"modelroom/{modelroom.__version__}"


def test_response_json_decodes_the_body():
    response = Response(status=200, headers={}, body=json.dumps({"a": 1}).encode("utf-8"))
    assert response.json() == {"a": 1}


def test_response_header_lookup_is_case_insensitive():
    response = Response(status=200, headers={"Content-Type": "application/json"}, body=b"{}")
    assert response.header("content-type") == "application/json"
    assert response.header("Content-Type") == "application/json"
    assert response.header("missing") is None


def test_urllib_transport_constructs_with_a_default_timeout():
    transport = UrllibTransport()
    assert transport._timeout > 0


# --- P2-1: UrllibTransport refuses file://, ftp:// and any host outside its allow-list --------
#
# Probe (fix-round 5 brief): UrllibTransport()("GET", "file:///.../pyvenv.cfg") returned status
# None and 178 bytes of a local file -- build_opener(_NoAutoRedirect) still carries urllib's
# default FileHandler/FTPHandler/DataHandler/HTTPHandler alongside it. Every case below must
# raise before urllib ever opens anything, never a FileNotFoundError/URLError from actually
# trying.


def test_urllib_transport_refuses_a_file_url_without_opening_it():
    transport = UrllibTransport()
    # a path that does not exist: if the transport ever actually opened it, this would raise
    # FileNotFoundError instead of TransportSecurityError.
    missing = "file:///this/path/does/not/exist/pyvenv.cfg"

    with pytest.raises(TransportSecurityError, match="file"):
        transport("GET", missing)


def test_urllib_transport_refuses_an_ftp_url():
    transport = UrllibTransport()
    with pytest.raises(TransportSecurityError):
        transport("GET", "ftp://example.test/x")


def test_urllib_transport_refuses_a_foreign_https_host():
    transport = UrllibTransport()
    with pytest.raises(TransportSecurityError, match="evil.example"):
        transport("GET", "https://evil.example/x")


def test_default_allow_list_accepts_exactly_the_three_hosts_this_package_talks_to():
    """The production default covers exactly `hf.py::HF_API`'s host, `ollama.py::TAGS_URL`'s
    host and `ollama.py::MANIFEST_URL`'s host over HTTPS -- read from those modules, never
    guessed (fix-round 5 brief). Checked against `_check_allowed` directly, never by actually
    opening a socket: this module's tests never touch the real network (see module docstring).
    """
    for url in (
        "https://huggingface.co/api/models/acme/Nova-7B",
        "https://ollama.com/library/nova/tags",
        "https://registry.ollama.ai/v2/library/nova/manifests/7b",
    ):
        _check_allowed(url, _ALLOWED_HTTPS_HOSTS, _ALLOWED_HTTP_ORIGINS)  # must not raise

    with pytest.raises(TransportSecurityError):
        _check_allowed("https://huggingface.co.evil.example/x", _ALLOWED_HTTPS_HOSTS, _ALLOWED_HTTP_ORIGINS)


# --- R7-4 (fix-round 6): the local HTTP allow-list is an origin (host + port), not just a host -
#
# Probe: `_check_allowed("http://127.0.0.1:59999/x", ...)` was accepted with the production
# defaults -- a poisoned `Location`/`Link` header could reach any local port, not only the real
# Ollama daemon's `11434`.


def test_check_allowed_refuses_a_loopback_port_other_than_11434_with_production_defaults():
    with pytest.raises(TransportSecurityError):
        _check_allowed("http://127.0.0.1:59999/x", _ALLOWED_HTTPS_HOSTS, _ALLOWED_HTTP_ORIGINS)


def test_check_allowed_accepts_the_real_ollama_daemon_origin_with_production_defaults():
    _check_allowed("http://127.0.0.1:11434/api/tags", _ALLOWED_HTTPS_HOSTS, _ALLOWED_HTTP_ORIGINS)  # must not raise


def test_check_allowed_refuses_a_loopback_url_with_no_explicit_port():
    """A missing port must never fall back to matching by host alone."""
    with pytest.raises(TransportSecurityError):
        _check_allowed("http://127.0.0.1/api/tags", _ALLOWED_HTTPS_HOSTS, _ALLOWED_HTTP_ORIGINS)


def test_ollama_local_default_base_url_is_in_the_production_http_allow_list():
    """`ollama_local.DEFAULT_BASE_URL` and `_ALLOWED_HTTP_ORIGINS` must name the same origin, or
    the daemon `hardware` actually talks to would be refused by `UrllibTransport`'s own default.
    """
    parsed = urllib.parse.urlsplit(DEFAULT_BASE_URL)
    assert (parsed.hostname, parsed.port) in _ALLOWED_HTTP_ORIGINS


def test_urllib_transport_rejects_loopback_http_on_a_port_other_than_11434_with_production_defaults():
    """The local Ollama daemon's `http://127.0.0.1:11434` is the *only* production default
    loopback origin (F1 text/R7-4): a real server on a different port -- the loopback server
    fixture gives one by construction, since the OS assigns an ephemeral port -- is refused with
    no explicit relaxation, while an allowed HTTPS host still passes the same check.
    """
    with _loopback_server() as (base_url, _requests_seen):
        port = urllib.parse.urlsplit(base_url).port
        assert port != 11434
        transport = UrllibTransport()  # production defaults, no relaxation

        with pytest.raises(TransportSecurityError):
            transport("GET", f"{base_url}/a")

        _check_allowed("https://huggingface.co/api/models/acme/Nova-7B", _ALLOWED_HTTPS_HOSTS, _ALLOWED_HTTP_ORIGINS)


# --- P2-2: a real loopback server proves _NoAutoRedirect actually stops urllib's own following,
# and doubles as P2-1's redirect-refusal test bed ---------------------------------------------


def test_urllib_transport_returns_a_302_raw_instead_of_following_it():
    with _loopback_server() as (base_url, requests_seen):
        port = urllib.parse.urlsplit(base_url).port
        transport = UrllibTransport(allowed_http_origins=frozenset({("127.0.0.1", port)}))

        response = transport("GET", f"{base_url}/a")

        assert response.status == 302
        assert response.header("Location") == "/b"
        assert requests_seen == ["/a"]


def test_redirecting_transport_over_a_real_server_refuses_a_file_url_location():
    with _loopback_server(redirect_target="file:///etc/passwd") as (base_url, requests_seen):
        port = urllib.parse.urlsplit(base_url).port
        transport = UrllibTransport(allowed_http_origins=frozenset({("127.0.0.1", port)}))
        redirecting = RedirectingTransport(transport)

        with pytest.raises(TransportSecurityError, match="file"):
            redirecting("GET", f"{base_url}/a")

        # the poisoned Location was never opened: only the first hop reached the server
        assert requests_seen == ["/a"]


def test_redirecting_transport_over_a_real_server_refuses_a_foreign_host_location():
    with _loopback_server(redirect_target="https://evil.example/x") as (base_url, requests_seen):
        port = urllib.parse.urlsplit(base_url).port
        transport = UrllibTransport(allowed_http_origins=frozenset({("127.0.0.1", port)}))
        redirecting = RedirectingTransport(transport)

        with pytest.raises(TransportSecurityError, match="evil.example"):
            redirecting("GET", f"{base_url}/a")

        assert requests_seen == ["/a"]


# --- R7-5 (fix-round 6): `_build_opener` honors HTTP_PROXY/HTTPS_PROXY/NO_PROXY from the
# environment -- `build_opener` (the previous implementation) added a `ProxyHandler` for free;
# the explicit handler list `_build_opener` replaced it with (P2-1/fix-round 5) dropped that,
# silently losing proxy support. The allow-list check still runs against the TARGET url, before
# the proxy is ever consulted (`UrllibTransport.__call__`: `_check_allowed` first, `opener.open`
# second) -- a proxy can never widen what this transport is willing to reach.


def test_urllib_transport_honors_the_http_proxy_env_var(monkeypatch: pytest.MonkeyPatch):
    """A real loopback server stands in for the proxy: when `urllib.request` actually proxies an
    HTTP request, it sends the request line with the absolute target URL (not a bare path) -- the
    one thing a direct request could never produce, so `requests_seen` capturing the absolute URL
    proves the proxy was used, not just reachable.
    """
    with _loopback_server() as (proxy_base_url, proxy_requests_seen):
        monkeypatch.setenv("HTTP_PROXY", proxy_base_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        # An address in TEST-NET-1 (RFC 5737, guaranteed unroutable): if this were ever connected
        # to directly instead of through the proxy, the request would time out, not succeed.
        target_url = "http://192.0.2.1:80/somepath"
        transport = UrllibTransport(allowed_http_origins=frozenset({("192.0.2.1", 80)}))

        response = transport("GET", target_url)

        assert response.status == 200
        assert proxy_requests_seen == [target_url]


def test_no_proxy_env_var_bypasses_the_proxy_for_the_target_host(monkeypatch: pytest.MonkeyPatch):
    with _loopback_server() as (proxy_base_url, proxy_requests_seen):
        with _loopback_server() as (target_base_url, target_requests_seen):
            monkeypatch.setenv("HTTP_PROXY", proxy_base_url)
            monkeypatch.setenv("NO_PROXY", "127.0.0.1")
            target_port = urllib.parse.urlsplit(target_base_url).port
            transport = UrllibTransport(allowed_http_origins=frozenset({("127.0.0.1", target_port)}))

            response = transport("GET", f"{target_base_url}/direct")

            assert response.status == 200
            assert target_requests_seen == ["/direct"]
            assert proxy_requests_seen == []  # the proxy was never contacted at all


# --- FixtureTransport -----------------------------------------------------------------------


def test_fixture_transport_returns_the_recorded_response_and_records_the_call():
    response = Response(status=200, headers={}, body=b"{}")
    transport = FixtureTransport({("GET", "https://example.test/a"): response})

    result = transport("GET", "https://example.test/a")

    assert result is response
    assert transport.calls == [("GET", "https://example.test/a")]


def test_fixture_transport_raises_on_an_unrecorded_call():
    transport = FixtureTransport({})
    with pytest.raises(KeyError):
        transport("GET", "https://example.test/missing")


def test_fixture_transport_records_every_call_including_repeats():
    response = Response(status=200, headers={}, body=b"{}")
    transport = FixtureTransport({("GET", "https://example.test/a"): response})

    transport("GET", "https://example.test/a")
    transport("GET", "https://example.test/a")

    assert transport.calls == [
        ("GET", "https://example.test/a"),
        ("GET", "https://example.test/a"),
    ]


# --- BudgetedTransport ------------------------------------------------------------------------


def test_budgeted_transport_passes_calls_through_under_budget():
    response = Response(status=200, headers={}, body=b"{}")
    inner = FixtureTransport({("GET", "https://example.test/a"): response})
    budgeted = BudgetedTransport(inner, budget=5)

    result = budgeted("GET", "https://example.test/a")

    assert result is response
    assert budgeted.used == 1
    assert budgeted.remaining == 4


def test_budgeted_transport_raises_once_the_budget_is_exhausted():
    response = Response(status=200, headers={}, body=b"{}")
    inner = FixtureTransport({("GET", "https://example.test/a"): response})
    budgeted = BudgetedTransport(inner, budget=1)

    budgeted("GET", "https://example.test/a")
    with pytest.raises(BudgetExhaustedError, match="budget exhausted"):
        budgeted("GET", "https://example.test/a")

    # the exhausted call never reached the inner transport
    assert inner.calls == [("GET", "https://example.test/a")]


def test_budgeted_transport_default_budget_is_400():
    inner = FixtureTransport({})
    budgeted = BudgetedTransport(inner)
    assert budgeted.remaining == 400


# --- R5: RedirectingTransport follows a chain by calling `inner` once per hop --------------
#
# R5 inverts F10's composition: UrllibTransport makes exactly one request per call and returns
# a 3xx as an ordinary Response; RedirectingTransport(inner) is what follows the chain, calling
# `inner` once per hop, so the run's request budget (BudgetedTransport) can be composed as the
# *inner* of a RedirectingTransport and see -- and book -- every hop before it is made, failure
# paths included. Response.requests_made and BudgetedTransport's post-call accounting are gone.


def test_redirecting_transport_follows_a_single_redirect_and_costs_two_requests():
    inner = FixtureTransport(
        {
            ("GET", "https://example.test/a"): Response(status=302, headers={"Location": "https://example.test/b"}, body=b""),
            ("GET", "https://example.test/b"): Response(status=200, headers={}, body=b'{"ok": true}'),
        }
    )
    budgeted = BudgetedTransport(inner, budget=10)
    redirecting = RedirectingTransport(budgeted)

    result = redirecting("GET", "https://example.test/a")

    assert result.status == 200
    assert result.json() == {"ok": True}
    assert budgeted.used == 2
    assert budgeted.remaining == 8


def test_redirecting_transport_a_six_hop_chain_raises_and_books_only_five_requests():
    # R5: a chain needing 6 requests to resolve exceeds MAX_REDIRECTS (5) -- exactly 5 requests
    # are made (booked against the budget) and the would-be sixth is never made.
    mapping = {}
    for i in range(1, 6):
        mapping[("GET", f"https://example.test/{i}")] = Response(
            status=302, headers={"Location": f"https://example.test/{i + 1}"}, body=b""
        )
    # deliberately no entry for https://example.test/6 -- it must never be requested
    inner = FixtureTransport(mapping)
    budgeted = BudgetedTransport(inner, budget=10)
    redirecting = RedirectingTransport(budgeted)

    with pytest.raises(RuntimeError, match="redirect"):
        redirecting("GET", "https://example.test/1")

    assert budgeted.used == 5
    assert ("GET", "https://example.test/6") not in inner.calls


def test_redirecting_transport_budget_of_one_raises_budget_exhausted_on_the_second_hop():
    inner = FixtureTransport(
        {
            ("GET", "https://example.test/a"): Response(status=302, headers={"Location": "https://example.test/b"}, body=b""),
            ("GET", "https://example.test/b"): Response(status=200, headers={}, body=b"{}"),
        }
    )
    budgeted = BudgetedTransport(inner, budget=1)
    redirecting = RedirectingTransport(budgeted)

    with pytest.raises(BudgetExhaustedError):
        redirecting("GET", "https://example.test/a")

    assert inner.calls == [("GET", "https://example.test/a")]  # the second hop was never made


def test_redirecting_transport_never_follows_a_redirect_for_a_non_get_head_method():
    inner = FixtureTransport(
        {("POST", "https://example.test/a"): Response(status=302, headers={"Location": "https://example.test/b"}, body=b"")}
    )
    redirecting = RedirectingTransport(inner)

    result = redirecting("POST", "https://example.test/a")

    assert result.status == 302
    assert inner.calls == [("POST", "https://example.test/a")]


def test_redirecting_transport_a_redirect_status_without_a_location_header_is_returned_as_is():
    inner = FixtureTransport({("GET", "https://example.test/a"): Response(status=302, headers={}, body=b"")})
    redirecting = RedirectingTransport(inner)

    result = redirecting("GET", "https://example.test/a")

    assert result.status == 302
