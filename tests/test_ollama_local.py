"""Tests for modelroom.ollama_local: the local Ollama daemon's `/api/tags` inventory."""

from __future__ import annotations

from datetime import datetime, timezone

from modelroom.http import FixtureTransport, Response
from modelroom.ollama_local import DEFAULT_BASE_URL, fetch_installed_models

from fixture_support import json_response

NOW = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)


def _transport(mapping: dict[tuple[str, str], Response]) -> FixtureTransport:
    return FixtureTransport(mapping)


def test_fetch_installed_models_parses_the_recorded_tags_response():
    transport = _transport({("GET", f"{DEFAULT_BASE_URL}/api/tags"): json_response("ollama_tags_local.json")})

    installed, reason = fetch_installed_models(transport, NOW)

    assert reason is None
    assert installed is not None
    assert len(installed) == 3
    assert installed[0].name == "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL"
    assert installed[0].digest == "sha256:c66cf836b05b503a80e2c488e800dd4f1c1ad2497cb93c212cd41582373b8dda"
    assert installed[0].size_bytes == 18490325682
    assert installed[0].observed_at == NOW


def test_fetch_installed_models_normalizes_the_bare_hex_digest():
    # The real daemon returns the digest without a "sha256:" prefix (measured 2026-09-22); every
    # entry in the fixture must come back prefixed.
    transport = _transport({("GET", f"{DEFAULT_BASE_URL}/api/tags"): json_response("ollama_tags_local.json")})
    installed, _ = fetch_installed_models(transport, NOW)
    assert all(model.digest.startswith("sha256:") for model in installed)


def test_fetch_installed_models_transport_error_is_unavailable_not_a_crash():
    def _raise(method: str, url: str, headers=None):
        raise ConnectionRefusedError("no daemon listening")

    installed, reason = fetch_installed_models(_raise, NOW)

    assert installed is None
    assert reason is not None
    assert "no daemon listening" in reason


def test_fetch_installed_models_non_200_status_is_unavailable():
    transport = _transport({("GET", f"{DEFAULT_BASE_URL}/api/tags"): Response(status=500, headers={}, body=b"error")})

    installed, reason = fetch_installed_models(transport, NOW)

    assert installed is None
    assert "500" in reason


def test_fetch_installed_models_malformed_body_is_unavailable():
    transport = _transport({("GET", f"{DEFAULT_BASE_URL}/api/tags"): Response(status=200, headers={}, body=b"not json")})

    installed, reason = fetch_installed_models(transport, NOW)

    assert installed is None
    assert reason is not None


def test_fetch_installed_models_missing_models_field_is_unavailable():
    transport = _transport({("GET", f"{DEFAULT_BASE_URL}/api/tags"): Response(status=200, headers={}, body=b"{}")})

    installed, reason = fetch_installed_models(transport, NOW)

    assert installed is None
    assert reason is not None


def test_fetch_installed_models_empty_models_list_is_available_and_empty():
    transport = _transport({("GET", f"{DEFAULT_BASE_URL}/api/tags"): Response(status=200, headers={}, body=b'{"models": []}')})

    installed, reason = fetch_installed_models(transport, NOW)

    assert reason is None
    assert installed == []


def test_fetch_installed_models_respects_a_custom_base_url():
    transport = _transport({("GET", "http://example-host:11434/api/tags"): json_response("ollama_tags_local.json")})

    installed, reason = fetch_installed_models(transport, NOW, base_url="http://example-host:11434")

    assert reason is None
    assert len(installed) == 3
