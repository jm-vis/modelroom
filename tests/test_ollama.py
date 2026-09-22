"""Tests for modelroom.ollama: tags-page parsing and manifest-based package assembly.

The real recorded tags page (`ollama_tags_qwen35.html`, 64 tags, each linked from both a
mobile and a desktop layout row) is used to test `parse_library_tags` in isolation -- it is
much larger than any base model this project's example configuration lists, so the full
`fetch_ollama_area` happy-path tests use a small, hand-written HTML snippet instead, keeping
control over exactly which manifests need a fixture. The three real qwen3.5 manifests (a
shared-digest sibling pair for quantization inheritance, and a tensor-only manifest) still
back every manifest-shape assertion; the HEAD digest header, a missing-header fallback, and
error/budget paths use small synthetic Response objects, matching test_provenance.py's style.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from modelroom.contracts import Architecture, BaseModelSpec
from modelroom.http import BudgetedTransport, Response
from modelroom.ollama import fetch_ollama_area, parse_library_tags

from fixture_support import FIXTURES, build_transport, envelope_response, json_response

RUN_AT = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
TAGS_URL = "https://ollama.com/library/qwen3.5/tags"

_SMALL_TAGS_HTML = (
    b'<html><body>'
    b'<a href="/library/qwen3.5:9b">9b</a>'
    b'<a href="/library/qwen3.5:9b">9b</a>'  # the real page links every tag twice (mobile+desktop)
    b'<a href="/library/qwen3.5:9b-q4_K_M">9b-q4_K_M</a>'
    b'<a href="/library/qwen3.5:9b-mlx-bf16">9b-mlx-bf16</a>'
    b'<a href="/library/qwen3.5:27b">27b</a>'  # a different size, must never be kept for a "9b" base model
    b'</body></html>'
)


def _unknown_architecture(source_repo: str) -> Architecture:
    return Architecture(source_repo=source_repo, source_revision=None, kind="unknown")


def _qwen35_9b(**overrides) -> BaseModelSpec:
    defaults = dict(
        hf_repo="Qwen/Qwen3.5-9B",
        repo_aliases=["Qwen3.5-9B-GGUF"],
        ollama_base="qwen3.5",
        ollama_tag="9b",
        publisher="Qwen",
        parameters_b=9.0,  # a round number so the Ollama size-token check in decide_provenance matches "9b"
        architecture=_unknown_architecture("Qwen/Qwen3.5-9B"),
    )
    defaults.update(overrides)
    return BaseModelSpec(**defaults)


def _manifest_url(tag: str) -> str:
    return f"https://registry.ollama.ai/v2/library/qwen3.5/manifests/{tag}"


def _digest_header_response() -> Response:
    return envelope_response("ollama_manifest_head_9b.json")


def _no_digest_header_response() -> Response:
    return Response(status=200, headers={}, body=b"")


def _happy_path_transport():
    return build_transport(
        {
            ("GET", TAGS_URL): Response(status=200, headers={}, body=_SMALL_TAGS_HTML),
            ("GET", _manifest_url("9b")): json_response("ollama_qwen35_9b.json"),
            ("HEAD", _manifest_url("9b")): _digest_header_response(),
            ("GET", _manifest_url("9b-q4_K_M")): json_response("ollama_qwen35_9b-q4_K_M.json"),
            ("HEAD", _manifest_url("9b-q4_K_M")): _no_digest_header_response(),
            ("GET", _manifest_url("9b-mlx-bf16")): json_response("ollama_qwen35_9b-mlx-bf16.json"),
            ("HEAD", _manifest_url("9b-mlx-bf16")): _no_digest_header_response(),
        }
    )


# --- parse_library_tags, against the real recorded page -----------------------------------


def test_parse_library_tags_extracts_every_unique_tag_from_the_real_page():
    html_text = (FIXTURES / "ollama_tags_qwen35.html").read_text(encoding="utf-8")

    tags = parse_library_tags(html_text, "qwen3.5")

    assert len(tags) == 64
    assert len(tags) == len(set(tags)), "the mobile+desktop duplicate anchors must be de-duplicated"
    assert "9b" in tags
    assert "9b-q4_K_M" in tags


def test_parse_library_tags_ignores_anchors_for_a_different_base():
    tags = parse_library_tags('<a href="/library/other-model:9b">x</a>', "qwen3.5")
    assert tags == []


# --- fetch_ollama_area: which tags are kept ------------------------------------------------


def test_fetch_ollama_area_keeps_only_tags_matching_the_configured_size():
    outcome = fetch_ollama_area(_happy_path_transport(), _qwen35_9b(), RUN_AT)

    assert outcome.status == "complete"
    assert outcome.error is None
    names = {pkg.ollama_name for pkg in outcome.packages}
    assert names == {"qwen3.5:9b", "qwen3.5:9b-q4_K_M", "qwen3.5:9b-mlx-bf16"}


def test_fetch_ollama_area_zero_tags_parsed_is_incomplete():
    transport = build_transport({("GET", TAGS_URL): Response(status=200, headers={}, body=b"<html></html>")})

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error == "no tags parsed"
    assert outcome.packages == []


def test_fetch_ollama_area_no_ollama_configuration_is_trivially_complete_and_empty():
    base_model = _qwen35_9b(ollama_base=None, ollama_tag=None)
    transport = build_transport({})

    outcome = fetch_ollama_area(transport, base_model, RUN_AT)

    assert outcome.status == "complete"
    assert outcome.packages == []
    assert transport.calls == []


# --- gguf vs tensor manifests --------------------------------------------------------------


def test_fetch_ollama_area_gguf_manifest_becomes_a_gguf_package():
    outcome = fetch_ollama_area(_happy_path_transport(), _qwen35_9b(), RUN_AT)

    by_name = {pkg.ollama_name: pkg for pkg in outcome.packages}
    plain = by_name["qwen3.5:9b"]
    assert plain.format == "gguf"
    assert plain.source == "ollama"
    assert plain.manifest_digest == "sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
    assert plain.complete is True
    assert plain.observed_at == RUN_AT
    assert plain.active is True


def test_fetch_ollama_area_tensor_manifest_becomes_a_tensor_package_unresolved_by_format():
    outcome = fetch_ollama_area(_happy_path_transport(), _qwen35_9b(), RUN_AT)

    by_name = {pkg.ollama_name: pkg for pkg in outcome.packages}
    tensor_pkg = by_name["qwen3.5:9b-mlx-bf16"]
    assert tensor_pkg.format == "tensor"
    assert tensor_pkg.provenance == "unresolved"
    assert tensor_pkg.unresolved_reason == "format"


# --- quantization: parsed directly, or inherited from a digest sibling --------------------


def test_fetch_ollama_area_inherits_quantization_from_a_digest_sibling():
    outcome = fetch_ollama_area(_happy_path_transport(), _qwen35_9b(), RUN_AT)

    by_name = {pkg.ollama_name: pkg for pkg in outcome.packages}
    # "9b" alone carries no quantization token of its own; "9b-q4_K_M" is byte-identical
    # (same weights-layer digest), so "9b" inherits Q4_K_M from it.
    assert by_name["qwen3.5:9b"].quantization == "Q4_K_M"
    assert by_name["qwen3.5:9b-q4_K_M"].quantization == "Q4_K_M"


def test_fetch_ollama_area_gguf_package_with_matching_tags_is_metadata_ok():
    outcome = fetch_ollama_area(_happy_path_transport(), _qwen35_9b(), RUN_AT)

    by_name = {pkg.ollama_name: pkg for pkg in outcome.packages}
    assert by_name["qwen3.5:9b-q4_K_M"].provenance == "metadata_ok"


# --- manifest digest: HEAD header, or a hash-of-body fallback -----------------------------


def test_fetch_manifest_digest_falls_back_to_hashing_the_get_body_when_head_has_no_header():
    body = (
        b'{"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model",'
        b'"digest":"sha256:aaaa","size":1}]}'
    )
    transport = build_transport(
        {
            ("GET", TAGS_URL): Response(status=200, headers={}, body=b'<a href="/library/qwen3.5:9b"></a>'),
            ("GET", _manifest_url("9b")): Response(status=200, headers={}, body=body),
            ("HEAD", _manifest_url("9b")): Response(status=200, headers={}, body=b""),
        }
    )

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT)

    expected = f"sha256:{hashlib.sha256(body).hexdigest()}"
    assert outcome.packages[0].manifest_digest == expected


# --- failures end the area incomplete, never raise ----------------------------------------


def test_fetch_ollama_area_tags_page_failure_is_incomplete():
    transport = build_transport({("GET", TAGS_URL): Response(status=500, headers={}, body=b"error")})

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


def test_fetch_ollama_area_manifest_failure_is_incomplete():
    transport = build_transport(
        {
            ("GET", TAGS_URL): Response(status=200, headers={}, body=b'<a href="/library/qwen3.5:9b"></a>'),
            ("GET", _manifest_url("9b")): Response(status=500, headers={}, body=b"error"),
        }
    )

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


def test_fetch_ollama_area_budget_exhausted_mid_area_is_incomplete():
    inner = build_transport(
        {("GET", TAGS_URL): Response(status=200, headers={}, body=b'<a href="/library/qwen3.5:9b"></a>')}
    )
    budgeted = BudgetedTransport(inner, budget=1)
    budgeted("GET", TAGS_URL)  # spend the only slot

    outcome = fetch_ollama_area(budgeted, _qwen35_9b(), RUN_AT)

    assert outcome.status == "incomplete"
    assert "budget exhausted" in outcome.error
