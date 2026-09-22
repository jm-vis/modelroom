"""Tests for modelroom.ollama: tags-page parsing and manifest-based package assembly.

The real recorded tags page (`ollama_tags_qwen35.html`, 64 tags, each linked from both a
mobile and a desktop layout row) is used to test `parse_library_tags` in isolation -- it is
much larger than any base model this project's example configuration lists, so the full
`fetch_ollama_area` happy-path tests use a small, hand-written HTML snippet instead, keeping
control over exactly which manifests need a fixture. The three real qwen3.5 manifests (a
shared-digest sibling pair for quantization inheritance, and a tensor-only manifest) still back
every manifest-shape assertion; the digest is always `sha256:` + sha256 of the manifest `GET`
body (F5 -- no `HEAD` request is made at all any more, measured 2026-09-22 to equal the old
`HEAD` header exactly for the `9b` fixture, see tests/fixtures/README.md); error/budget paths
use small synthetic Response objects, matching test_provenance.py's style.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone

from modelroom.contracts import Approval, Architecture, BaseModelSpec, Package, PackageFile
from modelroom.http import BudgetedTransport, Response
from modelroom.ollama import fetch_ollama_area, parse_library_tags
from modelroom.quantization import package_identity_key

from fixture_support import FIXTURES, build_transport, json_response

RUN_AT = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
TAGS_URL = "https://ollama.com/library/qwen3.5/tags"
_MANIFEST_9B_DIGEST = f"sha256:{hashlib.sha256((FIXTURES / 'ollama_qwen35_9b.json').read_bytes()).hexdigest()}"

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


def _happy_path_transport():
    return build_transport(
        {
            ("GET", TAGS_URL): Response(status=200, headers={}, body=_SMALL_TAGS_HTML),
            ("GET", _manifest_url("9b")): json_response("ollama_qwen35_9b.json"),
            ("GET", _manifest_url("9b-q4_K_M")): json_response("ollama_qwen35_9b-q4_K_M.json"),
            ("GET", _manifest_url("9b-mlx-bf16")): json_response("ollama_qwen35_9b-mlx-bf16.json"),
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
    assert plain.manifest_digest == _MANIFEST_9B_DIGEST
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


# --- manifest digest: sha256 of the GET body, no HEAD request at all (F5) ------------------


def test_fetch_manifest_digest_is_sha256_of_the_get_body_and_makes_exactly_one_call():
    from modelroom.ollama import _fetch_manifest

    body = (FIXTURES / "ollama_qwen35_9b.json").read_bytes()
    transport = build_transport({("GET", _manifest_url("9b")): Response(status=200, headers={}, body=body)})

    manifest, digest = _fetch_manifest(transport, "qwen3.5", "9b")

    assert digest == f"sha256:{hashlib.sha256(body).hexdigest()}"
    assert digest == _MANIFEST_9B_DIGEST  # matches the real HEAD header exactly (see fixtures/README.md)
    assert transport.calls == [("GET", _manifest_url("9b"))]


def test_fetch_ollama_area_transport_mapping_with_no_head_entry_at_all_passes():
    # F5: a transport mapping that carries no HEAD entry for any manifest URL must still let
    # the whole area complete -- if the fetcher ever made a HEAD call, FixtureTransport would
    # raise KeyError immediately (no HEAD entry is recorded here).
    transport = _happy_path_transport()

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT)

    assert outcome.status == "complete"
    assert all(method == "GET" for method, _ in transport.calls)


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


# --- F3: a malformed manifest layer ends the area incomplete, never raises -----------------


def test_fetch_ollama_area_manifest_layer_not_a_dict_ends_area_incomplete():
    body = b'{"schemaVersion":2,"layers":[null]}'
    transport = build_transport(
        {
            ("GET", TAGS_URL): Response(status=200, headers={}, body=b'<a href="/library/qwen3.5:9b"></a>'),
            ("GET", _manifest_url("9b")): Response(status=200, headers={}, body=body),
        }
    )

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


# --- F4: a weights layer without a size is a shape error, the area ends incomplete ---------


def test_fetch_ollama_area_weights_layer_without_size_ends_area_incomplete():
    body = (
        b'{"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model",'
        b'"digest":"sha256:' + b"a" * 64 + b'"}]}'  # no "size" field at all
    )
    transport = build_transport(
        {
            ("GET", TAGS_URL): Response(status=200, headers={}, body=b'<a href="/library/qwen3.5:9b"></a>'),
            ("GET", _manifest_url("9b")): Response(status=200, headers={}, body=body),
        }
    )

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


# --- F6: a previous approval survives a fetch when its content still matches ---------------


def test_fetch_ollama_area_carries_forward_an_approval_bound_to_the_current_digest():
    transport = _happy_path_transport()
    previous = Package(
        source="ollama",
        ollama_name="qwen3.5:9b",
        manifest_digest=_MANIFEST_9B_DIGEST,
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format="gguf",
        files=[PackageFile(name="9b.gguf", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization="Q4_K_M",
        default_context=None,
        provenance="approved",
        unresolved_reason=None,
        approval=Approval(date=date(2026, 9, 1), content=_MANIFEST_9B_DIGEST, by="acme-ai-team"),
        observed_at=RUN_AT,
        last_seen=RUN_AT,
        active=True,
    )
    previous_by_key = {package_identity_key(previous): previous}

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT, previous_by_key=previous_by_key)

    by_name = {pkg.ollama_name: pkg for pkg in outcome.packages}
    assert by_name["qwen3.5:9b"].provenance == "approved"
    assert by_name["qwen3.5:9b"].approval.content == _MANIFEST_9B_DIGEST


def test_fetch_ollama_area_approval_bound_to_an_older_digest_is_not_approved():
    transport = _happy_path_transport()
    stale_content = "sha256:" + "1" * 64
    previous = Package(
        source="ollama",
        ollama_name="qwen3.5:9b",
        manifest_digest=stale_content,
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format="gguf",
        files=[PackageFile(name="9b.gguf", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization="Q4_K_M",
        default_context=None,
        provenance="approved",
        unresolved_reason=None,
        approval=Approval(date=date(2026, 1, 1), content=stale_content, by="acme-ai-team"),
        observed_at=RUN_AT,
        last_seen=RUN_AT,
        active=True,
    )
    previous_by_key = {package_identity_key(previous): previous}

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT, previous_by_key=previous_by_key)

    by_name = {pkg.ollama_name: pkg for pkg in outcome.packages}
    assert by_name["qwen3.5:9b"].provenance != "approved"


def test_fetch_ollama_area_approval_does_not_follow_the_package_identity_to_another_base_model():
    # R3: same rule as hf.py's -- an approval is bound to the base model it was given for, not
    # just the bare ollama_name identity.
    transport = _happy_path_transport()
    previous = Package(
        source="ollama",
        ollama_name="qwen3.5:9b",
        manifest_digest=_MANIFEST_9B_DIGEST,
        base_model_hf_repo="SomeOther/Different-Model",  # not Qwen/Qwen3.5-9B
        format="gguf",
        files=[PackageFile(name="9b.gguf", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization="Q4_K_M",
        default_context=None,
        provenance="approved",
        unresolved_reason=None,
        approval=Approval(date=date(2026, 9, 1), content=_MANIFEST_9B_DIGEST, by="acme-ai-team"),
        observed_at=RUN_AT,
        last_seen=RUN_AT,
        active=True,
    )
    previous_by_key = {package_identity_key(previous): previous}

    outcome = fetch_ollama_area(transport, _qwen35_9b(), RUN_AT, previous_by_key=previous_by_key)

    by_name = {pkg.ollama_name: pkg for pkg in outcome.packages}
    assert by_name["qwen3.5:9b"].provenance != "approved"
    assert by_name["qwen3.5:9b"].approval is None


def test_fetch_ollama_area_budget_exhausted_mid_area_is_incomplete():
    inner = build_transport(
        {("GET", TAGS_URL): Response(status=200, headers={}, body=b'<a href="/library/qwen3.5:9b"></a>')}
    )
    budgeted = BudgetedTransport(inner, budget=1)
    budgeted("GET", TAGS_URL)  # spend the only slot

    outcome = fetch_ollama_area(budgeted, _qwen35_9b(), RUN_AT)

    assert outcome.status == "incomplete"
    assert "budget exhausted" in outcome.error
