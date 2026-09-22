"""Ollama fetching: tags for a base model's library page, and one package per kept tag.

Only base models that configure `ollama_base`/`ollama_tag` are looked at; everything else is
a trivially "complete, empty" area (CONTRACTS.md/AGENTS.md's Ollama tier only ever names a
base model this project's configuration already lists -- there is no discovery step here).

The tags page is plain, unauthenticated web data extraction: one `GET` of one page per
configured base model, the tag names read out of `<a href="/library/<base>:<tag>">` anchors
with the standard-library `html.parser`, nothing else taken from the page, no login, no
crawling beyond it (see the `datenextraktion` skill this project follows for scraping work).
The manifest digest is read from a `HEAD` request's `ollama-content-digest` header, the only
place the registry states it (measured 2026-09-22, see tests/fixtures/README.md); a `HEAD`
response without that header falls back to hashing the `GET` body.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from html.parser import HTMLParser
from typing import Literal

from .contracts import BaseModelSpec, Package, PackageFile, shards_complete
from .fetch_types import AreaOutcome
from .http import Transport
from .provenance import decide_provenance
from .quantization import inherit_from_siblings, parse_ollama_tag_quant

TAGS_URL = "https://ollama.com/library/{base}/tags"
MANIFEST_URL = "https://registry.ollama.ai/v2/library/{base}/manifests/{tag}"

_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
_TENSOR_MEDIA_TYPE = "application/vnd.ollama.image.tensor"


class _TagLinkParser(HTMLParser):
    """Collects the tag part of every `<a href="/library/<base>:<tag>">` anchor."""

    def __init__(self, base: str) -> None:
        super().__init__()
        self._prefix = f"/library/{base}:"
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href and href.startswith(self._prefix):
            self.tags.append(href[len(self._prefix) :])


def parse_library_tags(html_text: str, ollama_base: str) -> list[str]:
    """Every unique tag name linked from an Ollama library "tags" page, in first-seen order.

    The real page links each tag from two layout variants (a mobile and a desktop row); this
    de-duplicates them.
    """
    parser = _TagLinkParser(ollama_base)
    parser.feed(html_text)
    seen: set[str] = set()
    tags: list[str] = []
    for tag in parser.tags:
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def _keep_relevant_tags(tags: list[str], ollama_tag: str) -> list[str]:
    return [tag for tag in tags if tag == ollama_tag or tag.startswith(f"{ollama_tag}-")]


def _incomplete(base_model: BaseModelSpec, error: str) -> AreaOutcome:
    return AreaOutcome(
        source="ollama", base_model_hf_repo=base_model.hf_repo, packager=None, status="incomplete", error=error, packages=[]
    )


def fetch_ollama_area(transport: Transport, base_model: BaseModelSpec, run_at: datetime) -> AreaOutcome:
    """Fetch every kept tag of `base_model`'s Ollama library entry and assemble their packages."""
    if not base_model.ollama_base or not base_model.ollama_tag:
        return AreaOutcome(
            source="ollama", base_model_hf_repo=base_model.hf_repo, packager=None, status="complete", error=None, packages=[]
        )

    try:
        tags_response = transport("GET", TAGS_URL.format(base=base_model.ollama_base))
    except Exception as exc:
        return _incomplete(base_model, str(exc))
    if tags_response.status != 200:
        return _incomplete(base_model, f"unexpected status {tags_response.status} fetching tags page")
    try:
        html_text = tags_response.body.decode("utf-8")
    except Exception as exc:
        return _incomplete(base_model, f"cannot decode tags page: {exc}")

    all_tags = parse_library_tags(html_text, base_model.ollama_base)
    if not all_tags:
        return _incomplete(base_model, "no tags parsed")
    kept_tags = _keep_relevant_tags(all_tags, base_model.ollama_tag)

    manifests: dict[str, dict] = {}
    digests: dict[str, str] = {}
    for tag in kept_tags:
        try:
            manifest, digest = _fetch_manifest(transport, base_model.ollama_base, tag)
        except Exception as exc:
            return _incomplete(base_model, str(exc))
        manifests[tag] = manifest
        digests[tag] = digest

    tag_quants = {tag: parse_ollama_tag_quant(tag) for tag in kept_tags}
    tag_digests = {tag: _weights_digest(manifests[tag]) for tag in kept_tags}
    resolved_quants = inherit_from_siblings(tag_digests, tag_quants)

    packages = [
        _build_package(base_model, tag, manifests[tag], digests[tag], resolved_quants[tag], run_at)
        for tag in kept_tags
    ]
    return AreaOutcome(
        source="ollama", base_model_hf_repo=base_model.hf_repo, packager=None, status="complete", error=None, packages=packages
    )


def _fetch_manifest(transport: Transport, base: str, tag: str) -> tuple[dict, str]:
    url = MANIFEST_URL.format(base=base, tag=tag)
    get_response = transport("GET", url)
    if get_response.status != 200:
        raise RuntimeError(f"{base}:{tag}: unexpected status {get_response.status} fetching manifest")
    manifest = get_response.json()
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{base}:{tag}: manifest response is not an object")

    head_response = transport("HEAD", url)
    header_digest = head_response.header("ollama-content-digest")
    digest = f"sha256:{header_digest}" if header_digest else f"sha256:{hashlib.sha256(get_response.body).hexdigest()}"
    return manifest, digest


def _weights_digest(manifest: dict) -> str | None:
    """The digest used to detect digest-sibling tags: the model layer's, else the first tensor layer's."""
    layers = manifest.get("layers") or []
    for layer in layers:
        if layer.get("mediaType") == _MODEL_MEDIA_TYPE:
            return layer.get("digest")
    for layer in layers:
        if layer.get("mediaType") == _TENSOR_MEDIA_TYPE:
            return layer.get("digest")
    return None


def _package_files(tag: str, manifest: dict) -> tuple[Literal["gguf", "tensor", "unknown"], list[PackageFile]]:
    layers = manifest.get("layers") or []
    model_layers = [layer for layer in layers if layer.get("mediaType") == _MODEL_MEDIA_TYPE]
    tensor_layers = [layer for layer in layers if layer.get("mediaType") == _TENSOR_MEDIA_TYPE]

    if model_layers:
        files = [
            PackageFile(name=f"{tag}.gguf", role="weights", size_bytes=layer.get("size") or 0, digest=layer.get("digest"))
            for layer in model_layers
        ]
        return "gguf", files
    if tensor_layers:
        # A tensor image can carry hundreds of per-tensor layers; it is always unresolved by
        # format regardless of quantization or file layout, so they are aggregated into one
        # synthetic file rather than modelled one-for-one (documented in CONTRACTS.md).
        total_size = sum(layer.get("size") or 0 for layer in tensor_layers)
        return "tensor", [PackageFile(name=f"{tag}.safetensors", role="weights", size_bytes=total_size, digest=None)]
    return "unknown", []


def _finalize(stub: Package, provenance: str, unresolved_reason: str | None) -> Package:
    data = stub.model_dump(mode="python")
    data["provenance"] = provenance
    data["unresolved_reason"] = unresolved_reason
    return Package(**data)


def _build_package(
    base_model: BaseModelSpec, tag: str, manifest: dict, digest: str, quantization: str | None, run_at: datetime
) -> Package:
    format_, files = _package_files(tag, manifest)
    stub = Package(
        source="ollama",
        ollama_name=f"{base_model.ollama_base}:{tag}",
        manifest_digest=digest,
        base_model_hf_repo=base_model.hf_repo,
        format=format_,
        files=files,
        complete=shards_complete(files),
        quantization=quantization,
        default_context=None,
        provenance="unresolved",
        unresolved_reason="format",
        approval=None,
        observed_at=run_at,
        last_seen=run_at,
        active=True,
    )
    provenance, unresolved_reason = decide_provenance(stub, base_model, None, [])
    return _finalize(stub, provenance, unresolved_reason)
