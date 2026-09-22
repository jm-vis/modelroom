"""Ollama fetching: tags for a base model's library page, and one package per kept tag.

Only base models that configure `ollama_base`/`ollama_tag` are looked at; everything else is
a trivially "complete, empty" area (CONTRACTS.md/AGENTS.md's Ollama tier only ever names a
base model this project's configuration already lists -- there is no discovery step here).

The tags page is plain, unauthenticated web data extraction: one `GET` of one page per
configured base model, the tag names read out of `<a href="/library/<base>:<tag>">` anchors
with the standard-library `html.parser`, nothing else taken from the page, no login, no
crawling beyond it (see the `datenextraktion` skill this project follows for scraping work).
The manifest digest is `sha256:` plus the sha256 of the manifest `GET` body -- **no `HEAD`
request is made at all** (F5): measured 2026-09-22 against the real registry
(`registry.ollama.ai/v2/library/qwen3.5/manifests/9b`, 709 bytes), hashing the `GET` body gives
exactly the same digest the registry's `ollama-content-digest` `HEAD` header used to state (see
tests/fixtures/README.md), so the extra request bought nothing and is dropped.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from html.parser import HTMLParser
from typing import Literal

from .contracts import Approval, BaseModelSpec, Package, PackageFile, shards_complete
from .fetch_types import AreaOutcome
from .http import Transport
from .provenance import decide_provenance
from .quantization import inherit_from_siblings, package_identity_key, parse_ollama_tag_quant

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


def fetch_ollama_area(
    transport: Transport,
    base_model: BaseModelSpec,
    run_at: datetime,
    previous_by_key: dict[tuple[str, str, str], Package] | None = None,
) -> AreaOutcome:
    """Fetch every kept tag of `base_model`'s Ollama library entry and assemble their packages.

    `previous_by_key` (F6, default `None`) maps a package identity key to that package as it
    stood in the previous snapshot -- see `fetch_hf_area`'s docstring for the exact rule; the
    same carry-forward applies here via `_build_package`.
    """
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

    try:
        packages = _assemble_packages(
            base_model, kept_tags, manifests, digests, run_at, previous_by_key or {}
        )
    except Exception as exc:
        # F3: a malformed manifest (a layer in an unexpected shape, a weight layer with no
        # size, ...) ends this area incomplete like any other genuine failure, never raises.
        return _incomplete(base_model, f"{base_model.ollama_base}: {exc}")

    return AreaOutcome(
        source="ollama", base_model_hf_repo=base_model.hf_repo, packager=None, status="complete", error=None, packages=packages
    )


def _assemble_packages(
    base_model: BaseModelSpec,
    kept_tags: list[str],
    manifests: dict[str, dict],
    digests: dict[str, str],
    run_at: datetime,
    previous_by_key: dict[tuple[str, str, str], Package],
) -> list[Package]:
    tag_layers = {tag: _validated_layers(base_model.ollama_base, tag, manifests[tag]) for tag in kept_tags}
    tag_quants = {tag: parse_ollama_tag_quant(tag) for tag in kept_tags}
    tag_digests = {tag: _weights_digest(tag_layers[tag]) for tag in kept_tags}
    resolved_quants = inherit_from_siblings(tag_digests, tag_quants)

    return [
        _build_package(base_model, tag, tag_layers[tag], digests[tag], resolved_quants[tag], run_at, previous_by_key)
        for tag in kept_tags
    ]


def _fetch_manifest(transport: Transport, base: str, tag: str) -> tuple[dict, str]:
    """`GET` the manifest and return it with its digest: `sha256:` + sha256 of the raw body.

    F5: no `HEAD` request is made at all any more -- see the module docstring for the
    measurement showing the body hash equals the registry's old `HEAD` header exactly.
    """
    url = MANIFEST_URL.format(base=base, tag=tag)
    get_response = transport("GET", url)
    if get_response.status != 200:
        raise RuntimeError(f"{base}:{tag}: unexpected status {get_response.status} fetching manifest")
    manifest = get_response.json()
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{base}:{tag}: manifest response is not an object")
    digest = f"sha256:{hashlib.sha256(get_response.body).hexdigest()}"
    return manifest, digest


def _validated_layers(base: str, tag: str, manifest: dict) -> list[dict]:
    """`manifest['layers']`, validated: F3/F4 -- every layer must be an object with `mediaType`/
    `digest` as strings when present; a weight (`.image.model`/`.image.tensor`) layer's own
    `size` is validated later, per layer, once its role is known (`_weight_layer_size`).
    """
    layers = manifest.get("layers")
    if layers is None:
        return []
    if not isinstance(layers, list):
        raise ValueError(f"{base}:{tag}: manifest 'layers' is not a list")
    return [_validated_layer(base, tag, layer) for layer in layers]


def _validated_layer(base: str, tag: str, layer: object) -> dict:
    if not isinstance(layer, dict):
        raise ValueError(f"{base}:{tag}: manifest layer is not an object: {layer!r}")
    media_type = layer.get("mediaType")
    if media_type is not None and not isinstance(media_type, str):
        raise ValueError(f"{base}:{tag}: manifest layer mediaType is not a string: {media_type!r}")
    digest = layer.get("digest")
    if digest is not None and not isinstance(digest, str):
        raise ValueError(f"{base}:{tag}: manifest layer digest is not a string: {digest!r}")
    return layer


def _weight_layer_size(layer: dict) -> int:
    """F4: a weight layer (`.image.model`/`.image.tensor`) without a real size is a shape error."""
    size = layer.get("size")
    if size is None:
        raise ValueError(f"weight layer has no 'size': {layer!r}")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"weight layer 'size' is not a non-negative integer: {size!r}")
    return size


def _weights_digest(layers: list[dict]) -> str | None:
    """The digest used to detect digest-sibling tags: the model layer's, else the first tensor layer's."""
    for layer in layers:
        if layer.get("mediaType") == _MODEL_MEDIA_TYPE:
            return layer.get("digest")
    for layer in layers:
        if layer.get("mediaType") == _TENSOR_MEDIA_TYPE:
            return layer.get("digest")
    return None


def _package_files(tag: str, layers: list[dict]) -> tuple[Literal["gguf", "tensor", "unknown"], list[PackageFile]]:
    model_layers = [layer for layer in layers if layer.get("mediaType") == _MODEL_MEDIA_TYPE]
    tensor_layers = [layer for layer in layers if layer.get("mediaType") == _TENSOR_MEDIA_TYPE]

    if model_layers:
        files = [
            PackageFile(name=f"{tag}.gguf", role="weights", size_bytes=_weight_layer_size(layer), digest=layer.get("digest"))
            for layer in model_layers
        ]
        return "gguf", files
    if tensor_layers:
        # A tensor image can carry hundreds of per-tensor layers; it is always unresolved by
        # format regardless of quantization or file layout, so they are aggregated into one
        # synthetic file rather than modelled one-for-one (documented in CONTRACTS.md).
        total_size = sum(_weight_layer_size(layer) for layer in tensor_layers)
        return "tensor", [PackageFile(name=f"{tag}.safetensors", role="weights", size_bytes=total_size, digest=None)]
    return "unknown", []


def _finalize(stub: Package, provenance: str, unresolved_reason: str | None) -> Package:
    data = stub.model_dump(mode="python")
    data["provenance"] = provenance
    data["unresolved_reason"] = unresolved_reason
    return Package(**data)


def _carry_forward_approval(
    stub: Package, previous_by_key: dict[tuple[str, str, str], Package]
) -> tuple[Package, list[Approval]]:
    """F6: the same rule as `hf._carry_forward_approval`, see its docstring."""
    previous = previous_by_key.get(package_identity_key(stub))
    if previous is None or previous.approval is None:
        return stub, []
    return stub.model_copy(update={"approval": previous.approval}), [previous.approval]


def _build_package(
    base_model: BaseModelSpec,
    tag: str,
    layers: list[dict],
    digest: str,
    quantization: str | None,
    run_at: datetime,
    previous_by_key: dict[tuple[str, str, str], Package],
) -> Package:
    format_, files = _package_files(tag, layers)
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
    stub, approvals = _carry_forward_approval(stub, previous_by_key)
    provenance, unresolved_reason = decide_provenance(stub, base_model, None, approvals)
    return _finalize(stub, provenance, unresolved_reason)
