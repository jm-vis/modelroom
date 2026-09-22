"""Decide whether a package's link to its base model can be trusted.

A package is trustworthy either because a human approved its exact content (an approval
survives only as long as the content it names is still current) or because its own metadata
proves the link on its own: a packager naming convention plus a `base_model:` tag on Hugging
Face, or a name/tag convention in the Ollama library. Anything else is left `unresolved`,
never guessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .quantization import QUANT_ORDER, normalize_file_stem

if TYPE_CHECKING:
    from .contracts import Approval, BaseModelSpec, Package


def decide_provenance(
    package: "Package",
    base_model: "BaseModelSpec",
    hf_tags: list[str] | None,
    approvals: list["Approval"],
) -> tuple[str, str | None]:
    """Return `(provenance, unresolved_reason)` for `package` against `base_model`.

    Order of decision, matching `CONTRACTS.md`:
    1. A non-GGUF package (`tensor`/`unknown` format) is always `("unresolved", "format")`.
    2. An approval whose `content` equals the package's *current* revision or manifest digest
       makes it `("approved", None)`. An approval bound to older content is void: it never
       falls back to `metadata_ok` by itself, the metadata rules below decide instead.
    3. Otherwise the metadata rules for the package's source decide.
    """
    if package.format != "gguf":
        return "unresolved", "format"

    current_content = package.revision if package.source == "huggingface" else package.manifest_digest
    for approval in approvals:
        if approval.content == current_content:
            return "approved", None

    if package.source == "huggingface":
        return _decide_huggingface(package, base_model, hf_tags or [])
    return _decide_ollama(package, base_model)


def _decide_huggingface(
    package: "Package", base_model: "BaseModelSpec", hf_tags: list[str]
) -> tuple[str, str | None]:
    base_name = base_model.hf_repo.split("/", 1)[1]
    repo_name = package.repo.split("/", 1)[1] if package.repo and "/" in package.repo else package.repo

    name_ok = repo_name == f"{base_name}-GGUF" or repo_name in base_model.repo_aliases
    if not name_ok:
        return "unresolved", "repo_name"

    if f"base_model:{base_model.hf_repo}" not in hf_tags:
        return "unresolved", "base_model_tag"

    weight_files = [f for f in package.files if f.role in ("weights", "weights_shard")]
    if not weight_files:
        return "unresolved", "no_weights"

    prefix = f"{base_name}-"
    for weight_file in weight_files:
        stem = normalize_file_stem(weight_file.name)
        if not stem.startswith(prefix) or stem[len(prefix) :] not in QUANT_ORDER:
            return "unresolved", "file_stem"

    return "metadata_ok", None


def _decide_ollama(package: "Package", base_model: "BaseModelSpec") -> tuple[str, str | None]:
    """Decide provenance for an Ollama package against the tag conventions in CONTRACTS.md.

    Order: the base name (`<ollama_base>:`) must prefix `ollama_name`; the base model must
    declare an `ollama_tag`; the tag's first `-`-separated token must equal
    `f"{parameters_b:g}b"` exactly (so `nova:7banana` and `nova:70b-q4_K_M` are both rejected
    for a 7B model -- neither has a first token that equals `7b`); and the full tag must equal
    `base_model.ollama_tag` or start with `base_model.ollama_tag + "-"` (token boundary, so
    `7b-q4_K_M` and `7b-instruct-q4_K_M` match a `7b` base tag but a divergent base tag such as
    `7b-preview` does not).
    """
    ollama_name = package.ollama_name or ""

    if not base_model.ollama_base:
        return "unresolved", "ollama_base"
    if not ollama_name.startswith(f"{base_model.ollama_base}:"):
        return "unresolved", "ollama_name"

    tag = ollama_name.split(":", 1)[1] if ":" in ollama_name else ""
    if not base_model.ollama_tag:
        return "unresolved", "ollama_tag"

    first_token = tag.split("-", 1)[0]
    size_token = f"{base_model.parameters_b:g}b"
    if first_token != size_token:
        return "unresolved", "size_token"

    if tag != base_model.ollama_tag and not tag.startswith(f"{base_model.ollama_tag}-"):
        return "unresolved", "ollama_tag"

    return "metadata_ok", None
