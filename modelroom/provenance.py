"""Decide whether a package's link to its base model can be trusted.

A package is trustworthy either because a human approved its exact content (an approval
survives only as long as the content it names is still current) or because its own metadata
proves the link on its own: a packager naming convention plus a `base_model:` tag on Hugging
Face, or a name/tag convention in the Ollama library. Anything else is left `unresolved`,
never guessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .quantization import SIZE_TOKEN_RE, match_base_prefix, stem_basename

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
    """Decide provenance for a Hugging Face package: repo name, `base_model:` tag, file stem.

    The file-stem check (`quantization.match_base_prefix`) matches each weight file's own
    *basename*, never the packager's folder layout -- unsloth ships per-quant subfolders
    (`BF16/GLM-5.2-BF16-00001-of-00033.gguf`), so the folder segment must never defeat the
    prefix match, nor let an unrelated file (e.g. a draft/MTP model dropped in the same repo)
    pass under a foreign folder name. It accepts both the common `<base>-<QUANT>` convention
    and mradermacher's `<base>.<QUANT>` one.
    """
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

    for weight_file in weight_files:
        basename = stem_basename(weight_file.name)
        if match_base_prefix(basename, base_name) is None:
            return "unresolved", "file_stem"

    return "metadata_ok", None


# A declared size token is accepted when it is within this fraction of the measured
# `parameters_b`: `safetensors.total` counts embeddings, so a 9.653B model is legitimately
# tagged `9b` by its packager -- an exact-equality rule can never match real data (see
# CONTRACTS.md, "Ollama size-token tolerance").
_SIZE_TOKEN_TOLERANCE = 0.15


def _size_token_matches(first_token: str, parameters_b: float) -> bool:
    """Whether `first_token` (the tag's first `-`-separated token) plausibly names `parameters_b`.

    `first_token` must fully match `SIZE_TOKEN_RE` (`<number>b` or `<number>m`, case-insensitive
    -- `7banana` is not a size token at all and never matches). A `b` token is compared directly
    in billions; an `m` token is converted to billions first. The declared size must be within
    `_SIZE_TOKEN_TOLERANCE` (15 %) of `parameters_b`.
    """
    match = SIZE_TOKEN_RE.fullmatch(first_token)
    if match is None:
        return False
    declared = float(match.group("number"))
    if match.group("unit").lower() == "m":
        declared /= 1000
    return abs(parameters_b - declared) <= _SIZE_TOKEN_TOLERANCE * declared


def _decide_ollama(package: "Package", base_model: "BaseModelSpec") -> tuple[str, str | None]:
    """Decide provenance for an Ollama package against the tag conventions in CONTRACTS.md.

    Order: the base name (`<ollama_base>:`) must prefix `ollama_name`; the base model must
    declare an `ollama_tag`; the tag's first `-`-separated token must be a size token
    (`SIZE_TOKEN_RE`) whose declared size is within 15 % of `parameters_b` (`_size_token_matches`
    -- `nova:7banana` is rejected because it is not a size token at all, `nova:70b-q4_K_M` is
    rejected for a 7B model because 70 is nowhere near 7 within tolerance); and the full tag
    must equal `base_model.ollama_tag` or start with `base_model.ollama_tag + "-"` (token
    boundary, so `7b-q4_K_M` and `7b-instruct-q4_K_M` match a `7b` base tag but a divergent base
    tag such as `7b-preview` does not).
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
    if not _size_token_matches(first_token, base_model.parameters_b):
        return "unresolved", "size_token"

    if tag != base_model.ollama_tag and not tag.startswith(f"{base_model.ollama_tag}-"):
        return "unresolved", "ollama_tag"

    return "metadata_ok", None
