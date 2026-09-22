"""Quantization parsing and package ordering.

Pure functions only: parsing a GGUF filename or an Ollama tag into a quantization label,
inheriting a quantization from a digest sibling when a tag carries none of its own, and the
stable sort key that orders packages of the same base model from smallest/most-compressed to
largest/least-compressed.

No dependency on `modelroom.contracts` at runtime: the functions here take plain values
(filenames, tags, digests) or duck-type on the small subset of `Package` attributes they need,
so `contracts.py` can import this module without creating a cycle. Type hints that name
`Package`/`PackageFile` are import-guarded behind `TYPE_CHECKING`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import Package

# Ascending: smallest/most-compressed first, largest/least-compressed last.
QUANT_ORDER: tuple[str, ...] = (
    "IQ2_XXS",
    "IQ2_M",
    "UD-IQ2_XXS",
    "UD-IQ2_M",
    "Q2_K",
    "UD-Q2_K_XL",
    "IQ3_XXS",
    "UD-IQ3_XXS",
    "Q3_K_S",
    "Q3_K_M",
    "UD-Q3_K_XL",
    "IQ4_XS",
    "IQ4_NL",
    "Q4_0",
    "Q4_1",
    "Q4_K_S",
    "Q4_K_M",
    "UD-Q4_K_XL",
    "Q5_K_S",
    "Q5_K_M",
    "UD-Q5_K_XL",
    "Q6_K",
    "UD-Q6_K_XL",
    "Q8_0",
    "UD-Q8_K_XL",
    "F16",
    "BF16",
    "F32",
)

_SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")
_OLLAMA_SIZE_TAG_RE = re.compile(r"^\d+(?:\.\d+)?[bBmMkK]-(.+)$")


def normalize_file_stem(filename: str) -> str:
    """Strip the `.gguf` extension and a `-NNNNN-of-NNNNN` shard suffix, in that order."""
    stem = filename
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    stem = _SHARD_SUFFIX_RE.sub("", stem)
    return stem


def parse_hf_quant(filename: str) -> str | None:
    """Read the quantization off a Hugging Face GGUF filename.

    The quantization is the suffix after the last `-` of the normalized stem, matched
    case-sensitively against the longest member of `QUANT_ORDER` that is a `-<QUANT>` suffix
    (so `UD-Q4_K_XL`, which itself contains a `-`, wins over a shorter false match). An mmproj
    file (its stem starts with `mmproj`, e.g. `mmproj-BF16.gguf`) is a projector, not a model
    weight, and never carries a quantization even though its suffix looks like one.
    """
    stem = normalize_file_stem(filename)
    if stem.lower().startswith("mmproj"):
        return None
    candidates = [quant for quant in QUANT_ORDER if stem.endswith(f"-{quant}")]
    if not candidates:
        return None
    return max(candidates, key=len)


def parse_ollama_tag_quant(tag: str) -> str | None:
    """Read the quantization off an Ollama tag, e.g. `9b-q4_K_M` -> `Q4_K_M`.

    The tag is expected to start with a size token (`9b-`, `70b-`, ...) followed by the
    quantization, matched case-insensitively against `QUANT_ORDER`. A tag with no such suffix,
    or whose suffix is not a `QUANT_ORDER` member (e.g. `9b-mlx-bf16`, a tensor-format build),
    returns `None`; callers fall back to `inherit_from_siblings` or leave it unknown.
    """
    match = _OLLAMA_SIZE_TAG_RE.match(tag)
    if not match:
        return None
    candidate = match.group(1)
    for quant in QUANT_ORDER:
        if quant.lower() == candidate.lower():
            return quant
    return None


def inherit_from_siblings(
    tag_digests: dict[str, str], tag_quants: dict[str, str | None]
) -> dict[str, str | None]:
    """Fill in a tag's missing quantization from another tag that shares its weights digest.

    `tag_digests` maps a tag to its weights-layer digest, `tag_quants` maps a tag to the
    quantization `parse_ollama_tag_quant` already found for it (or `None`). A tag without a
    digest, or whose digest matches no other tag's known quantization, keeps its `None`.
    """
    digest_to_quant: dict[str, str] = {}
    for tag, quant in tag_quants.items():
        digest = tag_digests.get(tag)
        if digest is not None and quant is not None and digest not in digest_to_quant:
            digest_to_quant[digest] = quant

    result = dict(tag_quants)
    for tag, quant in tag_quants.items():
        if quant is not None:
            continue
        digest = tag_digests.get(tag)
        if digest is not None and digest in digest_to_quant:
            result[tag] = digest_to_quant[digest]
    return result


def package_identity_key(package: "Package") -> tuple[str, str, str]:
    """The `(source, repo|ollama_name, filename-or-tag)` triple that identifies a package.

    For Hugging Face, the third element is the normalized stem of its first weights or
    weights-shard file (so the shards of one sharded package collapse to the same identity).
    For Ollama, it is the tag part of `ollama_name` after the colon.
    """
    if package.source == "huggingface":
        repo_or_name = package.repo or ""
    else:
        repo_or_name = package.ollama_name or ""
    return (package.source, repo_or_name, _weights_filename_or_tag(package))


def _weights_filename_or_tag(package: "Package") -> str:
    if package.source == "ollama":
        name = package.ollama_name or ""
        return name.split(":", 1)[1] if ":" in name else name
    for file in package.files:
        if file.role in ("weights", "weights_shard"):
            return normalize_file_stem(file.name)
    return ""


def _packager_name(package: "Package") -> str:
    if package.source == "huggingface":
        repo = package.repo or ""
        return repo.split("/", 1)[0] if "/" in repo else repo
    return "ollama"


def _weights_size_bytes(package: "Package") -> int:
    return sum(file.size_bytes for file in package.files if file.role in ("weights", "weights_shard"))


def sort_key(package: "Package") -> tuple[int, int, str, tuple[str, str, str]]:
    """Order packages of the same base model from smallest/most-compressed to largest.

    Primary key is the `QUANT_ORDER` index (unknown quantization sorts last), then the total
    weights size (smaller first), then the packager name alphabetically (the Hugging Face
    owner, or the literal `"ollama"` for the library), then the full package identity as a
    stable, deterministic final tie-break.
    """
    quant = package.quantization
    quant_index = QUANT_ORDER.index(quant) if quant in QUANT_ORDER else len(QUANT_ORDER)
    return (
        quant_index,
        _weights_size_bytes(package),
        _packager_name(package),
        package_identity_key(package),
    )
