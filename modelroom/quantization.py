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

# Ascending: smallest/most-compressed first, largest/least-compressed last. Extended 2026-09-22
# against a live snapshot of the allow-listed packagers (13 base models, 79 areas, 358
# packages; see docs/ and CONTRACTS.md) -- the ternary (`TQ*`) and 1-bit `Q1_0` builds sort
# below `IQ1_S`, the rest follow the documented `<` chains between existing families.
QUANT_ORDER: tuple[str, ...] = (
    "TQ1_0",
    "UD-TQ1_0",
    "UD-Q1_0",
    "TQ2_0",
    "UD-TQ2_0",
    "IQ1_S",
    "IQ1_M",
    "UD-IQ1_S",
    "UD-IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "UD-IQ2_XXS",
    "UD-IQ2_M",
    "Q2_K",
    "Q2_K_L",
    "UD-Q2_K_XL",
    "IQ3_XXS",
    "IQ3_XS",
    "UD-IQ3_XXS",
    "IQ3_S",
    "UD-IQ3_S",
    "IQ3_M",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q3_K_XL",
    "UD-Q3_K_XL",
    "IQ4_XS",
    "UD-IQ4_XS",
    "IQ4_NL",
    "UD-IQ4_NL",
    "Q4_0",
    "Q4_1",
    "Q4_K_S",
    "Q4_K_M",
    "UD-Q4_K_XL",
    "MXFP4",
    "MXFP4_MOE",
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

# Markers that identify a basename as not a model-weight file, and the `PackageFile.role` each
# implies: a projector companion (`mmproj`) is its own role, an importance-matrix helper
# (`imatrix`) is `other`. Checked as "contains", not "starts with" -- e.g.
# `Qwen3-VL-235B-A22B-Instruct.mmproj-Q8_0.gguf` and `MiniMax-M3-imatrix.gguf` both carry the
# marker mid-name, never at the start. The single source both `hf._classify_role` and
# `parse_hf_quant` read, so a package can never end up a "weight" under one and an "extra"
# under the other.
_NON_WEIGHT_ROLES: dict[str, str] = {"mmproj": "mmproj", "imatrix": "other"}

_SHARD_SUFFIX_RE = re.compile(r"-(?:split-)?\d{5}-of-\d{5}$")
_STEM_EXTENSIONS = (".gguf", ".safetensors")
# An Ollama tag starts with a size token such as `9b`, `1.5b` or `270m`; anything else is not
# a library size tag and carries no quantization of its own. Named groups so `provenance.py`
# can read the declared number/unit back out for its within-tolerance size check.
SIZE_TOKEN_RE = re.compile(r"(?i)(?P<number>\d+(?:\.\d+)?)(?P<unit>[bm])")


def non_weight_role(basename: str) -> str | None:
    """The role forced by a non-weight marker in `basename`, or `None` for a real weight file."""
    lower = basename.lower()
    for marker, role in _NON_WEIGHT_ROLES.items():
        if marker in lower:
            return role
    return None


def has_shard_suffix(stem: str) -> bool:
    """Whether `stem` (extension already stripped) ends in a shard suffix.

    Shared by `normalize_file_stem` (strips it) and `hf._classify_role` (only checks for it),
    so the shard pattern -- including Qwen's own optional `-split-` marker before the
    `-NNNNN-of-NNNNN` counter -- is defined exactly once.
    """
    return _SHARD_SUFFIX_RE.search(stem) is not None


def canonical_quant(token: str) -> str | None:
    """Map `token` to its canonical `QUANT_ORDER` member, case-insensitively.

    mradermacher writes the lowercase `f16`/`bf16`; this returns the canonical uppercase
    member (`F16`, `BF16`) so `Package.quantization` always stores the same spelling regardless
    of which packager's casing produced it. `None` when `token` matches no member at all.
    """
    for quant in QUANT_ORDER:
        if quant.lower() == token.lower():
            return quant
    return None


def match_base_prefix(basename: str, base_name: str) -> str | None:
    """Match the packager naming convention `<base_name><sep><QUANT>` against `basename`.

    `sep` is `-` (most packagers, e.g. `GLM-5.2-BF16`) or `.` (mradermacher's
    `<base>.<QUANT>.gguf` convention, e.g. `Qwen3.5-9B.IQ4_XS`). The quantization token is
    matched case-insensitively; the return value is the canonical `QUANT_ORDER` member, or
    `None` when `basename` follows neither convention for this `base_name`.
    """
    for separator in ("-", "."):
        prefix = f"{base_name}{separator}"
        if basename.startswith(prefix):
            canonical = canonical_quant(basename[len(prefix) :])
            if canonical is not None:
                return canonical
    return None


def stem_basename(filename: str) -> str:
    """The basename (last `/` or `\\` segment) of the normalized, shard-stripped file stem.

    Used wherever a naming convention must match against the file's own name, never the
    packager's folder layout (e.g. unsloth's per-quant subfolders such as
    `BF16/GLM-5.2-BF16-00001-of-00033.gguf`).
    """
    return re.split(r"[\\/]", normalize_file_stem(filename))[-1]


def identity_stem(filename: str) -> str:
    """The stem used for package identity: the shard suffix is removed, the extension kept.

    `Nova-7B-BF16.gguf` and `Nova-7B-BF16.safetensors` are two different builds of the same
    repo and must never collapse into one identity; the shards of one build
    (`name-00001-of-00002.gguf`, `name-00002-of-00002.gguf`) must.
    """
    lower = filename.lower()
    for extension in _STEM_EXTENSIONS:
        if lower.endswith(extension):
            return normalize_file_stem(filename) + extension
    return normalize_file_stem(filename)


def normalize_file_stem(filename: str) -> str:
    """Strip a trailing `.gguf`/`.safetensors` extension, then a `-NNNNN-of-NNNNN` shard suffix.

    Order matters: the extension is stripped first so the shard suffix regex, which is
    anchored at the end of the string, matches regardless of which of the two extensions (or
    neither) the file carries -- e.g. both `name-00001-of-00002.gguf` and
    `model.safetensors-00001-of-00002.safetensors` collapse to their un-sharded stem.
    """
    stem = filename
    lower = stem.lower()
    for extension in _STEM_EXTENSIONS:
        if lower.endswith(extension):
            stem = stem[: -len(extension)]
            break
    stem = _SHARD_SUFFIX_RE.sub("", stem)
    return stem


def parse_hf_quant(filename: str) -> str | None:
    """Read the quantization off a Hugging Face GGUF filename.

    The quantization is the suffix after the normalized stem's last `-` or `.` separator
    (mradermacher writes `<base>.<QUANT>.gguf`, most other packagers `<base>-<QUANT>.gguf`),
    matched case-insensitively against the longest member of `QUANT_ORDER` that is a suffix
    (so `UD-Q4_K_XL`, which itself contains a `-`, wins over a shorter false match, and
    lowercase `f16` still resolves to canonical `F16`). A non-weight file (a projector or an
    importance matrix, see `non_weight_role`) never carries a quantization even though its
    suffix may look like one; that check looks at the basename only (the last `/` or `\\`
    segment), so a path like `projectors/mmproj-F16.gguf` is still recognized as a projector.
    """
    stem = normalize_file_stem(filename)
    basename = re.split(r"[\\/]", stem)[-1]
    if non_weight_role(basename) is not None:
        return None
    lower_stem = stem.lower()
    candidates = [
        quant
        for quant in QUANT_ORDER
        if lower_stem.endswith(f"-{quant.lower()}") or lower_stem.endswith(f".{quant.lower()}")
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def parse_ollama_tag_quant(tag: str) -> str | None:
    """Read the quantization off an Ollama tag, e.g. `9b-q4_K_M` -> `Q4_K_M`.

    The tag is split on `-`; the quantization is the LAST token when it matches a
    `QUANT_ORDER` member case-insensitively (so `70b-instruct-q4_K_M` -> `Q4_K_M`, the
    descriptive middle tokens are ignored). Any middle token equal to `mlx` (case-insensitive)
    marks a tensor build and forces `None` regardless of the last token (`9b-mlx-bf16` stays
    `None`). A tag with fewer than two tokens (e.g. `9b` alone), or whose last token is not a
    `QUANT_ORDER` member, returns `None`; callers fall back to `inherit_from_siblings` or leave
    it unknown.
    """
    tokens = tag.split("-")
    if len(tokens) < 2 or not SIZE_TOKEN_RE.fullmatch(tokens[0]):
        return None
    if any(token.lower() == "mlx" for token in tokens):
        return None
    return canonical_quant(tokens[-1])


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

    For Hugging Face, the third element is the sorted, de-duplicated set of the identity stems
    (shard suffix removed, extension kept) of every weights/weights-shard file, joined with
    `|`: the shards of one sharded package collapse to the same identity, two file formats of
    the same build stay distinct, and the result never depends on the order `files` happens
    to list them in. For Ollama, it is the tag part of `ollama_name` after the colon.
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
    stems = {identity_stem(file.name) for file in package.files if file.role in ("weights", "weights_shard")}
    return "|".join(sorted(stems))


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
