"""A Hugging Face tensor repository's quantization, read from its `config.json` (AP9-SV2-P, P2).

Separate from `quantization.py`: that module parses a GGUF filename or an Ollama tag, this one
parses a vLLM tensor repository's `config.json`, a wholly different source and shape. Both read
and write `quantization.QUANT_ORDER` labels (`docs/adr/0014-runtime-entries-and-vllm-packages.md`),
never anything of their own.

Also here: `SAFE_REPO_PATH_RE`, the one path-safety rule a tensor group's files are checked
against (weights and start files alike, `hf.py` and later the on-disk check and the remote
operation of other SV2 sub-packages), and `start_files`, the one definition of what vLLM loads
at startup -- the same list the disk-space check and the "already on disk" check use.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    from .contracts import PackageFile

# P2: every path of a tensor group's files (weight and start files alike) must match this shape --
# checked by `hf.py` while a repository's tree is assembled into packages, and read again by
# `hf_cache.py` (I3) and the remote operation (R2) against the very same files once they are on
# disk. Each segment starts with a letter or digit (which already rules out a bare `..` segment,
# since `..` starts with `.`), at most four segments (`{0,3}` further ones), never a leading `/`,
# a `\`, a `:` or a UNC prefix -- none of those characters is in the allowed class at all, so a
# Windows drive letter, a UNC share or a backslash-separated path never matches, whatever position
# it appears in.
SAFE_REPO_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*){0,3}$")

# P2: a start file beyond the weights is one of these suffixes -- the files vLLM reads at startup
# besides the weights themselves (README, images and `.gitattributes` are not, Codex r1-6).
_START_FILE_SUFFIXES = (".json", ".tiktoken", ".jinja")
_START_FILE_NAMES = ("tokenizer.model",)

# The upper bound on a `config.json` read for a quantization label (P2): a real transformers
# config is a few KiB; a response this large is refused outright rather than parsed, the same
# defensive bound `hf.py` already applies to other registry-controlled sizes.
MAX_CONFIG_JSON_BYTES = 64 * 1024

QuantizationSource = Literal["config"]


def is_safe_repo_path(path: str) -> bool:
    """Whether `path` (a tree entry's full path) is safe to use as a cache-relative file path.

    Used for both the weight files of a tensor group and its start files (`start_files`) --
    `hf.py` checks every one of them against this rule while assembling a tensor package, the
    same rule `hf_cache.py` (I3) and the remote operation (R2) check again once the files are on
    disk, so a manipulated name or a symlink target outside the cache can never be read as safe
    just because one of the three places forgot to check.
    """
    return bool(SAFE_REPO_PATH_RE.fullmatch(path))


def start_files(files: Sequence["PackageFile"]) -> list["PackageFile"]:
    """The files vLLM reads to start serving a tensor package: weights plus the small extras.

    `files` is a package's whole file list (weight shards and the non-weight extras already
    attached to it, `hf.py::_assemble_packages`'s `[*group_files, *extra_files]`). The weight
    shards always count; an extra file counts only when its name ends in `.json`,
    `tokenizer.model`, `.tiktoken` or `.jinja` -- this is also "download" (I2) and the one list
    "already on disk" (I3) is checked against (A5): README, `.gitattributes` and images are
    never among them (Codex r1-6).
    """
    return [file for file in files if file.role in ("weights", "weights_shard") or _is_start_extra(file.name)]


def _is_start_extra(name: str) -> bool:
    basename = re.split(r"[\\/]", name)[-1].lower()
    if basename in _START_FILE_NAMES:
        return True
    return basename.endswith(_START_FILE_SUFFIXES)


def _layer_config(config: dict) -> dict:
    """The language-model fields: `text_config` when it is an object, else the top level."""
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _compressed_tensors_label(quantization_config: dict) -> str | None:
    """`W4A16`, `W8A8` or `FP8` from a `compressed-tensors` config's first weight group.

    Reads `config_groups.*.weights.num_bits`/`.type` -- every group of a repository quantizes
    its weights the same way (the format `vLLM`'s compressed-tensors loader itself assumes), so
    the first one found decides. Any other shape (no groups, no `weights` sub-object, an
    unlisted `(num_bits, type)` pair) is `None`: an unknown method, never guessed.
    """
    groups = quantization_config.get("config_groups")
    if not isinstance(groups, dict):
        return None
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if not isinstance(weights, dict):
            continue
        num_bits, kind = weights.get("num_bits"), weights.get("type")
        if (num_bits, kind) == (4, "int"):
            return "W4A16"
        if (num_bits, kind) == (8, "int"):
            return "W8A8"
        if (num_bits, kind) == (8, "float"):
            return "FP8"
        return None
    return None


def _quantization_config_label(quantization_config: dict) -> str | None:
    """The `QUANT_ORDER` label from a `quantization_config` object, or `None` for an unknown method."""
    method = quantization_config.get("quant_method")
    if method == "awq" and quantization_config.get("bits") == 4:
        return "AWQ"
    if method == "gptq":
        bits = quantization_config.get("bits")
        if bits == 4:
            return "GPTQ-4"
        if bits == 8:
            return "GPTQ-8"
        return None
    if method == "fp8":
        return "FP8"
    if method == "compressed-tensors":
        return _compressed_tensors_label(quantization_config)
    return None


_DTYPE_LABELS = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}


def _dtype_label(config: dict, layer_config: dict) -> str | None:
    """`BF16`/`F16`/`F32` from `torch_dtype`, else `dtype`, else `text_config.dtype` (SV2-0's
    documented search order -- Qwen3.5 carries `dtype` only under `text_config`)."""
    dtype = config.get("torch_dtype")
    if dtype is None:
        dtype = config.get("dtype")
    if dtype is None and layer_config is not config:
        dtype = layer_config.get("dtype")
    return _DTYPE_LABELS.get(dtype) if isinstance(dtype, str) else None


def tensor_quant_label(config: dict) -> tuple[str | None, QuantizationSource | None]:
    """`(quantization, quantization_source)` for one tensor repository's parsed `config.json`.

    A `quantization_config` object decides first (AWQ, GPTQ, FP8 or one of compressed-tensors'
    three shapes); without one, the model's own dtype decides (`BF16`/`F16`/`F32`). Any other
    shape -- an unknown method, an unlisted bit width, no dtype at all -- is `(None, None)`:
    `quantization_source` is only ever `"config"` when a label was actually produced, never a
    bare "we looked" marker (`Package.quantization_source`, SV2-0).
    """
    layer_config = _layer_config(config)
    quantization_config = config.get("quantization_config")
    label = (
        _quantization_config_label(quantization_config)
        if isinstance(quantization_config, dict)
        else _dtype_label(config, layer_config)
    )
    return (label, "config") if label is not None else (None, None)


__all__ = [
    "MAX_CONFIG_JSON_BYTES",
    "SAFE_REPO_PATH_RE",
    "QuantizationSource",
    "is_safe_repo_path",
    "start_files",
    "tensor_quant_label",
]
