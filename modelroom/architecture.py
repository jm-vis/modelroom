"""A base model's architecture read from `config.json`, and the memory it keeps per request.

`architecture_from_hf_config` reads the publisher repo's `config.json` into an `Architecture`
(`contracts.py`). Two sizes follow from it (AP9-UX2 P1, ADR 0010):

- `kv_token_bytes`: the 16-bit KV cache of one token -- key and value, per layer of full
  attention, KV head and head width. A dense model counts every layer, a hybrid model only its
  `full_attention` layers; `None` for an architecture nobody could read.
- `state_bytes`: what a hybrid model keeps per request whatever the context -- the state of each
  `linear_attention` layer (the recurrent matrix plus the convolution window), at the width of
  `mamba_ssm_dtype`. `0` for a dense model.

This module imports from `contracts`, never the other way round.
"""

from __future__ import annotations

from typing import Literal

from .contracts import HYBRID_LAYER_TYPES, Architecture

KV_BYTES_PER_ELEMENT = 2

# Any of these fields, present at the top level or under `text_config` with a value other
# than None, marks a mixture-of-experts config and rules out `dense_classic` and `hybrid_linear`
# regardless of the plain numeric fields. A dense decoder does not carry them at all, so presence
# is the signal; the value is not interpreted (a routed expert count of 1 is still a MoE layout).
_MOE_FIELDS = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
)

# Bytes of one state element by `mamba_ssm_dtype`; a config without the field is read as 4 bytes
# (float32), a documented assumption (ADR 0010). An unlisted dtype is not read at all.
_STATE_DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}
STATE_BYTES_ASSUMED = 4

_LINEAR_CONFIG_FIELDS = (
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)


def _is_moe_config(config: dict, layer_config: dict) -> bool:
    return any(source.get(field) is not None for source in (config, layer_config) for field in _MOE_FIELDS)


def architecture_from_hf_config(
    source_repo: str, source_revision: str | None, config: dict | None
) -> Architecture:
    """Build an `Architecture` from a publisher repo's parsed `config.json`.

    Some publishers nest the language-model fields under a `text_config` key (multimodal
    configs); this reads from there when present, otherwise from the top level. `kind` is
    `dense_classic` when the three numeric fields are all present, `layer_types` is either absent
    or entirely `"full_attention"`, and none of the MoE fields (checked at both the top level and
    in `text_config`) signal a mixture-of-experts model. It is `hybrid_linear` under exactly the
    invariants `Architecture` checks for it: `layer_types` of `num_hidden_layers` entries, only
    `full_attention` and `linear_attention` and both present, every `linear_*` field, and no MoE
    field. Anything else, or a missing file, resolves to `"unknown"` with the numeric fields left
    at `None`; `max_context` is kept either way.
    """
    if config is None:
        return Architecture(source_repo=source_repo, source_revision=source_revision, kind="unknown")

    layer_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    numeric = {name: layer_config.get(name) for name in ("num_hidden_layers", "num_key_value_heads", "head_dim")}
    layer_types = layer_config.get("layer_types")
    max_context = layer_config.get("max_position_embeddings") or config.get("max_position_embeddings")
    # AP9-SV2-P: read independently of `kind`, same as `max_context` -- a tensor-parallel package
    # may need the head count even when the rest of the shape could not be read (`kind == "unknown"`).
    num_attention_heads = layer_config.get("num_attention_heads")
    kind = _kind(numeric, layer_types, layer_config, _is_moe_config(config, layer_config))
    fields: dict = {}
    if kind != "unknown":
        fields = {**numeric, "layer_types": layer_types}
    if kind == "hybrid_linear":
        fields.update({name: layer_config.get(name) for name in _LINEAR_CONFIG_FIELDS})
        fields["linear_state_bytes"] = _state_dtype_bytes(layer_config)
    return Architecture(
        source_repo=source_repo,
        source_revision=source_revision,
        kind=kind,
        max_context=max_context,
        num_attention_heads=num_attention_heads,
        **fields,
    )


def _kind(
    numeric: dict, layer_types: object, layer_config: dict, is_moe: bool
) -> Literal["dense_classic", "hybrid_linear", "unknown"]:
    """Which of the three kinds a config reads as (`architecture_from_hf_config`)."""
    if is_moe or None in numeric.values():
        return "unknown"
    if layer_types is None or all(t == "full_attention" for t in layer_types):
        return "dense_classic"
    hybrid = (
        isinstance(layer_types, list)
        and len(layer_types) == numeric["num_hidden_layers"]
        and set(layer_types) == set(HYBRID_LAYER_TYPES)
        and all(layer_config.get(name) is not None for name in _LINEAR_CONFIG_FIELDS)
        and _state_dtype_bytes(layer_config) is not None
    )
    return "hybrid_linear" if hybrid else "unknown"


def _state_dtype_bytes(layer_config: dict) -> int | None:
    """Bytes of one state element: from `mamba_ssm_dtype`, 4 when absent, `None` when unreadable."""
    dtype = layer_config.get("mamba_ssm_dtype")
    if dtype is None:
        return STATE_BYTES_ASSUMED
    return _STATE_DTYPE_BYTES.get(dtype) if isinstance(dtype, str) else None  # a list or a map reads as nothing


def _full_attention_layers(arch: Architecture) -> int:
    if arch.kind == "hybrid_linear":
        return sum(1 for layer in arch.layer_types if layer == "full_attention")
    return arch.num_hidden_layers


def kv_token_bytes(arch: Architecture) -> int | None:
    """The 16-bit KV cache of one token in bytes: `2 x n_attn x kv_heads x head_dim x 2`.

    `n_attn` is every layer of a `dense_classic` model and the `full_attention` layers of a
    `hybrid_linear` one; `None` for `unknown`.
    """
    if arch.kind == "unknown":
        return None
    return 2 * _full_attention_layers(arch) * arch.num_key_value_heads * arch.head_dim * KV_BYTES_PER_ELEMENT


def state_bytes(arch: Architecture) -> int:
    """The per-request state of the `linear_attention` layers in bytes; `0` unless `hybrid_linear`.

    Per linear layer: the recurrent state `value_heads x key_dim x value_dim` plus the convolution
    window `(kernel - 1) x (2 x key_heads x key_dim + value_heads x value_dim)`, each element
    `linear_state_bytes` wide.
    """
    if arch.kind != "hybrid_linear":
        return 0
    n_linear = sum(1 for layer in arch.layer_types if layer == "linear_attention")
    recurrent = arch.linear_num_value_heads * arch.linear_key_head_dim * arch.linear_value_head_dim
    conv_width = 2 * arch.linear_num_key_heads * arch.linear_key_head_dim + arch.linear_num_value_heads * arch.linear_value_head_dim
    window = (arch.linear_conv_kernel_dim - 1) * conv_width
    return n_linear * (recurrent + window) * arch.linear_state_bytes


__all__ = [
    "KV_BYTES_PER_ELEMENT",
    "STATE_BYTES_ASSUMED",
    "architecture_from_hf_config",
    "kv_token_bytes",
    "state_bytes",
]
