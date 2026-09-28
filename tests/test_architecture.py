"""Tests for modelroom.architecture: the hybrid architecture of Qwen3.5 and the two sizes it keeps.

Qwen3.5-9B and -4B (AP9-UX2 P1, ADR 0010) have 32 layers, 24 of them `linear_attention` and 8
`full_attention`, 4 KV heads of width 256, and a linear-layer state of 16 key heads, 32 value heads
of width 128 and a convolution window of 4, in `float32`. By hand:

- KV cache per token: `2 x 8 x 4 x 256 x 2` = 32_768 bytes;
- state per request: `24 x (32 x 128 x 128 + 3 x (2 x 16 x 128 + 32 x 128)) x 4` = 52_690_944 bytes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.architecture import architecture_from_hf_config, kv_token_bytes, state_bytes
from modelroom.contracts import LINEAR_FIELDS, Architecture, Snapshot, load_snapshot
from modelroom.examples import EXAMPLES

FIXTURES = Path(__file__).parent / "fixtures"
QWEN35_KV_TOKEN_BYTES = 32_768
QWEN35_STATE_BYTES = 52_690_944


def _qwen35_config() -> dict:
    return json.loads((FIXTURES / "hf_qwen_qwen35_9b_config.json").read_text(encoding="utf-8"))


def _qwen35(config: dict | None = None) -> Architecture:
    model = json.loads((FIXTURES / "hf_qwen_qwen35_9b_model.json").read_text(encoding="utf-8"))
    return architecture_from_hf_config("Qwen/Qwen3.5-9B", model["sha"], config or _qwen35_config())


def _hybrid_fields() -> dict:
    return {
        "source_repo": "Qwen/Qwen3.5-9B",
        "kind": "hybrid_linear",
        "num_hidden_layers": 4,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "linear_state_bytes": 4,
    }


# --- reading config.json ------------------------------------------------------------------------


def test_qwen35_9b_config_is_hybrid_linear():
    architecture = _qwen35()
    assert architecture.kind == "hybrid_linear"
    assert (architecture.num_hidden_layers, architecture.num_key_value_heads, architecture.head_dim) == (32, 4, 256)
    assert architecture.layer_types.count("linear_attention") == 24
    assert architecture.layer_types.count("full_attention") == 8
    assert architecture.linear_num_key_heads == 16
    assert architecture.linear_num_value_heads == 32
    assert architecture.linear_key_head_dim == 128
    assert architecture.linear_value_head_dim == 128
    assert architecture.linear_conv_kernel_dim == 4
    assert architecture.linear_state_bytes == 4
    assert architecture.max_context == 262144


def test_kv_token_bytes_of_qwen35_is_32768():
    assert kv_token_bytes(_qwen35()) == QWEN35_KV_TOKEN_BYTES


def test_state_bytes_of_qwen35_is_52690944():
    assert state_bytes(_qwen35()) == QWEN35_STATE_BYTES


def test_a_bfloat16_state_halves_the_state_bytes():
    config = _qwen35_config()
    config["text_config"]["mamba_ssm_dtype"] = "bfloat16"
    architecture = _qwen35(config)
    assert architecture.linear_state_bytes == 2
    assert state_bytes(architecture) == QWEN35_STATE_BYTES // 2


def test_a_missing_state_dtype_assumes_four_bytes():
    config = _qwen35_config()
    del config["text_config"]["mamba_ssm_dtype"]
    architecture = _qwen35(config)
    assert architecture.linear_state_bytes == 4
    assert state_bytes(architecture) == QWEN35_STATE_BYTES


def test_a_mixture_config_with_linear_layers_stays_unknown():
    config = _qwen35_config()
    config["text_config"]["num_experts"] = 64
    architecture = _qwen35(config)
    assert architecture.kind == "unknown"
    assert architecture.linear_num_key_heads is None and architecture.layer_types is None


def test_a_layer_type_other_than_full_or_linear_stays_unknown():
    config = _qwen35_config()
    config["text_config"]["layer_types"][0] = "sliding_attention"
    assert _qwen35(config).kind == "unknown"


def test_layer_types_of_another_length_stay_unknown():
    config = _qwen35_config()
    config["text_config"]["layer_types"] = config["text_config"]["layer_types"][:-4]
    assert _qwen35(config).kind == "unknown"


def test_only_linear_layers_stay_unknown():
    config = _qwen35_config()
    config["text_config"]["layer_types"] = ["linear_attention"] * 32
    assert _qwen35(config).kind == "unknown"


def test_dense_classic_kv_is_unchanged():
    """Bestandsinvariante (`examples.py:196-200`): 32 layers x 8 KV heads x 128 wide, no state."""
    architecture = Architecture.model_validate(EXAMPLES["Architecture"])
    assert kv_token_bytes(architecture) == 2 * 32 * 8 * 128 * 2
    assert state_bytes(architecture) == 0


def test_an_unknown_architecture_has_no_kv_token_bytes():
    architecture = Architecture(source_repo="acme/Nova-7B", kind="unknown")
    assert kv_token_bytes(architecture) is None
    assert state_bytes(architecture) == 0


# --- the contract ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["num_hidden_layers", "num_key_value_heads", "head_dim", "layer_types", *LINEAR_FIELDS]
)
def test_hybrid_linear_needs_every_field(field):
    fields = _hybrid_fields()
    del fields[field]
    with pytest.raises(ValidationError):
        Architecture(**fields)


@pytest.mark.parametrize(
    "layer_types",
    [
        ["linear_attention", "linear_attention", "full_attention"],
        ["linear_attention"] * 4,
        ["full_attention"] * 4,
        ["linear_attention", "linear_attention", "sliding_attention", "full_attention"],
    ],
)
def test_hybrid_linear_needs_layer_types_of_both_kinds_and_every_layer(layer_types):
    with pytest.raises(ValidationError):
        Architecture(**{**_hybrid_fields(), "layer_types": layer_types})


def test_a_complete_hybrid_architecture_is_accepted():
    assert Architecture(**_hybrid_fields()).kind == "hybrid_linear"


@pytest.mark.parametrize("kind", ["dense_classic", "unknown"])
@pytest.mark.parametrize("field", LINEAR_FIELDS)
def test_dense_classic_and_unknown_with_a_linear_field_are_refused(kind, field):
    fields = {"source_repo": "acme/Nova-7B", "kind": kind, field: 4}
    if kind == "dense_classic":
        fields.update(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)
    with pytest.raises(ValidationError):
        Architecture(**fields)


def _snapshot_with(architecture: dict) -> dict:
    payload = copy.deepcopy(EXAMPLES["Snapshot"])
    payload["base_models"][0]["architecture"] = architecture
    return payload


def test_a_snapshot_with_an_invalid_hybrid_architecture_is_refused():
    fields = _hybrid_fields()
    fields["source_repo"] = "acme/Nova-7B"
    del fields["linear_conv_kernel_dim"]
    with pytest.raises(ValidationError):
        load_snapshot(_snapshot_with(fields))


def test_an_old_snapshot_without_the_new_fields_still_reads():
    payload = copy.deepcopy(EXAMPLES["Snapshot"])
    for name in LINEAR_FIELDS:
        assert name not in payload["base_models"][0]["architecture"]
    snapshot = load_snapshot(payload)
    architecture = snapshot.base_models[0].architecture
    assert architecture.kind == "dense_classic"
    assert all(getattr(architecture, name) is None for name in LINEAR_FIELDS)


def test_a_hybrid_snapshot_round_trips():
    fields = {**_hybrid_fields(), "source_repo": "acme/Nova-7B"}
    snapshot = load_snapshot(_snapshot_with(fields))
    written = json.loads(json.dumps(snapshot.model_dump(mode="json")))
    again = load_snapshot(written)
    assert again == snapshot
    assert isinstance(again, Snapshot)
    assert again.base_models[0].architecture.kind == "hybrid_linear"
