"""Tests for modelroom.tensor_quant: a tensor repository's quantization label, and its file-path
and start-file rules (AP9-SV2-P, P2).

The AWQ, FP8 and GPTQ fixtures are real `config.json` files, fetched once by the builder from the
repositories `CONTRACTS.md`'s "Extern belegt" table documents; see `tests/fixtures/README.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelroom.contracts import PackageFile
from modelroom.tensor_quant import (
    MAX_CONFIG_JSON_BYTES,
    SAFE_REPO_PATH_RE,
    is_safe_repo_path,
    start_files,
    tensor_quant_label,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _config(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --- quantization labels, config.json -> QUANT_ORDER ----------------------------------------------


def test_awq_4_bit_is_awq():
    assert tensor_quant_label(_config("hf_qwen3_8b_awq_config.json")) == ("AWQ", "config")


def test_a_plain_fp8_quant_method_is_fp8():
    assert tensor_quant_label(_config("hf_qwen3_8b_fp8_config.json")) == ("FP8", "config")


def test_compressed_tensors_float_8_is_fp8():
    assert tensor_quant_label(_config("hf_redhatai_qwen3_8b_fp8_dynamic_config.json")) == ("FP8", "config")


def test_compressed_tensors_int_4_is_w4a16():
    assert tensor_quant_label(_config("hf_redhatai_qwen3_8b_w4a16_config.json")) == ("W4A16", "config")


def test_compressed_tensors_int_8_is_w8a8():
    config = _config("hf_redhatai_qwen3_8b_w4a16_config.json")
    config["quantization_config"]["config_groups"]["group_0"]["weights"]["num_bits"] = 8
    assert tensor_quant_label(config) == ("W8A8", "config")


def test_gptq_4_is_gptq_4():
    assert tensor_quant_label(_config("hf_qwen25_7b_gptq_int4_config.json")) == ("GPTQ-4", "config")


def test_gptq_8_is_gptq_8():
    config = _config("hf_qwen25_7b_gptq_int4_config.json")
    config["quantization_config"]["bits"] = 8
    assert tensor_quant_label(config) == ("GPTQ-8", "config")


def test_no_quantization_config_reads_dtype_then_text_config():
    # Top-level torch_dtype:
    awq = _config("hf_qwen3_8b_awq_config.json")
    del awq["quantization_config"]
    assert tensor_quant_label(awq) == ("F16", "config")

    # dtype only under text_config (Qwen3.5's own shape):
    nested = _config("hf_qwen_qwen35_4b_config.json")
    assert "quantization_config" not in nested
    assert nested["text_config"]["dtype"] == "bfloat16"
    assert tensor_quant_label(nested) == ("BF16", "config")


def test_an_unknown_method_is_none_and_the_area_stays_complete():
    config = _config("hf_qwen3_8b_awq_config.json")
    config["quantization_config"]["quant_method"] = "something-new"
    assert tensor_quant_label(config) == (None, None)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["quantization_config"].update(bits=5),  # awq: an unlisted bit width
        lambda c: c["quantization_config"].update(quant_method="gptq", bits=5),
        lambda c: c.pop("quantization_config") and c.update(torch_dtype="float8_e4m3"),  # unlisted dtype
    ],
)
def test_an_unlisted_shape_is_none(mutation):
    config = _config("hf_qwen3_8b_awq_config.json")
    mutation(config)
    assert tensor_quant_label(config) == (None, None)


def test_compressed_tensors_without_config_groups_is_none():
    config = _config("hf_redhatai_qwen3_8b_w4a16_config.json")
    del config["quantization_config"]["config_groups"]
    assert tensor_quant_label(config) == (None, None)


@pytest.mark.parametrize("wrong_shape", [[], "awq", 4, True])
def test_a_wrongly_typed_quantization_config_is_none_never_a_dtype_fallback(wrong_shape):
    """A `quantization_config` that is *present* but not an object
    is a shape error, not an absence -- `{"quantization_config": [], "torch_dtype": "bfloat16"}`
    must never read as `BF16`. Only a genuinely *missing* (or explicit `null`) `quantization_config`
    falls back to dtype; `test_no_quantization_config_reads_dtype_then_text_config` above proves
    that case still works."""
    config = _config("hf_qwen3_8b_awq_config.json")
    config["quantization_config"] = wrong_shape
    assert "torch_dtype" in config  # the fallback this bug reached for is still right there
    assert tensor_quant_label(config) == (None, None)


def test_an_explicit_null_quantization_config_still_falls_back_to_dtype():
    """The one shape that *is* an absence: `null` reads the same as the key missing entirely."""
    config = _config("hf_qwen3_8b_awq_config.json")
    config["quantization_config"] = None
    assert tensor_quant_label(config) == ("F16", "config")


def test_config_over_64_kib_is_none():
    """P2: the 64 KiB bound is enforced by the caller (`hf.py`) before the body is even parsed as
    JSON -- this proves the bound's own size, so a change to one of the real fixtures cannot
    silently make this test meaningless."""
    assert MAX_CONFIG_JSON_BYTES == 64 * 1024


# --- safe repository paths -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "model-00001-of-00002.safetensors",
        "config.json",
        "a/b/c/d.json",  # four segments, the maximum
    ],
)
def test_an_ordinary_tensor_path_is_safe(path):
    assert is_safe_repo_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/x.safetensors",
        "../x.safetensors",
        "a/../../x.json",
        "..\\x.json",
        "C:\\x.json",
        "\\\\host\\share\\x.json",
        "a:b.json",
        "a/b/c/d/e.json",  # five segments
    ],
)
def test_an_unsafe_tensor_path_is_rejected(path):
    assert not is_safe_repo_path(path)
    assert SAFE_REPO_PATH_RE.fullmatch(path) is None


# --- start files ------------------------------------------------------------------------------------


def _file(name: str, role: str) -> PackageFile:
    return PackageFile(name=name, role=role, size_bytes=1)


def test_start_files_are_weights_plus_json_tokenizer_and_template():
    files = [
        _file("model-00001-of-00002.safetensors", "weights_shard"),
        _file("model-00002-of-00002.safetensors", "weights_shard"),
        _file("config.json", "other"),
        _file("tokenizer.json", "other"),
        _file("tokenizer.model", "other"),
        _file("tokenizer_config.json", "other"),
        _file("chat_template.jinja", "other"),
        _file("vocab.tiktoken", "other"),
        _file("README.md", "other"),
        _file(".gitattributes", "other"),
        _file("image.png", "other"),
    ]

    names = {file.name for file in start_files(files)}

    assert names == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.tiktoken",
    }
    assert "README.md" not in names
    assert ".gitattributes" not in names
    assert "image.png" not in names
