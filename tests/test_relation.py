"""Tests for modelroom.relation: does a repo declare itself a quantization of one base model?"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelroom.relation import check_relation

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "acme/Nova-7B"


def test_real_unsloth_repo_passes_with_the_relation_only_in_its_tags():
    """The recorded packager repo carries `base_model:quantized:Qwen/Qwen3.5-9B` in `tags` and
    `base_model` in `cardData` without a `base_model_relation` -- the check must still pass."""
    model = json.loads((FIXTURES / "hf_unsloth_qwen35_9b_gguf_model.json").read_text(encoding="utf-8"))
    assert "base_model:quantized:Qwen/Qwen3.5-9B" in model["tags"]
    status, reason = check_relation(model["tags"], model.get("cardData"), "Qwen/Qwen3.5-9B")
    assert status == "quantized", reason


def test_real_unsloth_repo_fails_against_another_base_model():
    model = json.loads((FIXTURES / "hf_unsloth_qwen35_9b_gguf_model.json").read_text(encoding="utf-8"))
    status, _ = check_relation(model["tags"], model.get("cardData"), "Qwen/Qwen3.5-4B")
    assert status == "base_model_tag"


@pytest.mark.parametrize(
    "tags, card, expected",
    [
        ([f"base_model:{BASE}", f"base_model:quantized:{BASE}"], None, "quantized"),
        ([], {"base_model": BASE, "base_model_relation": "quantized"}, "quantized"),
        ([], {"base_model": [BASE], "base_model_relation": "quantized"}, "quantized"),
        ([f"base_model:{BASE}", f"base_model:finetune:{BASE}"], None, "derivative"),
        ([], {"base_model": BASE, "base_model_relation": "adapter"}, "derivative"),
        ([], {"base_model": BASE, "base_model_relation": "merge"}, "derivative"),
        ([f"base_model:{BASE}"], None, "relation_unknown"),
        ([f"base_model:{BASE}", f"base_model:distilled:{BASE}"], None, "relation_unknown"),
        ([], None, "base_model_tag"),
        (["base_model:acme/Other-7B", "base_model:quantized:acme/Other-7B"], None, "base_model_tag"),
        ([f"base_model:{BASE}", "base_model:acme/Other-7B"], None, "base_model_tag"),
        ([f"base_model:{BASE}"], {"base_model": "acme/Other-7B"}, "metadata_conflict"),
        ([f"base_model:{BASE}", "base_model:quantized:acme/Other-7B"], None, "metadata_conflict"),
        ([f"base_model:{BASE}", f"base_model:quantized:{BASE}"], {"base_model": BASE, "base_model_relation": "finetune"}, "metadata_conflict"),
        ([f"base_model:{BASE}", f"base_model:quantized:{BASE}", f"base_model:finetune:{BASE}"], None, "metadata_conflict"),
    ],
)
def test_relation_outcomes(tags, card, expected):
    status, reason = check_relation(tags, card, BASE)
    assert status == expected, reason
    assert reason


@pytest.mark.parametrize(
    "card, tags",
    [
        ({"base_model": {BASE: 123}, "base_model_relation": "quantized"}, []),
        ({"base_model": [BASE, 5], "base_model_relation": "quantized"}, []),
        ({"base_model": 123, "base_model_relation": "quantized"}, []),
        ({"base_model": BASE, "base_model_relation": ["quantized"]}, []),
        (None, [f"base_model:{BASE}", f"base_model:quantized:{BASE}", 7]),
        (None, {f"base_model:{BASE}": 1, f"base_model:quantized:{BASE}": 1}),
        (None, 5),
        (None, f"base_model:quantized:{BASE}"),
        ([f"base_model:{BASE}"], [f"base_model:{BASE}", f"base_model:quantized:{BASE}"]),
        ("quantized", [f"base_model:{BASE}", f"base_model:quantized:{BASE}"]),
    ],
)
def test_malformed_metadata_is_a_conflict_never_a_pass_or_a_crash(card, tags):
    assert check_relation(tags, card, BASE)[0] == "metadata_conflict"


def test_contradicting_relations_are_a_conflict_before_the_base_comparison():
    tags = ["base_model:acme/Other", "base_model:quantized:acme/Other"]
    card = {"base_model": "acme/Other", "base_model_relation": "finetune"}
    assert check_relation(tags, card, BASE)[0] == "metadata_conflict"


def test_untagged_noise_is_ignored():
    tags = ["gguf", "text-generation", f"base_model:{BASE}", f"base_model:quantized:{BASE}", "license:apache-2.0"]
    assert check_relation(tags, {"license": "apache-2.0"}, BASE)[0] == "quantized"


def test_missing_tags_and_card_are_tolerated():
    assert check_relation(None, None, BASE)[0] == "base_model_tag"
