"""Tests for modelroom.quantization: parsing, digest inheritance, ordering."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.contracts import Package, PackageFile
from modelroom.quantization import (
    QUANT_ORDER,
    inherit_from_siblings,
    normalize_file_stem,
    parse_hf_quant,
    parse_ollama_tag_quant,
    sort_key,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _unsloth_gguf_filenames() -> list[str]:
    model = json.loads((FIXTURES / "hf_unsloth_qwen35_9b_gguf_model.json").read_text(encoding="utf-8"))
    return [s["rfilename"] for s in model["siblings"] if s["rfilename"].endswith(".gguf")]


# --- normalize_file_stem ---------------------------------------------------------------


def test_normalize_file_stem_strips_extension():
    assert normalize_file_stem("Nova-7B-Q4_K_M.gguf") == "Nova-7B-Q4_K_M"


def test_normalize_file_stem_strips_shard_suffix():
    assert normalize_file_stem("Nova-70B-Q4_K_M-00002-of-00003.gguf") == "Nova-70B-Q4_K_M"


def test_normalize_file_stem_keeps_non_gguf_extension_as_is():
    # the Unsloth imatrix file has a `.gguf_file` extension, not `.gguf`
    assert normalize_file_stem("imatrix_unsloth.gguf_file") == "imatrix_unsloth.gguf_file"


# --- parse_hf_quant ----------------------------------------------------------------------


def test_parse_hf_quant_all_real_unsloth_weight_files_resolve_to_a_known_quant():
    filenames = [f for f in _unsloth_gguf_filenames() if not f.lower().startswith("mmproj")]
    assert len(filenames) == 22  # 22 quantized weight files, mmproj excluded already
    for filename in filenames:
        quant = parse_hf_quant(filename)
        assert quant in QUANT_ORDER, f"{filename} did not resolve to a known quant, got {quant!r}"


def test_parse_hf_quant_resolves_q4_1_specifically():
    assert parse_hf_quant("Qwen3.5-9B-Q4_1.gguf") == "Q4_1"


def test_parse_hf_quant_resolves_the_longest_suffix_match():
    assert parse_hf_quant("Qwen3.5-9B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"


def test_parse_hf_quant_mmproj_files_are_none():
    for filename in _unsloth_gguf_filenames():
        if filename.lower().startswith("mmproj"):
            assert parse_hf_quant(filename) is None, filename


def test_parse_hf_quant_imatrix_file_is_none():
    assert parse_hf_quant("imatrix_unsloth.gguf_file") is None


def test_parse_hf_quant_unrecognized_suffix_is_none():
    assert parse_hf_quant("Nova-7B-experimental.gguf") is None


# --- parse_ollama_tag_quant ----------------------------------------------------------------


def test_parse_ollama_tag_quant_reads_the_suffix_after_the_size_token():
    assert parse_ollama_tag_quant("9b-q4_K_M") == "Q4_K_M"


def test_parse_ollama_tag_quant_plain_size_tag_is_none():
    assert parse_ollama_tag_quant("9b") is None


def test_parse_ollama_tag_quant_tensor_build_tag_is_none():
    assert parse_ollama_tag_quant("9b-mlx-bf16") is None


def test_parse_ollama_tag_quant_matches_case_insensitively():
    assert parse_ollama_tag_quant("9b-Q4_k_m") == "Q4_K_M"


# --- inherit_from_siblings -----------------------------------------------------------------


def _ollama_weights_digest(fixture_name: str) -> str | None:
    manifest = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    for layer in manifest["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            return layer["digest"]
    return None


def test_inherit_from_siblings_fills_in_from_a_tag_with_the_same_digest():
    digest_9b = _ollama_weights_digest("ollama_qwen35_9b.json")
    digest_q4 = _ollama_weights_digest("ollama_qwen35_9b-q4_K_M.json")
    assert digest_9b == digest_q4  # the fixture's real-world fact this test relies on

    tag_digests = {"9b": digest_9b, "9b-q4_K_M": digest_q4}
    tag_quants = {"9b": parse_ollama_tag_quant("9b"), "9b-q4_K_M": parse_ollama_tag_quant("9b-q4_K_M")}
    assert tag_quants == {"9b": None, "9b-q4_K_M": "Q4_K_M"}

    result = inherit_from_siblings(tag_digests, tag_quants)
    assert result == {"9b": "Q4_K_M", "9b-q4_K_M": "Q4_K_M"}


def test_inherit_from_siblings_leaves_a_tag_without_a_matching_sibling_alone():
    tag_digests = {"9b": "sha256:aaaa", "9b-mlx-bf16": "sha256:bbbb"}
    tag_quants = {"9b": None, "9b-mlx-bf16": None}
    result = inherit_from_siblings(tag_digests, tag_quants)
    assert result == {"9b": None, "9b-mlx-bf16": None}


def test_ollama_mlx_bf16_manifest_has_no_weights_model_layer():
    manifest = json.loads((FIXTURES / "ollama_qwen35_9b-mlx-bf16.json").read_text(encoding="utf-8"))
    media_types = {layer["mediaType"] for layer in manifest["layers"]}
    assert "application/vnd.ollama.image.model" not in media_types
    assert "application/vnd.ollama.image.tensor" in media_types


# --- sort_key ---------------------------------------------------------------------------

_NOW = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)


def _hf_package(*, repo: str, filename: str, size_bytes: int, quant: str | None) -> Package:
    return Package(
        source="huggingface",
        repo=repo,
        revision="9f8e7d6c5b4a3928170615243342516078960a1b",
        base_model_hf_repo="acme/Nova-7B",
        format="gguf",
        files=[PackageFile(name=filename, role="weights", size_bytes=size_bytes, digest=None)],
        complete=True,
        quantization=quant,
        default_context=None,
        provenance="metadata_ok",
        unresolved_reason=None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )


def test_sort_key_orders_by_quant_index_first():
    small = _hf_package(repo="acme-packager/Nova-7B-GGUF", filename="Nova-7B-Q2_K.gguf", size_bytes=1, quant="Q2_K")
    large = _hf_package(repo="acme-packager/Nova-7B-GGUF", filename="Nova-7B-Q8_0.gguf", size_bytes=1, quant="Q8_0")
    assert sort_key(small) < sort_key(large)


def test_sort_key_puts_unknown_quant_last():
    known = _hf_package(repo="acme-packager/Nova-7B-GGUF", filename="Nova-7B-Q2_K.gguf", size_bytes=1, quant="Q2_K")
    unknown = _hf_package(
        repo="acme-packager/Nova-7B-GGUF", filename="Nova-7B-mystery.gguf", size_bytes=1, quant=None
    )
    assert sort_key(known) < sort_key(unknown)


def test_sort_key_breaks_equal_quant_tie_by_smaller_size_first():
    smaller = _hf_package(
        repo="acme-packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", size_bytes=100, quant="Q4_K_M"
    )
    larger = _hf_package(
        repo="acme-packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", size_bytes=200, quant="Q4_K_M"
    )
    assert sort_key(smaller) < sort_key(larger)


def test_sort_key_breaks_equal_quant_and_size_tie_by_packager_name():
    from_acme = _hf_package(
        repo="acme-packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", size_bytes=100, quant="Q4_K_M"
    )
    from_zeta = _hf_package(
        repo="zeta-packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", size_bytes=100, quant="Q4_K_M"
    )
    assert sort_key(from_acme) < sort_key(from_zeta)


def test_sort_key_final_tie_break_is_the_full_package_key():
    first = _hf_package(
        repo="acme-packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", size_bytes=100, quant="Q4_K_M"
    )
    # Same packager, same size, same quant, different repo name -> falls through to the
    # full identity key as the stable, deterministic final criterion.
    second = _hf_package(
        repo="acme-packager/Nova-7B-Instruct-GGUF",
        filename="Nova-7B-Instruct-Q4_K_M.gguf",
        size_bytes=100,
        quant="Q4_K_M",
    )
    keys = sorted([sort_key(second), sort_key(first)])
    assert keys[0][:3] == keys[1][:3]  # quant, size and packager tie
    assert keys[0][3] != keys[1][3]  # the full key still orders them deterministically
