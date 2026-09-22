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
    identity_stem,
    package_identity_key,
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


# --- normalize_file_stem: shard suffix regardless of extension (finding 6) ----------------


def test_normalize_file_stem_strips_shard_suffix_on_a_safetensors_file():
    assert (
        normalize_file_stem("model.safetensors-00001-of-00002.safetensors") == "model.safetensors"
    )


# --- package_identity_key: sorted, de-duplicated, order-independent (finding 6) -----------


class _StubPackageFile:
    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role


class _StubPackage:
    """Duck-types the subset of `Package` that `package_identity_key` reads."""

    def __init__(self, *, source: str, repo: str | None, ollama_name: str | None, files: list) -> None:
        self.source = source
        self.repo = repo
        self.ollama_name = ollama_name
        self.files = files


def test_package_identity_key_third_element_is_sorted_and_order_independent():
    file_a = _StubPackageFile("A-Q4_K_M.gguf", "weights")
    file_b = _StubPackageFile("B-Q4_K_M.gguf", "weights")

    forward = _StubPackage(source="huggingface", repo="acme/Nova-Bundle-GGUF", ollama_name=None, files=[file_a, file_b])
    reversed_ = _StubPackage(source="huggingface", repo="acme/Nova-Bundle-GGUF", ollama_name=None, files=[file_b, file_a])

    key_forward = package_identity_key(forward)
    key_reversed = package_identity_key(reversed_)
    assert key_forward == key_reversed
    assert key_forward[2] == "A-Q4_K_M.gguf|B-Q4_K_M.gguf"


def test_package_identity_key_deduplicates_repeated_stems():
    file_1 = _StubPackageFile("Nova-70B-Q4_K_M-00001-of-00002.gguf", "weights_shard")
    file_2 = _StubPackageFile("Nova-70B-Q4_K_M-00002-of-00002.gguf", "weights_shard")
    package = _StubPackage(source="huggingface", repo="acme/Nova-70B-GGUF", ollama_name=None, files=[file_1, file_2])
    assert package_identity_key(package)[2] == "Nova-70B-Q4_K_M.gguf"


# --- parse_ollama_tag_quant: last token, descriptive middle tokens ignored (finding 7) -----


def test_parse_ollama_tag_quant_reads_the_last_token_past_a_descriptive_middle_token():
    assert parse_ollama_tag_quant("70b-instruct-q4_K_M") == "Q4_K_M"


# --- parse_hf_quant: mmproj check applies to the basename only (finding 8) ----------------


def test_parse_hf_quant_mmproj_check_applies_to_the_basename_with_forward_slash():
    assert parse_hf_quant("projectors/mmproj-F16.gguf") is None


def test_parse_hf_quant_mmproj_check_applies_to_the_basename_with_backslash():
    assert parse_hf_quant("projectors\\mmproj-F16.gguf") is None


# --- QUANT_ORDER is pinned, and the longest-suffix rule is exercised on a real overlap ----
# (finding 13; the pinned tuple itself moved to the AP3-extension test below, finding 4)


def test_parse_hf_quant_resolves_the_longest_suffix_on_a_real_overlap():
    # "UD-IQ2_M" ends with "-IQ2_M", so both "IQ2_M" and "UD-IQ2_M" are real QUANT_ORDER
    # candidates for this filename; the longer one must win, and the plain "IQ2_M" filename
    # must still resolve to the shorter label rather than always preferring the longer one.
    assert parse_hf_quant("Nova-7B-UD-IQ2_M.gguf") == "UD-IQ2_M"
    assert parse_hf_quant("Nova-7B-IQ2_M.gguf") == "IQ2_M"


# --- the HF tree fixture actually drives a size-based ordering test (finding 14) -----------


def _unsloth_tree_entries() -> list[dict]:
    tree = json.loads((FIXTURES / "hf_unsloth_qwen35_9b_gguf_tree.json").read_text(encoding="utf-8"))
    return [entry for entry in tree if entry["type"] == "file" and entry["path"].endswith(".gguf")]


def _hf_package_from_tree_entry(entry: dict) -> Package:
    filename = entry["path"]
    return Package(
        source="huggingface",
        repo="unsloth/Qwen3.5-9B-GGUF",
        revision="3885219b6810b007914f3a7950a8d1b469d598a5",
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format="gguf",
        files=[PackageFile(name=filename, role="weights", size_bytes=entry["size"], digest=None)],
        complete=True,
        quantization=parse_hf_quant(filename),
        default_context=None,
        provenance="metadata_ok",
        unresolved_reason=None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )


def test_sort_key_orders_real_tree_fixture_packages_from_smallest_to_largest():
    entries_by_name = {entry["path"]: entry for entry in _unsloth_tree_entries()}
    ordered_filenames = [
        "Qwen3.5-9B-UD-IQ2_XXS.gguf",  # UD-IQ2_XXS: QUANT_ORDER index 2
        "Qwen3.5-9B-Q4_K_M.gguf",  # Q4_K_M: QUANT_ORDER index 16
        "Qwen3.5-9B-Q8_0.gguf",  # Q8_0: QUANT_ORDER index 23
    ]
    packages = [_hf_package_from_tree_entry(entries_by_name[name]) for name in ordered_filenames]
    # Sanity: the fixture's real sizes also happen to increase in this order, so this genuinely
    # exercises sort_key against real recorded sizes, not just the quant index in isolation.
    sizes = [entries_by_name[name]["size"] for name in ordered_filenames]
    assert sizes == sorted(sizes)

    shuffled = [packages[2], packages[0], packages[1]]
    assert sorted(shuffled, key=sort_key) == packages


def test_sort_key_uses_the_size_when_two_real_packages_share_a_quantization():
    # Same quantization from two packager repos: only the recorded size can order them, so a
    # sort key that dropped the size component would fail here.
    entry = {e["path"]: e for e in _unsloth_tree_entries()}["Qwen3.5-9B-Q4_K_M.gguf"]
    smaller = _hf_package_from_tree_entry(entry)
    larger = _hf_package_from_tree_entry({**entry, "size": entry["size"] * 2}).model_copy(
        update={"repo": "aaa-packager/Qwen3.5-9B-GGUF"}
    )
    assert larger.quantization == smaller.quantization
    assert sorted([larger, smaller], key=sort_key) == [smaller, larger]


# --- Codex round 2: size token and mlx anywhere; identity keeps the file format ------------


def test_ollama_tag_without_a_size_token_carries_no_quantization():
    assert parse_ollama_tag_quant("banana-q4_K_M") is None
    assert parse_ollama_tag_quant("mlx-bf16") is None
    assert parse_ollama_tag_quant("1.5b-q8_0") == "Q8_0"
    assert parse_ollama_tag_quant("270m-fp16") is None


def test_identity_stem_keeps_the_extension_but_drops_the_shard_suffix():
    assert identity_stem("Nova-7B-BF16.gguf") == "Nova-7B-BF16.gguf"
    assert identity_stem("Nova-7B-BF16.safetensors") == "Nova-7B-BF16.safetensors"
    assert identity_stem("Nova-7B-BF16-00001-of-00002.gguf") == "Nova-7B-BF16.gguf"
    assert identity_stem("Nova-7B-BF16-00002-of-00002.gguf") == "Nova-7B-BF16.gguf"


# --- AP3 acceptance findings, real 2026-09-22 filenames -----------------------------------
# --- F1: unsloth per-quant subfolders (BF16/, UD-Q2_K_XL/, ...) ----------------------------


def test_parse_hf_quant_resolves_unsloth_per_quant_subfolder_filenames():
    assert parse_hf_quant("BF16/GLM-5.2-BF16-00001-of-00033.gguf") == "BF16"
    assert parse_hf_quant("UD-Q2_K_XL/Kimi-K2.6-UD-Q2_K_XL-00001-of-00008.gguf") == "UD-Q2_K_XL"
    assert parse_hf_quant("Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf") == "Q8_0"


def test_identity_stem_keeps_the_folder_for_per_quant_subfolders():
    # Two different unsloth subfolders can carry a file with the same basename (e.g. the same
    # shard index); they must never collapse into one package identity.
    a = identity_stem("BF16/model-00001-of-00002.gguf")
    b = identity_stem("Q8_0/model-00001-of-00002.gguf")
    assert a != b


# --- F2: mradermacher's dot convention (`<base>.<QUANT>.gguf`) and lowercase f16 -----------


def test_parse_hf_quant_accepts_the_mradermacher_dot_convention():
    assert parse_hf_quant("Qwen3.5-9B.IQ4_XS.gguf") == "IQ4_XS"
    assert parse_hf_quant("gemma-4-31B-it.Q2_K.gguf") == "Q2_K"
    assert parse_hf_quant("EuroLLM-9B-Instruct-2512.Q8_0.gguf") == "Q8_0"


def test_parse_hf_quant_accepts_lowercase_f16_and_bf16():
    assert parse_hf_quant("Qwen3.5-9B.f16.gguf") == "F16"
    assert parse_hf_quant("Nova-7B-bf16.gguf") == "BF16"


def test_parse_hf_quant_mmproj_marker_anywhere_in_the_basename_is_none():
    # Not only a basename that *starts* with "mmproj" -- mradermacher's dot convention puts it
    # mid-name.
    assert parse_hf_quant("Qwen3.5-9B.mmproj-f16.gguf") is None
    assert parse_hf_quant("Qwen3-VL-235B-A22B-Instruct.mmproj-Q8_0.gguf") is None


# --- F3: Qwen's own "-split-NNNNN-of-NNNNN" shard suffix -----------------------------------


def test_normalize_file_stem_strips_the_qwen_split_shard_suffix():
    assert (
        normalize_file_stem("Qwen3VL-235B-A22B-Instruct-F16-split-00001-of-00010.gguf")
        == "Qwen3VL-235B-A22B-Instruct-F16"
    )


def test_parse_hf_quant_resolves_the_qwen_split_shard_filename():
    assert parse_hf_quant("Qwen3VL-235B-A22B-Instruct-Q4_K_M-split-00001-of-00003.gguf") == "Q4_K_M"


# --- F5: an importance-matrix file is never a weight, never its own package ----------------


def test_parse_hf_quant_imatrix_marker_anywhere_is_none():
    assert parse_hf_quant("MiniMax-M3-imatrix.gguf") is None


# --- F4: QUANT_ORDER extended for tokens observed live but previously unparsed -------------


def test_quant_order_is_exactly_this_tuple_in_order_after_ap3_extension():
    assert QUANT_ORDER == (
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
    assert len(QUANT_ORDER) == 50


def test_unsloth_ud_iq4_builds_parse_to_the_ud_member():
    # Live snapshot 2026-09-22: unsloth/GLM-5.2-GGUF and unsloth/MiniMax-M3-GGUF ship these.
    assert parse_hf_quant("UD-IQ4_NL/GLM-5.2-UD-IQ4_NL-00001-of-00009.gguf") == "UD-IQ4_NL"
    assert parse_hf_quant("UD-IQ4_XS/MiniMax-M3-UD-IQ4_XS-00001-of-00006.gguf") == "UD-IQ4_XS"
    assert QUANT_ORDER.index("IQ4_XS") < QUANT_ORDER.index("UD-IQ4_XS") < QUANT_ORDER.index("IQ4_NL")
    assert QUANT_ORDER.index("IQ4_NL") < QUANT_ORDER.index("UD-IQ4_NL") < QUANT_ORDER.index("Q4_0")


def test_quant_order_ternary_and_1bit_members_sort_below_iq1_s():
    below_iq1_s = ("TQ1_0", "UD-TQ1_0", "UD-Q1_0", "TQ2_0", "UD-TQ2_0")
    iq1_s_index = QUANT_ORDER.index("IQ1_S")
    for member in below_iq1_s:
        assert QUANT_ORDER.index(member) < iq1_s_index, member


def test_quant_order_iq1_and_iq2_chain_is_ascending():
    chain = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M"]
    indices = [QUANT_ORDER.index(m) for m in chain]
    assert indices == sorted(indices)


def test_quant_order_q2_k_l_sorts_above_q2_k():
    assert QUANT_ORDER.index("Q2_K") < QUANT_ORDER.index("Q2_K_L")


def test_quant_order_iq3_chain_is_ascending():
    chain = ["IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M"]
    indices = [QUANT_ORDER.index(m) for m in chain]
    assert indices == sorted(indices)


def test_quant_order_q3_k_chain_is_ascending():
    chain = ["Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_XL"]
    indices = [QUANT_ORDER.index(m) for m in chain]
    assert indices == sorted(indices)


def test_quant_order_mxfp4_sorts_between_q4_k_m_and_q5_k_s():
    q4_k_m = QUANT_ORDER.index("Q4_K_M")
    q5_k_s = QUANT_ORDER.index("Q5_K_S")
    assert q4_k_m < QUANT_ORDER.index("MXFP4") < q5_k_s
    assert q4_k_m < QUANT_ORDER.index("MXFP4_MOE") < q5_k_s


def test_two_file_formats_of_the_same_build_have_distinct_identities():
    def build(name: str, fmt: str) -> Package:
        return Package(
            source="huggingface",
            repo="acme/Nova-7B",
            revision="1" * 40,
            base_model_hf_repo="acme/Nova-7B",
            format=fmt,
            files=[PackageFile(name=name, role="weights", size_bytes=1, digest=None)],
            complete=True,
            quantization=None,
            default_context=None,
            provenance="unresolved",
            unresolved_reason="format" if fmt != "gguf" else "repo_name",
            approval=None,
            observed_at=_NOW,
            last_seen=_NOW,
            active=True,
        )

    gguf = build("Nova-7B-BF16.gguf", "gguf")
    tensor = build("Nova-7B-BF16.safetensors", "tensor")
    assert package_identity_key(gguf) != package_identity_key(tensor)
