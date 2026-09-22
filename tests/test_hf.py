"""Tests for modelroom.hf: base-model architecture/meta and packager-area package assembly.

Real fixtures cover the ordinary paths (the Unsloth Qwen3.5-9B GGUF packager repo, the
Qwen/Qwen3.5-9B publisher repo, the real 401-for-nonexistent-repo response, the real
paginated-tree mechanism against a synthetic two-page fixture); everything that needs a
specific defect planted (alias repos, a broken tree fetch, an owner outside the allow-list)
uses small synthetic Response objects built directly in this file, the same way
test_provenance.py plants defects no single real repo happens to have.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modelroom.config import Configuration
from modelroom.contracts import Architecture, BaseModelSpec
from modelroom.hf import candidate_owners, fetch_base_model_meta, fetch_hf_area
from modelroom.http import Response

from fixture_support import build_transport, envelope_response, json_response

RUN_AT = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
QWEN_SHA = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
UNSLOTH_SHA = "3885219b6810b007914f3a7950a8d1b469d598a5"


def _unknown_architecture(source_repo: str) -> Architecture:
    return Architecture(source_repo=source_repo, source_revision=None, kind="unknown")


def _qwen35_9b(**overrides) -> BaseModelSpec:
    defaults = dict(
        hf_repo="Qwen/Qwen3.5-9B",
        repo_aliases=["Qwen3.5-9B-GGUF"],
        ollama_base="qwen3.5",
        ollama_tag="9b",
        publisher="Qwen",
        parameters_b=9.653104368,
        architecture=_unknown_architecture("Qwen/Qwen3.5-9B"),
    )
    defaults.update(overrides)
    return BaseModelSpec(**defaults)


def _base_config(**overrides) -> dict:
    defaults = dict(
        schema_version=1,
        families=[
            {
                "name": "qwen3.5",
                "base_models": [
                    {
                        "hf_repo": "Qwen/Qwen3.5-9B",
                        "repo_aliases": ["Qwen3.5-9B-GGUF"],
                        "ollama_base": "qwen3.5",
                        "ollama_tag": "9b",
                    }
                ],
            }
        ],
        packagers=["unsloth", "bartowski"],
        publishers=["Qwen"],
        machines={},
        paths={"state": "//x/state", "markdown": "//x/models.md"},
    )
    defaults.update(overrides)
    return Configuration.from_dict(defaults)


# --- fetch_base_model_meta --------------------------------------------------------------


def test_fetch_base_model_meta_reads_parameters_and_resolves_a_hybrid_config_to_unknown():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"): json_response(
                "hf_qwen_qwen35_9b_model.json"
            ),
            (
                "GET",
                f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{QWEN_SHA}/config.json",
            ): json_response("hf_qwen_qwen35_9b_config.json"),
        }
    )

    meta = fetch_base_model_meta(transport, "Qwen/Qwen3.5-9B")

    assert meta.publisher == "Qwen"
    assert meta.parameters_b == pytest.approx(9653104368 / 1e9)
    assert meta.architecture.kind == "unknown"
    assert meta.architecture.source_revision == QWEN_SHA


def test_fetch_base_model_meta_repo_not_found_resolves_unknown_with_no_parameters():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/unsloth/Does-Not-Exist-GGUF"): envelope_response(
                "hf_unsloth_does_not_exist_model.json"
            ),
        }
    )
    meta = fetch_base_model_meta(transport, "unsloth/Does-Not-Exist-GGUF")

    assert meta.publisher == "unsloth"
    assert meta.parameters_b is None
    assert meta.architecture.kind == "unknown"
    assert meta.architecture.source_revision is None


def test_fetch_base_model_meta_missing_config_json_is_unknown_but_never_raises():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"): json_response(
                "hf_qwen_qwen35_9b_model.json"
            ),
            (
                "GET",
                f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{QWEN_SHA}/config.json",
            ): Response(status=404, headers={}, body=b'{"error": "not found"}'),
        }
    )

    meta = fetch_base_model_meta(transport, "Qwen/Qwen3.5-9B")

    assert meta.architecture.kind == "unknown"
    # safetensors.total is still read off the model-info response, independent of config.json
    assert meta.parameters_b == pytest.approx(9653104368 / 1e9)


# --- candidate_owners --------------------------------------------------------------------


def test_candidate_owners_is_configured_packagers_plus_the_base_models_own_owner():
    config = _base_config()
    owners = candidate_owners(config, _qwen35_9b())
    assert owners == ["unsloth", "bartowski", "Qwen"]


def test_candidate_owners_never_includes_an_owner_outside_the_allow_list():
    config = _base_config(packagers=["unsloth"])
    owners = candidate_owners(config, _qwen35_9b())
    assert "SomeRandomOwner" not in owners
    assert set(owners) <= config.allowed_owners()


def test_candidate_owners_deduplicates_when_the_base_owner_is_also_a_packager():
    config = _base_config(packagers=["unsloth", "Qwen"])
    owners = candidate_owners(config, _qwen35_9b())
    assert owners == ["unsloth", "Qwen"]


# --- fetch_hf_area: happy path, real fixtures ---------------------------------------------


def test_fetch_hf_area_assembles_one_package_per_quantization_from_the_tree():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF"): json_response(
                "hf_unsloth_qwen35_9b_gguf_model.json"
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF/tree/{UNSLOTH_SHA}?recursive=true",
            ): json_response("hf_unsloth_qwen35_9b_gguf_tree.json"),
        }
    )

    outcome = fetch_hf_area(transport, _qwen35_9b(), owner="unsloth", run_at=RUN_AT)

    assert outcome.status == "complete"
    assert outcome.error is None
    assert len(outcome.packages) == 22
    quantizations = {pkg.quantization for pkg in outcome.packages}
    assert "Q4_1" in quantizations
    assert "BF16" in quantizations
    for pkg in outcome.packages:
        assert pkg.format == "gguf"
        assert pkg.repo == "unsloth/Qwen3.5-9B-GGUF"
        assert pkg.revision == UNSLOTH_SHA
        assert pkg.complete is True
        assert pkg.provenance == "metadata_ok"
        assert pkg.observed_at == RUN_AT
        assert pkg.last_seen == RUN_AT
        assert pkg.active is True
        # the repo-wide extras (mmproj, imatrix, README, .gitattributes) travel with every
        # quantization package; the identity-defining weight file is always exactly one.
        weight_files = [f for f in pkg.files if f.role in ("weights", "weights_shard")]
        assert len(weight_files) == 1


def test_fetch_hf_area_mmproj_and_other_files_never_take_a_quantization():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF"): json_response(
                "hf_unsloth_qwen35_9b_gguf_model.json"
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF/tree/{UNSLOTH_SHA}?recursive=true",
            ): json_response("hf_unsloth_qwen35_9b_gguf_tree.json"),
        }
    )

    outcome = fetch_hf_area(transport, _qwen35_9b(), owner="unsloth", run_at=RUN_AT)

    mmproj_names = {f.name for pkg in outcome.packages for f in pkg.files if f.role == "mmproj"}
    assert mmproj_names == {"mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"}
    other_names = {f.name for pkg in outcome.packages for f in pkg.files if f.role == "other"}
    assert other_names == {".gitattributes", "README.md", "imatrix_unsloth.gguf_file"}


# --- fetch_hf_area: pagination over the tree (synthetic, documented in fixtures/README.md) -


def test_fetch_hf_area_follows_link_header_pagination_across_tree_pages():
    sha = "1111111111111111111111111111111111111111"
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Big-Repo-GGUF"): Response(
                status=200,
                headers={},
                body=(
                    b'{"sha": "' + sha.encode() + b'", '
                    b'"tags": ["base_model:synthetic/Big-Repo"]}'
                ),
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Big-Repo-GGUF/tree/{sha}?recursive=true",
            ): envelope_response("hf_synthetic_paginated_tree_page1.json"),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Big-Repo-GGUF/tree/{sha}?recursive=true&cursor=page2",
            ): envelope_response("hf_synthetic_paginated_tree_page2.json"),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/Big-Repo", repo_aliases=["Big-Repo-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "complete"
    assert {pkg.quantization for pkg in outcome.packages} == {"Q4_K_M", "Q8_0"}
    assert len(transport.calls) == 3


# --- fetch_hf_area: a 404/401 candidate is skipped, the area still completes --------------


def test_fetch_hf_area_candidate_repo_not_found_leaves_the_area_complete_and_empty():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/unsloth/Does-Not-Exist-GGUF"): envelope_response(
                "hf_unsloth_does_not_exist_model.json"
            ),
        }
    )
    base_model = _qwen35_9b(hf_repo="unsloth/Does-Not-Exist", repo_aliases=[], ollama_base=None, ollama_tag=None)

    outcome = fetch_hf_area(transport, base_model, owner="unsloth", run_at=RUN_AT)

    assert outcome.status == "complete"
    assert outcome.error is None
    assert outcome.packages == []


# --- fetch_hf_area: an alias repo under a packager is a distinct candidate ----------------


def test_fetch_hf_area_finds_a_package_only_under_its_alias_repo():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/packager/Nova-7B-GGUF"): envelope_response(
                "hf_unsloth_does_not_exist_model.json"
            ),
            ("GET", "https://huggingface.co/api/models/packager/Nova-7B-Legacy-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + b"a" * 40 + b'", "tags": ["base_model:acme/Nova-7B"]}',
            ),
            (
                "GET",
                "https://huggingface.co/api/models/packager/Nova-7B-Legacy-GGUF/tree/"
                + "a" * 40
                + "?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=(
                    b'[{"type": "file", "path": "Nova-7B-Q4_K_M.gguf", "size": 123, '
                    b'"lfs": {"oid": "' + b"c" * 64 + b'"}}]'
                ),
            ),
        }
    )
    base_model = BaseModelSpec(
        hf_repo="acme/Nova-7B",
        repo_aliases=["Nova-7B-Legacy-GGUF"],
        publisher="acme",
        parameters_b=7.0,
        architecture=_unknown_architecture("acme/Nova-7B"),
    )

    outcome = fetch_hf_area(transport, base_model, owner="packager", run_at=RUN_AT)

    assert outcome.status == "complete"
    assert len(outcome.packages) == 1
    package = outcome.packages[0]
    assert package.repo == "packager/Nova-7B-Legacy-GGUF"
    assert package.provenance == "metadata_ok"


# --- fetch_hf_area: errors other than 404/401 end the area incomplete --------------------


def test_fetch_hf_area_tree_failure_ends_the_area_incomplete_with_no_packages():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF"): json_response(
                "hf_unsloth_qwen35_9b_gguf_model.json"
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF/tree/{UNSLOTH_SHA}?recursive=true",
            ): Response(status=500, headers={}, body=b"internal error"),
        }
    )

    outcome = fetch_hf_area(transport, _qwen35_9b(), owner="unsloth", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


def test_fetch_hf_area_model_info_server_error_ends_the_area_incomplete():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF"): Response(
                status=500, headers={}, body=b"internal error"
            ),
        }
    )

    outcome = fetch_hf_area(transport, _qwen35_9b(), owner="unsloth", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error


# --- AP3 acceptance findings ----------------------------------------------------------------
# --- F2: an mmproj marker anywhere in the basename (not only a prefix) is role "mmproj" ------


def test_fetch_hf_area_mmproj_marker_anywhere_in_the_basename_is_role_mmproj():
    sha = "2" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Marker-Repo-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/Marker-Repo"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Marker-Repo-GGUF/tree/{sha}?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=(
                    b'[{"type": "file", "path": "Marker-Repo-Q4_K_M.gguf", "size": 10, '
                    b'"lfs": {"oid": "' + b"a" * 64 + b'"}},'
                    # mradermacher's dot convention: "mmproj" sits mid-name, not at the start.
                    b'{"type": "file", "path": "Marker-Repo.mmproj-Q8_0.gguf", "size": 5, '
                    b'"lfs": {"oid": "' + b"b" * 64 + b'"}}]'
                ),
            ),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/Marker-Repo", repo_aliases=["Marker-Repo-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "complete"
    assert len(outcome.packages) == 1
    package = outcome.packages[0]
    roles = {f.name: f.role for f in package.files}
    assert roles["Marker-Repo.mmproj-Q8_0.gguf"] == "mmproj"
    assert package.quantization == "Q4_K_M"


# --- F5: an importance-matrix file is an extra, never its own weight package -----------------


def test_fetch_hf_area_imatrix_file_is_an_extra_not_its_own_weight_package():
    # The base/repo name deliberately avoids the substring "imatrix" itself (unlike the real
    # "MiniMax-M3-imatrix.gguf" example) -- otherwise the weight file's own name would also
    # carry the marker and this test would not isolate the finding.
    sha = "4" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/MiniMax-M3-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/MiniMax-M3"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/MiniMax-M3-GGUF/tree/{sha}?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=(
                    b'[{"type": "file", "path": "MiniMax-M3-Q4_K_M.gguf", "size": 10, '
                    b'"lfs": {"oid": "' + b"a" * 64 + b'"}},'
                    b'{"type": "file", "path": "MiniMax-M3-imatrix.gguf", "size": 2, '
                    b'"lfs": {"oid": "' + b"d" * 64 + b'"}}]'
                ),
            ),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/MiniMax-M3", repo_aliases=["MiniMax-M3-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "complete"
    assert len(outcome.packages) == 1
    package = outcome.packages[0]
    roles = {f.name: f.role for f in package.files}
    assert roles["MiniMax-M3-imatrix.gguf"] == "other"
    weight_files = [f for f in package.files if f.role in ("weights", "weights_shard")]
    assert len(weight_files) == 1


# --- F3: Qwen's own "-split-NNNNN-of-NNNNN" shard suffix is role weights_shard --------------


def test_fetch_hf_area_qwen_split_shard_suffix_is_role_weights_shard_and_one_package():
    sha = "5" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Split-Repo-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/Split-Repo"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Split-Repo-GGUF/tree/{sha}?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=(
                    b'[{"type": "file", "path": "Split-Repo-F16-split-00001-of-00002.gguf", "size": 10, '
                    b'"lfs": {"oid": "' + b"e" * 64 + b'"}},'
                    b'{"type": "file", "path": "Split-Repo-F16-split-00002-of-00002.gguf", "size": 10, '
                    b'"lfs": {"oid": "' + b"f" * 64 + b'"}}]'
                ),
            ),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/Split-Repo", repo_aliases=["Split-Repo-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "complete"
    assert len(outcome.packages) == 1
    package = outcome.packages[0]
    assert package.quantization == "F16"
    assert package.complete is True
    assert {f.role for f in package.files} == {"weights_shard"}


def test_fetch_hf_area_budget_exhausted_mid_area_ends_it_incomplete():
    from modelroom.http import BudgetedTransport

    inner = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF"): json_response(
                "hf_unsloth_qwen35_9b_gguf_model.json"
            ),
        }
    )
    budgeted = BudgetedTransport(inner, budget=1)
    budgeted("GET", "https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF")  # spend the only slot

    outcome = fetch_hf_area(budgeted, _qwen35_9b(), owner="unsloth", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert "budget exhausted" in outcome.error
    assert outcome.packages == []
