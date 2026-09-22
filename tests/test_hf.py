"""Tests for modelroom.hf: base-model architecture/meta and packager-area package assembly.

Real fixtures cover the ordinary paths (the Unsloth Qwen3.5-9B GGUF packager repo, the
Qwen/Qwen3.5-9B publisher repo, the real 401-for-nonexistent-repo response, the real
paginated-tree mechanism against a synthetic two-page fixture); everything that needs a
specific defect planted (alias repos, a broken tree fetch, an owner outside the allow-list)
uses small synthetic Response objects built directly in this file, the same way
test_provenance.py plants defects no single real repo happens to have.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from modelroom.config import Configuration
from modelroom.contracts import Approval, Architecture, BaseModelSpec, Package, PackageFile
from modelroom.hf import _MAX_TREE_PAGES, candidate_owners, fetch_base_model_meta, fetch_hf_area
from modelroom.http import Response
from modelroom.quantization import package_identity_key

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


def test_fetch_base_model_meta_zero_safetensors_total_is_none_not_a_fabricated_zero():
    # R2: safetensors.total == 0 must resolve to parameters_b=None, never a bare 0.0 that
    # BaseModelSpec.parameters_b (gt=0) would reject once fetch.py tries to build a spec from it.
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + QWEN_SHA.encode() + b'", "safetensors": {"total": 0}}',
            ),
            (
                "GET",
                f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{QWEN_SHA}/config.json",
            ): Response(status=404, headers={}, body=b"{}"),
        }
    )

    meta = fetch_base_model_meta(transport, "Qwen/Qwen3.5-9B")

    assert meta.parameters_b is None


def test_fetch_base_model_meta_subnormal_safetensors_total_is_none_not_a_crash():
    # Fix-round 3: 1e-320 passes the R2 "> 0" check but underflows to 0.0 once divided by 1e9,
    # which BaseModelSpec.parameters_b's own gt=0 constraint would reject -- caught here instead.
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + QWEN_SHA.encode() + b'", "safetensors": {"total": 1e-320}}',
            ),
            (
                "GET",
                f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{QWEN_SHA}/config.json",
            ): Response(status=404, headers={}, body=b"{}"),
        }
    )

    meta = fetch_base_model_meta(transport, "Qwen/Qwen3.5-9B")

    assert meta.parameters_b is None


def test_fetch_base_model_meta_huge_json_integer_safetensors_total_is_none_not_a_crash():
    # Fix-round 3: a JSON integer far outside float range (json.loads parses it without
    # complaint) overflows the "/ 1e9" division itself -- caught here instead of raising.
    huge_total = str(10**400).encode()
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + QWEN_SHA.encode() + b'", "safetensors": {"total": ' + huge_total + b"}}",
            ),
            (
                "GET",
                f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{QWEN_SHA}/config.json",
            ): Response(status=404, headers={}, body=b"{}"),
        }
    )

    meta = fetch_base_model_meta(transport, "Qwen/Qwen3.5-9B")

    assert meta.parameters_b is None


def test_fetch_base_model_meta_malformed_sha_resolves_architecture_unknown_and_never_raises():
    # R2: a non-40-hex 'sha' must never reach Architecture.source_revision's validator -- that
    # would raise a pydantic ValidationError straight out of fetch_base_model_meta, which its
    # own docstring promises never happens.
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"): Response(
                status=200, headers={}, body=b'{"sha": "bad"}'
            ),
        }
    )

    meta = fetch_base_model_meta(transport, "Qwen/Qwen3.5-9B")

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


# --- P3-3 (fix-round 5): a Link: rel="next" chain that never terminates must not fetch forever

def test_fetch_hf_area_tree_pagination_stops_at_a_hard_cap():
    sha = "2222222222222222222222222222222222222222"
    base_url = "https://huggingface.co/api/models/synthetic/Forever-GGUF/tree/" + sha + "?recursive=true"

    def page_url(n: int) -> str:
        return base_url if n == 1 else f"{base_url}&cursor=page{n}"

    mapping = {
        ("GET", "https://huggingface.co/api/models/synthetic/Forever-GGUF"): Response(
            status=200, headers={}, body=(b'{"sha": "' + sha.encode() + b'", "tags": []}')
        ),
    }
    # a Link header on every page, including the last one this cap allows -- the chain never
    # terminates on its own, so the cap is the only thing that ever stops it.
    for n in range(1, _MAX_TREE_PAGES + 2):
        mapping[("GET", page_url(n))] = Response(
            status=200, headers={"Link": f'<{page_url(n + 1)}>; rel="next"'}, body=b"[]"
        )
    transport = build_transport(mapping)
    base_model = _qwen35_9b(
        hf_repo="synthetic/Forever", repo_aliases=["Forever-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert str(_MAX_TREE_PAGES) in outcome.error
    # the model-info request, plus exactly _MAX_TREE_PAGES tree-page requests -- the page that
    # would exceed the cap is never made.
    assert len(transport.calls) == 1 + _MAX_TREE_PAGES
    assert ("GET", page_url(_MAX_TREE_PAGES + 1)) not in transport.calls


# --- P3-4 (fix-round 5): a malformed 'sha' from the packager repo's own model-info must never
# reach the tree-fetch URL unchecked ---------------------------------------------------------


def test_fetch_hf_area_rejects_a_malformed_sha_before_fetching_the_tree():
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/unsloth/Nova-7B-GGUF"): Response(
                status=200,
                headers={},
                # not 40-hex: a packager repo response this project has never actually seen but
                # must still never be trusted to build a URL path segment unchecked.
                body=b'{"sha": "not-a-real-sha", "tags": []}',
            ),
        }
    )
    base_model = _qwen35_9b(hf_repo="acme/Nova-7B", repo_aliases=[], ollama_base=None, ollama_tag=None, publisher="acme")

    outcome = fetch_hf_area(transport, base_model, owner="unsloth", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert "sha" in outcome.error
    # the tree was never requested -- only the one model-info call was ever made.
    assert transport.calls == [("GET", "https://huggingface.co/api/models/unsloth/Nova-7B-GGUF")]


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


# --- F3: a malformed tree entry ends the area incomplete, never raises out of fetch_hf_area --


def test_fetch_hf_area_tree_entry_not_a_dict_ends_area_incomplete():
    sha = "6" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Null-Entry-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/Null-Entry"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Null-Entry-GGUF/tree/{sha}?recursive=true",
            ): Response(status=200, headers={}, body=b"[null]"),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/Null-Entry", repo_aliases=["Null-Entry-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


# --- F4: a weight file without a size is a shape error, the area ends incomplete -----------


def test_fetch_hf_area_weight_file_without_size_ends_area_incomplete():
    sha = "7" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/No-Size-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/No-Size"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/No-Size-GGUF/tree/{sha}?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=b'[{"type": "file", "path": "No-Size-Q4_K_M.gguf", '
                b'"lfs": {"oid": "' + b"a" * 64 + b'"}}]',  # no "size" field at all
            ),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/No-Size", repo_aliases=["No-Size-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


def test_fetch_hf_area_non_weight_file_without_size_still_completes():
    # A missing size on an mmproj/other file is not a shape error -- only weight files require
    # a real size (CONTRACTS.md/F4).
    sha = "8" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Extra-No-Size-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/Extra-No-Size"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Extra-No-Size-GGUF/tree/{sha}?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=(
                    b'[{"type": "file", "path": "Extra-No-Size-Q4_K_M.gguf", "size": 10, '
                    b'"lfs": {"oid": "' + b"a" * 64 + b'"}},'
                    b'{"type": "file", "path": "README.md"}]'  # no size, role "other"
                ),
            ),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/Extra-No-Size", repo_aliases=["Extra-No-Size-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "complete"
    assert len(outcome.packages) == 1
    readme = next(f for f in outcome.packages[0].files if f.name == "README.md")
    assert readme.role == "other"
    assert readme.size_bytes == 0


# --- F8 (fix-round 5): a weight file 'size' far outside any real file size is a shape error --
#
# `json.loads` parses a JSON integer literal like `10**400` without complaint (Python ints have
# no fixed width); `_tree_entry_size`'s old `isinstance(int) and >= 0` check let it straight
# through as a package's `size_bytes`, and `fit.py::compute_fit`'s `sum(...) / GIB` later raised
# `OverflowError` (`int too large to convert to float`) out of `render`/`fit` instead of this
# module ending the area `incomplete` like every other shape problem.


def test_fetch_hf_area_weight_file_size_far_too_large_ends_area_incomplete():
    sha = "5" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Huge-Size-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/Huge-Size"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Huge-Size-GGUF/tree/{sha}?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=(
                    b'[{"type": "file", "path": "Huge-Size-Q4_K_M.gguf", "size": '
                    + str(10**400).encode()
                    + b', "lfs": {"oid": "'
                    + b"a" * 64
                    + b'"}}]'
                ),
            ),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/Huge-Size", repo_aliases=["Huge-Size-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


# --- F7 (fix-round 5): a malformed 'path' on a file entry is a shape error, never silently
# dropped -- dropping every entry this way would make the whole tree look genuinely empty, and
# a `complete` area with zero files deactivates every one of the old area's packages (merge
# rule) instead of ending `incomplete` and leaving them untouched.


def test_fetch_hf_area_file_entry_with_a_non_string_path_ends_area_incomplete():
    sha = "9" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Bad-Path-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/Bad-Path"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Bad-Path-GGUF/tree/{sha}?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=b'[{"type": "file", "path": 0, "size": 10}]',  # path is an int, not a string
            ),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/Bad-Path", repo_aliases=["Bad-Path-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


def test_fetch_hf_area_file_entry_with_an_empty_path_ends_area_incomplete():
    sha = "6" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Empty-Path-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": ["base_model:synthetic/Empty-Path"]}',
            ),
            (
                "GET",
                f"https://huggingface.co/api/models/synthetic/Empty-Path-GGUF/tree/{sha}?recursive=true",
            ): Response(
                status=200,
                headers={},
                body=b'[{"type": "file", "path": "", "size": 10}]',
            ),
        }
    )
    base_model = _qwen35_9b(
        hf_repo="synthetic/Empty-Path", repo_aliases=["Empty-Path-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(transport, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error
    assert outcome.packages == []


# --- F6: a previous approval survives a fetch when its content still matches ---------------


def test_fetch_hf_area_carries_forward_an_approval_bound_to_the_current_revision():
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
    approval = Approval(date=date(2026, 9, 1), content=UNSLOTH_SHA, by="acme-ai-team")
    previous = Package(
        source="huggingface",
        repo="unsloth/Qwen3.5-9B-GGUF",
        revision=UNSLOTH_SHA,
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format="gguf",
        files=[PackageFile(name="Qwen3.5-9B-Q4_1.gguf", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization="Q4_1",
        default_context=None,
        provenance="approved",
        unresolved_reason=None,
        approval=approval,
        observed_at=RUN_AT,
        last_seen=RUN_AT,
        active=True,
    )
    previous_by_key = {package_identity_key(previous): previous}

    outcome = fetch_hf_area(
        transport, _qwen35_9b(), owner="unsloth", run_at=RUN_AT, previous_by_key=previous_by_key
    )

    by_key = {package_identity_key(p): p for p in outcome.packages}
    refreshed = by_key[package_identity_key(previous)]
    assert refreshed.provenance == "approved"
    assert refreshed.approval == approval


def test_fetch_hf_area_approval_bound_to_an_older_revision_is_not_approved():
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
    stale_approval = Approval(date=date(2026, 1, 1), content="0" * 40, by="acme-ai-team")
    previous = Package(
        source="huggingface",
        repo="unsloth/Qwen3.5-9B-GGUF",
        revision="0" * 40,
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format="gguf",
        files=[PackageFile(name="Qwen3.5-9B-Q4_1.gguf", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization="Q4_1",
        default_context=None,
        provenance="approved",
        unresolved_reason=None,
        approval=stale_approval,
        observed_at=RUN_AT,
        last_seen=RUN_AT,
        active=True,
    )
    previous_by_key = {package_identity_key(previous): previous}

    outcome = fetch_hf_area(
        transport, _qwen35_9b(), owner="unsloth", run_at=RUN_AT, previous_by_key=previous_by_key
    )

    by_key = {package_identity_key(p): p for p in outcome.packages}
    refreshed = by_key[package_identity_key(previous)]
    assert refreshed.provenance != "approved"
    assert refreshed.approval == stale_approval  # kept for history, but no longer active


def test_fetch_hf_area_approval_does_not_follow_the_package_identity_to_another_base_model():
    # R3: an approval must be bound to the base model it was given for, not just the bare
    # (repo, filename) identity -- the same packager repo reassembled under a different base
    # model (e.g. a family reconfigured to a different hf_repo) must not inherit the approval.
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
    approval = Approval(date=date(2026, 9, 1), content=UNSLOTH_SHA, by="acme-ai-team")
    previous = Package(
        source="huggingface",
        repo="unsloth/Qwen3.5-9B-GGUF",
        revision=UNSLOTH_SHA,
        base_model_hf_repo="SomeOther/Different-Model",  # not Qwen/Qwen3.5-9B
        format="gguf",
        files=[PackageFile(name="Qwen3.5-9B-Q4_1.gguf", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization="Q4_1",
        default_context=None,
        provenance="approved",
        unresolved_reason=None,
        approval=approval,
        observed_at=RUN_AT,
        last_seen=RUN_AT,
        active=True,
    )
    previous_by_key = {package_identity_key(previous): previous}

    outcome = fetch_hf_area(
        transport, _qwen35_9b(), owner="unsloth", run_at=RUN_AT, previous_by_key=previous_by_key
    )

    by_key = {package_identity_key(p): p for p in outcome.packages}
    refreshed = by_key[package_identity_key(previous)]
    assert refreshed.provenance != "approved"
    assert refreshed.approval is None


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
    # P3-12 (fix-round 5): exact text, not a substring -- CONTRACTS.md, "Request budget", says
    # the fetcher "ends its current area incomplete with that message", `that message` being
    # exactly `BudgetExhaustedError`'s own `"budget exhausted"`, never a repo-prefixed variant.
    # The budget runs out on the model-info call itself here, before there is even a `repo` to
    # prefix with.
    assert outcome.error == "budget exhausted"
    assert outcome.packages == []


def test_fetch_hf_area_budget_exhausted_mid_tree_fetch_ends_it_incomplete_with_the_exact_message():
    """P3-12 (fix-round 5): the budget can also run out *inside* `_fetch_tree`'s pagination,
    after a real `repo` is already known -- CONTRACTS.md's exact-text promise still applies:
    `Area.error` is `"budget exhausted"`, never `f"{repo}: budget exhausted"` (measured before
    this fix: the tree-fetch except-clause prefixed every exception with `repo`, including this
    one, unlike the model-info except-clause a few lines above it).
    """
    sha = "3" * 40
    inner = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/synthetic/Budget-GGUF"): Response(
                status=200,
                headers={},
                body=b'{"sha": "' + sha.encode() + b'", "tags": []}',
            ),
        }
    )
    from modelroom.http import BudgetedTransport

    budgeted = BudgetedTransport(inner, budget=2)
    budgeted.used = 1  # one slot already spent elsewhere -- the model-info call below spends the
    # second (last) one, so the tree-fetch call after it is refused before it is ever made.
    base_model = _qwen35_9b(
        hf_repo="synthetic/Budget", repo_aliases=["Budget-GGUF"], ollama_base=None, ollama_tag=None
    )

    outcome = fetch_hf_area(budgeted, base_model, owner="synthetic", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error == "budget exhausted"
    assert outcome.packages == []


# --- P3-14 (fix-round 5): a message-less exception from a caller-supplied transport must never
# become an empty Area.error -- Area's own validator requires a non-empty error whenever status
# is incomplete, so an empty string here would raise ValidationError out of merge_snapshot ------


def test_fetch_hf_area_treats_a_message_less_transport_exception_as_a_real_error():
    def transport(method, url, headers=None):
        raise Exception()  # no message at all -- str(Exception()) == ""

    outcome = fetch_hf_area(transport, _qwen35_9b(), owner="unsloth", run_at=RUN_AT)

    assert outcome.status == "incomplete"
    assert outcome.error  # must be truthy, never ""
