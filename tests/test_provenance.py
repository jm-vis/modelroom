"""Tests for modelroom.provenance.decide_provenance.

Real Hugging Face metadata (the Unsloth Qwen3.5-9B GGUF repo, which the real base model tags
correctly) covers the ordinary metadata_ok path; the remaining branches (aliases, wrong file
stems, void approvals, tensor format) are exercised with small synthetic fixtures, since they
need to plant a specific defect that no single real repo happens to have.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from modelroom.contracts import Approval, Architecture, BaseModelSpec, Package, PackageFile
from modelroom.provenance import decide_provenance

FIXTURES = Path(__file__).parent / "fixtures"
_NOW = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
_HF_REVISION_OLD = "0" * 39 + "a"
_HF_REVISION_NEW = "0" * 39 + "b"
_OLLAMA_DIGEST_OLD = "sha256:" + "0" * 63 + "1"
_OLLAMA_DIGEST_NEW = "sha256:" + "0" * 63 + "2"


def _unknown_architecture(source_repo: str) -> Architecture:
    return Architecture(source_repo=source_repo, source_revision=None, kind="unknown")


def _nova_base_model(**overrides) -> BaseModelSpec:
    defaults = dict(
        hf_repo="acme/Nova-7B",
        repo_aliases=["Nova-7B-Legacy-GGUF"],
        ollama_base="nova",
        ollama_tag="7b",
        publisher="acme",
        parameters_b=7.0,
        architecture=_unknown_architecture("acme/Nova-7B"),
    )
    defaults.update(overrides)
    return BaseModelSpec(**defaults)


def _hf_package(*, repo: str, weight_filename: str, revision: str = _HF_REVISION_NEW, format_: str = "gguf") -> Package:
    # The package's own `provenance`/`unresolved_reason` fields are irrelevant scaffolding here:
    # `decide_provenance` never reads them, it recomputes a fresh verdict from `format`, `source`,
    # `revision`/`manifest_digest`, `repo` and `files`. They still have to satisfy Package's own
    # invariants (ADR: a non-gguf package must be provenance='unresolved' with reason 'format'),
    # so a gguf package gets an arbitrary valid placeholder instead.
    return Package(
        source="huggingface",
        repo=repo,
        revision=revision,
        base_model_hf_repo="acme/Nova-7B",
        format=format_,
        files=[PackageFile(name=weight_filename, role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization=None,
        default_context=None,
        provenance="unresolved" if format_ != "gguf" else "metadata_ok",
        unresolved_reason="format" if format_ != "gguf" else None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )


def _ollama_package(*, ollama_name: str, manifest_digest: str = _OLLAMA_DIGEST_NEW) -> Package:
    # Same scaffolding note as `_hf_package`: this helper only ever builds gguf-format packages,
    # so the placeholder provenance is always 'metadata_ok'/None.
    return Package(
        source="ollama",
        ollama_name=ollama_name,
        manifest_digest=manifest_digest,
        base_model_hf_repo="acme/Nova-7B",
        format="gguf",
        files=[PackageFile(name="weights", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization=None,
        default_context=None,
        provenance="metadata_ok",
        unresolved_reason=None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )


# --- real Hugging Face metadata: the Unsloth Qwen3.5-9B GGUF repo ------------------------


def _qwen_base_model() -> BaseModelSpec:
    return BaseModelSpec(
        hf_repo="Qwen/Qwen3.5-9B",
        repo_aliases=[],
        ollama_base="qwen3.5",
        ollama_tag="9b",
        publisher="Qwen",
        parameters_b=9.0,
        architecture=_unknown_architecture("Qwen/Qwen3.5-9B"),
    )


def _unsloth_gguf_model() -> dict:
    return json.loads((FIXTURES / "hf_unsloth_qwen35_9b_gguf_model.json").read_text(encoding="utf-8"))


def test_real_unsloth_repo_is_metadata_ok():
    model = _unsloth_gguf_model()
    package = Package(
        source="huggingface",
        repo=model["id"],
        revision=model["sha"],
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format="gguf",
        files=[
            PackageFile(name="Qwen3.5-9B-Q4_K_M.gguf", role="weights", size_bytes=5680522464, digest=None),
        ],
        complete=True,
        quantization="Q4_K_M",
        default_context=None,
        provenance="metadata_ok",
        unresolved_reason=None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )
    provenance, reason = decide_provenance(package, _qwen_base_model(), model["tags"], approvals=[])
    assert (provenance, reason) == ("metadata_ok", None)


def test_real_unsloth_repo_ud_quant_file_stem_is_metadata_ok():
    # UD-Q4_K_XL contains a '-' itself; the file-stem rule must still recognize it.
    model = _unsloth_gguf_model()
    package = Package(
        source="huggingface",
        repo=model["id"],
        revision=model["sha"],
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format="gguf",
        files=[
            PackageFile(name="Qwen3.5-9B-UD-Q4_K_XL.gguf", role="weights", size_bytes=5966095584, digest=None),
        ],
        complete=True,
        quantization="UD-Q4_K_XL",
        default_context=None,
        provenance="metadata_ok",
        unresolved_reason=None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )
    provenance, reason = decide_provenance(package, _qwen_base_model(), model["tags"], approvals=[])
    assert (provenance, reason) == ("metadata_ok", None)


# --- format gate takes priority over everything else -------------------------------------


def test_tensor_format_is_always_unresolved_format():
    package = _hf_package(repo="packager/Nova-7B-GGUF", weight_filename="Nova-7B-F16.gguf", format_="tensor")
    base_model = _nova_base_model()
    approval = Approval(date=date(2026, 9, 1), content=package.revision, by="acme-ai-team")
    provenance, reason = decide_provenance(package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[approval])
    assert (provenance, reason) == ("unresolved", "format")


# --- alias repo names ----------------------------------------------------------------------


def test_alias_repo_name_is_metadata_ok():
    package = _hf_package(repo="packager/Nova-7B-Legacy-GGUF", weight_filename="Nova-7B-Q4_K_M.gguf")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(
        package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[]
    )
    assert (provenance, reason) == ("metadata_ok", None)


# --- fine-tune: correct base_model tag, wrong file stem -----------------------------------


def test_correct_tag_but_wrong_file_stem_is_unresolved():
    package = _hf_package(repo="packager/Nova-7B-GGUF", weight_filename="Nova-7B-finetuned-Q4_K_M.gguf")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(
        package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[]
    )
    assert (provenance, reason) == ("unresolved", "file_stem")


def test_missing_base_model_tag_is_unresolved():
    package = _hf_package(repo="packager/Nova-7B-GGUF", weight_filename="Nova-7B-Q4_K_M.gguf")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(package, base_model, hf_tags=[], approvals=[])
    assert (provenance, reason) == ("unresolved", "base_model_tag")


def test_unrelated_repo_name_is_unresolved():
    package = _hf_package(repo="packager/SomethingElse-GGUF", weight_filename="SomethingElse-Q4_K_M.gguf")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(
        package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[]
    )
    assert (provenance, reason) == ("unresolved", "repo_name")


# --- approvals: current content wins, stale content is void -------------------------------


def test_approval_matching_current_revision_is_approved():
    package = _hf_package(repo="anyone/Anything-GGUF", weight_filename="whatever.gguf", revision=_HF_REVISION_NEW)
    base_model = _nova_base_model()
    approval = Approval(date=date(2026, 9, 1), content=_HF_REVISION_NEW, by="acme-ai-team")
    provenance, reason = decide_provenance(package, base_model, hf_tags=[], approvals=[approval])
    assert (provenance, reason) == ("approved", None)


def test_stale_approval_falls_back_to_metadata_ok_when_metadata_still_fits():
    package = _hf_package(repo="packager/Nova-7B-GGUF", weight_filename="Nova-7B-Q4_K_M.gguf", revision=_HF_REVISION_NEW)
    base_model = _nova_base_model()
    stale_approval = Approval(date=date(2026, 1, 1), content=_HF_REVISION_OLD, by="acme-ai-team")
    provenance, reason = decide_provenance(
        package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[stale_approval]
    )
    assert (provenance, reason) == ("metadata_ok", None)


def test_stale_approval_with_missing_tag_now_is_unresolved():
    package = _hf_package(repo="packager/Nova-7B-GGUF", weight_filename="Nova-7B-Q4_K_M.gguf", revision=_HF_REVISION_NEW)
    base_model = _nova_base_model()
    stale_approval = Approval(date=date(2026, 1, 1), content=_HF_REVISION_OLD, by="acme-ai-team")
    provenance, reason = decide_provenance(package, base_model, hf_tags=[], approvals=[stale_approval])
    assert (provenance, reason) == ("unresolved", "base_model_tag")


def test_ollama_approval_matching_current_digest_is_approved():
    package = _ollama_package(ollama_name="nova:7b-q4_K_M", manifest_digest=_OLLAMA_DIGEST_NEW)
    base_model = _nova_base_model()
    approval = Approval(date=date(2026, 9, 1), content=_OLLAMA_DIGEST_NEW, by="acme-ai-team")
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[approval])
    assert (provenance, reason) == ("approved", None)


def test_stale_ollama_approval_falls_back_to_metadata_rule():
    package = _ollama_package(ollama_name="nova:7b-q4_K_M", manifest_digest=_OLLAMA_DIGEST_NEW)
    base_model = _nova_base_model()
    stale_approval = Approval(date=date(2026, 1, 1), content=_OLLAMA_DIGEST_OLD, by="acme-ai-team")
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[stale_approval])
    assert (provenance, reason) == ("metadata_ok", None)


# --- Ollama library metadata rule -----------------------------------------------------------


def test_ollama_library_tag_is_metadata_ok():
    package = _ollama_package(ollama_name="nova:7b-q4_K_M")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("metadata_ok", None)


def test_ollama_wrong_base_name_is_unresolved():
    package = _ollama_package(ollama_name="other-family:7b-q4_K_M")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("unresolved", "ollama_name")


def test_ollama_wrong_size_token_is_unresolved():
    package = _ollama_package(ollama_name="nova:70b-q4_K_M")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("unresolved", "size_token")


def test_ollama_base_model_without_ollama_base_is_unresolved():
    package = _ollama_package(ollama_name="nova:7b-q4_K_M")
    base_model = _nova_base_model(ollama_base=None, ollama_tag=None)
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("unresolved", "ollama_base")


# --- Ollama tag rule: token boundaries, never a naive prefix match (finding 5) -------------


def test_ollama_tag_that_merely_starts_with_the_size_digits_is_unresolved():
    # The historic bug: "7banana" satisfied `tag.startswith("7b")`. The fix requires the first
    # '-'-separated token to equal the size token exactly.
    package = _ollama_package(ollama_name="nova:7banana")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert provenance == "unresolved"


def test_ollama_tag_with_descriptive_middle_token_still_matches_the_base_tag():
    package = _ollama_package(ollama_name="nova:7b-instruct-q4_K_M")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("metadata_ok", None)


def test_ollama_tag_with_size_match_but_divergent_base_tag_is_unresolved():
    # first_token ("7b") matches the size derived from parameters_b, but the base model's own
    # ollama_tag ("7b-preview") is not a prefix of "7b-q4_K_M" -- the tag boundary must still
    # be checked against the actual ollama_tag, not just the numeric size.
    package = _ollama_package(ollama_name="nova:7b-q4_K_M")
    base_model = _nova_base_model(ollama_tag="7b-preview")
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("unresolved", "ollama_tag")


def test_ollama_base_model_without_ollama_tag_is_unresolved():
    # BaseModelSpec itself couples ollama_base/ollama_tag (both set or both None), so this
    # exercises the defensive check in `_decide_ollama` directly with a duck-typed stand-in,
    # the way `quantization.py`'s own functions duck-type on `Package`/`PackageFile`.
    class _StubBaseModel:
        ollama_base = "nova"
        ollama_tag = None
        parameters_b = 7.0

    package = _ollama_package(ollama_name="nova:7b-q4_K_M")
    provenance, reason = decide_provenance(package, _StubBaseModel(), hf_tags=None, approvals=[])
    assert (provenance, reason) == ("unresolved", "ollama_tag")


# --- Hugging Face metadata rule requires at least one weight file (finding 3) --------------


def test_mmproj_only_package_is_unresolved_no_weights():
    # mmproj is not a weight file: a package that carries only one has no weights at all, so
    # `complete` derives to False (shards_complete ignores mmproj/other-role files).
    package = Package(
        source="huggingface",
        repo="packager/Nova-7B-GGUF",
        revision=_HF_REVISION_NEW,
        base_model_hf_repo="acme/Nova-7B",
        format="gguf",
        files=[PackageFile(name="mmproj-F16.gguf", role="mmproj", size_bytes=1, digest=None)],
        complete=False,
        quantization=None,
        default_context=None,
        provenance="metadata_ok",
        unresolved_reason=None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(
        package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[]
    )
    assert (provenance, reason) == ("unresolved", "no_weights")


# --- real Ollama registry fixtures drive decide_provenance, not just stored media types ----
# (finding 14): the format is derived from the weights-layer media type the same way a real
# fetcher would, and the resulting Package is fed through decide_provenance end to end.


def _qwen35_ollama_base_model() -> BaseModelSpec:
    return BaseModelSpec(
        hf_repo="Qwen/Qwen3.5-9B",
        repo_aliases=[],
        ollama_base="qwen3.5",
        ollama_tag="9b",
        publisher="Qwen",
        parameters_b=9.0,
        architecture=_unknown_architecture("Qwen/Qwen3.5-9B"),
    )


def _format_from_ollama_manifest(fixture_name: str) -> str:
    manifest = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    media_types = {layer["mediaType"] for layer in manifest["layers"]}
    if "application/vnd.ollama.image.model" in media_types:
        return "gguf"
    if "application/vnd.ollama.image.tensor" in media_types:
        return "tensor"
    return "unknown"


def _package_from_ollama_manifest(*, tag: str, fixture_name: str) -> Package:
    format_ = _format_from_ollama_manifest(fixture_name)
    weight_file = PackageFile(name=tag, role="weights", size_bytes=1, digest=None)
    return Package(
        source="ollama",
        ollama_name=f"qwen3.5:{tag}",
        manifest_digest="sha256:" + "0" * 64,
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format=format_,
        files=[weight_file],
        complete=True,
        quantization=None,
        default_context=None,
        provenance="unresolved" if format_ != "gguf" else "metadata_ok",
        unresolved_reason="format" if format_ != "gguf" else None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )


def test_real_ollama_mlx_tensor_build_is_unresolved_format():
    package = _package_from_ollama_manifest(tag="9b-mlx-bf16", fixture_name="ollama_qwen35_9b-mlx-bf16.json")
    provenance, reason = decide_provenance(package, _qwen35_ollama_base_model(), hf_tags=None, approvals=[])
    assert (provenance, reason) == ("unresolved", "format")


def test_real_ollama_gguf_quant_tag_is_metadata_ok():
    package = _package_from_ollama_manifest(tag="9b-q4_K_M", fixture_name="ollama_qwen35_9b-q4_K_M.json")
    provenance, reason = decide_provenance(package, _qwen35_ollama_base_model(), hf_tags=None, approvals=[])
    assert (provenance, reason) == ("metadata_ok", None)


# --- AP3 acceptance findings ----------------------------------------------------------------
# --- F1: unsloth per-quant subfolders; the file-stem prefix must ignore the folder ----------


def test_unsloth_per_quant_subfolder_file_stem_is_metadata_ok():
    # Real layout observed 2026-09-22: unsloth ships one subfolder per quantization
    # (`BF16/`, `UD-Q2_K_XL/`, ...); the folder segment must never defeat the base-name prefix
    # match on the file's own basename.
    package = _hf_package(repo="packager/Nova-7B-GGUF", weight_filename="BF16/Nova-7B-BF16.gguf")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(
        package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[]
    )
    assert (provenance, reason) == ("metadata_ok", None)


def test_draft_model_in_a_foreign_subfolder_stays_unresolved_file_stem():
    # Real observed layout: a packager repo can carry a draft/MTP model dropped under its own
    # subfolder (e.g. `MTP/mtp-gemma-4-31B-it-BF16.gguf`); it must never be mistaken for the
    # base model just because it landed inside the repo's tree.
    package = _hf_package(repo="packager/Nova-7B-GGUF", weight_filename="MTP/mtp-Nova-7B-BF16.gguf")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(
        package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[]
    )
    assert (provenance, reason) == ("unresolved", "file_stem")


# --- F2: mradermacher's `<base>.<QUANT>.gguf` (dot separator) convention -------------------


def test_mradermacher_dot_convention_file_stem_is_metadata_ok():
    package = _hf_package(repo="packager/Nova-7B-GGUF", weight_filename="Nova-7B.Q4_K_M.gguf")
    base_model = _nova_base_model()
    provenance, reason = decide_provenance(
        package, base_model, hf_tags=["base_model:acme/Nova-7B"], approvals=[]
    )
    assert (provenance, reason) == ("metadata_ok", None)


# --- F3: Qwen's own "-split-NNNNN-of-NNNNN" shard suffix; a real, documented mismatch -------


def test_qwen_split_shard_stem_mismatch_stays_unresolved_file_stem():
    # Real observed name (2026-09-22): the packager's own shard filename spells the base
    # "Qwen3VL-..." (no dash before VL) while the base repo is "Qwen3-VL-235B-A22B-Instruct" --
    # a genuine mismatch under the file_stem rule, not a bug to work around. Only 1 of 10
    # shards is modelled here, so `complete` is `False` (shards_complete's own concern, not
    # this test's), built directly rather than through `_hf_package` (a single-file helper).
    base_model = _nova_base_model(
        hf_repo="Qwen/Qwen3-VL-235B-A22B-Instruct", repo_aliases=[], parameters_b=235.0
    )
    package = Package(
        source="huggingface",
        repo="packager/Qwen3-VL-235B-A22B-Instruct-GGUF",
        revision=_HF_REVISION_NEW,
        base_model_hf_repo="Qwen/Qwen3-VL-235B-A22B-Instruct",
        format="gguf",
        files=[
            PackageFile(
                name="Qwen3VL-235B-A22B-Instruct-F16-split-00001-of-00010.gguf",
                role="weights_shard",
                size_bytes=1,
                digest=None,
            )
        ],
        complete=False,
        quantization="F16",
        default_context=None,
        provenance="unresolved",
        unresolved_reason="file_stem",
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )
    provenance, reason = decide_provenance(
        package,
        base_model,
        hf_tags=["base_model:Qwen/Qwen3-VL-235B-A22B-Instruct"],
        approvals=[],
    )
    assert (provenance, reason) == ("unresolved", "file_stem")


# --- F6: Ollama size token accepted within 15% of the measured parameters_b ----------------


def test_ollama_size_token_within_tolerance_of_measured_parameters_is_metadata_ok():
    # Real measurement 2026-09-22: Qwen/Qwen3.5-9B's safetensors.total gives
    # parameters_b = 9.653104368 (it counts embeddings), but the library tag is "9b" --
    # exact equality can never match this, only a tolerance can.
    package = _ollama_package(ollama_name="nova:9b-q4_K_M")
    base_model = _nova_base_model(parameters_b=9.653104368, ollama_tag="9b")
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("metadata_ok", None)


def test_ollama_size_token_30b_matches_measured_30_5b():
    package = _ollama_package(ollama_name="nova:30b-q4_K_M")
    base_model = _nova_base_model(parameters_b=30.5, ollama_tag="30b")
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("metadata_ok", None)


def test_ollama_size_token_1_5b_matches_measured_1_54b():
    package = _ollama_package(ollama_name="nova:1.5b-q4_K_M")
    base_model = _nova_base_model(parameters_b=1.54, ollama_tag="1.5b")
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("metadata_ok", None)


def test_ollama_base_model_with_no_measured_parameters_is_unresolved_parameters_unknown():
    # F11: parameters_b is None (never measured this run, and no previous reading exists) --
    # the tag can never be size-checked, so this is its own reason, not "size_token".
    package = _ollama_package(ollama_name="nova:7b-q4_K_M")
    base_model = _nova_base_model(parameters_b=None)
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("unresolved", "parameters_unknown")


def test_ollama_size_token_declared_size_far_outside_tolerance_is_unresolved():
    # The tag itself matches the base model's own declared `ollama_tag` ("70b"), so the
    # tag-boundary check alone would pass -- the *measured* parameters_b must still gate it.
    package = _ollama_package(ollama_name="nova:70b-q4_K_M")
    base_model = _nova_base_model(parameters_b=7.0, ollama_tag="70b")
    provenance, reason = decide_provenance(package, base_model, hf_tags=None, approvals=[])
    assert (provenance, reason) == ("unresolved", "size_token")
