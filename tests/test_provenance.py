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
        provenance="unresolved",
        unresolved_reason="format" if format_ != "gguf" else None,
        approval=None,
        observed_at=_NOW,
        last_seen=_NOW,
        active=True,
    )


def _ollama_package(*, ollama_name: str, manifest_digest: str = _OLLAMA_DIGEST_NEW) -> Package:
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
        provenance="unresolved",
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
        provenance="unresolved",
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
        provenance="unresolved",
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
