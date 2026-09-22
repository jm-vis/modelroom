"""Tests for modelroom.contracts: the Pydantic models, EXAMPLES, and CONTRACTS.md in step."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.contracts import (
    EXAMPLES,
    SNAPSHOT_SCHEMA_RANGE,
    SNAPSHOT_SCHEMA_VERSION,
    Approval,
    Architecture,
    Area,
    BaseModelSpec,
    Family,
    Package,
    PackageFile,
    SchemaVersionError,
    Snapshot,
    architecture_from_hf_config,
    check_schema_version,
    shards_complete,
)

REPO = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"

MODEL_CLASSES = {
    "Architecture": Architecture,
    "BaseModelSpec": BaseModelSpec,
    "Family": Family,
    "PackageFile": PackageFile,
    "Approval": Approval,
    "Package": Package,
    "Area": Area,
    "Snapshot": Snapshot,
}

_NOW = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)


# --- EXAMPLES validate, and cover every model, and only every model -----------------------


@pytest.mark.parametrize("name, model_cls", MODEL_CLASSES.items())
def test_example_validates_against_its_model(name, model_cls):
    assert name in EXAMPLES, f"no EXAMPLES entry for {name}"
    model_cls.model_validate(EXAMPLES[name])


def test_examples_has_no_entries_for_unknown_models():
    assert set(EXAMPLES) == set(MODEL_CLASSES)


# --- CONTRACTS.md documents every model and repeats its example verbatim ------------------


def _contracts_md_examples() -> dict[str, dict]:
    text = (REPO / "CONTRACTS.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^### (\w+)\s*$", text, flags=re.MULTILINE))
    found: dict[str, dict] = {}
    for i, heading in enumerate(headings):
        name = heading.group(1)
        start = heading.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[start:end]
        code_block = re.search(r"```json\n(.*?)\n```", section, flags=re.DOTALL)
        assert code_block, f"### {name} has no ```json example block"
        found[name] = json.loads(code_block.group(1))
    return found


def test_contracts_md_names_every_model_and_no_extra_ones():
    documented = _contracts_md_examples()
    assert set(documented) == set(MODEL_CLASSES)


def test_contracts_md_examples_match_examples_module_verbatim():
    documented = _contracts_md_examples()
    for name, example in EXAMPLES.items():
        assert documented[name] == example, f"CONTRACTS.md example for {name} drifted from EXAMPLES"


# --- extra=forbid and required fields ------------------------------------------------------


def test_extra_field_is_rejected():
    payload = dict(EXAMPLES["Family"])
    payload["unexpected"] = "nope"
    with pytest.raises(ValidationError):
        Family.model_validate(payload)


def test_missing_required_field_is_rejected():
    payload = dict(EXAMPLES["BaseModelSpec"])
    del payload["publisher"]
    with pytest.raises(ValidationError):
        BaseModelSpec.model_validate(payload)


def test_family_requires_at_least_one_base_model():
    payload = dict(EXAMPLES["Family"])
    payload["base_models"] = []
    with pytest.raises(ValidationError):
        Family.model_validate(payload)


# --- Package: source-coupled fields --------------------------------------------------------


def test_huggingface_package_without_revision_is_rejected():
    payload = dict(EXAMPLES["Package"])
    del payload["revision"]
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_ollama_package_with_repo_is_rejected():
    payload = dict(EXAMPLES["Package"])
    payload["source"] = "ollama"
    payload["ollama_name"] = "nova:7b-q4_K_M"
    payload["manifest_digest"] = "sha256:" + "a" * 64
    # repo/revision are still set from the huggingface example -> must be rejected
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_huggingface_package_with_ollama_fields_is_rejected():
    payload = dict(EXAMPLES["Package"])
    payload["ollama_name"] = "nova:7b-q4_K_M"
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_valid_ollama_package_is_accepted():
    payload = dict(EXAMPLES["Package"])
    payload["source"] = "ollama"
    del payload["repo"]
    del payload["revision"]
    payload["ollama_name"] = "nova:7b-q4_K_M"
    payload["manifest_digest"] = "sha256:" + "a" * 64
    Package.model_validate(payload)


def test_provenance_approved_requires_an_approval():
    payload = dict(EXAMPLES["Package"])
    payload["provenance"] = "approved"
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


# --- BaseModelSpec: ollama_base/ollama_tag both-or-neither ---------------------------------


def test_ollama_base_without_tag_is_rejected():
    payload = dict(EXAMPLES["BaseModelSpec"])
    payload["ollama_tag"] = None
    with pytest.raises(ValidationError):
        BaseModelSpec.model_validate(payload)


def test_ollama_tag_without_base_is_rejected():
    payload = dict(EXAMPLES["BaseModelSpec"])
    payload["ollama_base"] = None
    with pytest.raises(ValidationError):
        BaseModelSpec.model_validate(payload)


def test_neither_ollama_base_nor_tag_is_accepted():
    payload = dict(EXAMPLES["BaseModelSpec"])
    payload["ollama_base"] = None
    payload["ollama_tag"] = None
    BaseModelSpec.model_validate(payload)


# --- Architecture: dense_classic completeness ----------------------------------------------


def test_dense_classic_without_numeric_fields_is_rejected():
    with pytest.raises(ValidationError):
        Architecture(source_repo="acme/Nova-7B", kind="dense_classic")


def test_dense_classic_with_hybrid_layer_types_is_rejected():
    with pytest.raises(ValidationError):
        Architecture(
            source_repo="acme/Nova-7B",
            kind="dense_classic",
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            layer_types=["full_attention", "linear_attention"],
        )


def test_unknown_kind_allows_no_numeric_fields():
    Architecture(source_repo="acme/Nova-7B", kind="unknown")


# --- shards_complete -------------------------------------------------------------------------


def _weights_file(name: str) -> PackageFile:
    return PackageFile(name=name, role="weights_shard", size_bytes=1, digest=None)


def test_shards_complete_all_present():
    files = [_weights_file(f"Nova-70B-Q4_K_M-{i:05d}-of-00003.gguf") for i in (1, 2, 3)]
    assert shards_complete(files) is True


def test_shards_complete_missing_one():
    files = [_weights_file(f"Nova-70B-Q4_K_M-{i:05d}-of-00003.gguf") for i in (1, 2)]
    assert shards_complete(files) is False


def test_shards_complete_single_non_sharded_file():
    files = [PackageFile(name="Nova-7B-Q4_K_M.gguf", role="weights", size_bytes=1, digest=None)]
    assert shards_complete(files) is True


def test_shards_complete_no_weight_files_is_false():
    files = [PackageFile(name="mmproj-F16.gguf", role="mmproj", size_bytes=1, digest=None)]
    assert shards_complete(files) is False


# --- architecture_from_hf_config --------------------------------------------------------------


def test_architecture_from_missing_config_is_unknown():
    architecture = architecture_from_hf_config("acme/Nova-7B", None, None)
    assert architecture.kind == "unknown"
    assert architecture.num_hidden_layers is None


def test_architecture_from_real_qwen_hybrid_config_is_unknown():
    config = json.loads((FIXTURES / "hf_qwen_qwen35_9b_config.json").read_text(encoding="utf-8"))
    model = json.loads((FIXTURES / "hf_qwen_qwen35_9b_model.json").read_text(encoding="utf-8"))
    architecture = architecture_from_hf_config("Qwen/Qwen3.5-9B", model["sha"], config)
    assert architecture.kind == "unknown"


def test_architecture_from_synthetic_dense_config_is_dense_classic():
    config = json.loads((FIXTURES / "hf_synthetic_dense_config.json").read_text(encoding="utf-8"))
    architecture = architecture_from_hf_config(
        "acme/Nova-7B", "1a2b3c4d5e6f7890abcdef1234567890abcdef12", config
    )
    assert architecture.kind == "dense_classic"
    assert architecture.num_hidden_layers == 32
    assert architecture.num_key_value_heads == 8
    assert architecture.head_dim == 128
    assert architecture.max_context == 32768


# --- Snapshot: cross-references and identity uniqueness -------------------------------------


def test_snapshot_rejects_a_package_with_an_unknown_base_model():
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    payload["packages"][0]["base_model_hf_repo"] = "someone/else"
    with pytest.raises(ValidationError):
        Snapshot.model_validate(payload)


def test_snapshot_rejects_duplicate_package_identity():
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    payload["packages"].append(json.loads(json.dumps(payload["packages"][0])))
    with pytest.raises(ValidationError):
        Snapshot.model_validate(payload)


def test_snapshot_accepts_two_packages_with_different_quantizations():
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    second = json.loads(json.dumps(payload["packages"][0]))
    second["quantization"] = "Q8_0"
    second["files"][0]["name"] = "Nova-7B-Q8_0.gguf"
    payload["packages"].append(second)
    Snapshot.model_validate(payload)


# --- schema version ---------------------------------------------------------------------------


def test_check_schema_version_accepts_current_version():
    check_schema_version(SNAPSHOT_SCHEMA_VERSION, SNAPSHOT_SCHEMA_RANGE, "snapshot")


def test_check_schema_version_rejects_out_of_range_version():
    with pytest.raises(SchemaVersionError) as excinfo:
        check_schema_version(2, (1, 2), "snapshot")
    message = str(excinfo.value)
    assert "snapshot" in message
    assert "2" in message
    assert "[1, 2)" in message


def test_snapshot_model_itself_rejects_wrong_schema_version():
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        Snapshot.model_validate(payload)
