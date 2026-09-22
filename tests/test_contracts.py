"""Tests for modelroom.contracts: the Pydantic models, EXAMPLES, and CONTRACTS.md in step."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.examples import EXAMPLES
from modelroom.contracts import (
    HARDWARE_SCHEMA_RANGE,
    HARDWARE_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_RANGE,
    SNAPSHOT_SCHEMA_VERSION,
    Approval,
    Architecture,
    Area,
    BaseModelSpec,
    Family,
    Fit,
    HardwareSnapshot,
    InstalledModel,
    Measurement,
    Package,
    PackageFile,
    Rating,
    SchemaVersionError,
    Snapshot,
    architecture_from_hf_config,
    check_schema_version,
    load_hardware_snapshot,
    load_snapshot,
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
    "InstalledModel": InstalledModel,
    "Measurement": Measurement,
    "HardwareSnapshot": HardwareSnapshot,
    "Fit": Fit,
    "Rating": Rating,
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


# --- BaseModelSpec.parameters_b: None is "not measured", never a fabricated number (F11) ----


def test_parameters_b_none_is_accepted():
    payload = dict(EXAMPLES["BaseModelSpec"])
    payload["parameters_b"] = None
    BaseModelSpec.model_validate(payload)


def test_parameters_b_zero_is_still_rejected():
    payload = dict(EXAMPLES["BaseModelSpec"])
    payload["parameters_b"] = 0.0
    with pytest.raises(ValidationError):
        BaseModelSpec.model_validate(payload)


def test_parameters_b_negative_is_still_rejected():
    payload = dict(EXAMPLES["BaseModelSpec"])
    payload["parameters_b"] = -1.0
    with pytest.raises(ValidationError):
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


# --- Package.complete is derived from files, never accepted freely (finding 1) ------------


def test_complete_true_with_no_files_is_rejected():
    payload = dict(EXAMPLES["Package"])
    payload["files"] = []
    payload["complete"] = True
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_complete_false_with_a_single_complete_weight_file_is_rejected():
    payload = dict(EXAMPLES["Package"])
    payload["complete"] = False  # files already hold one non-sharded weight file -> derived True
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_complete_matching_the_derived_value_is_accepted():
    payload = dict(EXAMPLES["Package"])
    payload["complete"] = True
    Package.model_validate(payload)


# --- shards_complete: stems, duplicate indices (finding 2) ---------------------------------


def test_shards_complete_mixed_stems_is_incomplete():
    files = [
        _weights_file("Nova-7B-Q4_K_M-00001-of-00002.gguf"),
        _weights_file("Nova-7B-Q8_0-00002-of-00002.gguf"),
    ]
    assert shards_complete(files) is False


def test_shards_complete_duplicate_index_is_incomplete():
    files = [
        _weights_file("Nova-70B-Q4_K_M-00001-of-00002.gguf"),
        _weights_file("Nova-70B-Q4_K_M-00001-of-00002.gguf"),
    ]
    assert shards_complete(files) is False


def test_shards_complete_duplicate_index_among_three_files_is_incomplete():
    # Converting indices straight to a set loses the duplicate: {1, 1, 2} -> {1, 2}, which
    # equals range(1, 3) even though the actual file count/index pairing is wrong.
    files = [
        _weights_file("Nova-70B-Q4_K_M-00001-of-00002.gguf"),
        _weights_file("Nova-70B-Q4_K_M-00001-of-00002.gguf"),
        _weights_file("Nova-70B-Q4_K_M-00002-of-00002.gguf"),
    ]
    assert shards_complete(files) is False


def test_shards_complete_mixed_sharded_and_non_sharded_is_incomplete():
    files = [
        _weights_file("Nova-70B-Q4_K_M-00001-of-00002.gguf"),
        PackageFile(name="Nova-70B-Q4_K_M.gguf", role="weights", size_bytes=1, digest=None),
    ]
    assert shards_complete(files) is False


# --- shards_complete: Qwen's own "-split-NNNNN-of-NNNNN" shard suffix (AP3 acceptance, F3) ---


def test_shards_complete_accepts_the_qwen_split_shard_suffix():
    files = [
        _weights_file("Qwen3VL-235B-A22B-Instruct-F16-split-00001-of-00002.gguf"),
        _weights_file("Qwen3VL-235B-A22B-Instruct-F16-split-00002-of-00002.gguf"),
    ]
    assert shards_complete(files) is True


def test_shards_complete_qwen_split_shard_missing_one_is_incomplete():
    files = [_weights_file("Qwen3VL-235B-A22B-Instruct-F16-split-00001-of-00010.gguf")]
    assert shards_complete(files) is False


# --- Package provenance invariants (finding 4) ----------------------------------------------


def test_non_gguf_package_must_be_unresolved_format():
    payload = dict(EXAMPLES["Package"])
    payload["format"] = "tensor"
    # provenance/unresolved_reason still say metadata_ok/None from the gguf example -> rejected
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_non_gguf_package_with_unresolved_format_is_accepted():
    payload = dict(EXAMPLES["Package"])
    payload["format"] = "tensor"
    payload["provenance"] = "unresolved"
    payload["unresolved_reason"] = "format"
    Package.model_validate(payload)


def test_unresolved_provenance_requires_a_reason():
    payload = dict(EXAMPLES["Package"])
    payload["provenance"] = "unresolved"
    payload["unresolved_reason"] = None
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_approved_package_with_mismatched_approval_content_is_rejected():
    payload = dict(EXAMPLES["Package"])
    payload["provenance"] = "approved"
    payload["approval"] = {"date": "2026-09-01", "content": "a" * 40, "by": "acme-ai-team"}
    # approval.content ("a"*40) does not match the package's revision -> rejected
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_approved_package_with_matching_approval_content_is_accepted():
    payload = dict(EXAMPLES["Package"])
    payload["provenance"] = "approved"
    payload["approval"] = {"date": "2026-09-01", "content": payload["revision"], "by": "acme-ai-team"}
    Package.model_validate(payload)


# --- documented string formats (finding 9) --------------------------------------------------


def test_repo_alias_empty_string_is_rejected():
    payload = dict(EXAMPLES["BaseModelSpec"])
    payload["repo_aliases"] = [""]
    with pytest.raises(ValidationError):
        BaseModelSpec.model_validate(payload)


def test_approval_content_must_be_a_sha_or_sha256_digest():
    payload = dict(EXAMPLES["Approval"])
    payload["content"] = "not-a-digest"
    with pytest.raises(ValidationError):
        Approval.model_validate(payload)


def test_approval_content_accepts_sha256_prefixed_digest():
    payload = dict(EXAMPLES["Approval"])
    payload["content"] = "sha256:" + "a" * 64
    Approval.model_validate(payload)


def test_package_quantization_must_be_a_quant_order_member():
    payload = dict(EXAMPLES["Package"])
    payload["quantization"] = "not-a-real-quant"
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_package_quantization_none_is_accepted():
    payload = dict(EXAMPLES["Package"])
    payload["quantization"] = None
    Package.model_validate(payload)


def test_package_ollama_name_must_match_base_colon_tag_shape():
    payload = dict(EXAMPLES["Package"])
    payload["source"] = "ollama"
    del payload["repo"]
    del payload["revision"]
    payload["ollama_name"] = "not valid"
    payload["manifest_digest"] = "sha256:" + "a" * 64
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_package_ollama_name_valid_shape_is_accepted():
    payload = dict(EXAMPLES["Package"])
    payload["source"] = "ollama"
    del payload["repo"]
    del payload["revision"]
    payload["ollama_name"] = "nova:7b-q4_K_M"
    payload["manifest_digest"] = "sha256:" + "a" * 64
    Package.model_validate(payload)


# --- re.fullmatch, never re.match, for every regex check (finding 10) ----------------------


def test_hf_repo_with_trailing_newline_is_rejected():
    payload = dict(EXAMPLES["BaseModelSpec"])
    payload["hf_repo"] = payload["hf_repo"] + "\n"
    with pytest.raises(ValidationError):
        BaseModelSpec.model_validate(payload)


def test_source_revision_with_trailing_newline_is_rejected():
    with pytest.raises(ValidationError):
        Architecture(
            source_repo="acme/Nova-7B",
            source_revision="1a2b3c4d5e6f7890abcdef1234567890abcdef12\n",
            kind="unknown",
        )


def test_package_revision_with_trailing_newline_is_rejected():
    payload = dict(EXAMPLES["Package"])
    payload["revision"] = payload["revision"] + "\n"
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


def test_package_manifest_digest_with_trailing_newline_is_rejected():
    payload = dict(EXAMPLES["Package"])
    payload["source"] = "ollama"
    del payload["repo"]
    del payload["revision"]
    payload["ollama_name"] = "nova:7b-q4_K_M"
    payload["manifest_digest"] = "sha256:" + "a" * 64 + "\n"
    with pytest.raises(ValidationError):
        Package.model_validate(payload)


# --- load_snapshot: schema_version checked before Pydantic field validation (finding 11) ---


def test_load_snapshot_rejects_out_of_range_version_with_schema_version_error():
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    payload["schema_version"] = 2
    with pytest.raises(SchemaVersionError):
        load_snapshot(payload)


def test_load_snapshot_rejects_missing_schema_version_with_schema_version_error():
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    del payload["schema_version"]
    with pytest.raises(SchemaVersionError):
        load_snapshot(payload)


def test_load_snapshot_rejects_non_integer_schema_version_with_schema_version_error():
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    payload["schema_version"] = "1"
    with pytest.raises(SchemaVersionError):
        load_snapshot(payload)


def test_load_snapshot_accepts_a_valid_snapshot():
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    snapshot = load_snapshot(payload)
    assert isinstance(snapshot, Snapshot)
    assert snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION


def test_check_schema_version_message_states_the_half_open_range_explicitly():
    with pytest.raises(SchemaVersionError) as excinfo:
        check_schema_version(2, (1, 2), "snapshot")
    assert "accepted: >= 1 and < 2" in str(excinfo.value)


# --- architecture_from_hf_config: MoE configs are never dense_classic (finding 12) ---------


def test_moe_config_with_all_dense_numeric_fields_is_still_unknown():
    config = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 1408,
    }
    architecture = architecture_from_hf_config("acme/Nova-8x7B", None, config)
    assert architecture.kind == "unknown"
    assert architecture.num_hidden_layers is None


def test_moe_field_under_text_config_is_also_detected():
    config = {
        "text_config": {
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_experts_per_tok": 2,
        }
    }
    architecture = architecture_from_hf_config("acme/Nova-8x7B", None, config)
    assert architecture.kind == "unknown"


def test_any_moe_field_present_signals_moe_even_with_a_count_of_one():
    # A dense decoder does not carry expert fields at all, so their presence is the signal;
    # a routed expert count of 1 is still a MoE layout and must not pass as dense_classic.
    config = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_experts_per_tok": 1,
    }
    architecture = architecture_from_hf_config("acme/Nova-7B", None, config)
    assert architecture.kind == "unknown"


def test_dense_config_without_any_moe_field_stays_dense():
    config = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
    assert architecture_from_hf_config("acme/Nova-7B", None, config).kind == "dense_classic"


# --- InstalledModel: digest format, aware-UTC observed_at ---------------------------------


def test_installed_model_rejects_a_bare_hex_digest_without_sha256_prefix():
    payload = dict(EXAMPLES["InstalledModel"])
    payload["digest"] = "ab" * 32  # the Ollama daemon's own /api/tags shape -- must be rejected raw
    with pytest.raises(ValidationError):
        InstalledModel.model_validate(payload)


def test_installed_model_rejects_a_naive_observed_at():
    payload = dict(EXAMPLES["InstalledModel"])
    payload["observed_at"] = datetime(2026, 9, 22, 9, 0, 0)  # no tzinfo
    with pytest.raises(ValidationError):
        InstalledModel.model_validate(payload)


# --- Measurement: content_source-coupled fields, tps_range --------------------------------


def test_measurement_ollama_content_rejects_a_stray_hf_field():
    payload = dict(EXAMPLES["Measurement"])
    payload["hf_repo"] = "acme/Nova-7B"
    with pytest.raises(ValidationError):
        Measurement.model_validate(payload)


def test_measurement_huggingface_content_requires_all_three_hf_fields():
    payload = dict(EXAMPLES["Measurement"])
    payload["content_source"] = "huggingface"
    payload["ollama_manifest_digest"] = None
    payload["hf_repo"] = "acme/Nova-7B"
    payload["hf_revision"] = "1a2b3c4d5e6f7890abcdef1234567890abcdef12"
    # hf_file_digest deliberately left at None
    with pytest.raises(ValidationError):
        Measurement.model_validate(payload)


def test_measurement_huggingface_content_validates_with_all_three_hf_fields():
    payload = dict(EXAMPLES["Measurement"])
    payload["content_source"] = "huggingface"
    payload["ollama_manifest_digest"] = None
    payload["hf_repo"] = "acme/Nova-7B"
    payload["hf_revision"] = "1a2b3c4d5e6f7890abcdef1234567890abcdef12"
    payload["hf_file_digest"] = "sha256:" + "cd" * 32
    Measurement.model_validate(payload)


def test_measurement_rejects_a_tps_range_with_low_above_high():
    payload = dict(EXAMPLES["Measurement"])
    payload["tps_range"] = [45.0, 40.0]
    with pytest.raises(ValidationError):
        Measurement.model_validate(payload)


def test_measurement_rejects_a_naive_measured_at():
    payload = dict(EXAMPLES["Measurement"])
    payload["measured_at"] = "2026-09-22T09:05:00"  # no offset
    with pytest.raises(ValidationError):
        Measurement.model_validate(payload)


# --- HardwareSnapshot: schema version, machine name reuse, installed/reason XOR -----------


def test_hardware_snapshot_rejects_unsupported_schema_version():
    payload = dict(EXAMPLES["HardwareSnapshot"])
    payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        HardwareSnapshot.model_validate(payload)


def test_hardware_snapshot_rejects_a_bad_machine_name():
    payload = dict(EXAMPLES["HardwareSnapshot"])
    payload["machine"] = "Bad Name"
    with pytest.raises(ValidationError):
        HardwareSnapshot.model_validate(payload)


def test_hardware_snapshot_rejects_installed_and_reason_both_set():
    payload = dict(EXAMPLES["HardwareSnapshot"])
    payload["installed_unavailable_reason"] = "daemon unreachable"
    with pytest.raises(ValidationError):
        HardwareSnapshot.model_validate(payload)


def test_hardware_snapshot_rejects_neither_installed_nor_reason():
    payload = dict(EXAMPLES["HardwareSnapshot"])
    payload["installed"] = None
    payload["installed_unavailable_reason"] = None
    with pytest.raises(ValidationError):
        HardwareSnapshot.model_validate(payload)


def test_hardware_snapshot_accepts_installed_none_with_a_reason():
    payload = dict(EXAMPLES["HardwareSnapshot"])
    payload["installed"] = None
    payload["installed_unavailable_reason"] = "daemon unreachable"
    HardwareSnapshot.model_validate(payload)


def test_hardware_snapshot_rejects_a_naive_measured_at():
    payload = dict(EXAMPLES["HardwareSnapshot"])
    payload["measured_at"] = "2026-09-22T09:00:00"  # no offset
    with pytest.raises(ValidationError):
        HardwareSnapshot.model_validate(payload)


def test_load_hardware_snapshot_rejects_unsupported_schema_version():
    payload = dict(EXAMPLES["HardwareSnapshot"])
    payload["schema_version"] = 99
    with pytest.raises(SchemaVersionError):
        load_hardware_snapshot(payload)


def test_load_hardware_snapshot_accepts_the_example():
    load_hardware_snapshot(EXAMPLES["HardwareSnapshot"])


# --- Fit: extra=forbid, no schema_version (never persisted) --------------------------------


def test_fit_example_has_no_schema_version_field():
    assert "schema_version" not in EXAMPLES["Fit"]


def test_fit_rejects_an_extra_field():
    payload = dict(EXAMPLES["Fit"])
    payload["extra_field"] = "nope"
    with pytest.raises(ValidationError):
        Fit.model_validate(payload)


# --- Rating: stars range/step, source non-empty, extra=forbid ------------------------------


def test_rating_accepts_a_whole_and_a_half_star_value():
    Rating.model_validate({"stars": 5.0, "source": "market index"})
    Rating.model_validate({"stars": 0.5, "source": "market index"})


def test_rating_rejects_a_non_half_step_value():
    with pytest.raises(ValidationError):
        Rating.model_validate({"stars": 3.3, "source": "market index"})


def test_rating_rejects_a_value_below_the_minimum():
    with pytest.raises(ValidationError):
        Rating.model_validate({"stars": 0.0, "source": "market index"})


def test_rating_rejects_a_value_above_the_maximum():
    with pytest.raises(ValidationError):
        Rating.model_validate({"stars": 5.5, "source": "market index"})


def test_rating_rejects_an_empty_source():
    with pytest.raises(ValidationError):
        Rating.model_validate({"stars": 3.0, "source": ""})


def test_rating_rejects_an_extra_field():
    payload = dict(EXAMPLES["Rating"])
    payload["extra_field"] = "nope"
    with pytest.raises(ValidationError):
        Rating.model_validate(payload)
