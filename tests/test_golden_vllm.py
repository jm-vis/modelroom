"""The rendered document of a machine with vLLM, held against golden files (AP9-SV2-F).

Every other golden file of this repository (`tests/golden/guided-screen*.txt`) is a reasoning
machine without vLLM, so none of them changes a byte for this sub-package (CONTRACTS.md, "Fit for
vLLM"). This machine -- four equal 22.49 GiB L4 cards, `profiles_v4/four_l4_server_vllm.json` --
is new, and only here: one GGUF row and one Hugging Face tensor row of the very same base model,
so the Markdown table, the terminal view and the JSON document all show the runtime prefix, the
`runtime` row, the legend's extra line and the `memory` line's own vLLM piece together.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.config import Configuration
from modelroom.contracts import Architecture, BaseModelSpec, Package, PackageFile, Snapshot
from modelroom.measurements import Scenario
from modelroom.profile import HardwareProfile
from modelroom.render import MachineProfile, build_render_document, document_json
from modelroom.views import document_markdown, document_terminal

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"
SNAPSHOT_RUN_AT = datetime(2026, 9, 30, 8, 0, 0, tzinfo=timezone.utc)
RENDERED_AT = datetime(2026, 9, 30, 8, 5, 0, tzinfo=timezone.utc)
BASE_REPO = "Qwen/Qwen3-8B"


def _profile() -> HardwareProfile:
    data = json.loads((FIXTURES / "profiles_v4" / "four_l4_server_vllm.json").read_text(encoding="utf-8"))
    return HardwareProfile.model_validate(data)


def _architecture() -> Architecture:
    # The brief's own rechenbeispiel (F2): 36 layers, 8 KV heads, head_dim 128, 32 attention heads.
    return Architecture(
        source_repo=BASE_REPO, source_revision="a" * 40, kind="dense_classic",
        num_hidden_layers=36, num_key_value_heads=8, head_dim=128, num_attention_heads=32, max_context=131072,
    )


def _base_model() -> BaseModelSpec:
    return BaseModelSpec(hf_repo=BASE_REPO, repo_aliases=[], ollama_base=None, ollama_tag=None, publisher="Qwen", parameters_b=8.0, architecture=_architecture())


def _gguf_package() -> Package:
    return Package(
        source="huggingface", repo="unsloth/Qwen3-8B-GGUF", revision="b" * 40,
        base_model_hf_repo=BASE_REPO, format="gguf",
        files=[PackageFile(name="Qwen3-8B-Q4_K_M.gguf", role="weights", size_bytes=5_027_785_216, digest="sha256:" + "1" * 64)],
        complete=True, quantization="Q4_K_M", quantization_source="filename", default_context=None,
        provenance="from_metadata", unresolved_reason=None, approval=None,
        observed_at=SNAPSHOT_RUN_AT, last_seen=SNAPSHOT_RUN_AT, active=True,
    )


def _tensor_package() -> Package:
    # The brief's own rechenbeispiel (F2): 6,098,581,864 bytes, AWQ.
    return Package(
        source="huggingface", repo="RedHatAI/Qwen3-8B-quantized.w4a16", revision="c" * 40,
        base_model_hf_repo=BASE_REPO, format="tensor",
        files=[PackageFile(name="model.safetensors", role="weights", size_bytes=6_098_581_864, digest=None)],
        complete=True, quantization="AWQ", quantization_source="config", default_context=None,
        provenance="from_metadata", unresolved_reason=None, approval=None,
        observed_at=SNAPSHOT_RUN_AT, last_seen=SNAPSHOT_RUN_AT, active=True,
    )


def _config() -> Configuration:
    return Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [{"name": "qwen3", "base_models": [{"hf_repo": BASE_REPO}]}],
            "packagers": ["unsloth", "RedHatAI"],
            "publishers": ["Qwen"],
            "machines": {
                "server": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True, "profile": _profile().profile_id}
            },
            "paths": {"state": str(FIXTURES / "state"), "markdown": str(FIXTURES / "docs" / "models.md")},
        }
    )


def _snapshot() -> Snapshot:
    from modelroom.contracts import Area

    return Snapshot(
        schema_version=2,
        run_at=SNAPSHOT_RUN_AT,
        areas=[
            Area(source="huggingface", base_model_hf_repo=BASE_REPO, packager="unsloth", status="complete", last_success=SNAPSHOT_RUN_AT, error=None),
            Area(source="huggingface", base_model_hf_repo=BASE_REPO, packager="RedHatAI", status="complete", last_success=SNAPSHOT_RUN_AT, error=None),
        ],
        base_models=[_base_model()],
        packages=[_gguf_package(), _tensor_package()],
    )


def _document():
    config = _config()
    profiles = {"server": MachineProfile(status="ranked", label=_profile().display_name, profile=_profile(), reason=None)}
    scenario = Scenario(context_requested=32768, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=10)
    return build_render_document(config, _snapshot(), profiles, {}, scenario, RENDERED_AT)


def _compare(name: str, text: str) -> None:
    golden = GOLDEN / name
    if not golden.is_file():  # pragma: no cover - only while a new golden file is written
        pytest.fail(f"{golden} does not exist; write it from the run and read it before committing")
    assert text == golden.read_text(encoding="utf-8")


def test_the_markdown_of_a_machine_with_vllm_reads_as_the_golden_file():
    _compare("golden_vllm_four_l4.md", document_markdown(_document()))


def test_the_terminal_view_of_a_machine_with_vllm_reads_as_the_golden_file():
    _compare("golden_vllm_four_l4.txt", document_terminal(_document()) + "\n")


def test_the_json_of_a_machine_with_vllm_reads_as_the_golden_file():
    text = json.dumps(document_json(_document()), indent=2, sort_keys=True) + "\n"
    _compare("golden_vllm_four_l4.json", text)


def test_the_tensor_row_carries_the_runtime_and_the_gguf_row_does_not():
    """The one fact every other test here only shows through formatted text: `Fit.runtime` itself,
    on the ranked rows of the real document (not a hand-built one)."""
    document = _document()
    runtimes = {entry.fit.runtime for entry in document.machines[0].ranked}

    assert runtimes == {"ollama", "vllm"}
    assert document.machines[0].ranked_total == 2
