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


# --- Fix r1 (Codex r1 on 3b7e4b0): through a full render document, not a helper ---------------------


def _render(
    profile: HardwareProfile, base_models: list[BaseModelSpec], packages: list[Package], *, context: int = 32768, requests: int = 10
):
    """A full `RenderDocument` for `profile`, not through the golden scenario's own fixed helpers --
    every Fix r1 test below builds its own minimal snapshot, so one test's packages can never leak
    into another's ranking."""
    from modelroom.contracts import Area

    machine = "server"
    accounts = sorted({(p.repo or "").split("/", 1)[0] for p in packages if p.repo})
    config = Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [{"name": "fam", "base_models": [{"hf_repo": b.hf_repo} for b in base_models]}],
            "packagers": accounts,
            "publishers": sorted({b.publisher for b in base_models}),
            "machines": {machine: {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True, "profile": profile.profile_id}},
            "paths": {"state": str(FIXTURES / "state"), "markdown": str(FIXTURES / "docs" / "models.md")},
        }
    )
    areas = [
        Area(source="huggingface", base_model_hf_repo=b.hf_repo, packager=accounts[0] if accounts else "acct", status="complete", last_success=SNAPSHOT_RUN_AT, error=None)
        for b in base_models
    ]
    snapshot = Snapshot(schema_version=2, run_at=SNAPSHOT_RUN_AT, areas=areas, base_models=base_models, packages=packages)
    profiles = {machine: MachineProfile(status="ranked", label=profile.display_name, profile=profile, reason=None)}
    scenario = Scenario(context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=requests)
    return build_render_document(config, snapshot, profiles, {}, scenario, RENDERED_AT)


def _architecture_without_heads(repo: str) -> Architecture:
    return Architecture(
        source_repo=repo, source_revision="a" * 40, kind="dense_classic",
        num_hidden_layers=36, num_key_value_heads=8, head_dim=128, num_attention_heads=None, max_context=131072,
    )


def test_a_missing_head_count_lands_in_fit_reason_and_is_rendered():
    """Codex r1-2: `t` stopping at one card for lack of a head count is not a guard failure (the
    fit still computes), but a reader must still see why -- in `Fit.reason` (so the JSON carries
    it) and in the row's own note text (Markdown and terminal alike), through the real render
    pipeline, not `vllm_fit_as_fit` called by hand."""
    from modelroom.fit_vllm import NO_HEAD_COUNT_REASON

    repo = "acme/NoHeads-7B"
    base = BaseModelSpec(
        hf_repo=repo, repo_aliases=[], ollama_base=None, ollama_tag=None, publisher="acme",
        parameters_b=7.0, architecture=_architecture_without_heads(repo),
    )
    package = Package(
        source="huggingface", repo="acme/NoHeads-7B-AWQ", revision="d" * 40, base_model_hf_repo=repo, format="tensor",
        files=[PackageFile(name="model.safetensors", role="weights", size_bytes=5_000_000_000, digest=None)],
        complete=True, quantization="AWQ", quantization_source="config", default_context=None,
        provenance="from_metadata", unresolved_reason=None, approval=None,
        observed_at=SNAPSHOT_RUN_AT, last_seen=SNAPSHOT_RUN_AT, active=True,
    )
    document = _render(_profile(), [base], [package])

    entry = document.machines[0].ranked[0]
    assert entry.fit.cards == 1 and entry.fit.reason == NO_HEAD_COUNT_REASON
    assert NO_HEAD_COUNT_REASON[0].upper() + NO_HEAD_COUNT_REASON[1:] + "." in entry.note.text

    markdown = document_markdown(document)
    assert "Tensor parallel needs the model's head count; computed on one card." in markdown
    dumped = document_json(document)
    assert dumped["machines"][0]["ranked"][0]["fit"]["reason"] == NO_HEAD_COUNT_REASON


def test_a_not_covered_vllm_row_carries_its_runtime_in_json_and_the_packager_cell():
    """Codex r1-3: every guard path of `vllm_fit_as_fit` must say `runtime == "vllm"` too, not the
    `Fit` default `"ollama"`. A weight file of size 0 (`fit_vllm.ZERO_WEIGHT_REASON`) is the guard
    reachable through a real render: `render._eligible_packages` already requires `complete`
    before a package is judged at all, so an *incomplete* package never reaches a fit here (that
    guard is `guided_vllm._fit`'s own entry point instead, proven directly in `test_fit_vllm.py`).
    Its JSON and its packager cell must both still read as vLLM's, not Ollama's."""
    from modelroom.fit_vllm import ZERO_WEIGHT_REASON
    from modelroom.views_runtime import packager_cell

    repo = "acme/ZeroWeight-7B"
    base = BaseModelSpec(
        hf_repo=repo, repo_aliases=[], ollama_base=None, ollama_tag=None, publisher="acme",
        parameters_b=7.0, architecture=_architecture_without_heads(repo),
    )
    package = Package(
        source="huggingface", repo="acme/ZeroWeight-7B-AWQ", revision="e" * 40, base_model_hf_repo=repo, format="tensor",
        files=[PackageFile(name="model.safetensors", role="weights", size_bytes=0, digest=None)],
        complete=True, quantization="AWQ", quantization_source="config", default_context=None,
        provenance="from_metadata", unresolved_reason=None, approval=None,
        observed_at=SNAPSHOT_RUN_AT, last_seen=SNAPSHOT_RUN_AT, active=True,
    )
    document = _render(_profile(), [base], [package])

    assert document.machines[0].ranked == []
    entry = document.machines[0].not_covered[0]
    assert entry.fit.fit_class == "unknown" and entry.fit.reason == ZERO_WEIGHT_REASON
    assert entry.fit.runtime == "vllm"
    assert packager_cell(entry) == "vLLM · acme"

    dumped = document_json(document)
    not_covered = dumped["machines"][0]["not_covered"][0]
    assert not_covered["fit"]["runtime"] == "vllm"
    markdown = document_markdown(document)
    assert "vLLM · acme" in markdown


def test_an_unequal_cards_machine_describes_the_group_not_a_packages_own_usage():
    """Codex r1-4: `mixed_cards_server.json` (three equal 22.49 GiB L4 cards, one smaller 11 GiB
    T4). The `runtime` row's continuation line names the *available* group ("can split over the 3
    cards"); the `memory` line's own vLLM piece names each row's *actual* card count from
    `Fit.cards` -- two packages here land on different `t` (one divides the group of 3 down to
    `t = 2`, the other has no head count and stays at `t = 1`), so the two numbers must differ.
    """
    profile = json.loads((FIXTURES / "profiles_v4" / "mixed_cards_server.json").read_text(encoding="utf-8"))
    profile = HardwareProfile.model_validate(profile)

    def _base(repo: str, heads: int | None) -> BaseModelSpec:
        arch = Architecture(
            source_repo=repo, source_revision="a" * 40, kind="dense_classic",
            num_hidden_layers=36, num_key_value_heads=8, head_dim=128, num_attention_heads=heads, max_context=131072,
        )
        return BaseModelSpec(hf_repo=repo, repo_aliases=[], ollama_base=None, ollama_tag=None, publisher="acme", parameters_b=7.0, architecture=arch)

    def _pkg(repo: str, base_repo: str, size: int) -> Package:
        return Package(
            source="huggingface", repo=repo, revision="f" * 40, base_model_hf_repo=base_repo, format="tensor",
            files=[PackageFile(name="model.safetensors", role="weights", size_bytes=size, digest=None)],
            complete=True, quantization="AWQ", quantization_source="config", default_context=None,
            provenance="from_metadata", unresolved_reason=None, approval=None,
            observed_at=SNAPSHOT_RUN_AT, last_seen=SNAPSHOT_RUN_AT, active=True,
        )

    base_t2 = _base("acme/Eight-7B", 8)  # group of 3 -> t = 2 (8 heads divide 2)
    base_t1 = _base("acme/NoHeads-7B", None)  # no head count -> t = 1
    packages = [_pkg("acme/Eight-7B-AWQ", base_t2.hf_repo, 5_000_000_000), _pkg("acme/NoHeads-7B-AWQ", base_t1.hf_repo, 5_000_000_000)]

    document = _render(profile, [base_t2, base_t1], packages, context=8192, requests=4)

    entries = {entry.base_model_hf_repo: entry for entry in document.machines[0].ranked}
    assert {repo: entry.fit.cards for repo, entry in entries.items()} == {"acme/Eight-7B": 2, "acme/NoHeads-7B": 1}
    # Sweep (Codex r1): the row's own note must say its own card count too, not the machine's four
    # (`profile.gpus`) -- the same "per-row, not per-machine" bug class as findings 1 and 4.
    assert "spread over 2 cards" in entries["acme/Eight-7B"].note.text
    assert "the reserve of 1.8" in entries["acme/Eight-7B"].note.text
    assert "spread over" not in entries["acme/NoHeads-7B"].note.text  # t = 1: mode "gpu", one card

    terminal = document_terminal(document)
    assert "vLLM can split over the 3 cards of 22.49 GB; 1 smaller card is left out" in terminal
    assert "vLLM on 1 card:" in terminal and "vLLM on 2 cards:" in terminal
