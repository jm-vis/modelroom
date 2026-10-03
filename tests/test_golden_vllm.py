"""The rendered document of a machine with vLLM, held against golden files (AP9-SV2-F).

Every other golden file of this repository (`tests/golden/guided-screen*.txt`) is a reasoning
machine without vLLM, so none of them changes a byte for this sub-package (CONTRACTS.md, "Fit for
vLLM"). This machine -- four equal 22.49 GiB L4 cards, `profiles_v4/four_l4_server_vllm.json` --
is new, and only here: one GGUF row and one Hugging Face tensor row of the very same base model,
so the Markdown file and the terminal view show one section per runtime (F5) -- each with its own
table, `memory` line and key -- under the machine's `runtime` row; the JSON stays one block.
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
from modelroom.screen import FIT_HINT
from modelroom.views import document_markdown, document_terminal

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"
SNAPSHOT_RUN_AT = datetime(2026, 9, 30, 8, 0, 0, tzinfo=timezone.utc)
RENDERED_AT = datetime(2026, 9, 30, 8, 5, 0, tzinfo=timezone.utc)
BASE_REPO = "Qwen/Qwen3-8B"
# F5: what vLLM's own key says its `gpu_split` is (never Ollama's "all cards together"), and its start.
VLLM_SPLIT_PIECE = "gpu_split: 2, 4 or 8 equal cards of the largest size, as the head count allows"
VLLM_KEY_START = "Fit is memory only: how many requests of this context the cards hold at once"


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
    profile: HardwareProfile, base_models: list[BaseModelSpec], packages: list[Package], *, context: int = 32768, requests: int = 10,
    live_vllm: dict | None = None,
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
    return build_render_document(config, snapshot, profiles, {}, scenario, RENDERED_AT, live_vllm=live_vllm)


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
    assert packager_cell(entry) == "acme"  # F5: no prefix, the machine's vLLM key says whose row it is

    dumped = document_json(document)
    not_covered = dumped["machines"][0]["not_covered"][0]
    assert not_covered["fit"]["runtime"] == "vllm"
    markdown = document_markdown(document)
    assert "| acme |" in markdown
    assert VLLM_KEY_START in markdown and FIT_HINT[0] not in markdown


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

    # Rule (b) (SV2-D): 17 GB of weights hold 2 of 4 requests on one card (`tight`) and 20 on two
    # (`fits`), so the smallest `t` that fits is 2 -- a smaller package would take one card.
    base_t2 = _base("acme/Eight-7B", 8)  # group of 3 -> t = 2 (8 heads divide 2, and one card is too few)
    base_t1 = _base("acme/NoHeads-7B", None)  # no head count -> t = 1
    packages = [_pkg("acme/Eight-7B-AWQ", base_t2.hf_repo, 17_000_000_000), _pkg("acme/NoHeads-7B-AWQ", base_t1.hf_repo, 17_000_000_000)]

    document = _render(profile, [base_t2, base_t1], packages, context=8192, requests=4)

    entries = {entry.base_model_hf_repo: entry for entry in document.machines[0].ranked}
    assert {repo: entry.fit.cards for repo, entry in entries.items()} == {"acme/Eight-7B": 2, "acme/NoHeads-7B": 1}
    # Sweep (Codex r1): the row's own note must say its own card count too, not the machine's four
    # (`profile.gpus`) -- the same "per-row, not per-machine" bug class as findings 1 and 4. Since
    # Fix r2 the note is vLLM's own (requests at once on its cards), never Ollama's reserve wording.
    assert "at once on 2 cards of 22.5 GiB" in entries["acme/Eight-7B"].note.text
    assert "at once on one card of 22.5 GiB" in entries["acme/NoHeads-7B"].note.text
    for entry in entries.values():
        assert "graphics memory" not in entry.note.text and "after the reserve" not in entry.note.text

    terminal = document_terminal(document)
    assert "vLLM can split over at most 2 of the 3 cards of 22.49 GB; 1 smaller card is left out" in terminal
    assert "vLLM on 1 card:" in terminal and "vLLM on 2 cards:" in terminal
    # Codex r2-3, F5: the key may not claim `all cards together` for vLLM, which uses two of the
    # three equal cards here and never the smaller T4 -- a table of vLLM rows carries vLLM's key alone.
    markdown = document_markdown(document)
    for text in (markdown, terminal):
        assert VLLM_SPLIT_PIECE in text
        assert "gpu_split: all cards together" not in text


# --- Fix r2 (Codex r2 on 8d05e92): every text that explains a row reads `Fit.runtime` ---------------


def _qwen3_base(heads: int | None = 32) -> BaseModelSpec:
    """The brief's own rechenbeispiel model; `heads=None` forces `t = 1` (F1)."""
    return _base_model().model_copy(update={"architecture": _architecture().model_copy(update={"num_attention_heads": heads})})


def _awq(size: int = 6_098_581_864) -> Package:
    return _tensor_package().model_copy(
        update={"repo": "acme/Qwen3-8B-AWQ", "files": [PackageFile(name="model.safetensors", role="weights", size_bytes=size, digest=None)]}
    )


# One case per class vLLM's formula reaches (F2): the head count (None -> `t = 1`), the requests, the
# weights, and the words the row's note must carry instead of Ollama's "Fits into graphics memory".
VLLM_CLASSES = {
    # The rechenbeispiel at `t = 4`: 15 at once, 10 assumed -> fits.
    "fits": (32, 10, 6_098_581_864, "vLLM holds 15 requests at once on 4 cards of 22.5 GiB"),
    # Rule (b): one request assumed, the smallest `t` (one card) holds 3: twice them or more -> ample.
    "ample": (32, 1, 6_098_581_864, "vLLM holds 3 requests at once on one card of 22.5 GiB"),
    # The rechenbeispiel at `t = 1`: 3 of 10 at once -> tight (52.2 GiB needed against a 20.7 GiB pool).
    "tight": (None, 10, 6_098_581_864, "vLLM holds only 3 of 10 requests at once on one card of 22.5 GiB"),
    # 20 GB of weights on one card leave no KV cache at all -> no fit.
    "no_fit": (None, 10, 20_000_000_000, "vLLM holds not even one request at once on one card of 22.5 GiB"),
}


@pytest.mark.parametrize("fit_class", list(VLLM_CLASSES))
def test_every_vllm_class_explains_its_row_by_requests_at_once(fit_class):
    """Codex r2-1: a vLLM row's note says how many requests its cards hold at once against the
    requests the ranking assumes -- never Ollama's "Fits into graphics memory" or its reserve,
    which for `tight` (3 of 10) claimed a fit that does not hold. Through the full document:
    the row's note, the Markdown table, the JSON and the terminal view."""
    heads, requests, size, said = VLLM_CLASSES[fit_class]
    document = _render(_profile(), [_qwen3_base(heads)], [_awq(size)], requests=requests)

    block = document.machines[0]
    entry = (block.no_fit if fit_class == "no_fit" else block.ranked)[0]
    assert (entry.fit.runtime, entry.fit.fit_class) == ("vllm", fit_class)
    assert said in entry.note.text
    for ollama_words in ("graphics memory", "after the reserve", "Fit contract v1"):
        assert ollama_words not in entry.note.text
    assert entry.note.text in document_markdown(document)
    listed = document_json(document)["machines"][0]["no_fit" if fit_class == "no_fit" else "ranked"][0]
    assert listed["note"]["text"] == entry.note.text and listed["fit"].get("at_once", 0) == entry.fit.at_once
    terminal = document_terminal(document)
    if fit_class == "no_fit":
        assert "1 no fit (acme AWQ)" in terminal
    else:
        row = next(line for line in terminal.splitlines() if "acme · AWQ" in line)
        assert f" {fit_class} " in row


def test_an_ollama_only_table_on_a_vllm_machine_keeps_its_own_key():
    """The counter-proof: a machine with vLLM that shows no vLLM row keeps Ollama's key, word for word."""
    document = _render(_profile(), [_base_model()], [_gguf_package()])

    assert " · ".join(FIT_HINT) in document_markdown(document)
    assert VLLM_KEY_START not in document_markdown(document) + document_terminal(document)


@pytest.mark.parametrize("where", ["no_fit", "not_covered"])
def test_vllm_rows_set_aside_get_the_legend_beside_ranked_ollama_rows(where):
    """Codex r2-4, F5: Ollama ranks here, and the one vLLM row stands under `No fit` or `Not
    covered` -- it gets the vLLM section with vLLM's own key, in the Markdown and the terminal."""
    tensor = _awq(20_000_000_000 if where == "no_fit" else 0)
    document = _render(_profile(), [_qwen3_base(None)], [_gguf_package(), tensor])

    block = document.machines[0]
    assert {entry.fit.runtime for entry in block.ranked} == {"ollama"}
    assert [entry.fit.runtime for entry in getattr(block, where)] == ["vllm"]
    terminal = document_terminal(document)
    for text, heading in ((document_markdown(document), "### with vLLM"), (terminal, " with vLLM")):
        vllm = text.index(heading)
        assert VLLM_KEY_START in text[vllm:] and FIT_HINT[0] in text[:vllm]
    assert "acme AWQ" in terminal.split("\n with vLLM\n")[1]  # the `shown` line of the vLLM section


def _gguf(size: int) -> Package:
    return _gguf_package().model_copy(
        update={"files": [PackageFile(name="Qwen3-8B-Q4_K_M.gguf", role="weights", size_bytes=size, digest="sha256:" + "1" * 64)]}
    )


@pytest.mark.parametrize("where", ["no_fit", "not_covered"])
def test_a_machine_whose_only_rows_are_vllm_set_aside_still_gets_the_legend(where):
    """Codex r3-1: no row is ranked at all, and the one row the tables show is vLLM's, under `No
    fit` or `Not covered`. The key is owed to that row all the same -- in the Markdown file and in
    the terminal, without the speed hint (no table with a speed column is shown)."""
    tensor = _awq(20_000_000_000 if where == "no_fit" else 0)
    document = _render(_profile(), [_qwen3_base(None)], [tensor])

    block = document.machines[0]
    assert block.ranked == [] and [entry.fit.runtime for entry in getattr(block, where)] == ["vllm"]
    markdown, terminal = document_markdown(document), document_terminal(document)
    for text in (markdown, terminal):
        assert VLLM_KEY_START in text and FIT_HINT[0] not in text
    assert any(line.startswith(" key ") for line in terminal.splitlines())
    assert "Speed is measured, see its column." not in markdown + terminal


@pytest.mark.parametrize("size", [300_000_000_000, 0], ids=["no_fit", "not_covered"])
def test_an_ollama_machine_with_nothing_ranked_still_gets_no_key(size):
    """The counter-proof of Codex r3-1: an Ollama row set aside, nothing ranked -- no key at all,
    exactly as before AP9-SV2, on a machine with vLLM as on any other."""
    document = _render(_profile(), [_base_model()], [_gguf(size)])

    block = document.machines[0]
    assert block.ranked == [] and len(block.no_fit) + len(block.not_covered) == 1
    markdown, terminal = document_markdown(document), document_terminal(document)
    assert FIT_HINT[0] not in markdown and FIT_HINT[0] not in terminal
    assert not any(line.startswith(" key ") for line in terminal.splitlines())


def test_the_terminal_view_of_a_machine_with_vllm_stays_within_100_columns():
    """Sweep (Fix r2): the legend's vLLM sentences are pieces the terminal can wrap between; one
    sentence of 200 characters ran past the 100-column window the view is drawn for."""
    lines = document_terminal(_document()).splitlines()

    assert [line for line in lines if len(line) > 100] == []
