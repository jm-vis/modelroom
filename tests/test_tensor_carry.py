"""Tensor packages across fetch runs with and without vLLM, through `run_fetch` and the merge.

Two kinds of knowledge about a tensor package are kept apart here:

- **Content facts** (the quantization label and its source, read from `config.json`): they depend
  on the repository and its revision alone, so a run without vLLM keeps them at the same revision,
  whatever the provenance verdict was -- an `unresolved` package included.
- **Verdicts** (provenance, the reason): they also depend on which base model the package is
  assigned to now. A run without vLLM never copies one; it decides again, without a new request,
  from the facts it kept and the current assignment, and must reach the verdict a run with vLLM
  would reach.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from modelroom.cli import fetch_with_config, render_with_config
from modelroom.config import Configuration
from modelroom.fetch import run_fetch
from modelroom.http import Response
from modelroom.state import read_snapshot, write_snapshot

from fixture_support import build_transport, place_v2_profile

HF = "https://huggingface.co"
RUN1 = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
RUN2 = datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc)
RUN3 = datetime(2026, 9, 22, 11, 0, 0, tzinfo=timezone.utc)

BASE = "acme/Nova-7B"
OTHER_BASE = "acme/Nova-7B-Chat"
TENSOR_REPO = "builder/Nova-7B-FP8"
SHA_BASE = "a" * 40
SHA_TENSOR = "c" * 40


def config_for(tmp_path, base: str = BASE, *, other: bool = False, profile_id: str | None = None) -> Configuration:
    """One family; the tensor repository is a `repos` target of `base`. With `other`, a second
    base model is configured beside it, so the repository can move between the two."""
    base_models = [{"hf_repo": base, "repos": [TENSOR_REPO]}]
    if other:
        moved_from = BASE if base == OTHER_BASE else OTHER_BASE
        base_models.insert(0, {"hf_repo": moved_from})
    machine = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}
    if profile_id is not None:
        machine["profile"] = profile_id
    return Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [{"name": "nova", "base_models": base_models}],
            "packagers": [],
            "publishers": ["acme"],
            "machines": {"workstation": machine},
            "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "models.md")},
        }
    )


def _json(payload) -> Response:
    return Response(status=200, headers={}, body=json.dumps(payload).encode("utf-8"))


def _base_answers(base: str) -> dict:
    return {
        ("GET", f"{HF}/api/models/{base}"): Response(status=200, headers={}, body=b'{"sha": "' + SHA_BASE.encode() + b'"}'),
        ("GET", f"{HF}/{base}/resolve/{SHA_BASE}/config.json"): Response(status=404, headers={}, body=b"{}"),
        ("GET", f"{HF}/api/models/{base}-GGUF"): Response(status=401, headers={}, body=b"{}"),
    }


def mapping(*, with_config: bool, tree_ok: bool = True) -> dict:
    """Every answer a fetch of either configuration asks for. `with_config=False` leaves the
    tensor repository's `config.json` unmapped: asking for it would fail the test."""
    tree = [
        {"type": "file", "path": "model.safetensors", "size": 8_000_000_000, "lfs": {"oid": "m" * 64}},
        {"type": "file", "path": "draft.safetensors", "size": 1_000_000_000, "lfs": {"oid": "d" * 64}},
        {"type": "file", "path": "config.json", "size": 900},
    ]
    answers = {
        **_base_answers(BASE),
        **_base_answers(OTHER_BASE),
        ("GET", f"{HF}/api/models/{TENSOR_REPO}"): _json(
            {
                "sha": SHA_TENSOR,
                "tags": [f"base_model:{BASE}", f"base_model:quantized:{BASE}"],
                "cardData": {"base_model": [BASE]},
            }
        ),
        ("GET", f"{HF}/api/models/{TENSOR_REPO}/tree/{SHA_TENSOR}?recursive=true"): (
            _json(tree) if tree_ok else Response(status=500, headers={}, body=b"{}")
        ),
    }
    if with_config:
        answers[("GET", f"{HF}/{TENSOR_REPO}/resolve/{SHA_TENSOR}/config.json")] = _json({"torch_dtype": "bfloat16"})
    return answers


def tensor_packages(snapshot, base: str | None = None) -> dict[str, tuple]:
    """`file -> (base model, quantization, source, provenance, reason)` of the tensor repository."""
    return {
        package.files[0].name: (
            package.base_model_hf_repo,
            package.quantization,
            package.quantization_source,
            package.provenance,
            package.unresolved_reason,
        )
        for package in snapshot.packages
        if package.repo == TENSOR_REPO and package.active and (base is None or package.base_model_hf_repo == base)
    }


def config_requests(transport) -> list[str]:
    return [url for _method, url in transport.calls if url.endswith(f"{TENSOR_REPO}/resolve/{SHA_TENSOR}/config.json")]


def builder_area(snapshot, base: str = BASE):
    return next(a for a in snapshot.areas if a.packager == "builder" and a.base_model_hf_repo == base)


EXPECTED = {
    "model.safetensors": (BASE, "BF16", "config", "from_metadata", None),
    # The draft model in the same repository: its label is a fact of this repository and
    # revision; its verdict stays `file_stem` (its weight file is not a library file name).
    "draft.safetensors": (BASE, "BF16", "config", "unresolved", "file_stem"),
}


def test_with_vllm_then_without_then_with_keeps_facts_and_verdicts(tmp_path):
    config = config_for(tmp_path)
    first = run_fetch(config, build_transport(mapping(with_config=True)), RUN1, None, build_tensor=True)
    assert builder_area(first.snapshot).status == "complete"
    assert tensor_packages(first.snapshot) == EXPECTED

    without = build_transport(mapping(with_config=False))
    second = run_fetch(config, without, RUN2, first.snapshot, build_tensor=False)
    assert builder_area(second.snapshot).status == "complete", builder_area(second.snapshot).error
    assert config_requests(without) == []
    assert tensor_packages(second.snapshot) == EXPECTED

    again = build_transport(mapping(with_config=True))
    third = run_fetch(config, again, RUN3, second.snapshot, build_tensor=True)
    assert len(config_requests(again)) == 1
    assert tensor_packages(third.snapshot) == EXPECTED


def test_the_draft_keeps_its_label_without_vllm_although_it_is_unresolved(tmp_path):
    """An `unresolved` verdict is no reason to drop a known content fact."""
    config = config_for(tmp_path)
    first = run_fetch(config, build_transport(mapping(with_config=True)), RUN1, None, build_tensor=True)
    second = run_fetch(config, build_transport(mapping(with_config=False)), RUN2, first.snapshot, build_tensor=False)

    assert tensor_packages(second.snapshot)["draft.safetensors"] == (BASE, "BF16", "config", "unresolved", "file_stem")


def test_without_any_earlier_vllm_run_a_tensor_group_stays_locked(tmp_path):
    config = config_for(tmp_path)
    first = run_fetch(config, build_transport(mapping(with_config=False)), RUN1, None, build_tensor=False)
    second = run_fetch(config, build_transport(mapping(with_config=False)), RUN2, first.snapshot, build_tensor=False)

    locked = {
        "model.safetensors": (BASE, None, None, "unresolved", "format"),
        "draft.safetensors": (BASE, None, None, "unresolved", "format"),
    }
    assert tensor_packages(first.snapshot) == locked
    assert tensor_packages(second.snapshot) == locked


def test_a_new_assignment_at_the_same_revision_is_decided_again_not_copied(tmp_path):
    """The revision confirms the content, not its relation to the base model
    it is assigned to now. Without vLLM the verdict is decided again from the kept facts, with
    no new request, and equals what a run with vLLM decides for the same assignment."""
    before = config_for(tmp_path, BASE, other=True)
    first = run_fetch(before, build_transport(mapping(with_config=True)), RUN1, None, build_tensor=True)
    assert tensor_packages(first.snapshot, BASE)["model.safetensors"][3] == "from_metadata"

    moved = config_for(tmp_path, OTHER_BASE, other=True)
    without = build_transport(mapping(with_config=False))
    second = run_fetch(moved, without, RUN2, first.snapshot, build_tensor=False)
    with_vllm = run_fetch(moved, build_transport(mapping(with_config=True)), RUN2, first.snapshot, build_tensor=True)

    assert config_requests(without) == []
    kept = tensor_packages(second.snapshot, OTHER_BASE)
    assert kept == tensor_packages(with_vllm.snapshot, OTHER_BASE)
    assert kept["model.safetensors"][1:3] == ("BF16", "config")
    assert kept["model.safetensors"][3] == "unresolved"
    assert kept["model.safetensors"][4] != "format"


def test_moving_back_to_the_first_assignment_restores_its_verdict_without_a_request(tmp_path):
    first = run_fetch(config_for(tmp_path, BASE, other=True), build_transport(mapping(with_config=True)), RUN1, None, build_tensor=True)
    moved = run_fetch(
        config_for(tmp_path, OTHER_BASE, other=True), build_transport(mapping(with_config=False)), RUN2, first.snapshot,
        build_tensor=False,
    )
    without = build_transport(mapping(with_config=False))
    back = run_fetch(config_for(tmp_path, BASE, other=True), without, RUN3, moved.snapshot, build_tensor=False)

    assert config_requests(without) == []
    assert tensor_packages(back.snapshot, BASE) == EXPECTED


# --- what the output of a machine without vLLM shows --------------------------------------------


def _folder(tmp_path, name: str) -> Configuration:
    """A results folder whose one machine has a measured profile without vLLM."""
    bare = config_for(tmp_path / name)
    profile = place_v2_profile(bare.paths.hardware_dir)
    return config_for(tmp_path / name, profile_id=profile.profile_id)


def _outputs(config: Configuration) -> dict[str, bytes]:
    assert render_with_config(config, now=RUN3) == 0
    markdown = config.paths.markdown
    return {"markdown": markdown.read_bytes(), "json": markdown.with_suffix(".json").read_bytes()}


def test_tensor_packages_from_an_earlier_vllm_run_leave_no_trace_in_the_output_without_vllm(tmp_path):
    """The whole Markdown and JSON output for a machine without vLLM, byte for
    byte -- the document-wide counts included -- is the same whether the snapshot holds tensor
    packages a vLLM run resolved (and an incomplete fetch since then kept), tensor packages that
    were only ever locked, or no tensor package at all. The snapshot itself keeps its raw facts."""
    seen = _folder(tmp_path, "seen")
    fetch_with_config(seen, "workstation", build_transport(mapping(with_config=True)), RUN1, build_tensor=True)
    fetch_with_config(seen, "workstation", build_transport(mapping(with_config=False, tree_ok=False)), RUN2)
    kept = read_snapshot(seen)
    assert builder_area(kept).status == "incomplete"
    assert tensor_packages(kept) == EXPECTED

    never = _folder(tmp_path, "never")
    fetch_with_config(never, "workstation", build_transport(mapping(with_config=False)), RUN1)
    fetch_with_config(never, "workstation", build_transport(mapping(with_config=False, tree_ok=False)), RUN2)
    assert {row[3] for row in tensor_packages(read_snapshot(never)).values()} == {"unresolved"}

    stripped = _folder(tmp_path, "stripped")
    data = kept.model_dump(mode="json")
    data["packages"] = [package for package in data["packages"] if package["format"] != "tensor"]
    write_snapshot(stripped, data)

    shown = _outputs(seen)
    assert shown == _outputs(never)
    assert shown == _outputs(stripped)
