"""The served model is found through the configuration a guided run writes (AP9-UX2 P3, SV1-F).

Until AP9-UX2 a guided run wrote only the repository of the hit that led to a base model as a
`repos` target -- a packager's GGUF repository, as a rule -- so the fetch never read the
publisher's own `tensor` package and `match_served` found nothing for a vLLM server serving the
base model (second hand run of 2026-09-28). The tests of `tests/test_guided_vllm.py` lay the
configuration out by hand with the served repository in `repos`; they prove the measurement, not
this mapping. Here the configuration is the one `guided._new_configuration` writes, changed only by
`apply_hits`, and fetched from public Hub recordings of `Qwen/Qwen3.5-4B` (`tests/fixtures/README.md`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from modelroom.catalog import load_catalog
from modelroom.config import Configuration
from modelroom.fetch import run_fetch
from modelroom.fit_vllm import vllm_fit_as_fit
from modelroom.guided_contracts import SearchHit
from modelroom.guided_vllm import match_served
from modelroom.measurements import Scenario
from modelroom.profile import HardwareProfile
from modelroom.search_apply import apply_hits
from modelroom.vllm_server import ServedModel

from fixture_support import FIXTURES, QWEN35_4B_REPO, build_transport, packager_probes, qwen35_4b_transport_mapping

RUN_AT = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
SHARDS = (5_329_398_688, 3_990_429_408)  # the two safetensors shards of the public tree
L4_CARD_GIB = 22.49  # the card of `profiles_v4/two_l4_server.json`, as the profile records it


def _guided_configuration(folder: Path) -> Configuration:
    """Exactly what `guided._new_configuration` writes (`packagers = []`), for this folder."""
    return Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [],
            "packagers": [],
            "publishers": [],
            "machines": {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}},
            "paths": {"state": str(folder / "state"), "markdown": str(folder / "docs" / "models.md")},
            "guided": {"results": str(folder)},
        }
    )


def _packager_hit() -> SearchHit:
    return SearchHit(
        repo="unsloth/Qwen3.5-4B-GGUF",
        publisher_status="listed packager",
        base_model=[QWEN35_4B_REPO],
        base_model_relation="quantized",
        resolved=True,
        resolved_base_model=QWEN35_4B_REPO,
    )


def test_the_served_model_is_found_through_the_guided_configuration(tmp_path: Path):
    config = apply_hits(_guided_configuration(tmp_path.resolve()), [_packager_hit()], catalog=load_catalog())
    transport = build_transport({**packager_probes(config), **qwen35_4b_transport_mapping()})

    snapshot = run_fetch(config, transport, RUN_AT, None).snapshot
    matched = match_served(snapshot.packages, ServedModel(id=QWEN35_4B_REPO, root=QWEN35_4B_REPO, max_model_len=32768))

    assert not isinstance(matched, str), matched
    assert matched.format == "tensor" and matched.repo == QWEN35_4B_REPO
    weights = sum(file.size_bytes for file in matched.files if file.role in ("weights", "weights_shard"))
    assert weights == sum(SHARDS)
    base = next(spec for spec in snapshot.base_models if spec.hf_repo == QWEN35_4B_REPO)
    assert base.architecture.kind == "hybrid_linear"
    # The fit through its integration point `guided_vllm._fit` (Codex r1 m9), now `runtimes.fit_for`
    # itself (AP9-SV2-F, "eine Rechnung, eine Wahrheit"): the profile this folder binds this machine
    # to is the recorded L4 server, two equal 22.49 GiB cards; the base model's 16 attention heads
    # divide evenly by 2, so tensor parallel spreads the package over both (`t = 2`).
    fit, reason, notes = _guided_fit(tmp_path.resolve(), config, matched, snapshot.base_models)
    assert reason is None and fit.basis == "architecture" and fit.cards == 2
    profile = HardwareProfile.model_validate(
        json.loads((FIXTURES / "profiles_v4" / "two_l4_server.json").read_text(encoding="utf-8"))
    )
    scenario = Scenario(context_requested=32768, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1)
    # The very function `runtimes.fit_for` calls for this row: one calculation, one truth, so a
    # render of this same folder could never compute anything else for it.
    assert fit == vllm_fit_as_fit(profile, matched, base, scenario, config.machines["workstation"])
    assert any(f"{fit.at_once} requests of context 32k at once" in note for note in notes)


def _guided_fit(folder: Path, config: Configuration, package, base_models):
    """`guided_vllm._fit` for `folder`, bound to the recorded L4 profile; returns fit, reason, notes."""
    import json
    from types import SimpleNamespace

    from fixture_support import FIXTURES, TWO_L4_ID
    from modelroom.binding import POINTER_SCHEMA_VERSION, GuidedPointer, write_pointer
    from modelroom.guided_vllm import _fit
    from modelroom.measurements import Scenario

    profile = json.loads((FIXTURES / "profiles_v4" / "two_l4_server.json").read_text(encoding="utf-8"))
    config.paths.hardware_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.hardware_dir / f"{TWO_L4_ID}.json").write_bytes(json.dumps(profile).encode("utf-8"))
    data = config.model_dump(mode="json")
    data["machines"]["workstation"]["profile"] = TWO_L4_ID
    bound = Configuration.from_dict(data)
    pointer = folder / "guided.json"
    write_pointer(pointer, GuidedPointer(schema_version=POINTER_SCHEMA_VERSION, current=str(folder), bindings={str(folder): TWO_L4_ID}))
    notes: list[str] = []
    run = SimpleNamespace(pointer_path=pointer, screen=SimpleNamespace(note=notes.append))
    scenario = Scenario(context_requested=32768, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1)
    fit, reason = _fit(run, bound, folder, package, base_models, scenario)
    return fit, reason, notes
