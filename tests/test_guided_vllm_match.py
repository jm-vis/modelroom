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

from datetime import datetime, timezone
from pathlib import Path

from modelroom.catalog import load_catalog
from modelroom.config import Configuration
from modelroom.fetch import run_fetch
from modelroom.fit_vllm import vllm_fit
from modelroom.guided_contracts import SearchHit
from modelroom.guided_vllm import match_served
from modelroom.search_apply import apply_hits
from modelroom.vllm_server import ServedModel

from fixture_support import QWEN35_4B_REPO, build_transport, packager_probes, qwen35_4b_transport_mapping

RUN_AT = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
SHARDS = (5_329_398_688, 3_990_429_408)  # the two safetensors shards of the public tree
L4_CARD_GIB = 22.494


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
    # The fit of `guided_vllm._fit`, on one L4 as the recorded server profile has it.
    fit = vllm_fit(L4_CARD_GIB, weights, context=32768, requests=1, architecture=base.architecture)
    assert fit.basis == "architecture"
    kv_bytes = int(fit.kv_gib * 1024**3)
    assert fit.at_once == kv_bytes // (32_768 * 32_768 + 52_690_944)
