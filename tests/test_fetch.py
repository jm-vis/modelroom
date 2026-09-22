"""Tests for modelroom.fetch: orchestrating hf.py + ollama.py into one merged Snapshot."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone

from modelroom.config import Configuration
from modelroom.contracts import Approval, Architecture, BaseModelSpec, Package, PackageFile, Snapshot, load_snapshot
from modelroom.fetch import run_fetch
from modelroom.http import Response
from modelroom.quantization import package_identity_key

from fixture_support import (
    FIXTURES,
    build_transport,
    envelope_response,
    qwen35_example_config_dict,
    qwen35_transport_mapping,
)

RUN1 = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
RUN2 = datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc)


def _config(tmp_path) -> Configuration:
    return Configuration.from_dict(qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md")))


def test_run_fetch_all_areas_complete_builds_a_valid_snapshot(tmp_path):
    config = _config(tmp_path)
    transport = build_transport(qwen35_transport_mapping())

    result = run_fetch(config, transport, RUN1, old_snapshot=None)

    assert isinstance(result.snapshot, Snapshot)
    assert result.snapshot.run_at == RUN1
    assert all(area.status == "complete" for area in result.snapshot.areas)
    # unsloth (packages), Qwen-as-its-own-packager (empty), ollama
    assert len(result.snapshot.areas) == 3
    assert len(result.snapshot.base_models) == 1
    assert result.snapshot.base_models[0].hf_repo == "Qwen/Qwen3.5-9B"
    assert any(pkg.source == "huggingface" for pkg in result.snapshot.packages)
    assert any(pkg.source == "ollama" for pkg in result.snapshot.packages)


def test_run_fetch_parameters_b_is_none_with_no_prior_snapshot_and_no_safetensors_total(tmp_path):
    # F11: no fabricated placeholder any more -- "not measured this run, and never measured
    # before" is represented as None, never a value that looks like a real reading.
    config = _config(tmp_path)
    mapping = qwen35_transport_mapping()
    mapping[("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B")] = Response(
        status=200, headers={}, body=b'{"sha": "' + b"a" * 40 + b'"}'  # no "safetensors" field
    )
    mapping[("GET", f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{'a' * 40}/config.json")] = Response(
        status=404, headers={}, body=b"{}"
    )
    transport = build_transport(mapping)

    result = run_fetch(config, transport, RUN1, old_snapshot=None)

    assert result.snapshot.base_models[0].parameters_b is None
    assert isinstance(load_snapshot(result.snapshot.model_dump(mode="json")), Snapshot)


def test_run_fetch_parameters_b_falls_back_to_the_previous_snapshots_value(tmp_path):
    config = _config(tmp_path)
    mapping = qwen35_transport_mapping()
    mapping[("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B")] = Response(
        status=200, headers={}, body=b'{"sha": "' + b"a" * 40 + b'"}'
    )
    mapping[("GET", f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{'a' * 40}/config.json")] = Response(
        status=404, headers={}, body=b"{}"
    )
    transport = build_transport(mapping)
    old_snapshot = Snapshot(
        schema_version=1,
        run_at=RUN1,
        areas=[],
        base_models=[
            BaseModelSpec(
                hf_repo="Qwen/Qwen3.5-9B",
                repo_aliases=["Qwen3.5-9B-GGUF"],
                ollama_base="qwen3.5",
                ollama_tag="9b",
                publisher="Qwen",
                parameters_b=42.0,
                architecture=Architecture(source_repo="Qwen/Qwen3.5-9B", source_revision=None, kind="unknown"),
            )
        ],
        packages=[],
    )

    result = run_fetch(config, transport, RUN2, old_snapshot=old_snapshot)

    assert result.snapshot.base_models[0].parameters_b == 42.0


def test_run_fetch_a_failing_area_ends_the_run_partially_incomplete(tmp_path):
    config = _config(tmp_path)
    mapping = qwen35_transport_mapping()
    mapping[
        (
            "GET",
            f"https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5?recursive=true",
        )
    ] = Response(status=500, headers={}, body=b"error")
    transport = build_transport(mapping)

    result = run_fetch(config, transport, RUN1, old_snapshot=None)

    statuses = {(area.source, area.packager): area.status for area in result.snapshot.areas}
    assert statuses[("huggingface", "unsloth")] == "incomplete"
    assert statuses[("ollama", None)] == "complete"


def test_run_fetch_reports_the_requests_used_and_the_default_budget(tmp_path):
    config = _config(tmp_path)
    transport = build_transport(qwen35_transport_mapping())

    result = run_fetch(config, transport, RUN1, old_snapshot=None)

    assert result.request_budget == 400
    assert 0 < result.request_used < 400


# --- F3: a malformed registry response in one area never aborts the whole run --------------


def test_run_fetch_survives_a_malformed_hf_tree_and_still_completes_the_ollama_area(tmp_path):
    config = _config(tmp_path)
    mapping = qwen35_transport_mapping()
    mapping[
        (
            "GET",
            f"https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5?recursive=true",
        )
    ] = Response(status=200, headers={}, body=b"[null]")  # a shape that used to raise AttributeError
    transport = build_transport(mapping)

    result = run_fetch(config, transport, RUN1, old_snapshot=None)

    statuses = {(area.source, area.packager): area.status for area in result.snapshot.areas}
    assert statuses[("huggingface", "unsloth")] == "incomplete"
    assert statuses[("ollama", None)] == "complete"


# --- F6: an approval bound to the current content survives a second fetch ------------------


def test_run_fetch_carries_forward_an_approval_bound_to_the_current_content(tmp_path):
    config = _config(tmp_path)
    transport = build_transport(qwen35_transport_mapping())
    # The digest this fixture actually resolves to at fetch time (sha256 of the recorded "9b"
    # manifest body, F5) -- an approval bound to exactly this content must survive the fetch.
    real_digest = f"sha256:{hashlib.sha256((FIXTURES / 'ollama_qwen35_9b.json').read_bytes()).hexdigest()}"
    old_ollama_package = Package(
        source="ollama",
        ollama_name="qwen3.5:9b",
        manifest_digest=real_digest,
        base_model_hf_repo="Qwen/Qwen3.5-9B",
        format="gguf",
        files=[PackageFile(name="9b.gguf", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization="Q4_K_M",
        default_context=None,
        provenance="approved",
        unresolved_reason=None,
        approval=Approval(date=date(2026, 9, 1), content=real_digest, by="acme-ai-team"),
        observed_at=RUN1,
        last_seen=RUN1,
        active=True,
    )
    old_snapshot = Snapshot(
        schema_version=1,
        run_at=RUN1,
        areas=[],
        base_models=[
            BaseModelSpec(
                hf_repo="Qwen/Qwen3.5-9B",
                repo_aliases=["Qwen3.5-9B-GGUF"],
                ollama_base="qwen3.5",
                ollama_tag="9b",
                publisher="Qwen",
                parameters_b=9.653104368,
                architecture=Architecture(source_repo="Qwen/Qwen3.5-9B", source_revision=None, kind="unknown"),
            )
        ],
        packages=[old_ollama_package],
    )

    result = run_fetch(config, transport, RUN2, old_snapshot=old_snapshot)

    by_key = {package_identity_key(p): p for p in result.snapshot.packages}
    refreshed = by_key[package_identity_key(old_ollama_package)]
    assert refreshed.provenance == "approved"
    assert refreshed.approval.content == real_digest


def test_run_fetch_stops_making_requests_once_the_budget_is_exhausted(tmp_path):
    config = _config(tmp_path)
    transport = build_transport(qwen35_transport_mapping())

    result = run_fetch(config, transport, RUN1, old_snapshot=None, budget=2)

    assert result.request_used == 2
    assert result.request_budget == 2
    assert any(area.status == "incomplete" for area in result.snapshot.areas)
    assert any("budget exhausted" in (area.error or "") for area in result.snapshot.areas)


# --- F9: a base model this run never reached keeps its previous spec, never "unknown" -------


def test_run_fetch_budget_end_never_touches_a_base_model_it_did_not_reach(tmp_path):
    config = Configuration.from_dict(
        {
            "schema_version": 1,
            "families": [
                {"name": "nova", "base_models": [{"hf_repo": "acme/Nova-7B", "repo_aliases": []}]},
                {"name": "otherfam", "base_models": [{"hf_repo": "other-org/Other-7B", "repo_aliases": []}]},
            ],
            "packagers": [],
            "publishers": ["acme", "other-org"],
            "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "models.md")},
        }
    )
    sha = "a" * 40
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/acme/Nova-7B"): Response(
                status=200, headers={}, body=b'{"sha": "' + sha.encode() + b'"}'
            ),
            (
                "GET",
                f"https://huggingface.co/acme/Nova-7B/resolve/{sha}/config.json",
            ): Response(status=404, headers={}, body=b"{}"),
        }
    )
    old_architecture = Architecture(
        source_repo="other-org/Other-7B",
        source_revision="b" * 40,
        kind="dense_classic",
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    old_other_spec = BaseModelSpec(
        hf_repo="other-org/Other-7B", repo_aliases=[], publisher="other-org", parameters_b=7.0, architecture=old_architecture
    )
    old_snapshot = Snapshot(schema_version=1, run_at=RUN1, areas=[], base_models=[old_other_spec], packages=[])

    result = run_fetch(config, transport, RUN2, old_snapshot=old_snapshot, budget=2)

    assert result.request_used == 2  # never touched other-org/Other-7B at all
    other_spec = next(bm for bm in result.snapshot.base_models if bm.hf_repo == "other-org/Other-7B")
    assert other_spec == old_other_spec
    other_areas = [a for a in result.snapshot.areas if a.base_model_hf_repo == "other-org/Other-7B"]
    assert other_areas, "the never-reached base model must still get an incomplete area entry"
    assert all(a.status == "incomplete" for a in other_areas)
    assert all(a.error == "budget exhausted before this area was started" for a in other_areas)
