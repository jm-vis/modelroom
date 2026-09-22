"""Tests for modelroom.fetch: orchestrating hf.py + ollama.py into one merged Snapshot."""

from __future__ import annotations

from datetime import datetime, timezone

from modelroom.config import Configuration
from modelroom.contracts import Architecture, BaseModelSpec, Snapshot
from modelroom.fetch import run_fetch
from modelroom.http import Response

from fixture_support import build_transport, envelope_response, qwen35_example_config_dict, qwen35_transport_mapping

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


def test_run_fetch_parameters_b_falls_back_to_a_placeholder_with_no_prior_snapshot(tmp_path):
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

    assert result.snapshot.base_models[0].parameters_b == 1.0


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


def test_run_fetch_stops_making_requests_once_the_budget_is_exhausted(tmp_path):
    config = _config(tmp_path)
    transport = build_transport(qwen35_transport_mapping())

    result = run_fetch(config, transport, RUN1, old_snapshot=None, budget=2)

    assert result.request_used == 2
    assert result.request_budget == 2
    assert any(area.status == "incomplete" for area in result.snapshot.areas)
    assert any("budget exhausted" in (area.error or "") for area in result.snapshot.areas)
