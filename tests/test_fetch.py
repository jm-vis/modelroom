"""Tests for modelroom.fetch: orchestrating hf.py + ollama.py into one merged Snapshot."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone

from modelroom.config import Configuration
from modelroom.contracts import Approval, Architecture, BaseModelSpec, Package, PackageFile, Snapshot, load_snapshot
from modelroom.fetch import run_fetch
from modelroom.http import RequestBudget, Response
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


# --- R2: malformed base-model metadata must never escape run_fetch -----------------------


def test_run_fetch_zero_safetensors_total_returns_instead_of_raising(tmp_path):
    config = _config(tmp_path)
    mapping = qwen35_transport_mapping()
    mapping[("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B")] = Response(
        status=200, headers={}, body=b'{"sha": "' + b"a" * 40 + b'", "safetensors": {"total": 0}}'
    )
    mapping[("GET", f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{'a' * 40}/config.json")] = Response(
        status=404, headers={}, body=b"{}"
    )
    transport = build_transport(mapping)

    result = run_fetch(config, transport, RUN1, old_snapshot=None)  # must not raise

    assert result.snapshot.base_models[0].parameters_b is None
    assert isinstance(load_snapshot(result.snapshot.model_dump(mode="json")), Snapshot)


def test_run_fetch_subnormal_safetensors_total_returns_instead_of_raising(tmp_path):
    # Fix-round 3: 1e-320 passes "> 0" but underflows to 0.0 once divided by 1e9, which
    # BaseModelSpec.parameters_b's gt=0 constraint would reject if it ever reached fetch.py.
    config = _config(tmp_path)
    mapping = qwen35_transport_mapping()
    mapping[("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B")] = Response(
        status=200, headers={}, body=b'{"sha": "' + b"a" * 40 + b'", "safetensors": {"total": 1e-320}}'
    )
    mapping[("GET", f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{'a' * 40}/config.json")] = Response(
        status=404, headers={}, body=b"{}"
    )
    transport = build_transport(mapping)

    result = run_fetch(config, transport, RUN1, old_snapshot=None)  # must not raise

    assert result.snapshot.base_models[0].parameters_b is None
    assert isinstance(load_snapshot(result.snapshot.model_dump(mode="json")), Snapshot)


def test_run_fetch_huge_json_integer_safetensors_total_returns_instead_of_raising(tmp_path):
    # Fix-round 3: a JSON integer far outside float range overflows the "/ 1e9" division itself.
    config = _config(tmp_path)
    huge_total = str(10**400).encode()
    mapping = qwen35_transport_mapping()
    mapping[("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B")] = Response(
        status=200,
        headers={},
        body=b'{"sha": "' + b"a" * 40 + b'", "safetensors": {"total": ' + huge_total + b"}}",
    )
    mapping[("GET", f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{'a' * 40}/config.json")] = Response(
        status=404, headers={}, body=b"{}"
    )
    transport = build_transport(mapping)

    result = run_fetch(config, transport, RUN1, old_snapshot=None)  # must not raise

    assert result.snapshot.base_models[0].parameters_b is None
    assert isinstance(load_snapshot(result.snapshot.model_dump(mode="json")), Snapshot)


def test_run_fetch_malformed_sha_returns_instead_of_raising(tmp_path):
    config = _config(tmp_path)
    mapping = qwen35_transport_mapping()
    mapping[("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B")] = Response(
        status=200, headers={}, body=b'{"sha": "bad"}'
    )
    transport = build_transport(mapping)

    result = run_fetch(config, transport, RUN1, old_snapshot=None)  # must not raise

    assert result.snapshot.base_models[0].architecture.kind == "unknown"
    assert isinstance(load_snapshot(result.snapshot.model_dump(mode="json")), Snapshot)


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


# --- AP9-K: a budget object shared with the caller's own requests ---------------------------


def test_run_fetch_books_against_a_shared_budget_the_caller_already_spent_on(tmp_path):
    config = _config(tmp_path)
    fresh = run_fetch(config, build_transport(qwen35_transport_mapping()), RUN1, old_snapshot=None)
    shared = RequestBudget(400)
    for _ in range(10):  # e.g. the guided search's own requests
        shared.charge()

    result = run_fetch(config, build_transport(qwen35_transport_mapping()), RUN1, old_snapshot=None, budget=shared)

    assert result.request_budget == 400
    assert result.request_used == 10 + fresh.request_used
    assert shared.used == result.request_used


def test_run_fetch_ends_where_the_shared_budget_ends(tmp_path):
    config = _config(tmp_path)
    shared = RequestBudget(3)
    shared.charge()

    result = run_fetch(config, build_transport(qwen35_transport_mapping()), RUN1, old_snapshot=None, budget=shared)

    assert shared.used == 3 and shared.remaining == 0
    assert result.request_used == 3
    assert any("budget exhausted" in (area.error or "") for area in result.snapshot.areas)


# --- R4: the budget is checked before each base model AND before each area -----------------


def test_run_fetch_budget_zero_from_the_start_never_touches_any_base_model(tmp_path):
    config = Configuration.from_dict(
        {
            "schema_version": 1,
            "families": [{"name": "nova", "base_models": [{"hf_repo": "acme/Nova-7B", "repo_aliases": []}]}],
            "packagers": [],
            "publishers": ["acme"],
            "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "models.md")},
        }
    )
    old_architecture = Architecture(
        source_repo="acme/Nova-7B",
        source_revision="a" * 40,
        kind="dense_classic",
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    old_spec = BaseModelSpec(
        hf_repo="acme/Nova-7B", repo_aliases=[], publisher="acme", parameters_b=7.0, architecture=old_architecture
    )
    old_snapshot = Snapshot(schema_version=1, run_at=RUN1, areas=[], base_models=[old_spec], packages=[])
    transport = build_transport({})  # must never be called

    result = run_fetch(config, transport, RUN2, old_snapshot=old_snapshot, budget=0)

    assert result.request_used == 0
    assert transport.calls == []
    assert result.snapshot.base_models == [old_spec]
    assert result.snapshot.areas, "the never-started area must still be recorded"
    assert all(area.status == "incomplete" for area in result.snapshot.areas)
    assert all(area.error == "budget exhausted before this area was started" for area in result.snapshot.areas)


def test_run_fetch_budget_ending_between_two_areas_of_the_same_base_model_is_not_a_transport_error(tmp_path):
    # R4: the budget must be checked before each area too, not only before each base model --
    # otherwise the second area's fetcher is actually invoked, sees BudgetedTransport raise
    # BudgetExhaustedError mid-call, and records the bare "budget exhausted" message instead of
    # "budget exhausted before this area was started".
    config = Configuration.from_dict(
        {
            "schema_version": 1,
            "families": [
                {
                    "name": "nova",
                    "base_models": [
                        {"hf_repo": "acme/Nova-7B", "repo_aliases": [], "ollama_base": "nova", "ollama_tag": "7b"}
                    ],
                }
            ],
            "packagers": [],
            "publishers": ["acme"],
            "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "models.md")},
        }
    )
    transport = build_transport(
        {
            ("GET", "https://huggingface.co/api/models/acme/Nova-7B"): Response(status=404, headers={}, body=b"{}"),
            ("GET", "https://huggingface.co/api/models/acme/Nova-7B-GGUF"): Response(
                status=404, headers={}, body=b"{}"
            ),
        }
    )

    # budget: 1 for fetch_base_model_meta's model-info call, 1 for the HF area's only candidate
    # (also 404) -- exhausted exactly as the HF area finishes, before the Ollama area starts.
    result = run_fetch(config, transport, RUN1, old_snapshot=None, budget=2)

    assert result.request_used == 2
    statuses = {(area.source, area.packager): area for area in result.snapshot.areas}
    assert statuses[("huggingface", "acme")].status == "complete"
    ollama_area = statuses[("ollama", None)]
    assert ollama_area.status == "incomplete"
    assert ollama_area.error == "budget exhausted before this area was started"


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


# --- AP9-B: the target set, grouped by owner, is one area per (base model, owner) ----------


def _two_repo_config(tmp_path) -> Configuration:
    """One base model with two packager repos under the *same* owner, plus one under another."""
    return Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [
                {
                    "name": "nova",
                    "base_models": [
                        {
                            "hf_repo": "acme/Nova-7B",
                            "repos": ["packager/Nova-7B-GGUF", "packager/Nova-7B-i1-GGUF"],
                        }
                    ],
                }
            ],
            "packagers": [],
            "publishers": ["acme"],
            "machines": {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}},
            "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "models.md")},
        }
    )


_SHA_A = "a" * 40
_SHA_B = "b" * 40


def _repo_info(sha: str, base: str = "acme/Nova-7B") -> Response:
    payload = {
        "sha": sha,
        "tags": [f"base_model:{base}", f"base_model:quantized:{base}"],
        "cardData": {"base_model": [base]},
    }
    return Response(status=200, headers={}, body=json.dumps(payload).encode("utf-8"))


def _tree(filename: str) -> Response:
    entry = [{"type": "file", "path": filename, "size": 4_000_000_000}]
    return Response(status=200, headers={}, body=json.dumps(entry).encode("utf-8"))


def _two_repo_mapping() -> dict:
    return {
        ("GET", "https://huggingface.co/api/models/acme/Nova-7B"): Response(
            status=200, headers={}, body=b'{"sha": "' + _SHA_A.encode() + b'"}'
        ),
        ("GET", f"https://huggingface.co/acme/Nova-7B/resolve/{_SHA_A}/config.json"): Response(
            status=404, headers={}, body=b"{}"
        ),
        ("GET", "https://huggingface.co/api/models/packager/Nova-7B-GGUF"): _repo_info(_SHA_A),
        (
            "GET",
            f"https://huggingface.co/api/models/packager/Nova-7B-GGUF/tree/{_SHA_A}?recursive=true",
        ): _tree("Nova-7B-Q4_K_M.gguf"),
        ("GET", "https://huggingface.co/api/models/packager/Nova-7B-i1-GGUF"): _repo_info(_SHA_B),
        (
            "GET",
            f"https://huggingface.co/api/models/packager/Nova-7B-i1-GGUF/tree/{_SHA_B}?recursive=true",
        ): _tree("Nova-7B-Q4_K_S.gguf"),
        ("GET", "https://huggingface.co/api/models/acme/Nova-7B-GGUF"): Response(
            status=401, headers={}, body=b"{}"
        ),
    }


def test_two_repos_of_one_owner_share_a_single_area_and_both_stay_active(tmp_path):
    config = _two_repo_config(tmp_path)

    result = run_fetch(config, build_transport(_two_repo_mapping()), RUN1, old_snapshot=None)

    hf_areas = [area for area in result.snapshot.areas if area.source == "huggingface"]
    packager_areas = [area for area in hf_areas if area.packager == "packager"]
    assert len(packager_areas) == 1
    assert packager_areas[0].status == "complete"
    packages = [p for p in result.snapshot.packages if p.repo and p.repo.startswith("packager/")]
    assert {p.repo for p in packages} == {"packager/Nova-7B-GGUF", "packager/Nova-7B-i1-GGUF"}
    assert all(p.active for p in packages)
    assert all(p.provenance == "metadata_ok" for p in packages)


def test_a_repos_target_is_fetched_even_though_its_owner_is_no_configured_packager(tmp_path):
    config = _two_repo_config(tmp_path)

    result = run_fetch(config, build_transport(_two_repo_mapping()), RUN1, old_snapshot=None)

    assert config.packagers == []
    assert any(area.packager == "packager" for area in result.snapshot.areas)


def test_a_failure_on_the_second_target_leaves_the_owners_whole_area_incomplete(tmp_path):
    config = _two_repo_config(tmp_path)
    first = run_fetch(config, build_transport(_two_repo_mapping()), RUN1, old_snapshot=None)
    broken = _two_repo_mapping()
    broken[("GET", "https://huggingface.co/api/models/packager/Nova-7B-i1-GGUF")] = Response(
        status=500, headers={}, body=b""
    )

    result = run_fetch(config, build_transport(broken), RUN2, old_snapshot=first.snapshot)

    area = next(a for a in result.snapshot.areas if a.packager == "packager")
    assert area.status == "incomplete"
    assert "500" in (area.error or "")
    # The whole owner's previous stock survives *byte for byte*, both repos included -- not just
    # the repo names and the active flag: an `incomplete` area must not refresh `last_seen` or
    # any other field of a package it could not read this run.
    before = {p.repo: p.model_dump(mode="json") for p in first.snapshot.packages}
    after = {p.repo: p.model_dump(mode="json") for p in result.snapshot.packages}
    assert after == before


def test_a_target_that_disappears_is_deactivated_when_the_owners_area_completes(tmp_path):
    config = _two_repo_config(tmp_path)
    first = run_fetch(config, build_transport(_two_repo_mapping()), RUN1, old_snapshot=None)
    gone = _two_repo_mapping()
    gone[("GET", "https://huggingface.co/api/models/packager/Nova-7B-i1-GGUF")] = Response(
        status=401, headers={}, body=b"{}"
    )

    result = run_fetch(config, build_transport(gone), RUN2, old_snapshot=first.snapshot)

    by_repo = {p.repo: p for p in result.snapshot.packages}
    assert by_repo["packager/Nova-7B-GGUF"].active is True
    assert by_repo["packager/Nova-7B-i1-GGUF"].active is False


def test_a_budget_that_runs_out_between_two_targets_of_one_owner_keeps_the_previous_stock(tmp_path):
    # The budget is a whole-run resource, so running out at the second target of an owner is a
    # failure of that owner's *area*: `incomplete`, the error exactly "budget exhausted", and the
    # previous stock of both targets untouched -- not a half-read area published as complete.
    config = _two_repo_config(tmp_path)
    first = run_fetch(config, build_transport(_two_repo_mapping()), RUN1, old_snapshot=None)
    # 2 for the base model (info + config.json), 1 for the not-found own-owner candidate,
    # 2 for the first target (info + tree), then nothing left for the second target.
    budget = RequestBudget(5)

    result = run_fetch(
        config, build_transport(_two_repo_mapping()), RUN2, old_snapshot=first.snapshot, budget=budget
    )

    area = next(a for a in result.snapshot.areas if a.packager == "packager")
    assert area.status == "incomplete"
    assert area.error == "budget exhausted"
    before = {p.repo: p.model_dump(mode="json") for p in first.snapshot.packages}
    after = {p.repo: p.model_dump(mode="json") for p in result.snapshot.packages}
    assert after == before
    assert budget.remaining == 0


def test_a_budget_that_runs_out_before_an_owner_names_that_owners_area(tmp_path):
    config = _two_repo_config(tmp_path)

    result = run_fetch(config, build_transport(_two_repo_mapping()), RUN1, old_snapshot=None, budget=0)

    owners = {area.packager for area in result.snapshot.areas if area.source == "huggingface"}
    assert owners == {"packager", "acme"}
    assert all(
        area.error == "budget exhausted before this area was started" for area in result.snapshot.areas
    )
