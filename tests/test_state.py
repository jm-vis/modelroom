"""Tests for modelroom.state: the lock file, the snapshot merge rule, and atomic writes.

Real filesystem, real `datetime` values passed in explicitly (never `datetime.now()` inside
the code under test) -- no mocking, per AGENTS.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.contracts import (
    Architecture,
    Area,
    BaseModelSpec,
    HardwareSnapshot,
    Package,
    PackageFile,
    SchemaVersionError,
    Snapshot,
)
from modelroom.fetch_types import AreaOutcome
from modelroom.state import (
    LockHeldError,
    StaleRunError,
    acquire_lock,
    atomic_write_json,
    check_run_is_newer,
    hardware_snapshot_path,
    load_existing_hardware_snapshot,
    load_existing_snapshot,
    merge_snapshot,
    release_lock,
    write_hardware_snapshot,
    write_run_status,
    write_snapshot,
)

RUN1 = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
RUN2 = datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc)


def _unknown_architecture(source_repo: str) -> Architecture:
    return Architecture(source_repo=source_repo, source_revision=None, kind="unknown")


def _base_model(**overrides) -> BaseModelSpec:
    defaults = dict(
        hf_repo="acme/Nova-7B",
        repo_aliases=["Nova-7B-GGUF"],
        publisher="acme",
        parameters_b=7.0,
        architecture=_unknown_architecture("acme/Nova-7B"),
    )
    defaults.update(overrides)
    return BaseModelSpec(**defaults)


def _hf_package(*, repo: str, filename: str, observed_at: datetime, last_seen: datetime, active: bool = True) -> Package:
    return Package(
        source="huggingface",
        repo=repo,
        revision="a" * 40,
        base_model_hf_repo="acme/Nova-7B",
        format="gguf",
        files=[PackageFile(name=filename, role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization=None,
        default_context=None,
        provenance="unresolved",
        unresolved_reason="repo_name",
        approval=None,
        observed_at=observed_at,
        last_seen=last_seen,
        active=active,
    )


# --- lock -----------------------------------------------------------------------------------


def test_acquire_lock_writes_pid_command_started_at_and_a_token(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"

    token = acquire_lock(lock_path, "fetch", RUN1)

    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["command"] == "fetch"
    assert data["started_at"] == RUN1.isoformat()
    assert data["token"] == token
    assert token  # non-empty


def test_acquire_lock_raises_when_held_by_a_young_lock(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"
    acquire_lock(lock_path, "fetch", RUN1)

    with pytest.raises(LockHeldError):
        acquire_lock(lock_path, "fetch", RUN1 + timedelta(minutes=5))


def test_acquire_lock_is_not_a_check_then_write_race_two_calls_get_different_tokens(tmp_path: Path):
    # F1: a second acquire_lock call against the same path, in the same process, at the same
    # instant, must never both succeed with the file simply overwritten -- the second call
    # finds the first call's still-young lock (via the atomic O_CREAT|O_EXCL create) and raises.
    lock_path = tmp_path / "modelroom.lock"
    first_token = acquire_lock(lock_path, "fetch", RUN1)

    with pytest.raises(LockHeldError):
        acquire_lock(lock_path, "fetch", RUN1)

    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["token"] == first_token  # the second call never overwrote the first


def test_acquire_lock_overwrites_a_stale_lock_older_than_two_hours(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"
    acquire_lock(lock_path, "fetch", RUN1)

    new_token = acquire_lock(lock_path, "fetch", RUN1 + timedelta(hours=2, minutes=1))

    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["started_at"] == (RUN1 + timedelta(hours=2, minutes=1)).isoformat()
    assert data["token"] == new_token  # the takeover's content is verified to be ours


def test_release_lock_removes_the_file(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"
    token = acquire_lock(lock_path, "fetch", RUN1)

    release_lock(lock_path, token)

    assert not lock_path.exists()


def test_release_lock_is_a_no_op_when_no_lock_exists(tmp_path: Path):
    release_lock(tmp_path / "modelroom.lock", "some-token")  # must not raise


def test_release_lock_with_a_foreign_token_does_not_delete(tmp_path: Path):
    # F1: release_lock must remove the file only when its content carries the caller's own
    # token -- a foreign token (another process's lock, or a stale takeover) is left alone.
    lock_path = tmp_path / "modelroom.lock"
    acquire_lock(lock_path, "fetch", RUN1)

    release_lock(lock_path, "not-the-real-token")

    assert lock_path.exists()


def test_release_lock_with_a_foreign_token_leaves_no_leftover_release_claim_file(tmp_path: Path):
    # R1: release_lock's atomic claim (os.replace(lock, lock.release.<token>)) must always give
    # the file back under its original name when the token does not match -- never leave the
    # renamed claim file lying around.
    lock_path = tmp_path / "modelroom.lock"
    real_token = acquire_lock(lock_path, "fetch", RUN1)

    release_lock(lock_path, "not-the-real-token")

    assert lock_path.exists()
    assert list(tmp_path.glob(f"{lock_path.name}.release.*")) == []
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["token"] == real_token


_RACE_SCRIPT = """
import sys, time
from pathlib import Path
from datetime import datetime
from modelroom.state import acquire_lock, LockHeldError

lock_path = Path(sys.argv[1])
go_path = Path(sys.argv[2])
now = datetime.fromisoformat(sys.argv[3])

while not go_path.exists():
    time.sleep(0.001)

try:
    won_token = acquire_lock(lock_path, "fetch", now)
    print(f"acquired:{won_token}")
except LockHeldError:
    print("held")
"""


def test_acquire_lock_stale_takeover_is_exclusive_across_real_processes(tmp_path: Path):
    # R1: two real processes race the takeover of the same stale lock. The old
    # write-then-reread implementation could let both see their own token (a race between one
    # process's os.replace and its own re-read, with the other process's replace landing in
    # between); the atomic rename-based claim must never let more than one process win.
    script_path = tmp_path / "race_acquire.py"
    script_path.write_text(_RACE_SCRIPT, encoding="utf-8")
    stale_started = (RUN1 - timedelta(hours=3)).isoformat()

    for i in range(5):
        lock_path = tmp_path / f"race-{i}.lock"
        go_path = tmp_path / f"go-{i}"
        lock_path.write_text(
            json.dumps({"pid": 999999, "command": "fetch", "started_at": stale_started, "token": "stale-token"}),
            encoding="utf-8",
        )

        procs = [
            subprocess.Popen(
                [sys.executable, str(script_path), str(lock_path), str(go_path), RUN1.isoformat()],
                stdout=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        go_path.write_text("go", encoding="utf-8")
        outputs = [proc.communicate(timeout=10)[0].strip() for proc in procs]

        acquired = [line for line in outputs if line.startswith("acquired:")]
        held = [line for line in outputs if line == "held"]
        assert len(acquired) == 1, f"round {i}: expected exactly one 'acquired', got {outputs!r}"
        assert len(held) == 1, f"round {i}: expected exactly one 'held', got {outputs!r}"

        winner_token = acquired[0].split(":", 1)[1]
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        assert data["token"] == winner_token
        # no leftover claim files from either process's takeover attempt
        assert list(tmp_path.glob(f"{lock_path.name}.stale.*")) == []


def test_lock_is_released_after_an_exception_in_the_caller(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"

    with pytest.raises(RuntimeError):
        token = acquire_lock(lock_path, "fetch", RUN1)
        try:
            raise RuntimeError("boom")
        finally:
            release_lock(lock_path, token)

    assert not lock_path.exists()


# --- atomic writes ----------------------------------------------------------------------


def test_atomic_write_json_leaves_no_tmp_file_on_success(tmp_path: Path):
    target = tmp_path / "state" / "modelroom.json"

    atomic_write_json(target, {"a": 1})

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert list(tmp_path.rglob("*.tmp")) == []


def test_atomic_write_json_leaves_no_tmp_file_on_a_serialization_failure(tmp_path: Path):
    target = tmp_path / "state" / "modelroom.json"

    with pytest.raises(TypeError):
        atomic_write_json(target, {"a": {1, 2, 3}})  # a set is not JSON serializable

    assert not target.exists()
    assert list(tmp_path.rglob("*.tmp")) == []


# --- write_snapshot / load_existing_snapshot ---------------------------------------------


def test_write_snapshot_then_load_existing_snapshot_round_trips(tmp_path: Path):
    from modelroom.config import Configuration

    config = Configuration.from_dict(
        {
            "schema_version": 1,
            "families": [{"name": "nova", "base_models": [{"hf_repo": "acme/Nova-7B", "repo_aliases": []}]}],
            "packagers": ["packager"],
            "publishers": ["acme"],
            "paths": {"state": str(tmp_path), "markdown": str(tmp_path / "models.md")},
        }
    )
    snapshot = Snapshot(schema_version=1, run_at=RUN1, areas=[], base_models=[_base_model()], packages=[])

    write_snapshot(config, snapshot.model_dump(mode="json"))
    loaded = load_existing_snapshot(config)

    assert loaded is not None
    assert loaded.run_at == RUN1
    assert loaded.base_models[0].hf_repo == "acme/Nova-7B"
    assert list((tmp_path).rglob("*.tmp")) == []


def test_load_existing_snapshot_returns_none_when_no_file_exists(tmp_path: Path):
    from modelroom.config import Configuration

    config = Configuration.from_dict(
        {
            "schema_version": 1,
            "families": [{"name": "nova", "base_models": [{"hf_repo": "acme/Nova-7B", "repo_aliases": []}]}],
            "packagers": ["packager"],
            "publishers": ["acme"],
            "paths": {"state": str(tmp_path), "markdown": str(tmp_path / "models.md")},
        }
    )
    assert load_existing_snapshot(config) is None


# --- check_run_is_newer -------------------------------------------------------------------


def test_check_run_is_newer_passes_when_no_old_snapshot():
    check_run_is_newer(None, RUN1)  # must not raise


def test_check_run_is_newer_passes_for_a_strictly_newer_run():
    old = Snapshot(schema_version=1, run_at=RUN1, areas=[], base_models=[], packages=[])
    check_run_is_newer(old, RUN2)  # must not raise


def test_check_run_is_newer_raises_for_an_older_run():
    old = Snapshot(schema_version=1, run_at=RUN2, areas=[], base_models=[], packages=[])
    with pytest.raises(StaleRunError):
        check_run_is_newer(old, RUN1)


def test_check_run_is_newer_raises_for_the_same_instant():
    old = Snapshot(schema_version=1, run_at=RUN1, areas=[], base_models=[], packages=[])
    with pytest.raises(StaleRunError):
        check_run_is_newer(old, RUN1)


# --- merge_snapshot: areas and packages ---------------------------------------------------


def test_merge_snapshot_first_run_has_no_old_snapshot():
    outcome = AreaOutcome(
        source="huggingface",
        base_model_hf_repo="acme/Nova-7B",
        packager="packager",
        status="complete",
        error=None,
        packages=[_hf_package(repo="packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", observed_at=RUN1, last_seen=RUN1)],
    )

    snapshot = merge_snapshot(None, RUN1, [outcome], [_base_model()])

    assert snapshot.run_at == RUN1
    assert len(snapshot.packages) == 1
    assert snapshot.packages[0].active is True
    assert snapshot.areas[0].status == "complete"
    assert snapshot.areas[0].last_success == RUN1


def test_merge_snapshot_complete_area_deactivates_a_package_that_disappeared():
    old_package = _hf_package(repo="packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", observed_at=RUN1, last_seen=RUN1)
    old = Snapshot(
        schema_version=1,
        run_at=RUN1,
        areas=[
            Area(source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", last_success=RUN1, error=None)
        ],
        base_models=[_base_model()],
        packages=[old_package],
    )
    outcome = AreaOutcome(
        source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", error=None, packages=[]
    )

    snapshot = merge_snapshot(old, RUN2, [outcome], [_base_model()])

    assert len(snapshot.packages) == 1
    assert snapshot.packages[0].active is False
    assert snapshot.packages[0].observed_at == RUN1  # observed_at is never bumped
    assert snapshot.areas[0].status == "complete"
    assert snapshot.areas[0].last_success == RUN2


def test_merge_snapshot_complete_area_keeps_observed_at_and_bumps_last_seen_for_a_still_present_package():
    old_package = _hf_package(repo="packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", observed_at=RUN1, last_seen=RUN1)
    old = Snapshot(
        schema_version=1,
        run_at=RUN1,
        areas=[
            Area(source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", last_success=RUN1, error=None)
        ],
        base_models=[_base_model()],
        packages=[old_package],
    )
    fresh_package = _hf_package(repo="packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", observed_at=RUN2, last_seen=RUN2)
    outcome = AreaOutcome(
        source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", error=None, packages=[fresh_package]
    )

    snapshot = merge_snapshot(old, RUN2, [outcome], [_base_model()])

    assert len(snapshot.packages) == 1
    assert snapshot.packages[0].observed_at == RUN1
    assert snapshot.packages[0].last_seen == RUN2
    assert snapshot.packages[0].active is True


def test_merge_snapshot_incomplete_area_keeps_old_packages_untouched_and_records_the_error():
    old_package = _hf_package(repo="packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", observed_at=RUN1, last_seen=RUN1)
    old = Snapshot(
        schema_version=1,
        run_at=RUN1,
        areas=[
            Area(source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", last_success=RUN1, error=None)
        ],
        base_models=[_base_model()],
        packages=[old_package],
    )
    outcome = AreaOutcome(
        source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="incomplete", error="tree fetch failed", packages=[]
    )

    snapshot = merge_snapshot(old, RUN2, [outcome], [_base_model()])

    assert snapshot.packages == [old_package]
    assert snapshot.areas[0].status == "incomplete"
    assert snapshot.areas[0].error == "tree fetch failed"
    assert snapshot.areas[0].last_success == RUN1  # the OLD last_success, not bumped


def test_merge_snapshot_drops_packages_of_an_area_no_longer_configured():
    # F2: a base model (or area) removed from this run's configuration must not leave orphan
    # packages behind -- only carry an old package over when its area is among this run's
    # area_outcomes. An incomplete area's old packages are still kept untouched.
    kept_package = _hf_package(repo="packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", observed_at=RUN1, last_seen=RUN1)
    removed_package = Package(
        source="huggingface",
        repo="packager/Other-7B-GGUF",
        revision="b" * 40,
        base_model_hf_repo="acme/Other-7B",
        format="gguf",
        files=[PackageFile(name="Other-7B-Q4_K_M.gguf", role="weights", size_bytes=1, digest=None)],
        complete=True,
        quantization=None,
        default_context=None,
        provenance="unresolved",
        unresolved_reason="repo_name",
        approval=None,
        observed_at=RUN1,
        last_seen=RUN1,
        active=True,
    )
    incomplete_area_package = _hf_package(
        repo="other-packager/Nova-7B-GGUF", filename="Nova-7B-BF16.gguf", observed_at=RUN1, last_seen=RUN1
    )
    old = Snapshot(
        schema_version=1,
        run_at=RUN1,
        areas=[
            Area(source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", last_success=RUN1, error=None),
            Area(source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="other-packager", status="complete", last_success=RUN1, error=None),
            Area(source="huggingface", base_model_hf_repo="acme/Other-7B", packager="packager", status="complete", last_success=RUN1, error=None),
        ],
        base_models=[_base_model(), _base_model(hf_repo="acme/Other-7B")],
        packages=[kept_package, incomplete_area_package, removed_package],
    )
    # This run configures acme/Nova-7B again, under both packager areas (one complete, one now
    # failing) -- acme/Other-7B is no longer configured at all this run.
    outcomes = [
        AreaOutcome(source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", error=None, packages=[kept_package]),
        AreaOutcome(source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="other-packager", status="incomplete", error="tree fetch failed", packages=[]),
    ]

    snapshot = merge_snapshot(old, RUN2, outcomes, [_base_model()])

    assert isinstance(snapshot, Snapshot)
    result_keys = {(p.repo, p.base_model_hf_repo) for p in snapshot.packages}
    assert result_keys == {
        ("packager/Nova-7B-GGUF", "acme/Nova-7B"),
        ("other-packager/Nova-7B-GGUF", "acme/Nova-7B"),
    }


def test_merge_snapshot_base_models_are_always_replaced_by_this_runs_results():
    old = Snapshot(schema_version=1, run_at=RUN1, areas=[], base_models=[_base_model(parameters_b=1.0)], packages=[])

    snapshot = merge_snapshot(old, RUN2, [], [_base_model(parameters_b=99.0)])

    assert snapshot.base_models[0].parameters_b == 99.0


def test_merge_snapshot_result_validates_as_a_snapshot():
    old_package = _hf_package(repo="packager/Nova-7B-GGUF", filename="Nova-7B-Q4_K_M.gguf", observed_at=RUN1, last_seen=RUN1)
    outcome = AreaOutcome(
        source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", error=None, packages=[old_package]
    )
    snapshot = merge_snapshot(None, RUN1, [outcome], [_base_model()])
    assert isinstance(snapshot, Snapshot)


# --- write_run_status ---------------------------------------------------------------------


def test_write_run_status_lists_every_area_with_run_at_budget_and_candidates(tmp_path: Path):
    from modelroom.config import Configuration

    config = Configuration.from_dict(
        {
            "schema_version": 1,
            "families": [{"name": "nova", "base_models": [{"hf_repo": "acme/Nova-7B", "repo_aliases": []}]}],
            "packagers": ["packager"],
            "publishers": ["acme"],
            "paths": {"state": str(tmp_path), "markdown": str(tmp_path / "models.md")},
        }
    )
    snapshot = Snapshot(
        schema_version=1,
        run_at=RUN1,
        areas=[
            Area(source="huggingface", base_model_hf_repo="acme/Nova-7B", packager="packager", status="complete", last_success=RUN1, error=None),
            Area(source="ollama", base_model_hf_repo="acme/Nova-7B", packager=None, status="incomplete", last_success=None, error="no tags parsed"),
        ],
        base_models=[_base_model()],
        packages=[],
    )

    write_run_status(config, snapshot, request_used=12, request_budget=400)

    data = json.loads(config.paths.run_status_file.read_text(encoding="utf-8"))
    assert data["run_at"] == RUN1.isoformat()
    assert data["request_budget"] == {"used": 12, "total": 400}
    assert data["candidates"] == []
    assert len(data["areas"]) == 2
    assert {area["status"] for area in data["areas"]} == {"complete", "incomplete"}


# --- hardware snapshot: path, load, write (AP4) --------------------------------------------


def _config(tmp_path: Path):
    from modelroom.config import Configuration

    return Configuration.from_dict(
        {
            "schema_version": 1,
            "families": [{"name": "nova", "base_models": [{"hf_repo": "acme/Nova-7B", "repo_aliases": []}]}],
            "packagers": ["packager"],
            "publishers": ["acme"],
            "machines": {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}},
            "paths": {"state": str(tmp_path), "markdown": str(tmp_path / "models.md")},
        }
    )


def _hardware_snapshot_dict(**overrides) -> dict:
    data = {
        "schema_version": 1,
        "machine": "workstation",
        "measured_at": RUN1.isoformat(),
        "llmfit_version": "1.1.16",
        "vram_gib": 11.94,
        "ram_gib": 127.46,
        "free_ram_gib_at_measurement": 76.64,
        "gpu_name": "Nova GPU",
        "backend": "CUDA",
        "unified_memory": False,
        "installed": None,
        "installed_unavailable_reason": "daemon unreachable",
        "measurements": [],
    }
    data.update(overrides)
    return data


def test_hardware_snapshot_path_is_under_the_hardware_subdir(tmp_path: Path):
    config = _config(tmp_path)
    assert hardware_snapshot_path(config, "workstation") == config.paths.hardware_dir / "workstation.json"


def test_load_existing_hardware_snapshot_returns_none_when_no_file_exists(tmp_path: Path):
    config = _config(tmp_path)
    assert load_existing_hardware_snapshot(config, "workstation") is None


def test_write_hardware_snapshot_then_load_round_trips(tmp_path: Path):
    config = _config(tmp_path)

    written = write_hardware_snapshot(config, "workstation", _hardware_snapshot_dict())
    loaded = load_existing_hardware_snapshot(config, "workstation")

    assert isinstance(written, HardwareSnapshot)
    assert loaded is not None
    assert loaded.machine == "workstation"
    assert loaded.vram_gib == 11.94
    assert list(config.paths.hardware_dir.glob("*.tmp")) == []


def test_write_hardware_snapshot_is_scoped_to_its_own_machine_file(tmp_path: Path):
    config = _config(tmp_path)
    write_hardware_snapshot(config, "workstation", _hardware_snapshot_dict())

    assert load_existing_hardware_snapshot(config, "inference-server") is None
    assert (config.paths.hardware_dir / "workstation.json").exists()
    assert not (config.paths.hardware_dir / "inference-server.json").exists()


def test_load_existing_hardware_snapshot_raises_schema_version_error(tmp_path: Path):
    config = _config(tmp_path)
    path = hardware_snapshot_path(config, "workstation")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    with pytest.raises(SchemaVersionError):
        load_existing_hardware_snapshot(config, "workstation")


def test_write_hardware_snapshot_validates_before_writing(tmp_path: Path):
    config = _config(tmp_path)
    bad = _hardware_snapshot_dict(ram_gib=-1.0)  # ram_gib must be > 0

    with pytest.raises(ValidationError):
        write_hardware_snapshot(config, "workstation", bad)

    assert not hardware_snapshot_path(config, "workstation").exists()
