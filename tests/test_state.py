"""Tests for modelroom.state: the lock file, the snapshot merge rule, and atomic writes.

Real filesystem, real `datetime` values passed in explicitly (never `datetime.now()` inside
the code under test) -- no mocking, per AGENTS.md.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
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
#
# Fix-round 3: the lock is a kernel lock on a stable file that is never renamed or deleted (see
# CONTRACTS.md, "Lock file"). `acquire_lock` returns a `LockHandle` (no more token), and
# `release_lock` takes only that handle.


def _read_lock_content(path: Path) -> dict:
    """The lock file's JSON content, skipping the single reserved lock byte at offset 0 (see
    `modelroom/state.py`'s `_LOCK_OFFSET`/`_CONTENT_OFFSET`) -- reading the whole file from
    offset 0 would touch that byte and raise `PermissionError` on Windows while it is locked."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.lseek(fd, 1, 0)
        raw = os.read(fd, 65536)
    finally:
        os.close(fd)
    return json.loads(raw.decode("utf-8"))


def test_acquire_lock_writes_pid_command_and_started_at(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"

    handle = acquire_lock(lock_path, "fetch", RUN1)
    try:
        data = _read_lock_content(lock_path)
        assert data["pid"] == os.getpid()
        assert data["command"] == "fetch"
        assert data["started_at"] == RUN1.isoformat()
        assert "token" not in data
    finally:
        release_lock(handle)


def test_acquire_lock_raises_while_a_handle_is_open(tmp_path: Path):
    # A second acquire_lock call against the same path, while the first handle is still open,
    # must raise -- regardless of how long the first has held it (a kernel lock does not age).
    # This works in-process because `flock` is scoped per open file description and a Windows
    # byte-range lock is scoped per handle, so a second `os.open` of the same path never inherits
    # the first one's lock.
    lock_path = tmp_path / "modelroom.lock"
    handle = acquire_lock(lock_path, "fetch", RUN1)
    try:
        with pytest.raises(LockHeldError, match=str(os.getpid())):
            acquire_lock(lock_path, "fetch", RUN1 + timedelta(hours=100))

        # the failed second attempt never touched our own content -- readable the whole time,
        # since the lock covers only the reserved byte at offset 0, never the content
        data = _read_lock_content(lock_path)
        assert data["started_at"] == RUN1.isoformat()
    finally:
        release_lock(handle)


def test_release_lock_empties_the_file_and_a_new_acquire_succeeds_afterwards(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"
    handle = acquire_lock(lock_path, "fetch", RUN1)

    release_lock(handle)

    assert lock_path.exists()  # the file is never deleted
    assert lock_path.read_text(encoding="utf-8") == ""

    second = acquire_lock(lock_path, "fetch", RUN1 + timedelta(minutes=1))
    try:
        data = _read_lock_content(lock_path)
        assert data["started_at"] == (RUN1 + timedelta(minutes=1)).isoformat()
    finally:
        release_lock(second)


def test_release_lock_is_idempotent(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"
    handle = acquire_lock(lock_path, "fetch", RUN1)

    release_lock(handle)
    release_lock(handle)  # must not raise

    assert lock_path.exists()
    assert handle.released is True


def test_second_release_never_touches_a_file_that_reused_the_descriptor_number(tmp_path: Path):
    # Fix-round 4 (Codex P1): after `release_lock` closes the fd, the OS hands the same descriptor
    # number to the next `os.open` in this process. A second release of the stale handle must not
    # truncate, unlock or close that unrelated file -- it must be a no-op on the handle's own
    # "released" state, never a descriptor operation.
    lock_path = tmp_path / "modelroom.lock"
    other_path = tmp_path / "other.json"
    other_path.write_text('{"keep": "me"}', encoding="utf-8")

    handle = acquire_lock(lock_path, "fetch", RUN1)
    release_lock(handle)

    other_fd = os.open(other_path, os.O_RDWR)
    try:
        assert other_fd == handle.fd  # the descriptor number was reused, so the trap is armed

        release_lock(handle)

        assert other_path.read_text(encoding="utf-8") == '{"keep": "me"}'
        os.write(other_fd, b"")  # still open: a close by the stale release would raise EBADF here
    finally:
        os.close(other_fd)


class _BrokenClock:
    """A `now` whose `isoformat` fails -- the simplest way to make acquire_lock fail *after* it
    already holds the kernel lock, without mocking anything in the module under test."""

    def isoformat(self) -> str:
        raise RuntimeError("clock broken")


def test_acquire_lock_releases_the_kernel_lock_when_writing_the_holder_fails(tmp_path: Path):
    # Fix-round 4 (Codex P2): a failure between winning the lock and returning the handle must
    # not leave the fd open with the lock held -- otherwise every later acquire in this process
    # (and every other process) would be blocked until this process exits.
    lock_path = tmp_path / "modelroom.lock"

    with pytest.raises(RuntimeError, match="clock broken"):
        acquire_lock(lock_path, "fetch", _BrokenClock())  # type: ignore[arg-type]

    handle = acquire_lock(lock_path, "fetch", RUN1)  # would raise LockHeldError if the fd leaked
    release_lock(handle)


def test_lock_is_released_after_an_exception_in_the_caller(tmp_path: Path):
    lock_path = tmp_path / "modelroom.lock"

    with pytest.raises(RuntimeError):
        handle = acquire_lock(lock_path, "fetch", RUN1)
        try:
            raise RuntimeError("boom")
        finally:
            release_lock(handle)

    # the file stays on disk, empty, and a fresh acquire succeeds -- release_lock ran in the
    # finally block despite the exception propagating past it
    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8") == ""
    release_lock(acquire_lock(lock_path, "fetch", RUN1 + timedelta(minutes=1)))


_HOLDER_SCRIPT = """
import sys, time
from pathlib import Path
from datetime import datetime
from modelroom.state import acquire_lock

lock_path = Path(sys.argv[1])
stop_path = Path(sys.argv[2])
now = datetime.fromisoformat(sys.argv[3])

acquire_lock(lock_path, "fetch", now)
print("holding", flush=True)

deadline = time.monotonic() + 10
while not stop_path.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""


def test_acquire_lock_succeeds_after_a_holder_process_is_killed_without_releasing(tmp_path: Path):
    # A real holder process acquires the lock and is then killed (simulating a crash, never
    # releasing) -- this process gets LockHeldError naming the holder's pid while it is alive,
    # and acquires successfully once it is gone: the kernel releases a crashed holder's lock on
    # process exit, however "old" the file's content still looks.
    #
    # The pid to kill, and to match in the error, comes from the lock file's own content, not
    # from `Popen.pid`: on this host `sys.executable` is a `uv`-managed launcher that spawns the
    # real interpreter as its own child, so `holder.pid` names the launcher, not the process that
    # actually calls `acquire_lock` -- killing `holder.pid` alone leaves that real process (and
    # its lock) running. The file's content is always the ground truth for who actually holds
    # the lock, and (by this design's whole point) it stays readable the entire time the lock is
    # held, so reading it costs nothing extra.
    lock_path = tmp_path / "modelroom.lock"
    stop_path = tmp_path / "stop"
    script_path = tmp_path / "holder.py"
    script_path.write_text(_HOLDER_SCRIPT, encoding="utf-8")

    holder = subprocess.Popen(
        [sys.executable, str(script_path), str(lock_path), str(stop_path), RUN1.isoformat()],
        stdout=subprocess.PIPE,
        text=True,
    )
    holder_pid: int | None = None
    try:
        assert holder.stdout.readline().strip() == "holding"
        holder_pid = _read_lock_content(lock_path)["pid"]

        with pytest.raises(LockHeldError, match=str(holder_pid)):
            acquire_lock(lock_path, "fetch", RUN1)
    finally:
        if holder_pid is not None:
            try:
                os.kill(holder_pid, signal.SIGTERM)
            except OSError:
                pass  # already gone
        holder.kill()
        holder.wait(timeout=5)

    # `os.kill`/`TerminateProcess` requests termination but does not itself wait for it to
    # finish -- the OS can still be tearing the killed process down (and releasing its handles,
    # including this lock) for a brief moment after the call returns. Retry on a short deadline
    # rather than assume that teardown is always complete by the time we get here.
    deadline = time.monotonic() + 2
    while True:
        try:
            handle = acquire_lock(lock_path, "fetch", RUN1 + timedelta(minutes=1))
            break
        except LockHeldError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)
    release_lock(handle)


_RACE_SCRIPT = """
import sys, time
from pathlib import Path
from datetime import datetime
from modelroom.state import acquire_lock, release_lock, LockHeldError

lock_path = Path(sys.argv[1])
go_path = Path(sys.argv[2])
release_path = Path(sys.argv[3])
now = datetime.fromisoformat(sys.argv[4])

deadline = time.monotonic() + 10
while not go_path.exists() and time.monotonic() < deadline:
    time.sleep(0.001)

try:
    handle = acquire_lock(lock_path, "fetch", now)
except LockHeldError:
    print("held", flush=True)
else:
    print("acquired", flush=True)
    # hold the lock until the test says so -- exiting here would release it (the kernel drops a
    # lock with its process) and let a slower sibling win legitimately, which is not a race
    deadline = time.monotonic() + 10
    while not release_path.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    release_lock(handle)
"""


def _race_round(script_path: Path, lock_path: Path, go_path: Path, release_path: Path) -> list[str]:
    """Launch three processes that all wait on `go_path`, then race `acquire_lock` against
    `lock_path` the instant it appears. The winner keeps holding until `release_path` exists,
    so every loser's attempt happens while the lock is genuinely held. Returns each process's
    one line of stdout."""
    procs = [
        subprocess.Popen(
            [sys.executable, str(script_path), str(lock_path), str(go_path), str(release_path), RUN1.isoformat()],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    go_path.write_text("go", encoding="utf-8")
    outputs = [proc.stdout.readline().strip() for proc in procs]
    release_path.write_text("release", encoding="utf-8")
    for proc in procs:
        proc.communicate(timeout=10)
    return outputs


def test_acquire_lock_is_exclusive_across_three_real_processes(tmp_path: Path):
    # Three processes wait on a go file, then all call acquire_lock against the same fresh lock
    # path at once; the winner holds the lock until the test releases it. Exactly one process
    # must win in every round, with no retry: a kernel lock has no timing window to allow for.
    # (A first draft of this test let the winner exit right after acquiring, which releases the
    # lock and lets a slower sibling acquire legitimately -- that looked like "two winners" and
    # was misread as a lock-placement problem. It was a test flaw; see CONTRACTS.md, "Lock file".)
    script_path = tmp_path / "race.py"
    script_path.write_text(_RACE_SCRIPT, encoding="utf-8")

    for i in range(5):
        outputs = _race_round(script_path, tmp_path / f"race-{i}.lock", tmp_path / f"go-{i}", tmp_path / f"release-{i}")
        assert sorted(outputs) == ["acquired", "held", "held"], f"round {i}: {outputs!r}"


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
