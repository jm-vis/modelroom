"""Persisted run state for `fetch`: the lock file, the snapshot merge rule, atomic writes.

Every "now" the functions here need is passed in explicitly (never read from the wall clock
inside this module) so tests control time exactly, per AGENTS.md's "no mocking" rule.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path

from .config import Configuration
from .contracts import (
    Area,
    BaseModelSpec,
    HardwareSnapshot,
    Package,
    Snapshot,
    SNAPSHOT_SCHEMA_VERSION,
    load_hardware_snapshot,
    load_snapshot,
)
from .fetch_types import AreaOutcome
from .quantization import package_identity_key

LOCK_MAX_AGE = timedelta(hours=2)

# R1: how many times acquire_lock retries a stale-lock takeover before giving up. A retry is
# only ever needed when this attempt's own claim rename loses a race to a concurrent takeover
# (see acquire_lock's docstring); three attempts is generous headroom over the one retry that
# race can actually cause.
MAX_TAKEOVER_ATTEMPTS = 3


class LockHeldError(Exception):
    """Another process holds the lock and it is not old enough to be considered stale."""


class StaleRunError(Exception):
    """The run being started is not strictly newer than the stored snapshot's `run_at`."""


# --- lock -------------------------------------------------------------------------------


def acquire_lock(path: Path, command: str, now: datetime) -> str:
    """Create the lock file exclusively, or raise `LockHeldError` if a live lock holds it.

    Uses `os.open` with `O_CREAT | O_EXCL` so the create-or-fail is one atomic syscall, never a
    `path.exists()` check followed by a separate write (a window a second process could win).
    On `FileExistsError` the existing holder is read: younger than `LOCK_MAX_AGE` blocks
    immediately with `LockHeldError` -- this project never waits for a lock.

    A stale (or unreadable) holder is taken over by claiming it first: `os.replace(path,
    <path>.stale.<token>)` atomically renames the existing file away -- of two processes racing
    the same stale file, only one rename can ever succeed (the other gets `FileNotFoundError`
    and loops back to retry the `O_EXCL` create, see below), so the claim itself can never be
    won twice (R1: the previous "write our content, then re-read to check whose token survived"
    approach let two processes each see their own token when the other's write landed between
    this process's own write and its re-read -- an atomic rename has no such window). Only the
    process that won the claim then creates the lock fresh with `O_CREAT | O_EXCL` and unlinks
    the claimed file; if that create itself loses a race to a third party, the loop retries (up
    to `MAX_TAKEOVER_ATTEMPTS`) rather than assuming success. Returns the token the caller must
    pass to `release_lock`.
    """
    token = secrets.token_hex(16)
    content = json.dumps(
        {"pid": os.getpid(), "command": command, "started_at": now.isoformat(), "token": token}
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(MAX_TAKEOVER_ATTEMPTS):
        if _create_lock_file(path, content):
            return token

        holder = _read_lock(path)
        if holder is not None and now - holder["started_at"] < LOCK_MAX_AGE:
            raise LockHeldError(
                f"{path}: locked by pid {holder['pid']} running '{holder['command']}' since "
                f"{holder['started_at'].isoformat()} (younger than {LOCK_MAX_AGE})"
            )

        claim_path = path.with_name(f"{path.name}.stale.{token}")
        try:
            os.replace(path, claim_path)
        except FileNotFoundError:
            continue  # another process's takeover claimed (or removed) it first; retry from the top

        try:
            if _create_lock_file(path, content):
                return token
        finally:
            claim_path.unlink(missing_ok=True)
        # Our claim won the rename race but a third party's create won the next one; loop back
        # and let the next O_EXCL attempt see whatever is there now.

    raise LockHeldError(f"{path}: lock takeover did not succeed after {MAX_TAKEOVER_ATTEMPTS} attempts")


def _create_lock_file(path: Path, content: str) -> bool:
    """Create `path` exclusively with `content`, or return `False` if it already exists."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    return True


def _read_lock(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "pid": data["pid"],
            "command": data["command"],
            "started_at": datetime.fromisoformat(data["started_at"]),
            "token": data.get("token"),
        }
    except Exception:
        return None


def release_lock(path: Path, token: str) -> None:
    """Remove the lock file, but only when its content still carries `token`.

    A no-op if the file is already gone. R1: releasing is a claim-then-decide, not a
    check-then-unlink -- `os.replace(path, <path>.release.<token>)` atomically claims whatever
    is currently there (a concurrent stale-lock takeover in `acquire_lock` can never land
    between our read and our unlink, because there is no such window any more). Once claimed,
    its content decides the outcome: if it still carries `token`, it is ours and gets unlinked;
    otherwise -- another process took the lock over as stale while we still held what we
    thought was ours -- it is not ours to delete, so it is given back under its original name.
    """
    claim_path = path.with_name(f"{path.name}.release.{token}")
    try:
        os.replace(path, claim_path)
    except FileNotFoundError:
        return

    holder = _read_lock(claim_path)
    if holder is not None and holder["token"] == token:
        claim_path.unlink(missing_ok=True)
    else:
        os.replace(claim_path, path)


# --- atomic writes ------------------------------------------------------------------------


def atomic_write_json(path: Path, data: dict) -> None:
    """Write `data` as JSON to `path` atomically: a `.tmp` file, then `os.replace`.

    A failure at any point (serialization, the write, the replace) leaves no `.tmp` file
    behind and re-raises.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_existing_snapshot(config: Configuration) -> Snapshot | None:
    """The current snapshot, or `None` if `fetch` has never run against this state directory.

    `SchemaVersionError` propagates unchanged, exactly like `load_config` -- the CLI maps it
    to exit code 3.
    """
    path = config.paths.snapshot_file
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return load_snapshot(data)


def write_snapshot(config: Configuration, data: dict) -> Snapshot:
    """Validate `data` as a `Snapshot`, then atomically replace the snapshot file with it.

    Validation happens before any file is touched, so a validation failure never leaves a
    `.tmp` file behind (CONTRACTS.md: "Producers validate before they write").
    """
    snapshot = load_snapshot(data)
    atomic_write_json(config.paths.snapshot_file, snapshot.model_dump(mode="json"))
    return snapshot


def write_run_status(config: Configuration, snapshot: Snapshot, request_used: int, request_budget: int) -> None:
    """Write `run-status.json`: per-area status/error/last_success, run_at, and the budget.

    `candidates` is always an empty list in this work package -- market-ranking suggestions
    are a later feature (CONTRACTS.md). Written on every run, even one that ends exit 1.
    """
    data = {
        "run_at": snapshot.run_at.isoformat(),
        "request_budget": {"used": request_used, "total": request_budget},
        "areas": [
            {
                "source": area.source,
                "base_model_hf_repo": area.base_model_hf_repo,
                "packager": area.packager,
                "status": area.status,
                "error": area.error,
                "last_success": area.last_success.isoformat() if area.last_success else None,
            }
            for area in snapshot.areas
        ],
        "candidates": [],
    }
    atomic_write_json(config.paths.run_status_file, data)


# --- hardware profile (AP4) ----------------------------------------------------------------
#
# No lock guards these functions: unlike the snapshot (one shared file every writer machine's
# `fetch` merges into), each machine's hardware profile is a file only that machine ever
# writes (`<state>/hardware/<machine>.json`), so there is nothing to serialize against another
# process (CONTRACTS.md, "Hardware profile (AP4)").


def hardware_snapshot_path(config: Configuration, machine: str) -> Path:
    return config.paths.hardware_dir / f"{machine}.json"


def load_existing_hardware_snapshot(config: Configuration, machine: str) -> HardwareSnapshot | None:
    """The current hardware profile for `machine`, or `None` if `hardware` never ran there.

    `SchemaVersionError` propagates unchanged, exactly like `load_existing_snapshot` -- the CLI
    maps it to exit code 3.
    """
    path = hardware_snapshot_path(config, machine)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return load_hardware_snapshot(data)


def write_hardware_snapshot(config: Configuration, machine: str, data: dict) -> HardwareSnapshot:
    """Validate `data` as a `HardwareSnapshot`, then atomically replace this machine's file.

    Validation happens before any file is touched, same convention as `write_snapshot`.
    """
    snapshot = load_hardware_snapshot(data)
    atomic_write_json(hardware_snapshot_path(config, machine), snapshot.model_dump(mode="json"))
    return snapshot


# --- version rule -------------------------------------------------------------------------


def check_run_is_newer(old: Snapshot | None, run_at: datetime) -> None:
    """Raise `StaleRunError` unless `run_at` is strictly newer than `old.run_at`.

    `None` (no prior snapshot) always passes. Equal instants are rejected, not just older
    ones -- a second `fetch` started within the same second as another is a stale run too.
    """
    if old is not None and run_at <= old.run_at:
        raise StaleRunError(
            f"run_at {run_at.isoformat()} is not newer than the stored snapshot's run_at "
            f"{old.run_at.isoformat()}; nothing written"
        )


# --- merge ----------------------------------------------------------------------------------


def _area_key_of_package(package: Package) -> tuple[str, str, str | None]:
    if package.source == "huggingface":
        owner = package.repo.split("/", 1)[0] if package.repo else None
        return (package.source, package.base_model_hf_repo, owner)
    return (package.source, package.base_model_hf_repo, None)


def merge_snapshot(
    old: Snapshot | None,
    run_at: datetime,
    area_outcomes: list[AreaOutcome],
    base_models: list[BaseModelSpec],
) -> Snapshot:
    """Build this run's `Snapshot` from `area_outcomes`, merged against `old`.

    Per area: `complete` replaces its packages (a package that disappeared is kept with
    `active=False`, one still found is `active=True`); `incomplete` keeps its old packages
    untouched and records the error, carrying over the old `last_success`. Base models are
    always replaced by this run's results. Never mutates `old`.
    """
    old_packages_by_key = {package_identity_key(p): p for p in (old.packages if old else [])}
    old_areas_by_key = {(a.source, a.base_model_hf_repo, a.packager): a for a in (old.areas if old else [])}

    # F2: an old package is only ever carried into this run's result when its area is among
    # this run's area_outcomes -- a base model or area no longer configured is dropped here,
    # not kept forever just because merge_snapshot never explicitly deletes anything.
    configured_area_keys = {(o.source, o.base_model_hf_repo, o.packager) for o in area_outcomes}
    result_packages = {
        key: package
        for key, package in old_packages_by_key.items()
        if _area_key_of_package(package) in configured_area_keys
    }
    new_areas: list[Area] = []

    for outcome in area_outcomes:
        area_key = (outcome.source, outcome.base_model_hf_repo, outcome.packager)
        old_area = old_areas_by_key.get(area_key)

        if outcome.status == "complete":
            old_keys_in_area = {k for k, p in old_packages_by_key.items() if _area_key_of_package(p) == area_key}
            new_keys: set[tuple[str, str, str]] = set()
            for package in outcome.packages:
                key = package_identity_key(package)
                new_keys.add(key)
                previous = old_packages_by_key.get(key)
                observed_at = previous.observed_at if previous else run_at
                result_packages[key] = package.model_copy(
                    update={"observed_at": observed_at, "last_seen": run_at, "active": True}
                )
            for stale_key in old_keys_in_area - new_keys:
                previous = old_packages_by_key[stale_key]
                if previous.active:
                    result_packages[stale_key] = previous.model_copy(update={"active": False})
            new_areas.append(
                Area(
                    source=outcome.source,
                    base_model_hf_repo=outcome.base_model_hf_repo,
                    packager=outcome.packager,
                    status="complete",
                    last_success=run_at,
                    error=None,
                )
            )
        else:
            new_areas.append(
                Area(
                    source=outcome.source,
                    base_model_hf_repo=outcome.base_model_hf_repo,
                    packager=outcome.packager,
                    status="incomplete",
                    last_success=old_area.last_success if old_area else None,
                    error=outcome.error,
                )
            )

    return Snapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        run_at=run_at,
        areas=new_areas,
        base_models=base_models,
        packages=list(result_packages.values()),
    )
