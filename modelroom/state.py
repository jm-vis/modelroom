"""Persisted run state for `fetch`: the lock file, the snapshot merge rule, atomic writes.

Every "now" the functions here need is passed in explicitly (never read from the wall clock
inside this module) so tests control time exactly, per AGENTS.md's "no mocking" rule.
"""

from __future__ import annotations

import json
import os
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


class LockHeldError(Exception):
    """Another process holds the lock and it is not old enough to be considered stale."""


class StaleRunError(Exception):
    """The run being started is not strictly newer than the stored snapshot's `run_at`."""


# --- lock -------------------------------------------------------------------------------


def acquire_lock(path: Path, command: str, now: datetime) -> None:
    """Create the lock file, or raise `LockHeldError` if a live lock is already there.

    A lock younger than `LOCK_MAX_AGE` blocks immediately -- this project never waits for a
    lock. A lock at or beyond `LOCK_MAX_AGE`, or one whose content cannot be read, is treated
    as abandoned and overwritten.
    """
    if path.exists():
        holder = _read_lock(path)
        if holder is not None and now - holder["started_at"] < LOCK_MAX_AGE:
            raise LockHeldError(
                f"{path}: locked by pid {holder['pid']} running '{holder['command']}' since "
                f"{holder['started_at'].isoformat()} (younger than {LOCK_MAX_AGE})"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pid": os.getpid(), "command": command, "started_at": now.isoformat()}), encoding="utf-8"
    )


def _read_lock(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "pid": data["pid"],
            "command": data["command"],
            "started_at": datetime.fromisoformat(data["started_at"]),
        }
    except Exception:
        return None


def release_lock(path: Path) -> None:
    """Remove the lock file. A no-op if it is already gone."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


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

    result_packages = dict(old_packages_by_key)
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
