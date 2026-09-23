"""`modelroom migrate`: schema-1 profiles and configuration to schema 2, in one locked run.

Every schema-1 profile `<state>/hardware/<name>.json` becomes `<profile_id>.json` with a new
random `profile_id`; its embedded measurements become measurement files (`protocol: none`); the
old file is kept as `<name>.json.v1.bak`. A schema-1 `modelroom.toml` becomes schema 2 with
`[machines.<name>].profile` set, the old file kept as `modelroom.toml.v1.bak`. Idempotent: a
second run finds nothing to do. The whole run is planned and checked -- versions, conversions,
existing targets and backups -- before the lock is taken, and again under the lock with the same
new ids and a freshly read configuration, before anything is written (CONTRACTS.md, "Migration
to schema 2").
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from .config import (
    ConfigError,
    Configuration,
    config_from_text,
    load_config,
    normalize_config_v1,
    read_config_schema_version,
)
from .contracts import HardwareSnapshot, SchemaVersionError
from .measurements import MeasurementRecord, legacy_measurement_record, measurement_path, write_measurement
from .profile import HardwareProfile, new_profile_id, normalize_profile_v1, read_profile_document
from .state import LockHandle, acquire_lock, atomic_write_json, atomic_write_text, release_lock
from .toml_writer import dump_toml

BACKUP_SUFFIX = ".v1.bak"
NOTHING_TO_DO = "nothing to do"
_CONFIG_HEADER = "modelroom configuration, schema 2 (written by `modelroom migrate`)."


class MigrationError(Exception):
    """A stored file cannot be migrated as it is (unreadable, not convertible, or a target or
    backup that exists with other content)."""


@dataclass
class _ProfileScan:
    legacy: dict[Path, HardwareSnapshot] = field(default_factory=dict)
    current: dict[Path, HardwareProfile] = field(default_factory=dict)


@dataclass
class _ProfileStep:
    source: Path
    target: Path
    backup: Path
    machine: str
    profile: HardwareProfile
    records: list[MeasurementRecord]


@dataclass
class _Plan:
    profiles: list[_ProfileStep]
    config_text: str | None


def _scan_profiles(hardware_dir: Path) -> _ProfileScan:
    """Read every `*.json` in the hardware folder; `SchemaVersionError` for schema 3 or later."""
    scan = _ProfileScan()
    if not hardware_dir.is_dir():
        return scan
    for path in sorted(hardware_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise MigrationError(f"{path}: expected a JSON object at the root")
            document = read_profile_document(data)
        except SchemaVersionError as exc:
            raise SchemaVersionError(f"{path}: {exc}") from exc
        except (OSError, ValueError) as exc:  # ValueError: bad UTF-8/JSON, over-long integers, ValidationError
            raise MigrationError(f"{path}: cannot read hardware profile: {exc}") from exc
        if isinstance(document, HardwareSnapshot):
            scan.legacy[path] = document
        else:
            scan.current[path] = document
    return scan


def _read_config_version(config_path: Path) -> tuple[dict, int]:
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{config_path}: cannot read config file: {exc}") from exc
    return raw, read_config_schema_version(raw, str(config_path))


def migrate(config_path: Path, now: datetime, fresh_id: Callable[[], str] = new_profile_id) -> list[str]:
    """Migrate the configuration at `config_path` and its profiles; one line per changed file.

    Returns `[NOTHING_TO_DO]` when everything is already schema 2. Raises `SchemaVersionError`
    (a file of schema 3 or later), `ConfigError`, `MigrationError` or `state.LockHeldError`;
    none of them leaves a file written.
    """
    config_path = config_path.resolve()
    config = load_config(config_path)
    id_for = _ids_per_machine(fresh_id)
    _plan(config_path, config, id_for)
    handle = acquire_lock(config.paths.lock_file, "migrate", now)
    try:
        # Planned again under the lock: another process may have changed the files meanwhile.
        current = load_config(config_path)
        if current.paths != config.paths:
            raise MigrationError(f"{config_path}: [paths] changed while waiting for the lock; run again")
        return _execute(_plan(config_path, current, id_for), config_path, current, handle)
    finally:
        release_lock(handle)


def _ids_per_machine(fresh_id: Callable[[], str]) -> Callable[[str], str]:
    """One new `profile_id` per machine, drawn once: the check and the run use the same ids."""
    drawn: dict[str, str] = {}

    def id_for(machine: str) -> str:
        if machine not in drawn:
            drawn[machine] = fresh_id()
        return drawn[machine]

    return id_for


def _plan(config_path: Path, config: Configuration, id_for: Callable[[str], str]) -> _Plan:
    """Read, convert and check everything the run would write; writes nothing."""
    raw, version = _read_config_version(config_path)
    scan = _scan_profiles(config.paths.hardware_dir)
    # A run that stopped halfway left a schema-2 profile of the same machine: reuse its id.
    ids_by_name = {p.display_name: p.profile_id for p in scan.current.values() if p.os_fingerprint_source == "legacy"}
    steps: list[_ProfileStep] = []
    for path, legacy in scan.legacy.items():
        if any(step.machine == legacy.machine for step in steps):
            raise MigrationError(f"{path}: a second schema-1 profile for machine {legacy.machine!r}")
        profile_id = ids_by_name.get(legacy.machine) or id_for(legacy.machine)
        ids_by_name[legacy.machine] = profile_id
        steps.append(_plan_profile(config.paths.state, path, legacy, profile_id))
    config_text = _plan_config(config_path, raw, ids_by_name) if version == 1 else None
    return _Plan(steps, config_text)


def _plan_profile(state_dir: Path, path: Path, legacy: HardwareSnapshot, profile_id: str) -> _ProfileStep:
    """Convert one schema-1 profile; every file it would touch must be absent or identical."""
    try:
        profile = normalize_profile_v1(legacy, profile_id)
        records = [legacy_measurement_record(m, profile_id) for m in legacy.measurements]
    except ValidationError as exc:
        raise MigrationError(f"{path}: cannot convert to schema 2: {exc}") from exc
    target = path.with_name(f"{profile_id}.json")
    _require_absent_or_same(target, profile.model_dump(mode="json"))
    for record in records:
        _require_absent_or_same(measurement_path(state_dir, profile_id, record.measurement_id), record.model_dump(mode="json"))
    backup = _backup_path(path)
    _require_backup_absent_or_same(backup, path)
    return _ProfileStep(path, target, backup, legacy.machine, profile, records)


def _require_absent_or_same(path: Path, content: dict) -> None:
    blocked = next((parent for parent in path.parents if parent.exists() and not parent.is_dir()), None)
    if blocked is not None:
        raise MigrationError(f"{blocked}: is not a folder, but {path} would be written below it")
    if not path.exists():
        return
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # ValueError covers bad UTF-8, bad JSON and over-long integers
        raise MigrationError(f"{path}: cannot read existing file: {exc}") from exc
    if stored != content:
        raise MigrationError(f"{path}: exists with other content; move it away and run again")


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + BACKUP_SUFFIX)


def _require_backup_absent_or_same(backup: Path, original: Path) -> None:
    if not backup.exists():
        return
    try:
        same = backup.read_bytes() == original.read_bytes()
    except OSError as exc:
        raise MigrationError(f"{backup}: cannot read existing backup: {exc}") from exc
    if not same:
        raise MigrationError(f"{backup}: backup exists with other content; move it away and run again")


def _plan_config(config_path: Path, raw: dict, ids_by_name: dict[str, str]) -> str:
    """The schema-2 text for a schema-1 configuration, validated like `load_config` would."""
    try:
        normalized = normalize_config_v1(raw)
    except ValueError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc
    for name, machine in normalized.get("machines", {}).items():
        if name in ids_by_name:
            machine["profile"] = ids_by_name[name]
    text = dump_toml(normalized, header=_CONFIG_HEADER)
    config_from_text(text, config_path)
    _require_backup_absent_or_same(_backup_path(config_path), config_path)
    return text


def _execute(plan: _Plan, config_path: Path, config: Configuration, handle: LockHandle) -> list[str]:
    """Write what `_plan` checked. A file that is already there is identical (checked), so it is
    skipped; a repeated run after a stop converges instead of duplicating."""
    lines: list[str] = []
    for step in plan.profiles:
        if not step.target.exists():
            atomic_write_json(step.target, step.profile.model_dump(mode="json"))
        for record in step.records:
            if not measurement_path(config.paths.state, record.profile_id, record.measurement_id).exists():
                write_measurement(config.paths.state, record, handle)
        if step.backup.exists():
            step.source.unlink()
        else:
            step.source.replace(step.backup)
        count = len(step.records)
        lines.append(f"migrated {step.source.name} -> {step.target.name} ({count} measurement(s)), backup {step.backup.name}")
    if plan.config_text is not None:
        backup = _backup_path(config_path)
        if not backup.exists():
            # newline="\n" in atomic_write_text leaves "\r\n" untouched, so the bytes stay the same.
            atomic_write_text(backup, config_path.read_bytes().decode("utf-8"))
        atomic_write_text(config_path, plan.config_text)
        lines.append(f"migrated {config_path.name} to schema 2, backup {backup.name}")
    return lines or [NOTHING_TO_DO]
