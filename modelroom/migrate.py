"""`modelroom migrate`: profiles and configuration of an earlier schema to the current one, locked.

Every schema-1 profile `<state>/hardware/<name>.json` becomes `<profile_id>.json` with a new
random `profile_id`; its embedded measurements become measurement files (`protocol: none`); the
old file is kept as `<profile_id>.json.v1.bak`. A schema-1 `modelroom.toml` gets
`[machines.<name>].profile` set, the old file kept as `modelroom.toml.v1.bak`. Every file this run
writes, the backups and their names included, goes through the sink of machine names first (FR-N):
nothing it publishes carries an old key, however the run ends. A schema-2
profile or configuration, or a schema-3 profile, is rewritten in place under the same name, the
old file kept as `<name>.v2.bak` or `<name>.v3.bak` (decided 2026-09-25 and 2026-09-26): every
backup is named after the version it left, so a folder that went from 1 to 2, to 3 and then to 4
keeps each of them. Idempotent: a second run finds nothing to
do. The whole run is planned and checked -- versions, conversions, existing targets and backups
-- before the lock is taken, and again under the lock with the same new ids and a freshly read
configuration, before anything is written (CONTRACTS.md, "Migration to schema 2").
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
    CONFIG_SCHEMA_VERSION,
    ConfigError,
    Configuration,
    config_from_text,
    load_config,
    normalize_config,
    read_config_schema_version,
)
from .contracts import HardwareSnapshot, SchemaVersionError
from .measurements import MeasurementRecord, legacy_measurement_record, measurement_path, write_measurement
from .profile import (
    PROFILE_SCHEMA_VERSION,
    HardwareProfile,
    new_profile_id,
    normalize_profile_v1,
    read_profile_document,
    stored_profile_version,
)
from .binding import default_pointer_path
from .neutral import Names, legacy_data, machine_keys, neutral_name, neutral_profile, neutral_records, renamed_note
from .neutral_config import config_text, entries, expected
from .neutral_sweep import legacy_owners, load_names, sweep
from .state import LockHandle, acquire_lock, atomic_write_json, atomic_write_text, release_lock
from .toml_writer import dump_toml

BACKUP_SUFFIX = ".v1.bak"
NOTHING_TO_DO = "nothing to do"
_CONFIG_HEADER = f"modelroom configuration, schema {CONFIG_SCHEMA_VERSION} (written by `modelroom migrate`)."


def backup_suffix(version: int) -> str:
    """The backup suffix of a file that left schema `version`: `.v1.bak`, `.v2.bak`."""
    return f".v{version}.bak"


class MigrationError(Exception):
    """A stored file cannot be migrated as it is (unreadable, not convertible, or a target or
    backup that exists with other content)."""


@dataclass
class _ProfileScan:
    legacy: dict[Path, HardwareSnapshot] = field(default_factory=dict)
    current: dict[Path, HardwareProfile] = field(default_factory=dict)
    # Stored as schema 2 or 3 and read as the current one: rewritten in place under the same name,
    # with the version it was stored as for the name of its backup.
    stale: dict[Path, tuple[int, HardwareProfile]] = field(default_factory=dict)


@dataclass
class _ProfileStep:
    source: Path
    target: Path
    backup: Path
    machine: str
    profile: HardwareProfile
    records: list[MeasurementRecord]
    kept: dict  # the schema-1 file as its backup holds it, through the sink (`neutral.legacy_data`)


@dataclass
class _RewriteStep:
    """A schema-2 or schema-3 profile rewritten as the current one under its own name, with its
    `.v2.bak` or `.v3.bak`."""

    path: Path
    backup: Path
    profile: HardwareProfile


@dataclass
class _Plan:
    profiles: list[_ProfileStep]
    config_text: str | None
    rewrites: list[_RewriteStep] = field(default_factory=list)
    config_version: int = CONFIG_SCHEMA_VERSION
    config_backup: str | None = None  # the configuration as its backup holds it, through the sink
    renames: dict[str, str] = field(default_factory=dict)  # the machine keys it takes to neutral ones


def _scan_profiles(hardware_dir: Path) -> _ProfileScan:
    """Read every `*.json` in the hardware folder; `SchemaVersionError` for schema 5 or later.

    A schema-2 or schema-3 file reads as the current schema (`read_profile_document`), so its
    stored version is taken from the raw file: it is `current` for everything that reads it, and
    `stale`, with that version, for the rewrite.
    """
    scan = _ProfileScan()
    if not hardware_dir.is_dir():
        return scan
    for path in sorted(hardware_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise MigrationError(f"{path}: expected a JSON object at the root")
            stored = stored_profile_version(data)
            document = read_profile_document(data)
        except SchemaVersionError as exc:
            raise SchemaVersionError(f"{path}: {exc}") from exc
        except (OSError, ValueError) as exc:  # ValueError: bad UTF-8/JSON, over-long integers, ValidationError
            raise MigrationError(f"{path}: cannot read hardware profile: {exc}") from exc
        if isinstance(document, HardwareSnapshot):
            scan.legacy[path] = document
            continue
        scan.current[path] = document
        if stored < PROFILE_SCHEMA_VERSION:
            scan.stale[path] = (stored, document)
    return scan


def _read_config_version(config_path: Path) -> tuple[dict, int]:
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{config_path}: cannot read config file: {exc}") from exc
    return raw, read_config_schema_version(raw, str(config_path))


def migrate(
    config_path: Path,
    now: datetime,
    fresh_id: Callable[[], str] = new_profile_id,
    *,
    pointer_path: Path | None = None,
    hostname: str | None = None,
) -> list[str]:
    """Migrate the configuration at `config_path` and its profiles; one line per changed file.

    Returns `[NOTHING_TO_DO]` when everything is already on the current schema. Raises
    `SchemaVersionError` (a file of a later schema), `ConfigError`, `MigrationError` or
    `state.LockHeldError`; none of them leaves a file written.

    The sweep of machine names comes first (FR-N): the files of the current schemas, so every backup
    this run keeps is taken from a neutral file. A schema-1 configuration and schema-1 profiles are
    this command's own to convert (their keys name the profile files): the keys, the profiles, the
    records and both backups are planned neutral, before anything is written.
    """
    config_path = config_path.resolve()
    names = load_names(config_path.parent, pointer_path or default_pointer_path(), hostname)
    swept = sweep(config_path, now, "migrate", pointer_path=pointer_path, hostname=hostname).notes
    config = load_config(config_path)
    id_for = _ids_per_machine(fresh_id)
    _plan(config_path, config, id_for, names)
    handle = acquire_lock(config.paths.lock_file, "migrate", now)
    try:
        # Planned again under the lock: another process may have changed the files meanwhile.
        current = load_config(config_path)
        if current.paths != config.paths:
            raise MigrationError(f"{config_path}: [paths] changed while waiting for the lock; run again")
        return [*swept, *_execute(_plan(config_path, current, id_for, names), config_path, current, handle)]
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


def _plan(config_path: Path, config: Configuration, id_for: Callable[[str], str], names: Names) -> _Plan:
    """Read, convert and check everything the run would write; writes nothing."""
    raw, version = _read_config_version(config_path)
    scan = _scan_profiles(config.paths.hardware_dir)
    # A run that stopped halfway left a profile of the same machine: reuse its id. It is found by its
    # content, not by its name (the sweep before this run gave it a neutral one, FR-N), and only where
    # exactly one profile answers and no other file of this run takes it too.
    left = {p.profile_id: p.model_dump(mode="json") for p in scan.current.values() if p.os_fingerprint_source == "legacy"}
    ids_by_name: dict[str, str] = {}
    steps: list[_ProfileStep] = []
    for path, legacy in scan.legacy.items():
        if any(step.machine == legacy.machine for step in steps):
            raise MigrationError(f"{path}: a second schema-1 profile for machine {legacy.machine!r}")
        profile_id = _left_by(path, legacy, left, set(ids_by_name.values())) or id_for(legacy.machine)
        ids_by_name[legacy.machine] = profile_id
        steps.append(_plan_profile(config.paths.state, path, legacy, profile_id))
    rewrites = [_plan_rewrite(path, stored, profile) for path, (stored, profile) in scan.stale.items()]
    if version >= CONFIG_SCHEMA_VERSION:
        return _Plan(steps, None, rewrites, version)
    text, backup, renames = _plan_config(config_path, raw, ids_by_name if version == 1 else {}, version, names)
    return _Plan(steps, text, rewrites, version, backup, renames)


def _left_by(path: Path, legacy: HardwareSnapshot, left: dict[str, dict], taken: set[str]) -> str | None:
    """The id of the profile an earlier run made of this schema-1 file; two that answer, or one another
    file of this run already took, cannot be told apart and stop the run (nothing written)."""
    owners = legacy_owners(legacy.model_dump(mode="json"), left)
    if len(owners) > 1 or set(owners) & taken:
        raise MigrationError(f"{path}: more than one profile of this folder was made from it; move the extra ones away and run again")
    return owners[0] if owners else None


def _plan_rewrite(path: Path, stored: int, profile: HardwareProfile) -> _RewriteStep:
    """A profile of an earlier schema rewritten in place, its backup named after the version it
    left; a backup of another content stops the run."""
    backup = _backup_path(path, stored)
    _require_backup_absent_or_same(backup, path)
    return _RewriteStep(path, backup, profile)


def _plan_profile(state_dir: Path, path: Path, legacy: HardwareSnapshot, profile_id: str) -> _ProfileStep:
    """Convert one schema-1 profile; every file it would touch must be absent or identical."""
    try:
        # Through the sink (FR-N): the old machine key is no name for the profile, the old runtime no version.
        profile = neutral_profile(normalize_profile_v1(legacy, profile_id))
        records = neutral_records([legacy_measurement_record(m, profile_id) for m in legacy.measurements])
    except ValidationError as exc:
        raise MigrationError(f"{path}: cannot convert to schema {PROFILE_SCHEMA_VERSION}: {exc}") from exc
    target = path.with_name(f"{profile_id}.json")
    # A run of an earlier version that stopped halfway left the same profile as schema 2 or 3; the
    # rewrite step of this run takes it on to the current one (`_plan_rewrite`).
    _require_profile_absent_or_same(target, profile)
    for record in records:
        _require_absent_or_same(measurement_path(state_dir, profile_id, record.measurement_id), record.model_dump(mode="json"))
    # The backup takes the profile's name and a neutral key: its old name was the machine's key.
    backup = path.with_name(f"{profile_id}.json{BACKUP_SUFFIX}")
    kept = legacy_data(legacy.model_dump(mode="json"), neutral_name(profile_id, "unknown"))
    _require_absent_or_same(backup, kept)
    return _ProfileStep(path, target, backup, legacy.machine, profile, records, kept)


def _require_absent_or_same(path: Path, content: dict) -> None:
    """`path` is absent, or holds `content` exactly."""
    stored = _existing_json(path)
    if stored is not None and stored != content:
        raise MigrationError(f"{path}: exists with other content; move it away and run again")


def _require_profile_absent_or_same(path: Path, profile: HardwareProfile) -> None:
    """`path` is absent, or holds `profile` in any schema the reader takes -- compared after reading."""
    stored = _existing_json(path)
    if stored is None:
        return
    try:
        same = read_profile_document(stored).model_dump(mode="json") == profile.model_dump(mode="json")
    except (SchemaVersionError, ValueError, AttributeError):  # not a profile of a readable schema
        same = False
    if not same:
        raise MigrationError(f"{path}: exists with other content; move it away and run again")


def _existing_json(path: Path) -> object | None:
    """The JSON a file already holds, or `None` when there is none; a file below a file stops the run."""
    blocked = next((parent for parent in path.parents if parent.exists() and not parent.is_dir()), None)
    if blocked is not None:
        raise MigrationError(f"{blocked}: is not a folder, but {path} would be written below it")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # ValueError covers bad UTF-8, bad JSON and over-long integers
        raise MigrationError(f"{path}: cannot read existing file: {exc}") from exc


def _backup_path(path: Path, version: int = 1) -> Path:
    return path.with_name(path.name + backup_suffix(version))


def _require_backup_absent_or_same(backup: Path, original: Path) -> None:
    if not backup.exists():
        return
    try:
        same = backup.read_bytes() == original.read_bytes()
    except OSError as exc:
        raise MigrationError(f"{backup}: cannot read existing backup: {exc}") from exc
    if not same:
        raise MigrationError(f"{backup}: backup exists with other content; move it away and run again")


def _plan_config(config_path: Path, raw: dict, ids_by_name: dict[str, str], version: int, names: Names) -> tuple[str, str, dict]:
    """The current text for a configuration of an earlier schema, validated like `load_config` would,
    and the text of its backup -- both through the sink (FR-N): the keys by the key rule, the paths
    inside the folder relative, the backup's other bytes as they stand.

    `ids_by_name` sets `[machines.<name>].profile` for machines of a schema-1 file; a schema-2
    file already names its profiles and keeps them. The backup is named after `version`.
    """
    try:
        normalized = normalize_config(raw)
    except ValueError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc
    for name, machine in normalized.get("machines", {}).items():
        if name in ids_by_name:
            machine["profile"] = ids_by_name[name]
    keys = machine_keys(entries(normalized), {}, names)
    renames = {old: new for old, new in keys.items() if old != new}
    normalized = expected(normalized, renames, config_path.parent)
    try:
        text = dump_toml(normalized, header=_CONFIG_HEADER)
    except TypeError as exc:
        # `inf` is valid TOML and passes `ge=0`, but the writer refuses a non-finite float: name
        # the file instead of ending in a traceback (exit 2, nothing written), as the import does.
        raise ConfigError(f"{config_path}: holds a value the TOML writer cannot write back: {exc}") from exc
    config_from_text(text, config_path)
    kept = config_text(config_path.read_text(encoding="utf-8"), raw, renames, config_path.parent)
    backup = _backup_path(config_path, version)
    try:
        same = not backup.exists() or backup.read_text(encoding="utf-8") == kept
    except (OSError, UnicodeDecodeError) as exc:
        raise MigrationError(f"{backup}: cannot read existing backup: {exc}") from exc
    if not same:
        raise MigrationError(f"{backup}: backup exists with other content; move it away and run again")
    return text, kept, renames


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
        if not step.backup.exists():
            atomic_write_json(step.backup, step.kept)
        step.source.unlink()
        count = len(step.records)
        lines.append(f"migrated {step.source.name} -> {step.target.name} ({count} measurement(s)), backup {step.backup.name}")
    lines += [_rewrite_profile(step) for step in plan.rewrites]
    if plan.config_text is not None:
        backup = _backup_path(config_path, plan.config_version)
        if not backup.exists():
            atomic_write_text(backup, plan.config_backup)
        atomic_write_text(config_path, plan.config_text)
        lines.append(f"migrated {config_path.name} to schema {CONFIG_SCHEMA_VERSION}, backup {backup.name}")
        lines += [renamed_note(new) for new in plan.renames.values()]
    return lines or [NOTHING_TO_DO]


def _rewrite_profile(step: _RewriteStep) -> str:
    """Keep the schema-2 file as its backup, then write the same profile as schema 3 in its place."""
    _keep_backup(step.backup, step.path)
    atomic_write_json(step.path, step.profile.model_dump(mode="json"))
    return f"migrated {step.path.name} to schema {PROFILE_SCHEMA_VERSION}, backup {step.backup.name}"


def _keep_backup(backup: Path, original: Path) -> None:
    """The original file as it stands, byte for byte, unless the same backup is already there."""
    if not backup.exists():
        # newline="\n" in atomic_write_text leaves "\r\n" untouched, so the bytes stay the same.
        atomic_write_text(backup, original.read_bytes().decode("utf-8"))
