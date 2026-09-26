"""Export one machine's profile with its measurements, and import such a file somewhere else.

`export-profile` writes an `ExportObject` v1 (profile plus every measurement of that profile)
as one file; `import-profile` reads such a file into another results folder, under
`modelroom.lock`. The two decisions are independent: the profile follows the freshness rule
(`recorded_at`; a younger stored profile is never overwritten), each measurement follows the
import rule of `measurements.plan_measurement_import` (idempotent / conflict / new). An import
adds a `[machines.<name>]` entry for the imported machine with the reserves from `[defaults]`
and `writer = false`, and it never touches this machine's own binding (the pointer file is not
even read). See CONTRACTS.md, "Export and import".

The whole run is planned under the lock and only then written, so the plan is the freshness
check: every writer of a profile, a measurement or the configuration holds the same lock, so
nothing can change between the plan and the writes.
"""

from __future__ import annotations

import copy
import json
import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .binding import read_pointer
from .config import (
    CONFIG_SCHEMA_VERSION,
    ConfigError,
    Configuration,
    config_from_text,
    load_config,
    normalize_config,
    read_config_schema_version,
)
from .contracts import PROFILE_ID_RE, HardwareSnapshot, SchemaVersionError, validate_machine_name
from .measurements import (
    EXPORT_SCHEMA_VERSION,
    ExportObject,
    MeasurementRecord,
    load_export,
    measurement_path,
    plan_measurement_import,
    read_measurements,
    write_measurement,
)
from .migrate import NOTHING_TO_DO, backup_suffix  # one wording and one backup name, defined once
from .profile import HardwareProfile, read_profile_document
from .state import LockHandle, acquire_lock, atomic_write_json, atomic_write_text, release_lock
from .toml_writer import dump_toml

SHORT_ID_LEN = 8
CLONE_NOTE = "same hardware id (cloned image?)"
MIGRATE_HINT = "run `modelroom migrate` first"
NOTHING_WRITTEN = "nothing written"
# The one backup of a hand-written configuration: written before the first rewrite, never again.
CONFIG_BACKUP_SUFFIX = ".bak"

_CONFIG_HEADER = f"modelroom configuration, schema {CONFIG_SCHEMA_VERSION} (rewritten by `modelroom import-profile`)."
_MACHINE_UNSAFE_RE = re.compile(r"[^a-z0-9]+")


class StoredFileError(Exception):
    """A file this command reads (the export file, a stored profile) is not readable as itself.

    The CLI maps it to exit `3`, exactly like `SchemaVersionError`.
    """


class PrerequisiteError(Exception):
    """Something has to happen before this command can run: a migration, a measurement, or a
    `--profile` argument. The CLI maps it to exit `2`."""


class ImportConflictError(Exception):
    """The import would have to overwrite or contradict a stored file; nothing is written.

    The CLI maps it to exit `1`, like a held lock: the run can be repeated once the conflict is
    resolved by hand.
    """


# --- reading what is stored --------------------------------------------------------------------


def _load_config(config_path: Path) -> Configuration:
    """`load_config`, with a configuration that is not valid UTF-8 named as a `ConfigError`.

    `load_config` catches `OSError` but lets `UnicodeDecodeError` through (it is a `ValueError`),
    and its message carries the byte position, not the file -- so both commands would end in a
    traceback. Here it becomes exit 2 with the path, like every other unusable configuration.
    """
    try:
        return load_config(config_path)
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{config_path}: cannot read config file: {exc}") from exc


@dataclass
class ProfileScan:
    """Every `*.json` in a results folder's `hardware/`, sorted into what an import can use.

    `profiles` are schema-2 profiles whose file name is their own `profile_id`; `legacy` are
    schema-1 files (they need `modelroom migrate`); `unreadable` are files that do not read,
    do not validate, carry an unsupported `schema_version`, or disagree with their own name.
    A broken file is listed, never loaded, and never stops an import of another profile.
    """

    profiles: dict[str, HardwareProfile] = field(default_factory=dict)
    legacy: list[Path] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)


def scan_profiles(hardware_dir: Path) -> ProfileScan:
    """Read every profile file in `hardware_dir`; nothing raises, everything broken is listed."""
    scan = ProfileScan()
    if not hardware_dir.is_dir():
        return scan
    for path in sorted(hardware_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object at the root")
            document = read_profile_document(data)
        # ValueError: bad UTF-8/JSON, validation. RecursionError: extreme nesting (see
        # `read_export_file`) -- a foreign broken file must be listed, never end the import.
        except (OSError, ValueError, SchemaVersionError, RecursionError) as exc:
            scan.unreadable.append((path, str(exc)))
            continue
        if isinstance(document, HardwareSnapshot):
            scan.legacy.append(path)
        elif document.profile_id != path.stem:
            scan.unreadable.append((path, "file name does not match profile_id"))
        else:
            scan.profiles[document.profile_id] = document
    return scan


def read_export_file(path: Path) -> ExportObject:
    """Validate an export file: `SchemaVersionError` outside the accepted range, else
    `StoredFileError` when it does not read or does not validate (both exit `3`)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    # ValueError: bad UTF-8/JSON, over-long integers. RecursionError is neither a ValueError nor
    # an OSError, and `json.loads` raises it for extreme nesting (measured: 20000 nested arrays
    # on CPython 3.12), so without it a prepared file would end in a traceback instead of exit 3.
    except (OSError, ValueError, RecursionError) as exc:
        raise StoredFileError(f"{path}: cannot read export file: {exc}") from exc
    if not isinstance(data, dict):
        raise StoredFileError(f"{path}: expected a JSON object at the root")
    try:
        return load_export(data)
    except SchemaVersionError as exc:
        raise SchemaVersionError(f"{path}: {exc}") from exc
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        raise StoredFileError(f"{path}: invalid export file: {exc}") from exc


# --- export --------------------------------------------------------------------------------------


def export_profile(config_path: Path, out_path: Path, profile_id: str | None, pointer_path: Path) -> list[str]:
    """Write one profile and all of its measurements to `out_path`; returns the report lines.

    The profile is the one named with `--profile`, else this machine's binding for the results
    folder (the configuration file's own folder). An unreadable measurement file is reported
    and left out; it does not stop the export. Writing is atomic (`atomic_write_json`), so a
    reader of `out_path` never sees a half-written file. No lock is taken: nothing inside the
    state folder is written, and a measurement file is published whole and never changed, so
    the only effect of a concurrent write is that a measurement published during the export may
    not be in it.
    """
    resolved_config = config_path.resolve()
    config = _load_config(resolved_config)
    results_dir = resolved_config.parent
    if profile_id is not None and not PROFILE_ID_RE.fullmatch(profile_id):
        raise PrerequisiteError(f"--profile must be a profile_id of 16 lowercase hex characters: {profile_id!r}")
    chosen = profile_id or read_pointer(pointer_path).binding_for(results_dir)
    if chosen is None:
        raise PrerequisiteError(
            f"{results_dir}: no profile of this machine is bound to this results folder; "
            "measure this machine first, or name a profile with --profile <profile_id>"
        )
    _check_out_target(out_path, config, resolved_config, pointer_path)
    profile = _profile_to_export(config.paths.hardware_dir, chosen)
    stored = read_measurements(config.paths.state, chosen)
    export = ExportObject(schema_version=EXPORT_SCHEMA_VERSION, profile=profile, measurements=stored.records)
    out = out_path.resolve()
    atomic_write_json(out, export.model_dump(mode="json"))
    lines = [f"exported {profile.display_name} ({chosen}) with {len(stored.records)} measurement(s) to {out}"]
    return lines + [f"skipped unreadable measurement file {path.name}: {reason}" for path, reason in stored.unreadable]


def _check_out_target(out_path: Path, config: Configuration, config_path: Path, pointer_path: Path) -> None:
    """`--out` is a user argument: it must not be a file this package owns and writes itself.

    Refused: anywhere inside `paths.state` (the profiles, the snapshot, the run status and the
    lock file live there -- the same rule `PathsConfig` holds for `paths.markdown`), the
    configuration file itself, and the pointer file. Each of them would otherwise be replaced
    by an export object, outside the lock, and the next run would read an export where it
    expects its own file.
    """
    out = out_path.resolve()
    state = config.paths.state.resolve(strict=False)
    if out == state or out.is_relative_to(state):
        raise PrerequisiteError(
            f"{out_path}: --out must not lie inside paths.state ({config.paths.state}) -- "
            "that folder holds the profiles, the snapshot and the lock file"
        )
    for owned, name in ((config_path, "the configuration file"), (pointer_path.resolve(), "the pointer file")):
        if out == owned:
            raise PrerequisiteError(f"{out_path}: --out must not be {name}")


def _profile_to_export(hardware_dir: Path, profile_id: str) -> HardwareProfile:
    """The stored schema-2 profile with this id, or the reason why it cannot be exported."""
    scan = scan_profiles(hardware_dir)
    profile = scan.profiles.get(profile_id)
    if profile is not None:
        return profile
    path = hardware_dir / f"{profile_id}.json"
    for broken, reason in scan.unreadable:
        if broken == path:
            raise StoredFileError(f"{path}: cannot read hardware profile: {reason}")
    if path in scan.legacy:
        raise PrerequisiteError(f"{path}: is still a schema-1 profile; {MIGRATE_HINT}")
    raise PrerequisiteError(f"{hardware_dir}: holds no profile {profile_id}; measure this machine first")


# --- import --------------------------------------------------------------------------------------


@dataclass
class ImportPlan:
    """What the import writes, and the line it reports for every decision it took."""

    profile: HardwareProfile | None = None
    measurements: list[MeasurementRecord] = field(default_factory=list)
    config_text: str | None = None
    # The backup to write before the configuration is rewritten; `None` when one is already
    # there (it is never overwritten) or when the configuration stays as it is.
    config_backup: Path | None = None
    # The `.v2.bak` of a configuration this run takes on to the current schema; `None` otherwise.
    schema_backup: Path | None = None
    lines: list[str] = field(default_factory=list)

    def writes_nothing(self) -> bool:
        return self.profile is None and not self.measurements and self.config_text is None


def import_profile(config_path: Path, export_path: Path, now: datetime) -> list[str]:
    """Import an export file into the results folder of `config_path`; returns the report lines.

    Raises `ConfigError`/`PrerequisiteError` (exit 2), `SchemaVersionError`/`StoredFileError`
    (exit 3), `state.LockHeldError`/`ImportConflictError` (exit 1). Every one of them leaves
    the results folder as it was, except for the lock file itself.

    The configuration is read twice: once before the lock, to refuse a missing, invalid or
    schema-1 file without even creating a lock file, and once under the lock, because the run
    writes that same file. The second read is what the plan uses -- another import may have
    added its own `[machines.<name>]` entry between the two reads, and planning from the first
    read would drop it again (the two-import concurrency test reproduces exactly that). If
    `[paths]` changed meanwhile, the lock in hand belongs to the old state folder, so the run
    stops and asks to be repeated.
    """
    resolved_config = config_path.resolve()
    config = _load_config(resolved_config)
    _read_raw_config(resolved_config)
    export = read_export_file(export_path)
    handle = acquire_lock(config.paths.lock_file, "import-profile", now)
    try:
        current = _load_config(resolved_config)
        if current.paths != config.paths:
            raise ImportConflictError(
                f"{resolved_config}: [paths] changed while waiting for the lock; run again\n{NOTHING_WRITTEN}"
            )
        plan = plan_import(current, resolved_config, _read_raw_config(resolved_config), export)
        return _execute(plan, current, resolved_config, handle)
    finally:
        release_lock(handle)


def _read_raw_config(config_path: Path) -> dict:
    """The configuration as the raw table the rewrite starts from; schema 1 needs `migrate`.

    Schema 2 is read as it is stored: `_plan_machine` writes it back as schema 3, even when it
    adds no machine (decided 2026-09-25).
    """
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{config_path}: cannot read config file: {exc}") from exc
    if read_config_schema_version(raw, str(config_path)) == 1:
        raise PrerequisiteError(f"{config_path}: is still a schema-1 configuration; {MIGRATE_HINT}")
    return raw


def plan_import(config: Configuration, config_path: Path, raw: dict, export: ExportObject) -> ImportPlan:
    """Decide everything the import would write, and write nothing.

    Raises `ImportConflictError` as soon as one decision cannot be taken without overwriting
    something, so a conflict anywhere leaves the whole run without a single write.
    """
    scan = scan_profiles(config.paths.hardware_dir)
    plan = ImportPlan()
    plan.lines += [f"skipped unreadable profile file {path.name}: {reason}" for path, reason in scan.unreadable]
    plan.lines += [f"skipped schema-1 profile file {path.name}; {MIGRATE_HINT}" for path in scan.legacy]
    plan.profile, line = _plan_profile(config, export.profile, scan)
    plan.lines.append(line)
    plan.lines += _notes(export.profile, scan)
    records, measurement_lines = _plan_measurements(config, export)
    plan.measurements = records
    plan.lines += measurement_lines
    name, plan.config_text = _plan_machine(config, config_path, raw, export.profile)
    if name is not None:
        plan.lines.append(f'note: machine "{name}" added with the reserves from [defaults] and writer = false')
    if plan.config_text is not None:
        plan.config_backup, line = _plan_config_backup(config_path)
        plan.lines.append(line)
        plan.schema_backup, schema_line = _plan_schema_backup(config_path, raw)
        plan.lines += [schema_line] if schema_line else []
    return plan


def _plan_schema_backup(config_path: Path, raw: dict) -> tuple[Path | None, str | None]:
    """The backup of a configuration this run takes on to the current schema, and its line.

    Named after the version it leaves, as `modelroom migrate` names it (`.v2.bak`), and kept
    apart from the one `.bak` above, which is never overwritten and may hold an earlier state:
    the bytes of the schema-2 file are kept in every case (decided 2026-09-25). A backup of that
    name with other content is a conflict; nothing is written.
    """
    version = raw.get("schema_version")
    if version == CONFIG_SCHEMA_VERSION:
        return None, None
    backup = config_path.with_name(config_path.name + backup_suffix(version))
    line = f"note: the configuration is written as schema {CONFIG_SCHEMA_VERSION}; schema {version} kept as {backup.name}"
    if not backup.exists():
        return backup, line
    try:
        same = backup.read_bytes() == config_path.read_bytes()
    except OSError as exc:
        raise ImportConflictError(f"{backup}: cannot read existing backup ({exc})\n{NOTHING_WRITTEN}") from exc
    if not same:
        raise ImportConflictError(f"{backup}: backup exists with other content; move it away and run again\n{NOTHING_WRITTEN}")
    return None, line


def _plan_config_backup(config_path: Path) -> tuple[Path | None, str]:
    """The one backup of the configuration, and the line that names it.

    The rewrite goes through `dump_toml` and keeps values, not comments, so the file as the user
    wrote it is kept once as `<config>.bak`. An existing backup is never overwritten: it holds an
    older state, and replacing it would throw away the comments this run is about to drop.
    """
    backup = config_path.with_name(config_path.name + CONFIG_BACKUP_SUFFIX)
    if backup.exists():
        return None, f"note: the configuration is rewritten without its comments; {backup} is kept as it is"
    return backup, f"note: the configuration is rewritten without its comments; backup: {backup}"


def _plan_profile(
    config: Configuration, incoming: HardwareProfile, scan: ProfileScan
) -> tuple[HardwareProfile | None, str]:
    """The freshness rule: the younger `recorded_at` wins, a younger stored profile stays."""
    label = _label(incoming, scan)
    existing = scan.profiles.get(incoming.profile_id)
    if existing is None:
        target = config.paths.hardware_dir / f"{incoming.profile_id}.json"
        _check_writable_path(target)
        if target in scan.legacy:
            raise ImportConflictError(f"{target}: is still a schema-1 profile; {MIGRATE_HINT}\n{NOTHING_WRITTEN}")
        if target.exists():
            raise ImportConflictError(
                f"{target}: a file is in the way that does not read as this profile; "
                f"move it away and run again\n{NOTHING_WRITTEN}"
            )
        return incoming, f"profile {label}: new"
    if incoming.recorded_at > existing.recorded_at:
        return incoming, f"profile {label}: updated, the imported profile is newer ({_stamp(incoming)} > {_stamp(existing)})"
    if incoming.recorded_at == existing.recorded_at:
        return None, f"profile {label}: unchanged, same recorded_at ({_stamp(incoming)})"
    return None, f"profile {label}: kept, the local profile is newer ({_stamp(existing)} > {_stamp(incoming)})"


def _plan_measurements(config: Configuration, export: ExportObject) -> tuple[list[MeasurementRecord], list[str]]:
    """The measurement import rule, plus the check that no file is in the way of a new one."""
    state, profile_id = config.paths.state, export.profile.profile_id
    stored = read_measurements(state, profile_id)
    lines = [f"skipped unreadable measurement file {path.name}: {reason}" for path, reason in stored.unreadable]
    plan = plan_measurement_import(stored.records, export.measurements)
    if plan.conflicts:
        raise ImportConflictError(
            "\n".join(
                [
                    f"measurement {record.measurement_id}: conflict, the stored file of that id has other content"
                    for record in plan.conflicts
                ]
                + [NOTHING_WRITTEN]
            )
        )
    for record in plan.new:
        _check_writable_path(measurement_path(state, profile_id, record.measurement_id))
    blocked = [record for record in plan.new if measurement_path(state, profile_id, record.measurement_id).exists()]
    if blocked:
        raise ImportConflictError(
            "\n".join(
                [
                    f"{measurement_path(state, profile_id, record.measurement_id)}: a file is in the way that does "
                    "not read as a measurement; move it away and run again"
                    for record in blocked
                ]
                + [NOTHING_WRITTEN]
            )
        )
    lines += [f"measurement {record.measurement_id}: unchanged (already stored)" for record in plan.idempotent]
    return plan.new, lines + [f"measurement {record.measurement_id}: new" for record in plan.new]


def _plan_machine(
    config: Configuration, config_path: Path, raw: dict, profile: HardwareProfile
) -> tuple[str | None, str | None]:
    """The `[machines.<name>]` entry an imported machine gets, and the configuration text for it.

    `(None, None)` when a machine already carries this `profile` and the file is on the current
    schema; a file of an earlier schema is rewritten all the same, with no machine added. The
    rewritten file is validated exactly as `load_config` would before it is ever written; it
    carries no comments (the TOML writer emits data, not comments), so a hand-written remark is
    lost on the first import -- named in the report line and in CONTRACTS.md.
    """
    known = any(machine.profile == profile.profile_id for machine in config.machines.values())
    if known and raw.get("schema_version") == CONFIG_SCHEMA_VERSION:
        return None, None
    try:
        data = copy.deepcopy(normalize_config(raw))
    except ValueError as exc:
        raise ConfigError(f"{config_path}: invalid configuration: {exc}") from exc
    name = None
    if not known:
        name = machine_name_for(profile.display_name, profile.profile_id, set(config.machines))
        data.setdefault("machines", {})[name] = {
            "reserve_ram_gib": config.defaults.reserve_ram_gib,
            "reserve_vram_gib": config.defaults.reserve_vram_gib,
            "writer": False,
            "profile": profile.profile_id,
        }
    try:
        text = dump_toml(data, header=_CONFIG_HEADER)
    except TypeError as exc:
        # `inf` is valid TOML and passes `ge=0`, but the writer refuses a non-finite float:
        # name the file instead of ending in a traceback (exit 2, nothing written).
        raise ConfigError(f"{config_path}: holds a value the TOML writer cannot write back: {exc}") from exc
    config_from_text(text, config_path)
    return name, text


def machine_name_for(display_name: str, profile_id: str, taken: set[str]) -> str:
    """A configuration key for an imported machine: `display_name` normalized, then made unique.

    Everything outside `a-z0-9` becomes a single `-` and the ends are trimmed; a name that is
    taken, or empty after that, gets the short id (`profile_id`'s first 8 hex) appended or, as
    a last resort, the whole `profile_id`.
    """
    base = _MACHINE_UNSAFE_RE.sub("-", display_name.lower()).strip("-")
    short = profile_id[:SHORT_ID_LEN]
    for candidate in (base, f"{base}-{short}" if base else short, f"{base}-{profile_id}" if base else profile_id):
        if candidate and candidate not in taken:
            return validate_machine_name(candidate)
    raise ImportConflictError(f"no free machine name for {display_name!r} ({profile_id}); rename a machine first")


def _check_writable_path(path: Path) -> None:
    """Refuse a target whose folder chain is blocked by a file, before anything is written.

    `Path.mkdir(parents=True, exist_ok=True)` inside the atomic writers raises `OSError` when an
    existing part of the chain is a file (a file named `hardware`, or one named after a profile
    where the measurement folder belongs). Without this check the profile was written first and
    the measurement write then crashed the CLI with a traceback -- the same rule `migrate`
    applies before it writes (`migrate._require_absent_or_same`).
    """
    blocked = next((parent for parent in path.parents if parent.exists() and not parent.is_dir()), None)
    if blocked is not None:
        raise ImportConflictError(
            f"{blocked}: is not a folder, but {path} would be written below it; "
            f"move it away and run again\n{NOTHING_WRITTEN}"
        )


def _notes(incoming: HardwareProfile, scan: ProfileScan) -> list[str]:
    """The two "both are kept" notes: same hardware id under another id, and the same name."""
    notes = []
    for other in scan.profiles.values():
        if other.profile_id == incoming.profile_id:
            continue
        if other.os_fingerprint != "none" and other.os_fingerprint == incoming.os_fingerprint:
            notes.append(f"note: {_short(other)} has the {CLONE_NOTE} -- both are kept")
        if other.display_name == incoming.display_name:
            notes.append(f"note: {_short(other)} has the same name -- both are kept, shown with their short id")
    return notes


def _label(profile: HardwareProfile, scan: ProfileScan) -> str:
    """The display name, with the short id when another stored profile carries the same name."""
    ambiguous = any(
        other.profile_id != profile.profile_id and other.display_name == profile.display_name
        for other in scan.profiles.values()
    )
    return _short(profile) if ambiguous else profile.display_name


def _short(profile: HardwareProfile) -> str:
    return f"{profile.display_name} ({profile.profile_id[:SHORT_ID_LEN]})"


def _stamp(profile: HardwareProfile) -> str:
    return profile.recorded_at.isoformat()


def _execute(plan: ImportPlan, config: Configuration, config_path: Path, handle: LockHandle) -> list[str]:
    """Write what `plan_import` decided, under the held lock. Every file write is atomic on its
    own; a run that stops in between leaves valid files and converges when it is repeated."""
    if plan.profile is not None:
        atomic_write_json(
            config.paths.hardware_dir / f"{plan.profile.profile_id}.json", plan.profile.model_dump(mode="json")
        )
    for record in plan.measurements:
        write_measurement(config.paths.state, record, handle)
    for backup in (plan.config_backup, plan.schema_backup):
        if backup is not None:
            # The file as it stands, byte for byte: `newline="\n"` leaves an existing "\r\n" alone,
            # and the configuration was already read as UTF-8 by `load_config` (same as `migrate`).
            atomic_write_text(backup, config_path.read_bytes().decode("utf-8"))
    if plan.config_text is not None:
        atomic_write_text(config_path, plan.config_text)
    return plan.lines if not plan.writes_nothing() else [*plan.lines, NOTHING_TO_DO]
