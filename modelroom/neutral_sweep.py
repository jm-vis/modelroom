"""`sweep`: the sink of `modelroom/neutral.py` over every file a results folder already holds (FR-N).

Every writing command runs it first, under `modelroom.lock`: the guided mode, `hardware`, `fetch`,
`render`, `migrate`, `import-profile` and `export-profile` (CONTRACTS.md, "Neutral names"). It goes
over

- the profiles (`display_name`), in the schema they are stored in -- only that field changes;
- the configuration (the keys of `[machines.<name>]`), by replacing the table heads in the text,
  so its comments stay; a file whose machines are not written as table heads is written anew;
- the measurement records (the record rules), rewritten under their own `measurement_id`;
- modelroom's own backups (`modelroom.toml.bak`, `.vN.bak`, `<profile_id>.json.vN.bak`): table heads
  replaced in the text, `display_name` in a profile backup. A backup that cannot be cleaned is a
  note that names only the file, never its content; no new backup is ever made.

Key, profile name and records are checked independently of each other, so a run that stopped
halfway converges on the next one, and a second sweep changes nothing. The folder is read without
the lock first; only when something is to change is the lock taken and the folder read again. A file
that cannot be read (a backup aside), a write that fails and a lock another process holds end the
command with `neutral.FAILED`, before it writes anything else.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .binding import PointerFileError, default_pointer_path, read_pointer
from .config import ConfigError, Configuration, config_from_text, load_config
from .contracts import PROFILE_ID_RE, Package, SchemaVersionError
from .measurements import MeasurementRecord
from .neutral import FAILED, Names, NeutralError, display_name_for, known_package, machine_keys, neutral_record, old_key, renamed_note
from .remote_store import RemoteStoreError, read_remote, remote_path_for, this_machine
from .state import LockHeldError, UnreadableStateFileError, acquire_lock, atomic_write_json, atomic_write_text, read_snapshot, release_lock
from .toml_writer import dump_toml

_HEAD_RE = re.compile(
    r"""^(?P<lead>[ \t]*\[[ \t]*machines[ \t]*\.[ \t]*)(?P<key>"[^"\n]*"|'[^'\n]*'|[A-Za-z0-9_-]+)(?P<tail>[ \t]*\][ \t]*(?:\#.*)?)$""",
    re.MULTILINE,
)
_HEADER = "modelroom configuration (machine names made neutral by modelroom)."
_BACKUP_NOTE = "a backup of modelroom could not be cleaned of machine names: {name}; delete it by hand"
_LEGACY_BACKUP = "a schema-1 profile backup in hardware/"


@dataclass
class Swept:
    """What a sweep did: the machines it renamed (old key -> new key) and its notes."""

    renames: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Plan:
    writes: list[tuple[Path, dict | str]] = field(default_factory=list)
    config_text: str | None = None
    renames: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def writes_something(self) -> bool:
        return bool(self.writes) or self.config_text is not None

    def swept(self) -> Swept:
        return Swept(dict(self.renames), [*(renamed_note(new) for new in self.renames.values()), *self.notes])


def load_names(results_dir: Path, pointer_path: Path, hostname: str | None, remote_path: Path | None = None) -> Names:
    """This user's `remote.json` (`RemoteStoreError` when it is broken), the folder's binding, the 0.1.0 key."""
    path = remote_path if remote_path is not None else remote_path_for(pointer_path)
    stored = read_remote(path)
    try:
        bound = read_pointer(pointer_path).binding_for(results_dir)
    except (PointerFileError, SchemaVersionError, OSError):
        bound = None  # the command that reads the pointer file says so itself
    return Names(stored.this_machine, lambda: this_machine(path), bound, dict(stored.ssh), old_key(hostname))


def sweep(
    config_file: Path,
    now: datetime,
    command: str,
    *,
    pointer_path: Path | None = None,
    hostname: str | None = None,
    remote_path: Path | None = None,
) -> Swept:
    """Sweep the results folder of `config_file`; a configuration that does not load is left to the
    command, which says so with its own exit code."""
    config_file = config_file.resolve()
    try:
        config = load_config(config_file)
    except (ConfigError, SchemaVersionError, ValueError, OSError):
        return Swept()
    names = load_names(config_file.parent, pointer_path or default_pointer_path(), hostname, remote_path)
    first = _plan(config_file, config, names)
    if not first.writes_something():
        return first.swept()
    try:
        handle = acquire_lock(config.paths.lock_file, command, now)
    except (LockHeldError, OSError) as exc:
        raise NeutralError(FAILED) from exc
    try:
        return sweep_locked(config_file, names)
    finally:
        release_lock(handle)


def sweep_locked(config_file: Path, names: Names) -> Swept:
    """The sweep itself; the caller holds `modelroom.lock` of this folder."""
    try:
        config = load_config(config_file)
    except (ConfigError, SchemaVersionError, ValueError, OSError):
        return Swept()
    plan = _plan(config_file, config, names)
    try:
        for path, content in plan.writes:
            (atomic_write_json if isinstance(content, dict) else atomic_write_text)(path, content)
        if plan.config_text is not None:
            atomic_write_text(config_file, plan.config_text)
    except OSError as exc:
        raise NeutralError(FAILED) from exc
    return plan.swept()


def _plan(config_file: Path, config: Configuration, names: Names) -> _Plan:
    """Everything the sweep would write; writes nothing but, on first need, `this_machine`."""
    plan = _Plan()
    classes, entered = _profiles(config.paths.hardware_dir, names, plan)
    if names.bound in entered:  # a binding to a machine entered by hand makes no row this computer's
        names = replace(names, bound=None)
    text = _read_text(config_file)
    raw = _toml(text)
    if raw is not None and raw.get("schema_version") != 1:  # schema 1 is `migrate`'s: its keys name its profile files
        keys = machine_keys(_entries(raw), classes, names)
        plan.renames = {old: new for old, new in keys.items() if old != new}
        if plan.renames:
            plan.config_text = _renamed_config(text, raw, plan.renames, config_file)
    _records(config, plan)
    for backup in _backups(config_file, config.paths.hardware_dir):
        _backup(backup, config_file.name, classes, names, plan)
    return plan


def _profiles(hardware_dir: Path, names: Names, plan: _Plan) -> tuple[dict[str, str], set[str]]:
    """Plan the `display_name` of every profile that reads; return each profile's class, and the
    profiles entered by hand (`guided_machines._measured_here` counts none of them as measured here)."""
    classes: dict[str, str] = {}
    entered: set[str] = set()
    for path in sorted(hardware_dir.glob("*.json")) if hardware_dir.is_dir() else []:
        data = _read_json(path)
        profile_id = _profile_id(data)
        if profile_id is None:  # a schema-1 file (`migrate` converts it) or one that does not read
            continue
        classes[profile_id] = str(data.get("machine_class") or "unknown")
        if data.get("ram_physical_source") == "entered" and data.get("origin") == "entered":
            entered.add(profile_id)
        name = display_name_for(profile_id, classes[profile_id], data.get("display_name"), names.aliases.get(profile_id))
        if data.get("display_name") != name:
            plan.writes.append((path, {**data, "display_name": name}))
    return classes, entered


def snapshot_packages(config: Configuration) -> list[Package]:
    """The packages of the folder's own snapshot, the identities a record's name may be built from;
    none where there is no snapshot or it does not read (its readers say so)."""
    try:
        snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError):
        return []
    return list(snapshot.packages) if snapshot is not None else []


def _records(config: Configuration, plan: _Plan) -> None:
    """Plan every measurement record that the record rules change; a file that does not read stays."""
    root = config.paths.state / "measurements"
    packages = snapshot_packages(config)
    for path in sorted(root.glob("*/*.json")) if root.is_dir() else []:
        data = _read_json(path)
        try:
            record = MeasurementRecord.model_validate(data)
        except ValueError:  # not a record: listed by its readers, never read as one
            continue
        clean = neutral_record(record, known_package(record, packages))
        if clean != record:
            plan.writes.append((path, clean.model_dump(mode="json")))


def _backups(config_file: Path, hardware_dir: Path) -> list[Path]:
    folder = config_file.parent
    found = [*folder.glob(f"{config_file.name}.bak"), *folder.glob(f"{config_file.name}.v*.bak")]
    found += sorted(hardware_dir.glob("*.json.v*.bak")) if hardware_dir.is_dir() else []
    return [path for path in found if path.is_file()]


def _backup(path: Path, config_name: str, classes: dict[str, str], names: Names, plan: _Plan) -> None:
    """Plan one backup, or note that it cannot be cleaned; its content is never shown."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return _note(plan, path, config_name)
    if path.name.startswith(config_name):
        raw = _toml(text)
        if raw is None:
            return _note(plan, path, config_name)
        # A key the configuration itself is renamed from takes the same new name there.
        entries = _entries(raw)
        keys = machine_keys({key: profile for key, profile in entries.items() if key not in plan.renames}, classes, names)
        keys.update({key: plan.renames[key] for key in entries if key in plan.renames})
        renames = {old: new for old, new in keys.items() if old != new}
        cleaned = _rename_heads(text, renames)
        if renames and _machines(cleaned) != _renamed(raw, renames):
            return _note(plan, path, config_name)
        if renames:
            plan.writes.append((path, cleaned))
        return None
    data = _loads(text)
    profile_id = _profile_id(data)
    if profile_id is None:
        return _note(plan, path, config_name)
    name = display_name_for(profile_id, data.get("machine_class"), data.get("display_name"), names.aliases.get(profile_id))
    if data.get("display_name") != name:
        plan.writes.append((path, {**data, "display_name": name}))
    return None


def _note(plan: _Plan, path: Path, config_name: str) -> None:
    """Name the file only where its name is modelroom's own: the configuration's or a profile_id."""
    own = path.name.startswith(config_name) or PROFILE_ID_RE.fullmatch(path.name.split(".", 1)[0]) is not None
    plan.notes.append(_BACKUP_NOTE.format(name=path.name if own else _LEGACY_BACKUP))


def _renamed_config(text: str, raw: dict, renames: dict[str, str], config_file: Path) -> str:
    """The configuration with its machines renamed: the table heads replaced, else written anew."""
    expected = _renamed(raw, renames)
    candidate = _rename_heads(text, renames)
    if _machines(candidate) != expected:
        try:
            candidate = dump_toml({**raw, "machines": expected}, header=_HEADER)
        except TypeError as exc:
            raise NeutralError(FAILED) from exc
    try:
        config_from_text(candidate, config_file)
    except (ConfigError, SchemaVersionError) as exc:
        raise NeutralError(FAILED) from exc
    return candidate


def _rename_heads(text: str, renames: dict[str, str]) -> str:
    def swap(match: re.Match) -> str:
        key = match.group("key").strip("\"'")
        return f"{match.group('lead')}{renames.get(key, key)}{match.group('tail')}" if key in renames else match.group(0)

    return _HEAD_RE.sub(swap, text)


def _renamed(raw: dict, renames: dict[str, str]) -> dict:
    machines = raw.get("machines") if isinstance(raw.get("machines"), dict) else {}
    return {renames.get(key, key): value for key, value in machines.items()}


def _machines(text: str) -> dict | None:
    raw = _toml(text)
    machines = raw.get("machines", {}) if raw is not None else None
    return machines if isinstance(machines, dict) else None


def _entries(raw: dict) -> dict[str, str | None]:
    machines = raw.get("machines") if isinstance(raw.get("machines"), dict) else {}
    return {key: _profile_of(entry) for key, entry in machines.items()}


def _profile_of(entry: object) -> str | None:
    profile = entry.get("profile") if isinstance(entry, dict) else None
    return profile if isinstance(profile, str) and PROFILE_ID_RE.fullmatch(profile) else None


def _profile_id(data: object) -> str | None:
    profile_id = data.get("profile_id") if isinstance(data, dict) else None
    return profile_id if isinstance(profile_id, str) and PROFILE_ID_RE.fullmatch(profile_id) else None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise NeutralError(FAILED) from exc


def _read_json(path: Path) -> object | None:
    """The JSON of a file of the folder; `None` where it does not parse. A file that cannot be read at
    all may hold a name the sweep cannot clean: that ends the command."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError as exc:
        raise NeutralError(FAILED) from exc
    return _loads(text)


def _loads(text: str) -> object | None:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _toml(text: str) -> dict | None:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None


# --- the commands ------------------------------------------------------------------------------


def sweep_for_command(
    config_file: Path,
    command: str,
    now: datetime | None,
    pointer_path: Path | None,
    hostname: Callable[[], str] | None,
) -> tuple[int | None, Swept]:
    """A subcommand's first step: the exit code it ends with (`1` swept nothing, `3` `remote.json`
    is broken), or `None` and what the sweep did, its notes printed."""
    started = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    try:
        swept = sweep(config_file, started, command, pointer_path=pointer_path, hostname=hostname() if hostname else None)
    except NeutralError as exc:
        print(str(exc), file=sys.stderr)
        return 1, Swept()
    except RemoteStoreError as exc:
        print(str(exc), file=sys.stderr)
        return 3, Swept()
    for note in swept.notes:
        print(f"note: {note}")
    return None, swept


def moved_machine(name: str | None, swept: Swept) -> str | None:
    """A `--machine` of this call, led to its new key; the old name is not repeated."""
    if name is None or name not in swept.renames:
        return name
    print(f"note: --machine now names {swept.renames[name]}")
    return swept.renames[name]


__all__ = ["Swept", "load_names", "moved_machine", "sweep", "sweep_for_command", "sweep_locked"]
