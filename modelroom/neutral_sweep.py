"""`sweep`: the sink of `modelroom/neutral.py` over every file a results folder already holds (FR-N).

Every writing command runs it first, under `modelroom.lock` (CONTRACTS.md, "Neutral names"): the
profiles (`display_name` only), the configuration (keys and paths, `neutral_config`), the records
(field by field), modelroom's own backups (a schema-1 profile's `<old key>.json.v1.bak` takes a neutral
name), and the result documents -- removed where anything else changes or where they name a machine.
A profile or record that does not read as a JSON object cannot be cleaned: `neutral.FAILED`, exit `1`;
only a backup is a note instead. No new backup is ever made; a second sweep changes nothing. The
folder is read without the lock first; only with something to change is the lock taken, the folder
read again and checked to be that lock's, and the files written.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .binding import PointerFileError, default_pointer_path, read_pointer
from .config import ConfigError, Configuration, config_from_text, load_config
from .contracts import PROFILE_ID_RE, HardwareSnapshot, Package, SchemaVersionError
from .neutral import FAILED, Names, NeutralError, display_name_for, is_alias, is_neutral, known_package, legacy_data
from .neutral import machine_keys, neutral_name, old_key, record_data, renamed_note
from .neutral_config import config_text, entries, expected
from .profile import HardwareProfile, normalize_profile_v1, read_profile_document
from .remote_store import RemoteStoreError, read_remote, remote_path_for, this_machine
from .state import LockHandle, LockHeldError, UnreadableStateFileError, acquire_lock, atomic_write_json, atomic_write_text, read_snapshot, release_lock

_BACKUP_NOTE = "a backup of modelroom could not be cleaned of machine names: {name}; delete it by hand"
DOCUMENTS_NOTE = "the result documents were made from names this sweep changed; they were removed, the next render writes them"


@dataclass
class Swept:
    """What a sweep did: the machines it renamed (old key -> new key) and its notes."""

    renames: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Plan:
    writes: list[tuple[Path, dict | str]] = field(default_factory=list)
    moves: list[tuple[Path, Path, dict]] = field(default_factory=list)
    removals: list[Path] = field(default_factory=list)
    config_text: str | None = None
    renames: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def writes_something(self) -> bool:
        return bool(self.writes or self.moves or self.removals) or self.config_text is not None

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
    config = _config(config_file)
    if config is None:
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
        return sweep_locked(config_file, names, handle)
    finally:
        release_lock(handle)


def sweep_locked(config_file: Path, names: Names, handle: LockHandle) -> Swept:
    """The sweep itself, under `handle`; a configuration whose state folder is no longer the one of that
    lock (it changed while the lock was taken) is not swept but refused."""
    config = _config(config_file)
    if config is None:
        return Swept()
    if config.paths.lock_file.resolve() != handle.path.resolve():
        raise NeutralError(FAILED)
    plan = _plan(config_file, config, names)
    try:
        for path, content in plan.writes:
            (atomic_write_json if isinstance(content, dict) else atomic_write_text)(path, content)
        for old, new, content in plan.moves:
            atomic_write_json(new, content)
            old.unlink()
        for path in plan.removals:
            path.unlink(missing_ok=True)
        if plan.config_text is not None:
            atomic_write_text(config_file, plan.config_text)
    except OSError as exc:
        raise NeutralError(FAILED) from exc
    return plan.swept()


def _config(config_file: Path) -> Configuration | None:
    try:
        return load_config(config_file)
    except (ConfigError, SchemaVersionError, ValueError, OSError):
        return None


def _plan(config_file: Path, config: Configuration, names: Names) -> _Plan:
    """Everything the sweep would write; writes nothing but, on first need, `this_machine`."""
    plan = _Plan()
    profiles, entered = _profiles(config.paths.hardware_dir, names, plan)
    if names.bound in entered:  # a binding to a machine entered by hand makes no row this computer's
        names = replace(names, bound=None)
    classes = {key: str(data.get("machine_class") or "unknown") for key, data in profiles.items()}
    text = _read_text(config_file)
    raw = _toml(text)
    if raw is not None and raw.get("schema_version") != 1:  # schema 1 is `migrate`'s: its keys name its profile files
        keys = machine_keys(entries(raw), classes, names)
        plan.renames = {old: new for old, new in keys.items() if old != new}
        if expected(raw, plan.renames, config_file.parent) != raw:
            plan.config_text = _valid(config_text(text, raw, plan.renames, config_file.parent), config_file)
    _records(config, plan)
    for backup in _backups(config_file, config.paths.hardware_dir):
        _backup(backup, config_file, classes, names, profiles, plan)
    # A schema-1 configuration keeps its keys until `migrate` (its keys name its profile files), so its
    # documents are judged by what the sweep changes alone.
    _documents(config, names, plan, raw is not None and raw.get("schema_version") != 1)
    return plan


def _valid(text: str, config_file: Path) -> str:
    try:
        config_from_text(text, config_file)
    except (ConfigError, SchemaVersionError, ValueError) as exc:
        raise NeutralError(FAILED) from exc
    return text


def _profiles(hardware_dir: Path, names: Names, plan: _Plan) -> tuple[dict[str, dict], set[str]]:
    """Plan the `display_name` of every profile; return each profile as stored, and the ones entered by
    hand (`guided_machines._measured_here` counts none of them as measured here)."""
    profiles: dict[str, dict] = {}
    entered: set[str] = set()
    for path in sorted(hardware_dir.glob("*.json")) if hardware_dir.is_dir() else []:
        data = _strict(path)
        profile_id = _profile_id(data)
        if profile_id is None:
            if data.get("schema_version") == 1 and "machine" in data:
                continue  # a schema-1 file: `migrate` converts it, and its name with it
            if "display_name" in data:
                raise NeutralError(FAILED)  # a name without the id its neutral one is made of
            continue
        profiles[profile_id] = data
        if data.get("ram_physical_source") == "entered" and data.get("origin") == "entered":
            entered.add(profile_id)
        name = display_name_for(profile_id, data.get("machine_class"), data.get("display_name"), names.aliases.get(profile_id))
        if data.get("display_name") != name:
            plan.writes.append((path, {**data, "display_name": name}))
    return profiles, entered


def snapshot_packages(config: Configuration) -> list[Package]:
    """The packages of the folder's own snapshot, the identities a record's name may be built from;
    none where there is no snapshot or it does not read (its readers say so)."""
    try:
        snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError):
        return []
    return list(snapshot.packages) if snapshot is not None else []


def _records(config: Configuration, plan: _Plan) -> None:
    """Plan every measurement record the record rules change, whether or not it validates otherwise."""
    root = config.paths.state / "measurements"
    packages = snapshot_packages(config)
    for path in sorted(root.glob("*/*.json")) if root.is_dir() else []:
        data = _strict(path)
        clean = record_data(data, known_package(data.get("package"), packages))
        if clean != data:
            plan.writes.append((path, clean))


def _backups(config_file: Path, hardware_dir: Path) -> list[Path]:
    folder = config_file.parent
    found = [*folder.glob(f"{config_file.name}.bak"), *folder.glob(f"{config_file.name}.v*.bak")]
    found += sorted(hardware_dir.glob("*.json.v*.bak")) if hardware_dir.is_dir() else []
    return [path for path in found if path.is_file()]


def _backup(path: Path, config_file: Path, classes: dict, names: Names, profiles: dict[str, dict], plan: _Plan) -> None:
    """Plan one backup, or note that it cannot be cleaned; its content is never shown."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return _note(plan, path)
    if path.name.startswith(config_file.name):
        return _config_backup(path, text, config_file.parent, classes, names, plan)
    data = _loads(text)
    if isinstance(data, dict) and data.get("schema_version") == 1 and "machine" in data:
        return _legacy_backup(path, text, data, profiles, plan)
    profile_id = _profile_id(data)
    if profile_id is None:
        return _note(plan, path)
    name = display_name_for(profile_id, data.get("machine_class"), data.get("display_name"), names.aliases.get(profile_id))
    if data.get("display_name") != name:
        plan.writes.append((path, {**data, "display_name": name}))
    return None


def _config_backup(path: Path, text: str, folder: Path, classes: dict, names: Names, plan: _Plan) -> None:
    """A backup of the configuration, by the same key rule; a key the configuration itself is renamed from
    takes the same new name here."""
    raw = _toml(text)
    if raw is None:
        return _note(plan, path)
    rows = entries(raw)
    keys = machine_keys({key: profile for key, profile in rows.items() if key not in plan.renames}, classes, names)
    keys.update({key: plan.renames[key] for key in rows if key in plan.renames})
    renames = {old: new for old, new in keys.items() if old != new}
    if expected(raw, renames, folder) != raw:
        plan.writes.append((path, config_text(text, raw, renames, folder)))
    return None


def _legacy_backup(path: Path, text: str, data: dict, profiles: dict[str, dict], plan: _Plan) -> None:
    """A schema-1 profile kept by `migrate`: its name was the old key. It takes the name of the profile it
    became (`<profile_id>.json.v1.bak`), else one of its own content (`legacy-<8 hex>`), and a neutral key."""
    stem = path.name.split(".", 1)[0]
    owners = legacy_owners(data, profiles)
    short = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    if PROFILE_ID_RE.fullmatch(stem) or stem.startswith("legacy-"):
        target, key = path, data.get("machine")
    elif len(owners) == 1:
        target, key = path.with_name(f"{owners[0]}.json.v1.bak"), neutral_name(owners[0], "unknown")
    else:
        target, key = path.with_name(f"legacy-{short}.json.v1.bak"), f"computer-{short}"
    key = key if is_neutral(key) else f"computer-{short}"
    clean = legacy_data(data, key)
    if target == path:
        if clean != data:
            plan.writes.append((path, clean))
    elif target.exists():
        _note(plan, target)
    else:
        plan.moves.append((path, target, clean))
    return None


def legacy_owners(data: dict, profiles: dict[str, dict]) -> list[str]:
    """Every profile a schema-1 file became: all of it the same but its id and its name (`migrate`)."""
    try:
        fresh = normalize_profile_v1(HardwareSnapshot.model_validate(data), "0" * 16)
    except ValueError:
        return []

    def shape(profile: HardwareProfile) -> dict:
        return {**profile.model_dump(mode="json"), "profile_id": None, "display_name": None}

    owners = []
    for profile_id, stored in profiles.items():
        try:
            profile = read_profile_document(stored)
        except (SchemaVersionError, ValueError):
            continue
        if isinstance(profile, HardwareProfile) and profile.os_fingerprint_source == "legacy" and shape(profile) == shape(fresh):
            owners.append(profile_id)
    return owners


def _documents(config: Configuration, names: Names, plan: _Plan, by_names: bool) -> None:
    """The result documents go where the sweep changes anything else, or where they name a machine."""
    markdown = config.paths.markdown
    present = [path for path in (markdown.with_suffix(".json"), markdown) if path.is_file()]
    if present and (plan.writes_something() or (by_names and _names_a_machine(markdown.with_suffix(".json"), names))):
        plan.removals += present
        plan.notes.append(DOCUMENTS_NOTE)


def _names_a_machine(view: Path, names: Names) -> bool:
    """Whether the JSON view names a machine by anything but a neutral name or an alias of this user."""
    try:
        blocks = json.loads(view.read_text(encoding="utf-8"))["machines"]
        shown = [(block["machine"], str(block["label"]).split(" ")[0], (block.get("profile") or {}).get("display_name")) for block in blocks]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return True  # a view that does not read cannot be shown to be clean
    allowed = set(names.aliases.values())
    return any(not (is_neutral(value) or (is_alias(value) and value in allowed)) for row in shown for value in row if value is not None)


def _note(plan: _Plan, path: Path) -> None:
    """Name the file only where its name is modelroom's own: the configuration's or a profile_id."""
    stem = path.name.split(".", 1)[0]
    own = path.name.startswith("modelroom.toml") or PROFILE_ID_RE.fullmatch(stem) is not None or stem.startswith("legacy-")
    plan.notes.append(_BACKUP_NOTE.format(name=path.name if own else "a profile backup in hardware/"))


def _profile_id(data: object) -> str | None:
    profile_id = data.get("profile_id") if isinstance(data, dict) else None
    return profile_id if isinstance(profile_id, str) and PROFILE_ID_RE.fullmatch(profile_id) else None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise NeutralError(FAILED) from exc


def _strict(path: Path) -> dict:
    """A profile or a record of the folder as a JSON object; anything else cannot be cleaned and ends
    the command -- a broken file may hold a name as well as any other."""
    data = _loads(_read_text(path))
    if not isinstance(data, dict):
        raise NeutralError(FAILED)
    return data


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


__all__ = ["DOCUMENTS_NOTE", "Swept", "legacy_owners", "load_names", "moved_machine", "snapshot_packages", "sweep", "sweep_for_command", "sweep_locked"]
