"""The sweep's plan and modelroom's own backups in it (FR-N; `neutral_sweep` runs it).

A backup is cleaned by the rules of the file it keeps: a configuration's by the key rule (a key the
configuration itself is renamed from takes the same new name), a profile's `display_name` by the
name rule. A schema-1 profile `migrate` kept as `<old key>.json.v1.bak` names a machine by its file
name as well: it takes the name of the profile it became (`<profile_id>.json.v1.bak`), else one of
its own content (`legacy-<8 hex>.json.v1.bak`), and a neutral key. Every target is reserved in the
plan before it is written; a target that already holds the same cleaned content (another copy, or a
run that stopped between writing the target and removing the source) takes the source's place, and
one of other content sends the source on to its content name. A backup that cannot be read is a
note naming the file where its name is modelroom's own; its content is never shown.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import PROFILE_ID_RE, HardwareSnapshot, SchemaVersionError
from .neutral import Names, display_name_for, is_neutral, legacy_data, machine_keys, neutral_name, renamed_note
from .neutral_config import config_text, entries, expected
from .profile import HardwareProfile, normalize_profile_v1, read_profile_document

BACKUP_NOTE = "a backup of modelroom could not be cleaned of machine names: {name}; delete it by hand"
_LEGACY_RE = re.compile(r"legacy-[0-9a-f]{8}")


@dataclass
class Swept:
    """What a sweep did: the machines it renamed (old key -> new key) and its notes."""

    renames: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """Everything a sweep writes: files, moves (new name, then the old one removed), removals, and the
    configuration last. `reserved` holds every backup name the plan leaves taken, with its content."""

    writes: list[tuple[Path, dict | str]] = field(default_factory=list)
    moves: list[tuple[Path, Path, dict]] = field(default_factory=list)
    removals: list[Path] = field(default_factory=list)
    config_text: str | None = None
    renames: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    reserved: dict[Path, dict] = field(default_factory=dict)

    def writes_something(self) -> bool:
        return bool(self.writes or self.moves or self.removals) or self.config_text is not None

    def swept(self) -> Swept:
        return Swept(dict(self.renames), [*(renamed_note(new) for new in self.renames.values()), *self.notes])


def backups(config_file: Path, hardware_dir: Path) -> list[Path]:
    """Modelroom's own backups of a folder; the ones already under a neutral name come first, so the
    names they hold are reserved before a legacy backup looks for its own."""
    folder = config_file.parent
    found = [*folder.glob(f"{config_file.name}.bak"), *folder.glob(f"{config_file.name}.v*.bak")]
    found += sorted(hardware_dir.glob("*.json.v*.bak")) if hardware_dir.is_dir() else []
    return sorted((path for path in found if path.is_file()), key=lambda path: not own_name(path))


def plan_backup(path: Path, config_file: Path, classes: dict, names: Names, profiles: dict[str, dict], plan: Plan) -> None:
    """Plan one backup, or note that it cannot be cleaned."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return note(plan, path)
    if path.name.startswith(config_file.name):
        return _config_backup(path, text, config_file.parent, classes, names, plan)
    data = _loads(text)
    if isinstance(data, dict) and data.get("schema_version") == 1 and "machine" in data:
        return _legacy_backup(path, text, data, profiles, plan)
    profile_id = profile_id_of(data)
    if profile_id is None:
        return note(plan, path)
    name = display_name_for(profile_id, data.get("machine_class"), data.get("display_name"), names.aliases.get(profile_id))
    if data.get("display_name") != name:
        plan.writes.append((path, {**data, "display_name": name}))
    return None


def _config_backup(path: Path, text: str, folder: Path, classes: dict, names: Names, plan: Plan) -> None:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return note(plan, path)
    rows = entries(raw)
    keys = machine_keys({key: profile for key, profile in rows.items() if key not in plan.renames}, classes, names)
    keys.update({key: plan.renames[key] for key in rows if key in plan.renames})
    renames = {old: new for old, new in keys.items() if old != new}
    if expected(raw, renames, folder) != raw:
        plan.writes.append((path, config_text(text, raw, renames, folder)))
    return None


def _legacy_backup(path: Path, text: str, data: dict, profiles: dict[str, dict], plan: Plan) -> None:
    short = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    if own_name(path):
        key = data.get("machine") if is_neutral(data.get("machine")) else f"computer-{short}"
        clean = legacy_data(data, key)
        if clean != data:
            plan.writes.append((path, clean))
        plan.reserved[path] = clean
        return None
    owners = legacy_owners(data, profiles)
    targets = [(path.with_name(f"{owners[0]}.json.v1.bak"), neutral_name(owners[0], "unknown"))] if len(owners) == 1 else []
    targets.append((path.with_name(f"legacy-{short}.json.v1.bak"), f"computer-{short}"))
    for target, key in targets:
        clean = legacy_data(data, key)
        if target not in plan.reserved and not target.exists():
            plan.moves.append((path, target, clean))
            plan.reserved[target] = clean
            return None
        if plan.reserved.get(target) == clean:  # the same backup is there already: this copy goes
            plan.removals.append(path)
            return None
    return note(plan, path)


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


def own_name(path: Path) -> bool:
    """A file name modelroom gave: the configuration's, a `profile_id`'s or `legacy-<8 hex>`."""
    stem = path.name.split(".", 1)[0]
    return path.name.startswith("modelroom.toml") or bool(PROFILE_ID_RE.fullmatch(stem) or _LEGACY_RE.fullmatch(stem))


def note(plan: Plan, path: Path) -> None:
    """Name the file only where its name is modelroom's own."""
    plan.notes.append(BACKUP_NOTE.format(name=path.name if own_name(path) else "a profile backup in hardware/"))


def profile_id_of(data: object) -> str | None:
    profile_id = data.get("profile_id") if isinstance(data, dict) else None
    return profile_id if isinstance(profile_id, str) and PROFILE_ID_RE.fullmatch(profile_id) else None


def _loads(text: str) -> object | None:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


__all__ = ["BACKUP_NOTE", "Plan", "Swept", "backups", "legacy_owners", "note", "own_name", "plan_backup", "profile_id_of"]
