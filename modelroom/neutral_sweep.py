"""`sweep`: the sink of `modelroom/neutral.py` over every file a results folder already holds (FR-N).

Every writing command runs it first, under `modelroom.lock` (CONTRACTS.md, "Neutral names"): the
profiles (`display_name` only), the configuration (keys, comments and paths, `neutral_config`), the
records (field by field), modelroom's own backups (`neutral_backups`), and the result documents --
removed where anything else changes or where either of them names a machine. A profile or record that
does not read as a JSON object cannot be cleaned: `neutral.FAILED`, exit `1`; only a backup is a note
instead. No new backup is ever made; a second sweep changes nothing. The folder is read without the
lock first; only with something to change is the lock taken, the folder read again and checked to be
that lock's, and the files written.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .binding import PointerFileError, default_pointer_path, read_pointer
from .config import ConfigError, Configuration, config_from_text, load_config
from .contracts import Package, SchemaVersionError
from .neutral import FAILED, Names, NeutralError, display_name_for, is_alias, is_neutral, known_package, machine_keys, old_key, record_data
from .neutral_backups import Plan, Swept, backups, legacy_owners, plan_backup, profile_id_of
from .neutral_config import config_text, entries, expected
from .remote_store import RemoteStoreError, read_remote, remote_path_for, this_machine
from .state import LockHandle, LockHeldError, UnreadableStateFileError, acquire_lock, atomic_write_json, atomic_write_text, read_snapshot, release_lock

DOCUMENTS_NOTE = "the result documents were made from names this sweep changed; they were removed, the next render writes them"
_HEADING_RE = re.compile(r"^## [^:\n]+: (.+)$", re.M)


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


def _plan(config_file: Path, config: Configuration, names: Names) -> Plan:
    """Everything the sweep would write; writes nothing but, on first need, `this_machine`."""
    plan = Plan()
    profiles, entered = _profiles(config.paths.hardware_dir, names, plan)
    if names.bound in entered:  # a binding to a machine entered by hand makes no row this computer's
        names = replace(names, bound=None)
    classes = {key: str(data.get("machine_class") or "unknown") for key, data in profiles.items()}
    text = _read_text(config_file)
    raw = _toml(text)
    current = raw is not None and raw.get("schema_version") != 1  # schema 1 is `migrate`'s: its keys name its profile files
    if current:
        keys = machine_keys(entries(raw), classes, names)
        plan.renames = {old: new for old, new in keys.items() if old != new}
        if expected(raw, plan.renames, config_file.parent) != raw:
            plan.config_text = _valid(config_text(text, raw, plan.renames, config_file.parent), config_file)
    _records(config, plan)
    for backup in backups(config_file, config.paths.hardware_dir):
        plan_backup(backup, config_file, classes, names, profiles, plan)
    documents(config, names, plan, by_names=current)
    return plan


def _valid(text: str, config_file: Path) -> str:
    try:
        config_from_text(text, config_file)
    except (ConfigError, SchemaVersionError, ValueError) as exc:
        raise NeutralError(FAILED) from exc
    return text


def _profiles(hardware_dir: Path, names: Names, plan: Plan) -> tuple[dict[str, dict], set[str]]:
    """Plan the `display_name` of every profile; return each profile as stored, and the ones entered by
    hand (`guided_machines._measured_here` counts none of them as measured here)."""
    profiles: dict[str, dict] = {}
    entered: set[str] = set()
    for path in sorted(hardware_dir.glob("*.json")) if hardware_dir.is_dir() else []:
        data = _strict(path)
        profile_id = profile_id_of(data)
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


def _records(config: Configuration, plan: Plan) -> None:
    """Plan every measurement record the record rules change, whether or not it validates otherwise."""
    root = config.paths.state / "measurements"
    packages = snapshot_packages(config)
    for path in sorted(root.glob("*/*.json")) if root.is_dir() else []:
        data = _strict(path)
        clean = record_data(data, known_package(data.get("package"), packages))
        if clean != data:
            plan.writes.append((path, clean))


def documents(config: Configuration, names: Names, plan: Plan, *, by_names: bool) -> None:
    """The result documents go where the plan changes anything else, or where either of them names a
    machine by anything but a neutral name or an alias of this user (`by_names`; a schema-1
    configuration's documents are `migrate`'s to judge, its keys being still the old ones)."""
    markdown = config.paths.markdown
    view = markdown.with_suffix(".json")
    present = [path for path in (view, markdown) if path.is_file()]
    named = by_names and any(not _allowed(value, names) for value in [*_view_shows(view), *_markdown_shows(markdown)])
    if present and (plan.writes_something() or named):
        plan.removals += present
        plan.notes.append(DOCUMENTS_NOTE)


def _allowed(value: object, names: Names) -> bool:
    return is_neutral(value) or (is_alias(value) and value in set(names.aliases.values()))


def _view_shows(view: Path) -> list[object]:
    """What the JSON view names machines by; a view that does not read cannot be shown to be clean."""
    if not view.is_file():
        return []
    try:
        blocks = json.loads(view.read_text(encoding="utf-8"))["machines"]
        rows = [(block["machine"], str(block["label"]).split(" ")[0], (block.get("profile") or {}).get("display_name")) for block in blocks]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return [None]
    return [value for row in rows for value in row if value is not None]


def _markdown_shows(markdown: Path) -> list[object]:
    """What the Markdown view names machines by: the machine headings (`## <what>: <machine>`) and the
    first two columns of the Machines table (key; profile name or label)."""
    if not markdown.is_file():
        return []
    try:
        text = markdown.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [None]
    shown: list[object] = [value.strip() for value in _HEADING_RE.findall(text)]
    table = re.split(r"^## Machines$", text, maxsplit=1, flags=re.M)
    for line in (table[1].split("\n## ", 1)[0] if len(table) == 2 else "").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if len(cells) >= 2 and cells[0] != "Machine" and set(cells[0]) - set("-: "):
            shown += [cells[0], cells[1].split(" ")[0]]
    return shown


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise NeutralError(FAILED) from exc


def _strict(path: Path) -> dict:
    """A profile or a record of the folder as a JSON object; anything else cannot be cleaned and ends
    the command -- a broken file may hold a name as well as any other."""
    try:
        data = json.loads(_read_text(path))
    except (ValueError, RecursionError) as exc:
        raise NeutralError(FAILED) from exc
    if not isinstance(data, dict):
        raise NeutralError(FAILED)
    return data


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


__all__ = [
    "DOCUMENTS_NOTE",
    "Swept",
    "documents",
    "legacy_owners",
    "load_names",
    "moved_machine",
    "snapshot_packages",
    "sweep",
    "sweep_for_command",
    "sweep_locked",
]
