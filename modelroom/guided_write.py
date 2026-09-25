"""Every write of the guided mode's configuration: read, change, write -- and once more on a conflict.

The dialog takes as long as the user takes, and another run may write `modelroom.toml` meanwhile
(an `import-profile` on a shared folder, a second guided run). Until 2026-09-25 a step read the
file, computed its change and wrote the result under `modelroom.lock`; a run that wrote between
that read and that write was written over. Now every step hands in its **change** -- a function
from the configuration it reads to the one it writes, or `None` when there is nothing to write --
and `update_config`:

1. reads the file once, keeping the text of that very read (`read_stored`);
2. applies the change to that configuration;
3. writes through `search_apply.write_configuration`, which compares the file with that text under
   the lock and raises `ConfigChangedError` when it differs;
4. on that conflict reads again, applies the **same change** to the new state and writes once more;
   a second conflict in a row goes to the caller (the guided mode ends with exit `2`).

It returns the configuration that was really written, or the one it read when there was nothing
to write -- the caller works on with that, never with its own earlier copy. No merge of two
configurations is attempted anywhere: each step only ever applies its own change again.

The first write of a new file has a rule of its own (`create_config`): a complete initial state
with empty families is no change that could be applied again, so when another run created the file
meanwhile, that file is taken over as it is and nothing is written. Its values stay; this machine's
own entry is added by the writer step, as in any folder someone else set up.

The changes of the individual steps live here as well (`with_writer`, `with_profile`,
`with_context`, `with_found_profiles`), so each can be tested on its own. CONTRACTS.md,
"Configuration" and "Guided mode".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

from .config import ConfigError, Configuration, PathsConfig, config_from_text
from .importer import ImportConflictError, machine_name_for, scan_profiles
from .search_apply import ConfigChangedError, write_configuration

Change = Callable[[Configuration], "Configuration | None"]


class Stored(NamedTuple):
    """One read of the configuration file: the parsed configuration and the text it came from."""

    config: Configuration
    text: str


@dataclass
class Said:
    """What a change has to say once it is written: notes, and problems the run reports.

    A change may run twice (once more after a conflict), so it clears this first; what is left is
    what the attempt that counted found.
    """

    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.notes.clear()
        self.problems.clear()


def read_stored(config_file: Path) -> Stored:
    """Read the configuration once, with the same checks `load_config` applies, and keep the text."""
    path = config_file.resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: config file not found") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: cannot read config file: {exc}") from exc
    return Stored(config_from_text(text, path), text)


def update_config(config_file: Path, change: Change, *, now: datetime) -> Configuration:
    """Apply `change` to the file as it is and write the result; once more after a conflict."""
    try:
        return _apply(config_file, change, now)
    except ConfigChangedError:
        return _apply(config_file, change, now)


def _apply(config_file: Path, change: Change, now: datetime) -> Configuration:
    stored = read_stored(config_file)
    updated = change(stored.config)
    if updated is None:
        return stored.config
    write_configuration(config_file, updated, now=now, expected_text=stored.text)
    return updated


def create_config(config_file: Path, config: Configuration, *, now: datetime) -> Configuration:
    """Write a new configuration file, or take over the one another run created meanwhile."""
    try:
        write_configuration(config_file, config, now=now, expected_text=None)
    except ConfigChangedError:
        return read_stored(config_file).config
    return config


def with_writer(name: str) -> Change:
    """`[machines.<name>]` present and a writer: this machine is the one that fetches."""

    def change(current: Configuration) -> Configuration | None:
        machine = current.machines.get(name)
        if machine is not None and machine.writer:
            return None
        data = current.model_dump(mode="json")
        entry = data["machines"].get(name, {})
        entry.setdefault("reserve_ram_gib", current.defaults.reserve_ram_gib)
        entry.setdefault("reserve_vram_gib", current.defaults.reserve_vram_gib)
        entry["writer"] = True
        data["machines"][name] = entry
        return Configuration.from_dict(data)

    return change


def with_profile(name: str, profile_id: str | None, measured_paths: PathsConfig, config_file: Path, said: Said) -> Change:
    """`[machines.<name>].profile` names the profile this machine was just measured as.

    The two checks hold for every read, the one after a conflict included: a `[paths]` that is no
    longer the one the measurement wrote into, or a machine entry that is gone, is a problem the
    run reports, and nothing is written.
    """

    def change(current: Configuration) -> Configuration | None:
        said.clear()
        if current.paths != measured_paths:
            said.problems.append(f"{config_file}: [paths] changed while this machine was measured; run again")
            return None
        if name not in current.machines:
            said.problems.append(f"{config_file}: the machine {name!r} is gone from the configuration; run again")
            return None
        if profile_id is None or current.machines[name].profile == profile_id:
            return None
        data = current.model_dump(mode="json")
        data["machines"][name]["profile"] = profile_id
        return Configuration.from_dict(data)

    return change


def with_context(context: int) -> Change:
    """`[guided].context` holds the context this run chose, compared with the file, not a copy."""

    def change(current: Configuration) -> Configuration | None:
        if context == current.guided.context:
            return None
        return current.model_copy(update={"guided": current.guided.model_copy(update={"context": context})})

    return change


def with_found_profiles(said: Said) -> Change:
    """Every profile of the folder without a `[machines.<name>]` entry gets one.

    The whole matching runs on every read -- scan, which profiles are configured, the names, the
    reserves -- so a name another run took meanwhile is not written over and a profile it entered
    is not entered twice. Name from the `display_name`, reserves from `[defaults]`, `writer` false:
    the rule `import-profile` follows.
    """

    def change(current: Configuration) -> Configuration | None:
        said.clear()
        scan = scan_profiles(current.paths.hardware_dir)
        configured = {machine.profile for machine in current.machines.values() if machine.profile is not None}
        data = current.model_dump(mode="json")
        for key, profile in sorted(scan.profiles.items()):
            if key in configured:
                continue
            try:
                name = machine_name_for(profile.display_name, profile.profile_id, set(data["machines"]))
            except ImportConflictError as exc:
                said.problems.append(f"the profile {profile.profile_id} stays without a machine entry: {exc}")
                continue
            data["machines"][name] = {
                "reserve_ram_gib": current.defaults.reserve_ram_gib,
                "reserve_vram_gib": current.defaults.reserve_vram_gib,
                "writer": False,
                "profile": profile.profile_id,
            }
            said.notes.append(f"machine {name!r} added for the profile {profile.profile_id} found in this folder")
        return Configuration.from_dict(data) if said.notes else None

    return change


__all__ = [
    "Change",
    "Said",
    "Stored",
    "create_config",
    "read_stored",
    "update_config",
    "with_context",
    "with_found_profiles",
    "with_profile",
    "with_writer",
]
