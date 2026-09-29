"""The sink for a whole configuration (FR-N): neutral machine keys and paths relative to its folder.

A configuration names machines by their keys and may name its folder by absolute paths
(`paths.state`, `paths.markdown`, `[guided].results`), and a folder's path can carry a user's or a
machine's name. Every configuration modelroom writes goes through here first:

- `config_text` rewrites a stored text: each renamed key replaced as a TOML key token -- a table
  head, a dotted key, an inline table, quoted or bare -- and each path inside the folder replaced by
  its relative form, so every comment and every other byte stays. Where the text cannot be brought to
  exactly the expected table that way (an escaped key, a key spelled like a field of the file), the
  table is written anew by the TOML writer (the comments go; the file then says so in its head);
- `relative_paths` gives the table form of a configuration with those paths relative, for the
  writers that dump a whole table (`search_apply.write_configuration`, `import-profile`, `migrate`);
- `neutral_config` renames the keys of a `Configuration` by the key rule (`neutral.machine_keys`),
  for the guided mode's writes, applied to the configuration it read last, right before it writes.

A path outside the folder is the user's own choice and stays as it is.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from .config import Configuration
from .contracts import PROFILE_ID_RE
from .neutral import Names, machine_keys
from .toml_writer import dump_toml

HEADER = "modelroom configuration (machine names made neutral by modelroom)."
# The names of the configuration's own fields: a machine key spelled like one of them is never
# replaced as a token (it could be the field); such a file is written anew instead.
_FIELDS = frozenset(
    "schema_version families name base_models hf_repo repo_aliases repos ollama_base ollama_tag packagers publishers "
    "machines reserve_ram_gib reserve_vram_gib writer profile defaults updates check paths state markdown llmfit "
    "min_version guided results context users requests requests_origin".split()
)
_PATH_FIELDS = (("paths", "state"), ("paths", "markdown"), ("guided", "results"))


def entries(raw: dict) -> dict[str, str | None]:
    """`key -> profile` of every `[machines.<key>]` of a configuration table."""
    machines = raw.get("machines") if isinstance(raw.get("machines"), dict) else {}
    return {key: _profile_of(entry) for key, entry in machines.items()}


def _profile_of(entry: object) -> str | None:
    profile = entry.get("profile") if isinstance(entry, dict) else None
    return profile if isinstance(profile, str) and PROFILE_ID_RE.fullmatch(profile) else None


def relative_paths(raw: dict, folder: Path) -> dict:
    """`raw` with every absolute path inside `folder` relative to it (`.`, `state`, `docs/models.md`)."""
    data = {**raw}
    for section, key in _PATH_FIELDS:
        table = data.get(section)
        value = table.get(key) if isinstance(table, dict) else None
        relative = _relative(value, folder)
        if relative is not None:
            data[section] = {**table, key: relative}
    return data


def _relative(value: object, folder: Path) -> str | None:
    if not isinstance(value, str) or not Path(value).is_absolute():
        return None
    try:
        return Path(value).resolve().relative_to(folder.resolve()).as_posix() or "."
    except ValueError:  # outside the folder: the user's own choice
        return None


def expected(raw: dict, renames: dict[str, str], folder: Path) -> dict:
    """The table the neutral text has to read as: keys renamed, paths relative."""
    data = relative_paths(raw, folder)
    if isinstance(raw.get("machines"), dict):
        data["machines"] = {renames.get(key, key): value for key, value in raw["machines"].items()}
    return data


def config_text(text: str, raw: dict, renames: dict[str, str], folder: Path) -> str:
    """The stored text brought to `expected`, its bytes kept where they can be; else the table anew."""
    wanted = expected(raw, renames, folder)
    candidate = _replace_paths(_rename_tokens(text, renames), raw, folder)
    if _loads(candidate) == wanted:
        return candidate
    return dump_toml(wanted, header=HEADER)


def _rename_tokens(text: str, renames: dict[str, str]) -> str:
    """Every renamed key where it stands as a TOML key: followed by `.`, `=` or `]`, bare or quoted."""
    olds = [old for old in renames if old not in _FIELDS]
    if not olds:
        return text
    words = "|".join(re.escape(old) for old in sorted(olds, key=len, reverse=True))
    pattern = re.compile(rf"""(?<![A-Za-z0-9_"'-])(?P<quote>["']?)(?P<key>{words})(?P=quote)(?=[ \t]*[.=\]])""")
    return pattern.sub(lambda match: renames[match.group("key")], text)


def _replace_paths(text: str, raw: dict, folder: Path) -> str:
    for section, key in _PATH_FIELDS:
        table = raw.get(section)
        value = table.get(key) if isinstance(table, dict) else None
        relative = _relative(value, folder)
        if relative is not None:
            text = text.replace(json.dumps(value, ensure_ascii=False), json.dumps(relative, ensure_ascii=False))
    return text


def _loads(text: str) -> dict | None:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None


def neutral_config(config: Configuration, names: Names) -> Configuration:
    """`config` with every key by the key rule; a configuration already neutral comes back as it is."""
    from .importer import scan_profiles  # imported here: the importer reaches this module in turn

    classes = {key: profile.machine_class for key, profile in scan_profiles(config.paths.hardware_dir).profiles.items()}
    keys = machine_keys({key: machine.profile for key, machine in config.machines.items()}, classes, names)
    if all(old == new for old, new in keys.items()):
        return config
    data = config.model_dump(mode="json")
    data["machines"] = {keys[key]: value for key, value in data["machines"].items()}
    return Configuration.from_dict(data)


__all__ = ["HEADER", "config_text", "entries", "expected", "neutral_config", "relative_paths"]
