"""The sink for a whole configuration (FR-N): neutral machine keys, comments and paths.

A configuration names machines by their keys, may name them again in its comments, and may name its
folder by absolute paths (`paths.state`, `paths.markdown`, `[guided].results`); a folder's path can
carry a user's or a machine's name. Every configuration modelroom writes goes through here first:

- `config_text` rewrites a stored text: each renamed key replaced wherever it stands as the key of a
  machine row -- a table head, a dotted key, an inline table; bare, quoted or escaped -- and wherever
  it stands in a comment (the same replacement, the comment otherwise kept), and each path replaced
  by its neutral form, so every other byte stays. Whether a place is a key of a row or a comment is
  decided by TOML itself: the place is replaced by a probe word and the file read again. Only where
  the text cannot be brought to exactly the expected table that way is the table written anew by
  the TOML writer (the comments go; the file then says so in its head);
- `relative_paths` gives the table form of a configuration with those paths neutral, for the
  writers that dump a whole table (`search_apply.write_configuration`, `import-profile`, `migrate`);
- `neutral_config` renames the keys of a `Configuration` by the key rule (`neutral.machine_keys`),
  for the guided mode's writes, applied to the configuration it read last, right before it writes.

A path inside the folder is written relative to it (`.`, `state`, `docs/models.md`); one outside it,
below the user's home folder, as `~/...` (`load_config` expands it); any other stays as it is -- the
one place a path of the user's own choice can still name something (CONTRACTS.md, "Neutral names").
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
_PATH_FIELDS = (("paths", "state"), ("paths", "markdown"), ("guided", "results"))
_PROBE = "modelroom-probe-0f1e2d3c"
_QUOTED_RE = re.compile(r""""(?:[^"\\\n]|\\.)*"|'[^'\n]*'""")


def entries(raw: dict) -> dict[str, str | None]:
    """`key -> profile` of every `[machines.<key>]` of a configuration table."""
    machines = raw.get("machines") if isinstance(raw.get("machines"), dict) else {}
    return {key: _profile_of(entry) for key, entry in machines.items()}


def _profile_of(entry: object) -> str | None:
    profile = entry.get("profile") if isinstance(entry, dict) else None
    return profile if isinstance(profile, str) and PROFILE_ID_RE.fullmatch(profile) else None


def relative_paths(raw: dict, folder: Path) -> dict:
    """`raw` with every absolute path in its neutral form: relative inside `folder`, `~/...` below home."""
    data = {**raw}
    for section, key in _PATH_FIELDS:
        table = data.get(section)
        value = table.get(key) if isinstance(table, dict) else None
        neutral = _neutral_path(value, folder)
        if neutral is not None:
            data[section] = {**table, key: neutral}
    return data


def _neutral_path(value: object, folder: Path) -> str | None:
    if not isinstance(value, str) or not Path(value).is_absolute():
        return None
    path = Path(value).resolve()
    for base, prefix in ((folder.resolve(), ""), (Path.home().resolve(), "~/")):
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError:
            continue
        return prefix + relative if prefix else relative or "."
    return None  # outside the folder and the home folder: the user's own choice


def expected(raw: dict, renames: dict[str, str], folder: Path) -> dict:
    """The table the neutral text has to read as: keys renamed, paths neutral."""
    data = relative_paths(raw, folder)
    if isinstance(raw.get("machines"), dict):
        data["machines"] = {renames.get(key, key): value for key, value in raw["machines"].items()}
    return data


def config_text(text: str, raw: dict, renames: dict[str, str], folder: Path) -> str:
    """The stored text brought to `expected`, its bytes kept where they can be; else the table anew."""
    wanted = expected(raw, renames, folder)
    candidate = _replace_paths(_rename(text, raw, renames), raw, folder)
    if _loads(candidate) == wanted:
        return candidate
    return dump_toml(wanted, header=HEADER)


def _rename(text: str, raw: dict, renames: dict[str, str]) -> str:
    """Every renamed key where it is the key of a machine row or stands in a comment."""
    outside = {key: value for key, value in raw.items() if key != "machines"}
    found: dict[tuple[int, int], tuple[bool, str]] = {}  # a place two keys match (other case) takes the exact one's
    for old, new in renames.items():
        for start, end, exact in _places(text, old):
            probe = _loads(text[:start] + _PROBE + text[end:])
            if probe is None:
                continue
            row = exact and _PROBE in (probe.get("machines") or {}) and {k: v for k, v in probe.items() if k != "machines"} == outside
            if (row or probe == raw) and (exact or (start, end) not in found):  # a key of a row, or a comment
                found[(start, end)] = (exact, new)
    for (start, end), (_exact, new) in sorted(found.items(), reverse=True):
        text = text[:start] + new + text[end:]
    return text


def _places(text: str, old: str) -> list[tuple[int, int, bool]]:
    """Where `old` may stand: bare or quoted, in any case; only a place spelled exactly as `old` (read as
    TOML reads it) can be the key of a row, a place in any case can be a comment (Codex r3-1)."""
    bare = re.compile(rf"""(?<![A-Za-z0-9_"'-]){re.escape(old)}(?![A-Za-z0-9_"'-])""", re.IGNORECASE)
    places = [(match.start(), match.end(), match.group() == old) for match in bare.finditer(text)]
    for match in _QUOTED_RE.finditer(text):
        read = (_loads(f"k = {match.group()}") or {}).get("k")
        if isinstance(read, str) and read.lower() == old.lower():
            places.append((match.start(), match.end(), read == old))
    return places


def _replace_paths(text: str, raw: dict, folder: Path) -> str:
    for section, key in _PATH_FIELDS:
        table = raw.get(section)
        value = table.get(key) if isinstance(table, dict) else None
        neutral = _neutral_path(value, folder)
        if neutral is not None:
            for written in {json.dumps(value, ensure_ascii=False), f"'{value}'"}:
                text = text.replace(written, json.dumps(neutral, ensure_ascii=False))
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
