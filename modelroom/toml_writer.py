"""A small TOML writer for the data shapes modelroom writes (its configuration).

The standard library reads TOML (`tomllib`) but does not write it, and a dependency for this
one purpose is not worth it. Supported: strings, booleans, integers, finite floats, paths
(written as strings), lists of those, tables (`dict`) and arrays of tables (`list[dict]`).
`None` means "not set" and is left out, since TOML has no null. Anything else raises
`TypeError`; the caller re-reads the result with `tomllib` before it is written anywhere.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import PurePath

_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(name: str) -> str:
    return name if _BARE_KEY_RE.fullmatch(name) else json.dumps(name, ensure_ascii=False)


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"TOML cannot hold a non-finite float: {value!r}")
        return repr(value)
    if isinstance(value, PurePath):
        return json.dumps(value.as_posix(), ensure_ascii=False)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list) and not any(isinstance(item, dict) for item in value):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    raise TypeError(f"not a TOML value: {value!r}")


def _is_table_array(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


def _emit_table(data: dict, prefix: list[str], lines: list[str]) -> None:
    plain = {k: v for k, v in data.items() if v is not None and not isinstance(v, dict) and not _is_table_array(v)}
    for name, value in plain.items():
        lines.append(f"{_key(name)} = {_scalar(value)}")
    for name, value in data.items():
        if isinstance(value, dict):
            path = [*prefix, _key(name)]
            # A table that holds only tables needs no header of its own: `[a.b]` defines `a`.
            if not value or any(v is not None and not isinstance(v, dict) for v in value.values()):
                lines.extend(["", f"[{'.'.join(path)}]"])
            _emit_table(value, path, lines)
    for name, value in data.items():
        if _is_table_array(value):
            path = [*prefix, _key(name)]
            for item in value:
                lines.extend(["", f"[[{'.'.join(path)}]]"])
                _emit_table(item, path, lines)


def dump_toml(data: dict, header: str = "") -> str:
    """`data` as TOML text; `header` lines are written first as `#` comments."""
    lines = [f"# {line}" if line else "#" for line in header.splitlines()]
    if lines:
        lines.append("")
    _emit_table(data, [], lines)
    return "\n".join(lines).strip("\n") + "\n"
