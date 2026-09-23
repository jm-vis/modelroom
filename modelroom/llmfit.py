"""The `llmfit` subprocess binding: the version gate and `llmfit system --json`.

`Runner` is the shape every caller depends on: a plain callable `(args) -> CompletedProcess`,
passed in by the caller -- the same dependency-injection shape `modelroom/http.py::Transport`
uses for HTTP. There is no mocking library anywhere in this project: a test that needs a
particular `llmfit` output constructs a `FixtureRunner` and passes it in like any other value.
`SubprocessRunner` is the only implementation that spawns a real process, and it is used
nowhere in the test suite.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from typing import Callable, Protocol

DEFAULT_TIMEOUT_SECONDS = 10.0
# F9b (fix-round 5): `Which = (name) -> absolute path | None`, the same dependency-injection
# shape `Transport`/`Runner` already use -- the real implementation is `_default_which` (an
# explicit PATH walk, see its own docstring). Only `SubprocessRunner` takes one (a test never
# constructs it -- see its own docstring -- so this never touches a `FixtureRunner`-based test):
# a test that needs to prove which resolved path `SubprocessRunner` invoked injects a fake
# resolver instead of needing a real binary on `PATH`.
Which = Callable[[str], "str | None"]

# The documented Windows default when `PATHEXT` is unset.
_DEFAULT_PATHEXT = ".EXE;.CMD;.BAT;.COM"


def _default_which(name: str) -> str | None:
    """R7-2 (fix-round 6): resolve `name` on `PATH` with an explicit walk, never via `shutil.which`.

    On Python 3.11 on Windows, `shutil.which` prepends the current working directory to the
    search list regardless of the `path` argument (`_win_path_needs_curdir` was only added in
    3.12, gated on `NeedCurrentDirectoryForExePath`) -- silently reopening exactly the hole F9b
    was written to close: a same-named file placed in the current working directory could run
    instead of the real one `PATH` points to. This walks `os.environ["PATH"]` by hand instead,
    so the current directory is consulted only if it is itself listed there.

    Only absolute directory entries are searched -- a relative `PATH` entry would still resolve
    against the current working directory (the same trust problem in a different disguise), so
    it is dropped rather than followed. On Windows, each `PATHEXT` extension (the documented
    default `.EXE;.CMD;.BAT;.COM` when the variable is unset) is tried per directory, plus the
    bare name. Returns the first candidate that is a regular file and passes
    `os.access(..., os.X_OK)`, else `None`.
    """
    path_value = os.environ.get("PATH", os.defpath)
    directories = [entry for entry in path_value.split(os.pathsep) if entry and os.path.isabs(entry)]

    if sys.platform == "win32":
        pathext_value = os.environ.get("PATHEXT") or _DEFAULT_PATHEXT
        extensions = [ext for ext in pathext_value.split(os.pathsep) if ext]
        candidate_names = [name + ext for ext in extensions] + [name]
    else:
        candidate_names = [name]

    for directory in directories:
        for candidate_name in candidate_names:
            candidate = os.path.join(directory, candidate_name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None

# `llmfit --help` (checked 2026-09-22) prints no install line of its own; this names the
# project's own documented source instead (README.md, "Planned usage").
INSTALL_HINT = "install llmfit (see https://github.com/AlexsJones/llmfit)"

_VERSION_OUTPUT_RE = re.compile(r"llmfit (\d+)\.(\d+)\.(\d+)")
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class LlmfitError(Exception):
    """`llmfit` is missing, too old, or its output could not be parsed.

    The CLI maps this to exit code 2, the same code AGENTS.md already documents for "a
    required external tool is missing or too old".
    """


class Runner(Protocol):
    """`(args) -> CompletedProcess`. Every `llmfit` caller takes one of these as a parameter."""

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess: ...


class SubprocessRunner:
    """The real runner: `subprocess.run` with a fixed timeout, capturing text output.

    F9a (fix-round 5): decodes with a fixed `encoding="utf-8", errors="replace"` rather than
    `text=True` alone, which decodes with `locale.getpreferredencoding()` -- on Windows that is
    the system codepage, not UTF-8, so a non-ASCII `gpu_name`/similar in `llmfit`'s real UTF-8
    output could raise `UnicodeDecodeError` here, uncaught by anything in this module, instead
    of the process's own output ever reaching `check_llmfit_version`/`fetch_llmfit_system` to be
    turned into an ordinary `LlmfitError`.

    F9b (fix-round 5): resolves `args[0]` (always the bare name `"llmfit"`, from
    `check_llmfit_version`/`fetch_llmfit_system`) to its absolute location via `which`
    (`_default_which` by default) *before* calling `subprocess.run`, rather than passing the bare
    name straight through -- an unqualified executable name's search order on Windows checks the
    current working directory before `PATH`, so a same-named file placed in the CWD could
    otherwise run instead of the real one `PATH` points to. `which` returning `None` raises
    `FileNotFoundError`, exactly the exception `subprocess.run` itself would raise for a name it
    cannot find at all -- `check_llmfit_version` already turns that into `LlmfitError` naming an
    install hint, `fetch_llmfit_system` into one via its broader `OSError` catch
    (`FileNotFoundError` is an `OSError` subclass).

    R7-2 (fix-round 6): the default was `shutil.which` through fix-round 5; on Python 3.11 on
    Windows it prepends the current directory to the search list regardless of any `path`
    argument, reopening the same hole. `_default_which` (module-level) is an explicit PATH walk
    that never does that.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS, which: Which = _default_which) -> None:
        self._timeout = timeout
        self._which = which

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        resolved = self._which(args[0])
        if resolved is None:
            raise FileNotFoundError(f"{args[0]}: not found on PATH")
        return subprocess.run(
            [resolved, *args[1:]], capture_output=True, encoding="utf-8", errors="replace", timeout=self._timeout
        )


class FixtureRunner:
    """Test runner: maps an exact `args` tuple to a recorded `CompletedProcess`, records calls.

    Never spawns a process. A call for an `args` tuple with no recorded response is a
    test-authoring error, surfaced immediately as a `KeyError`. An `args` tuple may instead be
    mapped to an `Exception` *instance* (e.g. `subprocess.TimeoutExpired(...)`, `OSError(...)`)
    -- the runner raises it instead of returning it, so a test can simulate a subprocess that
    times out or a binary that cannot be spawned, the same way a real `Runner` would raise.
    """

    def __init__(self, responses: dict[tuple[str, ...], subprocess.CompletedProcess | Exception]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        key = tuple(args)
        self.calls.append(key)
        response = self._responses[key]
        if isinstance(response, Exception):
            raise response
        return response


def _parse_semver(text: str, label: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.fullmatch(text.strip())
    if not match:
        raise LlmfitError(f"cannot parse {label} as a '<major>.<minor>.<patch>' version: {text!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_llmfit_version(runner: Runner, min_version: str) -> str:
    """Raise `LlmfitError` unless `llmfit` is installed, responds, and is at least `min_version`.

    Returns the installed version string (e.g. `"1.1.16"`) on success. A missing binary
    (`FileNotFoundError`), any other failure to run it at all (`subprocess.TimeoutExpired`, a
    more general `OSError`), a non-zero exit code, or a version below `min_version` all raise
    `LlmfitError` naming the minimum version and an install hint; the CLI maps every case to
    exit code 2, indistinguishable from any other "required external tool" failure.
    """
    try:
        result = runner(["llmfit", "--version"])
    except FileNotFoundError as exc:
        raise LlmfitError(f"llmfit is not installed (requires >= {min_version}); {INSTALL_HINT}") from exc
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise LlmfitError(f"llmfit --version did not respond (requires >= {min_version}); {INSTALL_HINT}: {exc}") from exc

    if result.returncode != 0:
        raise LlmfitError(f"llmfit --version failed (exit {result.returncode}): {result.stderr.strip()}")

    match = _VERSION_OUTPUT_RE.search(result.stdout)
    if not match:
        raise LlmfitError(f"cannot parse llmfit --version output: {result.stdout!r}")
    installed = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    installed_str = ".".join(str(part) for part in installed)

    required = _parse_semver(min_version, "min_version")
    if installed < required:
        raise LlmfitError(f"llmfit {installed_str} is older than the required {min_version}; {INSTALL_HINT}")
    return installed_str


def fetch_llmfit_system(runner: Runner) -> dict:
    """Run `llmfit system --json` and return the parsed JSON object.

    A failure to run the subprocess at all (`subprocess.TimeoutExpired`, a more general
    `OSError`), a non-zero exit code, or a body that fails to parse as JSON all raise
    `LlmfitError`; the CLI maps every case to exit code 2, like every other `llmfit` failure.
    """
    try:
        result = runner(["llmfit", "system", "--json"])
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise LlmfitError(f"llmfit system --json did not respond: {exc}") from exc
    if result.returncode != 0:
        raise LlmfitError(f"llmfit system --json failed (exit {result.returncode}): {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LlmfitError(f"llmfit system --json returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LlmfitError("llmfit system --json returned a non-object JSON value")
    return data


def hardware_fields_from_llmfit_system(data: dict) -> dict:
    """Map `llmfit system --json`'s `system` block to `HardwareSnapshot`'s hardware fields.

    `system.gpu_vram_gb`/`total_ram_gb`/`available_ram_gb` are already GiB in llmfit's own
    output (division by 1024**3 in its source, e.g. 127.46 for a 128 GB machine), taken
    unchanged (CONTRACTS.md, "Local llmfit binding"). `providers` and `gpus[]` are ignored in
    v1. `total_ram_gb` is the one field this cannot proceed without, and (F12/R7) it must be a
    real, finite positive number, not just present -- `null`, a string, zero, negative, `NaN` or
    infinite all raise `LlmfitError` naming the field, the same as `HardwareSnapshot.ram_gib`'s
    own `gt=0` would eventually reject, but caught here so it becomes an ordinary exit `2`, never
    a pydantic crash (`gt=0`/`ge=0` alone would not catch `NaN`: every ordinary comparison
    against `NaN` is `False`, so a naive "value <= 0 is bad" check never fires for it -- R7 checks
    `math.isfinite` explicitly). A GPU-less machine legitimately omits
    `gpu_vram_gb`/`gpu_name`/`backend`, so `gpu_vram_gb` defaults to `0.0` rather than raising;
    when present, it and `available_ram_gb` must still be finite non-negative numbers. R7:
    `gpu_name`/`backend` must be a string or `None` when present, and `unified_memory` must be a
    real `bool` when present (never silently coerced, e.g. `bool("no")` is `True`) -- every field
    this function passes on to `HardwareSnapshot` is validated here, so a shape problem always
    surfaces as this function's own `LlmfitError`, never an uncaught pydantic `ValidationError`
    from `write_hardware_snapshot` further down the line.
    """
    system = data.get("system")
    if not isinstance(system, dict) or "total_ram_gb" not in system:
        raise LlmfitError("llmfit system --json response has no 'system.total_ram_gb'")
    return {
        "vram_gib": _non_negative_number(system.get("gpu_vram_gb"), "system.gpu_vram_gb", default=0.0),
        "ram_gib": _positive_number(system.get("total_ram_gb"), "system.total_ram_gb"),
        "free_ram_gib_at_measurement": _non_negative_number(
            system.get("available_ram_gb"), "system.available_ram_gb", default=None
        ),
        "gpu_name": _optional_str(system.get("gpu_name"), "system.gpu_name"),
        "backend": _optional_str(system.get("backend"), "system.backend"),
        "unified_memory": _optional_bool(system.get("unified_memory"), "system.unified_memory"),
    }


def _is_finite_number(value: object) -> bool:
    """`True` only for a real, finite `int`/`float` (never `bool`).

    Fix-round 3: `math.isfinite` itself raises `OverflowError` for an `int` too large to convert
    to `float` (e.g. a JSON integer like `10**400`, which `json.loads` parses without complaint)
    -- caught here and treated as "not finite", the same verdict as `NaN`/infinity.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive_number(value: object, field: str) -> float:
    if not _is_finite_number(value) or value <= 0:
        raise LlmfitError(f"llmfit system --json response field {field!r} is not a positive number: {value!r}")
    return value


def _non_negative_number(value: object, field: str, default: float | None) -> float | None:
    if value is None:
        return default
    if not _is_finite_number(value) or value < 0:
        raise LlmfitError(f"llmfit system --json response field {field!r} is not a non-negative number: {value!r}")
    return value


def _optional_str(value: object, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise LlmfitError(f"llmfit system --json response field {field!r} is not a string or null: {value!r}")
    return value


def _optional_bool(value: object, field: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise LlmfitError(f"llmfit system --json response field {field!r} is not a boolean: {value!r}")
    return value
