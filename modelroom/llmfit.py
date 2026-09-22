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
import re
import subprocess
from typing import Protocol

DEFAULT_TIMEOUT_SECONDS = 10.0

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
    """The real runner: `subprocess.run` with a fixed timeout, capturing text output."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=self._timeout)


class FixtureRunner:
    """Test runner: maps an exact `args` tuple to a recorded `CompletedProcess`, records calls.

    Never spawns a process. A call for an `args` tuple with no recorded response is a
    test-authoring error, surfaced immediately as a `KeyError`.
    """

    def __init__(self, responses: dict[tuple[str, ...], subprocess.CompletedProcess]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        key = tuple(args)
        self.calls.append(key)
        return self._responses[key]


def _parse_semver(text: str, label: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.fullmatch(text.strip())
    if not match:
        raise LlmfitError(f"cannot parse {label} as a '<major>.<minor>.<patch>' version: {text!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_llmfit_version(runner: Runner, min_version: str) -> str:
    """Raise `LlmfitError` unless `llmfit` is installed and at least `min_version`.

    Returns the installed version string (e.g. `"1.1.16"`) on success. A missing binary
    (`FileNotFoundError` from the runner) and a version below `min_version` both raise
    `LlmfitError` naming the minimum version and an install hint; the CLI maps both to exit
    code 2, indistinguishable from any other "required external tool" failure.
    """
    try:
        result = runner(["llmfit", "--version"])
    except FileNotFoundError as exc:
        raise LlmfitError(f"llmfit is not installed (requires >= {min_version}); {INSTALL_HINT}") from exc

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

    A non-zero exit code or a body that fails to parse as JSON both raise `LlmfitError`; the
    CLI maps this to exit code 2, like every other `llmfit` failure.
    """
    result = runner(["llmfit", "system", "--json"])
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
    v1. `total_ram_gb` is the one field this cannot proceed without; a GPU-less machine
    legitimately omits `gpu_vram_gb`/`gpu_name`/`backend`, so those default rather than raise.
    """
    system = data.get("system")
    if not isinstance(system, dict) or "total_ram_gb" not in system:
        raise LlmfitError("llmfit system --json response has no 'system.total_ram_gb'")
    return {
        "vram_gib": system.get("gpu_vram_gb") or 0.0,
        "ram_gib": system["total_ram_gb"],
        "free_ram_gib_at_measurement": system.get("available_ram_gb"),
        "gpu_name": system.get("gpu_name"),
        "backend": system.get("backend"),
        "unified_memory": bool(system.get("unified_memory", False)),
    }
