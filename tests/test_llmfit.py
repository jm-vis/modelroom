"""Tests for modelroom.llmfit: the `llmfit` subprocess binding and its version gate."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from modelroom.llmfit import (
    FixtureRunner,
    LlmfitError,
    SubprocessRunner,
    check_llmfit_version,
    fetch_llmfit_system,
    hardware_fields_from_llmfit_system,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _completed(args: list[str], returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def _version_ok_runner(stdout: str = "llmfit 1.1.16\n") -> FixtureRunner:
    return FixtureRunner({("llmfit", "--version"): _completed(["llmfit", "--version"], stdout=stdout)})


# --- version gate ---------------------------------------------------------------------------


def test_check_llmfit_version_accepts_a_version_at_the_minimum():
    runner = _version_ok_runner("llmfit 1.1.16\n")
    assert check_llmfit_version(runner, "1.1.16") == "1.1.16"


def test_check_llmfit_version_accepts_a_version_above_the_minimum():
    runner = _version_ok_runner("llmfit 1.2.0\n")
    assert check_llmfit_version(runner, "1.1.16") == "1.2.0"


def test_check_llmfit_version_rejects_a_version_below_the_minimum():
    runner = _version_ok_runner("llmfit 1.1.0\n")
    with pytest.raises(LlmfitError, match="1.1.16"):
        check_llmfit_version(runner, "1.1.16")


def test_check_llmfit_version_rejects_a_missing_binary():
    def _missing(args: list[str]) -> subprocess.CompletedProcess:
        raise FileNotFoundError("llmfit not found")

    with pytest.raises(LlmfitError, match="1.1.16"):
        check_llmfit_version(_missing, "1.1.16")


def test_check_llmfit_version_rejects_unparseable_output():
    runner = FixtureRunner({("llmfit", "--version"): _completed(["llmfit", "--version"], stdout="not a version\n")})
    with pytest.raises(LlmfitError):
        check_llmfit_version(runner, "1.1.16")


# --- F12: subprocess-level failures (timeout, OSError, nonzero exit) are LlmfitError --------


def test_check_llmfit_version_rejects_a_timeout():
    runner = FixtureRunner({("llmfit", "--version"): subprocess.TimeoutExpired(cmd=["llmfit", "--version"], timeout=10.0)})
    with pytest.raises(LlmfitError, match="1.1.16"):
        check_llmfit_version(runner, "1.1.16")


def test_check_llmfit_version_rejects_a_generic_os_error():
    runner = FixtureRunner({("llmfit", "--version"): OSError("permission denied")})
    with pytest.raises(LlmfitError, match="1.1.16"):
        check_llmfit_version(runner, "1.1.16")


def test_check_llmfit_version_rejects_a_nonzero_returncode():
    runner = FixtureRunner(
        {("llmfit", "--version"): _completed(["llmfit", "--version"], returncode=1, stdout="llmfit 1.1.16\n", stderr="boom")}
    )
    with pytest.raises(LlmfitError, match="boom"):
        check_llmfit_version(runner, "1.1.16")


def test_fetch_llmfit_system_rejects_a_timeout():
    runner = FixtureRunner(
        {("llmfit", "system", "--json"): subprocess.TimeoutExpired(cmd=["llmfit", "system", "--json"], timeout=10.0)}
    )
    with pytest.raises(LlmfitError):
        fetch_llmfit_system(runner)


def test_fixture_runner_raises_the_mapped_exception_instance():
    exc = OSError("boom")
    runner = FixtureRunner({("llmfit", "--version"): exc})
    with pytest.raises(OSError, match="boom"):
        runner(["llmfit", "--version"])
    assert runner.calls == [("llmfit", "--version")]


# --- llmfit system --json ---------------------------------------------------------------


def test_fetch_llmfit_system_parses_the_recorded_laptop_fixture():
    stdout = (FIXTURES / "llmfit_system_laptop.json").read_text(encoding="utf-8")
    runner = FixtureRunner({("llmfit", "system", "--json"): _completed(["llmfit", "system", "--json"], stdout=stdout)})

    data = fetch_llmfit_system(runner)

    assert data["system"]["gpu_vram_gb"] == 11.94


def test_fetch_llmfit_system_nonzero_exit_is_llmfit_error():
    runner = FixtureRunner(
        {("llmfit", "system", "--json"): _completed(["llmfit", "system", "--json"], returncode=1, stderr="boom")}
    )
    with pytest.raises(LlmfitError, match="boom"):
        fetch_llmfit_system(runner)


def test_fetch_llmfit_system_invalid_json_is_llmfit_error():
    runner = FixtureRunner({("llmfit", "system", "--json"): _completed(["llmfit", "system", "--json"], stdout="not json")})
    with pytest.raises(LlmfitError):
        fetch_llmfit_system(runner)


# --- mapping to the HardwareSnapshot fields -----------------------------------------------


def test_hardware_fields_from_llmfit_system_maps_the_recorded_laptop_fixture():
    import json

    data = json.loads((FIXTURES / "llmfit_system_laptop.json").read_text(encoding="utf-8"))

    fields = hardware_fields_from_llmfit_system(data)

    # llmfit's own `_gb` fields are already GiB (division by 1024**3 in its source), taken
    # unchanged -- 127.46 for the measured 128 GB laptop (CONTRACTS.md, "Local llmfit binding").
    assert fields == {
        "vram_gib": 11.94,
        "ram_gib": 127.46,
        "free_ram_gib_at_measurement": 76.64,
        "gpu_name": "NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU",
        "backend": "CUDA",
        "unified_memory": False,
    }


def test_hardware_fields_from_llmfit_system_requires_total_ram_gb():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {}})


def test_hardware_fields_from_llmfit_system_defaults_vram_to_zero_when_absent():
    fields = hardware_fields_from_llmfit_system(
        {"system": {"total_ram_gb": 7.56, "available_ram_gb": 2.0, "unified_memory": False}}
    )
    assert fields["vram_gib"] == 0.0
    assert fields["gpu_name"] is None
    assert fields["backend"] is None


# --- F12: hardware_fields_from_llmfit_system validates shapes, never crashes ----------------


def test_hardware_fields_from_llmfit_system_rejects_a_null_total_ram_gb():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": None}})


def test_hardware_fields_from_llmfit_system_rejects_a_zero_total_ram_gb():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 0}})


def test_hardware_fields_from_llmfit_system_rejects_a_string_total_ram_gb():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": "127.46"}})


def test_hardware_fields_from_llmfit_system_rejects_a_negative_gpu_vram_gb():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 7.56, "gpu_vram_gb": -1.0}})


def test_hardware_fields_from_llmfit_system_rejects_a_negative_available_ram_gb():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 7.56, "available_ram_gb": -1.0}})


# --- R7: every field this function passes on is validated, never trusted by shape alone -----


def test_hardware_fields_from_llmfit_system_rejects_a_nan_total_ram_gb():
    # nan is neither > 0 nor <= 0, so a naive "value <= 0 is bad" check lets it slip through --
    # math.isfinite must reject it explicitly.
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": float("nan")}})


def test_hardware_fields_from_llmfit_system_rejects_an_infinite_gpu_vram_gb():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 7.56, "gpu_vram_gb": float("inf")}})


def test_hardware_fields_from_llmfit_system_rejects_a_non_string_gpu_name():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 7.56, "gpu_name": []}})


def test_hardware_fields_from_llmfit_system_rejects_a_huge_json_integer_total_ram_gb():
    # Fix-round 3: math.isfinite(10**400) itself raises OverflowError (an int too large to
    # convert to float) -- _is_finite_number must catch it and treat the value as not finite,
    # not let the OverflowError escape hardware_fields_from_llmfit_system as an uncaught crash.
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 10**400}})


def test_hardware_fields_from_llmfit_system_rejects_a_non_string_backend():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 7.56, "backend": 42}})


def test_hardware_fields_from_llmfit_system_rejects_a_non_bool_unified_memory():
    with pytest.raises(LlmfitError):
        hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 7.56, "unified_memory": "yes"}})


def test_hardware_fields_from_llmfit_system_accepts_a_true_unified_memory():
    fields = hardware_fields_from_llmfit_system({"system": {"total_ram_gb": 7.56, "unified_memory": True}})
    assert fields["unified_memory"] is True


# --- F9 (fix-round 5): SubprocessRunner -- the only implementation that spawns a real process --
#
# `check_llmfit_version`/`fetch_llmfit_system` are unaffected by F9b: they still build
# `["llmfit", ...]`, exactly as every `FixtureRunner`-based test above already exercises: the
# name is resolved inside `SubprocessRunner` itself, the one runner a test never constructs to
# fake a fixed response (per this module's own docstring) -- so it is tested directly here,
# against a real process, never a `FixtureRunner`.


def _write_llmfit_shim(directory: Path, version_line: str) -> None:
    """A minimal real `llmfit` stand-in on disk: `--version` prints `version_line`."""
    directory.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        (directory / "llmfit.bat").write_text(f"@echo off\r\necho {version_line}\r\n", encoding="utf-8")
    else:
        script = directory / "llmfit"
        script.write_text(f"#!/bin/sh\necho '{version_line}'\n", encoding="utf-8")
        script.chmod(0o755)


def test_subprocess_runner_resolves_the_bare_name_before_invoking_it(tmp_path: Path):
    on_path = tmp_path / "on_path"
    _write_llmfit_shim(on_path, "llmfit 1.1.16")

    def which(name: str) -> str | None:
        assert name == "llmfit"
        return str(on_path / ("llmfit.bat" if sys.platform == "win32" else "llmfit"))

    runner = SubprocessRunner(which=which)
    result = runner(["llmfit", "--version"])

    assert result.returncode == 0
    assert "llmfit 1.1.16" in result.stdout


def test_subprocess_runner_raises_file_not_found_when_which_finds_nothing():
    runner = SubprocessRunner(which=lambda name: None)
    with pytest.raises(FileNotFoundError):
        runner(["llmfit", "--version"])


def test_subprocess_runner_missing_binary_becomes_an_ordinary_llmfit_error():
    """End to end through `check_llmfit_version`, proving F9b's `FileNotFoundError` still maps
    to the same `LlmfitError` `test_check_llmfit_version_rejects_a_missing_binary` already
    covers for a hand-written fake -- here via the real `SubprocessRunner` path instead.
    """
    runner = SubprocessRunner(which=lambda name: None)
    with pytest.raises(LlmfitError, match="1.1.16"):
        check_llmfit_version(runner, "1.1.16")


def test_subprocess_runner_prefers_the_path_entry_over_a_same_named_file_in_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """F9b: a same-named `llmfit` shim placed only in the current working directory must never
    be the one that runs -- `which` (real `shutil.which` here, via `SubprocessRunner`'s own
    default) only ever searches `PATH`, and `SubprocessRunner` invokes the exact resolved path,
    never the bare name, so the current directory is never consulted at all.
    """
    on_path = tmp_path / "on_path"
    _write_llmfit_shim(on_path, "llmfit 1.1.16")
    planted_cwd = tmp_path / "planted_cwd"
    _write_llmfit_shim(planted_cwd, "llmfit 9.9.9")  # the impostor -- must never be chosen

    monkeypatch.chdir(planted_cwd)
    monkeypatch.setenv("PATH", str(on_path))

    result = SubprocessRunner()(["llmfit", "--version"])

    assert "llmfit 1.1.16" in result.stdout
    assert "9.9.9" not in result.stdout


def test_subprocess_runner_decodes_output_as_utf8_with_replace_on_undecodable_bytes(tmp_path: Path):
    """F9a: a fixed `encoding="utf-8", errors="replace"` never raises `UnicodeDecodeError`, even
    for bytes that are not valid UTF-8 at all -- `text=True` alone decodes with
    `locale.getpreferredencoding()` instead, which is not UTF-8 on Windows.
    """
    directory = tmp_path / "on_path"
    directory.mkdir()
    if sys.platform == "win32":
        script = directory / "llmfit.bat"
        # a raw non-UTF-8 byte (0x80) after "llmfit 1.1.16 " -- @echo off with this byte in a
        # .bat comment line is enough to make cmd.exe's own output undecodable as UTF-8.
        script.write_bytes(b"@echo off\r\necho llmfit 1.1.16 \x80\r\n")
    else:
        script = directory / "llmfit"
        script.write_bytes(b"#!/bin/sh\nprintf 'llmfit 1.1.16 \\x80\\n'\n")
        os.chmod(script, 0o755)

    runner = SubprocessRunner(which=lambda name: str(script))
    result = runner(["llmfit", "--version"])

    assert result.returncode == 0
    assert "llmfit 1.1.16" in result.stdout
