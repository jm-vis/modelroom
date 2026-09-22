"""Tests for modelroom.llmfit: the `llmfit` subprocess binding and its version gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from modelroom.llmfit import (
    FixtureRunner,
    LlmfitError,
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
