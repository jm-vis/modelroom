"""The memory reserve in `modelroom.toml`: optional, computed where it is left out (AP9-RM RM-1, ADR 0012).

Read: every file 0.1.0 wrote carries a number and reads as `entered`; a machine without the key
reads and computes. Written: a new folder leaves the key out, a number in `[defaults]` is copied
into a **new** entry as before, a missing one is left out, and an entry that is there keeps its
reserve through every write of the guided mode (Codex r1-1, r2-1).
"""

from __future__ import annotations

import json
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from modelroom.cli import main
from modelroom.config import Configuration, DefaultsConfig, MachineConfig, load_config
from modelroom.fit import GpuPool, fit_from_parameters
from modelroom.guided_write import Said, update_config, with_context, with_found_profiles, with_writer
from modelroom.reserve import ram_reserve
from modelroom.search_apply import write_configuration
from modelroom.state import atomic_write_json

from fixture_neutral import THIS_MACHINE
from fixture_support import qwen35_example_config_dict, v2_profile

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures"
LAPTOP_RAM = 127.70


def _config_file(tmp_path: Path, machines: dict | None = None, defaults: dict | None = None) -> Path:
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "docs" / "models.md"))
    data["schema_version"] = 3
    if machines is not None:
        data["machines"] = machines
    if defaults is not None:
        data["defaults"] = defaults
    path = tmp_path / "modelroom.toml"
    write_configuration(path, Configuration.from_dict(data), now=NOW, expected_text=None)
    return path


def _stored_machines(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))["machines"]


def _place_profile(tmp_path: Path) -> str:
    profile = v2_profile(display_name="server")
    atomic_write_json(tmp_path / "state" / "hardware" / f"{profile.profile_id}.json", profile.model_dump(mode="json"))
    return profile.profile_id


# --- reading ------------------------------------------------------------------------------------


def test_a_machine_without_a_memory_reserve_reads():
    machine = MachineConfig.model_validate({"reserve_vram_gib": 1.0, "writer": True})
    assert machine.reserve_ram_gib is None
    assert DefaultsConfig().reserve_ram_gib is None
    assert ram_reserve(machine, LAPTOP_RAM, unified=False) == (16.0, "computed")


def test_defaults_keep_ge_0():
    with pytest.raises(ValidationError):
        MachineConfig(reserve_ram_gib=-1.0, reserve_vram_gib=1.0, writer=True)
    with pytest.raises(ValidationError):
        DefaultsConfig(reserve_ram_gib=-1.0)


@pytest.mark.parametrize("schema", ["config_v1", "config_v2", "config_v3"])
def test_a_0_1_0_configuration_reads_with_its_number_entered(tmp_path, schema):
    """Every file 0.1.0 wrote carries the number: it reads as it did, origin `entered`."""
    shutil.copytree(FIXTURES / schema, tmp_path / schema)
    config = load_config(tmp_path / schema / "config.toml")
    laptop = config.machines["laptop" if schema == "config_v1" else "laptop-c0ffee00"]  # FR-N: neutral fixture keys
    assert laptop.reserve_ram_gib == 8.0
    assert ram_reserve(laptop, LAPTOP_RAM, unified=False) == (8.0, "entered")


def test_a_configuration_without_memory_reserve_reads_computed(tmp_path):
    path = _config_file(tmp_path, machines={"workstation": {"reserve_vram_gib": 1.0, "writer": True}})
    machine = load_config(path).machines["workstation"]
    assert ram_reserve(machine, LAPTOP_RAM, unified=False) == (16.0, "computed")


def test_a_written_configuration_without_the_key_round_trips(tmp_path):
    path = _config_file(tmp_path, machines={"workstation": {"reserve_vram_gib": 1.0, "writer": True}}, defaults={"reserve_vram_gib": 1.0})
    text = path.read_text(encoding="utf-8")
    assert "reserve_ram_gib" not in text
    config = load_config(path)
    changed = config.model_copy(update={"packagers": ["unsloth", "bartowski"]})
    write_configuration(path, changed, now=NOW, expected_text=text)
    assert "reserve_ram_gib" not in path.read_text(encoding="utf-8")
    assert load_config(path).machines == config.machines


# --- writing: a new folder, the defaults, a writer change -----------------------------------------


def test_a_new_folder_writes_no_memory_reserve(tmp_path):
    from modelroom.guided import _new_configuration

    run = SimpleNamespace(probes=SimpleNamespace(hostname=lambda: "laptop"), now=NOW, pointer_path=tmp_path / "pointer.json", out=print)
    config_file = tmp_path / "results" / "modelroom.toml"
    config_file.parent.mkdir()
    config = _new_configuration(run, config_file)
    assert config.machines[THIS_MACHINE].reserve_ram_gib is None  # FR-N: its neutral key, not the host name
    assert _stored_machines(config_file)[THIS_MACHINE] == {"reserve_vram_gib": 1.0, "writer": True}
    assert "reserve_ram_gib" not in config_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(("defaults", "expected"), [({"reserve_ram_gib": 1.5, "reserve_vram_gib": 0.5}, 1.5), ({"reserve_vram_gib": 0.5}, None)])
def test_a_found_profile_takes_a_numeric_default_and_leaves_a_missing_one_out(tmp_path, defaults, expected):
    """`test_a_numeric_default_is_still_copied` and `test_a_missing_default_writes_no_memory_reserve`, guided."""
    path = _config_file(tmp_path, defaults=defaults)
    profile_id = _place_profile(tmp_path)
    update_config(path, with_found_profiles(Said()), now=NOW)
    entry = _stored_machines(path)["computer-3f9a0c21"]  # FR-N: the neutral key
    assert entry["profile"] == profile_id and entry["reserve_vram_gib"] == 0.5
    assert entry.get("reserve_ram_gib") == expected


@pytest.mark.parametrize(("defaults_line", "expected"), [("reserve_ram_gib = 1.5\n", 1.5), ("", None)])
def test_an_imported_machine_takes_a_numeric_default_and_leaves_a_missing_one_out(tmp_path, capsys, defaults_line, expected):
    """`test_a_numeric_default_is_still_copied` and `test_a_missing_default_writes_no_memory_reserve`, import."""
    root = tmp_path / "target"
    root.mkdir()
    text = (FIXTURES / "config_v3" / "config.toml").read_text(encoding="utf-8")
    config_path = root / "modelroom.toml"
    config_path.write_text(text.replace("reserve_ram_gib = 6.0\n", defaults_line), encoding="utf-8")
    export = tmp_path / "exchange.json"
    export.write_text((FIXTURES / "export_v1.json").read_text(encoding="utf-8"), encoding="utf-8")

    code = main(["import-profile", str(export), "--config", str(config_path)], now=NOW)

    out = capsys.readouterr().out
    assert code == 0
    entry = _stored_machines(config_path)["workstation-3f9a0c21"]
    assert entry.get("reserve_ram_gib") == expected
    assert _stored_machines(config_path)["laptop-c0ffee00"]["reserve_ram_gib"] == 8.0
    said = "the reserves from [defaults]" if expected is not None else "the memory reserve is computed from its memory"
    assert f'machine "workstation-3f9a0c21" added with {said} and writer = false' in out


def test_an_entered_reserve_survives_a_guided_write(tmp_path):
    path = _config_file(tmp_path, machines={"workstation": {"reserve_ram_gib": 1.5, "reserve_vram_gib": 1.0, "writer": True}})
    update_config(path, with_context(32768), now=NOW)
    assert _stored_machines(path)["workstation"]["reserve_ram_gib"] == 1.5


def test_a_writer_change_keeps_a_computed_reserve(tmp_path):
    """A machine without the key stays without it, whatever `[defaults]` says: pool 111.7, not 119.7."""
    path = _config_file(
        tmp_path,
        machines={"laptop": {"reserve_vram_gib": 1.0, "writer": False}},
        defaults={"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0},
    )
    written = update_config(path, with_writer("laptop"), now=NOW)
    assert _stored_machines(path)["laptop"] == {"reserve_vram_gib": 1.0, "writer": True}
    fit = fit_from_parameters(10.0, GpuPool(()), LAPTOP_RAM, written.machines["laptop"], 1024)
    assert fit.pool_gib == pytest.approx(111.70)


def test_a_writer_change_keeps_an_entered_reserve(tmp_path):
    path = _config_file(
        tmp_path,
        machines={"laptop": {"reserve_ram_gib": 1.5, "reserve_vram_gib": 1.0, "writer": False}},
        defaults={"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0},
    )
    update_config(path, with_writer("laptop"), now=NOW)
    assert _stored_machines(path)["laptop"] == {"reserve_ram_gib": 1.5, "reserve_vram_gib": 1.0, "writer": True}


def test_a_new_writer_entry_takes_the_numeric_default(tmp_path):
    path = _config_file(tmp_path, defaults={"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0})
    update_config(path, with_writer("laptop"), now=NOW)
    assert _stored_machines(path)["laptop"] == {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}


def test_the_export_fixture_names_the_imported_machine_workstation():
    """The import tests above rest on this name, a neutral one since FR-N."""
    assert json.loads((FIXTURES / "export_v1.json").read_text(encoding="utf-8"))["profile"]["display_name"] == "workstation-3f9a0c21"
