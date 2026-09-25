"""Tests for `modelroom export-profile` and `modelroom import-profile`.

Every test builds a real deployment in a temporary folder -- a schema-3 configuration
(`fixtures/config_v3/config.toml`), profile files and measurement files -- and drives the
real CLI through `main`. The export fixture `fixtures/export_v1.json` is the exchanged file.
The two concurrency tests start real processes, like `tests/test_state.py` does.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from modelroom.binding import read_pointer
from modelroom.cli import main
from modelroom.config import load_config
from modelroom.importer import machine_name_for
from modelroom.measurements import load_export, measurement_path, read_measurements
from modelroom.profile import HardwareProfile, read_profile_document
from modelroom.state import acquire_lock, release_lock

FIXTURES = Path(__file__).parent / "fixtures"
EXPORT = json.loads((FIXTURES / "export_v1.json").read_text(encoding="utf-8"))
PROFILE_ID = EXPORT["profile"]["profile_id"]
OTHER_ID = "b1b2b3b4b5b6b7b8"
NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
# Measured on CPython 3.12: `json.loads` still parses 2000 nested arrays and raises
# RecursionError (not a ValueError) from 20000 on -- the depth these two tests need.
_NESTING_DEPTH = 20000


# --- helpers ---------------------------------------------------------------------------------


def _deployment(root: Path) -> Path:
    """A schema-3 configuration in its own folder (the results folder); returns its path."""
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "config_v3" / "config.toml", root / "modelroom.toml")
    return root / "modelroom.toml"


def _export_data(profile_updates: dict | None = None, measurements: list[dict] | None = None) -> dict:
    """A copy of the export fixture; every measurement is re-keyed to the (possibly new) profile."""
    data = copy.deepcopy(EXPORT)
    if profile_updates:
        data["profile"].update(profile_updates)
    if measurements is not None:
        data["measurements"] = copy.deepcopy(measurements)
    for measurement in data["measurements"]:
        measurement["profile_id"] = data["profile"]["profile_id"]
    return data


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _install(state: Path, data: dict) -> None:
    """Put an export's profile and measurements into a state folder as stored files."""
    profile_id = data["profile"]["profile_id"]
    _write_json(state / "hardware" / f"{profile_id}.json", data["profile"])
    for measurement in data["measurements"]:
        _write_json(state / "measurements" / profile_id / f"{measurement['measurement_id']}.json", measurement)


def _pointer_file(home: Path, results_dir: Path, profile_id: str) -> Path:
    pointer = home / ".modelroom" / "guided.json"
    return _write_json(
        pointer,
        {"schema_version": 1, "current": str(results_dir), "bindings": {str(results_dir): profile_id}},
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "modelroom.lock"
    }


def _machines(config_path: Path) -> dict:
    return tomllib.loads(config_path.read_text(encoding="utf-8"))["machines"]


def _read_profile(text: str) -> HardwareProfile:
    """A stored profile as every reader reads it: the export fixture's schema 2 as schema 3."""
    profile = read_profile_document(json.loads(text))
    assert isinstance(profile, HardwareProfile)
    return profile


# --- export ----------------------------------------------------------------------------------


def test_export_writes_the_bound_profile_with_every_measurement(tmp_path, capsys):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    pointer = _pointer_file(tmp_path / "home", config_path.parent, PROFILE_ID)
    out = tmp_path / "exchange" / "workstation.json"

    code = main(["export-profile", "--config", str(config_path), "--out", str(out)], pointer_path=pointer)

    assert code == 0
    export = load_export(json.loads(out.read_text(encoding="utf-8")))
    assert export.profile.profile_id == PROFILE_ID
    assert [m.measurement_id for m in export.measurements] == sorted(m["measurement_id"] for m in EXPORT["measurements"])
    assert "workstation" in capsys.readouterr().out


def test_export_takes_the_profile_named_with_the_profile_flag_without_any_binding(tmp_path):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    out = tmp_path / "out.json"

    code = main(
        ["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(out)],
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
    )

    assert code == 0
    assert load_export(json.loads(out.read_text(encoding="utf-8"))).profile.profile_id == PROFILE_ID


def test_the_profile_flag_wins_over_the_home_binding(tmp_path):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    _install(tmp_path / "source" / "state", _export_data({"profile_id": OTHER_ID, "display_name": "server"}, []))
    pointer = _pointer_file(tmp_path / "home", config_path.parent, PROFILE_ID)
    out = tmp_path / "out.json"

    code = main(
        ["export-profile", "--config", str(config_path), "--profile", OTHER_ID, "--out", str(out)],
        pointer_path=pointer,
    )

    assert code == 0
    export = load_export(json.loads(out.read_text(encoding="utf-8")))
    assert (export.profile.profile_id, export.measurements) == (OTHER_ID, [])


def test_export_without_a_binding_and_without_the_flag_exits_2(tmp_path, capsys):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    out = tmp_path / "out.json"

    code = main(
        ["export-profile", "--config", str(config_path), "--out", str(out)],
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
    )

    assert code == 2
    assert "--profile" in capsys.readouterr().err
    assert not out.exists()


def test_export_exits_2_when_the_named_profile_is_not_in_the_results_folder(tmp_path, capsys):
    config_path = _deployment(tmp_path / "source")
    out = tmp_path / "out.json"

    code = main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(out)])

    assert code == 2
    assert PROFILE_ID in capsys.readouterr().err
    assert not out.exists()


def test_export_of_a_schema_1_profile_file_exits_2_and_names_migrate(tmp_path, capsys):
    config_path = _deployment(tmp_path / "source")
    legacy = json.loads((FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json").read_text(encoding="utf-8"))
    _write_json(tmp_path / "source" / "state" / "hardware" / f"{PROFILE_ID}.json", legacy)
    out = tmp_path / "out.json"

    code = main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(out)])

    assert code == 2
    assert "migrate" in capsys.readouterr().err
    assert not out.exists()


def test_export_lists_an_unreadable_measurement_file_and_still_writes_the_others(tmp_path, capsys):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    broken = tmp_path / "source" / "state" / "measurements" / PROFILE_ID / "20260101T000000Z-aaaaaaaa.json"
    broken.write_text("{not json", encoding="utf-8")
    out = tmp_path / "out.json"

    code = main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(out)])

    assert code == 0
    assert len(load_export(json.loads(out.read_text(encoding="utf-8"))).measurements) == 2
    assert broken.name in capsys.readouterr().out


def test_export_exits_3_for_an_unreadable_profile_file(tmp_path, capsys):
    config_path = _deployment(tmp_path / "source")
    path = tmp_path / "source" / "state" / "hardware" / f"{PROFILE_ID}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    out = tmp_path / "out.json"

    code = main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(out)])

    assert code == 3
    assert path.name in capsys.readouterr().err
    assert not out.exists()


def test_export_exits_3_for_a_profile_of_an_unsupported_schema(tmp_path):
    config_path = _deployment(tmp_path / "source")
    _write_json(
        tmp_path / "source" / "state" / "hardware" / f"{PROFILE_ID}.json",
        {**EXPORT["profile"], "schema_version": 4},
    )
    out = tmp_path / "out.json"

    code = main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(out)])

    assert code == 3
    assert not out.exists()


def test_export_replaces_an_existing_file_and_leaves_no_tmp_file_behind(tmp_path):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    out = tmp_path / "exchange" / "out.json"
    out.parent.mkdir()
    out.write_text("older content", encoding="utf-8")

    code = main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(out)])

    assert code == 0
    assert load_export(json.loads(out.read_text(encoding="utf-8"))).profile.profile_id == PROFILE_ID
    assert [p.name for p in out.parent.iterdir()] == [out.name]


@pytest.mark.parametrize("out_name", ["state/hardware/beef.json", "state", "state/modelroom.json"])
def test_export_refuses_to_write_into_the_state_folder(tmp_path, out_name, capsys):
    """`--out` is a user argument; it must not be able to overwrite a state file."""
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    before = _tree(tmp_path / "source")

    code = main(
        ["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(tmp_path / "source" / out_name)]
    )

    assert code == 2
    assert "paths.state" in capsys.readouterr().err
    assert _tree(tmp_path / "source") == before


def test_export_refuses_to_write_over_the_configuration_file(tmp_path, capsys):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    before = config_path.read_bytes()

    code = main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(config_path)])

    assert code == 2
    assert config_path.read_bytes() == before
    assert "configuration" in capsys.readouterr().err


def test_export_refuses_to_write_over_the_pointer_file(tmp_path, capsys):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    pointer = _pointer_file(tmp_path / "home", config_path.parent, PROFILE_ID)
    before = pointer.read_bytes()

    code = main(
        ["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(pointer)],
        pointer_path=pointer,
    )

    assert code == 2
    assert pointer.read_bytes() == before
    assert "pointer" in capsys.readouterr().err


def test_export_exits_2_when_the_out_path_cannot_be_written(tmp_path, capsys):
    """An `--out` that is an existing folder is a user mistake, not a crash."""
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    out = tmp_path / "a-folder"
    out.mkdir()

    code = main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(out)])

    assert code == 2
    assert out.name in capsys.readouterr().err
    assert list(out.iterdir()) == []
    assert not list(tmp_path.glob("*.tmp"))


def test_an_import_stops_when_a_file_blocks_the_measurements_folder(tmp_path, capsys):
    """A regular file where the measurement folder belongs: exit 1, and the profile stays put."""
    config_path = _deployment(tmp_path / "target")
    blocked = tmp_path / "target" / "state" / "measurements" / PROFILE_ID
    blocked.parent.mkdir(parents=True)
    blocked.write_text("not a folder", encoding="utf-8")
    before = _tree(tmp_path / "target")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 1
    assert _tree(tmp_path / "target") == before
    assert str(PROFILE_ID) in capsys.readouterr().err


def test_an_import_stops_when_a_file_blocks_the_hardware_folder(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    blocked = tmp_path / "target" / "state" / "hardware"
    blocked.parent.mkdir(parents=True)
    blocked.write_text("not a folder", encoding="utf-8")
    before = _tree(tmp_path / "target")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 1
    assert _tree(tmp_path / "target") == before
    assert "hardware" in capsys.readouterr().err


@pytest.mark.parametrize("profile_id", ["not-hex", "3F9A0C21D4E6B870", "3f9a0c21", "../../evil"])
def test_export_refuses_a_profile_argument_that_is_not_a_profile_id(tmp_path, profile_id, capsys):
    config_path = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    out = tmp_path / "out.json"

    code = main(["export-profile", "--config", str(config_path), "--profile", profile_id, "--out", str(out)])

    assert code == 2
    assert "profile_id" in capsys.readouterr().err
    assert not out.exists()


def test_export_exits_2_for_a_missing_configuration(tmp_path):
    assert main(["export-profile", "--config", str(tmp_path / "gone.toml"), "--out", str(tmp_path / "o.json")]) == 2


# --- import: the profile decision --------------------------------------------------------------


def test_import_writes_the_profile_its_measurements_and_a_machine_entry(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    export_file = _write_json(tmp_path / "exchange.json", EXPORT)

    code = main(["import-profile", str(export_file), "--config", str(config_path)], now=NOW)

    out = capsys.readouterr().out
    assert code == 0
    stored = _read_profile(
        (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8")
    )
    assert stored.display_name == "workstation"
    records = read_measurements(tmp_path / "target" / "state", PROFILE_ID)
    assert len(records.records) == 2 and not records.unreadable
    machine = _machines(config_path)["workstation"]
    assert machine == {"reserve_ram_gib": 6.0, "reserve_vram_gib": 0.5, "writer": False, "profile": PROFILE_ID}
    assert "workstation" in out and "new" in out
    for measurement in EXPORT["measurements"]:
        assert measurement["measurement_id"] in out


def test_the_first_rewrite_of_the_configuration_leaves_a_backup(tmp_path, capsys):
    """A hand-written configuration keeps its comments in `<config>.bak`, written once."""
    config_path = _deployment(tmp_path / "target")
    original = config_path.read_bytes()
    backup = config_path.with_name(config_path.name + ".bak")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    out = capsys.readouterr().out
    assert code == 0
    assert backup.read_bytes() == original
    assert "# A schema-3 configuration" in backup.read_text(encoding="utf-8")
    assert "[machines.workstation]" in config_path.read_text(encoding="utf-8")
    assert backup.name in out


def test_an_existing_backup_of_the_configuration_is_never_overwritten(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    backup = config_path.with_name(config_path.name + ".bak")
    backup.write_text("# an older backup, kept as it is\n", encoding="utf-8")
    before = backup.read_bytes()

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert backup.read_bytes() == before
    assert backup.name in capsys.readouterr().out
    assert "[machines.workstation]" in config_path.read_text(encoding="utf-8")


def test_no_backup_is_written_when_the_configuration_is_not_rewritten(tmp_path):
    config_path = _deployment(tmp_path / "target")
    text = config_path.read_text(encoding="utf-8").replace('profile = "c0ffee0000000001"', f'profile = "{PROFILE_ID}"')
    config_path.write_text(text, encoding="utf-8")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert not config_path.with_name(config_path.name + ".bak").exists()


def test_a_conflicting_import_writes_no_backup_either(tmp_path):
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", EXPORT)
    changed = copy.deepcopy(EXPORT["measurements"][0])
    changed["daemon_version"] = "0.12.4"
    before = _tree(tmp_path / "target")

    code = main(
        ["import-profile", str(_write_json(tmp_path / "e.json", _export_data(None, [changed]))), "--config", str(config_path)],
        now=NOW,
    )

    assert code == 1
    assert _tree(tmp_path / "target") == before
    assert not config_path.with_name(config_path.name + ".bak").exists()


def test_import_keeps_every_other_configuration_value(tmp_path):
    config_path = _deployment(tmp_path / "target")
    before = load_config(config_path)
    main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)
    after = load_config(config_path)

    assert after.machines["laptop"] == before.machines["laptop"]
    assert (after.families, after.packagers, after.publishers, after.paths, after.llmfit, after.defaults) == (
        before.families,
        before.packagers,
        before.publishers,
        before.paths,
        before.llmfit,
        before.defaults,
    )


def test_a_second_import_changes_nothing_and_says_nothing_to_do(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    export_file = _write_json(tmp_path / "exchange.json", EXPORT)
    main(["import-profile", str(export_file), "--config", str(config_path)], now=NOW)
    before = _tree(tmp_path / "target")
    capsys.readouterr()

    code = main(["import-profile", str(export_file), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert _tree(tmp_path / "target") == before
    assert "nothing to do" in capsys.readouterr().out


def test_a_newer_profile_replaces_the_stored_one(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", EXPORT)
    newer = _export_data({"recorded_at": "2026-09-24T08:00:00Z", "gpu_name": "Nova GPU 2"}, [])

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", newer)), "--config", str(config_path)], now=NOW)

    assert code == 0
    stored = _read_profile(
        (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8")
    )
    assert stored.gpu_name == "Nova GPU 2"
    assert "updated" in capsys.readouterr().out


def test_an_older_profile_never_overwrites_the_stored_one(tmp_path, capsys):
    """The stored files stay byte for byte; only the machine entry of that profile is added."""
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", EXPORT)
    before = _tree(tmp_path / "target" / "state")
    older = _export_data({"recorded_at": "2026-09-22T08:00:00Z", "gpu_name": "Nova GPU 0"}, [])

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", older)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert _tree(tmp_path / "target" / "state") == before
    assert _machines(config_path)["workstation"]["profile"] == PROFILE_ID
    assert "kept" in capsys.readouterr().out


def test_a_profile_of_the_same_age_changes_nothing_even_with_other_content(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", EXPORT)
    before = _tree(tmp_path / "target" / "state")
    same_age = _export_data({"gpu_name": "Nova GPU X"}, [])

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", same_age)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert _tree(tmp_path / "target" / "state") == before
    assert "same recorded_at" in capsys.readouterr().out


def test_the_same_hardware_id_under_another_profile_id_keeps_both_with_a_note(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", _export_data(None, []))
    clone = _export_data({"profile_id": OTHER_ID, "display_name": "clone"}, [])

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", clone)), "--config", str(config_path)], now=NOW)

    hardware = tmp_path / "target" / "state" / "hardware"
    assert code == 0
    assert sorted(p.stem for p in hardware.iterdir()) == sorted([PROFILE_ID, OTHER_ID])
    assert "same hardware id (cloned image?)" in capsys.readouterr().out


def test_the_same_display_name_keeps_both_and_shows_the_short_id(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", _export_data({"os_fingerprint": "1111111111111111"}, []))
    twin = _export_data({"profile_id": OTHER_ID, "os_fingerprint": "2222222222222222"}, [])

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", twin)), "--config", str(config_path)], now=NOW)

    out = capsys.readouterr().out
    hardware = tmp_path / "target" / "state" / "hardware"
    assert code == 0
    assert sorted(p.stem for p in hardware.iterdir()) == sorted([PROFILE_ID, OTHER_ID])
    assert f"workstation ({OTHER_ID[:8]})" in out


# --- import: measurements ----------------------------------------------------------------------


def test_a_measurement_with_the_same_id_and_other_content_is_a_conflict_that_writes_nothing(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", EXPORT)
    before = _tree(tmp_path / "target")
    changed = copy.deepcopy(EXPORT["measurements"][0])
    changed["daemon_version"] = "0.12.4"

    code = main(
        ["import-profile", str(_write_json(tmp_path / "e.json", _export_data(None, [changed]))), "--config", str(config_path)],
        now=NOW,
    )

    assert code == 1
    assert _tree(tmp_path / "target") == before
    assert changed["measurement_id"] in capsys.readouterr().err


def test_known_and_new_measurements_are_reported_one_line_each(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", _export_data(None, [EXPORT["measurements"][0]]))
    capsys.readouterr()

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("measurement ")]
    assert code == 0
    assert len(lines) == 2
    assert any(EXPORT["measurements"][0]["measurement_id"] in line and "unchanged" in line for line in lines)
    assert any(EXPORT["measurements"][1]["measurement_id"] in line and "new" in line for line in lines)
    assert len(read_measurements(tmp_path / "target" / "state", PROFILE_ID).records) == 2


def test_a_file_in_the_way_of_an_imported_measurement_is_a_conflict(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    state = tmp_path / "target" / "state"
    blocked = measurement_path(state, PROFILE_ID, EXPORT["measurements"][0]["measurement_id"])
    blocked.parent.mkdir(parents=True)
    blocked.write_text("{not json", encoding="utf-8")
    before = _tree(tmp_path / "target")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 1
    assert _tree(tmp_path / "target") == before
    assert blocked.name in capsys.readouterr().err


# --- import: broken files in the target ----------------------------------------------------------


def test_an_unreadable_profile_file_in_the_target_is_listed_and_does_not_stop_the_import(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    broken = tmp_path / "target" / "state" / "hardware" / f"{OTHER_ID}.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{not json", encoding="utf-8")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    out = capsys.readouterr().out
    assert code == 0
    assert broken.name in out
    assert (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").exists()


def test_a_profile_of_an_unsupported_schema_in_the_target_is_listed_and_does_not_stop_the_import(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    future = _write_json(
        tmp_path / "target" / "state" / "hardware" / f"{OTHER_ID}.json",
        {**EXPORT["profile"], "schema_version": 4, "profile_id": OTHER_ID},
    )

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert future.name in capsys.readouterr().out
    assert (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").exists()


def test_a_schema_1_profile_in_the_way_of_the_imported_profile_names_migrate(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    legacy = json.loads((FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json").read_text(encoding="utf-8"))
    _write_json(tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json", legacy)
    before = _tree(tmp_path / "target")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 1
    assert _tree(tmp_path / "target") == before
    assert "migrate" in capsys.readouterr().err


def test_a_file_in_the_way_of_the_imported_profile_is_a_conflict(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    blocked = tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json"
    blocked.parent.mkdir(parents=True)
    blocked.write_text("{not json", encoding="utf-8")
    before = _tree(tmp_path / "target")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 1
    assert _tree(tmp_path / "target") == before
    assert blocked.name in capsys.readouterr().err


# --- import: the local binding, the lock, unreadable input -----------------------------------------


def test_import_never_changes_the_local_binding(tmp_path, monkeypatch):
    config_path = _deployment(tmp_path / "target")
    home = tmp_path / "home"
    pointer = _pointer_file(home, config_path.parent, "c0ffee0000000001")
    before = pointer.read_bytes()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert pointer.read_bytes() == before
    assert read_pointer(pointer).binding_for(config_path.parent) == "c0ffee0000000001"


def test_import_stops_at_a_held_lock_and_writes_nothing(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    before = _tree(tmp_path / "target")
    handle = acquire_lock(load_config(config_path).paths.lock_file, "fetch", NOW)
    try:
        code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)
    finally:
        release_lock(handle)

    assert code == 1
    assert _tree(tmp_path / "target") == before
    assert "lock" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    "content",
    ["{not json", "[]", "null", '"a string"'],
    ids=["broken-json", "array-root", "null-root", "string-root"],
)
def test_an_export_file_that_is_not_an_export_object_exits_3(tmp_path, content, capsys):
    config_path = _deployment(tmp_path / "target")
    export_file = tmp_path / "e.json"
    export_file.write_text(content, encoding="utf-8")

    code = main(["import-profile", str(export_file), "--config", str(config_path)], now=NOW)

    assert code == 3
    assert export_file.name in capsys.readouterr().err
    assert not (tmp_path / "target" / "state").exists()


def test_an_export_file_that_is_not_valid_utf_8_exits_3(tmp_path):
    config_path = _deployment(tmp_path / "target")
    export_file = tmp_path / "e.json"
    export_file.write_bytes(b'{"schema_version": 1, "profile": "\xff\xfe"}')

    assert main(["import-profile", str(export_file), "--config", str(config_path)], now=NOW) == 3


def test_a_missing_export_file_exits_3(tmp_path):
    config_path = _deployment(tmp_path / "target")
    assert main(["import-profile", str(tmp_path / "gone.json"), "--config", str(config_path)], now=NOW) == 3


@pytest.mark.parametrize("version", [0, 2, 3, "1", None])
def test_an_export_of_an_unsupported_schema_version_exits_3(tmp_path, version):
    config_path = _deployment(tmp_path / "target")
    data = {**copy.deepcopy(EXPORT), "schema_version": version}
    if version is None:
        del data["schema_version"]

    assert main(["import-profile", str(_write_json(tmp_path / "e.json", data)), "--config", str(config_path)], now=NOW) == 3


def test_an_export_whose_measurement_belongs_to_another_profile_exits_3(tmp_path):
    config_path = _deployment(tmp_path / "target")
    data = copy.deepcopy(EXPORT)
    data["measurements"][0]["profile_id"] = OTHER_ID

    assert main(["import-profile", str(_write_json(tmp_path / "e.json", data)), "--config", str(config_path)], now=NOW) == 3


def test_a_schema_1_configuration_exits_2_and_names_migrate(tmp_path, capsys):
    root = tmp_path / "target"
    root.mkdir()
    shutil.copy(FIXTURES / "config_v1" / "config.toml", root / "modelroom.toml")

    code = main(
        ["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(root / "modelroom.toml")],
        now=NOW,
    )

    assert code == 2
    assert "migrate" in capsys.readouterr().err
    assert not (root / "state").exists()


def test_a_missing_configuration_exits_2(tmp_path):
    export_file = _write_json(tmp_path / "e.json", EXPORT)
    assert main(["import-profile", str(export_file), "--config", str(tmp_path / "gone.toml")], now=NOW) == 2


def test_a_configuration_that_is_not_valid_utf_8_exits_2_for_both_commands(tmp_path, capsys):
    """`load_config` lets `UnicodeDecodeError` through; the CLI must not show a traceback."""
    config_path = _deployment(tmp_path / "target")
    config_path.write_bytes(b'schema_version = 2\n# \xff\xfe\n[paths]\nstate = "state"\nmarkdown = "d/m.md"\n')
    export_file = _write_json(tmp_path / "e.json", EXPORT)

    assert main(["import-profile", str(export_file), "--config", str(config_path)], now=NOW) == 2
    assert main(["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(tmp_path / "o.json")]) == 2
    assert config_path.name in capsys.readouterr().err
    assert not (tmp_path / "target" / "state").exists()


def test_an_export_file_of_extreme_nesting_exits_3(tmp_path, capsys):
    """Deeply nested JSON makes `json.loads` raise `RecursionError`, not `ValueError`."""
    config_path = _deployment(tmp_path / "target")
    export_file = tmp_path / "e.json"
    export_file.write_text("[" * _NESTING_DEPTH + "]" * _NESTING_DEPTH, encoding="utf-8")

    code = main(["import-profile", str(export_file), "--config", str(config_path)], now=NOW)

    assert code == 3
    assert export_file.name in capsys.readouterr().err


def test_a_profile_file_of_extreme_nesting_is_listed_and_does_not_stop_the_import(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    broken = tmp_path / "target" / "state" / "hardware" / f"{OTHER_ID}.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("[" * _NESTING_DEPTH + "]" * _NESTING_DEPTH, encoding="utf-8")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert broken.name in capsys.readouterr().out
    assert (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").exists()


def test_a_configuration_value_the_toml_writer_cannot_write_exits_2(tmp_path, capsys):
    """`inf` is valid TOML and passes `ge=0`, but `dump_toml` refuses it: exit 2, nothing written."""
    config_path = _deployment(tmp_path / "target")
    text = config_path.read_text(encoding="utf-8").replace("reserve_ram_gib = 6.0", "reserve_ram_gib = inf")
    config_path.write_text(text, encoding="utf-8")
    before = _tree(tmp_path / "target")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 2
    assert _tree(tmp_path / "target") == before
    assert config_path.name in capsys.readouterr().err


# --- import: the machine name ---------------------------------------------------------------------


def test_the_machine_name_gets_the_short_id_when_the_name_is_already_taken(tmp_path):
    config_path = _deployment(tmp_path / "target")
    text = config_path.read_text(encoding="utf-8").replace("[machines.laptop]", "[machines.workstation]")
    config_path.write_text(text, encoding="utf-8")

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    machines = _machines(config_path)
    assert code == 0
    assert machines[f"workstation-{PROFILE_ID[:8]}"]["profile"] == PROFILE_ID
    assert machines["workstation"]["profile"] == "c0ffee0000000001"


def test_a_display_name_without_usable_characters_becomes_the_short_id(tmp_path):
    config_path = _deployment(tmp_path / "target")
    data = _export_data({"display_name": "*** ***"}, [])

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", data)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert _machines(config_path)[PROFILE_ID[:8]]["profile"] == PROFILE_ID


def test_a_machine_that_already_carries_this_profile_keeps_its_entry(tmp_path):
    config_path = _deployment(tmp_path / "target")
    text = config_path.read_text(encoding="utf-8").replace('profile = "c0ffee0000000001"', f'profile = "{PROFILE_ID}"')
    config_path.write_text(text, encoding="utf-8")
    before = config_path.read_bytes()

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert config_path.read_bytes() == before
    assert sorted(_machines(config_path)) == ["laptop"]


@pytest.mark.parametrize(
    "display_name, taken, expected",
    [
        ("workstation", set(), "workstation"),
        ("Workstation_01", set(), "workstation-01"),
        ("a b", {"a-b"}, "a-b-3f9a0c21"),
        ("...", set(), "3f9a0c21"),
        ("-lead-", set(), "lead"),
    ],
)
def test_machine_name_for_normalizes_and_avoids_collisions(display_name, taken, expected):
    assert machine_name_for(display_name, PROFILE_ID, taken) == expected


# --- concurrency (real processes) ------------------------------------------------------------------

_IMPORT_SCRIPT = """
import sys, time
from pathlib import Path
from modelroom.cli import main

go_path = Path(sys.argv[3])
ready_path = Path(sys.argv[4])
ready_path.write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 10
while not go_path.exists() and time.monotonic() < deadline:
    time.sleep(0.001)
raise SystemExit(main(["import-profile", sys.argv[1], "--config", sys.argv[2]]))
"""

_MEASURE_SCRIPT = """
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from modelroom.measurements import MeasurementRecord, write_measurement
from modelroom.state import acquire_lock, release_lock

state = Path(sys.argv[1])
record = MeasurementRecord.model_validate(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
release_path = Path(sys.argv[3])
handle = acquire_lock(state / "modelroom.lock", "measure", datetime.now(timezone.utc))
write_measurement(state, record, handle)
print("holding", flush=True)
deadline = time.monotonic() + 10
while not release_path.exists() and time.monotonic() < deadline:
    time.sleep(0.005)
release_lock(handle)
print("released", flush=True)
"""


def _wait_for(paths: list[Path], seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not all(path.exists() for path in paths) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert all(path.exists() for path in paths), "a child process did not signal in time"


@pytest.mark.parametrize("round_index", range(3))
def test_two_imports_at_once_serialize_without_loss_or_a_half_written_file(tmp_path, round_index):
    """Two processes import two different profiles into the same state folder at the same time.

    The lock is non-blocking, so a loser exits 1 without writing; whoever wins leaves complete,
    valid files. Repeating a loser afterwards has to bring the folder to the same result as two
    sequential runs: both profiles, all measurements, all three machine entries. Several rounds,
    because which process wins is the operating system's decision and both orders have to hold
    -- this test is the regression guard for the configuration read that the import repeats
    under the lock (the first draft read it before the lock and the second import silently
    dropped the first one's machine entry).
    """
    config_path = _deployment(tmp_path / "target")
    first = _write_json(tmp_path / "first.json", EXPORT)
    second = _write_json(tmp_path / "second.json", _export_data({"profile_id": OTHER_ID, "display_name": "server"}))
    script = tmp_path / "importer.py"
    script.write_text(_IMPORT_SCRIPT, encoding="utf-8")
    go = tmp_path / "go"
    ready = [tmp_path / "ready-0", tmp_path / "ready-1"]

    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(export), str(config_path), str(go), str(ready[index])],
            stdout=subprocess.PIPE,
            text=True,
        )
        for index, export in enumerate((first, second))
    ]
    _wait_for(ready)
    go.write_text("go", encoding="utf-8")
    for proc in procs:
        proc.communicate(timeout=30)
    codes = [proc.returncode for proc in procs]

    assert sorted(codes) in ([0, 0], [0, 1]), codes
    _assert_state_is_consistent(tmp_path / "target", config_path)
    for export, code in zip((first, second), codes):
        if code == 1:
            assert main(["import-profile", str(export), "--config", str(config_path)], now=NOW) == 0
    _assert_state_is_consistent(tmp_path / "target", config_path)
    for profile_id in (PROFILE_ID, OTHER_ID):
        assert len(read_measurements(tmp_path / "target" / "state", profile_id).records) == 2
    assert sorted(_machines(config_path)) == ["laptop", "server", "workstation"]


def test_an_import_during_a_measurement_write_stops_at_the_lock_and_loses_nothing(tmp_path):
    """A second process holds the lock and publishes a measurement of the same profile.

    The import must stop at the lock (exit 1) with nothing written, and must then import
    cleanly next to the measurement the other process published.
    """
    config_path = _deployment(tmp_path / "target")
    state = tmp_path / "target" / "state"
    extra = copy.deepcopy(EXPORT["measurements"][0])
    extra["measurement_id"] = "20260923T090000Z-11112222"
    extra["measured_at"] = "2026-09-23T09:00:00Z"
    record_file = _write_json(tmp_path / "extra.json", extra)
    script = tmp_path / "measure.py"
    script.write_text(_MEASURE_SCRIPT, encoding="utf-8")
    release = tmp_path / "release"

    holder = subprocess.Popen(
        [sys.executable, str(script), str(state), str(record_file), str(release)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "holding"
        before = _tree(tmp_path / "target")
        code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)
        assert code == 1
        assert _tree(tmp_path / "target") == before
    finally:
        release.write_text("release", encoding="utf-8")
        holder.communicate(timeout=30)

    assert main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW) == 0
    records = read_measurements(state, PROFILE_ID)
    assert not records.unreadable
    assert len(records.records) == 3


def _assert_state_is_consistent(root: Path, config_path: Path) -> None:
    """Every stored file reads and validates, and the configuration still loads."""
    for path in (root / "state" / "hardware").glob("*.json"):
        _read_profile(path.read_text(encoding="utf-8"))
    for folder in (root / "state" / "measurements").glob("*"):
        assert not read_measurements(root / "state", folder.name).unreadable
    load_config(config_path)
    assert not list(root.rglob("*.tmp"))


def test_the_state_folder_holds_no_temporary_file_after_an_import(tmp_path):
    config_path = _deployment(tmp_path / "target")
    main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)
    assert not list((tmp_path / "target").rglob("*.tmp"))


def test_the_import_of_an_older_profile_still_imports_its_new_measurements(tmp_path):
    """The profile decision and the measurement decision are independent (schema-2 contract)."""
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", _export_data(None, []))
    older = _export_data({"recorded_at": "2026-09-22T08:00:00Z"}, EXPORT["measurements"])

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", older)), "--config", str(config_path)], now=NOW)

    stored = _read_profile(
        (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8")
    )
    assert code == 0
    assert stored.recorded_at == datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc)
    assert len(read_measurements(tmp_path / "target" / "state", PROFILE_ID).records) == 2


def test_an_import_into_an_empty_results_folder_creates_the_state_folder(tmp_path):
    config_path = _deployment(tmp_path / "target")
    assert not (tmp_path / "target" / "state").exists()

    code = main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)

    assert code == 0
    assert (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").exists()


def test_export_then_import_round_trips_between_two_folders(tmp_path):
    source_config = _deployment(tmp_path / "source")
    _install(tmp_path / "source" / "state", EXPORT)
    target_config = _deployment(tmp_path / "target")
    out = tmp_path / "exchange" / "workstation.json"

    assert main(["export-profile", "--config", str(source_config), "--profile", PROFILE_ID, "--out", str(out)]) == 0
    assert main(["import-profile", str(out), "--config", str(target_config)], now=NOW) == 0

    source = _read_profile(
        (tmp_path / "source" / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8")
    )
    target = _read_profile(
        (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8")
    )
    assert source == target
    assert len(read_measurements(tmp_path / "target" / "state", PROFILE_ID).records) == 2


def test_a_newer_profile_arriving_within_the_same_second_is_compared_by_recorded_at(tmp_path):
    """`recorded_at` is the only freshness key -- the file's own modification time is never read."""
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", _export_data({"recorded_at": "2026-09-23T08:00:01Z"}, []))
    incoming = _export_data({"recorded_at": "2026-09-23T08:00:00Z", "gpu_name": "Nova GPU 9"}, [])

    main(["import-profile", str(_write_json(tmp_path / "e.json", incoming)), "--config", str(config_path)], now=NOW)

    stored = _read_profile(
        (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8")
    )
    assert stored.gpu_name == "Nova GPU"
    assert stored.recorded_at == datetime(2026, 9, 23, 8, 0, 1, tzinfo=timezone.utc)


def test_an_import_that_writes_reports_the_reserves_it_used(tmp_path, capsys):
    config_path = _deployment(tmp_path / "target")
    main(["import-profile", str(_write_json(tmp_path / "e.json", EXPORT)), "--config", str(config_path)], now=NOW)
    out = capsys.readouterr().out
    assert "[defaults]" in out and "writer = false" in out


def test_timedelta_is_not_used_for_the_freshness_rule(tmp_path):
    """A guard against a tolerance creeping in: one second younger already wins."""
    config_path = _deployment(tmp_path / "target")
    _install(tmp_path / "target" / "state", _export_data(None, []))
    stored_at = datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc)
    incoming = _export_data(
        {"recorded_at": (stored_at + timedelta(seconds=1)).isoformat().replace("+00:00", "Z"), "gpu_name": "Nova GPU 1"},
        [],
    )

    main(["import-profile", str(_write_json(tmp_path / "e.json", incoming)), "--config", str(config_path)], now=NOW)

    stored = _read_profile(
        (tmp_path / "target" / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8")
    )
    assert stored.gpu_name == "Nova GPU 1"
