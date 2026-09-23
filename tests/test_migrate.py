"""Tests for `modelroom migrate` and the TOML writer it uses.

Every test works on a copy of the schema-1 fixtures (`fixtures/config_v1/modelroom.toml`, two
machines, and `fixtures/profiles_v1/`) in a temporary folder.
"""

from __future__ import annotations

import copy
import json
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

import modelroom.migrate as migrate_module
from modelroom.cli import main
from modelroom.config import ConfigError, load_config
from modelroom.contracts import HardwareSnapshot, SchemaVersionError
from modelroom.examples import EXAMPLES
from modelroom.measurements import legacy_measurement_record, measurement_path, read_measurements
from modelroom.migrate import NOTHING_TO_DO, MigrationError, migrate
from modelroom.profile import HardwareProfile
from modelroom.state import acquire_lock, release_lock
from modelroom.toml_writer import dump_toml

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
IDS = ["aaaaaaaaaaaaaaa1", "aaaaaaaaaaaaaaa2", "aaaaaaaaaaaaaaa3"]


def _ids():
    iterator = iter(IDS)
    return lambda: next(iterator)


def _deployment(tmp_path: Path, profiles=("windows-nvidia-laptop.json", "linux-cpu-server.json")) -> Path:
    shutil.copy(FIXTURES / "config_v1" / "modelroom.toml", tmp_path / "modelroom.toml")
    hardware = tmp_path / "state" / "hardware"
    hardware.mkdir(parents=True)
    names = {"windows-nvidia-laptop.json": "laptop.json", "linux-cpu-server.json": "server.json"}
    for fixture in profiles:
        shutil.copy(FIXTURES / "profiles_v1" / fixture, hardware / names.get(fixture, fixture))
    return tmp_path / "modelroom.toml"


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and p.name != "modelroom.lock"}


# --- first run, second run ------------------------------------------------------------------


def test_first_run_migrates_profiles_measurements_and_configuration(tmp_path):
    config_path = _deployment(tmp_path)
    original_config = config_path.read_bytes()
    original_laptop = (tmp_path / "state" / "hardware" / "laptop.json").read_bytes()

    lines = migrate(config_path, NOW, fresh_id=_ids())

    hardware = tmp_path / "state" / "hardware"
    assert len(lines) == 3
    assert sorted(p.name for p in hardware.iterdir()) == [
        "aaaaaaaaaaaaaaa1.json",
        "aaaaaaaaaaaaaaa2.json",
        "laptop.json.v1.bak",
        "server.json.v1.bak",
    ]
    assert (hardware / "laptop.json.v1.bak").read_bytes() == original_laptop
    assert (tmp_path / "modelroom.toml.v1.bak").read_bytes() == original_config

    laptop = HardwareProfile.model_validate_json((hardware / "aaaaaaaaaaaaaaa1.json").read_text(encoding="utf-8"))
    assert laptop.display_name == "laptop" and laptop.gpu_state == "legacy_unknown"
    records = read_measurements(tmp_path / "state", "aaaaaaaaaaaaaaa1")
    assert len(records.records) == 2 and not records.unreadable
    assert {r.protocol for r in records.records} == {"none"}
    assert read_measurements(tmp_path / "state", "aaaaaaaaaaaaaaa2").records == []

    config = load_config(config_path)
    assert tomllib.loads(config_path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert config.machines["laptop"].profile == "aaaaaaaaaaaaaaa1"
    assert config.machines["server"].profile == "aaaaaaaaaaaaaaa2"
    assert config.machines["laptop"].writer is True
    assert [f.name for f in config.families] == ["nova"]


def test_migrated_configuration_keeps_every_schema_1_value(tmp_path):
    config_path = _deployment(tmp_path)
    before = load_config(config_path)
    migrate(config_path, NOW, fresh_id=_ids())
    after = load_config(config_path)
    for machine in ("laptop", "server"):
        assert after.machines[machine].model_copy(update={"profile": None}) == before.machines[machine]
    assert (after.families, after.packagers, after.publishers, after.paths, after.llmfit) == (
        before.families,
        before.packagers,
        before.publishers,
        before.paths,
        before.llmfit,
    )


def test_second_run_says_nothing_to_do_and_writes_nothing(tmp_path):
    config_path = _deployment(tmp_path)
    migrate(config_path, NOW, fresh_id=_ids())
    before = _tree(tmp_path)
    assert migrate(config_path, NOW, fresh_id=_ids()) == [NOTHING_TO_DO]
    assert _tree(tmp_path) == before


def test_a_schema_2_deployment_without_profiles_is_nothing_to_do(tmp_path):
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text('schema_version = 2\n\n[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n', encoding="utf-8")
    assert migrate(config_path, NOW) == [NOTHING_TO_DO]


def test_a_run_that_stopped_after_the_profile_converges_to_the_same_result(tmp_path):
    """Simulate a stop after the laptop profile was written but before its backup: the next run
    reuses the profile_id (same display_name, fingerprint source legacy) and writes the same files."""
    clean = tmp_path / "clean"
    clean.mkdir()
    migrate(_deployment(clean), NOW, fresh_id=_ids())

    broken = tmp_path / "broken"
    broken.mkdir()
    config_path = _deployment(broken)
    hardware = broken / "state" / "hardware"
    shutil.copy(clean / "state" / "hardware" / "aaaaaaaaaaaaaaa1.json", hardware / "aaaaaaaaaaaaaaa1.json")
    shutil.copytree(clean / "state" / "measurements", broken / "state" / "measurements")

    migrate(config_path, NOW, fresh_id=iter(["aaaaaaaaaaaaaaa2"]).__next__)
    assert _tree(broken) == _tree(clean)


# --- refusals: nothing is written ------------------------------------------------------------


def test_a_schema_3_profile_stops_before_anything_is_written(tmp_path):
    config_path = _deployment(tmp_path)
    (tmp_path / "state" / "hardware" / "future.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(SchemaVersionError):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(tmp_path) == before


def test_a_schema_3_configuration_stops_before_anything_is_written(tmp_path):
    config_path = _deployment(tmp_path)
    config_path.write_text(config_path.read_text(encoding="utf-8").replace("schema_version = 1", "schema_version = 3"), encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(SchemaVersionError):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(tmp_path) == before


@pytest.mark.parametrize(
    "content", ["{", '{"schema_version": 1' + "0" * 4300 + "}"], ids=["broken-json", "integer-too-long"]
)
def test_an_unreadable_profile_stops_before_anything_is_written(tmp_path, content):
    config_path = _deployment(tmp_path)
    (tmp_path / "state" / "hardware" / "broken.json").write_text(content, encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(MigrationError, match="broken.json"):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(tmp_path) == before


def test_a_backup_with_other_content_stops_before_anything_is_written(tmp_path):
    config_path = _deployment(tmp_path, profiles=("windows-nvidia-laptop.json",))
    backup = tmp_path / "state" / "hardware" / "laptop.json.v1.bak"
    backup.write_text("someone else's file", encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(MigrationError, match="backup exists"):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(tmp_path) == before
    assert not (tmp_path / "state" / "modelroom.lock").exists()  # refused before the lock


def test_a_configuration_backup_with_other_content_stops_before_anything_is_written(tmp_path):
    config_path = _deployment(tmp_path)
    (tmp_path / "modelroom.toml.v1.bak").write_text("someone else's file", encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(MigrationError, match="backup exists"):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("backup_name", ["modelroom.toml.v1.bak", "state/hardware/laptop.json.v1.bak"])
def test_cli_migrate_with_a_folder_at_a_backup_path_exits_3_without_writing(tmp_path, capsys, backup_name):
    config_path = _deployment(tmp_path, profiles=("windows-nvidia-laptop.json",))
    (tmp_path / backup_name).mkdir()
    before = _tree(tmp_path)
    assert main(["migrate", "--config", str(config_path)], now=NOW) == 3
    assert Path(backup_name).name in capsys.readouterr().err
    assert _tree(tmp_path) == before
    assert not (tmp_path / "state" / "modelroom.lock").exists()  # refused before the lock


def test_an_existing_target_profile_with_other_content_stops_before_anything_is_written(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    migrate(_deployment(clean, profiles=("windows-nvidia-laptop.json",)), NOW, fresh_id=_ids())
    target = json.loads((clean / "state" / "hardware" / "aaaaaaaaaaaaaaa1.json").read_text(encoding="utf-8"))
    target["ram_physical_gib"] = 15.6  # an older reading of the same machine

    broken = tmp_path / "broken"
    broken.mkdir()
    config_path = _deployment(broken, profiles=("windows-nvidia-laptop.json",))
    (broken / "state" / "hardware" / "aaaaaaaaaaaaaaa1.json").write_text(json.dumps(target), encoding="utf-8")
    before = _tree(broken)
    with pytest.raises(MigrationError, match="aaaaaaaaaaaaaaa1.json"):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(broken) == before


def test_a_measurement_file_with_other_content_stops_before_anything_is_written(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    migrate(_deployment(clean, profiles=("windows-nvidia-laptop.json",)), NOW, fresh_id=_ids())
    broken = tmp_path / "broken"
    broken.mkdir()
    config_path = _deployment(broken, profiles=("windows-nvidia-laptop.json",))
    shutil.copy(clean / "state" / "hardware" / "aaaaaaaaaaaaaaa1.json", broken / "state" / "hardware")
    shutil.copytree(clean / "state" / "measurements", broken / "state" / "measurements")
    one = next((broken / "state" / "measurements").rglob("*.json"))
    one.write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    before = _tree(broken)
    with pytest.raises(MigrationError, match=one.name):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(broken) == before


def test_a_measurement_file_under_the_new_id_is_refused_before_the_lock(tmp_path):
    config_path = _deployment(tmp_path, profiles=("windows-nvidia-laptop.json",))
    legacy = HardwareSnapshot.model_validate(
        json.loads((FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json").read_text(encoding="utf-8"))
    )
    record = legacy_measurement_record(legacy.measurements[0], IDS[0])
    target = measurement_path(tmp_path / "state", IDS[0], record.measurement_id)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(MigrationError, match=target.name):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(tmp_path) == before
    assert not (tmp_path / "state" / "modelroom.lock").exists()  # refused before the lock


def test_every_machine_draws_one_id_for_the_check_and_the_run(tmp_path):
    drawn: list[str] = []
    ids = _ids()

    def counting_id() -> str:
        drawn.append(ids())
        return drawn[-1]

    migrate(_deployment(tmp_path), NOW, fresh_id=counting_id)
    assert drawn == IDS[:2]


def test_an_unrelated_profile_does_not_block_the_migration(tmp_path):
    config_path = _deployment(tmp_path, profiles=("windows-nvidia-laptop.json",))
    other = copy.deepcopy(EXAMPLES["HardwareProfile"])
    other.update(profile_id="0" * 16, display_name="elsewhere")
    (tmp_path / "state" / "hardware" / ("0" * 16 + ".json")).write_text(json.dumps(other), encoding="utf-8")
    assert migrate(config_path, NOW, fresh_id=_ids()) != [NOTHING_TO_DO]
    assert (tmp_path / "state" / "hardware" / f"{IDS[0]}.json").exists()


def test_paths_changed_while_waiting_for_the_lock_stop_the_run(tmp_path, monkeypatch):
    config_path = _deployment(tmp_path, profiles=("windows-nvidia-laptop.json",))
    original = config_path.read_text(encoding="utf-8")
    real_acquire = migrate_module.acquire_lock

    def acquire_after_an_edit(*args, **kwargs):
        # Another process edits the configuration between the check and the lock.
        config_path.write_text(original.replace('state = "state"', 'state = "state-b"'), encoding="utf-8")
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(migrate_module, "acquire_lock", acquire_after_an_edit)
    with pytest.raises(MigrationError, match="changed"):
        migrate(config_path, NOW, fresh_id=_ids())
    assert sorted(p.name for p in (tmp_path / "state" / "hardware").iterdir()) == ["laptop.json"]
    assert not (tmp_path / "state-b").exists()


def test_a_configuration_named_like_the_lock_file_is_refused_unchanged(tmp_path):
    config_path = tmp_path / "modelroom.lock"
    text = f'schema_version = 2\n[paths]\nstate = "."\nmarkdown = "{tmp_path.parent.as_posix()}/models.md"\n'
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        migrate(config_path, NOW, fresh_id=_ids())
    assert config_path.read_text(encoding="utf-8") == text


def test_a_measurements_path_that_is_a_file_stops_before_anything_is_written(tmp_path):
    config_path = _deployment(tmp_path, profiles=("windows-nvidia-laptop.json",))
    (tmp_path / "state" / "measurements").write_text("in the way", encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(MigrationError, match="not a folder"):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(tmp_path) == before
    assert not (tmp_path / "state" / "modelroom.lock").exists()  # refused before the lock


def test_two_schema_1_files_for_one_machine_are_refused(tmp_path):
    config_path = _deployment(tmp_path, profiles=("windows-nvidia-laptop.json",))
    shutil.copy(FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json", tmp_path / "state" / "hardware" / "copy.json")
    before = _tree(tmp_path)
    with pytest.raises(MigrationError, match="laptop"):
        migrate(config_path, NOW, fresh_id=_ids())
    assert _tree(tmp_path) == before


def test_a_schema_1_range_starting_at_zero_migrates_unchanged(tmp_path):
    config_path = _deployment(tmp_path, profiles=("windows-nvidia-laptop.json",))
    path = tmp_path / "state" / "hardware" / "laptop.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["measurements"][0]["tps_range"] = [0.0, 45.0]
    path.write_text(json.dumps(data), encoding="utf-8")
    migrate(config_path, NOW, fresh_id=_ids())
    records = read_measurements(tmp_path / "state", "aaaaaaaaaaaaaaa1").records
    assert sorted((r.tps_min, r.tps_max) for r in records) == [(0.0, 45.0), (21.5, 23.0)]


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_migrate_twice_then_schema_3_exits_3_without_writing(tmp_path, capsys):
    config_path = _deployment(tmp_path)

    assert main(["migrate", "--config", str(config_path)], now=NOW) == 0
    first = capsys.readouterr().out
    assert "migrated laptop.json" in first and "migrated modelroom.toml" in first

    assert main(["migrate", "--config", str(config_path)], now=NOW) == 0
    assert capsys.readouterr().out.strip() == NOTHING_TO_DO

    (tmp_path / "state" / "hardware" / "future.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    before = _tree(tmp_path)
    assert main(["migrate", "--config", str(config_path)], now=NOW) == 3
    err = capsys.readouterr().err
    assert "schema_version" in err and "future.json" in err  # the message names the file
    assert _tree(tmp_path) == before


def test_cli_migrate_missing_configuration_exits_2(tmp_path, capsys):
    assert main(["migrate", "--config", str(tmp_path / "missing.toml")], now=NOW) == 2
    assert "missing.toml" in capsys.readouterr().err


def test_cli_migrate_with_a_held_lock_exits_1_without_writing(tmp_path, capsys):
    config_path = _deployment(tmp_path)
    handle = acquire_lock(tmp_path / "state" / "modelroom.lock", "fetch", NOW)
    try:
        before = _tree(tmp_path)
        assert main(["migrate", "--config", str(config_path)], now=NOW) == 1
        assert _tree(tmp_path) == before
    finally:
        release_lock(handle)


# --- TOML writer ---------------------------------------------------------------------------------


def test_dump_toml_round_trips_the_configuration_shapes():
    data = {
        "schema_version": 2,
        "packagers": ["packager", "other"],
        "families": [
            {"name": "qwen3.5", "base_models": [{"hf_repo": "acme/Nova-7B", "repo_aliases": [], "ollama_base": None}]},
            {"name": "nova", "base_models": [{"hf_repo": "acme/Nova-1B"}, {"hf_repo": "acme/Nova-3B"}]},
        ],
        "machines": {"laptop": {"reserve_ram_gib": 8.0, "writer": True}, "odd name": {"reserve_ram_gib": 0.5}},
        "paths": {"state": Path("state"), "markdown": 'docs/"quoted" models.md'},
        "text": "line one\nline two ä",
    }
    parsed = tomllib.loads(dump_toml(data, header="first line\n\nthird line"))
    expected = json.loads(json.dumps(data, default=lambda p: p.as_posix()))
    expected["families"][0]["base_models"][0].pop("ollama_base")  # None is left out
    assert parsed == expected


def test_dump_toml_writes_the_header_as_comments_and_ends_with_one_newline():
    text = dump_toml({"a": 1}, header="modelroom\nschema 2")
    assert text == "# modelroom\n# schema 2\n\na = 1\n"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), [{"a": 1}, 2]])
def test_dump_toml_refuses_values_toml_cannot_hold(value):
    with pytest.raises(TypeError):
        dump_toml({"value": value})


def test_dump_toml_writes_no_empty_header_for_a_table_of_tables():
    text = dump_toml({"machines": {"laptop": {"writer": True}}, "empty": {}})
    assert text == "[machines.laptop]\nwriter = true\n\n[empty]\n"
    assert tomllib.loads(text) == {"machines": {"laptop": {"writer": True}}, "empty": {}}


def test_dump_toml_writes_scalars_before_tables_before_arrays_of_tables():
    text = dump_toml({"b": {"x": 1}, "c": [{"k": "v"}], "a": 2})
    assert text == 'a = 2\n\n[b]\nx = 1\n\n[[c]]\nk = "v"\n'
