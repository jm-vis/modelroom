"""The persisted step to hardware profile schema 4 (decided 2026-09-26).

`modelroom migrate` rewrites a schema-2 or schema-3 profile in place and keeps the file it left
as `.v2.bak` or `.v3.bak`, named after the stored version. A run of an earlier version that
stopped halfway is recognized by reading the target it left, whatever its schema. The files are
real files of 0.1.0 (`tests/fixtures/profiles_v2`, `profiles_v3`), never the schema-4 example
with its number set back.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.cli import main
from modelroom.contracts import HardwareSnapshot
from modelroom.migrate import NOTHING_TO_DO, MigrationError, migrate
from modelroom.profile import HardwareProfile, normalize_profile_v1, read_profile_document

from fixture_support import v3_profile

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
V2_ID, V3_ID = "c0ffee0000000002", "c0ffee0000000003"


def _folder(root: Path) -> Path:
    """A results folder of 0.1.0: a schema-3 configuration, one schema-2 and one schema-3 profile."""
    root.mkdir(parents=True, exist_ok=True)
    config = root / "modelroom.toml"
    shutil.copy(FIXTURES / "config_v3" / "config.toml", config)
    hardware = root / "state" / "hardware"
    hardware.mkdir(parents=True)
    for version, profile_id in ((2, V2_ID), (3, V3_ID)):
        data = v3_profile(schema_version=version, profile_id=profile_id, display_name=f"box {version}")
        (hardware / f"{profile_id}.json").write_bytes(json.dumps(data, indent=2).encode("utf-8"))
    return config


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and p.name != "modelroom.lock"}


def _schema(path: Path) -> int:
    return json.loads(path.read_text(encoding="utf-8"))["schema_version"]


def test_a_schema_2_and_a_schema_3_profile_both_become_schema_4_each_with_its_own_backup(tmp_path: Path):
    config = _folder(tmp_path)
    hardware = tmp_path / "state" / "hardware"
    before = {name: (hardware / f"{name}.json").read_bytes() for name in (V2_ID, V3_ID)}

    lines = migrate(config, NOW)

    assert _schema(hardware / f"{V2_ID}.json") == 4 and _schema(hardware / f"{V3_ID}.json") == 4
    assert (hardware / f"{V2_ID}.json.v2.bak").read_bytes() == before[V2_ID]
    assert (hardware / f"{V3_ID}.json.v3.bak").read_bytes() == before[V3_ID]
    assert not (hardware / f"{V3_ID}.json.v2.bak").exists()
    for profile_id in (V2_ID, V3_ID):
        stored = HardwareProfile.model_validate_json((hardware / f"{profile_id}.json").read_text(encoding="utf-8"))
        assert stored == read_profile_document(json.loads(before[profile_id]))
        assert [gpu.name for gpu in stored.gpus] == ["Nova GPU"] and stored.machine_class == "unknown"
    assert len(lines) == 2 and any("v2.bak" in line for line in lines) and any("v3.bak" in line for line in lines)


def test_a_second_run_writes_nothing(tmp_path: Path):
    config = _folder(tmp_path)
    migrate(config, NOW)
    before = _tree(tmp_path)

    assert migrate(config, NOW) == [NOTHING_TO_DO]
    assert _tree(tmp_path) == before


def test_a_v3_backup_of_other_content_stops_the_run_before_any_write(tmp_path: Path):
    config = _folder(tmp_path)
    (tmp_path / "state" / "hardware" / f"{V3_ID}.json.v3.bak").write_text("{}", encoding="utf-8")
    before = _tree(tmp_path)

    with pytest.raises(MigrationError, match="backup exists with other content"):
        migrate(config, NOW)
    assert _tree(tmp_path) == before


def test_a_folder_that_went_from_1_to_2_and_to_3_keeps_every_backup(tmp_path: Path):
    config = _folder(tmp_path)
    hardware = tmp_path / "state" / "hardware"
    v1 = (FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json").read_bytes()
    v2 = json.dumps(v3_profile(schema_version=2, profile_id=V3_ID)).encode("utf-8")
    (hardware / "laptop.json.v1.bak").write_bytes(v1)
    (hardware / f"{V3_ID}.json.v2.bak").write_bytes(v2)

    migrate(config, NOW)

    assert (hardware / "laptop.json.v1.bak").read_bytes() == v1
    assert (hardware / f"{V3_ID}.json.v2.bak").read_bytes() == v2
    assert (hardware / f"{V3_ID}.json.v3.bak").is_file() and _schema(hardware / f"{V3_ID}.json") == 4
    assert migrate(config, NOW) == [NOTHING_TO_DO]


def _v1_folder(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "config_v1" / "config.toml", root / "modelroom.toml")
    hardware = root / "state" / "hardware"
    hardware.mkdir(parents=True)
    shutil.copy(FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json", hardware / "laptop.json")
    return root / "modelroom.toml"


def _left_by_0_1_0(hardware: Path, profile_id: str) -> dict:
    """The profile 0.1.0 wrote for the schema-1 file before it stopped: schema 3, no card list."""
    legacy = HardwareSnapshot.model_validate_json((hardware / "laptop.json").read_text(encoding="utf-8"))
    dumped = normalize_profile_v1(legacy, profile_id).model_dump(mode="json")
    added = ("gpus", "machine_class", "machine_class_source")
    return {**{key: value for key, value in dumped.items() if key not in added}, "schema_version": 3}


def test_a_schema_1_profile_lands_on_schema_4_in_one_run(tmp_path: Path):
    config = _v1_folder(tmp_path)
    migrate(config, NOW, fresh_id=lambda: "aaaaaaaaaaaaaaa1")
    hardware = tmp_path / "state" / "hardware"
    assert _schema(hardware / "aaaaaaaaaaaaaaa1.json") == 4
    assert sorted(p.name for p in hardware.iterdir()) == ["aaaaaaaaaaaaaaa1.json", "laptop.json.v1.bak"]


def test_a_run_of_0_1_0_that_stopped_with_a_schema_3_target_converges_on_schema_4(tmp_path: Path):
    config = _v1_folder(tmp_path)
    hardware = tmp_path / "state" / "hardware"
    left = _left_by_0_1_0(hardware, "aaaaaaaaaaaaaaa1")
    (hardware / "aaaaaaaaaaaaaaa1.json").write_bytes(json.dumps(left).encode("utf-8"))

    migrate(config, NOW, fresh_id=lambda: "bbbbbbbbbbbbbbb1")

    assert sorted(p.name for p in hardware.iterdir()) == [
        "aaaaaaaaaaaaaaa1.json",
        "aaaaaaaaaaaaaaa1.json.v3.bak",
        "laptop.json.v1.bak",
    ]
    assert _schema(hardware / "aaaaaaaaaaaaaaa1.json") == 4
    assert migrate(config, NOW) == [NOTHING_TO_DO]


def test_import_profile_takes_an_export_of_0_1_0_with_a_schema_3_profile(tmp_path: Path):
    config = _folder(tmp_path / "results")
    export = tmp_path / "export.json"
    shutil.copy(FIXTURES / "export_v1_profile_v3.json", export)

    assert main(["import-profile", str(export), "--config", str(config)], now=NOW) == 0

    stored = tmp_path / "results" / "state" / "hardware" / "3f9a0c21d4e6b870.json"
    assert _schema(stored) == 4
    assert [gpu.name for gpu in HardwareProfile.model_validate_json(stored.read_text(encoding="utf-8")).gpus] == ["Nova GPU"]


def test_a_target_of_other_content_stops_the_resumed_run(tmp_path: Path):
    config = _v1_folder(tmp_path)
    hardware = tmp_path / "state" / "hardware"
    left = {**_left_by_0_1_0(hardware, "aaaaaaaaaaaaaaa1"), "ram_physical_gib": 1.0}
    (hardware / "aaaaaaaaaaaaaaa1.json").write_bytes(json.dumps(left).encode("utf-8"))
    before = _tree(tmp_path)

    with pytest.raises(MigrationError, match="exists with other content"):
        migrate(config, NOW, fresh_id=lambda: "bbbbbbbbbbbbbbb1")
    assert _tree(tmp_path) == before
