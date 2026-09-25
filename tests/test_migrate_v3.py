"""The persisted step from schema 2 to profile schema 3 and configuration schema 3.

Reading turns a schema-2 file into schema 3 in memory and never writes it
(`tests/test_config_v3.py`, `tests/test_profile_v3.py`). Three paths write it back: `modelroom
migrate`, its trigger at the start of a guided run, and `import-profile`. Each keeps a backup
named after the version it left (`.v2.bak` next to an earlier `.v1.bak`), and a second run writes
nothing.
"""

from __future__ import annotations

import copy
import json
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.cli import main
from modelroom.guided import GuidedError
from modelroom.config import load_config
from modelroom.contracts import HardwareSnapshot
from modelroom.examples import EXAMPLES
from modelroom.migrate import NOTHING_TO_DO, migrate
from modelroom.profile import HardwareProfile, normalize_profile_v1

from test_guided import _config_file, _results, _run

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
EXPORT = json.loads((FIXTURES / "export_v1.json").read_text(encoding="utf-8"))
V2_PROFILE_ID = "c0ffee0000000001"


def _v2_profile(profile_id: str = V2_PROFILE_ID) -> dict:
    """A schema-2 profile file exactly as version 0.1.0 wrote it."""
    data = copy.deepcopy(EXAMPLES["HardwareProfile"])
    data.update(schema_version=2, profile_id=profile_id, display_name="laptop")
    return data


def _v2_folder(root: Path) -> Path:
    """A results folder as version 0.1.0 left it: configuration and one profile, both schema 2."""
    root.mkdir(parents=True, exist_ok=True)
    config = root / "modelroom.toml"
    shutil.copy(FIXTURES / "config_v2" / "config.toml", config)
    hardware = root / "state" / "hardware"
    hardware.mkdir(parents=True)
    (hardware / f"{V2_PROFILE_ID}.json").write_text(json.dumps(_v2_profile(), indent=2), encoding="utf-8", newline="\n")
    return config


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and p.name != "modelroom.lock"}


def _schema(path: Path) -> int:
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))["schema_version"]
    return json.loads(path.read_text(encoding="utf-8"))["schema_version"]


# --- modelroom migrate ---------------------------------------------------------------------------


def test_migrate_rewrites_a_schema_2_configuration_and_profile_in_place_with_backups(tmp_path):
    config = _v2_folder(tmp_path)
    profile = tmp_path / "state" / "hardware" / f"{V2_PROFILE_ID}.json"
    config_before, profile_before = config.read_bytes(), profile.read_bytes()

    lines = migrate(config, NOW)

    assert _schema(config) == 3 and _schema(profile) == 3
    assert (tmp_path / "modelroom.toml.v2.bak").read_bytes() == config_before
    assert (profile.parent / f"{V2_PROFILE_ID}.json.v2.bak").read_bytes() == profile_before
    assert sorted(p.name for p in profile.parent.iterdir()) == [f"{V2_PROFILE_ID}.json", f"{V2_PROFILE_ID}.json.v2.bak"]
    stored = HardwareProfile.model_validate_json(profile.read_text(encoding="utf-8"))
    assert stored.model_dump(exclude={"schema_version"}) == HardwareProfile.model_validate(
        {**_v2_profile(), "schema_version": 3}
    ).model_dump(exclude={"schema_version"})
    migrated = load_config(config)
    assert migrated.machines["laptop"].profile == V2_PROFILE_ID
    assert (migrated.defaults.reserve_ram_gib, migrated.defaults.reserve_vram_gib) == (6.0, 0.5)
    assert len(lines) == 2 and all("v2.bak" in line for line in lines)


def test_a_second_migrate_after_schema_2_says_nothing_to_do_and_writes_nothing(tmp_path):
    config = _v2_folder(tmp_path)
    assert migrate(config, NOW) != [NOTHING_TO_DO]
    before = _tree(tmp_path)

    assert migrate(config, NOW) == [NOTHING_TO_DO]
    assert _tree(tmp_path) == before


def test_the_cli_migrates_schema_2_with_exit_0_and_then_has_nothing_to_do(tmp_path, capsys):
    config = _v2_folder(tmp_path)
    assert main(["migrate", "--config", str(config)], now=NOW) == 0
    assert "modelroom.toml.v2.bak" in capsys.readouterr().out
    assert _schema(config) == 3
    assert main(["migrate", "--config", str(config)], now=NOW) == 0
    assert NOTHING_TO_DO in capsys.readouterr().out


def test_v1_to_v2_to_v3_in_the_same_folder_keeps_both_backups(tmp_path):
    """A folder 0.1.0 migrated from schema 1 carries `.v1.bak` files; the next step adds `.v2.bak`."""
    config = _v2_folder(tmp_path)
    hardware = tmp_path / "state" / "hardware"
    v1_config = (FIXTURES / "config_v1" / "config.toml").read_bytes()
    v1_profile = (FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json").read_bytes()
    (tmp_path / "modelroom.toml.v1.bak").write_bytes(v1_config)
    (hardware / "laptop.json.v1.bak").write_bytes(v1_profile)

    migrate(config, NOW)

    assert (tmp_path / "modelroom.toml.v1.bak").read_bytes() == v1_config
    assert (hardware / "laptop.json.v1.bak").read_bytes() == v1_profile
    assert (tmp_path / "modelroom.toml.v2.bak").is_file()
    assert (hardware / f"{V2_PROFILE_ID}.json.v2.bak").is_file()
    assert _schema(config) == 3 and _schema(hardware / f"{V2_PROFILE_ID}.json") == 3
    assert migrate(config, NOW) == [NOTHING_TO_DO]


def test_a_schema_1_folder_lands_on_schema_3_in_one_run(tmp_path):
    shutil.copy(FIXTURES / "config_v1" / "config.toml", tmp_path / "modelroom.toml")
    hardware = tmp_path / "state" / "hardware"
    hardware.mkdir(parents=True)
    shutil.copy(FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json", hardware / "laptop.json")

    migrate(tmp_path / "modelroom.toml", NOW, fresh_id=lambda: "aaaaaaaaaaaaaaa1")

    assert _schema(tmp_path / "modelroom.toml") == 3
    assert _schema(hardware / "aaaaaaaaaaaaaaa1.json") == 3
    assert (tmp_path / "modelroom.toml.v1.bak").is_file()
    assert not (tmp_path / "modelroom.toml.v2.bak").exists()
    assert migrate(tmp_path / "modelroom.toml", NOW) == [NOTHING_TO_DO]


def test_a_run_of_an_earlier_version_that_stopped_halfway_converges_on_schema_3(tmp_path):
    """0.1.0 wrote the converted profile as schema 2 and stopped before its backup of the old file."""
    shutil.copy(FIXTURES / "config_v1" / "config.toml", tmp_path / "modelroom.toml")
    hardware = tmp_path / "state" / "hardware"
    hardware.mkdir(parents=True)
    shutil.copy(FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json", hardware / "laptop.json")
    legacy = HardwareSnapshot.model_validate_json((hardware / "laptop.json").read_text(encoding="utf-8"))
    left = {**normalize_profile_v1(legacy, "aaaaaaaaaaaaaaa1").model_dump(mode="json"), "schema_version": 2}
    (hardware / "aaaaaaaaaaaaaaa1.json").write_text(json.dumps(left), encoding="utf-8", newline="\n")

    migrate(tmp_path / "modelroom.toml", NOW, fresh_id=lambda: "bbbbbbbbbbbbbbb1")

    assert sorted(p.name for p in hardware.iterdir()) == [
        "aaaaaaaaaaaaaaa1.json",
        "aaaaaaaaaaaaaaa1.json.v2.bak",
        "laptop.json.v1.bak",
    ]
    assert _schema(hardware / "aaaaaaaaaaaaaaa1.json") == 3
    assert load_config(tmp_path / "modelroom.toml").machines["laptop"].profile == "aaaaaaaaaaaaaaa1"
    assert migrate(tmp_path / "modelroom.toml", NOW) == [NOTHING_TO_DO]


def test_a_stored_v2_backup_with_other_content_stops_the_run_before_any_write(tmp_path):
    config = _v2_folder(tmp_path)
    (tmp_path / "modelroom.toml.v2.bak").write_text("something else\n", encoding="utf-8")
    before = _tree(tmp_path)

    try:
        migrate(config, NOW)
    except Exception as exc:  # noqa: BLE001 -- the kind is MigrationError, the point is: nothing written
        assert "backup" in str(exc)
    else:
        raise AssertionError("a foreign backup must stop the run")
    assert _tree(tmp_path) == before


# --- the guided mode's trigger -------------------------------------------------------------------


def _v2_guided_config(tmp_path: Path) -> None:
    (tmp_path / "home").mkdir(exist_ok=True)
    _results(tmp_path).mkdir(exist_ok=True)
    _config_file(tmp_path).write_text(
        "\n".join(
            [
                "schema_version = 2",
                'packagers = ["unsloth"]',
                'publishers = ["Qwen"]',
                "[[families]]",
                'name = "qwen3.5"',
                "  [[families.base_models]]",
                '  hf_repo = "Qwen/Qwen3.5-9B"',
                "[machines.workstation]",
                "reserve_ram_gib = 8",
                "reserve_vram_gib = 1",
                "writer = true",
                "[paths]",
                'state = "state"',
                'markdown = "docs/models.md"',
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def test_a_guided_run_rewrites_a_schema_2_configuration_and_says_so(tmp_path):
    _v2_guided_config(tmp_path)

    code, lines = _run(tmp_path)

    assert code == 0
    assert (_results(tmp_path) / "modelroom.toml.v2.bak").is_file()
    assert _schema(_config_file(tmp_path)) == 3
    assert (
        "this folder holds a configuration from an earlier version; it was updated, "
        "backup kept: modelroom.toml.v2.bak"
    ) in [line.strip() for line in lines]


def test_a_guided_run_that_cannot_migrate_names_the_file_and_claims_no_update(tmp_path):
    _v2_guided_config(tmp_path)
    broken = _results(tmp_path) / "state" / "hardware" / "broken.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{", encoding="utf-8")

    with pytest.raises(GuidedError, match="broken.json") as caught:
        _run(tmp_path)

    assert "move it away" in str(caught.value) or "cannot read" in str(caught.value)
    assert _schema(_config_file(tmp_path)) == 2
    assert not (_results(tmp_path) / "modelroom.toml.v2.bak").exists()


# --- import-profile ------------------------------------------------------------------------------


def _import(tmp_path: Path, config: Path) -> int:
    export_file = tmp_path / "export.json"
    export_file.write_text(json.dumps(EXPORT), encoding="utf-8")
    return main(["import-profile", str(export_file), "--config", str(config)], now=NOW)


def test_import_rewrites_a_schema_2_configuration_when_it_adds_a_machine(tmp_path):
    config = _v2_folder(tmp_path / "results")
    before = config.read_bytes()

    assert _import(tmp_path, config) == 0

    assert _schema(config) == 3
    assert (config.parent / "modelroom.toml.bak").read_bytes() == before
    stored = config.parent / "state" / "hardware" / f"{EXPORT['profile']['profile_id']}.json"
    assert _schema(stored) == 3


def test_import_keeps_the_schema_2_bytes_even_when_its_own_backup_is_already_there(tmp_path, capsys):
    """`import-profile` never overwrites its `.bak`; the step to schema 3 keeps `.v2.bak` of its own."""
    config = _v2_folder(tmp_path / "results")
    (config.parent / "modelroom.toml.bak").write_text("# an earlier import's backup\n", encoding="utf-8")
    before = config.read_bytes()

    assert _import(tmp_path, config) == 0

    assert _schema(config) == 3
    assert (config.parent / "modelroom.toml.v2.bak").read_bytes() == before
    out = capsys.readouterr().out
    assert "schema 3" in out and "modelroom.toml.v2.bak" in out


def test_import_rewrites_a_schema_2_configuration_even_when_the_machine_is_already_there(tmp_path):
    config = _v2_folder(tmp_path / "results")
    text = config.read_text(encoding="utf-8").replace(V2_PROFILE_ID, EXPORT["profile"]["profile_id"])
    config.write_text(text, encoding="utf-8", newline="\n")
    before = config.read_bytes()

    assert _import(tmp_path, config) == 0

    assert _schema(config) == 3
    assert (config.parent / "modelroom.toml.bak").read_bytes() == before
    assert load_config(config).machines["laptop"].profile == EXPORT["profile"]["profile_id"]
    after = _tree(tmp_path / "results")
    assert _import(tmp_path, config) == 0
    assert _tree(tmp_path / "results") == after
