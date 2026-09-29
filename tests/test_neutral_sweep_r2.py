"""The sink of machine names, third round (FR-N, Codex diff review r2): the documents one by one, the
comments and the paths of a configuration, the backups of 0.1.0's profiles, and references of records.

Every test here failed on the code of the second round (`2b983c5`); the report names each.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from fixture_neutral import SENTINEL_HOST, THIS_MACHINE, files_with
from fixture_support import PROFILE_ID, windows_probes
from modelroom.cli import main
from modelroom.config import load_config
from modelroom.migrate import migrate
from modelroom.neutral import FAILED, is_neutral, legacy_data, neutral_name
from modelroom.neutral_sweep import sweep

from test_neutral import NOW, _folder, _leaky_record, _pointer, _profile, _write
from test_neutral_sweep import _v1_folder

P1 = "a1aaaaaaaaaaaaaa"


# --- r2-1: the documents, each on its own; migrate takes them along ---------------------------------------


@pytest.mark.parametrize(
    "markdown",
    [f"## Ranking: {SENTINEL_HOST}\n", f"## Machines\n\n| Machine | Profile |\n| --- | --- |\n| {THIS_MACHINE} | {SENTINEL_HOST} (3f9a0c21d4e6b870) |\n"],
    ids=["heading", "machines-table"],
)
def test_a_markdown_view_that_names_a_machine_goes_although_the_json_view_is_neutral(tmp_path, markdown):
    config = _folder(tmp_path / "f", {THIS_MACHINE: PROFILE_ID}, writer=THIS_MACHINE)
    _profile(config.parent, PROFILE_ID, "laptop-3f9a0c21", "laptop")
    docs = config.parent / "docs"
    _write(docs / "models.json", {"machines": [{"machine": THIS_MACHINE, "label": "laptop-3f9a0c21", "profile": {"display_name": "laptop-3f9a0c21"}}]})
    _write(docs / "models.md", markdown)

    sweep(config, NOW, "render", pointer_path=_pointer(tmp_path / "home"))

    assert not (docs / "models.md").exists() and not (docs / "models.json").exists()


def test_migrate_of_0_1_0_takes_the_documents_that_name_its_machines(tmp_path):
    config = _v1_folder(tmp_path / "old")
    _write(config.parent / "docs" / "models.md", f"## Ranking: {SENTINEL_HOST}\n")
    _write(config.parent / "docs" / "models.json", {"machines": [{"machine": SENTINEL_HOST, "label": SENTINEL_HOST, "profile": None}]})

    lines = migrate(config, NOW, fresh_id=lambda: P1, pointer_path=_pointer(tmp_path / "home"))

    assert files_with(config.parent, SENTINEL_HOST) == [] and not (config.parent / "docs" / "models.md").exists()
    assert any("the next render writes them" in line for line in lines)


# --- r2-2/6: comments, escaped and reserved keys, paths below home ----------------------------------------


def test_a_comment_that_names_a_renamed_machine_is_kept_with_the_new_name(tmp_path):
    config = _folder(tmp_path / "f", {SENTINEL_HOST: None}, writer=SENTINEL_HOST)
    text = config.read_text(encoding="utf-8").replace(f"[machines.{SENTINEL_HOST}]", f"# {SENTINEL_HOST.upper()} is my desk\n[machines.{SENTINEL_HOST}]  # '{SENTINEL_HOST}'")
    config.write_bytes(text.encode("utf-8"))

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"), hostname=SENTINEL_HOST)

    written = config.read_text(encoding="utf-8")
    assert SENTINEL_HOST not in written.lower() and f"# {THIS_MACHINE} is my desk\n" in written
    assert written.startswith("# written by hand, kept by the sweep")


@pytest.mark.parametrize("quoted", [f"'{SENTINEL_HOST.upper()}'", f'"{SENTINEL_HOST.title()}"', f"'{SENTINEL_HOST}'"], ids=["single-upper", "double-title", "single-same"])
def test_a_quoted_name_in_a_comment_goes_in_any_case(tmp_path, quoted):
    """Codex r3-1: a quoted name in a comment is found in any case (a key itself is found exactly)."""
    config = _folder(tmp_path / "f", {SENTINEL_HOST: None}, writer=SENTINEL_HOST)
    text = config.read_text(encoding="utf-8").replace(f"[machines.{SENTINEL_HOST}]", f"# the desk {quoted}\n[machines.{SENTINEL_HOST}]")
    config.write_bytes(text.encode("utf-8"))

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"), hostname=SENTINEL_HOST)

    written = config.read_text(encoding="utf-8")
    assert SENTINEL_HOST not in written.lower() and f"# the desk {THIS_MACHINE}\n" in written and written.startswith("# written by hand")
    assert list(tomllib.loads(written)["machines"]) == [THIS_MACHINE]


def test_escaped_keys_and_keys_spelled_like_fields_keep_the_comments(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    escaped = "".join(f"\\u{ord(char):04X}" for char in SENTINEL_HOST)
    config = folder / "modelroom.toml"
    config.write_bytes(
        (
            "# kept\nschema_version = 3\npackagers = []\npublishers = []\n\n"
            f'[machines."{escaped}"]\nreserve_vram_gib = 1.0\nwriter = true\n\n'
            "[machines.state]  # the machine called state\nreserve_vram_gib = 1.0\nwriter = false\n\n"
            '[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n'
        ).encode("utf-8")
    )

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"), hostname=SENTINEL_HOST)

    text = config.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    assert text.startswith("# kept\n") and escaped not in text and "the machine called " in text
    assert all(is_neutral(key) for key in raw["machines"]) and raw["paths"]["state"] == "state"


def test_a_path_outside_the_folder_below_home_is_kept_as_tilde(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for variable in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(variable, str(home))
    markdown = home / "notes" / "models.md"
    config = _folder(tmp_path / "f", {THIS_MACHINE: None}, writer=THIS_MACHINE)
    config.write_bytes(config.read_text(encoding="utf-8").replace('markdown = "docs/models.md"', f"markdown = {json.dumps(str(markdown))}").encode("utf-8"))

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "pointer-home"))

    assert tomllib.loads(config.read_text(encoding="utf-8"))["paths"]["markdown"] == "~/notes/models.md"
    assert load_config(config).paths.markdown == markdown.resolve()


# --- r2-4: backups of 0.1.0's profiles ----------------------------------------------------------------------


def _migrated(tmp_path: Path) -> tuple[Path, Path, dict]:
    """A folder `migrate` converted; its profile's own backup taken away again, the schema-1 file returned."""
    config = _v1_folder(tmp_path / "old")
    source = json.loads((config.parent / "state" / "hardware" / f"{SENTINEL_HOST}.json").read_text(encoding="utf-8"))
    migrate(config, NOW, fresh_id=lambda: P1, pointer_path=_pointer(tmp_path / "home"))
    hardware = config.parent / "state" / "hardware"
    return config, hardware, source


def _kept(source: dict, key: str) -> dict:
    from modelroom.contracts import HardwareSnapshot

    return legacy_data(HardwareSnapshot.model_validate(source).model_dump(mode="json"), key)


def test_two_backups_of_one_profile_both_stay_under_their_own_names(tmp_path):
    config, hardware, source = _migrated(tmp_path)
    (hardware / f"{P1}.json.v1.bak").unlink()
    _write(hardware / f"{SENTINEL_HOST}.json.v1.bak", source)
    _write(hardware / "buildhost.json.v1.bak", {**source, "machine": "buildhost", "measurements": []})

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    kept = sorted((json.loads(path.read_text(encoding="utf-8")) for path in hardware.glob("*.json.v1.bak")), key=lambda data: len(data["measurements"]))
    assert len(kept) == 2 and [len(data["measurements"]) for data in kept] == [0, len(source["measurements"])]
    assert files_with(config.parent, SENTINEL_HOST) == [] and files_with(config.parent, "buildhost") == []


def test_a_backup_whose_copy_is_already_there_goes_after_a_stopped_run(tmp_path):
    config, hardware, source = _migrated(tmp_path)
    target = hardware / f"{P1}.json.v1.bak"
    assert json.loads(target.read_text(encoding="utf-8")) == _kept(source, neutral_name(P1, "unknown"))
    _write(hardware / f"{SENTINEL_HOST}.json.v1.bak", source)  # the source a stopped sweep left beside its copy
    before = target.read_bytes()

    swept = sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    assert not (hardware / f"{SENTINEL_HOST}.json.v1.bak").exists() and target.read_bytes() == before
    assert not any("could not be cleaned" in line for line in swept.notes)


def test_only_legacy_and_eight_hex_counts_as_a_neutral_backup_name(tmp_path):
    config, hardware, source = _migrated(tmp_path)
    (hardware / f"{P1}.json.v1.bak").unlink()
    _write(hardware / f"legacy-{SENTINEL_HOST}.json.v1.bak", source)

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    assert files_with(config.parent, SENTINEL_HOST) == [] and not list(hardware.glob(f"*{SENTINEL_HOST}*"))


# --- r2-5: a reference of a record that is no text -----------------------------------------------------------


@pytest.mark.parametrize("field, value", [("hf_file_digest", []), ("hf_repo", {"a": 1}), ("content_source", 7)])
def test_a_record_whose_reference_is_of_another_shape_stops_the_command(tmp_path, capsys, field, value):
    config = _folder(tmp_path / "f", {THIS_MACHINE: None}, writer=THIS_MACHINE)
    record = _leaky_record().model_dump(mode="json")
    record["package"] = {**record["package"], field: value}
    path = _write(config.parent / "state" / "measurements" / PROFILE_ID / "x.json", record)
    before = path.read_bytes()

    code = main(["render", "--config", str(config)], pointer_path=_pointer(tmp_path / "home"), probes=windows_probes(), now=NOW)

    assert code == 1 and FAILED in capsys.readouterr().err and path.read_bytes() == before
