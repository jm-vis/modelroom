"""The sink of machine names, second round (FR-N, Codex diff review r1): what the sweep covers beyond
keys, profiles and records, and where it has to stop instead of writing.

Every test here failed on the code of the first round (`73fb12e`); the report names each.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import tomllib
from pathlib import Path

import pytest

import modelroom.state as state_module
from fixture_neutral import SENTINEL_HOST, THIS_MACHINE, files_with
from fixture_support import FIXTURES, PROFILE_ID, build_transport, guided_transport_mapping, offline_daemon, offline_servers, windows_probes
from modelroom import neutral_sweep, remote_store
from modelroom.cli import main
from modelroom.dialog import FileAsker
from modelroom.guided import run_guided
from modelroom.migrate import MigrationError, migrate
from modelroom.neutral import FAILED, NeutralError, is_neutral, old_key
from modelroom.neutral_sweep import sweep
from modelroom.state import acquire_lock, release_lock

from test_neutral import EXPORT, NOW, OTHER_ID, _clean, _folder, _leaky_record, _machines, _old_folder, _pointer, _profile, _snapshot, _store, _write

RUN_ANSWERS = {"results": "here", "machines": ["this-machine"], "machine_class": "laptop", "search": "qwen",
               "filter_owners": True, "select": ["unsloth/Qwen3.5-9B-GGUF"], "users": 1, "context": "8192"}


def _guided(results: Path, home: Path, answers: dict, asker=FileAsker, host: str = SENTINEL_HOST) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = run_guided(asker(answers), here=results, pointer_path=home, transport=build_transport(guided_transport_mapping()),
                      probes=windows_probes(host=host), daemon=offline_daemon(), servers=offline_servers(), now=NOW, out=lines.append)
    return code, lines


def _names(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in root.rglob("*") if SENTINEL_HOST in path.name]


# --- r1-1: the result documents ----------------------------------------------------------------------


def test_documents_that_name_a_machine_go_with_the_sweep(tmp_path, capsys):
    config = _old_folder(tmp_path / "old")
    docs = tmp_path / "old" / "docs"
    _write(docs / "models.md", f"## Ranking: {SENTINEL_HOST}\n")
    _write(docs / "models.json", {"machines": [{"machine": SENTINEL_HOST, "label": SENTINEL_HOST, "profile": None}]})

    assert main(["export-profile", "--config", str(config), "--profile", PROFILE_ID, "--out", str(tmp_path / "out.json")],
                pointer_path=_pointer(tmp_path / "home")) == 0

    assert not docs.joinpath("models.md").exists() and not docs.joinpath("models.json").exists()
    assert "the next render writes them" in capsys.readouterr().out
    _clean(tmp_path / "old")


def test_neutral_documents_stay_where_nothing_else_changes(tmp_path):
    config = _folder(tmp_path / "f", {THIS_MACHINE: PROFILE_ID}, writer=THIS_MACHINE)
    _profile(config.parent, PROFILE_ID, "laptop-3f9a0c21", "laptop")
    view = _write(config.parent / "docs" / "models.json", {"machines": [{"machine": THIS_MACHINE, "label": "laptop-3f9a0c21 (entered)", "profile": {"display_name": "laptop-3f9a0c21"}}]})

    assert sweep(config, NOW, "render", pointer_path=_pointer(tmp_path / "home")).notes == []
    assert view.is_file()


# --- r1-2: the whole configuration ---------------------------------------------------------------------


def test_a_results_folder_named_like_a_host_leaves_the_name_in_no_file(tmp_path):
    results = tmp_path / SENTINEL_HOST / "results"
    results.mkdir(parents=True)
    home = _pointer(tmp_path / "home")

    code, _lines = _guided(results, home, RUN_ANSWERS)
    out = tmp_path / "share" / "out.json"
    exported = main(["export-profile", "--config", str(results / "modelroom.toml"), "--out", str(out)], pointer_path=home, probes=windows_probes())

    assert (code, exported) == (0, 0)
    assert files_with(results, SENTINEL_HOST) == [] and files_with(out.parent, SENTINEL_HOST) == []
    assert tomllib.loads((results / "modelroom.toml").read_text(encoding="utf-8"))["paths"]["state"] == "state"


def test_a_folder_of_0_1_0_gets_its_own_paths_relative(tmp_path):
    folder = tmp_path / SENTINEL_HOST
    config = _folder(folder, {THIS_MACHINE: None}, writer=THIS_MACHINE)
    text = config.read_text(encoding="utf-8").replace('state = "state"', f"state = {json.dumps(str(folder / 'state'))}")
    config.write_bytes((text + f"\n[guided]\nresults = {json.dumps(str(folder))}\n").encode("utf-8"))

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    assert (raw["paths"]["state"], raw["guided"]["results"]) == ("state", ".")
    assert config.read_text(encoding="utf-8").startswith("# written by hand, kept by the sweep")


def test_a_key_another_run_writes_during_the_dialog_is_written_neutral(tmp_path):
    results, home = tmp_path / "results", _pointer(tmp_path / "home")
    results.mkdir()
    assert _guided(results, home, RUN_ANSWERS, host="laptop-host")[0] == 0

    class Meanwhile(FileAsker):
        def text(self, key, question, *args, **kwargs):
            if key == "search":  # another run of 0.1.0 adds its machine under its host name
                config = results / "modelroom.toml"
                config.write_text(config.read_text(encoding="utf-8") + f"\n[machines.{SENTINEL_HOST}]\nreserve_vram_gib = 1.0\nwriter = false\n", encoding="utf-8")
            return super().text(key, question, *args, **kwargs)

    _guided(results, home, {**RUN_ANSWERS, "results": "here", "machines": []}, asker=Meanwhile, host="laptop-host")

    assert SENTINEL_HOST not in (results / "modelroom.toml").read_text(encoding="utf-8")


# --- r1-3: files that cannot be cleaned -----------------------------------------------------------------


@pytest.mark.parametrize("content", [b"{", b"\xff\xfe{}", b"[]"], ids=["not-json", "not-utf-8", "not-an-object"])
@pytest.mark.parametrize("where", ["hardware", "measurements"])
def test_a_file_that_cannot_be_cleaned_stops_the_command_before_any_write(tmp_path, capsys, content, where):
    config = _old_folder(tmp_path / "old")
    broken = config.parent / "state" / where / (f"{OTHER_ID}.json" if where == "hardware" else f"{PROFILE_ID}/x.json")
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(content)
    before = {path: path.read_bytes() for path in config.parent.rglob("*") if path.is_file()}

    code = main(["render", "--config", str(config)], pointer_path=_pointer(tmp_path / "home"), probes=windows_probes(), now=NOW)

    assert code == 1 and FAILED in capsys.readouterr().err
    assert {path: path.read_bytes() for path in config.parent.rglob("*") if path.is_file() and path.name != "modelroom.lock"} == before


def test_a_record_that_does_not_validate_is_cleaned_field_by_field(tmp_path):
    config = _folder(tmp_path / "f", {THIS_MACHINE: None}, writer=THIS_MACHINE)
    record = {**_leaky_record().model_dump(mode="json"), "tps_mean": -1}  # breaks the model, keeps the names
    path = _write(config.parent / "state" / "measurements" / PROFILE_ID / "x.json", record)

    sweep(config, NOW, "render", pointer_path=_pointer(tmp_path / "home"))

    assert SENTINEL_HOST not in path.read_text(encoding="utf-8") and json.loads(path.read_text(encoding="utf-8"))["tps_mean"] == -1


def test_a_profile_whose_name_is_a_number_is_cleaned_not_a_crash(tmp_path):
    config = _folder(tmp_path / "f", {THIS_MACHINE: PROFILE_ID}, writer=THIS_MACHINE)
    _write(config.parent / "state" / "hardware" / f"{PROFILE_ID}.json", {**EXPORT["profile"], "display_name": 1234})

    sweep(config, NOW, "render", pointer_path=_pointer(tmp_path / "home"))

    assert json.loads((config.parent / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8"))["display_name"] == "computer-3f9a0c21"


# --- r1-4/5: migrate ------------------------------------------------------------------------------------


def _v1_folder(root: Path, host: str = SENTINEL_HOST) -> Path:
    root.mkdir(parents=True)
    text = (FIXTURES / "config_v1" / "config.toml").read_text(encoding="utf-8").replace("[machines.laptop]", f"[machines.{host}]")
    (root / "modelroom.toml").write_bytes(text.encode("utf-8"))
    legacy = json.loads((FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json").read_text(encoding="utf-8"))
    _write(root / "state" / "hardware" / f"{host}.json", {**legacy, "machine": host})
    return root / "modelroom.toml"


def _watch(monkeypatch, fail_after: int | None = None) -> list[str]:
    """Every write of the package, checked: never a machine name in a path or a content; optionally a
    stop (`OSError`) at the given write, as a crash would make it."""
    seen: list[str] = []

    def watched(real):
        def write(path, content, *args):
            text = content if isinstance(content, str) else json.dumps(content)
            assert SENTINEL_HOST not in text and SENTINEL_HOST not in Path(path).name, path
            if fail_after is not None and len(seen) >= fail_after:
                raise OSError("stopped here")
            seen.append(Path(path).name)
            return real(path, content, *args)

        return write

    import sys

    for module in [module for name, module in sys.modules.items() if name.startswith("modelroom") and module is not None]:
        for name in ("atomic_write_json", "atomic_write_text"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, watched(getattr(state_module, name)))
    return seen


def test_migrate_publishes_no_old_key_at_any_point(tmp_path, monkeypatch):
    config = _v1_folder(tmp_path / "old")
    seen = _watch(monkeypatch)

    migrate(config, NOW, fresh_id=lambda: "a1aaaaaaaaaaaaaa", pointer_path=_pointer(tmp_path / "home"))

    assert "a1aaaaaaaaaaaaaa.json.v1.bak" in seen and "modelroom.toml.v1.bak" in seen
    assert _names(config.parent) == [] and files_with(config.parent, SENTINEL_HOST) == []


def test_migrate_stopped_at_every_write_leaves_no_name_and_converges(tmp_path, monkeypatch):
    for stop in range(6):
        config = _v1_folder(tmp_path / f"run{stop}")
        with monkeypatch.context() as patch:
            _watch(patch, fail_after=stop)
            with pytest.raises(OSError):
                migrate(config, NOW, fresh_id=lambda: "a1aaaaaaaaaaaaaa", pointer_path=_pointer(tmp_path / "home"))
        written = [path for path in config.parent.rglob("*") if path.is_file() and path.name != f"{SENTINEL_HOST}.json"]
        assert all(SENTINEL_HOST not in path.read_text(encoding="utf-8", errors="replace") or path.name == "modelroom.toml" for path in written)
        migrate(config, NOW, fresh_id=lambda: "a1aaaaaaaaaaaaaa", pointer_path=_pointer(tmp_path / "home"))
        assert _names(config.parent) == [] and files_with(config.parent, SENTINEL_HOST) == [], stop


def test_two_profiles_a_stopped_run_may_have_left_are_a_conflict(tmp_path):
    config = _v1_folder(tmp_path / "old")
    migrate(config, NOW, fresh_id=lambda: "a1aaaaaaaaaaaaaa", pointer_path=_pointer(tmp_path / "home"))
    hardware = config.parent / "state" / "hardware"
    twin = json.loads((hardware / "a1aaaaaaaaaaaaaa.json").read_text(encoding="utf-8")) | {"profile_id": "a2aaaaaaaaaaaaaa"}
    _write(hardware / "a2aaaaaaaaaaaaaa.json", twin)
    shutil.copy(hardware / "a1aaaaaaaaaaaaaa.json.v1.bak", hardware / "other.json")  # the schema-1 file, back again

    with pytest.raises(MigrationError, match="more than one profile"):
        migrate(config, NOW, fresh_id=lambda: "a3aaaaaaaaaaaaaa", pointer_path=_pointer(tmp_path / "home"))


# --- r1-6: the lock -------------------------------------------------------------------------------------


def test_an_export_stops_at_a_held_lock_even_in_a_neutral_folder(tmp_path, capsys):
    config = _folder(tmp_path / "f", {THIS_MACHINE: PROFILE_ID}, writer=THIS_MACHINE)
    _profile(config.parent, PROFILE_ID, "laptop-3f9a0c21", "laptop")
    out = tmp_path / "out.json"
    handle = acquire_lock(config.parent / "state" / "modelroom.lock", "another process", NOW)
    try:
        code = main(["export-profile", "--config", str(config), "--profile", PROFILE_ID, "--out", str(out)], pointer_path=_pointer(tmp_path / "home"))
    finally:
        release_lock(handle)

    assert code == 1 and not out.exists()


def test_a_sweep_whose_state_folder_moved_while_it_took_the_lock_writes_nothing(tmp_path, monkeypatch):
    config = _old_folder(tmp_path / "old")
    real = neutral_sweep.acquire_lock

    def acquire_after_an_edit(*args, **kwargs):
        config.write_text(config.read_text(encoding="utf-8").replace('state = "state"', 'state = "state-b"'), encoding="utf-8")
        return real(*args, **kwargs)

    monkeypatch.setattr(neutral_sweep, "acquire_lock", acquire_after_an_edit)
    with pytest.raises(NeutralError):
        sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"))

    assert not (tmp_path / "old" / "state-b").exists() and SENTINEL_HOST in _machines(config)


# --- r1-7: the first key of this computer ------------------------------------------------------------------


def test_the_first_key_waits_for_another_process_making_it(tmp_path):
    path = tmp_path / ".modelroom" / "remote.json"
    handle = acquire_lock(path.with_name("remote.lock"), "another process", NOW)
    found: list[str] = []
    worker = threading.Thread(target=lambda: found.append(remote_store.this_machine(path)))
    worker.start()
    time.sleep(0.3)
    _write(path, {"schema_version": 1, "this_machine": "computer-0e0e0e0e"})
    release_lock(handle)
    worker.join(timeout=10)

    assert found == ["computer-0e0e0e0e"]


def test_a_remote_json_that_cannot_be_written_is_exit_3_with_a_fixed_sentence(tmp_path, capsys):
    config = _folder(tmp_path / "f", {SENTINEL_HOST: None}, writer=SENTINEL_HOST)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".modelroom").write_text("a file where the folder belongs", encoding="utf-8")

    code = main(["fetch", "--config", str(config), "--machine", SENTINEL_HOST], pointer_path=home / ".modelroom" / "guided.json",
                probes=windows_probes(host=SENTINEL_HOST), transport=build_transport({}), now=NOW)

    assert code == 3
    err = capsys.readouterr().err
    assert "could not be written" in err and str(home) not in err


# --- r1-8/9: the old key, the sentences, the screen -------------------------------------------------------


def test_the_key_of_0_1_0_keeps_every_dash(tmp_path):
    assert (old_key("Node  01"), old_key("Büro-PC"), old_key("--")) == ("node--01", None, None)  # 0.1.0 had no key for the last two
    config = _folder(tmp_path / "f", {"node--01": None}, writer="node--01")

    assert sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"), hostname="Node  01").renames == {"node--01": THIS_MACHINE}


def test_a_broken_remote_json_is_said_without_its_path_or_its_values(tmp_path, capsys):
    config = _folder(tmp_path / "f", {THIS_MACHINE: None}, writer=THIS_MACHINE)
    home = _pointer(tmp_path / "home")
    _write(home.parent / "remote.json", {"schema_version": 1, "this_machine": SENTINEL_HOST})

    assert main(["render", "--config", str(config)], pointer_path=home, probes=windows_probes(), now=NOW) == 3

    err = capsys.readouterr().err
    assert SENTINEL_HOST not in err and str(tmp_path) not in err and "remote.json" in err


def test_the_name_typed_for_a_machine_entered_by_hand_is_not_on_the_screen(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    answers = {**RUN_ANSWERS, "machines": ["enter"], "entered_name": SENTINEL_HOST, "entered_class": "server",
               "entered_ram": 64, "entered_gpu": "none"}

    code, lines = _guided(results, _pointer(tmp_path / "home"), answers, host="laptop-host")

    assert code == 0 and not any(SENTINEL_HOST in line for line in lines)


# --- r1-10: other ways to write a key ----------------------------------------------------------------------


def test_dotted_and_inline_keys_are_renamed_and_the_comments_stay(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    config = folder / "modelroom.toml"
    config.write_text(
        "# kept\nschema_version = 3\npackagers = []\npublishers = []\n"
        f"machines.{SENTINEL_HOST}.reserve_vram_gib = 1.0\nmachines.{SENTINEL_HOST}.writer = true\n"
        f'machines."{SENTINEL_HOST}-2" = {{ reserve_vram_gib = 1.0, writer = false }}\n\n'
        '[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n',
        encoding="utf-8",
    )

    sweep(config, NOW, "fetch", pointer_path=_pointer(tmp_path / "home"), hostname=SENTINEL_HOST)

    text = config.read_text(encoding="utf-8")
    assert text.startswith("# kept\n") and SENTINEL_HOST not in text
    assert all(is_neutral(key) for key in tomllib.loads(text)["machines"])
