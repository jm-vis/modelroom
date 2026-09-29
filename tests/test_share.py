"""Tests for `modelroom/guided_share.py`: the shared folder of step 1 (FR-S, GF question 7 (c)).

Two layers: pure-function tests of the row, the stand and the single choice (no file system at
all), and end-to-end guided runs through `run_guided`/`main` with a real drop folder under
`tmp_path`. Export files are built from `tests/fixtures/export_v1.json`, the same fixture
`test_importer.py` uses, so a profile and its measurements are always a valid export.
`~/.modelroom/remote.json` is written directly where a test needs a folder already stored.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelroom import guided_share
from modelroom.cli import main
from modelroom.dialog import FileAsker
from modelroom.guided import run_guided
from modelroom.guided_share import (
    ENV_VAR,
    FORM_PROBLEM,
    KEPT_NOTE,
    NO_SHARE_ROW,
    NONE_SET_YET,
    REMOVED_NOTE,
    SET_BY_ENV,
    UNREACHABLE_NOTE,
    WITH_SHARE_ROW,
    WRITE_PROBLEM,
    env_value,
    machine_list_row,
    share_folder,
)
from modelroom.remote_store import REMOTE_SCHEMA_VERSION, RemoteStore, read_remote, remote_path_for
from modelroom.screen import Screen
from modelroom.state import atomic_write_json

from fixture_neutral import THIS_MACHINE, files_with
from fixture_room import fixed_room
from fixture_support import build_transport, guided_transport_mapping, offline_daemon, offline_servers, windows_probes
from test_guided import FULL_ANSWERS, RUN1, _WatchingAsker, _pointer, _results

FIXTURES = Path(__file__).parent / "fixtures"
EXPORT = json.loads((FIXTURES / "export_v1.json").read_text(encoding="utf-8"))
PROFILE_ID = EXPORT["profile"]["profile_id"]
OTHER_ID = "b1b2b3b4b5b6b7b8"
# A form-valid stored path for tests that only need *some* absolute path, not a particular shape
# (Codex r1-7): native to whatever platform the suite runs on, so `RemoteStore`'s own validator
# (real `Path(...).is_absolute()`, unlike the injected `windows_probes()` fixtures) accepts it
# everywhere. UNC paths and drive letters are their own, Windows-only tests below.
EXAMPLE_SHARE = "Z:\\profiles" if sys.platform == "win32" else "/mnt/profiles"
ANOTHER_SHARE = "D:\\other" if sys.platform == "win32" else "/mnt/other"
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="UNC paths are a Windows path shape")


@pytest.fixture(autouse=True)
def _results_folder(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()


# --- helpers ---------------------------------------------------------------------------------


def _export(profile_updates: dict | None = None, measurements: list[dict] | None = None) -> dict:
    """A copy of `EXPORT`, with the profile changed and, optionally, its own measurements; every
    measurement is re-keyed to the (possibly new) profile id (`tests/test_importer.py`'s own shape)."""
    data = copy.deepcopy(EXPORT)
    if profile_updates:
        data["profile"].update(profile_updates)
    if measurements is not None:
        data["measurements"] = copy.deepcopy(measurements)
    for measurement in data["measurements"]:
        measurement["profile_id"] = data["profile"]["profile_id"]
    return data


def _drop_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "share"
    folder.mkdir(exist_ok=True)
    return folder


def _write(folder: Path, name: str, data: dict) -> Path:
    path = folder / name
    atomic_write_json(path, data)
    return path


def _store(profile_share: str | None = None) -> RemoteStore:
    return RemoteStore(schema_version=REMOTE_SCHEMA_VERSION, profile_share=profile_share)


def _write_remote(tmp_path: Path, profile_share: str | None) -> Path:
    path = remote_path_for(_pointer(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"schema_version": REMOTE_SCHEMA_VERSION, "this_machine": THIS_MACHINE, "ssh": {}, "profile_share": profile_share})
    return path


def _value_has(value: object, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(_value_has(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_value_has(item, needle) for item in value)
    return False


def _decoded_leak_search(folder: Path, needle: str) -> list[str]:
    """Every file under `folder` that carries `needle`, relative to it: a `.json` file is checked
    on its *decoded* string values (Codex r1-6 -- a path with backslashes reads differently once
    JSON has escaped them, so a raw byte search alone would miss a real leak there), everything
    else -- `docs/`, backups, the rendered document -- as raw bytes."""
    hits: list[str] = []
    if not folder.exists():
        return hits
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        found = False
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                data = None
            found = data is not None and _value_has(data, needle)
        if not found and needle.encode("utf-8") in path.read_bytes():
            found = True
        if found:
            hits.append(str(path.relative_to(folder)))
    return hits


def _share_run(tmp_path: Path, share_answers: dict, *, environ: dict | None = None, machines=("share",), now=RUN1):
    """One guided run whose machine list picks `share`; returns the exit code, every printed line
    and the asker (so a test can read the choices a list was built with, `_WatchingAsker`)."""
    answers = {**FULL_ANSWERS, "machines": list(machines), **share_answers}
    lines: list[str] = []
    asker = _WatchingAsker(answers, lines)
    code = run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        servers=offline_servers(),
        now=now,
        out=lines.append,
        room=fixed_room(environ=environ or {}),
    )
    return code, lines, asker


def _share_main(tmp_path: Path, answers: dict, *, environ: dict | None = None) -> int:
    """The same run through `main`, for the tests that need the CLI's own exit-code mapping
    (`AnswerInvalidError`/`AnswerMissingError` -> exit `2`, the shape `test_guided_folder.py` uses)."""
    full = {**FULL_ANSWERS, "machines": ["share"], **answers}
    return main(
        [],
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        pointer_path=_pointer(tmp_path),
        now=RUN1,
        asker=FileAsker(full),
        here=_results(tmp_path),
        out=lambda line: None,
        daemon=offline_daemon(),
        servers=offline_servers(),
        room=fixed_room(environ=environ or {}),
    )


class _TerminalAsker:
    """Not a `FileAsker`: `_ask_folder` retries with it instead of raising, the terminal shape."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.calls = 0

    def text(self, key: str, question: str, default: str = "") -> str:
        self.calls += 1
        return self._answers.pop(0)


def _stub_run() -> tuple[SimpleNamespace, list[str]]:
    lines: list[str] = []
    return SimpleNamespace(asker=None, screen=Screen(lines.append, colored=False)), lines


# --- the row, the stand, the single choice (pure) ---------------------------------------------


def test_the_row_is_always_there_and_says_set_up_without_a_folder():
    assert machine_list_row({}, _store()) == NO_SHARE_ROW


def test_the_row_says_profiles_from_the_shared_folder_with_a_folder_stored_or_in_the_environment():
    assert machine_list_row({}, _store(EXAMPLE_SHARE)) == WITH_SHARE_ROW
    assert machine_list_row({ENV_VAR: EXAMPLE_SHARE}, _store()) == WITH_SHARE_ROW


def test_empty_share_env_counts_as_unset():
    assert env_value({ENV_VAR: ""}) is None
    assert share_folder({ENV_VAR: ""}, _store()) is None
    assert machine_list_row({ENV_VAR: ""}, _store()) == NO_SHARE_ROW


def test_the_row_never_touches_the_folder(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("the folder was touched")

    monkeypatch.setattr(Path, "exists", _boom)
    monkeypatch.setattr(Path, "is_dir", _boom)
    monkeypatch.setattr("builtins.open", _boom)

    assert machine_list_row({}, _store()) == NO_SHARE_ROW
    assert machine_list_row({ENV_VAR: EXAMPLE_SHARE}, _store(ANOTHER_SHARE)) == WITH_SHARE_ROW


def test_the_step_shows_the_stored_path_and_set_by_environment_without_path():
    stand = guided_share._stand_lines({}, _store())[0]
    assert NONE_SET_YET in stand

    stand = guided_share._stand_lines({}, _store(EXAMPLE_SHARE))[0]
    assert EXAMPLE_SHARE in stand

    stand = guided_share._stand_lines({ENV_VAR: EXAMPLE_SHARE}, _store(EXAMPLE_SHARE))[0]
    assert SET_BY_ENV in stand and EXAMPLE_SHARE not in stand


def test_choose_is_grayed_when_the_environment_sets_the_folder():
    ungrayed = guided_share._setup_choices({}, _store())
    assert next(choice for choice in ungrayed if choice.value == "choose").disabled is None

    grayed = guided_share._setup_choices({ENV_VAR: EXAMPLE_SHARE}, _store())
    choose = next(choice for choice in grayed if choice.value == "choose")
    assert choose.disabled == SET_BY_ENV
    assert not any(choice.value == "use" for choice in ungrayed)
    assert any(choice.value == "use" and choice.checked for choice in grayed)


def test_a_relative_or_device_path_is_refused():
    run, lines = _stub_run()
    run.asker = _TerminalAsker(["profiles", "\\\\.\\pipe\\x", "\\\\?\\C:\\x", ""])

    assert guided_share._ask_folder(run) is None
    assert run.asker.calls == 4
    assert sum(FORM_PROBLEM in line for line in lines) == 3


def test_an_empty_input_changes_nothing():
    run, lines = _stub_run()
    run.asker = _TerminalAsker([""])

    assert guided_share._ask_folder(run) is None
    assert run.asker.calls == 1


def test_a_failed_write_probe_is_the_fixed_sentence_and_stores_nothing(tmp_path: Path):
    missing = tmp_path / "missing" / "share"  # no parent: the write itself fails
    run, lines = _stub_run()
    run.asker = _TerminalAsker([str(missing), ""])

    assert guided_share._ask_folder(run) is None
    assert any(WRITE_PROBLEM in line for line in lines)


def test_a_probe_file_left_behind_is_a_note_and_the_folder_counts(tmp_path: Path, monkeypatch):
    folder = tmp_path / "share"
    folder.mkdir()
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(OSError("busy")))
    run, lines = _stub_run()
    run.asker = _TerminalAsker([str(folder)])

    assert guided_share._ask_folder(run) == str(folder)
    assert any("delete it by hand" in line for line in lines)


# --- the "Shared folder" step, end to end -------------------------------------------------------


def test_choose_writes_the_folder_after_the_write_probe_as_typed_without_resolve(tmp_path: Path):
    share = tmp_path / "share"
    share.mkdir()
    typed = str(share / ".." / "share")  # `resolve()` would collapse the ".."; plain `Path` does not

    _code, lines, _asker = _share_run(tmp_path, {"share_setup": "choose", "share_folder": typed})

    store = read_remote(remote_path_for(_pointer(tmp_path)))
    assert store.profile_share == str(Path(typed).expanduser())
    assert ".." in store.profile_share
    assert any(KEPT_NOTE in line for line in lines)
    assert list(share.iterdir()) == []  # the probe file cleaned up after itself


@WINDOWS_ONLY
def test_a_unc_path_is_accepted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(guided_share, "_write_probe", lambda folder: (True, None))
    unc = "\\\\sentinel-nas\\share\\profiles"

    _code, _lines, _asker = _share_run(tmp_path, {"share_setup": "choose", "share_folder": unc})

    store = read_remote(remote_path_for(_pointer(tmp_path)))
    assert store.profile_share == unc


def test_none_removes_the_stored_folder(tmp_path: Path):
    remote = _write_remote(tmp_path, str(tmp_path / "old-share"))

    _code, lines, _asker = _share_run(tmp_path, {"share_setup": "none"})

    store = read_remote(remote)
    assert store.profile_share is None
    assert store.this_machine == THIS_MACHINE  # the write touches only `profile_share`
    assert any(REMOVED_NOTE in line for line in lines)


def test_answers_without_share_setup_are_exit_2(tmp_path: Path):
    assert _share_main(tmp_path, {}) == 2


def test_answers_choose_with_a_bad_folder_are_exit_2(tmp_path: Path):
    assert _share_main(tmp_path, {"share_setup": "choose", "share_folder": "relative"}) == 2
    store = read_remote(remote_path_for(_pointer(tmp_path)))
    assert store.profile_share is None


def test_the_environment_wins_and_leaves_the_stored_folder_unchanged(tmp_path: Path):
    stored_folder = tmp_path / "stored-share"
    stored_folder.mkdir()
    env_folder = _drop_folder(tmp_path)
    _write(env_folder, "one.json", _export({"recorded_at": "2026-09-20T08:00:00Z"}))
    remote = _write_remote(tmp_path, str(stored_folder))

    _code, _lines, asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(env_folder)}
    )

    assert len(asker.choices["share_files"]) == 1  # read from the environment's folder, not the stored one
    assert read_remote(remote).profile_share == str(stored_folder)  # untouched


def test_an_unreachable_share_is_one_fixed_line_and_the_run_goes_on(tmp_path: Path):
    missing = tmp_path / "does-not-exist"

    code, lines, _asker = _share_run(tmp_path, {"share_setup": "use"}, environ={ENV_VAR: str(missing)})

    assert sum(UNREACHABLE_NOTE in line for line in lines) == 1
    assert code in (0, 1)


def test_bad_files_are_one_note_with_a_count(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    (folder / "broken1.json").write_bytes(b"{not json")
    (folder / "broken2.json").write_bytes(b"{}")
    (folder / "broken3.json").write_bytes(json.dumps({"schema_version": 1, "profile": {}, "measurements": []}).encode("utf-8"))

    _code, lines, _asker = _share_run(tmp_path, {"share_setup": "use"}, environ={ENV_VAR: str(folder)})

    assert any("3 files in the shared folder do not read as a profile export" in line for line in lines)


def test_the_limits_skip_a_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(guided_share, "MAX_FILES", 1)
    folder = _drop_folder(tmp_path)
    _write(folder, "a_first.json", _export({"recorded_at": "2026-09-20T08:00:00Z"}))
    _write(
        folder, "z_second.json",
        _export({"profile_id": OTHER_ID, "display_name": "server-99990000", "recorded_at": "2026-09-21T08:00:00Z"}, measurements=[]),
    )

    _code, lines, asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert len(asker.choices["share_files"]) == 1  # the second file, beyond the limit, was never read
    assert not any("do not read as a profile export" in line for line in lines)  # not even counted as bad


def test_one_row_per_profile_id_described_by_the_newest_file(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    _write(folder, "a.json", _export({"recorded_at": "2026-09-20T08:00:00Z", "display_name": "server-11112222"}))
    _write(folder, "b.json", _export({"recorded_at": "2026-09-25T08:00:00Z", "display_name": "server-33334444"}))

    _code, _lines, asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    rows = asker.choices["share_files"]
    assert len(rows) == 1
    assert "server-33334444" in rows[0].label  # the newer file's own name, not the older one's


def test_file_names_never_appear(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    _write(folder, "sentinel-host-export.json", _export({"recorded_at": "2026-09-20T08:00:00Z"}))

    _code, lines, _asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert not files_with(_results(tmp_path), "sentinel-host-export")
    assert not any("sentinel-host-export" in line for line in lines)


def test_a_chosen_file_is_imported_with_a_neutral_name(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    _write(folder, "one.json", _export({"display_name": "server-77778888", "recorded_at": "2026-09-20T08:00:00Z"}))

    _share_run(tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)})

    stored = json.loads((_results(tmp_path) / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8"))
    assert stored["display_name"] == "server-77778888"


def test_every_export_of_a_chosen_profile_is_imported_oldest_first_and_the_measurements_add_up(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    older = _export({"recorded_at": "2026-09-20T08:00:00Z"})  # both measurements of the fixture
    newer = _export({"recorded_at": "2026-09-25T08:00:00Z"}, measurements=[EXPORT["measurements"][0]])

    _write(folder, "older.json", older)
    _write(folder, "newer.json", newer)
    code, _lines, _asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert code == 0
    stored = json.loads((_results(tmp_path) / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8"))
    assert stored["recorded_at"].startswith("2026-09-25")  # the younger profile won
    measurements = {path.stem for path in (_results(tmp_path) / "state" / "measurements" / PROFILE_ID).glob("*.json")}
    assert measurements == {m["measurement_id"] for m in EXPORT["measurements"]}  # the older-only one was added too


def test_a_younger_profile_wins_whatever_the_order_of_the_files(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    # File names sort the opposite way of `recorded_at`: proves the import order is by time, not name.
    _write(folder, "a_newest.json", _export({"recorded_at": "2026-09-25T08:00:00Z", "display_name": "server-aaaa1111"}))
    _write(folder, "z_oldest.json", _export({"recorded_at": "2026-09-20T08:00:00Z", "display_name": "server-zzzz2222"}))

    _share_run(tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)})

    stored = json.loads((_results(tmp_path) / "state" / "hardware" / f"{PROFILE_ID}.json").read_text(encoding="utf-8"))
    assert stored["display_name"] == "server-aaaa1111"


def test_a_contradicting_measurement_id_across_two_files_fails_only_the_second_file(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    measurement_id = EXPORT["measurements"][0]["measurement_id"]
    original = copy.deepcopy(EXPORT["measurements"][0])
    changed = copy.deepcopy(EXPORT["measurements"][0])
    changed["load_state"]["gpu_utilization_percent"] = 5.0  # differs from the original, still a valid record
    _write(folder, "older.json", _export({"recorded_at": "2026-09-20T08:00:00Z"}, measurements=[original]))
    _write(folder, "newer.json", _export({"recorded_at": "2026-09-25T08:00:00Z"}, measurements=[changed]))

    code, lines, _asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert code == 1
    assert any("contradicts one imported before" in line for line in lines)
    stored = json.loads(
        (_results(tmp_path) / "state" / "measurements" / PROFILE_ID / f"{measurement_id}.json").read_text(encoding="utf-8")
    )
    # The field that actually differs between the two files (Codex r1-6): `tps_mean` is identical
    # in both, so asserting it would pass even if the wrong file had won.
    assert stored["load_state"]["gpu_utilization_percent"] == original["load_state"]["gpu_utilization_percent"]


def test_one_failing_file_does_not_stop_the_others(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    original = copy.deepcopy(EXPORT["measurements"][0])
    changed = copy.deepcopy(EXPORT["measurements"][0])
    changed["load_state"]["gpu_utilization_percent"] = 5.0  # differs from the original, still a valid record
    _write(folder, "a1.json", _export({"recorded_at": "2026-09-20T08:00:00Z"}, measurements=[original]))
    _write(folder, "a2.json", _export({"recorded_at": "2026-09-25T08:00:00Z"}, measurements=[changed]))
    _write(
        folder, "b1.json",
        _export({"profile_id": OTHER_ID, "display_name": "server-55556666", "recorded_at": "2026-09-21T08:00:00Z"}, measurements=[]),
    )

    code, lines, _asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID, OTHER_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert code == 1
    assert any("contradicts one imported before" in line for line in lines)
    assert (_results(tmp_path) / "state" / "hardware" / f"{OTHER_ID}.json").is_file()


def test_an_answer_naming_an_unknown_id_is_exit_2(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    _write(folder, "one.json", _export({"recorded_at": "2026-09-20T08:00:00Z"}))

    assert _share_main(
        tmp_path, {"share_setup": "use", "share_files": [OTHER_ID]}, environ={ENV_VAR: str(folder)}
    ) == 2


def test_import_errors_never_carry_the_path(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    original = copy.deepcopy(EXPORT["measurements"][0])
    changed = copy.deepcopy(EXPORT["measurements"][0])
    changed["load_state"]["gpu_utilization_percent"] = 5.0  # differs from the original, still a valid record
    _write(folder, "older.json", _export({"recorded_at": "2026-09-20T08:00:00Z"}, measurements=[original]))
    _write(folder, "newer.json", _export({"recorded_at": "2026-09-25T08:00:00Z"}, measurements=[changed]))

    _code, lines, _asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert not any(str(folder) in line for line in lines)
    assert not any("older.json" in line or "newer.json" in line for line in lines)


# --- the leak proofs (FR-S's own "Warum der Pfad nicht hinausgeht") -----------------------------


def test_the_stored_share_path_reaches_no_file_but_remote_json(tmp_path: Path):
    """Codex r1-6: a real, readable shared folder (its own name carries the sentinel, so the path
    is genuine, not a string nobody ever opens), a successful import, then `render` and
    `export-profile` on the same results folder -- the sentinel is allowed in exactly one place on
    screen (the stand line) and nowhere any of the three commands write."""
    sentinel = "sentinel-nas-share-sentinel-user"
    folder = tmp_path / sentinel
    folder.mkdir()
    _write(folder, "one.json", _export({"display_name": "server-11112222", "recorded_at": "2026-09-20T08:00:00Z"}))
    remote = _write_remote(tmp_path, str(folder))

    code, lines, _asker = _share_run(tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]})

    assert code == 0
    assert (_results(tmp_path) / "state" / "hardware" / f"{PROFILE_ID}.json").is_file()  # the import really ran
    carrying = [line for line in lines if sentinel in line]
    assert carrying and all(line.strip().startswith("shared folder") for line in carrying)

    config_path = _results(tmp_path) / "modelroom.toml"
    assert main(["render", "--config", str(config_path)], pointer_path=_pointer(tmp_path), now=RUN1) == 0
    export_path = tmp_path / "out.json"
    assert main(
        ["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(export_path)],
        pointer_path=_pointer(tmp_path),
    ) == 0

    assert _decoded_leak_search(_results(tmp_path), sentinel) == []
    assert _decoded_leak_search(folder, sentinel) == []  # the shared folder itself (Codex r2-2)
    assert sentinel not in export_path.read_text(encoding="utf-8")
    assert json.loads(remote.read_text(encoding="utf-8"))["profile_share"] == str(folder)


def test_the_environment_path_never_appears(tmp_path: Path):
    """Same as above, from `MODELROOM_PROFILE_SHARE`: no screen line, no stored value, ever."""
    sentinel = "sentinel-nas-share-sentinel-user"
    folder = tmp_path / sentinel
    folder.mkdir()
    _write(folder, "one.json", _export({"display_name": "server-22223333", "recorded_at": "2026-09-20T08:00:00Z"}))

    code, lines, _asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert code == 0
    assert (_results(tmp_path) / "state" / "hardware" / f"{PROFILE_ID}.json").is_file()
    assert not any(sentinel in line for line in lines)

    config_path = _results(tmp_path) / "modelroom.toml"
    assert main(["render", "--config", str(config_path)], pointer_path=_pointer(tmp_path), now=RUN1) == 0
    export_path = tmp_path / "out.json"
    assert main(
        ["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(export_path)],
        pointer_path=_pointer(tmp_path),
    ) == 0

    assert _decoded_leak_search(_results(tmp_path), sentinel) == []
    assert _decoded_leak_search(folder, sentinel) == []  # the shared folder itself (Codex r2-2)
    assert sentinel not in export_path.read_text(encoding="utf-8")
    remote = remote_path_for(_pointer(tmp_path))
    assert not remote.exists() or sentinel not in remote.read_text(encoding="utf-8")


def test_a_non_neutral_display_name_never_reaches_the_list_the_answer_or_the_import(tmp_path: Path):
    """Codex r1-6, the "Rechnernamen" gap: a valid export whose source forgot to neutralize its
    `display_name` -- the listing, the picked-answer line and the written profile all carry the
    sink's own fallback name (`neutral.display_name_for`), never the raw one. Codex r2-2: checked
    against the whole results folder, not one profile field, and after a subsequent export."""
    name = "sentinel-host"
    folder = _drop_folder(tmp_path)
    _write(folder, "one.json", _export({"display_name": name, "recorded_at": "2026-09-20T08:00:00Z"}))

    code, lines, asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert code == 0
    assert not any(name in line for line in lines)
    rows = asker.choices["share_files"]
    assert name not in rows[0].label
    assert _decoded_leak_search(_results(tmp_path), name) == []

    config_path = _results(tmp_path) / "modelroom.toml"
    export_path = tmp_path / "out.json"
    assert main(
        ["export-profile", "--config", str(config_path), "--profile", PROFILE_ID, "--out", str(export_path)],
        pointer_path=_pointer(tmp_path),
    ) == 0
    assert name not in export_path.read_text(encoding="utf-8")


def test_a_non_neutral_display_name_never_reaches_an_error_sentence(tmp_path: Path):
    folder = _drop_folder(tmp_path)
    original = copy.deepcopy(EXPORT["measurements"][0])
    changed = copy.deepcopy(EXPORT["measurements"][0])
    changed["load_state"]["gpu_utilization_percent"] = 5.0
    _write(
        folder, "older.json",
        _export({"display_name": "sentinel-host", "recorded_at": "2026-09-20T08:00:00Z"}, measurements=[original]),
    )
    _write(
        folder, "newer.json",
        _export({"display_name": "sentinel-host", "recorded_at": "2026-09-25T08:00:00Z"}, measurements=[changed]),
    )

    _code, lines, _asker = _share_run(
        tmp_path, {"share_setup": "use", "share_files": [PROFILE_ID]}, environ={ENV_VAR: str(folder)}
    )

    assert any("contradicts one imported before" in line for line in lines)
    assert not any("sentinel-host" in line for line in lines)


def test_none_with_the_environment_set_skips_this_run_and_keeps_the_stored_folder(tmp_path: Path):
    """Codex r1-2: while the environment wins, `none` must not clear a stored folder -- only this
    run skips it."""
    remote = _write_remote(tmp_path, EXAMPLE_SHARE)

    _code, lines, _asker = _share_run(tmp_path, {"share_setup": "none"}, environ={ENV_VAR: ANOTHER_SHARE})

    assert read_remote(remote).profile_share == EXAMPLE_SHARE
    assert any(guided_share.NOT_THIS_RUN_NOTE in line for line in lines)
    assert not any(guided_share.REMOVED_NOTE in line for line in lines)


def test_a_write_failure_after_create_is_a_write_problem_and_cleans_up(tmp_path: Path, monkeypatch):
    """Codex r1-3: writing (or closing) fails after the exclusive create already succeeded -- the
    probe file is still removed, not left behind."""
    import builtins

    folder = tmp_path / "share"
    folder.mkdir()
    real_open = builtins.open

    def flaky_open(path, mode="r", *args, **kwargs):
        if mode == "xb" and Path(path).parent == folder:
            handle = real_open(path, mode, *args, **kwargs)

            class _Flaky:
                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    handle.close()
                    return False

                def write(self, _data):
                    raise OSError("disk full")

            return _Flaky()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky_open)

    ok, note = guided_share._write_probe(folder)

    assert (ok, note) == (False, WRITE_PROBLEM)
    assert list(folder.iterdir()) == []  # cleaned up despite the write failure


def test_a_write_failure_and_a_failed_cleanup_names_the_leftover_file(tmp_path: Path, monkeypatch):
    """Codex r2-1: writing fails **and** the cleanup that follows fails too -- the plain
    `WRITE_PROBLEM` sentence alone would drop the leftover file silently; its own (neutral) name
    must be named so the leftover can be found and removed by hand."""
    import builtins

    folder = tmp_path / "share"
    folder.mkdir()
    real_open = builtins.open

    def flaky_open(path, mode="r", *args, **kwargs):
        if mode == "xb" and Path(path).parent == folder:
            handle = real_open(path, mode, *args, **kwargs)

            class _Flaky:
                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    handle.close()
                    return False

                def write(self, _data):
                    raise OSError("disk full")

            return _Flaky()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky_open)
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(OSError("busy")))
    fixed_token = "deadbeef"

    ok, note = guided_share._write_probe(folder, token=lambda: fixed_token)

    assert ok is False
    assert WRITE_PROBLEM in note
    assert f"{guided_share.PROBE_PREFIX}{fixed_token}.tmp" in note


def test_a_failed_create_never_deletes_an_existing_file(tmp_path: Path):
    """Codex r1-3: `open(..., "xb")` itself failing (a name collision) must never delete whatever
    is already at that path."""
    folder = tmp_path / "share"
    folder.mkdir()
    fixed_token = "cafebabe"
    decoy = folder / f"{guided_share.PROBE_PREFIX}{fixed_token}.tmp"
    decoy.write_bytes(b"do not touch")

    ok, note = guided_share._write_probe(folder, token=lambda: fixed_token)

    assert (ok, note) == (False, WRITE_PROBLEM)
    assert decoy.read_bytes() == b"do not touch"


def test_the_machine_list_choice_carries_the_row_label(tmp_path: Path):
    """Codex r1-5: the label is shown once, as the checkbox's own `Choice` -- not a second time as
    an interactive note beside it."""
    _code, _lines, asker = _share_run(tmp_path, {"share_setup": "none"})

    share_choice = next(choice for choice in asker.choices["machines"] if choice.value == "share")
    assert share_choice.label == NO_SHARE_ROW


def test_a_plain_run_never_prints_the_share_row_as_a_note(tmp_path: Path):
    """Codex r1-5: the row's label used to be printed as a `run.screen.note(...)` right beside the
    checkbox, unconditionally -- even a run that never picks `share` showed it. A run that only
    picks `this-machine` must never print it at all now."""
    _code, lines, _asker = _share_run(tmp_path, {}, machines=("this-machine",))

    assert not any(NO_SHARE_ROW in line for line in lines)


def test_only_remote_store_and_guided_share_mention_profile_share():
    """Codex r2-3: the package-wide check from round 1, kept alongside the sharper AST one below --
    no other module of `modelroom` names `profile_share` at all, not even in a comment."""
    import modelroom

    package_dir = Path(modelroom.__file__).parent
    hits = {path.name for path in package_dir.glob("*.py") if "profile_share" in path.read_text(encoding="utf-8")}

    assert hits == {"remote_store.py", "guided_share.py"}


def test_only_share_folder_reads_profile_share_as_an_attribute():
    """Codex r1-4: a grep on the module name is not enough -- `store.profile_share` must be an
    attribute access inside `share_folder` and nowhere else in `guided_share.py`, function by
    function."""
    import ast

    source = Path(guided_share.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.hits: list[str] = []
            self._stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr == "profile_share":
                self.hits.append(self._stack[-1] if self._stack else "<module>")
            self.generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)

    assert visitor.hits == ["share_folder"]
