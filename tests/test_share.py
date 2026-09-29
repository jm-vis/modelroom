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
    assert machine_list_row({}, _store("Z:\\profiles")) == WITH_SHARE_ROW
    assert machine_list_row({ENV_VAR: "Z:\\profiles"}, _store()) == WITH_SHARE_ROW


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
    assert machine_list_row({ENV_VAR: "Z:\\profiles"}, _store("D:\\other")) == WITH_SHARE_ROW


def test_the_step_shows_the_stored_path_and_set_by_environment_without_path():
    stand = guided_share._stand_lines({}, _store())[0]
    assert NONE_SET_YET in stand

    stand = guided_share._stand_lines({}, _store("Z:\\profiles"))[0]
    assert "Z:\\profiles" in stand

    stand = guided_share._stand_lines({ENV_VAR: "Z:\\profiles"}, _store("Z:\\profiles"))[0]
    assert SET_BY_ENV in stand and "Z:\\profiles" not in stand


def test_choose_is_grayed_when_the_environment_sets_the_folder():
    ungrayed = guided_share._setup_choices({}, _store())
    assert next(choice for choice in ungrayed if choice.value == "choose").disabled is None

    grayed = guided_share._setup_choices({ENV_VAR: "Z:\\profiles"}, _store())
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
    assert stored["tps_mean"] == original["tps_mean"]  # the older, first-imported file stayed


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
    sentinel = "\\\\sentinel-nas\\share\\sentinel-user"
    remote = _write_remote(tmp_path, sentinel)

    _code, lines, _asker = _share_run(tmp_path, {"share_setup": "use"})

    assert any(sentinel in line for line in lines)  # the step's own screen, only
    assert not files_with(_results(tmp_path), sentinel)
    assert json.loads(remote.read_text(encoding="utf-8"))["profile_share"] == sentinel  # JSON escapes the backslashes


def test_the_environment_path_never_appears(tmp_path: Path):
    sentinel = "\\\\sentinel-nas\\share\\sentinel-user"

    _code, lines, _asker = _share_run(tmp_path, {"share_setup": "use"}, environ={ENV_VAR: sentinel})

    assert not any(sentinel in line for line in lines)
    assert not files_with(_results(tmp_path), sentinel)
    remote = remote_path_for(_pointer(tmp_path))
    assert not remote.exists() or sentinel not in remote.read_text(encoding="utf-8")


def test_only_guided_share_reads_profile_share():
    import modelroom

    package_dir = Path(modelroom.__file__).parent
    hits = {path.name for path in package_dir.glob("*.py") if "profile_share" in path.read_text(encoding="utf-8")}

    assert hits == {"remote_store.py", "guided_share.py"}
