"""Tests for `search_apply.write_configuration`: a change another run wrote in between is not lost.

The guided mode reads the configuration, computes the new one and writes it back under
`modelroom.lock`. Until 2026-09-25 a second run that wrote the file between that read and that
write was simply overwritten. The writer now compares, under the lock, the file with the text the
change was computed from, and refuses when they differ (CONTRACTS.md, "Configuration").
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom import search_apply
from modelroom.config import ConfigError, Configuration, config_from_text, load_config
from modelroom.search_apply import ConfigChangedError, write_configuration
from modelroom.state import LockHeldError, acquire_lock, release_lock

from fixture_support import qwen35_example_config_dict

NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
CHANGED = "changed by another run since it was read"


def _first(tmp_path: Path) -> tuple[Path, str]:
    """A configuration file as a first run leaves it, and the text a second read of it gives."""
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data.update({"schema_version": 2, "families": [], "publishers": []})
    path = tmp_path / "modelroom.toml"
    write_configuration(path, Configuration.from_dict(data), now=NOW, expected_text=None)
    return path, path.read_text(encoding="utf-8")


def _with_context(config: Configuration, context: int) -> Configuration:
    return config.model_copy(update={"guided": config.guided.model_copy(update={"context": context})})


def _with_publisher(config: Configuration, publisher: str) -> Configuration:
    data = config.model_dump(mode="json")
    data["publishers"] = [*data["publishers"], publisher]
    return Configuration.from_dict(data)


def test_a_change_another_run_wrote_between_read_and_write_is_not_lost(tmp_path: Path):
    path, text = _first(tmp_path)
    mine = _with_context(config_from_text(text, path), 4096)
    # The second writer read the same text and wrote first.
    other = _with_publisher(config_from_text(text, path), "acme")
    write_configuration(path, other, now=NOW, expected_text=text)

    with pytest.raises(ConfigChangedError, match=CHANGED):
        write_configuration(path, mine, now=NOW, expected_text=text)

    stored = load_config(path)
    assert stored.publishers == ["acme"]
    assert stored.guided.context is None


def test_a_file_that_is_still_what_was_read_is_written(tmp_path: Path):
    path, text = _first(tmp_path)

    write_configuration(path, _with_context(config_from_text(text, path), 4096), now=NOW, expected_text=text)

    assert load_config(path).guided.context == 4096


def test_a_file_that_appeared_since_the_read_is_the_same_conflict(tmp_path: Path):
    """Two runs that create the configuration at once: the second one must not write over the first."""
    path, text = _first(tmp_path)

    with pytest.raises(ConfigChangedError, match=CHANGED):
        write_configuration(path, _with_context(config_from_text(text, path), 4096), now=NOW, expected_text=None)

    assert path.read_text(encoding="utf-8") == text


def test_a_file_that_was_removed_since_the_read_is_a_conflict(tmp_path: Path):
    path, text = _first(tmp_path)
    path.unlink()

    with pytest.raises(ConfigChangedError):
        write_configuration(path, config_from_text(text, path), now=NOW, expected_text=text)

    assert not path.exists()


def test_the_conflict_names_the_file_and_is_a_configuration_error(tmp_path: Path):
    path, text = _first(tmp_path)
    path.write_text(text + "\n# edited\n", encoding="utf-8")

    with pytest.raises(ConfigError) as refused:
        write_configuration(path, config_from_text(text, path), now=NOW, expected_text=text)

    assert isinstance(refused.value, ConfigChangedError)
    assert str(path.resolve()) in str(refused.value)


def test_the_lock_is_the_one_of_the_configuration_that_was_read(tmp_path: Path):
    """A run that moves `paths.state` still stops at the lock every run that read this text takes."""
    path, text = _first(tmp_path)
    read = config_from_text(text, path)
    moved = read.model_copy(update={"paths": read.paths.model_copy(update={"state": tmp_path / "state-2"})})
    handle = acquire_lock(read.paths.lock_file, "fetch", NOW)
    try:
        with pytest.raises(LockHeldError):
            write_configuration(path, moved, now=NOW, expected_text=text)
    finally:
        release_lock(handle)

    assert path.read_text(encoding="utf-8") == text


def test_a_conflict_releases_the_lock(tmp_path: Path):
    path, text = _first(tmp_path)
    config = config_from_text(text, path)
    path.write_text(text + "\n# edited\n", encoding="utf-8")

    with pytest.raises(ConfigChangedError):
        write_configuration(path, config, now=NOW, expected_text=text)

    release_lock(acquire_lock(config.paths.lock_file, "fetch", NOW))


# --- a new file is created exclusively (decided 2026-09-25) -----------------------------------------


def _new_config(tmp_path: Path, state: str) -> Configuration:
    data = qwen35_example_config_dict(str(tmp_path / state), str(tmp_path / "models.md"))
    data.update({"schema_version": 2, "families": [], "publishers": []})
    return Configuration.from_dict(data)


def test_a_file_another_run_created_after_the_comparison_is_the_same_conflict(tmp_path: Path, monkeypatch):
    """The window between the comparison and the write: another run creates the file right there."""
    path = tmp_path / "modelroom.toml"
    theirs = "# written by the other run\n"
    real = search_apply.publish_new_text

    def other_run_first(target: Path, text: str) -> None:
        target.write_text(theirs, encoding="utf-8")
        real(target, text)

    monkeypatch.setattr(search_apply, "publish_new_text", other_run_first)

    with pytest.raises(ConfigChangedError, match=CHANGED):
        write_configuration(path, _new_config(tmp_path, "state"), now=NOW, expected_text=None)

    assert path.read_text(encoding="utf-8") == theirs


def _refused(target: Path) -> str:
    raise PermissionError(13, "the file is held by another process", str(target))


def test_a_new_file_another_run_is_publishing_at_the_comparison_is_the_same_conflict(tmp_path: Path, monkeypatch):
    """Measured on Windows (2026-09-25): a read of a file another process is renaming into place is
    refused for a moment. For a file that was not there when this run read, that is its answer."""
    path = tmp_path / "modelroom.toml"
    monkeypatch.setattr(search_apply, "_text_on_disk", _refused)

    with pytest.raises(ConfigChangedError, match=CHANGED):
        write_configuration(path, _new_config(tmp_path, "state"), now=NOW, expected_text=None)

    assert not path.exists()


def test_a_refused_read_of_a_directory_where_the_new_file_belongs_is_the_error(tmp_path: Path, monkeypatch):
    """A directory is nobody's configuration: the fault goes up, it is not called a change."""
    path = tmp_path / "modelroom.toml"
    path.mkdir()
    monkeypatch.setattr(search_apply, "_text_on_disk", _refused)

    with pytest.raises(PermissionError):
        write_configuration(path, _new_config(tmp_path, "state"), now=NOW, expected_text=None)


def test_a_refused_read_of_a_file_that_was_read_before_is_no_conflict_but_the_error(tmp_path: Path, monkeypatch):
    """Runs that read the same text share one lock, so no other of them can be writing: a refusal
    there is a fault of its own and is not called a change."""
    path, text = _first(tmp_path)
    monkeypatch.setattr(search_apply, "_text_on_disk", _refused)

    with pytest.raises(PermissionError):
        write_configuration(path, _with_context(config_from_text(text, path), 4096), now=NOW, expected_text=text)


_CREATE_SCRIPT = """
import json, sys, time
from datetime import datetime
from pathlib import Path
from modelroom.config import Configuration
from modelroom.search_apply import ConfigChangedError, write_configuration

target, config_json, go_path, ready_path = (Path(arg) for arg in sys.argv[1:5])
now = datetime.fromisoformat(sys.argv[5])
config = Configuration.from_dict(json.loads(config_json.read_text(encoding="utf-8")))
ready_path.write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 10
while not go_path.exists() and time.monotonic() < deadline:
    time.sleep(0.001)
try:
    write_configuration(target, config, now=now, expected_text=None)
except ConfigChangedError:
    print("changed", flush=True)
else:
    print("written", flush=True)
"""


def _create_round(folder: Path, script: Path) -> tuple[list[str], Path, list[Path]]:
    """Two processes create the same configuration at once, each with its own `paths.state`.

    Different state folders mean different locks, so the lock does not keep them apart: only the
    creation itself can. Each process signals it is ready and waits on the go file, so both reach
    `write_configuration` together. Returns what each process printed, the file, and the state
    folder each one wrote into its configuration.
    """
    folder.mkdir()
    target = folder / "modelroom.toml"
    go_path = folder / "go"
    states: list[Path] = []
    procs: list[subprocess.Popen] = []
    ready = [folder / f"ready-{i}" for i in range(2)]
    try:
        for i in range(2):
            config = _new_config(folder, f"state-{i}")
            config_json = folder / f"config-{i}.json"
            config_json.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
            states.append(config.paths.state.resolve())
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(script), str(target), str(config_json), str(go_path), str(ready[i]),
                     NOW.isoformat()],
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + 20
        while not all(path.exists() for path in ready) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert all(path.exists() for path in ready), "not every process signaled ready in time"
        go_path.write_text("go", encoding="utf-8")
        outputs = [proc.communicate(timeout=20)[0].strip() for proc in procs]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
    return outputs, target, states


def test_two_processes_that_create_the_file_at_once_leave_exactly_one_winner(tmp_path: Path):
    script = tmp_path / "create.py"
    script.write_text(_CREATE_SCRIPT, encoding="utf-8")

    for round_no in range(3):
        folder = tmp_path / f"round-{round_no}"
        outputs, target, states = _create_round(folder, script)

        assert sorted(outputs) == ["changed", "written"], f"round {round_no}: {outputs!r}"
        # The file on disk is the winner's, whole: it names the winner's state folder.
        on_disk = config_from_text(target.read_text(encoding="utf-8"), target).paths.state.resolve()
        assert on_disk == states[outputs.index("written")]
        assert list(folder.glob("*.tmp")) == []
