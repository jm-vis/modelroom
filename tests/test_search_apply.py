"""Tests for `search_apply.write_configuration`: a change another run wrote in between is not lost.

The guided mode reads the configuration, computes the new one and writes it back under
`modelroom.lock`. Until 2026-09-25 a second run that wrote the file between that read and that
write was simply overwritten. The writer now compares, under the lock, the file with the text the
change was computed from, and refuses when they differ (CONTRACTS.md, "Configuration").
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

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
