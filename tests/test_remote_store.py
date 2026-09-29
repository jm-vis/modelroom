"""`~/.modelroom/remote.json` (`modelroom/remote_store.py`): the parts FR-N uses -- this computer's
neutral key and the aliases the sink allows. The writer of an alias is FR-B1's; its tests follow there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixture_neutral import THIS_MACHINE
from modelroom.cli import main
from modelroom.remote_store import RemoteStoreError, aliases, read_remote, remote_path_for, this_machine

PROFILE_ID = "3f9a0c21d4e6b870"


def _write(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((data if isinstance(data, str) else json.dumps(data)).encode("utf-8"))
    return path


def test_a_missing_file_is_empty(tmp_path):
    stored = read_remote(tmp_path / "remote.json")
    assert (stored.this_machine, stored.ssh) == (None, {})
    assert not (tmp_path / "remote.json").exists()


def test_this_machine_is_made_once_and_kept_next_to_the_pointer_file(tmp_path):
    path = remote_path_for(tmp_path / ".modelroom" / "guided.json")
    assert path == tmp_path / ".modelroom" / "remote.json"
    assert this_machine(path) == THIS_MACHINE
    _write(path, {**json.loads(path.read_text(encoding="utf-8")), "ssh": {PROFILE_ID: "gpu1"}})
    assert this_machine(path) == THIS_MACHINE
    assert aliases(path) == {PROFILE_ID: "gpu1"}


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        {"schema_version": 2},
        {"schema_version": 1, "extra": True},
        {"schema_version": 1, "this_machine": "sentinel-host"},
        {"schema_version": 1, "ssh": {PROFILE_ID: "192.0.2.10"}},
        {"schema_version": 1, "ssh": {PROFILE_ID: "10"}},
        {"schema_version": 1, "ssh": {PROFILE_ID: "server-3f9a0c21"}},
        {"schema_version": 1, "ssh": {PROFILE_ID: "gpu1", "b1b2b3b4b5b6b7b8": "gpu1"}},
    ],
)
def test_a_broken_file_is_refused(tmp_path, content):
    with pytest.raises(RemoteStoreError):
        read_remote(_write(tmp_path / "remote.json", content))


def test_a_broken_file_ends_a_command_with_exit_3(tmp_path, capsys):
    folder = tmp_path / "results"
    config = _write(folder / "modelroom.toml", 'schema_version = 3\n\n[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n')
    pointer = tmp_path / "home" / ".modelroom" / "guided.json"
    _write(remote_path_for(pointer), "{not json")

    assert main(["render", "--config", str(config)], pointer_path=pointer) == 3
    assert "remote.json" in capsys.readouterr().err
