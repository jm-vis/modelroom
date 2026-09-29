"""`~/.modelroom/remote.json` (`modelroom/remote_store.py`): the parts FR-N uses -- this computer's
neutral key and the aliases the sink allows -- and `profile_share` (FR-S), the field
`modelroom/guided_share.py` is the only reader of; its own tests are `tests/test_share.py`. The
writer of an alias is FR-B1's; its tests follow there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# UNC paths and drive letters are a Windows path shape; `Path(...).is_absolute()` reads them as
# absolute only there (Codex r1-7), so the tests that rely on that stay Windows-only.
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="drive letters and UNC paths are a Windows path shape")

from fixture_neutral import THIS_MACHINE
from modelroom.cli import main
from modelroom.remote_store import (
    RemoteStoreError,
    aliases,
    is_share_path,
    read_remote,
    remote_path_for,
    set_profile_share,
    this_machine,
)

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


def test_is_share_path_accepts_a_native_absolute_path_and_rejects_the_rest(tmp_path):
    assert is_share_path(str(tmp_path / "profiles"))
    assert not is_share_path("relative")
    assert not is_share_path("")
    assert not is_share_path("a\x00b")
    assert not is_share_path("x" * 1025)


@WINDOWS_ONLY
def test_is_share_path_accepts_unc_and_a_drive_letter():
    assert is_share_path("Z:\\profiles")
    assert is_share_path("\\\\sentinel-nas\\share\\profiles")


@WINDOWS_ONLY
def test_is_share_path_rejects_device_and_pipe_prefixes():
    assert not is_share_path("\\\\.\\pipe\\x")
    assert not is_share_path("\\\\?\\C:\\x")


@pytest.mark.parametrize(
    "value",
    ["", "relative", "\\\\.\\pipe\\x", "\\\\?\\C:\\x", "x" * 1025, "a\x00b"],
)
def test_a_broken_profile_share_is_refused(tmp_path, value):
    with pytest.raises(RemoteStoreError):
        read_remote(_write(tmp_path / "remote.json", {"schema_version": 1, "profile_share": value}))


def test_set_profile_share_stores_as_given_and_clears(tmp_path):
    path = remote_path_for(tmp_path / ".modelroom" / "guided.json")

    set_profile_share(path, "Z:\\profiles")
    assert read_remote(path).profile_share == "Z:\\profiles"

    set_profile_share(path, None)
    assert read_remote(path).profile_share is None


def test_set_profile_share_never_touches_this_machine_or_ssh(tmp_path):
    path = remote_path_for(tmp_path / ".modelroom" / "guided.json")
    assert this_machine(path) == THIS_MACHINE

    set_profile_share(path, "Z:\\profiles")

    assert this_machine(path) == THIS_MACHINE
    assert aliases(path) == {}


def test_a_broken_file_ends_a_command_with_exit_3(tmp_path, capsys):
    folder = tmp_path / "results"
    config = _write(folder / "modelroom.toml", 'schema_version = 3\n\n[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n')
    pointer = tmp_path / "home" / ".modelroom" / "guided.json"
    _write(remote_path_for(pointer), "{not json")

    assert main(["render", "--config", str(config)], pointer_path=pointer) == 3
    assert "remote.json" in capsys.readouterr().err
