"""B3/B4 write order and B5 row choice of the SSH way (AP9-FR, FR-B1; Codex FR-B1 r2 findings 4 and 5),
end to end through `run_guided` with `FakeSsh` redirecting into `remote_server.main`.

B3/B4: the answer is validated first, then `remote.json` maps the id to the alias (atomic), then the
import renames the old row and plans under one and the same lock -- a failed `remote.json` write
leaves every other file as it was, and a failed rename aborts this import, no half state.
B5: the row of the active SSH machine chosen in `pull_other`'s list, whichever rank, is the one
pulled over SSH -- never simply the first row.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import pytest

from fixture_ssh import FakeSsh
from modelroom.config import load_config
from modelroom.daemon import PULL_PATH, FixtureDaemon
from modelroom.guided import run_guided
from modelroom.http import Response
from modelroom.remote_store import RemoteStoreError, aliases as remote_aliases
from modelroom.remote_store import remote_path_for
from modelroom.toml_writer import dump_toml

from fixture_support import build_transport, guided_transport_mapping, offline_daemon, offline_servers, windows_probes
from test_guided import FULL_ANSWERS, RUN1, RUN2, _WatchingAsker, _pointer, _results
from test_guided_remote import _measurable_context, _server_context

OLD_ID = "3f9a0c21d4e6b870"
NEW_ID = "7b1e4d09c2a85f36"


@pytest.fixture(autouse=True)
def _results_folder(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()


def _run(tmp_path: Path, answers: dict, ssh, *, now=RUN1, ids: list[str] | None = None):
    full = {**FULL_ANSWERS, "machines": ["ssh"], "ssh_alias": "gpu1", "ssh_machine_class": "server", **answers}
    lines: list[str] = []
    code = run_guided(
        _WatchingAsker(full, lines),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(ids=ids),
        daemon=offline_daemon(),
        servers=offline_servers(),
        now=now,
        out=lines.append,
        ssh=ssh,
    )
    return code, lines


def _config_file(tmp_path: Path) -> Path:
    return _results(tmp_path) / "modelroom.toml"


def _profile_file(tmp_path: Path, profile_id: str) -> Path:
    return _results(tmp_path) / "state" / "hardware" / f"{profile_id}.json"


def _first_run(tmp_path: Path) -> None:
    """gpu1 read once under `OLD_ID`, with reserves of its own on its row."""
    _run(tmp_path, {}, FakeSsh(context=_server_context("a" * 32)), ids=[OLD_ID])
    config_file = _config_file(tmp_path)
    data = tomllib.loads(config_file.read_text(encoding="utf-8"))
    data["machines"]["gpu1"]["reserve_vram_gib"] = 2.5
    data["machines"]["gpu1"]["reserve_ram_gib"] = 6.0
    config_file.write_bytes(dump_toml(data).encode("utf-8"))
    assert load_config(config_file).machines["gpu1"].profile == OLD_ID


def _new_machine_run(tmp_path: Path):
    """gpu1 now answers with another fingerprint; `new` takes the fresh id `NEW_ID`."""
    return _run(tmp_path, {"ssh_clone": "new"}, FakeSsh(context=_server_context("b" * 32)), now=RUN2, ids=[NEW_ID])


def test_new_behind_an_alias_keeps_the_old_machine_and_puts_the_new_one_exactly_under_the_alias(tmp_path):
    _first_run(tmp_path)

    _new_machine_run(tmp_path)

    machines = load_config(_config_file(tmp_path)).machines
    old_key = f"server-{OLD_ID[:8]}"
    assert machines["gpu1"].profile == NEW_ID
    assert machines[old_key].profile == OLD_ID
    assert (machines[old_key].reserve_vram_gib, machines[old_key].reserve_ram_gib) == (2.5, 6.0)
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {NEW_ID: "gpu1"}


def test_new_falls_back_to_a_free_neutral_key_when_the_neutral_name_is_taken(tmp_path):
    _first_run(tmp_path)
    config_file = _config_file(tmp_path)
    data = tomllib.loads(config_file.read_text(encoding="utf-8"))
    data["machines"][f"server-{OLD_ID[:8]}"] = {"reserve_vram_gib": 1.0, "writer": False}
    config_file.write_bytes(dump_toml(data).encode("utf-8"))

    _new_machine_run(tmp_path)

    machines = load_config(config_file).machines
    moved = [key for key, machine in machines.items() if machine.profile == OLD_ID]
    assert machines["gpu1"].profile == NEW_ID
    assert len(moved) == 1 and moved[0].startswith("computer-") and len(moved[0]) == len("computer-") + 8
    assert machines[moved[0]].reserve_vram_gib == 2.5


def test_a_failed_remote_json_write_writes_nothing_else(tmp_path, monkeypatch):
    """Codex FR-B1 r2 finding 4: the old row is not renamed before `remote.json` is written -- a
    failing alias write leaves the configuration and the profiles exactly as they were."""
    _first_run(tmp_path)
    machines_before = load_config(_config_file(tmp_path)).machines
    import modelroom.guided_remote_bind as bind

    def refuse(*args, **kwargs):
        raise RemoteStoreError("the file of remote machines could not be written")

    monkeypatch.setattr(bind, "set_alias", refuse)

    code, lines = _new_machine_run(tmp_path)

    assert load_config(_config_file(tmp_path)).machines == machines_before  # the old row not renamed
    assert not _profile_file(tmp_path, NEW_ID).exists()
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {OLD_ID: "gpu1"}
    assert code == 1


def test_a_failed_rename_of_the_old_row_aborts_this_import(tmp_path, monkeypatch):
    """Codex FR-B1 r2 finding 4: the rename of the old row belongs to the import (same lock, before
    the plan) -- when the configuration cannot be written for it, nothing of the new profile is
    written; `remote.json` already names the fresh id, which the next run converges (B3)."""
    _first_run(tmp_path)
    config_file = _config_file(tmp_path)
    armed = {"on": False, "failed": 0}
    real_replace = os.replace

    def replace_once_failing(src, dst, *args, **kwargs):
        if armed["on"] and Path(dst).name == "modelroom.toml" and not armed["failed"]:
            armed["failed"] += 1
            raise PermissionError("the configuration is locked by another program")
        return real_replace(src, dst, *args, **kwargs)

    class _ArmAfterProfile(FakeSsh):
        def run(self, argv, *rest):
            outcome = super().run(argv, *rest)
            armed["on"] = armed["on"] or argv[-1] == "profile"
            return outcome

    monkeypatch.setattr(os, "replace", replace_once_failing)

    code, lines = _run(tmp_path, {"ssh_clone": "new"}, _ArmAfterProfile(context=_server_context("b" * 32)), now=RUN2, ids=[NEW_ID])

    assert armed["failed"] == 1
    assert not _profile_file(tmp_path, NEW_ID).exists()
    machines = load_config(config_file).machines
    old_rows = [machine for machine in machines.values() if machine.profile == OLD_ID]
    assert len(old_rows) == 1 and old_rows[0].reserve_vram_gib == 2.5  # the old machine is kept whole
    assert NEW_ID not in {machine.profile for machine in machines.values()}
    assert any("the profile of gpu1 could not be imported" in line for line in lines)
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {NEW_ID: "gpu1"}  # converges next run (B3)
    assert code == 1


def test_the_chosen_row_of_the_active_target_is_pulled_over_ssh_not_the_first(tmp_path):
    """Codex FR-B1 r2 finding 5 (B5, A3): `pull_other = "gpu1 <name of #3>"` pulls #3 onto gpu1,
    through the shared `PullTarget`/`pull_chosen` chain -- its install line, question and POST all
    name #3, never #1."""
    third = "hf.co/unsloth/Qwen3.5-9B-GGUF:Q6_K"
    pull_body = b"\n".join([json.dumps({"status": "pulling manifest"}).encode(), json.dumps({"status": "success"}).encode()])
    tags_after = Response(200, {}, json.dumps({"models": [{"name": third, "digest": "e" * 64}]}).encode())
    daemon = FixtureDaemon(
        {
            ("GET", "/api/version"): Response(200, {}, b"{}"),
            ("GET", "/api/tags"): [Response(200, {}, b'{"models": []}'), tags_after],
            ("POST", PULL_PATH): Response(200, {}, pull_body),
        }
    )

    code, lines = _run(tmp_path, {"pull_other": f"gpu1 {third}", "pull_remote": True}, FakeSsh(context=_measurable_context(daemon=daemon)))

    assert any(f"#3 on gpu1  ollama pull {third}" in line for line in lines)
    assert any("Pull #3 onto gpu1 now?" in line for line in lines)
    assert [call[2]["model"] for call in daemon.calls if call[1] == PULL_PATH] == [third]
    assert any("Pulled" in line and third in line and "on gpu1" in line for line in lines)


def test_without_a_pull_other_answer_nothing_is_pulled_over_ssh(tmp_path):
    """B5: the pull over SSH follows a chosen row only -- no row chosen, no POST (the first row is
    never taken on its own)."""
    daemon = FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}"), ("GET", "/api/tags"): Response(200, {}, b'{"models": []}')})

    _code, lines = _run(tmp_path, {"pull_remote": True}, FakeSsh(context=_measurable_context(daemon=daemon)))

    assert not any(call[1] == PULL_PATH for call in daemon.calls)
    assert not any("onto gpu1" in line for line in lines)
