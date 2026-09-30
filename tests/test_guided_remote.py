"""`modelroom/guided_remote.py` (FR-B1, B3-B5): the `ssh` row of step 1, end to end through
`run_guided`, with `FakeSsh` redirecting into `remote_server.main` -- no real ssh, no real server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixture_room import fixed_room
from fixture_ssh import FakeSsh, FaultSsh
from modelroom.config import load_config
from modelroom.daemon import FixtureDaemon
from modelroom.dialog import AnswerMissingError, FileAsker
from modelroom.guided import run_guided
from modelroom.guided_remote import active_target
from modelroom.http import Response
from modelroom.measure import Probes
from modelroom.remote_store import aliases as remote_aliases
from modelroom.remote_store import read_remote, remote_path_for
from modelroom.ssh import ProcessOutput, SshError

from fixture_support import (
    LAPTOP_RAM_BYTES,
    build_transport,
    guided_transport_mapping,
    offline_daemon,
    offline_servers,
    windows_probes,
    windows_runner,
)
from test_guided import FULL_ANSWERS, RUN1, RUN2, _WatchingAsker, _pointer, _results

MACHINE_ID_FILE = "/etc/machine-id"


class _NothingRunner:
    """Every external command answers "not found" -- a hardware measurement with no tool at all,
    never a fault (mirrors `tests/test_remote_server.py::_NothingRunner`)."""

    def __call__(self, args):
        raise FileNotFoundError(args[0])


def _server_probes(machine_id: str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") -> Probes:
    def read_text(path):
        if str(path) == MACHINE_ID_FILE:
            return machine_id
        raise OSError("no such file")

    return Probes(platform="linux", runner=_NothingRunner(), read_text=read_text, memory_bytes=lambda: 0)


def _server_context(machine_id: str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"):
    from modelroom.remote_server import ServerContext

    daemon = FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}")})
    return ServerContext(probes=_server_probes(machine_id), room=fixed_room(), daemon=daemon, now=lambda: RUN1, environ={})


def _measurable_context(guid: str = "11111111-2222-3333-4444-555555555555", daemon=None, now=RUN1):
    """A server whose hardware measures fully (the same Windows/NVIDIA fixture `this-machine`'s own
    tests use, `fixture_support.windows_runner`): enough for the render to rank a package and give
    `pull_remote_step` a real row to pull."""
    from modelroom.measure import WINDOWS_MACHINE_GUID_ARGS
    from modelroom.remote_server import ServerContext
    import subprocess

    guid_line = f"\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\n    MachineGuid    REG_SZ    {guid}\n"
    overrides = {WINDOWS_MACHINE_GUID_ARGS: subprocess.CompletedProcess(list(WINDOWS_MACHINE_GUID_ARGS), 0, guid_line, "")}
    probes = Probes(platform="win32", runner=windows_runner(overrides), read_text=lambda p: (_ for _ in ()).throw(OSError()), memory_bytes=lambda: LAPTOP_RAM_BYTES)
    active_daemon = daemon if daemon is not None else FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}")})
    return ServerContext(probes=probes, room=fixed_room(), daemon=active_daemon, now=lambda: now, environ={})


def _run(tmp_path: Path, answers: dict, *, ssh=None, now=RUN1, machines=("ssh",)):
    full = {**FULL_ANSWERS, "machines": list(machines), **answers}
    lines: list[str] = []
    asker = _WatchingAsker(full, lines)
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
        ssh=ssh,
    )
    return code, lines, asker


@pytest.fixture(autouse=True)
def _results_folder(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()


# --- B3: a new alias ---------------------------------------------------------------------------


def test_an_alias_adds_the_machine_under_the_alias(tmp_path):
    ssh = FakeSsh(context=_server_context())

    code, lines, _asker = _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)

    config = load_config(_results(tmp_path) / "modelroom.toml")
    assert "gpu1" in config.machines
    assert any(call[1] == "hello" for call in ssh.calls) and any(call[1] == "profile" for call in ssh.calls)
    profile_id = config.machines["gpu1"].profile
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {profile_id: "gpu1"}
    profile_path = _results(tmp_path) / "state" / "hardware" / f"{profile_id}.json"
    assert profile_path.is_file()
    import json

    assert json.loads(profile_path.read_text(encoding="utf-8"))["display_name"] == "gpu1"
    assert any("gpu1" in line and "over SSH" in line for line in lines)


def test_the_same_machine_keeps_its_id(tmp_path):
    ssh = FakeSsh(context=_server_context())
    _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)
    first_id = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile

    ssh2 = FakeSsh(context=_server_context())
    _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh2, now=RUN2)

    second_id = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile
    assert first_id == second_id


def test_another_fingerprint_asks_and_new_takes_a_fresh_id(tmp_path):
    ssh = FakeSsh(context=_server_context("a" * 32))
    _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)
    first_id = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile

    ssh2 = FakeSsh(context=_server_context("b" * 32))
    code, lines, _asker = _run(
        tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server", "ssh_clone": "new"}, ssh=ssh2, now=RUN2
    )

    second_id = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile
    assert second_id != first_id
    assert any("gpu1" in line for line in lines if "same machine or a new one" in line or "new machine" in line)


def test_a_failed_connection_writes_nothing_and_the_run_goes_on(tmp_path):
    ssh = FaultSsh(ProcessOutput([], 255, stderr=b"Permission denied"))

    code, lines, _asker = _run(tmp_path, {"ssh_alias": "gpu1"}, ssh=ssh)

    assert not (_results(tmp_path) / "modelroom.toml").is_file() is False  # config exists (from step 1's own write)
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {}
    assert any("refused the login" in line for line in lines)
    assert code == 1


def test_an_invalid_alias_asks_nothing_more(tmp_path):
    code, lines, _asker = _run(tmp_path, {"ssh_alias": "192.0.2.10"}, ssh=FaultSsh(ProcessOutput([], 0)))

    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {}
    assert any("a host alias is a short name of your own" in line for line in lines)


# --- B4: the one active target -----------------------------------------------------------------


def test_active_target_prefers_a_fresh_alias():
    class _Run:
        pass

    assert active_target(_Run(), None, (), "gpu1") == "gpu1"


def test_a_stored_ssh_profile_picked_directly_is_checked_before_pull_and_measurement(tmp_path):
    ssh = FakeSsh(context=_server_context("a" * 32))
    _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)
    old_id = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile

    ssh2 = FakeSsh(context=_server_context("b" * 32))
    code, lines, asker = _run(
        tmp_path,
        {"machines": [f"profile:{old_id}"], "ssh_clone": "new"},
        ssh=ssh2,
        now=RUN2,
        machines=(f"profile:{old_id}",),
    )

    assert any("counts from the next run" in line for line in lines)
    new_id = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile
    assert new_id != old_id  # rebound to the fresh id (B3's own write, done inside verify_active)


def test_a_changed_fingerprint_without_answer_is_exit_2_with_no_write_after_it(tmp_path):
    ssh = FakeSsh(context=_server_context("a" * 32))
    _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)
    old_id = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile
    before = read_remote(remote_path_for(_pointer(tmp_path))).ssh

    ssh2 = FakeSsh(context=_server_context("b" * 32))
    with pytest.raises(AnswerMissingError):
        _run(tmp_path, {"machines": [f"profile:{old_id}"]}, ssh=ssh2, now=RUN2, machines=(f"profile:{old_id}",))

    assert read_remote(remote_path_for(_pointer(tmp_path))).ssh == before


def test_two_stored_ssh_targets_connect_only_to_the_active_one(tmp_path):
    ssh_a = FakeSsh(context=_server_context("a" * 32))
    _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh_a)
    ssh_b = FakeSsh(context=_server_context("c" * 32))
    _run(tmp_path, {"ssh_alias": "gpu2", "ssh_machine_class": "server"}, ssh=ssh_b, now=RUN2)
    config = load_config(_results(tmp_path) / "modelroom.toml")
    gpu1_id, gpu2_id = config.machines["gpu1"].profile, config.machines["gpu2"].profile

    ssh3 = FakeSsh(context=_server_context("c" * 32))  # gpu2's fingerprint unchanged: no clone question
    code, lines, _asker = _run(
        tmp_path,
        {"machines": [f"profile:{gpu1_id}", f"profile:{gpu2_id}"], "ssh_active": "gpu2"},
        ssh=ssh3,
        now=RUN2,
        machines=(f"profile:{gpu1_id}", f"profile:{gpu2_id}"),
    )

    assert not any(call[0] == "gpu1" for call in ssh3.calls)
    assert any(call[0] == "gpu2" for call in ssh3.calls)


def test_several_stored_targets_without_ssh_active_connect_to_none(tmp_path):
    ssh_a = FakeSsh(context=_server_context("a" * 32))
    _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh_a)
    ssh_b = FakeSsh(context=_server_context("c" * 32))
    _run(tmp_path, {"ssh_alias": "gpu2", "ssh_machine_class": "server"}, ssh=ssh_b, now=RUN2)
    config = load_config(_results(tmp_path) / "modelroom.toml")
    gpu1_id, gpu2_id = config.machines["gpu1"].profile, config.machines["gpu2"].profile

    ssh3 = FakeSsh(context=_server_context("a" * 32))
    code, lines, _asker = _run(
        tmp_path,
        {"machines": [f"profile:{gpu1_id}", f"profile:{gpu2_id}"]},
        ssh=ssh3,
        now=RUN2,
        machines=(f"profile:{gpu1_id}", f"profile:{gpu2_id}"),
    )

    assert ssh3.calls == []


# --- the screen: origin and the free line -------------------------------------------------------


def test_the_head_says_over_ssh(tmp_path):
    ssh = FakeSsh(context=_server_context())
    _code, lines, _asker = _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)

    assert any("gpu1" in line and "over SSH" in line for line in lines)


def test_the_card_says_read_over_ssh_not_this_computer(tmp_path):
    """B3, the brief's own wording, on the run's summary card (`result_card.ORIGIN_WORDS`) --
    distinct from the table head's `· over SSH` suffix, checked above."""
    ssh = FakeSsh(context=_server_context())
    _code, lines, _asker = _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)

    assert any(line.strip() == "read over SSH · not this computer" for line in lines)


def test_the_free_line_names_the_disk_on_gpu1_without_a_path(tmp_path):
    ssh = FakeSsh(context=_server_context())
    _code, lines, _asker = _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)

    disk_lines = [line for line in lines if "disk" in line and "gpu1" in line]
    assert disk_lines
    assert not any("\\" in line or "/mnt" in line or "C:" in line for line in disk_lines)
    assert any("free on gpu1 (where Ollama keeps its models)" in line for line in disk_lines)


# --- B5: the pull over SSH -----------------------------------------------------------------------


def test_pull_remote_checks_the_disk_again_and_pulls_over_ssh(tmp_path):
    pull_body = b"\n".join(
        [json.dumps({"status": "pulling manifest"}).encode(), json.dumps({"status": "success"}).encode()]
    )
    daemon = FixtureDaemon(
        {
            ("GET", "/api/version"): Response(200, {}, b"{}"),
            ("GET", "/api/tags"): Response(200, {}, b'{"models": []}'),
            ("POST", "/api/pull"): Response(200, {}, pull_body),
        }
    )
    ssh = FakeSsh(context=_measurable_context(daemon=daemon))

    code, lines, _asker = _run(
        tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server", "pull_remote": True}, ssh=ssh
    )

    assert any("onto gpu1" in line for line in lines)
    assert any("Pulled" in line and "gpu1" in line for line in lines)


def test_a_package_local_on_the_server_is_a_note_not_a_pull(tmp_path):
    daemon = FixtureDaemon(
        {("GET", "/api/version"): Response(200, {}, b"{}"), ("GET", "/api/tags"): Response(200, {}, b'{"models": []}')}
    )
    ssh = FakeSsh(context=_measurable_context(daemon=daemon))
    _run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh=ssh)
    config = load_config(_results(tmp_path) / "modelroom.toml")

    import modelroom.guided_remote as guided_remote

    original = guided_remote._held_on
    try:
        guided_remote._held_on = lambda daemon, name: True
        ssh2 = FakeSsh(context=_measurable_context(daemon=daemon, now=RUN2))
        code, lines, _asker = _run(
            tmp_path,
            {"machines": [f"profile:{config.machines['gpu1'].profile}"], "pull_remote": True},
            ssh=ssh2,
            now=RUN2,
            machines=(f"profile:{config.machines['gpu1'].profile}",),
        )
    finally:
        guided_remote._held_on = original

    assert any("is local already on gpu1" in line for line in lines)
    assert not any("onto gpu1" in line for line in lines)
