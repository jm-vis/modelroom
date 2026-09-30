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
from modelroom.http import Response
from modelroom.measure import Probes
from modelroom.remote_store import aliases as remote_aliases
from modelroom.remote_store import read_remote, remote_path_for
from modelroom.ssh import ProcessOutcome, SshError

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
    `pull_remote_row` a real row to pull."""
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
    ssh = FaultSsh(ProcessOutcome(255, stderr=b"Permission denied"))

    code, lines, _asker = _run(tmp_path, {"ssh_alias": "gpu1"}, ssh=ssh)

    assert not (_results(tmp_path) / "modelroom.toml").is_file() is False  # config exists (from step 1's own write)
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {}
    assert any("refused the login" in line for line in lines)
    assert code == 1


def test_an_invalid_alias_under_answers_is_exit_2_and_never_reaches_the_screen(tmp_path):
    """Codex r1 finding 6: `--answers` (a `FileAsker`) gets `AnswerInvalidError` -- the fixed
    sentence only, the invalid input itself never echoed or repeated anywhere."""
    from modelroom.dialog import AnswerInvalidError

    with pytest.raises(AnswerInvalidError) as excinfo:
        _run(tmp_path, {"ssh_alias": "sentinel-host.example"}, ssh=FaultSsh(ProcessOutcome(0)))

    assert "a host alias is a short name of your own" in str(excinfo.value)
    assert "sentinel-host" not in str(excinfo.value)
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {}


def test_an_invalid_alias_at_a_terminal_asks_again_without_echoing_it():
    """Codex r1 finding 6: a terminal-shaped asker (not a `FileAsker`) is asked the alias question
    again, and the invalid attempt never reaches the screen log in between -- `_ask_alias` tested
    directly, with a minimal stand-in for the two fields it reads (`run.asker`, `run.screen`)."""
    from modelroom.guided_remote import _ask_alias

    answers = iter(["sentinel-host.example", "gpu1"])
    notes: list[str] = []
    screen_answers: list[tuple] = []

    class _ReaskingAsker:
        def text(self, key, question, default=""):
            assert key == "ssh_alias"
            return next(answers)

    class _Screen:
        def note(self, text):
            notes.append(text)

        def answer(self, question, value, **kw):
            screen_answers.append((question, value))

    class _Run:
        asker = _ReaskingAsker()
        screen = _Screen()

    alias = _ask_alias(_Run())

    assert alias == "gpu1"
    assert notes == ["a host alias is a short name of your own, like gpu1 (letters, digits, dashes)"]
    assert screen_answers == [("Host alias: your own short name for it in your SSH configuration", "gpu1")]
    assert not any("sentinel-host" in note or "user@" in note for note in notes)


# --- B4: the one active target -----------------------------------------------------------------


def test_resolve_active_prefers_a_fresh_alias_and_never_falls_back(monkeypatch):
    """Codex r1 finding 8: a run that entered a fresh alias this step stands or falls with it alone
    -- `active_target` (the stored-target picker) must not even be consulted, whether the fresh
    attempt succeeded or failed, so a failed fresh attempt can never fall back to a stored one."""
    import modelroom.guided_remote as guided_remote

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("active_target must not be consulted when a fresh alias was attempted")

    monkeypatch.setattr(guided_remote, "active_target", _must_not_be_called)

    class _Run:
        pass

    assert guided_remote.resolve_active(_Run(), None, None, None, (), "gpu1", True) == "gpu1"
    assert guided_remote.resolve_active(_Run(), None, None, None, (), None, True) is None


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
    # The first row this hardware fixture ranks for the default `FULL_ANSWERS` selection -- the
    # same name `test_guided_install.py`'s `FIRST_NAME` pins for the identical Windows/NVIDIA
    # fixture (`_measurable_context`'s own docstring).
    first_name = "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0"
    pull_body = b"\n".join(
        [json.dumps({"status": "pulling manifest"}).encode(), json.dumps({"status": "success"}).encode()]
    )
    # `_pull` (reused unchanged, Codex r1 finding 9) reads `/api/tags` again after a pull to prove
    # the daemon now lists what it just fetched (`pulled_problem`) -- so the second answer must show
    # `first_name`, the way a real Ollama daemon's inventory would after a real pull.
    tags_before = Response(200, {}, b'{"models": []}')
    tags_after = Response(200, {}, json.dumps({"models": [{"name": first_name, "digest": "e" * 64}]}).encode())
    daemon = FixtureDaemon(
        {
            ("GET", "/api/version"): Response(200, {}, b"{}"),
            # Three reads (FR-B2): step 4's own `candidates` call (nothing installed yet), then the
            # pre-pull digest check, then `pulled_problem` after the pull -- `tags_before` twice.
            ("GET", "/api/tags"): [tags_before, tags_before, tags_after],
            ("POST", "/api/pull"): Response(200, {}, pull_body),
        }
    )
    ssh = FakeSsh(context=_measurable_context(daemon=daemon))

    code, lines, _asker = _run(
        tmp_path,
        {"ssh_alias": "gpu1", "ssh_machine_class": "server", "pull_other": f"gpu1 {first_name}", "pull_remote": True},
        ssh=ssh,
    )

    assert any(f"#1 on gpu1  ollama pull {first_name}" in line for line in lines)
    assert any("onto gpu1" in line for line in lines)
    assert any("Pulled" in line and "gpu1" in line for line in lines)


def test_a_package_local_on_the_server_is_a_note_not_a_pull(tmp_path):
    """Codex r1 finding 9: `pull_remote_row` reuses `pull_chosen` whole, including its
    local-already branch -- proven directly on a `PullTarget` built the way `pull_remote_row`
    builds one (`where="gpu1"`, `key="pull_remote"`, FR-B1), against a real daemon fixture
    (`granite4.2:8b`, already local by its manifest digest -- `fixture_servers.pull_daemon`'s own
    pinned `/api/tags` -- no monkeypatch of a private helper)."""
    from fixture_servers import granite_ollama_package, pull_daemon
    from modelroom.daemon import PULL_PATH
    from modelroom.dialog import FileAsker
    from modelroom.guided_install import FirstRow, PullTarget, pull_chosen
    from modelroom.pull_choice import weights_bytes_of
    from modelroom.screen import Screen

    daemon = pull_daemon()
    lines: list[str] = []
    package = granite_ollama_package()
    first = FirstRow(1, "granite4.2:8b", 5.0, package, weights_bytes_of(package))

    def _disk_must_not_be_read():
        raise AssertionError("a local package is a note, never checked against the disk")

    def _must_not_fail(message):
        raise AssertionError(message)

    target = PullTarget(
        daemon=daemon,
        disk=_disk_must_not_be_read,
        now=RUN1,
        screen=Screen(lines.append, colored=False),
        asker=FileAsker({}),
        failed=_must_not_fail,
        where="gpu1",
        key="pull_remote",
    )

    pull_chosen(target, first)

    assert any("is local already on gpu1" in line for line in lines)
    assert not any("onto gpu1" in line for line in lines)
    assert not any(call[1] == PULL_PATH for call in daemon.calls)
