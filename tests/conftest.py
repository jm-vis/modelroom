"""Two rules for the whole suite: no test may reach the home folder, nor the room, of the person running it.

Every test already passes its own `pointer_path` into `tmp_path`, so nothing here reads or writes
`~/.modelroom` by design. This fixture is the guard behind that design rather than a second way of
doing it: it moves `HOME` and `USERPROFILE` into a folder of the test, so a call that forgets to
pass a path -- `binding.default_pointer_path()`, `Path.home()`, `Path.expanduser()` -- lands in
`tmp_path` and not in a real home folder. `scripts/selftest.py` does the same for its own run.

The room (AP9-UX2 U8, Codex r2-5): the free graphics memory and disk a guided run reads live come
from `fixture_room.fixed_room()` for every test; the real probes refuse to run under pytest
(`room.real_room_probes`), so a test that forgot this fails instead of reading the machine.
"""

from __future__ import annotations

import pytest

from fixture_neutral import THIS_MACHINE
from fixture_room import fixed_room


@pytest.fixture(autouse=True)
def _home_in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    # The application-data folders move with it (AP9-RM U11b): a Windows PowerShell started under a
    # USERPROFILE without `AppData\Local` gets an empty `LocalApplicationData` and writes its module
    # analysis cache relative to its working directory -- into the repository.
    for name, folder in home_app_data(home).items():
        monkeypatch.setenv(name, str(folder))
    return home


def home_app_data(home) -> dict:
    """`LOCALAPPDATA` and `APPDATA` inside `home`, created: the folders Windows resolves them to."""
    folders = {"LOCALAPPDATA": home / "AppData" / "Local", "APPDATA": home / "AppData" / "Roaming"}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


@pytest.fixture(autouse=True)
def _shipped_catalog(monkeypatch: pytest.MonkeyPatch):
    """A guided run reads the published catalog unless `MODELROOM_CATALOG=shipped` (AP9-KA-L); a test
    that passes a transport for something else meets the shipped catalog and no request for the
    dataset. `tests/test_catalog_remote.py` takes the variable out again and tests the way that is on."""
    monkeypatch.setenv("MODELROOM_CATALOG", "shipped")


@pytest.fixture(autouse=True)
def _fixed_room(monkeypatch: pytest.MonkeyPatch):
    from modelroom import room

    monkeypatch.setattr(room, "default_room_probes", lambda: fixed_room())


@pytest.fixture(autouse=True)
def _no_real_local_servers(monkeypatch: pytest.MonkeyPatch):
    """`modelroom hardware` reads which runtimes this machine has (AP9-SV2-I R1-1); a test that passes no
    daemon or server transport meets silence, never a real Ollama or vLLM of the person running it."""
    from modelroom import cli
    from modelroom.daemon import DaemonError

    def _silent(*args, **kwargs):
        raise DaemonError("no daemon in this test")

    class _SilentDaemon:
        base_url = "http://127.0.0.1:0"
        __call__ = _silent

    monkeypatch.setattr(cli, "_default_daemon", lambda: _SilentDaemon(), raising=False)
    monkeypatch.setattr(cli, "default_transports", lambda: {}, raising=False)


@pytest.fixture(autouse=True)
def _no_real_server_vllm(monkeypatch: pytest.MonkeyPatch):
    """`modelroom remote` reads the vLLM of the computer it runs on (AP9-SV2-R); a `ServerContext` a test builds
    without a `vllm=` meets silence, never the vLLM server of the person running the suite."""
    from modelroom import remote_server
    from modelroom.vllm_server import FixtureVllmServer

    monkeypatch.setattr(
        remote_server, "default_vllm",
        lambda: FixtureVllmServer({}, unreachable="http://127.0.0.1:8000/v1/models: the vLLM server did not answer (refused)"),
        raising=False,
    )


class _UnusedSsh:
    """The default `GuidedRun.ssh` of every test that never touches FR-B1 at all: a loud failure
    the moment anything tries to use it, instead of a silent real connection (FR-B1, "Teststrategie
    ohne echten Server"). A test of the SSH way builds its own `FakeSsh`/`FaultSsh` and passes it in."""

    def run(self, *args, **kwargs):  # pragma: no cover - only reached by a test-authoring mistake
        raise AssertionError("a test reached the default ssh runner; pass ssh=FakeSsh(...)/FaultSsh(...) instead")


@pytest.fixture(autouse=True)
def _no_real_ssh(monkeypatch: pytest.MonkeyPatch):
    from modelroom import ssh

    monkeypatch.setattr(ssh, "default_ssh", lambda: _UnusedSsh())


@pytest.fixture(autouse=True)
def _fixed_this_machine(monkeypatch: pytest.MonkeyPatch):
    """This computer's neutral key (FR-N) is random in a real run; a test gets the same one every time,
    so a screen or a configuration compared with a golden file reads the same. `remote.json` itself
    lands in the test's home folder (above), never in a real one."""
    from modelroom import remote_store

    monkeypatch.setattr(remote_store, "_NEW_KEY", lambda: THIS_MACHINE)
