"""Sentinel test (AP9-FR, FR-B1): after a whole guided run over SSH, none of the server's own
identity ever reached a file of this folder, `remote.json`, or the screen -- only the alias does,
and this computer's own name once, in the start line (FR-N).

Sentinels stand for what a real server or a real command line could carry: its host name, its
address, ssh's own stderr, an Ollama error text, a local model name and its `/api/ps` name. All of
them are RFC 5737/`.example` (hygiene-fest, `tests/fixture_neutral.py`'s own convention).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixture_neutral import files_with
from fixture_room import fixed_room
from fixture_ssh import FakeSsh
from modelroom.daemon import FixtureDaemon
from modelroom.http import Response
from modelroom.measure import Probes
from modelroom.remote_server import ServerContext
from modelroom.ssh import ProcessOutcome, SshError

from fixture_support import build_transport, guided_transport_mapping, offline_daemon, offline_servers, windows_probes
from test_guided import FULL_ANSWERS, RUN1, _WatchingAsker, _pointer, _results

SENTINEL_HOST = "sentinel-host.example"
SENTINEL_USER = "sentinel-user"
SENTINEL_ADDRESS = "203.0.113.77"
SENTINEL_PATH = "/srv/sentinel-path"


@pytest.fixture(autouse=True)
def _results_folder(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()


class _SentinelRunner:
    """Every command answers "not found" but for the machine-id probe, whose text is a sentinel
    stderr-shaped sentence -- the way a real machine's raw identifier could look, and never once
    the value stored (only its salted digest, `profile.os_fingerprint`, ever reaches a file)."""

    def __call__(self, args):
        raise FileNotFoundError(args[0])


def _server_context() -> ServerContext:
    probes = Probes(
        platform="linux",
        runner=_SentinelRunner(),
        read_text=lambda path: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" if "machine-id" in str(path) else (_ for _ in ()).throw(OSError()),
        memory_bytes=lambda: 0,
    )
    daemon = FixtureDaemon(
        {
            ("GET", "/api/version"): Response(200, {}, b"{}"),
            ("GET", "/api/tags"): Response(200, {}, b'{"models": [{"name": "' + SENTINEL_HOST.encode() + b':latest"}]}'),
        }
    )
    return ServerContext(probes=probes, room=fixed_room(), daemon=daemon, now=lambda: RUN1, environ={})


def _run(tmp_path: Path, ssh, answers: dict | None = None):
    lines: list[str] = []
    full = {**FULL_ANSWERS, "machines": ["ssh"], "ssh_alias": "gpu1", "ssh_machine_class": "server", **(answers or {})}
    asker = _WatchingAsker(full, lines)
    from modelroom.guided import run_guided

    code = run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        servers=offline_servers(),
        now=RUN1,
        out=lines.append,
        ssh=ssh,
    )
    return code, lines


def _leak_search(folder: Path, needle: str) -> list[str]:
    """Every tracked-shape file under `folder` (state, docs, config, backups) that carries `needle`,
    decoded the way the folder's own leak test reads them (`fixture_neutral.files_with`)."""
    return files_with(folder, needle)


def test_a_run_over_ssh_leaves_no_sentinel_in_the_results_folder(tmp_path: Path):
    ssh = FakeSsh(context=_server_context())

    code, lines = _run(tmp_path, ssh)

    results = _results(tmp_path)
    for sentinel in (SENTINEL_HOST, SENTINEL_USER, SENTINEL_ADDRESS, SENTINEL_PATH):
        assert _leak_search(results, sentinel) == [], f"{sentinel} leaked into a file of the results folder"
        assert _leak_search(_pointer(tmp_path).parent, sentinel) == [], f"{sentinel} leaked into remote.json"
    joined = "\n".join(lines)
    assert SENTINEL_HOST not in joined
    assert SENTINEL_USER not in joined
    assert SENTINEL_ADDRESS not in joined


def test_ssh_stderr_with_a_sentinel_never_reaches_the_screen(tmp_path: Path):
    class _FaultRunner:
        def run(self, argv, stdin, total, per_line, limits, on_line):
            return ProcessOutcome(255, stderr=f"unknown user {SENTINEL_USER} on {SENTINEL_HOST}".encode())

    code, lines = _run(tmp_path, _FaultRunner())

    joined = "\n".join(lines)
    assert SENTINEL_HOST not in joined
    assert SENTINEL_USER not in joined


def test_the_servers_own_model_name_is_never_shown(tmp_path: Path):
    ssh = FakeSsh(context=_server_context())

    code, lines = _run(tmp_path, ssh)

    assert not any(f"{SENTINEL_HOST}:latest" in line for line in lines)


def test_this_computers_own_name_stands_only_once_in_the_start_line(tmp_path: Path):
    """FR-N: the laptop's own host name is never written to a file, and on screen stands only in
    the run's own start line -- a distinctive name, unlike a machine-class word such as
    "workstation", which legitimately appears in the document for other reasons."""
    laptop_host = "sentinel-laptop.example"
    ssh = FakeSsh(context=_server_context())
    lines: list[str] = []
    full = {**FULL_ANSWERS, "machines": ["ssh"], "ssh_alias": "gpu1", "ssh_machine_class": "server"}
    asker = _WatchingAsker(full, lines)
    from modelroom.guided import run_guided

    run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(host=laptop_host),
        daemon=offline_daemon(),
        servers=offline_servers(),
        now=RUN1,
        out=lines.append,
        ssh=ssh,
    )

    results = _results(tmp_path)
    assert _leak_search(results, laptop_host) == []
    assert _leak_search(_pointer(tmp_path).parent, laptop_host) == []
    assert sum(laptop_host in line for line in lines) == 1


# --- sentinels fed in at every place a server or ssh could carry them (Codex FR-B1 r1-11, r2-6) -----

SENTINEL_NAS = "\\\\sentinel-nas\\share"
SENTINELS = (SENTINEL_HOST, SENTINEL_USER, SENTINEL_ADDRESS, SENTINEL_PATH, SENTINEL_NAS, "sentinel-host")
_FIRST_ROW = "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0"


def _sentinel_room():
    """A server room whose every path, drive and mount point is a sentinel -- `room` answers only
    numbers and flags (B2), so none of it may arrive."""
    from modelroom.llmfit import FixtureRunner
    from modelroom.room import NVIDIA_SMI_FREE_ARGS, RoomProbes

    return RoomProbes(
        environ={"OLLAMA_MODELS": f"{SENTINEL_PATH}/models", "HOME": SENTINEL_NAS},
        home=Path(SENTINEL_PATH),
        system="linux",
        exists=lambda path: True,
        resolve=lambda path: path,
        disk_usage=lambda path: 900 * 1024**3,
        mount_of=lambda path: SENTINEL_PATH,
        runner=FixtureRunner({NVIDIA_SMI_FREE_ARGS: FileNotFoundError(f"{SENTINEL_PATH}/nvidia-smi")}),
    )


def _sentinel_daemon() -> FixtureDaemon:
    """Ollama on the server: its version, its local model names and every line of a failing pull
    carry sentinels."""
    pull = b"\n".join(
        json.dumps(line).encode()
        for line in (
            {"status": "pulling manifest"},
            {"status": f"pulling {SENTINEL_PATH}/blob", "total": 10, "completed": 1},
            {"error": f"unknown user {SENTINEL_USER} on {SENTINEL_HOST} ({SENTINEL_ADDRESS}), {SENTINEL_NAS}"},
        )
    )
    tags = {"models": [{"name": f"{SENTINEL_HOST}:latest", "digest": "e" * 64}, {"name": f"{SENTINEL_USER}/x:1", "digest": "f" * 64}]}
    return FixtureDaemon(
        {
            ("GET", "/api/version"): Response(200, {}, json.dumps({"version": f"0.9.1-{SENTINEL_HOST}"}).encode()),
            ("GET", "/api/tags"): Response(200, {}, json.dumps(tags).encode()),
            ("POST", "/api/pull"): Response(200, {}, pull),
        }
    )


def _sentinel_server() -> ServerContext:
    """A fully measurable server (the Windows/NVIDIA fixture) whose host name, room and Ollama are
    sentinels -- the host name probe must never even be called (B2)."""
    import subprocess

    from fixture_support import LAPTOP_RAM_BYTES, windows_runner
    from modelroom.measure import WINDOWS_MACHINE_GUID_ARGS

    guid = "\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\n    MachineGuid    REG_SZ    11111111-2222-3333-4444-555555555555\n"
    overrides = {WINDOWS_MACHINE_GUID_ARGS: subprocess.CompletedProcess(list(WINDOWS_MACHINE_GUID_ARGS), 0, guid, "")}
    probes = Probes(
        platform="win32",
        runner=windows_runner(overrides),
        read_text=lambda path: (_ for _ in ()).throw(OSError()),
        memory_bytes=lambda: LAPTOP_RAM_BYTES,
        hostname=lambda: SENTINEL_HOST,
    )
    return ServerContext(probes=probes, room=_sentinel_room(), daemon=_sentinel_daemon(), now=lambda: RUN1, environ={"USER": SENTINEL_USER})


class _LyingServer(FakeSsh):
    """The server's `profile` answer arrives with a sentinel as its `display_name` (a server that
    does not keep to B2): the import names the machine by its alias all the same."""

    def run(self, argv, stdin, total, per_line, limits, on_line):
        def rewrite(raw: bytes) -> None:
            envelope = json.loads(raw)
            if "export" in envelope.get("result", {}):
                envelope["result"]["export"]["profile"]["display_name"] = f"{SENTINEL_USER} {SENTINEL_HOST}"
            on_line(json.dumps(envelope).encode())

        return super().run(argv, stdin, total, per_line, limits, rewrite)


def _assert_no_sentinel(tmp_path: Path, lines: list[str], *extra: Path) -> None:
    joined = "\n".join(lines)
    for sentinel in SENTINELS:
        assert _leak_search(_results(tmp_path), sentinel) == [], f"{sentinel} reached the results folder"
        assert _leak_search(_pointer(tmp_path).parent, sentinel) == [], f"{sentinel} reached remote.json"
        for folder in extra:
            assert _leak_search(folder, sentinel) == [], f"{sentinel} reached {folder.name}"
        assert sentinel not in joined, f"{sentinel} reached the screen"


def test_sentinels_of_a_server_reach_no_file_no_export_and_no_screen(tmp_path: Path, monkeypatch):
    """A whole run over SSH -- the machine added, imported, its room read, a pull that fails with Ollama's own text --
    then an export of the profile it imported: no sentinel in the results folder (configuration,
    profiles, records, `docs/` Markdown and JSON, backups), `remote.json`, the export file, the
    system temp folder or any line on the screen, the failure's own line included."""
    systemp = tmp_path / "systemp"
    systemp.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(systemp))
    ssh = _LyingServer(context=_sentinel_server())

    code, lines = _run(tmp_path, ssh, {"pull_other": f"gpu1 {_FIRST_ROW}", "pull_remote": True})

    assert code == 1  # the pull failed: one fixed sentence, a step that did not finish
    assert any("Ollama on gpu1 reported an error" in line for line in lines)
    assert any("gpu1" in line and "over SSH" in line for line in lines)
    from modelroom.config import load_config
    from modelroom.importer import export_profile

    profile_id = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile
    exports = tmp_path / "exports"
    exports.mkdir()
    export_profile(_results(tmp_path) / "modelroom.toml", exports / "gpu1.json", profile_id, _pointer(tmp_path))
    assert json.loads((exports / "gpu1.json").read_text(encoding="utf-8"))["profile"]["display_name"] != f"{SENTINEL_USER} {SENTINEL_HOST}"
    assert list(systemp.iterdir()) == []  # the temporary export of the import is gone
    _assert_no_sentinel(tmp_path, lines, exports)
    assert any(call[1] == "room" for call in ssh.calls)  # the sentinel room was read, not skipped


def test_sentinels_in_ssh_stderr_reach_no_file_and_no_screen(tmp_path: Path):
    class _Refused:
        def run(self, argv, stdin, total, per_line, limits, on_line):
            stderr = f"{SENTINEL_USER}@{SENTINEL_HOST} ({SENTINEL_ADDRESS}): Permission denied, {SENTINEL_PATH} {SENTINEL_NAS}"
            return ProcessOutcome(255, stderr=stderr.encode())

    code, lines = _run(tmp_path, _Refused())

    assert any(line.strip().endswith("gpu1 refused the login; modelroom uses your SSH keys and agent, never a password") for line in lines)
    _assert_no_sentinel(tmp_path, lines)


def test_an_invalid_alias_with_a_sentinel_reaches_no_file_and_no_screen(tmp_path: Path):
    """The alias itself as a sentinel (an address and a domain typed where an alias belongs): never
    echoed, never stored -- at a terminal it is asked again, a valid alias goes on."""
    from modelroom.dialog import AnswerInvalidError

    with pytest.raises(AnswerInvalidError) as excinfo:
        _run(tmp_path, FakeSsh(context=_server_context()), {"ssh_alias": f"{SENTINEL_USER}@{SENTINEL_HOST}"})

    assert "sentinel" not in str(excinfo.value)
    _assert_no_sentinel(tmp_path, [])
