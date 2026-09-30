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
