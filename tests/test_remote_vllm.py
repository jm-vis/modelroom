"""vLLM at a server over SSH (AP9-SV2-R, R1 to R3): `modelroom remote vllm` and `vllm-measure`.

The counterpart is `remote_server.main` in-process behind `FakeSsh` (`tests/fixture_ssh.py`), as in FR; at
the "server" stands a `FixtureVllmServer` with the SV1 recordings (`vllm_*_l4.*`) and, where a cache is
read, a fake file system (`CacheProbes`) or real files in `tmp_path`. No ssh, no network, no real vLLM:
nothing here starts a program or opens a port. What crosses the wire is held against sentinels -- a
person's name in `approval.by`, a model path and a `--served-model-name` the server chose, a file name of
its cache -- and the commands a table prints for `gpu1` are the very ones SV2-I builds (I1).

Every test imports what AP9-SV2-R added inside itself (or inside a helper it calls), so a red run against
the basis fails test by test and not as one module that cannot be collected.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from fixture_room import fixed_room
from fixture_ssh import FakeSsh, FaultSsh
from fixture_support import (
    VLLM_UNREACHABLE,
    json_response,
    qwen25_tensor_package,
    vllm_servers,
)
from modelroom.contracts import Approval, PackageFile
from modelroom.daemon import FixtureDaemon
from modelroom.dialog import FileAsker
from modelroom.http import Response
from modelroom.measure import Probes
from modelroom.measurements import Scenario
from modelroom.room import GIB, Disk, LiveRoom
from modelroom.screen import Screen, plain
from modelroom.ssh import ProcessOutcome, SshError
from modelroom.vllm_server import CHAT_PATH, METRICS_PATH, MODELS_PATH, FixtureVllmServer

from test_golden_vllm import FIXTURES, _base_model, _gguf_package, _profile
from test_install_vllm import DOCKER, EXPRESSION, RELATIVE_NOTE, REPO as AWQ_REPO, REVISION as AWQ_REVISION, SERVE
from test_install_vllm import _package as _awq_package

NOW = datetime(2026, 10, 2, 8, 0, 0, tzinfo=timezone.utc)
SENTINEL_HOST = "sentinel-host"
SENTINEL_PERSON = "sentinel-person"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # the model the recorded server serves, at context 4096
FORM = "gpu1 answered in a form this modelroom does not read"


# --- the packages ---------------------------------------------------------------------------------------------------


def _file(name: str, size: int, role: str = "other") -> PackageFile:
    return PackageFile(name=name, role=role, size_bytes=size, digest=None)


def _small(repo: str = MODEL, revision: str | None = None, tokenizer: int = 50, **changes):
    """The recorded server's package with three start files (weights, tokenizer, config) and a README."""
    files = [
        _file("model.safetensors", 1000, "weights"), _file("tokenizer.json", tokenizer), _file("config.json", 30),
        _file("README.md", 99_999),
    ]
    package = qwen25_tensor_package().model_copy(update={"repo": repo, "files": files, **changes})
    return package.model_copy(update={"revision": revision}) if revision is not None else package


def _scenario(context: int = 4096, requests: int = 20) -> Scenario:
    return Scenario(context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=requests)


# --- the "server" ---------------------------------------------------------------------------------------------------


def _nothing(args):
    raise FileNotFoundError(args[0])


def _probes(platform: str = "linux") -> Probes:
    return Probes(platform=platform, runner=_nothing, read_text=lambda path: (_ for _ in ()).throw(OSError("no")), memory_bytes=lambda: 0)


def _room(tmp_path: Path, *, hub: Path | None = None, free: int = 50 * GIB, environ: dict | None = None, **changes):
    """The room of a Linux-like server: its own home, the Hugging Face cache where `hub` says, `free` bytes free."""
    env = dict(environ or {})
    if hub is not None:
        env["HF_HUB_CACHE"] = str(hub)
    return dataclasses.replace(
        fixed_room(free_disk_bytes=free, environ=env), **{"system": sys.platform, "home": tmp_path / "home", **changes}
    )


def _models(root: str, model_id: str | None = None, length: int = 4096) -> Response:
    """What `GET /v1/models` of a vLLM server answers, with the model named as the test says."""
    entry = {"id": model_id or root, "object": "model", "owned_by": "vllm", "root": root, "parent": None, "max_model_len": length}
    return Response(200, {}, json.dumps({"object": "list", "data": [entry]}).encode("utf-8"))


def _serving(root: str = MODEL, model_id: str | None = None, length: int = 4096) -> FixtureVllmServer:
    """A vLLM server that states `root`, and answers one load test of 20 requests with the SV1 recordings."""
    from fixture_servers import vllm_chat_answer, vllm_metrics_pages

    return FixtureVllmServer(
        {
            ("GET", MODELS_PATH): _models(root, model_id, length),
            ("GET", METRICS_PATH): vllm_metrics_pages(20),
            ("POST", CHAT_PATH): vllm_chat_answer(),
        }
    )


def _silent() -> FixtureVllmServer:
    return FixtureVllmServer({}, unreachable=VLLM_UNREACHABLE)


def _asks_key() -> FixtureVllmServer:
    return FixtureVllmServer({("GET", MODELS_PATH): Response(401, {}, b"{}")})


def _context(tmp_path: Path, *, vllm=None, platform: str = "linux", room=None, **changes):
    """A server: Ollama answering its version, vLLM as `vllm` says, no vLLM installed, an empty fake cache."""
    from modelroom.remote_server import ServerContext

    daemon = FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}")})
    base = dict(
        probes=_probes(platform), room=room if room is not None else _room(tmp_path), daemon=daemon, now=lambda: NOW,
        environ={}, vllm=vllm if vllm is not None else _silent(), is_file=lambda path: False,
    )
    base.update(changes)
    return ServerContext(**base)


def _call(op: str, payload: dict, context) -> list[dict]:
    from modelroom.remote_server import main

    lines: list[bytes] = []
    assert main(op, json.dumps(payload).encode("utf-8"), lines.append, context) == 0
    return [json.loads(line) for line in lines]


def _fake_cache(hub: Path, present: dict[tuple[str, str, str], int], stats: list | None = None):
    """A cache that holds exactly `present` (`(repo, revision, file name) -> size`): every path is its own real
    path, and `stat` knows these files alone -- no file system of this machine is touched."""
    from modelroom.hf_cache import CacheProbes

    wanted = {
        os.path.join(str(hub), "models--" + repo.replace("/", "--"), "snapshots", revision, *name.split("/")): size
        for (repo, revision, name), size in present.items()
    }

    def stat_of(path: str):
        if stats is not None:
            stats.append(path)
        if path in wanted:
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_size=wanted[path])
        raise FileNotFoundError(path)

    return CacheProbes(realpath=lambda path: path, stat=stat_of)


def _present(package, skip: tuple[str, ...] = ()) -> dict[tuple[str, str, str], int]:
    """Every start file of `package` there with its declared size, but `skip`."""
    from modelroom.tensor_quant import start_files

    return {(package.repo, package.revision, f.name): f.size_bytes for f in start_files(package.files) if f.name not in skip}


# --- the laptop's run ---------------------------------------------------------------------------------------------------


class _Screen:
    glyphs = SimpleNamespace(dot="·", skip="-")

    def __init__(self) -> None:
        self.notes: list[str] = []
        self.answers: list[tuple] = []

    def note(self, text: str) -> None:
        self.notes.append(text)

    def answer(self, question: str, value: str, **_kw) -> None:
        self.answers.append((question, value))


class _Run:
    """The few fields of a `GuidedRun` the glue of R3 reads: the ssh seam, the active computer, what the run
    knows of its vLLM, the screen, the chosen models. No `.probes`, no daemon: nothing local is read."""

    def __init__(self, ssh, ssh_active: str | None = "gpu1", models: tuple[str, ...] = (MODEL,), asker=None) -> None:
        self.ssh = ssh
        self.ssh_active = ssh_active
        self.remote_vllm: dict = {}
        self.screen = _Screen()
        self.problems: list[str] = []
        self.selection = SimpleNamespace(models=models)
        self.asker = asker
        self.pointer_path = Path("home") / "guided.json"

    def failed(self, message: str) -> None:
        self.problems.append(message)


def _hello(run: _Run, alias: str = "gpu1") -> dict:
    """`hello` at the server, kept in the run as step 1 keeps it (`guided_remote_vllm.remember_hello`)."""
    from modelroom.guided_remote_vllm import remember_hello
    from modelroom.remote_client import hello

    info = hello(alias, run.ssh, ssh_path="ssh")
    remember_hello(run, alias, info)
    return info


def _offer(run: _Run, packages: list, monkeypatch, scenario: Scenario | None = None):
    """Step 4's vLLM part for the active computer, for the run's `packages`."""
    import modelroom.guided_remote_vllm as grv

    monkeypatch.setattr(grv, "run_packages", lambda config, selection: packages)
    return grv.vllm_offer(run, None, scenario or _scenario())


# --- R1: the three optional fields --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state, vllm, which",
    [
        ("serving", lambda: _serving(), None),
        ("installed", lambda: _silent(), "/usr/bin/vllm"),
        ("absent", lambda: _silent(), None),
        ("asks_key", lambda: _asks_key(), None),
    ],
)
def test_hello_names_the_vllm_state_on_the_server(tmp_path, state, vllm, which):
    """R1: `hello.vllm` is the vLLM entry's own `detect` at the server, one of the four states."""
    from modelroom.remote_client import hello

    room = _room(tmp_path, which=lambda name: which if name == "vllm" else None)
    ssh = FakeSsh(context=_context(tmp_path, vllm=vllm(), room=room))

    assert _call("hello", {}, ssh.context)[0]["result"]["vllm"] == state
    assert hello("gpu1", ssh, ssh_path="ssh")["vllm"] == state  # and it passes the laptop's strict reading


def test_an_old_hello_without_vllm_reads_absent():
    """R1: an older counterpart sends no `vllm`, no `model_disk` and no `runtimes`: they read as not there."""
    from modelroom.remote_client import hello, profile, room

    old_hello = b'{"protocol":1,"result":{"version":"0.2.0","system":"linux","ollama":"reachable","agent_forwarded":false,"os_fingerprint":"none"}}'
    assert hello("gpu1", FaultSsh(ProcessOutcome(0), lines=(old_hello,)), ssh_path="ssh")["vllm"] == "absent"
    old_room = b'{"protocol":1,"result":{"disk":{"known":true,"free_bytes":5,"unclear":false},"graphics_free_mib":null}}'
    assert room("gpu1", [], FaultSsh(ProcessOutcome(0), lines=(old_room,)), ssh_path="ssh").get("model_disk") is None


def test_the_vllm_op_answers_with_an_index_never_a_name(tmp_path, monkeypatch):
    """R2: the model the server names (`id`, `root`) never leaves it. A local path and a free `--served-model-name`
    are no index; the answer, the screen and every file stay free of them."""
    package = _small()
    results = tmp_path / "results"
    results.mkdir()
    for root, served_name, index in (("/srv/sentinel-host/model", SENTINEL_HOST, None), (MODEL, SENTINEL_HOST, 0)):
        server = _serving(root=root, model_id=served_name)
        ssh = FakeSsh(context=_context(tmp_path, vllm=server))
        run = _Run(ssh)
        _hello(run)

        _offer(run, [package], monkeypatch)

        answer = json.dumps([json.loads(line) for line in ssh.lines_seen])
        assert SENTINEL_HOST not in answer and "sentinel" not in answer
        assert run.remote_vllm["gpu1"].served_index == index
        assert not any(SENTINEL_HOST in note or "/srv" in note for note in run.screen.notes), run.screen.notes
        assert run.problems == []
    assert list(results.rglob("*")) == []  # nothing is written here, and no file of the tree carries a sentinel
    assert not any(SENTINEL_HOST.encode() in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())


def test_the_served_repo_of_a_run_package_gives_its_index(tmp_path):
    """R2: the index is the position of the one package of the request whose repo the server serves; a repo of
    two revisions, which the server does not state, is no index at all."""
    from modelroom.remote_vllm import package_ref

    first, second = _small("acme/Other-AWQ", "e" * 40), _small()
    refs = lambda *packages: [package_ref(p) for p in packages]  # noqa: E731
    context = _context(tmp_path, vllm=_serving())

    served = _call("vllm", {"packages": refs(first, second)}, context)[0]["result"]["served"]
    assert served == {"package": 1, "context": 4096}

    twin = _small(revision="f" * 40)
    ambiguous = _call("vllm", {"packages": refs(second, twin)}, context)[0]["result"]["served"]
    assert ambiguous == {"package": None, "context": 4096}

    other = _call("vllm", {"packages": refs(first)}, context)[0]["result"]["served"]
    assert other == {"package": None, "context": 4096}


# --- R2: vllm-measure runs at the server ----------------------------------------------------------------------------------


def test_vllm_measure_runs_at_the_server_not_here(tmp_path, monkeypatch):
    """R2: the load test is run against the server's own `127.0.0.1:8000`; this computer's transport to a local
    vLLM is never touched, not even constructed to call."""
    import modelroom.guided_remote_measure as grm
    import modelroom.vllm_server as vllm_server
    from modelroom.guided_remote_vllm import remember_hello  # noqa: F401
    from modelroom.remote_vllm import vllm_measure

    def _never_here(self, *args, **kwargs):
        raise AssertionError("this computer's own vLLM transport was asked")

    monkeypatch.setattr(vllm_server.LocalVllmServer, "__call__", _never_here)
    server = _serving()
    ssh = FakeSsh(context=_context(tmp_path, vllm=server))
    package = _small()

    measurement = vllm_measure("gpu1", [package], 0, _scenario(), ssh, ssh_path="ssh")

    assert measurement.problem is None and measurement.total_tps > 0 and measurement.requests == 20
    assert {(method, path) for method, path, _body in server.calls} == {("GET", MODELS_PATH), ("GET", METRICS_PATH), ("POST", CHAT_PATH)}
    assert [call[1] for call in ssh.calls] == ["vllm-measure"]

    # and through step 4: the picked row is measured over SSH, the number stays in the run
    run = _Run(FakeSsh(context=_context(tmp_path, vllm=_serving())), asker=SimpleNamespace(
        confirm=lambda key, question, default=True: True, checkbox=lambda key, question, choices: ["vllm"],
    ))
    _hello(run)
    monkeypatch.setattr(grm, "run_packages", lambda config, selection: [])
    import modelroom.guided_remote_vllm as grv

    monkeypatch.setattr(grv, "run_packages", lambda config, selection: [package])
    assert grm.remote_load_test(run, None, _scenario()) == "1 of 1 measured on gpu1"
    assert run.problems == []
    assert any(note.startswith("measured vLLM on gpu1:") for note in run.screen.notes), run.screen.notes
    assert "identity stated by the server; not ranked, nothing written" in run.screen.notes
    assert run.remote_vllm["gpu1"].outcome is not None


def test_measure_progress_carries_no_name(tmp_path):
    """R2: `progress` is `{"stage", "number", "of"}`: the warm-up, then three batches. No model, no path."""
    from modelroom.remote_vllm import vllm_measure

    ssh = FakeSsh(context=_context(tmp_path, vllm=_serving(root=MODEL, model_id=SENTINEL_HOST)))
    seen: list[dict] = []

    vllm_measure("gpu1", [_small()], 0, _scenario(), ssh, ssh_path="ssh", on_progress=seen.append)

    assert seen == [
        {"stage": "warmup", "number": 1, "of": 1}, {"stage": "batch", "number": 1, "of": 3},
        {"stage": "batch", "number": 2, "of": 3}, {"stage": "batch", "number": 3, "of": 3},
    ]
    wire = b"\n".join(ssh.lines_seen).decode("utf-8")
    assert "Qwen" not in wire and "sentinel" not in wire


@pytest.mark.parametrize("code", ["banana", "Invalid_Status", "", 3, True])
def test_an_unknown_problem_code_is_a_form_fault(code):
    """R2: a `problem` outside the fixed set is a form fault (B1), never read as a sentence."""
    from modelroom.ssh import call

    speeds = {name: 1.5 for name in ("total_tps", "total_min", "total_max", "request_tps", "request_min", "request_max")}
    measurement = {"requests": 20, "ranking_requests": 20, "context": 4096, **speeds, "problem": code}
    line = json.dumps({"protocol": 1, "result": {"measurement": measurement}}).encode("utf-8")

    with pytest.raises(SshError, match=FORM):
        call("gpu1", "vllm_measure", {}, FaultSsh(ProcessOutcome(0), lines=(line,)), ssh_path="ssh")
    good = json.dumps({"protocol": 1, "result": {"measurement": {**measurement, "problem": "other_context"}}}).encode("utf-8")
    assert call("gpu1", "vllm_measure", {}, FaultSsh(ProcessOutcome(0), lines=(good,)), ssh_path="ssh")["result"]["measurement"]["problem"] == "other_context"


@pytest.mark.parametrize(
    "code, sentence, server",
    [
        ("vllm_unreachable", "vLLM does not answer on gpu1", lambda: _silent()),
        ("vllm_asks_key", "the vLLM server on gpu1 asks for an API key; modelroom reads an open server only", lambda: _asks_key()),
        ("other_model", "vLLM on gpu1 serves another model now", lambda: _serving(root="acme/Other-AWQ")),
    ],
)
def test_each_new_error_code_maps_to_its_sentence(tmp_path, code, sentence, server):
    """R2: the three codes of the server, each one fixed sentence at this computer -- and the server's real
    reasons produce them: no answer, a key asked for, a model that is not the package asked about."""
    from modelroom.remote_vllm import vllm_measure
    from modelroom.ssh import ERROR_SENTENCES

    assert ERROR_SENTENCES[code].format(alias="gpu1") == sentence
    ssh = FakeSsh(context=_context(tmp_path, vllm=server()))

    assert _call("vllm-measure", _measure_request([_small()]), ssh.context)[0] == {"protocol": 1, "error": code}
    with pytest.raises(SshError) as excinfo:
        vllm_measure("gpu1", [_small()], 0, _scenario(), ssh, ssh_path="ssh")
    assert str(excinfo.value) == sentence


def _measure_request(packages: list, index: int = 0, scenario: Scenario | None = None) -> dict:
    from modelroom.remote_vllm import package_ref

    scenario = scenario or _scenario()
    return {
        "packages": [package_ref(package) for package in packages], "package": index,
        "scenario": {"context_requested": scenario.context_requested, "requests": scenario.requests},
    }


def _snapshot(folder: Path) -> dict[str, int]:
    return {str(path.relative_to(folder)): path.stat().st_size for path in sorted(folder.rglob("*")) if path.is_file()}


def test_the_server_writes_nothing_and_measures_nothing_persisted(tmp_path, monkeypatch):
    """R2 (FR B2, E3): the server's own folders and the laptop's results folder are the same after a `vllm` and a
    `vllm-measure`; the server's vLLM was asked the three calls and nothing else (no start, no load)."""
    server_home, results = tmp_path / "server", tmp_path / "results"
    hub = server_home / "hub"
    package = _small()
    (hub / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / package.revision).mkdir(parents=True)
    (hub / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / package.revision / "model.safetensors").write_bytes(b"x" * 1000)
    results.mkdir()
    server = _serving()
    ssh = FakeSsh(context=_context(tmp_path, vllm=server, room=_room(server_home, hub=hub)))
    before = (_snapshot(server_home), _snapshot(results), _snapshot(tmp_path / "home") if (tmp_path / "home").exists() else {})
    run = _Run(ssh)
    _hello(run)

    offer = _offer(run, [package], monkeypatch)
    from modelroom.guided_remote_vllm import measure

    assert measure(run, offer, _scenario()) is not None

    after = (_snapshot(server_home), _snapshot(results), _snapshot(tmp_path / "home") if (tmp_path / "home").exists() else {})
    assert after == before
    assert {(method, path) for method, path, _body in server.calls} <= {("GET", MODELS_PATH), ("GET", METRICS_PATH), ("POST", CHAT_PATH)}
    assert [call[1] for call in ssh.calls] == ["hello", "vllm", "vllm-measure"]


def test_room_model_disk_carries_no_path(tmp_path):
    """R1: `room.model_disk` is four numbers and flags: no drive, no path -- through the laptop's strict reading too."""
    from modelroom.remote_client import room as ssh_room

    hub = tmp_path / "sentinel-host" / "hub"
    ssh = FakeSsh(context=_context(tmp_path, room=_room(tmp_path, hub=hub, free=7 * GIB)))

    result = _call("room", {"cards": []}, ssh.context)[0]["result"]

    assert result["model_disk"] == {"known": True, "free_bytes": 7 * GIB, "same_as_ollama": True, "relative": False}
    wire = json.dumps(result)
    assert "sentinel" not in wire and str(tmp_path) not in wire and "C:" not in wire and "hub" not in wire
    assert ssh_room("gpu1", [], ssh, ssh_path="ssh")["model_disk"] == result["model_disk"]
    other = dataclasses.replace(_room(tmp_path, hub=hub), system="linux", mount_of=lambda path: "/vllm" if "hub" in str(path) else "/")
    assert _call("room", {"cards": []}, _context(tmp_path, room=other))[0]["result"]["model_disk"]["same_as_ollama"] is False


def test_missing_bytes_on_the_server_per_package(tmp_path):
    """R2: for each package, in the request's order, a number or `null`: `0` only for all start files there."""
    from modelroom.remote_vllm import package_ref

    hub = tmp_path / "hub"
    complete = _small("acme/Complete-AWQ", "a" * 40)
    no_shard = _small("acme/NoShard-AWQ", "b" * 40)
    nothing = _small("acme/Nothing-AWQ", "c" * 40)
    zero_missing = _small("acme/Zero-AWQ", "d" * 40, files=[_file("model.safetensors", 1000, "weights"), _file("tokenizer.json", 0)])
    present = {**_present(complete), **_present(no_shard, skip=("model.safetensors",)), **_present(zero_missing, skip=("tokenizer.json",))}
    context = _context(tmp_path, room=_room(tmp_path, hub=hub), cache=_fake_cache(hub, present))
    packages = [complete, no_shard, nothing, zero_missing]

    result = _call("vllm", {"packages": [package_ref(p) for p in packages]}, context)[0]["result"]

    assert result["missing_bytes"] == [0, 1000, 1000 + 50 + 30, None]


def test_only_the_active_ssh_computer_is_read_for_vllm(tmp_path, monkeypatch):
    """FR B4: of two stored SSH computers only the active one is asked for its vLLM; the other, not connected
    in this run, is not in the live runtimes either."""
    from modelroom.guided_remote_vllm import live_map
    from modelroom.remote_vllm import RemoteVllm

    ssh = FakeSsh(context=_context(tmp_path, vllm=_serving()))
    run = _Run(ssh, ssh_active="gpu2")
    run.remote_vllm["gpu1"] = RemoteVllm(state="serving")  # stored earlier in the run, not connected now
    _hello(run, "gpu2")

    _offer(run, [_small()], monkeypatch)

    assert {call[0] for call in ssh.calls} == {"gpu2"} and [call[1] for call in ssh.calls] == ["hello", "vllm"]
    assert set(live_map(run)) == {"gpu2"}


def test_a_connection_lost_mid_vllm_measure_is_one_line(tmp_path):
    """B1: the connection ends after a progress line: one fixed sentence, `run.failed`, nothing kept."""
    from modelroom.guided_remote_vllm import RemoteOffer, measure
    from modelroom.remote_vllm import RemoteVllm
    from modelroom.vllm_server import ServedModel

    package = _small()
    line = b'{"protocol":1,"progress":{"stage":"warmup","number":1,"of":1}}'
    run = _Run(FaultSsh(ProcessOutcome(255), lines=(line,)))
    run.remote_vllm["gpu1"] = RemoteVllm(state="serving", packages=(package,), served_index=0, served_context=4096)
    offer = RemoteOffer("gpu1", 0, package, ServedModel(MODEL, MODEL, 4096), 20, None)

    assert measure(run, offer, _scenario()) is None

    assert run.problems == ["gpu1 could not be reached over SSH; try ssh gpu1 yourself to see why"]
    assert run.remote_vllm["gpu1"].outcome is None


def test_vllm_on_the_server_asking_a_key_is_one_note(tmp_path, monkeypatch):
    """E5, R3: a server that asks for a key is said so once, offers no row and does not count as detected."""
    import modelroom.guided_remote_vllm as grv
    from modelroom.runtimes import present_runtimes

    run = _Run(FakeSsh(context=_context(tmp_path, vllm=_asks_key())))
    assert _hello(run)["vllm"] == "asks_key"

    assert _offer(run, [_small()], monkeypatch) is None

    assert run.screen.notes == ["the vLLM server on gpu1 asks for an API key; modelroom reads an open server only"]
    assert run.problems == [] and grv.live_map(run) == {}
    assert "vllm" not in present_runtimes(_profile().model_copy(update={"runtimes": ["ollama"]}), grv.live_map(run).get("gpu1"))


# --- the commands for gpu1: printed, never run ---------------------------------------------------------------------------


def _gpu1_profile(**changes):
    return _profile().model_copy(update={"display_name": "gpu1", **changes})


def _document(profile, packages, machine: str = "gpu1"):
    """A render document of `profile` as machine `machine` (the key and the label of a computer over SSH)."""
    from modelroom.config import Configuration
    from modelroom.contracts import Area, Snapshot
    from modelroom.render import MachineProfile, build_render_document

    base = _base_model()
    run_at = datetime(2026, 9, 30, 8, 0, 0, tzinfo=timezone.utc)
    accounts = sorted({(p.repo or "").split("/", 1)[0] for p in packages if p.repo})
    config = Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [{"name": "fam", "base_models": [{"hf_repo": base.hf_repo}]}],
            "packagers": accounts, "publishers": [base.publisher],
            "machines": {machine: {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True, "profile": profile.profile_id}},
            "paths": {"state": str(FIXTURES / "state"), "markdown": str(FIXTURES / "docs" / "models.md")},
        }
    )
    areas = [Area(source="huggingface", base_model_hf_repo=base.hf_repo, packager=accounts[0], status="complete", last_success=run_at, error=None)]
    snapshot = Snapshot(schema_version=2, run_at=run_at, areas=areas, base_models=[base], packages=packages)
    profiles = {machine: MachineProfile(status="ranked", label=profile.display_name, profile=profile, reason=None)}
    scenario = Scenario(context_requested=32768, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=10)
    return build_render_document(config, snapshot, profiles, {}, scenario, run_at)


def _lines(document, packages, live: LiveRoom | None = None, live_vllm=None) -> list[str]:
    """The terminal view of the table of `gpu1`: this run is on another computer, `gpu1` is the active one over SSH."""
    from modelroom.views import RunView, terminal_blocks

    view = RunView(
        install_for="laptop", live={"gpu1": live} if live is not None else {}, packages=packages, ssh_active="gpu1",
        origins={"gpu1": "ssh"}, pull_other_machines=frozenset({"gpu1"}), live_vllm=live_vllm or {},
    )
    return [plain(line) for line in terminal_blocks(document, view=view)]


def _read_room(run: _Run, document) -> LiveRoom:
    from modelroom.guided_remote import read_room

    return read_room(run, "gpu1", document)


def _packages():
    return [_gguf_package(), _awq_package()]


def test_the_serve_command_for_gpu1_is_printed_not_run(tmp_path):
    """GF question F1 a: choosing a vLLM row of the active computer prints the two commands and starts nothing:
    no call over SSH, no yes/no question, no pull."""
    from modelroom.guided_install import PullContext
    from modelroom.pull_choice import ambiguous_names
    from modelroom.pull_other import pull_other_step

    packages = _packages()
    document = _document(_gpu1_profile(), packages)
    ssh = FakeSsh(context=_context(tmp_path))
    lines: list[str] = []
    run = _Run(ssh)
    run.asker = FileAsker({"pull_other": f"gpu1 vllm {AWQ_REPO}"})
    run.screen = Screen(lines.append, colored=False)
    run.out = lines.append

    pull_other_step(run, PullContext(document, "laptop", packages, ambiguous_names(packages)))

    text = "\n".join(lines)
    assert f"#2 on gpu1  {SERVE}" in text and DOCKER in text
    assert "modelroom pull" not in text and "ollama pull" not in text and "Pull #" not in text
    assert ssh.calls == []


def test_the_outgoing_request_carries_no_approval_by(tmp_path):
    """Codex r1-1: a package with a person's name in `approval.by` goes over SSH as a `PackageRef`, never whole."""
    from modelroom.remote_vllm import read_state, vllm_measure

    revision = "9" * 40
    package = _small(
        revision=revision, provenance="approved",
        approval=Approval(date=date(2026, 9, 30), content=revision, by=SENTINEL_PERSON),
    )
    ssh = FakeSsh(context=_context(tmp_path, vllm=_serving()))

    read_state("gpu1", [package], ssh, ssh_path="ssh")
    vllm_measure("gpu1", [package], 0, _scenario(), ssh, ssh_path="ssh")

    assert [call[1] for call in ssh.calls] == ["vllm", "vllm-measure"]
    for _alias, _op, stdin in ssh.calls:
        assert SENTINEL_PERSON.encode() not in stdin and b"approval" not in stdin and b"provenance" not in stdin


def test_the_package_ref_has_exactly_repo_revision_and_start_files():
    """R2: `package_ref` is `{"repo", "revision", "files"}`, the files exactly the start files (P2): weights, `*.json`,
    `tokenizer.model`, `*.tiktoken`, `*.jinja` -- not the README, not an image."""
    from modelroom.remote_vllm import package_ref

    files = [
        _file("model-00001-of-00002.safetensors", 10, "weights_shard"), _file("model-00002-of-00002.safetensors", 20, "weights_shard"),
        _file("config.json", 3), _file("tokenizer.model", 4), _file("vocab.tiktoken", 5), _file("chat.jinja", 6),
        _file("README.md", 7), _file("assets/logo.png", 8), _file(".gitattributes", 9),
    ]
    package = _small(files=files)

    ref = package_ref(package)

    assert set(ref) == {"repo", "revision", "files"} and ref["repo"] == MODEL and ref["revision"] == package.revision
    assert [file["name"] for file in ref["files"]] == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "config.json", "tokenizer.model", "vocab.tiktoken", "chat.jinja",
    ]
    assert all(set(file) == {"name", "size_bytes"} for file in ref["files"])
    assert package_ref(_small(files=[_file("../x.safetensors", 1, "weights")])) is None  # an unsafe name is not sent
    assert package_ref(_small(repo="Qü/Qwen")) is None


@pytest.mark.parametrize("op", ["vllm", "vllm-measure"])
@pytest.mark.parametrize(
    "change",
    [
        {"files": [{"name": "/etc/passwd", "size_bytes": 1}]},
        {"files": [{"name": "../x", "size_bytes": 1}]},
        {"files": [{"name": "\\\\host\\share\\x", "size_bytes": 1}]},
        {"files": [{"name": "C:\\x", "size_bytes": 1}]},
        {"files": [{"name": "a/../../x.json", "size_bytes": 1}]},
        {"repo": "../x"},
        {"repo": "a/b;rm"},
        {"revision": "main"},
        {"revision": "A" * 40},
        {"revision": "1" * 39},
        {"extra": True},
    ],
    ids=["absolute", "dotdot", "unc", "drive", "nested dotdot", "repo dotdot", "repo shell", "branch", "upper hex", "short", "extra field"],
)
def test_a_manipulated_ref_is_bad_request_without_cache_access(tmp_path, op, change):
    """Codex r1-2: the server checks every field again before any file is touched; a trap on every `stat` and
    `realpath` proves none is. `bad_request`, nothing else."""
    from modelroom.hf_cache import CacheProbes
    from modelroom.remote_vllm import package_ref

    touched: list[str] = []

    def trap(path: str):
        touched.append(path)
        raise AssertionError(f"the cache was touched: {path}")

    ref = {**package_ref(_small()), **change}
    payload = {"packages": [ref]} if op == "vllm" else {"packages": [ref], "package": 0, "scenario": {"context_requested": 4096, "requests": 20}}
    server = _serving()
    context = _context(tmp_path, vllm=server, cache=CacheProbes(realpath=trap, stat=trap))

    assert _call(op, payload, context) == [{"protocol": 1, "error": "bad_request"}]
    assert touched == [] and server.calls == []


def test_a_link_outside_the_repo_cache_on_the_server_is_not_on_disk(tmp_path):
    """Codex r1-2, I3: a start file whose resolved path leaves the repo's folder is unknown (`null`): no `stat`
    is made of it, and this computer counts the whole download, the package is not on disk."""
    from modelroom.hf_cache import CacheProbes
    from modelroom.remote_vllm import RemoteVllm, read_state

    hub = tmp_path / "hub"
    package = _small()
    stats: list[str] = []
    inner = _fake_cache(hub, _present(package), stats)
    outside = str(tmp_path / "outside" / "secret")
    probes = CacheProbes(realpath=lambda path: outside if path.endswith("model.safetensors") else path, stat=inner.stat)
    ssh = FakeSsh(context=_context(tmp_path, vllm=_serving(), room=_room(tmp_path, hub=hub), cache=probes))

    state = read_state("gpu1", [package], ssh, ssh_path="ssh")

    assert state.missing_bytes == (None,) and stats == []
    needs, held = RemoteVllm(packages=(package,), missing_bytes=state.missing_bytes).needs()
    assert held == frozenset() and list(needs.values()) == [1000 + 50 + 30]


# --- Rev 3: the bytes still to fetch decide the disk rule -----------------------------------------------------------------------


def _ab():
    """Two tensor packages of one base model: A with every shard and no tokenizer (11 MB declared), B nowhere in the cache (6.1 GB)."""
    def files(tokenizer: int):
        return [
            _file("model-00001-of-00002.safetensors", 3_050_000_000, "weights_shard"),
            _file("model-00002-of-00002.safetensors", 3_050_000_000, "weights_shard"),
            _file("tokenizer.json", tokenizer), _file("config.json", 1000),
        ]

    a = _awq_package(repo="RedHatAI/Qwen3-8B-quantized.w4a16", revision="a" * 40, files=files(11_000_000), quantization="AWQ")
    b = _awq_package(repo="RedHatAI/Qwen3-8B-FP8-dynamic", revision="b" * 40, files=files(12_000_000), quantization="FP8")
    return a, b


def _ranks(document, *packages) -> list[int]:
    from modelroom.quantization import package_identity_key

    entries = {tuple(entry.package_identity): entry.rank for entry in document.machines[0].ranked}
    return [entries[package_identity_key(package)] for package in packages]


def test_remote_tokenizer_missing_and_all_missing_differ_at_the_same_free_space(tmp_path):
    """Codex r2-6: the same 2 GB free at the server. A has every shard and misses its tokenizer (about 11 MB to
    fetch): its row is not marked, its command carries no disk warning. B is nowhere in the cache (6.1 GB):
    its row is marked as in U11 and cannot be chosen from the list."""
    from modelroom.guided_remote import read_room
    from modelroom.pull_choice import ambiguous_names
    from modelroom.pull_other import other_rows, pull_other_choices
    from modelroom.remote_vllm import read_state

    a, b = _ab()
    hub = tmp_path / "hub"
    context = _context(
        tmp_path, vllm=_silent(), room=_room(tmp_path, hub=hub, free=2 * GIB), cache=_fake_cache(hub, _present(a, skip=("tokenizer.json",))),
        is_file=lambda path: True,  # installed
    )
    run = _Run(FakeSsh(context=context))
    _hello(run)
    run.remote_vllm["gpu1"] = read_state("gpu1", [a, b], run.ssh, ssh_path="ssh")
    assert run.remote_vllm["gpu1"].missing_bytes == (11_000_000, 3_050_000_000 * 2 + 12_000_000 + 1000)
    document = _document(_gpu1_profile(), [_gguf_package(), a, b])
    live = read_room(run, "gpu1", document)

    lines = [line.strip() for line in _lines(document, [a, b], live, {"gpu1": run.remote_vllm["gpu1"].live()})]

    rank_a, rank_b = _ranks(document, a, b)
    row = lambda rank: next(line for line in lines if line.startswith(f"{rank} "))  # noqa: E731
    assert not row(rank_a).endswith("×") and row(rank_b).endswith("×")
    install = [line for line in lines if line.startswith("install")]
    assert any(f"#{rank_a} on gpu1" in line and a.repo in line for line in install)
    assert not any(f"#{rank_b} on gpu1" in line for line in install)
    disk_lines = [line for line in lines if line.startswith("disk") and "×" in line]
    assert any(f"#{rank_b}" in line and "2.0 GB free on gpu1" in line for line in disk_lines)
    assert not any(f"#{rank_a}" in line for line in disk_lines)  # the red `disk` line names B alone; A has no warning
    packages = [a, b]
    rows = other_rows(document, "laptop", ambiguous_names(packages), "·", packages, {"gpu1": run.remote_vllm["gpu1"].disk})
    choices = {entry.row.rank: choice for entry, choice in zip(rows, pull_other_choices(rows)[1:])}
    assert choices[rank_a].disabled is None and choices[rank_b].disabled is not None and "plus 1 GB room, 2.0 GB free" in choices[rank_b].disabled


def test_missing_bytes_carry_no_names(tmp_path):
    """Codex r2-6: the answer is a list of numbers; a file name, a repo folder or a path of the server's cache that
    carries a sentinel never reaches the wire."""
    from modelroom.remote_vllm import package_ref

    package = _small()
    hub = tmp_path / "sentinel-host" / "hub"
    snapshot = hub / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / package.revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"x" * 1000)
    (snapshot / "sentinel-host.bin").write_bytes(b"x")  # a file of the cache the request never named
    ssh = FakeSsh(context=_context(tmp_path, room=_room(tmp_path, hub=hub)))

    lines = _call("vllm", {"packages": [package_ref(package)]}, ssh.context)

    assert lines[0]["result"]["missing_bytes"] == [50 + 30]
    assert "sentinel" not in json.dumps(lines) and str(tmp_path) not in json.dumps(lines)


def test_a_missing_bytes_list_of_wrong_length_or_range_is_a_form_fault():
    """Codex r2-6, B1: the laptop checks the list against what it sent: as many entries as packages, each `null` or
    between `0` and the sum of that package's start files."""
    from modelroom.remote_vllm import read_state

    package = _small()  # start files: 1000 + 50 + 30
    ok = {"state": "installed", "served": None}

    def read(**answer):
        line = json.dumps({"protocol": 1, "result": {**ok, **answer}}).encode("utf-8")
        return read_state("gpu1", [package], FaultSsh(ProcessOutcome(0), lines=(line,)), ssh_path="ssh")

    assert read(missing_bytes=[0]).missing_bytes == (0,) and read(missing_bytes=[1080]).missing_bytes == (1080,)
    assert read(missing_bytes=[None]).missing_bytes == (None,)
    for bad in ([], [0, 0], [1081], [-1], ["5"], [1.5], [True]):
        with pytest.raises(SshError, match=FORM):
            read(missing_bytes=bad)
    with pytest.raises(SshError, match=FORM):  # a served model only where the server serves; an index only of what was sent
        read(missing_bytes=[0], served={"package": 0, "context": 4096})
    with pytest.raises(SshError, match=FORM):
        read(state="serving", missing_bytes=[0], served={"package": 1, "context": 4096})


def test_missing_bytes_null_uses_the_whole_download():
    """R2: `null` is the whole download of the package's start files for the disk rule (U11): not held, not less."""
    from modelroom.disk_fit import over_disk
    from modelroom.remote_vllm import RemoteVllm
    from modelroom.quantization import package_identity_key

    package = _small()
    state = RemoteVllm(state="installed", packages=(package,), missing_bytes=(None,))

    needs, held = state.needs()

    identity = package_identity_key(package)
    assert needs == {identity: 1000 + 50 + 30} and held == frozenset()
    disk = Disk(1000 + 50 + 30 + GIB - 1, "gpu1", needs=needs)
    assert over_disk(1000 / GIB, disk, identity) is True
    assert over_disk(1000 / GIB, Disk(1000 + 50 + 30 + GIB, "gpu1", needs=needs), identity) is False
    zero = RemoteVllm(state="installed", packages=(package,), missing_bytes=(0,))
    assert zero.needs()[1] == frozenset({identity}) and zero.needs()[0] == {identity: 0}


def test_local_and_remote_use_the_same_missing_bytes_function(tmp_path, monkeypatch):
    """Codex r2-6, r3-6: one function counts the bytes still to fetch -- this computer's disk rule and the server's
    `vllm` operation both call `hf_cache.missing_bytes`; a spy on it sees both."""
    from modelroom import hf_cache
    from modelroom.remote_vllm import package_ref

    package = _small()
    seen: list[str] = []
    original = hf_cache.missing_bytes

    def spy(ref, cache, probes=None):
        seen.append(ref.repo)
        return original(ref, cache, probes)

    monkeypatch.setattr(hf_cache, "missing_bytes", spy)
    hub = tmp_path / "hub"

    local = hf_cache.vllm_disk(_room(tmp_path, hub=hub), [package])
    assert seen == [MODEL] and local.needs[next(iter(local.needs))] == 1000 + 50 + 30

    remote = _call("vllm", {"packages": [package_ref(package)]}, _context(tmp_path, room=_room(tmp_path, hub=hub)))
    assert seen == [MODEL, MODEL] and remote[0]["result"]["missing_bytes"] == [1000 + 50 + 30]


# --- Rev 4 -----------------------------------------------------------------------------------------------------------------------


def test_remote_zero_size_tokenizer_missing_is_null_not_on_disk(tmp_path):
    """Codex r3-6: every weight is there, `tokenizer.json` (declared size 0) is not. The server says `null`, not `0`:
    this computer prints no `on disk already` and the disk rule counts the whole download."""
    from modelroom.guided_remote import read_room
    from modelroom.remote_vllm import read_state

    package = _awq_package(
        files=[_file("model.safetensors", 6_098_581_864, "weights"), _file("tokenizer.json", 0), _file("config.json", 1000)],
    )
    hub = tmp_path / "hub"
    context = _context(
        tmp_path, vllm=_silent(), room=_room(tmp_path, hub=hub, free=2 * GIB), is_file=lambda path: True,
        cache=_fake_cache(hub, _present(package, skip=("tokenizer.json",))),
    )
    run = _Run(FakeSsh(context=context))
    _hello(run)
    state = run.remote_vllm["gpu1"] = read_state("gpu1", [package], run.ssh, ssh_path="ssh")
    assert state.missing_bytes == (None,)
    document = _document(_gpu1_profile(), [package])
    live = read_room(run, "gpu1", document)

    lines = [line.strip() for line in _lines(document, [package], live, {"gpu1": state.live()})]

    assert not any("on disk already" in line for line in lines)
    (rank,) = _ranks(document, package)
    assert next(line for line in lines if line.startswith(f"{rank} ")).endswith("×")  # the whole download against 2 GB
    assert state.needs()[0][next(iter(state.needs()[0]))] == 6_098_581_864 + 1000


# --- Rev 5, Rev 6 -------------------------------------------------------------------------------------------------------------------


def _gpu1_view(tmp_path, environ: dict, platform: str = "linux", installed: bool = True, profile=None):
    """The table of `gpu1` for a server whose environment is `environ`: its profile as the server would send it."""
    from modelroom.remote_vllm import read_state

    packages = _packages()
    room = _room(tmp_path, environ=environ, system="linux", which=lambda name: "/usr/bin/vllm" if name == "vllm" and installed else None)
    context = _context(tmp_path, vllm=_serving(AWQ_REPO), platform=platform, room=room)
    run = _Run(FakeSsh(context=context), models=("Qwen/Qwen3-8B",))
    info = _hello(run)
    run.remote_vllm["gpu1"] = read_state("gpu1", [_awq_package()], run.ssh, ssh_path="ssh")
    document = _document(profile if profile is not None else _gpu1_profile(), packages)
    live = _read_room(run, document)
    lines = [line.strip() for line in _lines(document, packages, live, {"gpu1": run.remote_vllm["gpu1"].live()})]
    return info, run, document, lines


def test_remote_prints_both_lines_for_gpu1_from_its_profile(tmp_path):
    """Rev 5 (F2 b): the two commands under the table of `gpu1` come from the profile the server sent -- its cards,
    its context -- with `on gpu1` in the label; no server path, the cache only as the shell expression."""
    info, _run_, _document_, lines = _gpu1_view(tmp_path, {"HF_HOME": "/srv/hf"})

    assert info["vllm"] == "serving"
    start = next(index for index, line in enumerate(lines) if line.startswith("install   #2 on gpu1  "))
    assert lines[start] == f"install   #2 on gpu1  {SERVE}"
    assert lines[start + 1] == f"docker    {DOCKER}"
    assert "on that computer, in a terminal that stays open" in lines[start + 2]
    text = "\n".join(lines)
    assert EXPRESSION in text and "/srv/hf" not in text and str(tmp_path) not in text and "C:\\" not in text
    assert "--mount \"type=bind,src=" + EXPRESSION + ",dst=/root/.cache/huggingface/hub\"" in text  # the fixed container path only


def test_remote_relative_cache_leaves_out_the_docker_line(tmp_path):
    """Rev 6: `room.model_disk.relative` at the server -- the Hugging Face cache is set as a relative path there:
    the note instead of the Docker line for `gpu1`; the `vllm serve` line stays, and the handover agrees."""
    from modelroom.guided_install import PullContext
    from modelroom.pull_choice import ambiguous_names
    from modelroom.pull_other import pull_other_step

    _info, run, document, lines = _gpu1_view(tmp_path, {"HF_HOME": "hf-cache"})

    assert run.remote_vllm["gpu1"].relative is True
    start = next(index for index, line in enumerate(lines) if line.startswith("install   #2 on gpu1  "))
    assert lines[start] == f"install   #2 on gpu1  {SERVE}"
    assert RELATIVE_NOTE in lines[start + 1] and not any(line.startswith("docker    ") for line in lines)
    out: list[str] = []
    run.asker, run.screen, run.out = FileAsker({"pull_other": f"gpu1 vllm {AWQ_REPO}"}), Screen(out.append, colored=False), out.append

    packages = _packages()
    pull_other_step(run, PullContext(document, "laptop", packages, ambiguous_names(packages)))

    assert RELATIVE_NOTE in "\n".join(out) and "docker --host" not in "\n".join(out)


def test_remote_non_linux_server_gets_neither_line(tmp_path):
    """Rev 6: `hello.system = win32` -- vLLM is `absent` there, whatever else is on the server (a serving
    server, `vllm` in the PATH); its profile names no vLLM, so its table has no vLLM row and prints neither line."""
    context = _context(
        tmp_path, vllm=_serving(AWQ_REPO), platform="win32", room=_room(tmp_path, which=lambda name: "C:/vllm.exe"),
    )
    hello = _call("hello", {}, context)[0]["result"]
    assert hello["system"] == "win32" and hello["vllm"] == "absent"
    profile = _call("profile", {"profile_id": "3f9a0c21d4e6b870", "machine_class": "server"}, context)[0]["result"]["export"]["profile"]
    assert profile["runtimes"] == ["ollama"]
    linux = _context(tmp_path, vllm=_serving(AWQ_REPO), room=_room(tmp_path, which=lambda name: "/usr/bin/vllm"))
    assert _call("profile", {"profile_id": "3f9a0c21d4e6b870", "machine_class": "server"}, linux)[0]["result"]["export"]["profile"]["runtimes"] == ["ollama", "vllm"]

    info, run, _document_, lines = _gpu1_view(tmp_path, {}, platform="win32", profile=_gpu1_profile(runtimes=["ollama"]))

    assert info["vllm"] == "absent" and info["system"] == "win32"
    text = "\n".join(lines)
    assert "vllm serve" not in text and "docker" not in text and "vLLM" not in text


# --- Codex r1 of SV2-R (fix round r1) -----------------------------------------------------------------------------------------


def _ollama_sig(rows):
    return [(r.machine, r.row.rank, r.row.runtime, r.row.locked, r.row.over_disk, r.row.too_big) for r in rows if r.row.runtime == "ollama"]


def _install_text(block, packages, disk, over_ssh):
    from modelroom.views_install import install_lines

    lines = install_lines(block, "laptop", frozenset(), disk, "-", frozenset({"gpu1"}), packages, None, None, over_ssh)
    return [plain(line) for line in lines]


def test_the_ssh_disk_rule_leaves_ollama_rows_as_in_fr():
    """R1-1: over SSH the disk rule is for the vLLM rows of the active computer (R3 step 5). A computer whose
    Ollama rows meet a far too small `room.disk` shows the same install line, and the same list rows, as in FR."""
    from modelroom.pull_choice import ambiguous_names
    from modelroom.pull_other import other_rows

    packages = [_gguf_package()]
    document = _document(_gpu1_profile(runtimes=["ollama"]), packages)
    block = document.machines[0]
    assert block.ranked
    tiny, huge = Disk(1000, "gpu1"), Disk(500 * GIB, "gpu1")

    fr = _install_text(block, packages, tiny, False)  # the basis: another machine's block never reads a disk
    assert any("ollama pull" in line for line in fr)
    assert _install_text(block, packages, tiny, True) == fr
    assert _install_text(block, packages, huge, True) == fr

    ambiguous = ambiguous_names(packages)
    plain_rows = other_rows(document, "laptop", ambiguous, "·", packages, {})
    assert plain_rows and not any(r.row.over_disk or r.row.locked for r in plain_rows)
    assert _ollama_sig(other_rows(document, "laptop", ambiguous, "·", packages, {"gpu1": tiny})) == _ollama_sig(plain_rows)


def test_the_ssh_disk_rule_marks_vllm_rows_only_in_a_mixed_list():
    """R1-1: on a computer with Ollama and vLLM rows a small disk locks the vLLM rows by `room.model_disk`, the Ollama
    rows stay what they are without it."""
    from modelroom.pull_choice import ambiguous_names
    from modelroom.pull_other import other_rows

    packages = _packages()
    document = _document(_gpu1_profile(), packages)
    ambiguous = ambiguous_names(packages)
    disk = Disk(1000, "gpu1", vllm=Disk(1000, "gpu1"))

    rows = other_rows(document, "laptop", ambiguous, "·", packages, {"gpu1": disk})
    without = other_rows(document, "laptop", ambiguous, "·", packages, {})

    assert _ollama_sig(rows) == _ollama_sig(without) and any(r.row.runtime == "ollama" for r in rows)
    vllm_rows = [r for r in rows if r.row.runtime == "vllm"]
    assert vllm_rows and all(r.row.over_disk for r in vllm_rows)


def _measure_wire(wire: dict, scenario: Scenario):
    from modelroom.remote_vllm import vllm_measure

    line = json.dumps({"protocol": 1, "result": {"measurement": wire}}).encode("utf-8")
    return vllm_measure("gpu1", [_small()], 0, scenario, FaultSsh(ProcessOutcome(0), lines=(line,)), ssh_path="ssh")


def _speeds_wire(**changes) -> dict:
    speeds = {name: 1.5 for name in ("total_tps", "total_min", "total_max", "request_tps", "request_min", "request_max")}
    return {"requests": 20, "ranking_requests": 20, "context": 4096, **speeds, "problem": None, **changes}


@pytest.mark.parametrize(
    "scenario, change",
    [
        (_scenario(requests=128), {"requests": 64, "ranking_requests": 128}),  # fewer requests measured, no code
        (_scenario(), {"problem": "other_requests"}),  # the code without the numbers
        (_scenario(), {"context": 2048}),  # a shorter context, no code
        (_scenario(), {"problem": "other_context"}),  # the code without the numbers
        (_scenario(requests=128), {"requests": 64, "ranking_requests": 128, "problem": "other_context", "context": 4096}),
    ],
    ids=["requests no code", "other_requests no numbers", "context no code", "other_context no numbers", "wrong code for requests"],
)
def test_a_measurement_that_contradicts_its_own_scenario_is_a_form_fault(scenario, change):
    """R1-2: what this computer can check from its own scenario it checks -- the numbers and the code must agree."""
    with pytest.raises(SshError, match=FORM):
        _measure_wire(_speeds_wire(**change), scenario)


def test_a_measurement_whose_numbers_and_code_agree_is_taken():
    """R1-2: the same deviations, stated with their code, are not comparable -- and nothing else changes."""
    short = _measure_wire(_speeds_wire(requests=64, ranking_requests=128, problem="other_requests"), _scenario(requests=128))
    assert short.comparable is False and short.comparable_reason.code == "other_requests" and short.validity == "valid"
    other = _measure_wire(_speeds_wire(context=2048, problem="other_context"), _scenario())
    assert other.comparable is False and other.comparable_reason.code == "other_context"
    fine = _measure_wire(_speeds_wire(), _scenario())
    assert fine.comparable is True and fine.validity == "valid"
    nothing = {name: None for name in ("total_tps", "total_min", "total_max", "request_tps", "request_min", "request_max")}
    masked = _measure_wire(_speeds_wire(requests=64, ranking_requests=128, problem="invalid_tokens", **nothing), _scenario(requests=128))
    assert masked.validity == "invalid"  # an invalid answer names its own rule first, whatever else deviates


def _server_without_model_disk(tmp_path, missing: int | None):
    """The active computer with the vLLM package of the run, a server of the old form (`room` has no `model_disk`),
    `missing` the bytes still to fetch it counted: `0` every start file there; `None` every weight there and the
    tokenizer of declared size 0 missing (Rev 4), so the server cannot say -- the answer is checked to be exactly that."""
    from modelroom.guided_remote_vllm import with_model_disk
    from modelroom.remote_vllm import read_state

    if missing == 0:
        package = _awq_package()
        present = _present(package)
    else:
        package = _awq_package(
            files=[_file("model.safetensors", 6_098_581_864, "weights"), _file("tokenizer.json", 0), _file("config.json", 1000)],
        )
        present = _present(package, skip=("tokenizer.json",))
    hub = tmp_path / "hub"
    context = _context(
        tmp_path, vllm=_silent(), room=_room(tmp_path, hub=hub, free=50 * GIB), is_file=lambda path: True, cache=_fake_cache(hub, present),
    )
    run = _Run(FakeSsh(context=context))
    _hello(run)
    state = run.remote_vllm["gpu1"] = read_state("gpu1", [package], run.ssh, ssh_path="ssh")
    assert state.missing_bytes == (missing,)
    disk = with_model_disk(run, "gpu1", Disk(50 * GIB, "gpu1"), None)
    return run, package, state, disk


def test_a_server_without_model_disk_still_gives_on_disk_already(tmp_path):
    """R1-3: `missing_bytes=[0]` counts without `room.model_disk`: the cache result is taken whatever the room said,
    vLLM's place of unknown free space -- `on disk already` in the table and in the handover."""
    from modelroom.guided_install import PullContext
    from modelroom.pull_choice import ambiguous_names
    from modelroom.pull_other import pull_other_step

    run, package, state, disk = _server_without_model_disk(tmp_path, 0)

    assert disk.vllm is not None and disk.vllm.known is False and disk.vllm.held == state.needs()[1] != frozenset()
    packages = _packages()
    document = _document(_gpu1_profile(), packages)
    lines = [line.strip() for line in _lines(document, packages, LiveRoom(disk=disk), {"gpu1": state.live()})]
    assert any("on disk already" in line for line in lines)
    out: list[str] = []
    run.asker, run.screen, run.out = FileAsker({"pull_other": f"gpu1 vllm {AWQ_REPO}"}), Screen(out.append, colored=False), out.append
    pull_other_step(run, PullContext(document, "laptop", packages, ambiguous_names(packages)))
    assert "on disk already" in "\n".join(out)


def test_a_server_without_model_disk_keeps_the_whole_download_unknown_disk(tmp_path):
    """R1-3: `null` (not determinable) is the whole download against a place of unknown free space: nothing marked,
    no `on disk already`, the needs are kept and the room line says the place is unknown."""
    from modelroom.remote_vllm import download_bytes

    run, package, state, disk = _server_without_model_disk(tmp_path, None)

    assert disk.vllm is not None and disk.vllm.known is False and disk.vllm.held == frozenset()
    assert list(disk.vllm.needs.values()) == [download_bytes(package)]
    packages = _packages()
    document = _document(_gpu1_profile(), packages)
    lines = [line.strip() for line in _lines(document, packages, LiveRoom(disk=disk), {"gpu1": state.live()})]
    assert not any("on disk already" in line for line in lines)
    assert any("disk unknown on gpu1" in line for line in lines)


@pytest.mark.parametrize("packages", [[], [_gguf_package()]], ids=["no packages", "no tensor package"])
def test_a_key_asking_server_is_one_note_without_a_sendable_package(tmp_path, monkeypatch, packages):
    """R1-4: `hello.vllm = asks_key` is known; the note is said although the run has no tensor package to send --
    and `vllm` is not called for it."""
    run = _Run(FakeSsh(context=_context(tmp_path, vllm=_asks_key())))
    assert _hello(run)["vllm"] == "asks_key"

    assert _offer(run, packages, monkeypatch) is None

    assert run.screen.notes == ["the vLLM server on gpu1 asks for an API key; modelroom reads an open server only"]
    assert [call[1] for call in run.ssh.calls] == ["hello"]


def test_a_mixed_table_says_where_vllm_keeps_its_models_when_the_server_did_not(tmp_path):
    """R1-5: Ollama and vLLM both shown, the server sent no `model_disk`: Ollama's piece and `disk unknown on gpu1`."""
    from modelroom.room_head import head_pieces

    document = _document(_gpu1_profile(), _packages())
    live = LiveRoom(disk=Disk(50 * GIB, "gpu1"))

    pieces = head_pieces(document.machines[0], live, "ssh", frozenset({"ollama", "vllm"}))

    assert "disk unknown on gpu1" in pieces
    assert any("free on gpu1" in piece for piece in pieces if piece != "disk unknown on gpu1")
