"""FR-B2: measurement on the server (`candidates`/`measure`), and `guided_remote_measure.py`'s own
glue -- step 4's second question round for the active SSH target.

Server-level tests (`_context`/`_call`/`_one`, mirroring `tests/test_remote_server.py`) prove the
digest rule and the sink run at the server, never with an installed name. `guided_remote_measure`
tests use `_FakeRun`, a minimal stand-in that carries no `.probes` attribute at all -- proving the
whole round trip never reads a laptop-side hardware probe, because nothing here could.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import modelroom.ssh_process as ssh_process
from fixture_room import fixed_room
from fixture_ssh import FaultSsh, FakeSsh
from fixture_support import (
    DAEMON_OBSERVED_AT,
    GRANITE_MANIFEST_DIGEST,
    LAPTOP_RAM_BYTES,
    build_transport,
    deepseek_package,
    default_generates,
    granite_ollama_package,
    guided_transport_mapping,
    json_response,
    loadtest_daemon,
    offline_daemon,
    offline_servers,
    ps_answer,
    windows_probes,
    windows_runner,
)

import modelroom.guided_remote_measure as grm
from modelroom.config import load_config
from modelroom.contracts import Package
from modelroom.daemon import GENERATE_PATH, PS_PATH, TAGS_PATH, VERSION_PATH, FixtureDaemon
from modelroom.dialog import AnswerMissingError
from modelroom.guided import run_guided
from modelroom.http import Response
from modelroom.loadtest import GPU_UTILIZATION_ARGS, installed_candidates, installed_models, run_load_test, weight_digest
from modelroom.measure import Probes, WINDOWS_MACHINE_GUID_ARGS
from modelroom.measurements import default_scenario, measurement_path, read_measurements
from modelroom.quantization import package_identity_key
from modelroom.remote_client import FORM_ERROR
from modelroom.remote_protocol import OP_BUDGETS, OpBudget
from modelroom.remote_server import ServerContext, main as server_main
from modelroom.ssh import ProcessOutcome, SshError, SubprocessSsh, call

from test_guided import DEEPSEEK, DEEPSEEK_OLLAMA_NAME, FULL_ANSWERS, QWEN_GGUF, UNSLOTH, _WatchingAsker, _pointer, _results
from test_guided import RUN1 as GUIDED_RUN1
from test_guided import RUN2 as GUIDED_RUN2

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)
PROFILE_ID = "3f9a0c21d4e6b870"


# --- server-level: candidates/measure at the counterpart (mirrors tests/test_remote_server.py) -----


def _probes() -> Probes:
    return Probes(
        platform="linux",
        runner=lambda a: (_ for _ in ()).throw(FileNotFoundError(a[0])),
        read_text=lambda p: (_ for _ in ()).throw(OSError("no such file")),
        memory_bytes=lambda: 0,
    )


def _context(**kw) -> ServerContext:
    daemon = kw.pop("daemon", None) or loadtest_daemon()
    base = dict(probes=_probes(), room=fixed_room(), daemon=daemon, now=lambda: NOW, environ={})
    base.update(kw)
    return ServerContext(**base)


def _call(op: str, payload: dict, context: ServerContext) -> tuple[list[bytes], int]:
    lines: list[bytes] = []
    code = server_main(op, json.dumps(payload).encode(), lines.append, context)
    return lines, code


def _one(lines: list[bytes]) -> dict:
    assert len(lines) == 1
    return json.loads(lines[0])


def _measure_payload(identity: tuple, package: Package, profile_id: str = PROFILE_ID) -> dict:
    return {
        "package_identity": list(identity),
        "packages": [package.model_dump(mode="json")],
        "scenario": default_scenario().model_dump(mode="json"),
        "profile_id": profile_id,
    }


def test_candidates_follow_the_digest_rule_on_the_server():
    """B2: identities only, never an installed name -- and cloud/unmatched as one count."""
    package = deepseek_package()
    context = _context()
    result = _one(_call("candidates", {"packages": [package.model_dump(mode="json")]}, context)[0])["result"]
    assert result["candidates"] == [{"package_identity": list(package_identity_key(package))}]
    assert result["set_aside"] == 1  # the fixture's one cloud entry, glm-5.3-flash:cloud
    assert "hf.co" not in json.dumps(result)  # no installed name, identity only


def test_an_unmatched_installed_model_counts_as_set_aside_too():
    """A model that names a configured repository but whose digest the server cannot prove is
    `unmatched`, counted the same way as a cloud entry (B2: one number, no names, no reasons)."""
    from modelroom.contracts import PackageFile

    wrong_digest = PackageFile(
        name="DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf", role="weights", size_bytes=1, digest="sha256:" + "1" * 64
    )
    package = deepseek_package(files=[wrong_digest])  # same repo, a digest `/api/show` will not confirm
    context = _context()
    result = _one(_call("candidates", {"packages": [package.model_dump(mode="json")]}, context)[0])["result"]
    assert result["candidates"] == []
    assert result["set_aside"] == 2  # the cloud entry, plus the now-unmatched deepseek entry


def test_measure_runs_the_protocol_at_the_server_and_sinks_the_record():
    package = deepseek_package()
    identity = package_identity_key(package)
    context = _context()
    lines, code = _call("measure", _measure_payload(identity, package), context)
    assert code == 0
    kinds = [next(iter(set(json.loads(line)) - {"protocol"})) for line in lines]
    assert kinds == ["progress", "progress", "progress", "progress", "result"]
    progress = [json.loads(line)["progress"] for line in lines[:-1]]
    assert progress[0] == {"stage": "warmup", "number": 1, "of": 1}
    assert progress[1] == {"stage": "run", "number": 1, "of": 3}
    assert progress[-1] == {"stage": "run", "number": 3, "of": 3}
    record = json.loads(lines[-1])["result"]["record"]
    assert record["profile_id"] == PROFILE_ID
    assert record["ollama_name"] == "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"


def test_a_candidate_no_longer_found_is_internal():
    """The identity the laptop asks for is no longer among the server's own candidates (its state
    changed between `candidates` and `measure`) -- `internal`, the one code left for this server's
    own fault, never a traceback."""
    package = deepseek_package()
    other_identity = ("huggingface", "sentinel/does-not-exist-GGUF", "does-not-exist.gguf")
    context = _context()
    error = _one(_call("measure", _measure_payload(other_identity, package), context)[0])
    assert error["error"] == "internal"


def test_candidates_and_measure_are_ollama_unreachable_when_tags_cannot_be_read():
    package = deepseek_package()
    context = _context(daemon=FixtureDaemon({}, unreachable="down"))
    error = _one(_call("candidates", {"packages": [package.model_dump(mode="json")]}, context)[0])
    assert error["error"] == "ollama_unreachable"
    error = _one(_call("measure", _measure_payload(package_identity_key(package), package), context)[0])
    assert error["error"] == "ollama_unreachable"


def test_a_candidates_request_with_an_extra_field_is_bad_request():
    context = _context()
    error = _one(_call("candidates", {"packages": [], "extra": True}, context)[0])
    assert error["error"] == "bad_request"


def test_a_candidates_request_without_packages_is_bad_request():
    """Codex r1 minor 1: `packages` is mandatory (`remote_protocol.CandidatesRequest`), an empty run
    sends `"packages": []`, it never omits the key -- a request without it at all is a form fault,
    not a silently assumed empty list."""
    context = _context()
    error = _one(_call("candidates", {}, context)[0])
    assert error["error"] == "bad_request"


def test_a_measure_request_without_packages_is_bad_request():
    """Codex r1 minor 1, `MeasureRequest.packages`."""
    package = deepseek_package()
    payload = _measure_payload(package_identity_key(package), package)
    del payload["packages"]
    context = _context()
    error = _one(_call("measure", payload, context)[0])
    assert error["error"] == "bad_request"


def test_a_candidates_result_missing_candidates_is_a_form_fault():
    """Codex r1 minor 1: `CandidatesResult.candidates` is mandatory on the laptop's own reading of
    the answer too -- a line with only `set_aside` is a form fault, mirroring
    `tests/test_ssh.py::test_a_result_that_fails_the_strict_op_model_is_a_form_fault`."""
    line = b'{"protocol":1,"result":{"set_aside":0}}'
    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "candidates", {"packages": []}, FaultSsh(ProcessOutcome(0), lines=(line,)))


# --- sentinels: no name of the server's own machine survives the sink (Codex r2-3) -----------------


def _sentinel_ollama_daemon(observations=None) -> FixtureDaemon:
    """`granite_ollama_package()`'s own manifest digest, installed at the server under the local
    name `sentinel-host:latest` -- the digest rule matches it regardless of that name (B2)."""
    tags = Response(
        200, {},
        json.dumps({"models": [{
            "name": "sentinel-host:latest", "model": "sentinel-host:latest",
            "size": 5_347_929_757, "digest": GRANITE_MANIFEST_DIGEST.removeprefix("sha256:"), "details": {},
        }]}).encode(),
    )
    default_obs = [ps_answer(digest=GRANITE_MANIFEST_DIGEST.removeprefix("sha256:"), name="sentinel-host:latest")] * 4
    return FixtureDaemon({
        ("GET", VERSION_PATH): json_response("ollama_version_local.json"),
        ("GET", TAGS_PATH): tags,
        ("GET", PS_PATH): [*(observations if observations is not None else default_obs), ps_answer()],
        ("POST", GENERATE_PATH): default_generates(),
    })


def test_a_registry_package_installed_as_sentinel_host_is_shown_by_its_package_name():
    package = granite_ollama_package()
    identity = package_identity_key(package)
    context = _context(daemon=_sentinel_ollama_daemon())

    cand_result = _one(_call("candidates", {"packages": [package.model_dump(mode="json")]}, context)[0])["result"]
    assert cand_result["candidates"] == [{"package_identity": list(identity)}]
    assert "sentinel" not in json.dumps(cand_result)

    lines, code = _call("measure", _measure_payload(identity, package), context)
    assert code == 0
    assert "sentinel" not in json.dumps([line.decode() for line in lines])
    record = json.loads(lines[-1])["result"]["record"]
    assert record["ollama_name"] == "granite4.2:8b"


def test_another_model_in_ps_named_sentinel_host_gives_the_fixed_reason():
    package = deepseek_package()
    identity = package_identity_key(package)
    daemon = loadtest_daemon(observations=[ps_answer(name="sentinel-host:latest")] * 4)
    context = _context(daemon=daemon)

    lines, code = _call("measure", _measure_payload(identity, package), context)

    assert code == 0
    record = json.loads(lines[-1])["result"]["record"]
    assert record["comparable"] is False
    assert record["comparable_reason"] == "another model was loaded during the measurement"
    assert "sentinel" not in json.dumps([line.decode() for line in lines])


def test_a_daemon_version_with_a_sentinel_is_null():
    package = deepseek_package()
    identity = package_identity_key(package)
    daemon = loadtest_daemon()
    daemon._answers[("GET", VERSION_PATH)] = Response(200, {}, b'{"version": "9.9.9-sentinel-host"}')
    context = _context(daemon=daemon)

    lines, code = _call("measure", _measure_payload(identity, package), context)

    assert code == 0
    record = json.loads(lines[-1])["result"]["record"]
    assert record["daemon_version"] is None
    assert "sentinel" not in json.dumps([line.decode() for line in lines])


# --- the transport budget: measure's own is 660s per line, no total (B1's own table) ---------------
#
# A real functional run over the FR-B1 process boundary (`_ScriptSsh`, `SubprocessSsh` against a
# small Python script), with a fake clock (`_FastClock`, the same one
# `test_ssh_process.py::test_a_generate_silent_for_more_than_60_s_still_succeeds` uses): one real
# second of `time.sleep` counts as 300 fake ones, so the script's own 1.0 s sleep between the
# `progress` and `result` line reads as 300 s of silence to `ssh_process`'s own per-line timer --
# real wall time stays under two seconds. `FakeSsh` cannot show this at all (no real time passes
# through it, B2's own report of 2026-09-30 named this a scope gap); this replaces that.


class _FastClock:
    """A fake clock for `ssh_process`: every real second counts as `factor` seconds of the limits."""

    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.origin = time.monotonic()

    def monotonic(self) -> float:
        return self.origin + (time.monotonic() - self.origin) * self.factor

    @staticmethod
    def sleep(seconds: float) -> None:
        time.sleep(seconds)


class _ScriptSsh:
    """A `SshRunner` whose "ssh" is a real child process running `script` (B1's own process seam)."""

    def __init__(self, script: Path) -> None:
        self.script = script

    def run(self, argv, stdin, total, per_line, limits, on_line):
        return SubprocessSsh().run([sys.executable, str(self.script), *argv[1:]], stdin, total, per_line, limits, on_line)


def _measure_script(tmp_path: Path, record_json: str) -> Path:
    """One `progress` line, a real 1.0 s sleep, then the `result` line -- `record_json` already a
    complete, valid `MeasurementRecord` so the answer passes `remote_protocol.validate_result`."""
    path = tmp_path / "fake_ssh_measure.py"
    body = (
        "import json, sys, time\n"
        "sys.stdin.buffer.read()\n"
        "print(json.dumps({'protocol': 1, 'progress': {'stage': 'warmup', 'number': 1, 'of': 1}}), flush=True)\n"
        "time.sleep(1.0)\n"
        f"print({record_json!r}, flush=True)\n"
    )
    path.write_bytes(body.encode("utf-8"))
    return path


def test_a_measurement_silent_for_more_than_60_s_succeeds(tmp_path, monkeypatch):
    """B1's own table: `measure` has 660 s **per line**, not the pull's 60 s -- 300 s of fake-clock
    silence between the warm-up's progress line and the result is no fault."""
    package = deepseek_package()
    record = _local_record(package, PROFILE_ID)
    result_line = json.dumps({"protocol": 1, "result": {"record": record.model_dump(mode="json")}})
    script = _measure_script(tmp_path, result_line)
    request = _measure_payload(package_identity_key(package), package)

    monkeypatch.setattr(ssh_process, "time", _FastClock(300.0))
    answer = call("gpu1", "measure", request, _ScriptSsh(script), ssh_path="ssh")

    assert answer["result"]["record"]["profile_id"] == PROFILE_ID


def test_a_60s_budget_would_time_the_identical_silence_out(tmp_path, monkeypatch):
    """The test above discriminates, it does not merely observe a call that never waits: with the
    pull's own 60 s per-line budget in place of `measure`'s 660 s, the identical 300 s (fake-clock)
    silence times out -- proof the 660 s budget, not the test's own leniency, is what succeeds
    above (abnahme note 2026-09-30, point 2)."""
    package = deepseek_package()
    record = _local_record(package, PROFILE_ID)
    result_line = json.dumps({"protocol": 1, "result": {"record": record.model_dump(mode="json")}})
    script = _measure_script(tmp_path, result_line)
    request = _measure_payload(package_identity_key(package), package)
    original = OP_BUDGETS["measure"]

    monkeypatch.setitem(OP_BUDGETS, "measure", OpBudget(original.total_seconds, 60.0, original.limits))
    monkeypatch.setattr(ssh_process, "time", _FastClock(300.0))

    with pytest.raises(SshError, match="gpu1 did not answer within 60 s"):
        call("gpu1", "measure", request, _ScriptSsh(script), ssh_path="ssh")


# --- guided_remote_measure.py: step 4's second round, with a real record but a fake wire -----------


def _local_record(package: Package, profile_id: str, daemon=None) -> "grm.MeasurementRecord":
    """A real measurement of `package`, made the same way `measure_result` makes one at the server
    -- used as ground truth for the laptop-side tests below, never sent over any wire."""
    daemon = daemon if daemon is not None else loadtest_daemon()
    installed, _reason = installed_models(daemon, NOW)
    inventory = installed_candidates([package], installed, lambda name: weight_digest(daemon, name))
    candidate = next(c for c in inventory.candidates if package_identity_key(c.package) == package_identity_key(package))
    return run_load_test(daemon, candidate, default_scenario(), profile_id, probes=windows_probes(), measured_at=NOW)


@dataclass
class _FakeScreen:
    notes: list = field(default_factory=list)
    answers: list = field(default_factory=list)
    glyphs: object = field(default_factory=lambda: SimpleNamespace(dot="·", skip="-"))

    def note(self, text: str) -> None:
        self.notes.append(text)

    def answer(self, question: str, value: str, **_kw) -> None:
        self.answers.append((question, value))


class _FakeRun:
    """A minimal stand-in for `GuidedRun`, carrying only what `guided_remote_measure.py` reads --
    deliberately **no** `.probes` attribute: any code path that touched one would raise
    `AttributeError` here, which is exactly how `test_measure_reads_load_on_the_server_not_here`
    proves the load is never read on this side."""

    def __init__(self, asker=None, ssh_active: str | None = "gpu1", pointer_path: Path | None = None):
        self.asker = asker
        self.ssh = object()  # never used directly: `ssh_candidates`/`ssh_measure` are stubbed below
        self.ssh_active = ssh_active
        self.pointer_path = pointer_path if pointer_path is not None else Path("home") / "guided.json"
        self.selection = None
        self.now = NOW
        self.screen = _FakeScreen()
        self.problems: list[str] = []

    def failed(self, message: str) -> None:
        self.problems.append(message)


class _TrapAsker:
    """Every call is a test failure: proves a round that should ask nothing really asks nothing."""

    def confirm(self, key, question, default=True):
        raise AssertionError(f"must not ask {key!r}: there is no candidate to measure")

    def checkbox(self, key, question, choices):
        raise AssertionError(f"must not ask {key!r}: there is no candidate to measure")


class _MissingAnswerAsker:
    """`confirm` behaves like `--answers` without the key (`AnswerMissingError`); `checkbox` must
    never be reached once that default (`no`) is taken."""

    def confirm(self, key, question, default=True):
        raise AnswerMissingError(key, question)

    def checkbox(self, key, question, choices):
        raise AssertionError("checkbox must not be reached: load_test_remote defaulted to no")


class _YesAsker:
    def __init__(self, picked: list[str]):
        self._picked = picked

    def confirm(self, key, question, default=True):
        return True

    def checkbox(self, key, question, choices):
        return list(self._picked)


def _shown(package: Package) -> str:
    from modelroom.pull_choice import package_pull_name

    return package_pull_name(package) or package.base_model_hf_repo


def test_no_candidate_asks_nothing(monkeypatch):
    monkeypatch.setattr(grm, "remote_aliases", lambda path: {PROFILE_ID: "gpu1"})
    monkeypatch.setattr(grm, "run_packages", lambda config, selection: [deepseek_package()])
    monkeypatch.setattr(grm, "ssh_candidates", lambda alias, packages, runner: {"candidates": [], "set_aside": 2})
    run = _FakeRun(asker=_TrapAsker())

    result = grm.remote_load_test(run, config=None, scenario=default_scenario())

    assert result is None
    assert run.problems == []
    assert run.screen.answers == []


def test_no_active_ssh_target_asks_nothing(monkeypatch):
    """B4: no active SSH target means no question -- `ssh_active` is `None`, so `candidates` is
    never even called."""
    monkeypatch.setattr(grm, "ssh_candidates", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no active target")))
    run = _FakeRun(asker=_TrapAsker(), ssh_active=None)

    result = grm.remote_load_test(run, config=None, scenario=default_scenario())

    assert result is None
    assert run.problems == []


def test_a_missing_answer_measures_nothing(monkeypatch):
    package = deepseek_package()
    identity = package_identity_key(package)
    monkeypatch.setattr(grm, "remote_aliases", lambda path: {PROFILE_ID: "gpu1"})
    monkeypatch.setattr(grm, "run_packages", lambda config, selection: [package])
    monkeypatch.setattr(
        grm, "ssh_candidates",
        lambda alias, packages, runner: {"candidates": [{"package_identity": list(identity)}], "set_aside": 0},
    )
    monkeypatch.setattr(grm, "ssh_measure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not measure")))
    run = _FakeRun(asker=_MissingAnswerAsker())

    result = grm.remote_load_test(run, config=None, scenario=default_scenario())

    assert result is None
    assert run.problems == []
    assert run.screen.answers == [("Measure the speed on gpu1 now?", "No")]


def test_two_ssh_machines_measure_only_the_active_one(monkeypatch):
    package = deepseek_package()
    identity = package_identity_key(package)
    calls: list[str] = []

    def _candidates(alias, packages, runner):
        calls.append(alias)
        return {"candidates": [{"package_identity": list(identity)}], "set_aside": 0}

    monkeypatch.setattr(grm, "remote_aliases", lambda path: {"pid1": "gpu1", "pid2": "gpu2"})
    monkeypatch.setattr(grm, "run_packages", lambda config, selection: [package])
    monkeypatch.setattr(grm, "ssh_candidates", _candidates)
    monkeypatch.setattr(grm, "ssh_measure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not measure")))
    run = _FakeRun(asker=_MissingAnswerAsker(), ssh_active="gpu2")

    grm.remote_load_test(run, config=None, scenario=default_scenario())

    assert calls == ["gpu2"]


# --- end to end: a real answer run over FakeSsh, a real measurement, not a canned record -----------


def _measurable_server_context(daemon, gpu_percent: float, guid: str, now=GUIDED_RUN1) -> ServerContext:
    """A server whose hardware measures fully (Windows/NVIDIA) and whose own GPU utilization for the
    load test's own reading (`loadtest.gpu_utilization`) is fixed and distinctive -- proves that
    value, not another server's or the laptop's, ends up in the stored record."""
    guid_line = f"\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\n    MachineGuid    REG_SZ    {guid}\n"
    overrides = {
        WINDOWS_MACHINE_GUID_ARGS: subprocess.CompletedProcess(list(WINDOWS_MACHINE_GUID_ARGS), 0, guid_line, ""),
        GPU_UTILIZATION_ARGS: subprocess.CompletedProcess(list(GPU_UTILIZATION_ARGS), 0, f"{gpu_percent}\n", ""),
    }
    probes = Probes(platform="win32", runner=windows_runner(overrides), read_text=lambda p: (_ for _ in ()).throw(OSError()), memory_bytes=lambda: LAPTOP_RAM_BYTES)
    return ServerContext(probes=probes, room=fixed_room(), daemon=daemon, now=lambda: now, environ={})


def _guarded_laptop_probes() -> Probes:
    """This computer's own `Probes`: ordinary hardware reading (step 1 still shows this machine), but
    the load test's own GPU command fails the test outright -- proves this run's own hardware probe
    is never asked for a remote candidate's load, for either server."""
    base = windows_probes()

    def guarded_runner(args):
        if list(args) == list(GPU_UTILIZATION_ARGS):
            raise AssertionError("the laptop's own runner must never be asked for a remote server's load")
        return base.runner(args)

    return base.__class__(**{**base.__dict__, "runner": guarded_runner})


_SELECT = [UNSLOTH, QWEN_GGUF, DEEPSEEK]
GUID_GPU1 = "11111111-2222-3333-4444-555555555555"
GUID_GPU2 = "66666666-7777-8888-9999-aaaaaaaaaaaa"


def _guided_run(tmp_path: Path, answers: dict, ssh, now=GUIDED_RUN1, machines=("ssh",), probes=None):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "home").mkdir(exist_ok=True)
    full = {**FULL_ANSWERS, "select": _SELECT, "machines": list(machines), **answers}
    lines: list[str] = []
    asker = _WatchingAsker(full, lines)
    code = run_guided(
        asker, here=_results(tmp_path), pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()), probes=probes or windows_probes(),
        daemon=offline_daemon(), servers=offline_servers(), now=now, out=lines.append, ssh=ssh,
    )
    return code, lines, asker


def test_two_ssh_machines_measure_only_the_active_one_with_a_real_measurement(tmp_path):
    """Codex r1 minor 2: not a canned record -- a real answer run over `FakeSsh`, with the real
    `measure` op running the protocol at the active target's own server process. gpu1 and gpu2 give
    different GPU utilization; the laptop's own runner raises if it is ever asked for a load reading
    at all; gpu1 stays without a connection during the active round (B4).

    `cpu_load` has no injection seam of its own (`loadtest.read_cpu_load` reads `os.getloadavg`
    directly, and this machine's own Windows has none -- `getattr` returns `None` before any call):
    the same limitation the local load test has, no special case for the remote path (Codex r3-2).
    """
    ssh_a = FakeSsh(context=_measurable_server_context(loadtest_daemon(), gpu_percent=11.0, guid=GUID_GPU1))
    _guided_run(tmp_path, {"ssh_alias": "gpu1", "ssh_machine_class": "server"}, ssh_a)
    ssh_b = FakeSsh(context=_measurable_server_context(loadtest_daemon(), gpu_percent=88.0, guid=GUID_GPU2))
    _guided_run(tmp_path, {"ssh_alias": "gpu2", "ssh_machine_class": "server"}, ssh_b, now=GUIDED_RUN2)
    config = load_config(_results(tmp_path) / "modelroom.toml")
    gpu1_id, gpu2_id = config.machines["gpu1"].profile, config.machines["gpu2"].profile

    ssh_active = FakeSsh(context=_measurable_server_context(loadtest_daemon(), gpu_percent=88.0, guid=GUID_GPU2))
    _code, lines, _asker = _guided_run(
        tmp_path,
        {
            "machines": [f"profile:{gpu1_id}", f"profile:{gpu2_id}"], "ssh_active": "gpu2",
            "load_test_remote": True, "load_test_remote_packages": [DEEPSEEK_OLLAMA_NAME],
        },
        ssh_active, now=GUIDED_RUN2, machines=(f"profile:{gpu1_id}", f"profile:{gpu2_id}"),
        probes=_guarded_laptop_probes(),
    )

    assert not any(call[0] == "gpu1" for call in ssh_active.calls), "gpu1 (not active) must stay without a connection"
    assert any(call[0] == "gpu2" and call[1] == "measure" for call in ssh_active.calls)

    gpu1_dir = _results(tmp_path) / "state" / "measurements" / gpu1_id
    gpu2_dir = _results(tmp_path) / "state" / "measurements" / gpu2_id
    assert not (gpu1_dir.is_dir() and list(gpu1_dir.glob("*.json")))
    gpu2_files = list(gpu2_dir.glob("*.json"))
    assert len(gpu2_files) == 1
    record = json.loads(gpu2_files[0].read_text(encoding="utf-8"))
    assert record["profile_id"] == gpu2_id
    assert record["load_state"]["gpu_utilization_percent"] == 88.0


def test_the_record_is_stored_under_the_alias_profile(tmp_path):
    package = deepseek_package()
    record = _local_record(package, PROFILE_ID)
    run = _FakeRun()
    config = SimpleNamespace(paths=SimpleNamespace(state=tmp_path / "state"))
    offer = grm.RemoteOffer("gpu1", PROFILE_ID, (package,), (package_identity_key(package),))

    ok = grm._store_checked(run, config, offer, package, record, _shown(package))

    assert ok is True
    assert run.problems == []
    stored = read_measurements(tmp_path / "state", PROFILE_ID)
    assert [entry.measurement_id for entry in stored.records] == [record.measurement_id]
    assert all(entry.profile_id == PROFILE_ID for entry in stored.records)
    assert measurement_path(tmp_path / "state", PROFILE_ID, record.measurement_id).is_file()


def test_an_invalid_record_is_refused(monkeypatch):
    """A record measured under another `profile_id` than the alias's own: the double sink does not
    touch `profile_id`, so this is caught by the explicit check, not by `neutral_record` alone."""
    package = deepseek_package()
    identity = package_identity_key(package)
    foreign_record = _local_record(package, "0" * 16)
    monkeypatch.setattr(grm, "ssh_measure", lambda *a, **k: {"record": foreign_record.model_dump(mode="json")})
    run = _FakeRun()
    offer = grm.RemoteOffer("gpu1", PROFILE_ID, (package,), (identity,))

    ok = grm._measure_one(run, None, offer, identity, default_scenario())

    assert ok is False
    assert run.problems == [FORM_ERROR.format(alias="gpu1")]


def test_a_record_with_a_foreign_name_is_refused_on_the_laptop(monkeypatch):
    """The server did not hold the rules: its `ollama_name` is a foreign, local name -- the second
    sink pass rebuilds it from the known package and the mismatch is discarded, never stored."""
    package = deepseek_package()
    identity = package_identity_key(package)
    record = _local_record(package, PROFILE_ID)
    tampered = record.model_copy(update={"ollama_name": "sentinel-host:latest"})
    monkeypatch.setattr(grm, "ssh_measure", lambda *a, **k: {"record": tampered.model_dump(mode="json")})
    run = _FakeRun()
    offer = grm.RemoteOffer("gpu1", PROFILE_ID, (package,), (identity,))

    ok = grm._measure_one(run, None, offer, identity, default_scenario())

    assert ok is False
    assert run.problems == [FORM_ERROR.format(alias="gpu1")]
    assert "sentinel" not in "".join(run.screen.notes)


def test_a_broken_connection_mid_measurement_is_one_line(monkeypatch):
    def _boom(*a, **k):
        raise SshError("the connection to gpu1 ended before the answer")

    monkeypatch.setattr(grm, "ssh_measure", _boom)
    package = deepseek_package()
    identity = package_identity_key(package)
    run = _FakeRun()
    offer = grm.RemoteOffer("gpu1", PROFILE_ID, (package,), (identity,))

    ok = grm._measure_one(run, None, offer, identity, default_scenario())

    assert ok is False
    assert run.problems == ["the connection to gpu1 ended before the answer"]
    assert run.screen.notes == []


def test_measure_reads_load_on_the_server_not_here(monkeypatch, tmp_path):
    """B2: the machine's own load (`gpu_utilization`/`read_cpu_load`) is read where `run_load_test`
    runs -- at the server, inside `measure_result`. This side never reads one: `_FakeRun` carries no
    `.probes` at all, so a code path that tried would fail with `AttributeError`, not a wrong value."""
    package = deepseek_package()
    identity = package_identity_key(package)
    record = _local_record(package, PROFILE_ID)  # its own `load_state` came from a *separate* Probes
    assert record.load_state is not None

    monkeypatch.setattr(grm, "remote_aliases", lambda path: {PROFILE_ID: "gpu1"})
    monkeypatch.setattr(grm, "run_packages", lambda config, selection: [package])
    monkeypatch.setattr(
        grm, "ssh_candidates",
        lambda alias, packages, runner: {"candidates": [{"package_identity": list(identity)}], "set_aside": 0},
    )
    monkeypatch.setattr(grm, "ssh_measure", lambda *a, **k: {"record": record.model_dump(mode="json")})
    run = _FakeRun(asker=_YesAsker([_shown(package)]))
    config = SimpleNamespace(paths=SimpleNamespace(state=tmp_path / "state"))

    result = grm.remote_load_test(run, config, default_scenario())

    assert result == "1 of 1 measured on gpu1"
    assert read_measurements(tmp_path / "state", PROFILE_ID).records
