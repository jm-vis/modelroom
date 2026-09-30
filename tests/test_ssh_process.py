"""The real process boundary of the SSH transport (FR-B1, B1; Codex FR-B1 r2 findings 1-3 and 6):
a small Python script stands in for `ssh` as a real child process (`Teststrategie ohne echten
Server`, "Fake-ssh als Prozess"). No network, no real `ssh`.

Every way a call can end -- a clean answer, a form fault with a full queue, an over-limit line, a
timeout, Ctrl-C -- is checked for the same end state: the child process has exited (by its own PID,
at the operating system), its three pipes are closed, and no reader or writer thread of the
transport is left running (`threading.enumerate`).
"""

from __future__ import annotations

import _thread
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from modelroom.remote_client import SshDaemon
from modelroom.ssh import OP_BUDGETS, Limits, SshError, SubprocessSsh, call

# The interpreter itself, not a launcher that would start it as a grandchild (a venv's `python.exe`
# on Windows can be one): the PID the script reports is then the PID of the child process.
PYTHON = getattr(sys, "_base_executable", None) or sys.executable
_TRANSPORT_THREADS = ("modelroom-ssh-", "(_read_stdout)", "(_read_stderr)", "(_feed_stdin)")
_HELLO = {"version": "0.2.0", "system": "linux", "ollama": "reachable", "agent_forwarded": False, "os_fingerprint": "none"}
_PULL = {"model": "qwen3.5:9b"}


class _ScriptSsh:
    """A `SshRunner` whose "ssh" is `script`: the argument vector after the program is passed on
    unchanged, so the script sees exactly what a real `ssh` would."""

    def __init__(self, script: Path) -> None:
        self.script = script

    def run(self, argv, stdin, total, per_line, limits, on_line):
        return SubprocessSsh().run([PYTHON, str(self.script), *argv[1:]], stdin, total, per_line, limits, on_line)


@pytest.fixture
def children(monkeypatch) -> list[subprocess.Popen]:
    """Every child process the transport starts in this test, recorded as it is started."""
    started: list[subprocess.Popen] = []
    real = subprocess.Popen

    def recording(*args, **kwargs):
        process = real(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording)
    return started


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_ssh.py"
    path.write_bytes(("import json, os, sys, time\n" + body).encode("utf-8"))
    return path


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _leftover_threads() -> list[str]:
    return [thread.name for thread in threading.enumerate() if thread.is_alive() and any(mark in thread.name for mark in _TRANSPORT_THREADS)]


def _assert_all_ended(children: list[subprocess.Popen]) -> None:
    assert children, "no child process was started"
    for process in children:
        assert process.poll() is not None, "the child process still runs"
        assert not _pid_alive(process.pid), "the operating system still lists the child process"
        for name in ("stdin", "stdout", "stderr"):
            assert getattr(process, name).closed, f"the {name} pipe of the child was left open"
    assert _leftover_threads() == []


# --- the argument vector and stdin, as a real process receives them -----------------------------


def test_the_argv_and_stdin_a_real_process_receives_are_exact(tmp_path, children):
    seen = tmp_path / "seen.json"
    script = _script(
        tmp_path,
        f"data = sys.stdin.buffer.read()\n"
        f"open({str(seen)!r}, 'w').write(json.dumps({{'argv': sys.argv[1:], 'stdin': data.decode()}}))\n"
        f"print(json.dumps({{'protocol': 1, 'result': {_HELLO!r}}}))\n",
    )

    result = call("gpu1", "hello", {"k": "x; rm -rf /$(y)"}, _ScriptSsh(script), ssh_path="ssh")

    received = json.loads(seen.read_text())
    assert received["argv"][received["argv"].index("--") + 1 :] == ["gpu1", "modelroom", "remote", "hello"]
    assert "StrictHostKeyChecking=yes" in received["argv"] and "BatchMode=yes" in received["argv"]
    assert not any("rm -rf" in part for part in received["argv"])
    assert json.loads(received["stdin"]) == {"k": "x; rm -rf /$(y)"}  # read to EOF: stdin was closed
    assert result["result"]["system"] == "linux"
    _assert_all_ended(children)


# --- streaming: a progress line reaches the caller while the process still runs -----------------


def test_a_progress_line_reaches_the_caller_before_the_process_ends(tmp_path, children):
    marker = tmp_path / "finished"
    script = _script(
        tmp_path,
        "sys.stdin.buffer.read()\n"
        "print(json.dumps({'protocol': 1, 'progress': {'status': 'pulling manifest'}}), flush=True)\n"
        "time.sleep(1.0)\n"
        f"open({str(marker)!r}, 'w').write('done')\n"
        "print(json.dumps({'protocol': 1, 'result': {'status': 200, 'done': True}}), flush=True)\n",
    )
    arrivals: list[tuple[float, bool]] = []

    def progress(line: dict) -> None:
        arrivals.append((time.monotonic(), marker.exists()))

    answer = SshDaemon("gpu1", _ScriptSsh(script), ssh_path="ssh")("POST", "/api/pull", _PULL, progress=progress)
    ended = time.monotonic()

    assert json.loads(answer.body) == {"status": "success"}
    assert [finished for _at, finished in arrivals] == [False]  # the script had not finished yet
    assert ended - arrivals[0][0] >= 0.5
    _assert_all_ended(children)


# --- every way out ends the process, the pipes and the threads ----------------------------------


def test_a_form_fault_with_a_full_queue_ends_the_process_the_pipes_and_every_thread(tmp_path, children):
    """Codex FR-B1 r2 finding 2: the reader is blocked on a full queue when the consumer stops at the
    first (malformed) line; the stop signal still ends it."""
    script = _script(
        tmp_path,
        "sys.stdin.buffer.read()\n"
        "print('not an envelope', flush=True)\n"
        "for n in range(5000):\n"
        "    print(json.dumps({'protocol': 1, 'progress': {'status': 'pulling manifest'}}))\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n",
    )
    started = time.monotonic()

    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "daemon_pull", {"method": "POST", "path": "/api/pull", "body": _PULL}, _ScriptSsh(script), ssh_path="ssh")

    assert time.monotonic() - started < 10.0
    _assert_all_ended(children)


def test_an_exception_of_the_caller_with_a_full_queue_ends_everything(tmp_path, children):
    script = _script(
        tmp_path,
        "for n in range(5000):\n"
        "    print(json.dumps({'protocol': 1, 'progress': {'status': 'pulling manifest'}}))\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n",
    )

    def stop(_line):
        raise RuntimeError("the caller gave up")

    with pytest.raises(RuntimeError, match="gave up"):
        _ScriptSsh(script).run(["ssh", "gpu1"], b"{}", 20.0, None, OP_BUDGETS["daemon_pull"].limits, stop)

    _assert_all_ended(children)


def test_ctrl_c_during_a_call_ends_the_process_the_pipes_and_every_thread(tmp_path, children):
    """Ctrl-C (`KeyboardInterrupt` in the main thread) while the transport waits for a silent
    server: it reaches the caller promptly -- on Windows too -- and nothing is left behind."""
    script = _script(
        tmp_path,
        "print(json.dumps({'protocol': 1, 'progress': {'status': 'pulling manifest'}}), flush=True)\n"
        "time.sleep(30)\n",
    )
    timer = threading.Timer(0.5, _thread.interrupt_main)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            call("gpu1", "daemon_pull", {"method": "POST", "path": "/api/pull", "body": _PULL}, _ScriptSsh(script), ssh_path="ssh")
    finally:
        timer.cancel()

    assert time.monotonic() - started < 5.0
    _assert_all_ended(children)


def test_a_timeout_ends_the_process_the_pipes_and_every_thread(tmp_path, children):
    script = _script(tmp_path, "time.sleep(30)\n")
    started = time.monotonic()

    outcome = _ScriptSsh(script).run(["ssh", "gpu1"], b"{}", 0.5, None, OP_BUDGETS["hello"].limits, lambda raw: None)

    assert outcome.timed_out is True
    assert time.monotonic() - started < 5.0
    _assert_all_ended(children)


def test_an_endless_line_is_caught_on_its_first_read_and_ends_everything(tmp_path, children):
    script = _script(
        tmp_path,
        "sys.stdin.buffer.read()\n"
        "sys.stdout.write('x' * (8 * 1024 * 1024))\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n",
    )
    started = time.monotonic()

    outcome = _ScriptSsh(script).run(["ssh", "gpu1"], b"{}", 20.0, None, Limits(line_bytes=1024), lambda raw: None)

    assert outcome.over_limit is True
    assert time.monotonic() - started < 5.0
    _assert_all_ended(children)


def test_a_child_that_never_reads_stdin_does_not_block_the_call(tmp_path, children):
    script = _script(tmp_path, f"print(json.dumps({{'protocol': 1, 'result': {{'ok': 1}}}}), flush=True)\n")
    started = time.monotonic()

    outcome = _ScriptSsh(script).run(["ssh", "gpu1"], b'{"pad": "' + b"x" * 400_000 + b'"}', 20.0, None, OP_BUDGETS["hello"].limits, lambda raw: None)

    assert time.monotonic() - started < 10.0
    assert outcome.timed_out is False and outcome.over_limit is False
    _assert_all_ended(children)


def test_the_deadline_counts_from_the_start_of_the_process(tmp_path, children):
    script = _script(tmp_path, "time.sleep(2.0)\nsys.stdin.buffer.read()\nprint('{}', flush=True)\n")
    started = time.monotonic()

    outcome = _ScriptSsh(script).run(["ssh", "gpu1"], b"{}", 0.5, None, OP_BUDGETS["hello"].limits, lambda raw: None)

    assert outcome.timed_out is True
    assert time.monotonic() - started < 1.8
    _assert_all_ended(children)


# --- after stdout ends: the real exit code and the whole of stderr (Codex FR-B1 r2 finding 3) -----


_CLOSE_STDOUT_THEN = "sys.stdin.buffer.read()\nsys.stdout.flush()\nos.close(1)\ntime.sleep(0.5)\n"


def test_exit_255_after_stdout_ended_keeps_its_stderr_and_its_sentence(tmp_path, children):
    script = _script(
        tmp_path,
        _CLOSE_STDOUT_THEN + "sys.stderr.write('Permission denied (publickey).\\n')\nsys.stderr.flush()\ntime.sleep(0.2)\nsys.exit(255)\n",
    )

    with pytest.raises(SshError) as excinfo:
        call("gpu1", "hello", {}, _ScriptSsh(script), ssh_path="ssh")

    assert str(excinfo.value) == "gpu1 refused the login; modelroom uses your SSH keys and agent, never a password"
    _assert_all_ended(children)


def test_exit_2_after_stdout_ended_says_the_server_is_too_old(tmp_path, children):
    script = _script(tmp_path, _CLOSE_STDOUT_THEN + "sys.stderr.write('usage: modelroom [-h]\\n')\nsys.exit(2)\n")

    with pytest.raises(SshError, match="older than 0.2.0"):
        call("gpu1", "hello", {}, _ScriptSsh(script), ssh_path="ssh")

    _assert_all_ended(children)


def test_exit_127_after_stdout_ended_tries_the_local_bin_path_once(tmp_path, children):
    counter = tmp_path / "runs"
    script = _script(
        tmp_path,
        f"open({str(counter)!r}, 'a').write('x')\n"
        "if '~/.local/bin/modelroom' not in sys.argv:\n"
        "    sys.stdin.buffer.read()\n"
        "    sys.stdout.flush()\n"
        "    os.close(1)\n"
        "    time.sleep(0.5)\n"
        "    sys.exit(127)\n"
        "sys.stdin.buffer.read()\n"
        f"print(json.dumps({{'protocol': 1, 'result': {_HELLO!r}}}), flush=True)\n",
    )

    result = call("gpu1", "hello", {}, _ScriptSsh(script), ssh_path="ssh")

    assert result["result"]["system"] == "linux"
    assert counter.read_text() == "xx"
    _assert_all_ended(children)


def test_data_after_the_result_from_a_real_process_is_a_form_fault(tmp_path, children):
    script = _script(
        tmp_path,
        "sys.stdin.buffer.read()\n"
        f"print(json.dumps({{'protocol': 1, 'result': {_HELLO!r}}}), flush=True)\n"
        "time.sleep(0.3)\n"
        "print('and one more thing', flush=True)\n",
    )

    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "hello", {}, _ScriptSsh(script), ssh_path="ssh")

    _assert_all_ended(children)


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


def test_a_generate_silent_for_more_than_60_s_still_succeeds(tmp_path, children, monkeypatch):
    """B1: `POST /api/generate` has 660 s in total and no per-line limit -- 300 s (fake clock) without
    a line is no fault; the same silence on a short `daemon` call (30 s) is."""
    import modelroom.ssh_process as ssh_process

    monkeypatch.setattr(ssh_process, "time", _FastClock(300.0))
    script = _script(
        tmp_path,
        "sys.stdin.buffer.read()\ntime.sleep(1.0)\n"
        "print(json.dumps({'protocol': 1, 'result': {'status': 200, 'body': {'done': True}}}), flush=True)\n",
    )
    request = {"method": "POST", "path": "/api/generate", "body": {"model": "qwen3.5:9b"}}

    result = call("gpu1", "daemon_generate", request, _ScriptSsh(script), ssh_path="ssh")

    assert result["result"]["status"] == 200
    with pytest.raises(SshError, match="gpu1 did not answer within 30 s"):
        call("gpu1", "daemon_short", {"method": "GET", "path": "/api/version", "body": None}, _ScriptSsh(script), ssh_path="ssh")
    _assert_all_ended(children)


@pytest.mark.parametrize(("op", "seconds"), [("hello", 30), ("room", 30), ("profile", 180), ("candidates", 60)])
def test_each_op_has_its_time_limit(tmp_path, children, monkeypatch, op, seconds):
    """B1's table, on a fake clock: a server silent for longer than the op's total limit is stopped
    with that op's own number of seconds."""
    import modelroom.ssh_process as ssh_process

    monkeypatch.setattr(ssh_process, "time", _FastClock(seconds / 0.3))
    script = _script(tmp_path, "time.sleep(30)\n")

    with pytest.raises(SshError, match=f"gpu1 did not answer within {seconds} s"):
        call("gpu1", op, {}, _ScriptSsh(script), ssh_path="ssh")
    _assert_all_ended(children)


def test_the_program_parameter_stands_in_for_ssh(tmp_path, children):
    """B1's own seam for a process test: `SubprocessSsh(program=[python, script])`."""
    script = _script(
        tmp_path,
        "sys.stdin.buffer.read()\n"
        f"print(json.dumps({{'protocol': 1, 'result': {_HELLO!r}}}), flush=True)\n",
    )

    result = call("gpu1", "hello", {}, SubprocessSsh(program=[PYTHON, str(script)]), ssh_path="ssh")

    assert result["result"]["ollama"] == "reachable"
    _assert_all_ended(children)
