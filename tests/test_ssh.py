"""`modelroom/ssh.py` (FR-B1, B1): the fixed argument vector, the alias check, the timeouts and
limits, and the mapping of every failure to one fixed local sentence -- never ssh's own stderr,
never a text of the server. The real process boundary itself (bounded reads, a bounded queue, a
deadline from process start, guaranteed cleanup) is proven against real child processes below, not
just canned `FaultSsh` outcomes -- Codex r1 findings 1-3.
"""

from __future__ import annotations

import json
import sys
import time

import pytest

from fixture_ssh import FakeSsh, FaultSsh
from modelroom.remote_server import ServerContext
from modelroom.ssh import (
    OP_BUDGETS,
    Limits,
    ProcessOutcome,
    SshError,
    SubprocessSsh,
    build_argv,
    call,
    default_ssh,
    is_alias,
)


def test_the_command_line_is_fixed_and_carries_every_safety_option():
    argv = build_argv("/usr/bin/ssh", "gpu1", "hello", "modelroom")
    assert argv[0] == "/usr/bin/ssh"
    for option in (
        "-T", "BatchMode=yes", "StrictHostKeyChecking=yes", "ConnectTimeout=10", "ServerAliveInterval=15",
        "ServerAliveCountMax=4", "ForwardAgent=no", "ForwardX11=no", "ClearAllForwardings=yes",
        "PermitLocalCommand=no", "RemoteCommand=none", "RequestTTY=no", "LogLevel=ERROR",
    ):
        assert option in argv
    assert argv[argv.index("--") + 1 :] == ["gpu1", "modelroom", "remote", "hello"]


@pytest.mark.parametrize(
    "alias",
    ["192.0.2.10", "10", "2001:db8::1", "gpu1.example", "-oProxyCommand=x", "user@gpu1", "gpu1:22", "../x", "a b", "GPU1", "x" * 32],
)
def test_alias_re_refuses_addresses_domains_options_logins_paths(alias):
    assert not is_alias(alias)
    with pytest.raises(SshError):
        call(alias, "hello", {}, FaultSsh(ProcessOutcome(0)))


def test_a_valid_alias_is_accepted():
    assert is_alias("gpu1")


def test_data_never_reaches_the_command_line():
    """A package name with shell metacharacters travels only in the JSON body, never in argv."""
    line = b'{"protocol":1,"result":{"status":200,"body":null}}'
    runner = FaultSsh(ProcessOutcome(0), lines=(line,))
    dangerous = "x; rm -rf /$(echo y)`z`\nname"
    call("gpu1", "daemon_short", {"method": "GET", "path": "/api/version", "body": {"name": dangerous}}, runner)
    argv, stdin = runner.calls[0][0], runner.calls[0][1]
    assert not any(dangerous in part for part in argv)
    assert b"rm -rf" in stdin  # the data traveled over stdin, JSON-escaped, never argv


def test_a_line_over_the_limit_is_a_fault():
    with pytest.raises(SshError, match="more than modelroom reads"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(None, over_limit=True)))


def test_a_line_of_the_wrong_shape_is_a_fault():
    bad = b'{"not_protocol": 1}'
    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(0), lines=(bad,)))


def test_a_line_that_is_not_json_is_a_fault():
    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(0), lines=(b"not json",)))


def test_a_result_that_fails_the_strict_op_model_is_a_form_fault():
    """Codex r1 finding 4: an empty `hello` result (missing every required field) is a form fault
    here, never a bare `KeyError` once something later reads `result["os_fingerprint"]`."""
    line = b'{"protocol":1,"result":{}}'
    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(0), lines=(line,)))


def test_missing_ssh_says_how_to_install():
    class NoSsh:
        def run(self, *a, **k):
            raise AssertionError("never reached: the runner is not asked without a path")

    with pytest.raises(SshError, match="no ssh command"):
        call("gpu1", "hello", {}, NoSsh(), which=lambda name: None)


def test_the_local_bin_fallback_runs_once():
    runner = FakeSsh(context=ServerContext(), known_programs=("~/.local/bin/modelroom",))
    result = call("gpu1", "hello", {}, runner)
    assert len(runner.calls) == 2
    assert result["result"]["ollama"] in ("reachable", "unreachable")


def test_each_op_has_its_time_limit():
    assert OP_BUDGETS["hello"].total_seconds == 30.0
    assert OP_BUDGETS["room"].total_seconds == 30.0
    assert OP_BUDGETS["profile"].total_seconds == 180.0
    assert OP_BUDGETS["daemon_short"].total_seconds == 30.0
    assert OP_BUDGETS["daemon_generate"].total_seconds == 660.0
    assert OP_BUDGETS["daemon_pull"].per_line_seconds == 60.0


@pytest.mark.parametrize(
    ("returncode", "stderr", "message"),
    [
        (255, b"Host key verification failed.", "host key of gpu1 is not known"),
        (255, b"REMOTE HOST IDENTIFICATION HAS CHANGED!", "host key of gpu1 is not known"),
        (255, b"Permission denied (publickey).", "refused the login"),
        (255, b"ssh: connect to host gpu1 port 22: Connection refused", "could not be reached over SSH"),
        (127, b"", "not installed on gpu1"),
    ],
)
def test_each_failure_maps_to_its_sentence(returncode, stderr, message):
    runner = FaultSsh(ProcessOutcome(returncode, stderr=stderr))
    with pytest.raises(SshError, match=message):
        call("gpu1", "hello", {}, runner)


def test_an_old_server_is_reported_as_such():
    with pytest.raises(SshError, match="older than 0.2.0"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(2, stderr=b"usage: modelroom ...")))


def test_a_timeout_names_the_alias_and_the_seconds():
    with pytest.raises(SshError, match=r"gpu1 did not answer within 30 s"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(None, timed_out=True)))


def test_ssh_stderr_never_reaches_screen_or_problems():
    runner = FaultSsh(ProcessOutcome(255, stderr=b"sentinel-host has left the building"))
    try:
        call("gpu1", "hello", {}, runner)
        raise AssertionError("expected SshError")
    except SshError as exc:
        assert "sentinel-host" not in str(exc)


def test_a_connection_cut_mid_pull_gets_its_own_resumable_sentence():
    with pytest.raises(SshError, match="the same pull resumes"):
        call("gpu1", "daemon_pull", {"method": "POST", "path": "/api/pull", "body": {}}, FaultSsh(ProcessOutcome(0)))


def test_a_pull_cut_by_exit_255_after_progress_also_gets_the_resumable_sentence():
    """Codex r1 finding 5: exit 255 mid-pull must not fall back to the generic host-key/login
    classification -- the resumable hint always wins for a pull that ends without its final line."""
    progress = json.dumps({"protocol": 1, "progress": {"status": "pulling manifest"}}).encode()
    runner = FaultSsh(ProcessOutcome(255, stderr=b"kex_exchange_identification: read: reset"), lines=(progress,))
    with pytest.raises(SshError, match="the same pull resumes"):
        call("gpu1", "daemon_pull", {"method": "POST", "path": "/api/pull", "body": {}}, runner)


def test_a_connection_cut_outside_a_pull_gets_the_generic_sentence():
    with pytest.raises(SshError, match="ended before the answer"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(0)))


def test_another_protocol_version_on_hello_names_the_number():
    line = b'{"protocol": 2, "result": {}}'
    with pytest.raises(SshError, match="speaks protocol 2"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(0), lines=(line,)))


def test_the_real_ssh_is_refused_under_pytest():
    with pytest.raises(RuntimeError):
        default_ssh()


# --- real child processes (Codex r1 findings 1-3): the process boundary itself ---------------------


def _write_script(tmp_path, text: str):
    script = tmp_path / "fake_ssh.py"
    script.write_text(text, encoding="utf-8")
    return script


def _real_argv(tmp_path, script, op="hello"):
    return [sys.executable, str(script), *build_argv("ssh", "gpu1", op, "modelroom")[1:]]


def test_the_real_ssh_argv_is_exact_as_a_subprocess(tmp_path):
    """A small Python script stands in for `ssh`; it echoes its own argv back, so the test checks
    the *actual* vector a real process received, not one this module merely built and never sent."""
    script = _write_script(
        tmp_path,
        "import sys, json\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'protocol': 1, 'result': {'argv': sys.argv[1:]}}))\n",
    )
    seen = []
    SubprocessSsh().run(_real_argv(tmp_path, script), b"{}", 10.0, None, OP_BUDGETS["hello"].limits, seen.append)
    # `outcome.returncode` is not asserted here: once the terminal line is read, this module stops
    # the process itself rather than waiting for its own natural exit, and a `terminate()` on an
    # already-exiting process can report a code of its own (e.g. 1 on Windows) instead of the
    # child's -- harmless, since `call()` never reads `returncode` once it has a terminal envelope.
    payload = json.loads(seen[0])
    received = payload["result"]["argv"]
    assert received[received.index("--") + 1 :] == ["gpu1", "modelroom", "remote", "hello"]
    assert "StrictHostKeyChecking=yes" in received and "BatchMode=yes" in received


def test_a_delayed_progress_line_survives_a_generous_deadline_but_a_short_one_times_out(tmp_path):
    """Codex r1 finding 3: the deadline is counted from process start, not after `stdin.write` --
    a script that sleeps *before reading stdin* still finishes well inside a generous total, and a
    short one still times out close to its own limit, not close to "generous minus the sleep"."""
    script = _write_script(
        tmp_path,
        "import sys, json, time\n"
        "time.sleep(1.5)\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'protocol': 1, 'result': {'ok': True}}))\n",
    )
    argv = _real_argv(tmp_path, script)
    started = time.monotonic()
    outcome = SubprocessSsh().run(argv, b"{}", 10.0, None, OP_BUDGETS["hello"].limits, lambda raw: True)
    assert outcome.timed_out is False
    assert time.monotonic() - started >= 1.5

    started = time.monotonic()
    outcome = SubprocessSsh().run(argv, b"{}", 0.3, None, OP_BUDGETS["hello"].limits, lambda raw: True)
    elapsed = time.monotonic() - started
    assert outcome.timed_out is True
    assert elapsed < 1.0, f"the deadline should fire near 0.3s, not after the full 1.5s sleep (took {elapsed:.2f}s)"


def test_an_oversized_line_from_a_real_process_is_caught_without_reading_it_all(tmp_path):
    """Codex r1 finding 2: a process that writes one endless line (no newline at all) must be
    caught on its first bounded read, not after megabytes of it arrived."""
    script = _write_script(
        tmp_path,
        "import sys, time\n"
        "sys.stdin.read()\n"
        "sys.stdout.write('x' * (8 * 1024 * 1024))\n"
        "sys.stdout.flush()\n"
        "time.sleep(5)\n",
    )
    argv = _real_argv(tmp_path, script)
    started = time.monotonic()
    outcome = SubprocessSsh().run(argv, b"{}", 10.0, None, Limits(line_bytes=1024), lambda raw: True)
    elapsed = time.monotonic() - started
    assert outcome.over_limit is True
    assert elapsed < 2.0, f"an oversized line should be caught almost immediately, not after {elapsed:.2f}s"


def test_stdin_the_child_never_reads_does_not_hang(tmp_path):
    """A script that never touches stdin at all, and answers right away regardless -- writing
    stdin must not block waiting for a reader that will never come."""
    script = _write_script(
        tmp_path,
        "import sys, json\n"
        "print(json.dumps({'protocol': 1, 'result': {'ok': True}}))\n",
    )
    argv = _real_argv(tmp_path, script)
    huge_stdin = b'{"pad": "' + b"x" * 200_000 + b'"}'
    started = time.monotonic()
    outcome = SubprocessSsh().run(argv, huge_stdin, 10.0, None, OP_BUDGETS["hello"].limits, lambda raw: True)
    assert time.monotonic() - started < 5.0
    assert outcome.timed_out is False and outcome.over_limit is False


def test_the_process_is_reaped_after_a_timeout(tmp_path):
    """A script that sleeps far longer than the deadline is still gone (not a zombie, not still
    running) once `run` returns -- proven by starting another one right after and getting a fresh
    PID that answers normally, which a leaked process table entry would not prevent either, but a
    hung *thread* from the previous call would still show up as a long total test time."""
    script = _write_script(
        tmp_path,
        "import sys, time\n"
        "time.sleep(30)\n",
    )
    argv = _real_argv(tmp_path, script)
    started = time.monotonic()
    outcome = SubprocessSsh().run(argv, b"{}", 0.3, None, OP_BUDGETS["hello"].limits, lambda raw: True)
    elapsed = time.monotonic() - started
    assert outcome.timed_out is True
    assert elapsed < 2.0, f"the process should be killed near the 0.3s deadline, not wait out its 30s sleep (took {elapsed:.2f}s)"


def test_an_exception_from_on_line_still_kills_and_reaps_the_process(tmp_path):
    """Codex r1 finding 3: an exception raised inside `on_line` (a form fault, in real use) must
    still end the process promptly through the same cleanup -- proven by timing, since a leaked
    process holding the pipe open would otherwise stall the next read indefinitely."""
    script = _write_script(
        tmp_path,
        "import sys, json, time\n"
        "print(json.dumps({'protocol': 1, 'progress': {'status': 'pulling manifest'}}))\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n",
    )
    argv = _real_argv(tmp_path, script, op="daemon")

    def blows_up(raw):
        raise SshError("boom")

    started = time.monotonic()
    with pytest.raises(SshError, match="boom"):
        SubprocessSsh().run(argv, b"{}", 10.0, None, OP_BUDGETS["daemon_pull"].limits, blows_up)
    assert time.monotonic() - started < 2.0
