"""`modelroom/ssh.py` (FR-B1, B1): the fixed argument vector, the alias check, the timeouts and
limits, and the mapping of every failure to one fixed local sentence -- never ssh's own stderr,
never a text of the server.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from fixture_ssh import FakeSsh, FaultSsh
from modelroom.remote_server import ServerContext
from modelroom.ssh import (
    OP_BUDGETS,
    ProcessOutput,
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
        call(alias, "hello", {}, FaultSsh(ProcessOutput([], 0)))


def test_a_valid_alias_is_accepted():
    assert is_alias("gpu1")


def test_data_never_reaches_the_command_line():
    """A package name with shell metacharacters travels only in the JSON body, never in argv."""
    runner = FaultSsh(ProcessOutput([b'{"protocol":1,"result":{"status":200,"body":null}}\n'], 0))
    dangerous = "x; rm -rf /$(echo y)`z`\nname"
    call("gpu1", "daemon_short", {"method": "GET", "path": "/api/version", "body": {"name": dangerous}}, runner)
    argv, stdin = runner.calls[0][0], runner.calls[0][1]
    assert not any(dangerous in part for part in argv)
    assert b"rm -rf" in stdin  # the data traveled over stdin, JSON-escaped, never argv


def test_a_line_over_the_limit_is_a_fault():
    huge = b'{"protocol":1,"result":{}}' + b" " * (2 * 1024 * 1024)
    with pytest.raises(SshError, match="more than modelroom reads"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutput([huge], None, over_limit=True)))


def test_an_answer_over_the_total_limit_is_a_fault():
    with pytest.raises(SshError, match="more than modelroom reads"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutput([b"{}"], None, over_limit=True)))


def test_a_line_of_the_wrong_shape_is_a_fault():
    bad = b'{"not_protocol": 1}\n'
    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutput([bad], 0)))


def test_a_line_that_is_not_json_is_a_fault():
    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutput([b"not json\n"], 0)))


def test_missing_ssh_says_how_to_install():
    class NoSsh:
        def run(self, *a, **k):
            raise AssertionError("never reached: the runner is not asked without a path")

    with pytest.raises(SshError, match="no ssh command"):
        call("gpu1", "hello", {}, NoSsh(), which=lambda name: None)


def test_the_local_bin_fallback_runs_once():
    runner = FakeSsh(context=ServerContext(), known_programs=("~/.local/bin/modelroom",))
    result = call("gpu1", "hello", {}, runner)
    programs = [c[0] for c in runner.calls]
    assert len(runner.calls) == 2
    assert result[-1]["result"]["ollama"] in ("reachable", "unreachable")


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
    runner = FaultSsh(ProcessOutput([], returncode, stderr=stderr))
    with pytest.raises(SshError, match=message):
        call("gpu1", "hello", {}, runner)


def test_an_old_server_is_reported_as_such():
    with pytest.raises(SshError, match="older than 0.2.0"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutput([], 2, stderr=b"usage: modelroom ...")))


def test_a_timeout_names_the_alias_and_the_seconds():
    with pytest.raises(SshError, match=r"gpu1 did not answer within 30 s"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutput([], None, timed_out=True)))


def test_ssh_stderr_never_reaches_screen_or_problems():
    runner = FaultSsh(ProcessOutput([], 255, stderr=b"sentinel-host has left the building"))
    try:
        call("gpu1", "hello", {}, runner)
        raise AssertionError("expected SshError")
    except SshError as exc:
        assert "sentinel-host" not in str(exc)


def test_a_connection_cut_mid_pull_gets_its_own_resumable_sentence():
    with pytest.raises(SshError, match="the same pull resumes"):
        call("gpu1", "daemon_pull", {"method": "POST", "path": "/api/pull", "body": {}}, FaultSsh(ProcessOutput([], 0)))


def test_a_connection_cut_outside_a_pull_gets_the_generic_sentence():
    with pytest.raises(SshError, match="ended before the answer"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutput([], 0)))


def test_another_protocol_version_on_hello_names_the_number():
    line = b'{"protocol": 2, "result": {}}\n'
    with pytest.raises(SshError, match="speaks protocol 2"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutput([line], 0)))


def test_the_real_ssh_is_refused_under_pytest():
    with pytest.raises(RuntimeError):
        default_ssh()


def test_the_real_ssh_argv_is_exact_as_a_subprocess(tmp_path):
    """A small Python script stands in for `ssh`, and gets exactly the fixed argument vector."""
    script = tmp_path / "fake_ssh.py"
    script.write_bytes(
        b"import sys, json\n"
        b"sys.stdin.read()\n"
        b'sys.stdout.write(json.dumps({"protocol": 1, "result": {"argv": sys.argv[1:]}}) + "\\n")\n'
    )
    runner = SubprocessSsh()
    argv = [sys.executable, str(script), *build_argv("ssh", "gpu1", "hello", "modelroom")[1:]]
    outcome = runner.run(argv, b"{}", 10.0, None, OP_BUDGETS["hello"].limits)
    assert outcome.returncode == 0
    assert outcome.lines
