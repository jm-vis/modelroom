"""`modelroom/ssh.py` (FR-B1, B1): the fixed argument vector, the alias check, the timeouts and
limits, and the mapping of every failure to one fixed local sentence -- never ssh's own stderr,
never a text of the server; and the strict reading of the answer stream to EOF (one envelope per
line, exactly one terminal line, known error codes). The real process boundary itself is proven
against real child processes in `tests/test_ssh_process.py`.
"""

from __future__ import annotations

import json

import pytest

from fixture_ssh import FakeSsh, FaultSsh
from modelroom.remote_server import ServerContext
from modelroom.ssh import (
    OP_BUDGETS,
    ProcessOutcome,
    SshError,
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


_PULL = {"method": "POST", "path": "/api/pull", "body": {}}
_PROGRESS = json.dumps({"protocol": 1, "progress": {"status": "pulling manifest"}}).encode()


def test_a_connection_cut_mid_pull_gets_its_own_resumable_sentence():
    with pytest.raises(SshError, match="the same pull resumes"):
        call("gpu1", "daemon_pull", _PULL, FaultSsh(ProcessOutcome(0), lines=(_PROGRESS,)))


_RESUMES = "the connection to gpu1 ended; Ollama there keeps what arrived, the same pull resumes"


@pytest.mark.parametrize(
    ("outcome", "sentence"),
    [
        (ProcessOutcome(255, stderr=b"Host key verification failed."), "the host key of gpu1 is not known here or has changed; connect once with ssh gpu1 yourself and check it"),
        (ProcessOutcome(255, stderr=b"@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @"), "the host key of gpu1 is not known here or has changed; connect once with ssh gpu1 yourself and check it"),
        (ProcessOutcome(255, stderr=b"Permission denied (publickey,password)."), "gpu1 refused the login; modelroom uses your SSH keys and agent, never a password"),
        (ProcessOutcome(127), "modelroom is not installed on gpu1, or not on the PATH of an SSH command (see README)"),
        (ProcessOutcome(2, stderr=b"usage: modelroom"), "modelroom on gpu1 is older than 0.2.0; update it there"),
    ],
    ids=["host-key-unknown", "host-key-changed", "login", "missing-127", "too-old-2"],
)
def test_a_pull_that_fails_before_it_began_keeps_the_sentence_of_its_start_fault(outcome, sentence):
    """Codex FR-B1 r3 finding 3: a fault that clearly came before the call reached the server keeps
    its own sentence, a pull included."""
    with pytest.raises(SshError) as excinfo:
        call("gpu1", "daemon_pull", _PULL, FaultSsh(outcome))
    assert str(excinfo.value) == sentence


@pytest.mark.parametrize(
    "outcome",
    [
        ProcessOutcome(0),
        ProcessOutcome(255, stderr=b"Connection to gpu1 closed by remote host."),
        ProcessOutcome(255, stderr=b"client_loop: send disconnect: Connection reset"),
        ProcessOutcome(1),
        ProcessOutcome(None),
    ],
    ids=["eof-exit-0", "255-closed", "255-reset", "exit-1", "no-code"],
)
def test_a_pull_cut_before_its_first_line_keeps_the_resumable_hint(outcome):
    """Codex FR-B1 r3 finding 3: Ollama may have taken the pull on before it sent a line -- a pull
    that ends without its final line, and not by a clear start fault, gets the resumable hint."""
    with pytest.raises(SshError) as excinfo:
        call("gpu1", "daemon_pull", _PULL, FaultSsh(outcome))
    assert str(excinfo.value) == _RESUMES


def test_the_same_cut_outside_a_pull_is_a_connection_end():
    with pytest.raises(SshError, match="ended before the answer"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(0)))
    with pytest.raises(SshError, match="could not be reached over SSH"):
        call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(255, stderr=b"Connection reset")))


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


def test_no_test_reaches_the_real_ssh():
    """The autouse fixture of `tests/conftest.py` stands in for `ssh.default_ssh` in every test: the
    runner a guided run gets without `ssh=` fails loudly instead of starting a process."""
    from modelroom import ssh

    with pytest.raises(AssertionError, match="default ssh runner"):
        ssh.default_ssh().run(["ssh"], b"{}", 1.0, None, OP_BUDGETS["hello"].limits, lambda raw: None)


# --- the stream to EOF: one envelope per line, exactly one terminal line (Codex FR-B1 r2 finding 1) ---

_HELLO_RESULT = {"version": "0.2.0", "system": "linux", "ollama": "reachable", "agent_forwarded": False, "os_fingerprint": "none"}


def _line(**fields) -> bytes:
    return json.dumps(fields).encode()


def _hello_with(*lines: bytes, code: int = 0):
    return call("gpu1", "hello", {}, FaultSsh(ProcessOutcome(code), lines=lines))


def test_a_clean_hello_reads_to_eof_and_returns_its_result():
    assert _hello_with(_line(protocol=1, result=_HELLO_RESULT))["result"]["system"] == "linux"


@pytest.mark.parametrize(
    "after",
    [
        _line(protocol=1, result=_HELLO_RESULT),
        _line(protocol=1, error="internal"),
        _line(protocol=1, progress={"status": "pulling manifest"}),
        b"trailing text of the server",
    ],
)
def test_a_line_after_the_last_is_a_fault(after):
    with pytest.raises(SshError, match="form this modelroom does not read"):
        _hello_with(_line(protocol=1, result=_HELLO_RESULT), after)


def test_a_line_after_the_last_pull_result_is_a_fault():
    done = _line(protocol=1, result={"status": 200, "done": True})
    with pytest.raises(SshError, match="form this modelroom does not read"):
        call("gpu1", "daemon_pull", _PULL, FaultSsh(ProcessOutcome(0), lines=(_PROGRESS, done, _PROGRESS)))


@pytest.mark.parametrize(
    "envelope",
    [
        {"protocol": 1, "result": _HELLO_RESULT, "note": "extra"},
        {"protocol": 1, "result": _HELLO_RESULT, "error": "internal"},
        {"protocol": True, "result": _HELLO_RESULT},
        {"protocol": 1.0, "result": _HELLO_RESULT},
        {"protocol": "1", "result": _HELLO_RESULT},
        {"result": _HELLO_RESULT},
        {"protocol": 1},
    ],
    ids=["extra-field", "two-fields", "protocol-true", "protocol-1.0", "protocol-string", "no-protocol", "no-field"],
)
def test_every_line_has_protocol_1_and_exactly_one_field(envelope):
    with pytest.raises(SshError, match="form this modelroom does not read"):
        _hello_with(json.dumps(envelope).encode())


def test_a_line_with_a_duplicate_key_is_a_fault():
    raw = b'{"protocol": 1, "result": {}, "result": ' + json.dumps(_HELLO_RESULT).encode() + b"}"
    with pytest.raises(SshError, match="form this modelroom does not read"):
        _hello_with(raw)


@pytest.mark.parametrize("code", ["unknown", "", "internal error on sentinel-host.example", 7, None])
def test_an_unknown_error_code_is_a_form_fault(code):
    with pytest.raises(SshError, match="form this modelroom does not read") as excinfo:
        _hello_with(_line(protocol=1, error=code))
    assert "sentinel" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("code", "sentence"),
    [
        ("bad_request", "modelroom on gpu1 could not read the request of this one; update the older one"),
        ("call_refused", "modelroom on gpu1 refused the call to its Ollama"),
        ("ollama_unreachable", "Ollama does not answer on gpu1"),
        ("too_large", "the request to gpu1 was larger than modelroom there reads"),
        ("internal", "modelroom on gpu1 failed on its own side before it could answer"),
    ],
)
def test_each_error_code_has_its_own_sentence(code, sentence):
    with pytest.raises(SshError) as excinfo:
        _hello_with(_line(protocol=1, error=code))
    assert str(excinfo.value) == sentence


def test_an_error_code_during_a_pull_that_began_is_its_own_sentence_not_a_resume():
    with pytest.raises(SshError) as excinfo:
        call("gpu1", "daemon_pull", _PULL, FaultSsh(ProcessOutcome(0), lines=(_PROGRESS, _line(protocol=1, error="ollama_unreachable"))))
    assert str(excinfo.value) == "Ollama does not answer on gpu1"


def test_the_error_codes_of_both_ends_are_the_same_five():
    from modelroom import remote_server
    from modelroom.remote_protocol import ERROR_CODES
    from modelroom.ssh import ERROR_SENTENCES

    server = {getattr(remote_server, name) for name in dir(remote_server) if name.startswith("ERROR_")}
    assert server == set(ERROR_CODES) == set(ERROR_SENTENCES)


@pytest.mark.parametrize(
    "result",
    [
        {**_HELLO_RESULT, "version": "0.2.0 on sentinel-host.example"},
        {**_HELLO_RESULT, "os_fingerprint": "/srv/sentinel-path"},
        {**_HELLO_RESULT, "agent_forwarded": "false"},
    ],
)
def test_a_hello_field_with_free_text_is_a_form_fault(result):
    with pytest.raises(SshError, match="form this modelroom does not read"):
        _hello_with(_line(protocol=1, result=result))
