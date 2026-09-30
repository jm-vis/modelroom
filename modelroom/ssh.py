"""The SSH transport to another computer's `modelroom remote <op>` (AP9-FR, FR-B1, B1).

One process, one fixed argument vector: the system `ssh` (`shutil.which`), the safety options B1
names (never overridable by a user's own config), `--`, the checked host alias, then the program
path and the fixed op name -- never a name, a path or a package name of any kind. All data of the
call goes as one JSON object over stdin, then stdin is closed (EOF); the answer is JSON lines on
stdout, each one envelope of "The remote protocol v1" (`modelroom/remote_server.py`).

`SshRunner` is the one injectable seam (`run(argv, stdin, total, per_line, limits)`); a test either
builds a `FakeSsh` that redirects into `remote_server.main` in-process (`tests/fixture_ssh.py`), or
runs a small Python script as the "ssh" program to check the real argument vector (`SubprocessSsh`
with an injected `program`). `default_ssh()` refuses under pytest, the same guard `room.py` and
`measure.py` use for the real machine -- no test may reach a real `ssh`.

Every failure here becomes one fixed local sentence with the alias in it (Codex r1-2): ssh's own
stderr and any text the server sent are never shown, stored or put in `run.problems` -- classified
by pattern and dropped. `call()` is the one function that does this classification; `modelroom/
remote_client.py` reads only its return value (parsed envelope lines) or catches `SshError`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable, Protocol, Sequence

from .neutral import ALIAS_RE, is_alias  # noqa: F401 -- re-exported, B1's own name for the check

NO_SSH = "no ssh command on this computer; on Windows: Settings, Optional features, OpenSSH Client (see README)"
INVALID_ALIAS = "a host alias is a short name of your own, like gpu1 (letters, digits, dashes)"

# The fixed argument vector (B1): never a user's config overrides these, and nothing here is data.
SSH_OPTIONS: tuple[str, ...] = (
    "-T",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
    "-o", "ForwardAgent=no",
    "-o", "ForwardX11=no",
    "-o", "ClearAllForwardings=yes",
    "-o", "PermitLocalCommand=no",
    "-o", "RemoteCommand=none",
    "-o", "RequestTTY=no",
    "-o", "LogLevel=ERROR",
)
PRIMARY_PROGRAM = "modelroom"
FALLBACK_PROGRAM = "~/.local/bin/modelroom"
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Limits:
    """The answer grenzen of one op (B1): every line at most `line_bytes`; the whole answer at most
    `total_bytes`, or unbounded (`None`) where only the line matters -- a pull's status lines."""

    line_bytes: int
    total_bytes: int | None = None


@dataclass(frozen=True)
class OpBudget:
    total_seconds: float | None
    per_line_seconds: float | None
    limits: Limits


# The six ops of the protocol (B2); `candidates`/`measure` are FR-B2's own op, reserved here so the
# transport already knows their budget -- `remote_server.py` does not answer them in this package.
_MIB = 1024 * 1024
OP_BUDGETS: dict[str, OpBudget] = {
    "hello": OpBudget(30.0, None, Limits(_MIB, 64 * 1024)),
    "room": OpBudget(30.0, None, Limits(_MIB, 64 * 1024)),
    "profile": OpBudget(180.0, None, Limits(_MIB, 4 * _MIB)),
    "daemon_short": OpBudget(30.0, None, Limits(_MIB, 16 * _MIB)),
    "daemon_generate": OpBudget(660.0, None, Limits(_MIB, 16 * _MIB)),
    "daemon_pull": OpBudget(None, 60.0, Limits(64 * 1024, None)),
    "candidates": OpBudget(60.0, None, Limits(_MIB, 4 * _MIB)),
    "measure": OpBudget(None, 660.0, Limits(_MIB, None)),
}
# The op name that reaches the wire (`modelroom remote <op>`) for each budget key above.
WIRE_OP: dict[str, str] = {
    "hello": "hello", "room": "room", "profile": "profile", "candidates": "candidates", "measure": "measure",
    "daemon_short": "daemon", "daemon_generate": "daemon", "daemon_pull": "daemon",
}


class SshError(Exception):
    """One fixed local sentence, with the alias already in it; never ssh's stderr, never server text."""


@dataclass(frozen=True)
class ProcessOutput:
    """What one `SshRunner.run` call produced: every line of stdout read, and how the process ended.

    `timed_out`/`over_limit` mark why reading stopped early -- the process is killed either way, so
    `returncode` is `None` then. A real runner enforces `total`/`per_line`/`limits` itself; a fake
    one for a behavior test returns exactly the lines and code it was built with.
    """

    lines: list[bytes] = field(default_factory=list)
    returncode: int | None = None
    timed_out: bool = False
    over_limit: bool = False
    stderr: bytes = b""


class SshRunner(Protocol):
    def run(
        self, argv: Sequence[str], stdin: bytes, total: float | None, per_line: float | None, limits: Limits
    ) -> ProcessOutput: ...


def default_ssh() -> SshRunner:
    """This computer's real `ssh`, refused under pytest (the guard `room.real_room_probes` uses)."""
    if "pytest" in sys.modules:
        raise RuntimeError("the real ssh runner was reached from a test; pass ssh= or use a fixture")
    return SubprocessSsh()


class SubprocessSsh:
    """The real transport: one `subprocess.Popen`, stdin written and closed, stdout read line by
    line on a reader thread so a per-line timeout can be enforced without blocking forever on a
    server that never writes another byte. Whether `ssh` exists at all is `call`'s own check
    (`which`), before any runner is asked to do anything."""

    def run(
        self, argv: Sequence[str], stdin: bytes, total: float | None, per_line: float | None, limits: Limits
    ) -> ProcessOutput:
        process = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        queue: "Queue[bytes | None]" = Queue()
        reader = threading.Thread(target=_read_lines, args=(process.stdout, queue), daemon=True)
        reader.start()
        try:
            process.stdin.write(stdin)
            process.stdin.close()
        except OSError:
            pass
        return _collect(process, queue, reader, total, per_line, limits)


def _read_lines(stream, queue: "Queue[bytes | None]") -> None:
    try:
        for line in iter(stream.readline, b""):
            queue.put(line)
    except OSError:
        pass
    finally:
        queue.put(None)


def _collect(
    process: subprocess.Popen, queue: "Queue[bytes | None]", reader: threading.Thread,
    total: float | None, per_line: float | None, limits: Limits
) -> ProcessOutput:
    import time

    deadline = None if total is None else time.monotonic() + total
    lines: list[bytes] = []
    total_read = 0
    while True:
        wait = _wait_for(deadline, per_line)
        try:
            line = queue.get(timeout=wait) if wait is not None else queue.get()
        except Empty:
            return _end(process, lines, timed_out=True)
        if line is None:
            break
        if len(line) > limits.line_bytes:
            return _end(process, lines, over_limit=True)
        total_read += len(line)
        if limits.total_bytes is not None and total_read > limits.total_bytes:
            return _end(process, lines, over_limit=True)
        lines.append(line)
    reader.join(timeout=1.0)
    code = process.wait(timeout=5)
    return ProcessOutput(lines, code, stderr=_drain(process.stderr))


def _wait_for(deadline: float | None, per_line: float | None) -> float | None:
    import time

    waits = [w for w in (None if deadline is None else max(deadline - time.monotonic(), 0.0), per_line) if w is not None]
    return min(waits) if waits else None


def _end(process: subprocess.Popen, lines: list[bytes], *, timed_out: bool = False, over_limit: bool = False) -> ProcessOutput:
    process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return ProcessOutput(lines, None, timed_out=timed_out, over_limit=over_limit, stderr=_drain(process.stderr))


def _drain(stream) -> bytes:
    try:
        return stream.read() or b""
    except (OSError, ValueError):
        return b""


# --- classifying a finished process into one fixed sentence, or the parsed envelope lines --------


def _budget(key: str) -> OpBudget:
    return OP_BUDGETS[key]


def build_argv(ssh_path: str, alias: str, op: str, program: str = PRIMARY_PROGRAM) -> list[str]:
    """The fixed argument vector (B1): options, `--`, the alias, then the program and the op --
    never anything else, and never through a shell string."""
    return [ssh_path, *SSH_OPTIONS, "--", alias, program, "remote", op]


def _host_key_sentence(alias: str, stderr: bytes) -> str | None:
    text = stderr.decode("utf-8", errors="replace")
    if "REMOTE HOST IDENTIFICATION HAS CHANGED" in text or "Host key verification failed" in text:
        return f"the host key of {alias} is not known here or has changed; connect once with ssh {alias} yourself and check it"
    return None


def _classify_255(alias: str, stderr: bytes) -> str:
    key = _host_key_sentence(alias, stderr)
    if key is not None:
        return key
    text = stderr.decode("utf-8", errors="replace")
    if "Permission denied" in text:
        return f"{alias} refused the login; modelroom uses your SSH keys and agent, never a password"
    return f"{alias} could not be reached over SSH; try ssh {alias} yourself to see why"


def _run_once(runner: SshRunner, ssh_path: str, alias: str, budget_key: str, program: str, payload: bytes) -> ProcessOutput:
    budget = _budget(budget_key)
    argv = build_argv(ssh_path, alias, WIRE_OP[budget_key], program)
    return runner.run(argv, payload, budget.total_seconds, budget.per_line_seconds, budget.limits)


def call(
    alias: str,
    budget_key: str,
    request: dict,
    runner: SshRunner,
    *,
    ssh_path: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[dict]:
    """One request-response call to `alias`'s `modelroom remote <op>`: the parsed envelope lines, or
    a fixed-sentence `SshError`. `budget_key` is one of `OP_BUDGETS`. `ssh_path` is the resolved path
    of the system `ssh`; without one it is resolved here (`which`, `shutil.which` by default) --
    missing, `NO_SSH` (B1) before any runner is ever asked to do anything."""
    if not is_alias(alias):
        raise SshError(INVALID_ALIAS)
    resolved = ssh_path if ssh_path is not None else which("ssh")
    if resolved is None:
        raise SshError(NO_SSH)
    payload = json.dumps(request).encode("utf-8")
    outcome = _run_once(runner, resolved, alias, budget_key, PRIMARY_PROGRAM, payload)
    if outcome.returncode == 127 and not outcome.lines:
        outcome = _run_once(runner, resolved, alias, budget_key, FALLBACK_PROGRAM, payload)
    return _read_outcome(alias, budget_key, outcome)


def _read_outcome(alias: str, budget_key: str, outcome: ProcessOutput) -> list[dict]:
    budget = _budget(budget_key)
    if outcome.timed_out:
        raise SshError(f"{alias} did not answer within {int(budget.total_seconds or budget.per_line_seconds)} s")
    if outcome.over_limit:
        raise SshError(f"{alias} answered with more than modelroom reads")
    if outcome.returncode == 127:
        raise SshError(f"modelroom is not installed on {alias}, or not on the PATH of an SSH command (see README)")
    if outcome.returncode == 2 and not outcome.lines:
        raise SshError(f"modelroom on {alias} is older than 0.2.0; update it there")
    if outcome.returncode == 255:
        raise SshError(_classify_255(alias, outcome.stderr))
    return _parse_lines(alias, budget_key, outcome)


def _parse_lines(alias: str, budget_key: str, outcome: ProcessOutput) -> list[dict]:
    envelopes = [_envelope(alias, budget_key, line) for line in outcome.lines]
    if not envelopes or ("result" not in envelopes[-1] and "error" not in envelopes[-1]):
        if budget_key == "daemon_pull":
            raise SshError(
                f"the pull on {alias} did not finish: the connection ended; "
                "Ollama keeps what arrived, the same pull resumes"
            )
        raise SshError(f"the connection to {alias} ended before the answer")
    return envelopes


def _envelope(alias: str, budget_key: str, line: bytes) -> dict:
    form = f"{alias} answered in a form this modelroom does not read"
    try:
        payload = json.loads(line)
    except (ValueError, RecursionError):
        raise SshError(form) from None
    if not isinstance(payload, dict):
        raise SshError(form)
    protocol = payload.get("protocol")
    fields = [key for key in ("progress", "result", "error") if key in payload]
    if protocol != PROTOCOL_VERSION and budget_key == "hello" and isinstance(protocol, int):
        raise SshError(f"modelroom on {alias} speaks protocol {protocol}; this one speaks 1; update the older one")
    if protocol != PROTOCOL_VERSION or len(fields) != 1:
        raise SshError(form)
    return payload


__all__ = [
    "ALIAS_RE",
    "FALLBACK_PROGRAM",
    "INVALID_ALIAS",
    "Limits",
    "NO_SSH",
    "OP_BUDGETS",
    "PRIMARY_PROGRAM",
    "PROTOCOL_VERSION",
    "ProcessOutput",
    "SSH_OPTIONS",
    "SshError",
    "SshRunner",
    "SubprocessSsh",
    "build_argv",
    "call",
    "default_ssh",
    "is_alias",
]
