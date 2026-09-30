"""The SSH transport to another computer's `modelroom remote <op>` (AP9-FR, FR-B1, B1).

One process, one fixed argument vector: the system `ssh` (`shutil.which`), the safety options B1
names (never overridable by a user's own config), `--`, the checked host alias, then the program
path and the fixed op name -- never a name, a path or a package name of any kind. All data of the
call goes as one JSON object over stdin, then stdin is closed (EOF); the answer is JSON lines on
stdout, each one envelope of "The remote protocol v1" (`modelroom/remote_server.py`), each read
and handed to its caller **as it arrives** -- a pull's `progress` lines are never collected, so a
laptop reading a multi-gigabyte download holds at most one line of it at a time.

`SshRunner` is the one injectable seam (`run(argv, stdin, total, per_line, limits, on_line)`); a
test either builds a `FakeSsh` that redirects into `remote_server.main` in-process (`tests/
fixture_ssh.py`), a `FaultSsh` with a canned `ProcessOutcome`, or runs a small Python script as the
real "ssh" program (`SubprocessSsh` with an injected path) to prove the process boundary itself:
the fixed argument vector, a bounded read, a bounded queue, a deadline counted from the moment the
process starts (never after the possibly-blocking `stdin.write`), and a guaranteed kill-and-reap on
every way the call can end -- a clean answer, a timeout, an over-limit line, or any exception,
`KeyboardInterrupt` included. `default_ssh()` refuses under pytest, the same guard `room.py` and
`measure.py` use for the real machine -- no test may reach a real `ssh`.

Every failure here becomes one fixed local sentence with the alias in it (Codex r1-2): ssh's own
stderr and any text the server sent are never shown, stored or put in `run.problems` -- classified
by pattern and dropped. `call()` is the one function that does this classification, together with
the strict per-op validation of "The remote protocol v1" (`remote_protocol.py`, Codex r1 finding 4);
`modelroom/remote_client.py` reads only its return value (the terminal envelope) or catches
`SshError`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Protocol, Sequence

from . import remote_protocol
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
class ProcessOutcome:
    """How one `SshRunner.run` call ended. `returncode` is `None` when this module stopped the
    process itself (a timeout or an over-limit line) rather than letting it exit on its own; a
    real runner enforces `total`/`per_line`/`limits`, a fake one for a behavior test returns
    exactly what it was built with."""

    returncode: int | None = None
    timed_out: bool = False
    over_limit: bool = False
    stderr: bytes = b""


# `on_line(raw_line)` is called once per line of stdout, in the order they arrive, as soon as each
# is read -- never collected first. It returns `True` to keep reading, `False` to stop (the
# terminal envelope line arrived and nothing more is needed); an exception it raises propagates
# out of `run()`, which still kills and reaps the process first (a `finally`, not a best effort).
OnLine = Callable[[bytes], bool]


class SshRunner(Protocol):
    def run(
        self, argv: Sequence[str], stdin: bytes, total: float | None, per_line: float | None, limits: Limits, on_line: OnLine
    ) -> ProcessOutcome: ...


def default_ssh() -> "SshRunner":
    """This computer's real `ssh`, refused under pytest (the guard `room.real_room_probes` uses)."""
    if "pytest" in sys.modules:
        raise RuntimeError("the real ssh runner was reached from a test; pass ssh= or use a fixture")
    return SubprocessSsh()


# --- the real process: bounded reads, a bounded queue, a deadline from process start --------------

_QUEUE_MAXSIZE = 8
_STOP_WAIT_SECONDS = 5.0


class SubprocessSsh:
    """The real transport. `stdout` and `stderr` are each drained by their own thread from the
    moment the process starts, so a full stderr pipe nobody is reading can never block a stdout
    read (the classic subprocess deadlock) and vice versa; stdin is written by a third thread, so
    a slow or stalled reader on the other end cannot delay the deadline, which starts right after
    `Popen` -- never after the (possibly blocking) write. Every way the call can end -- a clean
    answer, a timeout, an over-limit line, or any exception including `KeyboardInterrupt` -- goes
    through the same `finally`: terminate, wait, kill if it did not stop, wait again.
    """

    def run(self, argv: Sequence[str], stdin: bytes, total, per_line, limits: Limits, on_line: OnLine) -> ProcessOutcome:
        process = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        started = time.monotonic()
        stdout_q: "Queue[tuple[str, bytes] | None]" = Queue(maxsize=_QUEUE_MAXSIZE)
        stderr_state = _StderrState()
        threads = _start_threads(process, stdin, stdout_q, limits.line_bytes, stderr_state)
        try:
            return _consume(process, stdout_q, started, total, per_line, limits, on_line, stderr_state)
        finally:
            _stop(process)
            for thread in threads:
                thread.join(timeout=1.0)


class _StderrState:
    """stderr, drained continuously and kept bounded -- classified by pattern, never shown, so a
    server that writes gigabytes of stderr cannot grow this module's own memory with it."""

    _CAP = 64 * 1024

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._chunks.append(chunk)
            total = sum(len(piece) for piece in self._chunks)
            while total > self._CAP and len(self._chunks) > 1:
                total -= len(self._chunks.pop(0))

    def joined(self) -> bytes:
        with self._lock:
            return b"".join(self._chunks)


def _start_threads(process, stdin: bytes, stdout_q, line_bytes: int, stderr_state: _StderrState) -> list[threading.Thread]:
    stdout_thread = threading.Thread(target=_read_stdout, args=(process.stdout, stdout_q, line_bytes), daemon=True)
    stderr_thread = threading.Thread(target=_read_stderr, args=(process.stderr, stderr_state), daemon=True)
    stdin_thread = threading.Thread(target=_feed_stdin, args=(process, stdin), daemon=True)
    for thread in (stdout_thread, stderr_thread, stdin_thread):
        thread.start()
    return [stdout_thread, stderr_thread, stdin_thread]


def _feed_stdin(process: subprocess.Popen, data: bytes) -> None:
    try:
        process.stdin.write(data)
    except (OSError, ValueError):
        pass
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass


def _read_stdout(stream, queue: "Queue[tuple[str, bytes] | None]", max_line_bytes: int) -> None:
    """One line at a time, bounded: `readline(size)` never returns more than `size` bytes even
    without a newline in sight, so a single, endless "line" is caught on its very first read, not
    after the whole of it has arrived. Blocks on `queue.put` when the consumer falls behind
    (`_QUEUE_MAXSIZE`) -- backpressure, not an unbounded buffer of read-ahead lines."""
    size = max_line_bytes + 1
    try:
        while True:
            raw = stream.readline(size)
            if not raw:
                queue.put(None)
                return
            over = len(raw) > max_line_bytes
            queue.put(("over", raw) if over else ("line", raw))
            if over:
                return
    except (OSError, ValueError):
        queue.put(None)


def _read_stderr(stream, state: _StderrState) -> None:
    try:
        while True:
            raw = stream.read(65536)
            if not raw:
                return
            state.append(raw)
    except (OSError, ValueError):
        return


def _consume(process, queue, started: float, total, per_line, limits: Limits, on_line: OnLine, stderr_state: _StderrState) -> ProcessOutcome:
    total_read = 0
    while True:
        wait = _next_wait(started, total, per_line)
        try:
            item = queue.get(timeout=wait) if wait is not None else queue.get()
        except Empty:
            return ProcessOutcome(None, timed_out=True, stderr=stderr_state.joined())
        if item is None:
            return ProcessOutcome(_stop(process), stderr=stderr_state.joined())
        kind, raw = item
        if kind == "over":
            return ProcessOutcome(None, over_limit=True, stderr=stderr_state.joined())
        total_read += len(raw)
        if limits.total_bytes is not None and total_read > limits.total_bytes:
            return ProcessOutcome(None, over_limit=True, stderr=stderr_state.joined())
        if not on_line(raw):
            return ProcessOutcome(_stop(process), stderr=stderr_state.joined())


def _next_wait(started: float, total: float | None, per_line: float | None) -> float | None:
    remaining_total = None if total is None else max(total - (time.monotonic() - started), 0.0)
    waits = [w for w in (remaining_total, per_line) if w is not None]
    return min(waits) if waits else None


def _stop(process: subprocess.Popen) -> int | None:
    """Terminate, wait, kill if that did not work, wait once more -- idempotent (a process already
    reaped is left alone) and called from every exit path, `run`'s own `finally` included, so
    nothing here ever leaves a process unreaped. Returns the real exit code once it is known."""
    code = process.poll()
    if code is not None:
        return code
    for signal in (process.terminate, process.kill):
        try:
            signal()
        except OSError:
            pass
        try:
            return process.wait(timeout=_STOP_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            continue
    return process.poll()


# --- classifying a finished process into one fixed sentence, or the validated terminal envelope ---


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


_PULL_RESUMES = "the pull on {alias} did not finish: the connection ended; Ollama keeps what arrived, the same pull resumes"
_CONNECTION_ENDED = "the connection to {alias} ended before the answer"


class _CallState:
    """One call's own `on_line`: parses the envelope, validates its `progress`/`result` shape
    (`remote_protocol.py`, Codex r1 finding 4) before anything reads a field of it, forwards a
    `progress` line and keeps reading, or keeps the terminal line and stops. Raises `SshError`
    directly the moment a line does not fit -- the process is still killed and reaped, by `run`'s
    own `finally`, whichever way this call ends."""

    def __init__(self, alias: str, budget_key: str, on_progress: Callable[[dict], None] | None) -> None:
        self.alias = alias
        self.budget_key = budget_key
        self.on_progress = on_progress
        self.terminal: dict | None = None
        self.saw_progress = False

    def on_line(self, raw: bytes) -> bool:
        envelope = _parsed_envelope(self.alias, self.budget_key, raw)
        if "progress" in envelope:
            self.saw_progress = True
            validated = _validated(self.alias, self.budget_key, "progress", envelope["progress"])
            if self.on_progress is not None:
                self.on_progress(validated)
            return True
        self.terminal = envelope
        if "result" in envelope:
            envelope["result"] = _validated(self.alias, self.budget_key, "result", envelope["result"])
        return False


def _parsed_envelope(alias: str, budget_key: str, raw: bytes) -> dict:
    form = f"{alias} answered in a form this modelroom does not read"
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        raise SshError(form) from None
    if not isinstance(payload, dict):
        raise SshError(form)
    protocol = payload.get("protocol")
    fields = [key for key in ("progress", "result", "error") if key in payload]
    if protocol != PROTOCOL_VERSION and budget_key == "hello" and isinstance(protocol, int) and not isinstance(protocol, bool):
        raise SshError(f"modelroom on {alias} speaks protocol {protocol}; this one speaks 1; update the older one")
    if protocol != PROTOCOL_VERSION or len(fields) != 1:
        raise SshError(form)
    return payload


def _validated(alias: str, budget_key: str, field_name: str, data: object) -> dict:
    form = f"{alias} answered in a form this modelroom does not read"
    try:
        if field_name == "progress":
            return remote_protocol.validate_progress(budget_key, data)
        return remote_protocol.validate_result(budget_key, data)
    except ValueError:
        raise SshError(form) from None


def _run_once(runner: SshRunner, ssh_path: str, alias: str, budget_key: str, program: str, payload: bytes, on_progress) -> tuple[dict | None, ProcessOutcome, bool]:
    budget = OP_BUDGETS[budget_key]
    argv = build_argv(ssh_path, alias, WIRE_OP[budget_key], program)
    state = _CallState(alias, budget_key, on_progress)
    outcome = runner.run(argv, payload, budget.total_seconds, budget.per_line_seconds, budget.limits, state.on_line)
    return state.terminal, outcome, state.saw_progress


def call(
    alias: str,
    budget_key: str,
    request: dict,
    runner: SshRunner,
    *,
    ssh_path: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """One call to `alias`'s `modelroom remote <op>`: the validated terminal envelope (`result` or
    `error`), or a fixed-sentence `SshError`. `budget_key` is one of `OP_BUDGETS`. `on_progress`,
    given, is called for every `progress` line, in order, **as it arrives** -- never collected.
    `ssh_path` is the resolved path of the system `ssh`; without one it is resolved here (`which`,
    `shutil.which` by default) -- missing, `NO_SSH` (B1), before any runner is ever asked to do
    anything, real or fake."""
    if not is_alias(alias):
        raise SshError(INVALID_ALIAS)
    resolved = ssh_path if ssh_path is not None else which("ssh")
    if resolved is None:
        raise SshError(NO_SSH)
    payload = json.dumps(request).encode("utf-8")
    terminal, outcome, saw_progress = _run_once(runner, resolved, alias, budget_key, PRIMARY_PROGRAM, payload, on_progress)
    if terminal is None and not saw_progress and outcome.returncode == 127:
        terminal, outcome, saw_progress = _run_once(runner, resolved, alias, budget_key, FALLBACK_PROGRAM, payload, on_progress)
    return _resolved(alias, budget_key, terminal, outcome, saw_progress)


def _resolved(alias: str, budget_key: str, terminal: dict | None, outcome: ProcessOutcome, saw_progress: bool) -> dict:
    budget = OP_BUDGETS[budget_key]
    if outcome.timed_out:
        raise SshError(f"{alias} did not answer within {int(budget.total_seconds or budget.per_line_seconds)} s")
    if outcome.over_limit:
        raise SshError(f"{alias} answered with more than modelroom reads")
    if outcome.returncode == 127:
        raise SshError(f"modelroom is not installed on {alias}, or not on the PATH of an SSH command (see README)")
    if outcome.returncode == 2 and terminal is None and not saw_progress:
        raise SshError(f"modelroom on {alias} is older than 0.2.0; update it there")
    if terminal is not None:
        return terminal
    # Any premature end of a pull -- EOF or exit 255, with or without progress already seen --
    # gets the resumable hint (B1's own table): Ollama may already hold part of the download, and
    # this module cannot tell the difference between "nothing started" and "started, then cut".
    if budget_key == "daemon_pull":
        raise SshError(_PULL_RESUMES.format(alias=alias))
    if outcome.returncode == 255:
        raise SshError(_classify_255(alias, outcome.stderr))
    raise SshError(_CONNECTION_ENDED.format(alias=alias))


__all__ = [
    "ALIAS_RE",
    "FALLBACK_PROGRAM",
    "INVALID_ALIAS",
    "Limits",
    "NO_SSH",
    "OP_BUDGETS",
    "PRIMARY_PROGRAM",
    "PROTOCOL_VERSION",
    "ProcessOutcome",
    "SSH_OPTIONS",
    "SshError",
    "SshRunner",
    "SubprocessSsh",
    "build_argv",
    "call",
    "default_ssh",
    "is_alias",
]
