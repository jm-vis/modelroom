"""The SSH transport to another computer's `modelroom remote <op>` (AP9-FR, FR-B1, B1).

One process, one fixed argument vector: the system `ssh` (`shutil.which`), the safety options B1
names (never overridable by a user's own config), `--`, the checked host alias, then the program
path and the fixed op name -- never a name, a path or a package name of any kind. All data of the
call goes as one JSON object over stdin, then stdin is closed (EOF); the answer is JSON lines on
stdout, each one envelope of "The remote protocol v1", read **to EOF** and checked line by line as
it arrives: a `progress` line is handed on at once (never collected), exactly one terminal line
(`result` or `error`) ends the answer, and any line after it is a form fault.

`SshRunner` is the one injectable seam (`run(argv, stdin, total, per_line, limits, on_line)`): the
real process is `ssh_process.SubprocessSsh`; a test builds a `FakeSsh` that redirects into
`remote_server.main` in-process, a `FaultSsh` with a canned `ProcessOutcome`, or a
`SubprocessSsh(program=[sys.executable, script])` whose script stands in for `ssh`.
`default_ssh()` refuses under pytest -- no test may reach a real `ssh`.

Every failure becomes one fixed local sentence with the alias in it: ssh's own stderr and any text
the server sent are never shown, stored or put in `run.problems` -- stderr is classified by pattern
and dropped, and an `error` line carries only one of five known codes, each with its own sentence.
"""

from __future__ import annotations

import json
import shutil
import sys
from typing import Callable, Protocol, Sequence

from . import remote_protocol
from .neutral import ALIAS_RE, is_alias  # noqa: F401 -- re-exported, B1's own name for the check
from .remote_protocol import ERROR_CODES, OP_BUDGETS, WIRE_OP, OpBudget  # noqa: F401 -- re-exported
from .ssh_process import Limits, OnLine, ProcessOutcome, SubprocessSsh

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
PROTOCOL_VERSION = remote_protocol.PROTOCOL_VERSION

# One fixed sentence per error code of the wire (B1, the table's last row; CONTRACTS.md,
# "The remote protocol v1"). The server never sends text; these are the laptop's own words.
ERROR_SENTENCES: dict[str, str] = {
    "bad_request": "modelroom on {alias} could not read the request of this one; update the older one",
    "call_refused": "modelroom on {alias} refused the call to its Ollama",
    "ollama_unreachable": "Ollama does not answer on {alias}",
    "too_large": "the request to {alias} was larger than modelroom there reads",
    "internal": "modelroom on {alias} failed on its own side before it could answer",
}
assert set(ERROR_SENTENCES) == set(ERROR_CODES), ERROR_SENTENCES

_FORM = "{alias} answered in a form this modelroom does not read"
_PULL_RESUMES = "the pull on {alias} did not finish: the connection ended; Ollama keeps what arrived, the same pull resumes"
_CONNECTION_ENDED = "the connection to {alias} ended before the answer"
_ENVELOPE_FIELDS = ("progress", "result", "error")


class SshError(Exception):
    """One fixed local sentence, with the alias already in it; never ssh's stderr, never server text."""


class SshRunner(Protocol):
    def run(
        self, argv: Sequence[str], stdin: bytes, total: float | None, per_line: float | None, limits: Limits, on_line: OnLine
    ) -> ProcessOutcome: ...


def default_ssh() -> "SshRunner":
    """This computer's real `ssh`, refused under pytest (the guard `room.real_room_probes` uses)."""
    if "pytest" in sys.modules:
        raise RuntimeError("the real ssh runner was reached from a test; pass ssh= or use a fixture")
    return SubprocessSsh()


def build_argv(ssh_path: str, alias: str, op: str, program: str = PRIMARY_PROGRAM) -> list[str]:
    """The fixed argument vector (B1): options, `--`, the alias, then the program and the op --
    never anything else, and never through a shell string."""
    return [ssh_path, *SSH_OPTIONS, "--", alias, program, "remote", op]


# --- reading the answer: one envelope per line, exactly one terminal line, nothing after it --------


def _no_duplicate_keys(pairs: list) -> dict:
    keys = [key for key, _value in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


class _CallState:
    """One call's own `on_line`: every line is parsed and checked the moment it arrives -- the
    envelope (`protocol` exactly the integer 1, exactly one more field of `progress`/`result`/
    `error`), then its content (`remote_protocol`, strict per op). A `progress` line is handed on
    at once; the first terminal line is kept; any line after it is a form fault (`SshError`)."""

    def __init__(self, alias: str, budget_key: str, on_progress: Callable[[dict], None] | None) -> None:
        self.alias = alias
        self.budget_key = budget_key
        self.on_progress = on_progress
        self.terminal: dict | None = None
        self.saw_progress = False

    def _form(self) -> SshError:
        return SshError(_FORM.format(alias=self.alias))

    def on_line(self, raw: bytes) -> None:
        field, value = self._envelope(raw)
        if self.terminal is not None:
            raise self._form()
        if field == "error":
            if not isinstance(value, str) or value not in ERROR_CODES:
                raise self._form()
            self.terminal = {"error": value}
        elif field == "result":
            self.terminal = {"result": self._validated(remote_protocol.validate_result, value)}
        else:
            validated = self._validated(remote_protocol.validate_progress, value)
            self.saw_progress = True
            if self.on_progress is not None:
                self.on_progress(validated)

    def _envelope(self, raw: bytes) -> tuple[str, object]:
        try:
            payload = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
        except (ValueError, RecursionError):
            raise self._form() from None
        if not isinstance(payload, dict):
            raise self._form()
        protocol = payload.get("protocol")
        whole = type(protocol) is int  # never `True`, never `1.0`
        if whole and protocol != PROTOCOL_VERSION and self.budget_key == "hello":
            raise SshError(f"modelroom on {self.alias} speaks protocol {protocol}; this one speaks 1; update the older one")
        fields = [key for key in payload if key != "protocol"]
        if not whole or protocol != PROTOCOL_VERSION or len(fields) != 1 or fields[0] not in _ENVELOPE_FIELDS:
            raise self._form()
        return fields[0], payload[fields[0]]

    def _validated(self, validate: Callable[[str, object], dict], value: object) -> dict:
        try:
            return validate(self.budget_key, value)
        except ValueError:
            raise self._form() from None


# --- one call ----------------------------------------------------------------------------------


def _run_once(runner: SshRunner, ssh_path: str, alias: str, budget_key: str, program: str, payload: bytes, on_progress) -> tuple[_CallState, ProcessOutcome]:
    budget = OP_BUDGETS[budget_key]
    argv = build_argv(ssh_path, alias, WIRE_OP[budget_key], program)
    state = _CallState(alias, budget_key, on_progress)
    outcome = runner.run(argv, payload, budget.total_seconds, budget.per_line_seconds, budget.limits, state.on_line)
    return state, outcome


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
    """One call to `alias`'s `modelroom remote <op>`: `{"result": {...}}`, validated, or a fixed-
    sentence `SshError` (an `error` line of the server included). `budget_key` is one of
    `OP_BUDGETS`. `on_progress` is called for every `progress` line, in order, as it arrives.
    `ssh_path` is the resolved path of the system `ssh`; without one it is resolved here (`which`)
    -- missing, `NO_SSH` (B1), before any runner is asked to do anything."""
    if not is_alias(alias):
        raise SshError(INVALID_ALIAS)
    resolved = ssh_path if ssh_path is not None else which("ssh")
    if resolved is None:
        raise SshError(NO_SSH)
    payload = json.dumps(request).encode("utf-8")
    state, outcome = _run_once(runner, resolved, alias, budget_key, PRIMARY_PROGRAM, payload, on_progress)
    if _program_missing(state, outcome):
        state, outcome = _run_once(runner, resolved, alias, budget_key, FALLBACK_PROGRAM, payload, on_progress)
    return _resolved(state, outcome)


def _program_missing(state: _CallState, outcome: ProcessOutcome) -> bool:
    return outcome.returncode == 127 and state.terminal is None and not state.saw_progress and not (outcome.timed_out or outcome.over_limit)


def _resolved(state: _CallState, outcome: ProcessOutcome) -> dict:
    """The answer as read to EOF, or the one sentence for how it ended (B1's own table)."""
    alias, budget = state.alias, OP_BUDGETS[state.budget_key]
    if outcome.over_limit:
        raise SshError(f"{alias} answered with more than modelroom reads")
    if outcome.timed_out:
        raise SshError(f"{alias} did not answer within {int(budget.total_seconds or budget.per_line_seconds or 0)} s")
    if state.terminal is not None:
        if "error" in state.terminal:
            raise SshError(ERROR_SENTENCES[state.terminal["error"]].format(alias=alias))
        return state.terminal
    return _without_answer(state, outcome)


def _without_answer(state: _CallState, outcome: ProcessOutcome) -> dict:
    """No terminal line arrived: the exit code and stderr say why. A pull's resumable hint only where
    the pull had begun (a `progress` line arrived); a host-key or login fault before that keeps its
    own sentence."""
    alias, code = state.alias, outcome.returncode
    if code == 127 and not state.saw_progress:
        raise SshError(f"modelroom is not installed on {alias}, or not on the PATH of an SSH command (see README)")
    if code == 2 and not state.saw_progress:
        raise SshError(f"modelroom on {alias} is older than 0.2.0; update it there")
    if state.budget_key == "daemon_pull" and state.saw_progress:
        raise SshError(_PULL_RESUMES.format(alias=alias))
    if code == 255:
        raise SshError(_classify_255(alias, outcome.stderr))
    raise SshError(_CONNECTION_ENDED.format(alias=alias))


def _classify_255(alias: str, stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace")
    if "REMOTE HOST IDENTIFICATION HAS CHANGED" in text or "Host key verification failed" in text:
        return f"the host key of {alias} is not known here or has changed; connect once with ssh {alias} yourself and check it"
    if "Permission denied" in text:
        return f"{alias} refused the login; modelroom uses your SSH keys and agent, never a password"
    return f"{alias} could not be reached over SSH; try ssh {alias} yourself to see why"


__all__ = [
    "ALIAS_RE",
    "ERROR_SENTENCES",
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
