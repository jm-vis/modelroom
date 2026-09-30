"""In-process fake `ssh` (FR-B1): a `SshRunner` that redirects a call into `remote_server.main`
with a fake daemon/room/probes, instead of spawning a real process (`Teststrategie ohne echten
Server`). Streams like the real transport: each line `remote_server.main` writes reaches `on_line`
**inside** that `write`, before the server has written its next line or returned -- never collected
first. An exception `on_line` raises (a form fault, a callback of the caller) stops the server there
and leaves `run` unchanged, as the real transport's own `finally` would. `FaultSsh` returns a canned
`ProcessOutcome` instead -- exit codes, stderr patterns, timeouts and over-limit answers.

The real process boundary is proven with a script as the `ssh` program (`SubprocessSsh(program=
[sys.executable, script])`, `tests/test_ssh_process.py`), not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelroom.remote_server import ServerContext, main as server_main
from modelroom.ssh import Limits, ProcessOutcome


class _Stop(BaseException):
    """Ends `remote_server.main` from inside its own `write` (it catches `Exception`, not this)."""


@dataclass
class FakeSsh:
    """Redirects a connection to `alias` into `remote_server.main(op, stdin, write, context)` --
    the same round trip a real ssh call to a real `modelroom remote <op>` would make, minus the
    process. `calls` records `(alias, op, stdin)` once per connection; `lines_seen` every line
    `on_line` was given, across every call this fake ever answered."""

    context: ServerContext
    known_programs: tuple[str, ...] = ("modelroom",)
    calls: list[tuple[str, str, bytes]] = field(default_factory=list)
    lines_seen: list[bytes] = field(default_factory=list)

    def run(self, argv, stdin: bytes, total, per_line, limits: Limits, on_line) -> ProcessOutcome:
        alias, program, _remote, op = argv[-4], argv[-3], argv[-2], argv[-1]
        self.calls.append((alias, op, stdin))
        if program not in self.known_programs:
            return ProcessOutcome(127)
        read = [0]
        raised: list[BaseException] = []
        over: list[bool] = []

        def write(line: bytes) -> None:
            read[0] += len(line) + 1
            if len(line) > limits.line_bytes or (limits.total_bytes is not None and read[0] > limits.total_bytes):
                over.append(True)
                raise _Stop()
            self.lines_seen.append(line)
            try:
                on_line(line + b"\n")
            except BaseException as exc:  # noqa: BLE001 - handed back unchanged below
                raised.append(exc)
                raise _Stop() from None

        try:
            code = server_main(op, stdin, write, self.context)
        except _Stop:
            if raised:
                raise raised[0] from None
            return ProcessOutcome(None, over_limit=bool(over))
        return ProcessOutcome(code)


@dataclass
class FaultSsh:
    """A canned `ProcessOutcome` for every call -- one connection-level fault, argv unchecked. When
    `lines` is given, each is offered to `on_line` first (in order), exactly like a real answer
    that then ends the way `outcome` says (a mid-pull disconnect, a timeout after some progress)."""

    outcome: ProcessOutcome
    lines: tuple[bytes, ...] = ()
    calls: list[tuple] = field(default_factory=list)

    def run(self, argv, stdin: bytes, total, per_line, limits: Limits, on_line) -> ProcessOutcome:
        self.calls.append((argv, stdin, total, per_line, limits))
        for line in self.lines:
            on_line(line)
        return self.outcome


__all__ = ["FakeSsh", "FaultSsh"]
