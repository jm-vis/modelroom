"""In-process fake `ssh` (FR-B1): a `SshRunner` that redirects a call into `remote_server.main`
with a fake daemon/room/probes, instead of spawning a real process (`Teststrategie ohne echten
Server`). Streams exactly like the real transport: each line `remote_server.main` writes is handed
to `on_line` immediately, one at a time, so a test can watch a pull's progress arrive without ever
collecting the whole answer first. `FaultSsh` returns a canned `ProcessOutcome` instead -- exit
codes, stderr patterns, timeouts and over-limit answers, without any process at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelroom.remote_server import ServerContext, main as server_main
from modelroom.ssh import Limits, ProcessOutcome


@dataclass
class FakeSsh:
    """Redirects a connection to `alias` into `remote_server.main(op, stdin, write, context)` --
    the same round trip a real ssh call to a real `modelroom remote <op>` would make, minus the
    process. `calls` records `(alias, op, stdin)` once per connection; `lines_seen` every line
    `on_line` was actually given, across every call this fake ever answered."""

    context: ServerContext
    known_programs: tuple[str, ...] = ("modelroom",)
    calls: list[tuple[str, str, bytes]] = field(default_factory=list)
    lines_seen: list[bytes] = field(default_factory=list)

    def run(self, argv, stdin: bytes, total, per_line, limits: Limits, on_line) -> ProcessOutcome:
        alias, program, _remote, op = argv[-4], argv[-3], argv[-2], argv[-1]
        self.calls.append((alias, op, stdin))
        if program not in self.known_programs:
            return ProcessOutcome(127)
        lines: list[bytes] = []

        def write(line: bytes) -> None:
            lines.append(line)

        code = server_main(op, stdin, write, self.context)
        for line in lines:
            self.lines_seen.append(line)
            if len(line) > limits.line_bytes:
                return ProcessOutcome(None, over_limit=True)
            if not on_line(line):
                return ProcessOutcome(code)
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
            if not on_line(line):
                return ProcessOutcome(self.outcome.returncode)
        return self.outcome


__all__ = ["FakeSsh", "FaultSsh"]
