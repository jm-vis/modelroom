"""In-process fake `ssh` (FR-B1): a `SshRunner` that redirects a call into `remote_server.main`
with a fake daemon/room/probes, instead of spawning a real process (`Teststrategie ohne echten
Server`). `FaultSsh` returns a canned `ProcessOutput` instead -- exit codes, stderr patterns,
timeouts and over-limit answers, without any process at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelroom.remote_server import ServerContext, main as server_main
from modelroom.ssh import Limits, ProcessOutput


@dataclass
class FakeSsh:
    """Redirects a connection to `alias` into `remote_server.main(op, stdin, context)` -- the same
    round trip a real ssh call to a real `modelroom remote <op>` would make, minus the process."""

    context: ServerContext
    known_programs: tuple[str, ...] = ("modelroom",)
    calls: list[tuple[str, str, bytes]] = field(default_factory=list)

    def run(self, argv, stdin: bytes, total, per_line, limits: Limits) -> ProcessOutput:
        alias, program, _remote, op = argv[-4], argv[-3], argv[-2], argv[-1]
        self.calls.append((alias, op, stdin))
        if program not in self.known_programs:
            return ProcessOutput([], 127)
        lines, code = server_main(op, stdin, self.context)
        return ProcessOutput([line + b"\n" for line in lines], code)


@dataclass
class FaultSsh:
    """A canned `ProcessOutput` for every call -- one connection-level fault, argv unchecked."""

    outcome: ProcessOutput
    calls: list[tuple] = field(default_factory=list)

    def run(self, argv, stdin: bytes, total, per_line, limits: Limits) -> ProcessOutput:
        self.calls.append((argv, stdin, total, per_line, limits))
        return self.outcome


__all__ = ["FakeSsh", "FaultSsh"]
