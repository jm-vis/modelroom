"""`modelroom remote <op>`: the CLI entry point of the counterpart (AP9-FR, FR-B1, B2).

Reads at most `remote_server.input_limit(op) + 1` bytes from stdin (never the whole of an
unbounded stream before the size is judged, Codex r1 finding 2), hands them to
`remote_server.main`, and writes every answer line to stdout, one JSON object per line, flushed as
it is written -- a pull's `progress` lines reach the SSH pipe while the pull runs, not only once it
ends. Never interactive, never a terminal: `modelroom --answers` and every dialog question belong
to the guided mode, not to this command.
"""

from __future__ import annotations

import sys
from typing import BinaryIO

from .remote_server import OPS, ServerContext, input_limit, main as server_main


def run_remote(op: str, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    """Read stdin (bounded), dispatch `op`, write every line as it is produced; the exit code is
    always `0` once a line was written -- every fault of the call itself is an `error` line
    (`remote_server.main`)."""
    reader = stdin if stdin is not None else sys.stdin.buffer
    writer = stdout if stdout is not None else sys.stdout.buffer
    payload = reader.read(input_limit(op) + 1)

    def write(line: bytes) -> None:
        writer.write(line + b"\n")
        writer.flush()

    return server_main(op, payload, write, ServerContext())


__all__ = ["OPS", "run_remote"]
