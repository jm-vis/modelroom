"""`modelroom remote <op>`: the CLI entry point of the gegenstelle (AP9-FR, FR-B1, B2).

Reads exactly one JSON object from stdin (bytes, not text: a line ending must never be assumed),
hands it to `remote_server.main`, and writes every answer line to stdout, one JSON object per line,
flushed as it is written -- a pull's `progress` lines reach the SSH pipe while the pull runs, not
only once it ends. Never interactive, never a terminal: `modelroom --answers` and every dialog
question belong to the guided mode, not to this command.
"""

from __future__ import annotations

import sys
from typing import BinaryIO

from .remote_server import OPS, ServerContext, main as server_main

MIN_VERSION_TEXT = "modelroom remote needs modelroom 0.2.0 or newer on this computer"


def run_remote(op: str, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    """Read stdin to EOF, dispatch `op`, write every line; the exit code is always `0` once a line
    was written -- every fault of the call itself is an `error` line (`remote_server.main`)."""
    reader = stdin if stdin is not None else sys.stdin.buffer
    writer = stdout if stdout is not None else sys.stdout.buffer
    payload = reader.read()
    lines, code = server_main(op, payload, ServerContext())
    for line in lines:
        writer.write(line + b"\n")
        writer.flush()
    return code


__all__ = ["OPS", "run_remote"]
