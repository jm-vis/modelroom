"""The counterpart's own selftest step (AP9-FR, FR-B1): `modelroom remote hello`, run as a real
local process (never in-process), stdin `{}`, stdout checked against the wire envelope.

`scripts/selftest.py` stays without SSH itself (no alias, no `ssh` process): this is the one place
the self-test proves the *other end* of the SSH way -- the `modelroom remote <op>` entry point a
real server runs -- answers correctly when called exactly as ssh would call it (a JSON object on
stdin, EOF, one JSON line on stdout).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def remote_step() -> tuple[str, bool, str]:
    """Step (13): `modelroom remote hello`, a real process, stdin `{}`, checked as B2 describes it."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from modelroom.cli import main; sys.exit(main(['remote', 'hello']))"],
            input=b"{}",
            capture_output=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "(13) modelroom remote hello answers the wire envelope", False, f"the process could not be run: {exc}"
    return _checked(result)


def _checked(result: subprocess.CompletedProcess) -> tuple[str, bool, str]:
    name = "(13) modelroom remote hello answers the wire envelope"
    if result.returncode != 0:
        return name, False, f"exit {result.returncode}: {result.stderr.decode('utf-8', errors='replace')[:300]}"
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return name, False, f"expected exactly one line, got {len(lines)}: {result.stdout[:300]!r}"
    try:
        payload = json.loads(lines[0])
    except ValueError as exc:
        return name, False, f"the line is not JSON: {exc}"
    problem = _envelope_problem(payload)
    if problem is not None:
        return name, False, problem
    return name, True, f"protocol 1, ollama {payload['result']['ollama']!r}, system {payload['result']['system']!r}"


def _envelope_problem(payload: object) -> str | None:
    if not isinstance(payload, dict) or payload.get("protocol") != 1:
        return f"no protocol-1 envelope: {payload!r}"
    if "result" not in payload:
        return f"no 'result' field: {payload!r}"
    result = payload["result"]
    needed = {"version", "system", "ollama", "agent_forwarded", "os_fingerprint"}
    if not isinstance(result, dict) or not needed <= set(result):
        return f"'result' is missing a field of {sorted(needed)}: {result!r}"
    return None


if __name__ == "__main__":
    _name, ok, detail = remote_step()
    print(f"{_name}: {'OK' if ok else 'FAIL'}\n    {detail}")
    sys.exit(0 if ok else 1)
