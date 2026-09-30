"""The gegenstelle `modelroom remote <op>` (AP9-FR, FR-B1, B2): what runs on the other computer.

Not interactive: reads exactly one JSON object from stdin (at most `MAX_INPUT_BYTES`), writes JSON
lines to stdout, and writes nothing to the disk of the machine it runs on -- except what Ollama
itself writes during a pull. Every line is one envelope object, `{"protocol": 1, <one of "progress",
"result", "error">}` (CONTRACTS.md, "The remote protocol v1"); free text of this machine -- its own
error text, a foreign package name, its host name -- never stands in one, because nothing here reads
`socket.gethostname()` or forwards a caught exception's own message to the wire.

`candidates`/`measure` are FR-B2's own ops; this module answers only `hello`, `profile`, `room` and
`daemon` (0.1.0's four verbs plus the fixed HTTP allowlist `daemon.py` already has). `main()` is the
one entry point the CLI (`cli_remote.py`) and the in-process fake ssh of the test suite both call.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .contracts import PROFILE_ID_RE
from .daemon import ALLOWED_CALLS, Daemon, DaemonError, LocalDaemon
from .llmfit import read_llmfit_reference
from .measure import MeasuredHardware, Probes, build_profile, measure_hardware, read_os_identity
from .measurements import EXPORT_SCHEMA_VERSION, ExportObject
from .neutral import neutral_version
from .profile import MACHINE_CLASSES, os_fingerprint
from .pull_choice import PULL_NAME_RE
from . import room as room_module
from .room import RoomProbes, free_disk, free_vram_mib
from .room import UNCLEAR_REASON as _UNCLEAR

MAX_INPUT_BYTES = 1 * 1024 * 1024
PROTOCOL_VERSION = 1
OPS = ("hello", "profile", "room", "daemon")
# The 0.1.0 default (`config.LlmfitConfig.min_version`): the server has no configuration to read one
# from, so the profile op uses the same floor a fresh configuration would.
_DEFAULT_LLMFIT_MIN_VERSION = "1.1.16"
REMOTE_VERSION = "0.2.0"

ERROR_BAD_REQUEST = "bad_request"
ERROR_CALL_REFUSED = "call_refused"
ERROR_OLLAMA_UNREACHABLE = "ollama_unreachable"
ERROR_TOO_LARGE = "too_large"
ERROR_INTERNAL = "internal"


class RemoteRequestError(Exception):
    """One of the four error codes the wire allows; `main` turns it into an `error` line."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ServerContext:
    """Everything one call reads from the machine it runs on, injectable for a test."""

    probes: Probes = field(default_factory=Probes)
    room: RoomProbes = field(default_factory=lambda: room_module.default_room_probes())
    daemon: Daemon = field(default_factory=LocalDaemon)
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    environ: dict = field(default_factory=lambda: dict(os.environ))


def _envelope(**fields: object) -> bytes:
    return json.dumps({"protocol": PROTOCOL_VERSION, **fields}, separators=(",", ":")).encode("utf-8")


def _result(payload: dict) -> bytes:
    return _envelope(result=payload)


def _error(code: str) -> bytes:
    return _envelope(error=code)


def _progress(payload: dict) -> bytes:
    return _envelope(progress=payload)


def read_request(stdin: bytes) -> dict:
    """The one JSON object of the call, or `RemoteRequestError` (`too_large`/`bad_request`)."""
    if len(stdin) > MAX_INPUT_BYTES:
        raise RemoteRequestError(ERROR_TOO_LARGE)
    try:
        payload = json.loads(stdin.decode("utf-8"))
    except (ValueError, RecursionError, UnicodeDecodeError):
        raise RemoteRequestError(ERROR_BAD_REQUEST) from None
    if not isinstance(payload, dict):
        raise RemoteRequestError(ERROR_BAD_REQUEST)
    return payload


def hello_result(context: ServerContext) -> dict:
    identity = read_os_identity(context.probes.platform, context.probes.runner, context.probes.read_text)
    fingerprint = os_fingerprint(identity.raw_id) if identity.raw_id else "none"
    return {
        "version": neutral_version(REMOTE_VERSION),
        "system": context.probes.platform,
        "ollama": "reachable" if _ollama_reachable(context.daemon) else "unreachable",
        "agent_forwarded": bool(context.environ.get("SSH_AUTH_SOCK")),
        "os_fingerprint": fingerprint,
    }


def _ollama_reachable(daemon: Daemon) -> bool:
    try:
        daemon("GET", "/api/version", None)
    except DaemonError:
        return False
    return True


def profile_result(payload: dict, context: ServerContext) -> dict:
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or not PROFILE_ID_RE.fullmatch(profile_id):
        raise RemoteRequestError(ERROR_BAD_REQUEST)
    machine_class = payload.get("machine_class")
    if machine_class is not None and machine_class not in MACHINE_CLASSES:
        raise RemoteRequestError(ERROR_BAD_REQUEST)
    reference = read_llmfit_reference(context.probes.runner, _DEFAULT_LLMFIT_MIN_VERSION)
    measured: MeasuredHardware = measure_hardware(
        context.probes.platform, context.probes.runner, context.probes.read_text, context.probes.memory_bytes,
        False, machine_class, context.probes.list_dir, reference,
    )
    profile = build_profile(measured, reference, profile_id, "remote", context.now())
    export = ExportObject(schema_version=EXPORT_SCHEMA_VERSION, profile=profile, measurements=[])
    return {"export": export.model_dump(mode="json")}


def room_result(payload: dict, context: ServerContext) -> dict:
    cards = _cards(payload.get("cards"))
    disk = free_disk(context.room)
    graphics = free_vram_mib(context.room, cards) if cards else None
    return {
        "disk": {"known": disk.known, "free_bytes": disk.free_bytes if disk.known else None, "unclear": disk.reason == _UNCLEAR},
        "graphics_free_mib": graphics,
    }


def _cards(value: object) -> list[tuple[int, str | None]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RemoteRequestError(ERROR_BAD_REQUEST)
    cards: list[tuple[int, str | None]] = []
    for entry in value:
        if not (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], int) and isinstance(entry[0], bool) is False):
            raise RemoteRequestError(ERROR_BAD_REQUEST)
        name = entry[1]
        if name is not None and not isinstance(name, str):
            raise RemoteRequestError(ERROR_BAD_REQUEST)
        cards.append((entry[0], name))
    return cards


PULL_PATH = "/api/pull"
_STATUS_STAGES = ("pulling manifest", "verifying sha256 digest", "writing manifest", "success")


def _pull_status(status: object) -> str:
    if isinstance(status, str) and (status in _STATUS_STAGES or status.startswith("pulling ")):
        return status
    return "working"


def daemon_lines(payload: dict, context: ServerContext) -> list[bytes]:
    method, path, body = payload.get("method"), payload.get("path"), payload.get("body")
    if not isinstance(method, str) or not isinstance(path, str) or not (body is None or isinstance(body, dict)):
        raise RemoteRequestError(ERROR_BAD_REQUEST)
    if (method, path) not in ALLOWED_CALLS:
        raise RemoteRequestError(ERROR_CALL_REFUSED)
    if (method, path) == ("POST", PULL_PATH) and not _safe_pull_body(body):
        raise RemoteRequestError(ERROR_CALL_REFUSED)
    if (method, path) == ("POST", PULL_PATH):
        return _pull_lines(context.daemon, body)
    return _short_lines(context.daemon, method, path, body)


def _safe_pull_body(body: dict | None) -> bool:
    name = (body or {}).get("model")
    return isinstance(name, str) and PULL_NAME_RE.fullmatch(name) is not None


def _short_lines(daemon: Daemon, method: str, path: str, body: dict | None) -> list[bytes]:
    try:
        answer = daemon(method, path, body)
    except DaemonError:
        raise RemoteRequestError(ERROR_OLLAMA_UNREACHABLE) from None
    try:
        parsed = answer.json() if answer.body else None
    except ValueError:
        parsed = None
    return [_result({"status": answer.status, "body": parsed})]


def _pull_lines(daemon: Daemon, body: dict | None) -> list[bytes]:
    lines: list[bytes] = []

    def follow(entry: dict) -> None:
        if "error" in entry:
            lines.append(_progress({"error": True}))
        else:
            lines.append(_progress({k: entry[k] for k in ("total", "completed") if k in entry} | {"status": _pull_status(entry.get("status"))}))

    try:
        answer = daemon("POST", PULL_PATH, body, progress=follow)
    except DaemonError:
        raise RemoteRequestError(ERROR_OLLAMA_UNREACHABLE) from None
    done = _pull_done(answer.body)
    lines.append(_result({"status": answer.status, "done": done}))
    return lines


def _pull_done(body: bytes) -> bool:
    try:
        payload = json.loads(body) if body else {}
    except (ValueError, RecursionError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "success"


def dispatch(op: str, payload: dict, context: ServerContext) -> list[bytes]:
    if op == "hello":
        return [_result(hello_result(context))]
    if op == "profile":
        return [_result(profile_result(payload, context))]
    if op == "room":
        return [_result(room_result(payload, context))]
    if op == "daemon":
        return daemon_lines(payload, context)
    raise RemoteRequestError(ERROR_BAD_REQUEST)


def main(op: str, stdin: bytes, context: ServerContext | None = None) -> tuple[list[bytes], int]:
    """Every line this call writes, and its exit code (always `0`: a fault is an `error` line, not
    a process exit code -- the SSH side (B1) reads exit codes only for the OS/protocol pre-checks
    127/2/255, never for a call that ran and answered)."""
    context = context if context is not None else ServerContext()
    try:
        request = read_request(stdin)
        if op not in OPS:
            raise RemoteRequestError(ERROR_BAD_REQUEST)
        return dispatch(op, request, context), 0
    except RemoteRequestError as exc:
        return [_error(exc.code)], 0
    except Exception:  # noqa: BLE001 -- nothing of this machine's own text ever reaches the wire
        return [_error(ERROR_INTERNAL)], 0


__all__ = [
    "ERROR_BAD_REQUEST",
    "ERROR_CALL_REFUSED",
    "ERROR_INTERNAL",
    "ERROR_OLLAMA_UNREACHABLE",
    "ERROR_TOO_LARGE",
    "MAX_INPUT_BYTES",
    "OPS",
    "PROTOCOL_VERSION",
    "REMOTE_VERSION",
    "RemoteRequestError",
    "ServerContext",
    "daemon_lines",
    "dispatch",
    "hello_result",
    "main",
    "profile_result",
    "read_request",
    "room_result",
]
