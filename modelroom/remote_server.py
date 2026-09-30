"""The gegenstelle `modelroom remote <op>` (AP9-FR, FR-B1, B2): what runs on the other computer.

Not interactive: reads at most `MAX_INPUT_BYTES` from stdin (never more, even off a stream that
keeps sending -- `read_request` is handed an already-bounded read, `cli_remote.py`'s own job),
writes JSON lines to stdout **as each one is ready, not collected** (Codex r1 finding 1: a pull's
`progress` lines reach the pipe while Ollama is still sending them, through the same callback
`daemon.py`'s own streaming reader already calls per line), and writes nothing to the disk of the
machine it runs on -- except what Ollama itself writes during a pull. Every line is one envelope
object, `{"protocol": 1, <one of "progress", "result", "error">}` (CONTRACTS.md, "The remote
protocol v1"); free text of this machine -- its own error text, a foreign package name, its host
name -- never stands in one, because nothing here reads `socket.gethostname()` or forwards a caught
exception's own message to the wire.

Every request is validated against its own strict Pydantic model first (`remote_protocol.py`,
Codex r1 finding 4) -- a missing, extra or mis-typed field is `bad_request` before anything about
it is read a second time, by hand.

`hello`/`profile`/`room`/`daemon` are 0.1.0's four verbs plus the fixed HTTP allowlist `daemon.py`
already has; `candidates`/`measure` (FR-B2) run stage 1's digest rule and the load test itself at
the server, never with this machine's own installed name -- `candidates_result`/`measure_result`
below. `main()` is the one entry point the CLI (`cli_remote.py`) and the in-process fake ssh of the
test suite both call.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import remote_protocol
from .contracts import Package
from .daemon import ALLOWED_CALLS, Daemon, DaemonError, LocalDaemon
from .llmfit import read_llmfit_reference
from .loadtest import installed_candidates, installed_models, run_load_test, weight_digest
from .measure import MeasuredHardware, Probes, build_profile, measure_hardware, read_os_identity
from .measurements import EXPORT_SCHEMA_VERSION, ExportObject, Scenario, shipped_protocol
from .neutral import neutral_record, neutral_version
from .profile import os_fingerprint
from .pull_choice import PULL_NAME_RE
from .quantization import package_identity_key
from . import room as room_module
from .room import RoomProbes, free_disk, free_vram_mib
from .room import UNCLEAR_REASON as _UNCLEAR

MAX_INPUT_BYTES = 1 * 1024 * 1024
PROTOCOL_VERSION = 1
OPS = ("hello", "profile", "room", "daemon", "candidates", "measure")
# The 0.1.0 default (`config.LlmfitConfig.min_version`): the server has no configuration to read one
# from, so the profile op uses the same floor a fresh configuration would.
_DEFAULT_LLMFIT_MIN_VERSION = "1.1.16"
REMOTE_VERSION = "0.2.0"

ERROR_BAD_REQUEST = "bad_request"
ERROR_CALL_REFUSED = "call_refused"
ERROR_OLLAMA_UNREACHABLE = "ollama_unreachable"
ERROR_TOO_LARGE = "too_large"
ERROR_INTERNAL = "internal"

Write = Callable[[bytes], None]


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
    """The one JSON object of the call, or `RemoteRequestError` (`too_large`/`bad_request`).
    `stdin` is expected already bounded to at most `MAX_INPUT_BYTES + 1` by the caller
    (`cli_remote.run_remote` reads no more than that) -- checked again here regardless, so this
    function is safe to call with an unbounded buffer too (every test that builds one by hand)."""
    if len(stdin) > MAX_INPUT_BYTES:
        raise RemoteRequestError(ERROR_TOO_LARGE)
    try:
        payload = json.loads(stdin.decode("utf-8"))
    except (ValueError, RecursionError, UnicodeDecodeError):
        raise RemoteRequestError(ERROR_BAD_REQUEST) from None
    if not isinstance(payload, dict):
        raise RemoteRequestError(ERROR_BAD_REQUEST)
    return payload


def _validated_request(op: str, payload: dict) -> dict:
    try:
        return remote_protocol.validate_request(op, payload)
    except ValueError:
        raise RemoteRequestError(ERROR_BAD_REQUEST) from None


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
    profile_id = payload["profile_id"]
    machine_class = payload.get("machine_class")
    reference = read_llmfit_reference(context.probes.runner, _DEFAULT_LLMFIT_MIN_VERSION)
    measured: MeasuredHardware = measure_hardware(
        context.probes.platform, context.probes.runner, context.probes.read_text, context.probes.memory_bytes,
        False, machine_class, context.probes.list_dir, reference,
    )
    profile = build_profile(measured, reference, profile_id, "remote", context.now())
    export = ExportObject(schema_version=EXPORT_SCHEMA_VERSION, profile=profile, measurements=[])
    return {"export": export.model_dump(mode="json")}


def room_result(payload: dict, context: ServerContext) -> dict:
    cards = [tuple(card) for card in payload.get("cards", [])]
    disk = free_disk(context.room)
    graphics = free_vram_mib(context.room, cards) if cards else None
    return {
        "disk": {"known": disk.known, "free_bytes": disk.free_bytes if disk.known else None, "unclear": disk.reason == _UNCLEAR},
        "graphics_free_mib": graphics,
    }


PULL_PATH = "/api/pull"
_STATUS_STAGES = ("pulling manifest", "verifying sha256 digest", "writing manifest", "success")
_DIGEST_STAGE_RE = re.compile(r"^pulling [0-9a-f]{64}$")


def _pull_status(status: object) -> str:
    """One of Ollama's own fixed stages, or the digest stage with its exact 64-hex digest -- never
    any other text after `pulling `, which could otherwise carry a server path or name through
    (Codex r1 finding 4: the old check only looked at the prefix, not the whole of what followed it)."""
    if isinstance(status, str) and (status in _STATUS_STAGES or _DIGEST_STAGE_RE.fullmatch(status)):
        return status
    return "working"


def daemon_lines(payload: dict, context: ServerContext, write: Write) -> None:
    method, path, body = payload["method"], payload["path"], payload.get("body")
    if (method, path) not in ALLOWED_CALLS:
        raise RemoteRequestError(ERROR_CALL_REFUSED)
    if (method, path) == ("POST", PULL_PATH):
        if not _safe_pull_body(body):
            raise RemoteRequestError(ERROR_CALL_REFUSED)
        _pull_lines(context.daemon, body, write)
        return
    _short_lines(context.daemon, method, path, body, write)


def _safe_pull_body(body: dict | None) -> bool:
    name = (body or {}).get("model")
    return isinstance(name, str) and PULL_NAME_RE.fullmatch(name) is not None


def _short_lines(daemon: Daemon, method: str, path: str, body: dict | None, write: Write) -> None:
    try:
        answer = daemon(method, path, body)
    except DaemonError:
        raise RemoteRequestError(ERROR_OLLAMA_UNREACHABLE) from None
    try:
        parsed = answer.json() if answer.body else None
    except ValueError:
        parsed = None
    write(_result({"status": answer.status, "body": parsed}))


def _pull_lines(daemon: Daemon, body: dict | None, write: Write) -> None:
    """Every progress line reaches `write` (the real stdout, flushed) the moment `daemon.py`'s own
    streaming reader hands it to this callback -- nothing here waits for the pull to end before the
    first byte leaves this process (Codex r1 finding 1)."""

    def follow(entry: dict) -> None:
        if "error" in entry:
            write(_progress({"error": True}))
        else:
            counts = {k: entry[k] for k in ("total", "completed") if k in entry}
            write(_progress({**counts, "status": _pull_status(entry.get("status"))}))

    try:
        answer = daemon("POST", PULL_PATH, body, progress=follow)
    except DaemonError:
        raise RemoteRequestError(ERROR_OLLAMA_UNREACHABLE) from None
    write(_result({"status": answer.status, "done": _pull_done(answer.body)}))


def _pull_done(body: bytes) -> bool:
    try:
        payload = json.loads(body) if body else {}
    except (ValueError, RecursionError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "success"


def _local_inventory(packages: list[dict], context: ServerContext):
    """This server's own installed candidates for `packages` (B2): the same digest rule
    `loadtest.installed_candidates` applies against this machine, never a name the daemon's
    owner gave a package. `packages` are payload dicts (already validated), turned back into
    `Package` objects here -- the one place either op needs them as such."""
    parsed = [Package.model_validate(entry) for entry in packages]
    installed, _reason = installed_models(context.daemon, context.now())
    if installed is None:
        raise RemoteRequestError(ERROR_OLLAMA_UNREACHABLE)
    inventory = installed_candidates(parsed, installed, lambda name: weight_digest(context.daemon, name))
    return inventory, parsed


def candidates_result(payload: dict, context: ServerContext) -> dict:
    """Stage 1's own rule, run at the server (B2): identities only, cloud and unmatched entries as
    one count -- never an installed name of this machine."""
    inventory, _packages = _local_inventory(payload["packages"], context)
    identities = [{"package_identity": list(package_identity_key(candidate.package))} for candidate in inventory.candidates]
    set_aside = len(inventory.unmatched) + len(inventory.cloud)
    return {"candidates": identities, "set_aside": set_aside}


def _measure_progress(write: Write, warmup_runs: int, measured_runs: int):
    """`{"stage", "number", "of"}` (B2), counted from the order `run_load_test` already calls its
    `progress` callback in -- the first `warmup_runs` calls are the warm-up, the rest the measured
    runs -- never the text `run_load_test` builds for the local screen, which names the package."""
    seen = {"n": 0}

    def progress(_line: str) -> None:
        seen["n"] += 1
        number = seen["n"]
        if number <= warmup_runs:
            write(_progress({"stage": "warmup", "number": number, "of": warmup_runs}))
        else:
            write(_progress({"stage": "run", "number": number - warmup_runs, "of": measured_runs}))

    return progress


def measure_result(payload: dict, context: ServerContext, write: Write) -> None:
    """B2: the server finds its own candidate by identity (the digest rule again, freshly), runs
    the protocol against its own daemon with its own load, and sends the record through the sink
    (FR-N) with the package it just matched -- so `ollama_name` is the built name, never this
    machine's own. A candidate the fresh lookup no longer finds is `internal` (Codex: no code of
    the five names this specific case, and it is this server's own state that changed)."""
    identity = tuple(payload["package_identity"])
    inventory, packages = _local_inventory(payload["packages"], context)
    candidate = next((c for c in inventory.candidates if package_identity_key(c.package) == identity), None)
    if candidate is None:
        raise RemoteRequestError(ERROR_INTERNAL)
    protocol = shipped_protocol()
    scenario = Scenario.model_validate(payload["scenario"])
    record = run_load_test(
        context.daemon, candidate, scenario, payload["profile_id"],
        probes=context.probes, measured_at=context.now(),
        progress=_measure_progress(write, protocol.warmup_runs, protocol.measured_runs),
    )
    clean = neutral_record(record, candidate.package)
    write(_result({"record": clean.model_dump(mode="json")}))


def dispatch(op: str, payload: dict, context: ServerContext, write: Write) -> None:
    if op == "hello":
        write(_result(hello_result(context)))
    elif op == "profile":
        write(_result(profile_result(payload, context)))
    elif op == "room":
        write(_result(room_result(payload, context)))
    elif op == "daemon":
        daemon_lines(payload, context, write)
    elif op == "candidates":
        write(_result(candidates_result(payload, context)))
    elif op == "measure":
        measure_result(payload, context, write)
    else:
        raise RemoteRequestError(ERROR_BAD_REQUEST)


def main(op: str, stdin: bytes, write: Write, context: ServerContext | None = None) -> int:
    """Every line this call writes goes to `write`, one at a time, as it is ready. The exit code is
    always `0`: a fault is an `error` line, not a process exit code -- the SSH side (B1) reads exit
    codes only for the OS/protocol pre-checks 127/2/255, never for a call that ran and answered."""
    context = context if context is not None else ServerContext()
    try:
        request = _validated_request(op, read_request(stdin)) if op in OPS else None
        if request is None:
            raise RemoteRequestError(ERROR_BAD_REQUEST)
        dispatch(op, request, context, write)
        return 0
    except RemoteRequestError as exc:
        write(_error(exc.code))
        return 0
    except Exception:  # noqa: BLE001 -- nothing of this machine's own text ever reaches the wire
        write(_error(ERROR_INTERNAL))
        return 0


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
    "Write",
    "candidates_result",
    "daemon_lines",
    "dispatch",
    "hello_result",
    "main",
    "measure_result",
    "profile_result",
    "read_request",
    "room_result",
]
