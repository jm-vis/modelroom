"""The `modelroom` command line: `fetch` and `hardware` today, `render`/`check` later.

`main(argv)` is the real entry point (`[project.scripts]` in `pyproject.toml`); it takes
optional `transport`/`runner`/`now` keyword arguments purely for dependency injection in tests
-- the real CLI never passes them, so it always uses `UrllibTransport()`, `SubprocessRunner()`
and the real clock. Exit codes follow AGENTS.md/CONTRACTS.md: `0` success (`fetch`: every area
complete; `hardware`: the profile was measured and written, whether or not the local Ollama
daemon could be reached), `1` (`fetch` only) at least one area incomplete, the lock is held, or
this run is not newer than the stored snapshot, `2` the configuration is missing/invalid, the
named machine is not a writer (`fetch`) or not configured at all (`hardware`), or a required
external tool (`llmfit`) is missing or too old, `3` a stored file's schema_version is
unsupported.

`fetch_with_config`/`hardware_with_config` are the programmatic entry points for a caller that
already has a `Configuration` object (e.g. built with `Configuration.from_dict`) and wants to
run a command without going through argv/`load_config` -- `_cmd_fetch`/`_cmd_hardware` are thin
wrappers around them.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .config import ConfigError, Configuration, load_config
from .contracts import HARDWARE_SCHEMA_VERSION, SchemaVersionError
from .fetch import run_fetch
from .http import Transport, UrllibTransport
from .llmfit import (
    LlmfitError,
    Runner,
    SubprocessRunner,
    check_llmfit_version,
    fetch_llmfit_system,
    hardware_fields_from_llmfit_system,
)
from .ollama_local import fetch_installed_models
from .state import (
    LockHeldError,
    StaleRunError,
    acquire_lock,
    check_run_is_newer,
    load_existing_hardware_snapshot,
    load_existing_snapshot,
    release_lock,
    write_hardware_snapshot,
    write_run_status,
    write_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelroom")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch package metadata for every configured base model from this machine."
    )
    fetch_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")
    fetch_parser.add_argument("--machine", required=True, help="This machine's name in [machines]")

    hardware_parser = subparsers.add_parser(
        "hardware", help="Measure this machine's hardware and local Ollama inventory."
    )
    hardware_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")
    hardware_parser.add_argument("--machine", required=True, help="This machine's name in [machines]")

    return parser


def main(
    argv: list[str] | None = None,
    transport: Transport | None = None,
    runner: Runner | None = None,
    now: datetime | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return _cmd_fetch(args, transport, now)
    if args.command == "hardware":
        return _cmd_hardware(args, runner, transport, now)
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - argparse.error already exits


def _cmd_fetch(args: argparse.Namespace, transport: Transport | None, now: datetime | None) -> int:
    try:
        config = load_config(args.config)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return fetch_with_config(config, args.machine, transport, now)


def fetch_with_config(
    config: Configuration, machine: str, transport: Transport | None = None, now: datetime | None = None
) -> int:
    """Run `fetch` against an already-loaded `Configuration` -- the programmatic entry point.

    Everything `_cmd_fetch` does once it has a `Configuration` in hand: the writer check, the
    lock, the stale-run check, the fetch itself, the snapshot/run-status writes and the exit
    code. `_cmd_fetch` is a thin CLI wrapper around this (`load_config` plus argv parsing); a
    caller that already has a `Configuration` -- e.g. `Configuration.from_dict` -- and a
    `Transport` (a `FixtureTransport` in tests, `UrllibTransport()` in production) calls this
    directly, with no argparse/CLI process involved.
    """
    machine_config = config.machines.get(machine)
    if machine_config is None or not machine_config.writer:
        writers = sorted(name for name, candidate in config.machines.items() if candidate.writer)
        print(
            f"{machine!r} is not a writer in this configuration; "
            f"writers: {', '.join(writers) if writers else '(none configured)'}",
            file=sys.stderr,
        )
        return 2

    run_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    active_transport = transport if transport is not None else UrllibTransport()

    # F13: the schema-version gate on the existing snapshot runs *before* the lock is taken --
    # an unsupported schema_version must exit 3 with nothing written, not even a lock file. The
    # snapshot is read again inside the lock (`_run_locked`), since it may change between the
    # two reads.
    try:
        load_existing_snapshot(config)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    try:
        handle = acquire_lock(config.paths.lock_file, "fetch", run_at)
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        return _run_locked(config, active_transport, run_at)
    finally:
        release_lock(handle)


def _run_locked(config, transport: Transport, run_at: datetime) -> int:
    try:
        old_snapshot = load_existing_snapshot(config)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    try:
        check_run_is_newer(old_snapshot, run_at)
    except StaleRunError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    result = run_fetch(config, transport, run_at, old_snapshot)
    write_snapshot(config, result.snapshot.model_dump(mode="json"))
    write_run_status(config, result.snapshot, result.request_used, result.request_budget)

    return 0 if all(area.status == "complete" for area in result.snapshot.areas) else 1


def _cmd_hardware(
    args: argparse.Namespace, runner: Runner | None, transport: Transport | None, now: datetime | None
) -> int:
    try:
        config = load_config(args.config)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return hardware_with_config(config, args.machine, runner, transport, now)


def hardware_with_config(
    config: Configuration,
    machine: str,
    runner: Runner | None = None,
    transport: Transport | None = None,
    now: datetime | None = None,
) -> int:
    """Run `hardware` against an already-loaded `Configuration` -- the programmatic entry point.

    Unlike `fetch_with_config`, `machine` only has to be *configured*, not a writer -- hardware
    is measured on every machine, and each machine writes only its own
    `<state>/hardware/<machine>.json` file, so there is no lock to acquire (CONTRACTS.md,
    "Hardware profile (AP4)").
    """
    if machine not in config.machines:
        names = sorted(config.machines)
        print(
            f"{machine!r} is not a configured machine; "
            f"configured machines: {', '.join(names) if names else '(none configured)'}",
            file=sys.stderr,
        )
        return 2

    measured_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    active_runner = runner if runner is not None else SubprocessRunner()
    active_transport = transport if transport is not None else UrllibTransport()

    try:
        llmfit_version = check_llmfit_version(active_runner, config.llmfit.min_version)
        system_data = fetch_llmfit_system(active_runner)
        fields = hardware_fields_from_llmfit_system(system_data)
    except LlmfitError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    installed, unavailable_reason = fetch_installed_models(active_transport, measured_at)

    try:
        existing = load_existing_hardware_snapshot(config, machine)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    measurements = existing.measurements if existing is not None else []

    data = {
        "schema_version": HARDWARE_SCHEMA_VERSION,
        "machine": machine,
        "measured_at": measured_at.isoformat(),
        "llmfit_version": llmfit_version,
        "vram_gib": fields["vram_gib"],
        "ram_gib": fields["ram_gib"],
        "free_ram_gib_at_measurement": fields["free_ram_gib_at_measurement"],
        "gpu_name": fields["gpu_name"],
        "backend": fields["backend"],
        "unified_memory": fields["unified_memory"],
        "installed": [model.model_dump(mode="json") for model in installed] if installed is not None else None,
        "installed_unavailable_reason": unavailable_reason,
        "measurements": [measurement.model_dump(mode="json") for measurement in measurements],
    }
    # R7: second line of defense. hardware_fields_from_llmfit_system already validates every
    # field it produces, but a llmfit output shape neither it nor this function has anticipated
    # must still exit 2 -- never crash with an uncaught pydantic ValidationError. Validation
    # happens before any file is touched (write_hardware_snapshot's own convention), so nothing
    # is written either way.
    try:
        snapshot = write_hardware_snapshot(config, machine, data)
    except ValidationError as exc:
        print(f"llmfit output did not validate as a hardware profile: {exc}", file=sys.stderr)
        return 2

    if snapshot.installed is not None:
        installed_summary = f"installed: {len(snapshot.installed)}"
    else:
        installed_summary = f"installed: unknown ({snapshot.installed_unavailable_reason})"
    print(
        f"{machine}: vram {snapshot.vram_gib:.2f} GiB, ram {snapshot.ram_gib:.2f} GiB, {installed_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
