"""The `modelroom` command line: `fetch` today, `hardware`/`render`/`check` later.

`main(argv)` is the real entry point (`[project.scripts]` in `pyproject.toml`); it takes
optional `transport`/`now` keyword arguments purely for dependency injection in tests -- the
real CLI never passes them, so it always uses `UrllibTransport()` and the real clock. Exit
codes follow AGENTS.md/CONTRACTS.md: `0` every area complete, `1` at least one area
incomplete (also: the lock is held, or this run is not newer than the stored snapshot -- both
leave the state directory untouched), `2` the configuration is missing/invalid or the named
machine is not a writer, `3` a stored file's schema_version is unsupported.

`fetch_with_config` is the programmatic entry point for a caller that already has a
`Configuration` object (e.g. built with `Configuration.from_dict`) and wants to run `fetch`
without going through argv/`load_config` -- `_cmd_fetch` is a thin wrapper around it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ConfigError, Configuration, load_config
from .contracts import SchemaVersionError
from .fetch import run_fetch
from .http import Transport, UrllibTransport
from .state import (
    LockHeldError,
    StaleRunError,
    acquire_lock,
    check_run_is_newer,
    load_existing_snapshot,
    release_lock,
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

    return parser


def main(argv: list[str] | None = None, transport: Transport | None = None, now: datetime | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return _cmd_fetch(args, transport, now)
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

    try:
        acquire_lock(config.paths.lock_file, "fetch", run_at)
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        return _run_locked(config, active_transport, run_at)
    finally:
        release_lock(config.paths.lock_file)


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


if __name__ == "__main__":
    raise SystemExit(main())
