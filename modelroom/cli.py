"""The `modelroom` command line: `fetch`, `hardware` and `render` today, `check` later.

`main(argv)` is the real entry point (`[project.scripts]` in `pyproject.toml`); it takes
optional `transport`/`runner`/`now` keyword arguments purely for dependency injection in tests
-- the real CLI never passes them, so it always uses `UrllibTransport()`, `SubprocessRunner()`
and the real clock. Exit codes follow AGENTS.md/CONTRACTS.md: `0` success (`fetch`: every area
complete; `hardware`: the profile was measured and written, whether or not the local Ollama
daemon could be reached; `render`: the document was written, even when its rating source
failed), `1` (`fetch`/`render`) at least one `fetch` area incomplete, the lock is held, a
`fetch` run is not newer than the stored snapshot, `render` has no snapshot to read, or
`render`'s existing document was rendered from a newer snapshot, `2` the configuration is
missing/invalid, the named machine is not a writer (`fetch`) or not configured at all
(`hardware`), or a required external tool (`llmfit`) is missing or too old, `3` a stored file's
(configuration, snapshot or hardware profile) `schema_version` is unsupported.

`fetch_with_config`/`hardware_with_config`/`render_with_config` are the programmatic entry
points for a caller that already has a `Configuration` object (e.g. built with
`Configuration.from_dict`) and wants to run a command without going through argv/`load_config`
-- `_cmd_fetch`/`_cmd_hardware`/`_cmd_render` are thin wrappers around them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .config import ConfigError, Configuration, load_config
from .contracts import HARDWARE_SCHEMA_VERSION, HardwareSnapshot, SchemaVersionError, Snapshot
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
from .render import RatingSource, build_document, parse_header_line
from .state import (
    LockHeldError,
    StaleRunError,
    acquire_lock,
    atomic_write_text,
    check_run_is_newer,
    hardware_snapshot_path,
    load_existing_hardware_snapshot,
    load_existing_snapshot,
    release_lock,
    write_hardware_snapshot,
    write_run_status,
    write_snapshot,
)


class _UnreadableStateFileError(Exception):
    """A stored snapshot/hardware file is not valid JSON, or does not match its model (P2-3).

    `state.load_existing_snapshot`/`load_existing_hardware_snapshot` let `json.JSONDecodeError`
    and pydantic `ValidationError` propagate unchanged (only `SchemaVersionError` is their own);
    `_read_snapshot`/`_read_hardware_snapshot` below catch those two and re-raise this instead,
    naming the file, so every load site in `cli.py` maps it to exit `3` exactly like
    `SchemaVersionError` -- a corrupt or wrong-shape file must never crash `main()` with an
    uncaught exception (probe: a naive `Snapshot.run_at`, before P2-3's contract fix, validated
    fine and then blew up `state.check_run_is_newer` with an uncaught `TypeError` instead; a
    truncated file or one missing required fields did the same via `JSONDecodeError`/
    `ValidationError`).
    """


def _read_snapshot(config: Configuration) -> Snapshot | None:
    """`load_existing_snapshot`, wrapping a corrupt/wrong-shape file with its path (P2-3)."""
    try:
        return load_existing_snapshot(config)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise _UnreadableStateFileError(f"{config.paths.snapshot_file}: cannot read snapshot: {exc}") from exc


def _read_hardware_snapshot(config: Configuration, machine: str) -> HardwareSnapshot | None:
    """`load_existing_hardware_snapshot`, wrapping a corrupt/wrong-shape file with its path (P2-3)."""
    try:
        return load_existing_hardware_snapshot(config, machine)
    except (json.JSONDecodeError, ValidationError) as exc:
        path = hardware_snapshot_path(config, machine)
        raise _UnreadableStateFileError(f"{path}: cannot read hardware profile: {exc}") from exc


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

    render_parser = subparsers.add_parser(
        "render", help="Render the current snapshot and every machine's hardware profile to Markdown."
    )
    render_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")

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
    if args.command == "render":
        return _cmd_render(args, now)
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
        _read_snapshot(config)
    except (SchemaVersionError, _UnreadableStateFileError) as exc:
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
        old_snapshot = _read_snapshot(config)
    except (SchemaVersionError, _UnreadableStateFileError) as exc:
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
        existing = _read_hardware_snapshot(config, machine)
    except (SchemaVersionError, _UnreadableStateFileError) as exc:
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


def _cmd_render(args: argparse.Namespace, now: datetime | None) -> int:
    try:
        config = load_config(args.config)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return render_with_config(config, now=now)


def render_with_config(
    config: Configuration, rating: RatingSource | None = None, now: datetime | None = None
) -> int:
    """Run `render` against an already-loaded `Configuration` -- see `fetch_with_config`.

    A pure reader: acquires the same lock `fetch` uses (`modelroom/state.py::acquire_lock`),
    reads the snapshot and every configured machine's hardware profile, and writes exactly one
    Markdown file (`config.paths.markdown`) via `modelroom.render.build_document`. `rating` is a
    `RatingSource` for the Package table's Stars column; `None` (the CLI's own default -- there
    is no `--rating` flag) renders every Stars cell as `–` with no failure note.
    """
    rendered_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)

    # Same ordering as fetch_with_config (F13): the schema-version gate on the existing
    # snapshot runs before the lock is taken -- an unsupported version exits 3 with nothing
    # written, not even a lock file.
    try:
        _read_snapshot(config)
    except (SchemaVersionError, _UnreadableStateFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    try:
        handle = acquire_lock(config.paths.lock_file, "render", rendered_at)
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        return _render_locked(config, rendered_at, rating)
    finally:
        release_lock(handle)


def _render_locked(config: Configuration, rendered_at: datetime, rating: RatingSource | None) -> int:
    try:
        snapshot = _read_snapshot(config)
    except (SchemaVersionError, _UnreadableStateFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if snapshot is None:
        print("nothing to render, run fetch first", file=sys.stderr)
        return 1

    try:
        hardware_by_machine = _load_hardware_profiles(config)
    except (SchemaVersionError, _UnreadableStateFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    refusal = _refusal_against_existing_document(config.paths.markdown, snapshot.run_at)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1

    document = build_document(config, snapshot, hardware_by_machine, rendered_at, rating)
    atomic_write_text(config.paths.markdown, document)
    return 0


def _load_hardware_profiles(config: Configuration) -> dict[str, HardwareSnapshot | None]:
    return {name: _read_hardware_snapshot(config, name) for name in config.machines}


def _refusal_against_existing_document(markdown_path: Path, new_snapshot_run_at: datetime) -> str | None:
    """`None` when `render` may write `markdown_path`, else the message to print and exit 1 with.

    A missing file, one whose first line is not the fixed header, one whose header timestamps
    are not aware UTC (`parse_header_line` returns `None` for all three), or one that is not
    valid UTF-8 at all (fix-round 5, F4: `read_text` would otherwise raise `UnicodeDecodeError`
    straight out of `render` for a corrupted/foreign-encoding existing file) is never a reason to
    refuse -- only a header whose own `snapshot_run_at` is strictly newer than the snapshot about
    to be rendered blocks the write.
    """
    if not markdown_path.exists():
        return None
    try:
        existing_text = markdown_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    header = parse_header_line(existing_text)
    if header is None or header.snapshot_run_at <= new_snapshot_run_at:
        return None
    return (
        f"{markdown_path}: already rendered from a newer snapshot "
        f"({header.snapshot_run_at.isoformat()} > {new_snapshot_run_at.isoformat()}); nothing written"
    )


if __name__ == "__main__":
    raise SystemExit(main())
