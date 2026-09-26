"""The `modelroom` command line: the guided mode, `fetch`, `hardware`, `render` and the rest.

`modelroom` with no subcommand is the guided mode (`modelroom/guided.py`); the subcommands never
ask a question. `main(argv)` is the real entry point (`[project.scripts]` in `pyproject.toml`); it takes
optional `transport`/`probes`/`pointer_path`/`now` keyword arguments purely for dependency
injection in tests -- the real CLI never passes them, so it always uses `UrllibTransport()`,
the real `Probes()` (this machine), the user's own pointer file and the real clock. Exit codes
follow AGENTS.md/CONTRACTS.md: `0` success (`fetch`: every area complete; `hardware`: the
profile was measured and written, whether or not `llmfit` was there to cross-check it;
`render`: the document was written, even when its rating source failed), `1` (`fetch`/`render`/
`hardware`) at least one `fetch` area incomplete, the lock is held, a `fetch` run is not newer
than the stored snapshot, `render` has no snapshot to read, or `render`'s existing document was
rendered from a newer snapshot, `2` the configuration is missing/invalid, the named machine is
not a writer (`fetch`) or not configured at all (`hardware`), or `hardware`'s bound profile
belongs to another machine (`--new-identity`), `3` a stored file's (configuration, snapshot,
hardware profile or pointer file) `schema_version` is unsupported or it cannot be read.

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
from typing import Callable

from pydantic import ValidationError

from . import __version__
from .answers import AnswerFileError, read_answers
from .binding import (
    KnownProfile,
    PointerFileError,
    default_pointer_path,
    read_pointer,
    resolve_profile_target,
    write_pointer,
)
from .config import ConfigError, Configuration, load_config
from .contracts import HardwareSnapshot, SchemaVersionError
from .daemon import Daemon
from .dialog import (
    AnswerInvalidError,
    AnswerMissingError,
    Asker,
    Canceled,
    FileAsker,
    TerminalAsker,
    is_interactive,
)
from .fetch import run_fetch
from .guided import GuidedError, run_guided
from .http import DEFAULT_REQUEST_BUDGET, RequestBudget, Transport, UrllibTransport
from .importer import (
    ImportConflictError,
    PrerequisiteError,
    StoredFileError,
    export_profile,
    import_profile,
)
from .llmfit import LlmfitReference, read_llmfit_reference
from .measure import MeasuredHardware, Probes, build_profile, measure_hardware
from .migrate import MigrationError, migrate
from .profile import HardwareProfile, fit_block_reason, os_fingerprint, read_profile_document
from .render_cmd import render_with_config
from .state import (
    LockHeldError,
    StaleRunError,
    StateFileShapeError,
    UnreadableStateFileError,
    acquire_lock,
    atomic_write_json,
    check_run_is_newer,
    read_snapshot,
    release_lock,
    write_run_status,
    write_snapshot,
)


_FRESH_ID_ATTEMPTS = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelroom",
        description=(
            "Without a subcommand: the guided mode, which asks, writes modelroom.toml and runs "
            "hardware, fetch and render. The subcommands never ask a question."
        ),
    )
    parser.add_argument(
        "--config", type=Path, help="Path to modelroom.toml (guided mode: wins over the remembered folder)"
    )
    parser.add_argument(
        "--answers", type=Path, help="Guided mode: take the dialog's answers from this TOML file instead of asking"
    )
    # Answered by argparse itself, before a subcommand, a terminal or the pointer file is looked at.
    parser.add_argument("-V", "--version", action="version", version=f"modelroom {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch package metadata for every configured base model from this machine."
    )
    fetch_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")
    fetch_parser.add_argument("--machine", required=True, help="This machine's name in [machines]")

    hardware_parser = subparsers.add_parser("hardware", help="Measure this machine and write its hardware profile.")
    hardware_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")
    hardware_parser.add_argument(
        "--machine", help="This machine's name in [machines], when it has one (its profile is adopted)"
    )
    hardware_parser.add_argument(
        "--cpu-only", action="store_true", help="Judge this machine as a CPU machine; no GPU source is read"
    )
    hardware_parser.add_argument(
        "--new-identity", action="store_true", help="Write a new profile for this machine instead of the bound one"
    )
    hardware_parser.add_argument(
        "--same-machine",
        action="store_true",
        help="Measure again under the bound profile's own id, when that profile is missing or foreign",
    )

    render_parser = subparsers.add_parser(
        "render", help="Render the current snapshot and every machine's hardware profile to Markdown."
    )
    render_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")

    migrate_parser = subparsers.add_parser(
        "migrate", help="Move hardware profiles and configuration of an earlier schema to the current one (backups are kept)."
    )
    migrate_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")

    export_parser = subparsers.add_parser(
        "export-profile", help="Write one machine's hardware profile and its measurements to a file."
    )
    export_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")
    export_parser.add_argument("--profile", help="The profile_id to export (default: this machine's own profile)")
    export_parser.add_argument("--out", required=True, type=Path, help="The file to write")

    import_parser = subparsers.add_parser(
        "import-profile", help="Read a hardware profile and its measurements from such a file."
    )
    import_parser.add_argument("file", type=Path, help="The export file to read")
    import_parser.add_argument("--config", required=True, type=Path, help="Path to modelroom.toml")

    return parser


def main(
    argv: list[str] | None = None,
    transport: Transport | None = None,
    now: datetime | None = None,
    probes: Probes | None = None,
    pointer_path: Path | None = None,
    asker: Asker | None = None,
    here: Path | None = None,
    out: Callable[[str], None] | None = None,
    daemon: "Daemon | None" = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return _cmd_guided(args, parser, transport, probes, pointer_path, now, asker, here, out, daemon)
    if args.answers is not None:
        print(f"--answers belongs to the guided mode; {args.command} never asks a question", file=sys.stderr)
        return 2
    if args.command == "fetch":
        return _cmd_fetch(args, transport, now)
    if args.command == "hardware":
        return _cmd_hardware(args, probes, pointer_path, now)
    if args.command == "render":
        return _cmd_render(args, now)
    if args.command == "migrate":
        return _cmd_migrate(args, now)
    if args.command == "export-profile":
        return _cmd_export_profile(args, pointer_path)
    if args.command == "import-profile":
        return _cmd_import_profile(args, now)
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - argparse.error already exits


def _cmd_guided(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    transport: Transport | None,
    probes: Probes | None,
    pointer_path: Path | None,
    now: datetime | None,
    asker: Asker | None,
    here: Path | None,
    out: Callable[[str], None] | None,
    daemon: "Daemon | None" = None,
) -> int:
    """`modelroom` with no subcommand: the guided mode (`modelroom/guided.py`).

    Exit codes: `0` a document was written, `1` another process holds the lock or a step reported
    it, `2` without a terminal and without `--answers` (the help goes to stderr), at an end of
    input, at a missing or unusable answer, or when a step cannot go on, `3` a stored file's
    `schema_version` is unsupported or it does not read, `130` at Ctrl-C. Every failure of a
    stored file keeps the exit code that command would give it on its own -- the guided mode does
    not flatten them into one. Ctrl-C and an end of input leave no half-written file: every write
    of this mode is atomic and happens after the last question of its step.
    """
    try:
        chosen = asker if asker is not None else _asker_for(args, parser)
        if chosen is None:
            return 2
        return run_guided(
            chosen,
            here=here if here is not None else Path.cwd(),
            config_arg=args.config,
            pointer_path=pointer_path,
            transport=transport,
            probes=probes,
            now=now,
            out=out if out is not None else print,
            daemon=daemon,
            # A run that answers from a file is read from a log: no color, whatever the
            # terminal it was started in (CONTRACTS.md, "Guided mode", the start screen).
            colored=None if args.answers is None else False,
        )
    except KeyboardInterrupt:
        print("stopped at your request; nothing was left half written", file=sys.stderr)
        return 130
    except (Canceled, AnswerMissingError, AnswerInvalidError, AnswerFileError, GuidedError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (SchemaVersionError, MigrationError, PointerFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3


def _asker_for(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Asker | None:
    """The asker this run uses, or `None` when there is no way to ask (help on stderr, exit 2)."""
    if args.answers is not None:
        return FileAsker(read_answers(args.answers))
    if is_interactive():
        return TerminalAsker()
    parser.print_help(sys.stderr)
    print(
        "\nmodelroom without a subcommand needs an interactive terminal, or "
        "`modelroom --answers <file>` with the dialog's answers.",
        file=sys.stderr,
    )
    return None


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
    config: Configuration,
    machine: str,
    transport: Transport | None = None,
    now: datetime | None = None,
    budget: int | RequestBudget = DEFAULT_REQUEST_BUDGET,
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
        read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    try:
        handle = acquire_lock(config.paths.lock_file, "fetch", run_at)
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        return _run_locked(config, active_transport, run_at, budget)
    finally:
        release_lock(handle)


def _run_locked(config, transport: Transport, run_at: datetime, budget: int | RequestBudget) -> int:
    try:
        old_snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    try:
        check_run_is_newer(old_snapshot, run_at)
    except StaleRunError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    result = run_fetch(config, transport, run_at, old_snapshot, budget)
    write_snapshot(config, result.snapshot.model_dump(mode="json"))
    write_run_status(config, result.snapshot, result.request_used, result.request_budget)

    return 0 if all(area.status == "complete" for area in result.snapshot.areas) else 1


def _cmd_hardware(
    args: argparse.Namespace, probes: Probes | None, pointer_path: Path | None, now: datetime | None
) -> int:
    try:
        config = load_config(args.config)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return hardware_with_config(
        config,
        args.machine,
        probes,
        now,
        pointer_path,
        cpu_only=args.cpu_only,
        new_identity=args.new_identity,
        same_machine=args.same_machine,
        results_dir=args.config.resolve().parent,
    )


def hardware_with_config(
    config: Configuration,
    machine: str | None = None,
    probes: Probes | None = None,
    now: datetime | None = None,
    pointer_path: Path | None = None,
    cpu_only: bool = False,
    new_identity: bool = False,
    results_dir: Path | None = None,
    # Behind `results_dir` and by keyword only, so that no caller of the eight positional
    # arguments this function had before hands its folder in as a "yes" to overwriting a profile
    # (second-model round, 2026-09-24).
    *,
    same_machine: bool = False,
    summary: bool = True,
) -> int:
    """Run `hardware` against an already-loaded `Configuration` -- the programmatic entry point.

    Measures this machine itself (`modelroom/measure.py`), cross-checks the two comparable
    readings against `llmfit` and writes one schema-2 profile,
    `<state>/hardware/<profile_id>.json` (CONTRACTS.md, "Hardware measurement"). `machine` is
    optional: it is only consulted for `[machines.<name>].profile`, the configured profile the
    takeover rule may adopt. The measurement itself runs before the lock -- it reads the machine,
    not the state -- and the lock covers reading the existing profiles, writing the new one and
    binding it.

    `new_identity` and `same_machine` are the two answers to the takeover rule's own question and
    are mutually exclusive: refused here with exit `2`, before anything is measured or written.
    `summary` prints the one line with the readings of the profile; the guided mode passes `False`
    and says in its own note what was measured (`modelroom render --config` and `modelroom
    hardware` are unchanged).
    """
    if new_identity and same_machine:
        print(
            "--new-identity and --same-machine are two answers to one question: "
            "--new-identity measures this machine as a new one, --same-machine measures it again "
            "under the profile id this results folder already binds it to. Pass one of them",
            file=sys.stderr,
        )
        return 2
    if machine is not None and machine not in config.machines:
        names = sorted(config.machines)
        print(
            f"{machine!r} is not a configured machine; "
            f"configured machines: {', '.join(names) if names else '(none configured)'}",
            file=sys.stderr,
        )
        return 2

    active = probes if probes is not None else Probes()
    recorded_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    pointer_file = pointer_path if pointer_path is not None else default_pointer_path()

    measured = measure_hardware(active.platform, active.runner, active.read_text, active.memory_bytes, cpu_only)
    reference = read_llmfit_reference(active.runner, config.llmfit.min_version)

    try:
        handle = acquire_lock(config.paths.lock_file, "hardware", recorded_at)
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        return _hardware_locked(
            config, machine, active, measured, reference, recorded_at, pointer_file, new_identity,
            _binding_key(config, results_dir), same_machine, summary,
        )
    finally:
        release_lock(handle)


def _hardware_locked(
    config: Configuration,
    machine: str | None,
    probes: Probes,
    measured: MeasuredHardware,
    reference: LlmfitReference,
    recorded_at: datetime,
    pointer_file: Path,
    new_identity: bool,
    binding_key: Path,
    same_machine: bool = False,
    summary: bool = True,
) -> int:
    """Pick this machine's profile, write it and bind it -- everything the lock has to cover.

    The pointer file is read *here*, not before the lock, and as late as the takeover rule
    allows: two runs that both read "no binding yet" before either wrote one would each create a
    profile for the same machine, which is exactly what the binding exists to prevent.
    """
    try:
        profiles = _existing_profiles(config)
    except (SchemaVersionError, UnreadableStateFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    raw_id = measured.identity.raw_id
    machine_config = config.machines.get(machine) if machine is not None else None
    known = {key: KnownProfile(key, p.os_fingerprint, p.origin, p.ram_physical_source) for key, p in profiles.items()}
    fresh_profile_id = _fresh_profile_id(probes, _taken_profile_ids(config, profiles))
    try:
        pointer = read_pointer(pointer_file)
    except (SchemaVersionError, PointerFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    target = resolve_profile_target(
        pointer.binding_for(binding_key),
        machine_config.profile if machine_config is not None else None,
        known,
        os_fingerprint(raw_id) if raw_id else "none",
        fresh_profile_id,
        new_identity,
        same_machine,
    )
    if target.action == "ask_clone":
        print(
            f"{target.reason} ({target.profile_id}); this is either the same machine or a clone -- "
            "run `modelroom hardware --same-machine` to measure it again under that same profile id, "
            "or `modelroom hardware --new-identity` to measure it as its own machine",
            file=sys.stderr,
        )
        return 2

    existing = profiles.get(target.profile_id)
    display_name = existing.display_name if existing is not None else probes.hostname()
    # Second line of defense, the same convention the llmfit path already follows: every source
    # guards its own value (a byte count that is not a real positive number never becomes a
    # reading), but a combination none of them anticipated must still end as an exit code, never
    # as an uncaught pydantic error. Validation happens before the write, so nothing is written
    # either way (probe: a naive `now`, which `recorded_at` refuses).
    try:
        profile = build_profile(measured, reference, target.profile_id, display_name, recorded_at)
    except ValidationError as exc:
        print(f"this measurement did not validate as a hardware profile: {exc}", file=sys.stderr)
        return 2
    profile_file = config.paths.hardware_dir / f"{profile.profile_id}.json"
    try:
        atomic_write_json(profile_file, profile.model_dump(mode="json"))
    except OSError as exc:
        # A state folder that cannot hold the profile -- a regular file where the `hardware`
        # folder belongs, a full disk, no permission -- is a message and an exit code, never an
        # `OSError` out of the command. Nothing is written and nothing is bound.
        print(f"{profile_file}: the profile could not be written ({exc})", file=sys.stderr)
        return 1
    bound = _bind_this_machine(pointer_file, binding_key, profile.profile_id)
    _print_hardware_summary(profile, measured, reference, summary)
    return 0 if bound else 1


def _bind_this_machine(pointer_file: Path, results_dir: Path, profile_id: str) -> bool:
    """Record this machine's profile in the pointer file; `False` when that could not be done.

    The pointer file is one file per user for *every* results folder, and each folder has its own
    lock, so this is a merge onto the newest content rather than a write-back of the copy this run
    read -- a binding another folder's run added meanwhile must not be dropped. The remaining
    window between this read and this write is not covered by any lock; it is the same trade-off
    as "atomicity, not durability" for the state files, and the cost of losing it is one extra
    profile on the next run, never a lost measurement.

    The profile itself is already written when this runs: a pointer file that cannot be read or
    written (a folder in the way, no permission, a corrupt file) must therefore not throw the
    measurement away. It is reported instead, and the caller ends with exit `1`.
    """
    try:
        write_pointer(pointer_file, read_pointer(pointer_file).with_binding(results_dir, profile_id))
        return True
    except (OSError, ValueError, SchemaVersionError, PointerFileError) as exc:
        print(
            f"{pointer_file}: the profile was written, but this machine could not be bound to it "
            f"({exc}); the next run may write a second profile for this machine",
            file=sys.stderr,
        )
        return False


def _binding_key(config: Configuration, results_dir: Path | None) -> Path:
    """The folder this machine's profile is bound to in the home pointer file.

    The results folder *is* the folder the configuration file sits in -- the same key
    `export-profile`/`import-profile` and the guided mode use, so one machine never ends up with
    two profiles in one folder. A configuration handed in as an object without a file (an
    embedding adapter) has no such folder and binds on the state folder, where the profiles live.
    """
    return results_dir if results_dir is not None else config.paths.state


def _existing_profiles(config: Configuration) -> dict[str, HardwareProfile]:
    """Every schema-2 profile in the hardware folder, by `profile_id`.

    A schema-1 file (`<machine>.json`, written before `modelroom migrate` ran) is read, version
    checked and then skipped: it is not a profile v2, it carries no `profile_id`, and this
    command neither adopts nor overwrites it.
    """
    folder = config.paths.hardware_dir
    if not folder.is_dir():
        return {}
    profiles: dict[str, HardwareProfile] = {}
    for path in sorted(folder.glob("*.json")):
        document = _read_profile_file(path)
        if isinstance(document, HardwareProfile):
            profiles[document.profile_id] = document
    return profiles


def _read_profile_file(path: Path) -> HardwareSnapshot | HardwareProfile:
    """One stored profile file, with the file named in every failure (exit `3` at the call site).

    `ValueError` covers more than `json.JSONDecodeError`: a JSON integer above Python's
    int/str conversion limit raises a plain `ValueError` out of `json.loads` (probe: a
    `schema_version` of 5000 digits), and pydantic's `ValidationError`, `StateFileShapeError` and
    `UnicodeDecodeError` are all `ValueError`s as well. They are named anyway, for the reader.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise StateFileShapeError(f"expected a JSON object at the root, got {type(data).__name__}")
        return read_profile_document(data)
    except SchemaVersionError as exc:
        raise SchemaVersionError(f"{path}: {exc}") from exc
    except (json.JSONDecodeError, ValidationError, StateFileShapeError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise UnreadableStateFileError(f"{path}: cannot read hardware profile: {exc}") from exc


def _taken_profile_ids(config: Configuration, profiles: dict[str, HardwareProfile]) -> set[str]:
    """Every name a new profile must not take: each file name in the folder and each `profile_id`.

    The file names matter on their own, because a schema-1 file (`<machine>.json`) carries no
    `profile_id` and a machine name may well look like one -- a fresh id landing on it would
    overwrite a profile this command promises never to touch.
    """
    folder = config.paths.hardware_dir
    stems = {path.stem for path in folder.glob("*.json")} if folder.is_dir() else set()
    return stems | set(profiles)


def _fresh_profile_id(probes: Probes, taken: set[str]) -> str:
    """A `profile_id` no file in this folder uses yet (16 random hex characters, `Probes`)."""
    for _ in range(_FRESH_ID_ATTEMPTS):
        candidate = probes.new_id()
        if candidate not in taken:
            return candidate
    raise ValueError(f"no unused profile_id after {_FRESH_ID_ATTEMPTS} attempts; the id source repeats itself")


def _print_hardware_summary(
    profile: HardwareProfile, measured: MeasuredHardware, reference: LlmfitReference, summary: bool = True
) -> None:
    """One summary line, then every note the user has to read -- facts only, no advice.

    `summary` is `False` for the guided mode, which says in a note of its own what was measured and
    who confirmed it; the readings themselves are in the profile file (decided 2026-09-24). The
    notes are printed either way -- they are what a reader has to act on.
    """
    checks = profile.llmfit_crosscheck
    if summary:
        print(
            f"{profile.display_name} ({profile.profile_id}): "
            f"ram {_gib(profile.ram_physical_gib)} ({profile.ram_physical_source}), "
            f"vram {_gib(profile.vram_gib)} ({profile.vram_source}), gpu {profile.gpu_state}, "
            f"llmfit ram {checks.ram_physical.status} / vram {checks.vram.status}"
        )
    for note in measured.notes:
        print(f"note: {note}")
    if reference.reason is not None:
        print(f"note: the llmfit cross-check is {reference.status}: {reference.reason}")
    blocked = fit_block_reason(profile)
    if blocked is not None:
        print(f"note: no fit is computed for this profile -- {blocked}")


def _gib(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f} GiB"


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


def _cmd_migrate(args: argparse.Namespace, now: datetime | None) -> int:
    """`migrate`: exit 0 migrated or nothing to do, 1 lock held, 2 configuration invalid, 3 a
    file has an unsupported schema version or cannot be read -- nothing written in 1, 2, 3."""
    started_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    try:
        lines = migrate(args.config, started_at)
    except (SchemaVersionError, MigrationError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


def _cmd_export_profile(args: argparse.Namespace, pointer_path: Path | None) -> int:
    """`export-profile`: exit 0 written, 2 the configuration, `--out` or the chosen profile is
    not usable (no binding, no such profile, still schema 1, the file cannot be written), 3 a
    stored file does not read."""
    try:
        lines = export_profile(args.config, args.out, args.profile, pointer_path or default_pointer_path())
    except (SchemaVersionError, StoredFileError, PointerFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (ConfigError, PrerequisiteError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        # An --out that is a folder, a read-only target, a full disk: a user-actionable failure,
        # never a traceback. `atomic_write_json` leaves no temporary file behind either way.
        print(f"{args.out}: cannot write the export file: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


def _cmd_import_profile(args: argparse.Namespace, now: datetime | None) -> int:
    """`import-profile`: exit 0 written or nothing to do, 1 the lock is held or a stored file
    contradicts the import, 2 the configuration is missing/invalid or still schema 1, 3 the
    export file does not read or its schema is not supported -- 1, 2 and 3 write nothing."""
    started_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    try:
        lines = import_profile(args.config, args.file, started_at)
    except (SchemaVersionError, StoredFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (ConfigError, PrerequisiteError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (LockHeldError, ImportConflictError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        # A state folder that is a file, a read-only folder, a full disk. Every file the run had
        # already written is complete and valid (each write is atomic), and repeating the run
        # after the cause is fixed finishes the import -- so this says what failed, not that
        # nothing was written.
        print(f"{args.config}: the import could not finish: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
