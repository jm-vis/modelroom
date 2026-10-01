"""The `modelroom` command line: the guided mode, `fetch`, `hardware`, `render` and the rest.

`modelroom` with no subcommand is the guided mode (`modelroom/guided.py`); the subcommands never
ask a question, except `pull`, which asks once before it pulls unless `--yes` is given.
`main(argv)` is the real entry point (`[project.scripts]` in `pyproject.toml`); it takes
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
from .cli_pull import run_pull
from .cli_remote import OPS as REMOTE_OPS
from .cli_remote import run_remote
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
from .guided_entered import fresh_profile_id
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
from .views_adapters import adapter_lines
from .migrate import MigrationError, migrate
from .neutral import NeutralError, neutral_name, neutral_profile
from .neutral_sweep import moved_machine, sweep_for_command
from .profile import MACHINE_CLASSES, HardwareProfile, MachineClass, fit_block_reason, os_fingerprint, read_profile_document
from .remote_store import RemoteStoreError
from .render_cmd import render_with_config
from .room import RoomProbes
from .ssh import SshRunner
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelroom",
        description=(
            "Without a subcommand: the guided mode, which asks, writes modelroom.toml and runs "
            "hardware, fetch and render. The subcommands never ask a question, except pull, "
            "which asks once before it pulls unless --yes is given."
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
    hardware_parser.add_argument("--machine-class", choices=MACHINE_CLASSES, help="The machine class, not the chassis's")
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

    pull_parser = subparsers.add_parser(
        "pull",
        help="Pull one package into this machine's Ollama, checking Ollama and the disk first "
        "(A4; the command a guided run's handover line names, no results folder needed).",
    )
    pull_parser.add_argument("name", help="The Ollama pull name (hf.co/<owner>/<repo>:<quant> or <name>:<tag>)")
    pull_parser.add_argument("--weights-bytes", type=int, help="The download's size, for the disk check before it")
    pull_parser.add_argument("--yes", action="store_true", help="Pull without asking (no terminal needs this)")

    remote_parser = subparsers.add_parser(
        "remote",
        help="The counterpart of another computer's SSH way in (FR-B1): reads one JSON request from "
        "stdin, writes JSON answer lines to stdout. Not interactive; run by modelroom itself, over SSH.",
    )
    remote_parser.add_argument("op", choices=REMOTE_OPS, help="hello, profile, room, daemon, candidates, or measure")

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
    servers: dict | None = None,
    room: "RoomProbes | None" = None,
    ssh: "SshRunner | None" = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return _cmd_guided(args, parser, transport, probes, pointer_path, now, asker, here, out, daemon, servers, room, ssh)
    if args.answers is not None:
        print(f"--answers belongs to the guided mode; {args.command} never asks a question", file=sys.stderr)
        return 2
    # Every subcommand sweeps its folder of machine names first (FR-N); the live host name is read
    # only to find this computer's 0.1.0 key, never written.
    host = (probes if probes is not None else Probes()).hostname
    if args.command == "fetch":
        return _cmd_fetch(args, transport, now, pointer_path, host)
    if args.command == "hardware":
        return _cmd_hardware(args, probes, pointer_path, now)
    if args.command == "render":
        return _cmd_render(args, now, pointer_path, host)
    if args.command == "migrate":
        return _cmd_migrate(args, now, pointer_path, host)
    if args.command == "export-profile":
        return _cmd_export_profile(args, pointer_path, host)
    if args.command == "import-profile":
        return _cmd_import_profile(args, now, pointer_path, host)
    if args.command == "pull":
        # No results folder and no configuration (A4): nothing here is swept, nothing bound.
        return run_pull(args.name, args.weights_bytes, args.yes, daemon=daemon, room=room, now=now, out=out)
    if args.command == "remote":
        # The counterpart (B2): stdin/stdout only, no results folder, nothing swept, nothing bound.
        return run_remote(args.op)
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
    servers: dict | None = None,
    room: "RoomProbes | None" = None,
    ssh: "SshRunner | None" = None,
) -> int:
    """`modelroom` with no subcommand: the guided mode (`modelroom/guided.py`).

    Exit codes: `0` a document was written, `1` another process holds the lock or a step reported
    it, `2` without a terminal and without `--answers` (the help goes to stderr), at an end of
    input, at a missing or unusable answer, or when a step cannot go on, `3` a stored file's
    `schema_version` is unsupported or it does not read, `130` at Ctrl-C (a stored file keeps its own
    command's exit code; every write is atomic and follows the last question of its step).
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
            servers=servers,
            room=room,  # `None`: the real room of this machine (`room.default_room_probes`, AP9-UX2 U8)
            ssh=ssh,  # `None`: the real ssh of this machine (`ssh.default_ssh`, FR-B1)
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
    except NeutralError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (SchemaVersionError, MigrationError, PointerFileError, RemoteStoreError) as exc:
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


def _cmd_fetch(
    args: argparse.Namespace, transport: Transport | None, now: datetime | None, pointer_path: Path | None, host: Callable[[], str]
) -> int:
    code, swept = sweep_for_command(args.config, "fetch", now, pointer_path, host)
    if code is not None:
        return code
    try:
        config = load_config(args.config)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return fetch_with_config(config, moved_machine(args.machine, swept), transport, now)


def fetch_with_config(
    config: Configuration,
    machine: str,
    transport: Transport | None = None,
    now: datetime | None = None,
    budget: int | RequestBudget = DEFAULT_REQUEST_BUDGET,
    build_tensor: bool = False,
) -> int:
    """Run `fetch` against an already-loaded `Configuration` -- the programmatic entry point.

    Everything `_cmd_fetch` does once it has a `Configuration` in hand: the writer check, the
    lock, the stale-run check, the fetch itself, the snapshot/run-status writes and the exit
    code. `_cmd_fetch` is a thin CLI wrapper around this (`load_config` plus argv parsing); a
    caller that already has a `Configuration` -- e.g. `Configuration.from_dict` -- and a
    `Transport` (a `FixtureTransport` in tests, `UrllibTransport()` in production) calls this
    directly, with no argparse/CLI process involved.

    `build_tensor` (AP9-SV2-P, default `False`) is passed on to `run_fetch`
    unchanged. Neither this function nor `_cmd_fetch` (a plain `modelroom fetch --config
    --machine`, with no pointer/profile binding of its own) ever sets it to `True`: only the
    guided mode's fetch step, which knows the folder's bound profile, does.
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
        return _run_locked(config, active_transport, run_at, budget, build_tensor)
    finally:
        release_lock(handle)


def _run_locked(
    config, transport: Transport, run_at: datetime, budget: int | RequestBudget, build_tensor: bool = False
) -> int:
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

    result = run_fetch(config, transport, run_at, old_snapshot, budget, build_tensor)
    write_snapshot(config, result.snapshot.model_dump(mode="json"))
    write_run_status(config, result.snapshot, result.request_used, result.request_budget)

    return 0 if all(area.status == "complete" for area in result.snapshot.areas) else 1


def _cmd_hardware(
    args: argparse.Namespace, probes: Probes | None, pointer_path: Path | None, now: datetime | None
) -> int:
    active = probes if probes is not None else Probes()
    code, swept = sweep_for_command(args.config, "hardware", now, pointer_path, active.hostname)
    if code is not None:
        return code
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
        moved_machine(args.machine, swept),
        active,
        now,
        pointer_path,
        cpu_only=args.cpu_only,
        new_identity=args.new_identity,
        same_machine=args.same_machine,
        results_dir=args.config.resolve().parent,
        machine_class=args.machine_class,
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
    machine_class: MachineClass | None = None,
    note: Callable[[str], None] | None = None,
) -> int:
    """Run `hardware` against an already-loaded `Configuration` -- the programmatic entry point.

    `note` receives every note of the measurement without its `note: ` prefix; the guided mode
    passes its screen's own note, so the notes stand indented like their neighbors (AP9-UX G).
    Without it they are printed as `note: <text>`, as `modelroom hardware` always did.

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
    hardware` are unchanged). `machine_class` is the user's own word for the class (`hardware
    --machine-class`, or the guided mode's answer where it differs from the chassis); without it the
    chassis type decides.
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

    # llmfit first: where modelroom counts no card itself, llmfit's cards may count (AP9-GPU, mode B).
    reference = read_llmfit_reference(active.runner, config.llmfit.min_version)
    measured = measure_hardware(active.platform, active.runner, active.read_text, active.memory_bytes, cpu_only,
                                machine_class, active.list_dir, reference)

    try:
        handle = acquire_lock(config.paths.lock_file, "hardware", recorded_at)
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        return _hardware_locked(
            config, machine, active, measured, reference, recorded_at, pointer_file, new_identity,
            _binding_key(config, results_dir), same_machine, summary, note,
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
    note: Callable[[str], None] | None = None,
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
    fresh_id = fresh_profile_id(probes.new_id, config.paths.hardware_dir)  # no name or id of the folder, any letter case
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
        fresh_id,
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
    display_name = existing.display_name if existing is not None else neutral_name(target.profile_id, measured.machine_class.machine_class)
    # Second line of defense, the same convention the llmfit path already follows: every source
    # guards its own value (a byte count that is not a real positive number never becomes a
    # reading), but a combination none of them anticipated must still end as an exit code, never
    # as an uncaught pydantic error. Validation happens before the write, so nothing is written
    # either way (probe: a naive `now`, which `recorded_at` refuses).
    try:
        profile = neutral_profile(build_profile(measured, reference, target.profile_id, display_name, recorded_at))
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
    _print_hardware_summary(profile, measured, reference, summary, note)
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


def _print_hardware_summary(
    profile: HardwareProfile,
    measured: MeasuredHardware,
    reference: LlmfitReference,
    summary: bool = True,
    note: Callable[[str], None] | None = None,
) -> None:
    """One summary line, then every note the user has to read -- facts only, no advice.

    `summary` is `False` for the guided mode, which says in a note of its own what was measured and
    who confirmed it; the readings themselves are in the profile file (decided 2026-09-24). The
    notes are said either way -- they are what a reader has to act on -- through `note` where the
    caller passes one (the guided mode's screen, AP9-UX G), else printed behind `note: `.
    """
    say = note if note is not None else (lambda text: print(f"note: {text}"))
    checks = profile.llmfit_crosscheck
    cards = f", {len(profile.gpus)} cards" if len(profile.gpus) > 1 else ""
    if summary:
        print(
            f"{profile.display_name} ({profile.profile_id}): "
            f"ram {_gib(profile.ram_physical_gib)} ({profile.ram_physical_source}), "
            f"vram {_gib(profile.vram_gib)} ({profile.vram_source}{cards}), gpu {profile.gpu_state}, "
            f"class {profile.machine_class} ({profile.machine_class_source}), "
            f"llmfit ram {checks.ram_physical.status} / vram {checks.vram.status}"
        )
    # One line per graphics unit (AP9-GPU G8): printed as they are, or said like a note in the guided mode.
    for line in adapter_lines(profile.adapters, profile.gpu_state, measured.gpu.note):
        (say if note is not None else print)(line)
    for text in measured.notes:
        say(text)
    if reference.reason is not None:
        say(f"the llmfit cross-check is {reference.status}: {reference.reason}")
    blocked = fit_block_reason(profile)
    if blocked is not None:
        say(f"no fit is computed for this profile -- {blocked}")


def _gib(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f} GiB"


def _cmd_render(args: argparse.Namespace, now: datetime | None, pointer_path: Path | None, host: Callable[[], str]) -> int:
    code, _swept = sweep_for_command(args.config, "render", now, pointer_path, host)
    if code is not None:
        return code
    try:
        config = load_config(args.config)
    except SchemaVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return render_with_config(config, now=now)


def _cmd_migrate(args: argparse.Namespace, now: datetime | None, pointer_path: Path | None, host: Callable[[], str]) -> int:
    """`migrate`: exit 0 migrated or nothing to do, 1 lock held or the folder not swept, 2 configuration
    invalid, 3 a file has an unsupported schema version or cannot be read -- nothing written in 1, 2, 3."""
    started_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    try:
        lines = migrate(args.config, started_at, pointer_path=pointer_path, hostname=host())
    except (SchemaVersionError, MigrationError, RemoteStoreError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (LockHeldError, NeutralError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


def _cmd_export_profile(args: argparse.Namespace, pointer_path: Path | None, host: Callable[[], str]) -> int:
    """`export-profile`: exit 0 written, 2 the configuration, `--out` or the chosen profile is
    not usable (no binding, no such profile, still schema 1, the file cannot be written), 3 a
    stored file does not read."""
    try:
        lines = export_profile(args.config, args.out, args.profile, pointer_path or default_pointer_path(), hostname=host())
    except (SchemaVersionError, StoredFileError, PointerFileError, RemoteStoreError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (NeutralError, LockHeldError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
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


def _cmd_import_profile(args: argparse.Namespace, now: datetime | None, pointer_path: Path | None, host: Callable[[], str]) -> int:
    """`import-profile`: exit 0 written or nothing to do, 1 the lock is held or a stored file
    contradicts the import, 2 the configuration is missing/invalid or still schema 1, 3 the
    export file does not read or its schema is not supported -- 1, 2 and 3 write nothing."""
    started_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    try:
        lines = import_profile(args.config, args.file, started_at, pointer_path=pointer_path, hostname=host())
    except (SchemaVersionError, StoredFileError, RemoteStoreError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (ConfigError, PrerequisiteError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (LockHeldError, ImportConflictError, NeutralError) as exc:
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
