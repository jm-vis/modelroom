"""The `render` command: the lock, the schema gates, the reads and the two atomic writes.

`modelroom/render.py` builds the document and `modelroom/views.py` formats it, both without any
I/O; this module is the layer around them -- the same split `fetch`/`hardware` have between
`modelroom/fetch.py` and `modelroom/cli.py`. `modelroom/cli.py` re-exports
`render_with_config`, so a caller still finds every command's entry point in one place.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Configuration
from .contracts import SchemaVersionError
from .importer import ProfileScan, scan_profiles
from .measurements import (
    MeasurementRecord,
    Scenario,
    default_scenario,
    read_measurements,
)
from .render import (
    MachineProfile,
    RatingSource,
    build_render_document,
    document_json,
    machine_profile,
    parse_header_line,
)
from .state import (
    LockHeldError,
    UnreadableStateFileError,
    acquire_lock,
    atomic_write_json,
    atomic_write_text,
    read_snapshot,
    release_lock,
)
from .views import document_markdown, document_terminal


def scenario_from_config(config: Configuration) -> Scenario:
    """The context `render` computes for when no caller passes one: `[guided].context`, else 8192.

    A guided run keeps the context its ranking was computed for in the configuration, so a later
    `modelroom render --config` of that folder shows the same ranking and leaves a measurement of
    that context in measured group 0. A kept context is `entered`, 8192 included: it is what
    that run chose, and the dialog names it so -- `default` is only what a configuration no
    guided run has chosen a context in gets, through `default_scenario()`, exactly as before.
    """
    if config.guided.context is None:
        return default_scenario()
    return Scenario(
        context_requested=config.guided.context,
        context_origin="entered",
        kv_type="f16",
        kv_type_assumed=True,
        requests=1,
    )


def render_with_config(
    config: Configuration,
    rating: RatingSource | None = None,
    now: datetime | None = None,
    scenario: Scenario | None = None,
    echo: Callable[[str], None] | None = None,
) -> int:
    """Run `render` against an already-loaded `Configuration` -- the programmatic entry point.

    A pure reader: acquires the same lock `fetch` uses (`modelroom/state.py::acquire_lock`),
    reads the snapshot and every configured machine's schema-2 hardware profile with its
    measurements, and writes two files from one `document.RenderDocument` -- the Markdown view
    to `config.paths.markdown` and the JSON view next to it under the same stem.

    `rating` is a `RatingSource` for the Stars column; `None` (the CLI's own default -- there is
    no `--rating` flag) renders every Stars cell as `–` with no failure note. `scenario` is the
    one context the whole document is computed for; `None` is `scenario_from_config(config)` --
    the context a guided run kept in `[guided].context`, else 8192. `modelroom render` passes
    nothing and so shows the ranking of the last guided run of that folder; the guided mode
    itself passes the context the user just chose. `echo`, when given, receives the terminal view
    (the guided mode passes `print`); `modelroom render` passes nothing and stays silent on
    success.
    """
    rendered_at = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    json_path = config.paths.markdown.with_suffix(".json")
    if json_path == config.paths.markdown:
        print(
            f"{config.paths.markdown}: paths.markdown must not be a .json file -- the render "
            "writes its JSON view next to it under the same stem",
            file=sys.stderr,
        )
        return 2

    # Same ordering as `fetch` (F13): the schema-version gate on the existing snapshot runs
    # before the lock is taken -- an unsupported version exits 3 with nothing written, not even a
    # lock file. And R7-12: "nothing to render" is decided from this same pre-read, still before
    # the lock, because `acquire_lock` creates the state directory and the lock file as a side
    # effect of opening it. `_render_locked` re-reads the snapshot once the lock is held.
    try:
        pre_read = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if pre_read is None:
        print("nothing to render, run fetch first", file=sys.stderr)
        return 1

    try:
        handle = acquire_lock(config.paths.lock_file, "render", rendered_at)
    except LockHeldError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        return _render_locked(config, rendered_at, rating, scenario or scenario_from_config(config), json_path, echo)
    finally:
        release_lock(handle)


def _render_locked(
    config: Configuration,
    rendered_at: datetime,
    rating: RatingSource | None,
    scenario: Scenario,
    json_path: Path,
    echo: Callable[[str], None] | None,
) -> int:
    """Read, build and write both views; a broken profile file is a note, never the whole render.

    A hardware file that does not read, does not validate or is still schema 1 is listed by
    `scan_profiles` and shown as that machine's own status with its reason -- one unreadable
    file must not keep every other machine's ranking out of the document. Only the snapshot's
    own `schema_version` still ends the run with exit `3`.
    """
    try:
        snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if snapshot is None:
        print("nothing to render, run fetch first", file=sys.stderr)
        return 1

    refusal = refusal_against_existing_document(config.paths.markdown, snapshot.run_at)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1

    scan = scan_profiles(config.paths.hardware_dir)
    profiles = _machine_profiles(config, scan)
    measurements, notes = _measurements_of(config, profiles)
    document = build_render_document(config, snapshot, profiles, measurements, scenario, rendered_at, rating)
    # Each file is replaced atomically, but two files cannot be replaced as one: between the two
    # writes a reader can see the new JSON view next to the old Markdown one. The JSON view goes
    # first on purpose -- the Markdown header is the promise "this render happened", and a later
    # run compares against it, so it must only be made once the JSON view is on disk. A failure in
    # between says so, and the next render replaces both.
    try:
        atomic_write_json(json_path, document_json(document))
    except OSError as exc:
        print(f"{json_path}: the JSON view could not be written ({exc}); nothing else written", file=sys.stderr)
        return 1
    try:
        atomic_write_text(config.paths.markdown, document_markdown(document))
    except OSError as exc:
        print(
            f"{config.paths.markdown}: the Markdown view could not be written ({exc}); "
            f"{json_path} is already the new one, so run this again",
            file=sys.stderr,
        )
        return 1
    if echo is not None:
        echo(document_terminal(document))
    for note in [*(f"skipped {path.name}: {reason}" for path, reason in scan.unreadable), *notes]:
        print(f"note: {note}")
    return 0


def _machine_profiles(config: Configuration, scan: ProfileScan) -> dict[str, MachineProfile]:
    """Which profile each `[machines.<name>]` is rendered from (`render.machine_profile`)."""
    return {name: machine_profile(name, machine, scan) for name, machine in config.machines.items()}


def _measurements_of(
    config: Configuration, profiles: dict[str, MachineProfile]
) -> tuple[dict[str, list[MeasurementRecord]], list[str]]:
    """Every ranked profile's measurement records, plus one note per file that does not read."""
    records: dict[str, list[MeasurementRecord]] = {}
    notes: list[str] = []
    for found in profiles.values():
        if found.profile is None or found.profile.profile_id in records:
            continue
        result = read_measurements(config.paths.state, found.profile.profile_id)
        records[found.profile.profile_id] = result.records
        notes += [f"skipped measurement {path.name}: {reason}" for path, reason in result.unreadable]
    return records, notes


def refusal_against_existing_document(markdown_path: Path, new_snapshot_run_at: datetime) -> str | None:
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


__all__ = [
    "refusal_against_existing_document",
    "render_with_config",
    "scenario_from_config",
]
