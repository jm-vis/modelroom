"""Persisted run state for `fetch`: the lock file, the snapshot merge rule, atomic writes.

Every "now" the functions here need is passed in explicitly (never read from the wall clock
inside this module) so tests control time exactly, per AGENTS.md's "no mocking" rule.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from pydantic import ValidationError

from .config import Configuration
from .contracts import (
    Area,
    BaseModelSpec,
    HardwareSnapshot,
    Package,
    Snapshot,
    SNAPSHOT_SCHEMA_VERSION,
    load_hardware_snapshot,
    load_snapshot,
)
from .fetch_types import AreaOutcome
from .quantization import package_identity_key

# Fix-round 3: the kernel lock covers exactly one byte, byte 0, which is never part of the JSON;
# the content starts at `_CONTENT_OFFSET`, one byte later. Windows locking is mandatory, not
# advisory -- a read that overlaps a locked byte raises `OSError`/`PermissionError` instead of
# returning data, from any handle but the one holding the lock -- so if the JSON started at
# offset 0, nobody but the holder could ever read it while the lock was held, including this
# module's own `_read_lock` trying to name the holder in `LockHeldError`. Reserving one
# otherwise-unused byte for the lock and starting the content one byte later keeps both
# properties: exclusivity is the kernel's, and the content stays readable by any other process
# the whole time the lock is held. (`fcntl.flock` locks the whole file and is advisory, so the
# offset only matters on Windows; keeping one layout on both keeps the file format one thing.)
_LOCK_OFFSET = 0
_CONTENT_OFFSET = 1


class LockHeldError(Exception):
    """Another live process holds the lock (age no longer matters, see `acquire_lock`)."""


class StaleRunError(Exception):
    """The run being started is not strictly newer than the stored snapshot's `run_at`."""


# --- lock -------------------------------------------------------------------------------


@dataclass
class LockHandle:
    """An open, locked file descriptor. Keep it until `release_lock` -- the kernel lock lives
    exactly as long as this fd stays open (`release_lock` closes it and sets `released`).

    `released` is the handle's own state, not the OS's: once the fd is closed its number is
    reused by the next `os.open` in this process, so a second release must never touch the
    descriptor again (fix-round 4, Codex P1)."""

    path: Path
    fd: int
    released: bool = False


def acquire_lock(path: Path, command: str, now: datetime) -> LockHandle:
    """Open `path` (creating it if needed) and take an exclusive, non-blocking kernel lock on it.

    Fix-round 3: the lock is a kernel lock on a stable file that is never renamed or deleted --
    `os.open(path, O_RDWR | O_CREAT)` (never `O_EXCL`: the file persists across runs and across
    crashes), then `msvcrt.locking` (Windows) or `fcntl.flock` (elsewhere) with the non-blocking
    flag. This replaces the previous rename-based takeover design entirely: that design's
    "claim the stale file, then create fresh" could not be made exclusive against a third
    process, because `os.replace` is not bound to the file generation a process actually read --
    process A could claim a stale lock, create a fresh one and return, and process B could then
    rename *A's* fresh lock away and create its own, with neither os.replace call ever failing.
    Giving a foreign lock back on release had the same hole: it could overwrite a third process's
    fresh lock instead of the one it took from. A kernel lock has no such window -- the OS, not
    this module's own bookkeeping, decides who holds it, and it needs no "how old is too old"
    rule at all: a crashed holder's lock is released by the kernel the moment its process exits,
    and a live holder keeps the lock no matter how long it has held it.

    The lock is taken at `_LOCK_OFFSET` on the file exactly as `os.open` leaves it: nothing is
    written or truncated before the lock call, because until the lock is held this process is
    not the only one touching the file. On failure (`OSError`, including `BlockingIOError`) the
    fd is closed and `LockHeldError` is raised, naming the current holder's pid/command/started_at
    when a read of the content (`_CONTENT_OFFSET` onward, never overlapping the locked byte)
    succeeds and parses -- it may still be empty or unparseable while the holder is mid-write, in
    which case the message says only that another process holds it. On success -- now the sole
    owner of the lock byte -- the file is truncated to exactly `_CONTENT_OFFSET` bytes (which
    also grows an empty file to that length) and rewritten from `_CONTENT_OFFSET` with this run's
    own `{"pid", "command", "started_at"}` (no token any more -- there is nothing left to
    arbitrate). A `LockHandle` is returned for `release_lock`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        _lock_exclusive(fd)
    except OSError:
        os.close(fd)
        raise LockHeldError(_describe_holder(path)) from None

    try:
        _write_holder(fd, command, now)
    except BaseException:
        # fix-round 4 (Codex P2): a failure after winning the lock must not leave the fd open
        # with the lock held -- nobody would ever get a handle to release it. Closing the fd
        # is enough: the kernel drops the lock with it, and a single close leaves no second
        # step that could fail and skip the first (round 5).
        os.close(fd)
        raise
    return LockHandle(path=path, fd=fd)


def _write_holder(fd: int, command: str, now: datetime) -> None:
    content = json.dumps({"pid": os.getpid(), "command": command, "started_at": now.isoformat()})
    os.ftruncate(fd, _CONTENT_OFFSET)
    os.lseek(fd, _CONTENT_OFFSET, 0)
    os.write(fd, content.encode("utf-8"))


def _lock_exclusive(fd: int) -> None:
    """Take a non-blocking exclusive lock on `fd`'s `_LOCK_OFFSET` byte, raising `OSError` if
    another fd already holds it."""
    if sys.platform == "win32":
        os.lseek(fd, _LOCK_OFFSET, 0)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        finally:
            os.lseek(fd, 0, 0)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_exclusive(fd: int) -> None:
    if sys.platform == "win32":
        os.lseek(fd, _LOCK_OFFSET, 0)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.lseek(fd, 0, 0)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _describe_holder(path: Path) -> str:
    holder = _read_lock(path)
    if holder is None:
        return f"{path}: held by another process, holder not yet written"
    return (
        f"{path}: locked by pid {holder['pid']} running '{holder['command']}' since "
        f"{holder['started_at'].isoformat()}"
    )


def _read_lock(path: Path) -> dict | None:
    """`path`'s holder info, or `None` if it cannot be read or parsed right now.

    Reads from `_CONTENT_OFFSET` onward through a fresh handle, deliberately never touching the
    locked byte at `_LOCK_OFFSET` -- that is what lets this succeed while another process holds
    the lock. Still tolerant: the holder may be reading this file mid-write (empty or truncated
    content), so any failure here -- a short read, bad JSON, or the file not existing yet --
    simply means the holder is unknown, never a reason for `acquire_lock`'s own failure path to
    raise.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.lseek(fd, _CONTENT_OFFSET, 0)
            raw = os.read(fd, 65536)
        finally:
            os.close(fd)
        data = json.loads(raw.decode("utf-8"))
        return {
            "pid": data["pid"],
            "command": data["command"],
            "started_at": datetime.fromisoformat(data["started_at"]),
        }
    except Exception:
        return None


def release_lock(handle: LockHandle) -> None:
    """Empty `handle`'s lock file, release the kernel lock, and close the fd.

    The file is never deleted -- it stays on disk, empty, ready for the next `acquire_lock`
    (which grows it back to `_CONTENT_OFFSET` bytes itself, once it has the lock). Idempotent
    through the handle's own `released` flag: a second release returns before touching the
    descriptor at all, because its number may already belong to another file opened since
    (fix-round 4, Codex P1). The fd is closed in `finally`, so a failing truncate or unlock still
    closes it (closing releases the kernel lock too) and the failure propagates (Codex P2).
    """
    if handle.released:
        return
    handle.released = True
    try:
        os.ftruncate(handle.fd, 0)
        _unlock_exclusive(handle.fd)
    finally:
        os.close(handle.fd)


# --- atomic writes ------------------------------------------------------------------------


def atomic_write_json(path: Path, data: dict) -> None:
    """Write `data` as JSON to `path` atomically: a `.tmp` file, then `os.replace`.

    A failure at any point (serialization, the write, the replace) leaves no `.tmp` file
    behind and re-raises.

    P3-9 (fix-round 5, decided): atomic, not durable -- no `os.fsync` on the temp file or its
    directory, so a power loss in the narrow window after `os.replace` returns but before the OS
    flushes could still lose the write. Not fixed on purpose: every file this writes is a
    reconstructible cache of the registries/local daemon, recovered by re-running the command,
    not by restoring a backup (CONTRACTS.md, "Atomic writes: atomicity, not durability").
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: a `.tmp` file, then `os.replace`.

    Same convention as `atomic_write_json` (used by `render` for `config.paths.markdown`,
    which is not JSON): a failure at any point leaves no `.tmp` file behind and re-raises.
    Both writers emit LF line endings on every platform (`newline="\n"`), so a file that
    lands in a Git working tree on Windows is not rewritten with CRLF.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def publish_new_text(path: Path, text: str) -> None:
    """Create `path` with `text` only if it does not exist yet; `FileExistsError` if it does.

    The one exception to the `os.replace` convention above (decided 2026-09-25): a file two runs may
    create at the same moment -- a new configuration -- must not have the second run's text replace
    the first's. So the text is written, whole and with LF line endings, into a temporary file of
    its own -- its name carries the process id and a random part, and `O_EXCL` refuses a name that
    is taken, so two calls never share one -- and then published by **one** system call that never
    replaces an existing file: `os.rename` on Windows, `os.link` (and removing the temporary name)
    elsewhere. The mode is the one `write_text` gives (`0o666` before the umask), so a configuration
    other users of the folder read stays readable. Whatever ends the call early -- an error, Ctrl-C
    -- the temporary file goes with it.

    A file system without hard links, and a network drive that does not keep `link`/`rename`
    atomic, are outside this guarantee.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        # The descriptor stays this function's: an early end before the file object takes it over
        # would leave it open, and Windows removes no file that is open (acceptance round).
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n", closefd=False) as handle:
                handle.write(text)
        finally:
            os.close(descriptor)
        if sys.platform == "win32":
            os.rename(tmp_path, path)
        else:
            os.link(tmp_path, path)
            os.unlink(tmp_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class StateFileShapeError(ValueError):
    """R7-3 (fix-round 6): a stored state file's JSON root is not an object.

    `load_snapshot`/`load_hardware_snapshot` (`contracts.py`) both start with
    `data.get("schema_version")`, which assumes `data` is a `dict`. A root of `[]`, `null` or a
    bare string parses without complaint (`json.loads` accepts any JSON value at the top level)
    and then raises an uncaught `AttributeError` instead of the ordinary "corrupt file" error
    every other shape problem produces. Raised by `load_existing_snapshot`/
    `load_existing_hardware_snapshot` before either loader is called; `cli.py`'s
    `_read_snapshot`/`_read_hardware_snapshot` catch it (alongside `UnicodeDecodeError`, for a
    file that is not valid UTF-8 at all) and map it to exit code 3, the same as
    `json.JSONDecodeError`/`ValidationError`.
    """


def load_existing_snapshot(config: Configuration) -> Snapshot | None:
    """The current snapshot, or `None` if `fetch` has never run against this state directory.

    `SchemaVersionError` propagates unchanged, exactly like `load_config` -- the CLI maps it
    to exit code 3. `StateFileShapeError` propagates for a JSON root that is not an object.
    """
    path = config.paths.snapshot_file
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise StateFileShapeError(f"{path}: expected a JSON object at the root, got {type(data).__name__}")
    return load_snapshot(data)


class UnreadableStateFileError(Exception):
    """A stored snapshot is not valid JSON, or does not match its model (P2-3).

    `load_existing_snapshot` lets `json.JSONDecodeError`, pydantic `ValidationError`,
    `StateFileShapeError` and `UnicodeDecodeError` propagate unchanged (only
    `SchemaVersionError` is its own); `read_snapshot` below catches those four and raises this
    instead, naming the file, so every load site maps it to exit `3` exactly like
    `SchemaVersionError` -- a corrupt or wrong-shape file must never crash a command with an
    uncaught exception (probe: a naive `Snapshot.run_at`, before P2-3's contract fix, validated
    fine and then blew up `check_run_is_newer` with an uncaught `TypeError`; a truncated file or
    one missing required fields did the same via `JSONDecodeError`/`ValidationError`; R7-3: a
    JSON root of `[]`/`null`/a bare string did the same via an uncaught `AttributeError` inside
    `contracts.load_snapshot`, and a file that is not valid UTF-8 via `UnicodeDecodeError`).
    """


def read_snapshot(config: Configuration) -> Snapshot | None:
    """`load_existing_snapshot`, wrapping a corrupt or wrong-shape file with its path (P2-3).

    The one snapshot reader every command uses (`fetch` in `modelroom/cli.py`, `render` in
    `modelroom/render_cmd.py`), so a broken file reads the same way and names the same path
    wherever it is met.
    """
    try:
        return load_existing_snapshot(config)
    except (json.JSONDecodeError, ValidationError, StateFileShapeError, UnicodeDecodeError, OSError) as exc:
        # `OSError` as well (found by the second-model review of AP9-C, and true of this reader
        # since AP3): `load_existing_snapshot` checks `Path.exists()` and then reads, so a folder
        # named `modelroom.json`, a file without read permission or a race between the two raised
        # an `IsADirectoryError`/`PermissionError`/`FileNotFoundError` straight out of `fetch` and
        # `render` -- a traceback where the contract promises a message and exit `3`.
        raise UnreadableStateFileError(f"{config.paths.snapshot_file}: cannot read snapshot: {exc}") from exc


def write_snapshot(config: Configuration, data: dict) -> Snapshot:
    """Validate `data` as a `Snapshot`, then atomically replace the snapshot file with it.

    Validation happens before any file is touched, so a validation failure never leaves a
    `.tmp` file behind (CONTRACTS.md: "Producers validate before they write").
    """
    snapshot = load_snapshot(data)
    atomic_write_json(config.paths.snapshot_file, snapshot.model_dump(mode="json"))
    return snapshot


def write_run_status(config: Configuration, snapshot: Snapshot, request_used: int, request_budget: int) -> None:
    """Write `run-status.json`: per-area status/error/last_success, run_at, and the budget.

    `candidates` is always an empty list in this work package -- market-ranking suggestions
    are a later feature (CONTRACTS.md). Written on every run, even one that ends exit 1.
    """
    data = {
        "run_at": snapshot.run_at.isoformat(),
        "request_budget": {"used": request_used, "total": request_budget},
        "areas": [
            {
                "source": area.source,
                "base_model_hf_repo": area.base_model_hf_repo,
                "packager": area.packager,
                "status": area.status,
                "error": area.error,
                "last_success": area.last_success.isoformat() if area.last_success else None,
            }
            for area in snapshot.areas
        ],
        "candidates": [],
    }
    atomic_write_json(config.paths.run_status_file, data)


SEARCH_LOG_NAME = "search.json"


def search_log_path(config: Configuration) -> Path:
    """Where one search's own log goes: next to the snapshot, under `paths.state`."""
    return config.paths.state / SEARCH_LOG_NAME


def write_search_log(config: Configuration, data: dict) -> None:
    """Write `search.json`: which account answered what, and what could not be picked, and why.

    The file a reader goes to when an account answered nothing, or when a page was full: those
    seven lines and the request count left the screen of the guided mode in the test round of
    2026-09-24, because they say nothing about the answer a reader is looking at. Written once per
    search, atomically, and replaced by the next one -- it is about the search that just ran
    (CONTRACTS.md, "Search log").
    """
    atomic_write_json(search_log_path(config), data)


# --- hardware profile (AP4) ----------------------------------------------------------------
#
# No lock guards these functions: unlike the snapshot (one shared file every writer machine's
# `fetch` merges into), each machine's hardware profile is a file only that machine ever
# writes (`<state>/hardware/<machine>.json`), so there is nothing to serialize against another
# process (CONTRACTS.md, "Hardware profile (AP4)").


def hardware_snapshot_path(config: Configuration, machine: str) -> Path:
    return config.paths.hardware_dir / f"{machine}.json"


def load_existing_hardware_snapshot(config: Configuration, machine: str) -> HardwareSnapshot | None:
    """The current hardware profile for `machine`, or `None` if `hardware` never ran there.

    `SchemaVersionError` propagates unchanged, exactly like `load_existing_snapshot` -- the CLI
    maps it to exit code 3. `StateFileShapeError` propagates for a JSON root that is not an
    object.
    """
    path = hardware_snapshot_path(config, machine)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise StateFileShapeError(f"{path}: expected a JSON object at the root, got {type(data).__name__}")
    return load_hardware_snapshot(data)


def write_hardware_snapshot(config: Configuration, machine: str, data: dict) -> HardwareSnapshot:
    """Validate `data` as a `HardwareSnapshot`, then atomically replace this machine's file.

    Validation happens before any file is touched, same convention as `write_snapshot`.
    """
    snapshot = load_hardware_snapshot(data)
    atomic_write_json(hardware_snapshot_path(config, machine), snapshot.model_dump(mode="json"))
    return snapshot


# --- version rule -------------------------------------------------------------------------


def check_run_is_newer(old: Snapshot | None, run_at: datetime) -> None:
    """Raise `StaleRunError` unless `run_at` is strictly newer than `old.run_at`.

    `None` (no prior snapshot) always passes. Equal instants are rejected, not just older
    ones -- a second `fetch` started within the same second as another is a stale run too.
    """
    if old is not None and run_at <= old.run_at:
        raise StaleRunError(
            f"run_at {run_at.isoformat()} is not newer than the stored snapshot's run_at "
            f"{old.run_at.isoformat()}; nothing written"
        )


# --- merge ----------------------------------------------------------------------------------


def _area_key_of_package(package: Package) -> tuple[str, str, str | None]:
    if package.source == "huggingface":
        owner = package.repo.split("/", 1)[0] if package.repo else None
        return (package.source, package.base_model_hf_repo, owner)
    return (package.source, package.base_model_hf_repo, None)


def merge_snapshot(
    old: Snapshot | None,
    run_at: datetime,
    area_outcomes: list[AreaOutcome],
    base_models: list[BaseModelSpec],
) -> Snapshot:
    """Build this run's `Snapshot` from `area_outcomes`, merged against `old`.

    Per area: `complete` replaces its packages (a package that disappeared is kept with
    `active=False`, one still found is `active=True`); `incomplete` keeps its old packages
    untouched and records the error, carrying over the old `last_success`. Base models are
    always replaced by this run's results. Never mutates `old`.
    """
    old_packages_by_key = {package_identity_key(p): p for p in (old.packages if old else [])}
    old_areas_by_key = {(a.source, a.base_model_hf_repo, a.packager): a for a in (old.areas if old else [])}

    # F2: an old package is only ever carried into this run's result when its area is among
    # this run's area_outcomes -- a base model or area no longer configured is dropped here,
    # not kept forever just because merge_snapshot never explicitly deletes anything.
    configured_area_keys = {(o.source, o.base_model_hf_repo, o.packager) for o in area_outcomes}
    result_packages = {
        key: package
        for key, package in old_packages_by_key.items()
        if _area_key_of_package(package) in configured_area_keys
    }
    new_areas: list[Area] = []

    for outcome in area_outcomes:
        area_key = (outcome.source, outcome.base_model_hf_repo, outcome.packager)
        old_area = old_areas_by_key.get(area_key)

        if outcome.status == "complete":
            old_keys_in_area = {k for k, p in old_packages_by_key.items() if _area_key_of_package(p) == area_key}
            new_keys: set[tuple[str, str, str]] = set()
            for package in outcome.packages:
                key = package_identity_key(package)
                new_keys.add(key)
                previous = old_packages_by_key.get(key)
                observed_at = previous.observed_at if previous else run_at
                result_packages[key] = package.model_copy(
                    update={"observed_at": observed_at, "last_seen": run_at, "active": True}
                )
            for stale_key in old_keys_in_area - new_keys:
                previous = old_packages_by_key[stale_key]
                current = result_packages.get(stale_key)
                if current is not None and _area_key_of_package(current) != area_key:
                    # Another area published this identity this run (the repository moved to a
                    # different base model); its fresh record wins regardless of area order.
                    continue
                if previous.active:
                    result_packages[stale_key] = previous.model_copy(update={"active": False})
            new_areas.append(
                Area(
                    source=outcome.source,
                    base_model_hf_repo=outcome.base_model_hf_repo,
                    packager=outcome.packager,
                    status="complete",
                    last_success=run_at,
                    error=None,
                )
            )
        else:
            new_areas.append(
                Area(
                    source=outcome.source,
                    base_model_hf_repo=outcome.base_model_hf_repo,
                    packager=outcome.packager,
                    status="incomplete",
                    last_success=old_area.last_success if old_area else None,
                    error=outcome.error,
                )
            )

    return Snapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        run_at=run_at,
        areas=new_areas,
        base_models=base_models,
        packages=list(result_packages.values()),
    )
