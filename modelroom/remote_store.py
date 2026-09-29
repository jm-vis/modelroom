"""`~/.modelroom/remote.json`: this computer's neutral key, and the SSH aliases of other computers.

A file of the user account, next to the pointer file (`binding.default_pointer_path`), never inside a
results folder and never exported, rendered or copied anywhere (CONTRACTS.md, "Remote machines").
It holds what the neutral names of AP9-FR need to know about this user:

- `this_machine`: the key of this computer in `[machines.<name>]`, `computer-<8 hex>`, random, made
  once on first need and kept (FR-N). It cannot be derived from the host name, and it is the same in
  every results folder this user runs on this computer.
- `ssh`: `profile_id -> alias`, the host alias a profile was read over SSH under (FR-B1 writes it; a
  results folder may name that profile's machine by the alias, and by nothing else of the SSH side).
- `profile_share`: the shared folder `modelroom/guided_share.py` offers in step 1 (FR-S, GF question
  7 (c)), set up in its own menu step and kept exactly as it was typed (after `expanduser`, never
  `resolve`d -- CONTRACTS.md, "Remote machines"). It may carry a user's or a machine's name, because
  this file never leaves the account it lives in: no export, render, results folder or backup ever
  reads it, and `guided_share.share_folder` is its only reader.

Every write reads the file, changes exactly one field and writes it back atomically, so the fields
never overwrite one another. A missing file is an empty one; a broken one is `RemoteStoreError`, which
every command maps to exit `3`, like a broken pointer file.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contracts import PROFILE_ID_RE, SchemaVersionError, check_schema_version
from .neutral import THIS_MACHINE_RE, is_alias, random_key
from .state import LockHandle, LockHeldError, acquire_lock, atomic_write_json, release_lock

REMOTE_SCHEMA_VERSION = 1
REMOTE_SCHEMA_RANGE: tuple[int, int] = (1, 2)
REMOTE_FILE_NAME = "remote.json"
LOCK_FILE_NAME = "remote.lock"
# Fixed sentences: no path of the user account, no value of the file (FR-N, the screen).
BROKEN = "remote.json of your user account (next to guided.json) does not read as modelroom's; move it away and run again"
UNWRITTEN = "remote.json of your user account (next to guided.json) could not be written; nothing else was written"
# The form `profile_share` may take (FR-S): after `expanduser`, absolute, not empty, no NUL, at most
# this many characters, and never a Windows device or pipe prefix (a write there could open a named
# pipe instead of a file) -- checked again here because the file may be edited by hand.
SHARE_PATH_MAX = 1024
SHARE_DEVICE_PREFIXES = ("\\\\.\\", "\\\\?\\")


def is_share_path(value: str) -> bool:
    """Whether `value` is a form `profile_share` may take: `guided_share`'s own rule, checked again
    on every read so a hand-edited file is caught the same way an interactive answer would be."""
    if not value or len(value) > SHARE_PATH_MAX or "\x00" in value:
        return False
    if value.startswith(SHARE_DEVICE_PREFIXES):
        return False
    return Path(value).is_absolute()


class RemoteStoreError(Exception):
    """`remote.json` exists but does not read as itself; the caller ends with exit `3`."""


def remote_path_for(pointer_path: Path) -> Path:
    """`remote.json` next to the pointer file: the home folder's in a real run, `tmp_path`'s in a test."""
    return pointer_path.parent / REMOTE_FILE_NAME


class RemoteStore(BaseModel):
    """The file: this computer's neutral key and the aliases of profiles read over SSH."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    this_machine: str | None = None
    ssh: dict[str, str] = Field(default_factory=dict)
    profile_share: str | None = None

    @field_validator("this_machine")
    @classmethod
    def _check_this_machine(cls, value: str | None) -> str | None:
        if value is not None and not THIS_MACHINE_RE.fullmatch(value):
            raise ValueError(f"this_machine must be computer-<8 lowercase hex>: {value!r}")
        return value

    @field_validator("profile_share")
    @classmethod
    def _check_profile_share(cls, value: str | None) -> str | None:
        if value is not None and not is_share_path(value):
            raise ValueError("profile_share does not read as a folder path")
        return value

    @field_validator("ssh")
    @classmethod
    def _check_ssh(cls, value: dict[str, str]) -> dict[str, str]:
        for profile_id, alias in value.items():
            if not PROFILE_ID_RE.fullmatch(profile_id):
                raise ValueError(f"ssh keys are profile_ids of 16 lowercase hex characters: {profile_id!r}")
            if not is_alias(alias):
                raise ValueError(f"ssh values are host aliases (letters, digits, dashes): {alias!r}")
        if len(set(value.values())) != len(value):
            raise ValueError("one alias belongs to one profile_id at most")
        return value

    @model_validator(mode="after")
    def _check_schema_version(self) -> "RemoteStore":
        if self.schema_version != REMOTE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REMOTE_SCHEMA_VERSION}, got {self.schema_version}")
        return self


def read_remote(path: Path) -> RemoteStore:
    """The file at `path`; an empty store when there is none. `RemoteStoreError` when it is broken."""
    if not path.exists():
        return RemoteStore(schema_version=REMOTE_SCHEMA_VERSION)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise RemoteStoreError(BROKEN) from exc
    version = data.get("schema_version") if isinstance(data, dict) else None
    if not isinstance(version, int) or isinstance(version, bool):
        raise RemoteStoreError(BROKEN)
    try:
        check_schema_version(version, REMOTE_SCHEMA_RANGE, REMOTE_FILE_NAME)
        return RemoteStore.model_validate(data)
    except (SchemaVersionError, ValidationError) as exc:
        raise RemoteStoreError(BROKEN) from exc


# The maker of a new key; tests replace it to get a fixed one (`tests/conftest.py`). How long a first
# key waits for another process making it (seconds).
_WAIT = 10.0
_NEW_KEY: Callable[[], str] = random_key


def this_machine(path: Path) -> str:
    """This computer's key: the stored one, else a new random one, stored before it is returned.

    The first one is made under a lock of the user account (`remote.lock` next to the file, the kernel
    lock of `state.acquire_lock`), read once more under it: of two processes that start at once the
    second waits and takes the first one's key. A file that cannot be written is `RemoteStoreError`.
    """
    stored = read_remote(path)
    if stored.this_machine is not None:
        return stored.this_machine
    handle = _account_lock(path)
    try:
        stored = read_remote(path)
        if stored.this_machine is not None:
            return stored.this_machine
        key = _NEW_KEY()
        atomic_write_json(path, stored.model_copy(update={"this_machine": key}).model_dump(mode="json"))
        return key
    except OSError as exc:
        raise RemoteStoreError(UNWRITTEN) from exc
    finally:
        release_lock(handle)


def _account_lock(path: Path) -> LockHandle:
    """The lock of this user's `remote.json`; another process holding it is waited for, `_WAIT` at most."""
    deadline = time.monotonic() + _WAIT
    while True:
        try:
            return acquire_lock(path.with_name(LOCK_FILE_NAME), "remote", datetime.now(timezone.utc))
        except LockHeldError:
            if time.monotonic() > deadline:
                raise RemoteStoreError(UNWRITTEN) from None
            time.sleep(0.05)
        except OSError as exc:
            raise RemoteStoreError(UNWRITTEN) from exc


def aliases(path: Path) -> dict[str, str]:
    """`profile_id -> alias` of every profile read over SSH by this user."""
    return dict(read_remote(path).ssh)


def set_profile_share(path: Path, value: str | None) -> None:
    """Store or clear `profile_share`, the one field `guided_share` ever writes here (FR-S).

    Read, changed, written back atomically under the account lock (`_account_lock`), like every
    other write of this file -- a concurrent write of `this_machine` or `ssh` is never lost. `value`
    is stored exactly as given (already checked by `is_share_path`); `None` removes it.
    """
    handle = _account_lock(path)
    try:
        stored = read_remote(path)
        atomic_write_json(path, stored.model_copy(update={"profile_share": value}).model_dump(mode="json"))
    except OSError as exc:
        raise RemoteStoreError(UNWRITTEN) from exc
    finally:
        release_lock(handle)


__all__ = [
    "REMOTE_FILE_NAME",
    "REMOTE_SCHEMA_VERSION",
    "SHARE_DEVICE_PREFIXES",
    "SHARE_PATH_MAX",
    "RemoteStore",
    "RemoteStoreError",
    "aliases",
    "is_share_path",
    "read_remote",
    "remote_path_for",
    "set_profile_share",
    "this_machine",
]
