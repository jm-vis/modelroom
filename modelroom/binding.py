"""Which profile is "this machine": the pointer file in the home folder and the takeover rule.

The pointer file (`~/.modelroom/guided.json`) remembers the results folder the guided mode
last used and, per results folder, the `profile_id` of this machine's own profile there (the
local binding). `resolve_profile_target` decides, without any I/O, which profile a measurement
of this machine writes to (CONTRACTS.md, "Profile binding").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contracts import PROFILE_ID_RE, SchemaVersionError, check_schema_version
from .state import atomic_write_json

POINTER_SCHEMA_VERSION = 1
POINTER_SCHEMA_RANGE: tuple[int, int] = (1, 2)


class PointerFileError(Exception):
    """The pointer file exists but is not valid JSON or does not match `GuidedPointer`."""


def default_pointer_path() -> Path:
    """`~/.modelroom/guided.json` of the current user."""
    return Path.home() / ".modelroom" / "guided.json"


class GuidedPointer(BaseModel):
    """The pointer file: last results folder and, per results folder, this machine's profile.

    Keys of `bindings` and `current` are absolute folder paths as the guided mode resolved them.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    current: str | None = None
    bindings: dict[str, str] = Field(default_factory=dict)

    @field_validator("bindings")
    @classmethod
    def _check_bindings(cls, value: dict[str, str]) -> dict[str, str]:
        for folder, profile_id in value.items():
            if not Path(folder).is_absolute():
                raise ValueError(f"bindings keys are absolute results folders: {folder!r}")
            if not PROFILE_ID_RE.fullmatch(profile_id):
                raise ValueError(f"bindings values are profile_ids of 16 lowercase hex characters: {profile_id!r}")
        return value

    @field_validator("current")
    @classmethod
    def _check_current(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).is_absolute():
            raise ValueError(f"current must be an absolute results folder: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_schema_version(self) -> "GuidedPointer":
        if self.schema_version != POINTER_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {POINTER_SCHEMA_VERSION}, got {self.schema_version}")
        return self

    def binding_for(self, results_dir: Path) -> str | None:
        return self.bindings.get(str(results_dir))

    def with_binding(self, results_dir: Path, profile_id: str) -> "GuidedPointer":
        """A copy that binds `profile_id` to `results_dir` and makes it the current folder."""
        return GuidedPointer.model_validate(
            {
                "schema_version": POINTER_SCHEMA_VERSION,
                "current": str(results_dir),
                "bindings": {**self.bindings, str(results_dir): profile_id},
            }
        )


def read_pointer(path: Path) -> GuidedPointer:
    """The pointer file at `path`; an empty pointer when it does not exist yet.

    A version outside the accepted range raises `SchemaVersionError`; broken content raises
    `PointerFileError` naming the file.
    """
    if not path.exists():
        return GuidedPointer(schema_version=POINTER_SCHEMA_VERSION)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # ValueError: bad UTF-8/JSON and over-long integers
        raise PointerFileError(f"{path}: cannot read pointer file: {exc}") from exc
    if not isinstance(data, dict):
        raise PointerFileError(f"{path}: expected a JSON object at the root")
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"{path}: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, POINTER_SCHEMA_RANGE, str(path))
    try:
        return GuidedPointer.model_validate(data)
    except ValidationError as exc:
        raise PointerFileError(f"{path}: invalid pointer file: {exc}") from exc


def write_pointer(path: Path, pointer: GuidedPointer) -> None:
    """Replace the pointer file atomically (it belongs to one user; no shared lock needed)."""
    atomic_write_json(path, pointer.model_dump(mode="json"))


@dataclass(frozen=True)
class KnownProfile:
    """What the takeover rule needs to know about a profile file in the results folder."""

    profile_id: str
    os_fingerprint: str


@dataclass(frozen=True)
class ProfileTarget:
    """The decision: which profile a measurement of this machine goes to, and why.

    `bound` -- the home binding holds; `adopt_config` -- `[machines.<name>].profile` becomes the
    binding (no new profile); `new` -- a new profile, bound and written to the configuration;
    `ask_clone` -- the bound or configured profile is missing or its fingerprint differs: the
    guided mode asks "same machine or a clone?", automation stops and points to
    `hardware --new-identity`.
    """

    action: Literal["bound", "adopt_config", "new", "ask_clone"]
    profile_id: str
    reason: str


def _fingerprints_agree(known: KnownProfile, local_fingerprint: str) -> bool:
    # A profile without a fingerprint (migrated, or unreadable on its machine) cannot disagree.
    return "none" in (known.os_fingerprint, local_fingerprint) or known.os_fingerprint == local_fingerprint


def resolve_profile_target(
    home_binding: str | None,
    config_profile: str | None,
    profiles: dict[str, KnownProfile],
    local_fingerprint: str,
    fresh_profile_id: str,
    new_identity: bool = False,
) -> ProfileTarget:
    """Pick the profile "this machine" writes to, in the order of the takeover rule.

    (0) `new_identity` (`hardware --new-identity`, or "a clone" in the guided mode): a new
    profile, replacing the binding. (1) The home binding for this results folder. (2) Else the
    selected configuration machine's `profile`, adopted as the binding. (3) Else a new profile.
    A binding or configured profile that is missing from `profiles` (the profile files found in
    the results folder), or whose `os_fingerprint` differs from `local_fingerprint`, is
    `ask_clone`. `local_fingerprint` is `none` when this machine's OS identifier is unreadable.
    """
    if not PROFILE_ID_RE.fullmatch(fresh_profile_id) or fresh_profile_id in profiles:
        raise ValueError(f"fresh_profile_id must be a new profile_id: {fresh_profile_id!r}")
    if new_identity:
        return ProfileTarget("new", fresh_profile_id, "new identity requested")
    for candidate, action, source in (
        (home_binding, "bound", "home binding"),
        (config_profile, "adopt_config", "configuration"),
    ):
        if candidate is None:
            continue
        known = profiles.get(candidate)
        if known is None:
            return ProfileTarget("ask_clone", candidate, f"profile from {source} is missing in the results folder")
        if not _fingerprints_agree(known, local_fingerprint):
            return ProfileTarget("ask_clone", candidate, f"profile from {source} has another os_fingerprint")
        return ProfileTarget(action, candidate, f"profile from {source}")
    return ProfileTarget("new", fresh_profile_id, "no binding and no configured profile")
