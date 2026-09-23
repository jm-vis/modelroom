"""Hardware profile v2: one machine's identity and memory, each value with its source.

Schema 2 replaces the `HardwareSnapshot` of schema 1 as the file `hardware` writes, keyed by a
random `profile_id` instead of the machine name (`<state>/hardware/<profile_id>.json`). The
schema-1 model stays in `contracts.py` as its own validator; `normalize_profile_v1` turns one
into the other without guessing (CONTRACTS.md, "Hardware profile v2"). Kept apart from
`contracts.py` so each module stays one subject.
"""

from __future__ import annotations

import hashlib
import math
import secrets
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import (
    PROFILE_ID_RE,
    HardwareSnapshot,
    SchemaVersionError,
    check_aware_utc,
    check_schema_version,
)

PROFILE_SCHEMA_VERSION = 2
# Half-open, same convention as every other reader: schema 1 (legacy, read-only) and 2.
PROFILE_SCHEMA_RANGE: tuple[int, int] = (1, 3)

FINGERPRINT_SALT = "modelroom"
CROSSCHECK_TOLERANCE = 0.05

GpuState = Literal[
    "none",
    "measured",
    "present_unmeasured",
    "multi_gpu_not_covered",
    "unified_memory",
    "unsupported_platform",
    "legacy_unknown",
]
# The only GPU states the fit computes for: no GPU at all (CPU) or one measured GPU.
FIT_GPU_STATES: frozenset[str] = frozenset({"none", "measured"})

_GPU_STATE_REASONS = {
    "present_unmeasured": "a graphics adapter is present but its memory was not measured",
    "multi_gpu_not_covered": "more than one GPU is not covered by fit v1",
    "unified_memory": "unified memory is not covered by fit v1",
    "unsupported_platform": "this platform is not measured yet",
    "legacy_unknown": "profile from schema 1 does not show the GPU layout; measure again",
}


class CrossCheck(BaseModel):
    """One llmfit cross-check: own reading against llmfit's reading of the same quantity.

    `confirmed`/`deviation` carry both values; `absent` (llmfit not installed, or not asked)
    and `error` (llmfit failed) carry no llmfit value.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["confirmed", "deviation", "absent", "error"]
    own_gib: float | None = Field(default=None, ge=0)
    llmfit_gib: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_values(self) -> "CrossCheck":
        if self.status in ("confirmed", "deviation"):
            if self.own_gib is None or self.llmfit_gib is None:
                raise ValueError(f"status {self.status!r} requires own_gib and llmfit_gib")
            if _crosscheck_status(self.own_gib, self.llmfit_gib) != self.status:
                raise ValueError(f"status {self.status!r} contradicts the 5 % rule for {self.own_gib} / {self.llmfit_gib}")
        elif self.llmfit_gib is not None:
            raise ValueError(f"status {self.status!r} carries no llmfit_gib")
        return self


class LlmfitCrosscheck(BaseModel):
    """The two cross-checked fields: physical RAM against llmfit `total_ram_gb`, VRAM against
    llmfit `gpu_vram_gb`. Nothing else is compared (only like with like)."""

    model_config = ConfigDict(extra="forbid")

    ram_physical: CrossCheck
    vram: CrossCheck


class HardwareProfile(BaseModel):
    """One machine's hardware, schema 2 (`<state>/hardware/<profile_id>.json`).

    Reserves are not here, same reason as in schema 1: they are operator policy in the
    configuration. `ram_limit_gib` is a note only and never enters the fit.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    profile_id: str
    display_name: str = Field(min_length=1, max_length=128)
    os_fingerprint: str
    os_fingerprint_source: Literal[
        "windows_machineguid", "linux_machine_id", "macos_platform_uuid", "none", "legacy"
    ]
    origin: Literal["measured", "entered"]
    recorded_at: datetime
    ram_physical_gib: float | None = Field(default=None, gt=0)
    ram_physical_source: Literal["os", "llmfit", "unknown"]
    ram_limit_gib: float | None = Field(default=None, gt=0)
    ram_limit_scope: str = Field(min_length=1)
    vram_gib: float | None = Field(default=None, ge=0)
    vram_source: Literal["nvidia-smi", "llmfit", "none", "unknown"]
    gpu_state: GpuState
    gpu_name: str | None = None
    llmfit_crosscheck: LlmfitCrosscheck
    llmfit_version: str | None = None

    @field_validator("profile_id")
    @classmethod
    def _check_profile_id(cls, value: str) -> str:
        if not PROFILE_ID_RE.fullmatch(value):
            raise ValueError(f"profile_id must be 16 lowercase hex characters: {value!r}")
        return value

    @field_validator("recorded_at")
    @classmethod
    def _check_recorded_at(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "recorded_at")

    @model_validator(mode="after")
    def _check_schema_version(self) -> "HardwareProfile":
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PROFILE_SCHEMA_VERSION}, got {self.schema_version}")
        return self

    @model_validator(mode="after")
    def _check_fingerprint(self) -> "HardwareProfile":
        without = self.os_fingerprint_source in ("none", "legacy")
        if without and self.os_fingerprint != "none":
            raise ValueError(f"os_fingerprint_source {self.os_fingerprint_source!r} requires os_fingerprint 'none'")
        if not without and not PROFILE_ID_RE.fullmatch(self.os_fingerprint):
            raise ValueError(f"os_fingerprint must be 16 lowercase hex characters: {self.os_fingerprint!r}")
        if self.origin == "entered" and self.os_fingerprint_source != "none":
            raise ValueError("an entered profile has no os_fingerprint (source 'none')")
        return self

    @model_validator(mode="after")
    def _check_memory_sources(self) -> "HardwareProfile":
        if (self.ram_physical_source == "unknown") != (self.ram_physical_gib is None):
            raise ValueError("ram_physical_gib is None exactly when ram_physical_source is 'unknown'")
        if (self.ram_limit_scope == "none") != (self.ram_limit_gib is None):
            raise ValueError("ram_limit_gib is None exactly when ram_limit_scope is 'none'")
        if (self.vram_source == "unknown") != (self.vram_gib is None):
            raise ValueError("vram_gib is None exactly when vram_source is 'unknown'")
        if self.vram_source == "none" and self.vram_gib != 0:
            raise ValueError("vram_source 'none' requires vram_gib 0")
        return self

    @model_validator(mode="after")
    def _check_gpu_state(self) -> "HardwareProfile":
        if self.gpu_state == "none" and self.vram_source != "none":
            raise ValueError("gpu_state 'none' requires vram_source 'none'")
        if self.gpu_state == "measured" and (self.vram_source in ("none", "unknown") or not self.vram_gib):
            raise ValueError("gpu_state 'measured' requires a measured vram_gib above 0")
        if self.gpu_state == "legacy_unknown" and self.os_fingerprint_source != "legacy":
            raise ValueError("gpu_state 'legacy_unknown' only comes from a schema-1 profile")
        return self

    @model_validator(mode="after")
    def _check_crosscheck_matches_own_values(self) -> "HardwareProfile":
        pairs = (
            (self.llmfit_crosscheck.ram_physical, self.ram_physical_gib, "ram_physical"),
            (self.llmfit_crosscheck.vram, self.vram_gib, "vram"),
        )
        for check, own, name in pairs:
            if check.own_gib is not None and check.own_gib != own:
                raise ValueError(f"llmfit_crosscheck.{name}.own_gib must equal the profile's own value")
        return self


def new_profile_id() -> str:
    """A fresh random `profile_id`: 16 lowercase hex characters."""
    return secrets.token_hex(8)


def os_fingerprint(raw_os_id: str) -> str:
    """SHA-256 of the raw OS identifier followed by the salt `modelroom`, first 16 hex characters.

    The raw identifier (Windows `MachineGuid`, Linux `/etc/machine-id`, macOS platform UUID)
    never leaves the machine; only this digest is stored.
    """
    if not raw_os_id.strip():
        raise ValueError("raw_os_id must not be empty")
    return hashlib.sha256(f"{raw_os_id.strip()}{FINGERPRINT_SALT}".encode("utf-8")).hexdigest()[:16]


def crosscheck(own_gib: float, llmfit_gib: float | None) -> CrossCheck:
    """Compare one own reading with llmfit's: |a-b| / max(a, b) within 5 % is `confirmed`.

    Both zero is `confirmed` (no GPU on either side). `llmfit_gib is None` means llmfit gave no
    reading: `absent`.
    """
    if llmfit_gib is None:
        return CrossCheck(status="absent", own_gib=own_gib)
    return CrossCheck(status=_crosscheck_status(own_gib, llmfit_gib), own_gib=own_gib, llmfit_gib=llmfit_gib)


def _crosscheck_status(own_gib: float, llmfit_gib: float) -> Literal["confirmed", "deviation"]:
    larger = max(own_gib, llmfit_gib)
    deviation = 0.0 if larger == 0 else abs(own_gib - llmfit_gib) / larger
    # The bound is inclusive; isclose keeps an exact 5 % (8.0 vs 7.6) from failing on float rounding.
    within = deviation <= CROSSCHECK_TOLERANCE or math.isclose(deviation, CROSSCHECK_TOLERANCE, rel_tol=1e-9)
    return "confirmed" if within else "deviation"


def fit_block_reason(profile: HardwareProfile) -> str | None:
    """Why the fit must not compute for `profile`, or `None` when it may.

    The fit computes only for `gpu_state` `none` (CPU) or `measured`, with a known physical RAM
    and no llmfit deviation on RAM or VRAM (then the user measures again or enters the value).
    """
    if profile.gpu_state not in FIT_GPU_STATES:
        return f"gpu_state {profile.gpu_state}: {_GPU_STATE_REASONS[profile.gpu_state]}"
    if profile.ram_physical_gib is None:
        return "physical RAM unknown"
    for name, check in (("RAM", profile.llmfit_crosscheck.ram_physical), ("VRAM", profile.llmfit_crosscheck.vram)):
        if check.status == "deviation":
            return f"{name} differs from llmfit by more than 5 %; measure again or enter the value"
    return None


def read_profile_document(data: dict) -> HardwareSnapshot | HardwareProfile:
    """Validate a stored profile file: schema 1 as `HardwareSnapshot`, schema 2 as `HardwareProfile`.

    Anything else -- a missing or non-integer version, or 3 and above -- raises
    `SchemaVersionError` before any field is validated (the CLI maps it to exit 3).
    """
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"hardware profile: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, PROFILE_SCHEMA_RANGE, "hardware profile")
    if version == 1:
        return HardwareSnapshot.model_validate(data)
    return HardwareProfile.model_validate(data)


def normalize_profile_v1(legacy: HardwareSnapshot, profile_id: str) -> HardwareProfile:
    """The schema-2 profile for a schema-1 one; pure, the caller supplies the new `profile_id`.

    `display_name` is the old machine key; no OS fingerprint (source `legacy`); physical RAM and
    VRAM keep their llmfit readings and say so. GPU: `unified_memory = true` becomes
    `unified_memory`, everything else `legacy_unknown` -- a positive VRAM does not prove a single
    GPU (the schema-1 adapter dropped llmfit's per-GPU list) and a zero does not prove there is
    none. No fit until the machine is measured again. Measurements embedded in the old file are
    converted separately (`measurements.legacy_measurement_record`).
    """
    no_check = CrossCheck(status="absent")
    return HardwareProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        display_name=legacy.machine,
        os_fingerprint="none",
        os_fingerprint_source="legacy",
        origin="measured",
        recorded_at=legacy.measured_at,
        ram_physical_gib=legacy.ram_gib,
        ram_physical_source="llmfit",
        ram_limit_gib=None,
        ram_limit_scope="none",
        vram_gib=legacy.vram_gib,
        vram_source="llmfit",
        gpu_state="unified_memory" if legacy.unified_memory else "legacy_unknown",
        gpu_name=legacy.gpu_name,
        llmfit_crosscheck=LlmfitCrosscheck(ram_physical=no_check, vram=no_check),
        llmfit_version=legacy.llmfit_version,
    )
