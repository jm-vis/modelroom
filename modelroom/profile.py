"""Hardware profile v2: one machine's identity and memory, each value with its source.

Schema 2 replaces the `HardwareSnapshot` of schema 1 as the file `hardware` writes, keyed by a
random `profile_id` instead of the machine name (`<state>/hardware/<profile_id>.json`). The
schema-1 model stays in `contracts.py` as its own validator; `normalize_profile_v1` turns one
into the other without guessing (CONTRACTS.md, "Hardware profile v2"). Kept apart from
`contracts.py` so each module stays one subject.

Schema 3 (decided 2026-09-25) keeps every field and adds the value `entered` for a machine
entered by hand: its memory, its graphics memory and its GPU state. `unified_memory` becomes a
state the fit computes for. A schema-2 file reads as schema 3 (`normalize_profile_v2`); only
`modelroom migrate` writes it back.

Schema 4 (decided 2026-09-26) adds every graphics card `nvidia-smi` lists (`gpus`), the state
`multi_gpu` for two cards or more, and the machine class with its source. A schema-3 file reads
as schema 4 (`normalize_profile_v3`), a schema-2 file through both steps.
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

PROFILE_SCHEMA_VERSION = 4
# Half-open, same convention as every other reader: schema 1 (legacy, read-only), 2 and 3 (read
# as 4) and 4.
PROFILE_SCHEMA_RANGE: tuple[int, int] = (1, 5)
# The tolerance of the card list's sum against `vram_gib`: both carry two decimals.
CARD_SUM_TOLERANCE = 0.01

FINGERPRINT_SALT = "modelroom"
CROSSCHECK_TOLERANCE = 0.05

GpuState = Literal[
    "none",
    "measured",
    "entered",
    "present_unmeasured",
    "multi_gpu",
    "multi_gpu_not_covered",
    "unified_memory",
    "unsupported_platform",
    "legacy_unknown",
]
MachineClass = Literal["laptop", "workstation", "server", "unknown"]
# The classes a user may state for a machine (`hardware --machine-class`, the guided questions).
MACHINE_CLASSES: tuple[str, ...] = ("laptop", "workstation", "server")
# The GPU states the fit computes for: no GPU at all (CPU), one measured GPU, one GPU entered by
# hand, two graphics cards or more (decided 2026-09-26), and unified memory (one pool shared by
# the system and the model; decided 2026-09-25).
FIT_GPU_STATES: frozenset[str] = frozenset({"none", "measured", "entered", "multi_gpu", "unified_memory"})
# The GPU states a machine entered by hand can have: its own graphics card, several graphics cards
# of one size, none, unified memory.
ENTERED_GPU_STATES: frozenset[str] = frozenset({"entered", "multi_gpu", "none", "unified_memory"})
# The GPU states that carry a card list; every other state has an empty one.
CARD_GPU_STATES: frozenset[str] = frozenset({"measured", "entered", "multi_gpu"})

_GPU_STATE_REASONS = {
    "present_unmeasured": "a graphics adapter is present but its memory was not measured",
    "multi_gpu_not_covered": "more than one GPU, measured before this version covered it; measure again",
    "unsupported_platform": "this platform is not measured yet",
    "legacy_unknown": "profile from schema 1 does not show the GPU layout; measure again",
}
# A schema-1 profile of unified memory carries llmfit's readings only, never measured ones.
_LEGACY_REASON = "profile from schema 1 carries llmfit readings only; measure again"
# The values schema 3 added; a file that says it is schema 2 cannot carry them.
_SCHEMA_3_VALUES = {"ram_physical_source": "entered", "vram_source": "entered", "gpu_state": "entered"}
# The fields and the value schema 4 added; a file that says it is schema 2 or 3 carries none of them.
_SCHEMA_4_FIELDS = ("gpus", "machine_class", "machine_class_source")
_SCHEMA_4_VALUES = {"gpu_state": "multi_gpu"}


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


class GpuAdapter(BaseModel):
    """One graphics card of the profile's card list, at its position in `nvidia-smi`'s list.

    `name` has no length rule, like `gpu_name`: `nvidia-smi`'s trimmed field, or `None` for a card
    entered by hand and for a profile read from schema 3 without a name. A view says "graphics
    card" for a card without one.
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    name: str | None = None
    vram_gib: float = Field(gt=0)
    vram_source: Literal["nvidia-smi", "llmfit", "entered"]


class HardwareProfile(BaseModel):
    """One machine's hardware, schema 4 (`<state>/hardware/<profile_id>.json`).

    Reserves are not here, same reason as in schema 1: they are operator policy in the
    configuration. `ram_limit_gib` is a note only and never enters the fit. A machine entered
    by hand (`ram_physical_source == "entered"`) has exactly one of four shapes: its own
    graphics card of N GiB, several graphics cards of N GiB each, no graphics card, or unified
    memory (`_check_entered`). `gpus` lists the cards of the states `measured`, `entered` and
    `multi_gpu`; the machine class changes no number of the fit.
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
    ram_physical_source: Literal["os", "llmfit", "entered", "unknown"]
    ram_limit_gib: float | None = Field(default=None, gt=0)
    ram_limit_scope: str = Field(min_length=1)
    vram_gib: float | None = Field(default=None, ge=0)
    vram_source: Literal["nvidia-smi", "llmfit", "entered", "none", "unknown"]
    gpu_state: GpuState
    gpu_name: str | None = None
    gpus: list[GpuAdapter]
    machine_class: MachineClass
    machine_class_source: Literal["chassis", "entered", "unknown"]
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
        if self.gpu_state in ("measured", "multi_gpu") and (self.vram_source in ("none", "unknown") or not self.vram_gib):
            raise ValueError(f"gpu_state {self.gpu_state!r} requires a measured vram_gib above 0")
        if self.gpu_state == "legacy_unknown" and self.os_fingerprint_source != "legacy":
            raise ValueError("gpu_state 'legacy_unknown' only comes from a schema-1 profile")
        return self

    @model_validator(mode="after")
    def _check_entered(self) -> "HardwareProfile":
        """The coupled rules of a value entered by hand (schema 3, decided 2026-09-25).

        An entered memory or graphics memory belongs to an entered profile; `vram_source`
        `entered` comes exactly with `gpu_state` `entered`, or with `multi_gpu` of an entered
        profile (schema 4) -- so a measured GPU never carries an entered size -- and needs a size
        above 0 on a machine whose memory was entered as well. A machine whose memory was entered
        is one of the four shapes (`_check_entered_shape`). `hardware --cpu-only` stays what it
        was: `origin` `entered` with a memory the OS measured.
        """
        hand_ram = self.ram_physical_source == "entered"
        if (hand_ram or self.vram_source == "entered") and self.origin != "entered":
            raise ValueError("an entered ram_physical_source or vram_source requires origin 'entered'")
        entered_cards = self.gpu_state == "entered" or (self.gpu_state == "multi_gpu" and self.origin == "entered")
        if entered_cards != (self.vram_source == "entered"):
            raise ValueError("vram_source 'entered' comes with gpu_state 'entered' or with entered 'multi_gpu' cards")
        if entered_cards and not (hand_ram and self.vram_gib):
            raise ValueError("entered graphics cards require vram_gib above 0 and an entered ram_physical_gib")
        if hand_ram:
            _check_entered_shape(self)
        return self

    @model_validator(mode="after")
    def _check_gpus(self) -> "HardwareProfile":
        """The card list matches the GPU state and `vram_gib` (schema 4, decided 2026-09-26)."""
        if bool(self.gpus) != (self.gpu_state in CARD_GPU_STATES):
            raise ValueError(f"gpus is not empty exactly for gpu_state {sorted(CARD_GPU_STATES)}")
        if [gpu.index for gpu in self.gpus] != list(range(len(self.gpus))):
            raise ValueError("gpus carries the indexes 0 ... n-1 in order")
        if any(gpu.vram_source != self.vram_source for gpu in self.gpus):
            raise ValueError("every card of gpus has the profile's vram_source")
        if self.gpu_state == "multi_gpu":
            _check_card_sum(self)
        elif self.gpus and (len(self.gpus) != 1 or (self.gpus[0].vram_gib, self.gpus[0].name) != (self.vram_gib, self.gpu_name)):
            raise ValueError(f"gpu_state {self.gpu_state!r} has one card with the profile's vram_gib and gpu_name")
        return self

    @model_validator(mode="after")
    def _check_machine_class(self) -> "HardwareProfile":
        if (self.machine_class == "unknown") != (self.machine_class_source == "unknown"):
            raise ValueError("machine_class is 'unknown' exactly when machine_class_source is 'unknown'")
        if self.machine_class_source == "chassis" and self.ram_physical_source == "entered":
            raise ValueError("a machine entered by hand has no chassis reading (source 'entered' or 'unknown')")
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


def _check_card_sum(profile: HardwareProfile) -> None:
    """Two cards or more, whose sizes add up to `vram_gib`."""
    if len(profile.gpus) < 2:
        raise ValueError("gpu_state 'multi_gpu' lists two graphics cards or more")
    difference = abs(sum(gpu.vram_gib for gpu in profile.gpus) - (profile.vram_gib or 0.0))
    # Inclusive, and isclose keeps an exact 0.01 (80.00 against 79.99) from failing on float rounding.
    if difference > CARD_SUM_TOLERANCE and not math.isclose(difference, CARD_SUM_TOLERANCE, rel_tol=1e-9):
        raise ValueError("vram_gib of gpu_state 'multi_gpu' is the sum of its cards")


def _check_entered_shape(profile: HardwareProfile) -> None:
    """A machine entered by hand: a graphics card of N GiB, n cards of N GiB each, no graphics
    card, or unified memory.

    Nothing of it was cross-checked with llmfit, and unified memory has no graphics memory of its
    own -- the fit takes both reserves from the one memory instead (`fit._choose_pool`). The cards
    of one machine entered by hand have one size: the dialog asks for one.
    """
    if profile.gpu_state not in ENTERED_GPU_STATES:
        raise ValueError(f"a machine entered by hand has gpu_state {sorted(ENTERED_GPU_STATES)}, got {profile.gpu_state!r}")
    if len({gpu.vram_gib for gpu in profile.gpus}) > 1:
        raise ValueError("the graphics cards of a machine entered by hand have one size")
    if profile.gpu_state == "unified_memory" and profile.vram_source != "none":
        raise ValueError("unified memory entered by hand has vram_source 'none' and vram_gib 0")
    checks = profile.llmfit_crosscheck
    if checks.ram_physical.status != "absent" or checks.vram.status != "absent":
        raise ValueError("a machine entered by hand has no llmfit cross-check (status 'absent')")


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

    The fit computes only for `gpu_state` `none` (CPU), `measured`, `entered`, `multi_gpu` or
    `unified_memory`, with a known physical RAM and no llmfit deviation on RAM or VRAM (then the
    user measures again or enters the value). A profile from schema 1 never computes: its
    `unified_memory` carries llmfit readings only, and the machine is measured again first.
    """
    if profile.gpu_state not in FIT_GPU_STATES:
        return f"gpu_state {profile.gpu_state}: {_GPU_STATE_REASONS[profile.gpu_state]}"
    if profile.os_fingerprint_source == "legacy":
        return f"gpu_state {profile.gpu_state}: {_LEGACY_REASON}"
    if profile.ram_physical_gib is None:
        return "physical RAM unknown"
    for name, check in (("RAM", profile.llmfit_crosscheck.ram_physical), ("VRAM", profile.llmfit_crosscheck.vram)):
        if check.status == "deviation":
            return f"{name} differs from llmfit by more than 5 %; measure again or enter the value"
    return None


def read_profile_document(data: dict) -> HardwareSnapshot | HardwareProfile:
    """Validate a stored profile file: schema 1 as `HardwareSnapshot`, 2 to 4 as `HardwareProfile`.

    A schema-2 or schema-3 file is read as schema 4 (`normalize_stored_profile`); the file itself
    is not changed. Anything else -- a missing or non-integer version, or 5 and above -- raises
    `SchemaVersionError` before any field is validated (the CLI maps it to exit 3).
    """
    if stored_profile_version(data) == 1:
        return HardwareSnapshot.model_validate(data)
    return HardwareProfile.model_validate(normalize_stored_profile(data))


def stored_profile_version(data: dict) -> int:
    """The `schema_version` of a raw profile dict; `SchemaVersionError` outside the range."""
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"hardware profile: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, PROFILE_SCHEMA_RANGE, "hardware profile")
    return version


def normalize_stored_profile(data: dict) -> dict:
    """A stored profile dict of schema 2, 3 or 4 as the schema-4 dict; `data` is left unchanged.

    The one chain both readers of a stored profile take (`read_profile_document` and
    `measurements.load_export`): schema 2 goes through both steps, schema 3 through the second.
    """
    version = stored_profile_version(data)
    if version == 2:
        return normalize_profile_v3(normalize_profile_v2(data))
    return normalize_profile_v3(data) if version == 3 else data


def normalize_profile_v2(data: dict) -> dict:
    """A schema-2 profile dict as the equivalent schema-3 dict; `data` is left unchanged.

    Lossless: schema 3 has the same fields and only adds values, so every value is kept and
    `schema_version` becomes 3 -- a fixed number, not the current one. A file that says it is
    schema 2 but carries what schema 3 or 4 added is refused (`ValueError`), never read as a file
    of a later schema.
    """
    _refuse_later_values(data, 2, _SCHEMA_3_VALUES)
    return {**data, "schema_version": 3}


def normalize_profile_v3(data: dict) -> dict:
    """A schema-3 profile dict as the equivalent schema-4 dict; `data` is left unchanged.

    The card list comes from the state: one card with the profile's own `vram_gib`, `gpu_name`
    and `vram_source` for `measured` and `entered`, none for every other state -- 0.1.0 never
    stored a second card. The machine class was never read, so it is `unknown`. A file that says
    it is schema 3 but carries a field or the value schema 4 added is refused (`ValueError`).
    """
    _refuse_later_values(data, 3, {})
    card = {"index": 0, "name": data.get("gpu_name"), "vram_gib": data.get("vram_gib"), "vram_source": data.get("vram_source")}
    one_card = data.get("gpu_state") in ("measured", "entered")
    return {**data, "schema_version": 4, "gpus": [card] if one_card else [], "machine_class": "unknown", "machine_class_source": "unknown"}


def _refuse_later_values(data: dict, version: int, values: dict[str, str]) -> None:
    """`ValueError` for a file of schema `version` that carries what a later schema added."""
    for name, value in [*values.items(), *_SCHEMA_4_VALUES.items()]:  # a list: both may name gpu_state
        if data.get(name) == value:
            raise ValueError(f"hardware profile: schema {version} has no {name} {value!r}")
    for name in _SCHEMA_4_FIELDS:
        if name in data:
            raise ValueError(f"hardware profile: schema {version} has no field {name!r}")


def normalize_profile_v1(legacy: HardwareSnapshot, profile_id: str) -> HardwareProfile:
    """The current profile for a schema-1 one; pure, the caller supplies the new `profile_id`.

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
        gpus=[],
        machine_class="unknown",
        machine_class_source="unknown",
        llmfit_crosscheck=LlmfitCrosscheck(ram_physical=no_check, vram=no_check),
        llmfit_version=legacy.llmfit_version,
    )
