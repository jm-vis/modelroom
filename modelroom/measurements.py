"""Scenario, speed measurement (schema 2), export object and measurement protocol v1.

A measurement is its own file, `<state>/measurements/<profile_id>/<measurement_id>.json`,
written once under `modelroom.lock` and never changed. The load test that produces it follows
the protocol shipped as `protocol_v1.toml`; this module holds the shapes, the validity rule and
the file rules, not the load test itself (CONTRACTS.md, "MeasurementRecord", "Measurement
files and import rule").
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contracts import (
    PROFILE_ID_RE,
    Measurement,
    SchemaVersionError,
    check_aware_utc,
    check_schema_version,
    validate_hf_repo,
)
from .profile import HardwareProfile
from .state import LockHandle, atomic_write_json

MEASUREMENT_SCHEMA_VERSION = 2
EXPORT_SCHEMA_VERSION = 1
EXPORT_SCHEMA_RANGE: tuple[int, int] = (1, 2)
DEFAULT_CONTEXT_REQUESTED = 8192
PROTOCOL_FILE = Path(__file__).with_name("protocol_v1.toml")
LEGACY_REASON = "measured before protocol v1 (migrated from a schema-1 profile)"

_MEASUREMENT_ID_FORMAT = "%Y%m%dT%H%M%SZ"
_MEASUREMENT_ID_TOKEN_LEN = 8
_SHA1_HEX = 40
_SHA256_PREFIX = "sha256:"


class MeasurementExistsError(Exception):
    """A measurement file with this `measurement_id` already exists; measurements never change."""


def _check_sha256_digest(value: str, name: str) -> str:
    hex_part = value.removeprefix(_SHA256_PREFIX)
    if not value.startswith(_SHA256_PREFIX) or len(hex_part) != 64 or any(c not in "0123456789abcdef" for c in hex_part):
        raise ValueError(f"{name} must be 'sha256:' plus 64 hex characters: {value!r}")
    return value


class Scenario(BaseModel):
    """What one ranking assumes: one context for every package, KV cache type, concurrent requests.

    `context_origin` says where `context_requested` came from (`default` 8192, `entered` by the
    user, `legacy` from a schema-1 measurement). The daemon's KV cache type cannot be read over
    its API, so `f16` (Ollama's default) is always `kv_type_assumed = true` today. `requests`
    above 1 only appears in the reverse calculation, never in a measurement.
    """

    model_config = ConfigDict(extra="forbid")

    context_requested: int = Field(gt=0, le=2**31 - 1)
    context_origin: Literal["default", "entered", "legacy"]
    kv_type: Literal["f16"]
    kv_type_assumed: Literal[True]
    requests: int = Field(ge=1, le=1024)

    @model_validator(mode="after")
    def _check_default_context(self) -> "Scenario":
        if self.context_origin == "default" and self.context_requested != DEFAULT_CONTEXT_REQUESTED:
            raise ValueError(f"context_origin 'default' means context_requested {DEFAULT_CONTEXT_REQUESTED}")
        return self


def default_scenario() -> Scenario:
    """The ranking scenario when the user chose nothing: context 8192, KV 16-bit, one request."""
    return Scenario(
        context_requested=DEFAULT_CONTEXT_REQUESTED,
        context_origin="default",
        kv_type="f16",
        kv_type_assumed=True,
        requests=1,
    )


class PackageRef(BaseModel):
    """The exact package content a measurement ran: an Ollama manifest digest, or a Hugging Face
    repo, revision and weight-file digest. Same field rules as the schema-1 `Measurement`."""

    model_config = ConfigDict(extra="forbid")

    content_source: Literal["ollama", "huggingface"]
    ollama_manifest_digest: str | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None
    hf_file_digest: str | None = None

    @model_validator(mode="after")
    def _check_fields(self) -> "PackageRef":
        hf_fields = (self.hf_repo, self.hf_revision, self.hf_file_digest)
        if self.content_source == "ollama":
            if self.ollama_manifest_digest is None or any(f is not None for f in hf_fields):
                raise ValueError("content_source 'ollama' requires ollama_manifest_digest and no hf_* fields")
            _check_sha256_digest(self.ollama_manifest_digest, "ollama_manifest_digest")
            return self
        if self.ollama_manifest_digest is not None or any(f is None for f in hf_fields):
            raise ValueError(
                "content_source 'huggingface' requires hf_repo, hf_revision and hf_file_digest, "
                "and no ollama_manifest_digest"
            )
        validate_hf_repo(self.hf_repo)
        if len(self.hf_revision) != _SHA1_HEX or any(c not in "0123456789abcdef" for c in self.hf_revision):
            raise ValueError(f"hf_revision must be a 40-hex commit sha: {self.hf_revision!r}")
        _check_sha256_digest(self.hf_file_digest, "hf_file_digest")
        return self


class RunCounters(BaseModel):
    """The raw counters of one measured `/api/generate` run, as the daemon returned them (ns)."""

    model_config = ConfigDict(extra="forbid")

    done_reason: str
    eval_count: int = Field(ge=0)
    eval_duration: int = Field(ge=0)
    prompt_eval_count: int = Field(ge=0)
    prompt_eval_duration: int = Field(ge=0)
    load_duration: int = Field(ge=0)

    def tokens_per_second(self) -> float:
        return self.eval_count / (self.eval_duration / 1e9)


class LoadState(BaseModel):
    """Machine load read just before the measured runs; load during the runs is not measurable."""

    model_config = ConfigDict(extra="forbid")

    gpu_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    cpu_load: float | None = Field(default=None, ge=0)


class ProtocolOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: Literal[True]
    stream: Literal[False]
    num_predict: int = Field(gt=0)
    temperature: float = Field(ge=0)
    seed: int


class ProtocolValidity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done_reason: str = Field(min_length=1)
    require_full_eval_count: Literal[True]
    require_positive_eval_duration: Literal[True]
    require_context_headroom: Literal[True]


class Protocol(BaseModel):
    """Measurement protocol v1 as shipped in `protocol_v1.toml` (`num_ctx` is the scenario's)."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1]
    warmup_runs: int = Field(ge=0)
    measured_runs: int = Field(gt=0)
    prompt: str = Field(min_length=1)
    options: ProtocolOptions
    validity: ProtocolValidity


def load_protocol(path: Path = PROTOCOL_FILE) -> Protocol:
    """Read and validate a protocol file; the default is the one shipped with the package."""
    try:
        return Protocol.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ValueError(f"{path}: invalid measurement protocol: {exc}") from exc


@lru_cache(maxsize=1)
def shipped_protocol() -> Protocol:
    return load_protocol()


def run_invalid_reason(run: RunCounters, protocol: Protocol, num_ctx: int) -> str | None:
    """Why one measured run breaks the protocol's validity rule, or `None` when it is valid."""
    num_predict = protocol.options.num_predict
    if run.done_reason != protocol.validity.done_reason:
        return f"done_reason {run.done_reason!r}, expected {protocol.validity.done_reason!r}"
    if run.eval_count != num_predict:
        return f"eval_count {run.eval_count}, expected {num_predict}"
    if run.eval_duration <= 0:
        return "eval_duration is not above zero"
    if run.prompt_eval_count + num_predict > num_ctx:
        return f"prompt_eval_count {run.prompt_eval_count} + {num_predict} exceeds num_ctx {num_ctx}"
    return None


def measurement_invalid_reason(runs: list[RunCounters], protocol: Protocol, num_ctx: int) -> str | None:
    """The first reason the measured runs as a whole are invalid, or `None` when all are valid."""
    if len(runs) != protocol.measured_runs:
        return f"{len(runs)} measured runs, protocol requires {protocol.measured_runs}"
    for index, run in enumerate(runs, start=1):
        reason = run_invalid_reason(run, protocol, num_ctx)
        if reason is not None:
            return f"run {index}: {reason}"
    return None


class MeasurementRecord(BaseModel):
    """One speed measurement of one package on one profile (schema 2), never changed once written.

    `protocol` is `v1` for a load test after protocol v1, `none` for a measurement migrated from
    a schema-1 profile (`validity` `unchecked`, never comparable, never ranked). `comparable`
    is decided by the load test (daemon digest and `context_length` equal to the package and
    the scenario at every observation); a measurement that is not comparable says why.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    measurement_id: str
    profile_id: str
    protocol: Literal["v1", "none"]
    measured_at: datetime
    package: PackageRef
    ollama_name: str | None = None
    daemon_version: str | None = None
    scenario: Scenario
    runs: list[RunCounters] = Field(default_factory=list)
    tps_mean: float | None = Field(default=None, gt=0)
    # No lower bound here: a migrated schema-1 range only ever had low <= high (e.g. 0.0).
    # Protocol v1 requires both above zero (`_check_reasons`).
    tps_min: float | None = None
    tps_max: float | None = None
    validity: Literal["valid", "invalid", "unchecked"]
    validity_reason: str | None = None
    comparable: bool
    comparable_reason: str | None = None
    load_state: LoadState | None = None

    @field_validator("measured_at")
    @classmethod
    def _check_measured_at(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "measured_at")

    @field_validator("profile_id")
    @classmethod
    def _check_profile_id(cls, value: str) -> str:
        if not PROFILE_ID_RE.fullmatch(value):
            raise ValueError(f"profile_id must be 16 lowercase hex characters: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_identity(self) -> "MeasurementRecord":
        if self.schema_version != MEASUREMENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {MEASUREMENT_SCHEMA_VERSION}, got {self.schema_version}")
        stamp, _, token = self.measurement_id.partition("-")
        if stamp != self.measured_at.strftime(_MEASUREMENT_ID_FORMAT) or not _is_hex(token, _MEASUREMENT_ID_TOKEN_LEN):
            raise ValueError(
                f"measurement_id must be '<measured_at as YYYYMMDDTHHMMSSZ>-<8 hex>': {self.measurement_id!r}"
            )
        if self.scenario.requests != 1:
            raise ValueError("a measurement runs exactly one request (scenario.requests == 1)")
        return self

    @model_validator(mode="after")
    def _check_reasons(self) -> "MeasurementRecord":
        if (self.validity == "valid") != (self.validity_reason is None):
            raise ValueError("validity_reason is set exactly when validity is not 'valid'")
        if self.comparable != (self.comparable_reason is None):
            raise ValueError("comparable_reason is set exactly when comparable is false")
        speeds = (self.tps_mean, self.tps_min, self.tps_max)
        if any(value is not None for value in speeds):
            if any(value is None for value in speeds) or self.tps_min > self.tps_max:
                raise ValueError("tps_mean, tps_min and tps_max are all set or none, tps_min <= tps_max")
            # A schema-1 range was never checked against its mean; it is carried over unchanged.
            if self.protocol == "v1" and not (0 < self.tps_min <= self.tps_mean <= self.tps_max):
                raise ValueError("protocol v1 requires 0 < tps_min <= tps_mean <= tps_max")
        return self

    @model_validator(mode="after")
    def _check_protocol(self) -> "MeasurementRecord":
        if self.protocol == "none":
            if self.validity != "unchecked" or self.runs or self.comparable:
                raise ValueError("protocol 'none' requires validity 'unchecked', no runs and comparable false")
            return self
        if self.validity == "unchecked":
            raise ValueError("protocol 'v1' is either 'valid' or 'invalid'")
        if self.validity == "valid":
            reason = measurement_invalid_reason(self.runs, shipped_protocol(), self.scenario.context_requested)
            if reason is not None:
                raise ValueError(f"validity 'valid' contradicts the protocol rule: {reason}")
            if self.tps_mean is None:
                raise ValueError("a valid measurement carries tps_mean, tps_min and tps_max")
        if self.tps_mean is not None:
            if not self.runs:
                raise ValueError("protocol v1 tps values come from runs; a record without runs carries none")
            _check_speeds_match_runs(self)
        return self


def _check_speeds_match_runs(record: "MeasurementRecord") -> None:
    """Protocol v1: `tps_mean`/`tps_min`/`tps_max` are the mean, min and max over `runs`."""
    try:
        speeds = [run.tokens_per_second() for run in record.runs if run.eval_duration > 0]
    except OverflowError as exc:
        raise ValueError(f"a run's eval_count or eval_duration is too large to compute a speed: {exc}") from exc
    if len(speeds) != len(record.runs):
        raise ValueError("tps values need every run's eval_duration above zero")
    expected = (sum(speeds) / len(speeds), min(speeds), max(speeds))
    stored = (record.tps_mean, record.tps_min, record.tps_max)
    if any(not math.isclose(a, b, rel_tol=1e-6) for a, b in zip(stored, expected)):
        raise ValueError(f"tps_mean/tps_min/tps_max {stored} do not match the runs {expected}")


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(c in "0123456789abcdef" for c in value)


def new_measurement_id(measured_at: datetime, token: str | None = None) -> str:
    """`<measured_at as YYYYMMDDTHHMMSSZ>-<8 hex>`; `token` defaults to 8 random hex characters."""
    check_aware_utc(measured_at, "measured_at")
    token = token if token is not None else secrets.token_hex(_MEASUREMENT_ID_TOKEN_LEN // 2)
    if not _is_hex(token, _MEASUREMENT_ID_TOKEN_LEN):
        raise ValueError(f"token must be 8 lowercase hex characters: {token!r}")
    return f"{measured_at.strftime(_MEASUREMENT_ID_FORMAT)}-{token}"


def legacy_measurement_record(legacy: Measurement, profile_id: str) -> MeasurementRecord:
    """A schema-1 embedded measurement as a schema-2 record with `protocol: none`.

    The `measurement_id` token is derived from the old measurement's content (first 8 hex of
    its SHA-256), not random, so migrating the same file twice yields the same id.
    """
    content = json.dumps(legacy.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    low, high = legacy.tps_range
    return MeasurementRecord(
        schema_version=MEASUREMENT_SCHEMA_VERSION,
        measurement_id=new_measurement_id(legacy.measured_at, hashlib.sha256(content).hexdigest()[:8]),
        profile_id=profile_id,
        protocol="none",
        measured_at=legacy.measured_at,
        package=PackageRef(
            content_source=legacy.content_source,
            ollama_manifest_digest=legacy.ollama_manifest_digest,
            hf_repo=legacy.hf_repo,
            hf_revision=legacy.hf_revision,
            hf_file_digest=legacy.hf_file_digest,
        ),
        daemon_version=legacy.runtime,
        scenario=Scenario(
            context_requested=legacy.context,
            context_origin="legacy",
            kv_type="f16",
            kv_type_assumed=True,
            requests=1,
        ),
        tps_mean=legacy.tps_mean,
        tps_min=low,
        tps_max=high,
        validity="unchecked",
        validity_reason=LEGACY_REASON,
        comparable=False,
        comparable_reason=LEGACY_REASON,
    )


# --- files ------------------------------------------------------------------------------------


def measurement_path(state_dir: Path, profile_id: str, measurement_id: str) -> Path:
    return state_dir / "measurements" / profile_id / f"{measurement_id}.json"


def write_measurement(state_dir: Path, record: MeasurementRecord, lock: LockHandle) -> Path:
    """Publish `record` as its own file; the caller holds `modelroom.lock` of `state_dir`.

    Checks under the lock that no file with this id exists (`MeasurementExistsError`
    otherwise), then writes it with `atomic_write_json`. A measurement is never overwritten.
    """
    if lock.released or lock.path != state_dir / "modelroom.lock":
        raise ValueError(f"write_measurement requires the held lock of {state_dir / 'modelroom.lock'}")
    path = measurement_path(state_dir, record.profile_id, record.measurement_id)
    if path.exists():
        raise MeasurementExistsError(f"{path}: measurement already exists")
    atomic_write_json(path, record.model_dump(mode="json"))
    return path


@dataclass
class MeasurementReadResult:
    records: list[MeasurementRecord] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)


def read_measurements(state_dir: Path, profile_id: str) -> MeasurementReadResult:
    """Every measurement file of one profile, oldest id first; a broken file is listed, not loaded.

    A file is unreadable when it is not valid JSON or UTF-8, fails validation, or its name or
    folder disagree with its own `measurement_id`/`profile_id`.
    """
    result = MeasurementReadResult()
    folder = state_dir / "measurements" / profile_id
    if not folder.is_dir():
        return result
    for path in sorted(folder.glob("*.json")):
        try:
            record = MeasurementRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, RecursionError) as exc:  # ValueError: bad UTF-8/JSON, over-long integers, ValidationError; RecursionError: nesting beyond the parser
            result.unreadable.append((path, str(exc)))
            continue
        if record.measurement_id != path.stem or record.profile_id != profile_id:
            result.unreadable.append((path, "file name or folder does not match measurement_id/profile_id"))
            continue
        result.records.append(record)
    return result


# --- import rule ------------------------------------------------------------------------------

ImportOutcome = Literal["new", "idempotent", "conflict"]


def classify_measurement_import(existing: MeasurementRecord | None, incoming: MeasurementRecord) -> ImportOutcome:
    """Same id and same content: `idempotent`; same id, other content: `conflict`; else `new`."""
    if existing is None:
        return "new"
    same = existing.model_dump(mode="json") == incoming.model_dump(mode="json")
    return "idempotent" if same else "conflict"


@dataclass
class MeasurementImportPlan:
    new: list[MeasurementRecord] = field(default_factory=list)
    idempotent: list[MeasurementRecord] = field(default_factory=list)
    conflicts: list[MeasurementRecord] = field(default_factory=list)


def plan_measurement_import(
    existing: list[MeasurementRecord], incoming: list[MeasurementRecord]
) -> MeasurementImportPlan:
    """Sort `incoming` by the import rule against `existing`; conflicts are listed, never imported."""
    by_id = {record.measurement_id: record for record in existing}
    plan = MeasurementImportPlan()
    for record in incoming:
        outcome = classify_measurement_import(by_id.get(record.measurement_id), record)
        getattr(plan, "conflicts" if outcome == "conflict" else outcome).append(record)
    return plan


# --- export object ----------------------------------------------------------------------------


class ExportObject(BaseModel):
    """What one machine hands to another: its profile and all of its measurements."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    profile: HardwareProfile
    measurements: list[MeasurementRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ExportObject":
        if self.schema_version != EXPORT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EXPORT_SCHEMA_VERSION}, got {self.schema_version}")
        ids = [m.measurement_id for m in self.measurements]
        if len(ids) != len(set(ids)):
            raise ValueError("measurement_id must be unique within one export")
        if any(m.profile_id != self.profile.profile_id for m in self.measurements):
            raise ValueError("every measurement must belong to the exported profile")
        return self


def load_export(data: dict) -> ExportObject:
    """Check `schema_version` first (`SchemaVersionError`), then validate an export file."""
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"export: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, EXPORT_SCHEMA_RANGE, "export")
    return ExportObject.model_validate(data)
