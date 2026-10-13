"""Shared data shapes for modelroom: family/base-model identity, packages, snapshots.

Pydantic v2 models with `extra="forbid"`, explicit value ranges and validators for coupled
fields. This is the only definition; `CONTRACTS.md` describes each model for readers and
repeats `EXAMPLES` verbatim. See `CONTRACTS.md` for the rules these models follow.

Snapshot schema 2 (AP9-SV2-0, decided 2026-09-29, docs/adr/0014-runtime-entries-and-vllm-
packages.md, E8) is the first real version step this form takes: earlier additive changes
(`aliases`, the provenance rename of ADR 0007) stayed inside schema 1. A tensor package may now
be `from_metadata`/`approved` when it came from Hugging Face (`Package.quantization_source`,
the provenance invariant below), and `quantization.QUANT_ORDER` gained the vLLM labels -- a
0.1.0 reader meeting either would fail on a field it does not know, an opaque validation error
rather than the named schema-version refusal (exit `3`) this step gives it instead. Schema 1 is
still accepted and reads as 2 unchanged (`load_snapshot`, the one entry point that normalizes;
`Snapshot.model_validate` alone still enforces the current version only).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts_fit import Fit, Rating  # noqa: F401 -- re-exported, see "render (AP5)" below
from .quantization import QUANT_ORDER, package_identity_key

SNAPSHOT_SCHEMA_VERSION = 2
# Half-open range: a reader accepts schema_version >= low and < high. Readers accept 1 and 2
# (AP9-SV2-0, 2026-09-29): 1 reads as 2, unchanged (`load_snapshot`'s own normalization, below).
SNAPSHOT_SCHEMA_RANGE: tuple[int, int] = (1, 3)

HARDWARE_SCHEMA_VERSION = 1
# Own range, same half-open convention as SNAPSHOT_SCHEMA_RANGE -- a hardware profile and a
# fetch snapshot version independently of each other.
HARDWARE_SCHEMA_RANGE: tuple[int, int] = (1, 2)

# Each half starts with a letter or digit (the Hub's own rule): `Qwen/..`, `../x` or `Qwen/-a`
# would otherwise pass the character class and become a path segment of a URL this package builds.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_APPROVAL_CONTENT_RE = re.compile(r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")
_OLLAMA_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*:[A-Za-z0-9][A-Za-z0-9._-]*$")
# The tag half of an `ollama_name` on its own, for the other tags of one manifest (`Package.aliases`).
_OLLAMA_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MACHINE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# A hardware profile's identity (profile v2, `modelroom/profile.py`): 16 lowercase hex
# characters, random, created once per profile file. Defined here because the configuration's
# `[machines.<name>].profile` refers to it too.
PROFILE_ID_RE = re.compile(r"^[0-9a-f]{16}$")
# Qwen's own convention inserts an extra `-split` marker before the counter
# (`Qwen3VL-235B-A22B-Instruct-F16-split-00001-of-00010.gguf`); the counter itself is otherwise
# identical to every other packager's `-NNNNN-of-NNNNN` suffix.
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?:split-)?(?P<idx>\d{5})-of-(?P<total>\d{5})(?:\.[^.]+)?$")


class SchemaVersionError(Exception):
    """A persisted file's `schema_version` is outside the range its reader accepts."""


def check_schema_version(version: int, accepted: tuple[int, int], file_label: str) -> None:
    """Raise `SchemaVersionError` unless `accepted[0] <= version < accepted[1]`.

    The message names the file, the version found and the accepted range, so the CLI can
    print it as-is before mapping the failure to exit code 3.
    """
    low, high = accepted
    if not (low <= version < high):
        raise SchemaVersionError(
            f"{file_label}: schema_version {version} is outside the accepted range "
            f"(accepted: >= {low} and < {high}; equivalently [{low}, {high}))"
        )


def validate_hf_repo(value: str) -> str:
    """Validate that `value` looks like a Hugging Face `owner/name` repo id.

    Shared by `BaseModelSpec` and `config.BaseModelConfig` so the rule is defined once.
    """
    if not _HF_REPO_RE.fullmatch(value):
        raise ValueError(f"hf_repo must look like 'owner/name': {value!r}")
    return value


def hf_id_key(value: str) -> str:
    """The form in which two Hugging Face account or repository ids are compared (AP9-KO): the Hub treats them
    case-insensitively, so `othercorp/Nova-7B-GGUF` and `OtherCorp/nova-7b-gguf` are one repository. Only for
    comparing, indexing and keying -- an id is stored and shown as it was written; file names and revisions
    are never folded."""
    return value.casefold()


def validate_repo_aliases(value: list[str]) -> list[str]:
    """Validate that every alias is a non-empty repo *name*, never `owner/name`.

    Shared by `BaseModelSpec` and `config.BaseModelConfig` so the rule is defined once.
    """
    for alias in value:
        if not alias:
            raise ValueError("repo_aliases entries must be non-empty")
        if "/" in alias:
            raise ValueError(f"repo_aliases holds repo names, not 'owner/name': {alias!r}")
    return value


def validate_ollama_name(value: str) -> str:
    """Validate an Ollama model name `<base>:<tag>`; shared by `Package` and the guided shapes."""
    if not _OLLAMA_NAME_RE.fullmatch(value):
        raise ValueError(f"ollama name must look like '<base>:<tag>': {value!r}")
    return value


def validate_ollama_pair(ollama_base: str | None, ollama_tag: str | None) -> None:
    """Validate that `ollama_base` and `ollama_tag` are both set, or both `None`.

    Shared by `BaseModelSpec` and `config.BaseModelConfig` so the rule is defined once.
    """
    if (ollama_base is None) != (ollama_tag is None):
        raise ValueError("ollama_base and ollama_tag must both be set, or both be None")


def validate_machine_name(value: str) -> str:
    """Validate that `value` looks like a configured machine name (`^[a-z0-9][a-z0-9-]*$`).

    Shared by `HardwareSnapshot.machine` (this module) and the keys of `Configuration.machines`
    (`modelroom/config.py`) so a machine name means the same thing everywhere it appears.
    """
    if not _MACHINE_NAME_RE.fullmatch(value):
        raise ValueError(f"machine name must match '[a-z0-9][a-z0-9-]*$': {value!r}")
    return value


def check_aware_utc(value: datetime, field_name: str) -> datetime:
    """Validate that `value` is a timezone-aware datetime at UTC offset.

    Every persisted timestamp in this module uses this: AP4's hardware/measurement fields
    (`HardwareSnapshot.measured_at`, `InstalledModel.observed_at`,
    `Measurement.profile_measured_at`/`measured_at`) and, since fix-round 5 (P2-3),
    `Package.observed_at`/`last_seen`, `Area.last_success` and `Snapshot.run_at` -- a naive
    datetime, or one at a non-UTC offset, is rejected rather than silently accepted and later
    misread (probe: a naive `Snapshot.run_at` validated without complaint, then
    `state.check_run_is_newer` raised `TypeError: can't compare offset-naive and offset-aware
    datetimes` out of `cli.main`). Earlier snapshot/config datetimes did not need this check
    spelled out because every producer already emitted UTC ISO-8601 with a `Z`/`+00:00` suffix;
    AP4 made the rule explicit and checked for its own new fields, rather than only
    conventional, and fix-round 5 closes the gap for the fields AP3 had already shipped.
    """
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be an aware UTC datetime: {value!r}")
    return value


def shards_complete(files: list["PackageFile"]) -> bool:
    """Whether a package's weight files are a complete set.

    A single non-sharded weights file is complete. Zero weight files is never complete. A
    sharded set (filenames ending in `-NNNNN-of-NNNNN`, optionally preceded by `-split` and
    optionally before an extension -- Qwen's own convention) is complete only when every shard
    shares the same normalized stem and the same total `N`, the indices present are exactly
    `{1..N}` with no duplicate index, and no non-sharded file is mixed in. Mixed stems, a mixed
    sharded/non-sharded set, or any duplicate index makes the package incomplete. Files with
    role `mmproj` or `other` are not weights and are ignored.
    """
    weight_files = [f for f in files if f.role in ("weights", "weights_shard")]
    if not weight_files:
        return False

    matches = [_SHARD_RE.match(f.name) for f in weight_files]
    if all(m is None for m in matches):
        return len(weight_files) == 1
    if any(m is None for m in matches):
        return False

    stems = {m.group("stem") for m in matches}
    if len(stems) != 1:
        return False

    totals = {int(m.group("total")) for m in matches}
    if len(totals) != 1:
        return False
    total = totals.pop()

    indices = [int(m.group("idx")) for m in matches]
    if len(indices) != len(set(indices)):
        return False
    return set(indices) == set(range(1, total + 1))


class Architecture(BaseModel):
    """A base model's transformer shape, read from the publisher repo's `config.json`.

    Only the publisher repo carries this file; packager (GGUF) repos never do, so this is
    always attached to a `BaseModelSpec`, never derived from a `Package`.
    """

    model_config = ConfigDict(extra="forbid")

    source_repo: str
    source_revision: str | None = None
    kind: Literal["dense_classic", "hybrid_linear", "unknown"]
    # R7-10 (fix-round 6): `le=2**31 - 1` bounds every numeric field -- `gt=0` alone never
    # bounded the top end, so an implausible value like `10**400` (which `json.loads` parses
    # without complaint) validated fine and then overflowed `fit.py::compute_fit`'s arithmetic
    # (`OverflowError: integer division result too large for a float`), aborting the whole
    # render instead of ending this one package's fit `"unknown"`. The bound is documented in
    # CONTRACTS.md ("Fit contract v1"); `compute_fit` also catches `OverflowError` directly as a
    # second line of defense for a combination of in-bound values that still overflows together.
    num_hidden_layers: int | None = Field(default=None, gt=0, le=2**31 - 1)
    num_key_value_heads: int | None = Field(default=None, gt=0, le=2**31 - 1)
    head_dim: int | None = Field(default=None, gt=0, le=2**31 - 1)
    # AP9-SV2-P (ADR 0014): the attention head count, read whenever the config names one --
    # independent of `kind` (kept even for `unknown`, the same way `max_context` already is --
    # a tensor-parallel package may still need it when the rest of the shape could not be read).
    # vLLM's own tensor-parallel rule requires the head count to divide evenly by the card count;
    # `t = 1` is used wherever this is `None` (SV2-F).
    num_attention_heads: int | None = Field(default=None, gt=0, le=2**31 - 1)
    layer_types: list[str] | None = None
    max_context: int | None = Field(default=None, gt=0, le=2**31 - 1)
    # `hybrid_linear` only (AP9-UX2 P1, ADR 0010): the linear-attention layers' state, read from
    # `config.json`; `linear_state_bytes` is the byte width of one state element (`mamba_ssm_dtype`).
    linear_num_key_heads: int | None = Field(default=None, gt=0, le=2**31 - 1)
    linear_num_value_heads: int | None = Field(default=None, gt=0, le=2**31 - 1)
    linear_key_head_dim: int | None = Field(default=None, gt=0, le=2**31 - 1)
    linear_value_head_dim: int | None = Field(default=None, gt=0, le=2**31 - 1)
    linear_conv_kernel_dim: int | None = Field(default=None, gt=0, le=2**31 - 1)
    linear_state_bytes: int | None = Field(default=None, gt=0, le=2**31 - 1)

    @field_validator("source_revision")
    @classmethod
    def _check_source_revision(cls, value: str | None) -> str | None:
        if value is not None and not _SHA1_RE.fullmatch(value):
            raise ValueError(f"source_revision must be a 40-hex commit sha or None: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_completeness(self) -> "Architecture":
        linear = [getattr(self, name) for name in LINEAR_FIELDS]
        if self.kind != "hybrid_linear" and any(value is not None for value in linear):
            raise ValueError(f"{self.kind} carries none of the linear_* fields")
        if self.kind == "unknown":
            return self
        if None in (self.num_hidden_layers, self.num_key_value_heads, self.head_dim):
            raise ValueError(f"{self.kind} requires num_hidden_layers, num_key_value_heads and head_dim")
        types = self.layer_types
        if self.kind == "dense_classic":
            if types is not None and any(layer != "full_attention" for layer in types):
                raise ValueError("dense_classic requires layer_types to be None or all 'full_attention'")
            return self
        if types is None or len(types) != self.num_hidden_layers or set(types) != set(HYBRID_LAYER_TYPES):
            raise ValueError(
                "hybrid_linear requires layer_types of num_hidden_layers entries, "
                "each full_attention or linear_attention and both present"
            )
        if None in linear:
            raise ValueError("hybrid_linear requires every linear_* field")
        return self


# The fields only a `hybrid_linear` architecture carries, and the two layer types it is made of.
LINEAR_FIELDS = (
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "linear_state_bytes",
)
HYBRID_LAYER_TYPES = ("full_attention", "linear_attention")


class BaseModelSpec(BaseModel):
    """One publisher base model that packagers build GGUF/tensor packages from."""

    model_config = ConfigDict(extra="forbid")

    hf_repo: str
    repo_aliases: list[str] = Field(default_factory=list)
    ollama_base: str | None = None
    ollama_tag: str | None = None
    publisher: str
    # F11: None means "not measured this run, and no previous reading exists" -- never a
    # fabricated placeholder that could be mistaken for a real reading. A float value is still
    # required to be a real, positive parameter count.
    parameters_b: float | None = Field(default=None, gt=0)
    architecture: Architecture

    @field_validator("hf_repo")
    @classmethod
    def _check_hf_repo(cls, value: str) -> str:
        return validate_hf_repo(value)

    @field_validator("repo_aliases")
    @classmethod
    def _check_repo_aliases(cls, value: list[str]) -> list[str]:
        return validate_repo_aliases(value)

    @model_validator(mode="after")
    def _check_ollama_pair(self) -> "BaseModelSpec":
        validate_ollama_pair(self.ollama_base, self.ollama_tag)
        return self


class Family(BaseModel):
    """A search term (e.g. `qwen3.5`) and the base models it resolves to."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_models: list[BaseModelSpec] = Field(min_length=1)


class PackageFile(BaseModel):
    """One file inside a package: a weight (shard), an mmproj projector, or something else."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: Literal["weights", "weights_shard", "mmproj", "other"]
    size_bytes: int = Field(ge=0)
    digest: str | None = None


class Approval(BaseModel):
    """A human decision that vouches for a specific package content, despite its metadata."""

    model_config = ConfigDict(extra="forbid")

    date: date
    content: str
    by: str

    @field_validator("content")
    @classmethod
    def _check_content(cls, value: str) -> str:
        if not _APPROVAL_CONTENT_RE.fullmatch(value):
            raise ValueError(
                f"content must be a 40-hex commit sha or 'sha256:' plus 64 hex characters: {value!r}"
            )
        return value


class Package(BaseModel):
    """One packaged build of a base model, as observed on Hugging Face or in Ollama.

    Exactly the fields of its `source` are set: a Hugging Face package carries `repo` and
    `revision`, never `ollama_name`/`manifest_digest`, and vice versa.

    `aliases` are the other tags of the very same Ollama manifest, which is why they are no
    packages of their own (`ollama._assemble_packages`, decided 2026-09-25); optional and empty by
    default, so a snapshot written before it reads unchanged and `schema_version` stays where it is.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["huggingface", "ollama"]
    repo: str | None = None
    revision: str | None = None
    ollama_name: str | None = None
    manifest_digest: str | None = None
    aliases: list[str] = Field(default_factory=list)
    base_model_hf_repo: str
    format: Literal["gguf", "tensor", "unknown"]
    files: list[PackageFile] = Field(default_factory=list)
    complete: bool
    quantization: str | None = None
    # Where `quantization` came from: a GGUF filename, a tensor repo's `config.json`, or nothing
    # yet (also Ollama's own case, which reads a tag, never a config -- AP9-SV2-0, E8). Additive,
    # defaults to `None`, so a snapshot written before it reads unchanged.
    quantization_source: Literal["filename", "config"] | None = None
    default_context: int | None = Field(default=None, gt=0)
    provenance: Literal["from_metadata", "approved", "unresolved"]
    unresolved_reason: str | None = None
    approval: Approval | None = None
    observed_at: datetime
    last_seen: datetime
    active: bool

    @field_validator("observed_at")
    @classmethod
    def _check_observed_at(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "observed_at")

    @field_validator("last_seen")
    @classmethod
    def _check_last_seen(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "last_seen")

    @field_validator("ollama_name")
    @classmethod
    def _check_ollama_name(cls, value: str | None) -> str | None:
        if value is not None and not _OLLAMA_NAME_RE.fullmatch(value):
            raise ValueError(f"ollama_name must look like '<base>:<tag>': {value!r}")
        return value

    @field_validator("quantization")
    @classmethod
    def _check_quantization(cls, value: str | None) -> str | None:
        if value is not None and value not in QUANT_ORDER:
            raise ValueError(f"quantization must be a QUANT_ORDER member or None: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_aliases(self) -> "Package":
        """Every alias is another tag of this package's own manifest, each named once. A Hugging
        Face package has none: one repository holds one build under one file name."""
        if self.source == "huggingface":
            if self.aliases:
                raise ValueError("a huggingface package carries no aliases")
            return self
        own_tag = (self.ollama_name or "").partition(":")[2]
        for index, alias in enumerate(self.aliases):
            if not _OLLAMA_TAG_RE.fullmatch(alias):
                raise ValueError(f"an alias must be an Ollama tag: {alias!r}")
            if alias == own_tag:
                raise ValueError(f"an alias must not repeat the package's own tag: {alias!r}")
            if alias in self.aliases[:index]:
                raise ValueError(f"the alias {alias!r} is named twice")
        return self

    @model_validator(mode="after")
    def _check_source_fields(self) -> "Package":
        if self.source == "huggingface":
            if self.ollama_name is not None or self.manifest_digest is not None:
                raise ValueError("a huggingface package must not carry ollama_name/manifest_digest")
            if self.repo is None or self.revision is None:
                raise ValueError("a huggingface package requires repo and revision")
            if not _HF_REPO_RE.fullmatch(self.repo):
                raise ValueError(f"repo must look like 'owner/name': {self.repo!r}")
            if not _SHA1_RE.fullmatch(self.revision):
                raise ValueError(f"revision must be a 40-hex commit sha: {self.revision!r}")
        else:
            if self.repo is not None or self.revision is not None:
                raise ValueError("an ollama package must not carry repo/revision")
            if self.ollama_name is None or self.manifest_digest is None:
                raise ValueError("an ollama package requires ollama_name and manifest_digest")
            if not _MANIFEST_DIGEST_RE.fullmatch(self.manifest_digest):
                raise ValueError(
                    f"manifest_digest must be 'sha256:' plus 64 hex characters: "
                    f"{self.manifest_digest!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_complete_is_derived_from_files(self) -> "Package":
        derived = shards_complete(self.files)
        if self.complete != derived:
            raise ValueError(
                f"complete must be the value derived from files by shards_complete: "
                f"expected {derived}, got {self.complete}"
            )
        return self

    @model_validator(mode="after")
    def _check_provenance_invariants(self) -> "Package":
        # AP9-SV2-0 (E8, docs/adr/0014-…): a Hugging Face tensor package is the one exception to
        # the format lock below -- SV2-P's provenance rule may resolve it. Every other non-gguf
        # package (Ollama's own `tensor` packages, an MLX tag included -- noted 2026-09-28 --
        # and any `unknown` format) stays locked, same as before this step.
        is_hf_tensor = self.format == "tensor" and self.source == "huggingface"
        if self.format != "gguf" and not is_hf_tensor and (self.provenance != "unresolved" or self.unresolved_reason != "format"):
            raise ValueError(
                "a non-gguf package must have provenance='unresolved' and unresolved_reason='format', "
                "unless it is a Hugging Face tensor package"
            )
        if self.provenance == "unresolved":
            if not self.unresolved_reason:
                raise ValueError("unresolved_reason is required when provenance is 'unresolved'")
        elif self.unresolved_reason is not None:
            raise ValueError("unresolved_reason is only set when provenance is 'unresolved'")
        if self.provenance == "approved":
            if self.approval is None:
                raise ValueError("provenance 'approved' requires an approval")
            current_content = self.revision if self.source == "huggingface" else self.manifest_digest
            if self.approval.content != current_content:
                raise ValueError(
                    "an 'approved' package's approval.content must match its current "
                    "revision (huggingface) or manifest_digest (ollama)"
                )
        return self


class Area(BaseModel):
    """One fetch unit: a base model on one source, optionally scoped to one HF packager."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["huggingface", "ollama"]
    base_model_hf_repo: str
    packager: str | None = None
    status: Literal["complete", "incomplete"]
    last_success: datetime | None = None
    error: str | None = None

    @field_validator("last_success")
    @classmethod
    def _check_last_success(cls, value: datetime | None) -> datetime | None:
        return check_aware_utc(value, "last_success") if value is not None else None

    @model_validator(mode="after")
    def _check_error_present_when_incomplete(self) -> "Area":
        if self.status == "incomplete" and not self.error:
            raise ValueError("error is required when status is 'incomplete'")
        return self


class Snapshot(BaseModel):
    """One `fetch` run: the areas it covered, the base models it knows, the packages it found."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    run_at: datetime
    areas: list[Area] = Field(default_factory=list)
    base_models: list[BaseModelSpec] = Field(default_factory=list)
    packages: list[Package] = Field(default_factory=list)

    @field_validator("run_at")
    @classmethod
    def _check_run_at(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "run_at")

    @model_validator(mode="after")
    def _check_schema_version(self) -> "Snapshot":
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SNAPSHOT_SCHEMA_VERSION}, got {self.schema_version}"
            )
        return self

    @model_validator(mode="after")
    def _check_packages_reference_known_base_models(self) -> "Snapshot":
        known = {base_model.hf_repo for base_model in self.base_models}
        for package in self.packages:
            if package.base_model_hf_repo not in known:
                raise ValueError(
                    f"package base_model_hf_repo {package.base_model_hf_repo!r} is not in base_models"
                )
        return self

    @model_validator(mode="after")
    def _check_package_identity_is_unique(self) -> "Snapshot":
        seen: set[tuple[str, str, str]] = set()
        for package in self.packages:
            key = package_identity_key(package)
            if key in seen:
                raise ValueError(f"duplicate package identity: {key}")
            seen.add(key)
        return self


def load_snapshot(data: dict) -> Snapshot:
    """Validate `schema_version` before field validation, then build a `Snapshot`.

    A raw `Snapshot.model_validate(data)` call runs the schema-version check as one of the
    model's own validators, so an out-of-range or missing version surfaces as a pydantic
    `ValidationError` mixed in with every other field problem. Callers that need to map a
    schema-version problem to a specific exit code (`3` in this project) call `load_snapshot`
    instead: it raises `SchemaVersionError` -- for a missing, non-integer, or out-of-range
    `schema_version` -- before Pydantic ever sees the payload. The model's own version check
    stays in place as a second line of defense for callers that build a `Snapshot` some other
    way.
    """
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"snapshot: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, SNAPSHOT_SCHEMA_RANGE, "snapshot")
    return Snapshot.model_validate(_read_schema_1_as_2(_read_provenance_of_0_1_0(data)))


def _read_schema_1_as_2(data: dict) -> dict:
    """The snapshot dict with `schema_version` 1 read as 2; `data` is left unchanged otherwise.

    Schema 2 (AP9-SV2-0) is purely additive over schema 1 -- every new field
    (`Package.quantization_source`, the new `QUANT_ORDER` labels a tensor package's
    `quantization` may now carry) has a default, so a schema-1 file needs no other change to
    validate as schema 2. This is the one place that makes the substitution; `data["schema_version"]`
    is otherwise never rewritten by a reader.
    """
    if data.get("schema_version") == 1:
        return {**data, "schema_version": SNAPSHOT_SCHEMA_VERSION}
    return data


def _read_provenance_of_0_1_0(data: dict) -> dict:
    """The snapshot dict with the provenance `metadata_ok` read as `from_metadata`; `data` is left unchanged.

    Snapshot schema 1 written by 0.1.0 spells the provenance `metadata_ok`; ADR 0007
    (2026-09-27) renamed it `from_metadata` and kept schema 1, so the reader takes the old
    spelling once and every write spells the new one. No other field is touched.
    """
    packages = data.get("packages")
    if not isinstance(packages, list):
        return data
    renamed = [
        {**package, "provenance": "from_metadata"}
        if isinstance(package, dict) and package.get("provenance") == "metadata_ok"
        else package
        for package in packages
    ]
    return {**data, "packages": renamed}


# --- hardware profile (AP4) -----------------------------------------------------------------


class InstalledModel(BaseModel):
    """One model the local Ollama daemon reports installed, as `modelroom/ollama_local.py` saw it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    digest: str
    size_bytes: int = Field(ge=0)
    observed_at: datetime

    @field_validator("digest")
    @classmethod
    def _check_digest(cls, value: str) -> str:
        if not _MANIFEST_DIGEST_RE.fullmatch(value):
            raise ValueError(f"digest must be 'sha256:' plus 64 hex characters: {value!r}")
        return value

    @field_validator("observed_at")
    @classmethod
    def _check_observed_at(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "observed_at")


class Measurement(BaseModel):
    """One recorded `bench` result, bound to the exact package content and hardware profile it measured.

    `content_source` discriminates which of the two field groups below is populated, exactly
    the pattern `Package.source` already uses for `repo`/`revision` vs.
    `ollama_name`/`manifest_digest`. The alternative the brief allowed -- one polymorphic
    `content` field holding either a bare digest string or a `{repo, revision, file_digest}`
    object -- was not chosen, so that this model stays the same shape as every other
    coupled-field model in this file. `profile_measured_at` is not cross-checked against a
    `HardwareSnapshot.measured_at` here: AP4 only defines and validates `Measurement`, the
    renderer (a later work package) is what refuses a measurement whose profile does not match
    the snapshot it is being read alongside (see CONTRACTS.md, "Fit contract v1").
    """

    model_config = ConfigDict(extra="forbid")

    content_source: Literal["ollama", "huggingface"]
    ollama_manifest_digest: str | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None
    hf_file_digest: str | None = None
    context: int = Field(gt=0)
    runtime: str
    profile_measured_at: datetime
    measured_at: datetime
    tps_mean: float = Field(gt=0)
    tps_range: tuple[float, float]

    @field_validator("profile_measured_at")
    @classmethod
    def _check_profile_measured_at(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "profile_measured_at")

    @field_validator("measured_at")
    @classmethod
    def _check_measured_at(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "measured_at")

    @model_validator(mode="after")
    def _check_content_fields(self) -> "Measurement":
        hf_fields = (self.hf_repo, self.hf_revision, self.hf_file_digest)
        if self.content_source == "ollama":
            if self.ollama_manifest_digest is None or any(f is not None for f in hf_fields):
                raise ValueError(
                    "content_source 'ollama' requires ollama_manifest_digest and no hf_* fields"
                )
            if not _MANIFEST_DIGEST_RE.fullmatch(self.ollama_manifest_digest):
                raise ValueError(
                    f"ollama_manifest_digest must be 'sha256:' plus 64 hex characters: "
                    f"{self.ollama_manifest_digest!r}"
                )
        else:
            if self.ollama_manifest_digest is not None or any(f is None for f in hf_fields):
                raise ValueError(
                    "content_source 'huggingface' requires hf_repo, hf_revision and "
                    "hf_file_digest, and no ollama_manifest_digest"
                )
            if not _HF_REPO_RE.fullmatch(self.hf_repo):
                raise ValueError(f"hf_repo must look like 'owner/name': {self.hf_repo!r}")
            if not _SHA1_RE.fullmatch(self.hf_revision):
                raise ValueError(f"hf_revision must be a 40-hex commit sha: {self.hf_revision!r}")
            if not _MANIFEST_DIGEST_RE.fullmatch(self.hf_file_digest):
                raise ValueError(
                    f"hf_file_digest must be 'sha256:' plus 64 hex characters: {self.hf_file_digest!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_tps_range(self) -> "Measurement":
        low, high = self.tps_range
        if low > high:
            raise ValueError(f"tps_range low must be <= high: {self.tps_range!r}")
        return self


class HardwareSnapshot(BaseModel):
    """One machine's measured hardware, as `hardware` wrote it to `<state>/hardware/<machine>.json`.

    Reserved headroom (`reserve_ram_gib`/`reserve_vram_gib`) is deliberately not a field here:
    it is operator policy chosen once per machine, not a measured fact, and it can change
    without a new hardware measurement -- it lives in the configuration's `machines.<name>`
    (`config.MachineConfig`) instead, and `modelroom/fit.py::compute_fit` takes both a
    `HardwareSnapshot` and a `MachineConfig` as separate parameters rather than merging them.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    machine: str
    measured_at: datetime
    llmfit_version: str
    vram_gib: float = Field(ge=0)
    ram_gib: float = Field(gt=0)
    free_ram_gib_at_measurement: float | None = Field(default=None, ge=0)
    gpu_name: str | None = None
    backend: str | None = None
    unified_memory: bool
    installed: list[InstalledModel] | None = None
    installed_unavailable_reason: str | None = None
    measurements: list[Measurement] = Field(default_factory=list)

    @field_validator("machine")
    @classmethod
    def _check_machine(cls, value: str) -> str:
        return validate_machine_name(value)

    @field_validator("measured_at")
    @classmethod
    def _check_measured_at(cls, value: datetime) -> datetime:
        return check_aware_utc(value, "measured_at")

    @model_validator(mode="after")
    def _check_schema_version(self) -> "HardwareSnapshot":
        if self.schema_version != HARDWARE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {HARDWARE_SCHEMA_VERSION}, got {self.schema_version}"
            )
        return self

    @model_validator(mode="after")
    def _check_installed_xor_reason(self) -> "HardwareSnapshot":
        if self.installed is not None:
            if self.installed_unavailable_reason is not None:
                raise ValueError("installed_unavailable_reason must be None when installed is a list")
        elif not self.installed_unavailable_reason:
            raise ValueError("installed_unavailable_reason is required when installed is None")
        return self


def load_hardware_snapshot(data: dict) -> HardwareSnapshot:
    """Validate `schema_version` before field validation, then build a `HardwareSnapshot`.

    Same convention as `load_snapshot`: raises `SchemaVersionError` -- for a missing,
    non-integer, or out-of-range `schema_version` -- before Pydantic ever sees the payload.
    """
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(
            f"hardware snapshot: schema_version is missing or not an integer: {version!r}"
        )
    check_schema_version(version, HARDWARE_SCHEMA_RANGE, "hardware snapshot")
    return HardwareSnapshot.model_validate(data)


# --- render (AP5) ---------------------------------------------------------------------------

# `Fit` and `Rating` moved to `contracts_fit.py` (AP9-SV2-0, docs/adr/0014-runtime-entries-and-
# vllm-packages.md) to keep this module under the size guard as the snapshot schema grows; both
# names are imported and re-exported at the top of this module, so every existing
# `from .contracts import Fit`/`Rating` keeps working, unchanged and pointing at the same classes.
