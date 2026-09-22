"""Shared data shapes for modelroom: family/base-model identity, packages, snapshots.

Pydantic v2 models with `extra="forbid"`, explicit value ranges and validators for coupled
fields. This is the only definition; `CONTRACTS.md` describes each model for readers and
repeats `EXAMPLES` verbatim. See `CONTRACTS.md` for the rules these models follow.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .quantization import package_identity_key

SNAPSHOT_SCHEMA_VERSION = 1
# Inclusive lower bound, exclusive upper bound: readers accept schema_version 1 for now.
SNAPSHOT_SCHEMA_RANGE: tuple[int, int] = (1, 2)

_HF_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})(?=\.[^.]+$)")


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
            f"[{low}, {high})"
        )


def shards_complete(files: list["PackageFile"]) -> bool:
    """Whether a package's weight files are a complete set.

    A single non-sharded weights file is complete. A sharded set (filenames ending in
    `-NNNNN-of-NNNNN` before the extension) is complete when every index `1..N` is present
    for a shared `N`. Files with role `mmproj` or `other` are not weights and are ignored.
    """
    weight_files = [f for f in files if f.role in ("weights", "weights_shard")]
    if not weight_files:
        return False

    matches = [_SHARD_RE.search(f.name) for f in weight_files]
    if all(m is None for m in matches):
        return len(weight_files) == 1
    if any(m is None for m in matches):
        return False

    totals = {int(m.group(2)) for m in matches}
    if len(totals) != 1:
        return False
    total = totals.pop()
    indices = {int(m.group(1)) for m in matches}
    return indices == set(range(1, total + 1))


class Architecture(BaseModel):
    """A base model's transformer shape, read from the publisher repo's `config.json`.

    Only the publisher repo carries this file; packager (GGUF) repos never do, so this is
    always attached to a `BaseModelSpec`, never derived from a `Package`.
    """

    model_config = ConfigDict(extra="forbid")

    source_repo: str
    source_revision: str | None = None
    kind: Literal["dense_classic", "unknown"]
    num_hidden_layers: int | None = Field(default=None, gt=0)
    num_key_value_heads: int | None = Field(default=None, gt=0)
    head_dim: int | None = Field(default=None, gt=0)
    layer_types: list[str] | None = None
    max_context: int | None = Field(default=None, gt=0)

    @field_validator("source_revision")
    @classmethod
    def _check_source_revision(cls, value: str | None) -> str | None:
        if value is not None and not _SHA1_RE.match(value):
            raise ValueError(f"source_revision must be a 40-hex commit sha or None: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_dense_classic_completeness(self) -> "Architecture":
        if self.kind == "dense_classic":
            if None in (self.num_hidden_layers, self.num_key_value_heads, self.head_dim):
                raise ValueError(
                    "dense_classic requires num_hidden_layers, num_key_value_heads and head_dim"
                )
            if self.layer_types is not None and any(
                layer != "full_attention" for layer in self.layer_types
            ):
                raise ValueError(
                    "dense_classic requires layer_types to be None or all 'full_attention'"
                )
        return self


def architecture_from_hf_config(
    source_repo: str, source_revision: str | None, config: dict | None
) -> Architecture:
    """Build an `Architecture` from a publisher repo's parsed `config.json`.

    Some publishers nest the language-model fields under a `text_config` key (multimodal
    configs); this reads from there when present, otherwise from the top level. `kind` is
    `dense_classic` only when the three numeric fields are all present and `layer_types` is
    either absent or entirely `"full_attention"` (a hybrid or MoE config, or a missing file,
    resolves to `"unknown"` with the numeric fields left at `None`).
    """
    if config is None:
        return Architecture(source_repo=source_repo, source_revision=source_revision, kind="unknown")

    layer_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config

    num_hidden_layers = layer_config.get("num_hidden_layers")
    num_key_value_heads = layer_config.get("num_key_value_heads")
    head_dim = layer_config.get("head_dim")
    layer_types = layer_config.get("layer_types")
    max_context = layer_config.get("max_position_embeddings") or config.get("max_position_embeddings")

    has_numeric = None not in (num_hidden_layers, num_key_value_heads, head_dim)
    all_full_attention = layer_types is None or all(t == "full_attention" for t in layer_types)
    kind: Literal["dense_classic", "unknown"] = (
        "dense_classic" if has_numeric and all_full_attention else "unknown"
    )

    return Architecture(
        source_repo=source_repo,
        source_revision=source_revision,
        kind=kind,
        num_hidden_layers=num_hidden_layers if kind == "dense_classic" else None,
        num_key_value_heads=num_key_value_heads if kind == "dense_classic" else None,
        head_dim=head_dim if kind == "dense_classic" else None,
        layer_types=layer_types if kind == "dense_classic" else None,
        max_context=max_context,
    )


class BaseModelSpec(BaseModel):
    """One publisher base model that packagers build GGUF/tensor packages from."""

    model_config = ConfigDict(extra="forbid")

    hf_repo: str
    repo_aliases: list[str] = Field(default_factory=list)
    ollama_base: str | None = None
    ollama_tag: str | None = None
    publisher: str
    parameters_b: float = Field(gt=0)
    architecture: Architecture

    @field_validator("hf_repo")
    @classmethod
    def _check_hf_repo(cls, value: str) -> str:
        if not _HF_REPO_RE.match(value):
            raise ValueError(f"hf_repo must look like 'owner/name': {value!r}")
        return value

    @field_validator("repo_aliases")
    @classmethod
    def _check_repo_aliases(cls, value: list[str]) -> list[str]:
        for alias in value:
            if "/" in alias:
                raise ValueError(f"repo_aliases holds repo names, not 'owner/name': {alias!r}")
        return value

    @model_validator(mode="after")
    def _check_ollama_pair(self) -> "BaseModelSpec":
        if (self.ollama_base is None) != (self.ollama_tag is None):
            raise ValueError("ollama_base and ollama_tag must both be set, or both be None")
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
    """A human decision that a specific package content is trustworthy, despite its metadata."""

    model_config = ConfigDict(extra="forbid")

    date: date
    content: str
    by: str


class Package(BaseModel):
    """One packaged build of a base model, as observed on Hugging Face or in Ollama.

    Exactly the fields of its `source` are set: a Hugging Face package carries `repo` and
    `revision`, never `ollama_name`/`manifest_digest`, and vice versa.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["huggingface", "ollama"]
    repo: str | None = None
    revision: str | None = None
    ollama_name: str | None = None
    manifest_digest: str | None = None
    base_model_hf_repo: str
    format: Literal["gguf", "tensor", "unknown"]
    files: list[PackageFile] = Field(default_factory=list)
    complete: bool
    quantization: str | None = None
    default_context: int | None = Field(default=None, gt=0)
    provenance: Literal["metadata_ok", "approved", "unresolved"]
    unresolved_reason: str | None = None
    approval: Approval | None = None
    observed_at: datetime
    last_seen: datetime
    active: bool

    @model_validator(mode="after")
    def _check_source_fields(self) -> "Package":
        if self.source == "huggingface":
            if self.ollama_name is not None or self.manifest_digest is not None:
                raise ValueError("a huggingface package must not carry ollama_name/manifest_digest")
            if self.repo is None or self.revision is None:
                raise ValueError("a huggingface package requires repo and revision")
            if not _HF_REPO_RE.match(self.repo):
                raise ValueError(f"repo must look like 'owner/name': {self.repo!r}")
            if not _SHA1_RE.match(self.revision):
                raise ValueError(f"revision must be a 40-hex commit sha: {self.revision!r}")
        else:
            if self.repo is not None or self.revision is not None:
                raise ValueError("an ollama package must not carry repo/revision")
            if self.ollama_name is None or self.manifest_digest is None:
                raise ValueError("an ollama package requires ollama_name and manifest_digest")
            if not _MANIFEST_DIGEST_RE.match(self.manifest_digest):
                raise ValueError(
                    f"manifest_digest must be 'sha256:' plus 64 hex characters: "
                    f"{self.manifest_digest!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_provenance_coupling(self) -> "Package":
        if self.provenance == "approved" and self.approval is None:
            raise ValueError("provenance 'approved' requires an approval")
        if self.provenance != "unresolved" and self.unresolved_reason is not None:
            raise ValueError("unresolved_reason is only set when provenance is 'unresolved'")
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


EXAMPLES: dict[str, dict] = {
    "Architecture": {
        "source_repo": "acme/Nova-7B",
        "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
        "kind": "dense_classic",
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "layer_types": None,
        "max_context": 131072,
    },
    "BaseModelSpec": {
        "hf_repo": "acme/Nova-7B",
        "repo_aliases": ["Nova-7B-Instruct-GGUF"],
        "ollama_base": "nova",
        "ollama_tag": "7b",
        "publisher": "acme",
        "parameters_b": 7.0,
        "architecture": {
            "source_repo": "acme/Nova-7B",
            "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
            "kind": "dense_classic",
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layer_types": None,
            "max_context": 131072,
        },
    },
    "Family": {
        "name": "nova",
        "base_models": [
            {
                "hf_repo": "acme/Nova-7B",
                "repo_aliases": ["Nova-7B-Instruct-GGUF"],
                "ollama_base": "nova",
                "ollama_tag": "7b",
                "publisher": "acme",
                "parameters_b": 7.0,
                "architecture": {
                    "source_repo": "acme/Nova-7B",
                    "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
                    "kind": "dense_classic",
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "layer_types": None,
                    "max_context": 131072,
                },
            }
        ],
    },
    "PackageFile": {
        "name": "Nova-7B-Q4_K_M.gguf",
        "role": "weights",
        "size_bytes": 4500000000,
        "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
    },
    "Approval": {
        "date": "2026-09-01",
        "content": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
        "by": "acme-ai-team",
    },
    "Package": {
        "source": "huggingface",
        "repo": "packager/Nova-7B-GGUF",
        "revision": "9f8e7d6c5b4a3928170615243342516078960a1b",
        "base_model_hf_repo": "acme/Nova-7B",
        "format": "gguf",
        "files": [
            {
                "name": "Nova-7B-Q4_K_M.gguf",
                "role": "weights",
                "size_bytes": 4500000000,
                "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
            }
        ],
        "complete": True,
        "quantization": "Q4_K_M",
        "default_context": None,
        "provenance": "metadata_ok",
        "unresolved_reason": None,
        "approval": None,
        "observed_at": "2026-09-22T09:00:00Z",
        "last_seen": "2026-09-22T09:00:00Z",
        "active": True,
    },
    "Area": {
        "source": "huggingface",
        "base_model_hf_repo": "acme/Nova-7B",
        "packager": "packager",
        "status": "complete",
        "last_success": "2026-09-22T09:00:00Z",
        "error": None,
    },
    "Snapshot": {
        "schema_version": 1,
        "run_at": "2026-09-22T09:00:00Z",
        "areas": [
            {
                "source": "huggingface",
                "base_model_hf_repo": "acme/Nova-7B",
                "packager": "packager",
                "status": "complete",
                "last_success": "2026-09-22T09:00:00Z",
                "error": None,
            }
        ],
        "base_models": [
            {
                "hf_repo": "acme/Nova-7B",
                "repo_aliases": ["Nova-7B-Instruct-GGUF"],
                "ollama_base": "nova",
                "ollama_tag": "7b",
                "publisher": "acme",
                "parameters_b": 7.0,
                "architecture": {
                    "source_repo": "acme/Nova-7B",
                    "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
                    "kind": "dense_classic",
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "layer_types": None,
                    "max_context": 131072,
                },
            }
        ],
        "packages": [
            {
                "source": "huggingface",
                "repo": "packager/Nova-7B-GGUF",
                "revision": "9f8e7d6c5b4a3928170615243342516078960a1b",
                "base_model_hf_repo": "acme/Nova-7B",
                "format": "gguf",
                "files": [
                    {
                        "name": "Nova-7B-Q4_K_M.gguf",
                        "role": "weights",
                        "size_bytes": 4500000000,
                        "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
                    }
                ],
                "complete": True,
                "quantization": "Q4_K_M",
                "default_context": None,
                "provenance": "metadata_ok",
                "unresolved_reason": None,
                "approval": None,
                "observed_at": "2026-09-22T09:00:00Z",
                "last_seen": "2026-09-22T09:00:00Z",
                "active": True,
            }
        ],
    },
}
