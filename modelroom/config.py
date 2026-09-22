"""The `Configuration` model tree and the TOML reader for `modelroom.toml`.

Pydantic v2 models with `extra="forbid"`, explicit value ranges and validators for coupled
fields, following the same rules as `modelroom/contracts.py`. `CONTRACTS.md` describes each
model for readers and repeats `EXAMPLES` verbatim under its "Configuration" section.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contracts import (
    SchemaVersionError,
    check_schema_version,
    validate_hf_repo,
    validate_machine_name,
    validate_ollama_pair,
    validate_repo_aliases,
)

CONFIG_SCHEMA_VERSION = 1
# Half-open range: a reader accepts schema_version >= low and < high, same convention as the
# snapshot's SNAPSHOT_SCHEMA_RANGE. Readers accept exactly schema_version 1 for now.
CONFIG_SCHEMA_RANGE: tuple[int, int] = (1, 2)

_FAMILY_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_MIN_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


class ConfigError(Exception):
    """A configuration file is missing, is not valid TOML, or fails validation.

    Raised by `load_config` and `Configuration.from_dict`; the CLI maps it to exit code 2.
    `SchemaVersionError` is raised separately and is never wrapped in a `ConfigError`.
    """


def _validate_owner_list(value: list[str], field_name: str) -> list[str]:
    """Validate a list of Hugging Face owner names: non-empty, no `/`, unique."""
    seen: set[str] = set()
    for owner in value:
        if not owner:
            raise ValueError(f"{field_name} entries must be non-empty")
        if "/" in owner:
            raise ValueError(f"{field_name} entries are owners, not 'owner/name': {owner!r}")
        if owner in seen:
            raise ValueError(f"{field_name} has a duplicate entry: {owner!r}")
        seen.add(owner)
    return value


class BaseModelConfig(BaseModel):
    """One base model a family resolves to, as the user configures it.

    Carries only what a user has to supply by hand. `publisher`, `parameters_b` and
    `architecture` are read from Hugging Face at fetch time and live only in the snapshot's
    `BaseModelSpec` -- duplicating them here would let the config and the snapshot disagree.
    """

    model_config = ConfigDict(extra="forbid")

    hf_repo: str
    repo_aliases: list[str] = Field(default_factory=list)
    ollama_base: str | None = None
    ollama_tag: str | None = None

    @field_validator("hf_repo")
    @classmethod
    def _check_hf_repo(cls, value: str) -> str:
        return validate_hf_repo(value)

    @field_validator("repo_aliases")
    @classmethod
    def _check_repo_aliases(cls, value: list[str]) -> list[str]:
        return validate_repo_aliases(value)

    @model_validator(mode="after")
    def _check_ollama_pair(self) -> "BaseModelConfig":
        validate_ollama_pair(self.ollama_base, self.ollama_tag)
        return self


class FamilyConfig(BaseModel):
    """A family name and the base models it resolves to, as the user configures it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_models: list[BaseModelConfig] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _FAMILY_NAME_RE.fullmatch(value):
            raise ValueError(f"name must match '[a-z0-9][a-z0-9.-]*': {value!r}")
        return value


class MachineConfig(BaseModel):
    """One machine's reserved headroom and whether it runs `fetch`.

    The machine's name is not a field here: it is the key under `Configuration.machines`,
    validated there against every key at once.
    """

    model_config = ConfigDict(extra="forbid")

    reserve_ram_gib: float = Field(ge=0)
    reserve_vram_gib: float = Field(ge=0)
    writer: bool


class PathsConfig(BaseModel):
    """Where modelroom keeps its state and its rendered Markdown output.

    Both paths must be absolute once this model validates. `load_config` resolves a relative
    `state` or `markdown` against the config file's own directory before validation; a
    programmatic caller (`Configuration.from_dict`) has to pass paths that are already
    absolute -- this package never resolves a path against the process' working directory.
    """

    model_config = ConfigDict(extra="forbid")

    state: Path
    markdown: Path

    @field_validator("state", "markdown")
    @classmethod
    def _check_absolute(cls, value: Path, info: Any) -> Path:
        if not value.is_absolute():
            raise ValueError(
                f"paths.{info.field_name} must be absolute, got {value!r}. The TOML reader "
                "resolves a relative path against the config file's own directory; a "
                "Configuration built programmatically must already pass absolute paths."
            )
        return value

    @property
    def snapshot_file(self) -> Path:
        return self.state / "modelroom.json"

    @property
    def lock_file(self) -> Path:
        return self.state / "modelroom.lock"

    @property
    def run_status_file(self) -> Path:
        return self.state / "run-status.json"

    @property
    def hardware_dir(self) -> Path:
        return self.state / "hardware"


class LlmfitConfig(BaseModel):
    """The minimum `llmfit` version this configuration requires.

    Only the requirement is stored here; checking the installed tool against it is a later
    work package.
    """

    model_config = ConfigDict(extra="forbid")

    min_version: str = "1.1.16"

    @field_validator("min_version")
    @classmethod
    def _check_min_version(cls, value: str) -> str:
        if not _MIN_VERSION_RE.fullmatch(value):
            raise ValueError(f"min_version must look like '<major>.<minor>.<patch>': {value!r}")
        return value


class Configuration(BaseModel):
    """The whole `modelroom.toml`: families, allow-listed owners, machines, paths, tool gate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    families: list[FamilyConfig] = Field(min_length=1)
    packagers: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    machines: dict[str, MachineConfig] = Field(default_factory=dict)
    paths: PathsConfig
    llmfit: LlmfitConfig = Field(default_factory=LlmfitConfig)

    @field_validator("packagers")
    @classmethod
    def _check_packagers(cls, value: list[str]) -> list[str]:
        return _validate_owner_list(value, "packagers")

    @field_validator("publishers")
    @classmethod
    def _check_publishers(cls, value: list[str]) -> list[str]:
        return _validate_owner_list(value, "publishers")

    @field_validator("machines")
    @classmethod
    def _check_machine_names(cls, value: dict[str, MachineConfig]) -> dict[str, MachineConfig]:
        for name in value:
            validate_machine_name(name)
        return value

    @model_validator(mode="after")
    def _check_schema_version(self) -> "Configuration":
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {self.schema_version}")
        return self

    @model_validator(mode="after")
    def _check_family_names_unique(self) -> "Configuration":
        names = [family.name for family in self.families]
        if len(names) != len(set(names)):
            raise ValueError("family names must be unique")
        return self

    @model_validator(mode="after")
    def _check_hf_repos_unique_across_families(self) -> "Configuration":
        repos = [
            base_model.hf_repo for family in self.families for base_model in family.base_models
        ]
        if len(repos) != len(set(repos)):
            raise ValueError("hf_repo must be unique across all families")
        return self

    @model_validator(mode="after")
    def _check_owners_are_publishers(self) -> "Configuration":
        publishers = set(self.publishers)
        for family in self.families:
            for base_model in family.base_models:
                owner = base_model.hf_repo.split("/", 1)[0]
                if owner not in publishers:
                    raise ValueError(
                        f"hf_repo owner {owner!r} ({base_model.hf_repo}) is not in publishers"
                    )
        return self

    def allowed_owners(self) -> frozenset[str]:
        """The positive list a package's repo owner must be a member of."""
        return frozenset(self.packagers) | frozenset(self.publishers)

    @classmethod
    def from_dict(cls, data: dict) -> "Configuration":
        """Build a `Configuration` from an in-memory dict (e.g. for an integrating tool).

        Checks `schema_version` before field validation, exactly like `load_config`. Paths
        under `paths` are not resolved here -- they must already be absolute. The `paths.state`
        confinement check (F7) is enforced only by `load_config`, the file reader; a
        programmatic caller owns its own paths and is trusted to have already confined them.
        """
        return _build_configuration(data, "config")


def _validate_schema_version(data: dict, label: str) -> None:
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"{label}: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, CONFIG_SCHEMA_RANGE, label)


def _build_configuration(data: dict, label: str) -> Configuration:
    _validate_schema_version(data, label)
    try:
        return Configuration.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{label}: invalid configuration: {exc}") from exc


def _resolve_relative_paths(data: dict, base_dir: Path) -> None:
    """Resolve `paths.state` and `paths.markdown` against `base_dir` when they are relative.

    This is the only place this package resolves a path against anything -- the config
    file's own directory, never the process' working directory -- and it runs only for
    `load_config`; `Configuration.from_dict` requires paths to already be absolute.
    """
    paths = data.get("paths")
    if not isinstance(paths, dict):
        return
    for key in ("state", "markdown"):
        value = paths.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            paths[key] = str(base_dir / value)


def _check_state_confinement(config: Configuration, config_dir: Path, label: str) -> None:
    """F7: `paths.state` must resolve to somewhere inside `config_dir`'s own directory tree.

    Symlinks are followed (`Path.resolve()`) before the containment check, so a symlink that
    only *looks* like it is under the config directory but actually points elsewhere is still
    rejected. `paths.markdown` is deliberately not confined -- CONTRACTS.md, "PathsConfig": it
    may live elsewhere by design.
    """
    state_resolved = config.paths.state.resolve()
    config_dir_resolved = config_dir.resolve()
    try:
        state_resolved.relative_to(config_dir_resolved)
    except ValueError:
        raise ConfigError(
            f"{label}: paths.state ({config.paths.state}) must lie inside the config file's "
            f"own directory tree ({config_dir_resolved}), but resolves to {state_resolved}"
        ) from None


def load_config(path: Path) -> Configuration:
    """Read, parse and validate a `modelroom.toml` configuration file.

    `schema_version` is checked before field validation, exactly like `load_snapshot`. `path`
    is resolved to an absolute path first (F8), so a relative `--config` argument still
    resolves `paths.state`/`paths.markdown` against the config file's real directory rather
    than against a relative `path.parent` that never becomes absolute on its own. Relative
    `paths.state`/`paths.markdown` are then resolved against that directory before validation,
    and `paths.state` is additionally confined to lie inside it (F7, `_check_state_confinement`)
    -- `paths.markdown` is not. A missing file, a TOML syntax error, or a validation error each
    become a `ConfigError` naming `path` and the underlying cause; `SchemaVersionError` passes
    through unchanged.
    """
    path = path.resolve()
    label = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"{label}: config file not found") from exc
    except OSError as exc:
        raise ConfigError(f"{label}: cannot read config file: {exc}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{label}: invalid TOML: {exc}") from exc
    _resolve_relative_paths(data, path.parent)
    config = _build_configuration(data, label)
    _check_state_confinement(config, path.parent, label)
    return config


EXAMPLES: dict[str, dict] = {
    "BaseModelConfig": {
        "hf_repo": "acme/Nova-7B",
        "repo_aliases": ["Nova-7B-Instruct-GGUF"],
        "ollama_base": "nova",
        "ollama_tag": "7b",
    },
    "FamilyConfig": {
        "name": "nova",
        "base_models": [
            {
                "hf_repo": "acme/Nova-7B",
                "repo_aliases": ["Nova-7B-Instruct-GGUF"],
                "ollama_base": "nova",
                "ollama_tag": "7b",
            }
        ],
    },
    "MachineConfig": {
        "reserve_ram_gib": 8.0,
        "reserve_vram_gib": 1.0,
        "writer": True,
    },
    "PathsConfig": {
        "state": "//models/modelroom/state",
        "markdown": "//models/modelroom/docs/models.md",
    },
    "LlmfitConfig": {
        "min_version": "1.1.16",
    },
    "Configuration": {
        "schema_version": 1,
        "families": [
            {
                "name": "nova",
                "base_models": [
                    {
                        "hf_repo": "acme/Nova-7B",
                        "repo_aliases": ["Nova-7B-Instruct-GGUF"],
                        "ollama_base": "nova",
                        "ollama_tag": "7b",
                    }
                ],
            }
        ],
        "packagers": ["packager"],
        "publishers": ["acme"],
        "machines": {
            "workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True},
        },
        "paths": {
            "state": "//models/modelroom/state",
            "markdown": "//models/modelroom/docs/models.md",
        },
        "llmfit": {"min_version": "1.1.16"},
    },
}
