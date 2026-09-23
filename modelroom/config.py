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

    @model_validator(mode="after")
    def _check_markdown_does_not_collide_with_state(self) -> "PathsConfig":
        """R7-1 (fix-round 6): the rendered document must not land inside the state directory.

        `paths.state` holds the snapshot, the lock file, the run-status file and the hardware
        profiles -- all of them internal, machine-written files. If `paths.markdown` resolved
        to one of them, or anywhere inside `paths.state`, a render would silently corrupt state
        the next `fetch` or `hardware` run depends on. Resolved with `strict=False` so this
        catches the collision even before the directories exist.
        """
        state_resolved = self.state.resolve(strict=False)
        markdown_resolved = self.markdown.resolve(strict=False)
        if markdown_resolved == state_resolved or markdown_resolved.is_relative_to(state_resolved):
            raise ValueError(
                f"paths.markdown ({self.markdown}) must not equal and must not lie inside "
                f"paths.state ({self.state}) -- the state directory holds the snapshot, the "
                "lock file, the run-status file and the hardware profiles"
            )
        return self

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
    def _check_repo_names_do_not_collide_across_base_models(self) -> "Configuration":
        """P3-6 (fix-round 5), corrected by R7-8 (fix-round 6): no two base models may claim the
        same packager repo, `owner/name`.

        `hf.py::fetch_hf_area` probes `<owner>/<name>-GGUF` and every `repo_aliases` entry, for
        every base model, under every owner in `packagers` (global to the whole configuration)
        plus that base model's own owner -- so a full `owner/name` candidate shared by two base
        models would be fetched into two areas that both produce a `Package` under the *same*
        identity key (`CONTRACTS.md`, "Identity": identity never carries `base_model_hf_repo`),
        and `state.merge_snapshot`'s `result_packages` dict would let whichever area is processed
        last silently overwrite the other's package with no error and no trace. Caught here, at
        configuration load, instead of at merge time.

        R7-8: P3-6's own check compared the bare repo *name* only, ignoring the owner -- so
        `acme/Nova` and `other/Nova` (distinct owners, `packagers = []`) were wrongly rejected as
        `'Nova-GGUF' is claimed by both`, even though `fetch_hf_area` would probe `acme/Nova-GGUF`
        and `other/Nova-GGUF`, never the same repo. Fixed by comparing the full `owner/name`
        candidate: for each base model, `owners` is `packagers` plus its own owner, `names` is
        its default `<name>-GGUF` plus `repo_aliases`, and a collision is only real when the same
        `owner/name` is produced by two *different* base models.
        """
        claimed_by: dict[str, str] = {}
        for family in self.families:
            for base_model in family.base_models:
                own_owner = base_model.hf_repo.split("/", 1)[0]
                owners = {*self.packagers, own_owner}
                default_name = f"{base_model.hf_repo.split('/', 1)[1]}-GGUF"
                names = {default_name, *base_model.repo_aliases}
                for owner in owners:
                    for name in names:
                        candidate = f"{owner}/{name}"
                        existing = claimed_by.get(candidate)
                        if existing is not None and existing != base_model.hf_repo:
                            raise ValueError(
                                f"packager repo {candidate!r} is claimed by both {existing!r} "
                                f"and {base_model.hf_repo!r} -- a repo must name at most one base model"
                            )
                        claimed_by[candidate] = base_model.hf_repo
        return self

    @model_validator(mode="after")
    def _check_ollama_tags_do_not_collide_across_base_models(self) -> "Configuration":
        """R7-7 (fix-round 6): two base models sharing an `ollama_base` must not claim
        colliding `ollama_tag` identities.

        `ollama.py::_keep_relevant_tags` keeps `tag == ollama_tag or tag.startswith(ollama_tag +
        "-")`, so a base model configured with `ollama_tag="7b"` also owns any real registry tag
        like `"7b-instruct"`. Two base models on the same `ollama_base` with equal tags, or with
        one tag a `"<tag>-"`-prefix of the other, would both resolve packages under the *same*
        Ollama identity (`<ollama_base>:<tag>`) -- `quantization.package_identity_key` does not
        carry `base_model_hf_repo`, so `state.merge_snapshot`'s `result_packages` dict lets
        whichever base model's area is processed last silently overwrite the other's package,
        exactly the P3-6 hazard this closes for Ollama instead of Hugging Face packager names.
        Caught here, at configuration load, instead of at merge time.
        """
        tags_by_base: dict[str, list[tuple[str, str]]] = {}
        for family in self.families:
            for base_model in family.base_models:
                if base_model.ollama_base is None:
                    continue
                tags_by_base.setdefault(base_model.ollama_base, []).append(
                    (base_model.ollama_tag, base_model.hf_repo)
                )
        for ollama_base, entries in tags_by_base.items():
            for index, (tag, hf_repo) in enumerate(entries):
                for other_tag, other_hf_repo in entries[index + 1 :]:
                    if tag == other_tag or tag.startswith(f"{other_tag}-") or other_tag.startswith(f"{tag}-"):
                        raise ValueError(
                            f"ollama identity collision on ollama_base {ollama_base!r}: "
                            f"{hf_repo!r} (tag {tag!r}) and {other_hf_repo!r} (tag {other_tag!r}) "
                            "would resolve packages under the same or overlapping Ollama identity"
                        )
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


def _check_markdown_not_the_config_file(config: Configuration, config_path: Path, label: str) -> None:
    """R7-1 (fix-round 6): `paths.markdown` must not resolve to the config file itself.

    `PathsConfig` already refuses a `markdown` that collides with `paths.state`; this closes
    the remaining collision that only `load_config` can see -- the config file it just read.
    """
    if config.paths.markdown.resolve(strict=False) == config_path.resolve(strict=False):
        raise ConfigError(
            f"{label}: paths.markdown ({config.paths.markdown}) must not be the config file itself"
        )


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
    _check_markdown_not_the_config_file(config, path, label)
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
