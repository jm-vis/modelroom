"""The shipped catalog (`catalog.toml`): publishers, current models, successors, Ollama names.

The search uses it to tell a publisher from a packager, to call a model `latest` or `legacy`
when the repo itself says nothing (`new_version`), and to know an Ollama name. Every model row
names its evidence; nothing in it is inferred (CONTRACTS.md, "Catalog rules").
"""

from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contracts import SchemaVersionError, check_schema_version, validate_hf_repo, validate_ollama_pair

CATALOG_FILE = Path(__file__).with_name("catalog.toml")
CATALOG_SCHEMA_VERSION = 1
CATALOG_SCHEMA_RANGE: tuple[int, int] = (1, 2)

Age = Literal["latest", "legacy", "unknown"]
_HUB = "https://huggingface.co/"
# `urlsplit(...).hostname` is lowercase and without the port.
_HUB_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})


class CatalogError(Exception):
    """The catalog file cannot be read or does not validate."""


class CatalogModel(BaseModel):
    """One publisher base model: current (`latest`), superseded (`successor`), or neither.

    `latest` is a statement about this one model, never derived from a line or a number: it
    carries the publisher page or collection that says so (`latest_source`) and the day the
    maintainer checked it (`latest_checked`). Without them the model's age is `unknown`.
    """

    model_config = ConfigDict(extra="forbid")

    hf_repo: str
    latest: bool = False
    latest_source: str | None = None
    latest_checked: date | None = None
    successor: str | None = None
    ollama_base: str | None = None
    ollama_tag: str | None = None
    source: str

    @field_validator("hf_repo")
    @classmethod
    def _check_hf_repo(cls, value: str) -> str:
        return validate_hf_repo(value)

    @field_validator("source")
    @classmethod
    def _check_source(cls, value: str) -> str:
        if not value.startswith(_HUB):
            raise ValueError(f"source must be a https://huggingface.co/ page: {value!r}")
        return value

    @field_validator("latest_source")
    @classmethod
    def _check_latest_source(cls, value: str | None) -> str | None:
        if value is not None:
            parts = urlsplit(value)
            if parts.scheme != "https" or not parts.hostname:
                raise ValueError(f"latest_source must be an https page with a host: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_fields(self) -> "CatalogModel":
        validate_ollama_pair(self.ollama_base, self.ollama_tag)
        evidence = (self.latest_source, self.latest_checked)
        if self.latest and None in evidence:
            raise ValueError(f"{self.hf_repo}: latest needs latest_source and latest_checked")
        if not self.latest and evidence != (None, None):
            raise ValueError(f"{self.hf_repo}: latest_source and latest_checked belong to a latest model")
        if self.successor is not None:
            validate_hf_repo(self.successor)
            if self.latest:
                raise ValueError(f"{self.hf_repo}: a model with a successor is not latest")
            if self.successor == self.hf_repo:
                raise ValueError(f"{self.hf_repo}: successor must be another model")
        return self


class CatalogFamily(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    models: list[CatalogModel] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_models(self) -> "CatalogFamily":
        repos = [model.hf_repo for model in self.models]
        if len(repos) != len(set(repos)):
            raise ValueError(f"family {self.name}: hf_repo listed twice")
        for model in self.models:
            for repo in (model.hf_repo, model.successor):
                if repo is not None and repo.split("/", 1)[0] != self.publisher:
                    raise ValueError(f"family {self.name}: {repo} is not under publisher {self.publisher}")
            if model.latest_source is not None and not _is_publisher_evidence(model.latest_source, self.publisher):
                raise ValueError(
                    f"family {self.name}: latest_source of {model.hf_repo} on the Hub must be the page or a "
                    f"collection of publisher {self.publisher}: {model.latest_source}"
                )
        _check_no_successor_cycle(self.models, f"family {self.name}")
        return self


def _is_publisher_evidence(url: str, publisher: str) -> bool:
    """A Hub page proves `latest` only as the publisher's own page or one of its collections; a
    page off the Hub (the publisher's site) is taken as the maintainer confirmed it."""
    parts = urlsplit(url)
    if parts.hostname not in _HUB_HOSTS:
        return True
    path = parts.path.strip("/")
    return path == publisher or path.startswith(f"collections/{publisher}/")


def _check_no_successor_cycle(models: list[CatalogModel], label: str) -> None:
    successor_of = {model.hf_repo: model.successor for model in models}
    for start in successor_of:
        seen = {start}
        current = successor_of.get(start)
        while current is not None:
            if current in seen:
                raise ValueError(f"{label}: successor cycle through {current}")
            seen.add(current)
            current = successor_of.get(current)


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    families: list[CatalogFamily] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "Catalog":
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CATALOG_SCHEMA_VERSION}, got {self.schema_version}")
        names = [family.name for family in self.families]
        repos = [model.hf_repo for family in self.families for model in family.models]
        if len(names) != len(set(names)) or len(repos) != len(set(repos)):
            raise ValueError("family names and hf_repo entries must be unique across the catalog")
        _check_no_successor_cycle([m for family in self.families for m in family.models], "catalog")
        return self

    def model_for(self, hf_repo: str) -> CatalogModel | None:
        return next((m for f in self.families for m in f.models if m.hf_repo == hf_repo), None)

    def is_publisher(self, owner: str) -> bool:
        return any(family.publisher == owner for family in self.families)

    def age_of(self, hf_repo: str) -> tuple[Age, str | None]:
        """`(age, successor)` from the catalog alone: `latest`, `legacy` with its successor, or
        `unknown` when the model is not listed or listed without either statement."""
        model = self.model_for(hf_repo)
        if model is None:
            return "unknown", None
        if model.successor is not None:
            return "legacy", model.successor
        return ("latest", None) if model.latest else ("unknown", None)


def load_catalog(path: Path = CATALOG_FILE) -> Catalog:
    """Read and validate a catalog file; the default is the one shipped with the package."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CatalogError(f"{path}: cannot read catalog: {exc}") from exc
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"{path}: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, CATALOG_SCHEMA_RANGE, str(path))
    try:
        return Catalog.model_validate(data)
    except ValidationError as exc:
        raise CatalogError(f"{path}: invalid catalog: {exc}") from exc
