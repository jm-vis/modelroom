"""The schema-2 catalog document (`catalog.json`): the flat, publicly published shape.

A module of its own so both stay under the package's size guard. `catalog.py` owns the shared `Catalog`/`CatalogFamily`/`CatalogModel`
shape every reader works against; this module owns the rules that apply to the **document** a
schema-2 file carries before it is grouped into that shape -- the ones `CatalogModel` alone
cannot check, because they need sibling fields (`publisher` against the document's `publishers`
list) or the whole list (`models` count, duplicate `hf_repo`, one publisher per `family`).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .catalog import (
    _HUB_HOSTS,
    ACCOUNT_RE,
    CATALOG_MAX_GGUF_REPOS,  # re-exported next to the other document limits
    CATALOG_MAX_TEXT_LEN,
    CatalogError,
    CatalogModel,
    Catalog,
    PublisherInfo,
    CATALOG_JSON_SCHEMA_RANGE,
    check_version_field,
)

CATALOG_DATA_LICENSE = "CC-BY-4.0"
CATALOG_ATTRIBUTION = "modelroom catalog, jm-vis, CC BY 4.0, https://huggingface.co/datasets/jm-vis/modelroom-catalog"

# Abuse limits on the schema-2 document: a publicly published file is adversarial input. Counted
# on the parsed structure (entries), never on raw bytes or line breaks -- a byte limit alone lets
# many short, valid rows through. The per-row `gguf_repos` limit (CATALOG_MAX_GGUF_REPOS) and the
# text limit sit on the row and publisher shapes in `catalog.py`.
CATALOG_MAX_MODELS = 5000
CATALOG_MAX_PUBLISHERS = 500
CATALOG_MAX_PACKAGERS = 50


def _check_successor_evidence(model: CatalogModel) -> None:
    """`successor`, `successor_source` and `successor_checked` are set together or not at all,
    and then `successor_source` is one of three Hub pages -- the successor's own card, this
    model's own card (the `new_version` case), or a collection of the model's publisher -- or a
    page off the Hub, taken as the maintainer confirmed it."""
    filled = {model.successor is not None, model.successor_source is not None, model.successor_checked is not None}
    if len(filled) != 1:
        raise ValueError(f"{model.hf_repo}: successor, successor_source and successor_checked must be set together")
    if model.successor is None:
        return
    parts = urlsplit(model.successor_source)
    if parts.hostname not in _HUB_HOSTS:
        return
    path = parts.path.strip("/")
    allowed = {model.successor.strip("/"), model.hf_repo.strip("/")}
    if path in allowed or path.startswith(f"collections/{model.publisher}/"):
        return
    raise ValueError(
        f"{model.hf_repo}: successor_source must be the successor's card, the model's own card, "
        f"or a collection of {model.publisher}: {model.successor_source}"
    )


def _check_gguf_owners(model: CatalogModel, packagers: set[str]) -> None:
    """Every `gguf_repos` entry belongs to the model's own publisher or a listed packager -- a
    syntactically valid id of a foreign, unlisted account names no relation this catalog vouches
    for."""
    for repo in model.gguf_repos:
        owner = repo.split("/", 1)[0]
        if owner != model.publisher and owner not in packagers:
            raise ValueError(f"{model.hf_repo}: gguf_repos owner {owner!r} is neither the publisher nor a listed packager")


class CatalogDocumentV2(BaseModel):
    """The schema-2 catalog file (`catalog.json`): a flat, publicly published table.

    Every model row stands on its own (no family nesting). `to_catalog` groups the rows by
    `family` into the same shape schema 1 builds.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    published: date
    data_license: str
    attribution: str
    publishers: list[PublisherInfo] = Field(default_factory=list)
    packagers: list[str] = Field(default_factory=list)
    models: list[CatalogModel] = Field(default_factory=list)

    @field_validator("data_license")
    @classmethod
    def _check_data_license(cls, value: str) -> str:
        if value != CATALOG_DATA_LICENSE:
            raise ValueError(f"data_license must be {CATALOG_DATA_LICENSE!r}: {value!r}")
        return value

    @field_validator("attribution")
    @classmethod
    def _check_attribution(cls, value: str) -> str:
        if value != CATALOG_ATTRIBUTION:
            raise ValueError(f"attribution must be the fixed text: {value!r}")
        return value

    @field_validator("publishers")
    @classmethod
    def _check_publishers(cls, value: list[PublisherInfo]) -> list[PublisherInfo]:
        if len(value) > CATALOG_MAX_PUBLISHERS:
            raise ValueError(f"publishers exceeds {CATALOG_MAX_PUBLISHERS} entries")
        accounts = [publisher.account for publisher in value]
        if len(accounts) != len(set(accounts)):
            raise ValueError("publishers: an account is listed twice")
        return value

    @field_validator("packagers")
    @classmethod
    def _check_packagers(cls, value: list[str]) -> list[str]:
        if len(value) > CATALOG_MAX_PACKAGERS:
            raise ValueError(f"packagers exceeds {CATALOG_MAX_PACKAGERS} entries")
        for account in value:
            if len(account) > CATALOG_MAX_TEXT_LEN or not ACCOUNT_RE.fullmatch(account):
                raise ValueError(f"packagers: not an account name: {account!r}")
        if len(value) != len(set(value)):
            raise ValueError("packagers: an account is listed twice")
        return value

    @field_validator("models")
    @classmethod
    def _check_models_count(cls, value: list[CatalogModel]) -> list[CatalogModel]:
        if len(value) > CATALOG_MAX_MODELS:
            raise ValueError(f"models exceeds {CATALOG_MAX_MODELS} entries")
        return value

    @model_validator(mode="after")
    def _check(self) -> "CatalogDocumentV2":
        if self.schema_version != 2:
            raise ValueError(f"schema_version must be 2, got {self.schema_version}")
        accounts = {publisher.account for publisher in self.publishers}
        packagers = set(self.packagers)
        repos = [model.hf_repo for model in self.models]
        if len(repos) != len(set(repos)):
            raise ValueError("models: hf_repo listed twice")
        family_publisher: dict[str, str] = {}
        for model in self.models:
            # `kind` has a default (`"text"`, the shape every schema-1 row has always had), so a
            # missing value would otherwise pass silently; `model_fields_set` tells an explicit
            # `"text"` apart from the default.
            if "kind" not in model.model_fields_set:
                raise ValueError(f"{model.hf_repo}: kind is required")
            if model.family is None or model.publisher is None or model.added is None:
                raise ValueError(f"{model.hf_repo}: family, publisher and added are required")
            if model.publisher not in accounts:
                raise ValueError(f"{model.hf_repo}: publisher {model.publisher!r} is not in publishers")
            if model.hf_repo.split("/", 1)[0] != model.publisher:
                raise ValueError(f"{model.hf_repo}: publisher {model.publisher!r} does not own hf_repo")
            owner = family_publisher.setdefault(model.family, model.publisher)
            if owner != model.publisher:
                raise ValueError(f"family {model.family}: rows of two different publishers")
            _check_successor_evidence(model)
            _check_gguf_owners(model, packagers)
        return self

    def to_catalog(self) -> Catalog:
        order: list[str] = []
        grouped: dict[str, dict] = {}
        for model in self.models:
            group = grouped.setdefault(model.family, {"name": model.family, "publisher": model.publisher, "models": []})
            if model.family not in order:
                order.append(model.family)
            group["models"].append(model.model_dump(mode="json"))
        data = {
            "schema_version": self.schema_version,
            "families": [grouped[key] for key in order],
            "packagers": list(self.packagers),
            "publisher_info": [publisher.model_dump(mode="json") for publisher in self.publishers],
            "data_license": self.data_license,
            "attribution": self.attribution,
            "published": self.published.isoformat(),
        }
        return Catalog.model_validate(data)


def build_catalog_v2(text: str, path: Path) -> Catalog:
    """Parse `text` (already read and size-capped by `load_catalog`) as a schema-2 document.

    Every failure -- malformed JSON, a body too deeply nested for Python's recursion limit, a
    schema-version outside range, or a `CatalogDocumentV2`/`to_catalog` validation error -- ends
    as `CatalogError`, the one exception `load_catalog`'s callers handle (`SchemaVersionError` is
    still raised on its own, same as the TOML path, so callers can map it to its own exit code).
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise CatalogError(f"{path}: cannot read catalog: {exc}") from exc
    check_version_field(data, CATALOG_JSON_SCHEMA_RANGE, path)
    try:
        document = CatalogDocumentV2.model_validate(data)
        return document.to_catalog()
    except (ValidationError, RecursionError) as exc:
        raise CatalogError(f"{path}: invalid catalog: {exc}") from exc
