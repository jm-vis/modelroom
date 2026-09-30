"""The shipped catalog (`catalog.json`): publishers, current models, successors, Ollama names.

The search uses it to tell a publisher from a packager, to call a model `latest` or `legacy`
when the repo itself says nothing (`new_version`), and to know an Ollama name. Every model row
names its evidence; nothing in it is inferred (CONTRACTS.md, "Catalog rules").

Two file formats build the same in-memory `Catalog`, told apart by their suffix: `.toml` is
schema 1 (legacy, nested by family), `.json` is schema 2 (flat, the publicly published form;
every row carries its own `family` and `publisher`). `load_catalog` reads either into the same
`Catalog`/`CatalogFamily`/`CatalogModel` shape, so every other reader of this package keeps
working against one search API regardless of which schema shipped it: `model_for`,
`is_publisher`, `age_of`, `families`.
"""

from __future__ import annotations

import json
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contracts import SchemaVersionError, check_schema_version, validate_hf_repo, validate_ollama_pair
from .contracts import _OLLAMA_TAG_RE as OLLAMA_TAG_RE  # the same tag rule `Package.aliases` uses

CATALOG_FILE = Path(__file__).with_name("catalog.json")
CATALOG_REJECTED_FILE = Path(__file__).with_name("catalog-rejected.json")
# Half-open, like every schema range here; each format keeps its own narrower range too, so a
# `.toml` claiming schema 2 is still refused (CATALOG_SCHEMA_RANGE: a direct model_validate net).
CATALOG_SCHEMA_RANGE: tuple[int, int] = (1, 3)
CATALOG_TOML_SCHEMA_RANGE: tuple[int, int] = (1, 2)
CATALOG_JSON_SCHEMA_RANGE: tuple[int, int] = (2, 3)

# Abuse limits on the schema-2 file (a publicly published file is adversarial input).
CATALOG_MAX_BYTES, CATALOG_MAX_LINES = 1024 * 1024, 5000
CATALOG_MAX_PUBLISHERS, CATALOG_MAX_PACKAGERS = 500, 50
CATALOG_MAX_GGUF_REPOS, CATALOG_MAX_TEXT_LEN = 50, 300

CATALOG_DATA_LICENSE = "CC-BY-4.0"
CATALOG_ATTRIBUTION = "modelroom catalog, jm-vis, CC BY 4.0, https://huggingface.co/datasets/jm-vis/modelroom-catalog"

Age = Literal["latest", "legacy", "unknown"]
Kind = Literal["text", "vision", "embedding", "speech-recognition", "speech-synthesis", "audio", "image-generation"]
_HUB = "https://huggingface.co/"
# `urlsplit(...).hostname` is lowercase and without the port.
_HUB_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})

# `pipeline_tag` on the Hub -> `kind` in this schema, the eleven values a repository without a
# `pipeline_tag`, or with one outside this table, never becomes a row for. A future addition is a
# schema change with a test of its own, never a silent new value.
PIPELINE_TAG_TO_KIND: dict[str, Kind] = {
    "text-generation": "text", "image-text-to-text": "vision",
    "feature-extraction": "embedding", "sentence-similarity": "embedding",
    "automatic-speech-recognition": "speech-recognition", "text-to-speech": "speech-synthesis",
    "text-to-audio": "audio", "audio-to-audio": "audio", "audio-text-to-text": "audio",
    "text-to-image": "image-generation", "image-to-image": "image-generation",
}
# The two kinds a page of a search with no word shows: a positive list, so a future kind never
# appears there unannounced.
DISPLAYED_KINDS: frozenset[Kind] = frozenset({"text", "vision"})

_OLLAMA_BASE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# Same rule as `config.py`'s `_FAMILY_NAME_RE`, kept as its own literal (neither `config.py` nor
# `contracts.py` is touched by this work package).
_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
# One half of `contracts.py`'s `_HF_REPO_RE`: a publisher or packager account name on its own.
_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9][\w.-]*$")
_REGION_RE = re.compile(r"^(?:EU|[A-Z]{2})$")


class CatalogError(Exception):
    """The catalog file cannot be read or does not validate."""


def _check_https_page(value: str | None, field_name: str) -> str | None:
    """Shared shape check for `latest_source` and `successor_source`: an https page with a host."""
    if value is not None:
        parts = urlsplit(value)
        if parts.scheme != "https" or not parts.hostname:
            raise ValueError(f"{field_name} must be an https page with a host: {value!r}")
    return value


def _check_printable_note(value: str | None) -> str | None:
    if value is not None:
        if len(value) > CATALOG_MAX_TEXT_LEN:
            raise ValueError(f"note exceeds {CATALOG_MAX_TEXT_LEN} characters")
        if not value.isascii() or not value.isprintable():
            raise ValueError(f"note must be printable ASCII without control characters: {value!r}")
    return value


class CatalogModel(BaseModel):
    """One publisher base model: current (`latest`), superseded (`successor`), or neither.

    `latest` is a statement about this one model, never derived from a line or a number: it
    carries the publisher page or collection that says so (`latest_source`) and the day the
    maintainer checked it (`latest_checked`). Without them the model's age is `unknown`.

    The fields below `source` are schema 2 only: optional here so a schema-1 row stays exactly
    what it always was, but required in practice of a schema-2 *document* (`CatalogDocumentV2`
    checks `family`, `publisher` and `added` are set, and the ownership rule of
    `successor_source`). A schema-1 row keeps `kind` at its default `text`; every other new
    field stays unset.
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

    # --- schema 2 (AP9-KA) --------------------------------------------------------------------
    kind: Kind = "text"
    family: str | None = None
    publisher: str | None = None
    added: date | None = None
    parameters_b: float | None = None
    license: str | None = None
    hub_created: date | None = None
    successor_source: str | None = None
    successor_checked: date | None = None
    gguf_repos: list[str] = Field(default_factory=list)
    note: str | None = None

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

    @field_validator("latest_source", "successor_source")
    @classmethod
    def _check_evidence_page(cls, value: str | None, info) -> str | None:
        return _check_https_page(value, info.field_name)

    @field_validator("ollama_base")
    @classmethod
    def _check_ollama_base(cls, value: str | None) -> str | None:
        if value is not None and not _OLLAMA_BASE_RE.fullmatch(value):
            raise ValueError(f"ollama_base must be a library name, no '/': {value!r}")
        return value

    @field_validator("ollama_tag")
    @classmethod
    def _check_ollama_tag(cls, value: str | None) -> str | None:
        if value is not None and not OLLAMA_TAG_RE.fullmatch(value):
            raise ValueError(f"ollama_tag does not look like a tag: {value!r}")
        return value

    @field_validator("family")
    @classmethod
    def _check_family(cls, value: str | None) -> str | None:
        if value is not None and not _FAMILY_RE.fullmatch(value):
            raise ValueError(f"family must match '[a-z0-9][a-z0-9.-]*$': {value!r}")
        return value

    @field_validator("parameters_b")
    @classmethod
    def _check_parameters_b(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError(f"parameters_b must be >= 0: {value!r}")
        return value

    @field_validator("gguf_repos")
    @classmethod
    def _check_gguf_repos(cls, value: list[str]) -> list[str]:
        if len(value) > CATALOG_MAX_GGUF_REPOS:
            raise ValueError(f"gguf_repos exceeds {CATALOG_MAX_GGUF_REPOS} entries")
        return [validate_hf_repo(repo) for repo in value]

    @field_validator("note")
    @classmethod
    def _check_note(cls, value: str | None) -> str | None:
        return _check_printable_note(value)

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


class PublisherInfo(BaseModel):
    """One account named in a schema-2 catalog's `publishers` list (CONTRACTS.md, "Catalog rules")."""

    model_config = ConfigDict(extra="forbid")

    account: str
    name: str = Field(min_length=1, max_length=CATALOG_MAX_TEXT_LEN)
    region: str | None = None

    @field_validator("account")
    @classmethod
    def _check_account(cls, value: str) -> str:
        if not _ACCOUNT_RE.fullmatch(value):
            raise ValueError(f"account does not look like a Hub account: {value!r}")
        return value

    @field_validator("region")
    @classmethod
    def _check_region(cls, value: str | None) -> str | None:
        if value is not None and not _REGION_RE.fullmatch(value):
            raise ValueError(f"region must be 'EU' or an ISO-2 country code: {value!r}")
        return value


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    families: list[CatalogFamily] = Field(default_factory=list)
    # Schema 2 only (AP9-KA): empty/`None` for a catalog read from `.toml`.
    packagers: list[str] = Field(default_factory=list)
    publisher_info: list[PublisherInfo] = Field(default_factory=list)
    data_license: str | None = None
    attribution: str | None = None
    published: date | None = None

    @model_validator(mode="after")
    def _check(self) -> "Catalog":
        low, high = CATALOG_SCHEMA_RANGE
        if not (low <= self.schema_version < high):
            raise ValueError(f"schema_version must be within [{low}, {high}), got {self.schema_version}")
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


def _check_successor_source_ownership(model: CatalogModel) -> None:
    """Schema 2's own rule: set exactly when `successor` is, and then one of three Hub pages --
    the successor's own card, this model's own card (the `new_version` case), or a collection of
    the model's publisher -- or a page off the Hub, taken as the maintainer confirmed it."""
    has_evidence = model.successor_source is not None and model.successor_checked is not None
    if (model.successor is not None) != has_evidence:
        raise ValueError(f"{model.hf_repo}: successor_source and successor_checked belong to a successor")
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
            if not _ACCOUNT_RE.fullmatch(account):
                raise ValueError(f"packagers: not an account name: {account!r}")
        if len(value) != len(set(value)):
            raise ValueError("packagers: an account is listed twice")
        return value

    @model_validator(mode="after")
    def _check(self) -> "CatalogDocumentV2":
        if self.schema_version != 2:
            raise ValueError(f"schema_version must be 2, got {self.schema_version}")
        accounts = {publisher.account for publisher in self.publishers}
        repos = [model.hf_repo for model in self.models]
        if len(repos) != len(set(repos)):
            raise ValueError("models: hf_repo listed twice")
        family_publisher: dict[str, str] = {}
        for model in self.models:
            if model.family is None or model.publisher is None or model.added is None:
                raise ValueError(f"{model.hf_repo}: family, publisher and added are required")
            if model.publisher not in accounts:
                raise ValueError(f"{model.hf_repo}: publisher {model.publisher!r} is not in publishers")
            if model.hf_repo.split("/", 1)[0] != model.publisher:
                raise ValueError(f"{model.hf_repo}: publisher {model.publisher!r} does not own hf_repo")
            owner = family_publisher.setdefault(model.family, model.publisher)
            if owner != model.publisher:
                raise ValueError(f"family {model.family}: rows of two different publishers")
            _check_successor_source_ownership(model)
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


def load_catalog(path: Path = CATALOG_FILE) -> Catalog:
    """Read and validate a catalog file; the default is the one shipped with the package.

    The format follows the suffix: `.toml` is schema 1 (legacy, nested by family), `.json` is
    schema 2 (flat, the publicly published form). Both build the same `Catalog`.
    """
    if path.suffix == ".json":
        return _load_catalog_v2(path)
    return _load_catalog_v1(path)


def _check_version_field(data: object, accepted: tuple[int, int], path: Path) -> None:
    version = data.get("schema_version") if isinstance(data, dict) else None
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaVersionError(f"{path}: schema_version is missing or not an integer: {version!r}")
    check_schema_version(version, accepted, str(path))


def _load_catalog_v1(path: Path) -> Catalog:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CatalogError(f"{path}: cannot read catalog: {exc}") from exc
    _check_version_field(data, CATALOG_TOML_SCHEMA_RANGE, path)
    try:
        return Catalog.model_validate(data)
    except ValidationError as exc:
        raise CatalogError(f"{path}: invalid catalog: {exc}") from exc


def _load_catalog_v2(path: Path) -> Catalog:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CatalogError(f"{path}: cannot read catalog: {exc}") from exc
    if len(raw) > CATALOG_MAX_BYTES:
        raise CatalogError(f"{path}: catalog file exceeds {CATALOG_MAX_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogError(f"{path}: cannot read catalog: {exc}") from exc
    if text.count("\n") + 1 > CATALOG_MAX_LINES:
        raise CatalogError(f"{path}: catalog file exceeds {CATALOG_MAX_LINES} lines")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{path}: cannot read catalog: {exc}") from exc
    _check_version_field(data, CATALOG_JSON_SCHEMA_RANGE, path)
    try:
        document = CatalogDocumentV2.model_validate(data)
    except ValidationError as exc:
        raise CatalogError(f"{path}: invalid catalog: {exc}") from exc
    return document.to_catalog()
