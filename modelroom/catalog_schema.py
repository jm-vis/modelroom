"""The schema-2, schema-3 and schema-4 catalog documents (`catalog.json`): the flat, publicly published shape.

A module of its own so both stay under the package's size guard. `catalog.py` owns the shared `Catalog`/`CatalogFamily`/`CatalogModel`
shape every reader works against; this module owns the rules that apply to the **document** a
schema-2 file carries before it is grouped into that shape -- the ones `CatalogModel` alone
cannot check, because they need sibling fields (`publisher` against the document's `publishers`
list) or the whole list (`models` count, duplicate `hf_repo`, one publisher per `family`).

Schema 3 is schema 2 plus the two findings of a row (`gguf_checks`, `setup_note`). A schema-2
document is exactly the shape release 1.0.0 reads and carries neither key, not even at its default;
`project_to_schema2` is the one way from a schema-3 document without findings to the file of the old
channel (CONTRACTS.md, "Published catalog"). Schema 4 is schema 3 plus the `vllm_args` of a setup note;
`project_to_schema3` is the one way back from it to the file release 1.1.0 reads.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_serializer, model_validator

from .catalog import (
    _HUB_HOSTS,
    ACCOUNT_RE,
    CATALOG_MAX_BYTES,
    CATALOG_MAX_GGUF_REPOS,  # re-exported next to the other document limits
    CATALOG_MAX_TEXT_LEN,
    CATALOG_V2_SCHEMA_RANGE,
    CATALOG_V3_MAX_BYTES,
    CATALOG_V3_SCHEMA_RANGE,
    CATALOG_V4_SCHEMA_RANGE,
    CatalogError,
    PIPELINE_TAG_TO_KIND,
    CatalogModel,
    Catalog,
    PublisherInfo,
    CATALOG_JSON_SCHEMA_RANGE,
    check_size,
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

# The row fields of schema 3 and the value each has when nothing is known: the whole of what separates a
# row of schema 3 from a row of schema 2.
SCHEMA3_ROW_DEFAULTS: dict[str, object] = {"gguf_checks": [], "setup_note": None}


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


def _check_kind_evidence(model: CatalogModel) -> None:
    """`kind` is the view of `hub_pipeline_tag` (CONTRACTS.md, "Catalog rules"): a task of the
    table gives exactly that kind and no `kind_source`; any other task, or none, needs the
    `kind_source` that decided `kind`, and a `decision` needs a `note` that says on what."""
    from_task = PIPELINE_TAG_TO_KIND.get(model.hub_pipeline_tag)
    if from_task is not None:
        if model.kind != from_task:
            raise ValueError(
                f"{model.hf_repo}: kind {model.kind!r} contradicts hub_pipeline_tag "
                f"{model.hub_pipeline_tag!r}, which gives {from_task!r}"
            )
        if model.kind_source is not None:
            raise ValueError(
                f"{model.hf_repo}: kind_source {model.kind_source!r} must be null, hub_pipeline_tag "
                f"{model.hub_pipeline_tag!r} is in the kind table and already gives kind {model.kind!r}"
            )
        return
    if model.kind_source is None:
        raise ValueError(
            f"{model.hf_repo}: kind {model.kind!r} needs a kind_source, hub_pipeline_tag "
            f"{model.hub_pipeline_tag!r} is not in the kind table"
        )
    if model.kind_source == "decision" and model.note is None:
        raise ValueError(f"{model.hf_repo}: kind_source 'decision' needs a note that says why kind {model.kind!r} stands")


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
    `family` into the same shape schema 1 builds. A row carries none of the keys of schema 3
    (`SCHEMA3_ROW_DEFAULTS`), not even at its default: release 1.0.0 refuses an unknown key in a row.
    """

    model_config = ConfigDict(extra="forbid")
    SCHEMA: ClassVar[int] = 2

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

    @model_serializer(mode="wrap")
    def _as_its_schema(self, handler):
        """A schema-2 document is written as schema 2: the two keys of schema 3 are not written (they hold their
        default, `_check_findings` saw to that), so what this release writes is what release 1.0.0 reads."""
        data = handler(self)
        if self.SCHEMA == 2:
            data["models"] = [{k: v for k, v in row.items() if k not in SCHEMA3_ROW_DEFAULTS} for row in data["models"]]
        return data

    def _check_findings(self, model: CatalogModel) -> None:
        """Schema 2: a row has none of the findings' keys."""
        present = sorted(model.model_fields_set & SCHEMA3_ROW_DEFAULTS.keys())
        if present:
            raise ValueError(f"schema 2 has no {', '.join(present)} ({model.hf_repo}); that needs schema 3")

    @model_validator(mode="after")
    def _check(self) -> "CatalogDocumentV2":
        if self.schema_version != self.SCHEMA:
            raise ValueError(f"schema_version must be {self.SCHEMA}, got {self.schema_version}")
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
            _check_kind_evidence(model)
            _check_gguf_owners(model, packagers)
            self._check_findings(model)
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


class CatalogDocumentV3(CatalogDocumentV2):
    """The schema-3 catalog file: schema 2 plus `gguf_checks` and `setup_note` in a row (AP9-KO).

    Their rules sit on the row (`CatalogModel`); the document adds none but one: a note carries no `vllm_args`, which
    release 1.1.0 would refuse as an unknown key. A row without the findings reads as unknown.
    """

    SCHEMA: ClassVar[int] = 3

    def _check_findings(self, model: CatalogModel) -> None:
        if model.setup_note is not None and "vllm_args" in model.setup_note.model_fields_set:  # the key, whatever it holds: 1.1.0 refuses it
            raise ValueError(f"schema 3 has no vllm_args ({model.hf_repo}); that needs schema 4")


class CatalogDocumentV4(CatalogDocumentV3):
    """The schema-4 catalog file: schema 3 plus `vllm_args` in a setup note (V7), the switches of the card's start block.

    The field sits on `SetupNote` and is not written at its default, so a document without switches differs from
    schema 3 in its `schema_version` alone.
    """

    SCHEMA: ClassVar[int] = 4

    def _check_findings(self, model: CatalogModel) -> None:
        return None


def _build(
    text: str, path: Path, accepted: tuple[int, int], document_for
) -> Catalog:
    """The common reading: JSON, the version gate, the document class `document_for(version)`, the size rule of it."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise CatalogError(f"{path}: cannot read catalog: {exc}") from exc
    check_version_field(data, accepted, path)
    document_class = document_for(data["schema_version"])
    check_size(text, CATALOG_V3_MAX_BYTES if document_class in (CatalogDocumentV3, CatalogDocumentV4) else CATALOG_MAX_BYTES, path)
    try:
        return document_class.model_validate(data).to_catalog()
    except (ValidationError, RecursionError) as exc:
        raise CatalogError(f"{path}: invalid catalog: {exc}") from exc


def build_catalog_v2(text: str, path: Path) -> Catalog:
    """Parse `text` (already read and size-capped by `load_catalog`) as a schema-2 document.

    Every failure -- malformed JSON, a body too deeply nested for Python's recursion limit, a
    schema-version outside range, or a `CatalogDocumentV2`/`to_catalog` validation error -- ends
    as `CatalogError`, the one exception `load_catalog`'s callers handle (`SchemaVersionError` is
    still raised on its own, same as the TOML path, so callers can map it to its own exit code).
    This is what the old channel holds and what release 1.0.0 reads: schema 2 and nothing else.
    """
    return _build(text, path, CATALOG_V2_SCHEMA_RANGE, lambda _version: CatalogDocumentV2)


def build_catalog_v3(text: str, path: Path) -> Catalog:
    """The same for a schema-3 document, the file of the new channel."""
    return _build(text, path, CATALOG_V3_SCHEMA_RANGE, lambda _version: CatalogDocumentV3)


def build_catalog_v4(text: str, path: Path) -> Catalog:
    """The same for a schema-4 document, the file of the channel release 1.2.0 asks."""
    return _build(text, path, CATALOG_V4_SCHEMA_RANGE, lambda _version: CatalogDocumentV4)


_DOCUMENTS = {2: CatalogDocumentV2, 3: CatalogDocumentV3, 4: CatalogDocumentV4}


def build_catalog(text: str, path: Path) -> Catalog:
    """The same for any of them: the content says schema 2, 3 or 4 (what `load_catalog` reads)."""
    return _build(text, path, CATALOG_JSON_SCHEMA_RANGE, _DOCUMENTS.__getitem__)


def project_to_schema3(document: dict) -> dict:
    """A copy of a schema-4 `document` (as read from JSON) in the exact shape of schema 3: what release 1.1.0 reads.

    The `vllm_args` of a setup note are dropped. They are an addition to a note that stands without them (it still
    says that the card holds instructions, and where), so nothing a 1.1.0 reader could act on is lost, and nothing
    but `vllm_args` and `schema_version` differs. The file of the channel of release 1.1.0 is made through this and
    nowhere else.
    """
    if not isinstance(document, dict) or document.get("schema_version") not in (3, 4):
        raise ValueError("only a document of schema 3 or 4 is projected")
    result = copy.deepcopy(document)
    result["schema_version"] = 3
    for row in result.get("models", []):
        note = row.get("setup_note")
        if isinstance(note, dict):
            note.pop("vllm_args", None)
    return result


def project_to_schema2(document: dict) -> dict:
    """A copy of a schema-3 `document` (as read from JSON) in the exact shape of schema 2.

    The two keys of schema 3 are dropped from every row -- they hold their default, or this raises
    `ValueError`: a finding cannot be carried by a file release 1.0.0 reads, and is not dropped silently.
    The file of the old channel is made through this and nowhere else.
    """
    if not isinstance(document, dict) or document.get("schema_version") not in (2, 3):
        raise ValueError("only a document of schema 2 or 3 is projected")
    result = copy.deepcopy(document)
    result["schema_version"] = 2
    for row in result.get("models", []):
        for name, default in SCHEMA3_ROW_DEFAULTS.items():
            if name in row and row.pop(name) != default:
                raise ValueError(f"{row.get('hf_repo')}: {name} holds a finding, which schema 2 cannot carry")
    return result
