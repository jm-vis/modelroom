"""Shapes the guided mode shows: a search hit, a computed requirement, a plain-language note.

Only the shapes live here; the search, the reverse calculation and the note texts are later
work packages (CONTRACTS.md, "SearchHit", "Requirement", "Note").
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .catalog import Age
from .contracts import check_aware_utc, validate_hf_repo, validate_ollama_name
from .fit import FIXED_OVERHEAD_GIB, WEIGHTS_OVERHEAD_RATIO
from .measurements import Scenario
from .relation import RelationStatus

UNKNOWN = "unknown"


class SearchHit(BaseModel):
    """One Hugging Face repository from the search, as the selection list shows it.

    Every field but `repo` may be unknown: `None`, or the literal `unknown` for the labeled
    fields. `resolved` means exactly one publisher base model was proven (relation `quantized`,
    publisher per catalog); an unresolved hit carries its reason and gets no fit and no Ollama
    name. `repo_created_at` is shown as "repo created", never as the model's release date;
    `downloads` is shown as it is and is no rank.
    """

    model_config = ConfigDict(extra="forbid")

    repo: str
    publisher_status: Literal["publisher", "listed packager", "other", "unknown"] = UNKNOWN
    base_model: list[str] | None = None
    base_model_relation: str = UNKNOWN
    resolved: bool = False
    unresolved_reason: RelationStatus | Literal["publisher_unknown"] | None = None
    resolved_base_model: str | None = None
    repo_created_at: datetime | None = None
    # How often the Hub says this repository was downloaded: an indication of what many people
    # take, never a rank. The ranking rule alone orders the result (decided 2026-09-24).
    downloads: int | None = Field(default=None, ge=0)
    parameters_b: float | None = Field(default=None, gt=0)
    license: str = UNKNOWN
    age: Age = UNKNOWN
    successor: str | None = None
    # Where `age` comes from: `stated` by the publisher's repository or the catalog, or `computed`
    # from the version numbers of the family (`search_release`, decided 2026-09-25). `None` is an
    # `unknown` age, or a hit built without it, and reads as `stated`.
    release_basis: Literal["stated", "computed"] | None = None
    ollama: str | None = None

    @field_validator("repo")
    @classmethod
    def _check_repo(cls, value: str) -> str:
        return validate_hf_repo(value)

    @field_validator("repo_created_at")
    @classmethod
    def _check_created(cls, value: datetime | None) -> datetime | None:
        return check_aware_utc(value, "repo_created_at") if value is not None else None

    @model_validator(mode="after")
    def _check_coupling(self) -> "SearchHit":
        if self.resolved:
            if self.resolved_base_model is None or self.unresolved_reason is not None:
                raise ValueError("a resolved hit names resolved_base_model and no unresolved_reason")
            validate_hf_repo(self.resolved_base_model)
            if self.base_model != [self.resolved_base_model] or self.base_model_relation != "quantized":
                raise ValueError("a resolved hit declares exactly its resolved_base_model with relation 'quantized'")
        else:
            if self.unresolved_reason is None or self.resolved_base_model is not None or self.ollama is not None:
                raise ValueError("an unresolved hit has an unresolved_reason, no resolved_base_model, no ollama")
        if (self.age == "legacy") != (self.successor is not None):
            raise ValueError("successor is set exactly when age is 'legacy'")
        if self.release_basis == "computed" and self.age == UNKNOWN:
            raise ValueError("release_basis 'computed' needs a known age")
        if self.ollama is not None:
            validate_ollama_name(self.ollama)
        return self


class Requirement(BaseModel):
    """What one package needs for a scenario (computed, never measured): the reverse calculation.

    `kv_gib_total` is `kv_gib_per_request` x `scenario.requests`; `need_gib` is fit v1's formula
    with that KV total; `perfect_reachable` is false in CPU mode, where fit v1 caps at `good`.
    It carries no speed statement.
    """

    model_config = ConfigDict(extra="forbid")

    origin: Literal["computed"]
    package_identity: tuple[str, str, str]
    scenario: Scenario
    mode: Literal["gpu", "cpu_gpu", "cpu"]
    weights_gib: float = Field(ge=0)
    kv_gib_per_request: float = Field(ge=0)
    kv_gib_total: float = Field(ge=0)
    reserve_gib: float = Field(ge=0)
    need_gib: float = Field(ge=0)
    perfect_reachable: bool

    @model_validator(mode="after")
    def _check(self) -> "Requirement":
        if not math.isclose(self.kv_gib_total, self.kv_gib_per_request * self.scenario.requests, rel_tol=1e-9):
            raise ValueError("kv_gib_total must be kv_gib_per_request x scenario.requests")
        expected_need = self.weights_gib * WEIGHTS_OVERHEAD_RATIO + self.kv_gib_total + FIXED_OVERHEAD_GIB
        if not math.isclose(self.need_gib, expected_need, rel_tol=1e-9):
            raise ValueError(f"need_gib must be fit v1's formula with kv_gib_total ({expected_need})")
        if self.mode == "cpu" and self.perfect_reachable:
            raise ValueError("'perfect' is not reachable in cpu mode")
        return self


class Note(BaseModel):
    """One plain-language note next to a package, a machine, a measurement or a ranking.

    Built only from facts: `facts` names the fields it was derived from, so a reader can check
    it. `origin` is the provenance of those facts (measured, entered or computed).
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    subject: Literal["package", "machine", "measurement", "ranking"]
    origin: Literal["measured", "entered", "computed"]
    text: str = Field(min_length=1, max_length=240)
    facts: list[str] = Field(min_length=1)
