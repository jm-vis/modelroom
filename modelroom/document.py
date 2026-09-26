"""The render document: one object per run, from which terminal, Markdown and JSON are written.

`modelroom render` (and the guided mode's last step) builds exactly one `RenderDocument` and
hands it to three writers in `modelroom/render.py`. Nothing is computed twice and no output can
say something another one does not: a number that is not in this object is in no output either.

Every number carries where it came from. `weights_gib`, `fit` and everything derived from them
are `computed`; `speed_tps` is `measured` and exists only in measured group 0; a profile's own
readings carry their source in the profile (`ram_physical_source`, `vram_source`, `origin`).
See CONTRACTS.md, "Render (schema 2)".
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import MAX_USERS, RequestsOrigin, check_requests_origin
from .contracts import Area, Fit, Rating, check_aware_utc, validate_hf_repo
from .guided_contracts import Note
from .measurements import Scenario
from .profile import HardwareProfile

# Schema 2 (decided 2026-09-25): every fit carries `requests`, the document carries `users` and
# `requests_origin`.
DOCUMENT_SCHEMA_VERSION = 2

# What a machine's block says about itself: a profile (schema 2 or later) that the fit could use
# (`ranked`), a schema-1 file that has to be migrated and measured again (`legacy`), or no
# profile at all yet (`no_profile`). Only `ranked` ever carries entries.
MachineStatus = Literal["ranked", "legacy", "no_profile"]

_RANKABLE_CLASSES = ("perfect", "good", "marginal")


class PackageLine(BaseModel):
    """The package facts every row shows, whether the package was ranked or set aside.

    `quantization` and `package_context` are `unknown`/`None` when the package declares none --
    the renderer never substitutes a plausible value (CONTRACTS.md, "Language standard").
    """

    model_config = ConfigDict(extra="forbid")

    package_identity: tuple[str, str, str]
    base_model_hf_repo: str
    packager: str
    quantization: str = Field(min_length=1)
    format: str = Field(min_length=1)
    weights_gib: float = Field(ge=0)
    package_context: int | None = Field(default=None, gt=0)
    provenance: str
    note: Note

    @model_validator(mode="after")
    def _check_repo(self) -> "PackageLine":
        validate_hf_repo(self.base_model_hf_repo)
        return self


class RankedEntry(PackageLine):
    """One row of one machine's ranking: its place, its computed fit, its measured speed.

    `measurement_group` 0 means a measurement counted for this ranking (protocol v1, valid,
    comparable, same context) -- then and only then `measurement_id` and `speed_tps` are set.
    """

    rank: int = Field(ge=1)
    fit: Fit
    measurement_group: Literal[0, 1]
    measurement_id: str | None = None
    speed_tps: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_entry(self) -> "RankedEntry":
        if self.fit.fit_class not in _RANKABLE_CLASSES:
            raise ValueError(f"a ranked entry has a fit class of {_RANKABLE_CLASSES}, got {self.fit.fit_class!r}")
        measured = self.measurement_id is not None
        if measured != (self.measurement_group == 0) or measured != (self.speed_tps is not None):
            raise ValueError("measurement_group 0 carries measurement_id and speed_tps, group 1 carries neither")
        return self


class SetAsideEntry(PackageLine):
    """One package outside the ranking -- not covered, or too tight -- with the reason shown."""

    fit: Fit
    reason: str = Field(min_length=1)


class MachineRanking(BaseModel):
    """One machine's block: which profile it was computed from, and the three lists.

    `label` is what the document shows for the machine's profile: the profile's `display_name`
    for `ranked`, the file name for `legacy`, the machine name for `no_profile`. A machine
    that is not `ranked` carries its `reason` and no entries at all -- there is no fit to show.
    """

    model_config = ConfigDict(extra="forbid")

    machine: str
    status: MachineStatus
    label: str = Field(min_length=1)
    profile: HardwareProfile | None = None
    reserve_ram_gib: float = Field(ge=0)
    reserve_vram_gib: float = Field(ge=0)
    reason: str | None = None
    ranked: list[RankedEntry] = Field(default_factory=list)
    # How many packages the rule ranked in all; `ranked` shows the first `ranking.TOP_LIMIT`
    # of them, so a reader can see that the list is a top ten and not the whole result.
    ranked_total: int = Field(default=0, ge=0)
    not_covered: list[SetAsideEntry] = Field(default_factory=list)
    too_tight: list[SetAsideEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_status(self) -> "MachineRanking":
        if self.status == "ranked":
            if self.profile is None or self.reason is not None:
                raise ValueError("status 'ranked' carries a profile and no reason")
            return self
        if self.profile is not None or not self.reason:
            raise ValueError(f"status {self.status!r} carries a reason and no profile")
        if self.ranked or self.not_covered or self.too_tight or self.ranked_total:
            raise ValueError(f"status {self.status!r} carries no entries -- there is no fit to show")
        return self

    @model_validator(mode="after")
    def _check_ranks(self) -> "MachineRanking":
        ranks = [entry.rank for entry in self.ranked]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"rank must count 1, 2, 3 ... in order, got {ranks}")
        if self.ranked_total < len(ranks):
            raise ValueError(f"ranked_total {self.ranked_total} is below the {len(ranks)} entries shown")
        return self


class RenderDocument(BaseModel):
    """Everything one render says: the snapshot it read, the scenario, and one block per machine.

    `ratings` holds the market rating of a base model the `RatingSource` answered for; a base
    model that is not a key has no rating (`-` in the Stars column). `rating_unavailable` is the
    source's own message when it failed; then no rating is shown at all and the run still ends 0.

    `users` and `requests_origin` say where `scenario.requests` came from, copied from the
    configuration's `[guided]` when the rendered scenario is that configuration's; both `None`
    when a caller passed a scenario of other requests, so the document never names a wrong
    origin. Every ranked and too-tight fit was computed for `scenario.requests`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    snapshot_run_at: datetime
    rendered_at: datetime
    base_model_count: int = Field(ge=0)
    package_count: int = Field(ge=0)
    scenario: Scenario
    ranking_rule: str = Field(min_length=1)
    rating_unavailable: str | None = None
    ratings: dict[str, Rating] = Field(default_factory=dict)
    areas: list[Area] = Field(default_factory=list)
    machines: list[MachineRanking] = Field(default_factory=list)
    users: int | None = Field(default=None, ge=1, le=MAX_USERS)
    requests_origin: RequestsOrigin | None = None

    @model_validator(mode="after")
    def _check_requests(self) -> "RenderDocument":
        if self.requests_origin is None:
            if self.users is not None:
                raise ValueError("users is only set together with requests_origin")
        else:
            check_requests_origin(self.requests_origin, self.scenario.requests, self.users)
        for block in self.machines:
            for entry in [*block.ranked, *block.too_tight]:
                if entry.fit.requests != self.scenario.requests:
                    raise ValueError(f"a fit for {entry.fit.requests} requests in a document of {self.scenario.requests}")
        return self

    @model_validator(mode="after")
    def _check_document(self) -> "RenderDocument":
        if self.schema_version != DOCUMENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {DOCUMENT_SCHEMA_VERSION}, got {self.schema_version}")
        check_aware_utc(self.snapshot_run_at, "snapshot_run_at")
        check_aware_utc(self.rendered_at, "rendered_at")
        names = [block.machine for block in self.machines]
        if len(names) != len(set(names)):
            raise ValueError("a machine has exactly one block in a document")
        if self.rating_unavailable is not None and self.ratings:
            raise ValueError("a failed rating source leaves no ratings behind")
        return self
