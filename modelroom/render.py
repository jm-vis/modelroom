"""`render`: one document object per run, then three writers that only format it.

`build_render_document(...) -> document.RenderDocument` is the pure builder: it takes an
already-loaded `Configuration`, `Snapshot`, one `MachineProfile` per configured machine, that
profile's measurement records, the ranking `Scenario` and `rendered_at`, computes every fit
(`fit.compute_fit_v2`) and every ranking (`ranking.rank_packages`), and returns the one object
the three writers read: `document_markdown`, `document_json`, `document_terminal`. No I/O
anywhere in this module -- `modelroom/cli.py::render_with_config` owns the lock, the
schema-version gates, reading the files and the atomic writes.

`machine_profile` is the other pure decision here: which profile a `[machines.<name>]` entry is
rendered from, given the profile files found in the results folder. See CONTRACTS.md,
"Render (schema 2)".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from .config import Configuration, MachineConfig
from .contracts import BaseModelSpec, Fit, Package, Rating, Snapshot
from .document import (
    DOCUMENT_SCHEMA_VERSION,
    MachineRanking,
    MachineStatus,
    RankedEntry,
    RenderDocument,
    SetAsideEntry,
)
from .fit import GIB, compute_fit_v2
from .guided_contracts import Note
from .importer import ProfileScan
from .measurements import MeasurementRecord, Scenario
from .profile import PROFILE_SCHEMA_VERSION, CrossCheck, HardwareProfile, LlmfitCrosscheck, fit_block_reason
from .quantization import package_identity_key
from .ranking import RANKING_RULE, RankedPackage, Ranking, SetAside, rank_packages

RatingSource = Callable[[str], Rating | None]

_HEADER_RE = re.compile(
    r"^<!-- modelroom render: snapshot_run_at=(?P<snapshot_run_at>\S+) rendered_at=(?P<rendered_at>\S+) -->$"
)
_ELIGIBLE_PROVENANCE = ("metadata_ok", "approved")
_WEIGHT_ROLES = ("weights", "weights_shard")
_UNKNOWN = "unknown"
_NOTE_LIMIT = 240

NO_PROFILE_REASON = "no hardware profile yet; measure this machine with `modelroom hardware`"


class RatingUnavailableError(Exception):
    """A `RatingSource` could not answer at all for the base model it was asked about."""


@dataclass(frozen=True)
class HeaderInfo:
    """The two timestamps in a rendered document's first line, as `parse_header_line` reads it."""

    snapshot_run_at: datetime
    rendered_at: datetime


def format_header_line(snapshot_run_at: datetime, rendered_at: datetime) -> str:
    """The fixed header comment `document_markdown` always writes as the document's first line."""
    return f"<!-- modelroom render: snapshot_run_at={snapshot_run_at.isoformat()} rendered_at={rendered_at.isoformat()} -->"


def _is_aware_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == timedelta(0)


def parse_header_line(text: str) -> HeaderInfo | None:
    """The header line's two timestamps, or `None` when `text` has no such first line.

    `None` covers an empty file, one whose first line does not match the fixed format (e.g. a
    document from before `render` wrote this header), and one whose `snapshot_run_at`/
    `rendered_at` parses but is not an aware UTC datetime (`format_header_line` never writes
    anything else, same rule as `contracts.check_aware_utc`; a hand-edited or otherwise
    corrupted header with a naive timestamp would otherwise make
    `cli.py::_refusal_against_existing_document`'s comparison against the new, always-aware
    snapshot's `run_at` raise `TypeError` instead of just refusing to compare). Either way, the
    caller (`cli.py`) treats it as nothing to compare against, never a reason to fail.
    """
    first_line = text.splitlines()[0] if text else ""
    match = _HEADER_RE.match(first_line)
    if match is None:
        return None
    try:
        snapshot_run_at = datetime.fromisoformat(match.group("snapshot_run_at"))
        rendered_at = datetime.fromisoformat(match.group("rendered_at"))
    except ValueError:
        return None
    if not (_is_aware_utc(snapshot_run_at) and _is_aware_utc(rendered_at)):
        return None
    return HeaderInfo(snapshot_run_at=snapshot_run_at, rendered_at=rendered_at)


# --- which profile one machine is rendered from -----------------------------------------------


@dataclass(frozen=True)
class MachineProfile:
    """What the results folder holds for one `[machines.<name>]`, as `machine_profile` decided.

    `ranked` carries the schema-2 profile the fit computes with; `legacy` and `no_profile`
    carry no profile at all and say why in `reason`. `label` is what the document shows: the
    profile's `display_name`, the schema-1 file's name, or the machine name.
    """

    status: MachineStatus
    label: str
    profile: HardwareProfile | None
    reason: str | None


def _legacy_fit_reason() -> str:
    """The reason the fit rule gives for *any* schema-1 profile -- it is always the same one.

    `fit_block_reason` decides it from `gpu_state`, and `profile.normalize_profile_v1` gives
    every schema-1 profile without unified memory the state `legacy_unknown`, so the text does
    not depend on the file. Taken from the rule itself rather than copied, so the two can never
    drift apart (`tests/test_render.py` checks it against a really normalized profile). The
    probe below is never written anywhere and never leaves this function.
    """
    absent = CrossCheck(status="absent")
    probe = HardwareProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id="0" * 16,
        display_name="legacy",
        os_fingerprint="none",
        os_fingerprint_source="legacy",
        origin="measured",
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ram_physical_gib=1.0,
        ram_physical_source="llmfit",
        ram_limit_gib=None,
        ram_limit_scope="none",
        vram_gib=0.0,
        vram_source="llmfit",
        gpu_state="legacy_unknown",
        gpu_name=None,
        llmfit_crosscheck=LlmfitCrosscheck(ram_physical=absent, vram=absent),
        llmfit_version=None,
    )
    reason = fit_block_reason(probe)
    assert reason is not None  # `legacy_unknown` is never a state the fit computes for
    return reason


LEGACY_FIT_REASON = _legacy_fit_reason()


def machine_profile(machine: str, machine_config: MachineConfig, scan: ProfileScan) -> MachineProfile:
    """Which profile `[machines.<machine>]` is rendered from, given the folder's profile files.

    `[machines.<name>].profile` names the schema-2 file; when it is set and readable, that is
    the profile. A machine without `profile` may still have a schema-1 file `<name>.json` from
    before `modelroom migrate` -- it is shown as `legacy` with the fit rule's own reason and
    nothing is persisted. Everything else is `no_profile` with the reason named.
    """
    wanted = machine_config.profile
    if wanted is not None:
        found = scan.profiles.get(wanted)
        if found is not None:
            return MachineProfile("ranked", found.display_name, found, None)
        return MachineProfile("no_profile", machine, None, _missing_profile_reason(wanted, scan))
    legacy = next((path for path in scan.legacy if path.stem == machine), None)
    if legacy is not None:
        return MachineProfile("legacy", legacy.name, None, LEGACY_FIT_REASON)
    return MachineProfile("no_profile", machine, None, NO_PROFILE_REASON)


def _missing_profile_reason(profile_id: str, scan: ProfileScan) -> str:
    """Why the configured `profile` is not the profile this machine is rendered from."""
    broken = next((reason for path, reason in scan.unreadable if path.stem == profile_id), None)
    if broken is not None:
        return f"the configured profile {profile_id} does not read: {broken}"
    if any(path.stem == profile_id for path in scan.legacy):
        return f"the configured profile {profile_id} is still a schema-1 file; {LEGACY_FIT_REASON}"
    return f"the configured profile {profile_id} is not in this results folder"


# --- eligibility and package facts -------------------------------------------------------------


def _eligible_packages(snapshot: Snapshot) -> list[Package]:
    """Every package a fit may be computed for: shown provenance, complete and active.

    The base model is not checked here: `Snapshot` itself refuses a package whose
    `base_model_hf_repo` is not among its `base_models`, so every package that reaches this
    point has an architecture the fit can read.
    """
    return [
        package
        for package in snapshot.packages
        if package.provenance in _ELIGIBLE_PROVENANCE and package.complete and package.active
    ]


def _packager_name(package: Package) -> str:
    if package.source == "ollama":
        return "ollama"
    return (package.repo or "").split("/", 1)[0]


def _weights_gib(package: Package) -> float:
    return sum(f.size_bytes for f in package.files if f.role in _WEIGHT_ROLES) / GIB


def _package_line(package: Package, note: Note) -> dict:
    """The fields `RankedEntry` and `SetAsideEntry` share, all of them about the package."""
    return {
        "package_identity": package_identity_key(package),
        "base_model_hf_repo": package.base_model_hf_repo,
        "packager": _packager_name(package),
        "quantization": package.quantization or _UNKNOWN,
        "format": package.format,
        "weights_gib": _weights_gib(package),
        "package_context": package.default_context,
        "provenance": package.provenance,
        "note": note,
    }


# --- the plain-language note, built from named facts only ---------------------------------------


def _clip(text: str) -> str:
    return text if len(text) <= _NOTE_LIMIT else text[: _NOTE_LIMIT - 1].rstrip() + "…"


def _measured_note(measurement: MeasurementRecord) -> Note:
    return Note(
        code="measured_here",
        subject="package",
        origin="measured",
        text=_clip(
            f"Measured on this machine: {measurement.tps_mean:.1f} tokens per second at a "
            f"context of {measurement.scenario.context_requested}."
        ),
        facts=["measurement.tps_mean", "measurement.scenario.context_requested"],
    )


_MODE_NOTES = {
    "gpu": (
        "fits_in_graphics_memory",
        "Fits into graphics memory: it needs about {need:.1f} GiB of the {pool:.1f} GiB left "
        "after the reserve.",
        ["fit.mode", "fit.need_gib", "fit.pool_gib"],
    ),
    "cpu_gpu": (
        "shared_between_memories",
        "Too large for graphics memory alone, so it runs in system memory with the graphics "
        "card helping; the best rating it can reach here is 'good'.",
        ["fit.mode", "fit.fit_class"],
    ),
    "cpu": (
        "cpu_caps_at_good",
        "Runs in system memory only, so the best possible rating on this machine is 'good'.",
        ["fit.mode", "fit.fit_class"],
    ),
}
# Unified memory computes in mode `gpu` as well, but it is no graphics memory of its own: the
# system and the model share it, and both reserves are taken from it (decided 2026-09-25).
_SHARED_MEMORY_NOTE = (
    "fits_in_shared_memory",
    "Fits into shared memory: it needs about {need:.1f} GiB of the {pool:.1f} GiB left after both "
    "reserves.",
    ["fit.mode", "fit.need_gib", "fit.pool_gib", "profile.gpu_state"],
)


# The one sentence a fit on the size basis adds to its note, whatever the pool: the number comes
# from the size of the package, not from an architecture this package's base model never made
# readable (decided 2026-09-24). The pool's own sentence stays in front of it -- in the acceptance
# of 2026-09-24 a row in system memory read as the equal of one in graphics memory without it.
_FROM_SIZE_SENTENCE = " From the size of the package, not its architecture."


def _computed_note(fit: Fit, shared_memory: bool = False) -> Note:
    """The row note of a computed fit; `shared_memory` names a `gpu` row of unified memory so."""
    shared = shared_memory and fit.mode == "gpu"
    code, template, facts = _SHARED_MEMORY_NOTE if shared else _MODE_NOTES[str(fit.mode)]
    text = template.format(need=fit.need_gib, pool=fit.pool_gib)
    if fit.basis == "size":
        text += _FROM_SIZE_SENTENCE
        facts = [*facts, "fit.basis"]
    return Note(code=code, subject="package", origin="computed", text=_clip(text), facts=facts)


def _set_aside_note(set_aside: SetAside, covered: bool, shared_memory: bool = False) -> Note:
    if not covered:
        return Note(
            code="not_covered",
            subject="package",
            origin="computed",
            text=_clip(f"Fit contract v1 cannot judge this package here: {set_aside.reason}."),
            facts=["fit.fit_class", "fit.reason"],
        )
    # Not "more than the machine has": fit v1 already calls a package `too_tight` above 98 % of
    # the pool, so the need can be below the pool and the note must not claim an overrun
    # (probe: need 7.125 GiB against a pool of 7.2 GiB is `too_tight`).
    fit = set_aside.fit
    # A package set aside on the size basis says on what basis: the number that put it there was
    # computed from its size, not from its architecture (second-model round, 2026-09-24).
    from_size = _FROM_SIZE_SENTENCE if fit.basis == "size" else ""
    return Note(
        code="too_tight",
        subject="package",
        origin="computed",
        text=_clip(
            f"Fit contract v1 does not count this as a fit: it needs about {fit.need_gib:.1f} GiB "
            f"of the {fit.pool_gib:.1f} GiB left after {'both reserves' if shared_memory else 'the reserve'}.{from_size}"
        ),
        facts=["fit.need_gib", "fit.pool_gib"] + (["fit.basis"] if fit.basis == "size" else []),
    )


# --- building the document ----------------------------------------------------------------------


def _row_note(ranked: RankedPackage, shared_memory: bool) -> Note:
    """What one ranked row says: its measurement, why a measurement did not count, or its fit.

    A measurement that counts but for `requests` alone is named in the row, so a reader does not
    look for a speed the ranking of more requests cannot use (`RankedPackage.measurement_note`).
    """
    if ranked.measurement is not None:
        return _measured_note(ranked.measurement)
    if ranked.measurement_note is not None:
        return Note(
            code="measured_with_one_request",
            subject="package",
            origin="computed",
            text=_clip(ranked.measurement_note),
            facts=["scenario.requests", "measurement_group"],
        )
    return _computed_note(ranked.fit, shared_memory)


def _ranked_entries(ranking: Ranking, shared_memory: bool = False) -> list[RankedEntry]:
    entries = []
    for ranked in ranking.top:
        note = _row_note(ranked, shared_memory)
        entries.append(
            RankedEntry(
                rank=ranked.rank,
                fit=ranked.fit,
                measurement_group=ranked.measurement_group,
                measurement_id=ranked.measurement.measurement_id if ranked.measurement is not None else None,
                speed_tps=ranked.measurement.tps_mean if ranked.measurement is not None else None,
                **_package_line(ranked.package, note),
            )
        )
    return entries


def _set_aside_entries(items: list[SetAside], covered: bool, shared_memory: bool = False) -> list[SetAsideEntry]:
    return [
        SetAsideEntry(
            fit=item.fit, reason=item.reason, **_package_line(item.package, _set_aside_note(item, covered, shared_memory))
        )
        for item in items
    ]


def _machine_block(
    machine: str,
    machine_config: MachineConfig,
    found: MachineProfile,
    packages: list[Package],
    base_model_by_repo: dict[str, BaseModelSpec],
    measurements: list[MeasurementRecord],
    scenario: Scenario,
) -> MachineRanking:
    """One machine's block: its three lists, or its reason when no fit can be computed at all."""
    reserves = {
        "reserve_ram_gib": machine_config.reserve_ram_gib,
        "reserve_vram_gib": machine_config.reserve_vram_gib,
    }
    if found.profile is None:
        return MachineRanking(
            machine=machine, status=found.status, label=found.label, reason=found.reason, **reserves
        )
    entries = [
        (package, compute_fit_v2(found.profile, package, base_model_by_repo[package.base_model_hf_repo], scenario, machine_config))
        for package in packages
    ]
    ranking = rank_packages(entries, measurements, scenario)
    shared = found.profile.gpu_state == "unified_memory"  # one memory, both reserves taken from it
    return MachineRanking(
        machine=machine,
        status="ranked",
        label=found.label,
        profile=found.profile,
        ranked=_ranked_entries(ranking, shared_memory=shared),
        ranked_total=len(ranking.ranked),
        not_covered=_set_aside_entries(ranking.not_covered, covered=False),
        too_tight=_set_aside_entries(ranking.too_tight, covered=True, shared_memory=shared),
        **reserves,
    )


def _repos_with_rows(blocks: list[MachineRanking]) -> list[str]:
    """Every base model that has at least one row anywhere, in the order the rows name it."""
    seen: list[str] = []
    for block in blocks:
        for entry in [*block.ranked, *block.not_covered, *block.too_tight]:
            if entry.base_model_hf_repo not in seen:
                seen.append(entry.base_model_hf_repo)
    return seen


def _collect_ratings(
    rating_source: RatingSource | None, base_model_repos: list[str]
) -> tuple[dict[str, Rating], str | None]:
    """Call `rating_source` once per base model; the first exception's message ends the column."""
    if rating_source is None:
        return {}, None
    ratings: dict[str, Rating] = {}
    for repo in base_model_repos:
        try:
            answer = rating_source(repo)
        except Exception as exc:  # noqa: BLE001 -- every exception is treated the same (brief)
            return {}, str(exc)
        if answer is not None:
            ratings[repo] = answer
    return ratings, None


def build_render_document(
    config: Configuration,
    snapshot: Snapshot,
    profiles: dict[str, MachineProfile],
    measurements: dict[str, list[MeasurementRecord]],
    scenario: Scenario,
    rendered_at: datetime,
    rating_source: RatingSource | None = None,
) -> RenderDocument:
    """The one object a render produces; pure, no I/O and no clock read.

    `profiles` maps every `config.machines` key to the `MachineProfile` the results folder
    holds for it (`machine_profile`); `measurements` maps a `profile_id` to that profile's
    measurement records. Every fit is computed against `scenario`, one context for the whole
    document (CONTRACTS.md, "Render (schema 2)").
    """
    base_model_by_repo = {bm.hf_repo: bm for bm in snapshot.base_models}
    packages = _eligible_packages(snapshot)
    blocks = [
        _machine_block(
            machine,
            machine_config,
            profiles[machine],
            packages,
            base_model_by_repo,
            measurements.get(getattr(profiles[machine].profile, "profile_id", ""), []),
            scenario,
        )
        for machine, machine_config in config.machines.items()
    ]
    ratings, rating_error = _collect_ratings(rating_source, _repos_with_rows(blocks))
    return RenderDocument(
        schema_version=DOCUMENT_SCHEMA_VERSION,
        snapshot_run_at=snapshot.run_at,
        rendered_at=rendered_at,
        base_model_count=len(snapshot.base_models),
        package_count=len(snapshot.packages),
        scenario=scenario,
        ranking_rule=RANKING_RULE,
        rating_unavailable=rating_error,
        ratings=ratings,
        areas=snapshot.areas,
        machines=blocks,
    )


# --- writer: JSON --------------------------------------------------------------------------------


def document_json(document: RenderDocument) -> dict:
    """The document as the JSON file next to the Markdown one: the object itself, nothing added."""
    return document.model_dump(mode="json")
