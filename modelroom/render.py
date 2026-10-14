"""`render`: one document object per run, then three writers that only format it.

`build_render_document(...) -> document.RenderDocument` is the pure builder: it takes an
already-loaded `Configuration`, `Snapshot`, one `MachineProfile` per configured machine, that
profile's measurement records, the ranking `Scenario` and `rendered_at`, computes every fit
(`runtimes.fit_for`, AP9-SV2-L: the runtime a package's table row belongs to judges it --
`fit.compute_fit_v2` for a GGUF or Ollama-library package, `fit_vllm.vllm_fit_as_fit` for a
Hugging Face tensor one, AP9-SV2-F) and every ranking (`ranking.rank_packages`), and returns the
one object the three
writers read: `document_markdown`, `document_json`, `document_terminal`. No I/O anywhere in this
module -- `modelroom/cli.py::render_with_config` owns the lock, the schema-version gates, reading
the files and the atomic writes.

`machine_profile` is the other pure decision here: which profile a `[machines.<name>]` entry is
rendered from, given the profile files found in the results folder. See CONTRACTS.md,
"Render (schema 2)".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, Sequence

from .catalog import Catalog
from .catalog_findings import PackageCheck, SetupNote, load_blocked_text
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
from .fit import GIB, unknown_fit
from .guided_contracts import Note
from .importer import ProfileScan
from .measurements import MeasurementRecord, Scenario
from .origin import listed_packagers_of, package_origin
from .pull_choice import blocked_runtimes
from .profile import PROFILE_SCHEMA_VERSION, CrossCheck, HardwareProfile, LlmfitCrosscheck, fit_block_reason
from .quantization import package_identity_key
from .parameters import parameter_count
from .reserve import ram_reserve
from .runtimes import MachineRuntimes, fit_for, runtimes_among, visible_packages
from .views_sections import gguf_rule_name
from .vllm_words import is_vllm, reason_sentence as vllm_reason_sentence, row_note as vllm_row_note
from .ranking import (
    RANKING_RULE,
    RANKING_RULE_PER_MODEL,
    ModelFacts,
    RankedPackage,
    Ranking,
    SetAside,
    ShownRow,
    both_runtimes_rule,
    rank_packages,
    row_sentence,
    shown_rows,
)

RatingSource = Callable[[str], Rating | None]

_HEADER_RE = re.compile(
    r"^<!-- modelroom render: snapshot_run_at=(?P<snapshot_run_at>\S+) rendered_at=(?P<rendered_at>\S+) -->$"
)
_ELIGIBLE_PROVENANCE = ("from_metadata", "approved")
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
    profile's `display_name` (with `(entered)` behind it for a machine entered by hand), the
    schema-1 file's name, or the machine name.
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
        gpus=[],
        machine_class="unknown",
        machine_class_source="unknown",
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
            # A machine entered by hand says so wherever its name stands (decided 2026-09-26).
            entered = found.ram_physical_source == "entered"
            return MachineProfile("ranked", f"{found.display_name} (entered)" if entered else found.display_name, found, None)
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


def _eligible_packages(snapshot: Snapshot, machines: Sequence[MachineRuntimes] = ()) -> list[Package]:
    """Every package a fit may be computed for: shown provenance, complete, active, and visible
    for `machines` (`runtimes.visible_packages`, the one visibility rule).

    The base model is not checked here: `Snapshot` itself refuses a package whose
    `base_model_hf_repo` is not among its `base_models`, so every package that reaches this
    point has an architecture the fit can read.

    `machines` are the machines the list is read for -- one machine for its own block or scale,
    every shown machine for a document-wide number. No machines means no runtime present:
    no package a runtime claims (a Hugging Face tensor package).
    """
    return [
        package
        for package in visible_packages(snapshot.packages, machines)
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
    # Two graphics cards or more, when no single one holds the package (decided 2026-09-26).
    "gpu_split": (
        "spread_over_graphics_cards",
        "Fits into graphics memory spread over {cards} cards: it needs about {need:.1f} GiB of the "
        "{pool:.1f} GiB left after the reserve of {per_card:.1f} GiB on each card.",
        ["fit.mode", "fit.need_gib", "fit.pool_gib", "fit.reserve_gib", "profile.gpus"],
    ),
    "cpu_gpu": (
        "shared_between_memories",
        "Too large for graphics memory alone, so it runs in system memory with the graphics "
        "card helping; the best rating it can reach here is 'fits'.",
        ["fit.mode", "fit.fit_class"],
    ),
    "cpu": (
        "cpu_caps_at_fits",
        "Runs in system memory only, so the best possible rating on this machine is 'fits'.",
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


def _computed_note(fit: Fit, shared_memory: bool = False, cards: int = 1) -> Note:
    """The row note of a computed fit; `shared_memory` names a `gpu` row of unified memory so, and
    `cards` is the number of graphics cards a `gpu_split` row is spread over -- the whole machine's,
    as Ollama spreads over every usable card there is.

    A vLLM row is judged by another rule (requests at once on its own `Fit.cards`), so its note is
    vLLM's own (`vllm_words.row_note`, Codex r2-1) and none of the mode sentences above reaches it.
    """
    if is_vllm(fit):
        code, text, facts = vllm_row_note(fit)
    else:
        shared = shared_memory and fit.mode == "gpu"
        code, template, facts = _SHARED_MEMORY_NOTE if shared else _MODE_NOTES[str(fit.mode)]
        text = template.format(need=fit.need_gib, pool=fit.pool_gib, cards=cards, per_card=fit.reserve_gib / max(cards, 1))
    if fit.basis == "size":
        text += _FROM_SIZE_SENTENCE
        facts = [*facts, "fit.basis"]
    reason_text, reason_facts = vllm_reason_sentence(fit)
    return Note(code=code, subject="package", origin="computed", text=_clip(text + reason_text), facts=[*facts, *reason_facts])


def _vllm_no_fit_note(fit: Fit) -> Note:
    """A vLLM row under `No fit`: not even one request at once on its cards (`vllm_words.row_note`),
    never Ollama's need against the pool left after its reserve (Codex r2-1)."""
    code, text, facts = vllm_row_note(fit)
    from_size = _FROM_SIZE_SENTENCE if fit.basis == "size" else ""
    reason_text, reason_facts = vllm_reason_sentence(fit)
    facts = [*facts, *(["fit.basis"] if from_size else []), *reason_facts]
    return Note(code=code, subject="package", origin="computed", text=_clip(f"{text}{from_size}{reason_text}"), facts=facts)


def _set_aside_note(set_aside: SetAside, covered: bool, shared_memory: bool = False) -> Note:
    if not covered:
        return Note(
            code="not_covered",
            subject="package",
            origin="computed",
            text=_clip(f"Fit contract v1 cannot judge this package here: {set_aside.reason}."),
            facts=["fit.fit_class", "fit.reason"],
        )
    # Not "more than the machine has": fit v1 already calls a package `no_fit` above 98 % of
    # the pool, so the need can be below the pool and the note must not claim an overrun
    # (probe: need 7.125 GiB against a pool of 7.2 GiB is `no_fit`).
    fit = set_aside.fit
    if is_vllm(fit):
        return _vllm_no_fit_note(fit)
    # A package set aside on the size basis says on what basis: the number that put it there was
    # computed from its size, not from its architecture (second-model round, 2026-09-24).
    from_size = _FROM_SIZE_SENTENCE if fit.basis == "size" else ""
    reserves = "both reserves" if shared_memory else "the reserve"
    if fit.pool_gib > 0:
        room = f"of the {fit.pool_gib:.1f} GiB left after {reserves}"
    else:
        # A pool of 0 or less is said in words, never as a memory below zero (AP9-RM RM-2): the
        # memory in all is the pool and the reserve together.
        room = (
            f"and nothing is left after {reserves} of {fit.reserve_gib:.1f} GiB "
            f"({fit.pool_gib + fit.reserve_gib:.1f} GiB in all)"
        )
    return Note(
        code="no_fit",
        subject="package",
        origin="computed",
        text=_clip(f"Fit contract v1 does not count this as a fit: it needs about {fit.need_gib:.1f} GiB {room}.{from_size}"),
        facts=["fit.need_gib", "fit.pool_gib"]
        + ([] if fit.pool_gib > 0 else ["fit.reserve_gib"])
        + (["fit.basis"] if fit.basis == "size" else []),
    )


# --- building the document ----------------------------------------------------------------------


def _row_note(ranked: RankedPackage, shared_memory: bool, cards: int = 1) -> Note:
    """What one ranked row says: its measurement, why a measurement did not count, or its fit.

    A measurement that counts but for `requests` alone is named in the row, so a reader does not
    look for a speed the ranking of more requests cannot use (`RankedPackage.measurement_note`).
    """
    if ranked.measurement is not None:
        return _measured_note(ranked.measurement)
    if ranked.measurement_note is not None:
        text, facts = ranked.measurement_note, ["scenario.requests", "measurement_group"]
        # The size-basis sentence stays whatever else the row says (see `_FROM_SIZE_SENTENCE`).
        if ranked.fit.basis == "size":
            text, facts = f"{text}.{_FROM_SIZE_SENTENCE}", [*facts, "fit.basis"]
        return Note(code="measured_with_one_request", subject="package", origin="computed", text=_clip(text), facts=facts)
    if ranked.measured_other is not None:
        # A measurement of this package that counts nowhere here (another context, not comparable, not
        # valid): the row says why, and no speed stands in it (Codex F3, ADR 0009).
        text, facts = f"{ranked.measured_other}.", ["measurement.scenario.context_requested", "measurement_group"]
        if ranked.fit.basis == "size":
            text, facts = f"{text}{_FROM_SIZE_SENTENCE}", [*facts, "fit.basis"]
        return Note(code="measured_not_comparable", subject="package", origin="computed", text=_clip(text), facts=facts)
    return _computed_note(ranked.fit, shared_memory, cards)


def _with_rule_sentence(note: Note, row: ShownRow, per_model: bool) -> Note:
    """The row note with the sentence the rule gives the row appended: it adds, never replaces."""
    said = row_sentence(row, per_model)
    if said is None:
        return note
    sentence, fact = said
    return note.model_copy(update={"text": _clip(f"{note.text} {sentence}"), "facts": [*note.facts, fact]})


def _ranked_entries(
    ranking: Ranking, shared_memory: bool = False, cards: int = 1, per_model: bool = False, start: int = 1, whole: bool = False
) -> list[RankedEntry]:
    """The rows of one machine (`ranking.shown_rows`); `rank` is the number of the row shown,
    counted on from `start`. `whole`: every ranked package, no cut and no rule sentence (F5, the
    sections after the first, so their count stays exact: `views_sections.runtime_sections`)."""
    entries = []
    rows = [ShownRow(ranked, "rule", ranked.rank, len(ranking.ranked)) for ranked in ranking.ranked] if whole else None
    per_model = per_model and not whole
    for number, row in enumerate(rows if rows is not None else shown_rows(ranking, per_model), start=start):
        ranked = row.ranked
        note = _with_rule_sentence(_row_note(ranked, shared_memory, cards), row, per_model)
        entries.append(
            RankedEntry(
                rank=number,
                fit=ranked.fit,
                measurement_group=ranked.measurement_group,
                measurement_id=ranked.measurement.measurement_id if ranked.measurement is not None else None,
                speed_tps=ranked.measurement.tps_mean if ranked.measurement is not None else None,
                speed_protocol=ranked.measurement.protocol if ranked.measurement is not None else None,
                **_package_line(ranked.package, note),
            )
        )
    return entries


def _set_aside_entries(
    items: list[SetAside], covered: bool, shared_memory: bool = False, blocked: Mapping[tuple[str, str, str], str] | None = None
) -> list[SetAsideEntry]:
    """`blocked` maps a package whose `no_fit` became `not_covered` by an `absent` check (`_blocked_or`) to the red
    line that is its reason; its note is that line, not a statement about the fit rule."""
    return [
        SetAsideEntry(
            fit=item.fit,
            reason=item.reason,
            **_package_line(item.package, _blocked_note(item.reason) if (blocked or {}).get(package_identity_key(item.package)) == item.reason else _set_aside_note(item, covered, shared_memory)),
        )
        for item in items
    ]


def _blocked_note(reason: str) -> Note:
    """The note of a blocked package: the red line itself."""
    return Note(code="load_blocked", subject="package", origin="computed", text=_clip(reason), facts=["fit.reason"])


def _machine_block(
    machine: str,
    machine_config: MachineConfig,
    found: MachineProfile,
    packages: list[Package],
    base_model_by_repo: dict[str, BaseModelSpec],
    measurements: list[MeasurementRecord],
    scenario: Scenario,
    models: Mapping[str, ModelFacts] | None = None,
    per_model: bool = False,
    checks: Mapping[tuple[str, str, str], PackageCheck] | None = None,
) -> MachineRanking:
    """One machine's block: its three lists, or its reason when no fit can be computed at all.

    The block carries the memory reserve the fit took and where it came from (`reserve.ram_reserve`,
    AP9-RM): without a profile the memory is unknown, so only a number entered stays one.
    """
    profile = found.profile
    ram_gib = profile.ram_physical_gib if profile is not None else None
    unified = profile is not None and profile.gpu_state == "unified_memory"
    ram_reserve_gib, origin = ram_reserve(machine_config, ram_gib, unified)
    reserves = {
        "reserve_ram_gib": ram_reserve_gib,
        "reserve_vram_gib": machine_config.reserve_vram_gib,
        "reserve_ram_origin": origin,
    }
    if found.profile is None:
        return MachineRanking(
            machine=machine, status=found.status, label=found.label, reason=found.reason, **reserves
        )
    entries: list[tuple[Package, Fit]] = []
    blocked: dict[tuple[str, str, str], str] = {}  # packages whose `no_fit` an `absent` check turned into `not_covered`
    for package in packages:
        fit = fit_for(found.profile, package, base_model_by_repo[package.base_model_hf_repo], scenario, machine_config)
        gated = _blocked_or(fit, package, found.profile, checks)
        if gated is not fit:
            blocked[package_identity_key(package)] = str(gated.reason)
        entries.append((package, gated))
    shared = found.profile.gpu_state == "unified_memory"  # one memory, both reserves taken from it
    ranked: list[RankedEntry] = []
    rankings = _rankings_by_runtime(entries, measurements, scenario, models)
    for index, ranking in enumerate(rankings):
        cards = len(found.profile.gpus)
        ranked += _ranked_entries(ranking, shared, cards, per_model, start=len(ranked) + 1, whole=index > 0)
    return MachineRanking(
        machine=machine,
        status="ranked",
        label=found.label,
        profile=found.profile,
        ranked=ranked,
        ranked_total=sum(len(ranking.ranked) for ranking in rankings),
        not_covered=[
            entry for ranking in rankings for entry in _set_aside_entries(ranking.not_covered, covered=False, blocked=blocked)
        ],
        no_fit=[entry for ranking in rankings for entry in _set_aside_entries(ranking.no_fit, True, shared)],
        **reserves,
    )


def _blocked_or(fit: Fit, package: Package, profile: HardwareProfile, checks: Mapping[tuple[str, str, str], PackageCheck] | None) -> Fit:
    """`fit`, except that a package an `absent` check blocks which the fit would set aside as `no_fit` is `unknown`
    with the red line of the block as its reason: it is carried as `not_covered`, with the reason, not as `no_fit`
    (V5; `catalog_findings.load_blocked_text`). A ranked row keeps its place and its class -- it is locked in the
    views and in the pull list; a package the fit already cannot judge keeps its own reason."""
    check = (checks or {}).get(package_identity_key(package))
    if check is None or check.result != "absent" or fit.fit_class != "no_fit":
        return fit
    return unknown_fit(load_blocked_text(blocked_runtimes(profile), package.base_model_hf_repo, check))


def _rankings_by_runtime(
    entries: list[tuple[Package, Fit]], measurements: list[MeasurementRecord], scenario: Scenario, models
) -> list[Ranking]:
    """One ranking per runtime whose fits are among `entries`, in the order of `runtimes.RUNTIMES`
    (F5): the rows of two runtimes are alternatives on the same cards, never one list, so each is
    ranked on its own -- a machine of one runtime ranks exactly as before. The document keeps one
    block with continuous numbers (Ollama's rows first); the views split it into sections."""
    keys = runtimes_among(fit for _package, fit in entries)  # the one predicate (Codex r6-1)
    if len(keys) < 2:
        return [rank_packages(entries, measurements, scenario, models=models)]
    return [rank_packages([e for e in entries if e[1].runtime == key], measurements, scenario, models=models) for key in keys]


def _repos_with_rows(blocks: list[MachineRanking]) -> list[str]:
    """Every base model that has at least one row anywhere, in the order the rows name it."""
    seen: list[str] = []
    for block in blocks:
        for entry in [*block.ranked, *block.not_covered, *block.no_fit]:
            if entry.base_model_hf_repo not in seen:
                seen.append(entry.base_model_hf_repo)
    return seen


def _package_checks(packages: Sequence[Package], catalog: Catalog | None) -> dict[tuple[str, str, str], PackageCheck]:
    """The check the catalog holds for each package that has one, by package identity; none without a catalog."""
    if catalog is None:
        return {}
    return {package_identity_key(package): found for package in packages if (found := catalog.package_check(package)) is not None}


def _with_evidence(
    blocks: list[MachineRanking], packages: Sequence[Package], catalog: Catalog | None
) -> tuple[list[MachineRanking], dict[str, SetupNote]]:
    """The blocks with the catalog's check on each row of a package it has one for, and the setup note of each
    base model that has a row (AP9-KO, V11). No catalog gives the blocks back as they are and no note: the document
    of release 1.0.0. A catalog that holds nothing for these rows gives that document only where it also changes the
    origin of no row (`origin: "other"` needs no check). The rows keep their place and their class; what a check
    changes in a view is not decided here."""
    if catalog is None:
        return blocks, {}
    updates: dict[tuple[str, str, str], dict] = {}
    listed = listed_packagers_of(catalog)
    for package in packages:
        update: dict = {}
        found = catalog.package_check(package)
        if found is not None:
            update["load_check"] = found
        if package.source == "huggingface" and package.format == "gguf" and package.repo is not None:
            if package_origin(package.repo.split("/", 1)[0], package.base_model_hf_repo, listed) == "other":
                update["origin"] = "other"
        if update:
            updates[package_identity_key(package)] = update

    def checked(entries):
        return [
            entry.model_copy(update=updates[tuple(entry.package_identity)]) if tuple(entry.package_identity) in updates else entry
            for entry in entries
        ]

    if updates:
        blocks = [
            block.model_copy(
                update={"ranked": checked(block.ranked), "not_covered": checked(block.not_covered), "no_fit": checked(block.no_fit)}
            )
            for block in blocks
        ]
    notes = {repo: note for repo in _repos_with_rows(blocks) if (note := catalog.setup_note_for(repo)) is not None}
    return blocks, notes


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


def snapshot_model_facts(snapshot: Snapshot) -> dict[str, ModelFacts]:
    """What the snapshot alone says of its base models: the parameter count, and no `legacy`.

    The same rule the list of step 2 uses (`parameters.parameter_count`), with the count the snapshot
    keeps; a render without a guided run knows no release, so the parameters decide (ADR 0009).
    """
    return {
        spec.hf_repo: ModelFacts(parameter_count(spec.parameters_b, spec.hf_repo.partition("/")[2]), False)
        for spec in snapshot.base_models
    }


def build_render_document(
    config: Configuration,
    snapshot: Snapshot,
    profiles: dict[str, MachineProfile],
    measurements: dict[str, list[MeasurementRecord]],
    scenario: Scenario,
    rendered_at: datetime,
    rating_source: RatingSource | None = None,
    *,
    models: Mapping[str, ModelFacts] | None = None,
    live_vllm: Mapping[str, object] | None = None,
    catalog: Catalog | None = None,
) -> RenderDocument:
    """The one object a render produces; pure, no I/O and no clock read.

    `catalog` is the evidence (AP9-KO, V11): the check it holds for a package goes on that package's row
    (`PackageLine.load_check`), the setup note it holds for a base model with rows into `setup_notes`. `None`
    is the default and leaves the document exactly as it was: every package is `unknown`. A catalog that knows
    nothing of these packages does so only where it changes the origin of no row (`_with_evidence`; CONTRACTS.md
    V11).

    `profiles` maps every `config.machines` key to the `MachineProfile` the results folder
    holds for it (`machine_profile`); `measurements` maps a `profile_id` to that profile's
    measurement records. Every fit is computed against `scenario`, one context for the whole
    document (CONTRACTS.md, "Render (schema 2)"). `models` is what the rule knows of each base
    model (ADR 0009); without it the snapshot says it (`snapshot_model_facts`), and no model is
    `legacy`. Several base models among the packages give one row per model (`ranking.shown_rows`).

    `live_vllm` maps a machine key to whatever that very run's own step 1 found for `"vllm"`
    (`GuidedRun.servers.found.get("vllm")`) -- a guided run's own render step passes it for the
    one machine it is running on; `modelroom render` (no guided run, no server talked to) passes
    nothing, so every machine is judged by its stored profile `runtimes` alone.

    Visibility (`runtimes.visible_packages`): each machine's block reads the packages visible for
    that machine alone; every document-wide number (`package_count`, the ranking rule) reads the
    packages visible for at least one shown machine. A Hugging Face tensor package therefore leaves no trace in
    a document none of whose machines has vLLM; the snapshot keeps its raw facts.
    """
    base_model_by_repo = {bm.hf_repo: bm for bm in snapshot.base_models}
    live_vllm = live_vllm or {}
    machine_runtimes = {machine: (profiles[machine].profile, live_vllm.get(machine)) for machine in config.machines}
    shown = list(machine_runtimes.values())
    per_model = len({package.base_model_hf_repo for package in _eligible_packages(snapshot, shown)}) > 1
    facts = models if models is not None else snapshot_model_facts(snapshot)
    checks = _package_checks(snapshot.packages, catalog)
    blocks = [
        _machine_block(
            machine,
            machine_config,
            profiles[machine],
            _eligible_packages(snapshot, [machine_runtimes[machine]]),
            base_model_by_repo,
            measurements.get(getattr(profiles[machine].profile, "profile_id", ""), []),
            scenario,
            facts,
            per_model,
            checks,
        )
        for machine, machine_config in config.machines.items()
    ]
    blocks, setup_notes = _with_evidence(blocks, snapshot.packages, catalog)
    ratings, rating_error = _collect_ratings(rating_source, _repos_with_rows(blocks))
    rule = RANKING_RULE_PER_MODEL if per_model else RANKING_RULE
    # F5 (Codex r5-1): the rule says the per-runtime selection wherever a machine ranks both.
    # The same predicate as the sections and the whole vLLM ranking, over every row shown (Codex r6-1).
    gguf_name = gguf_rule_name(blocks)  # S4: the runtime of the GGUF rows; `None` where no machine shows rows of two
    rule += both_runtimes_rule(gguf_name) if gguf_name else ""
    return RenderDocument(
        schema_version=DOCUMENT_SCHEMA_VERSION,
        snapshot_run_at=snapshot.run_at,
        rendered_at=rendered_at,
        base_model_count=len(snapshot.base_models),
        package_count=len(visible_packages(snapshot.packages, shown)),
        scenario=scenario,
        ranking_rule=rule,
        rating_unavailable=rating_error,
        ratings=ratings,
        areas=snapshot.areas,
        machines=blocks,
        setup_notes=setup_notes,
    )


# --- writer: JSON --------------------------------------------------------------------------------


def document_json(document: RenderDocument) -> dict:
    """The document as the JSON file next to the Markdown one: the object itself, nothing added."""
    return document.model_dump(mode="json")
