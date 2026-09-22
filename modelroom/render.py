"""`render`: a pure reader of the snapshot and every configured machine's hardware profile.

`build_document(config, snapshot, hardware_by_machine, rendered_at, rating_source=None) -> str`
is the whole module's public surface besides `RatingSource`/`RatingUnavailableError` and the
header-line helpers -- no I/O anywhere here, exactly like `modelroom/fit.py::compute_fit`.
`modelroom/cli.py::render_with_config` is the orchestration layer (the lock, the schema-version
gates, reading the snapshot and hardware files, the atomic write); this module only turns
already-loaded data into Markdown. See CONTRACTS.md, "Render (AP5)", for every rule below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from .config import Configuration
from .contracts import (
    Area,
    BaseModelSpec,
    HardwareSnapshot,
    Measurement,
    Package,
    Rating,
    Snapshot,
)
from .fit import GIB, compute_fit
from .quantization import QUANT_ORDER, package_identity_key, sort_key

RatingSource = Callable[[str], Rating | None]

_HEADER_RE = re.compile(
    r"^<!-- modelroom render: snapshot_run_at=(?P<snapshot_run_at>\S+) rendered_at=(?P<rendered_at>\S+) -->$"
)
_ELIGIBLE_PROVENANCE = ("metadata_ok", "approved")
_QUALIFYING_FIT_CLASSES = ("good", "perfect")


class RatingUnavailableError(Exception):
    """A `RatingSource` could not answer at all for the base model it was asked about."""


@dataclass(frozen=True)
class HeaderInfo:
    """The two timestamps in a rendered document's first line, as `parse_header_line` reads it."""

    snapshot_run_at: datetime
    rendered_at: datetime


def format_header_line(snapshot_run_at: datetime, rendered_at: datetime) -> str:
    """The fixed header comment `build_document` always writes as the document's first line."""
    return f"<!-- modelroom render: snapshot_run_at={snapshot_run_at.isoformat()} rendered_at={rendered_at.isoformat()} -->"


def _is_aware_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == timedelta(0)


def parse_header_line(text: str) -> HeaderInfo | None:
    """The header line's two timestamps, or `None` when `text` has no such first line.

    `None` covers an empty file, one whose first line does not match the fixed format (e.g. a
    document from before `render` wrote this header), and -- fix-round 5, F4 -- one whose
    `snapshot_run_at`/`rendered_at` parses but is not an aware UTC datetime (`format_header_line`
    never writes anything else, same rule as `contracts.check_aware_utc`; a hand-edited or
    otherwise corrupted header with a naive timestamp would otherwise make
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


# --- eligibility and grouping ----------------------------------------------------------------


def _eligible_packages(packages: list[Package]) -> list[Package]:
    return [p for p in packages if p.provenance in _ELIGIBLE_PROVENANCE and p.complete and p.active]


def _packager_name(package: Package) -> str:
    if package.source == "ollama":
        return "ollama"
    return (package.repo or "").split("/", 1)[0]


def _groups_for_base_model(repo: str, packages: list[Package]) -> list[tuple[str, str, list[Package]]]:
    by_packager: dict[str, list[Package]] = {}
    for package in packages:
        by_packager.setdefault(_packager_name(package), []).append(package)
    ordered_packagers = sorted(by_packager, key=lambda name: (name == "ollama", name))
    return [(repo, packager, by_packager[packager]) for packager in ordered_packagers]


def _ordered_groups(config: Configuration, eligible: list[Package]) -> list[tuple[str, str, list[Package]]]:
    """Groups (base model x packager) in config order, packagers alphabetical with ollama last."""
    by_repo: dict[str, list[Package]] = {}
    for package in eligible:
        by_repo.setdefault(package.base_model_hf_repo, []).append(package)

    groups: list[tuple[str, str, list[Package]]] = []
    for family in config.families:
        for base_model_config in family.base_models:
            repo = base_model_config.hf_repo
            groups.extend(_groups_for_base_model(repo, by_repo.get(repo, [])))
    return groups


def _unique_repos_with_rows(groups: list[tuple[str, str, list[Package]]]) -> list[str]:
    seen: list[str] = []
    for repo, _packager, packages in groups:
        if packages and repo not in seen:
            seen.append(repo)
    return seen


# --- selection rule ------------------------------------------------------------------------


def _quant_index(package: Package) -> int:
    """Same convention as `quantization.sort_key`: unknown quantization sorts as if largest."""
    return QUANT_ORDER.index(package.quantization) if package.quantization in QUANT_ORDER else len(QUANT_ORDER)


def _select_for_machine(packages: list[Package], base_model: BaseModelSpec, hardware, machine_config) -> Package | None:
    """The picked package for one machine, largest good-or-better quant, else the smallest
    *judged* package -- or `None` when this machine has nothing to recommend at all.

    F5 (fix-round 5): a machine with no hardware profile, or one where fit v1 cannot judge a
    single package of the group (`fit_class == "unknown"` for every one -- typically the whole
    architecture is not covered by v1, e.g. the real Qwen3.5 case measured 2026-09-22), makes no
    pick, rather than falling back to an arbitrary "smallest" package fit v1 never actually
    scored (the old behavior: a useless recommendation such as `UD-IQ2_XXS` for an architecture
    fit v1 cannot judge at all). The "else the smallest" fallback still applies once at least one
    package *was* judged (a real class, not "unknown") but none reached good/perfect -- scoped to
    the judged packages only, never back to an unknown one that only looks smallest by quant
    label.
    """
    if hardware is None:
        return None

    fits_by_key = {package_identity_key(p): compute_fit(p, base_model, hardware, machine_config) for p in packages}
    judged = [p for p in packages if fits_by_key[package_identity_key(p)].fit_class != "unknown"]
    if not judged:
        return None

    qualifying = [p for p in judged if fits_by_key[package_identity_key(p)].fit_class in _QUALIFYING_FIT_CLASSES]
    if not qualifying:
        return min(judged, key=sort_key)

    max_index = max(_quant_index(p) for p in qualifying)
    candidates = [p for p in qualifying if _quant_index(p) == max_index]
    return min(candidates, key=sort_key)


def _select_group_packages(
    packages: list[Package], base_model: BaseModelSpec, config: Configuration, hardware_by_machine: dict
) -> list[Package]:
    """Every distinct package picked by at least one machine -- never one a machine had nothing
    to recommend for (F5: `_select_for_machine` returning `None` contributes nothing here).
    """
    picked_by_key: dict[tuple, Package] = {}
    for machine_name, machine_config in config.machines.items():
        hardware = hardware_by_machine.get(machine_name)
        picked = _select_for_machine(packages, base_model, hardware, machine_config)
        if picked is not None:
            picked_by_key[package_identity_key(picked)] = picked
    return list(picked_by_key.values())


def _no_recommendation_reason(packages: list[Package], base_model: BaseModelSpec, hardware, machine_config) -> str:
    """F5: the reason text for a machine that made no pick -- `compute_fit`'s own reason (every
    package shares it when the cause is architecture-level, the common case) computed against a
    deterministic representative package (`sort_key`'s smallest) when it is package-level
    instead, or the literal `"no profile"` when this machine has no hardware profile at all.
    """
    if hardware is None:
        return "no profile"
    representative = min(packages, key=sort_key)
    return compute_fit(representative, base_model, hardware, machine_config).reason or "unknown"


@dataclass(frozen=True)
class PackageRow:
    base_model_hf_repo: str
    packager: str
    package: Package | None
    base_model: BaseModelSpec
    variants: int
    # F5: set only on a "no recommendation" row (`package is None`) -- maps each configured
    # machine name to the reason text its fit cell shows (`"no recommendation: <reason>"`).
    no_recommendation: dict[str, str] | None = None


def _build_rows(
    groups: list[tuple[str, str, list[Package]]],
    base_model_by_repo: dict[str, BaseModelSpec],
    config: Configuration,
    hardware_by_machine: dict,
) -> list[PackageRow]:
    rows: list[PackageRow] = []
    for repo, packager, packages in groups:
        if not packages:
            continue
        base_model = base_model_by_repo[repo]
        selected = _select_group_packages(packages, base_model, config, hardware_by_machine)
        # F5: every configured machine made no pick for this group (config.machines is never
        # empty here -- an empty one would make `selected` trivially empty for an unrelated
        # reason, and must fall through to the ordinary, zero-row path unchanged).
        if not selected and config.machines:
            no_recommendation = {
                name: _no_recommendation_reason(packages, base_model, hardware_by_machine.get(name), machine_config)
                for name, machine_config in config.machines.items()
            }
            rows.append(PackageRow(repo, packager, None, base_model, len(packages), no_recommendation))
            continue
        variants = len(packages) - len(selected)
        for package in sorted(selected, key=sort_key):
            rows.append(PackageRow(repo, packager, package, base_model, variants))
    return rows


# --- ratings ---------------------------------------------------------------------------------


def _collect_ratings(
    rating_source: RatingSource | None, base_model_repos: list[str]
) -> tuple[dict[str, Rating | None], str | None]:
    """Call `rating_source` once per base model; the first exception's message, if any."""
    if rating_source is None:
        return {}, None

    ratings: dict[str, Rating | None] = {}
    error_message: str | None = None
    for repo in base_model_repos:
        try:
            ratings[repo] = rating_source(repo)
        except Exception as exc:  # noqa: BLE001 -- every exception is treated the same (brief)
            if error_message is None:
                error_message = str(exc)
            ratings[repo] = None
    return ratings, error_message


def _format_stars(stars: float) -> str:
    full_stars, half = divmod(round(stars * 2), 2)
    return "★" * full_stars + ("½" if half else "")


def _stars_cell(rating: Rating | None, rating_failed: bool) -> str:
    if rating_failed or rating is None:
        return "–"
    return _format_stars(rating.stars)


# --- fit / installed / speed cells ------------------------------------------------------------


def _fit_cell(package: Package, base_model: BaseModelSpec, machine_config, hardware) -> str:
    if hardware is None:
        return "no profile"
    fit = compute_fit(package, base_model, hardware, machine_config)
    if fit.fit_class == "unknown":
        return f"unknown: {fit.reason}"
    return f"{fit.fit_class} ({fit.mode}, need {fit.need_gib:.1f} / pool {fit.pool_gib:.1f} GiB)"


def _installed_cell(package: Package, hardware: HardwareSnapshot | None) -> str:
    if package.source == "huggingface":
        return "–"
    if hardware is None or hardware.installed is None:
        return "unknown"
    matched = any(model.digest == package.manifest_digest for model in hardware.installed)
    return "yes" if matched else "no"


def _measurement_matches_package(measurement: Measurement, package: Package) -> bool:
    if measurement.content_source == "ollama":
        return package.source == "ollama" and measurement.ollama_manifest_digest == package.manifest_digest
    if package.source != "huggingface":
        return False
    if measurement.hf_repo != package.repo or measurement.hf_revision != package.revision:
        return False
    weight_digests = {f.digest for f in package.files if f.role in ("weights", "weights_shard") and f.digest}
    return measurement.hf_file_digest in weight_digests


def _speed_cell(package: Package, machine_names: list[str], hardware_by_machine: dict) -> str:
    for machine_name in machine_names:
        hardware = hardware_by_machine.get(machine_name)
        if hardware is None:
            continue
        for measurement in hardware.measurements:
            if measurement.profile_measured_at != hardware.measured_at:
                continue
            if _measurement_matches_package(measurement, package):
                return f"{measurement.tps_mean:.1f} tps @{measurement.context} ({machine_name})"
    return "–"


def _weights_gib(package: Package) -> float:
    return sum(f.size_bytes for f in package.files if f.role in ("weights", "weights_shard")) / GIB


def _context_cell(package: Package) -> str:
    if package.default_context:
        return str(package.default_context)
    return "8192 (assumed)"


# --- rendering: summary, areas, machines, packages ---------------------------------------------


def _summary_lines(snapshot: Snapshot, rendered_at: datetime, rating_failed: bool, rating_error: str | None) -> list[str]:
    lines = [
        "# Model packages",
        "",
        f"Snapshot run at: {snapshot.run_at.isoformat()}",
        f"Rendered at: {rendered_at.isoformat()}",
        f"Base models / packages: {len(snapshot.base_models)} / {len(snapshot.packages)}",
    ]
    if rating_failed:
        lines.append(f"Market rating unavailable: {rating_error}")
    return lines


def _render_areas_table(areas: list[Area]) -> str:
    header = "| Source | Base model | Packager | Status | Last success | Error |\n|---|---|---|---|---|---|"
    lines = [header]
    for area in areas:
        last_success = area.last_success.isoformat() if area.last_success else "–"
        lines.append(
            f"| {area.source} | {area.base_model_hf_repo} | {area.packager or '–'} | {area.status} | "
            f"{last_success} | {area.error or ''} |"
        )
    return "\n".join(lines)


def _installed_summary(hardware: HardwareSnapshot) -> str:
    if hardware.installed is not None:
        return str(len(hardware.installed))
    return f"unknown ({hardware.installed_unavailable_reason})"


def _machine_row(name: str, machine_config, hardware: HardwareSnapshot | None, rendered_at: datetime) -> str:
    reserve_vram = f"{machine_config.reserve_vram_gib:.2f}"
    reserve_ram = f"{machine_config.reserve_ram_gib:.2f}"
    if hardware is None:
        no_profile = "no hardware profile yet"
        return (
            f"| {name} | {no_profile} | {no_profile} | {reserve_vram} | {reserve_ram} | {no_profile} | "
            f"{no_profile} | {no_profile} | {no_profile} |"
        )
    backend_gpu = f"{hardware.backend or '–'} / {hardware.gpu_name or '–'}"
    age_days = (rendered_at - hardware.measured_at).days
    return (
        f"| {name} | {hardware.vram_gib:.2f} | {hardware.ram_gib:.2f} | {reserve_vram} | {reserve_ram} | "
        f"{backend_gpu} | {hardware.measured_at.isoformat()} | {age_days} | {_installed_summary(hardware)} |"
    )


def _render_machines_table(config: Configuration, hardware_by_machine: dict, rendered_at: datetime) -> str:
    header = (
        "| Machine | VRAM GiB | RAM GiB | Reserve VRAM GiB | Reserve RAM GiB | Backend / GPU | "
        "Profile measured at | Profile age (days) | Installed |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for name, machine_config in config.machines.items():
        lines.append(_machine_row(name, machine_config, hardware_by_machine.get(name), rendered_at))
    return "\n".join(lines)


def _packages_table_header(machine_names: list[str]) -> str:
    columns = ["Base model", "Stars", "Packager", "Quant", "Format", "Size GiB", "Context"]
    columns += [f"fit (computed, v1): {name}" for name in machine_names]
    columns += ["Installed", "Speed", "Provenance", "Observed", "Variants"]
    return "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)


def _no_recommendation_row_line(
    row: PackageRow, ratings: dict, rating_failed: bool, machine_names: list[str]
) -> str:
    """F5: every package cell "–" except the fit cells (`"no recommendation: <reason>"`) and
    Variants (`"<N> variants, none judged"`) -- there is no picked `Package` to read the other
    cells from.
    """
    assert row.no_recommendation is not None  # only ever built alongside a no-recommendation row
    cells = [
        row.base_model_hf_repo,
        _stars_cell(ratings.get(row.base_model_hf_repo), rating_failed),
        row.packager,
        "–",  # Quant
        "–",  # Format
        "–",  # Size GiB
        "–",  # Context
    ]
    cells += [f"no recommendation: {row.no_recommendation[name]}" for name in machine_names]
    cells.append("–")  # Installed
    cells.append("–")  # Speed
    cells.append("–")  # Provenance
    cells.append("–")  # Observed
    cells.append(f"{row.variants} variants, none judged")
    return "| " + " | ".join(cells) + " |"


def _package_row_line(
    row: PackageRow, config: Configuration, hardware_by_machine: dict, ratings: dict, rating_failed: bool, machine_names: list[str]
) -> str:
    if row.package is None:
        return _no_recommendation_row_line(row, ratings, rating_failed, machine_names)
    package = row.package
    cells = [
        package.base_model_hf_repo,
        _stars_cell(ratings.get(row.base_model_hf_repo), rating_failed),
        row.packager,
        package.quantization or "–",
        package.format,
        f"{_weights_gib(package):.2f}",
        _context_cell(package),
    ]
    cells += [
        _fit_cell(package, row.base_model, config.machines[name], hardware_by_machine.get(name)) for name in machine_names
    ]
    installed = "; ".join(f"{name}: {_installed_cell(package, hardware_by_machine.get(name))}" for name in machine_names)
    cells.append(installed)
    cells.append(_speed_cell(package, machine_names, hardware_by_machine))
    cells.append(package.provenance)
    cells.append(package.observed_at.date().isoformat())
    cells.append(f"+{row.variants} more" if row.variants else "–")
    return "| " + " | ".join(cells) + " |"


def _render_packages_table(rows: list[PackageRow], config: Configuration, hardware_by_machine: dict, ratings: dict, rating_failed: bool) -> str:
    machine_names = list(config.machines)
    lines = [_packages_table_header(machine_names)]
    for row in rows:
        lines.append(_package_row_line(row, config, hardware_by_machine, ratings, rating_failed, machine_names))
    return "\n".join(lines)


# --- entry point -------------------------------------------------------------------------------


def build_document(
    config: Configuration,
    snapshot: Snapshot,
    hardware_by_machine: dict[str, HardwareSnapshot | None],
    rendered_at: datetime,
    rating_source: RatingSource | None = None,
) -> str:
    """Render `snapshot` and every machine's hardware profile into the Markdown package view.

    Pure: no I/O, no clock read (`rendered_at` is passed in). `hardware_by_machine` maps every
    `config.machines` key to its `HardwareSnapshot` or `None` (no profile measured yet). See
    CONTRACTS.md, "Render (AP5)", for the header format, eligibility, selection and cell rules.
    """
    base_model_by_repo = {bm.hf_repo: bm for bm in snapshot.base_models}
    eligible = _eligible_packages(snapshot.packages)
    groups = _ordered_groups(config, eligible)
    ratings, rating_error = _collect_ratings(rating_source, _unique_repos_with_rows(groups))
    rating_failed = rating_error is not None
    rows = _build_rows(groups, base_model_by_repo, config, hardware_by_machine)

    sections = [
        format_header_line(snapshot.run_at, rendered_at),
        "",
        *_summary_lines(snapshot, rendered_at, rating_failed, rating_error),
        "",
        "## Areas",
        "",
        _render_areas_table(snapshot.areas),
        "",
        "## Machines",
        "",
        _render_machines_table(config, hardware_by_machine, rendered_at),
        "",
        "## Packages",
        "",
        _render_packages_table(rows, config, hardware_by_machine, ratings, rating_failed),
        "",
    ]
    return "\n".join(sections)
