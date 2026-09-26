"""The ranking rule: one device's packages ordered best fit first, a total order.

Pure and deterministic: the same packages, fits and measurements give the same ranking in any
input order. It sorts facts; it recommends nothing (CONTRACTS.md, "Ranking rule").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import Fit, Package
from .measurements import MeasurementRecord, Scenario
from .quantization import QUANT_ORDER, package_identity_key

TOP_LIMIT = 10
RANKING_RULE = (
    "fit class (perfect, good, marginal), then measured group (valid comparable measurement "
    "first), then measured speed (faster first), then quantization, then larger weights, "
    "then package identity"
)

_FIT_ORDER = {"perfect": 0, "good": 1, "marginal": 2}
_WEIGHT_ROLES = ("weights", "weights_shard")


@dataclass(frozen=True)
class RankedPackage:
    """One ranked package; `measurement_note` says why a measurement of it did not count.

    Set only when a measurement would count in group 0 but for `requests` alone (a measurement
    runs one request): `measured with 1 request, ranking assumes N`. Never written anywhere but
    the render document's row note (decided 2026-09-25).
    """

    rank: int
    package: Package
    fit: Fit
    measurement_group: int
    measurement: MeasurementRecord | None
    measurement_note: str | None = None


@dataclass(frozen=True)
class SetAside:
    """A package outside the ranking, with the reason shown next to it."""

    package: Package
    fit: Fit
    reason: str


@dataclass
class Ranking:
    scenario: Scenario
    rule: str = RANKING_RULE
    ranked: list[RankedPackage] = field(default_factory=list)
    not_covered: list[SetAside] = field(default_factory=list)
    too_tight: list[SetAside] = field(default_factory=list)

    @property
    def top(self) -> list[RankedPackage]:
        return self.ranked[:TOP_LIMIT]


def _matches(record: MeasurementRecord, package: Package) -> bool:
    ref = record.package
    if package.source == "ollama":
        return ref.content_source == "ollama" and ref.ollama_manifest_digest == package.manifest_digest
    weight_digests = {f.digest for f in package.files if f.role in _WEIGHT_ROLES and f.digest}
    return (
        ref.content_source == "huggingface"
        and ref.hf_repo == package.repo
        and ref.hf_revision == package.revision
        and ref.hf_file_digest in weight_digests
    )


def group_zero_measurement(
    package: Package, measurements: list[MeasurementRecord], scenario: Scenario
) -> MeasurementRecord | None:
    """The newest measurement of `package` that counts for this ranking, or `None` (group 1).

    Counts: protocol v1, valid, comparable, the same `context_requested` and the same
    `requests` as the ranking (a measurement always runs one request, so a ranking of more
    requests has no group 0). Ties on `measured_at` break on `measurement_id`.
    """
    counting = [
        record
        for record in _counting_but_requests(package, measurements, scenario)
        if record.scenario.requests == scenario.requests
    ]
    return max(counting, key=lambda r: (r.measured_at, r.measurement_id), default=None)


def _counting_but_requests(
    package: Package, measurements: list[MeasurementRecord], scenario: Scenario
) -> list[MeasurementRecord]:
    """The measurements of `package` that count for this ranking, `requests` not yet asked."""
    return [
        record
        for record in measurements
        if record.protocol == "v1"
        and record.validity == "valid"
        and record.comparable
        and record.scenario.context_requested == scenario.context_requested
        and _matches(record, package)
    ]


def measurement_note(package: Package, measurements: list[MeasurementRecord], scenario: Scenario) -> str | None:
    """Why a measurement of `package` did not count, when `requests` alone is the reason.

    `None` when one counts, or when none would count anyway (another context, another package).
    """
    if group_zero_measurement(package, measurements, scenario) is not None:
        return None
    others = _counting_but_requests(package, measurements, scenario)
    if not others:
        return None
    measured = max(others, key=lambda r: (r.measured_at, r.measurement_id)).scenario.requests
    return f"measured with {measured} request{'s' if measured != 1 else ''}, ranking assumes {scenario.requests}"


def _sort_key(package: Package, fit: Fit, measurement: MeasurementRecord | None) -> tuple:
    quant = package.quantization
    quant_index = QUANT_ORDER.index(quant) if quant in QUANT_ORDER else len(QUANT_ORDER)
    weights = sum(f.size_bytes for f in package.files if f.role in _WEIGHT_ROLES)
    speed = measurement.tps_mean if measurement is not None else 0.0
    return (
        _FIT_ORDER[fit.fit_class],
        0 if measurement is not None else 1,
        -speed,
        quant_index,
        -weights,
        package_identity_key(package),
    )


def rank_packages(
    entries: list[tuple[Package, Fit]], measurements: list[MeasurementRecord], scenario: Scenario
) -> Ranking:
    """Rank one device's packages; `measurements` are that device's own measurement records.

    `perfect`/`good`/`marginal` are ranked by the tuple key; `unknown` goes to `not_covered`
    with the fit's reason; `too_tight` is listed apart and never in the top 10. Every computed
    fit must belong to the ranking's context and requests (a fit for another context or another
    number of requests is a caller error).
    """
    ranking = Ranking(scenario=scenario)
    rankable: list[tuple[tuple, Package, Fit, MeasurementRecord | None]] = []
    for package, fit in entries:
        if fit.fit_class == "unknown":
            ranking.not_covered.append(SetAside(package, fit, fit.reason or "not covered"))
            continue
        _check_fit_belongs(fit, scenario)
        if fit.fit_class == "too_tight":
            ranking.too_tight.append(SetAside(package, fit, "does not fit this machine"))
            continue
        measurement = group_zero_measurement(package, measurements, scenario)
        rankable.append((_sort_key(package, fit, measurement), package, fit, measurement))
    rankable.sort(key=lambda item: item[0])
    ranking.ranked = [
        RankedPackage(
            rank,
            package,
            fit,
            0 if measurement is not None else 1,
            measurement,
            measurement_note(package, measurements, scenario),
        )
        for rank, (_, package, fit, measurement) in enumerate(rankable, start=1)
    ]
    ranking.not_covered.sort(key=lambda s: package_identity_key(s.package))
    ranking.too_tight.sort(key=lambda s: package_identity_key(s.package))
    return ranking


def _check_fit_belongs(fit: Fit, scenario: Scenario) -> None:
    if fit.context != scenario.context_requested:
        raise ValueError(f"fit context {fit.context} differs from the ranking's {scenario.context_requested}")
    if fit.requests != scenario.requests:
        raise ValueError(f"fit requests {fit.requests} differ from the ranking's {scenario.requests}")
