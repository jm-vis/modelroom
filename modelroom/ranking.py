"""The ranking rule: one device's packages ordered, a total order (ADR 0009).

Among the packages that fit: graphics memory before system memory, every model that is not
`legacy` before a `legacy` one, then the largest model first and, within one model, the least
compression that fits; a measurement only breaks a tie between two builds of the same compression.
`ample`, `fits` and `tight` rank alike -- each still says its word in its row.

Pure and deterministic: the same packages, fits, measurements and model facts give the same ranking
in any input order. It sorts facts; it recommends nothing and rates no model family against another
(CONTRACTS.md, "Ranking rule").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from .contracts import Fit, Package
from .measurements import MeasurementRecord, Scenario
from .quantization import QUANT_ORDER, package_identity_key

TOP_LIMIT = 10
_RULE_KEY = (
    "graphics memory before system memory, then every model that is not legacy before a legacy one, "
    "then the larger model (parameter count), then the least compression, then measured before "
    "unmeasured (a valid comparable measurement), then larger weights, then package identity"
)
RANKING_RULE = f"{_RULE_KEY}; the first {TOP_LIMIT} rows, then every measured package the rule leaves out"
RANKING_RULE_PER_MODEL = (
    f"{_RULE_KEY}; one row per model, its first package under this rule, then every measured package "
    "the rule leaves out"
)
# The facts a row note names for the sentence it gains from the rule (`row_sentence`); the words of
# the result (`ranking_words`) read the case of a block from them.
BEST_FACT = "ranking.best_per_model"
SHOWN_FACT = "ranking.shown_for_measurement"

_WEIGHT_ROLES = ("weights", "weights_shard")
_GRAPHICS_MODES = ("gpu", "gpu_split")
# 16- and 32-bit builds are one step of compression; within it the smaller weights come first
# (GF 2026-09-28, question 5). `F16` stands first of the three in `QUANT_ORDER`.
_FULL_PRECISION = ("F16", "BF16", "F32")
_FULL_STEP = QUANT_ORDER.index("F16")


@dataclass(frozen=True)
class ModelFacts:
    """What the rule knows about one base model: its parameter count, and whether it is `legacy`."""

    parameters_b: float | None
    legacy: bool


_NOTHING_KNOWN = ModelFacts(None, False)


@dataclass(frozen=True)
class RankedPackage:
    """One ranked package; `measurement_note` says why a measurement of it did not count.

    Set only when a measurement would count in group 0 but for `requests` alone (a measurement
    runs one request): `measured with 1 request, ranking assumes N`. Never written anywhere but
    the render document's row note (decided 2026-09-25). `rank` is the package's place under the
    rule; the row number of a document is `document.RankedEntry.rank`.
    """

    rank: int
    package: Package
    fit: Fit
    measurement_group: int
    measurement: MeasurementRecord | None
    measurement_note: str | None = None
    # Why a measurement of exactly this package counts nowhere here for another reason than `requests`:
    # another context, not comparable, not valid (`other_measurement_note`, Codex F3). The package stays
    # in sight; the number is not shown as comparable.
    measured_other: str | None = None


@dataclass(frozen=True)
class SetAside:
    """A package outside the ranking, with the reason shown next to it."""

    package: Package
    fit: Fit
    reason: str


@dataclass
class Ranking:
    scenario: Scenario
    # No `rule` field any more: the document names the rule of its case (`render.build_render_document`),
    # and a field that always said the rule of one case was read by nobody (Diff-Review N6).
    ranked: list[RankedPackage] = field(default_factory=list)
    not_covered: list[SetAside] = field(default_factory=list)
    no_fit: list[SetAside] = field(default_factory=list)

    @property
    def top(self) -> list[RankedPackage]:
        return self.ranked[:TOP_LIMIT]


@dataclass(frozen=True)
class ShownRow:
    """One row a result shows: why it is there, and its place under the rule.

    `why` is `rule` for a row the rule itself brings (the first ten, or one per model) and
    `measured` for a measured package of this machine the rule leaves out, appended behind them.
    `place` of `of`: the place among every ranked package of the machine, or, one row per model,
    among the ranked packages of that model.
    """

    ranked: RankedPackage
    why: Literal["rule", "measured"]
    place: int
    of: int


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


def _tokens(context: int) -> str:
    return f"{context // 1024}k" if context % 1024 == 0 else f"{context} tokens"


def other_measurement_note(package: Package, measurements: list[MeasurementRecord], scenario: Scenario) -> str | None:
    """Why the newest measurement of `package` counts nowhere here, where neither group 0 nor the
    `requests` note covers it: `None` when there is no measurement of this content at all."""
    if group_zero_measurement(package, measurements, scenario) is not None:
        return None
    if measurement_note(package, measurements, scenario) is not None:
        return None
    mine = [record for record in measurements if _matches(record, package)]
    if not mine:
        return None
    newest = max(mine, key=lambda r: (r.measured_at, r.measurement_id))
    if newest.protocol != "v1" or newest.validity != "valid":
        return f"measured, not valid ({newest.validity_reason or newest.validity})"
    if not newest.comparable:
        return f"measured, not comparable ({newest.comparable_reason})"
    measured, ranked = newest.scenario.context_requested, scenario.context_requested
    return f"measured at context {_tokens(measured)}, this ranking's is {_tokens(ranked)}"


def _compression_key(quantization: str | None, weights: int) -> tuple:
    """Least compression first; an unknown quantization behind every known one; 16/32 bit one step."""
    if quantization not in QUANT_ORDER:
        return (1, 0, 0)
    if quantization in _FULL_PRECISION:
        return (0, -_FULL_STEP, weights)
    return (0, -QUANT_ORDER.index(quantization), 0)


def _sort_key(package: Package, fit: Fit, measurement: MeasurementRecord | None, facts: ModelFacts) -> tuple:
    """The seven steps of ADR 0009, in order; the speed of a measurement is no part of it."""
    weights = sum(f.size_bytes for f in package.files if f.role in _WEIGHT_ROLES)
    size = facts.parameters_b
    return (
        0 if fit.mode in _GRAPHICS_MODES else 1,
        1 if facts.legacy else 0,
        (0, -size) if size is not None else (1, 0.0),
        _compression_key(package.quantization, weights),
        0 if measurement is not None else 1,
        -weights,
        package_identity_key(package),
    )


def rank_packages(
    entries: list[tuple[Package, Fit]],
    measurements: list[MeasurementRecord],
    scenario: Scenario,
    *,
    models: Mapping[str, ModelFacts] | None = None,
) -> Ranking:
    """Rank one device's packages; `measurements` are that device's own measurement records.

    `models` maps a `base_model_hf_repo` to what the rule knows about it; a model it does not name
    (or `models is None`) has an unknown parameter count and is not `legacy`. `ample`/`fits`/`tight`
    are ranked by the tuple key; `unknown` goes to `not_covered` with the fit's reason; `no_fit` is
    listed apart in its own list and never ranked. Every computed fit must belong to the ranking's
    context and requests (a fit for another context or another number of requests is a caller error).
    """
    known = models or {}
    ranking = Ranking(scenario=scenario)
    rankable: list[tuple[tuple, Package, Fit, MeasurementRecord | None]] = []
    for package, fit in entries:
        if fit.fit_class == "unknown":
            ranking.not_covered.append(SetAside(package, fit, fit.reason or "not covered"))
            continue
        _check_fit_belongs(fit, scenario)
        if fit.fit_class == "no_fit":
            ranking.no_fit.append(SetAside(package, fit, "does not fit this machine"))
            continue
        measurement = group_zero_measurement(package, measurements, scenario)
        facts = known.get(package.base_model_hf_repo, _NOTHING_KNOWN)
        rankable.append((_sort_key(package, fit, measurement, facts), package, fit, measurement))
    rankable.sort(key=lambda item: item[0])
    ranking.ranked = [
        RankedPackage(
            rank,
            package,
            fit,
            0 if measured is not None else 1,
            measured,
            measurement_note(package, measurements, scenario),
            other_measurement_note(package, measurements, scenario),
        )
        for rank, (_, package, fit, measured) in enumerate(rankable, start=1)
    ]
    ranking.not_covered.sort(key=lambda s: package_identity_key(s.package))
    ranking.no_fit.sort(key=lambda s: package_identity_key(s.package))
    return ranking


def best_per_model(ranking: Ranking) -> list[RankedPackage]:
    """The best package of every model: its first under the whole key, in the order of the key."""
    seen: set[str] = set()
    best = []
    for ranked in ranking.ranked:
        if ranked.package.base_model_hf_repo not in seen:
            seen.add(ranked.package.base_model_hf_repo)
            best.append(ranked)
    return best


def _was_measured(ranked: RankedPackage) -> bool:
    """Any measurement of exactly this package on this machine: counting (group 0), failing on
    `requests` alone, or counting nowhere here for another reason (Codex F3)."""
    return ranked.measurement is not None or ranked.measurement_note is not None or ranked.measured_other is not None


def _places(ranking: Ranking, per_model: bool) -> dict[int, tuple[int, int]]:
    """Each ranked package's place, by its rank: among all, or among the packages of its model."""
    if not per_model:
        return {ranked.rank: (ranked.rank, len(ranking.ranked)) for ranked in ranking.ranked}
    counts: dict[str, int] = {}
    for ranked in ranking.ranked:
        counts[ranked.package.base_model_hf_repo] = counts.get(ranked.package.base_model_hf_repo, 0) + 1
    seen: dict[str, int] = {}
    places = {}
    for ranked in ranking.ranked:
        base = ranked.package.base_model_hf_repo
        seen[base] = seen.get(base, 0) + 1
        places[ranked.rank] = (seen[base], counts[base])
    return places


def shown_rows(ranking: Ranking, per_model: bool) -> list[ShownRow]:
    """The rows of a result: the rule's first, then every measured package the rule leaves out.

    The rule's rows are the first ten (one model) or the best package of every model (several);
    no row of the rule moves for a measurement. A measured package of this machine that is no
    such row is appended behind them, in the order of the key, with its place (ADR 0009).
    """
    places = _places(ranking, per_model)
    rule = best_per_model(ranking) if per_model else ranking.top
    ruled = {ranked.rank for ranked in rule}
    rows = [ShownRow(ranked, "rule", *places[ranked.rank]) for ranked in rule]
    rows += [
        ShownRow(ranked, "measured", *places[ranked.rank])
        for ranked in ranking.ranked
        if ranked.rank not in ruled and _was_measured(ranked)
    ]
    return rows


def row_sentence(row: ShownRow, per_model: bool) -> tuple[str, str] | None:
    """The sentence a row note gains from the rule, and the fact it names; `None` where there is none."""
    if row.why == "measured":
        where = f"place {row.place} of this model's {row.of}" if per_model else f"place {row.place}"
        return f"Shown for its measurement; the rule puts it at {where}.", SHOWN_FACT
    if not per_model:
        return None
    if row.of == 1:
        return "The only package of this model that fits.", BEST_FACT
    return f"Best of {row.of} packages of this model that fit.", BEST_FACT


def _check_fit_belongs(fit: Fit, scenario: Scenario) -> None:
    if fit.context != scenario.context_requested:
        raise ValueError(f"fit context {fit.context} differs from the ranking's {scenario.context_requested}")
    if fit.requests != scenario.requests:
        raise ValueError(f"fit requests {fit.requests} differ from the ranking's {scenario.requests}")
