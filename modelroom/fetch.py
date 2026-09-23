"""Orchestrates one `fetch` run: every configured base model, both sources, one merged Snapshot.

The one place that decides what happens when a base model's `parameters_b` cannot be read this
run (`safetensors.total` missing or the publisher repo unreachable): fall back to the previous
snapshot's value for that `hf_repo` when one exists, else `None` -- `parameters_b` is
`float | None` (F11), `None` meaning "not measured this run, and no previous reading exists",
never a fabricated placeholder that could be mistaken for a real one. Documented in
CONTRACTS.md.

F9: once this run's request budget is exhausted, it stops fetching entirely -- a base model
this run never even started keeps its previous `BaseModelSpec` unchanged (or an unknown one
when there is no previous snapshot), and every area it would have needed is recorded
`incomplete` with `"budget exhausted before this area was started"`, rather than being silently
skipped or, worse, overwritten with a fresh but empty reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import BaseModelConfig, Configuration, package_targets
from .contracts import Architecture, BaseModelSpec, Package, Snapshot
from .fetch_types import AreaOutcome
from .hf import BaseModelMeta, fetch_base_model_meta, fetch_hf_area, target_owners
from .http import DEFAULT_REQUEST_BUDGET, BudgetedTransport, RedirectingTransport, RequestBudget, Transport
from .ollama import fetch_ollama_area
from .quantization import package_identity_key
from .state import merge_snapshot

_BUDGET_EXHAUSTED_MESSAGE = "budget exhausted before this area was started"


@dataclass(frozen=True)
class FetchResult:
    snapshot: Snapshot
    request_used: int
    request_budget: int


def run_fetch(
    config: Configuration,
    transport: Transport,
    run_at: datetime,
    old_snapshot: Snapshot | None,
    budget: int | RequestBudget = DEFAULT_REQUEST_BUDGET,
) -> FetchResult:
    """Fetch every configured base model on both sources and merge the result against `old_snapshot`.

    `budget` is either a limit (a fresh budget for this run, the default as before) or a
    `RequestBudget` the caller already spent on (e.g. its search): the run then books against
    that same count and stops where the shared budget ends. `request_used`/`request_budget` in
    the result are the shared budget's totals.
    """
    shared = budget if isinstance(budget, RequestBudget) else RequestBudget(budget)
    budgeted = BudgetedTransport(transport, shared)
    # R5: RedirectingTransport is the *outer* layer -- every hop of a redirect chain arrives at
    # budgeted as its own separate call, so each hop is checked and booked against the run's
    # request budget before it is made, failure paths included, rather than only afterwards.
    redirecting = RedirectingTransport(budgeted)
    old_base_models_by_repo = {bm.hf_repo: bm for bm in (old_snapshot.base_models if old_snapshot else [])}
    # F6: every package identity this project has ever seen, so a fetcher can carry an existing
    # Approval forward when it reassembles the same package this run.
    previous_by_key: dict[tuple[str, str, str], Package] = {
        package_identity_key(p): p for p in (old_snapshot.packages if old_snapshot else [])
    }

    base_models: list[BaseModelSpec] = []
    area_outcomes: list[AreaOutcome] = []

    for family in config.families:
        for base_model_config in family.base_models:
            # R4: checked before this base model is even started, not only after the previous
            # one finished -- a budget that is already exhausted (or exhausted by a request made
            # earlier in this very base model's own areas) must never let fetch_base_model_meta
            # or a fetcher run at all; each of those would otherwise see BudgetedTransport raise
            # BudgetExhaustedError and quietly resolve to an "unknown"/empty result instead of
            # never having been attempted.
            if budgeted.remaining <= 0:
                base_model = _carried_over_base_model_spec(base_model_config, old_base_models_by_repo)
                base_models.append(base_model)
                area_outcomes.extend(_budget_exhausted_areas(config, base_model_config, base_model))
                continue

            meta = fetch_base_model_meta(redirecting, base_model_config.hf_repo)
            base_model = _build_base_model_spec(base_model_config, meta, old_base_models_by_repo)
            base_models.append(base_model)

            # AP9-B: one area per owner of this base model's package target set, never one per
            # target -- `state.merge_snapshot` identifies an area by `(source, base_model,
            # owner)` and, on `complete`, deactivates every package of that area it does not
            # contain, so two targets of the same owner reported as two areas would deactivate
            # each other's packages.
            targets = package_targets(config, base_model_config)
            for owner in target_owners(targets):
                if budgeted.remaining <= 0:
                    area_outcomes.append(_budget_exhausted_hf_area(base_model, owner))
                    continue
                area_outcomes.append(
                    fetch_hf_area(redirecting, base_model, owner, targets, run_at, previous_by_key)
                )

            if base_model.ollama_base and base_model.ollama_tag:
                if budgeted.remaining <= 0:
                    area_outcomes.append(_budget_exhausted_ollama_area(base_model))
                else:
                    area_outcomes.append(fetch_ollama_area(redirecting, base_model, run_at, previous_by_key))

    snapshot = merge_snapshot(old_snapshot, run_at, area_outcomes, base_models)
    return FetchResult(snapshot=snapshot, request_used=shared.used, request_budget=shared.limit)


def _build_base_model_spec(
    base_model_config: BaseModelConfig, meta: BaseModelMeta, old_base_models_by_repo: dict[str, BaseModelSpec]
) -> BaseModelSpec:
    parameters_b = meta.parameters_b
    if parameters_b is None:
        old = old_base_models_by_repo.get(base_model_config.hf_repo)
        parameters_b = old.parameters_b if old is not None else None
    return BaseModelSpec(
        hf_repo=base_model_config.hf_repo,
        repo_aliases=base_model_config.repo_aliases,
        ollama_base=base_model_config.ollama_base,
        ollama_tag=base_model_config.ollama_tag,
        publisher=meta.publisher,
        parameters_b=parameters_b,
        architecture=meta.architecture,
    )


def _carried_over_base_model_spec(
    base_model_config: BaseModelConfig, old_base_models_by_repo: dict[str, BaseModelSpec]
) -> BaseModelSpec:
    """F9: the spec for a base model this run never even started reading.

    Its previous `BaseModelSpec` is kept byte-for-byte when one exists; otherwise an unknown
    one is built straight from the configuration, with `parameters_b=None` (F11) rather than a
    fabricated number.
    """
    old = old_base_models_by_repo.get(base_model_config.hf_repo)
    if old is not None:
        return old
    return BaseModelSpec(
        hf_repo=base_model_config.hf_repo,
        repo_aliases=base_model_config.repo_aliases,
        ollama_base=base_model_config.ollama_base,
        ollama_tag=base_model_config.ollama_tag,
        publisher=base_model_config.hf_repo.split("/", 1)[0],
        parameters_b=None,
        architecture=Architecture(source_repo=base_model_config.hf_repo, source_revision=None, kind="unknown"),
    )


def _budget_exhausted_hf_area(base_model: BaseModelSpec, owner: str) -> AreaOutcome:
    """R4: the `incomplete` outcome for one Hugging Face area the budget ran out before starting."""
    return AreaOutcome(
        source="huggingface",
        base_model_hf_repo=base_model.hf_repo,
        packager=owner,
        status="incomplete",
        error=_BUDGET_EXHAUSTED_MESSAGE,
        packages=[],
    )


def _budget_exhausted_ollama_area(base_model: BaseModelSpec) -> AreaOutcome:
    """R4: the `incomplete` outcome for the Ollama area the budget ran out before starting."""
    return AreaOutcome(
        source="ollama",
        base_model_hf_repo=base_model.hf_repo,
        packager=None,
        status="incomplete",
        error=_BUDGET_EXHAUSTED_MESSAGE,
        packages=[],
    )


def _budget_exhausted_areas(
    config: Configuration, base_model_config: BaseModelConfig, base_model: BaseModelSpec
) -> list[AreaOutcome]:
    """F9: one `incomplete` `AreaOutcome` per area this run would have attempted, had the
    budget not already run out before this base model was ever reached.
    """
    outcomes = [
        _budget_exhausted_hf_area(base_model, owner)
        for owner in target_owners(package_targets(config, base_model_config))
    ]
    if base_model_config.ollama_base and base_model_config.ollama_tag:
        outcomes.append(_budget_exhausted_ollama_area(base_model))
    return outcomes
