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

from .config import BaseModelConfig, Configuration
from .contracts import Architecture, BaseModelSpec, Package, Snapshot
from .fetch_types import AreaOutcome
from .hf import BaseModelMeta, candidate_owners, fetch_base_model_meta, fetch_hf_area
from .http import DEFAULT_REQUEST_BUDGET, BudgetedTransport, Transport
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
    budget: int = DEFAULT_REQUEST_BUDGET,
) -> FetchResult:
    """Fetch every configured base model on both sources and merge the result against `old_snapshot`."""
    budgeted = BudgetedTransport(transport, budget)
    old_base_models_by_repo = {bm.hf_repo: bm for bm in (old_snapshot.base_models if old_snapshot else [])}
    # F6: every package identity this project has ever seen, so a fetcher can carry an existing
    # Approval forward when it reassembles the same package this run.
    previous_by_key: dict[tuple[str, str, str], Package] = {
        package_identity_key(p): p for p in (old_snapshot.packages if old_snapshot else [])
    }

    base_models: list[BaseModelSpec] = []
    area_outcomes: list[AreaOutcome] = []
    budget_exhausted = False

    for family in config.families:
        for base_model_config in family.base_models:
            if budget_exhausted:
                base_model = _carried_over_base_model_spec(base_model_config, old_base_models_by_repo)
                base_models.append(base_model)
                area_outcomes.extend(_budget_exhausted_areas(config, base_model_config, base_model))
                continue

            meta = fetch_base_model_meta(budgeted, base_model_config.hf_repo)
            base_model = _build_base_model_spec(base_model_config, meta, old_base_models_by_repo)
            base_models.append(base_model)

            for owner in candidate_owners(config, base_model):
                area_outcomes.append(fetch_hf_area(budgeted, base_model, owner, run_at, previous_by_key))

            if base_model.ollama_base and base_model.ollama_tag:
                area_outcomes.append(fetch_ollama_area(budgeted, base_model, run_at, previous_by_key))

            if budgeted.remaining <= 0:
                budget_exhausted = True

    snapshot = merge_snapshot(old_snapshot, run_at, area_outcomes, base_models)
    return FetchResult(snapshot=snapshot, request_used=budgeted.used, request_budget=budget)


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


def _budget_exhausted_areas(
    config: Configuration, base_model_config: BaseModelConfig, base_model: BaseModelSpec
) -> list[AreaOutcome]:
    """F9: one `incomplete` `AreaOutcome` per area this run would have attempted, had the
    budget not already run out before this base model was ever reached.
    """
    outcomes = [
        AreaOutcome(
            source="huggingface",
            base_model_hf_repo=base_model.hf_repo,
            packager=owner,
            status="incomplete",
            error=_BUDGET_EXHAUSTED_MESSAGE,
            packages=[],
        )
        for owner in candidate_owners(config, base_model)
    ]
    if base_model_config.ollama_base and base_model_config.ollama_tag:
        outcomes.append(
            AreaOutcome(
                source="ollama",
                base_model_hf_repo=base_model.hf_repo,
                packager=None,
                status="incomplete",
                error=_BUDGET_EXHAUSTED_MESSAGE,
                packages=[],
            )
        )
    return outcomes
