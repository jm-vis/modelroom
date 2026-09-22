"""Orchestrates one `fetch` run: every configured base model, both sources, one merged Snapshot.

The one place that decides what happens when a base model's `parameters_b` cannot be read
this run (`safetensors.total` missing or the publisher repo unreachable): fall back to the
previous snapshot's value for that `hf_repo` when one exists, else an explicit placeholder of
`1.0` (billion) -- never `0.0`, which would fail `BaseModelSpec.parameters_b`'s `gt=0`
constraint, and never a fabricated "real-looking" number. Documented in CONTRACTS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import BaseModelConfig, Configuration
from .contracts import BaseModelSpec, Snapshot
from .fetch_types import AreaOutcome
from .hf import BaseModelMeta, candidate_owners, fetch_base_model_meta, fetch_hf_area
from .http import DEFAULT_REQUEST_BUDGET, BudgetedTransport, Transport
from .ollama import fetch_ollama_area
from .state import merge_snapshot

PARAMETERS_B_PLACEHOLDER = 1.0


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

    base_models: list[BaseModelSpec] = []
    area_outcomes: list[AreaOutcome] = []

    for family in config.families:
        for base_model_config in family.base_models:
            meta = fetch_base_model_meta(budgeted, base_model_config.hf_repo)
            base_model = _build_base_model_spec(base_model_config, meta, old_base_models_by_repo)
            base_models.append(base_model)

            for owner in candidate_owners(config, base_model):
                area_outcomes.append(fetch_hf_area(budgeted, base_model, owner, run_at))

            if base_model.ollama_base and base_model.ollama_tag:
                area_outcomes.append(fetch_ollama_area(budgeted, base_model, run_at))

    snapshot = merge_snapshot(old_snapshot, run_at, area_outcomes, base_models)
    return FetchResult(snapshot=snapshot, request_used=budgeted.used, request_budget=budget)


def _build_base_model_spec(
    base_model_config: BaseModelConfig, meta: BaseModelMeta, old_base_models_by_repo: dict[str, BaseModelSpec]
) -> BaseModelSpec:
    parameters_b = meta.parameters_b
    if parameters_b is None:
        old = old_base_models_by_repo.get(base_model_config.hf_repo)
        parameters_b = old.parameters_b if old is not None else PARAMETERS_B_PLACEHOLDER
    return BaseModelSpec(
        hf_repo=base_model_config.hf_repo,
        repo_aliases=base_model_config.repo_aliases,
        ollama_base=base_model_config.ollama_base,
        ollama_tag=base_model_config.ollama_tag,
        publisher=meta.publisher,
        parameters_b=parameters_b,
        architecture=meta.architecture,
    )
