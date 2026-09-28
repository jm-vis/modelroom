"""What one guided run chose: its machines and its models, held in memory and never persisted.

The result of a guided run covers only this selection (AP9-UX B): the card, the tables and
`docs/models.md/json` name the machines step 1 of this run chose and the models step 2 chose. The
configuration of the folder keeps every family and every machine, and the fetch still fetches them
all; a later `modelroom render` without a run shows the whole folder again (GF 2026-09-28, question 4).

`narrow` is pure: it builds a filtered `Configuration` and a filtered `Snapshot` through their own
constructors, so every validator runs, and changes neither of the objects it was given.
`model_facts` hands the rule of ADR 0009 the numbers the list of step 2 showed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .catalog import Age
from .config import Configuration
from .contracts import Snapshot
from .ranking import ModelFacts

# The note of a run that chose no model (70 characters).
NO_MODEL_CHOSEN = "this run chose no model; the result shows the models this folder holds"


@dataclass(frozen=True)
class RunSelection:
    """The `[machines.<name>]` and the base models of one run, with the release and size step 2 showed.

    An empty `machines` or `models` is no filter: the result then shows what the folder holds.
    `parameters` carries the very number the list of step 2 ordered by (`parameters.parameter_count`),
    so the list and the result never order differently; `release` is `SearchHit.age` of that run.
    """

    machines: tuple[str, ...]
    models: tuple[str, ...]
    release: Mapping[str, Age] = field(default_factory=dict)
    parameters: Mapping[str, float | None] = field(default_factory=dict)


def narrow(config: Configuration, snapshot: Snapshot, selection: RunSelection) -> tuple[Configuration, Snapshot]:
    """The configuration and the snapshot of this run's selection alone; nothing is written."""
    return narrow_config(config, selection), narrow_snapshot(snapshot, selection)


def narrow_config(config: Configuration, selection: RunSelection) -> Configuration:
    """`config` with only the chosen machines and the families' chosen base models (validated anew)."""
    data = config.model_dump()
    if selection.machines:
        data["machines"] = {name: machine for name, machine in data["machines"].items() if name in selection.machines}
    if selection.models:
        families = []
        for family in data["families"]:
            kept = [spec for spec in family["base_models"] if spec["hf_repo"] in selection.models]
            if kept:
                families.append({**family, "base_models": kept})
        data["families"] = families
    return Configuration.model_validate(data)


def narrow_snapshot(snapshot: Snapshot, selection: RunSelection) -> Snapshot:
    """`snapshot` with only the chosen base models, their packages and their areas (validated anew)."""
    if not selection.models:
        return snapshot
    chosen = set(selection.models)
    data = snapshot.model_dump()
    data["base_models"] = [spec for spec in data["base_models"] if spec["hf_repo"] in chosen]
    data["packages"] = [package for package in data["packages"] if package["base_model_hf_repo"] in chosen]
    data["areas"] = [area for area in data["areas"] if area["base_model_hf_repo"] in chosen]
    return Snapshot.model_validate(data)


def model_facts(selection: RunSelection) -> dict[str, ModelFacts]:
    """What the rule knows of each chosen model: the list's size, and whether step 2 called it `legacy`."""
    return {
        model: ModelFacts(selection.parameters.get(model), selection.release.get(model) == "legacy")
        for model in selection.models
    }


__all__ = ["NO_MODEL_CHOSEN", "RunSelection", "model_facts", "narrow", "narrow_config", "narrow_snapshot"]
