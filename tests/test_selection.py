"""Tests for modelroom.selection: the result of a guided run covers only what this run chose."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from modelroom.config import Configuration
from modelroom.contracts import Area, BaseModelSpec, Package, Snapshot
from modelroom.examples import EXAMPLES
from modelroom.ranking import ModelFacts
from modelroom.selection import RunSelection, model_facts, narrow

NOW = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)
NOVA, VEGA = "acme/Nova-8B", "acme/Vega-9B"


def _config() -> Configuration:
    machine = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0}
    return Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [
                {"name": "nova", "base_models": [{"hf_repo": NOVA}]},
                {"name": "vega", "base_models": [{"hf_repo": VEGA}]},
            ],
            "packagers": ["packager"],
            "publishers": ["acme"],
            "machines": {"laptop": {**machine, "writer": True}, "server": {**machine, "writer": False}},
            "paths": {"state": "//models/state", "markdown": "//models/models.md"},
        }
    )


def _base(repo: str) -> BaseModelSpec:
    payload = copy.deepcopy(EXAMPLES["BaseModelSpec"])
    payload["hf_repo"] = repo
    payload["architecture"]["source_repo"] = repo
    return BaseModelSpec.model_validate(payload)


def _package(base: str) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["base_model_hf_repo"] = base
    payload["repo"] = f"packager/{base.split('/')[1]}-GGUF"
    return Package.model_validate(payload)


def _snapshot() -> Snapshot:
    areas = [Area(source="huggingface", base_model_hf_repo=base, packager="packager", status="complete") for base in (NOVA, VEGA)]
    return Snapshot(
        schema_version=1,
        run_at=NOW,
        areas=areas,
        base_models=[_base(NOVA), _base(VEGA)],
        packages=[_package(NOVA), _package(VEGA)],
    )


SELECTION = RunSelection(machines=("laptop",), models=(VEGA,), release={VEGA: "legacy"}, parameters={VEGA: 9.15})


def test_narrow_keeps_only_the_chosen_models_and_machines():
    config, snapshot = narrow(_config(), _snapshot(), SELECTION)

    assert list(config.machines) == ["laptop"]
    assert [spec.hf_repo for family in config.families for spec in family.base_models] == [VEGA]
    assert [spec.hf_repo for spec in snapshot.base_models] == [VEGA]
    assert {package.base_model_hf_repo for package in snapshot.packages} == {VEGA}
    assert [area.base_model_hf_repo for area in snapshot.areas] == [VEGA]
    assert snapshot.run_at == NOW


def test_the_configuration_keeps_every_family_and_machine():
    """Nothing of the selection is persisted: the original objects are the same after the narrowing."""
    original, snapshot = _config(), _snapshot()

    narrow(original, snapshot, SELECTION)

    assert list(original.machines) == ["laptop", "server"]
    assert [family.name for family in original.families] == ["nova", "vega"]
    assert len(snapshot.packages) == 2


def test_a_run_that_chose_no_model_shows_none_of_the_folder():
    """An explicitly empty choice is a choice (Codex F1): no model and no machine of the folder is shown.
    Only `modelroom render` without a selection shows the whole folder."""
    config, snapshot = narrow(_config(), _snapshot(), RunSelection((), (), {}, {}))

    assert (config.families, config.machines) == ([], {})
    assert (snapshot.base_models, snapshot.packages, snapshot.areas) == ([], [], [])
    assert snapshot.run_at == NOW


def test_model_facts_carry_the_lists_numbers():
    selection = RunSelection(
        machines=("laptop",),
        models=(NOVA, VEGA),
        release={NOVA: "latest", VEGA: "legacy"},
        parameters={NOVA: None, VEGA: 9.15},
    )

    assert model_facts(selection) == {NOVA: ModelFacts(None, False), VEGA: ModelFacts(9.15, True)}
