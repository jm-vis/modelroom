"""Tests for the task Hugging Face gives a model (`hub_pipeline_tag`) and the evidence for `kind` (`kind_source`).

The rules live in `modelroom/catalog_schema.py`; the last tests read the shipped `catalog.json` against them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelroom.catalog import Catalog, CatalogError, load_catalog

from fixture_support import FIXTURES
from test_catalog import _CARD_READ, _document, _row


# --- the task Hugging Face gives a model (`hub_pipeline_tag`) and the evidence for `kind` ---
# One broken row breaks one rule; everything else in it is valid (rows of `catalog_v2_valid.json`).


def _load(tmp_path: Path, *rows: dict) -> Catalog:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_document(models=list(rows))), encoding="utf-8")
    return load_catalog(path)


def _evidence_row(**fields) -> dict:
    """Row 0 of the fixture (text, task `text-generation`) with the given fields changed."""
    return _row(0, **fields)


def test_a_vision_row_with_a_matching_task_and_no_kind_source_loads(tmp_path):
    catalog = _load(tmp_path, _evidence_row(hub_pipeline_tag="image-text-to-text", kind="vision", kind_source=None))
    model = catalog.model_for("acme/Nova-7B-2512")
    assert (model.hub_pipeline_tag, model.kind, model.kind_source) == ("image-text-to-text", "vision", None)


def test_a_task_that_contradicts_kind_is_rejected_with_both_kinds_named(tmp_path):
    with pytest.raises(CatalogError, match="acme/Nova-7B-2512") as caught:
        _load(tmp_path, _evidence_row(hub_pipeline_tag="text-generation", kind="vision", kind_source=None))
    assert "vision" in str(caught.value) and "text" in str(caught.value)


def test_a_matching_task_with_a_kind_source_is_rejected(tmp_path):
    """The task already decides `kind`; naming an evidence stage as well is a second, unneeded claim."""
    with pytest.raises(CatalogError, match="kind_source") as caught:
        _load(tmp_path, _evidence_row(hub_pipeline_tag="text-generation", kind="text", kind_source="tags"))
    message = str(caught.value)
    for named in ("acme/Nova-7B-2512", "kind_source 'tags'", "hub_pipeline_tag 'text-generation'", "'text'"):
        assert named in message, (named, message)


@pytest.mark.parametrize("source", ["tags", "base_model", "config", "decision"])
def test_a_row_without_a_task_loads_with_each_evidence_stage(tmp_path, source):
    fields = {"hub_pipeline_tag": None, "kind": "vision", "kind_source": source}
    if source == "decision":
        fields["note"] = "Kind vision from the model card, checked 2026-10-01."
    assert _load(tmp_path, _evidence_row(**fields)).model_for("acme/Nova-7B-2512").kind_source == source


def test_a_row_without_a_task_and_without_a_kind_source_is_rejected(tmp_path):
    with pytest.raises(CatalogError, match="kind_source"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=None, kind="vision", kind_source=None))


def test_a_decision_without_a_note_is_rejected(tmp_path):
    with pytest.raises(CatalogError, match="note") as caught:
        _load(tmp_path, _evidence_row(hub_pipeline_tag=None, kind="vision", kind_source="decision", note=None))
    message = str(caught.value)
    for named in ("acme/Nova-7B-2512", "kind_source 'decision'", "kind 'vision'", "note"):
        assert named in message, (named, message)


@pytest.mark.parametrize("source", ["pipeline_tag", "", "Tags", "guess"])
def test_a_kind_source_outside_the_four_values_is_rejected(tmp_path, source):
    with pytest.raises(CatalogError, match="kind_source"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=None, kind="vision", kind_source=source))


@pytest.mark.parametrize(("task", "kind"), [("any-to-any", "vision"), ("translation", "text")])
def test_a_task_outside_the_kind_table_is_kept_verbatim_when_a_kind_source_is_named(tmp_path, task, kind):
    catalog = _load(tmp_path, _evidence_row(hub_pipeline_tag=task, kind=kind, kind_source="tags"))
    assert catalog.model_for("acme/Nova-7B-2512").hub_pipeline_tag == task


@pytest.mark.parametrize(("task", "kind"), [("any-to-any", "vision"), ("translation", "text")])
def test_a_task_outside_the_kind_table_without_a_kind_source_is_rejected(tmp_path, task, kind):
    with pytest.raises(CatalogError, match="kind_source"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=task, kind=kind, kind_source=None))


@pytest.mark.parametrize("task", ["Text-Generation", "text generation", "text_generation", "-text", "x" * 200, "a" * 65])
def test_a_task_that_breaks_the_pattern_or_the_length_is_rejected(tmp_path, task):
    with pytest.raises(CatalogError, match="hub_pipeline_tag"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=task, kind="text", kind_source="tags"))


def test_a_task_of_sixty_four_characters_loads(tmp_path):
    task = "a" * 64
    catalog = _load(tmp_path, _evidence_row(hub_pipeline_tag=task, kind="text", kind_source="tags"))
    assert catalog.model_for("acme/Nova-7B-2512").hub_pipeline_tag == task


# The fifteen table values, repeated here on purpose: the shipped rows are checked against this
# copy, never against the constant the schema itself uses.
TABLE_KIND_OF_TASK = {
    "text-generation": "text",
    "image-text-to-text": "vision",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
    "automatic-speech-recognition": "speech-recognition",
    "text-to-speech": "speech-synthesis",
    "text-to-audio": "audio",
    "audio-to-audio": "audio",
    "audio-text-to-text": "audio",
    "text-to-image": "image-generation",
    "image-to-image": "image-generation",
    "text-to-video": "video-generation",
    "image-to-video": "video-generation",
    "image-text-to-video": "video-generation",
    "video-to-video": "video-generation",
}
# The rows of the shipped catalog whose task is not in the table, as the one filling run measured
# them (2026-10-03): task, evidence stage and note. `decision` marks the seven Mistral repos whose
# model card was read on 2026-10-01; the note keeps any older sentence in front of it.
OFF_TABLE_ROWS: dict[str, tuple[str | None, str, str | None]] = {
    "mistralai/Ministral-3-14B-Instruct-2512": (None, "decision", _CARD_READ),
    "mistralai/Ministral-3-8B-Instruct-2512": (None, "decision", _CARD_READ),
    "mistralai/Ministral-3-3B-Instruct-2512": (None, "decision", _CARD_READ),
    "mistralai/Mistral-Small-4-119B-2603": (None, "decision", _CARD_READ),
    "mistralai/Devstral-Small-2-24B-Instruct-2512": (
        None,
        "decision",
        "No Ollama mapping as of 2026-09-25: the Ollama library's devstral:24b is the 2505 build, a different model. "
        + _CARD_READ,
    ),
    "mistralai/Magistral-Small-2509": (
        None,
        "decision",
        "No Ollama mapping as of 2026-09-25: the Ollama library's magistral:24b is the 2506 build, a different model. "
        + _CARD_READ,
    ),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": (
        None,
        "decision",
        "No successor as of 2026-09-25: the newer Mistral Small 4 card compares against Mistral Small 3 but names no "
        "specific replaced model. " + _CARD_READ,
    ),
    "google/gemma-4-12B-it": ("any-to-any", "tags", None),
    "google/gemma-4-E4B-it": ("any-to-any", "tags", None),
    "tencent/Hy-MT2-30B-A3B": (
        "translation",
        "tags",
        "No row for the publisher's larger current model Hy4-preview (780B) as of 2026-09-25: no GGUF build of a "
        "listed account.",
    ),
}
DECISION_ONLY_REPOS = frozenset(
    {
        "mistralai/Ministral-3-14B-Instruct-2512",
        "mistralai/Ministral-3-8B-Instruct-2512",
        "mistralai/Ministral-3-3B-Instruct-2512",
        "mistralai/Mistral-Small-4-119B-2603",
        "mistralai/Devstral-Small-2-24B-Instruct-2512",
        "mistralai/Magistral-Small-2509",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    }
)


def test_the_shipped_catalog_has_a_task_or_an_evidence_stage_on_every_one_of_its_rows():
    rows = [model for family in load_catalog().families for model in family.models]
    for model in rows:
        if model.hf_repo in OFF_TABLE_ROWS:
            task, source, note = OFF_TABLE_ROWS[model.hf_repo]
            assert (model.hub_pipeline_tag, model.kind_source, model.note) == (task, source, note), model.hf_repo
        elif model.kind_source is None:
            assert model.hub_pipeline_tag in TABLE_KIND_OF_TASK, model.hf_repo
            assert model.kind == TABLE_KIND_OF_TASK[model.hub_pipeline_tag], model.hf_repo
        else:
            # A row added by a later publication, kind from a stage of scripts/catalog_scan.py
            # (`tags`, `base_model`, `config`); the task is missing or outside the table.
            assert model.kind_source in {"tags", "base_model", "config"}, model.hf_repo
            assert model.hub_pipeline_tag is None or model.hub_pipeline_tag not in TABLE_KIND_OF_TASK, model.hf_repo
            assert model.kind is not None, model.hf_repo


def test_only_the_seven_listed_repos_of_the_shipped_catalog_carry_a_decision():
    rows = [model for family in load_catalog().families for model in family.models]
    decided = {model.hf_repo for model in rows if model.kind_source == "decision"}
    assert decided == DECISION_ONLY_REPOS
    assert len(OFF_TABLE_ROWS) == 10


def test_a_schema_1_row_reads_with_no_task_and_no_kind_source_and_no_rule_applies():
    rows = [model for family in load_catalog(FIXTURES / "catalog_excerpt.toml").families for model in family.models]
    assert len(rows) == 3
    for model in rows:
        assert (model.hub_pipeline_tag, model.kind_source, model.kind) == (None, None, "text"), model.hf_repo
