"""Tests for the task Hugging Face gives a model (`hub_pipeline_tag`) and the evidence for `kind` (`kind_source`).

The rules live in `modelroom/catalog_schema.py`; the last tests read the shipped `catalog.json` against them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelroom.catalog import Catalog, CatalogError, load_catalog

from fixture_support import FIXTURES
from test_catalog import _CARD_READ, _document, _maintainer_kind_note, _row


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


def _maintainer_form():
    """`MAINTAINER_KIND_NOTE` from `scripts/`; the test is skipped where `scripts/` is not shipped (the sdist), as in test_catalog."""
    form = _maintainer_kind_note()
    if form is None:
        pytest.skip("scripts/ is not shipped in the sdist; the form of the maintainer's note is not at hand")
    return form


def _is_maintainer_decision(model, form) -> bool:
    return model.kind_source == "decision" and form.fullmatch(model.note or "") is not None


def _row_problems(rows, off_table, form) -> list[str]:
    """The repos among `rows` that carry none of the four forms a shipped row may have (empty: every row is sound).

    1. a row listed in `off_table`, as the filling run measured it; 2. a task of the table and no `kind_source`;
    3. a maintainer's decision from `--complete`: `decision`, the note in the form of `MAINTAINER_KIND_NOTE`, no task of
    the table, and `kind` the kind that note names; 4. a stage of the scanner (`tags`, `base_model`, `config`) with
    a task that is missing or outside the table."""
    kinds = set(TABLE_KIND_OF_TASK.values())
    problems = []
    for model in rows:
        if model.hf_repo in off_table:
            task, source, note = off_table[model.hf_repo]
            sound = (model.hub_pipeline_tag, model.kind_source, model.note) == (task, source, note)
        elif model.kind_source is None:
            sound = model.hub_pipeline_tag in TABLE_KIND_OF_TASK and model.kind == TABLE_KIND_OF_TASK[model.hub_pipeline_tag]
        elif _is_maintainer_decision(model, form):
            named = form.fullmatch(model.note)  # `MAINTAINER_KIND_NOTE` itself: the group `kind` names the kind of the note
            sound = (
                named is not None and named.group("kind") in kinds and model.kind == named.group("kind")
                and (model.hub_pipeline_tag is None or model.hub_pipeline_tag not in TABLE_KIND_OF_TASK)
            )
        else:
            # A row added by a later publication, kind from a stage of scripts/catalog_scan.py
            # (`tags`, `base_model`, `config`); the task is missing or outside the table.
            sound = (
                model.kind_source in {"tags", "base_model", "config"}
                and (model.hub_pipeline_tag is None or model.hub_pipeline_tag not in TABLE_KIND_OF_TASK)
                and model.kind is not None
            )
        if not sound:
            problems.append(model.hf_repo)
    return problems


def _decision_problems(rows, decision_only, form) -> list[str]:
    """What is wrong with the rows that carry a `decision`: the card decisions (`decision`, no note of the maintainer's
    form) must be exactly `decision_only`; a maintainer's decision may not be one of those, and none of those may have the form."""
    card = {m.hf_repo for m in rows if m.kind_source == "decision" and not _is_maintainer_decision(m, form)}
    maintainer = {m.hf_repo for m in rows if _is_maintainer_decision(m, form)}
    problems = [f"card decisions differ from the listed repos: {sorted(card ^ decision_only)}"] if card != decision_only else []
    problems += [f"{repo} is listed as a card decision but carries the maintainer's note" for repo in sorted(maintainer & decision_only)]
    return problems


def _shipped_rows():
    return [model for family in load_catalog().families for model in family.models]


def test_the_shipped_catalog_has_a_task_or_an_evidence_stage_on_every_one_of_its_rows():
    assert _row_problems(_shipped_rows(), OFF_TABLE_ROWS, _maintainer_form()) == []


def test_only_the_seven_listed_repos_carry_a_card_decision_and_the_other_decisions_are_the_maintainers():
    """The seven Mistral repos are the card decisions (`decision`, a note without the form of `--complete`); every other
    `decision` row is a kind the maintainer set (note in the form of `MAINTAINER_KIND_NOTE`) and is none of the seven."""
    form = _maintainer_form()
    assert _decision_problems(_shipped_rows(), DECISION_ONLY_REPOS, form) == []
    assert len(OFF_TABLE_ROWS) == 10


# One row of the fixture, read as the shipped catalog would hold it: the first row `--complete` made (a kind the
# maintainer set, no task at all), a note with no kind, and one of the "seven" that carries the maintainer's note.
MAINTAINER_NOTE = "Kind video-generation set by the maintainer on 2026-10-27."


def _fixture_rows(tmp_path, **fields):
    return [m for family in _load(tmp_path, _evidence_row(**fields)).families for m in family.models]


def test_a_row_with_a_maintainers_kind_is_sound_where_its_note_names_its_kind(tmp_path):
    form = _maintainer_form()
    rows = _fixture_rows(tmp_path, hub_pipeline_tag=None, kind="video-generation", kind_source="decision", note=MAINTAINER_NOTE)
    assert _row_problems(rows, {}, form) == []
    assert _decision_problems(rows, frozenset(), form) == []


def test_a_note_that_names_no_kind_is_a_card_decision_and_makes_the_suite_red(tmp_path):
    form = _maintainer_form()
    banana = "Kind banana set by the maintainer on 2026-10-27."
    rows = _fixture_rows(tmp_path, hub_pipeline_tag=None, kind="video-generation", kind_source="decision", note=banana)
    assert _row_problems(rows, {}, form) == ["acme/Nova-7B-2512"]
    assert _decision_problems(rows, frozenset(), form) != []


def test_a_maintainers_note_that_names_another_kind_than_the_row_makes_the_suite_red(tmp_path):
    form = _maintainer_form()
    rows = _fixture_rows(tmp_path, hub_pipeline_tag=None, kind="vision", kind_source="decision", note=MAINTAINER_NOTE)
    assert _row_problems(rows, {}, form) == ["acme/Nova-7B-2512"]


def test_one_of_the_seven_with_the_maintainers_note_makes_the_suite_red(tmp_path):
    form = _maintainer_form()
    rows = _fixture_rows(tmp_path, hub_pipeline_tag=None, kind="video-generation", kind_source="decision", note=MAINTAINER_NOTE)
    problems = _decision_problems(rows, frozenset({"acme/Nova-7B-2512"}), form)
    assert any("carries the maintainer's note" in problem for problem in problems), problems


def test_a_maintainers_note_on_a_row_with_a_task_of_the_table_does_not_load(tmp_path):
    """The schema, not this suite, keeps a kind of the maintainer off a row whose task already gives the kind."""
    with pytest.raises(CatalogError, match="kind_source"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag="text-to-video", kind="video-generation", kind_source="decision", note=MAINTAINER_NOTE))


def test_a_schema_1_row_reads_with_no_task_and_no_kind_source_and_no_rule_applies():
    rows = [model for family in load_catalog(FIXTURES / "catalog_excerpt.toml").families for model in family.models]
    assert len(rows) == 3
    for model in rows:
        assert (model.hub_pipeline_tag, model.kind_source, model.kind) == (None, None, "text"), model.hf_repo
