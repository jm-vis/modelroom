"""Tests for `scripts/catalog_fill_pipeline_tag.py`, the one-off filling of the shipped catalog's task column.

Every answer comes from a `FixtureTransport`; nothing here opens a socket. The script is removed
after its review (the answers it read are kept in the fixtures named below), and so is this file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from modelroom.catalog import CATALOG_FILE, load_catalog
from modelroom.http import FixtureTransport, Response

from fixture_support import FIXTURES, envelope_response, json_response

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("catalog_fill_pipeline_tag", REPO / "scripts" / "catalog_fill_pipeline_tag.py")
fill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fill)

API = "https://huggingface.co/api/models/"
SENTENCE = "Kind vision from the model card, checked 2026-10-01."
# The seven repos the script may mark `decision`, repeated here on purpose (not imported).
DECISION_REPOS = (
    "mistralai/Ministral-3-14B-Instruct-2512",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Mistral-Small-4-119B-2603",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "mistralai/Magistral-Small-2509",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
)
REAL_ANSWER = "hf_mistralai_ministral_3_8b_instruct_2512_model.json"
TASK_OF_KIND = {"text": "text-generation", "vision": "image-text-to-text", "embedding": "feature-extraction"}


def _dump(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _answer(body: object, status: int = 200, headers: dict[str, str] | None = None) -> Response:
    return Response(status=status, headers=headers or {}, body=json.dumps(body).encode("utf-8"))


def _shipped() -> dict:
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


def _blank_catalog() -> dict:
    """The shipped catalog as it was before the filling: no task, no evidence stage, no sentence."""
    data = _shipped()
    for row in data["models"]:
        row["hub_pipeline_tag"] = None
        row["kind_source"] = None
        if row["note"] is not None and SENTENCE.replace("vision", row["kind"]) in row["note"]:
            rest = row["note"].replace(SENTENCE.replace("vision", row["kind"]), "").strip()
            row["note"] = rest or None
    return data


def _hub_answers(filled: dict) -> dict[str, dict]:
    """What the Hub says about each row of `filled`: its task as the top field, its tags as evidence."""
    answers = {}
    for row in filled["models"]:
        body: dict = {"id": row["hf_repo"], "tags": ["transformers", "safetensors"]}
        if row["hub_pipeline_tag"] is not None:
            body["pipeline_tag"] = row["hub_pipeline_tag"]
        if row["kind_source"] == "tags":
            body["tags"].append(TASK_OF_KIND[row["kind"]])
        answers[row["hf_repo"]] = body
    return answers


def _setup(tmp_path: Path, catalog: dict | None = None, answers: dict[str, object] | None = None):
    """A catalog file (the blank shipped one by default), a transport over `answers` and the file's bytes."""
    catalog = _blank_catalog() if catalog is None else catalog
    path = tmp_path / "catalog.json"
    path.write_bytes(_dump(catalog))
    bodies = _hub_answers(_shipped()) if answers is None else answers
    mapping = {
        ("GET", API + repo): body if isinstance(body, Response) else _answer(body) for repo, body in bodies.items()
    }
    return path, FixtureTransport(mapping), path.read_bytes()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(catalog: dict, repo: str) -> dict:
    return next(row for row in catalog["models"] if row["hf_repo"] == repo)


def _assert_refused(path: Path, before: str, problems: list[str], *needles: str) -> None:
    assert problems, "the run was expected to fail"
    for needle in needles:
        assert any(needle in problem for problem in problems), (needle, problems)
    assert _sha(path) == before
    assert [entry.name for entry in path.parent.iterdir()] == ["catalog.json"]


def test_the_fill_asks_the_hub_once_per_row_and_reproduces_the_shipped_catalog_byte_for_byte(tmp_path):
    path, transport, _ = _setup(tmp_path)

    assert fill.fill(path, transport) == []

    assert len(transport.calls) == 62
    assert sorted(url for _, url in transport.calls) == sorted(API + row["hf_repo"] for row in _shipped()["models"])
    assert path.read_bytes() == CATALOG_FILE.read_bytes()
    assert set(transport.limits) == {1024 * 1024}


def test_a_table_task_wins_over_a_tag_of_another_kind(tmp_path):
    repo = "Qwen/Qwen3.6-27B"
    blank = _blank_catalog()
    kind = _row(blank, repo)["kind"]
    answers = _hub_answers(_shipped())
    answers[repo] = {"pipeline_tag": TASK_OF_KIND[kind], "tags": ["image-text-to-text", "text-generation"]}
    path, transport, _ = _setup(tmp_path, blank, answers)

    assert fill.fill(path, transport) == []

    row = _row(json.loads(path.read_text(encoding="utf-8")), repo)
    assert (row["hub_pipeline_tag"], row["kind_source"]) == (TASK_OF_KIND[kind], None)


def test_a_table_task_that_contradicts_kind_is_an_error_and_kind_is_never_changed(tmp_path):
    repo = "Qwen/Qwen3-Embedding-8B"
    answers = _hub_answers(_shipped())
    answers[repo] = {"pipeline_tag": "text-generation", "tags": []}
    path, transport, _ = _setup(tmp_path, answers=answers)
    before = _sha(path)

    problems = fill.fill(path, transport)

    _assert_refused(path, before, problems, repo, "embedding")


def test_a_row_without_a_task_whose_table_tags_all_give_its_kind_is_marked_tags(tmp_path):
    repo = "google/gemma-4-12B-it"
    answers = _hub_answers(_shipped())
    answers[repo] = {"pipeline_tag": "any-to-any", "tags": ["transformers", "image-text-to-text"]}
    path, transport, _ = _setup(tmp_path, answers=answers)

    assert fill.fill(path, transport) == []

    row = _row(json.loads(path.read_text(encoding="utf-8")), repo)
    assert (row["hub_pipeline_tag"], row["kind_source"], row["kind"]) == ("any-to-any", "tags", "vision")


def test_a_row_without_a_task_whose_table_tags_name_two_kinds_is_an_error_outside_the_decision_list(tmp_path):
    repo = "google/gemma-4-12B-it"
    answers = _hub_answers(_shipped())
    answers[repo] = {"pipeline_tag": "any-to-any", "tags": ["image-text-to-text", "text-generation"]}
    path, transport, _ = _setup(tmp_path, answers=answers)
    before = _sha(path)

    problems = fill.fill(path, transport)

    _assert_refused(path, before, problems, repo)


def test_a_listed_repo_without_evidence_gets_the_fixed_decision_sentence_dated_2026_10_01(tmp_path):
    repo = DECISION_REPOS[0]
    path, transport, _ = _setup(tmp_path)

    assert fill.fill(path, transport) == []

    row = _row(json.loads(path.read_text(encoding="utf-8")), repo)
    assert (row["hub_pipeline_tag"], row["kind_source"], row["kind"]) == (None, "decision", "vision")
    assert row["note"] == SENTENCE


def test_a_decision_sentence_is_appended_after_an_existing_note_with_one_space(tmp_path):
    repo = "mistralai/Magistral-Small-2509"
    blank = _blank_catalog()
    assert _row(blank, repo)["note"] is not None
    old = _row(blank, repo)["note"]
    path, transport, _ = _setup(tmp_path, blank)

    assert fill.fill(path, transport) == []

    assert _row(json.loads(path.read_text(encoding="utf-8")), repo)["note"] == f"{old} {SENTENCE}"


def test_a_listed_repo_with_a_table_task_gets_no_sentence(tmp_path):
    repo = DECISION_REPOS[1]
    answers = _hub_answers(_shipped())
    answers[repo] = {"pipeline_tag": "image-text-to-text", "tags": []}
    path, transport, _ = _setup(tmp_path, answers=answers)

    assert fill.fill(path, transport) == []

    row = _row(json.loads(path.read_text(encoding="utf-8")), repo)
    assert (row["hub_pipeline_tag"], row["kind_source"], row["note"]) == ("image-text-to-text", None, None)


def test_an_unlisted_repo_without_evidence_is_an_error_that_names_the_row(tmp_path):
    repo = "Qwen/Qwen3.5-9B"
    answers = _hub_answers(_shipped())
    answers[repo] = {"tags": ["transformers"]}
    path, transport, _ = _setup(tmp_path, answers=answers)
    before = _sha(path)

    problems = fill.fill(path, transport)

    _assert_refused(path, before, problems, repo)


def test_a_listed_repo_whose_table_tags_name_one_other_kind_is_an_error_and_gets_no_sentence(tmp_path):
    """The decision list only covers a repo with no table evidence; one tag of another kind is evidence against `kind`."""
    repo = DECISION_REPOS[0]
    answers = _hub_answers(_shipped())
    answers[repo] = {"tags": ["transformers", "text-generation"]}
    path, transport, _ = _setup(tmp_path, answers=answers)
    before = _sha(path)

    problems = fill.fill(path, transport)

    _assert_refused(path, before, problems, repo)
    assert "decision" not in path.read_text(encoding="utf-8")


def test_a_note_of_exactly_three_hundred_characters_after_the_sentence_is_written_and_loads(tmp_path):
    repo = DECISION_REPOS[0]
    blank = _blank_catalog()
    _row(blank, repo)["note"] = "n" * (300 - len(SENTENCE) - 1)
    path, transport, _ = _setup(tmp_path, blank)

    assert fill.fill(path, transport) == []

    assert len(load_catalog(path).model_for(repo).note) == 300


def test_a_note_of_three_hundred_and_one_characters_after_the_sentence_is_refused_and_nothing_is_written(tmp_path):
    repo = DECISION_REPOS[0]
    blank = _blank_catalog()
    _row(blank, repo)["note"] = "n" * (300 - len(SENTENCE))
    path, transport, _ = _setup(tmp_path, blank)
    before = _sha(path)

    problems = fill.fill(path, transport)

    _assert_refused(path, before, problems, repo)


def test_a_second_run_over_the_filled_file_changes_no_byte_and_keeps_the_sentence_once(tmp_path):
    path, transport, _ = _setup(tmp_path)
    assert fill.fill(path, transport) == []
    first = _sha(path)

    assert fill.fill(path, transport) == []

    assert _sha(path) == first
    text = path.read_text(encoding="utf-8")
    assert text.count("checked 2026-10-01.") == 7


def test_only_the_task_the_stage_and_the_decision_notes_change_and_no_kind_is_touched(tmp_path):
    path, transport, _ = _setup(tmp_path)
    blank = _blank_catalog()

    assert fill.fill(path, transport) == []

    filled = json.loads(path.read_text(encoding="utf-8"))
    for before, after in zip(blank["models"], filled["models"], strict=True):
        changed = {key for key in before if before[key] != after[key]}
        assert changed <= {"hub_pipeline_tag", "kind_source", "note"}, before["hf_repo"]
        if "note" in changed:
            assert after["kind_source"] == "decision", before["hf_repo"]
    assert {key: value for key, value in blank.items() if key != "models"} == {
        key: value for key, value in filled.items() if key != "models"
    }


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(Response(status=404, body=b'{"error": "Repository not found"}'), id="404"),
        pytest.param(envelope_response("hf_unsloth_does_not_exist_model.json"), id="401-as-the-hub-answers-a-missing-repo"),
        pytest.param(
            Response(status=307, headers={"Location": "https://huggingface.co/api/models/other/Renamed"}), id="redirect"
        ),
        pytest.param(_answer({"pipeline_tag": "Text-Generation"}), id="task-against-the-pattern"),
        pytest.param(_answer({"pipeline_tag": 7}), id="task-not-a-string"),
        pytest.param(Response(status=200, body=b"not json"), id="no-json"),
        pytest.param(Response(status=200, body=b"[]"), id="not-an-object"),
        pytest.param(Response(status=200, body=b" " * (1024 * 1024 + 1)), id="longer-than-a-mebibyte"),
    ],
)
def test_an_answer_that_is_no_usable_model_record_fails_the_run_and_names_the_row(tmp_path, response):
    repo = "Qwen/Qwen3.5-9B"
    answers = _hub_answers(_shipped())
    answers[repo] = response
    path, transport, _ = _setup(tmp_path, answers=answers)
    before = _sha(path)

    problems = fill.fill(path, transport)

    _assert_refused(path, before, problems, repo)


def test_a_redirect_is_never_followed_and_the_row_after_it_is_still_asked(tmp_path):
    repo = "Qwen/Qwen3.5-9B"
    target = API + "other/Renamed"
    answers = _hub_answers(_shipped())
    answers[repo] = Response(status=307, headers={"Location": target})
    path, transport, _ = _setup(tmp_path, answers=answers)
    before = _sha(path)

    problems = fill.fill(path, transport)

    _assert_refused(path, before, problems, repo, "redirect")
    assert ("GET", target) not in transport.calls
    assert transport.calls.count(("GET", API + repo)) == 1
    assert len(transport.calls) == 62


def test_a_429_stops_at_once_names_the_retry_after_and_asks_no_further_row(tmp_path):
    repo = "mistralai/Ministral-3-14B-Instruct-2512"
    answers = _hub_answers(_shipped())
    answers[repo] = Response(status=429, headers={"Retry-After": "120"}, body=b"{}")
    path, transport, _ = _setup(tmp_path, answers=answers)
    before = _sha(path)
    position = [row["hf_repo"] for row in _shipped()["models"]].index(repo)

    problems = fill.fill(path, transport)

    _assert_refused(path, before, problems, repo, "120")
    assert len(transport.calls) == position + 1


def test_the_budget_ending_after_sixty_one_requests_fails_the_run_and_names_the_row(tmp_path):
    path, transport, _ = _setup(tmp_path)
    before = _sha(path)

    problems = fill.fill(path, transport, budget=61)

    _assert_refused(path, before, problems, _shipped()["models"][61]["hf_repo"])
    assert len(transport.calls) == 61


def test_a_filled_catalog_the_loader_refuses_at_the_end_is_one_problem_and_leaves_the_file_and_the_folder_alone(
    tmp_path, capsys
):
    """The script checks the rows it fills, not the whole document; the final load is the last guard."""
    blank = _blank_catalog()
    _row(blank, "Qwen/Qwen3.5-9B")["source"] = "http://huggingface.co/Qwen/Qwen3.5-9B"
    path, transport, _ = _setup(tmp_path, blank)
    before = _sha(path)

    problems = fill.fill(path, transport)

    assert len(problems) == 1
    assert problems[0].startswith("the filled catalog does not load: ")
    assert len(transport.calls) == 62
    assert _sha(path) == before
    assert [entry.name for entry in path.parent.iterdir()] == ["catalog.json"]

    assert fill.main(["--catalog", str(path)], transport=transport) == 1
    assert "the filled catalog does not load" in capsys.readouterr().err
    assert _sha(path) == before
    assert [entry.name for entry in path.parent.iterdir()] == ["catalog.json"]


def test_the_main_function_exits_one_on_a_problem_and_zero_on_success(tmp_path, capsys):
    path, transport, _ = _setup(tmp_path)

    assert fill.main(["--catalog", str(path)], transport=transport) == 0
    missing = {("GET", API + row["hf_repo"]): Response(status=404) for row in _shipped()["models"]}
    assert fill.main(["--catalog", str(path)], transport=FixtureTransport(missing)) == 1
    assert "error" in capsys.readouterr().err.lower()


def test_the_real_answer_of_a_repo_without_a_task_gives_none():
    body = json.loads((FIXTURES / REAL_ANSWER).read_text(encoding="utf-8"))

    assert "pipeline_tag" not in body
    assert fill.task_of(body) is None


def test_the_task_is_the_top_field_and_never_the_card_data_one():
    body = json.loads(json_response("hf_deepseek_r1_qwen3_8b_model.json").body)
    body["cardData"]["pipeline_tag"] = "image-text-to-text"

    assert fill.task_of(body) == "text-generation"
