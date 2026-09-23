"""Tests for modelroom.guided_contracts: search hit, requirement and note shapes."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from modelroom.examples import EXAMPLES
from modelroom.guided_contracts import Note, Requirement, SearchHit
from modelroom.http import BudgetExhaustedError, BudgetedTransport, RequestBudget, Response


def _hit(**changes) -> dict:
    payload = copy.deepcopy(EXAMPLES["SearchHit"])
    payload.update(changes)
    return payload


def _unresolved(**changes) -> dict:
    return _hit(resolved=False, resolved_base_model=None, unresolved_reason="relation_unknown", ollama=None, **changes)


# --- SearchHit -------------------------------------------------------------------------------


def test_a_hit_with_only_a_repo_is_unknown_everywhere_but_must_say_why_it_is_unresolved():
    with pytest.raises(ValidationError):
        SearchHit.model_validate({"repo": "someone/Nova-7B-GGUF"})
    hit = SearchHit.model_validate({"repo": "someone/Nova-7B-GGUF", "unresolved_reason": "publisher_unknown"})
    assert (hit.publisher_status, hit.license, hit.age, hit.base_model_relation) == ("unknown",) * 4


def test_resolved_hit_names_its_base_model_and_no_reason():
    with pytest.raises(ValidationError):
        SearchHit.model_validate(_hit(resolved_base_model=None))
    with pytest.raises(ValidationError):
        SearchHit.model_validate(_hit(unresolved_reason="derivative"))


def test_unresolved_hit_gets_no_ollama_name_and_no_base_model():
    SearchHit.model_validate(_unresolved())
    with pytest.raises(ValidationError):
        SearchHit.model_validate({**_unresolved(), "ollama": "nova:7b"})
    with pytest.raises(ValidationError):
        SearchHit.model_validate({**_unresolved(), "resolved_base_model": "acme/Nova-7B"})


@pytest.mark.parametrize(
    "changes",
    [
        {"base_model": ["acme/Nova-7B", "acme/Other-7B"]},
        {"base_model": ["acme/Other-7B"]},
        {"base_model": None},
        {"base_model_relation": "merge"},
        {"base_model_relation": "unknown"},
    ],
)
def test_resolved_hit_needs_exactly_its_base_model_and_relation_quantized(changes):
    with pytest.raises(ValidationError):
        SearchHit.model_validate(_hit(**changes))


def test_successor_exactly_when_legacy():
    with pytest.raises(ValidationError):
        SearchHit.model_validate(_hit(age="latest"))
    SearchHit.model_validate(_hit(age="latest", successor=None))
    with pytest.raises(ValidationError):
        SearchHit.model_validate(_hit(successor=None))


@pytest.mark.parametrize(
    "changes",
    [
        {"publisher_status": "packager"},
        {"age": "current"},
        {"ollama": "nova"},
        {"repo_created_at": "2026-03-02T10:00:00"},
        {"parameters_b": 0},
    ],
)
def test_search_hit_rejects_words_outside_the_vocabulary_and_bad_values(changes):
    with pytest.raises(ValidationError):
        SearchHit.model_validate(_hit(**changes))


# --- Requirement ------------------------------------------------------------------------------


def test_requirement_kv_total_is_per_request_times_requests():
    payload = copy.deepcopy(EXAMPLES["Requirement"])
    payload["kv_gib_total"] = 1.0
    with pytest.raises(ValidationError, match="kv_gib_total"):
        Requirement.model_validate(payload)


def test_requirement_need_is_fit_v1_formula_with_the_kv_total():
    payload = copy.deepcopy(EXAMPLES["Requirement"])
    payload["need_gib"] = 6.45  # the formula with one request's KV only
    with pytest.raises(ValidationError, match="need_gib"):
        Requirement.model_validate(payload)


def test_requirement_perfect_is_not_reachable_in_cpu_mode():
    payload = copy.deepcopy(EXAMPLES["Requirement"])
    payload["mode"] = "cpu"
    with pytest.raises(ValidationError, match="cpu"):
        Requirement.model_validate(payload)
    payload["perfect_reachable"] = False
    Requirement.model_validate(payload)


def test_requirement_origin_is_always_computed():
    with pytest.raises(ValidationError):
        Requirement.model_validate({**copy.deepcopy(EXAMPLES["Requirement"]), "origin": "measured"})


# --- Note --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "x" * 241},
        {"text": ""},
        {"facts": []},
        {"origin": "guessed"},
        {"code": "Not A Code"},
        {"subject": "user"},
    ],
)
def test_note_rejects_long_text_missing_facts_and_other_words(changes):
    with pytest.raises(ValidationError):
        Note.model_validate({**EXAMPLES["Note"], **changes})


# --- RequestBudget (shared by search, resolution and fetch) -----------------------------------


def test_two_transports_on_one_budget_book_against_the_same_count():
    budget = RequestBudget(3)
    first = BudgetedTransport(lambda method, url, headers=None: Response(status=200, headers={}, body=b""), budget)
    second = BudgetedTransport(lambda method, url, headers=None: Response(status=200, headers={}, body=b""), budget)
    first("GET", "https://huggingface.co/a")
    second("GET", "https://huggingface.co/b")
    second("GET", "https://huggingface.co/c")
    assert (budget.used, budget.remaining, first.used, second.remaining) == (3, 0, 3, 0)
    with pytest.raises(BudgetExhaustedError):
        first("GET", "https://huggingface.co/d")


def test_an_integer_budget_still_makes_a_fresh_budget():
    transport = BudgetedTransport(lambda method, url, headers=None: Response(status=200, headers={}, body=b""), 1)
    transport("GET", "https://huggingface.co/a")
    assert transport.used == 1 and transport.remaining == 0
