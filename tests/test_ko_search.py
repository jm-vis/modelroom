"""Contract tests of AP9-KO P0, part 3: the direct find in the search contract (V8).

Nothing of the search itself is tested here (that is P2): only the shapes it will fill.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modelroom.guided_contracts import DIRECT_FIND_MAX, DIRECT_FIND_SOURCES, UNKNOWN, DirectLogEntry, SearchHit


def _direct(**changes) -> dict:
    data = {
        "repo": "acme/Nova-9B",
        "publisher_status": "publisher",
        "base_model": None,
        "base_model_relation": UNKNOWN,
        "resolved": True,
        "unresolved_reason": None,
        "resolved_base_model": "acme/Nova-9B",
        "repo_created_at": "2026-10-01T10:00:00Z",
        "downloads": 1200,
        "parameters_b": 9.0,
        "license": "apache-2.0",
        "age": "latest",
        "successor": None,
        "release_basis": "stated",
        "ollama": None,
        "found_as": "direct",
    }
    data.update(changes)
    return data


# --- V8: the direct find is a hit of its own kind -------------------------------------------------------------------


def test_a_hit_is_a_package_hit_unless_it_says_it_is_a_direct_find():
    from modelroom.examples import EXAMPLES

    assert SearchHit.model_validate(EXAMPLES["SearchHit"]).found_as == "package"
    assert SearchHit.model_validate(_direct()).found_as == "direct"


def test_a_direct_find_is_the_publishers_own_model_and_invents_no_quantization_relation():
    hit = SearchHit.model_validate(_direct())
    assert hit.resolved and hit.resolved_base_model == hit.repo
    assert hit.base_model is None and hit.base_model_relation == UNKNOWN


FAULTY_DIRECT = {
    "another base model": {"resolved_base_model": "acme/Nova-7B"},
    "not resolved": {"resolved": False, "unresolved_reason": "base_model_tag", "resolved_base_model": None},
    "not the publisher's": {"publisher_status": "listed packager"},
    "unknown owner": {"publisher_status": "unknown"},
    "a declared base model": {"base_model": ["acme/Nova-9B"]},
    "a quantized relation": {"base_model_relation": "quantized"},
}


@pytest.mark.parametrize("name", sorted(FAULTY_DIRECT))
def test_a_direct_find_that_is_not_the_publishers_own_model_is_refused(name):
    with pytest.raises(ValidationError):
        SearchHit.model_validate(_direct(**FAULTY_DIRECT[name]))


def test_a_direct_find_is_itself_whatever_the_case_of_its_id():
    """The catalog row may write `acme/Nova-9B`, the Hub answer `Acme/nova-9b`: one repository (ids are case-insensitive)."""
    hit = SearchHit.model_validate(_direct(repo="Acme/nova-9b"))
    assert (hit.repo, hit.resolved_base_model) == ("Acme/nova-9b", "acme/Nova-9B")  # both kept as written


def test_a_direct_find_may_carry_the_ollama_name_of_the_catalog_and_an_age():
    hit = SearchHit.model_validate(_direct(ollama="nova:9b", age="legacy", successor="acme/Nova-9B-2610"))
    assert (hit.ollama, hit.age, hit.successor) == ("nova:9b", "legacy", "acme/Nova-9B-2610")


def test_a_package_hit_keeps_every_rule_it_had():
    from modelroom.examples import EXAMPLES

    broken = dict(EXAMPLES["SearchHit"], base_model=["other/Model"])
    with pytest.raises(ValidationError):
        SearchHit.model_validate(broken)
    with pytest.raises(ValidationError):
        SearchHit.model_validate(dict(EXAMPLES["SearchHit"], found_as="elsewhere"))


# --- V8: what the log says about it ------------------------------------------------------------------------------------


def test_the_direct_find_is_read_at_most_ten_times_per_search_and_by_three_inputs_only():
    assert DIRECT_FIND_MAX == 10
    assert DIRECT_FIND_SOURCES == ("id", "word")  # an empty search makes no direct find


def test_a_log_entry_says_which_model_was_asked_how_and_what_the_hub_answered():
    found = DirectLogEntry.model_validate({"repo": "acme/Nova-9B", "source": "word", "status": "found", "requests": 1, "reason": None})
    assert (found.status, found.requests) == ("found", 1)
    unread = DirectLogEntry.model_validate({"repo": "acme/Nova-9B", "source": "id", "status": "not_read", "requests": 1, "reason": "HTTP 404"})
    assert unread.reason == "HTTP 404"
    cut = DirectLogEntry.model_validate({"repo": "acme/Nova-9B", "source": "word", "status": "budget", "requests": 0, "reason": "request budget ended"})
    assert cut.requests == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "found", "reason": "x"},  # a find has no reason
        {"status": "not_read", "reason": None},  # a failure says why
        {"status": "budget", "reason": None},
        {"status": "budget", "reason": "request budget ended", "requests": 1},  # nothing was asked
        {"status": "found", "requests": 0},  # a find costs its request
        {"source": "empty"},
        {"status": "missing"},
        {"requests": -1},
        {"repo": "Nova"},
    ],
)
def test_a_faulty_log_entry_is_refused(bad):
    data = {"repo": "acme/Nova-9B", "source": "word", "status": "found", "requests": 1, "reason": None, **bad}
    with pytest.raises(ValidationError):
        DirectLogEntry.model_validate(data)
