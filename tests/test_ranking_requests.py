"""The ranking rule and parallel requests (decided 2026-09-25).

A measurement always runs one request, so it counts in measured group 0 only for a ranking of one
request. Where a measurement would count but for `requests` alone, the ranked package carries the
reason (`RankedPackage.measurement_note`), and the render hands it on as the row's note -- never
into the measurement file.
"""

from __future__ import annotations

import pytest

from modelroom.contracts import Fit
from modelroom.measurements import Scenario
from modelroom.ranking import group_zero_measurement, rank_packages
from modelroom.render import _ranked_entries

from test_ranking import Q4, Q5, _fit, _measurement


def _scenario(requests: int = 1, context: int = 8192) -> Scenario:
    return Scenario(
        context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=requests
    )


def _fit_for(requests: int, fit_class: str = "perfect", context: int = 8192) -> Fit:
    return _fit(fit_class, context).model_copy(update={"requests": requests})


def test_a_measurement_counts_in_group_zero_only_for_the_same_requests():
    measurement = _measurement(Q4, 40.0)
    assert group_zero_measurement(Q4, [measurement], _scenario(1)) == measurement
    assert group_zero_measurement(Q4, [measurement], _scenario(2)) is None


def test_a_ranking_of_more_requests_puts_the_measured_package_in_group_one_and_says_why():
    ranking = rank_packages([(Q4, _fit_for(3))], [_measurement(Q4, 40.0)], _scenario(3))

    ranked = ranking.ranked[0]
    assert (ranked.measurement_group, ranked.measurement) == (1, None)
    assert ranked.measurement_note == "measured with 1 request, ranking assumes 3"


def test_a_ranking_of_one_request_has_no_note():
    ranking = rank_packages([(Q4, _fit_for(1))], [_measurement(Q4, 40.0)], _scenario(1))
    assert ranking.ranked[0].measurement_group == 0
    assert ranking.ranked[0].measurement_note is None


def test_no_note_when_the_measurement_fails_on_the_context_as_well():
    """Only a measurement that fails on `requests` alone gives the note."""
    ranking = rank_packages([(Q4, _fit_for(3, context=16384))], [_measurement(Q4, 40.0)], _scenario(3, 16384))
    assert ranking.ranked[0].measurement_note is None


def test_no_note_for_a_package_nobody_measured():
    ranking = rank_packages([(Q5, _fit_for(3))], [_measurement(Q4, 40.0)], _scenario(3))
    assert ranking.ranked[0].measurement_note is None


def test_a_fit_for_other_requests_than_the_ranking_is_a_caller_error():
    with pytest.raises(ValueError, match="requests"):
        rank_packages([(Q4, _fit_for(2))], [], _scenario(3))


def test_the_render_hands_the_reason_on_as_the_row_note():
    ranking = rank_packages([(Q4, _fit_for(3))], [_measurement(Q4, 40.0)], _scenario(3))

    entry = _ranked_entries(ranking, shared_memory=False)[0]

    assert entry.note.text == "measured with 1 request, ranking assumes 3"
    assert entry.note.code == "measured_with_one_request"
    assert entry.note.subject == "package"
    assert entry.measurement_group == 1 and entry.speed_tps is None
