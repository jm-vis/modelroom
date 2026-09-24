"""Tests for modelroom.document: the render document, the one object all three outputs read."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from modelroom.document import (
    DOCUMENT_SCHEMA_VERSION,
    MachineRanking,
    RankedEntry,
    RenderDocument,
    SetAsideEntry,
)
from modelroom.examples import EXAMPLES

_NOW = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)


def _ranked(**overrides) -> dict:
    return {**EXAMPLES["RankedEntry"], **overrides}


def _machine(**overrides) -> dict:
    return {**EXAMPLES["MachineRanking"], **overrides}


def _document(**overrides) -> dict:
    return {**EXAMPLES["RenderDocument"], **overrides}


def test_schema_version_is_checked():
    with pytest.raises(ValidationError):
        RenderDocument.model_validate(_document(schema_version=DOCUMENT_SCHEMA_VERSION + 1))


def test_timestamps_must_be_aware_utc():
    with pytest.raises(ValidationError):
        RenderDocument.model_validate(_document(rendered_at=_NOW.replace(tzinfo=None).isoformat()))


def test_group_zero_carries_its_measurement():
    with pytest.raises(ValidationError) as excinfo:
        RankedEntry.model_validate(_ranked(measurement_group=0, measurement_id=None, speed_tps=None))
    assert "measurement_group" in str(excinfo.value)


def test_group_one_carries_no_measurement():
    with pytest.raises(ValidationError):
        RankedEntry.model_validate(_ranked(measurement_group=1))


def test_a_ranked_entry_is_never_unknown_or_too_tight():
    fit = {**EXAMPLES["RankedEntry"]["fit"], "fit_class": "unknown", "reason": "not covered"}
    with pytest.raises(ValidationError):
        RankedEntry.model_validate(_ranked(fit=fit))


def test_a_ranked_machine_carries_its_profile():
    with pytest.raises(ValidationError) as excinfo:
        MachineRanking.model_validate(_machine(status="ranked", profile=None))
    assert "profile" in str(excinfo.value)


def test_a_machine_without_a_profile_carries_a_reason_and_no_entries():
    with pytest.raises(ValidationError):
        MachineRanking.model_validate(_machine(status="legacy", profile=None, reason=None))
    without = MachineRanking.model_validate(
        _machine(status="no_profile", profile=None, reason="no hardware profile yet", ranked=[],
                 ranked_total=0, not_covered=[], too_tight=[])
    )
    assert without.ranked == []
    with pytest.raises(ValidationError):
        MachineRanking.model_validate(
            _machine(status="no_profile", profile=None, reason="no hardware profile yet")
        )


def test_ranks_are_one_two_three():
    second = _ranked(rank=2, measurement_group=1, measurement_id=None, speed_tps=None, note=EXAMPLES["Note"])
    with pytest.raises(ValidationError) as excinfo:
        MachineRanking.model_validate(_machine(ranked=[_ranked(rank=1), _ranked(rank=1)], ranked_total=2))
    assert "rank must count" in str(excinfo.value)
    MachineRanking.model_validate(_machine(ranked=[_ranked(rank=1), second], ranked_total=2))


def test_ranked_total_is_never_below_what_is_shown():
    with pytest.raises(ValidationError) as excinfo:
        MachineRanking.model_validate(_machine(ranked_total=0))
    assert "ranked_total" in str(excinfo.value)


def test_a_set_aside_entry_states_its_reason():
    with pytest.raises(ValidationError):
        SetAsideEntry.model_validate({**EXAMPLES["SetAsideEntry"], "reason": ""})


def test_machine_names_are_unique_in_one_document():
    twice = [EXAMPLES["MachineRanking"], EXAMPLES["MachineRanking"]]
    with pytest.raises(ValidationError):
        RenderDocument.model_validate(_document(machines=twice))


def test_the_example_document_round_trips_as_json():
    document = RenderDocument.model_validate(EXAMPLES["RenderDocument"])
    assert RenderDocument.model_validate(document.model_dump(mode="json")) == document
