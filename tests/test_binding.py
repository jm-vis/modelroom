"""Tests for modelroom.binding: the pointer file and the profile takeover rule."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from modelroom.binding import (
    GuidedPointer,
    KnownProfile,
    PointerFileError,
    read_pointer,
    resolve_profile_target,
    write_pointer,
)
from modelroom.contracts import SchemaVersionError
from modelroom.examples import EXAMPLES

BOUND = "3f9a0c21d4e6b870"
CONFIGURED = "1111111111111111"
FRESH = "ffffffffffffffff"
HERE = "9d2f4b6a8c0e1357"
ELSEWHERE = "0000000000000001"


def _profiles(**fingerprints: str) -> dict[str, KnownProfile]:
    ids = {"bound": BOUND, "configured": CONFIGURED}
    return {ids[key]: KnownProfile(ids[key], value, "measured", "os") for key, value in fingerprints.items()}


# --- pointer file ------------------------------------------------------------------------------


def test_missing_pointer_file_reads_as_empty(tmp_path):
    pointer = read_pointer(tmp_path / "guided.json")
    assert pointer.current is None and pointer.bindings == {}


def test_pointer_round_trip_and_with_binding(tmp_path):
    results = tmp_path / "results"
    path = tmp_path / ".modelroom" / "guided.json"
    pointer = read_pointer(path).with_binding(results, BOUND)
    write_pointer(path, pointer)
    again = read_pointer(path)
    assert again == pointer
    assert again.current == str(results)
    assert again.binding_for(results) == BOUND
    assert again.binding_for(tmp_path / "other") is None


def test_with_binding_keeps_other_folders(tmp_path):
    pointer = GuidedPointer.model_validate(EXAMPLES["GuidedPointer"]).with_binding(tmp_path, CONFIGURED)
    assert pointer.bindings["//models/modelroom"] == BOUND
    assert pointer.bindings[str(tmp_path)] == CONFIGURED


@pytest.mark.parametrize(
    "changes",
    [
        {"current": "relative/folder"},
        {"bindings": {"relative": BOUND}},
        {"bindings": {"//models/modelroom": "workstation"}},
        {"schema_version": 2},
    ],
)
def test_pointer_rejects_relative_folders_bad_ids_and_other_versions(changes):
    with pytest.raises(ValidationError):
        GuidedPointer.model_validate({**copy.deepcopy(EXAMPLES["GuidedPointer"]), **changes})


def test_broken_pointer_file_is_a_pointer_file_error(tmp_path):
    path = tmp_path / "guided.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(PointerFileError, match="guided.json"):
        read_pointer(path)
    path.write_text(json.dumps({"schema_version": 1, "current": "relative"}), encoding="utf-8")
    with pytest.raises(PointerFileError):
        read_pointer(path)
    # An integer literal too long for Python raises ValueError inside json.loads, not JSONDecodeError.
    path.write_text('{"schema_version": 1' + "0" * 4300 + "}", encoding="utf-8")
    with pytest.raises(PointerFileError, match="guided.json"):
        read_pointer(path)


def test_pointer_file_of_a_newer_schema_is_a_schema_version_error(tmp_path):
    path = tmp_path / "guided.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        read_pointer(path)


# --- takeover rule -----------------------------------------------------------------------------


def test_home_binding_wins_over_the_configuration():
    target = resolve_profile_target(BOUND, CONFIGURED, _profiles(bound=HERE, configured=HERE), HERE, FRESH)
    assert (target.action, target.profile_id) == ("bound", BOUND)


def test_configured_profile_is_adopted_without_a_home_binding():
    target = resolve_profile_target(None, CONFIGURED, _profiles(configured=HERE), HERE, FRESH)
    assert (target.action, target.profile_id) == ("adopt_config", CONFIGURED)


def test_neither_binding_nor_configuration_gives_a_new_profile():
    target = resolve_profile_target(None, None, {}, HERE, FRESH)
    assert (target.action, target.profile_id) == ("new", FRESH)


def test_new_identity_always_gives_a_new_profile():
    target = resolve_profile_target(BOUND, CONFIGURED, _profiles(bound=HERE), HERE, FRESH, new_identity=True)
    assert (target.action, target.profile_id) == ("new", FRESH)


def test_a_bound_profile_missing_from_the_folder_asks():
    target = resolve_profile_target(BOUND, None, {}, HERE, FRESH)
    assert (target.action, target.profile_id) == ("ask_clone", BOUND)
    assert "missing" in target.reason


def test_a_configured_profile_of_another_machine_asks_same_machine_or_clone():
    target = resolve_profile_target(None, CONFIGURED, _profiles(configured=ELSEWHERE), HERE, FRESH)
    assert (target.action, target.profile_id) == ("ask_clone", CONFIGURED)
    assert "os_fingerprint" in target.reason


@pytest.mark.parametrize("stored, local", [("none", HERE), (HERE, "none"), ("none", "none")])
def test_a_missing_fingerprint_on_either_side_never_disagrees(stored, local):
    target = resolve_profile_target(None, CONFIGURED, _profiles(configured=stored), local, FRESH)
    assert target.action == "adopt_config"


def test_the_same_machine_measures_again_under_a_binding_that_is_missing():
    """The way out of the clone question (decided 2026-09-24): the old id keeps its measurement."""
    target = resolve_profile_target(BOUND, None, {}, HERE, FRESH, same_machine=True)

    assert (target.action, target.profile_id) == ("rewrite", BOUND)
    assert "missing" in target.reason and "same id" in target.reason


def test_the_same_machine_measures_again_under_a_profile_of_another_fingerprint():
    target = resolve_profile_target(None, CONFIGURED, _profiles(configured=ELSEWHERE), HERE, FRESH, same_machine=True)

    assert (target.action, target.profile_id) == ("rewrite", CONFIGURED)
    assert "os_fingerprint" in target.reason and "same machine" in target.reason


@pytest.mark.parametrize(
    "home_binding, config_profile, profiles, expected",
    [
        (BOUND, CONFIGURED, {"bound": HERE}, ("bound", BOUND)),
        (None, CONFIGURED, {"configured": HERE}, ("adopt_config", CONFIGURED)),
        (None, None, {}, ("new", FRESH)),
    ],
)
def test_same_machine_changes_nothing_where_the_rule_does_not_ask(home_binding, config_profile, profiles, expected):
    known = _profiles(**{key: value for key, value in profiles.items()})
    target = resolve_profile_target(home_binding, config_profile, known, HERE, FRESH, same_machine=True)

    assert (target.action, target.profile_id) == expected


def test_both_switches_at_once_are_a_value_error():
    with pytest.raises(ValueError):
        resolve_profile_target(BOUND, None, {}, HERE, FRESH, new_identity=True, same_machine=True)


def test_the_fresh_id_must_be_new():
    with pytest.raises(ValueError):
        resolve_profile_target(None, None, _profiles(bound=HERE), HERE, BOUND)
    with pytest.raises(ValueError):
        resolve_profile_target(None, None, {}, HERE, "not-an-id")
