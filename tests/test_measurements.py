"""Tests for modelroom.measurements: scenario, protocols v1 and v2, measurement records, files, export."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.contracts import HardwareSnapshot, SchemaVersionError
from modelroom.examples import EXAMPLES
from modelroom.measurements import (
    PROTOCOL_FILES,
    ExportObject,
    MeasurementExistsError,
    MeasurementRecord,
    RunCounters,
    Scenario,
    default_scenario,
    legacy_measurement_record,
    load_export,
    load_protocol,
    measurement_invalid_reason,
    measurement_path,
    new_measurement_id,
    plan_measurement_import,
    protocol_for,
    read_measurements,
    run_invalid_reason,
    shipped_protocol,
    write_measurement,
)
from modelroom.state import acquire_lock, release_lock

FIXTURES = Path(__file__).parent / "fixtures"
PROFILE_ID = "3f9a0c21d4e6b870"
NOW = datetime(2026, 9, 23, 10, 0, 0, tzinfo=timezone.utc)


def _record(**changes) -> dict:
    payload = copy.deepcopy(EXAMPLES["MeasurementRecord"])
    payload.update(changes)
    return payload


def _run(**changes) -> RunCounters:
    return RunCounters.model_validate({**EXAMPLES["RunCounters"], **changes})


# --- scenario ---------------------------------------------------------------------------------


def test_default_scenario_is_8192_f16_assumed_one_request():
    scenario = default_scenario()
    assert (scenario.context_requested, scenario.context_origin) == (8192, "default")
    assert (scenario.kv_type, scenario.kv_type_assumed, scenario.requests) == ("f16", True, 1)


@pytest.mark.parametrize(
    "changes",
    [{"kv_type_assumed": False}, {"context_origin": "default", "context_requested": 4096}],
)
def test_scenario_rejects_contradicting_origin_and_kv_statements(changes):
    with pytest.raises(ValidationError):
        Scenario.model_validate({**EXAMPLES["Scenario"], **changes})


def test_scenario_entered_context_may_be_any_value():
    Scenario.model_validate({**EXAMPLES["Scenario"], "context_origin": "entered", "context_requested": 4096})


VERSIONS = ("v1", "v2")


@pytest.mark.parametrize("protocol", VERSIONS)
def test_speeds_must_be_above_zero(protocol):
    base = _record(protocol=protocol, validity="invalid", validity_reason="x")  # the example's runs stay
    with pytest.raises(ValidationError, match=f"protocol {protocol} requires 0 < tps_min"):
        MeasurementRecord.model_validate({**base, "tps_min": 0.0})


@pytest.mark.parametrize("field, value", [("context_requested", 0), ("requests", 0), ("kv_type", "q8_0")])
def test_scenario_rejects_out_of_range_values(field, value):
    with pytest.raises(ValidationError):
        Scenario.model_validate({**EXAMPLES["Scenario"], field: value})


# --- protocol v1 ------------------------------------------------------------------------------


PACKAGE_DIR = Path(__file__).parent.parent / "modelroom"


def test_shipped_protocol_is_version_2_without_raw_and_with_fixed_sampling():
    protocol = shipped_protocol()
    assert protocol is protocol_for("v2")
    assert protocol.protocol_version == 2
    assert (protocol.warmup_runs, protocol.measured_runs) == (1, 3)
    assert protocol.options.raw is False and protocol.options.stream is False
    assert (protocol.options.num_predict, protocol.options.temperature, protocol.options.seed) == (128, 0.0, 42)
    assert protocol.validity.done_reason == "length"


def test_protocol_for_v1_is_version_1_with_raw_and_the_same_prompt():
    v1, v2 = protocol_for("v1"), protocol_for("v2")
    assert v1.protocol_version == 1 and v1.options.raw is True
    assert v2.protocol_version == 2 and v2.options.raw is False
    assert v1.prompt == v2.prompt


def test_protocol_v2_differs_from_v1_only_in_version_and_raw():
    v1, v2 = protocol_for("v1").model_dump(), protocol_for("v2").model_dump()
    for protocol, version, raw in ((v1, 1, True), (v2, 2, False)):
        assert (protocol.pop("protocol_version"), protocol["options"].pop("raw")) == (version, raw)
    assert v1 == v2


def test_protocol_for_an_unknown_version_is_a_value_error_with_its_name():
    with pytest.raises(ValueError, match="no measurement protocol v3"):
        protocol_for("v3")


def test_shipped_protocol_files_are_inside_the_package():
    import modelroom

    for name in ("protocol_v1.toml", "protocol_v2.toml"):
        assert (Path(modelroom.__file__).parent / name).is_file()
    assert set(PROTOCOL_FILES) == {"v1", "v2"}


def test_protocol_v1_file_is_unchanged_from_the_base():
    import hashlib

    data = (PACKAGE_DIR / "protocol_v1.toml").read_bytes()
    assert hashlib.sha256(data).hexdigest() == "97f2f13bbea7d1c3c1b5df3ccc89c5310c4c853065226b7a93fa71f63a55221f"


def test_load_protocol_rejects_version_1_with_raw_false_as_a_regression(tmp_path):
    """Regression, green on the base too: version 1 ships raw = true."""
    text = (PACKAGE_DIR / "protocol_v1.toml").read_text(encoding="utf-8")
    broken = tmp_path / "protocol.toml"
    broken.write_text(text.replace("raw = true", "raw = false"), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid measurement protocol"):
        load_protocol(broken)


def test_load_protocol_rejects_version_2_with_raw_true_by_the_validator(tmp_path):
    text = (PACKAGE_DIR / "protocol_v2.toml").read_text(encoding="utf-8")
    broken = tmp_path / "protocol.toml"
    broken.write_text(text.replace("raw = false", "raw = true"), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol version 2 ships raw = false"):
        load_protocol(broken)


def test_load_protocol_rejects_version_1_with_raw_false_by_the_validator(tmp_path):
    text = (PACKAGE_DIR / "protocol_v1.toml").read_text(encoding="utf-8")
    broken = tmp_path / "protocol.toml"
    broken.write_text(text.replace("raw = true", "raw = false"), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol version 1 ships raw = true"):
        load_protocol(broken)


def test_load_protocol_accepts_version_2_with_raw_false(tmp_path):
    copy_of_v2 = tmp_path / "protocol.toml"
    copy_of_v2.write_bytes((PACKAGE_DIR / "protocol_v2.toml").read_bytes())
    assert load_protocol(copy_of_v2).options.raw is False


@pytest.mark.parametrize(
    "changes, num_ctx, fragment",
    [
        ({"done_reason": "stop"}, 8192, "done_reason"),
        ({"eval_count": 127}, 8192, "eval_count"),
        ({"eval_duration": 0}, 8192, "eval_duration"),
        ({"prompt_eval_count": 8100}, 8192, "exceeds num_ctx"),
    ],
)
def test_run_invalid_reason_names_the_broken_rule(changes, num_ctx, fragment):
    assert fragment in run_invalid_reason(_run(**changes), shipped_protocol(), num_ctx)


def test_run_at_the_exact_context_limit_is_valid():
    run = _run(prompt_eval_count=8192 - 128)
    assert run_invalid_reason(run, shipped_protocol(), 8192) is None


def test_measurement_needs_exactly_the_measured_run_count():
    assert "2 measured runs" in measurement_invalid_reason([_run(), _run()], shipped_protocol(), 8192)
    assert measurement_invalid_reason([_run(), _run(), _run()], shipped_protocol(), 8192) is None


# --- MeasurementRecord ----------------------------------------------------------------------


def test_measurement_id_follows_measured_at():
    assert new_measurement_id(NOW, "0a1b2c3d") == "20260923T100000Z-0a1b2c3d"
    random_id = new_measurement_id(NOW)
    assert random_id.startswith("20260923T100000Z-") and len(random_id) == 25


@pytest.mark.parametrize("bad_id", ["20260923T083001Z-5c1e9a07", "20260923T083000Z-5C1E9A07", "20260923T083000Z"])
def test_measurement_id_must_agree_with_measured_at_and_be_hex(bad_id):
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(_record(measurement_id=bad_id))


def test_a_measurement_runs_exactly_one_request():
    with pytest.raises(ValidationError, match="one request"):
        MeasurementRecord.model_validate(_record(scenario={**EXAMPLES["Scenario"], "requests": 2}))


@pytest.mark.parametrize("protocol", VERSIONS)
def test_valid_measurement_must_satisfy_the_protocol_rule(protocol):
    runs = copy.deepcopy(EXAMPLES["MeasurementRecord"]["runs"])
    runs[0]["done_reason"] = "stop"
    with pytest.raises(ValidationError, match="contradicts the protocol rule"):
        MeasurementRecord.model_validate(_record(protocol=protocol, runs=runs))


@pytest.mark.parametrize("protocol", VERSIONS)
def test_valid_measurement_without_runs_is_refused(protocol):
    with pytest.raises(ValidationError, match="contradicts the protocol rule"):
        MeasurementRecord.model_validate(
            _record(protocol=protocol, runs=[], tps_mean=None, tps_min=None, tps_max=None)
        )


@pytest.mark.parametrize("protocol", VERSIONS)
def test_valid_speeds_must_be_the_mean_min_max_of_the_runs(protocol):
    with pytest.raises(ValidationError, match="do not match the runs"):
        MeasurementRecord.model_validate(_record(protocol=protocol, tps_mean=51.0))  # inside min..max, not the mean


@pytest.mark.parametrize("field, value", [("tps_min", 49.0), ("tps_max", 52.0)])
@pytest.mark.parametrize("protocol", VERSIONS)
def test_minimum_and_maximum_are_each_checked_against_the_runs(protocol, field, value):
    """Only one bound differs and 0 < min <= mean <= max still holds, so the comparison with `runs` is what fails."""
    with pytest.raises(ValidationError, match="do not match the runs"):
        MeasurementRecord.model_validate(_record(protocol=protocol, **{field: value}))


@pytest.mark.parametrize("protocol", VERSIONS)
def test_invalid_without_a_validity_reason_is_refused(protocol):
    with pytest.raises(ValidationError, match="validity_reason is set exactly when"):
        MeasurementRecord.model_validate(_record(protocol=protocol, validity="invalid", validity_reason=None))


@pytest.mark.parametrize("protocol", VERSIONS)
def test_comparable_with_a_comparable_reason_is_refused(protocol):
    with pytest.raises(ValidationError, match="comparable_reason is set exactly when"):
        MeasurementRecord.model_validate(_record(protocol=protocol, comparable=True, comparable_reason="x"))


@pytest.mark.parametrize("protocol", VERSIONS)
def test_a_valid_record_is_checked_against_the_file_of_its_version(protocol, monkeypatch):
    asked: list[str] = []
    real = protocol_for

    def spy(version):
        asked.append(version)
        return real(version)

    monkeypatch.setattr("modelroom.measurements.protocol_for", spy)
    MeasurementRecord.model_validate(_record(protocol=protocol))
    assert asked == [protocol]


def test_an_overflowing_run_counter_is_a_validation_error():
    record = _record()
    record["runs"][0]["eval_duration"] = 10**400
    with pytest.raises(ValidationError, match="eval_duration"):
        MeasurementRecord.model_validate(record)


def test_one_broken_measurement_file_does_not_stop_reading_the_others(tmp_path):
    good = MeasurementRecord.model_validate(_record())
    folder = tmp_path / "measurements" / good.profile_id
    folder.mkdir(parents=True)
    (folder / f"{good.measurement_id}.json").write_text(good.model_dump_json(), encoding="utf-8")
    broken = _record(measurement_id=good.measurement_id[:-8] + "ffffffff")
    broken["runs"][0]["eval_duration"] = 10**400
    (folder / f"{broken['measurement_id']}.json").write_text(json.dumps(broken), encoding="utf-8")
    result = read_measurements(tmp_path, good.profile_id)
    assert result.records == [good]
    assert [path.stem for path, _ in result.unreadable] == [broken["measurement_id"]]


@pytest.mark.parametrize("protocol", VERSIONS)
def test_speeds_need_runs_to_come_from(protocol):
    with pytest.raises(ValidationError, match=f"protocol {protocol} tps values come from runs"):
        MeasurementRecord.model_validate(
            _record(protocol=protocol, runs=[], validity="invalid", validity_reason="no runs")
        )
    MeasurementRecord.model_validate(
        _record(
            protocol=protocol, runs=[], validity="invalid", validity_reason="no runs",
            tps_mean=None, tps_min=None, tps_max=None,
        )
    )


@pytest.mark.parametrize("protocol", VERSIONS)
def test_a_version_is_never_unchecked_and_reasons_follow_the_flags(protocol):
    with pytest.raises(ValidationError, match=f"protocol '{protocol}' is either 'valid' or 'invalid'"):
        MeasurementRecord.model_validate(_record(protocol=protocol, validity="unchecked", validity_reason="x"))
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(_record(protocol=protocol, validity_reason="should not be here"))
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(_record(protocol=protocol, comparable=False))
    MeasurementRecord.model_validate(
        _record(protocol=protocol, comparable=False, comparable_reason="daemon digest changed")
    )


def test_the_sink_lets_a_v2_reason_through_unchanged():
    """The reason of a v2 record has the same shape as v1's: no new pattern is needed in `neutral.py`."""
    from modelroom.neutral import neutral_record

    reason = "run 1: done_reason 'stop', expected 'length'"
    runs = copy.deepcopy(EXAMPLES["MeasurementRecord"]["runs"])
    runs[0].update(done_reason="stop", eval_count=2)
    record = MeasurementRecord.model_validate(
        _record(
            protocol="v2", runs=runs, validity="invalid", validity_reason=reason,
            tps_mean=None, tps_min=None, tps_max=None,
        )
    )
    assert neutral_record(record).validity_reason == reason


def test_a_protocol_word_outside_v1_v2_none_is_refused():
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(_record(protocol="v3"))


def test_invalid_measurement_fixture_validates_and_keeps_its_reason():
    data = json.loads((FIXTURES / "measurement_v2_invalid.json").read_text(encoding="utf-8"))
    record = MeasurementRecord.model_validate(data)
    assert record.validity == "invalid"
    assert "done_reason" in record.validity_reason


def test_protocol_none_is_unchecked_without_runs_and_never_comparable():
    legacy = _record(protocol="none", runs=[], validity="unchecked", validity_reason="old", comparable=False, comparable_reason="old")
    MeasurementRecord.model_validate(legacy)
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate({**legacy, "comparable": True, "comparable_reason": None})
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate({**legacy, "runs": EXAMPLES["MeasurementRecord"]["runs"]})


def test_tps_values_are_all_set_or_none_and_min_not_above_max():
    # protocol 'none' carries tps values without runs, so only the coupling rule can refuse these.
    base = _record(protocol="none", runs=[], validity="unchecked", validity_reason="old", comparable=False, comparable_reason="old")
    MeasurementRecord.model_validate(base)
    with pytest.raises(ValidationError, match="all set or none"):
        MeasurementRecord.model_validate({**base, "tps_min": None})
    with pytest.raises(ValidationError, match="tps_min <= tps_max"):
        MeasurementRecord.model_validate({**base, "tps_min": base["tps_max"] + 1.0})
    MeasurementRecord.model_validate({**base, "tps_mean": None, "tps_min": None, "tps_max": None})


def test_a_number_too_long_for_json_loads_leaves_the_other_files_readable(tmp_path):
    good = MeasurementRecord.model_validate(_record())
    folder = tmp_path / "measurements" / good.profile_id
    folder.mkdir(parents=True)
    (folder / f"{good.measurement_id}.json").write_text(good.model_dump_json(), encoding="utf-8")
    broken = _record(measurement_id=good.measurement_id[:-8] + "ffffffff")
    broken["runs"][0]["eval_duration"] = "TOO_LONG"
    # Python refuses to convert an integer literal of more than 4300 digits (ValueError, not JSONDecodeError).
    text = json.dumps(broken).replace('"TOO_LONG"', "1" + "0" * 4300)
    (folder / f"{broken['measurement_id']}.json").write_text(text, encoding="utf-8")
    result = read_measurements(tmp_path, good.profile_id)
    assert result.records == [good]
    assert [path.stem for path, _ in result.unreadable] == [broken["measurement_id"]]


# --- legacy conversion ------------------------------------------------------------------------


def _legacy_laptop() -> HardwareSnapshot:
    data = json.loads((FIXTURES / "profiles_v1" / "windows-nvidia-laptop.json").read_text(encoding="utf-8"))
    return HardwareSnapshot.model_validate(data)


def test_legacy_measurement_is_protocol_none_with_the_legacy_context():
    legacy = _legacy_laptop().measurements[0]
    record = legacy_measurement_record(legacy, PROFILE_ID)
    assert record.protocol == "none" and record.validity == "unchecked" and not record.comparable
    assert record.scenario.context_requested == legacy.context
    assert record.scenario.context_origin == "legacy"
    assert record.package.ollama_manifest_digest == legacy.ollama_manifest_digest
    assert record.daemon_version == legacy.runtime
    assert record.measured_at == legacy.measured_at


def test_legacy_measurement_id_is_deterministic():
    legacy = _legacy_laptop().measurements[0]
    assert legacy_measurement_record(legacy, PROFILE_ID) == legacy_measurement_record(legacy, PROFILE_ID)


def test_legacy_range_is_carried_over_unchanged_even_when_it_misses_the_mean():
    legacy = _legacy_laptop().measurements[1]
    assert not legacy.tps_range[0] <= legacy.tps_mean  # the fixture's schema-1 oddity
    record = legacy_measurement_record(legacy, PROFILE_ID)
    assert (record.tps_mean, record.tps_min, record.tps_max) == (legacy.tps_mean, *legacy.tps_range)


# --- files ------------------------------------------------------------------------------------


def _record_model(**changes) -> MeasurementRecord:
    return MeasurementRecord.model_validate(_record(**changes))


def test_write_measurement_needs_the_held_lock_of_this_state_folder(tmp_path):
    other = tmp_path / "other"
    handle = acquire_lock(other / "modelroom.lock", "test", NOW)
    try:
        with pytest.raises(ValueError, match="held lock"):
            write_measurement(tmp_path / "state", _record_model(), handle)
    finally:
        release_lock(handle)
    with pytest.raises(ValueError, match="held lock"):
        write_measurement(other, _record_model(), handle)  # released


def test_write_measurement_never_overwrites(tmp_path):
    state = tmp_path / "state"
    handle = acquire_lock(state / "modelroom.lock", "test", NOW)
    try:
        path = write_measurement(state, _record_model(), handle)
        assert path == measurement_path(state, PROFILE_ID, EXAMPLES["MeasurementRecord"]["measurement_id"])
        with pytest.raises(MeasurementExistsError):
            write_measurement(state, _record_model(comparable=False, comparable_reason="other content"), handle)
    finally:
        release_lock(handle)
    assert json.loads(path.read_text(encoding="utf-8")) == _record_model().model_dump(mode="json")


def test_read_measurements_lists_a_file_nested_too_deep_for_the_json_parser_as_unreadable(tmp_path):
    """`json.loads` raises `RecursionError` (neither `ValueError` nor `OSError`) at extreme
    nesting; such a file is listed like any other broken one, never a crash of the reader."""
    state = tmp_path / "state"
    folder = state / "measurements" / PROFILE_ID
    folder.mkdir(parents=True)
    good = _record_model()
    (folder / f"{good.measurement_id}.json").write_text(json.dumps(good.model_dump(mode="json")), encoding="utf-8")
    depth = 100_000
    (folder / "20260923T090000Z-22222222.json").write_text("[" * depth + "]" * depth, encoding="utf-8")
    result = read_measurements(state, PROFILE_ID)
    assert result.records == [good]
    assert [p.name for p, _ in result.unreadable] == ["20260923T090000Z-22222222.json"]


def test_read_measurements_lists_broken_and_misplaced_files_as_unreadable(tmp_path):
    state = tmp_path / "state"
    folder = state / "measurements" / PROFILE_ID
    folder.mkdir(parents=True)
    good = _record_model()
    (folder / f"{good.measurement_id}.json").write_text(json.dumps(good.model_dump(mode="json")), encoding="utf-8")
    (folder / "20260923T090000Z-00000000.json").write_text("{not json", encoding="utf-8")
    (folder / "20260923T091500Z-11111111.json").write_text(json.dumps(good.model_dump(mode="json")), encoding="utf-8")
    result = read_measurements(state, PROFILE_ID)
    assert result.records == [good]
    assert sorted(p.name for p, _ in result.unreadable) == [
        "20260923T090000Z-00000000.json",
        "20260923T091500Z-11111111.json",
    ]


def test_read_measurements_of_an_unknown_profile_is_empty(tmp_path):
    result = read_measurements(tmp_path, PROFILE_ID)
    assert result.records == [] and result.unreadable == []


# --- import rule -------------------------------------------------------------------------------


def test_import_rule_new_idempotent_conflict():
    existing = _record_model()
    same = _record_model()
    changed = _record_model(comparable=False, comparable_reason="different content")
    fresh = _record_model(measurement_id="20260923T083000Z-00000001")
    plan = plan_measurement_import([existing], [same, changed, fresh])
    assert plan.idempotent == [same]
    assert plan.conflicts == [changed]
    assert plan.new == [fresh]


# --- export ------------------------------------------------------------------------------------


def test_export_fixture_loads_with_both_protocols():
    export = load_export(json.loads((FIXTURES / "export_v1.json").read_text(encoding="utf-8")))
    assert [m.protocol for m in export.measurements] == ["v1", "none"]


def test_export_rejects_foreign_or_duplicate_measurements():
    payload = copy.deepcopy(EXAMPLES["ExportObject"])
    payload["measurements"].append(copy.deepcopy(payload["measurements"][0]))
    with pytest.raises(ValidationError, match="unique"):
        ExportObject.model_validate(payload)
    payload = copy.deepcopy(EXAMPLES["ExportObject"])
    payload["measurements"][0]["profile_id"] = "0000000000000000"
    with pytest.raises(ValidationError, match="exported profile"):
        ExportObject.model_validate(payload)


@pytest.mark.parametrize("version", [2, None])
def test_load_export_checks_the_version_first(version):
    with pytest.raises(SchemaVersionError):
        load_export({"schema_version": version, "profile": None})
