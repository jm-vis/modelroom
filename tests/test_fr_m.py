"""Tests for AP9-FR-M (measurement protocol v2): the version on the card, and the Step 4 sentence for stored records."""

from __future__ import annotations

import pytest

from modelroom.measurements import MeasurementRecord

from test_fr_nachlese import _speed_fact, early_line


def _record_at(package, *, tps: float, protocol: str, hour: int, token: str):
    """A valid stored record of `package`, measured at `hour` o'clock under `protocol`."""
    from test_render import _measurement

    base = _measurement(package, tps=tps, token=token)
    at = base.measured_at.replace(hour=hour)
    data = base.model_dump()
    data.update(protocol=protocol, measured_at=at, measurement_id=f"{at:%Y%m%dT%H%M%SZ}-{token}")
    return MeasurementRecord.model_validate(data)


def _two_packages():
    from test_render import _hf_package

    slow = _hf_package(file_digest="sha256:" + "c" * 64)
    fast = _hf_package(file_digest="sha256:" + "d" * 64, quantization="Q8_0")
    return slow, fast


def test_the_card_names_the_protocol_of_the_counted_measurement_not_of_an_older_one():
    """Ranking, document and card: the newer slower v2 record counts, so tempo and version are its own."""
    from test_render import PROFILE_ID, _for_requests

    package, _other = _two_packages()
    older_faster = _record_at(package, tps=80.0, protocol="v1", hour=9, token="aaaaaaa1")
    newer_slower = _record_at(package, tps=50.0, protocol="v2", hour=11, token="aaaaaaa2")
    document = _for_requests(1, None, None, [package], {PROFILE_ID: [older_faster, newer_slower]})

    entry = document.machines[0].ranked[0]
    fact = _speed_fact(document)

    assert (entry.speed_protocol, round(entry.speed_tps)) == ("v2", 50)
    assert (fact.value, fact.note) == ("1 measured", "fastest Nova-8B at 50.0 tok/s, protocol v2")


def test_the_fastest_package_is_chosen_by_tempo_and_brings_its_own_version():
    from test_render import PROFILE_ID, _for_requests

    slow, fast = _two_packages()
    records = [
        _record_at(slow, tps=40.0, protocol="v1", hour=9, token="bbbbbbb1"),
        _record_at(fast, tps=90.0, protocol="v2", hour=10, token="bbbbbbb2"),
    ]
    document = _for_requests(1, None, None, [slow, fast], {PROFILE_ID: records})

    fact = _speed_fact(document)

    assert (fact.value, fact.note) == ("2 measured", "fastest Nova-8B at 90.0 tok/s, protocol v2")
    versions = {round(entry.speed_tps): entry.speed_protocol for entry in document.machines[0].ranked}
    assert versions == {40: "v1", 90: "v2"}


def test_a_stored_v1_record_still_counts_and_the_card_says_protocol_v1():
    from test_render import PROFILE_ID, _for_requests

    package, _other = _two_packages()
    document = _for_requests(1, None, None, [package], {PROFILE_ID: [_record_at(package, tps=50.0, protocol="v1", hour=9, token="ccccccc1")]})

    assert _speed_fact(document).note == "fastest Nova-8B at 50.0 tok/s, protocol v1"


def test_a_document_without_speed_protocol_gives_the_old_line_and_never_a_silent_v1():
    from modelroom.document import RenderDocument
    from test_render import PROFILE_ID, _for_requests

    package, _other = _two_packages()
    document = _for_requests(1, None, None, [package], {PROFILE_ID: [_record_at(package, tps=50.0, protocol="v2", hour=9, token="ddddddd1")]})
    data = document.model_dump(mode="json")
    for machine in data["machines"]:
        for entry in machine["ranked"]:
            entry.pop("speed_protocol")
    old = RenderDocument.model_validate(data)

    assert old.machines[0].ranked[0].speed_protocol is None
    assert _speed_fact(old).note == "fastest Nova-8B at 50.0 tok/s"


def _stored_early_record(protocol: str, **run_changes):
    """A stored invalid record of `protocol` whose first run ended after 2 tokens (no writer involved)."""
    import copy

    from modelroom.examples import EXAMPLES

    data = copy.deepcopy(EXAMPLES["MeasurementRecord"])
    data["runs"][0].update({"done_reason": "stop", "eval_count": 2, **run_changes})
    data.update(protocol=protocol, validity="invalid", tps_mean=None, tps_min=None, tps_max=None)
    data["validity_reason"] = "run 1: stored reason"
    return MeasurementRecord.model_validate(data)


class _Notes:
    def __init__(self):
        self.lines: list[str] = []
        self.screen = self

    def note(self, text: str) -> None:
        self.lines.append(text)


def test_a_stored_v1_record_with_an_early_end_gets_the_retry_hint():
    from modelroom.guided_loadtest import early_end_line, report_problem

    record = _stored_early_record("v1")
    notes = _Notes()

    assert early_end_line(record) == early_line(record.ollama_name, 2, version=1)
    assert report_problem(notes, record) is True
    assert notes.lines[1] == early_line(record.ollama_name, 2, version=1)
    assert notes.lines[1].endswith("say Yes to measure again")
    assert record.validity_reason == "run 1: stored reason"  # the stored reason is not touched


def test_a_stored_v2_record_with_an_early_end_gets_the_sentence_without_the_hint():
    from modelroom.guided_loadtest import early_end_line

    record = _stored_early_record("v2")

    assert early_end_line(record) == early_line(record.ollama_name, 2, version=2)
    assert "say Yes" not in early_end_line(record)


@pytest.mark.parametrize("protocol", ["v1", "v2"])
def test_a_stored_record_with_another_invalid_reason_gets_no_extra_line(protocol):
    from modelroom.guided_loadtest import early_end_line, report_problem

    record = _stored_early_record(protocol, done_reason="length", eval_count=128, eval_duration=0)
    notes = _Notes()

    assert early_end_line(record) is None
    assert report_problem(notes, record) is True and len(notes.lines) == 1


@pytest.mark.parametrize("protocol, word", [("v1", "protocol v1"), ("v2", "protocol v2")])
def test_the_measured_note_names_the_version_of_the_record(protocol, word):
    import copy

    from modelroom.examples import EXAMPLES
    from modelroom.guided_loadtest import _measurement_note

    data = copy.deepcopy(EXAMPLES["MeasurementRecord"])
    data["protocol"] = protocol
    record = MeasurementRecord.model_validate(data)

    assert _measurement_note(record, "-").endswith(f", context 8k, {word}")
