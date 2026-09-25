"""Render document schema 2: `Fit.requests`, and where the requests came from.

`docs/models.json` carries every fit's `requests`, the head count `users` and `requests_origin`.
The origin is copied from the configuration only when the rendered scenario is the one of that
configuration -- also when a caller passed it explicitly and it is equal; an explicitly passed
scenario of other requests leaves both empty, so the document never names a wrong origin
(decided 2026-09-25). Also here: the wording `shared memory` for a row of unified memory.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.cli import render_with_config
from modelroom.config import Configuration
from modelroom.contracts import Fit
from modelroom.document import DOCUMENT_SCHEMA_VERSION, MachineRanking, RenderDocument
from modelroom.examples import EXAMPLES
from modelroom.measurements import Scenario
from modelroom.ranking import SetAside
from modelroom.render import _computed_note, _set_aside_note
from modelroom.views import _pool_pieces

from test_cli import RUN2, _rendered
from fixture_support import place_v2_profile, with_profile


def _document(**overrides) -> dict:
    return {**copy.deepcopy(EXAMPLES["RenderDocument"]), **overrides}


def test_the_document_is_schema_2():
    assert DOCUMENT_SCHEMA_VERSION == 2
    assert EXAMPLES["RenderDocument"]["schema_version"] == 2


def test_the_example_carries_the_requests_and_their_origin():
    document = RenderDocument.model_validate(_document())
    assert document.requests_origin == "default" and document.users is None
    assert document.machines[0].ranked[0].fit.requests == 1


def test_the_document_round_trips_through_json_with_every_new_field():
    payload = _document(
        scenario={**EXAMPLES["RenderDocument"]["scenario"], "requests": 3}, users=25, requests_origin="from_users"
    )
    for block in payload["machines"]:
        for entry in [*block["ranked"], *block["not_covered"], *block["too_tight"]]:
            entry["fit"]["requests"] = 3
    document = RenderDocument.model_validate(payload)
    again = RenderDocument.model_validate(json.loads(json.dumps(document.model_dump(mode="json"))))
    assert again == document
    assert (again.users, again.requests_origin, again.machines[0].ranked[0].fit.requests) == (25, "from_users", 3)


@pytest.mark.parametrize(
    "fields",
    [
        {"requests_origin": "from_users"},  # a derived number needs the head count
        {"users": 25, "requests_origin": "default"},  # a head count is never a default
        {"users": 25},  # a head count without an origin
        {"requests_origin": "default", "scenario": {**EXAMPLES["Scenario"], "requests": 2}},
    ],
)
def test_an_origin_that_contradicts_the_document_is_invalid(fields):
    with pytest.raises(ValidationError):
        RenderDocument.model_validate(_document(**fields))


def test_a_ranked_fit_for_other_requests_than_the_scenario_is_invalid():
    payload = _document()
    payload["machines"][0]["ranked"][0]["fit"]["requests"] = 2
    with pytest.raises(ValidationError, match="requests"):
        RenderDocument.model_validate(payload)


# --- where the origin comes from ----------------------------------------------------------------


def _with_requests(config: Configuration, requests: int, origin: str, users: int | None = None) -> Configuration:
    data = config.model_dump(mode="json")
    data["guided"].update(requests=requests, requests_origin=origin, users=users)
    return Configuration.from_dict(data)


def _payload(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))


def _ready(tmp_path: Path) -> Configuration:
    config = _rendered(tmp_path)
    profile = place_v2_profile(config.paths.hardware_dir)
    return _with_requests(with_profile(config, "workstation", profile.profile_id), 3, "from_users", users=25)


def test_the_origin_comes_from_the_configuration_when_no_scenario_is_passed(tmp_path):
    config = _ready(tmp_path)

    assert render_with_config(config, now=RUN2) == 0

    payload = _payload(tmp_path)
    assert payload["schema_version"] == 2
    assert (payload["scenario"]["requests"], payload["users"], payload["requests_origin"]) == (3, 25, "from_users")
    fits = [entry["fit"] for block in payload["machines"] for entry in block["ranked"]]
    assert fits and all(fit["requests"] == 3 for fit in fits)


def test_the_origin_comes_from_the_configuration_for_an_equal_scenario_passed_explicitly(tmp_path):
    config = _ready(tmp_path)
    scenario = Scenario(context_requested=8192, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=3)

    assert render_with_config(config, now=RUN2, scenario=scenario) == 0

    assert (_payload(tmp_path)["users"], _payload(tmp_path)["requests_origin"]) == (25, "from_users")


def test_no_origin_for_a_scenario_of_other_requests_passed_explicitly(tmp_path):
    config = _ready(tmp_path)
    scenario = Scenario(context_requested=8192, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=2)

    assert render_with_config(config, now=RUN2, scenario=scenario) == 0

    payload = _payload(tmp_path)
    assert payload["scenario"]["requests"] == 2
    assert (payload["users"], payload["requests_origin"]) == (None, None)


# --- shared memory ------------------------------------------------------------------------------


def _unified_profile() -> dict:
    data = copy.deepcopy(EXAMPLES["HardwareProfile"])
    data.update(
        os_fingerprint="none",
        os_fingerprint_source="none",
        origin="entered",
        ram_physical_gib=32.0,
        ram_physical_source="entered",
        vram_gib=0.0,
        vram_source="none",
        gpu_state="unified_memory",
        gpu_name=None,
        llmfit_crosscheck={"ram_physical": {"status": "absent"}, "vram": {"status": "absent"}},
        llmfit_version=None,
    )
    return data


def test_the_pool_line_says_shared_memory_for_unified_memory():
    block = MachineRanking.model_validate({**copy.deepcopy(EXAMPLES["MachineRanking"]), "profile": _unified_profile()})
    pieces = _pool_pieces(block, "-")
    assert any("shared memory" in piece and "after both reserves" in piece for piece in pieces)
    assert not any("graphics memory" in piece for piece in pieces)


def test_the_pool_line_still_says_graphics_memory_for_a_graphics_card():
    block = MachineRanking.model_validate(EXAMPLES["MachineRanking"])
    assert any("graphics memory" in piece for piece in _pool_pieces(block, "-"))


def test_the_too_tight_note_names_both_reserves_for_unified_memory():
    fit = Fit.model_validate({**EXAMPLES["RankedEntry"]["fit"], "fit_class": "too_tight"})
    item = SetAside(package=None, fit=fit, reason="does not fit this machine")  # type: ignore[arg-type]
    assert "after both reserves" in _set_aside_note(item, covered=True, shared_memory=True).text
    assert "after the reserve" in _set_aside_note(item, covered=True).text


def test_the_row_note_says_shared_memory_for_unified_memory():
    fit = EXAMPLES["RankedEntry"]["fit"]
    note = _computed_note(Fit.model_validate(fit), shared_memory=True)
    assert "shared memory" in note.text and "graphics memory" not in note.text
    assert "graphics memory" in _computed_note(Fit.model_validate(fit), shared_memory=False).text
