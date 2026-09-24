"""scripts/selftest.py: the pieces with logic of their own -- the answer file and the criteria.

The whole run (two guided runs, a real measurement, the live search smoke) is the acceptance
step itself, not a unit test; these tests check that every criterion really goes red when the
thing it is about is wrong.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "selftest.py"
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the self-test is tested from a clone", allow_module_level=True)
_SPEC = importlib.util.spec_from_file_location("modelroom_selftest", SCRIPT)
st = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(st)

RESULTS = Path("//models/results")


def _config_text(**changes) -> str:
    raw = {
        "schema_version": 2,
        "machines": {"workstation": {"reserve_ram_gib": 8, "reserve_vram_gib": 1, "writer": True}},
        "guided": {"results": str(RESULTS)},
    }
    raw.update(changes)
    machines = "\n".join(
        f"[machines.{name}]\nreserve_ram_gib = {entry['reserve_ram_gib']}\n"
        f"reserve_vram_gib = {entry['reserve_vram_gib']}\nwriter = {'true' if entry['writer'] else 'false'}"
        for name, entry in raw["machines"].items()
    )
    return (
        f"schema_version = {raw['schema_version']}\n{machines}\n"
        f"[guided]\nresults = {json.dumps(raw['guided']['results'])}\n"
    )


def _pointer(**changes) -> dict:
    data = {"schema_version": 1, "current": str(RESULTS), "bindings": {str(RESULTS): "3f9a0c21d4e6b870"}}
    data.update(changes)
    return data


def _profile(**changes) -> dict:
    data = {
        "gpu_state": "measured",
        "origin": "measured",
        "llmfit_crosscheck": {"ram_physical": {"status": "confirmed"}, "vram": {"status": "confirmed"}},
    }
    data.update(changes)
    return data


# --- the answer file ------------------------------------------------------------------------------


def test_the_answer_file_is_valid_toml_with_every_answer():
    text = st.answers_toml(st.ANSWERS_FIRST)

    raw = tomllib.loads(text)
    assert raw["schema_version"] == 1
    assert raw["results"] == "here"
    assert raw["machines"] == ["this-machine"]
    assert raw["filter_owners"] is True
    assert raw["select"] == [st.UNSLOTH_GGUF]


def test_the_second_answer_file_asks_no_folder_question():
    assert "results" not in tomllib.loads(st.answers_toml(st.ANSWERS_SECOND))


def test_every_answer_the_guided_mode_can_ask_for_has_a_key():
    from modelroom.guided import QUESTIONS

    assert set(st.ANSWERS_FIRST) <= set(QUESTIONS)


# --- criterion 1 -------------------------------------------------------------------------------------


def test_criterion_one_passes_on_a_first_start():
    assert st.first_start_problems(_config_text(), _pointer(), RESULTS, "workstation") == []


def test_criterion_one_names_a_machine_that_is_not_a_writer():
    text = _config_text(machines={"workstation": {"reserve_ram_gib": 8, "reserve_vram_gib": 1, "writer": False}})
    assert any("writer" in problem for problem in st.first_start_problems(text, _pointer(), RESULTS, "workstation"))


def test_criterion_one_names_a_pointer_that_remembers_nothing():
    problems = st.first_start_problems(_config_text(), _pointer(current="", bindings={}), RESULTS, "workstation")
    assert len(problems) == 2


# --- criterion 2 -------------------------------------------------------------------------------------


def test_criterion_two_passes_on_a_measured_profile():
    assert st.measurement_problems(_profile()) == []


@pytest.mark.parametrize("change", [{"gpu_state": "present_unmeasured"}, {"origin": "entered"}])
def test_criterion_two_names_a_profile_that_was_not_measured(change):
    assert st.measurement_problems(_profile(**change))


def test_criterion_two_names_a_cross_check_that_is_not_confirmed():
    crosscheck = {"ram_physical": {"status": "deviation"}, "vram": {"status": "absent"}}
    assert len(st.measurement_problems(_profile(llmfit_crosscheck=crosscheck))) == 2


# --- criterion 3 -------------------------------------------------------------------------------------


GOOD_SEARCH = [
    "8 repositories, 3 resolved, 5 unresolved, budget 3/60",
    "unresolved community-user/Qwen3.5-9B-Plain-GGUF: relation_unknown",
]


def test_criterion_three_passes_on_a_real_search_output():
    assert st.search_problems(GOOD_SEARCH) == []


def test_criterion_three_names_a_missing_summary():
    assert st.search_problems(["nothing here"]) == ["no search summary line was printed"]


def test_criterion_three_names_a_search_that_resolved_nothing():
    lines = ["8 repositories, 0 resolved, 8 unresolved, budget 3/60", "unresolved a/B: derivative"]
    assert any("resolved 0" in problem for problem in st.search_problems(lines))


def test_criterion_three_names_an_unresolved_hit_without_a_reason():
    lines = [GOOD_SEARCH[0], "unresolved community-user/X-GGUF: None"]
    assert any("without a reason" in problem for problem in st.search_problems(lines))


# --- criterion 4 -------------------------------------------------------------------------------------


GOOD_DOCUMENT = "\n".join(
    [
        "Scenario: context 8192 (default), KV cache f16 (assumed), 1 request",
        "Ranking rule: fit class (perfect, good, marginal), then measured group",
        "## Ranking: workstation",
    ]
)
NOT_COVERED = [{"base_model_hf_repo": st.QWEN_BASE, "reason": "architecture not covered by v1"}]


def test_criterion_four_passes_on_a_ranking_with_its_header():
    assert st.ranking_problems(GOOD_DOCUMENT, "workstation", NOT_COVERED) == []


def test_criterion_four_names_a_missing_header_line():
    without_rule = GOOD_DOCUMENT.replace("Ranking rule: ", "Rule: ")
    assert any("Ranking rule" in problem for problem in st.ranking_problems(without_rule, "workstation", NOT_COVERED))


def test_criterion_four_names_an_empty_not_covered_block():
    assert any("empty" in problem for problem in st.ranking_problems(GOOD_DOCUMENT, "workstation", []))


def test_criterion_four_names_another_reason_than_the_fit_rules_own():
    other = [{"base_model_hf_repo": st.QWEN_BASE, "reason": "no profile"}]
    assert any("not the fit rule" in problem for problem in st.ranking_problems(GOOD_DOCUMENT, "workstation", other))


# --- criterion 6 -------------------------------------------------------------------------------------


BEFORE = {"machines": {"workstation": {"profile": "3f9a0c21d4e6b870"}}, "guided": {"results": str(RESULTS)}}


def test_criterion_six_passes_when_nothing_changed():
    assert st.second_start_problems(BEFORE, dict(BEFORE), ["3f9a0c21d4e6b870.json"], ["a line"]) == []


def test_criterion_six_names_a_second_profile_file():
    problems = st.second_start_problems(BEFORE, dict(BEFORE), ["a.json", "b.json"], [])
    assert any("2 profile files" in problem for problem in problems)


def test_criterion_six_names_a_changed_machine_table():
    after = {"machines": {"workstation": {"profile": "0000000000000001"}}, "guided": BEFORE["guided"]}
    assert any("[machines]" in problem for problem in st.second_start_problems(BEFORE, after, ["a.json"], []))


def test_criterion_six_names_a_clone_question():
    lines = ["profile from home binding is missing. Is this the same machine or a clone?"]
    assert any("clone question" in problem for problem in st.second_start_problems(BEFORE, dict(BEFORE), ["a.json"], lines))


# --- the report ------------------------------------------------------------------------------------------


def test_the_report_marks_each_step_and_indents_its_detail():
    text = st.format_report([st.Step("one", True, "fine"), st.Step("two", False, "broken\n")])

    assert "[1] one: OK" in text
    assert "    fine" in text
    assert "[2] two: FAIL" in text
