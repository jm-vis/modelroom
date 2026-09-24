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
    assert raw["select"] == [st.UNSLOTH_GGUF, st.DEEPSEEK_GGUF]
    assert raw["load_test"] is True
    assert raw["load_test_packages"] == [st.DEEPSEEK_OLLAMA_NAME]


def test_the_second_answer_file_declines_the_load_test():
    raw = tomllib.loads(st.answers_toml(st.ANSWERS_SECOND))

    assert raw["load_test"] is False
    assert "load_test_packages" not in raw


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
    "2 repositories cannot be picked: a fine-tune or a merge, not a quantization of one base model",
    "3 repositories cannot be picked: the repository does not say it packages a base model",
]


def test_criterion_three_passes_on_a_real_search_output():
    assert st.search_problems(GOOD_SEARCH) == []


def test_criterion_three_names_a_missing_summary():
    assert st.search_problems(["nothing here"]) == ["no search summary line was printed"]


def test_criterion_three_names_a_search_that_resolved_nothing():
    lines = ["8 repositories, 0 resolved, 8 unresolved, budget 3/60", "8 repositories cannot be picked: derivative"]
    assert any("resolved 0" in problem for problem in st.search_problems(lines))


def test_criterion_three_names_a_run_that_showed_no_reason_at_all():
    assert any("no reason was shown" in problem for problem in st.search_problems([GOOD_SEARCH[0]]))


def test_criterion_three_names_a_reason_that_is_a_raw_status():
    lines = [GOOD_SEARCH[0], "5 repositories cannot be picked: None"]
    assert any("raw status" in problem for problem in st.search_problems(lines))


def test_criterion_three_names_grouped_lines_that_leave_repositories_out():
    """One line per reason has to account for every unresolved repository of the summary."""
    lines = [GOOD_SEARCH[0], "1 repository cannot be picked: the repository does not say it packages a base model"]

    assert any("cover 1 repositories" in problem for problem in st.search_problems(lines))


# --- criterion 4 -------------------------------------------------------------------------------------


GOOD_DOCUMENT = "\n".join(
    [
        "Scenario: context 8192 (entered), KV cache f16 (assumed), 1 request",
        "Ranking rule: fit class (perfect, good, marginal), then measured group",
        "## Ranking: workstation",
    ]
)
def _block(**changes) -> dict:
    """A machine block in which Qwen3.5 is ranked from the size of its packages."""
    entry = {"base_model_hf_repo": st.QWEN_BASE, "fit": {"basis": "size"}, "reason": None}
    data = {"ranked": [entry], "too_tight": [], "not_covered": []}
    data.update(changes)
    return data


def test_criterion_four_passes_on_a_ranking_with_its_header():
    assert st.ranking_problems(GOOD_DOCUMENT, "workstation", _block()) == []


def test_criterion_four_names_a_missing_header_line():
    without_rule = GOOD_DOCUMENT.replace("Ranking rule: ", "Rule: ")
    assert any("Ranking rule" in problem for problem in st.ranking_problems(without_rule, "workstation", _block()))


def test_criterion_four_names_a_machine_that_judged_no_package_of_the_model():
    empty = _block(ranked=[])
    assert any("no package" in problem for problem in st.ranking_problems(GOOD_DOCUMENT, "workstation", empty))


def test_criterion_four_names_a_package_judged_on_another_basis():
    architecture = _block(ranked=[{"base_model_hf_repo": st.QWEN_BASE, "fit": {"basis": "architecture"}}])
    problems = st.ranking_problems(GOOD_DOCUMENT, "workstation", architecture)
    assert any("another basis" in problem for problem in problems)


def test_criterion_four_names_packages_still_set_aside_as_not_covered():
    stale = _block(not_covered=[{"base_model_hf_repo": st.QWEN_BASE, "reason": "architecture not covered by v1"}])
    problems = st.ranking_problems(GOOD_DOCUMENT, "workstation", stale)
    assert any("still set aside" in problem for problem in problems)


# --- criterion 5 -------------------------------------------------------------------------------------


def _record(**changes) -> dict:
    data = {
        "protocol": "v1",
        "validity": "valid",
        "validity_reason": None,
        "comparable": True,
        "comparable_reason": None,
        "package": {"content_source": "huggingface", "hf_repo": st.DEEPSEEK_GGUF},
        "ollama_name": st.DEEPSEEK_OLLAMA_NAME,
    }
    data.update(changes)
    return data


def _row(**changes) -> dict:
    data = {"rank": 1, "measurement_group": 0, "speed_tps": 58.4}
    data.update(changes)
    return data


def test_criterion_five_passes_on_one_valid_comparable_ranked_measurement():
    assert st.load_test_problems([_record()], _row()) == []


def test_criterion_five_names_a_machine_the_package_is_not_installed_on():
    problems = st.load_test_problems([], None)

    assert len(problems) == 1
    assert st.DEEPSEEK_OLLAMA_NAME in problems[0]


def test_criterion_five_names_a_second_measurement_file():
    assert any("2 measurement files" in problem for problem in st.load_test_problems([_record(), _record()], _row()))


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"validity": "invalid", "validity_reason": "run 1: done_reason 'stop'"}, "validity is"),
        ({"comparable": False, "comparable_reason": "the digest changed"}, "not comparable"),
        ({"protocol": "none"}, "protocol is"),
        ({"package": {"content_source": "huggingface", "hf_repo": "other/Repo-GGUF"}}, "the measurement names"),
    ],
    ids=["invalid", "not comparable", "no protocol", "another package"],
)
def test_criterion_five_names_a_measurement_that_does_not_count(change, expected):
    problems = st.load_test_problems([_record(**change)], _row())

    assert any(expected in problem for problem in problems)


@pytest.mark.parametrize("row", [None, _row(measurement_group=1, speed_tps=None)], ids=["absent", "group 1"])
def test_criterion_five_names_a_package_that_is_not_ranked_with_its_speed(row):
    assert st.load_test_problems([_record()], row)


# --- criterion 6 -------------------------------------------------------------------------------------


BEFORE = {"machines": {"workstation": {"profile": "3f9a0c21d4e6b870"}}, "guided": {"results": str(RESULTS)}}
ANSWERED_CONTEXT = st.ANSWERED_CONTEXT
KEPT = {"machines": BEFORE["machines"], "guided": {"results": str(RESULTS), "context": ANSWERED_CONTEXT}}
GOOD_HEADER = f"Scenario: context {ANSWERED_CONTEXT} (entered), KV cache f16 (assumed), 1 request"


def test_criterion_six_passes_when_nothing_changed():
    assert st.second_start_problems(BEFORE, dict(BEFORE), ["3f9a0c21d4e6b870.json"], ["a line"]) == []


def test_criterion_six_passes_when_the_measurement_of_the_first_run_is_still_ranked():
    assert st.second_start_problems(BEFORE, dict(BEFORE), ["a.json"], [], [_record()], _row()) == []


def test_criterion_six_names_a_second_measurement_file():
    problems = st.second_start_problems(BEFORE, dict(BEFORE), ["a.json"], [], [_record(), _record()], _row())
    assert any("2 measurement files after the second run" in problem for problem in problems)


def test_criterion_six_names_a_measurement_that_dropped_out_of_group_zero():
    problems = st.second_start_problems(BEFORE, dict(BEFORE), ["a.json"], [], [_record()], _row(measurement_group=1))
    assert any("no longer ranked in group 0" in problem for problem in problems)


def test_criterion_six_names_a_second_measurement_of_the_same_package():
    lines = [f"{st.DEEPSEEK_OLLAMA_NAME}: measured 58.4 tok/s (57.0-60.0), context 8192, valid, comparable"]
    problems = st.second_start_problems(BEFORE, dict(BEFORE), ["a.json"], lines, [_record()], _row())
    assert any("a second time" in problem for problem in problems)


def test_criterion_six_names_a_second_profile_file():
    problems = st.second_start_problems(BEFORE, dict(BEFORE), ["a.json", "b.json"], [])
    assert any("2 profile files" in problem for problem in problems)


def test_criterion_six_names_a_changed_machine_table():
    after = {"machines": {"workstation": {"profile": "0000000000000001"}}, "guided": BEFORE["guided"]}
    assert any("[machines]" in problem for problem in st.second_start_problems(BEFORE, after, ["a.json"], []))


def test_criterion_six_names_a_clone_question():
    lines = ["profile from home binding is missing. Is this the same machine or a clone?"]
    assert any("clone question" in problem for problem in st.second_start_problems(BEFORE, dict(BEFORE), ["a.json"], lines))


def test_criterion_six_passes_when_the_context_is_kept_offered_and_rendered():
    assert st.stored_context_problems(KEPT, ANSWERED_CONTEXT, GOOD_HEADER, _row()) == []


def test_criterion_six_names_a_context_the_first_run_did_not_keep():
    problems = st.stored_context_problems(BEFORE, ANSWERED_CONTEXT, GOOD_HEADER, _row())
    assert any("[guided].context is None" in problem for problem in problems)


def test_criterion_six_names_a_configuration_that_starts_the_question_at_another_context():
    problems = st.stored_context_problems(KEPT, 4096, GOOD_HEADER, _row())
    assert any("starts the question at 4096" in problem for problem in problems)


def test_criterion_six_names_a_standalone_render_with_another_context():
    header = "Scenario: context 4096 (entered), KV cache f16 (assumed), 1 request"
    problems = st.stored_context_problems(KEPT, ANSWERED_CONTEXT, header, _row())
    assert any("expected context" in problem for problem in problems)


def test_criterion_six_names_a_document_without_a_scenario_line():
    problems = st.stored_context_problems(KEPT, ANSWERED_CONTEXT, None, _row())
    assert any("no Scenario line" in problem for problem in problems)


@pytest.mark.parametrize("row", [None, _row(measurement_group=1)], ids=["absent", "group 1"])
def test_criterion_six_names_a_standalone_render_that_lost_the_measurement(row):
    problems = st.stored_context_problems(KEPT, ANSWERED_CONTEXT, GOOD_HEADER, row)
    assert any("group 0" in problem for problem in problems)


def test_criterion_six_names_a_scale_that_starts_on_another_level():
    problems = st.stored_context_problems(KEPT, ANSWERED_CONTEXT, GOOD_HEADER, _row(), "XXL")

    assert any("starts on level 'XXL'" in problem for problem in problems)


# --- criterion 7 -------------------------------------------------------------------------------------

GOOD_SCALE = [
    "XS      4k      3,000 words     short questions and answers           all 7 packages fit",
    "S       8k      6,000 words     a long conversation                   all 7 packages fit",
    "M       16k     12,000 words    a conversation plus a few documents   5 of 7 packages fit",
    "L       32k     24,000 words    a report or a long contract           5 of 7 packages fit",
    "XL      64k     48,000 words    several documents at once             2 of 7 packages fit",
    "XXL     128k    96,000 words    a whole book                          none of 7 packages fits",
]
GOOD_DIALOG = [
    " ## ## :: ##   ModelRoom",
    " ## ## ## ..   Which local model packages fit your machine.",
    " :: ## .. ..   modelroom 0.1.0",
    "",
    " folder    //models/results        a configuration, nothing rendered yet",
    " daemon    Ollama 0.34.2           reachable, 19 models installed",
    " machine   workstation             one graphics card, 12 GB, 128 GB memory",
    "",
    "Step 1 of 5  Configuration",
    "ok Configuration //models/results/modelroom.toml, 1 machine(s)",
    "Step 2 of 5  Packages",
    "Step 3 of 5  Context",
    "checking workstation   one graphics card, 12 GB, 128 GB memory",
    "Step 4 of 5  Measurement",
    "Step 5 of 5  Results\n #   Model                    Package                  Fit                 Speed"
    "         Memory\n not covered: 7 packages -- a weight file has no size (a Q4, b Q4, c Q4 and 4 more)",
]
GOOD_MODELS = [
    "Qwen3.5-9B                    good          9B       Qwen, unsloth               13.6M    qwen3.5:9b",
    "DeepSeek-R1-0528-Qwen3-8B     good          8B       unsloth                     68k      none known",
]


def test_criterion_seven_passes_on_a_run_that_reads_as_a_guided_dialog():
    assert st.guided_mode_problems(GOOD_DIALOG, GOOD_SCALE, GOOD_MODELS) == []


def test_criterion_seven_names_a_model_list_that_is_empty():
    assert any("no model" in problem for problem in st.guided_mode_problems(GOOD_DIALOG, GOOD_SCALE, []))


def test_criterion_seven_names_a_model_line_that_is_too_wide():
    wide = [GOOD_MODELS[0] + " " * 20]

    assert any("characters wide" in problem for problem in st.model_list_problems(wide))


def test_criterion_seven_names_a_model_line_without_a_fit():
    assert any("carries no fit" in problem for problem in st.model_list_problems(["Qwen3.5-9B    9B"]))


def test_criterion_seven_names_a_result_table_without_the_model_column():
    without = [line for line in GOOD_DIALOG if "Package" not in line]

    assert any("table head" in problem for problem in st.guided_mode_problems(without, GOOD_SCALE, GOOD_MODELS))


def test_criterion_seven_names_a_missing_start_screen():
    without = [line for line in GOOD_DIALOG if "ModelRoom" not in line]

    assert any("start screen" in problem for problem in st.guided_mode_problems(without, GOOD_SCALE))


def test_criterion_seven_names_a_missing_step_head():
    without = [line for line in GOOD_DIALOG if not line.startswith("Step 3")]

    assert any("step heads" in problem for problem in st.guided_mode_problems(without, GOOD_SCALE))


def test_criterion_seven_names_a_scale_without_its_last_column():
    bare = [label.split("  ")[0] for label in GOOD_SCALE]

    assert any("no count of what fits" in problem for problem in st.guided_mode_problems(GOOD_DIALOG, bare))


def test_criterion_seven_names_a_result_view_that_lists_every_package_again():
    one_per_package = [
        line for line in GOOD_DIALOG if "not covered" not in line
    ] + [
        "Step 5 of 5  Results\n #   Model                    Package                  Fit                 Speed"
        "         Memory\n not covered: a Q4 -- a weight file has no size"
    ]

    problems = st.guided_mode_problems(one_per_package, GOOD_SCALE, GOOD_MODELS)

    assert any("not grouped by reason" in problem for problem in problems)


def test_criterion_seven_names_a_step_three_without_its_machine():
    without = [line for line in GOOD_DIALOG if not line.startswith("checking ")]

    assert any("which machine" in problem for problem in st.guided_mode_problems(without, GOOD_SCALE))


# --- the report ------------------------------------------------------------------------------------------


def test_the_report_marks_each_step_and_indents_its_detail():
    text = st.format_report([st.Step("one", True, "fine"), st.Step("two", False, "broken\n")])

    assert "[1] one: OK" in text
    assert "    fine" in text
    assert "[2] two: FAIL" in text
