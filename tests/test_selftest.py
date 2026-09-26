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
_SET_SPEC = importlib.util.spec_from_file_location("modelroom_searchset", REPO / "scripts" / "searchset.py")
searchset = importlib.util.module_from_spec(_SET_SPEC)
_SET_SPEC.loader.exec_module(searchset)

RESULTS = Path("//models/results")


def _config_text(**changes) -> str:
    raw = {
        "schema_version": 3,
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


def test_the_live_self_test_never_pulls_without_being_asked():
    """Both runs say `pull = false` outright: the self-test on a real machine never pulls."""
    assert tomllib.loads(st.answers_toml(st.ANSWERS_FIRST))["pull"] is False
    assert tomllib.loads(st.answers_toml(st.ANSWERS_SECOND))["pull"] is False


# --- criteria 9 and 10: the pull of step 5 -----------------------------------------------------------


@pytest.fixture
def sp():
    """`scripts/selftest_pull.py`, the two pull criteria the self-test imports."""
    spec = importlib.util.spec_from_file_location("modelroom_selftest_pull", REPO / "scripts" / "selftest_pull.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NAME = "hf.co/unsloth/Qwen3.5-9B-GGUF:UD-IQ2_XXS"
PULLED_LINES = [
    " ? Pull #1 into Ollama now? (2.9 GB)  Yes",
    f" pulling   {NAME}",
    f" ✓ Pulled    {NAME} · 2.9 GB · say Yes in step 4 of the next run to measure it",
]
PULL_CALLS = [("GET", "/api/tags", None), ("POST", "/api/pull", {"model": NAME, "stream": True}), ("GET", "/api/tags", None)]


def test_criterion_nine_passes_on_a_pull_that_ended_in_success_and_was_listed(sp):
    assert sp.pulled_problems(PULLED_LINES, PULL_CALLS, NAME) == []


def test_criterion_nine_names_a_pull_without_its_closing_line(sp):
    assert any("Pulled" in problem for problem in sp.pulled_problems(PULLED_LINES[:2], PULL_CALLS, NAME))


def test_criterion_nine_names_a_pull_that_was_never_sent_or_never_checked(sp):
    assert any("POST /api/pull" in problem for problem in sp.pulled_problems(PULLED_LINES, PULL_CALLS[:1], NAME))
    assert any("/api/tags" in problem for problem in sp.pulled_problems(PULLED_LINES, PULL_CALLS[:2], NAME))


INSTALL = f" install   #1  ollama pull {NAME}"


@pytest.mark.parametrize(
    "tail",
    [
        [" ? Pull #1 into Ollama now? (2.9 GB)  No"],
        [" install   #1 is local already · say Yes in step 4 to measure it"],
        [],  # a daemon the start screen did not reach: no question at all
    ],
)
def test_criterion_ten_passes_on_a_run_that_did_not_pull(sp, tail):
    assert sp.declined_problems([INSTALL, " ✓ Results         docs\\models.md", *tail]) == []


@pytest.mark.parametrize(
    "lines, words",
    [
        ([INSTALL, " ? Pull #1 into Ollama now? (2.9 GB)  Yes"], "Yes"),
        ([INSTALL, f" pulling   {NAME}"], "pulled"),
        ([INSTALL, f" ✓ Pulled    {NAME} · 2.9 GB"], "pulled"),
        ([" ✓ Results         docs\\models.md"], "install line"),
    ],
)
def test_criterion_ten_names_a_run_that_pulled_or_lost_its_install_line(sp, lines, words):
    assert any(words in problem for problem in sp.declined_problems(lines))


# --- criterion 11: a machine entered by hand, in a run of its own ----------------------------------------


@pytest.fixture
def se():
    """`scripts/selftest_entered.py`, the criterion of a machine entered by hand."""
    spec = importlib.util.spec_from_file_location("modelroom_selftest_entered", REPO / "scripts" / "selftest_entered.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entered_payload(**studio) -> dict:
    """A document of the run: this machine without a profile, and `studio` entered by hand."""
    block = {
        "machine": "studio",
        "status": "ranked",
        "label": "studio (entered)",
        "profile": {"origin": "entered", "ram_physical_source": "entered", "gpu_state": "unified_memory"},
        "ranked": [{"fit": {"mode": "gpu", "pool_gib": 23.0}, "note": {"origin": "computed", "code": "fits_in_shared_memory"}}],
        "too_tight": [],
    }
    block.update(studio)
    own = {"machine": "workstation", "status": "no_profile", "label": "workstation", "profile": None, "ranked": [], "too_tight": []}
    return {"machines": [own, block]}


def test_the_entered_answer_file_marks_only_the_machine_entered_by_hand(se):
    from modelroom.guided import QUESTIONS

    raw = tomllib.loads(st.answers_toml(se.ENTERED_ANSWERS))
    assert raw["machines"] == ["enter"]
    assert (raw["entered_name"], raw["entered_ram"], raw["entered_gpu"]) == ("studio", 32, "unified")
    assert raw["pull"] is False
    assert set(se.ENTERED_ANSWERS) <= set(QUESTIONS)


def test_the_first_two_answer_files_stay_as_they_were():
    """Criterion 6 counts one profile file after the second run; nothing is entered there."""
    assert st.ANSWERS_FIRST["machines"] == ["this-machine"]
    assert not any(key.startswith("entered_") for key in (*st.ANSWERS_FIRST, *st.ANSWERS_SECOND))


def test_criterion_eleven_passes_on_a_document_with_the_machine_entered_by_hand(se):
    assert se.entered_problems(_entered_payload(), "workstation") == []


@pytest.mark.parametrize(
    "studio, words",
    [
        ({"label": "studio"}, "(entered)"),
        ({"status": "no_profile", "profile": None, "ranked": []}, "no block"),
        ({"ranked": []}, "no fit"),
        ({"ranked": [{"fit": {"mode": "gpu", "pool_gib": 31.0}, "note": {"origin": "computed", "code": "fits_in_shared_memory"}}]}, "23"),
        ({"ranked": [{"fit": {"mode": "gpu", "pool_gib": 23.0}, "note": {"origin": "measured", "code": "measured"}}]}, "computed"),
        ({"ranked": [{"fit": {"mode": "gpu", "pool_gib": 23.0}, "note": {"origin": "computed", "code": "fits_in_gpu"}}]}, "shared memory"),
        ({"ranked": [{"fit": {"mode": "cpu", "pool_gib": 23.0}, "note": {"origin": "computed", "code": "fits_in_ram"}}]}, "shared memory"),
        (
            {
                "profile": {"origin": "entered", "ram_physical_source": "entered", "gpu_state": "none"},
                "ranked": [{"fit": {"mode": "cpu", "pool_gib": 23.0}, "note": {"origin": "computed", "code": "fits_in_ram"}}],
            },
            "unified memory",
        ),
    ],
)
def test_criterion_eleven_names_what_the_machine_entered_by_hand_lacks(se, studio, words):
    assert any(words in problem for problem in se.entered_problems(_entered_payload(**studio), "workstation"))


def test_criterion_eleven_names_a_document_without_this_machines_own_entry(se):
    payload = _entered_payload()
    payload["machines"] = payload["machines"][1:]

    assert any("machine blocks: 1, not two" in problem for problem in se.entered_problems(payload, "workstation"))
    assert any("no_profile" in problem for problem in se.entered_problems(payload, "workstation"))


def test_criterion_eleven_passes_on_a_real_isolated_run(se, tmp_path: Path):
    """The whole criterion, in process: its own results folder, its own pointer, no daemon, no net."""
    from datetime import datetime, timezone

    steps = se.entered_steps(tmp_path, st.answers_toml, datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc))

    assert len(steps) == 1
    name, ok, detail = steps[0]
    assert name.startswith("(11) ")
    assert ok, detail
    assert "23.0" in detail


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


GOOD_SEARCH_LOG = {
    "schema_version": 3,
    "word": "qwen",
    "mode": "word",
    "filter_owners": True,
    "run_at": "2026-09-25T08:00:00+00:00",
    "accounts": [
        {"account": "Qwen", "class": "publisher", "hits": 1, "page_full": False},
        {"account": "unsloth", "class": "packager", "hits": 3, "page_full": False},
    ],
    "requests": 2,
    "budget": {"used": 3, "limit": 150},
    "resolved": 3,
    "unresolved": [{"reason": "the repository does not say it packages a base model", "count": 5}],
    "models": [
        {"base_model": "Qwen/Qwen3.5-9B", "age": "legacy", "successor": "Qwen/Qwen3.8-27B", "release_basis": "computed"},
        {"base_model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "age": "latest", "successor": None,
         "release_basis": "computed"},
    ],
}
GOOD_SEARCH = [
    " searched Hugging Face at the publisher Qwen and the five listed packagers",
    " 8 repositories, 3 of them models you can pick from",
]


def test_criterion_three_passes_on_a_real_search_log():
    assert st.search_problems(GOOD_SEARCH_LOG, GOOD_SEARCH) == []


def test_criterion_three_names_a_log_of_another_schema():
    problems = st.search_problems({"schema_version": 1}, GOOD_SEARCH)

    assert problems == ["search.json has schema_version 1, expected 3"]


def test_criterion_three_names_a_log_of_the_schema_before_the_release_models():
    """A log of schema 2 carries no `models`: a run before the computed release is no evidence."""
    log = {key: value for key, value in GOOD_SEARCH_LOG.items() if key != "models"} | {"schema_version": 2}

    assert st.search_problems(log, GOOD_SEARCH) == ["search.json has schema_version 2, expected 3"]


@pytest.mark.parametrize("models", [None, [], "Qwen/Qwen3.5-9B"])
def test_criterion_three_names_a_log_without_the_release_of_its_models(models):
    log = GOOD_SEARCH_LOG | {"models": models}

    assert st.search_problems(log, GOOD_SEARCH) == ["search.json names no release for the models it resolved"]


def test_criterion_three_names_a_log_of_another_mode():
    log = GOOD_SEARCH_LOG | {"mode": "catalog"}

    assert any("names mode 'catalog'" in problem for problem in st.search_problems(log, GOOD_SEARCH))


def test_criterion_three_names_a_search_that_resolved_nothing():
    log = GOOD_SEARCH_LOG | {"resolved": 0}

    assert any("resolved 0" in problem for problem in st.search_problems(log, GOOD_SEARCH))


def test_criterion_three_names_a_log_with_no_reason_at_all():
    log = GOOD_SEARCH_LOG | {"unresolved": []}

    assert any("names no reason" in problem for problem in st.search_problems(log, GOOD_SEARCH))


def test_criterion_three_names_a_reason_that_is_empty_or_covers_nothing():
    log = GOOD_SEARCH_LOG | {"unresolved": [{"reason": "", "count": 0}]}

    assert any("empty or covers no repository" in problem for problem in st.search_problems(log, GOOD_SEARCH))


def test_criterion_three_names_a_log_that_asked_no_publisher():
    log = GOOD_SEARCH_LOG | {"accounts": [{"account": "unsloth", "class": "packager", "hits": 3, "page_full": False}]}

    assert any("no publisher account" in problem for problem in st.search_problems(log, GOOD_SEARCH))


def test_criterion_three_names_a_screen_that_says_nothing_about_the_search():
    problems = st.search_problems(GOOD_SEARCH_LOG, ["nothing here"])

    assert any("does not say where the search asked" in problem for problem in problems)
    assert any("how much of the answer is a choice" in problem for problem in problems)


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
    lines = [f"   measured {st.DEEPSEEK_OLLAMA_NAME}: 58.4 tok/s (57.0–60.0), context 8k"]
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
def _head(number: int, name: str) -> str:
    """One step head as the screen draws it: a rule, the step, its name, a rule to 72 characters."""
    return " " + f"-- Step {number} of 5  {name} ".ljust(72, "-")


GOOD_DIALOG = [
    " ## ## :: ##   ModelRoom",
    " ## ## ## ..   Which local model packages fit your machine.",
    " :: ## .. ..   modelroom 0.1.0",
    "",
    " folder    //models/results        a configuration, nothing rendered yet",
    " daemon    Ollama 0.34.2           reachable, 19 models installed",
    " machine   workstation             one graphics card, 12 GB, 128 GB memory",
    "",
    _head(1, "Configuration"),
    " ? Where should results live?  this folder",
    " ? Which machines should the result cover?  this machine",
    "   measured: one graphics card, 12 GB, 128 GB memory, confirmed by llmfit",
    " ok Configuration   //models/results, 1 machine",
    _head(2, "Packages"),
    " ? What are you looking for?  qwen",
    " ? Show only repositories of a publisher or a listed packager?  Yes",
    " ok Packages        2 models, 25 packages from 3 repositories",
    _head(3, "Context"),
    " ? How much text should a model handle at once?  L  32k  24,000 words  a report",
    " ok Context         L 32k",
    _head(4, "Measurement"),
    " - Measurement     none in this run",
    _head(5, "Results"),
    "",
    " folder    //models/results",
    " machine   workstation             12 GB graphics, 128 GB memory, measured today",
    " models    Qwen3.5-9B              25 packages from unsloth and Ollama",
    " context   L 32k                   24,000 words, a report or a long contract",
    " speed     not measured            say Yes in step 4 to measure an installed package",
    " result    docs/models.md          25 packages ranked, none too tight",
    "",
    " workstation - context L 32k",
    " #    Model                     Package                 Fit                   Speed         Memory",
    " 1    Qwen3.5-9B                unsloth - UD-IQ2_XXS    good                  -             8.3 GB",
    " shown     1 of 25 packages",
    " memory    #1 fits into graphics memory, 11.0 GB free after the reserve",
    " speed     nothing measured in the rows shown - say Yes in step 4 to measure an installed package",
    "",
    " install   #1  ollama pull hf.co/unsloth/Qwen3.5-9B-GGUF:UD-IQ2_XXS",
    "",
    " ok Results         docs/models.md",
]
GOOD_MODELS = [
    "Qwen3.5-9B              good          9B     legacy   Qwen, unsloth   13.6M   qwen3.5:9b",
    "DeepSeek-R1-0528-Qwen3-  good          8B     –        unsloth         68k     –",
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
    without = [line for line in GOOD_DIALOG if "Step 3" not in line]

    assert any("step heads" in problem for problem in st.guided_mode_problems(without, GOOD_SCALE))


def test_criterion_seven_names_a_step_head_that_is_not_seventy_two_wide():
    short = [line.rstrip("-") if "Step 3" in line else line for line in GOOD_DIALOG]

    assert any("characters wide, expected 72" in problem for problem in st.step_head_problems(short))


def test_criterion_seven_names_a_scale_without_its_last_column():
    bare = [label.split("  ")[0] for label in GOOD_SCALE]

    assert any("no count of what fits" in problem for problem in st.guided_mode_problems(GOOD_DIALOG, bare))


def test_criterion_seven_names_a_run_without_answer_lines():
    without = [line for line in GOOD_DIALOG if not line.strip().startswith("? ")]

    assert any("answer lines" in problem for problem in st.answer_line_problems(without))


def test_criterion_seven_names_the_libraries_own_answer():
    with_library = [*GOOD_DIALOG, "? Which of these models should the result cover? done (2 selections)"]

    assert any("library wrote an answer" in problem for problem in st.answer_line_problems(with_library))


@pytest.mark.parametrize(
    "line",
    [
        "wrote x with y as the writer of this results folder",
        "kept context 32768 in x",
        "checking workstation   one graphics card",
        "Written to a   and   b",
    ],
)
def test_criterion_seven_names_a_line_the_screen_no_longer_carries(line):
    problems = st.answer_line_problems([*GOOD_DIALOG, line])

    assert any("no longer carries" in problem for problem in problems)


def test_criterion_seven_names_a_run_that_closes_without_the_card():
    without = [line for line in GOOD_DIALOG if not line.strip().startswith(("models ", "speed ", "result "))]

    assert any("card of step 5" in problem for problem in st.card_problems(without))


def test_criterion_seven_names_a_result_view_that_repeats_one_note_per_row():
    repeated = [*GOOD_DIALOG, " #2 fits into graphics memory", " #3 needs system memory", " #4 needs system memory"]

    problems = st.result_table_problems(repeated)

    assert any("notes about the memory pool" in problem for problem in problems)


def test_criterion_seven_names_a_row_that_still_says_unknown():
    with_unknown = [
        *GOOD_DIALOG,
        " 1    Qwen3.5-9B                unsloth - Q4_K_M        good                  unknown       8.3 GB",
    ]

    assert any("says `unknown`" in problem for problem in st.result_table_problems(with_unknown))


# --- criterion 8: the search test set --------------------------------------------------------------------


def test_the_shipped_test_set_loads_and_holds_the_cases_the_decision_named():
    cases = searchset.load_testset()

    typed = [case["input"] for case in cases]
    for expected in ("qwen", "qwen 3.5 9b", "Qwen3.5-9B", "unsloth/Qwen3.5-9B-GGUF", "mistral", "deepseek r1", "qwn", ""):
        assert expected in typed, expected
    assert all(isinstance(case["filter"], bool) for case in cases)
    assert all(case.get("why") for case in cases)


def test_a_test_set_of_another_schema_is_a_setup_error(tmp_path: Path):
    path = tmp_path / "set.toml"
    path.write_text("schema_version = 9\n[[cases]]\ninput = 'qwen'\nfilter = true\nmodels = []\n", encoding="utf-8")

    with pytest.raises(searchset.TestsetError) as exc:
        searchset.load_testset(path)

    assert "schema_version is 9" in str(exc.value)


@pytest.mark.parametrize(
    "body, message",
    [
        ("schema_version = 1\n", "holds no case"),
        ("schema_version = 1\n[[cases]]\nfilter = true\nmodels = []\n", "has no input"),
        ("schema_version = 1\n[[cases]]\ninput = 'qwen'\nmodels = []\n", "no filter or no models"),
        ("not toml at all = \n", "cannot read the search test set"),
    ],
)
def test_a_test_set_that_is_not_a_test_set_is_a_setup_error(tmp_path: Path, body, message):
    path = tmp_path / "set.toml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(searchset.TestsetError) as exc:
        searchset.load_testset(path)

    assert message in str(exc.value)


def test_a_case_whose_models_all_came_back_is_reported_as_ok():
    case = {"input": "qwen 3.5 9b", "filter": True, "models": ["Qwen/Qwen3.5-9B"]}

    line = searchset.testset_line(case, ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B"], [])

    assert line == "OK   qwen 3.5 9b: 2 models"


def test_a_missing_model_is_a_warning_and_names_what_is_missing():
    case = {"input": "mistral", "filter": True, "models": ["mistralai/Ministral-3-8B-Instruct-2512"]}

    line = searchset.testset_line(case, [], [])

    assert line.startswith("WARN mistral: 0 models, missing mistralai/Ministral-3-8B-Instruct-2512")


def test_a_typo_case_reports_the_candidates_it_was_offered_instead():
    case = {"input": "qwn", "filter": True, "models": []}

    line = searchset.testset_line(case, [], ["qwen", "qwen3.5"])

    assert line == "OK   qwn: 0 models, did you mean qwen, qwen3.5"


def test_a_case_with_no_word_says_so_instead_of_showing_a_blank():
    case = {"input": "", "filter": True, "models": []}

    assert searchset.testset_line(case, [], []) == "OK   (no word): 0 models"


def test_a_live_search_that_did_not_answer_is_a_warning_and_never_a_failure():
    case = {"input": "qwen", "filter": True, "models": ["Qwen/Qwen3.8-27B"]}

    line = searchset.testset_line(case, [], [], "SearchError: unexpected status 503")

    assert line == "WARN qwen: SearchError: unexpected status 503"


def test_the_criterion_watches_the_report_and_not_the_hub():
    cases = [{"input": "qwen", "filter": True, "models": []}]

    assert searchset.testset_problems(cases, ["WARN qwen: 0 models, missing x"]) == []
    assert searchset.testset_problems(cases, []) == ["0 report lines for 1 cases"]
    assert any("carries no verdict" in problem for problem in searchset.testset_problems(cases, ["qwen: nothing"]))


# --- the report ------------------------------------------------------------------------------------------


def test_the_report_marks_each_step_and_indents_its_detail():
    text = st.format_report([st.Step("one", True, "fine"), st.Step("two", False, "broken\n")])

    assert "[1] one: OK" in text
    assert "    fine" in text
    assert "[2] two: FAIL" in text
