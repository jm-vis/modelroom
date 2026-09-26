"""Tests for modelroom.guided_context and fit.count_fitting: the size scale of step 3.

The last column of the scale is a fit, so it is checked against the fit itself: every text is
compared with what `compute_fit_v2` says about the very same packages at the very same context,
package by package. A scale that agreed with nothing but itself would be a second calculation,
which is exactly what this package does not have.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.config import Configuration, MachineConfig
from modelroom.contracts import BaseModelSpec, Package
from modelroom.examples import EXAMPLES
from modelroom.fit import compute_fit_v2, count_fitting
from modelroom.guided import GuidedError
from modelroom.guided_context import (
    DEFAULT_CONTEXT,
    LEVELS,
    NO_MACHINE_LINE,
    NO_PACKAGES,
    NUMBER_VALUE,
    USERS_EXPLANATION,
    USERS_QUESTION,
    Checked,
    RequestsAnswer,
    answer_words,
    context_cap,
    context_scenario,
    fit_texts,
    machine_checked,
    requests_answer,
    scale_choices,
    tokens_of,
    users_choices,
)
from modelroom.profile import HardwareProfile
from modelroom.ranking import rank_packages

GIB = 1024**3
MACHINE = MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)
NOW = datetime(2026, 9, 23, 8, 0, 0, tzinfo=timezone.utc)


def _base_model(max_context: int | None = 131072, repo: str | None = None) -> BaseModelSpec:
    payload = copy.deepcopy(EXAMPLES["BaseModelSpec"])
    payload["architecture"]["max_context"] = max_context
    if repo is not None:
        payload["hf_repo"] = repo
        payload["architecture"]["source_repo"] = repo
    return BaseModelSpec.model_validate(payload)


def _package(weights_gib: float, repo: str = "packager/Nova-8B-GGUF") -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["files"][0]["size_bytes"] = int(weights_gib * GIB)
    payload["repo"] = repo
    return Package.model_validate(payload)


def _profile(vram_gib: float = 11.94) -> HardwareProfile:
    """The example profile with a graphics card of `vram_gib`; llmfit confirms the same number."""
    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    payload["vram_gib"] = vram_gib
    payload["llmfit_crosscheck"]["vram"] = {"status": "confirmed", "own_gib": vram_gib, "llmfit_gib": vram_gib}
    return HardwareProfile.model_validate(payload)


def _checked(profile: HardwareProfile | None = None) -> Checked:
    if profile is None:
        return Checked(reason="no measured machine")
    return Checked(name="workstation", profile=profile, machine_config=MACHINE)


PACKAGES = [_package(3.0, "a/Nova-8B-GGUF"), _package(6.0, "b/Nova-8B-GGUF"), _package(9.0, "c/Nova-8B-GGUF")]
BY_REPO = {PACKAGES[0].base_model_hf_repo: _base_model()}


def _fitting_by_hand(context: int) -> int:
    """How many packages the ranking ranks here, computed package by package with the fit itself."""
    fits = [(package, compute_fit_v2(_profile(), package, _base_model(), context_scenario(context), MACHINE)) for package in PACKAGES]
    return len(rank_packages(fits, [], context_scenario(context)).ranked)


# --- the levels ------------------------------------------------------------------------------------


def test_the_six_levels_are_the_powers_of_two_from_4k_to_128k():
    assert [(level.name, level.tokens) for level in LEVELS] == [
        ("XS", 4096),
        ("S", 8192),
        ("M", 16384),
        ("L", 32768),
        ("XL", 65536),
        ("XXL", 131072),
    ]


def test_every_level_shows_its_context_and_roughly_its_words():
    assert [level.shown_tokens for level in LEVELS] == ["4k", "8k", "16k", "32k", "64k", "128k"]
    assert [level.words for level in LEVELS] == [
        "3,000 words",
        "6,000 words",
        "12,000 words",
        "24,000 words",
        "48,000 words",
        "96,000 words",
    ]


def test_the_default_level_is_l():
    assert DEFAULT_CONTEXT == 32768


@pytest.mark.parametrize("answer, expected", [("XS", 4096), ("XXL", 131072), ("5000", 5000), ("32768", 32768)])
def test_a_level_or_a_number_becomes_a_number_of_tokens(answer, expected):
    assert tokens_of(answer) == expected


def test_everything_the_dialog_writes_is_entered_even_the_number_of_the_three_commands():
    """8192 is `default` only where nobody chose it; in the scale it is the level `S`."""
    assert context_scenario(8192).context_origin == "entered"
    assert context_scenario(32768).context_origin == "entered"


# --- the last column -------------------------------------------------------------------------------


def test_every_line_says_how_many_packages_fit_and_agrees_with_the_fit_itself():
    contexts = [level.tokens for level in LEVELS]

    texts = fit_texts(PACKAGES, BY_REPO, _checked(_profile()), contexts)

    for context in contexts:
        fitting = _fitting_by_hand(context)
        assert count_fitting(_profile(), MACHINE, PACKAGES, BY_REPO, context) == fitting
        if fitting == len(PACKAGES):
            assert texts[context] == "all 3 packages fit"
        elif fitting == 0:
            assert texts[context] == "none of 3 packages fits"
        else:
            assert texts[context] == f"{fitting} of 3 packages fit"


def test_the_column_really_changes_with_the_level():
    """A scale whose column said the same everywhere would be worth nothing to look at."""
    contexts = [level.tokens for level in LEVELS]

    texts = fit_texts(PACKAGES, BY_REPO, _checked(_profile()), contexts)

    assert len(set(texts.values())) > 1


def test_a_machine_that_nothing_fits_says_so_in_the_plural_of_none():
    tight = Checked(name="workstation", profile=_profile(), machine_config=MachineConfig(
        reserve_ram_gib=31.0, reserve_vram_gib=11.0, writer=True
    ))

    assert fit_texts(PACKAGES, BY_REPO, tight, [131072])[131072] == "none of 3 packages fits"


def test_a_bigger_context_can_make_a_package_fit_again_off_the_graphics_card():
    """The count of the scale is not a falling line, and it must not be made into one.

    Fit v1 judges a package against the graphics memory as long as it fits there, and against
    system memory once it does not (`fit._choose_pool`). The 9 GiB package is too tight on the
    card by a hair at 4k; at 8k it no longer fits the card at all, is judged against the much
    larger system memory, and comes out `good`. The scale shows what the ranking of this same
    folder will show, whatever that is -- it has no second opinion of its own.
    """
    assert count_fitting(_profile(), MACHINE, PACKAGES, BY_REPO, 4096) == 2
    assert count_fitting(_profile(), MACHINE, PACKAGES, BY_REPO, 8192) == 3


def test_without_a_measured_machine_there_is_no_column_at_all():
    assert fit_texts(PACKAGES, BY_REPO, _checked(None), [4096]) == {}


def test_without_packages_every_line_says_so():
    assert fit_texts([], {}, _checked(_profile()), [4096, 8192]) == {4096: NO_PACKAGES, 8192: NO_PACKAGES}


def test_one_package_is_counted_in_the_singular():
    texts = fit_texts(PACKAGES[:1], BY_REPO, _checked(_profile()), [4096])

    assert texts[4096] == "all 1 package fit"


def test_a_package_whose_base_model_is_unknown_is_not_counted():
    assert count_fitting(_profile(), MACHINE, PACKAGES, {}, 4096) == 0


# --- which machine the scale is about ---------------------------------------------------------------


def _folder(tmp_path, profile: HardwareProfile, machine: dict) -> tuple[Path, Configuration]:
    """A results folder that holds `profile`, with the pointer binding this machine to it."""
    from modelroom.binding import GuidedPointer, write_pointer
    from modelroom.state import atomic_write_json

    results = tmp_path / "results"
    config = Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [],
            "machines": {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, **machine}},
            "paths": {"state": str(results / "state"), "markdown": str(results / "docs" / "models.md")},
        }
    )
    atomic_write_json(
        config.paths.hardware_dir / f"{profile.profile_id}.json", profile.model_dump(mode="json")
    )
    pointer = tmp_path / "home" / "guided.json"
    write_pointer(
        pointer,
        GuidedPointer(schema_version=1, current=str(results), bindings={str(results): profile.profile_id}),
    )
    return pointer, config


def test_the_scale_is_about_the_machine_the_pointer_binds_this_folder_to(tmp_path):
    profile = _profile()
    pointer, config = _folder(tmp_path, profile, {"writer": True, "profile": profile.profile_id})

    checked = machine_checked(pointer, config, tmp_path / "results")

    assert (checked.name, checked.reason) == ("workstation", None)
    assert checked.profile is not None and checked.profile.profile_id == profile.profile_id
    assert checked.machine_config is not None and checked.machine_config.reserve_vram_gib == 1.0


def test_no_machine_entry_for_the_bound_profile_means_no_column(tmp_path):
    """Another machine's reserves under another machine's name would be a made-up number."""
    profile = _profile()
    pointer, config = _folder(tmp_path, profile, {"writer": True})  # no `profile` in the entry

    checked = machine_checked(pointer, config, tmp_path / "results")

    assert checked.profile is None and checked.machine_config is None
    assert profile.profile_id in checked.reason
    assert fit_texts(PACKAGES, BY_REPO, checked, [4096]) == {}


def test_without_a_binding_the_scale_says_there_is_no_measured_machine(tmp_path):
    from modelroom.binding import GuidedPointer, write_pointer

    profile = _profile()
    pointer, config = _folder(tmp_path, profile, {"writer": True, "profile": profile.profile_id})
    write_pointer(pointer, GuidedPointer(schema_version=1, current=None, bindings={}))

    checked = machine_checked(pointer, config, tmp_path / "results")

    assert checked.reason == NO_MACHINE_LINE


def test_a_profile_the_fit_may_not_compute_for_gets_no_column_either(tmp_path):
    blocked = _profile().model_copy(update={"gpu_state": "multi_gpu_not_covered"})
    pointer, config = _folder(tmp_path, blocked, {"writer": True, "profile": blocked.profile_id})

    checked = machine_checked(pointer, config, tmp_path / "results")

    assert checked.profile is None
    assert checked.reason.startswith("no fit on this machine: gpu_state multi_gpu_not_covered")


# --- the cap ---------------------------------------------------------------------------------------


def test_the_cap_is_the_smallest_window_the_base_models_declare():
    models = [_base_model(131072, "a/Nova-8B"), _base_model(32768, "b/Small-8B")]

    assert context_cap(models) == (32768, "b/Small-8B")


def test_a_base_model_without_a_window_caps_nothing():
    models = [_base_model(32768, "b/Small-8B"), _base_model(None, "c/Silent-8B")]

    assert context_cap(models) is None


def test_no_base_model_at_all_caps_nothing():
    assert context_cap([]) is None


# --- the list -----------------------------------------------------------------------------------------


def test_the_pointer_starts_on_the_kept_level_and_the_number_entry_is_last():
    choices = scale_choices(LEVELS, 16384, {}, None)

    assert [choice.value for choice in choices][-1] == NUMBER_VALUE
    assert [choice.value for choice in choices if choice.checked] == ["M"]


def test_a_kept_context_that_is_no_level_gets_a_line_of_its_own():
    choices = scale_choices(LEVELS, 5000, {}, None)

    custom = next(choice for choice in choices if choice.value == "5000")
    assert custom.checked is True
    assert "kept in this folder" in custom.label
    assert [choice.value for choice in choices if choice.checked] == ["5000"]


def test_the_last_column_stands_in_every_line_of_the_list():
    fits = fit_texts(PACKAGES, BY_REPO, _checked(_profile()), [level.tokens for level in LEVELS])

    choices = scale_choices(LEVELS, 32768, fits, None)

    for level in LEVELS:
        label = next(choice.label for choice in choices if choice.value == level.name)
        assert label.endswith(fits[level.tokens])


# --- the head count: how many people, and the requests the ranking assumes for them ----------------


def test_the_question_and_the_derivation_under_it_say_what_the_number_is():
    """A head count is not a number of requests; the rule between them is said where it is asked."""
    assert USERS_QUESTION == "How many people use it on a typical day?"
    assert USERS_EXPLANATION == "assumes 1 in 10 of them at once (rule of thumb, not measured)"


@pytest.mark.parametrize(
    "answered, kept, expected",
    [
        ("25", None, RequestsAnswer(25, 3, "from_users")),
        # One person is a head count too: `from_users`, never `default` -- the user named it.
        ("1", None, RequestsAnswer(1, 1, "from_users")),
        ("10", None, RequestsAnswer(10, 1, "from_users")),
        ("11", None, RequestsAnswer(11, 2, "from_users")),
        ("10240", None, RequestsAnswer(10240, 1024, "from_users")),
        ("12 requests", None, RequestsAnswer(None, 12, "entered")),
        # Requests named outright keep the head count the folder holds, for the display.
        ("12 requests", 25, RequestsAnswer(25, 12, "entered")),
        ("1 request", None, RequestsAnswer(None, 1, "entered")),
        ("1024 requests", None, RequestsAnswer(None, 1024, "entered")),
    ],
)
def test_an_answer_is_a_head_count_or_the_requests_named_outright(answered, kept, expected):
    assert requests_answer(answered, kept) == expected


@pytest.mark.parametrize(
    "answered",
    ["0", "-1", "abc", "3 people", "10241", "0 requests", "1025 requests", "", "2.5", "requests", "+5", "1_000", "9" * 400],
)
def test_anything_else_ends_the_run_and_names_both_forms_with_their_bounds(answered):
    with pytest.raises(GuidedError) as raised:
        requests_answer(answered, None)

    message = str(raised.value)
    assert repr(answered) in message
    assert "a whole number of people (1 to 10240)" in message
    assert "`N requests` (1 to 1024)" in message


@pytest.mark.parametrize(
    "answer, words",
    [
        (RequestsAnswer(25, 3, "from_users"), "25 (3 requests)"),
        (RequestsAnswer(1, 1, "from_users"), "1 (1 request)"),
        (RequestsAnswer(None, 12, "entered"), "12 requests"),
        (RequestsAnswer(25, 1, "entered"), "1 request"),
    ],
)
def test_the_answer_line_names_the_requests_behind_the_head_count(answer, words):
    assert answer_words(answer) == words


def test_the_list_offers_five_head_counts_and_a_number_of_your_own_starting_on_one():
    choices = users_choices(1)

    assert [choice.value for choice in choices] == ["1", "5", "10", "25", "100", NUMBER_VALUE]
    assert [choice.value for choice in choices if choice.checked] == ["1"]
    assert "3 requests" in next(choice.label for choice in choices if choice.value == "25")


def test_a_kept_head_count_puts_the_pointer_on_its_line():
    assert [choice.value for choice in users_choices(25) if choice.checked] == ["25"]


def test_a_kept_head_count_that_is_no_line_of_the_list_gets_a_line_of_its_own():
    choices = users_choices(42)

    kept = next(choice for choice in choices if choice.value == "42")
    assert kept.checked is True
    assert "kept in this folder" in kept.label and "5 requests" in kept.label
    assert [choice.value for choice in choices if choice.checked] == ["42"]
    assert choices[-1].value == NUMBER_VALUE


def test_the_scenario_of_the_run_carries_the_requests_of_the_answer():
    scenario = context_scenario(32768, 3)

    assert (scenario.context_requested, scenario.requests, scenario.context_origin) == (32768, 3, "entered")
    assert context_scenario(32768).requests == 1


def test_the_last_column_counts_with_the_requests_of_the_run():
    """Three requests hold three KV caches: at 128k the example machine keeps two packages for one
    request and none for three -- and the scale says so, as the ranking of the run will."""
    one = fit_texts(PACKAGES, BY_REPO, _checked(_profile()), [131072])
    three = fit_texts(PACKAGES, BY_REPO, _checked(_profile()), [131072], requests=3)

    assert count_fitting(_profile(), MACHINE, PACKAGES, BY_REPO, 131072, 3) == 0
    assert one[131072] == "2 of 3 packages fit"
    assert three[131072] == "none of 3 packages fits"
