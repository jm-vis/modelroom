"""End-to-end tests for the guided mode: `modelroom` with no subcommand.

Every run goes through injected layers -- a `FixtureTransport` for the search and the fetch, a
`Probes` whose sources are all fixtures, a fixed clock, a pointer file inside `tmp_path` and a
`FileAsker` for the dialog. No process is spawned, no network is touched, and nothing outside
`tmp_path` is written; in particular the real pointer file in the user's home folder is never
read or written.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.binding import GuidedPointer, read_pointer, write_pointer
from modelroom.cli import main
from modelroom.config import load_config
from modelroom.dialog import AnswerMissingError, Canceled, Choice, FileAsker
from modelroom.examples import EXAMPLES
from modelroom.guided import GuidedError, run_guided
from modelroom.guided_loadtest import NO_CANDIDATE_LINE
from modelroom.intro import STEP_NAMES
from modelroom.profile import HardwareProfile
from modelroom.state import acquire_lock, atomic_write_json, release_lock

from fixture_support import (
    DEEPSEEK_OLLAMA_NAME,
    build_transport,
    guided_transport_mapping,
    loadtest_daemon,
    offline_daemon,
    ps_answer,
    windows_probes,
    v2_profile,
)

RUN1 = datetime(2026, 9, 23, 8, 0, 0, tzinfo=timezone.utc)
RUN2 = datetime(2026, 9, 23, 9, 0, 0, tzinfo=timezone.utc)
# A third stamp: a `fetch` is refused when its run is not newer than the stored snapshot, so a
# test with three runs needs three clocks.
RUN3 = datetime(2026, 9, 23, 10, 0, 0, tzinfo=timezone.utc)
UNSLOTH = "unsloth/Qwen3.5-9B-GGUF"
QWEN_GGUF = "Qwen/Qwen3.5-9B-GGUF"
# The one repository in the pinned search answer whose base model fit v1 can judge, and whose
# `Q4_K_M` package is the one the load test fixtures show as installed.
DEEPSEEK = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
DEEPSEEK_BASE = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


class _WatchingAsker(FileAsker):
    """A `FileAsker` that keeps the choices every list was built with, for the list's own shape."""

    def __init__(self, answers: dict, lines: list[str]) -> None:
        super().__init__(answers)
        self.choices: dict[str, list[Choice]] = {}
        self._lines = lines

    def checkbox(self, key: str, question: str, choices) -> list[str]:
        self.choices[key] = list(choices)
        return super().checkbox(key, question, choices)

FULL_ANSWERS = {
    "results": "here",
    "machines": ["this-machine"],
    "search": "qwen",
    "filter_owners": True,
    "select": [UNSLOTH, QWEN_GGUF],
    "context": "8192",
}
# The same run, plus the repository the load test measures and its two answers.
LOAD_TEST_ANSWERS = {
    **FULL_ANSWERS,
    "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK],
    "load_test": True,
    "load_test_packages": [DEEPSEEK_OLLAMA_NAME],
}


def _pointer(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".modelroom" / "guided.json"


def _run(tmp_path: Path, answers: dict | None = None, *, here: Path | None = None, now=RUN1, **kwargs) -> tuple[int, list[str]]:
    """One guided run; returns its exit code and every line it printed.

    The daemon is a fixture like every other source: without one of its own a run gets a machine
    with no Ollama daemon at all, so no test here ever reaches the real one on port 11434.
    """
    lines: list[str] = []
    asker = FileAsker(FULL_ANSWERS if answers is None else answers)
    code = run_guided(
        asker,
        here=here if here is not None else tmp_path / "results",
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=kwargs.pop("probes", None) or windows_probes(),
        daemon=kwargs.pop("daemon", None) or offline_daemon(),
        now=now,
        out=lines.append,
        **kwargs,
    )
    return code, lines


def _results(tmp_path: Path) -> Path:
    return tmp_path / "results"


def _config_file(tmp_path: Path) -> Path:
    return _results(tmp_path) / "modelroom.toml"


def _profiles(tmp_path: Path) -> list[Path]:
    folder = _results(tmp_path) / "state" / "hardware"
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


@pytest.fixture(autouse=True)
def _results_folder(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()


# --- the start screen and the five steps ----------------------------------------------------------


def test_the_run_begins_with_the_start_screen_and_walks_five_numbered_steps(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    assert any("ModelRoom" in line for line in lines[:4])
    # Step 5's head comes with the whole result view in one line, so only its first line is a head.
    heads = [line.splitlines()[0] for line in lines if line.startswith("Step ")]
    assert [head.split("  ")[-1] for head in heads] == list(STEP_NAMES)
    assert [head.split(" of ")[0] for head in heads] == ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]


def test_every_step_leaves_one_line_behind_with_what_it_did(tmp_path: Path):
    _code, lines = _run(tmp_path)

    done = [line for line in lines if line.startswith(("ok ", "✓ "))]
    assert [line.split()[1] for line in done] == list(STEP_NAMES[:4])
    assert any("repositories added" in line and "packages fetched" in line for line in done)
    assert any(line.endswith("context S 8k") for line in done)


def test_the_start_screen_names_the_folder_and_the_hardware_of_a_second_run(tmp_path: Path):
    """Once a folder is remembered and measured, the start screen says so before the first question."""
    assert _run(tmp_path)[0] == 0

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    _code, lines = _run(tmp_path, answers, now=RUN2)

    folder = next(line for line in lines if line.strip().startswith("folder"))
    machine = next(line for line in lines if line.strip().startswith("machine"))
    assert str(_results(tmp_path)) in folder
    assert "one graphics card" in machine and "memory" in machine


def test_the_fetch_runs_before_the_context_question(tmp_path: Path):
    """The scale counts the packages of the snapshot, so they have to be there when it is asked."""
    order: list[str] = []

    class _Watching(FileAsker):
        def select_or_text(self, key, question, choices, text_value, text_question):
            order.append(f"asked {key}")
            return super().select_or_text(key, question, choices, text_value, text_question)

    lines: list[str] = []
    run_guided(
        _Watching(FULL_ANSWERS),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lambda line: order.append(line) or lines.append(line),
    )

    assert order.index("Step 2 of 5  Packages") < order.index("asked context")
    fetched = next(index for index, line in enumerate(order) if "packages fetched" in line)
    assert fetched < order.index("asked context")


def test_the_scale_is_asked_with_the_packages_of_this_runs_own_fetch(tmp_path: Path):
    """The last column is not a promise: it is a count over what the snapshot really holds."""
    seen: dict[str, list] = {}

    class _Watching(FileAsker):
        def select_or_text(self, key, question, choices, text_value, text_question):
            seen[key] = list(choices)
            return super().select_or_text(key, question, choices, text_value, text_question)

    run_guided(
        _Watching(FULL_ANSWERS),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lambda _line: None,
    )

    labels = [choice.label for choice in seen["context"]]
    assert all("packages" in label for label in labels[:6])
    assert any("24 packages" in label for label in labels)


def test_the_last_line_says_where_the_result_was_written(tmp_path: Path):
    _code, lines = _run(tmp_path)

    written = next(line for line in lines if line.startswith("Written to "))
    assert str(_results(tmp_path) / "docs" / "models.md") in written
    assert str(_results(tmp_path) / "state") in written


# --- first start: the folder question, the configuration, the pointer file ----------------------


def test_the_first_start_writes_a_configuration_with_this_device_as_the_writer(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    raw = tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    assert raw["machines"]["workstation"]["writer"] is True
    assert Path(raw["guided"]["results"]) == _results(tmp_path)
    assert any("as the writer of this results folder" in line for line in lines)


def test_the_first_start_remembers_the_folder_in_the_pointer_file(tmp_path: Path):
    assert _run(tmp_path)[0] == 0

    pointer = read_pointer(_pointer(tmp_path))
    assert Path(pointer.current) == _results(tmp_path)
    assert pointer.binding_for(_results(tmp_path)) is not None


def test_another_path_is_asked_for_and_used(tmp_path: Path):
    elsewhere = tmp_path / "elsewhere"
    answers = {**FULL_ANSWERS, "results": "path", "results_path": str(elsewhere)}

    code, _lines = _run(tmp_path, answers)

    assert code == 0
    assert (elsewhere / "modelroom.toml").is_file()


def test_a_folder_with_results_but_no_configuration_is_a_question(tmp_path: Path):
    (_results(tmp_path) / "state").mkdir()
    answers = {**FULL_ANSWERS, "write_config": False}

    with pytest.raises(GuidedError, match="no configuration"):
        _run(tmp_path, answers)
    assert not _config_file(tmp_path).exists()


def test_a_folder_with_results_is_used_once_the_question_is_answered_yes(tmp_path: Path):
    (_results(tmp_path) / "state").mkdir()

    code, _lines = _run(tmp_path, {**FULL_ANSWERS, "write_config": True})

    assert code == 0
    assert _config_file(tmp_path).is_file()


def test_the_second_run_reuses_the_configuration_and_the_binding(tmp_path: Path):
    assert _run(tmp_path)[0] == 0
    first = _profiles(tmp_path)
    assert len(first) == 1

    # No folder question this time: the pointer file remembers it. `here` is somewhere else on
    # purpose, so a run that fell back to "this folder" would write a second configuration.
    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    code, lines = _run(tmp_path, answers, here=tmp_path / "somewhere-else", now=RUN2)

    assert code == 0
    # The bound profile is measured again in place: one file, the same id, a newer reading.
    assert [path.name for path in _profiles(tmp_path)] == [first[0].name]
    profile = HardwareProfile.model_validate(json.loads(_profiles(tmp_path)[0].read_text(encoding="utf-8")))
    assert profile.profile_id == first[0].stem
    assert profile.recorded_at == RUN2
    assert not (tmp_path / "somewhere-else" / "modelroom.toml").exists()
    assert not any("clone" in line for line in lines)


def test_a_config_argument_that_points_nowhere_asks_again(tmp_path: Path):
    code, lines = _run(tmp_path, config_arg=tmp_path / "nowhere" / "modelroom.toml")

    assert code == 0
    assert any("there is no configuration at this path" in line for line in lines)
    assert _config_file(tmp_path).is_file()


def test_a_config_argument_wins_over_the_remembered_folder(tmp_path: Path):
    assert _run(tmp_path)[0] == 0
    elsewhere = tmp_path / "elsewhere"
    answers = {**FULL_ANSWERS, "results": "path", "results_path": str(elsewhere)}
    assert _run(tmp_path, answers, now=RUN2)[0] == 0

    # The pointer now remembers `elsewhere`; `--config` must still work on the first folder.
    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    code, lines = _run(tmp_path, answers, config_arg=_config_file(tmp_path), now=RUN3)

    assert code == 0
    assert not any("there is no configuration" in line for line in lines)
    assert read_pointer(_pointer(tmp_path)).current == str(_results(tmp_path))


def test_a_remembered_folder_that_is_gone_asks_again(tmp_path: Path):
    write_pointer(_pointer(tmp_path), GuidedPointer(schema_version=1, current=str(tmp_path / "gone")))

    code, lines = _run(tmp_path)

    assert code == 0
    assert any("no longer holds a modelroom.toml" in line for line in lines)


# --- schema 1 in the folder ----------------------------------------------------------------------


def test_a_schema_one_configuration_is_migrated_before_the_first_write(tmp_path: Path):
    _config_file(tmp_path).write_text(
        "\n".join(
            [
                "schema_version = 1",
                'packagers = ["unsloth"]',
                'publishers = ["Qwen"]',
                "[[families]]",
                'name = "qwen3.5"',
                "  [[families.base_models]]",
                '  hf_repo = "Qwen/Qwen3.5-9B"',
                "[machines.workstation]",
                "reserve_ram_gib = 8",
                "reserve_vram_gib = 1",
                "writer = true",
                "[paths]",
                'state = "state"',
                'markdown = "docs/models.md"',
            ]
        ),
        encoding="utf-8",
    )

    code, lines = _run(tmp_path)

    assert code == 0
    assert (_results(tmp_path) / "modelroom.toml.v1.bak").is_file()
    assert tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))["schema_version"] == 2
    assert any("modelroom.toml" in line for line in lines)
    # The migration lines are for a maintainer; the sentence in front of them is for a user.
    assert (
        "This folder holds a configuration from an earlier version; it was updated, "
        "backup kept: modelroom.toml.v1.bak"
    ) in lines


# --- the machine list ------------------------------------------------------------------------------


def test_the_measured_machine_is_written_into_the_configuration(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    profile = HardwareProfile.model_validate(json.loads(_profiles(tmp_path)[0].read_text(encoding="utf-8")))
    assert profile.gpu_state == "measured"
    assert profile.llmfit_crosscheck.ram_physical.status == "confirmed"
    raw = tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))
    assert raw["machines"]["workstation"]["profile"] == profile.profile_id
    assert any(f"measured as profile {profile.profile_id}" in line for line in lines)


def test_no_machine_chosen_measures_nothing(tmp_path: Path):
    code, _lines = _run(tmp_path, {**FULL_ANSWERS, "machines": []})

    assert code == 0
    assert _profiles(tmp_path) == []


def _recorded_machines(tmp_path: Path, now, answers: dict) -> list:
    """The machine list of one run, as the dialog was offered it."""
    asked: dict[str, list] = {}

    class _Recording(FileAsker):
        def checkbox(self, key, question, choices):
            asked[key] = list(choices)
            return super().checkbox(key, question, choices)

    run_guided(
        _Recording(answers),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=now,
        out=lambda _line: None,
    )
    return asked["machines"]


def test_a_first_run_offers_to_measure_this_machine_and_marks_it(tmp_path: Path):
    this_machine = _recorded_machines(tmp_path, RUN1, FULL_ANSWERS)[0]

    assert this_machine.value == "this-machine"
    assert this_machine.label == "this machine (measure now)"
    assert this_machine.checked is True


def test_a_machine_that_is_already_measured_here_is_not_marked_again(tmp_path: Path):
    """Enter alone measured all three again in the hand test; measuring again is a decision now."""
    assert _run(tmp_path)[0] == 0
    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}

    this_machine = _recorded_machines(tmp_path, RUN2, answers)[0]

    assert this_machine.checked is False
    assert this_machine.label == "this machine (measure again, last measured 2026-09-23)"


def test_every_profile_in_the_folder_is_listed_grouped_and_not_selectable(tmp_path: Path):
    asked: dict[str, list] = {}

    class _Recording(FileAsker):
        def checkbox(self, key, question, choices):
            asked[key] = list(choices)
            return super().checkbox(key, question, choices)

    _run(tmp_path, {**FULL_ANSWERS, "machines": []})
    hardware = _results(tmp_path) / "state" / "hardware"
    for index, name in enumerate(("alpha", "beta")):
        profile = v2_profile(profile_id=f"{index + 3}" * 16, display_name=name)
        atomic_write_json(hardware / f"{profile.profile_id}.json", profile.model_dump(mode="json"))

    lines: list[str] = []
    run_guided(
        _Recording({key: value for key, value in FULL_ANSWERS.items() if key != "results"} | {"machines": []}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=lines.append,
    )

    machines = asked["machines"]
    groups = [choice for choice in machines if choice.value.startswith("group:")]
    assert len(groups) == 1
    assert "2 machines: alpha, beta" in groups[0].label
    assert groups[0].disabled == "already in this results folder"
    assert [choice.value for choice in machines if choice.disabled is None] == ["this-machine", "import"]
    assert any(choice.value == "enter" and choice.disabled == "stage 2" for choice in machines)


def test_a_schema_one_profile_and_a_broken_one_are_listed_with_their_reason(tmp_path: Path):
    asked: dict[str, list] = {}

    class _Recording(FileAsker):
        def checkbox(self, key, question, choices):
            asked[key] = list(choices)
            return super().checkbox(key, question, choices)

    _run(tmp_path, {**FULL_ANSWERS, "machines": []})
    hardware = _results(tmp_path) / "state" / "hardware"
    hardware.mkdir(parents=True, exist_ok=True)
    legacy = dict(EXAMPLES["HardwareSnapshot"], machine="old")
    (hardware / "old.json").write_text(json.dumps(legacy), encoding="utf-8")
    (hardware / "broken.json").write_text("{", encoding="utf-8")

    run_guided(
        _Recording({key: value for key, value in FULL_ANSWERS.items() if key != "results"} | {"machines": []}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=[].append,
    )

    labels = {choice.value: choice.disabled for choice in asked["machines"]}
    assert labels["legacy:old.json"] == "run `modelroom migrate`"
    assert "does not read" in labels["broken:broken.json"]


def test_a_clone_answer_writes_a_second_profile(tmp_path: Path):
    assert _run(tmp_path)[0] == 0
    first = _profiles(tmp_path)[0]
    # The bound profile is gone: the takeover rule asks whether this is the same machine.
    first.unlink()

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"} | {"clone": "clone"}
    fresh = ["a" * 16, "b" * 16]
    code, lines = _run(tmp_path, answers, now=RUN2, probes=windows_probes(ids=fresh))

    assert code == 0
    written = _profiles(tmp_path)
    assert len(written) == 1
    assert written[0].stem != first.stem
    assert read_pointer(_pointer(tmp_path)).binding_for(_results(tmp_path)) == written[0].stem
    assert not any("nothing measured" in line for line in lines)


def test_the_same_machine_answer_measures_again_under_the_bound_id(tmp_path: Path):
    """The way out of the clone question, decided 2026-09-24: the run measures, it does not stop.

    The results folder was emptied and the pointer file was not -- the case a hand test ran into
    on 2026-09-24, where "the same machine" left the machine unmeasured and every step after it
    empty.
    """
    assert _run(tmp_path)[0] == 0
    bound = read_pointer(_pointer(tmp_path)).binding_for(_results(tmp_path))
    _profiles(tmp_path)[0].unlink()

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"} | {"clone": "same"}
    code, lines = _run(tmp_path, answers, now=RUN2)

    assert code == 0
    assert [path.stem for path in _profiles(tmp_path)] == [bound]
    assert read_pointer(_pointer(tmp_path)).binding_for(_results(tmp_path)) == bound
    assert not any("nothing measured" in line for line in lines)
    assert not any("did not finish" in line for line in lines)


def test_the_same_machine_answer_leads_to_a_ranking_of_this_machine(tmp_path: Path):
    """The whole point of the way out: the steps after it have a measured machine to work with."""
    assert _run(tmp_path)[0] == 0
    _profiles(tmp_path)[0].unlink()

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"} | {"clone": "same"}
    code, lines = _run(tmp_path, answers, now=RUN2)

    assert code == 0
    assert any("packages fit" in line for line in lines)  # step 3 has a machine to count against
    assert "Ranking: " in "\n".join(lines)  # step 5 has a ranking
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    assert payload["machines"][0]["status"] == "ranked"


def test_an_import_adds_the_machine_and_never_changes_the_binding(tmp_path: Path):
    assert _run(tmp_path)[0] == 0
    bound = read_pointer(_pointer(tmp_path)).binding_for(_results(tmp_path))
    export = tmp_path / "server.json"
    profile = v2_profile(profile_id="4" * 16, display_name="server")
    export.write_text(
        json.dumps({"schema_version": 1, "profile": profile.model_dump(mode="json"), "measurements": []}),
        encoding="utf-8",
    )

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    answers |= {"machines": ["import"], "import_file": str(export)}
    code, lines = _run(tmp_path, answers, now=RUN2)

    assert code == 0
    assert (_results(tmp_path) / "state" / "hardware" / f"{'4' * 16}.json").is_file()
    assert read_pointer(_pointer(tmp_path)).binding_for(_results(tmp_path)) == bound
    assert any("profile server" in line for line in lines)


def test_a_profile_already_in_the_folder_gets_a_machine_entry(tmp_path: Path):
    """Otherwise the render, which covers configured machines only, would leave it out -- while
    the machine list calls it "already in this results folder"."""
    hardware = _results(tmp_path) / "state" / "hardware"
    hardware.mkdir(parents=True)
    profile = v2_profile(profile_id="5" * 16, display_name="server 01")
    atomic_write_json(hardware / f"{profile.profile_id}.json", profile.model_dump(mode="json"))

    code, lines = _run(tmp_path, {**FULL_ANSWERS, "write_config": True})

    assert code == 0
    config = load_config(_config_file(tmp_path))
    entry = next(machine for machine in config.machines.values() if machine.profile == profile.profile_id)
    assert entry.writer is False
    assert (entry.reserve_ram_gib, entry.reserve_vram_gib) == (
        config.defaults.reserve_ram_gib,
        config.defaults.reserve_vram_gib,
    )
    assert any("added for the profile" in line for line in lines)
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    assert "server-01" in {block["machine"] for block in payload["machines"]}


def test_a_change_made_during_the_dialog_is_not_written_over(tmp_path: Path):
    """The configuration is read again right before it is changed, not carried through the dialog."""
    assert _run(tmp_path)[0] == 0

    class _MeanwhileImporting(FileAsker):
        """Adds a machine to the file while the search question is on screen."""

        def text(self, key, question, default=""):
            if key == "search":
                text = _config_file(tmp_path).read_text(encoding="utf-8")
                addition = '\n[machines.other]\nreserve_ram_gib = 8.0\nreserve_vram_gib = 1.0\nwriter = false\n'
                _config_file(tmp_path).write_text(text + addition, encoding="utf-8")
            return super().text(key, question, default)

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    code = run_guided(
        _MeanwhileImporting(answers),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=[].append,
    )

    assert code == 0
    assert "other" in load_config(_config_file(tmp_path)).machines


def _adding_a_machine(tmp_path: Path, when_key: str, name: str = "other"):
    """A `FileAsker` that adds `[machines.<name>]` to the file while one question is on screen."""

    def add() -> None:
        entry = f'\n[machines.{name}]\nreserve_ram_gib = 8.0\nreserve_vram_gib = 1.0\nwriter = false\n'
        path = _config_file(tmp_path)
        path.write_text(path.read_text(encoding="utf-8") + entry, encoding="utf-8")

    class _Meanwhile(FileAsker):
        def text(self, key, question, default=""):
            if key == when_key:
                add()
            return super().text(key, question, default)

        def checkbox(self, key, question, choices):
            if key == when_key:
                add()
            return super().checkbox(key, question, choices)

        def select_or_text(self, key, question, choices, text_value, text_question):
            if key == when_key:
                add()
            return super().select_or_text(key, question, choices, text_value, text_question)

    return _Meanwhile


@pytest.mark.parametrize("when_key", ["search", "select", "machines", "context"])
def test_a_change_made_during_any_question_is_not_written_over(tmp_path: Path, when_key: str):
    """Every write reads the file again first, whichever question the change happened under."""
    assert _run(tmp_path)[0] == 0
    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}

    code = run_guided(
        _adding_a_machine(tmp_path, when_key)(answers),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=[].append,
    )

    assert code == 0
    assert "other" in load_config(_config_file(tmp_path)).machines


def _guided_run_for(tmp_path: Path, out: list[str]):
    from modelroom.catalog import load_catalog
    from modelroom.guided import GuidedRun

    return GuidedRun(
        asker=FileAsker({}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        catalog=load_catalog(),
        out=out.append,
    )


def test_a_configuration_that_moved_on_leaves_the_measured_profile_unrecorded(tmp_path: Path):
    """The guard between the measurement and the entry it writes, called directly.

    Through the dialog this cannot happen any more (`_ensure_this_machine` reads the file again
    right before the measurement), so the guard is the second line of defense the review asked
    for: a `[paths]` that moved, or a machine entry that is gone, must be a reported problem and
    not a `KeyError` after the profile was already written and bound.
    """
    from modelroom.guided import _record_profile

    assert _run(tmp_path)[0] == 0
    config = load_config(_config_file(tmp_path))
    out: list[str] = []
    run = _guided_run_for(tmp_path, out)

    moved = config.model_copy(update={"paths": config.paths.model_copy(update={"state": tmp_path / "elsewhere"})})
    assert _record_profile(run, _config_file(tmp_path), moved, "workstation", _results(tmp_path)) is not None
    assert any("[paths] changed while this machine was measured" in line for line in out)

    out.clear()
    _record_profile(run, _config_file(tmp_path), config, "gone-machine", _results(tmp_path))
    assert any("is gone from the configuration" in line for line in out)


def test_a_machine_name_that_cannot_be_freed_leaves_one_profile_unconfigured(tmp_path: Path):
    """`machine_name_for` raises when base, short-id and full-id names are all taken.

    Found by the second-model review: that has to be one profile without an entry and a reported
    problem, not a traceback out of the folder step.
    """
    assert _run(tmp_path, {**FULL_ANSWERS, "machines": []})[0] == 0
    profile = v2_profile(profile_id="7" * 16, display_name="taken")
    hardware = _results(tmp_path) / "state" / "hardware"
    hardware.mkdir(parents=True, exist_ok=True)
    atomic_write_json(hardware / f"{profile.profile_id}.json", profile.model_dump(mode="json"))
    blocked = "".join(
        f'\n[machines.{name}]\nreserve_ram_gib = 8.0\nreserve_vram_gib = 1.0\nwriter = false\n'
        for name in ("taken", f"taken-{profile.profile_id[:8]}", f"taken-{profile.profile_id}")
    )
    path = _config_file(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + blocked, encoding="utf-8")

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    code, lines = _run(tmp_path, answers | {"machines": []}, now=RUN2)

    assert code == 1
    assert any("stays without a machine entry" in line for line in lines)
    assert profile.profile_id not in {
        machine.profile for machine in load_config(path).machines.values()
    }


def test_a_fetch_that_does_not_finish_makes_the_run_exit_1(tmp_path: Path):
    """The document is still written; the exit code says that a step did not finish."""
    assert _run(tmp_path)[0] == 0

    # The same clock again: `fetch` refuses a run that is not newer than the stored snapshot.
    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    code, lines = _run(tmp_path, answers, now=RUN1)

    assert code == 1
    assert any("the fetch ended with exit" in line for line in lines)
    assert (_results(tmp_path) / "docs" / "models.md").is_file()


def test_a_second_run_in_the_remembered_folder_does_not_rewrite_the_pointer_file(tmp_path: Path):
    """A write that changes nothing could still drop a binding another run added meanwhile."""
    assert _run(tmp_path)[0] == 0
    before = _pointer(tmp_path).read_bytes()
    stamp = _pointer(tmp_path).stat().st_mtime_ns

    # Nothing is measured this time, so `hardware` does not rewrite the binding either.
    answers = {key: value for key, value in FULL_ANSWERS.items() if key not in ("results", "machines")}
    assert _run(tmp_path, answers | {"machines": []}, now=RUN2)[0] == 0

    assert _pointer(tmp_path).read_bytes() == before
    assert _pointer(tmp_path).stat().st_mtime_ns == stamp


def test_an_import_of_a_file_that_is_not_there_is_reported_and_the_run_goes_on(tmp_path: Path):
    answers = {**FULL_ANSWERS, "machines": ["import"], "import_file": str(tmp_path / "nowhere.json")}

    code, lines = _run(tmp_path, answers)

    # Exit 1: the ranking is still written, but the import the user asked for did not happen.
    assert code == 1
    assert any("nothing imported" in line for line in lines)


# --- search, choice, context ------------------------------------------------------------------------


def test_the_search_summary_and_the_grouped_reasons_are_printed(tmp_path: Path):
    """One line per reason with its number, never one line per repository (decided 2026-09-24)."""
    _code, lines = _run(tmp_path)

    assert any(line.startswith("4 repositories, 3 resolved, 1 unresolved, 7 requests, budget ") for line in lines)
    grouped = [line for line in lines if "cannot be picked:" in line]
    assert grouped
    assert not any(line.startswith("unresolved ") for line in lines)
    assert sum(int(line.split(" ", 1)[0]) for line in grouped) == 1
    assert any("the repository does not say it packages a base model" in line for line in grouped)


def test_with_the_filter_on_every_account_gets_its_line_and_no_open_list_is_asked(tmp_path: Path):
    """Decided 2026-09-24: one request per account, so a reader can see that theirs was asked."""
    _code, lines = _run(tmp_path)

    assert "Qwen 1" in lines
    assert "unsloth 3" in lines
    assert "bartowski 0" in lines
    assert not any(line.startswith(("most downloaded", "newest")) for line in lines)


def test_with_the_filter_off_the_two_open_lists_are_named_as_well(tmp_path: Path):
    _code, lines = _run(tmp_path, {**FULL_ANSWERS, "filter_owners": False})

    assert "Qwen 1" in lines
    assert "most downloaded 3, 1 of them already listed" in lines
    assert "newest 3" in lines
    assert any(line.startswith("9 repositories, 3 resolved, 6 unresolved, 9 requests, budget ") for line in lines)


def test_without_the_filter_a_derivative_is_visible_and_grayed_out_with_its_reason(tmp_path: Path):
    """"That it is apparent" is the point: a fine-tune is shown, not hidden (decided 2026-09-24)."""
    _code, lines = _run(tmp_path, {**FULL_ANSWERS, "filter_owners": False})

    assert any("a fine-tune or a merge, not a quantization of one base model" in line for line in lines)


def test_the_answer_file_keys_of_step_2_are_unchanged(tmp_path: Path):
    # The three keys stay `search`, `filter_owners` and `select`; switching the filter off only
    # lengthens the list a `select` may pick from.
    code, _lines = _run(tmp_path, {**FULL_ANSWERS, "filter_owners": False})

    assert code == 0
    assert sorted(load_config(_config_file(tmp_path)).families[0].base_models[0].repos) == sorted(
        [QWEN_GGUF, UNSLOTH]
    )


def test_the_chosen_repositories_reach_the_configuration(tmp_path: Path):
    assert _run(tmp_path)[0] == 0

    config = load_config(_config_file(tmp_path))
    base_model = config.families[0].base_models[0]
    assert base_model.hf_repo == "Qwen/Qwen3.5-9B"
    assert sorted(base_model.repos) == sorted([QWEN_GGUF, UNSLOTH])
    assert (base_model.ollama_base, base_model.ollama_tag) == ("qwen3.5", "9b")
    assert config.publishers == ["Qwen"]


def test_the_list_shows_one_line_per_model_with_its_fit_and_its_packagers(tmp_path: Path):
    """The list a person picks from, printed by the run itself (decided 2026-09-24)."""
    lines: list[str] = []
    asker = _WatchingAsker(FULL_ANSWERS, lines)
    code = run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    assert code == 0
    listed = asker.choices["select"]
    # Two models, then the repositories an answer file of the older shape may name instead.
    assert [choice.value for choice in listed[:2]] == ["Qwen/Qwen3.5-9B", DEEPSEEK_BASE]
    assert [choice.value for choice in listed[2:]] == [QWEN_GGUF, UNSLOTH, DEEPSEEK]
    assert all(choice.disabled is None for choice in listed)
    assert listed[0].label.startswith("Qwen3.5-9B")
    assert "good" in listed[0].label
    assert "Qwen, unsloth" in listed[0].label
    assert all(len(choice.label) <= 100 for choice in listed[:2])


def test_the_count_line_and_the_hint_line_stand_around_the_list(tmp_path: Path):
    _code, lines = _run(tmp_path)

    assert any(line.startswith("2 models can be picked; 1 repository cannot:") for line in lines)
    assert any(line.startswith("fit at 8k context") for line in lines)
    assert any(line.startswith("Space marks a model") for line in lines)


def test_a_model_is_fetched_from_the_publisher_and_from_one_listed_packager(tmp_path: Path):
    """Two repositories per model, because every one costs the fetch about ten requests."""
    answers = {**FULL_ANSWERS, "select": ["Qwen/Qwen3.5-9B"]}

    code, _lines = _run(tmp_path, answers)

    assert code == 0
    base_model = load_config(_config_file(tmp_path)).families[0].base_models[0]
    assert sorted(base_model.repos) == sorted([QWEN_GGUF, UNSLOTH])


def test_choosing_nothing_leaves_the_configuration_as_it_is(tmp_path: Path):
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "select": []})

    assert code == 1  # nothing to render: no family, so no snapshot
    assert load_config(_config_file(tmp_path)).families == []
    assert any("nothing chosen" in line for line in lines)
    assert any("nothing to fetch" in line for line in lines)


def test_an_entered_context_reaches_the_document(tmp_path: Path):
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "context": "4096"})

    assert code == 0
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    assert payload["scenario"] == {
        "context_requested": 4096,
        "context_origin": "entered",
        "kv_type": "f16",
        "kv_type_assumed": True,
        "requests": 1,
    }
    # The terminal view says the scenario short, the two files carry it in full.
    printed = "\n".join(lines)
    assert "context XS 4k" in printed and "1 request" in printed and "KV cache f16 (assumed)" in printed


def test_the_chosen_context_is_kept_in_the_configuration(tmp_path: Path):
    assert _run(tmp_path, {**FULL_ANSWERS, "context": "4096"})[0] == 0

    assert tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))["guided"]["context"] == 4096
    assert load_config(_config_file(tmp_path)).guided.context == 4096


def test_the_default_context_is_kept_too_because_choosing_it_is_a_decision(tmp_path: Path):
    """A stored 8192 is the answer the user gave, not the value the question started with."""
    assert _run(tmp_path, FULL_ANSWERS)[0] == 0

    assert load_config(_config_file(tmp_path)).guided.context == 8192


def test_the_second_run_offers_the_kept_context_as_the_default(tmp_path: Path):
    """The pointer of the scale starts on the level this folder kept, not on the default level."""
    pointed_at: dict[str, str] = {}

    class _Recording(FileAsker):
        def select_or_text(self, key, question, choices, text_value, text_question):
            pointed_at[key] = next(choice.value for choice in choices if choice.checked)
            return super().select_or_text(key, question, choices, text_value, text_question)

    assert _run(tmp_path, {**FULL_ANSWERS, "context": "4096"})[0] == 0

    run_guided(
        _Recording({**FULL_ANSWERS, "context": "4096"}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=lambda _line: None,
    )

    assert pointed_at["context"] == "XS"


def test_the_same_context_again_is_not_written_a_second_time(tmp_path: Path):
    """The step says so when it writes, so a run that writes nothing is visible in its output."""
    code, first = _run(tmp_path, {**FULL_ANSWERS, "context": "4096"})
    assert code == 0
    assert any(line.startswith("kept context 4096 in ") for line in first)

    code, second = _run(tmp_path, {**FULL_ANSWERS, "context": "4096"}, now=RUN2)

    assert code == 0
    assert not any(line.startswith("kept context ") for line in second)
    assert load_config(_config_file(tmp_path)).guided.context == 4096


def _store_context(config_file: Path, context: int) -> None:
    """Write `[guided].context` straight into the file -- what another process would leave behind."""
    lines = [
        f"context = {context}" if line.startswith("context = ") else line
        for line in config_file.read_text(encoding="utf-8").splitlines()
    ]
    config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_answer_is_compared_with_the_file_not_with_this_runs_own_copy(tmp_path: Path):
    """Another process may write the file while the question stands; the file on disk decides.

    Without this the run would skip its write because its own copy already said 4096, and leave
    the other process' 8192 in the file -- the ranking of this run and the one a later
    `modelroom render` computes would disagree again, which is the whole point of keeping it.
    """

    class _WritesInBetween(FileAsker):
        def select_or_text(self, key, question, choices, text_value, text_question):
            if key == "context":
                _store_context(_config_file(tmp_path), 8192)
            return super().select_or_text(key, question, choices, text_value, text_question)

    assert _run(tmp_path, {**FULL_ANSWERS, "context": "4096"})[0] == 0

    run_guided(
        _WritesInBetween({**FULL_ANSWERS, "context": "4096"}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=lambda _line: None,
    )

    assert load_config(_config_file(tmp_path)).guided.context == 4096


def test_a_later_render_of_its_own_uses_the_kept_context(tmp_path: Path):
    """The whole point: `modelroom render` shows the ranking the guided run showed."""
    # The daemon has to report the context the ranking asks for, or the measurement is stored as
    # not comparable and would count in no ranking at all (`ps_answer`'s `context_length`).
    daemon = loadtest_daemon(observations=[ps_answer(context_length=4096)] * 4)
    assert _run(tmp_path, LOAD_TEST_ANSWERS | {"context": "4096"}, daemon=daemon)[0] == 0
    measured = json.loads(_measurement_files(tmp_path)[0].read_text(encoding="utf-8"))
    assert (measured["scenario"]["context_requested"], measured["comparable"]) == (4096, True)

    assert main(["render", "--config", str(_config_file(tmp_path))], now=RUN2) == 0

    text = (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")
    assert "Scenario: context 4096 (entered)" in text
    assert _ranked_deepseek(tmp_path)["measurement_group"] == 0


def test_a_configuration_without_a_kept_context_still_renders_with_8192(tmp_path: Path):
    assert _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())[0] == 0
    config_file = _config_file(tmp_path)
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace("context = 8192\n", ""), encoding="utf-8"
    )

    assert main(["render", "--config", str(config_file)], now=RUN2) == 0

    text = (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")
    assert "Scenario: context 8192 (default)" in text


def test_a_kept_context_of_8192_is_entered_when_render_runs_alone(tmp_path: Path):
    """A kept `[guided].context` is what a guided run chose, 8192 included -- `default` is only
    what `render` assumes for a folder that kept none. The dialog and a later `render --config`
    of the same folder must name the same origin."""
    assert _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())[0] == 0
    assert "context = 8192\n" in _config_file(tmp_path).read_text(encoding="utf-8")

    assert main(["render", "--config", str(_config_file(tmp_path))], now=RUN2) == 0

    text = (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")
    assert "Scenario: context 8192 (entered)" in text


def test_a_context_that_is_not_a_number_ends_the_run(tmp_path: Path):
    with pytest.raises(GuidedError, match="whole number"):
        _run(tmp_path, {**FULL_ANSWERS, "context": "lots"})


@pytest.mark.parametrize("answer", ["0", "-1", str(2**31)])
def test_a_context_the_scenario_cannot_take_ends_the_run_without_a_traceback(tmp_path: Path, answer: str):
    """`Scenario.context_requested` is bounded; an answer outside it is an input error, not a crash."""
    with pytest.raises(GuidedError):
        _run(tmp_path, {**FULL_ANSWERS, "context": answer})


# --- the last steps: fetch, the load test, render -----------------------------------------------------


def test_the_run_ends_with_both_views_and_a_word_on_the_load_test(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    assert any(line.startswith("no load test: ") for line in lines)
    assert (_results(tmp_path) / "docs" / "models.md").is_file()
    assert (_results(tmp_path) / "docs" / "models.json").is_file()
    printed = "\n".join(lines)
    # The rule and the snapshot time are in the Markdown file; the screen shows the table.
    assert "Ranking rule: fit class" in (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")
    assert "Ranking: workstation" in printed
    assert "Model" in printed and "Package" in printed


def _ranked_deepseek(tmp_path: Path) -> dict:
    """The row of the measured example package in the machine block the render wrote."""
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    rows = payload["machines"][0]["ranked"] + payload["machines"][0]["too_tight"]
    return next(row for row in rows if row["package_identity"][1] == DEEPSEEK)


def _measurement_files(tmp_path: Path) -> list[Path]:
    folder = _results(tmp_path) / "state" / "measurements"
    return sorted(folder.rglob("*.json")) if folder.is_dir() else []


def test_the_load_test_measures_the_picked_model_and_the_ranking_shows_the_speed(tmp_path: Path):
    code, lines = _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())

    assert code == 0
    files = _measurement_files(tmp_path)
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert (record["validity"], record["comparable"]) == ("valid", True)
    assert record["package"]["hf_repo"] == DEEPSEEK
    assert record["ollama_name"] == DEEPSEEK_OLLAMA_NAME
    measured = next(line for line in lines if line.startswith(f"{DEEPSEEK_OLLAMA_NAME}: measured "))
    assert "tok/s" in measured and measured.endswith("valid, comparable")
    row = _ranked_deepseek(tmp_path)
    assert row["measurement_group"] == 0
    assert row["speed_tps"] == pytest.approx(record["tps_mean"])


def test_a_no_to_the_load_test_measures_nothing_and_asks_for_no_package(tmp_path: Path):
    answers = {**LOAD_TEST_ANSWERS, "load_test": False}
    answers.pop("load_test_packages")

    code, lines = _run(tmp_path, answers, daemon=loadtest_daemon())

    assert code == 0
    assert _measurement_files(tmp_path) == []
    assert not any(line.startswith(f"{DEEPSEEK_OLLAMA_NAME}:") for line in lines)


def test_nothing_installed_that_is_ranked_is_one_line_and_no_question(tmp_path: Path):
    """The answer file has no load-test answer at all, and the run still ends with exit 0."""
    code, lines = _run(tmp_path, FULL_ANSWERS, daemon=loadtest_daemon())

    assert code == 0
    assert NO_CANDIDATE_LINE in lines
    assert _measurement_files(tmp_path) == []


def test_a_cloud_model_is_named_as_one_and_never_measured(tmp_path: Path):
    _code, lines = _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())

    assert any("glm-5.3-flash:cloud" in line and "not measured" in line for line in lines)


def test_the_selection_list_shows_every_candidate_unchecked_with_its_four_facts(tmp_path: Path):
    asked: dict[str, list] = {}

    class _Recording(FileAsker):
        def checkbox(self, key, question, choices):
            asked[key] = list(choices)
            return super().checkbox(key, question, choices)

    lines: list[str] = []
    run_guided(
        _Recording(LOAD_TEST_ANSWERS),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=loadtest_daemon(),
        now=RUN1,
        out=lines.append,
    )

    choices = asked["load_test_packages"]
    assert [choice.value for choice in choices] == [DEEPSEEK_OLLAMA_NAME]
    # Nothing is marked: Enter alone must not measure what the user never picked (decided 2026-09-24).
    assert choices[0].checked is False
    assert choices[0].label.split() == [DEEPSEEK_OLLAMA_NAME, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Q4_K_M", "4.68", "GiB"]


def test_a_daemon_that_goes_away_after_the_yes_is_a_step_that_did_not_finish(tmp_path: Path):
    from modelroom.daemon import DaemonError

    daemon = loadtest_daemon(generates=[DaemonError("the Ollama daemon did not answer (refused)")])

    code, lines = _run(tmp_path, LOAD_TEST_ANSWERS, daemon=daemon)

    assert code == 1
    assert _measurement_files(tmp_path) == []
    assert any("nothing measured for" in line for line in lines)
    assert (_results(tmp_path) / "docs" / "models.md").is_file()


def test_a_digest_that_changes_mid_run_is_stored_as_not_comparable_and_ranks_in_group_one(tmp_path: Path):
    observations = [ps_answer(), ps_answer(), ps_answer(digest="d" * 64), ps_answer()]
    # This one model only: a measurement that does not count ranks behind every package that
    # does, and the ranking of the document shows the first ten of them.
    answers = {**LOAD_TEST_ANSWERS, "select": [DEEPSEEK]}

    code, lines = _run(tmp_path, answers, daemon=loadtest_daemon(observations=observations))

    assert code == 0
    record = json.loads(_measurement_files(tmp_path)[0].read_text(encoding="utf-8"))
    assert record["comparable"] is False
    assert "digest" in record["comparable_reason"]
    assert any("not comparable" in line for line in lines)
    row = _ranked_deepseek(tmp_path)
    assert (row["measurement_group"], row["speed_tps"]) == (1, None)


def test_an_answer_file_that_does_not_mention_the_load_test_measures_nothing(tmp_path: Path):
    """The one optional answer: a file written for another machine still runs to the end."""
    answers = {**LOAD_TEST_ANSWERS}
    answers.pop("load_test")

    code, lines = _run(tmp_path, answers, daemon=loadtest_daemon())

    assert code == 0
    assert _measurement_files(tmp_path) == []
    assert not any(line.startswith(f"{DEEPSEEK_OLLAMA_NAME}:") for line in lines)


def test_a_yes_that_names_no_package_is_a_missing_answer(tmp_path: Path):
    answers = {**LOAD_TEST_ANSWERS}
    answers.pop("load_test_packages")

    with pytest.raises(AnswerMissingError, match="load_test_packages"):
        _run(tmp_path, answers, daemon=loadtest_daemon())


def test_a_yes_against_a_daemon_that_is_not_there_is_a_step_that_did_not_finish(tmp_path: Path):
    answers = {**LOAD_TEST_ANSWERS}
    answers.pop("load_test_packages")

    code, lines = _run(tmp_path, answers, daemon=offline_daemon())

    assert code == 1
    assert _measurement_files(tmp_path) == []
    assert any(line.startswith("nothing measured: ") for line in lines)
    assert (_results(tmp_path) / "docs" / "models.md").is_file()


def test_a_configuration_that_names_a_profile_is_not_proof_that_it_is_this_machine(tmp_path: Path):
    """Only the pointer file's binding decides which profile this machine measures into."""
    assert _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())[0] == 0
    first = _measurement_files(tmp_path)
    assert len(first) == 1
    # What a shared results folder looks like on a machine that never measured itself: the
    # configuration still names the profile, the local binding does not.
    pointer = read_pointer(_pointer(tmp_path))
    write_pointer(_pointer(tmp_path), GuidedPointer(schema_version=1, current=pointer.current))
    answers = {key: value for key, value in LOAD_TEST_ANSWERS.items() if key != "results"}
    answers["machines"] = []

    code, lines = _run(tmp_path, answers, now=RUN2, daemon=loadtest_daemon())

    assert code == 0
    assert _measurement_files(tmp_path) == first
    assert any("not bound to a profile" in line for line in lines)


def test_a_bound_profile_the_folder_no_longer_holds_measures_nothing(tmp_path: Path):
    assert _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())[0] == 0
    for path in _profiles(tmp_path):
        path.unlink()
    answers = {key: value for key, value in LOAD_TEST_ANSWERS.items() if key != "results"}
    answers["machines"] = []

    code, lines = _run(tmp_path, answers, now=RUN2, daemon=loadtest_daemon())

    assert code == 0
    assert any("is missing in the results folder" in line for line in lines)
    assert len(_measurement_files(tmp_path)) == 1


def test_a_bound_profile_of_another_machine_measures_nothing(tmp_path: Path):
    """The takeover rule's own fingerprint check: a declined `same machine` must not measure here."""
    assert _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())[0] == 0
    bound = _profiles(tmp_path)[0]
    profile = json.loads(bound.read_text(encoding="utf-8"))
    profile["os_fingerprint"] = "a" * 16
    bound.write_text(json.dumps(profile), encoding="utf-8")
    answers = {key: value for key, value in LOAD_TEST_ANSWERS.items() if key != "results"}
    answers["machines"] = []

    code, lines = _run(tmp_path, answers, now=RUN2, daemon=loadtest_daemon())

    assert code == 0
    assert any("another os_fingerprint" in line for line in lines)
    assert len(_measurement_files(tmp_path)) == 1


def test_a_no_against_a_daemon_that_is_not_there_is_one_line_and_exit_zero(tmp_path: Path):
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "load_test": False}, daemon=offline_daemon())

    assert code == 0
    assert any(line.startswith("no load test: ") for line in lines)


def test_a_second_run_keeps_the_measurement_and_writes_no_second_file(tmp_path: Path):
    assert _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())[0] == 0
    first = _measurement_files(tmp_path)
    answers = {key: value for key, value in LOAD_TEST_ANSWERS.items() if key != "results"}
    answers["load_test"] = False
    answers.pop("load_test_packages")

    code, _lines = _run(tmp_path, answers, now=RUN2, daemon=loadtest_daemon())

    assert code == 0
    assert _measurement_files(tmp_path) == first
    assert _ranked_deepseek(tmp_path)["measurement_group"] == 0


def test_qwen35_is_judged_from_the_size_of_its_packages(tmp_path: Path):
    """Its architecture is a hybrid one fit v1 cannot read, so the size answers (2026-09-24)."""
    assert _run(tmp_path)[0] == 0

    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    block = payload["machines"][0]
    assert block["status"] == "ranked"
    assert block["ranked"], "a package of a hybrid architecture is judged from its size now"
    assert {entry["fit"]["basis"] for entry in block["ranked"]} == {"size"}
    assert not any(entry["reason"] == "architecture not covered by v1" for entry in block["not_covered"])


# --- the entry point: the TTY rule and --answers -------------------------------------------------------


def test_a_missing_answer_is_exit_2_and_names_the_question(tmp_path: Path, capsys):
    answers = tmp_path / "answers.toml"
    answers.write_text('schema_version = 1\nresults = "here"\n', encoding="utf-8")

    code = main(
        ["--answers", str(answers)],
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        pointer_path=_pointer(tmp_path),
        now=RUN1,
        here=_results(tmp_path),
        out=[].append,
    )

    assert code == 2
    assert "no answer for 'machines'" in capsys.readouterr().err


def test_an_answer_file_that_does_not_read_is_exit_2(tmp_path: Path, capsys):
    answers = tmp_path / "answers.toml"
    answers.write_text("results = \n", encoding="utf-8")

    assert main(["--answers", str(answers)], here=_results(tmp_path), pointer_path=_pointer(tmp_path)) == 2
    assert "answers.toml" in capsys.readouterr().err


def test_an_answer_file_of_an_unsupported_schema_is_exit_3(tmp_path: Path, capsys):
    answers = tmp_path / "answers.toml"
    answers.write_text("schema_version = 9\n", encoding="utf-8")

    assert main(["--answers", str(answers)], here=_results(tmp_path), pointer_path=_pointer(tmp_path)) == 3
    assert "schema_version" in capsys.readouterr().err


def test_answers_with_a_subcommand_is_refused(tmp_path: Path, capsys):
    code = main(["--answers", str(tmp_path / "a.toml"), "render", "--config", str(_config_file(tmp_path))])

    assert code == 2
    assert "--answers belongs to the guided mode" in capsys.readouterr().err


def test_an_end_of_input_is_exit_2(tmp_path: Path, capsys):
    class _Ending(FileAsker):
        def select(self, key, question, choices):
            raise Canceled("the dialog ended without an answer")

    code = main(
        [],
        asker=_Ending({}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        now=RUN1,
        out=[].append,
    )

    assert code == 2
    assert "ended without an answer" in capsys.readouterr().err


def test_control_c_is_exit_130_and_leaves_no_configuration(tmp_path: Path, capsys):
    class _Interrupting(FileAsker):
        def select(self, key, question, choices):
            raise KeyboardInterrupt

    code = main(
        [],
        asker=_Interrupting({}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        probes=windows_probes(),
        now=RUN1,
        out=[].append,
    )

    assert code == 130
    assert "nothing was left half written" in capsys.readouterr().err
    assert not _config_file(tmp_path).exists()


def test_an_unsupported_configuration_schema_keeps_its_own_exit_code(tmp_path: Path, capsys):
    """Exit 3, not 2: the guided mode does not flatten a stored file's failure into a dialog one."""
    _config_file(tmp_path).write_text("schema_version = 9\n", encoding="utf-8")
    answers = tmp_path / "answers.toml"
    answers.write_text('schema_version = 1\nresults = "here"\n', encoding="utf-8")

    code = main(
        ["--answers", str(answers)],
        pointer_path=_pointer(tmp_path),
        probes=windows_probes(),
        here=_results(tmp_path),
        now=RUN1,
        out=[].append,
    )

    assert code == 3
    assert "schema_version" in capsys.readouterr().err


def test_a_held_lock_keeps_its_own_exit_code(tmp_path: Path, capsys):
    """Exit 1, not 2: another process holding the state lock is the documented `1`."""
    answers = tmp_path / "answers.toml"
    # `acquire_lock` creates the state folder, so the folder question comes up as well.
    answers.write_text('schema_version = 1\nresults = "here"\nwrite_config = true\n', encoding="utf-8")
    lock = _results(tmp_path) / "state" / "modelroom.lock"
    handle = acquire_lock(lock, "test", RUN1)
    try:
        code = main(
            ["--answers", str(answers)],
            pointer_path=_pointer(tmp_path),
            probes=windows_probes(),
            here=_results(tmp_path),
            now=RUN1,
            out=[].append,
        )
    finally:
        release_lock(handle)

    assert code == 1
    assert "modelroom.lock" in capsys.readouterr().err


def test_without_a_terminal_the_help_goes_to_stderr_and_the_exit_is_2(tmp_path: Path):
    """A real child process, stdin from the null device and stdout in a file: no terminal at all.

    Nothing is asked and nothing is written -- not even in the user's home folder, which the
    child gets a fresh copy of inside `tmp_path` so a failure of this rule would be visible
    rather than silently landing in the real one.
    """
    home = tmp_path / "child-home"
    home.mkdir()
    out_path = tmp_path / "stdout.txt"
    program = "import sys; from modelroom.cli import main; sys.exit(main([]))"
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}
    with out_path.open("w", encoding="utf-8") as out, open(os.devnull) as devnull:
        finished = subprocess.run(
            [sys.executable, "-c", program],
            stdin=devnull,
            stdout=out,
            stderr=subprocess.PIPE,
            cwd=tmp_path,
            env=env,
            text=True,
            encoding="utf-8",
        )

    assert finished.returncode == 2
    assert "usage: modelroom" in finished.stderr
    assert "needs an interactive terminal" in finished.stderr
    assert out_path.read_text(encoding="utf-8") == ""
    assert list(home.rglob("*")) == []


def test_a_missing_answer_deep_in_the_dialog_leaves_the_configuration_valid(tmp_path: Path):
    """A run that stops at a question keeps every file it already wrote readable."""
    with pytest.raises(AnswerMissingError, match="search"):
        _run(tmp_path, {"results": "here", "machines": []})

    assert load_config(_config_file(tmp_path)).schema_version == 2
