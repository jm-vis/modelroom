"""End-to-end tests for the guided mode: `modelroom` with no subcommand.

Every run goes through injected layers -- a `FixtureTransport` for the search and the fetch, a
`Probes` whose sources are all fixtures, a fixed clock, a pointer file inside `tmp_path` and a
`FileAsker` for the dialog. No process is spawned, no network is touched, and nothing outside
`tmp_path` is written; in particular the real pointer file in the user's home folder is never
read or written.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import re
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
from modelroom.fit import OLLAMA_REQUEST_HINT_TEXT
from modelroom.examples import EXAMPLES
from modelroom.guided import GuidedError, run_guided
from modelroom.guided_loadtest import NO_CANDIDATE_LINE
from modelroom.http import Response
from modelroom.intro import STEP_NAMES
from modelroom.profile import HardwareProfile
from modelroom.state import acquire_lock, atomic_write_json, release_lock

from fixture_support import (
    DEEPSEEK_OLLAMA_NAME,
    build_transport,
    guided_transport_mapping,
    json_response,
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
        self.instructions: dict[str, str | None] = {}
        self._lines = lines

    def checkbox(self, key: str, question: str, choices, instruction=None) -> list[str]:
        self.choices[key] = list(choices)
        self.instructions[key] = instruction
        return super().checkbox(key, question, choices, instruction)


FULL_ANSWERS = {
    "results": "here",
    "machines": ["this-machine"],
    "search": "qwen",
    "filter_owners": True,
    "select": [UNSLOTH, QWEN_GGUF],
    "users": 1,
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
        transport=kwargs.pop("transport", None) or build_transport(guided_transport_mapping()),
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


def _heads(lines: list[str]) -> list[str]:
    """Every step head of a run: a rule, the step, its name and a rule to 72 characters."""
    return [line.strip() for line in lines if line.strip().startswith(("── Step ", "-- Step "))]


def _summaries(lines: list[str]) -> list[str]:
    """Every balance line of a run: a check mark or a dash, the step's name and its balance."""
    marks = ("✓ ", "ok ", "– ", "- ")
    return [line.strip() for line in lines if line.strip().startswith(marks) and not line.strip().startswith("-- ")]


def test_the_run_begins_with_the_start_screen_and_walks_five_numbered_steps(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    assert any("ModelRoom" in line for line in lines[:4])
    heads = [re.match(r"\S+ Step (\d) of 5  (\w+) ", head) for head in _heads(lines)]
    assert [head.group(2) for head in heads] == list(STEP_NAMES)
    assert [head.group(1) for head in heads] == [str(number) for number in range(1, 6)]
    assert all(len(head.string) == 72 for head in heads)


def test_every_step_leaves_one_line_behind_with_what_it_did(tmp_path: Path):
    _code, lines = _run(tmp_path)

    done = _summaries(lines)
    assert [line.split()[1] for line in done] == list(STEP_NAMES)
    # One model: this answer file names two repositories of the same base model, and the
    # configuration and the card hold one (second-model round, 2026-09-24).
    assert any(re.search(r"1 model, \d+ packages from \d+ repositor", line) for line in done)
    assert any(line.endswith("S 8k") for line in done)
    assert any(line.endswith("models.md") for line in done)


def test_a_step_that_did_nothing_carries_a_dash_instead_of_a_check_mark(tmp_path: Path):
    """A step that measured nothing did what it was told; a check mark cannot say that."""
    _code, lines = _run(tmp_path)

    measurement = next(line for line in _summaries(lines) if "Measurement" in line)
    assert measurement.startswith(("– ", "- "))
    assert measurement.endswith("none in this run")


def test_every_answer_is_written_back_in_words(tmp_path: Path):
    """The run writes the answer line, not the library: `this machine`, not `[this machine (...)]`."""
    _code, lines = _run(tmp_path)

    answers = [line.strip() for line in lines if line.strip().startswith("? ")]
    assert "? Where should results live?  this folder" in answers
    assert "? Which machines should the result cover?  this machine" in answers
    assert "? What are you looking for?  qwen" in answers
    assert "? Show only repositories of a publisher or a listed packager?  Yes" in answers
    # This answer file names repositories, as a file always could: an answer of the user appears
    # in the words of the user. A marked model appears as the name the list showed.
    assert any(answer.endswith(f"  {UNSLOTH}, {QWEN_GGUF}") for answer in answers)
    assert any(answer.endswith("S  8k  6,000 words  a long conversation") for answer in answers)
    assert not any("done (" in line for line in lines)


def test_a_marked_model_is_written_back_as_the_name_the_list_showed(tmp_path: Path):
    _code, lines = _run(tmp_path, {**FULL_ANSWERS, "select": ["Qwen/Qwen3.5-9B"]})

    assert any(line.strip().endswith("  Qwen3.5-9B") for line in lines)


def test_the_balance_of_step_two_counts_models_and_not_answers(tmp_path: Path):
    """Two repositories of one base model are one model, and that is what the card holds too."""
    two_models = {**FULL_ANSWERS, "select": ["Qwen/Qwen3.5-9B", DEEPSEEK_BASE]}

    _code, lines = _run(tmp_path, two_models)

    assert any(line.strip().startswith("✓ Packages        2 models,") for line in lines)


def test_a_search_log_that_cannot_be_written_is_a_fault_and_no_traceback(tmp_path: Path):
    """The search went through; a folder that cannot hold its log must not end the run (2026-09-24)."""
    state = _results(tmp_path) / "state"
    state.mkdir(parents=True)
    (state / "search.json").mkdir()  # a directory where the file belongs

    code, lines = _run(tmp_path, {**FULL_ANSWERS, "write_config": True})

    assert code == 1  # a step reported it, and the document is still written
    assert any("the log of this search was not written" in line for line in lines)
    assert (_results(tmp_path) / "docs" / "models.md").is_file()


def test_the_install_line_follows_the_machine_this_run_is_on(tmp_path: Path):
    """Several writers are allowed; the local name of a package is about this machine only."""
    _code, lines = _run(tmp_path)

    install = next(line for line in lines if line.strip().startswith("install   #1"))
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    ranked = next(block for block in payload["machines"] if block["machine"] == "workstation")["ranked"]
    assert ranked[0]["quantization"] in install


def test_an_import_that_brought_no_profile_says_what_it_did_instead(tmp_path: Path):
    """The stored profile was the newer one: then the import's own report is the only statement."""
    assert _run(tmp_path)[0] == 0
    profile = HardwareProfile.model_validate(json.loads(_profiles(tmp_path)[0].read_text(encoding="utf-8")))
    older = profile.model_copy(update={"recorded_at": RUN1.replace(hour=1)})
    export = tmp_path / "again.json"
    export.write_text(
        json.dumps({"schema_version": 1, "profile": older.model_dump(mode="json"), "measurements": []}),
        encoding="utf-8",
    )

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    answers |= {"machines": ["import"], "import_file": str(export)}
    _code, lines = _run(tmp_path, answers, now=RUN2)

    assert not any(line.strip().startswith("imported ") for line in lines)
    assert any("kept, the local profile is newer" in line for line in lines)


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
        def select_or_text(self, key, question, choices, text_value, text_question, instruction=None):
            order.append(f"asked {key}")
            return super().select_or_text(key, question, choices, text_value, text_question, instruction)

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

    head = next(index for index, line in enumerate(order) if "Step 2 of 5  Packages" in line)
    assert head < order.index("asked context")
    chosen = next(index for index, line in enumerate(order) if "of them models you can pick from" in line)
    assert chosen < order.index("asked context")


def test_the_scale_is_asked_with_the_packages_of_this_runs_own_fetch(tmp_path: Path):
    """The last column is not a promise: it is a count over what the snapshot really holds."""
    seen: dict[str, list] = {}

    class _Watching(FileAsker):
        def select_or_text(self, key, question, choices, text_value, text_question, instruction=None):
            seen[key] = list(choices)
            return super().select_or_text(key, question, choices, text_value, text_question, instruction)

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
    assert any("23 packages" in label for label in labels)


def _card(lines: list[str]) -> dict[str, str]:
    """The rows of the card of step 5, by their label: the block between the head and the table.

    Read as a block and not by label alone -- the start screen has a `folder` row of its own, and
    the notes under the result table carry the card's own labels (`speed`, `shown`, `memory`).
    """
    head = max(index for index, line in enumerate(lines) if "Step 5 of 5" in line)
    tail = lines[head + 1 :]
    start = next(index for index, line in enumerate(tail) if line.strip().startswith("folder "))
    rows = itertools.takewhile(lambda line: line.strip(), tail[start:])
    return {line.split()[0]: line for line in (row.strip() for row in rows)}


def test_the_run_closes_with_the_card_and_the_relative_path_of_the_document(tmp_path: Path):
    """The card is the report a reader looks at once the run is over (decided 2026-09-24)."""
    _code, lines = _run(tmp_path)

    card = _card(lines)
    assert str(_results(tmp_path)) in card["folder"]
    assert "GB graphics" in card["machine"] and "GB memory" in card["machine"]
    assert "Qwen3.5-9B" in card["models"] and "packages from" in card["models"]
    assert card["context"] == "context   8k (entered), KV cache f16 (assumed), 1 request (from 1 user)"
    assert card["speed"].startswith("speed     not measured")
    assert "models.md" in card["result"] and "packages ranked" in card["result"]
    assert not any(line.startswith("Written to ") for line in lines)


# --- first start: the folder question, the configuration, the pointer file ----------------------


def test_the_first_start_writes_a_configuration_with_this_device_as_the_writer(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    raw = tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 3
    assert raw["machines"]["workstation"]["writer"] is True
    assert Path(raw["guided"]["results"]) == _results(tmp_path)
    # The line that named the writer is gone from the screen: the configuration says it, and the
    # balance of step 1 says the folder and how many machines the result covers.
    assert not any("as the writer of this results folder" in line for line in lines)
    assert any(line.strip().endswith(f"{_results(tmp_path)}, 1 machine") for line in lines)


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
    assert tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))["schema_version"] == 3
    assert any("modelroom.toml" in line for line in lines)
    # The migration lines are for a maintainer; the sentence in front of them is for a user. Both
    # are notes of step 1 now, indented under the question they belong to.
    assert (
        "this folder holds a configuration from an earlier version; it was updated, "
        "backup kept: modelroom.toml.v1.bak"
    ) in [line.strip() for line in lines]


# --- the machine list ------------------------------------------------------------------------------


def test_the_measured_machine_is_written_into_the_configuration(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    profile = HardwareProfile.model_validate(json.loads(_profiles(tmp_path)[0].read_text(encoding="utf-8")))
    assert profile.gpu_state == "measured"
    assert profile.llmfit_crosscheck.ram_physical.status == "confirmed"
    raw = tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))
    assert raw["machines"]["workstation"]["profile"] == profile.profile_id
    # One note instead of the readings and the profile id: what was measured, and who confirmed it.
    assert not any(profile.profile_id in line for line in lines)
    note = next(line.strip() for line in lines if line.strip().startswith("measured"))
    assert note == "measured: one graphics card, 12 GB, 128 GB memory, confirmed by llmfit"


def test_no_machine_chosen_measures_nothing(tmp_path: Path):
    code, _lines = _run(tmp_path, {**FULL_ANSWERS, "machines": []})

    assert code == 0
    assert _profiles(tmp_path) == []


def _recorded_machines(tmp_path: Path, now, answers: dict) -> list:
    """The machine list of one run, as the dialog was offered it."""
    asked: dict[str, list] = {}

    class _Recording(FileAsker):
        def checkbox(self, key, question, choices, instruction=None):
            asked[key] = list(choices)
            return super().checkbox(key, question, choices, instruction)

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
        def checkbox(self, key, question, choices, instruction=None):
            asked[key] = list(choices)
            return super().checkbox(key, question, choices, instruction)

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
    assert [choice.value for choice in machines if choice.disabled is None] == ["this-machine", "import", "enter"]


def test_a_schema_one_profile_and_a_broken_one_are_listed_with_their_reason(tmp_path: Path):
    asked: dict[str, list] = {}

    class _Recording(FileAsker):
        def checkbox(self, key, question, choices, instruction=None):
            asked[key] = list(choices)
            return super().checkbox(key, question, choices, instruction)

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
    assert not any("nothing measured:" in line for line in lines)
    # A clone is another machine, so its note is the note of a first measurement.
    assert any(line.strip().startswith("measured: ") for line in lines)


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
    assert not any("nothing measured:" in line for line in lines)
    assert not any("did not finish" in line for line in lines)
    assert any(line.strip().startswith("measured again:") for line in lines)


def test_the_same_machine_answer_leads_to_a_ranking_of_this_machine(tmp_path: Path):
    """The whole point of the way out: the steps after it have a measured machine to work with."""
    assert _run(tmp_path)[0] == 0
    _profiles(tmp_path)[0].unlink()

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"} | {"clone": "same"}
    code, lines = _run(tmp_path, answers, now=RUN2)

    assert code == 0
    # Step 3 has a machine to count against, and step 5 a ranking of it.
    assert not any("no measured machine" in line for line in lines)
    assert any("· context L" in line or "· context S" in line for line in lines)
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
    assert any(line.strip().startswith("imported server: ") for line in lines)


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

        def checkbox(self, key, question, choices, instruction=None):
            if key == when_key:
                add()
            return super().checkbox(key, question, choices, instruction)

        def select_or_text(self, key, question, choices, text_value, text_question, instruction=None):
            if key == when_key:
                add()
            return super().select_or_text(key, question, choices, text_value, text_question, instruction)

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


# What a second run leaves in a new results folder: settings this run must not replace.
FOREIGN_CONFIGURATION = (
    "schema_version = 2\n"
    "families = []\n"
    "packagers = []\n"
    'publishers = ["acme"]\n\n'
    "[machines.other]\nreserve_ram_gib = 8.0\nreserve_vram_gib = 1.0\nwriter = false\n\n"
    '[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n\n'
    "[defaults]\nreserve_ram_gib = 12.0\nreserve_vram_gib = 2.0\n\n"
    "[updates]\ncheck = false\n"
)


def test_two_runs_that_create_the_configuration_at_once_keep_the_first_ones_file(tmp_path: Path):
    """The second run found no file, and one is there when it writes: it takes it over as it is.

    The other run writes its file while this one asks this machine's host name -- after the
    folder was chosen and found empty, before the first write. The initial values of a new file
    replace nothing; this machine's own entry comes from the writer step, as in any folder.
    """
    config_file = _config_file(tmp_path)
    chosen: list[str] = []

    class _Asker(FileAsker):
        def select(self, key, question, choices, instruction=None):
            chosen.append(key)
            return super().select(key, question, choices, instruction)

    def hostname() -> str:
        if "results" in chosen and not config_file.exists():
            config_file.write_text(FOREIGN_CONFIGURATION, encoding="utf-8")
        return "workstation"

    probes = dataclasses.replace(windows_probes(), hostname=hostname)
    lines: list[str] = []
    code = run_guided(
        _Asker(FULL_ANSWERS),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=probes,
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    assert code == 0, lines
    stored = load_config(config_file)
    assert "other" in stored.machines
    assert (stored.defaults.reserve_ram_gib, stored.updates.check) == (12.0, False)
    assert "acme" in stored.publishers
    assert stored.machines["workstation"].writer is True


def test_a_second_conflict_in_a_row_is_a_dialog_error(tmp_path: Path):
    """Exit `2` like every other step that cannot go on; the other run's file stays as it is."""
    from modelroom.guided import _write_config

    assert _run(tmp_path)[0] == 0
    run = _guided_run_for(tmp_path, [])
    config_file = _config_file(tmp_path)
    calls = {"count": 0}

    def change(current):
        calls["count"] += 1
        text = config_file.read_text(encoding="utf-8")
        config_file.write_text(text + f"\n[machines.other-{calls['count']}]\nreserve_ram_gib = 8.0\n"
                               "reserve_vram_gib = 1.0\nwriter = false\n", encoding="utf-8")
        return current.model_copy(update={"guided": current.guided.model_copy(update={"context": 4096})})

    with pytest.raises(GuidedError, match="changed by another run since it was read"):
        _write_config(run, config_file, change)

    assert {"other-1", "other-2"} <= set(load_config(config_file).machines)


def test_an_ollama_pair_of_the_configuration_stands_in_the_list(tmp_path: Path):
    """A pair the catalog does not know was a dash in the list, while the fetch used it."""
    assert _run(tmp_path)[0] == 0
    config_file = _config_file(tmp_path)
    data = load_config(config_file).model_dump(mode="json")
    data["families"].append(
        {"name": "deepseek-r1", "base_models": [{"hf_repo": DEEPSEEK_BASE, "ollama_base": "my-deepseek", "ollama_tag": "8b"}]}
    )
    data["publishers"] = [*data["publishers"], "deepseek-ai"]
    from modelroom.toml_writer import dump_toml

    config_file.write_text(dump_toml(data, "a pair the catalog does not know"), encoding="utf-8")
    lines: list[str] = []
    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"}
    asker = _WatchingAsker(answers, lines)

    run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=lines.append,
    )

    row = next(choice for choice in asker.choices["select"] if choice.value == DEEPSEEK_BASE)
    assert row.label.rstrip().endswith("my-deepseek:8b")


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


def _search_log(tmp_path: Path) -> dict:
    return json.loads((_results(tmp_path) / "state" / "search.json").read_text(encoding="utf-8"))


def test_the_search_leaves_two_notes_and_puts_the_accounts_into_search_json(tmp_path: Path):
    """Where it asked, and how much of it is a choice -- the rest is a file (decided 2026-09-24)."""
    _code, lines = _run(tmp_path)

    notes = [line.strip() for line in lines]
    assert "searched Hugging Face at the publisher Qwen and the five listed packagers" in notes
    assert "4 repositories, 2 of them models you can pick from" in notes
    # The seven account lines, the request count and the budget are in the file, not on the screen.
    assert not any(line.strip().startswith(("Qwen 2", "unsloth 6", "bartowski 0")) for line in lines)
    assert not any("cannot be picked:" in line for line in lines)
    assert not any("budget " in line for line in lines)
    log = _search_log(tmp_path)
    assert log["schema_version"] == 3
    assert log["mode"] == "word"
    # Fourteen requests for seven accounts since 2026-09-25: newest and most downloaded each.
    assert (log["word"], log["filter_owners"], log["requests"], log["resolved"]) == ("qwen", True, 14, 3)
    assert log["budget"]["limit"] == 150
    # Two, because both sort orders of `Qwen` answer with its one repository (the fixture binds
    # the same curated page to each); the duplicate is listed once and counted as already listed.
    assert {entry["account"]: entry["hits"] for entry in log["accounts"]}["Qwen"] == 2
    assert {entry["account"]: entry["class"] for entry in log["accounts"]}["unsloth"] == "packager"
    assert log["unresolved"] == [{"reason": "the repository does not say it packages a base model", "count": 1}]


def test_the_search_log_holds_the_same_numbers_the_old_lines_said(tmp_path: Path):
    """`group_lines`/`summary_line` are the counter-probe: one search, one set of numbers."""
    _code, _lines = _run(tmp_path)
    log = _search_log(tmp_path)

    assert log["resolved"] + sum(entry["count"] for entry in log["unresolved"]) == 4
    # `hits` is what the two pages of an account answered with together, duplicates included.
    assert sum(entry["hits"] for entry in log["accounts"]) == 8
    assert log["requests"] == 2 * len(log["accounts"])


def test_with_the_filter_off_the_two_open_lists_are_in_the_note_and_in_the_file(tmp_path: Path):
    _code, lines = _run(tmp_path, {**FULL_ANSWERS, "filter_owners": False})

    assert any(line.strip().endswith("listed packagers, and the two open lists") for line in lines)
    log = _search_log(tmp_path)
    assert [entry["account"] for entry in log["accounts"] if entry["class"] == "open"] == [
        "most downloaded",
        "newest",
    ]
    assert log["requests"] == 16
    assert log["filter_owners"] is False


def test_without_the_filter_a_derivative_is_in_the_file_with_its_reason(tmp_path: Path):
    """"That it is apparent" is the point: a fine-tune is named, not hidden (decided 2026-09-24)."""
    _code, _lines = _run(tmp_path, {**FULL_ANSWERS, "filter_owners": False})

    reasons = [entry["reason"] for entry in _search_log(tmp_path)["unresolved"]]
    assert "a fine-tune or a merge, not a quantization of one base model" in reasons


def test_a_full_account_page_invites_a_more_specific_word(tmp_path: Path):
    from modelroom.search_pages import account_downloads_url, account_search_url

    page = json_response("hf_search_qwen_page_full.json")
    empty = json_response("hf_search_none.json")
    mapping = {
        **guided_transport_mapping(),
        ("GET", account_search_url("qwen", "unsloth")): page,
        ("GET", account_downloads_url("qwen", "unsloth")): empty,
    }
    lines: list[str] = []
    run_guided(
        FileAsker({**FULL_ANSWERS, "select": []}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(mapping),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    assert any("a more specific word shortens the list" in line for line in lines)
    assert any(entry["page_full"] for entry in _search_log(tmp_path)["accounts"])


def test_an_empty_search_answer_asks_the_catalog_pages_and_says_what_it_did(tmp_path: Path):
    """"I have no model in mind, show me what fits" is a valid answer (decided 2026-09-25)."""
    from fixture_support import catalog_transport_mapping

    mapping = {**guided_transport_mapping(), **catalog_transport_mapping()}
    lines: list[str] = []
    run_guided(
        FileAsker({**FULL_ANSWERS, "search": "", "select": []}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(mapping),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    assert any("anything that fits this machine" in line for line in lines)
    assert any("no search word: the catalog's current models, one page each" in line for line in lines)
    assert any("current models of this catalog" in line for line in lines)
    assert not any("open lists" in line for line in lines)
    log = _search_log(tmp_path)
    assert (log["mode"], log["word"]) == ("catalog", "")
    assert [entry["class"] for entry in log["accounts"]] == ["catalog"] * len(log["accounts"])


def test_an_empty_search_answer_offers_the_models_those_pages_resolved(tmp_path: Path):
    from fixture_support import CATALOG_PAGE_REPO, catalog_transport_mapping

    mapping = {**guided_transport_mapping(), **catalog_transport_mapping()}
    lines: list[str] = []
    asker = _WatchingAsker({**FULL_ANSWERS, "search": "", "select": []}, lines)
    run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(mapping),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    assert "Qwen3.8-27B" in [choice.value.partition("/")[2] for choice in asker.choices["select"]]
    assert CATALOG_PAGE_REPO in [choice.value for choice in asker.choices["select"]]


def test_a_typed_repository_id_is_searched_as_one_page_and_reaches_the_configuration(tmp_path: Path):
    from fixture_support import typed_transport_mapping

    mapping = {**guided_transport_mapping(), **typed_transport_mapping()}
    code, lines = _run(
        tmp_path,
        {**FULL_ANSWERS, "search": UNSLOTH, "select": [UNSLOTH]},
        transport=build_transport(mapping),
    )

    assert code == 0
    assert any(f"? What are you looking for?  {UNSLOTH}" in line for line in lines)
    assert any(f"searched Hugging Face for the repository {UNSLOTH}" in line for line in lines)
    log = _search_log(tmp_path)
    assert (log["mode"], log["word"], log["requests"]) == ("id", UNSLOTH, 1)
    assert [entry["account"] for entry in log["accounts"]] == ["typed"]
    assert [entry["class"] for entry in log["accounts"]] == ["typed"]
    assert UNSLOTH in load_config(_config_file(tmp_path)).families[0].base_models[0].repos


def test_a_typed_id_without_a_gguf_file_says_so_and_the_word_search_follows(tmp_path: Path):
    from fixture_support import TYPED_NO_GGUF_REPO, typed_transport_mapping

    mapping = {**guided_transport_mapping(), **typed_transport_mapping()}
    code, lines = _run(
        tmp_path,
        {**FULL_ANSWERS, "search": TYPED_NO_GGUF_REPO},
        transport=build_transport(mapping),
    )

    assert code == 0
    assert any(f"{TYPED_NO_GGUF_REPO} holds no GGUF file" in line for line in lines)
    assert _search_log(tmp_path)["mode"] == "word"


def test_a_typo_ends_in_a_list_of_candidates_and_the_search_they_lead_to(tmp_path: Path):
    from modelroom.search import DEFAULT_PACKAGERS
    from modelroom.search_pages import account_downloads_url, account_search_url

    empty = json_response("hf_search_none.json")
    mapping = {**guided_transport_mapping()}
    for account in ("Qwen", "deepseek-ai", *DEFAULT_PACKAGERS):
        mapping[("GET", account_search_url("qwn", account))] = empty
        mapping[("GET", account_downloads_url("qwn", account))] = empty
    lines: list[str] = []
    asker = _WatchingAsker({**FULL_ANSWERS, "search": "qwn", "did_you_mean": "qwen"}, lines)

    code = run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(mapping),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    assert code == 0
    assert "did_you_mean" in asker.asked
    # The second search is the one the step reports on, so the file and the notes are about `qwen`.
    assert _search_log(tmp_path)["word"] == "qwen"
    assert any("? Nothing was found. Did you mean one of these?  qwen" in line for line in lines)
    assert load_config(_config_file(tmp_path)).families[0].base_models[0].hf_repo == "Qwen/Qwen3.5-9B"


def test_none_of_these_keeps_the_word_and_the_step_stays_empty(tmp_path: Path):
    from modelroom.search import DEFAULT_PACKAGERS
    from modelroom.search_pages import account_downloads_url, account_search_url

    empty = json_response("hf_search_none.json")
    mapping = {**guided_transport_mapping()}
    for account in ("Qwen", "deepseek-ai", *DEFAULT_PACKAGERS):
        mapping[("GET", account_search_url("qwn", account))] = empty
        mapping[("GET", account_downloads_url("qwn", account))] = empty
    lines: list[str] = []
    run_guided(
        FileAsker({**FULL_ANSWERS, "search": "qwn", "did_you_mean": "keep", "select": []}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(mapping),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    assert any("none of these" in line for line in lines)
    assert any("nothing added" in line for line in lines)
    assert _search_log(tmp_path)["word"] == "qwn"


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
    # The column head, then two models, then the repositories an answer file of the older shape may
    # name instead. The order is the memory pool first: at 32k the smaller model fits the graphics
    # card and the 9B does not, so the 8B stands above it (the fit of the list is computed at the
    # context the scale starts on since 2026-09-24).
    assert listed[0].heading is True
    assert listed[0].label.split() == ["Model", "Fit", "Size", "Release", "Packagers", "Downl.", "Ollama"]
    assert sorted(choice.value for choice in listed[1:3]) == sorted(["Qwen/Qwen3.5-9B", DEEPSEEK_BASE])
    assert sorted(choice.value for choice in listed[3:]) == sorted([QWEN_GGUF, UNSLOTH, DEEPSEEK])
    assert all(choice.disabled is None for choice in listed)
    qwen = next(choice for choice in listed[1:3] if choice.value == "Qwen/Qwen3.5-9B")
    assert qwen.label.startswith("Qwen3.5-9B")
    assert any(word in qwen.label for word in ("good", "marginal", "too tight", "unknown"))
    # Twelve characters for the packagers since 2026-09-25: the first account and a count.
    assert "Qwen +1" in qwen.label
    assert all(len(choice.label) <= 95 for choice in listed[:3])


def test_the_hint_of_the_list_is_its_instruction_line_and_names_the_context_of_the_fit(tmp_path: Path):
    """A sentence that explains a list has nothing to say once the list is gone (2026-09-24)."""
    lines: list[str] = []
    asker = _WatchingAsker(FULL_ANSWERS, lines)
    run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    instruction = asker.instructions["select"]
    assert instruction.startswith("latest: the publisher's current release of its family")
    assert "at 32k context" in instruction
    assert not any(line.strip().startswith("fit at ") for line in lines)
    assert not any("can be picked;" in line for line in lines)


def test_a_model_is_fetched_from_the_publisher_and_from_one_listed_packager(tmp_path: Path):
    """Two repositories per model, because every one costs the fetch about ten requests."""
    answers = {**FULL_ANSWERS, "select": ["Qwen/Qwen3.5-9B"]}

    code, _lines = _run(tmp_path, answers)

    assert code == 0
    base_model = load_config(_config_file(tmp_path)).families[0].base_models[0]
    assert sorted(base_model.repos) == sorted([QWEN_GGUF, UNSLOTH])


def test_a_search_that_resolves_no_model_asks_no_list(tmp_path: Path):
    """Test round of 2026-09-24 22:49: `mistral` with the filter on found 59 repositories and no
    model to pick, and the list question was asked anyway -- an empty list crashes the dialog."""
    from modelroom.catalog import load_catalog
    from modelroom.search import DEFAULT_PACKAGERS
    from modelroom.search_pages import account_downloads_url, account_search_url, search_accounts

    class _Refusing(FileAsker):
        def checkbox(self, key: str, question: str, choices, instruction=None) -> list[str]:
            assert list(choices), f"{key}: a list with nothing in it was asked"
            return super().checkbox(key, question, choices, instruction)

    empty = Response(status=200, headers={}, body=b"[]")
    # Every account the search asks, the catalog's matching publishers included -- `mistralai` is
    # a publisher since 2026-09-25 -- each answering with the empty list: this test is about the
    # list question that follows a search nothing resolves, not about Mistral.
    asked = search_accounts(load_catalog(), "mistral", DEFAULT_PACKAGERS)
    mapping = {
        **guided_transport_mapping(),
        **{("GET", account_search_url("mistral", account)): empty for account in asked},
        **{("GET", account_downloads_url("mistral", account)): empty for account in asked},
    }
    lines: list[str] = []
    code = run_guided(
        _Refusing({**FULL_ANSWERS, "search": "mistral", "select": []}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(mapping),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )

    assert code == 1  # nothing to render: no family, so no snapshot
    assert any("none of them a model you can pick from" in line for line in lines)
    assert any("nothing added" in line for line in lines)
    assert any(line.strip() == "– Packages        nothing chosen, the folder stays as it is" for line in lines)
    assert load_config(_config_file(tmp_path)).families == []


def test_choosing_nothing_leaves_the_configuration_as_it_is(tmp_path: Path):
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "select": []})

    assert code == 1  # nothing to render: no family, so no snapshot
    assert load_config(_config_file(tmp_path)).families == []
    assert any("nothing chosen, the folder stays as it is" in line for line in lines)
    assert any("nothing to fetch" in line for line in lines)


def test_a_step_five_without_a_snapshot_says_so_and_still_draws_its_card(tmp_path: Path):
    """The last step was a head and nothing under it (test round, 2026-09-25): a run that chose
    nothing ended on an empty screen, with the reason on stderr where no reader of the run is."""
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "select": []})
    stripped = [line.strip() for line in lines]

    assert code == 1
    assert "nothing to rank: no package in this folder yet" in stripped
    card = _card(lines)
    assert str(_results(tmp_path)) in card["folder"]
    assert "GB graphics" in card["machine"]
    assert card["models"].startswith("models    none")
    assert card["context"].startswith("context   S 8k")
    assert card["speed"].startswith("speed     not measured")  # the card, not the note under the table
    assert card["result"].split() == ["result", "–"]
    assert stripped[-1] == "– Results         nothing to rank"


def test_a_render_that_was_stopped_says_nothing_about_the_folder(tmp_path: Path):
    """`nothing to rank` is a statement about the folder, and a held lock is one about this run.

    Every render that writes no document leaves step 5 without a card; only one of them means the
    folder holds no package (second-model round, 2026-09-25).
    """
    from modelroom.catalog import load_catalog
    from modelroom.guided import GuidedRun, _render_step
    from modelroom.guided_context import context_scenario
    from modelroom.screen import Screen

    assert _run(tmp_path)[0] == 0
    config = load_config(_config_file(tmp_path))
    lines: list[str] = []
    run = GuidedRun(
        asker=FileAsker({}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport({}),
        probes=windows_probes(),
        now=RUN1,
        catalog=load_catalog(),
        out=lines.append,
        screen=Screen(lines.append, colored=False),
    )
    handle = acquire_lock(config.paths.lock_file, "fetch", RUN1)
    try:
        code, first = _render_step(run, _config_file(tmp_path), config, context_scenario(8192))
    finally:
        release_lock(handle)

    assert code == 1
    assert first is None  # no document, so no first row a pull could be about
    assert not any("nothing to rank" in line for line in lines)
    assert not any(line.strip().startswith("result ") for line in lines)


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
    # The head of the machine's table says the context; the whole scenario sentence -- the KV cache
    # and the requests with their origin -- stands once on the screen, in the card's `context` row
    # (decided 2026-09-26; until then only the two files carried it, decided 2026-09-24).
    printed = "\n".join(lines)
    assert "· context XS 4k" in printed
    assert [line.split()[0] for line in lines if "KV cache" in line] == ["context"]


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
        def select_or_text(self, key, question, choices, text_value, text_question, instruction=None):
            pointed_at[key] = next(choice.value for choice in choices if choice.checked)
            return super().select_or_text(key, question, choices, text_value, text_question, instruction)

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
    """The write is not a line of its own any more; the file's own modification time shows it."""
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "context": "4096"})
    assert code == 0
    assert not any("kept context" in line for line in lines)
    written = _config_file(tmp_path).read_bytes()

    code, _second = _run(tmp_path, {**FULL_ANSWERS, "context": "4096"}, now=RUN2)

    assert code == 0
    assert _config_file(tmp_path).read_bytes() == written
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
        def select_or_text(self, key, question, choices, text_value, text_question, instruction=None):
            if key == "context":
                _store_context(_config_file(tmp_path), 8192)
            return super().select_or_text(key, question, choices, text_value, text_question, instruction)

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
    assert "Scenario: context 4k (entered)" in text
    assert _ranked_deepseek(tmp_path)["measurement_group"] == 0


def test_a_configuration_without_a_kept_context_still_renders_with_8192(tmp_path: Path):
    assert _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())[0] == 0
    config_file = _config_file(tmp_path)
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace("context = 8192\n", ""), encoding="utf-8"
    )

    assert main(["render", "--config", str(config_file)], now=RUN2) == 0

    text = (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")
    assert "Scenario: context 8k (default)" in text


def test_a_kept_context_of_8192_is_entered_when_render_runs_alone(tmp_path: Path):
    """A kept `[guided].context` is what a guided run chose, 8192 included -- `default` is only
    what `render` assumes for a folder that kept none. The dialog and a later `render --config`
    of the same folder must name the same origin."""
    assert _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())[0] == 0
    assert "context = 8192\n" in _config_file(tmp_path).read_text(encoding="utf-8")

    assert main(["render", "--config", str(_config_file(tmp_path))], now=RUN2) == 0

    text = (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")
    assert "Scenario: context 8k (entered)" in text


def test_a_context_that_is_not_a_number_ends_the_run(tmp_path: Path):
    with pytest.raises(GuidedError, match="whole number"):
        _run(tmp_path, {**FULL_ANSWERS, "context": "lots"})


@pytest.mark.parametrize("answer", ["0", "-1", str(2**31)])
def test_a_context_the_scenario_cannot_take_ends_the_run_without_a_traceback(tmp_path: Path, answer: str):
    """`Scenario.context_requested` is bounded; an answer outside it is an input error, not a crash."""
    with pytest.raises(GuidedError):
        _run(tmp_path, {**FULL_ANSWERS, "context": answer})



# --- step 3, the head count: the requests the ranking assumes -----------------------------------------

USERS_QUESTION_LINE = "? How many people use it on a typical day?"
ONE_REQUEST_LINE = "measurements run one request; this ranking assumes {requests}"


def _document(tmp_path: Path) -> dict:
    return json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))


def _markdown(tmp_path: Path) -> str:
    return (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")


def _kept_requests(tmp_path: Path) -> tuple:
    guided = load_config(_config_file(tmp_path)).guided
    return guided.users, guided.requests, guided.requests_origin


def _answer_line(lines: list[str], question: str) -> str:
    return next(line.strip() for line in lines if line.strip().startswith(question))


def _pointed_at_in_a_second_run(tmp_path: Path, answers: dict) -> dict[str, str]:
    """The entry each list of a second run in the same folder starts on."""
    pointed_at: dict[str, str] = {}

    class _Recording(FileAsker):
        def select_or_text(self, key, question, choices, text_value, text_question, instruction=None):
            pointed_at[key] = next(choice.value for choice in choices if choice.checked)
            return super().select_or_text(key, question, choices, text_value, text_question, instruction)

    run_guided(
        _Recording(answers),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=lambda _line: None,
    )
    return pointed_at


def test_the_head_count_is_asked_before_the_scale(tmp_path: Path):
    """The last column of the scale counts with the requests, so they have to be known first."""
    asker = FileAsker({**FULL_ANSWERS, "users": 25})
    run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lambda _line: None,
    )

    assert asker.asked.index("users") + 1 == asker.asked.index("context")


def test_a_head_count_is_kept_with_its_requests_and_reaches_the_document(tmp_path: Path):
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "users": 25})

    assert code == 0
    assert _kept_requests(tmp_path) == (25, 3, "from_users")
    raw = tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))["guided"]
    assert (raw["users"], raw["requests"], raw["requests_origin"]) == (25, 3, "from_users")
    payload = _document(tmp_path)
    assert (payload["users"], payload["requests_origin"], payload["scenario"]["requests"]) == (25, "from_users", 3)
    fits = [row["fit"] for row in payload["machines"][0]["ranked"] + payload["machines"][0]["too_tight"]]
    assert fits and {fit["requests"] for fit in fits} == {3}
    assert _answer_line(lines, USERS_QUESTION_LINE) == f"{USERS_QUESTION_LINE}  25 (3 requests)"
    sentence = "context 8k (entered), KV cache f16 (assumed), 3 requests (from 25 users)"
    assert f"Scenario: {sentence}" in _markdown(tmp_path)
    assert _card(lines)["context"] == "context   " + sentence.removeprefix("context ")


def test_one_person_is_a_head_count_too_and_says_so_in_the_singular(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    assert _kept_requests(tmp_path) == (1, 1, "from_users")
    assert (_document(tmp_path)["users"], _document(tmp_path)["requests_origin"]) == (1, "from_users")
    assert _answer_line(lines, USERS_QUESTION_LINE) == f"{USERS_QUESTION_LINE}  1 (1 request)"
    assert "Scenario: context 8k (entered), KV cache f16 (assumed), 1 request (from 1 user)" in _markdown(tmp_path)
    # One request: no hint, no sentence in step 4.
    printed = "\n".join(lines)
    assert "throughput" not in printed and "measurements run one request" not in printed
    assert "throughput" not in _markdown(tmp_path)


def test_requests_named_outright_are_entered_and_carry_the_hint_beyond_eight(tmp_path: Path):
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "users": "12 requests"})

    assert code == 0
    assert _kept_requests(tmp_path) == (None, 12, "entered")
    assert "users" not in tomllib.loads(_config_file(tmp_path).read_text(encoding="utf-8"))["guided"]
    payload = _document(tmp_path)
    assert (payload["users"], payload["requests_origin"], payload["scenario"]["requests"]) == (None, "entered", 12)
    assert _answer_line(lines, USERS_QUESTION_LINE) == f"{USERS_QUESTION_LINE}  12 requests"
    markdown = _markdown(tmp_path)
    assert "Scenario: context 8k (entered), KV cache f16 (assumed), 12 requests (entered)" in markdown
    assert f"Load: {OLLAMA_REQUEST_HINT_TEXT}" in markdown
    card = _card(lines)
    assert card["load"].startswith("load      beyond 8 requests at once")
    assert " ".join(" ".join(line.split()) for line in lines).count(OLLAMA_REQUEST_HINT_TEXT.split(": ")[1]) == 1
    assert all(len(line) <= 100 for line in lines if "throughput" in line or "serving stack" in line)


def test_the_second_run_offers_the_kept_head_count(tmp_path: Path):
    assert _run(tmp_path, {**FULL_ANSWERS, "users": 25})[0] == 0

    assert _pointed_at_in_a_second_run(tmp_path, {**FULL_ANSWERS, "users": 25})["users"] == "25"


def test_requests_without_a_head_count_leave_the_list_on_one(tmp_path: Path):
    """Twelve slots say nothing about how many people there are; no head count is made up."""
    assert _run(tmp_path, {**FULL_ANSWERS, "users": "12 requests"})[0] == 0

    assert _pointed_at_in_a_second_run(tmp_path, {**FULL_ANSWERS, "users": "12 requests"})["users"] == "1"


def test_requests_named_outright_keep_the_head_count_the_folder_holds(tmp_path: Path):
    assert _run(tmp_path, {**FULL_ANSWERS, "users": 25})[0] == 0

    assert _run(tmp_path, {**FULL_ANSWERS, "users": "12 requests"}, now=RUN2)[0] == 0

    assert _kept_requests(tmp_path) == (25, 12, "entered")
    assert "12 requests (entered)" in _markdown(tmp_path)


def test_the_same_head_count_again_writes_nothing(tmp_path: Path):
    assert _run(tmp_path, {**FULL_ANSWERS, "users": 25})[0] == 0
    written = _config_file(tmp_path).read_bytes()

    assert _run(tmp_path, {**FULL_ANSWERS, "users": 25}, now=RUN2)[0] == 0

    assert _config_file(tmp_path).read_bytes() == written


def test_a_change_another_process_writes_while_the_head_count_is_asked_is_kept(tmp_path: Path):
    """The head count is written on top of the file as it is then, not over it."""

    class _WritesInBetween(FileAsker):
        def select_or_text(self, key, question, choices, text_value, text_question, instruction=None):
            if key == "users":
                config_file = _config_file(tmp_path)
                text = config_file.read_text(encoding="utf-8")
                config_file.write_text(text.replace("reserve_ram_gib = 8.0", "reserve_ram_gib = 12.0", 1), encoding="utf-8")
            return super().select_or_text(key, question, choices, text_value, text_question, instruction)

    assert _run(tmp_path)[0] == 0
    run_guided(
        _WritesInBetween({**FULL_ANSWERS, "users": 25}),
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=lambda _line: None,
    )

    stored = load_config(_config_file(tmp_path))
    assert 12.0 in [stored.defaults.reserve_ram_gib, *(machine.reserve_ram_gib for machine in stored.machines.values())]
    assert (stored.guided.users, stored.guided.requests, stored.guided.requests_origin) == (25, 3, "from_users")


@pytest.mark.parametrize("answer", ["3 people", "0", "10241", "0 requests"])
def test_a_head_count_the_ranking_cannot_take_ends_the_run_naming_both_forms(tmp_path: Path, answer):
    with pytest.raises(GuidedError, match="N requests"):
        _run(tmp_path, {**FULL_ANSWERS, "users": answer})


def test_an_answer_file_without_the_head_count_is_exit_2_and_names_it(tmp_path: Path, capsys):
    """An answer file of 0.1.0 has no `users`; like every other missing answer it stops the run."""
    answers = tmp_path / "answers.toml"
    body = [f"{key} = {json.dumps(value)}" for key, value in FULL_ANSWERS.items() if key != "users"]
    answers.write_text("schema_version = 1\n" + "\n".join(body) + "\n", encoding="utf-8")

    code = main(
        ["--answers", str(answers)],
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        pointer_path=_pointer(tmp_path),
        daemon=offline_daemon(),
        now=RUN1,
        here=_results(tmp_path),
        out=[].append,
    )

    assert code == 2
    assert "no answer for 'users'" in capsys.readouterr().err


class _EditsWhileAsked(FileAsker):
    """A `FileAsker` that lets another process rewrite the configuration while `key` is asked."""

    def __init__(self, answers: dict, config_file: Path, key: str, edits: dict[str, str]) -> None:
        super().__init__(answers)
        self._config_file, self._key, self._edits = config_file, key, edits
        self.labels: dict[str, list[str]] = {}

    def select_or_text(self, key, question, choices, text_value, text_question, instruction=None):
        self.labels[key] = [choice.label for choice in choices]
        if key == self._key:
            text = self._config_file.read_text(encoding="utf-8")
            for old, new in self._edits.items():
                assert old in text, old
                text = text.replace(old, new)
            self._config_file.write_text(text, encoding="utf-8")
        return super().select_or_text(key, question, choices, text_value, text_question, instruction)


def _run_with(tmp_path: Path, asker: FileAsker, now=RUN2) -> int:
    return run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=now,
        out=lambda _line: None,
    )


def test_a_head_count_written_by_another_run_during_the_scale_does_not_split_the_run(tmp_path: Path):
    """The run answered 25 people; another run writes 100 while the scale is asked. The ranking of
    this run, its document and the configuration it leaves all say 25 -- the last write of step 3
    applies the whole answer again, so the document never names an origin its number did not
    come from (second-model round, 2026-09-26)."""
    assert _run(tmp_path)[0] == 0
    other = {"users = 25\nrequests = 3\n": "users = 100\nrequests = 10\n"}
    asker = _EditsWhileAsked({**FULL_ANSWERS, "users": 25}, _config_file(tmp_path), "context", other)

    assert _run_with(tmp_path, asker) == 0

    assert _kept_requests(tmp_path) == (25, 3, "from_users")
    payload = _document(tmp_path)
    assert (payload["users"], payload["requests_origin"], payload["scenario"]["requests"]) == (25, "from_users", 3)


def test_requests_named_outright_keep_a_head_count_another_run_wrote_meanwhile(tmp_path: Path):
    """`12 requests` answers no head count, so the head count is the file's -- as it is when the
    answer is written, not as it was when the question was asked."""
    assert _run(tmp_path, {**FULL_ANSWERS, "users": 25})[0] == 0
    other = {"users = 25\nrequests = 3\n": "users = 50\nrequests = 5\n"}
    asker = _EditsWhileAsked({**FULL_ANSWERS, "users": "12 requests"}, _config_file(tmp_path), "users", other)

    assert _run_with(tmp_path, asker) == 0

    assert _kept_requests(tmp_path) == (50, 12, "entered")


def test_the_scale_counts_with_the_reserves_the_file_holds_after_the_head_count(tmp_path: Path):
    """Reserves another run wrote while the head count was asked are the ones the ranking uses, so
    the last column of the scale uses them too."""
    assert _run(tmp_path)[0] == 0
    tight = {"reserve_ram_gib = 8.0": "reserve_ram_gib = 127.0", "reserve_vram_gib = 1.0": "reserve_vram_gib = 11.5"}
    asker = _EditsWhileAsked(dict(FULL_ANSWERS), _config_file(tmp_path), "users", tight)

    assert _run_with(tmp_path, asker) == 0

    levels = asker.labels["context"][:6]
    assert all("none of" in label for label in levels), levels

def test_the_load_test_measures_one_request_and_the_ranking_says_why_it_does_not_count(tmp_path: Path):
    """Step 4 measures with a copy of the scenario at one request and the same context; the render
    computes for three. The measurement stays, the row names the reason, and neither the table nor
    the card tells the user to measure what was just measured."""
    answers = {**LOAD_TEST_ANSWERS, "select": [DEEPSEEK], "users": 25}

    code, lines = _run(tmp_path, answers, daemon=loadtest_daemon())

    assert code == 0
    measured = json.loads(_measurement_files(tmp_path)[0].read_text(encoding="utf-8"))
    payload = _document(tmp_path)
    assert measured["scenario"]["requests"] == 1
    assert measured["scenario"]["context_requested"] == payload["scenario"]["context_requested"] == 8192
    assert payload["scenario"]["requests"] == 3
    row = _ranked_deepseek(tmp_path)
    assert (row["measurement_group"], row["speed_tps"]) == (1, None)
    assert row["note"]["text"].startswith("measured with 1 request, ranking assumes 3")
    step_4 = lines[next(index for index, line in enumerate(lines) if "Step 4 of 5" in line) :]
    sentence = next(index for index, line in enumerate(step_4) if ONE_REQUEST_LINE.format(requests=3) in line)
    assert sentence < next(index for index, line in enumerate(step_4) if "Measure the speed" in line)
    speed = [line for line in lines if line.strip().startswith("speed ")]
    assert any(f"#{row['rank']} measured with 1 request, ranking assumes 3" in line for line in speed)
    assert not any("say Yes in step 4" in line for line in speed)
    assert _card(lines)["speed"].endswith(f"#{row['rank']} measured with 1 request, ranking assumes 3")


def test_a_run_of_one_request_says_nothing_about_one_request_in_step_4(tmp_path: Path):
    code, lines = _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())

    assert code == 0
    assert not any("measurements run one request" in line for line in lines)
    assert _ranked_deepseek(tmp_path)["measurement_group"] == 0

# --- the last steps: fetch, the load test, render -----------------------------------------------------


def test_the_run_ends_with_both_views_and_a_word_on_the_load_test(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    assert any(line.strip().startswith("no load test: ") for line in lines)
    assert (_results(tmp_path) / "docs" / "models.md").is_file()
    assert (_results(tmp_path) / "docs" / "models.json").is_file()
    printed = "\n".join(lines)
    # The rule and the snapshot time are in the Markdown file; the screen shows the table.
    assert "Ranking rule: fit class" in (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")
    assert "workstation · context" in printed
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
    measured = next(line.strip() for line in lines if line.strip().startswith("measured hf.co/"))
    assert measured.startswith(f"measured {DEEPSEEK_OLLAMA_NAME}: ")
    assert "tok/s" in measured and measured.endswith("context 8k")
    assert "valid" not in measured and "comparable" not in measured
    assert any(line.strip().endswith("1 of 1 measured") for line in lines)
    row = _ranked_deepseek(tmp_path)
    assert row["measurement_group"] == 0
    assert row["speed_tps"] == pytest.approx(record["tps_mean"])


def test_a_no_to_the_load_test_measures_nothing_and_asks_for_no_package(tmp_path: Path):
    answers = {**LOAD_TEST_ANSWERS, "load_test": False}
    answers.pop("load_test_packages")

    code, lines = _run(tmp_path, answers, daemon=loadtest_daemon())

    assert code == 0
    assert _measurement_files(tmp_path) == []
    assert not any(line.strip().startswith("measured hf.co/") for line in lines)


def test_nothing_installed_that_is_ranked_is_one_note_and_no_question(tmp_path: Path):
    """The answer file has no load-test answer at all, and the run still ends with exit 0."""
    code, lines = _run(tmp_path, FULL_ANSWERS, daemon=loadtest_daemon())

    assert code == 0
    assert NO_CANDIDATE_LINE in [line.strip() for line in lines]
    assert _measurement_files(tmp_path) == []


def test_what_cannot_be_measured_is_a_count_and_a_reason_without_names(tmp_path: Path):
    """Without a candidate the reason is the note; the names of the cloud models are not."""
    _code, lines = _run(tmp_path, FULL_ANSWERS, daemon=loadtest_daemon())

    reason = next(line.strip() for line in lines if "cloud model" in line)
    assert reason.startswith("1 installed model -- ")
    assert "glm-5.3-flash:cloud" not in reason


def test_with_a_candidate_on_the_list_what_was_left_out_is_not_said_at_all(tmp_path: Path):
    """The twelve names of the cloud models pushed the question off the screen (2026-09-24)."""
    _code, lines = _run(tmp_path, LOAD_TEST_ANSWERS, daemon=loadtest_daemon())

    assert not any("cloud" in line for line in lines)


def test_the_selection_list_shows_every_candidate_unchecked_with_its_four_facts(tmp_path: Path):
    asked: dict[str, list] = {}

    class _Recording(FileAsker):
        def checkbox(self, key, question, choices, instruction=None):
            asked[key] = list(choices)
            return super().checkbox(key, question, choices, instruction)

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
    # The note names what is missing instead of a speed; the run still ends `0`, because the
    # record is written and the ranking says where such a measurement stands.
    assert any("not comparable (" in line for line in lines)
    assert not any(line.strip().startswith("measured hf.co/") for line in lines)
    row = _ranked_deepseek(tmp_path)
    assert (row["measurement_group"], row["speed_tps"]) == (1, None)


def test_an_answer_file_that_does_not_mention_the_load_test_measures_nothing(tmp_path: Path):
    """The one optional answer: a file written for another machine still runs to the end."""
    answers = {**LOAD_TEST_ANSWERS}
    answers.pop("load_test")

    code, lines = _run(tmp_path, answers, daemon=loadtest_daemon())

    assert code == 0
    assert _measurement_files(tmp_path) == []
    assert not any(line.strip().startswith("measured hf.co/") for line in lines)


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
    assert any(line.startswith("nothing measured: ") for line in lines)  # a fault line, not a note
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
    assert any(line.strip().startswith("no load test: ") for line in lines)


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

    assert load_config(_config_file(tmp_path)).schema_version == 3


# --- a machine entered by hand (step 1, `modelroom/guided_entered.py`) ------------------------------

ENTERED_UNIFIED = {
    **FULL_ANSWERS,
    "machines": ["enter"],
    "entered_name": "studio",
    "entered_ram": 32,
    "entered_gpu": "unified",
}
ENTERED_CARD = {**ENTERED_UNIFIED, "entered_name": "tower", "entered_ram": 64, "entered_gpu": "card", "entered_vram": 24}
ENTERED_NONE = {**ENTERED_UNIFIED, "entered_name": "box", "entered_ram": 64, "entered_gpu": "none"}


def _entered_files(tmp_path: Path) -> list[HardwareProfile]:
    """Every profile of the results folder whose memory was entered by hand."""
    profiles = [HardwareProfile.model_validate_json(path.read_text(encoding="utf-8")) for path in _profiles(tmp_path)]
    return [profile for profile in profiles if profile.ram_physical_source == "entered"]


def _answers_without_folder(answers: dict) -> dict:
    return {key: value for key, value in answers.items() if key != "results"}


def _measured_stem(tmp_path: Path) -> str:
    """The one other profile next to the entered one (`c...c`): the measurement of this machine."""
    stems = [path.stem for path in _profiles(tmp_path)]
    assert len(stems) == 2 and "c" * 16 in stems, stems
    return next(stem for stem in stems if stem != "c" * 16)


def _watched(tmp_path: Path, answers: dict) -> _WatchingAsker:
    """One run with an asker that keeps the choices of every list; returns the asker."""
    lines: list[str] = []
    asker = _WatchingAsker(answers, lines)
    run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=lines.append,
    )
    return asker


def test_the_machine_list_offers_entering_a_machine_by_hand_after_measuring_and_importing(tmp_path: Path):
    machines = _watched(tmp_path, {**FULL_ANSWERS, "machines": []}).choices["machines"]

    assert [choice.value for choice in machines if choice.disabled is None] == ["this-machine", "import", "enter"]
    assert next(choice.label for choice in machines if choice.value == "enter") == "enter a machine by hand"


@pytest.mark.parametrize(
    ("answers", "asked"),
    [
        (ENTERED_CARD, ["entered_name", "entered_ram", "entered_gpu", "entered_vram"]),
        (ENTERED_NONE, ["entered_name", "entered_ram", "entered_gpu"]),
        (ENTERED_UNIFIED, ["entered_name", "entered_ram", "entered_gpu"]),
    ],
)
def test_entering_asks_its_questions_in_order_and_graphics_memory_only_for_a_card(tmp_path: Path, answers, asked):
    asker = _watched(tmp_path, answers)

    assert [key for key in asker.asked if key.startswith("entered_")] == asked
    assert _entered_files(tmp_path), "the run did not get as far as the profile"


def test_every_answer_of_the_machine_entered_by_hand_is_written_back_in_words(tmp_path: Path):
    code, lines = _run(tmp_path, ENTERED_CARD)

    assert code == 0
    answers = [line.strip() for line in lines if line.strip().startswith("? ")]
    assert "? Which machines should the result cover?  a machine entered by hand" in answers
    assert "? What is the machine called?  tower" in answers
    assert "? How much memory does it have, in GiB?  64 GiB" in answers
    assert "? What runs the model?  a graphics card with its own memory" in answers
    assert "? How much graphics memory, in GiB?  24 GiB" in answers


@pytest.mark.parametrize(
    ("key", "answer", "said"),
    [
        ("entered_name", "", "1 to 128 characters"),
        ("entered_name", "   ", "1 to 128 characters"),
        ("entered_name", "x" * 129, "1 to 128 characters"),
        ("entered_ram", "", "not a number of GiB above 0"),
        ("entered_ram", "a lot", "not a number of GiB above 0"),
        ("entered_ram", 0, "not a number of GiB above 0"),
        ("entered_ram", -16, "not a number of GiB above 0"),
        ("entered_ram", "-1.5", "not a number of GiB above 0"),
        ("entered_ram", "nan", "not a number of GiB above 0"),
        ("entered_ram", "inf", "not a number of GiB above 0"),
        ("entered_vram", "", "not a number of GiB above 0"),
        ("entered_vram", "big", "not a number of GiB above 0"),
        ("entered_vram", 0, "not a number of GiB above 0"),
        ("entered_vram", -8, "not a number of GiB above 0"),
    ],
)
def test_an_answer_a_machine_entered_by_hand_cannot_take_ends_the_run_and_writes_no_profile(
    tmp_path: Path, key, answer, said
):
    with pytest.raises(GuidedError, match=said) as raised:
        _run(tmp_path, {**ENTERED_CARD, key: answer})

    assert key in str(raised.value)
    assert _profiles(tmp_path) == []


@pytest.mark.parametrize("answer", ["31.5", "31.999999"])
def test_a_memory_size_may_be_a_decimal_number_given_as_text_and_is_said_as_given(tmp_path: Path, answer):
    code, lines = _run(tmp_path, {**ENTERED_UNIFIED, "entered_ram": answer})

    assert code == 0
    assert [profile.ram_physical_gib for profile in _entered_files(tmp_path)] == [float(answer)]
    assert f"? How much memory does it have, in GiB?  {answer} GiB" in [line.strip() for line in lines]


def test_the_answer_file_names_the_question_of_the_machine_entered_by_hand_it_has_no_answer_for(tmp_path: Path, capsys):
    answers = tmp_path / "answers.toml"
    answers.write_text(
        'schema_version = 1\nresults = "here"\nmachines = ["enter"]\nentered_name = "studio"\n', encoding="utf-8"
    )

    code = main(
        ["--answers", str(answers)],
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        pointer_path=_pointer(tmp_path),
        now=RUN1,
        here=_results(tmp_path),
        out=[].append,
        daemon=offline_daemon(),
    )

    assert code == 2
    assert "no answer for 'entered_ram'" in capsys.readouterr().err


def test_an_answer_file_from_before_the_machine_entered_by_hand_runs_unchanged(tmp_path: Path):
    asker = FileAsker(FULL_ANSWERS)
    code = run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN1,
        out=[].append,
    )

    assert code == 0
    assert not any(key.startswith("entered_") for key in asker.asked)


@pytest.mark.parametrize(
    ("shape", "vram", "state", "source", "stored_vram"),
    [
        ("card", 24.0, "entered", "entered", 24.0),
        ("none", 0.0, "none", "none", 0.0),
        ("unified", 0.0, "unified_memory", "none", 0.0),
    ],
)
def test_the_profile_of_a_machine_entered_by_hand_is_one_the_fit_computes_for(shape, vram, state, source, stored_vram):
    from modelroom.guided_entered import entered_profile
    from modelroom.profile import fit_block_reason

    profile = entered_profile("studio", 64.0, shape, vram, "a" * 16, RUN1)

    assert HardwareProfile.model_validate(profile.model_dump(mode="json")) == profile
    assert fit_block_reason(profile) is None
    assert (profile.gpu_state, profile.vram_source, profile.vram_gib) == (state, source, stored_vram)
    assert (profile.origin, profile.ram_physical_source, profile.ram_physical_gib) == ("entered", "entered", 64.0)
    assert (profile.os_fingerprint, profile.os_fingerprint_source) == ("none", "none")
    assert (profile.ram_limit_gib, profile.ram_limit_scope) == (None, "none")
    assert (profile.gpu_name, profile.llmfit_version, profile.recorded_at) == (None, None, RUN1)
    checks = profile.llmfit_crosscheck
    assert (checks.ram_physical.status, checks.vram.status) == ("absent", "absent")


def test_a_machine_entered_by_hand_is_a_profile_file_and_an_entry_of_its_own_and_no_binding(tmp_path: Path):
    from modelroom.profile import read_profile_document

    code, lines = _run(tmp_path, ENTERED_CARD, probes=windows_probes(ids=["c" * 16]))

    assert code == 0
    path = _results(tmp_path) / "state" / "hardware" / f"{'c' * 16}.json"
    stored = read_profile_document(json.loads(path.read_text(encoding="utf-8")))
    assert (stored.display_name, stored.gpu_state, stored.vram_gib) == ("tower", "entered", 24.0)
    assert read_pointer(_pointer(tmp_path)).bindings == {}
    config = load_config(_config_file(tmp_path))
    entry = config.machines["tower"]
    assert (entry.writer, entry.profile) == (False, "c" * 16)
    assert (entry.reserve_ram_gib, entry.reserve_vram_gib) == (
        config.defaults.reserve_ram_gib,
        config.defaults.reserve_vram_gib,
    )
    assert "entered tower: 64 GiB memory, graphics card 24 GiB" in [line.strip() for line in lines]
    # Step 1's balance counts it: this machine's own entry and the one entered by hand.
    assert any(line.endswith(", 2 machines") for line in _summaries(lines))


@pytest.mark.parametrize(
    ("answers", "note"),
    [
        (ENTERED_NONE, "entered box: 64 GiB memory, no graphics card"),
        (ENTERED_UNIFIED, "entered studio: 32 GiB memory, shared memory"),
    ],
)
def test_the_note_of_a_machine_entered_by_hand_says_what_it_is(tmp_path: Path, answers, note):
    _code, lines = _run(tmp_path, answers)

    assert note in [line.strip() for line in lines]


def test_a_name_another_machine_has_gets_the_short_id(tmp_path: Path):
    """`workstation` is this machine's own entry; the one entered by hand steps aside."""
    answers = {**ENTERED_UNIFIED, "entered_name": "workstation"}
    code, _lines = _run(tmp_path, answers, probes=windows_probes(ids=["d" * 16]))

    assert code == 0
    config = load_config(_config_file(tmp_path))
    assert config.machines["workstation"].writer is True
    assert config.machines[f"workstation-{'d' * 8}"].profile == "d" * 16


def test_the_new_profile_id_never_lands_on_a_file_of_the_folder_that_does_not_read_as_a_profile(tmp_path: Path):
    """A schema-1 file and a broken one carry no `profile_id` a scan could see; their names count."""
    hardware = _results(tmp_path) / "state" / "hardware"
    hardware.mkdir(parents=True)
    legacy = hardware / f"{'a' * 16}.json"
    legacy.write_text(json.dumps(dict(EXAMPLES["HardwareSnapshot"], machine="old")), encoding="utf-8")
    broken = hardware / f"{'b' * 16}.json"
    broken.write_text("{", encoding="utf-8")
    before = {path: path.read_bytes() for path in (legacy, broken)}

    answers = {**ENTERED_UNIFIED, "write_config": True}
    code, _lines = _run(tmp_path, answers, probes=windows_probes(ids=["a" * 16, "b" * 16, "e" * 16]))

    assert code == 0
    assert {path: path.read_bytes() for path in (legacy, broken)} == before
    assert [path.stem for path in _profiles(tmp_path)] == ["a" * 16, "b" * 16, "e" * 16]
    assert load_config(_config_file(tmp_path)).machines["studio"].profile == "e" * 16


def test_a_file_name_in_another_letter_case_counts_as_taken(tmp_path: Path):
    """On a file system that ignores letter case, `AAAA….json` and `aaaa….json` are one file."""
    hardware = _results(tmp_path) / "state" / "hardware"
    hardware.mkdir(parents=True)
    upper = hardware / f"{'A' * 16}.json"
    upper.write_text("{", encoding="utf-8")

    answers = {**ENTERED_UNIFIED, "write_config": True}
    code, _lines = _run(tmp_path, answers, probes=windows_probes(ids=["a" * 16, "e" * 16]))

    assert code == 0
    assert upper.read_text(encoding="utf-8") == "{"
    assert load_config(_config_file(tmp_path)).machines["studio"].profile == "e" * 16


def test_the_new_profile_id_never_takes_the_profile_id_a_file_of_another_name_carries(tmp_path: Path):
    """`aaaa….json` carrying `profile_id` `bbbb…` names two taken ids: `hardware` reads the id, the
    scan of the guided mode sets the file aside -- one rule for both writers (Codex round 1)."""
    from modelroom.guided_entered import entered_profile

    hardware = _results(tmp_path) / "state" / "hardware"
    hardware.mkdir(parents=True)
    renamed = hardware / f"{'a' * 16}.json"
    renamed.write_text(entered_profile("old", 16.0, "unified", 0.0, "b" * 16, RUN1).model_dump_json(), encoding="utf-8")
    before = renamed.read_bytes()

    answers = {**ENTERED_UNIFIED, "write_config": True}
    code, _lines = _run(tmp_path, answers, probes=windows_probes(ids=["b" * 16, "e" * 16]))

    assert code == 0
    assert renamed.read_bytes() == before
    assert [path.stem for path in _profiles(tmp_path)] == ["a" * 16, "e" * 16]
    assert load_config(_config_file(tmp_path)).machines["studio"].profile == "e" * 16


def test_the_empty_card_of_a_folder_whose_only_machine_was_entered_by_hand_says_it_is_never_measured(tmp_path: Path):
    """A run that chose nothing still draws its card; step 4 has nothing to measure for a machine
    entered by hand, so the card says why instead of the advice (Codex round 1)."""
    _code, lines = _run(tmp_path, {**ENTERED_UNIFIED, "select": []})

    speed = _card(lines)["speed"]
    assert speed.startswith("speed     not measured")
    assert speed.endswith("a machine entered by hand is never measured"), speed


def test_a_lock_another_process_holds_stops_the_entry_before_the_profile_is_written(tmp_path: Path):
    """Every writer of a profile holds `modelroom.lock`; the machine entered by hand is no exception."""
    from modelroom.state import LockHeldError

    held = []

    class _LockingAsker(FileAsker):
        """Another process takes the lock while the shape question is on screen."""

        def select(self, key, question, choices, instruction=None):
            if key == "entered_gpu":
                held.append(acquire_lock(_results(tmp_path) / "state" / "modelroom.lock", "test", RUN1))
            return super().select(key, question, choices, instruction)

    try:
        with pytest.raises(LockHeldError):
            run_guided(
                _LockingAsker(ENTERED_UNIFIED),
                here=_results(tmp_path),
                pointer_path=_pointer(tmp_path),
                transport=build_transport(guided_transport_mapping()),
                probes=windows_probes(),
                daemon=offline_daemon(),
                now=RUN1,
                out=[].append,
            )
    finally:
        for handle in held:
            release_lock(handle)

    assert held, "the shape question was never asked"
    assert _profiles(tmp_path) == []
    assert "studio" not in load_config(_config_file(tmp_path)).machines


def test_the_machine_list_shows_each_shape_entered_by_hand_as_what_it_is(tmp_path: Path):
    from modelroom.guided_entered import entered_profile

    hardware = _results(tmp_path) / "state" / "hardware"
    for shape, ram, vram, profile_id, name in (
        ("card", 64.0, 24.0, "1" * 16, "tower"),
        ("none", 64.0, 0.0, "2" * 16, "box"),
        ("unified", 32.0, 0.0, "3" * 16, "studio"),
    ):
        profile = entered_profile(name, ram, shape, vram, profile_id, RUN1)
        atomic_write_json(hardware / f"{profile_id}.json", profile.model_dump(mode="json"))

    asker = _watched(tmp_path, {**FULL_ANSWERS, "machines": [], "write_config": True})

    groups = {choice.label: choice.disabled for choice in asker.choices["machines"] if choice.value.startswith("group:")}
    assert groups == {
        "graphics card 24.00 GiB (entered) / 64.00 GiB RAM -- 1 machine: tower": "already in this results folder",
        "no graphics card (entered) / 64.00 GiB RAM -- 1 machine: box": "already in this results folder",
        "shared memory 32.00 GiB (entered) -- 1 machine: studio": "already in this results folder",
    }


def test_a_machine_entered_by_hand_is_named_so_and_computed_on_shared_memory_in_every_view(tmp_path: Path):
    """Unified memory, 32 GiB, reserves 8 + 1: one pool of 23 GiB, both reserves taken from it."""
    code, lines = _run(tmp_path, ENTERED_UNIFIED)

    assert code == 0
    stripped = [line.strip() for line in lines]
    assert any(line.startswith("studio (entered) · context") for line in stripped)
    assert any("fit into shared memory, 23.0 GB free after both reserves" in line for line in stripped)
    card = next(line for line in stripped if line.startswith("machine") and "studio" in line)
    assert card.endswith("shared memory 32 GB, entered")
    assert "measured" not in card
    markdown = (_results(tmp_path) / "docs" / "models.md").read_text(encoding="utf-8")
    assert "studio (entered) -- ranked" in markdown
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    block = next(block for block in payload["machines"] if block["machine"] == "studio")
    gpu = [entry for entry in block["ranked"] if entry["fit"]["mode"] == "gpu"]
    assert gpu, "unified memory computes in mode gpu"
    assert {entry["fit"]["pool_gib"] for entry in gpu} == {23.0}
    assert {entry["note"]["origin"] for entry in block["ranked"] + block["too_tight"]} == {"computed"}
    own = next(block for block in payload["machines"] if block["machine"] == "workstation")
    assert own["status"] == "no_profile"


def test_a_machine_entered_by_hand_leaves_a_card_without_an_install_line_and_no_load_test(tmp_path: Path):
    code, lines = _run(tmp_path, ENTERED_CARD)

    assert code == 0
    stripped = [line.strip() for line in lines]
    assert any(line.startswith("no load test: ") for line in stripped)
    assert not any(line.startswith("install ") for line in stripped)
    card = next(line for line in stripped if line.startswith("machine") and "tower" in line)
    assert card.endswith("graphics card 24 GB, 64 GB memory, entered")


def test_a_home_binding_on_a_machine_entered_by_hand_measures_this_machine_into_a_profile_of_its_own(tmp_path: Path):
    """No clone question (the answer file has no `clone`); the entered file stays byte for byte."""
    assert _run(tmp_path, ENTERED_UNIFIED, probes=windows_probes(ids=["c" * 16]))[0] == 0
    hand = _results(tmp_path) / "state" / "hardware" / f"{'c' * 16}.json"
    before = hand.read_bytes()
    write_pointer(_pointer(tmp_path), read_pointer(_pointer(tmp_path)).with_binding(_results(tmp_path), "c" * 16))

    code, _lines = _run(tmp_path, _answers_without_folder(FULL_ANSWERS), now=RUN2)

    assert code == 0
    assert hand.read_bytes() == before
    measured = _measured_stem(tmp_path)
    assert read_pointer(_pointer(tmp_path)).binding_for(_results(tmp_path)) == measured
    config = load_config(_config_file(tmp_path))
    assert (config.machines["studio"].profile, config.machines["workstation"].profile) == ("c" * 16, measured)
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    assert [block["status"] for block in payload["machines"]] == ["ranked", "ranked"]


def _bound_by_hand_to_an_entered_profile(tmp_path: Path) -> tuple[_WatchingAsker, list[str]]:
    """A folder whose pointer binds this machine to a profile entered by hand, then one run measuring it."""
    assert _run(tmp_path, ENTERED_UNIFIED, probes=windows_probes(ids=["c" * 16]))[0] == 0
    write_pointer(_pointer(tmp_path), read_pointer(_pointer(tmp_path)).with_binding(_results(tmp_path), "c" * 16))
    lines: list[str] = []
    asker = _WatchingAsker(_answers_without_folder(FULL_ANSWERS), lines)
    run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        now=RUN2,
        out=lines.append,
    )
    return asker, [line.strip() for line in lines]


def test_a_home_binding_on_a_machine_entered_by_hand_offers_to_measure_now_and_marks_it(tmp_path: Path):
    """Nothing of this machine was measured here, so there is no `last measured` date to name."""
    asker, _lines = _bound_by_hand_to_an_entered_profile(tmp_path)

    this_machine = asker.choices["machines"][0]
    assert this_machine.value == "this-machine"
    assert this_machine.label == "this machine (measure now, the bound profile was entered by hand)"
    assert this_machine.checked is True


def test_the_first_measurement_after_a_binding_on_a_machine_entered_by_hand_is_no_second_one(tmp_path: Path):
    _asker, lines = _bound_by_hand_to_an_entered_profile(tmp_path)

    assert any(line.startswith("measured: ") for line in lines)
    assert not any(line.startswith("measured again") for line in lines)


def test_a_local_entry_on_a_machine_entered_by_hand_is_bound_to_the_new_measurement(tmp_path: Path):
    """The entry is taken over by the measurement; the entered file stays and is entered again next run."""
    from modelroom.guided_write import update_config

    assert _run(tmp_path, ENTERED_UNIFIED, probes=windows_probes(ids=["c" * 16]))[0] == 0
    hand = _results(tmp_path) / "state" / "hardware" / f"{'c' * 16}.json"
    before = hand.read_bytes()

    def only_the_local_entry(current):
        data = current.model_dump(mode="json")
        del data["machines"]["studio"]
        data["machines"]["workstation"]["profile"] = "c" * 16
        return type(current).from_dict(data)

    update_config(_config_file(tmp_path), only_the_local_entry, now=RUN1)
    answers = _answers_without_folder(FULL_ANSWERS)

    code, _lines = _run(tmp_path, answers, now=RUN2)

    assert code == 0
    assert hand.read_bytes() == before
    assert load_config(_config_file(tmp_path)).machines["workstation"].profile == _measured_stem(tmp_path)
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    assert [block["machine"] for block in payload["machines"]] == ["workstation"]

    code, _lines = _run(tmp_path, {**answers, "machines": []}, now=RUN3)

    assert code == 0
    assert hand.read_bytes() == before
    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    assert sorted(block["machine"] for block in payload["machines"]) == ["studio", "workstation"]
