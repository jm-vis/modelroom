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
from modelroom.dialog import AnswerMissingError, Canceled, FileAsker
from modelroom.examples import EXAMPLES
from modelroom.guided import LOAD_TEST_LINE, GuidedError, _hit_choices, _hit_label, run_guided
from modelroom.guided_contracts import SearchHit
from modelroom.profile import HardwareProfile
from modelroom.state import acquire_lock, atomic_write_json, release_lock

from fixture_support import (
    build_transport,
    guided_transport_mapping,
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

FULL_ANSWERS = {
    "results": "here",
    "machines": ["this-machine"],
    "search": "qwen",
    "filter_owners": True,
    "select": [UNSLOTH, QWEN_GGUF],
    "context": "8192",
}


def _pointer(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".modelroom" / "guided.json"


def _run(tmp_path: Path, answers: dict | None = None, *, here: Path | None = None, now=RUN1, **kwargs) -> tuple[int, list[str]]:
    """One guided run; returns its exit code and every line it printed."""
    lines: list[str] = []
    asker = FileAsker(FULL_ANSWERS if answers is None else answers)
    code = run_guided(
        asker,
        here=here if here is not None else tmp_path / "results",
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=kwargs.pop("probes", None) or windows_probes(),
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


def test_the_same_machine_answer_measures_nothing_and_says_what_to_do(tmp_path: Path):
    assert _run(tmp_path)[0] == 0
    _profiles(tmp_path)[0].unlink()

    answers = {key: value for key, value in FULL_ANSWERS.items() if key != "results"} | {"clone": "same"}
    code, lines = _run(tmp_path, answers, now=RUN2)

    # Exit 1, not 0: the document is written, but a step did not do what it was asked.
    assert code == 1
    assert _profiles(tmp_path) == []
    assert any("nothing measured for this machine" in line for line in lines)
    assert any("1 step(s) did not finish" in line for line in lines)


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


def test_the_search_summary_and_the_unresolved_hits_are_printed(tmp_path: Path):
    _code, lines = _run(tmp_path)

    assert any(line.startswith("8 repositories, 3 resolved, 5 unresolved, budget ") for line in lines)
    assert any("unresolved community-user/Qwen3.5-9B-Roleplay-GGUF: derivative" in line for line in lines)


def test_the_chosen_repositories_reach_the_configuration(tmp_path: Path):
    assert _run(tmp_path)[0] == 0

    config = load_config(_config_file(tmp_path))
    base_model = config.families[0].base_models[0]
    assert base_model.hf_repo == "Qwen/Qwen3.5-9B"
    assert sorted(base_model.repos) == sorted([QWEN_GGUF, UNSLOTH])
    assert (base_model.ollama_base, base_model.ollama_tag) == ("qwen3.5", "9b")
    assert config.publishers == ["Qwen"]


def _hit(repo: str, publisher_status: str) -> SearchHit:
    return SearchHit(
        repo=repo,
        publisher_status=publisher_status,
        base_model=["Qwen/Qwen3.5-9B"],
        base_model_relation="quantized",
        resolved=True,
        resolved_base_model="Qwen/Qwen3.5-9B",
    )


def test_the_owner_filter_grays_out_every_other_account():
    """No resolved hit of the pinned answer is `other`, so the rule itself is checked here."""
    hits = [_hit(UNSLOTH, "listed packager"), _hit("community-user/Qwen3.5-9B-GGUF", "other")]

    on = {choice.value: choice.disabled for choice in _hit_choices(hits, True)}
    off = {choice.value: choice.disabled for choice in _hit_choices(hits, False)}

    assert on["community-user/Qwen3.5-9B-GGUF"] == "not a publisher or a listed packager"
    assert on[UNSLOTH] is None
    assert set(off.values()) == {None}


def test_every_selectable_hit_shows_the_seven_facts():
    label = _hit_label(_hit(UNSLOTH, "listed packager"))

    assert label.startswith(UNSLOTH)
    assert "listed packager" in label
    assert "repo created unknown" in label
    assert "unknown" in label  # size and license, both absent from this hit
    assert "ollama: none known" in label


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
    assert any("context 4096 (entered)" in line for line in lines)


def test_a_context_that_is_not_a_number_ends_the_run(tmp_path: Path):
    with pytest.raises(GuidedError, match="whole number"):
        _run(tmp_path, {**FULL_ANSWERS, "context": "lots"})


@pytest.mark.parametrize("answer", ["0", "-1", str(2**31)])
def test_a_context_the_scenario_cannot_take_ends_the_run_without_a_traceback(tmp_path: Path, answer: str):
    """`Scenario.context_requested` is bounded; an answer outside it is an input error, not a crash."""
    with pytest.raises(GuidedError):
        _run(tmp_path, {**FULL_ANSWERS, "context": answer})


# --- the last steps: fetch, the load test's place, render ---------------------------------------------


def test_the_run_ends_with_both_views_and_the_load_test_line(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    assert LOAD_TEST_LINE in lines
    assert (_results(tmp_path) / "docs" / "models.md").is_file()
    assert (_results(tmp_path) / "docs" / "models.json").is_file()
    assert any("Ranking rule: fit class" in line for line in lines)
    assert any("Ranking: workstation" in line for line in lines)


def test_qwen35_is_not_covered_by_fit_v1_and_says_so(tmp_path: Path):
    assert _run(tmp_path)[0] == 0

    payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
    block = payload["machines"][0]
    assert block["status"] == "ranked"
    assert block["ranked"] == []
    reasons = {entry["reason"] for entry in block["not_covered"]}
    assert reasons == {"architecture not covered by v1"}


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
