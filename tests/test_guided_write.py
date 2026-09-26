"""Tests for `modelroom.guided_write`: every write of the guided mode keeps a change made meanwhile.

Each test lets a second writer change the file **between** the read a step computes its change
from and that step's write -- the one window the lock alone never closed. The step then reads the
file again, applies its own change to what it finds, writes once more and hands back what it
wrote. A second conflict in a row ends the run (CONTRACTS.md, "Configuration").
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from modelroom.catalog import load_catalog
from modelroom.config import Configuration, load_config
from modelroom.guided_contracts import SearchHit
from modelroom.guided_write import (
    Said,
    combined,
    create_config,
    read_stored,
    update_config,
    with_context,
    with_found_profiles,
    with_profile,
    with_requests,
    with_writer,
)
from modelroom.search_apply import ConfigChangedError, apply_hits, write_configuration
from modelroom.state import atomic_write_json

from fixture_support import qwen35_example_config_dict, v2_profile

NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
QWEN = "Qwen/Qwen3.5-9B"


def _config_file(tmp_path: Path, **changes) -> Path:
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "docs" / "models.md"))
    data.update({"schema_version": 2, **changes})
    path = tmp_path / "modelroom.toml"
    write_configuration(path, Configuration.from_dict(data), now=NOW, expected_text=None)
    return path


def _second_writer(path: Path, edit: Callable[[dict], None]) -> None:
    """Another run: reads the file, changes it and writes it back through the same writer."""
    stored = read_stored(path)
    data = stored.config.model_dump(mode="json")
    edit(data)
    write_configuration(path, Configuration.from_dict(data), now=NOW, expected_text=stored.text)


def _add_machine(name: str) -> Callable[[dict], None]:
    def edit(data: dict) -> None:
        data["machines"][name] = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": False}

    return edit


def _meanwhile(path: Path, change, edit: Callable[[dict], None], times: int = 1):
    """`change`, with a second writer at work after the first `times` of its calls computed.

    The change is computed first and the other run writes afterwards, so what the first attempt
    saw (the file, the profiles of the folder) is truly out of date when it comes to the write.
    """
    calls = {"count": 0}

    def interleaved(current: Configuration):
        calls["count"] += 1
        result = change(current)
        if calls["count"] <= times:
            _second_writer(path, edit)
        return result

    return interleaved, calls


# A line no configuration this package writes carries: while it stands, nothing wrote the file.
UNTOUCHED = "# nobody wrote this file since\n"


def _mark_untouched(path: Path) -> None:
    path.write_text(path.read_text(encoding="utf-8") + UNTOUCHED, encoding="utf-8")


# --- the mechanism ------------------------------------------------------------------------------


def test_the_change_is_applied_again_to_what_the_other_run_wrote(tmp_path: Path):
    path = _config_file(tmp_path)
    change, calls = _meanwhile(path, with_context(4096), _add_machine("other"))

    written = update_config(path, change, now=NOW)

    stored = load_config(path)
    assert "other" in stored.machines, "the other run's change was written over"
    assert stored.guided.context == 4096, "this run's own change was lost"
    assert calls["count"] == 2
    assert written.model_dump(mode="json") == stored.model_dump(mode="json")


def test_a_second_conflict_in_a_row_ends_the_step_and_keeps_the_other_runs_file(tmp_path: Path):
    path = _config_file(tmp_path)
    names = iter(["other", "third"])

    def add_another(data: dict) -> None:
        _add_machine(next(names))(data)

    change, calls = _meanwhile(path, with_context(4096), add_another, times=2)

    with pytest.raises(ConfigChangedError):
        update_config(path, change, now=NOW)

    stored = load_config(path)
    assert stored.guided.context is None
    assert {"other", "third"} <= set(stored.machines)
    assert calls["count"] == 2, "one retry, never a loop"


def test_a_change_with_nothing_to_do_writes_nothing(tmp_path: Path):
    path = _config_file(tmp_path, guided={"results": str(tmp_path), "context": 4096})
    _mark_untouched(path)

    written = update_config(path, with_context(4096), now=NOW)

    assert path.read_text(encoding="utf-8").endswith(UNTOUCHED)
    assert written.guided.context == 4096


def test_the_first_write_of_a_new_file_takes_over_a_file_another_run_created(tmp_path: Path):
    """Two runs create the configuration at once: the first one's families and settings stay."""
    other = _config_file(tmp_path, defaults={"reserve_ram_gib": 12.0, "reserve_vram_gib": 2.0})
    kept = other.read_bytes()
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "docs" / "models.md"))
    data.update({"schema_version": 2, "families": [], "publishers": []})

    adopted = create_config(other, Configuration.from_dict(data), now=NOW)

    assert other.read_bytes() == kept, "the initial values of a new file replaced existing ones"
    assert [family.name for family in adopted.families] == ["qwen3.5"]
    assert adopted.defaults.reserve_ram_gib == 12.0


def test_the_first_write_of_a_new_file_writes_it_when_nobody_else_did(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "docs" / "models.md"))
    data.update({"schema_version": 2})
    path = tmp_path / "modelroom.toml"

    written = create_config(path, Configuration.from_dict(data), now=NOW)

    assert load_config(path).model_dump(mode="json") == written.model_dump(mode="json")


# --- each write of the guided mode --------------------------------------------------------------


def test_this_machine_becomes_a_writer_on_top_of_the_other_runs_change(tmp_path: Path):
    path = _config_file(tmp_path, machines={})
    change, _ = _meanwhile(path, with_writer("laptop"), _add_machine("other"))

    written = update_config(path, change, now=NOW)

    assert written.machines["laptop"].writer is True
    assert "other" in written.machines


def test_a_machine_another_run_made_a_writer_meanwhile_is_not_written_again(tmp_path: Path):
    path = _config_file(tmp_path, machines={})

    def make_writer(data: dict) -> None:
        data["machines"]["laptop"] = {"reserve_ram_gib": 20.0, "reserve_vram_gib": 2.0, "writer": True}

    change, _ = _meanwhile(path, with_writer("laptop"), make_writer)
    written = update_config(path, change, now=NOW)

    assert written.machines["laptop"].reserve_ram_gib == 20.0


def test_the_profile_is_recorded_on_top_of_the_other_runs_change(tmp_path: Path):
    path = _config_file(tmp_path)
    config = load_config(path)
    said = Said()
    change, _ = _meanwhile(path, with_profile("workstation", "a" * 16, config.paths, path, said), _add_machine("other"))

    written = update_config(path, change, now=NOW)

    assert written.machines["workstation"].profile == "a" * 16
    assert "other" in written.machines
    assert said.problems == []


def test_the_checks_of_the_profile_record_hold_again_after_the_reload(tmp_path: Path):
    """The machine entry is gone in the file the second attempt reads: reported, nothing written."""
    path = _config_file(tmp_path)
    config = load_config(path)
    said = Said()

    def remove(data: dict) -> None:
        del data["machines"]["workstation"]
        data["machines"]["other"] = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}

    change, _ = _meanwhile(path, with_profile("workstation", "a" * 16, config.paths, path, said), remove)
    written = update_config(path, change, now=NOW)

    assert "workstation" not in written.machines
    assert said.problems == [f"{path}: the machine 'workstation' is gone from the configuration; run again"]


def test_the_context_is_kept_on_top_of_the_other_runs_change(tmp_path: Path):
    path = _config_file(tmp_path)
    change, _ = _meanwhile(path, with_context(16384), _add_machine("other"))

    written = update_config(path, change, now=NOW)

    assert (written.guided.context, "other" in written.machines) == (16384, True)



# --- the head count and the requests: written together, compared with the file ---------------------


def _guided(path: Path) -> tuple:
    guided = load_config(path).guided
    return guided.users, guided.requests, guided.requests_origin


def test_the_head_count_and_its_requests_are_written_together(tmp_path: Path):
    path = _config_file(tmp_path)

    written = update_config(path, with_requests(25, 3, "from_users"), now=NOW)

    assert _guided(path) == (25, 3, "from_users")
    assert (written.guided.users, written.guided.requests, written.guided.requests_origin) == (25, 3, "from_users")


def test_requests_named_outright_are_written_without_a_head_count(tmp_path: Path):
    path = _config_file(tmp_path)

    update_config(path, with_requests(None, 12, "entered"), now=NOW)

    assert _guided(path) == (None, 12, "entered")
    assert "users" not in path.read_text(encoding="utf-8")


def test_requests_named_outright_keep_the_head_count_the_file_holds_when_they_are_written(tmp_path: Path):
    """The head count of `entered` is the file's own, read in the change -- also on a retry."""
    path = _config_file(tmp_path)
    update_config(path, with_requests(25, 3, "from_users"), now=NOW)
    edit = lambda data: data["guided"].update(users=50, requests=5, requests_origin="from_users")  # noqa: E731
    change, calls = _meanwhile(path, with_requests(25, 12, "entered"), edit)

    update_config(path, change, now=NOW)

    assert _guided(path) == (50, 12, "entered")
    assert calls["count"] == 2

def test_requests_the_file_already_holds_write_nothing(tmp_path: Path):
    path = _config_file(tmp_path)
    update_config(path, with_requests(25, 3, "from_users"), now=NOW)
    _mark_untouched(path)

    written = update_config(path, with_requests(25, 3, "from_users"), now=NOW)

    assert path.read_text(encoding="utf-8").endswith(UNTOUCHED)
    assert written.guided.users == 25


def test_the_requests_are_compared_with_the_file_and_not_with_the_runs_own_copy(tmp_path: Path):
    """The run read 25; another run wrote 5 since. The answer is 25, so 25 is written again."""
    path = _config_file(tmp_path)
    update_config(path, with_requests(25, 3, "from_users"), now=NOW)
    _second_writer(path, lambda data: data["guided"].update(users=5, requests=1, requests_origin="from_users"))

    update_config(path, with_requests(25, 3, "from_users"), now=NOW)

    assert _guided(path) == (25, 3, "from_users")


def test_the_requests_are_kept_on_top_of_the_other_runs_change(tmp_path: Path):
    path = _config_file(tmp_path)
    change, calls = _meanwhile(path, with_requests(25, 3, "from_users"), _add_machine("other"))

    written = update_config(path, change, now=NOW)

    assert "other" in load_config(path).machines, "the other run's change was written over"
    assert _guided(path) == (25, 3, "from_users")
    assert calls["count"] == 2
    assert written.model_dump(mode="json") == load_config(path).model_dump(mode="json")

def _place(tmp_path: Path, profile_id: str, display_name: str) -> None:
    hardware = tmp_path / "state" / "hardware"
    hardware.mkdir(parents=True, exist_ok=True)
    profile = v2_profile(profile_id=profile_id, display_name=display_name)
    atomic_write_json(hardware / f"{profile.profile_id}.json", profile.model_dump(mode="json"))


def test_the_found_profiles_are_worked_out_again_on_the_new_file(tmp_path: Path):
    """A name another run took meanwhile is not written over: the whole matching runs again."""
    path = _config_file(tmp_path)
    _place(tmp_path, "5" * 16, "server")
    said = Said()
    change, _ = _meanwhile(path, with_found_profiles(said), _add_machine("server"))

    written = update_config(path, change, now=NOW)

    assert written.machines["server"].profile is None, "the other run's machine was written over"
    assert written.machines[f"server-{'5' * 8}"].profile == "5" * 16
    assert said.notes == [f"machine 'server-{'5' * 8}' added for the profile {'5' * 16} found in this folder"]


def test_the_second_attempt_scans_again_and_takes_the_reserves_the_file_holds_then(tmp_path: Path):
    """Nothing of the first attempt is reused: a profile placed meanwhile and new defaults count."""
    path = _config_file(tmp_path)
    _place(tmp_path, "5" * 16, "server")
    said = Said()

    def import_and_raise_the_defaults(data: dict) -> None:
        _place(tmp_path, "6" * 16, "laptop")
        data["defaults"] = {"reserve_ram_gib": 24.0, "reserve_vram_gib": 3.0}

    change, _ = _meanwhile(path, with_found_profiles(said), import_and_raise_the_defaults)
    written = update_config(path, change, now=NOW)

    entered = {machine.profile: machine for machine in written.machines.values() if machine.profile}
    assert set(entered) == {"5" * 16, "6" * 16}
    assert {(m.reserve_ram_gib, m.reserve_vram_gib) for m in entered.values()} == {(24.0, 3.0)}
    assert len(said.notes) == 2


def test_a_profile_the_other_run_already_entered_is_not_entered_twice(tmp_path: Path):
    path = _config_file(tmp_path)
    _place(tmp_path, "5" * 16, "server")
    said = Said()

    def enter(data: dict) -> None:
        data["machines"]["srv"] = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": False, "profile": "5" * 16}

    entered, calls = _meanwhile(path, with_found_profiles(said), enter)

    def change(current: Configuration):
        result = entered(current)
        if calls["count"] == 1:
            _mark_untouched(path)  # after the other run's write: a rewrite would drop this line
        return result

    written = update_config(path, change, now=NOW)

    assert path.read_text(encoding="utf-8").endswith(UNTOUCHED), "nothing left to add, and still written"
    assert [name for name, machine in written.machines.items() if machine.profile == "5" * 16] == ["srv"]
    assert said.notes == []
    assert calls["count"] == 2


def _every_name_of(tmp_path: Path, display_name: str, profile_id: str) -> dict:
    """Machine entries that take every name `machine_name_for` would try for that profile."""
    taken = (display_name, f"{display_name}-{profile_id[:8]}", f"{display_name}-{profile_id}")
    return {name: {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": False} for name in taken}


def test_a_problem_the_other_run_resolved_meanwhile_is_not_reported_after_the_retry(tmp_path: Path):
    """What the first attempt had to say is cleared: only the attempt that counted reports."""
    path = _config_file(tmp_path, machines=_every_name_of(tmp_path, "server", "5" * 16))
    _place(tmp_path, "5" * 16, "server")
    _place(tmp_path, "6" * 16, "laptop")
    said = Said()

    def free_the_name(data: dict) -> None:
        del data["machines"][f"server-{'5' * 16}"]

    change, calls = _meanwhile(path, with_found_profiles(said), free_the_name)
    written = update_config(path, change, now=NOW)

    assert calls["count"] == 2
    assert said.problems == [], "the first attempt's problem was reported although the retry had none"
    assert written.machines[f"server-{'5' * 16}"].profile == "5" * 16
    assert [note.split("'")[1] for note in said.notes] == [f"server-{'5' * 16}", "laptop"]


def test_a_problem_that_stays_is_reported_once_after_the_retry(tmp_path: Path):
    path = _config_file(tmp_path, machines=_every_name_of(tmp_path, "server", "5" * 16))
    _place(tmp_path, "5" * 16, "server")
    _place(tmp_path, "6" * 16, "laptop")
    said = Said()

    change, calls = _meanwhile(path, with_found_profiles(said), _add_machine("other"))
    written = update_config(path, change, now=NOW)

    assert calls["count"] == 2
    assert len(said.problems) == 1, "the same problem was reported once per attempt"
    assert said.problems[0].startswith(f"the profile {'5' * 16} stays without a machine entry: ")
    assert written.machines["laptop"].profile == "6" * 16
    assert [note.split("'")[1] for note in said.notes] == ["laptop"]


def test_hits_without_an_ollama_name_leave_the_configured_pair_after_the_reload(tmp_path: Path):
    """The search writes its choice again on top of a pair another run configured meanwhile."""
    path = _config_file(tmp_path)
    hit = SearchHit(
        repo="unsloth/Qwen3.5-9B-GGUF",
        publisher_status="listed packager",
        base_model=[QWEN],
        base_model_relation="quantized",
        resolved=True,
        resolved_base_model=QWEN,
    )

    def pair(data: dict) -> None:
        entry = data["families"][0]["base_models"][0]
        entry["ollama_base"], entry["ollama_tag"] = "my-qwen", "9b"

    change, _ = _meanwhile(path, lambda current: apply_hits(current, [hit], catalog=load_catalog()), pair)
    written = update_config(path, change, now=NOW)

    entry = written.families[0].base_models[0]
    assert (entry.ollama_base, entry.ollama_tag) == ("my-qwen", "9b")
    assert "unsloth/Qwen3.5-9B-GGUF" in entry.repos


def test_changes_combined_are_one_write_and_nothing_when_none_has_anything_to_do(tmp_path: Path):
    path = _config_file(tmp_path)

    written = update_config(path, combined(with_requests(25, 3, "from_users"), with_context(4096)), now=NOW)

    assert (_guided(path), load_config(path).guided.context) == ((25, 3, "from_users"), 4096)
    assert written.guided.context == 4096
    _mark_untouched(path)
    update_config(path, combined(with_requests(25, 3, "from_users"), with_context(4096)), now=NOW)
    assert path.read_text(encoding="utf-8").endswith(UNTOUCHED)


# --- the entry of a machine entered by hand ------------------------------------------------------


def _with_entered_machine(profile_id: str, display_name: str, said: Said):
    from modelroom.guided_write import with_entered_machine

    return with_entered_machine(profile_id, display_name, said)


def test_the_machine_entered_by_hand_gets_its_entry_on_top_of_the_other_runs_change(tmp_path: Path):
    path = _config_file(tmp_path, defaults={"reserve_ram_gib": 6.0, "reserve_vram_gib": 2.0})
    _place(tmp_path, "7" * 16, "studio")
    said = Said()
    change, calls = _meanwhile(path, _with_entered_machine("7" * 16, "studio", said), _add_machine("other"))

    written = update_config(path, change, now=NOW)

    assert calls["count"] == 2
    assert "other" in written.machines, "the other run's machine was written over"
    entry = written.machines["studio"]
    assert (entry.writer, entry.profile, entry.reserve_ram_gib, entry.reserve_vram_gib) == (False, "7" * 16, 6.0, 2.0)
    assert said.problems == []
    assert load_config(path).machines["studio"].profile == "7" * 16


def test_a_name_another_run_took_meanwhile_is_not_written_over_by_the_entry(tmp_path: Path):
    path = _config_file(tmp_path)
    _place(tmp_path, "7" * 16, "studio")
    said = Said()
    change, _ = _meanwhile(path, _with_entered_machine("7" * 16, "studio", said), _add_machine("studio"))

    written = update_config(path, change, now=NOW)

    assert written.machines["studio"].profile is None
    assert written.machines[f"studio-{'7' * 8}"].profile == "7" * 16


def test_an_entry_another_run_made_for_the_same_profile_is_not_made_twice(tmp_path: Path):
    path = _config_file(tmp_path)
    _place(tmp_path, "7" * 16, "studio")
    said = Said()

    def enter(data: dict) -> None:
        data["machines"]["mine"] = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": False, "profile": "7" * 16}

    change, _ = _meanwhile(path, _with_entered_machine("7" * 16, "studio", said), enter)
    written = update_config(path, change, now=NOW)

    assert [name for name, machine in written.machines.items() if machine.profile == "7" * 16] == ["mine"]
    assert said.problems == []


def test_a_machine_entered_by_hand_without_a_free_name_stays_without_an_entry_and_says_so(tmp_path: Path):
    path = _config_file(tmp_path, machines=_every_name_of(tmp_path, "studio", "7" * 16))
    _place(tmp_path, "7" * 16, "studio")
    _mark_untouched(path)
    said = Said()

    update_config(path, _with_entered_machine("7" * 16, "studio", said), now=NOW)

    assert path.read_text(encoding="utf-8").endswith(UNTOUCHED)
    assert len(said.problems) == 1
    assert said.problems[0].startswith(f"the profile {'7' * 16} stays without a machine entry: ")


def test_a_state_folder_that_changed_meanwhile_leaves_the_entry_unwritten_and_says_so(tmp_path: Path):
    """The profile was written into the folder `[paths]` named before; the file names another now."""
    path = _config_file(tmp_path)
    _place(tmp_path, "7" * 16, "studio")
    said = Said()

    def move_the_state(data: dict) -> None:
        data["paths"]["state"] = str(tmp_path / "elsewhere")

    change, _ = _meanwhile(path, _with_entered_machine("7" * 16, "studio", said), move_the_state)
    written = update_config(path, change, now=NOW)

    assert "studio" not in written.machines
    assert len(said.problems) == 1
    assert "[paths] changed" in said.problems[0]


def test_a_state_folder_that_changed_is_reported_even_when_another_run_entered_the_profile(tmp_path: Path):
    """The other run names the profile **and** moves the state folder: the entry now points nowhere."""
    path = _config_file(tmp_path)
    _place(tmp_path, "7" * 16, "studio")
    profile_file = tmp_path / "state" / "hardware" / f"{'7' * 16}.json"
    before = profile_file.read_bytes()
    said = Said()

    def enter_and_move(data: dict) -> None:
        data["machines"]["mine"] = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": False, "profile": "7" * 16}
        data["paths"]["state"] = str(tmp_path / "elsewhere")

    change, _ = _meanwhile(path, _with_entered_machine("7" * 16, "studio", said), enter_and_move)
    update_config(path, change, now=NOW)

    assert len(said.problems) == 1
    assert "[paths] changed" in said.problems[0]
    assert profile_file.read_bytes() == before
