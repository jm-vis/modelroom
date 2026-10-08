"""AP9-FR-N3, finding 5: the name of a computer read over SSH, and where its key may differ from it.

A computer read over SSH is called by its alias, the user's own word. The screen says the `display_name` of its
profile for every machine (FR-N3), and `room_head` and `pull_remote_row` name an SSH computer by the alias they
are handed. These tests hold, on real flows (`run_guided` over `FakeSsh`, as `tests/test_fr_nachlese.py`), where
key, alias and `display_name` agree and where they do not:

* the computer this run is connected to (`run.ssh_active`) is found in the document by its alias *as key*
  (`guided_remote.active_room`, `pull_other_step`): key = alias there, and the sweep of the import makes its
  `display_name` the alias too -- after a first bind, a clone answered `new`, and a quiet next run;
* a stored SSH computer that is not connected in a run can have another key: a profile found in the folder
  without a row gets a neutral key (`guided_write.with_found_profiles`, no alias is passed to `key_for`), while
  its profile keeps the alias as name. The screen says the name for it (`room_head.head_pieces`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixture_ssh import FakeSsh
from modelroom.config import load_config
from modelroom.importer import scan_profiles
from modelroom.neutral import display_name_for, is_neutral
from modelroom.remote_store import aliases as remote_aliases
from modelroom.remote_store import remote_path_for

from test_fr_nachlese import NEW_ID, OLD_ID, SSH_BASE, _again, _bind_first, _idle, _run
from test_guided import RUN3, _pointer, _results


@pytest.fixture(autouse=True)
def _results_folder(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()


def _folder(tmp_path: Path):
    """The configuration, the profiles and the aliases of the results folder as the next run would read them."""
    config = load_config(_results(tmp_path) / "modelroom.toml")
    return config, scan_profiles(config.paths.hardware_dir).profiles, remote_aliases(remote_path_for(_pointer(tmp_path)))


def _agree(tmp_path: Path) -> None:
    """Key = alias = `display_name` for every profile that has an alias and a row; `display_name_for` agrees."""
    config, profiles, aliases = _folder(tmp_path)
    assert aliases
    for profile_id, alias in aliases.items():
        if profile_id not in profiles:
            continue  # an alias that points at a profile with no file yet (a failed import): nothing is shown for it
        profile = profiles[profile_id]
        keys = [key for key, machine in config.machines.items() if machine.profile == profile_id]
        assert keys == [alias], (profile_id, keys)
        assert profile.display_name == alias
        assert display_name_for(profile_id, profile.machine_class, profile.display_name, alias) == alias
    named = {profile.display_name for profile_id, profile in profiles.items() if profile_id not in aliases}
    assert not named & set(aliases.values())  # no other profile carries an alias as its name


def test_the_computer_bound_over_ssh_has_its_alias_as_key_and_as_name(tmp_path):
    assert _bind_first(tmp_path) == OLD_ID

    _agree(tmp_path)


def test_the_alias_stays_key_and_name_when_it_is_bound_to_a_new_profile(tmp_path):
    """A clone answered `new`: the alias moves to a fresh profile; the old row goes neutral, its profile loses the alias."""
    first = _bind_first(tmp_path, "a" * 32)
    _again(tmp_path, first, "b" * 32, {"ssh_clone": "new"}, ids=[NEW_ID])

    _agree(tmp_path)
    config, profiles, aliases = _folder(tmp_path)
    assert aliases == {NEW_ID: "gpu1"} and config.machines["gpu1"].profile == NEW_ID
    old = [key for key, machine in config.machines.items() if machine.profile == OLD_ID]
    assert len(old) == 1 and is_neutral(old[0]) and is_neutral(profiles[OLD_ID].display_name)  # the old one is called neutrally now


def test_the_alias_stays_key_and_name_on_the_quiet_next_run(tmp_path):
    first = _bind_first(tmp_path, "a" * 32)
    _again(tmp_path, first, "b" * 32, {"ssh_clone": "new"}, ids=[NEW_ID])

    _again(tmp_path, NEW_ID, "b" * 32, now=RUN3)

    _agree(tmp_path)


def test_a_profile_found_in_the_folder_without_a_row_gets_a_neutral_key_while_its_name_stays_the_alias(tmp_path):
    """Where key and name part: the row of the alias is gone from the configuration, the next run adds one for the
    profile it finds (`with_found_profiles`) -- no alias is passed there, so the key is a neutral one; the profile
    keeps `gpu1` as its name, which is what the screen says (the `room` line of `tests/test_screen_names.py`)."""
    assert _bind_first(tmp_path) == OLD_ID
    config_file = _results(tmp_path) / "modelroom.toml"
    kept, skipping = [], False
    for line in config_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            skipping = line.strip() == "[machines.gpu1]"
        if not skipping:
            kept.append(line)
    config_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert "gpu1" not in load_config(config_file).machines

    _run(tmp_path, {"machines": [f"profile:{OLD_ID}"]}, FakeSsh(context=_idle("a" * 32)), now=RUN3, machines=(f"profile:{OLD_ID}",))

    config, profiles, aliases = _folder(tmp_path)
    keys = [key for key, machine in config.machines.items() if machine.profile == OLD_ID]
    assert aliases == {OLD_ID: "gpu1"} and profiles[OLD_ID].display_name == "gpu1"
    assert len(keys) == 1 and is_neutral(keys[0]) and keys[0] != "gpu1"  # key and alias part here, name and alias do not
