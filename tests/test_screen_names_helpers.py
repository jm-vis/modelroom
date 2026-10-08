"""AP9-FR-N3: the three helpers of `modelroom/screen_names.py`, on their own.

They live in a file of their own because the behaviour tests (`tests/test_screen_names.py`) must be
collectable against the unchanged production code of the basis: a behaviour test fails there on the key
it still prints, not on a module that does not exist yet. Only this file imports `modelroom.screen_names`.
"""

from __future__ import annotations

from modelroom.screen_names import profile_name, shown_as, shown_names

from test_screen_names import HERE_KEY, HERE_NAME, OTHER_KEY, OTHER_NAME, _document


def test_the_names_of_a_document_are_the_display_names_of_the_machines_with_a_profile():
    names = shown_names(_document())

    assert names == {HERE_KEY: HERE_NAME, OTHER_KEY: OTHER_NAME}
    assert shown_as(names, HERE_KEY) == HERE_NAME


def test_a_machine_without_a_resolved_profile_keeps_its_key():
    names = shown_names(_document(other_profile=False))

    assert names == {HERE_KEY: HERE_NAME}
    assert shown_as(names, OTHER_KEY) == OTHER_KEY  # no name made up for it


def test_a_profile_with_an_empty_display_name_is_called_by_its_neutral_name():
    profile = _document().machines[0].profile.model_copy(update={"display_name": "", "machine_class": "laptop"})

    assert profile_name(profile) == f"laptop-{profile.profile_id[:8]}"
