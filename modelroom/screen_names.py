"""The name a machine has on the screen of the guided run (AP9-FR-N3, after AP9-FR-N2).

A machine has two names in a run. Its key (`[machines.<key>]`, `neutral.random_key`) is the internal
token: the lookups, the fit, the selection and the install commands go by it, and the files carry it.
The name the screen says is the one its profile carries: `display_name`, or `neutral.neutral_name`
where the profile has none -- the name the card, the table head and step 1 already say. A line of the
guided screen that names a machine says that name; a machine whose profile cannot be resolved keeps
its key, because the screen has no other name for it.

Nothing here reads a file or the configuration: the callers hand in the profile they already hold
(`guided_context.Checked.profile`, `document.MachineRanking.profile`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from .neutral import neutral_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import RenderDocument
    from .profile import HardwareProfile


def profile_name(profile: "HardwareProfile") -> str:
    """The name the screen gives the machine of `profile`: its `display_name`, else its neutral name."""
    return profile.display_name or neutral_name(profile.profile_id, profile.machine_class)


def shown_names(document: "RenderDocument") -> dict[str, str]:
    """`{key: name}` for every machine of the document whose profile is resolved; the others are left out
    on purpose, so that `shown.get(key, key)` keeps the key for them (`views.RunView.shown`)."""
    return {block.machine: profile_name(block.profile) for block in document.machines if block.profile is not None}


def shown_as(shown: Mapping[str, str], key: str) -> str:
    """The name of machine `key` in `shown`, or the key where no profile gave it one."""
    return shown.get(key, key)


__all__ = ["profile_name", "shown_as", "shown_names"]
