"""Step 1 of the guided mode, second half: the machine list, and which machines this run chose.

Moved out of `modelroom/guided.py` (AP9-UX, size guard): `_hardware_class`, `_gib`,
`_profile_choices`, `_machine_choices` and `_measured_here`. This module never imports `guided`;
the machine's key (`guided._machine_key`) stays there and is not printed by this module (FR-N2).

The list, from the top (CONTRACTS.md, "Guided mode", step 1; AP9-UX C and C1):

1. the bound profile of this machine, where it was measured here (`profile:<profile_id>`);
2. `this machine (<name>, measure again)` with row 1, else `this machine (measure now)` (`this-machine`);
   `<name>` is the `display_name` of the profile (FR-N2), never the configuration key;
3. every other profile of the folder a machine of the configuration names, by class, then label;
4. `import a profile file`; 5. the shared folder (`share`, `modelroom/guided_share.py`, FR-S) --
   always there, its label carried in from outside since this module never touches `remote.json` or
   the environment; 6. `enter a machine by hand`;
7. grayed out: profiles no machine names, then `legacy` files, then files that do not read.

Nothing is marked in advance: `Enter` alone takes the first row -- the profile this machine already
has, where there is one, else a first measurement.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .config import Configuration
from .dialog import Choice
from .guided_entered import entered_class
from .guided_remote import SSH_ROW
from .importer import ProfileScan, _label
from .intro import card_names, class_words, graphics_words
from .neutral import is_neutral
from .profile import HardwareProfile
from .screen import THIS_MACHINE, this_machine_words

OVER_SSH = "  over SSH"

_UNKNOWN = "unknown"
PROFILE_PREFIX = "profile:"
NO_MACHINE_NAMES = "no machine of the configuration names this profile"
# The note in front of the list (71 characters): the way to a machine this run does not run on.
ANOTHER_COMPUTER = "another computer: pick its profile below, or import the file made there"
_NO_ALIASES: Mapping[str, str] = {}


def _hardware_class(profile: HardwareProfile) -> str:
    """What makes two machines the same to the user: the class, the cards, their memory, the system memory."""
    if profile.ram_physical_source == "entered":
        return entered_class(profile)
    if profile.gpus:
        gpu = card_names(profile)
    else:  # a machine whose cards were named but none counted says which (AP9-GPU G8)
        gpu = graphics_words(profile) if profile.gpu_state == "none" and profile.adapters else profile.gpu_name or profile.gpu_state
    return f"{class_words(profile)}{gpu} {_gib(profile.vram_gib)} VRAM / {_gib(profile.ram_physical_gib)} RAM"


def _gib(value: float | None) -> str:
    return _UNKNOWN if value is None else f"{value:.2f} GiB"


def _measured_here(bound: HardwareProfile | None) -> bool:
    """Whether this machine was measured in this folder: bound, and not to a profile entered by hand.

    A binding to a profile entered by hand comes about only by hand; the list and the note of the
    first measurement then say `measure now` and `measured`, never `again` (decided 2026-09-26).
    """
    return bound is not None and bound.ram_physical_source != "entered"


def _class_word(profile: HardwareProfile | None) -> str:
    """The machine class of a profile measured here, or nothing where it is unknown."""
    return "" if profile is None or profile.machine_class == _UNKNOWN else profile.machine_class


def _class_detail(profile: HardwareProfile) -> str:
    """The class for row 1, unless the name says it already: `laptop-<8 hex>` of a laptop (FR-N2).

    An alias, or a neutral name whose prefix is another class than the profile's now, keeps the detail.
    """
    word = _class_word(profile)
    return "" if is_neutral(profile.display_name) and profile.display_name.partition("-")[0] == word else word


def _profile_choices(
    scan: ProfileScan, configured: set[str], skip: str | None = None, ssh_aliases: Mapping[str, str] = _NO_ALIASES
) -> tuple[list[Choice], list[Choice]]:
    """Every profile of the folder as a row of its own: the ones a machine names, and the grayed out.

    A profile a `[machines.<name>]` names is a choice (`profile:<profile_id>`, AP9-UX C1): its label as
    the import forms it, with the short id where two carry one name, and its hardware. One no machine
    names stays grayed out with that reason; so do `legacy` files and files that do not read.
    `skip` is the bound profile of row 1, which stands there and not here. `ssh_aliases`
    (`profile_id -> alias`, FR-B1, B3) marks a profile read over SSH with `OVER_SSH` behind its label
    -- the label itself is already the alias (`neutral.display_name_for`), so this only says *how*
    it was read.
    """
    rows = sorted(
        ((_hardware_class(profile), _label(profile, scan), profile) for key, profile in scan.profiles.items() if key != skip),
        key=lambda row: (row[0], row[1]),
    )
    chosen, blocked = [], []
    for hardware, label, profile in rows:
        over_ssh = OVER_SSH if profile.profile_id in ssh_aliases else ""
        text = f"{label}{over_ssh}  {hardware}"
        if profile.profile_id in configured:
            chosen.append(Choice(f"{PROFILE_PREFIX}{profile.profile_id}", text))
        else:
            blocked.append(Choice(f"{PROFILE_PREFIX}{profile.profile_id}", text, disabled=NO_MACHINE_NAMES))
    for path in sorted(scan.legacy):
        blocked.append(Choice(f"legacy:{path.name}", f"{path.name} (legacy)", disabled="run `modelroom migrate`"))
    for path, reason in sorted(scan.unreadable):
        blocked.append(Choice(f"broken:{path.name}", f"{path.name}", disabled=f"does not read: {reason}"))
    return chosen, blocked


def _machine_choices(
    scan: ProfileScan,
    bound: HardwareProfile | None,
    configured: Iterable[str],
    share_label: str,
    ssh_aliases: Mapping[str, str] = _NO_ALIASES,
) -> list[Choice]:
    """The machine list of step 1, rows 1 to 8 (module docstring); nothing is marked.

    `bound` is the profile this folder binds this machine to, `configured` every `profile` a machine
    of the configuration names. This machine is named by the `display_name` of that profile -- the
    name the result shows (FR-N2); its key in `[machines.<name>]` stays the run's internal token and
    is not printed, and without a profile of this machine there is no name yet. A binding
    to a profile entered by hand is no row 1: nothing of this machine was measured here, so its row
    says `measure now` and the profile stands among the others (decided 2026-09-26, Review r3-N3).
    `share_label` is row 5's own text (FR-S, `guided_share.machine_list_row`): always there, whatever
    it says -- the shared folder is meant to be found in this list. `ssh_aliases` (FR-B1, B3) is
    every profile this user already read over SSH (`remote_store.aliases`); the `ssh` row itself
    (`a computer over SSH`) always stands, right after `import`.
    """
    here = bound if _measured_here(bound) else None
    shown = here.display_name if here is not None else None  # FR-N2: the name the result shows, not the key
    rows = []
    if here is not None:
        measured = f"as measured {here.recorded_at.date().isoformat()}"
        rows.append(Choice(f"{PROFILE_PREFIX}{here.profile_id}", this_machine_words(shown, _class_detail(here), measured)))
    rows.append(Choice("this-machine", this_machine_words(shown, "measure again" if here is not None else "measure now")))
    chosen, blocked = _profile_choices(scan, set(configured), here.profile_id if here is not None else None, ssh_aliases)
    tail = [
        Choice("import", "import a profile file"),
        Choice("ssh", SSH_ROW),
        Choice("share", share_label),
        Choice("enter", "enter a machine by hand"),
    ]
    return [*rows, *chosen, *tail, *blocked]


def machine_answer_words(scan: ProfileScan, bound: HardwareProfile | None) -> dict[str, str]:
    """The words of the answer line for each row that has words of its own (`screen.machines_words`)."""
    here = bound if _measured_here(bound) else None
    # The answer line names this machine by its class alone: its neutral key says nothing to a reader
    # and would push the line over the width of the screen (FR-N); the list shows the profile's name.
    own = f"{THIS_MACHINE} ({_class_word(here)})" if _class_word(here) else THIS_MACHINE
    words = {"this-machine": own}
    for key, profile in scan.profiles.items():
        words[f"{PROFILE_PREFIX}{key}"] = _label(profile, scan)
    if here is not None:
        words[f"{PROFILE_PREFIX}{here.profile_id}"] = own
    return words


def chosen_machines(config: Configuration, picked: Sequence[str], name: str, arrived: Iterable[str] = ()) -> tuple[str, ...]:
    """The `[machines.<name>]` this run chose: this machine where it was measured, every machine whose
    profile was marked, imported or entered in this run (`arrived`). Nothing is measured or written here."""
    names = [name] if "this-machine" in picked and name in config.machines else []
    ids = {value[len(PROFILE_PREFIX) :] for value in picked if value.startswith(PROFILE_PREFIX)} | set(arrived)
    names += [machine for machine, entry in config.machines.items() if entry.profile in ids and machine not in names]
    return tuple(names)


__all__ = [
    "ANOTHER_COMPUTER",
    "NO_MACHINE_NAMES",
    "OVER_SSH",
    "PROFILE_PREFIX",
    "chosen_machines",
    "machine_answer_words",
]
