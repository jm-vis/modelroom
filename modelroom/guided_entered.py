"""Step 1's third entry of the machine list: a machine entered by hand (decided 2026-09-26).

Four questions (`QUESTIONS`) size a machine from its data sheet: its name, its memory, what runs
the model -- a graphics card with its own memory, no graphics card, or unified memory -- and, for
a graphics card, its memory. That is exactly one of the three shapes a schema-3 profile allows for
a machine entered by hand (`entered_profile`, CONTRACTS.md, "Hardware profile v2"), and nothing of
it was measured: every number the ranking shows for it is computed.

The profile is written as `<state>/hardware/<profile_id>.json` with the writer `hardware` uses,
and gets a `[machines.<name>]` entry of its own (`guided_write.with_entered_machine`), so the
render covers it. It is never bound to this machine -- the pointer file stays as it is -- and a
later measurement of this machine never writes into it (`binding.resolve_profile_target`).
Entering the same machine again makes a new profile; the one before stays, as a measured one
would. CONTRACTS.md, "Guided mode", step 1.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .config import Configuration
from .dialog import Choice
from .guided_write import Said, with_entered_machine
from .measure import DISPLAY_NAME_MAX
from .profile import PROFILE_SCHEMA_VERSION, CrossCheck, HardwareProfile, LlmfitCrosscheck
from .screen import entered_note, gib_words
from .state import acquire_lock, atomic_write_json, release_lock

if TYPE_CHECKING:
    from .guided import GuidedRun

QUESTIONS: dict[str, str] = {
    "entered_name": "What is the machine called?",
    "entered_ram": "How much memory does it have, in GiB?",
    "entered_gpu": "What runs the model?",
    "entered_vram": "How much graphics memory, in GiB?",
}
# The entry of a size list that leads to a number of one's own, and the question it asks then.
NUMBER_VALUE = "number"
NUMBER_LABEL = "enter a number"
_NUMBER_QUESTIONS = {"entered_ram": "Memory in GiB", "entered_vram": "Graphics memory in GiB"}
_SIZES = {
    "entered_ram": ("16", "32", "64", "128", "256"),
    "entered_vram": ("8", "12", "16", "24", "48", "80"),
}
# The three shapes: the answer file's value, the words of the list, and the profile's `gpu_state`.
SHAPES: dict[str, str] = {
    "card": "a graphics card with its own memory",
    "none": "no graphics card",
    "unified": "unified memory (system and graphics share one pool)",
}
_GPU_STATES = {"card": "entered", "none": "none", "unified": "unified_memory"}
_FRESH_ID_ATTEMPTS = 8
ENTERED_MARK = "(entered)"


@dataclass(frozen=True)
class EnteredMachine:
    """The answers of the four questions: a name, the memory, the shape and a graphics memory."""

    name: str
    ram_gib: float
    shape: str
    vram_gib: float


def _guided_error(message: str) -> Exception:
    from .guided import GuidedError  # imported here: the guided mode imports this module in turn

    return GuidedError(message)


def gib_of(key: str, answered: str) -> float:
    """An answer as a number of GiB above 0; anything else ends the run (exit `2`) and names it."""
    try:
        value = float(answered)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise _guided_error(f"{key}: {answered!r} is not a number of GiB above 0")
    return value


def _ask_name(run: "GuidedRun") -> str:
    name = run.asker.text("entered_name", QUESTIONS["entered_name"]).strip()
    if not 1 <= len(name) <= DISPLAY_NAME_MAX:
        raise _guided_error(
            f"entered_name: a machine name has 1 to {DISPLAY_NAME_MAX} characters, this one has {len(name)}"
        )
    run.screen.answer(QUESTIONS["entered_name"], name)
    return name


def _ask_gib(run: "GuidedRun", key: str) -> float:
    choices = [Choice(size, f"{size} GiB") for size in _SIZES[key]] + [Choice(NUMBER_VALUE, NUMBER_LABEL)]
    answered = run.asker.select_or_text(key, QUESTIONS[key], choices, NUMBER_VALUE, _NUMBER_QUESTIONS[key])
    value = gib_of(key, answered.strip())
    run.screen.answer(QUESTIONS[key], gib_words(value))
    return value


def ask_machine(run: "GuidedRun") -> EnteredMachine:
    """The four questions, in their order; the graphics memory only for a graphics card."""
    name = _ask_name(run)
    ram = _ask_gib(run, "entered_ram")
    shape = run.asker.select("entered_gpu", QUESTIONS["entered_gpu"], [Choice(value, words) for value, words in SHAPES.items()])
    run.screen.answer(QUESTIONS["entered_gpu"], SHAPES[shape])
    vram = _ask_gib(run, "entered_vram") if shape == "card" else 0.0
    return EnteredMachine(name, ram, shape, vram)


def entered_profile(
    name: str, ram_gib: float, shape: str, vram_gib: float, profile_id: str, now: datetime
) -> HardwareProfile:
    """The schema-3 profile of a machine entered by hand, in one of its three shapes.

    Nothing of it was measured or cross-checked: no fingerprint, no RAM limit, no llmfit reading,
    no GPU name. A graphics card carries its memory as `entered`; no graphics card and unified
    memory carry none of their own (`vram_gib` 0, source `none`).
    """
    if shape not in _GPU_STATES:
        raise ValueError(f"a machine entered by hand is one of {sorted(_GPU_STATES)}, not {shape!r}")
    card = shape == "card"
    absent = CrossCheck(status="absent")
    return HardwareProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        display_name=name,
        os_fingerprint="none",
        os_fingerprint_source="none",
        origin="entered",
        recorded_at=now,
        ram_physical_gib=ram_gib,
        ram_physical_source="entered",
        ram_limit_gib=None,
        ram_limit_scope="none",
        vram_gib=vram_gib if card else 0.0,
        vram_source="entered" if card else "none",
        gpu_state=_GPU_STATES[shape],
        gpu_name=None,
        llmfit_crosscheck=LlmfitCrosscheck(ram_physical=absent, vram=absent),
        llmfit_version=None,
    )


def entered_class(profile: HardwareProfile) -> str:
    """The group of the machine list for a machine entered by hand: what it is, and that it was entered."""
    ram = f"{profile.ram_physical_gib:.2f} GiB"
    if profile.gpu_state == "unified_memory":
        return f"shared memory {ram} {ENTERED_MARK}"
    card = f"graphics card {profile.vram_gib:.2f} GiB" if profile.gpu_state == "entered" else "no graphics card"
    return f"{card} {ENTERED_MARK} / {ram} RAM"


def taken_profile_ids(hardware_dir: Path) -> set[str]:
    """Every name a new profile must not take: each file name in the folder, in any letter case.

    A schema-1 file (`<machine>.json`) and a file that does not read carry no `profile_id` a scan
    could see, and `atomic_write_json` would replace either; so the file names count on their
    own (the rule `hardware` follows, `cli._taken_profile_ids`). Compared without letter case: on
    a file system that ignores it, `ABCD….json` and `abcd….json` are one file.
    """
    return {path.stem.casefold() for path in hardware_dir.glob("*.json")} if hardware_dir.is_dir() else set()


def _fresh_profile_id(run: "GuidedRun", hardware_dir: Path) -> str:
    taken = taken_profile_ids(hardware_dir)
    for _ in range(_FRESH_ID_ATTEMPTS):
        candidate = run.probes.new_id()
        if candidate.casefold() not in taken and not os.path.lexists(hardware_dir / f"{candidate}.json"):
            return candidate
    raise _guided_error(f"no unused profile_id after {_FRESH_ID_ATTEMPTS} attempts; the id source repeats itself")


def _built(machine: EnteredMachine, profile_id: str, now: datetime) -> HardwareProfile:
    """The profile of the answers; a value the profile refuses ends the run with its reason."""
    try:
        return entered_profile(machine.name, machine.ram_gib, machine.shape, machine.vram_gib, profile_id, now)
    except ValidationError as exc:
        reasons = "; ".join(error["msg"] for error in exc.errors())
        raise _guided_error(f"this machine does not validate as a hardware profile: {reasons}") from exc


def _record_entry(run: "GuidedRun", config_file: Path, profile: HardwareProfile) -> Configuration:
    from .guided import _write_config  # imported here: the guided mode imports this module in turn

    said = Said()
    config = _write_config(run, config_file, with_entered_machine(profile.profile_id, profile.display_name, said))
    for problem in said.problems:
        run.failed(problem)
    return config


def _write_profile(run: "GuidedRun", config: Configuration, machine: EnteredMachine) -> HardwareProfile | None:
    """Pick the id and write the profile under `modelroom.lock`, like every writer of a profile.

    The id is checked against the folder under the lock, immediately before the write, so no
    other writer can put a file there in between. A lock another process holds ends the run with
    its own exit code (`1`) before anything is written; a profile that cannot be written is a
    problem the run reports, and `None`.
    """
    hardware_dir = config.paths.hardware_dir
    try:
        handle = acquire_lock(config.paths.lock_file, "guided", run.now)  # `LockHeldError` goes on up
    except OSError as exc:
        run.failed(f"nothing entered: {config.paths.lock_file}: the lock could not be taken ({exc})")
        return None
    try:
        profile = _built(machine, _fresh_profile_id(run, hardware_dir), run.now)
        path = hardware_dir / f"{profile.profile_id}.json"
        try:
            atomic_write_json(path, profile.model_dump(mode="json"))
        except OSError as exc:
            run.failed(f"nothing entered: {path}: the profile could not be written ({exc})")
            return None
        return profile
    finally:
        release_lock(handle)


def enter_step(run: "GuidedRun", config_file: Path, config: Configuration) -> Configuration:
    """Ask, write the profile, then its `[machines.<name>]` entry; the pointer file stays as it is.

    A profile that cannot be written is reported and the run goes on with the machines that are
    there; an entry that cannot be made is reported too, and the profile stays in the folder,
    where the next run finds it and enters it (`guided_write.with_found_profiles`).
    """
    machine = ask_machine(run)
    profile = _write_profile(run, config, machine)
    if profile is None:
        return config
    run.screen.note(entered_note(profile))
    return _record_entry(run, config_file, profile)
