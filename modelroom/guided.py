"""The guided mode: `modelroom` with no subcommand, a configuration generator over the three
commands.

It asks, writes `modelroom.toml`, and then calls exactly the functions `hardware`, `fetch` and
`render` call -- there is no second way of computing anything here. Every question is asked
through `dialog.Asker`, so the same run works at a terminal and from an answer file
(`modelroom --answers <file>`). The questions, their keys and the order of the steps are
CONTRACTS.md, "Guided mode".

Five steps, each with a head of its own and one line left behind when it is done: 1 the results
folder, the configuration and the machines, 2 the search, the choice and the fetch, 3 the
context (`modelroom/guided_context.py`), 4 the load test of the installed packages
(`modelroom/guided_loadtest.py`), 5 the render. The fetch belongs to step 2 and not to a step
of its own, because step 3 counts how many of the packages it found still fit.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .binding import KnownProfile, PointerFileError, default_pointer_path, read_pointer, resolve_profile_target, write_pointer
from .catalog import Catalog, load_catalog
from .config import (
    CONFIG_SCHEMA_VERSION, DEFAULT_RESERVE_VRAM_GIB, ConfigError, Configuration, load_config, read_config_schema_version,
)
from .contracts import SchemaVersionError
from .daemon import Daemon, LocalDaemon
from .dialog import Asker, Choice
from .guided_context import DEFAULT_CONTEXT, Checked
from .guided_context import QUESTION as CONTEXT_QUESTION
from .guided_context import USERS_KEY, USERS_QUESTION
from .guided_context import chosen_runtimes, context_step, snapshot_facts, snapshot_packages, vllm_among_chosen
from .guided_entered import QUESTIONS as ENTERED_QUESTIONS, CLASS_CHOICES
from . import room as room_module
# The folder question moved to `guided_folder.py` (AP9-UX2 U1/U7); the names stay readable here.
from .guided_folder import CONFIG_NAME, MARKDOWN_PATH, STATE_FOLDER  # noqa: F401
from .guided_folder import QUESTIONS as FOLDER_QUESTIONS
from .guided_folder import config_file as _config_file
from .guided_folder import known_config as _known_config
from .guided_folder import last_elsewhere
from .guided_folder import looks_like_results_folder as _looks_like_results_folder
from .guided_folder import remember_folder as _remember_folder
from .guided_install import QUESTIONS as PULL_QUESTIONS, PullContext, held_identities, other_runtimes_held, pull_step
from .pull_other import KEY as PULL_OTHER_KEY, QUESTION as PULL_OTHER_QUESTION
from .pull_other import eligible_other_machines, pull_other_step
from .pull_choice import ambiguous_names, run_packages
from .room import Disk, RoomProbes, free_disk, live_room
# `_hardware_class` is re-exported: `tests/test_guided_cards.py` reads it from here (AP9-UX move).
from .guided_machines import ANOTHER_COMPUTER, _hardware_class, _machine_choices, _measured_here  # noqa: F401
from .guided_machines import chosen_machines, machine_answer_words
from .guided_loadtest import NOTHING_MEASURED
from .guided_loadtest import QUESTIONS as LOAD_TEST_QUESTIONS
from .guided_remote_measure import QUESTIONS as REMOTE_MEASURE_QUESTIONS
from .guided_remote_measure import combine_measured
from .local_servers import QUESTIONS as SERVER_QUESTIONS, ServerRun, card_facts, default_transports, load_tests
from .guided_models import (
    ModelChoice, ask_models, chosen_hits, hint_line, list_context, model_choices, picked_names, unusable_counts,
)
from .guided_search import DID_YOU_MEAN_KEY, DID_YOU_MEAN_QUESTION, FILTER_QUESTION, SEARCH_QUESTION, search_step
from .http import RequestBudget, Transport, UrllibTransport
from .guided_write import Change, Said, create_config, neutral_change, update_config, with_found_profiles, with_profile, with_writer
from .importer import (
    ImportConflictError, PrerequisiteError, ProfileScan, StoredFileError, import_profile, scan_profiles,
)
from .guided_share import QUESTIONS as SHARE_QUESTIONS
from .guided_share import machine_list_row
from .guided_remote import QUESTIONS as SSH_QUESTIONS
from .guided_remote import active_room, gather_arrivals, resolve_active
from . import ssh as ssh_module
from .intro import STEP_COUNT, collect_intro, print_intro
from .measure import Probes, read_machine_class, read_os_identity
from .measurements import Scenario
from .migrate import MigrationError, backup_suffix, migrate
from .neutral_sweep import load_names, sweep
from .remote_store import aliases as ssh_aliases_of, read_remote, remote_path_for, this_machine
from .profile import MACHINE_CLASSES, HardwareProfile, os_fingerprint
from .screen import (
    CLONE_QUESTION, NOTHING_CHOSEN_SUMMARY, THIS_FOLDER, Screen, clone_words, import_notes, machines_summary,
    machines_words, measured_note, names_words, packages_summary, yes_no,
)
from .search import DEFAULT_GUIDED_BUDGET, DEFAULT_PACKAGERS, SearchError, SearchOutcome, apply_hits, run_search
from .search_apply import apply_tensor_repos
from .selection import NO_MACHINE_CHOSEN, NO_MODEL_CHOSEN, RunSelection, machine_origins, narrow_config
from .state import LockHeldError, write_search_log
from .views import RunView, context_short, show_nothing, show_result

# How this machine is measured, as `_clone_mode` answers it: the takeover rule as it stands, a new
# profile ("a clone"), or the bound profile written again under its own id ("the same machine").
MODE_NORMAL = "normal"
MODE_NEW_IDENTITY = "new_identity"
MODE_SAME_MACHINE = "same_machine"
_FRESH_ID_ATTEMPTS = 8
_UNKNOWN = "unknown"

QUESTIONS: dict[str, str] = {
    **FOLDER_QUESTIONS,  # `results`, `results_path` and `write_home` (`modelroom/guided_folder.py`)
    "write_config": "Write a configuration into this folder?",
    "machines": "Which machines should the result cover?",
    "import_file": "Path to the profile file to import",
    "clone": "Is this the same machine or a clone?",
    "machine_class": "What kind of machine is this?",  # on every measurement (decided 2026-09-27)
    **ENTERED_QUESTIONS,  # step 1's machine entered by hand (`modelroom/guided_entered.py`)
    **SHARE_QUESTIONS,  # step 1's shared folder (`modelroom/guided_share.py`, FR-S)
    **SSH_QUESTIONS,  # step 1's ssh row and step 5's pull over SSH (`modelroom/guided_remote.py`, FR-B1)
    # Step 2's two search questions live with the search (`modelroom/guided_search.py`).
    "search": SEARCH_QUESTION,
    "filter_owners": FILTER_QUESTION,
    # Asked only when the search found no model and the catalog knows a word close to the one that
    # was typed, so an answer file that never runs into that case never needs the key.
    DID_YOU_MEAN_KEY: DID_YOU_MEAN_QUESTION,
    "select": "Which of these models should the result cover?",
    "context": CONTEXT_QUESTION,
    # Step 3 asks it before the scale: `N` people, or `N requests` named outright.
    USERS_KEY: USERS_QUESTION,
    # Step 4's own two (`modelroom/guided_loadtest.py`) and step 5's pull (`guided_install.py`),
    # then a package for another chosen machine (`pull_other.py`, A3).
    **LOAD_TEST_QUESTIONS, **PULL_QUESTIONS, **SERVER_QUESTIONS, **REMOTE_MEASURE_QUESTIONS,
    PULL_OTHER_KEY: PULL_OTHER_QUESTION,
}


class GuidedError(Exception):
    """The guided run cannot go on, and the message says why (the caller ends with exit `2`)."""


@dataclass
class GuidedRun:
    """Everything one guided run carries from step to step; nothing global, nothing guessed."""

    asker: Asker
    here: Path
    pointer_path: Path
    transport: Transport
    probes: Probes
    now: datetime
    catalog: Catalog
    daemon: Daemon = field(default_factory=LocalDaemon)
    servers: ServerRun = field(default_factory=lambda: ServerRun(default_transports()))  # `local_servers.py`
    out: Callable[[str], None] = print
    # Every line with a pattern goes through the screen (`modelroom/screen.py`); `out` stays for the
    # fault sentences and for the blocks of the result view, which bring their own layout.
    screen: Screen = field(default_factory=lambda: Screen(print))
    budget: RequestBudget = field(default_factory=lambda: RequestBudget(DEFAULT_GUIDED_BUDGET))
    # Every step that could not do what it was asked, in the words the user already read. The run
    # goes on with what is there -- a failed measurement must not cost the ranking of the other
    # machines -- but the exit code says that something did not finish (`1`).
    problems: list[str] = field(default_factory=list)
    # What steps 1 and 2 of this run chose; the steps after them and the result cover only this
    # (`selection.py`, AP9-UX B). In memory only, never written.
    selection: RunSelection = field(default_factory=lambda: RunSelection((), ()))
    # The free graphics memory and disk of this machine, read live in step 5 (`room.py`, AP9-UX2 U8).
    room: RoomProbes = field(default_factory=lambda: room_module.default_room_probes())
    # The SSH transport (FR-B1, B1), and this run's one active target set in step 1 (B4); `None`
    # without one -- every other chosen SSH profile is ranked without a connection (`pull_other.py`).
    ssh: "ssh_module.SshRunner" = field(default_factory=lambda: ssh_module.default_ssh())
    ssh_active: str | None = None

    def failed(self, message: str) -> None:
        """Say it once and remember it: the run continues, its exit code does not stay `0`."""
        self.out(message)
        self.problems.append(message)


# --- step 1: which folder, which configuration in it, which machines -----------------------------


def _configuration_step(run: GuidedRun, config_arg: Path | None) -> tuple[Path, Configuration]:
    """Step 1: the results folder, the configuration in it, and the machines the result covers."""
    run.screen.blank()
    run.screen.head(1)
    config_file = _config_file(run, config_arg)
    config = _configuration(run, config_file)
    config = _machines_step(run, config_file, config)
    run.screen.done(1, f"{config_file.parent}, {machines_summary(len(config.machines))}")
    return config_file, config


def _machine_key(run: GuidedRun) -> str:
    """This machine's key in `[machines.<name>]`: its neutral key in `remote.json`, never its host name (FR-N)."""
    return this_machine(remote_path_for(run.pointer_path))


def _sweep(run: GuidedRun, config_file: Path) -> None:
    """The run's first write to the folder: no file of it keeps a machine name (`neutral_sweep`, FR-N)."""
    swept = sweep(config_file, run.now, "guided", pointer_path=run.pointer_path, hostname=run.probes.hostname())
    for note in swept.notes:
        run.screen.note(note)


def _new_configuration(run: GuidedRun, config_file: Path) -> Configuration:
    """Write a fresh schema-2 configuration with this device as the writer, families empty.

    `packagers` stays empty: the search writes every repository it resolved as an owner-bound
    `repos` target, so a speculative `<packager>/<name>-GGUF` probe under five accounts per base
    model would only spend the shared budget on repositories nobody asked about. The owner classes
    the list shows come from `search.DEFAULT_PACKAGERS`; whoever wants the probes writes the list.
    A file another run created meanwhile is taken over as it is (`guided_write.create_config`).
    """
    folder = config_file.parent
    name = _machine_key(run)
    config = Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [],
            "packagers": [],
            "publishers": [],
            "machines": {
                # No `reserve_ram_gib`: the memory reserve is computed from the machine (ADR 0012).
                name: {
                    "reserve_vram_gib": DEFAULT_RESERVE_VRAM_GIB,
                    "writer": True,
                }
            },
            "paths": {"state": str(folder / STATE_FOLDER), "markdown": str(folder.joinpath(*MARKDOWN_PATH))},
            "guided": {"results": str(folder)},
        }
    )
    config = _write_config(run, config_file, config)
    _remember_folder(run, folder)
    return config


def _write_config(run: GuidedRun, config_file: Path, change: Change | Configuration) -> Configuration:
    """Write and return what is in the file now: a new file's first state, or a step's change.

    A change goes through `guided_write.update_config` -- read, change, write, and once more on a
    conflict -- and a second conflict in a row is a dialog error (exit `2`). A held lock keeps its
    own exit code (`1`), it is not a dialog error. Every configuration written goes through the sink
    first, the one read last included: a key another run added meanwhile is neutral too (FR-N).
    """
    names = load_names(config_file.parent, run.pointer_path, run.probes.hostname())
    try:
        if isinstance(change, Configuration):
            return create_config(config_file, change, now=run.now)
        return update_config(config_file, neutral_change(change, names), now=run.now)
    except (ConfigError, OSError) as exc:
        raise GuidedError(f"{config_file}: the configuration could not be written ({exc})") from exc


def _configuration(run: GuidedRun, config_file: Path) -> Configuration:
    """The configuration to work with: migrated, loaded, or newly written after a question."""
    if not config_file.is_file():
        if _looks_like_results_folder(config_file.parent):
            question = f"{config_file.parent} already holds results but no {CONFIG_NAME}. {QUESTIONS['write_config']}"
            answered = run.asker.confirm("write_config", question)
            run.screen.answer(QUESTIONS["write_config"], yes_no(answered))
            if not answered:
                raise GuidedError("no configuration to work with; run again and name a folder")
        _new_configuration(run, config_file)
        return _configure_found_profiles(run, config_file)
    stored = _stored_schema_version(config_file)
    if stored < CONFIG_SCHEMA_VERSION:
        try:
            lines = migrate(config_file, run.now, pointer_path=run.pointer_path, hostname=run.probes.hostname())
        except (ConfigError, MigrationError) as exc:
            raise GuidedError(f"{exc}; nothing was changed -- move that file away and run again") from exc
        run.screen.note(
            "this folder holds a configuration from an earlier version; it was updated, "
            f"backup kept: {config_file.name}{backup_suffix(stored)}"
        )
        for line in lines:
            run.screen.note(line)
    _load(config_file)  # a file that does not read ends the run before the folder is remembered
    _remember_folder(run, config_file.parent)
    return _configure_found_profiles(run, config_file)


def _load(config_file: Path) -> Configuration:
    """`load_config`, with a missing or invalid file as a dialog error (exit `2`).

    `SchemaVersionError` is not caught: an unsupported `schema_version` keeps its own exit code
    `3`, the same as for every other command.
    """
    try:
        return load_config(config_file)
    except ConfigError as exc:
        raise GuidedError(str(exc)) from exc


def _stored_schema_version(config_file: Path) -> int:
    try:
        raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise GuidedError(f"{config_file}: cannot read the configuration ({exc})") from exc
    return read_config_schema_version(raw, str(config_file))


def _configure_found_profiles(run: GuidedRun, config_file: Path) -> Configuration:
    """Give every profile in the folder a `[machines.<name>]` entry, so the render covers it.

    The render computes one ranking per configured machine, so a profile the configuration does not
    name is left out of the result -- and the machine list would then say "already in this results
    folder" about a machine that is in no result at all. This is the same rule `import-profile`
    follows for a profile it finds without an entry (CONTRACTS.md, "Export and import"): the name
    comes from the `display_name`, the reserves from `[defaults]`, `writer` is false. The matching is
    worked out on the file as it is at the write, and again after a conflict (`with_found_profiles`).
    """
    _sweep(run, config_file)
    said = Said()
    config = _write_config(run, config_file, with_found_profiles(said))
    for problem in said.problems:
        run.failed(problem)
    for note in said.notes:
        run.screen.note(note)
    return config


# --- step 1, second half: the machines this result covers ---------------------------------------


def _ensure_this_machine(run: GuidedRun, config_file: Path, name: str) -> Configuration:
    """Make sure `[machines.<name>]` is there and is a writer -- this machine runs the fetch.

    Reads the file again first: the machine question stood on screen in between, and the copy this
    run loaded before it would write over whatever another process changed meanwhile.
    """
    return _write_config(run, config_file, with_writer(name))


def _fresh_profile_id(run: GuidedRun, scan: ProfileScan) -> str:
    for _ in range(_FRESH_ID_ATTEMPTS):
        candidate = run.probes.new_id()
        if candidate not in scan.profiles:
            return candidate
    raise GuidedError(f"no unused profile_id after {_FRESH_ID_ATTEMPTS} attempts; the id source repeats itself")


def _clone_mode(run: GuidedRun, config: Configuration, name: str, scan: ProfileScan, results_dir: Path) -> str:
    """How this machine is measured: `normal`, `new_identity` or `same_machine`.

    The takeover rule decides (`resolve_profile_target`); only its `ask_clone` case is a question
    here. The local fingerprint is read on its own (`read_os_identity`), so the question comes
    before the measurement rather than after it. Since 2026-09-24 both answers measure: "a clone"
    writes a new profile, "the same machine" writes the bound profile again under its own id. A
    results folder someone emptied is a folder to measure into, not a dead end.
    """
    identity = read_os_identity(run.probes.platform, run.probes.runner, run.probes.read_text)
    local = os_fingerprint(identity.raw_id) if identity.raw_id else "none"
    known = {
        key: KnownProfile(key, profile.os_fingerprint, profile.origin, profile.ram_physical_source)
        for key, profile in scan.profiles.items()
    }
    pointer = read_pointer(run.pointer_path)
    target = resolve_profile_target(
        pointer.binding_for(results_dir),
        config.machines[name].profile,
        known,
        local,
        _fresh_profile_id(run, scan),
    )
    if target.action != "ask_clone":
        return MODE_NORMAL
    question = f"{target.reason} ({target.profile_id}). {QUESTIONS['clone']}"
    answer = run.asker.select("clone", question, [Choice("same", "the same machine"), Choice("clone", "a clone")])
    # The short form in the answer line: the profile id belongs to the question, not to the answer.
    run.screen.answer(CLONE_QUESTION, clone_words(answer))
    return MODE_NEW_IDENTITY if answer == "clone" else MODE_SAME_MACHINE


def _measure_this_machine(
    run: GuidedRun, config_file: Path, config: Configuration, scan: ProfileScan, measured_before: bool
) -> Configuration:
    """Measure this machine, then write `[machines.<name>].profile` -- the one write A leaves to C."""
    from .cli import hardware_with_config  # imported here: the CLI imports this module in turn

    results_dir = config_file.parent
    name = _machine_key(run)
    config = _ensure_this_machine(run, config_file, name)
    mode = _clone_mode(run, config, name, scan, results_dir)
    code = hardware_with_config(
        config,
        name,
        run.probes,
        run.now,
        run.pointer_path,
        new_identity=mode == MODE_NEW_IDENTITY,
        same_machine=mode == MODE_SAME_MACHINE,
        results_dir=results_dir,
        machine_class=_asked_class(run),
        # The readings are the profile file's business; this step says in one note what was measured.
        summary=False,
        note=run.screen.note,  # every note of the measurement indented like its neighbors (AP9-UX G)
    )
    if code != 0:
        run.failed(
            f"the measurement ended with exit {code}; the ranking uses the profiles the folder already holds"
        )
        return config
    config = _record_profile(run, config_file, config, name, results_dir)
    _measured(run, config, results_dir, again=measured_before or mode == MODE_SAME_MACHINE)
    return config


def _asked_class(run: GuidedRun) -> str | None:
    """The class of this machine, asked on every measurement (decided 2026-09-27).

    The chassis is read first: where it names a class, the list is the three classes with that one
    marked, and the answer that keeps it returns `None`, so the profile keeps the source `chassis`;
    any other answer is that class, entered. Where it names none, the list is the four entries with
    nothing marked, and `unknown` returns `None` as well.
    """
    read = read_machine_class(run.probes.platform, run.probes.runner, run.probes.read_text)
    if read.source == "unknown":
        choices = [Choice(value, words) for value, words in CLASS_CHOICES.items()]
    else:
        choices = [Choice(value, CLASS_CHOICES[value], checked=value == read.machine_class) for value in MACHINE_CLASSES]
    answer = run.asker.select("machine_class", QUESTIONS["machine_class"], choices)
    run.screen.answer(QUESTIONS["machine_class"], CLASS_CHOICES[answer])
    return None if answer in ("unknown", read.machine_class) else answer


def _measured(run: GuidedRun, config: Configuration, results_dir: Path, *, again: bool) -> None:
    """The one note a measurement leaves behind: what this machine is, and who confirmed it."""
    profile = _bound_profile(run, scan_profiles(config.paths.hardware_dir), results_dir)
    if profile is not None:
        run.screen.note(measured_note(profile, again=again))


def _record_profile(
    run: GuidedRun, config_file: Path, config: Configuration, name: str, results_dir: Path
) -> Configuration:
    """Name the profile this machine is now bound to in `[machines.<name>].profile`.

    The configuration is read again first: the dialog took time, and another process may have changed
    the file meanwhile (an `import-profile` adds a `[machines.<name>]` entry). Writing the copy this
    run loaded before the dialog would drop that change. That re-read can also show a file this
    measurement no longer belongs to -- the machine entry gone or renamed, or `[paths]` pointing at
    another state folder, where this `profile_id` names nothing. Then the entry is left alone and the
    run says so; the profile itself is written and bound either way, the checks held again after a
    conflict (`guided_write.with_profile`).
    """
    said = Said()
    profile_id = read_pointer(run.pointer_path).binding_for(results_dir)
    config = _write_config(run, config_file, with_profile(name, profile_id, config.paths, config_file, said))
    for problem in said.problems:
        run.failed(problem)
    return config


def _import_a_profile(run: GuidedRun, config_file: Path, config: Configuration) -> tuple[Configuration, set[str]]:
    """`modelroom import-profile` from inside the guided mode; the binding is never changed.

    Returns the configuration and the profile ids this import brought or named: the result of the
    run covers their machines (AP9-UX B).
    """
    answered = Path(run.asker.text("import_file", QUESTIONS["import_file"]).strip()).expanduser()
    path = answered if answered.is_absolute() else run.here / answered
    run.screen.answer(QUESTIONS["import_file"], str(path), path=True)
    before = set(scan_profiles(config.paths.hardware_dir).profiles)
    try:
        report = import_profile(config_file, path, run.now, pointer_path=run.pointer_path, hostname=run.probes.hostname())
    except (
        SchemaVersionError,
        StoredFileError,
        ConfigError,
        PrerequisiteError,
        ImportConflictError,
        LockHeldError,
    ) as exc:
        # An import that does not go through is reported and the run goes on with the machines
        # that are there -- it is one entry of the machine list, not the whole run. The exit code
        # still says that something did not finish.
        run.failed(f"nothing imported: {exc}")
        return config, set()
    except OSError as exc:
        run.failed(f"nothing imported: {path}: {exc}")
        return config, set()
    updated = _load(config_file)
    found = sorted(scan_profiles(updated.paths.hardware_dir).profiles.items())
    for note in import_notes([profile for key, profile in found if key not in before], report):
        run.screen.note(note)
    arrived = {key for key, _profile in found if key not in before}
    return updated, arrived | _exported_profile_id(path)


def _exported_profile_id(path: Path) -> set[str]:
    """The `profile_id` an export file names -- it was just imported, so it reads; else nothing."""
    try:
        profile_id = json.loads(path.read_text(encoding="utf-8"))["profile"]["profile_id"]
    except (OSError, ValueError, KeyError, TypeError):
        return set()
    return {profile_id} if isinstance(profile_id, str) else set()


def _machines_step(run: GuidedRun, config_file: Path, config: Configuration) -> Configuration:
    """The machine list (`guided_machines`), then what was marked: measured, imported, entered, or
    an existing profile of the folder taken as it is; the run's result covers exactly these (B)."""
    scan = scan_profiles(config.paths.hardware_dir)
    bound = _bound_profile(run, scan, config_file.parent)
    name = _machine_key(run)
    configured = [machine.profile for machine in config.machines.values() if machine.profile is not None]
    run.screen.note(ANOTHER_COMPUTER)
    # Row 5's own label (FR-S): a `Choice`, like `import`/`enter` next to it -- the checkbox itself
    # is the one place it is shown, never a note of its own beside it (Codex r1-5: that was a
    # product-visible double display, not the "always in the list" the row asks for).
    share_label = machine_list_row(run.room.environ, read_remote(remote_path_for(run.pointer_path)))
    aliases = ssh_aliases_of(remote_path_for(run.pointer_path))
    choices = _machine_choices(scan, bound, name, configured, share_label, aliases)
    picked = run.asker.checkbox("machines", QUESTIONS["machines"], choices)
    run.screen.answer(QUESTIONS["machines"], machines_words(picked, machine_answer_words(scan, bound)))
    if "this-machine" in picked:
        config = _measure_this_machine(run, config_file, config, scan, _measured_here(bound))
    config, imported, ssh_ids, shared, entered, ssh_alias = gather_arrivals(run, config_file, config, picked)
    # An empty choice stays empty -- nothing marked in a file, or an import that failed: no machine the
    # run did not choose is added, the bound one included (Codex r2, finding 1). At a terminal `Enter`
    # takes the pointer row, so a terminal run always chooses one.
    machines = chosen_machines(config, picked, name, imported | ssh_ids | shared | entered)
    run.ssh_active = resolve_active(
        run, config, config_file, scan_profiles(config.paths.hardware_dir), machines, ssh_alias, "ssh" in picked
    )
    # Every machine with a stored SSH alias, active or not this run (Codex r1 finding 10) -- not
    # only the ones that arrived just now (`ssh_ids`), which a re-chosen stored profile never is.
    all_ssh_ids = set(ssh_aliases_of(remote_path_for(run.pointer_path)))
    origins = _origins(run, config, config_file, machines, picked, imported | shared, entered, all_ssh_ids)
    run.selection = replace(run.selection, machines=machines, origins=origins)
    return config


def _origins(run: GuidedRun, config: Configuration, config_file: Path, machines, picked, imported, entered, ssh: Iterable[str] = ()) -> dict:
    """Where each chosen machine came from (U2, FR-B1 B3): this, ssh, imported, entered, folder."""
    bound = _bound_profile(run, scan_profiles(config.paths.hardware_dir), config_file.parent)
    own = bound.profile_id if _measured_here(bound) else None
    name = _machine_key(run)
    this = {name} if "this-machine" in picked else set()
    this |= {machine for machine in machines if own is not None and config.machines[machine].profile == own}
    return machine_origins(config, machines, this=this, ssh=ssh, imported=imported, entered=entered)


def _bound_profile(run: GuidedRun, scan: ProfileScan, results_dir: Path) -> HardwareProfile | None:
    """The profile this folder binds this machine to, if the folder still holds it."""
    bound = read_pointer(run.pointer_path).binding_for(results_dir)
    return scan.profiles.get(bound) if bound is not None else None


# --- step 2: search, choose, fetch ---------------------------------------------------------------


def _search_step(run: GuidedRun, config_file: Path, config: Configuration) -> tuple[Configuration, int]:
    """The search of step 2 (`modelroom/guided_search.py`), then the choice, then the write-back.

    The list holds one line per **model** with the fit its size allows, and only the ones that can be
    picked: a hand test of 2026-09-24 showed 120 repository lines nobody could choose from. The
    repositories behind a chosen model are `guided_models.chosen_hits`', unchanged through `apply_hits`.
    Returns the configuration and how many models were chosen.

    **Every way out of this step reads the file again**, the two that change nothing included: the
    fetch runs on what this step returns, and after a dialog that took as long as the user took, the
    copy loaded before it may name another writer, state directory or models than the file does now.
    """
    step = search_step(run, config, config_file.parent)
    if not step.models:
        # A list with nothing in it is not a question: the dialog cannot show one, and there is
        # nothing an answer could add (test round of 2026-09-24, `mistral` with the filter on).
        run.screen.note("no repository of a publisher or a listed packager was resolved; nothing added")
        return _load(config_file), 0
    picked = ask_models(run.asker, QUESTIONS["select"], step.models, hint_line(step.checked, step.context))
    run.screen.answer(QUESTIONS["select"], names_words(picked_names(step.models, picked)))
    chosen = chosen_hits(step.models, picked, config.packagers or DEFAULT_PACKAGERS)
    if not chosen:
        return _load(config_file), 0
    # Base models, not answers (second-model round, 2026-09-24) -- a base model nobody picked is
    # never written for, whatever P1's search found for it (`step.tensor_repos`).
    bases = {str(hit.resolved_base_model) for hit in chosen}
    tensor_for_chosen = {base: repos for base, repos in step.tensor_repos.items() if base in bases}

    # Read again here, immediately before the change is applied and written (see `_record_profile`).
    def change(current: Configuration) -> Configuration:
        try:
            updated = apply_hits(current, chosen, catalog=run.catalog)
        except ConfigError as exc:
            raise GuidedError(str(exc)) from exc
        if tensor_for_chosen:
            updated = apply_tensor_repos(updated, tensor_for_chosen, catalog=run.catalog)
        return updated

    updated = _write_config(run, config_file, change)
    _remember_models(run, [model for model in step.models if model.base_model in bases])
    return updated, len(bases)


def _remember_models(run: GuidedRun, models: Sequence[ModelChoice]) -> None:
    """The models of this run, with the release and the size the list showed (AP9-UX B, ADR 0009)."""
    run.selection = replace(
        run.selection,
        models=tuple(model.base_model for model in models),
        release={model.base_model: model.age for model in models},
        parameters={model.base_model: model.parameters_b for model in models},
    )


def _packages_step(run: GuidedRun, config_file: Path, config: Configuration) -> tuple[Configuration, str, bool]:
    """Step 2: search, choose, and fetch -- so that step 3 can count what really is in the folder."""
    run.screen.blank()
    run.screen.head(2)
    config, chosen = _search_step(run, config_file, config)
    _fetch_step(run, config)
    if not chosen:
        return config, NOTHING_CHOSEN_SUMMARY, False
    # All three numbers about one thing -- the snapshot this folder now holds. `apply_hits` keeps
    # what was there, so a balance may not mix the two frames (second-model round, 2026-09-24);
    # counted for the chosen machines, as the result document will be.
    facts = snapshot_facts(config, run.selection, chosen_runtimes(run, config))
    return config, packages_summary(len(facts.model_names), facts.packages, facts.repositories), True


def context_default(config: Configuration) -> int:
    """What the size scale starts at: the context this folder kept, else the default level.

    A kept context is a decision of an earlier run, so it is the level the pointer starts on --
    the default level is only ever offered to a folder no guided run has chosen a context in.
    """
    return DEFAULT_CONTEXT if config.guided.context is None else config.guided.context


# --- the fetch of step 2, and step 5: the render --------------------------------------------------


def _fetch_step(run: GuidedRun, config: Configuration) -> None:
    """Step 2's fetch. `build_tensor` is `guided_context.vllm_among_chosen`, the same gate the
    tensor search of this step reads; `False` reads and writes exactly as every fetch always did."""
    from .cli import fetch_with_config

    if not config.families:
        run.screen.note("no base model is configured yet, so there is nothing to fetch")
        return
    name = _machine_key(run)
    build_tensor = vllm_among_chosen(run, config)
    code = fetch_with_config(config, name, run.transport, run.now, budget=run.budget, build_tensor=build_tensor)
    if code != 0:
        run.failed(f"the fetch ended with exit {code}; the ranking uses what the snapshot holds")


def _render_step(
    run: GuidedRun, config_file: Path, config: Configuration, scenario: Scenario, daemon_reachable: bool = False
) -> tuple[int, PullContext | None]:
    """Step 5: render, then the card of the whole run (decided 2026-09-24) and one table per machine.

    Built from the document the render just wrote, so no number is computed twice; with nothing to
    render the card is drawn all the same (`views.show_nothing`, 2026-09-25). Returns the render's
    exit code, and what the pull that follows chooses from (`None` without a document).
    """
    from .cli import render_with_config

    run.screen.blank()
    run.screen.head(STEP_COUNT)
    written: list = []
    here = _machine_key(run)
    chosen = run.selection
    if not chosen.machines:
        run.screen.note(NO_MACHINE_CHOSEN)
    if not chosen.models:
        run.screen.note(NO_MODEL_CHOSEN)
    machines = chosen_runtimes(run, config)  # the card counts for the machines the document shows
    live = run.servers.found.get("vllm")

    def seen(document) -> None:
        # Read inside the callback, while the render still holds the lock: card, table and pull packages
        # rest on one snapshot (second-model round, 2026-09-24).
        packages = run_packages(config, chosen)  # before the eligibility filter (U4, Codex r3-1)
        facts = snapshot_facts(config, chosen, machines)
        written.append((document, facts, PullContext(document, here, packages, ambiguous_names(packages), disk)))

    # Free disk read once, before the render: Markdown, table and pull list mark the same rows (AP9-RM
    # U11); the check before a pull reads it again.
    disk = free_disk(run.room)
    if daemon_reachable:  # a package the daemon holds already needs no disk and is never marked (U11c)
        disk = replace(disk, held=held_identities(run, run_packages(config, chosen)))
    code = render_with_config(
        config, now=run.now, scenario=scenario, on_document=seen, selection=chosen, disks={here: disk}, live_vllm={here: live}
    )
    if not written:
        shown = narrow_config(config, chosen)  # the card of an empty step 5 names this run's machines only
        show_nothing(run.screen, shown, config_file.parent, scenario, run.now, card_facts(run), chosen.origins)
        return code, None
    return _shown_render_result(run, config, config_file, chosen, here, disk, daemon_reachable, written, code)


def _shown_render_result(
    run: GuidedRun,
    config: Configuration,
    config_file: Path,
    chosen: RunSelection,
    here: str,
    disk: Disk,
    daemon_reachable: bool,
    written: list,
    code: int,
) -> tuple[int, PullContext]:
    """The non-empty tail of `_render_step`: build the card's view from the one document just written."""
    document, facts, context = written[0]
    own = next((block.profile for block in document.machines if block.machine == here), None)
    live = {here: live_room(run.room, own, run.daemon if daemon_reachable else None, disk)}  # U10: this machine only
    live.update(active_room(run, document))  # B5: the active SSH machine's own room, if it has a table here
    # A3 (AP9-FR): other machines `pull_other_step` lists after this table, computed once so the note
    # under a table and the list that may follow never disagree.
    other_machines = eligible_other_machines(document, here, context.ambiguous)
    view = RunView(ambiguous=context.ambiguous, live=live, origins=chosen.origins, pull_other_machines=other_machines,
                   ssh_active=run.ssh_active, packages=context.packages, held=other_runtimes_held(run, context))
    show_result(run.screen, document, facts, config, config_file.parent, run.now, here, card_facts(run), view)
    return code, context


def run_guided(
    asker: Asker,
    *,
    here: Path,
    config_arg: Path | None = None,
    pointer_path: Path | None = None,
    transport: Transport | None = None,
    probes: Probes | None = None,
    now: datetime | None = None,
    out: Callable[[str], None] = print,
    catalog: Catalog | None = None,
    daemon: Daemon | None = None,
    colored: bool | None = None,
    servers: dict | None = None,
    room: RoomProbes | None = None,
    ssh: "ssh_module.SshRunner | None" = None,
) -> int:
    """Run the whole guided mode; returns the exit code (`0` when a document was written).

    `here` is the folder "this folder" means -- the caller passes it in, this package never resolves
    a path against the working directory on its own. Everything else is the usual dependency
    injection: the transport, the daemon and the other local servers of this machine by key
    (`local_servers.py`), its probes, the clock, the pointer file and the shipped catalog. `colored`
    says whether the start screen may carry color; `modelroom --answers <file>` passes `False`, `None`
    asks the stream (`intro.use_color`). `room`: the free memory and disk of this machine (`room.py`);
    `None` is the real one.
    """
    run = GuidedRun(
        asker=asker,
        here=here.resolve(),
        pointer_path=pointer_path if pointer_path is not None else default_pointer_path(),
        transport=transport if transport is not None else UrllibTransport(),
        probes=probes if probes is not None else Probes(),
        now=(now or datetime.now(timezone.utc)).replace(microsecond=0),
        catalog=catalog if catalog is not None else load_catalog(),
        daemon=daemon if daemon is not None else LocalDaemon(),
        servers=ServerRun(servers if servers is not None else default_transports()),
        out=out,
        screen=Screen(out, colored=colored),
        room=room if room is not None else room_module.default_room_probes(),
        ssh=ssh if ssh is not None else ssh_module.default_ssh(),
    )
    known = _known_config(run, config_arg)
    elsewhere = last_elsewhere(run, config_arg, known)
    intro = collect_intro(run.daemon, run.probes, run.pointer_path, known, run.servers, last_elsewhere=elsewhere)
    print_intro(intro, run.out, colored=colored)
    config_file, config = _configuration_step(run, config_arg)
    config, packages, chose = _packages_step(run, config_file, config)
    run.screen.summary(2, packages, chose)
    scenario, config = context_step(run, config_file, config)
    # `views.context_short` names a context the way the scale of step 3 does: `L 32k`, or the number.
    run.screen.done(3, context_short(scenario.context_requested))
    measured = load_tests(run, config, scenario, config_file.parent)
    measured = combine_measured(run, config, scenario, measured)  # B2: the active SSH target, if any
    run.screen.summary(4, measured, measured != NOTHING_MEASURED)
    code, first = _render_step(run, config_file, config, scenario, intro.daemon_reachable)
    pull_step(run, first, intro.daemon_reachable)
    pull_other_step(run, first)  # a row of the active SSH target is pulled there (B5)
    if code == 0 and run.problems:
        # The document was written, but a step did not do what it was asked -- the pull included.
        # Exit `1` is the documented "a step reported it": an automated caller must not read this
        # run as a clean one, and the lines the user already saw say which step it was.
        run.out(f"{len(run.problems)} step(s) did not finish; see the lines above")
        return 1
    return code
