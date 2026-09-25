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

import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .binding import KnownProfile, PointerFileError, default_pointer_path, read_pointer, resolve_profile_target, write_pointer
from .catalog import Catalog, load_catalog
from .config import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_RESERVE_RAM_GIB,
    DEFAULT_RESERVE_VRAM_GIB,
    ConfigError,
    Configuration,
    load_config,
    read_config_schema_version,
)
from .contracts import SchemaVersionError, validate_machine_name
from .daemon import Daemon, LocalDaemon
from .dialog import Asker, Choice
from .guided_context import DEFAULT_CONTEXT, Checked
from .guided_context import QUESTION as CONTEXT_QUESTION
from .guided_context import context_step, machine_checked, snapshot_facts, snapshot_packages
from .guided_install import QUESTIONS as PULL_QUESTIONS, FirstRow, first_row, pull_step
from .guided_loadtest import NOTHING_MEASURED
from .guided_loadtest import QUESTIONS as LOAD_TEST_QUESTIONS
from .guided_loadtest import load_test_step
from .guided_models import (
    ModelChoice,
    ask_models,
    chosen_hits,
    hint_line,
    list_context,
    model_choices,
    picked_names,
    unusable_counts,
)
from .guided_search import (
    DID_YOU_MEAN_KEY,
    DID_YOU_MEAN_QUESTION,
    FILTER_QUESTION,
    SEARCH_QUESTION,
    search_step,
)
from .http import RequestBudget, Transport, UrllibTransport
from .guided_write import Change, Said, create_config, update_config, with_found_profiles, with_profile, with_writer
from .importer import (
    ImportConflictError,
    PrerequisiteError,
    ProfileScan,
    StoredFileError,
    import_profile,
    scan_profiles,
)
from .intro import STEP_COUNT, collect_intro, print_intro
from .measure import Probes, read_os_identity
from .measurements import Scenario
from .migrate import MigrationError, backup_suffix, migrate
from .profile import HardwareProfile, os_fingerprint
from .screen import (
    CLONE_QUESTION,
    NOTHING_CHOSEN_SUMMARY,
    THIS_FOLDER,
    Screen,
    clone_words,
    import_notes,
    machines_summary,
    machines_words,
    measured_note,
    names_words,
    packages_summary,
    yes_no,
)
from .search import (
    DEFAULT_GUIDED_BUDGET,
    DEFAULT_PACKAGERS,
    SearchError,
    SearchOutcome,
    apply_hits,
    run_search,
)
from .state import LockHeldError, write_search_log
from .views import context_short, show_nothing, show_result

CONFIG_NAME = "modelroom.toml"
# How this machine is measured, as `_clone_mode` answers it: the takeover rule as it stands, a new
# profile ("a clone"), or the bound profile written again under its own id ("the same machine").
MODE_NORMAL = "normal"
MODE_NEW_IDENTITY = "new_identity"
MODE_SAME_MACHINE = "same_machine"
STATE_FOLDER = "state"
MARKDOWN_PATH = ("docs", "models.md")
_FRESH_ID_ATTEMPTS = 8
_UNKNOWN = "unknown"

QUESTIONS: dict[str, str] = {
    "results": "Where should results live?",
    "results_path": "Path to the results folder",
    "write_config": "Write a configuration into this folder?",
    "machines": "Which machines should the result cover?",
    "import_file": "Path to the profile file to import",
    "clone": "Is this the same machine or a clone?",
    # Step 2's two search questions live with the search (`modelroom/guided_search.py`).
    "search": SEARCH_QUESTION,
    "filter_owners": FILTER_QUESTION,
    # Asked only when the search found no model and the catalog knows a word close to the one that
    # was typed, so an answer file that never runs into that case never needs the key.
    DID_YOU_MEAN_KEY: DID_YOU_MEAN_QUESTION,
    "select": "Which of these models should the result cover?",
    "context": CONTEXT_QUESTION,
    # Step 4's own two (`modelroom/guided_loadtest.py`) and step 5's pull (`guided_install.py`).
    **LOAD_TEST_QUESTIONS, **PULL_QUESTIONS,
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
    out: Callable[[str], None] = print
    # Every line with a pattern goes through the screen (`modelroom/screen.py`); `out` stays for the
    # fault sentences and for the blocks of the result view, which bring their own layout.
    screen: Screen = field(default_factory=lambda: Screen(print))
    budget: RequestBudget = field(default_factory=lambda: RequestBudget(DEFAULT_GUIDED_BUDGET))
    # Every step that could not do what it was asked, in the words the user already read. The run
    # goes on with what is there -- a failed measurement must not cost the ranking of the other
    # machines -- but the exit code says that something did not finish (`1`).
    problems: list[str] = field(default_factory=list)

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


def _known_config(run: GuidedRun, config_arg: Path | None) -> Path | None:
    """The configuration this run can already read before its first question, for the start screen.

    `--config` when it names a file, else the folder the pointer file remembers when it still
    holds a configuration. Neither is decided here -- `_config_file` asks and says what it found;
    this is only what the start screen may already show.
    """
    if config_arg is not None and config_arg.is_file():
        return config_arg.resolve()
    current = read_pointer(run.pointer_path).current
    if current is None:
        return None
    candidate = Path(current) / CONFIG_NAME
    return candidate.resolve() if candidate.is_file() else None


def _ask_config_file(run: GuidedRun) -> Path:
    """The configuration file the run works on, from the folder question."""
    choice = run.asker.select(
        "results",
        QUESTIONS["results"],
        [Choice("here", f"{THIS_FOLDER} ({run.here})"), Choice("path", "another path")],
    )
    if choice == "here":
        folder = run.here
        run.screen.answer(QUESTIONS["results"], THIS_FOLDER)
    else:
        answered = Path(run.asker.text("results_path", QUESTIONS["results_path"]).strip()).expanduser()
        folder = answered if answered.is_absolute() else run.here / answered
        run.screen.answer(QUESTIONS["results"], str(folder.resolve()), path=True)
    return folder.resolve() / CONFIG_NAME


def _config_file(run: GuidedRun, config_arg: Path | None) -> Path:
    """`--config` wins over the pointer file; an unreachable target asks again, never guesses."""
    if config_arg is not None:
        if config_arg.is_file():
            return config_arg.resolve()
        run.screen.note(f"{config_arg}: there is no configuration at this path")
        return _ask_config_file(run)
    pointer = read_pointer(run.pointer_path)
    if pointer.current is not None:
        candidate = Path(pointer.current) / CONFIG_NAME
        if candidate.is_file():
            return candidate.resolve()
        # Without the path in front of it: the answer to the question that follows says which
        # folder this run works in, and it said the same path twice (test round, 2026-09-24).
        run.screen.note(f"the results folder of the last run no longer holds a {CONFIG_NAME}")
    return _ask_config_file(run)


def _looks_like_results_folder(folder: Path) -> bool:
    """Whether a folder already holds results, so writing a configuration into it is a question."""
    state = folder / STATE_FOLDER
    return state.is_dir() or (folder / MARKDOWN_PATH[0]).is_dir()


def _machine_key(hostname: str) -> str:
    """This machine's key in `[machines.<name>]`: its host name in the shape a key has to have."""
    slug = "".join(letter if letter.isalnum() else "-" for letter in hostname.lower()).strip("-")
    if not slug:
        raise GuidedError(f"this machine's host name ({hostname!r}) cannot be a machine name; write one by hand")
    return validate_machine_name(slug)


def _new_configuration(run: GuidedRun, config_file: Path) -> Configuration:
    """Write a fresh schema-2 configuration with this device as the writer, families empty.

    `packagers` stays empty: the search writes every repository it resolved as an owner-bound
    `repos` target, so a speculative `<packager>/<name>-GGUF` probe under five accounts per base
    model would only spend the shared budget on repositories nobody asked about. The owner classes
    the list shows come from `search.DEFAULT_PACKAGERS`; whoever wants the probes writes the list.
    A file another run created meanwhile is taken over as it is (`guided_write.create_config`).
    """
    folder = config_file.parent
    name = _machine_key(run.probes.hostname())
    config = Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [],
            "packagers": [],
            "publishers": [],
            "machines": {
                name: {
                    "reserve_ram_gib": DEFAULT_RESERVE_RAM_GIB,
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
    own exit code (`1`), it is not a dialog error.
    """
    try:
        if isinstance(change, Configuration):
            return create_config(config_file, change, now=run.now)
        return update_config(config_file, change, now=run.now)
    except (ConfigError, OSError) as exc:
        raise GuidedError(f"{config_file}: the configuration could not be written ({exc})") from exc


def _remember_folder(run: GuidedRun, folder: Path) -> None:
    """Record the results folder in the pointer file, so the next run finds it without asking.

    Written only when it really changes: the file is one per user for every results folder and is
    replaced as a whole, without a lock (the trade-off `cli._bind_this_machine` documents), so even
    a write that changes nothing could drop a binding a concurrent `hardware` run added in between.
    """
    pointer = read_pointer(run.pointer_path)
    if pointer.current == str(folder):
        return
    try:
        write_pointer(run.pointer_path, pointer.with_current(folder))
    except (OSError, ValueError) as exc:
        run.out(f"{run.pointer_path}: this folder could not be remembered ({exc}); the next run asks again")


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
            lines = migrate(config_file, run.now)
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

    The render computes one ranking per configured machine, so a profile the configuration does
    not name is left out of the result -- and the machine list would then say "already in this
    results folder" about a machine that is in no result at all. This is the same rule
    `import-profile` follows for a profile it finds without an entry (CONTRACTS.md, "Export and
    import"): the name comes from the `display_name`, the reserves from `[defaults]`, `writer`
    is false. The matching is worked out on the file as it is at the write, and again after a
    conflict (`guided_write.with_found_profiles`).
    """
    said = Said()
    config = _write_config(run, config_file, with_found_profiles(said))
    for problem in said.problems:
        run.failed(problem)
    for note in said.notes:
        run.screen.note(note)
    return config


# --- step 1, second half: the machines this result covers ---------------------------------------


def _hardware_class(profile: HardwareProfile) -> str:
    """What makes two machines the same to the user: the GPU, its memory and the system memory."""
    gpu = profile.gpu_name or profile.gpu_state
    return f"{gpu} {_gib(profile.vram_gib)} VRAM / {_gib(profile.ram_physical_gib)} RAM"


def _gib(value: float | None) -> str:
    return _UNKNOWN if value is None else f"{value:.2f} GiB"


def _profile_choices(scan: ProfileScan) -> list[Choice]:
    """Every profile the results folder holds, grouped by hardware class, alphabetical, grayed out.

    The group is a matter of operation only -- the ranking is computed per device, and every one
    of these profiles is already part of the result, so none of them is an action to pick.
    """
    groups: dict[str, list[str]] = {}
    for profile in scan.profiles.values():
        groups.setdefault(_hardware_class(profile), []).append(profile.display_name)
    choices = []
    for hardware_class, names in sorted(groups.items(), key=lambda item: sorted(item[1])):
        listed = ", ".join(sorted(names))
        count = f"{len(names)} machine" if len(names) == 1 else f"{len(names)} machines"
        choices.append(
            Choice(
                f"group:{hardware_class}",
                f"{hardware_class} -- {count}: {listed}",
                disabled="already in this results folder",
            )
        )
    for path in sorted(scan.legacy):
        choices.append(Choice(f"legacy:{path.name}", f"{path.name} (legacy)", disabled="run `modelroom migrate`"))
    for path, reason in sorted(scan.unreadable):
        choices.append(Choice(f"broken:{path.name}", f"{path.name}", disabled=f"does not read: {reason}"))
    return choices


def _machine_choices(scan: ProfileScan, bound: HardwareProfile | None) -> list[Choice]:
    """The machine list. `bound` is the profile this folder already holds for this machine.

    A machine that is already measured here is **not** marked: measuring it again is a decision,
    not the default, and the entry says when it was last measured so that the decision can be
    made. A machine with no profile in this folder is marked, because that is the one thing a
    first run is for.
    """
    if bound is None:
        this_machine = Choice("this-machine", "this machine (measure now)", checked=True)
    else:
        measured = bound.recorded_at.date().isoformat()
        this_machine = Choice("this-machine", f"this machine (measure again, last measured {measured})")
    return [
        this_machine,
        *_profile_choices(scan),
        Choice("import", "import a profile file"),
        Choice("enter", "enter a machine by hand", disabled="stage 2"),
    ]


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
    name = _machine_key(run.probes.hostname())
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
        # The readings are the profile file's business; this step says in one note what was measured.
        summary=False,
    )
    if code != 0:
        run.failed(
            f"the measurement ended with exit {code}; the ranking uses the profiles the folder already holds"
        )
        return config
    config = _record_profile(run, config_file, config, name, results_dir)
    _measured(run, config, results_dir, again=measured_before or mode == MODE_SAME_MACHINE)
    return config


def _measured(run: GuidedRun, config: Configuration, results_dir: Path, *, again: bool) -> None:
    """The one note a measurement leaves behind: what this machine is, and who confirmed it."""
    profile = _bound_profile(run, scan_profiles(config.paths.hardware_dir), results_dir)
    if profile is not None:
        run.screen.note(measured_note(profile, again=again))


def _record_profile(
    run: GuidedRun, config_file: Path, config: Configuration, name: str, results_dir: Path
) -> Configuration:
    """Name the profile this machine is now bound to in `[machines.<name>].profile`.

    The configuration is read again first: the dialog took time, and another process may have
    changed the file meanwhile (an `import-profile` adds a `[machines.<name>]` entry). Writing
    the copy this run loaded before the dialog would drop that change. That re-read can also show
    a file this measurement no longer belongs to -- the machine entry gone or renamed, or
    `[paths]` pointing at another state folder, where this `profile_id` names nothing. Then the
    entry is left alone and the run says so; the profile itself is written and bound either way.
    Both checks hold again for the read after a conflict (`guided_write.with_profile`).
    """
    said = Said()
    profile_id = read_pointer(run.pointer_path).binding_for(results_dir)
    config = _write_config(run, config_file, with_profile(name, profile_id, config.paths, config_file, said))
    for problem in said.problems:
        run.failed(problem)
    return config


def _import_a_profile(run: GuidedRun, config_file: Path, config: Configuration) -> Configuration:
    """`modelroom import-profile` from inside the guided mode; the binding is never changed."""
    answered = Path(run.asker.text("import_file", QUESTIONS["import_file"]).strip()).expanduser()
    path = answered if answered.is_absolute() else run.here / answered
    run.screen.answer(QUESTIONS["import_file"], str(path), path=True)
    before = set(scan_profiles(config.paths.hardware_dir).profiles)
    try:
        report = import_profile(config_file, path, run.now)
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
        return config
    except OSError as exc:
        run.failed(f"nothing imported: {path}: {exc}")
        return config
    updated = _load(config_file)
    found = sorted(scan_profiles(updated.paths.hardware_dir).profiles.items())
    for note in import_notes([profile for key, profile in found if key not in before], report):
        run.screen.note(note)
    return updated


def _machines_step(run: GuidedRun, config_file: Path, config: Configuration) -> Configuration:
    scan = scan_profiles(config.paths.hardware_dir)
    bound = _bound_profile(run, scan, config_file.parent)
    picked = run.asker.checkbox("machines", QUESTIONS["machines"], _machine_choices(scan, bound))
    run.screen.answer(QUESTIONS["machines"], machines_words(picked))
    if "this-machine" in picked:
        config = _measure_this_machine(run, config_file, config, scan, bound is not None)
    if "import" in picked:
        config = _import_a_profile(run, config_file, config)
    return config


def _bound_profile(run: GuidedRun, scan: ProfileScan, results_dir: Path) -> HardwareProfile | None:
    """The profile this folder binds this machine to, if the folder still holds it."""
    bound = read_pointer(run.pointer_path).binding_for(results_dir)
    return scan.profiles.get(bound) if bound is not None else None


# --- step 2: search, choose, fetch ---------------------------------------------------------------


def _search_step(run: GuidedRun, config_file: Path, config: Configuration) -> tuple[Configuration, int]:
    """The search of step 2 (`modelroom/guided_search.py`), then the choice, then the write-back.

    The list holds one line per **model** with the fit its size allows, and only the ones that can
    be picked: a hand test of 2026-09-24 showed 120 repository lines nobody could choose from. The
    repositories behind a chosen model are `guided_models.chosen_hits`', and they go through the
    unchanged `apply_hits`. Returns the configuration and how many models were chosen.

    **Every way out of this step reads the file again**, the two that change nothing included: the
    fetch runs on what this step returns, and after a dialog that took as long as the user took, the
    copy this run loaded before it may name another writer, another state directory or other models
    than the file does (second-model round, 2026-09-25).
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
    # Read again here, after the last question of this step and immediately before the change is
    # applied and written: the search and the selection list both took time (see `_record_profile`).
    # `apply_hits` is pure, so after a conflict it is simply applied to the new state again.
    def change(current: Configuration) -> Configuration:
        try:
            return apply_hits(current, chosen, catalog=run.catalog)
        except ConfigError as exc:
            raise GuidedError(str(exc)) from exc

    updated = _write_config(run, config_file, change)
    # Base models, not answers: an answer file may name two repositories of one model, and the
    # configuration and the card then hold one (second-model round, 2026-09-24).
    return updated, len({str(hit.resolved_base_model) for hit in chosen})


def _packages_step(run: GuidedRun, config_file: Path, config: Configuration) -> tuple[Configuration, str, bool]:
    """Step 2: search, choose, and fetch -- so that step 3 can count what really is in the folder."""
    run.screen.blank()
    run.screen.head(2)
    config, chosen = _search_step(run, config_file, config)
    _fetch_step(run, config)
    if not chosen:
        return config, NOTHING_CHOSEN_SUMMARY, False
    # All three numbers about one thing -- the snapshot this folder now holds. `apply_hits` keeps
    # what was there, so a balance may not mix the two frames (second-model round, 2026-09-24).
    facts = snapshot_facts(config)
    return config, packages_summary(len(facts.model_names), facts.packages, facts.repositories), True


def context_default(config: Configuration) -> int:
    """What the size scale starts at: the context this folder kept, else the default level.

    A kept context is a decision of an earlier run, so it is the level the pointer starts on --
    the default level is only ever offered to a folder no guided run has chosen a context in.
    """
    return DEFAULT_CONTEXT if config.guided.context is None else config.guided.context


# --- the fetch of step 2, and step 5: the render --------------------------------------------------


def _fetch_step(run: GuidedRun, config: Configuration) -> None:
    from .cli import fetch_with_config

    if not config.families:
        run.screen.note("no base model is configured yet, so there is nothing to fetch")
        return
    name = _machine_key(run.probes.hostname())
    code = fetch_with_config(config, name, run.transport, run.now, budget=run.budget)
    if code != 0:
        run.failed(f"the fetch ended with exit {code}; the ranking uses what the snapshot holds")


def _render_step(run: GuidedRun, config_file: Path, config: Configuration, scenario: Scenario) -> tuple[int, FirstRow | None]:
    """Step 5: render, then the card of the whole run (decided 2026-09-24) and one table per machine.

    Built from the document the render just wrote, so no number is computed twice; with nothing to
    render the card is drawn all the same (`views.show_nothing`, 2026-09-25). Returns the render's
    exit code, and this machine's first row for the pull that follows (`None` without one).
    """
    from .cli import render_with_config

    run.screen.blank()
    run.screen.head(STEP_COUNT)
    written: list = []
    here = _machine_key(run.probes.hostname())

    def seen(document) -> None:
        # Read inside the callback, which runs while the render still holds the lock: the card, the
        # table and the package of the first row rest on one snapshot (second-model round, 2026-09-24).
        written.append((document, snapshot_facts(config), first_row(document, here, snapshot_packages(config)[0])))

    code = render_with_config(config, now=run.now, scenario=scenario, on_document=seen)
    if not written:
        show_nothing(run.screen, config, config_file.parent, scenario, run.now)
        return code, None
    show_result(run.screen, *written[0][:2], config, config_file.parent, run.now, here)
    return code, written[0][2]


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
) -> int:
    """Run the whole guided mode; returns the exit code (`0` when a document was written).

    `here` is the folder "this folder" means -- the caller passes it in, this package never
    resolves a path against the working directory on its own. Everything else is the usual
    dependency injection: the transport, the daemon of this machine, the probes of this machine,
    the clock, the pointer file and the shipped catalog. `colored` says whether the start screen
    may carry color; `modelroom --answers <file>` passes `False`, `None` asks the stream
    (`intro.use_color`).
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
        out=out,
        screen=Screen(out, colored=colored),
    )
    intro = collect_intro(run.daemon, run.probes, run.pointer_path, _known_config(run, config_arg))
    print_intro(intro, run.out, colored=colored)
    config_file, config = _configuration_step(run, config_arg)
    config, packages, chose = _packages_step(run, config_file, config)
    run.screen.summary(2, packages, chose)
    scenario, config = context_step(run, config_file, config)
    # `views.context_short` names a context the way the scale of step 3 does: `L 32k`, or the number.
    run.screen.done(3, context_short(scenario.context_requested))
    measured = load_test_step(run, config, scenario, config_file.parent)
    run.screen.summary(4, measured, measured != NOTHING_MEASURED)
    code, first = _render_step(run, config_file, config, scenario)
    pull_step(run, first, intro.daemon_reachable)
    if code == 0 and run.problems:
        # The document was written, but a step did not do what it was asked -- the pull included.
        # Exit `1` is the documented "a step reported it": an automated caller must not read this
        # run as a clean one, and the lines the user already saw say which step it was.
        run.out(f"{len(run.problems)} step(s) did not finish; see the lines above")
        return 1
    return code
