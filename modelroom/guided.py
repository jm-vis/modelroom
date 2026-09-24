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
from typing import Callable

from .binding import KnownProfile, PointerFileError, default_pointer_path, read_pointer, resolve_profile_target, write_pointer
from .catalog import Catalog, load_catalog
from .config import (
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
from .guided_context import DEFAULT_CONTEXT
from .guided_context import QUESTION as CONTEXT_QUESTION
from .guided_context import context_step, machine_checked, snapshot_packages
from .guided_loadtest import QUESTIONS as LOAD_TEST_QUESTIONS
from .guided_loadtest import load_test_step
from .guided_models import (
    ask_models,
    chosen_hits,
    count_line,
    fit_line,
    hint_line,
    list_context,
    model_choices,
    unusable_reasons,
)
from .http import RequestBudget, Transport, UrllibTransport
from .importer import (
    ImportConflictError,
    PrerequisiteError,
    ProfileScan,
    StoredFileError,
    import_profile,
    machine_name_for,
    scan_profiles,
)
from .intro import collect_intro, done_line, print_intro, step_head
from .measure import Probes, read_os_identity
from .measurements import Scenario
from .migrate import BACKUP_SUFFIX, MigrationError, migrate
from .profile import HardwareProfile, os_fingerprint
from .search import (
    DEFAULT_GUIDED_BUDGET,
    DEFAULT_PACKAGERS,
    SearchError,
    apply_hits,
    run_search,
    write_configuration,
)
from .state import LockHeldError

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
    "search": "What are you looking for?",
    "filter_owners": "Show only repositories of a publisher or a listed packager?",
    "select": "Which of these models should the result cover?",
    "context": CONTEXT_QUESTION,
    # Step 4's two questions live with the step (`modelroom/guided_loadtest.py`); the answer file
    # has one table of keys, so they are merged in here.
    **LOAD_TEST_QUESTIONS,
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
    run.out("")
    run.out(step_head(1))
    config_file = _config_file(run, config_arg)
    config = _configuration(run, config_file)
    config = _machines_step(run, config_file, config)
    run.out(done_line(1, f"{config_file}, {len(config.machines)} machine(s)"))
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
        [Choice("here", f"this folder ({run.here})"), Choice("path", "another path")],
    )
    if choice == "here":
        folder = run.here
    else:
        answered = Path(run.asker.text("results_path", QUESTIONS["results_path"]).strip()).expanduser()
        folder = answered if answered.is_absolute() else run.here / answered
    return folder.resolve() / CONFIG_NAME


def _config_file(run: GuidedRun, config_arg: Path | None) -> Path:
    """`--config` wins over the pointer file; an unreachable target asks again, never guesses."""
    if config_arg is not None:
        if config_arg.is_file():
            return config_arg.resolve()
        run.out(f"{config_arg}: there is no configuration at this path")
        return _ask_config_file(run)
    pointer = read_pointer(run.pointer_path)
    if pointer.current is not None:
        candidate = Path(pointer.current) / CONFIG_NAME
        if candidate.is_file():
            return candidate.resolve()
        run.out(f"{pointer.current}: the results folder of the last run no longer holds a {CONFIG_NAME}")
    return _ask_config_file(run)


def _looks_like_results_folder(folder: Path) -> bool:
    """Whether a folder already holds results, so writing a configuration into it is a question."""
    state = folder / STATE_FOLDER
    return state.is_dir() or (folder / MARKDOWN_PATH[0]).is_dir()


def _machine_key(hostname: str) -> str:
    """This machine's key in `[machines.<name>]`: its host name in the shape a key has to have."""
    slug = "".join(character if character.isalnum() else "-" for character in hostname.lower()).strip("-")
    if not slug:
        raise GuidedError(f"this machine's host name ({hostname!r}) cannot be a machine name; write one by hand")
    return validate_machine_name(slug)


def _new_configuration(run: GuidedRun, config_file: Path) -> Configuration:
    """Write a fresh schema-2 configuration with this device as the writer, families empty.

    `packagers` stays empty: the search writes every repository it resolved as an owner-bound
    `repos` target, so a speculative `<packager>/<name>-GGUF` probe under five accounts per base
    model would only spend the shared request budget on repositories nobody asked about. The
    owner classes the selection list shows come from `search.DEFAULT_PACKAGERS` while the
    configuration lists none, and a user who wants the speculative probes writes the list by hand.
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
    _write_config(run, config_file, config)
    _remember_folder(run, folder)
    run.out(f"wrote {config_file} with {name} as the writer of this results folder")
    return config


def _write_config(run: GuidedRun, config_file: Path, config: Configuration) -> None:
    """Write the configuration; a held lock keeps its own exit code (`1`), it is not a dialog error."""
    try:
        write_configuration(config_file, config, now=run.now)
    except (ConfigError, OSError) as exc:
        raise GuidedError(f"{config_file}: the configuration could not be written ({exc})") from exc


def _remember_folder(run: GuidedRun, folder: Path) -> None:
    """Record the results folder in the pointer file, so the next run finds it without asking.

    Written only when it really changes. The pointer file is one file per user for every results
    folder and is replaced as a whole, without a lock (the same trade-off `cli._bind_this_machine`
    documents), so a write that changes nothing could still drop a binding a concurrent
    `hardware` run added between this read and this write. A run in the folder it already
    remembers therefore does not write at all.
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
            if not run.asker.confirm("write_config", question):
                raise GuidedError("no configuration to work with; run again and name a folder")
        return _configure_found_profiles(run, config_file, _new_configuration(run, config_file))
    if _stored_schema_version(config_file) == 1:
        run.out(
            "This folder holds a configuration from an earlier version; it was updated, "
            f"backup kept: {config_file.name}{BACKUP_SUFFIX}"
        )
        try:
            for line in migrate(config_file, run.now):
                run.out(line)
        except ConfigError as exc:
            raise GuidedError(str(exc)) from exc
    config = _load(config_file)
    _remember_folder(run, config_file.parent)
    return _configure_found_profiles(run, config_file, config)


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


def _configure_found_profiles(run: GuidedRun, config_file: Path, config: Configuration) -> Configuration:
    """Give every profile in the folder a `[machines.<name>]` entry, so the render covers it.

    The render computes one ranking per configured machine, so a profile the configuration does
    not name is left out of the result -- and the machine list would then say "already in this
    results folder" about a machine that is in no result at all. This is the same rule
    `import-profile` follows for a profile it finds without an entry (CONTRACTS.md, "Export and
    import"): the name comes from the `display_name`, the reserves from `[defaults]`, `writer`
    is false.
    """
    scan = scan_profiles(config.paths.hardware_dir)
    configured = {machine.profile for machine in config.machines.values() if machine.profile is not None}
    missing = [profile for key, profile in sorted(scan.profiles.items()) if key not in configured]
    if not missing:
        return config
    data = config.model_dump(mode="json")
    added = 0
    for profile in missing:
        try:
            name = machine_name_for(profile.display_name, profile.profile_id, set(data["machines"]))
        except ImportConflictError as exc:
            run.failed(f"the profile {profile.profile_id} stays without a machine entry: {exc}")
            continue
        data["machines"][name] = {
            "reserve_ram_gib": config.defaults.reserve_ram_gib,
            "reserve_vram_gib": config.defaults.reserve_vram_gib,
            "writer": False,
            "profile": profile.profile_id,
        }
        added += 1
        run.out(f"machine {name!r} added for the profile {profile.profile_id} found in this folder")
    if not added:
        return config
    updated = Configuration.from_dict(data)
    _write_config(run, config_file, updated)
    return updated


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


def _ensure_this_machine(run: GuidedRun, config_file: Path, config: Configuration, name: str) -> Configuration:
    """Make sure `[machines.<name>]` is there and is a writer -- this machine runs the fetch.

    Reads the file again first: the machine question stood on screen in between, and the copy this
    run loaded before it would write over whatever another process changed meanwhile.
    """
    config = _load(config_file)
    machine = config.machines.get(name)
    if machine is not None and machine.writer:
        return config
    data = config.model_dump(mode="json")
    entry = data["machines"].get(name, {})
    entry.update(
        {
            "reserve_ram_gib": entry.get("reserve_ram_gib", config.defaults.reserve_ram_gib),
            "reserve_vram_gib": entry.get("reserve_vram_gib", config.defaults.reserve_vram_gib),
            "writer": True,
        }
    )
    data["machines"][name] = entry
    updated = Configuration.from_dict(data)
    _write_config(run, config_file, updated)
    run.out(f"{name} is now a writer of this results folder")
    return updated


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
    known = {key: KnownProfile(key, profile.os_fingerprint) for key, profile in scan.profiles.items()}
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
    return MODE_NEW_IDENTITY if answer == "clone" else MODE_SAME_MACHINE


def _measure_this_machine(
    run: GuidedRun, config_file: Path, config: Configuration, scan: ProfileScan
) -> Configuration:
    """Measure this machine, then write `[machines.<name>].profile` -- the one write A leaves to C."""
    from .cli import hardware_with_config  # imported here: the CLI imports this module in turn

    results_dir = config_file.parent
    name = _machine_key(run.probes.hostname())
    config = _ensure_this_machine(run, config_file, config, name)
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
    )
    if code != 0:
        run.failed(
            f"the measurement ended with exit {code}; the ranking uses the profiles the folder already holds"
        )
        return config
    return _record_profile(run, config_file, config, name, results_dir)


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
    """
    profile_id = read_pointer(run.pointer_path).binding_for(results_dir)
    stored_paths = config.paths
    config = _load(config_file)
    if config.paths != stored_paths:
        run.failed(f"{config_file}: [paths] changed while this machine was measured; run again")
        return config
    if name not in config.machines:
        run.failed(f"{config_file}: the machine {name!r} is gone from the configuration; run again")
        return config
    if profile_id is None or config.machines[name].profile == profile_id:
        return config
    data = config.model_dump(mode="json")
    data["machines"][name]["profile"] = profile_id
    updated = Configuration.from_dict(data)
    _write_config(run, config_file, updated)
    run.out(f"{name} is measured as profile {profile_id}")
    return updated


def _import_a_profile(run: GuidedRun, config_file: Path, config: Configuration) -> Configuration:
    """`modelroom import-profile` from inside the guided mode; the binding is never changed."""
    answered = Path(run.asker.text("import_file", QUESTIONS["import_file"]).strip()).expanduser()
    path = answered if answered.is_absolute() else run.here / answered
    try:
        for line in import_profile(config_file, path, run.now):
            run.out(line)
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
    return _load(config_file)


def _machines_step(run: GuidedRun, config_file: Path, config: Configuration) -> Configuration:
    scan = scan_profiles(config.paths.hardware_dir)
    bound = _bound_profile(run, scan, config_file.parent)
    picked = run.asker.checkbox("machines", QUESTIONS["machines"], _machine_choices(scan, bound))
    if "this-machine" in picked:
        config = _measure_this_machine(run, config_file, config, scan)
    if "import" in picked:
        config = _import_a_profile(run, config_file, config)
    return config


def _bound_profile(run: GuidedRun, scan: ProfileScan, results_dir: Path) -> HardwareProfile | None:
    """The profile this folder binds this machine to, if the folder still holds it."""
    bound = read_pointer(run.pointer_path).binding_for(results_dir)
    return scan.profiles.get(bound) if bound is not None else None


# --- step 2: search, choose, fetch ---------------------------------------------------------------


def _search_step(run: GuidedRun, config_file: Path, config: Configuration) -> Configuration:
    """Search, then the list of **models** (`modelroom/guided_models.py`), then the choice.

    One line per model with the fit its size allows, and only the ones that can be picked: a
    hand test of 2026-09-24 showed 120 repository lines nobody could choose from. The
    repositories behind a chosen model are `guided_models.chosen_hits`', and they go through the
    unchanged `apply_hits`.
    """
    name = run.asker.text("search", QUESTIONS["search"]).strip()
    if not name:
        raise GuidedError("no model name to search for")
    filtered = run.asker.confirm("filter_owners", QUESTIONS["filter_owners"], default=True)
    listed_packagers = config.packagers or DEFAULT_PACKAGERS
    try:
        outcome = run_search(
            run.transport,
            name,
            catalog=run.catalog,
            listed_packagers=listed_packagers,
            budget=run.budget,
            # The owner filter is a question about the request, not only about the list: with it
            # on, only the accounts are asked; switching it off adds the two open lists on top
            # (decided 2026-09-24).
            open_pages=not filtered,
        )
    except SearchError as exc:
        raise GuidedError(str(exc)) from exc
    for line in outcome.group_lines():
        run.out(line)
    run.out(outcome.summary_line())
    context = list_context(config.guided.context)
    checked = machine_checked(run.pointer_path, config, config_file.parent)
    models = model_choices(outcome.hits, filtered=filtered, checked=checked, context=context)
    run.out(count_line(models, outcome.hits, filtered))
    for line in unusable_reasons(outcome.hits, filtered):
        run.out(line)
    fit_at = fit_line(context, checked)
    if fit_at is not None:
        run.out(fit_at)
    run.out(hint_line(checked))
    picked = ask_models(run.asker, QUESTIONS["select"], models)
    chosen = chosen_hits(models, picked, listed_packagers)
    if not chosen:
        if not models:
            run.out("no repository of a publisher or a listed packager was resolved; nothing added")
        else:
            run.out("nothing chosen; the configuration stays as it is")
        return config
    # Read again here, after the last question of this step and immediately before the change is
    # applied and written: the search and the selection list both took time (see `_record_profile`).
    config = _load(config_file)
    try:
        updated = apply_hits(config, chosen, catalog=run.catalog)
    except ConfigError as exc:
        raise GuidedError(str(exc)) from exc
    _write_config(run, config_file, updated)
    run.out(f"added {len(chosen)} repository target(s) to {config_file}")
    return updated


def _packages_step(run: GuidedRun, config_file: Path, config: Configuration) -> tuple[Configuration, str]:
    """Step 2: search, choose, and fetch -- so that step 3 can count what really is in the folder."""
    run.out("")
    run.out(step_head(2))
    before = _repository_count(config)
    config = _search_step(run, config_file, config)
    added = _repository_count(config) - before
    _fetch_step(run, config)
    fetched, _base_models = snapshot_packages(config)
    return config, f"{added} repositories added, {len(fetched)} packages fetched"


def _repository_count(config: Configuration) -> int:
    """How many packaging repositories the configuration names, over every base model."""
    return sum(len(base_model.repos) for family in config.families for base_model in family.base_models)


def context_default(config: Configuration) -> int:
    """What the size scale starts at: the context this folder kept, else the default level.

    A kept context is a decision of an earlier run, so it is the level the pointer starts on --
    the default level is only ever offered to a folder no guided run has chosen a context in.
    """
    return DEFAULT_CONTEXT if config.guided.context is None else config.guided.context


def context_label(context: int) -> str:
    """How a chosen context is named in a line that looks back at it: `L 32k`, or the number.

    One definition for the dialog and for the result view, which says the same thing in its own
    first line (`views.context_short`).
    """
    from .views import context_short  # imported here: the view reads the scale, not the reverse

    return context_short(context)


# --- the fetch of step 2, and step 5: the render --------------------------------------------------


def _fetch_step(run: GuidedRun, config: Configuration) -> None:
    from .cli import fetch_with_config

    if not config.families:
        run.out("no base model is configured yet, so there is nothing to fetch")
        return
    name = _machine_key(run.probes.hostname())
    code = fetch_with_config(config, name, run.transport, run.now, budget=run.budget)
    if code != 0:
        run.failed(f"the fetch ended with exit {code}; the ranking uses what the snapshot holds")


def _render_step(run: GuidedRun, config: Configuration, scenario: Scenario) -> int:
    from .cli import render_with_config

    run.out("")
    code = render_with_config(config, now=run.now, scenario=scenario, echo=run.out)
    if code == 0:
        run.out(f"Written to {config.paths.markdown}   and   {config.paths.state}")
    return code


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
    )
    intro = collect_intro(run.daemon, run.probes, run.pointer_path, _known_config(run, config_arg))
    print_intro(intro, run.out, colored=colored)
    config_file, config = _configuration_step(run, config_arg)
    config, packages = _packages_step(run, config_file, config)
    run.out(done_line(2, packages))
    scenario, config = context_step(run, config_file, config)
    run.out(done_line(3, f"context {context_label(scenario.context_requested)}"))
    measured = load_test_step(run, config, scenario, config_file.parent)
    run.out(done_line(4, measured))
    code = _render_step(run, config, scenario)
    if code == 0 and run.problems:
        # The document was written, but a step before it did not do what it was asked. Exit `1`
        # is the documented "a step reported it": an automated caller must not read this run as
        # a clean one, and the lines the user already saw say which step it was.
        run.out(f"{len(run.problems)} step(s) did not finish; see the lines above")
        return 1
    return code
