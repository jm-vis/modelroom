"""Step 1's fifth entry: a shared folder of profile exports, for a computer without SSH (FR-S, GF
question 7 (c)).

Where the folder is, in this order: the environment variable `MODELROOM_PROFILE_SHARE` (for
scripts, and it wins over anything stored), else `profile_share` of `~/.modelroom/remote.json`
(set up here, in "Shared folder", and kept per user account), else none. `share_folder` is the
**only** place `store.profile_share` is read as an attribute -- checked by AST, not a grep on the
module name (Codex r1-4); a caller that wants the stored value alone calls `share_folder({}, store)`
rather than touch the field. Nothing this module writes carries the path further than `remote.json`
and this step's own screen -- not a results folder, an export, a render, a failure sentence or a
log (CONTRACTS.md, "Remote machines", "Why the path does not leave").

The menu step never touches the folder before it is chosen (`machine_list_row`, the row of the
machine list): a network drive that happens to be unreachable must not hold up every run that
never opens the shared folder step at all. Once opened, a folder to write to is checked once by
writing and removing a small probe file (`_write_probe`) -- never by `os.access`, which reports
group and world bits a network share does not really enforce.

The folder is never trusted once it is read: `import_profile` gets `expected_profile_id` and
`max_bytes` (Codex r6-2), enforced on the object it actually reads, and every export of a chosen
profile is imported in full (Codex r2-5) -- oldest file first, so the stored freshness rule ends up
exactly where it would if the files had arrived one at a time. Every exception `import_profile` can
raise is mapped to one of three fixed sentences, none of them naming a path or a file name -- the
texts of `importer.py` do (`importer.py:149-157`), so none of them ever reaches the screen as they
are.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from .config import Configuration, load_config
from .contracts import SchemaVersionError
from .dialog import AnswerInvalidError, Choice, FileAsker
from .guided_machines import _hardware_class
from .importer import (
    ConfigError,
    ImportConflictError,
    PrerequisiteError,
    StaleExportError,
    StoredFileError,
    import_profile,
    read_export_file,
)
from .neutral import display_name_for
from .profile import HardwareProfile
from .remote_store import RemoteStore, is_share_path, read_remote, remote_path_for, set_profile_share
from .state import LockHeldError

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

ENV_VAR = "MODELROOM_PROFILE_SHARE"
# The folder's own limits (FR-S): a listing reads at most this many files, and refuses one larger
# than this -- both to keep one careless drop folder from turning a run into a scan of everything
# in it, never a rule about what a real export can hold.
MAX_FILES = 200
MAX_BYTES = 16 * 1024 * 1024
PROBE_PREFIX = ".modelroom-write-probe-"

NO_SHARE_ROW = "set up a shared folder for profiles"
WITH_SHARE_ROW = "profiles from the shared folder"
SET_BY_ENV = f"set by environment ({ENV_VAR})"
NONE_SET_YET = "none set yet"
HINT_LINES = (
    "a folder on this computer, a network drive such as Z:\\, or your cloud sync folder;",
    "other computers leave their profile exports there",
)
CHECKING_NOTE = "checking that modelroom can write there"
KEPT_NOTE = "shared folder kept for your user account"
REMOVED_NOTE = "the shared folder is no longer kept"
NOT_THIS_RUN_NOTE = "the shared folder set by the environment is not used in this run"
FORM_PROBLEM = "give the whole path of a folder, such as Z:\\profiles"
WRITE_PROBLEM = "modelroom cannot write to that folder; check that it exists and that you may write there"
UNREACHABLE_NOTE = "the shared folder cannot be read now; nothing was imported from it"
EMPTY_NOTE = "the shared folder holds no profile export yet"
CHANGED_NOTE = "a file in the shared folder changed after it was listed"

QUESTIONS: dict[str, str] = {
    "share_setup": "What should modelroom do with the shared folder?",
    "share_folder": "Folder for the profile exports:",
    "share_files": "Which profiles should be imported?",
}


# --- where the folder is, and the one row of the machine list ------------------------------------


def env_value(environ: Mapping[str, str]) -> str | None:
    """`MODELROOM_PROFILE_SHARE`, or `None` where it is unset or empty: an empty variable counts as
    unset everywhere in this module (`test_empty_share_env_counts_as_unset`)."""
    value = environ.get(ENV_VAR)
    return value if value else None


def share_folder(environ: Mapping[str, str], store: RemoteStore) -> str | None:
    """The folder this run would read from, or none: the environment first, else `profile_share`.

    Pure -- a string in, a string or `None` out, never a look at the file system
    (`test_the_row_never_touches_the_folder`). The only function that reads `profile_share`.
    """
    return env_value(environ) or store.profile_share


def machine_list_row(environ: Mapping[str, str], store: RemoteStore) -> str:
    """Row 5's own label of the machine list (`guided_machines._machine_choices`): always one of
    these two, whatever picking it would go on to do -- the folder is not read to build the label."""
    return WITH_SHARE_ROW if share_folder(environ, store) is not None else NO_SHARE_ROW


# --- the "Shared folder" menu step ----------------------------------------------------------------


def _stand_lines(environ: Mapping[str, str], store: RemoteStore) -> list[str]:
    """The step's own head: what is stored or set, then the two-line hint (module docstring mockup).

    `share_folder({}, store)` -- an empty environment -- reads as `store.profile_share` exactly (no
    variable can ever win against nothing), so this is `profile_share` alone, through the one
    reader, not a second direct touch of the field (Codex r1-4).
    """
    env = env_value(environ)
    stand = SET_BY_ENV if env is not None else (share_folder({}, store) or NONE_SET_YET)
    return [f"shared folder  {stand}", *HINT_LINES]


def _setup_choices(environ: Mapping[str, str], store: RemoteStore) -> list[Choice]:
    """`use` only where a folder is there at all; `choose` always, grayed while the environment
    wins; `none` always. The pointer starts on `use` where it exists, else on `choose`.

    `stored`/`location` both come from `share_folder` (Codex r1-4): `share_folder({}, store)` for
    the stored value alone, `share_folder(environ, store)` for the one this run would use.
    """
    env = env_value(environ)
    stored = share_folder({}, store)
    location = share_folder(environ, store)
    choices: list[Choice] = []
    if location is not None:
        choices.append(Choice("use", "take profiles from it", checked=True))
    choose_label = "change the folder" if stored is not None else "choose a folder"
    choices.append(Choice("choose", choose_label, disabled=SET_BY_ENV if env is not None else None, checked=location is None))
    choices.append(Choice("none", "none, no shared folder"))
    return choices


def _form_problem(value: str) -> str | None:
    return None if is_share_path(value) else FORM_PROBLEM


def _write_probe(folder: Path, token: Callable[[], str] = lambda: secrets.token_hex(4)) -> tuple[bool, str | None]:
    """Write, then remove, a probe file of our own: `(True, None)` on full success, `(True, <note>)`
    when only the removal failed (the folder still counts), `(False, WRITE_PROBLEM)` when nothing
    could be written there -- its own file name added when a write/close failure left it behind and
    the cleanup after that failed too (Codex r2-1). `open(..., "xb")` never touches a file already
    there; a create that fails outright never deletes one either (Codex r1-3)."""
    name = f"{PROBE_PREFIX}{token()}.tmp"
    path = folder / name
    created = False
    try:
        with open(path, "xb") as handle:
            created = True
            handle.write(b"0")
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                return False, f"{WRITE_PROBLEM}; its test file {name} could not be removed either -- delete it by hand"
        return False, WRITE_PROBLEM
    try:
        path.unlink()
    except OSError:
        return True, f"modelroom could not remove its test file {name} from that folder; delete it by hand"
    return True, None


def _ask_folder(run: "GuidedRun") -> str | None:
    """The folder question: a form or write problem asks it again at a terminal, an empty answer
    ends the step without change, and an answer file gets `AnswerInvalidError` on the first problem
    it hits (the shape of `guided_folder._allowed_folder`, decided there and kept here)."""
    while True:
        answered = run.asker.text("share_folder", QUESTIONS["share_folder"]).strip()
        if not answered:
            return None
        expanded = str(Path(answered).expanduser())
        run.screen.answer(QUESTIONS["share_folder"], expanded, path=True)
        problem = _form_problem(expanded)
        if problem is None:
            run.screen.note(CHECKING_NOTE)
            ok, note = _write_probe(Path(expanded))
            if ok:
                if note is not None:
                    run.screen.note(note)
                return expanded
            problem = note
        run.screen.note(problem)
        if isinstance(run.asker, FileAsker):
            raise AnswerInvalidError(problem)


def _apply_none(run: "GuidedRun", store_path: Path, store: RemoteStore, environ: Mapping[str, str]) -> None:
    """`none`: while the environment wins, a stored folder is untouched -- this run alone skips it
    (Codex r1-2); only without the environment does `none` actually clear what is kept."""
    if env_value(environ) is not None:
        run.screen.note(NOT_THIS_RUN_NOTE)
        return
    if share_folder({}, store) is not None:
        set_profile_share(store_path, None)
        run.screen.note(REMOVED_NOTE)


def share_step(run: "GuidedRun", config_file: Path, config: Configuration) -> tuple[Configuration, set[str]]:
    """The whole "Shared folder" step: stand, the single choice, then whatever it leads to.

    Returns the configuration (changed only by an import) and the `profile_id`s this step chose --
    `_machines_step` folds them into the same "imported" bucket `_import_a_profile` fills, because
    that is exactly what they are (Herkunft `imported`, the bau-brief's own words).
    """
    store_path = remote_path_for(run.pointer_path)
    store = read_remote(store_path)
    for line in _stand_lines(run.room.environ, store):
        run.screen.note(line)
    choices = _setup_choices(run.room.environ, store)
    setup = run.asker.select("share_setup", QUESTIONS["share_setup"], choices)
    run.screen.answer(QUESTIONS["share_setup"], next(choice.label for choice in choices if choice.value == setup))
    if setup == "none":
        _apply_none(run, store_path, store, run.room.environ)
        return config, set()
    if setup == "choose":
        folder = _ask_folder(run)
        if folder is None:
            return config, set()
        set_profile_share(store_path, folder)
        run.screen.note(KEPT_NOTE)
        location = folder
    else:  # "use"
        location = share_folder(run.room.environ, store)
    return _import_from_folder(run, config_file, config, Path(location))


# --- reading the folder, and importing what was chosen --------------------------------------------


@dataclass(frozen=True)
class _ShareFile:
    path: Path
    profile: HardwareProfile


class ShareUnreadableError(Exception):
    """The shared folder itself could not be listed: missing, unreachable, or no permission."""


def _scan(folder: Path) -> tuple[list[_ShareFile], int]:
    """Every `*.json` directly in `folder` (never a subfolder), at most `MAX_FILES`, each read as an
    export of at most `MAX_BYTES`. Returns what read as one, and how many did not -- a file beyond
    `MAX_FILES` is left alone, neither counted nor read (`test_the_limits_skip_a_file`)."""
    try:
        if not folder.is_dir():
            raise ShareUnreadableError(f"{folder}: not a folder")
        names = sorted(path for path in folder.iterdir() if path.suffix == ".json" and path.is_file())
    except OSError as exc:
        raise ShareUnreadableError(str(exc)) from exc
    files: list[_ShareFile] = []
    bad = 0
    for path in names[:MAX_FILES]:
        try:
            export = read_export_file(path, max_bytes=MAX_BYTES)
        except (StoredFileError, SchemaVersionError):
            bad += 1
            continue
        files.append(_ShareFile(path, export.profile))
    return files, bad


def _by_profile(files: Sequence[_ShareFile]) -> dict[str, list[_ShareFile]]:
    grouped: dict[str, list[_ShareFile]] = {}
    for entry in files:
        grouped.setdefault(entry.profile.profile_id, []).append(entry)
    return grouped


def _safe_name(profile: HardwareProfile) -> str:
    """The name shown for `profile`: itself where it is already neutral, else the sink's own
    fallback (`neutral.display_name_for`, FR-N) -- a foreign export whose source forgot to
    neutralize never reaches a list, an answer line or a failure sentence with its real name
    (Codex r1-6). No alias applies here: the shared folder is not the SSH way."""
    return display_name_for(profile.profile_id, profile.machine_class, profile.display_name, None)


def _rows(grouped: dict[str, list[_ShareFile]]) -> list[Choice]:
    """One row per `profile_id`, described from the file with the newest `recorded_at`; the file
    names themselves never reach a label (`test_file_names_never_appear`)."""
    labeled = []
    for profile_id, entries in grouped.items():
        newest = max(entries, key=lambda entry: entry.profile.recorded_at).profile
        label = f"{_safe_name(newest)}  {_hardware_class(newest)}  recorded {newest.recorded_at.date().isoformat()}"
        labeled.append((label, profile_id))
    labeled.sort()
    return [Choice(profile_id, label) for label, profile_id in labeled]


def _bad_note(bad: int) -> str:
    return f"{bad} file{'' if bad == 1 else 's'} in the shared folder do not read as a profile export"


def _picked_words(picked: Sequence[str], grouped: dict[str, list[_ShareFile]]) -> str:
    names = [_safe_name(grouped[profile_id][0].profile) for profile_id in picked if profile_id in grouped]
    return ", ".join(names) if names else "none"


def _import_from_folder(run: "GuidedRun", config_file: Path, config: Configuration, folder: Path) -> tuple[Configuration, set[str]]:
    try:
        files, bad = _scan(folder)
    except ShareUnreadableError:
        run.failed(UNREACHABLE_NOTE)
        return config, set()
    if bad:
        run.screen.note(_bad_note(bad))
    grouped = _by_profile(files)
    if not grouped:
        run.screen.note(EMPTY_NOTE)
        return config, set()
    choices = _rows(grouped)
    picked = run.asker.checkbox("share_files", QUESTIONS["share_files"], choices)
    run.screen.answer(QUESTIONS["share_files"], _picked_words(picked, grouped))
    if not picked:
        return config, set()
    _import_chosen(run, config_file, grouped, picked)
    return _reload(config_file), set(picked)


def _import_chosen(run: "GuidedRun", config_file: Path, grouped: dict[str, list[_ShareFile]], picked: Sequence[str]) -> None:
    """Every readable export of every chosen `profile_id`, oldest first (Codex r2-5); one file's
    failure never stops the others, of this id or of the next one (`test_one_failing_file_does_not_stop_the_others`)."""
    for profile_id in picked:
        entries = sorted(grouped.get(profile_id, ()), key=lambda entry: (entry.profile.recorded_at, entry.path.name))
        name = _safe_name(entries[0].profile) if entries else profile_id
        for entry in entries:
            _import_one(run, config_file, entry.path, profile_id, name)


def _import_one(run: "GuidedRun", config_file: Path, path: Path, profile_id: str, name: str) -> None:
    """One file of the shared folder, through the unchanged `import_profile`; every exception it can
    raise becomes one of the three fixed sentences, none of them naming this file (`importer.py`'s
    own texts carry the path -- `test_import_errors_never_carry_the_path`)."""
    try:
        import_profile(
            config_file,
            path,
            run.now,
            pointer_path=run.pointer_path,
            hostname=run.probes.hostname(),
            expected_profile_id=profile_id,
            max_bytes=MAX_BYTES,
        )
    except StaleExportError:
        run.failed(CHANGED_NOTE)
    except ImportConflictError:
        run.failed(f"a file of {name} in the shared folder contradicts one imported before")
    except (StoredFileError, SchemaVersionError, ConfigError, PrerequisiteError, LockHeldError, OSError):
        run.failed(f"a file of {name} in the shared folder could not be imported")


def _reload(config_file: Path) -> Configuration:
    from .guided import GuidedError  # imported here: the guided mode imports this module in turn

    try:
        return load_config(config_file)
    except ConfigError as exc:
        raise GuidedError(str(exc)) from exc


__all__ = [
    "CHANGED_NOTE",
    "EMPTY_NOTE",
    "ENV_VAR",
    "FORM_PROBLEM",
    "KEPT_NOTE",
    "MAX_BYTES",
    "MAX_FILES",
    "NO_SHARE_ROW",
    "NONE_SET_YET",
    "NOT_THIS_RUN_NOTE",
    "PROBE_PREFIX",
    "QUESTIONS",
    "REMOVED_NOTE",
    "SET_BY_ENV",
    "UNREACHABLE_NOTE",
    "WITH_SHARE_ROW",
    "WRITE_PROBLEM",
    "ShareUnreadableError",
    "env_value",
    "machine_list_row",
    "share_folder",
    "share_step",
]
