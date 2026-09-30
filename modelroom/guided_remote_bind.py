"""B3 (AP9-FR, FR-B1): binding a freshly entered SSH alias to a profile id, and the write order of
its answer -- the alias question, the machine-class question, the "same or new machine" question
where a stored fingerprint changed, and the validate-then-write sequence that turns a `profile`
answer into `remote.json`'s alias mapping and an imported profile.

Split out of `guided_remote.py` (Codex r2: split this module by topic once it grows too large):
B3's own mechanics, used by `guided_remote.ssh_step` (a fresh alias) and `guided_remote.verify_active`
(a stored one whose fingerprint changed) alike -- both call `_bind_and_import` to write the same way.
Nothing here is a step's own entry point; `guided_remote.py` keeps every name `guided.py` or a test
calls by itself.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Configuration
from .contracts import SchemaVersionError
from .dialog import AnswerInvalidError, Choice, FileAsker
from .guided_entered import CLASS_CHOICES
from .importer import (
    ConfigError,
    ImportConflictError,
    ProfileScan,
    PrerequisiteError,
    StoredFileError,
)
from .importer import import_profile as _import_profile
from .measurements import ExportObject
from .neutral import NeutralError
from .profile import HardwareProfile
from .remote_store import RemoteStoreError, aliases as remote_aliases
from .remote_store import remote_path_for, set_alias
from .ssh import INVALID_ALIAS, is_alias
from .state import LockHeldError

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

KEY_ALIAS = "ssh_alias"
KEY_CLASS = "ssh_machine_class"
KEY_CLONE = "ssh_clone"
QUESTIONS: dict[str, str] = {
    KEY_ALIAS: "Host alias: your own short name for it in your SSH configuration",
    KEY_CLASS: "What kind of machine is it?",
    KEY_CLONE: "Is it the same machine or a new one?",
}

_FRESH_ATTEMPTS = 8
_ATTEMPTS_ERROR = "no unused profile_id after 8 attempts; the id source repeats itself"
_ID_MISMATCH = "{alias} answered in a form this modelroom does not read"


def _form_error(alias: str) -> str:
    return _ID_MISMATCH.format(alias=alias)


def _ask_alias(run: "GuidedRun") -> str:
    """The alias, checked *before* anything about it ever reaches the screen (Codex r1 finding 6):
    an invalid one is never echoed and never repeated in a message, only the one fixed sentence
    (`ssh.INVALID_ALIAS`) -- a terminal asks again, `--answers` ends the run with `AnswerInvalidError`
    (exit 2), exactly as B1's own table says."""
    while True:
        alias = run.asker.text(KEY_ALIAS, QUESTIONS[KEY_ALIAS]).strip()
        if is_alias(alias):
            run.screen.answer(QUESTIONS[KEY_ALIAS], alias)
            return alias
        if isinstance(run.asker, FileAsker):
            raise AnswerInvalidError(INVALID_ALIAS)
        run.screen.note(INVALID_ALIAS)


def _ask_class(run: "GuidedRun", alias: str) -> str | None:
    question = f"What kind of machine is {alias}?"
    choices = [Choice(value, words, checked=value == "server") for value, words in CLASS_CHOICES.items()]
    answer = run.asker.select(KEY_CLASS, question, choices)
    run.screen.answer(question, next(choice.label for choice in choices if choice.value == answer))
    return None if answer == "unknown" else answer


def _ask_clone(run: "GuidedRun", alias: str) -> str:
    question = f"{alias} now answers with another machine than before. Is it the same machine or a new one?"
    choices = [Choice("same", "the same machine"), Choice("new", "a new machine", checked=True)]
    answer = run.asker.select(KEY_CLONE, question, choices)
    run.screen.answer(question, next(choice.label for choice in choices if choice.value == answer))
    return answer


def _fresh_id(run: "GuidedRun", scan: ProfileScan) -> str:
    from .guided import GuidedError  # imported here: the guided mode imports this module in turn

    for _ in range(_FRESH_ATTEMPTS):
        candidate = run.probes.new_id()
        if candidate not in scan.profiles:
            return candidate
    raise GuidedError(_ATTEMPTS_ERROR)


def _resolve_profile_id(run: "GuidedRun", scan: ProfileScan, alias: str) -> str:
    """B4: the id `alias` already stands for in this folder, when that profile is still there, else
    a fresh one (`_fresh_id`) -- the id `profile` (B2) is asked to measure under."""
    stored = remote_aliases(remote_path_for(run.pointer_path))
    existing = next((profile_id for profile_id, name in stored.items() if name == alias), None)
    return existing if existing is not None and existing in scan.profiles else _fresh_id(run, scan)


def _identity_changed(known: HardwareProfile | None, live_fingerprint: str) -> bool:
    """B4: a stored profile's own fingerprint differs from the one `hello` reads now -- `none` on
    either side never counts as a difference (there is nothing to compare)."""
    if known is None or known.os_fingerprint == "none" or live_fingerprint == "none":
        return False
    return known.os_fingerprint != live_fingerprint


def _bind_and_import(run: "GuidedRun", config_file: Path, config: Configuration, alias: str, profile_id: str, result: dict):
    """The write order (B3, B4): the answer is validated fully *first*; then (1) `remote.json` maps
    `profile_id` to `alias`, atomically; then (2) the import, which renames and plans under one lock.

    B4, `new`: the old profile's row `[machines.<alias>]` must move to a neutral key before the fresh
    id is imported exactly under `[machines.<alias>]`. Nothing extra is written for that here: once
    (1) moved the alias to the fresh id, the old row's key is no longer an alias of its own profile,
    so the import's own sweep (`neutral_sweep.sweep_locked`, the first thing `import_profile` does
    under its lock) renames it to the old profile's neutral name -- or a free `computer-<8 hex>`
    where that is taken (`neutral.machine_keys`) -- before `plan_import` reads the rows. A failure
    of that rename is a failure of this import: nothing of the profile is written, and `remote.json`
    points at a profile not (yet) in the folder, which the next run with the same alias converges
    (B3)."""
    try:
        export = ExportObject.model_validate(result["export"])
    except Exception:  # noqa: BLE001 - any shape mismatch is one fixed sentence, never a traceback
        run.failed(_form_error(alias))
        return config, set(), None
    if export.profile.profile_id != profile_id or export.measurements:
        run.failed(_form_error(alias))
        return config, set(), None
    store_path = remote_path_for(run.pointer_path)
    try:
        set_alias(store_path, profile_id, alias)
    except RemoteStoreError as exc:
        run.failed(str(exc))
        return config, set(), None
    updated = _import_export(run, config_file, alias, profile_id, export)
    return updated, {profile_id}, alias


def _import_export(run: "GuidedRun", config_file: Path, alias: str, profile_id: str, export: ExportObject) -> Configuration:
    """The write step B3 leaves to the unchanged `import_profile` (B3's own file, temp, deleted
    after): a failure here leaves `remote.json` pointing at a profile not yet on disk, which a later
    run's import (same alias) writes in full -- convergence, no rollback (B3)."""
    handle, raw_path = tempfile.mkstemp(suffix=".json")
    tmp = Path(raw_path)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as opened:
            opened.write(json.dumps(export.model_dump(mode="json")))
        _import_profile(
            config_file, tmp, run.now, name=alias, pointer_path=run.pointer_path, hostname=run.probes.hostname(),
            expected_profile_id=profile_id,
        )
    except (StoredFileError, SchemaVersionError, ConfigError, PrerequisiteError, ImportConflictError, LockHeldError, NeutralError, OSError):
        run.failed(f"the profile of {alias} could not be imported")
    finally:
        tmp.unlink(missing_ok=True)
    return _reload(config_file)


def _reload(config_file: Path) -> Configuration:
    from .config import load_config
    from .guided import GuidedError

    try:
        return load_config(config_file)
    except ConfigError as exc:
        raise GuidedError(str(exc)) from exc


__all__ = [
    "KEY_ALIAS",
    "KEY_CLASS",
    "KEY_CLONE",
    "QUESTIONS",
]
