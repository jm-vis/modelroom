"""Step 1's `ssh` row, the identity check of a stored SSH profile, and step 5's pull over SSH
(AP9-FR, FR-B1, B3-B5): a computer read over the system `ssh`, never a password, never a server's
own text on the screen.

One SSH target is active in a run (GF question 6, B4): a freshly entered alias, else the one
stored SSH profile chosen, else -- with several -- the answer to `ssh_active`. Every other chosen
SSH profile is ranked from what the folder already holds, without a connection
(`pull_other.pull_other_skip` already leaves it out of that list; this module never touches it).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .config import Configuration
from .contracts import SchemaVersionError
from .dialog import AnswerMissingError, Choice
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
from .profile import HardwareProfile
from .pull_choice import package_of, pull_rows, weights_bytes_of
from .remote_client import hello as ssh_hello
from .remote_client import profile as ssh_profile
from .remote_client import room as ssh_room
from .remote_store import RemoteStoreError, aliases as remote_aliases
from .remote_store import remote_path_for, set_alias
from .room import Disk, LiveRoom, gb, pull_need_bytes
from .room import UNCLEAR_REASON as _UNCLEAR
from .room import profile_cards as _profile_cards
from .screen import PULL_LABEL, pulled_line
from .ssh import SshError, is_alias
from .state import LockHeldError

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .document import RenderDocument
    from .guided import GuidedRun
    from .guided_install import PullContext

KEY_ALIAS = "ssh_alias"
KEY_CLASS = "ssh_machine_class"
KEY_CLONE = "ssh_clone"
KEY_ACTIVE = "ssh_active"
KEY_PULL = "pull_remote"

QUESTIONS: dict[str, str] = {
    KEY_ALIAS: "Host alias: your own short name for it in your SSH configuration",
    KEY_CLASS: "What kind of machine is it?",
    KEY_CLONE: "Is it the same machine or a new one?",
    KEY_ACTIVE: "Which computer over SSH should this run connect to?",
    KEY_PULL: "Pull it now?",
}

SSH_ROW = "a computer over SSH (its host alias)"
_FRESH_ATTEMPTS = 8
_ATTEMPTS_ERROR = "no unused profile_id after 8 attempts; the id source repeats itself"
_ANOTHER_MACHINE_NOTE = "{alias} is another machine now; its new profile counts from the next run"
_ID_MISMATCH = "{alias} answered in a form this modelroom does not read"


def _form_error(alias: str) -> str:
    return _ID_MISMATCH.format(alias=alias)


# --- B3: a new alias, entered in step 1 -----------------------------------------------------------


def _ask_alias(run: "GuidedRun") -> str:
    alias = run.asker.text(KEY_ALIAS, QUESTIONS[KEY_ALIAS]).strip()
    run.screen.answer(QUESTIONS[KEY_ALIAS], alias)
    return alias


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


def gather_arrivals(run: "GuidedRun", config_file: Path, config: Configuration, picked) -> tuple:
    """`import`, `ssh` (B3), `share` (FR-S) and `enter`, in machine-list order; each optional.
    `guided._machines_step`'s one call for step 1's second half. Returns the configuration and, per
    source, the profile ids that arrived -- `ssh` also the alias a freshly entered row chose (`None`
    for a stored one picked directly)."""
    from .guided import _import_a_profile  # imported here: guided.py imports this module in turn
    from .guided_entered import enter_step
    from .guided_share import share_step
    from .importer import scan_profiles

    imported = ssh_ids = shared = entered = set()
    ssh_alias = None
    if "import" in picked:
        config, imported = _import_a_profile(run, config_file, config)
    if "ssh" in picked:
        config, ssh_ids, ssh_alias = ssh_step(run, config_file, config, scan_profiles(config.paths.hardware_dir))
    if "share" in picked:
        config, shared = share_step(run, config_file, config)
    if "enter" in picked:
        before = set(scan_profiles(config.paths.hardware_dir).profiles)
        config = enter_step(run, config_file, config)
        entered = set(scan_profiles(config.paths.hardware_dir).profiles) - before
    return config, imported, ssh_ids, shared, entered, ssh_alias


def ssh_step(run: "GuidedRun", config_file: Path, config: Configuration, scan: ProfileScan) -> tuple[Configuration, set[str], str | None]:
    """Row `ssh` of the machine list (B3): alias, `hello`, the id (B4), the class question, `profile`,
    then the write order. Returns the configuration, the profile id this step chose (for `_origins`'
    "ssh" bucket) and the alias that becomes this run's active target -- `None` on any fault, before
    anything is written (`run.failed` already said why)."""
    alias = _ask_alias(run)
    if not is_alias(alias):
        run.failed(f"{alias!r}: a host alias is a short name of your own, like gpu1 (letters, digits, dashes)")
        return config, set(), None
    try:
        info = ssh_hello(alias, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return config, set(), None
    old_id = _resolve_profile_id(run, scan, alias)
    profile_id = old_id
    known = scan.profiles.get(profile_id)
    if _identity_changed(known, info["os_fingerprint"]):
        if _ask_clone(run, alias) == "new":
            profile_id = _fresh_id(run, scan)
            _rename_old_row(run, config_file, config, alias, old_id, scan)
    machine_class = _ask_class(run, alias)
    try:
        result = ssh_profile(alias, profile_id, machine_class, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return config, set(), None
    return _bind_and_import(run, config_file, config, alias, profile_id, result)


def _rename_old_row(run: "GuidedRun", config_file: Path, config: Configuration, alias: str, old_profile_id: str, scan: ProfileScan) -> None:
    """B4, "new": the row `[machines.<alias>]` still names the old profile -- moved to a neutral key
    (`neutral.key_for`, the profile's own `neutral_name`, else a free `computer-<8 hex>`) before the
    fresh id is imported exactly under `[machines.<alias>]` (Codex r6-1). Best effort: a failure here
    is not this step's own write order (B3) and does not stop it -- the next `sweep` still finds and
    fixes a row two profiles would otherwise share no key for."""
    from .config import ConfigError
    from .guided_write import update_config, neutral_change
    from .neutral import key_for
    from .neutral_sweep import load_names

    old = scan.profiles.get(old_profile_id)
    if old is None or alias not in config.machines:
        return
    taken = set(config.machines) - {alias}
    new_key = key_for(old_profile_id, old.machine_class, old.display_name, taken)

    def change(current: Configuration):
        if alias not in current.machines or new_key in current.machines:
            return None
        data = current.model_dump(mode="json")
        data["machines"][new_key] = data["machines"].pop(alias)
        return Configuration.from_dict(data)

    names = load_names(config_file.parent, run.pointer_path, run.probes.hostname())
    try:
        update_config(config_file, neutral_change(change, names), now=run.now)
    except (ConfigError, OSError):
        pass


def _bind_and_import(run: "GuidedRun", config_file: Path, config: Configuration, alias: str, profile_id: str, result: dict):
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
    except (StoredFileError, SchemaVersionError, ConfigError, PrerequisiteError, ImportConflictError, LockHeldError, OSError):
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


# --- B4: a stored SSH profile picked directly, and the one active target of the run ----------------


def active_target(run: "GuidedRun", config: Configuration, machines: Sequence[str], new_alias: str | None) -> str | None:
    """The alias this run connects to (B4, GF question 6): the alias just entered, else the one
    chosen stored SSH profile, else -- with several -- `ssh_active` (optional, `None` skips it)."""
    if new_alias is not None:
        return new_alias
    store_path = remote_path_for(run.pointer_path)
    known = remote_aliases(store_path)
    chosen = sorted({known[config.machines[m].profile] for m in machines if config.machines[m].profile in known})
    if len(chosen) <= 1:
        return chosen[0] if chosen else None
    choices = [Choice(alias, alias, checked=index == 0) for index, alias in enumerate(chosen)]
    try:
        answer = run.asker.select(KEY_ACTIVE, QUESTIONS[KEY_ACTIVE], choices)
    except AnswerMissingError:
        return None
    run.screen.answer(QUESTIONS[KEY_ACTIVE], answer)
    return answer


def verify_active(run: "GuidedRun", config: Configuration, config_file: Path, scan: ProfileScan, alias: str) -> bool:
    """B4, "verbindlich vor jeder aktiven Nutzung" (binding before every active use): a stored
    profile picked directly is checked against a live `hello` before room, pull or measurement ever
    use it. `True`: go on with `alias` (same profile, or a confirmed clone already re-imported under
    a fresh id -- the caller reads the folder again to see it). `False`: no further SSH action this run."""
    store_path = remote_path_for(run.pointer_path)
    profile_id = next((pid for pid, name in remote_aliases(store_path).items() if name == alias), None)
    if profile_id is None:
        return False
    try:
        info = ssh_hello(alias, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return False
    known = scan.profiles.get(profile_id)
    if not _identity_changed(known, info["os_fingerprint"]):
        return True
    if _ask_clone(run, alias) == "same":
        return True
    run.screen.note(_ANOTHER_MACHINE_NOTE.format(alias=alias))
    fresh_id = _fresh_id(run, scan)
    _rename_old_row(run, config_file, config, alias, profile_id, scan)
    machine_class = known.machine_class if known is not None and known.machine_class != "unknown" else None
    try:
        result = ssh_profile(alias, fresh_id, machine_class, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return False
    _bind_and_import(run, config_file, config, alias, fresh_id, result)
    return False  # this run's table still counts the old profile; nothing more is done with `alias`


def resolve_active(
    run: "GuidedRun", config: Configuration, config_file: Path, scan: ProfileScan, machines: Sequence[str], ssh_alias: str | None
) -> str | None:
    """`_machines_step`'s one call for B4: the active target (`active_target`), a freshly entered
    alias trusted outright, a stored one checked first (`verify_active`) -- a failed or refused
    check leaves this run's `ssh_active` `None`, and it touches that alias no further."""
    active = active_target(run, config, machines, ssh_alias)
    if active is None or ssh_alias is not None:
        return active
    return active if verify_active(run, config, config_file, scan, active) else None


# --- B5: the room and the pull of the active SSH machine, in step 5 --------------------------------


def active_room(run: "GuidedRun", document: "RenderDocument") -> dict:
    """`{run.ssh_active: LiveRoom}`, or `{}` without an active target or a table for it here --
    `guided.py`'s one call into this module for step 5's `live` mapping (U10, B5)."""
    if run.ssh_active is None or not any(block.machine == run.ssh_active for block in document.machines):
        return {}
    return {run.ssh_active: read_room(run, run.ssh_active, document)}


def read_room(run: "GuidedRun", alias: str, document: "RenderDocument") -> LiveRoom | None:
    """The room of the active target, read once for step 5 (B5): the disk (named `alias`, never a
    path) and the free graphics memory of its own cards. `disk.drive` always carries the alias
    (never a real drive letter or path), so `room_head._ssh_disk_piece` can name it in every case,
    known, unclear, or the connection failed outright."""
    profile = next((block.profile for block in document.machines if block.machine == alias), None)
    try:
        result = ssh_room(alias, _profile_cards(profile) if profile is not None else [], run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return LiveRoom(disk=Disk(None, alias, f"{alias} not reached"))
    disk_info = result["disk"]
    reason = _UNCLEAR if disk_info["unclear"] else None
    disk = Disk(disk_info["free_bytes"], alias, reason)
    graphics = result.get("graphics_free_mib")
    return LiveRoom(disk=disk, graphics_free_mib=tuple(graphics) if graphics else None)


def pull_remote_step(run: "GuidedRun", alias: str, context: "PullContext | None") -> None:
    """B5's own pull question for the active SSH machine's row: a fresh `room` reading (B5: "frische
    room-Lesung", never the one step 5's table already read), the digest proof over `SshDaemon`,
    then the pull itself -- the same shape `pull_step` uses for this machine, with `alias` in every
    message instead of nothing."""
    if context is None:
        return
    from .remote_client import SshDaemon

    rows = [row for row in pull_rows(context.document, alias, context.ambiguous, run.screen.glyphs.dot) if row.locked is None]
    if not rows:
        return
    row = rows[0]
    package = package_of(row, context.packages)
    if package is None or row.name is None:
        return
    daemon = SshDaemon(alias, run.ssh)
    weights_bytes = weights_bytes_of(package)
    if _held_on(daemon, row.name):
        run.screen.note(f"#{row.rank} is local already on {alias}")
        return
    live = read_room(run, alias, context.document)
    disk = live.disk if live is not None else None
    if disk is not None and disk.known and disk.free_bytes < pull_need_bytes(weights_bytes):
        run.screen.note(f"not enough disk on {alias} for #{row.rank}: needs {gb(pull_need_bytes(weights_bytes))}, {gb(disk.free_bytes)} free")
        return
    question = f"Pull #{row.rank} onto {alias} now? ({row.weights_gib:.1f} GB)"
    try:
        wants = run.asker.confirm(KEY_PULL, question, default=False)
    except AnswerMissingError:
        wants = False
    run.screen.answer(question, "Yes" if wants else "No")
    if wants:
        _pull_remote(run, alias, daemon, row.rank, row.name, row.weights_gib)


def _held_on(daemon, name: str) -> bool:
    from .loadtest import installed_models

    from datetime import datetime, timezone

    installed, _reason = installed_models(daemon, datetime.now(timezone.utc))
    return installed is not None and any(model.name == name for model in installed)


def _pull_remote(run: "GuidedRun", alias: str, daemon, rank: int, name: str, weights_gib: float) -> None:
    from .daemon import DaemonError, PULL_PATH
    from .intro import label_line

    run.screen.write(label_line(PULL_LABEL, [("", name)]))
    try:
        answer = daemon("POST", PULL_PATH, {"model": name, "stream": True})
    except DaemonError as exc:
        run.failed(f"the pull of {name} did not finish: {exc}")
        return
    body = json.loads(answer.body) if answer.body else {}
    if answer.status != 200 or body.get("status") != "success":
        run.failed(f"the pull of {name} did not finish on {alias}")
        return
    run.screen.write(pulled_line(name, weights_gib, run.screen.glyphs, hint=f"on {alias}"))


__all__ = [
    "KEY_ACTIVE",
    "KEY_ALIAS",
    "KEY_CLASS",
    "KEY_CLONE",
    "KEY_PULL",
    "QUESTIONS",
    "SSH_ROW",
    "active_room",
    "active_target",
    "gather_arrivals",
    "pull_remote_step",
    "read_room",
    "resolve_active",
    "ssh_step",
    "verify_active",
]
