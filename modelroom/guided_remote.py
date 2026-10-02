"""Step 1's `ssh` row, the identity check of a stored SSH profile, and step 5's pull over SSH
(AP9-FR, FR-B1, B3-B5): a computer read over the system `ssh`, never a password, never a server's
own text on the screen.

One SSH target is active in a run (GF question 6, B4): a freshly entered alias -- trusted outright,
whether it succeeds or fails, and never falls back to a stored one (Codex r1 finding 8: a failed
fresh attempt still "used the slot") -- else the one stored SSH profile chosen, else -- with
several -- the answer to `ssh_active`, which always offers an explicit `none`. Every other chosen
SSH profile is ranked from what the folder already holds, without a connection, and in
`pull_other`'s list its row is handed over like a computer without SSH (B4); a row of the active
target chosen there is pulled over SSH instead (`pull_remote_row`, B5).

B3's own mechanics -- binding a freshly entered alias to a profile id, and the write order of its
answer -- live in `guided_remote_bind.py` (Codex r2: split this module by topic once it grows too
large); this module re-exports the names a test or `guided.py` calls by themselves (`_ask_alias`,
`QUESTIONS`) and keeps every step's own entry point (`ssh_step`, `gather_arrivals`, B4, B5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .config import Configuration
from .dialog import AnswerMissingError, Choice
from .guided_remote_bind import KEY_ALIAS, KEY_CLASS, KEY_CLONE
from .guided_remote_bind import QUESTIONS as _BIND_QUESTIONS
from .guided_remote_vllm import remember_hello, with_model_disk
from .guided_remote_bind import (
    _ask_alias,
    _ask_class,
    _ask_clone,
    _bind_and_import,
    _fresh_id,
    _identity_changed,
    _resolve_profile_id,
)
from .importer import ProfileScan
from .remote_client import hello as ssh_hello
from .remote_client import profile as ssh_profile
from .remote_client import room as ssh_room
from .remote_store import aliases as remote_aliases
from .remote_store import remote_path_for
from .room import Disk, LiveRoom
from .room import UNCLEAR_REASON as _UNCLEAR
from .room import profile_cards as _profile_cards
from .ssh import SshError

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from pathlib import Path

    from .document import RenderDocument
    from .guided import GuidedRun
    from .guided_install import PullContext
    from .pull_choice import PullRow

KEY_ACTIVE = "ssh_active"
KEY_PULL = "pull_remote"

QUESTIONS: dict[str, str] = {
    **_BIND_QUESTIONS,
    KEY_ACTIVE: "Which computer over SSH should this run connect to?",
    KEY_PULL: "Pull it now?",
}

SSH_ROW = "a computer over SSH (its host alias)"
_ANOTHER_MACHINE_NOTE = "{alias} is another machine now; its new profile counts from the next run"
_ACTIVE_NONE = "none, no connection"


def gather_arrivals(run: "GuidedRun", config_file: "Path", config: Configuration, picked) -> tuple:
    """`import`, `ssh` (B3), `share` (FR-S) and `enter`, in machine-list order; each optional.
    `guided._machines_step`'s one call for step 1's second half. Returns the configuration and, per
    source, the profile ids that arrived -- `ssh` also the alias a freshly entered row chose (`None`
    for a stored one picked directly, or for a freshly entered one that failed)."""
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
    """Row `ssh` of the machine list (B3): alias, `hello`, the id (B4), the class question, `profile`.
    Returns the configuration, the profile id this step chose (for `_origins`' "ssh" bucket) and the
    alias that becomes this run's active target -- `None` on any fault, before anything is written
    (`run.failed` already said why). Nothing is written or renamed until `profile`'s answer is
    validated (Codex r1 finding 7): the questions alone never touch a file."""
    alias = _ask_alias(run)
    try:
        info = ssh_hello(alias, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return config, set(), None
    remember_hello(run, alias, info)  # its vLLM, live in this run (AP9-SV2-R, R3 step 1)
    profile_id = _resolve_profile_id(run, scan, alias)
    if _identity_changed(scan.profiles.get(profile_id), info["os_fingerprint"]) and _ask_clone(run, alias) == "new":
        profile_id = _fresh_id(run, scan)
    machine_class = _ask_class(run, alias)
    try:
        result = ssh_profile(alias, profile_id, machine_class, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return config, set(), None
    return _bind_and_import(run, config_file, config, alias, profile_id, result)


# --- B4: a stored SSH profile picked directly, and the one active target of the run ----------------


def active_target(run: "GuidedRun", config: Configuration, machines: Sequence[str]) -> str | None:
    """The alias of the one **stored** SSH profile this run connects to (B4, GF question 6), for a
    run that entered no fresh alias itself: none chosen, `None`; exactly one, it; several, the
    answer to `ssh_active` -- always offering an explicit `none` (Codex r1 finding 8), pointer on
    the first alias, `None` where the question is skipped or answered `none`."""
    store_path = remote_path_for(run.pointer_path)
    known = remote_aliases(store_path)
    chosen = sorted({known[config.machines[m].profile] for m in machines if config.machines[m].profile in known})
    if len(chosen) <= 1:
        return chosen[0] if chosen else None
    choices = [Choice(alias, alias, checked=index == 0) for index, alias in enumerate(chosen)]
    choices.append(Choice("none", _ACTIVE_NONE))
    try:
        answer = run.asker.select(KEY_ACTIVE, QUESTIONS[KEY_ACTIVE], choices)
    except AnswerMissingError:
        return None
    run.screen.answer(QUESTIONS[KEY_ACTIVE], next(choice.label for choice in choices if choice.value == answer))
    return None if answer == "none" else answer


def verify_active(run: "GuidedRun", config: Configuration, config_file: Path, scan: ProfileScan, alias: str) -> bool:
    """B4, binding before every active use: a stored
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
    remember_hello(run, alias, info)  # its vLLM, live in this run (AP9-SV2-R, R3 step 1)
    known = scan.profiles.get(profile_id)
    if not _identity_changed(known, info["os_fingerprint"]):
        return True
    if _ask_clone(run, alias) == "same":
        return True
    run.screen.note(_ANOTHER_MACHINE_NOTE.format(alias=alias))
    fresh_id = _fresh_id(run, scan)
    machine_class = known.machine_class if known is not None and known.machine_class != "unknown" else None
    try:
        result = ssh_profile(alias, fresh_id, machine_class, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return False
    _bind_and_import(run, config_file, config, alias, fresh_id, result)
    return False  # this run's table still counts the old profile; nothing more is done with `alias`


def resolve_active(
    run: "GuidedRun", config: Configuration, config_file: Path, scan: ProfileScan, machines: Sequence[str],
    ssh_alias: str | None, ssh_attempted: bool,
) -> str | None:
    """`_machines_step`'s one call for B4: a run that entered a fresh alias this step (`ssh_attempted`)
    stands or falls with it alone -- `ssh_alias` if it succeeded, `None` if it failed -- and never
    falls back to a stored target (Codex r1 finding 8: the run's one SSH "slot" was already spent on
    the attempt, whether or not it worked). Only a run that entered no fresh alias at all considers a
    stored one, checked first (`verify_active`)."""
    if ssh_attempted:
        return ssh_alias
    active = active_target(run, config, machines)
    if active is None:
        return None
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
    disk = with_model_disk(run, alias, Disk(disk_info["free_bytes"], alias, reason), result.get("model_disk"))  # vLLM's place (R3 step 5)
    graphics = result.get("graphics_free_mib")
    return LiveRoom(disk=disk, graphics_free_mib=tuple(graphics) if graphics else None)


def pull_remote_row(run: "GuidedRun", alias: str, row: "PullRow", context: "PullContext") -> None:
    """B5: the row of the active SSH machine chosen in `pull_other`'s list (A3) -- whichever rank it
    is -- goes through the unchanged `PullTarget`/`pull_chosen` chain, not a separate implementation:
    the install line `#<rank> on <alias>  ollama pull <name>`, the digest proof (`_is_local`/`_holds`,
    over `SshDaemon`), a fresh disk reading (`target.disk`, called right when it is checked), the
    question (key `pull_remote`), and the pull with its progress/error/completion handling
    (`_send`/`_check_line`/`_end_reason`/`pulled_problem`), `target.where=alias` naming the machine."""
    from .guided_install import FirstRow, PullTarget, pull_chosen
    from .pull_choice import package_of, weights_bytes_of
    from .remote_client import SshDaemon

    package = package_of(row, context.packages)
    if package is None or row.name is None:
        return
    first = FirstRow(row.rank, str(row.name), row.weights_gib, package, weights_bytes_of(package))
    target = PullTarget(
        daemon=SshDaemon(alias, run.ssh),
        disk=lambda: _fresh_disk(run, alias, context.document),
        now=run.now,
        screen=run.screen,
        asker=run.asker,
        failed=run.failed,
        where=alias,
        key=KEY_PULL,
    )
    pull_chosen(target, first)


def _fresh_disk(run: "GuidedRun", alias: str, document: "RenderDocument") -> Disk:
    live = read_room(run, alias, document)
    if live is not None and live.disk is not None:
        return live.disk
    return Disk(None, alias, f"{alias} not reached")


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
    "pull_remote_row",
    "read_room",
    "resolve_active",
    "ssh_step",
    "verify_active",
]
