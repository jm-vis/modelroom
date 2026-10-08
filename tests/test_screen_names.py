"""AP9-FR-N3: a line of the guided screen that names a machine says the name its profile gives it.

The key of a machine (`[machines.<key>]`) is the internal token of a run; the screen says
`display_name`, as the card, the table head and step 1 do (AP9-FR-N2). These tests hold the lines
of the guided screen below the instruction line of the scale (`tests/test_guided_context.py`) and
the full run (`tests/test_guided_screen.py`): the `local` line of a table, the list that offers a
package for another computer and its handover. Outside the guided screen nothing changes: the
Markdown document and the plain terminal view (`modelroom render`, a log) keep the key.

The synthetic document has two machines whose keys differ from their names, so that a line that
still prints the key is told from one that prints the name.

This file does not import `modelroom.screen_names`: its tests must be collectable, and fail on the
key they see, against the unchanged production code of the basis. The helpers have their own tests
(`tests/test_screen_names_helpers.py`).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelroom.document import RenderDocument
from modelroom.guided_install import PullContext
from modelroom.pull_choice import ambiguous_names, weights_bytes_of
from modelroom.pull_other import KEY, other_rows, pull_other_step
from modelroom.quantization import package_identity_key
from modelroom.room import Disk, LiveRoom
from modelroom.room_head import RENDER_DISK
from modelroom.screen import Screen, plain
from modelroom.views import RunView, document_markdown, show_result, terminal_blocks, terminal_lines

from fixture_rows import document as document_one, document_multi, package
from test_pull_other import NAME_Q4, NAME_Q8, UNSLOTH, _Recording, _run

NOW = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
HERE_KEY, HERE_NAME = "computer-aaaaaaaa", "server-bbbbbbbb"
OTHER_KEY, OTHER_NAME = "computer-cccccccc", "laptop-dddddddd"
Q4 = package(UNSLOTH, "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
Q8 = package(UNSLOTH, "Nova-7B-Q8_0.gguf", "Q8_0")
LOCAL_OTHER = "local     #1 already in Ollama on "


def _document(*, other_profile: bool = True, other_reason: str | None = None) -> RenderDocument:
    """Two machines, keys `computer-<8 hex>`, names `server-<8 hex>` and `laptop-<8 hex>`; the second one
    without a resolved profile where `other_profile` is false (the key is then its only name)."""
    document = document_multi({HERE_KEY: [Q4], OTHER_KEY: [Q8]})
    first, second = document.machines
    first = first.model_copy(update={"label": HERE_NAME, "profile": first.profile.model_copy(update={"display_name": HERE_NAME, "machine_class": "server"})})
    if other_profile:
        second = second.model_copy(update={"label": OTHER_NAME, "profile": second.profile.model_copy(update={"display_name": OTHER_NAME, "machine_class": "laptop"})})
    else:
        second = second.model_copy(update={"profile": None, "status": "no_profile", "reason": other_reason, "label": OTHER_KEY})
    return document.model_copy(update={"machines": [first, second]})


def _live() -> dict[str, LiveRoom]:
    """Ollama holds row #1 of each machine; both disks were read (as a guided run reads them)."""
    return {
        HERE_KEY: LiveRoom(disk=Disk(10**12, "C:", held=frozenset({package_identity_key(Q4)}))),
        OTHER_KEY: LiveRoom(disk=Disk(10**12, "gpu", held=frozenset({package_identity_key(Q8)}))),
    }


def _screen_of_a_run(document: RenderDocument, view: RunView) -> list[str]:
    """What `show_result` draws -- the one entry of the guided screen's step 5 -- as plain text lines."""
    lines: list[str] = []
    facts = SimpleNamespace(model_names=("Nova-7B",), packages=2, accounts=("unsloth",))
    config = SimpleNamespace(paths=SimpleNamespace(markdown=Path("docs") / "models.md"))
    show_result(Screen(lines.append, colored=False), document, facts, config, Path("results"), NOW, HERE_KEY, (), view)
    return [line.strip() for line in lines]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unresolved_document(tmp_path: Path, configured: str | None) -> RenderDocument:
    """`_document()` whose second machine goes the render's own way to a block without a profile: its row of the
    configuration (`profile` = `configured`, or none), `scan_profiles` of a results folder that holds the first
    machine's profile only, `machine_profile`, `_machine_block`. Nothing of the block is written by hand."""
    from modelroom.config import MachineConfig
    from modelroom.importer import scan_profiles
    from modelroom.render import _machine_block, machine_profile
    from modelroom.state import atomic_write_json

    document = _document()
    hardware = tmp_path / "hardware"
    here = document.machines[0].profile
    atomic_write_json(hardware / f"{here.profile_id}.json", here.model_dump(mode="json"))
    row = MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=False, profile=configured)
    found = machine_profile(OTHER_KEY, row, scan_profiles(hardware))
    block = _machine_block(OTHER_KEY, row, found, [], {}, [], document.scenario)
    assert block.profile is None and block.status == "no_profile" and block.machine == OTHER_KEY  # the render's own outcome
    return document.model_copy(update={"machines": [document.machines[0], block]})


UNKNOWN_PROFILE = "0123456789abcdef"
NO_PROFILE = pytest.mark.parametrize(
    "configured, reason",
    [(None, "profile"), (UNKNOWN_PROFILE, f"the configured profile {UNKNOWN_PROFILE} is not in this results folder")],
    ids=["entry without a profile", "profile not in the folder"],
)


# --- 2: the `local` line, in three outputs -----------------------------------------------------------


def test_the_local_line_of_another_machine_says_its_name_in_the_guided_screen():
    """2 (a). The second machine is not `install_for`, so its line names it; the key stays out of the screen."""
    text = _screen_of_a_run(_document(), RunView(live=_live()))

    assert LOCAL_OTHER + OTHER_NAME in text
    assert LOCAL_OTHER + OTHER_KEY not in text
    assert "local     #1 already in Ollama on this computer" in text  # install_for keeps its contextual words
    assert not any(OTHER_KEY in line or HERE_KEY in line for line in text)


def _stored_ssh_document() -> RenderDocument:
    """`_document()` whose second machine is a stored SSH computer: alias `gpu9` as its name, a neutral key."""
    document = _document()
    second = document.machines[1]
    second = second.model_copy(update={"label": SSH_ALIAS, "profile": second.profile.model_copy(update={"display_name": SSH_ALIAS})})
    return document.model_copy(update={"machines": [document.machines[0], second]})


SSH_ALIAS = "gpu9"
SSH_TERMINAL_SHA = "273f7a43e37e67944255c488eb29632b082b409815416d30700889755b24e5f2"
SSH_MARKDOWN_SHA = "fcc9ac5c35bce081f5fa48b9286099681ba50dc1802122737bd0940051479a41"
SSH_ORIGINS = {HERE_KEY: "this", OTHER_KEY: "ssh"}  # origin `ssh`; `ssh_active` stays None: not connected in this run


def test_a_stored_ssh_computer_not_connected_in_this_run_is_called_by_its_name_in_the_room_line():
    """AP9-FR-N3, SSH exception (finding 5). A computer read over SSH is called by its alias: the profile's `display_name`.
    Its key is that alias where the import could give it (`neutral.key_for`), but not where the alias was taken or the
    profile was found in the folder without a row (`guided_write.with_found_profiles`): the key is then a neutral one.
    The `room` line of such a computer, not connected in this run, says the name, never the key."""
    text = _screen_of_a_run(_stored_ssh_document(), RunView(live=_live(), origins=SSH_ORIGINS))

    assert f"disk unknown ({SSH_ALIAS} not connected in this run)" in "\n".join(text)
    assert not any(OTHER_KEY in line or HERE_KEY in line for line in text)


def test_the_name_in_the_room_line_of_a_stored_ssh_computer_reaches_only_the_guided_screen():
    """fn3-2, finding 1. The same stored SSH computer (origin `ssh`, key != name, not the active one, so the guided
    screen's `where` is `ssh_inactive`) through three outputs: the guided screen says the name; `terminal_lines`
    (`modelroom render`, the logs; no live reading, so `render`) and the Markdown document are byte for byte
    what `c5714d1` produced (the hashes were taken from the unchanged `git archive c5714d1 modelroom`)."""
    document = _stored_ssh_document()
    view = RunView(live=_live(), origins=SSH_ORIGINS)
    assert view.ssh_active is None and view.origins[OTHER_KEY] == "ssh"  # the fixture: a stored SSH computer, not connected

    guided = _screen_of_a_run(document, view)
    assert any(f"disk unknown ({SSH_ALIAS} not connected in this run)" in line for line in guided)
    assert not any(f"disk unknown ({OTHER_KEY} " in line for line in guided)

    plain_lines = terminal_lines(document, HERE_KEY)
    assert [line.strip() for line in plain_lines if "disk unknown" in line] == [RENDER_DISK, RENDER_DISK]  # one per machine
    assert not any("not connected" in line for line in plain_lines)  # the `render` words: no place named, no name, no key
    assert _sha("\n".join(plain_lines)) == SSH_TERMINAL_SHA  # bytes at c5714d1
    assert _sha(document_markdown(document)) == SSH_MARKDOWN_SHA  # bytes at c5714d1


def test_the_plain_terminal_view_keeps_the_key_in_the_local_line():
    """2 (b), kept. A view that carries no `shown` (every caller but the guided screen) is unchanged."""
    document = _document()
    blocks = [plain(line).strip() for line in terminal_blocks(document, HERE_KEY, view=RunView(install_for=HERE_KEY, live=_live()))]

    assert LOCAL_OTHER + OTHER_KEY in blocks
    assert LOCAL_OTHER + OTHER_NAME not in blocks
    # `modelroom render` and the logs: no live reading, so no `local` line, and the same bytes as the default view.
    plain_lines = terminal_lines(document, HERE_KEY)
    assert plain_lines == [plain(line) for line in terminal_blocks(document, HERE_KEY, view=RunView(install_for=HERE_KEY))]
    assert not any("already in Ollama" in line for line in plain_lines)
    assert _sha("\n".join(plain_lines)) == "3e714c878b636cb6445fdde1435701601b398a3598025cb45e1b3d07d768e5b8"  # bytes at c5714d1


def test_the_markdown_document_keeps_the_key_in_its_ranking_heading():
    """2 (c), kept: `## Ranking: <key>` and the head line under it, byte for byte as at c5714d1."""
    markdown = document_markdown(_document())

    assert f"## Ranking: {HERE_KEY}" in markdown and f"## Ranking: {OTHER_KEY}" in markdown
    assert _sha(markdown) == "1c5270db7e30dd948902cb0bdcc2c20060c81e603c384400ab79aa5b1ad1cf3e"  # bytes at c5714d1


# --- 3: no profile outside the scale ----------------------------------------------------------------


@NO_PROFILE
def test_a_machine_without_a_profile_keeps_its_key_on_the_screen(tmp_path, configured, reason):
    """3. The block comes out of the render's own resolution (a configuration row without `profile`, a row that names a
    profile the results folder does not hold); the screen then keeps the key -- no name made up, no exception."""
    document = _unresolved_document(tmp_path, configured)
    assert reason in document.machines[1].reason

    text = _screen_of_a_run(document, RunView(live=_live()))  # no exception

    assert any(line.startswith(f"{OTHER_KEY} ") and line.endswith("no_profile") for line in text)  # the head of its block: the key stays
    assert any(line.startswith(f"machine   {OTHER_KEY} ") and reason in line for line in text)  # the card row, with the reason
    assert not any(OTHER_NAME in line for line in text)
    assert [line for line in text if line.startswith("local")] == ["local     #1 already in Ollama on this computer"]  # a block with no rows has none
    assert not any(HERE_KEY in line for line in text)


# --- 4: the key stays the identity ---------------------------------------------------------------------


def test_lookups_and_the_install_line_still_go_by_the_key():
    """The live reading is found by the key (a view keyed by the name finds none); the install line is `install_for`'s."""
    document = _document()
    by_name = {HERE_NAME: _live()[HERE_KEY], OTHER_NAME: _live()[OTHER_KEY]}

    keyed = _screen_of_a_run(document, RunView(live=_live()))
    named = _screen_of_a_run(document, RunView(live=by_name))

    assert any(line.startswith("local") for line in keyed)
    assert not any(line.startswith("local") for line in named)
    # Each block installs its own package: `install_for`'s block the Q4 of this machine, with no place named; the
    # other block its Q8, with the label it already says. Neither carries the command of the other machine.
    installs = [line for line in keyed if line.startswith("install")]
    assert installs == [f"install   #1  ollama pull {NAME_Q4}", f"install   #1 on {OTHER_NAME}  ollama pull {NAME_Q8}"]


# --- the list for another computer and its handover ----------------------------------------------------


def _context(document: RenderDocument) -> PullContext:
    packages = [Q4, Q8]
    return PullContext(document, HERE_KEY, packages, ambiguous_names(packages))


def test_the_list_for_another_computer_names_the_machine_by_its_name_and_answers_by_its_key(tmp_path: Path):
    document = _document()
    asker = _Recording({KEY: f"{OTHER_KEY} {NAME_Q8}"})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, _context(document))

    choices = asker.lists[KEY]
    assert [choice.label for choice in choices if choice.heading] == [OTHER_NAME]
    assert [choice.value for choice in choices if not choice.heading][0] == f"{OTHER_KEY} {NAME_Q8}"  # the answer file's value: the key
    assert other_rows(document, HERE_KEY)[0].machine == OTHER_KEY  # the identity of the row


def test_the_chosen_row_and_its_handover_name_the_machine_by_its_name(tmp_path: Path):
    document = _document()
    asker = _Recording({KEY: f"{OTHER_KEY} {NAME_Q8}"})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, _context(document))

    text = [line.strip() for line in lines]
    assert any(line.endswith(f"{OTHER_NAME}  #1  {NAME_Q8}") for line in text)  # the answer line
    handover = next(line for line in text if "modelroom pull" in line)
    assert handover.startswith(f"install   #1 on {OTHER_NAME}  ") and f"--weights-bytes {round(weights_bytes_of(Q8))}" in handover
    assert not any(OTHER_KEY in line or HERE_KEY in line for line in text)


DECOY_KEY, DECOY_NAME = "computer-eeeeeeee", "decoy-ffffffff"


def _rekeyed(document: RenderDocument, key: str, name: str) -> RenderDocument:
    """`document` with its one machine under `key`, its profile naming it `name` (as the render does for the label),
    and in front of it a block under another key and another name that ranks nothing: a lookup that takes the first
    block finds no package of the row there, so it yields no command (or another one) and the test fails."""
    block = document.machines[0]
    decoy = block.model_copy(
        update={
            "machine": DECOY_KEY,
            "label": DECOY_NAME,
            "profile": block.profile.model_copy(update={"display_name": DECOY_NAME}),
            "ranked": [],
            "ranked_total": 0,
            "not_covered": [],
            "no_fit": [],
        }
    )
    block = block.model_copy(update={"machine": key, "label": name, "profile": block.profile.model_copy(update={"display_name": name})})
    return document.model_copy(update={"machines": [decoy, block]})


def test_the_vllm_handover_names_the_machine_by_its_name_and_finds_its_block_by_the_key(tmp_path: Path):
    """A vLLM row of another computer: both commands unchanged, the place says the name, the key is nowhere."""
    from test_install_vllm import DOCKER, REPO, SERVE, _both, _gguf_package, _package, _run as _vllm_run

    document = _rekeyed(_both(), OTHER_KEY, OTHER_NAME)
    packages = [_gguf_package(), _package()]
    asker = _Recording({KEY: f"{OTHER_KEY} vllm {REPO}"})
    run, lines = _vllm_run(tmp_path, asker)

    pull_other_step(run, PullContext(document, HERE_KEY, packages, ambiguous_names(packages)))

    text = "\n".join(lines)
    rows = [line.strip() for line in lines]
    assert f"install   #2 on {OTHER_NAME}  {SERVE}" in rows  # the block was found by the key; the place says the name
    assert f"docker    {DOCKER}" in rows  # the second command, whole and unchanged
    assert OTHER_KEY not in text and HERE_KEY not in text and DECOY_KEY not in text and DECOY_NAME not in text
    assert "modelroom pull" not in text and "ollama pull" not in text


def test_the_lm_studio_handover_names_the_machine_by_its_name_and_keeps_the_command(tmp_path: Path):
    """An LM Studio row of another computer: the `lms get` line unchanged, the place says the name, no key."""
    from test_lmstudio_runs import QUANT, _q4
    from test_runtimes import _on_machine

    q4 = _q4()
    document = _rekeyed(_on_machine(document_one([q4]), ["lmstudio"]), OTHER_KEY, OTHER_NAME)
    asker = _Recording({KEY: f"{OTHER_KEY} lmstudio {QUANT}"})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, PullContext(document, HERE_KEY, [q4], ambiguous_names([q4])))

    text = [line.strip() for line in lines]
    assert f"install   #1 on {OTHER_NAME}  lms get --gguf {QUANT}" in text  # the command is the one of the basis
    assert not any(key in line for key in (OTHER_KEY, HERE_KEY, DECOY_KEY, DECOY_NAME) for line in text)
    assert not any("modelroom pull" in line or "ollama pull" in line for line in text)
