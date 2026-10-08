"""AP9-LS-E, round 1 of the fixes: the choice of a runtime that is not Ollama's, the width of what a chosen row
prints, and whole renders and runs in place of prepared data (review findings 1, 2, 3 and 5).

Every test goes through the real functions of a render or a guided step; none starts a process, none reaches
a server. A fake `ssh` stands in for the connection of the SSH test (`fixture_ssh.FakeSsh`).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from modelroom import runtimes
from modelroom.disk_fit import entry_over_disk, entry_place, place
from modelroom.guided_install import PullContext, pull_step
from modelroom.pull_choice import ambiguous_names, pull_rows
from modelroom.pull_other import pull_other_step
from modelroom.room import GIB, Disk, LiveRoom, free_disk
from modelroom.screen import plain
from modelroom.views import RunView, document_markdown, terminal_blocks

from fixture_rows import document, package
from test_pull_choice import _Recording, _run
from test_runtimes import _on_machine, _third

HERE = "workstation"
QUANT = "unsloth/Nova-7B-GGUF@Q4_K_M"


def _q4():
    return package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")


# --- finding 1: S1, every choice that is not Ollama's is printed, never pulled ---------------------------------


def test_a_third_runtime_choice_prints_its_command_and_pulls_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(runtimes, "RUNTIMES", (*runtimes.RUNTIMES, _third()))
    q4 = _q4()
    doc = _on_machine(document([q4]), ["ollama", "third"])
    asker = _Recording({"pull_package": "third acme/nova:Q4_K_M", "pull": True})
    run, lines = _run(tmp_path, asker)

    pull_step(run, PullContext(doc, HERE, [q4], ambiguous_names([q4])), True)

    said = [line.strip() for line in lines]
    assert "install   #1  third get acme/nova:Q4_K_M" in said
    assert "pull" not in asker.asked  # the yes/no question of a pull is Ollama's alone
    assert not any(call[0] == "POST" for call in run.daemon.calls)
    assert not any("pulling" in line for line in said)


def test_a_third_runtime_choice_the_runtime_holds_already_says_so_and_prints_no_command(tmp_path, monkeypatch):
    from modelroom.quantization import package_identity_key

    q4 = _q4()
    held = frozenset({package_identity_key(q4)})
    monkeypatch.setattr(runtimes, "RUNTIMES", (*runtimes.RUNTIMES, _third(on_disk=lambda target, packages: held)))
    doc = _on_machine(document([q4]), ["ollama", "third"])
    run, lines = _run(tmp_path, _Recording({"pull_package": "third acme/nova:Q4_K_M"}))

    pull_step(run, PullContext(doc, HERE, [q4], ambiguous_names([q4])), True)

    said = [line.strip() for line in lines]
    assert any(line.startswith("install   #1 is local already") for line in said)
    assert not any("third get" in line for line in said)


def test_an_explicit_runtime_is_judged_at_its_own_place_only():
    """S7: with `runtime` given, exactly that runtime's place decides -- a runtime of the table with no place of its
    own (`place` is `None`) is judged at none, whatever the row's own `Fit.runtime` and a full Ollama disk say; only
    without `runtime` the row's own runtime is the fallback."""
    big = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M", int(20 * GIB))
    entry = document([big]).machines[0].ranked[0]
    full = Disk(int(1 * GIB), "C:")
    assert place("third", full) is None
    assert entry_place(entry, full, "third") is None
    assert entry_over_disk(entry, full, "third") is False
    assert entry_place(entry, full) is full and entry_over_disk(entry, full) is True  # no `runtime`: the row's own, Ollama's
    assert entry_over_disk(entry, full, "ollama") is True


# --- finding 3 (c): the new branch of the choice for another computer ---------------------------------------------


def _other_run(tmp_path, answer):
    asker = _Recording({"pull_other": answer})
    run, lines = _run(tmp_path, asker)
    return run, lines


def _other_context(doc, packages):
    return PullContext(doc, "laptop", packages, ambiguous_names(packages))  # the document's one machine is another one


def test_choosing_an_lms_row_of_another_computer_hands_over_the_lms_line_and_pulls_nothing(tmp_path, monkeypatch):
    from modelroom import guided_remote

    q4 = _q4()
    doc = _on_machine(document([q4]), ["lmstudio"])
    rows = [row for row in pull_rows(doc, HERE, packages=[q4])]
    assert [row.runtime for row in rows] == ["lmstudio"]
    run, lines = _other_run(tmp_path, f"{HERE} lmstudio {QUANT}")
    run.ssh_active = HERE  # even the active computer is handed the line: no SSH call is made for it (G7)
    monkeypatch.setattr(guided_remote, "pull_remote_row", lambda *args, **kwargs: pytest.fail("an LM Studio row went over SSH"))

    pull_other_step(run, _other_context(doc, [q4]))

    said = [line.strip() for line in lines]
    assert f"install   #1 on {HERE}  lms get --gguf {QUANT}" in said
    assert not any("modelroom pull" in line or "ollama pull" in line for line in said)
    assert [call for call in run.daemon.calls if call[0] == "POST"] == []


# --- finding 2: the notes under a chosen LM Studio row break to the width of the screen -----------------------------


def _form_home(run):
    run.room = replace(run.room, which=lambda name: None, exists=lambda path: True)


def _form_path(run):
    run.room = replace(run.room, which=lambda name: "C:/tools/lms" if name == "lms" else None)


def _printed_lines(tmp_path, form: str) -> tuple[list[str], int]:
    """The raw lines a chosen LM Studio row prints, in the call form `form`, and the context of the run."""
    q4 = _q4()
    if form == "handover":
        doc = _on_machine(document([q4]), ["lmstudio"])
        run, lines = _other_run(tmp_path, f"{HERE} lmstudio {QUANT}")
        pull_other_step(run, _other_context(doc, [q4]))
        return list(lines), doc.scenario.context_requested
    doc = _on_machine(document([q4]), ["ollama", "lmstudio"])
    run, lines = _run(tmp_path, _Recording({"pull_package": f"lmstudio {QUANT}"}))
    {"path": _form_path, "home": _form_home}[form](run)
    pull_step(run, PullContext(doc, HERE, [q4], ambiguous_names([q4])), True)
    return list(lines), doc.scenario.context_requested


@pytest.mark.parametrize("form", ["path", "home", "handover"])
def test_no_note_under_a_chosen_lms_row_is_wider_than_a_hundred_characters(tmp_path, form):
    lines, context = _printed_lines(tmp_path, form)
    sentence = f"in LM Studio: load it with context length {context}, then start its server (the server switch in LM Studio, or lms server start)"
    for line in lines:
        if "get --gguf" in line:
            continue  # a command to copy is as long as it is, and is never broken
        assert len(line) <= 100, line
    # the wording of the brief is kept: the broken lines read as the one sentence
    start = next(index for index, line in enumerate(lines) if line.strip().startswith("in LM Studio: load it"))
    stop = next(index for index, line in enumerate(lines) if line.strip().endswith("lms server start)"))
    assert stop > start  # it did break
    assert " ".join(line.strip() for line in lines[start : stop + 1]) == sentence
    assert any("get --gguf" in line for line in lines)  # the command line is there


def test_the_command_lines_are_whole_in_every_form(tmp_path):
    for form, expected in (
        ("path", f"lms get --gguf {QUANT}"),
        ("home", f'& "$env:USERPROFILE\\.lmstudio\\bin\\lms.exe" get --gguf {QUANT}'),
        ("handover", f"lms get --gguf {QUANT}"),
    ):
        lines, _context = _printed_lines(tmp_path / form, form)
        assert any(line.strip().endswith(expected) for line in lines), form


# --- finding 3 (a) and finding 5: a vLLM found live on a machine whose profile names LM Studio alone ---------------


def _lms_alone(cards: str = "four_l4_server_vllm.json"):
    from test_golden_vllm import FIXTURES
    from modelroom.profile import HardwareProfile

    data = json.loads((FIXTURES / "profiles_v4" / cards).read_text(encoding="utf-8"))
    data["runtimes"] = ["lmstudio"]
    return HardwareProfile.model_validate(data)


def _served():
    from modelroom.vllm_server import ServedModel

    return ServedModel(id="x", root="x", max_model_len=32768)


def _room_with_places(ollama: float, lms: float, vllm: float):
    """A room whose three places lie on three drives of their own, with the free space given in GiB."""
    from fixture_room import fixed_room

    def free(path: Path) -> int:
        parts = {part.lower() for part in Path(path).parts}
        return int((lms if ".lmstudio" in parts else vllm if ".cache" in parts or "huggingface" in parts else ollama) * GIB)

    return replace(fixed_room(), exists=lambda path: True, disk_usage=free)


def _live_vllm_disk(room, packages, profile) -> Disk:
    """The disk of a machine the way a guided run builds it: Ollama's, vLLM's out of `hf_cache.vllm_disk`, LM Studio's
    out of `lmstudio_install.lmstudio_disk_at` with the runtimes of the machine."""
    from modelroom.hf_cache import vllm_disk
    from modelroom.lmstudio_install import lmstudio_disk_at
    from modelroom.runtimes import machine_runtime_keys

    disk = free_disk(room)
    disk = replace(disk, vllm=vllm_disk(room, packages))
    return replace(disk, runtimes=machine_runtime_keys(profile), lmstudio=lmstudio_disk_at(room, packages))


@pytest.mark.parametrize("lms, vllm", [(50.0, 50.0), (50.0, 2.0), (2.0, 50.0), (2.0, 2.0)])
def test_a_live_vllm_on_an_lmstudio_profile_is_judged_at_each_places_disk_in_terminal_and_markdown(lms, vllm):
    from test_golden_vllm import _base_model, _gguf_package, _render, _tensor_package
    from test_lmstudio_weichen import _disk_place, _marked_in_markdown, _marked_in_terminal

    profile, packages = _lms_alone(), [_gguf_package(), _tensor_package()]
    live = {"server": _served()}
    doc = _render(profile, [_base_model()], packages, live_vllm=live)
    block = doc.machines[0]
    ranks = {entry.fit.runtime: entry.rank for entry in block.ranked}
    assert set(ranks) == {"ollama", "vllm"}  # the live vLLM brought its tensor row in beside the LM Studio profile
    disk = _live_vllm_disk(_room_with_places(50.0, lms, vllm), packages, profile)

    view = RunView(install_for="server", live={"server": LiveRoom(disk=disk)}, packages=packages, live_vllm=live)
    terminal_lines = terminal_blocks(doc, "server", view=view)
    terminal = "\n".join("".join(text for _style, text in line) for line in terminal_lines)
    markdown = document_markdown(doc, {"server": disk}, live)

    expected = ({ranks["ollama"]} if lms < 10 else set()) | ({ranks["vllm"]} if vllm < 10 else set())
    assert _marked_in_terminal(terminal_lines) == expected == _marked_in_markdown(markdown)
    assert _disk_place(terminal) == _disk_place(markdown)


def test_the_unequal_cards_note_follows_the_runtimes_in_effect():
    """Finding 5: it is vLLM's note. It stays where vLLM is among the runtimes of the machine, live or by profile,
    and is gone where LM Studio is there without vLLM -- a profile with LM Studio alone and a vLLM found live included."""
    from modelroom.views_runtime import unequal_cards_note

    mixed = _lms_alone("mixed_cards_server.json")
    note = "vLLM can split over"
    assert unequal_cards_note(mixed) is None
    assert unequal_cards_note(mixed, _served()) is not None and unequal_cards_note(mixed, _served()).startswith(note)
    assert unequal_cards_note(mixed.model_copy(update={"runtimes": ["ollama", "lmstudio", "vllm"]})) is not None
    assert unequal_cards_note(mixed.model_copy(update={"runtimes": ["ollama", "lmstudio"]}), _served()) is not None


def test_the_unequal_cards_note_reaches_the_screen_for_a_live_vllm_on_an_lmstudio_profile():
    from test_golden_vllm import _base_model, _gguf_package, _render, _tensor_package

    mixed, packages = _lms_alone("mixed_cards_server.json"), [_gguf_package(), _tensor_package()]
    live = {"server": _served()}
    doc = _render(mixed, [_base_model()], packages, live_vllm=live)
    with_live = "\n".join(plain(line) for line in terminal_blocks(doc, "server", view=RunView(install_for="server", packages=packages, live_vllm=live)))
    plain_doc = _render(mixed, [_base_model()], packages)
    without = "\n".join(plain(line) for line in terminal_blocks(plain_doc, "server", view=RunView(install_for="server", packages=packages)))
    assert "vLLM can split over" in with_live and "smaller card" in with_live
    assert "vLLM can split over" not in without  # LM Studio alone, nothing live: no vLLM, no note


# --- finding 3 (b): an active computer over SSH reads LM Studio as absent ------------------------------------------


def test_an_active_computer_over_ssh_with_lms_on_its_path_reads_lmstudio_absent(tmp_path):
    """The server has `lms` on its `PATH`; its `hello` carries no field for LM Studio, its measured profile names no
    LM Studio, and the table made of that profile has no LM Studio entry -- where the profile the run held before it
    (one that named LM Studio) did have one."""
    from fixture_ssh import FakeSsh
    from modelroom.profile import HardwareProfile
    from modelroom.remote_client import hello, profile
    from test_remote_vllm import _context, _room

    server_room = replace(_room(tmp_path), which=lambda name: f"/usr/bin/{name}")
    ssh = FakeSsh(context=_context(tmp_path, room=server_room))

    said = hello("gpu1", ssh, ssh_path="ssh")
    assert "lmstudio" not in json.dumps(said)

    fresh = HardwareProfile.model_validate(profile("gpu1", "3f9a0c21d4e6b870", None, ssh)["export"]["profile"])
    assert "lmstudio" not in fresh.runtimes

    q4 = _q4()
    before = _on_machine(document([q4]), ["ollama", "lmstudio"])
    after = _on_machine(document([q4]), list(fresh.runtimes) or ["ollama"])
    assert {row.runtime for row in pull_rows(before, HERE, packages=[q4])} == {"ollama", "lmstudio"}
    assert {row.runtime for row in pull_rows(after, HERE, packages=[q4])} == {"ollama"}
