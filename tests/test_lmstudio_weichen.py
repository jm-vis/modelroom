"""AP9-LS-E: the places where the table of runtimes needed more than a row (S1-S8 of the brief, F6 a, F7 a).

LM Studio is the third entry of `RUNTIMES`. The pull list gives each Hugging Face GGUF package a second
entry for it, the table shows the package once (F7 a), and the places that were written for Ollama and vLLM
learn the third one: what a choice in the list does (S1), the lock texts (S2), the place and the words of
the disk (S3, S7), the title of a section (S4), the `runtime` row (S5), the form of a table (S6) and the line
for a package that is on disk already (S8). Every test goes through prepared documents or a whole render and
the views a guided run draws; none starts a process or reaches a server.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from modelroom import runtimes
from modelroom.pull_choice import ambiguous_names, pull_rows
from modelroom.quantization import package_identity_key
from modelroom.room import GIB, Disk, LiveRoom
from modelroom.screen import plain
from modelroom.views import RunView, document_markdown, terminal_blocks

from fixture_rows import document, ollama_package, package
from test_runtimes import _on_machine

HERE = "workstation"
BOTH = ["ollama", "lmstudio"]


def _gguf(file: str = "Nova-7B-Q4_K_M.gguf", quant: str = "Q4_K_M", gib: float = 5.0, repo: str = "unsloth/Nova-7B-GGUF", **changes):
    return package(repo, file, quant, int(gib * GIB), **changes)


def _disk(ollama: float | None, lms: float | None = None, keys=BOTH, held=frozenset(), lms_held=frozenset(), vllm: float | None = None) -> Disk:
    """A disk for the machine: Ollama's free GiB (the disk itself), LM Studio's and vLLM's nested in it."""
    return Disk(
        None if ollama is None else int(ollama * GIB), "C:", held=held,
        lmstudio=None if lms is None else Disk(int(lms * GIB), "D:", held=lms_held),
        vllm=None if vllm is None else Disk(int(vllm * GIB), "E:"),
        runtimes=None if keys is None else frozenset(keys),
    )


def _terminal(doc, disk, packages=(), held=None, live_vllm=None) -> list[list[tuple[str, str]]]:
    view = RunView(install_for=HERE, live={HERE: LiveRoom(disk=disk)}, packages=packages, held=held or {}, live_vllm=live_vllm or {})
    return terminal_blocks(doc, HERE, view=view)


def _marked_in_terminal(lines) -> set[int]:
    return {int(re.match(r"\s*(\d+)\s", text).group(1)) for style, text in (line[0] for line in lines if len(line) == 1) if style == "class:over-disk"}


def _marked_in_markdown(text: str) -> set[int]:
    return {int(re.match(r"\|\s*(\d+)\s", line).group(1)) for line in text.splitlines() if line.startswith("|") and " ×" in line and re.match(r"\|\s*\d+\s", line)}


def _disk_place(text: str) -> list[tuple[str, str]]:
    """`(free, drive)` of every `need more than the X free on Y` sentence in a view."""
    return re.findall(r"more than the ([\d.]+ GB) free on (\S+?)(?=\.?(?:\s|$))", text)


def _views(doc, disk, packages=(), held=None, live_vllm=None):
    terminal_lines = _terminal(doc, disk, packages, held, live_vllm)
    terminal = "\n".join("".join(text for _style, text in line) for line in terminal_lines)
    markdown = document_markdown(doc, {HERE: disk}, live_vllm)
    return terminal_lines, terminal, markdown


# --- S1, S2, S8: the list, the lock texts and the line under the table ---------------------------------------


def _pull(tmp_path, doc, packages, answers: dict, which=None, home_lms: bool = False):
    """The pull of step 5 as the guided run calls it, with `lms` found by a path search (or not); `home_lms` puts
    the program in the home folder of the room, where the fixed room has none."""
    from modelroom.guided_install import PullContext, pull_step

    from test_pull_choice import _Recording, _run

    asker = _Recording(answers)
    run, lines = _run(tmp_path, asker)
    if which is not None:
        run.room = replace(run.room, which=which, exists=(lambda path: True) if home_lms else run.room.exists)
    pull_step(run, PullContext(doc, HERE, packages, ambiguous_names(packages)), True)
    return [line.strip() for line in lines], asker.lists, run


def _in_path(name):
    return "C:/tools/lms" if name == "lms" else None


def test_an_ollama_and_an_lmstudio_row_share_the_one_list(tmp_path):
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    said, lists, _run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M", "pull": False}, _in_path)
    values = [choice.value for choice in lists["pull_package"]]
    assert values == ["hf.co/unsloth/Nova-7B-GGUF:Q4_K_M", "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M", "none"]


def test_choosing_an_lms_row_asks_pull_and_never_calls_the_daemon_pull(tmp_path):
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    # `pull = no`: the question of the download is asked (the key is `pull`, as Ollama's), nothing is sent to the daemon.
    said, _lists, run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M", "pull": False}, _in_path)
    assert "install   #1  lms get --gguf https://huggingface.co/unsloth/Nova-7B-GGUF@Q4_K_M" in said
    assert not any(call[0] == "POST" for call in run.daemon.calls)
    assert not any(line.startswith("? Pull") or "pulling" in line for line in said)
    assert "? Download #1 with lms now? (5.0 GB)  No" in said
    assert run.asker.asked == ["pull_package", "pull"]
    assert not any(line.startswith(("download", "✓ Downloaded")) for line in said)


def test_a_yes_for_an_lms_row_goes_to_lms_get_and_not_to_the_daemon(tmp_path, monkeypatch):
    """The whole step with a yes: the process call is the one of `run_lms`, here recorded and not started."""
    import subprocess

    calls: list = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(subprocess, "run", fake)
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    said, _lists, run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M", "pull": True}, _in_path)
    assert calls == [((["C:/tools/lms", "get", "--gguf", "https://huggingface.co/unsloth/Nova-7B-GGUF@Q4_K_M", "--yes"],), {"check": False})]
    assert not any(call[0] == "POST" for call in run.daemon.calls)
    assert "download  lms get --gguf https://huggingface.co/unsloth/Nova-7B-GGUF@Q4_K_M --yes" in said
    assert any(line.startswith("#1: lms get ended, but Nova-7B-Q4_K_M.gguf is not in ~/.lmstudio/models/unsloth/Nova-7B-GGUF") for line in said)


def test_choosing_an_lms_row_without_a_found_lms_asks_nothing(tmp_path, monkeypatch):
    """Form `None` (no `lms` in the PATH, none in the home folder): the old output, no hint, no question, no process."""
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("a process was started"))
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    said, _lists, run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M", "pull": True}, lambda name: None)
    assert run.asker.asked == ["pull_package"]
    assert any(line.startswith("install   #1  lms get") for line in said)
    assert "if lms is not found: it is in .lmstudio/bin in your home folder" in said
    assert not any("service" in line or "Download" in line for line in said)


def test_choosing_an_lms_row_prints_and_pulls_nothing(tmp_path):
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    said, _lists, run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M"}, _in_path)
    assert any(line.startswith("install   #1  lms get") for line in said)
    assert [call for call in run.daemon.calls if call[1] == "/api/pull"] == []
    assert "? Download #1 with lms now? (5.0 GB)  No" in said  # no answer for `pull` is a No


def test_too_little_disk_for_an_lms_row_ends_after_the_fault_line_without_a_question(tmp_path, monkeypatch):
    import subprocess

    from modelroom import lmstudio_install
    from modelroom.room import Disk

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("a process was started"))
    monkeypatch.setattr(lmstudio_install, "lmstudio_disk_at", lambda room, packages=(): Disk(1 * GIB, "D:"))
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    said, _lists, run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M", "pull": True}, _in_path)
    assert any(line.startswith("install   not enough disk for #1: needs") and line.endswith("free on D:") for line in said)
    assert run.asker.asked == ["pull_package"]
    assert not any("service" in line or "Download" in line or "in LM Studio:" in line for line in said)


def test_the_chosen_row_prints_the_block_of_the_brief(tmp_path):
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    said, _lists, _run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M"}, _in_path)
    start = said.index("install   #1  lms get --gguf https://huggingface.co/unsloth/Nova-7B-GGUF@Q4_K_M")
    context = doc.scenario.context_requested
    # the hint about the service, then the question, then the sentences of AP9-LS-F; each broken between words to the width of the screen
    assert " ".join(said[start + 1 : start + 3]) == (
        "lms starts LM Studio's service in the background; a vision model also fetches its projector file"
    )
    assert said[start + 3] == "? Download #1 with lms now? (5.0 GB)  No"
    assert " ".join(said[start + 4 : start + 6]) == (
        f"in LM Studio: load it with context length {context}, then start its server (the server switch in LM Studio, or lms server start)"
    )
    assert said[start + 6] == "then run modelroom again: it finds the server; LM Studio is not measured yet"


def test_an_answer_file_picks_an_lms_row(tmp_path):
    q4, q8 = _gguf(), _gguf("Nova-7B-Q8_0.gguf", "Q8_0", 8.0)
    doc = _on_machine(document([q8, q4]), BOTH)
    said, _lists, _run = _pull(tmp_path, doc, [q8, q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M"}, _in_path)
    assert "install   #2  lms get --gguf https://huggingface.co/unsloth/Nova-7B-GGUF@Q4_K_M" in said


def test_an_lms_only_in_the_home_folder_prints_the_home_call_and_its_hint(tmp_path):
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    said, _lists, _run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M"}, lambda name: None, home_lms=True)
    # the fixed room has a Windows home with every file present, so `lms.exe` is found there
    assert 'install   #1  & "$env:USERPROFILE\\.lmstudio\\bin\\lms.exe" get --gguf https://huggingface.co/unsloth/Nova-7B-GGUF@Q4_K_M' in said
    assert "lms is not on the PATH yet; this line calls it where LM Studio put it (PowerShell)" in said


def test_the_home_call_block_reads_as_the_golden_file(tmp_path):
    from test_golden_vllm import _compare

    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    said, _lists, _run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M"}, lambda name: None, home_lms=True)
    start = next(index for index, line in enumerate(said) if line.startswith("install   #1"))
    _compare("golden_install_lms_home_win.txt", "\n".join(said[start : start + 8]) + "\n")


def test_the_lock_reasons_of_an_lms_row():
    q4 = _gguf()
    subfolder = _gguf("Q4/Nova-7B-Q4_K_M.gguf", repo="unsloth/Other-GGUF")
    ud = _gguf("Nova-7B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL", repo="unsloth/Third-GGUF")
    shards = _gguf(repo="unsloth/Fourth-GGUF")
    shards = shards.model_copy(update={"files": [
        file.model_copy(update={"name": f"Fourth-{n:05d}-of-00002.gguf", "role": "weights_shard"}) for n, file in enumerate([shards.files[0]] * 2, start=1)
    ]})
    packages = [q4, subfolder, ud, shards]
    doc = _on_machine(document(packages), BOTH)
    rows = {(row.rank, row.runtime): row.locked for row in pull_rows(doc, HERE, packages=packages)}
    assert rows[(1, "lmstudio")] is None
    assert rows[(2, "lmstudio")] == "computed only: lms get for files in a subfolder is not checked"
    assert rows[(3, "lmstudio")] == "computed only: lms get for UD-Q4_K_XL is not checked"
    assert rows[(4, "lmstudio")] == "computed only: lms get for split files is not checked"
    # Ollama's own rows keep their words
    assert rows[(1, "ollama")] is None


def test_the_lock_text_is_said_under_the_table_where_nothing_else_is_offered():
    ud = _gguf("Nova-7B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL")
    doc = _on_machine(document([ud]), ["lmstudio"])
    text = "\n".join(plain(line) for line in terminal_blocks(doc, HERE, view=RunView(install_for=HERE, packages=[ud])))
    assert "install   #1 computed only: lms get for UD-Q4_K_XL is not checked" in text


def test_the_line_under_the_table_is_the_path_form():
    q4 = _gguf()
    doc = _on_machine(document([q4]), ["lmstudio"])  # a machine with LM Studio alone: its entry is the one offered
    text = "\n".join(plain(line) for line in terminal_blocks(doc, HERE, view=RunView(install_for=HERE, packages=[q4])))
    assert "install   #1  lms get --gguf https://huggingface.co/unsloth/Nova-7B-GGUF@Q4_K_M" in text
    assert "ollama pull" not in text and "in LM Studio:" not in text  # no hint lines under the table


def test_an_lms_package_on_disk_replaces_the_download_line_under_the_table():
    q4 = _gguf()
    doc = _on_machine(document([q4]), ["lmstudio"])
    held = {"lmstudio": frozenset({package_identity_key(q4)})}
    view = RunView(install_for=HERE, packages=[q4], held=held)
    text = "\n".join(plain(line) for line in terminal_blocks(doc, HERE, view=view))
    assert "install   #1 is on disk already (size checked)" in text
    assert "lms get" not in text
    # Ollama's line with a held package stays as it was
    ollama_doc = document([q4])
    ollama_text = "\n".join(plain(line) for line in terminal_blocks(ollama_doc, HERE, view=RunView(install_for=HERE, packages=[q4], held=held)))
    assert "install   #1  ollama pull hf.co/unsloth/Nova-7B-GGUF:Q4_K_M" in ollama_text


def test_choosing_a_held_lms_row_says_it_is_on_disk_and_keeps_the_load_hint(tmp_path, monkeypatch):
    q4 = _gguf()
    doc = _on_machine(document([q4]), BOTH)
    held = frozenset({package_identity_key(q4)})
    monkeypatch.setattr(runtimes, "LMSTUDIO", replace(runtimes.LMSTUDIO, on_disk=lambda target, packages: held))
    monkeypatch.setattr(runtimes, "RUNTIMES", (runtimes.OLLAMA, runtimes.VLLM, runtimes.LMSTUDIO))
    said, _lists, _run = _pull(tmp_path, doc, [q4], {"pull_package": "lmstudio unsloth/Nova-7B-GGUF@Q4_K_M"}, _in_path)
    assert "install   #1 is on disk already (size checked)" in said
    assert not any(line.startswith("install   #1  lms get") for line in said)
    assert any(line.startswith("in LM Studio: load it with context length") for line in said)


def test_a_profile_from_another_computer_with_lmstudio_prints_the_command():
    q4 = _gguf()
    doc = _on_machine(document([q4]), ["lmstudio"])
    view = RunView(install_for="gpu1", packages=[q4])  # this run is on another machine: the block is that machine's
    text = "\n".join(plain(line) for line in terminal_blocks(doc, "gpu1", view=view))
    assert "#1 on workstation  lms get --gguf https://huggingface.co/unsloth/Nova-7B-GGUF@Q4_K_M" in text


# --- S4: the title of a section and the rule ------------------------------------------------------------------


def _doc_with_vllm_row(keys):
    q4 = _gguf()
    tensor = package("RedHatAI/Nova-7B-quantized.w4a16", "model.safetensors", "AWQ", 6 * GIB, format="tensor")
    doc = _on_machine(document([q4, tensor]), keys)
    blocks = []
    for block in doc.machines:
        ranked = [row.model_copy(update={"fit": row.fit.model_copy(update={"runtime": "vllm"})}) if row.rank == 2 else row for row in block.ranked]
        blocks.append(block.model_copy(update={"ranked": ranked}))
    return doc.model_copy(update={"machines": blocks}), [q4, tensor]


def test_runtime_sections_name_the_gguf_part_after_the_machine():
    from modelroom.views_sections import runtime_sections

    doc, _packages = _doc_with_vllm_row(["lmstudio", "vllm"])
    assert [key for key, _part in runtime_sections(doc.machines[0])] == ["lmstudio", "vllm"]
    assert [len(part.ranked) for _key, part in runtime_sections(doc.machines[0])] == [1, 1]
    both, _packages = _doc_with_vllm_row(["ollama", "lmstudio", "vllm"])
    assert [key for key, _part in runtime_sections(both.machines[0])] == ["ollama", "vllm"]
    plain_doc, _packages = _doc_with_vllm_row(["ollama", "vllm"])
    assert [key for key, _part in runtime_sections(plain_doc.machines[0])] == ["ollama", "vllm"]


def test_a_machine_with_lmstudio_and_vllm_without_ollama_names_lm_studio():
    doc, packages = _doc_with_vllm_row(["lmstudio", "vllm"])
    terminal = "\n".join(plain(line) for line in terminal_blocks(doc, HERE, view=RunView(install_for=HERE, packages=packages)))
    markdown = document_markdown(doc)
    assert "with LM Studio" in terminal and "with vLLM" in terminal and "with Ollama" not in terminal
    assert "### with LM Studio" in markdown and "### with vLLM" in markdown
    assert "with Ollama" not in markdown
    for ollama_machine in (["ollama", "vllm"], ["ollama", "lmstudio", "vllm"]):
        other, packages = _doc_with_vllm_row(ollama_machine)
        shown = "\n".join(plain(line) for line in terminal_blocks(other, HERE, view=RunView(install_for=HERE, packages=packages)))
        assert "with Ollama" in shown and "with LM Studio" not in shown


def test_the_both_runtimes_rule_for_ollama_is_the_old_constant():
    from modelroom.ranking import BOTH_RUNTIMES_RULE, both_runtimes_rule

    assert both_runtimes_rule("Ollama") == BOTH_RUNTIMES_RULE
    assert "LM Studio and vLLM rows" in both_runtimes_rule("LM Studio")
    assert BOTH_RUNTIMES_RULE.startswith("; on a machine with both Ollama and vLLM rows")


def test_the_rule_names_the_gguf_side_after_the_machines():
    from test_golden_vllm import _base_model, _gguf_package, _profile, _render, _tensor_package

    ollama = _render(_profile(), [_base_model()], [_gguf_package(), _tensor_package()])
    assert "Ollama and vLLM rows" in ollama.ranking_rule
    lms = _render(_profile().model_copy(update={"runtimes": ["lmstudio", "vllm"]}), [_base_model()], [_gguf_package(), _tensor_package()])
    assert "LM Studio and vLLM rows" in lms.ranking_rule and "Ollama" not in lms.ranking_rule.split("; on a machine")[-1]


# --- S7: the place of an entry, in the list, in the terminal and in the Markdown file ---------------------


def test_each_list_value_reads_the_disk_of_its_own_runtime():
    big, small = _gguf(gib=20.0), _gguf("Nova-7B-Q8_0.gguf", "Q8_0", 4.0)
    packages = [big, small]
    doc = _on_machine(document(packages), BOTH)
    # Ollama's place is too small for the big package, LM Studio's is not
    rows = {(row.rank, row.runtime): row for row in pull_rows(doc, HERE, disk=_disk(10.0, 30.0), packages=packages)}
    assert rows[(1, "ollama")].over_disk and rows[(1, "ollama")].locked == "needs 20.0 GB plus 1 GB room, 10.0 GB free"
    assert not rows[(1, "lmstudio")].over_disk and rows[(1, "lmstudio")].locked is None
    # and the other way round: the sentence names the free space of the entry's own place
    rows = {(row.rank, row.runtime): row for row in pull_rows(doc, HERE, disk=_disk(30.0, 10.0), packages=packages)}
    assert not rows[(1, "ollama")].over_disk
    assert rows[(1, "lmstudio")].over_disk and rows[(1, "lmstudio")].locked == "needs 20.0 GB plus 1 GB room, 10.0 GB free"


def test_a_package_lm_studio_holds_needs_no_disk_in_its_entry():
    big = _gguf(gib=20.0)
    doc = _on_machine(document([big]), BOTH)
    held = {"lmstudio": frozenset({package_identity_key(big)})}
    disk = _disk(10.0, 10.0, lms_held=held["lmstudio"])
    rows = {row.runtime: row for row in pull_rows(doc, HERE, disk=disk, packages=[big], held=held)}
    assert rows["lmstudio"].locked is None and not rows["lmstudio"].too_big
    assert rows["ollama"].too_big


CASES = {
    "ollama too small, LM Studio fits": ((10.0, 50.0), set()),
    "LM Studio too small, Ollama fits": ((50.0, 10.0), set()),
    "both too small": ((10.0, 10.0), {"big"}),
    "both fit": ((50.0, 50.0), set()),
}


def _real_render(keys, packages):
    """A whole render (profile, snapshot, ranking, fit) of the laptop of `profiles_v4/laptop_lmstudio.json` with the
    runtimes `keys` -- not a prepared document -- and the rank every package got, by `package_identity_key`."""
    from test_lmstudio_render import _laptop
    from test_golden_vllm import _base_model, _render

    document = _render(_laptop(keys), [_base_model()], packages, context=8192, requests=1)
    return document, {tuple(entry.package_identity): entry.rank for entry in document.machines[0].ranked}


def _real_package(file: str, quant: str, gib: float):
    from test_golden_vllm import _gguf_package

    package = _gguf_package()
    weights = package.files[0].model_copy(update={"name": file, "size_bytes": int(gib * GIB)})
    return package.model_copy(update={"quantization": quant, "files": [weights]})


def _real_library_package(gib: float):
    from test_golden_vllm import _gguf_package

    package = _gguf_package()
    weights = package.files[0].model_copy(update={"name": "model", "size_bytes": int(gib * GIB)})
    return package.model_copy(update={
        "source": "ollama", "repo": None, "revision": None, "ollama_name": "qwen3:8b",
        "manifest_digest": "sha256:" + "a" * 64, "quantization": "Q8_0", "files": [weights],
    })


def _both_views(document, disk, packages):
    """The one disk, built once, handed to the terminal and to the Markdown file of the same real document."""
    view = RunView(install_for="server", live={"server": LiveRoom(disk=disk)}, packages=packages)
    terminal_lines = terminal_blocks(document, "server", view=view)
    terminal = "\n".join("".join(text for _style, text in line) for line in terminal_lines)
    return terminal_lines, terminal, document_markdown(document, {"server": disk})


@pytest.mark.parametrize("case", list(CASES))
def test_terminal_and_markdown_mark_the_same_ranks_for_each_disk_case(case):
    """Two Hugging Face GGUF packages and one Ollama-library package (the library one is judged at Ollama's place
    alone), the places of Ollama and LM Studio with opposite room: out of a real render, one disk for both views."""
    (ollama, lms), expected = CASES[case]
    big, small, library = _real_package("Qwen3-8B-Q4_K_M.gguf", "Q4_K_M", 12.0), _real_package("Qwen3-8B-Q2_K.gguf", "Q2_K", 2.0), _real_library_package(12.0)
    packages = [big, small, library]
    document, ranks = _real_render(BOTH, packages)
    rank = {"small": ranks[("huggingface", big.repo, "Qwen3-8B-Q2_K.gguf")], "library": ranks[("ollama", "qwen3:8b", "8b")], "big": ranks[("huggingface", big.repo, "Qwen3-8B-Q4_K_M.gguf")]}
    assert sorted(rank.values()) == [1, 2, 3] and len(ranks) == 3  # the render shows all three, whatever order it gives them
    terminal_lines, terminal, markdown = _both_views(document, _disk(ollama, lms), packages)
    marked = _marked_in_terminal(terminal_lines)
    wanted = {rank[name] for name in expected} | ({rank["library"]} if ollama < 13 else set())
    assert marked == wanted
    assert _marked_in_markdown(markdown) == marked
    assert _disk_place(terminal) == _disk_place(markdown)
    if "big" in expected:  # both too small: the sentence names Ollama's place
        assert _disk_place(markdown)[0] == ("10.0 GB", "C:")


@pytest.mark.parametrize("lms, expected", [(10.0, True), (50.0, False)])
def test_a_machine_with_lmstudio_alone_is_judged_at_the_lmstudio_place_in_both_views(lms, expected):
    big = _real_package("Qwen3-8B-Q4_K_M.gguf", "Q4_K_M", 12.0)
    document, ranks = _real_render(["lmstudio"], [big])
    assert len(ranks) == 1
    disk = _disk(1.0, lms, keys=["lmstudio"])  # Ollama's place is tiny but this machine has no Ollama
    terminal_lines, terminal, markdown = _both_views(document, disk, [big])
    wanted = set(ranks.values()) if expected else set()
    assert _marked_in_terminal(terminal_lines) == wanted == _marked_in_markdown(markdown)
    if expected:
        assert _disk_place(terminal) == _disk_place(markdown) == [("10.0 GB", "D:")]


def test_a_disk_without_runtimes_judges_each_row_by_its_own_runtime():
    from modelroom.disk_fit import entry_over_disk, over_disk, place

    big = _gguf(gib=20.0)
    doc = _on_machine(document([big]), BOTH)
    entry = doc.machines[0].ranked[0]
    disk = _disk(10.0, 50.0, keys=None)  # no `runtimes`: a disk from before LM Studio
    assert entry_over_disk(entry, disk) is over_disk(entry.weights_gib, place("ollama", disk), entry.package_identity) is True


def test_the_markdown_alternatives_follow_lmstudio_accepts():
    from modelroom.disk_fit import hf_gguf_row

    hf_gguf, tensor = _gguf(), package("RedHatAI/Nova-7B-quantized.w4a16", "model.safetensors", "AWQ", format="tensor")
    library = ollama_package("nova:7b")
    doc = document([hf_gguf, tensor, library])
    for entry, pkg in zip(doc.machines[0].ranked, [hf_gguf, tensor, library]):
        assert hf_gguf_row(entry) is runtimes.LMSTUDIO.accepts(pkg)


def test_markdown_of_an_ollama_only_profile_is_byte_equal():
    """A disk with no `runtimes`, and one whose runtimes are Ollama's alone, change no byte of the Markdown file."""
    big = _gguf(gib=20.0)
    doc = document([big])
    without = document_markdown(doc, {HERE: _disk(10.0, keys=None)})
    with_only_ollama = document_markdown(doc, {HERE: _disk(10.0, keys=["ollama"])})
    assert without == with_only_ollama and " ×" in without


def test_a_live_vllm_tensor_row_on_an_lmstudio_profile_is_judged_alike_in_terminal_and_markdown():
    from modelroom.vllm_server import ServedModel

    served = ServedModel(id="x", root="x", max_model_len=32768)
    for vllm_free, tensor_marked in ((50.0, False), (3.0, True)):
        doc, packages = _doc_with_vllm_row(["lmstudio"])
        live = {HERE: served}
        rows = pull_rows(doc, HERE, disk=_disk(1.0, 50.0, keys=["lmstudio"], vllm=vllm_free), packages=packages)
        assert [row.runtime for row in rows if row.rank == 2] == ["vllm"]  # one vLLM entry, profile or not
        disk = _disk(1.0, 50.0, keys=["lmstudio"], vllm=vllm_free)
        terminal_lines, terminal, markdown = _views(doc, disk, packages, live_vllm=live)
        marked = _marked_in_terminal(terminal_lines)
        assert (2 in marked) is tensor_marked and 1 not in marked  # the GGUF row: only LM Studio's place, which is large
        assert _marked_in_markdown(markdown) == marked
        assert _disk_place(terminal) == _disk_place(markdown)


def test_rows_other_than_hf_gguf_ignore_disk_runtimes():
    from modelroom.disk_fit import entry_over_disk, over_disk, place

    doc, _packages = _doc_with_vllm_row(["lmstudio"])
    tensor_entry = doc.machines[0].ranked[1]
    disk = _disk(1.0, 1.0, keys=["lmstudio"], vllm=2.0)  # `vllm` is not among `Disk.runtimes`
    assert entry_over_disk(tensor_entry, disk) is over_disk(tensor_entry.weights_gib, place("vllm", disk), tensor_entry.package_identity)
    library = ollama_package("nova:7b")
    row = document([library]).machines[0].ranked[0]
    assert entry_over_disk(row, disk) is over_disk(row.weights_gib, place("ollama", disk), row.package_identity)
