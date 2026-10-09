"""AP9-LS-G, the acceptance test: modelroom runs `lms` once, for the download the reader said yes to, and not otherwise.

A program called `lms` stands first in the `PATH` and in the home folder, the LM Studio server lists a
loaded model, and the profile of the machine names LM Studio. The run goes from the start screen through the
table and the list (an LM Studio entry is chosen) and step 4 to the card, and then to the question of the
download (`pull`, the key the Ollama pull uses as well).

- No: no answer for `pull` and `pull: false`, each under the full trap (`subprocess.Popen` and `os.system`
  fail the test at any call): the line to copy is printed, no process starts, the stand-in leaves no mark.
- Yes: the unchanged `run_lms` of `lmstudio_get.py` runs. The trap lets exactly one call through, the one whose
  first argument is the expected list (the program the `PATH` names, `get`, `--gguf`, the URL, `--yes`); any
  other call fails the test. The stand-in appends its arguments to a mark file, so the test counts one process
  call and one entry. The stand-in creates no model file, so the run ends on the check of the file (exit 1).

The tests of `test_lmstudio_no_process.py` that read the functions alone stay; this one is the same claim for the
registered entry. Everything the run asks of processes (hardware, `llmfit`, Ollama, `ssh`) comes from the fixed
probes of the test frame.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from modelroom.guided import run_guided
from modelroom.dialog import FileAsker

from fixture_lmstudio import (
    chat_calls, fake_lms_on_path, lms_in_the_home_folder, lmstudio_servers, trap_processes, trap_processes_except,
)
from fixture_room import fixed_room
from fixture_support import build_transport, guided_transport_mapping, offline_daemon, offline_servers, windows_probes
from test_guided_install import ANSWERS, RUN1

LIST_VALUE = "lmstudio unsloth/Qwen3.5-9B-GGUF@Q8_0"
URL = "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF@Q8_0"


def _guided(tmp_path: Path, monkeypatch, answers: dict, trap=trap_processes):
    mark = fake_lms_on_path(tmp_path, monkeypatch)
    home = tmp_path / "lms-home"
    lms_in_the_home_folder(home, mark)
    program = shutil.which("lms")
    assert program is not None  # the stand-in is found, or the probe would run empty
    for folder in ("results", "home"):
        (tmp_path / folder).mkdir(exist_ok=True)
    server = lmstudio_servers(["lmstudio_v1_models_loaded.json"])
    room = replace(fixed_room(), which=shutil.which, home=home, exists=lambda path: Path(path).exists())
    through = trap(monkeypatch) if trap is trap_processes else trap(monkeypatch, [program, "get", "--gguf", URL, "--yes"])
    lines: list[str] = []
    asker = FileAsker(answers)
    code = run_guided(
        asker, here=tmp_path / "results", pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport(guided_transport_mapping()), probes=windows_probes(), daemon=offline_daemon(),
        servers={**offline_servers(), "lmstudio": server}, now=RUN1, out=lines.append, colored=False, room=room,
    )
    return code, [line.strip() for line in lines], asker, server, mark, through


def test_the_trap_catches_a_direct_and_a_shell_call(monkeypatch):
    """The trap of the runs below is no empty one: a direct call and a shell call, whatever the first argument,
    each end as a failed test."""
    import os
    import subprocess

    trap_processes(monkeypatch)
    with pytest.raises(pytest.fail.Exception):
        subprocess.Popen(["lms", "ls", "--json"])
    with pytest.raises(pytest.fail.Exception):
        os.system("powershell -c lms ls")


def test_the_narrow_trap_lets_one_exact_call_through_and_nothing_else(tmp_path, monkeypatch):
    """The trap of the yes run: the listed call is the one that passes (and is recorded); a call with another
    first argument, the same list with one more element, and `os.system` fail."""
    import os
    import subprocess

    mark = fake_lms_on_path(tmp_path, monkeypatch)
    program = shutil.which("lms")
    allowed = [program, "get", "--gguf", URL, "--yes"]
    through = trap_processes_except(monkeypatch, allowed)
    subprocess.run(allowed, check=False)
    assert through == [allowed] and mark.read_text(encoding="ascii").split() == ["get", "--gguf", URL, "--yes"]
    with pytest.raises(pytest.fail.Exception):  # the same list a second time fails at the trap
        subprocess.Popen(allowed)
    with pytest.raises(pytest.fail.Exception):
        subprocess.Popen([program, "get", "--gguf", URL, "--yes", "--extra"])
    with pytest.raises(pytest.fail.Exception):
        subprocess.Popen([program, "ls"])
    with pytest.raises(pytest.fail.Exception):
        os.system("lms get")
    assert through == [allowed]  # exactly one call let through
    assert len(mark.read_text(encoding="ascii").splitlines()) == 1  # exactly one entry


@pytest.mark.parametrize("answer", [{}, {"pull": False}], ids=["no answer for pull", "pull is false"])
def test_a_no_runs_no_lms_and_prints_the_line_to_copy(tmp_path, monkeypatch, answer):
    answers = {**ANSWERS, "pull_package": LIST_VALUE, **answer}
    code, lines, asker, server, mark, _through = _guided(tmp_path, monkeypatch, answers)
    text = "\n".join(lines)

    assert code == 0, text
    assert "server    LM Studio" in text and "1 loaded: qwen3.5-4b, context 8192" in text  # the start screen names it
    assert "LM Studio is not measured" not in text and "load_test" in asker.asked  # its row is offered, so step 4 asks; no chat is sent
    assert "pull_package" in asker.asked and "pull" in asker.asked  # the question was reached and answered No
    assert any(line.startswith("install   #1  lms get --gguf https://huggingface.co/unsloth/Qwen3.5-9B-GGUF@") for line in lines)
    assert any(line.startswith("? Download #1 with lms now?") and line.endswith("No") for line in lines)
    assert not any(line.startswith(("download", "✓ Downloaded")) for line in lines)
    assert any("then run modelroom again" in line for line in lines)  # the hints stay for a no
    assert chat_calls(server) == []  # not picked, so no chat
    assert not mark.exists()  # `lms` was never run; and the trap saw no process


def test_a_yes_runs_lms_once_with_the_one_list(tmp_path, monkeypatch):
    answers = {**ANSWERS, "pull_package": LIST_VALUE, "pull": True}
    code, lines, asker, server, mark, through = _guided(tmp_path, monkeypatch, answers, trap=trap_processes_except)
    text = "\n".join(lines)

    program = shutil.which("lms")
    assert len(through) == 1 and through[0] == [program, "get", "--gguf", URL, "--yes"], text  # one process call
    entries = mark.read_text(encoding="ascii").splitlines()
    assert len(entries) == 1 and entries[0].split() == ["get", "--gguf", URL, "--yes"]  # and one entry of the stand-in
    assert "pull" in asker.asked
    assert any(line.startswith("? Download #1 with lms now?") and line.endswith("Yes") for line in lines)
    assert any(line.startswith("download  ") and line.endswith("get --gguf " + URL + " --yes") for line in lines)
    # the stand-in made no file: the check of the file speaks, the run ends with 1
    assert any(line.startswith("#1: lms get ended, but") and "catalog size" in line for line in lines), text
    assert code == 1, text
    assert not any(line.startswith("✓ Downloaded") for line in lines)
    assert chat_calls(server) == []
