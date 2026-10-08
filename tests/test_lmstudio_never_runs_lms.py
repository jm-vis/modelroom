"""AP9-LS-E, the acceptance test: modelroom never runs `lms`, in a whole guided run with the entry registered.

A program called `lms` stands first in the `PATH` and in the home folder, the LM Studio server lists a
loaded model, and the profile of the machine names LM Studio. The run goes from the start screen through the
table and the list (an LM Studio entry is chosen) and step 4 to the card. `subprocess.Popen` and `os.system`
are traps for the length of the run: any call, whatever its first argument is, fails the test. Everything the
run asks of processes (hardware, `llmfit`, Ollama, `ssh`) comes from the fixed probes of the test frame, so the
run needs no process at all. The two tests of `test_lmstudio_no_process.py` that read the functions alone stay;
this one is the same claim for the registered entry.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from modelroom.guided import run_guided
from modelroom.dialog import FileAsker

from fixture_lmstudio import chat_calls, fake_lms_on_path, lms_in_the_home_folder, lmstudio_servers, trap_processes
from fixture_room import fixed_room
from fixture_support import build_transport, guided_transport_mapping, offline_daemon, offline_servers, windows_probes
from test_guided_install import ANSWERS, RUN1

LIST_VALUE = "lmstudio unsloth/Qwen3.5-9B-GGUF@Q8_0"


def _guided(tmp_path: Path, monkeypatch, answers: dict):
    mark = fake_lms_on_path(tmp_path, monkeypatch)
    home = tmp_path / "lms-home"
    lms_in_the_home_folder(home, mark)
    assert shutil.which("lms") is not None  # the stand-in is found, or the probe would run empty
    for folder in ("results", "home"):
        (tmp_path / folder).mkdir(exist_ok=True)
    server = lmstudio_servers(["lmstudio_v1_models_loaded.json"])
    room = replace(fixed_room(), which=shutil.which, home=home, exists=lambda path: Path(path).exists())
    trap_processes(monkeypatch)
    lines: list[str] = []
    asker = FileAsker(answers)
    code = run_guided(
        asker, here=tmp_path / "results", pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport(guided_transport_mapping()), probes=windows_probes(), daemon=offline_daemon(),
        servers={**offline_servers(), "lmstudio": server}, now=RUN1, out=lines.append, colored=False, room=room,
    )
    return code, [line.strip() for line in lines], asker, server, mark


def test_the_trap_catches_a_direct_and_a_shell_call(monkeypatch):
    """The trap of the run below is no empty one: a direct call and a shell call, whatever the first argument,
    each end as a failed test."""
    import os
    import subprocess

    trap_processes(monkeypatch)
    with pytest.raises(pytest.fail.Exception):
        subprocess.Popen(["lms", "ls", "--json"])
    with pytest.raises(pytest.fail.Exception):
        os.system("powershell -c lms ls")


def test_modelroom_never_runs_lms(tmp_path, monkeypatch):
    answers = {**ANSWERS, "pull_package": LIST_VALUE}
    code, lines, asker, server, mark = _guided(tmp_path, monkeypatch, answers)
    text = "\n".join(lines)

    assert code == 0, text
    assert "server    LM Studio" in text and "1 loaded: qwen3.5-4b, context 8192" in text  # the start screen names it
    assert "LM Studio is not measured: it may load a model on request" in text  # step 4 says why there is no row
    assert "pull_package" in asker.asked
    assert any(line.startswith("install   #1  lms get --gguf unsloth/Qwen3.5-9B-GGUF@") for line in lines)  # printed, not run
    assert chat_calls(server) == []  # no chat, not even one
    assert not mark.exists()  # `lms` was never run; and the trap saw no process


def test_the_run_pulls_nothing_for_an_lms_row(tmp_path, monkeypatch):
    answers = {**ANSWERS, "pull_package": LIST_VALUE, "pull": True}
    code, lines, asker, _server, mark = _guided(tmp_path, monkeypatch, answers)
    assert code == 0 and "pull" not in asker.asked and not mark.exists()
    assert any(line.startswith("install   #1  lms get --gguf ") for line in lines)  # the entry was chosen and printed
    assert not any(line.startswith(("pulling", "✓ Pulled")) for line in lines)  # and nothing was pulled
