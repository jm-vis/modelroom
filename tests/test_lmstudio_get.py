"""AP9-LS-G: `modelroom/lmstudio_get.py`, the one place that runs `lms get`, with an injected runner.

`download(...)` takes the runner as `run=`: the tests pass a function that records the argument list and
returns an exit code, so no process starts here. The real `run_lms` runs in `test_lmstudio_runs_lms_once.py`
(the whole guided run under a trap that lets exactly one call through).
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelroom import lmstudio_get
from modelroom.lmstudio_get import LmsStartError, argv, download, program
from modelroom.screen import Screen

from fixture_room import fixed_room
from fixture_rows import package

REPO = "unsloth/Qwen3.5-4B-GGUF"
FILE = "Qwen3.5-4B-Q5_K_M.gguf"
NAME = f"{REPO}@Q5_K_M"
URL = f"https://huggingface.co/{NAME}"
SIZE = 3_000_000


def _package():
    return package(REPO, FILE, "Q5_K_M", SIZE, format="gguf")


def _guided(tmp_path: Path, *, system: str = "linux", which=lambda name: "/usr/bin/lms"):
    """A stand-in for the guided run as far as `download` reads it: `room`, `screen` and `failed`."""
    lines: list[str] = []
    failed: list[str] = []
    room = replace(fixed_room(), home=tmp_path, system=system, environ={}, which=which, exists=lambda path: Path(path).exists())
    guided = SimpleNamespace(room=room, screen=Screen(lines.append, colored=False), failed=failed.append)
    return guided, lines, failed


def _context(size: int = 8192):
    return SimpleNamespace(document=SimpleNamespace(scenario=SimpleNamespace(context_requested=size)))


def _row():
    return SimpleNamespace(rank=7, name=NAME, weights_gib=2.9)


def _lay_the_file(tmp_path: Path, size: int = SIZE) -> None:
    target = tmp_path / ".lmstudio" / "models" / REPO / FILE
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * size)


class _Runner:
    """Records the argument lists it is given and answers with `code` (or raises `error`)."""

    def __init__(self, code: int = 0, error: BaseException | None = None, after=None) -> None:
        self.calls: list[list[str]] = []
        self._code, self._error, self._after = code, error, after

    def __call__(self, call: list[str]) -> int:
        self.calls.append(list(call))
        if self._after is not None:
            self._after()
        if self._error is not None:
            raise self._error
        return self._code


# --- the argument list --------------------------------------------------------------------------------


def test_argv_is_the_list_of_five_with_the_program_the_url_and_yes(tmp_path):
    guided, _lines, _failed = _guided(tmp_path, which=lambda name: "/opt/lm studio/lms")
    assert argv("path", guided.room, NAME) == ["/opt/lm studio/lms", "get", "--gguf", URL, "--yes"]  # one element, even with a blank
    assert len(argv("path", guided.room, NAME)) == 5


def test_the_url_is_the_one_the_printed_command_has(tmp_path):
    from modelroom import lmstudio_install

    guided, _lines, _failed = _guided(tmp_path)
    assert lmstudio_install.command(NAME).endswith(argv("path", guided.room, NAME)[3])


def test_the_home_form_names_the_file_lms_form_tested(tmp_path):
    posix, _l, _f = _guided(tmp_path, system="linux")
    windows, _l, _f = _guided(tmp_path, system="win32")
    assert program("home", posix.room) == str(tmp_path / ".lmstudio" / "bin" / "lms")
    assert program("home", windows.room) == str(tmp_path / ".lmstudio" / "bin" / "lms.exe")
    assert argv("home", windows.room, NAME)[1:] == ["get", "--gguf", URL, "--yes"]


def test_the_path_form_names_what_which_gives(tmp_path):
    guided, _lines, _failed = _guided(tmp_path, which=lambda name: "C:/tools/lms.exe" if name == "lms" else None)
    assert program("path", guided.room) == "C:/tools/lms.exe"


def test_no_form_and_a_lost_lms_are_controlled_errors(tmp_path):
    guided, _lines, _failed = _guided(tmp_path, which=lambda name: None)
    with pytest.raises(LmsStartError):
        program(None, guided.room)
    with pytest.raises(LmsStartError):
        program("path", guided.room)  # the form said `path`, but `which` has nothing now: no other path is tried


# --- the steps after a yes ----------------------------------------------------------------------------


def test_exit_0_and_the_file_there_is_the_downloaded_line(tmp_path):
    _lay_the_file(tmp_path)
    guided, lines, failed = _guided(tmp_path)
    runner = _Runner(0)

    download(guided, _context(), _row(), _package(), "path", run=runner)

    assert runner.calls == [["/usr/bin/lms", "get", "--gguf", URL, "--yes"]]
    assert failed == []
    said = [line.strip() for line in lines]
    assert said[0] == f"download  lms get --gguf {URL} --yes"
    assert said[1] == (
        f"✓ Downloaded #7  {FILE} (2.9 GB) · in LM Studio: load it with context length 8192, then start its server"
    )
    assert said[2] == "then run modelroom again: it finds the server; LM Studio is not measured yet"
    assert len(said) == 3


def test_the_home_form_prints_the_shell_expression_and_runs_the_resolved_program(tmp_path):
    _lay_the_file(tmp_path)
    guided, lines, failed = _guided(tmp_path, system="win32", which=lambda name: None)
    runner = _Runner(0)

    download(guided, _context(), _row(), _package(), "home", run=runner)

    assert failed == []
    printed = lines[0].strip()
    assert printed.startswith('download  & "$env:USERPROFILE\\.lmstudio\\bin\\lms.exe" get --gguf ') and printed.endswith(" --yes")
    assert runner.calls == [[str(tmp_path / ".lmstudio" / "bin" / "lms.exe"), "get", "--gguf", URL, "--yes"]]  # not the expression


def test_exit_0_without_the_file_is_a_fault(tmp_path):
    guided, lines, failed = _guided(tmp_path)
    download(guided, _context(), _row(), _package(), "path", run=_Runner(0))
    assert failed == [f"#7: lms get ended, but {FILE} is not in ~/.lmstudio/models/{REPO} with the catalog size"]
    assert not any("Downloaded" in line for line in lines)


def test_a_file_of_another_size_is_a_fault_too(tmp_path):
    _lay_the_file(tmp_path, SIZE - 1)
    guided, _lines, failed = _guided(tmp_path)
    download(guided, _context(), _row(), _package(), "path", run=_Runner(0))
    assert len(failed) == 1 and "catalog size" in failed[0]


def test_exit_3_is_a_fault_and_the_file_is_not_looked_at(tmp_path):
    _lay_the_file(tmp_path)  # even a file there does not turn a failed exit into a success
    guided, lines, failed = _guided(tmp_path)
    download(guided, _context(), _row(), _package(), "path", run=_Runner(3))
    assert failed == ["#7: lms get ended with exit 3; a canceled download resumes with the same line"]
    assert not any("Downloaded" in line for line in lines)


def test_a_model_that_was_there_already_ends_in_the_downloaded_line(tmp_path):
    """`Model already downloaded` is exit 0 with the file in the folder: a success like any other."""
    _lay_the_file(tmp_path)
    guided, lines, failed = _guided(tmp_path)
    download(guided, _context(), _row(), _package(), "path", run=_Runner(0))
    assert failed == [] and any(line.strip().startswith("✓ Downloaded #7") for line in lines)


def test_the_file_is_read_after_the_call_not_before(tmp_path):
    guided, lines, failed = _guided(tmp_path)
    download(guided, _context(), _row(), _package(), "path", run=_Runner(0, after=lambda: _lay_the_file(tmp_path)))
    assert failed == [] and any(line.strip().startswith("✓ Downloaded #7") for line in lines)


def test_a_keyboard_interrupt_says_one_hint_and_goes_on(tmp_path):
    guided, lines, failed = _guided(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        download(guided, _context(), _row(), _package(), "path", run=_Runner(error=KeyboardInterrupt()))
    said = [line.strip() for line in lines]
    assert said.count("a canceled download resumes with the same line") == 1
    assert failed == [] and not any("Downloaded" in line or "ended" in line for line in said)


def test_an_oserror_at_the_start_is_one_line(tmp_path):
    guided, lines, failed = _guided(tmp_path)
    download(guided, _context(), _row(), _package(), "path", run=_Runner(error=OSError("[WinError 193]\nnot a valid\tprogram")))
    assert failed == ["#7: could not start lms get: [WinError 193] not a valid program"]
    assert not any("Downloaded" in line for line in lines)


def test_which_giving_nothing_starts_no_process(tmp_path):
    guided, lines, failed = _guided(tmp_path, which=lambda name: None)
    runner = _Runner(0)
    download(guided, _context(), _row(), _package(), "path", run=runner)
    assert runner.calls == []
    assert failed == ["#7: could not start lms get: lms is not in the PATH any more"]


def test_no_form_starts_no_process(tmp_path):
    guided, _lines, failed = _guided(tmp_path, which=lambda name: None)
    runner = _Runner(0)
    download(guided, _context(), _row(), _package(), None, run=runner)
    assert runner.calls == [] and len(failed) == 1 and failed[0].startswith("#7: could not start lms get:")


def test_the_context_of_the_run_is_in_the_hint(tmp_path):
    _lay_the_file(tmp_path)
    guided, lines, _failed = _guided(tmp_path)
    download(guided, _context(32768), _row(), _package(), "path", run=_Runner(0))
    assert "load it with context length 32768, then start its server" in "\n".join(lines)


def test_the_default_runner_is_run_lms_and_it_takes_the_list_alone(monkeypatch):
    """`run_lms(argv)` is `subprocess.run(argv, check=False)` and gives the exit code back; nothing else is passed."""
    seen: list = []

    def fake(*args, **kwargs):
        seen.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 5)

    monkeypatch.setattr(subprocess, "run", fake)
    assert lmstudio_get.run_lms(["lms", "get"]) == 5
    assert seen == [((["lms", "get"],), {"check": False})]
    assert download.__kwdefaults__["run"] is lmstudio_get.run_lms
