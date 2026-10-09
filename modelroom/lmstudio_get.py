"""The one place that starts `lms`: the download of an LM Studio entry after a yes (AP9-LS-G).

Since AP9-LS (R5, 2026-10-08) modelroom ran no `lms`: every call wakes LM Studio's service in the
background, so even `lms ls` was not a read. After the reader has chosen an LM Studio entry of this machine,
has seen the line and the hint and has said yes to `Download #7 with lms now?`, the wake-up is wanted: the
reader loads the model in LM Studio next, and the service runs then anyway. So this module runs
`lms get --gguf https://huggingface.co/<owner>/<repo>@<quant> --yes` once, for that yes, and nowhere else:
not to detect `lms`, not to measure, not to load or unload, not in `modelroom pull`, not for another computer.

The call is a list, never a string and never a shell: `[<program>, "get", "--gguf", <url>, "--yes"]`. The
program is a single element, resolved from what `lms_form` found (the `PATH` search of `room.which`, or the
file in `~/.lmstudio/bin`); the shell expression that is printed for the reader is never handed to a process.
Standard input, output and error go through to `lms` (no capture, no live line of modelroom's own), the
working directory and the environment are those of the running process. The only process call is the
`run` of `subprocess` with `argv` and `check=False`, in `run_lms`; `tests/test_lmstudio_no_process.py`
reads this file by its syntax tree and allows nothing else, and no other module of LM Studio imports
`subprocess`.

After the call the file is looked at again (`package_on_disk`, `stat` only): the exit code alone does not say
that the package lies in LM Studio's folder with the size of the catalog. `Model already downloaded` (exit 0, the
file is there) is a success like any other.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Callable

from . import lmstudio_install
from .intro import label_line
from .lmstudio_install import HF_URL_PREFIX
from .screen import downloaded_line

if TYPE_CHECKING:  # pragma: no cover - the run and its context are passed in, never constructed here
    from .guided import GuidedRun
    from .guided_install import PullContext
    from .pull_choice import PullRow
    from .contracts import Package
    from .room import RoomProbes

DOWNLOAD_LABEL = "download"
CANCELED_HINT = "a canceled download resumes with the same line"
LOAD_HINT = "in LM Studio: load it with context length {context}, then start its server"
AGAIN_HINT = "then run modelroom again: it finds the server and measures the loaded model"


class LmsStartError(Exception):
    """`lms` cannot be named any more: no process is started, and the message says why in one line."""


def program(form: str | None, room: "RoomProbes") -> str:
    """The program to start, as one element: the path `room.which("lms")` gives (form `path`), or the file in
    `~/.lmstudio/bin` that `lms_form` tested (form `home`). Anything else is an `LmsStartError`; no other
    path is tried."""
    if form == "path":
        found = room.which("lms")
        if not found:
            raise LmsStartError("lms is not in the PATH any more")
        return str(found)
    if form == "home":
        return str(room.home / ".lmstudio" / "bin" / ("lms.exe" if room.system == "win32" else "lms"))
    raise LmsStartError("lms was not found on this machine")


def argv(form: str | None, room: "RoomProbes", name: str) -> list[str]:
    """`[<program>, "get", "--gguf", "https://huggingface.co/<owner>/<repo>@<quant>", "--yes"]`: the URL is
    the one `lmstudio_install.command(name)` prints, `--yes` stands last (the question was asked here)."""
    return [program(form, room), "get", "--gguf", HF_URL_PREFIX + name, "--yes"]


def run_lms(argv: list[str]) -> int:
    """The one process of LM Studio's modules: `lms` runs with the terminal of the reader, and its exit code comes back."""
    return subprocess.run(argv, check=False).returncode


def _one_line(text: str) -> str:
    return " ".join(text.split())


def download(
    guided: "GuidedRun", context: "PullContext", row: "PullRow", package: "Package", form: str | None,
    *, run: Callable[[list[str]], int] = run_lms,
) -> None:
    """Steps 6 to 8 of the choice of an LM Studio entry, after a yes: the line, the call, the check of the file.

    `guided` is the guided run (`screen`, `room`, `failed`); `run` only the runner that takes the argument list and
    returns the exit code (a test passes its own). A fault is one line through `guided.failed`; a
    `KeyboardInterrupt` says once that the same line resumes the download and goes on (the central handler
    ends the run with 130)."""
    room, screen, rank, name = guided.room, guided.screen, row.rank, str(row.name)
    shown = lmstudio_install.command_for(form, name, room.system) + " --yes"
    screen.write(label_line(DOWNLOAD_LABEL, [("class:command", shown)]))
    try:
        call = argv(form, room, name)
        code = run(call)
    except LmsStartError as exc:
        guided.failed(f"#{rank}: could not start lms get: {_one_line(str(exc))}")
        return
    except OSError as exc:
        guided.failed(f"#{rank}: could not start lms get: {_one_line(str(exc))}")
        return
    except KeyboardInterrupt:
        screen.note(CANCELED_HINT)
        raise
    if code != 0:
        guided.failed(f"#{rank}: lms get ended with exit {code}; {CANCELED_HINT}")
        return
    place = lmstudio_install.model_place(room.environ, room.home, room.system)[0]
    file = lmstudio_install.weight_files(package)[0].name
    if lmstudio_install.package_on_disk(package, place) is not True:
        guided.failed(f"#{rank}: lms get ended, but {file} is not in ~/.lmstudio/models/{package.repo} with the catalog size")
        return
    hint = LOAD_HINT.format(context=context.document.scenario.context_requested)
    screen.write(downloaded_line(rank, file, row.weights_gib, screen.glyphs, hint))
    screen.write(label_line("", [("class:note", AGAIN_HINT)]))


__all__ = ["AGAIN_HINT", "CANCELED_HINT", "DOWNLOAD_LABEL", "LmsStartError", "argv", "download", "program", "run_lms"]
