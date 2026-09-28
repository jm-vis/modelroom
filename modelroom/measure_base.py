"""The two injected layers of the hardware measurement and the value one source returns.

Every source of `modelroom.measure` and the `gpu_*` modules is a pure function over a command layer
(`Runner`, `(args) -> CompletedProcess`, `modelroom/llmfit.py`) and a file layer (`FileText`,
`(path) -> text`, plus `ListDir`, `(folder) -> names`). Nothing here reads `sys.platform`, spawns a
process by itself or raises for a source that is unavailable. Moved out of `measure.py` (AP9-GPU G0,
size guard); `measure.py` exports every name back.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Literal

from .llmfit import Runner
from .profile import GpuAdapter, GpuState

GIB = 1024**3
MIB = 1024**2
KIB = 1024
TIMEOUT_SECONDS = 10.0

FileText = Callable[[str], str]
# The names in one folder (files and folders alike), or an `OSError` naming it.
ListDir = Callable[[str], "list[str]"]
MemoryBytes = Callable[[], int]


def read_file_text(path: str) -> str:
    """The real file layer: the file's text, or an `OSError` naming it."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def list_directory(path: str) -> list[str]:
    """The real folder layer: the sorted names in `path`, or an `OSError` naming it."""
    return sorted(os.listdir(path))


def no_listing(path: str) -> list[str]:
    """A folder layer that knows no folder: every listing is a `FileNotFoundError`.

    The layer a caller gets who injects a file layer of its own without a folder layer: the real
    folders of the machine the suite runs on must never mix with a test's files.
    """
    raise FileNotFoundError(f"{path}: no such folder")


class FixtureFiles:
    """Test file layer: a path-to-text mapping; every other path is a `FileNotFoundError`.

    The same shape `FixtureRunner`/`FixtureTransport` have and for the same reason -- there is
    no mocking library in this project, a test that needs a particular `/proc` content
    constructs one of these and passes it in like any other value.

    `list_dir` is the folder layer of the same mapping (AP9-GPU G0): a folder lists the next
    part of every path below it, a folder no path lies below is a `FileNotFoundError`.
    """

    def __init__(self, files: dict[str, str]) -> None:
        self._files = dict(files)
        self.reads: list[str] = []

    def __call__(self, path: str) -> str:
        self.reads.append(path)
        if path not in self._files:
            raise FileNotFoundError(f"{path}: no such file")
        return self._files[path]

    def list_dir(self, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/"
        names = {name[len(prefix):].split("/", 1)[0] for name in self._files if name.startswith(prefix)}
        if not names:
            raise FileNotFoundError(f"{path}: no such folder")
        return sorted(names)


@dataclass(frozen=True)
class Reading:
    """One measured quantity in GiB with its source, or no value and the reason why not."""

    value: float | None
    source: Literal["os", "unknown"]
    reason: str | None = None


@dataclass(frozen=True)
class GpuReading:
    """The GPU verdict: the state the fit gates on, the VRAM with its source, a note, and every
    card `nvidia-smi` measured (none for every state but `measured` and `multi_gpu`)."""

    gpu_state: GpuState
    vram_gib: float | None
    vram_source: Literal["nvidia-smi", "none", "unknown"]
    gpu_name: str | None = None
    note: str | None = None
    adapters: tuple[GpuAdapter, ...] = ()


def _whole_number(text: str) -> int | None:
    """`text` as an integer, or `None` when it is not one this process can convert.

    The range is left to each caller: a negative or zero value is rejected where it matters
    (`_gib_reading`, the `memory.max` filter, the adapter's index and memory).

    `int(text)` is not enough on its own: a digit string above Python's int/str conversion limit
    (4300 digits by default) raises `ValueError`, and a source must never raise -- not even for
    output no real kernel or driver would print (probe: `MemTotal` with 5000 digits).
    """
    try:
        return int(text)
    except ValueError:
        return None


def _run(runner: Runner, args: tuple[str, ...]) -> tuple[str | None, str | None]:
    """`(stdout, None)` when the command answered with exit 0, else `(None, reason)`.

    Every way a command can fail ends here: not installed, not runnable, no answer within the
    timeout the runner enforces, or a non-zero exit code. Nothing propagates to the caller.
    """
    try:
        result = runner(list(args))
    except FileNotFoundError as exc:
        return None, f"{args[0]} is not installed ({exc})"
    except subprocess.TimeoutExpired:
        return None, f"{args[0]} did not answer within {TIMEOUT_SECONDS:.0f} s"
    except OSError as exc:
        return None, f"{args[0]} could not be run ({exc})"
    if result.returncode != 0:
        return None, f"{args[0]} failed (exit {result.returncode}): {result.stderr.strip()}"
    return result.stdout, None


def _gib(total: object) -> float | None:
    """`total` bytes in GiB, or `None` when that is not a usable number.

    Every memory value in the contract is `> 0`, so a `0` from a call that reported success, a
    negative number, an infinity, a value so small it rounds to `0.00` GiB, or an integer too
    large to turn into a float all come back `None` instead of reaching the contract or raising.
    Both `math.isfinite` and the division itself raise `OverflowError` for an integer of that
    size (probe: `nvidia-smi` reporting 400 digits of memory), which counts as "not usable" --
    the same rule `llmfit.py::_is_finite_number` follows.
    """
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        return None
    try:
        if not math.isfinite(total) or total <= 0:
            return None
        gib = round(total / GIB, 2)
    except OverflowError:
        return None
    return gib if gib > 0 else None


def _gib_reading(total: object, label: str) -> Reading:
    """`_gib` as a `Reading`: the value with source `os`, or unknown naming the source."""
    gib = _gib(total)
    if gib is None:
        return Reading(None, "unknown", f"{label} is not a usable number of bytes: {str(total)[:40]}")
    return Reading(gib, "os")
