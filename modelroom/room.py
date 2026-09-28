"""The room this machine has left: free graphics memory and free disk where Ollama keeps its models.

AP9-UX2 U8 (GF F5, F6, ADR 0010). Both are read live, never persisted, and both may be `unknown`
-- never `none`:

- **graphics memory**: `nvidia-smi --query-gpu=index,name,memory.free` (MiB), each card matched to
  the card of the profile with the same index; a count or a name that differs is `unknown`, as is a
  machine without an NVIDIA card or with unified memory;
- **disk**: the place Ollama keeps its models, as an approximation (Ollama's own setting is not
  readable from outside, CONTRACTS.md, "The pull"): `OLLAMA_MODELS` where it is set, else the user
  default `~/.ollama/models` and, on Linux, the service default `/usr/share/ollama/.ollama/models`
  -- the one of them that exists; with none, the user default; with both, unclear. A folder that is
  not there yet is measured at the nearest folder above it that is.

The path decision is pure (`ollama_models_candidates`); everything that touches the machine goes
through `RoomProbes`, so a test passes its own. `default_room_probes()` is the real machine and
refuses to run inside the test suite, where `tests/conftest.py` replaces it (Codex r2-5).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Callable, Mapping, Sequence

from .measure import TIMEOUT_SECONDS
from .llmfit import Runner, SubprocessRunner

GIB = 1024**3
MIB = 1024**2
# The disk a pull needs besides the weight files (U8): the manifest, the layers Ollama writes
# alongside, and a margin. One GiB, in bytes.
PULL_DISK_MARGIN_BYTES = 1 * GIB
NVIDIA_SMI_FREE_ARGS = ("nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits")
UNKNOWN = "unknown"
UNCLEAR_REASON = "two places Ollama may use"
NETWORK_SHARE = "network share"
_SERVICE_DEFAULT = Path("/usr/share/ollama/.ollama/models")


@dataclass(frozen=True)
class RoomProbes:
    """Everything the room is read from, in one injectable value (the shape of `measure.Probes`)."""

    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    home: Path = field(default_factory=Path.home)
    system: str = sys.platform
    exists: Callable[[Path], bool] = lambda path: path.exists()
    resolve: Callable[[Path], Path] = lambda path: path.resolve()
    disk_usage: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free
    mount_of: Callable[[Path], str] = lambda path: _mount_point(path)
    runner: Runner = field(default_factory=lambda: SubprocessRunner(timeout=TIMEOUT_SECONDS))


def default_room_probes() -> RoomProbes:
    """This machine's room. Never inside the test suite: a test reads a fixed room instead
    (`tests/conftest.py` replaces this function; `real_room_probes` is the guarded original)."""
    return real_room_probes()


def real_room_probes() -> RoomProbes:
    """The real probes, refused where `pytest` is loaded: no test may depend on the machine it runs on."""
    if "pytest" in sys.modules:
        raise RuntimeError("the real room probes were reached from a test; pass room= or use the conftest fixture")
    return RoomProbes()


@dataclass(frozen=True)
class Disk:
    """The free disk where Ollama keeps its models: bytes and the drive, or why it is not known."""

    free_bytes: int | None
    drive: str | None
    reason: str | None = None

    @property
    def known(self) -> bool:
        return self.free_bytes is not None and self.drive is not None


def ollama_models_candidates(environ: Mapping[str, str], home: Path, system: str) -> list[Path]:
    """Where Ollama may keep its models: `OLLAMA_MODELS` alone where it is set, else the defaults."""
    configured = environ.get("OLLAMA_MODELS")
    if configured:
        return [Path(configured).expanduser()]
    candidates = [home / ".ollama" / "models"]
    if system.startswith("linux"):
        candidates.append(_SERVICE_DEFAULT)
    return candidates


def free_disk(room: RoomProbes) -> Disk:
    """The free disk of the place Ollama keeps its models, measured now (F6: an approximation)."""
    try:
        place = _models_place(room)
        if place is None:
            return Disk(None, None, UNCLEAR_REASON)
        measured = _nearest_existing(room, room.resolve(place))
        return Disk(room.disk_usage(measured), _drive_name(room, measured))
    except (OSError, ValueError, RuntimeError):
        return Disk(None, None)


def _models_place(room: RoomProbes) -> Path | None:
    """The one place to measure, or `None` where two defaults both exist (unclear)."""
    candidates = ollama_models_candidates(room.environ, room.home, room.system)
    if room.environ.get("OLLAMA_MODELS"):
        return candidates[0]  # taken even where it does not exist yet
    existing = [path for path in candidates if room.exists(path)]
    if len(existing) > 1:
        return None
    return existing[0] if existing else candidates[0]


def _nearest_existing(room: RoomProbes, path: Path) -> Path:
    """`path`, or the nearest folder above it that exists (a models folder not created yet)."""
    current = path
    while not room.exists(current):
        if current.parent == current:
            raise OSError(f"no part of {path} exists")
        current = current.parent
    return current


def _drive_name(room: RoomProbes, path: Path) -> str:
    """`C:` on Windows, `network share` for a UNC path (never its server), else the mount point."""
    if room.system == "win32":
        windows = PureWindowsPath(str(path))
        if windows.drive.startswith("\\\\") or str(path).startswith("\\\\"):
            return NETWORK_SHARE
        return windows.drive or NETWORK_SHARE
    return room.mount_of(path)


def _mount_point(path: Path) -> str:
    """The mount point `path` lies on (POSIX): the nearest folder above it that is one."""
    current = path
    while not os.path.ismount(current) and current.parent != current:
        current = current.parent
    return str(current)


def free_vram_mib(room: RoomProbes, cards: Sequence[tuple[int, str | None]]) -> list[float] | None:
    """The free graphics memory of each card of the profile, in MiB, in its order; `None` if unknown.

    `cards` are the profile's `(index, name)`; the answer must name exactly these cards by index, each
    with the same name, or nothing is said about any of them.
    """
    if not cards:
        return None
    try:
        result = room.runner(list(NVIDIA_SMI_FREE_ARGS))
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    read = _read_free(result.stdout)
    if read is None or sorted(read) != sorted(index for index, _name in cards):
        return None
    if any(name is not None and read[index][0] != name for index, name in cards):
        return None
    return [read[index][1] for index, _name in cards]


def _read_free(text: str) -> dict[int, tuple[str, float]] | None:
    """`index -> (name, free MiB)` of every line, or `None` where a line does not read."""
    found: dict[int, tuple[str, float]] = {}
    for line in (row.strip() for row in text.splitlines()):
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit():
            return None
        try:
            free = float(parts[2])
        except ValueError:
            return None
        if free < 0 or int(parts[0]) in found:
            return None
        found[int(parts[0])] = (parts[1], free)
    return found or None


def gb(value_bytes: float) -> str:
    """Bytes as the screen writes them: binary GiB, called `GB`, one decimal."""
    return f"{value_bytes / GIB:.1f} GB"


def graphics_words(free_mib: list[float] | None) -> str:
    """`graphics 10.2 GB free now`, one number per card joined by `+`, or `graphics unknown`."""
    if not free_mib:
        return f"graphics {UNKNOWN}"
    return "graphics " + " + ".join(f"{mib / 1024:.1f}" for mib in free_mib) + " GB free now"


def disk_words(disk: Disk) -> str:
    """`disk 41.2 GB free on C: (Ollama models)`, or `disk unknown` with the reason where there is one."""
    if not disk.known:
        return f"disk {UNKNOWN}" + (f" ({disk.reason})" if disk.reason else "")
    return f"disk {gb(disk.free_bytes)} free on {disk.drive} (Ollama models)"


def profile_cards(profile) -> list[tuple[int, str | None]]:
    """The cards of a profile whose free memory `nvidia-smi` can say: none for unified memory."""
    if profile is None or profile.gpu_state not in ("measured", "multi_gpu", "entered"):
        return []
    return [(gpu.index, gpu.name) for gpu in profile.gpus]


def free_words(room: RoomProbes, profile, dot: str = "·") -> str:
    """The `free` line of this machine's table head: graphics and disk, each measured now."""
    return f"{graphics_words(free_vram_mib(room, profile_cards(profile)))} {dot} {disk_words(free_disk(room))}"


__all__ = [
    "Disk",
    "free_words",
    "profile_cards",
    "NVIDIA_SMI_FREE_ARGS",
    "PULL_DISK_MARGIN_BYTES",
    "RoomProbes",
    "default_room_probes",
    "disk_words",
    "free_disk",
    "free_vram_mib",
    "gb",
    "graphics_words",
    "ollama_models_candidates",
]
