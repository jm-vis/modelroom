"""The room this machine has left: free graphics memory and free disk where Ollama keeps its models.

AP9-UX2 U8 (GF F5, F6, ADR 0010). Both are read live, never persisted, and both may be `unknown`
-- never `none`:

- **graphics memory**: `nvidia-smi --query-gpu=index,name,memory.free` (MiB), each card matched to
  the card of the profile with the same index; a count or a name that differs is `unknown`, as is a
  machine without an NVIDIA card or with unified memory; an AMD card counted from sysfs reads
  `mem_info_vram_total - mem_info_vram_used` by its PCI address, cards llmfit counted are `unknown`
  (AP9-GPU G9);
- **disk**: the place Ollama keeps its models, as an approximation (Ollama's own setting is not
  readable from outside, CONTRACTS.md, "The pull"): `OLLAMA_MODELS` where it is set, else the user
  default `~/.ollama/models` and, on Linux, the service default `/usr/share/ollama/.ollama/models`
  -- the one of them that exists; with none, the user default; with both, unclear. A folder that is
  not there yet is measured at the nearest folder above it that is.

Since AP9-RM (U10, ADR 0012) none of it is a head's budget any more: a `LiveRoom` carries what was
read, per machine, and `room_head` shows it as the free disk and the `loaded` hint -- what the daemon
holds (`loaded_models`, `GET /api/ps`) and the free graphics memory of now.

The path decision is pure (`ollama_models_candidates`); everything that touches the machine goes
through `RoomProbes`, so a test passes its own. `default_room_probes()` is the real machine and
refuses to run inside the test suite, where `tests/conftest.py` replaces it (Codex r2-5).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Callable, Mapping, Sequence

from .gpu_linux import VRAM_TOTAL_FILE, VRAM_USED_FILE, drm_slots, sysfs_bytes
from .llmfit import Runner, SubprocessRunner
from .measure_base import TIMEOUT_SECONDS, FileText, ListDir, list_directory, read_file_text

GIB = 1024**3
MIB = 1024**2
# The disk a pull needs besides the weight files (U8): the manifest, the layers Ollama writes
# alongside, and a margin. One GiB, in bytes.
PULL_DISK_MARGIN_BYTES = 1 * GIB


def pull_need_bytes(weights_bytes: float, held: bool = False) -> float:
    """The disk a pull of `weights_bytes` of weight files needs: the files plus the margin. The one rule
    of the check before a pull and of the marks of the table and the pull list (AP9-RM U11). A package
    the daemon provably holds already (`held`) needs nothing: it is not pulled (U11c, Codex r3-1)."""
    return 0.0 if held else weights_bytes + PULL_DISK_MARGIN_BYTES


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
    # The file and folder layer of sysfs, for the free memory of an AMD card (AP9-GPU G9).
    read_text: FileText = read_file_text
    list_dir: ListDir = list_directory
    # Whether a program is on the PATH (a runtime installed but not serving, AP9-SV2-I R1-1).
    which: Callable[[str], str | None] = lambda name: shutil.which(name)


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
    """The free disk where Ollama keeps its models: bytes and the drive, or why it is not known.

    `held`: the identities (`package_identity_key`) of the packages the daemon provably holds there
    already, by the load test's digest rule -- they need no disk (`pull_need_bytes`, U11c); empty
    where nobody asked the daemon.
    """

    free_bytes: int | None
    drive: str | None
    reason: str | None = None
    held: frozenset[tuple[str, str, str]] = frozenset()
    # AP9-SV2-I (I2): the free disk where vLLM keeps its models, the Hugging Face cache, next to
    # this one's place where Ollama keeps its own -- `None` where nobody read it. On that disk
    # `needs` is, per package identity, the bytes still to download (`hf_cache.vllm_disk`; an
    # identity it does not name needs its whole weights), `held` the packages already there, and
    # `relative` says that a cache variable of the machine is a relative path (`hf_cache.cache_is_relative`).
    vllm: "Disk | None" = None
    needs: Mapping[tuple[str, str, str], int] = field(default_factory=dict)
    relative: bool = False
    # AP9-SV2-R (R1): on a computer over SSH the server says itself whether the two places are on one
    # drive (`room.model_disk.same_as_ollama`) -- no drive of it is ever named here. `None`: compare the drives.
    same_place: bool | None = None

    @property
    def known(self) -> bool:
        return self.free_bytes is not None and self.drive is not None


def ollama_models_candidates(environ: Mapping[str, str], home: Path, system: str) -> list[Path]:
    """Where Ollama may keep its models: `OLLAMA_MODELS` alone where it is set, else the defaults."""
    configured = environ.get("OLLAMA_MODELS")
    if configured:
        # `~` is the `home` passed in, never this process's own (Codex r1 m5): the decision stays pure.
        # Every separator behind it goes, not just one: `~//models` would join a root path onto home.
        if configured == "~" or configured.startswith(("~/", "~\\")):
            rest = configured[1:].lstrip("/\\")
            return [home / rest] if rest else [home]
        return [Path(configured)]
    candidates = [home / ".ollama" / "models"]
    if system.startswith("linux"):
        candidates.append(_SERVICE_DEFAULT)
    return candidates


def free_disk(room: RoomProbes) -> Disk:
    """The free disk of the place Ollama keeps its models, measured now (F6: an approximation)."""
    try:
        place = _models_place(room)
    except (OSError, ValueError, RuntimeError):
        return Disk(None, None)
    if place is None:
        return Disk(None, None, UNCLEAR_REASON)
    return free_disk_at(room, place)


def free_disk_at(room: RoomProbes, place: Path) -> Disk:
    """The free disk at `place`, or at the nearest folder above it that exists (a place not created
    yet); unknown, without a reason, where it cannot be measured. Ollama's place and vLLM's both."""
    try:
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
        if not math.isfinite(free) or free < 0 or int(parts[0]) in found:
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


@dataclass(frozen=True)
class LoadedModel:
    """One model the daemon holds right now (`GET /api/ps`): its name, its size and the part of it
    on the graphics card, both in bytes."""

    name: str
    size_bytes: int
    vram_bytes: int


@dataclass(frozen=True)
class LiveRoom:
    """What was read live about one machine, only in memory (AP9-RM U10): never an input to a fit.

    `disk`: the free disk where Ollama keeps its models, `None` when it was not read. `loaded`: the
    models the daemon holds, `()` when it holds none, `None` when nobody asked or the answer could
    not be read. `graphics_free_mib`: the free memory of each card of the profile, in MiB, in the
    profile's order, `None` when unknown.
    """

    disk: Disk | None = None
    loaded: tuple[LoadedModel, ...] | None = None
    graphics_free_mib: tuple[float, ...] | None = None


def live_room(room: RoomProbes, profile, daemon=None, disk: Disk | None = None) -> LiveRoom:
    """This machine read now: the disk (or the one read a moment ago, `disk`), the free graphics
    memory and, with a `daemon`, what it holds."""
    free = free_graphics_mib(room, profile)
    return LiveRoom(
        disk=disk if disk is not None else free_disk(room),
        loaded=loaded_models(daemon) if daemon is not None else None,
        graphics_free_mib=tuple(free) if free else None,
    )


def loaded_models(daemon) -> tuple[LoadedModel, ...] | None:
    """The models the daemon holds (`GET /api/ps`), or `None` for every answer that cannot be read.

    A reader of its own next to the load test's (`loadtest._observe`), which raises instead: a hint
    that cannot be read is no hint, and says nothing (W4). An entry without a name or without two
    whole, non-negative sizes is left out.
    """
    from .daemon import PS_PATH

    try:
        answer = daemon("GET", PS_PATH, None)
        data = answer.json() if answer.status == 200 else None
    except Exception:  # noqa: BLE001 -- every failure of a hint is silence, never a fault of the run (W4)
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    return tuple(model for model in (_loaded(entry) for entry in models) if model is not None)


def _loaded(entry: object) -> LoadedModel | None:
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
        return None
    size, vram = entry.get("size"), entry.get("size_vram")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (size, vram)):
        return None
    return LoadedModel(entry["name"], size, vram)


def free_graphics_mib(room: RoomProbes, profile) -> list[float] | None:
    """The free graphics memory of the counted cards, by the source that counted them (AP9-GPU G9, D14).

    `nvidia-smi` as before; sysfs per card by its PCI address; llmfit's cards are never read here:
    llmfit gives no free memory, so they are `unknown`.
    """
    source = getattr(profile, "vram_source", None)
    if source == "llmfit":
        return None
    if source == "sysfs":
        return free_sysfs_mib(room, [adapter.pci_slot for adapter in profile.adapters if adapter.counted])
    return free_vram_mib(room, profile_cards(profile))


def free_sysfs_mib(room: RoomProbes, slots: Sequence[str | None]) -> list[float] | None:
    """`mem_info_vram_total - mem_info_vram_used` of each card, found by its address; all or nothing."""
    if not slots or None in slots:
        return None
    cards = drm_slots(room.read_text, room.list_dir)
    free = []
    for slot in slots:
        found = cards.get(slot, [])
        if len(found) != 1:
            return None
        total = sysfs_bytes(room.read_text, found[0], VRAM_TOTAL_FILE)
        used = sysfs_bytes(room.read_text, found[0], VRAM_USED_FILE, minimum=0)
        if total is None or used is None or used > total:
            return None
        free.append((total - used) / MIB)
    return free


__all__ = [
    "Disk",
    "LiveRoom",
    "LoadedModel",
    "live_room",
    "loaded_models",
    "profile_cards",
    "NVIDIA_SMI_FREE_ARGS",
    "PULL_DISK_MARGIN_BYTES",
    "RoomProbes",
    "default_room_probes",
    "disk_words",
    "free_disk",
    "free_disk_at",
    "free_graphics_mib",
    "free_sysfs_mib",
    "free_vram_mib",
    "gb",
    "graphics_words",
    "ollama_models_candidates",
]
