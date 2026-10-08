"""A fixed room for the tests: the free graphics memory and disk of the fixture laptop (AP9-UX2 U8).

Not a test module (no `test_` prefix). `tests/conftest.py` puts `fixed_room()` in place of
`room.default_room_probes` for the whole suite, so no test depends on the machine it runs on
(Codex r2-5); a test that needs other numbers passes `room=fixed_room(...)` itself. The card is
the one of `windows_probes()` (`tests/fixtures/nvidia_smi_one_gpu.csv`), the drive `C:`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from modelroom.llmfit import FixtureRunner
from modelroom.room import NVIDIA_SMI_FREE_ARGS, RoomProbes

LAPTOP_CARD = "NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU"
FREE_VRAM_MIB = 10_444.0  # 10.2 GB
FREE_DISK_BYTES = 44_236_955_648  # 41.2 GB
HOME = Path("C:\\fixture-home")


def exists(path: Path) -> bool:
    """Every path of the fixture room exists -- except LM Studio's own folder: the fixture laptop has no LM
    Studio installed (a guided run meets no `lms.exe` there, AP9-LS-E), so no test of an Ollama machine finds
    a third runtime that nobody put there."""
    return ".lmstudio" not in Path(path).parts


def fixed_room(
    free_disk_bytes: int | None = FREE_DISK_BYTES,
    free_vram_mib: float | None = FREE_VRAM_MIB,
    *,
    card: str = LAPTOP_CARD,
    environ: dict | None = None,
) -> RoomProbes:
    """The room of the fixture laptop, with the two numbers given; `None` makes that number unknown."""
    if free_vram_mib is None:
        answer: subprocess.CompletedProcess | Exception = FileNotFoundError("nvidia-smi")
    else:
        answer = subprocess.CompletedProcess(list(NVIDIA_SMI_FREE_ARGS), 0, stdout=f"0, {card}, {free_vram_mib:g}\n", stderr="")

    def disk_usage(path: Path) -> int:
        if free_disk_bytes is None:
            raise OSError("no disk in this fixture")
        return free_disk_bytes

    return RoomProbes(
        environ=environ or {},
        home=HOME,
        system="win32",
        exists=exists,
        resolve=lambda path: path,
        disk_usage=disk_usage,
        mount_of=lambda path: "/",
        runner=FixtureRunner({NVIDIA_SMI_FREE_ARGS: answer}),
        which=lambda name: None,  # no program on this room's PATH, whatever the machine running the suite has
    )


class SequenceDisk:
    """A disk whose free bytes change from one reading to the next (the head, then the pull check)."""

    def __init__(self, *readings: int) -> None:
        self._readings = list(readings)
        self.calls = 0

    def __call__(self, path: Path) -> int:
        self.calls += 1
        return self._readings[min(self.calls, len(self._readings)) - 1]


def room_with_disk(disk) -> RoomProbes:
    """`fixed_room()` with its disk read through `disk` (e.g. a `SequenceDisk`)."""
    from dataclasses import replace

    return replace(fixed_room(), disk_usage=disk)
