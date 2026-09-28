"""AP9-GPU G9: the free graphics memory by the source that counted the cards (D14)."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from fixture_support import v2_profile
from modelroom.llmfit import FixtureRunner
from modelroom.measure_base import FixtureFiles
from modelroom.profile import HardwareProfile
from modelroom.room import NVIDIA_SMI_FREE_ARGS, RoomProbes, free_graphics_mib, graphics_words

DRM = "/sys/class/drm"
GIB = 1024**3
SLOTS = ("0000:03:00.0", "0000:0c:00.0")


def _card(slot: str, gib: float) -> dict:
    return {"vendor": "amd", "name": "AMD Radeon RX 7900 XTX", "pci_id": "1002:744c", "pci_slot": slot, "memory_gib": gib,
            "memory_kind": "dedicated", "form": "discrete", "source": "sysfs", "evidence": "kfd", "backend": "rocm",
            "usable": "yes", "counted": True, "reason": None}


def _amd_profile(*slots: str) -> HardwareProfile:
    data = v2_profile().model_dump(mode="json")
    cards = [_card(slot, 24.0) for slot in slots]
    gpus = [{"index": i, "name": "AMD Radeon RX 7900 XTX", "vram_gib": 24.0, "vram_source": "sysfs"} for i in range(len(slots))]
    data.update(gpu_state="measured" if len(slots) == 1 else "multi_gpu", vram_gib=24.0 * len(slots), vram_source="sysfs",
                gpu_name="AMD Radeon RX 7900 XTX", gpus=gpus, adapters=cards)
    return HardwareProfile.model_validate(data)


def _room(files: dict[str, str], runner: FixtureRunner | None = None) -> RoomProbes:
    layer = FixtureFiles(files)
    return RoomProbes(environ={}, home=Path("/srv/ada"), system="linux", runner=runner or FixtureRunner({}), read_text=layer, list_dir=layer.list_dir)


def _sysfs(**cards: tuple[str, str, str]) -> dict[str, str]:
    files = {}
    for card, (slot, total, used) in cards.items():
        files[f"{DRM}/{card}/device/uevent"] = f"PCI_SLOT_NAME={slot}\n"
        if total is not None:
            files[f"{DRM}/{card}/device/mem_info_vram_total"] = total
        if used is not None:
            files[f"{DRM}/{card}/device/mem_info_vram_used"] = used
    return files


def test_two_amd_cards_each_by_their_address():
    files = _sysfs(card1=(SLOTS[1], str(24 * GIB), str(4 * GIB)), card0=(SLOTS[0], str(24 * GIB), "0"))
    free = free_graphics_mib(_room(files), _amd_profile(*SLOTS))
    assert free == [24 * 1024.0, 20 * 1024.0]
    assert graphics_words(free) == "graphics 24.0 + 20.0 GB free now"


@pytest.mark.parametrize(
    "files",
    [
        _sysfs(card0=(SLOTS[0], None, "0")),
        _sysfs(card0=(SLOTS[0], "lots", "0")),
        _sysfs(card0=(SLOTS[0], str(24 * GIB), str(25 * GIB))),
        _sysfs(card0=(SLOTS[0], str(24 * GIB), "0"), card1=(SLOTS[0], str(24 * GIB), "0")),
        _sysfs(card0=("0000:0f:00.0", str(24 * GIB), "0")),
    ],
    ids=["missing", "unreadable", "used-above-total", "two-matches", "no-match"],
)
def test_any_card_that_does_not_read_leaves_all_unknown(files):
    assert free_graphics_mib(_room(files), _amd_profile(SLOTS[0])) is None


def test_cards_llmfit_counted_are_unknown():
    data = v2_profile().model_dump(mode="json")
    data.update(vram_source="llmfit", gpus=[{**data["gpus"][0], "vram_source": "llmfit"}])
    runner = FixtureRunner({})  # nvidia-smi is not asked: any call would raise
    assert free_graphics_mib(_room({}, runner), HardwareProfile.model_validate(data)) is None and runner.calls == []


def test_an_entered_profile_and_the_nvidia_path_stay_as_they_were():
    answer = subprocess.CompletedProcess(list(NVIDIA_SMI_FREE_ARGS), 0, stdout="0, Nova GPU, 10240\n", stderr="")
    room = _room({}, FixtureRunner({NVIDIA_SMI_FREE_ARGS: answer}))
    assert free_graphics_mib(room, v2_profile()) == [10240.0]
    data = v2_profile().model_dump(mode="json")
    data.update(origin="entered", os_fingerprint="none", os_fingerprint_source="none", ram_physical_source="entered",
                vram_source="entered", gpu_state="entered", gpu_name=None,
                gpus=[{"index": 0, "name": None, "vram_gib": 11.94, "vram_source": "entered"}])
    entered = HardwareProfile.model_validate(data)
    assert free_graphics_mib(replace(room), entered) == [10240.0]
