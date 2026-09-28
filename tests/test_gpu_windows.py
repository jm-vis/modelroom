"""AP9-GPU G3: Windows -- the video controllers, and the registry's memory size shown, never counted."""

from __future__ import annotations

import pytest

from fixture_gpu import cim, fixture, gpu_runner, registry
from modelroom.gpu import measure_gpu
from modelroom.gpu_windows import (
    WINDOWS_ADAPTER_ARGS,
    WINDOWS_REGISTRY_ARGS,
    list_video_controllers,
    registry_bytes,
    registry_memory,
)

NVIDIA_PNP = "PCI\\VEN_10DE&DEV_2F38&SUBSYS_00000000&REV_A1\\X"
NVIDIA_KEY = "pci\\ven_10de&dev_2f38&subsys_00000000"
AMD_PNP = "PCI\\VEN_1002&DEV_744C&SUBSYS_00000000&REV_C8\\X"
LAPTOP = {"cim": fixture("win_video_controllers_laptop.json.txt"), "registry": fixture("win_registry_display_laptop.json.txt")}
SMI = "0, NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU, 12227\n"


def test_one_controller_alone_and_a_list_both_read():
    one, _ = list_video_controllers(gpu_runner(cim=cim(("NVIDIA L4", NVIDIA_PNP))))
    two, _ = list_video_controllers(gpu_runner(cim=LAPTOP["cim"]))
    assert [entry.vendor_id for entry in one] == ["10de"]
    assert [(entry.name, entry.vendor_id, entry.device_id) for entry in two] == [
        ("Intel(R) Graphics", "8086", "7d67"),
        ("NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU", "10de", "2f38"),
    ]


def test_the_listing_never_asks_for_adapterram():
    assert not any("AdapterRAM" in part for part in WINDOWS_ADAPTER_ARGS)
    assert not any("AdapterRAM" in part for part in WINDOWS_REGISTRY_ARGS)


def test_the_registry_query_never_reads_the_32_bit_memorysize():
    query = WINDOWS_REGISTRY_ARGS[-1]
    assert "HardwareInformation.qwMemorySize" in query and "'HardwareInformation.MemorySize'" not in query


@pytest.mark.parametrize(
    ("value", "number"),
    [
        (12820938752, 12820938752),
        ([0, 0, 0, 0, 3, 0, 0, 0], 12884901888),
        ({"value": [0, 0, 0, 0, 1, 0, 0, 0], "Count": 8}, 4294967296),
        ([0, 0, 0, 0, 1, 0, 0], None),
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], None),
        (67108863, None),
        (67108864, 67108864),
        (None, None),
        (True, None),
        ("12820938752", None),
    ],
    ids=["number", "8-bytes", "wrapped", "7-bytes", "9-bytes", "below-64-mib", "64-mib", "null", "bool", "text"],
)
def test_qwmemorysize_is_a_number_or_exactly_8_bytes_from_64_mib_on(value, number):
    assert registry_bytes(value) == number


@pytest.mark.parametrize("size", [4294967295, 4294967296, 4294967297])
def test_values_around_4_gib_are_all_valid_and_all_shown_as_4_0_gb(size):
    from modelroom.views_adapters import memory_words

    runner = gpu_runner(cim=cim(("NVIDIA L4", NVIDIA_PNP)), registry=registry((NVIDIA_KEY, size)))
    reading = measure_gpu("win32", runner)
    (adapter,) = reading.graphics
    assert adapter.source == "registry" and memory_words(adapter) == "4.0 GB"


def _memory(entries, listed_pnp=(NVIDIA_PNP,)):
    listed, _ = list_video_controllers(gpu_runner(cim=cim(*[("card", pnp) for pnp in listed_pnp])))
    return registry_memory(gpu_runner(registry=registry(*entries)), listed)


def test_an_entry_without_an_adapter_is_left_out():
    """A driver left behind in the registry (a card that was removed) is no memory of this machine."""
    found = _memory([(NVIDIA_KEY, 12820938752), ("PCI\\VEN_1002&DEV_73BF", 17163091968)])
    assert list(found) == ["VEN_10DE&DEV_2F38"]


def test_a_stale_second_entry_of_the_same_value_keeps_the_value():
    assert _memory([(NVIDIA_KEY, 12820938752), (NVIDIA_KEY, 12820938752)])["VEN_10DE&DEV_2F38"] == (11.94, None)


def test_two_same_cards_with_different_values_have_no_memory_and_say_why():
    found = _memory([(NVIDIA_KEY, 12820938752), (NVIDIA_KEY, 25769803776)], listed_pnp=(NVIDIA_PNP, NVIDIA_PNP))
    assert found["VEN_10DE&DEV_2F38"] == (None, "the registry holds 2 entries for 2 adapters")


def test_the_registry_memory_is_shown_and_never_counted():
    runner = gpu_runner(cim=cim(("AMD Radeon RX 7900 XTX", AMD_PNP)), registry=registry(("PCI\\VEN_1002&DEV_744C", 25753026560)))
    reading = measure_gpu("win32", runner)
    (adapter,) = reading.graphics
    assert (adapter.memory_gib, adapter.source, adapter.counted) == (23.98, "registry", False)
    assert reading.gpu_state == "none" and reading.vram_source == "none"


def test_the_laptop_recording_counts_the_nvidia_card_and_shows_the_intel_graphics():
    reading = measure_gpu("win32", gpu_runner(smi=SMI, cc="0, 12.0\n", **LAPTOP))
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source) == ("measured", 11.94, "nvidia-smi")
    nvidia, intel = reading.graphics
    assert (nvidia.source, nvidia.counted, nvidia.backend) == ("nvidia-smi", True, "cuda")
    assert (intel.name, intel.form, intel.counted, intel.memory_kind) == ("Intel(R) Graphics", "integrated", False, "shared")


def test_the_laptop_without_nvidia_smi_shows_the_registry_size_and_counts_nothing():
    """Handprobe 2 without llmfit: the NVIDIA card with 12 GB from the registry, `unconfirmed`, not counted."""
    reading = measure_gpu("win32", gpu_runner(**LAPTOP))
    intel, nvidia = reading.graphics
    assert (nvidia.memory_gib, nvidia.source, nvidia.usable, nvidia.counted) == (11.94, "registry", "unconfirmed", False)
    assert reading.gpu_state == "none" and "install the NVIDIA driver tools (nvidia-smi)" in reading.note


@pytest.mark.parametrize(
    "answer",
    [FileNotFoundError("powershell"), "not json", "[1, 2]", OSError("cannot start")],
    ids=["not-installed", "not-json", "not-objects", "oserror"],
)
def test_a_listing_that_fails_is_never_an_exception(answer):
    reading = measure_gpu("win32", gpu_runner(cim=answer))
    assert reading.gpu_state == "present_unmeasured" and reading.note.startswith("not recognized: ")


@pytest.mark.parametrize("answer", [FileNotFoundError("powershell"), "not json", "{}", "[]"], ids=["not-installed", "not-json", "empty-object", "empty"])
def test_a_registry_that_fails_only_leaves_the_memory_out(answer):
    reading = measure_gpu("win32", gpu_runner(cim=cim(("NVIDIA L4", NVIDIA_PNP)), registry=answer))
    (adapter,) = reading.graphics
    assert (adapter.memory_gib, adapter.source) == (None, "windows")


def test_the_registry_is_not_asked_when_nvidia_smi_covers_every_card():
    runner = gpu_runner(smi="0, NVIDIA L4, 23034\n", cim=cim(("NVIDIA L4", NVIDIA_PNP)))
    measure_gpu("win32", runner)
    assert WINDOWS_REGISTRY_ARGS not in runner.calls
