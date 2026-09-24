"""Tests for modelroom.measure: this machine's own hardware measurement.

Every source is exercised through the injected command layer (`Runner`, the same shape
`modelroom/llmfit.py` defines) and the injected file layer (`FixtureFiles`), never through a
mocking library and never against the machine the suite happens to run on -- the two real
implementations (`windows_physical_memory_bytes`, `read_file_text`) have their own tests at the
bottom, guarded by the platform they belong to.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.llmfit import FixtureRunner, LlmfitReference
from modelroom.measure import (
    DISPLAY_ONLY_VENDORS,
    INTEL_VENDOR,
    LINUX_MACHINE_ID_FILE,
    LSPCI_ARGS,
    MACOS_MEMSIZE_ARGS,
    MACOS_PLATFORM_UUID_ARGS,
    NVIDIA_SMI_ARGS,
    PROC_MEMINFO_FILE,
    PROC_SELF_CGROUP_FILE,
    WINDOWS_ADAPTER_ARGS,
    WINDOWS_MACHINE_GUID_ARGS,
    FixtureFiles,
    build_crosscheck,
    build_profile,
    measure_gpu,
    measure_hardware,
    measure_physical_ram,
    measure_ram_limit,
    read_file_text,
    read_os_identity,
    windows_physical_memory_bytes,
)
from modelroom.profile import fit_block_reason, os_fingerprint

FIXTURES = Path(__file__).parent / "fixtures"
RECORDED_AT = datetime(2026, 9, 23, 8, 0, 0, tzinfo=timezone.utc)
MACHINE_GUID = "4f2c1a68-9b03-4d5e-8a77-0c1de2f34567"
PLATFORM_UUID = "0A1B2C3D-4E5F-6071-8293-A4B5C6D7E8F9"
LAPTOP_RAM_BYTES = 137_112_547_328  # 127.70 GiB, as GlobalMemoryStatusEx reports it
# Not spelled inline below: `version="x.y.z"` would trip the single-version-source guard
# (`tests/test_version_single_source.py`), which is about this package's own version.
LLMFIT_VERSION = "1.1.16"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _done(args: tuple[str, ...], stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr=stderr)


def _runner(**responses) -> FixtureRunner:
    """A `FixtureRunner` keyed by the argument lists `measure.py` uses, by short name."""
    known = {
        "nvidia_smi": NVIDIA_SMI_ARGS,
        "lspci": LSPCI_ARGS,
        "adapters": WINDOWS_ADAPTER_ARGS,
        "machine_guid": WINDOWS_MACHINE_GUID_ARGS,
        "memsize": MACOS_MEMSIZE_ARGS,
        "platform_uuid": MACOS_PLATFORM_UUID_ARGS,
    }
    return FixtureRunner({known[name]: response for name, response in responses.items()})


def _missing(name: str) -> FileNotFoundError:
    return FileNotFoundError(f"{name}: not found on PATH")


def _windows_adapters(*pnp_device_ids: str) -> subprocess.CompletedProcess:
    return _done(WINDOWS_ADAPTER_ARGS, stdout="".join(f"{value}\n" for value in pnp_device_ids))


def _reg_query(guid: str = MACHINE_GUID) -> subprocess.CompletedProcess:
    stdout = (
        "\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\n"
        f"    MachineGuid    REG_SZ    {guid}\n\n"
    )
    return _done(WINDOWS_MACHINE_GUID_ARGS, stdout=stdout)


def _linux_files(**extra: str) -> FixtureFiles:
    files = {
        PROC_MEMINFO_FILE: _fixture("proc_meminfo_linux.txt"),
        LINUX_MACHINE_ID_FILE: "7c9e6679a1b04f0e8c2d3b5a6f7e8d90\n",
    }
    files.update(extra)
    return FixtureFiles(files)


# --- physical RAM ---------------------------------------------------------------------------


def test_windows_physical_ram_comes_from_the_injected_byte_count():
    reading = measure_physical_ram("win32", _runner(), FixtureFiles({}), lambda: LAPTOP_RAM_BYTES)
    assert reading.value == pytest.approx(127.7, abs=0.01)
    assert reading.source == "os" and reading.reason is None


def test_windows_physical_ram_that_cannot_be_read_is_unknown_with_a_reason():
    def _failing() -> int:
        raise OSError("GlobalMemoryStatusEx failed")

    reading = measure_physical_ram("win32", _runner(), FixtureFiles({}), _failing)
    assert reading.value is None and reading.source == "unknown"
    assert "GlobalMemoryStatusEx" in reading.reason


def test_linux_physical_ram_comes_from_proc_meminfo_in_gib():
    reading = measure_physical_ram("linux", _runner(), _linux_files(), lambda: 0)
    # 32612348 kB / 1024 / 1024
    assert reading.value == pytest.approx(31.1, abs=0.01)
    assert reading.source == "os"


def test_linux_physical_ram_without_the_file_is_unknown_and_names_it():
    reading = measure_physical_ram("linux", _runner(), FixtureFiles({}), lambda: 0)
    assert reading.value is None and reading.source == "unknown"
    assert PROC_MEMINFO_FILE in reading.reason


def test_linux_physical_ram_without_a_memtotal_line_is_unknown():
    reading = measure_physical_ram("linux", _runner(), FixtureFiles({PROC_MEMINFO_FILE: "MemFree: 12 kB\n"}), lambda: 0)
    assert reading.value is None and reading.source == "unknown"
    assert "MemTotal" in reading.reason


def test_macos_physical_ram_comes_from_sysctl_for_display_only():
    runner = _runner(memsize=_done(MACOS_MEMSIZE_ARGS, stdout="137438953472\n"))
    reading = measure_physical_ram("darwin", runner, FixtureFiles({}), lambda: 0)
    assert reading.value == pytest.approx(128.0)
    assert reading.source == "os"


def test_macos_physical_ram_from_a_failing_sysctl_is_unknown():
    runner = _runner(memsize=_done(MACOS_MEMSIZE_ARGS, stdout="", returncode=1, stderr="unknown oid"))
    reading = measure_physical_ram("darwin", runner, FixtureFiles({}), lambda: 0)
    assert reading.value is None and reading.source == "unknown"
    assert "sysctl" in reading.reason


def test_an_unmeasured_platform_has_an_unknown_physical_ram():
    reading = measure_physical_ram("aix", _runner(), FixtureFiles({}), lambda: 0)
    assert reading.value is None and reading.source == "unknown"
    assert "aix" in reading.reason


@pytest.mark.parametrize("total", [0, -1, float("inf")])
def test_a_byte_count_that_is_not_a_real_positive_number_is_unknown(total):
    """`HardwareProfile.ram_physical_gib` is `> 0` and must be a real number.

    A `0` from a memory call that reported success, or an infinity, would otherwise reach the
    contract and raise a pydantic error out of the command instead of being an unknown reading.
    """
    reading = measure_physical_ram("win32", _runner(), FixtureFiles({}), lambda: total)
    assert reading.value is None and reading.source == "unknown"
    assert "not a usable number of bytes" in reading.reason


_OVER_LONG_DIGITS = "1" * 5000


def test_an_over_long_memtotal_is_unknown_not_a_crash():
    """`int("1" * 5000)` raises `ValueError` (Python's int/str conversion limit), and a source
    must never raise -- not even for output no real kernel would print."""
    files = FixtureFiles({PROC_MEMINFO_FILE: f"MemTotal:  {_OVER_LONG_DIGITS} kB\n"})
    reading = measure_physical_ram("linux", _runner(), files, lambda: 0)
    assert reading.value is None and reading.source == "unknown"


def test_an_over_long_vram_number_is_present_unmeasured_not_a_crash():
    runner = _runner(nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout=f"0, NVIDIA Nova, {_OVER_LONG_DIGITS}\n"))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "present_unmeasured" and gpu.vram_gib is None


def test_an_over_long_cgroup_limit_is_no_limit_not_a_crash():
    files = _linux_files(
        **{PROC_SELF_CGROUP_FILE: "0::/\n", "/sys/fs/cgroup/memory.max": _OVER_LONG_DIGITS}
    )
    limit = measure_ram_limit("linux", files)
    assert limit.limit_gib is None and limit.scope == "none"


def test_a_memtotal_of_zero_is_unknown():
    files = FixtureFiles({PROC_MEMINFO_FILE: "MemTotal:              0 kB\n"})
    reading = measure_physical_ram("linux", _runner(), files, lambda: 0)
    assert reading.value is None and reading.source == "unknown"


def test_a_memsize_of_zero_is_unknown():
    runner = _runner(memsize=_done(MACOS_MEMSIZE_ARGS, stdout="0\n"))
    reading = measure_physical_ram("darwin", runner, FixtureFiles({}), lambda: 0)
    assert reading.value is None and reading.source == "unknown"


def test_a_sysctl_that_never_answers_is_unknown_not_an_exception():
    runner = _runner(memsize=subprocess.TimeoutExpired(cmd=list(MACOS_MEMSIZE_ARGS), timeout=10.0))
    reading = measure_physical_ram("darwin", runner, FixtureFiles({}), lambda: 0)
    assert reading.value is None and reading.source == "unknown"
    assert "did not answer" in reading.reason


# --- VRAM through nvidia-smi ----------------------------------------------------------------


def test_one_nvidia_adapter_is_measured_vram_from_nvidia_smi():
    runner = _runner(nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout=_fixture("nvidia_smi_one_gpu.csv")))
    gpu = measure_gpu("win32", runner)
    assert gpu.gpu_state == "measured"
    assert gpu.vram_gib == pytest.approx(11.99, abs=0.01)  # 12282 MiB
    assert gpu.vram_source == "nvidia-smi"
    assert gpu.gpu_name == "NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU"
    assert gpu.note is None


def test_two_adapters_are_not_covered_and_carry_no_vram():
    runner = _runner(nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout=_fixture("nvidia_smi_two_gpus.csv")))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "multi_gpu_not_covered"
    assert gpu.vram_gib is None and gpu.vram_source == "unknown"
    assert "2" in gpu.note


def test_an_adapter_that_is_not_index_0_is_present_unmeasured():
    runner = _runner(nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout="1, NVIDIA Nova, 8192\n"))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "present_unmeasured"
    assert "index" in gpu.note


def test_an_adapter_reporting_zero_memory_is_present_unmeasured():
    runner = _runner(nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout="0, NVIDIA Nova, 0\n"))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "present_unmeasured"
    assert gpu.vram_gib is None and gpu.vram_source == "unknown"


def test_unparsable_nvidia_smi_output_is_present_unmeasured_never_a_crash():
    runner = _runner(nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout="not, a, number\n"))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "present_unmeasured"
    assert "nvidia-smi" in gpu.note


def test_a_failing_nvidia_smi_falls_through_to_the_adapter_rule():
    runner = _runner(
        nvidia_smi=_done(NVIDIA_SMI_ARGS, returncode=9, stderr="driver not loaded"),
        lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")),
    )
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "none" and gpu.vram_gib == 0.0 and gpu.vram_source == "none"


def test_an_nvidia_smi_that_never_answers_falls_through_to_the_adapter_rule():
    runner = _runner(
        nvidia_smi=subprocess.TimeoutExpired(cmd=list(NVIDIA_SMI_ARGS), timeout=10.0),
        lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")),
    )
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "none"


# --- the adapter rule without nvidia-smi ----------------------------------------------------


def test_no_display_class_device_at_all_is_no_gpu():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "none" and gpu.vram_gib == 0.0 and gpu.vram_source == "none"
    assert gpu.note is None


def test_a_virtual_display_adapter_is_no_gpu_with_a_display_adapter_only_note():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_cpu_server.txt")))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "none" and gpu.vram_gib == 0.0 and gpu.vram_source == "none"
    assert gpu.note == "display adapter only"


def test_intel_graphics_is_present_unmeasured_and_names_the_cpu_only_flag():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_intel_laptop.txt")))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "present_unmeasured"
    assert gpu.vram_gib is None and gpu.vram_source == "unknown"
    assert "Ollama uses the CPU on Intel graphics" in gpu.note
    assert "hardware --cpu-only" in gpu.note


def test_an_nvidia_adapter_without_nvidia_smi_is_present_unmeasured():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_nvidia_laptop.txt")))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "present_unmeasured"
    assert "10de" in gpu.note
    # The same machine also has Intel graphics, but a discrete card is present: the CPU advice
    # that belongs to an Intel-only machine must not be given here.
    assert "Ollama uses the CPU" not in gpu.note


def test_an_adapter_list_that_cannot_be_read_is_present_unmeasured_not_no_gpu():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_missing("lspci"))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "present_unmeasured"
    assert "lspci" in gpu.note


def test_the_display_only_vendor_list_is_exactly_the_eight_the_contract_names():
    """Written out here on purpose, not derived from the module: the parametrised test below
    would lose a case together with the entry it tests, so removing a vendor from the rule would
    stay green. These eight are the list in CONTRACTS.md, "Hardware measurement"."""
    assert set(DISPLAY_ONLY_VENDORS) == {"1234", "1af4", "15ad", "1b36", "80ee", "1414", "5853", "1d0f"}
    assert INTEL_VENDOR not in DISPLAY_ONLY_VENDORS


@pytest.mark.parametrize("vendor", sorted(DISPLAY_ONLY_VENDORS))
def test_every_listed_display_only_vendor_is_no_gpu_on_windows(vendor):
    adapters = _windows_adapters(f"PCI\\VEN_{vendor.upper()}&DEV_1111&SUBSYS_11111234&REV_02\\3&2411e6fe&0&10")
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), adapters=adapters)
    gpu = measure_gpu("win32", runner)
    assert gpu.gpu_state == "none" and gpu.vram_source == "none"
    assert gpu.note == "display adapter only"


def test_a_windows_adapter_from_another_vendor_is_present_unmeasured():
    adapters = _windows_adapters("PCI\\VEN_1002&DEV_164E&SUBSYS_00000000&REV_C1\\3&2411e6fe&0&41")
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), adapters=adapters)
    gpu = measure_gpu("win32", runner)
    assert gpu.gpu_state == "present_unmeasured" and "1002" in gpu.note


def test_a_windows_adapter_without_a_vendor_id_counts_as_an_unknown_vendor():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), adapters=_windows_adapters("ROOT\\BasicDisplay\\0000"))
    gpu = measure_gpu("win32", runner)
    assert gpu.gpu_state == "present_unmeasured" and "unknown" in gpu.note


def test_no_windows_display_adapter_at_all_is_no_gpu():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), adapters=_windows_adapters())
    gpu = measure_gpu("win32", runner)
    assert gpu.gpu_state == "none" and gpu.vram_gib == 0.0


def test_a_measurable_vendor_wins_over_a_display_only_one_in_the_same_machine():
    adapters = _windows_adapters(
        "PCI\\VEN_1234&DEV_1111&SUBSYS_11111234&REV_02\\3&2411e6fe&0&10",
        f"PCI\\VEN_{INTEL_VENDOR.upper()}&DEV_7D51&SUBSYS_00000000&REV_08\\3&2411e6fe&0&10",
    )
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), adapters=adapters)
    gpu = measure_gpu("win32", runner)
    assert gpu.gpu_state == "present_unmeasured"


def test_macos_is_an_unsupported_platform_without_a_unified_memory_claim():
    gpu = measure_gpu("darwin", _runner())
    assert gpu.gpu_state == "unsupported_platform"
    assert gpu.vram_gib is None and gpu.vram_source == "unknown"
    assert "unified" not in gpu.note.lower()


def test_an_unmeasured_platform_is_an_unsupported_platform():
    gpu = measure_gpu("aix", _runner())
    assert gpu.gpu_state == "unsupported_platform" and "aix" in gpu.note


# --- the RAM limit, a note only -------------------------------------------------------------


def test_a_cgroup_v2_limit_is_the_smallest_effective_value_up_to_the_root():
    files = _linux_files(
        **{
            PROC_SELF_CGROUP_FILE: "0::/user.slice/session.scope\n",
            "/sys/fs/cgroup/user.slice/session.scope/memory.max": "12884901888\n",
            "/sys/fs/cgroup/user.slice/memory.max": "8589934592\n",
            "/sys/fs/cgroup/memory.max": "max\n",
        }
    )
    limit = measure_ram_limit("linux", files)
    assert limit.limit_gib == pytest.approx(8.0)
    assert limit.scope == "cgroup"


def test_the_root_cgroup_counts_too_when_it_holds_the_smallest_limit():
    """A walk that stopped one level above the root would stay green on the test above, where the
    root is `max`. Here the root holds the smallest value of the three."""
    files = _linux_files(
        **{
            PROC_SELF_CGROUP_FILE: "0::/user.slice/session.scope\n",
            "/sys/fs/cgroup/user.slice/session.scope/memory.max": "12884901888\n",
            "/sys/fs/cgroup/user.slice/memory.max": "8589934592\n",
            "/sys/fs/cgroup/memory.max": "4294967296\n",
        }
    )
    limit = measure_ram_limit("linux", files)
    assert limit.limit_gib == pytest.approx(4.0) and limit.scope == "cgroup"


def test_a_cgroup_without_any_numeric_limit_is_no_limit():
    files = _linux_files(
        **{
            PROC_SELF_CGROUP_FILE: "0::/\n",
            "/sys/fs/cgroup/memory.max": "max\n",
        }
    )
    limit = measure_ram_limit("linux", files)
    assert limit.limit_gib is None and limit.scope == "none"


def test_a_cgroup_v1_only_machine_reports_no_limit():
    files = _linux_files(**{PROC_SELF_CGROUP_FILE: "7:memory:/\n1:name=systemd:/user.slice\n"})
    limit = measure_ram_limit("linux", files)
    assert limit.limit_gib is None and limit.scope == "none"


def test_a_machine_without_proc_self_cgroup_reports_no_limit():
    limit = measure_ram_limit("linux", _linux_files())
    assert limit.limit_gib is None and limit.scope == "none"


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_the_ram_limit_is_only_read_on_linux(platform):
    limit = measure_ram_limit(platform, FixtureFiles({}))
    assert limit.limit_gib is None and limit.scope == "none"


# --- the OS identifier ----------------------------------------------------------------------


def test_the_windows_machine_guid_is_read_from_the_registry():
    identity = read_os_identity("win32", _runner(machine_guid=_reg_query()), FixtureFiles({}))
    assert identity.raw_id == MACHINE_GUID
    assert identity.source == "windows_machineguid" and identity.reason is None


def test_a_registry_query_that_fails_leaves_the_machine_without_an_identifier():
    runner = _runner(machine_guid=_done(WINDOWS_MACHINE_GUID_ARGS, returncode=1, stderr="access denied"))
    identity = read_os_identity("win32", runner, FixtureFiles({}))
    assert identity.raw_id is None and identity.source == "none"
    assert "MachineGuid" in identity.reason


def test_a_registry_output_without_the_value_leaves_no_identifier():
    runner = _runner(machine_guid=_done(WINDOWS_MACHINE_GUID_ARGS, stdout="HKEY_LOCAL_MACHINE\\SOFTWARE\n"))
    identity = read_os_identity("win32", runner, FixtureFiles({}))
    assert identity.raw_id is None and identity.source == "none"


def test_the_linux_machine_id_is_read_from_etc():
    identity = read_os_identity("linux", _runner(), _linux_files())
    assert identity.raw_id == "7c9e6679a1b04f0e8c2d3b5a6f7e8d90"
    assert identity.source == "linux_machine_id"


def test_a_missing_linux_machine_id_leaves_no_identifier():
    identity = read_os_identity("linux", _runner(), FixtureFiles({}))
    assert identity.raw_id is None and identity.source == "none"
    assert LINUX_MACHINE_ID_FILE in identity.reason


def test_the_macos_platform_uuid_comes_from_ioreg():
    stdout = f'  +-o Root  <class IORegistryEntry>\n    "IOPlatformUUID" = "{PLATFORM_UUID}"\n'
    identity = read_os_identity("darwin", _runner(platform_uuid=_done(MACOS_PLATFORM_UUID_ARGS, stdout=stdout)), FixtureFiles({}))
    assert identity.raw_id == PLATFORM_UUID and identity.source == "macos_platform_uuid"


@pytest.mark.parametrize("value", [" ", ""])
def test_a_blank_platform_uuid_leaves_no_identifier_instead_of_hashing_it(value):
    """`os_fingerprint` refuses an empty identifier, so a blank one must never reach it."""
    stdout = f'    "IOPlatformUUID" = "{value}"\n'
    identity = read_os_identity("darwin", _runner(platform_uuid=_done(MACOS_PLATFORM_UUID_ARGS, stdout=stdout)), FixtureFiles({}))
    assert identity.raw_id is None and identity.source == "none"


def test_a_blank_machine_guid_leaves_no_identifier():
    runner = _runner(machine_guid=_done(WINDOWS_MACHINE_GUID_ARGS, stdout="    MachineGuid    REG_SZ    \n"))
    identity = read_os_identity("win32", runner, FixtureFiles({}))
    assert identity.raw_id is None and identity.source == "none"


def test_an_unmeasured_platform_has_no_identifier():
    identity = read_os_identity("aix", _runner(), FixtureFiles({}))
    assert identity.raw_id is None and identity.source == "none" and "aix" in identity.reason


# --- the whole measurement, per fixture machine ---------------------------------------------


def _laptop() -> tuple[FixtureRunner, FixtureFiles]:
    runner = _runner(
        nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout=_fixture("nvidia_smi_one_gpu.csv")),
        machine_guid=_reg_query(),
    )
    return runner, FixtureFiles({})


def test_a_windows_nvidia_laptop_measures_ram_and_vram_without_a_note():
    runner, files = _laptop()
    measured = measure_hardware("win32", runner, files, lambda: LAPTOP_RAM_BYTES)
    assert measured.ram.source == "os" and measured.gpu.gpu_state == "measured"
    assert measured.identity.source == "windows_machineguid"
    assert measured.notes == ()


def test_a_cpu_server_with_a_virtual_display_adapter_measures_as_a_cpu_machine():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_cpu_server.txt")))
    measured = measure_hardware("linux", runner, _linux_files(), lambda: 0)
    assert measured.gpu.gpu_state == "none" and measured.gpu.vram_gib == 0.0
    assert measured.ram.value == pytest.approx(31.1, abs=0.01)
    assert "display adapter only" in measured.notes


def test_a_smaller_cgroup_limit_is_a_note_and_says_the_fit_uses_the_physical_ram():
    files = _linux_files(
        **{
            PROC_SELF_CGROUP_FILE: "0::/payload.scope\n",
            "/sys/fs/cgroup/payload.scope/memory.max": "8589934592\n",
        }
    )
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")))
    measured = measure_hardware("linux", runner, files, lambda: 0)
    note = next(line for line in measured.notes if "cgroup" in line)
    assert "8.00 GiB" in note and "31.1" in note and "physical" in note


def test_a_cgroup_limit_above_the_physical_ram_is_no_note():
    files = _linux_files(
        **{
            PROC_SELF_CGROUP_FILE: "0::/payload.scope\n",
            "/sys/fs/cgroup/payload.scope/memory.max": str(200 * 1024**3) + "\n",
        }
    )
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")))
    measured = measure_hardware("linux", runner, files, lambda: 0)
    assert measured.ram_limit.limit_gib == pytest.approx(200.0)
    assert not any("cgroup" in line for line in measured.notes)


def test_an_intel_laptop_measures_present_unmeasured_and_keeps_the_note():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_intel_laptop.txt")))
    measured = measure_hardware("linux", runner, _linux_files(), lambda: 0)
    assert measured.gpu.gpu_state == "present_unmeasured"
    assert any("Intel graphics" in line for line in measured.notes)


def test_a_multi_gpu_machine_measures_as_not_covered():
    runner = _runner(nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout=_fixture("nvidia_smi_two_gpus.csv")))
    measured = measure_hardware("linux", runner, _linux_files(), lambda: 0)
    assert measured.gpu.gpu_state == "multi_gpu_not_covered"
    assert measured.notes != ()


def test_macos_measures_ram_for_display_and_leaves_the_gpu_unsupported():
    runner = _runner(
        memsize=_done(MACOS_MEMSIZE_ARGS, stdout="137438953472\n"),
        platform_uuid=_done(MACOS_PLATFORM_UUID_ARGS, stdout=f'"IOPlatformUUID" = "{PLATFORM_UUID}"\n'),
    )
    measured = measure_hardware("darwin", runner, FixtureFiles({}), lambda: 0)
    assert measured.ram.value == pytest.approx(128.0) and measured.gpu.gpu_state == "unsupported_platform"
    assert measured.identity.source == "macos_platform_uuid"


def test_cpu_only_does_not_override_the_platform_gate_on_macos():
    """`--cpu-only` is a statement about the GPU, not about the platform: on macOS the state
    stays `unsupported_platform`, so the fit stays blocked. Otherwise a `--cpu-only` run on a Mac
    would produce a computing profile from a RAM reading the contract calls display only."""
    runner = _runner(memsize=_done(MACOS_MEMSIZE_ARGS, stdout="137438953472\n"))
    measured = measure_hardware("darwin", runner, FixtureFiles({}), lambda: 0, cpu_only=True)
    assert measured.gpu.gpu_state == "unsupported_platform"
    profile = build_profile(measured, _absent(), "3f9a0c21d4e6b870", "mac", RECORDED_AT)
    assert fit_block_reason(profile) is not None


def test_an_unrepresentable_vram_number_is_present_unmeasured_not_an_overflow():
    """`int("9" * 400)` converts fine, but dividing it into GiB raises `OverflowError`."""
    runner = _runner(nvidia_smi=_done(NVIDIA_SMI_ARGS, stdout=f"0, NVIDIA Nova, {'9' * 400}\n"))
    gpu = measure_gpu("linux", runner)
    assert gpu.gpu_state == "present_unmeasured" and gpu.vram_gib is None


def test_an_unrepresentable_cgroup_limit_is_no_limit_not_an_overflow():
    files = _linux_files(**{PROC_SELF_CGROUP_FILE: "0::/\n", "/sys/fs/cgroup/memory.max": "9" * 400})
    limit = measure_ram_limit("linux", files)
    assert limit.limit_gib is None and limit.scope == "none"


def test_an_unusable_number_is_not_quoted_at_full_length_in_the_reason():
    """A reason line is read by a person; 400 digits of it are not a fact anybody needs."""
    files = FixtureFiles({PROC_MEMINFO_FILE: f"MemTotal:  {'9' * 400} kB\n"})
    reading = measure_physical_ram("linux", _runner(), files, lambda: 0)
    assert reading.value is None and len(reading.reason) < 200


def test_cpu_only_skips_every_gpu_source_and_the_os_identifier():
    runner = _runner()  # any call at all would raise KeyError
    measured = measure_hardware("linux", runner, _linux_files(), lambda: 0, cpu_only=True)
    assert measured.gpu.gpu_state == "none" and measured.gpu.vram_gib == 0.0 and measured.gpu.vram_source == "none"
    assert measured.identity.raw_id is None and measured.identity.source == "none"
    assert runner.calls == []
    assert any("--cpu-only" in line for line in measured.notes)


def test_an_unreadable_machine_identifier_is_a_note_of_its_own():
    """The contract prints the reason of every source that failed, and this one has a consequence
    the user has to know: without a fingerprint a clone of this machine cannot be told apart."""
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")))
    files = FixtureFiles({PROC_MEMINFO_FILE: _fixture("proc_meminfo_linux.txt")})  # no /etc/machine-id
    measured = measure_hardware("linux", runner, files, lambda: 0)
    assert measured.identity.source == "none"
    assert any(LINUX_MACHINE_ID_FILE in line for line in measured.notes)


def test_a_cpu_only_run_does_not_report_the_missing_identifier_as_a_failure():
    """`--cpu-only` leaves the identifier out by contract, so there is nothing to report: the
    single `--cpu-only` note already says why."""
    measured = measure_hardware("linux", _runner(), _linux_files(), lambda: 0, cpu_only=True)
    assert len(measured.notes) == 1 and "--cpu-only" in measured.notes[0]


def test_an_unknown_physical_ram_is_a_note_of_its_own():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")))
    measured = measure_hardware("linux", runner, FixtureFiles({}), lambda: 0)
    assert measured.ram.value is None
    assert any(PROC_MEMINFO_FILE in line for line in measured.notes)


# --- the llmfit cross-check and the profile -------------------------------------------------


def _available(ram_gib: float = 127.46, vram_gib: float | None = 11.94) -> LlmfitReference:
    return LlmfitReference(status="available", version=LLMFIT_VERSION, ram_gib=ram_gib, vram_gib=vram_gib, reason=None)


def _absent() -> LlmfitReference:
    return LlmfitReference(status="absent", version=None, ram_gib=None, vram_gib=None, reason="llmfit is not installed")


def _measured_laptop():
    runner, files = _laptop()
    return measure_hardware("win32", runner, files, lambda: LAPTOP_RAM_BYTES)


def test_readings_within_five_percent_of_llmfit_are_confirmed():
    check = build_crosscheck(_measured_laptop(), _available())
    assert check.ram_physical.status == "confirmed" and check.vram.status == "confirmed"
    assert check.vram.llmfit_gib == 11.94


def test_a_vram_reading_that_differs_by_more_than_five_percent_is_a_deviation():
    check = build_crosscheck(_measured_laptop(), _available(vram_gib=6.0))
    assert check.vram.status == "deviation"
    assert check.ram_physical.status == "confirmed"


def test_a_missing_llmfit_leaves_both_cross_checks_absent():
    reference = LlmfitReference(status="absent", version=None, ram_gib=None, vram_gib=None, reason="llmfit is not installed")
    check = build_crosscheck(_measured_laptop(), reference)
    assert (check.ram_physical.status, check.vram.status) == ("absent", "absent")
    assert check.ram_physical.own_gib == pytest.approx(127.7, abs=0.01)


def test_a_failing_llmfit_leaves_both_cross_checks_in_error():
    reference = LlmfitReference(status="error", version=LLMFIT_VERSION, ram_gib=None, vram_gib=None, reason="exit 1")
    check = build_crosscheck(_measured_laptop(), reference)
    assert (check.ram_physical.status, check.vram.status) == ("error", "error")
    assert check.vram.llmfit_gib is None


def test_an_unmeasured_vram_is_never_compared_with_llmfits_value():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_intel_laptop.txt")))
    measured = measure_hardware("linux", runner, _linux_files(), lambda: 0)
    check = build_crosscheck(measured, _available(vram_gib=2.0))
    assert check.vram.status == "absent" and check.vram.own_gib is None
    assert check.ram_physical.status == "deviation"  # 31.1 against llmfit's 127.46


def test_a_measured_profile_carries_the_fingerprint_digest_and_every_source():
    measured = _measured_laptop()
    profile = build_profile(measured, _available(), "3f9a0c21d4e6b870", "workstation", RECORDED_AT)
    assert profile.origin == "measured"
    assert profile.os_fingerprint == os_fingerprint(MACHINE_GUID)
    assert profile.os_fingerprint_source == "windows_machineguid"
    assert MACHINE_GUID not in profile.model_dump_json()
    assert profile.ram_physical_source == "os" and profile.vram_source == "nvidia-smi"
    assert profile.gpu_state == "measured" and profile.llmfit_version == LLMFIT_VERSION
    assert profile.ram_limit_gib is None and profile.ram_limit_scope == "none"
    assert profile.display_name == "workstation"


def test_a_cpu_only_profile_is_entered_and_has_no_fingerprint():
    runner = _runner()
    measured = measure_hardware("linux", runner, _linux_files(), lambda: 0, cpu_only=True)
    profile = build_profile(measured, _available(ram_gib=31.2, vram_gib=0.0), "3f9a0c21d4e6b870", "server", RECORDED_AT)
    assert profile.origin == "entered"
    assert profile.os_fingerprint == "none" and profile.os_fingerprint_source == "none"
    assert profile.gpu_state == "none" and profile.vram_gib == 0.0 and profile.vram_source == "none"
    assert profile.llmfit_crosscheck.vram.status == "absent"


def test_a_cpu_only_vram_stays_absent_even_when_llmfit_failed():
    """The check was skipped on purpose, so it is `absent`, not `error`: `error` would say that
    the comparison was attempted and llmfit let it down. The RAM check is still `error`."""
    measured = measure_hardware("linux", _runner(), _linux_files(), lambda: 0, cpu_only=True)
    reference = LlmfitReference(status="error", version=LLMFIT_VERSION, ram_gib=None, vram_gib=None, reason="exit 1")
    check = build_crosscheck(measured, reference)
    assert check.vram.status == "absent"
    assert check.ram_physical.status == "error"


def test_an_entered_cpu_only_vram_is_never_held_against_llmfits_gpu_reading():
    """The whole point of `--cpu-only` is a CPU fit; llmfit's VRAM must not block it.

    On a machine that does have a card, comparing the entered `0` with llmfit's reading was a
    `deviation`, and a deviation blocks the fit (`fit_block_reason`).
    """
    measured = measure_hardware("linux", _runner(), _linux_files(), lambda: 0, cpu_only=True)
    check = build_crosscheck(measured, _available(ram_gib=31.2, vram_gib=11.94))
    assert check.vram.status == "absent" and check.vram.llmfit_gib is None
    assert check.ram_physical.status == "confirmed"


def test_a_profile_without_a_readable_physical_ram_says_unknown():
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")))
    measured = measure_hardware("linux", runner, FixtureFiles({}), lambda: 0)
    profile = build_profile(measured, _available(), "3f9a0c21d4e6b870", "server", RECORDED_AT)
    assert profile.ram_physical_gib is None and profile.ram_physical_source == "unknown"
    assert profile.llmfit_crosscheck.ram_physical.status == "absent"


def test_a_cgroup_limit_is_stored_with_its_scope_but_never_as_the_ram():
    files = _linux_files(
        **{
            PROC_SELF_CGROUP_FILE: "0::/payload.scope\n",
            "/sys/fs/cgroup/payload.scope/memory.max": "8589934592\n",
        }
    )
    runner = _runner(nvidia_smi=_missing("nvidia-smi"), lspci=_done(LSPCI_ARGS, stdout=_fixture("lspci_nn_headless.txt")))
    measured = measure_hardware("linux", runner, files, lambda: 0)
    profile = build_profile(measured, _available(ram_gib=31.2), "3f9a0c21d4e6b870", "server", RECORDED_AT)
    assert profile.ram_limit_gib == pytest.approx(8.0) and profile.ram_limit_scope == "cgroup"
    assert profile.ram_physical_gib == pytest.approx(31.1, abs=0.01)


def test_a_display_name_longer_than_the_contract_allows_is_cut_to_fit():
    profile = build_profile(_measured_laptop(), _available(), "3f9a0c21d4e6b870", "n" * 300, RECORDED_AT)
    assert len(profile.display_name) == 128


def test_an_empty_display_name_falls_back_to_a_readable_placeholder():
    profile = build_profile(_measured_laptop(), _available(), "3f9a0c21d4e6b870", "   ", RECORDED_AT)
    assert profile.display_name == "this machine"


# --- the two real implementations, each on its own platform ---------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="GlobalMemoryStatusEx is a Windows call")
def test_the_real_windows_memory_call_returns_a_plausible_byte_count():
    total = windows_physical_memory_bytes()
    assert total > 1024**3  # more than 1 GiB on any machine that runs this suite


def test_the_real_file_layer_reads_a_file_and_reports_a_missing_one(tmp_path: Path):
    target = tmp_path / "meminfo"
    target.write_text("MemTotal: 1 kB\n", encoding="utf-8")
    assert read_file_text(str(target)) == "MemTotal: 1 kB\n"
    with pytest.raises(OSError):
        read_file_text(str(tmp_path / "missing"))
