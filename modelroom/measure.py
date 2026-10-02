"""This machine's own hardware measurement: physical RAM, VRAM, the graphics cards, a limit, the
machine class.

modelroom measures the machine itself and asks `llmfit` only for a second opinion
(CONTRACTS.md, "Hardware measurement"). Every source here is a pure function over two injected
layers -- a command layer (`Runner`, the same `(args) -> CompletedProcess` shape
`modelroom/llmfit.py` defines and `SubprocessRunner` implements with a 10 s timeout) and a file
layer (`FileText`, `(path) -> text`). Nothing in this module reads `sys.platform`, spawns a
process by itself or raises for a source that is unavailable: every reading is a value with its
source, or no value with the reason, so one unreadable source never ends the command.
"""

from __future__ import annotations

import ctypes
import re
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Callable, Literal, Sequence

from .gpu import (  # noqa: F401 -- exported back: the tests and callers import these from here
    DISPLAY_ONLY_NOTE,
    DISPLAY_ONLY_VENDORS,
    INTEL_VENDOR,
    MEASURED_PLATFORMS,
    measure_gpu,
)
from .gpu_linux import LSPCI_ARGS, UNKNOWN_VENDOR, WSL_OSRELEASE_FILE, _lspci_vendors  # noqa: F401
from .gpu_llmfit import vram_crosscheck
from .gpu_macos import MACOS_ARM64_ARGS, MACOS_CHIP_ARGS, SYSTEM_PROFILER_ARGS  # noqa: F401
from .gpu_nvidia import NVIDIA_CC_ARGS, NVIDIA_SMI_ARGS, _nvidia_adapter  # noqa: F401
from .gpu_windows import WINDOWS_ADAPTER_ARGS, WINDOWS_REGISTRY_ARGS  # noqa: F401
from .llmfit import LlmfitReference, Runner, SubprocessRunner
from .measure_base import (  # noqa: F401
    GIB,
    KIB,
    MIB,
    TIMEOUT_SECONDS,
    FileText,
    FixtureFiles,
    GpuReading,
    ListDir,
    MemoryBytes,
    list_directory,
    no_listing,
    Reading,
    _gib,
    _gib_reading,
    _run,
    _whole_number,
    read_file_text,
)
from .profile import (
    PROFILE_SCHEMA_VERSION,
    CrossCheck,
    HardwareProfile,
    LlmfitCrosscheck,
    MachineClass,
    crosscheck,
    new_profile_id,
    os_fingerprint,
)

DISPLAY_NAME_MAX = 128
DISPLAY_NAME_FALLBACK = "this machine"

# Every command this module runs, as a fixed argument list -- no user-controlled string ever
# reaches a command line (AGENTS.md, "Security, definition of done").
WINDOWS_CHASSIS_ARGS = (
    "powershell",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "(Get-CimInstance Win32_SystemEnclosure).ChassisTypes",
)
WINDOWS_MACHINE_GUID_ARGS = ("reg", "query", "HKLM\\SOFTWARE\\Microsoft\\Cryptography", "/v", "MachineGuid")
MACOS_MEMSIZE_ARGS = ("sysctl", "-n", "hw.memsize")
MACOS_PLATFORM_UUID_ARGS = ("ioreg", "-rd1", "-c", "IOPlatformExpertDevice")

PROC_MEMINFO_FILE = "/proc/meminfo"
PROC_SELF_CGROUP_FILE = "/proc/self/cgroup"
LINUX_MACHINE_ID_FILE = "/etc/machine-id"
LINUX_CHASSIS_FILE = "/sys/class/dmi/id/chassis_type"
CGROUP_ROOT = "/sys/fs/cgroup"

# The SMBIOS chassis types (System Enclosure, type 3) of each machine class, a positive list per
# class; every other number -- 1 Other, 2 Unknown, anything a guest machine without a chassis reports -- is
# `unknown` (decided 2026-09-26). The class changes no number of the fit.
CHASSIS_CLASSES: dict[int, MachineClass] = {
    **dict.fromkeys((8, 9, 10, 11, 14, 30, 31, 32), "laptop"),
    **dict.fromkeys((3, 4, 5, 6, 7, 13, 15, 16, 24, 34, 35, 36), "workstation"),
    **dict.fromkeys((17, 23, 25, 28, 29), "server"),
}
# The chassis type is the first word of the answer, a whole number of ASCII digits and nothing else.
_CHASSIS_NUMBER_RE = re.compile(r"[0-9]+")

CPU_ONLY_NOTE = "GPU sources skipped: --cpu-only was given, so this machine is judged as a CPU machine"
CPU_ONLY_IDENTITY_REASON = "an entered profile carries no OS fingerprint"

_MACHINE_GUID_RE = re.compile(r"MachineGuid\s+REG_SZ\s+(\S+)")
_PLATFORM_UUID_RE = re.compile(r'"IOPlatformUUID"\s*=\s*"([^"]+)"')
_MEMTOTAL_RE = re.compile(r"^MemTotal:\s+(\d+)\s+kB", flags=re.MULTILINE)


class _MemoryStatusEx(ctypes.Structure):
    """`MEMORYSTATUSEX` as `GlobalMemoryStatusEx` fills it (Windows only; harmless elsewhere)."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def windows_physical_memory_bytes() -> int:
    """Installed physical memory in bytes, from `GlobalMemoryStatusEx` (`kernel32`).

    Windows only; anywhere else `ctypes.windll` does not exist and this raises `AttributeError`,
    which `measure_physical_ram` records as an unknown reading like any other failure.
    """
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        raise OSError("the Windows GlobalMemoryStatusEx call reported a failure")
    return int(status.ullTotalPhys)


# --- what one source returns ------------------------------------------------------------------


@dataclass(frozen=True)
class MachineClassReading:
    """The machine class with its source: the chassis type, the user's own word, or unknown and why."""

    machine_class: MachineClass
    source: Literal["chassis", "entered", "unknown"]
    reason: str | None = None


@dataclass(frozen=True)
class RamLimitReading:
    """A process or container memory limit -- a note only, never an input to the fit."""

    limit_gib: float | None
    scope: str


@dataclass(frozen=True)
class OsIdentity:
    """The raw OS identifier of this machine; only its salted digest is ever stored."""

    raw_id: str | None
    source: Literal["windows_machineguid", "linux_machine_id", "macos_platform_uuid", "none"]
    reason: str | None = None


@dataclass(frozen=True)
class MeasuredHardware:
    """One whole measurement of this machine, with the notes a user has to read."""

    ram: Reading
    gpu: GpuReading
    ram_limit: RamLimitReading
    identity: OsIdentity
    notes: tuple[str, ...]
    cpu_only: bool
    machine_class: MachineClassReading = MachineClassReading("unknown", "unknown")


_CPU_ONLY_GPU = GpuReading(gpu_state="none", vram_gib=0.0, vram_source="none", gpu_name=None, note=CPU_ONLY_NOTE)


@dataclass(frozen=True)
class Probes:
    """Everything one `hardware` run reads from outside itself, in one injectable value.

    The defaults are this machine: its platform name, a `SubprocessRunner` with the 10 s timeout,
    the real file layer, the Windows memory call, the host name and a fresh random `profile_id`.
    The clock is not here -- `now` is passed separately, like everywhere else in this project. A
    test constructs its own `Probes` instead of patching anything, the same dependency-injection
    shape `Transport`/`Runner` already use.
    """

    platform: str = sys.platform
    runner: Runner = field(default_factory=lambda: SubprocessRunner(timeout=TIMEOUT_SECONDS))
    read_text: FileText = read_file_text
    memory_bytes: MemoryBytes = windows_physical_memory_bytes
    hostname: Callable[[], str] = socket.gethostname
    new_id: Callable[[], str] = new_profile_id
    # The folder layer (sysfs, AP9-GPU): `None` is the real one with the real file layer, else the
    # injected file layer's own `list_dir` where it has one, else no folder at all.
    list_dir: ListDir | None = None


def folder_layer(read_text: FileText | None, list_dir: ListDir | None) -> ListDir:
    """The folder layer that belongs to a file layer: a test's files never mix with this machine's folders."""
    if list_dir is not None:
        return list_dir
    if read_text is None or read_text is read_file_text:
        return list_directory
    return getattr(read_text, "list_dir", no_listing)


# --- physical RAM ------------------------------------------------------------------------------


def measure_physical_ram(platform: str, runner: Runner, read_text: FileText, memory_bytes: MemoryBytes) -> Reading:
    """Installed physical RAM in GiB: Windows `GlobalMemoryStatusEx`, Linux `/proc/meminfo`,
    macOS `sysctl hw.memsize` (the unified memory of a Mac, AP9-GPU G5)."""
    if platform == "win32":
        return _windows_ram(memory_bytes)
    if platform == "linux":
        return _linux_ram(read_text)
    if platform == "darwin":
        return _macos_ram(runner)
    return Reading(None, "unknown", f"physical RAM is not measured on {platform}")


def _windows_ram(memory_bytes: MemoryBytes) -> Reading:
    try:
        total = memory_bytes()
    except (OSError, AttributeError, ValueError) as exc:
        return Reading(None, "unknown", f"the Windows GlobalMemoryStatusEx call failed ({exc})")
    return _gib_reading(total, "GlobalMemoryStatusEx")


def _linux_ram(read_text: FileText) -> Reading:
    try:
        text = read_text(PROC_MEMINFO_FILE)
    except OSError as exc:
        return Reading(None, "unknown", f"cannot read {PROC_MEMINFO_FILE} ({exc})")
    match = _MEMTOTAL_RE.search(text)
    if match is None:
        return Reading(None, "unknown", f"{PROC_MEMINFO_FILE} has no MemTotal line")
    total_kib = _whole_number(match.group(1))
    if total_kib is None:
        return Reading(None, "unknown", f"{PROC_MEMINFO_FILE} MemTotal is not a readable number of kB")
    return _gib_reading(total_kib * KIB, f"{PROC_MEMINFO_FILE} MemTotal")


def _macos_ram(runner: Runner) -> Reading:
    stdout, reason = _run(runner, MACOS_MEMSIZE_ARGS)
    if stdout is None:
        return Reading(None, "unknown", f"cannot read the memory size ({reason})")
    try:
        total = int(stdout.strip())
    except ValueError:
        return Reading(None, "unknown", f"sysctl hw.memsize did not answer with a byte count: {stdout.strip()!r}")
    return _gib_reading(total, "sysctl hw.memsize")


# --- the RAM limit, a note only -----------------------------------------------------------------


def measure_ram_limit(platform: str, read_text: FileText) -> RamLimitReading:
    """The effective cgroup v2 memory limit: the smallest `memory.max` from this process's own
    cgroup up to the root. Linux only, and never an input to the fit -- the fit computes with
    the physical RAM, because the daemon's own cgroup is not knowable from here."""
    if platform != "linux":
        return RamLimitReading(None, "none")
    try:
        own_path = _cgroup_v2_path(read_text(PROC_SELF_CGROUP_FILE))
    except OSError:
        return RamLimitReading(None, "none")
    if own_path is None:
        return RamLimitReading(None, "none")
    read = (_memory_max(read_text, folder) for folder in _cgroup_folders(own_path))
    # `> 0` as well as "not None": a `memory.max` of `0` is not a limit anybody could run under,
    # and `HardwareProfile.ram_limit_gib` is `> 0` by contract.
    limits = [limit for limit in read if limit is not None and limit > 0]
    limit_gib = _gib(min(limits)) if limits else None
    if limit_gib is None:
        return RamLimitReading(None, "none")
    return RamLimitReading(limit_gib, "cgroup")


def _cgroup_v2_path(text: str) -> str | None:
    """The `0::<path>` line's path (cgroup v2), or `None` on a v1-only machine."""
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:].strip() or "/"
    return None


def _cgroup_folders(own_path: str) -> list[str]:
    """This cgroup's folder and every parent up to the root, as absolute paths."""
    relative = PurePosixPath(own_path)
    parts = [part for part in relative.parts if part != "/"]
    return [str(PurePosixPath(CGROUP_ROOT, *parts[:depth])) for depth in range(len(parts), -1, -1)]


def _memory_max(read_text: FileText, folder: str) -> int | None:
    """One cgroup's `memory.max` in bytes; `None` for `max`, an unreadable file or a value that
    is not a number this process can convert."""
    try:
        value = read_text(f"{folder}/memory.max").strip()
    except OSError:
        return None
    return _whole_number(value)


# --- the OS identifier ----------------------------------------------------------------------------


def read_os_identity(platform: str, runner: Runner, read_text: FileText) -> OsIdentity:
    """This machine's raw OS identifier; `profile.os_fingerprint` is the only form ever stored."""
    if platform == "win32":
        return _windows_identity(runner)
    if platform == "linux":
        return _linux_identity(read_text)
    if platform == "darwin":
        return _macos_identity(runner)
    return OsIdentity(None, "none", f"the machine identifier is not read on {platform}")


def _windows_identity(runner: Runner) -> OsIdentity:
    stdout, reason = _run(runner, WINDOWS_MACHINE_GUID_ARGS)
    if stdout is None:
        return OsIdentity(None, "none", f"MachineGuid could not be read ({reason})")
    match = _MACHINE_GUID_RE.search(stdout)
    if match is None or not match.group(1).strip():
        return OsIdentity(None, "none", "the registry query returned no MachineGuid value")
    return OsIdentity(match.group(1).strip(), "windows_machineguid")


def _linux_identity(read_text: FileText) -> OsIdentity:
    try:
        raw = read_text(LINUX_MACHINE_ID_FILE).strip()
    except OSError as exc:
        return OsIdentity(None, "none", f"cannot read {LINUX_MACHINE_ID_FILE} ({exc})")
    if not raw:
        return OsIdentity(None, "none", f"{LINUX_MACHINE_ID_FILE} is empty")
    return OsIdentity(raw, "linux_machine_id")


def _macos_identity(runner: Runner) -> OsIdentity:
    stdout, reason = _run(runner, MACOS_PLATFORM_UUID_ARGS)
    if stdout is None:
        return OsIdentity(None, "none", f"the platform UUID could not be read ({reason})")
    match = _PLATFORM_UUID_RE.search(stdout)
    # A blank value counts as none: `os_fingerprint` refuses an empty identifier and would
    # otherwise raise out of the command over one line of whitespace.
    if match is None or not match.group(1).strip():
        return OsIdentity(None, "none", "ioreg returned no IOPlatformUUID value")
    return OsIdentity(match.group(1).strip(), "macos_platform_uuid")


# --- the machine class ----------------------------------------------------------------------------


def read_machine_class(platform: str, runner: Runner, read_text: FileText) -> MachineClassReading:
    """This machine's class from its chassis type: Windows `Win32_SystemEnclosure`, Linux DMI.

    No GPU source, so `--cpu-only` reads it too. A chassis that cannot be read, or a type no class
    names, is `unknown` with the reason; macOS and every other platform have no source here.
    """
    if platform == "win32":
        text, reason = _run(runner, WINDOWS_CHASSIS_ARGS)
    elif platform == "linux":
        text, reason = _linux_chassis(read_text)
    else:
        return _unknown_class(f"the machine class is not read on {platform}")
    if text is None:
        return _unknown_class(f"the machine class is unknown: the chassis type could not be read ({reason})")
    first = (text.split() or [""])[0]
    number = _whole_number(first) if _CHASSIS_NUMBER_RE.fullmatch(first) else None
    if number is None:
        return _unknown_class(f"the machine class is unknown: the chassis type is not a number: {text.strip()[:40]!r}")
    machine_class = CHASSIS_CLASSES.get(number)
    if machine_class is None:
        return _unknown_class(f"the machine class is unknown: chassis type {number} names no machine class")
    return MachineClassReading(machine_class, "chassis")


def _linux_chassis(read_text: FileText) -> tuple[str | None, str | None]:
    try:
        return read_text(LINUX_CHASSIS_FILE), None
    except OSError as exc:
        return None, f"cannot read {LINUX_CHASSIS_FILE} ({exc})"


def _unknown_class(reason: str) -> MachineClassReading:
    return MachineClassReading("unknown", "unknown", reason)


# --- one whole measurement -------------------------------------------------------------------------


def measure_hardware(
    platform: str,
    runner: Runner | None = None,
    read_text: FileText | None = None,
    memory_bytes: MemoryBytes | None = None,
    cpu_only: bool = False,
    machine_class: MachineClass | None = None,
    list_dir: ListDir | None = None,
    reference: LlmfitReference | None = None,
) -> MeasuredHardware:
    """Every source of this machine at once, plus the notes a user has to read.

    `machine_class` (`hardware --machine-class`) is the user's own word for the class: it is taken
    with the source `entered` and the chassis is not read. `reference` is llmfit's reading of this
    machine: the fallback where modelroom counts no card itself (AP9-GPU, mode B); without it llmfit
    adds nothing. `list_dir` is the folder layer (`folder_layer`).

    `cpu_only` (`hardware --cpu-only`) is the user's own statement that this machine runs on the
    CPU: no GPU source is asked at all (F21), and the profile becomes an `entered` one, which carries
    no OS fingerprint -- so the OS identifier is not read either. It is a statement about the
    GPU, not about the platform: on a platform this does not measure, the state stays
    `unsupported_platform` and the fit stays blocked.
    """
    active_runner = runner if runner is not None else SubprocessRunner(timeout=TIMEOUT_SECONDS)
    active_read = read_text if read_text is not None else read_file_text
    active_memory = memory_bytes if memory_bytes is not None else windows_physical_memory_bytes
    active_list = folder_layer(read_text, list_dir)

    ram = measure_physical_ram(platform, active_runner, active_read, active_memory)
    if cpu_only and platform in MEASURED_PLATFORMS:
        gpu = _CPU_ONLY_GPU
    else:
        gpu = measure_gpu(platform, active_runner, active_read, active_list, reference, ram.value)
    ram_limit = measure_ram_limit(platform, active_read)
    identity = (
        OsIdentity(None, "none", CPU_ONLY_IDENTITY_REASON)
        if cpu_only
        else read_os_identity(platform, active_runner, active_read)
    )
    if machine_class is not None:
        chassis = MachineClassReading(machine_class, "entered")
    else:
        chassis = read_machine_class(platform, active_runner, active_read)
    notes = _notes(ram, gpu, ram_limit, identity, cpu_only) + ((chassis.reason,) if chassis.reason else ())
    return MeasuredHardware(ram, gpu, ram_limit, identity, notes, cpu_only, chassis)


def _notes(
    ram: Reading, gpu: GpuReading, ram_limit: RamLimitReading, identity: OsIdentity, cpu_only: bool
) -> tuple[str, ...]:
    """The plain-language notes of one measurement, in reading order; facts only.

    The identifier's reason is one of them, because it has a consequence the user has to know: a
    profile without a fingerprint cannot be told apart from a clone of this machine. It is left
    out for `--cpu-only`, where the identifier is skipped by contract and the `--cpu-only` note
    already says so.
    """
    identity_note = None
    if identity.reason is not None and not cpu_only:
        identity_note = f"{identity.reason}; without it a clone of this machine cannot be told apart"
    notes = [note for note in (ram.reason, gpu.note, identity_note) if note]
    if ram_limit.limit_gib is not None and ram.value is not None and ram_limit.limit_gib < ram.value:
        notes.append(
            f"a cgroup limits this process to {ram_limit.limit_gib:.2f} GiB; the fit computes with "
            f"the physical RAM of {ram.value:.2f} GiB, so it can be too optimistic here"
        )
    return tuple(notes)


# --- the llmfit cross-check and the profile -------------------------------------------------------


def build_crosscheck(measured: MeasuredHardware, reference: LlmfitReference) -> LlmfitCrosscheck:
    """Compare like with like: physical RAM against llmfit's, VRAM against llmfit's.

    A quantity this machine could not measure is never compared -- there is nothing to hold
    llmfit's reading against, so the check is `absent` (or `error`, when llmfit itself failed).

    A `cpu_only` VRAM is not compared either, for the same reason in the other direction: the
    `0` is the user's own statement, not a reading, and llmfit's number is about hardware the
    user has just said not to use. Measured against a machine that does have a card, that
    comparison was a `deviation`, which blocked the very CPU fit `--cpu-only` exists to produce
    (probe: `hardware --cpu-only` on a laptop with one NVIDIA adapter, 2026-09-23) and would be
    shown as one since a deviation is no block (2026-09-27).

    Cards of different sizes are `not_comparable`: llmfit multiplies one card's size by the count,
    so its sum says nothing about them (decided 2026-09-27). The graphics memory follows the table
    of ADR 0011 (`gpu_llmfit.vram_crosscheck`): compared only where llmfit's entries of the counted
    backend are exactly the counted cards, never where llmfit itself gave the counted value.
    """
    return LlmfitCrosscheck(
        ram_physical=_one_check(measured.ram.value, reference.ram_gib, reference.status),
        vram=vram_crosscheck(measured.gpu, reference, measured.cpu_only),
    )


def _one_check(own_gib: float | None, llmfit_gib: float | None, status: str) -> CrossCheck:
    if status == "available" and own_gib is not None and llmfit_gib is not None:
        return crosscheck(own_gib, llmfit_gib)
    return CrossCheck(status="error" if status == "error" else "absent", own_gib=own_gib)


def build_profile(
    measured: MeasuredHardware,
    reference: LlmfitReference,
    profile_id: str,
    display_name: str,
    recorded_at: datetime,
    runtimes: Sequence[str] = (),
) -> HardwareProfile:
    """The current profile of one measurement (CONTRACTS.md, "Hardware profile v2").

    The raw OS identifier never reaches the file: only `os_fingerprint`'s salted digest does.
    A `cpu_only` measurement is an `entered` profile, which the contract keeps without a
    fingerprint.
    """
    identity = measured.identity
    return HardwareProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        display_name=_display_name(display_name),
        os_fingerprint=os_fingerprint(identity.raw_id) if identity.raw_id else "none",
        os_fingerprint_source=identity.source,
        origin="entered" if measured.cpu_only else "measured",
        recorded_at=recorded_at,
        ram_physical_gib=measured.ram.value,
        ram_physical_source=measured.ram.source,
        ram_limit_gib=measured.ram_limit.limit_gib,
        ram_limit_scope=measured.ram_limit.scope,
        vram_gib=measured.gpu.vram_gib,
        vram_source=measured.gpu.vram_source,
        gpu_state=measured.gpu.gpu_state,
        gpu_name=measured.gpu.gpu_name,
        gpus=list(measured.gpu.adapters),
        machine_class=measured.machine_class.machine_class,
        machine_class_source=measured.machine_class.source,
        llmfit_crosscheck=build_crosscheck(measured, reference),
        llmfit_version=reference.version,
        adapters=list(measured.gpu.graphics),
        runtimes=list(runtimes),
    )


def _display_name(candidate: str) -> str:
    """The contract's 1 to 128 characters: a blank name becomes a readable placeholder."""
    name = candidate.strip()
    return name[:DISPLAY_NAME_MAX] if name else DISPLAY_NAME_FALLBACK
