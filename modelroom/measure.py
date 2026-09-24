"""This machine's own hardware measurement: physical RAM, VRAM, the graphics adapter, a limit.

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
import math
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Callable, Literal

from .llmfit import LlmfitReference, Runner, SubprocessRunner
from .profile import (
    PROFILE_SCHEMA_VERSION,
    CrossCheck,
    GpuState,
    HardwareProfile,
    LlmfitCrosscheck,
    crosscheck,
    new_profile_id,
    os_fingerprint,
)

GIB = 1024**3
MIB = 1024**2
KIB = 1024
TIMEOUT_SECONDS = 10.0
# The platforms whose GPU this measures at all; everywhere else is `unsupported_platform`.
MEASURED_PLATFORMS: frozenset[str] = frozenset({"win32", "linux"})
DISPLAY_NAME_MAX = 128
DISPLAY_NAME_FALLBACK = "this machine"

# Every command this module runs, as a fixed argument list -- no user-controlled string ever
# reaches a command line (AGENTS.md, "Security, definition of done").
NVIDIA_SMI_ARGS = ("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits")
LSPCI_ARGS = ("lspci", "-nn")
WINDOWS_ADAPTER_ARGS = (
    "powershell",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID }",
)
WINDOWS_MACHINE_GUID_ARGS = ("reg", "query", "HKLM\\SOFTWARE\\Microsoft\\Cryptography", "/v", "MachineGuid")
MACOS_MEMSIZE_ARGS = ("sysctl", "-n", "hw.memsize")
MACOS_PLATFORM_UUID_ARGS = ("ioreg", "-rd1", "-c", "IOPlatformExpertDevice")

PROC_MEMINFO_FILE = "/proc/meminfo"
PROC_SELF_CGROUP_FILE = "/proc/self/cgroup"
LINUX_MACHINE_ID_FILE = "/etc/machine-id"
CGROUP_ROOT = "/sys/fs/cgroup"

# PCI vendors whose class-03 devices are pure display or virtual adapters: they never run a
# model, so a machine that has only these is a CPU machine, not a machine with an unmeasured
# GPU. A positive list on purpose -- an unknown vendor is never silently declared harmless.
DISPLAY_ONLY_VENDORS = {
    "1234": "QEMU/Bochs",
    "1af4": "virtio",
    "15ad": "VMware",
    "1b36": "Red Hat",
    "80ee": "VirtualBox",
    "1414": "Microsoft Hyper-V",
    "5853": "Xen",
    "1d0f": "Amazon",
}
INTEL_VENDOR = "8086"
UNKNOWN_VENDOR = "unknown"
INTEL_NOTE = "Ollama uses the CPU on Intel graphics; run `hardware --cpu-only` for the CPU fit"
DISPLAY_ONLY_NOTE = "display adapter only"
CPU_ONLY_NOTE = "GPU sources skipped: --cpu-only was given, so this machine is judged as a CPU machine"
CPU_ONLY_IDENTITY_REASON = "an entered profile carries no OS fingerprint"

_LSPCI_CLASS_RE = re.compile(r"^\S+\s+.*?\[([0-9a-fA-F]{4})\]:")
_LSPCI_ID_RE = re.compile(r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]")
_PNP_VENDOR_RE = re.compile(r"VEN_([0-9a-fA-F]{4})")
_MACHINE_GUID_RE = re.compile(r"MachineGuid\s+REG_SZ\s+(\S+)")
_PLATFORM_UUID_RE = re.compile(r'"IOPlatformUUID"\s*=\s*"([^"]+)"')
_MEMTOTAL_RE = re.compile(r"^MemTotal:\s+(\d+)\s+kB", flags=re.MULTILINE)


# --- the two injected layers ------------------------------------------------------------------

FileText = Callable[[str], str]
MemoryBytes = Callable[[], int]


def read_file_text(path: str) -> str:
    """The real file layer: the file's text, or an `OSError` naming it."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


class FixtureFiles:
    """Test file layer: a path-to-text mapping; every other path is a `FileNotFoundError`.

    The same shape `FixtureRunner`/`FixtureTransport` have and for the same reason -- there is
    no mocking library in this project, a test that needs a particular `/proc` content
    constructs one of these and passes it in like any other value.
    """

    def __init__(self, files: dict[str, str]) -> None:
        self._files = dict(files)
        self.reads: list[str] = []

    def __call__(self, path: str) -> str:
        self.reads.append(path)
        if path not in self._files:
            raise FileNotFoundError(f"{path}: no such file")
        return self._files[path]


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
class Reading:
    """One measured quantity in GiB with its source, or no value and the reason why not."""

    value: float | None
    source: Literal["os", "unknown"]
    reason: str | None = None


@dataclass(frozen=True)
class GpuReading:
    """The GPU verdict: the state the fit gates on, the VRAM with its source, and a note."""

    gpu_state: GpuState
    vram_gib: float | None
    vram_source: Literal["nvidia-smi", "none", "unknown"]
    gpu_name: str | None = None
    note: str | None = None


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


# --- the command layer, with one place for every failure --------------------------------------


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


# --- physical RAM ------------------------------------------------------------------------------


def measure_physical_ram(platform: str, runner: Runner, read_text: FileText, memory_bytes: MemoryBytes) -> Reading:
    """Installed physical RAM in GiB: Windows `GlobalMemoryStatusEx`, Linux `/proc/meminfo`,
    macOS `sysctl hw.memsize` (display only -- macOS has no fit, see `measure_gpu`)."""
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


# --- VRAM and the graphics adapter --------------------------------------------------------------


def measure_gpu(platform: str, runner: Runner) -> GpuReading:
    """The GPU state and, for exactly one NVIDIA adapter, its measured VRAM.

    `nvidia-smi` first; when it is not there or cannot answer, the adapter rule decides between
    "no GPU at all" and "an adapter is present but unmeasured" -- a graphics adapter's own
    reported memory (`AdapterRAM`) is never read, it is not VRAM. macOS and everything else is
    `unsupported_platform`: no unified-memory statement is invented.
    """
    if platform not in MEASURED_PLATFORMS:
        return GpuReading("unsupported_platform", None, "unknown", None, f"the GPU is not measured on {platform} yet")
    measured = _nvidia_smi_gpu(runner)
    if measured is not None:
        return measured
    return _adapter_rule(platform, runner)


def _nvidia_smi_gpu(runner: Runner) -> GpuReading | None:
    """The `nvidia-smi` verdict, or `None` when it did not answer at all (adapter rule then)."""
    stdout, _ = _run(runner, NVIDIA_SMI_ARGS)
    if stdout is None:
        return None
    rows = [line for line in stdout.splitlines() if line.strip()]
    if not rows:
        return None
    if len(rows) > 1:
        note = f"nvidia-smi lists {len(rows)} adapters; fit v1 covers one"
        return GpuReading("multi_gpu_not_covered", None, "unknown", None, note)
    return _one_nvidia_adapter(rows[0])


def _one_nvidia_adapter(row: str) -> GpuReading:
    fields = [field.strip() for field in row.split(",")]
    index = _whole_number(fields[0]) if fields else None
    mib = _whole_number(fields[2]) if len(fields) == 3 else None
    if len(fields) != 3 or index is None or mib is None:
        return GpuReading("present_unmeasured", None, "unknown", None, f"nvidia-smi output could not be read: {row.strip()[:120]!r}")
    name = fields[1]
    if index != 0:
        note = f"nvidia-smi reports this adapter at index {index}, not index 0"
        return GpuReading("present_unmeasured", None, "unknown", name, note)
    vram_gib = _gib(mib * MIB)
    if vram_gib is None:
        note = f"nvidia-smi reports no usable memory for this adapter: {str(mib)[:40]} MiB"
        return GpuReading("present_unmeasured", None, "unknown", name, note)
    return GpuReading("measured", vram_gib, "nvidia-smi", name)


def _adapter_rule(platform: str, runner: Runner) -> GpuReading:
    """No nvidia-smi: decide from the PCI vendor ids of the class-03 (display) devices."""
    vendors, reason = _display_adapter_vendors(platform, runner)
    if vendors is None:
        note = f"the graphics adapters could not be listed, so a GPU cannot be ruled out ({reason})"
        return GpuReading("present_unmeasured", None, "unknown", None, note)
    if not vendors:
        return GpuReading("none", 0.0, "none")
    measurable = [vendor for vendor in vendors if vendor not in DISPLAY_ONLY_VENDORS]
    if not measurable:
        return GpuReading("none", 0.0, "none", None, DISPLAY_ONLY_NOTE)
    return GpuReading("present_unmeasured", None, "unknown", None, _unmeasured_adapter_note(measurable))


def _unmeasured_adapter_note(vendors: list[str]) -> str:
    """Name every adapter that could hold memory, not just the first one found.

    The Intel advice is only true for a machine whose *only* measurable adapter is Intel
    graphics: a laptop that also carries a discrete card would otherwise be told to compute a
    CPU fit although it has a GPU whose driver tools were simply not installed.
    """
    listed = ", ".join(dict.fromkeys(vendors))
    note = f"a graphics adapter is present (PCI vendor {listed}), but its memory was not measured"
    if set(vendors) == {INTEL_VENDOR}:
        return f"{note}; {INTEL_NOTE}"
    return f"{note}; run `hardware --cpu-only` for the CPU fit"


def _display_adapter_vendors(platform: str, runner: Runner) -> tuple[list[str] | None, str | None]:
    """The PCI vendor id of every display adapter, or `(None, reason)` when none could be listed."""
    if platform == "linux":
        stdout, reason = _run(runner, LSPCI_ARGS)
        return (None, reason) if stdout is None else (_lspci_vendors(stdout), None)
    stdout, reason = _run(runner, WINDOWS_ADAPTER_ARGS)
    return (None, reason) if stdout is None else (_pnp_vendors(stdout), None)


def _lspci_vendors(stdout: str) -> list[str]:
    """Every `lspci -nn` line of PCI class 03 (display controllers), by vendor id."""
    vendors = []
    for line in stdout.splitlines():
        device_class = _LSPCI_CLASS_RE.match(line)
        if device_class is None or not device_class.group(1).startswith("03"):
            continue
        ids = _LSPCI_ID_RE.findall(line)
        vendors.append(ids[-1][0].lower() if ids else UNKNOWN_VENDOR)
    return vendors


def _pnp_vendors(stdout: str) -> list[str]:
    """Every `Win32_VideoController.PNPDeviceID`, by vendor id; a line without one is unknown."""
    vendors = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        match = _PNP_VENDOR_RE.search(line)
        vendors.append(match.group(1).lower() if match else UNKNOWN_VENDOR)
    return vendors


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


# --- one whole measurement -------------------------------------------------------------------------


def measure_hardware(
    platform: str,
    runner: Runner | None = None,
    read_text: FileText | None = None,
    memory_bytes: MemoryBytes | None = None,
    cpu_only: bool = False,
) -> MeasuredHardware:
    """Every source of this machine at once, plus the notes a user has to read.

    `cpu_only` (`hardware --cpu-only`) is the user's own statement that this machine runs on the
    CPU: no GPU source is asked at all, and the profile becomes an `entered` one, which carries
    no OS fingerprint -- so the OS identifier is not read either. It is a statement about the
    GPU, not about the platform: on a platform this does not measure, the state stays
    `unsupported_platform` and the fit stays blocked, because the RAM reading there is display
    only (probe: `--cpu-only` on macOS produced a computing profile before this).
    """
    active_runner = runner if runner is not None else SubprocessRunner(timeout=TIMEOUT_SECONDS)
    active_read = read_text if read_text is not None else read_file_text
    active_memory = memory_bytes if memory_bytes is not None else windows_physical_memory_bytes

    ram = measure_physical_ram(platform, active_runner, active_read, active_memory)
    gpu = _CPU_ONLY_GPU if cpu_only and platform in MEASURED_PLATFORMS else measure_gpu(platform, active_runner)
    ram_limit = measure_ram_limit(platform, active_read)
    identity = (
        OsIdentity(None, "none", CPU_ONLY_IDENTITY_REASON)
        if cpu_only
        else read_os_identity(platform, active_runner, active_read)
    )
    return MeasuredHardware(ram, gpu, ram_limit, identity, _notes(ram, gpu, ram_limit, identity, cpu_only), cpu_only)


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
    (probe: `hardware --cpu-only` on a laptop with one NVIDIA adapter, 2026-09-23).
    """
    if measured.cpu_only:
        # Skipped on purpose, whatever llmfit did: `absent` says "never asked", `error` would say
        # the comparison was attempted and llmfit let it down.
        vram = CrossCheck(status="absent", own_gib=measured.gpu.vram_gib)
    else:
        vram = _one_check(measured.gpu.vram_gib, reference.vram_gib, reference.status)
    return LlmfitCrosscheck(
        ram_physical=_one_check(measured.ram.value, reference.ram_gib, reference.status),
        vram=vram,
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
) -> HardwareProfile:
    """The schema-2 profile of one measurement (CONTRACTS.md, "Hardware profile v2").

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
        llmfit_crosscheck=build_crosscheck(measured, reference),
        llmfit_version=reference.version,
    )


def _display_name(candidate: str) -> str:
    """The contract's 1 to 128 characters: a blank name becomes a readable placeholder."""
    name = candidate.strip()
    return name[:DISPLAY_NAME_MAX] if name else DISPLAY_NAME_FALLBACK
