"""Every graphics card `nvidia-smi` lists, and the machine class from the chassis (decided 2026-09-26).

One card is `measured` as before; two or more are `multi_gpu` with their sum. The chassis type
(Windows `Win32_SystemEnclosure`, Linux DMI) names the machine class; whatever cannot be read, or
names no class, is `unknown` with a note. Every source goes through the injected layers.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.cli import hardware_with_config, main
from modelroom.config import Configuration
from modelroom.llmfit import FixtureRunner, LlmfitGpu, LlmfitReference, cuda_vram_gib, read_llmfit_reference
from modelroom.measure import (
    LINUX_CHASSIS_FILE,
    LSPCI_ARGS,
    NVIDIA_CC_ARGS,
    NVIDIA_SMI_ARGS,
    WINDOWS_ADAPTER_ARGS,
    PROC_MEMINFO_FILE,
    WINDOWS_CHASSIS_ARGS,
    FixtureFiles,
    Probes,
    build_crosscheck,
    build_profile,
    measure_gpu,
    measure_hardware,
    read_machine_class,
)
from modelroom.profile import HardwareProfile, fit_block_reason

from fixture_support import LAPTOP_RAM_BYTES, windows_runner

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 26, 8, 0, 0, tzinfo=timezone.utc)
# The cards of the two rented machines of 2026-09-27 (`tests/fixtures/README.md`, recorded).
A100 = "NVIDIA A100-SXM4-80GB"
L4 = "NVIDIA L4"


def _done(args, stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr=stderr)


def _smi(stdout: str) -> FixtureRunner:
    """`nvidia-smi` alone; the sources AP9-GPU added are not installed (`tests/test_gpu_*.py` test them)."""
    missing = {args: FileNotFoundError(args[0]) for args in (NVIDIA_CC_ARGS, LSPCI_ARGS, WINDOWS_ADAPTER_ARGS)}
    return FixtureRunner({**missing, NVIDIA_SMI_ARGS: _done(NVIDIA_SMI_ARGS, stdout)})


def _chassis(response) -> FixtureRunner:
    return FixtureRunner({WINDOWS_CHASSIS_ARGS: response})


# --- the cards ------------------------------------------------------------------------------------


def test_two_cards_are_multi_gpu_with_their_sum_the_common_name_and_a_note():
    reading = measure_gpu("win32", _smi((FIXTURES / "nvidia_smi_two_gpus.csv").read_text(encoding="utf-8")))
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source, reading.gpu_name) == ("multi_gpu", 160.0, "nvidia-smi", A100)
    assert [(card.index, card.name, card.vram_gib) for card in reading.adapters] == [(0, A100, 80.0), (1, A100, 80.0)]
    assert reading.note == f"nvidia-smi lists 2 adapters: 2 x {A100}"


def test_the_recorded_two_l4_list_is_multi_gpu_with_its_sum():
    reading = measure_gpu("linux", _smi((FIXTURES / "nvidia_smi_two_l4.csv").read_text(encoding="utf-8")))
    assert (reading.gpu_state, reading.vram_gib, reading.vram_source, reading.gpu_name) == ("multi_gpu", 44.98, "nvidia-smi", L4)
    assert [(card.index, card.name, card.vram_gib) for card in reading.adapters] == [(0, L4, 22.49), (1, L4, 22.49)]
    assert reading.note == f"nvidia-smi lists 2 adapters: 2 x {L4}"


def test_three_mixed_cards_carry_no_common_name_and_name_each_card():
    reading = measure_gpu("linux", _smi("0, NVIDIA A6000, 49152\n1, NVIDIA A4000, 16384\n2, NVIDIA A4000, 16384\n"))
    assert (reading.gpu_state, reading.vram_gib, reading.gpu_name) == ("multi_gpu", 80.0, None)
    assert reading.note == "nvidia-smi lists 3 adapters: NVIDIA A6000 48.00 GiB, NVIDIA A4000 16.00 GiB, NVIDIA A4000 16.00 GiB"


def test_one_card_stays_measured_with_its_one_adapter():
    reading = measure_gpu("win32", _smi((FIXTURES / "nvidia_smi_one_gpu.csv").read_text(encoding="utf-8")))
    assert (reading.gpu_state, reading.vram_gib, reading.note) == ("measured", 11.99, None)
    assert [(card.index, card.vram_gib, card.vram_source) for card in reading.adapters] == [(0, 11.99, "nvidia-smi")]


def test_indexes_other_than_0_to_n_minus_1_are_present_unmeasured():
    reading = measure_gpu("win32", _smi(f"0, {A100}, 81920\n2, {A100}, 81920\n"))
    assert (reading.gpu_state, reading.vram_gib, reading.adapters) == ("present_unmeasured", None, ())
    assert reading.note == "nvidia-smi reports the adapters at indexes 0, 2, not 0 to 1"


@pytest.mark.parametrize("second", [f"1, {A100}", f"1, {A100}, 0", f"1, {A100}, lots"])
def test_one_unreadable_card_makes_the_whole_reading_present_unmeasured(second):
    reading = measure_gpu("win32", _smi(f"0, {A100}, 81920\n{second}\n"))
    assert (reading.gpu_state, reading.vram_gib, reading.adapters) == ("present_unmeasured", None, ())


def test_no_path_writes_multi_gpu_not_covered_any_more():
    for stdout in ("0, A, 8192\n1, A, 8192\n", "0, A, 8192\n1, B, 4096\n2, C, 2048\n3, D, 1024\n4, E, 512\n"):
        assert measure_gpu("win32", _smi(stdout)).gpu_state == "multi_gpu"


def test_a_two_card_measurement_writes_a_valid_profile_the_fit_computes_for():
    runner = windows_runner({NVIDIA_SMI_ARGS: _done(NVIDIA_SMI_ARGS, (FIXTURES / "nvidia_smi_two_gpus.csv").read_text(encoding="utf-8"))})
    measured = measure_hardware("win32", runner, FixtureFiles({}), lambda: LAPTOP_RAM_BYTES)
    reference = LlmfitReference("available", "1.1.16", 127.46, 80.0, None, cuda_vram_gib=160.0, gpus=(LlmfitGpu(A100, "CUDA", 80.0, 2),))
    profile = build_profile(measured, reference, "0123456789abcdef", "server", NOW)
    assert profile.gpu_state == "multi_gpu" and profile.vram_gib == 160.0 and len(profile.gpus) == 2
    assert profile.llmfit_crosscheck.vram.status == "confirmed"
    assert fit_block_reason(profile) is None


# --- the llmfit cross-check of several cards --------------------------------------------------------


def _system(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["system"]


def test_llmfit_counts_its_cuda_cards_times_their_count():
    assert cuda_vram_gib(_system("llmfit_system_two_gpus.json")) == 160.0


def test_llmfit_counts_the_recorded_two_l4_cards():
    assert cuda_vram_gib(_system("llmfit_system_two_l4.json")) == 44.98


def test_the_intel_graphics_of_the_laptop_recording_is_not_counted():
    assert cuda_vram_gib(_system("llmfit_system_laptop.json")) == 11.94


@pytest.mark.parametrize(
    "gpus",
    [None, "cards", [{"backend": "CUDA", "vram_gb": "40"}], [{"backend": "CUDA", "vram_gb": 40.0, "count": 0}], ["x"],
     [{"vram_gb": 40.0, "count": 1}], [{"backend": "CUDA", "vram_gb": 40.0, "count": 10**400}],
     [{"backend": "CUDA", "vram_gb": 40.0}], [{"backend": "CUDA", "vram_gb": 10**308, "count": 2}]],
    ids=["missing", "not-a-list", "size-text", "count-zero", "entry-text", "no-backend", "count-huge", "count-missing",
         "product-overflows"],
)
def test_a_card_list_that_does_not_read_gives_no_sum(gpus):
    system = {"total_ram_gb": 64.0} if gpus is None else {"total_ram_gb": 64.0, "gpus": gpus}
    assert cuda_vram_gib(system) is None


def test_read_llmfit_reference_carries_the_sum():
    runner = windows_runner({("llmfit", "system", "--json"): _done(("llmfit", "system", "--json"), (FIXTURES / "llmfit_system_two_gpus.json").read_text(encoding="utf-8"))})
    reference = read_llmfit_reference(runner, "1.1.16")
    assert (reference.status, reference.ram_gib, reference.vram_gib, reference.cuda_vram_gib) == ("available", 235.92, 80.0, 160.0)


@pytest.mark.parametrize(("llmfit_sum", "status"), [(160.0, "confirmed"), (80.0, "deviation"), (None, "not_comparable")])
def test_several_cards_are_checked_against_the_sum_of_llmfits_cards(llmfit_sum, status):
    """AP9-GPU G6 row 5/6: compared against llmfit's CUDA entries; without them `not_comparable` (was `absent`)."""
    runner = windows_runner({NVIDIA_SMI_ARGS: _done(NVIDIA_SMI_ARGS, (FIXTURES / "nvidia_smi_two_gpus.csv").read_text(encoding="utf-8"))})
    measured = measure_hardware("win32", runner, FixtureFiles({}), lambda: LAPTOP_RAM_BYTES)
    gpus = (LlmfitGpu(A100, "CUDA", llmfit_sum / 2, 2),) if llmfit_sum else ()
    reference = LlmfitReference("available", "1.1.16", 127.46, 80.0, None, cuda_vram_gib=llmfit_sum, gpus=gpus)
    assert build_crosscheck(measured, reference).vram.status == status


@pytest.mark.parametrize(("status", "llmfit_sum"), [("available", 96.0), ("available", None), ("absent", None), ("error", None)])
def test_cards_of_different_sizes_are_not_comparable_with_llmfits_sum(status, llmfit_sum):
    """llmfit multiplies one card's size by the count, so its sum says nothing about mixed cards
    (decided 2026-09-27): never `absent` for that case, and llmfit's number is kept where it said one."""
    runner = windows_runner({NVIDIA_SMI_ARGS: _done(NVIDIA_SMI_ARGS, "0, NVIDIA A6000, 49152\n1, NVIDIA A4000, 16384\n")})
    measured = measure_hardware("win32", runner, FixtureFiles({}), lambda: LAPTOP_RAM_BYTES)
    reference = LlmfitReference(status, "1.1.16", 127.46, 48.0, None, cuda_vram_gib=llmfit_sum)
    vram = build_crosscheck(measured, reference).vram
    assert (vram.status, vram.own_gib, vram.llmfit_gib) == ("not_comparable", 64.0, llmfit_sum)


def test_a_profile_of_mixed_cards_writes_not_comparable_and_computes():
    runner = windows_runner({NVIDIA_SMI_ARGS: _done(NVIDIA_SMI_ARGS, "0, NVIDIA A6000, 49152\n1, NVIDIA A4000, 16384\n")})
    measured = measure_hardware("win32", runner, FixtureFiles({}), lambda: LAPTOP_RAM_BYTES)
    reference = LlmfitReference("available", "1.1.16", 127.46, 48.0, None, cuda_vram_gib=96.0)
    profile = build_profile(measured, reference, "0123456789abcdef", "server", NOW)
    assert profile.llmfit_crosscheck.vram.status == "not_comparable"
    assert fit_block_reason(profile) is None


def test_the_recorded_two_l4_machine_is_confirmed_by_its_own_llmfit_answer():
    """Both answers from the same rented machine: `nvidia-smi` and `llmfit system --json` agree.

    `MemTotal` is the first line of that machine's `/proc/meminfo`, 98870364 kB (94.29 GiB).
    """
    smi = _done(NVIDIA_SMI_ARGS, (FIXTURES / "nvidia_smi_two_l4.csv").read_text(encoding="utf-8"))
    files = FixtureFiles({PROC_MEMINFO_FILE: "MemTotal:       98870364 kB\n", LINUX_CHASSIS_FILE: "1\n"})
    lspci = _done(LSPCI_ARGS, (FIXTURES / "lspci_nnD_two_l4_server.txt").read_text(encoding="utf-8"))
    runner = FixtureRunner({NVIDIA_SMI_ARGS: smi, NVIDIA_CC_ARGS: _done(NVIDIA_CC_ARGS, "0, 8.9\n1, 8.9\n"), LSPCI_ARGS: lspci})
    measured = measure_hardware("linux", runner, files, lambda: 0)
    llmfit = _done(("llmfit", "system", "--json"), (FIXTURES / "llmfit_system_two_l4.json").read_text(encoding="utf-8"))
    version = _done(("llmfit", "--version"), (FIXTURES / "llmfit_version.txt").read_text(encoding="utf-8"))
    reference = read_llmfit_reference(FixtureRunner({("llmfit", "--version"): version, ("llmfit", "system", "--json"): llmfit}), "1.1.16")
    check = build_crosscheck(measured, reference)
    assert (measured.gpu.vram_gib, measured.ram.value, reference.cuda_vram_gib) == (44.98, 94.29, 44.98)
    assert (check.vram.status, check.ram_physical.status) == ("confirmed", "confirmed")


# --- the chassis ------------------------------------------------------------------------------------


def test_cards_without_a_name_are_named_graphics_card_in_the_note():
    reading = measure_gpu("linux", _smi("0, , 40960\n1, , 40960\n"))
    assert reading.note == "nvidia-smi lists 2 adapters: 2 x graphics card"
    assert [card.name for card in reading.adapters] == ["", ""]


@pytest.mark.parametrize(("chassis", "machine_class"), [("10", "laptop"), ("3", "workstation"), ("23", "server"), ("9\n10", "laptop"), ("17\r\n", "server")])
def test_the_windows_chassis_names_the_class(chassis, machine_class):
    reading = read_machine_class("win32", _chassis(_done(WINDOWS_CHASSIS_ARGS, f"{chassis}\n")), FixtureFiles({}))
    assert (reading.machine_class, reading.source, reading.reason) == (machine_class, "chassis", None)


@pytest.mark.parametrize(
    ("response", "words"),
    [
        (_done(WINDOWS_CHASSIS_ARGS, "1\n"), "chassis type 1 names no machine class"),
        (_done(WINDOWS_CHASSIS_ARGS, "2\n"), "chassis type 2 names no machine class"),
        (_done(WINDOWS_CHASSIS_ARGS, ""), "not a number"),
        (_done(WINDOWS_CHASSIS_ARGS, "Notebook\n"), "not a number"),
        (_done(WINDOWS_CHASSIS_ARGS, "9" * 5000), "not a number"),
        (_done(WINDOWS_CHASSIS_ARGS, "", returncode=1, stderr="no access"), "could not be read"),
        (OSError("cannot start"), "could not be read"),
        (FileNotFoundError("no powershell"), "could not be read"),
    ],
    ids=["other", "unknown", "empty", "text", "huge", "exit-1", "oserror", "not-installed"],
)
def test_a_windows_chassis_that_says_nothing_is_unknown_with_a_note(response, words):
    reading = read_machine_class("win32", _chassis(response), FixtureFiles({}))
    assert (reading.machine_class, reading.source) == ("unknown", "unknown")
    assert reading.reason is not None and words in reading.reason and reading.reason.startswith("the machine class is unknown")


def test_the_linux_chassis_file_names_the_class():
    reading = read_machine_class("linux", FixtureRunner({}), FixtureFiles({LINUX_CHASSIS_FILE: "3\n"}))
    assert (reading.machine_class, reading.source) == ("workstation", "chassis")


def test_the_recorded_linux_chassis_type_1_names_no_class():
    """Recorded 2026-09-27: both rented GPU machines answer `1` (Other) in the DMI file."""
    reading = read_machine_class("linux", FixtureRunner({}), FixtureFiles({LINUX_CHASSIS_FILE: "1\n"}))
    assert (reading.machine_class, reading.source) == ("unknown", "unknown")
    assert reading.reason == "the machine class is unknown: chassis type 1 names no machine class"


class _BrokenFiles:
    def __call__(self, path: str) -> str:
        raise PermissionError(f"{path}: permission denied")


@pytest.mark.parametrize(
    ("files", "words"),
    [(FixtureFiles({}), "cannot read"), (_BrokenFiles(), "permission denied"), (FixtureFiles({LINUX_CHASSIS_FILE: "desktop\n"}), "not a number"),
     (FixtureFiles({LINUX_CHASSIS_FILE: "-10\n"}), "not a number"), (FixtureFiles({LINUX_CHASSIS_FILE: "3.5\n"}), "not a number")],
    ids=["missing", "oserror", "text", "negative", "decimal"],
)
def test_a_linux_chassis_that_says_nothing_is_unknown_with_a_note(files, words):
    reading = read_machine_class("linux", FixtureRunner({}), files)
    assert (reading.machine_class, reading.source) == ("unknown", "unknown")
    assert words in reading.reason


def test_macos_has_no_chassis_source():
    reading = read_machine_class("darwin", FixtureRunner({}), FixtureFiles({}))
    assert (reading.machine_class, reading.reason) == ("unknown", "the machine class is not read on darwin")


def test_a_fixture_runner_without_the_chassis_answer_is_the_test_authors_error():
    with pytest.raises(KeyError):
        read_machine_class("win32", FixtureRunner({}), FixtureFiles({}))


def test_an_unreadable_chassis_is_a_note_of_the_measurement():
    runner = windows_runner({WINDOWS_CHASSIS_ARGS: _done(WINDOWS_CHASSIS_ARGS, "")})
    measured = measure_hardware("win32", runner, FixtureFiles({}), lambda: LAPTOP_RAM_BYTES)
    assert measured.machine_class.source == "unknown"
    assert any(note.startswith("the machine class is unknown") for note in measured.notes)


# --- --cpu-only and --machine-class ------------------------------------------------------------------


def test_cpu_only_reads_the_chassis_and_writes_a_valid_profile_with_it():
    runner = windows_runner()
    measured = measure_hardware("win32", runner, FixtureFiles({}), lambda: LAPTOP_RAM_BYTES, cpu_only=True)
    profile = build_profile(measured, LlmfitReference("absent", None, None, None, "not installed"), "0123456789abcdef", "box", NOW)
    assert (profile.origin, profile.ram_physical_source, profile.gpu_state) == ("entered", "os", "none")
    assert (profile.machine_class, profile.machine_class_source, profile.gpus) == ("laptop", "chassis", [])
    assert WINDOWS_CHASSIS_ARGS in runner.calls and NVIDIA_SMI_ARGS not in runner.calls


def test_cpu_only_on_linux_still_leaves_exactly_one_note():
    files = FixtureFiles({PROC_MEMINFO_FILE: (FIXTURES / "proc_meminfo_linux.txt").read_text(encoding="utf-8"), LINUX_CHASSIS_FILE: "3\n"})
    measured = measure_hardware("linux", FixtureRunner({}), files, lambda: 0, cpu_only=True)
    assert len(measured.notes) == 1 and measured.machine_class.machine_class == "workstation"


def test_the_machine_class_given_is_entered_and_the_chassis_is_not_asked():
    runner = windows_runner()
    measured = measure_hardware("win32", runner, FixtureFiles({}), lambda: LAPTOP_RAM_BYTES, machine_class="server")
    assert (measured.machine_class.machine_class, measured.machine_class.source) == ("server", "entered")
    assert WINDOWS_CHASSIS_ARGS not in runner.calls


def _config(tmp_path: Path) -> Configuration:
    return Configuration.from_dict({
        "schema_version": 3, "families": [], "packagers": [], "publishers": [],
        "machines": {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}},
        "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "models.md")},
    })


def _probes(runner: FixtureRunner) -> Probes:
    return Probes(platform="win32", runner=runner, read_text=FixtureFiles({}), memory_bytes=lambda: LAPTOP_RAM_BYTES,
                  hostname=lambda: "workstation", new_id=lambda: "1111111111111111")


def _stored(config: Configuration) -> HardwareProfile:
    (path,) = sorted(config.paths.hardware_dir.glob("*.json"))
    return HardwareProfile.model_validate_json(path.read_text(encoding="utf-8"))


def test_hardware_with_config_reads_the_chassis_without_the_argument(tmp_path: Path):
    config = _config(tmp_path)
    assert hardware_with_config(config, "workstation", _probes(windows_runner()), NOW, tmp_path / "pointer.json") == 0
    assert (_stored(config).machine_class, _stored(config).machine_class_source) == ("laptop", "chassis")


def test_the_machine_class_flag_is_entered_and_printed(tmp_path: Path, capsys):
    config = _config(tmp_path)
    runner = windows_runner()
    code = hardware_with_config(config, "workstation", _probes(runner), NOW, tmp_path / "pointer.json", machine_class="server")
    assert code == 0 and WINDOWS_CHASSIS_ARGS not in runner.calls
    assert (_stored(config).machine_class, _stored(config).machine_class_source) == ("server", "entered")
    assert "class server (entered)" in capsys.readouterr().out


def test_the_summary_line_names_the_cards_and_the_class(tmp_path: Path, capsys):
    config = _config(tmp_path)
    two = windows_runner({NVIDIA_SMI_ARGS: _done(NVIDIA_SMI_ARGS, (FIXTURES / "nvidia_smi_two_gpus.csv").read_text(encoding="utf-8"))})
    assert hardware_with_config(config, "workstation", _probes(two), NOW, tmp_path / "pointer.json") == 0
    out = capsys.readouterr().out
    assert "vram 160.00 GiB (nvidia-smi, 2 cards), gpu multi_gpu, class laptop (chassis)" in out


def test_the_summary_line_of_one_card_only_adds_the_class(tmp_path: Path, capsys):
    config = _config(tmp_path)
    assert hardware_with_config(config, "workstation", _probes(windows_runner()), NOW, tmp_path / "pointer.json") == 0
    assert "vram 11.99 GiB (nvidia-smi), gpu measured, class laptop (chassis), llmfit" in capsys.readouterr().out


def test_the_cli_refuses_a_machine_class_it_does_not_know(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["hardware", "--config", str(tmp_path / "modelroom.toml"), "--machine-class", "tablet"])
    assert stop.value.code == 2
    assert "--machine-class" in capsys.readouterr().err
