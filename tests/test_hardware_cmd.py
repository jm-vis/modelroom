"""End-to-end tests for `modelroom hardware`: measure this machine, write its profile v2.

Every run here goes through injected layers (`Probes`): a `FixtureRunner` for every command
(`nvidia-smi`, `llmfit`, the registry query), a `FixtureFiles` file layer, a fixed byte count
for the Windows memory call, a fixed host name and a fixed sequence of profile ids. No process
is spawned and no file outside `tmp_path` is touched -- in particular the pointer file, which
in production lives in the user's home folder, is redirected with `pointer_path`.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.binding import GuidedPointer, read_pointer, write_pointer
from modelroom.cli import hardware_with_config, main
from modelroom.config import Configuration
from modelroom.examples import EXAMPLES
from modelroom.llmfit import FixtureRunner
from modelroom.measure import (
    LINUX_MACHINE_ID_FILE,
    LSPCI_ARGS,
    MACOS_MEMSIZE_ARGS,
    NVIDIA_SMI_ARGS,
    PROC_MEMINFO_FILE,
    PROC_SELF_CGROUP_FILE,
    WINDOWS_ADAPTER_ARGS,
    WINDOWS_MACHINE_GUID_ARGS,
    FixtureFiles,
    Probes,
)
from modelroom.profile import HardwareProfile, fit_block_reason, os_fingerprint
from modelroom.state import atomic_write_json

FIXTURES = Path(__file__).parent / "fixtures"
RUN1 = datetime(2026, 9, 23, 8, 0, 0, tzinfo=timezone.utc)
RUN2 = datetime(2026, 9, 23, 9, 0, 0, tzinfo=timezone.utc)
MACHINE_GUID = "4f2c1a68-9b03-4d5e-8a77-0c1de2f34567"
LAPTOP_RAM_BYTES = 137_112_547_328  # 127.70 GiB
LAPTOP_FINGERPRINT = os_fingerprint(MACHINE_GUID)
ID_ONE = "1111111111111111"
ID_TWO = "2222222222222222"


def _completed(args: tuple[str, ...], stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr=stderr)


def _llmfit_system(**system) -> str:
    data = json.loads((FIXTURES / "llmfit_system_laptop.json").read_text(encoding="utf-8"))
    data["system"].update(system)
    return json.dumps(data)


def _runner(llmfit_system: str | None = None, llmfit_missing: bool = False, overrides: dict | None = None) -> FixtureRunner:
    """Every command a Windows/NVIDIA `hardware` run makes, answered from the fixtures."""
    guid_stdout = f"\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\n    MachineGuid    REG_SZ    {MACHINE_GUID}\n"
    responses = {
        NVIDIA_SMI_ARGS: _completed(NVIDIA_SMI_ARGS, stdout=(FIXTURES / "nvidia_smi_one_gpu.csv").read_text(encoding="utf-8")),
        WINDOWS_MACHINE_GUID_ARGS: _completed(WINDOWS_MACHINE_GUID_ARGS, stdout=guid_stdout),
        ("llmfit", "--version"): _completed(("llmfit", "--version"), stdout="llmfit 1.1.16\n"),
        ("llmfit", "system", "--json"): _completed(("llmfit", "system", "--json"), stdout=llmfit_system or _llmfit_system()),
    }
    if llmfit_missing:
        responses[("llmfit", "--version")] = FileNotFoundError("llmfit not found")
        del responses[("llmfit", "system", "--json")]
    responses.update(overrides or {})
    return FixtureRunner(responses)


def _probes(runner: FixtureRunner | None = None, host: str = "workstation", ids: list[str] | None = None) -> Probes:
    remaining = list(ids or [ID_ONE, ID_TWO])
    return Probes(
        platform="win32",
        runner=runner if runner is not None else _runner(),
        read_text=FixtureFiles({}),
        memory_bytes=lambda: LAPTOP_RAM_BYTES,
        hostname=lambda: host,
        new_id=lambda: remaining.pop(0),
    )


def _config(tmp_path: Path, machines: dict | None = None) -> Configuration:
    data = {
        "schema_version": 2,
        "families": [],
        "packagers": [],
        "publishers": [],
        "machines": machines if machines is not None else {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}},
        "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "models.md")},
    }
    return Configuration.from_dict(data)


def _profiles(config: Configuration) -> list[Path]:
    folder = config.paths.hardware_dir
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def _written(config: Configuration) -> HardwareProfile:
    files = _profiles(config)
    assert len(files) == 1, f"expected exactly one profile, found {[path.name for path in files]}"
    return HardwareProfile.model_validate(json.loads(files[0].read_text(encoding="utf-8")))


def _place_profile(config: Configuration, profile_id: str, fingerprint: str = LAPTOP_FINGERPRINT, **changes) -> Path:
    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    payload.update({"profile_id": profile_id, "os_fingerprint": fingerprint, **changes})
    path = config.paths.hardware_dir / f"{profile_id}.json"
    atomic_write_json(path, HardwareProfile.model_validate(payload).model_dump(mode="json"))
    return path


def _run(config: Configuration, tmp_path: Path, **kwargs) -> int:
    kwargs.setdefault("probes", _probes())
    kwargs.setdefault("now", RUN1)
    kwargs.setdefault("pointer_path", tmp_path / "home" / "guided.json")
    return hardware_with_config(config, **kwargs)


# --- the measured profile -----------------------------------------------------------------


def test_a_measured_run_writes_one_profile_v2_with_every_source(tmp_path: Path, capsys):
    config = _config(tmp_path)

    assert _run(config, tmp_path, machine="workstation") == 0

    profile = _written(config)
    assert profile.schema_version == 2 and profile.profile_id == ID_ONE
    assert profile.origin == "measured" and profile.recorded_at == RUN1
    assert profile.gpu_state == "measured" and profile.vram_source == "nvidia-smi"
    assert profile.vram_gib == pytest.approx(11.99, abs=0.01)
    assert profile.ram_physical_source == "os" and profile.ram_physical_gib == pytest.approx(127.7, abs=0.01)
    assert profile.os_fingerprint == LAPTOP_FINGERPRINT
    assert profile.llmfit_crosscheck.ram_physical.status == "confirmed"
    assert profile.llmfit_crosscheck.vram.status == "confirmed"
    assert profile.llmfit_version == "1.1.16"
    assert profile.display_name == "workstation"
    assert fit_block_reason(profile) is None
    assert not list(config.paths.hardware_dir.glob("*.tmp"))
    assert "11.99" in capsys.readouterr().out


def test_the_raw_machine_identifier_never_reaches_the_file(tmp_path: Path):
    config = _config(tmp_path)
    _run(config, tmp_path)
    assert MACHINE_GUID not in _profiles(config)[0].read_text(encoding="utf-8")


def test_the_run_binds_this_machine_in_the_home_pointer(tmp_path: Path):
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"

    _run(config, tmp_path, pointer_path=pointer_path)

    pointer = read_pointer(pointer_path)
    assert pointer.binding_for(config.paths.state) == ID_ONE
    assert pointer.current == str(config.paths.state)


def test_the_command_line_binds_on_the_folder_the_configuration_file_sits_in(tmp_path: Path):
    """The results folder is the configuration's own folder -- the key `export-profile` and
    `import-profile` read the binding under -- not the state folder inside it."""
    results = tmp_path / "results"
    results.mkdir()
    config_path = results / "modelroom.toml"
    config_path.write_text(
        'schema_version = 2\nfamilies = []\npackagers = []\npublishers = []\n'
        '[machines.workstation]\nreserve_ram_gib = 8.0\nreserve_vram_gib = 1.0\nwriter = true\n'
        '[paths]\nstate = "state"\nmarkdown = "models.md"\n',
        encoding="utf-8",
    )
    pointer_path = tmp_path / "home" / "guided.json"

    assert main(["hardware", "--config", str(config_path)], probes=_probes(), now=RUN1, pointer_path=pointer_path) == 0

    pointer = read_pointer(pointer_path)
    assert pointer.binding_for(results.resolve()) == ID_ONE
    assert pointer.binding_for(results.resolve() / "state") is None


def test_a_second_run_writes_the_same_profile_again_and_no_second_file(tmp_path: Path):
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    assert _run(config, tmp_path, pointer_path=pointer_path) == 0

    assert _run(config, tmp_path, pointer_path=pointer_path, now=RUN2, probes=_probes(ids=[ID_TWO])) == 0

    profile = _written(config)
    assert profile.profile_id == ID_ONE and profile.recorded_at == RUN2


def test_a_display_name_the_user_changed_survives_the_next_measurement(tmp_path: Path):
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    _run(config, tmp_path, pointer_path=pointer_path)
    path = _profiles(config)[0]
    renamed = json.loads(path.read_text(encoding="utf-8")) | {"display_name": "the build server"}
    atomic_write_json(path, renamed)

    _run(config, tmp_path, pointer_path=pointer_path, now=RUN2, probes=_probes(host="something-else", ids=[ID_TWO]))

    assert _written(config).display_name == "the build server"


def test_the_configured_profile_is_adopted_when_the_home_pointer_is_empty(tmp_path: Path):
    machines = {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True, "profile": ID_TWO}}
    config = _config(tmp_path, machines=machines)
    _place_profile(config, ID_TWO)

    assert _run(config, tmp_path, machine="workstation") == 0

    assert _written(config).profile_id == ID_TWO


def test_a_run_without_a_machine_name_measures_and_writes_a_new_profile(tmp_path: Path):
    config = _config(tmp_path)
    assert _run(config, tmp_path) == 0
    assert _written(config).profile_id == ID_ONE


def test_a_configured_machine_that_is_not_a_writer_is_measured_too(tmp_path: Path):
    """Unlike `fetch`, `hardware` never asks for `writer = true`: every machine is measured,
    including inference-only ones. (The schema-1 command had this test; it must not be lost.)"""
    machines = {"inference-server": {"reserve_ram_gib": 16.0, "reserve_vram_gib": 2.0, "writer": False}}
    config = _config(tmp_path, machines=machines)

    assert _run(config, tmp_path, machine="inference-server") == 0

    assert _written(config).profile_id == ID_ONE
    assert read_pointer(tmp_path / "home" / "guided.json").binding_for(config.paths.state) == ID_ONE


def test_a_state_folder_that_cannot_hold_the_profile_is_reported_not_a_crash(tmp_path: Path, capsys):
    """A regular file where the `hardware` folder belongs: the write fails, and that has to be a
    message and an exit code, not an `OSError` out of the command."""
    config = _config(tmp_path)
    config.paths.state.mkdir(parents=True)
    config.paths.hardware_dir.write_text("not a folder", encoding="utf-8")

    code = _run(config, tmp_path)

    assert code == 1
    assert "could not be written" in capsys.readouterr().err
    assert not (tmp_path / "home" / "guided.json").exists()


def test_an_unknown_machine_name_is_exit_2_and_writes_nothing(tmp_path: Path, capsys):
    config = _config(tmp_path)
    assert _run(config, tmp_path, machine="no-such-machine") == 2
    assert _profiles(config) == []
    assert "no-such-machine" in capsys.readouterr().err


# --- the flags ------------------------------------------------------------------------------


def test_cpu_only_writes_an_entered_cpu_profile_and_asks_no_gpu_source(tmp_path: Path):
    config = _config(tmp_path)
    # llmfit still sees the card this machine has; the entered CPU profile must stand anyway.
    runner = _runner(llmfit_system=_llmfit_system(total_ram_gb=127.46))

    assert _run(config, tmp_path, probes=_probes(runner), cpu_only=True) == 0

    profile = _written(config)
    assert profile.origin == "entered" and profile.gpu_state == "none"
    assert profile.vram_gib == 0.0 and profile.vram_source == "none"
    assert profile.os_fingerprint == "none" and profile.os_fingerprint_source == "none"
    assert NVIDIA_SMI_ARGS not in runner.calls and WINDOWS_MACHINE_GUID_ARGS not in runner.calls
    assert profile.llmfit_crosscheck.vram.status == "absent"
    assert fit_block_reason(profile) is None


def test_new_identity_writes_a_second_profile_and_rebinds_this_machine(tmp_path: Path):
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    _run(config, tmp_path, pointer_path=pointer_path)

    assert _run(config, tmp_path, pointer_path=pointer_path, probes=_probes(ids=[ID_TWO]), new_identity=True) == 0

    assert [path.name for path in _profiles(config)] == [f"{ID_ONE}.json", f"{ID_TWO}.json"]
    assert read_pointer(pointer_path).binding_for(config.paths.state) == ID_TWO


def test_a_bound_profile_that_is_gone_stops_the_run_and_names_new_identity(tmp_path: Path, capsys):
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    write_pointer(pointer_path, GuidedPointer(schema_version=1).with_binding(config.paths.state, ID_TWO))

    assert _run(config, tmp_path, pointer_path=pointer_path) == 2

    assert _profiles(config) == []
    assert "--new-identity" in capsys.readouterr().err


def test_a_bound_profile_of_another_machine_stops_the_run(tmp_path: Path, capsys):
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    _place_profile(config, ID_TWO, fingerprint="abcdefabcdefabcd")
    write_pointer(pointer_path, GuidedPointer(schema_version=1).with_binding(config.paths.state, ID_TWO))

    assert _run(config, tmp_path, pointer_path=pointer_path) == 2

    assert "--new-identity" in capsys.readouterr().err
    assert json.loads((config.paths.hardware_dir / f"{ID_TWO}.json").read_text(encoding="utf-8"))["display_name"] == "workstation"


# --- llmfit as a cross-check, never a requirement -------------------------------------------


def test_a_missing_llmfit_is_not_a_failure_and_leaves_the_cross_check_absent(tmp_path: Path, capsys):
    config = _config(tmp_path)

    assert _run(config, tmp_path, probes=_probes(_runner(llmfit_missing=True))) == 0

    profile = _written(config)
    assert profile.llmfit_version is None
    assert profile.llmfit_crosscheck.ram_physical.status == "absent"
    assert fit_block_reason(profile) is None
    assert "llmfit" in capsys.readouterr().out


def test_a_deviating_llmfit_reading_is_stored_and_blocks_the_fit(tmp_path: Path, capsys):
    config = _config(tmp_path)
    runner = _runner(llmfit_system=_llmfit_system(gpu_vram_gb=6.0))

    assert _run(config, tmp_path, probes=_probes(runner)) == 0

    profile = _written(config)
    assert profile.llmfit_crosscheck.vram.status == "deviation"
    assert fit_block_reason(profile) is not None
    assert "deviation" in capsys.readouterr().out


def test_a_failing_llmfit_is_recorded_as_an_error_not_as_a_deviation(tmp_path: Path):
    broken = {("llmfit", "--version"): _completed(("llmfit", "--version"), returncode=1, stderr="broken")}
    runner = _runner(llmfit_missing=True, overrides=broken)
    config = _config(tmp_path)

    assert _run(config, tmp_path, probes=_probes(runner)) == 0

    assert _written(config).llmfit_crosscheck.vram.status == "error"


# --- notes the user has to read --------------------------------------------------------------


def test_an_unmeasured_adapter_is_reported_as_a_note_and_leaves_the_fit_unknown(tmp_path: Path, capsys):
    config = _config(tmp_path)
    runner = _runner(
        overrides={
            NVIDIA_SMI_ARGS: FileNotFoundError("nvidia-smi not found"),
            WINDOWS_ADAPTER_ARGS: _completed(
                WINDOWS_ADAPTER_ARGS, stdout="PCI\\VEN_8086&DEV_7D51&SUBSYS_00000000&REV_08\\3&2411e6fe&0&10\n"
            ),
        }
    )

    assert _run(config, tmp_path, probes=_probes(runner)) == 0

    profile = _written(config)
    assert profile.gpu_state == "present_unmeasured"
    assert fit_block_reason(profile) is not None
    assert "Ollama uses the CPU on Intel graphics" in capsys.readouterr().out


# --- existing files in the results folder ------------------------------------------------------


def test_a_schema_1_profile_beside_the_new_one_is_left_untouched(tmp_path: Path):
    config = _config(tmp_path)
    legacy = config.paths.hardware_dir / "workstation.json"
    atomic_write_json(legacy, EXAMPLES["HardwareSnapshot"])
    before = legacy.read_text(encoding="utf-8")

    assert _run(config, tmp_path) == 0

    assert legacy.read_text(encoding="utf-8") == before
    assert (config.paths.hardware_dir / f"{ID_ONE}.json").is_file()


def test_a_profile_file_of_a_future_schema_is_exit_3_and_writes_nothing(tmp_path: Path, capsys):
    config = _config(tmp_path)
    broken = config.paths.hardware_dir / "future.json"
    atomic_write_json(broken, {**EXAMPLES["HardwareProfile"], "schema_version": 3})

    assert _run(config, tmp_path) == 3

    assert _profiles(config) == [broken]
    assert "schema_version" in capsys.readouterr().err


def test_an_unreadable_profile_file_is_exit_3_and_names_it(tmp_path: Path, capsys):
    config = _config(tmp_path)
    broken = config.paths.hardware_dir
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "truncated.json").write_text('{"schema_version": 2, "profile_id"', encoding="utf-8")

    assert _run(config, tmp_path) == 3

    assert "truncated.json" in capsys.readouterr().err


def test_a_measurement_that_does_not_validate_as_a_profile_is_exit_2_not_a_crash(tmp_path: Path, capsys):
    """Second line of defense: every source guards its own value, but a combination none of them
    anticipated must still end as an exit code. Probe: a naive `now`, which
    `HardwareProfile.recorded_at` refuses (the schema-1 command answered `2` for it too)."""
    config = _config(tmp_path)

    code = _run(config, tmp_path, now=datetime(2026, 9, 23, 8, 0, 0))

    assert code == 2
    assert _profiles(config) == []
    assert "recorded_at" in capsys.readouterr().err


def test_a_broken_pointer_file_is_exit_3_and_writes_nothing(tmp_path: Path, capsys):
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_text("{", encoding="utf-8")

    assert _run(config, tmp_path, pointer_path=pointer_path) == 3

    assert _profiles(config) == []
    assert str(pointer_path) in capsys.readouterr().err


# --- the pointer file: read late, written as a merge, failures reported ----------------------


def test_a_binding_written_while_this_run_started_is_adopted_instead_of_duplicated(tmp_path: Path):
    """The pointer is read inside the lock, as late as possible.

    Two first runs at the same time must not end with two profiles for one machine. The competing
    write is injected through `new_id`, which runs inside the lock just before the pointer is
    read -- a read taken any earlier (before the lock, as the first version did) would miss it and
    write a second profile.
    """
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    _place_profile(config, ID_TWO)

    def _competing_write() -> str:
        write_pointer(pointer_path, GuidedPointer(schema_version=1).with_binding(config.paths.state, ID_TWO))
        return ID_ONE

    probes = _probes()
    probes = Probes(platform=probes.platform, runner=probes.runner, read_text=probes.read_text,
                    memory_bytes=probes.memory_bytes, hostname=probes.hostname, new_id=_competing_write)

    assert _run(config, tmp_path, probes=probes, pointer_path=pointer_path) == 0

    assert _written(config).profile_id == ID_TWO
    assert not (config.paths.hardware_dir / f"{ID_ONE}.json").exists()


def test_a_binding_for_another_results_folder_survives_this_runs_write(tmp_path: Path):
    """The pointer file is one file for every results folder, and each folder has its own lock.

    The write is therefore a merge onto the newest content, not a write-back of the copy this run
    read. The competing entry is injected through `hostname`, which runs after this run resolved
    its own profile and before the pointer is written.
    """
    config = _config(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    other_folder = tmp_path / "other-results"

    def _competing_write() -> str:
        write_pointer(pointer_path, read_pointer(pointer_path).with_binding(other_folder, ID_TWO))
        return "workstation"

    probes = _probes()
    probes = Probes(platform=probes.platform, runner=probes.runner, read_text=probes.read_text,
                    memory_bytes=probes.memory_bytes, hostname=_competing_write, new_id=probes.new_id)

    assert _run(config, tmp_path, probes=probes, pointer_path=pointer_path) == 0

    pointer = read_pointer(pointer_path)
    assert pointer.binding_for(other_folder) == ID_TWO
    assert pointer.binding_for(config.paths.state) == ID_ONE


def test_a_pointer_file_that_cannot_be_written_is_exit_1_with_the_profile_kept(tmp_path: Path, capsys):
    """The profile is written first, the binding after it. A binding that cannot be recorded is
    not a lost measurement, but it is not a complete run either: exit `1`, and the message says
    that the next run may write a second profile."""
    config = _config(tmp_path)
    blocking_file = tmp_path / "home"
    blocking_file.write_text("not a folder", encoding="utf-8")

    code = _run(config, tmp_path, pointer_path=blocking_file / "guided.json")

    assert code == 1
    assert _written(config).profile_id == ID_ONE
    assert "second profile" in capsys.readouterr().err


# --- files in the folder that are not schema-2 profiles ---------------------------------------


def test_a_fresh_id_never_lands_on_a_file_that_is_already_there(tmp_path: Path):
    """"Never overwritten" has to hold for a schema-1 file too, although it carries no
    `profile_id`: every file name in the folder is a taken name, whatever its schema."""
    config = _config(tmp_path)
    legacy_name = "aaaaaaaaaaaaaaaa"  # a valid machine name and a valid profile_id shape
    legacy = config.paths.hardware_dir / f"{legacy_name}.json"
    atomic_write_json(legacy, {**EXAMPLES["HardwareSnapshot"], "machine": legacy_name})
    before = legacy.read_bytes()

    assert _run(config, tmp_path, probes=_probes(ids=[legacy_name, ID_ONE])) == 0

    assert legacy.read_bytes() == before
    assert (config.paths.hardware_dir / f"{ID_ONE}.json").is_file()


def test_a_profile_file_with_an_over_long_number_is_exit_3_not_a_crash(tmp_path: Path, capsys):
    """`json.loads` raises a plain `ValueError`, not a `JSONDecodeError`, for an integer above
    Python's int/str conversion limit -- it must still be an ordinary unreadable file."""
    config = _config(tmp_path)
    config.paths.hardware_dir.mkdir(parents=True, exist_ok=True)
    broken = config.paths.hardware_dir / "huge.json"
    broken.write_text('{"schema_version": ' + "1" * 5000 + "}", encoding="utf-8")

    assert _run(config, tmp_path) == 3

    assert "huge.json" in capsys.readouterr().err


def test_hardware_stops_at_a_lock_held_by_another_process(tmp_path: Path):
    config = _config(tmp_path)
    holder_script = tmp_path / "lock_holder.py"
    holder_script.write_text(
        "import sys, time\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n"
        "from modelroom.state import acquire_lock\n"
        "acquire_lock(Path(sys.argv[1]), 'hardware', datetime.now(timezone.utc))\n"
        "print('holding', flush=True)\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    holder = subprocess.Popen([sys.executable, str(holder_script), str(config.paths.lock_file)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "holding", "the holder process never acquired the lock"
        assert _run(config, tmp_path) == 1
        assert _profiles(config) == []
    finally:
        holder.kill()
        holder.wait(timeout=5)


# --- the command line itself -------------------------------------------------------------------


def _write_config_file(tmp_path: Path) -> Path:
    path = tmp_path / "modelroom.toml"
    path.write_text(
        "schema_version = 2\n"
        "[machines.workstation]\n"
        "reserve_ram_gib = 8.0\n"
        "reserve_vram_gib = 1.0\n"
        "writer = true\n"
        "[paths]\n"
        'state = "state"\n'
        'markdown = "models.md"\n',
        encoding="utf-8",
    )
    return path


def test_the_command_line_measures_and_writes_the_profile(tmp_path: Path):
    config_path = _write_config_file(tmp_path)

    code = main(
        ["hardware", "--config", str(config_path)],
        probes=_probes(),
        now=RUN1,
        pointer_path=tmp_path / "home" / "guided.json",
    )

    assert code == 0
    assert (tmp_path / "state" / "hardware" / f"{ID_ONE}.json").is_file()


def test_the_command_line_passes_cpu_only_and_new_identity_through(tmp_path: Path):
    config_path = _write_config_file(tmp_path)
    pointer_path = tmp_path / "home" / "guided.json"
    runner = _runner(llmfit_system=_llmfit_system(gpu_vram_gb=0.0))

    first = main(["hardware", "--config", str(config_path), "--cpu-only"], probes=_probes(runner), now=RUN1,
                 pointer_path=pointer_path)
    second = main(["hardware", "--config", str(config_path), "--cpu-only", "--new-identity"],
                  probes=_probes(_runner(llmfit_system=_llmfit_system(gpu_vram_gb=0.0)), ids=[ID_TWO]), now=RUN2,
                  pointer_path=pointer_path)

    assert (first, second) == (0, 0)
    assert sorted(path.name for path in (tmp_path / "state" / "hardware").glob("*.json")) == [
        f"{ID_ONE}.json",
        f"{ID_TWO}.json",
    ]


def test_a_missing_configuration_file_is_exit_2(tmp_path: Path):
    assert main(["hardware", "--config", str(tmp_path / "missing.toml")], probes=_probes()) == 2


def test_the_real_probes_are_used_when_none_are_injected(tmp_path: Path):
    """The default `Probes` is this machine: it must build without an argument and name a
    platform. Nothing is measured here -- that is what the fixture runs above do."""
    probes = Probes()
    assert probes.platform == sys.platform
    assert callable(probes.runner) and callable(probes.read_text)


def test_a_linux_cpu_server_writes_a_profile_the_fit_can_compute(tmp_path: Path):
    """The second machine class of stage 1: a virtual server with no card, measured through the
    file layer (`/proc/meminfo`, `/etc/machine-id`) and `lspci -nn`, with a cgroup limit as a
    note. A real run of this belongs on the server; this is the same code path end to end."""
    config = _config(tmp_path)
    runner = FixtureRunner(
        {
            NVIDIA_SMI_ARGS: FileNotFoundError("nvidia-smi not found"),
            LSPCI_ARGS: _completed(LSPCI_ARGS, stdout=(FIXTURES / "lspci_nn_cpu_server.txt").read_text(encoding="utf-8")),
            ("llmfit", "--version"): FileNotFoundError("llmfit not found"),
        }
    )
    files = FixtureFiles(
        {
            PROC_MEMINFO_FILE: (FIXTURES / "proc_meminfo_linux.txt").read_text(encoding="utf-8"),
            LINUX_MACHINE_ID_FILE: "7c9e6679a1b04f0e8c2d3b5a6f7e8d90\n",
            PROC_SELF_CGROUP_FILE: "0::/payload.scope\n",
            "/sys/fs/cgroup/payload.scope/memory.max": "8589934592\n",
        }
    )
    probes = Probes(platform="linux", runner=runner, read_text=files, memory_bytes=lambda: 0,
                    hostname=lambda: "cpu-server", new_id=lambda: ID_ONE)

    assert _run(config, tmp_path, probes=probes) == 0

    profile = _written(config)
    assert profile.gpu_state == "none" and profile.vram_gib == 0.0 and profile.vram_source == "none"
    assert profile.ram_physical_gib == pytest.approx(31.1, abs=0.01) and profile.ram_physical_source == "os"
    assert profile.ram_limit_gib == pytest.approx(8.0) and profile.ram_limit_scope == "cgroup"
    assert profile.os_fingerprint_source == "linux_machine_id"
    assert profile.origin == "measured" and profile.llmfit_version is None
    assert fit_block_reason(profile) is None


def test_a_macos_run_writes_an_unsupported_platform_profile(tmp_path: Path):
    config = _config(tmp_path)
    runner = FixtureRunner(
        {
            MACOS_MEMSIZE_ARGS: _completed(MACOS_MEMSIZE_ARGS, stdout="137438953472\n"),
            ("ioreg", "-rd1", "-c", "IOPlatformExpertDevice"): _completed(("ioreg",), stdout='"IOPlatformUUID" = "A-B-C"\n'),
            ("llmfit", "--version"): FileNotFoundError("llmfit not found"),
        }
    )
    probes = Probes(platform="darwin", runner=runner, read_text=FixtureFiles({}), memory_bytes=lambda: 0,
                    hostname=lambda: "laptop", new_id=lambda: ID_ONE)

    assert _run(config, tmp_path, probes=probes) == 0

    profile = _written(config)
    assert profile.gpu_state == "unsupported_platform" and profile.vram_gib is None
    assert profile.os_fingerprint_source == "macos_platform_uuid"
    assert fit_block_reason(profile) is not None
