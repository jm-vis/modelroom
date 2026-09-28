"""Tests for modelroom.room: free graphics memory and disk, read through fakes only (AP9-UX2 U8).

No test here reads this machine: every probe is a fake, and `room.real_room_probes` refuses to run
under pytest (Codex r2-5). Paths are compared in their POSIX spelling, so the tests read the same
on Windows and Linux; the Windows cases name their folder through `OLLAMA_MODELS`.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from modelroom.llmfit import FixtureRunner
from modelroom.room import (
    NVIDIA_SMI_FREE_ARGS,
    RoomProbes,
    disk_words,
    free_disk,
    free_vram_mib,
    graphics_words,
    ollama_models_candidates,
    real_room_probes,
)

GIB = 1024**3


def _key(path, system: str) -> str:
    return str(path) if system == "win32" else Path(path).as_posix()


def _room(existing: set[str], *, system: str = "linux", environ: dict | None = None, links: dict | None = None):
    """A room whose file system holds exactly `existing`; `links` maps a path to where it points."""
    measured: list[str] = []

    def disk_usage(path) -> int:
        measured.append(_key(path, system))
        return 7 * GIB

    room = RoomProbes(
        environ=environ or {},
        home=Path("/srv/ada"),
        system=system,
        exists=lambda path: _key(path, system) in existing,
        resolve=lambda path: Path((links or {}).get(_key(path, system), str(path))),
        disk_usage=disk_usage,
        mount_of=lambda path: "/data" if _key(path, system).startswith("/data") else "/",
        runner=FixtureRunner({}),
    )
    return room, measured


def test_ollama_models_set_in_the_environment_wins():
    environ = {"OLLAMA_MODELS": "/data/models"}
    assert ollama_models_candidates(environ, Path("/srv/ada"), "linux") == [Path("/data/models")]
    # Taken even where it does not exist yet, and measured at the nearest folder above it.
    room, measured = _room({"/data", "/srv/ada/.ollama/models"}, environ=environ)
    disk = free_disk(room)
    assert disk.known and disk.drive == "/data" and measured == ["/data"]


@pytest.mark.parametrize("value", ["~/models", "~"])
def test_a_tilde_in_ollama_models_is_the_home_that_was_passed_in(value):
    """Codex r1 m5: the path decision is pure -- `~` is the `home` argument, never this process's."""
    candidates = ollama_models_candidates({"OLLAMA_MODELS": value}, Path("/srv/ada"), "linux")
    assert [path.as_posix() for path in candidates] == [("/srv/ada/models" if value != "~" else "/srv/ada")]


def test_the_user_default_when_only_it_exists():
    room, measured = _room({"/srv/ada/.ollama/models", "/srv/ada", "/"})
    disk = free_disk(room)
    assert disk.free_bytes == 7 * GIB and measured == ["/srv/ada/.ollama/models"]


def test_the_service_default_when_only_it_exists():
    room, measured = _room({"/usr/share/ollama/.ollama/models"})
    free_disk(room)
    assert measured == ["/usr/share/ollama/.ollama/models"]


def test_both_defaults_existing_is_unclear():
    room, measured = _room({"/srv/ada/.ollama/models", "/usr/share/ollama/.ollama/models"})
    disk = free_disk(room)
    assert not disk.known and measured == []
    assert disk_words(disk) == "disk unknown (two places Ollama may use)"


def test_a_missing_folder_is_measured_at_its_parent():
    room, measured = _room({"/srv/ada", "/home", "/"})
    disk = free_disk(room)
    assert disk.known and measured == ["/srv/ada"]


def test_a_symlink_is_measured_where_it_points():
    links = {"/srv/ada/.ollama/models": "/data/ollama"}
    room, measured = _room({"/srv/ada/.ollama/models", "/data/ollama"}, links=links)
    disk = free_disk(room)
    assert measured == ["/data/ollama"] and disk.drive == "/data"


def test_a_windows_drive_is_named_by_its_letter():
    models = "D:\\ollama\\models"
    room, _measured = _room({models}, system="win32", environ={"OLLAMA_MODELS": models})
    disk = free_disk(room)
    assert disk.drive == "D:"
    assert disk_words(disk) == "disk 7.0 GB free on D: (Ollama models)"


def test_a_unc_path_is_named_network_share():
    unc = "\\\\fileserver\\share\\models"
    room, _measured = _room({unc}, system="win32", environ={"OLLAMA_MODELS": unc})
    disk = free_disk(room)
    assert disk.drive == "network share"
    assert "fileserver" not in disk_words(disk)


def test_a_linux_path_is_named_by_its_mount():
    room, _measured = _room({"/data/models"}, environ={"OLLAMA_MODELS": "/data/models"})
    assert free_disk(room).drive == "/data"


def test_an_access_error_is_unknown():
    room, _measured = _room({"/srv/ada/.ollama/models"})

    def refused(path):
        raise PermissionError("denied")

    disk = free_disk(replace(room, disk_usage=refused))
    assert not disk.known and disk_words(disk) == "disk unknown"


def _smi(stdout: str | Exception) -> RoomProbes:
    if isinstance(stdout, Exception):
        answer: subprocess.CompletedProcess | Exception = stdout
    else:
        answer = subprocess.CompletedProcess(list(NVIDIA_SMI_FREE_ARGS), 0, stdout=stdout, stderr="")
    room, _measured = _room(set())
    return replace(room, runner=FixtureRunner({NVIDIA_SMI_FREE_ARGS: answer}))


def test_free_vram_per_card_in_mib():
    room = _smi("0, NVIDIA L4, 21504\n1, NVIDIA L4, 20889\n")
    free = free_vram_mib(room, [(0, "NVIDIA L4"), (1, "NVIDIA L4")])
    assert free == [21504.0, 20889.0]
    assert graphics_words(free) == "graphics 21.0 + 20.4 GB free now"


def test_missing_nvidia_smi_is_unknown():
    room = _smi(FileNotFoundError("nvidia-smi"))
    assert free_vram_mib(room, [(0, "NVIDIA L4")]) is None
    assert graphics_words(None) == "graphics unknown"


def test_a_timeout_is_unknown():
    room = _smi(subprocess.TimeoutExpired(list(NVIDIA_SMI_FREE_ARGS), 10))
    assert free_vram_mib(room, [(0, "NVIDIA L4")]) is None


@pytest.mark.parametrize(
    "stdout, cards",
    [
        ("0, NVIDIA L4, 21504\n", [(0, "NVIDIA L4"), (1, "NVIDIA L4")]),  # one card fewer
        ("0, NVIDIA A100, 21504\n", [(0, "NVIDIA L4")]),  # another name
        ("0, NVIDIA L4, lots\n", [(0, "NVIDIA L4")]),  # a line that does not read
        ("0, NVIDIA L4, nan\n", [(0, "NVIDIA L4")]),  # not finite (Codex r1 m6)
        ("0, NVIDIA L4, inf\n", [(0, "NVIDIA L4")]),
        ("0, NVIDIA L4, 21504\n", []),  # no card in the profile (unified memory, none)
    ],
)
def test_cards_that_differ_from_the_profile_are_unknown(stdout, cards):
    assert free_vram_mib(_smi(stdout), cards) is None


def test_the_self_test_checks_the_free_line_for_its_form_only():
    """`scripts/selftest.py` runs on the real machine: it checks the form of the `free` line, not its numbers.

    Here and not in `tests/test_selftest.py`, which is over the size limit (AP9-UX2, size guard).
    """
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "selftest.py"
    if not (script.parent.parent / ".git").exists():
        pytest.skip("scripts/ is not shipped in the sdist; the self-test is tested from a clone")
    spec = importlib.util.spec_from_file_location("modelroom_selftest_free", script)
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    head = " #    Model                   Package                 Fit           Speed         Memory    Download"
    lines = [head, " shown     1 of 1 package", " install   #1  ollama pull x:y"]

    def named(free: list[str]) -> bool:
        return any("free line" in problem for problem in st.result_table_problems([*free, *lines]))

    assert named([])
    assert not named([" free      graphics 21.0 + 20.4 GB free now · disk 41.2 GB free on C: (Ollama models)"])
    assert not named([" free      graphics unknown - disk unknown (two places Ollama may use)"])
    assert named([" free      graphics none · disk none"])


def test_no_test_reaches_the_real_room_probes():
    """The guard behind `tests/conftest.py`: the real probes refuse to run under pytest."""
    with pytest.raises(RuntimeError, match="real room probes"):
        real_room_probes()
