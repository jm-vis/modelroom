"""AP9-LS-E: the LM Studio entry's own functions (`modelroom/lmstudio_install.py`) and its row of the table.

The command, its allowed names, the model folder, the "already there" check, the disk read for the
folder and the helper that hangs both on the disk of a guided run. Everything that reads the folder
uses a real temporary folder; nothing here starts a process or calls a server.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from modelroom import runtimes
from modelroom.contracts import Package
from modelroom.examples import EXAMPLES
from modelroom.hf_cache import CacheProbes
from modelroom.quantization import QUANT_ORDER, package_identity_key
from modelroom.room import GIB, Disk, RoomProbes

from fixture_room import fixed_room


def _package(repo: str = "unsloth/Qwen3.5-4B-GGUF", file: str = "Qwen3.5-4B-Q4_K_M.gguf", quant: str | None = "Q4_K_M",
             size: int = 2_783_446_304, **changes) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload.update(repo=repo, quantization=quant, format="gguf")
    payload["files"] = [{"name": file, "role": "weights", "size_bytes": size, "digest": None}]
    payload.update(changes)
    return Package.model_validate(payload)


def _split(**changes) -> Package:
    package = _package(**changes)
    files = [
        {"name": f"Qwen3.5-4B-Q4_K_M-0000{n}-of-00002.gguf", "role": "weights_shard", "size_bytes": 1000, "digest": None}
        for n in (1, 2)
    ]
    return package.model_copy(update={"files": [package.files[0].__class__.model_validate(file) for file in files]})


# --- the row of the table -----------------------------------------------------------------------------


def test_runtime_key_names_lmstudio_and_every_key_has_an_entry():
    from typing import get_args

    from modelroom.profile import RuntimeKey

    assert get_args(RuntimeKey) == ("ollama", "vllm", "lmstudio")
    assert [runtime.key for runtime in runtimes.RUNTIMES] == list(get_args(RuntimeKey))
    assert [runtime.name for runtime in runtimes.RUNTIMES][-1] == "LM Studio"


def test_the_entry_has_every_field_of_the_brief():
    entry = runtimes.runtime_of("lmstudio")
    assert entry.key == "lmstudio" and entry.name == "LM Studio" and entry.hello_field == "lmstudio"
    assert entry.server.origin == "http://127.0.0.1:1234"
    assert entry.words.package_prefix == "" and entry.words.key_text is None
    assert entry.words.question == "Which package should be installed?"
    assert entry.fit is runtimes.OLLAMA.fit


def test_lmstudio_accepts_hf_gguf_only():
    entry = runtimes.runtime_of("lmstudio")
    ollama_package = _package(source="ollama", repo=None, revision=None, ollama_name="nova:7b", manifest_digest="sha256:" + "a" * 64)
    tensor = _package(format="tensor")
    gguf = _package()
    assert entry.accepts(gguf) is True
    assert entry.accepts(tensor) is False and entry.accepts(ollama_package) is False
    assert entry.shown_only_with(gguf) is True
    assert entry.shown_only_with(tensor) is False and entry.shown_only_with(ollama_package) is False


def test_an_ollama_library_package_gets_no_lms_row():
    ollama_package = _package(source="ollama", repo=None, revision=None, ollama_name="nova:7b", manifest_digest="sha256:" + "a" * 64)
    profile = _profile(["ollama", "lmstudio"])
    keys = [runtime.key for runtime in runtimes.accepting_on_machine(ollama_package, profile)]
    assert keys == ["ollama"]


def _profile(keys):
    from modelroom.profile import HardwareProfile

    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    payload["runtimes"] = keys
    return HardwareProfile.model_validate(payload)


def test_a_profile_with_lmstudio_in_runtimes_reads_and_writes():
    profile = _profile(["ollama", "lmstudio"])
    assert profile.runtimes == ["ollama", "lmstudio"]
    assert profile.model_dump(mode="json")["runtimes"] == ["ollama", "lmstudio"]


def test_fit_runtime_lmstudio_serializes_and_ollama_fit_stays_byte_equal():
    from modelroom.contracts import Fit

    payload = copy.deepcopy(EXAMPLES["Fit"])
    ollama = Fit.model_validate(payload).model_dump(mode="json")
    assert "runtime" not in ollama
    payload.update(runtime="lmstudio")
    assert Fit.model_validate(payload).model_dump(mode="json")["runtime"] == "lmstudio"


def test_an_answer_with_an_unknown_runtime_value_is_exit_2():
    from modelroom.dialog import AnswerInvalidError

    from fixture_rows import document, package
    from test_pull_choice import _Recording, _run

    from modelroom.guided_install import PullContext, pull_step
    from modelroom.pull_choice import ambiguous_names

    q4 = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
    doc = document([q4])
    asker = _Recording({"pull_package": "mlx unsloth/Nova-7B-GGUF@Q4_K_M", "pull": False})
    run, _lines = _run(Path("."), asker)
    with pytest.raises(AnswerInvalidError):
        pull_step(run, PullContext(doc, "workstation", [q4], ambiguous_names([q4])), True)


# --- the command ----------------------------------------------------------------------------------------


def test_the_command_is_lms_get_gguf_owner_repo_at_quant():
    from modelroom import lmstudio_install as lms

    package = _package()
    name = lms.install_name(package)
    assert name == "unsloth/Qwen3.5-4B-GGUF@Q4_K_M"
    assert runtimes.runtime_of("lmstudio").install_name(package) == name
    assert runtimes.runtime_of("lmstudio").command(name) == "lms get --gguf unsloth/Qwen3.5-4B-GGUF@Q4_K_M"


@pytest.mark.parametrize(
    "repo, quant",
    [
        ("a/b;rm", "Q4_K_M"),
        ("a/b$(x)", "Q4_K_M"),
        ("a/b`x`", "Q4_K_M"),
        ("a/b\u00e9", "Q4_K_M"),  # a non-ASCII letter
        ("unsloth/Qwen3.5-4B-GGUF", "unknown"),
        ("unsloth/Qwen3.5-4B-GGUF", None),
    ],
)
def test_an_unsafe_repo_or_quant_has_no_command(repo, quant):
    from modelroom import lmstudio_install as lms

    # `model_copy` skips the validation of `Package`: the stand-in is what a pull row would meet if a
    # Hub answer got past it.
    package = _package().model_copy(update={"repo": repo, "quantization": quant})
    assert lms.install_name(package) is None
    assert lms.lms_reason(package) == "no safe command"


def test_the_command_has_only_safe_characters_for_every_lms_quant():
    import re

    from modelroom import lmstudio_install as lms

    assert lms.LMS_QUANTS
    for quant in lms.LMS_QUANTS:
        name = lms.install_name(_package(quant=quant))
        line = runtimes.runtime_of("lmstudio").command(name)
        assert re.fullmatch(r"lms get --gguf [A-Za-z0-9_./@-]+", line), line
        assert "@" in name and not name.startswith("@")


def test_tensor_labels_are_not_lms_quants():
    from modelroom import lmstudio_install as lms

    for label in ("AWQ", "FP8", "W4A16", "GPTQ-4", "GPTQ-8", "W8A8"):
        assert label in QUANT_ORDER and label not in lms.LMS_QUANTS
    assert not any(label.startswith("UD-") for label in lms.LMS_QUANTS)
    assert "Q4_K_M" in lms.LMS_QUANTS


def test_a_ud_label_is_computed_only():
    from modelroom import lmstudio_install as lms

    package = _package(quant="UD-Q4_K_XL")
    assert lms.install_name(package) is None
    assert lms.lms_reason(package) == "computed only: lms get for UD-Q4_K_XL is not checked"


def test_a_split_package_is_computed_only():
    from modelroom import lmstudio_install as lms

    package = _split()
    assert lms.install_name(package) is None
    assert lms.lms_reason(package) == "computed only: lms get for split files is not checked"


def test_a_single_file_in_a_subfolder_is_computed_only():
    from modelroom import lmstudio_install as lms

    package = _package(file="Q4/model-Q4_K_M.gguf")
    assert lms.install_name(package) is None
    assert lms.lms_reason(package) == "computed only: lms get for files in a subfolder is not checked"


def test_the_command_reads_the_same_in_posix_and_powershell():
    import shlex

    from modelroom import lmstudio_install as lms

    line = runtimes.runtime_of("lmstudio").command(lms.install_name(_package()))
    assert shlex.split(line) == ["lms", "get", "--gguf", "unsloth/Qwen3.5-4B-GGUF@Q4_K_M"]
    assert _powershell_tokens(line) == ["lms", "get", "--gguf", "unsloth/Qwen3.5-4B-GGUF@Q4_K_M"]


def _powershell_tokens(line: str) -> list[str]:
    """A simple PowerShell rule: words split at spaces; an `@` at the start of a word would be splatting."""
    tokens = line.split(" ")
    assert not any(token.startswith(("@", "$", "&")) for token in tokens)
    return tokens


def test_the_home_call_on_windows_is_the_powershell_form():
    from modelroom import lmstudio_install as lms

    line = lms.home_command("unsloth/Qwen3.5-4B-GGUF@Q4_K_M", "win32")
    assert line == '& "$env:USERPROFILE\\.lmstudio\\bin\\lms.exe" get --gguf unsloth/Qwen3.5-4B-GGUF@Q4_K_M'


def test_lms_only_in_home_prints_the_posix_call_elsewhere():
    from modelroom import lmstudio_install as lms

    line = lms.home_command("unsloth/Qwen3.5-4B-GGUF@Q4_K_M", "linux")
    assert line == '"$HOME/.lmstudio/bin/lms" get --gguf unsloth/Qwen3.5-4B-GGUF@Q4_K_M'


def test_the_home_call_carries_no_resolved_path(tmp_path):
    from modelroom import lmstudio_install as lms

    sentinel = str(tmp_path / "sentinel-home")
    for system in ("win32", "linux"):
        line = lms.home_command("unsloth/Qwen3.5-4B-GGUF@Q4_K_M", system)
        assert sentinel not in line and ("$env:USERPROFILE" in line or "$HOME" in line)


@pytest.mark.skipif(sys.platform != "win32", reason="the PowerShell form runs under Windows")
def test_the_powershell_home_call_resolves_its_variable_and_runs_a_batch_stand_in(tmp_path):
    """What this proves, and no more: PowerShell resolves `$env:USERPROFILE` in the form and calls a program of the
    shape `& "<path>" get --gguf <name>` with those arguments. The stand-in is a batch file (`lms.cmd`); the
    unchanged call of an executable (`lms.exe`) is not run, no native stand-in being at hand."""
    from modelroom import lmstudio_install as lms

    program = tmp_path / ".lmstudio" / "bin"
    program.mkdir(parents=True)
    (program / "lms.exe").write_bytes(b"")
    shell = "powershell"
    # A stand-in that prints its arguments: a batch file cannot be called as `lms.exe`, so the call is rebuilt
    # with the same tokens against a script of the same shape.
    (program / "lms.cmd").write_text("@echo off\r\necho %*\r\n", encoding="ascii")
    line = lms.home_command("unsloth/Qwen3.5-4B-GGUF@Q4_K_M", "win32").replace("lms.exe", "lms.cmd")
    result = subprocess.run(
        [shell, "-NoProfile", "-Command", line], env={**os.environ, "USERPROFILE": str(tmp_path)}, capture_output=True, text=True, timeout=60
    )
    assert result.stdout.strip() == "get --gguf unsloth/Qwen3.5-4B-GGUF@Q4_K_M"


def test_unknown_call_form_prints_the_path_form_with_the_hint():
    from modelroom import lmstudio_install as lms

    lines = lms.hint_lines(None, "win32", 32768)
    assert lines[0] == "if lms is not found: it is in .lmstudio/bin in your home folder"
    assert lms.command_for(None, "unsloth/Qwen3.5-4B-GGUF@Q4_K_M", "win32") == "lms get --gguf unsloth/Qwen3.5-4B-GGUF@Q4_K_M"


def test_the_hint_names_the_context_of_the_run():
    from modelroom import lmstudio_install as lms

    lines = lms.hint_lines("path", "win32", 32768)
    assert lines == [
        "in LM Studio: load it with context length 32768, then start its server (the server switch in LM Studio, or lms server start)",
        "then run modelroom again: it finds the server; LM Studio is not measured yet",
    ]
    assert "context length 8192" in lms.hint_lines("path", "linux", 8192)[0]


def test_the_home_hint_names_the_shell_on_windows_only():
    from modelroom import lmstudio_install as lms

    win = lms.hint_lines("home", "win32", 8192)[0]
    posix = lms.hint_lines("home", "linux", 8192)[0]
    assert win == "lms is not on the PATH yet; this line calls it where LM Studio put it (PowerShell)"
    assert posix == "lms is not on the PATH yet; this line calls it where LM Studio put it"


# --- the model folder and "already there" -------------------------------------------------------------


def test_the_lmstudio_place_is_home_lmstudio_models_on_each_system():
    from modelroom import lmstudio_install as lms

    for system in ("win32", "linux", "darwin"):
        assert lms.model_place({"LMSTUDIO_HOME": "ignored"}, Path("/h"), system) == [Path("/h") / ".lmstudio" / "models"]
    assert runtimes.runtime_of("lmstudio").model_place({}, Path("/h"), "linux") == [Path("/h") / ".lmstudio" / "models"]


def _folder(tmp_path: Path) -> Path:
    place = tmp_path / ".lmstudio" / "models"
    place.mkdir(parents=True)
    return place


def _write(place: Path, package: Package, size: int | None = None) -> Path:
    file = package.files[0]
    target = place / package.repo / file.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * (file.size_bytes if size is None else size))
    return target


def test_the_single_file_with_its_size_is_on_disk(tmp_path):
    from modelroom import lmstudio_install as lms

    place = _folder(tmp_path)
    package = _package(size=1500)
    _write(place, package)
    assert lms.package_on_disk(package, place) is True


def test_a_missing_models_folder_is_not_on_disk_and_no_error(tmp_path):
    from modelroom import lmstudio_install as lms

    assert lms.package_on_disk(_package(size=1500), tmp_path / "nowhere" / "models") is False


def test_a_wrong_size_is_not_on_disk(tmp_path):
    from modelroom import lmstudio_install as lms

    place = _folder(tmp_path)
    package = _package(size=1500)
    _write(place, package, size=1499)
    assert lms.package_on_disk(package, place) is False


def test_a_zero_size_file_missing_is_null(tmp_path):
    from modelroom import lmstudio_install as lms

    place = _folder(tmp_path)
    assert lms.package_on_disk(_package(size=0), place) is None


def test_split_or_subfolder_packages_are_unknown_without_touching_files(tmp_path):
    from modelroom import lmstudio_install as lms

    touched = []
    probes = CacheProbes(realpath=lambda path: touched.append(path) or path, stat=lambda path: touched.append(path))
    place = _folder(tmp_path)
    assert lms.package_on_disk(_split(), place, probes) is None
    assert lms.package_on_disk(_package(file="Q4/model-Q4_K_M.gguf"), place, probes) is None
    assert lms.package_on_disk(_package(quant="UD-Q4_K_XL"), place, probes) is None
    assert touched == []


def test_on_disk_refuses_unsafe_names_without_touching_files(tmp_path):
    from modelroom import lmstudio_install as lms

    touched = []
    probes = CacheProbes(realpath=lambda path: touched.append(path) or path, stat=lambda path: touched.append(path))
    place = _folder(tmp_path)
    unsafe = _package().model_copy(update={"repo": "a/b;rm"})
    odd_file = _package(file="..\\x.gguf")
    assert lms.package_on_disk(unsafe, place, probes) is None
    assert lms.package_on_disk(odd_file, place, probes) is None
    assert touched == []


def test_a_link_leaving_the_repo_folder_is_unknown(tmp_path):
    from modelroom import lmstudio_install as lms

    place = _folder(tmp_path)
    package = _package(size=1500)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / package.files[0].name).write_bytes(b"x" * 1500)
    owner = place / "unsloth"
    owner.mkdir()
    try:
        os.symlink(outside, owner / "Qwen3.5-4B-GGUF", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this account may not create links")
    assert lms.package_on_disk(package, place) is None


def test_a_file_that_is_a_link_to_another_place_is_unknown(tmp_path):
    from modelroom import lmstudio_install as lms

    place = _folder(tmp_path)
    package = _package(size=1500)
    elsewhere = tmp_path / "elsewhere.gguf"
    elsewhere.write_bytes(b"x" * 1500)
    folder = place / "unsloth" / "Qwen3.5-4B-GGUF"
    folder.mkdir(parents=True)
    try:
        os.symlink(elsewhere, folder / package.files[0].name)
    except (OSError, NotImplementedError):
        pytest.skip("this account may not create links")
    assert lms.package_on_disk(package, place) is None


def test_disk_and_on_disk_use_the_catalog_revision_sizes(tmp_path):
    """The check counts the files and sizes of the catalog revision of the package, whatever else is there."""
    from modelroom import lmstudio_install as lms

    place = _folder(tmp_path)
    package = _package(size=1500)
    _write(place, package, size=2000)  # a newer, larger file of the same name is not the catalogued one
    assert lms.package_on_disk(package, place) is False
    room = _room(tmp_path)
    disk = lms.lmstudio_disk_at(room, [package])
    assert disk.held == frozenset()
    _write(place, package)
    assert lms.lmstudio_disk_at(room, [package]).held == frozenset({package_identity_key(package)})


def _room(tmp_path: Path, free: int = 100 * GIB) -> RoomProbes:
    return replace(
        fixed_room(), home=tmp_path, system="linux", environ={}, exists=lambda path: Path(path).exists(),
        resolve=lambda path: path, disk_usage=lambda path: free, mount_of=lambda path: "/data",
    )


def test_on_disk_reads_the_run_through_the_table(tmp_path):
    place = _folder(tmp_path)
    package = _package(size=1500)
    _write(place, package)
    run = type("Run", (), {"room": _room(tmp_path)})()
    held = runtimes.runtime_of("lmstudio").on_disk(run, [package])
    assert held == frozenset({package_identity_key(package)})
    assert runtimes.runtime_of("lmstudio").on_disk(object(), [package]) == frozenset()  # no room to read


def test_the_disk_of_the_place_is_measured_where_the_folder_is_or_above(tmp_path):
    from modelroom import lmstudio_install as lms

    room = _room(tmp_path, free=7 * GIB)
    disk = lms.lmstudio_disk_at(room, [])
    assert disk.free_bytes == 7 * GIB and disk.drive == "/data"


# --- the helper that hangs the entry on the disk of a run ---------------------------------------------


def _config_with(tmp_path: Path, runtimes_: list[str] | None, here: str = "here"):
    """A configuration whose machine `here` has a stored profile with these runtimes (`None`: no profile entry)."""
    import json

    from modelroom.config import Configuration

    hardware = tmp_path / "state" / "hardware"
    hardware.mkdir(parents=True, exist_ok=True)
    machines: dict = {}
    if runtimes_ is not None:
        payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
        payload["runtimes"] = runtimes_
        profile_id = payload["profile_id"]
        (hardware / f"{profile_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        machines[here] = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True, "profile": profile_id}
    return Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [{"name": "q", "base_models": [{"hf_repo": "Qwen/Qwen3-8B"}]}],
            "packagers": ["unsloth"],
            "publishers": ["Qwen"],
            "machines": machines,
            "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "docs" / "models.md")},
        }
    )


def test_lmstudio_disk_reads_the_profile_of_here_like_the_render(tmp_path):
    from modelroom import lmstudio_install as lms

    room = _room(tmp_path)
    disk = Disk(50 * GIB, "C:")
    with_it = lms.lmstudio_disk(disk, room, _config_with(tmp_path, ["ollama", "lmstudio"]), "here", [])
    assert with_it.runtimes == frozenset({"ollama", "lmstudio"})
    assert with_it.lmstudio is not None and with_it.lmstudio.free_bytes == 100 * GIB
    # every other case: the very same object comes back
    assert lms.lmstudio_disk(disk, room, _config_with(tmp_path, ["ollama"]), "here", []) is disk
    assert lms.lmstudio_disk(disk, room, _config_with(tmp_path, ["ollama", "lmstudio"]), "elsewhere", []) is disk
    assert lms.lmstudio_disk(disk, room, _config_with(tmp_path, None), "here", []) is disk
    config = _config_with(tmp_path, ["ollama", "lmstudio"])
    for stored in (tmp_path / "state" / "hardware").glob("*.json"):
        stored.unlink()  # a machine whose profile file is gone
    assert lms.lmstudio_disk(disk, room, config, "here", []) is disk


def test_lmstudio_install_and_runtimes_import_in_either_order():
    for first, second in (("modelroom.lmstudio_install", "modelroom.runtimes"), ("modelroom.runtimes", "modelroom.lmstudio_install")):
        done = subprocess.run(
            [sys.executable, "-c", f"import {first}; import {second}"], capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        )
        assert done.returncode == 0, done.stderr


def test_the_head_imports_of_the_module_stay_small():
    import ast

    source = (Path(__file__).resolve().parent.parent / "modelroom" / "lmstudio_install.py").read_text(encoding="utf-8")
    heads = {node.module for node in ast.parse(source).body if isinstance(node, ast.ImportFrom) and node.level == 1}
    assert not heads & {"runtimes", "guided", "importer", "intro", "views"}
