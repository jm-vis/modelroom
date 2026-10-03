"""The install line of a vLLM row: two commands, no start, no pull (AP9-SV2-I, I1 and I4).

modelroom prints the commands and starts nothing (GF 2026-09-29, F1 a): `vllm serve ...` and the
same server as one `docker run` line (F2 b), both built from the same checked values by
`install_vllm.vllm_install`. The list of step 5 takes the vLLM rows beside Ollama's (I4); the
disk and "already there" tests live in `test_hf_cache.py`. Hub facts are the brief's own worked
example: `Qwen/Qwen3-8B-AWQ` on four 22.49 GiB L4 cards, context 32 768, ten requests, `t = 4`.
Every test imports what AP9-SV2-I added inside itself, so a red run against the basis fails test
by test and not as one module that cannot be collected.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.daemon import PULL_PATH, FixtureDaemon
from modelroom.dialog import AnswerInvalidError, FileAsker
from modelroom.guided_install import PullContext, pull_step
from modelroom.measurements import Scenario
from modelroom.pull_choice import ambiguous_names
from modelroom.screen import plain

from fixture_support import json_response
from test_golden_vllm import FIXTURES, GOLDEN, _base_model, _gguf_package, _profile, _qwen3_base, _render, _tensor_package

NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
REPO = "Qwen/Qwen3-8B-AWQ"
REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
WEIGHTS = 6_098_581_864
EXPRESSION = "${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}"
SERVE = (
    "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 "
    f"vllm serve {REPO} --revision {REVISION} --max-model-len 32768 --tensor-parallel-size 4 "
    "--gpu-memory-utilization 0.92 --host 127.0.0.1 --port 8000"
)
DOCKER = (
    f'mkdir -p "{EXPRESSION}" && docker --host unix:///var/run/docker.sock run --rm --ipc=host '
    "--gpus '\"device=0,1,2,3\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 "
    f'-p 127.0.0.1:8000:8000 --mount "type=bind,src={EXPRESSION},dst=/root/.cache/huggingface/hub" '
    f"vllm/vllm-openai:v0.30.0 {REPO} --revision {REVISION} --max-model-len 32768 --tensor-parallel-size 4 "
    "--gpu-memory-utilization 0.92 --host 0.0.0.0 --port 8000"
)
COMPUTED_ONLY = "computed only: modelroom cannot name these cards to vLLM"
RELATIVE_NOTE = "docker: left out, the Hugging Face cache is set as a relative path; set it absolute to get this line"
STOP_LINE = "vLLM serves another model on port 8000 now; stop it first (one model per server)"
CACHE_VARIABLES = ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "CUDA_VISIBLE_DEVICES", "DOCKER_HOST", "DOCKER_CONTEXT")


# --- fixtures ---------------------------------------------------------------------------------------


def _package(repo: str = REPO, revision: str = REVISION, **changes):
    """The brief's AWQ package of `Qwen/Qwen3-8B` (6 098 581 864 bytes of weights)."""
    from modelroom.contracts import PackageFile

    files = [PackageFile(name="model.safetensors", role="weights", size_bytes=WEIGHTS, digest=None)]
    return _tensor_package().model_copy(update={"repo": repo, "revision": revision, "files": files, **changes})


def _scenario(context: int = 32768, requests: int = 10) -> Scenario:
    return Scenario(context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=requests)


def _cards(sizes, source: str = "nvidia-smi"):
    """The four-L4 profile with its cards replaced: one `GpuAdapter` per size, `index` = position."""
    from modelroom.profile import GpuAdapter

    gpus = [GpuAdapter(index=index, name="NVIDIA L4", vram_gib=size, vram_source=source) for index, size in enumerate(sizes)]
    return _profile().model_copy(update={"gpus": gpus})


def _fit(profile, package, heads: int | None = 32, scenario: Scenario | None = None):
    from modelroom.fit_vllm import vllm_fit_as_fit

    return vllm_fit_as_fit(profile, package, _qwen3_base(heads), scenario or _scenario(), None)


def _install(profile=None, package=None, heads: int | None = 32, *, relative: bool = False):
    from modelroom.install_vllm import vllm_install

    profile = profile if profile is not None else _profile()
    package = package if package is not None else _package()
    return vllm_install(package, _fit(profile, package, heads), profile, relative_cache=relative)


def _four(sizes=(22.49,) * 4):
    return _cards(list(sizes))


def _both():
    return _render(_profile(), [_base_model()], [_gguf_package(), _package()])


def _ollama_only():
    return _render(_profile().model_copy(update={"runtimes": ["ollama"]}), [_base_model()], [_gguf_package(), _package()])


def _sh() -> str:
    found = shutil.which("sh")
    if found is None:  # pragma: no cover - a machine without a POSIX shell cannot run the shell tests
        pytest.skip("no POSIX shell on this machine")
    return found


def _env(**extra: str) -> dict[str, str]:
    """The environment of this process without the variables a command line reads, plus `extra`."""
    env = {key: value for key, value in os.environ.items() if key not in CACHE_VARIABLES}
    env.update(extra)
    return env


def _attrappe(folder: Path, name: str, body: str) -> Path:
    """A one-file stand-in for a program, written as bytes with LF, executable, first in the `PATH` of a test.

    Its first output line is `STUB <name>`: a test checks it, so a real `vllm` or `docker` of the machine
    running the suite never answers unnoticed (Codex r1, R1-5).
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(f'#!/bin/sh\necho "STUB {name}"\n{body}\n'.encode("utf-8"))
    path.chmod(0o755)
    return folder


def _on_path(folder: Path, shell: str | None = None) -> str:
    """The stand-ins first, then only the system default path (`mkdir`, `sh`): never the inventory of the
    machine's own `PATH`. On Windows the tools come from the Git installation of `shell` alone (Codex r2,
    R1-5): its own folder (`usr\\bin`, where `mkdir.exe` lives) and, where `shell` is the launcher in
    `<git>\\bin`, the `usr\\bin` beside it; nothing is looked up in the inherited `PATH`."""
    parts = [str(folder)]
    if shell is not None and os.name == "nt":
        shell_dir = Path(shell).parent
        parts.append(str(shell_dir))
        usr_bin = shell_dir.parent / "usr" / "bin"
        if usr_bin.is_dir():
            parts.append(str(usr_bin))
    parts.append(os.defpath)
    return os.pathsep.join(parts)


# --- I1: the vllm serve line -------------------------------------------------------------------------


def test_the_command_carries_revision_context_tp_utilization_loopback_and_no_tracking():
    install = _install()

    assert install.serve == SERVE
    for part in (
        f"--revision {REVISION}", "--max-model-len 32768", "--tensor-parallel-size 4", "--gpu-memory-utilization 0.92",
        "--host 127.0.0.1", "--port 8000", "VLLM_NO_USAGE_STATS=1", "DO_NOT_TRACK=1",
    ):
        assert part in install.serve
    assert "0.0.0.0" not in install.serve and install.reason is None


@pytest.mark.parametrize("repo", ["a/b;rm", "$(x)/y", "a/`b`", "a/b c", "../x", "-x/y", "a/b\n", "ü/ü", "a/b|c", "ab", ""])
def test_an_unsafe_repo_or_revision_has_no_command_repo(repo):
    install = _install(package=_package(repo=repo))

    assert install.serve is None and install.docker is None and install.reason == "no safe command"


@pytest.mark.parametrize("revision", [None, "main", "a" * 39, "A" * 40, "g" * 40, "a" * 40 + "\n", "a" * 40 + ";x", " " + "a" * 39])
def test_an_unsafe_repo_or_revision_has_no_command(revision):
    install = _install(package=_package(revision=revision))

    assert install.serve is None and install.docker is None and install.reason == "no safe command"


@pytest.mark.parametrize("sizes, bound", [([8, 24, 24], "1,2"), ([24, 8, 24], "0,2"), ([24, 24, 8], "0,1")])
def test_cards_are_bound_by_index_in_every_order(sizes, bound):
    install = _install(_cards(sizes))

    assert install.serve.startswith(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={bound} ")
    assert "--tensor-parallel-size 2" in install.serve
    assert f"--gpus '\"device={bound}\"'" in install.docker and "-e CUDA_DEVICE_ORDER=PCI_BUS_ID" in install.docker


def test_all_cards_are_bound_too():
    install = _install(_four())

    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 " in install.serve
    assert "--tensor-parallel-size 4" in install.serve


def test_a_subset_of_equal_cards_is_bound():
    install = _install(_four(), heads=14)  # 14 heads: two of the four cards

    assert install.serve.startswith("CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 ")
    assert "--tensor-parallel-size 2" in install.serve


HINT = "larger --tensor-parallel-size spreads the load over more cards, if the server is yours alone"


def _block_text(profile, package, scenario: Scenario | None = None, heads: int | None = 32) -> str:
    from modelroom.install_vllm import install_block_lines, vllm_install

    fit = _fit(profile, package, heads, scenario)
    lines = install_block_lines("install", 1, vllm_install(package, fit, profile), repo=package.repo)
    return "\n".join(plain(line) for line in lines)


def test_the_hint_under_the_commands_only_where_more_equal_cards_exist_than_t():
    """AP9-SV2-D, rule (b): 1 GB and 3 requests give `t = 1` of four cards -- the hint, once; the
    worked example takes all four cards, and two cards with `t = 2` take both: no hint."""
    base = _package()
    small = base.model_copy(update={"files": [base.files[0].model_copy(update={"size_bytes": 1_000_000_000})]})

    spare = _block_text(_four(), small, _scenario(requests=3))
    assert spare.count(HINT) == 1 and "--tensor-parallel-size 1 " in spare
    assert spare.index(HINT) > spare.index("docker") and spare.index(HINT) < spare.index("run one of the two")

    assert HINT not in _block_text(_four(), _package())  # t = 4 of 4
    assert HINT not in _block_text(_cards([22.49, 22.49]), _package(), _scenario(requests=7))  # t = 2 of 2


def test_the_hint_needs_a_larger_size_the_model_allows():
    """D1-3 (Codex d1): four cards and 14 attention heads allow `t = 1, 2` only, so `t = 2` has no larger
    size to point to; an unknown head count computes on one card and names no size at all. Neither says it."""
    four = _four()
    assert HINT not in _block_text(four, _package(), heads=14)  # t = 2 (best try), 4 does not divide 14
    assert HINT not in _block_text(four, _package(), heads=None)  # t = 1 for want of a head count
    assert HINT in _block_text(four, _package(), _scenario(requests=3), heads=14)  # t = 1: 2 is allowed


def test_an_inherited_mask_is_overridden_by_the_prefix(tmp_path: Path):
    """The line, run in a POSIX shell whose environment already masks the cards to `0,1`, reaches
    `vllm` with all four (a stand-in prints the variable it sees)."""
    shell = _sh()
    folder = _attrappe(tmp_path / "bin", "vllm", 'printf "%s\\n" "$CUDA_VISIBLE_DEVICES"')
    env = _env(PATH=_on_path(folder, shell), CUDA_VISIBLE_DEVICES="0,1")

    done = subprocess.run([shell, "-c", _install().serve], env=env, capture_output=True, text=True, check=True)

    assert done.stdout.split() == ["STUB", "vllm", "0,1,2,3"]  # the stand-in ran, and saw all four cards


def test_the_stand_ins_are_executable_and_the_inventory_of_the_path_is_not_searched(tmp_path: Path, monkeypatch):
    """R1-5: a stand-in is executable, and `PATH` holds the stand-ins and the system default only: a `vllm`
    that lives elsewhere on the machine's own `PATH` is never found, so a real program cannot answer."""
    shell = _sh()
    folder = _attrappe(tmp_path / "bin", "vllm", "true")
    assert os.name == "nt" or (folder / "vllm").stat().st_mode & 0o111 == 0o111
    inventory = _attrappe(tmp_path / "inventory", "vllm-real", "true")
    (inventory / "vllm-real").rename(inventory / "inventory-only")
    monkeypatch.setenv("PATH", f"{inventory}{os.pathsep}{os.environ.get('PATH', '')}")
    env = _env(PATH=_on_path(folder, shell))

    found = subprocess.run([shell, "-c", "command -v inventory-only"], env=env, capture_output=True, text=True)
    own = subprocess.run([shell, "-c", "vllm"], env=env, capture_output=True, text=True)

    assert found.returncode != 0 and found.stdout.strip() == ""
    assert own.returncode == 0 and own.stdout.split() == ["STUB", "vllm"]


@pytest.mark.parametrize("source", ["sysfs", "entered", "sysctl", "llmfit"])
@pytest.mark.parametrize("heads", [14, 32], ids=["subset", "all-cards"])
def test_cards_without_nvidia_smi_index_are_computed_only(source, heads):
    install = _install(_cards([22.49] * 4, source), heads=heads)

    assert install.serve is None and install.docker is None and install.reason == COMPUTED_ONLY


def test_a_card_left_out_may_come_from_anywhere():
    """Only the cards the line names must be `nvidia-smi`'s: the smaller T4-like card left out of the group is not named."""
    from modelroom.profile import GpuAdapter

    profile = _four()
    gpus = [*profile.gpus[:3], GpuAdapter(index=3, name="odd", vram_gib=8.0, vram_source="entered")]
    install = _install(profile.model_copy(update={"gpus": gpus}), heads=32)

    assert install.reason is None and "CUDA_VISIBLE_DEVICES=0,1" in install.serve


def test_computed_only_prints_neither_line():
    from modelroom.pull_choice import pull_rows
    from modelroom.views import RunView, terminal_blocks

    only = FIXTURES / "profiles_v4" / "four_l4_server_vllm_only.json"
    profile = _four_only_sysfs(only)
    package = _package()
    document = _render(profile, [_base_model()], [_gguf_package(), package])

    row = pull_rows(document, "server", packages=[package])[0]
    text = "\n".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", packages=[package])))

    assert row.runtime == "vllm" and row.locked == COMPUTED_ONLY and document.machines[0].ranked  # the fit stays
    assert COMPUTED_ONLY in text and "vllm serve" not in text and "docker" not in text


def test_a_no_safe_command_lock_prints_its_note_under_the_table():
    """R1-6: a package without a safe command name (a repo letters the Hub allows and a shell line never gets; a
    snapshot refuses a revision that is no digest, so the repo is the way in) says why under the table, like
    `computed only`, and the list row stays locked."""
    from modelroom.pull_choice import pull_rows
    from modelroom.views import RunView, terminal_blocks

    package = _package(repo="Qü/Qwen3-8B-AWQ")
    document = _render(_profile(), [_base_model()], [package])

    row = pull_rows(document, "server", packages=[package])[0]
    text = "\n".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", packages=[package])))

    assert row.runtime == "vllm" and row.locked == "no safe command"
    assert re.search(r"^ *install +#\d+ no safe command$", text, re.M), text
    assert "vllm serve" not in text and "docker" not in text


def _four_only_sysfs(path: Path):
    import json

    from modelroom.profile import HardwareProfile

    profile = HardwareProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    gpus = [gpu.model_copy(update={"vram_source": "sysfs"}) for gpu in profile.gpus]
    return HardwareProfile.model_validate({**profile.model_dump(mode="json"), "vram_source": "sysfs", "gpus": [gpu.model_dump(mode="json") for gpu in gpus]})


# --- I1: the docker line (F2 b, Rev 5 to 7) ----------------------------------------------------------


def test_both_lines_match_their_golden():
    install = _install()

    assert install.serve + "\n" == (GOLDEN / "golden_install_vllm_serve.txt").read_text(encoding="utf-8")
    assert install.docker + "\n" == (GOLDEN / "golden_install_vllm_docker.txt").read_text(encoding="utf-8")
    assert install.docker == DOCKER


def test_docker_binds_a_subset_and_all_cards():
    assert "--gpus '\"device=0,1\"'" in _install(_four(), heads=14).docker
    assert "--gpus '\"device=0,1,2,3\"'" in _install(_four()).docker
    assert "--gpus '\"device=1,2\"'" in _install(_cards([8, 24, 24])).docker
    for install in (_install(_four(), heads=14), _install(_four()), _install(_cards([8, 24, 24]))):
        assert "-e CUDA_DEVICE_ORDER=PCI_BUS_ID" in install.docker


def _shell_place(environ: dict[str, str], home: Path) -> Path:
    shell = _sh()
    env = _env(HOME=home.as_posix(), **environ)
    done = subprocess.run([shell, "-c", f'printf %s "{EXPRESSION}"'], env=env, capture_output=True, text=True, check=True)
    return Path(done.stdout)


def _same_place(shell: Path, expected: Path, variables: dict) -> bool:
    """Equal places; where only `$HOME` decides, equal below it -- an MSYS shell writes the home folder
    of a Windows machine in its own form (`/tmp/...` for `C:/.../Temp`), so only the part the rule adds is compared."""
    return shell.parts[-3:] == expected.parts[-3:] if not variables else shell == expected


@pytest.mark.parametrize(
    "variables",
    [{"HF_HUB_CACHE": "{t}/direct"}, {"HF_HOME": "{t}/hfhome"}, {"XDG_CACHE_HOME": "{t}/xdg"}, {}],
    ids=["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "none"],
)
def test_docker_mounts_the_same_cache_expression_as_hf_cache_candidates(variables, tmp_path: Path):
    """The expression in the line, run by a POSIX shell with one of the variables (or none), names the
    place `hf_cache_candidates` names: the download lands where the disk was checked."""
    from modelroom.hf_cache import hf_cache_candidates

    environ = {key: value.format(t=tmp_path.as_posix()) for key, value in variables.items()}
    home = tmp_path / "home"

    expected = hf_cache_candidates(environ, home, "linux")[0]

    assert re.search(r'--mount "type=bind,src=(.*?),dst=', _install().docker).group(1) == EXPRESSION
    assert _same_place(_shell_place(environ, home), expected, variables)


def test_docker_line_carries_no_path_and_only_checked_values():
    """Injection (Rev 5, Rev 6, Rev 7): an unsafe repo or revision, a number given as text -- no line at
    all, and the line that does come carries the fixed container path and the constant expression only."""
    install = _install()
    for hostile in ("a/b;rm", "$(x)/y", "a/`b`"):
        refused = _install(package=_package(repo=hostile))
        assert refused.docker is None and refused.serve is None and refused.reason == "no safe command"
    profile, package = _profile(), _package()
    fit = _fit(profile, package)
    from modelroom.install_vllm import vllm_install

    for field, value in (("context", "32768; rm -rf x"), ("cards", "4 && x"), ("context", True), ("cards", 0)):
        broken = fit.model_copy(update={field: value})
        refused = vllm_install(package, broken, profile)
        assert refused.docker is None and refused.serve is None and refused.reason == "no safe command", (field, value)
    assert re.search(r"(/home/|/Users/|[A-Za-z]:\\|/mnt/)", install.docker) is None
    assert "/root/.cache/huggingface/hub" in install.docker
    assert install.docker.count(EXPRESSION) == 2  # the `mkdir` part and the mount, nothing else


def test_docker_mounts_with_type_bind_never_dash_v():
    docker = _install().docker

    assert len(re.findall(r'--mount "type=bind,src=[^"]*,dst=/root/.cache/huggingface/hub"', docker)) == 1
    assert " -v " not in docker and "--volume" not in docker


@pytest.mark.parametrize("variable", ["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"])
def test_a_relative_cache_value_leaves_out_the_docker_line(variable, tmp_path: Path):
    from modelroom.hf_cache import cache_is_relative
    from modelroom.pull_choice import pull_rows
    from modelroom.views import RunView, terminal_blocks

    assert cache_is_relative({variable: "hf-cache"}, "linux") and not cache_is_relative({variable: "/abs/hf"}, "linux")
    relative = _install(relative=True)
    assert relative.docker is None and relative.docker_note == RELATIVE_NOTE and relative.serve == SERVE
    package = _package()
    document = _render(_profile(), [_base_model()], [_gguf_package(), package])
    from modelroom.room import Disk, LiveRoom

    vllm = Disk(10**12, "/", relative=True)
    live = {"server": LiveRoom(disk=Disk(10**12, "/", vllm=vllm))}
    text = "\n".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", live=live, packages=[package])))
    assert RELATIVE_NOTE in text and "mkdir -p" not in text and "vllm serve" in text
    assert pull_rows(document, "server", packages=[package])[-1].runtime == "vllm"


def test_a_relative_value_is_the_effective_one_only():
    from modelroom.hf_cache import cache_is_relative

    assert not cache_is_relative({"HF_HUB_CACHE": "/abs", "HF_HOME": "rel"}, "linux")  # HF_HUB_CACHE wins
    assert cache_is_relative({"HF_HOME": "rel", "XDG_CACHE_HOME": "/abs"}, "linux")  # HF_HOME wins
    assert not cache_is_relative({"HF_HOME": "", "XDG_CACHE_HOME": "/abs"}, "linux")  # empty is unset, as in the shell
    assert not cache_is_relative({}, "linux")


def test_docker_line_golden_has_mkdir_and_local_host():
    golden = (GOLDEN / "golden_install_vllm_docker.txt").read_text(encoding="utf-8").rstrip("\n")

    assert golden.startswith(f'mkdir -p "{EXPRESSION}" && docker --host unix:///var/run/docker.sock run')
    assert re.search(r'^mkdir -p "(.*?)" && ', golden).group(1) == re.search(r'src=(.*?),dst=', golden).group(1) == EXPRESSION


def test_docker_host_and_context_do_not_change_the_line(monkeypatch):
    golden = (GOLDEN / "golden_install_vllm_docker.txt").read_text(encoding="utf-8")
    for name, value in (("DOCKER_HOST", "tcp://192.0.2.10:2376"), ("DOCKER_CONTEXT", "remote1")):
        monkeypatch.setenv(name, value)
        assert _install().docker + "\n" == golden
        monkeypatch.delenv(name)


def test_first_start_creates_the_missing_cache(tmp_path: Path):
    """The line, run in a POSIX shell with an empty `HF_HOME` and a stand-in for `docker`: the `mkdir`
    part makes `<HF_HOME>/hub`, and the stand-in gets the local socket and a bind mount of that place."""
    shell = _sh()
    folder = _attrappe(tmp_path / "bin", "docker", 'printf "%s\\n" "$@"')
    hf_home = tmp_path / "hfhome"
    hf_home.mkdir()
    env = _env(PATH=_on_path(folder, shell), HF_HOME=hf_home.as_posix(), HOME=(tmp_path / "home").as_posix())

    done = subprocess.run([shell, "-c", _install().docker], env=env, capture_output=True, text=True, check=True)

    assert (hf_home / "hub").is_dir()
    assert done.stdout.splitlines()[0] == "STUB docker"
    arguments = done.stdout.splitlines()[1:]
    assert arguments[:4] == ["--host", "unix:///var/run/docker.sock", "run", "--rm"]
    assert f"type=bind,src={hf_home.as_posix()}/hub,dst=/root/.cache/huggingface/hub" in arguments
    assert '"device=0,1,2,3"' in arguments


# --- I1: where the lines may appear --------------------------------------------------------------------


def test_native_windows_gets_neither_line():
    """`vllm` in the `PATH` and Docker there, on `win32`: `detect` says `absent`, so the machine has no vLLM row at
    all and neither `vllm serve` nor a `docker` line is ever built."""
    from modelroom.runtimes import VLLM
    from modelroom.views import RunView, terminal_blocks

    for system in ("win32", "darwin"):
        state = VLLM.detect(None, system=system, which=lambda name: f"/bin/{name}", home=Path("/srv/user"), is_file=lambda path: True)
        assert state == "absent"
    document = _render(_profile().model_copy(update={"runtimes": ["ollama"]}), [_base_model()], [_gguf_package(), _package()])
    text = "\n".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", packages=[_package()])))
    assert "vllm serve" not in text and "docker" not in text


@pytest.mark.parametrize(
    "found, is_file, expected",
    [(None, False, "absent"), ("/usr/bin/vllm", False, "installed"), (None, True, "installed")],
)
def test_vllm_is_installed_by_path_or_the_user_bin_folder(found, is_file, expected):
    from modelroom.runtimes import VLLM

    state = VLLM.detect(None, system="linux", which=lambda name: found, home=Path("/srv/user"), is_file=lambda path: is_file)

    assert state == expected


def test_a_serving_or_key_asking_server_is_read_first():
    from modelroom.runtimes import VLLM
    from modelroom.vllm_server import FixtureVllmServer, VllmKeyError

    refuse = dict(system="linux", which=lambda name: None, home=Path("/srv/user"), is_file=lambda path: False)
    none = FixtureVllmServer({}, unreachable="no server")
    key = FixtureVllmServer({("GET", "/v1/models"): VllmKeyError("key")})
    assert VLLM.detect(none, **refuse) == "absent" and VLLM.detect(key, **refuse) == "asks_key"
    served = FixtureVllmServer({("GET", "/v1/models"): json_response("vllm_v1_models_l4.json")})
    assert VLLM.detect(served, **refuse) == "serving"


def test_modelroom_never_calls_docker(monkeypatch):
    """A process stand-in that throws at `docker`, over the detection, the table and the install line:
    the Docker line is output only, never a call (Codex r5-1, r5-3)."""
    from modelroom.runtimes import VLLM
    from modelroom.views import RunView, terminal_blocks

    seen: list[object] = []

    def refuse(*args, **kwargs):
        seen.append(args)
        if "docker" in repr(args) or "docker" in repr(kwargs):
            raise AssertionError("modelroom called docker")
        raise AssertionError("no process is started by the install line")

    real_which = shutil.which

    def which(name, *args, **kwargs):
        seen.append(name)
        assert name != "docker", "modelroom looked for docker"
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)
    monkeypatch.setattr(shutil, "which", which)
    package = _package()
    VLLM.detect(None, system="linux", which=shutil.which, home=Path("/srv/user"), is_file=lambda path: False)
    document = _render(_profile(), [_base_model()], [_gguf_package(), package])
    text = "\n".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", packages=[package])))

    assert "docker --host unix:///var/run/docker.sock run" in text  # printed, not run
    assert all(item != "docker" for item in seen if isinstance(item, str))


# --- I1: the lines under the table -------------------------------------------------------------------------


def _install_lines(view_extra: dict | None = None, document=None, package=None) -> list[str]:
    from modelroom.views import RunView, terminal_blocks

    package = package if package is not None else _package()
    document = document if document is not None else _both()
    view = RunView(install_for="server", packages=[_gguf_package(), package], **(view_extra or {}))
    return [plain(line) for line in terminal_blocks(document, view=view)]


def test_the_table_prints_both_commands_and_the_two_notes():
    text = _install_lines()
    vllm = next(index for index, line in enumerate(text) if line.startswith(" install   #2  "))

    assert text[vllm] == f" install   #2  {SERVE}"
    assert text[vllm + 1] == f"     docker    {DOCKER}"
    assert text[vllm + 2] == "           run one of the two in a terminal that stays open; vLLM downloads the model at its first start"
    assert text[vllm + 3] == "           then run modelroom again: step 4 finds the server and measures it"
    assert any(line.startswith(" install   #1  ollama pull ") for line in text)  # Ollama's own section is untouched


def test_another_served_model_adds_the_stop_line():
    from modelroom.vllm_server import ServedModel

    other = ServedModel(id="acme/Other", root="acme/Other", max_model_len=4096)
    same = ServedModel(id=REPO, root=REPO, max_model_len=32768)

    with_other = _install_lines({"live_vllm": {"server": other}})
    with_same = _install_lines({"live_vllm": {"server": same}})
    index = next(index for index, line in enumerate(with_other) if line.startswith(" install   #2  "))

    assert f"           {STOP_LINE}" in with_other[index + 1 : index + 6]
    assert not any(STOP_LINE in line for line in with_same)


def test_the_handover_for_another_computer_carries_the_serve_command(tmp_path: Path):
    """A machine this run cannot reach: both commands with the machine named, and no `modelroom pull`."""
    from modelroom.pull_other import pull_other_step

    package = _package()
    document = _both()
    asker = FileAsker({"pull_other": f"server vllm {REPO}"})
    run, lines = _run(tmp_path, asker)

    pull_other_step(run, PullContext(document, "laptop", [_gguf_package(), package], ambiguous_names([_gguf_package(), package])))

    text = "\n".join(lines)
    assert f"#2 on server  {SERVE}" in text and DOCKER in text
    assert "modelroom pull" not in text and "ollama pull" not in text


# --- I4: the list and the question ----------------------------------------------------------------------------


class _Asking(FileAsker):
    """An answer file that keeps the question and the choices of every list it was asked."""

    def __init__(self, answers: dict) -> None:
        super().__init__(answers)
        self.questions: dict[str, str] = {}
        self.lists: dict[str, list] = {}

    def select(self, key, question, choices, instruction=None):
        self.questions[key] = question
        self.lists[key] = list(choices)
        return super().select(key, question, choices, instruction)


def _daemon() -> FixtureDaemon:
    from modelroom.daemon import TAGS_PATH

    return FixtureDaemon({("GET", TAGS_PATH): json_response("ollama_tags_local_loadtest.json")})


def _run(tmp_path: Path, asker, daemon=None, room=None):
    from fixture_room import fixed_room
    from fixture_support import offline_servers, windows_probes
    from modelroom.catalog import load_catalog
    from modelroom.guided import GuidedRun
    from modelroom.http import FixtureTransport
    from modelroom.local_servers import ServerRun
    from modelroom.screen import Screen

    lines: list[str] = []
    run = GuidedRun(
        asker=asker, here=tmp_path, pointer_path=tmp_path / "guided.json", transport=FixtureTransport({}),
        probes=windows_probes(), now=NOW, catalog=load_catalog(), daemon=daemon if daemon is not None else _daemon(),
        servers=ServerRun(offline_servers()), out=lines.append, screen=Screen(lines.append, colored=False),
        room=room if room is not None else fixed_room(),
    )
    return run, lines


def _step(tmp_path: Path, document, answers: dict, packages=None, daemon=None):
    packages = packages if packages is not None else [_gguf_package(), _package()]
    asker = _Asking(answers)
    run, lines = _run(tmp_path, asker, daemon)
    pull_step(run, PullContext(document, "server", packages, ambiguous_names(packages)), True)
    return asker, lines, run


def test_the_list_question_changes_only_with_vllm_rows(tmp_path: Path):
    asker, _lines, _run_ = _step(tmp_path, _both(), {"pull_package": "none"})
    assert asker.questions["pull_package"] == "Which package should be installed?"
    values = [choice.value for choice in asker.lists["pull_package"]]
    assert values[-2:] == [f"vllm {REPO}", "none"] and values[0].startswith("hf.co/")

    ollama_asker, _lines, _run_ = _step(tmp_path, _ollama_only(), {"pull_package": "none"})
    assert ollama_asker.questions["pull_package"] == "Which package should be pulled into Ollama?"
    assert not any(choice.value.startswith("vllm ") for choice in ollama_asker.lists["pull_package"])


def test_choosing_a_vllm_row_prints_and_pulls_nothing(tmp_path: Path):
    daemon = _daemon()

    asker, lines, _run_ = _step(tmp_path, _both(), {"pull_package": f"vllm {REPO}", "pull": True}, daemon=daemon)

    text = "\n".join(lines)
    assert SERVE in text and DOCKER in text
    assert [call for call in daemon.calls if call[1] == PULL_PATH] == []
    assert "pull" not in asker.asked and "Pull #" not in text  # no yes/no question, no pull


def test_a_vllm_only_machine_is_offered_its_rows_without_a_daemon(tmp_path: Path):
    """No Ollama on the machine (the start screen did not reach a daemon): its vLLM rows are still the list."""
    import json

    from modelroom.profile import HardwareProfile

    data = json.loads((FIXTURES / "profiles_v4" / "four_l4_server_vllm_only.json").read_text(encoding="utf-8"))
    document = _render(HardwareProfile.model_validate(data), [_base_model()], [_gguf_package(), _package()])
    packages = [_gguf_package(), _package()]
    asker = _Asking({"pull_package": f"vllm {REPO}"})
    run, lines = _run(tmp_path, asker)

    pull_step(run, PullContext(document, "server", packages, ambiguous_names(packages)), False)

    assert [choice.value for choice in asker.lists["pull_package"]] == [f"vllm {REPO}", "none"]
    assert SERVE in "\n".join(lines)


def test_an_answer_file_picks_a_vllm_row(tmp_path: Path):
    _asker, lines, _run_ = _step(tmp_path, _both(), {"pull_package": f"vllm {REPO}"})
    assert any(line.endswith(f"#2  {REPO}") or f"#2  {REPO}" in line for line in lines)

    with pytest.raises(AnswerInvalidError):  # an unknown value is an error, as before
        _step(tmp_path, _both(), {"pull_package": "vllm acme/Unknown-AWQ"})


def test_a_list_row_of_vllm_says_its_runtime_and_a_locked_one_says_why(tmp_path: Path):
    asker, _lines, _run_ = _step(tmp_path, _both(), {"pull_package": "none"})
    row = next(choice for choice in asker.lists["pull_package"] if choice.value == f"vllm {REPO}")
    assert row.label.startswith("#2  vLLM") and row.label.endswith(REPO) and row.disabled is None

    unsafe = _package(repo="Qü/Qwen3-8B-AWQ")  # letters the hub name rule allows, a shell line never gets
    document = _render(_profile(), [_base_model()], [_gguf_package(), unsafe])
    locked, _lines, _run_ = _step(tmp_path, document, {"pull_package": "none"}, packages=[_gguf_package(), unsafe])
    row = next(choice for choice in locked.lists["pull_package"] if choice.label.startswith("#2"))
    assert row.disabled == "no safe command"
