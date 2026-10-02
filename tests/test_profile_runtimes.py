"""A measurement remembers the runtimes of its machine (AP9-SV2-I, Codex r1, R1-1; brief F4 "Gemerkt").

`hardware` and the guided run's measurement write `HardwareProfile.runtimes` from the `detect` of every
row of `RUNTIMES` (`runtimes.local_runtime_keys`): `serving` and `installed` count, `absent` and `asks_key`
do not, and an empty list is not written. A machine with vLLM installed and not running is the first-start
case of I1 and gets its vLLM rows. Every probe is the run's own (daemon, transport, room); never Docker.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from modelroom.cli import main
from modelroom.daemon import DaemonError
from modelroom.dialog import FileAsker
from modelroom.guided import run_guided
from modelroom.profile import HardwareProfile
from modelroom.runtimes import local_runtime_keys, machine_has_vllm, present_runtimes
from modelroom.screen import plain
from modelroom.vllm_server import MODELS_PATH, FixtureVllmServer, VllmKeyError

from fixture_room import fixed_room
from fixture_support import offline_daemon, vllm_servers, windows_probes
from test_hardware_cmd import ID_ONE, RUN1, _probes, _write_config_file
from test_install_vllm import SERVE, _package
from test_golden_vllm import _base_model, _gguf_package, _profile, _render
from test_intro import _daemon as _ollama_daemon

NO_SERVER = FixtureVllmServer({}, unreachable="no server")


class _Down:
    """An Ollama daemon that does not answer."""

    base_url = "http://127.0.0.1:11434"

    def __call__(self, *args, **kwargs):
        raise DaemonError("no daemon")


def _linux(*, vllm_in_path: bool = True, ollama_in_path: bool = False, system: str = "linux"):
    """A room of a machine of `system` whose `PATH` holds `vllm` (and `ollama`) or not."""
    present = {name for name, there in (("vllm", vllm_in_path), ("ollama", ollama_in_path)) if there}
    return dataclasses.replace(
        fixed_room(),
        system=system,
        home=Path("/srv/user"),
        exists=lambda path: False,
        which=lambda name: f"/usr/bin/{name}" if name in present else None,
    )


def _hardware(tmp_path: Path, *, daemon, servers, room) -> dict:
    """`modelroom hardware` with these probes; the profile it wrote, as the file holds it."""
    config_path = _write_config_file(tmp_path)
    code = main(
        ["hardware", "--config", str(config_path)],
        probes=_probes(),
        now=RUN1,
        pointer_path=tmp_path / "home" / "guided.json",
        daemon=daemon,
        servers=servers,
        room=room,
    )
    assert code == 0
    return json.loads((tmp_path / "state" / "hardware" / f"{ID_ONE}.json").read_text(encoding="utf-8"))


# --- hardware ----------------------------------------------------------------------------------------------


def test_vllm_installed_and_not_serving_is_remembered_beside_ollama(tmp_path: Path):
    """The first-start case: Linux, `vllm` in the `PATH`, no server answering, Ollama up."""
    written = _hardware(tmp_path, daemon=_ollama_daemon(), servers={"vllm": NO_SERVER}, room=_linux())

    assert written["runtimes"] == ["ollama", "vllm"]
    profile = HardwareProfile.model_validate(written)
    assert present_runtimes(profile, None) == {"ollama", "vllm"} and machine_has_vllm(profile, None)


def test_a_serving_vllm_is_remembered_the_same_way(tmp_path: Path):
    written = _hardware(tmp_path, daemon=_ollama_daemon(), servers=vllm_servers(), room=_linux(vllm_in_path=False))

    assert written["runtimes"] == ["ollama", "vllm"]


def test_without_ollama_only_vllm_is_remembered(tmp_path: Path):
    written = _hardware(tmp_path, daemon=_Down(), servers={"vllm": NO_SERVER}, room=_linux())

    assert written["runtimes"] == ["vllm"]


def test_an_installed_ollama_that_does_not_answer_still_counts(tmp_path: Path):
    written = _hardware(tmp_path, daemon=_Down(), servers={"vllm": NO_SERVER}, room=_linux(ollama_in_path=True))

    assert written["runtimes"] == ["ollama", "vllm"]


@pytest.mark.parametrize("system", ["win32", "darwin"])
def test_vllm_in_the_path_on_windows_or_a_mac_is_no_vllm(tmp_path: Path, system):
    written = _hardware(tmp_path, daemon=_ollama_daemon(), servers={"vllm": NO_SERVER}, room=_linux(system=system))

    assert written["runtimes"] == ["ollama"]
    assert "vllm" not in written["runtimes"]
    assert "vllm" not in present_runtimes(HardwareProfile.model_validate(written), None)


def test_a_server_that_asks_for_a_key_does_not_count(tmp_path: Path):
    key = FixtureVllmServer({("GET", MODELS_PATH): VllmKeyError("key")})
    written = _hardware(tmp_path, daemon=_ollama_daemon(), servers={"vllm": key}, room=_linux(vllm_in_path=False))

    assert written["runtimes"] == ["ollama"]


def test_nothing_found_writes_no_field_and_the_profile_reads_as_ollama_only(tmp_path: Path):
    written = _hardware(tmp_path, daemon=_Down(), servers={"vllm": NO_SERVER}, room=_linux(vllm_in_path=False))

    assert "runtimes" not in written
    assert present_runtimes(HardwareProfile.model_validate(written), None) == {"ollama"}


def test_the_keys_come_in_table_order_and_never_ask_docker():
    asked: list[str] = []

    def which(name: str):
        asked.append(name)
        return None

    room = dataclasses.replace(_linux(), which=which)

    assert local_runtime_keys(_ollama_daemon(), {"vllm": FixtureVllmServer({}, unreachable="x")}, room) == ["ollama"]
    assert "docker" not in asked


def test_an_installed_vllm_gets_its_rows_and_install_lines_in_the_table():
    """The profile a measurement of that machine writes (`runtimes == ["ollama", "vllm"]`) gives the vLLM rows
    and their install lines; one that names Ollama alone gives none."""
    from modelroom.views import RunView, terminal_blocks

    keys = local_runtime_keys(_ollama_daemon(), {"vllm": NO_SERVER}, _linux())
    assert keys == ["ollama", "vllm"]
    packages = [_gguf_package(), _package()]
    for runtimes, there in ((keys, True), (["ollama"], False)):
        document = _render(_profile().model_copy(update={"runtimes": runtimes}), [_base_model()], packages)
        text = "\n".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", packages=[_package()])))
        assert (SERVE in text and "docker" in text) is there, runtimes


def test_the_old_profile_without_the_field_is_unchanged():
    profile = _profile()
    old = HardwareProfile.model_validate({key: value for key, value in profile.model_dump(mode="json").items() if key != "runtimes"})

    assert old.runtimes == [] and present_runtimes(old, None) == {"ollama"}
    assert "runtimes" not in old.model_dump(mode="json")


# --- the guided measurement ----------------------------------------------------------------------------------


def test_the_guided_measurement_writes_the_runtimes_too(tmp_path: Path):
    from test_guided_vllm import FRESH
    from test_guided_vllm import RUN_AT

    results = tmp_path / "results"
    results.mkdir(parents=True)
    (tmp_path / "home").mkdir()
    from fixture_support import build_transport, guided_transport_mapping

    code = run_guided(
        FileAsker({**FRESH, "load_test": False}),
        here=results,
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=offline_daemon(),
        servers={"vllm": NO_SERVER},
        room=_linux(),
        now=RUN_AT,
        out=lambda line: None,
        colored=False,
    )

    assert code == 0
    files = sorted((results / "state" / "hardware").glob("*.json")) or sorted(results.rglob("hardware/*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["runtimes"] == ["vllm"]
