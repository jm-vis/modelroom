"""End-to-end tests for modelroom.cli: `modelroom fetch --config ... --machine ...`.

`main()` takes optional `transport`/`now` keyword arguments precisely so these tests never
touch the network or the wall clock -- the real CLI entry point (`transport=None`) uses
`UrllibTransport()` and the real clock, exercised nowhere in this suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from modelroom.cli import fetch_with_config, hardware_with_config, main
from modelroom.config import Configuration
from modelroom.contracts import load_hardware_snapshot, load_snapshot
from modelroom.http import FixtureTransport, Response
from modelroom.llmfit import FixtureRunner

from fixture_support import FIXTURES, build_transport, json_response, qwen35_example_config_dict, qwen35_transport_mapping

RUN1 = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
RUN2 = datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc)


def _write_config(tmp_path: Path, **overrides) -> Path:
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data.update(overrides)
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(_to_toml(data), encoding="utf-8")
    return config_path


def _to_toml(data: dict) -> str:
    # A tiny, purpose-built writer for the shapes qwen35_example_config_dict produces --
    # good enough for these tests, not a general TOML serializer.
    lines = [f"schema_version = {data['schema_version']}"]
    lines.append(f"packagers = {json.dumps(data['packagers'])}")
    lines.append(f"publishers = {json.dumps(data['publishers'])}")
    for family in data["families"]:
        lines.append("[[families]]")
        lines.append(f'name = "{family["name"]}"')
        for base_model in family["base_models"]:
            lines.append("  [[families.base_models]]")
            lines.append(f'  hf_repo = "{base_model["hf_repo"]}"')
            lines.append(f'  repo_aliases = {json.dumps(base_model["repo_aliases"])}')
            if base_model.get("ollama_base"):
                lines.append(f'  ollama_base = "{base_model["ollama_base"]}"')
                lines.append(f'  ollama_tag = "{base_model["ollama_tag"]}"')
    for name, machine in data["machines"].items():
        lines.append(f"[machines.{name}]")
        lines.append(f"reserve_ram_gib = {machine['reserve_ram_gib']}")
        lines.append(f"reserve_vram_gib = {machine['reserve_vram_gib']}")
        lines.append(f"writer = {str(machine['writer']).lower()}")
    lines.append("[paths]")
    lines.append(f'state = "{Path(data["paths"]["state"]).as_posix()}"')
    lines.append(f'markdown = "{Path(data["paths"]["markdown"]).as_posix()}"')
    text = "\n".join(lines) + "\n"
    tomllib.loads(text)  # fail fast if this hand-written TOML is ever malformed
    return text


def _transport():
    return build_transport(qwen35_transport_mapping())


def _llmfit_runner(version_stdout: str = "llmfit 1.1.16\n", system_stdout: str | None = None) -> FixtureRunner:
    if system_stdout is None:
        system_stdout = (FIXTURES / "llmfit_system_laptop.json").read_text(encoding="utf-8")
    return FixtureRunner(
        {
            ("llmfit", "--version"): subprocess.CompletedProcess(["llmfit", "--version"], 0, stdout=version_stdout, stderr=""),
            ("llmfit", "system", "--json"): subprocess.CompletedProcess(
                ["llmfit", "system", "--json"], 0, stdout=system_stdout, stderr=""
            ),
        }
    )


def _ollama_transport(status: int = 200) -> FixtureTransport:
    url = "http://127.0.0.1:11434/api/tags"
    if status == 200:
        return FixtureTransport({("GET", url): json_response("ollama_tags_local.json")})
    return FixtureTransport({("GET", url): Response(status=status, headers={}, body=b"error")})


# --- configuration / writer errors --------------------------------------------------------


def test_fetch_missing_config_file_is_exit_2(tmp_path: Path):
    code = main(["fetch", "--config", str(tmp_path / "missing.toml"), "--machine", "workstation"])
    assert code == 2


def test_fetch_unknown_machine_is_exit_2(tmp_path: Path):
    config_path = _write_config(tmp_path)
    code = main(["fetch", "--config", str(config_path), "--machine", "no-such-machine"], transport=_transport(), now=RUN1)
    assert code == 2


def test_fetch_non_writer_machine_is_exit_2(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data["machines"]["inference-server"] = {"reserve_ram_gib": 16.0, "reserve_vram_gib": 2.0, "writer": False}
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(_to_toml(data), encoding="utf-8")

    code = main(["fetch", "--config", str(config_path), "--machine", "inference-server"], transport=_transport(), now=RUN1)
    assert code == 2


def test_fetch_snapshot_with_unsupported_schema_version_is_exit_3(tmp_path: Path):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "modelroom.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1)
    assert code == 3


# --- F13: the schema-version gate runs before the lock, never touching it -------------------


def test_fetch_snapshot_with_unsupported_schema_version_never_creates_a_lock_file(tmp_path: Path):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "modelroom.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1)

    assert code == 3
    assert not (state_dir / "modelroom.lock").exists()


def test_fetch_snapshot_with_unsupported_schema_version_leaves_a_stale_lock_untouched(tmp_path: Path):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "modelroom.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    lock_path = state_dir / "modelroom.lock"
    stale_content = json.dumps({"pid": 999999, "command": "fetch", "started_at": RUN1.isoformat(), "token": "x"})
    lock_path.write_text(stale_content, encoding="utf-8")

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN2)

    assert code == 3
    assert lock_path.read_text(encoding="utf-8") == stale_content


# --- end-to-end against the fixture transport ----------------------------------------------


def test_fetch_end_to_end_exit_0_writes_a_valid_snapshot_and_run_status(tmp_path: Path):
    config_path = _write_config(tmp_path)

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1)

    assert code == 0
    snapshot_path = tmp_path / "state" / "modelroom.json"
    loaded = load_snapshot(json.loads(snapshot_path.read_text(encoding="utf-8")))
    assert loaded.run_at == RUN1
    assert len(loaded.areas) == 3
    assert all(area.status == "complete" for area in loaded.areas)

    run_status = json.loads((tmp_path / "state" / "run-status.json").read_text(encoding="utf-8"))
    assert len(run_status["areas"]) == 3
    assert run_status["candidates"] == []
    assert not list((tmp_path / "state").glob("*.tmp"))
    assert not (tmp_path / "state" / "modelroom.lock").exists()


def test_fetch_end_to_end_exit_1_when_one_area_is_forced_incomplete(tmp_path: Path):
    config_path = _write_config(tmp_path)
    mapping = qwen35_transport_mapping()
    mapping[("GET", "https://ollama.com/library/qwen3.5/tags")] = Response(status=500, headers={}, body=b"error")
    transport = build_transport(mapping)

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=transport, now=RUN1)

    assert code == 1
    snapshot_path = tmp_path / "state" / "modelroom.json"
    loaded = load_snapshot(json.loads(snapshot_path.read_text(encoding="utf-8")))
    statuses = {area.source: area.status for area in loaded.areas}
    assert statuses["ollama"] == "incomplete"
    assert statuses["huggingface"] == "complete"
    run_status = json.loads((tmp_path / "state" / "run-status.json").read_text(encoding="utf-8"))
    assert len(run_status["areas"]) == 3


# --- version rule at the CLI level -----------------------------------------------------


def test_fetch_a_second_run_older_than_the_stored_snapshot_is_exit_1_and_writes_nothing(tmp_path: Path):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN2) == 0
    snapshot_path = tmp_path / "state" / "modelroom.json"
    before = snapshot_path.read_text(encoding="utf-8")

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1)

    assert code == 1
    assert snapshot_path.read_text(encoding="utf-8") == before


# --- the lock, against a real second process -----------------------------------------------


def test_fetch_stops_at_a_lock_held_by_another_process(tmp_path: Path):
    config_path = _write_config(tmp_path)
    lock_path = tmp_path / "state" / "modelroom.lock"
    holder_script = tmp_path / "lock_holder.py"
    holder_script.write_text(
        "import json, sys, time\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n"
        "path = Path(sys.argv[1])\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_text(json.dumps({'pid': 999999, 'command': 'fetch', "
        "'started_at': datetime.now(timezone.utc).isoformat()}))\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    holder = subprocess.Popen([sys.executable, str(holder_script), str(lock_path)])
    try:
        deadline = time.monotonic() + 5
        while not lock_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert lock_path.exists(), "the holder process never created the lock file"

        code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1)

        assert code == 1
    finally:
        holder.kill()
        holder.wait(timeout=5)


# --- F7: fetch_with_config, the programmatic entry point for an in-memory Configuration ----


def test_fetch_with_config_runs_end_to_end_from_an_in_memory_configuration(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)

    code = fetch_with_config(config, "workstation", transport=_transport(), now=RUN1)

    assert code == 0
    snapshot_path = tmp_path / "state" / "modelroom.json"
    loaded = load_snapshot(json.loads(snapshot_path.read_text(encoding="utf-8")))
    assert loaded.run_at == RUN1
    assert len(loaded.areas) == 3
    assert all(area.status == "complete" for area in loaded.areas)


def test_fetch_with_config_non_writer_machine_is_exit_2(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data["machines"]["inference-server"] = {"reserve_ram_gib": 16.0, "reserve_vram_gib": 2.0, "writer": False}
    config = Configuration.from_dict(data)

    code = fetch_with_config(config, "inference-server", transport=_transport(), now=RUN1)

    assert code == 2


def test_fetch_with_config_defaults_to_the_real_transport_and_clock(tmp_path: Path):
    # No transport/now given -- must fall back exactly like `_cmd_fetch` does, never raise for
    # a missing keyword argument. A non-writer machine short-circuits before either default is
    # ever exercised against the network or the wall clock.
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data["machines"]["laptop"] = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": False}
    config = Configuration.from_dict(data)

    assert fetch_with_config(config, "laptop") == 2


# --- hardware: configuration / machine errors -----------------------------------------------


def test_hardware_missing_config_file_is_exit_2(tmp_path: Path):
    code = main(["hardware", "--config", str(tmp_path / "missing.toml"), "--machine", "workstation"])
    assert code == 2


def test_hardware_unknown_machine_is_exit_2(tmp_path: Path):
    config_path = _write_config(tmp_path)
    code = main(
        ["hardware", "--config", str(config_path), "--machine", "no-such-machine"],
        runner=_llmfit_runner(),
        transport=_ollama_transport(),
        now=RUN1,
    )
    assert code == 2


def test_hardware_does_not_require_a_writer_machine(tmp_path: Path):
    # Unlike fetch, hardware only requires the machine to be configured, not a writer -- it is
    # measured on every machine (CONTRACTS.md).
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data["machines"]["inference-server"] = {"reserve_ram_gib": 16.0, "reserve_vram_gib": 2.0, "writer": False}
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(_to_toml(data), encoding="utf-8")

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "inference-server"],
        runner=_llmfit_runner(),
        transport=_ollama_transport(),
        now=RUN1,
    )
    assert code == 0


# --- hardware: llmfit gate ------------------------------------------------------------------


def test_hardware_llmfit_missing_is_exit_2(tmp_path: Path):
    config_path = _write_config(tmp_path)

    def _missing_runner(args):
        raise FileNotFoundError("llmfit not found")

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=_missing_runner,
        transport=_ollama_transport(),
        now=RUN1,
    )
    assert code == 2


def test_hardware_llmfit_too_old_is_exit_2(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner = _llmfit_runner(version_stdout="llmfit 1.0.0\n")

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=runner,
        transport=_ollama_transport(),
        now=RUN1,
    )
    assert code == 2


# --- F12: llmfit subprocess errors and shape problems are exit 2, not a crash --------------


def test_hardware_llmfit_version_timeout_is_exit_2_and_writes_nothing(tmp_path: Path):
    config_path = _write_config(tmp_path)

    def _timeout_runner(args):
        raise subprocess.TimeoutExpired(cmd=args, timeout=10.0)

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=_timeout_runner,
        transport=_ollama_transport(),
        now=RUN1,
    )

    assert code == 2
    assert not (tmp_path / "state" / "hardware" / "workstation.json").exists()


def test_hardware_llmfit_version_nonzero_returncode_is_exit_2(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner = FixtureRunner(
        {
            ("llmfit", "--version"): subprocess.CompletedProcess(
                ["llmfit", "--version"], returncode=1, stdout="llmfit 1.1.16\n", stderr="boom"
            )
        }
    )

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=runner,
        transport=_ollama_transport(),
        now=RUN1,
    )

    assert code == 2
    assert not (tmp_path / "state" / "hardware" / "workstation.json").exists()


def test_hardware_llmfit_system_total_ram_null_is_exit_2_and_writes_nothing(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner = _llmfit_runner(system_stdout=json.dumps({"system": {"total_ram_gb": None}}))

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=runner,
        transport=_ollama_transport(),
        now=RUN1,
    )

    assert code == 2
    assert not (tmp_path / "state" / "hardware" / "workstation.json").exists()


# --- R7: invalid hardware fields must exit 2, never bypass it as an uncaught pydantic crash --


def test_hardware_llmfit_system_gpu_name_a_list_is_exit_2_and_writes_nothing(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner = _llmfit_runner(
        system_stdout=json.dumps({"system": {"total_ram_gb": 7.56, "gpu_name": []}})
    )

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=runner,
        transport=_ollama_transport(),
        now=RUN1,
    )

    assert code == 2
    assert not (tmp_path / "state" / "hardware" / "workstation.json").exists()


def test_hardware_llmfit_system_total_ram_nan_is_exit_2_and_writes_nothing(tmp_path: Path):
    # json.loads accepts the non-standard "NaN" literal -- a runner can genuinely produce this.
    config_path = _write_config(tmp_path)
    runner = _llmfit_runner(system_stdout='{"system": {"total_ram_gb": NaN}}')

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=runner,
        transport=_ollama_transport(),
        now=RUN1,
    )

    assert code == 2
    assert not (tmp_path / "state" / "hardware" / "workstation.json").exists()


def test_hardware_llmfit_system_unified_memory_a_string_is_exit_2_and_writes_nothing(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner = _llmfit_runner(
        system_stdout=json.dumps({"system": {"total_ram_gb": 7.56, "unified_memory": "yes"}})
    )

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=runner,
        transport=_ollama_transport(),
        now=RUN1,
    )

    assert code == 2
    assert not (tmp_path / "state" / "hardware" / "workstation.json").exists()


# --- hardware: schema-3 path ------------------------------------------------------------


def test_hardware_existing_snapshot_with_unsupported_schema_version_is_exit_3(tmp_path: Path):
    config_path = _write_config(tmp_path)
    hardware_dir = tmp_path / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    (hardware_dir / "workstation.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=_llmfit_runner(),
        transport=_ollama_transport(),
        now=RUN1,
    )
    assert code == 3


# --- hardware: end-to-end against the fixture runner/transport ------------------------------


def test_hardware_end_to_end_exit_0_writes_a_valid_hardware_snapshot(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=_llmfit_runner(),
        transport=_ollama_transport(),
        now=RUN1,
    )

    assert code == 0
    path = tmp_path / "state" / "hardware" / "workstation.json"
    loaded = load_hardware_snapshot(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.machine == "workstation"
    assert loaded.measured_at == RUN1
    assert loaded.llmfit_version == "1.1.16"
    assert loaded.vram_gib == 11.94
    assert loaded.ram_gib == 127.46
    assert loaded.installed is not None
    assert len(loaded.installed) == 3
    assert loaded.installed_unavailable_reason is None
    assert not list((tmp_path / "state" / "hardware").glob("*.tmp"))

    out = capsys.readouterr().out
    assert "workstation" in out
    assert "installed: 3" in out


def test_hardware_end_to_end_ollama_daemon_unreachable_is_still_exit_0(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=_llmfit_runner(),
        transport=_ollama_transport(status=500),
        now=RUN1,
    )

    assert code == 0
    path = tmp_path / "state" / "hardware" / "workstation.json"
    loaded = load_hardware_snapshot(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.installed is None
    assert loaded.installed_unavailable_reason is not None

    out = capsys.readouterr().out
    assert "installed: unknown" in out


def test_hardware_preserves_measurements_from_an_existing_snapshot(tmp_path: Path):
    config_path = _write_config(tmp_path)
    hardware_dir = tmp_path / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    existing = {
        "schema_version": 1,
        "machine": "workstation",
        "measured_at": RUN1.isoformat(),
        "llmfit_version": "1.1.16",
        "vram_gib": 11.94,
        "ram_gib": 127.46,
        "free_ram_gib_at_measurement": 76.64,
        "gpu_name": "Nova GPU",
        "backend": "CUDA",
        "unified_memory": False,
        "installed": None,
        "installed_unavailable_reason": "not queried yet",
        "measurements": [
            {
                "content_source": "ollama",
                "ollama_manifest_digest": "sha256:" + "ab" * 32,
                "hf_repo": None,
                "hf_revision": None,
                "hf_file_digest": None,
                "context": 8192,
                "runtime": "ollama 0.12.3",
                "profile_measured_at": RUN1.isoformat(),
                "measured_at": RUN1.isoformat(),
                "tps_mean": 42.5,
                "tps_range": [40.0, 45.0],
            }
        ],
    }
    (hardware_dir / "workstation.json").write_text(json.dumps(existing), encoding="utf-8")

    code = main(
        ["hardware", "--config", str(config_path), "--machine", "workstation"],
        runner=_llmfit_runner(),
        transport=_ollama_transport(),
        now=RUN2,
    )

    assert code == 0
    loaded = load_hardware_snapshot(json.loads((hardware_dir / "workstation.json").read_text(encoding="utf-8")))
    assert loaded.measured_at == RUN2
    assert len(loaded.measurements) == 1
    assert loaded.measurements[0].ollama_manifest_digest == "sha256:" + "ab" * 32


# --- hardware_with_config: the programmatic entry point for an in-memory Configuration -----


def test_hardware_with_config_runs_end_to_end_from_an_in_memory_configuration(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)

    code = hardware_with_config(config, "workstation", runner=_llmfit_runner(), transport=_ollama_transport(), now=RUN1)

    assert code == 0
    assert (tmp_path / "state" / "hardware" / "workstation.json").exists()


def test_hardware_with_config_unknown_machine_is_exit_2(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)

    code = hardware_with_config(config, "no-such-machine", runner=_llmfit_runner(), transport=_ollama_transport(), now=RUN1)

    assert code == 2
