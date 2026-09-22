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

from modelroom.cli import fetch_with_config, main
from modelroom.config import Configuration
from modelroom.contracts import load_snapshot
from modelroom.http import Response

from fixture_support import build_transport, qwen35_example_config_dict, qwen35_transport_mapping

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
