"""End-to-end tests for modelroom.cli: `modelroom fetch --config ... --machine ...`.

`main()` takes optional `transport`/`now` keyword arguments precisely so these tests never
touch the network or the wall clock -- the real CLI entry point (`transport=None`) uses
`UrllibTransport()` and the real clock, exercised nowhere in this suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.cli import fetch_with_config, main, render_with_config
from modelroom.config import Configuration
from modelroom.contracts import load_snapshot
from modelroom.http import Response
from modelroom.importer import scan_profiles
from modelroom.measurements import MeasurementRecord, PackageRef, RunCounters, Scenario
from modelroom.render_cmd import _machine_profiles, _measurements_of
from modelroom.render import RatingUnavailableError

from fixture_support import (
    build_transport,
    place_v2_profile,
    qwen35_example_config_dict,
    qwen35_transport_mapping,
    with_profile,
)

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


# --- P2-3 (fix-round 5): a corrupt or wrong-shape state file is exit 3, not an uncaught crash -


def test_fetch_snapshot_with_truncated_json_is_exit_3_and_names_the_file(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    snapshot_path = state_dir / "modelroom.json"
    snapshot_path.write_text('{"schema_version": 1, "run_at": ', encoding="utf-8")  # truncated

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1)

    assert code == 3
    assert str(snapshot_path) in capsys.readouterr().err


def test_fetch_snapshot_with_wrong_shape_is_exit_3(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    snapshot_path = state_dir / "modelroom.json"
    snapshot_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")  # missing run_at etc.

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1)

    assert code == 3
    assert str(snapshot_path) in capsys.readouterr().err


# --- R7-3 (fix-round 6): a structurally-broken state file (non-object root, or undecodable
# bytes) is exit 3, not an uncaught AttributeError/UnicodeDecodeError. `[]` and `null` both
# reach `state.load_snapshot`'s `data.get("schema_version")` -- a list/`None` has no `.get`.

_MALFORMED_STATE_PAYLOADS = [
    pytest.param(b"[]", id="list-root"),
    pytest.param(b"null", id="null-root"),
    pytest.param(b'"just a string"', id="string-root"),
    pytest.param(b"\xff\xfe{", id="invalid-utf8"),
]


@pytest.mark.parametrize("payload", _MALFORMED_STATE_PAYLOADS)
def test_fetch_snapshot_with_malformed_shape_is_exit_3_and_names_the_file(tmp_path: Path, capsys, payload: bytes):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    snapshot_path = state_dir / "modelroom.json"
    snapshot_path.write_bytes(payload)

    code = main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1)

    assert code == 3
    assert str(snapshot_path) in capsys.readouterr().err


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
    # Fix-round 3: the lock file is a stable kernel-lock target, never deleted -- release_lock
    # only empties it (see modelroom/state.py, "Lock file").
    lock_path = tmp_path / "state" / "modelroom.lock"
    assert lock_path.exists()
    assert lock_path.read_bytes() == b""


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
    # P3-11 (fix-round 5): the stale run still acquired and released the lock (the schema-version
    # gate runs before it, but the staleness check runs inside it, F13) -- the file itself is
    # never deleted (CONTRACTS.md, "Lock file"), but release_lock always empties it.
    assert (tmp_path / "state" / "modelroom.lock").read_bytes() == b""


# --- the lock, against a real second process -----------------------------------------------


def test_fetch_stops_at_a_lock_held_by_another_process(tmp_path: Path):
    # Fix-round 3: the lock is a real kernel lock now (see modelroom/state.py), so the holder
    # must take it through acquire_lock itself -- a plain file write, as the previous version of
    # this test used, would no longer block anything.
    config_path = _write_config(tmp_path)
    lock_path = tmp_path / "state" / "modelroom.lock"
    holder_script = tmp_path / "lock_holder.py"
    holder_script.write_text(
        "import sys, time\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n"
        "from modelroom.state import acquire_lock\n"
        "path = Path(sys.argv[1])\n"
        "acquire_lock(path, 'fetch', datetime.now(timezone.utc))\n"
        "print('holding', flush=True)\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    holder = subprocess.Popen(
        [sys.executable, str(holder_script), str(lock_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "holding", "the holder process never acquired the lock"

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


def test_fetch_with_config_accepts_omitted_transport_and_now_without_raising(tmp_path: Path):
    """P3-13 (fix-round 5, renamed): this only proves `fetch_with_config(config, machine)` --
    `transport`/`now` both omitted -- never raises `TypeError` for a missing keyword argument,
    exactly like `_cmd_fetch` calls it. It does **not** prove the real defaults
    (`UrllibTransport()`, `datetime.now(timezone.utc)`) ever run: a non-writer machine returns
    exit `2` before either default is even constructed, and this suite must never let a real
    `UrllibTransport()` make a live network call in the first place (AGENTS.md, "Network-facing
    code is tested against recorded fixtures, never against the live API"), so there is no way
    to exercise that default's actual network behavior from this test suite at all -- only that
    the call site itself is valid Python.
    """
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data["machines"]["laptop"] = {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": False}
    config = Configuration.from_dict(data)

    assert fetch_with_config(config, "laptop") == 2


# --- render: schema-version gates, lock, no-snapshot, newer-document refusal ----------------


def test_render_no_snapshot_is_exit_1_and_writes_nothing(tmp_path: Path):
    config_path = _write_config(tmp_path)

    code = main(["render", "--config", str(config_path)], now=RUN1)

    assert code == 1
    assert not (tmp_path / "models.md").exists()
    # R7-12 (fix-round 6): "nothing to render" must be decided BEFORE the lock is ever taken --
    # `state.acquire_lock` creates the state directory and the lock file as a side effect of
    # opening it, so a render that never gets past "no snapshot" must never call it at all.
    assert not (tmp_path / "state").exists()


def test_render_snapshot_with_unsupported_schema_version_is_exit_3(tmp_path: Path):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "modelroom.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    code = main(["render", "--config", str(config_path)], now=RUN1)

    assert code == 3
    assert not (state_dir / "modelroom.lock").exists()
    assert not (tmp_path / "models.md").exists()


def test_render_hardware_with_unsupported_schema_version_is_a_note_not_a_failure(tmp_path: Path, capsys):
    """Schema 2: one unreadable profile file no longer keeps every machine out of the document.

    `scan_profiles` lists a file that does not read, does not validate or carries an unsupported
    `schema_version`; the machine is shown with that reason and the render still writes both
    views. Only the snapshot's own `schema_version` still ends the run with exit 3.
    """
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    hardware_dir = tmp_path / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    (hardware_dir / "workstation.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 0
    assert "skipped workstation.json" in capsys.readouterr().out
    assert "no_profile" in (tmp_path / "models.md").read_text(encoding="utf-8")


# --- P2-3 (fix-round 5): a corrupt or wrong-shape snapshot/hardware file is exit 3 ----------


def test_render_snapshot_with_truncated_json_is_exit_3_and_names_the_file(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    snapshot_path = state_dir / "modelroom.json"
    snapshot_path.write_text('{"schema_version": 1, "run_at": ', encoding="utf-8")  # truncated

    code = main(["render", "--config", str(config_path)], now=RUN1)

    assert code == 3
    assert not (tmp_path / "models.md").exists()
    assert str(snapshot_path) in capsys.readouterr().err


def test_render_snapshot_with_wrong_shape_is_exit_3(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    snapshot_path = state_dir / "modelroom.json"
    snapshot_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")  # missing run_at etc.

    code = main(["render", "--config", str(config_path)], now=RUN1)

    assert code == 3
    assert not (tmp_path / "models.md").exists()
    assert str(snapshot_path) in capsys.readouterr().err


@pytest.mark.parametrize("payload", _MALFORMED_STATE_PAYLOADS)
def test_render_snapshot_with_malformed_shape_is_exit_3(tmp_path: Path, capsys, payload: bytes):
    config_path = _write_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    snapshot_path = state_dir / "modelroom.json"
    snapshot_path.write_bytes(payload)

    code = main(["render", "--config", str(config_path)], now=RUN1)

    assert code == 3
    assert not (tmp_path / "models.md").exists()
    assert str(snapshot_path) in capsys.readouterr().err


@pytest.mark.parametrize("payload", [b'{"schema_version": 1, "machine": ', *_MALFORMED_STATE_PAYLOADS])
def test_render_names_every_hardware_file_that_does_not_read(tmp_path: Path, capsys, payload: bytes):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    hardware_dir = tmp_path / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    hardware_path = hardware_dir / "workstation.json"
    hardware_path.write_bytes(payload)

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 0
    assert f"skipped {hardware_path.name}" in capsys.readouterr().out


def test_render_end_to_end_exit_0_writes_a_markdown_document(tmp_path: Path):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 0
    markdown_path = tmp_path / "models.md"
    text = markdown_path.read_text(encoding="utf-8")
    assert text.startswith("<!-- modelroom render: snapshot_run_at=")
    assert "# Model packages" in text
    assert not list((tmp_path / "state").glob("*.tmp"))
    assert not list(markdown_path.parent.glob("*.tmp"))
    # The lock is the same stable kernel-lock file `fetch` already created -- render only takes
    # and releases it, never deletes it (modelroom/state.py, "Lock file").
    lock_path = tmp_path / "state" / "modelroom.lock"
    assert lock_path.exists()
    assert lock_path.read_bytes() == b""


def test_render_stops_at_a_lock_held_by_another_process(tmp_path: Path):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    lock_path = tmp_path / "state" / "modelroom.lock"
    holder_script = tmp_path / "render_lock_holder.py"
    holder_script.write_text(
        "import sys, time\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n"
        "from modelroom.state import acquire_lock\n"
        "path = Path(sys.argv[1])\n"
        "acquire_lock(path, 'render', datetime.now(timezone.utc))\n"
        "print('holding', flush=True)\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    holder = subprocess.Popen(
        [sys.executable, str(holder_script), str(lock_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "holding", "the holder process never acquired the lock"

        code = main(["render", "--config", str(config_path)], now=RUN2)

        assert code == 1
        assert not (tmp_path / "models.md").exists()
    finally:
        holder.kill()
        holder.wait(timeout=5)


def test_render_refuses_when_the_existing_document_is_newer(tmp_path: Path):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN2) == 0
    assert main(["render", "--config", str(config_path)], now=RUN2) == 0
    markdown_path = tmp_path / "models.md"
    before = markdown_path.read_text(encoding="utf-8")

    # Simulate a snapshot older than the one already rendered underneath the existing document.
    state_dir = tmp_path / "state"
    old_snapshot = json.loads((state_dir / "modelroom.json").read_text(encoding="utf-8"))
    old_snapshot["run_at"] = RUN1.isoformat()
    (state_dir / "modelroom.json").write_text(json.dumps(old_snapshot), encoding="utf-8")

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 1
    assert markdown_path.read_text(encoding="utf-8") == before


def test_render_replaces_an_older_existing_document(tmp_path: Path):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    assert main(["render", "--config", str(config_path)], now=RUN1) == 0

    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN2) == 0
    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 0
    text = (tmp_path / "models.md").read_text(encoding="utf-8")
    assert f"snapshot_run_at={RUN2.isoformat()}" in text


def test_render_replaces_an_existing_document_that_is_not_valid_utf8(tmp_path: Path):
    """F4 (fix-round 5, own finding at the AP5 acceptance): an undecodable existing document
    counts as no header, exactly like a missing/empty one -- `render` must replace it, never
    raise `UnicodeDecodeError` out of `_refusal_against_existing_document`.
    """
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    markdown_path = tmp_path / "models.md"
    markdown_path.write_bytes(b"\xff\xfe not valid utf-8 \x80\x81")

    code = main(["render", "--config", str(config_path)], now=RUN1)

    assert code == 0
    text = markdown_path.read_text(encoding="utf-8")
    assert f"snapshot_run_at={RUN1.isoformat()}" in text


# --- render_with_config: the programmatic entry point, and the Rating source ----------------


def test_render_with_config_runs_end_to_end(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)
    assert fetch_with_config(config, "workstation", transport=_transport(), now=RUN1) == 0

    code = render_with_config(config, now=RUN2)

    assert code == 0
    assert (tmp_path / "models.md").exists()


def test_render_with_config_rating_source_failure_is_exit_0_with_a_note(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)
    assert fetch_with_config(config, "workstation", transport=_transport(), now=RUN1) == 0
    # The Stars column only exists where rows exist, so this machine needs a profile the fit
    # computes for -- without one the rating source is never asked at all.
    profile = place_v2_profile(config.paths.hardware_dir)
    config = with_profile(config, "workstation", profile.profile_id)

    def _failing(repo: str):
        raise RatingUnavailableError("market index unreachable")

    code = render_with_config(config, rating=_failing, now=RUN2)

    assert code == 0
    text = (tmp_path / "models.md").read_text(encoding="utf-8")
    assert "Market rating unavailable: market index unreachable" in text


def test_render_with_config_no_source_given_has_no_rating_note(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)
    assert fetch_with_config(config, "workstation", transport=_transport(), now=RUN1) == 0

    code = render_with_config(config, now=RUN2)

    assert code == 0
    text = (tmp_path / "models.md").read_text(encoding="utf-8")
    assert "Market rating unavailable" not in text


# --- render on schema 2: the ranking, the JSON view next to the Markdown one -----------------


def _rendered(tmp_path: Path, *, machines: dict | None = None, **render_kwargs):
    """Fetch from the fixtures, then render; returns `(code, config)`."""
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    if machines is not None:
        data["machines"] = machines
    config = Configuration.from_dict(data)
    writer = next(name for name, machine in config.machines.items() if machine.writer)
    assert fetch_with_config(config, writer, transport=_transport(), now=RUN1) == 0
    return config


def test_render_ranks_the_packages_of_a_schema_two_profile(tmp_path: Path):
    config = _rendered(tmp_path)
    profile = place_v2_profile(config.paths.hardware_dir)
    config = with_profile(config, "workstation", profile.profile_id)

    assert render_with_config(config, now=RUN2) == 0

    text = (tmp_path / "models.md").read_text(encoding="utf-8")
    assert "## Ranking: workstation" in text
    assert "Scenario: context 8192 (default), KV cache f16 (assumed), 1 request" in text
    assert "Ranking rule: fit class (perfect, good, marginal)" in text


def test_render_writes_the_json_view_next_to_the_markdown_one(tmp_path: Path):
    config = _rendered(tmp_path)
    profile = place_v2_profile(config.paths.hardware_dir)
    config = with_profile(config, "workstation", profile.profile_id)

    assert render_with_config(config, now=RUN2) == 0

    payload = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert datetime.fromisoformat(payload["rendered_at"]) == RUN2
    assert [block["machine"] for block in payload["machines"]] == ["workstation"]
    assert payload["ranking_rule"] in (tmp_path / "models.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("command", ["render", "fetch"])
def test_a_snapshot_path_that_is_a_folder_is_exit_3_not_a_traceback(tmp_path: Path, capsys, command: str):
    """`load_existing_snapshot` checks `exists()` and then reads: a folder raised `OSError`.

    Found by the second-model review of AP9-C and true of this reader since AP3; the contract
    promises a message and exit `3` for every stored file that does not read.
    """
    config_path = _write_config(tmp_path)
    (tmp_path / "state" / "modelroom.json").mkdir(parents=True)
    argv = [command, "--config", str(config_path)]
    if command == "fetch":
        argv += ["--machine", "workstation"]

    code = main(argv, transport=_transport(), now=RUN1)

    assert code == 3
    assert "cannot read snapshot" in capsys.readouterr().err


def test_render_refuses_a_markdown_path_that_is_itself_a_json_file(tmp_path: Path, capsys):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.json"))
    config = Configuration.from_dict(data)

    assert render_with_config(config, now=RUN2) == 2
    assert "must not be a .json file" in capsys.readouterr().err


def test_render_shows_a_schema_one_profile_as_legacy_and_persists_nothing(tmp_path: Path):
    config = _rendered(tmp_path)
    legacy_path = config.paths.hardware_dir / "workstation.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(_v1_profile("workstation")), encoding="utf-8")
    before = legacy_path.read_bytes()

    assert render_with_config(config, now=RUN2) == 0

    text = (tmp_path / "models.md").read_text(encoding="utf-8")
    assert "workstation.json -- legacy" in text
    assert "measure again" in text
    assert legacy_path.read_bytes() == before
    assert sorted(path.name for path in config.paths.hardware_dir.glob("*.json")) == ["workstation.json"]


def test_render_reads_the_measurement_files_of_the_profile(tmp_path: Path):
    """The speed comes from `<state>/measurements/<profile_id>/`, never from the profile file.

    Asserted on what the reader returns, not on the rendered document: the fixture snapshot is
    Qwen3.5, which fit v1 does not cover, so an empty ranking would also hold if the file had
    never been read (found by the second-model review).
    """
    config = _rendered(tmp_path)
    profile = place_v2_profile(config.paths.hardware_dir)
    config = with_profile(config, "workstation", profile.profile_id)
    ollama = next(p for p in load_snapshot(json.loads(config.paths.snapshot_file.read_text(encoding="utf-8"))).packages
                  if p.source == "ollama")
    record = _ollama_measurement(profile.profile_id, ollama.manifest_digest)
    (config.paths.state / "measurements" / profile.profile_id).mkdir(parents=True)
    (config.paths.state / "measurements" / profile.profile_id / f"{record.measurement_id}.json").write_text(
        json.dumps(record.model_dump(mode="json")), encoding="utf-8"
    )

    profiles = _machine_profiles(config, scan_profiles(config.paths.hardware_dir))
    records, notes = _measurements_of(config, profiles)

    assert notes == []
    assert [entry.measurement_id for entry in records[profile.profile_id]] == [record.measurement_id]
    assert records[profile.profile_id][0].tps_mean == pytest.approx(50.0)
    assert render_with_config(config, now=RUN2) == 0


def test_render_names_a_measurement_file_that_does_not_read(tmp_path: Path, capsys):
    config = _rendered(tmp_path)
    profile = place_v2_profile(config.paths.hardware_dir)
    config = with_profile(config, "workstation", profile.profile_id)
    folder = config.paths.state / "measurements" / profile.profile_id
    folder.mkdir(parents=True)
    (folder / "20260922T202000Z-4da3b585.json").write_text("{", encoding="utf-8")

    assert render_with_config(config, now=RUN2) == 0
    assert "skipped measurement 20260922T202000Z-4da3b585.json" in capsys.readouterr().out


def _ollama_measurement(profile_id: str, manifest_digest: str) -> MeasurementRecord:
    run = RunCounters(
        done_reason="length",
        eval_count=128,
        eval_duration=2_560_000_000,
        prompt_eval_count=42,
        prompt_eval_duration=90_000_000,
        load_duration=12_000_000,
    )
    measured_at = datetime(2026, 9, 22, 20, 20, 0, tzinfo=timezone.utc)
    return MeasurementRecord(
        schema_version=2,
        measurement_id=f"{measured_at.strftime('%Y%m%dT%H%M%SZ')}-4da3b585",
        profile_id=profile_id,
        protocol="v1",
        measured_at=measured_at,
        package=PackageRef(content_source="ollama", ollama_manifest_digest=manifest_digest),
        scenario=Scenario(
            context_requested=8192, context_origin="default", kv_type="f16", kv_type_assumed=True, requests=1
        ),
        runs=[run, run, run],
        tps_mean=run.tokens_per_second(),
        tps_min=run.tokens_per_second(),
        tps_max=run.tokens_per_second(),
        validity="valid",
        validity_reason=None,
        comparable=True,
        comparable_reason=None,
    )


def test_render_echoes_the_terminal_view_when_a_caller_asks_for_it(tmp_path: Path):
    config = _rendered(tmp_path)
    profile = place_v2_profile(config.paths.hardware_dir)
    config = with_profile(config, "workstation", profile.profile_id)
    lines: list[str] = []

    assert render_with_config(config, now=RUN2, echo=lines.append) == 0

    assert "workstation · context" in "\n".join(lines)


def test_render_hands_the_document_itself_to_a_caller_that_asks_for_it(tmp_path: Path):
    """The guided mode draws the card of its step 5 from the document the render just wrote."""
    config = _rendered(tmp_path)
    profile = place_v2_profile(config.paths.hardware_dir)
    config = with_profile(config, "workstation", profile.profile_id)
    seen: list = []

    assert render_with_config(config, now=RUN2, on_document=seen.append) == 0

    assert [block.machine for block in seen[0].machines] == ["workstation"]


def _v1_profile(machine: str) -> dict:
    return {
        "schema_version": 1,
        "machine": machine,
        "measured_at": RUN1.isoformat(),
        "llmfit_version": "1.1.16",
        "vram_gib": 11.94,
        "ram_gib": 127.46,
        "free_ram_gib_at_measurement": None,
        "gpu_name": "Nova GPU",
        "backend": "CUDA",
        "unified_memory": False,
        "installed": None,
        "installed_unavailable_reason": "not queried in this test",
        "measurements": [],
    }


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_prints_the_package_version_without_a_terminal_and_without_the_pointer_file(tmp_path: Path, flag: str):
    """A real child process, stdin from the null device: the flag needs no terminal and no home.

    Testers typed `modelroom --version` after the install and got an argparse error (2026-09-25).
    """
    import modelroom

    home = tmp_path / "child-home"
    # A pointer file that does not read: any look at it would end the run with exit 3.
    pointer = home / ".modelroom" / "guided.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{ not a pointer file", encoding="utf-8")
    program = "import sys; from modelroom.cli import main; sys.exit(main())"
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}
    with open(os.devnull) as devnull:
        finished = subprocess.run(
            [sys.executable, "-c", program, flag],
            stdin=devnull,
            capture_output=True,
            cwd=tmp_path,
            env=env,
            text=True,
            encoding="utf-8",
        )

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout == f"modelroom {modelroom.__version__}\n"
    assert finished.stderr == ""
    assert sorted(home.rglob("*")) == [pointer.parent, pointer]
    assert pointer.read_text(encoding="utf-8") == "{ not a pointer file"


def test_version_in_process_exits_0_with_the_package_version(capsys):
    import modelroom

    with pytest.raises(SystemExit) as stopped:
        main(["--version"])

    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == f"modelroom {modelroom.__version__}"


def test_the_help_names_the_version_flag_next_to_answers(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])

    help_text = capsys.readouterr().out
    assert "-V, --version" in help_text
    assert "--answers" in help_text


def test_render_with_config_defaults_to_the_real_clock(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)
    # No snapshot at all -> exit 1 regardless of the clock; the real-clock default path is
    # exercised without ever depending on its actual value.
    assert render_with_config(config) == 1
