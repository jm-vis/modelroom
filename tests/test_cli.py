"""End-to-end tests for modelroom.cli: `modelroom fetch --config ... --machine ...`.

`main()` takes optional `transport`/`now` keyword arguments precisely so these tests never
touch the network or the wall clock -- the real CLI entry point (`transport=None`) uses
`UrllibTransport()` and the real clock, exercised nowhere in this suite.
"""

from __future__ import annotations

import json
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
from modelroom.render import RatingUnavailableError

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


def test_render_hardware_with_unsupported_schema_version_is_exit_3(tmp_path: Path):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    hardware_dir = tmp_path / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    (hardware_dir / "workstation.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 3
    assert not (tmp_path / "models.md").exists()


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


def test_render_hardware_with_truncated_json_is_exit_3_and_names_the_file(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    hardware_dir = tmp_path / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    hardware_path = hardware_dir / "workstation.json"
    hardware_path.write_text('{"schema_version": 1, "machine": ', encoding="utf-8")  # truncated

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 3
    assert not (tmp_path / "models.md").exists()
    assert str(hardware_path) in capsys.readouterr().err


@pytest.mark.parametrize("payload", _MALFORMED_STATE_PAYLOADS)
def test_render_hardware_with_malformed_shape_is_exit_3(tmp_path: Path, capsys, payload: bytes):
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    hardware_dir = tmp_path / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    hardware_path = hardware_dir / "workstation.json"
    hardware_path.write_bytes(payload)

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 3
    assert not (tmp_path / "models.md").exists()
    assert str(hardware_path) in capsys.readouterr().err


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


def test_render_with_config_defaults_to_the_real_clock(tmp_path: Path):
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)
    # No snapshot at all -> exit 1 regardless of the clock; the real-clock default path is
    # exercised without ever depending on its actual value.
    assert render_with_config(config) == 1
