"""What `modelroom render` prints on stdout (AP9-T-L): its notes on unreadable profiles, then the quality link.

Only the CLI handler prints; `render_with_config` stays silent, because the guided mode calls it
itself and draws step 5 itself (so the line never stands twice). Kept out of `test_cli.py`, which
does not grow past 800 lines.
"""

from __future__ import annotations

import json
from pathlib import Path

from modelroom.cli import fetch_with_config, main, render_with_config
from modelroom.config import Configuration
from modelroom.tempo_text import QUALITY_LINK_LINE

from fixture_support import qwen35_example_config_dict
from test_cli import RUN1, RUN2, _transport, _write_config


def _fetched(tmp_path: Path) -> Path:
    config_path = _write_config(tmp_path)
    assert main(["fetch", "--config", str(config_path), "--machine", "workstation"], transport=_transport(), now=RUN1) == 0
    return config_path


def test_modelroom_render_prints_the_quality_link_on_success(tmp_path: Path, capsys):
    config_path = _fetched(tmp_path)
    capsys.readouterr()

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 0
    assert capsys.readouterr().out == QUALITY_LINK_LINE + "\n"


def test_modelroom_render_orders_note_then_link(tmp_path: Path, capsys):
    config_path = _fetched(tmp_path)
    hardware_dir = tmp_path / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    (hardware_dir / "workstation.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    capsys.readouterr()

    code = main(["render", "--config", str(config_path)], now=RUN2)

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("note: skipped workstation.json")
    assert lines[-1] == QUALITY_LINK_LINE
    assert len(lines) == 2


def test_modelroom_render_prints_no_link_on_failure(tmp_path: Path, capsys):
    config_path = _write_config(tmp_path)

    assert main(["render", "--config", str(config_path)], now=RUN1) == 1  # no snapshot
    assert QUALITY_LINK_LINE not in capsys.readouterr().out

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "modelroom.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    assert main(["render", "--config", str(config_path)], now=RUN1) == 3  # a schema this version does not read
    assert QUALITY_LINK_LINE not in capsys.readouterr().out


def test_modelroom_render_prints_no_link_for_an_invalid_configuration(tmp_path: Path, capsys):
    assert main(["render", "--config", str(tmp_path / "missing.toml")], now=RUN1) == 2
    assert QUALITY_LINK_LINE not in capsys.readouterr().out


def test_render_with_config_stays_silent_on_success(tmp_path: Path, capsys):
    config = Configuration.from_dict(qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md")))
    assert fetch_with_config(config, "workstation", transport=_transport(), now=RUN1) == 0
    capsys.readouterr()

    assert render_with_config(config, now=RUN2) == 0

    assert capsys.readouterr().out == ""
