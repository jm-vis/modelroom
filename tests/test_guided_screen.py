"""The whole screen of one guided run, held against a golden file -- the mockup as a test.

`modelroom --answers <file>` prints the same lines in the same order as a terminal run, without
color, and that is what a log of a run is read from. So the screen has a fixture of its own
(`tests/golden/guided-screen*.txt`): a change to it is a change to this file, made on purpose.

Everything variable is replaced before the comparison -- the results folder by `<results>`, a
profile id by `<profile>`, a measured speed by `<speed>` -- because a golden file may not depend
on which machine ran the test. Nothing outside `tmp_path` is written: `HOME`/`USERPROFILE` are
never read here, the pointer file lives in `tmp_path`, and every source is a fixture.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.dialog import FileAsker
from modelroom.guided import run_guided

from fixture_support import (
    DEEPSEEK_OLLAMA_NAME,
    build_transport,
    guided_transport_mapping,
    loadtest_daemon,
    offline_daemon,
    windows_probes,
)

GOLDEN = Path(__file__).parent / "golden"
RUN_AT = datetime(2026, 9, 25, 8, 0, 0, tzinfo=timezone.utc)
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen/Qwen3.5-9B"

CHOSEN = {
    "results": "here",
    "machines": ["this-machine"],
    "search": "qwen",
    "filter_owners": True,
    "select": [QWEN, DEEPSEEK],
    "context": "L",
}
NOTHING_CHOSEN = {**CHOSEN, "select": []}
# The level `S` for the run that measures: the fixture daemon loads 8192, so any other context
# would make the measurement not comparable and no row would carry a speed.
MEASURED = {**CHOSEN, "context": "S", "load_test": True, "load_test_packages": [DEEPSEEK_OLLAMA_NAME]}


def _screen(tmp_path: Path, answers: dict, *, daemon=None) -> str:
    """Every line one `--answers` run prints, with what varies per machine masked out."""
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    lines: list[str] = []
    run_guided(
        FileAsker(answers),
        here=results,
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(host="workstation", ids=["a" * 16]),
        daemon=daemon if daemon is not None else offline_daemon(),
        now=RUN_AT,
        out=lines.append,
        colored=False,
    )
    return _masked("\n".join(lines), results) + "\n"


def _masked(text: str, results: Path) -> str:
    """Replace what a golden file may not depend on: the folder, an id, a measured speed."""
    text = text.replace(str(results), "<results>")
    text = re.sub(r"\b[0-9a-f]{16}\b", "<profile>", text)
    text = re.sub(r"\d+\.\d tok/s \(\d+\.\d–\d+\.\d\)", "<speed>", text)
    return re.sub(r"\d+\.\d tok/s", "<speed>", text)


def _compare(name: str, screen: str) -> None:
    golden = GOLDEN / name
    if not golden.is_file():  # pragma: no cover - only while a new golden file is written
        pytest.fail(f"{golden} does not exist; write it from the run and read it before committing")
    assert screen == golden.read_text(encoding="utf-8")


def test_the_screen_of_a_run_with_two_models_reads_as_the_golden_file(tmp_path: Path):
    """Two models chosen, context `L`, no measurement -- the run the mockup of 2026-09-24 draws."""
    _compare("guided-screen.txt", _screen(tmp_path, CHOSEN))


def test_the_screen_of_a_run_that_chose_nothing_carries_a_dash(tmp_path: Path):
    _compare("guided-screen-nothing.txt", _screen(tmp_path, NOTHING_CHOSEN))


def test_the_screen_of_a_run_that_measured_carries_the_speed(tmp_path: Path):
    _compare("guided-screen-measured.txt", _screen(tmp_path, MEASURED, daemon=loadtest_daemon()))


def test_no_line_of_the_screen_is_wider_than_a_hundred_characters(tmp_path: Path):
    """The head is 72, the table 100; nothing the run draws may wrap in a 100-column window."""
    for line in _screen(tmp_path, CHOSEN).splitlines():
        if "<results>" in line or "Hugging Face" in line or "no load test" in line:
            continue  # a path and a registry's own message are as long as they are
        assert len(line) <= 100, line
