"""The link to model quality (AP9-T-L): one line, under the result of step 5, never fetched.

`modelroom/tempo_text.py` holds the sentences the speed, design and memory lines of AP9-T say;
this package adds the link. The line is text for a reader -- not a request: modelroom never
opens the address (CONTRACTS.md, "Render (schema 2)"; README, "Network access").
"""

from __future__ import annotations

import socket
import urllib.request
from pathlib import Path

import pytest

from modelroom.cli import main
from modelroom.render import document_json
from modelroom.tempo_text import QUALITY_LINK_LINE
from modelroom.views import document_markdown, document_terminal, terminal_lines

from fixture_rows import document, document_multi, package
from test_cli import RUN2
from test_cli_render import _fetched
from test_guided_screen import CHOSEN, NOTHING_CHOSEN, _screen

LINK_TEXT = "Quality of these models for your task: https://artificialanalysis.ai"
ONE = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
OTHER = package("unsloth/Nova-7B-Instruct-GGUF", "Nova-7B-Instruct-Q4_K_M.gguf", "Q4_K_M")


def test_the_quality_link_line_reads_as_decided():
    assert QUALITY_LINK_LINE == LINK_TEXT
    assert "\n" not in QUALITY_LINK_LINE and QUALITY_LINK_LINE == QUALITY_LINK_LINE.strip()


def test_the_quality_link_follows_the_last_block():
    lines = terminal_lines(document([ONE, OTHER]), "workstation")

    assert lines[-1] == f" {QUALITY_LINK_LINE}"  # indented like the other lines of the screen (`_INDENT`)
    assert sum(QUALITY_LINK_LINE in line for line in lines) == 1


def test_the_quality_link_stands_once_behind_the_last_of_several_blocks():
    lines = terminal_lines(document_multi({"workstation": [ONE], "studio": [OTHER]}), "workstation")

    assert lines[-1] == f" {QUALITY_LINK_LINE}"
    assert sum(QUALITY_LINK_LINE in line for line in lines) == 1
    assert lines.index(f" {QUALITY_LINK_LINE}") > max(i for i, line in enumerate(lines) if "studio" in line)


def test_the_terminal_view_ends_with_the_quality_link():
    assert document_terminal(document([ONE])).splitlines()[-1] == f" {QUALITY_LINK_LINE}"


def test_no_quality_link_without_a_ranked_row():
    empty = document([])

    assert not any(QUALITY_LINK_LINE in line for line in terminal_lines(empty, "workstation"))
    assert QUALITY_LINK_LINE not in document_terminal(empty)


def test_the_quality_link_is_not_in_the_markdown_nor_the_json_view():
    doc = document([ONE])

    assert "artificialanalysis" not in document_markdown(doc)
    assert "artificialanalysis" not in document_json(doc)


def test_the_guided_run_shows_the_link_once(tmp_path: Path):
    screen = _screen(tmp_path, CHOSEN)

    assert screen.count(QUALITY_LINK_LINE) == 1
    lines = screen.splitlines()
    assert lines.index(f" {QUALITY_LINK_LINE}") < max(i for i, line in enumerate(lines) if "Results" in line)


def test_a_guided_run_that_chose_nothing_shows_no_link(tmp_path: Path):
    assert QUALITY_LINK_LINE not in _screen(tmp_path, NOTHING_CHOSEN)


def test_no_request_to_the_quality_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """The address is text: a document and a `modelroom render` with every connection refused still show it."""
    config_path = _fetched(tmp_path)  # the snapshot first (a fake transport); the refusals then cover the CLI path too
    capsys.readouterr()

    def refuse(*args, **kwargs):
        raise AssertionError("a test opened a connection to the quality site")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)

    assert QUALITY_LINK_LINE in document_terminal(document([ONE]))
    assert main(["render", "--config", str(config_path)], now=RUN2) == 0
    assert capsys.readouterr().out == QUALITY_LINK_LINE + "\n"
