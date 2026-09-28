"""Rows that look alike: the repository under the table, and pull names no row may claim (AP9-UX2 U4, F7).

Terminal and Markdown view of synthetic documents (`tests/fixture_rows.py`). Nothing is merged and
the ranking rule is untouched (ADR 0009): the difference is only said, under the table.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.daemon import PULL_PATH, FixtureDaemon
from modelroom.dialog import AnswerInvalidError
from modelroom.guided_install import PullContext, pull_step
from modelroom.pull_choice import NOT_UNIQUE, ambiguous_names, pull_rows
from modelroom.views import document_markdown, terminal_lines

from fixture_rows import GIB, document, fillers, package

UNSLOTH = "unsloth/Nova-7B-GGUF"
NAME = "hf.co/unsloth/Nova-7B-GGUF"
Q4 = package(UNSLOTH, "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
Q4_OTHER_REPO = package("unsloth/Nova-7B-Instruct-GGUF", "Nova-7B-Instruct-Q4_K_M.gguf", "Q4_K_M")
BF16_FILE = package(UNSLOTH, "Nova-7B-BF16.gguf", "BF16", 14 * GIB)
BF16_FOLDER = package(UNSLOTH, "BF16/Nova-7B-BF16.gguf", "BF16", 7 * GIB)


def _stripped(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines]


def test_two_repositories_of_one_account_are_named_under_the_table():
    doc = document([Q4, Q4_OTHER_REPO])

    lines = _stripped(terminal_lines(doc, "workstation"))
    assert f"repo      #1  {UNSLOTH}" in lines
    assert "#2  unsloth/Nova-7B-Instruct-GGUF" in lines
    markdown = document_markdown(doc)
    assert f"#1  {UNSLOTH} · #2  unsloth/Nova-7B-Instruct-GGUF" in markdown


def test_two_file_forms_with_one_pull_name_are_named_and_say_so():
    doc = document([BF16_FILE, BF16_FOLDER, Q4])
    ambiguous = ambiguous_names([BF16_FILE, BF16_FOLDER, Q4])

    lines = _stripped(terminal_lines(doc, "workstation", ambiguous=ambiguous))

    assert f"repo      #1  {UNSLOTH}  (single file)" in lines
    assert f"#2  {UNSLOTH}  BF16/ (folder)" in lines
    shown = next(line for line in lines if line.startswith("shown"))
    assert "#1 has no unique pull name: 2 files answer to it" in " ".join(lines)
    assert "#2 has no unique pull name: 2 files answer to it" in " ".join(lines)
    assert shown  # the pieces stand behind the count
    assert "install   #1 has no command: its pull name is not unique" in lines


def test_a_pull_name_shared_with_a_hidden_package_at_place_11_is_not_unique():
    """The counterpart is not in the table (place 11): the row is still locked and has no command."""
    shown = [BF16_FILE, *fillers(9, avoid=("BF16",))]
    everything = [*shown, BF16_FOLDER]
    doc = document(shown, total=11)
    ambiguous = ambiguous_names(everything)

    rows = pull_rows(doc, "workstation", ambiguous)
    lines = _stripped(terminal_lines(doc, "workstation", ambiguous=ambiguous))

    assert rows[0].locked == NOT_UNIQUE and all(row.locked is None for row in rows[1:])
    assert not any(line.startswith("repo") for line in lines)  # the counterpart is not shown
    assert not any("ollama pull" in line for line in lines)


def test_two_file_forms_of_different_size_lock_both(tmp_path: Path):
    """Different weights behind one name: neither row can be pulled, no disk check, no POST."""
    from test_pull_choice import _Recording, _daemon, _run

    packages = [BF16_FILE, BF16_FOLDER]
    context = PullContext(document(packages), "workstation", packages, ambiguous_names(packages))
    daemon = _daemon()
    run, lines = _run(tmp_path, _Recording({"pull_package": f"{NAME}:BF16", "pull": True}), daemon)

    assert [row.locked for row in pull_rows(context.document, "workstation", context.ambiguous)] == [NOT_UNIQUE] * 2
    with pytest.raises(AnswerInvalidError):
        pull_step(run, context, True)
    assert not any(call[1] == PULL_PATH for call in daemon.calls)
    assert not any("disk" in line for line in lines)


def test_no_line_of_a_table_is_wider_than_100_columns_but_repositories_and_commands():
    """A long model name is cut in its column; `9999.9 tok/s` fills `Speed` exactly (U8 widths)."""
    long_base = "acme/Nova-7B"
    rows = [Q4, Q4_OTHER_REPO, BF16_FILE]
    doc = document(rows)
    block = doc.machines[0]
    measured = block.ranked[0].model_copy(
        update={"base_model_hf_repo": long_base, "measurement_group": 0, "measurement_id": "m" * 8, "speed_tps": 9999.9}
    )
    doc = doc.model_copy(update={"machines": [block.model_copy(update={"ranked": [measured, *block.ranked[1:]]})]})

    lines = terminal_lines(doc, "workstation")

    assert any("9999.9 tok/s" in line for line in lines)
    for line in lines:
        if line.strip().startswith(("repo", "#2  unsloth/", "install")):
            continue  # a repository and a command are as long as they are
        assert len(line) <= 100, line


@pytest.mark.parametrize("model", ["acme/" + "Nova-" * 12 + "7B"])
def test_a_long_model_name_is_cut_in_its_column(model):
    doc = document([package(UNSLOTH, "Nova-7B-Q4_K_M.gguf", "Q4_K_M", base_model_hf_repo=model)])
    row = next(line for line in terminal_lines(doc, "workstation") if line.strip().startswith("1 "))
    assert len(row) <= 100
