"""AP9-KO P1, fix round 1: Hugging Face ids are compared without case (`contracts.hf_id_key`), and `modelroom pull`
validates the name before it reads anything.

The setup note of a document is keyed by the id of its rows; a document read back may write that id in another
case than the rows do. The note must reach all four views, and it is shown with the id as the note wrote it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelroom.catalog_findings import SETUP_NOTE, SetupNote
from modelroom.dialog import FileAsker
from modelroom.guided_install import PullContext, pull_step
from modelroom.pull_choice import ambiguous_names
from modelroom.screen import plain
from modelroom.views import RunView, document_markdown, terminal_blocks

from test_golden_vllm import BASE_REPO
from test_install_vllm import REPO, SERVE, _both, _gguf_package, _package, _run

WRITTEN = BASE_REPO.upper().replace("QWEN/", "qwen/")  # same repository, another case
REVISION = "d" * 40


def _document():
    note = SetupNote.model_validate(
        {
            "model": WRITTEN, "status": "instructions", "card_revision": REVISION, "checked": "2026-10-13",
            "source_url": f"https://huggingface.co/{WRITTEN}/blob/{REVISION}/README.md", "package": None, "reason": None,
        }
    )
    return _both().model_copy(update={"setup_notes": {WRITTEN: note}}), SETUP_NOTE.format(model=WRITTEN, url=note.source_url)


def test_the_note_reaches_all_four_views_when_the_document_writes_the_id_in_another_case():
    assert WRITTEN != BASE_REPO
    document, sentence = _document()
    packages = [_gguf_package(), _package()]

    table = "\n".join(plain(line) for line in terminal_blocks(document, view=RunView(install_for="server", packages=packages)))
    assert sentence in table and table.index(sentence) < table.index("vllm serve")
    assert sentence in document_markdown(document)


def test_the_choice_in_step_5_and_the_handover_carry_it_too(tmp_path: Path):
    from modelroom.pull_other import pull_other_step

    document, sentence = _document()
    packages = [_gguf_package(), _package()]
    run, lines = _run(tmp_path / "a", FileAsker({"pull_package": f"vllm {REPO}"}))
    pull_step(run, PullContext(document, "server", packages, ambiguous_names(packages)), True)
    assert sentence in "\n".join(lines)

    run, lines = _run(tmp_path / "b", FileAsker({"pull_other": f"server vllm {REPO}"}))
    pull_other_step(run, PullContext(document, "laptop", packages, ambiguous_names(packages)))
    assert sentence in "\n".join(lines)


def test_an_invalid_name_is_refused_before_the_catalog_is_read(monkeypatch, tmp_path: Path, capsys):
    from modelroom import cli

    reads: list[object] = []
    monkeypatch.setattr(cli, "offline_catalog", lambda *args, **kwargs: reads.append(args) or pytest.fail("the catalog was read"))

    assert cli.main(["pull", "a;b"], pointer_path=tmp_path / "guided.json") == 2
    assert reads == [] and "a;b" in capsys.readouterr().err
