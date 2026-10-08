"""Tests for modelroom.loadtest_lmstudio: LM Studio is not measured (the negative branch, decided 2026-10-08).

The measuring switch stands on "negative branch" because the recording of a chat to the id of an unloaded
model proves neither a refusal nor a load (`tests/fixtures/README.md`). So there is no load test to run:
this module holds the one sentence that says so and the rule that picks it, and the tests prove the
absence -- no module of LM Studio names the chat call -- next to the behavior tests in
`tests/test_guided_lmstudio.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from modelroom.lmstudio_server import KEY_REASON, LmStudioModels
from modelroom.loadtest_lmstudio import NOT_MEASURED_NOTE, step_note

PACKAGE = Path(__file__).resolve().parent.parent / "modelroom"
MEASURING_MODULES = ("loadtest_lmstudio", "guided_lmstudio")
CHAT_NAMES = ("CHAT_PATH", "chat_body", "run_vllm_load_test", "loadtest_vllm")


def test_the_note_names_the_reason_in_the_words_of_the_contract():
    assert NOT_MEASURED_NOTE == "LM Studio is not measured: it may load a model on request"


def test_a_server_that_answered_gets_the_note_a_key_asking_one_its_reason_and_none_nothing():
    answered = LmStudioModels(loaded=(), downloaded=0, keys=frozenset())

    assert step_note(answered) == NOT_MEASURED_NOTE
    assert step_note(KEY_REASON) == KEY_REASON
    assert step_note(None) is None


@pytest.mark.parametrize("name", MEASURING_MODULES)
def test_no_measuring_module_names_the_chat_call(name):
    """The transport has no chat call (`ALLOWED_CALLS` is one `GET`), and this is the second guard: no caller
    of one exists. The two modules neither import a chat path nor the body of a load test, and write no chat
    path, so a transport that later gains the call finds no caller here."""
    tree = ast.parse((PACKAGE / f"{name}.py").read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    names |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    assert not names & set(CHAT_NAMES), names & set(CHAT_NAMES)
    assert not any("/chat" in text for text in literals)
    assert "POST" not in literals
