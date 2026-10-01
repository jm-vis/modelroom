"""Tests for modelroom.guided_search: what step 2 says about a search, and the candidate list.

All pure: a `search.json` reading in, a sentence out. The case these tests exist for is the one the
search of 2026-09-24 had no sentence for -- a publisher of the catalog was asked and answered with
nothing, which read "at no publisher of this catalog that answered" and left the reader guessing
which account that was (decided 2026-09-25).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from modelroom.catalog import Catalog, load_catalog
from modelroom.guided_context import Checked
from modelroom.guided_search import (
    DID_YOU_MEAN_KEY,
    KEEP_THE_WORD,
    KEEP_THE_WORD_LABEL,
    NO_WORD_ANSWER,
    _tensor_hits,
    candidate_choices,
    no_gguf_clause,
    search_answer,
    search_notes,
    searched_sentence,
)
from modelroom.http import FixtureTransport, RequestBudget, Response


def _log(accounts: list[dict]) -> dict:
    return {"schema_version": 2, "word": "deepseek", "mode": "word", "accounts": accounts}


def _account(name: str, klass: str, hits: int, page_full: bool = False) -> dict:
    return {"account": name, "class": klass, "hits": hits, "page_full": page_full}


PACKAGERS = ["unsloth", "bartowski", "mradermacher", "lmstudio-community", "ggml-org"]


def _packager_entries(hits: int = 0) -> list[dict]:
    return [_account(name, "packager", hits) for name in PACKAGERS]


# --- where the search asked ----------------------------------------------------------------------


def test_a_publisher_that_answered_is_named_as_before():
    log = _log([_account("Qwen", "publisher", 1), *_packager_entries()])

    assert searched_sentence(log) == "searched Hugging Face at the publisher Qwen and the five listed packagers"


def test_a_publisher_that_was_asked_and_answered_with_nothing_is_named_with_its_reason():
    log = _log([_account("deepseek-ai", "publisher", 0), *_packager_entries()])

    assert searched_sentence(log) == (
        "searched Hugging Face at the five listed packagers; "
        "the publisher deepseek-ai has no GGUF repository for this search"
    )


def test_two_publishers_that_answered_with_nothing_are_both_named():
    log = _log(
        [_account("Qwen", "publisher", 0), _account("deepseek-ai", "publisher", 0), *_packager_entries()]
    )

    assert searched_sentence(log).endswith("the publishers Qwen and deepseek-ai have no GGUF repository for this search")


def test_the_open_lists_are_named_in_that_sentence_too():
    log = _log(
        [
            _account("deepseek-ai", "publisher", 0),
            *_packager_entries(),
            _account("most downloaded", "open", 10),
            _account("newest", "open", 10),
        ]
    )

    assert searched_sentence(log) == (
        "searched Hugging Face at the five listed packagers, and the two open lists; "
        "the publisher deepseek-ai has no GGUF repository for this search"
    )


def test_a_word_no_publisher_matches_keeps_the_sentence_it_always_had():
    # Nothing to name: no publisher of the catalog was asked at all.
    log = _log(_packager_entries())

    assert searched_sentence(log) == (
        "searched Hugging Face at no publisher of this catalog that answered and the five listed packagers"
    )


def test_an_account_named_twice_in_the_log_is_still_counted_once():
    # The sentence counts **accounts**, not entries: a log that held one entry per page -- which is
    # what the search asks since 2026-09-25 -- must not read "the ten listed packagers".
    log = _log([_account("Qwen", "publisher", 2), *_packager_entries(2), *_packager_entries(0)])

    assert [entry["account"] for entry in log["accounts"]].count("unsloth") == 2
    assert "five listed packagers" in searched_sentence(log)


def test_the_two_notes_are_the_sentence_and_the_two_numbers():
    log = _log([_account("Qwen", "publisher", 2), *_packager_entries()])

    notes = search_notes(log, 4, 2)

    assert notes[0] == searched_sentence(log)
    assert notes[1] == "4 repositories, 2 of them models you can pick from"


def test_the_second_note_of_a_search_with_one_repository_counts_it_once():
    log = _log([_account("Qwen", "publisher", 1)])

    assert search_notes(log, 1, 1)[1] == "1 repository, a model you can pick from"
    assert search_notes(log, 1, 0)[1] == "1 repository, not a model you can pick from"


def test_a_full_account_page_reaches_the_second_note():
    log = _log([_account("unsloth", "packager", 20, page_full=True)])

    assert search_notes(log, 20, 3, dot="-")[1].endswith("- a more specific word shortens the list")


def test_a_typed_repository_id_gets_a_sentence_about_that_repository():
    # The account sentence read "at no publisher of this catalog that answered and the no listed
    # packagers, and the two open lists" for a run of one request (live probe, 2026-09-25).
    log = {
        "mode": "id",
        "word": "unsloth/Qwen3.5-9B-GGUF",
        "accounts": [{"account": "typed", "class": "typed", "hits": 1, "page_full": False}],
    }

    assert searched_sentence(log) == "searched Hugging Face for the repository unsloth/Qwen3.5-9B-GGUF"


def test_a_run_with_no_word_gets_a_sentence_about_the_catalog():
    log = {
        "mode": "catalog",
        "word": "",
        "accounts": [
            {"account": "Qwen3.8-27B", "class": "catalog", "hits": 10, "page_full": False},
            {"account": "gemma-4-31B-it", "class": "catalog", "hits": 0, "page_full": False},
        ],
    }

    assert searched_sentence(log) == "searched Hugging Face for the 2 current models of this catalog"


def test_neither_of_those_two_claims_an_open_list_or_counts_packagers():
    for mode, word in (("id", "a/b"), ("catalog", "")):
        log = {"mode": mode, "word": word, "accounts": [{"account": "x", "class": mode, "hits": 1, "page_full": False}]}
        sentence = searched_sentence(log)
        assert "open lists" not in sentence
        assert "listed packagers" not in sentence


def test_the_clause_names_one_publisher_or_several():
    assert no_gguf_clause(["Qwen"]) == "the publisher Qwen has no GGUF repository for this search"
    assert no_gguf_clause(["a", "b", "c"]) == "the publishers a, b and c have no GGUF repository for this search"


# --- the answer line of the search question -------------------------------------------------------


def test_the_word_is_the_answer_line_and_nothing_typed_says_what_it_means():
    assert search_answer(" qwen ") == "qwen"
    assert search_answer("") == NO_WORD_ANSWER == "anything that fits this machine"


# --- the candidates -------------------------------------------------------------------------------


def test_the_candidates_are_a_list_with_none_of_these_last_and_the_first_under_the_pointer():
    choices = candidate_choices(["qwen", "qwen3.5"])

    assert [choice.value for choice in choices] == ["qwen", "qwen3.5", KEEP_THE_WORD]
    assert [choice.checked for choice in choices] == [True, False, False]
    assert choices[-1].label == KEEP_THE_WORD_LABEL
    assert all(choice.disabled is None for choice in choices)


def test_the_key_of_that_question_is_the_one_the_answer_file_uses():
    assert DID_YOU_MEAN_KEY == "did_you_mean"


# --- the vLLM tensor search of step 2 (AP9-SV2-P, P1) ----------------------------------------------


@dataclass
class _FakeScreen:
    notes: list[str] = field(default_factory=list)

    def note(self, line: str) -> None:
        self.notes.append(line)


@dataclass
class _FakeRun:
    transport: Any
    budget: Any
    catalog: Catalog
    now: datetime = datetime(2026, 9, 30, 9, 0, 0, tzinfo=timezone.utc)
    screen: _FakeScreen = field(default_factory=_FakeScreen)
    problems: list[str] = field(default_factory=list)
    # `runtimes.machine_has_vllm`'s other source (fix round 0c): a server detected serving in
    # this very run's own step 1. Empty by default, like a run that never saw one.
    servers: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(found={}))

    def failed(self, message: str) -> None:
        self.problems.append(message)


def test_no_vllm_machine_asks_no_tensor_search():
    """P1: a machine without vLLM (no `checked.profile`, or one whose `runtimes` do not name it)
    costs no extra request at all -- an ordinary Ollama run and its goldens are unchanged."""
    hit = SimpleNamespace(resolved_base_model="acme/Nova-7B")

    def _unreachable(method: str, url: str, headers=None):
        raise AssertionError(f"no request should be made for a machine without vLLM: {method} {url}")

    run = _FakeRun(transport=_unreachable, budget=None, catalog=load_catalog())

    repos = _tensor_hits(run, Checked(), [hit])

    assert repos == {}
    assert run.problems == []
    assert run.screen.notes == []


def test_a_vllm_machine_finds_the_tensor_repos_a_base_model_has():
    """A machine whose profile names `vllm` among its `runtimes` is searched, once per distinct
    resolved base model of this very search; what is found is returned, not written -- the write
    happens later, through the same selection-and-write path a GGUF hit goes through
    (`guided.py::_search_step`, `search_apply.apply_hits`/`apply_tensor_repos`)."""
    from modelroom.tensor_search import tensor_search_url

    hit = SimpleNamespace(resolved_base_model="acme/Nova-7B")

    transport = FixtureTransport(
        {
            ("GET", tensor_search_url("Nova-7B", "acme")): _tensor_page(
                [
                    {
                        "id": "RedHatAI/Nova-7B-FP8",
                        "tags": ["base_model:acme/Nova-7B", "base_model:quantized:acme/Nova-7B"],
                    }
                ]
            ),
            ("GET", tensor_search_url("Nova-7B", "RedHatAI")): _tensor_page([]),
        }
    )
    run = _FakeRun(transport=transport, budget=RequestBudget(50), catalog=load_catalog())
    checked = Checked(profile=SimpleNamespace(runtimes=["ollama", "vllm"]))

    repos = _tensor_hits(run, checked, [hit, hit])  # two hits of the same base model: still one search

    assert len(transport.calls) == 2  # one account each, never once per hit
    assert repos == {"acme/Nova-7B": ["RedHatAI/Nova-7B-FP8"]}
    assert run.problems == []


def test_a_live_detected_server_with_no_stored_runtimes_also_searches():
    """Fix round 0c: the second of P1's two sources ("runtimes, or detected live") -- a machine
    whose profile names no `vllm` at all (or has none) still runs the tensor search when *this*
    run's own step 1 found a vLLM server actually serving a model, through the one function
    `runtimes.machine_has_vllm` (`test_runtimes.py` proves the function itself; this is the
    red-first case the fix round asked for: no stored `runtimes` is the whole point)."""
    from modelroom.tensor_search import tensor_search_url
    from modelroom.vllm_server import ServedModel

    hit = SimpleNamespace(resolved_base_model="acme/Nova-7B")

    transport = FixtureTransport(
        {
            ("GET", tensor_search_url("Nova-7B", "acme")): _tensor_page(
                [
                    {
                        "id": "RedHatAI/Nova-7B-FP8",
                        "tags": ["base_model:acme/Nova-7B", "base_model:quantized:acme/Nova-7B"],
                    }
                ]
            ),
            ("GET", tensor_search_url("Nova-7B", "RedHatAI")): _tensor_page([]),
        }
    )
    run = _FakeRun(transport=transport, budget=RequestBudget(50), catalog=load_catalog())
    run.servers.found["vllm"] = ServedModel(id="acme/Nova-7B", root="acme/Nova-7B", max_model_len=4096)
    checked = Checked(profile=None)  # no bound profile at all, never mind a stored runtimes list

    repos = _tensor_hits(run, checked, [hit])

    assert len(transport.calls) == 2
    assert repos == {"acme/Nova-7B": ["RedHatAI/Nova-7B-FP8"]}


def test_a_key_prompt_alone_does_not_count_as_detected():
    """E5: a server that asks for a key is not counted as vLLM detected -- neither source then
    says so, so the search asks nothing (red-first against the naive "any entry in `found`
    means vLLM" reading)."""
    hit = SimpleNamespace(resolved_base_model="acme/Nova-7B")

    def _unreachable(method: str, url: str, headers=None):
        raise AssertionError(f"no request should be made: {method} {url}")

    run = _FakeRun(transport=_unreachable, budget=None, catalog=load_catalog())
    run.servers.found["vllm"] = "asks for an API key"
    checked = Checked(profile=None)

    repos = _tensor_hits(run, checked, [hit])

    assert repos == {}


def _tensor_page(hits: list[dict]) -> Response:
    return Response(status=200, headers={}, body=json.dumps(hits).encode("utf-8"))
