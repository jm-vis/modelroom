"""Tests for modelroom.guided_lmstudio: the start row, step 4 and the card, with the server not measured.

The functions are not registered in `runtimes.RUNTIMES` (that is AP9-LS-E): they are called directly, with
a stand-in for the run (`fixture_lmstudio.run_of`: the screen's notes and the run's `ServerRun`) and the
recordings of LM Studio (`tests/fixtures/lmstudio_*.json`). Every test that could send a chat counts the
chats its fixture server was sent, and every count is 0: the measuring switch stands on its negative
branch (decided 2026-10-08; ground: a chat to the id of an unloaded model proves neither a refusal nor a
load, A-LS7), so there is no offer, no row, no chat and no number.
"""

from __future__ import annotations

import copy
import json

import pytest

from modelroom.http import Response
from modelroom.intro import Fact
from modelroom.lmstudio_server import KEY_REASON, MODELS_PATH, FixtureLmStudioServer, LmStudioModels, LoadedModel
from modelroom.guided_lmstudio import ROW, card_facts, measure, offer, start_fact
from modelroom.loadtest_lmstudio import NOT_MEASURED_NOTE
from modelroom.measurements import Scenario
from modelroom.runtimes import Offer, Origins

from fixture_http import FIXTURES
from fixture_lmstudio import chat_calls, list_calls, lmstudio_servers, run_of, trap_processes

SCENARIO = Scenario(context_requested=8192, context_origin="default", kv_type="f16", kv_type_assumed=True, requests=1)
LLM_KEY = "qwen3.5-4b"


def _recorded(name: str = "lmstudio_v1_models_loaded.json") -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _llm(payload: dict) -> dict:
    return next(entry for entry in payload["models"] if entry["type"] == "llm")


def _instance(name: str, context: int = 8192) -> dict:
    return {"id": name, "config": {"context_length": context, "parallel": 4}}


def _step_four(run, results_dir, picked: list[str]):
    """The three calls step 4 makes of the server, in order: the offer, then the measurement of what was picked."""
    given = offer(run, None, SCENARIO, results_dir)
    return given, measure(run, None, SCENARIO, results_dir, given, picked)


def _started(server):
    run = run_of(server)
    return run, start_fact(Origins(daemon=None, servers=run.servers))


# --- the start row ----------------------------------------------------------------------------------------


def test_the_start_row_names_the_loaded_model_and_context():
    server = lmstudio_servers(["lmstudio_v1_models_loaded.json"])

    run, fact = _started(server)

    assert fact == Fact("server", "LM Studio", f"1 loaded: {LLM_KEY}, context 8192")
    assert run.servers.found[ROW] == LmStudioModels(
        loaded=(LoadedModel(LLM_KEY, LLM_KEY, 8192, 4),), downloaded=1, keys=frozenset({LLM_KEY, "text-embedding-nomic-embed-text-v1.5"})
    )
    assert [call[:2] for call in server.calls] == [("GET", MODELS_PATH)]


def test_the_start_row_counts_several_loaded_models():
    payload = _recorded()
    second = copy.deepcopy(_llm(payload))
    second.update(key="other-model", loaded_instances=[_instance("other-model", 4096)])
    payload["models"].append(second)

    _run, fact = _started(lmstudio_servers([payload]))

    assert fact == Fact("server", "LM Studio", "2 models loaded")


@pytest.mark.parametrize(
    "recording",
    ["lmstudio_jit_on_after.json", "lmstudio_v1_models_empty.json", "lmstudio_v1_models_embedding_only.json"],
)
def test_a_server_with_no_loaded_model_has_a_row_that_says_zero(recording):
    _run, fact = _started(lmstudio_servers([recording]))

    assert fact == Fact("server", "LM Studio", "0 loaded")


def test_no_start_row_without_a_server():
    silent = FixtureLmStudioServer({}, unreachable="refused")
    foreign = FixtureLmStudioServer({("GET", MODELS_PATH): Response(status=200, body=b'{"data": []}')})
    broken = FixtureLmStudioServer({("GET", MODELS_PATH): Response(status=500, body=b"{}")})

    for server in (None, silent, foreign, broken):
        run, fact = _started(server)
        assert fact is None
        assert run.servers.found == {}


def test_a_key_asking_server_is_one_start_row_and_no_offer():
    server = FixtureLmStudioServer({("GET", MODELS_PATH): Response(status=401, body=b"{}")})
    run, fact = _started(server)

    given, said = _step_four(run, None, [ROW])

    assert fact == Fact("server", "127.0.0.1:1234", "asks for an API token")
    assert run.servers.found[ROW] == KEY_REASON
    assert given == Offer() and said is None
    assert run.notes == [KEY_REASON]
    assert chat_calls(server) == [] and len(list_calls(server)) == 1


# --- step 4: no offer, no row, no chat ---------------------------------------------------------------------


def test_a_loaded_model_gets_the_note_and_no_offer_and_no_chat():
    server = lmstudio_servers(["lmstudio_v1_models_loaded.json"])
    run, _fact = _started(server)

    given, said = _step_four(run, None, [ROW])

    assert given == Offer(rows=(), unavailable=None, nothing=None, state=None)
    assert said is None
    assert run.notes == [NOT_MEASURED_NOTE]
    assert chat_calls(server) == []
    assert len(list_calls(server)) == 1


def test_no_loaded_instance_offers_nothing_and_sends_no_chat():
    """No instance, an answers file that names `lmstudio` and a yes to the load test: still no chat."""
    server = lmstudio_servers(["lmstudio_jit_on_after.json"])
    run, _fact = _started(server)

    given, said = _step_four(run, None, [ROW, f"{ROW}:0"])

    assert given.rows == () and said is None
    assert chat_calls(server) == []
    assert len(list_calls(server)) == 1


def test_two_loaded_gguf_instances_offer_no_row_and_no_chat():
    """No row `lmstudio:0`, `lmstudio:1` is built: two models loaded, one of them twice (the second instance
    under an identifier of its own), an MLX and an embedding instance alike get no row and no chat."""
    payload = _recorded()
    first, second = _llm(payload), copy.deepcopy(_llm(payload))
    second.update(key="other-model", loaded_instances=[_instance("other-model", 4096)])
    mlx = copy.deepcopy(second)
    mlx.update(key="mlx-model", format="mlx", loaded_instances=[_instance("mlx-model")])
    payload["models"][1]["loaded_instances"] = [_instance("text-embedding-nomic-embed-text-v1.5", 2048)]
    payload["models"] += [second, mlx]
    first["loaded_instances"].append(_instance("mr-probe"))
    server = lmstudio_servers([payload])
    run, fact = _started(server)

    given, said = _step_four(run, None, [f"{ROW}:0", f"{ROW}:1"])

    assert fact == Fact("server", "LM Studio", "3 models loaded")
    assert given.rows == () and said is None
    assert chat_calls(server) == []


@pytest.mark.parametrize(
    ("after", "instance"),
    [
        ("lmstudio_jit_on_after.json", LLM_KEY),
        ("lmstudio_jit_on_identifier_after.json", "mr-probe"),
    ],
    ids=["by-model-key", "by-identifier"],
)
def test_jit_on_only_model_loaded_then_unloaded_sends_no_chat(after, instance):
    """The regression of the recorded case: the only downloaded model is loaded, the list says so, and it
    is unloaded before a chat could go out. The recorded answer to such a chat is a 400; no chat is sent,
    and the list read after the unloading is never even asked for."""
    loaded = _recorded()
    _llm(loaded)["loaded_instances"] = [_instance(instance)]
    server = lmstudio_servers([loaded, after])
    run, fact = _started(server)

    given, said = _step_four(run, None, [ROW])

    assert fact == Fact("server", "LM Studio", f"1 loaded: {instance}, context 8192")
    assert given.rows == () and said is None
    assert chat_calls(server) == []
    assert [call for call in server.calls if call[0] != "GET"] == []  # no POST of any kind
    assert len(list_calls(server)) == 1  # the unloaded state was never read: nothing asked for it
    assert run.notes == [NOT_MEASURED_NOTE]


def test_the_note_is_said_once_for_a_server_that_answered_and_not_at_all_for_one_that_did_not():
    answered = run_of(lmstudio_servers(["lmstudio_v1_models_loaded.json"]))
    answered_fact = start_fact(Origins(daemon=None, servers=answered.servers))
    silent = run_of(FixtureLmStudioServer({}, unreachable="refused"))
    start_fact(Origins(daemon=None, servers=silent.servers))

    offer(answered, None, SCENARIO, None)
    offer(silent, None, SCENARIO, None)

    assert answered_fact is not None
    assert answered.notes == [NOT_MEASURED_NOTE]
    assert silent.notes == []


def test_nothing_is_written_nothing_is_ranked_and_nothing_is_kept(tmp_path):
    """No number exists, so no measurement file, no profile and no snapshot: the results folder is as it was,
    and the run keeps no outcome for the card."""
    results = tmp_path / "results"
    results.mkdir()
    server = lmstudio_servers(["lmstudio_v1_models_loaded.json"])
    run, _fact = _started(server)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    _step_four(run, results, [ROW])

    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == before
    assert run.servers.measured == {}


def test_the_card_carries_no_lmstudio_row():
    run, _fact = _started(lmstudio_servers(["lmstudio_v1_models_loaded.json"]))
    _step_four(run, None, [ROW])

    assert card_facts(run) == ()
    assert card_facts(run_of(None)) == ()


def test_step_four_reads_no_model_folder_and_starts_no_process(tmp_path, monkeypatch):
    """The model folder holds the very file of the loaded model and a program `lms` stands in the home
    folder: nothing is looked at there (no stat, no listing) and no process is started."""
    from pathlib import Path

    folder = tmp_path / "home" / ".lmstudio" / "models" / "unsloth" / "Qwen3.5-4B-GGUF"
    folder.mkdir(parents=True)
    (folder / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"x" * 16)
    server = lmstudio_servers(["lmstudio_v1_models_loaded.json"])

    def refuse(*args, **kwargs):
        pytest.fail("the model folder was touched", pytrace=False)

    with monkeypatch.context() as patched:
        trap_processes(patched)
        patched.setattr(Path, "stat", refuse)
        patched.setattr(Path, "iterdir", refuse)
        patched.setattr(Path, "open", refuse)
        run, fact = _started(server)
        given, said = _step_four(run, None, [ROW])

    assert fact is not None and given.rows == () and said is None
