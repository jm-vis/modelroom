"""Tests for modelroom.guided_lmstudio: the start row, the row of step 4, its measurement and the card (AP9-LS-M).

The four functions are called directly, with a stand-in for the run (`fixture_lmstudio.run_of`: the screen's
notes, the run's `ServerRun` and its failures) and the recordings of LM Studio (`tests/fixtures/lmstudio_*.json`).
A row is offered for exactly one loaded GGUF instance; the measurement is one request at a time (four chats and
five reads against the fixture server), and a number reaches the card only when nothing in the table of the
contract ended the measurement. Every test that could send a chat counts the chats its fixture server got.
"""

from __future__ import annotations

import copy
import json

import pytest

from modelroom import guided_lmstudio, loadtest_lmstudio
from modelroom.dialog import Choice
from modelroom.http import Response
from modelroom.intro import UNICODE_GLYPHS, Fact
from modelroom.lmstudio_server import (
    KEY_REASON,
    MODELS_PATH,
    FixtureLmStudioServer,
    LmStudioModels,
    LmStudioServerError,
    LmStudioTimeoutError,
    LoadedModel,
)
from modelroom.guided_lmstudio import (
    MEASURED,
    NO_MODEL_NOTE,
    OTHER_CONTEXT_NOTE,
    ROW,
    SEVERAL_NOTE,
    card_facts,
    measure,
    offer,
    start_fact,
)
from modelroom.loadtest import LoadTestError
from modelroom.loadtest_lmstudio import LmStudioMeasurement
from modelroom.measurements import Scenario, shipped_protocol
from modelroom.runtimes import Offer, Origins
from modelroom.screen import card_lines, plain

from fixture_http import FIXTURES
from fixture_lmstudio import chat_calls, chat_to, list_calls, lmstudio_servers, run_of, trap_processes

SCENARIO = Scenario(context_requested=8192, context_origin="default", kv_type="f16", kv_type_assumed=True, requests=1)
LLM_KEY = "qwen3.5-4b"
SECOND = 1_000_000_000
DASH = UNICODE_GLYPHS.skip
LOADED = "lmstudio_v1_models_loaded.json"


def _recorded(name: str = LOADED) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _llm(payload: dict) -> dict:
    return next(entry for entry in payload["models"] if entry["type"] == "llm")


def _instance(name: str, context: int = 8192) -> dict:
    return {"id": name, "config": {"context_length": context, "parallel": 4}}


def _step_four(run, results_dir, picked: list[str], scenario: Scenario = SCENARIO):
    """The two calls step 4 makes of the server, in order: the offer, then the measurement of what was picked."""
    given = offer(run, None, scenario, results_dir)
    return given, measure(run, None, scenario, results_dir, given, picked)


def _started(server):
    run = run_of(server)
    return run, start_fact(Origins(daemon=None, servers=run.servers))


def _chats(*replies) -> list:
    """The four chats of a measurement: `replies` first, then the good answer for the rest."""
    return list(replies) + [chat_to(LLM_KEY)] * (4 - len(replies))


def _server(*replies, lists: list | None = None):
    return lmstudio_servers(lists or [LOADED], chats=_chats(*replies))


@pytest.fixture
def steady(monkeypatch):
    """Every chat takes 2 s on the clock the writer gets, with a long stretch between two chats: 64.0 tok/s."""
    real = loadtest_lmstudio.run_lmstudio_load_test

    def with_clock(server, instance, scenario, prompt):
        state = {"n": 0}

        def clock() -> int:
            n, state["n"] = state["n"], state["n"] + 1
            return (n // 2) * 100 * SECOND + (n % 2) * 2 * SECOND

        return real(server, instance, scenario, prompt, clock=clock)

    monkeypatch.setattr(guided_lmstudio, "run_lmstudio_load_test", with_clock)


# --- the start row ----------------------------------------------------------------------------------------


def test_the_start_row_names_the_loaded_model_and_context():
    server = lmstudio_servers([LOADED])

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


# --- step 4: the row, the notes, no chat before the answer -------------------------------------------------------


def test_a_key_asking_server_is_one_start_row_and_no_offer_and_no_note():
    server = FixtureLmStudioServer({("GET", MODELS_PATH): Response(status=401, body=b"{}")})
    run, fact = _started(server)

    given, said = _step_four(run, None, [ROW])

    assert fact == Fact("server", "127.0.0.1:1234", "asks for an API token")
    assert run.servers.found[ROW] == KEY_REASON
    assert given == Offer() and said is None
    assert run.notes == []  # the start row says it
    assert chat_calls(server) == [] and len(list_calls(server)) == 1


def test_one_loaded_model_is_offered_as_a_row_without_a_chat_and_without_a_note():
    server = _server()
    run, _fact = _started(server)

    given = offer(run, None, SCENARIO, None)

    assert given.rows == (Choice(ROW, f"LM Studio server  {LLM_KEY}, context 8192"),)
    assert isinstance(given.state, LmStudioModels) and given.unavailable is None and given.nothing is None
    assert run.notes == []
    assert chat_calls(server) == [] and len(list_calls(server)) == 1


def test_no_loaded_instance_offers_no_row_says_so_and_sends_no_chat():
    """No instance, an answers file that names `lmstudio` and a yes to the load test: still no chat."""
    server = lmstudio_servers(["lmstudio_jit_on_after.json"])
    run, _fact = _started(server)

    given, said = _step_four(run, None, [ROW, f"{ROW}:0"])

    assert given.rows == () and said is None
    assert run.notes == [NO_MODEL_NOTE] == ["LM Studio: no model loaded, nothing to measure"]
    assert chat_calls(server) == [] and len(list_calls(server)) == 1


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

    given, said = _step_four(run, None, [f"{ROW}:0", f"{ROW}:1", ROW])

    assert fact == Fact("server", "LM Studio", "3 models loaded")
    assert given.rows == () and said is None
    assert run.notes == [SEVERAL_NOTE.format(count=3)] == ["LM Studio: 3 models loaded, measure with one loaded model"]
    assert chat_calls(server) == []


@pytest.mark.parametrize(
    ("after", "instance"),
    [("lmstudio_jit_on_after.json", LLM_KEY), ("lmstudio_jit_on_identifier_after.json", "mr-probe")],
    ids=["by-model-key", "by-identifier"],
)
def test_a_model_unloaded_before_the_first_read_of_the_measurement_ends_it_without_a_chat(after, instance):
    """The regression of the recorded case: the only downloaded model is loaded at the start, and unloaded
    before the measurement reads the list again. The server is never asked to chat; the row of the table says
    the loaded model changed."""
    loaded = _recorded()
    _llm(loaded)["loaded_instances"] = [_instance(instance)]
    server = lmstudio_servers([loaded, after])
    run, fact = _started(server)

    given, said = _step_four(run, None, [ROW])

    assert fact == Fact("server", "LM Studio", f"1 loaded: {instance}, context 8192")
    assert len(given.rows) == 1 and said is None
    assert chat_calls(server) == [] and [call for call in server.calls if call[0] != "GET"] == []
    assert len(list_calls(server)) == 2  # the start row and the one read that showed the model gone
    assert run.notes == ["LM Studio: the loaded model changed during the measurement"]
    assert run.servers.measured == {} and card_facts(run) == ()


def test_the_note_of_a_server_with_no_model_is_said_once_and_a_silent_server_says_nothing():
    answered = run_of(lmstudio_servers(["lmstudio_v1_models_empty.json"]))
    start_fact(Origins(daemon=None, servers=answered.servers))
    silent = run_of(FixtureLmStudioServer({}, unreachable="refused"))
    start_fact(Origins(daemon=None, servers=silent.servers))

    offer(answered, None, SCENARIO, None)
    offer(silent, None, SCENARIO, None)

    assert answered.notes == [NO_MODEL_NOTE]
    assert silent.notes == []


# --- a context other than the run's: the note before the question, no chat --------------------------------------


def test_a_context_other_than_the_one_asked_for_is_said_before_the_question_and_no_chat_is_sent():
    server = _server()
    run, _fact = _started(server)
    asks_for = SCENARIO.model_copy(update={"context_requested": 4096})

    given = offer(run, None, asks_for, None)
    said = measure(run, None, asks_for, None, given, [ROW])  # a yes to the row leads to the note only

    assert len(given.rows) == 1 and said is None
    assert run.notes == [OTHER_CONTEXT_NOTE.format(server=8192, run=4096)]
    assert run.notes == ["the number will not count: LM Studio runs context 8192, this run asks for 4096"]
    assert chat_calls(server) == [] and len(list_calls(server)) == 1
    assert run.servers.measured == {} and card_facts(run) == ()


# --- the measurement: the line, the number and the card -----------------------------------------------------------


def test_the_measurement_sends_four_chats_and_five_reads_says_two_lines_and_keeps_the_number_for_the_card(steady):
    server = _server(chat_to(LLM_KEY))
    run, _fact = _started(server)

    given, said = _step_four(run, None, [ROW])

    assert said == MEASURED == "1 of 1 measured"
    assert run.notes == [
        f"measured LM Studio {LLM_KEY}: 64.0 tok/s (64.0{DASH}64.0), one request, wall time, context 8192",
        "LM Studio states no load, so a busy server reads slower; not ranked, nothing written",
    ]
    assert len(chat_calls(server)) == 4 and len(list_calls(server)) == 6  # the start row, then five reads
    assert {call[2]["messages"][0]["content"] for call in chat_calls(server)} == {shipped_protocol().prompt}
    measured = run.servers.measured[ROW]
    assert isinstance(measured, LmStudioMeasurement) and measured.problem is None and measured.calls == 9
    assert run.failures == []


def test_an_answers_file_naming_the_row_without_the_pick_measures_nothing():
    server = _server()
    run, _fact = _started(server)

    given, said = _step_four(run, None, [])

    assert len(given.rows) == 1 and said is None
    assert chat_calls(server) == [] and run.servers.measured == {}


def test_the_card_has_four_rows_in_the_agreed_words_and_none_wider_than_100_columns(steady):
    run, _fact = _started(_server())
    _step_four(run, None, [ROW])

    facts = card_facts(run)

    assert facts == (
        Fact("server", "LM Studio server", "64.0 tok/s, one request, wall time"),
        Fact("", "", "package unknown; not ranked, nothing written"),
        Fact("", "", "LM Studio states no load, so a busy server reads slower"),
        Fact("", "", "never asks to load; an unseen server load can affect the time"),
    )
    rendered = [plain(line) for line in card_lines(facts)]
    assert len(rendered) == 4 and all(len(line) <= 100 for line in rendered), [len(line) for line in rendered]
    assert all(len(fact.note) <= 65 for fact in facts)


def test_there_is_no_card_row_without_a_measurement():
    run, _fact = _started(_server())
    _step_four(run, None, [])

    assert card_facts(run) == ()
    assert card_facts(run_of(None)) == ()


@pytest.mark.parametrize(
    "fault",
    ["warm-up", "run 3", "fifth read"],
)
def test_a_measurement_that_ended_has_no_number_on_the_card_and_no_count_in_the_balance(fault, steady):
    stop = "lmstudio_chat_finish_stop.json"
    if fault == "fifth read":
        lists = [LOADED] * 5 + ["lmstudio_v1_models_empty.json"]  # the start row, four good reads, the fifth bad
        server = _server(lists=lists)
    else:
        server = _server(*([stop] if fault == "warm-up" else [chat_to(LLM_KEY)] * 3 + [stop]))
    run, _fact = _started(server)

    given, said = _step_four(run, None, [ROW])

    assert said is None and card_facts(run) == () and run.servers.measured == {}
    assert len(run.notes) == 1 and run.notes[0].startswith("LM Studio: ") and "1 of 1" not in " ".join(run.notes)


def test_a_measurement_with_a_problem_is_never_kept_even_when_handed_to_the_card_by_hand():
    run, _fact = _started(_server())
    run.servers.measured[ROW] = LmStudioMeasurement(LLM_KEY, 8192, None, None, None, "a request took no measurable time", 3)

    assert card_facts(run) == ()


# --- a fault outside the table, and the other servers of step 4 ---------------------------------------------------


def _with_a_second_server(monkeypatch, run):
    """`load_tests` over LM Studio and a second server that measures when picked; returns what it said."""
    from modelroom import local_servers, runtimes
    from modelroom.dialog import FileAsker
    from modelroom.runtimes import LMSTUDIO, LocalServer

    def second_measure(_run, _config, _scenario, _results, _offer, picked):
        return "second measured" if "second" in picked else None

    second = LocalServer(
        key="second", name="Second", origin="http://127.0.0.1:1", transport=lambda _run: None,
        start_fact=lambda _origins: None,
        offer=lambda _run, _config, _scenario, _results: Offer((Choice("second", "second row"),)),
        measure=second_measure, card_facts=lambda _run: (),
    )
    monkeypatch.setattr(runtimes, "local_servers", lambda: (LMSTUDIO.server, second))
    run.asker = FileAsker({"load_test": True, "load_test_packages": [ROW, "second"]})
    run.screen.blank = lambda: None
    run.screen.head = lambda _step: None
    run.screen.answer = lambda *_args, **_kwargs: None
    return local_servers.load_tests(run, None, SCENARIO, None)


def test_a_measurement_that_ended_leaves_step_4_going_with_the_other_server(monkeypatch):
    run, _fact = _started(_server("lmstudio_chat_finish_stop.json"))

    said = _with_a_second_server(monkeypatch, run)

    assert said == "second measured"
    assert "LM Studio: a request did not end with finish_reason length" in run.notes
    assert run.servers.measured == {}


_ROW_REPLIES = {  # the answer that ends a measurement on this row of the table, and the line the row gives
    3: (LmStudioServerError("gone"), "stopped answering during the measurement"),
    4: (LmStudioTimeoutError("limit"), "a request did not answer within 600 s"),
    5: (Response(status=400, body=(FIXTURES / "lmstudio_jit_on_unloaded_chat.json").read_bytes()),
        "failed to load the model for the request; nothing measured"),
    6: (Response(status=500, body=b"{}"), "a request did not answer with status 200"),
    7: ("lmstudio_chat_no_usage.json", "a request carried no readable finish_reason and usage"),
    8: ("lmstudio_chat_finish_stop.json", "a request did not end with finish_reason length"),
    9: ("lmstudio_chat_tokens_127.json", "a request did not carry 128 tokens"),
    10: ("lmstudio_chat_prompt_large.json", "a request did not fit the context 8192"),
    11: (chat_to(LLM_KEY), "a request took no measurable time"),  # a good answer, but the clock stands still
}


def _clock_standing_still_at(monkeypatch, chat_index: int) -> None:
    """Every chat takes 2 s on the clock, except the one at `chat_index`, which takes none."""
    real = loadtest_lmstudio.run_lmstudio_load_test

    def with_clock(server, instance, scenario, prompt):
        state = {"n": 0}

        def clock() -> int:
            n, state["n"] = state["n"], state["n"] + 1
            chat, end = divmod(n, 2)
            return chat * 100 * SECOND + (0 if chat == chat_index else end * 2 * SECOND)

        return real(server, instance, scenario, prompt, clock=clock)

    monkeypatch.setattr(guided_lmstudio, "run_lmstudio_load_test", with_clock)


@pytest.mark.parametrize("position", range(4), ids=["warm-up", "run 1", "run 2", "run 3"])
@pytest.mark.parametrize("row", sorted(_ROW_REPLIES))
def test_each_row_of_the_table_at_each_request_is_one_full_line_no_number_and_the_other_server_goes_on(
    monkeypatch, row, position
):
    reply, words = _ROW_REPLIES[row]
    if row == 11:
        _clock_standing_still_at(monkeypatch, position)
    server = _server(*([chat_to(LLM_KEY)] * position + [reply]))
    run, _fact = _started(server)

    said = _with_a_second_server(monkeypatch, run)

    assert said == "second measured"
    assert [note for note in run.notes if note.startswith("LM Studio: ")] == [f"LM Studio: {words}"]
    assert run.servers.measured == {} and card_facts(run) == () and run.failures == []
    assert len(chat_calls(server)) == position + 1  # no repeat


def test_a_success_and_the_other_server_are_both_counted(monkeypatch, steady):
    run, _fact = _started(_server())

    said = _with_a_second_server(monkeypatch, run)

    assert said == f"1 of 1 measured {UNICODE_GLYPHS.dot} second measured"


@pytest.mark.parametrize("error", [LmStudioServerError("boom"), LoadTestError("boom")])
def test_a_fault_outside_the_table_is_one_failed_line_and_step_4_goes_on(monkeypatch, error):
    def fails(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(guided_lmstudio, "run_lmstudio_load_test", fails)
    server = _server()
    run, _fact = _started(server)

    said = _with_a_second_server(monkeypatch, run)

    assert said == "second measured"
    assert run.failures == ["nothing measured for LM Studio: boom"]
    assert chat_calls(server) == [] and run.servers.measured == {} and card_facts(run) == ()


def test_a_programming_error_is_not_caught(monkeypatch):
    def breaks(*_args, **_kwargs):
        raise KeyError("a bug")

    monkeypatch.setattr(guided_lmstudio, "run_lmstudio_load_test", breaks)
    run, _fact = _started(_server())

    with pytest.raises(KeyError):
        _step_four(run, None, [ROW])


# --- nothing is written, nothing is looked at -----------------------------------------------------------------------


def test_nothing_is_written_nothing_is_ranked_and_the_number_stays_in_the_run(tmp_path, steady):
    """No measurement file, no profile and no snapshot: the results folder is as it was, byte for byte, and the
    run keeps the number in memory for the card."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "kept.json").write_bytes(b'{"kept": true}')
    run, _fact = _started(_server())
    before = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file()}

    _given, said = _step_four(run, results, [ROW])

    after = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file()}
    assert said == MEASURED and after == before
    assert list(run.servers.measured) == [ROW]


def test_step_four_reads_no_model_folder_and_starts_no_process(tmp_path, monkeypatch, steady):
    """The model folder holds the very file of the loaded model and a program `lms` stands in the home
    folder: nothing is looked at there (no stat, no listing) and no process is started."""
    from pathlib import Path

    folder = tmp_path / "home" / ".lmstudio" / "models" / "unsloth" / "Qwen3.5-4B-GGUF"
    folder.mkdir(parents=True)
    (folder / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"x" * 16)
    server = _server()
    shipped_protocol()  # the package's own protocol file is read once, before the folder is watched

    def refuse(*args, **kwargs):
        pytest.fail("the model folder was touched", pytrace=False)

    with monkeypatch.context() as patched:
        trap_processes(patched)
        patched.setattr(Path, "stat", refuse)
        patched.setattr(Path, "iterdir", refuse)
        patched.setattr(Path, "open", refuse)
        run, fact = _started(server)
        given, said = _step_four(run, None, [ROW])

    assert fact is not None and len(given.rows) == 1 and said == MEASURED
