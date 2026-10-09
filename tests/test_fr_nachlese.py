"""AP9-FR-N2, the lessons of the hand test on rented servers: what an SSH computer already holds (a),
what step 4 and the card say about a computer and a measurement (c), the one line of a cut pull (d),
the identity behind an alias (e) and the early end of a speed run (f).

Every test goes through an existing entry: `run_guided` with `FakeSsh` redirecting into
`remote_server.main` (as `tests/test_guided_remote.py`), `result_card` for the card of a document.
No real ssh, no real server, no network; names are the neutral `gpu1`, `server`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from fixture_http import FIXTURES
from fixture_room import fixed_room
from fixture_servers import _loadtest_answers
from fixture_ssh import FakeSsh
from fixture_support import (
    DEEPSEEK_OLLAMA_NAME,
    build_transport,
    guided_transport_mapping,
    loadtest_daemon,
    offline_daemon,
    offline_servers,
    ps_answer,
    windows_probes,
)
from modelroom.config import load_config
from modelroom.daemon import PULL_PATH, TAGS_PATH, FixtureDaemon
from modelroom.guided import run_guided
from modelroom.http import Response
from modelroom.importer import ImportConflictError
from modelroom.remote_store import aliases as remote_aliases
from modelroom.remote_store import remote_path_for
from modelroom.ssh import ProcessOutcome

from test_guided import FULL_ANSWERS, QWEN_GGUF, RUN1, RUN2, RUN3, UNSLOTH, _card, _pointer, _results, _WatchingAsker
from test_guided_remote import _measurable_context, _server_context

DEEPSEEK = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
LOCAL_IN_OLLAMA = "local     #1 already in Ollama on gpu1"
OLD_ID = "3f9a0c21d4e6b870"
NEW_ID = "7b1e4d09c2a85f36"
THIRD_ID = "9c2d5e18a4b7f031"
SENTINEL = "sentinel-host"

# gpu1 over SSH, DeepSeek measured there in step 4 (so its row stands at #1 in the table).
SSH_BASE = {"ssh_alias": "gpu1", "ssh_machine_class": "server"}
SSH_MEASURED = {
    **SSH_BASE,
    "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK],
    "load_test_remote": True,
    "load_test_remote_packages": [DEEPSEEK_OLLAMA_NAME],
}


@pytest.fixture(autouse=True)
def _results_folder(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()


def _run(tmp_path: Path, answers: dict, ssh, *, now=RUN1, machines=("ssh",), ids=None, daemon=None):
    """`run_guided` over `ssh`; returns the exit code, the lines as printed and the asker."""
    full = {**FULL_ANSWERS, "machines": list(machines), **answers}
    lines: list[str] = []
    asker = _WatchingAsker(full, lines)
    code = run_guided(
        asker,
        here=_results(tmp_path),
        pointer_path=_pointer(tmp_path),
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(ids=ids),
        daemon=daemon if daemon is not None else offline_daemon(),
        servers=offline_servers(),
        now=now,
        out=lines.append,
        ssh=ssh,
    )
    return code, lines, asker


def _holding(free_disk_bytes: int | None = None, daemon=None) -> FakeSsh:
    """A server whose Ollama holds the DeepSeek package, optionally with a small free disk."""
    context = _measurable_context(daemon=daemon if daemon is not None else loadtest_daemon())
    if free_disk_bytes is not None:
        context = replace(context, room=fixed_room(free_disk_bytes=free_disk_bytes))
    return FakeSsh(context=context)


def _table_lines(lines: list[str]) -> list[str]:
    """Everything printed from the head of step 5 on."""
    head = max(index for index, line in enumerate(lines) if "Step 5 of 5" in line)
    return lines[head:]


def _local_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip().startswith("local ")]


@dataclass
class _FailingOnce(FakeSsh):
    """`FakeSsh` that ends the `nth` call of `op` with exit 255 and counts every call per op."""

    op: str = "candidates"
    nth: int = 2
    counts: dict = field(default_factory=dict)
    failed_at: list = field(default_factory=list)

    def run(self, argv, stdin, total, per_line, limits, on_line):
        op = argv[-1]
        self.counts[op] = self.counts.get(op, 0) + 1
        if op == self.op and self.counts[op] == self.nth:
            self.failed_at.append(self.counts[op])
            self.calls.append((argv[-4], op, stdin))
            return ProcessOutcome(255, stderr=b"Connection reset by peer")
        return super().run(argv, stdin, total, per_line, limits, on_line)


# --- N2-a: what the active SSH computer holds ------------------------------------------------------


def test_the_table_names_the_rows_the_active_target_holds(tmp_path):
    code, lines, _asker = _run(tmp_path, {**SSH_MEASURED, "pull_other": "none"}, _holding())

    assert code == 0
    shown = _table_lines(lines)
    texts = [line.strip() for line in shown]
    assert LOCAL_IN_OLLAMA in texts
    speed = next(index for index, text in enumerate(texts) if text.startswith("speed ") and "#1 measured" in text)
    assert texts[speed + 1] == LOCAL_IN_OLLAMA  # directly behind the `speed` line of that table


def test_the_pull_other_list_marks_a_row_the_active_target_holds(tmp_path):
    value = f"gpu1 {DEEPSEEK_OLLAMA_NAME}"
    _code, lines, asker = _run(tmp_path, {**SSH_MEASURED, "pull_other": value, "pull_remote": True}, _holding())

    marked = next(choice for choice in asker.choices["pull_other"] if choice.value == value)
    assert marked.label.endswith("  local already")
    assert marked.disabled is None and not marked.warn
    assert any(line.strip() == "install   #1 is local already on gpu1" for line in lines)
    assert "pull_remote" not in asker.asked  # no question: nothing to pull


def test_a_package_the_active_target_holds_is_never_too_big(tmp_path):
    """U11c on the active SSH computer: its own free disk (3 GiB) is below both downloads, yet the
    row it holds is never marked; the row it does not hold (and needs 8.9 GB) still is."""
    code, lines, asker = _run(
        tmp_path, {**SSH_MEASURED, "pull_other": "none"}, _holding(free_disk_bytes=3 * 1024**3)
    )

    shown = _table_lines(lines)
    held_row = next(line for line in shown if line.strip().startswith("1 ") and "DeepSeek" in line)
    other_row = next(line for line in shown if line.strip().startswith("2 ") and "Qwen3.5-9B" in line)
    assert "×" not in held_row
    assert "×" in other_row  # the contrast: the same disk marks a package that is not held
    assert LOCAL_IN_OLLAMA in [line.strip() for line in shown]
    marked = [line for line in shown if line.strip().startswith("disk ") and "×" in line]
    assert len(marked) == 1 and "#2" in marked[0] and "#1" not in marked[0]
    held = next(choice for choice in asker.choices["pull_other"] if choice.value == f"gpu1 {DEEPSEEK_OLLAMA_NAME}")
    assert held.disabled is None and not held.warn


def test_step_5_calls_candidates_once(tmp_path):
    ssh = _holding()

    _code, _lines, _asker = _run(tmp_path, {**SSH_BASE, "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK], "pull_other": "none"}, ssh)

    ops = [call[1] for call in ssh.calls]
    # step 4 asks for the candidates of its own; step 5 asks exactly once more, before it reads the room
    assert ops.count("candidates") == 2
    assert ops[-2:] == ["candidates", "room"]


def test_a_failed_candidates_call_in_step_5_adds_no_problem(tmp_path):
    ssh = _FailingOnce(context=_holding().context)
    other = tmp_path / "second"
    (other / "results").mkdir(parents=True)
    (other / "home").mkdir()
    clean_ssh = FakeSsh(context=_holding().context)

    code, lines, _asker = _run(tmp_path, {**SSH_MEASURED, "pull_other": "none"}, ssh)
    clean_code, clean_lines, _ = _run(other, {**SSH_MEASURED, "pull_other": "none"}, clean_ssh)

    assert ssh.counts["candidates"] == 2 and ssh.failed_at == [2]  # the failure reached the call of step 5
    assert _local_lines(lines) == []
    assert clean_code == 0 and code == 0  # no problem of its own: the digest proof at the pull stays the check
    assert not any("did not finish" in line for line in lines)
    assert [line.strip() for line in lines if line.strip().startswith("disk ")] == [
        line.strip() for line in clean_lines if line.strip().startswith("disk ")
    ]


def test_a_failed_candidates_call_in_step_5_leaves_the_room_line_as_it_was(tmp_path):
    ssh = _FailingOnce(context=_holding().context)
    other = tmp_path / "second"
    (other / "results").mkdir(parents=True)
    (other / "home").mkdir()

    _code, lines, _asker = _run(tmp_path, {**SSH_MEASURED, "pull_other": "none"}, ssh)
    _clean_code, clean_lines, _ = _run(other, {**SSH_MEASURED, "pull_other": "none"}, FakeSsh(context=_holding().context))

    room = [line.strip() for line in lines if line.strip().startswith("room ")]
    assert room and room == [line.strip() for line in clean_lines if line.strip().startswith("room ")]


def test_a_failed_room_read_keeps_what_candidates_found(tmp_path):
    ssh = _FailingOnce(context=_holding().context, op="room", nth=1)

    _code, lines, asker = _run(tmp_path, {**SSH_MEASURED, "pull_other": "none"}, ssh)

    assert ssh.failed_at == [1]  # the failure reached the `room` call of step 5
    shown = _table_lines(lines)
    assert LOCAL_IN_OLLAMA in [line.strip() for line in shown]
    assert any("disk unknown (gpu1 not reached)" in line for line in shown)  # the disk line still says it
    held = next(choice for choice in asker.choices["pull_other"] if choice.value == f"gpu1 {DEEPSEEK_OLLAMA_NAME}")
    assert held.label.endswith("  local already")


def _sentinel_daemon() -> FixtureDaemon:
    """The server's Ollama holds DeepSeek and, besides it, a model named `sentinel-host:latest`."""
    data = json.loads((FIXTURES / "ollama_tags_local_loadtest.json").read_text(encoding="utf-8"))
    data["models"].append(
        {"name": f"{SENTINEL}:latest", "model": f"{SENTINEL}:latest", "size": 1_234_567_890, "digest": "f" * 64, "details": {}}
    )
    answers = _loadtest_answers()
    answers[("GET", TAGS_PATH)] = Response(200, {}, json.dumps(data).encode("utf-8"))
    return FixtureDaemon(answers)


def test_an_installed_name_of_the_server_reaches_neither_the_screen_nor_a_file(tmp_path):
    code, lines, _asker = _run(tmp_path, {**SSH_MEASURED, "pull_other": "none"}, _holding(daemon=_sentinel_daemon()))

    assert LOCAL_IN_OLLAMA in [line.strip() for line in lines]
    assert not any(SENTINEL in line for line in lines)
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files and not any(SENTINEL.encode() in path.read_bytes() for path in files)


def test_the_held_set_carries_identities_only():
    """A registry package installed at the server as `sentinel-host:latest` is proven by its manifest digest
    alone: `active_held` returns the identity, and the name the server gave it never leaves the server."""
    from types import SimpleNamespace

    from fixture_support import GRANITE_MANIFEST_DIGEST, granite_ollama_package
    from modelroom.daemon import GENERATE_PATH, PS_PATH, VERSION_PATH
    from modelroom.guided_remote import active_held
    from modelroom.quantization import package_identity_key

    package = granite_ollama_package()
    digest = GRANITE_MANIFEST_DIGEST.removeprefix("sha256:")
    named = f"{SENTINEL}:latest"
    tags = {"models": [{"name": named, "model": named, "size": 5_347_929_757, "digest": digest, "details": {}}]}
    daemon = FixtureDaemon({
        ("GET", VERSION_PATH): Response(200, {}, b"{}"),
        ("GET", TAGS_PATH): Response(200, {}, json.dumps(tags).encode("utf-8")),
        ("GET", PS_PATH): ps_answer(),
        ("POST", GENERATE_PATH): Response(200, {}, b"{}"),
    })
    ssh = FakeSsh(context=_measurable_context(daemon=daemon))

    held = active_held(SimpleNamespace(ssh=ssh), "gpu1", [package])

    assert held == frozenset({package_identity_key(package)})
    assert not any(SENTINEL.encode() in line for line in ssh.lines_seen)


def test_nothing_held_writes_no_local_line(tmp_path):
    """The goldens of a run that holds nothing stay as they are (`tests/golden/guided-screen*.txt`)."""
    empty = FixtureDaemon(_loadtest_answers(tags="ollama_tags_local_loadtest.json") | {("GET", TAGS_PATH): Response(200, {}, b'{"models": []}')})

    _code, lines, _asker = _run(tmp_path, {**SSH_BASE, "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK], "pull_other": "none"}, _holding(daemon=empty))

    assert _local_lines(lines) == []


def test_a_stored_target_not_connected_gets_no_local_line(tmp_path):
    """gpu1 is stored and measured, gpu2 is the active target of this run: gpu1 is not asked, its
    table has no `local` line and its rows in the list do not say `local already`."""
    first = _holding()
    _run(tmp_path, {**SSH_MEASURED, "pull_other": "none"}, first)
    second = FakeSsh(context=_measurable_context("66666666-7777-8888-9999-aaaaaaaaaaaa", daemon=loadtest_daemon()))
    _run(tmp_path, {**SSH_BASE, "ssh_alias": "gpu2", "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK]}, second, now=RUN2)
    config = load_config(_results(tmp_path) / "modelroom.toml")
    gpu1_id, gpu2_id = config.machines["gpu1"].profile, config.machines["gpu2"].profile
    third = FakeSsh(context=_measurable_context("66666666-7777-8888-9999-aaaaaaaaaaaa", daemon=loadtest_daemon()))
    picked = (f"profile:{gpu1_id}", f"profile:{gpu2_id}")

    _code, lines, asker = _run(
        tmp_path,
        {"machines": list(picked), "ssh_active": "gpu2", "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK], "pull_other": "none"},
        third, now=RUN3, machines=picked,
    )

    assert not any(call[0] == "gpu1" for call in third.calls)
    shown = _table_lines(lines)
    gpu1_head = next(index for index, line in enumerate(shown) if line.strip().startswith("gpu1 "))
    gpu2_head = next(index for index, line in enumerate(shown) if line.strip().startswith("gpu2 "))
    gpu1_block = shown[gpu1_head:gpu2_head] if gpu1_head < gpu2_head else shown[gpu1_head:]
    assert any("DeepSeek" in line for line in gpu1_block)  # the measured row is there, nothing says it is held
    assert not any(line.strip().startswith("local ") for line in gpu1_block)
    listed = [choice.label for choice in asker.choices["pull_other"] if choice.value.startswith("gpu1 ")]
    assert listed and not any(label.endswith("local already") for label in listed)


def test_the_local_line_changes_no_rank(tmp_path):
    """The same measured folder ranked twice, once with the package held at the server and once without."""
    ranks = []
    for index, daemon in enumerate((loadtest_daemon(), FixtureDaemon({("GET", "/api/version"): Response(200, {}, b"{}"), ("GET", TAGS_PATH): Response(200, {}, b'{"models": []}')}))):
        answers = {**SSH_MEASURED, "pull_other": "none"} if index == 0 else {**SSH_BASE, "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK], "pull_other": "none"}
        if index == 0:
            _run(tmp_path, answers, _holding(daemon=daemon))
        else:
            config = load_config(_results(tmp_path) / "modelroom.toml")
            picked = (f"profile:{config.machines['gpu1'].profile}",)
            _run(tmp_path, {**answers, "machines": list(picked)}, _holding(daemon=daemon), now=RUN2, machines=picked)
        payload = json.loads((_results(tmp_path) / "docs" / "models.json").read_text(encoding="utf-8"))
        block = next(block for block in payload["machines"] if block["machine"] == "gpu1")
        ranks.append([(row["rank"], tuple(row["package_identity"])) for row in block["ranked"]])

    assert ranks[0] == ranks[1] and ranks[0]


# --- N2-b: the version of a server is read by no rule of the laptop -------------------------------


@dataclass
class _VersionedSsh(FakeSsh):
    """`FakeSsh` whose `hello` answers claim the version `claimed`; `seen` keeps what the server said itself."""

    claimed: str = "9.9.9"
    seen: list = field(default_factory=list)

    def run(self, argv, stdin, total, per_line, limits, on_line):
        def rewrite(line: bytes):
            data = json.loads(line)
            if isinstance(data.get("result"), dict) and "version" in data["result"]:
                self.seen.append(data["result"]["version"])
                data["result"]["version"] = self.claimed
                line = json.dumps(data, separators=(",", ":")).encode("utf-8") + b"\n"
            return on_line(line)

        return super().run(argv, stdin, total, per_line, limits, rewrite)


def test_a_server_with_another_version_but_protocol_1_goes_on(tmp_path):
    ssh = _VersionedSsh(context=_idle("a" * 32))

    code, _lines, _asker = _run(tmp_path, SSH_BASE, ssh)

    assert ssh.seen  # the server did answer `hello`, and the laptop read the other number
    assert code == 0
    assert list(remote_aliases(remote_path_for(_pointer(tmp_path))).values()) == ["gpu1"]


# --- N2-c: which computer, which measurement -----------------------------------------------------


def test_step_4_names_this_computer_when_it_has_no_profile_here(tmp_path):
    _code, lines, _asker = _run(tmp_path, {**SSH_MEASURED, "pull_other": "none"}, _holding())

    texts = [line.strip() for line in lines]
    said = "no load test on this computer: it has no profile in this results folder"
    assert said in texts
    assert "no load test: this machine is not bound to a profile in this results folder" not in texts
    assert texts[texts.index(said) + 1].startswith("? Measure the speed on gpu1 now?")


def _stop_answer(count: int = 2) -> Response:
    """A `/api/generate` answer that ends early: `done_reason stop` after `count` tokens."""
    data = json.loads((FIXTURES / "ollama_generate_valid.json").read_text(encoding="utf-8"))
    data["done_reason"], data["eval_count"] = "stop", count
    return Response(status=200, body=json.dumps(data).encode("utf-8"))


def _local_run(tmp_path: Path, generates):
    from test_guided_install import MEASURED, _run as install_run

    return install_run(tmp_path, {**MEASURED, "pull": False}, loadtest_daemon(generates=generates))


def test_the_card_names_a_measurement_that_did_not_count(tmp_path):
    _code, lines, _asker = _local_run(tmp_path, [_stop_answer()] * 4)

    card = _card(lines)
    assert card["speed"].split() == ["speed", "none", "counted", "#1", "measured,", "not", "valid"]
    assert "say Yes in step 4" not in card["speed"]


def test_a_bound_chosen_computer_gets_no_note(tmp_path):
    _code, lines, _asker = _local_run(tmp_path, None)

    assert not any(line.startswith("no load test") for line in lines)


def _note(entry, text: str | tuple[str, str]):
    code, text = text if isinstance(text, tuple) else ("measured_not_comparable", text)
    return entry.note.model_copy(update={"code": code, "text": text})


def _speed_fact(document):
    from test_render import _card_facts

    return next(fact for fact in _card_facts(document) if fact.label == "speed")


def _document_with(notes: dict[int, str], valid: bool = False):
    from test_render import PROFILE_ID, _for_requests, _hf_package, _measurement

    package = _hf_package(file_digest="sha256:" + "c" * 64)
    document = _for_requests(1, None, None, [package], {PROFILE_ID: [_measurement(package)]} if valid else {})
    entry = document.machines[0].ranked[0]
    rows = [entry] if valid else []
    rows += [entry.model_copy(update={"rank": rank, "speed_tps": None, "note": _note(entry, text)}) for rank, text in notes.items()]
    block = document.machines[0].model_copy(update={"ranked": rows})
    return document.model_copy(update={"machines": [block]})


INVALID = "measured, not valid (run 1: done_reason 'stop', expected 'length')"
OTHER_CONTEXT = "measured at context 32k, this ranking's is 8k"
ONE_REQUEST = ("measured_with_one_request", "measured with 1 request, ranking assumes 3")


def test_two_reasons_too_wide_together_keep_the_first():
    document = _document_with({2: OTHER_CONTEXT, 4: INVALID})
    joined = "#2 measured at context 32k, this ranking's is 8k · #4 measured, not valid"
    assert len(joined) > 65  # the premise: the two different reasons do not fit one card row

    fact = _speed_fact(document)

    assert (fact.value, fact.note) == ("none counted", "#2 measured at context 32k, this ranking's is 8k")


def test_the_first_reason_is_the_one_the_ranking_meets_first():
    document = _document_with({4: INVALID, 2: OTHER_CONTEXT})
    document.machines[0].ranked.sort(key=lambda entry: entry.rank, reverse=True)  # rank 4 first

    assert _speed_fact(document).note == "#4 measured, not valid"


def test_a_one_request_reason_met_first_is_kept_where_both_kinds_are_too_wide():
    document = _document_with({1: ONE_REQUEST, 2: INVALID})
    joined = "#1 measured with 1 request, ranking assumes 3 · #2 measured, not valid"
    assert len(joined) > 65  # the premise: the two kinds do not fit one card row

    fact = _speed_fact(document)

    assert (fact.value, fact.note) == ("none counted", "#1 measured with 1 request, ranking assumes 3")


def test_another_reason_met_first_is_kept_before_the_one_request_reason():
    document = _document_with({1: INVALID, 2: ONE_REQUEST})

    fact = _speed_fact(document)

    assert (fact.value, fact.note) == ("none counted", "#1 measured, not valid")


def test_ranks_too_many_for_the_row_are_a_count():
    ranks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]  # no run of ranks: every rank is named
    document = _document_with({rank: INVALID for rank in ranks})
    named = "#1, #3, #5, #7, #9, #11, #13, #15, #17, #19, #21, #23 measured, not valid"
    assert len(named) > 65  # the premise: one reason alone, named by rank, is wider than the row

    fact = _speed_fact(document)

    assert (fact.value, fact.note) == ("none counted", "12 rows measured, not counted here")


def test_contiguous_invalid_rows_stay_ranks():
    document = _document_with({rank: INVALID for rank in range(1, 6)})

    fact = _speed_fact(document)

    assert (fact.value, fact.note) == ("none counted", "#1–5 measured, not valid")


def test_the_card_names_both_reasons_where_they_fit():
    document = _document_with({2: "measured, not comparable (another model was loaded during the measurement)", 8: INVALID})

    fact = _speed_fact(document)

    assert fact.note == "#2 measured, not comparable · #8 measured, not valid"


def test_the_card_still_says_measure_where_nothing_was_measured():
    from modelroom.screen import MEASURE_HINT

    fact = _speed_fact(_document_with({}))

    assert (fact.value, fact.note) == ("not measured", MEASURE_HINT)


def test_a_valid_speed_keeps_the_card_as_it_is():
    document = _document_with({3: INVALID}, valid=True)

    fact = _speed_fact(document)

    assert fact.value == "1 measured" and fact.note.startswith("fastest ") and fact.note.endswith(" tok/s, protocol v1")


def test_the_card_keeps_the_one_request_reason():
    from test_render import PROFILE_ID, _for_requests, _hf_package, _measurement

    package = _hf_package(file_digest="sha256:" + "c" * 64)
    document = _for_requests(3, 25, "from_users", [package], {PROFILE_ID: [_measurement(package)]})

    fact = _speed_fact(document)

    assert (fact.value, fact.note) == ("none counted", "#1 measured with 1 request, ranking assumes 3")


# --- N2-d: a cut pull over SSH is one line ---------------------------------------------------------


@dataclass
class _CutPullSsh(FakeSsh):
    """`FakeSsh`, except that the call carrying `POST /api/pull` ends after one progress line with exit 255."""

    def run(self, argv, stdin, total, per_line, limits, on_line):
        if argv[-1] == "daemon" and b"/api/pull" in stdin:
            self.calls.append((argv[-4], argv[-1], stdin))
            on_line(json.dumps({"protocol": 1, "progress": {"status": "pulling manifest"}}).encode() + b"\n")
            return ProcessOutcome(255, stderr=b"Connection reset by peer")
        return super().run(argv, stdin, total, per_line, limits, on_line)


def _empty_tags_daemon(status: int | None = None) -> FixtureDaemon:
    answers = {("GET", "/api/version"): Response(200, {}, b"{}"), ("GET", TAGS_PATH): Response(200, {}, b'{"models": []}')}
    if status is not None:
        answers[("POST", PULL_PATH)] = Response(status, {}, b"{}")
    return FixtureDaemon(answers)


FIRST_NAME = "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0"
PULL_ANSWERS = {**SSH_BASE, "pull_other": f"gpu1 {FIRST_NAME}", "pull_remote": True}


def _pull_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip().startswith("the pull of ")]


def test_a_connection_cut_mid_pull_is_one_line_with_one_did_not_finish(tmp_path):
    ssh = _CutPullSsh(context=_measurable_context(daemon=_empty_tags_daemon()))

    code, lines, _asker = _run(tmp_path, PULL_ANSWERS, ssh)

    expected = (
        f"the pull of {FIRST_NAME} did not finish: the connection to gpu1 ended; "
        "Ollama there keeps what arrived, the same pull resumes"
    )
    assert _pull_lines(lines) == [expected]
    assert expected.count("did not finish") == 1
    assert code == 1


def test_ollama_status_on_gpu1_keeps_its_line(tmp_path):
    ssh = FakeSsh(context=_measurable_context(daemon=_empty_tags_daemon(status=500)))

    _code, lines, _asker = _run(tmp_path, PULL_ANSWERS, ssh)

    assert _pull_lines(lines) == [f"the pull of {FIRST_NAME} did not finish: Ollama on gpu1 answered with status 500"]


def test_a_local_pull_cut_is_unchanged(tmp_path):
    from fixture_support import pull_daemon
    from modelroom.daemon import DaemonError
    from test_guided_install import ANSWERS, _run as install_run

    daemon = pull_daemon(pull=DaemonError("the connection ended"))

    code, lines, _asker = install_run(tmp_path, {**ANSWERS, "pull": True}, daemon)

    assert _pull_lines(lines) == [f"the pull of {FIRST_NAME} did not finish: the connection ended"]
    assert code == 1


# --- N2-e: the identity behind an alias -------------------------------------------------------------


def _idle(machine_id: str):
    """A server that answers with `machine_id` and holds no package (Ollama answers, nothing installed)."""
    return replace(_server_context(machine_id), daemon=_empty_tags_daemon())


def _bind_first(tmp_path: Path, machine_id: str = "a" * 32, ids=(OLD_ID,)) -> str:
    _run(tmp_path, SSH_BASE, FakeSsh(context=_idle(machine_id)), ids=list(ids))
    return load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile


def _again(tmp_path: Path, profile_id: str, machine_id: str, answers: dict | None = None, now=RUN2, ids=None):
    """Next run picking the stored profile directly: the identity is checked before anything uses it."""
    picked = (f"profile:{profile_id}",)
    return _run(tmp_path, {"machines": list(picked), **(answers or {})}, FakeSsh(context=_idle(machine_id)), now=now, machines=picked, ids=ids)


def test_a_third_machine_after_new_asks_again(tmp_path):
    first = _bind_first(tmp_path, "a" * 32)
    _again(tmp_path, first, "b" * 32, {"ssh_clone": "new"}, ids=[NEW_ID])
    second = load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile
    assert second == NEW_ID

    _code, _lines, asker = _again(tmp_path, second, "c" * 32, {"ssh_clone": "new"}, now=RUN3, ids=[THIRD_ID])

    assert "ssh_clone" in asker.asked


def test_the_new_machine_is_quiet_on_the_next_run(tmp_path):
    first = _bind_first(tmp_path, "a" * 32)
    _again(tmp_path, first, "b" * 32, {"ssh_clone": "new"}, ids=[NEW_ID])

    _code, _lines, asker = _again(tmp_path, NEW_ID, "b" * 32, now=RUN3)

    assert "ssh_clone" not in asker.asked
    assert load_config(_results(tmp_path) / "modelroom.toml").machines["gpu1"].profile == NEW_ID


def _profile_bytes(tmp_path: Path, profile_id: str) -> bytes:
    return (_results(tmp_path) / "state" / "hardware" / f"{profile_id}.json").read_bytes()


def _without_name(raw: bytes) -> dict:
    """A stored profile without its display name: the import renames the old machine to its neutral name."""
    data = json.loads(raw)
    data.pop("display_name")
    return data


def test_a_failed_import_after_new_is_said_and_the_next_run_converges(tmp_path, monkeypatch):
    import modelroom.guided_remote_bind as bind

    first = _bind_first(tmp_path, "a" * 32)
    assert first == OLD_ID
    old_bytes = _profile_bytes(tmp_path, OLD_ID)
    with monkeypatch.context() as patched:
        def conflict(*args, **kwargs):
            raise ImportConflictError("the stored file differs")

        patched.setattr(bind, "_import_profile", conflict)
        _code, lines, _asker = _run(tmp_path, {**SSH_BASE, "ssh_clone": "new"}, FakeSsh(context=_idle("b" * 32)), now=RUN2, ids=[NEW_ID])

    assert any("the profile of gpu1 could not be imported" in line for line in lines)
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {NEW_ID: "gpu1"}  # points at a profile with no file
    assert not (_results(tmp_path) / "state" / "hardware" / f"{NEW_ID}.json").exists()
    assert _profile_bytes(tmp_path, OLD_ID) == old_bytes

    code, _lines, asker = _run(tmp_path, SSH_BASE, FakeSsh(context=_idle("b" * 32)), now=RUN3, ids=[THIRD_ID])

    assert code == 0 and "ssh_clone" not in asker.asked
    assert remote_aliases(remote_path_for(_pointer(tmp_path))) == {THIRD_ID: "gpu1"}
    assert (_results(tmp_path) / "state" / "hardware" / f"{THIRD_ID}.json").is_file()
    assert _without_name(_profile_bytes(tmp_path, OLD_ID)) == _without_name(old_bytes)  # only its name turned neutral


NO_IDENTITY = "gpu1 gives no machine identity; modelroom cannot tell whether it is still the same machine"


def test_a_server_that_lost_its_machine_id_asks(tmp_path):
    first = _bind_first(tmp_path, "a" * 32)

    _code, _lines, asker = _again(tmp_path, first, "", {"ssh_clone": "same"})

    assert "ssh_clone" in asker.asked


def test_a_server_that_gained_a_machine_id_asks(tmp_path):
    first = _bind_first(tmp_path, "")

    _code, _lines, asker = _again(tmp_path, first, "a" * 32, {"ssh_clone": "same"})

    assert "ssh_clone" in asker.asked


@pytest.mark.parametrize("typed", [False, True], ids=["picked", "typed"])
def test_two_servers_without_machine_id_say_they_cannot_be_told_apart(tmp_path, typed):
    first = _bind_first(tmp_path, "")

    if typed:
        code, lines, asker = _run(tmp_path, SSH_BASE, FakeSsh(context=_idle("")), now=RUN2)
    else:
        code, lines, asker = _again(tmp_path, first, "")

    assert [line.strip() for line in lines].count(NO_IDENTITY) == 1
    assert "ssh_clone" not in asker.asked
    assert code == 0  # measurement and pull go on as usual
    assert any(line.strip().startswith("✓ Results") for line in lines)


def test_a_first_reading_without_machine_id_says_nothing_to_compare(tmp_path):
    _code, lines, asker = _run(tmp_path, SSH_BASE, FakeSsh(context=_idle("")))

    assert NO_IDENTITY not in [line.strip() for line in lines]  # nothing stored, nothing to compare with
    assert "ssh_clone" not in asker.asked


# --- N2-f: a run that ends early ---------------------------------------------------------------------


V1_RETRY = "; protocol v2 sends the prompt through the model's template, say Yes to measure again"


def early_line(name: str, count: int, version: int = 2) -> str:
    """The Step 4 line of a run that ended early; only a stored v1 record carries the retry hint."""
    line = (
        f"{name}: run 1 ended after {count} of 128 tokens (done_reason 'stop'); "
        f"protocol v{version} counts a run only when it reaches 128 tokens (done_reason 'length')"
    )
    return line + V1_RETRY if version == 1 else line


def test_an_answer_that_stops_early_names_tokens_and_rule(tmp_path):
    _code, lines, _asker = _local_run(tmp_path, [_stop_answer(2)] * 4)

    texts = [line.strip() for line in lines]
    assert early_line(DEEPSEEK_OLLAMA_NAME, 2) in texts


def test_an_answer_that_stops_early_names_tokens_and_rule_over_ssh(tmp_path):
    ssh = _holding(daemon=loadtest_daemon(generates=[_stop_answer(2)] * 4))

    _code, lines, _asker = _run(tmp_path, {**SSH_MEASURED, "pull_other": "none"}, ssh)

    assert early_line(DEEPSEEK_OLLAMA_NAME, 2) in [line.strip() for line in lines]


def test_127_tokens_with_stop_names_127(tmp_path):
    _code, lines, _asker = _local_run(tmp_path, [_stop_answer(127)] * 4)

    early = [line.strip() for line in lines if "ended after" in line]
    assert early == [early_line(DEEPSEEK_OLLAMA_NAME, 127)]
    assert "at once" not in early[0]  # 127 of 128 is not "at once": the line only counts


def test_an_answer_with_another_invalid_reason_gets_no_extra_line(tmp_path):
    from fixture_servers import generate_answer

    _code, lines, _asker = _local_run(tmp_path, [generate_answer(), *(generate_answer(0) for _ in range(3))])

    assert any("invalid (run 1: eval_duration is not above zero)" in line or "eval_duration" in line for line in lines)
    assert not any("ended after" in line for line in lines)


def test_the_stored_reason_is_unchanged(tmp_path):
    _local_run(tmp_path, [_stop_answer(2)] * 4)

    files = list((_results(tmp_path) / "state" / "measurements").rglob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["validity"] == "invalid"
    assert record["validity_reason"] == "run 1: done_reason 'stop', expected 'length'"
    assert "ended after" not in files[0].read_text(encoding="utf-8")
