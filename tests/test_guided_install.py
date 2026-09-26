"""Step 5, its last question: pull the first row of this machine into the local Ollama daemon.

Every run goes through injected layers, as in `tests/test_guided.py`: a `FixtureTransport` for
the search and the fetch, fixture probes, a fixed clock, a pointer file inside `tmp_path`, a
`FileAsker` -- and a `FixtureDaemon` that plays the recorded answer of a real pull
(`tests/fixtures/ollama_pull_smollm2_stream.ndjson`). No test here reaches the daemon on port
11434 or pulls anything.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from modelroom.daemon import PULL_PATH, SHOW_PATH, TAGS_PATH, DaemonError
from modelroom.dialog import Asker, FileAsker, TerminalAsker
from modelroom.document import RenderDocument
from modelroom.guided import run_guided
from modelroom.guided_install import (
    CANCELED_LINE,
    FirstRow,
    LiveLine,
    first_row,
    pull_step,
    pulled_problem,
)
from modelroom.guided_context import snapshot_packages
from modelroom.config import load_config
from modelroom.http import Response
from modelroom.loadtest import installed_models

from fixture_support import (
    DEEPSEEK_OLLAMA_NAME,
    PULL_STREAM_LINES,
    build_transport,
    deepseek_package,
    granite_ollama_package,
    guided_transport_mapping,
    json_response,
    offline_daemon,
    pull_daemon,
    tags_with,
    windows_probes,
)

RUN1 = datetime(2026, 9, 23, 8, 0, 0, tzinfo=timezone.utc)
UNSLOTH = "unsloth/Qwen3.5-9B-GGUF"
QWEN_GGUF = "Qwen/Qwen3.5-9B-GGUF"
DEEPSEEK = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
# The first row of this machine for the answers below, at 8192 tokens: the smallest Qwen3.5-9B build
# (`tests/golden/guided-screen.txt` shows the same row at 32k).
FIRST_NAME = "hf.co/unsloth/Qwen3.5-9B-GGUF:UD-IQ2_XXS"

ANSWERS = {
    "results": "here",
    "machines": ["this-machine"],
    "search": "qwen",
    "filter_owners": True,
    "select": [UNSLOTH, QWEN_GGUF],
    "users": 1,
    "context": "8192",
}
PULL = {**ANSWERS, "pull": True}
# DeepSeek is installed in the fixture daemon and proven by its digest; measured, it is the first row.
MEASURED = {
    **ANSWERS,
    "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK],
    "load_test": True,
    "load_test_packages": [DEEPSEEK_OLLAMA_NAME],
}


@pytest.fixture(autouse=True)
def _folders(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()


def _run(tmp_path: Path, answers: dict, daemon) -> tuple[int, list[str], FileAsker]:
    lines: list[str] = []
    asker = FileAsker(answers)
    code = run_guided(
        asker,
        here=tmp_path / "results",
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=daemon,
        now=RUN1,
        out=lines.append,
        colored=False,
    )
    return code, [line.strip() for line in lines], asker


def _first_weights(tmp_path: Path) -> float:
    payload = json.loads((tmp_path / "results" / "docs" / "models.json").read_text(encoding="utf-8"))
    return next(block for block in payload["machines"] if block["machine"] == "workstation")["ranked"][0]["weights_gib"]


def _pull_calls(daemon) -> list:
    return [call for call in daemon.calls if call[1] == PULL_PATH]


def _after_results(lines: list[str]) -> list[str]:
    """Everything the run printed after the balance of step 5."""
    return lines[next(index for index, line in enumerate(lines) if line.startswith("✓ Results")) + 1 :]


# --- when the question is asked ----------------------------------------------------------------------


def test_the_question_follows_the_card_with_the_size_of_the_first_row(tmp_path: Path):
    daemon = pull_daemon(after_pull=tags_with(FIRST_NAME))

    code, lines, asker = _run(tmp_path, {**ANSWERS, "pull": False}, daemon)

    assert code == 0
    size = _first_weights(tmp_path)
    assert f"? Pull #1 into Ollama now? ({size:.1f} GB)  No" in _after_results(lines)
    assert "pull" in asker.asked
    assert _pull_calls(daemon) == []
    # The line to copy stays, whatever the answer.
    assert f"install   #1  ollama pull {FIRST_NAME}" in lines


def test_a_daemon_the_start_screen_did_not_reach_is_not_asked(tmp_path: Path):
    code, lines, asker = _run(tmp_path, PULL, offline_daemon())

    assert code == 0
    assert "pull" not in asker.asked
    assert not any(line.startswith("? Pull ") for line in lines)
    assert f"install   #1  ollama pull {FIRST_NAME}" in lines


def test_a_run_that_wrote_no_document_is_not_asked(tmp_path: Path):
    daemon = pull_daemon()

    _code, lines, asker = _run(tmp_path, {**PULL, "select": []}, daemon)

    assert "pull" not in asker.asked
    assert _pull_calls(daemon) == []
    assert not any(line.startswith("install ") for line in lines)


def test_an_answer_file_without_the_key_does_not_pull(tmp_path: Path):
    """`pull` is optional like `load_test`: whether it is asked depends on the machine."""
    daemon = pull_daemon()

    code, lines, asker = _run(tmp_path, ANSWERS, daemon)

    assert code == 0
    assert "pull" in asker.asked
    assert any(line.startswith("? Pull #1 into Ollama now?") and line.endswith("  No") for line in lines)
    assert _pull_calls(daemon) == []


def test_an_answer_that_is_no_switch_is_an_error_like_any_other(tmp_path: Path):
    from modelroom.dialog import AnswerInvalidError

    with pytest.raises(AnswerInvalidError):
        _run(tmp_path, {**ANSWERS, "pull": "yes please"}, pull_daemon())


def test_a_first_row_that_is_local_already_is_a_note_and_no_question(tmp_path: Path):
    """DeepSeek is in `/api/tags` and `/api/show` proves its weights: nothing to pull."""
    daemon = pull_daemon()

    code, lines, asker = _run(tmp_path, {**MEASURED, "pull": True}, daemon)

    assert code == 0
    assert "pull" not in asker.asked
    assert "install   #1 is local already · say Yes in step 4 to measure it" in _after_results(lines)
    assert _pull_calls(daemon) == []


def test_a_name_the_daemon_lists_without_the_proven_weights_is_asked_like_a_missing_one(tmp_path: Path):
    """The name is there, but `/api/show` shows another blob: a pull brings the registry's own build.

    The pinned `/api/show` answer names DeepSeek's weight file, which is no weight file of the Qwen
    build -- old content, or several weight files, look the same to this step.
    """
    listed = pull_daemon(after_pull=tags_with(FIRST_NAME), tags=tags_with(FIRST_NAME))

    _code, lines, asker = _run(tmp_path, {**ANSWERS, "pull": False}, listed)

    assert "pull" in asker.asked
    assert any(call == ("POST", SHOW_PATH, {"model": FIRST_NAME}) for call in listed.calls)
    assert not any("is local already" in line for line in lines)


# --- the pull -----------------------------------------------------------------------------------------


def test_a_yes_pulls_the_first_row_and_checks_the_list_of_the_daemon_afterwards(tmp_path: Path):
    daemon = pull_daemon(after_pull=tags_with(FIRST_NAME))

    code, lines, _asker = _run(tmp_path, PULL, daemon)

    assert code == 0
    assert _pull_calls(daemon) == [("POST", PULL_PATH, {"model": FIRST_NAME, "stream": True})]
    after = _after_results(lines)
    size = _first_weights(tmp_path)
    assert f"? Pull #1 into Ollama now? ({size:.1f} GB)  Yes" in after
    # Under `--answers` the pull is its first line and its last: no line per progress step.
    assert [line for line in after if line.startswith("pulling")] == [f"pulling   {FIRST_NAME}"]
    assert after[-1] == (
        f"✓ Pulled    {FIRST_NAME} · {size:.1f} GB · say Yes in step 4 of the next run to measure it"
    )
    tags_after_pull = [index for index, call in enumerate(daemon.calls) if call[1] == TAGS_PATH]
    assert tags_after_pull[-1] > daemon.calls.index(_pull_calls(daemon)[0])


def test_a_pull_writes_nothing_into_the_results_folder(tmp_path: Path):
    from modelroom.guided_install import pull_step

    _run(tmp_path, {**ANSWERS, "pull": False}, pull_daemon())
    first = _first_row_of(tmp_path)
    before = _tree(tmp_path)
    lines: list[str] = []
    run = _bare_run(tmp_path, pull_daemon(after_pull=tags_with(FIRST_NAME)), {"pull": True}, lines)

    pull_step(run, first, True)

    assert lines[-1].strip().startswith("✓ Pulled")
    assert _tree(tmp_path) == before  # every file of the results folder and the home folder


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize(
    "answer, words",
    [
        (Response(status=200, body=b'{"status":"pulling manifest"}\n{"error":"max retries exceeded"}\n'), "max retries exceeded"),
        (Response(status=404, body=b'{"error":"pull model manifest: file does not exist"}'), "status 404"),
        (DaemonError("http://127.0.0.1:11434/api/pull: the Ollama daemon did not answer (timed out)"), "timed out"),
        (Response(status=200, body=b'{"status":"pulling manifest"}\n{"status":"writing manifest"}\n'), "before success"),
        (Response(status=200, body=b""), "before success"),
        (Response(status=200, body=b'{"status":"pulling manifest"}\n{"status": 5}\n'), "unexpected line"),
        (Response(status=200, body=b'{"status":"pulling a","total":"10"}\n'), "unexpected line"),
        (Response(status=200, body=b'{"status":"pulling manifest"}\n{no json\n'), "not JSON"),
        (Response(status=200, body=b'{"status":"pulling a","total":1' + b"0" * 400 + b"}\n"), "unexpected line"),
        (Response(status=200, body=b'{"error":"failed"}\n{"status":"success"}\n'), "failed"),
        (Response(status=200, body=b'{"error":"first\\nsecond\\rthird"}\n'), "first second third"),
        (Response(status=200, body=b'{"error":"failed","total":1,"completed":"bad"}\n'), "unexpected line"),
    ],
)
def test_a_pull_that_did_not_finish_is_one_line_and_exit_1(tmp_path: Path, answer, words):
    daemon = pull_daemon(pull=answer, after_pull=tags_with(FIRST_NAME))

    code, lines, _asker = _run(tmp_path, PULL, daemon)

    assert code == 1
    faults = [line for line in _after_results(lines) if line.startswith(f"the pull of {FIRST_NAME} did not finish")]
    assert len(faults) == 1 and words in faults[0]
    assert "\n" not in faults[0] and "\r" not in faults[0]
    assert not any(line.startswith("✓ Pulled") for line in lines)
    assert "1 step(s) did not finish; see the lines above" in lines


def test_a_pull_the_daemon_then_does_not_list_is_a_fault(tmp_path: Path):
    daemon = pull_daemon()  # the inventory after the pull is the one from before

    code, lines, _asker = _run(tmp_path, PULL, daemon)

    assert code == 1
    assert f"{FIRST_NAME}: pulled, but the daemon does not list it" in lines
    assert not any(line.startswith("✓ Pulled") for line in lines)


def test_ctrl_c_during_a_pull_says_it_resumes_and_goes_up_as_it_came(tmp_path: Path):
    daemon = pull_daemon()

    class _Stopped(type(daemon)):
        def __call__(self, method, path, body=None, progress=None):
            if path == PULL_PATH:
                self.calls.append((method, path, body))
                progress({"status": "pulling manifest"})
                raise KeyboardInterrupt
            return super().__call__(method, path, body, progress)

    stopped = _Stopped(daemon._answers)
    lines: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        run_guided(
            FileAsker(PULL),
            here=tmp_path / "results",
            pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
            transport=build_transport(guided_transport_mapping()),
            probes=windows_probes(),
            daemon=stopped,
            now=RUN1,
            out=lines.append,
            colored=False,
        )

    assert lines[-1].strip() == CANCELED_LINE == "a canceled pull resumes with the same command"


# --- the pieces -----------------------------------------------------------------------------------------


def _document(tmp_path: Path) -> RenderDocument:
    payload = json.loads((tmp_path / "results" / "docs" / "models.json").read_text(encoding="utf-8"))
    return RenderDocument.model_validate(payload)


def _first_row_of(tmp_path: Path) -> FirstRow:
    config = load_config(tmp_path / "results" / "modelroom.toml")
    packages, _base = snapshot_packages(config)
    first = first_row(_document(tmp_path), "workstation", packages)
    assert first is not None
    return first


def _bare_run(tmp_path: Path, daemon, answers: dict, lines: list[str], asker: Asker | None = None):
    from modelroom.catalog import load_catalog
    from modelroom.guided import GuidedRun
    from modelroom.screen import Screen

    return GuidedRun(
        asker=asker if asker is not None else FileAsker(answers),
        here=tmp_path / "results",
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport({}),
        probes=windows_probes(),
        now=RUN1,
        catalog=load_catalog(),
        daemon=daemon,
        out=lines.append,
        screen=Screen(lines.append, colored=False),
    )


def test_the_first_row_is_the_install_line_of_this_machine_with_its_package(tmp_path: Path):
    _run(tmp_path, {**ANSWERS, "pull": False}, offline_daemon())

    first = _first_row_of(tmp_path)

    assert (first.rank, first.name) == (1, FIRST_NAME)
    assert first.package.repo == UNSLOTH
    assert first.weights_gib == pytest.approx(_first_weights(tmp_path))


def test_a_first_row_without_an_ollama_name_or_of_another_machine_is_none(tmp_path: Path):
    _run(tmp_path, {**ANSWERS, "pull": False}, offline_daemon())
    config = load_config(tmp_path / "results" / "modelroom.toml")
    packages, _base = snapshot_packages(config)
    document = _document(tmp_path)
    block = document.machines[0]
    unnamed = block.ranked[0].model_copy(update={"quantization": "unknown"})
    without_name = document.model_copy(
        update={"machines": [block.model_copy(update={"ranked": [unnamed, *block.ranked[1:]]})]}
    )

    assert first_row(without_name, "workstation", packages) is None
    assert first_row(document, "inference-server", packages) is None
    assert first_row(document, "workstation", []) is None


def test_enter_alone_at_the_terminal_declines_the_pull(tmp_path: Path):
    """The pointer starts on No (`default=False`): Enter alone pulls nothing. An answer file cannot
    show that -- a `FileAsker` reads the answer and never the default (third-model round, 2026-09-25)."""
    _run(tmp_path, {**ANSWERS, "pull": False}, offline_daemon())
    first = _first_row_of(tmp_path)
    daemon = pull_daemon(after_pull=tags_with(FIRST_NAME))
    lines: list[str] = []

    with create_pipe_input() as pipe:
        pipe.send_text("\r")
        run = _bare_run(tmp_path, daemon, {}, lines, asker=TerminalAsker(input=pipe, output=DummyOutput()))
        pull_step(run, first, True)

    assert _pull_calls(daemon) == []
    assert [line.strip() for line in lines if "Pull #1" in line] == [f"? Pull #1 into Ollama now? ({first.weights_gib:.1f} GB)  No"]


def test_a_registry_package_counts_as_pulled_only_with_its_manifest_digest():
    """An Ollama registry name is proven by its manifest digest; a Hugging Face name by its name."""
    package = granite_ollama_package()
    registry = FirstRow(rank=1, name="granite4.2:8b", weights_gib=5.0, package=package)
    installed, _reason = installed_models(_Daemon(json_response("ollama_tags_local_loadtest.json")), RUN1)
    changed = package.model_copy(update={"manifest_digest": "sha256:" + "e" * 64})
    hugging_face = FirstRow(rank=1, name=DEEPSEEK_OLLAMA_NAME, weights_gib=4.7, package=deepseek_package())

    assert pulled_problem(installed, registry) is None
    assert "another manifest" in pulled_problem(installed, FirstRow(1, "granite4.2:8b", 5.0, changed))
    assert "does not list it" in pulled_problem([], registry)
    # `hf.co/…`: the manifest digest `/api/tags` shows is not the GGUF's, so the name decides.
    assert pulled_problem(installed, hugging_face) is None


class _Daemon:
    """Just `/api/tags`, for the reader the step shares with the load test."""

    base_url = "http://127.0.0.1:11434"

    def __init__(self, tags: Response) -> None:
        self._tags = tags

    def __call__(self, method, path, body=None, progress=None):
        return self._tags


def test_the_live_line_at_a_terminal_is_rewritten_in_place_and_erased():
    class _Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _Terminal()
    live = LiveLine(stream)

    live.show(" pulling   0.1 of 0.3 GB")
    live.show(" pulling   0.2 of 0.3 GB")
    live.clear()

    written = stream.getvalue()
    assert written.count("\r") >= 3
    assert "0.2 of 0.3 GB" in written
    assert "\n" not in written
    assert written.endswith("\r")


def test_the_live_line_is_cut_to_the_terminal_and_a_status_is_one_line():
    """A line that wraps cannot be written again in place; a daemon's text keeps no line break."""
    from modelroom.guided_install import one_line

    stream = io.StringIO()
    live = LiveLine(stream, columns=20)

    live.show(" pulling   " + "x" * 50)

    assert max(len(part) for part in stream.getvalue().split("\r")) == 19
    assert one_line("first\nsecond\r\x1b[2Jthird") == "first second  [2Jthird"
    wide = io.StringIO()
    LiveLine(wide, columns=10).show("界" * 10)  # two columns each: four fit into nine
    assert max(len(part) for part in wide.getvalue().split("\r")) == 4


def test_a_line_with_a_bad_count_never_reaches_the_live_line_at_a_terminal(tmp_path: Path):
    """At a terminal every line is formatted; a count that is no number is a fault, not a traceback."""
    from modelroom.guided_install import _pull

    _run(tmp_path, {**ANSWERS, "pull": False}, pull_daemon())
    bad = Response(status=200, body=b'{"error":"failed","total":1,"completed":"bad"}\n')
    lines: list[str] = []
    run = _bare_run(tmp_path, pull_daemon(pull=bad), {"pull": True}, lines)
    stream = io.StringIO()

    _pull(run, _first_row_of(tmp_path), LiveLine(stream, columns=80))

    assert run.problems and "unexpected line" in run.problems[0]
    assert stream.getvalue() == ""  # nothing was shown, nothing is left to erase


def test_the_recorded_stream_is_the_whole_answer_of_a_real_pull():
    body = pull_daemon()._answers[("POST", PULL_PATH)].body
    lines = [json.loads(line) for line in body.splitlines()]

    assert len(lines) == PULL_STREAM_LINES
    assert lines[0] == {"status": "pulling manifest"} and lines[-1] == {"status": "success"}
