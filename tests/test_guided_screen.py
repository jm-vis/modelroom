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
    offline_servers,
    pull_daemon,
    tags_with,
    vllm_servers,
    windows_probes,
)

GOLDEN = Path(__file__).parent / "golden"
RUN_AT = datetime(2026, 9, 25, 8, 0, 0, tzinfo=timezone.utc)
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen/Qwen3.5-9B"
# The model the recorded vLLM server states it serves: an id is as long as it is, and stands last.
SERVED_ID = "Qwen/Qwen2.5-0.5B-Instruct"

CHOSEN = {
    "results": "here",
    "machines": ["this-machine"],
    "machine_class": "laptop",
    "search": "qwen",
    "filter_owners": True,
    "select": [QWEN, DEEPSEEK],
    "users": 1,
    "context": "L",
}
NOTHING_CHOSEN = {**CHOSEN, "select": []}
# The level `S` for the run that measures: the fixture daemon loads 8192, so any other context
# would make the measurement not comparable and no row would carry a speed.
MEASURED = {**CHOSEN, "context": "S", "load_test": True, "load_test_packages": [DEEPSEEK_OLLAMA_NAME]}
# This machine measured, and a second one entered by hand from its data sheet: unified memory.
ENTERED = {
    **CHOSEN,
    "machines": ["this-machine", "enter"],
    "entered_name": "studio",
    "entered_class": "workstation",
    "entered_ram": 32,
    "entered_gpu": "unified",
}


def _screen(tmp_path: Path, answers: dict, *, daemon=None, servers=None) -> str:
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
        servers=servers if servers is not None else offline_servers(),
        now=RUN_AT,
        out=lines.append,
        colored=False,
    )
    return _masked("\n".join(lines), results) + "\n"


def _masked(text: str, results: Path) -> str:
    """Replace what a golden file may not depend on: the folder, an id, a measured speed, the day the
    catalog shipped by `<day>` (it changes with every publication)."""
    text = text.replace(str(results), "<results>")
    text = re.sub(r"\b[0-9a-f]{16}\b", "<profile>", text)
    text = re.sub(r"catalog: shipped \d{4}-\d{2}-\d{2}", "catalog: shipped <day>", text)
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


def test_the_screen_of_a_run_that_pulls_the_first_row_ends_with_the_pull(tmp_path: Path):
    """The daemon answers: the list of the rows shown, the command of the choice, the question, the pull.

    Qwen3.5-9B alone since AP9-UX: with DeepSeek chosen too, its one package -- installed on the
    fixture daemon -- is the first row (ADR 0009), and a local first row asks nothing. Since AP9-UX2
    (U6) the user chooses the package; this run takes the `Q4_K_S` build, not `#1`.
    """
    chosen = "hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_S"
    daemon = pull_daemon(after_pull=tags_with(chosen))
    answers = {**CHOSEN, "select": [QWEN], "pull_package": chosen, "pull": True}

    _compare("guided-screen-pull.txt", _screen(tmp_path, answers, daemon=daemon))


def test_the_screen_of_a_run_with_a_machine_entered_by_hand_reads_as_the_golden_file(tmp_path: Path):
    """Two blocks: this machine measured, and `studio` entered by hand, computed on shared memory."""
    _compare("guided-screen-entered.txt", _screen(tmp_path, ENTERED))


@pytest.mark.parametrize(
    ("answers", "servers"),
    [
        (CHOSEN, None),
        (ENTERED, None),
        ({**CHOSEN, "users": "20 requests", "load_test": True, "load_test_packages": ["vllm"]}, vllm_servers),
    ],
    ids=["measured", "entered", "vllm"],
)
def test_no_line_of_the_screen_is_wider_than_a_hundred_characters(tmp_path: Path, answers, servers):
    """The head is 72, the table 100; nothing the run draws may wrap in a 100-column window."""
    screen = _screen(tmp_path, answers, servers=servers() if servers else None)
    for line in screen.splitlines():
        if "<results>" in line or "Hugging Face" in line or "no load test" in line or SERVED_ID in line:
            continue  # a path, a registry's own message and a served model's id are as long as they are
        if "cannot reach the Ollama daemon" in line:
            continue  # the daemon's own reason, since AP9-UX E also as `Ollama: <reason>` before the question
        if "ollama pull " in line or "lms get " in line or line.strip().startswith(("repo ", "graphics  ")) or "hf.co/" in line:
            continue  # a command, a repository, a pull name and a graphics adapter (AP9-GPU G8) are as long as they are
        assert len(line) <= 100, line
    # Step 4 since AP9-UX E: the reason the number will not count stands before the one question.
    assert servers is None or "the number will not count: the server serves" in screen


def test_a_knowledge_hint_for_a_long_name_breaks_into_lines_of_at_most_a_hundred_columns_as_the_screen_writes_them():
    """The hint of the catalog (E3, fixed words) for `Qwen-Image-2.1-PE-I2I`: the first line stood at 98 characters, and the
    screen adds a blank in front (`plain`) and the indent of a note, so it measured 101. Counted here as the screen writes it."""
    from modelroom.catalog_findings import KNOWLEDGE_HINT
    from modelroom.guided_search import hint_lines
    from modelroom.screen import note_line, plain

    hint = KNOWLEDGE_HINT.format(model="Qwen-Image-2.1-PE-I2I", publisher="Qwen") + " as of 2026-10-27."
    lines = hint_lines(hint)
    assert len(lines) >= 2 and " ".join(lines) == hint  # nothing cut, nothing added
    for line in lines:
        assert len(plain(note_line(line))) <= 100, line


def test_a_reason_of_step_four_breaks_before_the_longest_name_of_the_catalog():
    """Reasons 1 and 3 name a model; beyond 100 columns the line breaks before the name, never inside it."""
    from modelroom.catalog import load_catalog
    from modelroom.guided_vllm import not_counting_reason, wrapped
    from modelroom.vllm_server import ServedModel

    names = [model.hf_repo for family in load_catalog().families for model in family.models]
    longest = max(names, key=len)
    served = ServedModel(id=longest, root=longest, max_model_len=4096)
    for reason in (not_counting_reason(served, (QWEN, DEEPSEEK), 4096, "none"), not_counting_reason(served, (longest,), 4096, "none")):
        lines = wrapped(reason)
        assert " ".join(lines) == reason  # nothing cut, nothing added
        for line in lines:
            # A line is at most 100 columns wide, unless it holds a name that is longer on its own.
            assert len(f"   {line}") <= 100 or line.startswith(longest.partition("/")[2]) or line.endswith(longest), line


# --- AP9-FR-N3: one name for one machine, on every line that names it --------------------------------------


class _Instructions(FileAsker):
    """An answer file that keeps the instruction line each scale or list was asked with, and the choices (headings
    included) of every list."""

    def __init__(self, answers: dict) -> None:
        super().__init__(answers)
        self.instructions: dict[str, str | None] = {}
        self.lists: dict[str, list] = {}

    def select_or_text(self, key, question, choices, text_value, text_question, instruction=None, *, required=False):
        self.instructions[key] = instruction
        return super().select_or_text(key, question, choices, text_value, text_question, instruction, required=required)

    def select(self, key, question, choices, instruction=None):
        self.lists[key] = list(choices)
        return super().select(key, question, choices, instruction)


def _go(tmp_path: Path, asker: FileAsker) -> list[str]:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    lines: list[str] = []
    run_guided(
        asker,
        here=results,
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(host="workstation", ids=["a" * 16]),
        daemon=offline_daemon(),
        servers=offline_servers(),
        now=RUN_AT,
        out=lines.append,
        colored=False,
    )
    return lines


def _each_machine_holds_its_packages(monkeypatch) -> None:
    """Controlled live readings for the second run. Step 5 reads a disk for this computer and for the active SSH
    computer only, and an SSH computer is called by its alias (key = alias = name), so a `local` line of a machine
    whose key differs from its name does not come up in a run by itself. The one seam is `guided.active_room`, whose
    result the run lays over its own reading of this computer: here every machine of the document has a disk read
    on which Ollama holds every package of its own table."""
    from modelroom import guided
    from modelroom.room import Disk, LiveRoom

    def rooms(run, document, packages=()):
        return {
            block.machine: LiveRoom(disk=Disk(10**12, "C:", held=frozenset(tuple(entry.package_identity) for entry in block.ranked)))
            for block in document.machines
        }

    monkeypatch.setattr(guided, "active_room", rooms)


def _table_of(text: list[str], head: str) -> list[str]:
    """The lines of the table that begins at `head`, up to the head of the next table."""
    heads = [index for index, line in enumerate(text) if " · context " in line]
    begin = text.index(head)
    return text[begin : next((index for index in heads if index > begin), len(text))]


def test_a_run_with_two_machines_says_each_of_them_by_its_name_and_never_by_its_key(tmp_path: Path, monkeypatch):
    """Two machines whose keys (`computer-<8 hex>`) differ from the names of their profiles. The first run
    writes the configuration; the entered machine is then renamed to a fixed key in it, and the second run
    over the same folder -- with a controlled disk reading for each machine -- is the one held here, line by line."""
    import tomllib

    _go(tmp_path, FileAsker(ENTERED))
    config = tmp_path / "results" / "modelroom.toml"
    first = tomllib.loads(config.read_text(encoding="utf-8"))["machines"]
    entered_key = next(key for key, machine in first.items() if machine["profile"] == "0000000000000002")
    config.write_text(config.read_text(encoding="utf-8").replace(entered_key, "computer-aaaaaaaa"), encoding="utf-8")
    keys = set(tomllib.loads(config.read_text(encoding="utf-8"))["machines"])
    assert "computer-aaaaaaaa" in keys and len(keys) == 2 and all(key.startswith("computer-") for key in keys)

    pull = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
    asker = _Instructions(
        {**ENTERED, "machines": ["this-machine", "profile:0000000000000002"], "pull_other": f"computer-aaaaaaaa {pull}"}
    )
    _each_machine_holds_its_packages(monkeypatch)
    lines = _go(tmp_path, asker)

    start = next(index for index, line in enumerate(lines) if line.startswith(" machine   workstation "))  # the host name of the probe
    after = lines[start + 1 :]  # everything from the line after the start line on
    for key in keys:  # no line after the start line names a machine by its key
        assert not any(key in line for line in after), key
    text = [line.strip() for line in after]
    # Step 1, the answer to the machine list; step 3, the instruction line of the scale.
    assert any("Which machines should the result cover?  this machine (laptop), workstation-00000000" in line for line in text)
    assert asker.instructions["context"] == "last column: how many fetched packages fit laptop-00000000"
    # Step 5, the card, one row per machine, and the head of each table.
    assert any(line.startswith("machine   laptop-00000000 ") for line in text)
    assert any(line.startswith("machine   workstation-00000000 (entered) ") for line in text)
    here_head, other_head = "laptop-00000000 · context L 32k", "workstation-00000000 (entered) · another computer · context L 32k"
    assert here_head in text and other_head in text
    # The `local` line of each table, one by one: this computer in its contextual words, the other machine by its name.
    here_local = [line for line in _table_of(text, here_head) if line.startswith("local")]
    other_local = [line for line in _table_of(text, other_head) if line.startswith("local")]
    assert len(here_local) == 1 and here_local[0].endswith("already in Ollama on this computer")
    assert len(other_local) == 1 and other_local[0].endswith("already in Ollama on workstation-00000000")
    # The list for another computer: its headings (the value of a row stays the key), the answer to it and the handover.
    choices = asker.lists["pull_other"]
    assert [choice.label for choice in choices if choice.heading] == ["workstation-00000000"]
    assert [choice.value for choice in choices if not choice.heading][0].startswith("computer-aaaaaaaa ")
    assert f"? Which package should go onto another computer?  workstation-00000000  #1  {pull}" in text
    assert any(line.startswith("install   #1 on workstation-00000000  modelroom pull ") and pull in line for line in text)
