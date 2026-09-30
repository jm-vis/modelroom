"""The acceptance run of the guided mode, one step per criterion:

    uv run --frozen python scripts/selftest.py

It runs the guided mode twice with `--answers` in a temporary results folder, against the pinned
search answers from `tests/fixture_support.py` -- one page per account since 2026-09-24 (criteria
3 and 4 must not depend on whichever repositories the live Hub answers with today) -- and reports
each criterion with the evidence it read. Two things in the run are real, and they are the two gates: this machine is
measured with its own sources (criterion 2), and the load test runs against the Ollama daemon
of this machine (criterion 5). Everything else answers from the fixtures.

Criteria (the plan's own numbering):

1. the first start writes `modelroom.toml` with this device as the writer, and remembers the
   results folder in the pointer file;
2. this machine is measured: `gpu_state: measured` with one card or `multi_gpu` with the sum of
   two and more, the machine class read from the chassis or, on a machine whose chassis names
   none (a guest machine at a provider, chassis type 1 or 2), the answered class `server`, and
   llmfit cross-check `confirmed`;
3. the search resolves at least one publisher model, `<state>/search.json` names every page it
   asked with its class, every reason a repository cannot be picked with its number and every
   resolved model with its release, and the screen says where it asked and how much of the answer
   is a choice;
4. the ranking carries the rule and the context in its header, and Qwen3.5 stands under
   "not covered";
5. the prepared package is measured against the real local Ollama daemon: the measurement is
   valid and comparable, its file lies under `<state>/measurements/<profile_id>/`, and the
   package stands in the ranking in measurement group 0 with its speed;
6. the second start reuses the configuration, the binding, the measurement and the context: no
   clone question, no second profile file, no second measurement, `[guided].context` holds the
   answered context, the configuration the second run reads starts the size scale at that context
   again and on its level (`guided.context_default` and `guided_context.scale_choices`, the
   functions the question itself uses), and a `modelroom render` of its own writes that context
   into the document and keeps the measurement in group 0.
   What this run cannot show is a context other than the answered one carrying through: its
   answer file stays at the level `S` (8192) so that criteria 4 and 5 keep their pinned numbers,
   and 8192 is also what a render with no stored context assumes. That case is a test of its own
   (`tests/test_guided.py`, 4096 answered, measured, and rendered on its own afterwards);
7. the run reads as the guided dialog it promises: the start screen with its three rows, five step
   heads of 72 characters, one answer line per question with its answer in words and none of the
   dialog library's own, a size scale that says how many packages fit on the measured machine, a
   result table whose notes are bundled by memory pool, and the card of six rows that closes the
   run (the screen of 2026-09-24);
8. the search test set (`tests/search_testset.toml`) is asked **live**, one search per case, and
   reported case by case: what a person types and which models come back for it -- a repository
   id, a name with blanks, a typo, and nothing at all. Gate-free by decision (2026-09-25): the
   Hub's answer for a word changes by the hour, so a missing model is a `WARN`. What the tests gate
   is the shape of that report (`tests/test_selftest.py`);
9. and 10. the pull of step 5 (`scripts/selftest_pull.py`): the first row pulled to `success` from a
   fixture daemon that plays a recorded pull, and the first live run's `pull = false`;
11. the head count of step 3, one person and one request, kept (`scripts/selftest_head.py`);
12. a machine entered by hand, in a run and a results folder of its own (`scripts/selftest_entered.py`).

The test set and the live search run as smokes afterwards and never decide the exit code.

Nothing outside the temporary folder is written: `HOME`/`USERPROFILE` are moved into it before
anything runs, and the run refuses to start if the pointer file would still land anywhere else.
The report goes to stdout. Exit code: 0 every criterion met, 1 one was not, 2 the environment
could not be set up.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
SEARCH_NAME = "qwen"
SEARCH_LOG_VERSION = 3  # `search.json`'s own version (CONTRACTS.md, "Search log")
QWEN_BASE = "Qwen/Qwen3.5-9B"
UNSLOTH_GGUF = "unsloth/Qwen3.5-9B-GGUF"
# The prepared package of criterion 5: the one repository in the pinned search answer whose base
# model fit v1 can judge, and whose `Q4_K_M` build has to be installed on this machine. The load
# test itself runs against the real local daemon -- that is what criterion 5 is about.
DEEPSEEK_GGUF = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
DEEPSEEK_OLLAMA_NAME = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
# Its quantization: criteria 5 and 6 find the row by repository **and** quantization (ADR 0009, AP9-UX).
DEEPSEEK_QUANT = "Q4_K_M"
# The scale's level `S`, 8192 tokens, `8k` in the scenario sentence: criteria 4 and 5 keep the numbers
# they were pinned with, and the answer file shows that a level is a valid answer (CONTRACTS.md).
ANSWERED_LEVEL, ANSWERED_CONTEXT, ANSWERED_SHOWN = "S", 8192, "8k"
# The class the run answers when the chassis names none. `guided._asked_class` asks on every
# measurement (decided 2026-09-27), so both answer files carry the class: the one the chassis reads,
# which keeps the source `chassis`, else this one -- a rented guest machine reports chassis type 1 or
# 2 and is a server (`answered_class`, set into both files before the first is written).
ANSWERED_CLASS = "server"
ANSWERS_FIRST = {
    "results": "here",
    "machines": ["this-machine"],
    "machine_class": ANSWERED_CLASS,
    "search": SEARCH_NAME,
    "filter_owners": True,
    "select": [UNSLOTH_GGUF, DEEPSEEK_GGUF],
    "users": 1,  # one person, one request: criterion 5's measurement stays in group 0 (criterion 11)
    "context": ANSWERED_LEVEL,
    "load_test": True,
    "load_test_packages": [DEEPSEEK_OLLAMA_NAME],
    "pull": False,  # said outright: the self-test on a real machine never pulls (criterion 10)
}
# The second run answers the same questions, minus the folder (the pointer file remembers it) and
# with the load test declined: criterion 6 is that the first run's measurement is still there.
ANSWERS_SECOND = {
    key: value for key, value in ANSWERS_FIRST.items() if key not in ("results", "load_test_packages")
} | {"load_test": False}


class Step(NamedTuple):
    name: str
    ok: bool
    detail: str


class SetupError(Exception):
    """The environment for the self-test could not be set up."""


# -- pieces with logic of their own (unit tested) ------------------------------------------------


def answers_toml(answers: dict) -> str:
    """The answer file for one run: `schema_version`, then one key per question."""
    lines = ["# modelroom guided answers, written by scripts/selftest.py", "schema_version = 1"]
    for key, value in answers.items():
        if isinstance(value, list):
            lines.append(f"{key} = [" + ", ".join(json.dumps(entry) for entry in value) + "]")
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        else:
            lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def first_start_problems(config_text: str, pointer: dict, results: Path, machine: str) -> list[str]:
    """Criterion 1: the configuration names this device as the writer, the pointer the folder."""
    problems = []
    raw = tomllib.loads(config_text)
    if raw.get("schema_version") != 3:
        problems.append(f"schema_version is {raw.get('schema_version')!r}, expected 3")
    entry = raw.get("machines", {}).get(machine)
    if entry is None:
        problems.append(f"the configuration has no [machines.{machine}]")
    elif entry.get("writer") is not True:
        problems.append(f"[machines.{machine}].writer is {entry.get('writer')!r}, expected true")
    # The configuration lies in `results`; a relative path is read against its folder, as
    # `load_config` reads it (FR-N writes the folder's own path as ".").
    written = raw.get("guided", {}).get("results", "")
    if not written or Path(os.path.normpath(results / written)) != Path(os.path.normpath(results)):
        problems.append(f"[guided].results is {written!r}, expected {results} or a path relative to it")
    if Path(pointer.get("current", "")) != results:
        problems.append(f"the pointer file's current is {pointer.get('current')!r}, expected {results}")
    if pointer.get("bindings", {}).get(str(results)) is None:
        problems.append(f"the pointer file binds no profile to {results}")
    return problems


def answered_class(reading) -> str:
    """The class the run answers: the one the chassis reads (the answer keeps `chassis`), else
    `ANSWERED_CLASS`. `reading` is a `measure.MachineClassReading` of this machine."""
    return reading.machine_class if reading.source == "chassis" else ANSWERED_CLASS


def measurement_problems(profile: dict) -> list[str]:
    """Criterion 2: this machine was really measured -- one graphics card, or two and more with their sum, and
    its adapters -- its class from the chassis or the answered one, and llmfit confirmed both readings."""
    problems, state, cards = [], profile.get("gpu_state"), profile.get("gpus") or []
    total = sum(card.get("vram_gib") or 0 for card in cards)
    if not ((state, len(cards)) == ("measured", 1) or (state == "multi_gpu" and len(cards) > 1 and abs(total - (profile.get("vram_gib") or 0)) <= 0.01)):
        problems.append(f"gpu_state {state!r} with {len(cards)} card(s), expected 'measured' with one or 'multi_gpu' with their sum")
    machine_class, source = profile.get("machine_class"), profile.get("machine_class_source")
    # The run answers the class the chassis reads wherever it reads one (`answered_class`), so
    # `entered` here means the chassis was silent and the run answered; any other class was not ours.
    read = source == "chassis" and machine_class not in (None, "unknown")
    answered = source == "entered" and machine_class == ANSWERED_CLASS
    if not (read or answered):
        problems.append(f"machine class is {machine_class!r} ({source!r}), expected one the chassis names or the answered {ANSWERED_CLASS!r}")
    if profile.get("origin") != "measured":
        problems.append(f"origin is {profile.get('origin')!r}, expected 'measured'")
    adapters = profile.get("adapters") or []  # AP9-GPU F18, in form only: there, every counted card one of `gpus`
    if not adapters or sum(1 for adapter in adapters if adapter.get("counted")) != len(cards):
        problems.append(f"{len(adapters)} graphics adapter(s), expected a list whose counted cards are the {len(cards)} of gpus")
    for field in ("ram_physical", "vram"):
        status = profile.get("llmfit_crosscheck", {}).get(field, {}).get("status")
        if status != "confirmed":
            problems.append(f"llmfit cross-check of {field} is {status!r}, expected 'confirmed'")
    return problems


def search_problems(log: dict, lines: list[str]) -> list[str]:
    """Criterion 3: at least one resolved publisher model, and every reason once with its number.

    Read from `search.json` since 2026-09-24: the seven account lines, the request count, the
    budget and the reasons left the screen -- they answer a question about the search, not about
    the answer a reader is looking at (CONTRACTS.md, "Search log"). The two notes that stayed say
    where the search asked and how much of what it answered is a choice, and those are held
    against the screen here.
    """
    if log.get("schema_version") != SEARCH_LOG_VERSION:
        return [f"search.json has schema_version {log.get('schema_version')!r}, expected {SEARCH_LOG_VERSION}"]
    problems = []
    if log.get("mode") != "word":
        problems.append(f"search.json names mode {log.get('mode')!r}, expected 'word'")
    resolved = log.get("resolved", 0)
    if resolved < 1:
        problems.append(f"the search resolved {resolved} repositories, expected at least one")
    reasons = log.get("unresolved", [])
    if not reasons:
        problems.append("search.json names no reason for the repositories that cannot be picked")
    if any(not entry.get("reason") or not entry.get("count") for entry in reasons):
        problems.append(f"a reason is empty or covers no repository: {reasons}")
    if not [entry for entry in log.get("accounts", []) if entry.get("class") == "publisher"]:
        problems.append("search.json names no publisher account among the pages that were asked")
    # Schema 3: every resolved base model with its release, the one record of the computed ones.
    if not isinstance(log.get("models"), list) or not log["models"]:
        problems.append("search.json names no release for the models it resolved")
    if not any("searched Hugging Face at " in line for line in lines):
        problems.append("the screen does not say where the search asked")
    if not any("you can pick from" in line for line in lines):
        problems.append("the screen does not say how much of the answer is a choice")
    return problems


def ranking_problems(text: str, machine: str, block: dict) -> list[str]:
    """Criterion 4: the header carries rule and context, and Qwen3.5 is judged on its architecture.

    `entered`, not `default`: everything the dialog writes is the user's decision (CONTRACTS.md,
    "Guided mode", step 3). Qwen3.5 is `hybrid_linear` since AP9-UX2 (P1, ADR 0010), `basis: "architecture"`.
    """
    lines = text.splitlines()
    problems = []
    for expected in ("Ranking rule: ", f"Scenario: context {ANSWERED_SHOWN} (entered)", f"## Ranking: {machine}"):
        if not any(line.startswith(expected) for line in lines):
            problems.append(f"the document has no line starting with {expected!r}")
    judged = [
        entry
        for entry in block["ranked"] + block["no_fit"]
        if entry["base_model_hf_repo"] == QWEN_BASE
    ]
    if not judged:
        problems.append(f"no package of {QWEN_BASE} was judged on this machine")
    if any(entry["fit"]["basis"] != "architecture" for entry in judged):
        problems.append(f"a package of {QWEN_BASE} was judged on another basis than its architecture")
    stale = [entry["reason"] for entry in block["not_covered"] if entry["reason"] == "architecture not covered by v1"]
    if stale:
        problems.append(f"{len(stale)} packages are still set aside as not covered by v1")
    return problems


def load_test_problems(records: list[dict], row: dict | None) -> list[str]:
    """Criterion 5: one valid, comparable measurement of the prepared package, ranked with it."""
    if not records:
        return [
            f"nothing was measured: {DEEPSEEK_OLLAMA_NAME} has to be installed on this machine "
            "for criterion 5 (`ollama list` shows what is)"
        ]
    if len(records) != 1:
        return [f"{len(records)} measurement files, expected exactly one"]
    record = records[0]
    problems = []
    if record["protocol"] != "v1":
        problems.append(f"protocol is {record['protocol']!r}, expected 'v1'")
    if record["validity"] != "valid":
        problems.append(f"validity is {record['validity']!r}: {record['validity_reason']}")
    if not record["comparable"]:
        problems.append(f"the measurement is not comparable: {record['comparable_reason']}")
    if record["package"].get("hf_repo") != DEEPSEEK_GGUF:
        problems.append(f"the measurement names {record['package'].get('hf_repo')!r}, expected {DEEPSEEK_GGUF}")
    # A rule row or a row appended for its measurement (ADR 0009): either way in group 0 with its speed.
    if row is None:
        problems.append(f"{DEEPSEEK_GGUF} {DEEPSEEK_QUANT} stands in no ranking of this machine")
    elif row.get("measurement_group") != 0 or row.get("speed_tps") is None:
        problems.append(f"the ranked row is group {row.get('measurement_group')} with speed {row.get('speed_tps')!r}")
    return problems


def second_start_problems(
    before: dict,
    after: dict,
    profiles: list[str],
    lines: list[str],
    measurements: list[dict] | None = None,
    row: dict | None = None,
) -> list[str]:
    """Criterion 6: the configuration, the binding and the measurement are reused, not remade."""
    problems = []
    if len(profiles) != 1:
        problems.append(f"{len(profiles)} profile files after the second run, expected exactly one: {profiles}")
    if before.get("machines") != after.get("machines"):
        problems.append("the second run changed [machines]")
    if before.get("guided") != after.get("guided"):
        problems.append("the second run changed [guided]")
    if any("clone" in line for line in lines):
        problems.append("the second run asked the clone question")
    if measurements is not None and len(measurements) != 1:
        problems.append(f"{len(measurements)} measurement files after the second run, expected exactly one")
    if measurements is not None and (row is None or row["measurement_group"] != 0):
        problems.append("the first run's measurement is no longer ranked in group 0")
    if any(line.strip().startswith(f"measured {DEEPSEEK_OLLAMA_NAME}: ") for line in lines):
        problems.append("the second run measured the package a second time")
    return problems


def stored_context_problems(
    before: dict, offered: object, header: str | None, row: dict | None, level: str | None = None
) -> list[str]:
    """Criterion 6, the kept context: written by run 1, offered again, and rendered on its own.

    `before` is the configuration as the first run left it, `offered` the context that
    configuration makes the scale start at (`guided.context_default`) and `level` the level of the
    scale whose line carries the pointer for it (`guided_context.scale_choices`), both read before
    the second run -- an answer file answers every question outright, so what the dialog was
    offered cannot be read off this run; `tests/test_guided.py` watches the real `Asker` for that.
    `header` is the `Scenario:` line of the document a standalone `modelroom render` wrote
    afterwards, and `row` the measured package's row in that document.
    """
    answered = ANSWERED_CONTEXT
    problems = []
    stored = before.get("guided", {}).get("context")
    if stored != answered:
        problems.append(f"[guided].context is {stored!r} after the first run, expected {answered}")
    if offered != answered:
        problems.append(f"the configuration of the second run starts the question at {offered!r}, expected {answered}")
    if level is not None and level != ANSWERED_LEVEL:
        problems.append(f"the scale of the second run starts on level {level!r}, expected {ANSWERED_LEVEL!r}")
    if header is None:
        problems.append("the document of the standalone render carries no Scenario line")
    elif f"context {ANSWERED_SHOWN} " not in header:
        problems.append(f"the standalone render wrote {header!r}, expected context {ANSWERED_SHOWN}")
    if row is None or row["measurement_group"] != 0:
        problems.append("the standalone render does not keep the first run's measurement in group 0")
    return problems


def format_report(steps: list[Step]) -> str:
    lines = []
    for number, step in enumerate(steps, start=1):
        lines.append(f"[{number}] {step.name}: {'OK' if step.ok else 'FAIL'}")
        lines += [f"    {line}" for line in step.detail.splitlines() if line.strip()]
    return "\n".join(lines)


# -- the run --------------------------------------------------------------------------------------


def _import_test_support():
    """The fixture transport lives with the tests; the self-test uses the very same one."""
    sys.path.insert(0, str(REPO / "tests"))
    from fixture_support import build_transport, guided_transport_mapping

    return build_transport, guided_transport_mapping


def _redirect_home(home: Path) -> Path:
    """Move this process' home folder into the temporary tree, and prove it took effect."""
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    # A Windows PowerShell of the hardware measurement started under this USERPROFILE without its
    # `AppData\Local` writes its module analysis cache into the working directory (AP9-RM U11b).
    for name, parts in (("LOCALAPPDATA", ("AppData", "Local")), ("APPDATA", ("AppData", "Roaming"))):
        home.joinpath(*parts).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(home.joinpath(*parts))
    from modelroom.binding import default_pointer_path

    pointer = default_pointer_path()
    if not pointer.is_relative_to(home):
        raise SetupError(f"the pointer file would be {pointer}, which is outside {home}")
    return pointer


def _guided_run(answers: Path, results: Path, transport, now: datetime) -> tuple[int, list[str]]:
    """One `modelroom --answers <file>` run, in process, with the pinned search answer.

    `now` is passed in and the second run gets a later one: `fetch` refuses a run that is not
    strictly newer than the stored snapshot, and two runs a fraction of a second apart would
    otherwise share the same whole second and the second fetch would be turned away.
    """
    from modelroom.cli import main

    lines: list[str] = []
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        code = main(
            ["--answers", str(answers)],
            transport=transport,
            now=now,
            here=results,
            out=lines.append,
        )
    return code, lines + captured.getvalue().splitlines()


def _profile_files(results: Path) -> list[Path]:
    folder = results / "state" / "hardware"
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def _search_log(results: Path) -> dict:
    """What the search of this run left behind (`<state>/search.json`), or an empty answer."""
    path = results / "state" / "search.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _document(results: Path) -> tuple[str, dict]:
    markdown = results / "docs" / "models.md"
    return markdown.read_text(encoding="utf-8"), json.loads(markdown.with_suffix(".json").read_text(encoding="utf-8"))


def _machine_block(payload: dict, machine: str) -> dict:
    return next(block for block in payload["machines"] if block["machine"] == machine)


def _measurement_records(results: Path) -> list[dict]:
    """Every measurement file the run left behind, under `<state>/measurements/<profile_id>/`."""
    folder = results / "state" / "measurements"
    if not folder.is_dir():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.rglob("*.json"))]


def _ranked_row(block: dict, repo: str, quantization: str) -> dict | None:
    """The row of one package -- its repository **and** its quantization -- wherever it stands.

    Since ADR 0009 a repository may have more than one row, and a measured package the rule does not
    show is appended behind the rule's rows; the repository alone could find the wrong one.
    """
    rows = block["ranked"] + block["no_fit"]
    return next((row for row in rows if row["package_identity"][1] == repo and row["quantization"] == quantization), None)


def row_kind(row: dict) -> str:
    """`measured` for a row appended for its measurement (ADR 0009), else `rule`."""
    return "measured" if "ranking.shown_for_measurement" in row.get("note", {}).get("facts", []) else "rule"


def _machine_name(config_text: str) -> str:
    """The one machine the guided mode wrote into the configuration: this device."""
    machines = tomllib.loads(config_text)["machines"]
    return next(name for name, entry in machines.items() if entry.get("writer"))


def _first_run(results: Path, pointer: Path, transport, now: datetime) -> tuple[list[Step], list[str]]:
    """Criteria 1 to 4, from one guided run and the files it left behind, plus that run's output."""
    from modelroom.measure import Probes, read_machine_class

    probes = Probes()
    reading = read_machine_class(probes.platform, probes.runner, probes.read_text)
    ANSWERS_FIRST["machine_class"] = ANSWERS_SECOND["machine_class"] = answered_class(reading)
    answers = results.parent / "answers-first.toml"
    answers.write_text(answers_toml(ANSWERS_FIRST), encoding="utf-8", newline="\n")
    code, lines = _guided_run(answers, results, transport, now)
    config_text = (results / "modelroom.toml").read_text(encoding="utf-8")
    machine = _machine_name(config_text)
    pointer_data = json.loads(pointer.read_text(encoding="utf-8"))
    profiles = _profile_files(results)
    profile = json.loads(profiles[0].read_text(encoding="utf-8")) if profiles else {}
    search_log = _search_log(results)
    text, payload = _document(results)
    block = _machine_block(payload, machine)
    steps = [
        Step("guided run 1 ended with exit 0", code == 0, f"exit {code}, {len(lines)} lines printed"),
        Step(
            "(1) first start writes the configuration and remembers the folder",
            not first_start_problems(config_text, pointer_data, results, machine),
            "\n".join(first_start_problems(config_text, pointer_data, results, machine))
            or f"[machines.{machine}].writer = true, pointer current = {results}",
        ),
        Step(
            "(2) this machine is measured, its class read or answered, llmfit confirms both readings",
            len(profiles) == 1 and not measurement_problems(profile),
            "\n".join(measurement_problems(profile))
            or f"{profiles[0].name}: gpu {profile['gpu_state']}, {len(profile['gpus'])} card{'s' * (len(profile['gpus']) != 1)}, class "
            f"{profile['machine_class']} ({profile['machine_class_source']}), vram {profile['vram_gib']} GiB ({profile['vram_source']}), "
            f"ram {profile['ram_physical_gib']} GiB ({profile['ram_physical_source']})",
        ),
        Step(
            "(3) the search resolves a publisher model and its log groups the reasons of the rest",
            not search_problems(search_log, lines),
            "\n".join(search_problems(search_log, lines))
            or f"{search_log['resolved']} resolved, {len(search_log['accounts'])} pages asked, "
            f"budget {search_log['budget']['used']}/{search_log['budget']['limit']}: "
            + next(line.strip() for line in lines if "you can pick from" in line),
        ),
        Step(
            "(4) the ranking carries rule and context, Qwen3.5 is judged on its architecture",
            not ranking_problems(text, machine, block),
            "\n".join(ranking_problems(text, machine, block))
            or f"{len(block['ranked'])} ranked, {len(block['no_fit'])} no fit, "
            f"{len(block['not_covered'])} not covered; "
            f"{sum(1 for e in block['ranked'] + block['no_fit'] if e['fit']['basis'] == 'architecture')} "
            f"rows judged on the architecture",
        ),
    ]
    return steps, lines


def _load_test_step(results: Path, machine: str) -> Step:
    """Criterion 5: the prepared package was really measured here, and it is ranked with its speed.

    The measurement is the one part of this run that talks to the real local Ollama daemon --
    that is the gate. Everything else answers from the pinned fixtures.
    """
    records = _measurement_records(results)
    _text, payload = _document(results)
    row = _ranked_row(_machine_block(payload, machine), DEEPSEEK_GGUF, DEEPSEEK_QUANT)
    problems = load_test_problems(records, row)
    if problems:
        return Step("(5) the prepared package is measured and ranked", False, "\n".join(problems))
    record = records[0]
    detail = (f"{record['ollama_name']}: {record['tps_mean']:.1f} tok/s ({record['tps_min']:.1f}-{record['tps_max']:.1f}), context "
              f"{record['scenario']['context_requested']}, daemon {record['daemon_version']}, row {row['rank']} "
              f"({row_kind(row)}) in measurement group {row['measurement_group']}")
    return Step("(5) the prepared package is measured and ranked", True, detail)


def _offered_context(config_file: Path) -> int:
    """What the size scale of the next run starts at, from the configuration on disk."""
    from modelroom.config import load_config
    from modelroom.guided import context_default

    return context_default(load_config(config_file))


def _offered_level(config_file: Path) -> str:
    """Which line of the scale carries the pointer for that context -- the level, not the number."""
    from modelroom.guided_context import LEVELS, scale_choices

    choices = scale_choices(LEVELS, _offered_context(config_file), {}, None)
    return next((choice.value for choice in choices if choice.checked), "none")


def _scale_labels(results: Path, pointer: Path) -> list[str]:
    """The six level lines of the scale for this folder, built by the step's own functions.

    An answer file answers the question outright, so the list itself is never printed. It is
    built here from the same folder with the same functions the step uses -- the snapshot the run
    fetched, the profile the pointer binds this machine to, the reserves of the configuration.
    """
    from modelroom.config import load_config
    from modelroom.guided_context import LEVELS, context_cap, fit_texts, machine_checked, scale_choices, snapshot_packages

    config = load_config(results / "modelroom.toml")
    packages, base_models = snapshot_packages(config)
    checked = machine_checked(pointer, config, results)
    contexts = [level.tokens for level in LEVELS]
    fits = fit_texts(packages, {spec.hf_repo: spec for spec in base_models}, checked, contexts)
    choices = scale_choices(LEVELS, ANSWERED_CONTEXT, fits, context_cap(base_models))
    return [choice.label for choice in choices if choice.value in {level.name for level in LEVELS}]


def _model_list_labels(results: Path, pointer: Path, transport) -> list[str]:
    """The lines of step 2's selection list for this folder, built by the step's own functions.

    An answer file answers the question outright, so the list itself is never printed -- the same
    reason the scale is rebuilt here. The search runs once more against the very same pinned
    answers, with a budget of its own so this rebuild cannot spend the run's.
    """
    from modelroom.catalog import load_catalog
    from modelroom.config import load_config
    from modelroom.guided_context import machine_checked
    from modelroom.guided_models import list_choices, list_context, model_choices
    from modelroom.http import RequestBudget
    from modelroom.search import DEFAULT_GUIDED_BUDGET, DEFAULT_PACKAGERS, run_search

    config = load_config(results / "modelroom.toml")
    outcome = run_search(
        transport,
        SEARCH_NAME,
        catalog=load_catalog(),
        listed_packagers=config.packagers or DEFAULT_PACKAGERS,
        budget=RequestBudget(DEFAULT_GUIDED_BUDGET),
        open_pages=False,
    )
    checked = machine_checked(pointer, config, results)
    models = model_choices(
        outcome.hits, filtered=True, checked=checked, context=list_context(config.guided.context)
    )
    return [choice.label for choice in list_choices(models)]


def _standalone_render(results: Path, config_file: Path, now: datetime) -> tuple[int, str | None]:
    """`modelroom render --config <file>` on its own, and the `Scenario:` line it wrote."""
    from modelroom.cli import main

    code = main(["render", "--config", str(config_file)], now=now)
    header = next((line for line in _document(results)[0].splitlines() if line.startswith("Scenario: ")), None)
    return code, header


def _second_run(results: Path, transport, now: datetime, machine: str) -> list[Step]:
    """Criterion 6: the same command again reuses the configuration, the binding, the measurement
    and the context -- the last one also for a `modelroom render` that runs on its own afterwards."""
    config_file = results / "modelroom.toml"
    before = tomllib.loads(config_file.read_text(encoding="utf-8"))
    offered = _offered_context(config_file)
    level = _offered_level(config_file)
    answers = results.parent / "answers-second.toml"
    answers.write_text(answers_toml(ANSWERS_SECOND), encoding="utf-8", newline="\n")
    code, lines = _guided_run(answers, results, transport, now)
    after = tomllib.loads(config_file.read_text(encoding="utf-8"))
    names = [path.name for path in _profile_files(results)]
    records = _measurement_records(results)
    row = _ranked_row(_machine_block(_document(results)[1], machine), DEEPSEEK_GGUF, DEEPSEEK_QUANT)
    problems = second_start_problems(before, after, names, lines, records, row)
    render_code, header = _standalone_render(results, config_file, now + timedelta(minutes=1))
    if render_code != 0:
        problems.append(f"the standalone render ended with exit {render_code}")
    render_row = _ranked_row(_machine_block(_document(results)[1], machine), DEEPSEEK_GGUF, DEEPSEEK_QUANT)
    problems += stored_context_problems(before, offered, header, render_row, level)
    return [
        Step("guided run 2 ended with exit 0", code == 0, f"exit {code}"),
        Step(
            "(6) the second start reuses the configuration, the binding, the measurement and the context",
            not problems,
            "\n".join(problems)
            or f"one profile file ({names[0]}), one measurement still in group 0, no clone question, "
            f"[guided].context {offered} starts the scale again on level {level} and is rendered on its "
            f"own ({header})",
        ),
    ]


def _live_search_smoke() -> Step:
    """The live search, as a smoke: it says what the Hub answers today and decides nothing.

    Since 2026-09-24 a search costs one request per account, so the budget has to cover the
    catalog's matching publishers plus the whole positive list plus one age lookup per distinct
    resolved base model. `DEFAULT_GUIDED_BUDGET` is the number a real run has; using it here keeps
    the smoke from reporting a budget failure that no real run would hit.
    """
    from modelroom.catalog import load_catalog
    from modelroom.http import RequestBudget, UrllibTransport
    from modelroom.search import DEFAULT_GUIDED_BUDGET, run_search

    try:
        outcome = run_search(
            UrllibTransport(), SEARCH_NAME, catalog=load_catalog(), budget=RequestBudget(DEFAULT_GUIDED_BUDGET)
        )
    except Exception as exc:  # noqa: BLE001 -- a smoke never fails the run
        return Step("live search (smoke, no gate effect)", True, f"not reached: {type(exc).__name__}: {exc}")
    return Step(
        "live search (smoke, no gate effect)",
        True,
        " | ".join([*outcome.group_lines(), outcome.summary_line()]),
    )


def _testset_step() -> Step:
    """Criterion 8: the search test set, asked live -- one line per case, no effect on the exit code.

    Live because that is the only place the question can be answered: whether a person who types
    `qwen 3.5 9b` gets the model is a fact about the Hub's answer today, not about a fixture. A
    missing model is therefore a `WARN` and never a failure (decided 2026-09-25); what the tests
    gate is the shape of this report (`scripts/searchset.py`, `tests/test_selftest.py`).
    """
    from searchset import TestsetError, ask_testset, load_testset, testset_problems

    try:
        cases = load_testset()
    except TestsetError as exc:
        return Step("(8) the search test set, live (no gate effect)", False, str(exc))
    lines = ask_testset(cases)
    problems = testset_problems(cases, lines)
    return Step(
        "(8) the search test set, live (no gate effect)",
        not problems,
        "\n".join(problems or lines),
    )


def run_steps(work: Path) -> list[Step]:
    build_transport, guided_transport_mapping = _import_test_support()
    from selftest_entered import entered_steps
    from selftest_head import head_count_step
    from selftest_pull import pull_steps
    from selftest_remote import remote_step
    pointer = _redirect_home(work / "home")
    results = work / "results"
    results.mkdir(parents=True)
    transport = build_transport(guided_transport_mapping())
    started = datetime.now(timezone.utc).replace(microsecond=0)
    steps, first_lines = _first_run(results, pointer, transport, started)
    steps.append(Step(*head_count_step(results)))
    machine = _machine_name((results / "modelroom.toml").read_text(encoding="utf-8"))
    steps.append(_load_test_step(results, machine))
    scale = _scale_labels(results, pointer)
    models = _model_list_labels(results, pointer, build_transport(guided_transport_mapping()))
    steps += _second_run(results, build_transport(guided_transport_mapping()), started + timedelta(minutes=1), machine)
    steps.append(_guided_mode_step(first_lines, scale, models))
    steps.append(_testset_step())
    steps += [Step(*step) for step in pull_steps(results, pointer, machine, first_lines, started)]
    steps += [Step(*step) for step in entered_steps(work, answers_toml, started + timedelta(minutes=2))]
    steps.append(Step(*remote_step()))  # FR-B1: the gegenstelle, a real local process
    steps.append(_live_search_smoke())
    return steps


def _guided_mode_step(lines: list[str], scale: list[str], models: list[str]) -> Step:
    """Criterion 7: the dialog of the first run is the one the guided mode promises."""
    from selftest_dialog import guided_mode_problems  # the checks themselves (size guard, AP9-RM)

    problems = guided_mode_problems(lines, scale, models)
    detail = "\n".join(problems) or (
        "start screen, five step heads, scale: "
        + " / ".join(scale[:2])
        + f", {len(models)} models in the list: "
        + " | ".join(label.split("  ")[0] for label in models)
    )
    return Step("(7) the run reads as a guided dialog: start screen, steps, scale, model list", not problems, detail)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        work = Path(tempfile.mkdtemp(prefix="modelroom-selftest-"))
    except OSError as exc:
        print(f"[1] setup: FAIL\n    temp directory: {type(exc).__name__}: {exc}")
        print("SELFTEST: FAIL")
        return 2
    try:
        steps = run_steps(work)
    except SetupError as exc:
        print(f"[1] setup: FAIL\n    {exc}")
        print("SELFTEST: FAIL")
        return 2
    except (OSError, KeyError, ValueError, TypeError, StopIteration, IndexError) as exc:
        steps = [Step("the run did not finish", False, f"{type(exc).__name__}: {exc}")]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(format_report(steps))
    failed = [step for step in steps if not step.ok]
    print(f"SELFTEST: {'OK' if not failed else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
