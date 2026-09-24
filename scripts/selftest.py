"""The acceptance run of the guided mode, one step per criterion:

    uv run --frozen python scripts/selftest.py

It runs the guided mode twice with `--answers` in a temporary results folder, against the
pinned search answer from `tests/fixture_support.py` (criteria 3 and 4 must not depend on
whichever repositories the live Hub answers with today), and reports each criterion with the
evidence it read. Two things in the run are real, and they are the two gates: this machine is
measured with its own sources (criterion 2), and the load test runs against the Ollama daemon
of this machine (criterion 5). Everything else answers from the fixtures.

Criteria (the plan's own numbering):

1. the first start writes `modelroom.toml` with this device as the writer, and remembers the
   results folder in the pointer file;
2. this machine is measured: `gpu_state: measured`, llmfit cross-check `confirmed`;
3. the search resolves at least one publisher model and shows the unresolved hits with a reason;
4. the ranking carries the rule and the context in its header, and Qwen3.5 stands under
   "not covered";
5. the prepared package is measured against the real local Ollama daemon: the measurement is
   valid and comparable, its file lies under `<state>/measurements/<profile_id>/`, and the
   package stands in the ranking in measurement group 0 with its speed;
6. the second start reuses the configuration, the binding and the measurement: no clone
   question, no second profile file, no second measurement.

The live search runs as a smoke afterwards and never decides the exit code.

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
QWEN_BASE = "Qwen/Qwen3.5-9B"
UNSLOTH_GGUF = "unsloth/Qwen3.5-9B-GGUF"
# The prepared package of criterion 5: the one repository in the pinned search answer whose base
# model fit v1 can judge, and whose `Q4_K_M` build has to be installed on this machine. The load
# test itself runs against the real local daemon -- that is what criterion 5 is about.
DEEPSEEK_GGUF = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
DEEPSEEK_OLLAMA_NAME = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
ANSWERS_FIRST = {
    "results": "here",
    "machines": ["this-machine"],
    "search": SEARCH_NAME,
    "filter_owners": True,
    "select": [UNSLOTH_GGUF, DEEPSEEK_GGUF],
    "context": "8192",
    "load_test": True,
    "load_test_packages": [DEEPSEEK_OLLAMA_NAME],
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
    if raw.get("schema_version") != 2:
        problems.append(f"schema_version is {raw.get('schema_version')!r}, expected 2")
    entry = raw.get("machines", {}).get(machine)
    if entry is None:
        problems.append(f"the configuration has no [machines.{machine}]")
    elif entry.get("writer") is not True:
        problems.append(f"[machines.{machine}].writer is {entry.get('writer')!r}, expected true")
    if Path(raw.get("guided", {}).get("results", "")) != results:
        problems.append(f"[guided].results is {raw.get('guided', {}).get('results')!r}, expected {results}")
    if Path(pointer.get("current", "")) != results:
        problems.append(f"the pointer file's current is {pointer.get('current')!r}, expected {results}")
    if pointer.get("bindings", {}).get(str(results)) is None:
        problems.append(f"the pointer file binds no profile to {results}")
    return problems


def measurement_problems(profile: dict) -> list[str]:
    """Criterion 2: this machine was really measured, and llmfit confirmed both readings."""
    problems = []
    if profile.get("gpu_state") != "measured":
        problems.append(f"gpu_state is {profile.get('gpu_state')!r}, expected 'measured'")
    if profile.get("origin") != "measured":
        problems.append(f"origin is {profile.get('origin')!r}, expected 'measured'")
    for field in ("ram_physical", "vram"):
        status = profile.get("llmfit_crosscheck", {}).get(field, {}).get("status")
        if status != "confirmed":
            problems.append(f"llmfit cross-check of {field} is {status!r}, expected 'confirmed'")
    return problems


def search_problems(lines: list[str]) -> list[str]:
    """Criterion 3: at least one resolved publisher model, every unresolved hit with a reason."""
    summary = next((line for line in lines if " repositories, " in line and " resolved, " in line), None)
    if summary is None:
        return ["no search summary line was printed"]
    problems = []
    resolved = int(summary.split(" repositories, ")[1].split(" resolved")[0])
    if resolved < 1:
        problems.append(f"the search resolved {resolved} repositories, expected at least one")
    unresolved = [line for line in lines if line.startswith("unresolved ")]
    if not unresolved:
        problems.append("no unresolved hit was shown with its reason")
    if any(line.rstrip().endswith(":") or line.rstrip().endswith("None") for line in unresolved):
        problems.append("an unresolved hit was shown without a reason")
    return problems


def ranking_problems(text: str, machine: str, not_covered: list[dict]) -> list[str]:
    """Criterion 4: the header carries rule and context, and Qwen3.5 stands under not covered."""
    lines = text.splitlines()
    problems = []
    for expected in ("Ranking rule: ", "Scenario: context 8192 (default)", f"## Ranking: {machine}"):
        if not any(line.startswith(expected) for line in lines):
            problems.append(f"the document has no line starting with {expected!r}")
    if not not_covered:
        problems.append("the not-covered block of this machine is empty")
    if any(entry["base_model_hf_repo"] != QWEN_BASE for entry in not_covered):
        problems.append(f"the not-covered block names something other than {QWEN_BASE}")
    if any(entry["reason"] != "architecture not covered by v1" for entry in not_covered):
        problems.append(f"a not-covered reason is not the fit rule's own: {[e['reason'] for e in not_covered]}")
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
    if row is None:
        problems.append(f"{DEEPSEEK_GGUF} stands in no ranking of this machine")
    elif row["measurement_group"] != 0 or row["speed_tps"] is None:
        problems.append(f"the ranked row is group {row['measurement_group']} with speed {row['speed_tps']!r}")
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
    if any(line.startswith(f"{DEEPSEEK_OLLAMA_NAME}: measured ") for line in lines):
        problems.append("the second run measured the package a second time")
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


def _ranked_row(block: dict, repo: str) -> dict | None:
    """The row of one packaging repository in a machine's block, wherever it was sorted to."""
    rows = block["ranked"] + block["too_tight"]
    return next((row for row in rows if row["package_identity"][1] == repo), None)


def _machine_name(config_text: str) -> str:
    """The one machine the guided mode wrote into the configuration: this device."""
    machines = tomllib.loads(config_text)["machines"]
    return next(name for name, entry in machines.items() if entry.get("writer"))


def _first_run(results: Path, pointer: Path, transport, now: datetime) -> list[Step]:
    """Criteria 1 to 4, from one guided run and the files it left behind."""
    answers = results.parent / "answers-first.toml"
    answers.write_text(answers_toml(ANSWERS_FIRST), encoding="utf-8", newline="\n")
    code, lines = _guided_run(answers, results, transport, now)
    config_text = (results / "modelroom.toml").read_text(encoding="utf-8")
    machine = _machine_name(config_text)
    pointer_data = json.loads(pointer.read_text(encoding="utf-8"))
    profiles = _profile_files(results)
    profile = json.loads(profiles[0].read_text(encoding="utf-8")) if profiles else {}
    text, payload = _document(results)
    block = _machine_block(payload, machine)
    return [
        Step("guided run 1 ended with exit 0", code == 0, f"exit {code}, {len(lines)} lines printed"),
        Step(
            "(1) first start writes the configuration and remembers the folder",
            not first_start_problems(config_text, pointer_data, results, machine),
            "\n".join(first_start_problems(config_text, pointer_data, results, machine))
            or f"[machines.{machine}].writer = true, pointer current = {results}",
        ),
        Step(
            "(2) this machine is measured, llmfit confirms both readings",
            len(profiles) == 1 and not measurement_problems(profile),
            "\n".join(measurement_problems(profile))
            or f"{profiles[0].name}: gpu_state {profile['gpu_state']}, vram {profile['vram_gib']} GiB "
            f"({profile['vram_source']}), ram {profile['ram_physical_gib']} GiB ({profile['ram_physical_source']})",
        ),
        Step(
            "(3) the search resolves a publisher model and shows the unresolved hits",
            not search_problems(lines),
            "\n".join(search_problems(lines))
            or next(line for line in lines if " repositories, " in line),
        ),
        Step(
            "(4) the ranking carries rule and context, Qwen3.5 is not covered",
            not ranking_problems(text, machine, block["not_covered"]),
            "\n".join(ranking_problems(text, machine, block["not_covered"]))
            or f"{len(block['ranked'])} ranked, {len(block['not_covered'])} not covered "
            f"({block['not_covered'][0]['reason']})",
        ),
    ]


def _load_test_step(results: Path, machine: str) -> Step:
    """Criterion 5: the prepared package was really measured here, and it is ranked with its speed.

    The measurement is the one part of this run that talks to the real local Ollama daemon --
    that is the gate. Everything else answers from the pinned fixtures.
    """
    records = _measurement_records(results)
    _text, payload = _document(results)
    row = _ranked_row(_machine_block(payload, machine), DEEPSEEK_GGUF)
    problems = load_test_problems(records, row)
    if problems:
        return Step("(5) the prepared package is measured and ranked", False, "\n".join(problems))
    record = records[0]
    detail = (
        f"{record['ollama_name']}: {record['tps_mean']:.1f} tok/s "
        f"({record['tps_min']:.1f}-{record['tps_max']:.1f}), context "
        f"{record['scenario']['context_requested']}, daemon {record['daemon_version']}, "
        f"rank {row['rank']} in measurement group {row['measurement_group']}"
    )
    return Step("(5) the prepared package is measured and ranked", True, detail)


def _second_run(results: Path, transport, now: datetime, machine: str) -> list[Step]:
    """Criterion 6: the same command again reuses the configuration, the binding and the measurement."""
    before = tomllib.loads((results / "modelroom.toml").read_text(encoding="utf-8"))
    answers = results.parent / "answers-second.toml"
    answers.write_text(answers_toml(ANSWERS_SECOND), encoding="utf-8", newline="\n")
    code, lines = _guided_run(answers, results, transport, now)
    after = tomllib.loads((results / "modelroom.toml").read_text(encoding="utf-8"))
    names = [path.name for path in _profile_files(results)]
    records = _measurement_records(results)
    row = _ranked_row(_machine_block(_document(results)[1], machine), DEEPSEEK_GGUF)
    problems = second_start_problems(before, after, names, lines, records, row)
    return [
        Step("guided run 2 ended with exit 0", code == 0, f"exit {code}"),
        Step(
            "(6) the second start reuses the configuration, the binding and the measurement",
            not problems,
            "\n".join(problems)
            or f"one profile file ({names[0]}), one measurement still in group 0, no clone question",
        ),
    ]


def _live_search_smoke() -> Step:
    """The live search, as a smoke: it says what the Hub answers today and decides nothing."""
    from modelroom.catalog import load_catalog
    from modelroom.http import RequestBudget, UrllibTransport
    from modelroom.search import run_search

    try:
        outcome = run_search(
            UrllibTransport(), SEARCH_NAME, catalog=load_catalog(), budget=RequestBudget(12)
        )
    except Exception as exc:  # noqa: BLE001 -- a smoke never fails the run
        return Step("live search (smoke, no gate effect)", True, f"not reached: {type(exc).__name__}: {exc}")
    return Step("live search (smoke, no gate effect)", True, outcome.summary_line())


def run_steps(work: Path) -> list[Step]:
    build_transport, guided_transport_mapping = _import_test_support()
    pointer = _redirect_home(work / "home")
    results = work / "results"
    results.mkdir(parents=True)
    transport = build_transport(guided_transport_mapping())
    started = datetime.now(timezone.utc).replace(microsecond=0)
    steps = _first_run(results, pointer, transport, started)
    machine = _machine_name((results / "modelroom.toml").read_text(encoding="utf-8"))
    steps.append(_load_test_step(results, machine))
    steps += _second_run(
        results, build_transport(guided_transport_mapping()), started + timedelta(minutes=1), machine
    )
    steps.append(_live_search_smoke())
    return steps


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
