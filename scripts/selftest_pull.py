"""Criteria 9 and 10 of `scripts/selftest.py`: the pull of step 5, and a run that does not pull.

9. **The pull, against a fixture daemon.** The first row of the self-test's own results folder is
   pulled through `guided_install.pull_step` from a `FixtureDaemon` that plays the recorded answer
   of a real pull (`tests/fixtures/ollama_pull_smollm2_stream.ndjson`) to its `success`, and lists
   the name afterwards. The screen has to carry the pull's first line and its closing line, and the
   daemon has to have been sent exactly that pull and read `/api/tags` after it. The real daemon of
   this machine is never asked to pull.
10. **`pull = false`.** The first live run answers `pull = false`: it pulls nothing, and the install
   line stands unchanged. Which of three things it did depends on the machine -- the question with
   `No`, the note that the row is local already, or nothing at all where the daemon did not answer
   the start screen -- and each of them passes.

Both are imported by `scripts/selftest.py`, which keeps the run and the report.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PULLED_WORD = "Pulled    "
PULL_QUESTION = "? Pull #"
INSTALL_LINE = "install   #1  ollama pull "


def pulled_problems(lines: list[str], calls: list[tuple], name: str) -> list[str]:
    """Criterion 9: the pull's two lines on the screen, the pull sent, the list read after it."""
    stripped = [line.strip() for line in lines]
    problems = []
    if f"pulling   {name}" not in stripped:
        problems.append(f"the screen has no line that starts the pull of {name}")
    if not any(PULLED_WORD + name + " " in line for line in stripped):
        problems.append(f"the screen has no `Pulled` line for {name}")
    pulls = [index for index, call in enumerate(calls) if call[:2] == ("POST", "/api/pull")]
    if [calls[index][2] for index in pulls] != [{"model": name, "stream": True}]:
        problems.append(f"the daemon was not sent exactly one POST /api/pull for {name}")
    elif not any(call[:2] == ("GET", "/api/tags") for call in calls[pulls[0] + 1 :]):
        problems.append("the daemon's /api/tags was not read after the pull")
    return problems


def declined_problems(lines: list[str]) -> list[str]:
    """Criterion 10: a run that answered `pull = false` pulled nothing and kept its install line."""
    stripped = [line.strip() for line in lines]
    problems = []
    if not any(line.startswith(INSTALL_LINE) for line in stripped):
        problems.append("the install line of the first row is gone")
    if any(line.startswith(PULL_QUESTION) and line.endswith("  Yes") for line in stripped):
        problems.append("the pull question was answered with Yes")
    if any(line.startswith("pulling ") or PULLED_WORD in line for line in stripped):
        problems.append("the run pulled a package although it was told not to")
    return problems


def _declined_detail(lines: list[str]) -> str:
    stripped = [line.strip() for line in lines]
    for line in stripped:
        if line.startswith(PULL_QUESTION) or " is local already " in line:
            return line
    return "no question: the start screen did not reach a daemon, or there was no first row to pull"


def pull_steps(results: Path, pointer: Path, machine: str, first_lines: list[str], now: datetime) -> list[tuple]:
    """Criteria 9 and 10 as `(name, ok, detail)`, for `scripts/selftest.py` to report."""
    problems, detail = _fixture_pull(results, pointer, machine, now)
    declined = declined_problems(first_lines)
    return [
        ("(9) the pull of step 5 runs to success against the recorded answer", not problems, "\n".join(problems) or detail),
        ("(10) pull = false pulls nothing and keeps the install line", not declined, "\n".join(declined) or _declined_detail(first_lines)),
    ]


def _fixture_pull(results: Path, pointer: Path, machine: str, now: datetime) -> tuple[list[str], str]:
    """Pull the first row of `results` from a fixture daemon; the problems, and what was pulled."""
    from fixture_support import build_transport, pull_daemon, tags_with

    from modelroom.catalog import load_catalog
    from modelroom.config import load_config
    from modelroom.dialog import FileAsker
    from modelroom.document import RenderDocument
    from modelroom.guided import GuidedRun
    from modelroom.guided_context import snapshot_packages
    from modelroom.guided_install import first_row, pull_step
    from modelroom.http import Response
    from modelroom.measure import Probes
    from modelroom.screen import Screen

    config = load_config(results / "modelroom.toml")
    payload = json.loads(config.paths.markdown.with_suffix(".json").read_text(encoding="utf-8"))
    first = first_row(RenderDocument.model_validate(payload), machine, snapshot_packages(config)[0])
    if first is None:
        return [f"the document of the first run has no first row with an Ollama name for {machine}"], ""
    # Nothing is installed before the pull, so the row is asked for and pulled whatever it is.
    daemon = pull_daemon(tags=Response(status=200, body=b'{"models": []}'), after_pull=tags_with(first.name))
    lines: list[str] = []
    run = GuidedRun(
        asker=FileAsker({"pull": True}),
        here=results,
        pointer_path=pointer,
        transport=build_transport({}),
        probes=Probes(),
        now=now,
        catalog=load_catalog(),
        daemon=daemon,
        out=lines.append,
        screen=Screen(lines.append, colored=False),
    )
    pull_step(run, first, True)
    problems = pulled_problems(lines, daemon.calls, first.name) + list(run.problems)
    return problems, f"{first.name}: {len(lines)} lines, last: {lines[-1].strip() if lines else '-'}"
