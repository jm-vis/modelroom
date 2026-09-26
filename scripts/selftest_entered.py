"""Criterion 11 of `scripts/selftest.py`: a machine entered by hand, in a guided run of its own.

11. **A machine entered by hand.** One more guided run, isolated from the two of criteria 1 to 6:
   a results folder of its own and a pointer file of its own, so the first two runs keep their one
   profile file and their binding. It marks only `enter` and enters unified memory of 32 GiB from a
   data sheet. The document then has two blocks: this machine's own entry without a profile
   (`no_profile` -- the guided mode always writes it) and the machine entered by hand, ranked, named
   `(entered)`, computed on shared memory with one pool of 23 GiB (32 minus the reserves 8 + 1 of
   `[defaults]`), and every note of its rows `computed`. Gate-free: the fixtures answer the search
   and the fetch, and no daemon is asked -- this machine is not measured in this run at all.

Imported by `scripts/selftest.py`, which keeps the run and the report.
"""

from __future__ import annotations

import contextlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

ENTERED_POOL_GIB = 23.0
ENTERED_ANSWERS = {
    "results": "here",
    "machines": ["enter"],
    "entered_name": "studio",
    "entered_ram": 32,
    "entered_gpu": "unified",
    "search": "qwen",
    "filter_owners": True,
    "select": ["unsloth/Qwen3.5-9B-GGUF", "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"],
    "context": "S",
    "pull": False,
}
NAME = "(11) a machine entered by hand ranks on shared memory, every number computed"


def _entered_block(blocks: list[dict]) -> dict | None:
    return next(
        (block for block in blocks if (block.get("profile") or {}).get("ram_physical_source") == "entered"), None
    )


def _row_problems(block: dict) -> list[str]:
    """The rows of the machine entered by hand: computed, on shared memory, one pool of 23 GiB."""
    rows = block["ranked"] + block["too_tight"]
    problems = []
    if not block["ranked"]:
        problems.append("the machine entered by hand has no fit in its ranking")
    if any(row["note"]["origin"] != "computed" for row in rows):
        problems.append("a row of the machine entered by hand is not marked computed")
    # Unified memory computes in mode `gpu` only, with no fallback onto the same memory: every
    # ranked row is a shared-memory row, and each of them carries the one pool.
    shared = [
        row for row in block["ranked"] if row["fit"]["mode"] == "gpu" and row["note"]["code"] == "fits_in_shared_memory"
    ]
    if len(shared) != len(block["ranked"]):
        problems.append("not every ranked row of the machine entered by hand says shared memory")
    if {row["fit"]["pool_gib"] for row in shared} != {ENTERED_POOL_GIB}:
        problems.append(f"the shared pool is not {ENTERED_POOL_GIB} GiB (32 minus the reserves 8 + 1)")
    return problems


def entered_problems(payload: dict, local: str) -> list[str]:
    """Criterion 11 on the document: this machine without a profile, and the one entered by hand."""
    blocks = payload["machines"]
    problems = [] if len(blocks) == 2 else [f"machine blocks: {len(blocks)}, not two"]
    own = next((block for block in blocks if block["machine"] == local), None)
    if own is None or own["status"] != "no_profile":
        problems.append(f"this machine's own entry {local!r} is not in the document as no_profile")
    entered = _entered_block(blocks)
    if entered is None or entered["status"] != "ranked":
        return problems + ["the document has no block of a machine entered by hand that was ranked"]
    if not entered["label"].endswith("(entered)"):
        problems.append(f"the block is labeled {entered['label']!r}, without (entered)")
    if entered["profile"].get("gpu_state") != "unified_memory":
        problems.append(f"the machine entered by hand is {entered['profile'].get('gpu_state')!r}, not unified memory")
    return problems + _row_problems(entered)


def _run(work: Path, answers_toml: Callable[[dict], str], now: datetime) -> tuple[int, list[str], Path, Path]:
    """The isolated guided run: its own results folder, its own pointer, fixture sources only."""
    from fixture_support import build_transport, guided_transport_mapping, offline_daemon, windows_probes

    from modelroom.cli import main

    results = work / "entered" / "results"
    results.mkdir(parents=True)
    pointer = work / "entered" / "home" / ".modelroom" / "guided.json"
    answers = work / "entered" / "answers.toml"
    answers.write_text(answers_toml(ENTERED_ANSWERS), encoding="utf-8", newline="\n")
    lines: list[str] = []
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        code = main(
            ["--answers", str(answers)],
            transport=build_transport(guided_transport_mapping()),
            now=now,
            probes=windows_probes(host="workstation"),
            pointer_path=pointer,
            here=results,
            out=lines.append,
            daemon=offline_daemon(),
        )
    return code, lines + captured.getvalue().splitlines(), results, pointer


def entered_steps(work: Path, answers_toml: Callable[[dict], str], now: datetime) -> list[tuple]:
    """Criterion 11 as `(name, ok, detail)`, for `scripts/selftest.py` to report."""
    code, lines, results, pointer = _run(work, answers_toml, now)
    document = results / "docs" / "models.json"
    if code != 0 or not document.is_file():
        return [(NAME, False, f"the run ended with exit {code}: " + " | ".join(lines[-3:]))]
    problems = entered_problems(json.loads(document.read_text(encoding="utf-8")), "workstation")
    bindings = json.loads(pointer.read_text(encoding="utf-8")).get("bindings", {}) if pointer.is_file() else {}
    if bindings:
        problems.append(f"the pointer file binds a profile after a run that measured nothing: {bindings}")
    entered = _entered_block(json.loads(document.read_text(encoding="utf-8"))["machines"]) or {"ranked": []}
    pools = sorted({row["fit"]["pool_gib"] for row in entered["ranked"] if row["fit"]["mode"] == "gpu"})
    detail = (
        f"{entered.get('label')}: {len(entered['ranked'])} ranked, shared pool {pools} GiB, every row computed; "
        "this machine's own entry no_profile, no binding"
    )
    return [(NAME, not problems, "\n".join(problems) or detail)]
