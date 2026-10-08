"""Criterion 7 of `scripts/selftest.py`: the run reads as the guided dialog it is.

The checks of the screen -- start screen, step heads, answer lines, card, model list and result
table -- moved here unchanged from `scripts/selftest.py` (size guard, AP9-RM); that script keeps
the run and the report and imports `guided_mode_problems` from here.
"""

from __future__ import annotations

import re


def guided_mode_problems(lines: list[str], scale: list[str], models: list[str] | None = None) -> list[str]:
    """Criterion 7: the run reads like the guided mode it is -- the screen of 2026-09-24.

    What the test round of 2026-09-24 called "a record, not a report": every step is a block with a
    head of its own, every question leaves one line behind with its answer in words, the notes are
    bundled instead of repeated, and the run closes with a card of six rows over the result table.
    """
    problems = []
    if models is not None:
        problems += model_list_problems(models)
    if not any("ModelRoom" in line for line in lines[:6]):
        problems.append("the run does not begin with the start screen")
    for label in ("folder", "daemon", "machine"):
        if not any(line.strip().startswith(label) for line in lines[:12]):
            problems.append(f"the start screen has no {label} row")
    problems += step_head_problems(lines)
    problems += answer_line_problems(lines)
    problems += card_problems(lines)
    if not all(" packages fit" in label or " packages fits" in label for label in scale):
        problems.append(f"a level of the scale carries no count of what fits: {scale}")
    problems += result_table_problems("\n".join(lines).splitlines())
    return problems


def step_head_problems(lines: list[str]) -> list[str]:
    """Criterion 7: five heads, each a rule of 72 characters around the step and its name."""
    heads = [line.strip() for line in lines if line.strip().startswith(("-- Step ", "── Step "))]
    expected = [f"Step {number} of 5" for number in range(1, 6)]
    numbered = [head.split("  ")[0].split(" ", 1)[1] for head in heads]
    if numbered != expected:
        problems = [f"the five step heads are {numbered}, expected {expected}"]
        return problems
    wrong = [head for head in heads if len(head) != 72]
    return [f"a step head is {len(wrong[0])} characters wide, expected 72: {wrong[0]}"] if wrong else []


def answer_line_problems(lines: list[str]) -> list[str]:
    """Criterion 7: the run writes one answer line per question, and the library writes none."""
    answers = [line.strip() for line in lines if line.strip().startswith("? ")]
    problems = []
    if len(answers) < 4:
        problems.append(f"{len(answers)} answer lines, expected one per question asked: {answers}")
    if any(line.endswith("?") for line in answers):
        problems.append("an answer line carries no answer")
    if any("done (" in line or " selections)" in line for line in lines):
        problems.append("the dialog library wrote an answer of its own")
    for gone in ("as the writer of this results folder", "kept context ", "checking ", "Written to "):
        if any(gone in line for line in lines):
            problems.append(f"a line the screen no longer carries is still printed: {gone!r}")
    return problems


def card_problems(lines: list[str]) -> list[str]:
    """Criterion 7: the run closes with the card -- six kinds of row, each with its own note."""
    labels = ("folder", "machine", "models", "context", "speed", "result")
    # The card is what follows its own first row (`folder`, the last one the run printed): a window of the last
    # lines would grow with every row the table above it gains and could take an old row for a missing one.
    stripped = [line.strip() for line in lines]
    starts = [index for index, line in enumerate(stripped) if line.startswith("folder ")]
    tail = stripped[starts[-1] :] if starts else []
    missing = [label for label in labels if not any(line.startswith(f"{label} ") for line in tail)]
    if missing:
        return [f"the card of step 5 has no {', '.join(missing)} row"]
    return []


def model_list_problems(models: list[str]) -> list[str]:
    """Criterion 7, the selection list of step 2: one line per model, each with its fit in a word.

    The list of 2026-09-24 had one line per repository, over 200 characters wide, `unknown` where
    a size belongs and no fit at all; this one has a model per line and says what it would cost
    (CONTRACTS.md, "Guided mode", step 2).
    """
    problems = []
    if not models:
        problems.append("the selection list of step 2 holds no model")
    for label in models:
        if len(label) > 100:
            problems.append(f"a line of the model list is {len(label)} characters wide: {label}")
        if not any(word in label for word in ("fits", "tight", "no_fit", "unknown")):
            problems.append(f"a line of the model list carries no fit: {label}")
    return problems


# AP9-RM U10: the pieces of a `room` block, the form and never the numbers. A budget is a number
# with one decimal, or said in words where nothing is left.
_BUDGET = r"(?:\d+\.\d GB|nothing left) for models"
_CARDS = r"(?:\d+\.\d GB card|\d+ x \d+\.\d GB cards|\d+\.\d(?: \+ \d+\.\d)+ GB cards|\d+ cards, \d+\.\d GB in all)"
_ROOM_PIECES = {
    "graphics": rf"graphics (?:{_BUDGET} \({_CARDS} minus reserve\)|none counted|unknown)",
    "memory": rf"memory (?:{_BUDGET} \(\d+\.\d GB minus reserve\)|unknown)",
    "shared": rf"shared memory (?:{_BUDGET} \(\d+\.\d GB minus both reserves\)|unknown)",
    # AP9-LS-E: a machine that names LM Studio (or vLLM) says whose models the place holds, one name or several.
    "disk": r"disk (?:\d+\.\d GB free on .+ \((?:Ollama|LM Studio|vLLM)(?:(?:, | and )(?:Ollama|LM Studio|vLLM))* models\)|unknown(?: .+)?)",
}


def room_blocks(lines: list[str]) -> list[str]:
    """Each `room` block as one text: the labeled line and the lines that continue it (no label)."""
    blocks, current = [], None
    for line in lines:
        if line.strip().startswith("room "):
            current = line.strip()[len("room ") :].strip()
            blocks.append(current)
        elif current is not None and line.startswith("  ") and line.strip():
            blocks[-1] = f"{blocks[-1]} · {line.strip()}"
        else:
            current = None
    return blocks


def room_problems(lines: list[str]) -> list[str]:
    """Every `room` block: graphics and memory, or shared memory, then the disk; nothing else, no `free now`."""
    blocks = room_blocks(lines)
    if not blocks:
        return ["the table head carries no room line"]
    problems = []
    for block in blocks:
        pieces = re.split(r" [·.-] ", block)
        kinds = [next((kind for kind, form in _ROOM_PIECES.items() if re.fullmatch(form, piece)), None) for piece in pieces]
        if None in kinds or kinds not in (["graphics", "memory", "disk"], ["shared", "disk"]):
            problems.append(f"a room line is not graphics and memory (or shared memory) and disk: {block}")
    return problems


def result_table_problems(lines: list[str]) -> list[str]:
    """Criterion 7, the result table: the columns of the mockup, and every note said once."""
    problems = []
    head = next((line for line in lines if "Model" in line and "Package" in line and "Fit" in line), None)
    if head is None:
        problems.append("the result view has no table head with Model, Package and Fit")
    elif head.split() != ["#", "Model", "Package", "Fit", "Speed", "Memory", "Download"]:
        problems.append(f"the columns of the result table are {head.split()}")
    problems += room_problems(lines)
    if not any(line.strip().startswith("shown ") for line in lines):
        problems.append("the result table says not how many of its rows are shown")
    if not any(line.strip().startswith("install ") for line in lines):
        problems.append("the result table carries no install line for the machine of this run")
    pooled = [line for line in _outside_key_block(lines) if ("graphics memory" in line or "system memory" in line) and not line.strip().startswith("graphics  ")]  # an adapter line (AP9-GPU) is no pool note
    if pooled and not all("#" in line for line in pooled):
        problems.append(f"a note about the memory pool names no rank: {pooled}")
    if len(pooled) > 3:
        problems.append(f"{len(pooled)} notes about the memory pool, expected one per pool at most")
    if any(" unknown " in line for line in lines if line.strip().startswith(("1 ", "2 ", "3 "))):
        problems.append("a row of the result table says `unknown` where a dash belongs")
    return problems


def _outside_key_block(lines: list[str]) -> list[str]:
    """The lines minus the `key` block of the card (the hint of 2026-09-27 names the pools in words):
    from the `key` label to the next label, where a label is the word at column 1."""
    kept, in_key = [], False
    for line in lines:
        labeled = len(line) > 1 and line[0] == " " and line[1] != " "
        if labeled:
            in_key = line.strip().startswith("key ")
        if not in_key:
            kept.append(line)
    return kept
