"""The two text views of a `document.RenderDocument`: Markdown and the terminal.

Both are pure formatters -- they read the document `modelroom/render.py::build_render_document`
produced and add no number of their own, so the Markdown file, the JSON file and what the
guided mode prints can never disagree. The JSON view is the document itself
(`render.document_json`). See CONTRACTS.md, "Render (schema 2)".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

from .contracts import Area, Fit, Rating
from .document import MachineRanking, RankedEntry, RenderDocument, SetAsideEntry
from .intro import LABEL_COLUMN, STEP_COUNT, glyphs, label_line
from .measurements import Scenario
from .pull_choice import collision_pieces, entry_pull_name, repo_lines
from .ranking_words import key_pieces, showing_line, shown_pieces
from .render import format_header_line
from .result_card import _DASH, _folder_is_empty, _model_name, context_fact, context_short, empty_card
from .result_card import relative_path, result_card
from .scenario_text import document_scenario, load_hint, never_measured, scenario_line, unused_measurements
from .screen import (
    JOIN_WIDTH,
    MEASURE_HINT,
    NO_FIT,
    Line,
    deviation_words,
    fit_word,
    head_line,
    install_pull_line,
    joined,
    plain,
    ranks_text,
)

_UNKNOWN = "unknown"
# The terminal view is what the last step of the guided mode shows, so it carries that step's head.
RESULTS_STEP = STEP_COUNT


# --- shared wording ------------------------------------------------------------------------------


def _cell(value: str) -> str:
    """Sanitize one Markdown table cell's external text (R7-11, fix-round 6).

    Applied to every cell that can carry text this package does not itself control the shape of
    -- an area's error message, a base model or packager repo name, a quantization/format label,
    a profile's `gpu_name`, a fit or set-aside reason, a note, a `RatingSource`'s error message.
    Escapes `|` (a literal pipe would otherwise read as a new column boundary) and collapses any
    `\\r\\n`/`\\n`/`\\r` to a single space (a raw newline would otherwise split one logical row
    across multiple physical lines). Probe: an `Area.error` of `"boom | extra\\nsecond line"`
    produced 4 physical lines and an extra column in the areas table instead of the one row it
    should have been.
    """
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    return normalized.replace("|", "\\|").strip()


def _fit_text(fit: Fit) -> str:
    return f"{fit.fit_class} ({fit.mode})"


def _stars(rating: Rating | None) -> str:
    if rating is None:
        return _DASH
    full_stars, half = divmod(round(rating.stars * 2), 2)
    return "★" * full_stars + ("½" if half else "")


def _package_context(entry) -> str:
    return _UNKNOWN if entry.package_context is None else str(entry.package_context)


# --- writer: Markdown ----------------------------------------------------------------------------

_RANKED_COLUMNS = (
    "#",
    "Base model",
    "Stars",
    "Packager",
    "Quant",
    "Format",
    "Weights GiB (computed)",
    "Fit (computed)",
    "Need GiB (computed)",
    "Pool GiB (computed)",
    "Package context",
    "Measured group",
    "Speed tok/s (measured)",
    "Provenance",
    "Note",
)
_SET_ASIDE_COLUMNS = (
    "Base model",
    "Stars",
    "Packager",
    "Quant",
    "Format",
    "Weights GiB (computed)",
    "Package context",
    "Reason",
    "Note",
)
_MACHINE_COLUMNS = (
    "Machine", "Profile", "Origin", "RAM GiB", "RAM source", "VRAM GiB", "VRAM source", "GPU state", "GPUs", "Class",
    "Reserve RAM GiB", "Reserve VRAM GiB", "Recorded at", "Profile age (days)",
)
_AREA_COLUMNS = ("Source", "Base model", "Packager", "Status", "Last success", "Error")
# One hint line under every table whose column has words of its own, in plain words and one short
# clause per word (decided 2026-09-27). The fit's own is `screen.FIT_HINT`, shared with the terminal.
SPEED_HINT = "Speed is measured, see its column."
GPU_STATE_HINT = (
    "GPU state: measured = read from nvidia-smi; entered = typed in by hand; multi_gpu = several cards, "
    "added up; present_unmeasured = a card may be there, its memory was not read; none = no usable card "
    "(none found, display adapter only, or --cpu-only); unified_memory = system and card share one "
    "memory; unsupported_platform = not read on this platform; multi_gpu_not_covered = several cards, "
    "measured before the fit covered them; legacy_unknown = from a schema-1 profile"
)
CLASS_HINT = (
    "Class: laptop, workstation or server, as the chassis reports it (chassis) or as the user gave it "
    "(entered); unknown where neither says. It changes no number."
)


def _table(columns: tuple[str, ...], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
    return "\n".join([head, *("| " + " | ".join(row) + " |" for row in rows)])


def _area_row(area: Area) -> list[str]:
    return [
        area.source,
        _cell(area.base_model_hf_repo),
        _cell(area.packager) if area.packager else _DASH,
        area.status,
        area.last_success.isoformat() if area.last_success else _DASH,
        _cell(area.error) if area.error else "",
    ]


def _machine_row(block: MachineRanking, rendered_at: datetime) -> list[str]:
    """One row of the Machines table; a machine without a schema-2 profile shows its status.

    Its readings are deliberately not shown: a schema-1 file's numbers came from llmfit under
    the old schema and no longer say what schema 2 measures, and a machine without a profile has
    no readings at all. The status word and the reason next to the machine's own ranking say
    what to do (`modelroom migrate`, then `modelroom hardware`).
    """
    profile = block.profile
    reserves = [f"{block.reserve_ram_gib:.2f}", f"{block.reserve_vram_gib:.2f}"]
    if profile is None:
        return [
            _cell(block.machine),
            _cell(block.label),
            *([block.status] * 8),
            *reserves,
            block.status,
            block.status,
        ]
    return [
        _cell(block.machine),
        # A machine entered by hand says so behind its name (decided 2026-09-26).
        _cell(f"{profile.display_name}{' (entered)' if profile.ram_physical_source == 'entered' else ''} ({profile.profile_id})"),
        profile.origin,
        _number(profile.ram_physical_gib),
        profile.ram_physical_source,
        _number(profile.vram_gib),
        profile.vram_source,
        profile.gpu_state,
        _cell("; ".join(f"{gpu.index} {gpu.name or 'graphics card'} {gpu.vram_gib:.2f}" for gpu in profile.gpus) or _DASH),
        f"{profile.machine_class} ({profile.machine_class_source})",
        *reserves,
        profile.recorded_at.isoformat(),
        str((rendered_at - profile.recorded_at).days),
    ]


def _number(value: float | None) -> str:
    return _UNKNOWN if value is None else f"{value:.2f}"


def _ranked_row(entry: RankedEntry, document: RenderDocument) -> list[str]:
    return [
        str(entry.rank),
        _cell(entry.base_model_hf_repo),
        _stars(document.ratings.get(entry.base_model_hf_repo)),
        _cell(entry.packager),
        _cell(entry.quantization),
        _cell(entry.format),
        f"{entry.weights_gib:.2f}",
        _fit_text(entry.fit),
        f"{entry.fit.need_gib:.2f}",
        f"{entry.fit.pool_gib:.2f}",
        _package_context(entry),
        str(entry.measurement_group),
        _DASH if entry.speed_tps is None else f"{entry.speed_tps:.1f}",
        entry.provenance,
        _cell(entry.note.text),
    ]


def _set_aside_row(entry: SetAsideEntry, document: RenderDocument) -> list[str]:
    return [
        _cell(entry.base_model_hf_repo),
        _stars(document.ratings.get(entry.base_model_hf_repo)),
        _cell(entry.packager),
        _cell(entry.quantization),
        _cell(entry.format),
        f"{entry.weights_gib:.2f}",
        _package_context(entry),
        _cell(entry.reason),
        _cell(entry.note.text),
    ]


def _summary_lines(document: RenderDocument) -> list[str]:
    lines = [
        "# Model packages",
        "",
        f"Snapshot run at: {document.snapshot_run_at.isoformat()}",
        f"Rendered at: {document.rendered_at.isoformat()}",
        f"Base models / packages: {document.base_model_count} / {document.package_count}",
        f"Scenario: {document_scenario(document)}",
        *(f"Load: {hint}" for hint in load_hint(document)),
        f"Ranking rule: {document.ranking_rule}",
    ]
    if document.rating_unavailable is not None:
        lines.append(f"Market rating unavailable: {_cell(document.rating_unavailable)}")
    return lines


def _machine_sections(block: MachineRanking, document: RenderDocument) -> list[str]:
    """One machine's sections: its ranking, then what was set aside, each as its own heading."""
    headline = f"{block.label} -- {block.status}{_deviation(block)}"
    if block.reason is not None:
        headline = f"{headline}: {_cell(block.reason)}"
    sections = [f"## Ranking: {block.machine}", "", headline, ""]
    if block.ranked:
        repos = repo_lines(block)
        sections += [
            showing_line(block),
            "",
            _table(_RANKED_COLUMNS, [_ranked_row(entry, document) for entry in block.ranked]),
            "",
            *([_cell(" · ".join(repos)), ""] if repos else []),
            " · ".join(key_pieces(block)),
            "",
            SPEED_HINT,
            "",
        ]
    else:
        sections += ["No package of the configured base models is ranked on this machine.", ""]
    for heading, items in (("Not covered", block.not_covered), (NO_FIT.capitalize(), block.no_fit)):
        if not items:
            continue
        sections += [
            f"## {heading}: {block.machine}",
            "",
            _table(_SET_ASIDE_COLUMNS, [_set_aside_row(entry, document) for entry in items]),
            "",
        ]
    return sections


def document_markdown(document: RenderDocument) -> str:
    """The Markdown view of `document`: the same numbers the JSON and the terminal view show."""
    sections = [
        format_header_line(document.snapshot_run_at, document.rendered_at),
        "",
        *_summary_lines(document),
        "",
        "## Areas",
        "",
        _table(_AREA_COLUMNS, [_area_row(area) for area in document.areas]),
        "",
        "## Machines",
        "",
        _table(_MACHINE_COLUMNS, [_machine_row(block, document.rendered_at) for block in document.machines]),
        "",
        GPU_STATE_HINT,
        "",
        CLASS_HINT,
        "",
    ]
    for block in document.machines:
        sections += _machine_sections(block, document)
    return "\n".join(sections)


# --- writer: terminal ----------------------------------------------------------------------------

# The result table as the mockup of 2026-09-24 draws it: what a reader recognizes a package by
# (its model and its packager) before what a fit is made of. 99 characters wide plus the one space
# every line of this view is indented by, so it fits a 100-column window without wrapping.
# Since AP9-UX2 (U8) the table carries the download of each row, the weight files of its package:
# `Fit` gave up 8 (its longest word, `tight (RAM)`, is 11) and `Model` 2, so the table stays 99 wide.
# `Package` says a cut with `…` instead of ending mid-word; `Speed` holds up to `9999.9 tok/s`.
_TERMINAL_COLUMNS = (
    ("#", 3),
    ("Model", 22),
    ("Package", 22),
    ("Fit", 12),
    ("Speed", 12),
    ("Memory", 8),
    ("Download", 8),
)
_INDENT = " "
# The labels of the lines under the table, in the order a reader reads them. They stand in the
# card's own column (`intro.label_line`), so the report of step 5 reads as one block from the
# `folder` row down to the install command: running text buried the command a reader came for
# (test round, 2026-09-25).
SHOWN_LABEL = "shown"
MEMORY_LABEL = "memory"
BASIS_LABEL = "basis"
SPEED_LABEL = "speed"
# The words of the table's own columns, under the notes and above the install line (decided
# 2026-09-27): the same hint the Markdown file carries under its ranking.
KEY_LABEL = "key"
INSTALL_LABEL = "install"
# U4 and U8 (AP9-UX2): the repositories of rows that look alike, and the room left on this machine.
REPO_LABEL = "repo"
FREE_LABEL = "free"
ANOTHER_COMPUTER_FREE = "unknown on another computer"
OTHER_INSTALL_NOTE = "run it on that computer; this run pulls only onto this one"
NOT_UNIQUE_INSTALL = "has no command: its pull name is not unique"
# What is left of the 100 characters for the text of a labeled line: the label's own column, and
# the one space every line of this screen carries.
_NOTE_WIDTH = JOIN_WIDTH - LABEL_COLUMN - len(_INDENT)
# How many of the packages behind one reason are named before the rest is a number.
_NAMED_PER_REASON = 3
# What a group of ranked packages says about the memory it runs in, plural and singular. The words
# come from the fields of `Fit` alone -- the pool, the mode and the rank -- and never from parsing
# the note's own sentence back apart (decided 2026-09-24).
_POOL_WORDS = {
    "gpu": ("fit into graphics memory", "fits into graphics memory"),
    "gpu_split": ("fit into graphics memory, spread over the cards", "fits into graphics memory, spread over the cards"),
    "cpu_gpu": ("need system memory, the graphics card helps", "needs system memory, the graphics card helps"),
    "cpu": ("need system memory, no graphics card", "needs system memory, no graphics card"),
}
_POOL_ORDER = tuple(_POOL_WORDS)
# Under the `speed` label, so the word "speed" is not said twice -- but still "the rows shown", not
# "this machine": the ranking rule sorts by fit class first, so a measured package can stand behind
# eleven unmeasured ones and be in no `RankedEntry` of this document at all. The card of the same
# screen counts that measurement, and the two may not contradict each other (second-model round,
# 2026-09-24 and 2026-09-25).
_NO_SPEED_YET = "nothing measured in the rows shown"
_FROM_SIZE_NOTE = "computed from the package size, not its architecture"
# What the ranking has nothing to say about at all: no snapshot, or nothing in it that a fit may be
# computed for. Step 5 says it where it happens; `render` says its own sentence on stderr.
NOTHING_TO_RANK = "nothing to rank: no package in this folder yet"
NOTHING_RANKED_SUMMARY = "nothing to rank"


def _terminal_row(cells: list[str]) -> str:
    """One row of the terminal table; the padding of the last column is not printed."""
    return "  ".join(text.ljust(width)[:width] for text, (_name, width) in zip(cells, _TERMINAL_COLUMNS)).rstrip()


def _terminal_rule() -> str:
    """The line under the table head, as wide as the columns are."""
    width = sum(width for _name, width in _TERMINAL_COLUMNS) + 2 * (len(_TERMINAL_COLUMNS) - 1)
    return glyphs().rule * width


def _set_aside_pieces(word: str, entries: list[SetAsideEntry]) -> list[str]:
    """One piece per reason: how many packages it covers, and up to three of their names.

    41 lines of `not covered`, one per package, said the same thing 41 times and pushed the
    ranking itself off the screen (hand test, 2026-09-24); since 2026-09-24 they stand behind the
    `shown` count as pieces of one line. The reason is named only where the list has more than
    one -- with a single reason the word (`no fit`, `not covered`) already says it, and the
    reason itself is in the Markdown file, which has room for it. The word `package` is the
    `shown` piece's own (`10 of 23 packages`), so a piece behind it is the count and the reason.
    """
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        grouped.setdefault(entry.reason, []).append(f"{entry.packager} {entry.quantization}")
    pieces = []
    for reason, names in sorted(grouped.items()):
        count = str(len(names))
        label = f"{word}, {reason}" if len(grouped) > 1 else word
        pieces.append(f"{count} {label}{_named(names, _NOTE_WIDTH - len(count) - len(label) - 1)}")
    return pieces


def _named(names: list[str], room: int) -> str:
    """Up to three of `names` in brackets, and fewer where the room is not there.

    The names are the evidence behind a count, so they are what gives way: three packages of an
    account with a long name made a piece of 137 characters, and `joined` leaves an over-long piece
    whole (measured in the second-model round of 2026-09-24). The count and the reason never give
    way, and nothing is cut mid-name -- a name ends where the next one would have begun.

    `room` is measured against everything this adds: the space in front, both brackets -- three
    characters, not two. With two the line came to 101 (second-model round, 2026-09-25).
    """
    for shown in range(min(_NAMED_PER_REASON, len(names)), 0, -1):
        rest = len(names) - shown
        listed = ", ".join(names[:shown]) + (f" and {rest} more" if rest > 0 else "")
        if len(listed) + 3 <= room:
            return f" ({listed})"
    return ""


def _package_cell(entry: RankedEntry) -> str:
    """`packager · quant`, cut to its column with `…` rather than ending mid-word."""
    width = dict(_TERMINAL_COLUMNS)["Package"]
    text = f"{entry.packager} {glyphs().dot} {entry.quantization}"
    return text if len(text) <= width else text[: width - 1] + "…"


def _terminal_speed(entry: RankedEntry, dash: str) -> str:
    """The speed of a row: the number where one was measured, a dash where none was.

    A dash and not `unknown`: `unknown` in every row of the table said nothing twelve times over
    (test round, 2026-09-24), and the line under the table says once that nothing was measured
    here yet.
    """
    return dash if entry.speed_tps is None else f"{entry.speed_tps:.1f} tok/s"


def _block_head(block: MachineRanking, scenario: Scenario, dot: str, origin: str | None = None) -> str:
    """The line above one machine's table: what the machine is called, and what was computed for it.

    Where llmfit read the profile otherwise, both readings follow (`screen.deviation_words`); a
    confirmed or absent check leaves the line as it is. A machine that is not the one of this run
    says so after its label (U2): `imported`, or `another computer`.
    """
    where = {"imported": f" {dot} imported", "entered": f" {dot} another computer", "folder": f" {dot} another computer"}
    label = f"{block.label}{where.get(origin or 'this', '')}"
    if block.status == "ranked":
        return f"{label} {dot} context {context_short(scenario.context_requested)}{_deviation(block)}"
    return f"{label} {dot} {block.status}"


def _deviation(block: MachineRanking) -> str:
    """The deviation clause of a machine's profile, or nothing where there is no profile."""
    return "" if block.profile is None else deviation_words(block.profile)


def _pool_pieces(block: MachineRanking, dash: str) -> list[str]:
    """The fit of the ranked rows, bundled by the memory they run in, with their rank ranges.

    Ten rows carried ten notes of the same two sentences in the test round of 2026-09-24. A reader
    needs each statement once, with the ranks it is about, and since 2026-09-25 each on a line of
    its own under the `memory` label -- two statements joined by a `·` read as one. The numbers come
    from the fields of `Fit`; `pool_gib` of a graphics-memory fit is what is left there after the
    reserve. Unified memory computes in mode `gpu` too, but it is no graphics memory of its own:
    those rows say `shared memory` (decided 2026-09-25).
    """
    by_mode: dict[str, list[RankedEntry]] = {}
    for entry in block.ranked:
        by_mode.setdefault(str(entry.fit.mode), []).append(entry)
    shared = block.profile is not None and block.profile.gpu_state == "unified_memory"
    pieces = []
    for mode in _POOL_ORDER:
        entries = by_mode.get(mode)
        if not entries:
            continue
        plural, singular = _POOL_WORDS[mode]
        if shared and mode == "gpu":
            plural, singular = "fit into shared memory", "fits into shared memory"
        text = f"{ranks_text([entry.rank for entry in entries], dash)} {singular if len(entries) == 1 else plural}"
        if mode in ("gpu", "gpu_split"):
            # The fit's own budget, labeled as such (U8, F5): the free memory of now is the `free` line.
            source = (
                "memory minus both reserves" if shared else "cards minus the reserve on each" if mode == "gpu_split"
                else "card minus reserve"
            )
            text += f", budget {entries[0].fit.pool_gib:.1f} GB ({source}, from the profile)"
        # A statement longer than the room breaks before its bracket, never inside a word.
        head, bracket, rest = text.partition(" (")
        pieces += [head, f"({rest}"] if len(text) > _NOTE_WIDTH and bracket else [text]
    return pieces


def _speed_pieces(block: MachineRanking, dash: str) -> list[str]:
    """What was measured on the rows this view can see, or that nothing was.

    One piece per measured row (`#3 measured 41.1 tok/s`), in the ranking's own order, then the rows
    whose measurement did not count for its requests alone (`#3 measured with 1 request, ranking
    assumes 3`). Only where neither is there does it say nothing was measured and what to do: a
    row that names its reason was measured, and to be told to measure it would be wrong -- as a
    machine entered by hand would be (`never_measured`).
    """
    if not block.ranked:
        return []
    pieces = [f"#{entry.rank} measured {entry.speed_tps:.1f} tok/s" for entry in block.ranked if entry.speed_tps is not None]
    # A measurement that counts nowhere here says why, and is never told to measure (Codex F3).
    pieces += [
        f"#{entry.rank} {entry.note.text.split('. ')[0].rstrip('.')}" for entry in block.ranked if entry.note.code == "measured_not_comparable"
    ]
    unused = unused_measurements(block, dash)
    return pieces + ([unused] if unused else []) or [_NO_SPEED_YET, never_measured(block) or MEASURE_HINT]


def _labeled_rows(label: str, texts: list[str]) -> list[Line]:
    """One row per text, the label on the first only -- a row that continues it carries none."""
    return [label_line(label if index == 0 else "", [("class:note", text)]) for index, text in enumerate(texts)]


def _labeled(label: str, pieces: list[str], dot: str) -> list[Line]:
    """One labeled statement out of its pieces, joined with ` <dot> ` and wrapped between them."""
    return _labeled_rows(label, joined(pieces, dot=dot, width=_NOTE_WIDTH, indent=0))


@dataclass(frozen=True)
class RunView:
    """What the guided run adds to the terminal view (AP9-UX2): the pull names no row may claim (U4),
    the room left on this machine (U8, `None` outside a guided run) and where each machine came from (U2)."""

    install_for: str | None = None
    ambiguous: frozenset[str] = frozenset()
    free: str | None = None
    origins: Mapping[str, str] = field(default_factory=dict)


def _note_lines(block: MachineRanking, dash: str, dot: str, view: RunView) -> list[Line]:
    """Everything under one machine's table, in the order a reader reads it, each line labeled."""
    lines = _labeled_rows(REPO_LABEL, repo_lines(block))
    shown = shown_pieces(block, dash)
    shown += _set_aside_pieces("not covered", block.not_covered)
    shown += _set_aside_pieces(NO_FIT, block.no_fit)
    shown += collision_pieces(block, view.ambiguous)
    lines += _labeled(SHOWN_LABEL, shown, dot)
    # One pool, one line: two statements about two memories joined by a `·` read as one.
    lines += _labeled_rows(MEMORY_LABEL, _pool_pieces(block, dash))
    from_size = [entry.rank for entry in block.ranked if entry.fit.basis == "size"]
    if from_size:
        lines += _labeled(BASIS_LABEL, [f"{ranks_text(from_size, dash)} {_FROM_SIZE_NOTE}"], dot)
    lines += _labeled(SPEED_LABEL, _speed_pieces(block, dash), dot)
    if block.ranked:
        lines += _labeled_rows(KEY_LABEL, joined(key_pieces(block), dot=dot, width=_NOTE_WIDTH, indent=0) + [SPEED_HINT])
    lines += _install_lines(block, view)
    return lines


def install_target(block: MachineRanking, install_for: str | None) -> tuple[RankedEntry, str] | None:
    """The first row of the machine this run is on, and its local Ollama name -- or `None`.

    Only for that machine: the local Ollama name of a package is about the machine that would
    install it. A package with no such name has none. The install line stands on this choice; the
    pull of step 5 lets the user choose another row (`pull_choice`, U6).
    """
    if install_for is None or block.machine != install_for or not block.ranked:
        return None
    entry = block.ranked[0]
    name = entry_pull_name(entry)
    return None if name is None else (entry, name)


def _install_lines(block: MachineRanking, view: RunView) -> list[Line]:
    """The one line a reader copies to get the first package of a machine.

    Nothing is called and nothing is looked up for it, and a row without a local name has no line
    (CONTRACTS.md, "Guided mode", "The screen"). It stands behind a blank line and the command
    itself is drawn as a command: in running text under the table it went under (test round,
    2026-09-25). It stays when step 5 offers the pull, for anyone who prefers to paste it. A pull
    name more than one package answers to has a note instead (U4); another machine of a guided run
    gets its command with the machine named, and the reminder that this run pulls only here (U3).
    """
    own = view.install_for is not None and block.machine == view.install_for
    target = install_target(block, block.machine if view.install_for is not None else None)
    if target is None:
        return []
    entry, name = target
    if name in view.ambiguous:
        return [[], label_line(INSTALL_LABEL, [("class:note", f"#{entry.rank} {NOT_UNIQUE_INSTALL}")])]
    if own:
        # The one line of this view that may pass 100 characters: it is a command to copy, and a
        # command that was cut is worse than a line that wraps (second-model round, 2026-09-24).
        return [[], label_line(INSTALL_LABEL, install_pull_line(entry.rank, name))]
    other = [("class:note", f"#{entry.rank} on {block.label}"), ("", "  "), ("class:command", f"ollama pull {name}")]
    return [[], label_line(INSTALL_LABEL, other), label_line("", [("class:note", OTHER_INSTALL_NOTE)])]


def _free_lines(block: MachineRanking, view: RunView) -> list[Line]:
    """The room left, under the head of a block (U8): measured now for this machine, else unknown."""
    if view.free is None:
        return []
    text = view.free if block.machine == view.install_for else ANOTHER_COMPUTER_FREE
    return [label_line(FREE_LABEL, [("class:note", text)])]


def _terminal_ranking(block: MachineRanking, scenario: Scenario, dash: str, dot: str, view: RunView) -> list[Line]:
    origin = view.origins.get(block.machine)
    lines: list[Line] = [[("", _block_head(block, scenario, dot, origin))], *_free_lines(block, view)]
    if block.reason is not None:
        lines.append([("", block.reason)])
    if not block.ranked:
        lines.append([("", "no package of the configured base models is ranked here")])
    else:
        lines.append([("", _terminal_row([name for name, _width in _TERMINAL_COLUMNS]))])
        lines.append([("", _terminal_rule())])
        for entry in block.ranked:
            lines.append(
                [
                    (
                        "",
                        _terminal_row(
                            [
                                str(entry.rank),
                                _model_name(entry),
                                _package_cell(entry),
                                fit_word(entry.fit.fit_class, entry.fit.mode),
                                _terminal_speed(entry, dash),
                                f"{entry.fit.need_gib:.1f} GB",
                                f"{entry.weights_gib:.1f} GB",
                            ]
                        ),
                    )
                ]
            )
    lines += _note_lines(block, dash, dot, view)
    return lines


def show_result(
    screen, document: RenderDocument, facts, config, folder: Path, now: datetime, install_for: str | None = None, extra=(),
    view: RunView | None = None,
) -> None:
    """Draw step 5 of the guided mode: the card of the whole run, then one table per machine.

    The head of the step is already on screen, so what is left is the report: the card, the tables
    under it, and the line that says where the document went (CONTRACTS.md, "Guided mode", "The
    screen"). `install_for` is the machine this run is on, the only one the pull may be about;
    `extra` the card rows of the local servers; `view` what the run adds (`RunView`). Every line goes
    through the screen, so the labels, the notes and the install command carry the colors of the card
    (decided 2026-09-25).
    """
    view = RunView(install_for=install_for) if view is None else replace(view, install_for=install_for)
    screen.blank()
    screen.card(
        result_card(
            document,
            folder=folder,
            markdown=config.paths.markdown,
            model_names=facts.model_names,
            packages=facts.packages,
            accounts=facts.accounts,
            now=now,
            machine=install_for,
            extra=extra,
            origins=view.origins,
        )
    )
    for line in terminal_blocks(document, install_for=install_for, view=view):
        screen.write(line)
    screen.blank()
    screen.done(STEP_COUNT, relative_path(config.paths.markdown, folder))


def show_nothing(screen, config, folder: Path, scenario: Scenario, now: datetime, extra=(), origins=None) -> None:
    """Draw step 5 where there is nothing to rank: the reason, the card, and a dash for a balance.

    A head with nothing under it was the last step of a run that chose nothing (test round,
    2026-09-25): the reason stood on stderr, and the card was not drawn. `extra`: the local servers.

    Only where the folder really holds no snapshot. A render that writes no document for another
    reason -- a lock another process holds, a snapshot of an unsupported schema, a view that could
    not be written -- draws nothing here, and the sentence the command already printed stands on its
    own (second-model round, 2026-09-25).
    """
    if not _folder_is_empty(config):
        return
    screen.note(NOTHING_TO_RANK)
    screen.blank()
    screen.card(empty_card(config, folder=folder, scenario=scenario, now=now, extra=extra, origins=origins))
    screen.blank()
    screen.skipped(STEP_COUNT, NOTHING_RANKED_SUMMARY)


def terminal_blocks(
    document: RenderDocument, install_for: str | None = None, *, ambiguous: frozenset[str] = frozenset(), view: RunView | None = None
) -> list[Line]:
    """One machine's block after another, each behind a blank line -- the table and its notes.

    Without the head of step 5 and without the card: those two belong to the guided mode, which
    draws them itself and then prints these lines (CONTRACTS.md, "Guided mode", "The screen").
    `ambiguous` are the pull names no row may claim (`pull_choice.pull_collisions`, U4).
    """
    view = view if view is not None else RunView(install_for=install_for, ambiguous=ambiguous)
    marks = glyphs()
    lines: list[Line] = []
    for block in document.machines:
        lines += [[], *_terminal_ranking(block, document.scenario, marks.skip, marks.dot, view)]
    return lines


def terminal_lines(
    document: RenderDocument, install_for: str | None = None, *, ambiguous: frozenset[str] = frozenset()
) -> list[str]:
    """The same blocks as plain text, for `modelroom render` and for a log of a run."""
    return [plain(line) for line in terminal_blocks(document, install_for, ambiguous=ambiguous)]


def document_terminal(
    document: RenderDocument, install_for: str | None = None, *, ambiguous: frozenset[str] = frozenset()
) -> str:
    """A compact terminal view of the same document: the ranking and the notes under it.

    It shows fewer columns than the Markdown table on purpose (a terminal is narrow), never
    other numbers: every value here is read from `document`, like the other two writers. The
    snapshot time and the ranking rule stay in the Markdown file a reader can take their time
    over, and since 2026-09-24 so does the scenario -- the head of each machine's block says the
    context, which is what a reader of a table needs from it. It is step 5 of the guided mode, so
    it carries that step's head; `modelroom render` writes the two files and stays silent unless a
    caller asks for it.
    """
    lines = [plain(head_line(RESULTS_STEP, glyphs()))]
    if document.rating_unavailable is not None:
        lines.append(f"{_INDENT}Market rating unavailable: {document.rating_unavailable}")
    lines += terminal_lines(document, install_for, ambiguous=ambiguous)
    return "\n".join(lines)


__all__ = [
    "NOTHING_RANKED_SUMMARY",
    "NOTHING_TO_RANK",
    "RunView",
    "context_fact",
    "context_short",
    "document_markdown",
    "document_terminal",
    "empty_card",
    "install_target",
    "relative_path",
    "result_card",
    "scenario_line",
    "show_nothing",
    "show_result",
    "terminal_blocks",
    "terminal_lines",
]
