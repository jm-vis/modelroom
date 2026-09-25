"""The two text views of a `document.RenderDocument`: Markdown and the terminal.

Both are pure formatters -- they read the document `modelroom/render.py::build_render_document`
produced and add no number of their own, so the Markdown file, the JSON file and what the
guided mode prints can never disagree. The JSON view is the document itself
(`render.document_json`). See CONTRACTS.md, "Render (schema 2)".
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from .contracts import Area, Fit, Rating, SchemaVersionError
from .document import MachineRanking, RankedEntry, RenderDocument, SetAsideEntry
from .guided_context import LEVELS
from .intro import LABEL_COLUMN, STEP_COUNT, Fact, glyphs, label_line
from .measurements import Scenario
from .render import format_header_line
from .screen import (
    JOIN_WIDTH,
    MEASURE_HINT,
    Line,
    fit_word,
    head_line,
    install_name,
    install_pull_line,
    joined,
    machine_fact,
    models_fact,
    plain,
    ranks_text,
    result_fact,
    speed_fact,
)

_UNKNOWN = "unknown"
_DASH = "–"
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


def scenario_line(scenario: Scenario) -> str:
    """The one sentence every output prints about what was computed."""
    assumed = " (assumed)" if scenario.kv_type_assumed else ""
    requests = "request" if scenario.requests == 1 else "requests"
    return (
        f"context {scenario.context_requested} ({scenario.context_origin}), "
        f"KV cache {scenario.kv_type}{assumed}, {scenario.requests} {requests}"
    )


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
    "Machine",
    "Profile",
    "Origin",
    "RAM GiB",
    "RAM source",
    "VRAM GiB",
    "VRAM source",
    "GPU state",
    "Reserve RAM GiB",
    "Reserve VRAM GiB",
    "Recorded at",
    "Profile age (days)",
)
_AREA_COLUMNS = ("Source", "Base model", "Packager", "Status", "Last success", "Error")


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
            *([block.status] * 6),
            *reserves,
            block.status,
            block.status,
        ]
    return [
        _cell(block.machine),
        _cell(f"{profile.display_name} ({profile.profile_id})"),
        profile.origin,
        _number(profile.ram_physical_gib),
        profile.ram_physical_source,
        _number(profile.vram_gib),
        profile.vram_source,
        profile.gpu_state,
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
        f"Scenario: {scenario_line(document.scenario)}",
        f"Ranking rule: {document.ranking_rule}",
    ]
    if document.rating_unavailable is not None:
        lines.append(f"Market rating unavailable: {_cell(document.rating_unavailable)}")
    return lines


def _machine_sections(block: MachineRanking, document: RenderDocument) -> list[str]:
    """One machine's sections: its ranking, then what was set aside, each as its own heading."""
    headline = f"{block.label} -- {block.status}"
    if block.reason is not None:
        headline = f"{headline}: {_cell(block.reason)}"
    sections = [f"## Ranking: {block.machine}", "", headline, ""]
    if block.ranked:
        sections += [
            f"Showing {len(block.ranked)} of {block.ranked_total} ranked packages.",
            "",
            _table(_RANKED_COLUMNS, [_ranked_row(entry, document) for entry in block.ranked]),
            "",
        ]
    else:
        sections += ["No package of the configured base models is ranked on this machine.", ""]
    for heading, items in (("Not covered", block.not_covered), ("Too tight", block.too_tight)):
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
    ]
    for block in document.machines:
        sections += _machine_sections(block, document)
    return "\n".join(sections)


# --- writer: terminal ----------------------------------------------------------------------------

# The result table as the mockup of 2026-09-24 draws it: what a reader recognizes a package by
# (its model and its packager) before what a fit is made of. 99 characters wide plus the one space
# every line of this view is indented by, so it fits a 100-column window without wrapping.
# `Fit` is 20 and not 18: `marginal (from size)` is 20 characters and was cut to
# `marginal (from siz` (second-model round, 2026-09-24). The two came off `Package`, whose cell
# says so with `…` instead of ending mid-word.
_TERMINAL_COLUMNS = (
    ("#", 3),
    ("Model", 24),
    ("Package", 22),
    ("Fit", 20),
    ("Speed", 12),
    ("Memory", 8),
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
INSTALL_LABEL = "install"
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
    "cpu_gpu": ("need system memory, the graphics card helps", "needs system memory, the graphics card helps"),
    "cpu": ("need system memory, no graphics card", "needs system memory, no graphics card"),
}
_POOL_ORDER = ("gpu", "cpu_gpu", "cpu")
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
    one -- with a single reason the word (`too tight`, `not covered`) already says it, and the
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


def context_short(context: int) -> str:
    """A context as the scale of the dialog names it: `L 32k`, or the number for a custom one."""
    level = next((level for level in LEVELS if level.tokens == context), None)
    return f"{level.name} {level.shown_tokens}" if level is not None else f"{context} tokens"


def _model_name(entry: RankedEntry) -> str:
    """What a reader calls the model: the name of its base model, without the account in front."""
    return entry.base_model_hf_repo.partition("/")[2] or entry.base_model_hf_repo


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


def _block_head(block: MachineRanking, scenario: Scenario, dot: str) -> str:
    """The line above one machine's table: what the machine is called, and what was computed for it."""
    if block.status == "ranked":
        return f"{block.label} {dot} context {context_short(scenario.context_requested)}"
    return f"{block.label} {dot} {block.status}"


def _pool_pieces(block: MachineRanking, dash: str) -> list[str]:
    """The fit of the ranked rows, bundled by the memory they run in, with their rank ranges.

    Ten rows carried ten notes of the same two sentences in the test round of 2026-09-24. A reader
    needs each statement once, with the ranks it is about, and since 2026-09-25 each on a line of
    its own under the `memory` label -- two statements joined by a `·` read as one. The numbers come
    from the fields of `Fit`; `pool_gib` of a graphics-memory fit is what is left there after the
    reserve.
    """
    by_mode: dict[str, list[RankedEntry]] = {}
    for entry in block.ranked:
        by_mode.setdefault(str(entry.fit.mode), []).append(entry)
    pieces = []
    for mode in _POOL_ORDER:
        entries = by_mode.get(mode)
        if not entries:
            continue
        plural, singular = _POOL_WORDS[mode]
        text = f"{ranks_text([entry.rank for entry in entries], dash)} {singular if len(entries) == 1 else plural}"
        if mode == "gpu":
            text += f", {entries[0].fit.pool_gib:.1f} GB free after the reserve"
        pieces.append(text)
    return pieces


def _speed_pieces(block: MachineRanking, dash: str) -> list[str]:
    """What was measured on the rows this view can see, or that nothing was.

    One piece per measured row (`#3 measured 41.1 tok/s`), in the ranking's own order; where no row
    carries a speed, the reason and what to do about it.
    """
    if not block.ranked:
        return []
    measured = [entry for entry in block.ranked if entry.speed_tps is not None]
    if not measured:
        return [_NO_SPEED_YET, MEASURE_HINT]
    return [f"#{entry.rank} measured {entry.speed_tps:.1f} tok/s" for entry in measured]


def _labeled_rows(label: str, texts: list[str]) -> list[Line]:
    """One row per text, the label on the first only -- a row that continues it carries none."""
    return [label_line(label if index == 0 else "", [("class:note", text)]) for index, text in enumerate(texts)]


def _labeled(label: str, pieces: list[str], dot: str) -> list[Line]:
    """One labeled statement out of its pieces, joined with ` <dot> ` and wrapped between them."""
    return _labeled_rows(label, joined(pieces, dot=dot, width=_NOTE_WIDTH, indent=0))


def _note_lines(block: MachineRanking, dash: str, dot: str, install_for: str | None) -> list[Line]:
    """Everything under one machine's table, in the order a reader reads it, each line labeled."""
    shown = [f"{len(block.ranked)} of {block.ranked_total} packages"]
    shown += _set_aside_pieces("not covered", block.not_covered)
    shown += _set_aside_pieces("too tight", block.too_tight)
    lines = _labeled(SHOWN_LABEL, shown, dot)
    # One pool, one line: two statements about two memories joined by a `·` read as one.
    lines += _labeled_rows(MEMORY_LABEL, _pool_pieces(block, dash))
    from_size = [entry.rank for entry in block.ranked if entry.fit.basis == "size"]
    if from_size:
        lines += _labeled(BASIS_LABEL, [f"{ranks_text(from_size, dash)} {_FROM_SIZE_NOTE}"], dot)
    lines += _labeled(SPEED_LABEL, _speed_pieces(block, dash), dot)
    lines += _install_lines(block, install_for)
    return lines


def _install_lines(block: MachineRanking, install_for: str | None) -> list[Line]:
    """The one line a reader copies to get the first package of the machine this run is on.

    Only for that machine: the local Ollama name of a package is about the machine that would
    install it. Nothing is called, nothing is looked up, and a package with no such name has no
    line (CONTRACTS.md, "Guided mode", "The screen"). It stands behind a blank line and the command
    itself is drawn as a command: in running text under the table it went under (test round,
    2026-09-25).
    """
    if install_for is None or block.machine != install_for or not block.ranked:
        return []
    entry = block.ranked[0]
    name = install_name(entry.package_identity, entry.quantization, entry.format)
    if name is None:
        return []
    # The one line of this view that may pass 100 characters: it is a command to copy, and a command
    # that was cut is worse than a line that wraps (second-model round, 2026-09-24).
    return [[], label_line(INSTALL_LABEL, install_pull_line(entry.rank, name))]


def _terminal_ranking(block: MachineRanking, scenario: Scenario, dash: str, dot: str, install_for: str | None) -> list[Line]:
    lines: list[Line] = [[("", _block_head(block, scenario, dot))]]
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
                            ]
                        ),
                    )
                ]
            )
    lines += _note_lines(block, dash, dot, install_for)
    return lines


# --- the card of the whole run ----------------------------------------------------------------------


def relative_path(path: Path, folder: Path) -> str:
    """A path as the card shows it: relative to the results folder where it lies inside it.

    `docs\\models.md` says what a reader needs; the whole path said it for the third time on one
    screen (test round, 2026-09-24).
    """
    try:
        return str(path.relative_to(folder))
    except ValueError:
        return str(path)


def context_fact(context: int) -> Fact:
    """The context row of the card: the level, and what that many tokens are good for."""
    level = next((level for level in LEVELS if level.tokens == context), None)
    if level is None:
        return Fact("context", context_short(context), f"about {int(context * 0.75):,} words")
    return Fact("context", context_short(context), f"{level.words}, {level.example}")


def _card_block(document: RenderDocument, machine: str | None) -> MachineRanking | None:
    """The machine the card counts: the one the run is on, else the first that has a ranking.

    One machine, not every one: a package that fits two machines is one package and two rows, and a
    sum over the machines would count it twice while the `models` row of the same card counts the
    packages of the snapshot once (second-model round, 2026-09-24).
    """
    blocks = document.machines
    return next(
        (block for block in blocks if block.machine == machine),
        next((block for block in blocks if block.status == "ranked"), blocks[0] if blocks else None),
    )


def _measured_speeds(block: MachineRanking | None) -> tuple[int, tuple[str, float] | None]:
    """How many rows of this machine's ranking carry a measured speed, and which is the fastest."""
    rows = [] if block is None else block.ranked
    measured = [(_model_name(entry), entry.speed_tps) for entry in rows if entry.speed_tps is not None]
    return len(measured), max(measured, key=lambda entry: entry[1]) if measured else None


def _set_aside_counts(block: MachineRanking | None) -> list[tuple[str, int]]:
    """How many packages this machine set aside, by the word that says why, where there are any."""
    if block is None:
        return []
    counted = (("too tight", len(block.too_tight)), ("not covered", len(block.not_covered)))
    return [pair for pair in counted if pair[1]]


def _folder_is_empty(config) -> bool:
    """Whether this results folder holds no snapshot at all -- the one case `show_nothing` is about.

    A snapshot that does not read is no answer to that question: the command that tried to render it
    said so and kept its own exit code, and a second sentence about the folder would be a guess.
    """
    from .state import UnreadableStateFileError, read_snapshot

    try:
        return read_snapshot(config) is None
    except (SchemaVersionError, UnreadableStateFileError):
        return False


def empty_card(config, *, folder: Path, scenario: Scenario, now: datetime) -> list[Fact]:
    """The card of a step 5 that had nothing to rank: the same rows, and a dash for the result.

    Everything a run without a snapshot still knows: the folder, the machines the configuration
    names with the profiles this folder holds (`render.machine_profile`, the same decision the
    document would have rested on), the models of the snapshot (none), the context that was chosen,
    and no measurement. The `result` row is a dash, because no document was written.
    """
    from .guided_context import snapshot_facts
    from .importer import scan_profiles
    from .render import machine_profile

    scan = scan_profiles(config.paths.hardware_dir)
    facts = [Fact("folder", str(folder), "")]
    for name, machine in config.machines.items():
        found = machine_profile(name, machine, scan)
        facts.append(machine_fact(found.label, found.profile, now, found.reason or found.status))
    snapshot = snapshot_facts(config)
    facts.append(models_fact(snapshot.model_names, snapshot.packages, snapshot.accounts))
    facts.append(context_fact(scenario.context_requested))
    facts.append(speed_fact(0, None))
    facts.append(Fact("result", _DASH, ""))
    return facts


def result_card(
    document: RenderDocument,
    *,
    folder: Path,
    markdown: Path,
    model_names: Sequence[str],
    packages: int,
    accounts: Sequence[str],
    now: datetime,
    machine: str | None = None,
) -> list[Fact]:
    """The whole run in six kinds of row: folder, machine, models, context, speed, result.

    The rows of the start screen, with what this run made of them (decided 2026-09-24): a reader
    who looks at the screen once the run is over sees everything in one place, and the table under
    it says the rest. Every number comes from the document the render just wrote or from the
    snapshot it was built from -- nothing here is computed a second time. `machine` is the machine
    of the run, whose ranking the `speed` and `result` rows count; the tables below carry them all.
    """
    block = _card_block(document, machine)
    facts = [Fact("folder", str(folder), "")]
    facts += [
        machine_fact(entry.label, entry.profile, now, entry.reason or entry.status) for entry in document.machines
    ]
    measured, fastest = _measured_speeds(block)
    facts.append(models_fact(model_names, packages, accounts))
    facts.append(context_fact(document.scenario.context_requested))
    facts.append(speed_fact(measured, fastest))
    facts.append(
        result_fact(
            relative_path(markdown, folder),
            0 if block is None else block.ranked_total,
            _set_aside_counts(block),
            glyphs().dot,
        )
    )
    return facts


def show_result(
    screen, document: RenderDocument, facts, config, folder: Path, now: datetime, install_for: str | None = None
) -> None:
    """Draw step 5 of the guided mode: the card of the whole run, then one table per machine.

    The head of the step is already on screen when this runs, so what is left is the report: the
    card a reader looks at once the run is over, the tables under it, and the one line that says
    where the document went (CONTRACTS.md, "Guided mode", "The screen"). `install_for` is the
    machine this run is on, and the only one an install line may be about. Every line goes through
    the screen, so the labels, the notes and the install command carry the colors of the card
    (decided 2026-09-25).
    """
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
        )
    )
    for line in terminal_blocks(document, install_for=install_for):
        screen.write(line)
    screen.blank()
    screen.done(STEP_COUNT, relative_path(config.paths.markdown, folder))


def show_nothing(screen, config, folder: Path, scenario: Scenario, now: datetime) -> None:
    """Draw step 5 where there is nothing to rank: the reason, the card, and a dash for a balance.

    A head with nothing under it was the whole last step of a run that chose nothing (test round,
    2026-09-25): the reason stood on stderr, where no reader of the run is, and the card that says
    what the run did do was not drawn at all.

    Only where the folder really holds no snapshot. Every render that writes no document leaves
    step 5 without one -- a lock another process holds, a snapshot of an unsupported schema, a view
    that could not be written -- and none of those is a folder without packages. Then this draws
    nothing at all and the sentence the command already printed stands on its own (second-model
    round, 2026-09-25).
    """
    if not _folder_is_empty(config):
        return
    screen.note(NOTHING_TO_RANK)
    screen.blank()
    screen.card(empty_card(config, folder=folder, scenario=scenario, now=now))
    screen.blank()
    screen.skipped(STEP_COUNT, NOTHING_RANKED_SUMMARY)


def terminal_blocks(document: RenderDocument, install_for: str | None = None) -> list[Line]:
    """One machine's block after another, each behind a blank line -- the table and its notes.

    Without the head of step 5 and without the card: those two belong to the guided mode, which
    draws them itself and then prints these lines (CONTRACTS.md, "Guided mode", "The screen").
    """
    marks = glyphs()
    lines: list[Line] = []
    for block in document.machines:
        lines += [[], *_terminal_ranking(block, document.scenario, marks.skip, marks.dot, install_for)]
    return lines


def terminal_lines(document: RenderDocument, install_for: str | None = None) -> list[str]:
    """The same blocks as plain text, for `modelroom render` and for a log of a run."""
    return [plain(line) for line in terminal_blocks(document, install_for)]


def document_terminal(document: RenderDocument, install_for: str | None = None) -> str:
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
    lines += terminal_lines(document, install_for)
    return "\n".join(lines)


__all__ = [
    "NOTHING_RANKED_SUMMARY",
    "NOTHING_TO_RANK",
    "context_fact",
    "context_short",
    "document_markdown",
    "document_terminal",
    "empty_card",
    "relative_path",
    "result_card",
    "scenario_line",
    "show_nothing",
    "show_result",
    "terminal_blocks",
    "terminal_lines",
]
