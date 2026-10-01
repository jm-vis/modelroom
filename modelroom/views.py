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
from typing import Mapping, Sequence

from . import runtimes
from .contracts import Area, Fit, Package, Rating
from .document import MachineRanking, RankedEntry, RenderDocument, SetAsideEntry
from .views_runtime import RUNTIME_LABEL, key_text, packager_cell, runtime_line, unequal_cards_note, vllm_pool_pieces
from .intro import LABEL_COLUMN, STEP_COUNT, glyphs, label_line
from .measurements import Scenario
from .disk_fit import DISK_LABEL, disk_words, entry_over_disk
from .pull_choice import collision_pieces, repo_lines
from .ranking_words import EACH_MACHINE_SENTENCE, several_ranked, showing_line, shown_pieces
from .render import format_header_line
from .room import Disk, LiveRoom
from .room_head import LOADED_LABEL, ROOM_LABEL, head_pieces, loaded_pieces, reserve_hint_rows
from .views_install import INSTALL_LABEL, install_lines, install_target, over_ranks_on_machine  # noqa: F401 -- `install_target` is re-exported
from .result_card import _DASH, _folder_is_empty, _model_name, context_fact, context_short, empty_card
from .result_card import relative_path, result_card
from .scenario_text import document_scenario, load_hint, never_measured, scenario_line, unused_measurements
from .views_adapters import ADAPTER_COLUMNS, ADAPTER_HINT, adapter_table_rows
from .screen import (
    JOIN_WIDTH,
    MEASURE_HINT,
    NO_FIT,
    Line,
    deviation_words,
    fit_word,
    head_line,
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
    "GPU state: measured = a counted card, read by modelroom or llmfit; entered = typed in by hand; multi_gpu = "
    "several counted cards, added up; present_unmeasured = a card may be there, nothing about it was read; none = "
    "no counted card (none found, none usable, display adapter only, or --cpu-only); unified_memory = system and card share one "
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
    reserves = [_reserve_cell(block), f"{block.reserve_vram_gib:.2f}"]
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


def _reserve_cell(block: MachineRanking) -> str:
    """The memory reserve and where it came from (AP9-RM): `16.00 (computed)`, `1.50`, or a dash."""
    computed = " (computed)" if block.reserve_ram_origin == "computed" else ""
    return _DASH if block.reserve_ram_origin == "unknown" else f"{block.reserve_ram_gib:.2f}{computed}"


def _number(value: float | None) -> str:
    return _UNKNOWN if value is None else f"{value:.2f}"


def _ranked_row(entry: RankedEntry, document: RenderDocument, disk: Disk | None = None) -> list[str]:
    row = [
        str(entry.rank),
        _cell(entry.base_model_hf_repo),
        _stars(document.ratings.get(entry.base_model_hf_repo)),
        _cell(packager_cell(entry)),
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
    row[6] += " ×" if entry_over_disk(entry, disk) else ""  # AP9-RM U11: the download does not fit the disk
    return row


def _set_aside_row(entry: SetAsideEntry, document: RenderDocument) -> list[str]:
    return [
        _cell(entry.base_model_hf_repo),
        _stars(document.ratings.get(entry.base_model_hf_repo)),
        _cell(packager_cell(entry)),
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


def _machine_sections(block: MachineRanking, document: RenderDocument, first: bool = False, disk: Disk | None = None) -> list[str]:
    """One machine's sections: its ranking, then what was set aside, each as its own heading; the
    reserve hint of 0.1.0 and, for the `first` of several, that each table is its machine's (AP9-RM);
    with the free disk of a guided run, ` ×` behind a download too large for it and the `disk` line (U11)."""
    over = disk_words(block, disk, "–", "×")
    headline = f"{block.label} -- {block.status}{_deviation(block)}"
    if block.reason is not None:
        headline = f"{headline}: {_cell(block.reason)}"
    sections = [f"## Ranking: {block.machine}", "", headline, ""]
    hint = " ".join(reserve_hint_rows(block))
    sections += [f"{hint[0].upper()}{hint[1:]}.", ""] if hint else []
    if block.ranked:
        repos = repo_lines(block)
        sections += [
            f"{showing_line(block)}{' ' + EACH_MACHINE_SENTENCE if first else ''}",
            "",
            _table(_RANKED_COLUMNS, [_ranked_row(entry, document, disk) for entry in block.ranked]),
            "",
            *([f"{over}.", ""] if over else []),
            *([_cell(" · ".join(repos)), ""] if repos else []),
            " · ".join(key_text(block)),
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


def document_markdown(document: RenderDocument, disks: Mapping[str, Disk] | None = None) -> str:
    """The Markdown view of `document`: the same numbers the JSON and the terminal view show; `disks`
    is the free disk a guided run read, per machine name (U11), and marks nothing where it is `None`."""
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
    adapters = [[_cell(value) for value in row] for block in document.machines for row in adapter_table_rows(block.machine, block.profile)]
    if adapters:
        sections += ["## Graphics adapters", "", _table(ADAPTER_COLUMNS, adapters), "", ADAPTER_HINT, ""]
    first = next((block.machine for block in document.machines if block.ranked), None) if several_ranked(document) else None
    for block in document.machines:
        sections += _machine_sections(block, document, block.machine == first, (disks or {}).get(block.machine))
    return "\n".join(sections)


# --- writer: terminal ----------------------------------------------------------------------------

# The result table as the mockup of 2026-09-24 draws it: what a reader recognizes a package by
# (its model and its packager) before what a fit is made of. 99 characters wide plus the one space
# every line of this view is indented by, so it fits a 100-column window without wrapping.
# Since AP9-UX2 (U8) the table carries the download of each row, the weight files of its package:
# `Fit` gave up 8 (its longest word, `tight (RAM)`, is 11) and `Model` 2, so the table stays 99 wide.
# `Package` says a cut with `…` instead of ending mid-word; `Speed` holds up to `9999.9 tok/s`.
# Since AP9-RM U11 a download too large for the free disk carries ` ×` (`100.5 GB ×`), so `Download`
# is 10 wide and `Fit` (11, its longest word) and `Package` (21) gave up one each.
_TERMINAL_COLUMNS = (
    ("#", 3),
    ("Model", 22),
    ("Package", 21),
    ("Fit", 11),
    ("Speed", 12),
    ("Memory", 8),
    ("Download", 10),
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
# U4 (AP9-UX2): the repositories of rows that look alike. The `room` and `loaded` lines under the
# head are `room_head`'s (AP9-RM U10), the `install` line `views_install`'s (U11).
REPO_LABEL = "repo"
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
        # The runtime's prefix as in every packager cell (F4): `vLLM · acme AWQ`, nothing for Ollama.
        grouped.setdefault(entry.reason, []).append(f"{packager_cell(entry)} {entry.quantization}")
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
    """`<prefix>packager · quant`, cut to its column with `…` rather than ending mid-word (F4:
    the runtime's own `words.package_prefix`, empty for Ollama)."""
    width = dict(_TERMINAL_COLUMNS)["Package"]
    text = f"{packager_cell(entry)} {glyphs().dot} {entry.quantization}"
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
    where = {
        "imported": f" {dot} imported",
        "entered": f" {dot} another computer",
        "folder": f" {dot} another computer",
        "ssh": f" {dot} over SSH",  # FR-B1, B3
    }
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
    # vLLM's rows share `Fit.mode`'s own words (`gpu`/`gpu_split`) with Ollama's, but not its
    # accounting (a reserved share of each card, not Ollama's single VRAM reserve): they never join
    # an Ollama piece, and form their own below instead (F4, `views_runtime.vllm_pool_pieces`).
    ollama_entries = [entry for entry in block.ranked if entry.fit.runtime != runtimes.VLLM.key]
    vllm_entries = [entry for entry in block.ranked if entry.fit.runtime == runtimes.VLLM.key]
    by_mode: dict[str, list[RankedEntry]] = {}
    for entry in ollama_entries:
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
        # The fit's own budget, labeled as such (U8, F5), for system memory too since AP9-RM: the
        # reserve of the system memory is no longer invisible.
        source = (
            "memory minus both reserves" if shared and mode == "gpu" else "cards minus the reserve on each"
            if mode == "gpu_split" else "card minus reserve" if mode == "gpu" else "memory minus reserve"
        )
        text += f", budget {entries[0].fit.pool_gib:.1f} GB ({source}, from the profile)"
        # A statement longer than the room breaks before its bracket, never inside a word.
        head, bracket, rest = text.partition(" (")
        pieces += [head, f"({rest}"] if len(text) > _NOTE_WIDTH and bracket else [text]
    pieces += vllm_pool_pieces(vllm_entries, dash)
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
    what was read live per machine name (AP9-RM U10; `None` outside a guided run, which reads no
    machine live), where each machine came from (U2), and the other machines `pull_other_step` will
    list a row for (A3, AP9-FR) -- so a table's note never promises a list that will not follow."""

    install_for: str | None = None
    ambiguous: frozenset[str] = frozenset()
    live: Mapping[str, LiveRoom] | None = None
    origins: Mapping[str, str] = field(default_factory=dict)
    pull_other_machines: frozenset[str] = frozenset()
    ssh_active: str | None = None  # FR-B1, B5: the `room` line names this one by its alias
    packages: Sequence[Package] = ()  # the run's packages: the install line's runtime (`views_install`)
    held: Mapping[str, frozenset] = field(default_factory=dict)  # what each runtime but Ollama holds itself


def _note_lines(block: MachineRanking, dash: str, dot: str, view: RunView, several: bool = False) -> list[Line]:
    """Everything under one machine's table, in the order a reader reads it, each line labeled."""
    lines = _labeled_rows(REPO_LABEL, repo_lines(block))
    shown = shown_pieces(block, dash, several)
    shown += _set_aside_pieces("not covered", block.not_covered)
    shown += _set_aside_pieces(NO_FIT, block.no_fit)
    shown += collision_pieces(block, view.ambiguous)
    lines += _labeled(SHOWN_LABEL, shown, dot)
    # One pool, one line: two statements about two memories joined by a `·` read as one.
    lines += _labeled_rows(MEMORY_LABEL, _pool_pieces(block, dash) + reserve_hint_rows(block))
    lines += _disk_lines(block, _disk(block, view), dash, _over(block, view))
    from_size = [entry.rank for entry in block.ranked if entry.fit.basis == "size"]
    if from_size:
        lines += _labeled(BASIS_LABEL, [f"{ranks_text(from_size, dash)} {_FROM_SIZE_NOTE}"], dot)
    lines += _labeled(SPEED_LABEL, _speed_pieces(block, dash), dot)
    if block.ranked:
        lines += _labeled_rows(KEY_LABEL, joined(key_text(block), dot=dot, width=_NOTE_WIDTH, indent=0) + [SPEED_HINT])
    disk = _disk(block, view)
    return lines + install_lines(block, view.install_for, view.ambiguous, disk, dash, view.pull_other_machines, view.packages, _held(block, view))


def _disk(block: MachineRanking, view: RunView) -> Disk | None:
    """The free disk read for this block's machine, where one was (U11); `None` in `render`."""
    live = None if view.live is None else view.live.get(block.machine)
    return None if live is None else live.disk


def _held(block: MachineRanking, view: RunView) -> Mapping[str, frozenset] | None:
    """What each runtime but Ollama holds, for the machine this run is on only (`RunView.held`)."""
    return view.held if block.machine == view.install_for else None


def _over(block: MachineRanking, view: RunView) -> list[int]:
    """The ranks of the rows no runtime of the machine can take within the free disk, each runtime
    with its own holding (`views_install.over_ranks_on_machine`)."""
    return over_ranks_on_machine(block, _disk(block, view), view.packages, _held(block, view))


def _disk_lines(block: MachineRanking, disk: Disk | None, dash: str, over: list[int] | None = None) -> list[Line]:
    """The red `disk` line under `memory` (U11): the rows whose download does not fit the free disk,
    broken before what to do where it passes the width."""
    text = disk_words(block, disk, dash, glyphs().cross, over)
    if text is None:
        return []
    head, sep, tail = text.partition(" -- ")
    rows = [text] if len(text) <= _NOTE_WIDTH or not sep else [head, f"-- {tail}"]
    return [label_line(DISK_LABEL if index == 0 else "", [("class:warn", row)]) for index, row in enumerate(rows)]


def _room_lines(block: MachineRanking, view: RunView, dot: str) -> list[Line]:
    """Under the head of a block (U10): the budgets and the disk (`room`), and what Ollama holds
    (`loaded`) where it holds something -- for the machine the live values were read on."""
    live = None if view.live is None else view.live.get(block.machine)
    where = (
        "render" if view.live is None
        else "this" if block.machine == view.install_for
        else "ssh" if block.machine == view.ssh_active
        else "ssh_inactive" if view.origins.get(block.machine) == "ssh"
        else "other"
    )
    lines = _labeled(ROOM_LABEL, head_pieces(block, live, where), dot)
    loaded = loaded_pieces(live)
    return lines + (_labeled(LOADED_LABEL, loaded, dot) if loaded else [])


def _runtime_lines(block: MachineRanking) -> list[Line]:
    """The `runtime` row under a machine's head (F4): `None`/`[]` for a machine without vLLM, so an
    Ollama-only table draws exactly what it drew before AP9-SV2.

    `live` is not threaded through `RunView` yet (`runtimes.VLLM.detect` is AP9-SV2-I/R's own), so
    this reads the machine's remembered profile alone -- the one source `modelroom render` ever
    has, and still correct, if not live, for a guided run on that same machine (F3).
    """
    text = runtime_line(block.profile)
    if text is None:
        return []
    texts = [text]
    note = unequal_cards_note(block.profile)
    if note is not None:
        texts.append(note)
    return _labeled_rows(RUNTIME_LABEL, texts)


def _terminal_ranking(
    block: MachineRanking, scenario: Scenario, dash: str, dot: str, view: RunView, several: bool = False
) -> list[Line]:
    origin = view.origins.get(block.machine)
    lines: list[Line] = [[("", _block_head(block, scenario, dot, origin))], *_room_lines(block, view, dot)]
    lines += _runtime_lines(block)
    if block.reason is not None:
        lines.append([("", block.reason)])
    if not block.ranked:
        lines.append([("", "no package of the configured base models is ranked here")])
    else:
        lines.append([("", _terminal_row([name for name, _width in _TERMINAL_COLUMNS]))])
        lines.append([("", _terminal_rule())])
        over_ranks = set(_over(block, view))
        for entry in block.ranked:
            # A download larger than the free disk: ` ×` behind it, and the whole row on the house red (U11).
            over = entry.rank in over_ranks
            download = f"{entry.weights_gib:.1f} GB" + (f" {glyphs().cross}" if over else "")
            cells = [str(entry.rank), _model_name(entry), _package_cell(entry), fit_word(entry.fit.fit_class, entry.fit.mode)]
            cells += [_terminal_speed(entry, dash), f"{entry.fit.need_gib:.1f} GB", download]
            lines.append([("class:over-disk" if over else "", _terminal_row(cells))])
    lines += _note_lines(block, dash, dot, view, several)
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
    several = several_ranked(document)
    for block in document.machines:
        lines += [[], *_terminal_ranking(block, document.scenario, marks.skip, marks.dot, view, several)]
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
