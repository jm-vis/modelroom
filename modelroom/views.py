"""The two text views of a `document.RenderDocument`: Markdown and the terminal.

Both are pure formatters -- they read the document `modelroom/render.py::build_render_document`
produced and add no number of their own, so the Markdown file, the JSON file and what the
guided mode prints can never disagree. The JSON view is the document itself
(`render.document_json`). See CONTRACTS.md, "Render (schema 2)".
"""

from __future__ import annotations

from datetime import datetime

from .contracts import Area, Fit, Rating
from .document import MachineRanking, RankedEntry, RenderDocument, SetAsideEntry
from .intro import STEP_COUNT, glyphs, step_head
from .measurements import Scenario
from .render import format_header_line

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

_TERMINAL_COLUMNS = (
    ("#", 3),
    ("packager", 20),
    ("quant", 12),
    ("GiB", 8),
    ("fit (computed)", 20),
    ("tok/s (measured)", 16),
)
# How many of the packages behind one reason are named before the rest is a number.
_NAMED_PER_REASON = 3


def _terminal_row(cells: list[str]) -> str:
    """One row of the terminal table; the padding of the last column is not printed."""
    return "  ".join(text.ljust(width)[:width] for text, (_name, width) in zip(cells, _TERMINAL_COLUMNS)).rstrip()


def _terminal_rule() -> str:
    """The line under the table head, as wide as the columns are."""
    width = sum(width for _name, width in _TERMINAL_COLUMNS) + 2 * (len(_TERMINAL_COLUMNS) - 1)
    return glyphs().rule * width


def _set_aside_lines(heading: str, entries: list[SetAsideEntry]) -> list[str]:
    """One line per reason, with the number of packages behind it and up to three of their names.

    41 lines of `not covered`, one per package, said the same thing 41 times and pushed the
    ranking itself off the screen (hand test, 2026-09-24). The reason is what a reader can
    act on, so the reason is the line, and the names are its evidence.
    """
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        grouped.setdefault(entry.reason, []).append(f"{entry.packager} {entry.quantization}")
    lines = []
    for reason, names in sorted(grouped.items()):
        shown = ", ".join(names[:_NAMED_PER_REASON])
        rest = len(names) - _NAMED_PER_REASON
        listed = f"{shown} and {rest} more" if rest > 0 else shown
        count = f"{len(names)} package" + ("" if len(names) == 1 else "s")
        lines.append(f"  {heading}: {count} -- {reason} ({listed})")
    return lines


def _terminal_ranking(block: MachineRanking) -> list[str]:
    lines = [f"Ranking: {block.machine} ({block.label}, {block.status})"]
    if block.reason is not None:
        lines.append(f"  {block.reason}")
    if not block.ranked:
        lines.append("  no package of the configured base models is ranked here")
    else:
        lines.append("  " + _terminal_row([name for name, _width in _TERMINAL_COLUMNS]))
        lines.append("  " + _terminal_rule())
        for entry in block.ranked:
            lines.append(
                "  "
                + _terminal_row(
                    [
                        str(entry.rank),
                        entry.packager,
                        entry.quantization,
                        f"{entry.weights_gib:.2f}",
                        _fit_text(entry.fit),
                        _UNKNOWN if entry.speed_tps is None else f"{entry.speed_tps:.1f}",
                    ]
                )
            )
        lines.append(f"  showing {len(block.ranked)} of {block.ranked_total} ranked packages")
        lines += [f"  #{entry.rank} {entry.note.text}" for entry in block.ranked]
    lines += _set_aside_lines("not covered", block.not_covered)
    lines += _set_aside_lines("too tight", block.too_tight)
    return lines


def document_terminal(document: RenderDocument) -> str:
    """A compact terminal view of the same document: the ranking, the notes and both blocks.

    It shows fewer columns than the Markdown table on purpose (a terminal is narrow), never
    other numbers: every value here is read from `document`, like the other two writers. It is
    step 5 of the guided mode, which is the only caller that asks for it, so it carries that
    step's head; `modelroom render` writes the two files and stays silent.
    """
    lines = [
        step_head(RESULTS_STEP),
        "",
        f"Snapshot run at: {document.snapshot_run_at.isoformat()}",
        f"Scenario: {scenario_line(document.scenario)}",
        f"Ranking rule: {document.ranking_rule}",
    ]
    if document.rating_unavailable is not None:
        lines.append(f"Market rating unavailable: {document.rating_unavailable}")
    for block in document.machines:
        lines += ["", *_terminal_ranking(block)]
    return "\n".join(lines)


__all__ = ["document_markdown", "document_terminal", "scenario_line"]
