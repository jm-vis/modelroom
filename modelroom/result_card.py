"""The card of the whole run: the rows of step 5 above the tables, and the card of an empty step 5.

Moved out of `modelroom/views.py` unchanged (AP9-UX, size guard): the card reads the same
`document.RenderDocument` the tables do and adds no number of its own. `views.py` imports from here
and re-exports the public names; this module never imports `views` (CONTRACTS.md, "Guided mode",
"The screen").
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from .contracts import SchemaVersionError
from .document import MachineRanking, RankedEntry, RenderDocument
from .guided_context import LEVELS
from .intro import Fact, glyphs
from .measurements import Scenario
from .scenario_text import load_hint, measurement_reason, never_measured, never_measured_of, scenario_words
from .screen import NO_FIT, machine_fact, models_fact, result_fact, speed_fact

_DASH = "–"
# The line under a machine that is not the one of this run (AP9-UX2 U2), by where its profile came
# from; a machine entered by hand says so in its own row already (`screen.entered_hardware`).
ORIGIN_WORDS = {
    "imported": "imported profile · not this computer",
    "folder": "profile from this folder · not this computer",
    "ssh": "read over SSH · not this computer",  # AP9-FR, FR-B1, B3
}


def _machine_facts(rows, now: datetime, origins) -> list[Fact]:
    """One machine row per `(machine, label, profile, reason)`, and its origin line under it (U2)."""
    facts = []
    for machine, label, profile, reason in rows:
        facts.append(machine_fact(label, profile, now, reason))
        words = ORIGIN_WORDS.get((origins or {}).get(machine, "this"))
        if words is not None:
            facts.append(Fact("", words, ""))
    return facts


def context_short(context: int) -> str:
    """A context as the scale of the dialog names it: `L 32k`, or the number for a custom one."""
    level = next((level for level in LEVELS if level.tokens == context), None)
    return f"{level.name} {level.shown_tokens}" if level is not None else f"{context} tokens"


def _model_name(entry: RankedEntry) -> str:
    """What a reader calls the model: the name of its base model, without the account in front."""
    return entry.base_model_hf_repo.partition("/")[2] or entry.base_model_hf_repo


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
    """The machine the card counts: the one the run is on where it has a ranking, else the first that has one.

    One machine, not every one: a package that fits two machines is one package and two rows, and a
    sum over the machines would count it twice while the `models` row of the same card counts the
    packages of the snapshot once (second-model round, 2026-09-24). A run on a machine without a
    ranking counted `0 packages` next to a table of 25 (live probe, 2026-09-26).
    """
    blocks = document.machines
    own = next((block for block in blocks if block.machine == machine), None)
    if own is not None and own.status == "ranked":
        return own
    return next((block for block in blocks if block.status == "ranked"), own or (blocks[0] if blocks else None))


def _measured_speeds(block: MachineRanking | None) -> tuple[int, tuple[str, float, str | None] | None]:
    """How many rows of this machine's ranking carry a measured speed, and which is the fastest
    (name, speed, protocol of the counted measurement -- `None` for a document that carries none)."""
    rows = [] if block is None else block.ranked
    measured = [
        (_model_name(entry), entry.speed_tps, entry.speed_protocol) for entry in rows if entry.speed_tps is not None
    ]
    return len(measured), max(measured, key=lambda entry: entry[1]) if measured else None


def _set_aside_counts(block: MachineRanking | None) -> list[tuple[str, int]]:
    """How many packages this machine set aside, by the word that says why, where there are any."""
    if block is None:
        return []
    counted = ((NO_FIT, len(block.no_fit)), ("not covered", len(block.not_covered)))
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


def empty_card(
    config, *, folder: Path, scenario: Scenario, now: datetime, extra: Sequence[Fact] = (), origins=None
) -> list[Fact]:
    """The card of a step 5 that had nothing to rank: the same rows, and a dash for the result.

    Everything a run without a snapshot still knows: the folder, the machines the configuration
    names with the profiles this folder holds (`render.machine_profile`, the same decision the
    document would have rested on), the models of the snapshot (none), the context that was chosen,
    and no measurement. The `result` row is a dash, because no document was written. `config` is the
    configuration of this run's own selection (`selection.narrow`), so it names only this run's machines.
    """
    from .guided_context import snapshot_facts
    from .importer import scan_profiles
    from .render import machine_profile

    scan = scan_profiles(config.paths.hardware_dir)
    facts = [Fact("folder", str(folder), "")]
    named = [(name, machine_profile(name, machine, scan)) for name, machine in config.machines.items()]
    found = [entry for _name, entry in named]
    rows = [(name, entry.label, entry.profile, entry.reason or entry.status) for name, entry in named]
    facts += _machine_facts(rows, now, origins)
    # Counted for the machines this card names, by their stored profiles (`runtimes.visible_packages`).
    snapshot = snapshot_facts(config, machines=[(entry.profile, None) for entry in found])
    facts.append(models_fact(snapshot.model_names, snapshot.packages, snapshot.accounts))
    facts.append(context_fact(scenario.context_requested))
    facts.append(speed_fact(0, None, never_measured_of(entry.profile for entry in found)))
    facts += extra  # the rows of the local servers (`local_servers.card_facts`)
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
    extra: Sequence[Fact] = (),
    origins=None,
) -> list[Fact]:
    """The whole run in six kinds of row: folder, machine, models, context, speed, result.

    The rows of the start screen, with what this run made of them (decided 2026-09-24), every number
    from the document just written or its snapshot. `machine` is the machine of the run, whose ranking
    the `speed` and `result` rows count; `extra` the rows of the local servers, above `result`.
    """
    block = _card_block(document, machine)
    facts = [Fact("folder", str(folder), "")]
    rows = [(entry.machine, entry.label, entry.profile, entry.reason or entry.status) for entry in document.machines]
    facts += _machine_facts(rows, now, origins)
    measured, fastest = _measured_speeds(block)
    facts.append(models_fact(model_names, packages, accounts))
    # The scenario sentence of the Markdown head, its first word being the row's own label; the hint
    # beyond 8 requests wrapped under it, so no line of the run is wider than 100 columns.
    facts.append(Fact("context", scenario_words(document.scenario, document.users, document.requests_origin), ""))
    facts += [Fact("" if index else "load", row, "") for index, row in enumerate(load_hint(document, 86))]
    # 65: what is left of 100 columns behind the screen's leading blank, the card's label and value
    # columns and the two gaps (`screen.plain`, `intro.fact_line`).
    reason = measurement_reason(block, glyphs().skip, glyphs().dot, 65)  # the rule of the `speed` note under the table (c2)
    facts.append(speed_fact(measured, fastest, reason or never_measured(block), from_measurement=reason is not None))
    facts += extra
    facts.append(
        result_fact(
            relative_path(markdown, folder),
            0 if block is None else block.ranked_total,
            _set_aside_counts(block),
            glyphs().dot,
        )
    )
    return facts


__all__ = ["context_fact", "context_short", "empty_card", "relative_path", "result_card"]
