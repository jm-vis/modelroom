"""What the views say about the scenario of a document: its sentence, the hint, the unused measurement.

Four pieces of wording the Markdown head, the card of the guided mode and the notes under the
terminal table share, so they say the same thing about the same document (decided 2026-09-26):

- the **scenario sentence** with the origin of the requests (`3 requests (from 25 users)`,
  `12 requests (entered)`, `1 request` for `default` or a scenario a caller passed in), read from
  the document alone -- the Markdown renderer sees no configuration; a context that is a level
  of the scale is said as the scale says it (`32k`), any other as its number (`12000`);
- the **hint** beyond `fit.OLLAMA_REQUEST_HINT` requests, the fit module's own text;
- the **reason a measurement did not count**: it ran one request, the ranking assumes more (the row
  note `measured_with_one_request` that `render._row_note` writes);
- the **reason a machine entered by hand has no speed**, in place of the advice to measure it.

Pure formatters, like `modelroom/views.py`, which imports them; nothing here computes a number.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable

from .document import MachineRanking, RenderDocument
from .fit import OLLAMA_REQUEST_HINT, OLLAMA_REQUEST_HINT_TEXT
from .guided_context import LEVELS
from .measurements import Scenario
from .profile import HardwareProfile
from .screen import ranks_text

# The code of the row note that says why a measurement did not count; the text before its first
# full stop is that reason (a size-basis row adds its own sentence after it).
ONE_REQUEST_NOTE = "measured_with_one_request"
# Under `speed` and on the card, where a machine entered by hand has no measured row.
ENTERED_NEVER_MEASURED = "a machine entered by hand is never measured"


def scenario_line(scenario: Scenario, users: int | None = None, requests_origin: str | None = None) -> str:
    """The one sentence every output prints about what was computed, and where the requests came from."""
    return f"context {scenario_words(scenario, users, requests_origin)}"


def scenario_words(scenario: Scenario, users: int | None, requests_origin: str | None) -> str:
    """The scenario sentence after its first word -- the card's `context` label says that one."""
    assumed = " (assumed)" if scenario.kv_type_assumed else ""
    requests = f"{scenario.requests} {'request' if scenario.requests == 1 else 'requests'}"
    if requests_origin == "from_users" and users is not None:
        requests += f" (from {users} {'user' if users == 1 else 'users'})"
    elif requests_origin == "entered":
        requests += " (entered)"
    context = context_words(scenario.context_requested)
    return f"{context} ({scenario.context_origin}), KV cache {scenario.kv_type}{assumed}, {requests}"


def context_words(context: int) -> str:
    """A context as the sentence says it: a level's tokens as the scale shows them (`32k`), else the number.

    The reader chose `L 32k` and reads `32k` again here; the level's letter is the dialog's own word
    and stays there (`views.context_short`). `docs/models.json` keeps the number.
    """
    level = next((level for level in LEVELS if level.tokens == context), None)
    return str(context) if level is None else level.shown_tokens


def document_scenario(document: RenderDocument) -> str:
    """The scenario sentence of a document, with the origin the document carries."""
    return scenario_line(document.scenario, document.users, document.requests_origin)


def load_hint(document: RenderDocument, width: int | None = None) -> list[str]:
    """Beyond `OLLAMA_REQUEST_HINT` requests the hint, as one line or wrapped at `width`; else none.

    A hint, never a limit: the fit computes up to 1024 requests all the same.
    """
    if document.scenario.requests <= OLLAMA_REQUEST_HINT:
        return []
    return [OLLAMA_REQUEST_HINT_TEXT] if width is None else textwrap.wrap(OLLAMA_REQUEST_HINT_TEXT, width=width)


def unused_measurements(block: MachineRanking | None, dash: str, width: int | None = None) -> str | None:
    """The rows whose measurement ran one request while the ranking assumes more, with that reason.

    By rank (`#1–2, #4 measured with 1 request, ranking assumes 3`); where that is wider than
    `width`, by count (`6 rows measured with ...`) -- the card has one line for it, the table notes
    wrap and always name the ranks.
    """
    rows = [] if block is None else [entry for entry in block.ranked if entry.note.code == ONE_REQUEST_NOTE]
    if not rows:
        return None
    reason = rows[0].note.text.partition(".")[0]
    named = f"{ranks_text([entry.rank for entry in rows], dash)} {reason}"
    if width is None or len(named) <= width:
        return named
    return f"{len(rows)} {'row' if len(rows) == 1 else 'rows'} {reason}"


def never_measured(block: MachineRanking | None) -> str | None:
    """Why a machine entered by hand has no speed, said in place of the advice to measure; else `None`.

    It is not this machine, and measuring this machine never writes into another machine's
    profile (CONTRACTS.md, "Guided mode"): step 4 is no way to a speed for it (decided 2026-09-26).
    """
    return None if block is None else never_measured_of([block.profile])


def never_measured_of(profiles: Iterable[HardwareProfile | None]) -> str | None:
    """The same reason for the machines of a folder: said where every profile there is was entered by hand.

    The card of a run that ranked nothing has no block to follow, only the machines of the
    configuration with the profiles the folder holds; a machine without a profile decides nothing.
    """
    found = [profile for profile in profiles if profile is not None]
    entered = bool(found) and all(profile.ram_physical_source == "entered" for profile in found)
    return ENTERED_NEVER_MEASURED if entered else None


__all__ = [
    "ENTERED_NEVER_MEASURED",
    "ONE_REQUEST_NOTE",
    "context_words",
    "document_scenario",
    "load_hint",
    "never_measured",
    "never_measured_of",
    "scenario_line",
    "scenario_words",
    "unused_measurements",
]
