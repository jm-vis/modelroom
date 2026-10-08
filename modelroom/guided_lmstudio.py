"""The LM Studio server in the guided mode: its start row, its note in step 4, and nothing to measure.

A server that answers on `127.0.0.1:1234` with the model list of LM Studio gets a row on the start screen
that says how many models it has loaded. Step 4 does not offer it: LM Studio is not measured
(`loadtest_lmstudio`), so there is no row in the one list, no request that names a model, no number and no
row on the card. The four functions are the ones the entry of the table takes (`LocalServer.start_fact`,
`offer`, `measure`, `card_facts`); they are not registered here -- AP9-LS-E does that -- so the table and
the guided run are unchanged by this module.

**Nothing is persisted**, and nothing is looked at beyond the model list: no model folder, no process, no
`lms`. What the start screen read lives in the run's `ServerRun` and nowhere else. This module never imports
`guided.py`, and it imports `runtimes.py` only when a step 4 asks for an `Offer` (`runtimes.py` will import
this module at load time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .intro import Fact
from .lmstudio_server import LmStudioKeyError, LmStudioServerError, read_lmstudio_models
from .loadtest_lmstudio import step_note

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from pathlib import Path

    from .config import Configuration
    from .guided import GuidedRun
    from .measurements import Scenario
    from .runtimes import Offer

# The key of this server in the run's `ServerRun`, and the label and name of its start row.
ROW = "lmstudio"
LABEL = "server"
NAME = "LM Studio"
_ORIGIN = "127.0.0.1:1234"


def start_fact(origins) -> Fact | None:
    """The start row: how many models the server has loaded, a server asking for a token, or no row at all.

    What was read is kept in the run for step 4: the model list, or the reason of a server that asks for a
    token. A server that does not answer, or answers with something that is no LM Studio model list, has no
    row, so a machine without LM Studio draws the start screen it always drew.
    """
    server = origins.servers.transports.get(ROW)
    if server is None:
        return None
    try:
        models = read_lmstudio_models(server)
    except LmStudioKeyError as exc:
        origins.servers.found[ROW] = str(exc)
        return Fact(LABEL, _ORIGIN, "asks for an API token")
    except LmStudioServerError:
        return None
    origins.servers.found[ROW] = models
    return Fact(LABEL, NAME, _loaded_words(models.loaded))


def _loaded_words(loaded) -> str:
    """`1 loaded: <id>, context <n>`, `<k> models loaded`, or `0 loaded`: the id is shown, never sent anywhere."""
    if len(loaded) == 1:
        return f"1 loaded: {loaded[0].instance_id}, context {loaded[0].context_length}"
    return f"{len(loaded)} models loaded" if loaded else "0 loaded"


def offer(run: "GuidedRun", config: Configuration | None, scenario: Scenario, results_dir: Path | None) -> "Offer":
    """Step 4 up to the question: the note, and no row -- there is nothing to measure.

    The note is the sentence of `loadtest_lmstudio` for a server that answered, the reason for one that asks
    for a token, and nothing where the start screen read no server. No row means no question for this
    server and no request: the server is not asked for anything here.
    """
    from .runtimes import Offer  # deferred: runtimes.py will import this module at load time (a cycle)

    note = step_note(run.servers.found.get(ROW))
    if note is not None:
        run.screen.note(note)
    return Offer()


def measure(
    run: "GuidedRun",
    config: Configuration | None,
    scenario: Scenario,
    results_dir: Path | None,
    offered: "Offer",
    picked: list[str],
) -> str | None:
    """Nothing was offered, so nothing is measured, whatever `picked` holds (an answers file may name `lmstudio`).

    The server is not called, nothing is kept for the card and the balance of step 4 hears nothing of it.
    """
    return None


def card_facts(run: "GuidedRun") -> tuple[Fact, ...]:
    """No row on the card: there is no number to show and no outcome the run keeps."""
    return ()


__all__ = ["LABEL", "NAME", "ROW", "card_facts", "measure", "offer", "start_fact"]
