"""The LM Studio server in the guided mode: its start row, its row in step 4, its four rows on the card.

A server that answers on `127.0.0.1:1234` with the model list of LM Studio gets a row on the start screen
that says how many models it has loaded. With **exactly one loaded GGUF instance** step 4 offers it as a row
of the one list; the measurement (`loadtest_lmstudio`) is timed on the wall -- one request at a time where the
instance states no `parallel` (or less than 2) or the scenario names one request, else batches of up to
`parallel` requests at once (AP9-PM-1) -- and the number is shown in step 4 and on the card with what it is:
not ranked, nothing written, a busy server reads slower, and the server's own load can stay unseen. With no or several loaded models there is a note and no
row; with a context other than the one this run asks for the number would not count, so there is a note and
no chat. The four functions are the ones the entry of the table takes (`LocalServer.start_fact`, `offer`,
`measure`, `card_facts`).

**Nothing is persisted**, and nothing is looked at beyond the model list and the one request: no model
folder, no process, no `lms`. What the start screen read and what step 4 measured live in the run's
`ServerRun` and nowhere else. This module never imports `guided.py`, and it imports `runtimes.py` only when a
step 4 asks for an `Offer` (`runtimes.py` imports this module at load time). The one thing it takes from
`measurements.py` at run time is the prompt of the protocol (`shipped_protocol().prompt`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .dialog import Choice
from .intro import Fact
from .lmstudio_server import LmStudioKeyError, LmStudioModels, LmStudioServerError, read_lmstudio_models
from .loadtest import LoadTestError
from .loadtest_lmstudio import LmStudioMeasurement, run_lmstudio_load_test
from .measurements import shipped_protocol

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
MEASURED = "1 of 1 measured"
NO_MODEL_NOTE = "LM Studio: no model loaded, nothing to measure"
SEVERAL_NOTE = "LM Studio: {count} models loaded, measure with one loaded model"
OTHER_CONTEXT_NOTE = "the number will not count: LM Studio runs context {server}, this run asks for {run}"
PARALLEL_NOTE = "LM Studio allows {parallel} at once for this model (Max Concurrent Predictions in LM Studio)"
NOT_RANKED_NOTE ="LM Studio states no load, so a busy server reads slower; not ranked, nothing written"
# The three further rows of the card under the number (each at most 65 characters, the note column).
CARD_NOT_RANKED = "package unknown; not ranked, nothing written"
CARD_BUSY = "LM Studio states no load, so a busy server reads slower"
CARD_UNSEEN = "never asks to load; an unseen server load can affect the time"


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
    """`1 loaded: <id>, context <n>`, `<k> models loaded`, or `0 loaded`: the id is shown, and sent to the chat."""
    if len(loaded) == 1:
        return f"1 loaded: {loaded[0].instance_id}, context {loaded[0].context_length}"
    return f"{len(loaded)} models loaded" if loaded else "0 loaded"


def offer(run: "GuidedRun", config: Configuration | None, scenario: Scenario, results_dir: Path | None) -> "Offer":
    """Step 4 up to the question: one row for exactly one loaded instance, else the note and no row.

    A server that asks for a token has neither row nor note (the start row says it), and one the start
    screen did not read has nothing. With a context other than the run's the row is still offered; the note
    that the number will not count comes before the question, and a yes then measures nothing.
    """
    from .runtimes import Offer  # deferred: runtimes.py imports this module at load time (a cycle)

    found = run.servers.found.get(ROW)
    if not isinstance(found, LmStudioModels):
        return Offer()
    if len(found.loaded) != 1:
        run.screen.note(SEVERAL_NOTE.format(count=len(found.loaded)) if found.loaded else NO_MODEL_NOTE)
        return Offer(state=found)
    instance = found.loaded[0]
    if instance.context_length != scenario.context_requested:
        run.screen.note(OTHER_CONTEXT_NOTE.format(server=instance.context_length, run=scenario.context_requested))
    row = Choice(ROW, f"LM Studio server  {instance.instance_id}, context {instance.context_length}")
    return Offer((row,), state=found)


def measure(
    run: "GuidedRun",
    config: Configuration | None,
    scenario: Scenario,
    results_dir: Path | None,
    offered: "Offer",
    picked: list[str],
) -> str | None:
    """Measure the one loaded instance where its row was picked; `1 of 1 measured`, or `None`.

    Nothing runs where the row was not picked (an answers file may name `lmstudio` all the same), where
    the offer holds no single instance, or where its context is not the run's. A row of the table of the
    contract is one line and `None`; a fault outside the table is `nothing measured for LM Studio`. In all
    three step 4 goes on with the other servers. Only a measurement without a problem is kept for the card.
    """
    found = offered.state
    if ROW not in picked or not isinstance(found, LmStudioModels) or len(found.loaded) != 1:
        return None
    instance = found.loaded[0]
    if instance.context_length != scenario.context_requested:
        return None
    try:
        measurement = run_lmstudio_load_test(
            run.servers.transports[ROW], instance, scenario, shipped_protocol().prompt
        )
    except (LoadTestError, LmStudioServerError) as exc:
        run.failed(f"nothing measured for LM Studio: {exc}")
        return None
    if measurement.problem is not None:
        run.screen.note(f"{NAME}: {measurement.problem}")
        return None
    for line in measurement_lines(measurement, run.screen.glyphs.skip):
        run.screen.note(line)
    run.servers.measured[ROW] = measurement
    return MEASURED


def measurement_lines(measurement: LmStudioMeasurement, dash: str) -> list[str]:
    """The notes of a number: the speed with its range, and what it is. With several requests at once (n of 2
    or more) the form of `guided_vllm.measurement_lines`: the total, the speed of one request, and the line
    that names the `parallel` of the instance where it was the limit."""
    speed = f"{measurement.request_tps:.1f} tok/s ({measurement.request_min:.1f}{dash}{measurement.request_max:.1f})"
    n = measurement.requests
    if n == 1:
        return [
            f"measured {NAME} {measurement.instance_id}: {speed}, one request, wall time, context {measurement.context}",
            NOT_RANKED_NOTE,
        ]
    lines = []
    if n < measurement.ranking_requests:
        lines.append(f"{n} of {measurement.ranking_requests} requests measured")
    total = f"{measurement.total_tps:.1f} tok/s total ({measurement.total_min:.1f}{dash}{measurement.total_max:.1f})"
    lines.append(f"measured {NAME} {measurement.instance_id}: {total}, {n} at once, wall time, context {measurement.context}")
    lines.append(f"per request: {speed}")
    if n == measurement.parallel < measurement.ranking_requests:
        lines.append(PARALLEL_NOTE.format(parallel=n))
    lines.append(NOT_RANKED_NOTE)
    return lines


def card_facts(run: "GuidedRun") -> tuple[Fact, ...]:
    """Four rows on the card where step 4 kept a number, else none: the number, then what it is."""
    measurement = run.servers.measured.get(ROW)
    if not isinstance(measurement, LmStudioMeasurement) or measurement.problem is not None:
        return ()
    if measurement.requests == 1:
        note = f"{measurement.request_tps:.1f} tok/s, one request, wall time"
    else:  # at most 52 characters: n x 128 tokens cannot make a total beyond 32768000000000.0 (the column holds 65)
        note = f"{measurement.total_tps:.1f} tok/s total, {measurement.requests} at once, wall time"
    return (
        Fact(LABEL, f"{NAME} server", note),
        Fact("", "", CARD_NOT_RANKED),
        Fact("", "", CARD_BUSY),
        Fact("", "", CARD_UNSEEN),
    )


__all__ = ["LABEL", "NAME", "ROW", "card_facts", "measure", "measurement_lines", "offer", "start_fact"]
