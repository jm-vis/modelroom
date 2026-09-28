"""The vLLM server in the guided mode: its start row, its part of step 4, its rows on the card.

A vLLM server that answers on `127.0.0.1:8000` states which model it serves (`stated`: its own
statement, never measured, never ranked, never provenance). Step 4 says so, says before the one
question whether its number will count for this run (AP9-UX E), offers the load test of
`modelroom/loadtest_vllm.py` -- n requests at once, three batches -- as a row of the one list, and
computes how many requests of the ranking's context one card holds (`modelroom/fit_vllm.py`), for
the one package of this folder the server serves. The card of step 5 carries one row for it.

**Nothing is persisted**: no measurement file, no profile, no snapshot, no configuration, no
render document changes (CONTRACTS.md, "Local servers and the vLLM load test", ADR 0008). What
this module found lives in the run's `ServerRun` and nowhere else. It never imports `guided.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Configuration
from .contracts import Package, SchemaVersionError
from .dialog import Choice
from .fit_vllm import VLLM_GPU_MEMORY_UTILIZATION, VllmFit, largest_card, vllm_fit
from .guided_context import machine_checked
from .guided_loadtest import local_measure_target
from .intro import Fact
from .loadtest import LoadTestError
from .loadtest_vllm import MAX_REQUESTS, MAX_TOKENS, VllmMeasurement, run_vllm_load_test
from .measurements import Scenario
from .scenario_text import requests_words
from .screen import NOTE_INDENT, context_tokens_text, names_words
from .state import UnreadableStateFileError, read_snapshot
from .vllm_server import KEY_REASON, ServedModel, ServedModelsError, VllmKeyError, VllmServerError, read_vllm_models

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

# The value of the server's row in the one list of step 4 (`load_test_packages`).
ROW = "vllm"
LABEL = "server"
NAME = "vLLM"
MEASURED = "vLLM measured"
NOT_COUNTED = "vLLM ran, not counted"
NOT_A_PACKAGE = "no package of this folder"
NOT_MEASURED = "not measured in this run"
REVISION_LINE = "revision unknown (not stated)"
NOTHING_WRITTEN_LINE = "identity stated by the server; not ranked, nothing written"
# Why there is no fit: `guided_context.machine_checked` found no profile the fit may compute for
# (not bound, blocked, or named by no machine of the configuration), or one without a card.
NO_PROFILE_REASON = "this machine has no profile in this folder a fit may compute for"
NO_CARD_REASON = "the profile of this machine names no graphics card"
# What is left of 100 columns for the note of a card row (`views.result_card`).
CARD_NOTE_WIDTH = 65
# A note line holds 100 columns minus the screen's leading blank and the note's own indent.
_NOTE_ROOM = 100 - 1 - len(NOTE_INDENT)
_OTHER_MODEL = "the number will not count: the server serves "
_OTHER_CONTEXT = "the number will not count: the server runs context {server}, this run asks for {run}"
_NO_FIT = "the number counts, no fit: no package of this folder comes from "


@dataclass(frozen=True)
class VllmOutcome:
    """What step 4 found for the vLLM server: the package, the measurement, the fit, or why not.

    `counts` says whether its number counts for this run (AP9-UX E): the served `root` is a model of
    this run and the server runs this run's context.
    """

    package_reason: str | None = None
    measurement: VllmMeasurement | None = None
    fit: VllmFit | None = None
    fit_reason: str | None = None
    counts: bool = True


@dataclass(frozen=True)
class VllmOffer:
    """What the server offers step 4: the served model, the package it serves, and whether it counts."""

    served: ServedModel
    requests: int
    matched: Package | str
    counts: bool
    reason_said: str | None = None


# --- the start screen -------------------------------------------------------------------------------


def start_fact(origins) -> Fact | None:
    """The start row: the model the server states, a server asking for a key, or no row at all.

    What was read is kept in the run for step 4: the served model, or the reason there is none.
    A server that does not answer, or answers with something that is no vLLM model list, has no
    row, so a machine without vLLM draws the start screen it always drew.
    """
    server = origins.servers.transports.get("vllm")
    if server is None:
        return None
    try:
        served = read_vllm_models(server)
    except VllmKeyError:
        origins.servers.found["vllm"] = KEY_REASON
        return Fact(LABEL, server.base_url.removeprefix("http://"), "asks for an API key")
    except ServedModelsError as exc:
        origins.servers.found["vllm"] = str(exc)
        return Fact(LABEL, NAME, str(exc))
    except VllmServerError:
        return None
    origins.servers.found["vllm"] = served
    return Fact(LABEL, NAME, f"context {served.max_model_len}, serves (stated) {served.id}")


# --- which package the server serves ------------------------------------------------------------------


def match_served(packages: list[Package], served: ServedModel) -> Package | str:
    """The one active complete `tensor` package whose repository is the server's `root`, or why not.

    Several such packages -- one repository at several revisions -- are no answer: the server
    does not state its revision, so it names none of them, and there is no fit.
    """
    matching = [
        package
        for package in packages
        if package.active and package.complete and package.format == "tensor" and package.repo == served.root
    ]
    if not matching:
        return f"no package of this folder comes from {served.root}"
    if len(matching) > 1:
        return f"the server names none of the {len(matching)} tensor packages of {served.root}"
    return matching[0]


def _snapshot(config: Configuration):
    try:
        return read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError):
        return None


def _name(repo: str) -> str:
    """What a reader calls a model: the part behind the account (`views._model_name`)."""
    return repo.partition("/")[2] or repo


def run_models(run: "GuidedRun", snapshot) -> tuple[str, ...]:
    """The base models of this run: the ones step 2 chose -- none where it chose none (B, Codex F1)."""
    return tuple(run.selection.models)


def not_counting_reason(served: ServedModel, models: tuple[str, ...], context: int, matched: Package | str) -> str | None:
    """The first reason, in order, why the number will not count or has no fit; `None` where it counts with a fit."""
    if served.root not in models:
        return f"{_OTHER_MODEL}{_name(served.root)}, this run chose {names_words([_name(model) for model in models])}"
    if served.max_model_len != context:
        return _OTHER_CONTEXT.format(server=context_tokens_text(served.max_model_len), run=context_tokens_text(context))
    if isinstance(matched, str):
        return f"{_NO_FIT}{served.root}"
    return None


def wrapped(text: str) -> list[str]:
    """A reason as note lines: one where it fits 100 columns, else broken before the name, never cut."""
    if len(text) <= _NOTE_ROOM:
        return [text]
    for opening in (_OTHER_MODEL, _NO_FIT):
        if text.startswith(opening):
            return [opening.rstrip(), text[len(opening) :]]
    return [text]


# --- step 4 -------------------------------------------------------------------------------------------


def vllm_offer(run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path) -> VllmOffer | None:
    """The vLLM part of step 4 up to the question: its notes, and the offer of its row.

    Only where the start screen read a served model; a server asking for a key, or one whose model
    list did not read, is a note and no row.
    """
    found = run.servers.found.get("vllm")
    if isinstance(found, str):
        run.screen.note(found if found == KEY_REASON else f"{NAME}: {found}")
    if not isinstance(found, ServedModel):
        return None
    if local_measure_target(run, config, results_dir, NAME) is None:
        # The daemon's own check (Codex F2, r3, r4): the server runs on this machine, so it counts only
        # where this machine is the machine of its bound profile and the run chose it.
        return None
    requests = min(scenario.requests, MAX_REQUESTS)
    run.screen.note(f"a vLLM server answers here, context {found.max_model_len}")
    run.screen.note(f"it states it serves {found.id}")
    run.screen.note(REVISION_LINE)
    run.screen.note(f"one warm-up request, then three batches of {requests_words(requests)}, {MAX_TOKENS} tokens each")
    snapshot = _snapshot(config)
    matched = match_served(snapshot.packages if snapshot is not None else [], found)
    reason = not_counting_reason(found, run_models(run, snapshot), scenario.context_requested, matched)
    for line in wrapped(reason) if reason is not None else []:
        run.screen.note(line)
    counts = reason is None or reason.startswith(_NO_FIT)
    return VllmOffer(found, requests, matched, counts, reason)


def vllm_rows(offer: VllmOffer | None) -> list[Choice]:
    """The server's row of the one list, nothing marked: `vLLM server  context 4k  20 requests at once  <id>`."""
    if offer is None:
        return []
    context = f"context {context_tokens_text(offer.served.max_model_len)}"
    return [Choice(ROW, f"vLLM server  {context}  {requests_words(offer.requests)} at once  {offer.served.id}")]


def measure_vllm(
    run: "GuidedRun", config: Configuration, offer: VllmOffer | None, scenario: Scenario, results_dir: Path, picked: list[str]
) -> str | None:
    """Measure the server where its row was picked; `vLLM measured`, `vLLM ran, not counted`, or `None`."""
    if offer is None:
        return None
    measurement = _measure(run, offer.served, scenario) if ROW in picked else None
    matched = offer.matched
    if isinstance(matched, str):
        if offer.reason_said is None or not offer.reason_said.startswith(_NO_FIT):
            run.screen.note(matched)
        outcome = VllmOutcome(package_reason=matched, measurement=measurement, counts=offer.counts)
    else:
        snapshot = _snapshot(config)
        base_models = snapshot.base_models if snapshot is not None else []
        fit, reason = _fit(run, config, results_dir, matched, base_models, scenario)
        outcome = VllmOutcome(measurement=measurement, fit=fit, fit_reason=reason, counts=offer.counts)
    run.servers.measured["vllm"] = outcome
    if measurement is None:
        return None
    # `vLLM measured` only where the number counts **and** the measurement holds: an invalid or not
    # comparable one says so on the card, and the balance may not say otherwise (Diff-Review M1).
    return MEASURED if offer.counts and measurement.problem is None else NOT_COUNTED


def _measure(run: "GuidedRun", served: ServedModel, scenario: Scenario) -> VllmMeasurement | None:
    """Run the load test; a fault is one line and a step that did not finish."""
    try:
        measurement = run_vllm_load_test(run.servers.transports["vllm"], served, scenario)
    except LoadTestError as exc:
        run.failed(f"nothing measured for vLLM: {exc}")
        return None
    for line in measurement_lines(measurement, run.screen.glyphs.skip):
        run.screen.note(line)
    return measurement


def measurement_lines(measurement: VllmMeasurement, dash: str) -> list[str]:
    """The notes of a measurement: the two speeds, or the reason in their place, then what it is."""
    lines = []
    if measurement.requests != measurement.ranking_requests:
        lines.append(f"{measurement.requests} of {requests_words(measurement.ranking_requests)} measured")
    if measurement.problem is not None:
        lines.append(f"{NAME}: {measurement.problem}")
    else:
        total = f"{measurement.total_tps:.1f} tok/s total ({measurement.total_min:.1f}{dash}{measurement.total_max:.1f})"
        request = f"{measurement.request_tps:.1f} tok/s ({measurement.request_min:.1f}{dash}{measurement.request_max:.1f})"
        lines.append(f"measured vLLM: {total}, context {context_tokens_text(measurement.context)}")
        lines.append(f"per request: {request}")
    lines.append(NOTHING_WRITTEN_LINE)
    return lines


def _fit(
    run: "GuidedRun", config: Configuration, results_dir: Path, package: Package, base_models: list, scenario: Scenario
) -> tuple[VllmFit | None, str | None]:
    """The fit on the largest card of the profile this machine is bound to, or the reason there is none."""
    checked = machine_checked(run.pointer_path, config, results_dir)
    card = largest_card(checked.profile) if checked.profile is not None else None
    if card is None:
        reason = NO_CARD_REASON if checked.profile is not None else NO_PROFILE_REASON
        run.screen.note(f"no vLLM fit: {reason}")
        return None, reason
    base = next((spec for spec in base_models if spec.hf_repo == package.base_model_hf_repo), None)
    weights = sum(file.size_bytes for file in package.files if file.role in ("weights", "weights_shard"))
    fit = vllm_fit(
        card,
        weights,
        context=scenario.context_requested,
        requests=scenario.requests,
        architecture=base.architecture if base is not None else None,
    )
    for line in fit_lines(fit):
        run.screen.note(line)
    return fit, None


def fit_lines(fit: VllmFit) -> list[str]:
    """The three notes of a vLLM fit: the card, the three parts of its budget, and the requests it holds."""
    basis = ", from size" if fit.basis == "size" else ""
    return [
        f"vLLM fit, computed: one card of {fit.card_gib:.2f} GiB, {VLLM_GPU_MEMORY_UTILIZATION} of it for vLLM",
        f"weights {fit.weights_gib:.2f} GiB, runtime {fit.runtime_gib:.2f} GiB (one recording), "
        f"KV cache {fit.kv_gib:.2f} GiB",
        f"{requests_words(fit.at_once)} of context {context_tokens_text(fit.context)} at once{basis}: "
        f"{fit.fit_class.replace('_', ' ')}",
    ]


# --- the card of step 5 ---------------------------------------------------------------------------------


def card_facts(run: "GuidedRun") -> tuple[Fact, ...]:
    """The vLLM rows of the card: the number and whether it counts, or why there is none; then the identity."""
    outcome: Any = run.servers.measured.get("vllm")
    if outcome is None:
        return ()
    return (Fact(LABEL, NAME, server_note(outcome)), Fact("", "", _identity_note(outcome, run.screen.glyphs.dot)))


def server_note(outcome: VllmOutcome) -> str:
    """The note of the server row (AP9-UX E): the number, the requests, and whether it counts."""
    measurement = outcome.measurement
    if measurement is None:
        return NOT_MEASURED
    if measurement.total_tps is None:
        return _clip(measurement.problem or NOT_MEASURED, CARD_NOTE_WIDTH)
    speed = f"{measurement.total_tps:.1f} tok/s total, {requests_words(measurement.requests)} at once"
    if not outcome.counts:
        # Another model or another context: the number stands, and that it does not count (reason 1 or 2),
        # which also covers the context the load test itself calls not comparable.
        return f"{speed}, not counted"
    if measurement.problem is not None:
        return _clip(measurement.problem, CARD_NOTE_WIDTH)
    return f"{speed}, {'counts, no fit' if outcome.package_reason is not None else 'counts for this run'}"


def _identity_note(outcome: VllmOutcome, dot: str) -> str:
    if outcome.package_reason is not None:
        return f"identity stated {dot} {NOT_A_PACKAGE}"
    if outcome.fit is None:
        return f"identity stated {dot} no fit computed"
    return f"identity stated {dot} fit: {outcome.fit.at_once} at once, {outcome.fit.fit_class.replace('_', ' ')}"


def _clip(text: str, width: int) -> str:
    """`text` in at most `width` characters, with `…` where the rest of the reason was cut."""
    return text if len(text) <= width else text[: width - 1] + "…"


__all__ = [
    "ROW",
    "VllmOffer",
    "VllmOutcome",
    "card_facts",
    "fit_lines",
    "match_served",
    "measure_vllm",
    "measurement_lines",
    "not_counting_reason",
    "server_note",
    "start_fact",
    "vllm_offer",
    "vllm_rows",
]
