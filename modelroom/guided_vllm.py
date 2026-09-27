"""The vLLM server in the guided mode: its start row, its part of step 4, its rows on the card.

A vLLM server that answers on `127.0.0.1:8000` states which model it serves (`stated`: its own
statement, never measured, never ranked, never provenance). Step 4 then says so, offers the load
test of `modelroom/loadtest_vllm.py` -- n requests at once, three batches -- and computes how many
requests of the ranking's context one card holds (`modelroom/fit_vllm.py`), for the one package of
this folder the server serves. The card of step 5 carries one row for it.

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
from .dialog import AnswerMissingError
from .fit_vllm import VLLM_GPU_MEMORY_UTILIZATION, VllmFit, largest_card, vllm_fit
from .guided_context import machine_checked
from .intro import Fact
from .loadtest import LoadTestError
from .loadtest_vllm import MAX_REQUESTS, MAX_TOKENS, VllmMeasurement, run_vllm_load_test
from .measurements import Scenario
from .screen import context_tokens_text, yes_no
from .state import UnreadableStateFileError, read_snapshot
from .vllm_server import KEY_REASON, ServedModel, ServedModelsError, VllmKeyError, VllmServerError, read_vllm_models

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

KEY = "load_test_vllm"
QUESTION = "Measure the vLLM server with {requests} requests at once?"
QUESTIONS: dict[str, str] = {KEY: QUESTION}
LABEL = "server"
NAME = "vLLM"
MEASURED = "vLLM measured"
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
_GIB = 1024**3


@dataclass(frozen=True)
class VllmOutcome:
    """What step 4 found for the vLLM server: the package, the measurement, the fit, or why not."""

    package_reason: str | None = None
    measurement: VllmMeasurement | None = None
    fit: VllmFit | None = None
    fit_reason: str | None = None


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


# --- step 4 -------------------------------------------------------------------------------------------


def vllm_step(run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path) -> str | None:
    """The vLLM part of step 4; returns `vLLM measured`, or `None` where nothing was measured.

    Only where the start screen read a served model. The measurement needs no package and no
    profile -- it measures the server, and nothing is written -- but the fit needs both.
    """
    found = run.servers.found.get("vllm")
    if isinstance(found, str):
        run.screen.note(found if found == KEY_REASON else f"{NAME}: {found}")
    if not isinstance(found, ServedModel):
        return None
    run.screen.note(f"a vLLM server answers here, context {found.max_model_len}")
    run.screen.note(f"it states it serves {found.id}")
    run.screen.note(REVISION_LINE)
    snapshot = _snapshot(config)
    matched = match_served(snapshot.packages if snapshot is not None else [], found)
    measurement = _measure(run, found, scenario)
    if isinstance(matched, str):
        run.screen.note(matched)
        run.servers.measured["vllm"] = VllmOutcome(package_reason=matched, measurement=measurement)
    else:
        fit, reason = _fit(run, config, results_dir, matched, snapshot.base_models, scenario)
        run.servers.measured["vllm"] = VllmOutcome(measurement=measurement, fit=fit, fit_reason=reason)
    return MEASURED if measurement is not None else None


def _wants(run: "GuidedRun", question: str) -> bool:
    """The third optional answer: an answer file without `load_test_vllm` measures nothing."""
    try:
        return run.asker.confirm(KEY, question, default=False)
    except AnswerMissingError:
        return False


def _measure(run: "GuidedRun", served: ServedModel, scenario: Scenario) -> VllmMeasurement | None:
    """Offer the load test and run it; a fault is one line and a step that did not finish."""
    requests = min(scenario.requests, MAX_REQUESTS)
    run.screen.note(f"one warm-up request, then three batches of {requests} requests, {MAX_TOKENS} tokens each")
    question = QUESTION.format(requests=requests)
    wants = _wants(run, question)
    run.screen.answer(question, yes_no(wants))
    if not wants:
        return None
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
        lines.append(f"{measurement.requests} of {measurement.ranking_requests} requests measured")
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
        f"{fit.at_once} requests of context {context_tokens_text(fit.context)} at once{basis}: "
        f"{fit.fit_class.replace('_', ' ')}",
    ]


# --- the card of step 5 ---------------------------------------------------------------------------------


def card_facts(run: "GuidedRun") -> tuple[Fact, ...]:
    """The vLLM rows of the card: the speed, or the reason there is none, and the identity and fit."""
    outcome: Any = run.servers.measured.get("vllm")
    if outcome is None:
        return ()
    if outcome.package_reason is not None:
        return (Fact(LABEL, NAME, NOT_A_PACKAGE),)
    measurement = outcome.measurement
    if measurement is None:
        note = NOT_MEASURED
    elif measurement.problem is not None:
        note = _clip(measurement.problem, CARD_NOTE_WIDTH)
    else:
        note = f"{measurement.total_tps:.1f} tok/s measured (vLLM, {measurement.requests} requests, total)"
    return (Fact(LABEL, NAME, note), Fact("", "", _identity_note(outcome, run.screen.glyphs.dot)))


def _identity_note(outcome: VllmOutcome, dot: str) -> str:
    if outcome.fit is None:
        return f"identity stated {dot} no fit computed"
    return f"identity stated {dot} fit: {outcome.fit.at_once} at once, {outcome.fit.fit_class.replace('_', ' ')}"


def _clip(text: str, width: int) -> str:
    """`text` in at most `width` characters, with `…` where the rest of the reason was cut."""
    return text if len(text) <= width else text[: width - 1] + "…"


__all__ = [
    "KEY",
    "QUESTION",
    "QUESTIONS",
    "VllmOutcome",
    "card_facts",
    "fit_lines",
    "match_served",
    "measurement_lines",
    "start_fact",
    "vllm_step",
]
