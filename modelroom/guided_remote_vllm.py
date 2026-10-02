"""vLLM at the active computer over SSH, in the guided mode (AP9-SV2-R, R3).

One computer over SSH is active in a run (FR B4). What `hello` says of its vLLM (`hello.vllm`) is kept in
the run (`GuidedRun.remote_vllm`, in memory only) and goes into the runtimes that computer has *live* in
this run -- the set its packages are chosen by, F3 -- exactly as a vLLM server this computer finds on its
own machine does. Step 4 asks the server which package of the run it serves (`vllm`, R2) and offers a row
`vLLM server on gpu1  context 32k  10 requests at once  <package name>` in the list of the other candidates
of that computer; measuring it is `vllm-measure`, with its progress, and the number lives in the run and
on the card, nothing written (E3). Step 5 turns `room.model_disk` and the bytes still to fetch into the
disk rule of the vLLM rows of that computer (I2). Choosing a vLLM row there prints the two commands of I1
with `on gpu1` in the label and starts nothing (GF question F1): `pull_other.py`.

Every name shown is built from this computer's own package (`remote_vllm.py`); this module opens no
connection by itself and never imports `guided.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .config import Configuration
from .dialog import Choice
from .guided_vllm import MEASURED, NOT_COUNTED, VllmOutcome, measurement_lines, not_counting_reason, run_models, server_note, wrapped
from .intro import Fact
from .loadtest_vllm import MAX_REQUESTS, MAX_TOKENS
from .measurements import Scenario
from .pull_choice import run_packages
from .remote_vllm import KEY_NOTE, NO_PACKAGE_NOTE, RemoteVllm, read_state, sendable, vllm_measure
from .room import Disk
from .scenario_text import requests_words
from .screen import context_tokens_text
from .ssh import SshError
from .vllm_server import ServedModel

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .contracts import Package
    from .guided import GuidedRun

ROW = "vllm"
LABEL = "server"


def _state(run: "GuidedRun", alias: str | None) -> RemoteVllm | None:
    return getattr(run, "remote_vllm", {}).get(alias) if alias is not None else None


def remember_hello(run: "GuidedRun", alias: str, info: dict) -> None:
    """Keep what `hello` of `alias` said of its vLLM (`absent` where an older server says nothing)."""
    run.remote_vllm[alias] = RemoteVllm(state=info.get("vllm", "absent"))


def live_map(run: "GuidedRun") -> dict[str, object]:
    """`{alias: live}` of the active computer over SSH, where its vLLM was detected live (serving or
    installed): what `render` and the views take as "found in this run" for a machine (`present_runtimes`).
    A server that asks for a key, or has no vLLM, is not in it (E5)."""
    alias = getattr(run, "ssh_active", None)
    state = _state(run, alias)
    live = state.live() if state is not None else None
    return {alias: live} if live is not None else {}


@dataclass(frozen=True)
class RemoteOffer:
    """What the active computer's vLLM offers step 4: the package it serves, its context and whether the
    number counts for this run -- the reason (`guided_vllm.not_counting_reason`) where it does not."""

    alias: str
    index: int
    package: "Package"
    served: ServedModel
    requests: int
    reason: str | None

    @property
    def counts(self) -> bool:
        return self.reason is None


def vllm_offer(run: "GuidedRun", config: Configuration, scenario: Scenario) -> RemoteOffer | None:
    """Step 4's vLLM part for the active computer, up to the question: asks the server (`vllm`) which package
    of the run it serves, says what it found, and returns the offer -- or `None` with the one note that says
    why there is none (a key asked for, a model that is no package of this run). Nothing for a computer whose
    `hello` found no vLLM, nothing where the run has no tensor package."""
    alias = getattr(run, "ssh_active", None)
    state = _state(run, alias)
    if alias is None or state is None or state.state == "absent":
        return None
    packages = sendable(run_packages(config, run.selection))
    if not packages:
        return None
    try:
        fresh = read_state(alias, packages, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return None
    fresh.relative, fresh.disk = state.relative, state.disk
    run.remote_vllm[alias] = fresh
    if fresh.state == "asks_key":
        run.screen.note(KEY_NOTE.format(alias=alias))
        return None
    if fresh.state != "serving":
        return None
    if fresh.served_index is None:
        run.screen.note(NO_PACKAGE_NOTE.format(alias=alias))
        return None
    package = packages[fresh.served_index]
    context = fresh.served_context or 1
    served = ServedModel(id=package.repo, root=package.repo, max_model_len=context)
    requests = min(scenario.requests, MAX_REQUESTS)
    run.screen.note(f"a vLLM server answers on {alias}, context {context}")
    run.screen.note(f"it serves {package.repo}, a package of this run")
    run.screen.note(f"one warm-up request, then three batches of {requests_words(requests)}, {MAX_TOKENS} tokens each")
    reason = not_counting_reason(served, run_models(run, None), scenario.context_requested, package)
    for line in wrapped(reason) if reason is not None else []:
        run.screen.note(line)
    return RemoteOffer(alias, fresh.served_index, package, served, requests, reason)


def vllm_row(offer: RemoteOffer) -> Choice:
    """`vLLM server on gpu1  context 32k  10 requests at once  Qwen/Qwen3-8B-AWQ`, nothing marked: in the
    one list of the computer's candidates (`load_test_remote_packages`, value `vllm`)."""
    context = f"context {context_tokens_text(offer.served.max_model_len)}"
    return Choice(ROW, f"vLLM server on {offer.alias}  {context}  {requests_words(offer.requests)} at once  {offer.package.repo}")


def row_words(alias: str, value: str) -> str:
    """The answer line's word for a picked value: the vLLM row names the computer, every other value is a name already."""
    return f"vLLM server on {alias}" if value == ROW else value


def measure(run: "GuidedRun", offer: RemoteOffer, scenario: Scenario) -> str | None:
    """`vllm-measure` at the server for the offered package: the progress, then the notes of a measurement
    (SV1's, naming the computer), kept in the run for the card. `vLLM measured` or `vLLM ran, not counted`;
    `None` where nothing was measured (the fault is one line, never a traceback)."""
    state = _state(run, offer.alias)
    name = offer.package.repo

    def on_progress(entry: dict) -> None:
        if entry.get("stage") == "warmup":
            run.screen.note(f"{name}: warm-up request")
        else:
            run.screen.note(f"{name}: batch {entry.get('number')} of {entry.get('of')}")

    try:
        measurement = vllm_measure(offer.alias, state.packages, offer.index, scenario, run.ssh, on_progress=on_progress)
    except SshError as exc:
        run.failed(str(exc))
        return None
    for line in measurement_lines(measurement, run.screen.glyphs.skip, f" on {offer.alias}"):
        run.screen.note(line)
    state.outcome = VllmOutcome(measurement=measurement, counts=offer.counts)
    return MEASURED if offer.counts and measurement.problem is None else NOT_COUNTED


def card_facts(run: "GuidedRun") -> tuple[Fact, ...]:
    """The rows of the card for the vLLM server of the active computer, if it was measured: the number and
    whether it counts, then what its identity is (stated by the server, not ranked)."""
    alias = getattr(run, "ssh_active", None)
    state = _state(run, alias)
    outcome = state.outcome if state is not None else None
    if outcome is None:
        return ()
    return (Fact(LABEL, f"vLLM on {alias}", server_note(outcome)), Fact("", "", f"identity stated {run.screen.glyphs.dot} not ranked"))


def with_model_disk(run: "GuidedRun", alias: str, disk: Disk, model_disk: dict | None) -> Disk:
    """`disk` with vLLM's own place nested in it, from what `room` said (`room.model_disk`, R1): the free bytes
    and whether it is the drive of Ollama's place, a relative cache variable, and -- from what step 4's `vllm`
    operation counted -- the bytes still to fetch per package and the packages that are there already (I2, I3).
    A server that said nothing of it leaves `disk` as it was (`disk unknown on gpu1` for vLLM's place)."""
    state = _state(run, alias)
    if model_disk is not None:
        needs, held = state.needs() if state is not None else ({}, frozenset())
        known = model_disk["known"]
        vllm = Disk(
            model_disk["free_bytes"] if known else None, alias if known else None, None,
            held=held, needs=needs, relative=model_disk["relative"], same_place=model_disk["same_as_ollama"],
        )
        disk = replace(disk, vllm=vllm)
    if state is not None:
        state.relative = model_disk["relative"] if model_disk is not None else None
        state.disk = disk
    return disk


__all__ = [
    "ROW",
    "RemoteOffer",
    "card_facts",
    "live_map",
    "measure",
    "remember_hello",
    "row_words",
    "vllm_offer",
    "vllm_row",
    "with_model_disk",
]
