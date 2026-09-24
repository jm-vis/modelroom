"""The load test, stage 1: measure packages that are already installed on this machine.

Stage 1 downloads nothing, removes nothing and says nothing about disk space. It reads the
daemon's own inventory (`GET /api/tags`), keeps the entries a configured package provably is --
provably meaning a digest the daemon itself shows, never a name that merely looks right -- and
runs measurement protocol v1 against each of them. The result is a `MeasurementRecord` written
as its own file under `modelroom.lock`; the renderer reads it back on its own.

This module writes measurements. It does not rank them and it does not render them
(CONTRACTS.md, "Load test (stage 1)").
"""

from __future__ import annotations

import os
import re
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .contracts import InstalledModel, Package
from .daemon import GENERATE_PATH, PS_PATH, SHOW_PATH, VERSION_PATH, Daemon, DaemonError
from .http import Response
from .llmfit import Runner
from .measure import Probes
from .measurements import (
    MEASUREMENT_SCHEMA_VERSION,
    LoadState,
    MeasurementRecord,
    PackageRef,
    Protocol,
    RunCounters,
    Scenario,
    measurement_invalid_reason,
    new_measurement_id,
    shipped_protocol,
    write_measurement,
)
from .ollama_local import fetch_installed_models
from .state import acquire_lock, release_lock

LOCK_COMMAND = "load-test"
# Read once, just before the runs; load *during* a run cannot be read from outside the run.
GPU_UTILIZATION_ARGS = ("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")

_WEIGHT_ROLES = ("weights", "weights_shard")
_SHA256_PREFIX = "sha256:"
# `hf.co/<owner>/<repo>:<quantization>` is the local name Ollama gives a package it pulled from
# Hugging Face; nothing else is read out of the name, the digest decides.
_HF_NAME_RE = re.compile(r"^hf\.co/(?P<repo>[^:/]+/[^:/]+):(?P<quantization>[^:/]+)$")
# The blob path may hold spaces (`FROM C:\Model Cache\blobs\sha256-<hex>`), so the part before the
# digest is `.*`, not `\S*`; only the commented lines `ollama show` writes above it are excluded,
# and they do not start with `FROM`.
_BLOB_RE = re.compile(r"(?m)^FROM\s+.*sha256-(?P<digest>[0-9a-f]{64})\s*$")

CLOUD_REASON = "a cloud model, not a package on this machine"
NO_DIGEST_REASON = "no weight digest in the daemon's /api/show answer"
SHARDED_REASON = (
    "the daemon shows one blob, and stage 1 cannot prove the content of a package whose weights "
    "are several files"
)


class LoadTestError(Exception):
    """The measurement could not be made, and the message says why.

    Every way the daemon can let a run down ends here -- unreachable, a status other than 200,
    an answer that carries no counters, a limit that was reached, a model that is gone -- so a
    caller reports one line and goes on. A run that raises this has written nothing.
    """


# --- the inventory ------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One installed model, the active package it provably is, and the reference to record."""

    installed: InstalledModel
    package: Package
    package_ref: PackageRef

    @property
    def ollama_name(self) -> str:
        return self.installed.name

    @property
    def weights_bytes(self) -> int:
        """How much this package's weight files weigh, the same sum the ranking sorts on."""
        return sum(file.size_bytes for file in self.package.files if file.role in _WEIGHT_ROLES)


@dataclass(frozen=True)
class Unmatched:
    """An installed model that names a configured package but cannot prove it is that package."""

    name: str
    reason: str


@dataclass
class Inventory:
    """What one look at `/api/tags` found: what can be measured, and what was left out why."""

    candidates: list[Candidate] = field(default_factory=list)
    unmatched: list[Unmatched] = field(default_factory=list)
    cloud: list[str] = field(default_factory=list)


WeightDigestLookup = Callable[[str], "tuple[str | None, str | None]"]


def installed_models(daemon: Daemon, now: datetime) -> tuple[list[InstalledModel] | None, str | None]:
    """The daemon's own inventory, through the reader `hardware` already uses.

    `ollama_local.fetch_installed_models` speaks the `(method, url, headers)` transport shape,
    so the daemon is handed to it through a two-line adapter rather than a second parser: one
    place reads `/api/tags`, whoever asks.
    """

    def transport(method: str, url: str, headers: dict[str, str] | None = None) -> Response:
        return daemon(method, urllib.parse.urlsplit(url).path)

    return fetch_installed_models(transport, now, daemon.base_url)


def is_cloud(installed: InstalledModel) -> bool:
    """Whether this entry is a model the daemon runs elsewhere, not a package on this machine.

    The daemon lists a cloud model with a stub of a few hundred bytes and a tag of `cloud` or
    one ending in `-cloud` (`kimi-k3:cloud`, `gpt-oss:20b-cloud`, read from a real daemon on
    2026-09-24). A size of zero says the same thing for an entry with any other tag.
    """
    tag = installed.name.rpartition(":")[2]
    return installed.size_bytes == 0 or tag == "cloud" or tag.endswith("-cloud")


def installed_candidates(
    packages: list[Package], installed: list[InstalledModel], show: WeightDigestLookup
) -> Inventory:
    """Which installed models are which active package, by digest and by digest only.

    An Ollama package is the entry whose `/api/tags` digest is its manifest digest. A Hugging
    Face package is an entry named `hf.co/<repo>:<quantization>` for that repository **and**
    whose weight digest, as `show` reads it from the daemon, is one of the package's own weight
    files. A name that matches without such a digest is listed under `unmatched` and never
    measured: the package may not claim what the daemon does not show.
    """
    active = [package for package in packages if package.active]
    inventory = Inventory()
    for entry in sorted(installed, key=lambda model: model.name):
        if is_cloud(entry):
            inventory.cloud.append(entry.name)
            continue
        candidate = _ollama_candidate(active, entry)
        if candidate is not None:
            inventory.candidates.append(candidate)
            continue
        _add_hugging_face(inventory, active, entry, show)
    return inventory


def _ollama_candidate(packages: list[Package], entry: InstalledModel) -> Candidate | None:
    for package in packages:
        if package.source == "ollama" and package.manifest_digest == entry.digest:
            ref = PackageRef(content_source="ollama", ollama_manifest_digest=package.manifest_digest)
            return Candidate(installed=entry, package=package, package_ref=ref)
    return None


def _add_hugging_face(
    inventory: Inventory, packages: list[Package], entry: InstalledModel, show: WeightDigestLookup
) -> None:
    """Add the Hugging Face candidate for `entry`, or the reason there is none to `unmatched`."""
    match = _HF_NAME_RE.match(entry.name)
    if match is None:
        return
    repo = match.group("repo")
    named = [package for package in packages if package.source == "huggingface" and package.repo == repo]
    if not named:
        return
    digest, reason = show(entry.name)
    if digest is None:
        inventory.unmatched.append(Unmatched(entry.name, reason or NO_DIGEST_REASON))
        return
    sharded = False
    for package in named:
        if digest not in _weight_digests(package):
            continue
        # One blob proves one file. A package whose weights are several files would be credited
        # with a measurement of whichever single file the daemon holds, so it stays unmatched
        # until stage 2 can prove the whole content.
        if len(_weight_files(package)) != 1:
            sharded = True
            continue
        ref = PackageRef(
            content_source="huggingface",
            hf_repo=package.repo,
            hf_revision=package.revision,
            hf_file_digest=digest,
        )
        inventory.candidates.append(Candidate(installed=entry, package=package, package_ref=ref))
        return
    inventory.unmatched.append(
        Unmatched(
            entry.name,
            SHARDED_REASON if sharded else f"the digest the daemon shows is no weight file of {repo}",
        )
    )


def _weight_files(package: Package) -> list:
    return [file for file in package.files if file.role in _WEIGHT_ROLES]


def _weight_digests(package: Package) -> set[str]:
    return {file.digest for file in _weight_files(package) if file.digest}


def weight_digest(daemon: Daemon, name: str) -> tuple[str | None, str | None]:
    """The digest of the blob the daemon built `name` from, or `(None, reason)`.

    `POST /api/show` answers with a Modelfile whose `FROM` line names the blob on disk, as
    `…/blobs/sha256-<hex>` (read from a real daemon on 2026-09-24,
    `tests/fixtures/ollama_show_hf_gguf.json`). For a package pulled from Hugging Face that hex
    is the `sha256` of the GGUF file itself, which is the digest the fetch recorded for that
    file -- which is why a match here is evidence and a matching name is not.
    """
    try:
        answer = daemon("POST", SHOW_PATH, {"model": name})
    except DaemonError as exc:
        return None, str(exc)
    if answer.status != 200:
        return None, f"unexpected status {answer.status} from {SHOW_PATH}"
    try:
        data = answer.json()
    except (ValueError, RecursionError) as exc:
        return None, f"cannot read the {SHOW_PATH} answer: {exc}"
    modelfile = data.get("modelfile") if isinstance(data, dict) else None
    found = _BLOB_RE.search(modelfile) if isinstance(modelfile, str) else None
    if found is None:
        return None, NO_DIGEST_REASON
    return _SHA256_PREFIX + found.group("digest"), None


# --- the measured run ----------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """What `/api/ps` showed about the measured model at one point in the run.

    `loaded` is every local name the daemon listed at that moment, so a measurement that found
    its own model missing can say what was there instead. `name` is `None` exactly when the
    measured model was not among them.
    """

    name: str | None = None
    digest: str | None = None
    context_length: int | None = None
    loaded: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return self.name is None


def run_load_test(
    daemon: Daemon,
    candidate: Candidate,
    scenario: Scenario,
    profile_id: str,
    *,
    probes: Probes,
    measured_at: datetime,
    cpu_load: "Callable[[], float | None]" = None,
    progress: Callable[[str], None] = lambda _line: None,
) -> MeasurementRecord:
    """Run the shipped measurement protocol against `candidate` and return the record; write nothing.

    The order is the protocol's: the daemon's version, the machine's load, the warm-up run,
    then the measured runs, with `/api/ps` read after every one of them -- so every measured run
    stands between two observations. The record is returned, never stored: `store_measurement`
    does that, under the lock.

    The protocol is not a parameter. The record this writes says `protocol: v1`, and
    `MeasurementRecord` checks a valid one against `shipped_protocol()` -- a run after another
    prompt, seed or warm-up count would be stored under a name it does not deserve. A new
    protocol version is a new writer, never an argument here (CONTRACTS.md, "Measurement
    protocol v1").
    """
    protocol = shipped_protocol()
    version = _daemon_version(daemon)
    load_state = LoadState(
        gpu_utilization_percent=gpu_utilization(probes.runner),
        cpu_load=(cpu_load or read_cpu_load)(),
    )
    observations: list[Observation] = []
    for _ in range(protocol.warmup_runs):
        progress(f"{candidate.ollama_name}: warm-up run")
        _generate(daemon, candidate, scenario, protocol)
        observations.append(_observe(daemon, candidate.ollama_name))
    runs: list[RunCounters] = []
    for number in range(1, protocol.measured_runs + 1):
        progress(f"{candidate.ollama_name}: run {number} of {protocol.measured_runs}")
        runs.append(_generate(daemon, candidate, scenario, protocol))
        observations.append(_observe(daemon, candidate.ollama_name))
    return _record(candidate, scenario, profile_id, version, load_state, runs, observations, measured_at, protocol)


def _daemon_version(daemon: Daemon) -> str | None:
    """The daemon's own version, stored with the measurement; it is never a validity criterion."""
    answer = _ask(daemon, "GET", VERSION_PATH, None)
    if answer.status != 200:
        raise LoadTestError(f"unexpected status {answer.status} from {VERSION_PATH}")
    try:
        data = answer.json()
    except (ValueError, RecursionError) as exc:
        raise LoadTestError(f"cannot read the {VERSION_PATH} answer: {exc}") from exc
    version = data.get("version") if isinstance(data, dict) else None
    return str(version) if version is not None else None


def _ask(daemon: Daemon, method: str, path: str, body: dict | None) -> Response:
    try:
        return daemon(method, path, body)
    except DaemonError as exc:
        raise LoadTestError(str(exc)) from exc


def _generate(daemon: Daemon, candidate: Candidate, scenario: Scenario, protocol: Protocol) -> RunCounters:
    """One `POST /api/generate` exactly as the protocol prescribes it, as raw counters."""
    body = {
        "model": candidate.ollama_name,
        "prompt": protocol.prompt,
        "raw": protocol.options.raw,
        "stream": protocol.options.stream,
        "options": {
            "num_ctx": scenario.context_requested,
            "num_predict": protocol.options.num_predict,
            "temperature": protocol.options.temperature,
            "seed": protocol.options.seed,
        },
    }
    answer = _ask(daemon, "POST", GENERATE_PATH, body)
    if answer.status != 200:
        raise LoadTestError(f"unexpected status {answer.status} from {GENERATE_PATH} for {candidate.ollama_name}")
    try:
        data = answer.json()
    except (ValueError, RecursionError) as exc:
        raise LoadTestError(f"cannot read the {GENERATE_PATH} answer: {exc}") from exc
    return _counters(data)


def _counters(data: object) -> RunCounters:
    """The five counters plus the end reason, as the daemon returned them.

    A counter the daemon leaves out because nothing happened (no prompt token was evaluated, the
    model was already loaded) reads as zero; `done_reason`, `eval_count` and `eval_duration` are
    what a measurement is, so an answer without them is not one.
    """
    if not isinstance(data, dict) or not isinstance(data.get("done_reason"), str):
        raise LoadTestError(f"the {GENERATE_PATH} answer carries no done_reason: {str(data)[:120]}")
    try:
        return RunCounters(
            done_reason=data["done_reason"],
            eval_count=data["eval_count"],
            eval_duration=data["eval_duration"],
            prompt_eval_count=data.get("prompt_eval_count", 0),
            prompt_eval_duration=data.get("prompt_eval_duration", 0),
            load_duration=data.get("load_duration", 0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LoadTestError(f"the {GENERATE_PATH} answer carries no usable counters: {exc}") from exc


def _observe(daemon: Daemon, name: str) -> Observation:
    """What `/api/ps` says about `name` right now, plus every model it listed.

    The measured model is looked up by its local name rather than taken from the first entry: a
    daemon can hold several models at once, and another one of them standing first is no
    statement about this measurement. An answer this module cannot read is an empty observation.
    """
    answer = _ask(daemon, "GET", PS_PATH, None)
    if answer.status != 200:
        raise LoadTestError(f"unexpected status {answer.status} from {PS_PATH}")
    try:
        data = answer.json()
    except (ValueError, RecursionError) as exc:
        raise LoadTestError(f"cannot read the {PS_PATH} answer: {exc}") from exc
    models = data.get("models") if isinstance(data, dict) else None
    entries = [entry for entry in models if isinstance(entry, dict)] if isinstance(models, list) else []
    loaded = tuple(str(entry.get("name")) for entry in entries)
    found = next((entry for entry in entries if entry.get("name") == name), None)
    if found is None:
        return Observation(loaded=loaded)
    digest = found.get("digest")
    return Observation(
        name=name,
        digest=_SHA256_PREFIX + digest if isinstance(digest, str) else None,
        context_length=found.get("context_length"),
        loaded=loaded,
    )


def not_comparable_reason(
    observations: list[Observation], candidate: Candidate, scenario: Scenario
) -> str | None:
    """Why this whole measurement may not be compared with another, or `None` when it may.

    Every observation has to show the measured model, under its own name, with the package's
    digest and the ranking's context. An observation that does not show it at all counts as a
    miss, and names what was loaded instead: a model that was not loaded at that moment cannot
    have produced the run around it.
    """
    if not observations:
        return "no observation of /api/ps was made"
    for observation in observations:
        if observation.empty:
            if observation.loaded:
                return f"/api/ps showed {', '.join(observation.loaded)}, not {candidate.ollama_name}"
            return "no observation of the measured model in /api/ps"
        if observation.digest != candidate.installed.digest:
            return f"the digest of {candidate.ollama_name} changed during the measurement"
        if observation.context_length != scenario.context_requested:
            return (
                f"the daemon loaded context {observation.context_length}, "
                f"not the ranking's {scenario.context_requested}"
            )
    return None


def _mean_within(speeds: list[float]) -> float | None:
    """The mean of `speeds` (sorted), held inside its own minimum and maximum.

    The mean of a set of numbers cannot lie outside it -- but a float sum can: three runs of
    2 100 000 000 ns each give `60.95238095238095` tok/s and a mean of `60.95238095238094`, one
    unit in the last place below the minimum (measured). `MeasurementRecord` requires
    `tps_min <= tps_mean <= tps_max` and would refuse that perfectly good measurement, so the
    rounding is corrected here rather than the measurement thrown away. The correction is one
    unit in the last place and stays well inside the `rel_tol` of `_check_speeds_match_runs`.
    """
    if not speeds:
        return None
    mean = sum(speeds) / len(speeds)
    return min(max(mean, speeds[0]), speeds[-1])


def _record(
    candidate: Candidate,
    scenario: Scenario,
    profile_id: str,
    version: str | None,
    load_state: LoadState,
    runs: list[RunCounters],
    observations: list[Observation],
    measured_at: datetime,
    protocol: Protocol,
) -> MeasurementRecord:
    """Everything the run learned, as the one record it is stored as."""
    invalid = measurement_invalid_reason(runs, protocol, scenario.context_requested)
    not_comparable = not_comparable_reason(observations, candidate, scenario)
    # `RunCounters` bounds nothing at the top end, so a nonsense duration passes both it and the
    # protocol rule and only overflows here. That is a daemon this module cannot measure, not a
    # measurement with a reason -- and `MeasurementRecord` would raise the same thing one step
    # later, outside every `except` the guided step has.
    try:
        speeds = sorted(run.tokens_per_second() for run in runs) if invalid is None else []
        return MeasurementRecord(
            schema_version=MEASUREMENT_SCHEMA_VERSION,
            measurement_id=new_measurement_id(measured_at),
            profile_id=profile_id,
            protocol="v1",
            measured_at=measured_at,
            package=candidate.package_ref,
            ollama_name=candidate.ollama_name,
            daemon_version=version,
            scenario=scenario,
            runs=runs,
            tps_mean=_mean_within(speeds),
            tps_min=speeds[0] if speeds else None,
            tps_max=speeds[-1] if speeds else None,
            validity="invalid" if invalid else "valid",
            validity_reason=invalid,
            comparable=not_comparable is None,
            comparable_reason=not_comparable,
            load_state=load_state,
        )
    # `RunCounters` bounds nothing at the top end, so a nonsense duration passes both it and the
    # protocol rule and only overflows when a speed is computed from it. And `MeasurementRecord`
    # has rules of its own that an answer no daemon should give could still break. Both are a
    # daemon this module cannot measure, not a measurement with a reason -- and both would
    # otherwise leave the guided step as a traceback rather than a line.
    except (OverflowError, ValueError) as exc:
        raise LoadTestError(f"the daemon's counters are not a measurement this protocol can record: {exc}") from exc


# --- the machine's load ----------------------------------------------------------------------


def gpu_utilization(runner: Runner) -> float | None:
    """The GPU's utilization in percent from `nvidia-smi`, or `None` when it cannot be read.

    A machine without an NVIDIA GPU, an `nvidia-smi` that is not installed, one that fails or
    one that answers with something else are all the same here: no value, and no failure of the
    measurement (the same rule `modelroom/measure.py` follows for every hardware source).
    """
    try:
        finished = runner(list(GPU_UTILIZATION_ARGS))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    first = (finished.stdout or "").strip().splitlines()
    try:
        value = float(first[0].strip())
    except (IndexError, ValueError):
        return None
    return value if 0 <= value <= 100 else None


def read_cpu_load() -> float | None:
    """The system's one-minute load average where the platform has one, else `None`.

    Windows has no load average; `os.getloadavg` does not exist there, and no value is claimed
    instead of one.
    """
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return None
    try:
        return float(getloadavg()[0])
    except (OSError, IndexError, ValueError):
        return None


# --- the file ----------------------------------------------------------------------------------


def store_measurement(state_dir: Path, record: MeasurementRecord, now: datetime) -> Path:
    """Publish `record` as its own file under `modelroom.lock`, the way every writer here does.

    `LockHeldError` reaches the caller unchanged -- another process holding the lock is exit `1`
    for every command in this package, and the load test is no exception.
    """
    handle = acquire_lock(state_dir / "modelroom.lock", LOCK_COMMAND, now)
    try:
        return write_measurement(state_dir, record, handle)
    finally:
        release_lock(handle)
