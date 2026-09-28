"""Step 4 of the guided mode: the daemon's part of the load test of the packages this machine has.

The daemon's notes, its candidates, the progress and the result lines; the one question and the
one list of step 4 are `local_servers.load_tests`' (AP9-UX E). The work is `modelroom/loadtest.py`'s,
the dialog is `modelroom/dialog.py`'s, and the order of the steps stays in `modelroom/guided.py`.
It never imports `guided.py` back (CONTRACTS.md, "Guided mode", step 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .binding import KnownProfile, read_pointer, resolve_profile_target
from .config import Configuration
from .contracts import Package, SchemaVersionError
from .dialog import AnswerMissingError, Choice, columns
from .importer import scan_profiles
from .loadtest import (
    CLOUD_REASON,
    Candidate,
    Inventory,
    LoadTestError,
    installed_candidates,
    installed_models,
    run_load_test,
    store_measurement,
    weight_digest,
)
from .measure import read_os_identity
from .measurements import MeasurementExistsError, MeasurementRecord, Scenario
from .profile import os_fingerprint
from .screen import context_tokens_text
from .selection import narrow_snapshot
from .state import LockHeldError, UnreadableStateFileError, read_snapshot

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

_GIB = 1024**3
_UNKNOWN = "unknown"
_CANDIDATE_WIDTHS = (46, 40, 10)
STEP = 4
# The summary of a step that measured nothing -- and then the step's balance carries a dash, not a
# check mark: it did nothing, which is not the same as being done. Deliberately not the wording of
# the fault lines above ("nothing measured: <reason>"): a reader of the log must be able to tell
# the step's own summary from a measurement that was asked for and did not happen.
NOTHING_MEASURED = "none in this run"

# One question and one list for every local server (AP9-UX E): the question asks whether to measure
# at all, the list which of what this machine offers.
QUESTIONS: dict[str, str] = {
    "load_test": "Measure the speed on this machine now?",
    "load_test_packages": "Which of these should be measured?",
}

NO_CANDIDATE_LINE = (
    "no ranked package is installed on this machine; the load test measures installed "
    "packages only (stage 1: no download)"
)
# Where another server offers a row, the daemon's side of the same fact names itself.
NO_INSTALLED_LINE = "Ollama: no package of this run's models is installed here"
# Said before anything else in step 4 when the ranking assumes more than one request: it explains
# the note a measured row then carries in step 5 (`measured with 1 request, ranking assumes N`).
ONE_REQUEST_LINE = "measurements run one request; this ranking assumes {requests}"
LOAD_NOTE_LINE = (
    "the load on this machine is read once, just before the runs -- load during a run is not "
    "measurable -- and the model behind the name must not change while the runs are made"
)


def _bound_profile_id(run: "GuidedRun", config: Configuration, results_dir: Path) -> str | None:
    """The profile **this** machine may measure into here, or `None` with a line saying so.

    Not `[machines.<name>].profile`: a configuration is shared, and a stale or foreign entry
    there would send this machine's measurements into another device's profile -- which is
    exactly the case the hardware step declines with "put that profile file back". The decision
    is the takeover rule itself (`binding.resolve_profile_target`), asked with the home binding
    alone: only its `bound` outcome measures. So the profile file has to be in the folder **and**
    its `os_fingerprint` has to agree with this machine's, which is what makes a declined
    `same machine` answer earlier in the run keep this step from measuring too.
    """
    bound = read_pointer(run.pointer_path).binding_for(results_dir)
    if bound is None:
        run.screen.note("no load test: this machine is not bound to a profile in this results folder")
        return None
    scan = scan_profiles(config.paths.hardware_dir)
    known = {
        key: KnownProfile(key, profile.os_fingerprint, profile.origin, profile.ram_physical_source)
        for key, profile in scan.profiles.items()
    }
    identity = read_os_identity(run.probes.platform, run.probes.runner, run.probes.read_text)
    local = os_fingerprint(identity.raw_id) if identity.raw_id else "none"
    # `resolve_profile_target` refuses a `fresh_profile_id` that is already a profile; any unused
    # one will do here, because only its `bound` outcome is accepted and the others never use it.
    unused = next(candidate for candidate in ("0" * 16, "1" * 16, "2" * 16) if candidate not in known)
    target = resolve_profile_target(bound, None, known, local, unused)
    if target.action != "bound":
        run.screen.note(f"no load test: {target.reason} ({target.profile_id})")
        return None
    return bound


def _snapshot_packages(run: "GuidedRun", config: Configuration) -> list[Package] | None:
    """The packages the fetch recorded, or `None` with a line saying why there are none.

    A stored file that does not read is said out loud here and left to the render, which ends
    the run with the exit code that file's own reader gives it (`3`) -- the load test never
    flattens it into one of its own.
    """
    try:
        snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError) as exc:
        run.screen.note(f"no load test: {exc}")
        return None
    if snapshot is None:
        run.screen.note("no load test: nothing has been fetched into this folder yet")
        return None
    # The candidates are packages of this run's models alone (AP9-UX B).
    return narrow_snapshot(snapshot, run.selection).packages


def _candidate_rows(candidates: list[Candidate]) -> list[str]:
    """The lines of the load test's selection list: the four facts, aligned in columns."""
    rows = [
        [
            candidate.ollama_name,
            candidate.package.base_model_hf_repo,
            candidate.package.quantization or _UNKNOWN,
            f"{candidate.weights_bytes / _GIB:.2f} GiB",
        ]
        for candidate in candidates
    ]
    return columns(rows, _CANDIDATE_WIDTHS)


def _candidate_choices(candidates: list[Candidate]) -> list[Choice]:
    """Nothing is marked: which models are measured is a decision, and Enter must not make it.

    A single candidate is a list with one entry as well -- one code path, and an answer file that
    names it reads the same on a machine with ten installed packages as on one with one.
    """
    return [
        Choice(candidate.ollama_name, label)
        for candidate, label in zip(candidates, _candidate_rows(candidates))
    ]


def _report_set_aside(run: "GuidedRun", inventory: Inventory, *, cloud: bool) -> None:
    """Say which installed models were left out, and why: one note per reason, never in silence.

    One note per reason with the number of installed models it covers and **no names** (decided
    2026-09-24): which model exactly is `ollama list`'s answer, and twelve names pushed the question
    off the screen. `cloud` is off once there is a candidate to measure -- that a machine also holds
    cloud models is then beside the point. An `unmatched` entry is not: it means the daemon did not
    show the digest of a model whose name matches a configured package, which is a fault about a
    package a reader may have wanted measured, and it stays either way (second-model round,
    2026-09-24).
    """
    grouped: dict[str, int] = {}
    for entry in inventory.unmatched:
        grouped[entry.reason] = grouped.get(entry.reason, 0) + 1
    if cloud and inventory.cloud:
        grouped[CLOUD_REASON] = len(inventory.cloud)
    for reason, count in sorted(grouped.items()):
        models = f"{count} installed model" + ("" if count == 1 else "s")
        run.screen.note(f"{models} -- {reason}")


def _measurement_note(record: MeasurementRecord, dash: str) -> str:
    """The one note the user reads about a measurement that was just written.

    The short form of it: the speed, its range and the context. That the record is valid and
    comparable is what makes it a measurement at all, so it is not said -- a record that is
    neither is a fault line instead (`_record_problem`).
    """
    context = f"context {context_tokens_text(record.scenario.context_requested)}"
    speed = f"{record.tps_mean:.1f} tok/s ({record.tps_min:.1f}{dash}{record.tps_max:.1f})"
    return f"measured {record.ollama_name}: {speed}, {context}"


def _record_problem(record: MeasurementRecord) -> str | None:
    """Why this measurement cannot be read as one, or `None` when it can."""
    if record.tps_mean is None:
        return f"no speed, {record.validity} ({record.validity_reason})"
    if record.validity != "valid":
        return f"{record.validity} ({record.validity_reason})"
    if not record.comparable:
        return f"not comparable ({record.comparable_reason})"
    return None


def _measure_candidate(
    run: "GuidedRun", config: Configuration, candidate: Candidate, scenario: Scenario, profile_id: str
) -> bool:
    """Measure one candidate and publish the record; a fault is one line, never a traceback."""
    try:
        record = run_load_test(
            run.daemon,
            candidate,
            scenario,
            profile_id,
            probes=run.probes,
            measured_at=run.now,
            # The progress of a run that takes minutes belongs on the screen, in the same column
            # the notes of this step stand in (decided 2026-09-24).
            progress=run.screen.note,
        )
    except LoadTestError as exc:
        run.failed(f"nothing measured for {candidate.ollama_name}: {exc}")
        return False
    try:
        store_measurement(config.paths.state, record, run.now)
    except (LockHeldError, MeasurementExistsError, OSError) as exc:
        run.failed(f"the measurement of {candidate.ollama_name} was not stored: {exc}")
        return False
    problem = _record_problem(record)
    if problem is not None:
        # The record is written and the run stays `0`: a measurement that does not count in
        # measured group 0 is still a measurement of this machine, and the ranking says where it
        # stands ("Load test (stage 1)"). What it is missing is said here, once.
        run.screen.note(f"{record.ollama_name}: {problem}")
        return True
    run.screen.note(_measurement_note(record, run.screen.glyphs.skip))
    return True


@dataclass(frozen=True)
class OllamaOffer:
    """What the daemon offers step 4: its candidates, or why it has none (the one question is asked
    for every server at once, `local_servers.load_tests`, AP9-UX E)."""

    candidates: tuple[Candidate, ...] = ()
    profile_id: str | None = None
    unavailable: str | None = None


def installed_note(count: int) -> str:
    """`Ollama: 2 installed packages of this run's models, one request each`, or none installed."""
    if not count:
        return NO_INSTALLED_LINE
    packages = "1 installed package" if count == 1 else f"{count} installed packages"
    return f"Ollama: {packages} of this run's models, one request each"


def ollama_offer(
    run: "GuidedRun", config: Configuration, results_dir: Path, ranking_requests: int = 1, named: bool = False
) -> OllamaOffer:
    """Step 4 of the daemon up to the question: its notes, and the candidates it offers.

    A folder this machine is not bound to a profile in, and one that holds no snapshot, are lines
    and no more -- there is nothing a load test could even be about. A daemon that does not answer
    is `unavailable`: a `yes` that only it could honor is a step that did not finish.
    """
    if ranking_requests > 1:
        run.screen.note(("Ollama " if named else "") + ONE_REQUEST_LINE.format(requests=ranking_requests))
    profile_id = _bound_profile_id(run, config, results_dir)
    if profile_id is None:
        return OllamaOffer()
    packages = _snapshot_packages(run, config)
    if packages is None:
        return OllamaOffer()
    installed, reason = installed_models(run.daemon, run.now)
    if installed is None:
        return OllamaOffer(unavailable=reason)
    inventory = installed_candidates(packages, installed, lambda name: weight_digest(run.daemon, name))
    _report_set_aside(run, inventory, cloud=not inventory.candidates)
    return OllamaOffer(tuple(inventory.candidates), profile_id)


def ollama_rows(offer: OllamaOffer) -> list[Choice]:
    """The daemon's rows of the one list: nothing marked (`_candidate_choices`)."""
    return _candidate_choices(list(offer.candidates))


def measure_ollama(run: "GuidedRun", config: Configuration, offer: OllamaOffer, scenario: Scenario, picked: list[str]) -> str | None:
    """Measure the picked candidates; `<k> of <n> measured`, or `None` where none was picked."""
    chosen = [candidate for candidate in offer.candidates if candidate.ollama_name in picked]
    if not chosen or offer.profile_id is None:
        return None
    run.screen.note(LOAD_NOTE_LINE)
    measured = sum(_measure_candidate(run, config, candidate, scenario, offer.profile_id) for candidate in chosen)
    return f"{measured} of {len(chosen)} measured"


def wants_load_test(run: "GuidedRun") -> bool:
    """The `load_test` answer, with `no` for an answer file that does not mention it.

    The one optional answer of the guided mode, and deliberately so: whether this question is
    reached at all depends on the machine -- whether it runs a server, and which packages it
    already has -- so one answer file has to work on a machine that would never be asked. At a
    terminal nothing is optional: the question is asked, with `no` as the default answer.
    """
    try:
        return run.asker.confirm("load_test", QUESTIONS["load_test"], default=False)
    except AnswerMissingError:
        return False

