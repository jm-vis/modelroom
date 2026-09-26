"""Step 4 of the guided mode: the load test of the packages this machine already has.

The step itself -- the two questions, the selection list, the progress and the result lines.
The work is `modelroom/loadtest.py`'s, the dialog is `modelroom/dialog.py`'s, and the order of
the steps stays in `modelroom/guided.py`, which calls `load_test_step` between the context and
the render. It lives in its own module so `guided.py` keeps holding the run and nothing else; it
never imports `guided.py` back (CONTRACTS.md, "Guided mode", step 4).
"""

from __future__ import annotations

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
from .screen import context_tokens_text, names_words, yes_no
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

QUESTIONS: dict[str, str] = {
    "load_test": "Measure the speed of the checked models that are already installed here?",
    "load_test_packages": "Which of these installed models should be measured?",
}

NO_CANDIDATE_LINE = (
    "no ranked package is installed on this machine; the load test measures installed "
    "packages only (stage 1: no download)"
)
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
    return snapshot.packages


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


def _wants_load_test(run: "GuidedRun") -> bool:
    """The `load_test` answer, with `no` for an answer file that does not mention it.

    The one optional answer of the guided mode, and deliberately so: whether this question is
    reached at all depends on the machine -- whether it runs a daemon, and which packages it
    already has -- so one answer file has to work on a machine that would never be asked. A file
    that wants a measurement says `load_test = true` (CONTRACTS.md, "Guided mode"). At a terminal
    nothing is optional: the question is asked, with `no` as the default answer.
    """
    try:
        return run.asker.confirm("load_test", QUESTIONS["load_test"], default=False)
    except AnswerMissingError:
        return False


def _no_daemon(run: "GuidedRun", reason: str) -> None:
    """The daemon could not be read: a user who asked for a measurement gets a fault, not a shrug.

    The question comes first, so a run that never wanted a load test ends `0` on a machine with
    no Ollama daemon at all -- the daemon is needed for this step and for nothing else. A yes
    that cannot be honored is a step that did not finish (exit `1`).
    """
    if not _wants_load_test(run):
        run.screen.answer(QUESTIONS["load_test"], yes_no(False))
        run.screen.note(f"no load test: {reason}")
        return
    run.screen.answer(QUESTIONS["load_test"], yes_no(True))
    run.failed(f"nothing measured: {reason}")


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


def load_test_step(
    run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path, ranking_requests: int = 1
) -> str:
    """Step 4: measure installed packages, with the context this ranking is computed for.

    `scenario` is the one a measurement runs, always one request; `ranking_requests` is what the
    ranking of this run assumes, and when that is more than one the step says so first.

    Returns the one line the finished step leaves behind. Stage 1 measures what the daemon
    already has. A folder this machine is not bound to a profile in, and one that holds no
    snapshot, are lines and no more -- there is nothing a load test could even be about. From the
    daemon on, a `load_test` yes that cannot be honored is a step that did not finish
    (`run.failed`, exit `1`), and the document is still written (CONTRACTS.md, "Load test
    (stage 1)").
    """
    run.screen.blank()
    run.screen.head(STEP)
    if ranking_requests > 1:
        run.screen.note(ONE_REQUEST_LINE.format(requests=ranking_requests))
    profile_id = _bound_profile_id(run, config, results_dir)
    if profile_id is None:
        return NOTHING_MEASURED
    packages = _snapshot_packages(run, config)
    if packages is None:
        return NOTHING_MEASURED
    installed, reason = installed_models(run.daemon, run.now)
    if installed is None:
        _no_daemon(run, reason)
        return NOTHING_MEASURED
    inventory = installed_candidates(packages, installed, lambda name: weight_digest(run.daemon, name))
    _report_set_aside(run, inventory, cloud=not inventory.candidates)
    if not inventory.candidates:
        run.screen.note(NO_CANDIDATE_LINE)
        return NOTHING_MEASURED
    wants = _wants_load_test(run)
    run.screen.answer(QUESTIONS["load_test"], yes_no(wants))
    if not wants:
        return NOTHING_MEASURED
    return _measure_picked(run, config, inventory, scenario, profile_id)


def _measure_picked(
    run: "GuidedRun", config: Configuration, inventory: Inventory, scenario: Scenario, profile_id: str
) -> str:
    """Ask which of the candidates to measure, measure them, and say how many were measured."""
    choices = _candidate_choices(inventory.candidates)
    picked = run.asker.checkbox("load_test_packages", QUESTIONS["load_test_packages"], choices)
    run.screen.answer(QUESTIONS["load_test_packages"], names_words(picked, "nothing"))
    if not picked:
        return NOTHING_MEASURED
    run.screen.note(LOAD_NOTE_LINE)
    measured = 0
    for candidate in inventory.candidates:
        if candidate.ollama_name in picked:
            measured += _measure_candidate(run, config, candidate, scenario, profile_id)
    return f"{measured} of {len(picked)} measured"
