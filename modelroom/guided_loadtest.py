"""Step 5 of the guided mode: the load test of the packages this machine already has.

The step itself -- the two questions, the selection list, the progress and the result lines.
The work is `modelroom/loadtest.py`'s, the dialog is `modelroom/dialog.py`'s, and the order of
the steps stays in `modelroom/guided.py`, which calls `load_test_step` between the fetch and the
render. It lives in its own module so `guided.py` keeps holding the run and nothing else; it
never imports `guided.py` back (CONTRACTS.md, "Guided mode", step 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .binding import KnownProfile, read_pointer, resolve_profile_target
from .config import Configuration
from .contracts import Package, SchemaVersionError
from .dialog import AnswerMissingError, Choice
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
from .state import LockHeldError, UnreadableStateFileError, read_snapshot

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

_GIB = 1024**3
_UNKNOWN = "unknown"

QUESTIONS: dict[str, str] = {
    "load_test": "Measure the speed of the checked models that are already installed here?",
    "load_test_packages": "Which of these installed models should be measured?",
}

NO_CANDIDATE_LINE = (
    "no ranked package is installed on this machine; the load test measures installed "
    "packages only (stage 1: no download)"
)
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
        run.out("no load test: this machine is not bound to a profile in this results folder")
        return None
    scan = scan_profiles(config.paths.hardware_dir)
    known = {key: KnownProfile(key, profile.os_fingerprint) for key, profile in scan.profiles.items()}
    identity = read_os_identity(run.probes.platform, run.probes.runner, run.probes.read_text)
    local = os_fingerprint(identity.raw_id) if identity.raw_id else "none"
    # `resolve_profile_target` refuses a `fresh_profile_id` that is already a profile; any unused
    # one will do here, because only its `bound` outcome is accepted and the others never use it.
    unused = next(candidate for candidate in ("0" * 16, "1" * 16, "2" * 16) if candidate not in known)
    target = resolve_profile_target(bound, None, known, local, unused)
    if target.action != "bound":
        run.out(f"no load test: {target.reason} ({target.profile_id})")
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
        run.out(f"no load test: {exc}")
        return None
    if snapshot is None:
        run.out("no load test: nothing has been fetched into this folder yet")
        return None
    return snapshot.packages


def _candidate_label(candidate: Candidate) -> str:
    """One line of the load test's selection list: the four facts about what would be measured."""
    package = candidate.package
    weights = f"{candidate.weights_bytes / _GIB:.2f} GiB"
    return (
        f"{candidate.ollama_name}  |  {package.base_model_hf_repo}  |  "
        f"{package.quantization or _UNKNOWN}  |  {weights}"
    )


def _candidate_choices(candidates: list[Candidate]) -> list[Choice]:
    return [Choice(candidate.ollama_name, _candidate_label(candidate), checked=True) for candidate in candidates]


def _report_set_aside(run: "GuidedRun", inventory: Inventory) -> None:
    """Say which installed models were left out, and why -- once per reason, never in silence."""
    for entry in inventory.unmatched:
        run.out(f"not measured: {entry.name} -- {entry.reason}")
    if inventory.cloud:
        run.out(f"not measured ({CLOUD_REASON}): {', '.join(inventory.cloud)}")


def _measurement_line(record: MeasurementRecord) -> str:
    """The one line the user reads about a measurement that was just written."""
    context = f"context {record.scenario.context_requested}"
    comparable = "comparable" if record.comparable else f"not comparable ({record.comparable_reason})"
    if record.tps_mean is None:
        return f"{record.ollama_name}: no speed, {context}, {record.validity} ({record.validity_reason})"
    speed = f"measured {record.tps_mean:.1f} tok/s ({record.tps_min:.1f}-{record.tps_max:.1f})"
    return f"{record.ollama_name}: {speed}, {context}, {record.validity}, {comparable}"


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
        run.out(f"no load test: {reason}")
        return
    run.failed(f"nothing measured: {reason}")


def _measure_candidate(
    run: "GuidedRun", config: Configuration, candidate: Candidate, scenario: Scenario, profile_id: str
) -> None:
    """Measure one candidate and publish the record; a fault is one line, never a traceback."""
    try:
        record = run_load_test(
            run.daemon,
            candidate,
            scenario,
            profile_id,
            probes=run.probes,
            measured_at=run.now,
            progress=run.out,
        )
    except LoadTestError as exc:
        run.failed(f"nothing measured for {candidate.ollama_name}: {exc}")
        return
    try:
        store_measurement(config.paths.state, record, run.now)
    except (LockHeldError, MeasurementExistsError, OSError) as exc:
        run.failed(f"the measurement of {candidate.ollama_name} was not stored: {exc}")
        return
    run.out(_measurement_line(record))


def load_test_step(run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path) -> None:
    """Step 5: measure installed packages, with the context this ranking is computed for.

    Stage 1 measures what the daemon already has. A folder this machine is not bound to a profile
    in, and one that holds no snapshot, are lines and no more -- there is nothing a load test
    could even be about. From the daemon on, a `load_test` yes that cannot be honored is a step
    that did not finish (`run.failed`, exit `1`), and the document is still written (CONTRACTS.md,
    "Load test (stage 1)").
    """
    profile_id = _bound_profile_id(run, config, results_dir)
    if profile_id is None:
        return
    packages = _snapshot_packages(run, config)
    if packages is None:
        return
    installed, reason = installed_models(run.daemon, run.now)
    if installed is None:
        _no_daemon(run, reason)
        return
    inventory = installed_candidates(packages, installed, lambda name: weight_digest(run.daemon, name))
    _report_set_aside(run, inventory)
    if not inventory.candidates:
        run.out(NO_CANDIDATE_LINE)
        return
    if not _wants_load_test(run):
        return
    choices = _candidate_choices(inventory.candidates)
    picked = run.asker.checkbox("load_test_packages", QUESTIONS["load_test_packages"], choices)
    if not picked:
        run.out("nothing measured: no installed model was picked")
        return
    run.out(LOAD_NOTE_LINE)
    for candidate in inventory.candidates:
        if candidate.ollama_name in picked:
            _measure_candidate(run, config, candidate, scenario, profile_id)
