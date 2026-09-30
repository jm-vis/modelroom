"""Step 4's second question round (AP9-FR, FR-B2): the load test on the active SSH target.

Runs after the local servers' own round (`local_servers.load_tests`), and only for `run.ssh_active`
-- no other stored SSH profile is ever measured, and a run with no active target asks nothing at
all (B4, GF question 6). Two calls to the counterpart (`remote_client.candidates`/`measure`) do the
work; this module is the guided-mode glue: the packages of this run, the two questions
(`load_test_remote`, `load_test_remote_packages`), the progress line, and the double sink pass on
the answer before it is stored (FR-N, Codex r2-3).

Every name this module shows or stores is built from a package identity the laptop already knows
(`pull_choice.package_pull_name`) -- never an installed name of the server, which `candidates`'s and
`measure`'s own results never carry (`CONTRACTS.md`, "The remote protocol v1").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import Configuration
from .contracts import Package
from .dialog import AnswerMissingError, Choice, columns
from .guided_loadtest import NOTHING_MEASURED, _CANDIDATE_WIDTHS, _GIB, _UNKNOWN, _measurement_note, _record_problem
from .loadtest import store_measurement
from .measurements import MeasurementExistsError, MeasurementRecord, Scenario
from .neutral import NeutralError, neutral_record
from .pull_choice import package_pull_name, run_packages, weights_bytes_of
from .quantization import package_identity_key
from .remote_client import candidates as ssh_candidates
from .remote_client import measure as ssh_measure
from .remote_client import FORM_ERROR
from .remote_store import aliases as remote_aliases
from .remote_store import remote_path_for
from .screen import names_words, yes_no
from .ssh import SshError
from .state import LockHeldError

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

KEY_REMOTE = "load_test_remote"
KEY_REMOTE_PACKAGES = "load_test_remote_packages"
QUESTIONS: dict[str, str] = {
    KEY_REMOTE: "Measure the speed on {alias} now?",
    KEY_REMOTE_PACKAGES: "Which of these should be measured on {alias}?",
}


@dataclass(frozen=True)
class RemoteOffer:
    """What the active SSH target offers step 4's second round, before the question: the packages
    of this run (to build every name and row from) and the candidate identities the server found
    among its own installed packages (B2, `candidates`)."""

    alias: str | None = None
    profile_id: str | None = None
    packages: tuple[Package, ...] = ()
    identities: tuple[tuple[str, str, str], ...] = ()


def _alias_profile_id(run: "GuidedRun", alias: str) -> str | None:
    """The `profile_id` the alias is bound to (B3/B4) -- the id every measurement of it is stored
    under (B2: stored under the alias's own profile_id)."""
    store_path = remote_path_for(run.pointer_path)
    return next((pid for pid, name in remote_aliases(store_path).items() if name == alias), None)


def remote_offer(run: "GuidedRun", config: Configuration) -> RemoteOffer:
    """The active SSH target's candidates, or an empty offer -- no active target, no binding, no
    packages of this run, or a failed `candidates` call all end the round here, before any question
    (B4: no active SSH target asks nothing; the same holds with no candidate at all)."""
    alias = run.ssh_active
    if alias is None:
        return RemoteOffer()
    profile_id = _alias_profile_id(run, alias)
    packages = tuple(run_packages(config, run.selection))
    if profile_id is None or not packages:
        return RemoteOffer()
    try:
        result = ssh_candidates(alias, packages, run.ssh)
    except SshError as exc:
        run.failed(str(exc))
        return RemoteOffer()
    identities = tuple(tuple(entry["package_identity"]) for entry in result["candidates"])
    return RemoteOffer(alias, profile_id, packages, identities)


def _package_of(identity: tuple[str, str, str], packages: tuple[Package, ...]) -> Package | None:
    return next((package for package in packages if package_identity_key(package) == identity), None)


def _shown_name(package: Package) -> str:
    return package_pull_name(package) or package.base_model_hf_repo


def _remote_rows(offer: RemoteOffer) -> list[Choice]:
    """One row per candidate the server found, named and sized from the package this laptop already
    knows (never the server's own installed name) -- same column widths as the local list."""
    entries = [(identity, package) for identity in offer.identities if (package := _package_of(identity, offer.packages)) is not None]
    cells = [
        [_shown_name(package), package.base_model_hf_repo, package.quantization or _UNKNOWN, f"{weights_bytes_of(package) / _GIB:.2f} GiB"]
        for _identity, package in entries
    ]
    labels = columns(cells, _CANDIDATE_WIDTHS)
    return [Choice(_shown_name(package), label) for (_identity, package), label in zip(entries, labels)]


def _ask_remote(run: "GuidedRun", alias: str, rows: list[Choice]) -> list[str]:
    """The one question and the one list of this round (B2), only reached with at least one row."""
    question = QUESTIONS[KEY_REMOTE].format(alias=alias)
    try:
        wants = run.asker.confirm(KEY_REMOTE, question, default=False)
    except AnswerMissingError:
        wants = False
    run.screen.answer(question, yes_no(wants))
    if not wants:
        return []
    list_question = QUESTIONS[KEY_REMOTE_PACKAGES].format(alias=alias)
    picked = run.asker.checkbox(KEY_REMOTE_PACKAGES, list_question, rows)
    run.screen.answer(list_question, names_words(list(picked), "nothing"))
    return list(picked)


def _progress_reporter(run: "GuidedRun", shown_name: str):
    def on_progress(entry: dict) -> None:
        if entry.get("stage") == "warmup":
            run.screen.note(f"{shown_name}: warm-up run")
        elif entry.get("stage") == "run":
            run.screen.note(f"{shown_name}: run {entry.get('number')} of {entry.get('of')}")

    return on_progress


def _measure_one(run: "GuidedRun", config: Configuration, offer: RemoteOffer, identity: tuple[str, str, str], scenario: Scenario) -> bool:
    """One candidate, measured on `offer.alias`; a fault is one line, never a traceback (mirrors
    `guided_loadtest._measure_candidate`)."""
    package = _package_of(identity, offer.packages)
    if package is None:
        return False
    shown_name = _shown_name(package)
    try:
        result = ssh_measure(
            offer.alias, identity, offer.packages, scenario, offer.profile_id, run.ssh,
            on_progress=_progress_reporter(run, shown_name),
        )
    except SshError as exc:
        run.failed(str(exc))
        return False
    try:
        record = MeasurementRecord.model_validate(result["record"])
    except Exception:  # noqa: BLE001 - any shape mismatch is the one fixed sentence, never a traceback
        run.failed(FORM_ERROR.format(alias=offer.alias))
        return False
    return _store_checked(run, config, offer, package, record, shown_name)


def _store_checked(run: "GuidedRun", config: Configuration, offer: RemoteOffer, package: Package, record: MeasurementRecord, shown_name: str) -> bool:
    """The double sink (B2): the laptop runs the record through the same sink (FR-N) again, and a
    result that differs from what the server sent -- the server did not hold the rules, or the
    `profile_id` is not the alias's own -- is discarded with the one fixed sentence, never stored."""
    clean = neutral_record(record, package)
    if clean.model_dump(mode="json") != record.model_dump(mode="json") or record.profile_id != offer.profile_id:
        run.failed(FORM_ERROR.format(alias=offer.alias))
        return False
    try:
        store_measurement(config.paths.state, record, run.now, package)
    except (LockHeldError, MeasurementExistsError, OSError, NeutralError) as exc:
        run.failed(f"the measurement of {shown_name} was not stored: {exc}")
        return False
    problem = _record_problem(record)
    if problem is not None:
        run.screen.note(f"{record.ollama_name}: {problem}")
        return True
    run.screen.note(_measurement_note(record, run.screen.glyphs.skip))
    return True


def remote_load_test(run: "GuidedRun", config: Configuration, scenario: Scenario) -> str | None:
    """Step 4's second round, whole: the offer, the two questions, then every picked candidate --
    `None` where nothing was asked or nothing was picked (no active target, no candidate, `no`, or
    an empty pick), a summary line otherwise."""
    offer = remote_offer(run, config)
    if offer.alias is None or not offer.identities:
        return None
    rows = _remote_rows(offer)
    if not rows:
        return None
    picked = _ask_remote(run, offer.alias, rows)
    if not picked:
        return None
    chosen = [identity for identity in offer.identities if (package := _package_of(identity, offer.packages)) is not None and _shown_name(package) in picked]
    one_request = scenario.model_copy(update={"requests": 1})
    measured = sum(_measure_one(run, config, offer, identity, one_request) for identity in chosen)
    return f"{measured} of {len(chosen)} measured on {offer.alias}"


def combine_measured(run: "GuidedRun", config: Configuration, scenario: Scenario, local_measured: str) -> str:
    """`run_guided`'s one call for step 4's summary line: the local servers' own result, joined with
    this round's, or unchanged where nothing here was asked or measured."""
    remote_measured = remote_load_test(run, config, scenario)
    if remote_measured is None:
        return local_measured
    dot = f" {run.screen.glyphs.dot} "
    return remote_measured if local_measured == NOTHING_MEASURED else local_measured + dot + remote_measured


__all__ = [
    "KEY_REMOTE",
    "KEY_REMOTE_PACKAGES",
    "QUESTIONS",
    "RemoteOffer",
    "combine_measured",
    "remote_load_test",
    "remote_offer",
]
