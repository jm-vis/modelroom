"""Step 3 of the guided mode: how many people use the model, then how much text it should handle.

First the head count: how many people use it on a typical day, turned into the parallel requests
the ranking assumes by a rule of thumb that is said under the list (`config.requests_from_users`),
or the requests named outright as `N requests`. Then the size scale: six levels from XS to XXL,
each with its number of tokens, roughly how many words that is, an example of what it is good
for, and how many of the packages of this results folder still fit
the machine being checked. The last column is the whole point of the scale: it is the fit of
fit contract v1, computed by `fit.count_fitting` over the packages the fetch of this same run
recorded, against the profile the pointer file binds this machine to for this folder. A level
the base models cannot even hold is grayed out.

Everything the scale knows comes from somewhere else: the formula from `modelroom/fit.py`, the
"it fits" predicate from `modelroom/ranking.py`, the packages from the snapshot, the reserves
from `[machines.<name>]`. It is read when the question is asked, like every other list of this
dialog; the write that follows reads the configuration again on its own.

The head count comes first because the last column of the scale counts with its requests: a
column computed for one request would show a number the ranking of the same run then contradicts.

This module asks both questions and writes the answers; the order of the steps stays in
`modelroom/guided.py` (CONTRACTS.md, "Guided mode", step 3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .binding import read_pointer
from .config import (
    ACTIVE_SHARE,
    MAX_REQUESTS,
    MAX_USERS,
    Configuration,
    GuidedConfig,
    MachineConfig,
    RequestsOrigin,
    requests_from_users,
)
from .contracts import BaseModelSpec, Package, SchemaVersionError
from .dialog import Choice, columns
from .fit import count_fitting
from .guided_write import combined, with_context, with_requests
from .importer import scan_profiles
from .measurements import Scenario
from .profile import HardwareProfile, fit_block_reason
from .state import UnreadableStateFileError, read_snapshot

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

STEP = 3

QUESTION = "How much text should a model handle at once?"
# What the last column of the list means, in the instruction line of the question rather than as a
# line of the run (decided 2026-09-24): it explains the list, so it belongs to the list and goes
# away with it. `<machine>` is the machine the column was computed against.
EXPLANATION = "last column: how many fetched packages fit {machine}"
NUMBER_VALUE = "number"
NUMBER_LABEL = "enter a number"
NUMBER_QUESTION = "How many tokens?"
CUSTOM_LABEL = "custom"

NO_MACHINE_LINE = "no measured machine in this folder yet, so the scale shows no fit"
NO_ENTRY_LINE = (
    "no machine of this configuration names the profile {profile_id} of this machine, "
    "so the scale shows no fit"
)
NO_PACKAGES = "no packages yet"

_COLUMN_WIDTHS = (6, 6, 14, 36)

# The head count (decided 2026-09-26): asked in people, because that is the number a user knows;
# the ranking computes with parallel requests, and the rule between the two stands under the list.
USERS_KEY = "users"
USERS_QUESTION = "How many people use it on a typical day?"
USERS_EXPLANATION = f"assumes 1 in {round(1 / ACTIVE_SHARE)} of them at once (rule of thumb, not measured)"
USERS_NUMBER_QUESTION = "How many people? (or: N requests)"
USERS_LEVELS = (1, 5, 10, 25, 100)
USERS_FORMS = f"a whole number of people (1 to {MAX_USERS}) or `N requests` (1 to {MAX_REQUESTS})"
# Digits 0-9 only: `int()` would also take `+25`, `1_000` and digits of other scripts.
_USERS_ANSWER = re.compile(r"[0-9]+")
_REQUESTS_ANSWER = re.compile(r"([0-9]+) requests?")
_USERS_WIDTHS = (6, 14)


@dataclass(frozen=True)
class Level:
    """One level of the scale: what it is called, its context, and what it is good for."""

    name: str
    tokens: int
    example: str

    @property
    def shown_tokens(self) -> str:
        return f"{self.tokens // 1024}k"

    @property
    def words(self) -> str:
        """Roughly how many words that many tokens are: three quarters of a token each."""
        return f"{(self.tokens // 1024) * 750:,} words"


LEVELS: tuple[Level, ...] = (
    Level("XS", 4096, "short questions and answers"),
    Level("S", 8192, "a long conversation"),
    Level("M", 16384, "a conversation plus a few documents"),
    Level("L", 32768, "a report or a long contract"),
    Level("XL", 65536, "several documents at once"),
    Level("XXL", 131072, "a whole book"),
)
# The level the pointer starts on for a folder that has kept no context of its own (decided 2026-09-24).
DEFAULT_LEVEL = "L"
DEFAULT_CONTEXT = next(level.tokens for level in LEVELS if level.name == DEFAULT_LEVEL)


@dataclass(frozen=True)
class Checked:
    """The machine the scale computes its last column against, or the reason there is none."""

    name: str | None = None
    profile: HardwareProfile | None = None
    machine_config: MachineConfig | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RequestsAnswer:
    """The answer to the head count: the people, the requests the ranking assumes, and why."""

    users: int | None
    requests: int
    origin: RequestsOrigin


def requests_words(requests: int) -> str:
    """`1 request`, `3 requests`."""
    return f"{requests} request" if requests == 1 else f"{requests} requests"


def answer_words(answer: RequestsAnswer) -> str:
    """The answer line: `25 (3 requests)` for a head count, `12 requests` for requests named outright."""
    if answer.origin == "from_users":
        return f"{answer.users} ({requests_words(answer.requests)})"
    return requests_words(answer.requests)


def users_choices(default_users: int) -> list[Choice]:
    """The list of the head count: five head counts with their requests, the kept one, a number.

    The pointer starts on the head count this folder kept, else on 1; a kept one that is no line
    of the list gets a line of its own, like a kept context that is no level of the scale.
    """
    rows = [[str(users), requests_words(requests_from_users(users))] for users in USERS_LEVELS]
    choices = [
        Choice(str(users), label, checked=users == default_users)
        for users, label in zip(USERS_LEVELS, columns(rows, _USERS_WIDTHS))
    ]
    if default_users not in USERS_LEVELS:
        kept = [[str(default_users), requests_words(requests_from_users(default_users)), "kept in this folder"]]
        choices.append(Choice(str(default_users), columns(kept, _USERS_WIDTHS)[0], checked=True))
    choices.append(Choice(NUMBER_VALUE, NUMBER_LABEL))
    return choices


def requests_answer(answered: str, kept_users: int | None) -> RequestsAnswer:
    """A head count or `N requests` as the answer; anything else ends the run (exit `2`).

    A head count is `from_users`, one person included: the user named it. Requests named outright
    are `entered` and keep the head count this folder holds, which only the display uses. The
    bounds are the configuration's own, checked by its model.
    """
    from .guided import GuidedError  # imported here: that module calls this step in turn

    named = _REQUESTS_ANSWER.fullmatch(answered)
    try:
        if named is not None:
            answer = RequestsAnswer(kept_users, int(named.group(1)), "entered")
        elif _USERS_ANSWER.fullmatch(answered) is not None and 1 <= int(answered) <= MAX_USERS:
            users = int(answered)
            answer = RequestsAnswer(users, requests_from_users(users), "from_users")
        else:
            raise ValueError("neither a whole number of people nor N requests")
        GuidedConfig(users=answer.users, requests=answer.requests, requests_origin=answer.origin)
    except (ValueError, ValidationError) as exc:
        raise GuidedError(f"{answered!r} is no head count this ranking can be computed for: answer {USERS_FORMS}") from exc
    return answer


def context_scenario(context: int, requests: int = 1) -> Scenario:
    """The scenario of one chosen context and the requests of the head count; all `entered`.

    Not `measurements.default_scenario`: that one is 8192 as `default`, the context the three
    automation commands assume when nobody chose one. A user who picks `S` in the scale has
    chosen it, so it is `entered` like every other level, and `render_cmd.scenario_from_config`
    reads a kept context back the same way (CONTRACTS.md, "Guided mode", step 3). The requests
    are the ones the head count stands for; the load test gets a copy of one request.
    """
    return Scenario(
        context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=requests
    )


def scale_words(context: int) -> str:
    """One answer of the scale in the words of its own line, without the column of what fits.

    What fits is about the list, and the list is gone once the question is answered; the level,
    the tokens, the words and the example are the answer itself. A context of its own is the
    number it is.
    """
    level = next((level for level in LEVELS if level.tokens == context), None)
    if level is None:
        return f"{context:,} tokens"
    return f"{level.name}  {level.shown_tokens}  {level.words}  {level.example}"


def context_step(run: "GuidedRun", config_file: Path, config: Configuration) -> tuple[Scenario, Configuration]:
    """Ask the head count, then the scale; keep both in `[guided]` -- one scenario for the ranking.

    The head count is written as `[guided].users`, `requests` and `requests_origin` together
    (`guided_write.with_requests`), right after it is answered, and the scale is then asked with
    its requests.

    The answer is written whenever it differs from what the file holds **when the answer comes
    in**, read again immediately before the write: the dialog takes as long as the user takes,
    and another process may have written the file meanwhile. A run that skipped its write
    because its own copy already said so would leave that other context in the file, and the
    next `modelroom render` would compute a ranking this run never showed.

    The screen of this step is the answered question and nothing else: which machine the column
    was computed against stands in the instruction line of the list, and that the context was kept
    is what `[guided].context` says (test round, 2026-09-24).
    """
    run.screen.blank()
    run.screen.head(STEP)
    checked = machine_checked(run.pointer_path, config, config_file.parent)
    if checked.reason is not None:
        run.screen.note(checked.reason)
    from .guided import _write_config  # imported here: the step is called by that module in turn

    answer = _answered_requests(run, config)
    run.screen.answer(USERS_QUESTION, answer_words(answer))
    # Read, compare, write -- and once more on the new state when another run wrote in between.
    requests = with_requests(answer.users, answer.requests, answer.origin)
    config = _write_config(run, config_file, requests)
    # The machine again, from the file as it is now: the ranking computes with these reserves.
    checked = _checked_again(run, config, config_file, checked)
    context = _answered_context(run, config, checked, answer.requests)
    run.screen.answer(QUESTION, scale_words(context))
    # The whole answer once more, so a head count another run wrote meanwhile cannot split the run.
    config = _write_config(run, config_file, combined(requests, with_context(context)))
    return context_scenario(context, config.guided.requests), config


def _checked_again(run: "GuidedRun", config: Configuration, config_file: Path, checked: Checked) -> Checked:
    """The machine of the scale from the configuration just written; its reason was said already."""
    again = machine_checked(run.pointer_path, config, config_file.parent)
    if again.reason is not None and again.reason != checked.reason:
        run.screen.note(again.reason)
    return again


def machine_checked(pointer_path: Path, config: Configuration, results_dir: Path) -> Checked:
    """The machine the scale is about: the profile this folder binds **this** machine to.

    The binding of the pointer file, not `[machines.<name>].profile`: a configuration is shared,
    and a stale or foreign entry there would show another device's memory as this one's. The
    same source `guided_loadtest._bound_profile_id` measures into; unlike that step the scale
    does not need this machine's fingerprint to agree, because it only reads a profile that is
    in the folder -- but it does need a profile the fit may compute for (`fit_block_reason`).
    """
    bound = read_pointer(pointer_path).binding_for(results_dir)
    profile = scan_profiles(config.paths.hardware_dir).profiles.get(bound) if bound is not None else None
    if profile is None:
        return Checked(reason=NO_MACHINE_LINE)
    blocked = fit_block_reason(profile)
    if blocked is not None:
        return Checked(reason=f"no fit on this machine: {blocked}")
    entry = _machine_entry(config, profile.profile_id)
    if entry is None:
        return Checked(reason=NO_ENTRY_LINE.format(profile_id=profile.profile_id))
    name, machine_config = entry
    return Checked(name=name, profile=profile, machine_config=machine_config)


def scale_choices(levels: tuple[Level, ...], default_context: int, fits: dict[int, str], cap: tuple[int, str] | None) -> list[Choice]:
    """The list the scale shows: one line per level, the kept context, and a number of your own.

    The pointer starts on the level whose context this folder kept, on `custom` when the kept
    context is no level, and on `L` when the folder kept none. A level above the window of the
    base models is grayed out with that reason and cannot be answered, in a file either.
    """
    rows = [[level.name, level.shown_tokens, level.words, level.example, fits.get(level.tokens, "")] for level in levels]
    labels = columns(rows, _COLUMN_WIDTHS)
    known = {level.tokens for level in levels}
    choices = [
        Choice(level.name, label, checked=level.tokens == default_context, disabled=_beyond(level.tokens, cap))
        for level, label in zip(levels, labels)
    ]
    if default_context not in known:
        custom = columns(
            [[CUSTOM_LABEL, str(default_context), "tokens", "kept in this folder", fits.get(default_context, "")]],
            _COLUMN_WIDTHS,
        )[0]
        choices.append(Choice(str(default_context), custom, checked=True))
    choices.append(Choice(NUMBER_VALUE, NUMBER_LABEL))
    return choices


def fit_texts(
    packages: list[Package], base_models: dict[str, BaseModelSpec], checked: Checked, contexts: list[int], requests: int = 1
) -> dict[int, str]:
    """The last column, one text per context: `all N packages fit`, `k of N`, `none of N`.

    Computed for the requests of the head count, so the scale and the ranking of the run agree.
    """
    if checked.profile is None or checked.machine_config is None:
        return {}
    total = len(packages)
    if not total:
        return {context: NO_PACKAGES for context in contexts}
    texts = {}
    for context in contexts:
        fitting = count_fitting(checked.profile, checked.machine_config, packages, base_models, context, requests)
        texts[context] = _fit_text(fitting, total)
    return texts


def context_cap(base_models: list[BaseModelSpec]) -> tuple[int, str] | None:
    """The smallest context window the base models declare, or `None` when one of them is silent.

    A window nobody knows is no reason to gray out a level: the packages themselves may hold a
    smaller one, and `Architecture.max_context` is read from the publisher repository, which not
    every base model publishes.
    """
    if not base_models:
        return None
    windows = [(spec.architecture.max_context, spec.hf_repo) for spec in base_models]
    if any(window is None for window, _repo in windows):
        return None
    return min(windows)  # type: ignore[arg-type,return-value]  -- every window is an int here


def snapshot_packages(config: Configuration) -> tuple[list[Package], list[BaseModelSpec]]:
    """The packages of this folder's snapshot a fit may be computed for, and its base models.

    Through the render's own filter (`render._eligible_packages`) and not a second one of this
    module's, so that the count in the scale is about exactly the packages the ranking of this
    same run will judge. A snapshot that does not read is not this step's business: the scale
    then shows no count, and the render that follows ends the run with the exit code that file's
    own reader gives it.
    """
    from .render import _eligible_packages

    try:
        snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError):
        return [], []
    if snapshot is None:
        return [], []
    return _eligible_packages(snapshot), snapshot.base_models


@dataclass(frozen=True)
class SnapshotFacts:
    """What the snapshot of this folder holds, in the four numbers the run says out loud.

    The balance of step 2 and the card of step 5 say the same thing about the same fetch, so they
    read it once, here: the base models by name, how many packages came in for them, from which
    packager accounts, and from how many repositories.
    """

    model_names: tuple[str, ...]
    packages: int
    accounts: tuple[str, ...]
    repositories: int


def snapshot_facts(config: Configuration) -> SnapshotFacts:
    """The four numbers of this folder's snapshot, read through the render's own filter."""
    packages, base_models = snapshot_packages(config)
    accounts: list[str] = []
    repositories: list[str] = []
    for package in packages:
        account = "Ollama" if package.source == "ollama" else (package.repo or "").partition("/")[0]
        repository = package.repo if package.source == "huggingface" else f"ollama:{package.base_model_hf_repo}"
        if account and account not in accounts:
            accounts.append(account)
        if repository and repository not in repositories:
            repositories.append(repository)
    return SnapshotFacts(
        model_names=tuple(spec.hf_repo.partition("/")[2] or spec.hf_repo for spec in base_models),
        packages=len(packages),
        accounts=tuple(accounts),
        repositories=len(repositories),
    )


def _answered_requests(run: "GuidedRun", config: Configuration) -> RequestsAnswer:
    """The head count the user gave, or the requests named outright; the list starts on the kept one.

    Only a kept head count moves the pointer: requests named outright say nothing about how many
    people there are, so a folder that kept `12 requests` and no head count starts on 1.
    """
    kept = config.guided.users
    answered = run.asker.select_or_text(
        USERS_KEY,
        USERS_QUESTION,
        users_choices(kept if kept is not None else 1),
        NUMBER_VALUE,
        USERS_NUMBER_QUESTION,
        USERS_EXPLANATION,
    ).strip()
    return requests_answer(answered, kept)


def _answered_context(run: "GuidedRun", config: Configuration, checked: Checked, requests: int = 1) -> int:
    """The context the user picked, as a number of tokens; a level is translated here."""
    packages, base_models = snapshot_packages(config)
    default_context = config.guided.context if config.guided.context is not None else DEFAULT_CONTEXT
    contexts = [level.tokens for level in LEVELS] + [default_context]
    fits = fit_texts(packages, {spec.hf_repo: spec for spec in base_models}, checked, contexts, requests)
    choices = scale_choices(LEVELS, default_context, fits, context_cap(base_models))
    instruction = EXPLANATION.format(machine=checked.name) if fits else None
    answered = run.asker.select_or_text(
        "context", QUESTION, choices, NUMBER_VALUE, NUMBER_QUESTION, instruction
    ).strip()
    return tokens_of(answered)


def tokens_of(answered: str) -> int:
    """A level name or a number of tokens as a number; anything else ends the run (exit `2`)."""
    from .guided import GuidedError  # imported here: that module calls this step in turn

    level = next((level for level in LEVELS if level.name == answered), None)
    if level is not None:
        return level.tokens
    try:
        context = int(answered)
    except ValueError as exc:
        raise GuidedError(f"{answered!r} is neither a level of the scale nor a whole number of tokens") from exc
    try:
        context_scenario(context)
    except ValidationError as exc:
        raise GuidedError(f"{context} is not a context this ranking can be computed for: {exc}") from exc
    return context


def _machine_entry(config: Configuration, profile_id: str) -> tuple[str, MachineConfig] | None:
    """The `[machines.<name>]` entry of that profile, for the name and for the reserves.

    `None` when no entry names this profile -- and then there is no column, rather than one
    computed with another machine's reserves under another machine's name. It happens: a profile
    whose `display_name` collides keeps its entry (`guided._configure_found_profiles` says so and
    goes on), and the ranking of such a folder computes per configured machine, not per profile.
    """
    for name, machine in sorted(config.machines.items()):
        if machine.profile == profile_id:
            return name, machine
    return None


def _beyond(tokens: int, cap: tuple[int, str] | None) -> str | None:
    if cap is None or tokens <= cap[0]:
        return None
    return f"beyond the window of {cap[1]}"


def _fit_text(fitting: int, total: int) -> str:
    word = "package" if total == 1 else "packages"
    if fitting == total:
        return f"all {total} {word} fit"
    if fitting == 0:
        return f"none of {total} {word} fits"
    return f"{fitting} of {total} {word} fit"


__all__ = [
    "DEFAULT_CONTEXT",
    "DEFAULT_LEVEL",
    "EXPLANATION",
    "LEVELS",
    "NO_MACHINE_LINE",
    "NUMBER_LABEL",
    "NUMBER_VALUE",
    "QUESTION",
    "USERS_EXPLANATION",
    "USERS_KEY",
    "USERS_QUESTION",
    "Checked",
    "Level",
    "RequestsAnswer",
    "SnapshotFacts",
    "answer_words",
    "context_cap",
    "context_scenario",
    "context_step",
    "fit_texts",
    "machine_checked",
    "requests_answer",
    "requests_words",
    "scale_choices",
    "scale_words",
    "snapshot_facts",
    "snapshot_packages",
    "tokens_of",
    "users_choices",
]
