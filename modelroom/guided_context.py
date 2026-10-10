"""Step 3 of the guided mode: how many people work with the model at the same time, then how much text it should handle.

First the people at the same time: the question assumes one request at a time per person, so the
number named is the parallel requests the ranking computes with. Whoever knows only a daily head
count types `N people`, which a rule of thumb turns into requests (`config.requests_from_users`,
said under the list). The older answer-file key `users` still reads as it did. Then the size scale: six levels from XS to XXL,
each with its number of tokens, roughly how many words that is, an example of what it is good
for, and how many of the packages of this results folder still fit
the machine being checked. The last column is the whole point of the scale: it is the fit of
fit contract v1, computed by `fit.count_fitting` over the packages the fetch of this same run
recorded, against the profile the pointer file binds this machine to for this folder -- through
`runtimes.fit_for` (AP9-SV2-F, `fit_texts`'s own `fit_function`), the one function a render's own
row reads, so the scale never disagrees with the finished ranking about which runtime judges a
package. A level the base models cannot even hold is grayed out.

Everything the scale knows comes from somewhere else: the formula from `modelroom/fit.py` and
`modelroom/fit_vllm.py` (through `modelroom/runtimes.py`), the
"it fits" predicate from `modelroom/ranking.py`, the packages from the snapshot, the reserves
from `[machines.<name>]`. It is read when the question is asked, like every other list of this
dialog; the write that follows reads the configuration again on its own.

The people at the same time come first because the last column of the scale counts with their
requests: a column computed for one request would show a number the ranking of the same run then
contradicts.

This module asks both questions and writes the answers; the order of the steps stays in
`modelroom/guided.py` (CONTRACTS.md, "Guided mode", step 3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

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
from .screen_names import profile_name
from .state import UnreadableStateFileError, read_snapshot

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun
    from .selection import RunSelection

STEP = 3

QUESTION = "How much text should a model handle at once?"
# What the last column of the list means, in the instruction line of the question rather than as a
# line of the run (decided 2026-09-24): it explains the list, so it belongs to the list and goes
# away with it. `<machine>` is the machine the column was computed against, by its profile's name (FR-N3).
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

# The question about people at the same time (decided 2026-10-10): one request at a time per
# person, so the number answered is the requests; `N people` names a daily head count instead and
# the rule of thumb between the two stands under the list. `at_once` is a key of the answer file
# only, never a field of `[guided]`.
AT_ONCE_KEY = "at_once"
AT_ONCE_QUESTION = "How many people work with it at the same time?"
AT_ONCE_EXPLANATION = (
    f"one request at a time per person; a daily head count is about {round(1 / ACTIVE_SHARE)} times as many "
    "(rule of thumb, not measured)"
)
AT_ONCE_NUMBER_QUESTION = "How many at the same time? (or: N people)"
AT_ONCE_LEVELS = (1, 5, 10, 50, 100)
AT_ONCE_FORMS = (
    f"a whole number of people at the same time (1 to {MAX_REQUESTS}) or `N people` for a daily head count "
    f"(1 to {MAX_USERS})"
)
AT_ONCE_INVALID = "{answered!r} is not a valid answer for step 3: answer {forms}"
AT_ONCE_BOTH_KEYS = "the answer file holds both users and at_once; answer only one of them"
# What each level is, the second cell of its line in the list.
_AT_ONCE_LABELS = {
    1: "one person",
    5: "a small team",
    10: "a department; consider a serving stack such as vLLM",
    50: "a serving stack such as vLLM",
    100: "a serving stack such as vLLM",
}
KEPT_VALUE = "kept"
KEPT_LABEL = "kept in this folder"
_AT_ONCE_WIDTHS = (5,)
# The singular is for 1 only: `5 requests` and `5 people`, but `1 request` and `1 person`.
_AT_ONCE_REQUESTS = re.compile(r"(?:([0-9]+)(?: requests)?|(1) request)")
_AT_ONCE_PEOPLE = re.compile(r"(?:([0-9]+) people|(1) person)")

# The earlier head count (decided 2026-09-26): kept for the answer-file key `users`, which still
# reads as it did (a number is a daily head count, `N requests` names requests outright).
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
    shown: str | None = None  # the name the screen says (`screen_names.profile_name`); `name` stays the key (FR-N3)


@dataclass(frozen=True)
class RequestsAnswer:
    """The answer to the question about people at the same time: the daily head count when one was
    named, the requests the ranking computes with (the number answered, or derived from the head
    count), and which of the two it was."""

    users: int | None
    requests: int
    origin: RequestsOrigin


def requests_words(requests: int) -> str:
    """`1 request`, `3 requests`."""
    return f"{requests} request" if requests == 1 else f"{requests} requests"


def people_words(people: int) -> str:
    """`1 person`, `25 people`."""
    return "1 person" if people == 1 else f"{people} people"


def answer_words(answer: RequestsAnswer) -> str:
    """The answer line: `5 at once`, or `25 people (3 at once)` for a daily head count."""
    if answer.origin == "from_users":
        return f"{people_words(answer.users)} ({answer.requests} at once)"  # type: ignore[arg-type]  -- from_users has users
    return f"{answer.requests} at once"


def kept_offered(guided: GuidedConfig) -> bool:
    """Whether the list gets a line for the answer this folder kept: a number that is no level of
    the list, or any answer derived from a daily head count. A folder that kept nothing has none."""
    if guided.requests_origin == "default":
        return False
    return guided.requests_origin == "from_users" or guided.requests not in AT_ONCE_LEVELS


def at_once_choices(guided: GuidedConfig) -> list[Choice]:
    """The list offers five levels, a kept line for a nonstandard number or any daily head count, and a number of your own.

    The pointer starts on the requests this folder kept (origin not `default`), else on 1. The line
    of a kept answer is the pointer's start when it exists, not the level with the same number.
    """
    start = guided.requests if guided.requests_origin != "default" else 1
    offered = kept_offered(guided)
    rows = [[str(level), _AT_ONCE_LABELS[level]] for level in AT_ONCE_LEVELS]
    choices = [
        Choice(str(level), label, checked=level == start and not offered)
        for level, label in zip(AT_ONCE_LEVELS, columns(rows, _AT_ONCE_WIDTHS))
    ]
    if offered:
        text = KEPT_LABEL if guided.requests_origin != "from_users" else f"{KEPT_LABEL} ({people_words(guided.users)})"  # type: ignore[arg-type]
        choices.append(Choice(KEPT_VALUE, columns([[str(guided.requests), text]], _AT_ONCE_WIDTHS)[0], checked=True))
    choices.append(Choice(NUMBER_VALUE, NUMBER_LABEL))
    return choices


def at_once_answer(answered: str, guided: GuidedConfig) -> RequestsAnswer:
    """People at the same time, `N people` or `kept` as the answer; anything else raises `GuidedError`.

    A whole number (also `N requests`) is the requests: `entered`, and the head count stays what
    the folder holds, which only the display uses. `N people` is a daily head count: `from_users`.
    `kept` is the stored triple, and only when the list offered its line. Digits 0-9 only, the
    whole answer matched; the bounds are the configuration's own, checked by its model.
    """
    from .guided import GuidedError  # imported here: that module calls this step in turn

    try:
        number = _AT_ONCE_REQUESTS.fullmatch(answered)
        people = _AT_ONCE_PEOPLE.fullmatch(answered)
        if number is not None:
            answer = RequestsAnswer(guided.users, int(number.group(1) or number.group(2)), "entered")
        elif people is not None:
            head = int(people.group(1) or people.group(2))
            answer = RequestsAnswer(head, requests_from_users(head), "from_users")
        elif answered == KEPT_VALUE and kept_offered(guided):
            return RequestsAnswer(guided.users, guided.requests, guided.requests_origin)
        else:
            raise ValueError("neither people at the same time, N people nor the kept line")
        GuidedConfig(users=answer.users, requests=answer.requests, requests_origin=answer.origin)
    except (ValueError, ValidationError) as exc:
        raise GuidedError(AT_ONCE_INVALID.format(answered=answered, forms=AT_ONCE_FORMS)) from exc
    return answer


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
    """The answer-file key `users`: a head count or `N requests`; anything else raises `GuidedError`.

    Kept for answer-file compatibility: the question of the dialog is `at_once_answer`'s.
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
    """The scenario of one chosen context and the requests answered at the same time; all `entered`.

    Not `measurements.default_scenario`: that one is 8192 as `default`, the context the three
    automation commands assume when nobody chose one. A user who picks `S` in the scale has
    chosen it, so it is `entered` like every other level, and `render_cmd.scenario_from_config`
    reads a kept context back the same way (CONTRACTS.md, "Guided mode", step 3). The requests
    are the ones the answer stands for; the load test gets a copy of one request.
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
    """Ask how many people work at the same time, then the scale; keep both in `[guided]` -- one scenario for the ranking.

    The answer is written as `[guided].users`, `requests` and `requests_origin` together
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
    run.screen.answer(AT_ONCE_QUESTION, answer_words(answer))
    # Read, compare, write -- and once more on the new state when another run wrote in between.
    requests = with_requests(answer.users, answer.requests, answer.origin)
    config = _write_config(run, config_file, requests)
    # The machine again, from the file as it is now: the ranking computes with these reserves.
    checked = _checked_again(run, config, config_file, checked)
    context = _answered_context(run, config, checked, answer.requests)
    run.screen.answer(QUESTION, scale_words(context))
    # The whole answer once more, so an answer another run wrote meanwhile cannot split the run.
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
    return Checked(name=name, profile=profile, machine_config=machine_config, shown=profile_name(profile))


def chosen_runtimes(run: "GuidedRun", config: Configuration) -> list[tuple[HardwareProfile | None, object]]:
    """`runtimes.machine_has_vllm`'s two inputs for every machine this run chose (F3): its stored
    profile (`[machines.<name>].profile`, the one the render ranks it with), and what this run's
    own step 1 detected live -- for the machine the run is on and the active computer over SSH
    (its `hello`, AP9-SV2-R); any other machine has no live detection here, so its stored profile
    alone speaks for it."""
    from .guided_remote_vllm import live_map
    from .remote_store import remote_path_for, this_machine

    here = this_machine(remote_path_for(run.pointer_path))
    scan = scan_profiles(config.paths.hardware_dir)
    over_ssh = live_map(run)  # the active computer over SSH: what its `hello` said of its vLLM (AP9-SV2-R, R3)
    found = []
    for name in run.selection.machines:
        machine = config.machines.get(name)
        profile = scan.profiles.get(machine.profile) if machine is not None and machine.profile else None
        found.append((profile, run.servers.found.get("vllm") if name == here else over_ssh.get(name)))
    return found


def vllm_among_chosen(run: "GuidedRun", config: Configuration) -> bool:
    """Whether at least one machine this run chose has vLLM -- the one gate of the tensor search
    (`guided_search.search_step`) and of `build_tensor` (`guided._fetch_step`), P1."""
    from .runtimes import any_has_vllm

    return any_has_vllm(chosen_runtimes(run, config))


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

    Computed for the requests answered at the same time, so the scale and the ranking of the run agree --
    through `runtimes.fit_for` (AP9-SV2-F), the one function a render's own row reads, not
    `fit.compute_fit_v2` on its own: a Hugging Face tensor package counts here exactly as the
    finished ranking would judge it. `packages` already is the set `runtimes.visible_packages`
    shows for this machine (`snapshot_packages`'s own filter through `render._eligible_packages`),
    so the scale and the ranking read the same packages too, not only the same formula.
    """
    from .runtimes import fit_for  # deferred: runtimes.py imports guided_vllm.py, which imports this module (a cycle)

    if checked.profile is None or checked.machine_config is None:
        return {}
    total = len(packages)
    if not total:
        return {context: NO_PACKAGES for context in contexts}
    texts = {}
    for context in contexts:
        fitting = count_fitting(
            checked.profile, checked.machine_config, packages, base_models, context, requests, fit_function=fit_for
        )
        texts[context] = _fit_text(fitting, total)
    return texts


CAP_WORDS = "every model of this run"


def context_cap(base_models: list[BaseModelSpec]) -> tuple[int, str] | None:
    """The largest context window the base models of this run declare, or `None` when one is silent.

    A level is grayed out only beyond the window of **every** model of this run (AP9-UX H): one
    model with a 32k window no longer locks XL and XXL for one with 256k -- a package beyond its own
    model's window is `not covered` in the fit itself (`fit.compute_fit_v2` for a GGUF or
    Ollama-library package, `fit_vllm.vllm_fit_as_fit` for a Hugging Face tensor one, AP9-SV2-F --
    both apply the same guard). A window nobody knows is no reason to gray out a level:
    `Architecture.max_context` is read from the publisher repository, which not every base model
    publishes.
    """
    if not base_models:
        return None
    windows = [spec.architecture.max_context for spec in base_models]
    if any(window is None for window in windows):
        return None
    return max(windows), CAP_WORDS  # type: ignore[type-var]  -- every window is an int here


def snapshot_packages(
    config: Configuration,
    selection: "RunSelection | None" = None,
    machines: Sequence[tuple[HardwareProfile | None, object]] = (),
) -> tuple[list[Package], list[BaseModelSpec]]:
    """The packages of this folder's snapshot a fit may be computed for, and its base models.

    Through the render's own filter (`render._eligible_packages`) and not a second one of this
    module's, so that the count in the scale is about exactly the packages the ranking of this
    same run will judge -- and through the same selection (`selection.narrow_snapshot`, AP9-UX B).
    `machines` are the machines the count is read for (`runtimes.visible_packages`): the one
    machine of the scale, or every chosen machine for the balance and the card. A snapshot that
    does not read is not this step's business: the scale then shows no count, and the render that
    follows ends the run with the exit code that file's own reader gives it.
    """
    from .render import _eligible_packages
    from .selection import narrow_snapshot

    try:
        snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError):
        return [], []
    if snapshot is None:
        return [], []
    if selection is not None:
        snapshot = narrow_snapshot(snapshot, selection)
    return _eligible_packages(snapshot, machines), snapshot.base_models


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


def snapshot_facts(
    config: Configuration,
    selection: "RunSelection | None" = None,
    machines: Sequence[tuple[HardwareProfile | None, object]] = (),
) -> SnapshotFacts:
    """The four numbers of this folder's snapshot, read through the render's own filter and this
    run's selection, so the balance of step 2 and the card count what this run chose; `machines`
    are the machines they are read for (`chosen_runtimes`, see `snapshot_packages`)."""
    packages, base_models = snapshot_packages(config, selection, machines)
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
    """The people at the same time the user named, as the requests the ranking computes with.

    An answer file may still answer the older key `users` (a number is a daily head count, read
    by `requests_answer` as before); a file holding both keys ends the run (exit `2`). Anything
    else is the key `at_once`, read by `at_once_answer`; the list starts on the kept answer.
    """
    from .guided import GuidedError, ask_checked  # imported here: that module calls this step in turn

    asker = run.asker
    if asker.holds(USERS_KEY) and asker.holds(AT_ONCE_KEY):
        raise GuidedError(AT_ONCE_BOTH_KEYS)
    if asker.holds(USERS_KEY):
        kept = config.guided.users
        answered = asker.select_or_text(
            USERS_KEY,
            USERS_QUESTION,
            users_choices(kept if kept is not None else 1),
            NUMBER_VALUE,
            USERS_NUMBER_QUESTION,
            USERS_EXPLANATION,
        ).strip()
        return requests_answer(answered, kept)
    choices = at_once_choices(config.guided)

    def ask(again: bool) -> str:
        return asker.select_or_text(
            AT_ONCE_KEY,
            AT_ONCE_QUESTION,
            again_choices(choices) if again else choices,
            NUMBER_VALUE,
            AT_ONCE_NUMBER_QUESTION,
            AT_ONCE_EXPLANATION,
            required=True,
        ).strip()

    return ask_checked(run, ask, lambda answered: at_once_answer(answered, config.guided))


def again_choices(choices: Sequence[Choice]) -> list[Choice]:
    """The same list for a question asked again: the pointer starts on the number of your own.

    The kept row and its label stay as they were; only the start of the pointer moves, off the row
    the refused answer came from and onto the entry that leads to the number field.
    """
    return [replace(choice, checked=choice.value == NUMBER_VALUE) for choice in choices]


def _answered_context(run: "GuidedRun", config: Configuration, checked: Checked, requests: int = 1) -> int:
    """The context the user picked, as a number of tokens; a level is translated here. The scale
    counts and caps with the models of this run alone (AP9-UX B and H)."""
    packages, base_models = snapshot_packages(config, run.selection, [(checked.profile, run.servers.found.get("vllm"))])
    default_context = config.guided.context if config.guided.context is not None else DEFAULT_CONTEXT
    contexts = [level.tokens for level in LEVELS] + [default_context]
    fits = fit_texts(packages, {spec.hf_repo: spec for spec in base_models}, checked, contexts, requests)
    choices = scale_choices(LEVELS, default_context, fits, context_cap(base_models))
    instruction = EXPLANATION.format(machine=checked.shown) if fits else None
    from .guided import ask_checked  # imported here: that module calls this step in turn

    def ask(again: bool) -> str:
        return run.asker.select_or_text(
            "context",
            QUESTION,
            again_choices(choices) if again else choices,
            NUMBER_VALUE,
            NUMBER_QUESTION,
            instruction,
            required=True,
        ).strip()

    return ask_checked(run, ask, tokens_of)


def tokens_of(answered: str) -> int:
    """A level name or a number of tokens as a number; anything else raises `GuidedError`."""
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
    "AT_ONCE_EXPLANATION",
    "AT_ONCE_KEY",
    "AT_ONCE_QUESTION",
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
    "at_once_answer",
    "at_once_choices",
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
