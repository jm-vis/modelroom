"""Step 2 of the guided mode: one line per model, with the fit its size allows.

Until 2026-09-24 the selection list showed one line per repository -- 120 lines of over 200
characters in a hand test, with repository ids, `unknown` sizes and 40 grayed-out rows whose
appended reason broke the columns. Nobody could tell from it what a choice would cost. So the
list is about **models** now: one line per resolved base model, only the ones that can be
picked, with the fit that follows from the size of the model on this machine.

Since 2026-09-25 the list carries a column head (`list_header`) and a `Release` column, and says
`–` where it has nothing: the test round read `latest` nowhere, could not tell what a cell meant,
and took `none known` for a statement about the model.

Three things live here, all pure: `model_choices` groups the search's hits by base model and
computes each model's fit (`fit.fit_from_parameters`, basis `size`), `list_choices` renders that
as the checkbox the dialog shows, and `chosen_hits` turns the answer back into the repositories
the fetch works through -- two per model, because every repository costs the fetch about ten
requests of the shared budget. The parameter count comes from the search when a repository
states one and from the model's own name otherwise (`parameters_from_name`).

See CONTRACTS.md, "Guided mode", step 2.
"""

from __future__ import annotations

import math
import re
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import Age
from .contracts import Fit, validate_hf_repo
from .dialog import Asker, Choice, FileAsker, columns
from .fit import fit_from_parameters, unknown_fit
from .guided_context import DEFAULT_CONTEXT, Checked
from .screen import context_tokens_text
from .guided_contracts import SearchHit
from .search_pages import format_downloads

SELECT_KEY = "select"
UNKNOWN = "unknown"
# The two reasons a model's fit cannot be computed before the fetch. Both are facts about this
# results folder, not about the model.
MACHINE_NOT_MEASURED = "machine not measured"
PARAMETER_COUNT_UNKNOWN = "parameter count unknown"

# Why a repository is no model of this list, in the words a reader can act on. The technical
# status stays in the answer file's own error message, which names the value a file may answer.
UNRESOLVED_REASONS: dict[str, str] = {
    "derivative": "a fine-tune or a merge, not a quantization of one base model",
    "relation_unknown": "the repository does not say it packages a base model",
    "metadata_conflict": "the repository's own data names more than one base model",
    "base_model_tag": "the repository names no base model of this search",
    "publisher_unknown": "the base model's owner is not a publisher in this catalog",
}
OTHER_OWNER_REASON = "not a publisher or a listed packager"
_PICKABLE_OWNERS = ("publisher", "listed packager")

# The word for where a model stands in its family, as the column head says it. One constant, so
# the head, the documentation and a test can never name it differently.
RELEASE_COLUMN = "Release"
# The columns of one line: name, fit, size, release, packagers, downloads -- and the Ollama name,
# which is the last one and takes what is left. Since 2026-09-25 the packagers have 12 characters
# (`unsloth +3`) and the Ollama name the 14 that frees: 27 of the 32 names of the shipped catalog
# stand whole, where 10 characters cut 24 of them.
COLUMN_NAMES = ("Model", "Fit", "Size", RELEASE_COLUMN, "Packagers", "Downl.", "Ollama")
_COLUMN_WIDTHS = (24, 14, 6, 7, 12, 6)
# 100 characters is what a terminal window holds without wrapping, and it is a promise, not a hope:
# every cell is cut to its column and the Ollama name to what is left of the line. Since 2026-09-25
# the promise covers the **whole** line: `questionary` draws ` ❯ ○ ` in front of a row -- the one
# space every line of this screen carries, the pointer and the marker -- and a label of 100 made a
# line of 105. A registry name longer than the Ollama column is cut; the install line of step 5
# carries the whole name.
LINE_LIMIT = 100
POINTER_WIDTH = 5
LABEL_LIMIT = LINE_LIMIT - POINTER_WIDTH
# A heading is drawn without the marker of a row (three spaces, not five), so the head carries the
# two characters that are missing itself and its columns stand over the cells.
_HEAD_INDENT = "  "
# What the padded columns and the two spaces between them take up, so the last cell knows its room.
_FIXED_WIDTH = sum(_COLUMN_WIDTHS) + 2 * len(_COLUMN_WIDTHS)
_OLLAMA_WIDTH = LABEL_LIMIT - _FIXED_WIDTH
# What a cell says where there is nothing to say: `none known` and `unknown` in a column of their
# own read as a fact about the model rather than as an empty cell (test round, 2026-09-25).
DASH = "–"
_RELEASE_WORDS = {"latest": "latest", "legacy": "legacy", UNKNOWN: DASH}
# Behind a release computed from the version numbers of the family, never behind a stated one:
# `latest*` and `legacy*` still fit the seven characters of the column (decided 2026-09-25).
COMPUTED_MARK = "*"
# `latest` first, then a release nobody knows, then `legacy`: a reader is choosing what to fetch,
# and a model whose family has moved on belongs under the ones that have not (decided 2026-09-25).
_RELEASE_ORDER = ("latest", UNKNOWN, "legacy")
_FIT_WORDS = {"perfect": "good", "good": "good", "marginal": "marginal", "too_tight": "too tight", "unknown": UNKNOWN}
_CLASS_ORDER = ("perfect", "good", "marginal", "too_tight", "unknown")
# A fit against system memory says so behind its word: fit v1 caps that pool at `good`, and live
# on 2026-09-24 a 122B model stood as `good` above a 9B `marginal` that fits the graphics card,
# with nothing to tell the two pools apart. The list is what a person chooses by, so a fit in
# graphics memory stands above one in system memory, whatever its class (decided 2026-09-24).
_RAM_MODES = ("cpu_gpu", "cpu")
_RAM_SUFFIX = " (RAM)"
_MODE_ORDER = ("gpu", "cpu_gpu", "cpu", None)
_SETTLED = ("too_tight", "unknown")

# A stated age is positive evidence about one model; a computed one compares the version numbers of
# one family and variant and carries the star (AGENTS.md, the language standard). `legacy` means a
# successor is named -- by the publisher, or by that comparison -- and that is what it says.
RELEASE_HINT = (
    "latest: the publisher's current release of its family · legacy: a successor is named · "
    "*: computed from the version numbers of the family, not stated by the publisher"
)
FIT_FROM_SIZE_HINT = "fit from the size at {context} context, exact after the fetch"
FIT_UNKNOWN_HINT = "fit unknown until this machine is measured"

# A parameter count in a model name: `9B`, `0.6B`, `2.4T`, `360M`, the effective size `E4B`, and
# never one that stands behind a letter, a digit or a decimal point -- `A3B`, `A17B`, `8x7B` are the
# active parameters of a mixture or a factor, not a total, and the `.` keeps the tail of such a
# decimal out as well (`Nova-A0.6B` read as 6 in the second-model round of 2026-09-24). `M` and `E`
# since 2026-09-25: the computed release takes its sizes from this one rule as well.
_PARAMETERS_RE = re.compile(r"(?<![0-9A-Za-z.])[Ee]?(\d+(?:\.\d+)?)([BbMmTt])(?![0-9A-Za-z])")
_TRILLION_FACTOR = 1000
_MILLION_DIVISOR = 1000
_GGUF_SUFFIX = "-GGUF"


class ModelChoice(BaseModel):
    """One line of the selection list: one base model, with every repository that packages it.

    `repos` are the resolved hits of this base model in the order the search's groups answered
    (the publisher's account first, then the positive list, then the open lists), so
    `chosen_hits` can pick the publisher's repository and the first listed packager's from it.
    `downloads` is the sum over the repositories that state one, `None` when none does;
    `parameters_b` is what a repository states, else what the name says, else `None` -- and where
    it is `None` the fit is `unknown` rather than a guess.
    """

    model_config = ConfigDict(extra="forbid")

    base_model: str
    name: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    repos: list[SearchHit] = Field(min_length=1)
    packagers: list[str] = Field(min_length=1)
    downloads: int | None = Field(default=None, ge=0)
    ollama: str | None = None
    age: Age = UNKNOWN
    # Where `age` comes from, taken from the same repository as `age` (`SearchHit.release_basis`).
    release_basis: Literal["stated", "computed"] | None = None
    parameters_b: float | None = Field(default=None, gt=0)
    fit: Fit

    @model_validator(mode="after")
    def _check(self) -> "ModelChoice":
        validate_hf_repo(self.base_model)
        if self.base_model != f"{self.publisher}/{self.name}":
            raise ValueError(f"name and publisher are the two halves of {self.base_model!r}")
        if self.release_basis == "computed" and self.age == UNKNOWN:
            raise ValueError("release_basis 'computed' needs a known age")
        return self

    @property
    def fit_word(self) -> str:
        """The fit as the list shows it: `good`, `marginal`, `too tight`, `unknown` -- and
        `good (RAM)` or `marginal (RAM)` where the graphics memory is too small and the fit is
        against system memory. `too tight` is too tight for either pool."""
        word = _FIT_WORDS[self.fit.fit_class]
        if self.fit.mode in _RAM_MODES and self.fit.fit_class not in _SETTLED:
            return f"{word}{_RAM_SUFFIX}"
        return word

    @property
    def size_text(self) -> str:
        """The parameter count as a size: `9B`, `0.6B`, `2.4T`, `unknown`.

        A count of a thousand billions and more is shown in `T`, as the model's own name says it:
        `2400B` is honest and nobody reads it (decided 2026-09-24).
        """
        if self.parameters_b is None:
            return UNKNOWN
        if self.parameters_b >= _TRILLION_FACTOR:
            return _count_text(self.parameters_b / _TRILLION_FACTOR, "T")
        return _count_text(self.parameters_b, "B")

    @property
    def release_text(self) -> str:
        """Where this model stands in its family, as the column shows it: `latest`, `legacy`, `–`,
        and `latest*` or `legacy*` where the version numbers of the family decided it.

        A dash and not `unknown`: the word is a fact about a model in the data, and in a column of
        its own it read as one about this model (test round, 2026-09-25).
        """
        word = _RELEASE_WORDS.get(str(self.age), DASH)
        return f"{word}{COMPUTED_MARK}" if self.release_basis == "computed" else word

    @property
    def legacy(self) -> bool:
        """Whether this row is drawn gray: a successor is named for this model, stated or computed."""
        return self.age == "legacy"

    @property
    def packagers_text(self) -> str:
        """The packager accounts in the width of their column: all of them where they fit, else the
        first one and how many more (`unsloth +3`), else cut (decided 2026-09-25)."""
        width = _COLUMN_WIDTHS[4]
        full = ", ".join(self.packagers)
        if len(full) <= width:
            return full
        counted = f"{self.packagers[0]} +{len(self.packagers) - 1}"
        return counted if len(counted) <= width else _clip(full, width)

    @property
    def ollama_text(self) -> str:
        """The Ollama name, or a dash where neither a hit nor the configuration names one
        (`search.NO_OLLAMA_LABEL` is what the search log says; a column of a list has room for one
        character, 2026-09-25)."""
        return self.ollama if self.ollama is not None else DASH


def _clip(text: str, width: int) -> str:
    """`text` in at most `width` characters, with `…` where something was left out."""
    return text if len(text) <= width else text[: width - 1] + "…"


def _count_text(value: float, unit: str) -> str:
    """`9B`, `0.6B`, `2.4T`: whole where it is whole, else one decimal."""
    rounded = round(value)
    if abs(value - rounded) < 0.05:
        return f"{rounded}{unit}"
    return f"{value:.1f}{unit}"


def list_context(kept: int | None) -> int:
    """The context the fit of this list is computed for: the folder's own, else the scale's default.

    Step 3 asks for the context and this list stands in step 2, so a folder that kept one is shown
    what it kept. A folder that kept none is shown the context of the level the scale will start on
    (`guided_context.DEFAULT_CONTEXT`, 32768) and not fit v1's own assumption of 8192: the list
    said `fit at 8k context` while the very next question started on `L 32k`, and two numbers for
    one run are one too many (test round, 2026-09-24). It is one constant, not two.
    """
    return kept if kept is not None else DEFAULT_CONTEXT


def parameters_from_name(name: str) -> float | None:
    """The total parameter count a model name states, in billions, or `None` when it states none.

    Every `<number>B`, `<number>M` and `<number>T` of the name that stands on its own counts, with
    an `E` in front of it as well (`E4B`, an effective size) -- a count behind a letter, a digit or a
    decimal point is the active parameters of a mixture (`A3B`, `A0.6B`) or a factor (`8x7B`),
    never a total -- and the largest of them is the model's size. `2.4T` is 2400, `360M` is 0.36.
    A name is registry-controlled text, so only a real, finite, positive number is a count:
    `Nova-0B` and a number of 400 digits both say nothing (the fit is then `unknown` with
    `parameter count unknown`, never a guess and never an exception out of the list).
    """
    stem = name[: -len(_GGUF_SUFFIX)] if name.upper().endswith(_GGUF_SUFFIX) else name
    counts = []
    for number, unit in _PARAMETERS_RE.findall(stem):
        value = float(number)  # digits and at most one dot: `float` raises for nothing else
        if unit in "Tt":
            value *= _TRILLION_FACTOR
        elif unit in "Mm":
            value /= _MILLION_DIVISOR
        if math.isfinite(value) and value > 0:
            counts.append(value)
    return max(counts) if counts else None


def hit_reason(hit: SearchHit, filtered: bool, *, typed: str | None = None) -> str | None:
    """Why this repository is no model of the list, in plain words, or `None` when it is one.

    `typed` is the repository id the person typed, when they typed one: the owner filter never
    hides it (decided 2026-09-25). Someone who names a repository has said which account they
    want, so a preference for the publishers and the listed packagers has nothing left to decide.
    """
    if not hit.resolved:
        return UNRESOLVED_REASONS.get(str(hit.unresolved_reason), str(hit.unresolved_reason))
    if filtered and hit.publisher_status not in _PICKABLE_OWNERS and hit.repo != typed:
        return OTHER_OWNER_REASON
    return None


def unusable_counts(hits: Sequence[SearchHit], filtered: bool, *, typed: str | None = None) -> dict[str, int]:
    """Why repositories of this search are no model of the list, each reason with its count.

    The counts of `search.json`'s `unresolved` and of the lines below, from one reading of the
    hits: the file and a line may not say different numbers about the same search.
    """
    counted: dict[str, int] = {}
    for hit in hits:
        reason = hit_reason(hit, filtered, typed=typed)
        if reason is not None:
            counted[reason] = counted.get(reason, 0) + 1
    return counted


def unusable_reasons(hits: Sequence[SearchHit], filtered: bool) -> list[str]:
    """One line per reason with the number of repositories behind it, instead of one line each."""
    return [
        f"{count} {_repositories(count)} cannot be picked: {reason}"
        for reason, count in sorted(unusable_counts(hits, filtered).items())
    ]


def model_choices(
    hits: Sequence[SearchHit],
    *,
    filtered: bool,
    checked: Checked,
    context: int,
    typed: str | None = None,
    configured_ollama: Mapping[str, str] | None = None,
) -> list[ModelChoice]:
    """One `ModelChoice` per resolved base model of the search, best fit first.

    Only what can be picked: with the owner filter on, a repository of an account that is neither
    a publisher nor a listed packager is no target of this run and no part of a model's `repos`
    either -- except the one repository `typed` names, which the filter never hides. The order is
    the memory pool (graphics memory first), then the fit class, then the downloads, then the name,
    with `too tight` and `unknown` last -- the fit is what a reader is choosing by, and the
    downloads say what many people take (never a rank).

    `configured_ollama` is the `ollama_base:ollama_tag` pair the configuration holds per base
    model: where no hit names an Ollama name, the list shows that pair, which the fetch uses anyway.
    """
    grouped: dict[str, list[SearchHit]] = {}
    for hit in hits:
        if hit_reason(hit, filtered, typed=typed) is not None or hit.resolved_base_model is None:
            continue
        grouped.setdefault(str(hit.resolved_base_model), []).append(hit)
    configured = configured_ollama or {}
    models = [_model_choice(base, repos, checked, context, configured.get(base)) for base, repos in grouped.items()]
    models.sort(key=_list_order)
    return models


def _list_order(model: ModelChoice) -> tuple:
    fit = model.fit
    settled = fit.fit_class in _SETTLED
    pool = _CLASS_ORDER.index(fit.fit_class) if settled else _MODE_ORDER.index(fit.mode)
    release = _RELEASE_ORDER.index(str(model.age)) if str(model.age) in _RELEASE_ORDER else len(_RELEASE_ORDER)
    return (
        settled,
        pool,
        _CLASS_ORDER.index(fit.fit_class),
        release,
        -(model.downloads or 0),
        model.name,
    )


def _model_choice(
    base_model: str, repos: list[SearchHit], checked: Checked, context: int, configured_ollama: str | None
) -> ModelChoice:
    publisher, _slash, name = base_model.partition("/")
    parameters_b = next((hit.parameters_b for hit in repos if hit.parameters_b is not None), None)
    if parameters_b is None:
        parameters_b = parameters_from_name(name)
    counts = [hit.downloads for hit in repos if hit.downloads is not None]
    # Age and basis from one and the same repository, so a row never shows one hit's age with
    # another hit's star.
    judged = next((hit for hit in repos if hit.age != UNKNOWN), None)
    return ModelChoice(
        base_model=base_model,
        name=name,
        publisher=publisher,
        repos=repos,
        packagers=_accounts(repos),
        downloads=sum(counts) if counts else None,
        ollama=next((hit.ollama for hit in repos if hit.ollama is not None), configured_ollama),
        age=judged.age if judged is not None else UNKNOWN,
        release_basis=judged.release_basis if judged is not None else None,
        parameters_b=parameters_b,
        fit=model_fit(parameters_b, checked, context),
    )


def _accounts(repos: Sequence[SearchHit]) -> list[str]:
    """The accounts of these repositories, each once, in the order the search answered them.

    Once, because an account with two builds of one model (`...-GGUF` and `...-UD-GGUF`) would
    otherwise stand twice in the column and make the `+N` count repositories rather than
    accounts -- measured live on 2026-09-24: `unsloth, unsloth +4`.
    """
    accounts: list[str] = []
    for hit in repos:
        owner = hit.repo.partition("/")[0]
        if owner not in accounts:
            accounts.append(owner)
    return accounts


def model_fit(parameters_b: float | None, checked: Checked, context: int) -> Fit:
    """The fit of one model before any package of it has been fetched, or why there is none.

    The machine is the one the size scale is about as well (`guided_context.machine_checked`):
    the profile this folder binds this machine to, with the reserves of its own
    `[machines.<name>]`. Without such a profile, and without a parameter count, the fit is
    `unknown` with that reason -- the list never shows a number nobody computed.
    """
    if checked.profile is None or checked.machine_config is None:
        return unknown_fit(MACHINE_NOT_MEASURED)
    if parameters_b is None:
        return unknown_fit(PARAMETER_COUNT_UNKNOWN)
    vram_gib = checked.profile.vram_gib if checked.profile.gpu_state == "measured" else 0.0
    return fit_from_parameters(
        parameters_b, vram_gib or 0.0, checked.profile.ram_physical_gib, checked.machine_config, context
    )


def list_header() -> Choice:
    """The column head of the list: the first line, and no entry anybody can pick.

    The list had none, so a reader had to guess what a cell meant and `latest` was invisible
    between two columns of numbers (test round, 2026-09-25).
    """
    return Choice("", _HEAD_INDENT + columns([list(COLUMN_NAMES)], _COLUMN_WIDTHS)[0], heading=True)


def list_choices(models: Sequence[ModelChoice]) -> list[Choice]:
    """The checkbox of step 2: one line per model, nothing grayed out, the base model as value.

    Every cell is cut to its column before the columns are laid out, so no name can push a line
    past `LABEL_LIMIT` -- `dialog.columns` pads, it never cuts, and that is right for the lists
    whose cells this package controls the length of. Here the length comes from a registry. A
    `legacy` row is dimmed as a whole: it is a model whose family has moved on, and that belongs in
    the reading of the row, not only in one cell of it.
    """
    rows = [
        [
            _clip(model.name, _COLUMN_WIDTHS[0]),
            _clip(model.fit_word, _COLUMN_WIDTHS[1]),
            _clip(model.size_text, _COLUMN_WIDTHS[2]),
            _clip(model.release_text, _COLUMN_WIDTHS[3]),
            model.packagers_text,  # already cut to its column
            _clip(format_downloads(model.downloads), _COLUMN_WIDTHS[5]),
            _clip(model.ollama_text, _OLLAMA_WIDTH),
        ]
        for model in models
    ]
    labels = columns(rows, _COLUMN_WIDTHS)
    return [Choice(model.base_model, label, dim=model.legacy) for model, label in zip(models, labels)]


def hint_line(checked: Checked, context: int) -> str:
    """What this list has to say about itself: the release words, the star, and where its fit is from.

    It is an instruction line of the question, not a line of the run (decided 2026-09-24): a
    sentence that explains a list has nothing to say once the list is gone. The context it names is
    the one the fit was really computed for, which is the one the next question starts on. What
    nothing marked does is no longer said here -- `Enter` takes the row under the pointer now
    (`dialog.TerminalAsker.checkbox`, decided 2026-09-25).
    """
    measured = checked.profile is not None and checked.machine_config is not None
    fit = FIT_FROM_SIZE_HINT.format(context=context_tokens_text(context)) if measured else FIT_UNKNOWN_HINT
    return f"{RELEASE_HINT} · {fit}"


def ask_models(asker: Asker, question: str, models: Sequence[ModelChoice], instruction: str) -> list[str]:
    """Ask the checkbox once, with what the asker can answer with.

    The list a person sees holds one line per model, and its values are base models. An answer
    file written before this list showed models names repositories instead
    (`unsloth/Qwen3.5-9B-GGUF`), and such a file keeps working: a file reads no list, so the
    repositories are added to the values an answer may name there -- and on no screen, where they
    would be the 120 lines this list exists to replace. The question is asked once either way, so
    a file's `select` is read once.

    The column head stands in front of the rows, as the first line of the list and no entry of it
    (`list_header`); a file reads no list, so it is no part of the choices an answer is held
    against (`dialog.selectable`) either way.
    """
    choices = [list_header(), *list_choices(models)]
    if isinstance(asker, FileAsker):
        choices = [*choices, *_repo_choices(models)]
    return asker.checkbox(SELECT_KEY, question, choices, instruction)


def _repo_choices(models: Sequence[ModelChoice]) -> list[Choice]:
    return [Choice(hit.repo, hit.repo) for model in models for hit in model.repos]


def picked_names(models: Sequence[ModelChoice], picked: Sequence[str]) -> list[str]:
    """What was marked, in the names the list showed: `Qwen3.5-9B`, not `Qwen/Qwen3.5-9B`.

    A value an answer file names as a repository keeps its own spelling -- it is what the file
    said, and an answer of the user appears in the words of the user (`AGENTS.md`, the language
    standard).
    """
    by_id = {model.base_model: model.name for model in models}
    return [by_id.get(value, value) for value in picked]


def chosen_hits(
    models: Sequence[ModelChoice], picked: Sequence[str], listed_packagers: Sequence[str]
) -> list[SearchHit]:
    """The repositories one answer stands for: two per model, or exactly the one it names.

    Per model: the publisher's own repository, if the search resolved one, and the repository of
    the **first** listed packager that has this model, in the order of the positive list. A model
    that only accounts outside both lists hold is fetched from the one with the most downloads --
    otherwise a person who picked such a model would get nothing at all. Every repository costs
    the fetch about ten requests of the shared budget (measured: two repositories spend 20 of
    150), and the ranking shows the packages of both, so two is the number (decided 2026-09-24).

    A value that is a base model **and** a repository of this search at once is read as the base
    model -- that is what the list offers, so it is what a person can have marked. Nothing forbids
    such a collision (a repository may package one model and be the base of another), so the
    precedence is stated here and in CONTRACTS.md rather than left to a branch order.
    """
    by_model = {model.base_model: model for model in models}
    chosen: list[SearchHit] = []
    for value in picked:
        model = by_model.get(value)
        found = _targets(model, listed_packagers) if model is not None else _named_repo(models, value)
        for hit in found:
            if hit.repo not in {already.repo for already in chosen}:
                chosen.append(hit)
    return chosen


def _account_hit(model: ModelChoice, account: str) -> SearchHit | None:
    """The repository of `account` that is fetched for this model, or `None` when it has none.

    The plain `<name>-GGUF` build where the account has one, else the account's first in the order
    of the search: `unsloth` answered `-MTP-GGUF`, `-GGUF` and `-UD-GGUF` for one model, newest
    first, and the newest was the special build (measured live 2026-09-24). The plain build is the
    one a person who picked the model expects; the other builds are stage 2's choice of packager.
    """
    own = [hit for hit in model.repos if hit.repo.partition("/")[0] == account]
    plain = f"{model.name}{_GGUF_SUFFIX}".lower()
    return next((hit for hit in own if hit.repo.partition("/")[2].lower() == plain), own[0] if own else None)


def _targets(model: ModelChoice, listed_packagers: Sequence[str]) -> list[SearchHit]:
    publisher = _account_hit(model, model.publisher)
    packager = None
    for account in listed_packagers:
        packager = _account_hit(model, account)
        if packager is not None:
            break
    found = [hit for hit in (publisher, packager) if hit is not None]
    if found:
        return found
    # Only accounts of neither list hold this model: the one many people take.
    return [max(model.repos, key=lambda hit: (hit.downloads or 0, hit.repo))]


def _named_repo(models: Sequence[ModelChoice], repo: str) -> list[SearchHit]:
    for model in models:
        for hit in model.repos:
            if hit.repo == repo:
                return [hit]
    return []


def _repositories(count: int) -> str:
    return "repository" if count == 1 else "repositories"


__all__ = [
    "COLUMN_NAMES",
    "COMPUTED_MARK",
    "DASH",
    "DEFAULT_CONTEXT",
    "LABEL_LIMIT",
    "LINE_LIMIT",
    "MACHINE_NOT_MEASURED",
    "OTHER_OWNER_REASON",
    "PARAMETER_COUNT_UNKNOWN",
    "POINTER_WIDTH",
    "RELEASE_COLUMN",
    "SELECT_KEY",
    "UNRESOLVED_REASONS",
    "ModelChoice",
    "ask_models",
    "chosen_hits",
    "hint_line",
    "hit_reason",
    "list_choices",
    "list_context",
    "list_header",
    "model_choices",
    "model_fit",
    "parameters_from_name",
    "picked_names",
    "unusable_counts",
    "unusable_reasons",
]
