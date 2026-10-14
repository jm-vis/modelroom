"""The search of step 2: its two questions, its notes, and what it offers when nothing was found.

New on 2026-09-25, beside `guided.py::_search_step` rather than inside it: the step grew the three
forms of input (a repository id, loose words, no word at all) and `guided.py` already stands at the
house code-mass threshold.

What lives here:

1. **the two questions** of the search -- the word and the owner filter -- and the search they lead
   to, up to and including the candidate question a typo ends in. `search_step` returns the models
   the list will offer; the choice itself, and the write-back into the configuration, stay in
   `guided.py`, which owns the file;
2. **the two notes** the step keeps on the screen. The second one is `screen.repositories_note`
   unchanged; the first one adds the case the search of 2026-09-24 had no sentence for -- a
   publisher of the catalog was asked and answered with nothing at all, which used to read "at no
   publisher of this catalog that answered" and left the reader guessing which one that was;
3. **the candidates** a typo leads to (`search.SearchOutcome.candidates`), as the entries of a
   list, so a search that found nothing ends in a choice rather than in an empty step;
4. **what the catalog adds** (AP9-KO, `search_direct.py`): with a vLLM machine among the chosen ones
   the publisher's own models the input names are asked for directly and join the list before it
   is built (V8); without one, a gray line says which of them no GGUF is known for (V9).

Everything but `search_step` is pure. `GuidedRun` is only an annotation here (`TYPE_CHECKING`):
`guided.py` imports this module, never the other way round at run time.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .config import Configuration
from .contracts import Package, SchemaVersionError, hf_id_key
from .dialog import Choice
from .guided_context import Checked, machine_checked, vllm_among_chosen
from .guided_models import LINE_LIMIT, ModelChoice, hit_reason, list_context, model_choices, unusable_counts
from .screen import NOTE_INDENT, number_word, repositories_note, searched_note, yes_no
from .search import DEFAULT_PACKAGERS, SearchError, SearchOutcome, run_search
from .search_direct import find_direct, knowledge_hints
from .search_word import parse_search
from .state import UnreadableStateFileError, read_snapshot, write_search_log
from .tensor_search import search_tensor_repos

if TYPE_CHECKING:  # pragma: no cover -- the annotation only; no import at run time
    from .guided import GuidedRun

# The two questions of the search, so that the step and `guided.QUESTIONS` name them once.
SEARCH_QUESTION = "What are you looking for?"
FILTER_QUESTION = "Show only repositories of a publisher or a listed packager?"

# The question the candidates are offered under, and its key in an answer file. It is asked only
# when the search found no model and the catalog knows a word close to the one that was typed, so
# an answer file that never runs into that case never needs the key.
DID_YOU_MEAN_KEY = "did_you_mean"
DID_YOU_MEAN_QUESTION = "Nothing was found. Did you mean one of these?"
# The one entry that is not a candidate: it keeps the run going with what the search found, which
# is nothing -- the folder then stays as it is, exactly as an empty selection leaves it.
KEEP_THE_WORD = "keep"
KEEP_THE_WORD_LABEL = "none of these"
# An account page is asked **with** the search word, so an empty answer says nothing about
# everything that account holds -- only that this search found none of it (second-model round,
# 2026-09-25, which is why both sentences end in "for this search").
NO_GGUF_ONE = "the publisher {publishers} has no GGUF repository for this search"
NO_GGUF_MANY = "the publishers {publishers} have no GGUF repository for this search"
# What the first note says for the two searches that ask no account page at all. Without them the
# sentence read "at no publisher of this catalog that answered and the no listed packagers, and the
# two open lists" -- for a run that asked one request and no open list (live probe, 2026-09-25).
TYPED_SENTENCE = "searched Hugging Face for the repository {repo}"
CATALOG_SENTENCE = "searched Hugging Face for the {count} current models of this catalog"
# What the answer line of the search question says when nothing was typed. The answer is a
# decision -- "I have no model in mind" -- so the line says what it means, never an empty space.
NO_WORD_ANSWER = "anything that fits this machine"


def search_answer(text: str) -> str:
    """The answer line of the search question: the word as it was typed, or what nothing means."""
    return text.strip() or NO_WORD_ANSWER


@dataclass(frozen=True)
class SearchStep:
    """What the search of step 2 leaves the choice with: the models, and what they were computed on.

    `checked`, `context` and `requests` are the machine, the context and the requests at once the fit
    of every line was computed for, so the instruction line of the list says the same numbers the
    list shows. `context_kept` and `requests_kept` say whether step 3 kept them in this folder
    (where the values come from, never read off the numbers).

    `tensor_repos` (P1) maps a resolved base model to the tensor
    repositories P1's search found for it -- a read only, no write: which of them end up in the
    configuration is decided later, by the selection, through the same write path a GGUF hit goes
    through (`guided.py::_search_step`, `search_apply.apply_hits`/`apply_tensor_repos` folded into
    one `change`). Empty for a machine without vLLM, or when the search found nothing.
    """

    outcome: SearchOutcome
    models: list[ModelChoice]
    checked: Checked
    context: int
    requests: int
    context_kept: bool
    requests_kept: bool
    filtered: bool
    tensor_repos: dict[str, list[str]] = field(default_factory=dict)
    # AP9-KO (V9): the gray lines of a search without a vLLM machine, shown after the two notes.
    hints: list[str] = field(default_factory=list)


def search_step(run: GuidedRun, config: Configuration, results_dir: Path) -> SearchStep:
    """Ask the word and the owner filter, search, and report -- the first half of step 2.

    The answer may be a repository id, a name with blanks in it, or nothing at all
    (`search_word.parse_search`, decided 2026-09-25) -- an empty answer means "show me what fits
    this machine" and is no longer an error. A search that finds no model and a catalog that knows a
    word close to the one that was typed end in one more question, the candidates, rather than in an
    empty step.

    Two notes stay on the screen: where the search asked, and how much of what it answered is a
    choice, both from the search that was really **used** -- a first search that led to the candidate
    question leaves no lines behind, so the step still carries one reading. Which account answered
    what, the request count, the budget and the reasons go into `search.json` (CONTRACTS.md,
    "Search log"), from that same reading.
    """
    name = run.asker.text("search", SEARCH_QUESTION)
    run.screen.answer(SEARCH_QUESTION, search_answer(name))
    filtered = run.asker.confirm("filter_owners", FILTER_QUESTION, default=True)
    run.screen.answer(FILTER_QUESTION, yes_no(filtered))
    context = list_context(config.guided.context)
    checked = machine_checked(run.pointer_path, config, results_dir)
    vllm = vllm_among_chosen(run, config)
    step = _searched(run, config, name, filtered, checked, context, direct=vllm)
    if not step.models and step.outcome.candidates:
        again = _did_you_mean(run, step.outcome.candidates)
        if again is not None:
            step = _searched(run, config, again, filtered, checked, context, direct=vllm)
    tensor_repos = _tensor_hits(run, vllm, step.outcome.hits)
    # The tensor requests are booked in the log's budget as well, like the direct find's: it says what is left for the fetch.
    step = replace(step, tensor_repos=tensor_repos, outcome=replace(step.outcome, budget_used=run.budget.used))
    if not vllm:
        step = _with_hints(run, config, step)
    _report(run, config, step)
    return step


def _searched(
    run: GuidedRun,
    config: Configuration,
    name: str,
    filtered: bool,
    checked: Checked,
    context: int,
    *,
    direct: bool = False,
) -> SearchStep:
    """One search and the models it offers -- no line written yet, so a retry leaves none behind.

    The owner filter is a question about the **request** and about the resolution, not only about
    the list: with it on, only the accounts are asked and only a base model of a catalog publisher
    resolves; switching it off adds the two open lists and lets any account's base model resolve
    (decided 2026-09-25). A typed repository id is never hidden by it.

    `SearchError` becomes the guided mode's own error here, so the caller keeps its exit code; the
    import is local because `guided.py` imports this module. A single page that did not answer is no
    `SearchError` (AP9-KO P5): it is a note of the outcome, and only a search with no answer at all ends here.
    """
    from .guided import GuidedError

    try:
        outcome = run_search(
            run.transport,
            name,
            catalog=run.catalog,
            listed_packagers=config.packagers or run.catalog.packagers or DEFAULT_PACKAGERS,
            budget=run.budget,
            open_pages=not filtered,
            publisher_required=filtered,
        )
    except SearchError as exc:
        raise GuidedError(str(exc)) from exc
    if direct:
        outcome = _with_direct_finds(run, name, filtered, outcome)
    # The pair the configuration holds per base model, for a model no hit names an Ollama name of:
    # the fetch uses it, so the list shows it rather than a dash (decided 2026-09-25).
    configured = {
        entry.hf_repo: f"{entry.ollama_base}:{entry.ollama_tag}"
        for family in config.families
        for entry in family.base_models
        if entry.ollama_base is not None
    }
    models = model_choices(
        outcome.hits,
        filtered=filtered,
        checked=checked,
        context=context,
        typed=outcome.typed,
        configured_ollama=configured,
        requests=config.guided.requests,
    )
    return SearchStep(
        outcome=outcome,
        models=models,
        checked=checked,
        context=context,
        requests=config.guided.requests,
        context_kept=config.guided.context is not None,
        requests_kept=config.guided.requests_origin != "default",
        filtered=filtered,
    )


def _with_direct_finds(run: GuidedRun, name: str, filtered: bool, outcome: SearchOutcome) -> SearchOutcome:
    """AP9-KO (V8): the publisher's own models the input names, beside the packages found by GGUF.

    Only with a vLLM machine among the chosen ones. Made before the list is built and before the question
    after a typo, so a model with no GGUF at all is a choice even when the owner filter is on. A base model a
    package hit already offers (`hit_reason` is `None`, by `hf_id_key`) is not asked for again. What is found
    is only read: the configuration is written for the models a person chooses, through the existing change.
    """
    offered = {
        hf_id_key(hit.resolved_base_model)
        for hit in outcome.hits
        if hit.resolved_base_model is not None and hit_reason(hit, filtered, typed=outcome.typed) is None
    }
    found = find_direct(run.transport, run.budget, run.catalog, parse_search(name), skip=offered)
    # The requests of the direct find are booked in the log's budget as well: it says what is left for the fetch.
    return replace(outcome, hits=[*outcome.hits, *found.hits], direct=found.log, budget_used=run.budget.used)


def _with_hints(run: GuidedRun, config: Configuration, step: SearchStep) -> SearchStep:
    """AP9-KO (V9): what the catalog knows about the models the input names, for a search with no vLLM machine.

    No request. A GGUF hit of this very search that resolved to a model counts as a known GGUF for it.
    """
    found = {hf_id_key(hit.resolved_base_model) for hit in step.outcome.hits if hit.resolved_base_model is not None}
    hints, left_out = knowledge_hints(run.catalog, parse_search(step.outcome.query), found, _tensor_proven(config))
    return replace(step, hints=hints, outcome=replace(step.outcome, knowledge_hint=left_out))


def _tensor_proven(config: Configuration) -> set[str]:
    """The models the snapshot of this folder holds a tensor package of: the positive proof of the hint's second sentence."""
    try:
        snapshot = read_snapshot(config)
    except (SchemaVersionError, UnreadableStateFileError):
        return set()
    return proven_tensor_models(snapshot.packages) if snapshot is not None else set()


def proven_tensor_models(packages: Sequence[Package]) -> set[str]:
    """The `hf_id_key` of every base model with an active, complete Hugging Face tensor package."""
    return {
        hf_id_key(package.base_model_hf_repo)
        for package in packages
        if package.source == "huggingface" and package.format == "tensor" and package.active and package.complete
    }


def _tensor_hits(run: GuidedRun, wanted: bool, hits: Sequence) -> dict[str, list[str]]:
    """P1: the vLLM tensor search, a read only -- no write.

    Only when `wanted` -- `guided_context.vllm_among_chosen`, the one gate `guided._fetch_step`'s
    `build_tensor` reads too: at least one chosen machine has vLLM -- and only then, every base
    model this very search resolved (`hits`, not only the ones a person later picks: this step
    runs *before* the choice, there is no "later" to wait for here) is asked once per
    `tensor_search.TENSOR_PACKAGERS` account. A run without a vLLM machine asks no account here
    at all, so an ordinary Ollama run spends no extra request and its goldens stay byte-identical.

    What is found is only *read*, never written: `guided.py::_search_step` decides, after the
    person has chosen, which of these base models to keep, and folds them into the exact same
    write `apply_hits` already makes for the resolved GGUF repositories of that choice -- the
    existing selection-and-write path the GGUF hits already go through
    (`search_apply.apply_hits`/`apply_tensor_repos`, one `change`, one lock, one re-read on
    conflict), never a write path of this step's own. A base model nobody picks is never written
    for, whatever this search found.
    """
    if not wanted:
        return {}
    bases = list(dict.fromkeys(hit.resolved_base_model for hit in hits if hit.resolved_base_model is not None))
    repos_by_base: dict[str, list[str]] = {}
    for base in bases:
        outcome = search_tensor_repos(run.transport, run.budget, base)
        if outcome.repos:
            repos_by_base[base] = list(outcome.repos)
        note = outcome.note()
        if note is not None:
            run.screen.note(f"{base}: {note}")
        for error in outcome.errors:  # a diagnosis, not a sentence: the account and the kind of error
            run.screen.note(f"{base}: {error}")
    return repos_by_base


def _did_you_mean(run: GuidedRun, candidates: Sequence[str]) -> str | None:
    """The candidate list of a search that found nothing; `None` when the word is kept as it was.

    A list to pick from and not a sentence to read: a person who mistyped a name has no use for
    being told that nothing was found (decided 2026-09-25).
    """
    choice = run.asker.select(DID_YOU_MEAN_KEY, DID_YOU_MEAN_QUESTION, candidate_choices(candidates))
    kept = choice == KEEP_THE_WORD
    run.screen.answer(DID_YOU_MEAN_QUESTION, KEEP_THE_WORD_LABEL if kept else choice)
    return None if kept else choice


def _report(run: GuidedRun, config: Configuration, step: SearchStep) -> None:
    """Write `search.json` and leave the search's own sentences plus the two notes on the screen."""
    outcome = step.outcome
    log = outcome.search_log(
        filtered=step.filtered,
        run_at=run.now,
        unresolved=unusable_counts(outcome.hits, step.filtered, typed=outcome.typed),
    )
    try:
        write_search_log(config, log)
    except OSError as exc:
        # The search itself went through; its log is a record beside the run. A folder that cannot
        # hold it is said out loud and the run goes on with the choice it has to offer.
        run.failed(f"the log of this search was not written: {exc}")
    notes = [*run.catalog_notes, *outcome.notes, *search_notes(log, len(outcome.hits), len(step.models), run.screen.glyphs.dot)]
    for note in [*notes, *(part for hint in step.hints for part in hint_lines(hint))]:
        run.screen.note(note)


def hint_lines(hint: str) -> list[str]:
    """One hint of the catalog (E3, fixed words) in the lines of a 100-column screen, broken at blanks only.

    The sentence is longer than a note line for a model with a long name; the words are the fixed ones and
    only the line break is ours (the layout rule of decision E3). The indent of a note is counted.
    """
    return textwrap.wrap(hint, LINE_LIMIT - len(NOTE_INDENT), break_long_words=False, break_on_hyphens=False)


def search_notes(log: dict, repositories: int, models: int, dot: str = "·") -> list[str]:
    """The two notes step 2 keeps on the screen, read from the log of that very search.

    One reading of the search for the file and for the screen (`search.SearchOutcome.search_log`):
    a line and a file may not say different numbers about the same thing.
    """
    accounts = log.get("accounts", [])
    return [
        searched_sentence(log),
        repositories_note(
            repositories, models, page_full=any(entry["page_full"] for entry in accounts), dot=dot
        ),
    ]


def searched_sentence(log: dict) -> str:
    """Where this search asked, and which asked publisher has no GGUF repository for the word.

    A typed repository id and a run with no word at all ask no account page, so they get a sentence
    of their own -- the account one said "at no publisher of this catalog that answered and the no
    listed packagers, and the two open lists" about a run that asked one request and no open list
    (live probe, 2026-09-25).

    For a word search with a publisher that answered, this is `screen.searched_note` unchanged. Where
    every publisher of the catalog that was asked came back empty, the sentence names them instead:
    "no publisher of this catalog that answered" is true but tells a reader nothing they can act on,
    and the accounts were asked -- which is the whole question the change of 2026-09-24 exists to
    answer (decided 2026-09-25).

    A publisher with no hit and a page that did not answer (`failed_pages`, AP9-KO P5) is not named as
    having no GGUF repository: that is not known, and the page's own note says what went wrong.
    """
    accounts = log.get("accounts", [])
    if log.get("mode") == "id":
        return TYPED_SENTENCE.format(repo=log.get("word", ""))
    if log.get("mode") == "catalog":
        return CATALOG_SENTENCE.format(count=len(accounts))
    unanswered = {entry["account"] for entry in log.get("failed_pages", [])}
    publishers = [
        entry["account"]
        for entry in accounts
        if entry["class"] == "publisher" and (entry["hits"] or entry["account"] not in unanswered)
    ]
    answered = [entry["account"] for entry in accounts if entry["class"] == "publisher" and entry["hits"]]
    packagers = len({entry["account"] for entry in accounts if entry["class"] == "packager"})
    open_lists = any(entry["class"] == "open" for entry in accounts)
    if answered or not publishers:
        return searched_note(answered, packagers, open_lists=open_lists)
    open_part = ", and the two open lists" if open_lists else ""
    return (
        f"searched Hugging Face at the {number_word(packagers)} listed packagers{open_part}; "
        f"{no_gguf_clause(publishers)}"
    )


def no_gguf_clause(publishers: Sequence[str]) -> str:
    """The publishers that were asked and answered with nothing, named."""
    if len(publishers) == 1:
        return NO_GGUF_ONE.format(publishers=publishers[0])
    listed = ", ".join(publishers[:-1]) + f" and {publishers[-1]}"
    return NO_GGUF_MANY.format(publishers=listed)


def candidate_choices(candidates: Sequence[str]) -> list[Choice]:
    """The candidates as the entries of a list, with `none of these` last.

    The pointer starts on the first candidate: it is the closest one `difflib` found, and a person
    who typed a word with a spelling mistake in it means the word, not the mistake.
    """
    entries = [Choice(word, word, checked=index == 0) for index, word in enumerate(candidates)]
    return [*entries, Choice(KEEP_THE_WORD, KEEP_THE_WORD_LABEL)]


__all__ = [
    "DID_YOU_MEAN_KEY",
    "DID_YOU_MEAN_QUESTION",
    "FILTER_QUESTION",
    "KEEP_THE_WORD",
    "KEEP_THE_WORD_LABEL",
    "NO_GGUF_MANY",
    "NO_GGUF_ONE",
    "NO_WORD_ANSWER",
    "CATALOG_SENTENCE",
    "SEARCH_QUESTION",
    "TYPED_SENTENCE",
    "SearchStep",
    "candidate_choices",
    "no_gguf_clause",
    "search_answer",
    "search_notes",
    "search_step",
    "searched_sentence",
]
