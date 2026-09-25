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
   list, so a search that found nothing ends in a choice rather than in an empty step.

Everything but `search_step` is pure. `GuidedRun` is only an annotation here (`TYPE_CHECKING`):
`guided.py` imports this module, never the other way round at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .config import Configuration
from .dialog import Choice
from .guided_context import Checked, machine_checked
from .guided_models import ModelChoice, list_context, model_choices, unusable_counts
from .screen import number_word, repositories_note, searched_note, yes_no
from .search import DEFAULT_PACKAGERS, SearchError, SearchOutcome, run_search
from .state import write_search_log

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

    `checked` and `context` are the machine and the context the fit of every line was computed for,
    so the instruction line of the list says the same numbers the list shows.
    """

    outcome: SearchOutcome
    models: list[ModelChoice]
    checked: Checked
    context: int
    filtered: bool


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
    step = _searched(run, config, name, filtered, checked, context)
    if not step.models and step.outcome.candidates:
        again = _did_you_mean(run, step.outcome.candidates)
        if again is not None:
            step = _searched(run, config, again, filtered, checked, context)
    _report(run, config, step)
    return step


def _searched(
    run: GuidedRun, config: Configuration, name: str, filtered: bool, checked: Checked, context: int
) -> SearchStep:
    """One search and the models it offers -- no line written yet, so a retry leaves none behind.

    The owner filter is a question about the **request** and about the resolution, not only about
    the list: with it on, only the accounts are asked and only a base model of a catalog publisher
    resolves; switching it off adds the two open lists and lets any account's base model resolve
    (decided 2026-09-25). A typed repository id is never hidden by it.

    `SearchError` becomes the guided mode's own error here, so the caller keeps its exit code; the
    import is local because `guided.py` imports this module.
    """
    from .guided import GuidedError

    try:
        outcome = run_search(
            run.transport,
            name,
            catalog=run.catalog,
            listed_packagers=config.packagers or DEFAULT_PACKAGERS,
            budget=run.budget,
            open_pages=not filtered,
            publisher_required=filtered,
        )
    except SearchError as exc:
        raise GuidedError(str(exc)) from exc
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
    return SearchStep(outcome=outcome, models=models, checked=checked, context=context, filtered=filtered)


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
    for note in [*outcome.notes, *search_notes(log, len(outcome.hits), len(step.models), run.screen.glyphs.dot)]:
        run.screen.note(note)


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
    """
    accounts = log.get("accounts", [])
    if log.get("mode") == "id":
        return TYPED_SENTENCE.format(repo=log.get("word", ""))
    if log.get("mode") == "catalog":
        return CATALOG_SENTENCE.format(count=len(accounts))
    publishers = [entry["account"] for entry in accounts if entry["class"] == "publisher"]
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
