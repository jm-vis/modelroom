"""The local inference servers of this machine, as one list: the one place a server is registered.

`LOCAL_SERVERS` names every origin the guided mode reads, in the order it reads them: the Ollama
daemon (what existed before, unchanged in behavior) and a vLLM server (AP9-SV1). Each entry says
where its transport comes from, the row it adds to the start screen, what it does in step 4, and
the rows it adds to the card of step 5. The start screen, step 4 and both card paths loop over
this list; a server added to it reaches all three without a change anywhere else.

Nothing a server states or measures here is persisted (CONTRACTS.md, "Local servers and the vLLM
load test", ADR 0008): what the start screen read and what step 4 found live in the run's
`ServerRun` for as long as the run lasts, and no longer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from . import guided_vllm
from .config import Configuration
from .dialog import Choice
from .guided_loadtest import NO_CANDIDATE_LINE, NO_INSTALLED_LINE, NOTHING_MEASURED, QUESTIONS as LOAD_TEST_QUESTIONS
from .guided_loadtest import STEP, installed_note, measure_ollama, ollama_offer, ollama_rows, wants_load_test
from .intro import Fact, daemon_fact
from .measurements import Scenario
from .screen import names_words, yes_no
from .vllm_server import VLLM_BASE_URL, LocalVllmServer

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

# The answer keys the registered servers ask beyond the one question and the one list of step 4
# (`load_test`, `load_test_packages`): none since AP9-UX E -- `load_test_vllm` is gone.
QUESTIONS: dict[str, str] = {}


@dataclass(frozen=True)
class Offer:
    """What one server offers step 4 before the one question (AP9-UX E).

    `rows` are its rows of the one list; `unavailable` is why a server that should answer did not (a
    `yes` only it could honor is a step that did not finish); `nothing` what it says where it offers
    no row: alone, and next to another server's rows. `state` is the server's own, for `measure`.
    """

    rows: tuple[Choice, ...] = ()
    unavailable: str | None = None
    nothing: tuple[str, str] | None = None
    state: Any = None


@dataclass
class ServerRun:
    """The local servers of one guided run: their transports, and what each step found.

    `transports` holds the transport of every registered server but Ollama's, whose transport is
    the run's `daemon`; `found` is what the start screen read, `measured` what step 4 did, both by
    the server's key.
    """

    transports: Mapping[str, Any]
    found: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Origins:
    """What a start row is read from outside a run: the daemon and the other servers' transports."""

    daemon: Any
    servers: ServerRun


@dataclass(frozen=True)
class LocalServer:
    """One registered server: its key, its name, its origin, and what each place of the run asks of it."""

    key: str
    name: str
    origin: str
    transport: Callable[[Any], Any]
    start_fact: Callable[[Any], Fact | None]
    offer: Callable[["GuidedRun", Configuration, Scenario, Path], Offer]
    measure: Callable[["GuidedRun", Configuration, Scenario, Path, Offer, list[str]], str | None]
    card_facts: Callable[["GuidedRun"], tuple[Fact, ...]]


def _ollama_start(origins) -> Fact | None:
    """The daemon row of the start screen, and whether it answered (the pull gate reads that)."""
    fact, reachable = daemon_fact(origins.daemon)
    origins.servers.found["ollama"] = reachable
    return fact


def _ollama_offer(run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path) -> Offer:
    """The daemon's notes and candidates. Where another server has a row on the start screen, the
    line about one request names Ollama, so it is not read as that server's."""
    others = any(key != "ollama" and value is not None for key, value in run.servers.found.items())
    found = ollama_offer(run, config, results_dir, ranking_requests=scenario.requests, named=others)
    if found.unavailable is not None:
        return Offer(unavailable=found.unavailable, state=found)
    if found.profile_id is None:
        return Offer(state=found)
    rows = ollama_rows(found)
    if not rows:
        return Offer(nothing=(NO_CANDIDATE_LINE, NO_INSTALLED_LINE), state=found)
    run.screen.note(installed_note(len(rows)))
    return Offer(tuple(rows), state=found)


def _ollama_measure(
    run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path, offer: Offer, picked: list[str]
) -> str | None:
    """Measure the picked candidates: a measurement runs one request (protocol v1).

    The load test gets a copy of the scenario at one request and the same context, the render the
    whole scenario (decided 2026-09-26).
    """
    one_request = scenario.model_copy(update={"requests": 1})
    return measure_ollama(run, config, offer.state, one_request, picked)


def _vllm_offer(run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path) -> Offer:
    found = guided_vllm.vllm_offer(run, config, scenario, results_dir)
    return Offer(tuple(guided_vllm.vllm_rows(found)), state=found)


def _vllm_measure(
    run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path, offer: Offer, picked: list[str]
) -> str | None:
    return guided_vllm.measure_vllm(run, config, offer.state, scenario, results_dir, picked)


LOCAL_SERVERS: tuple[LocalServer, ...] = (
    LocalServer(
        key="ollama",
        name="Ollama",
        origin="http://127.0.0.1:11434",
        transport=lambda run: run.daemon,
        start_fact=_ollama_start,
        offer=_ollama_offer,
        measure=_ollama_measure,
        card_facts=lambda run: (),
    ),
    LocalServer(
        key="vllm",
        name="vLLM",
        origin=VLLM_BASE_URL,
        transport=lambda run: run.servers.transports.get("vllm"),
        start_fact=guided_vllm.start_fact,
        offer=_vllm_offer,
        measure=_vllm_measure,
        card_facts=guided_vllm.card_facts,
    ),
)


def default_transports() -> dict[str, Any]:
    """The real transport of every registered server but Ollama's; constructing one connects nowhere."""
    return {"vllm": LocalVllmServer()}


def start_rows(daemon, servers: ServerRun) -> tuple[tuple[Fact, ...], bool]:
    """One row per registered server that answered, the daemon's first; and whether the daemon answered."""
    origins = Origins(daemon, servers)
    rows = tuple(fact for server in LOCAL_SERVERS if (fact := server.start_fact(origins)) is not None)
    return rows, bool(servers.found.get("ollama"))


def load_tests(run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path) -> str:
    """Step 4 from one mould (AP9-UX E): every server's notes, then **one** question, then **one** list.

    The notes of every registered server come first, in order, the reason a number will not count
    among them. The question `load_test` is asked where any server offers a row, or where a server
    that should answer did not (a `yes` then is a step that did not finish, as before). The list
    `load_test_packages` holds every row, nothing marked. The balance puts the daemon's own first and
    each other server's behind a dot; where nothing was measured anywhere it is `none in this run`.
    """
    run.screen.blank()
    run.screen.head(STEP)
    offers = [(server, server.offer(run, config, scenario, results_dir)) for server in LOCAL_SERVERS]
    rows = [row for _server, offer in offers for row in offer.rows]
    for server, offer in offers:
        if offer.nothing is not None:
            run.screen.note(offer.nothing[1] if rows else offer.nothing[0])
        if offer.unavailable is not None and rows:
            run.screen.note(f"{server.name}: {offer.unavailable}")
    picked = _asked(run, rows, [offer.unavailable for _server, offer in offers if offer.unavailable is not None])
    parts = [server.measure(run, config, scenario, results_dir, offer, picked) for server, offer in offers]
    said = [part for part in parts if part and part != NOTHING_MEASURED]
    return f" {run.screen.glyphs.dot} ".join(said) if said else NOTHING_MEASURED


def _asked(run: "GuidedRun", rows: list[Choice], unavailable: list[str]) -> list[str]:
    """The one question and the one list; what was picked, or nothing."""
    if not rows and not unavailable:
        return []
    wants = wants_load_test(run)
    run.screen.answer(LOAD_TEST_QUESTIONS["load_test"], yes_no(wants))
    if not rows:
        # The daemon could not be read and nothing else offers a row: a yes cannot be honored.
        if wants:
            run.failed(f"nothing measured: {unavailable[0]}")
        else:
            run.screen.note(f"no load test: {unavailable[0]}")
        return []
    if not wants:
        return []
    picked = run.asker.checkbox("load_test_packages", LOAD_TEST_QUESTIONS["load_test_packages"], rows)
    words = [ROW_WORDS.get(value, value) for value in picked]
    run.screen.answer(LOAD_TEST_QUESTIONS["load_test_packages"], names_words(words, "nothing"))
    return picked


# The answer line names a server's row in words; an Ollama row is its local name already.
ROW_WORDS = {guided_vllm.ROW: "vLLM server"}


def card_facts(run: "GuidedRun") -> tuple[Fact, ...]:
    """The rows every registered server adds to the card of step 5, in the order of the list."""
    return tuple(fact for server in LOCAL_SERVERS for fact in server.card_facts(run))


__all__ = [
    "LOCAL_SERVERS",
    "QUESTIONS",
    "LocalServer",
    "Offer",
    "Origins",
    "ServerRun",
    "card_facts",
    "default_transports",
    "load_tests",
    "start_rows",
]
