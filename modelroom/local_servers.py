"""The local inference servers of this machine, as one list: a view on `runtimes.RUNTIMES`.

`LOCAL_SERVERS` names every origin the guided mode reads, in the order it reads them: the Ollama
daemon (what existed before, unchanged in behavior) and a vLLM server (AP9-SV1). Each entry says
where its transport comes from, the row it adds to the start screen, what it does in step 4, and
the rows it adds to the card of step 5. The start screen, step 4 and both card paths loop over
this list; a server added to it reaches all three without a change anywhere else.

`LOCAL_SERVERS` and the types it holds (`LocalServer`, `Offer`, `ServerRun`, `Origins`) live in
`runtimes.py`: `runtimes.RUNTIMES` is the one place a server or a runtime is registered. This
module re-exports them, so every import of them from here stays valid, and keeps only what has
nothing runtime-specific in it: reading the whole list for the start screen (`start_rows`), running
step 4 over it (`load_tests`) and reading its card rows (`card_facts`) -- each one reads the table
as it is at the call (`runtimes.local_servers()`).

Nothing a server states or measures here is persisted (CONTRACTS.md, "Local servers and the vLLM
load test", ADR 0008): what the start screen read and what step 4 found live in the run's
`ServerRun` for as long as the run lasts, and no longer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import guided_vllm
from .config import Configuration
from .dialog import Choice
from .guided_loadtest import NOTHING_MEASURED, QUESTIONS as LOAD_TEST_QUESTIONS, STEP, wants_load_test
from .guided_remote_vllm import card_facts as remote_card_facts
from .intro import Fact
from .measurements import Scenario
from . import runtimes
from .runtimes import LOCAL_SERVERS, QUESTIONS, LocalServer, Offer, Origins, ServerRun
from .screen import names_words, yes_no
from .vllm_server import LocalVllmServer

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun


def default_transports() -> dict[str, Any]:
    """The real transport of every registered server but Ollama's; constructing one connects nowhere."""
    return {"vllm": LocalVllmServer()}


def start_rows(daemon, servers: ServerRun) -> tuple[tuple[Fact, ...], bool]:
    """One row per registered server that answered, the daemon's first; and whether the daemon answered."""
    origins = Origins(daemon, servers)
    rows = tuple(fact for server in runtimes.local_servers() if (fact := server.start_fact(origins)) is not None)
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
    offers = [(server, server.offer(run, config, scenario, results_dir)) for server in runtimes.local_servers()]
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
    """The rows every registered server adds to the card of step 5, in the order of the list, then the vLLM
    server of the active computer over SSH where step 4 measured it (AP9-SV2-R, R3)."""
    local = tuple(fact for server in runtimes.local_servers() for fact in server.card_facts(run))
    return local + remote_card_facts(run)


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
