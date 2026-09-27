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
from .guided_loadtest import NOTHING_MEASURED, load_test_step
from .intro import Fact, daemon_fact
from .measurements import Scenario
from .vllm_server import VLLM_BASE_URL, LocalVllmServer

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun

# The answer keys the registered servers ask beyond the daemon's own, for `guided.QUESTIONS`.
QUESTIONS: dict[str, str] = dict(guided_vllm.QUESTIONS)


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
    step: Callable[["GuidedRun", Configuration, Scenario, Path], str | None]
    card_facts: Callable[["GuidedRun"], tuple[Fact, ...]]


def _ollama_start(origins) -> Fact | None:
    """The daemon row of the start screen, and whether it answered (the pull gate reads that)."""
    fact, reachable = daemon_fact(origins.daemon)
    origins.servers.found["ollama"] = reachable
    return fact


def _ollama_step(run: "GuidedRun", config: Configuration, scenario: Scenario, results_dir: Path) -> str | None:
    """Step 4 of the daemon, as it was: a measurement runs one request (protocol v1).

    The load test gets a copy of the scenario at one request and the same context, the render the
    whole scenario (decided 2026-09-26). Where another server has a row on the start screen, the
    line about one request names Ollama, so it is not read as that server's.
    """
    one_request = scenario.model_copy(update={"requests": 1})
    others = any(key != "ollama" and value is not None for key, value in run.servers.found.items())
    return load_test_step(run, config, one_request, results_dir, ranking_requests=scenario.requests, named=others)


LOCAL_SERVERS: tuple[LocalServer, ...] = (
    LocalServer(
        key="ollama",
        name="Ollama",
        origin="http://127.0.0.1:11434",
        transport=lambda run: run.daemon,
        start_fact=_ollama_start,
        step=_ollama_step,
        card_facts=lambda run: (),
    ),
    LocalServer(
        key="vllm",
        name="vLLM",
        origin=VLLM_BASE_URL,
        transport=lambda run: run.servers.transports.get("vllm"),
        start_fact=guided_vllm.start_fact,
        step=guided_vllm.vllm_step,
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
    """Step 4 of every registered server, in order; returns the balance of the step.

    The daemon's own balance stands first where it measured anything, each other server's after it
    behind a dot; where nothing was measured anywhere it is the daemon's `none in this run`.
    """
    parts = [server.step(run, config, scenario, results_dir) for server in LOCAL_SERVERS]
    said = [part for part in parts if part and part != NOTHING_MEASURED]
    return f" {run.screen.glyphs.dot} ".join(said) if said else NOTHING_MEASURED


def card_facts(run: "GuidedRun") -> tuple[Fact, ...]:
    """The rows every registered server adds to the card of step 5, in the order of the list."""
    return tuple(fact for server in LOCAL_SERVERS for fact in server.card_facts(run))


__all__ = [
    "LOCAL_SERVERS",
    "QUESTIONS",
    "LocalServer",
    "Origins",
    "ServerRun",
    "card_facts",
    "default_transports",
    "load_tests",
    "start_rows",
]
