"""The one table of local-server runtimes: "Laufzeit als Eintrag" (AP9-SV2-L, decided 2026-09-29).

Everything that was hard-wired to "Ollama" -- package format, fit, pull command, model folder,
the "already there" check, the words a row or question uses, and the server itself (start screen,
step 4, the card) -- hangs from one row of `RUNTIMES` now, the **single** place a runtime or a
local server is registered (Codex r1, fix 1: `local_servers.py` no longer builds `LOCAL_SERVERS`
on its own; it derives it from `RUNTIMES` and stays the thin, generic view CONTRACTS.md, "Local
servers and the vLLM load test" and docs/adr/0014-runtime-entries-and-vllm-packages.md already
describe). A later runtime (vLLM's own fields first, then LM Studio, AP9-LS) is a new row, never a
plug-in: no dynamic loading, no registration at runtime.

AP9-SV2-L moves Ollama onto this table without a behavior change: `OLLAMA.accepts`, `.fit`,
`.command`, `.model_place` and `.on_disk` are today's functions, referenced here, not rewritten,
and every caller that used to read them directly now reads them through this table (`pull_choice.py`,
`views_install.py`, `guided_install.py` -- Codex r1, fixes 2-4). `VLLM` in this sub-package carries
only what AP9-SV1 already built -- its `server` and `hello_field` -- and `accepts = lambda package:
False`, so it never claims a package before AP9-SV2-P builds the Tensor package class and AP9-SV2-F
builds its fit. Its other fields are named stand-ins that raise if ever called, each one saying
which later sub-package fills it in; `test_every_runtime_has_every_field` only checks that every
field is set, never that a vLLM one runs.

`Runtime.detect`, `.command`, `.model_place` and `.on_disk` are typed loosely on purpose: the
brief that describes the finished table (`DetectProbes`, `RuntimeState`, `InstallCommand`,
`OnDiskProbe`) names types that do not exist yet -- SV2-I and SV2-R still have to design them.
SV2-L only wires today's functions in.

Two imports are deferred to call time, inside the functions that need them (the same choke-point
pattern `screen._safe_pull_name` already uses for `pull_choice.PULL_NAME_RE`): `pull_choice.py`
and `views_install.py` import this table for their own dispatch, and `guided_install.py` imports
both this table and `pull_choice.py`, so this module cannot import either of them back at load
time without a cycle. Neither field the deferred imports build (`OLLAMA.command`, `OLLAMA.on_disk`)
is ever called before a caller already holds both modules loaded, so the deferral costs nothing at
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from . import guided_vllm
from .config import Configuration, MachineConfig
from .contracts import BaseModelSpec, Fit, Package
from .dialog import Choice
from .fit import compute_fit_v2
from .guided_loadtest import NO_CANDIDATE_LINE, NO_INSTALLED_LINE, installed_note, measure_ollama, ollama_offer, ollama_rows
from .intro import Fact, daemon_fact
from .measurements import Scenario
from .profile import HardwareProfile, RuntimeKey
from .room import ollama_models_candidates
from .vllm_server import VLLM_BASE_URL, LocalVllmServer

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun


@dataclass(frozen=True)
class RuntimeWords:
    """The words one runtime's rows and questions use.

    `package_prefix` and `key_text` are display words for a legend of a table with this runtime's
    rows (SV2-F/AP9-LS build that table; empty/`None` for Ollama, unused in AP9-SV2-L). They are
    **not** what tells two runtimes' pull-list values or collision keys apart -- that is the
    runtime's own stable `key` (Codex r1, fix 2), never a word two runtimes could happen to share.
    `question` is the one question the pull/install list asks once any row of this runtime is
    shown. `install_note` is the note an install line of this runtime adds below the command
    (`None` where it needs none).
    """

    package_prefix: str
    key_text: str | None
    question: str
    install_note: str | None = None


@dataclass(frozen=True)
class Runtime:
    """One row of `RUNTIMES`: every place that read "Ollama" or "gguf" hard-wired now reads one
    field of this instead (AP9-SV2-L, "Laufzeit als Eintrag" in the sub-package's brief).

    `server` is the SV1 `LocalServer` this runtime already has (start screen, step 4, the card);
    `detect` is how a live machine's profile learns whether this runtime is serving, installed or
    absent; `accepts` is which packages this runtime loads; `fit` judges one package against one
    measured machine for it; `command` is the name (or the reason there is none) that installs a
    package -- read by the pull list, the install line and, chosen, by the pull itself (Codex r1,
    fix 4); `model_place` is where its models live on disk; `on_disk` is its own "already there"
    check; `words` are the words above.
    """

    key: RuntimeKey
    name: str
    server: "LocalServer"
    detect: Callable[..., Any]
    hello_field: str
    accepts: Callable[[Package], bool]
    fit: Callable[[HardwareProfile, Package, BaseModelSpec, Scenario, MachineConfig], Fit]
    command: Callable[[Package], str | None]
    model_place: Callable[..., Sequence[Path]]
    on_disk: Callable[..., Any]
    words: RuntimeWords


def _not_built_yet(field_name: str, sub_package: str) -> Callable[..., Any]:
    """A named stand-in for a vLLM field this sub-package does not build.

    `VLLM.accepts` refuses every package here, so none of these ever run in AP9-SV2-L; a call
    fails loudly (the sub-package that will fill the field in, named in the message) rather than
    guessing quietly.
    """

    def _placeholder(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(f"Runtime.{field_name} for vllm is built in {sub_package}, not AP9-SV2-L")

    return _placeholder


def _ollama_accepts(package: Package) -> bool:
    """Which packages Ollama loads: a GGUF build, from Hugging Face or the Ollama library, or any
    package the Ollama library itself packed -- unchanged (`fit.py`, `pull_choice.py`)."""
    return package.format == "gguf" or package.source == "ollama"


def _ollama_command(package: Package) -> str | None:
    """The name `ollama pull` would take, or `None` where the package has none.

    `pull_choice.py`/`views_install.py` import this table for their own dispatch, so the reverse
    import is deferred to call time here (module docstring).
    """
    from .pull_choice import package_pull_name

    return package_pull_name(package)


def _ollama_on_disk(*args: Any, **kwargs: Any) -> Any:
    """Whether the daemon's inventory provably holds a package: the digest rule of the load test,
    unchanged (`guided_install._holds`). Deferred to call time (module docstring); the shape here
    is `guided_install._holds`'s own -- AP9-SV2-I gives every runtime's `on_disk` its final,
    common shape."""
    from .guided_install import _holds

    return _holds(*args, **kwargs)


# --- the local servers a runtime carries (AP9-SV1, moved here from local_servers.py by Codex r1
# fix 1): `local_servers.py` used to build `LOCAL_SERVERS` and this table separately, so a third
# runtime added only here never reached the start screen, step 4 or the card. Building both the
# server plumbing and the table in the one module that owns it removes that gap and the import
# ring a split the other way round would need (`local_servers.py` stays a thin view: it derives
# `LOCAL_SERVERS` from `RUNTIMES` below and keeps only the generic, server-list-shaped machinery
# -- `start_rows`, `load_tests`, `card_facts` -- that has nothing runtime-specific to move).

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


OLLAMA = Runtime(
    key="ollama",
    name="Ollama",
    server=LocalServer(
        key="ollama",
        name="Ollama",
        origin="http://127.0.0.1:11434",
        transport=lambda run: run.daemon,
        start_fact=_ollama_start,
        offer=_ollama_offer,
        measure=_ollama_measure,
        card_facts=lambda run: (),
    ),
    detect=daemon_fact,
    hello_field="ollama",
    accepts=_ollama_accepts,
    fit=compute_fit_v2,
    command=_ollama_command,
    model_place=ollama_models_candidates,
    on_disk=_ollama_on_disk,
    words=RuntimeWords(package_prefix="", key_text=None, question="Which package should be pulled into Ollama?"),
)

VLLM = Runtime(
    key="vllm",
    name="vLLM",
    server=LocalServer(
        key="vllm",
        name="vLLM",
        origin=VLLM_BASE_URL,
        transport=lambda run: run.servers.transports.get("vllm"),
        start_fact=guided_vllm.start_fact,
        offer=_vllm_offer,
        measure=_vllm_measure,
        card_facts=guided_vllm.card_facts,
    ),
    detect=_not_built_yet("detect", "AP9-SV2-I/AP9-SV2-R"),
    hello_field="vllm",
    # No Tensor package claims a row before AP9-SV2-P builds the Tensor package class and
    # AP9-SV2-F builds vLLM's fit (the Paar-Regel's mechanism is built here; nothing exercises it
    # for real until then -- "Laufzeit als Eintrag", AP9-SV2-L point 2).
    accepts=lambda package: False,
    fit=_not_built_yet("fit", "AP9-SV2-F"),
    command=_not_built_yet("command", "AP9-SV2-I"),
    model_place=_not_built_yet("model_place", "AP9-SV2-I"),
    on_disk=_not_built_yet("on_disk", "AP9-SV2-I"),
    words=RuntimeWords(
        package_prefix="vLLM · ",
        key_text="vLLM · rows start with vllm serve; the others are pulled into Ollama",
        question="Which package should be installed?",
    ),
)

RUNTIMES: tuple[Runtime, ...] = (OLLAMA, VLLM)

# The one place a local server is registered now (Codex r1, fix 1): every local-server reader
# (`local_servers.start_rows`, `.load_tests`, `.card_facts`, and every existing import of
# `local_servers.LOCAL_SERVERS`) goes on reading `local_servers.LOCAL_SERVERS`, which derives it
# from here, so a new `Runtime` reaches the start screen, step 4 and the card the moment it is a
# row of `RUNTIMES` -- no second place to add its server to.
LOCAL_SERVERS: tuple[LocalServer, ...] = tuple(runtime.server for runtime in RUNTIMES)


def accepting_runtimes(package: Package) -> tuple[Runtime, ...]:
    """Every runtime of `RUNTIMES` whose `accepts` claims `package`, in table order -- whatever the
    machine; `accepting_on_machine` below narrows this to one machine's own runtimes.

    Exactly one runtime today (`OLLAMA`, for a GGUF or Ollama-library package) or none (every
    other package: `VLLM.accepts` never claims one yet). Two or more is the Paar-Regel this
    function exists to carry once AP9-SV2-P/F/AP9-LS give a second runtime a real `accepts`.
    """
    return tuple(runtime for runtime in RUNTIMES if runtime.accepts(package))


def machine_runtime_keys(profile: HardwareProfile | None) -> frozenset[str]:
    """Which runtimes one machine's profile says it has: `profile.runtimes` (AP9-SV2-0) where it
    names any, else Ollama alone -- every profile before that field existed, and every machine
    whose last measurement recorded nothing there, is read as Ollama's own default (Codex r1, fix
    3: today's only shape, and the one every existing fixture and golden file already assumes).
    """
    if profile is not None and profile.runtimes:
        return frozenset(profile.runtimes)
    return frozenset({"ollama"})


def accepting_on_machine(package: Package, profile: HardwareProfile | None) -> tuple[Runtime, ...]:
    """`accepting_runtimes`, narrowed to the runtimes `profile`'s machine actually has (Codex r1,
    fix 3): a runtime that would accept `package` in general but that this machine never reported
    pairs with nothing here -- the pull list of a machine without it shows no row for it."""
    keys = machine_runtime_keys(profile)
    return tuple(runtime for runtime in accepting_runtimes(package) if runtime.key in keys)


def fit_for(
    profile: HardwareProfile, package: Package, base_model: BaseModelSpec, scenario: Scenario, machine_config: MachineConfig
) -> Fit:
    """The fit of the first runtime that accepts `package`, or fit contract v1 unchanged where
    none does -- every package a render judges today, GGUF or not, gets exactly one fit
    (CONTRACTS.md, "Fit contract v1"). Equivalent to calling `compute_fit_v2` unconditionally, as
    `render.py` did before AP9-SV2-L: `OLLAMA.fit` **is** `compute_fit_v2`, and a package no
    runtime accepts falls back to it here the same way `compute_fit_v2` already judges it
    `unknown` on its own (`fit._package_reason`). Not machine-scoped on purpose: the ranked table's
    pair rule (a second row's own fit) comes with SV2-F (AGENTS.md hand-off, kept as accepted).
    """
    accepting = accepting_runtimes(package)
    runtime = accepting[0] if accepting else OLLAMA
    return runtime.fit(profile, package, base_model, scenario, machine_config)


__all__ = [
    "LOCAL_SERVERS",
    "OLLAMA",
    "Offer",
    "Origins",
    "QUESTIONS",
    "RUNTIMES",
    "LocalServer",
    "Runtime",
    "RuntimeWords",
    "ServerRun",
    "VLLM",
    "accepting_on_machine",
    "accepting_runtimes",
    "fit_for",
    "machine_runtime_keys",
]
