"""The one table of local-server runtimes (AP9-SV2-L, decided 2026-09-29).

Everything that was hard-wired to "Ollama" -- package format, fit, install name and command, model
folder, the "already there" check, the words a row or question uses, and the server itself (start
screen, step 4, the card) -- hangs from one row of `RUNTIMES`, the **single** place a runtime or a
local server is registered: `local_servers.py` reads its servers off this table at call time
(`local_servers()`), so a new row reaches the start screen, step 4 and the card with no second
place to add it to (CONTRACTS.md, "Local servers and the vLLM load test";
docs/adr/0014-runtime-entries-and-vllm-packages.md). A later runtime (vLLM's own fields first,
then LM Studio) is a new row, never a plug-in: no dynamic loading, no registration at runtime.

Ollama sits on this table without a behavior change: `OLLAMA.accepts`, `.fit`, `.install_name`,
`.command`, `.model_place` and `.on_disk` are the functions that did the work before, referenced
here, not rewritten, and every caller reads them through this table (`pull_choice.py`,
`views_install.py`, `guided_install.py`). `VLLM` carries only what AP9-SV1 built -- its `server`
and `hello_field` -- and `accepts = lambda package: False`, so it claims no package until the
Tensor package class and its fit exist. Its other fields are named stand-ins that raise if ever
called; `test_every_runtime_has_every_field` only checks that every field is set.

`Runtime.detect`, `.model_place` and `.on_disk` are typed loosely on purpose: the probe and state
types of the finished table do not exist yet. `install_name` and `command` are two fields on
purpose: the name is what the pull list's value and a collision key are built from
(`pull_choice.runtime_value`), the command is the line a reader copies (`ollama pull <name>`).

Two imports are deferred to call time, inside the functions that need them (the pattern
`screen._safe_pull_name` already uses for `pull_choice.PULL_NAME_RE`): `pull_choice.py` and
`views_install.py` import this table, and `guided_install.py` imports both this table and
`pull_choice.py`, so this module cannot import either of them back at load time without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from . import guided_vllm
from .config import Configuration, MachineConfig
from .contracts import BaseModelSpec, Fit, Package
from .dialog import Choice
from .fit import compute_fit_v2
from .fit_vllm import vllm_fit_as_fit
from .guided_loadtest import NO_CANDIDATE_LINE, NO_INSTALLED_LINE, installed_note, measure_ollama, ollama_offer, ollama_rows
from .intro import Fact, daemon_fact
from .measurements import Scenario
from .profile import HardwareProfile, RuntimeKey
from .room import ollama_models_candidates
from .screen import pull_command
from .vllm_server import VLLM_BASE_URL, ServedModel

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun


@dataclass(frozen=True)
class RuntimeWords:
    """The words one runtime's rows and questions use.

    `package_prefix` and `key_text` are display words for a legend of a table with this runtime's
    rows (empty/`None` for Ollama, not read yet). They are **not** what tells two runtimes'
    pull-list values or collision keys apart -- that is the runtime's own stable `key`, never a
    word two runtimes could happen to share.
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
    """One row of `RUNTIMES`: every place that read "Ollama" or "gguf" hard-wired reads one field
    of this instead.

    `server` is the SV1 `LocalServer` this runtime already has (start screen, step 4, the card);
    `detect` is how a live machine's profile learns whether this runtime is serving, installed or
    absent; `accepts` is which packages this runtime loads; `fit` judges one package against one
    measured machine for it; `install_name` is the name this runtime installs a package by, or
    `None` where it has none -- the base of a pull-list value and a collision key; `command` turns
    that name into the line a reader copies (the install line, the command above the pull
    question); `model_place` is where its models live on disk; `on_disk(target, packages)` is its
    own "already there" check, the identities of those `packages` it provably holds; `words` are
    the words above.

    `shown_only_with` names the packages that are this runtime's alone to serve, so an output
    shows them only for a machine that has it (F3, `visible_packages`). It is not `accepts`:
    `accepts` says what a runtime loads and pairs a package with rows, and a package two runtimes
    load is still everyone's to see. vLLM claims the Hugging Face tensor package, Ollama every
    other one (F5): a machine whose profile names vLLM alone shows no Ollama row at all, while a
    profile without the field still reads as Ollama's (`machine_runtime_keys`) and shows them all.
    """

    key: RuntimeKey
    name: str
    server: "LocalServer"
    detect: Callable[..., Any]
    hello_field: str
    accepts: Callable[[Package], bool]
    fit: Callable[[HardwareProfile, Package, BaseModelSpec, Scenario, MachineConfig], Fit]
    install_name: Callable[[Package], str | None]
    command: Callable[[str], str]
    model_place: Callable[..., Sequence[Path]]
    on_disk: Callable[[Any, Sequence[Package]], frozenset]
    words: RuntimeWords
    shown_only_with: Callable[[Package], bool] = lambda package: False


def _not_built_yet(field_name: str, sub_package: str) -> Callable[..., Any]:
    """A named stand-in for a vLLM field that is not built yet.

    `VLLM.accepts` refuses every package, so none of these ever runs; a call fails loudly (the
    package that will fill the field in, named in the message) rather than guessing quietly.
    """

    def _placeholder(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(f"Runtime.{field_name} for vllm is built in {sub_package}")

    return _placeholder


def _ollama_accepts(package: Package) -> bool:
    """Which packages Ollama loads: a GGUF build, from Hugging Face or the Ollama library, or any
    package the Ollama library itself packed -- unchanged (`fit.py`, `pull_choice.py`)."""
    return package.format == "gguf" or package.source == "ollama"


def _huggingface_tensor(package: Package) -> bool:
    """A Hugging Face tensor (safetensors) package: what only vLLM serves among the table's rows."""
    return package.source == "huggingface" and package.format == "tensor"


def _not_huggingface_tensor(package: Package) -> bool:
    """Every package but a Hugging Face tensor one: Ollama's to show (F5, `Runtime.shown_only_with`)."""
    return not _huggingface_tensor(package)


def _ollama_install_name(package: Package) -> str | None:
    """The name `ollama pull` takes, or `None` where the package has none (`pull_choice.package_pull_name`).

    Deferred to call time: `pull_choice.py` imports this table (module docstring).
    """
    from .pull_choice import package_pull_name

    return package_pull_name(package)


def _ollama_on_disk(target: Any, packages: Sequence[Package]) -> frozenset:
    """The identities of those `packages` the daemon provably holds: the digest rule of the load
    test (`guided_install.ollama_held`). Deferred to call time (module docstring)."""
    from .guided_install import ollama_held

    return ollama_held(target, packages)


# --- the local servers a runtime carries (AP9-SV1): the server plumbing lives next to the table
# that registers it, so `local_servers.py` imports both from here without an import ring and keeps
# only the generic, list-shaped machinery (`start_rows`, `load_tests`, `card_facts`).

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
    install_name=_ollama_install_name,
    command=pull_command,
    model_place=ollama_models_candidates,
    on_disk=_ollama_on_disk,
    words=RuntimeWords(package_prefix="", key_text=None, question="Which package should be pulled into Ollama?"),
    shown_only_with=_not_huggingface_tensor,
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
    # AP9-SV2-F: the Tensor package class and vLLM's fit now exist, so vLLM claims exactly the
    # Hugging Face tensor packages SV2-P's search finds -- the same predicate `shown_only_with`
    # below already used for visibility (`_huggingface_tensor`), one definition for both questions.
    # GGUF over vLLM stays unoffered (experimental, out of scope by design): `_huggingface_tensor`
    # never claims `format == "gguf"`, whatever the package's source.
    accepts=_huggingface_tensor,
    fit=vllm_fit_as_fit,
    # `install_name` returns `None` (a genuine, documented value of its own type: "no safe pull
    # name") rather than `_not_built_yet`'s loud failure -- unlike every other not-yet field, this
    # one is now reachable the moment `accepts` claims a real package: `pull_choice.pull_collisions`
    # and `.ambiguous_names` call `runtime.install_name(package)` over *every* package a runtime of
    # `RUNTIMES` accepts, independent of which machine (or whether any machine at all) has vLLM --
    # the comment at `pull_choice.py:93` ("today only Ollama ever accepts a package") no longer
    # holds once this row's `accepts` is real. SV2-I gives this its own pull name and the `vllm
    # serve` install line; until then a vLLM row is simply never offered to pull (deliberately out
    # of scope by design: modelroom does not start vLLM itself).
    install_name=lambda package: None,
    command=_not_built_yet("command", "AP9-SV2-I"),
    model_place=_not_built_yet("model_place", "AP9-SV2-I"),
    on_disk=_not_built_yet("on_disk", "AP9-SV2-I"),
    # F5: no prefix and no key line -- a vLLM row stands in a section or on a machine that names
    # its runtime already, and no start command is promised before AP9-SV2-I builds one.
    words=RuntimeWords(package_prefix="", key_text=None, question="Which package should be installed?"),
    shown_only_with=_huggingface_tensor,
)

RUNTIMES: tuple[Runtime, ...] = (OLLAMA, VLLM)

# The servers of the table as it was loaded, for the imports that name the list; the readers of
# the guided run (`local_servers.start_rows`, `.load_tests`, `.card_facts`) call `local_servers()`
# instead, so they always read the table as it is now.
LOCAL_SERVERS: tuple[LocalServer, ...] = tuple(runtime.server for runtime in RUNTIMES)


def local_servers() -> tuple[LocalServer, ...]:
    """The server of every row of `RUNTIMES`, in table order: a new row reaches the start screen,
    step 4 and the card the moment it is in the table."""
    return tuple(runtime.server for runtime in RUNTIMES)


def runtime_of(key: str) -> Runtime:
    """The row of `RUNTIMES` with this `key` (a `PullRow.runtime`); `OLLAMA` for a key no row has."""
    return next((runtime for runtime in RUNTIMES if runtime.key == key), OLLAMA)


def accepting_runtimes(package: Package) -> tuple[Runtime, ...]:
    """Every runtime of `RUNTIMES` whose `accepts` claims `package`, in table order -- whatever the
    machine; `accepting_on_machine` below narrows this to one machine's own runtimes.

    Exactly one runtime today (`OLLAMA`, for a GGUF or Ollama-library package) or none (every
    other package: `VLLM.accepts` never claims one yet). Two or more is a package that pairs with
    more than one runtime, each pair a row of its own.
    """
    return tuple(runtime for runtime in RUNTIMES if runtime.accepts(package))


def present_runtimes(profile: HardwareProfile | None, live: object) -> frozenset[str]:
    """The runtimes one machine has in *this* run (P1: "`runtimes`, or detected live") -- the one
    function every reader of a machine's runtime set calls (Codex r7): which packages it shows
    (`visible_packages`), its head and empty state (`views`), step 1's `runtime` line.

    `profile.runtimes` (AP9-SV2-0) where it names any, else Ollama alone -- every profile before
    that field existed, and every machine whose last measurement recorded nothing there. `live` is
    whatever a caller's own step-1 bookkeeping holds for `"vllm"` on that machine
    (`GuidedRun.servers.found.get("vllm")` for the machine the run is on, nothing for any other):
    only a real `ServedModel` answer counts as detected -- a server that asked for a key, one whose
    answer could not be read, or nothing at all, does not (E5: "does not count as detected").
    """
    keys = frozenset(profile.runtimes) if profile is not None and profile.runtimes else frozenset({"ollama"})
    return keys | {"vllm"} if isinstance(live, ServedModel) else keys


def machine_runtime_keys(profile: HardwareProfile | None) -> frozenset[str]:
    """`present_runtimes` without a live reading: no logic of its own (Codex r7), for a reader that
    has none -- a machine of `modelroom render`, or a question live detection cannot change (it only
    ever adds vLLM, so Ollama's membership is the same either way)."""
    return present_runtimes(profile, None)


def machine_has_vllm(profile: HardwareProfile | None, live: object) -> bool:
    """Whether vLLM is present on one machine in this run -- what every vLLM gate reads:
    `any_has_vllm` (the tensor search and `build_tensor` of a guided run)."""
    return "vllm" in present_runtimes(profile, live)


# `machine_has_vllm`'s two inputs for one machine: its profile, and what was detected live on it.
MachineRuntimes = tuple[HardwareProfile | None, object]


def any_has_vllm(machines: Sequence[MachineRuntimes]) -> bool:
    """Whether at least one of `machines` has vLLM (`machine_has_vllm` for each)."""
    return any(machine_has_vllm(profile, live) for profile, live in machines)


def visible_packages(packages: Sequence[Package], machines: Sequence[MachineRuntimes]) -> list[Package]:
    """The one visibility rule (F3, F5): the `packages` an output for `machines` may show or count.

    Read from the table, never from a package's format: a package one or more rows claim
    (`Runtime.shown_only_with`) is visible only when at least one of `machines` has one of those
    runtimes -- a Hugging Face tensor package needs vLLM, every other package (GGUF, the Ollama
    library) needs Ollama, which every profile without the `runtimes` field has
    (`machine_runtime_keys`). No machines at all reads as such a profile: Ollama's alone. One
    machine for its own table or scale, every shown machine for a document-wide number or summary.
    The snapshot itself is never filtered: it keeps the raw facts.
    """
    present = frozenset().union(*(present_runtimes(profile, live) for profile, live in machines))
    present = present if machines else machine_runtime_keys(None)
    return [package for package in packages if _shown_with(package, present)]


def _shown_with(package: Package, present: frozenset[str]) -> bool:
    claiming = [runtime.key for runtime in RUNTIMES if runtime.shown_only_with(package)]
    return not claiming or any(key in present for key in claiming)


def runtimes_among(fits: Iterable[Fit]) -> tuple[str, ...]:
    """The runtimes whose rows `fits` are, in table order: the one answer to "does this machine
    show the rows of two runtimes" (F5, Codex r6-1). Two or more means one section per runtime,
    each ranked on its own and every one after the first whole (`render._rankings_by_runtime`), and
    the rule says so (`render.build_render_document`); the views split by it too
    (`views_sections.runtime_sections`). One or none: one block, as before."""
    keys = {fit.runtime for fit in fits}
    return tuple(runtime.key for runtime in RUNTIMES if runtime.key in keys)


def accepting_on_machine(package: Package, profile: HardwareProfile | None) -> tuple[Runtime, ...]:
    """`accepting_runtimes`, narrowed to the runtimes `profile`'s machine actually has: a runtime
    that would accept `package` in general but that this machine never reported pairs with nothing
    here. Empty for a package no runtime of the machine accepts -- such a package gets no pull row
    and no install line (`pull_choice.pull_rows`, `views_install.install_choice`)."""
    keys = machine_runtime_keys(profile)
    return tuple(runtime for runtime in accepting_runtimes(package) if runtime.key in keys)


def fit_for(
    profile: HardwareProfile, package: Package, base_model: BaseModelSpec, scenario: Scenario, machine_config: MachineConfig
) -> Fit:
    """The fit of the first runtime that accepts `package`, or fit contract v1 unchanged where
    none does -- every package a render judges today, GGUF or not, gets exactly one fit
    (CONTRACTS.md, "Fit contract v1"). `OLLAMA.fit` **is** `compute_fit_v2` (equivalent to calling
    it unconditionally, as `render.py` did before AP9-SV2-L); `VLLM.fit` **is**
    `fit_vllm.vllm_fit_as_fit` (AP9-SV2-F) for the Hugging Face tensor packages `VLLM.accepts`
    claims. A package no runtime accepts falls back to `OLLAMA.fit` here the same way
    `compute_fit_v2` already judges it `unknown` on its own (`fit._package_reason`). Not
    machine-scoped, and one fit per package: a second row of the ranked table for a second
    runtime, with its own fit, comes with `Fit.runtime`. The one caller outside `render.py` is
    `guided_context.fit_texts` (AP9-SV2-F), the step-3 scale, through `fit.count_fitting`'s
    `fit_function` parameter -- so the scale and the finished ranking never disagree about which
    runtime judges a package.
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
    "MachineRuntimes",
    "accepting_on_machine",
    "accepting_runtimes",
    "any_has_vllm",
    "fit_for",
    "local_servers",
    "machine_has_vllm",
    "machine_runtime_keys",
    "present_runtimes",
    "runtime_of",
    "runtimes_among",
    "visible_packages",
]
