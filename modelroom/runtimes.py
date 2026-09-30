"""The one table of local-server runtimes: "Laufzeit als Eintrag" (AP9-SV2-L, decided 2026-09-29).

Everything that was hard-wired to "Ollama" -- package format, fit, pull command, model folder,
the "already there" check, the words a row or question uses -- hangs from one row of `RUNTIMES`
now. A later runtime (vLLM's own fields first, then LM Studio, AP9-LS) is a new row, never a
plug-in: no dynamic loading, no registration at runtime (CONTRACTS.md, "Local servers and the
vLLM load test"; docs/adr/0014-runtime-entries-and-vllm-packages.md).

AP9-SV2-L moves Ollama onto this table without a behavior change: `OLLAMA.accepts`, `.fit`,
`.command`, `.model_place` and `.on_disk` are today's functions, referenced here, not rewritten.
`VLLM` in this sub-package carries only what AP9-SV1 already built -- its `server` and
`hello_field` -- and `accepts = lambda package: False`, so it never claims a package before
AP9-SV2-P builds the Tensor package class and AP9-SV2-F builds its fit. Its other fields are
named stand-ins that raise if ever called, each one saying which later sub-package fills it in;
`test_every_runtime_has_every_field` only checks that every field is set, never that a vLLM one
runs.

`Runtime.detect`, `.command`, `.model_place` and `.on_disk` are typed loosely on purpose: the
brief that describes the finished table (`DetectProbes`, `RuntimeState`, `InstallCommand`,
`OnDiskProbe`) names types that do not exist yet -- SV2-I and SV2-R still have to design them.
SV2-L only wires today's functions in.

Two imports are deferred to call time, inside the functions that need them (the same choke-point
pattern `screen._safe_pull_name` already uses for `pull_choice.PULL_NAME_RE`): `pull_choice.py`
imports this table for its own dispatch (`accepting_runtimes`), so this module cannot import
`pull_choice` back at load time without a cycle; `guided_install.py` imports `pull_choice.py` at
its own top level, so the same is true of it. Neither field the deferred imports build
(`OLLAMA.command`, `OLLAMA.on_disk`) is ever called before a caller already holds both modules
loaded, so the deferral costs nothing at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from . import local_servers
from .config import MachineConfig
from .contracts import BaseModelSpec, Fit, Package
from .fit import compute_fit_v2
from .intro import daemon_fact
from .local_servers import LocalServer
from .measurements import Scenario
from .profile import HardwareProfile, RuntimeKey
from .room import ollama_models_candidates


@dataclass(frozen=True)
class RuntimeWords:
    """The words one runtime's rows and questions use.

    `package_prefix` goes in front of a pull-list or install-line name so two runtimes that both
    answer for the same package never share a value (empty for Ollama: the shape stays exactly
    what it is today). `key_text` is the legend line a table with this runtime's rows adds
    (`None` where a runtime never needs one, Ollama today). `question` is the one question the
    pull/install list asks once any row of this runtime is shown. `install_note` is the note an
    install line of this runtime adds below the command (`None` where it needs none).
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
    measured machine for it; `command` is the line (or the reason there is none) that installs a
    package; `model_place` is where its models live on disk; `on_disk` is its own "already there"
    check; `words` are the words above.
    """

    key: RuntimeKey
    name: str
    server: LocalServer
    detect: Callable[..., Any]
    hello_field: str
    accepts: Callable[[Package], bool]
    fit: Callable[[HardwareProfile, Package, BaseModelSpec, Scenario, MachineConfig], Fit]
    command: Callable[[Package], str | None]
    model_place: Callable[..., Sequence[Path]]
    on_disk: Callable[..., Any]
    words: RuntimeWords


def _not_built_yet(field: str, sub_package: str) -> Callable[..., Any]:
    """A named stand-in for a vLLM field this sub-package does not build.

    `VLLM.accepts` refuses every package here, so none of these ever run in AP9-SV2-L; a call
    fails loudly (the sub-package that will fill the field in, named in the message) rather than
    guessing quietly.
    """

    def _placeholder(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(f"Runtime.{field} for vllm is built in {sub_package}, not AP9-SV2-L")

    return _placeholder


def _ollama_accepts(package: Package) -> bool:
    """Which packages Ollama loads: a GGUF build, from Hugging Face or the Ollama library, or any
    package the Ollama library itself packed -- unchanged (`fit.py`, `pull_choice.py`)."""
    return package.format == "gguf" or package.source == "ollama"


def _ollama_command(package: Package) -> str | None:
    """The name `ollama pull` would take, or `None` where the package has none.

    `pull_choice.py` imports this table for its own dispatch, so the reverse import is deferred to
    call time here (module docstring).
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


_SERVERS_BY_KEY = {server.key: server for server in local_servers.LOCAL_SERVERS}

OLLAMA = Runtime(
    key="ollama",
    name="Ollama",
    server=_SERVERS_BY_KEY["ollama"],
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
    server=_SERVERS_BY_KEY["vllm"],
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


def accepting_runtimes(package: Package) -> tuple[Runtime, ...]:
    """Every runtime of `RUNTIMES` whose `accepts` claims `package`, in table order.

    Exactly one runtime today (`OLLAMA`, for a GGUF or Ollama-library package) or none (every
    other package: `VLLM.accepts` never claims one yet). Two or more is the Paar-Regel this
    function exists to carry once AP9-SV2-P/F/AP9-LS give a second runtime a real `accepts`.
    """
    return tuple(runtime for runtime in RUNTIMES if runtime.accepts(package))


def fit_for(
    profile: HardwareProfile, package: Package, base_model: BaseModelSpec, scenario: Scenario, machine_config: MachineConfig
) -> Fit:
    """The fit of the first runtime that accepts `package`, or fit contract v1 unchanged where
    none does -- every package a render judges today, GGUF or not, gets exactly one fit
    (CONTRACTS.md, "Fit contract v1"). Equivalent to calling `compute_fit_v2` unconditionally, as
    `render.py` did before AP9-SV2-L: `OLLAMA.fit` **is** `compute_fit_v2`, and a package no
    runtime accepts falls back to it here the same way `compute_fit_v2` already judges it
    `unknown` on its own (`fit._package_reason`).
    """
    accepting = accepting_runtimes(package)
    runtime = accepting[0] if accepting else OLLAMA
    return runtime.fit(profile, package, base_model, scenario, machine_config)


__all__ = [
    "OLLAMA",
    "VLLM",
    "RUNTIMES",
    "Runtime",
    "RuntimeWords",
    "accepting_runtimes",
    "fit_for",
]
