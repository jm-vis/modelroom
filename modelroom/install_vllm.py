"""The install lines of a vLLM row: two commands to paste, nothing started (AP9-SV2-I, I1).

modelroom starts no vLLM and pulls nothing for it (GF 2026-09-29, F1 a, "Not here"); it prints the
command and the reader runs it in a terminal that stays open. For one row `vllm_install` builds
**two** lines (F2 b) from the very same checked values: `vllm serve ...` and the same server as one
`docker run` line. Both are plain POSIX shell text, for the measured Linux machine only: the Docker
line is output, never a call -- nothing in this module starts a process or looks for `docker`.

Every value that reaches a line is checked here and nowhere else: the repo against the Hub's name rule
(and ASCII), the revision against 40 hex digits, the context, the card count and the card numbers as
whole numbers from the fit, never text; the image, the Docker socket, the cache expression and the
container path are constants. A value that fails gives no line and the lock `no safe command`.

The cards are bound by number, always, even when the line names all of them: `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<index>,...` with the `index` values of exactly the cards the fit chose
(`fit_vllm.chosen_cards`), so a mask the shell already holds cannot shrink the line and CUDA counts as
`nvidia-smi` does. A card whose memory was not read by `nvidia-smi` has no CUDA number modelroom can
name: no line then, the row stays in the ranking with its fit (`computed only`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import Fit, Package, _HF_REPO_RE
from .fit_vllm import VLLM_GPU_MEMORY_UTILIZATION, chosen_cards, equal_size_group
from .hf_cache import HF_CACHE_EXPRESSION
from .intro import label_line
from .profile import HardwareProfile
from .vllm_server import ServedModel

# Annahme A8: the image of the SV1 recordings and of the calculation (`vllm_*_l4.*`, 0.92, 1.50 GiB);
# a fixed tag, because `latest` moves without notice and would break "calculated so, started so".
# Source: the vLLM Docker documentation (docs.vllm.ai, "Using Docker"), image `vllm/vllm-openai`.
# A change is this one line and a hand test.
VLLM_IMAGE = "vllm/vllm-openai:v0.30.0"
# `--host` wins over `DOCKER_HOST` and a set `DOCKER_CONTEXT`: the container runs on the Docker of the
# machine whose shell the line is pasted into, the measured one. Rootless Docker uses another socket.
DOCKER_LOCAL_HOST = "unix:///var/run/docker.sock"
CONTAINER_CACHE = "/root/.cache/huggingface/hub"

NO_SAFE_COMMAND = "no safe command"
COMPUTED_ONLY = "computed only: modelroom cannot name these cards to vLLM"
RELATIVE_CACHE_NOTE = (
    "docker: left out, the Hugging Face cache is set as a relative path; set it absolute to get this line"
)
LARGER_HINT = "larger --tensor-parallel-size spreads the load over more cards, if the server is yours alone"
STOP_LINE = "vLLM serves another model on port 8000 now; stop it first (one model per server)"
_START_LATER = "vLLM downloads the model at its first start"
_AGAIN = "then run modelroom again: step 4 finds the server and measures it"
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_NVIDIA_SMI = "nvidia-smi"

Line = list[tuple[str, str]]


@dataclass(frozen=True)
class VllmInstall:
    """What one vLLM row prints: the `vllm serve` line and the Docker line, or why there is none.

    `docker` is `None` with `docker_note` where only the Docker line is left out (a relative cache
    variable); with `serve` also `None`, `reason` is the lock (`no safe command`, `computed only: ...`).
    `larger`: the model allows a `--tensor-parallel-size` above the one named (`Fit.larger_size`; rule (b)
    takes the smallest `t` that fits), so a note says a larger value spreads the load further.
    """

    serve: str | None
    docker: str | None = None
    docker_note: str | None = None
    reason: str | None = None
    larger: bool = False


def safe_repo(repo: object) -> bool:
    """Whether `repo` is a Hub repository name a command line may carry: the Hub's name rule, ASCII only."""
    return isinstance(repo, str) and repo.isascii() and bool(_HF_REPO_RE.fullmatch(repo))


def safe_revision(revision: object) -> bool:
    """Whether `revision` is 40 lower-case hex digits (`fullmatch`: a trailing line break is no match)."""
    return isinstance(revision, str) and bool(_REVISION_RE.fullmatch(revision))


def install_name(package: Package) -> str | None:
    """The name a vLLM row is chosen by: its repo, where repo and revision both are safe to put in a command."""
    return package.repo if safe_repo(package.repo) and safe_revision(package.revision) else None


def _whole(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def vllm_install(
    package: Package | None, fit: Fit, profile: HardwareProfile | None, *, relative_cache: bool = False
) -> VllmInstall:
    """The two lines for `package` as `fit` judged it on `profile`'s cards.

    `relative_cache`: a cache variable of the machine is relative (`hf_cache.cache_is_relative`), so only
    the Docker line goes, with its note. Nothing here reads a file or the machine.
    """
    refused = VllmInstall(None, reason=NO_SAFE_COMMAND)
    if package is None or install_name(package) is None or profile is None:
        return refused
    if not _whole(fit.context, 1) or not _whole(fit.cards, 1):
        return refused
    cards = chosen_cards(equal_size_group(profile), fit.cards)
    if len(cards) != fit.cards or not all(_whole(card.index, 0) for card in cards):
        return refused
    if any(card.vram_source != _NVIDIA_SMI for card in cards):
        return VllmInstall(None, reason=COMPUTED_ONLY)
    larger = fit.larger_size  # a larger `t` the model allows (`fit_vllm.allowed_sizes`), not just more cards
    numbers = ",".join(str(card.index) for card in cards)
    common = (
        f"{package.repo} --revision {package.revision} --max-model-len {fit.context} "
        f"--tensor-parallel-size {fit.cards} --gpu-memory-utilization {VLLM_GPU_MEMORY_UTILIZATION}"
    )
    serve = (
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={numbers} VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 "
        f"vllm serve {common} --host 127.0.0.1 --port 8000"
    )
    if relative_cache:
        return VllmInstall(serve, None, RELATIVE_CACHE_NOTE, larger=larger)
    docker = (
        f'mkdir -p "{HF_CACHE_EXPRESSION}" && docker --host {DOCKER_LOCAL_HOST} run --rm --ipc=host '
        f"--gpus '\"device={numbers}\"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 "
        f'-p 127.0.0.1:8000:8000 --mount "type=bind,src={HF_CACHE_EXPRESSION},dst={CONTAINER_CACHE}" '
        f"{VLLM_IMAGE} {common} --host 0.0.0.0 --port 8000"
    )
    return VllmInstall(serve, docker, larger=larger)


def install_block_lines(
    label: str, rank: int, install: VllmInstall, *, repo: str | None = None, served: object = None,
    on_disk: bool = False, where: str = "",
) -> list[Line]:
    """The lines of one vLLM install, under `label` (`install`): the `vllm serve` line, the Docker line
    (or why it is left out), then what to do -- the one block of the table, of the choice in step 5 and,
    with `where` naming another computer, of the handover.

    `served`: what the local server states it serves (`vllm_server.ServedModel`); another model than
    `repo` adds the stop line. `on_disk`: the package is on disk already, so the download clause goes.
    Without a command (`serve` is `None`) the one line says why.
    """
    if install.serve is None:
        return [label_line(label, [("class:note", f"#{rank} {install.reason or NO_SAFE_COMMAND}")])]
    place = f"#{rank} on {where}" if where else f"#{rank}"
    lines: list[Line] = [label_line(label, [("class:note", place), ("", "  "), ("class:command", install.serve)])]
    if install.docker is not None:
        # Under the command above it, which starts four columns behind the label: `#2  `.
        lines.append([("", "    "), ("class:note", "docker"), ("", "    "), ("class:command", install.docker)])
    elif install.docker_note is not None:
        lines.append(label_line("", [("class:note", install.docker_note)]))
    notes: list[str] = [LARGER_HINT] if install.larger else []
    if isinstance(served, ServedModel) and served.root != repo and not where:
        notes.append(STOP_LINE)
    both = "one of the two" if install.docker is not None else "it"
    terminal = f"{'on that computer, ' if where else ''}in a terminal that stays open"
    if on_disk:
        notes += [f"#{rank} is on disk already (size checked)", f"run {both} {terminal}"]
    else:
        notes.append(f"run {both} {terminal}; {_START_LATER}")
    if not where:
        notes.append(_AGAIN)
    return lines + [label_line("", [("class:note", text)]) for text in notes]


__all__ = [
    "COMPUTED_ONLY",
    "CONTAINER_CACHE",
    "DOCKER_LOCAL_HOST",
    "LARGER_HINT",
    "NO_SAFE_COMMAND",
    "RELATIVE_CACHE_NOTE",
    "STOP_LINE",
    "VLLM_IMAGE",
    "VllmInstall",
    "install_block_lines",
    "install_name",
    "safe_repo",
    "safe_revision",
    "vllm_install",
]
