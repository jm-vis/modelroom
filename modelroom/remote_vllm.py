"""vLLM at a server over SSH, the laptop's side (AP9-SV2-R, R2): what is sent, how an answer is checked,
and the words this computer says for it.

`modelroom remote vllm` and `vllm-measure` read and measure the server's **own** vLLM
(`127.0.0.1:8000` there, the three calls of `vllm_server.py`); nothing here opens a connection to
that port, forwards one, or starts a program. What crosses the wire is small on purpose:

- to the server only a `PackageRef` per package (`package_ref`: repo, 40-hex revision and exactly the
  start files, `tensor_quant.start_files`) -- never a whole `Package` (`approval.by` is a person's name),
  never a scenario beyond the context and the requests;
- from the server only an index of the package it serves (never the model's own `id` or `root`, so no
  local path and no `--served-model-name` reaches this computer), the context, the bytes still to
  fetch per package (a number or `null`), and for a measurement numbers and one code of a fixed set.

Every name this computer shows is built from its own package identity (FR-B2); the sentence of a code
(`loadtest_vllm.PROBLEM_WORDS`, `ssh.ERROR_SENTENCES`) is its own. An answer that does not fit what was
asked -- a list of the wrong length, a number beyond the size of the package's start files -- is a form
fault (`<alias> answered in a form this modelroom does not read`), checked here before anything reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from . import remote_protocol
from .contracts import Package
from .loadtest_vllm import INVALID_CODES, PROBLEM_WORDS, MAX_REQUESTS, Reason, VllmMeasurement
from .measurements import Scenario
from .quantization import package_identity_key
from .remote_client import FORM_ERROR, _result_of
from .ssh import SshError, SshRunner, call
from .tensor_quant import start_files
from .vllm_server import ServedModel

KEY_NOTE = "the vLLM server on {alias} asks for an API key; modelroom reads an open server only"
NO_PACKAGE_NOTE = "the vLLM server on {alias} serves a model that is no package of this run"
SERVER_GONE = "vLLM on {alias} stopped answering during the measurement"


def package_ref(package: Package) -> dict | None:
    """The `PackageRef` of `package` (R2): `{"repo", "revision", "files": [{"name", "size_bytes"}, ...]}`
    with exactly its start files, built from checked values -- `None` where a value is not safe to send
    (not a Hugging Face package, a name outside the Hub's rule, a revision that is no 40 hex digits, a file
    name outside `SAFE_REPO_PATH_RE`, no start file)."""
    ref = {
        "repo": package.repo,
        "revision": package.revision,
        "files": [{"name": file.name, "size_bytes": file.size_bytes} for file in start_files(package.files)],
    }
    try:
        remote_protocol.VllmPackageRef.model_validate(ref)
    except ValueError:  # pydantic's ValidationError
        return None
    return ref


def sendable(packages: Sequence[Package]) -> list[Package]:
    """The packages of a run a server is asked about, in their order: the active, complete Hugging Face tensor
    packages whose `PackageRef` is safe -- the index a server answers with is a position in this list."""
    return [
        package
        for package in packages
        if package.active and package.complete and package.source == "huggingface" and package.format == "tensor"
        and package_ref(package) is not None
    ]


def download_bytes(package: Package) -> int:
    """The whole download of `package`: the sizes of its start files (the disk rule's reading, I2)."""
    return sum(file.size_bytes for file in start_files(package.files))


@dataclass
class RemoteVllm:
    """What one run knows of the vLLM at its active SSH target, in memory only (nothing of it is stored).

    `state` is `hello.vllm`, refined by the `vllm` operation where it was made; `packages` the packages
    that operation was sent, `served_index` the one it serves (an index into them, `None` for none),
    `missing_bytes` the bytes still to fetch for each, `relative` whether the server's Hugging Face cache
    is set as a relative path (`room.model_disk`, read in step 5), `outcome` what step 4 measured and
    `disk` the free disk step 5 last read at the server (`room.Disk`, vLLM's own place nested in it)."""

    state: str = "absent"
    packages: tuple[Package, ...] = ()
    served_index: int | None = None
    served_context: int | None = None
    missing_bytes: tuple[int | None, ...] = ()
    relative: bool | None = None
    outcome: object = None
    disk: object = None

    def live(self) -> object:
        """What the views take for "vLLM detected live at this machine": a `ServedModel` where it serves
        (its name is this computer's own package's, never the server's; an empty one where it serves no
        package of this run), `"installed"` where it is there and does not serve, else `None` -- a server that
        asks for a key does not count as detected (E5)."""
        if self.state == "serving":
            repo = self.packages[self.served_index].repo if self.served_index is not None else ""
            return ServedModel(id=repo, root=repo, max_model_len=self.served_context or 1)
        return "installed" if self.state == "installed" else None

    def needs(self) -> tuple[dict[tuple[str, str, str], int], frozenset[tuple[str, str, str]]]:
        """`(needs, held)` for the disk rule (I2, I3): per package identity the bytes still to fetch (`null`
        reads as the whole download), and the identities whose start files are all there (`0`)."""
        needs: dict[tuple[str, str, str], int] = {}
        held: set[tuple[str, str, str]] = set()
        for package, missing in zip(self.packages, self.missing_bytes):
            identity = package_identity_key(package)
            needs[identity] = missing if missing is not None else download_bytes(package)
            if missing == 0:
                held.add(identity)
        return needs, frozenset(held)


def read_state(alias: str, packages: Sequence[Package], runner: SshRunner, ssh_path: str | None = None) -> RemoteVllm:
    """`vllm` at `alias` for `packages` (already `sendable`): the state, which package it serves, and the
    bytes still to fetch per package -- checked against what was sent (R2, B1)."""
    sent = tuple(packages)
    request = {"packages": [package_ref(package) for package in sent]}
    result = _result_of(alias, call(alias, "vllm", request, runner, ssh_path=ssh_path))
    return _checked_state(alias, result, sent)


def _checked_state(alias: str, result: dict, sent: tuple[Package, ...]) -> RemoteVllm:
    """The answer against the request: as many entries as packages, each `null` or between `0` and the sum of
    that package's start files, and a served index only where the server serves and names a package sent."""
    missing = result["missing_bytes"]
    served = result["served"]
    fault = SshError(FORM_ERROR.format(alias=alias))
    if len(missing) != len(sent):
        raise fault
    for package, value in zip(sent, missing):
        if value is not None and not 0 <= value <= download_bytes(package):
            raise fault
    if served is not None and result["state"] != "serving":
        raise fault
    index = served["package"] if served is not None else None
    if index is not None and index >= len(sent):
        raise fault
    return RemoteVllm(
        state=result["state"], packages=sent, served_index=index,
        served_context=served["context"] if served is not None else None, missing_bytes=tuple(missing),
    )


def vllm_measure(
    alias: str,
    packages: Sequence[Package],
    index: int,
    scenario: Scenario,
    runner: SshRunner,
    ssh_path: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> VllmMeasurement:
    """`vllm-measure` at `alias` for `packages[index]`: one warm-up and three batches at the server, `on_progress`
    called with `{"stage", "number", "of"}` (never a name). The result as a `VllmMeasurement` in memory, its
    problem in this computer's own words; a server lost in the run is one `SshError`."""
    request = {
        "packages": [package_ref(package) for package in packages],
        "package": index,
        "scenario": {"context_requested": scenario.context_requested, "requests": scenario.requests},
    }
    answer = _result_of(alias, call(alias, "vllm_measure", request, runner, ssh_path=ssh_path, on_progress=on_progress))
    return to_measurement(alias, answer["measurement"], scenario)


def to_measurement(alias: str, wire: dict, scenario: Scenario) -> VllmMeasurement:
    """The `VllmMeasurement` of a server's numbers and code. The requests it states must be the ones asked
    for (`min(requests, 64)` measured, `ranking_requests` the scenario's), else a form fault; the sentence
    of the code is built here from the fields this computer knows."""
    if wire["ranking_requests"] != scenario.requests or wire["requests"] != min(scenario.requests, MAX_REQUESTS):
        raise SshError(FORM_ERROR.format(alias=alias))
    code = wire["problem"]
    if code == "server_gone":
        raise SshError(SERVER_GONE.format(alias=alias))
    reason = None
    if code is not None:
        words = PROBLEM_WORDS[code].format(
            context=wire["context"], requested=scenario.context_requested, requests=wire["requests"], ranking=wire["ranking_requests"]
        )
        reason = Reason(words, code)
    invalid = code in INVALID_CODES
    speeds = {name: wire[name] for name in ("total_tps", "total_min", "total_max", "request_tps", "request_min", "request_max")}
    return VllmMeasurement(
        requests=wire["requests"], ranking_requests=wire["ranking_requests"], context=wire["context"],
        validity="invalid" if invalid else "valid", validity_reason=reason if invalid else None,
        comparable=not (code is not None and not invalid), comparable_reason=reason if code is not None and not invalid else None,
        **speeds,
    )


__all__ = [
    "KEY_NOTE",
    "NO_PACKAGE_NOTE",
    "SERVER_GONE",
    "RemoteVllm",
    "download_bytes",
    "package_ref",
    "read_state",
    "sendable",
    "to_measurement",
    "vllm_measure",
]
