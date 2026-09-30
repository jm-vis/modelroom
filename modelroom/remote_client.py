"""The laptop side of the remote protocol (AP9-FR, FR-B1, B2): `hello`/`profile`/`room`, and
`SshDaemon`, which makes a computer read over SSH answer to the unchanged `Daemon` protocol.

`SshDaemon` **translates** the wire envelope into exactly the shapes `guided_install.py`'s
`_send`/`_check_line`/`_end_reason`/`pulled_problem` already expect (Codex r2-6): nothing there
changes for a pull over SSH, only where the bytes come from -- a pull's progress reaches that
callback line by line, streamed by `ssh.call`'s own `on_progress`, never collected first (Codex r1
finding 1). `hello`/`profile`/`room` are the three read-only calls `guided_remote.py` (B3-B5) makes
directly; an `error` line of the server is already one fixed sentence per code by the time it
reaches this module (`ssh.ERROR_SENTENCES`, raised by `ssh.call` as `SshError`).

`ssh_path` is `None` by default everywhere in this module (Codex r1 finding 5): `ssh.call`'s own
`which` resolves it, so a missing `ssh` binary is always the fixed `NO_SSH` sentence, never an
uncaught process-start error from a hard-coded `"ssh"` that bypassed the resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from .contracts import Package
from .daemon import DaemonError, GENERATE_PATH, PULL_PATH, Progress
from .http import Response
from .measurements import Scenario
from .ssh import SshError, SshRunner, call

FORM_ERROR = "{alias} answered in a form this modelroom does not read"
AGENT_FORWARDED = "your SSH agent reached {alias} although modelroom switches forwarding off; check your SSH configuration"
OLLAMA_UNREACHABLE = "Ollama does not answer on {alias}"


def _result_of(alias: str, envelope: dict) -> dict:
    if "result" not in envelope:
        raise SshError(FORM_ERROR.format(alias=alias))
    return envelope["result"]


def hello(alias: str, runner: SshRunner, ssh_path: str | None = None) -> dict:
    """`{version, system, ollama, agent_forwarded, os_fingerprint}` -- and the agent check (B2):
    a forwarded agent stops the run here, before anything else of this alias is read."""
    result = _result_of(alias, call(alias, "hello", {}, runner, ssh_path=ssh_path))
    if result.get("agent_forwarded") is True:
        raise SshError(AGENT_FORWARDED.format(alias=alias))
    if result.get("ollama") == "unreachable":
        raise SshError(OLLAMA_UNREACHABLE.format(alias=alias))
    return result


def profile(alias: str, profile_id: str, machine_class: str | None, runner: SshRunner, ssh_path: str | None = None) -> dict:
    """`{"export": <ExportObject v1>}` of `profile_id`, measured now on `alias`."""
    request = {"profile_id": profile_id, "machine_class": machine_class}
    return _result_of(alias, call(alias, "profile", request, runner, ssh_path=ssh_path))


def room(alias: str, cards, runner: SshRunner, ssh_path: str | None = None) -> dict:
    """`{"disk": {...}, "graphics_free_mib": [...] | None}` of `alias`, read now, no path, no drive."""
    request = {"cards": [[index, name] for index, name in cards]}
    return _result_of(alias, call(alias, "room", request, runner, ssh_path=ssh_path))


def candidates(alias: str, packages: Sequence[Package], runner: SshRunner, ssh_path: str | None = None) -> dict:
    """`{"candidates": [{"package_identity": [...]}, ...], "set_aside": int}` of `alias`'s own
    installed packages (B2, stage 1's digest rule run at the server) -- never an installed name."""
    request = {"packages": [package.model_dump(mode="json") for package in packages]}
    return _result_of(alias, call(alias, "candidates", request, runner, ssh_path=ssh_path))


def measure(
    alias: str,
    package_identity: tuple[str, str, str],
    packages: Sequence[Package],
    scenario: Scenario,
    profile_id: str,
    runner: SshRunner,
    ssh_path: str | None = None,
    on_progress=None,
) -> dict:
    """`{"record": <MeasurementRecord>}` of `alias`'s own measurement of `package_identity` under
    `profile_id` (B2); `on_progress` is called with `{"stage", "number", "of"}`, never a name."""
    request = {
        "package_identity": list(package_identity),
        "packages": [package.model_dump(mode="json") for package in packages],
        "scenario": scenario.model_dump(mode="json"),
        "profile_id": profile_id,
    }
    return _result_of(alias, call(alias, "measure", request, runner, ssh_path=ssh_path, on_progress=on_progress))


@dataclass
class SshDaemon:
    """A `Daemon` (`daemon.py`'s protocol) whose calls travel to `alias`'s own Ollama over SSH.

    `installed_models`, `_is_local`, `_send`, `_check_line`, `_end_reason`, `pulled_problem`
    (`guided_install.py`) run against this exactly as they run against `LocalDaemon` -- nothing
    there was changed for FR-B1 (Codex r2-6). Every `ssh.SshError` this raises is translated to a
    `DaemonError` at the boundary (Codex r1 finding 5): the pull path in `guided_install.py` only
    ever catches `DaemonError`, and a connection lost mid-call must reach it as one, carrying the
    same fixed sentence (the resumable hint included), never escape uncaught.
    """

    alias: str
    runner: SshRunner
    ssh_path: str | None = None
    base_url: str = field(default="")

    def __post_init__(self) -> None:
        self.base_url = f"ssh://{self.alias}"

    def __call__(self, method: str, path: str, body: dict | None = None, progress: Progress | None = None) -> Response:
        try:
            if (method, path) == ("POST", PULL_PATH):
                return self._pull(method, path, body, progress)
            return self._short(method, path, body)
        except SshError as exc:
            raise DaemonError(str(exc)) from exc

    def _short(self, method: str, path: str, body: dict | None) -> Response:
        budget = "daemon_generate" if path == GENERATE_PATH else "daemon_short"
        result = _result_of(self.alias, call(self.alias, budget, {"method": method, "path": path, "body": body}, self.runner, ssh_path=self.ssh_path))
        status, answer_body = result["status"], result.get("body")
        return Response(status=status, headers={}, body=b"" if answer_body is None else json.dumps(answer_body).encode("utf-8"))

    def _pull(self, method: str, path: str, body: dict | None, progress: Progress | None) -> Response:
        """B2: a status other than 200 and an error line of Ollama end the pull with the brief's own
        fixed sentences (`Ollama on <alias> answered with status <n>`, `Ollama on <alias> reported an
        error; run ollama pull <name> there to see it`, `<name>` the laptop's own checked name) --
        raised here, turned into a `DaemonError` by `__call__`, and read by the unchanged `_pull` as
        the reason the pull did not finish."""
        state = {"error": False}  # one flag, however many error lines arrive -- never a collection

        def on_progress(entry: dict) -> None:
            if entry.get("error") is True:
                state["error"] = True
                entry = {"error": f"Ollama on {self.alias} reported an error"}
            if progress is not None:
                progress(entry)

        envelope = call(self.alias, "daemon_pull", {"method": method, "path": path, "body": body}, self.runner, ssh_path=self.ssh_path, on_progress=on_progress)
        result = _result_of(self.alias, envelope)
        if result["status"] != 200:
            raise SshError(f"Ollama on {self.alias} answered with status {result['status']}")
        if state["error"] and not result["done"]:
            raise SshError(f"Ollama on {self.alias} reported an error; run ollama pull {(body or {}).get('model')} there to see it")
        success = b'{"status":"success"}' if result["done"] else b""
        return Response(status=result["status"], headers={}, body=success)


__all__ = ["SshDaemon", "candidates", "hello", "measure", "profile", "room"]
