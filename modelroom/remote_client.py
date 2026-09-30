"""The laptop side of the remote protocol (AP9-FR, FR-B1, B2): `hello`/`profile`/`room`, and
`SshDaemon`, which makes a computer read over SSH answer to the unchanged `Daemon` protocol.

`SshDaemon` **translates** the wire envelope into exactly the shapes `guided_install.py`'s
`_send`/`_check_line`/`_end_reason`/`pulled_problem` already expect (Codex r2-6): nothing there
changes for a pull over SSH, only where the bytes come from. `hello`/`profile`/`room` are the three
read-only calls `guided_remote.py` (B3-B5) makes directly; each raises `ssh.SshError` on a business
error the server sent back (`bad_request`, `internal`, …), since a well-formed request from this
client should never draw one -- if it does, the fixed "answered in a form this modelroom does not
read" sentence is the honest one, not a guess at what went wrong on the other end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from .daemon import GENERATE_PATH, PULL_PATH, Progress
from .http import Response
from .ssh import SshError, SshRunner, call

FORM_ERROR = "{alias} answered in a form this modelroom does not read"
AGENT_FORWARDED = "your SSH agent reached {alias} although modelroom switches forwarding off; check your SSH configuration"
OLLAMA_UNREACHABLE = "Ollama does not answer on {alias}"


def _only_result(alias: str, lines: list[dict]) -> dict:
    last = lines[-1]
    if "result" not in last:
        raise SshError(FORM_ERROR.format(alias=alias))
    return last["result"]


def hello(alias: str, runner: SshRunner, ssh_path: str = "ssh") -> dict:
    """`{version, system, ollama, agent_forwarded, os_fingerprint}` -- and the agent check (B2):
    a forwarded agent stops the run here, before anything else of this alias is read."""
    result = _only_result(alias, call(alias, "hello", {}, runner, ssh_path=ssh_path))
    if result.get("agent_forwarded") is True:
        raise SshError(AGENT_FORWARDED.format(alias=alias))
    if result.get("ollama") == "unreachable":
        raise SshError(OLLAMA_UNREACHABLE.format(alias=alias))
    return result


def profile(alias: str, profile_id: str, machine_class: str | None, runner: SshRunner, ssh_path: str = "ssh") -> dict:
    """`{"export": <ExportObject v1>}` of `profile_id`, measured now on `alias`."""
    request = {"profile_id": profile_id, "machine_class": machine_class}
    return _only_result(alias, call(alias, "profile", request, runner, ssh_path=ssh_path))


def room(alias: str, cards: Sequence[tuple[int, str | None]], runner: SshRunner, ssh_path: str = "ssh") -> dict:
    """`{"disk": {...}, "graphics_free_mib": [...] | None}` of `alias`, read now, no path, no drive."""
    request = {"cards": [[index, name] for index, name in cards]}
    return _only_result(alias, call(alias, "room", request, runner, ssh_path=ssh_path))


@dataclass
class SshDaemon:
    """A `Daemon` (`daemon.py`'s protocol) whose calls travel to `alias`'s own Ollama over SSH.

    `installed_models`, `_is_local`, `_send`, `_check_line`, `_end_reason`, `pulled_problem`
    (`guided_install.py`) run against this exactly as they run against `LocalDaemon` -- nothing
    there was changed for FR-B1 (Codex r2-6).
    """

    alias: str
    runner: SshRunner
    ssh_path: str = "ssh"
    base_url: str = field(default="")

    def __post_init__(self) -> None:
        self.base_url = f"ssh://{self.alias}"

    def __call__(self, method: str, path: str, body: dict | None = None, progress: Progress | None = None) -> Response:
        if (method, path) == ("POST", PULL_PATH):
            return self._pull(method, path, body, progress)
        return self._short(method, path, body)

    def _short(self, method: str, path: str, body: dict | None) -> Response:
        budget = "daemon_generate" if path == GENERATE_PATH else "daemon_short"
        lines = call(self.alias, budget, {"method": method, "path": path, "body": body}, self.runner, ssh_path=self.ssh_path)
        result = _only_result(self.alias, lines)
        status, answer_body = result.get("status"), result.get("body")
        return Response(status=status, headers={}, body=b"" if answer_body is None else json.dumps(answer_body).encode("utf-8"))

    def _pull(self, method: str, path: str, body: dict | None, progress: Progress | None) -> Response:
        lines = call(self.alias, "daemon_pull", {"method": method, "path": path, "body": body}, self.runner, ssh_path=self.ssh_path)
        for line in lines:
            answer = self._pull_line(line, progress)
            if answer is not None:
                return answer
        raise SshError(FORM_ERROR.format(alias=self.alias))

    def _pull_line(self, line: dict, progress: Progress | None) -> Response | None:
        if "progress" in line:
            self._pull_progress(line["progress"], progress)
            return None
        if "result" in line:
            result = line["result"]
            success = b'{"status":"success"}' if result.get("done") else b""
            return Response(status=result.get("status"), headers={}, body=success)
        raise SshError(FORM_ERROR.format(alias=self.alias))

    def _pull_progress(self, entry: dict, progress: Progress | None) -> None:
        if progress is None:
            return
        if entry.get("error") is True:
            progress({"error": f"Ollama on {self.alias} reported an error"})
        else:
            progress(entry)


__all__ = ["SshDaemon", "hello", "profile", "room"]
