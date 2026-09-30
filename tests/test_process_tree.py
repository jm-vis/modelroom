"""The process tree of the SSH transport (FR-B1, B1; Codex FR-B1 r4 findings 1 and 2).

Windows: the child belongs to its job object before it runs a single instruction (created
suspended, assigned, resumed), and where the job cannot be set up the call ends fail closed, the
child included. POSIX: the process group is signaled once, and never after its leader was reaped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import modelroom.process_tree as process_tree
from modelroom.ssh import OP_BUDGETS, SshError, call
from test_ssh_process import _HELLO, _ScriptSsh, _assert_all_ended, _pid_alive, _script, _with_grandchild, children  # noqa: F401

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the job object is the Windows side of the tree")


@windows_only
def test_a_helper_started_before_the_job_could_be_set_up_still_ends_with_it(tmp_path, children, monkeypatch):
    """The job setup is slowed down by a second: a child that ran at once would start its helper in
    that second, outside the job. Created suspended, it starts nothing before it belongs to the job,
    and the helper ends with the call."""
    real = process_tree._job_for

    def slow(process):
        time.sleep(1.0)
        return real(process)

    monkeypatch.setattr(process_tree, "_job_for", slow)
    script = _with_grandchild(tmp_path, "time.sleep(30)\n")

    outcome = _ScriptSsh(script).run(["ssh", "gpu1"], b"{}", 4.0, None, OP_BUDGETS["hello"].limits, lambda raw: None)

    assert outcome.timed_out is True
    pid = int((tmp_path / "grandchild.pid").read_text())
    deadline = time.monotonic() + 3.0
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(pid), "the helper started outside the job and still runs"
    _assert_all_ended(children)


@windows_only
@pytest.mark.parametrize("refused", ["_job_for", "_resume"])
def test_a_refused_job_ends_the_call_fail_closed_before_the_child_ran(tmp_path, children, monkeypatch, refused):
    """Codex FR-B1 r4 finding 1: no job (or no resume) -- the child is ended before its first
    instruction, one fixed sentence, nothing sent, no thread, no pipe, no process left."""
    ran = tmp_path / "ran"
    monkeypatch.setattr(process_tree, refused, lambda process: None if refused == "_job_for" else False, raising=False)
    script = _script(tmp_path, f"open({str(ran)!r}, 'w').write('ran')\nsys.stdin.buffer.read()\nprint(json.dumps({{'protocol': 1, 'result': {_HELLO!r}}}), flush=True)\n")

    with pytest.raises(SshError) as excinfo:
        call("gpu1", "hello", {}, _ScriptSsh(script), ssh_path="ssh")

    assert str(excinfo.value) == "modelroom could not start ssh for gpu1 under its own control on this computer; nothing was sent to gpu1"
    time.sleep(0.3)
    assert not ran.exists(), "the child ran although it had no job"
    _assert_all_ended(children)


# --- POSIX: the group is signaled once, and never after its leader was reaped (Codex r4 finding 2) ---


def _posix_tree(monkeypatch, returncode=None):
    """A `ProcessTree` on the POSIX path around a stand-in child, with `killpg` recorded."""
    import os
    import signal

    sent: list[tuple[int, object]] = []
    child = SimpleNamespace(pid=4242, returncode=returncode, poll=lambda: returncode)
    monkeypatch.setattr(process_tree, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append((pid, child.returncode)), raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    return process_tree.ProcessTree(child), child, sent


def test_the_process_group_is_signaled_once(monkeypatch):
    tree, _child, sent = _posix_tree(monkeypatch)

    tree.kill()
    tree.kill()

    assert sent == [(4242, None)]  # once, while the leader was not yet reaped


def test_the_process_group_is_never_signaled_after_its_leader_was_reaped(monkeypatch):
    """Once the leader is reaped its PID may belong to someone else: no signal to that group."""
    tree, _child, sent = _posix_tree(monkeypatch, returncode=0)

    tree.kill()

    assert sent == []
