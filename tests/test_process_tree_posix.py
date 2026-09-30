"""The POSIX side of the process tree, and a setup interrupted on Windows (Codex FR-B1 r5 findings 1-3).

POSIX runs here on a real child process of this machine: the platform check of `process_tree` is set
to `linux`, and the three calls Windows lacks are **stand-ins**, named as such: `os.waitid(...,
WNOWAIT)` answers from the child's process handle (it sees the exit, reaps nothing); `os.killpg`
records whether the leader was still unreaped when its group was signaled, then ends the group's two
members -- the child and the helper it started, which inherited stderr -- the way `SIGKILL` to the
group would. A real `waitid`, kqueue or `killpg` is not exercised on this machine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import modelroom.process_tree as process_tree
from modelroom.ssh import OP_BUDGETS
from test_ssh_process import _HELLO, _ScriptSsh, _assert_all_ended, _pid_alive, _script, _with_grandchild, children  # noqa: F401

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the stand-ins read Windows process handles")
pytestmark = windows_only

_ANSWER = f"sys.stdin.buffer.read()\nprint(json.dumps({{'protocol': 1, 'result': {_HELLO!r}}}), flush=True)\n"


def _terminate_pid(pid: int) -> None:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if handle:
        kernel32.TerminateProcess(handle, 9)
        kernel32.CloseHandle(handle)


def _has_exited(child) -> bool:
    import ctypes

    return ctypes.windll.kernel32.WaitForSingleObject(int(child._handle), 0) == 0  # WAIT_OBJECT_0


def _posix_path(monkeypatch, children, tmp_path: Path, waitid: str) -> list[object]:
    """Put `process_tree` on its POSIX path; `waitid` is `"present"` (stand-in), `"missing"` (and no
    kqueue either: the last resort), or `"lost"` (the PID was reaped by someone else). Returns the
    record of group signals: the leader's `returncode` at the moment its group was signaled."""
    import os
    import signal

    signals: list[object] = []

    def killpg(pid, sig):
        leader = next(child for child in children if child.pid == pid)
        signals.append(leader.returncode)
        _terminate_pid(pid)
        helper = tmp_path / "grandchild.pid"
        if helper.exists():
            _terminate_pid(int(helper.read_text()))

    def waitid_stand_in(idtype, pid, options):
        if waitid == "lost":
            raise ChildProcessError(10, "No child processes")
        leader = next(child for child in children if child.pid == pid)
        return SimpleNamespace(si_pid=pid) if _has_exited(leader) else None

    monkeypatch.setattr(process_tree, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(process_tree, "select", SimpleNamespace(), raising=False)  # no kqueue
    monkeypatch.setattr(os, "killpg", killpg, raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    if waitid == "missing":
        monkeypatch.delattr(os, "waitid", raising=False)
    else:
        monkeypatch.setattr(os, "waitid", waitid_stand_in, raising=False)
        for name, value in (("P_PID", 1), ("WEXITED", 4), ("WNOHANG", 1), ("WNOWAIT", 0x1000000)):
            monkeypatch.setattr(os, name, value, raising=False)
    return signals


@pytest.mark.parametrize("waitid", ["present", "missing"])
def test_a_helper_holding_stderr_after_ssh_answered_and_exited_ends_with_the_group(tmp_path, children, monkeypatch, waitid):
    """Codex FR-B1 r5 finding 1: `ssh` answers and exits; the helper it started still holds stderr.
    With `os.waitid` and without it, the group is signaled once while the leader is still unreaped
    -- the helper ends, the call ends at once, no thread and no pipe is left."""
    signals = _posix_path(monkeypatch, children, tmp_path, waitid)
    script = _with_grandchild(tmp_path, _ANSWER)
    lines: list[bytes] = []
    started = time.monotonic()

    outcome = _ScriptSsh(script).run(["ssh", "gpu1"], b"{}", 10.0, None, OP_BUDGETS["hello"].limits, lines.append)

    assert len(lines) == 1 and outcome.timed_out is False
    assert signals == [None]  # once, before the leader was reaped
    assert time.monotonic() - started < 5.0
    assert not _pid_alive(int((tmp_path / "grandchild.pid").read_text())), "the helper still runs"
    _assert_all_ended(children)


def test_a_pid_reaped_behind_our_back_is_never_signaled_as_a_group(tmp_path, children, monkeypatch):
    """Codex FR-B1 r5 finding 3: `waitid` says the child is not ours any more (`ChildProcessError`):
    its PID may already name another process, so no group signal follows -- not now, not later."""
    signals = _posix_path(monkeypatch, children, tmp_path, "lost")
    script = _script(tmp_path, _ANSWER)
    lines: list[bytes] = []

    outcome = _ScriptSsh(script).run(["ssh", "gpu1"], b"{}", 10.0, None, OP_BUDGETS["hello"].limits, lines.append)

    assert len(lines) == 1 and outcome.timed_out is False
    assert signals == []
    _assert_all_ended(children)


def test_without_waitid_a_kqueue_watch_sees_the_exit_without_reaping(monkeypatch):
    """macOS with Python 3.11/3.12: no `os.waitid`, but a kqueue `NOTE_EXIT` watch (a stand-in
    `select` here) sees the exit; the leader stays unreaped until its group was signaled."""
    import os
    import signal

    events: list[list] = [[], ["exit"]]
    watched: list[tuple] = []

    class _Kqueue:
        def control(self, changes, count, timeout):
            if changes:
                watched.append(tuple(changes))
                return []
            return events.pop(0)

        def close(self):
            pass

    fake_select = SimpleNamespace(
        kqueue=_Kqueue, kevent=lambda ident, **kw: (ident, kw), KQ_FILTER_PROC=-5, KQ_EV_ADD=1, KQ_NOTE_EXIT=0x80000000,
    )
    sent: list[object] = []
    child = SimpleNamespace(pid=4242, returncode=None, poll=lambda: pytest.fail("poll() would reap the leader"))
    monkeypatch.setattr(process_tree, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(process_tree, "select", fake_select, raising=False)
    monkeypatch.delattr(os, "waitid", raising=False)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append(child.returncode), raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    tree = process_tree.ProcessTree(child)

    assert tree.exited() is False and tree.exited() is True
    tree.kill()
    tree.kill()

    assert watched[0][0][0] == 4242 and sent == [None]
    tree.close()


# --- Windows: an interrupted setup leaves no suspended child and no job handle (r5 finding 2) -------


class _Kernel32Watch:
    """`kernel32` with the job calls watched: jobs created, handles closed, and where told, a Ctrl-C
    (`KeyboardInterrupt`) raised inside `AssignProcessToJobObject`."""

    def __init__(self, interrupt_assign: bool) -> None:
        import ctypes
        from ctypes import wintypes

        self._real = ctypes.windll.kernel32
        self._real.CreateJobObjectW.restype = wintypes.HANDLE
        self.created: list[int] = []
        self.closed: list[int] = []
        self.interrupt_assign = interrupt_assign

        def create(*args):  # a plain function: the code under test sets `.restype` on it
            job = self._real.CreateJobObjectW(*args)
            self.created.append(int(job or 0))
            return job

        self.CreateJobObjectW = create

    def __getattr__(self, name):
        return getattr(self._real, name)

    def CloseHandle(self, handle):
        self.closed.append(int(getattr(handle, "value", handle) or 0))
        return self._real.CloseHandle(handle)

    def AssignProcessToJobObject(self, job, process):
        if self.interrupt_assign:
            raise KeyboardInterrupt
        return self._real.AssignProcessToJobObject(job, process)


@pytest.mark.parametrize("where", ["assign", "resume"])
def test_ctrl_c_during_the_setup_leaves_no_suspended_child_and_no_job(tmp_path, children, monkeypatch, where):
    """Codex FR-B1 r5 finding 2: a Ctrl-C inside the job setup (while assigning the job, or while
    resuming the child) reaches the caller -- with the suspended child ended and reaped, its pipes
    closed, and the job handle released."""
    watch = _Kernel32Watch(interrupt_assign=where == "assign")
    monkeypatch.setattr(process_tree, "_kernel32", lambda: watch)
    if where == "resume":
        def interrupted(process):
            raise KeyboardInterrupt

        monkeypatch.setattr(process_tree, "_resume", interrupted)
    script = _script(tmp_path, _ANSWER)

    with pytest.raises(KeyboardInterrupt):
        _ScriptSsh(script).run(["ssh", "gpu1"], b"{}", 10.0, None, OP_BUDGETS["hello"].limits, lambda raw: None)

    assert watch.created and watch.created[0] in watch.closed, "the job handle was not released"
    _assert_all_ended(children)


# --- the cleanup after a failed setup is itself fault-proof (Codex FR-B1 r6 minor) -----------------


def _faulty_popen(started: list, fault: str):
    """A `Popen` whose instance `kill` or `wait` fails the way `fault` names."""
    import subprocess

    def popen(*args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        started.append(process)
        real_kill, real_wait = process.kill, process.wait

        def wait(timeout=None):
            if fault == "wait-interrupt":
                raise KeyboardInterrupt
            if fault == "wait-timeout":
                raise subprocess.TimeoutExpired("ssh", timeout)
            return real_wait(timeout)

        def kill():
            real_kill()
            if fault == "kill-error":
                raise ValueError("the kill went wrong")

        process.wait, process.kill = wait, kill
        return process

    return popen


@pytest.mark.parametrize("fault", ["wait-interrupt", "wait-timeout", "kill-error"])
def test_a_fault_while_cleaning_up_a_failed_setup_keeps_the_original_and_closes_the_pipes(tmp_path, monkeypatch, fault):
    """The setup fails (here: the resume raises); cleaning up then fails too -- a second Ctrl-C in
    `wait()`, a `wait()` that times out, an error of `kill()`. The caller still gets the original
    exception, every pipe is closed, and a timed-out wait is recorded as an unconfirmed end."""
    import subprocess

    from modelroom.ssh_process import SubprocessSsh
    from test_ssh_process import PYTHON

    def broken_resume(process):
        raise RuntimeError("the setup broke")

    monkeypatch.setattr(process_tree, "_resume", broken_resume)
    started: list = []
    script = _script(tmp_path, _ANSWER)
    raised = None
    try:
        SubprocessSsh(popen=_faulty_popen(started, fault)).run(
            [PYTHON, str(script), "gpu1"], b"{}", 10.0, None, OP_BUDGETS["hello"].limits, lambda raw: None
        )
    except BaseException as exc:  # noqa: BLE001 - a KeyboardInterrupt here is what the test looks for
        raised = exc
    finally:
        for process in started:
            subprocess.Popen.kill(process)
            subprocess.Popen.wait(process, 5)

    assert isinstance(raised, RuntimeError) and str(raised) == "the setup broke", repr(raised)
    for name in ("stdin", "stdout", "stderr"):
        assert getattr(started[0], name).closed, f"the {name} pipe was left open"
    if fault == "wait-timeout":
        assert "the ssh process was killed, but its end was not confirmed within 5 s" in getattr(raised, "__notes__", [])
