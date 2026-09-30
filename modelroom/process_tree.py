"""A child process together with every process it starts (AP9-FR, FR-B1, B1).

`ssh` may start helpers of its own -- a `ProxyJump` hop, a `ProxyCommand` -- that inherit its pipes.
Ending only `ssh` itself would leave such a helper holding stderr (or stdout) open, and a thread
reading that pipe would never see its end. So the whole tree is ended:

- Windows: the child is created suspended (`CREATE_SUSPENDED`), put into a job object of its own,
  and only then resumed (`NtResumeProcess` on the process handle -- `subprocess` keeps no thread
  handle): it cannot start a helper before it belongs to the job, and every process it starts
  belongs to the job too (unless it explicitly breaks away). Where the job cannot be set up, the
  suspended child is terminated before it ran a single instruction and `ProcessTreeError` is raised
  (fail closed). `kill()` terminates the job; closing the job handle does the same
  (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
- POSIX: the child starts a session of its own (`start_new_session=True`, set before it runs), so its
  process group is its own PID. `exited()` sees the leader's exit without reaping it (`waitid` with
  `WNOWAIT`), so `kill()` sends `SIGKILL` to the group while its PID still belongs to it -- once, and
  never after the leader was reaped, so a reused PID can never be hit. Where `os.waitid` is missing
  (macOS with Python 3.11 or 3.12), `exited()` falls back to `poll()`, which reaps the leader as it
  sees its exit: after `ssh` ended on its own, the group is then not signaled at all, and a helper it
  left behind can survive (and hold stderr open until it ends). While `ssh` still runs -- a timeout,
  a fault, Ctrl-C -- the group is signaled as before.

What is not covered: a descendant that breaks away from the job or starts a session of its own; on
POSIX without `os.waitid`, a helper that outlives an `ssh` which ended on its own.
"""

from __future__ import annotations

import os
import subprocess
import sys

_CREATE_SUSPENDED = 0x00000004
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


class ProcessTreeError(Exception):
    """The child could not be put under the tree's control; it was ended before it ran."""


def popen_options() -> dict:
    """The extra `Popen` arguments the tree needs on this platform."""
    return {"creationflags": _CREATE_SUSPENDED} if sys.platform == "win32" else {"start_new_session": True}


class ProcessTree:
    """`process` and its descendants; `kill()` ends all of them (once), `close()` releases the job.

    On Windows `process` must have been started with `popen_options()` (suspended): the constructor
    puts it into the job and resumes it, or ends it and raises `ProcessTreeError`."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self._job = None
        self._killed = False
        if sys.platform == "win32":
            self._job = _job_for(process)
            if not self._job or not _resume(process):
                self._fail_closed()

    def _fail_closed(self) -> None:
        try:
            self.process.kill()
            self.process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.close()
        raise ProcessTreeError("the ssh process could not be put into a job object of its own")

    def exited(self) -> bool:
        """Whether the child has exited, without reaping it on POSIX (its PID stays its own). Where
        `os.waitid` is missing (macOS before Python 3.13), `poll()` both sees the exit and reaps the
        child; `kill()` then sends no group signal at all -- never to a reaped leader's PID."""
        if sys.platform == "win32" or self.process.returncode is not None or not hasattr(os, "waitid"):
            return self.process.poll() is not None
        try:
            info = os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return True
        return info is not None

    def kill(self) -> None:
        """End every process of the tree, once. On POSIX never after the leader was reaped."""
        if self._killed:
            return
        self._killed = True
        if sys.platform == "win32":
            if self._job:
                _kernel32().TerminateJobObject(self._job, 1)
            return
        if self.process.returncode is not None:
            return
        import signal

        try:
            os.killpg(self.process.pid, signal.SIGKILL)  # the group id is the child's own PID
        except (ProcessLookupError, PermissionError):
            pass

    def close(self) -> None:
        if self._job:
            _kernel32().CloseHandle(self._job)
            self._job = None


def cancel_blocking_read(thread) -> None:
    """Windows: cancel the synchronous pipe read `thread` is blocked in (`CancelSynchronousIo`); its
    read then fails with an `OSError` and the reader ends. Nothing on POSIX, where a blocked read
    cannot be canceled from outside -- there the ended process group has closed every pipe."""
    if sys.platform != "win32" or thread.native_id is None:
        return
    kernel32 = _kernel32()
    handle = kernel32.OpenThread(0x0001, False, thread.native_id)  # THREAD_TERMINATE
    if not handle:
        return
    try:
        kernel32.CancelSynchronousIo(handle)
    finally:
        kernel32.CloseHandle(handle)


def _kernel32():
    import ctypes

    return ctypes.windll.kernel32


def _resume(process: subprocess.Popen) -> bool:
    """Resume the suspended child through its process handle; `False` where that fails."""
    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.windll.ntdll
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    return ntdll.NtResumeProcess(wintypes.HANDLE(int(process._handle))) == 0


def _job_for(process: subprocess.Popen):
    """A job object holding `process`, set to end every process in it when it is closed; `None` where
    the system refuses one (the caller then ends the child, fail closed)."""
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = _kernel32()
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(wintypes.HANDLE(job), _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info))
    if ok and kernel32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(process._handle))):
        return wintypes.HANDLE(job)
    kernel32.CloseHandle(wintypes.HANDLE(job))
    return None


__all__ = ["ProcessTree", "ProcessTreeError", "cancel_blocking_read", "popen_options"]
