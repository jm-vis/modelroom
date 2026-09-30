"""A child process together with every process it starts (AP9-FR, FR-B1, B1).

`ssh` may start helpers of its own -- a `ProxyJump` hop, a `ProxyCommand` -- that inherit its pipes.
Ending only `ssh` itself would leave such a helper holding stderr (or stdout) open, and a thread
reading that pipe would never see its end. So the whole tree is ended:

- Windows: the child is put into a job object of its own right after it starts; every process it
  starts belongs to that job too (unless it explicitly breaks away). `kill()` terminates the job,
  closing the job handle does the same (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
- POSIX: the child starts a session of its own (`start_new_session=True`), so its process group is
  its own PID; `kill()` sends `SIGKILL` to that group.

What is not covered: a descendant that breaks away from the job or starts a session of its own.
"""

from __future__ import annotations

import os
import subprocess
import sys

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


def popen_options() -> dict:
    """The extra `Popen` arguments the tree needs on this platform."""
    return {} if sys.platform == "win32" else {"start_new_session": True}


class ProcessTree:
    """`process` and its descendants; `kill()` ends all of them, `close()` releases what was held."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self._job = _job_for(process) if sys.platform == "win32" else None

    def kill(self) -> None:
        if sys.platform == "win32":
            if self._job:
                _kernel32().TerminateJobObject(self._job, 1)
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


def _job_for(process: subprocess.Popen):
    """A job object holding `process`, set to end every process in it when it is closed; `None` where
    the system refuses one (the tree is then the child alone, as without this module)."""
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


__all__ = ["ProcessTree", "cancel_blocking_read", "popen_options"]
