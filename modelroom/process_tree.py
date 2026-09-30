"""A child process together with every process it starts (AP9-FR, FR-B1, B1).

`ssh` may start helpers of its own -- a `ProxyJump` hop, a `ProxyCommand` -- that inherit its pipes.
Ending only `ssh` itself would leave such a helper holding stderr (or stdout) open, and a thread
reading that pipe would never see its end. So the whole tree is ended:

- Windows: the child is created suspended (`CREATE_SUSPENDED`), put into a job object of its own,
  and only then resumed (`NtResumeProcess` on the process handle -- `subprocess` keeps no thread
  handle): it cannot start a helper before it belongs to the job, and every process it starts
  belongs to the job too (unless it explicitly breaks away). Where the job cannot be set up, the
  suspended child is terminated before it ran a single instruction and `ProcessTreeError` is raised
  (fail closed); an exception or Ctrl-C during the setup ends and reaps the child and releases the
  job the same way before it propagates. `kill()` terminates the job; closing the job handle does
  the same (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
- POSIX: the child starts a session of its own (`start_new_session=True`, set before it runs), so its
  process group is its own PID. The leader is **never reaped before its group was signaled**: while
  it is not reaped, its PID -- and with it the group id -- cannot pass to another process. Its exit
  is seen without reaping: `os.waitid(..., WNOWAIT)`; where Python has no `os.waitid` (macOS with
  Python 3.11 or 3.12), a kqueue `NOTE_EXIT` watch; where neither exists, the end of stdout plus a
  grace of `_EXIT_GRACE_SECONDS` stands for the exit (an `ssh` still running then is ended with its
  group, and its exit code is the signal's). `kill()` then sends one `SIGKILL` to the group, and only
  then is the leader reaped. If the child was reaped by someone else (`ChildProcessError`), the PID
  is no longer known to be ours and no group signal is sent at all.

What is not covered: a descendant that breaks away from the job or starts a session of its own; a
child whose PID was reaped behind this module's back (its helpers are then not signaled).
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time

_CREATE_SUSPENDED = 0x00000004
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_EXIT_GRACE_SECONDS = 0.5


class ProcessTreeError(Exception):
    """The child could not be put under the tree's control; it was ended before it ran."""


def popen_options() -> dict:
    """The extra `Popen` arguments the tree needs on this platform."""
    return {"creationflags": _CREATE_SUSPENDED} if sys.platform == "win32" else {"start_new_session": True}


class ProcessTree:
    """`process` and its descendants; `kill()` ends all of them (once), `close()` releases what was held.

    On Windows `process` must have been started with `popen_options()` (suspended): the constructor
    puts it into the job and resumes it, or ends it and raises (`ProcessTreeError`, or whatever
    interrupted the setup), every handle it took released."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self._job = None
        self._killed = False
        self._lost = False  # POSIX: the PID was reaped behind our back; never signal its group
        self._kqueue = None
        self._eof_at: float | None = None
        if sys.platform == "win32":
            try:
                self._job = _job_for(process)
                ready = bool(self._job) and _resume(process)
            except BaseException:
                self.close()
                raise
            if not ready:
                self.close()
                raise ProcessTreeError("the ssh process could not be put into a job object of its own")

    def exited(self) -> bool:
        """Whether the child has exited -- on POSIX without reaping it (its PID stays its own)."""
        if sys.platform == "win32":
            return self.process.poll() is not None
        if self.process.returncode is not None or self._lost:
            return True
        if hasattr(os, "waitid"):
            try:
                return os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
            except ChildProcessError:
                self._lost = True
                return True
        if hasattr(select, "kqueue"):
            return self._kqueue_exited()
        if self._eof_at is None:
            self._eof_at = time.monotonic()
        return time.monotonic() - self._eof_at >= _EXIT_GRACE_SECONDS

    def _kqueue_exited(self) -> bool:
        """A `NOTE_EXIT` watch on the child's PID (BSD, macOS): its exit, seen without reaping it."""
        if self._kqueue is None:
            self._kqueue = select.kqueue()
            watch = select.kevent(self.process.pid, filter=select.KQ_FILTER_PROC, flags=select.KQ_EV_ADD, fflags=select.KQ_NOTE_EXIT)
            try:
                self._kqueue.control([watch], 0, 0)
            except ProcessLookupError:
                return True  # exited already, a zombie of ours: its PID is still held
        return bool(self._kqueue.control(None, 1, 0))

    def kill(self) -> None:
        """End every process of the tree, once. On POSIX never after the leader was reaped, and never
        where its PID was lost."""
        if self._killed:
            return
        self._killed = True
        if sys.platform == "win32":
            if self._job:
                _kernel32().TerminateJobObject(self._job, 1)
            return
        if self.process.returncode is not None or self._lost:
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
        if self._kqueue is not None:
            self._kqueue.close()
            self._kqueue = None


def abandon(process: subprocess.Popen) -> None:
    """A child whose tree could not be set up: ended, reaped, its three pipes closed."""
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass
    for stream in (process.stdin, process.stdout, process.stderr):
        try:
            if stream is not None:
                stream.close()
        except (OSError, ValueError):
            pass


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


def _limit_information():
    """`JOBOBJECT_EXTENDED_LIMIT_INFORMATION` with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` set."""
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

    info = _Extended()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    return info


def _job_for(process: subprocess.Popen):
    """A job object holding `process`, set to end every process in it when it is closed; `None` where
    the system refuses one (the caller then ends the child, fail closed). The job handle is released
    on every way out but the successful one, an exception or Ctrl-C included."""
    import ctypes
    from ctypes import wintypes

    kernel32 = _kernel32()
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    handle = wintypes.HANDLE(job)
    try:
        info = _limit_information()
        ok = kernel32.SetInformationJobObject(handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info))
        if ok and kernel32.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):
            return handle
    except BaseException:
        kernel32.CloseHandle(handle)
        raise
    kernel32.CloseHandle(handle)
    return None


__all__ = ["ProcessTree", "ProcessTreeError", "abandon", "cancel_blocking_read", "popen_options"]
