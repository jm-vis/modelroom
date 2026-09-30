"""The real `ssh` process of the SSH transport (AP9-FR, FR-B1, B1): one child process, three pipes,
one reader thread per output pipe, one writer thread for stdin, and the caller's own thread as the
only consumer, with a deadline.

The whole of stdout is read to EOF within the op's time limits, line by line, every line bounded
(`readline(size)` never returns more than `line_bytes + 1` bytes) and handed to `on_line` as it
arrives. After EOF the process is waited for within what is left of the limits, and stderr is taken
in full once its reader has ended -- so an exit code 127/2/255 and its stderr are always the real
ones, never lost to a race between EOF and exit.

Nothing here can block for good: the stdout reader hands lines over through a small bounded queue
whose `put` gives up as soon as the stop signal is set (a consumer that stopped reading -- a form
fault, a limit, a timeout, Ctrl-C, any exception -- never leaves the reader hanging on a full
queue); the consumer waits in short slices, so a Ctrl-C reaches it on Windows too. Every way out of
`run` goes through the same `finally`: stop signal, the whole process tree ended once
(`process_tree.py`: job object / process group -- a helper `ssh` started, holding a pipe, ends with
it; on POSIX before the child is reaped), the child reaped, every thread joined, every pipe closed.
On Windows the child is created suspended and resumed only once it is in its job; where that
cannot be set up, `ProcessTreeError` ends the call before the child ran (fail closed). A pipe is closed only once its own thread has
ended (closing a pipe another thread still reads from can block the closer); a read still blocked by
a descendant outside the tree is canceled on Windows first. Every thread is a daemon thread named
`modelroom-ssh-*`.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Callable, Sequence

from .process_tree import UNCONFIRMED_END, ProcessTree, abandon, cancel_blocking_read, popen_options

THREAD_PREFIX = "modelroom-ssh-"
_QUEUE_MAXSIZE = 8
_SLICE_SECONDS = 0.05
_JOIN_SECONDS = 5.0
_STOP_WAIT_SECONDS = 5.0
_STDERR_CAP = 64 * 1024
_EXIT_WAIT_WITHOUT_LIMIT = 60.0


@dataclass(frozen=True)
class Limits:
    """The answer limits of one op (B1): every line at most `line_bytes`; the whole answer at most
    `total_bytes`, or unbounded (`None`) where only the line matters -- a pull's status lines."""

    line_bytes: int
    total_bytes: int | None = None


@dataclass(frozen=True)
class ProcessOutcome:
    """How one `SshRunner.run` call ended. `returncode` is `None` when the process was stopped here
    (a timeout or an over-limit answer) rather than exiting on its own."""

    returncode: int | None = None
    timed_out: bool = False
    over_limit: bool = False
    stderr: bytes = b""


# `on_line(raw_line)` is called once per stdout line, in order, as soon as it is read. An exception
# it raises propagates out of `run()`, after the process is stopped and every thread has ended.
OnLine = Callable[[bytes], None]


@dataclass
class _Clock:
    """The two limits of B1: `total` from process start, `per_line` from the last line (or start)."""

    total: float | None
    per_line: float | None
    started: float = field(default_factory=lambda: time.monotonic())  # looked up per call: a test's clock
    last: float = 0.0

    def __post_init__(self) -> None:
        self.last = self.started

    def remaining(self) -> float | None:
        now = time.monotonic()
        waits = []
        if self.total is not None:
            waits.append(self.started + self.total - now)
        if self.per_line is not None:
            waits.append(self.last + self.per_line - now)
        return min(waits) if waits else None

    def line_arrived(self) -> None:
        self.last = time.monotonic()


class _Pipes:
    """The three pipes of one process, their threads, the stop signal and the bounded stderr."""

    def __init__(self, process: subprocess.Popen, line_bytes: int, tree: ProcessTree) -> None:
        self.process = process
        self.tree = tree
        self.line_bytes = line_bytes
        self.stop = threading.Event()
        self.lines: "Queue[tuple[str, bytes] | None]" = Queue(maxsize=_QUEUE_MAXSIZE)
        self.threads: dict[str, threading.Thread] = {}
        self._stderr: list[bytes] = []
        self._stderr_size = 0
        self._lock = threading.Lock()

    def start(self, data: bytes) -> None:
        for name, target, args in (
            ("stdout", _read_stdout, (self,)),
            ("stderr", _read_stderr, (self,)),
            ("stdin", _feed_stdin, (self.process, data)),
        ):
            thread = threading.Thread(target=target, args=args, name=THREAD_PREFIX + name, daemon=True)
            self.threads[name] = thread
            thread.start()

    def put(self, item: "tuple[str, bytes] | None") -> bool:
        """Hand one item to the consumer; `False` once the stop signal is set -- never blocks for good."""
        while not self.stop.is_set():
            try:
                self.lines.put(item, timeout=_SLICE_SECONDS)
                return True
            except Full:
                continue
        return False

    def add_stderr(self, chunk: bytes) -> None:
        with self._lock:
            room = _STDERR_CAP - self._stderr_size
            if room > 0:
                self._stderr.append(chunk[:room])
                self._stderr_size += min(len(chunk), room)

    def stderr(self) -> bytes:
        """stderr as read so far; complete once the stderr thread has ended (`finish_stderr`)."""
        with self._lock:
            return b"".join(self._stderr)

    def finish_stderr(self) -> bytes:
        self.threads["stderr"].join(timeout=_JOIN_SECONDS)
        return self.stderr()

    def close(self) -> None:
        """Join every thread (the process tree is ended already, so every pipe has reached its end),
        then close the pipe it used. A read still blocked -- only a descendant that left the tree can
        hold a pipe open -- is canceled where the system allows it (Windows) and joined again; a pipe
        whose thread is still alive after that is never closed under it (the close would block)."""
        streams = {"stdin": self.process.stdin, "stdout": self.process.stdout, "stderr": self.process.stderr}
        for name, thread in self.threads.items():
            thread.join(timeout=_JOIN_SECONDS)
            if thread.is_alive():
                cancel_blocking_read(thread)
                thread.join(timeout=_JOIN_SECONDS)
            if not thread.is_alive():
                _close(streams[name])


def _close(stream) -> None:
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def _feed_stdin(process: subprocess.Popen, data: bytes) -> None:
    try:
        process.stdin.write(data)
    except (OSError, ValueError):
        pass
    finally:
        _close(process.stdin)


def _read_stdout(pipes: _Pipes) -> None:
    """Bounded lines until EOF; a line longer than `line_bytes` (not counting its newline) is
    handed over as `over` on its first read and ends the reading."""
    size = pipes.line_bytes + 1
    while True:
        try:
            raw = pipes.process.stdout.readline(size)
        except (OSError, ValueError):
            raw = b""
        if not raw:
            pipes.put(None)
            return
        content = raw[:-1] if raw.endswith(b"\n") else raw
        over = len(content) > pipes.line_bytes
        if not pipes.put(("over" if over else "line", raw)) or over:
            return


def _read_stderr(pipes: _Pipes) -> None:
    """stderr until EOF, kept up to `_STDERR_CAP` (classified by pattern, never shown)."""
    while True:
        try:
            raw = pipes.process.stderr.read1(65536) if hasattr(pipes.process.stderr, "read1") else pipes.process.stderr.read(65536)
        except (OSError, ValueError):
            return
        if not raw:
            return
        pipes.add_stderr(raw)


class SubprocessSsh:
    """The real transport. `program`, where given, replaces the first element of the argument
    vector (the `ssh` path) -- a test's `[sys.executable, script]` standing in for `ssh` (B1,
    "Fake-ssh als Prozess"); `popen` is the process factory, `subprocess.Popen` (looked up at each
    call) where none is given."""

    def __init__(self, program: Sequence[str] | None = None, popen: Callable[..., subprocess.Popen] | None = None) -> None:
        self.program = tuple(program) if program else None
        self.popen = popen

    def run(self, argv: Sequence[str], stdin: bytes, total, per_line, limits: Limits, on_line: OnLine) -> ProcessOutcome:
        command = [*self.program, *argv[1:]] if self.program else list(argv)
        popen = self.popen if self.popen is not None else subprocess.Popen
        process = popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_options())
        try:
            tree = ProcessTree(process)  # Windows: into its job, then resumed -- or ended (fail closed)
        except BaseException as original:  # ProcessTreeError, any exception, Ctrl-C: nothing of it is left behind
            _abandon_keeping(process, original)
            raise
        clock = _Clock(total, per_line)
        pipes = _Pipes(process, limits.line_bytes, tree)
        try:
            pipes.start(stdin)
            return _consume(pipes, clock, limits, on_line)
        finally:
            pipes.stop.set()
            stop_process(process, tree)
            pipes.close()
            tree.close()


def _abandon_keeping(process: subprocess.Popen, original: BaseException) -> None:
    """`abandon` the child after a failed setup without letting a fault of the cleanup -- a second
    Ctrl-C, an error of `kill()`/`wait()` -- take the place of `original`; an unconfirmed end is
    recorded on `original` as a note."""
    try:
        if not abandon(process):
            original.add_note(UNCONFIRMED_END)
    except BaseException as cleanup:  # noqa: BLE001 - the original stays the one raised
        original.add_note(f"cleaning up after it failed too: {type(cleanup).__name__}")


def _consume(pipes: _Pipes, clock: _Clock, limits: Limits, on_line: OnLine) -> ProcessOutcome:
    """Every line to EOF, within the limits; then the exit, within what is left of them."""
    total_read = 0
    while True:
        item = _next_item(pipes, clock)
        if item == "timeout":
            return ProcessOutcome(None, timed_out=True, stderr=pipes.stderr())
        if item is None:
            return _await_exit(pipes, clock)
        kind, raw = item
        total_read += len(raw)
        if kind == "over" or (limits.total_bytes is not None and total_read > limits.total_bytes):
            return ProcessOutcome(None, over_limit=True, stderr=pipes.stderr())
        clock.line_arrived()
        on_line(raw)


def _next_item(pipes: _Pipes, clock: _Clock):
    """The next item of the queue, `"timeout"` once a limit is reached; waits in short slices."""
    while True:
        left = clock.remaining()
        if left is not None and left <= 0:
            return "timeout"
        try:
            return pipes.lines.get(timeout=_SLICE_SECONDS if left is None else min(left, _SLICE_SECONDS))
        except Empty:
            continue


def _await_exit(pipes: _Pipes, clock: _Clock) -> ProcessOutcome:
    """stdout ended: the process exits on its own within what is left of the limits (in short
    slices), then stderr is taken in full -- the real exit code, the real stderr."""
    left = clock.remaining()
    deadline = time.monotonic() + (_EXIT_WAIT_WITHOUT_LIMIT if left is None else max(left, 0.0))
    while not pipes.tree.exited():  # POSIX: seen, not reaped -- the group id is still the child's
        if time.monotonic() >= deadline:
            return ProcessOutcome(None, timed_out=True, stderr=pipes.stderr())
        time.sleep(_SLICE_SECONDS)
    pipes.tree.kill()  # a helper `ssh` left behind (a `ProxyJump` hop) must not hold stderr open
    code = pipes.process.wait()  # reaped only now, after the group was signaled
    return ProcessOutcome(code, stderr=pipes.finish_stderr())


def stop_process(process: subprocess.Popen, tree: ProcessTree) -> int | None:
    """End the whole tree -- once, and on POSIX before the child is reaped -- then reap the child;
    a plain kill of the child where it still runs after that. Returns the exit code once known."""
    tree.kill()
    for last_resort in (None, process.kill):
        if last_resort is not None:
            try:
                last_resort()
            except OSError:
                pass
        try:
            return process.wait(timeout=_STOP_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            continue
    return process.poll()


__all__ = ["Limits", "OnLine", "ProcessOutcome", "SubprocessSsh", "THREAD_PREFIX", "stop_process"]
