"""Tests for modelroom.loadtest_batch: a batch is timed from its release to the last answer (AP9-PM-1, part C).

The clock is a double: the first read of a batch is the barrier's action and gives the release time, every
worker's own clock stands at the release time until its call moves it by the wall time the test gives that call.
"""

from __future__ import annotations

import threading

from modelroom.loadtest_batch import BatchError, release_batch

RELEASE = 1_000


class _Clock:
    def __init__(self) -> None:
        self.local = threading.local()
        self.lock = threading.Lock()
        self.reads: list[int] = []

    def __call__(self) -> int:
        with self.lock:
            first = not self.reads
            self.reads.append(0)
        return RELEASE if first else getattr(self.local, "now", RELEASE)


def test_the_wall_time_is_the_latest_end_minus_the_release_and_the_clock_is_read_1_plus_2n_times():
    """Worker 0 is answered first, the others wait for it inside their calls: the wall time is the latest end
    minus the release read in the barrier's action, and the clock is read once there and twice per worker."""
    clock = _Clock()
    walls = [50, 10, 30]
    first_done = threading.Event()

    def call(index: int) -> str:
        if index != 0:
            assert first_done.wait(3.0)  # worker 0 answered first, whatever the main thread was doing
        clock.local.now = RELEASE + walls[index]
        if index == 0:
            first_done.set()
        return f"answer {index}"

    released = release_batch(3, call, clock)

    assert released.results == ("answer 0", "answer 1", "answer 2")
    assert released.spans == (50, 10, 30)
    assert released.wall_ns == 50
    assert len(clock.reads) == 1 + 2 * 3  # the action, then a start and an end per worker


def test_a_slow_request_ends_inside_the_batch_time():
    """Two calls of 100 and 400 ns: the batch is 400 ns long, the longest request's span (not a start read later)."""
    clock = _Clock()

    def call(index: int) -> int:
        clock.local.now = RELEASE + (400 if index == 1 else 100)
        return index

    released = release_batch(2, call, clock)

    assert released.wall_ns == 400 and released.spans == (100, 400)


def test_a_call_that_raises_is_handed_back_in_its_place_and_every_thread_ends():
    before = threading.active_count()

    def call(index: int) -> int:
        if index == 1:
            raise ValueError("boom")
        return index

    released = release_batch(3, call, _Clock())

    assert released.results[0] == 0 and released.results[2] == 2
    assert isinstance(released.results[1], ValueError)
    assert threading.active_count() == before


def _watched(function, seconds: float = 5.0):
    """Run `function` in a thread of its own and wait at most `seconds`: a hang is a failure, not a stuck suite."""
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["value"] = function()
        except BaseException as exc:
            outcome["error"] = exc

    watcher = threading.Thread(target=target, daemon=True)
    watcher.start()
    watcher.join(seconds)
    assert not watcher.is_alive(), "release_batch hung"
    return outcome


def test_the_release_is_read_before_any_worker_is_released():
    """The barrier's action is held open: while it is, no call has begun (a start read in the main thread after the
    barrier, as before AP9-PM-1, would have let the calls run first). Then the action is let go."""
    action_entered, go = threading.Event(), threading.Event()
    begun: list[int] = []
    reads: list[str] = []

    def clock() -> int:
        if not reads:
            reads.append("release")
            action_entered.set()
            assert go.wait(5.0)
            return RELEASE
        return RELEASE

    def call(index: int) -> int:
        reads.append("call")
        begun.append(index)
        return index

    holder: dict = {}
    runner = threading.Thread(target=lambda: holder.update(done=release_batch(3, call, clock)), daemon=True)
    runner.start()
    assert action_entered.wait(5.0)
    threading.Event().wait(0.2)  # a worker released too early would have begun its call by now
    assert begun == [] and reads == ["release"]
    go.set()
    runner.join(5.0)
    assert not runner.is_alive() and sorted(begun) == [0, 1, 2] and reads[0] == "release"


def test_a_thread_that_cannot_start_ends_the_started_ones_and_raises_its_error(monkeypatch):
    real = threading.Thread
    alive_before = threading.active_count()
    starts = {"n": 0}

    class Flaky(real):
        def start(self):
            starts["n"] += 1
            if starts["n"] == 4:  # the watcher is the first start, two workers follow, the third worker fails
                raise RuntimeError("can't start new thread")
            super().start()

    monkeypatch.setattr(threading, "Thread", Flaky)
    outcome = _watched(lambda: release_batch(5, lambda index: index, _Clock()))
    monkeypatch.undo()

    assert isinstance(outcome.get("error"), RuntimeError) and starts["n"] == 4
    assert threading.active_count() == alive_before  # the two started threads ended, none waits at the barrier


def test_a_clock_that_fails_in_the_barriers_action_is_reported_and_leaves_no_thread():
    alive_before = threading.active_count()

    def clock() -> int:
        raise ValueError("clock gone")

    outcome = _watched(lambda: release_batch(4, lambda index: index, clock))

    error = outcome.get("error")
    assert isinstance(error, BatchError) and isinstance(error.__cause__, ValueError)
    assert "clock gone" in str(error)
    assert threading.active_count() == alive_before


def test_a_clock_that_fails_at_the_end_of_a_worker_is_reported():
    reads: list[int] = []
    lock = threading.Lock()

    def clock() -> int:
        with lock:
            reads.append(1)
            if len(reads) == 6:  # one of the later reads of the 7 (the action, then a start and an end per worker)
                raise ValueError("late failure")
        return RELEASE

    outcome = _watched(lambda: release_batch(3, lambda index: index, clock))

    assert isinstance(outcome.get("error"), BatchError) and "late failure" in str(outcome["error"])


def test_one_request_is_a_batch_too():
    clock = _Clock()

    def call(index: int) -> str:
        clock.local.now = RELEASE + 7
        return "ok"

    released = release_batch(1, call, clock)

    assert released.results == ("ok",) and released.wall_ns == 7 == released.spans[0]
