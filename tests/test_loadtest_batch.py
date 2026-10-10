"""Tests for modelroom.loadtest_batch: a batch is timed from its release to the last answer (AP9-PM-1, part C).

The clock is a double: the first read of a batch is the barrier's action and gives the release time, every
worker's own clock stands at the release time until its call moves it by the wall time the test gives that call.
"""

from __future__ import annotations

import threading

from modelroom.loadtest_batch import release_batch

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


def test_the_batch_starts_in_the_barriers_action_and_ends_at_the_latest_answer():
    """Worker 0 is answered before the others have even left the barrier (they wait for it), and the main
    thread does nothing between the release and the join: the wall time is the latest end minus the release."""
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


def test_a_batch_whose_first_worker_is_late_to_start_is_still_timed_from_the_release():
    """The old course took the start in the main thread after the barrier: a slow main thread cut the batch's time.
    Here the release is read before any worker runs, so the late worker's end lies inside the batch's time."""
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


def test_one_request_is_a_batch_too():
    clock = _Clock()

    def call(index: int) -> str:
        clock.local.now = RELEASE + 7
        return "ok"

    released = release_batch(1, call, clock)

    assert released.results == ("ok",) and released.wall_ns == 7 == released.spans[0]
