"""One batch of requests released together, timed from the release to the last answer (AP9-PM-1).

The helper of the two load tests that measure several requests at once (`loadtest_vllm`,
`loadtest_lmstudio`). n workers wait behind one `threading.Barrier`; **the batch's start is taken in the
barrier's own action**, so before any worker is released and not after the main thread got back to its own
line. Each worker takes its own end the moment its call has returned, and the batch's wall time is the
latest end minus that start. A late start therefore can no longer make a batch look shorter, and the total
throughput of a batch can no longer read too high.

The helper judges nothing: what a call returned, or the exception it raised, is handed back per worker, in
the order of the indexes, after every thread has ended.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

Clock = Callable[[], int]


@dataclass(frozen=True)
class Released:
    """What a released batch brought back, one entry per worker in the order of the indexes.

    `results[i]` is what call `i` returned, or the `Exception` it raised; `spans[i]` the nanoseconds from the
    release of worker `i` to the end of its call; `wall_ns` the batch's time, from the barrier's action to the
    latest end.
    """

    results: tuple[Any, ...]
    spans: tuple[int, ...]
    wall_ns: int


def release_batch(n: int, call: Callable[[int], Any], clock: Clock) -> Released:
    """Run `call(index)` for `index` in `0..n-1` in n threads at once; all threads have ended when this returns."""
    starts: list[int] = []
    barrier = threading.Barrier(n, action=lambda: starts.append(clock()))
    results: list[Any] = [None] * n
    spans = [0] * n
    ends = [0] * n

    def worker(index: int) -> None:
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return
        started = clock()
        try:
            results[index] = call(index)
        except Exception as exc:  # handed back, never raised here: the caller orders the failures
            results[index] = exc
        ends[index] = clock()
        spans[index] = ends[index] - started

    threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(n)]
    try:
        for thread in threads:
            thread.start()
    except BaseException:
        barrier.abort()  # a thread that could not start must not leave the others waiting for it
        raise
    for thread in threads:
        thread.join()
    return Released(tuple(results), tuple(spans), max(ends) - starts[0])


__all__ = ["Clock", "Released", "release_batch"]
