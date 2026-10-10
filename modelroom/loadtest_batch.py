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

from .loadtest import LoadTestError

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


class BatchError(LoadTestError):
    """The batch itself failed -- the clock, the barrier -- not a call in it. Nothing of it can be measured."""


def release_batch(n: int, call: Callable[[int], Any], clock: Clock) -> Released:
    """Run `call(index)` for `index` in `0..n-1` in n threads at once; every started thread has ended when this
    returns or raises.

    A call that raises is handed back in its place. A failure of the batch itself -- the clock raising in the
    barrier's action or in a worker, a broken barrier -- is a `BatchError` naming the first cause; a thread that
    cannot be started, or an interrupt while waiting, aborts the barrier, lets the started threads end and is
    raised as it is.
    """
    starts: list[int] = []
    problems: list[BaseException] = []
    broken: list[bool] = []
    barrier = threading.Barrier(n, action=lambda: starts.append(clock()))
    results: list[Any] = [None] * n
    spans = [0] * n
    ends: list[int | None] = [None] * n

    def worker(index: int) -> None:
        try:
            barrier.wait()
            started = clock()
        except threading.BrokenBarrierError:
            broken.append(True)
            return
        except Exception as exc:  # the clock in the action (raised here by the barrier) or in this worker
            problems.append(exc)
            barrier.abort()
            return
        try:
            results[index] = call(index)
        except Exception as exc:  # handed back, never raised here: the caller orders the failures
            results[index] = exc
        try:
            ends[index] = clock()
        except Exception as exc:
            problems.append(exc)
            return
        spans[index] = ends[index] - started

    threads: list[threading.Thread] = []
    try:
        for index in range(n):
            thread = threading.Thread(target=worker, args=(index,), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    except BaseException:
        barrier.abort()  # a thread that could not start must not leave the others waiting for it
        for thread in threads:
            thread.join()
        raise
    if problems:
        raise BatchError(f"a batch of requests could not be timed ({problems[0]})") from problems[0]
    if broken or not starts or any(end is None for end in ends):
        raise BatchError("a batch of requests could not be timed (its barrier broke)")
    return Released(tuple(results), tuple(spans), max(end for end in ends if end is not None) - starts[0])


__all__ = ["BatchError", "Clock", "Released", "release_batch"]
