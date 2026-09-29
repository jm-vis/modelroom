"""Tests for the reserve of the system memory, `modelroom/reserve.py` (AP9-RM RM-1, ADR 0012).

A quarter of the memory, at least 2 and at most 16 GiB; unified memory without the upper bound; a
number in the configuration wins and says `entered`. Every number is written out so it can be
recomputed by hand.
"""

from __future__ import annotations

import pytest

from modelroom.config import MachineConfig
from modelroom.fit import GpuPool, fit_from_parameters
from modelroom.reserve import computed_ram_reserve, old_default_hint, ram_reserve


def _machine(reserve: float | None = None) -> MachineConfig:
    return MachineConfig(reserve_ram_gib=reserve, reserve_vram_gib=1.0, writer=True)


@pytest.mark.parametrize(
    ("ram_gib", "reserve"),
    [
        (3.81, 2.0),  # the tester's machine: a quarter would be 0.95
        (8.0, 2.0),
        (16.0, 4.0),
        (32.0, 8.0),  # the one size where nothing changes against 0.1.0
        (64.0, 16.0),
        (127.70, 16.0),  # the laptop: a quarter would be 31.9
        (512.0, 16.0),  # the server
    ],
)
def test_the_computed_reserve_is_a_quarter_between_2_and_16(ram_gib, reserve):
    assert computed_ram_reserve(ram_gib, unified=False) == pytest.approx(reserve)


def test_unified_memory_has_no_upper_limit():
    assert computed_ram_reserve(128.0, unified=True) == pytest.approx(32.0)


@pytest.mark.parametrize("entered", [1.5, 8.0, 0.0])
def test_an_entered_reserve_wins_and_says_entered(entered):
    assert ram_reserve(_machine(entered), 127.70, unified=False) == (entered, "entered")


def test_an_entered_reserve_stays_entered_when_the_memory_is_unknown():
    assert ram_reserve(_machine(1.5), None, unified=False) == (1.5, "entered")


def test_a_missing_reserve_is_computed():
    assert ram_reserve(_machine(), 127.70, unified=False) == (16.0, "computed")
    assert ram_reserve(_machine(), 128.0, unified=True) == (32.0, "computed")


def test_no_number_and_unknown_memory_is_origin_unknown():
    assert ram_reserve(_machine(), None, unified=False) == (0.0, "unknown")


def test_memory_below_the_minimum_keeps_the_minimum():
    """1.5 GiB keeps 2.0: the pool is -0.5, and a pool of 0 or less is `no_fit`."""
    assert computed_ram_reserve(1.5, unified=False) == 2.0
    assert computed_ram_reserve(0.0, unified=False) == 2.0
    fit = fit_from_parameters(0.5, GpuPool(()), 1.5, _machine(), 1024)
    assert fit.pool_gib == pytest.approx(-0.5)
    assert (fit.fit_class, fit.reserve_gib) == ("no_fit", 2.0)


def test_the_old_default_gets_a_hint_only_where_the_computed_reserve_differs():
    """GF question 3 (a): an entered 8.0 says what the machine would keep; nothing else does."""
    assert old_default_hint(8.0, "entered", 127.70, unified=False) == (
        "memory reserve 8.0 GB is set in modelroom.toml; computed from this machine it would be 16.0 GB"
    )
    assert old_default_hint(8.0, "entered", 32.0, unified=False) is None  # the same number
    assert old_default_hint(8.0, "entered", None, unified=False) is None  # memory unknown
    assert old_default_hint(1.5, "entered", 127.70, unified=False) is None  # set on purpose
    assert old_default_hint(16.0, "computed", 127.70, unified=False) is None
