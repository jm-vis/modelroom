"""The reserve of the system memory: a share of the machine's own size, or the number entered.

Pure, no I/O (AP9-RM, ADR 0012). Until 0.1.0 every machine kept a fixed 8 GiB of its system memory
back from the fit: too little on a machine of 128 GB, where the operating system, a browser and an
editor keep more than that, and more than all of a machine of 4 GB, whose pool came out below zero.
Since AP9-RM the reserve is a share of the memory with a lower and an upper bound -- unless
`modelroom.toml` sets a number for the machine, which always wins and is never overwritten.

The graphics reserve stays what it was: `reserve_vram_gib`, 1 GiB per card by default (`config.py`).
"""

from __future__ import annotations

from typing import Literal

from .config import MachineConfig

# The share of the system memory kept back for everything that is not the model. llmfit's own
# proposal of a share instead of a fixed number (issue #1086, PR #1089, retrieved 2026-09-29) keeps
# "some capacity available for the OS, display, desktop applications"; a quarter is the line macOS
# draws for unified memory without any change (the GPU gets about 75 %, retrieved 2026-09-29).
RAM_RESERVE_SHARE = 0.25
# The least a machine keeps: Windows 11 idles at about 3-4 GB on a machine of 16 GB (user reports,
# no vendor figure, retrieved 2026-09-29); below 8 GB a quarter would leave the system nothing.
RAM_RESERVE_MIN_GIB = 2.0
# The most a machine with a graphics memory of its own keeps: the 8-16 GB Apple advises to leave
# when the limit of the graphics memory is raised (retrieved 2026-09-29). Unified memory has no upper
# bound: there the system and the model share the one memory (decided 2026-09-29, GF question 4).
RAM_RESERVE_MAX_GIB = 16.0
# The fixed reserve every machine of 0.1.0 was written with; a machine that still carries it gets a
# hint, never a change (decided 2026-09-29, GF question 3).
OLD_DEFAULT_RESERVE_RAM_GIB = 8.0

ReserveOrigin = Literal["entered", "computed", "unknown"]

# How the share reads in a sentence; any other share reads as a percentage.
_SHARE_WORDS = {0.25: "a quarter", 0.5: "half"}


def computed_ram_reserve(ram_gib: float, unified: bool) -> float:
    """The reserve of a machine of `ram_gib`: the share, at least the minimum, at most the maximum.

    Unified memory has no maximum. A memory of 0 or less keeps the minimum.
    """
    if ram_gib <= 0:
        return RAM_RESERVE_MIN_GIB
    reserve = max(RAM_RESERVE_MIN_GIB, RAM_RESERVE_SHARE * ram_gib)
    return reserve if unified else min(reserve, RAM_RESERVE_MAX_GIB)


def ram_reserve(machine_config: MachineConfig, ram_gib: float | None, unified: bool) -> tuple[float, ReserveOrigin]:
    """The reserve a fit takes from the system memory, and where it came from.

    A number in the configuration wins, `0.0` included and whether the memory is known or not
    (`entered`). Without one it is computed from the memory (`computed`); with neither a number nor
    a known memory it is `0.0` and `unknown` -- a fit without a known memory has no pool anyway.
    """
    if machine_config.reserve_ram_gib is not None:
        return machine_config.reserve_ram_gib, "entered"
    if ram_gib is None:
        return 0.0, "unknown"
    return computed_ram_reserve(ram_gib, unified), "computed"


def share_words() -> str:
    """The share as a sentence says it: `a quarter`."""
    return _SHARE_WORDS.get(RAM_RESERVE_SHARE, f"{RAM_RESERVE_SHARE * 100:g} %")


def old_default_hint(reserve_gib: float, origin: ReserveOrigin, ram_gib: float | None, unified: bool) -> str | None:
    """The hint for a machine that still carries the fixed reserve of 0.1.0, or `None`.

    Only for an entered `8.0`, a known memory, and a computed reserve that differs from it; nothing
    is changed, the hint only says what the machine would keep (GF question 3, answer (a)).
    """
    if origin != "entered" or reserve_gib != OLD_DEFAULT_RESERVE_RAM_GIB or ram_gib is None:
        return None
    computed = computed_ram_reserve(ram_gib, unified)
    if computed == OLD_DEFAULT_RESERVE_RAM_GIB:
        return None
    return (
        f"memory reserve {OLD_DEFAULT_RESERVE_RAM_GIB:.1f} GB is set in modelroom.toml; "
        f"computed from this machine it would be {computed:.1f} GB"
    )


__all__ = [
    "OLD_DEFAULT_RESERVE_RAM_GIB",
    "RAM_RESERVE_MAX_GIB",
    "RAM_RESERVE_MIN_GIB",
    "RAM_RESERVE_SHARE",
    "ReserveOrigin",
    "computed_ram_reserve",
    "old_default_hint",
    "ram_reserve",
    "share_words",
]
