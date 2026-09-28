"""A model's parameter count: the number the Hub states, else the one its name states.

One rule for every place that orders models by size -- the selection list of step 2 and the
ranking of the result (ADR 0009) -- so the two can never order differently. It lives on its own so
that `ranking`, `render` and the search can read it without pulling in the dialog
(`guided_models` imports `dialog`). Moved here unchanged from `guided_models` (AP9-UX).
"""

from __future__ import annotations

import math
import re

# A parameter count in a model name: `9B`, `0.6B`, `2.4T`, `360M`, the effective size `E4B`, and
# never one that stands behind a letter, a digit or a decimal point -- `A3B`, `A17B`, `8x7B` are the
# active parameters of a mixture or a factor, not a total, and the `.` keeps the tail of such a
# decimal out as well (`Nova-A0.6B` read as 6 in the second-model round of 2026-09-24). `M` and `E`
# since 2026-09-25: the computed release takes its sizes from this one rule as well.
_PARAMETERS_RE = re.compile(r"(?<![0-9A-Za-z.])[Ee]?(\d+(?:\.\d+)?)([BbMmTt])(?![0-9A-Za-z])")
_TRILLION_FACTOR = 1000
_MILLION_DIVISOR = 1000
_GGUF_SUFFIX = "-GGUF"


def parameters_from_name(name: str) -> float | None:
    """The total parameter count a model name states, in billions, or `None` when it states none.

    Every `<number>B`, `<number>M` and `<number>T` of the name that stands on its own counts, with
    an `E` in front of it as well (`E4B`, an effective size) -- a count behind a letter, a digit or a
    decimal point is the active parameters of a mixture (`A3B`, `A0.6B`) or a factor (`8x7B`),
    never a total -- and the largest of them is the model's size. `2.4T` is 2400, `360M` is 0.36.
    A name is registry-controlled text, so only a real, finite, positive number is a count:
    `Nova-0B` and a number of 400 digits both say nothing (the fit is then `unknown` with
    `parameter count unknown`, never a guess and never an exception out of the list).
    """
    stem = name[: -len(_GGUF_SUFFIX)] if name.upper().endswith(_GGUF_SUFFIX) else name
    counts = []
    for number, unit in _PARAMETERS_RE.findall(stem):
        value = float(number)  # digits and at most one dot: `float` raises for nothing else
        if unit in "Tt":
            value *= _TRILLION_FACTOR
        elif unit in "Mm":
            value /= _MILLION_DIVISOR
        if math.isfinite(value) and value > 0:
            counts.append(value)
    return max(counts) if counts else None


def parameter_count(stated: float | None, name: str) -> float | None:
    """The size a model is ordered by: the number the Hub states, else the name's, else `None`.

    `stated` is what a repository of the search states (`SearchHit.parameters_b`, step 2) or what
    the snapshot keeps (`BaseModelSpec.parameters_b`, `modelroom render`); `name` is the model's
    name without its account. `None` ranks behind every known count (ADR 0009).
    """
    return stated if stated is not None else parameters_from_name(name)


__all__ = ["parameter_count", "parameters_from_name"]
