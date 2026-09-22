"""The one result shape both `modelroom/hf.py` and `modelroom/ollama.py` return.

Kept separate from both so neither fetcher module has to import the other just for this
shape, and so `modelroom/state.py`'s merge logic (which does not care which registry an area
came from) has a single type to depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .contracts import Package


@dataclass(frozen=True)
class AreaOutcome:
    """The result of fetching one area: a base model on one source, optionally one HF packager."""

    source: Literal["huggingface", "ollama"]
    base_model_hf_repo: str
    packager: str | None
    status: Literal["complete", "incomplete"]
    error: str | None
    packages: list[Package]


def error_text(exc: BaseException) -> str:
    """`str(exc)`, or a fallback naming the exception's type when that is empty (P3-14, fix-round 5).

    A `Transport` an integrating caller supplies (never `hf.py`/`ollama.py`'s own tested paths,
    which always raise a real message) could raise an exception with no message at all
    (`str(exc) == ""`, e.g. a bare `raise SomeError()`); `Area`'s own validator requires a
    non-empty `error` whenever `status == "incomplete"` (CONTRACTS.md), so an empty string here
    would raise a pydantic `ValidationError` out of `state.merge_snapshot` at merge time instead
    of the ordinary `incomplete` `AreaOutcome` every other failure produces. Every `str(exc)`
    capture in `hf.py`/`ollama.py` that becomes an `AreaOutcome.error` goes through this.
    """
    text = str(exc)
    return text if text else f"{type(exc).__name__} (no message)"
