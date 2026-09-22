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
