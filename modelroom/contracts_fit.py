"""`Fit` and `Rating`: the two live, never-persisted models split out of `contracts.py`.

Moved here unchanged by AP9-SV2-0 to keep `contracts.py` under the 799-line size guard as the
snapshot schema grows (docs/adr/0014-…). `contracts.py` re-exports both names
(`from .contracts_fit import Fit, Rating`), so every existing import site
(`from .contracts import Fit`, `from .contracts import Rating`) keeps working unchanged --
`Fit`/`Rating` here and the names `contracts.py` re-exports are the very same class objects, not
copies. `CONTRACTS.md` still describes both under "Fit" and "Rating" as part of the one contract;
this module is a file split, not a new contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Fit(BaseModel):
    """Whether one GGUF package fits one measured machine (fit contract v1).

    Computed by `modelroom/fit.py::compute_fit`, never persisted by AP4 -- a later renderer
    (AP5) calls it fresh against the current snapshot and hardware profile every time it
    renders, and always labels the result "fit (computed, v1)", never "runs" (CONTRACTS.md,
    "Fit contract v1").

    `basis` says what the KV cache was computed from: the base model's own `architecture`, or
    the `size` of the package alone, for a package whose architecture fit v1 cannot read
    (decided 2026-09-24). The provenance of both is `computed`; `size` is the coarser of the two
    and never reaches `ample` (CONTRACTS.md, "Fit from size (basis `size`)").
    """

    model_config = ConfigDict(extra="forbid")

    fit_class: Literal["ample", "fits", "tight", "no_fit", "unknown"]
    mode: Literal["gpu", "gpu_split", "cpu_gpu", "cpu"] | None
    need_gib: float = Field(ge=0)
    weights_gib: float = Field(ge=0)
    kv_gib: float = Field(ge=0)
    pool_gib: float
    reserve_gib: float = Field(ge=0)
    context: int = Field(ge=0)
    context_assumed: bool
    basis: Literal["architecture", "size"] = "architecture"
    reason: str | None
    # The parallel requests the KV cache was computed for, once per request (decided 2026-09-25).
    requests: int = Field(default=1, ge=1, le=1024)
    # AP9-SV2-F: which row of `runtimes.RUNTIMES` this fit belongs to -- one of `profile.RuntimeKey`'s
    # own values, checked against that type at validation time rather than repeated here as a second
    # `Literal` (a field type cannot import `profile.py` at module load: `profile.py` imports
    # `contracts.py`, which imports this module for the re-export above it, so a module-level import
    # back would be a cycle; the validator's import is deferred to call time instead, the same
    # pattern `runtimes.py` already uses for its own deferred imports). Default `"ollama"`, excluded
    # at the default: a render only Ollama ever touches writes no new byte
    # (`test_fit_defaults_are_not_written`).
    runtime: str = Field(default="ollama", exclude_if=lambda value: value == "ollama")
    # How many requests of this fit's context the runtime holds at once on the cards of this row
    # (vLLM's `at_once`, `fit_vllm.vllm_fit_as_fit`). `0` means "not computed for this row" -- true
    # for every Ollama fit, which this repository's render document has never carried a server-side
    # concurrency count for (ADR 0008).
    at_once: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    # How many cards of the machine this row's runtime spreads the model's weights over (vLLM's
    # tensor-parallel size `t`). Ollama never splits a package this way, so its rows keep the
    # default of one card.
    cards: int = Field(default=1, ge=1, exclude_if=lambda value: value == 1)

    @field_validator("runtime")
    @classmethod
    def _check_runtime(cls, value: str) -> str:
        from typing import get_args

        from .profile import RuntimeKey  # deferred: see the field's own comment above

        allowed = get_args(RuntimeKey)
        if value not in allowed:
            raise ValueError(f"runtime must be one of {allowed}: {value!r}")
        return value


class Rating(BaseModel):
    """A market-index star rating for one base model, as a `render.RatingSource` returns it.

    AP5's only new contract: `render` (`modelroom/render.py`) calls a `RatingSource` once per
    base model it renders and turns the result into the Package table's Stars column
    (CONTRACTS.md, "Render (AP5)"). Not persisted anywhere -- a `RatingSource` is a live
    callable, never a file this module reads or writes.
    """

    model_config = ConfigDict(extra="forbid")

    stars: float
    source: str

    @field_validator("stars")
    @classmethod
    def _check_stars(cls, value: float) -> float:
        if not (0.5 <= value <= 5.0):
            raise ValueError(f"stars must be between 0.5 and 5.0: {value!r}")
        if round(value * 2) != value * 2:
            raise ValueError(f"stars must be a multiple of 0.5: {value!r}")
        return value

    @field_validator("source")
    @classmethod
    def _check_source(cls, value: str) -> str:
        if not value:
            raise ValueError("source must be a non-empty label")
        return value
