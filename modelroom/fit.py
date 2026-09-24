"""Fit contract v1: a pure computation of whether one GGUF package fits one measured machine.

No I/O anywhere in this module -- `compute_fit` takes a `Package`, the `BaseModelSpec` it
belongs to, a `HardwareSnapshot` and the machine's `MachineConfig` (for its reserves) and
returns a `Fit`. See CONTRACTS.md, "Fit contract v1", for the formula, the thresholds and the
exact wording a renderer must use for the result. `compute_fit_v2` is the same formula behind
the schema-2 GPU gate and a ranking `Scenario` (CONTRACTS.md, "Fit with profile v2").
"""

from __future__ import annotations

from .config import MachineConfig
from .contracts import BaseModelSpec, Fit, HardwareSnapshot, Package
from .measurements import Scenario
from .profile import HardwareProfile, fit_block_reason

GIB = 1024**3

DEFAULT_CONTEXT = 8192
KV_BYTES_PER_ELEMENT = 2
WEIGHTS_OVERHEAD_RATIO = 1.10
FIXED_OVERHEAD_GIB = 0.50

PERFECT_RATIO = 0.60
GOOD_RATIO = 0.85
MARGINAL_RATIO = 0.98

_WEIGHT_ROLES = ("weights", "weights_shard")


def compute_fit(
    package: Package, base_model: BaseModelSpec, hardware: HardwareSnapshot, machine_config: MachineConfig
) -> Fit:
    """Judge whether `package` fits `hardware`, given `machine_config`'s reserved headroom.

    Only a `complete` `format == "gguf"` package of a `dense_classic` architecture is judged;
    everything else comes back `fit_class == "unknown"` with a `reason`, never a guess.
    """
    base = _judgeable_or_reason(package, base_model)
    if base is not None:
        return base
    return _fit_for_memory(
        package,
        base_model,
        hardware.vram_gib,
        hardware.ram_gib,
        machine_config,
        context=package.default_context or DEFAULT_CONTEXT,
        context_assumed=package.default_context is None,
    )


def compute_fit_v2(
    profile: HardwareProfile,
    package: Package,
    base_model: BaseModelSpec,
    scenario: Scenario,
    machine_config: MachineConfig,
) -> Fit:
    """Fit contract v1 for a schema-2 profile and a ranking scenario.

    The GPU gate comes first: only `gpu_state` `none` (CPU; VRAM 0) and `measured` compute, as
    does only a profile with a known physical RAM and no llmfit deviation (`profile.
    fit_block_reason`); everything else is `unknown` with that reason. The context is the
    scenario's `context_requested` for every package -- the package's own `default_context` is
    never used here -- and fit v1 covers one request with a 16-bit KV cache only. The formula
    itself is fit v1's, unchanged (`_fit_for_memory`). Physical RAM is the RAM pool; a cgroup
    limit is a note, never an input.
    """
    blocked = fit_block_reason(profile)
    if blocked is not None:
        return _unknown(blocked)
    if scenario.requests != 1:
        return _unknown("fit v1 covers one request; more requests are the reverse calculation")
    vram_gib = profile.vram_gib if profile.gpu_state == "measured" else 0.0
    base = _judgeable_or_reason(package, base_model)
    if base is not None:
        return base
    return _fit_for_memory(
        package,
        base_model,
        vram_gib,
        profile.ram_physical_gib,
        machine_config,
        context=scenario.context_requested,
        context_assumed=False,
    )


def count_fitting(
    profile: HardwareProfile,
    machine_config: MachineConfig,
    packages: list[Package],
    base_models: dict[str, BaseModelSpec],
    context: int,
) -> int:
    """How many of `packages` the ranking would rank on `profile` at `context`.

    The one question the size scale of the guided mode asks, six times over -- once per level --
    so that each line can say how many packages still fit. Nothing new is computed here: the fit
    is `compute_fit_v2`'s, unchanged, for one request and a 16-bit KV cache, and "fits" is the
    ranking rule's own predicate, asked by handing the fits to `ranking.rank_packages` with no
    measurements. So the scale and the ranking of the same folder can never disagree. A package
    whose base model is not in `base_models` is not counted (the snapshot itself refuses one).
    """
    from .ranking import rank_packages  # imported here: the rule reads a fit, so it is the later layer

    scenario = Scenario(
        context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1
    )
    entries = [
        (package, compute_fit_v2(profile, package, base_models[package.base_model_hf_repo], scenario, machine_config))
        for package in packages
        if package.base_model_hf_repo in base_models
    ]
    return len(rank_packages(entries, [], scenario).ranked)


def _judgeable_or_reason(package: Package, base_model: BaseModelSpec) -> Fit | None:
    """`None` when fit v1 can judge this package, else the `unknown` result saying why."""
    if base_model.architecture.kind != "dense_classic":
        return _unknown("architecture not covered by v1")
    if package.format != "gguf" or not package.complete:
        return _unknown("only a complete gguf package is judged by fit contract v1")
    if any(f.size_bytes == 0 for f in package.files if f.role in _WEIGHT_ROLES):
        # F4: a weight file the fetcher never learned a real size for (size_bytes == 0) must
        # never silently compute as an empty, "fits everywhere" package.
        return _unknown("a weight file has no size")
    return None


def _fit_for_memory(
    package: Package,
    base_model: BaseModelSpec,
    vram_gib: float,
    ram_gib: float,
    machine_config: MachineConfig,
    context: int,
    context_assumed: bool,
) -> Fit:
    """Fit contract v1's formula and classes for given memory and context (no gate here)."""
    arch = base_model.architecture

    # R7-10 (fix-round 6): `Architecture`'s own numeric fields are bounded (`le=2**31 - 1`,
    # `contracts.py`), but `Package.default_context` is not, and `PackageFile.size_bytes` is
    # bounded only by the fetcher that produced it (`hf.py`/`ollama.py`), not by the contract
    # itself -- a package built some other way (a hand-built `Package`, a future fetcher) could
    # still carry an implausible value here. Every multiplication below stays an exact Python
    # `int` (arbitrary precision, never overflows); only the final `/ GIB` converts to `float`,
    # which raises `OverflowError` for a product too large to represent (probe: `num_hidden_layers
    # = 10**400` -- now caught before this point by the contract bound -- or, still possible
    # today, an implausible `default_context`/`size_bytes` combined with in-bound architecture
    # values). Caught here as the second line of defense: this one package ends `fit_class ==
    # "unknown"`, never the whole render.
    try:
        weights_gib = sum(f.size_bytes for f in package.files if f.role in _WEIGHT_ROLES) / GIB
        kv_gib = (
            2 * arch.num_hidden_layers * arch.num_key_value_heads * arch.head_dim * KV_BYTES_PER_ELEMENT * context
        ) / GIB
        need_gib = weights_gib * WEIGHTS_OVERHEAD_RATIO + kv_gib + FIXED_OVERHEAD_GIB
    except OverflowError:
        return _unknown("architecture values out of range")

    mode, pool_gib, reserve_gib, cap_at_good = _choose_pool(need_gib, vram_gib, ram_gib, machine_config)
    fit_class = _classify(need_gib, pool_gib, cap_at_good)

    return Fit(
        fit_class=fit_class,
        mode=mode,
        need_gib=need_gib,
        weights_gib=weights_gib,
        kv_gib=kv_gib,
        pool_gib=pool_gib,
        reserve_gib=reserve_gib,
        context=context,
        context_assumed=context_assumed,
        reason=None,
    )


def _choose_pool(
    need_gib: float, vram_gib: float, ram_gib: float, machine_config: MachineConfig
) -> tuple[str, float, float, bool]:
    """The memory pool `need_gib` is judged against, and whether the class is capped at "good".

    The package fits the GPU when `need_gib` is at most the VRAM left after
    `reserve_vram_gib`; otherwise it falls back to the RAM pool (minus `reserve_ram_gib`),
    mode `cpu_gpu` when there is some VRAM at all, `cpu` when there is none -- and the class is
    capped at "good" off the GPU, since "perfect" only ever describes a package that fits
    comfortably in VRAM.
    """
    available_vram = vram_gib - machine_config.reserve_vram_gib
    if need_gib <= available_vram:
        return "gpu", available_vram, machine_config.reserve_vram_gib, False

    pool_gib = ram_gib - machine_config.reserve_ram_gib
    mode = "cpu_gpu" if vram_gib > 0 else "cpu"
    return mode, pool_gib, machine_config.reserve_ram_gib, True


def _classify(need_gib: float, pool_gib: float, cap_at_good: bool) -> str:
    if pool_gib <= 0:
        return "too_tight"
    ratio = need_gib / pool_gib
    if ratio <= PERFECT_RATIO:
        return "good" if cap_at_good else "perfect"
    if ratio <= GOOD_RATIO:
        return "good"
    if ratio <= MARGINAL_RATIO:
        return "marginal"
    return "too_tight"


def _unknown(reason: str) -> Fit:
    return Fit(
        fit_class="unknown",
        mode=None,
        need_gib=0.0,
        weights_gib=0.0,
        kv_gib=0.0,
        pool_gib=0.0,
        reserve_gib=0.0,
        context=0,
        context_assumed=False,
        reason=reason,
    )
