"""Tests for modelroom.fit: the pure fit computation (contract v1).

Every hardware/package/architecture fixture here is built in memory (no I/O) -- `compute_fit`
takes a `Package`, a `BaseModelSpec`, a `HardwareSnapshot` and a `MachineConfig` and returns a
`Fit`, nothing else. See CONTRACTS.md, "Fit contract v1", for the formula and thresholds this
file checks against by hand in each test's comments.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modelroom.config import MachineConfig
from modelroom.contracts import Architecture, BaseModelSpec, HardwareSnapshot, Package, PackageFile
from modelroom.fit import compute_fit

NOW = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
GIB = 1024**3


def _architecture(*, kind: str = "dense_classic", num_hidden_layers=36, num_key_value_heads=8, head_dim=128) -> Architecture:
    if kind == "unknown":
        return Architecture(source_repo="acme/Nova-8B", source_revision=None, kind="unknown")
    return Architecture(
        source_repo="acme/Nova-8B",
        source_revision=None,
        kind="dense_classic",
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )


def _base_model(*, architecture: Architecture | None = None) -> BaseModelSpec:
    return BaseModelSpec(
        hf_repo="acme/Nova-8B",
        publisher="acme",
        parameters_b=8.0,
        architecture=architecture or _architecture(),
    )


def _gguf_package(
    *, weights_bytes: int, default_context: int | None = None, complete: bool = True, format: str = "gguf"
) -> Package:
    files = [PackageFile(name="Nova-8B-Q4_K_M.gguf", role="weights", size_bytes=weights_bytes, digest=None)] if complete else []
    return Package(
        source="huggingface",
        repo="packager/Nova-8B-GGUF",
        revision="a" * 40,
        base_model_hf_repo="acme/Nova-8B",
        format=format,
        files=files,
        complete=complete if files else False,
        quantization="Q4_K_M" if format == "gguf" else None,
        default_context=default_context,
        provenance="metadata_ok" if format == "gguf" else "unresolved",
        unresolved_reason=None if format == "gguf" else "format",
        approval=None,
        observed_at=NOW,
        last_seen=NOW,
        active=True,
    )


def _hardware(*, vram_gib: float, ram_gib: float, free_ram_gib_at_measurement: float | None = None) -> HardwareSnapshot:
    return HardwareSnapshot(
        schema_version=1,
        machine="workstation",
        measured_at=NOW,
        llmfit_version="1.1.16",
        vram_gib=vram_gib,
        ram_gib=ram_gib,
        free_ram_gib_at_measurement=free_ram_gib_at_measurement,
        gpu_name="Nova GPU" if vram_gib else None,
        backend="CUDA" if vram_gib else None,
        unified_memory=False,
        installed=None,
        installed_unavailable_reason="not queried in this test",
        measurements=[],
    )


def _machine(*, reserve_ram_gib: float, reserve_vram_gib: float) -> MachineConfig:
    return MachineConfig(reserve_ram_gib=reserve_ram_gib, reserve_vram_gib=reserve_vram_gib, writer=True)


LAPTOP = _hardware(vram_gib=11.94, ram_gib=127.46, free_ram_gib_at_measurement=76.64)
LAPTOP_MACHINE = _machine(reserve_ram_gib=16.0, reserve_vram_gib=1.0)
SERVER = _hardware(vram_gib=0.0, ram_gib=7.56)
SERVER_MACHINE = _machine(reserve_ram_gib=3.0, reserve_vram_gib=0.0)


# --- architecture/package coverage ---------------------------------------------------------


def test_unknown_architecture_is_unknown_fit():
    package = _gguf_package(weights_bytes=5 * GIB)
    base_model = _base_model(architecture=_architecture(kind="unknown"))

    fit = compute_fit(package, base_model, LAPTOP, LAPTOP_MACHINE)

    assert fit.fit_class == "unknown"
    assert fit.mode is None
    assert fit.reason == "architecture not covered by v1"


def test_incomplete_package_is_unknown_fit():
    # P3-7 (fix-round 5): `fit.reason is not None` alone would still pass even if `compute_fit`
    # mixed this case up with a different `unknown` branch (e.g. "architecture not covered by
    # v1") -- pin the exact reason this test's name is actually about.
    package = _gguf_package(weights_bytes=5 * GIB, complete=False)
    base_model = _base_model()

    fit = compute_fit(package, base_model, LAPTOP, LAPTOP_MACHINE)

    assert fit.fit_class == "unknown"
    assert fit.reason == "only a complete gguf package is judged by fit contract v1"


def test_non_gguf_package_is_unknown_fit():
    package = _gguf_package(weights_bytes=5 * GIB, format="tensor")
    base_model = _base_model()

    fit = compute_fit(package, base_model, LAPTOP, LAPTOP_MACHINE)

    assert fit.fit_class == "unknown"
    assert fit.reason is not None


def test_zero_byte_weight_file_is_unknown_fit_with_reason():
    # F4: a 0-byte weight file means the fetcher never learned a real size for it -- fit must
    # never silently ignore the weights and compute a fake "fits everywhere" verdict.
    package = _gguf_package(weights_bytes=0)
    base_model = _base_model()

    fit = compute_fit(package, base_model, LAPTOP, LAPTOP_MACHINE)

    assert fit.fit_class == "unknown"
    assert fit.reason == "a weight file has no size"


# --- brief's worked examples: 8B and 9B dense packages on laptop vs. server ----------------


def test_dense_8b_package_is_good_gpu_on_laptop():
    # L=36, KVH=8, D=128, context assumed 8192: kv_gib = 2*36*8*128*2*8192 / 1024**3 = 1.125
    # GiB exactly. weights_gib = 5.0. need_gib = 5.0*1.10 + 1.125 + 0.50 = 7.125 GiB.
    # laptop available_vram = 11.94 - 1.0 = 10.94; 7.125 <= 10.94 -> gpu.
    # ratio = 7.125 / 10.94 = 0.6513... -> "good" (> 0.60, <= 0.85).
    package = _gguf_package(weights_bytes=5 * GIB)
    base_model = _base_model(architecture=_architecture(num_hidden_layers=36, num_key_value_heads=8, head_dim=128))

    fit = compute_fit(package, base_model, LAPTOP, LAPTOP_MACHINE)

    assert fit.mode == "gpu"
    assert fit.fit_class == "good"
    assert fit.weights_gib == pytest.approx(5.0)
    assert fit.kv_gib == pytest.approx(1.125)
    assert fit.need_gib == pytest.approx(7.125)
    assert fit.context == 8192
    assert fit.context_assumed is True
    assert fit.reserve_gib == pytest.approx(1.0)


def test_dense_8b_package_is_too_tight_on_server():
    # server: available_vram = 0 - 0 = 0, need(7.125) > 0 -> cpu path.
    # pool = ram_gib(7.56) - reserve_ram(3.0) = 4.56; ratio = 7.125 / 4.56 = 1.5625 -> too_tight.
    package = _gguf_package(weights_bytes=5 * GIB)
    base_model = _base_model(architecture=_architecture(num_hidden_layers=36, num_key_value_heads=8, head_dim=128))

    fit = compute_fit(package, base_model, SERVER, SERVER_MACHINE)

    assert fit.mode == "cpu"
    assert fit.fit_class == "too_tight"
    assert fit.pool_gib == pytest.approx(4.56)


def test_dense_9b_package_is_good_gpu_on_laptop():
    # L=42, KVH=8, D=128: kv_gib = 2*42*8*128*2*8192 / 1024**3 = 1.3125 GiB.
    # weights_gib = 5.6. need_gib = 5.6*1.10 + 1.3125 + 0.50 = 7.9725 GiB.
    # laptop: available_vram = 10.94 >= 7.9725 -> gpu; ratio = 7.9725/10.94 = 0.7288 -> good.
    package = _gguf_package(weights_bytes=int(5.6 * GIB))
    base_model = _base_model(architecture=_architecture(num_hidden_layers=42, num_key_value_heads=8, head_dim=128))

    fit = compute_fit(package, base_model, LAPTOP, LAPTOP_MACHINE)

    assert fit.mode == "gpu"
    assert fit.fit_class == "good"
    assert fit.need_gib == pytest.approx(7.9725, abs=1e-3)


def test_dense_9b_package_is_too_tight_on_server():
    # server: pool = 4.56; need ~= 7.9725; ratio = 1.748 -> too_tight.
    package = _gguf_package(weights_bytes=int(5.6 * GIB))
    base_model = _base_model(architecture=_architecture(num_hidden_layers=42, num_key_value_heads=8, head_dim=128))

    fit = compute_fit(package, base_model, SERVER, SERVER_MACHINE)

    assert fit.fit_class == "too_tight"


# --- default_context vs. package.default_context -------------------------------------------


def test_default_context_is_used_when_the_package_states_one():
    package = _gguf_package(weights_bytes=5 * GIB, default_context=4096)
    base_model = _base_model()

    fit = compute_fit(package, base_model, LAPTOP, LAPTOP_MACHINE)

    assert fit.context == 4096
    assert fit.context_assumed is False


def test_context_8192_is_assumed_when_the_package_states_none():
    package = _gguf_package(weights_bytes=5 * GIB, default_context=None)
    base_model = _base_model()

    fit = compute_fit(package, base_model, LAPTOP, LAPTOP_MACHINE)

    assert fit.context == 8192
    assert fit.context_assumed is True


# --- ratio boundaries and the CPU cap, derived from compute_fit's own need_gib ------------


def _need_gib_of(package: Package, base_model: BaseModelSpec) -> float:
    # need_gib depends only on the package/base_model, never on hardware -- read it once
    # against an oversized GPU pool so mode/class never interfere with the read.
    huge = _hardware(vram_gib=1_000_000.0, ram_gib=1_000_000.0)
    huge_machine = _machine(reserve_ram_gib=0.0, reserve_vram_gib=0.0)
    return compute_fit(package, base_model, huge, huge_machine).need_gib


@pytest.mark.parametrize(
    "ratio, expected_class",
    [
        (0.60, "perfect"),
        (0.6000001, "good"),
        (0.85, "good"),
        (0.8500001, "marginal"),
        (0.98, "marginal"),
        (0.9800001, "too_tight"),
    ],
)
def test_gpu_mode_ratio_boundaries(ratio, expected_class):
    package = _gguf_package(weights_bytes=5 * GIB, default_context=1024)
    base_model = _base_model()
    need_gib = _need_gib_of(package, base_model)

    pool_gib = need_gib / ratio
    hardware = _hardware(vram_gib=pool_gib, ram_gib=1_000_000.0)
    machine = _machine(reserve_ram_gib=0.0, reserve_vram_gib=0.0)

    fit = compute_fit(package, base_model, hardware, machine)

    assert fit.mode == "gpu"
    assert fit.fit_class == expected_class


def test_pool_at_or_below_zero_is_too_tight():
    package = _gguf_package(weights_bytes=5 * GIB, default_context=1024)
    base_model = _base_model()
    # vram_gib 0, reserve 0 -> available_vram 0, need > 0 -> falls to cpu path with ram_gib
    # equal to the reserve, so pool_gib == 0.
    hardware = _hardware(vram_gib=0.0, ram_gib=5.0)
    machine = _machine(reserve_ram_gib=5.0, reserve_vram_gib=0.0)

    fit = compute_fit(package, base_model, hardware, machine)

    assert fit.pool_gib <= 0
    assert fit.fit_class == "too_tight"


def test_cpu_mode_is_capped_at_good_even_when_the_ratio_would_be_perfect():
    package = _gguf_package(weights_bytes=5 * GIB, default_context=1024)
    base_model = _base_model()
    need_gib = _need_gib_of(package, base_model)

    # Off the GPU (vram_gib=0): a pool sized to give ratio 0.30 (well under the 0.60 "perfect"
    # boundary) must still come back "good", never "perfect".
    pool_gib = need_gib / 0.30
    hardware = _hardware(vram_gib=0.0, ram_gib=pool_gib)
    machine = _machine(reserve_ram_gib=0.0, reserve_vram_gib=0.0)

    fit = compute_fit(package, base_model, hardware, machine)

    assert fit.mode == "cpu"
    assert fit.fit_class == "good"


def test_cpu_gpu_mode_when_some_vram_exists_but_package_does_not_fit_in_it():
    package = _gguf_package(weights_bytes=50 * GIB, default_context=1024)
    base_model = _base_model()
    hardware = _hardware(vram_gib=8.0, ram_gib=64.0)
    machine = _machine(reserve_ram_gib=4.0, reserve_vram_gib=1.0)

    fit = compute_fit(package, base_model, hardware, machine)

    assert fit.mode == "cpu_gpu"
    assert fit.reserve_gib == pytest.approx(4.0)
