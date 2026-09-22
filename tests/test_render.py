"""Tests for modelroom.render: the pure Markdown document builder for `modelroom render`.

Every fixture here is built in memory (no I/O), the same convention as `tests/test_fit.py` --
`build_document` takes a `Configuration`, a `Snapshot`, a `{machine: HardwareSnapshot | None}`
mapping and `rendered_at`, and returns Markdown text. See CONTRACTS.md, "Render (AP5)", for the
header format, eligibility, selection and cell rules this file checks against.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modelroom.config import Configuration, MachineConfig
from modelroom.contracts import (
    Architecture,
    Area,
    BaseModelSpec,
    HardwareSnapshot,
    InstalledModel,
    Measurement,
    Package,
    PackageFile,
    Rating,
    Snapshot,
)
from modelroom.render import (
    RatingUnavailableError,
    build_document,
    format_header_line,
    parse_header_line,
)

NOW = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
RENDERED_AT = datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc)
GIB = 1024**3


# --- fixtures shared by every test in this module -------------------------------------------


def _architecture(*, kind: str = "dense_classic") -> Architecture:
    if kind == "unknown":
        return Architecture(source_repo="acme/Nova-8B", source_revision=None, kind="unknown")
    return Architecture(
        source_repo="acme/Nova-8B",
        source_revision=None,
        kind="dense_classic",
        num_hidden_layers=36,
        num_key_value_heads=8,
        head_dim=128,
    )


def _base_model(*, architecture: Architecture | None = None) -> BaseModelSpec:
    return BaseModelSpec(
        hf_repo="acme/Nova-8B",
        publisher="acme",
        parameters_b=8.0,
        architecture=architecture or _architecture(),
    )


def _hf_package(
    *,
    weights_bytes: int = 5 * GIB,
    quantization: str = "Q4_K_M",
    repo: str = "packager/Nova-8B-GGUF",
    revision: str = "a" * 40,
    complete: bool = True,
    active: bool = True,
    provenance: str = "metadata_ok",
    file_digest: str | None = None,
    default_context: int | None = None,
    approval=None,
) -> Package:
    files = (
        [
            PackageFile(
                name=f"Nova-8B-{quantization}.gguf",
                role="weights",
                size_bytes=weights_bytes,
                digest=file_digest,
            )
        ]
        if complete
        else []
    )
    return Package(
        source="huggingface",
        repo=repo,
        revision=revision,
        base_model_hf_repo="acme/Nova-8B",
        format="gguf",
        files=files,
        complete=complete if files else False,
        quantization=quantization if files else None,
        default_context=default_context,
        provenance=provenance,
        unresolved_reason="metadata mismatch" if provenance == "unresolved" else None,
        approval=approval,
        observed_at=NOW,
        last_seen=NOW,
        active=active,
    )


def _ollama_package(
    *,
    weights_bytes: int = 5 * GIB,
    quantization: str = "Q4_K_M",
    tag: str = "8b-q4_K_M",
    manifest_digest: str = "sha256:" + "b" * 64,
    active: bool = True,
) -> Package:
    files = [PackageFile(name=f"nova:{tag}", role="weights", size_bytes=weights_bytes, digest=None)]
    return Package(
        source="ollama",
        ollama_name=f"nova:{tag}",
        manifest_digest=manifest_digest,
        base_model_hf_repo="acme/Nova-8B",
        format="gguf",
        files=files,
        complete=True,
        quantization=quantization,
        default_context=None,
        provenance="metadata_ok",
        unresolved_reason=None,
        approval=None,
        observed_at=NOW,
        last_seen=NOW,
        active=active,
    )


def _hardware(
    *,
    vram_gib: float = 11.94,
    ram_gib: float = 127.46,
    measured_at: datetime = NOW,
    installed: list[InstalledModel] | None = None,
    installed_unavailable_reason: str | None = "not queried in this test",
    measurements: list[Measurement] | None = None,
) -> HardwareSnapshot:
    return HardwareSnapshot(
        schema_version=1,
        machine="workstation",
        measured_at=measured_at,
        llmfit_version="1.1.16",
        vram_gib=vram_gib,
        ram_gib=ram_gib,
        free_ram_gib_at_measurement=None,
        gpu_name="Nova GPU" if vram_gib else None,
        backend="CUDA" if vram_gib else None,
        unified_memory=False,
        installed=installed,
        installed_unavailable_reason=None if installed is not None else installed_unavailable_reason,
        measurements=measurements or [],
    )


def _config(machines: dict[str, MachineConfig] | None = None) -> Configuration:
    return Configuration.from_dict(
        {
            "schema_version": 1,
            "families": [
                {
                    "name": "nova",
                    "base_models": [{"hf_repo": "acme/Nova-8B"}],
                }
            ],
            "packagers": ["packager"],
            "publishers": ["acme"],
            "machines": machines
            or {"workstation": {"reserve_ram_gib": 16.0, "reserve_vram_gib": 1.0, "writer": True}},
            "paths": {"state": "//models/state", "markdown": "//models/models.md"},
        }
    )


def _snapshot(packages: list[Package], *, base_models: list[BaseModelSpec] | None = None) -> Snapshot:
    return Snapshot(
        schema_version=1,
        run_at=NOW,
        areas=[
            Area(
                source="huggingface",
                base_model_hf_repo="acme/Nova-8B",
                packager="packager",
                status="complete",
                last_success=NOW,
                error=None,
            )
        ],
        base_models=base_models or [_base_model()],
        packages=packages,
    )


def _render(packages, *, machines=None, hardware_by_machine=None, rating_source=None) -> str:
    config = _config(machines)
    snapshot = _snapshot(packages)
    hardware_by_machine = hardware_by_machine if hardware_by_machine is not None else {"workstation": _hardware()}
    return build_document(config, snapshot, hardware_by_machine, RENDERED_AT, rating_source)


# --- header line: format/parse round-trip ---------------------------------------------------


def test_header_line_round_trips():
    line = format_header_line(NOW, RENDERED_AT)
    parsed = parse_header_line(line + "\n# Model packages\n")
    assert parsed.snapshot_run_at == NOW
    assert parsed.rendered_at == RENDERED_AT


def test_parse_header_line_returns_none_for_a_file_without_the_header():
    assert parse_header_line("# Model packages\n") is None


def test_parse_header_line_returns_none_for_an_empty_file():
    assert parse_header_line("") is None


# --- F4 (fix-round 5, own finding at the AP5 acceptance): a naive header timestamp is None --
#
# `datetime.fromisoformat` happily parses a timestamp with no UTC offset; before this fix
# `parse_header_line` returned a `HeaderInfo` carrying naive datetimes, and
# `cli.py::_refusal_against_existing_document`'s `header.snapshot_run_at <= new_snapshot_run_at`
# then raised `TypeError: can't compare offset-naive and offset-aware datetimes` (the new
# snapshot's `run_at` is always aware since P2-3). CONTRACTS.md, "Header and the newer-document
# refusal", documents a header with a naive timestamp as `None`, same as no header at all.


def test_parse_header_line_returns_none_for_a_naive_snapshot_run_at():
    line = "<!-- modelroom render: snapshot_run_at=2026-09-22T09:00:00 rendered_at=2026-09-22T10:00:00+00:00 -->"
    assert parse_header_line(line + "\n") is None


def test_parse_header_line_returns_none_for_a_naive_rendered_at():
    line = "<!-- modelroom render: snapshot_run_at=2026-09-22T09:00:00+00:00 rendered_at=2026-09-22T10:00:00 -->"
    assert parse_header_line(line + "\n") is None


def test_parse_header_line_returns_none_for_a_non_utc_offset():
    """Same rule as `contracts.check_aware_utc`: aware but not UTC is still rejected."""
    line = "<!-- modelroom render: snapshot_run_at=2026-09-22T11:00:00+02:00 rendered_at=2026-09-22T10:00:00+00:00 -->"
    assert parse_header_line(line + "\n") is None


def test_build_document_starts_with_the_header_line():
    document = _render([_hf_package()])
    assert document.startswith(format_header_line(NOW, RENDERED_AT))


# --- eligibility: unresolved / incomplete / inactive are excluded ---------------------------


def test_unresolved_package_is_excluded():
    document = _render([_hf_package(provenance="unresolved")])
    assert "Q4_K_M" not in document


def test_incomplete_package_is_excluded():
    document = _render([_hf_package(complete=False)])
    packages_table = document.split("## Packages")[1]
    assert "acme/Nova-8B" not in packages_table


def test_inactive_package_is_excluded():
    document = _render([_hf_package(active=False)])
    assert "Q4_K_M" not in document


def test_approved_provenance_is_eligible():
    from modelroom.contracts import Approval

    revision = "a" * 40
    approval = Approval(date="2026-09-01", content=revision, by="acme-team")
    package = _hf_package(provenance="approved", revision=revision, approval=approval)
    document = _render([package])
    assert "Q4_K_M" in document
    assert "approved" in document


# --- selection rule ---------------------------------------------------------------------------


# A hardware/machine pair where Q3_K_S=perfect, Q4_K_M/Q5_K_M=good, Q8_0=too_tight (worked by
# hand: need_gib = weights_gib*1.10 + 1.125 (kv, L=36/KVH=8/D=128 @ context 8192) + 0.50).
_SELECTION_MACHINES = {"workstation": MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)}


def _selection_hardware() -> HardwareSnapshot:
    return _hardware(vram_gib=11.94, ram_gib=16.0)


def test_selection_picks_the_largest_good_or_better_quant():
    small = _hf_package(quantization="Q3_K_S", weights_bytes=int(3.5 * GIB), repo="packager/Nova-8B-GGUF-small")
    large_good = _hf_package(quantization="Q5_K_M", weights_bytes=int(6.0 * GIB), repo="packager/Nova-8B-GGUF-good")
    too_big = _hf_package(quantization="Q8_0", weights_bytes=int(20.0 * GIB), repo="packager/Nova-8B-GGUF-huge")
    document = _render(
        [small, large_good, too_big],
        machines=_SELECTION_MACHINES,
        hardware_by_machine={"workstation": _selection_hardware()},
    )

    assert "Q5_K_M" in document
    assert "Q3_K_S" not in document
    assert "Q8_0" not in document


def test_selection_falls_back_to_the_smallest_when_none_qualify():
    small = _hf_package(quantization="Q3_K_S", weights_bytes=int(3.5 * GIB), repo="packager/Nova-8B-GGUF-small")
    huge = _hf_package(quantization="Q8_0", weights_bytes=int(50.0 * GIB), repo="packager/Nova-8B-GGUF-huge")
    document = _render(
        [small, huge],
        machines={"workstation": MachineConfig(reserve_ram_gib=3.0, reserve_vram_gib=0.0, writer=True)},
        hardware_by_machine={"workstation": _hardware(vram_gib=0.0, ram_gib=7.56)},
    )

    assert "Q3_K_S" in document
    assert "too_tight" in document
    assert "Q8_0" not in document


def test_selection_differs_per_machine_yields_two_rows():
    small = _hf_package(quantization="Q3_K_S", weights_bytes=int(3.5 * GIB), repo="packager/Nova-8B-GGUF-small")
    large_good = _hf_package(quantization="Q5_K_M", weights_bytes=int(6.0 * GIB), repo="packager/Nova-8B-GGUF-good")
    machines = {
        "workstation": MachineConfig(reserve_ram_gib=16.0, reserve_vram_gib=1.0, writer=True),
        "tiny-server": MachineConfig(reserve_ram_gib=3.0, reserve_vram_gib=0.0, writer=False),
    }
    hardware_by_machine = {
        "workstation": _hardware(vram_gib=11.94, ram_gib=127.46),
        "tiny-server": _hardware(vram_gib=0.0, ram_gib=7.56),
    }
    document = _render([small, large_good], machines=machines, hardware_by_machine=hardware_by_machine)

    assert "Q5_K_M" in document
    assert "Q3_K_S" in document


def test_variants_count_excludes_packages_not_shown_as_a_row():
    small = _hf_package(quantization="Q3_K_S", weights_bytes=int(3.5 * GIB), repo="packager/Nova-8B-GGUF-small")
    mid = _hf_package(quantization="Q4_K_M", weights_bytes=int(5.0 * GIB), repo="packager/Nova-8B-GGUF-mid")
    large_good = _hf_package(quantization="Q5_K_M", weights_bytes=int(6.0 * GIB), repo="packager/Nova-8B-GGUF-good")
    # All three qualify (perfect/good/good) on this hardware -> exactly one row (Q5_K_M, the
    # largest qualifying quant), the other two eligible packages become "+2 more".
    document = _render(
        [small, mid, large_good],
        machines=_SELECTION_MACHINES,
        hardware_by_machine={"workstation": _selection_hardware()},
    )

    assert "+2 more" in document


# --- F5 (fix-round 5): a group fit v1 cannot judge at all makes no pick, never a useless one --
#
# Before F5, `_select_for_machine` fell back to "the smallest package" whenever nothing
# qualified as good/perfect -- including when fit v1 could not judge *any* package of the group
# (`fit_class == "unknown"` for every one, e.g. a whole architecture fit v1 does not cover, the
# real Qwen3.5 case measured 2026-09-22: every package came back unknown, and the old fallback
# still picked an arbitrary smallest quant such as `UD-IQ2_XXS` with `+21 more`, a recommendation
# with no basis at all). The new rule: a machine with no judged package (all unknown, or no
# hardware profile) makes no pick; if no machine picks anything for the group, it renders one
# row with every package cell "–" except the fit cells ("no recommendation: <reason>") and
# Variants ("<N> variants, none judged"). The "else the smallest" fallback still applies, but
# only across the *judged* packages, when at least one is judged and none reach good/perfect.


def test_group_entirely_unknown_renders_one_no_recommendation_row():
    small = _hf_package(quantization="Q3_K_S", weights_bytes=int(3.5 * GIB), repo="packager/Nova-8B-GGUF-small")
    mid = _hf_package(quantization="Q4_K_M", weights_bytes=int(5.0 * GIB), repo="packager/Nova-8B-GGUF-mid")
    large = _hf_package(quantization="Q5_K_M", weights_bytes=int(6.0 * GIB), repo="packager/Nova-8B-GGUF-good")
    base_model = _base_model(architecture=_architecture(kind="unknown"))
    # `_render`'s helper always attaches the default dense_classic `_base_model()` -- build
    # directly so the group's base model is the unknown-architecture one instead.
    config = _config(_SELECTION_MACHINES)
    snapshot = _snapshot([small, mid, large], base_models=[base_model])
    document = build_document(config, snapshot, {"workstation": _selection_hardware()}, RENDERED_AT)

    assert "no recommendation: architecture not covered by v1" in document
    assert "3 variants, none judged" in document
    assert "Q3_K_S" not in document
    assert "Q4_K_M" not in document
    assert "Q5_K_M" not in document


def test_group_with_one_judged_bad_package_among_unknowns_picks_the_judged_one():
    unknown_a = _hf_package(
        quantization="Q3_K_S", weights_bytes=0, repo="packager/Nova-8B-GGUF-unknown-a"
    )  # a weight file with no size -> unknown, regardless of its (smallest) quant label
    unknown_b = _hf_package(
        quantization="Q4_K_M", weights_bytes=0, repo="packager/Nova-8B-GGUF-unknown-b"
    )
    judged_bad = _hf_package(
        quantization="Q8_0", weights_bytes=int(50.0 * GIB), repo="packager/Nova-8B-GGUF-huge"
    )  # judged (too_tight) on the tiny hardware below -- the only package fit v1 could judge
    document = _render(
        [unknown_a, unknown_b, judged_bad],
        machines={"workstation": MachineConfig(reserve_ram_gib=3.0, reserve_vram_gib=0.0, writer=True)},
        hardware_by_machine={"workstation": _hardware(vram_gib=0.0, ram_gib=7.56)},
    )

    assert "Q8_0" in document
    assert "too_tight" in document
    assert "no recommendation" not in document
    assert "+2 more" in document


# --- fit cell: unknown never prints the zeroed numbers --------------------------------------


def test_fit_cell_for_unknown_architecture_never_prints_numbers():
    """F5 (fix-round 5): every package in the group is unknown here (architecture-level, so it
    applies to every package regardless of its own properties) -- the group makes no pick at
    all and renders the "no recommendation" row (see the F5 section below), never a fit cell
    quoting the zeroed placeholder numbers `compute_fit` returns for the unknown case.
    """
    package = _hf_package()
    base_model = _base_model(architecture=_architecture(kind="unknown"))
    config = _config()
    snapshot = _snapshot([package], base_models=[base_model])
    document = build_document(config, snapshot, {"workstation": _hardware()}, RENDERED_AT)

    assert "no recommendation: architecture not covered by v1" in document
    assert "need 0.0" not in document
    assert "pool 0.0" not in document


def test_fit_cell_no_profile_when_machine_has_no_hardware():
    document = _render([_hf_package()], hardware_by_machine={"workstation": None})
    assert "no profile" in document


# --- installed cell ----------------------------------------------------------------------------


def test_installed_cell_yes_when_digest_matches():
    digest = "sha256:" + "c" * 64
    package = _ollama_package(manifest_digest=digest)
    installed = [InstalledModel(name="nova:8b-q4_K_M", digest=digest, size_bytes=5 * GIB, observed_at=NOW)]
    document = _render([package], hardware_by_machine={"workstation": _hardware(installed=installed)})
    assert "workstation: yes" in document


def test_installed_cell_no_when_digest_does_not_match():
    package = _ollama_package(manifest_digest="sha256:" + "c" * 64)
    installed = [InstalledModel(name="nova:other", digest="sha256:" + "d" * 64, size_bytes=5 * GIB, observed_at=NOW)]
    document = _render([package], hardware_by_machine={"workstation": _hardware(installed=installed)})
    assert "workstation: no" in document


def test_installed_cell_unknown_when_installed_is_none():
    package = _ollama_package()
    document = _render([package], hardware_by_machine={"workstation": _hardware(installed=None)})
    assert "workstation: unknown" in document


def test_installed_cell_dash_for_a_huggingface_package():
    package = _hf_package()
    document = _render([package])
    assert "workstation: –" in document


# --- speed cell ---------------------------------------------------------------------------------


def test_speed_cell_shown_for_a_matching_current_measurement():
    digest = "sha256:" + "c" * 64
    package = _ollama_package(manifest_digest=digest)
    measurement = Measurement(
        content_source="ollama",
        ollama_manifest_digest=digest,
        hf_repo=None,
        hf_revision=None,
        hf_file_digest=None,
        context=8192,
        runtime="ollama 0.12.3",
        profile_measured_at=NOW,
        measured_at=NOW,
        tps_mean=42.5,
        tps_range=(40.0, 45.0),
    )
    document = _render([package], hardware_by_machine={"workstation": _hardware(measurements=[measurement])})
    assert "42.5 tps @8192 (workstation)" in document


def test_speed_cell_dash_for_a_stale_measurement():
    digest = "sha256:" + "c" * 64
    package = _ollama_package(manifest_digest=digest)
    stale_measurement = Measurement(
        content_source="ollama",
        ollama_manifest_digest=digest,
        hf_repo=None,
        hf_revision=None,
        hf_file_digest=None,
        context=8192,
        runtime="ollama 0.12.3",
        profile_measured_at=NOW - timedelta(days=1),  # profile has since changed
        measured_at=NOW,
        tps_mean=42.5,
        tps_range=(40.0, 45.0),
    )
    document = _render([package], hardware_by_machine={"workstation": _hardware(measurements=[stale_measurement])})
    assert "42.5 tps" not in document


def test_speed_cell_dash_when_no_measurement_matches():
    package = _ollama_package()
    document = _render([package])
    row = next(line for line in document.splitlines() if line.startswith("| acme/Nova-8B"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    speed_cell_index = 9  # base model, stars, packager, quant, format, size, context, fit, installed, speed
    assert cells[speed_cell_index] == "–"


# --- stars / rating ------------------------------------------------------------------------------


def test_stars_cell_renders_half_stars():
    document = _render([_hf_package()], rating_source=lambda repo: Rating(stars=3.5, source="market index"))
    assert "★★★½" in document


def test_stars_cell_dash_when_source_returns_none():
    document = _render([_hf_package()], rating_source=lambda repo: None)
    assert "Market rating unavailable" not in document


def test_stars_cell_dash_and_no_note_when_no_source_given():
    document = _render([_hf_package()], rating_source=None)
    assert "Market rating unavailable" not in document


def test_rating_source_raising_unavailable_error_adds_a_note_and_still_renders():
    def _failing(repo: str):
        raise RatingUnavailableError("market index is down")

    document = _render([_hf_package()], rating_source=_failing)
    assert "Market rating unavailable: market index is down" in document
    assert "Q4_K_M" in document


def test_rating_source_raising_any_exception_is_treated_the_same():
    def _failing(repo: str):
        raise RuntimeError("boom")

    document = _render([_hf_package()], rating_source=_failing)
    assert "Market rating unavailable: boom" in document


# --- machines without a profile ----------------------------------------------------------------


def test_machine_without_a_profile_renders_no_hardware_profile_yet():
    document = _render([_hf_package()], hardware_by_machine={"workstation": None})
    assert "no hardware profile yet" in document
    assert "no profile" in document  # the package row's fit cell
