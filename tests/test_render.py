"""Tests for modelroom.render: the schema-2 render document and its three writers.

Every fixture here is built in memory (no I/O), the same convention as `tests/test_fit.py`:
`build_render_document` takes a `Configuration`, a `Snapshot`, one `MachineProfile` per
configured machine, that profile's measurement records, a `Scenario` and `rendered_at`, and
returns the one `RenderDocument` all three writers read. See CONTRACTS.md, "Render (schema 2)".
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from modelroom.config import Configuration, MachineConfig
from modelroom.contracts import (
    Architecture,
    Area,
    BaseModelSpec,
    HardwareSnapshot,
    Package,
    PackageFile,
    Rating,
    Snapshot,
)
from modelroom.document import RenderDocument
from modelroom.importer import ProfileScan
from modelroom.measurements import MeasurementRecord, PackageRef, RunCounters, Scenario, default_scenario
from modelroom.profile import CrossCheck, HardwareProfile, LlmfitCrosscheck, fit_block_reason, normalize_profile_v1
from modelroom.ranking import RANKING_RULE, TOP_LIMIT
from modelroom.render import (
    LEGACY_FIT_REASON,
    NO_PROFILE_REASON,
    MachineProfile,
    RatingUnavailableError,
    build_render_document,
    document_json,
    format_header_line,
    machine_profile,
    parse_header_line,
)
from modelroom.views import _cell, document_markdown, document_terminal

NOW = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
RENDERED_AT = datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc)
GIB = 1024**3
PROFILE_ID = "3f9a0c21d4e6b870"


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


def _profile(*, vram_gib: float = 11.94, ram_gib: float = 127.46, profile_id: str = PROFILE_ID) -> HardwareProfile:
    absent = CrossCheck(status="absent")
    return HardwareProfile(
        schema_version=2,
        profile_id=profile_id,
        display_name="workstation",
        os_fingerprint="9d2f4b6a8c0e1357",
        os_fingerprint_source="windows_machineguid",
        origin="measured",
        recorded_at=NOW,
        ram_physical_gib=ram_gib,
        ram_physical_source="os",
        ram_limit_gib=None,
        ram_limit_scope="none",
        vram_gib=vram_gib,
        vram_source="nvidia-smi" if vram_gib else "none",
        gpu_state="measured" if vram_gib else "none",
        gpu_name="Nova GPU" if vram_gib else None,
        llmfit_crosscheck=LlmfitCrosscheck(ram_physical=absent, vram=absent),
        llmfit_version=None,
    )


def _legacy_snapshot() -> HardwareSnapshot:
    return HardwareSnapshot(
        schema_version=1,
        machine="workstation",
        measured_at=NOW,
        llmfit_version="1.1.16",
        vram_gib=11.94,
        ram_gib=127.46,
        free_ram_gib_at_measurement=None,
        gpu_name="Nova GPU",
        backend="CUDA",
        unified_memory=False,
        installed=None,
        installed_unavailable_reason="not queried in this test",
        measurements=[],
    )


def _measurement(package: Package, *, tps: float = 50.0, context: int = 8192, token: str = "4da3b585"):
    eval_duration = round(128 / tps * 1e9)
    run = RunCounters(
        done_reason="length",
        eval_count=128,
        eval_duration=eval_duration,
        prompt_eval_count=42,
        prompt_eval_duration=90000000,
        load_duration=12000000,
    )
    measured_at = datetime(2026, 9, 22, 20, 20, 0, tzinfo=timezone.utc)
    digest = next(f.digest for f in package.files if f.digest)
    return MeasurementRecord(
        schema_version=2,
        measurement_id=f"{measured_at.strftime('%Y%m%dT%H%M%SZ')}-{token}",
        profile_id=PROFILE_ID,
        protocol="v1",
        measured_at=measured_at,
        package=PackageRef(
            content_source="huggingface",
            hf_repo=package.repo,
            hf_revision=package.revision,
            hf_file_digest=digest,
        ),
        scenario=Scenario(
            context_requested=context,
            context_origin="default" if context == 8192 else "entered",
            kv_type="f16",
            kv_type_assumed=True,
            requests=1,
        ),
        runs=[run, run, run],
        tps_mean=run.tokens_per_second(),
        tps_min=run.tokens_per_second(),
        tps_max=run.tokens_per_second(),
        validity="valid",
        validity_reason=None,
        comparable=True,
        comparable_reason=None,
    )


def _config(machines: dict[str, MachineConfig | dict] | None = None) -> Configuration:
    return Configuration.from_dict(
        {
            "schema_version": 2,
            "families": [{"name": "nova", "base_models": [{"hf_repo": "acme/Nova-8B"}]}],
            "packagers": ["packager"],
            "publishers": ["acme"],
            "machines": machines
            or {
                "workstation": {
                    "reserve_ram_gib": 16.0,
                    "reserve_vram_gib": 1.0,
                    "writer": True,
                    "profile": PROFILE_ID,
                }
            },
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
        base_models=[_base_model()] if base_models is None else base_models,
        packages=packages,
    )


def _ready(profile: HardwareProfile | None = None) -> MachineProfile:
    found = profile or _profile()
    return MachineProfile(status="ranked", label=found.display_name, profile=found, reason=None)


def _document(
    packages,
    *,
    machines=None,
    profiles=None,
    measurements=None,
    scenario: Scenario | None = None,
    rating_source=None,
    base_models=None,
) -> RenderDocument:
    config = _config(machines)
    return build_render_document(
        config,
        _snapshot(packages, base_models=base_models),
        profiles if profiles is not None else {"workstation": _ready()},
        measurements or {},
        scenario or default_scenario(),
        RENDERED_AT,
        rating_source,
    )


# --- header line: format/parse round-trip ---------------------------------------------------


def test_header_line_round_trips():
    line = format_header_line(NOW, RENDERED_AT)
    header = parse_header_line(line + "\n\n# Model packages\n")
    assert header is not None
    assert (header.snapshot_run_at, header.rendered_at) == (NOW, RENDERED_AT)


def test_parse_header_line_returns_none_for_a_file_without_the_header():
    assert parse_header_line("# Model packages\n") is None


def test_parse_header_line_returns_none_for_an_empty_file():
    assert parse_header_line("") is None


def test_parse_header_line_returns_none_for_a_naive_timestamp():
    naive = "<!-- modelroom render: snapshot_run_at=2026-09-22T09:00:00 rendered_at=2026-09-22T10:00:00+00:00 -->"
    assert parse_header_line(naive) is None


def test_parse_header_line_returns_none_for_a_non_utc_offset():
    shifted = (
        "<!-- modelroom render: snapshot_run_at=2026-09-22T09:00:00+02:00 "
        "rendered_at=2026-09-22T10:00:00+00:00 -->"
    )
    assert parse_header_line(shifted) is None


def test_markdown_starts_with_the_header_line():
    text = document_markdown(_document([_hf_package()]))
    assert text.splitlines()[0] == format_header_line(NOW, RENDERED_AT)


# --- eligibility ----------------------------------------------------------------------------


def _ranked_identities(document: RenderDocument) -> list[tuple]:
    return [tuple(entry.package_identity) for entry in document.machines[0].ranked]


def test_unresolved_package_is_excluded():
    assert _ranked_identities(_document([_hf_package(provenance="unresolved")])) == []


def test_incomplete_package_is_excluded():
    assert _ranked_identities(_document([_hf_package(complete=False)])) == []


def test_inactive_package_is_excluded():
    assert _ranked_identities(_document([_hf_package(active=False)])) == []


def test_approved_provenance_is_eligible():
    approved = _hf_package(
        provenance="approved",
        approval={"date": "2026-09-01", "content": "a" * 40, "by": "acme-ai-team"},
    )
    assert len(_ranked_identities(_document([approved]))) == 1


# --- the ranking ------------------------------------------------------------------------------


def test_every_eligible_package_is_ranked_not_just_one_per_packager():
    packages = [
        _hf_package(quantization="Q4_K_M", weights_bytes=5 * GIB),
        _hf_package(quantization="Q8_0", weights_bytes=8 * GIB),
        _hf_package(quantization="Q2_K", weights_bytes=3 * GIB),
    ]
    block = _document(packages).machines[0]
    assert [entry.rank for entry in block.ranked] == [1, 2, 3]
    assert {entry.quantization for entry in block.ranked} == {"Q4_K_M", "Q8_0", "Q2_K"}


def test_fit_class_beats_measured_speed():
    fits_gpu = _hf_package(quantization="Q4_K_M", weights_bytes=5 * GIB, file_digest="sha256:" + "c" * 64)
    needs_ram = _hf_package(quantization="Q8_0", weights_bytes=90 * GIB, file_digest="sha256:" + "d" * 64)
    measurements = {PROFILE_ID: [_measurement(needs_ram, tps=200.0)]}
    block = _document([fits_gpu, needs_ram], measurements=measurements).machines[0]
    assert block.ranked[0].quantization == "Q4_K_M"
    assert block.ranked[0].measurement_group == 1
    assert block.ranked[1].measurement_group == 0
    assert block.ranked[1].speed_tps == pytest.approx(200.0)


def test_a_measurement_of_another_context_does_not_count():
    package = _hf_package(file_digest="sha256:" + "c" * 64)
    measurements = {PROFILE_ID: [_measurement(package, context=4096)]}
    block = _document([package], measurements=measurements).machines[0]
    assert block.ranked[0].measurement_group == 1
    assert block.ranked[0].speed_tps is None


def test_measured_entry_carries_a_measured_note():
    package = _hf_package(file_digest="sha256:" + "c" * 64)
    measurements = {PROFILE_ID: [_measurement(package, tps=50.0)]}
    note = _document([package], measurements=measurements).machines[0].ranked[0].note
    assert note.origin == "measured"
    assert "50.0" in note.text
    assert note.facts == ["measurement.tps_mean", "measurement.scenario.context_requested"]


def test_an_architecture_fit_v1_cannot_judge_lands_in_not_covered():
    document = _document([_hf_package()], base_models=[_base_model(architecture=_architecture(kind="unknown"))])
    block = document.machines[0]
    assert block.ranked == []
    assert [entry.reason for entry in block.not_covered] == ["architecture not covered by v1"]
    assert block.not_covered[0].note.code == "not_covered"


def test_a_package_that_does_not_fit_lands_in_too_tight_and_never_in_the_ranking():
    huge = _hf_package(weights_bytes=400 * GIB)
    block = _document([huge]).machines[0]
    assert block.ranked == []
    assert len(block.too_tight) == 1
    assert block.too_tight[0].note.code == "too_tight"


def test_the_too_tight_note_claims_no_overrun_that_did_not_happen():
    """Fit v1 sets `too_tight` above 98 % of the pool, so the need can be below the pool.

    Found by the second-model review: the note used to say "more than the machine offers after
    the reserve", which is false for every package between 98 % and 100 % of the pool.
    """
    tight = _hf_package(weights_bytes=int(98.8 * GIB))
    entry = _document([tight]).machines[0].too_tight[0]

    assert entry.fit.fit_class == "too_tight"
    assert entry.fit.need_gib < entry.fit.pool_gib
    assert "more than" not in entry.note.text
    assert f"{entry.fit.need_gib:.1f}" in entry.note.text
    assert f"{entry.fit.pool_gib:.1f}" in entry.note.text


def test_the_ranking_is_capped_at_ten_and_reports_the_total():
    packages = [
        _hf_package(quantization=quant, weights_bytes=(3 + index) * GIB, repo=f"packager/Nova-8B-{index}-GGUF")
        for index, quant in enumerate(
            ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]
        )
    ]
    block = _document(packages).machines[0]
    assert len(block.ranked) == TOP_LIMIT
    assert block.ranked_total == len(packages)


def test_the_context_comes_from_the_scenario_not_from_the_package():
    package = _hf_package(default_context=4096)
    block = _document([package]).machines[0]
    assert block.ranked[0].fit.context == 8192
    assert block.ranked[0].fit.context_assumed is False
    assert block.ranked[0].package_context == 4096


# --- which profile a machine is rendered from -------------------------------------------------


def test_machine_profile_reads_the_configured_schema_two_profile():
    profile = _profile()
    scan = ProfileScan(profiles={PROFILE_ID: profile})
    found = machine_profile("workstation", _config().machines["workstation"], scan)
    assert found == MachineProfile(status="ranked", label="workstation", profile=profile, reason=None)


def test_machine_profile_without_a_configured_profile_is_no_profile():
    machines = {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}}
    found = machine_profile("workstation", _config(machines).machines["workstation"], ProfileScan())
    assert (found.status, found.profile, found.reason) == ("no_profile", None, NO_PROFILE_REASON)


def test_machine_profile_finds_a_schema_one_file_under_the_machine_name(tmp_path):
    machines = {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}}
    scan = ProfileScan(legacy=[tmp_path / "workstation.json"])
    found = machine_profile("workstation", _config(machines).machines["workstation"], scan)
    assert found.status == "legacy"
    assert found.label == "workstation.json"
    assert found.reason == LEGACY_FIT_REASON


def test_the_legacy_reason_is_the_fit_rule_s_own_reason():
    normalized = normalize_profile_v1(_legacy_snapshot(), PROFILE_ID)
    assert LEGACY_FIT_REASON == fit_block_reason(normalized)


def test_machine_profile_reports_a_configured_profile_that_is_not_in_the_folder():
    found = machine_profile("workstation", _config().machines["workstation"], ProfileScan())
    assert found.status == "no_profile"
    assert PROFILE_ID in found.reason


def test_machine_profile_reports_an_unreadable_profile_file(tmp_path):
    scan = ProfileScan(unreadable=[(tmp_path / f"{PROFILE_ID}.json", "not valid JSON")])
    found = machine_profile("workstation", _config().machines["workstation"], scan)
    assert found.status == "no_profile"
    assert "not valid JSON" in found.reason


def test_a_machine_without_a_profile_shows_its_reason_and_no_entries():
    profiles = {"workstation": MachineProfile("no_profile", "workstation", None, NO_PROFILE_REASON)}
    block = _document([_hf_package()], profiles=profiles).machines[0]
    assert (block.status, block.ranked, block.not_covered, block.too_tight) == ("no_profile", [], [], [])
    assert block.reason == NO_PROFILE_REASON


def test_a_legacy_machine_is_shown_as_legacy_with_the_fit_reason():
    profiles = {"workstation": MachineProfile("legacy", "workstation.json", None, LEGACY_FIT_REASON)}
    block = _document([_hf_package()], profiles=profiles).machines[0]
    assert block.status == "legacy"
    assert block.reason == LEGACY_FIT_REASON
    assert document_markdown(_document([_hf_package()], profiles=profiles)).count("legacy") >= 1


# --- stars / rating ---------------------------------------------------------------------------


def test_stars_are_collected_once_per_base_model():
    asked: list[str] = []

    def source(repo: str) -> Rating:
        asked.append(repo)
        return Rating(stars=3.5, source="market index")

    document = _document([_hf_package(), _hf_package(quantization="Q8_0")], rating_source=source)
    assert asked == ["acme/Nova-8B"]
    assert document.ratings["acme/Nova-8B"].stars == 3.5
    assert "★★★½" in document_markdown(document)


def test_a_rating_source_that_fails_leaves_a_note_and_no_ratings():
    def source(repo: str) -> Rating:
        raise RatingUnavailableError("index unreachable")

    document = _document([_hf_package()], rating_source=source)
    assert document.rating_unavailable == "index unreachable"
    assert document.ratings == {}
    assert "Market rating unavailable: index unreachable" in document_markdown(document)


def test_any_exception_from_the_rating_source_is_treated_the_same():
    def source(repo: str) -> Rating:
        raise RuntimeError("boom")

    assert _document([_hf_package()], rating_source=source).rating_unavailable == "boom"


def test_no_rating_source_means_no_stars_and_no_note():
    document = _document([_hf_package()])
    assert (document.ratings, document.rating_unavailable) == ({}, None)
    assert "Market rating unavailable" not in document_markdown(document)


# --- the three writers agree ------------------------------------------------------------------


def test_json_is_the_document_itself():
    package = _hf_package(file_digest="sha256:" + "c" * 64)
    document = _document([package], measurements={PROFILE_ID: [_measurement(package)]})
    payload = document_json(document)
    assert RenderDocument.model_validate(payload) == document


def test_markdown_and_json_show_the_same_speed_and_fit():
    package = _hf_package(file_digest="sha256:" + "c" * 64)
    document = _document([package], measurements={PROFILE_ID: [_measurement(package, tps=50.0)]})
    entry = document.machines[0].ranked[0]
    row = next(line for line in document_markdown(document).splitlines() if line.startswith("| 1 |"))
    assert f"{entry.speed_tps:.1f}" in row
    assert entry.fit.fit_class in row
    assert f"{entry.fit.need_gib:.2f}" in row


def test_every_writer_names_the_rule_and_the_scenario():
    document = _document([_hf_package()])
    for text in (document_markdown(document), document_terminal(document)):
        assert RANKING_RULE in text
        assert "context 8192" in text
        assert "f16" in text


def test_the_terminal_view_lists_the_ranking_and_the_blocks():
    huge = _hf_package(weights_bytes=400 * GIB, repo="packager/Nova-8B-XL-GGUF")
    text = document_terminal(_document([_hf_package(), huge]))
    assert "Ranking: workstation" in text
    assert "too tight" in text.lower()


# --- Markdown cell escaping (R7-11) -------------------------------------------------------------


def test_cell_escapes_pipes_and_collapses_newlines():
    assert _cell("boom | extra\nsecond line") == "boom \\| extra second line"


def test_cell_collapses_crlf_and_lone_cr():
    assert _cell("a\r\nb\rc") == "a b c"


def test_an_area_error_with_a_pipe_stays_one_row():
    snapshot = _snapshot([_hf_package()])
    broken = snapshot.model_copy(
        update={
            "areas": [
                snapshot.areas[0].model_copy(update={"status": "incomplete", "error": "boom | extra\nsecond line"})
            ]
        }
    )
    document = build_render_document(
        _config(), broken, {"workstation": _ready()}, {}, default_scenario(), RENDERED_AT, None
    )
    text = document_markdown(document)
    area_rows = [line for line in text.splitlines() if line.startswith("| huggingface |")]
    assert len(area_rows) == 1
    assert len(re.findall(r"(?<!\\)\|", area_rows[0])) == 7
