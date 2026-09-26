"""Tests for modelroom.render: the schema-2 render document and its three writers.

Every fixture here is built in memory (no I/O), the same convention as `tests/test_fit.py`:
`build_render_document` takes a `Configuration`, a `Snapshot`, one `MachineProfile` per
configured machine, that profile's measurement records, a `Scenario` and `rendered_at`, and
returns the one `RenderDocument` all three writers read. See CONTRACTS.md, "Render (schema 2)".
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

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
from modelroom.profile import (
    PROFILE_SCHEMA_VERSION,
    CrossCheck,
    HardwareProfile,
    LlmfitCrosscheck,
    fit_block_reason,
    normalize_profile_v1,
)
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
from modelroom.fit import OLLAMA_REQUEST_HINT_TEXT
from modelroom.screen import MEASURE_HINT
from modelroom.views import _cell, document_markdown, document_terminal, result_card, scenario_line

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
        schema_version=PROFILE_SCHEMA_VERSION,
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


def test_an_architecture_fit_v1_cannot_judge_is_ranked_from_the_size():
    """Not set aside any more (decided 2026-09-24): the size of the package answers instead."""
    document = _document([_hf_package()], base_models=[_base_model(architecture=_architecture(kind="unknown"))])
    block = document.machines[0]
    assert block.not_covered == []
    assert [entry.fit.basis for entry in block.ranked] == ["size"]
    # The pool's own sentence stays in front (acceptance of 2026-09-24: a row in system memory read
    # as the equal of one in graphics memory without it), the basis is the sentence after it.
    assert block.ranked[0].note.text.startswith("Fits into graphics memory:")
    assert block.ranked[0].note.text.endswith("From the size of the package, not its architecture.")
    assert block.ranked[0].note.facts == ["fit.mode", "fit.need_gib", "fit.pool_gib", "fit.basis"]


def test_a_package_set_aside_on_the_size_basis_names_that_basis_in_its_note():
    """The Markdown reader sees the basis of a `too_tight` row in its note (second-model round)."""
    unknown = [_base_model(architecture=_architecture(kind="unknown"))]
    huge = _hf_package(weights_bytes=400 * GIB)

    entry = _document([huge], base_models=unknown).machines[0].too_tight[0]

    assert entry.fit.basis == "size"
    assert entry.note.text.endswith("From the size of the package, not its architecture.")
    assert "fit.basis" in entry.note.facts


def test_a_weight_file_without_a_size_is_still_not_covered():
    """The size basis needs a size: a package the fetcher learned none for is no size at all."""
    block = _document([_hf_package(weights_bytes=0)]).machines[0]
    assert block.ranked == []
    assert [entry.reason for entry in block.not_covered] == ["a weight file has no size"]
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


def test_the_markdown_view_names_the_rule_and_the_scenario():
    text = document_markdown(_document([_hf_package()]))

    assert RANKING_RULE in text
    assert "context 8192" in text
    assert "f16" in text


def test_the_head_of_a_machine_says_the_context_and_leaves_the_rule_to_the_file():
    """A terminal is narrow: the rule, the snapshot time and the scenario stay in the Markdown view."""
    large = Scenario(
        context_requested=32768, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1
    )
    text = document_terminal(_document([_hf_package()], scenario=large))

    assert "workstation · context L 32k" in text
    assert RANKING_RULE not in text
    assert "Snapshot run at" not in text
    assert "KV cache" not in text


def test_the_terminal_table_reads_as_the_mockup_does():
    """Model, package, fit, speed and memory in plain words -- the columns of the mockup."""
    package = _hf_package(file_digest="sha256:" + "c" * 64)
    document = _document([package], measurements={PROFILE_ID: [_measurement(package, tps=41.1)]})

    lines = document_terminal(document).splitlines()

    head = next(line for line in lines if "Model" in line and "Package" in line)
    assert head.split() == ["#", "Model", "Package", "Fit", "Speed", "Memory"]
    row = lines[lines.index(head) + 2]
    entry = document.machines[0].ranked[0]
    assert row.split() == ["1", "Nova-8B", "packager", "·", "Q4_K_M", entry.fit.fit_class, "41.1", "tok/s", f"{entry.fit.need_gib:.1f}", "GB"]


def test_a_fit_in_system_memory_says_ram_and_never_from_size():
    """The same word the selection list of step 2 uses; where the size came from is a note."""
    unknown = [_base_model(architecture=_architecture(kind="unknown"))]
    # 90 GiB of weights: need 100.6 GiB against the system-memory pool of 111.46, ratio 0.90.
    marginal = _hf_package(weights_bytes=90 * GIB)

    text = document_terminal(_document([marginal], base_models=unknown))

    assert "marginal (RAM)" in text
    assert "basis     #1 computed from the package size" in text  # a note under the table, not the cell
    assert "marginal (from size)" not in text


def test_a_fit_in_graphics_memory_carries_no_suffix():
    text = document_terminal(_document([_hf_package()]))

    assert re.search(r"\bgood\s{2,}", text)
    assert "(RAM)" not in text


def test_a_row_without_a_measurement_shows_a_dash_and_never_unknown():
    """`unknown` in every row of the table said nothing twelve times over (test round, 2026-09-24)."""
    lines = document_terminal(_document([_hf_package()])).splitlines()

    row = next(line for line in lines if line.strip().startswith("1 "))
    assert "unknown" not in row
    assert "–" in row


def test_no_line_of_the_terminal_table_is_wider_than_a_hundred_characters():
    packages = [_hf_package(repo=f"packager-with-a-long-name/Nova-8B-{index}-GGUF") for index in range(3)]

    lines = document_terminal(_document(packages)).splitlines()

    for line in lines:
        assert len(line) <= 100, line


def test_a_machine_that_is_not_ranked_keeps_its_status_next_to_its_name():
    profiles = {"workstation": MachineProfile(status="no_profile", label="workstation", profile=None, reason="no profile yet")}

    text = document_terminal(_document([_hf_package()], profiles=profiles))

    assert "workstation · no_profile" in text
    assert "no profile yet" in text


def test_a_ranked_machine_is_named_without_its_status():
    text = document_terminal(_document([_hf_package()]))

    assert "workstation · context" in text
    assert "ranked" not in text.splitlines()[2]


def test_the_terminal_view_lists_the_ranking_and_the_blocks():
    huge = _hf_package(weights_bytes=400 * GIB, repo="packager/Nova-8B-XL-GGUF")
    text = document_terminal(_document([_hf_package(), huge]))
    assert "workstation" in text
    assert "too tight" in text.lower()


def test_the_terminal_view_carries_the_head_of_the_last_step():
    """It is what step 5 of the guided mode shows, and the only caller that asks for it."""
    head = document_terminal(_document([_hf_package()])).splitlines()[0]

    assert head.strip().startswith("── Step 5 of 5  Results ")
    assert len(head.strip()) == 72


def test_every_line_under_the_table_carries_the_label_of_what_it_says():
    """Running text under the table buried the install command (test round, 2026-09-25).

    The labels are the card's own column, so `shown`, `memory`, `basis`, `speed` and `install`
    stand in the same place as `folder` and `result` above them.
    """
    unknown = [_base_model(architecture=_architecture(kind="unknown"))]
    packages = [_hf_package(repo=f"packager/Nova-8B-{index}-GGUF") for index in range(3)]

    lines = document_terminal(_document(packages, base_models=unknown), install_for="workstation").splitlines()

    labels = [line.split()[0] for line in lines if line.startswith(" ") and not line.startswith("  ")]
    for label in ("shown", "memory", "basis", "speed", "install"):
        assert label in labels, label
    for label in ("shown", "memory", "basis", "speed", "install"):
        line = next(line for line in lines if line.startswith(f" {label}"))
        assert line.index(line.split()[1]) == 11, line


def test_the_notes_of_the_ranking_are_bundled_by_memory_pool_with_their_rank_ranges():
    """Ten rows carried ten notes of the same two sentences (test round, 2026-09-24)."""
    packages = [_hf_package(repo=f"packager/Nova-8B-{index}-GGUF") for index in range(3)]

    lines = document_terminal(_document(packages)).splitlines()

    pool = next(line for line in lines if "graphics memory" in line)
    assert pool.split(maxsplit=1)[0] == "memory"
    assert "#1–3 fit into graphics memory, " in pool
    assert pool.rstrip().endswith("GB free after the reserve")
    assert len([line for line in lines if "graphics memory" in line]) == 1


def test_a_second_memory_pool_is_a_line_of_its_own_under_the_first():
    """Two statements about two pools are two lines; only the first carries the label."""
    unknown = [_base_model(architecture=_architecture(kind="unknown"))]
    # 90 GiB of weights do not fit the graphics card; the small one does.
    packages = [_hf_package(), _hf_package(repo="packager/Nova-8B-XL-GGUF", weights_bytes=90 * GIB)]

    lines = [line for line in document_terminal(_document(packages, base_models=unknown)).splitlines() if "memory" in line]

    assert lines[0].split(maxsplit=1)[0] == "memory"
    assert lines[1].startswith(" " * 11)
    assert lines[1].strip() == "#2 needs system memory, the graphics card helps"


def test_a_single_rank_is_named_in_the_singular():
    lines = document_terminal(_document([_hf_package()])).splitlines()

    assert any("#1 fits into graphics memory, " in line for line in lines)


def test_the_speed_line_says_what_was_measured_or_that_nothing_was():
    package = _hf_package(file_digest="sha256:" + "c" * 64)
    nothing = document_terminal(_document([package]))
    measured = document_terminal(_document([package], measurements={PROFILE_ID: [_measurement(package, tps=41.1)]}))

    assert "speed     nothing measured in the rows shown · say Yes in step 4 to measure an installed package" in nothing
    assert "speed     #1 measured 41.1 tok/s" in measured
    assert "nothing measured yet" not in measured


def test_the_basis_line_stands_only_where_a_fit_was_computed_from_the_size():
    unknown = [_base_model(architecture=_architecture(kind="unknown"))]

    from_size = document_terminal(_document([_hf_package()], base_models=unknown))
    architecture = document_terminal(_document([_hf_package()]))

    assert "basis     #1 computed from the package size, not its architecture" in from_size
    assert "basis " not in architecture
    assert "(from size)" not in from_size


def test_the_install_line_stands_apart_with_the_command_to_copy():
    text = document_terminal(_document([_hf_package()]), install_for="workstation")

    assert "install   #1  ollama pull hf.co/packager/Nova-8B-GGUF:Q4_K_M" in text
    # A blank line in front of it: the command went under in running text (test round, 2026-09-25).
    lines = text.splitlines()
    assert lines[lines.index(next(line for line in lines if "ollama pull" in line)) - 1] == ""
    # Without the machine of the run there is no line: `render` cannot know which writer it is on.
    assert "ollama pull" not in document_terminal(_document([_hf_package()]))
    assert "ollama pull" not in document_terminal(_document([_hf_package()]), install_for="another")


def test_a_set_aside_piece_names_fewer_packages_where_the_room_is_not_there():
    """Three packages of an account with a long name made a piece of 137 characters (2026-09-24).

    The count and the reason never give way; the names do, one at a time, and nothing is cut inside
    a name -- so no `…` is printed where a console may not be able to encode it.
    """
    packages = [
        _hf_package(repo=f"packager-with-a-very-long-account-name/Nova-8B-{index}-GGUF", weights_bytes=0)
        for index in range(5)
    ]

    lines = document_terminal(_document(packages)).splitlines()

    set_aside = next(line for line in lines if "not covered" in line)
    assert len(set_aside) <= 100
    assert "5 not covered (" in set_aside
    assert set_aside.endswith("and 4 more)")  # one name fits, three do not
    assert "…" not in set_aside


@pytest.mark.parametrize("length", list(range(60, 80)))
def test_no_labeled_line_reaches_a_hundred_and_one_characters(length):
    """`_named` counted two characters of the three it adds -- the space, and both brackets -- so a
    name list of just the right length made a line of 101 (second-model round, 2026-09-25)."""
    from modelroom.views import _named

    piece = f"5 not covered{_named(['a' * length], 89 - len('5') - len('not covered') - 1)}"

    assert len(" " + "shown".ljust(8) + "  " + piece) <= 100


def test_a_name_so_long_that_none_of_them_fits_leaves_the_count_and_the_reason():
    from modelroom.views import _set_aside_pieces

    class _Entry:
        packager, quantization, reason = "a" * 120, "Q4_K_M", "a weight file has no size"

    pieces = _set_aside_pieces("not covered", [_Entry()])

    assert pieces == ["1 not covered"]


def test_the_terminal_view_groups_what_was_set_aside_behind_the_showing_count():
    """41 lines that all said `not covered` pushed the ranking off the screen (hand test, 2026-09-24)."""
    packages = [_hf_package(repo=f"packager/Nova-8B-{index}-GGUF", weights_bytes=0) for index in range(5)]

    text = document_terminal(_document(packages))

    set_aside = [line for line in text.splitlines() if "not covered" in line]
    assert len(set_aside) == 1
    assert "0 of 0 packages" in text
    assert "5 not covered (" in set_aside[0]
    assert set_aside[0].count("Q4_K_M") == 3
    assert set_aside[0].endswith("and 2 more)")


def test_one_package_behind_a_reason_is_named_in_the_singular():
    text = document_terminal(_document([_hf_package(weights_bytes=0)]))

    assert "shown     0 of 0 packages · 1 not covered (packager Q4_K_M)" in text


def test_two_reasons_in_one_list_are_named_with_their_reason():
    """With one reason the word says it; with two the reason has to stand next to the count."""
    from modelroom.views import _set_aside_pieces

    document = _document([_hf_package(weights_bytes=0), _hf_package(weights_bytes=0, repo="p/Other-GGUF")])
    entries = sorted(document.machines[0].not_covered, key=lambda entry: entry.packager)
    other = entries[1].model_copy(update={"reason": "more than one request"})

    one = _set_aside_pieces("not covered", entries)
    two = _set_aside_pieces("not covered", [entries[0], other])

    assert one == ["2 not covered (p Q4_K_M, packager Q4_K_M)"]
    assert two == [
        "1 not covered, a weight file has no size (p Q4_K_M)",
        "1 not covered, more than one request (packager Q4_K_M)",
    ]


MARKDOWN_FIXTURE = "markdown_view_ap9_c2.md"


def test_the_markdown_view_did_not_change_one_byte():
    """The Markdown file is the product; the terminal view was rewritten, this was not.

    The fixture was written by the code of the commit this work package started from, from the
    very same document, and is compared as bytes -- a space or a column that moved would be a
    change to a file other people diff.
    """
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / MARKDOWN_FIXTURE
    huge = _hf_package(weights_bytes=400 * GIB, repo="packager/Nova-8B-XL-GGUF")
    document = _document([_hf_package(file_digest="sha256:" + "c" * 64), huge])

    rendered = document_markdown(document)

    assert rendered.encode("utf-8") == fixture.read_bytes()


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


# --- the scenario of the requests: where the number came from, the hint, the reason under speed -----


def _for_requests(requests: int, users: int | None = None, origin: str | None = None, packages=None, measurements=None):
    """A document computed for `requests`, with the origin `render_cmd` copies from `[guided]`."""
    document = _document(
        packages or [_hf_package()],
        measurements=measurements,
        scenario=default_scenario().model_copy(update={"requests": requests}),
    )
    return RenderDocument.model_validate({**document.model_dump(), "users": users, "requests_origin": origin})


def _card_facts(document: RenderDocument) -> list:
    return result_card(
        document,
        folder=Path("results"),
        markdown=Path("results") / "docs" / "models.md",
        model_names=["Nova-8B"],
        packages=1,
        accounts=["packager"],
        now=RENDERED_AT,
        machine="workstation",
    )


@pytest.mark.parametrize(
    "requests, users, origin, said",
    [
        (3, 25, "from_users", "3 requests (from 25 users)"),
        (1, 1, "from_users", "1 request (from 1 user)"),
        (12, None, "entered", "12 requests (entered)"),
        (1, 25, "entered", "1 request (entered)"),
        (1, None, "default", "1 request"),
        # A scenario passed in by a caller: the configuration says nothing about its number.
        (2, None, None, "2 requests"),
    ],
)
def test_the_scenario_line_names_where_the_requests_came_from(requests, users, origin, said):
    scenario = default_scenario().model_copy(update={"requests": requests})

    assert scenario_line(scenario, users, origin) == f"context 8192 (default), KV cache f16 (assumed), {said}"


def test_the_markdown_head_and_the_card_say_the_same_sentence():
    document = _for_requests(3, 25, "from_users")
    sentence = "context 8192 (default), KV cache f16 (assumed), 3 requests (from 25 users)"

    assert f"Scenario: {sentence}\n" in document_markdown(document)
    context = next(fact for fact in _card_facts(document) if fact.label == "context")
    assert f"{context.label} {context.value}" == sentence and context.note == ""


def test_beyond_eight_requests_the_markdown_head_and_the_card_carry_the_hint():
    document = _for_requests(9, None, "entered")

    assert f"\nLoad: {OLLAMA_REQUEST_HINT_TEXT}\n" in document_markdown(document)
    facts = _card_facts(document)
    labels = [fact.label for fact in facts]
    hint = facts[labels.index("load") : labels.index("load") + 2]
    assert " ".join(fact.value for fact in hint) == OLLAMA_REQUEST_HINT_TEXT
    assert hint[1].label == ""
    assert labels.index("context") < labels.index("load") < labels.index("speed")


def test_eight_requests_carry_no_hint():
    document = _for_requests(8, None, "entered")

    assert "throughput" not in document_markdown(document)
    assert "load" not in [fact.label for fact in _card_facts(document)]


def test_a_measurement_that_counts_but_for_its_requests_is_named_under_speed_not_asked_for_again():
    package = _hf_package(file_digest="sha256:" + "c" * 64)
    document = _for_requests(3, 25, "from_users", [package], {PROFILE_ID: [_measurement(package)]})

    terminal = document_terminal(document)
    speed = [line for line in terminal.splitlines() if line.strip().startswith("speed ")]
    assert speed == [" speed     #1 measured with 1 request, ranking assumes 3"]
    assert MEASURE_HINT not in terminal
    assert "| measured with 1 request, ranking assumes 3 |" in document_markdown(document)
    card = next(fact for fact in _card_facts(document) if fact.label == "speed")
    assert (card.value, card.note) == ("not measured", "#1 measured with 1 request, ranking assumes 3")


def test_rows_without_any_measurement_still_say_so_and_what_to_do():
    terminal = document_terminal(_for_requests(3, 25, "from_users"))

    assert "nothing measured in the rows shown" in terminal and MEASURE_HINT in terminal
    card = next(fact for fact in _card_facts(_for_requests(3, 25, "from_users")) if fact.label == "speed")
    assert card.note == MEASURE_HINT


@pytest.mark.parametrize("requests", [100, 1024])
def test_a_long_list_of_unused_measurements_keeps_the_card_row_within_a_hundred_columns(requests):
    """Six scattered ranks at 100 or 1024 requests would make the card's speed row 101 columns wide
    as the screen prints it (its leading blank counted); the card then counts the rows instead of
    naming them -- the table notes below still name them."""
    from modelroom.intro import fact_line
    from modelroom.screen import plain

    package = _hf_package(file_digest="sha256:" + "c" * 64)
    document = _for_requests(3, 25, "from_users", [package], {PROFILE_ID: [_measurement(package)]})
    entry = document.machines[0].ranked[0]
    note = entry.note.model_copy(update={"text": f"measured with 1 request, ranking assumes {requests}"})
    rows = [entry.model_copy(update={"rank": rank, "note": note}) for rank in (1, 2, 4, 6, 8, 10)]
    block = document.machines[0].model_copy(update={"ranked": rows})
    scattered = document.model_copy(update={"machines": [block]})

    speed = next(fact for fact in _card_facts(scattered) if fact.label == "speed")
    assert len(plain(fact_line(speed))) <= 100
    assert speed.note == f"6 rows measured with 1 request, ranking assumes {requests}"
    assert f"#1–2, #4, #6, #8, #10 measured with 1 request, ranking assumes {requests}" in document_terminal(scattered)
