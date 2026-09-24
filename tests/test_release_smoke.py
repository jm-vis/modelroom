"""scripts/release-smoke.py: only the customer configuration and the reading of results.

The full run (build, fresh venv, install, hardware, fetch without network, render) is an
acceptance step, not a unit test; these tests cover the two pieces with logic of their own.
"""

import importlib.util
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fixture_support import build_transport, qwen35_example_config_dict, qwen35_transport_mapping
from modelroom.cli import fetch_with_config, render_with_config
from modelroom.config import Configuration, load_config
from modelroom.contracts import load_hardware_snapshot
from modelroom.examples import EXAMPLES

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "release-smoke.py"
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the smoke test is tested from a clone", allow_module_level=True)
_SPEC = importlib.util.spec_from_file_location("release_smoke", SCRIPT)
rs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rs)

EXAMPLE = (REPO / "modelroom.example.toml").read_text(encoding="utf-8")


def test_customer_configuration_is_derived_from_the_example(tmp_path):
    path = tmp_path / "modelroom.toml"
    path.write_text(rs.customer_config(EXAMPLE, {"Qwen/Qwen3.5-9B"}), encoding="utf-8")
    config = load_config(path)
    assert list(config.machines) == ["kunde"]
    machine = config.machines["kunde"]
    assert (machine.reserve_ram_gib, machine.reserve_vram_gib, machine.writer) == (0, 0, True)
    assert [family.name for family in config.families] == ["qwen3.5"]
    assert config.families[0].base_models[0].ollama_tag == "9b"
    assert config.paths.state == (tmp_path / "state").resolve()
    assert config.llmfit.min_version == "1.1.16"


def test_customer_configuration_without_a_matching_family_is_refused():
    with pytest.raises(ValueError, match="no family"):
        rs.customer_config(EXAMPLE, {"acme/Nova-7B"})


def test_an_example_without_a_needed_table_is_a_value_error():
    with pytest.raises(ValueError, match="llmfit"):
        rs.customer_config(EXAMPLE.replace("[llmfit]", "[llmfit_gone]"), {"Qwen/Qwen3.5-9B"})


def test_a_temp_directory_that_cannot_be_made_is_a_reported_setup_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rs.tempfile, "tempdir", str(tmp_path / "missing"))
    assert rs.main() == 2
    assert capsys.readouterr().out.splitlines()[-1] == "RELEASE-SMOKE: FAIL"


def machines_document(cell: str) -> str:
    return ("## Packages\n\n| Packager | Stars | fit (computed, v1): kunde |\n|---|---|---|\n| x | – | good |\n\n"
            f"## Machines\n\n| Machine | Installed |\n|---|---|\n| kunde | {cell} |\n")


def test_an_unknown_inventory_must_be_rendered_exactly_as_unknown_with_a_reason():
    assert rs.render_problems(machines_document("unknown (llmfit not found)"), "kunde", None) == []
    assert rs.render_problems(machines_document("unknownBROKEN"), "kunde", None)
    assert rs.render_problems(machines_document("unknown ()"), "kunde", None)


def rendered(tmp_path: Path, with_profile: bool) -> str:
    config = Configuration.from_dict(qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md")))
    run_at = datetime(2026, 9, 22, 9, tzinfo=timezone.utc)
    assert fetch_with_config(config, "workstation", build_transport(qwen35_transport_mapping()), run_at) == 0
    if with_profile:
        profile = dict(EXAMPLES["HardwareSnapshot"], machine="workstation")
        (tmp_path / "state" / "hardware").mkdir()
        (tmp_path / "state" / "hardware" / "workstation.json").write_text(json.dumps(profile), encoding="utf-8")
    assert render_with_config(config) == 0
    return (tmp_path / "models.md").read_text(encoding="utf-8")


def test_markdown_table_reads_the_rendered_package_table(tmp_path):
    rows = rs.markdown_table(rendered(tmp_path, with_profile=True), "Packages")
    assert len(rows) == 2
    assert {row["Packager"] for row in rows} == {"unsloth", "ollama"}
    assert all(row["Stars"] == "–" for row in rows)
    assert all(row["fit (computed, v1): workstation"].startswith("no recommendation:") for row in rows)


def test_the_placed_schema_1_profile_validates_and_carries_the_fixture_inventory():
    """The step that keeps `render` testable until the renderer reads schema 2: the profile it
    places has to be a valid schema-1 `HardwareSnapshot` with the fixtures' own numbers."""
    profile = rs.v1_profile_from_fixtures("kunde")

    assert load_hardware_snapshot(profile).machine == "kunde"
    assert profile["vram_gib"] == 11.94 and profile["ram_gib"] == 127.46
    assert len(profile["installed"]) == 3
    assert all(entry["digest"].startswith("sha256:") for entry in profile["installed"])


def test_render_checks_pass_on_a_document_with_a_profile(tmp_path):
    assert rs.render_problems(rendered(tmp_path, with_profile=True), "workstation", 1) == []


def test_render_checks_compare_installed_with_the_profile(tmp_path):
    problems = rs.render_problems(rendered(tmp_path, with_profile=True), "workstation", 2)
    assert any("Installed" in problem for problem in problems)


def test_render_checks_name_a_machine_without_a_profile(tmp_path):
    problems = rs.render_problems(rendered(tmp_path, with_profile=False), "workstation", None)
    assert any("Installed" in problem for problem in problems)


def test_render_checks_name_a_missing_fit_column(tmp_path):
    problems = rs.render_problems(rendered(tmp_path, with_profile=True), "kunde", 1)
    assert any("fit (computed, v1): kunde" in problem for problem in problems)


def test_markdown_table_keeps_an_escaped_pipe_inside_its_cell():
    text = "## T\n\n| A | B |\n|---|---|\n| x \\| y | z |\n"
    assert rs.markdown_table(text, "T") == [{"A": "x \\| y", "B": "z"}]


def test_expected_areas_follow_the_contract_rule():
    config = tomllib.loads(rs.customer_config(EXAMPLE, {"Qwen/Qwen3.5-9B"}))
    areas = rs.expected_areas(config)
    owners = ["unsloth", "bartowski", "mradermacher", "lmstudio-community", "ggml-org", "Qwen"]
    assert areas == {("huggingface", "Qwen/Qwen3.5-9B", owner) for owner in owners} | {
        ("ollama", "Qwen/Qwen3.5-9B", None)}


def test_area_problems_demand_every_expected_area_and_each_incomplete():
    expected = {("huggingface", "a/B", "x"), ("ollama", "a/B", None)}
    def area(source, packager, status):
        return {"source": source, "base_model_hf_repo": "a/B", "packager": packager, "status": status}
    assert rs.area_problems([area("huggingface", "x", "incomplete"), area("ollama", None, "incomplete")], expected) == []
    assert rs.area_problems([area("huggingface", "x", "incomplete")], expected) != []
    assert rs.area_problems([area("huggingface", "x", "complete"), area("ollama", None, "incomplete")], expected) != []


def test_report_lists_every_step_and_exit_is_the_worst():
    steps = [rs.Step("build", 0, 0, "built"), rs.Step("fetch", 1, 1, "area not incomplete"),
             rs.Step("cleanup", 0, 0, "")]
    text = rs.format_report(steps)
    assert "[1] build: exit 0, OK" in text
    assert "[2] fetch: exit 1, FAIL" in text
    assert "    area not incomplete" in text
    assert rs.worst(steps) == 1
    assert rs.worst([rs.Step("venv", 1, 2, "")]) == 2
