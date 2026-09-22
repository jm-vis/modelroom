"""Tests for modelroom.config: the Configuration model tree and the TOML reader."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from pydantic import ValidationError

from modelroom.config import (
    EXAMPLES,
    CONFIG_SCHEMA_RANGE,
    CONFIG_SCHEMA_VERSION,
    BaseModelConfig,
    ConfigError,
    Configuration,
    FamilyConfig,
    LlmfitConfig,
    MachineConfig,
    PathsConfig,
    load_config,
)
from modelroom.contracts import SchemaVersionError

REPO = Path(__file__).parent.parent
EXAMPLE_TOML = REPO / "modelroom.example.toml"

MODEL_CLASSES = {
    "BaseModelConfig": BaseModelConfig,
    "FamilyConfig": FamilyConfig,
    "MachineConfig": MachineConfig,
    "PathsConfig": PathsConfig,
    "LlmfitConfig": LlmfitConfig,
    "Configuration": Configuration,
}


# --- EXAMPLES validate, and cover every model, and only every model -----------------------


@pytest.mark.parametrize("name, model_cls", MODEL_CLASSES.items())
def test_example_validates_against_its_model(name, model_cls):
    assert name in EXAMPLES, f"no EXAMPLES entry for {name}"
    model_cls.model_validate(EXAMPLES[name])


def test_examples_has_no_entries_for_unknown_models():
    assert set(EXAMPLES) == set(MODEL_CLASSES)


@pytest.mark.parametrize("value", [EXAMPLES["PathsConfig"]["state"], EXAMPLES["Configuration"]["paths"]["markdown"]])
def test_example_paths_are_absolute_on_every_platform(value):
    # PathsConfig is validated with the running platform's Path; the EXAMPLES must therefore
    # validate on Windows and POSIX alike, or the contract tests turn red on the other OS.
    assert PurePosixPath(value).is_absolute(), f"{value!r} is not absolute on POSIX"
    assert PureWindowsPath(value).is_absolute(), f"{value!r} is not absolute on Windows"


# --- CONTRACTS.md documents every config model and repeats its example verbatim -----------


def _contracts_md_config_examples() -> dict[str, dict]:
    text = (REPO / "CONTRACTS.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^#### (\w+)\s*$", text, flags=re.MULTILINE))
    found: dict[str, dict] = {}
    for i, heading in enumerate(headings):
        name = heading.group(1)
        start = heading.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[start:end]
        code_block = re.search(r"```json\n(.*?)\n```", section, flags=re.DOTALL)
        assert code_block, f"#### {name} has no ```json example block"
        found[name] = json.loads(code_block.group(1))
    return found


def test_contracts_md_names_every_config_model_and_no_extra_ones():
    documented = _contracts_md_config_examples()
    assert set(documented) == set(MODEL_CLASSES)


def test_contracts_md_config_examples_match_examples_module_verbatim():
    documented = _contracts_md_config_examples()
    for name, example in EXAMPLES.items():
        assert documented[name] == example, f"CONTRACTS.md example for {name} drifted from EXAMPLES"


# --- extra=forbid ----------------------------------------------------------------------------


def test_base_model_config_extra_field_is_rejected():
    payload = dict(EXAMPLES["BaseModelConfig"])
    payload["publisher"] = "acme"  # belongs to the snapshot's BaseModelSpec, not here
    with pytest.raises(ValidationError):
        BaseModelConfig.model_validate(payload)


def test_configuration_extra_field_is_rejected():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["unexpected"] = "nope"
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


# --- BaseModelConfig: shared validators behave like BaseModelSpec's -------------------------


def test_base_model_config_bad_hf_repo_is_rejected():
    payload = dict(EXAMPLES["BaseModelConfig"])
    payload["hf_repo"] = "not-an-owner-slash-name"
    with pytest.raises(ValidationError):
        BaseModelConfig.model_validate(payload)


def test_base_model_config_ollama_base_without_tag_is_rejected():
    payload = dict(EXAMPLES["BaseModelConfig"])
    payload["ollama_tag"] = None
    with pytest.raises(ValidationError):
        BaseModelConfig.model_validate(payload)


def test_base_model_config_neither_ollama_field_is_accepted():
    payload = dict(EXAMPLES["BaseModelConfig"])
    payload["ollama_base"] = None
    payload["ollama_tag"] = None
    BaseModelConfig.model_validate(payload)


def test_base_model_config_empty_repo_alias_is_rejected():
    payload = dict(EXAMPLES["BaseModelConfig"])
    payload["repo_aliases"] = [""]
    with pytest.raises(ValidationError):
        BaseModelConfig.model_validate(payload)


# --- FamilyConfig ------------------------------------------------------------------------


def test_family_config_requires_at_least_one_base_model():
    payload = json.loads(json.dumps(EXAMPLES["FamilyConfig"]))
    payload["base_models"] = []
    with pytest.raises(ValidationError):
        FamilyConfig.model_validate(payload)


@pytest.mark.parametrize("bad_name", ["Nova", "-nova", "nova!", "", "nova_7b"])
def test_family_config_rejects_bad_names(bad_name):
    payload = json.loads(json.dumps(EXAMPLES["FamilyConfig"]))
    payload["name"] = bad_name
    with pytest.raises(ValidationError):
        FamilyConfig.model_validate(payload)


def test_family_config_accepts_dotted_lowercase_name():
    payload = json.loads(json.dumps(EXAMPLES["FamilyConfig"]))
    payload["name"] = "qwen3.5"
    FamilyConfig.model_validate(payload)


# --- MachineConfig -------------------------------------------------------------------------


def test_machine_config_negative_reserve_ram_is_rejected():
    payload = dict(EXAMPLES["MachineConfig"])
    payload["reserve_ram_gib"] = -1.0
    with pytest.raises(ValidationError):
        MachineConfig.model_validate(payload)


def test_machine_config_negative_reserve_vram_is_rejected():
    payload = dict(EXAMPLES["MachineConfig"])
    payload["reserve_vram_gib"] = -0.5
    with pytest.raises(ValidationError):
        MachineConfig.model_validate(payload)


def test_machine_config_zero_reserve_is_accepted():
    payload = dict(EXAMPLES["MachineConfig"])
    payload["reserve_ram_gib"] = 0.0
    payload["reserve_vram_gib"] = 0.0
    MachineConfig.model_validate(payload)


# --- PathsConfig: absolute paths and derived properties -------------------------------------


def test_paths_config_relative_state_is_rejected():
    payload = {"state": "state", "markdown": EXAMPLES["PathsConfig"]["markdown"]}
    with pytest.raises(ValidationError, match="absolute"):
        PathsConfig.model_validate(payload)


def test_paths_config_relative_markdown_is_rejected():
    payload = {"state": EXAMPLES["PathsConfig"]["state"], "markdown": "docs/models.md"}
    with pytest.raises(ValidationError, match="absolute"):
        PathsConfig.model_validate(payload)


def test_paths_config_derived_properties(tmp_path):
    state_dir = tmp_path / "state"
    paths = PathsConfig.model_validate({"state": str(state_dir), "markdown": str(tmp_path / "models.md")})
    assert paths.snapshot_file == state_dir / "modelroom.json"
    assert paths.lock_file == state_dir / "modelroom.lock"
    assert paths.run_status_file == state_dir / "run-status.json"
    assert paths.hardware_dir == state_dir / "hardware"


# --- LlmfitConfig --------------------------------------------------------------------------


def test_llmfit_config_default_min_version():
    llmfit = LlmfitConfig.model_validate({})
    assert llmfit.min_version == "1.1.16"


@pytest.mark.parametrize("bad_version", ["1.1", "1.1.16.2", "v1.1.16", "1.1.x"])
def test_llmfit_config_rejects_bad_min_version(bad_version):
    with pytest.raises(ValidationError):
        LlmfitConfig.model_validate({"min_version": bad_version})


# --- Configuration: cross-model checks -------------------------------------------------------


def test_configuration_rejects_duplicate_family_name():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    second_family = json.loads(json.dumps(payload["families"][0]))
    second_family["base_models"][0]["hf_repo"] = "acme/Other-7B"
    payload["families"].append(second_family)
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_rejects_duplicate_hf_repo_across_families():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    second_family = json.loads(json.dumps(payload["families"][0]))
    second_family["name"] = "other-family"
    payload["families"].append(second_family)
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_rejects_owner_missing_from_publishers():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["publishers"] = []
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_rejects_bad_machine_name():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["machines"] = {"Bad Name": {"reserve_ram_gib": 1.0, "reserve_vram_gib": 0.0, "writer": True}}
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_rejects_duplicate_packager():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["packagers"] = ["packager", "packager"]
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


# --- P3-6 (fix-round 5): a packager repo name claimed by two base models is a config error --
#
# `hf.py::fetch_hf_area` probes `<owner>/<name>-GGUF` and every `repo_aliases` entry under every
# owner in `packagers` (global to the whole configuration) plus each base model's own owner.
# `Package` identity (CONTRACTS.md, "Identity") does not carry `base_model_hf_repo`, so two base
# models sharing a repo name would each write into the *same* `result_packages` dict entry in
# `state.merge_snapshot` -- whichever area's outcome is processed last silently wins, the other's
# packages vanish with no error. Caught at configuration load instead.


def test_configuration_rejects_a_repo_alias_shared_by_two_base_models():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    second_family = json.loads(json.dumps(payload["families"][0]))
    second_family["name"] = "other-family"
    second_family["base_models"][0]["hf_repo"] = "acme/Other-7B"
    # same repo_aliases entry as the first base model's -- both hf_repo values differ (so the
    # existing hf_repo-uniqueness check does not fire), but the *packager repo name* collides.
    payload["families"].append(second_family)
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_rejects_a_default_gguf_name_colliding_with_another_base_models_alias():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    second_family = json.loads(json.dumps(payload["families"][0]))
    second_family["name"] = "other-family"
    second_family["base_models"][0]["hf_repo"] = "acme/Nova-7B-Instruct"
    # this base model's *default* candidate name ("Nova-7B-Instruct-GGUF") collides with the
    # first base model's explicit repo_aliases entry.
    second_family["base_models"][0]["repo_aliases"] = []
    payload["families"].append(second_family)
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_accepts_distinct_repo_aliases_across_base_models():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    second_family = json.loads(json.dumps(payload["families"][0]))
    second_family["name"] = "other-family"
    second_family["base_models"][0]["hf_repo"] = "acme/Other-7B"
    second_family["base_models"][0]["repo_aliases"] = ["Other-7B-Instruct-GGUF"]
    payload["families"].append(second_family)
    Configuration.model_validate(payload)  # must not raise


def test_configuration_accepts_empty_machines():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["machines"] = {}
    Configuration.model_validate(payload)


def test_configuration_wrong_schema_version_is_rejected():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_allowed_owners_is_the_union_of_packagers_and_publishers():
    config = Configuration.model_validate(EXAMPLES["Configuration"])
    assert config.allowed_owners() == frozenset(EXAMPLES["Configuration"]["packagers"]) | frozenset(
        EXAMPLES["Configuration"]["publishers"]
    )


# --- Configuration.from_dict --------------------------------------------------------------


def test_from_dict_round_trip_equality():
    config = Configuration.model_validate(EXAMPLES["Configuration"])
    round_tripped = Configuration.from_dict(config.model_dump(mode="json"))
    assert round_tripped == config


def test_from_dict_rejects_relative_paths():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["paths"] = {"state": "state", "markdown": "docs/models.md"}
    with pytest.raises((ConfigError, ValidationError), match="absolute"):
        Configuration.from_dict(payload)


def test_from_dict_wrong_schema_version_raises_schema_version_error_before_field_errors():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["schema_version"] = 2
    payload["publishers"] = []  # would also fail field validation, but must never get there
    with pytest.raises(SchemaVersionError):
        Configuration.from_dict(payload)


# --- load_config: file, TOML and schema-version handling ------------------------------------


def _write_minimal_toml(directory: Path) -> Path:
    config_path = directory / "modelroom.toml"
    config_path.write_text(
        """
schema_version = 1
packagers = ["packager"]
publishers = ["acme"]

[[families]]
name = "nova"

  [[families.base_models]]
  hf_repo = "acme/Nova-7B"
  repo_aliases = ["Nova-7B-Instruct-GGUF"]

[paths]
state = "state"
markdown = "docs/models.md"
""",
        encoding="utf-8",
    )
    return config_path


def test_load_config_resolves_relative_paths_against_the_config_file_directory(tmp_path):
    config_path = _write_minimal_toml(tmp_path)
    config = load_config(config_path)
    assert config.paths.state == tmp_path / "state"
    assert config.paths.markdown == tmp_path / "docs" / "models.md"
    assert config.paths.state.is_absolute()


def test_load_config_loads_the_shipped_example_file():
    config = load_config(EXAMPLE_TOML)
    assert config.schema_version == CONFIG_SCHEMA_VERSION
    assert {family.name for family in config.families} == {"qwen3.5", "deepseek-r1", "eurollm"}
    assert set(config.machines) == {"workstation", "inference-server"}
    assert config.paths.state.is_absolute()
    assert config.paths.markdown.is_absolute()


def test_load_config_missing_file_raises_config_error_naming_the_path(tmp_path):
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(ConfigError, match=re.escape(str(missing))):
        load_config(missing)


def test_load_config_toml_syntax_error_raises_config_error(tmp_path):
    config_path = tmp_path / "broken.toml"
    config_path.write_text("schema_version = 1\n[paths\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_load_config_wrong_schema_version_raises_schema_version_error_before_field_errors(tmp_path):
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(
        """
schema_version = 2
publishers = []

[paths]
state = "state"
markdown = "docs/models.md"
""",
        encoding="utf-8",
    )
    with pytest.raises(SchemaVersionError):
        load_config(config_path)


def test_load_config_validation_error_is_wrapped_in_config_error(tmp_path):
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(
        """
schema_version = 1
packagers = ["packager"]
publishers = []

[[families]]
name = "nova"

  [[families.base_models]]
  hf_repo = "acme/Nova-7B"

[paths]
state = "state"
markdown = "docs/models.md"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=re.escape(str(config_path))):
        load_config(config_path)


def test_config_schema_range_matches_snapshot_convention():
    assert CONFIG_SCHEMA_RANGE == (1, 2)
    assert CONFIG_SCHEMA_VERSION == 1


# --- F7: paths.state must be confined to the config file's own directory tree --------------


def _write_toml_with_state(directory: Path, state: str) -> Path:
    config_path = directory / "modelroom.toml"
    config_path.write_text(
        f"""
schema_version = 1
packagers = ["packager"]
publishers = ["acme"]

[[families]]
name = "nova"

  [[families.base_models]]
  hf_repo = "acme/Nova-7B"

[paths]
state = "{state}"
markdown = "docs/models.md"
""",
        encoding="utf-8",
    )
    return config_path


def test_load_config_rejects_a_state_path_outside_the_config_directory(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path = _write_toml_with_state(project_dir, "../outside")

    with pytest.raises(ConfigError):
        load_config(config_path)

    assert not (tmp_path / "outside").exists()


def test_load_config_accepts_a_state_path_inside_the_config_directory(tmp_path):
    config_path = _write_toml_with_state(tmp_path, "state")

    config = load_config(config_path)

    assert config.paths.state == tmp_path / "state"


def test_load_config_rejects_a_symlinked_state_path_pointing_outside(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    link_path = project_dir / "state-link"
    try:
        link_path.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        pytest.skip("this OS/user refuses symlink creation")
    config_path = _write_toml_with_state(project_dir, "state-link")

    with pytest.raises(ConfigError):
        load_config(config_path)


# --- F8: a relative --config path must still resolve paths.state absolutely ----------------


def test_load_config_with_a_relative_config_path_resolves_state_absolutely(tmp_path):
    _write_toml_with_state(tmp_path, "state")
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        config = load_config(Path("modelroom.toml"))
    finally:
        os.chdir(original_cwd)

    assert config.paths.state.is_absolute()
    assert config.paths.state == tmp_path / "state"
