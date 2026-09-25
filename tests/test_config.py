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
    DefaultsConfig,
    FamilyConfig,
    GuidedConfig,
    LlmfitConfig,
    MachineConfig,
    PathsConfig,
    UpdatesConfig,
    load_config,
    normalize_config_v1,
    package_targets,
)
from modelroom.contracts import SchemaVersionError

REPO = Path(__file__).parent.parent
EXAMPLE_TOML = REPO / "modelroom.example.toml"

MODEL_CLASSES = {
    "BaseModelConfig": BaseModelConfig,
    "FamilyConfig": FamilyConfig,
    "MachineConfig": MachineConfig,
    "DefaultsConfig": DefaultsConfig,
    "UpdatesConfig": UpdatesConfig,
    "GuidedConfig": GuidedConfig,
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


def test_paths_config_markdown_equal_to_snapshot_file_is_rejected(tmp_path):
    state_dir = tmp_path / "state"
    payload = {"state": str(state_dir), "markdown": str(state_dir / "modelroom.json")}
    with pytest.raises(ValidationError, match="markdown"):
        PathsConfig.model_validate(payload)


def test_paths_config_markdown_equal_to_lock_file_is_rejected(tmp_path):
    state_dir = tmp_path / "state"
    payload = {"state": str(state_dir), "markdown": str(state_dir / "modelroom.lock")}
    with pytest.raises(ValidationError, match="markdown"):
        PathsConfig.model_validate(payload)


def test_paths_config_markdown_inside_hardware_dir_is_rejected(tmp_path):
    state_dir = tmp_path / "state"
    payload = {"state": str(state_dir), "markdown": str(state_dir / "hardware" / "workstation.json")}
    with pytest.raises(ValidationError, match="markdown"):
        PathsConfig.model_validate(payload)


def test_paths_config_markdown_equal_to_state_dir_is_rejected(tmp_path):
    state_dir = tmp_path / "state"
    payload = {"state": str(state_dir), "markdown": str(state_dir)}
    with pytest.raises(ValidationError, match="markdown"):
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
    # R8-3 (fix-round 7): no Ollama mapping on the copy, so the R7-7 Ollama collision validator
    # cannot be the one raising; the match pins the HF collision message.
    second_family["base_models"][0]["ollama_base"] = None
    second_family["base_models"][0]["ollama_tag"] = None
    payload["families"].append(second_family)
    with pytest.raises(ValidationError, match="packager repo"):
        Configuration.model_validate(payload)


def test_configuration_rejects_a_default_gguf_name_colliding_with_another_base_models_alias():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    second_family = json.loads(json.dumps(payload["families"][0]))
    second_family["name"] = "other-family"
    second_family["base_models"][0]["hf_repo"] = "acme/Nova-7B-Instruct"
    # this base model's *default* candidate name ("Nova-7B-Instruct-GGUF") collides with the
    # first base model's explicit repo_aliases entry.
    second_family["base_models"][0]["repo_aliases"] = []
    second_family["base_models"][0]["ollama_base"] = None  # R8-3: see the test above
    second_family["base_models"][0]["ollama_tag"] = None
    payload["families"].append(second_family)
    with pytest.raises(ValidationError, match="packager repo"):
        Configuration.model_validate(payload)


# --- R7-8 (fix-round 6): the collision check compares full "owner/name" candidates, not name
# alone -- P3-6's own check compared bare repo names, so two base models with the same *name*
# under two different owners (no shared packager) were wrongly rejected even though
# `hf.py::fetch_hf_area` would never probe the same `owner/name` combination for them.
#
# Probe (fix-round 6 brief): `acme/Nova` + `other/Nova` with `packagers = []` was rejected
# ("packager repo name 'Nova-GGUF' is claimed by both"), although the two repos actually probed
# are `acme/Nova-GGUF` and `other/Nova-GGUF` -- distinct owners, no collision.


def _minimal_two_owner_payload(packagers: list[str]) -> dict:
    return {
        "schema_version": 2,
        "families": [
            {"name": "nova", "base_models": [{"hf_repo": "acme/Nova", "repo_aliases": []}]},
            {"name": "nova-other", "base_models": [{"hf_repo": "other/Nova", "repo_aliases": []}]},
        ],
        "packagers": packagers,
        "publishers": ["acme", "other"],
        "paths": dict(EXAMPLES["Configuration"]["paths"]),
    }


def test_configuration_accepts_the_same_repo_name_under_two_different_owners_with_no_shared_packager():
    Configuration.model_validate(_minimal_two_owner_payload(packagers=[]))  # must not raise


def test_configuration_still_rejects_the_same_repo_name_when_a_packager_is_shared():
    """The probe case above must pass with no packager in common -- but a *shared* packager
    still makes both base models probe the same `packager/Nova-GGUF` candidate, which is a real
    collision (P3-6's original hazard), so this must still fail.
    """
    with pytest.raises(ValidationError):
        Configuration.model_validate(_minimal_two_owner_payload(packagers=["packager"]))


# --- R7-7 (fix-round 6): two base models on the same ollama_base must not claim colliding
# ollama_tag identities -- `ollama._keep_relevant_tags` keeps `tag == ollama_tag or
# tag.startswith(ollama_tag + "-")`, so `ollama_tag="7b"` also owns a real registry tag
# `"7b-instruct"`; two base models configured that way would both resolve packages under the
# *same* Ollama identity (`nova:7b`), and `state.merge_snapshot` lets whichever is processed last
# silently overwrite the other's package, exactly like P3-6 for Hugging Face packager names.


def _second_base_model_payload(hf_repo: str, ollama_base: str, ollama_tag: str) -> dict:
    return {
        "name": "other-family",
        "base_models": [
            {"hf_repo": hf_repo, "repo_aliases": [], "ollama_base": ollama_base, "ollama_tag": ollama_tag}
        ],
    }


def test_configuration_rejects_equal_ollama_tags_on_the_same_ollama_base():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["families"].append(_second_base_model_payload("acme/Other-7B", "nova", "7b"))
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_rejects_an_ollama_tag_that_is_a_prefix_of_another():
    """Probe (fix-round 6 brief): `nova:7b` also owns the real registry tag `7b-instruct` --
    configuring a second base model with `ollama_tag="7b-instruct"` on the same `ollama_base`
    must be rejected exactly like an equal tag would be.
    """
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["families"].append(_second_base_model_payload("acme/Other-7B", "nova", "7b-instruct"))
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_accepts_distinct_non_overlapping_ollama_tags_on_the_same_ollama_base():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["families"].append(_second_base_model_payload("acme/Other-14B", "nova", "14b"))
    Configuration.model_validate(payload)  # must not raise


def test_configuration_accepts_distinct_repo_aliases_across_base_models():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    second_family = json.loads(json.dumps(payload["families"][0]))
    second_family["name"] = "other-family"
    second_family["base_models"][0]["hf_repo"] = "acme/Other-7B"
    second_family["base_models"][0]["repo_aliases"] = ["Other-7B-Instruct-GGUF"]
    # Schema 2: the copied owner-bound `repos` entry would be a real collision; this test is
    # about repo_aliases only.
    second_family["base_models"][0]["repos"] = []
    # R7-7 (fix-round 6): the first base model already claims ollama_base "nova"/tag "7b" -- a
    # distinct tag on the same base keeps this test about repo_aliases only, not an (also
    # correctly rejected) ollama identity collision.
    second_family["base_models"][0]["ollama_tag"] = "14b"
    payload["families"].append(second_family)
    Configuration.model_validate(payload)  # must not raise


def test_configuration_accepts_empty_machines():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["machines"] = {}
    Configuration.model_validate(payload)


@pytest.mark.parametrize("version", [1, 3])
def test_configuration_model_accepts_only_schema_2(version):
    # Schema 1 reaches the model only through `normalize_config_v1`; the model itself is v2.
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["schema_version"] = version
    with pytest.raises(ValidationError):
        Configuration.model_validate(payload)


def test_configuration_schema_2_accepts_empty_families():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["families"] = []
    assert Configuration.model_validate(payload).families == []


def test_configuration_rejects_an_owner_bound_repo_shared_by_two_base_models():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["families"].append(
        {
            "name": "other-family",
            "base_models": [{"hf_repo": "acme/Other-7B", "repos": ["community/Nova-7B-GGUF"]}],
        }
    )
    with pytest.raises(ValidationError, match="community/Nova-7B-GGUF"):
        Configuration.model_validate(payload)


def test_configuration_rejects_a_repo_listed_twice_in_repos():
    payload = json.loads(json.dumps(EXAMPLES["BaseModelConfig"]))
    payload["repos"] = ["community/Nova-7B-GGUF", "community/Nova-7B-GGUF"]
    with pytest.raises(ValidationError):
        BaseModelConfig.model_validate(payload)


def test_configuration_rejects_a_repos_entry_without_owner():
    payload = json.loads(json.dumps(EXAMPLES["BaseModelConfig"]))
    payload["repos"] = ["Nova-7B-GGUF"]
    with pytest.raises(ValidationError):
        BaseModelConfig.model_validate(payload)


def test_machine_config_rejects_a_profile_that_is_not_a_profile_id():
    payload = dict(EXAMPLES["MachineConfig"])
    payload["profile"] = "workstation"
    with pytest.raises(ValidationError):
        MachineConfig.model_validate(payload)


def test_defaults_updates_and_guided_have_their_documented_defaults():
    config = Configuration.model_validate({"schema_version": 2, "paths": EXAMPLES["PathsConfig"]})
    assert (config.defaults.reserve_ram_gib, config.defaults.reserve_vram_gib) == (8.0, 1.0)
    assert config.updates.check is True
    assert config.guided.results is None


# --- package_targets: one target set for the collision check (and later fetch/provenance) ----


def test_package_targets_is_owners_times_names_then_owner_bound_repos_without_duplicates():
    config = Configuration.model_validate(EXAMPLES["Configuration"])
    base_model = config.families[0].base_models[0]
    assert package_targets(config, base_model) == [
        "packager/Nova-7B-GGUF",
        "packager/Nova-7B-Instruct-GGUF",
        "acme/Nova-7B-GGUF",
        "acme/Nova-7B-Instruct-GGUF",
        "community/Nova-7B-GGUF",
    ]


def test_package_targets_lists_a_repo_once_when_repos_repeats_a_generated_target():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["families"][0]["base_models"][0]["repos"] = ["packager/Nova-7B-GGUF"]
    config = Configuration.model_validate(payload)
    targets = package_targets(config, config.families[0].base_models[0])
    assert targets.count("packager/Nova-7B-GGUF") == 1


def test_package_targets_counts_the_own_owner_once_when_it_is_also_a_packager():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["packagers"] = ["acme"]
    config = Configuration.model_validate(payload)
    targets = package_targets(config, config.families[0].base_models[0])
    assert targets.count("acme/Nova-7B-GGUF") == 1


# --- schema 1 -> 2 in memory (normalize_config_v1) -------------------------------------------


def _schema_1_payload() -> dict:
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["schema_version"] = 1
    for key in ("defaults", "updates", "guided"):
        del payload[key]
    del payload["machines"]["workstation"]["profile"]
    del payload["families"][0]["base_models"][0]["repos"]
    return payload


def test_normalize_config_v1_is_lossless_and_leaves_its_input_unchanged():
    legacy = _schema_1_payload()
    before = json.loads(json.dumps(legacy))
    normalized = normalize_config_v1(legacy)
    assert legacy == before
    assert normalized["schema_version"] == 2
    for key, value in legacy.items():
        if key != "schema_version":
            assert normalized[key] == value
    assert normalized["defaults"] == {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0}
    assert normalized["updates"] == {"check": True}
    assert "guided" not in normalized
    Configuration.model_validate(normalized)


def test_normalize_config_v1_keeps_schema_1s_rule_of_at_least_one_family():
    legacy = _schema_1_payload()
    legacy["families"] = []
    with pytest.raises(ValueError, match="at least one"):
        normalize_config_v1(legacy)


def test_from_dict_reads_a_schema_1_dict_as_schema_2():
    config = Configuration.from_dict(_schema_1_payload())
    assert config.schema_version == 2
    assert config.machines["workstation"].profile is None


def test_from_dict_schema_1_without_families_is_a_config_error():
    legacy = _schema_1_payload()
    legacy["families"] = []
    with pytest.raises(ConfigError, match="at least one"):
        Configuration.from_dict(legacy)


def test_shipped_example_file_is_written_as_schema_2():
    import tomllib

    assert tomllib.loads(EXAMPLE_TOML.read_text(encoding="utf-8"))["schema_version"] == 2


def test_schema_1_fixture_with_two_machines_still_loads():
    config = load_config(REPO / "tests" / "fixtures" / "config_v1" / "config.toml")
    assert config.schema_version == 2
    assert set(config.machines) == {"laptop", "server"}
    assert all(machine.profile is None for machine in config.machines.values())


def test_load_config_resolves_a_relative_guided_results_path(tmp_path):
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(
        'schema_version = 2\n\n[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n\n'
        '[guided]\nresults = "results"\n',
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.guided.results == tmp_path / "results"


# --- [guided].context: the context a guided run chose ----------------------------------------


def test_guided_config_keeps_the_context_a_guided_run_chose():
    assert GuidedConfig(results=Path(EXAMPLES["GuidedConfig"]["results"]), context=4096).context == 4096


def test_guided_config_without_a_context_means_no_guided_run_chose_one():
    assert GuidedConfig().context is None


@pytest.mark.parametrize("value", [0, -1, 2**31, "lots", 4096.5], ids=["zero", "negative", "too large", "text", "fraction"])
def test_guided_config_refuses_a_context_no_ranking_can_be_computed_for(value):
    with pytest.raises(ValidationError):
        GuidedConfig(context=value)


@pytest.mark.parametrize("value", [1, 2**31 - 1], ids=["smallest", "largest"])
def test_guided_context_takes_exactly_the_contexts_a_scenario_takes(value):
    """The stored context is a `Scenario.context_requested`, so the two bounds have to agree."""
    from modelroom.measurements import Scenario

    assert GuidedConfig(context=value).context == value
    assert Scenario(
        context_requested=value, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1
    ).context_requested == value


def test_load_config_reads_a_stored_context(tmp_path):
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(
        'schema_version = 2\n\n[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n\n'
        '[guided]\nresults = "results"\ncontext = 4096\n',
        encoding="utf-8",
    )
    assert load_config(config_path).guided.context == 4096


def test_load_config_without_a_stored_context_reads_the_rest_unchanged(tmp_path):
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(
        'schema_version = 2\n\n[paths]\nstate = "state"\nmarkdown = "docs/models.md"\n\n'
        '[guided]\nresults = "results"\n',
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.guided.context is None
    assert config.guided.results == tmp_path / "results"


def test_normalize_config_v1_adds_no_context(tmp_path):
    """Schema 1 has no `[guided]` table at all, and the normalization does not invent one."""
    assert "guided" not in normalize_config_v1(_schema_1_payload())


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


@pytest.mark.parametrize("name", ["modelroom.lock", "modelroom.json", "run-status.json"])
def test_load_config_refuses_a_configuration_that_is_a_state_file(tmp_path, name):
    # A configuration inside <state>/hardware or <state>/measurements is already outside the
    # confinement of paths.state; only the files directly in <state> need this check.
    config_path = tmp_path / name
    config_path.write_text(
        f'schema_version = 2\n[paths]\nstate = "."\nmarkdown = "{tmp_path.parent.as_posix()}/models.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="state file"):
        load_config(config_path)


def test_from_dict_rejects_a_relative_guided_results_path():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["guided"] = {"results": "results"}
    with pytest.raises((ConfigError, ValidationError), match="absolute"):
        Configuration.from_dict(payload)


def test_from_dict_wrong_schema_version_raises_schema_version_error_before_field_errors():
    payload = json.loads(json.dumps(EXAMPLES["Configuration"]))
    payload["schema_version"] = 3
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


def test_load_config_file_that_is_not_utf8_raises_config_error_naming_the_path(tmp_path):
    """`UnicodeDecodeError` is a `ValueError`, not an `OSError`: without its own clause it left
    every command that reads the configuration as a traceback."""
    config_path = tmp_path / "modelroom.toml"
    config_path.write_bytes(b"schema_version = 2\n# \xff\xfe not UTF-8\n")
    with pytest.raises(ConfigError, match=r"modelroom\.toml.*UTF-8"):
        load_config(config_path)


def test_load_config_wrong_schema_version_raises_schema_version_error_before_field_errors(tmp_path):
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(
        """
schema_version = 3
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
    assert CONFIG_SCHEMA_RANGE == (1, 3)
    assert CONFIG_SCHEMA_VERSION == 2


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


# --- R7-1: paths.markdown must not collide with the config file itself --------------------


def test_load_config_rejects_markdown_equal_to_the_config_file(tmp_path):
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(
        """
schema_version = 1
packagers = ["packager"]
publishers = ["acme"]

[[families]]
name = "nova"

  [[families.base_models]]
  hf_repo = "acme/Nova-7B"

[paths]
state = "state"
markdown = "modelroom.toml"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="markdown"):
        load_config(config_path)


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
