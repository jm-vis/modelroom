"""Configuration schema 3: the assumed parallel requests in `[guided]`, and reading schema 2.

`users` is the head count a user named, `requests` the parallel slots the ranking assumes, and
`requests_origin` where that number came from (CONTRACTS.md, "GuidedConfig"; decided 2026-09-25).
A schema-2 file reads unchanged in memory; only the three writing paths rewrite it
(`tests/test_migrate_v3.py`).
"""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

import modelroom.config as config_module
from modelroom.cli import main
from modelroom.config import (
    CONFIG_SCHEMA_RANGE,
    CONFIG_SCHEMA_VERSION,
    ConfigError,
    Configuration,
    GuidedConfig,
    load_config,
)
from modelroom.contracts import SchemaVersionError
from modelroom.render_cmd import scenario_from_config

FIXTURES = Path(__file__).parent / "fixtures"


def _v2_file(tmp_path: Path, guided: str = "") -> Path:
    path = tmp_path / "modelroom.toml"
    shutil.copy(FIXTURES / "config_v2" / "config.toml", path)
    if guided:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"\n[guided]\n{guided}\n")
    return path


def _v3_file(tmp_path: Path, guided: str) -> Path:
    path = _v2_file(tmp_path)
    text = path.read_text(encoding="utf-8").replace("schema_version = 2", "schema_version = 3")
    path.write_text(f"{text}\n[guided]\n{guided}\n", encoding="utf-8", newline="\n")
    return path


def _v3_dict(tmp_path: Path, **guided) -> dict:
    return {
        "schema_version": 3,
        "paths": {"state": str(tmp_path / "state"), "markdown": str(tmp_path / "docs" / "models.md")},
        "guided": guided,
    }


# --- the version ---------------------------------------------------------------------------------


def test_the_configuration_is_schema_3_and_readers_accept_1_to_3():
    assert CONFIG_SCHEMA_VERSION == 3
    assert CONFIG_SCHEMA_RANGE == (1, 4)


def test_schema_4_is_refused_before_any_field_is_read(tmp_path):
    with pytest.raises(SchemaVersionError):
        Configuration.from_dict({**_v3_dict(tmp_path), "schema_version": 4, "garbage": True})


# --- the three origins ---------------------------------------------------------------------------


def test_a_guided_table_without_the_new_keys_assumes_one_request_by_default():
    guided = GuidedConfig()
    assert (guided.users, guided.requests, guided.requests_origin) == (None, 1, "default")


def test_requests_entered_by_the_user_keep_the_head_count_for_the_display():
    guided = GuidedConfig.model_validate({"users": 25, "requests": 4, "requests_origin": "entered"})
    assert (guided.users, guided.requests, guided.requests_origin) == (25, 4, "entered")
    assert GuidedConfig.model_validate({"requests": 4, "requests_origin": "entered"}).users is None


def test_requests_from_users_follow_the_rule_of_thumb():
    guided = GuidedConfig.model_validate({"users": 25, "requests": 3, "requests_origin": "from_users"})
    assert guided.requests == 3


def test_a_head_count_alone_derives_the_requests_by_the_rule_of_thumb(tmp_path):
    """Only `users` in the table: `from_users` and the rule's number, on every reader (acceptance round, 2026-09-26)."""
    guided = GuidedConfig.model_validate({"users": 25})
    assert (guided.users, guided.requests, guided.requests_origin) == (25, 3, "from_users")
    config = Configuration.from_dict(_v3_dict(tmp_path, users=25))
    assert (config.guided.users, config.guided.requests, config.guided.requests_origin) == (25, 3, "from_users")
    path = _v3_file(tmp_path, "users = 25")
    before = path.read_bytes()
    loaded = load_config(path)
    assert (loaded.guided.users, loaded.guided.requests, loaded.guided.requests_origin) == (25, 3, "from_users")
    assert path.read_bytes() == before


def test_the_active_share_is_one_in_ten():
    assert config_module.ACTIVE_SHARE == 0.10


@pytest.mark.parametrize("users, requests", [(1, 1), (10, 1), (11, 2), (25, 3), (10239, 1024), (10240, 1024)])
def test_requests_from_users_is_ceil_of_a_tenth_between_1_and_1024(users, requests):
    assert config_module.requests_from_users(users) == requests


@pytest.mark.parametrize(
    "guided",
    [
        {"users": 25, "requests": 2, "requests_origin": "from_users"},  # not the rule's number
        {"requests": 3, "requests_origin": "from_users"},  # no head count to derive from
        {"users": 25, "requests": 1, "requests_origin": "default"},  # a head count is never a default
        {"requests": 2, "requests_origin": "default"},  # the default is one request
        {"requests": 3},  # requests without their origin
        {"requests_origin": "entered"},  # an origin without requests
        {"users": 0, "requests": 1, "requests_origin": "entered"},
        {"users": 10241, "requests": 1, "requests_origin": "entered"},
        {"users": 0},  # a head count alone still has to be a head count
        {"users": 10241},
        {"users": "25"},
        {"requests": 0, "requests_origin": "entered"},
        {"requests": 1025, "requests_origin": "entered"},
        {"requests": 2, "requests_origin": "guessed"},
    ],
)
def test_an_inconsistent_guided_table_is_invalid(tmp_path, guided):
    with pytest.raises(ConfigError):
        Configuration.from_dict(_v3_dict(tmp_path, **guided))


def test_requests_without_their_origin_end_a_command_with_exit_2(tmp_path, capsys):
    path = _v3_file(tmp_path, "requests = 3")
    assert main(["render", "--config", str(path)]) == 2
    assert "requests_origin" in capsys.readouterr().err


# --- reading schema 2 ----------------------------------------------------------------------------


def test_load_config_reads_a_schema_2_file_as_schema_3_without_changing_it(tmp_path):
    path = _v2_file(tmp_path, 'results = "."\ncontext = 4096')
    before = path.read_bytes()

    config = load_config(path)

    assert path.read_bytes() == before
    assert config.schema_version == 3
    assert config.guided.context == 4096
    assert (config.guided.users, config.guided.requests, config.guided.requests_origin) == (None, 1, "default")
    assert config.machines["laptop"].profile == "c0ffee0000000001"


def test_from_dict_reads_a_schema_2_dict_as_schema_3(tmp_path):
    data = {**_v3_dict(tmp_path), "schema_version": 2, "guided": {"context": 4096}}
    config = Configuration.from_dict(data)
    assert config.schema_version == 3
    assert config.guided.requests_origin == "default"
    assert data["schema_version"] == 2  # the caller's dict is left as it was


def test_a_schema_2_file_cannot_carry_a_key_of_schema_3(tmp_path):
    path = _v2_file(tmp_path, 'requests = 3\nrequests_origin = "entered"')
    with pytest.raises(ConfigError, match="schema 2"):
        load_config(path)


def test_a_schema_1_file_still_reads_and_lands_on_schema_3(tmp_path):
    shutil.copy(FIXTURES / "config_v1" / "config.toml", tmp_path / "modelroom.toml")
    config = load_config(tmp_path / "modelroom.toml")
    assert config.schema_version == 3
    assert config.guided.requests == 1


def test_a_schema_3_file_reads_all_three_keys(tmp_path):
    path = _v3_file(tmp_path, 'users = 25\nrequests = 3\nrequests_origin = "from_users"')
    config = load_config(path)
    assert (config.guided.users, config.guided.requests, config.guided.requests_origin) == (25, 3, "from_users")
    assert tomllib.loads(path.read_text(encoding="utf-8"))["schema_version"] == 3


# --- the scenario a later render computes for -----------------------------------------------------


def test_scenario_from_config_takes_the_requests_without_a_context(tmp_path):
    config = Configuration.from_dict(_v3_dict(tmp_path, requests=3, requests_origin="entered"))
    scenario = scenario_from_config(config)
    assert (scenario.context_requested, scenario.context_origin, scenario.requests) == (8192, "default", 3)


def test_scenario_from_config_takes_the_requests_with_a_kept_context(tmp_path):
    config = Configuration.from_dict(_v3_dict(tmp_path, context=4096, users=25, requests=3, requests_origin="from_users"))
    scenario = scenario_from_config(config)
    assert (scenario.context_requested, scenario.context_origin, scenario.requests) == (4096, "entered", 3)
