"""Tests for modelroom.catalog: the shipped catalog and its rules."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.catalog import Catalog, CatalogError, CatalogFamily, CatalogModel, load_catalog
from modelroom.contracts import SchemaVersionError
from modelroom.examples import EXAMPLES

FIXTURES = Path(__file__).parent / "fixtures"


def test_shipped_catalog_loads_and_covers_the_example_configurations_families():
    catalog = load_catalog()
    assert {family.name for family in catalog.families} == {"qwen3.5", "deepseek-r1", "eurollm"}
    assert catalog.is_publisher("Qwen") and not catalog.is_publisher("unsloth")


def test_shipped_catalog_claims_no_latest_without_evidence_and_one_successor():
    catalog = load_catalog()
    assert not any(model.latest for family in catalog.families for model in family.models)
    assert catalog.age_of("utter-project/EuroLLM-9B-Instruct") == ("legacy", "utter-project/EuroLLM-9B-Instruct-2512")
    assert catalog.age_of("Qwen/Qwen3.5-9B") == ("unknown", None)


def test_shipped_catalog_is_inside_the_package():
    import modelroom

    assert (Path(modelroom.__file__).parent / "catalog.toml").is_file()


def test_catalog_excerpt_gives_latest_legacy_and_unknown():
    catalog = load_catalog(FIXTURES / "catalog_excerpt.toml")
    assert catalog.age_of("acme/Nova-7B-2512") == ("latest", None)
    assert catalog.age_of("acme/Nova-7B") == ("legacy", "acme/Nova-7B-2512")
    assert catalog.age_of("acme/Nova-1B") == ("unknown", None)
    assert catalog.age_of("acme/Not-Listed") == ("unknown", None)
    assert catalog.model_for("acme/Nova-7B").ollama_base == "nova"


def test_a_model_with_a_successor_is_not_latest():
    evidence = {"latest_source": "https://huggingface.co/acme", "latest_checked": "2026-09-23"}
    with pytest.raises(ValidationError, match="successor is not latest"):
        CatalogModel.model_validate({**EXAMPLES["CatalogModel"], "latest": True, **evidence})


def _latest_model(**changes) -> dict:
    model = copy.deepcopy(EXAMPLES["CatalogFamily"]["models"][1])
    model.update(changes)
    return model


@pytest.mark.parametrize("missing", ["latest_source", "latest_checked"])
def test_latest_needs_its_evidence_and_check_date(missing):
    with pytest.raises(ValidationError, match="latest"):
        CatalogModel.model_validate(_latest_model(**{missing: None}))


def test_evidence_without_latest_is_refused():
    legacy = {**EXAMPLES["CatalogModel"], "latest_source": "https://huggingface.co/acme", "latest_checked": "2026-09-23"}
    with pytest.raises(ValidationError, match="latest"):
        CatalogModel.model_validate(legacy)


@pytest.mark.parametrize("source", ["http://acme.example/models", "https://", "https:///acme", "acme.example/models"])
def test_latest_source_is_an_https_page_with_a_host(source):
    with pytest.raises(ValidationError, match="https"):
        CatalogModel.model_validate(_latest_model(latest_source=source))


@pytest.mark.parametrize(
    "source",
    [
        "https://huggingface.co/acme",
        "https://huggingface.co/acme/",
        "https://huggingface.co/acme?tab=models",
        "https://huggingface.co/collections/acme/nova-2512-0123abcd",
        "https://acme.example/models",
    ],
)
def test_latest_source_on_the_hub_is_the_publisher_page_or_collection(source):
    family = copy.deepcopy(EXAMPLES["CatalogFamily"])
    family["models"][1]["latest_source"] = source
    CatalogFamily.model_validate(family)


@pytest.mark.parametrize(
    "source",
    [
        "https://huggingface.co/acme/Nova-7B-2512",  # the model's own page proves no ranking among models
        "https://huggingface.co/packager",
        "https://huggingface.co/collections/packager/nova-0123abcd",
        "https://huggingface.co:443/acme/Nova-7B-2512",  # the Hub with an explicit port is still the Hub
        "https://HuggingFace.co/acme/Nova-7B-2512",  # host names are case-insensitive
        "https://www.huggingface.co/acme/Nova-7B-2512",
        "https://huggingface.co/collections/acme",  # the collections root is no collection
    ],
)
def test_latest_source_on_the_hub_must_be_the_publishers(source):
    family = copy.deepcopy(EXAMPLES["CatalogFamily"])
    family["models"][1]["latest_source"] = source
    with pytest.raises(ValidationError, match="latest_source"):
        CatalogFamily.model_validate(family)


def test_latest_checked_is_a_date():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate(_latest_model(latest_checked="last week"))


def test_a_model_is_not_its_own_successor():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate({**EXAMPLES["CatalogModel"], "successor": "acme/Nova-7B"})


def test_source_must_be_a_hugging_face_page():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate({**EXAMPLES["CatalogModel"], "source": "https://example.org/nova"})


def test_ollama_name_comes_as_a_pair():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate({**EXAMPLES["CatalogModel"], "ollama_tag": None})


def test_models_stay_under_the_family_publisher():
    family = copy.deepcopy(EXAMPLES["CatalogFamily"])
    family["models"][1]["hf_repo"] = "packager/Nova-7B-2512"
    with pytest.raises(ValidationError, match="publisher"):
        CatalogFamily.model_validate(family)


def test_successor_cycle_is_rejected():
    family = copy.deepcopy(EXAMPLES["CatalogFamily"])
    family["models"][1].update({"latest": False, "latest_source": None, "latest_checked": None, "successor": "acme/Nova-7B"})
    with pytest.raises(ValidationError, match="cycle"):
        CatalogFamily.model_validate(family)


def test_successor_cycle_across_two_families_is_rejected():
    catalog = {
        "schema_version": 1,
        "families": [
            {"name": "a", "publisher": "acme", "models": [{"hf_repo": "acme/A", "successor": "acme/B", "source": "https://huggingface.co/acme/A"}]},
            {"name": "b", "publisher": "acme", "models": [{"hf_repo": "acme/B", "successor": "acme/A", "source": "https://huggingface.co/acme/B"}]},
        ],
    }
    with pytest.raises(ValidationError, match="cycle"):
        Catalog.model_validate(catalog)


def test_hf_repo_is_unique_across_the_catalog():
    catalog = copy.deepcopy(EXAMPLES["Catalog"])
    second = copy.deepcopy(catalog["families"][0])
    second["name"] = "nova-copy"
    catalog["families"].append(second)
    with pytest.raises(ValidationError):
        Catalog.model_validate(catalog)


def test_load_catalog_errors(tmp_path):
    path = tmp_path / "catalog.toml"
    path.write_text("schema_version = 2\n", encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        load_catalog(path)
    path.write_text("schema_version = 1\n[[families]]\nname = 'x'\n", encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)
    with pytest.raises(CatalogError):
        load_catalog(tmp_path / "missing.toml")
