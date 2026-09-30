"""Tests for modelroom.catalog: the shipped catalog and its rules."""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.catalog import (
    CATALOG_ATTRIBUTION,
    CATALOG_DATA_LICENSE,
    CATALOG_FILE,
    CATALOG_MAX_BYTES,
    CATALOG_MAX_LINES,
    DISPLAYED_KINDS,
    PIPELINE_TAG_TO_KIND,
    Catalog,
    CatalogDocumentV2,
    CatalogError,
    CatalogFamily,
    CatalogModel,
    load_catalog,
)
from modelroom.contracts import SchemaVersionError
from modelroom.examples import EXAMPLES
from modelroom.search_pages import publisher_accounts
from modelroom.search_word import latest_model_names

FIXTURES = Path(__file__).parent / "fixtures"

# Every publisher account the shipped catalog has to name (decided 2026-09-25). The catalog is the
# positive list of publishers: a search hit is resolved only when the account of its base model is
# a publisher here, so an account missing from this list is a model nobody can pick.
REQUIRED_PUBLISHERS = (
    "mistralai",
    "utter-project",
    "openGPT-X",
    "Qwen",
    "deepseek-ai",
    "meta-llama",
    "google",
    "microsoft",
    "openai",
    "zai-org",
    "moonshotai",
    "ibm-granite",
    "nvidia",
    "allenai",
    "tiiuae",
    "CohereLabs",
    "LiquidAI",
    "HuggingFaceTB",
    "openbmb",
    "tencent",
    "baidu",
    "MiniMaxAI",
)
# The day every `latest` row of the shipped catalog was confirmed against its publisher page.
LATEST_CHECKED = date(2026, 9, 25)
# The two embedding rows the shipped catalog names, unchanged since the AP9-KA conversion
# (bau-brief, "Reichweite der Tabelle ...", "Bestandsschutz").
EMBEDDING_REPOS = ("Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-0.6B")
# Rows well above the sizes the rest of the catalog lists, kept through the conversion on purpose
# (bau-brief: "die als groß markierten Zeilen (catalog.toml:10-11)").
LARGE_REPOS = (
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "zai-org/GLM-5.3-Flash",
)


def test_shipped_catalog_loads_and_covers_the_example_configurations_families():
    catalog = load_catalog()
    assert {"qwen3.5", "deepseek-r1", "eurollm"} <= {family.name for family in catalog.families}
    assert catalog.is_publisher("Qwen") and not catalog.is_publisher("unsloth")


def test_shipped_catalog_names_every_required_publisher_account():
    catalog = load_catalog()
    publishers = [family.publisher for family in catalog.families]
    assert [account for account in REQUIRED_PUBLISHERS if account not in publishers] == []
    assert catalog.is_publisher("mistralai")


def test_shipped_catalog_claims_no_latest_without_evidence_and_one_successor():
    catalog = load_catalog()
    latest = [model for family in catalog.families for model in family.models if model.latest]
    assert latest, "the catalog states no current model at all"
    for model in latest:
        assert model.latest_source is not None and model.latest_checked == LATEST_CHECKED, model.hf_repo
    assert catalog.age_of("utter-project/EuroLLM-9B-Instruct") == ("legacy", "utter-project/EuroLLM-9B-Instruct-2512")
    assert catalog.age_of("Qwen/Qwen3.5-9B") == ("unknown", None)
    assert catalog.age_of("mistralai/Ministral-3-14B-Instruct-2512") == ("latest", None)


# The rows that were given a successor on 2026-09-25, each with a `successor_source` naming the
# publisher evidence (schema 2, migrated by `scripts/catalog_convert_v1.py` from the comment that
# used to sit above the row).
SUCCESSORS_2026_09_25 = {
    "mistralai/Magistral-Small-2509": "mistralai/Mistral-Small-4-119B-2603",
}
# Candidates that were checked on 2026-09-25 and keep age `unknown`: the publisher's pages follow a
# whole series, compare in a table or share a size and a version number, but name no model as
# replaced. A current line does not prove that every model of the line before it was replaced.
CHECKED_WITHOUT_SUCCESSOR = (
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.6-35B-A3B",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V3.2",
)
# A successor that is no `latest` row, with the reason. The model card of EuroLLM-9B-Instruct-2512
# calls itself an improved version of EuroLLM-9B-Instruct, but the publisher's collection lists both
# lines side by side, so nothing states the 2512 model current.
SUCCESSORS_NOT_LATEST = {"utter-project/EuroLLM-9B-Instruct": "the successor's own card, no current collection"}


def _rows_with_successor():
    catalog = load_catalog()
    return catalog, [(family, model) for family in catalog.families for model in family.models if model.successor]


def test_every_successor_of_the_shipped_catalog_is_a_latest_row_of_the_same_publisher():
    catalog, rows = _rows_with_successor()

    assert rows, "the catalog names no successor at all"
    for family, model in rows:
        target = catalog.model_for(model.successor)
        assert target is not None, f"{model.hf_repo}: successor {model.successor} is no row of the catalog"
        assert model.successor.split("/", 1)[0] == family.publisher, model.hf_repo
        if model.hf_repo not in SUCCESSORS_NOT_LATEST:
            assert target.latest, f"{model.hf_repo}: successor {model.successor} is not a latest row"


def test_the_rows_given_a_successor_on_2026_09_25_are_legacy():
    catalog = load_catalog()

    for repo, successor in SUCCESSORS_2026_09_25.items():
        assert catalog.age_of(repo) == ("legacy", successor), repo


def test_a_line_the_publisher_carries_as_current_is_no_successor_of_each_model_before_it():
    """No stand-in rule: neither the current line, nor a model of the same size, nor a table."""
    catalog = load_catalog()

    for repo in CHECKED_WITHOUT_SUCCESSOR:
        assert catalog.age_of(repo) == ("unknown", None), repo


def test_every_successor_row_names_its_evidence_and_the_day_it_was_checked():
    """Schema 2 (AP9-KA): `successor_source`/`successor_checked` are fields, not a comment."""
    catalog = load_catalog()
    for repo, successor in SUCCESSORS_2026_09_25.items():
        model = catalog.model_for(repo)
        assert model.successor == successor
        assert model.successor_source is not None and model.successor_checked == LATEST_CHECKED, repo
    # The successor with no current collection (`SUCCESSORS_NOT_LATEST`) still names its evidence.
    euro = catalog.model_for("utter-project/EuroLLM-9B-Instruct")
    assert euro.successor_source == "https://huggingface.co/utter-project/EuroLLM-9B-Instruct-2512"
    assert euro.successor_checked == LATEST_CHECKED


def test_every_shipped_row_carries_a_hub_source_a_full_ollama_pair_and_a_unique_id():
    """The validators already state these rules; this asserts the **shipped file** passes them."""
    catalog = load_catalog()
    rows = [(family, model) for family in catalog.families for model in family.models]
    for family, model in rows:
        assert model.source.startswith(f"https://huggingface.co/{family.publisher}/"), model.hf_repo
        assert (model.ollama_base is None) == (model.ollama_tag is None), model.hf_repo
        # A floating tag would name a different package tomorrow than the row was checked against.
        assert model.ollama_tag != "latest", model.hf_repo
        if not model.latest:
            assert (model.latest_source, model.latest_checked) == (None, None), model.hf_repo
    names = [family.name for family in catalog.families]
    assert names == [name.lower() for name in names] and all(name.isascii() for name in names)
    assert len(names) == len(set(names))
    repos = [model.hf_repo for _family, model in rows]
    assert len(repos) == len(set(repos))


def test_the_word_mistral_asks_the_account_of_that_publisher_alone():
    assert publisher_accounts(load_catalog(), "mistral") == ["mistralai"]


def test_the_word_qwen_still_asks_exactly_the_two_accounts_the_fixtures_bind():
    """The pinned `qwen` fixtures bind one page per asked account, so this answer is a contract."""
    assert publisher_accounts(load_catalog(), "qwen") == ["Qwen", "deepseek-ai"]


def test_shipped_catalog_is_inside_the_package():
    import modelroom

    assert (Path(modelroom.__file__).parent / "catalog.json").is_file()
    assert (Path(modelroom.__file__).parent / "catalog-rejected.json").is_file()


def test_catalog_excerpt_gives_latest_legacy_and_unknown():
    """The v1 (`.toml`) reader is unchanged: schema 1 still loads over the same entry point."""
    catalog = load_catalog(FIXTURES / "catalog_excerpt.toml")
    assert catalog.schema_version == 1
    assert catalog.age_of("acme/Nova-7B-2512") == ("latest", None)
    assert catalog.age_of("acme/Nova-7B") == ("legacy", "acme/Nova-7B-2512")
    assert catalog.age_of("acme/Nova-1B") == ("unknown", None)
    assert catalog.age_of("acme/Not-Listed") == ("unknown", None)
    assert catalog.model_for("acme/Nova-7B").ollama_base == "nova"
    # v1 carries no `kind`; every row reads as `text` in memory (bau-brief, "Leser im Paket").
    assert catalog.model_for("acme/Nova-7B-2512").kind == "text"


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


# --- Schema 2 (AP9-KA): a document with every field, grouped into the same shape schema 1 builds ---


def _document(models: list[dict] | None = None, **overrides) -> dict:
    base = json.loads((FIXTURES / "catalog_v2_valid.json").read_text(encoding="utf-8"))
    if models is not None:
        base["models"] = models
    base.update(overrides)
    return base


def _row(index: int = 0, **overrides) -> dict:
    row = copy.deepcopy(json.loads((FIXTURES / "catalog_v2_valid.json").read_text(encoding="utf-8"))["models"][index])
    row.update(overrides)
    return row


def test_shipped_catalog_is_schema_2_with_the_fixed_license_and_attribution():
    raw = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    assert raw["data_license"] == CATALOG_DATA_LICENSE == "CC-BY-4.0"
    assert raw["attribution"] == CATALOG_ATTRIBUTION
    catalog = load_catalog()
    assert catalog.data_license == CATALOG_DATA_LICENSE
    assert catalog.attribution == CATALOG_ATTRIBUTION


def test_the_shipped_catalog_keeps_every_one_of_the_sixty_two_v1_rows():
    """Bestandsschutz (bau-brief, "Reichweite der Tabelle ..."): every row of the pre-AP9-KA
    catalog survives the conversion, including the embedding rows and the large-model rows."""
    catalog = load_catalog()
    repos = [model.hf_repo for family in catalog.families for model in family.models]
    assert len(repos) == 62
    for repo in EMBEDDING_REPOS:
        model = catalog.model_for(repo)
        assert model is not None and model.kind == "embedding", repo
    for repo in LARGE_REPOS:
        assert catalog.model_for(repo) is not None, repo


def test_catalog_v2_fixture_loads_with_a_latest_a_legacy_and_an_embedding_row():
    catalog = load_catalog(FIXTURES / "catalog_v2_valid.json")
    assert catalog.schema_version == 2
    assert catalog.age_of("acme/Nova-7B-2512") == ("latest", None)
    assert catalog.age_of("acme/Nova-7B") == ("legacy", "acme/Nova-7B-2512")
    embed = catalog.model_for("acme/Nova-Embed")
    assert embed.kind == "embedding" and embed.latest
    assert catalog.publisher_info[0].account == "acme"
    assert catalog.packagers == ["packager"]


def test_display_selection_shows_only_latest_text_and_vision_kinds():
    """Anzeige-Auswahl (Frage 3 a): `latest` alone is not enough, `kind` must be `text`/`vision`."""
    catalog = load_catalog(FIXTURES / "catalog_v2_kinds.json")
    names = latest_model_names(catalog)
    assert names == ["Line-Text", "Line-Vision"]
    for kind in ("embedding", "speech-recognition", "speech-synthesis", "audio", "image-generation"):
        assert kind not in DISPLAYED_KINDS


def test_todays_shipped_display_list_is_unchanged_by_the_kind_filter():
    """Every `latest` row of the shipped catalog is already `text` or `vision`, so the filter
    added by AP9-KA changes nothing it used to show (bau-brief, "Reichweite der Tabelle ...")."""
    catalog = load_catalog()
    names = latest_model_names(catalog)
    assert len(names) == 22
    assert "GLM-5.3-Flash" in names


def test_pipeline_tag_to_kind_names_all_eleven_values():
    """Frage 2 b: the mapping is a constant of the schema and is repeated here verbatim."""
    expected = {
        "text-generation": "text",
        "image-text-to-text": "vision",
        "feature-extraction": "embedding",
        "sentence-similarity": "embedding",
        "automatic-speech-recognition": "speech-recognition",
        "text-to-speech": "speech-synthesis",
        "text-to-audio": "audio",
        "audio-to-audio": "audio",
        "audio-text-to-text": "audio",
        "text-to-image": "image-generation",
        "image-to-image": "image-generation",
    }
    assert PIPELINE_TAG_TO_KIND == expected
    assert len(expected) == 11


def test_every_kind_of_the_schema_loads_a_valid_row():
    catalog = load_catalog(FIXTURES / "catalog_v2_kinds.json")
    kinds = {model.kind for family in catalog.families for model in family.models}
    assert kinds == set(PIPELINE_TAG_TO_KIND.values())
    assert len(kinds) == 7


def test_successor_source_migration_of_the_shipped_catalog_passes_the_ownership_rule():
    """Migration der Nachfolger: both bestand successors carry `successor_source` after the v1 to
    v2 conversion, and loading the shipped file already proves the ownership rule holds."""
    catalog = load_catalog()
    magistral = catalog.model_for("mistralai/Magistral-Small-2509")
    assert magistral.successor_source == "https://huggingface.co/mistralai/Mistral-Small-4-119B-2603"
    assert magistral.successor_checked == LATEST_CHECKED
    euro = catalog.model_for("utter-project/EuroLLM-9B-Instruct")
    assert euro.successor_source == "https://huggingface.co/utter-project/EuroLLM-9B-Instruct-2512"
    assert euro.successor_checked == LATEST_CHECKED


@pytest.mark.parametrize(
    "source",
    [
        "https://huggingface.co/packager/Nova-7B-2512",  # a foreign account's card
        "https://huggingface.co/acme/Some-Third-Model",  # a third model of the same publisher
    ],
)
def test_successor_source_rejects_a_card_that_is_neither_the_model_nor_its_successor(source):
    row = _row(1, successor_source=source)
    with pytest.raises(ValidationError, match="successor_source"):
        CatalogDocumentV2.model_validate(_document(models=[_row(0), row]))


def test_successor_source_rejects_http():
    row = _row(1, successor_source="http://huggingface.co/acme/Nova-7B-2512")
    with pytest.raises(ValidationError, match="https"):
        CatalogDocumentV2.model_validate(_document(models=[_row(0), row]))


def test_successor_source_set_without_a_successor_is_rejected():
    row = _row(successor_source="https://huggingface.co/acme/Nova-7B-2512", successor_checked="2026-09-23")
    with pytest.raises(ValidationError, match="successor"):
        CatalogDocumentV2.model_validate(_document(models=[row]))


def test_unknown_field_is_rejected():
    document = _document()
    document["unexpected"] = "nope"
    with pytest.raises(ValidationError):
        CatalogDocumentV2.model_validate(document)


def test_schema_version_3_is_rejected():
    with pytest.raises((ValidationError, SchemaVersionError)):
        CatalogDocumentV2.model_validate(_document(schema_version=3))
    path = FIXTURES / "catalog_v2_valid.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = 3
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "catalog.json"
        broken.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(SchemaVersionError):
            load_catalog(broken)


def test_kind_outside_the_list_is_rejected():
    with pytest.raises(ValidationError, match="kind"):
        CatalogDocumentV2.model_validate(_document(models=[_row(kind="text-to-video")]))


@pytest.mark.parametrize("value", ["acme/nova", "nova model", "Nova"])
def test_ollama_base_with_a_slash_space_or_uppercase_is_rejected(value):
    with pytest.raises(ValidationError, match="ollama_base"):
        CatalogDocumentV2.model_validate(_document(models=[_row(ollama_base=value, ollama_tag="7b")]))


def test_control_characters_in_note_are_rejected():
    with pytest.raises(ValidationError, match="note"):
        CatalogDocumentV2.model_validate(_document(models=[_row(note="line one\nline two")]))


def test_a_file_over_one_mebibyte_is_rejected(tmp_path):
    document = _document()
    document["models"][0]["note"] = "x" * (CATALOG_MAX_BYTES + 1)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CatalogError, match="bytes"):
        load_catalog(path)


def test_a_file_over_five_thousand_lines_is_rejected(tmp_path):
    path = tmp_path / "catalog.json"
    padding = "\n" * (CATALOG_MAX_LINES + 10)
    text = json.dumps(_document()) + padding
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CatalogError, match="lines"):
        load_catalog(path)


def test_publisher_not_in_publishers_is_rejected():
    with pytest.raises(ValidationError, match="publishers"):
        CatalogDocumentV2.model_validate(_document(models=[_row(publisher="ghost", hf_repo="ghost/Nova-7B-2512")]))


def test_gguf_repos_with_an_invalid_id_is_rejected():
    with pytest.raises(ValidationError):
        CatalogDocumentV2.model_validate(_document(models=[_row(gguf_repos=["not a repo id"])]))


def test_latest_source_without_https_is_rejected():
    with pytest.raises(ValidationError, match="https"):
        CatalogDocumentV2.model_validate(_document(models=[_row(latest_source="ftp://huggingface.co/acme")]))


@pytest.mark.parametrize("license_value", ["CC0-1.0", "MIT"])
def test_data_license_other_than_cc_by_4_0_is_rejected(license_value):
    with pytest.raises(ValidationError, match="data_license"):
        CatalogDocumentV2.model_validate(_document(data_license=license_value))


def test_data_license_missing_is_rejected():
    document = _document()
    del document["data_license"]
    with pytest.raises(ValidationError):
        CatalogDocumentV2.model_validate(document)


def test_attribution_missing_or_different_is_rejected():
    document = _document()
    del document["attribution"]
    with pytest.raises(ValidationError):
        CatalogDocumentV2.model_validate(document)
    with pytest.raises(ValidationError, match="attribution"):
        CatalogDocumentV2.model_validate(_document(attribution="modelroom catalog, someone else"))
