"""Tests for modelroom.catalog: the shipped catalog and its rules."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.catalog import (
    CATALOG_FILE,
    CATALOG_MAX_BYTES,
    CATALOG_MAX_TEXT_LEN,
    DISPLAYED_KINDS,
    PIPELINE_TAG_TO_KIND,
    Catalog,
    CatalogError,
    CatalogFamily,
    CatalogModel,
    load_catalog,
)
from modelroom.catalog_schema import (
    CATALOG_ATTRIBUTION,
    CATALOG_DATA_LICENSE,
    CATALOG_MAX_GGUF_REPOS,
    CATALOG_MAX_MODELS,
    CATALOG_MAX_PACKAGERS,
    CATALOG_MAX_PUBLISHERS,
    CatalogDocumentV2,
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
# The two embedding rows the shipped catalog names, kept through the move from schema 1 to 2.
EMBEDDING_REPOS = ("Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-0.6B")
# Rows well above the sizes the rest of the catalog lists, kept through the move on purpose.
LARGE_REPOS = (
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "zai-org/GLM-5.3-Flash",
)


# --- The dicts below set `kind` explicitly. `CatalogModel` itself defaults it to `"text"`; only
# a schema-2 document refuses a row without it (`CatalogDocumentV2`, through `model_fields_set`).


def _model_dict(**overrides) -> dict:
    return {**EXAMPLES["CatalogModel"], "kind": "text", **overrides}


def _family_dict() -> dict:
    family = copy.deepcopy(EXAMPLES["CatalogFamily"])
    for model in family["models"]:
        model.setdefault("kind", "text")
    return family


def _catalog_dict() -> dict:
    catalog = copy.deepcopy(EXAMPLES["Catalog"])
    for family in catalog["families"]:
        for model in family["models"]:
            model.setdefault("kind", "text")
    return catalog


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
# publisher evidence (schema 2 carries it in fields; schema 1 had it in the comment above the row).
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
    """Schema 2: `successor_source`/`successor_checked` are fields, not a comment."""
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
    # v1 carries no `kind`; every row reads as `text` in memory.
    assert catalog.model_for("acme/Nova-7B-2512").kind == "text"


def test_a_model_with_a_successor_is_not_latest():
    evidence = {"latest_source": "https://huggingface.co/acme", "latest_checked": "2026-09-23"}
    with pytest.raises(ValidationError, match="successor is not latest"):
        CatalogModel.model_validate(_model_dict(latest=True, **evidence))


def _latest_model(**changes) -> dict:
    model = copy.deepcopy(EXAMPLES["CatalogFamily"]["models"][1])
    model.setdefault("kind", "text")
    model.update(changes)
    return model


@pytest.mark.parametrize("missing", ["latest_source", "latest_checked"])
def test_latest_needs_its_evidence_and_check_date(missing):
    with pytest.raises(ValidationError, match="latest"):
        CatalogModel.model_validate(_latest_model(**{missing: None}))


def test_evidence_without_latest_is_refused():
    legacy = _model_dict(latest_source="https://huggingface.co/acme", latest_checked="2026-09-23")
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
    family = _family_dict()
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
    family = _family_dict()
    family["models"][1]["latest_source"] = source
    with pytest.raises(ValidationError, match="latest_source"):
        CatalogFamily.model_validate(family)


def test_latest_checked_is_a_date():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate(_latest_model(latest_checked="last week"))


def test_a_model_is_not_its_own_successor():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate(_model_dict(successor="acme/Nova-7B"))


def test_source_must_be_a_hugging_face_page():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate(_model_dict(source="https://example.org/nova"))


def test_ollama_name_comes_as_a_pair():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate(_model_dict(ollama_tag=None))


def test_models_stay_under_the_family_publisher():
    family = _family_dict()
    family["models"][1]["hf_repo"] = "packager/Nova-7B-2512"
    with pytest.raises(ValidationError, match="publisher"):
        CatalogFamily.model_validate(family)


def test_successor_cycle_is_rejected():
    family = _family_dict()
    family["models"][1].update({"latest": False, "latest_source": None, "latest_checked": None, "successor": "acme/Nova-7B"})
    with pytest.raises(ValidationError, match="cycle"):
        CatalogFamily.model_validate(family)


def test_successor_cycle_across_two_families_is_rejected():
    catalog = {
        "schema_version": 1,
        "families": [
            {"name": "a", "publisher": "acme", "models": [{"hf_repo": "acme/A", "successor": "acme/B", "source": "https://huggingface.co/acme/A", "kind": "text"}]},
            {"name": "b", "publisher": "acme", "models": [{"hf_repo": "acme/B", "successor": "acme/A", "source": "https://huggingface.co/acme/B", "kind": "text"}]},
        ],
    }
    with pytest.raises(ValidationError, match="cycle"):
        Catalog.model_validate(catalog)


def test_hf_repo_is_unique_across_the_catalog():
    catalog = _catalog_dict()
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


# --- Schema 2: a document with every field, grouped into the same shape schema 1 builds ---------


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
    """Every row of the schema-1 catalog survives the move to schema 2, including the embedding
    rows and the large-model rows."""
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
    """The pages of a search with no word: `latest` alone is not enough, `kind` must be `text`/`vision`."""
    catalog = load_catalog(FIXTURES / "catalog_v2_kinds.json")
    names = latest_model_names(catalog)
    assert names == ["Line-Text", "Line-Vision"]
    for kind in ("embedding", "speech-recognition", "speech-synthesis", "audio", "image-generation", "video-generation"):
        assert kind not in DISPLAYED_KINDS


def test_todays_shipped_display_list_is_unchanged_by_the_kind_filter():
    """Every `latest` row of the shipped catalog is already `text` or `vision`, so the `kind`
    filter changes nothing the list used to show."""
    catalog = load_catalog()
    names = latest_model_names(catalog)
    assert len(names) == 22
    assert "GLM-5.3-Flash" in names


def test_pipeline_tag_to_kind_names_all_fifteen_values():
    """The mapping is a constant of the schema and is repeated here verbatim."""
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
        "text-to-video": "video-generation",
        "image-to-video": "video-generation",
        "image-text-to-video": "video-generation",
        "video-to-video": "video-generation",
    }
    assert PIPELINE_TAG_TO_KIND == expected
    assert len(expected) == 15


def test_every_kind_of_the_schema_loads_a_valid_row():
    catalog = load_catalog(FIXTURES / "catalog_v2_kinds.json")
    kinds = {model.kind for family in catalog.families for model in family.models}
    assert kinds == set(PIPELINE_TAG_TO_KIND.values())
    assert len(kinds) == 8


def test_successor_source_migration_of_the_shipped_catalog_passes_the_ownership_rule():
    """Both successors of the schema-1 catalog carry `successor_source` in schema 2, and loading
    the shipped file already proves the ownership rule holds."""
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


def _padded(text: str, target_bytes: int) -> str:
    """`text` (pure ASCII JSON) with extra spaces after the opening brace, so its serialized
    byte length is exactly `target_bytes`; whitespace there is insignificant JSON."""
    pad = target_bytes - len(text.encode("utf-8"))
    assert pad >= 0, "text is already longer than target_bytes"
    return "{" + " " * pad + text[1:]


def test_reading_stops_at_the_byte_cap_boundary(tmp_path):
    """The reader takes at most `CATALOG_MAX_BYTES + 1` bytes off disk, so
    a file exactly at the cap loads and one byte more is rejected -- proven at the boundary,
    not by loading a multi-megabyte file into memory first."""
    text = json.dumps(_document())
    at_cap = _padded(text, CATALOG_MAX_BYTES)
    over_cap = _padded(text, CATALOG_MAX_BYTES + 1)
    path = tmp_path / "catalog.json"
    path.write_text(at_cap, encoding="utf-8")
    load_catalog(path)
    path.write_text(over_cap, encoding="utf-8")
    with pytest.raises(CatalogError, match="bytes"):
        load_catalog(path)


def test_more_than_five_thousand_models_is_rejected(tmp_path):
    """The limit counts model **entries**, not text lines -- 5000 short, unique, minimal rows
    load and 5001 are rejected, though the file stays under 1 MiB (every optional field left
    out, so each defaults, keeping the file compact on purpose)."""
    rows = [
        {"hf_repo": f"acme/G{i}", "source": f"https://huggingface.co/acme/G{i}", "kind": "text", "hub_pipeline_tag": "text-generation", "family": f"g{i}", "publisher": "acme", "added": "2026-09-25"}
        for i in range(CATALOG_MAX_MODELS + 1)
    ]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_document(models=rows[:CATALOG_MAX_MODELS])), encoding="utf-8")
    assert len(load_catalog(path).families) == CATALOG_MAX_MODELS
    text = json.dumps(_document(models=rows))
    assert len(text.encode("utf-8")) <= CATALOG_MAX_BYTES, "fixture drifted: no longer a compact-JSON scenario"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CatalogError, match="models"):
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


# --- Required fields, evidence, limits and broken input of a schema-2 document --------------------


# `kind` defaults to `text` on `CatalogModel`, but a schema-2 document refuses a row without it.


def test_kind_missing_in_schema_2_is_rejected_via_load_catalog(tmp_path):
    document = _document()
    del document["models"][0]["kind"]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CatalogError, match="kind"):
        load_catalog(path)


def test_kind_default_of_text_applies_only_to_the_v1_reader():
    """A v1 row with no `kind` at all reads as `text` (unchanged); the same omission in a v2
    document is a missing required field, not a silent default (previous section)."""
    catalog = load_catalog(FIXTURES / "catalog_excerpt.toml")
    assert catalog.model_for("acme/Nova-7B-2512").kind == "text"


# `successor`, `successor_source` and `successor_checked` must be set together.


@pytest.mark.parametrize(
    "fields",
    [
        {"successor": "acme/Nova-7B-2512"},
        {"successor_source": "https://huggingface.co/acme/Nova-7B-2512"},
        {"successor_checked": "2026-09-23"},
        {"successor": "acme/Nova-7B-2512", "successor_source": "https://huggingface.co/acme/Nova-7B-2512"},
        {"successor": "acme/Nova-7B-2512", "successor_checked": "2026-09-23"},
        {"successor_source": "https://huggingface.co/acme/Nova-7B-2512", "successor_checked": "2026-09-23"},
    ],
)
def test_every_partial_combination_of_successor_evidence_is_rejected(fields):
    overrides = {"successor": None, "successor_source": None, "successor_checked": None, **fields}
    row = _row(1, **overrides)
    with pytest.raises(ValidationError, match="successor"):
        CatalogDocumentV2.model_validate(_document(models=[_row(0), row]))


# Every text value caps at 300 characters, list entries included; every list has its limit.


def test_license_over_three_hundred_characters_is_rejected():
    with pytest.raises(ValidationError):
        CatalogModel.model_validate(_model_dict(license="x" * (CATALOG_MAX_TEXT_LEN + 1)))


def _filled(prefix: str, length: int, fill: str = "a") -> str:
    """`prefix` padded with `fill` to exactly `length` characters."""
    assert len(prefix) < length
    return prefix + fill * (length - len(prefix))


def _row_text(field: str, length: int) -> list[dict]:
    """The fixture's rows with one text value of row 0 (or row 1 for the successor) at `length`."""
    if field == "successor":
        target = _filled("acme/S", length)
        return [_row(0), _row(1, successor=target, successor_source="https://acme.example/successors")]
    if field == "successor_source":
        return [_row(0), _row(1, successor_source=_filled("https://acme.example/", length))]
    values = {
        "hf_repo": _filled("acme/N", length),
        "source": _filled("https://huggingface.co/acme/N", length),
        "latest_source": _filled("https://huggingface.co/collections/acme/n", length),
        "family": _filled("f", length),
        "ollama_base": _filled("o", length),
        "ollama_tag": _filled("t", length),
        "license": _filled("l", length),
        "note": _filled("n", length),
        "gguf_repos": [_filled("packager/G", length)],
    }
    return [_row(0, **{field: values[field]})]


def _document_text(field: str, length: int) -> dict:
    if field == "publishers.account":
        document = _document()
        document["publishers"].append({"account": _filled("p", length), "name": "Other", "region": None})
        return document
    if field == "publishers.name":
        document = _document()
        document["publishers"][0]["name"] = _filled("N", length)
        return document
    if field == "packagers":
        return _document(packagers=["packager", _filled("k", length)])
    return _document(models=_row_text(field, length))


TEXT_FIELDS = (
    "hf_repo", "source", "latest_source", "successor", "successor_source", "family", "ollama_base",
    "ollama_tag", "license", "note", "gguf_repos", "publishers.account", "publishers.name", "packagers",
)


@pytest.mark.parametrize("field", TEXT_FIELDS)
def test_every_text_value_loads_at_three_hundred_characters_and_fails_at_three_hundred_one(field, tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_document_text(field, CATALOG_MAX_TEXT_LEN)), encoding="utf-8")
    load_catalog(path)
    path.write_text(json.dumps(_document_text(field, CATALOG_MAX_TEXT_LEN + 1)), encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)


def _document_count(field: str, count: int) -> dict:
    if field == "gguf_repos":
        return _document(models=[_row(gguf_repos=[f"packager/Nova-{i}-GGUF" for i in range(count)])])
    if field == "publishers":
        extra = [{"account": f"p{i}", "name": f"P{i}", "region": None} for i in range(count - 1)]
        return _document(publishers=_document()["publishers"] + extra)
    extra = [f"k{i}" for i in range(count - 1)]
    return _document(packagers=["packager", *extra])


@pytest.mark.parametrize(
    ("field", "limit"),
    [("gguf_repos", CATALOG_MAX_GGUF_REPOS), ("publishers", CATALOG_MAX_PUBLISHERS), ("packagers", CATALOG_MAX_PACKAGERS)],
)
def test_every_list_loads_at_its_limit_and_fails_one_entry_above(field, limit, tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_document_count(field, limit)), encoding="utf-8")
    load_catalog(path)
    path.write_text(json.dumps(_document_count(field, limit + 1)), encoding="utf-8")
    with pytest.raises(CatalogError, match=field):
        load_catalog(path)


def test_the_list_limits_are_the_published_ones():
    assert (CATALOG_MAX_GGUF_REPOS, CATALOG_MAX_PUBLISHERS, CATALOG_MAX_PACKAGERS) == (50, 500, 50)
    assert (CATALOG_MAX_MODELS, CATALOG_MAX_TEXT_LEN, CATALOG_MAX_BYTES) == (5000, 300, 1024 * 1024)


# A `RecursionError` from deeply nested JSON or TOML, and a `to_catalog()` failure that only
# surfaces once rows are grouped into families, all end as `CatalogError`, not an uncaught
# exception or a raw pydantic `ValidationError`.


def test_deeply_nested_json_is_a_catalog_error_not_a_recursion_error(tmp_path):
    nested = "[" * 6000 + "]" * 6000
    text = '{"x": ' + nested + "}"
    path = tmp_path / "catalog.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("catalog.toml", "schema_version = 1\nx = " + "9" * 5000 + "\n"),
        ("catalog.json", '{"schema_version": 2, "x": ' + "9" * 5000 + "}"),
    ],
    ids=["toml", "json"],
)
def test_an_integer_past_the_digit_limit_is_a_catalog_error(name, text, tmp_path):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_deeply_nested_toml_is_a_catalog_error_not_a_recursion_error(tmp_path):
    text = "schema_version = 1\nx = " + "[" * 6000 + "]" * 6000 + "\n"
    path = tmp_path / "catalog.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_a_foreign_hub_publisher_in_latest_source_fails_in_to_catalog_as_a_catalog_error(tmp_path):
    """`latest_source` pointing at a foreign account passes `CatalogDocumentV2` (it has no
    publisher-evidence rule of its own) and is only caught once `to_catalog()` groups the row
    into its `CatalogFamily` -- outside the old code's `try/except ValidationError`."""
    row = _row(latest_source="https://huggingface.co/someone-else")
    document = _document(models=[row])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CatalogError, match="latest_source"):
        load_catalog(path)


# A `gguf_repos` entry's owner must be the model's own publisher or a listed packager.


def test_gguf_repos_with_a_foreign_owner_is_rejected():
    row = _row(gguf_repos=["foreign/Unrelated-GGUF"])
    with pytest.raises(ValidationError, match="gguf_repos"):
        CatalogDocumentV2.model_validate(_document(models=[row]))


def test_gguf_repos_owned_by_the_publisher_or_a_listed_packager_is_accepted():
    row = _row(gguf_repos=["acme/Nova-7B-2512-GGUF", "packager/Nova-7B-2512-GGUF"])
    CatalogDocumentV2.model_validate(_document(models=[row]))


# `parameters_b` accepts only finite, non-negative numbers or `null` -- no strings, no
# `NaN`/`Infinity` (Python's `json` module parses those non-standard bare tokens by default), and
# no integer too large for a float.


@pytest.mark.parametrize("broken", [float("nan"), float("inf"), float("-inf"), "7.5", pytest.param(10**400, id="int-beyond-float")])
def test_parameters_b_rejects_non_finite_and_non_numeric_values(broken, tmp_path):
    document = _document()
    document["models"][0]["parameters_b"] = broken
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CatalogError, match="parameters_b"):
        load_catalog(path)


def test_parameters_b_accepts_a_plain_integer():
    row = _row(parameters_b=7)
    document = CatalogDocumentV2.model_validate(_document(models=[row]))
    assert document.models[0].parameters_b == 7.0


# The schema is told apart by content, never by the file's extension.


def test_schema_2_json_under_a_toml_shaped_name_still_loads_as_schema_2(tmp_path):
    document = _document()
    path = tmp_path / "catalog.toml"
    path.write_text(json.dumps(document), encoding="utf-8")
    catalog = load_catalog(path)
    assert catalog.schema_version == 2


def test_schema_1_toml_under_a_json_shaped_name_still_loads_as_schema_1(tmp_path):
    reference_text = (FIXTURES / "catalog_v1_reference.toml").read_text(encoding="utf-8")
    path = tmp_path / "catalog.json"
    path.write_text(reference_text, encoding="utf-8")
    catalog = load_catalog(path)
    assert catalog.schema_version == 1
    assert catalog.is_publisher("mistralai")


# a fixed schema-1 reference (the last shipped `catalog.toml`), compared value by
# value across all 62 rows, plus the exact ordered display list -- not a count and a few names.


# Every field a schema-1 row can carry; each one has to come through the move unchanged.
V1_FIELDS = frozenset(
    {"hf_repo", "latest", "latest_source", "latest_checked", "successor", "ollama_base", "ollama_tag", "source"}
)
# SHA-256 of the reference file with LF line ends: the last `catalog.toml` that shipped, byte for byte.
V1_REFERENCE_SHA256 = "5bee020b9c47fde3d3f9dded72da585e53b4b859842d7bb363a6613d14f5bceb"


def _rows_in_order(catalog: Catalog) -> list[tuple[str, str, dict]]:
    return [
        (family.name, family.publisher, model.model_dump(include=V1_FIELDS))
        for family in catalog.families
        for model in family.models
    ]


def test_the_v1_reference_is_the_last_shipped_schema_1_file():
    raw = (FIXTURES / "catalog_v1_reference.toml").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == V1_REFERENCE_SHA256


def test_the_v1_fields_cover_every_field_a_schema_1_row_reads():
    reference = load_catalog(FIXTURES / "catalog_v1_reference.toml")
    set_fields = {name for family in reference.families for model in family.models for name in model.model_fields_set}
    assert set_fields <= V1_FIELDS
    assert {"hf_repo", "source", "latest", "successor", "ollama_base"} <= set_fields


def test_the_shipped_catalog_matches_the_v1_reference_value_by_value_for_every_row():
    """All 62 rows, in the same order, under the same family and publisher, with every schema-1
    field (`source` included) unchanged."""
    reference = _rows_in_order(load_catalog(FIXTURES / "catalog_v1_reference.toml"))
    shipped = _rows_in_order(load_catalog())
    assert len(reference) == 62
    assert [row[2]["hf_repo"] for row in shipped] == [row[2]["hf_repo"] for row in reference]
    for before, after in zip(reference, shipped, strict=True):
        assert after == before, before[2]["hf_repo"]
    magistral = load_catalog().model_for("mistralai/Magistral-Small-2509")
    assert magistral.source == "https://huggingface.co/mistralai/Magistral-Small-2509"
    assert magistral.successor_source == "https://huggingface.co/mistralai/Mistral-Small-4-119B-2603"


# The schema-1 file explained, in a comment above the row, why a statement is missing: no
# `latest`, no successor, no Ollama name, or no row for a sibling model. Schema 2 carries each of
# those reasons in `note`; every other row of the shipped catalog has none.
NOTES = {
    "mistralai/Devstral-Small-2-24B-Instruct-2512": "No Ollama mapping as of 2026-09-25: the Ollama library's devstral:24b is the 2505 build, a different model.",
    "mistralai/Magistral-Small-2509": "No Ollama mapping as of 2026-09-25: the Ollama library's magistral:24b is the 2506 build, a different model.",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "No successor as of 2026-09-25: the newer Mistral Small 4 card compares against Mistral Small 3 but names no specific replaced model.",
    "utter-project/EuroLLM-9B-Instruct-2512": "No latest as of 2026-09-25: the publisher's EuroLLM collection lists the 2024 and 2512 models side by side, so nothing states this model current.",
    "utter-project/EuroLLM-22B-Instruct-2512": "No latest as of 2026-09-25: the publisher's EuroLLM collection lists the 2024 and 2512 models side by side, naming the line but not which model of it is current.",
    "utter-project/EuroMoE-2.6B-A0.6B-Instruct-2512": "No latest as of 2026-09-25: the publisher's EuroLLM collection lists the 2024 and 2512 models side by side, naming the line but not which model of it is current.",
    "Qwen/Qwen3.5-27B": "No successor as of 2026-09-25: Qwen3.8-27B's card follows the Qwen3.5 and Qwen3.6 series as a whole, naming no specific replaced model.",
    "Qwen/Qwen3.6-27B": "No successor as of 2026-09-25: Qwen3.8-27B's card compares against this model in tables but does not name it as replaced.",
    "Qwen/Qwen3.8-27B": "No row for the other model of the Qwen3.8 release, Qwen3.8-2.4T-A95B, as of 2026-09-25: far beyond one machine.",
    "deepseek-ai/DeepSeek-V3.2": "No successor as of 2026-09-25: the DeepSeek-V4 collection lists several V4 builds side by side, so no catalog row is named current.",
    "meta-llama/Llama-3.3-70B-Instruct": "No successor as of 2026-09-25: Llama 3 cards are gated and no publisher page names a Llama 4 model as replacing this one.",
    "meta-llama/Llama-3.1-8B-Instruct": "No successor as of 2026-09-25: Llama 3 cards are gated and no publisher page names a Llama 4 model as replacing this one.",
    "meta-llama/Llama-3.2-3B-Instruct": "No successor as of 2026-09-25: Llama 3 cards are gated and no publisher page names a Llama 4 model as replacing this one.",
    "zai-org/GLM-4.7-Flash": "No latest as of 2026-09-25: the model's own collection is an older one, not the publisher's current release.",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": "No row for the publisher's other current models (Kimi-K3 and the K2 line) as of 2026-09-25: hundreds of billions of parameters each.",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "No Ollama mapping as of 2026-09-25: the Ollama library's nemotron is a Llama-3.1 model from 2024, not this one.",
    "tiiuae/Falcon-H1-34B-Instruct": "No Ollama mapping as of 2026-09-25: the Ollama library's falcon3 and falcon are earlier lines, not the Falcon-H1 line.",
    "tiiuae/Falcon-H1R-7B": "No Ollama mapping as of 2026-09-25: the Ollama library's falcon3 and falcon are earlier lines, not the Falcon-H1 line.",
    "tiiuae/Falcon-H1-7B-Instruct": "No Ollama mapping as of 2026-09-25: the Ollama library's falcon3 and falcon are earlier lines, not the Falcon-H1 line.",
    "CohereLabs/command-a-reasoning-08-2025": "No Ollama mapping as of 2026-09-25: the Ollama library's command-a:111b is the 03-2025 build, a different model.",
    "HuggingFaceTB/SmolLM3-3B": "No Ollama mapping as of 2026-09-25: the Ollama library carries SmolLM and SmolLM2, not this model.",
    "tencent/Hy-MT2-30B-A3B": "No row for the publisher's larger current model Hy4-preview (780B) as of 2026-09-25: no GGUF build of a listed account.",
    "MiniMaxAI/MiniMax-H3": "No row for the publisher's larger models (MiniMax-M3, 427B) as of 2026-09-25: beyond one machine.",
}


def test_every_reasoned_missing_statement_of_the_schema_1_file_is_a_note():
    catalog = load_catalog()
    # KA-S2 appended the card sentence to the seven `decision` notes; what stands before it is the statement.
    notes = {
        model.hf_repo: model.note.removesuffix(_CARD_READ).strip()
        for family in catalog.families
        for model in family.models
        if model.note
    }
    assert {repo: note for repo, note in notes.items() if note} == NOTES


# `kind` per row, from the Hub: `pipeline_tag` through the table where it is one of the fifteen
# values; otherwise from the tags and the model card (image and text in, text out is `vision`).
# `MiniMaxAI/MiniMax-H3` (`image-text-to-video`) makes video with sound.
VIDEO_REPOS = frozenset({"MiniMaxAI/MiniMax-H3"})
VISION_REPOS = frozenset(
    {
        "mistralai/Ministral-3-14B-Instruct-2512",
        "mistralai/Ministral-3-8B-Instruct-2512",
        "mistralai/Ministral-3-3B-Instruct-2512",
        "mistralai/Mistral-Small-4-119B-2603",
        "mistralai/Devstral-Small-2-24B-Instruct-2512",
        "mistralai/Magistral-Small-2509",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.8-27B",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "google/gemma-4-31B-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-12B-it",
        "google/gemma-4-E4B-it",
        "zai-org/GLM-5.3-Flash",
    }
)


def test_every_shipped_row_carries_the_kind_its_hub_evidence_gives():
    catalog = load_catalog()
    for family in catalog.families:
        for model in family.models:
            if model.hf_repo in VISION_REPOS:
                expected = "vision"
            elif model.hf_repo in VIDEO_REPOS:
                expected = "video-generation"
            elif model.hf_repo in EMBEDDING_REPOS:
                expected = "embedding"
            else:
                expected = "text"
            assert model.kind == expected, model.hf_repo


def test_the_shipped_display_list_matches_the_v1_reference_order_exactly():
    reference = load_catalog(FIXTURES / "catalog_v1_reference.toml")
    shipped = load_catalog()
    assert latest_model_names(shipped) == latest_model_names(reference)


# --- KA-S2: the task Hugging Face gives a model (`hub_pipeline_tag`) and the evidence for `kind` ---
# One broken row breaks one rule; everything else in it is valid (rows of `catalog_v2_valid.json`).


def _load(tmp_path: Path, *rows: dict) -> Catalog:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_document(models=list(rows))), encoding="utf-8")
    return load_catalog(path)


def _evidence_row(**fields) -> dict:
    """Row 0 of the fixture (text, task `text-generation`) with the given fields changed."""
    return _row(0, **fields)


def test_a_vision_row_with_a_matching_task_and_no_kind_source_loads(tmp_path):
    catalog = _load(tmp_path, _evidence_row(hub_pipeline_tag="image-text-to-text", kind="vision", kind_source=None))
    model = catalog.model_for("acme/Nova-7B-2512")
    assert (model.hub_pipeline_tag, model.kind, model.kind_source) == ("image-text-to-text", "vision", None)


def test_a_task_that_contradicts_kind_is_rejected_with_both_kinds_named(tmp_path):
    with pytest.raises(CatalogError, match="acme/Nova-7B-2512") as caught:
        _load(tmp_path, _evidence_row(hub_pipeline_tag="text-generation", kind="vision", kind_source=None))
    assert "vision" in str(caught.value) and "text" in str(caught.value)


def test_a_matching_task_with_a_kind_source_is_rejected(tmp_path):
    """The task already decides `kind`; naming an evidence stage as well is a second, unneeded claim."""
    with pytest.raises(CatalogError, match="kind_source"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag="text-generation", kind="text", kind_source="tags"))


@pytest.mark.parametrize("source", ["tags", "base_model", "config", "decision"])
def test_a_row_without_a_task_loads_with_each_evidence_stage(tmp_path, source):
    fields = {"hub_pipeline_tag": None, "kind": "vision", "kind_source": source}
    if source == "decision":
        fields["note"] = "Kind vision from the model card, checked 2026-10-01."
    assert _load(tmp_path, _evidence_row(**fields)).model_for("acme/Nova-7B-2512").kind_source == source


def test_a_row_without_a_task_and_without_a_kind_source_is_rejected(tmp_path):
    with pytest.raises(CatalogError, match="kind_source"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=None, kind="vision", kind_source=None))


def test_a_decision_without_a_note_is_rejected(tmp_path):
    with pytest.raises(CatalogError, match="note"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=None, kind="vision", kind_source="decision", note=None))


@pytest.mark.parametrize("source", ["pipeline_tag", "", "Tags", "guess"])
def test_a_kind_source_outside_the_four_values_is_rejected(tmp_path, source):
    with pytest.raises(CatalogError, match="kind_source"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=None, kind="vision", kind_source=source))


@pytest.mark.parametrize(("task", "kind"), [("any-to-any", "vision"), ("translation", "text")])
def test_a_task_outside_the_kind_table_is_kept_verbatim_when_a_kind_source_is_named(tmp_path, task, kind):
    catalog = _load(tmp_path, _evidence_row(hub_pipeline_tag=task, kind=kind, kind_source="tags"))
    assert catalog.model_for("acme/Nova-7B-2512").hub_pipeline_tag == task


@pytest.mark.parametrize(("task", "kind"), [("any-to-any", "vision"), ("translation", "text")])
def test_a_task_outside_the_kind_table_without_a_kind_source_is_rejected(tmp_path, task, kind):
    with pytest.raises(CatalogError, match="kind_source"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=task, kind=kind, kind_source=None))


@pytest.mark.parametrize("task", ["Text-Generation", "text generation", "text_generation", "-text", "x" * 200, "a" * 65])
def test_a_task_that_breaks_the_pattern_or_the_length_is_rejected(tmp_path, task):
    with pytest.raises(CatalogError, match="hub_pipeline_tag"):
        _load(tmp_path, _evidence_row(hub_pipeline_tag=task, kind="text", kind_source="tags"))


def test_a_task_of_sixty_four_characters_loads(tmp_path):
    task = "a" * 64
    catalog = _load(tmp_path, _evidence_row(hub_pipeline_tag=task, kind="text", kind_source="tags"))
    assert catalog.model_for("acme/Nova-7B-2512").hub_pipeline_tag == task


# The fifteen table values, repeated here on purpose: the shipped rows are checked against this
# copy, never against the constant the schema itself uses.
TABLE_KIND_OF_TASK = {
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
    "text-to-video": "video-generation",
    "image-to-video": "video-generation",
    "image-text-to-video": "video-generation",
    "video-to-video": "video-generation",
}
# The rows of the shipped catalog whose task is not in the table, as the one filling run measured
# them (2026-10-03): task, evidence stage and note. `decision` marks the seven Mistral repos whose
# model card was read on 2026-10-01; the note keeps any older sentence in front of it.
_CARD_READ = "Kind vision from the model card, checked 2026-10-01."
OFF_TABLE_ROWS: dict[str, tuple[str | None, str, str | None]] = {
    "mistralai/Ministral-3-14B-Instruct-2512": (None, "decision", _CARD_READ),
    "mistralai/Ministral-3-8B-Instruct-2512": (None, "decision", _CARD_READ),
    "mistralai/Ministral-3-3B-Instruct-2512": (None, "decision", _CARD_READ),
    "mistralai/Mistral-Small-4-119B-2603": (None, "decision", _CARD_READ),
    "mistralai/Devstral-Small-2-24B-Instruct-2512": (
        None,
        "decision",
        "No Ollama mapping as of 2026-09-25: the Ollama library's devstral:24b is the 2505 build, a different model. "
        + _CARD_READ,
    ),
    "mistralai/Magistral-Small-2509": (
        None,
        "decision",
        "No Ollama mapping as of 2026-09-25: the Ollama library's magistral:24b is the 2506 build, a different model. "
        + _CARD_READ,
    ),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": (
        None,
        "decision",
        "No successor as of 2026-09-25: the newer Mistral Small 4 card compares against Mistral Small 3 but names no "
        "specific replaced model. " + _CARD_READ,
    ),
    "google/gemma-4-12B-it": ("any-to-any", "tags", None),
    "google/gemma-4-E4B-it": ("any-to-any", "tags", None),
    "tencent/Hy-MT2-30B-A3B": (
        "translation",
        "tags",
        "No row for the publisher's larger current model Hy4-preview (780B) as of 2026-09-25: no GGUF build of a "
        "listed account.",
    ),
}
DECISION_ONLY_REPOS = frozenset(
    {
        "mistralai/Ministral-3-14B-Instruct-2512",
        "mistralai/Ministral-3-8B-Instruct-2512",
        "mistralai/Ministral-3-3B-Instruct-2512",
        "mistralai/Mistral-Small-4-119B-2603",
        "mistralai/Devstral-Small-2-24B-Instruct-2512",
        "mistralai/Magistral-Small-2509",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    }
)


def test_the_shipped_catalog_has_a_task_or_an_evidence_stage_on_every_one_of_its_sixty_two_rows():
    rows = [model for family in load_catalog().families for model in family.models]
    assert len(rows) == 62
    for model in rows:
        if model.hf_repo in OFF_TABLE_ROWS:
            task, source, note = OFF_TABLE_ROWS[model.hf_repo]
            assert (model.hub_pipeline_tag, model.kind_source, model.note) == (task, source, note), model.hf_repo
        else:
            assert model.hub_pipeline_tag in TABLE_KIND_OF_TASK, model.hf_repo
            assert model.kind == TABLE_KIND_OF_TASK[model.hub_pipeline_tag], model.hf_repo
            assert model.kind_source is None, model.hf_repo


def test_only_the_seven_listed_repos_of_the_shipped_catalog_carry_a_decision():
    rows = [model for family in load_catalog().families for model in family.models]
    decided = {model.hf_repo for model in rows if model.kind_source == "decision"}
    assert decided == DECISION_ONLY_REPOS
    assert len(OFF_TABLE_ROWS) == 10


def test_a_schema_1_row_reads_with_no_task_and_no_kind_source_and_no_rule_applies():
    rows = [model for family in load_catalog(FIXTURES / "catalog_excerpt.toml").families for model in family.models]
    assert len(rows) == 3
    for model in rows:
        assert (model.hub_pipeline_tag, model.kind_source, model.kind) == (None, None, "text"), model.hf_repo
