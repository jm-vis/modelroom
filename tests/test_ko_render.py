"""Contract tests of AP9-KO P0, part 5: how the evidence of the catalog reaches the render document (V11), the
guided run and the stand-alone `render`; and that, with the defaults, nothing of release 1.0.0 changes.

A row of the document carries its package's load check, a document carries the setup note of each
model that has rows; both are left out of the JSON where they hold their default, so a document without
evidence is byte for byte the document of 1.0.0.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.catalog import CATALOG_FILE, Catalog, load_catalog
from modelroom.catalog_findings import llamacpp_page_url
from modelroom.catalog_schema import build_catalog_v3
from modelroom.document import RenderDocument
from modelroom.measurements import default_scenario
from modelroom.quantization import identity_stem, package_identity_key
from modelroom.render import build_render_document, document_json

from test_render import GIB, RENDERED_AT, _config, _hf_package, _ready, _snapshot

COMMIT = "45451c366d4256f94fbc8845645e31a6b9155054"
DAY = date(2026, 10, 13)


def _check_for(package, **changes) -> dict:
    file = package.files[0].name
    data = {
        "repo": package.repo, "revision": package.revision, "identity": identity_stem(file), "file": file,
        "quantization": package.quantization, "architecture": "nova8", "architecture_source": "file_header",
        "result": "absent", "llamacpp_commit": COMMIT, "source_url": llamacpp_page_url(COMMIT), "checked": DAY.isoformat(),
    }
    data.update(changes)
    return data


def _catalog(checks: list[dict], note: dict | None = None) -> Catalog:
    row = {
        "hf_repo": "acme/Nova-8B", "latest": False, "latest_source": None, "latest_checked": None, "successor": None,
        "successor_source": None, "successor_checked": None, "ollama_base": None, "ollama_tag": None,
        "source": "https://huggingface.co/acme/Nova-8B", "kind": "text", "hub_pipeline_tag": "text-generation",
        "kind_source": None, "family": "nova", "publisher": "acme", "added": "2026-09-23", "parameters_b": 8.0,
        "license": None, "hub_created": None, "gguf_repos": ["packager/Nova-8B-GGUF"], "note": None,
        "gguf_checks": checks, "setup_note": note,
    }
    document = {
        "schema_version": 3, "published": "2026-10-13", "data_license": "CC-BY-4.0",
        "attribution": "modelroom catalog, jm-vis, CC BY 4.0, https://huggingface.co/datasets/jm-vis/modelroom-catalog",
        "publishers": [{"account": "acme", "name": "Acme Labs", "region": None}], "packagers": ["packager"], "models": [row],
    }
    return build_catalog_v3(json.dumps(document), Path("catalog.json"))


def _note() -> dict:
    revision = "d" * 40
    return {
        "model": "acme/Nova-8B", "status": "instructions", "card_revision": revision, "checked": DAY.isoformat(),
        "source_url": f"https://huggingface.co/acme/Nova-8B/blob/{revision}/README.md", "package": "nova-inference", "reason": None,
    }


def _build(packages, catalog=None) -> RenderDocument:
    return build_render_document(
        _config(), _snapshot(packages), {"workstation": _ready()}, {}, default_scenario(), RENDERED_AT, None, catalog=catalog
    )


# --- the default: nothing changes -----------------------------------------------------------------------------------


def test_without_a_catalog_and_with_a_catalog_that_knows_nothing_the_json_view_is_the_same_bytes():
    def dump(packages, catalog=None):
        return json.dumps(document_json(_build(packages, catalog)), indent=2, sort_keys=True)

    # a packager the shipped catalog lists, and one the small catalog below lists: both are recommendable accounts
    shipped_listed = [_hf_package(repo="unsloth/Nova-8B-GGUF"), _hf_package(repo="unsloth/Nova-8B-GGUF", quantization="Q8_0", weights_bytes=8 * GIB)]
    own_listed = [_hf_package(), _hf_package(quantization="Q8_0", weights_bytes=8 * GIB)]
    assert dump(shipped_listed, load_catalog()) == dump(shipped_listed)
    assert dump(own_listed, _catalog([])) == dump(own_listed)
    plain = dump(own_listed)
    assert "load_check" not in plain and "setup_notes" not in plain and '"origin": "other"' not in plain


def test_a_row_without_evidence_has_no_load_check_and_a_document_no_notes():
    document = _build([_hf_package()])
    entry = document.machines[0].ranked[0]
    assert entry.load_check is None and document.setup_notes == {}


# --- the evidence arrives ----------------------------------------------------------------------------------------------


def test_the_check_of_a_package_reaches_its_row_and_only_its_row():
    q4, q8 = _hf_package(), _hf_package(quantization="Q8_0", weights_bytes=8 * GIB)
    document = _build([q4, q8], _catalog([_check_for(q4)]))
    rows = {entry.quantization: entry for entry in document.machines[0].ranked}
    assert rows["Q4_K_M"].load_check is not None and rows["Q4_K_M"].load_check.result == "absent"
    assert rows["Q8_0"].load_check is None


def test_the_check_reaches_a_row_that_is_set_aside_too():
    huge = _hf_package(weights_bytes=900 * GIB)
    document = _build([huge], _catalog([_check_for(huge)]))
    block = document.machines[0]
    aside = [*block.not_covered, *block.no_fit]
    assert aside and all(entry.load_check is not None for entry in aside)


def test_a_check_of_another_revision_does_not_reach_the_row():
    package = _hf_package()
    stale = _check_for(package, revision="b" * 40)
    assert _build([package], _catalog([stale])).machines[0].ranked[0].load_check is None


def test_the_json_view_carries_the_check_and_the_note_and_reads_back():
    package = _hf_package()
    document = _build([package], _catalog([_check_for(package)], _note()))
    payload = document_json(document)
    assert payload["machines"][0]["ranked"][0]["load_check"]["architecture"] == "nova8"
    assert payload["setup_notes"]["acme/Nova-8B"]["package"] == "nova-inference"
    assert payload["schema_version"] == 3
    assert RenderDocument.model_validate(payload) == document


def test_the_note_belongs_to_a_model_that_has_rows():
    assert _build([], _catalog([], _note())).setup_notes == {}
    assert set(_build([_hf_package()], _catalog([], _note())).setup_notes) == {"acme/Nova-8B"}


def test_the_note_that_found_no_instructions_is_not_carried():
    none_found = {**_note(), "status": "none_recognized", "package": None}
    assert _build([_hf_package()], _catalog([], none_found)).setup_notes == {}


def test_a_document_keeps_its_evidence_through_the_copy_the_guided_mode_makes_of_it():
    from modelroom.render_cmd import with_requests_origin

    package = _hf_package()
    document = _build([package], _catalog([_check_for(package)], _note()))
    copied = with_requests_origin(document, _config())
    assert (copied.machines, copied.setup_notes) == (document.machines, document.setup_notes)
    assert copied.machines[0].ranked[0].load_check is not None


def test_the_note_reaches_a_row_whose_base_model_id_the_catalog_writes_in_another_case():
    """Hugging Face ids are case-insensitive: the catalog's `acme/NOVA-8b` is the snapshot's `acme/Nova-8B`. The
    document keys the note by the id of its rows and keeps the note as the catalog wrote it. (The account stays
    `acme`: the publisher rule of release 1.0.0 compares the account as written and is not P0's.)"""
    revision = "d" * 40
    note = {**_note(), "model": "acme/NOVA-8b", "source_url": f"https://huggingface.co/acme/NOVA-8b/blob/{revision}/README.md"}
    row = _catalog([]).families[0].models[0]
    shouting = build_catalog_v3(
        json.dumps(
            {
                "schema_version": 3, "published": "2026-10-13", "data_license": "CC-BY-4.0",
                "attribution": "modelroom catalog, jm-vis, CC BY 4.0, https://huggingface.co/datasets/jm-vis/modelroom-catalog",
                "publishers": [{"account": "acme", "name": "Acme Labs", "region": None}], "packagers": ["packager"],
                "models": [{**row.model_dump(mode="json"), "hf_repo": "acme/NOVA-8b", "source": "https://huggingface.co/acme/NOVA-8b", "setup_note": note}],
            }
        ),
        Path("catalog.json"),
    )
    document = _build([_hf_package()], shouting)
    assert {repo: found.model for repo, found in document.setup_notes.items()} == {"acme/Nova-8B": "acme/NOVA-8b"}
    assert RenderDocument.model_validate(document_json(document)) == document


def test_a_document_does_not_take_a_note_for_another_model_than_its_key():
    document = _build([_hf_package()], _catalog([], _note()))
    payload = document_json(document)
    payload["setup_notes"] = {"acme/Other": payload["setup_notes"]["acme/Nova-8B"]}
    with pytest.raises(ValidationError, match="setup_notes"):
        RenderDocument.model_validate(payload)


def test_the_load_check_of_a_row_is_a_package_check_of_that_package():
    package = _hf_package()
    entry = _build([package], _catalog([_check_for(package)])).machines[0].ranked[0]
    assert (entry.load_check.repo, entry.load_check.identity) == (package.repo, package_identity_key(package)[2])


def _catalog_listing(packagers: list[str]) -> Catalog:
    base = _catalog([])
    return base.model_copy(update={"packagers": packagers})


def test_the_origin_of_a_package_reaches_its_row_so_two_catalogs_give_two_documents():
    """V4: `packager/...` is a listed packager in one catalog and an other account in the next; the document says it."""
    package = _hf_package()
    listed = _build([package], _catalog_listing(["packager"]))
    unlisted = _build([package], _catalog_listing(["somebody-else"]))
    assert listed.machines[0].ranked[0].origin is None  # listed packager: recommendable, the default, not written
    assert unlisted.machines[0].ranked[0].origin == "other"
    assert '"origin": "other"' in json.dumps(document_json(unlisted)) and '"origin": "other"' not in json.dumps(document_json(listed))


def test_the_publisher_of_the_base_model_is_never_other_and_an_ollama_package_has_no_origin():
    mine = _hf_package(repo="acme/Nova-8B-GGUF")
    assert _build([mine], _catalog_listing([])).machines[0].ranked[0].origin is None


def test_without_a_catalog_no_row_has_an_origin():
    assert _build([_hf_package()]).machines[0].ranked[0].origin is None


# --- the stand-alone render and the guided run ------------------------------------------------------------------------


def _first_gguf_row(payload: dict) -> tuple[int, dict]:
    """`(index, entry)` of the first ranked row of the first machine that is a Hugging Face GGUF package."""
    for index, entry in enumerate(payload["machines"][0]["ranked"]):
        if entry["package_identity"][0] == "huggingface" and entry["format"] == "gguf":
            return index, entry
    raise AssertionError("the fixture shows no Hugging Face GGUF row")


def _catalog_checking(config, entry: dict) -> Catalog:
    """The shipped catalog as a schema 3 document that holds a check for the package of `entry`."""
    from modelroom.state import read_snapshot

    package = next(p for p in read_snapshot(config).packages if list(package_identity_key(p)) == entry["package_identity"])
    shipped = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    shipped["schema_version"] = 3
    row = next(m for m in shipped["models"] if m["hf_repo"] == package.base_model_hf_repo)
    row["gguf_checks"] = [_check_for(package)]
    return build_catalog_v3(json.dumps(shipped), Path("catalog.json"))


def test_the_stand_alone_render_hands_the_catalog_to_the_document(tmp_path: Path):
    from fixture_support import place_v2_profile, qwen35_example_config_dict, with_profile
    from modelroom.cli import fetch_with_config, main
    from modelroom.config import Configuration
    from test_cli import RUN1, RUN2, _to_toml, _transport

    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    config = Configuration.from_dict(data)
    assert fetch_with_config(config, "workstation", transport=_transport(), now=RUN1) == 0
    profile = place_v2_profile(config.paths.hardware_dir)
    config = with_profile(config, "workstation", profile.profile_id)
    config_path = tmp_path / "modelroom.toml"
    config_path.write_text(_to_toml(data).replace("writer = true", f'writer = true\nprofile = "{profile.profile_id}"'), encoding="utf-8")
    pointer = tmp_path / "pointer" / "guided.json"

    assert main(["render", "--config", str(config_path)], now=RUN2, pointer_path=pointer) == 0
    plain_bytes = (tmp_path / "models.json").read_bytes()
    index, entry = _first_gguf_row(json.loads(plain_bytes))
    catalog = _catalog_checking(config, entry)

    assert main(["render", "--config", str(config_path)], now=RUN2, pointer_path=pointer, catalog=catalog) == 0
    ranked = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))["machines"][0]["ranked"]
    assert ranked[index]["load_check"]["result"] == "absent"
    assert [row.get("load_check") for number, row in enumerate(ranked) if number != index] == [None] * (len(ranked) - 1)

    # with a catalog that knows nothing of the package the bytes are the bytes of 1.0.0
    assert main(["render", "--config", str(config_path)], now=RUN2, pointer_path=pointer, catalog=load_catalog()) == 0
    assert (tmp_path / "models.json").read_bytes() == plain_bytes


def test_the_guided_run_hands_its_catalog_to_the_render_step(tmp_path: Path):
    from modelroom.config import load_config
    from test_guided import _config_file, _results, _run

    assert _run(tmp_path / "first")[0] == 0
    first = json.loads((_results(tmp_path / "first") / "docs" / "models.json").read_text(encoding="utf-8"))
    index, entry = _first_gguf_row(first)
    catalog = _catalog_checking(load_config(_config_file(tmp_path / "first")), entry)

    assert _run(tmp_path / "second", catalog=catalog)[0] == 0
    second = json.loads((_results(tmp_path / "second") / "docs" / "models.json").read_text(encoding="utf-8"))
    assert second["machines"][0]["ranked"][index]["load_check"]["result"] == "absent"
    # and the run without evidence wrote no key of the new kind at all
    assert "load_check" not in json.dumps(first) and "setup_notes" not in json.dumps(first)
