"""Contract tests of AP9-KO P0, part 4: the producers of the catalog scanner, repaired for the new row
fields (V3, "Erzeuger-Reparatur"). The scanner itself is P4's; only `ROW_FIELDS`, `build_row`,
`new_row_change` and what takes a row into a document are P0's.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.catalog import CatalogModel

from test_catalog import FIXTURES, _document

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the scanner is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))
rules = importlib.import_module("catalog_scan_rules")

TODAY = "2026-10-13"


def _entry(repo: str = "acme/Nova-9B-2610"):
    created = datetime(2026, 10, 10, tzinfo=timezone.utc)
    return rules.Entry(
        repo=repo, created=created, likes=10, downloads=100, tags=("text-generation",), card={"license": "apache-2.0"},
        pipeline="text-generation", parameters_b=9.0,
    )


def _result():
    return rules.KindResult("text", None, False, {"stage": "pipeline_tag", "value": "text-generation"})


def _row() -> dict:
    return rules.build_row(_entry(), _result(), family="nova", today=TODAY, latest=False, latest_source=None)


def test_the_fields_of_a_row_are_the_fields_of_the_model_and_include_the_new_ones():
    assert rules.ROW_FIELDS == tuple(sorted(CatalogModel.model_fields))
    assert {"gguf_checks", "setup_note"} <= set(rules.ROW_FIELDS)


def test_a_new_row_has_every_field_and_the_new_ones_at_their_default():
    row = _row()
    assert set(row) == set(rules.ROW_FIELDS)
    assert (row["gguf_checks"], row["setup_note"]) == ([], None)


def test_a_new_row_change_names_every_field_and_breaks_on_none():
    change = rules.new_row_change(_row(), _result(), TODAY)
    assert set(change["fields"]) == set(rules.ROW_FIELDS)
    assert change["fields"]["gguf_checks"] == {"old": None, "new": []}
    assert change["fields"]["setup_note"] == {"old": None, "new": None}


def test_a_new_row_taken_into_a_schema_2_document_leaves_the_new_fields_out_so_the_old_reader_still_reads_it():
    document = _document()
    change = rules.new_row_change(_row(), _result(), TODAY)
    applied = rules.apply_changes(document, [change])
    added = applied["models"][-1]
    assert added["hf_repo"] == "acme/Nova-9B-2610"
    assert "gguf_checks" not in added and "setup_note" not in added
    assert document["schema_version"] == 2 and applied["schema_version"] == 2
    kept, notes = rules.check_changes(document, [change])
    assert kept == [change] and notes == []


def test_a_new_row_taken_into_a_schema_3_document_keeps_them():
    document = copy.deepcopy(_document())
    document["schema_version"] = 3
    applied = rules.apply_changes(document, [rules.new_row_change(_row(), _result(), TODAY)])
    added = applied["models"][-1]
    assert (added["gguf_checks"], added["setup_note"]) == ([], None)
    kept, _notes = rules.check_changes(document, [rules.new_row_change(_row(), _result(), TODAY)])
    assert len(kept) == 1


def _finding_change(row_repo: str = "acme/Nova-7B-2512") -> dict:
    check = {
        "repo": "othercorp/Nova-7B-GGUF", "revision": "a" * 40, "identity": "Nova-7B-Q8_0.gguf", "file": "Nova-7B-Q8_0.gguf",
        "quantization": "Q8_0", "architecture": "nova7", "architecture_source": "file_header", "result": "absent",
        "llamacpp_commit": "c" * 40, "source_url": f"https://github.com/ggml-org/llama.cpp/blob/{'c' * 40}/src/llama-arch.cpp",
        "checked": TODAY,
    }
    return {
        "change": "field", "field": "gguf_checks", "stage": "A9", "hf_repo": row_repo, "kind": "text", "open_since": TODAY,
        "evidence": [check["source_url"]], "fields": {"gguf_checks": {"old": [], "new": [check]}},
    }


def test_a_finding_is_left_out_of_a_schema_2_document_with_the_reason_and_taken_into_a_schema_3_one():
    v2 = _document()
    kept, notes = rules.check_changes(v2, [_finding_change()])
    assert kept == [] and len(notes) == 1 and "left out" in notes[0] and "schema 2" in notes[0]
    v3 = _document()
    v3["schema_version"] = 3
    kept, notes = rules.check_changes(v3, [_finding_change()])
    assert len(kept) == 1 and notes == []


def test_a_finding_taken_into_a_schema_3_document_is_the_value_of_its_field():
    v3 = _document()
    v3["schema_version"] = 3
    applied = rules.apply_changes(v3, [_finding_change()])
    row = next(model for model in applied["models"] if model["hf_repo"] == "acme/Nova-7B-2512")
    assert row["gguf_checks"][0]["result"] == "absent"


def test_the_fixture_the_other_scanner_tests_read_is_still_a_schema_2_catalog():
    assert json.loads((FIXTURES / "catalog_v2_valid.json").read_text(encoding="utf-8"))["schema_version"] == 2
