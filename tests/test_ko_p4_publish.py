"""P4, part 4: the publication of the catalog on its channels (AP9-KO, V3, "Published catalog"): the working document
may be schema 2, 3 or 4, the old channel `catalog-v2.json` stays what release 1.0.0 reads and is made only through
`project_to_schema2`, the channel `catalog-v3.json` is the schema-3 document (release 1.1.0), `catalog-v4.json` the
schema-4 one (release 1.2.0; `tests/test_vs_switches.py`), and the dataset card says so. The
upload and the verification are exercised with the doubles of `test_catalog_publish.py`; nothing is uploaded."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

import fixture_ko_p4  # noqa: F401  (puts scripts/ on the path, skips outside a clone)
from test_catalog_publish import (  # noqa: F401  (`work` is a fixture)
    APPLY, CATALOG_BYTES, COMMIT, FIX, P, PUBLISHED, R, TODAY, upload_env, work,
)

from modelroom.catalog_remote import CHANNEL_V2, CHANNEL_V3, CHANNEL_V4
from modelroom.catalog_schema import build_catalog_v2, build_catalog_v3
from modelroom.http import FixtureTransport, Response

card_tool = importlib.import_module("catalog_card")
publish = importlib.import_module("catalog_publish")


def finding() -> dict:
    return {
        "repo": "other/Nova-7B-GGUF", "revision": "a" * 40, "identity": "Nova-7B-Q8_0.gguf", "file": "Nova-7B-Q8_0.gguf", "quantization": "Q8_0",
        "architecture": "nova7", "architecture_source": "file_header", "result": "absent", "llamacpp_commit": "c" * 40,
        "source_url": f"https://github.com/ggml-org/llama.cpp/blob/{'c' * 40}/src/llama-arch.cpp", "checked": "2026-10-13", "same_quantization": 1,
    }


def v3_with_finding() -> tuple[dict, str]:
    document = json.loads(CATALOG_BYTES)
    document["schema_version"] = 3
    repo = document["models"][0]["hf_repo"]
    document["models"][0]["gguf_checks"] = [{**finding(), "repo": f"{document['packagers'][0]}/Nova-7B-GGUF"}]
    return document, repo


# --- the two files of the channels ------------------------------------------------------------------------------------------


def test_the_channels_are_three_files_of_the_dataset():
    assert (CHANNEL_V2.file_name, CHANNEL_V3.file_name, CHANNEL_V4.file_name) == ("catalog-v2.json", "catalog-v3.json", "catalog-v4.json")
    assert tuple(P.channel_files(CATALOG_BYTES)) == (CHANNEL_V4.file_name, CHANNEL_V3.file_name, CHANNEL_V2.file_name)


def test_a_working_catalog_of_schema_2_is_the_old_channel_byte_for_byte_and_the_new_channel_the_same_rows_as_schema_3():
    files = P.channel_files(CATALOG_BYTES)
    assert files[CHANNEL_V2.file_name] == CATALOG_BYTES
    v3 = json.loads(files[CHANNEL_V3.file_name])
    assert v3["schema_version"] == 3 and {k: v for k, v in v3.items() if k != "schema_version"} == {k: v for k, v in json.loads(CATALOG_BYTES).items() if k != "schema_version"}
    assert build_catalog_v3(files[CHANNEL_V3.file_name].decode(), Path("v3.json")).families
    assert files[CHANNEL_V3.file_name] == R.dump_json(v3)


def test_a_working_catalog_of_schema_3_is_the_new_channel_byte_for_byte_and_the_old_one_its_projection_without_findings():
    document, repo = v3_with_finding()
    data = R.dump_json(document)
    files = P.channel_files(data)
    assert files[CHANNEL_V3.file_name] == data
    old = json.loads(files[CHANNEL_V2.file_name])
    assert old["schema_version"] == 2
    assert all("gguf_checks" not in row and "setup_note" not in row for row in old["models"])
    assert [row["hf_repo"] for row in old["models"]] == [row["hf_repo"] for row in document["models"]]
    build_catalog_v2(files[CHANNEL_V2.file_name].decode(), Path("v2.json"))  # exactly what release 1.0.0 reads
    assert build_catalog_v3(data.decode(), Path("v3.json"))
    assert repo in files[CHANNEL_V2.file_name].decode()


def test_the_old_channel_is_made_through_the_projection_and_nowhere_else(monkeypatch):
    from modelroom import catalog_schema

    called = []
    real = catalog_schema.project_to_schema2
    monkeypatch.setattr(P, "project_to_schema2", lambda document: called.append(document["schema_version"]) or real(document))
    document, _ = v3_with_finding()
    P.channel_files(R.dump_json(document))
    assert called == [3]


def test_a_catalog_that_is_no_catalog_makes_no_channel_files():
    for data in (b"not json", b"[]", json.dumps({"schema_version": 5}).encode(), json.dumps({"schema_version": 4}).encode(), json.dumps({"schema_version": 1}).encode()):
        with pytest.raises(P.PublishError):
            P.channel_files(data)


def test_the_folder_upload_sends_both_files_to_the_root_in_one_commit():
    assert P.upload_folder_args("jm-vis/x", Path("a"), "c" * 40) == [
        "uvx", "--from", f"huggingface_hub=={P.HF_HUB_VERSION}", "hf", "upload", "jm-vis/x", str(Path("a")), ".",
        "--repo-type", "dataset", "--commit-message", "c" * 40]


# --- the table: a working document of schema 3 is taken, validated and kept as schema 3 ---------------------------------------


def test_a_finding_of_a_proposal_is_taken_into_a_working_catalog_of_schema_3_and_the_table_stays_schema_3():
    document, repo = v3_with_finding()
    clean = copy.deepcopy(document)
    clean["models"][0]["gguf_checks"] = []
    change = {"change": "field", "field": "gguf_checks", "stage": "A9", "hf_repo": repo, "kind": "text", "open_since": TODAY, "evidence": ["https://github.com/x"],
              "fields": {"gguf_checks": {"old": [], "new": document["models"][0]["gguf_checks"]}}, "n": 1}
    proposal = {"changes": [change], "candidates": []}
    request = P.parse_request("1", None, None)
    done = P.plan_apply(clean, R.dump_json(clean), b"[]", proposal, request, TODAY)
    result = json.loads(done.catalog_text)
    assert result["schema_version"] == 3 and result["models"][0]["gguf_checks"][0]["result"] == "absent" and result["published"] == TODAY


def test_a_finding_cannot_be_taken_into_a_working_catalog_that_is_still_schema_2():
    document = json.loads(CATALOG_BYTES)
    repo = document["models"][0]["hf_repo"]
    change = {"change": "field", "field": "gguf_checks", "stage": "A9", "hf_repo": repo, "kind": "text", "open_since": TODAY, "evidence": [],
              "fields": {"gguf_checks": {"old": [], "new": [{**finding(), "repo": f"{document['packagers'][0]}/Nova-7B-GGUF"}]}}, "n": 1}
    with pytest.raises(P.PublishError):
        P.plan_apply(document, CATALOG_BYTES, b"[]", {"changes": [change], "candidates": []}, P.parse_request("1", None, None), TODAY)


# --- the publication: both files in one upload, both verified -------------------------------------------------------------------


def answers(files: dict[str, bytes], dataset: str = publish.DATASET) -> FixtureTransport:
    revision = f"https://huggingface.co/api/datasets/{dataset}/revision/main"
    responses = {("GET", revision): Response(200, {}, json.dumps({"sha": PUBLISHED}).encode())}
    for name, body in files.items():
        responses[("GET", f"https://huggingface.co/datasets/{dataset}/resolve/{PUBLISHED}/{name}")] = Response(200, {"X-Repo-Commit": PUBLISHED}, body)
    return FixtureTransport(responses)


def test_publish_pending_uploads_the_three_channel_files_in_one_process_and_verifies_all(work, tmp_path):
    work.committed()
    env = upload_env(tmp_path)
    data = work.decision()["result_catalog"].encode()
    files = P.channel_files(data)
    work.transport = answers(files)
    result = work.call("--publish-pending", work.folder, env=env)
    assert result.code == 0 and work.state() == "published"
    (args, _), = work.runner.calls
    assert args[:5] == ["uvx", "--from", f"huggingface_hub=={P.HF_HUB_VERSION}", "hf", "upload"] and args[5] == "jm-vis/modelroom-catalog" and args[7] == "."
    assert work.runner.folders == [files]
    urls = [url for _, url in work.transport.calls]
    assert urls[0].endswith("/revision/main") and {u.rsplit("/", 1)[-1] for u in urls[1:]} == {"catalog-v2.json", "catalog-v3.json", "catalog-v4.json"}


@pytest.mark.parametrize("name", ["catalog-v2.json", "catalog-v3.json", "catalog-v4.json"])
def test_the_check_after_the_upload_fails_when_either_file_differs_and_names_it(work, tmp_path, name):
    work.committed()
    files = P.channel_files(work.decision()["result_catalog"].encode())
    work.transport = answers({**files, name: files[name] + b" "})
    result = work.call("--publish-pending", work.folder, env=upload_env(tmp_path))
    assert result.code == 1 and work.state() == "uploaded" and f"ALARM: the published {name} differs" in result.out


def test_verify_reads_both_files_of_the_dataset(work):
    files = P.channel_files(CATALOG_BYTES)
    work.transport = answers(files)
    result = work.call("--verify", env={})
    assert (result.code, result.out) == (0, "catalog: published files equal working tree\n")


def test_verify_says_which_file_could_not_be_read(work):
    files = P.channel_files(CATALOG_BYTES)
    work.transport = answers({"catalog-v2.json": files["catalog-v2.json"], "catalog-v3.json": b"", "catalog-v4.json": files["catalog-v4.json"]})
    work.transport._responses[("GET", f"https://huggingface.co/datasets/jm-vis/modelroom-catalog/resolve/{PUBLISHED}/catalog-v3.json")] = Response(404, {}, b"")
    result = work.call("--verify", env={})
    assert result.code == 1 and "catalog-v3.json" in result.out and "could not be read" in result.out


def test_the_hygiene_check_reads_both_files(work, tmp_path, monkeypatch):
    work.committed()
    seen = []
    real = publish.Tool._hygiene

    def spy(self, data, name=None):
        seen.append(name)
        return real(self, data) if name is None else real(self, data, name)

    monkeypatch.setattr(publish.Tool, "_hygiene", spy)
    work.transport = answers(P.channel_files(work.decision()["result_catalog"].encode()))
    work.call("--publish-pending", work.folder, env=upload_env(tmp_path))
    assert {"catalog-v2.json", "catalog-v3.json", "catalog-v4.json"} <= set(seen)


# --- the dataset card ----------------------------------------------------------------------------------------------------------


def test_the_card_names_both_channels_and_the_two_findings_and_still_follows_the_numbers():
    text = (Path(__file__).resolve().parent.parent / "docs" / "dataset-card.md").read_text(encoding="utf-8")
    for needle in ("catalog-v2.json", "catalog-v3.json", "catalog-v4.json", "gguf_checks", "setup_note", "vllm_args", "schema 3", "schema 4", "llama.cpp", "release 1.0.0"):
        assert needle in text, needle
    assert "data_files: catalog-v2.json" not in text
    numbers = card_tool.numbers(CATALOG_BYTES)
    again = card_tool.card_text(text, numbers)
    assert f"{numbers.models} models" in again and "model card" in text
    assert "no promise that a runtime loads the file" in text and "never about a runtime version" in text
