"""Package VS (release 1.2.0): the vLLM switches of a model card (CONTRACTS.md, V7 "Setup notes of model cards",
"Published catalog", "The forms and their versions").

A `SetupNote` carries `vllm_args` since schema 4, the switches of the card's `vllm serve` block for its model, from a
closed list; the catalog has a third channel (`catalog-v4.json`) and the second one (`catalog-v3.json`) becomes a
projection without the new field, as the first one (`catalog-v2.json`) already is for release 1.0.0. The command
builder puts the switches of the note of a package's base model into both vLLM lines. The rule that reads a card is
tested in `test_ko_p4_card.py`.

The readers of releases 1.0.0 and 1.1.0 are their own code, run from the repository's history in a process of their own.
"""

from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import timedelta
from pathlib import Path

import pytest

import fixture_ko_p4  # noqa: F401  (puts scripts/ on the path, skips outside a clone)
import catalog_publish_rules as P

from modelroom import catalog_remote
from modelroom.catalog import CATALOG_V3_MAX_BYTES, load_catalog
from modelroom.catalog_findings import VLLM_FLAG_SWITCHES, VLLM_VALUE_SWITCHES, SetupNote
from modelroom.catalog_schema import (
    CatalogDocumentV2, CatalogDocumentV3, CatalogDocumentV4, build_catalog, build_catalog_v3, build_catalog_v4, project_to_schema2,
    project_to_schema3,
)
from modelroom.contracts import SchemaVersionError
from modelroom.http import FixtureTransport, RequestBudget, Response
from modelroom.install_vllm import card_switches, vllm_install

from test_catalog import FIXTURES, _document, _row
from test_ko_catalog_v3 import NOW, REVISION_URL, SHA, _channel_url, _hub, _note
from test_install_vllm import DOCKER, REPO, SERVE, _both, _fit, _gguf_package, _package, _profile
from test_ko_p1_setup_note import BASE_REPO, REVISION, _vllm_text, _with_note

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RELEASE_100 = "2d322fb"
_RELEASE_110 = "3553050"

SWITCHES = ["--reasoning-parser qwen3", "--tool-call-parser hermes", "--enable-auto-tool-choice", "--kv-cache-dtype fp8"]
THREE = SWITCHES[:3]


@pytest.fixture(autouse=True)
def _published_catalog_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELROOM_CATALOG", raising=False)


def _v4(models: list[dict] | None = None, **overrides) -> dict:
    document = _document(models, **overrides)
    document["schema_version"] = 4
    return document


def _v4_with_switches(args: list[str] | None = None) -> dict:
    note = _note(vllm_args=SWITCHES if args is None else args)
    return _v4([_row(0, gguf_checks=[], setup_note=note), _row(1), _row(2)])


def _later(document: dict, days: int = 3) -> dict:
    document["published"] = (load_catalog().published + timedelta(days=days)).isoformat()
    return document


# --- the field of a note ------------------------------------------------------------------------------------------------


def test_a_note_without_switches_is_written_without_the_key_and_with_them_in_the_given_order():
    assert "vllm_args" not in SetupNote.model_validate(_note()).model_dump(mode="json")
    assert "vllm_args" not in SetupNote.model_validate(_note(vllm_args=[])).model_dump(mode="json")
    assert SetupNote.model_validate(_note(vllm_args=SWITCHES)).model_dump(mode="json")["vllm_args"] == SWITCHES


def test_the_list_of_switches_is_the_closed_list_of_the_decision():
    assert VLLM_VALUE_SWITCHES == ("--reasoning-parser", "--tool-call-parser", "--kv-cache-dtype")
    assert VLLM_FLAG_SWITCHES == ("--enable-auto-tool-choice",)


@pytest.mark.parametrize("args", [
    ["--trust-remote-code"],
    ["--max-model-len 4096"],
    ["--tensor-parallel-size 2"],
    ["--reasoning-parser"],
    ["--reasoning-parser "],
    ["--reasoning-parser  qwen3"],
    ["--reasoning-parser=qwen3"],
    ["--reasoning-parser qwen3 extra"],
    ["--reasoning-parser qwen 3"],
    ["--reasoning-parser qwen3;id"],
    ["--reasoning-parser $X"],
    ["--reasoning-parser -x"],
    ["--reasoning-parser .x"],
    ["--reasoning-parser " + "a" * 41],
    ["--reasoning-parser ä"],
    ["--reasoning-parser qwen3\n--trust-remote-code"],
    ["--enable-auto-tool-choice true"],
    ["--enable-auto-tool-choice=true"],
    ["--reasoning-parser a", "--reasoning-parser b"],
    ["--enable-auto-tool-choice", "--enable-auto-tool-choice"],
    ["--REASONING-PARSER qwen3"],
    ["reasoning-parser qwen3"],
    [""],
], ids=str)
def test_a_switch_off_the_list_or_with_a_broken_value_is_no_note(args):
    with pytest.raises(ValueError):
        SetupNote.model_validate(_note(vllm_args=args))


def test_a_value_of_forty_characters_and_a_dash_inside_are_fine():
    name = "a" + "b-c.d_" * 6 + "e"
    assert len(name) == 38 and SetupNote.model_validate(_note(vllm_args=[f"--kv-cache-dtype {name}"])).vllm_args == [f"--kv-cache-dtype {name}"]
    full = "a" * 40
    assert SetupNote.model_validate(_note(vllm_args=[f"--tool-call-parser {full}"])).vllm_args
    with pytest.raises(ValueError):
        SetupNote.model_validate(_note(vllm_args=[f"--tool-call-parser {full}a"]))


def test_switches_belong_to_a_note_that_found_instructions():
    for status in ("none_recognized",):
        with pytest.raises(ValueError, match="vllm_args"):
            SetupNote.model_validate(_note(status=status, package=None, vllm_args=THREE))
    with pytest.raises(ValueError):
        SetupNote.model_validate({
            "model": "acme/Nova-7B-2512", "status": "not_read", "card_revision": None, "checked": "2026-10-14",
            "source_url": "https://huggingface.co/acme/Nova-7B-2512", "reason": "x", "vllm_args": THREE,
        })


# --- schema 4 -----------------------------------------------------------------------------------------------------------


def test_a_schema_4_document_carries_the_switches_of_a_note(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_v4_with_switches()), encoding="utf-8")
    catalog = load_catalog(path)
    assert catalog.schema_version == 4
    assert catalog.model_for("acme/Nova-7B-2512").setup_note.vllm_args == SWITCHES
    assert catalog.setup_note_for("acme/Nova-7B-2512").vllm_args == SWITCHES


def test_a_schema_4_document_without_any_switch_reads_like_schema_3(tmp_path):
    path = tmp_path / "catalog.json"
    four = _v4([_row(0, gguf_checks=[], setup_note=_note()), _row(1), _row(2)])
    path.write_text(json.dumps(four), encoding="utf-8")
    assert load_catalog(path).model_for("acme/Nova-7B-2512").setup_note.vllm_args == []


def test_a_schema_4_document_must_say_4_and_schema_3_and_2_refuse_the_switches():
    with pytest.raises(Exception):
        CatalogDocumentV4.model_validate({**_v4(), "schema_version": 3})
    with pytest.raises(Exception, match="schema 3 has no vllm_args"):
        CatalogDocumentV3.model_validate({**_v4_with_switches(), "schema_version": 3})
    with pytest.raises(Exception):
        CatalogDocumentV2.model_validate({**_v4_with_switches(), "schema_version": 2})
    CatalogDocumentV4.model_validate(_v4_with_switches())


@pytest.mark.parametrize("value", [[], ["--enable-auto-tool-choice"]], ids=["explicitly empty", "filled"])
def test_schema_3_refuses_the_key_vllm_args_whatever_it_holds(value):
    """Release 1.1.0 refuses the key itself (`extra="forbid"`), so a v3 file with an empty list would not read there either."""
    document = {**_v4([_row(0, gguf_checks=[], setup_note=_note(vllm_args=value)), _row(1)]), "schema_version": 3}
    with pytest.raises(Exception, match="schema 3 has no vllm_args"):
        CatalogDocumentV3.model_validate(document)
    CatalogDocumentV4.model_validate({**document, "schema_version": 4})


def test_a_working_catalog_of_schema_3_with_an_explicit_empty_list_is_not_published_as_the_v3_file():
    document = {**_v4([_row(0, gguf_checks=[], setup_note=_note(vllm_args=[])), _row(1)]), "schema_version": 3}
    with pytest.raises(P.PublishError):
        P.channel_files(_bytes(document))


def test_a_working_catalog_of_schema_4_with_an_explicit_empty_list_makes_a_v3_file_without_the_key():
    files = P.channel_files(_bytes(_v4([_row(0, gguf_checks=[], setup_note=_note(vllm_args=[])), _row(1)])))
    assert "vllm_args" not in files["catalog-v3.json"].decode("utf-8")
    build_catalog_v3(files["catalog-v3.json"].decode("utf-8"), Path("x"))


def test_load_catalog_reads_schema_2_3_and_4_and_refuses_5(tmp_path):
    path = tmp_path / "five.json"
    path.write_text(json.dumps({**_v4(), "schema_version": 5}), encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        load_catalog(path)
    for version in (2, 3, 4):
        path.write_text(json.dumps({**_document(), "schema_version": version}), encoding="utf-8")
        assert load_catalog(path).schema_version == version


def test_the_builders_dispatch_on_the_schema_of_the_content_for_all_three():
    assert [build_catalog(json.dumps({**_document(), "schema_version": v}), Path("c.json")).schema_version for v in (2, 3, 4)] == [2, 3, 4]
    assert build_catalog_v4(json.dumps(_v4()), Path("c.json")).schema_version == 4
    for wrong in (2, 3):
        with pytest.raises(SchemaVersionError):
            build_catalog_v4(json.dumps({**_document(), "schema_version": wrong}), Path("c.json"))
    with pytest.raises(SchemaVersionError):
        build_catalog_v3(json.dumps(_v4()), Path("c.json"))


def test_a_schema_4_file_may_hold_four_mebibytes(tmp_path):
    body = json.dumps(_v4()) + " " * (2 * 1024 * 1024)
    assert len(body.encode("utf-8")) < CATALOG_V3_MAX_BYTES
    assert build_catalog_v4(body, Path("c.json")).schema_version == 4
    with pytest.raises(Exception):
        build_catalog_v4(json.dumps(_v4()) + " " * CATALOG_V3_MAX_BYTES, Path("c.json"))


# --- the projection to schema 3 -----------------------------------------------------------------------------------------


def test_the_projection_to_schema_3_drops_the_switches_and_the_version_and_nothing_else():
    document = _v4_with_switches()
    before = copy.deepcopy(document)
    projected = project_to_schema3(document)
    assert document == before  # the argument is left alone
    assert projected["schema_version"] == 3
    expected = copy.deepcopy(document)
    expected["schema_version"] = 3
    del expected["models"][0]["setup_note"]["vllm_args"]
    assert projected == expected
    CatalogDocumentV3.model_validate(projected)  # exactly what the 1.1.0 reader takes


def test_the_projection_of_a_document_without_switches_changes_the_version_alone():
    document = _v4([_row(0, gguf_checks=[], setup_note=_note()), _row(1)])
    assert project_to_schema3(document) == {**document, "schema_version": 3}


def test_the_projection_to_schema_3_refuses_other_documents():
    for version in (1, 2, 5, None):
        with pytest.raises(ValueError):
            project_to_schema3({**_v4(), "schema_version": version})
    with pytest.raises(ValueError):
        project_to_schema3([])


def test_the_projection_keeps_a_schema_3_document_as_it_is_with_its_version():
    three = {**_v4([_row(0, gguf_checks=[], setup_note=_note())]), "schema_version": 3}
    assert project_to_schema3(three) == three


# --- the channels -------------------------------------------------------------------------------------------------------


def test_the_new_channel_has_its_file_cache_schema_and_size_and_is_the_default():
    channel = catalog_remote.CHANNEL_V4
    assert (channel.file_name, channel.cache_name, channel.schema, channel.max_bytes) == ("catalog-v4.json", "catalog-v4.cache.json", 4, 4 * 1024 * 1024)
    assert catalog_remote.DEFAULT_CHANNEL is channel
    assert len({c.cache_name for c in (catalog_remote.CHANNEL_V2, catalog_remote.CHANNEL_V3, channel)}) == 3
    assert (catalog_remote.CHANNEL_V3.file_name, catalog_remote.CHANNEL_V3.schema) == ("catalog-v3.json", 3)  # unchanged: release 1.1.0 reads it
    assert (catalog_remote.CHANNEL_V2.file_name, catalog_remote.CHANNEL_V2.schema) == ("catalog-v2.json", 2)


def _ask(transport, tmp_path: Path, **kwargs):
    return catalog_remote.current_catalog(transport, NOW, tmp_path / "cache", RequestBudget(150), **kwargs)


def test_a_guided_run_asks_the_v4_file_with_the_same_two_requests_and_keeps_the_switches(tmp_path):
    transport = FixtureTransport(_hub(catalog_remote.CHANNEL_V4, _later(_v4_with_switches())))
    choice = _ask(transport, tmp_path)
    assert choice.source == "published"
    assert [url for _method, url in transport.calls] == [REVISION_URL, _channel_url(catalog_remote.CHANNEL_V4)]
    assert choice.catalog.setup_note_for("acme/Nova-7B-2512").vllm_args == SWITCHES
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == ["catalog-v4.cache.json"]


def test_the_offline_reading_takes_the_cache_of_the_v4_channel(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    envelope = {"cache_version": 1, "checked_at": "2026-10-14T08:00:00Z", "sha": SHA, "last_error": None, "catalog": _later(_v4_with_switches())}
    (cache / "catalog-v4.cache.json").write_text(json.dumps(envelope), encoding="utf-8")
    assert catalog_remote.offline_catalog(cache).setup_note_for("acme/Nova-7B-2512").vllm_args == SWITCHES
    # the caches of the other channels are other files
    assert catalog_remote.offline_catalog(cache, channel=catalog_remote.CHANNEL_V3).published == load_catalog().published


def _faulty(kind: str) -> dict:
    channel = catalog_remote.CHANNEL_V4
    good = _later(_v4_with_switches())
    mapping = _hub(channel, good)
    url = ("GET", _channel_url(channel))
    answer = lambda body: Response(200, {"X-Repo-Commit": SHA}, body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))  # noqa: E731
    if kind == "not_found":
        mapping[url] = Response(404, {}, b"{}")
    elif kind == "not_json":
        mapping[url] = answer(b"{not json")
    elif kind == "other_commit":
        mapping[url] = Response(200, {"X-Repo-Commit": "b" * 40}, json.dumps(good).encode("utf-8"))
    elif kind == "schema_3":
        mapping[url] = answer({**good, "models": [{**row, "setup_note": None} for row in good["models"]], "schema_version": 3})
    elif kind == "schema_5":
        mapping[url] = answer({**good, "schema_version": 5})
    elif kind == "unknown_key":
        broken = copy.deepcopy(good)
        broken["models"][0]["surprise"] = 1
        mapping[url] = answer(broken)
    elif kind == "switch_off_the_list":
        broken = copy.deepcopy(good)
        broken["models"][0]["setup_note"]["vllm_args"] = ["--trust-remote-code"]
        mapping[url] = answer(broken)
    elif kind == "switch_broken_value":
        broken = copy.deepcopy(good)
        broken["models"][0]["setup_note"]["vllm_args"] = ["--reasoning-parser qwen3;id"]
        mapping[url] = answer(broken)
    elif kind == "too_large":
        mapping[url] = answer(b" " * (CATALOG_V3_MAX_BYTES + 1))
    elif kind == "no_revision":
        mapping[("GET", REVISION_URL)] = Response(200, {}, b'{"sha": "nope"}')
    return mapping


@pytest.mark.parametrize("kind", [
    "not_found", "not_json", "other_commit", "schema_3", "schema_5", "unknown_key", "switch_off_the_list", "switch_broken_value", "too_large", "no_revision",
])
def test_the_v4_channel_falls_back_to_the_shipped_catalog_and_says_why(kind, tmp_path):
    choice = _ask(FixtureTransport(_faulty(kind)), tmp_path)
    assert choice.source == "shipped" and choice.catalog.published == load_catalog().published
    assert "the published one did not read" in choice.origin
    envelope = json.loads((tmp_path / "cache" / "catalog-v4.cache.json").read_text(encoding="utf-8"))
    assert envelope["catalog"] is None and envelope["last_error"]


def test_a_schema_3_file_on_the_v4_channel_and_a_schema_4_file_on_the_v3_channel_are_faults_naming_the_schema(tmp_path):
    assert "schema_version 3" in _ask(FixtureTransport(_faulty("schema_3")), tmp_path / "a").origin
    four = _later(_v4_with_switches())
    old = _ask(FixtureTransport(_hub(catalog_remote.CHANNEL_V3, four)), tmp_path / "b", channel=catalog_remote.CHANNEL_V3)
    assert old.source == "shipped" and "schema_version 4" in old.origin


def test_a_tampered_cache_of_the_v4_channel_with_a_schema_3_catalog_is_not_taken(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    three = {**_v4(), "schema_version": 3}
    envelope = {"cache_version": 1, "checked_at": "2026-10-14T08:00:00Z", "sha": SHA, "last_error": None, "catalog": _later(three)}
    (cache / "catalog-v4.cache.json").write_text(json.dumps(envelope), encoding="utf-8")

    def offline(method, url, headers=None, *, max_bytes=None):
        raise OSError("network is unreachable")

    assert _ask(offline, tmp_path).source == "shipped"


def test_a_good_cache_of_the_v4_channel_survives_a_day_the_file_does_not_read(tmp_path):
    assert _ask(FixtureTransport(_hub(catalog_remote.CHANNEL_V4, _later(_v4_with_switches()))), tmp_path).source == "published"
    new_sha = "4" * 40
    moved = {
        ("GET", REVISION_URL): Response(200, {}, json.dumps({"sha": new_sha}).encode("utf-8")),
        ("GET", _channel_url(catalog_remote.CHANNEL_V4, new_sha)): Response(200, {"X-Repo-Commit": new_sha}, b"{not json"),
    }
    later = catalog_remote.current_catalog(FixtureTransport(moved), NOW + timedelta(days=2), tmp_path / "cache", RequestBudget(150))
    assert later.source == "cached" and later.catalog.setup_note_for("acme/Nova-7B-2512").vllm_args == SWITCHES
    assert "the published one did not read" in later.origin


# --- the files one publication makes ------------------------------------------------------------------------------------


def _bytes(document: dict) -> bytes:
    return P.R.dump_json(document)


def test_a_working_catalog_of_schema_4_makes_three_files_each_read_by_its_own_reader():
    working = _bytes(_v4_with_switches())
    files = P.channel_files(working)
    assert tuple(files) == ("catalog-v4.json", "catalog-v3.json", "catalog-v2.json")
    assert files["catalog-v4.json"] == working
    three, two = json.loads(files["catalog-v3.json"]), json.loads(files["catalog-v2.json"])
    assert three["schema_version"] == 3 and two["schema_version"] == 2
    assert "vllm_args" not in json.dumps(three) and "vllm_args" not in json.dumps(two)
    assert three["models"][0]["setup_note"]["package"] == "nova-inference"  # the note itself stands in the 1.1.0 file
    assert two["models"][0].get("setup_note") is None and "setup_note" not in two["models"][0]
    assert build_catalog_v4(files["catalog-v4.json"].decode("utf-8"), Path("x")).setup_note_for("acme/Nova-7B-2512").vllm_args == SWITCHES
    assert build_catalog_v3(files["catalog-v3.json"].decode("utf-8"), Path("x")).setup_note_for("acme/Nova-7B-2512").vllm_args == []


def test_the_v3_file_of_a_schema_4_catalog_is_the_projection_and_nothing_else():
    working = _v4_with_switches()
    files = P.channel_files(_bytes(working))
    assert files["catalog-v3.json"] == _bytes(project_to_schema3(working))


def test_a_working_catalog_of_schema_3_makes_its_v3_file_byte_for_byte_and_a_v4_file_that_differs_in_the_version_alone():
    three = {**_v4([_row(0, gguf_checks=[], setup_note=_note()), _row(1), _row(2)]), "schema_version": 3}
    working = _bytes(three)
    files = P.channel_files(working)
    assert files["catalog-v3.json"] == working
    assert json.loads(files["catalog-v4.json"]) == {**three, "schema_version": 4}
    assert json.loads(files["catalog-v2.json"])["schema_version"] == 2


def test_a_working_catalog_of_schema_2_makes_the_three_files_with_its_own_bytes_for_the_old_one():
    working = _bytes(_document())
    files = P.channel_files(working)
    assert files["catalog-v2.json"] == working
    assert json.loads(files["catalog-v3.json"]) == {**_document(), "schema_version": 3}
    assert json.loads(files["catalog-v4.json"]) == {**_document(), "schema_version": 4}


def test_the_files_of_a_working_catalog_of_schema_5_or_without_a_reading_are_refused():
    with pytest.raises(P.PublishError):
        P.channel_files(_bytes({**_v4(), "schema_version": 5}))
    with pytest.raises(P.PublishError):
        P.channel_files(_bytes(_v4_with_switches(["--trust-remote-code"])))


# --- the readers of releases 1.0.0 and 1.1.0 ----------------------------------------------------------------------------

_OLD_RUN = (
    "import json, sys\n"
    "from datetime import datetime, timezone\n"
    "from pathlib import Path\n"
    "from modelroom.catalog_remote import current_catalog\n"
    "from modelroom.http import FixtureTransport, RequestBudget, Response\n"
    "data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
    "sha = data['sha']\n"
    "base = 'https://huggingface.co/datasets/jm-vis/modelroom-catalog'\n"
    "mapping = {('GET', 'https://huggingface.co/api/datasets/jm-vis/modelroom-catalog/revision/main'): Response(200, {}, json.dumps({'sha': sha}).encode())}\n"
    "for name, text in data['files'].items():\n"
    "    mapping[('GET', f'{base}/resolve/{sha}/{name}')] = Response(200, {'X-Repo-Commit': sha}, text.encode())\n"
    "transport = FixtureTransport(mapping)\n"
    "choice = current_catalog(transport, datetime(2026, 10, 14, 9, tzinfo=timezone.utc), Path(sys.argv[2]), RequestBudget(150))\n"
    "print(json.dumps({'calls': [url.rsplit('/', 1)[-1] for _m, url in transport.calls], 'source': choice.source,\n"
    "                  'schema': choice.catalog.schema_version, 'published': choice.catalog.published.isoformat(), 'origin': choice.origin,\n"
    "                  'notes': {m.hf_repo: m.setup_note.package for f in choice.catalog.families for m in f.models if getattr(m, 'setup_note', None)}}))\n"
)


def _release(commit: str, folder: Path) -> Path:
    if not (_REPO_ROOT / ".git").exists():
        pytest.skip("the reader of an earlier release is taken from the repository's history; not in a source distribution")
    archive = subprocess.run(["git", "archive", commit, "modelroom"], cwd=_REPO_ROOT, capture_output=True)
    if archive.returncode != 0:
        pytest.skip(f"commit {commit} is not in this clone")
    folder.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(folder, filter="data")
    return folder


def _old_reader(commit: str, tmp_path: Path, files: dict[str, dict]) -> dict:
    old = _release(commit, tmp_path / "old")
    data = tmp_path / "dataset.json"
    data.write_text(json.dumps({"sha": SHA, "files": {name: json.dumps(document) for name, document in files.items()}}), encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key != "MODELROOM_CATALOG"}
    run = subprocess.run(
        [sys.executable, "-c", _OLD_RUN, str(data), str(tmp_path / "old-cache")], cwd=tmp_path, capture_output=True, text=True,
        env={**env, "PYTHONPATH": str(old)},
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _dataset() -> dict[str, dict]:
    """The three files one publication makes from a working catalog of schema 4 that holds a note with switches."""
    files = P.channel_files(_bytes(_later(_v4_with_switches())))
    return {name: json.loads(data) for name, data in files.items()}


def test_release_1_0_0_reads_the_v2_file_unchanged_and_never_asks_for_another(tmp_path):
    files = _dataset()
    answer = _old_reader(_RELEASE_100, tmp_path, files)
    assert answer["calls"] == ["main", "catalog-v2.json"]
    assert (answer["source"], answer["schema"], answer["published"]) == ("published", 2, files["catalog-v2.json"]["published"])
    assert "did not read" not in answer["origin"]


def test_release_1_1_0_reads_the_v3_file_with_its_note_and_never_asks_for_the_v4_file(tmp_path):
    files = _dataset()
    answer = _old_reader(_RELEASE_110, tmp_path, files)
    assert answer["calls"] == ["main", "catalog-v3.json"]
    assert (answer["source"], answer["schema"], answer["published"]) == ("published", 3, files["catalog-v3.json"]["published"])
    assert answer["notes"] == {"acme/Nova-7B-2512": "nova-inference"} and "did not read" not in answer["origin"]


def test_release_1_1_0_would_refuse_the_v4_file_if_it_ever_met_it(tmp_path):
    """Why the channels are files of their own: a note with `vllm_args` fails the validation of release 1.1.0."""
    files = _dataset()
    swapped = {"catalog-v3.json": files["catalog-v4.json"]}
    answer = _old_reader(_RELEASE_110, tmp_path, swapped)
    assert answer["source"] == "shipped" and "schema_version 4" in answer["origin"]


def test_release_1_1_0_would_refuse_a_note_with_switches_even_under_its_own_schema(tmp_path):
    files = _dataset()
    bad = {"catalog-v3.json": {**files["catalog-v4.json"], "schema_version": 3}}
    answer = _old_reader(_RELEASE_110, tmp_path, bad)
    assert answer["source"] == "shipped" and "did not read" in answer["origin"]


def test_this_release_reads_the_v4_file_of_that_same_dataset_with_the_switches(tmp_path):
    files = _dataset()
    mapping = {
        **_hub(catalog_remote.CHANNEL_V4, files["catalog-v4.json"]),
        ("GET", _channel_url(catalog_remote.CHANNEL_V3)): Response(200, {"X-Repo-Commit": SHA}, json.dumps(files["catalog-v3.json"]).encode("utf-8")),
        ("GET", _channel_url(catalog_remote.CHANNEL_V2)): Response(200, {"X-Repo-Commit": SHA}, json.dumps(files["catalog-v2.json"]).encode("utf-8")),
    }
    transport = FixtureTransport(mapping)
    choice = _ask(transport, tmp_path)
    assert [url.rsplit("/", 1)[-1] for _method, url in transport.calls] == ["main", "catalog-v4.json"]
    assert choice.catalog.schema_version == 4 and choice.catalog.setup_note_for("acme/Nova-7B-2512").vllm_args == SWITCHES


# --- the command --------------------------------------------------------------------------------------------------------


def _note_for(args: list[str], *, status: str = "instructions") -> SetupNote:
    return SetupNote.model_validate({
        "model": BASE_REPO, "status": status, "card_revision": REVISION, "checked": "2026-10-14",
        "source_url": f"https://huggingface.co/{BASE_REPO}/blob/{REVISION}/README.md", "package": None, "reason": None, "vllm_args": args,
    })


def _install(note: SetupNote | None):
    profile, package = _profile(), _package()
    return vllm_install(package, _fit(profile, package), profile, setup_note=note)


def test_the_switches_stand_after_modelrooms_own_in_both_lines_and_before_host_and_port():
    install = _install(_note_for(THREE))
    tail = " --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --tool-call-parser hermes --enable-auto-tool-choice"
    assert install.serve == SERVE.replace(" --gpu-memory-utilization 0.92", tail)
    assert install.docker == DOCKER.replace(" --gpu-memory-utilization 0.92", tail)
    assert install.serve.endswith("--host 127.0.0.1 --port 8000") and install.docker.endswith("--host 0.0.0.0 --port 8000")


def test_all_four_switches_reach_both_lines_once_each():
    install = _install(_note_for(SWITCHES))
    for line in (install.serve, install.docker):
        for arg in SWITCHES:
            assert line.count(arg) == 1
        assert "--trust-remote-code" not in line


def test_without_switches_or_without_a_note_both_lines_are_the_lines_of_before():
    for note in (None, _note_for([]), _note_for([], status="none_recognized")):
        install = _install(note)
        assert (install.serve, install.docker) == (SERVE, DOCKER)


def test_a_note_that_found_no_instructions_adds_nothing_even_if_it_holds_switches():
    unread = SetupNote.model_construct(**{**_note_for([]).__dict__, "status": "none_recognized", "vllm_args": THREE})
    assert (_install(unread).serve, _install(unread).docker) == (SERVE, DOCKER)


def test_modelroom_wins_a_switch_the_command_names_already_is_not_set_again():
    taken = "repo --revision r --kv-cache-dtype auto --max-model-len 1"
    assert card_switches(_note_for(SWITCHES), taken) == " --reasoning-parser qwen3 --tool-call-parser hermes --enable-auto-tool-choice"
    assert card_switches(_note_for(SWITCHES), "x --reasoning-parser a --tool-call-parser b --enable-auto-tool-choice --kv-cache-dtype c") == ""
    assert card_switches(None, "x") == ""


def test_modelroom_sets_none_of_the_listed_switches_itself_so_the_card_decides_all_four():
    """The check the decision asked for: today no switch of the list is in modelroom's own part of the command."""
    own = _install(None).serve
    assert not any(name in own for name in (*VLLM_VALUE_SWITCHES, *VLLM_FLAG_SWITCHES))


def test_a_note_that_did_not_come_through_its_contract_cannot_add_anything_but_the_list():
    hostile = SetupNote.model_construct(
        model=BASE_REPO, status="instructions", vllm_args=["--trust-remote-code", "--reasoning-parser qwen3; rm -rf /", "--kv-cache-dtype fp8", "--host 0.0.0.0"]
    )
    assert card_switches(hostile, "x") == " --kv-cache-dtype fp8"


def test_a_row_without_a_command_gets_no_switches():
    from modelroom.install_vllm import VllmInstall

    assert vllm_install(None, _fit(_profile(), _package()), _profile(), setup_note=_note_for(THREE)) == VllmInstall(None, reason="no safe command")


# --- the views ----------------------------------------------------------------------------------------------------------


def test_the_table_the_choice_and_the_handover_carry_the_same_switches(tmp_path):
    from modelroom.dialog import FileAsker
    from modelroom.guided_install import PullContext, pull_step
    from modelroom.pull_choice import ambiguous_names
    from modelroom.pull_other import pull_other_step
    from test_install_vllm import _run

    note = _note_for(THREE)
    document = _with_note(_both(), note)
    with_switches = SERVE.replace(" --gpu-memory-utilization 0.92", " --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --tool-call-parser hermes --enable-auto-tool-choice")
    text = _vllm_text(document)
    assert any(line.strip().endswith(with_switches) for line in text)  # the table
    packages = [_gguf_package(), _package()]
    run, lines = _run(tmp_path / "a", FileAsker({"pull_package": f"vllm {REPO}"}))
    pull_step(run, PullContext(document, "server", packages, ambiguous_names(packages)), True)
    assert with_switches in "\n".join(lines)  # step 5
    run, lines = _run(tmp_path / "b", FileAsker({"pull_other": f"server vllm {REPO}"}))
    pull_other_step(run, PullContext(document, "laptop", packages, ambiguous_names(packages)))
    assert "--reasoning-parser qwen3 --tool-call-parser hermes --enable-auto-tool-choice" in "\n".join(lines)  # the handover


def test_the_table_of_a_note_without_switches_is_the_table_of_before():
    text = _vllm_text(_with_note(_both(), _note_for([])))
    assert any(line.strip().endswith(SERVE) for line in text) and not any("--reasoning-parser" in line for line in text)


def test_the_render_documents_json_carries_the_switches_where_there_are_some_and_no_key_where_not():
    from modelroom.render import document_json

    with_args = document_json(_with_note(_both(), _note_for(THREE)))
    assert with_args["setup_notes"][BASE_REPO]["vllm_args"] == THREE
    without = document_json(_with_note(_both(), _note_for([])))
    assert "vllm_args" not in without["setup_notes"][BASE_REPO]
    assert "vllm_args" not in json.dumps(document_json(_with_note(_both(), None)))


def test_the_markdown_view_is_the_same_with_or_without_switches():
    from modelroom.views import document_markdown

    assert document_markdown(_with_note(_both(), _note_for(THREE))) == document_markdown(_with_note(_both(), _note_for([])))


def test_the_terminal_lines_over_the_commands_do_not_change_with_switches():
    from modelroom.catalog_findings import setup_note_lines

    assert setup_note_lines(_note_for(THREE)) == setup_note_lines(_note_for([]))
