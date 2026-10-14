"""AP9-KO P5, part 2: the guided run asks the new channel (`catalog-v3.json`), falls back cleanly where it does not read,
and release 1.0.0 goes on reading the old channel (`catalog-v2.json`) exactly as it did (CONTRACTS.md, "Published
catalog", "Two channels"; "Package findings and the load check", V3).

The reader of release 1.0.0 is its own code: the package of the commit `main` stood at when AP9-KO began, run in a
process of its own against a dataset that holds both files.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import timedelta
from pathlib import Path

import pytest

from modelroom import catalog_remote
from modelroom.catalog import CATALOG_V3_MAX_BYTES, load_catalog
from modelroom.catalog_schema import project_to_schema2
from modelroom.http import FixtureTransport, RequestBudget, Response

from test_catalog import FIXTURES
from test_ko_catalog_v3 import NOW, REVISION_URL, SHA, _channel_url, _hub, _v3, _with_findings

_OLD_RELEASE = "2d322fb"
_REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _published_catalog_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELROOM_CATALOG", raising=False)


def _later(document: dict, days: int = 3) -> dict:
    document["published"] = (load_catalog().published + timedelta(days=days)).isoformat()
    return document


def _ask(transport, tmp_path: Path):
    """What a guided run does: `current_catalog` with no channel named."""
    return catalog_remote.current_catalog(transport, NOW, tmp_path / "cache", RequestBudget(150))


# --- the switch: one line, and everything that follows from it ----------------------------------------------------


def test_a_run_that_names_no_channel_asks_the_new_one():
    assert catalog_remote.DEFAULT_CHANNEL is catalog_remote.CHANNEL_V3
    assert catalog_remote.CHANNEL_V2.file_name == "catalog-v2.json"  # the old channel keeps its file for release 1.0.0


def test_the_guided_call_asks_the_new_file_and_keeps_the_new_cache_only(tmp_path):
    body = _later(_with_findings())
    transport = FixtureTransport(_hub(catalog_remote.CHANNEL_V3, body))

    choice = _ask(transport, tmp_path)

    assert choice.source == "published"
    assert [url for _method, url in transport.calls] == [REVISION_URL, _channel_url(catalog_remote.CHANNEL_V3)]
    assert choice.catalog.schema_version == 3 and choice.catalog.model_for("acme/Nova-7B-2512").gguf_checks
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == ["catalog-v3.cache.json"]


def test_the_offline_reading_of_render_and_pull_takes_the_new_cache_and_not_the_old_one(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    body = _later(_with_findings())
    old = _later(json.loads((FIXTURES / "catalog_v2_published.json").read_text(encoding="utf-8")), days=9)
    for name, document in (("catalog-v3.cache.json", body), ("catalog-v2.cache.json", old)):
        envelope = {"cache_version": 1, "checked_at": "2026-10-14T08:00:00Z", "sha": SHA, "last_error": None, "catalog": document}
        (cache / name).write_text(json.dumps(envelope), encoding="utf-8")

    found = catalog_remote.offline_catalog(cache)

    assert found.schema_version == 3 and found.published.isoformat() == body["published"]  # not the later day of the old file


def test_a_cache_release_1_0_0_left_behind_is_not_read_by_the_new_release_and_stays_as_it_was(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    old = _later(json.loads((FIXTURES / "catalog_v2_published.json").read_text(encoding="utf-8")))
    envelope = {"cache_version": 1, "checked_at": "2026-10-14T08:00:00Z", "sha": SHA, "last_error": None, "catalog": old}
    path = cache / "catalog-v2.cache.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    before = path.read_bytes()
    transport = FixtureTransport(_hub(catalog_remote.CHANNEL_V3, _later(_v3())))

    choice = catalog_remote.current_catalog(transport, NOW, cache, RequestBudget(150))

    assert choice.catalog.schema_version == 3
    assert path.read_bytes() == before


# --- every fault ends in the shipped catalog, with its reason -----------------------------------------------------


def _faulty(kind: str) -> dict:
    channel = catalog_remote.CHANNEL_V3
    good = _later(_with_findings())
    mapping = _hub(channel, good)
    url = ("GET", _channel_url(channel))
    if kind == "not_found":
        mapping[url] = Response(404, {}, b"{}")
    elif kind == "not_json":
        mapping[url] = Response(200, {"X-Repo-Commit": SHA}, b"{not json")
    elif kind == "other_commit":
        mapping[url] = Response(200, {"X-Repo-Commit": "b" * 40}, json.dumps(good).encode("utf-8"))
    elif kind == "schema_2":
        mapping[url] = Response(200, {"X-Repo-Commit": SHA}, json.dumps(_later(json.loads((FIXTURES / "catalog_v2_published.json").read_text(encoding="utf-8")))).encode("utf-8"))
    elif kind == "schema_4":
        mapping[url] = Response(200, {"X-Repo-Commit": SHA}, json.dumps({**good, "schema_version": 4}).encode("utf-8"))
    elif kind == "unknown_key":
        broken = _later(_with_findings())
        broken["models"][0]["surprise"] = 1
        mapping[url] = Response(200, {"X-Repo-Commit": SHA}, json.dumps(broken).encode("utf-8"))
    elif kind == "too_large":
        mapping[url] = Response(200, {"X-Repo-Commit": SHA}, b" " * (CATALOG_V3_MAX_BYTES + 1))
    elif kind == "no_revision":
        mapping[("GET", REVISION_URL)] = Response(200, {}, b'{"sha": "nope"}')
    return mapping


@pytest.mark.parametrize("kind", ["not_found", "not_json", "other_commit", "schema_2", "schema_4", "unknown_key", "too_large", "no_revision"])
def test_the_new_channel_falls_back_to_the_shipped_catalog_and_says_why(kind, tmp_path):
    transport = FixtureTransport(_faulty(kind))

    choice = _ask(transport, tmp_path)

    assert choice.source == "shipped" and choice.catalog.published == load_catalog().published
    assert "the published one did not read" in choice.origin
    envelope = json.loads((tmp_path / "cache" / "catalog-v3.cache.json").read_text(encoding="utf-8"))
    assert envelope["catalog"] is None and envelope["last_error"]  # the day's attempt is kept, nothing else is


def test_a_schema_2_file_on_the_new_channel_is_a_fault_naming_the_schema(tmp_path):
    choice = _ask(FixtureTransport(_faulty("schema_2")), tmp_path)

    assert choice.source == "shipped" and "schema_version 2" in choice.origin


def test_a_good_cache_survives_a_day_the_new_channel_does_not_read(tmp_path):
    first = _ask(FixtureTransport(_hub(catalog_remote.CHANNEL_V3, _later(_with_findings()))), tmp_path)
    assert first.source == "published"

    new_sha = "4" * 40  # the dataset moved on, and its new file does not read
    moved = {
        ("GET", REVISION_URL): Response(200, {}, json.dumps({"sha": new_sha}).encode("utf-8")),
        ("GET", _channel_url(catalog_remote.CHANNEL_V3, new_sha)): Response(200, {"X-Repo-Commit": new_sha}, b"{not json"),
    }
    later_day = catalog_remote.current_catalog(FixtureTransport(moved), NOW + timedelta(days=2), tmp_path / "cache", RequestBudget(150))

    assert later_day.source == "cached" and later_day.catalog.model_for("acme/Nova-7B-2512").gguf_checks
    assert "the published one did not read" in later_day.origin


# --- release 1.0.0 against a dataset that holds both channels ------------------------------------------------------

_OLD_RUN = (
    "import json, sys\n"
    "from datetime import datetime, timezone\n"
    "from pathlib import Path\n"
    "from modelroom.catalog_remote import current_catalog\n"
    "from modelroom.http import FixtureTransport, RequestBudget, Response\n"
    "data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
    "sha = data['sha']\n"
    "base = 'https://huggingface.co/datasets/jm-vis/modelroom-catalog'\n"
    "files = {'catalog-v2.json': data['v2'], 'catalog-v3.json': data['v3']}\n"
    "mapping = {('GET', 'https://huggingface.co/api/datasets/jm-vis/modelroom-catalog/revision/main'): Response(200, {}, json.dumps({'sha': sha}).encode())}\n"
    "for name, text in files.items():\n"
    "    mapping[('GET', f'{base}/resolve/{sha}/{name}')] = Response(200, {'X-Repo-Commit': sha}, text.encode())\n"
    "transport = FixtureTransport(mapping)\n"
    "choice = current_catalog(transport, datetime(2026, 10, 14, 9, tzinfo=timezone.utc), Path(sys.argv[2]), RequestBudget(150))\n"
    "print(json.dumps({'calls': [url.rsplit('/', 1)[-1] for _m, url in transport.calls], 'source': choice.source,\n"
    "                  'schema': choice.catalog.schema_version, 'published': choice.catalog.published.isoformat(), 'origin': choice.origin}))\n"
)


def _old_package(tmp_path: Path) -> Path:
    if not (_REPO / ".git").exists():
        pytest.skip("the reader of release 1.0.0 is taken from the repository's history; not in a source distribution")
    archive = subprocess.run(["git", "archive", _OLD_RELEASE, "modelroom"], cwd=_REPO, capture_output=True)
    if archive.returncode != 0:
        pytest.skip(f"commit {_OLD_RELEASE} is not in this clone")
    old = tmp_path / "old"
    old.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(old, filter="data")
    return old


def _old_reader(tmp_path: Path, v2: dict, v3: dict) -> dict:
    old = _old_package(tmp_path)
    data = tmp_path / "dataset.json"
    data.write_text(json.dumps({"sha": SHA, "v2": json.dumps(v2), "v3": json.dumps(v3)}), encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key != "MODELROOM_CATALOG"}
    run = subprocess.run(
        [sys.executable, "-c", _OLD_RUN, str(data), str(tmp_path / "old-cache")], cwd=tmp_path, capture_output=True, text=True,
        env={**env, "PYTHONPATH": str(old)},
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _dataset() -> tuple[dict, dict]:
    """The two files one publication makes from a working catalog of schema 3 that holds a finding."""
    three = _later(_with_findings())
    return _later(project_to_schema2({**three, "models": [{**row, "gguf_checks": [], "setup_note": None} for row in three["models"]]})), three


def test_release_1_0_0_reads_the_old_channel_unchanged_and_never_asks_for_the_new_file(tmp_path):
    two, three = _dataset()

    answer = _old_reader(tmp_path, two, three)

    assert answer["calls"] == ["main", f"{'catalog-v2.json'}"] or answer["calls"] == ["main", "catalog-v2.json"]
    assert "catalog-v3.json" not in answer["calls"]
    assert (answer["source"], answer["schema"], answer["published"]) == ("published", 2, two["published"])
    assert "did not read" not in answer["origin"]


def test_this_release_reads_the_new_file_of_that_same_dataset_with_its_finding(tmp_path):
    two, three = _dataset()
    mapping = {
        **_hub(catalog_remote.CHANNEL_V3, three),
        ("GET", _channel_url(catalog_remote.CHANNEL_V2)): Response(200, {"X-Repo-Commit": SHA}, json.dumps(two).encode("utf-8")),
    }
    transport = FixtureTransport(mapping)

    choice = _ask(transport, tmp_path)

    assert [url.rsplit("/", 1)[-1] for _method, url in transport.calls] == ["main", "catalog-v3.json"]
    assert choice.catalog.schema_version == 3 and choice.catalog.model_for("acme/Nova-7B-2512").gguf_checks[0].result == "absent"


def test_release_1_0_0_would_refuse_the_new_file_if_it_ever_met_it(tmp_path):
    """Why the channels are two files: the old reader takes a schema-3 document as an unsupported version."""
    two, three = _dataset()

    answer = _old_reader(tmp_path, three, two)  # the files swapped: the old channel holds the schema-3 document

    assert answer["source"] == "shipped" and "schema_version 3" in answer["origin"]
