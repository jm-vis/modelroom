"""Contract tests of AP9-KO P0, part 2: the catalog of schema 3, the old channel, the new channel, and the
offline reading that `render` and `pull` use (CONTRACTS.md, "Catalog rules", "Published catalog",
"Package findings and the load check").

The fictitious `acme/Nova-9B` stands for a model whose publisher has no GGUF, `packager`/`othercorp`
for the accounts that convert it. No model, publisher or packager of the real cases is named.
"""

from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from modelroom import catalog_remote
from modelroom.catalog import CATALOG_FILE, CATALOG_MAX_BYTES, Catalog, CatalogError, load_catalog
from modelroom.catalog_findings import PackageCheck, SetupNote, llamacpp_page_url
from modelroom.catalog_schema import CatalogDocumentV2, CatalogDocumentV3, build_catalog, build_catalog_v3, project_to_schema2
from modelroom.contracts import Package, SchemaVersionError
from modelroom.examples import EXAMPLES
from modelroom.http import FixtureTransport, RequestBudget, Response

from test_catalog import FIXTURES, _document, _row

COMMIT = "45451c366d4256f94fbc8845645e31a6b9155054"
REVISION = "a" * 40
OTHER_REVISION = "b" * 40
DAY = date(2026, 10, 13)
NOW = datetime(2026, 10, 14, 9, 0, 0, tzinfo=timezone.utc)
BASE = "acme/Nova-9B-2610"  # the first row of the fixture is `acme/Nova-7B-2512`; rows below use their own names


def _check(**changes) -> dict:
    data = {
        "repo": "othercorp/Nova-7B-GGUF",
        "revision": REVISION,
        "identity": "Nova-7B-Q8_0.gguf",
        "file": "Nova-7B-Q8_0.gguf",
        "quantization": "Q8_0",
        "architecture": "nova7",
        "architecture_source": "file_header",
        "result": "absent",
        "llamacpp_commit": COMMIT,
        "source_url": llamacpp_page_url(COMMIT),
        "checked": DAY.isoformat(),
        "same_quantization": 1,
    }
    data.update(changes)
    return data


def _note(**changes) -> dict:
    data = {
        "model": "acme/Nova-7B-2512",
        "status": "instructions",
        "card_revision": REVISION,
        "source_url": f"https://huggingface.co/acme/Nova-7B-2512/blob/{REVISION}/README.md",
        "checked": DAY.isoformat(),
        "package": "nova-inference",
        "reason": None,
    }
    data.update(changes)
    return data


def _v3(models: list[dict] | None = None, **overrides) -> dict:
    document = _document(models, **overrides)
    document["schema_version"] = 3
    return document


def _with_findings(checks: list[dict] | None = None, note: dict | None = None) -> dict:
    row = _row(0, gguf_checks=[_check()] if checks is None else checks, setup_note=_note() if note is None else note)
    return _v3([row, _row(1), _row(2)])


def _catalog(document: dict, tmp_path: Path) -> Catalog:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_catalog(path)


# --- the fields a row gets, and the defaults that mean unknown --------------------------------------------------


def test_a_row_without_findings_has_the_defaults_that_mean_unknown():
    row = load_catalog().families[0].models[0]
    assert (row.gguf_checks, row.setup_note) == ([], None)


def test_a_schema_3_document_carries_the_checks_and_the_note_of_a_row(tmp_path):
    catalog = _catalog(_with_findings(), tmp_path)
    row = catalog.model_for("acme/Nova-7B-2512")
    assert [check.repo for check in row.gguf_checks] == ["othercorp/Nova-7B-GGUF"]
    assert row.setup_note.package == "nova-inference"
    assert catalog.schema_version == 3


def test_a_schema_3_document_without_any_finding_is_valid_and_reads_like_schema_2(tmp_path):
    catalog = _catalog(_v3(), tmp_path)
    assert all(row.gguf_checks == [] and row.setup_note is None for family in catalog.families for row in family.models)


def test_the_shipped_catalog_stays_a_schema_2_document_until_it_holds_findings():
    assert json.loads(CATALOG_FILE.read_text(encoding="utf-8"))["schema_version"] == 2


# --- schema 2 stays exactly the shape release 1.0.0 reads -------------------------------------------------------


@pytest.mark.parametrize("fields", [{"gguf_checks": [_check()]}, {"setup_note": _note()}, {"gguf_checks": []}, {"setup_note": None}])
def test_a_schema_2_row_never_carries_a_schema_3_field_not_even_at_its_default(fields):
    """Release 1.0.0 refuses an unknown key in a row (`extra="forbid"`), so a key at its default would
    break it as much as a finding would."""
    with pytest.raises(Exception, match="schema 2"):
        CatalogDocumentV2.model_validate(_document([_row(0, **fields)]))


def test_the_schema_2_reader_refuses_schema_3_as_before():
    with pytest.raises(Exception):
        CatalogDocumentV2.model_validate(_v3())


def test_a_schema_3_document_must_say_3():
    with pytest.raises(Exception):
        CatalogDocumentV3.model_validate(_document())  # says 2


def test_load_catalog_reads_schema_2_and_3_and_refuses_4(tmp_path):
    assert load_catalog(FIXTURES / "catalog_v2_valid.json").schema_version == 2
    assert _catalog(_v3(), tmp_path).schema_version == 3
    broken = _v3()
    broken["schema_version"] = 4
    path = tmp_path / "four.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        load_catalog(path)


def test_the_projection_to_schema_2_drops_the_fields_at_their_default_and_only_those():
    document = _v3()
    for row in document["models"]:
        row.update(gguf_checks=[], setup_note=None)
    projected = project_to_schema2(document)
    assert projected["schema_version"] == 2
    assert all("gguf_checks" not in row and "setup_note" not in row for row in projected["models"])
    CatalogDocumentV2.model_validate(projected)  # exactly what the old reader takes
    assert {key: value for key, value in projected.items() if key != "models"} == {key: value for key, value in document.items() if key not in ("models", "schema_version")} | {"schema_version": 2}


def test_the_projection_refuses_a_document_that_holds_a_finding():
    with pytest.raises(ValueError, match="finding"):
        project_to_schema2(_with_findings())


def test_the_projection_leaves_the_document_it_is_given_alone():
    document = _v3()
    before = copy.deepcopy(document)
    project_to_schema2(document)
    assert document == before


def test_the_published_fixture_is_still_what_the_old_reader_takes():
    CatalogDocumentV2.model_validate(json.loads((FIXTURES / "catalog_v2_published.json").read_text(encoding="utf-8")))


# The reader of release 1.0.0 itself: the package of the commit `main` stood at when AP9-KO began, run in a
# process of its own against the files this release makes. Not a copy of its rules, the real code.
_OLD_RELEASE = "2d322fb"
_REPO = Path(__file__).resolve().parent.parent
_OLD_READER = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "from modelroom.catalog import load_catalog\n"
    "answers = {}\n"
    "for path in sorted(Path(sys.argv[1]).glob('*.json')):\n"
    "    try:\n"
    "        catalog = load_catalog(path)\n"
    "        answers[path.name] = ['read', catalog.schema_version, sum(len(f.models) for f in catalog.families)]\n"
    "    except Exception as exc:\n"
    "        answers[path.name] = [type(exc).__name__]\n"
    "print(json.dumps(answers))\n"
)


def test_the_release_1_0_0_reader_takes_the_old_channel_unchanged_and_never_meets_a_finding(tmp_path):
    import io
    import os
    import subprocess
    import sys
    import tarfile

    if not (_REPO / ".git").exists():
        pytest.skip("the reader of release 1.0.0 is taken from the repository's history; not in a source distribution")
    archive = subprocess.run(["git", "archive", _OLD_RELEASE, "modelroom"], cwd=_REPO, capture_output=True)
    if archive.returncode != 0:
        pytest.skip(f"commit {_OLD_RELEASE} is not in this clone")
    old = tmp_path / "old"
    old.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(old, filter="data")

    shipped = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    v3 = copy.deepcopy(shipped)
    v3["schema_version"] = 3
    for row in v3["models"]:
        row.update(gguf_checks=[], setup_note=None)
    files = tmp_path / "files"
    files.mkdir()
    (files / "shipped_v2.json").write_bytes(CATALOG_FILE.read_bytes())
    (files / "projected_v2.json").write_text(json.dumps(project_to_schema2(v3)), encoding="utf-8")
    (files / "v3_without_findings.json").write_text(json.dumps(v3), encoding="utf-8")
    with_finding = copy.deepcopy(v3)
    with_finding["models"][0]["gguf_checks"] = [_check(repo="othercorp/Nova-9B-GGUF")]
    (files / "v3_with_a_finding.json").write_text(json.dumps(with_finding), encoding="utf-8")
    default_key = copy.deepcopy(shipped)
    default_key["models"][0]["gguf_checks"] = []
    (files / "v2_with_a_key_at_its_default.json").write_text(json.dumps(default_key), encoding="utf-8")
    # what this release writes when it reads a schema-2 document and writes it again (V3, serialization)
    (files / "v2_written_again.json").write_text(CatalogDocumentV2.model_validate(shipped).model_dump_json(), encoding="utf-8")

    run = subprocess.run(
        [sys.executable, "-c", _OLD_READER, str(files)], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(old)},
    )
    assert run.returncode == 0, run.stderr
    answers = json.loads(run.stdout)
    rows = len(shipped["models"])
    assert answers["shipped_v2.json"] == ["read", 2, rows]
    assert answers["projected_v2.json"] == ["read", 2, rows]  # the old channel, made by the projection
    assert answers["v3_without_findings.json"] == ["SchemaVersionError"]
    assert answers["v3_with_a_finding.json"] == ["SchemaVersionError"]
    assert answers["v2_with_a_key_at_its_default.json"] == ["CatalogError"]  # why the old channel is made by projection only
    assert answers["v2_written_again.json"] == ["read", 2, rows]
    # the same files, read by this release
    assert load_catalog(files / "projected_v2.json").schema_version == 2
    assert load_catalog(files / "v3_with_a_finding.json").schema_version == 3


# --- the rules of a row with findings ----------------------------------------------------------------------------


def test_a_note_belongs_to_the_model_of_its_row(tmp_path):
    with pytest.raises(CatalogError, match="setup_note"):
        _catalog(_with_findings(note=_note(model="acme/Nova-7B", source_url=f"https://huggingface.co/acme/Nova-7B/blob/{REVISION}/README.md")), tmp_path)


def test_a_card_that_was_not_read_is_not_stored(tmp_path):
    unread = _note(status="not_read", card_revision=None, package=None, reason="HTTP 503", source_url="https://huggingface.co/acme/Nova-7B-2512")
    with pytest.raises(CatalogError, match="not_read"):
        _catalog(_with_findings(note=unread), tmp_path)


def test_one_package_has_one_check_in_a_row(tmp_path):
    with pytest.raises(CatalogError, match="twice"):
        _catalog(_with_findings(checks=[_check(), _check(result="present")]), tmp_path)


def test_a_row_holds_at_most_two_hundred_checks(tmp_path):
    many = [_check(repo=f"othercorp/Nova-7B-{i}-GGUF") for i in range(200)]
    assert len(_catalog(_with_findings(checks=many), tmp_path).families) >= 1
    with pytest.raises(CatalogError):
        _catalog(_with_findings(checks=[*many, _check(repo="othercorp/Nova-7B-extra-GGUF")]), tmp_path)


def test_a_check_may_name_a_repository_of_any_account(tmp_path):
    """V6: the repositories to check are found by their quantization relation, not by a release; an account
    the catalog does not list is not made a listed packager by that."""
    catalog = _catalog(_with_findings(checks=[_check(repo="stranger/Nova-7B-GGUF")]), tmp_path)
    assert "stranger" not in catalog.packagers
    assert catalog.model_for("acme/Nova-7B-2512").gguf_checks[0].repo == "stranger/Nova-7B-GGUF"


# --- the size of the file ---------------------------------------------------------------------------------------


def test_the_limit_of_a_schema_2_file_is_still_one_mebibyte_and_a_schema_3_file_may_hold_four(tmp_path):
    from modelroom.catalog import CATALOG_V3_MAX_BYTES

    assert (CATALOG_MAX_BYTES, CATALOG_V3_MAX_BYTES) == (1024 * 1024, 4 * 1024 * 1024)
    padded = json.dumps(_v3()) + " " * (2 * 1024 * 1024)
    path = tmp_path / "big.json"
    path.write_text(padded, encoding="utf-8")
    assert load_catalog(path).schema_version == 3
    two = json.dumps(_document()) + " " * (2 * 1024 * 1024)
    path.write_text(two, encoding="utf-8")
    with pytest.raises(CatalogError, match="exceeds"):
        load_catalog(path)
    path.write_text(json.dumps(_v3()) + " " * (5 * 1024 * 1024), encoding="utf-8")
    with pytest.raises(CatalogError, match="exceeds"):
        load_catalog(path)


# --- V1/V2/V10: the lookups ---------------------------------------------------------------------------------------


def _package(repo: str = "othercorp/Nova-7B-GGUF", file: str = "Nova-7B-Q8_0.gguf", quant: str | None = "Q8_0", revision: str = REVISION, **changes) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload.update(repo=repo, revision=revision, quantization=quant, base_model_hf_repo="acme/Nova-7B-2512")
    payload["files"] = [{"name": file, "role": "weights", "size_bytes": 5 * 1024**3, "digest": None}]
    payload.update(changes)
    return Package.model_validate(payload)


def _lookup_catalog(tmp_path: Path, checks: list[dict] | None = None) -> Catalog:
    return _catalog(_with_findings(checks=checks), tmp_path)


def test_a_package_gets_the_check_of_its_repository_file_and_revision(tmp_path):
    catalog = _lookup_catalog(tmp_path)
    check = catalog.package_check(_package())
    assert check is not None and (check.result, check.architecture) == ("absent", "nova7")


@pytest.mark.parametrize(
    "package",
    [
        _package(revision=OTHER_REVISION),  # the same package under another revision of the repository
        _package(repo="third/Nova-7B-GGUF"),  # the same file name in another repository
        _package(file="Nova-7B-Q4_K_M.gguf", quant="Q4_K_M"),  # another package of the same repository
        _package(file="sub/Nova-7B-Q8_0.gguf"),  # the same name in a folder is another identity
    ],
    ids=["other revision", "other repository", "other package", "other folder"],
)
def test_nothing_is_carried_to_another_revision_repository_or_package(package, tmp_path):
    assert _lookup_catalog(tmp_path).package_check(package) is None


def test_two_packages_of_one_repository_have_their_own_findings(tmp_path):
    other = _check(identity="Nova-7B-Q4_K_M.gguf", file="Nova-7B-Q4_K_M.gguf", quantization="Q4_K_M", result="present", architecture="nova7")
    catalog = _lookup_catalog(tmp_path, [_check(), other])
    assert catalog.package_check(_package()).result == "absent"
    assert catalog.package_check(_package(file="Nova-7B-Q4_K_M.gguf", quant="Q4_K_M")).result == "present"


def test_a_shard_set_is_found_by_its_identity(tmp_path):
    sharded = _check(identity="Q8_0/Nova-7B-Q8_0.gguf", file="Q8_0/Nova-7B-Q8_0-00001-of-00002.gguf")
    catalog = _lookup_catalog(tmp_path, [sharded])
    files = [
        {"name": f"Q8_0/Nova-7B-Q8_0-0000{i}-of-00002.gguf", "role": "weights_shard", "size_bytes": 1024**3, "digest": None} for i in (1, 2)
    ]
    assert catalog.package_check(_package(files=files, quantization="Q8_0")).result == "absent"


def test_an_ollama_package_and_a_tensor_package_have_no_check(tmp_path):
    catalog = _lookup_catalog(tmp_path)
    ollama = copy.deepcopy(EXAMPLES["Package"])
    ollama.update(source="ollama", repo=None, revision=None, ollama_name="nova:7b", manifest_digest="sha256:" + "a" * 64)
    assert catalog.package_check(Package.model_validate(ollama)) is None
    tensor = _package(file="model.safetensors", quant=None, format="tensor", provenance="unresolved", unresolved_reason="format")
    assert catalog.package_check(tensor) is None


def test_two_rows_that_contradict_each_other_leave_the_package_unknown(tmp_path):
    rows = [_row(0, gguf_checks=[_check()]), _row(1, gguf_checks=[_check(result="present")]), _row(2)]
    assert _catalog(_v3(rows), tmp_path).package_check(_package()) is None


def test_two_rows_that_agree_on_the_result_but_not_on_the_architecture_or_the_file_leave_it_unknown(tmp_path):
    for other in (_check(architecture="nova8"), _check(file="Nova-7B-Q8_0.gguf", identity="Nova-7B-Q8_0.gguf", quantization="Q8_0", llamacpp_commit="d" * 40, source_url=llamacpp_page_url("d" * 40))):
        rows = [_row(0, gguf_checks=[_check()]), _row(1, gguf_checks=[other]), _row(2)]
        found = _catalog(_v3(rows), tmp_path).package_check(_package())
        assert (found is None) == (other["architecture"] != "nova7")  # a different commit alone is no contradiction


def test_a_check_of_another_shard_set_does_not_reach_the_package(tmp_path):
    """The same stem: a set of two shards and a set of three are two packages."""
    sharded = _check(identity="Q8_0/Nova-7B-Q8_0.gguf", file="Q8_0/Nova-7B-Q8_0-00001-of-00002.gguf")
    catalog = _lookup_catalog(tmp_path, [sharded])

    def package(total: int) -> Package:
        files = [
            {"name": f"Q8_0/Nova-7B-Q8_0-{i:05d}-of-{total:05d}.gguf", "role": "weights_shard", "size_bytes": 1024**3, "digest": None}
            for i in range(1, total + 1)
        ]
        return _package(files=files, quantization="Q8_0")

    assert catalog.package_check(package(2)).result == "absent"
    assert catalog.package_check(package(3)) is None


def test_two_rows_that_agree_give_the_one_result(tmp_path):
    rows = [_row(0, gguf_checks=[_check()]), _row(1, gguf_checks=[_check()]), _row(2)]
    assert _catalog(_v3(rows), tmp_path).package_check(_package()).result == "absent"


# --- V10: a pull name, with no revision in it ----------------------------------------------------------------------


def test_a_pull_name_finds_the_check_of_its_repository_and_quantization_whatever_the_revision(tmp_path):
    catalog = _lookup_catalog(tmp_path)
    for name in ("hf.co/othercorp/Nova-7B-GGUF:Q8_0", "hf.co/othercorp/Nova-7B-GGUF:q8_0"):
        check = catalog.pull_check(name)
        assert check is not None and check.revision == REVISION and check.result == "absent"


@pytest.mark.parametrize(
    "name",
    [
        "hf.co/othercorp/Nova-7B-GGUF",  # no quantization: Ollama chooses, so nothing is shown to be this package
        "hf.co/othercorp/Nova-7B-GGUF:Q4_K_M",  # another package of the repository
        "hf.co/third/Nova-7B-GGUF:Q8_0",  # another repository
        "hf.co/othercorp/Nova-7B-GGUF:latest",  # not a quantization
        "nova:7b",  # the name of an Ollama package inherits no Hub finding through its base model
        "acme/nova:7b",
        "hf.co/othercorp/Nova-7B-GGUF:Q8_0/../x",
        "",
    ],
)
def test_a_name_that_cannot_be_assigned_to_a_package_is_unchecked(name, tmp_path):
    assert _lookup_catalog(tmp_path).pull_check(name) is None


def test_a_name_that_two_packages_answer_to_is_unchecked(tmp_path):
    twin = _check(identity="Q8_0/Nova-7B-Q8_0.gguf", file="Q8_0/Nova-7B-Q8_0-00001-of-00002.gguf")
    assert _lookup_catalog(tmp_path, [_check(), twin]).pull_check("hf.co/othercorp/Nova-7B-GGUF:Q8_0") is None


@pytest.mark.parametrize("held", [None, 2, 3])
def test_a_name_is_assigned_only_where_the_checked_state_shows_one_package_of_that_quantization(held, tmp_path):
    """The repository held two packages of `Q8_0` and only one was checked (or the state was not recorded)."""
    catalog = _lookup_catalog(tmp_path, [_check(same_quantization=held)])
    assert catalog.pull_check("hf.co/othercorp/Nova-7B-GGUF:Q8_0") is None
    assert catalog.package_check(_package()).result == "absent"  # the check still holds for its own package


def test_a_repository_id_in_another_case_finds_the_same_checks(tmp_path):
    """Hugging Face repository ids are case-insensitive (Codex b4): the stored `absent` must not become unknown.
    File names and revisions stay as they are."""
    catalog = _lookup_catalog(tmp_path)
    assert catalog.pull_check("hf.co/othercorp/nova-7b-gguf:Q8_0").result == "absent"
    assert catalog.pull_check("hf.co/OTHERCORP/Nova-7B-GGUF:q8_0").result == "absent"
    assert catalog.package_check(_package(repo="othercorp/nova-7b-gguf")).result == "absent"
    assert catalog.package_check(_package(repo="othercorp/nova-7b-gguf", file="nova-7b-q8_0.gguf")) is None


def test_two_checks_of_one_package_in_another_case_in_one_row_are_refused(tmp_path):
    with pytest.raises(CatalogError, match="twice"):
        _lookup_catalog(tmp_path, [_check(), _check(repo="othercorp/nova-7b-gguf")])


def test_the_note_of_a_model_is_found_and_bound_whatever_the_case_of_its_id(tmp_path):
    catalog = _catalog(_with_findings(note=_note(model="ACME/nova-7b-2512", source_url=f"https://huggingface.co/ACME/nova-7b-2512/blob/{REVISION}/README.md")), tmp_path)
    assert catalog.setup_note_for("acme/Nova-7B-2512").model == "ACME/nova-7b-2512"  # stored as written
    assert _lookup_catalog(tmp_path).setup_note_for("Acme/nova-7B-2512").package == "nova-inference"


def test_an_ollama_name_of_the_base_model_is_unchecked_even_when_the_base_model_has_checks(tmp_path):
    catalog = _lookup_catalog(tmp_path)
    row = catalog.model_for("acme/Nova-7B-2512")
    assert row.gguf_checks and f"{row.ollama_base}:{row.ollama_tag}" == "nova:7b"
    assert catalog.pull_check("nova:7b") is None


# --- V7: the note ---------------------------------------------------------------------------------------------------


def test_the_note_of_a_model_is_shown_only_where_instructions_were_found(tmp_path):
    assert _lookup_catalog(tmp_path).setup_note_for("acme/Nova-7B-2512").package == "nova-inference"
    none_found = _note(status="none_recognized", package=None)
    assert _catalog(_with_findings(note=none_found), tmp_path).setup_note_for("acme/Nova-7B-2512") is None
    assert _lookup_catalog(tmp_path).setup_note_for("acme/Nova-7B") is None
    assert _lookup_catalog(tmp_path).setup_note_for("unknown/Model") is None


# --- the new channel ----------------------------------------------------------------------------------------------


SHA = "3f5c1d9a7e" * 4
DATASET = "jm-vis/modelroom-catalog"
REVISION_URL = f"https://huggingface.co/api/datasets/{DATASET}/revision/main"


def _channel_url(channel, sha: str = SHA) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{sha}/{channel.file_name}"


def _hub(channel, body: dict, *, sha: str = SHA) -> dict:
    return {
        ("GET", REVISION_URL): Response(200, {}, json.dumps({"sha": sha}).encode("utf-8")),
        ("GET", _channel_url(channel, sha)): Response(200, {"X-Repo-Commit": sha}, json.dumps(body).encode("utf-8")),
    }


@pytest.fixture(autouse=True)
def _published_catalog_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELROOM_CATALOG", raising=False)


def test_the_old_channel_is_unchanged_and_stays_the_default_until_the_release_decides():
    assert (catalog_remote.REMOTE_FILE_NAME, catalog_remote.CACHE_FILE_NAME) == ("catalog-v2.json", "catalog-v2.cache.json")
    assert catalog_remote.DEFAULT_CHANNEL is catalog_remote.CHANNEL_V2
    assert (catalog_remote.CHANNEL_V2.file_name, catalog_remote.CHANNEL_V2.schema, catalog_remote.CHANNEL_V2.max_bytes) == ("catalog-v2.json", 2, CATALOG_MAX_BYTES)


def test_the_new_channel_has_its_own_file_cache_schema_and_size():
    channel = catalog_remote.CHANNEL_V3
    assert (channel.file_name, channel.cache_name, channel.schema, channel.max_bytes) == ("catalog-v3.json", "catalog-v3.cache.json", 3, 4 * 1024 * 1024)
    assert channel.cache_name != catalog_remote.CHANNEL_V2.cache_name  # release 1.0.0 never meets the cache of the new one


def test_the_new_channel_is_read_with_the_same_two_requests_when_no_redirect_stands_between_and_keeps_its_findings(tmp_path):
    body = _with_findings()
    body["published"] = (load_catalog().published + timedelta(days=3)).isoformat()
    transport = FixtureTransport(_hub(catalog_remote.CHANNEL_V3, body))
    choice = catalog_remote.current_catalog(transport, NOW, tmp_path / "cache", RequestBudget(150), channel=catalog_remote.CHANNEL_V3)
    assert choice.source == "published"
    assert [url for _method, url in transport.calls] == [REVISION_URL, _channel_url(catalog_remote.CHANNEL_V3)]
    assert choice.catalog.model_for("acme/Nova-7B-2512").gguf_checks[0].result == "absent"
    assert (tmp_path / "cache" / "catalog-v3.cache.json").exists() and not (tmp_path / "cache" / "catalog-v2.cache.json").exists()


def test_the_new_channel_refuses_a_schema_2_file_and_the_old_channel_refuses_a_schema_3_file(tmp_path):
    published = json.loads((FIXTURES / "catalog_v2_published.json").read_text(encoding="utf-8"))
    new = catalog_remote.current_catalog(
        FixtureTransport(_hub(catalog_remote.CHANNEL_V3, published)), NOW, tmp_path / "a", RequestBudget(150), channel=catalog_remote.CHANNEL_V3
    )
    assert new.source == "shipped" and "schema_version 2" in new.origin
    three = _v3()
    three["published"] = "2026-10-12"
    old = catalog_remote.current_catalog(FixtureTransport(_hub(catalog_remote.CHANNEL_V2, three)), NOW, tmp_path / "b", RequestBudget(150))
    assert old.source == "shipped" and "schema_version 3" in old.origin


def test_a_file_above_the_old_limit_is_read_on_the_new_channel_only(tmp_path):
    body = _v3()
    body["published"] = (load_catalog().published + timedelta(days=3)).isoformat()
    big = json.dumps(body) + " " * (2 * 1024 * 1024)
    mapping = _hub(catalog_remote.CHANNEL_V3, body)
    mapping[("GET", _channel_url(catalog_remote.CHANNEL_V3))] = Response(200, {"X-Repo-Commit": SHA}, big.encode("utf-8"))
    transport = FixtureTransport(mapping)
    choice = catalog_remote.current_catalog(transport, NOW, tmp_path / "c", RequestBudget(150), channel=catalog_remote.CHANNEL_V3)
    assert choice.source == "published"


def test_a_tampered_cache_of_the_new_channel_with_a_schema_2_catalog_is_not_taken(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    envelope = {
        "cache_version": 1, "checked_at": "2026-10-14T08:00:00Z", "sha": SHA, "last_error": None,
        "catalog": json.loads((FIXTURES / "catalog_v2_published.json").read_text(encoding="utf-8")),
    }
    (cache / "catalog-v3.cache.json").write_text(json.dumps(envelope), encoding="utf-8")

    def offline(method, url, headers=None, *, max_bytes=None):
        raise OSError("network is unreachable")

    choice = catalog_remote.current_catalog(offline, NOW, cache, RequestBudget(150), channel=catalog_remote.CHANNEL_V3)
    assert choice.source == "shipped"


# --- the offline reading of render and pull ---------------------------------------------------------------------------


def test_the_offline_catalog_is_the_shipped_one_where_nothing_is_stored(tmp_path):
    assert catalog_remote.offline_catalog(tmp_path / "none").published == load_catalog().published


def test_the_offline_catalog_takes_the_stored_published_one_when_it_is_newer_and_asks_nothing(tmp_path):
    body = _with_findings()
    body["published"] = (load_catalog().published + timedelta(days=3)).isoformat()
    cache = tmp_path / "cache"
    cache.mkdir()
    envelope = {"cache_version": 1, "checked_at": "2026-10-14T08:00:00Z", "sha": SHA, "last_error": None, "catalog": body}
    (cache / "catalog-v3.cache.json").write_text(json.dumps(envelope), encoding="utf-8")
    found = catalog_remote.offline_catalog(cache, channel=catalog_remote.CHANNEL_V3)
    assert found.model_for("acme/Nova-7B-2512").gguf_checks
    assert catalog_remote.offline_catalog(cache).published == load_catalog().published  # the old channel's cache is another file


def test_the_offline_catalog_honours_the_switch_for_the_shipped_catalog(tmp_path, monkeypatch):
    body = _with_findings()
    body["published"] = (load_catalog().published + timedelta(days=3)).isoformat()
    cache = tmp_path / "cache"
    cache.mkdir()
    envelope = {"cache_version": 1, "checked_at": "2026-10-14T08:00:00Z", "sha": SHA, "last_error": None, "catalog": body}
    (cache / "catalog-v3.cache.json").write_text(json.dumps(envelope), encoding="utf-8")
    monkeypatch.setenv("MODELROOM_CATALOG", "shipped")
    assert catalog_remote.offline_catalog(cache, channel=catalog_remote.CHANNEL_V3).published == load_catalog().published


def test_a_broken_cache_never_stops_the_offline_reading(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "catalog-v2.cache.json").write_text("{not json", encoding="utf-8")
    assert catalog_remote.offline_catalog(cache).published == load_catalog().published


def test_the_builders_dispatch_on_the_schema_of_the_content(tmp_path):
    assert build_catalog(json.dumps(_document()), Path("c.json")).schema_version == 2
    assert build_catalog(json.dumps(_v3()), Path("c.json")).schema_version == 3
    assert build_catalog_v3(json.dumps(_v3()), Path("c.json")).schema_version == 3
    with pytest.raises(SchemaVersionError):
        build_catalog_v3(json.dumps(_document()), Path("c.json"))
