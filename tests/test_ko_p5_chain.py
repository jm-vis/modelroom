"""AP9-KO P5, part 4: the whole chain of the brief ("Abnahme", P5 Kette) in one test module, on recorded answers only:

scanner proposal -> approval -> publication on all channels -> reading of the catalog by a guided run -> search ->
choice -> configuration -> fetch -> render -> the decision about the installation.

Nothing opens a socket. The model is the recorded `Qwen/Qwen3.5-9B` with its recorded GGUF repository of the guided-run
fixtures; the Hub answers of the scanner are synthetic after the real pattern (a GGUF file whose architecture is not in
the recorded llama.cpp source), so the one check that comes out is bound to a package the fetch of the guided run
really finds. A blocked choice starts no download; a package nobody blocked is pulled in the very same run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import fixture_ko_p4 as F
from fixture_support import (
    UNSLOTH_SHA,
    build_transport,
    guided_transport_mapping,
    offline_servers,
    pull_daemon,
    tags_with,
    windows_probes,
)
from test_catalog_publish import P, R
from test_catalog_scan import Hub, scanner
from test_guided_install import ANSWERS, FIRST_NAME, THIRD_NAME, _pull_calls
from test_ko_catalog_v3 import REVISION_URL
from test_ko_p5_channel import _old_reader

from modelroom import catalog_remote
from modelroom.catalog import CATALOG_FILE
from modelroom.cli import main
from modelroom.dialog import FileAsker
from modelroom.http import Response

ROW = "Qwen/Qwen3.5-9B"
REPO = "unsloth/Qwen3.5-9B-GGUF"
FIRST_FILE, FIRST_BYTES = "Qwen3.5-9B-Q8_0.gguf", 9_527_502_048  # the first row of the guided run at 8192 tokens (FIRST_NAME)
SCAN_DAY = datetime(2026, 10, 14, 9, 0, 0, tzinfo=timezone.utc)
PUBLISHED_DAY = "2026-10-20"  # later than the shipped catalog, so a published file wins the choice
DATASET_SHA = "5d" * 20
RUN = datetime(2026, 10, 21, 8, 0, 0, tzinfo=timezone.utc)
UNKNOWN_ARCH = "nova7"  # an identifier the recorded llama.cpp source does not hold
NO_PULL = "installation not offered"


@pytest.fixture(autouse=True)
def _folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("MODELROOM_CATALOG", raising=False)  # the suite pins the shipped catalog; this chain reads the published one


# --- stage 1: the scanner proposes -------------------------------------------------------------------------------------


def _working_catalog(folder: Path) -> tuple[Path, Path]:
    """A working catalog of schema 3 with the one model of the chain (its GGUF repository the listed packager's), and an empty rejection list."""
    shipped = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    row = next(m for m in shipped["models"] if m["hf_repo"] == ROW)
    row = {**row, "gguf_repos": [REPO], "latest": False, "latest_source": None, "latest_checked": None}
    document = {
        **shipped, "schema_version": 3, "models": [row], "packagers": ["unsloth"],
        "publishers": [p for p in shipped["publishers"] if p["account"] == "Qwen"],
    }
    folder.mkdir(parents=True, exist_ok=True)
    catalog, rejected = folder / "catalog.json", folder / "rejected.json"
    catalog.write_bytes(R.dump_json(document))
    rejected.write_bytes(b"[]\n")
    return catalog, rejected


def _hub() -> Hub:
    hub = Hub()
    for account in ("Qwen", "unsloth"):
        hub.page(account, [])
    F.source(hub)
    F.listing(hub, ROW, [F.entry(REPO, ROW, UNSLOTH_SHA, architecture=UNKNOWN_ARCH, total=FIRST_BYTES)])
    F.tree(hub, REPO, UNSLOTH_SHA, [(FIRST_FILE, FIRST_BYTES)])
    F.card(hub, ROW, "# A card\n\nNothing to set up here.\n")
    return hub


def _scan(folder: Path) -> tuple[dict, Path, Path]:
    catalog, rejected = _working_catalog(folder)
    result = scanner.scan(catalog, rejected, folder / "out", transport=F.Recording(_hub().m), now=SCAN_DAY, budget=300, sub_budget=30, packages=True)
    assert result.exit_code == 0, result.error
    proposal = json.loads((folder / "out" / "2026-10-14" / "proposal.json").read_bytes())
    return proposal, catalog, rejected


def _check_change(proposal: dict) -> dict:
    [change] = [c for c in proposal["changes"] if c["change"] == "field" and c["field"] == "gguf_checks" and c["hf_repo"] == ROW]
    return change


def test_the_scanner_proposes_an_absent_check_bound_to_the_package_of_the_fetch(tmp_path):
    proposal, _catalog, _rejected = _scan(tmp_path)

    [check] = _check_change(proposal)["fields"]["gguf_checks"]["new"]

    assert (check["repo"], check["file"], check["result"], check["architecture"]) == (REPO, FIRST_FILE, "absent", UNKNOWN_ARCH)
    assert check["revision"] == UNSLOTH_SHA and check["checked"] == "2026-10-14" and check["llamacpp_commit"] == F.LLAMA_COMMIT


# --- stage 2 and 3: approval, publication on both channels --------------------------------------------------------------


def _approved(folder: Path) -> tuple[bytes, dict[str, bytes]]:
    """The table after the maintainer approved the number of the check, and the two files one publication makes of it."""
    proposal, catalog, rejected = _scan(folder)
    number = _check_change(proposal)["n"]
    plan = P.plan_apply(json.loads(catalog.read_bytes()), catalog.read_bytes(), rejected.read_bytes(), proposal, P.parse_request(str(number), None, None), PUBLISHED_DAY)
    table = plan.catalog_text.encode("utf-8") if isinstance(plan.catalog_text, str) else plan.catalog_text
    return table, P.channel_files(table)


def test_the_approved_table_stays_schema_3_and_makes_the_three_channel_files_of_one_publication(tmp_path):
    table, files = _approved(tmp_path)

    document = json.loads(table)
    assert document["schema_version"] == 3 and document["published"] == PUBLISHED_DAY
    assert document["models"][0]["gguf_checks"][0]["result"] == "absent"
    old = json.loads(files["catalog-v2.json"])
    assert old["schema_version"] == 2 and "gguf_checks" not in old["models"][0]  # what release 1.0.0 reads: no finding in it
    assert files["catalog-v3.json"] == table  # what release 1.1.0 reads
    assert json.loads(files["catalog-v4.json"]) == {**document, "schema_version": 4}  # what release 1.2.0 reads


def test_release_1_0_0_reads_the_old_file_of_that_publication_and_never_the_new_one(tmp_path):
    _table, files = _approved(tmp_path)

    reader = tmp_path / "reader"
    reader.mkdir()
    answer = _old_reader(reader, json.loads(files["catalog-v2.json"]), json.loads(files["catalog-v3.json"]))

    assert answer["calls"] == ["main", "catalog-v2.json"] and (answer["source"], answer["schema"]) == ("published", 2)
    assert answer["published"] == PUBLISHED_DAY


# --- stage 4 to 9: the guided run reads the catalog and decides the installation ------------------------------------------


def _dataset(files: dict[str, bytes]) -> dict[tuple[str, str], Response]:
    """The dataset of the publication as the guided run meets it: its revision, and the newest channel's file at that commit."""
    base = "https://huggingface.co/datasets/jm-vis/modelroom-catalog/resolve"
    return {
        ("GET", REVISION_URL): Response(200, {}, json.dumps({"sha": DATASET_SHA}).encode("utf-8")),
        ("GET", f"{base}/{DATASET_SHA}/catalog-v4.json"): Response(200, {"X-Repo-Commit": DATASET_SHA}, files["catalog-v4.json"]),
    }


def _guided(tmp_path: Path, files: dict[str, bytes], answers: dict, daemon, capsys) -> tuple[int, list[str], str]:
    lines: list[str] = []
    transport = build_transport({**guided_transport_mapping(), **_dataset(files)})
    code = main(
        [], transport=transport, probes=windows_probes(), pointer_path=tmp_path / "home" / ".modelroom" / "guided.json", now=RUN,
        asker=FileAsker(answers), here=tmp_path / "results", out=lines.append, daemon=daemon, servers=offline_servers(),
    )
    return code, [line.strip() for line in lines], capsys.readouterr().err


def _models(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "results" / "docs" / "models.json").read_text(encoding="utf-8"))


def _row(payload: dict, quantization: str) -> dict:
    return next(row for row in payload["machines"][0]["ranked"] if row["package_identity"][1] == REPO and row["quantization"] == quantization)


def test_the_run_reads_the_published_catalog_and_the_blocked_package_reaches_the_table(tmp_path, capsys):
    _table, files = _approved(tmp_path / "scan")
    daemon = pull_daemon()

    code, lines, _err = _guided(tmp_path, files, {**ANSWERS, "pull": False}, daemon, capsys)

    assert code == 0
    assert any(line == f"catalog: published {PUBLISHED_DAY}" for line in lines)  # read through the default channel (catalog-v4.json)
    payload = _models(tmp_path)
    blocked = _row(payload, "Q8_0")
    assert blocked["load_check"]["result"] == "absent" and blocked["load_check"]["architecture"] == UNKNOWN_ARCH
    assert "load_check" not in _row(payload, "Q6_K")  # the finding is bound to its package, not to its repository
    text = "\n".join(lines)
    assert f"architecture {UNKNOWN_ARCH} is absent from the checked llama.cpp source (checked 2026-10-14)" in text
    config = (tmp_path / "results" / "modelroom.toml").read_text(encoding="utf-8")
    assert ROW in config  # search -> choice -> configuration
    assert _pull_calls(daemon) == []


def test_choosing_the_blocked_package_by_name_ends_the_run_and_downloads_nothing(tmp_path, capsys):
    _table, files = _approved(tmp_path / "scan")
    daemon = pull_daemon(after_pull=tags_with(FIRST_NAME))

    code, _lines, err = _guided(tmp_path, files, {**ANSWERS, "pull_package": FIRST_NAME, "pull": True}, daemon, capsys)

    assert code == 2 and "pull_package" in err
    assert _pull_calls(daemon) == []


def test_without_a_choice_the_run_does_not_move_to_another_row_and_pulls_nothing(tmp_path, capsys):
    """An answer file without `pull_package` takes row #1 or nothing; row #1 is the blocked package."""
    _table, files = _approved(tmp_path / "scan")
    daemon = pull_daemon(after_pull=tags_with(FIRST_NAME))

    code, lines, _err = _guided(tmp_path, files, {**ANSWERS, "pull": True}, daemon, capsys)

    assert _pull_calls(daemon) == []
    assert code == 0
    assert not any(line.startswith("✓ Pulled") for line in lines)
    assert any(NO_PULL in line for line in lines), lines[-12:]


def test_a_package_nobody_blocked_is_pulled_in_the_same_run(tmp_path, capsys):
    _table, files = _approved(tmp_path / "scan")
    daemon = pull_daemon(after_pull=tags_with(THIRD_NAME))

    code, lines, _err = _guided(tmp_path, files, {**ANSWERS, "pull_package": THIRD_NAME, "pull": True}, daemon, capsys)

    assert code == 0
    assert _pull_calls(daemon)  # the same catalog, the same repository: only the package with the finding is locked
    assert any(line.startswith(f"✓ Pulled    {THIRD_NAME}") for line in lines)


def test_the_run_falls_back_to_the_shipped_catalog_where_the_dataset_has_no_new_file_yet(tmp_path, capsys):
    """The order of the release: the dataset's `catalog-v4.json` first, then the release. A release that meets a dataset
    without that file reads its shipped catalog, says so, and the run goes on (nothing is blocked, nothing breaks)."""
    _table, files = _approved(tmp_path / "scan")
    dataset = _dataset(files)
    for key in [key for key in dataset if key[1].endswith("catalog-v4.json")]:
        dataset[key] = Response(404, {}, b"{}")
    lines: list[str] = []
    code = main(
        [], transport=build_transport({**guided_transport_mapping(), **dataset}), probes=windows_probes(),
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json", now=RUN, asker=FileAsker({**ANSWERS, "pull": False}),
        here=tmp_path / "results", out=lines.append, daemon=pull_daemon(), servers=offline_servers(),
    )

    assert code == 0
    assert any("the published one did not read: HTTP 404" in line for line in lines)
    assert "load_check" not in json.dumps(_models(tmp_path))


def test_the_default_channel_of_the_chain_is_the_newest_one():
    assert catalog_remote.DEFAULT_CHANNEL.file_name == "catalog-v4.json"
