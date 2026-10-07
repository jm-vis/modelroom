"""Tests for `scripts/catalog_scan.py`: the scanner that proposes changes to the catalog and never makes one (AP9-KA-T).

Every request goes through a `FixtureTransport` that holds recorded answers in the shape of the Hub's API
(`tests/fixtures/scan_*.json`, accounts and models invented); nothing here opens a socket, reads a token or
touches the real catalog. A call for a URL nobody recorded is a `KeyError`, so a test that must not ask
something fails by recording nothing for it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the scanner is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))
rules = importlib.import_module("catalog_scan_rules")
scanner = importlib.import_module("catalog_scan")

from modelroom.catalog import CatalogModel  # noqa: E402
from modelroom.catalog_schema import build_catalog_v2  # noqa: E402
from modelroom.http import FixtureTransport, Response  # noqa: E402

FIX = REPO / "tests" / "fixtures"
NOW = datetime(2026, 10, 4, 9, 0, 0, tzinfo=timezone.utc)
TODAY, YESTERDAY = "2026-10-04", "2026-10-03"
CATALOG_BYTES = (FIX / "scan_catalog.json").read_bytes()
PUBLISHERS, PACKAGERS = ["acme", "nova", "orbit"], ["packone", "packtwo"]
LATEST_TEXT_VISION = ["acme/Atlas-2-9B", "nova/Nimbus-1-30B", "orbit/Comet-3-8B"]
TABLE = {  # the fifteen tasks and their kinds, word for word from the brief
    "text-generation": "text", "image-text-to-text": "vision", "feature-extraction": "embedding",
    "sentence-similarity": "embedding", "automatic-speech-recognition": "speech-recognition",
    "text-to-speech": "speech-synthesis", "text-to-audio": "audio", "audio-to-audio": "audio",
    "audio-text-to-text": "audio", "text-to-image": "image-generation", "image-to-image": "image-generation",
    "text-to-video": "video-generation", "image-to-video": "video-generation",
    "image-text-to-video": "video-generation", "video-to-video": "video-generation",
}
BASELINE_CALLS = len(PUBLISHERS) + len(PACKAGERS) + 15 + 3 + 2  # lists, B1 pages, B2 pages, two collection pages
MISSING = object()


def stamp(days: float = 0, base: datetime = NOW) -> str:
    return (base - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def ent(repo: str, created: str | None = None, **over) -> dict:
    """One entry of a list answer: the fields the scanner asks for, in the Hub's shape."""
    data = {
        "_id": "6710a1b2c3d4e5f607182930", "id": repo, "createdAt": created or stamp(1), "likes": 12, "downloads": 340,
        "pipeline_tag": "text-generation", "tags": ["safetensors", "region:us"], "cardData": {"license": "apache-2.0"},
        "safetensors": {"parameters": {"BF16": 7_000_000_000}, "total": 7_000_000_000},
    }
    data.update(over)
    return {key: value for key, value in data.items() if value is not MISSING}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Hub:
    """The recorded Hub: every request of a quiet run (empty answers) plus what a test puts on top."""

    def __init__(self) -> None:
        self.m: dict[tuple[str, str], Response] = {}
        for account in PUBLISHERS + PACKAGERS:
            self.page(account, [])
        for tag in TABLE:
            self.put(rules.b1_url(tag), [])
        for repo in LATEST_TEXT_VISION:
            self.put(rules.b2_url(repo), [])
        for owner in ("acme", "orbit"):  # the two publishers whose latest rows cite a plain publisher page
            self.put(rules.collections_url(owner), [])

    def put(self, url: str, body, status: int = 200, headers: dict | None = None) -> None:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.m[("GET", url)] = Response(status=status, headers=headers or {}, body=raw)

    def page(self, account: str, entries: list, **kw) -> None:
        self.put(rules.list_url(account), entries, **kw)

    def chain(self, account: str, pages: list[list], url_of=None) -> None:
        """A list of pages, each linking to the next by `Link: rel="next"`; the first sits on the account's first URL."""
        url_of = url_of or (lambda n: f"https://huggingface.co/api/models?author={account}&cursor=p{n}&direction=-1&limit=100&sort=createdAt")
        for index, entries in enumerate(pages, start=1):
            url = rules.list_url(account) if index == 1 else url_of(index)
            link = {"Link": f'<{url_of(index + 1)}>; rel="next"'} if index < len(pages) else {}
            self.put(url, entries, headers=link)

    def config(self, repo: str, body, redirect: bool = False) -> None:
        if redirect:
            where = f"https://huggingface.co/api/resolve-cache/models/{repo}/0123abcd/config.json"
            self.put(rules.config_url(repo), b"", status=307, headers={"Location": where})
            self.put(where, body)
        else:
            self.put(rules.config_url(repo), body)


class Env:
    def __init__(self, root: Path) -> None:
        self.root, self.out = root, root / "out"
        root.mkdir(parents=True, exist_ok=True)
        self.catalog, self.rejected = root / "catalog.json", root / "rejected.json"
        self.catalog.write_bytes(CATALOG_BYTES)
        self.rejected.write_bytes((FIX / "scan_rejected.json").read_bytes())

    def run(self, hub: Hub, *, now: datetime = NOW, budget: int = 300, sub: int = 30, files=None):
        transport = FixtureTransport(hub.m)
        result = scanner.scan(self.catalog, self.rejected, self.out, transport=transport, now=now, budget=budget, sub_budget=sub, files=files)
        return result, transport

    def proposal(self, day: str = TODAY) -> dict:
        return json.loads((self.out / day / "proposal.json").read_bytes())

    def state(self) -> dict:
        return json.loads((self.out / "scan-state.json").read_bytes())

    def put_state(self, accounts: dict) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "scan-state.json").write_bytes(json.dumps({"schema": 1, "accounts": accounts}).encode("utf-8"))

    def prev(self, day: str, changes=(), candidates=(), decision: dict | None = None) -> None:
        folder = self.out / day
        folder.mkdir(parents=True, exist_ok=True)
        data = {"schema": 1, "date": day, "base_catalog_sha256": digest(CATALOG_BYTES), "notes": [], "summary": {},
                "changes": list(changes), "candidates": list(candidates)}
        (folder / "proposal.json").write_bytes(json.dumps(data).encode("utf-8"))
        if decision is not None:
            (folder / "decision.json").write_bytes(json.dumps(decision).encode("utf-8"))


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def go(env: Env, hub: Hub, **kw):
    """Run once; the proposal as written, the result and the transport."""
    result, transport = env.run(hub, **kw)
    return env.proposal((kw.get("now") or NOW).date().isoformat()), result, transport


def new_rows(p: dict) -> dict:
    return {c["hf_repo"]: c for c in p["changes"] if c["change"] == "new_row"}


def new_of(change: dict) -> dict:
    return {name: pair["new"] for name, pair in change["fields"].items()}


def fields_of(p: dict, repo: str, field: str) -> dict | None:
    return next((c for c in p["changes"] if c["change"] == "field" and c["hf_repo"] == repo and c["field"] == field), None)


def cands(p: dict, kind: str | None = None) -> list[dict]:
    return [c for c in p["candidates"] if kind is None or c["candidate"] == kind]


def asked(transport: FixtureTransport, *needles: str) -> list[str]:
    return [url for _, url in transport.calls if all(n in url for n in needles)]


def stage_calls(transport: FixtureTransport) -> int:
    return len([u for _, u in transport.calls if "/resolve" in u or "/resolve-cache/" in u or u.startswith("https://huggingface.co/api/models/")])


def b3(repo: str, open_since: str, *, budget: bool = False, tags=(), pipeline=None) -> dict:
    return {
        "k": "K1", "candidate": "B3", "hf_repo": repo, "account": repo.split("/")[0], "likes": 0, "downloads": 0,
        "created_at": stamp(14), "license": None, "parameters_b": None, "hub_pipeline_tag": pipeline, "tags": list(tags),
        "kind_evidence": {"stage": "budget", "value": "request budget ended"} if budget else {"stage": "config", "value": "no decisive evidence"},
        "open_since": open_since, "evidence": [f"https://huggingface.co/{repo}"],
    }


def applied(p: dict):
    """The base catalog with the proposal's changes applied, read back through the schema-2 validator."""
    return build_catalog_v2(json.dumps(rules.apply_changes(json.loads(CATALOG_BYTES), p["changes"])), Path("catalog.json"))


def b1_entry(account: str, likes: int = 120, downloads: int = 1500, age: timedelta = timedelta(days=3), **over) -> dict:
    return ent(f"{account}/Rising-1-7B", (NOW - age).strftime("%Y-%m-%dT%H:%M:%S.000Z"), likes=likes, downloads=downloads, **over)


# --- A1: a new base model of a listed publisher ------------------------------------------------------------------


def test_a_new_base_model_of_a_listed_publisher_becomes_a_complete_row_that_validates(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2), safetensors={"total": 9_100_000_000}, likes=57)])
    p, result, _ = go(env, hub)
    row = new_rows(p)["acme/Atlas-3-9B"]
    assert (row["n"], row["stage"], row["kind"], row["open_since"]) == (1, "A1", "text", TODAY)
    assert set(row["fields"]) == set(CatalogModel.model_fields) and all(pair["old"] is None for pair in row["fields"].values())
    new = new_of(row)
    assert (new["family"], new["publisher"], new["kind"], new["hub_pipeline_tag"], new["kind_source"]) == ("atlas", "acme", "text", "text-generation", None)
    assert (new["added"], new["parameters_b"], new["license"], new["hub_created"]) == (TODAY, 9.1, "apache-2.0", stamp(2)[:10])
    assert new["source"] == "https://huggingface.co/acme/Atlas-3-9B" and new["gguf_repos"] == [] and new["successor"] is None
    assert row["evidence"] == ["https://huggingface.co/acme/Atlas-3-9B"] and row["kind_evidence"] == {"stage": "pipeline_tag", "value": "text-generation"}
    assert p["summary"] == {"new": 1, "changed": 0, "candidates": 0, "incomplete": 0, "pending_publication": None}
    assert result.exit_code == 0 and applied(p).model_for("acme/Atlas-3-9B") is not None


def test_the_new_model_that_stands_over_the_latest_row_of_its_group_is_proposed_latest_without_a_successor_for_the_old_row(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2))])
    p, _, _ = go(env, hub)
    new = new_of(new_rows(p)["acme/Atlas-3-9B"])
    assert (new["latest"], new["latest_source"], new["latest_checked"]) == (True, "https://huggingface.co/acme", TODAY)
    assert not [c for c in p["changes"] if c["hf_repo"] == "acme/Atlas-2-9B"]


def test_a_new_model_in_a_group_without_a_latest_row_is_not_latest_and_has_no_latest_source(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(2))])
    new = new_of(new_rows(go(env, hub)[0])["acme/Zeta-1-3B"])
    assert (new["latest"], new["latest_source"], new["latest_checked"]) == (False, None, None)


def test_an_embedding_model_a_huge_model_and_one_with_no_size_all_become_rows_with_null_where_the_size_is_unknown(env):
    hub = Hub()
    hub.page("acme", [
        ent("acme/Zeta-Embed-1", stamp(1), pipeline_tag="feature-extraction", safetensors=MISSING),
        ent("acme/Giant-1-1000B", stamp(2), safetensors={"total": 1_000_000_000_000}),
    ])
    rows = new_rows(go(env, hub)[0])
    assert (rows["acme/Zeta-Embed-1"]["kind"], new_of(rows["acme/Zeta-Embed-1"])["parameters_b"]) == ("embedding", None)
    assert new_of(rows["acme/Giant-1-1000B"])["parameters_b"] == 1000.0


def test_a_repo_with_a_foreign_base_or_on_the_rejection_list_is_not_proposed_and_an_own_base_is(env):
    hub = Hub()
    hub.page("acme", [
        ent("acme/Zeta-1-3B", stamp(1), tags=["base_model:other/Thing-7B", "base_model:finetune:other/Thing-7B"]),
        ent("acme/Rejected-1-7B", stamp(1)),
        ent("acme/Zeta-2-3B", stamp(1), tags=["base_model:acme/Zeta-1-3B-Base", "base_model:finetune:acme/Zeta-1-3B-Base"]),
    ])
    assert list(new_rows(go(env, hub)[0])) == ["acme/Zeta-2-3B"]


def test_a_repo_of_a_rejected_account_is_not_proposed_by_the_open_pages(env):
    hub = Hub()
    hub.put(rules.b1_url("text-generation"), [ent("spamlabs/Hot-1-7B", stamp(1), likes=900, downloads=90_000)])
    assert cands(go(env, hub)[0]) == []


def test_a_speech_model_of_a_family_with_a_latest_text_row_neither_becomes_latest_nor_makes_a_successor(env):
    hub = Hub()
    hub.page("acme", [
        ent("acme/Atlas-Voice-3", stamp(2), pipeline_tag="text-to-speech"),
        ent("acme/Atlas-2-9B", stamp(5), cardData={"license": "apache-2.0", "new_version": "acme/Atlas-Voice-3"}),
    ])
    p, _, _ = go(env, hub)
    assert new_of(new_rows(p)["acme/Atlas-Voice-3"])["latest"] is False
    assert [c for c in p["changes"] if c["hf_repo"] == "acme/Atlas-2-9B"] == []


def test_the_open_pages_ask_fifteen_tasks_and_only_the_latest_text_and_vision_rows_of_the_table(env):
    p, _, transport = go(env, Hub())
    assert len(asked(transport, "sort=trendingScore")) == 15 and len(transport.calls) == BASELINE_CALLS
    assert sorted(asked(transport, "filter=base_model")) == sorted(rules.b2_url(r) for r in LATEST_TEXT_VISION)


def test_the_task_comes_from_the_top_field_even_when_the_card_says_something_else(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag="text-generation", cardData={"pipeline_tag": "image-text-to-text"})])
    assert new_of(new_rows(go(env, hub)[0])["acme/Zeta-1-3B"])["hub_pipeline_tag"] == "text-generation"


# --- the rest of stage A and stage B ----------------------------------------------------------------------------------


def quantized(repo: str, base: str, relation: str = "quantized", tags_without: str = "", **over) -> dict:
    tags = [t for t in ("gguf", f"base_model:{base}", f"base_model:{relation}:{base}") if t != tags_without]
    return ent(repo, stamp(1), tags=tags, cardData={"base_model": [base]}, pipeline_tag="text-generation", **over)


def test_a_gguf_repo_of_a_listed_packager_or_the_publisher_that_quantizes_exactly_this_row_is_proposed_for_its_list(env):
    hub = Hub()
    hub.page("packone", [quantized("packone/Atlas-2-9B-GGUF", "acme/Atlas-2-9B"), quantized("packone/Atlas-2-9B-ft-GGUF", "acme/Atlas-2-9B", "finetune"),
                         quantized("packone/Other-GGUF", "other/Thing-7B")])
    hub.page("acme", [quantized("acme/Atlas-2-9B-GGUF", "acme/Atlas-2-9B")])
    p, _, _ = go(env, hub)
    change = fields_of(p, "acme/Atlas-2-9B", "gguf_repos")
    assert change["stage"] == "A2" and change["fields"] == {"gguf_repos": {"old": [], "new": ["acme/Atlas-2-9B-GGUF", "packone/Atlas-2-9B-GGUF"]}}
    assert applied(p).model_for("acme/Atlas-2-9B").gguf_repos == ["acme/Atlas-2-9B-GGUF", "packone/Atlas-2-9B-GGUF"]
    assert [c["hf_repo"] for c in p["changes"]] == ["acme/Atlas-2-9B"]  # a quantization is no row of its own


def test_a_gguf_repo_already_in_the_list_and_one_of_an_unlisted_owner_are_no_change(env):
    raw = json.loads(CATALOG_BYTES)
    raw["models"][1]["gguf_repos"] = ["packone/Atlas-2-9B-GGUF"]
    env.catalog.write_bytes(json.dumps(raw).encode("utf-8"))
    hub = Hub()
    hub.page("packone", [quantized("packone/Atlas-2-9B-GGUF", "acme/Atlas-2-9B")])
    hub.page("orbit", [quantized("orbit/Atlas-2-9B-GGUF", "acme/Atlas-2-9B")])
    assert go(env, hub)[0]["changes"] == []


def test_a_new_version_on_the_card_of_a_row_proposes_a_successor_with_the_card_as_evidence(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-1-9B", stamp(1), cardData={"license": "mit", "new_version": "acme/Atlas-2-9B"})])
    p, _, _ = go(env, hub)
    change = fields_of(p, "acme/Atlas-1-9B", "successor")
    assert change["stage"] == "A3" and change["evidence"] == ["https://huggingface.co/acme/Atlas-1-9B"]
    assert {k: v["new"] for k, v in change["fields"].items()} == {
        "successor": "acme/Atlas-2-9B", "successor_source": "https://huggingface.co/acme/Atlas-1-9B", "successor_checked": TODAY}
    assert applied(p).age_of("acme/Atlas-1-9B") == ("legacy", "acme/Atlas-2-9B")


def test_a_new_row_that_replaces_a_latest_row_comes_as_a_pair_when_the_old_card_names_it(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2)), ent("acme/Atlas-2-9B", stamp(5), cardData={"license": "mit", "new_version": "acme/Atlas-3-9B"})])
    p, _, _ = go(env, hub)
    change = fields_of(p, "acme/Atlas-2-9B", "successor")
    assert new_of(change)["successor"] == "acme/Atlas-3-9B" and new_of(change)["latest"] is False
    assert (new_of(change)["latest_source"], new_of(change)["latest_checked"]) == (None, None) and change["fields"]["latest"]["old"] is True
    assert new_of(new_rows(p)["acme/Atlas-3-9B"])["latest"] is True
    assert applied(p).age_of("acme/Atlas-2-9B") == ("legacy", "acme/Atlas-3-9B")


def test_a_collection_of_the_publisher_that_holds_a_model_proposes_the_latest_source_and_is_asked_once_per_account(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2)), ent("acme/Atlas-2-9B", stamp(5))])
    hub.put(rules.collections_url("acme"), (FIX / "scan_collections.json").read_bytes())
    p, _, transport = go(env, hub)
    url = "https://huggingface.co/collections/acme/atlas-6a1b2c3d4e5f"
    change = fields_of(p, "acme/Atlas-2-9B", "latest_source")
    assert change["stage"] == "A3" and change["fields"] == {"latest_source": {"old": "https://huggingface.co/acme", "new": url}}
    assert new_of(new_rows(p)["acme/Atlas-3-9B"])["latest_source"] == url
    assert asked(transport, "/api/collections") == [rules.collections_url("acme"), rules.collections_url("orbit")] and applied(p)


def test_a_card_that_names_a_row_without_a_new_version_gives_a_note_and_never_a_successor(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2), cardData={"license": "mit", "description": "The big sibling of acme/Atlas-2-9B."})])
    p, _, _ = go(env, hub)
    assert any(n.startswith("card mentions acme/Atlas-2-9B") for n in p["notes"])
    assert fields_of(p, "acme/Atlas-2-9B", "successor") is None and fields_of(p, "acme/Atlas-2-9B", "latest") is None


@pytest.mark.parametrize("target", ["orbit/Comet-3-8B", "acme/Atlas-1-9B"])
def test_a_new_version_of_another_account_or_the_row_itself_gives_a_note_and_no_successor(env, target):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-1-9B", stamp(1), cardData={"license": "mit", "new_version": target})])
    p, _, _ = go(env, hub)
    assert p["changes"] == [] and any(f"new_version {target}" in n for n in p["notes"])


def test_a_repo_of_a_quantization_relation_without_the_gguf_tag_is_no_gguf_package(env):
    hub = Hub()
    hub.page("packone", [quantized("packone/Atlas-2-9B-FP8", "acme/Atlas-2-9B", tags_without="gguf")])
    assert go(env, hub)[0]["changes"] == []


def test_a_new_row_takes_the_family_its_group_has_in_the_table_and_avoids_a_family_of_another_publisher(env):
    hub = Hub()
    hub.page("orbit", [ent("orbit/Comet-4-8B", stamp(1))])
    hub.page("nova", [ent("nova/Atlas-1-1B", stamp(1))])
    rows = new_rows(go(env, hub)[0])
    assert new_of(rows["orbit/Comet-4-8B"])["family"] == "comet-core"  # the family of Comet-3-8B, not the one the name gives
    assert new_of(rows["nova/Atlas-1-1B"])["family"] == "atlas-nova"  # `atlas` is a family of acme in this table


def test_a_new_version_that_names_no_model_of_the_table_gives_a_note_only(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-1-9B", stamp(1), cardData={"license": "mit", "new_version": "acme/Nowhere-9B"})])
    p, _, _ = go(env, hub)
    assert p["changes"] == [] and any("new_version acme/Nowhere-9B" in n for n in p["notes"])


# --- the check of the existing rows -----------------------------------------------------------------------------------


def test_the_check_of_existing_rows_uses_the_lists_of_the_run_only_and_names_only_a_differing_row(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-1-9B", stamp(1)), ent("acme/Atlas-2-9B", stamp(1), pipeline_tag="text-to-image")])
    hub.page("nova", [ent("nova/Nimbus-1-30B", stamp(1), pipeline_tag=None, tags=[])])
    before = env.catalog.read_bytes()
    p, result, transport = go(env, hub)
    assert result.kind_check == {
        "acme/Atlas-1-9B": "confirmed", "acme/Atlas-2-9B": "differs", "nova/Nimbus-1-30B": "not checked",
        "orbit/Comet-3-8B": "not checked", "orbit/Comet-Embed-1": "not checked", "orbit/Comet-Echo-1": "not checked",
    }
    assert [n for n in p["notes"] if "differs" in n] == [
        "acme/Atlas-2-9B: kind differs from the table (table text, hub text-to-image); the row is not changed"]
    assert p["changes"] == [] and asked(transport, "/api/models/") == [] and asked(transport, "config.json") == []
    assert env.catalog.read_bytes() == before


# --- coverage: pages, marks, budget ------------------------------------------------------------------------------------


def test_a_list_of_three_pages_stops_in_the_page_where_the_mark_is_reached_and_moves_the_mark(env):
    env.put_state({"acme": {"covered_until": "2026-09-30T00:00:00.000Z", "pending": None}})
    hub = Hub()
    hub.chain("acme", [[ent("acme/N1-1B", stamp(1)), ent("acme/N2-1B", stamp(2))], [ent("acme/N3-1B", "2026-09-29T00:00:00.000Z"), ent("acme/N4-1B", "2026-09-20T00:00:00.000Z")], [ent("acme/N5-1B", stamp(40))]])
    p, _, transport = go(env, hub)
    assert len(asked(transport, "author=acme")) == 2 and sorted(new_rows(p)) == ["acme/N1-1B", "acme/N2-1B", "acme/N3-1B"]  # N4 is on the page but before the threshold: no row
    assert env.state()["accounts"]["acme"] == {"covered_until": stamp(1), "pending": None}


def test_a_page_is_read_on_until_an_entry_is_two_days_older_than_the_mark(env):
    env.put_state({"acme": {"covered_until": "2026-09-30T00:00:00.000Z", "pending": None}})
    hub = Hub()
    hub.chain("acme", [[ent("acme/N1-1B", stamp(1)), ent("acme/N2-1B", "2026-09-29T00:00:00.000Z")], [ent("acme/N3-1B", "2026-09-27T23:59:59.000Z"), ent("acme/N4-1B", "2026-09-01T00:00:00.000Z")], [ent("acme/N5-1B", "2026-08-01T00:00:00.000Z")]])
    p, _, transport = go(env, hub)
    assert len(asked(transport, "author=acme")) == 2 and sorted(new_rows(p)) == ["acme/N1-1B", "acme/N2-1B"]  # the second page is read: the first holds nothing older than 2026-09-28; N3 on it is before the threshold, so no row


MARK_30_09 = {"covered_until": "2026-09-30T00:00:00.000Z", "pending": None}  # the threshold of the account is 2026-09-28
UNDECIDED = {"pipeline_tag": None, "tags": []}  # no cheap kind: deciding it would ask for `config.json`


def test_an_entry_before_the_threshold_starts_no_card_request_and_no_row_while_a_younger_one_does(env):
    env.put_state({"acme": MARK_30_09})
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-4-9B", stamp(1)), ent("acme/Atlas-0-9B", "2026-09-20T00:00:00.000Z", **UNDECIDED)])
    hub.config("acme/Atlas-0-9B", {})
    p, _, transport = go(env, hub)
    assert list(new_rows(p)) == ["acme/Atlas-4-9B"] and cands(p, "B3") == []
    assert asked(transport, "Atlas-0-9B") == [] and stage_calls(transport) == 0


def test_a_row_of_the_table_before_the_threshold_is_still_checked_on_its_card(env):
    env.put_state({"acme": MARK_30_09})
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-4-9B", stamp(1)), ent("acme/Atlas-1-9B", "2026-09-01T00:00:00.000Z", cardData={"license": "mit", "new_version": "acme/Atlas-2-9B"})])
    p, _, _ = go(env, hub)
    change = fields_of(p, "acme/Atlas-1-9B", "successor")
    assert change["stage"] == "A3" and change["evidence"] == ["https://huggingface.co/acme/Atlas-1-9B"]


def test_a_gguf_repo_before_the_threshold_of_a_row_of_the_table_is_still_proposed_and_asks_for_nothing(env):
    env.put_state({"packone": MARK_30_09})
    hub = Hub()
    hub.page("packone", [quantized("packone/Atlas-2-9B-GGUF", "acme/Atlas-2-9B", createdAt="2026-09-01T00:00:00.000Z"), ent("packone/Young-1-1B", stamp(1))])
    p, _, transport = go(env, hub)
    change = fields_of(p, "acme/Atlas-2-9B", "gguf_repos")
    assert change["stage"] == "A2" and change["fields"]["gguf_repos"]["new"] == ["packone/Atlas-2-9B-GGUF"]
    assert stage_calls(transport) == 0 and asked(transport, "Atlas-2-9B-GGUF") == []  # the tags and the card of the list decide it


def test_an_entry_without_a_creation_time_counts_as_inside_the_threshold(env):
    env.put_state({"acme": MARK_30_09})
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-4-9B", stamp(1)), ent("acme/Atlas-5-9B", createdAt=MISSING)])
    p, _, _ = go(env, hub)
    assert sorted(new_rows(p)) == ["acme/Atlas-4-9B", "acme/Atlas-5-9B"]


def test_the_threshold_holds_in_the_pending_part_of_a_catch_up_too(env):
    env.put_state({"acme": {"covered_until": "2026-09-01T00:00:00.000Z", "pending": {"next_url": pg(11), "newest_seen": stamp(5)}}})
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1)), ent("acme/Old-1-3B", stamp(6))])
    hub.put(pg(11), [ent("acme/Mid-1-3B", "2026-09-10T00:00:00.000Z"), ent("acme/Ancient-1-3B", "2026-08-10T00:00:00.000Z", **UNDECIDED)])  # the threshold is 2026-08-30
    hub.config("acme/Ancient-1-3B", {})
    p, _, transport = go(env, hub)
    assert sorted(new_rows(p)) == ["acme/Mid-1-3B", "acme/Old-1-3B", "acme/Zeta-1-3B"] and cands(p, "B3") == []
    assert asked(transport, "Ancient-1-3B") == [] and stage_calls(transport) == 0


def test_the_proposal_names_the_oldest_threshold_of_the_run_in_a_note(env):
    p, _, _ = go(env, Hub())
    assert "watermark: entries created before 2026-08-28 are read for existing rows only" in p["notes"]
    env.put_state({a: MARK_30_09 for a in PUBLISHERS + PACKAGERS})
    p, _, _ = go(env, Hub(), now=NOW + timedelta(days=1))
    assert "watermark: entries created before 2026-09-28 are read for existing rows only" in p["notes"]


def test_the_note_names_the_older_threshold_when_the_accounts_have_different_marks(env):
    env.put_state({a: {"covered_until": "2026-09-20T00:00:00.000Z" if a == "nova" else "2026-09-30T00:00:00.000Z", "pending": None} for a in PUBLISHERS + PACKAGERS})
    p, _, _ = go(env, Hub())
    marks = [n for n in p["notes"] if n.startswith("watermark:")]
    assert marks == ["watermark: entries created before 2026-09-18 are read for existing rows only"]


def test_a_next_link_to_another_host_rejects_the_page_names_the_account_and_does_not_advance_it(env):
    env.put_state({"acme": {"covered_until": "2026-09-30T00:00:00.000Z", "pending": None}})
    hub = Hub()
    hub.page("acme", [ent("acme/N1-1B", stamp(1))], headers={"Link": '<https://evil.example/api/models?author=acme&cursor=x>; rel="next"'})
    p, result, transport = go(env, hub)
    assert any(n.startswith("acme: page error") for n in p["notes"]) and result.exit_code == 1
    assert env.state()["accounts"]["acme"]["covered_until"] == "2026-09-30T00:00:00.000Z" and asked(transport, "evil") == []


def pages_25(hub: Hub, newest: dict | None = None) -> None:
    """25 pages of one account, one repo each, newest first; the last page holds a repo older than the mark."""
    entries = [[ent(f"acme/Gen{i:02d}-3B", stamp(i / 24))] for i in range(1, 25)] + [[ent("acme/Gen99-3B", "2026-08-01T00:00:00.000Z")]]
    hub.chain("acme", entries, url_of=lambda n: f"https://huggingface.co/api/models?author=acme&cursor=pg{n}&limit=100")


def pg(n: int) -> str:
    return f"https://huggingface.co/api/models?author=acme&cursor=pg{n}&limit=100"


@pytest.mark.parametrize("top_pages", [1, 2])
def test_an_account_that_needs_25_pages_is_read_in_three_runs_and_every_repo_stands_once(env, top_pages):
    env.put_state({"acme": {"covered_until": "2026-09-01T00:00:00.000Z", "pending": None}})
    hub = Hub()
    pages_25(hub)
    p1, _, t1 = go(env, hub)
    assert len(asked(t1, "author=acme")) == 10
    state = env.state()["accounts"]["acme"]
    assert state == {"covered_until": "2026-09-01T00:00:00.000Z", "pending": {"next_url": pg(11), "newest_seen": stamp(1 / 24)}}
    assert any("read back to" in n and "only, continues next run" in n for n in p1["notes"])
    hub2 = Hub()
    pages_25(hub2)
    newest, older, seen = ent("acme/Late-2-3B", stamp(0.01)), ent("acme/Late-1-3B", stamp(0.02)), ent("acme/Gen01-3B", stamp(1 / 24))
    top2 = "https://huggingface.co/api/models?author=acme&cursor=top2&limit=100"
    if top_pages == 1:
        hub2.page("acme", [newest, seen])
    else:
        hub2.put(rules.list_url("acme"), [newest], headers={"Link": f'<{top2}>; rel="next"'})
        hub2.put(top2, [older, seen])
    p2, _, t2 = go(env, hub2, now=NOW + timedelta(days=1))
    calls = asked(t2, "author=acme")
    top = [rules.list_url("acme")] + ([] if top_pages == 1 else [top2])
    assert calls == top + [pg(n) for n in range(11, 11 + 10 - top_pages)]
    pending = env.state()["accounts"]["acme"]["pending"]
    assert pending == {"next_url": pg(11 + 10 - top_pages), "newest_seen": stamp(0.01)}
    hub3 = Hub()
    pages_25(hub3)
    hub3.page("acme", [newest])
    p3, _, t3 = go(env, hub3, now=NOW + timedelta(days=2))
    assert asked(t3, "author=acme")[1] == pg(11 + 10 - top_pages) and env.state()["accounts"]["acme"]["pending"] is None
    assert env.state()["accounts"]["acme"]["covered_until"] != "2026-09-01T00:00:00.000Z"
    repos = [c["hf_repo"] for c in p3["changes"]]
    assert len(repos) == len(set(repos)) and {f"acme/Gen{i:02d}-3B" for i in range(1, 25)} <= set(repos)


def test_a_next_url_that_answers_400_drops_the_pending_read_and_names_the_lost_position(env):
    env.put_state({"acme": {"covered_until": "2026-09-01T00:00:00.000Z", "pending": {"next_url": pg(11), "newest_seen": stamp(5)}}})
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1)), ent("acme/Old-1-3B", stamp(6))])
    hub.put(pg(11), b"", status=400)
    p, result, _ = go(env, hub)
    assert "acme: resume position lost, catching up again" in p["notes"] and result.exit_code == 1
    assert env.state()["accounts"]["acme"] == {"covered_until": "2026-09-01T00:00:00.000Z", "pending": None}


@pytest.mark.parametrize("url", [
    "https://evil.example/api/models?author=acme", "https://huggingface.co/api/datasets?author=acme", "https://huggingface.co/api/models?author=other",
])
def test_a_stored_next_url_with_another_host_path_or_author_is_dropped_without_a_request(env, url):
    env.put_state({"acme": {"covered_until": "2026-09-01T00:00:00.000Z", "pending": {"next_url": url, "newest_seen": stamp(5)}}})
    hub = Hub()
    hub.page("acme", [ent("acme/Old-1-3B", stamp(6))])
    p, _, transport = go(env, hub)
    assert asked(transport, "evil") == [] and url not in [u for _, u in transport.calls]
    assert env.state()["accounts"]["acme"]["pending"] is None


def test_a_run_that_ends_in_its_budget_writes_what_it_found_names_the_unread_accounts_and_keeps_their_marks(env):
    env.put_state({a: {"covered_until": "2026-09-30T00:00:00.000Z", "pending": None} for a in PUBLISHERS + PACKAGERS})
    hub = Hub()
    hub.page("acme", [ent("acme/N1-1B", stamp(1))])
    p, result, _ = go(env, hub, budget=3)
    assert result.exit_code == 1 and list(new_rows(p)) == ["acme/N1-1B"]
    assert any(n.startswith("budget ended: 2 accounts not read: packone, packtwo") for n in p["notes"])
    state = env.state()["accounts"]
    assert state["packone"]["covered_until"] == state["packtwo"]["covered_until"] == "2026-09-30T00:00:00.000Z"
    assert p["summary"]["incomplete"] >= 2


def test_a_page_that_answers_500_names_the_account_and_the_other_accounts_still_run(env):
    hub = Hub()
    hub.page("acme", b"oops", status=500)
    hub.page("nova", [ent("nova/N1-1B", stamp(1))])
    p, result, _ = go(env, hub)
    assert "acme: page error (HTTP 500), not advanced" in p["notes"] and list(new_rows(p)) == ["nova/N1-1B"] and result.exit_code == 1
    assert env.state()["accounts"]["acme"]["covered_until"] == "2026-08-30T00:00:00.000Z"


def test_a_first_run_without_a_mark_starts_thirty_days_before_the_published_day(env):
    hub = Hub()
    hub.page("acme", [ent("acme/N1-1B", stamp(1))])
    go(env, hub)
    assert env.state()["accounts"]["nova"]["covered_until"] == "2026-08-30T00:00:00.000Z"


# --- proposals from run to run, writing, files ------------------------------------------------------------------------


def test_two_runs_on_the_same_hub_and_table_give_the_same_changes_in_the_same_order_with_the_same_numbers(tmp_path):
    outcomes = []
    for name in ("one", "two"):
        env = Env(tmp_path / name)
        hub = Hub()
        hub.page("acme", [ent("acme/Zeta-2-3B", stamp(1)), ent("acme/Atlas-3-9B", stamp(2)), ent("acme/Zeta-1-3B", stamp(3))])
        hub.page("nova", [ent("nova/Aaa-1-1B", stamp(1))])
        hub.put(rules.b1_url("text-generation"), [b1_entry("lab1", 150), b1_entry("lab2", 150)])
        go(env, hub)
        outcomes.append(((env.out / TODAY / "proposal.json").read_bytes(), (env.out / TODAY / "proposal.md").read_bytes()))
    assert outcomes[0] == outcomes[1]
    p = json.loads(outcomes[0][0])
    assert [c["hf_repo"] for c in p["changes"]] == ["acme/Atlas-3-9B", "acme/Zeta-1-3B", "acme/Zeta-2-3B", "nova/Aaa-1-1B"]
    assert [c["n"] for c in p["changes"]] == [1, 2, 3, 4] and [c["k"] for c in p["candidates"]] == ["K1", "K2"]


def test_the_files_are_sorted_json_with_two_spaces_lf_and_a_final_newline_and_the_markdown_ends_on_the_checksum(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2))])
    go(env, hub)
    raw = (env.out / TODAY / "proposal.json").read_bytes()
    assert raw == (json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n").encode("utf-8") and b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    md = (env.out / TODAY / "proposal.md").read_bytes().decode("utf-8")
    assert md.endswith(f"proposal.json sha256: {digest(raw)}\n") and "\r" not in md
    assert json.loads(raw)["base_catalog_sha256"] == digest(CATALOG_BYTES) and json.loads(raw)["schema"] == 1
    assert set(json.loads(raw)) == {"schema", "date", "base_catalog_sha256", "summary", "notes", "changes", "candidates"}
    state_raw = (env.out / "scan-state.json").read_bytes()
    assert state_raw == (json.dumps(json.loads(state_raw), indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert json.loads(state_raw)["schema"] == 1 and not list(env.out.rglob("*.tmp"))


def test_a_change_not_decided_comes_back_with_its_first_date_and_number_even_when_the_mark_has_moved_on(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2)), ent("acme/Zeta-1-3B", stamp(2))])
    first, _, _ = go(env, hub)
    nxt, _, transport = go(env, Hub(), now=NOW + timedelta(days=1))
    rows = new_rows(nxt)
    assert sorted(rows) == ["acme/Atlas-3-9B", "acme/Zeta-1-3B"] and {r["open_since"] for r in rows.values()} == {TODAY}
    assert [r["n"] for r in rows.values()] == [r["n"] for r in new_rows(first).values()]


def test_a_decision_of_the_folder_removes_what_it_approved_or_rejected_and_keeps_what_was_left_open(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2)), ent("acme/Zeta-1-3B", stamp(2)), ent("acme/Zeta-2-3B", stamp(2))])
    go(env, hub)
    decision = {"state": "approved", "date": TODAY, "approved": [1], "rejected": [2], "admitted": []}
    (env.out / TODAY / "decision.json").write_bytes(json.dumps(decision).encode("utf-8"))
    nxt, _, _ = go(env, Hub(), now=NOW + timedelta(days=1))
    assert list(new_rows(nxt)) == ["acme/Zeta-2-3B"]
    assert nxt["summary"]["pending_publication"] == {"date": TODAY, "state": "approved"}


def test_a_rejected_repo_and_a_row_the_table_has_now_are_not_carried(env):
    env.prev(YESTERDAY, changes=[
        {"n": 1, "change": "new_row", "stage": "A1", "hf_repo": "acme/Rejected-1-7B", "kind": "text", "open_since": YESTERDAY, "evidence": [], "fields": {}},
        {"n": 2, "change": "new_row", "stage": "A1", "hf_repo": "acme/Atlas-1-9B", "kind": "text", "open_since": YESTERDAY, "evidence": [], "fields": {}},
    ])
    p, _, _ = go(env, Hub())
    assert p["changes"] == []


@pytest.mark.parametrize("state, expected", [
    ("approved", {"date": YESTERDAY, "state": "approved"}), ("uploaded", {"date": YESTERDAY, "state": "uploaded"}), ("published", None),
])
def test_an_unpublished_decision_is_named_in_the_summary_even_when_the_proposal_is_empty(env, state, expected):
    env.prev(YESTERDAY, decision={"state": state, "date": YESTERDAY, "approved": [], "rejected": [], "admitted": []})
    p, _, _ = go(env, Hub())
    assert p["summary"]["pending_publication"] == expected and p["changes"] == [] and p["summary"]["new"] == 0


class Crash(Exception):
    pass


def crashing(after: str | None):
    """A file layer that fails when it is about to write `after` (`None`: before the first file)."""
    class Files(scanner.DiskFiles):
        def write(self, path, data):
            if after is None or path.name == after:
                raise Crash(path.name)
            super().write(path, data)
    return Files()


def test_a_crash_before_the_first_file_leaves_every_file_as_it_was(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2))])
    with pytest.raises(Crash):
        env.run(hub, files=crashing(None))
    assert not (env.out / TODAY).exists() and not (env.out / "scan-state.json").exists()


def test_a_crash_between_the_proposal_and_the_state_keeps_the_state_and_the_next_run_gives_the_same_numbers(env):
    env.put_state({"acme": {"covered_until": "2026-09-30T00:00:00.000Z", "pending": None}})
    state_before = (env.out / "scan-state.json").read_bytes()
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2)), ent("acme/Zeta-1-3B", stamp(5))])
    with pytest.raises(Crash):
        env.run(hub, files=crashing("scan-state.json"))
    assert (env.out / "scan-state.json").read_bytes() == state_before and (env.out / TODAY / "proposal.md").exists()
    first = env.proposal()
    nxt, _, transport = go(env, hub, now=NOW + timedelta(days=1))
    assert asked(transport, "author=acme") == [rules.list_url("acme")]  # the page is read again
    assert [(c["hf_repo"], c["n"]) for c in nxt["changes"]] == [(c["hf_repo"], c["n"]) for c in first["changes"]]


def test_a_budget_candidate_checked_after_a_crash_replaces_the_candidate_with_a_row_and_keeps_the_other_numbers(env):
    env.put_state({"acme": {"covered_until": "2026-09-30T00:00:00.000Z", "pending": None}})
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2)), ent("acme/Zulu-1-7B", stamp(3), pipeline_tag=None, tags=["base_model:acme/Atlas-Base-9B"])])
    with pytest.raises(Crash):
        env.run(hub, sub=0, files=crashing("scan-state.json"))
    first = env.proposal()
    assert [c["hf_repo"] for c in first["candidates"]] == ["acme/Zulu-1-7B"] and first["candidates"][0]["kind_evidence"]["stage"] == "budget"
    hub.put(rules.base_url("acme/Atlas-Base-9B"), (FIX / "scan_model_answer.json").read_bytes())
    nxt, _, _ = go(env, hub, now=NOW + timedelta(days=1))
    rows = new_rows(nxt)
    assert list(rows) == ["acme/Atlas-3-9B", "acme/Zulu-1-7B"] and rows["acme/Zulu-1-7B"]["open_since"] == TODAY
    assert [c["hf_repo"] for c in nxt["changes"]].count("acme/Zulu-1-7B") == 1 and nxt["candidates"] == []
    assert rows["acme/Atlas-3-9B"]["n"] == first["changes"][0]["n"] == 1
    assert digest((env.out / "2026-10-05" / "proposal.json").read_bytes()) != digest((env.out / TODAY / "proposal.json").read_bytes())


class Recording(scanner.DiskFiles):
    def __init__(self) -> None:
        self.read_paths: list[Path] = []
        self.written: list[Path] = []

    def read(self, path):
        self.read_paths.append(Path(path))
        return super().read(path)

    def write(self, path, data):
        self.written.append(Path(path))
        super().write(path, data)


def test_the_scanner_opens_only_the_two_catalog_files_and_the_output_folder_and_writes_nothing_else(env):
    files = Recording()
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2))])
    env.prev(YESTERDAY)
    env.run(hub, files=files)
    out, inputs = env.out.resolve(), {env.catalog.resolve(), env.rejected.resolve()}
    assert files.read_paths and all(p.resolve() in inputs or out in p.resolve().parents for p in files.read_paths)
    assert {env.catalog.resolve(), env.rejected.resolve()} <= {p.resolve() for p in files.read_paths}
    assert files.written and all(out in p.resolve().parents for p in files.written)
    assert [p.name for p in files.written] == ["proposal.json", "proposal.md", "scan-state.json"]
    assert env.catalog.read_bytes() == CATALOG_BYTES and env.rejected.read_bytes() == (FIX / "scan_rejected.json").read_bytes()


def test_a_missing_or_invalid_catalog_ends_with_exit_two_and_writes_nothing(env):
    env.catalog.unlink()
    result, transport = env.run(Hub())
    assert result.exit_code == 2 and result.error and not env.out.exists() and transport.calls == []
    env.catalog.write_bytes(b"{not json")
    assert env.run(Hub())[0].exit_code == 2 and not env.out.exists()
    env.catalog.write_bytes(CATALOG_BYTES)
    env.rejected.unlink()
    assert env.run(Hub())[0].exit_code == 2 and not env.out.exists()


def test_a_day_that_already_has_a_decision_is_not_overwritten(env):
    env.prev(TODAY, decision={"state": "approved", "date": TODAY, "approved": [1], "rejected": [], "admitted": []})
    result, transport = env.run(Hub())
    assert result.exit_code == 2 and transport.calls == []


def test_the_command_line_runs_the_scan_and_gives_the_exit_code(env, capsys):
    transport = FixtureTransport(Hub().m)
    code = scanner.main(["--catalog", str(env.catalog), "--rejected", str(env.rejected), "--out", str(env.out), "--today", TODAY, "--budget", "300"], transport=transport)
    assert code == 0 and (env.out / TODAY / "proposal.json").exists() and len(transport.calls) == BASELINE_CALLS
    assert scanner.main(["--catalog", str(env.root / "none.json"), "--rejected", str(env.rejected), "--out", str(env.root / "o2")], transport=transport) == 2
    assert capsys.readouterr().err


def test_a_full_quiet_run_over_the_recorded_hub_asks_every_page_once_and_changes_nothing(env):
    p, result, transport = go(env, Hub())
    assert len(transport.calls) == len(set(transport.calls)) == BASELINE_CALLS == result.main_used and result.sub_used == 0
    assert p["changes"] == [] and p["candidates"] == [] and result.exit_code == 0 and env.catalog.read_bytes() == CATALOG_BYTES


def test_the_scanner_never_reads_a_token_or_a_credential_variable():
    source = (REPO / "scripts" / "catalog_scan.py").read_text(encoding="utf-8") + (REPO / "scripts" / "catalog_scan_rules.py").read_text(encoding="utf-8")
    assert "HF_TOKEN" not in source and "os.environ" not in source and "getenv" not in source


def test_the_recorded_list_page_of_the_fixture_files_is_read_through_its_link(env):
    page = json.loads((FIX / "scan_list_page.json").read_bytes())
    nxt = page["headers"]["Link"].split(">")[0][1:]
    hub = Hub()
    hub.put(rules.list_url("acme"), page["body"], headers=page["headers"])
    hub.put(nxt, [])
    env.put_state({"acme": {"covered_until": "2026-09-01T00:00:00.000Z", "pending": None}})
    p, _, transport = go(env, hub)
    assert asked(transport, "author=acme")[-1] == nxt and list(new_rows(p)) == ["acme/Atlas-3-9B"]


@pytest.mark.parametrize("name, tags", [
    ("acme/Zeta-1-3B-FP8", []), ("acme/Zeta-1-3B-GGUF", []), ("acme/Zeta-1-3B-bnb-4bit", []), ("acme/Zeta-1-3B", ["awq"]),
    ("acme/Zeta-1-3B", ["base_model:quantized:acme/Zeta-1-3B-Base"]),
])
def test_a_quantization_of_a_publisher_is_never_a_row_but_a_name_that_only_contains_a_suffix_is(env, name, tags):
    hub = Hub()
    hub.page("acme", [ent(name, stamp(1), tags=tags), ent("acme/Zeta-gguf-Lab-3B", stamp(1))])
    p, _, _ = go(env, hub)
    assert list(new_rows(p)) == ["acme/Zeta-gguf-Lab-3B"] and p["candidates"] == []


def test_a_gguf_change_not_decided_comes_back_and_goes_once_the_table_has_it(env):
    hub = Hub()
    hub.page("packone", [quantized("packone/Atlas-2-9B-GGUF", "acme/Atlas-2-9B")])
    go(env, hub)
    nxt, _, _ = go(env, Hub(), now=NOW + timedelta(days=1))
    change = fields_of(nxt, "acme/Atlas-2-9B", "gguf_repos")
    assert change["open_since"] == TODAY and change["fields"]["gguf_repos"] == {"old": [], "new": ["packone/Atlas-2-9B-GGUF"]}
    raw = json.loads(CATALOG_BYTES)
    raw["models"][1]["gguf_repos"] = ["packone/Atlas-2-9B-GGUF"]
    env.catalog.write_bytes(json.dumps(raw).encode("utf-8"))
    assert go(env, Hub(), now=NOW + timedelta(days=2))[0]["changes"] == []


def test_damaged_files_of_earlier_runs_cost_their_entries_and_never_the_run(env):
    env.out.mkdir(parents=True)
    (env.out / "scan-state.json").write_bytes(b"{")
    newest = env.out / YESTERDAY
    newest.mkdir()
    (newest / "proposal.json").write_bytes(b"not json")
    (newest / "decision.json").write_bytes(b"[1]")
    env.prev("2026-10-02", changes=[{"hf_repo": "acme/X-1-1B", "change": "new_row"}, "junk", {"n": 1}, {"hf_repo": "acme/Y-1-1B", "change": "new_row", "fields": 3}],
             candidates=[5, {"candidate": "B1"}, {"candidate": "B1", "account": "lab", "likes": "many", "open_since": TODAY}])
    p, result, _ = go(env, Hub())
    assert result.exit_code == 0 and p["changes"] == [] and p["candidates"] == [] and p["summary"]["pending_publication"] is None
    assert env.state()["accounts"]["acme"]["covered_until"] == "2026-08-30T00:00:00.000Z"
