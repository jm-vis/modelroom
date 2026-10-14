"""Tests for the name rule of the catalog scanner: it proposes a successor, the maintainer approves each one (NF, 2026-10-14).

`catalog_scan_latest.name_successors` is a pure function over rows: no request, no file, no clock. A row is a
`SimpleNamespace` with the attributes it reads. The scanner tests at the end run the whole scan against a recorded Hub
and against no Hub at all (`offline`).
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the scanner is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))
latest = importlib.import_module("catalog_scan_latest")
publish = importlib.import_module("catalog_publish_rules")

from modelroom.catalog_schema import build_catalog_v2, build_catalog_v3, project_to_schema2  # noqa: E402
from modelroom.http import FixtureTransport  # noqa: E402
from test_catalog_scan import (  # noqa: E402,F401  (the recorded Hub, the run and the readers of a proposal)
    CATALOG_BYTES, FIX, NOW, TODAY, YESTERDAY, Hub, ent, env, fields_of, go, new_of, rules, scanner, stamp,
)

DAY1, DAY2, DAY3 = date(2026, 1, 1), date(2026, 3, 1), date(2026, 6, 1)
HUB = "https://huggingface.co"


def row(repo, kind="text", *, latest_row=False, successor=None, size=None, created=DAY1):
    return SimpleNamespace(hf_repo=repo, kind=kind, latest=latest_row, successor=successor, parameters_b=size, hub_created=created)


def state_of(rows):
    return {r.hf_repo: (r.latest, r.successor) for r in rows}


def propose(rows, rejected=lambda repo: False, state=None):
    found, notes = latest.name_successors(rows, state if state is not None else state_of(rows), rejected)
    return {repo: pair.target for repo, pair in found.items()}, notes


# --- the pure rule: every condition, once positive and once negative ------------------------------------------------------


def test_a_row_gets_the_higher_version_of_its_group_as_successor_with_the_names_as_basis():
    rows = [row("acme/Nova-1", created=DAY1), row("acme/Nova-2", created=DAY2)]
    found, notes = latest.name_successors(rows, state_of(rows), lambda repo: False)
    assert {repo: pair.target for repo, pair in found.items()} == {"acme/Nova-1": "acme/Nova-2"} and notes == []
    assert found["acme/Nova-1"].basis == "name rule: version 2 stands over version 1 in acme/nova (text)"


def test_a_row_without_a_version_in_its_name_is_said_to_have_none():
    rows = [row("acme/Nova-7B", created=DAY1), row("acme/Nova-2-7B", created=DAY2)]
    found, _ = latest.name_successors(rows, state_of(rows), lambda repo: False)
    assert found["acme/Nova-7B"].basis == "name rule: version 2 stands over no version in acme/nova (text)"


@pytest.mark.parametrize("other", ["acme/Nova-2-LoRA", "acme/Nova-2-A2V", "acme/Nova-2-Omni", "acme/Nova-2-Flash-MOPD", "acme/Nova-2-Vision"])
def test_a_successor_has_the_same_variation_as_the_row(other):
    """Fix 1: a missing successor is allowed, a wrong one is not. `LTX-2.3 -> LTX-2.5-...-LoRA`, `SkyReels-V2 -> V3-A2V`,
    `jina-v2-small -> v5-omni-small`, `MiMo-7B-RL -> V2.6-Flash-MOPD` of the first list: another variation is no replacement."""
    assert propose([row("acme/Nova-1"), row(other, created=DAY2)]) == ({}, [])
    assert propose([row("acme/Nova-1-Flash-RL"), row("acme/Nova-2-Flash-MOPD", created=DAY2)]) == ({}, [])


def test_the_same_variation_in_any_case_and_order_is_the_same_variation_and_the_size_may_differ():
    assert propose([row("acme/Nova-1-Flash-VL", size=8), row("acme/Nova-2-vl-FLASH", size=70, created=DAY2)]) == ({"acme/Nova-1-Flash-VL": "acme/Nova-2-vl-FLASH"}, [])
    assert propose([row("acme/Nova-1-mini"), row("acme/Nova-2-mini", created=DAY2)]) == ({"acme/Nova-1-mini": "acme/Nova-2-mini"}, [])


def test_the_successor_is_the_highest_member_of_the_subgroup_of_the_same_variation_not_of_the_whole_group():
    """`LTX-2.3` points at `LTX-2.5`, though `LTX-2.5-...-LoRA` is as high: the subgroup (group and variation) decides."""
    rows = [row("acme/Nova-2.3"), row("acme/Nova-2.5", created=DAY2), row("acme/Nova-2.5-LoRA", created=DAY3), row("acme/Nova-2.4-LoRA", created=DAY2)]
    found, _ = propose(rows)
    assert found == {"acme/Nova-2.3": "acme/Nova-2.5", "acme/Nova-2.4-LoRA": "acme/Nova-2.5-LoRA"}


def test_a_proposal_the_maintainer_refused_is_not_made_again_but_another_successor_is():
    rows = [row("acme/Nova-1"), row("acme/Nova-2", created=DAY2)]
    refused = {("acme/Nova-1", "acme/Nova-2")}
    assert latest.name_successors(rows, state_of(rows), lambda repo: False, refused) == ({}, [])
    more = rows + [row("acme/Nova-3", created=DAY3)]
    found, _ = latest.name_successors(more, state_of(more), lambda repo: False, refused)
    assert {repo: pair.target for repo, pair in found.items()} == {"acme/Nova-1": "acme/Nova-3", "acme/Nova-2": "acme/Nova-3"}


def test_the_top_of_a_group_and_a_group_of_one_get_nothing():
    assert propose([row("acme/Nova-2"), row("acme/Other-1")]) == ({}, [])


@pytest.mark.parametrize("other", [row("acme/Nova-Image-2", "image-generation", created=DAY2), row("acme/Nova-Embed-2", "embedding", created=DAY2)])
def test_another_kind_is_no_successor_even_in_the_same_group(other):
    """`Qwen/Qwen-Image-2.1 -> Qwen/Qwen3.5-9B` and `Qwen3-Embedding-8B -> Qwen3.5-9B` of the probe: the kind separates the groups."""
    assert latest.line_key(other.hf_repo, None).group == latest.line_key("acme/Nova-1", None).group
    assert propose([row("acme/Nova-1"), other]) == ({}, [])


def test_text_and_vision_are_kept_apart_though_the_search_reads_the_names_alone():
    """The search has a name and no kind, so it groups a text and a vision model of one family; the catalog knows the kind
    and keeps the stricter reading: a vision model is no successor of a text model."""
    assert propose([row("acme/Nova-1", "text"), row("acme/Nova-2", "vision", created=DAY2)]) == ({}, [])
    assert propose([row("acme/Nova-1", "vision"), row("acme/Nova-2", "vision", created=DAY2)]) == ({"acme/Nova-1": "acme/Nova-2"}, [])


def test_another_publisher_or_another_line_is_another_group():
    assert propose([row("acme/Nova-1"), row("other/Nova-2", created=DAY2)]) == ({}, [])
    assert propose([row("acme/Nova-Small-1"), row("acme/Nova-Large-2", created=DAY2)]) == ({}, [])


def test_a_later_date_alone_is_no_higher_version():
    """The search stands a dated name over an undated one or an earlier date; a successor needs a version (U5 rule 4)."""
    assert propose([row("acme/Nova-2506"), row("acme/Nova-2507", created=DAY2)]) == ({}, [])
    assert propose([row("acme/Nova-7B"), row("acme/Nova-7B-2507", created=DAY2)]) == ({}, [])


@pytest.mark.parametrize("created, target_created", [(None, DAY2), (DAY1, None), (None, None), (DAY2, DAY2), (DAY3, DAY2)])
def test_the_successor_has_to_be_created_after_the_row_and_both_dates_have_to_be_known(created, target_created):
    """`sarvam-105b -> sarvam-1` and `Solar-Open-100B -> SOLAR-0-70b` of the probe: a higher number on an older model."""
    assert propose([row("acme/Nova-1", created=created), row("acme/Nova-2", created=target_created)]) == ({}, [])


def test_a_row_with_a_successor_keeps_it_and_a_target_with_a_successor_is_no_target():
    shipped = [row("acme/Nova-1", successor="acme/Nova-2"), row("acme/Nova-2", created=DAY2), row("acme/Nova-3", created=DAY3)]
    assert propose(shipped) == ({"acme/Nova-2": "acme/Nova-3"}, [])  # Nova-1 keeps its link, and is not asked again
    chained = [row("acme/Nova-1"), row("acme/Nova-2", created=DAY2), row("acme/Nova-3", created=DAY3)]
    state = {**state_of(chained), "acme/Nova-3": (False, "acme/Nova-4")}  # the end state of the run gives it a successor
    assert propose(chained, state=state) == ({}, [])


def test_a_latest_row_gets_no_successor_and_a_note_names_the_contradiction():
    rows = [row("acme/Nova-1", latest_row=True), row("acme/Nova-2", created=DAY2)]
    found, notes = propose(rows)
    assert found == {} and notes == ["acme/Nova-1: latest in the catalog, but acme/Nova-2 stands over it by version; no successor proposed, decide separately"]
    assert propose(rows, state={"acme/Nova-1": (False, None), "acme/Nova-2": (False, None)})[0] == {"acme/Nova-1": "acme/Nova-2"}  # the state of the run decides


def test_a_latest_row_without_a_complete_case_is_no_contradiction():
    assert propose([row("acme/Nova-1", latest_row=True), row("acme/Nova-2", created=None)]) == ({}, [])
    assert propose([row("acme/Nova-1", latest_row=True), row("acme/Nova-1-2507", created=DAY2)]) == ({}, [])


def test_the_nearest_size_of_the_highest_members_is_taken_in_the_order_of_the_search():
    rows = [row("acme/Nova-1-9B", size=9), row("acme/Nova-2-30B", size=30, created=DAY2), row("acme/Nova-2-9B", size=9, created=DAY2)]
    assert propose(rows)[0]["acme/Nova-1-9B"] == "acme/Nova-2-9B"
    rows = [row("acme/Nova-1-8B", size=8), row("acme/Nova-2-30B", size=30, created=DAY2), row("acme/Nova-2-12B", size=12, created=DAY2)]
    assert propose(rows)[0]["acme/Nova-1-8B"] == "acme/Nova-2-12B"


def test_only_a_highest_member_is_a_successor_so_two_old_rows_meet_at_the_top_and_no_chain_is_made():
    rows = [row("acme/Nova-1"), row("acme/Nova-2", created=DAY2), row("acme/Nova-3", created=DAY3)]
    assert propose(rows) == ({"acme/Nova-1": "acme/Nova-3", "acme/Nova-2": "acme/Nova-3"}, [])


def test_a_rejected_row_or_target_gives_no_proposal():
    rows = [row("acme/Nova-1"), row("acme/Nova-2", created=DAY2)]
    assert propose(rows, rejected=lambda repo: repo == "acme/Nova-1")[0] == {}
    assert propose(rows, rejected=lambda repo: repo == "acme/Nova-2")[0] == {}
    assert propose(rows, rejected=lambda repo: repo.startswith("other/"))[0] == {"acme/Nova-1": "acme/Nova-2"}


def test_a_name_without_a_reading_is_neither_row_nor_target():
    assert propose([row("acme/Nova-1"), row("acme/378-2", created=DAY2)]) == ({}, [])


def test_the_rule_is_deterministic_whatever_the_order_of_the_rows():
    rows = [row("acme/Nova-1"), row("acme/Nova-2", created=DAY2), row("acme/Nova-3", created=DAY3), row("acme/Zeta-1"), row("acme/Zeta-2", created=DAY2)]
    assert propose(rows) == propose(list(reversed(rows)))


# --- the scan: the proposal, the carry, the rejection list, no request offline -----------------------------------------


def dated_catalog(*, atlas_latest: bool = False, created=("2025-01-01", "2025-06-01")) -> bytes:
    """The fixture catalog with `Atlas-1-9B` and `Atlas-2-9B` dated; `Atlas-2-9B` loses `latest` unless `atlas_latest`."""
    document = json.loads(CATALOG_BYTES)
    for model, day in zip(document["models"][:2], created):
        model["hub_created"] = day
    if not atlas_latest:
        document["models"][1].update(latest=False, latest_source=None, latest_checked=None)
    return json.dumps(document).encode("utf-8")


def successor_changes(p):
    return [c for c in p["changes"] if c["change"] == "field" and c["field"] == "successor"]


def test_the_scan_proposes_the_name_rule_successor_as_its_own_numbered_change_with_the_page_of_the_successor_as_proof(env):
    env.catalog.write_bytes(dated_catalog())
    p, result, _ = go(env, Hub())
    [change] = successor_changes(p)
    assert (change["n"], change["stage"], change["hf_repo"], change["evidence"]) == (1, "A4", "acme/Atlas-1-9B", [f"{HUB}/acme/Atlas-2-9B"])
    assert new_of(change) == {"successor": "acme/Atlas-2-9B", "successor_source": f"{HUB}/acme/Atlas-2-9B", "successor_checked": TODAY}
    assert change["basis"] == "name rule: version 2 stands over version 1 in acme/atlas (text)" and result.exit_code == 0
    assert "basis: name rule: version 2 stands over version 1 in acme/atlas (text)" in rules.render_proposal(p)
    document = rules.apply_changes(json.loads(env.catalog.read_bytes()), p["changes"])
    assert build_catalog_v2(json.dumps(document), Path("catalog.json")).age_of("acme/Atlas-1-9B") == ("legacy", "acme/Atlas-2-9B")


def test_a_latest_row_that_a_higher_version_stands_over_is_a_note_and_no_change(env):
    env.catalog.write_bytes(dated_catalog(atlas_latest=True))
    document = json.loads(env.catalog.read_bytes())
    document["models"][0].update(latest=True, latest_source="https://huggingface.co/acme", latest_checked="2026-09-25")
    env.catalog.write_bytes(json.dumps(document).encode("utf-8"))
    hub = Hub()
    hub.put(rules.b2_url("acme/Atlas-1-9B"), [])
    p, _, _ = go(env, hub)
    assert successor_changes(p) == []
    assert "acme/Atlas-1-9B: latest in the catalog, but acme/Atlas-2-9B stands over it by version; no successor proposed, decide separately" in p["notes"]


def test_without_dates_on_the_rows_the_scan_proposes_no_successor(env):
    assert successor_changes(go(env, Hub())[0]) == []


def test_a_successor_the_card_names_is_the_only_proposal_for_that_row(env):
    env.catalog.write_bytes(dated_catalog(atlas_latest=True))
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-1-9B", stamp(6), cardData={"license": "mit", "new_version": "acme/Atlas-2-9B"})])
    [change] = successor_changes(go(env, hub)[0])
    assert (change["stage"], change["evidence"]) == ("A3", [f"{HUB}/acme/Atlas-1-9B"])


def test_a_rejected_row_comes_back_in_no_later_run(env):
    env.catalog.write_bytes(dated_catalog())
    assert successor_changes(go(env, Hub())[0])
    env.rejected.write_text(json.dumps([{"hf_repo": "acme/Atlas-1-9B", "date": TODAY}]), encoding="utf-8")
    assert successor_changes(go(env, Hub(), now=NOW + timedelta(days=3))[0]) == []
    assert successor_changes(go(env, Hub(), now=NOW + timedelta(days=4))[0]) == []


def a4_change(number=1):
    page = f"{HUB}/acme/Atlas-2-9B"
    return {"n": number, "change": "field", "field": "successor", "stage": "A4", "hf_repo": "acme/Atlas-1-9B", "kind": "text", "open_since": TODAY,
            "evidence": [page], "basis": "name rule: version 2 stands over version 1 in acme/atlas (text)",
            "fields": {"successor": {"old": None, "new": "acme/Atlas-2-9B"}, "successor_source": {"old": None, "new": page}, "successor_checked": {"old": None, "new": TODAY}}}


def reject_a4(rejected=b"[]"):
    request = publish.parse_request(None, "1", None)
    proposal = {"changes": [a4_change()], "candidates": []}
    publish.check_request(proposal, request)
    document = json.loads(dated_catalog())
    return publish.plan_apply(document, dated_catalog(), rejected, proposal, request, TODAY)


def test_rejecting_a_name_rule_proposal_lists_the_pair_and_not_the_row():
    plan = reject_a4()
    assert json.loads(plan.rejected_text) == [{"hf_repo": "acme/Atlas-1-9B", "successor": "acme/Atlas-2-9B", "date": TODAY}]
    assert rules.parse_rejected(plan.rejected_text.encode()) == (set(), set())  # no row, no account is rejected
    assert rules.parse_rejected_pairs(plan.rejected_text.encode()) == {("acme/Atlas-1-9B", "acme/Atlas-2-9B")}
    assert reject_a4(plan.rejected_text.encode()).rejected_text == plan.rejected_text  # once only


def test_a_list_of_the_old_form_reads_as_before_and_holds_no_pair():
    raw = (FIX / "scan_rejected.json").read_bytes()
    assert rules.parse_rejected_pairs(raw) == set() and rules.parse_rejected(raw) == ({"acme/Rejected-1-7B"}, {"spamlabs"})


@pytest.mark.parametrize("entry", [{"hf_repo": "acme/Atlas-1-9B", "successor": 3}, {"successor": "acme/Atlas-2-9B"}, {"hf_repo": "acme/Atlas-1-9B", "successor": ""}])
def test_a_damaged_pair_is_an_error_and_never_lifts_a_rejection(entry):
    with pytest.raises(ValueError):
        rules.parse_rejected(json.dumps([entry]).encode())
    with pytest.raises(ValueError):
        rules.parse_rejected_pairs(json.dumps([entry]).encode())


def test_a_refused_pair_comes_back_in_no_later_run_but_the_row_stays_open_for_the_rest(env):
    env.catalog.write_bytes(dated_catalog())
    assert successor_changes(go(env, Hub())[0])
    env.rejected.write_bytes(reject_a4().rejected_text.encode())
    for days in (3, 4):
        assert successor_changes(go(env, Hub(), now=NOW + timedelta(days=days))[0]) == []
    hub = Hub()  # the row is no longer rejected as a whole: its card is read and proposes a field of its own
    hub.page("acme", [ent("acme/Atlas-1-9B", stamp(6), cardData={"license": "mit", "new_version": "acme/Atlas-2-9B"})])
    document = json.loads(dated_catalog(atlas_latest=True))
    env.catalog.write_bytes(json.dumps(document).encode())
    p, _, _ = go(env, hub, now=NOW + timedelta(days=5))
    assert [c["stage"] for c in successor_changes(p)] == ["A3"]


def test_a_pair_the_decision_rejected_does_not_lock_the_other_successor_fields_of_the_row_in_the_next_scan(env):
    """Codex nf-b1 2: `Atlas-1 -> Atlas-2` rejected in a real decision file, `Atlas-3` in the catalog since: the next scan
    proposes `Atlas-1 -> Atlas-3`; the pair alone is locked, by the rejection list."""
    document = json.loads(dated_catalog())
    third = {**document["models"][1], "hf_repo": "acme/Atlas-3-9B", "source": f"{HUB}/acme/Atlas-3-9B", "hub_created": "2025-09-01"}
    document["models"].append(third)
    env.catalog.write_bytes(json.dumps(document).encode("utf-8"))
    env.prev(YESTERDAY, changes=[a4_change()], decision={"state": "published", "approved": [], "rejected": [1], "admitted": []})
    env.rejected.write_bytes(reject_a4().rejected_text.encode())
    p, _, _ = go(env, Hub())
    assert {c["hf_repo"]: new_of(c)["successor"] for c in successor_changes(p)} == {"acme/Atlas-1-9B": "acme/Atlas-3-9B", "acme/Atlas-2-9B": "acme/Atlas-3-9B"}


def test_offline_after_a_normal_scan_of_the_same_day_keeps_the_open_proposals_of_that_day(env):
    """Codex nf-b1 1: the offline run takes over today's proposal like an earlier day's, and makes the A4 changes again."""
    env.catalog.write_bytes(dated_catalog())
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2))])
    first, _, _ = go(env, hub)
    assert "acme/Atlas-3-9B" in {c["hf_repo"] for c in first["changes"] if c["change"] == "new_row"}
    scanner.scan(env.catalog, env.rejected, env.out, transport=FixtureTransport({}), now=NOW, offline=True)
    again = env.proposal()
    assert "acme/Atlas-3-9B" in {c["hf_repo"] for c in again["changes"] if c["change"] == "new_row"}
    assert [c["hf_repo"] for c in successor_changes(again)] == [c["hf_repo"] for c in successor_changes(first)] and len(successor_changes(again)) == 1


def test_an_open_proposal_is_carried_with_its_first_day_though_its_target_is_no_latest_row(env):
    env.catalog.write_bytes(dated_catalog())
    first = successor_changes(go(env, Hub())[0])[0]
    again, result, _ = go(env, Hub(), now=NOW + timedelta(days=1))
    [change] = successor_changes(again)
    assert (change["open_since"], change["evidence"], new_of(change)["successor"]) == (first["open_since"], first["evidence"], "acme/Atlas-2-9B")
    assert not any("dropped" in note for note in again["notes"]) and result.exit_code == 0


def test_a_proposal_the_rule_no_longer_makes_is_not_carried(env):
    env.catalog.write_bytes(dated_catalog())
    assert successor_changes(go(env, Hub())[0])
    env.catalog.write_bytes(dated_catalog(created=("2025-01-01", None)))  # the successor lost its date
    assert successor_changes(go(env, Hub(), now=NOW + timedelta(days=1))[0]) == []


def test_a_proposal_that_the_table_holds_is_not_made_again(env):
    document = json.loads(dated_catalog())
    document["models"][0].update(successor="acme/Atlas-2-9B", successor_source=f"{HUB}/acme/Atlas-2-9B", successor_checked="2026-10-01")
    env.catalog.write_bytes(json.dumps(document).encode("utf-8"))
    assert successor_changes(go(env, Hub())[0]) == []


def test_offline_the_scan_makes_the_proposal_from_the_table_alone_and_asks_nobody(env):
    env.catalog.write_bytes(dated_catalog())
    transport = FixtureTransport({})  # a request for any page is a KeyError
    result = scanner.scan(env.catalog, env.rejected, env.out, transport=transport, now=NOW, offline=True)
    assert result.exit_code == 0 and transport.calls == []
    p = env.proposal()
    assert [(c["hf_repo"], new_of(c)["successor"]) for c in successor_changes(p)] == [("acme/Atlas-1-9B", "acme/Atlas-2-9B")]
    assert p["summary"]["changed"] == 1 and p["candidates"] == [] and (env.out / TODAY / "proposal.md").exists()


def test_offline_does_not_move_the_marks_of_the_accounts(env):
    env.catalog.write_bytes(dated_catalog())
    scanner.scan(env.catalog, env.rejected, env.out, transport=FixtureTransport({}), now=NOW, offline=True)
    first = env.state()
    scanner.scan(env.catalog, env.rejected, env.out, transport=FixtureTransport({}), now=NOW + timedelta(days=5), offline=True)
    assert env.state() == first


def test_the_command_line_has_an_offline_switch(env):
    env.catalog.write_bytes(dated_catalog())
    argv = ["--catalog", str(env.catalog), "--rejected", str(env.rejected), "--out", str(env.out), "--offline", "--today", "2026-10-06"]
    assert scanner.main(argv, transport=FixtureTransport({})) == 0
    assert [c["hf_repo"] for c in successor_changes(env.proposal("2026-10-06"))] == ["acme/Atlas-1-9B"]


# --- compatibility: the readers of 1.1.0 (schema 3) and 1.0.0 (the v2 projection) ----------------------------------------


def test_a_catalog_with_name_rule_successors_reads_under_1_1_0_and_projects_to_the_reader_of_1_0_0(env, tmp_path):
    env.catalog.write_bytes(dated_catalog())
    scanner.scan(env.catalog, env.rejected, env.out, transport=FixtureTransport({}), now=NOW, offline=True)
    changes = env.proposal()["changes"]
    document = rules.apply_changes(json.loads(env.catalog.read_bytes()), changes)
    document["schema_version"] = 3
    for model in document["models"]:
        model.setdefault("gguf_checks", [])
        model.setdefault("setup_note", None)
    seen_by_1_1_0 = build_catalog_v3(json.dumps(document), Path("catalog-v3.json"))
    seen_by_1_0_0 = build_catalog_v2(json.dumps(project_to_schema2(document)), Path("catalog-v2.json"))
    for catalog in (seen_by_1_1_0, seen_by_1_0_0):
        assert catalog.age_of("acme/Atlas-1-9B") == ("legacy", "acme/Atlas-2-9B")
        assert catalog.model_for("acme/Atlas-1-9B").successor_source == f"{HUB}/acme/Atlas-2-9B"
        assert catalog.age_of("acme/Atlas-2-9B") == ("unknown", None)


def test_the_proposal_over_the_shipped_catalog_validates_in_both_readers_whatever_it_holds(env):
    shipped = (REPO / "modelroom" / "catalog.json").read_bytes()
    env.catalog.write_bytes(shipped)
    scanner.scan(env.catalog, env.rejected, env.out, transport=FixtureTransport({}), now=NOW, offline=True)
    changes = env.proposal()["changes"]
    document = rules.apply_changes(json.loads(shipped), changes)
    assert build_catalog_v3(json.dumps(document), Path("catalog-v3.json")) is not None
    for model in document["models"]:  # the projection refuses a finding by design: the old channel is made without them
        model.update(gguf_checks=[], setup_note=None)
    assert build_catalog_v2(json.dumps(project_to_schema2(document)), Path("catalog-v2.json")) is not None
