"""Tests for the scanner's carry from run to run and its writing, after the third review round (AP9-KA-T, t3).

What an earlier proposal hands on is read as data that may be damaged or out of date: a carried row counts
wherever a row of the run counts, a rejection made since then takes everything that names the rejected repo,
an entry of the wrong shape costs that entry, and nothing is written before all three files exist in memory.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the scanner is tested from a clone", allow_module_level=True)

from modelroom.catalog import CatalogModel  # noqa: E402
from test_catalog_scan import (  # noqa: E402,F401  (the recorded Hub, the run and the readers of a proposal)
    CATALOG_BYTES, NOW, TODAY, YESTERDAY, Hub, applied, asked, b1_entry, b3, cands, ent, env, fields_of, go, new_of,
    new_rows, quantized, rules, scanner, stage_calls, stamp,
)
from test_catalog_scan_rules import b2_hub, carried_row, reject  # noqa: E402

DAY2 = NOW + timedelta(days=1)


# --- T3-1: a carried new row stands over a fresh one ----------------------------------------------------------------


@pytest.mark.parametrize("fresh, latest", [("acme/Atlas-3-9B", False), ("acme/Atlas-5-9B", True)])
def test_a_carried_new_row_that_stands_over_a_fresh_row_keeps_it_from_latest_and_one_below_it_does_not(env, fresh, latest):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-4-9B", stamp(2))])
    first = new_rows(go(env, hub)[0])["acme/Atlas-4-9B"]
    assert new_of(first)["latest"] is True
    hub2 = Hub()
    hub2.page("acme", [ent(fresh, stamp(1, DAY2))])
    p, result, _ = go(env, hub2, now=DAY2)
    row = new_of(new_rows(p)[fresh])
    assert (row["latest"], row["latest_source"]) == ((True, "https://huggingface.co/acme") if latest else (False, None))
    assert new_rows(p)["acme/Atlas-4-9B"]["fields"] == first["fields"] and result.exit_code == 0 and applied(p)


# --- T3-2: a carried successor whose target is rejected ---------------------------------------------------------------


def test_a_carried_successor_whose_target_is_rejected_since_the_last_run_is_dropped(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2)), ent("acme/Atlas-2-9B", stamp(5), cardData={"license": "mit", "new_version": "acme/Atlas-3-9B"})])
    assert fields_of(go(env, hub)[0], "acme/Atlas-2-9B", "successor")
    reject(env, {"hf_repo": "acme/Atlas-3-9B"})
    p, result, _ = go(env, Hub(), now=DAY2)
    assert p["changes"] == [] and result.exit_code == 0 and [n for n in p["notes"] if "Atlas" in n] == []


def test_the_carry_hands_the_successor_and_the_base_model_of_a_kind_to_the_rejection_check():
    rows = {m["hf_repo"]: CatalogModel.model_validate(m) for m in json.loads(CATALOG_BYTES)["models"]}
    successor = {"n": 1, "change": "field", "field": "successor", "stage": "A3", "hf_repo": "acme/Atlas-1-9B", "kind": "text",
                 "open_since": YESTERDAY, "evidence": [], "fields": {"successor": {"old": None, "new": "acme/Atlas-2-9B"}}}
    row = {**carried_row("acme/Zeta-1-3B", 2), "kind_evidence": {"stage": "base_model", "value": "acme/Zeta-Base-3B"}}
    proposal = {"changes": [successor, row], "candidates": []}
    for named in ("acme/Atlas-2-9B", "acme/Zeta-Base-3B"):
        changes = rules.carry_over(json.loads(json.dumps(proposal)), None, rows, set(), lambda repo: repo == named)[0]
        assert len(changes) == 1 and all(named not in json.dumps(c) for c in changes.values())
    assert len(rules.carry_over(json.loads(json.dumps(proposal)), None, rows, set(), lambda repo: False)[0]) == 2


# --- T3-3: a rejected model is no evidence of a candidate --------------------------------------------------------------


def test_a_b1_page_whose_only_strong_entry_is_rejected_gives_no_candidate(env):
    reject(env, {"hf_repo": "newlab/Rising-1-7B"})
    hub = Hub()
    hub.put(rules.b1_url("text-generation"), [b1_entry("newlab", 500, 9000)])
    p, result, _ = go(env, hub)
    assert cands(p) == [] and result.exit_code == 0 and not [n for n in p["notes"] if "limit" in n]


def test_a_carried_b1_candidate_whose_model_is_rejected_since_the_last_run_is_dropped(env):
    hub = Hub()
    hub.put(rules.b1_url("text-generation"), [b1_entry("newlab")])
    assert [c["account"] for c in cands(go(env, hub)[0], "B1")] == ["newlab"]
    reject(env, {"hf_repo": "newlab/Rising-1-7B"})
    p, result, _ = go(env, Hub(), now=DAY2)
    assert p["candidates"] == [] and result.exit_code == 0


@pytest.mark.parametrize("repo", ["quantco/Comet-3-8B-GGUF", "quantco/Atlas-2-9B-GGUF"])
def test_a_carried_b2_candidate_with_a_model_rejected_since_the_last_run_is_dropped(env, repo):
    first = cands(go(env, b2_hub({r: 1 for r in ("acme/Atlas-2-9B", "nova/Nimbus-1-30B", "orbit/Comet-3-8B")}))[0], "B2")
    assert [c["hf_repo"] for c in first] == ["quantco/Comet-3-8B-GGUF"] and repo in {r["hf_repo"] for r in first[0]["repos"]}
    reject(env, {"hf_repo": repo})
    p, result, _ = go(env, Hub(), now=DAY2)
    assert p["candidates"] == [] and result.exit_code == 0


def test_a_rejected_base_model_is_never_asked_for_the_kind_of_a_new_repo(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-2-3B", stamp(1), pipeline_tag=None, tags=["base_model:acme/Rejected-1-7B"])])
    hub.put(rules.config_url("acme/Zeta-2-3B"), b"", status=404)
    p, result, transport = go(env, hub)
    assert asked(transport, "/api/models/") == [] and [c["hf_repo"] for c in cands(p, "B3")] == ["acme/Zeta-2-3B"]
    assert result.exit_code == 0


# --- T3-4: evidence of the wrong type, and nothing written before everything is made -----------------------------------


def test_carried_entries_with_evidence_of_the_wrong_type_or_none_or_a_name_that_is_no_repo_are_dropped(env):
    no_evidence = carried_row("acme/Zeta-2-3B", 2)
    del no_evidence["evidence"]
    env.prev(YESTERDAY, changes=[{**carried_row("acme/Zeta-1-3B"), "evidence": [1]}, no_evidence], candidates=[
        {**b3("acme/Odd-1-7B", YESTERDAY), "evidence": "x"}, {**b3("acme/Odd-2-7B", YESTERDAY), "hf_repo": "acme"},
        {"k": "K9", "candidate": "B1", "account": "lab", "hf_repo": "lab/Hot-1-7B", "likes": 150, "open_since": YESTERDAY, "evidence": [7]},
    ])
    p, result, _ = go(env, Hub())
    assert result.exit_code == 0 and p["changes"] == [] and p["candidates"] == []
    assert {f.name for f in (env.out / TODAY).iterdir()} == {"proposal.json", "proposal.md"} and (env.out / "scan-state.json").exists()


class Counting(scanner.DiskFiles):
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, path, data):
        self.writes.append(Path(path).name)
        super().write(path, data)


def test_a_proposal_that_cannot_be_rendered_writes_no_file_and_ends_with_exit_two(env, monkeypatch):
    def broken(*args, **kwargs):
        raise TypeError("sequence item 0: expected str instance, int found")

    monkeypatch.setattr(scanner.R, "render_proposal", broken)
    files = Counting()
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2))])
    result, _ = env.run(hub, files=files)
    assert files.writes == [] and result.exit_code == 2 and "cannot be written" in result.error and not env.out.exists()


# --- T3-5: the temporary file is the run's own ------------------------------------------------------------------------


def test_a_catalog_under_the_name_an_older_scanner_gave_its_temporary_file_stays_untouched(env):
    target = env.out / TODAY / f"proposal.json.{os.getpid()}.tmp"
    target.parent.mkdir(parents=True)
    target.write_bytes(CATALOG_BYTES)
    env.catalog = target
    p, result, _ = go(env, Hub())
    assert result.exit_code == 0 and target.read_bytes() == CATALOG_BYTES and p["base_catalog_sha256"]
    assert sorted(f.name for f in target.parent.iterdir()) == sorted(["proposal.json", "proposal.md", target.name])


@pytest.mark.parametrize("how", ["the target is a folder", "the replace fails"])
def test_a_write_that_fails_leaves_no_temporary_file_behind(tmp_path, monkeypatch, how):
    target = tmp_path / "proposal.json"
    if how == "the target is a folder":
        target.mkdir()
    else:
        def refuse(src, dst):
            raise PermissionError(dst)
        monkeypatch.setattr(scanner.os, "replace", refuse)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(OSError):
        scanner.DiskFiles().write(target, b"{}\n")
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# --- T3-6: a rejection list of the wrong shape is a usage error --------------------------------------------------------


@pytest.mark.parametrize("entries", [[{"hf_repo": ["acme/X"]}], [{"account": 5}], [["acme/X"]], [{"note": "x"}], ["acme/X", {"hf_repo": None}]])
def test_a_rejection_list_with_an_entry_of_the_wrong_shape_ends_with_exit_two_before_any_request(env, entries):
    env.rejected.write_bytes(json.dumps(entries).encode("utf-8"))
    result, transport = env.run(Hub())
    assert result.exit_code == 2 and "neither a name nor an object with a name under hf_repo or account" in result.error
    assert transport.calls == [] and not env.out.exists()


def test_a_rejection_list_of_names_and_objects_is_read_as_repos_and_accounts():
    assert rules.parse_rejected(json.dumps(["acme/X", {"account": "nova"}, {"hf_repo": "orbit/Y", "date": "2026-09-30"}]).encode()) == (
        {"acme/X", "orbit/Y"}, {"nova"})


# --- T3-7: a full list of GGUF repos ----------------------------------------------------------------------------------


def fill_gguf(env) -> list[str]:
    raw = json.loads(CATALOG_BYTES)
    raw["models"][1]["gguf_repos"] = full = [f"packone/Atlas-2-9B-Q{i:02d}-GGUF" for i in range(50)]
    env.catalog.write_bytes(json.dumps(raw).encode("utf-8"))
    return full


def test_a_full_gguf_list_takes_no_further_repo_and_makes_no_change(env):
    fill_gguf(env)
    hub = Hub()
    hub.page("packtwo", [quantized("packtwo/Atlas-2-9B-GGUF", "acme/Atlas-2-9B")])
    p, result, _ = go(env, hub)
    assert p["changes"] == [] and result.exit_code == 0


def test_a_carried_gguf_change_that_the_cap_turns_into_the_list_of_the_table_is_dropped(env):
    fill_gguf(env)
    env.prev(YESTERDAY, changes=[{"n": 1, "change": "field", "field": "gguf_repos", "stage": "A2", "hf_repo": "acme/Atlas-2-9B", "kind": "text",
                                  "open_since": YESTERDAY, "evidence": ["https://huggingface.co/packtwo/Atlas-2-9B-GGUF"],
                                  "fields": {"gguf_repos": {"old": [], "new": ["packtwo/Atlas-2-9B-GGUF"]}}}])
    p, result, _ = go(env, Hub())
    assert p["changes"] == [] and result.exit_code == 0


# --- the fixes after the fourth review round (t4) ---------------------------------------------------------------------


BASES = ("acme/Atlas-2-9B", "nova/Nimbus-1-30B", "orbit/Comet-3-8B")


def b12(kind: str, k: str, account: str, *repos: str, bases: tuple = ()) -> dict:
    """A carried B1 or B2 as the scanner writes one: the best repo on top, the list behind it, a B2 with its rows."""
    return {
        "k": k, "candidate": kind, "account": account, "hf_repo": repos[0], "likes": 150, "downloads": 2000, "created_at": stamp(3),
        "open_since": YESTERDAY, "repos": [{"hf_repo": r, "likes": 150 - i, "downloads": 2000} for i, r in enumerate(repos)],
        "evidence": [f"https://huggingface.co/{r}" for r in repos], **({"rows": len(bases), "bases": list(bases)} if kind == "B2" else {}),
    }


@pytest.mark.parametrize("decided", ["K1", "K2"])
def test_a_carried_candidate_of_an_account_that_a_decision_took_through_the_other_kind_is_dropped(env, decided):
    env.prev(YESTERDAY, candidates=[
        b12("B1", "K1", "lab", "lab/Hot-1-7B"),
        b12("B2", "K2", "lab", "lab/Atlas-2-9B-GGUF", "lab/Nimbus-1-30B-GGUF", "lab/Comet-3-8B-GGUF", bases=BASES),
    ], decision={"state": "approved", "date": YESTERDAY, "approved": [], "rejected": [], "admitted": [decided]})
    p, result, _ = go(env, Hub())
    assert p["candidates"] == [] and result.exit_code == 0


def test_a_carried_candidate_keeps_what_the_rejection_list_leaves_with_its_first_date_and_a_b2_needs_three_of_each(env):
    env.prev(YESTERDAY, candidates=[
        b12("B1", "K1", "lab", "lab/Hot-1-7B", "lab/Warm-1-7B"),
        b12("B2", "K2", "pack", "pack/Atlas-2-9B-GGUF", "pack/Nimbus-1-30B-GGUF", "pack/Comet-3-8B-GGUF", "pack/Atlas-2-9B-Q4-GGUF", bases=BASES),
        b12("B2", "K3", "thin", "thin/Atlas-2-9B-GGUF", "thin/Nimbus-1-30B-GGUF", "thin/Comet-3-8B-GGUF", bases=BASES),
    ])
    reject(env, {"hf_repo": "lab/Hot-1-7B"}, {"hf_repo": "pack/Atlas-2-9B-GGUF"}, {"hf_repo": "thin/Comet-3-8B-GGUF"})
    p, result, _ = go(env, Hub())
    (b1,), (b2,) = cands(p, "B1"), cands(p, "B2")
    assert (b1["hf_repo"], b1["likes"], b1["open_since"], b1["evidence"]) == ("lab/Warm-1-7B", 149, YESTERDAY, ["https://huggingface.co/lab/Warm-1-7B"])
    assert [r["hf_repo"] for r in b1["repos"]] == ["lab/Warm-1-7B"] and result.exit_code == 0
    assert (b2["account"], b2["hf_repo"], b2["rows"], b2["likes"]) == ("pack", "pack/Nimbus-1-30B-GGUF", 3, 149 + 148 + 147)
    assert [r["hf_repo"] for r in b2["repos"]] == ["pack/Nimbus-1-30B-GGUF", "pack/Comet-3-8B-GGUF", "pack/Atlas-2-9B-Q4-GGUF"]
    assert b2["evidence"] == [f"https://huggingface.co/{r['hf_repo']}" for r in b2["repos"]] and b2["bases"] == list(BASES)


def test_a_carried_gguf_addition_keeps_only_the_evidence_of_what_is_still_added(env):
    both = ["packone/Atlas-2-9B-GGUF", "packtwo/Atlas-2-9B-GGUF"]
    env.prev(YESTERDAY, changes=[{"n": 1, "change": "field", "field": "gguf_repos", "stage": "A2", "hf_repo": "acme/Atlas-2-9B", "kind": "text",
                                  "open_since": YESTERDAY, "evidence": [f"https://huggingface.co/{r}" for r in both],
                                  "fields": {"gguf_repos": {"old": [], "new": both}}}])
    reject(env, {"account": "packtwo"})
    p, result, _ = go(env, Hub())
    change = fields_of(p, "acme/Atlas-2-9B", "gguf_repos")
    assert new_of(change)["gguf_repos"] == ["packone/Atlas-2-9B-GGUF"] and change["evidence"] == ["https://huggingface.co/packone/Atlas-2-9B-GGUF"]
    assert change["open_since"] == YESTERDAY and result.exit_code == 0


def test_a_parameter_count_too_large_for_a_number_is_no_size_and_never_ends_the_run(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2), safetensors={"parameters": {"BF16": 10 ** 400}, "total": 10 ** 400})])
    p, result, _ = go(env, hub)
    assert result.exit_code == 0 and new_of(new_rows(p)["acme/Atlas-3-9B"])["parameters_b"] == 9.0


# --- the fixes after the fifth review round (t5) ----------------------------------------------------------------------


@pytest.mark.parametrize("likes", ["many", None])
def test_a_stored_repo_of_a_carried_candidate_with_likes_of_the_wrong_type_goes_like_a_rejected_one_and_never_ends_the_run(env, likes):
    one, two = b12("B1", "K1", "lab", "lab/Hot-1-7B"), b12("B1", "K2", "pal", "pal/Hot-1-7B", "pal/Warm-1-7B")
    one["repos"][0]["likes"] = two["repos"][0]["likes"] = likes
    env.prev(YESTERDAY, candidates=[one, two])
    p, result, _ = go(env, Hub())
    assert result.exit_code == 0 and [(c["account"], c["hf_repo"], c["likes"]) for c in cands(p)] == [("pal", "pal/Warm-1-7B", 149)]


def test_a_carried_b2_keeps_the_likes_of_all_its_repos_and_loses_only_those_of_a_rejected_one(env):
    cand = b12("B2", "K1", "pack", *[f"pack/Model-{i}-7B-GGUF" for i in range(10)], bases=BASES)
    cand["likes"] = 1755  # the ten stored repos (1455) and twenty more made the case
    env.prev(YESTERDAY, candidates=[cand])
    first = cands(go(env, Hub())[0], "B2")
    assert [(c["likes"], c["hf_repo"], len(c["repos"])) for c in first] == [(1755, "pack/Model-0-7B-GGUF", 10)]
    reject(env, {"hf_repo": "pack/Model-3-7B-GGUF"})
    p, result, _ = go(env, Hub(), now=DAY2)
    assert [(c["likes"], len(c["repos"]), c["open_since"]) for c in cands(p, "B2")] == [(1755 - 147, 9, YESTERDAY)] and result.exit_code == 0


def test_a_fresh_gguf_addition_that_the_cap_leaves_out_behind_a_carried_one_gives_no_evidence(env):
    raw = json.loads(CATALOG_BYTES)
    raw["models"][1]["gguf_repos"] = [f"packone/Atlas-2-9B-Q{i:02d}-GGUF" for i in range(49)]
    env.catalog.write_bytes(json.dumps(raw).encode("utf-8"))
    env.prev(YESTERDAY, changes=[{"n": 1, "change": "field", "field": "gguf_repos", "stage": "A2", "hf_repo": "acme/Atlas-2-9B", "kind": "text",
                                  "open_since": YESTERDAY, "evidence": ["https://huggingface.co/packtwo/Atlas-2-9B-GGUF"],
                                  "fields": {"gguf_repos": {"old": [], "new": ["packtwo/Atlas-2-9B-GGUF"]}}}])
    hub = Hub()
    hub.page("packone", [quantized("packone/Atlas-2-9B-GGUF", "acme/Atlas-2-9B")])
    p, result, _ = go(env, hub)
    change = fields_of(p, "acme/Atlas-2-9B", "gguf_repos")
    added = new_of(change)["gguf_repos"]
    assert len(added) == 50 and added[-1] == "packtwo/Atlas-2-9B-GGUF" and "packone/Atlas-2-9B-GGUF" not in added
    assert change["evidence"] == ["https://huggingface.co/packtwo/Atlas-2-9B-GGUF"] and change["open_since"] == YESTERDAY and result.exit_code == 0


# --- the fixes after the sixth review round (t6) ----------------------------------------------------------------------


def test_a_carried_row_that_the_table_would_not_take_does_not_keep_a_fresh_row_from_latest(env):
    env.prev(YESTERDAY, changes=[carried_row("acme/Atlas-4-9B", 1, family="comet")])  # comet is orbit's family: the row cannot stand
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-3-9B", stamp(2))])
    p, result, _ = go(env, hub)
    assert new_of(new_rows(p)["acme/Atlas-3-9B"])["latest"] is True and "acme/Atlas-4-9B" not in new_rows(p)
    assert [n for n in p["notes"] if n.startswith("acme/Atlas-4-9B: left out")] and result.exit_code == 0 and applied(p)


@pytest.mark.parametrize("which", ["the out folder", "the day's folder"])
def test_an_out_path_that_is_a_file_ends_with_exit_two_before_any_request(env, which):
    target = env.out if which == "the out folder" else env.out / TODAY
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a folder\n")
    result, transport = env.run(Hub())
    assert result.exit_code == 2 and "--out must name a folder" in result.error and transport.calls == []
    assert target.read_bytes() == b"not a folder\n"


# --- the fixes after the seventh review round (t7) --------------------------------------------------------------------


def test_a_carried_row_that_the_table_would_not_take_is_no_successor_target(env):
    env.prev(YESTERDAY, changes=[carried_row("acme/Atlas-4-9B", 1, family="comet")])
    hub = Hub()
    hub.page("acme", [ent("acme/Atlas-2-9B", stamp(5), cardData={"license": "mit", "new_version": "acme/Atlas-4-9B"})])
    p, result, _ = go(env, hub)
    assert fields_of(p, "acme/Atlas-2-9B", "successor") is None and result.exit_code == 0 and applied(p).model_for("acme/Atlas-2-9B").latest
    assert [n for n in p["notes"] if "new_version acme/Atlas-4-9B, which is not a model of the table or of this proposal" in n]


@pytest.mark.parametrize("target", [f"{TODAY}/proposal.json", f"{TODAY}/proposal.md", "scan-state.json"])
def test_an_output_path_that_is_a_folder_ends_with_exit_two_before_any_request_or_write(env, target):
    (env.out / target).mkdir(parents=True)
    result, transport = env.run(Hub())
    assert result.exit_code == 2 and "it is a folder" in result.error and transport.calls == []
    assert not [p for p in env.out.rglob("*") if p.is_file()]


def test_a_gguf_repo_that_answers_for_three_rows_counts_once_for_a_b2(env):
    hub = Hub()
    one = ent("quantco/Everything-GGUF", stamp(1), tags=["gguf", *(f"base_model:quantized:{b}" for b in BASES)], likes=100)
    for base in BASES:
        hub.put(rules.b2_url(base), [one])
    (cand,) = cands(go(env, hub)[0], "B2")
    assert (cand["likes"], cand["rows"], [r["hf_repo"] for r in cand["repos"]], cand["evidence"]) == (
        100, 3, ["quantco/Everything-GGUF"], ["https://huggingface.co/quantco/Everything-GGUF"])
