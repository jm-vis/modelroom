"""Tests for `scripts/catalog_publish_rules.py`: what the maintainer named, what it does to the table, the decision file (AP9-KA-P).

The rules are pure: every test hands one a value read from `tests/fixtures/publish_*.json` (accounts and models invented)
and reads the value back. Nothing here opens a file of the real catalog, a socket or a process.
"""

from __future__ import annotations

import copy
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the publisher is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))
P = importlib.import_module("catalog_publish_rules")
R = importlib.import_module("catalog_scan_rules")

from modelroom.catalog_schema import build_catalog_v2  # noqa: E402

FIX = REPO / "tests" / "fixtures"
CATALOG_BYTES, REJECTED_BYTES = (FIX / "publish_catalog.json").read_bytes(), (FIX / "scan_rejected.json").read_bytes()
PROPOSAL_BYTES = (FIX / "publish_proposal.json").read_bytes()
TODAY = "2026-10-04"
SHA = P.sha256_of(PROPOSAL_BYTES)


@pytest.fixture
def proposal() -> dict:
    return json.loads(PROPOSAL_BYTES)


@pytest.fixture
def document() -> dict:
    return json.loads(CATALOG_BYTES)


def plan(document, proposal, approve=None, reject=None, admit=None, rejected=REJECTED_BYTES):
    request = P.parse_request(approve, reject, admit)
    P.check_request(proposal, request)
    return P.plan_apply(document, CATALOG_BYTES, rejected, proposal, request, TODAY)


def refused(call, *args, **kw) -> str:
    with pytest.raises(P.PublishError) as caught:
        call(*args, **kw)
    return str(caught.value)


# --- the fixture itself ---------------------------------------------------------------------------------------------


def test_the_fixture_proposal_has_one_change_of_each_kind_and_the_scanner_accepts_all_of_them(proposal, document):
    kept, notes = R.check_changes(document, proposal["changes"])
    assert len(kept) == len(proposal["changes"]) == 6 and notes == []
    by = {(c["change"], c.get("field")): c for c in proposal["changes"]}
    assert set(by) == {("new_row", None), ("field", "gguf_repos"), ("field", "successor"), ("field", "latest_source"), ("field", "ollama")}
    gguf = [c["fields"]["gguf_repos"] for c in proposal["changes"] if c.get("field") == "gguf_repos"]
    assert any(len(g["new"]) > len(g["old"]) for g in gguf) and any(len(g["new"]) < len(g["old"]) for g in gguf)
    successor = by[("field", "successor")]
    assert set(successor["fields"]) == {"successor", "successor_source", "successor_checked"} and successor["evidence"]
    assert set(by[("field", "ollama")]["fields"]) == {"ollama_base", "ollama_tag"}
    assert [c["candidate"] for c in proposal["candidates"]] == ["B1", "B2", "B3"] and proposal["base_catalog_sha256"] == P.sha256_of(CATALOG_BYTES)
    assert R.dump_json(proposal) == PROPOSAL_BYTES and not PROPOSAL_BYTES.startswith(b"\xef\xbb\xbf") and b"\r" not in PROPOSAL_BYTES


# --- checksum, commit, remote, the upload call -----------------------------------------------------------------------


@pytest.mark.parametrize("given", [None, "", "abc", SHA.upper(), SHA[:-1], SHA + "0", "g" * 64])
def test_a_checksum_that_is_missing_or_no_64_lower_case_hex_is_refused(given):
    assert "missing or is not 64" in refused(P.check_sha, given, PROPOSAL_BYTES)


def test_a_checksum_of_other_bytes_is_refused_and_the_one_of_these_bytes_passes():
    assert P.check_sha(SHA, PROPOSAL_BYTES) == SHA
    assert "has changed" in refused(P.check_sha, SHA, PROPOSAL_BYTES.replace(b"acme", b"acmf", 1))


@pytest.mark.parametrize("commit", ["", "--force", "main", "A" * 40, "a" * 39, "a" * 41, "a" * 40 + "\n"])
def test_a_commit_must_be_a_full_lower_case_hash_so_git_never_reads_it_as_an_option(commit):
    refused(P.check_commit, commit)


def test_a_full_hash_and_a_plain_remote_name_pass_and_an_option_is_no_remote():
    assert P.check_commit("0123456789abcdef" * 2 + "01234567") and P.check_remote("forgejo") == "forgejo"
    for remote in ("", "-f", "--force", "a b", "a/b", "x" * 65):
        refused(P.check_remote, remote)


def test_the_upload_argument_list_is_fixed_and_names_the_pinned_tool_the_file_and_the_target():
    assert re.fullmatch(r"\d+\.\d+\.\d+", P.HF_HUB_VERSION)
    assert P.upload_args("jm-vis/x", Path("a") / "f", "c" * 40) == [
        "uvx", "--from", f"huggingface_hub=={P.HF_HUB_VERSION}", "hf", "upload", "jm-vis/x", str(Path("a") / "f"), "catalog-v2.json",
        "--repo-type", "dataset", "--commit-message", "c" * 40,
    ]


def test_the_eight_kinds_are_the_values_of_the_kind_table():
    assert len(P.KINDS) == 8 and set(P.KINDS) == {"text", "vision", "embedding", "speech-recognition", "speech-synthesis", "audio", "image-generation", "video-generation"}


# --- what the maintainer named ---------------------------------------------------------------------------------------


def test_the_lists_are_read_as_numbers_and_candidates_with_blanks_trimmed():
    request = P.parse_request("1, 2,5", "4,K3", "K1")
    assert (request.approve, request.reject, request.admit) == (("1", "2", "5"), ("4", "K3"), ("K1",))
    assert P.parse_request(None, None, None).named == ()


@pytest.mark.parametrize("approve, reject, admit", [("0", None, None), ("01", None, None), ("1,,2", None, None), ("one", None, None), (None, "k1", None),
                                                    (None, None, "3"), (None, None, "K0"), ("", None, None), ("1 2", None, None)])
def test_a_name_that_is_no_number_and_no_candidate_is_refused_and_admit_takes_candidates_only(approve, reject, admit):
    refused(P.parse_request, approve, reject, admit)


@pytest.mark.parametrize("approve, reject, admit, fragment", [
    (None, None, None, "nothing is named"), ("9", None, None, "9 is not in the proposal"), (None, None, "K9", "K9 is not in the proposal"),
    ("1", "1", None, "named twice"), ("1,1", None, None, "named twice"), (None, "K1", "K1", "named twice"),
    ("K1", None, None, "is a candidate"), (None, None, "K3", "only --complete"),
    ("9", "9", None, "9 is not in the proposal"), (None, None, None, "nothing"),
])
def test_a_request_the_proposal_cannot_carry_is_refused_in_the_order_of_the_brief(proposal, approve, reject, admit, fragment):
    assert fragment in refused(lambda: P.check_request(proposal, P.parse_request(approve, reject, admit)))


def test_a_candidate_of_a_new_account_may_be_admitted_and_a_number_or_candidate_rejected(proposal):
    P.check_request(proposal, P.parse_request("1", "4,K3", "K1,K2"))


# --- the result of an --apply -----------------------------------------------------------------------------------------


def test_only_the_approved_numbers_enter_the_table_and_the_day_of_the_call_is_the_published_day(document, proposal):
    result = plan(document, proposal, approve="1,2,5")
    target = json.loads(result.catalog_text)
    expected = R.apply_changes(document, [proposal["changes"][i] for i in (0, 1, 4)])
    expected["published"] = TODAY
    assert target == expected and result.catalog_text.encode() == R.dump_json(expected)
    assert (result.approved, result.rejected, result.admitted) == ([1, 2, 5], [], [])
    assert result.rejected_text == REJECTED_BYTES.decode() and build_catalog_v2(result.catalog_text, Path("c.json"))


def test_a_request_that_only_rejects_leaves_both_the_table_and_its_day_alone_and_adds_the_repo_of_the_change(document, proposal):
    result = plan(document, proposal, reject="4")
    assert result.catalog_text == CATALOG_BYTES.decode() and result.approved == [] and result.rejected == [4]
    entries = json.loads(result.rejected_text)
    assert entries[:2] == json.loads(REJECTED_BYTES) and entries[2] == {"hf_repo": "acme/Atlas-1-9B", "date": TODAY}
    assert R.parse_rejected(result.rejected_text.encode()) == ({"acme/Rejected-1-7B", "acme/Atlas-1-9B"}, {"spamlabs"})


def test_a_rejected_candidate_names_the_account_and_one_of_open_kind_names_the_repo_and_both_stay_readable(document, proposal):
    result = plan(document, proposal, reject="K1,K3,K2")
    assert result.rejected == ["K1", "K2", "K3"]
    assert json.loads(result.rejected_text)[2:] == [{"account": "newlab", "date": TODAY}, {"account": "packthree", "date": TODAY}, {"hf_repo": "acme/Odd-1-5B", "date": TODAY}]
    assert R.parse_rejected(result.rejected_text.encode()) == ({"acme/Rejected-1-7B", "acme/Odd-1-5B"}, {"spamlabs", "newlab", "packthree"})


def test_a_name_already_on_the_rejection_list_is_not_added_twice_and_the_bytes_stay_when_nothing_is_new(document, proposal):
    listed = R.dump_json([{"account": "newlab", "date": "2026-09-01"}])
    result = plan(document, proposal, reject="K1", rejected=listed)
    assert result.rejected_text == listed.decode()
    twice = copy.deepcopy(proposal)
    twice["changes"][3]["hf_repo"] = "acme/Atlas-1-9B"
    twice["changes"][4]["hf_repo"] = "acme/Atlas-1-9B"
    assert len(json.loads(plan(document, twice, reject="4,5").rejected_text)) == 3


def test_an_admitted_publisher_gets_an_entry_of_the_form_the_schema_asks_for_and_a_packager_a_plain_name(document, proposal):
    result = plan(document, proposal, admit="K1,K2")
    target = json.loads(result.catalog_text)
    assert target["publishers"][-1] == {"account": "newlab", "name": "newlab", "region": None} and target["packagers"][-1] == "packthree"
    assert target["models"] == document["models"] and target["published"] == TODAY and result.admitted == ["K1", "K2"]
    assert build_catalog_v2(result.catalog_text, Path("c.json")).packagers[-1] == "packthree"


def test_an_account_the_table_lists_already_cannot_be_admitted_and_nothing_is_made(document, proposal):
    listed = copy.deepcopy(proposal)
    listed["candidates"][0]["account"] = "acme"
    assert "would not validate" in refused(plan, document, listed, admit="K1")


def test_a_change_that_cannot_be_applied_to_the_table_is_refused_with_a_sentence(document, proposal):
    broken = copy.deepcopy(proposal)
    broken["changes"][3]["hf_repo"] = "acme/Gone-1-1B"
    assert "cannot be applied" in refused(plan, document, broken, approve="4")


# --- AP9-KA-H K1b: a row without a reading needs the maintainer's confirmation ----------------------------------------------------


def unread(proposal: dict, mark: object = "absent") -> dict:
    """The proposal with change 1 renamed to a repo the grammar has no reading for; `mark` is the `reading` mark of the
    proposal (absent: the form of an old proposal)."""
    data = copy.deepcopy(proposal)
    change = data["changes"][0]
    change["hf_repo"] = change["fields"]["hf_repo"]["new"] = "acme/Ovie-512"
    change["fields"]["source"]["new"] = "https://huggingface.co/acme/Ovie-512"
    if mark != "absent":
        change["reading"] = mark
    return data


def plan_unread(document, proposal, approve=None, approve_unread=None, reject=None):
    request = P.parse_request(approve, reject, None, approve_unread)
    P.check_request(proposal, request)
    return P.plan_apply(document, CATALOG_BYTES, REJECTED_BYTES, proposal, request, TODAY)


@pytest.mark.parametrize("mark", ["absent", None, True, False], ids=["old proposal", "null", "reading true", "reading false"])
def test_a_new_row_without_a_reading_is_refused_whatever_the_proposal_says_about_it(document, proposal, mark):
    """Not-readable is worked out from the repo and the size here; the mark of the proposal is for the reader only, so
    a missing, old, or wrong mark passes nothing."""
    assert refused(plan_unread, document, unread(proposal, mark), approve="1") == (
        "row 1 has no reading; approve it with --approve-unread 1 after adding it to SHIPPED_WITHOUT_READING")


def test_a_carried_new_row_without_a_reading_is_refused_too(document, proposal):
    carried = unread(proposal, False)
    carried["changes"][0]["open_since"] = "2026-09-01"  # a row that has waited in earlier proposals
    assert "has no reading" in refused(plan_unread, document, carried, approve="1,2")


def test_a_confirmed_row_without_a_reading_enters_the_table_and_the_mark_stays_out_of_the_row(document, proposal):
    data = unread(proposal, False)
    result = plan_unread(document, data, approve="1", approve_unread="1")
    rows = {m["hf_repo"]: m for m in json.loads(result.catalog_text)["models"]}
    assert "acme/Ovie-512" in rows and "reading" not in rows["acme/Ovie-512"]
    assert (result.approved, result.approved_unread) == ([1], [1]) and build_catalog_v2(result.catalog_text, Path("c.json"))


def test_a_row_with_a_reading_needs_no_confirmation_and_a_confirmation_of_it_does_no_harm(document, proposal):
    assert plan_unread(document, proposal, approve="1").approved_unread == []
    assert plan_unread(document, proposal, approve="1", approve_unread="1").approved_unread == [1]


def test_a_field_change_is_not_a_row_and_needs_no_confirmation(document, proposal):
    assert plan_unread(document, unread(proposal, False), approve="2,5").approved == [2, 5]  # the other changes of the fixture


@pytest.mark.parametrize("approve, approve_unread, reject, fragment", [
    ("1", "2", None, "--approve-unread 2 needs --approve 2"),  # no approval
    ("1", "9", None, "9 is not in the proposal"),  # an unknown number
    (None, "1", "1", "--approve-unread 1 needs --approve 1"),  # approval and rejection of one number are refused before
    ("1", "1,1", None, "named twice in --approve-unread: 1"),  # the repetition that the reading of the decision would refuse
    ("1,1", "1", None, "named twice: 1"),
])
def test_a_confirmation_is_a_subset_of_the_approval_and_named_once(proposal, approve, approve_unread, reject, fragment):
    assert fragment in refused(lambda: P.check_request(proposal, P.parse_request(approve, reject, None, approve_unread)))


def test_the_confirmation_may_overlap_the_approval_without_naming_twice(proposal):
    P.check_request(proposal, P.parse_request("1,2", None, None, "1"))
    assert P.parse_request("1", None, None, "1").named == ("1",)


@pytest.mark.parametrize("given", ["K1", "k1", "0", "one", "1,,2", ""])
def test_the_confirmation_takes_numbers_and_no_candidate(given):
    refused(P.parse_request, "1", None, None, given)


def test_the_confirmation_stands_in_the_decision_beside_the_approval_and_is_read_back(document, proposal):
    decision = P.decision_document(plan_unread(document, unread(proposal, False), approve="1,2", approve_unread="1"), SHA, TODAY,
                                   P.sha256_of(CATALOG_BYTES), P.sha256_of(REJECTED_BYTES))
    assert (decision["approved"], decision["approved_unread"]) == ([1, 2], [1])
    assert P.check_decision(R.dump_json(decision)) == decision


def test_a_decision_of_the_old_form_without_the_field_counts_as_an_empty_confirmation(document, proposal):
    decision = decision_of(document, proposal, approve="1")
    del decision["approved_unread"]
    assert P.check_decision(R.dump_json(decision)) == decision


@pytest.mark.parametrize("damage", [
    lambda d: d.update(approved_unread=[1, 1]), lambda d: d.update(approved_unread=[3]), lambda d: d.update(approved_unread="1"),
    lambda d: d.update(approved_unread=[True]), lambda d: d.update(approved_unread=["1"]), lambda d: d.update(approved_unread=None),
], ids=["twice", "outside approved", "no list", "bool", "text", "null"])
def test_a_damaged_confirmation_in_a_decision_is_an_error(document, proposal, damage):
    decision = decision_of(document, proposal, approve="1,2")
    damage(decision)
    refused(P.check_decision, R.dump_json(decision))


def test_the_lists_of_a_decision_hold_numbers_first_then_candidates_each_in_numeric_order(document, proposal):
    result = plan(document, proposal, approve="5,1,3", reject="K3,6,K1,4")
    assert (result.approved, result.rejected) == ([1, 3, 5], [4, 6, "K1", "K3"])


# --- the decision file ------------------------------------------------------------------------------------------------


def decision_of(document, proposal, **lists):
    result = plan(document, proposal, **lists)
    return P.decision_document(result, SHA, TODAY, P.sha256_of(CATALOG_BYTES), P.sha256_of(REJECTED_BYTES))


def test_the_decision_carries_the_lists_both_bases_both_targets_with_their_checksums_and_the_first_state(document, proposal):
    decision = decision_of(document, proposal, approve="1", reject="4")
    assert set(decision) == {"proposal_sha", "approved", "approved_unread", "rejected", "admitted", "date", "base_catalog_sha256", "base_rejected_sha256", "result_catalog",
                             "result_rejected", "result_sha256", "result_rejected_sha256", "state", "commit"}
    assert (decision["state"], decision["commit"], decision["date"], decision["proposal_sha"]) == ("approved", None, TODAY, SHA)
    assert decision["result_sha256"] == P.sha256_of(decision["result_catalog"].encode()) and decision["result_rejected_sha256"] == P.sha256_of(decision["result_rejected"].encode())
    assert decision["base_catalog_sha256"] == P.sha256_of(CATALOG_BYTES) and (decision["approved"], decision["rejected"], decision["admitted"]) == ([1], [4], [])
    assert P.check_decision(R.dump_json(decision)) == decision


def test_the_decision_is_read_by_the_scanner_so_that_what_it_names_is_no_longer_carried(document, proposal):
    decision = decision_of(document, proposal, approve="1,2", reject="4,K1", admit="K2")
    rows = {m.hf_repo: m for f in build_catalog_v2(CATALOG_BYTES.decode(), Path("c.json")).families for m in f.models}
    changes, candidates, known, fields, accounts = R.carry_over(proposal, decision, rows, {"acme", "nova", "orbit"}, lambda repo: False)
    assert sorted(c["n"] for c in changes.values()) == [5, 6]  # 3 drops out by itself: the scanner carries no removal from a list
    assert [c["k"] for c in candidates.values()] == ["K3"]  # K1 rejected, K2 admitted: neither is carried
    assert "acme/Atlas-3-9B" in known and ("nova/Nimbus-1-30B", "gguf_repos") in fields and accounts == {"packthree", "newlab"}


@pytest.mark.parametrize("damage", [
    lambda d: d.update(state="done"), lambda d: d.update(approved="1"), lambda d: d.update(result_sha256="0" * 64), lambda d: d.update(proposal_sha="x"),
    lambda d: d.update(result_rejected=d["result_rejected"] + " "), lambda d: d.update(commit=7), lambda d: d.pop("base_rejected_sha256"),
    lambda d: d.update(approved=[True]), lambda d: d.update(approved=["1"]), lambda d: d.update(rejected=["x1"]), lambda d: d.update(admitted=[1]),
])
def test_a_damaged_decision_is_an_error_and_never_guessed_at(document, proposal, damage):
    decision = decision_of(document, proposal, approve="1")
    damage(decision)
    assert "damaged" in refused(P.check_decision, R.dump_json(decision))


@pytest.mark.parametrize("raw", [b"", b"[1]", b"not json", b"\xff\xfe", b'"approved"'])
def test_bytes_that_are_no_decision_are_an_error(raw):
    assert "damaged" in refused(P.check_decision, raw)


# --- --complete -------------------------------------------------------------------------------------------------------


def complete(document, proposal, admit="K3", kind="vision", today=TODAY):
    return P.complete_proposal(document, proposal, P.parse_request(None, None, admit), kind, today)


def test_complete_marks_the_new_row_and_every_carried_row_by_repo_and_size(document, proposal):
    """Codex kah-1 finding 1: a row `--complete` makes, and one taken over, get the `reading` mark from the name."""
    proposal["candidates"][2].update(hf_repo="acme/Ovie-512", parameters_b=None)
    proposal["changes"][0]["reading"] = False  # a wrong mark on a readable new row (change 1 of the fixture)
    data, number = complete(document, proposal)
    new = next(c for c in data["changes"] if c["n"] == number)
    assert new["reading"] is False
    assert all(("reading" in c) == bool(R.unread_of(c)) for c in data["changes"])  # every mark follows the name, the wrong one is gone


def test_a_candidate_of_open_kind_becomes_a_complete_row_with_the_kind_the_maintainer_named(document, proposal):
    data, number = complete(document, proposal)
    change = next(c for c in data["changes"] if c["n"] == number)
    new = {name: pair["new"] for name, pair in change["fields"].items()}
    assert set(change["fields"]) == set(R.ROW_FIELDS) and all(pair["old"] is None for pair in change["fields"].values())
    assert (new["kind"], new["kind_source"], new["hf_repo"], new["publisher"], new["added"], new["family"]) == ("vision", "decision", "acme/Odd-1-5B", "acme", TODAY, "odd")
    assert new["note"] == f"Kind vision set by the maintainer on {TODAY}." and new["hub_pipeline_tag"] is None and new["latest"] is False
    assert change["kind_evidence"] == {"stage": "decision", "value": "vision"} and change["stage"] == "A1" and change["open_since"] == proposal["candidates"][2]["open_since"]
    assert [c["candidate"] for c in data["candidates"]] == ["B1", "B2"] and [c["k"] for c in data["candidates"]] == ["K1", "K2"]
    assert [c["n"] for c in data["changes"]] == list(range(1, 8)) and data["summary"]["new"] == 2 and data["summary"]["candidates"] == 2
    assert R.check_changes(document, data["changes"])[1] == [] and proposal["candidates"][2]["k"] == "K3"  # the argument is not changed


def test_the_note_of_complete_has_exactly_the_shared_form(document, proposal):
    """The suite hides a note of this form (`MAINTAINER_KIND_NOTE`); what `--complete` writes must be exactly it."""
    data, number = complete(document, proposal)
    note = next(c for c in data["changes"] if c["n"] == number)["fields"]["note"]["new"]
    assert P.MAINTAINER_KIND_NOTE.fullmatch(note) is not None
    assert all(P.MAINTAINER_KIND_NOTE.fullmatch(f"Kind {kind} set by the maintainer on 2026-10-27.") for kind in P.KINDS)
    for off in ("Kind banana set by the maintainer on 2026-10-27.", f"{note} ", f" {note}", note[:-1], f"{note}\n"):
        assert P.MAINTAINER_KIND_NOTE.fullmatch(off) is None, off


def test_the_numbers_are_made_again_in_the_order_of_the_scanner_and_the_row_takes_its_place_in_it(document, proposal):
    data, number = complete(document, proposal)
    assert number == 2 and [(c["stage"], c["hf_repo"]) for c in data["changes"]][:3] == [("A1", "acme/Atlas-3-9B"), ("A1", "acme/Odd-1-5B"), ("A2", "nova/Nimbus-1-30B")]


def test_a_row_that_stands_over_the_latest_of_its_group_is_made_latest_with_the_publisher_page_as_its_source(document, proposal):
    candidate = proposal["candidates"][2]
    candidate.update(hf_repo="acme/Atlas-4-9B", parameters_b=9.0, hub_pipeline_tag=None)
    data, number = complete(document, proposal, kind="text")
    new = {name: pair["new"] for name, pair in next(c for c in data["changes"] if c["n"] == number)["fields"].items()}
    assert (new["latest"], new["latest_source"], new["latest_checked"], new["family"]) == (True, "https://huggingface.co/acme", TODAY, "atlas")


@pytest.mark.parametrize("task, kind", [("text-to-video", "video-generation"), ("text-generation", "text"), ("image-text-to-text", "vision")])
def test_a_candidate_whose_task_is_in_the_table_is_refused_with_the_kind_it_gives_on_its_own(document, proposal, task, kind):
    """The scanner reads the task first (`catalog_scan_rules.py`, stage 1), so it never makes a B3 of such a repo; a hand-changed
    `proposal.json` could. `--complete` is for a candidate without a table task: its kind is no decision to make."""
    proposal["candidates"][2]["hub_pipeline_tag"] = task
    before = json.dumps(proposal, sort_keys=True)
    message = refused(complete, document, proposal, "K3", "vision")
    assert f"acme/Odd-1-5B carries the task {task}, which gives kind {kind} on its own" in message
    assert "--complete is for a candidate without a table task" in message
    assert json.dumps(proposal, sort_keys=True) == before  # nothing is changed


def test_a_task_outside_the_table_does_not_stop_a_completion(document, proposal):
    proposal["candidates"][2]["hub_pipeline_tag"] = "any-to-any"
    data, number = complete(document, proposal)
    assert any(c["n"] == number for c in data["changes"])


@pytest.mark.parametrize("admit, kind, fragment", [
    ("K3", None, "eight kinds"), ("K3", "classification", "eight kinds"), ("K3", "Vision", "eight kinds"), ("K3,K2", "vision", "exactly one"),
    ("K1", "vision", "B3 only"), ("K2", "vision", "B3 only"), ("K9", "vision", "B3 only"),
])
def test_a_completion_the_brief_does_not_allow_is_refused(document, proposal, admit, kind, fragment):
    assert fragment in refused(complete, document, proposal, admit, kind)


def test_a_completion_with_other_lists_is_refused(document, proposal):
    request = P.parse_request("1", None, "K3")
    assert "exactly one" in refused(P.complete_proposal, document, proposal, request, "vision", TODAY)


def test_a_name_without_a_family_word_makes_no_row(document, proposal):
    proposal["candidates"][2]["hf_repo"] = "acme/7B"
    assert "no family word" in refused(complete, document, proposal)


def test_a_row_the_schema_would_not_take_is_refused_with_the_reason(document, proposal):
    proposal["candidates"][2]["hf_repo"] = "acme/Atlas-1-9B"  # a repo the table has already
    assert refused(complete, document, proposal)


# --- --show -----------------------------------------------------------------------------------------------------------


def test_the_text_of_the_open_proposal_is_the_rendering_of_exactly_these_bytes_with_their_checksum():
    text = P.render_open(PROPOSAL_BYTES)
    assert text == R.render_proposal(json.loads(PROPOSAL_BYTES), SHA) and text.endswith(f"proposal.json sha256: {SHA}\n")
    assert P.render_open(PROPOSAL_BYTES) == text


@pytest.mark.parametrize("raw", [b"", b"[]", b"{}", b'{"date": 1}', b"\xff"])
def test_bytes_that_are_no_proposal_cannot_be_shown(raw):
    assert "cannot be shown" in refused(P.render_open, raw)


def test_the_base_line_says_current_for_the_catalog_the_proposal_was_made_against_and_stale_for_any_other():
    assert P.base_line(PROPOSAL_BYTES, P.sha256_of(CATALOG_BYTES)) == (True, "base: current")
    current, line = P.base_line(PROPOSAL_BYTES, "0" * 64)
    assert not current and line == "base: stale, the proposal was made against another catalog; run the scanner again"
    assert P.base_line(b"[]", "0" * 64)[0] is False


def test_a_pending_publication_is_named_with_its_day_state_and_commit():
    assert P.pending_line("2026-10-03", {"state": "applied", "commit": None}) == "publication pending: 2026-10-03, state applied, commit none"
    assert P.pending_line("2026-10-03", {"state": "pushed", "commit": "c" * 40}).endswith(f"state pushed, commit {'c' * 40}")
