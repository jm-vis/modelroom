"""Tests for `scripts/catalog_scan_notes.py`: notes, sections and small rules of the catalog scanner (AP9-KA-H).

Pure functions over values: no network, no file, no clock. A row is a `SimpleNamespace` with the attributes the
functions read; where a rule needs a real `CatalogModel` (`latest_among`), the tests are in `test_catalog_scan_rules.py`.
"""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the scanner is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))
notes = importlib.import_module("catalog_scan_notes")
latest = importlib.import_module("catalog_scan_latest")
rules = importlib.import_module("catalog_scan_rules")


def row(repo: str, kind: str = "text", latest: bool = False, successor: str | None = None, size: float | None = None, family: str = "nova"):
    return SimpleNamespace(hf_repo=repo, kind=kind, latest=latest, successor=successor, parameters_b=size, family=family)


def new(*repos: str, kind: str = "text") -> list[tuple[str, str, float | None]]:
    return [(repo, kind, None) for repo in repos]


# --- K2 rule 1: the top of a group without a latest row ---------------------------------------------------------------


def test_a_single_new_member_of_a_group_without_a_latest_row_is_its_top():
    assert latest.group_tops(new("acme/Nova-1"), [row("acme/Nova-1")]) == ({"acme/Nova-1": "family top"}, [])


def test_the_higher_of_two_versions_without_a_date_is_the_top():
    rows = [row("acme/Nova-1"), row("acme/Nova-2")]
    assert latest.group_tops(new("acme/Nova-1", "acme/Nova-2"), rows) == ({"acme/Nova-2": "family top"}, [])


def test_two_members_of_the_same_version_stand_level_and_there_is_no_top():
    rows = [row("acme/Nova-1"), row("acme/Nova-1-Base")]
    tops, found = latest.group_tops(new("acme/Nova-1", "acme/Nova-1-Base"), rows)
    assert tops == {} and found == ["acme/nova text: no latest, acme/Nova-1 and acme/Nova-1-Base stand level"]


def test_dates_of_two_forms_stand_level_and_there_is_no_top():
    rows = [row("acme/Nova-1-2506"), row("acme/Nova-1-2025-05-01")]
    tops, found = latest.group_tops(new("acme/Nova-1-2506", "acme/Nova-1-2025-05-01"), rows)
    assert tops == {} and len(found) == 1 and "stand level" in found[0]


def test_a_tie_below_the_top_does_not_prevent_it_and_a_tie_at_the_top_does():
    below = [row("acme/Nova-1"), row("acme/Nova-1-Base"), row("acme/Nova-2")]
    assert latest.group_tops(new("acme/Nova-2"), below)[0] == {"acme/Nova-2": "family top"}  # versions 1, 1, 2
    above = [row("acme/Nova-2"), row("acme/Nova-2-Base"), row("acme/Nova-1")]
    tops, found = latest.group_tops(new("acme/Nova-2", "acme/Nova-2-Base"), above)  # versions 2, 2, 1
    assert tops == {} and found == ["acme/nova text: no latest, acme/Nova-2 and acme/Nova-2-Base stand level"]


def test_three_maximal_members_are_all_named():
    rows = [row("acme/Nova-1-A"), row("acme/Nova-1-B"), row("acme/Nova-1-C")]
    found = latest.group_tops(new("acme/Nova-1-A"), rows)[1]
    assert found == ["acme/nova text: no latest, acme/Nova-1-A, acme/Nova-1-B and acme/Nova-1-C stand level"]


def test_the_lines_of_one_family_are_groups_of_their_own():
    rows = [row("acme/Nova-Small-3"), row("acme/Nova-Large-1")]
    tops, found = latest.group_tops(new("acme/Nova-Large-1"), rows)  # the small line is no member of the large one's group
    assert tops == {"acme/Nova-Large-1": "family top"} and found == []


def test_another_publisher_and_another_kind_are_no_members_of_the_group():
    rows = [row("other/Nova-5"), row("acme/Nova-6", kind="vision"), row("acme/Nova-2")]
    assert latest.group_tops(new("acme/Nova-2"), rows) == ({"acme/Nova-2": "family top"}, [])


def test_a_shipped_top_stays_and_is_only_named_once_per_group():
    rows = [row("acme/Nova-3"), row("acme/Nova-2"), row("acme/Nova-1")]
    tops, found = latest.group_tops(new("acme/Nova-2", "acme/Nova-1"), rows)
    assert tops == {} and found == ["acme/nova text: no latest row; the top is the shipped row acme/Nova-3"]


def test_a_group_with_a_latest_row_is_left_to_the_rule_over_latest():
    rows = [row("acme/Nova-1", latest=True), row("acme/Nova-2")]
    assert latest.group_tops(new("acme/Nova-2"), rows) == ({}, [])


def test_a_group_without_a_new_member_is_not_looked_at():
    rows = [row("acme/Nova-1"), row("acme/Nova-1-Base")]
    assert latest.group_tops([], rows) == ({}, [])


def test_a_name_without_a_reading_is_no_member():
    assert latest.group_tops(new("acme/Nova-378"), [row("acme/Nova-378")]) == ({}, [])


def test_the_ltx_case_the_new_top_over_two_older_rows_and_two_new_ones_of_one_version():
    shipped = [row("Lightricks/LTX-2", kind="video-generation"), row("Lightricks/LTX-2.3", kind="video-generation")]
    top = [("Lightricks/LTX-2.5", "video-generation", None)]
    assert latest.group_tops(top, shipped + [row("Lightricks/LTX-2.5", kind="video-generation")]) == ({"Lightricks/LTX-2.5": "family top"}, [])
    both = top + [("Lightricks/LTX-2.5-Diffusers", "video-generation", None)]
    members = shipped + [row("Lightricks/LTX-2.5", kind="video-generation"), row("Lightricks/LTX-2.5-Diffusers", kind="video-generation")]
    tops, found = latest.group_tops(both, members)
    assert tops == {} and found == ["lightricks/ltx video-generation: no latest, Lightricks/LTX-2.5 and Lightricks/LTX-2.5-Diffusers stand level"]


# --- K2 rule 3: successors against each other ---------------------------------------------------------------------------


def resolve(edges: dict[str, str], pointed: dict[str, str] | None = None):
    return latest.resolve_successors(edges, pointed or {})


def test_a_chain_of_three_gives_the_last_link_only():
    chosen, found = resolve({"acme/A": "acme/B", "acme/B": "acme/C"})
    assert chosen == {"acme/B": "acme/C"} and found == ["acme/A: successor acme/B not set, acme/B has a successor"]


def test_a_chain_of_four_gives_the_first_and_the_last_link():
    chosen, _ = resolve({"acme/A": "acme/B", "acme/B": "acme/C", "acme/C": "acme/D"})
    assert chosen == {"acme/A": "acme/B", "acme/C": "acme/D"}


def test_a_chain_gives_the_same_result_whatever_the_names_do_to_the_order():
    """`zeta` -> `mid` -> `alpha`: the order of the names runs against the direction of the chain."""
    assert resolve({"acme/zeta": "acme/mid", "acme/mid": "acme/alpha"})[0] == {"acme/mid": "acme/alpha"}
    chosen, _ = resolve({"acme/zeta": "acme/mid", "acme/mid": "acme/alpha", "acme/alpha": "acme/aaa"})
    assert chosen == {"acme/zeta": "acme/mid", "acme/alpha": "acme/aaa"}
    assert resolve({"acme/a": "acme/b", "acme/b": "acme/c", "acme/c": "acme/d"})[0] == {"acme/a": "acme/b", "acme/c": "acme/d"}


def test_a_cycle_sets_nothing_and_every_edge_is_named():
    chosen, found = resolve({"acme/A": "acme/B", "acme/B": "acme/A"})
    assert chosen == {} and found == ["acme/A: successor acme/B not set, part of a cycle", "acme/B: successor acme/A not set, part of a cycle"]


def test_a_cycle_of_three_sets_nothing_and_names_three_edges():
    chosen, found = resolve({"zeta/A": "zeta/B", "zeta/B": "zeta/C", "zeta/C": "zeta/A"})
    assert chosen == {} and len(found) == 3 and all(note.endswith("part of a cycle") for note in found)


def test_a_cycle_does_not_take_a_candidate_that_only_leads_into_it_along():
    """The two edges of the cycle are struck, so `A` has no successor and `X -> A` stands on its own."""
    chosen, found = resolve({"acme/X": "acme/A", "acme/A": "acme/B", "acme/B": "acme/A"})
    assert chosen == {"acme/X": "acme/A"} and len(found) == 2


def test_two_chains_that_run_together_give_the_last_link_only():
    chosen, found = resolve({"acme/A": "acme/B", "acme/X": "acme/B", "acme/B": "acme/C"})
    assert chosen == {"acme/B": "acme/C"} and len(found) == 2 and all("acme/B has a successor" in note for note in found)


def test_a_row_another_row_points_at_gets_no_successor_and_the_note_names_that_row():
    chosen, found = resolve({"acme/B": "acme/C"}, pointed={"acme/B": "acme/A"})
    assert chosen == {} and found == ["acme/B: successor acme/C not set, acme/A points at acme/B"]


def test_candidates_that_do_not_meet_all_pass():
    assert resolve({"acme/A": "acme/Z", "acme/B": "acme/Y"}) == ({"acme/A": "acme/Z", "acme/B": "acme/Y"}, [])


# --- the end state ------------------------------------------------------------------------------------------------------


def test_the_end_state_holds_shipped_rows_new_rows_and_the_fields_of_field_changes():
    rows = {"acme/A": row("acme/A", latest=True), "acme/B": row("acme/B", latest=True)}
    changes = {
        ("new_row", "acme/N"): {"fields": {"latest": {"old": None, "new": True}, "successor": {"old": None, "new": None}}},
        ("field", "acme/A", "successor"): {"fields": {"successor": {"old": None, "new": "acme/N"}, "latest": {"old": True, "new": False}}},
        ("field", "acme/B", "gguf_repos"): {"fields": {"gguf_repos": {"old": [], "new": ["x/y"]}}},
    }
    assert latest.end_state(rows, changes) == {"acme/A": (False, "acme/N"), "acme/B": (True, None), "acme/N": (True, None)}


# --- successor refusals ---------------------------------------------------------------------------------------------------


def test_a_target_counts_only_as_a_latest_row_without_a_successor_at_the_end_of_the_run():
    source = SimpleNamespace(hf_repo="acme/A", publisher="acme", kind="text", successor=None)
    state = {"acme/B": (True, None), "acme/C": (False, None), "acme/D": (False, "acme/B")}
    assert latest.successor_refusal(source, "acme/B", "text", state) is None
    for target in ("acme/C", "acme/D"):
        assert latest.successor_refusal(source, target, "text", state) == (
            f"acme/A: successor {target} from the card is not a latest row at the end of this run, not set")
    assert "another kind" in latest.successor_refusal(source, "acme/B", "vision", state)
    assert "not a model of the table" in latest.successor_refusal(source, "other/B", "text", state)


# --- K3: reading drift ------------------------------------------------------------------------------------------------------


def test_a_row_with_another_family_than_its_name_makes_is_named_with_the_family_of_today():
    assert notes.drift_notes([row("acme/Nova-1", family="old")]) == ["acme/Nova-1: reading drift, family old in the catalog, nova today"]


def test_every_row_of_a_group_with_the_old_family_is_named_and_a_row_without_drift_is_not():
    rows = [row("acme/Nova-1", family="old"), row("acme/Nova-2", family="old"), row("acme/Nova-3", family="nova")]
    assert [n.split(":")[0] for n in notes.drift_notes(rows)] == ["acme/Nova-1", "acme/Nova-2"]


def test_a_name_without_a_reading_is_judged_by_its_first_run_of_letters():
    assert notes.drift_notes([row("apple/DFN5B-CLIP-ViT-H-14-378", family="dfn")]) == []
    assert notes.drift_notes([row("apple/DFN5B-CLIP-ViT-H-14-378", family="clip")]) == [
        "apple/DFN5B-CLIP-ViT-H-14-378: reading drift, family clip in the catalog, dfn today"]


def test_the_family_with_the_publisher_added_is_a_form_of_the_name_and_no_drift():
    rows = [row("acme/Nova-1", family="nova"), row("other/Nova-1", family="nova-other")]
    assert notes.drift_notes(rows) == []
    assert notes.drift_notes([row("acme/Nova-1", family="nova-acme"), row("other/Nova-1", family="nova-other")]) == []  # the owner of `nova` is gone
    assert notes.drift_notes([row("apple/DFN5B-CLIP-ViT-H-14-378", family="dfn-apple")]) == []  # the same for a name without a reading


def test_a_family_at_the_length_limit_is_no_drift():
    long_name = "a" * 80
    family = f"{long_name}-acme"[: rules.CATALOG_MAX_TEXT_LEN]
    assert notes.drift_notes([row(f"acme/{long_name}-1", family=family)]) == []
    assert notes.drift_notes([row(f"acme/{long_name}-1", family=long_name[: rules.CATALOG_MAX_TEXT_LEN])]) == []


def test_a_name_with_no_family_form_today_says_so():
    assert notes.drift_notes([row("acme/7B-foo", family="foo")]) == ["acme/7B-foo: reading drift, family foo in the catalog, no family form today"]


def test_a_lost_reading_with_the_same_family_is_not_drift():
    assert notes.drift_notes([row("kyutai/ovie-512", family="ovie")]) == []


def test_drift_notes_name_only_triples_the_last_run_does_not_hold_unchanged():
    rows = [row("acme/Nova-1", family="old"), row("acme/Nova-2", family="old")]
    same = [["acme/Nova-1", "old", ["nova", "nova-acme"]]]
    assert [n.split(":")[0] for n in notes.drift_notes(rows, same)] == ["acme/Nova-2"]  # one is held, one is new
    assert [n.split(":")[0] for n in notes.drift_notes(rows, [])] == ["acme/Nova-1", "acme/Nova-2"]  # an empty state holds nothing
    assert [n.split(":")[0] for n in notes.drift_notes(rows, None)] == ["acme/Nova-1", "acme/Nova-2"]  # no state: the first run
    assert [n.split(":")[0] for n in notes.drift_notes(rows, [["acme/Nova-1", "older", ["nova", "nova-acme"]]])] == ["acme/Nova-1", "acme/Nova-2"]
    assert [n.split(":")[0] for n in notes.drift_notes(rows, [["acme/Nova-1", "old", ["nova"]]])] == ["acme/Nova-1", "acme/Nova-2"]  # other forms


def test_last_drift_reads_the_list_a_proposal_kept_and_nothing_from_a_proposal_of_the_old_form():
    assert notes.last_drift({"drift": [["a/B-1", "x", ["b"]], "junk", ["too", "short"]]}) == [["a/B-1", "x", ["b"]]]
    assert notes.last_drift({}) is None and notes.last_drift(None) is None and notes.last_drift({"drift": "x"}) is None


def test_a_field_change_counts_as_new_only_when_it_is_new_or_differs_from_the_carried_one():
    old = {("field", "acme/A", "gguf_repos"): {"fields": {"gguf_repos": {"old": [], "new": ["x/a"]}}}}
    before = notes.field_values(old)
    same = {("field", "acme/A", "gguf_repos"): {"fields": {"gguf_repos": {"old": ["zz"], "new": ["x/a"]}}}, ("new_row", "acme/N"): {"fields": {}}}
    assert notes.new_field_repos(before, same) == set()  # only the old value moved: the change is the same
    more = {**same, ("field", "acme/A", "gguf_repos"): {"fields": {"gguf_repos": {"old": [], "new": ["x/a", "x/b"]}}}}
    assert notes.new_field_repos(before, more) == {"acme/A"}
    assert notes.new_field_repos(before, {**same, ("field", "acme/B", "successor"): {"fields": {"successor": {"old": None, "new": "acme/C"}}}}) == {"acme/B"}
    assert notes.new_field_repos({}, old) == {"acme/A"}


def test_group_notes_count_a_field_change_as_a_new_member_but_promote_nothing():
    rows = [row("acme/Nova-1"), row("acme/Nova-2")]
    assert latest.group_tops([], rows) == ({}, [])  # no new member: no note
    tops, found = latest.group_tops([], rows, touched={"acme/Nova-1"})
    assert tops == {} and found == ["acme/nova text: no latest row; the top is the shipped row acme/Nova-2"]


# --- K4: the time to decide on a candidate --------------------------------------------------------------------------------


def candidate(open_since: object, **more) -> dict:
    return {"candidate": "B1", "account": "newlab", "hf_repo": "newlab/X-1", "likes": 1, "open_since": open_since, "evidence": [], **more}


def test_a_candidate_is_overdue_from_day_31_and_not_on_day_30():
    day31, day30 = candidate("2026-09-03"), candidate("2026-09-04")  # today is 2026-10-04
    overdue, rest = notes.split_overdue([day31, day30], "2026-10-04")
    assert overdue == [{**day31, "overdue_days": 31}] and rest == [day30]


def test_split_overdue_changes_nothing_it_is_given():
    """A pure function (Codex kah-1 finding 6): the copies are marked, the candidates given stay as they are."""
    day31, day30, stale = candidate("2026-09-03"), candidate("2026-09-04"), candidate("2026-10-01", overdue_days=40)
    given = [day31, day30, stale]
    before = copy.deepcopy(given)
    overdue, rest = notes.split_overdue(given, "2026-10-04")
    assert given == before and given[0] is day31 and "overdue_days" not in day31 and stale["overdue_days"] == 40
    assert overdue[0] is not day31 and rest[0] is not day30 and "overdue_days" not in rest[1]  # the stale mark goes from the copy only
    overdue[0]["likes"] = 99
    assert day31["likes"] == 1


def test_the_constant_is_thirty_and_a_candidate_of_today_is_in_time():
    assert rules.CANDIDATE_DECISION_DAYS == 30
    assert notes.split_overdue([candidate("2026-10-04")], "2026-10-04")[0] == []


def test_a_mark_from_an_earlier_run_does_not_outlive_the_day_it_was_true():
    stale = candidate("2026-10-01", overdue_days=40)
    assert notes.split_overdue([stale], "2026-10-04") == ([], [candidate("2026-10-01")])


@pytest.mark.parametrize("bad", [None, "", "yesterday", 7, "2026-13-01"])
def test_an_open_since_that_is_no_date_is_never_overdue(bad):
    assert notes.split_overdue([candidate(bad)], "2026-10-04")[0] == []


def test_days_open_counts_whole_days():
    assert notes.days_open("2026-09-04", "2026-10-04") == 30 and notes.days_open("2026-10-05", "2026-10-04") == -1


# --- K1b: the section of rows without a reading --------------------------------------------------------------------------------


def change(n: int, repo: str, **more) -> dict:
    return {"n": n, "change": "new_row", "hf_repo": repo, "fields": {"parameters_b": {"old": None, "new": None}}, **more}


def render(changes: list[dict]) -> str:
    data = {"schema": 1, "date": "2026-10-04", "base_catalog_sha256": "b" * 64, "notes": [], "changes": changes, "candidates": [],
            "summary": {"new": len(changes), "changed": 0, "candidates": 0, "incomplete": 0, "pending_publication": None}}
    for c in changes:
        c.update({"stage": "A1", "kind": "text", "open_since": "2026-10-04", "evidence": []})
    return rules.render_proposal(data, "0" * 64)


SECTION = "## Rows without a reading (approve only with an entry in SHIPPED_WITHOUT_READING)"


def test_the_section_lists_the_unread_rows_by_repo_and_size_with_the_reason_and_the_line_for_the_exception_list():
    """Whatever the mark says (here: none, `true`, `false`): an old or wrong mark hides nothing (Codex kah-1 finding 1)."""
    text = render([change(1, "acme/Nova-7B"), change(2, "kyutai/ovie-512"), change(3, "m-a-p/MERT-v2-30s", reading=True), change(4, "acme/Ovie-378", reading=True)])
    section = text.split(SECTION)[1].split("## Changes")[0]
    assert "2 | kyutai/ovie-512 | reading: false | token '512' has no class" in section and '  "kyutai/ovie-512",  # token \'512\' has no class' in section
    assert "4 | acme/Ovie-378 | reading: false | token '378' has no class" in section and "Nova-7B" not in section and "MERT" not in section


def test_there_is_no_section_without_an_unread_new_row():
    assert SECTION not in render([change(1, "acme/Nova-7B")])
    assert SECTION not in render([{"n": 1, "change": "field", "field": "gguf_repos", "hf_repo": "kyutai/ovie-512", "reading": False, "fields": {}}])


def test_the_mark_is_made_from_repo_and_size_for_every_new_row_and_cleaned_where_it_is_wrong():
    assert rules.mark_reading(change(1, "kyutai/ovie-512"))["reading"] is False
    assert "reading" not in rules.mark_reading(change(1, "acme/Nova-7B", reading=False))  # a wrong mark goes
    assert rules.mark_reading(change(1, "kyutai/ovie-512", reading=True))["reading"] is False  # a wrong mark is made right
    field = {"change": "field", "hf_repo": "kyutai/ovie-512", "fields": {}}
    assert rules.mark_reading(dict(field)) == field


def test_the_note_of_a_row_without_a_reading_names_the_token_and_a_readable_name_gets_none():
    assert notes.unread_notes("kyutai/ovie-512", None, "ovie") == ["kyutai/ovie-512: no reading (token '512' has no class), family ovie taken from its first word"]
    assert notes.unread_notes("acme/Nova-7B", None, "nova") == []


# --- Codex kah-1 finding 2: the successor of the starting row at the end of the run ------------------------------------------------


def test_a_row_that_has_a_carried_successor_is_refused_a_second_one_from_its_card():
    source = SimpleNamespace(hf_repo="acme/A", publisher="acme", kind="text", successor=None, latest=False)
    state = {"acme/A": (False, "acme/B"), "acme/B": (True, None), "acme/C": (True, None)}
    assert "or the row has a successor" in latest.successor_refusal(source, "acme/C", "text", state)
    assert latest.successor_refusal(source, "acme/C", "text", {**state, "acme/A": (False, None)}) is None
