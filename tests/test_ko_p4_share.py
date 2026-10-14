"""NB: the daily package run gives every row it takes a fixed share of the stage's budget, reads the repositories of a
row publisher first, then listed packagers, then the rest, and hands what is left to the rows that were cut (V6). The Hub,
the source and the heads are recorded answers of a `FixtureTransport`; nothing opens a socket."""

from __future__ import annotations

import json
from datetime import timedelta

import fixture_ko_p4 as F
import catalog_scan_packages as K
from test_catalog_scan import NOW, TODAY, Hub
from test_ko_p4_stage import _only_first_row_due, checks_change, env, new_checks, run, world  # noqa: F401  (env is a fixture)

PHASE_DAYS = 24  # the bound of the multi-run test: 16 repositories, a reserve of 8 (four repositories) a run, one row of four a run, with room to spare
FIRST, SECOND, THIRD, FOURTH = F.ROWS  # acme/Atlas-1-9B, acme/Atlas-2-9B, nova/Nimbus-1-30B, orbit/Comet-3-8B


def repos_of(hub, row, owners, stem):
    """The repositories `owners` (one each) of `row`: a list in this order, a file list of one package, a head of one request."""
    names = [f"{owner}/{stem}-GGUF" for owner in owners]
    F.listing(hub, row, [F.entry(name, row, F.SHA_A) for name in names])
    for name in names:
        F.tree(hub, name, F.SHA_A, [(f"{stem}-Q8_0.gguf", 900)])
        F.header(hub, name, F.SHA_A, f"{stem}-Q8_0.gguf", F.KNOWN, redirect=False)
    return names


def trees(transport):
    return [url for _, url in transport.calls if "/tree/" in url]


def stage_requests(transport):
    """What the package stage sent: the source, the model answers, the lists, the file lists, the cards and the heads."""
    models = {K.model_url(row) for row in F.ROWS}
    return [url for _, url in transport.calls
            if url.startswith("https://api.github.com/") or "filter=gguf" in url or "/tree/" in url or "/resolve/" in url or url in models]


def two_pages(hub, row, repos):
    """The list of `row` over two pages, one half of `repos` on each."""
    half = max(1, len(repos) // 2)
    second = K.list_url(row) + "&cursor=2"
    F.put(hub, K.list_url(row), [F.entry(r, row, F.SHA_A) for r in repos[:half]], headers={"Link": f'<{second}>; rel="next"'})
    F.put(hub, second, [F.entry(r, row, F.SHA_A) for r in repos[half:]])


def asked(transport, url):
    return sum(1 for _, called in transport.calls if called == url)


# --- rule 1: a fixed share for every row ---------------------------------------------------------------------------------------


def test_a_row_with_many_repositories_cannot_use_the_budget_of_the_rows_behind_it(env):
    hub = Hub()
    world(hub)
    repos_of(hub, FIRST, [f"s{i}" for i in range(10)], "Atlas-1")
    for row, stem in ((SECOND, "Atlas-2"), (THIRD, "Nimbus"), (FOURTH, "Comet")):
        repos_of(hub, row, ["t0"], stem)
    budget = 2 + 4 * 8  # the source, then eight requests for each of the four rows
    proposal, result, transport = run(env, hub, package_budget=budget)
    assert set(env.state()["package_attempts"]) == set(F.ROWS)
    for row in (SECOND, THIRD, FOURTH):
        assert len(new_checks(proposal, row)) == 1
    assert 0 < len(new_checks(proposal, FIRST)) < 10
    assert result.exit_code == 0  # what the share cuts is no fault


def test_a_row_cut_at_its_share_is_tried_and_its_unread_repositories_come_first_next_time(env):
    hub = Hub()
    world(hub)
    repos = repos_of(hub, FIRST, [f"s{i}" for i in range(4)], "Atlas-1")
    budget = 2 + 7  # the source, then the model answer, the card, the list and two repositories of two requests each
    proposal, result, transport = run(env, hub, package_rows=1, package_budget=budget)
    assert env.state()["package_attempts"] == {FIRST: TODAY} and result.exit_code == 0
    assert sorted(env.state()["package_unit_attempts"]) == sorted(repos[:2])  # the one the share cut is not recorded
    assert [c["repo"] for c in new_checks(proposal, FIRST)] == sorted(repos[:2])
    _only_first_row_due(env, (NOW + timedelta(days=1)).date().isoformat())
    proposal, _, transport = run(env, hub, now=NOW + timedelta(days=1), package_rows=1, package_budget=budget)
    assert trees(transport)[:2] == [K.tree_url(repo, F.SHA_A) for repo in repos[2:]]
    assert [c["repo"] for c in new_checks(proposal, FIRST)] == sorted(repos)


def test_every_repository_of_a_row_is_read_in_a_finite_number_of_runs_whatever_the_share(env):  # V6
    hub = Hub()
    world(hub)
    repos = repos_of(hub, FIRST, [f"s{i}" for i in range(7)], "Atlas-1")
    seen = []
    for day in range(7):
        moment = NOW + timedelta(days=day)
        if day:
            _only_first_row_due(env, moment.date().isoformat())
        proposal, _, _ = run(env, hub, now=moment, package_rows=1, package_budget=2 + 7)
        seen.append(len(new_checks(proposal, FIRST) or []))
    assert seen[-1] == 7 and sorted(c["repo"] for c in new_checks(proposal, FIRST)) == sorted(repos)
    assert seen == sorted(seen)  # nothing read is lost on the way


def test_a_run_whose_budget_cannot_give_each_row_the_minimum_takes_fewer_rows_and_says_so(env):
    hub = Hub()
    world(hub)
    rest = K.RESERVE_MIN + K.MIN_ROW_SHARE * 3 + K.MIN_ROW_SHARE - 1  # the reserve, three rows get the minimum, the fourth would not
    proposal, result, transport = run(env, hub, package_budget=2 + rest)
    assert len(env.state()["package_attempts"]) == 3 and result.exit_code == 1
    assert any("budget ended" in note and "3 of 4" in note for note in proposal["notes"])
    assert sum(1 for url in (u for _, u in transport.calls) if "filter=gguf" in url) == 3


# --- rule 2: the publisher's repositories, then the listed packagers', then the others ---------------------------------------


def test_the_repositories_of_a_row_are_read_publisher_first_then_listed_packagers_then_the_rest(env):
    hub = Hub()
    world(hub)
    names = repos_of(hub, SECOND, ["stranger", "other", "packone", "acme", "Packtwo"], "Atlas-2")
    _, _, transport = run(env, hub)
    expected = [names[3], names[2], names[4], names[0], names[1]]  # acme; packone, Packtwo; stranger, other (the list's order)
    assert [u for u in trees(transport)] == [K.tree_url(repo, F.SHA_A) for repo in expected]


def test_an_unread_repository_of_another_class_still_comes_before_one_read_yesterday(env):  # V6: the class decides among equals only
    hub = Hub()
    world(hub)
    names = repos_of(hub, FIRST, ["acme", "stranger"], "Atlas-1")
    run(env, hub, package_rows=1, package_budget=2 + 4)  # the source, the model answer, the card, the list: the share ends in the first tree
    state = env.state()
    state["package_unit_attempts"] = {names[0]: (NOW - timedelta(days=1)).date().isoformat()}
    (env.out / "scan-state.json").write_bytes(json.dumps(state).encode("utf-8"))
    _only_first_row_due(env, (NOW + timedelta(days=1)).date().isoformat())
    _, _, transport = run(env, hub, now=NOW + timedelta(days=1), package_rows=1)
    assert trees(transport)[0] == K.tree_url(names[1], F.SHA_A)


# --- rule 3: what is left goes to the rows that were cut -------------------------------------------------------------------------


def test_what_the_first_round_leaves_goes_to_the_rows_that_were_cut_without_asking_the_card_or_the_list_again(env):
    hub = Hub()
    world(hub)
    repos_of(hub, FIRST, [f"s{i}" for i in range(6)], "Atlas-1")
    budget = 2 + 2 * 8  # two rows: 16 requests, one kept back, share 7 each
    proposal, result, transport = run(env, hub, package_rows=2, package_budget=budget)
    assert len(new_checks(proposal, FIRST)) == 4  # one in the share of the row, three more from what the quiet row and the reserve left
    assert asked(transport, K.list_url(FIRST)) == 1 and asked(transport, K.model_url(FIRST)) == 1
    assert result.exit_code == 0 and len(stage_requests(transport)) <= budget and result.main_used <= 300
    assert set(env.state()["package_attempts"]) == {FIRST, SECOND}
    evidence = checks_change(proposal, FIRST)["evidence"]
    assert sum(1 for e in evidence if "/s" in e and e.startswith("https://huggingface.co/")) == len(new_checks(proposal, FIRST))  # both rounds


def test_a_budget_the_first_round_spends_whole_leaves_nothing_to_a_second_round(env):
    hub = Hub()
    world(hub)
    repos_of(hub, FIRST, [f"s{i}" for i in range(6)], "Atlas-1")
    repos_of(hub, SECOND, [f"t{i}" for i in range(6)], "Atlas-2")
    _, result, transport = run(env, hub, package_rows=2, package_budget=2 + 2 * 7)
    assert len(trees(transport)) <= 2 * 3 + 2 and len(stage_requests(transport)) <= 2 + 2 * 7 and result.main_used <= 300


# --- the first review of NB: the rest is shared over runs, a cut list is continued, evidence of both rounds ------------------------


def test_every_unit_of_every_row_is_reached_with_the_least_share_competing_rows_and_a_list_of_two_pages(env):  # V6
    hub = Hub()
    world(hub)
    wanted = []
    for row, stem in zip(F.ROWS, ("Atlas-1", "Atlas-2", "Nimbus", "Comet")):
        names = [f"{owner}/{stem}-GGUF" for owner in ("s0", "s1", "s2", "s3")]
        two_pages(hub, row, names)
        for name in names:
            F.tree(hub, name, F.SHA_A, [(f"{stem}-Q8_0.gguf", 900)])
            F.header(hub, name, F.SHA_A, f"{stem}-Q8_0.gguf", F.KNOWN, redirect=False)
        wanted += names
    for day in range(40):
        run(env, hub, now=NOW + timedelta(days=day), package_rows=4, package_budget=2 + 20)  # share 4: model, card, two list pages
        if set(wanted) <= set(env.state().get("package_unit_attempts", {})):
            break
    assert set(wanted) <= set(env.state()["package_unit_attempts"])
    assert day < 12  # the rest goes round by round, not to the first row for ever


def test_the_reserve_is_never_nil_so_spent_minimum_shares_still_reach_every_unit_and_new_packages_of_visited_repositories(env):  # V6
    hub = Hub()
    world(hub)
    stems = dict(zip(F.ROWS, ("Atlas-1", "Atlas-2", "Nimbus", "Comet")))
    repos = {}
    for row, stem in stems.items():
        repos[row] = [f"{owner}/{stem}-GGUF" for owner in ("s0", "s1", "s2", "s3")]
        two_pages(hub, row, repos[row])
        for name in repos[row]:
            F.tree(hub, name, F.SHA_A, [(f"{stem}-Q8_0.gguf", 900)])
            F.header(hub, name, F.SHA_A, f"{stem}-Q8_0.gguf", F.KNOWN, redirect=False)
    budget = 2 + 16  # four minimum shares of four would take it whole: the reserve (8) takes two rows' worth, so two rows a run

    def files_proposed(proposal):
        return {(row, c["repo"], c["file"]) for row in F.ROWS for c in (new_checks(proposal, row) or [])}

    def wanted(files_of):
        return {(row, name, f"{stems[row]}-{quant}.gguf") for row in F.ROWS for name in repos[row] for quant in files_of}

    day = 0
    for day in range(PHASE_DAYS):  # phase 1: every package of every repository, none of them seen before
        proposal, _, _ = run(env, hub, now=NOW + timedelta(days=day), package_rows=4, package_budget=budget)
        if wanted(["Q8_0"]) <= files_proposed(proposal):
            break
    assert wanted(["Q8_0"]) <= files_proposed(proposal), f"not reached within {PHASE_DAYS} runs"
    first = day
    for row, stem in stems.items():  # phase 2: every repository, already visited, gets a new package
        for name in repos[row]:
            F.tree(hub, name, F.SHA_A, [(f"{stem}-Q8_0.gguf", 900), (f"{stem}-Q4_K_M.gguf", 500)])
            F.header(hub, name, F.SHA_A, f"{stem}-Q4_K_M.gguf", F.KNOWN, redirect=False)
    for day in range(first + 1, first + 1 + PHASE_DAYS):
        proposal, _, _ = run(env, hub, now=NOW + timedelta(days=day), package_rows=4, package_budget=budget)
        if wanted(["Q8_0", "Q4_K_M"]) <= files_proposed(proposal):
            break
    assert wanted(["Q8_0", "Q4_K_M"]) <= files_proposed(proposal), f"not reached within {PHASE_DAYS} runs"


def _stored_for(env, row, checks):
    document = json.loads(env.catalog.read_bytes())
    for model in document["models"]:
        if model["hf_repo"] == row:
            model["gguf_checks"] = checks
    env.catalog.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def test_a_leftover_of_the_reserve_without_progress_books_no_reserve_day_and_the_row_keeps_its_priority(env):  # V6
    from test_ko_p4_stage import check_dict

    hub = Hub()
    world(hub)
    a_repos = [f"s{i}/Atlas-1-GGUF" for i in range(8)]
    F.listing(hub, FIRST, [F.entry(r, FIRST, F.SHA_A) for r in a_repos])
    for repo in a_repos:
        F.tree(hub, repo, F.SHA_A, [("Atlas-1-Q8_0.gguf", 900)])
    _stored_for(env, FIRST, [check_dict(repo=r, identity="Atlas-1-Q8_0.gguf", file="Atlas-1-Q8_0.gguf", result="present", architecture=F.KNOWN) for r in a_repos])
    b_repo = "t0/Atlas-2-GGUF"
    F.listing(hub, SECOND, [F.entry(b_repo, SECOND, F.SHA_A)])
    F.tree(hub, b_repo, F.SHA_A, [("Atlas-2-Q8_0.gguf", 900), ("Atlas-2-Q4_K_M.gguf", 500)])
    F.header(hub, b_repo, F.SHA_A, "Atlas-2-Q4_K_M.gguf", F.KNOWN, redirect=False)
    _stored_for(env, SECOND, [check_dict(repo=b_repo, identity="Atlas-2-Q8_0.gguf", file="Atlas-2-Q8_0.gguf", result="present", architecture=F.KNOWN)])
    run(env, hub, package_rows=2, package_budget=2 + 16)
    booked = env.state().get("package_reserve_days", {})
    assert FIRST in booked and SECOND not in booked  # the eight repositories A read to their end are progress, B's one request is none
    state = env.state()  # the next day takes A and B again: the two other rows were tried that day
    state["package_attempts"].update({row: (NOW + timedelta(days=1)).date().isoformat() for row in (THIRD, FOURTH)})
    state["package_attempts"].pop(FIRST), state["package_attempts"].pop(SECOND)
    (env.out / "scan-state.json").write_bytes(json.dumps(state).encode("utf-8"))
    proposal, _, _ = run(env, hub, now=NOW + timedelta(days=1), package_rows=2, package_budget=2 + 16)
    assert any(c["file"] == "Atlas-2-Q4_K_M.gguf" for c in new_checks(proposal, SECOND) or [])


def _long_list(hub, pages):  # the heads are redirected to the CDN, as the Hub does: two requests each
    urls = [K.list_url(FIRST)] + [K.list_url(FIRST) + f"&cursor={i}" for i in range(1, pages)]
    repos = [f"s{i}/Atlas-1-GGUF" for i in range(pages)]
    for i, url in enumerate(urls):
        F.put(hub, url, [F.entry(repos[i], FIRST, F.SHA_A)], headers={"Link": f'<{urls[i + 1]}>; rel="next"'} if i + 1 < len(urls) else None)
        F.tree(hub, repos[i], F.SHA_A, [("Atlas-1-Q8_0.gguf", 900)])
        F.header(hub, repos[i], F.SHA_A, "Atlas-1-Q8_0.gguf", F.KNOWN, redirect=True)
    return repos


def test_the_limit_of_a_long_list_just_inside_is_reached_and_just_beyond_is_the_documented_stand_still_without_a_fault(env):
    # budget 2 + 16, two rows: share 4 (model, card, two pages), reserve 8 plus the 1 the quiet row leaves: the other pages, one
    # file list and one head (two requests, redirected) must fit, so a list of eight pages is the last that is reached, one of nine
    # is not (CONTRACTS, V6)
    hub = Hub()
    world(hub)
    _long_list(hub, 8)
    proposal, result, _ = run(env, hub, package_rows=2, package_budget=2 + 16)
    assert len(new_checks(proposal, FIRST) or []) >= 1 and result.exit_code == 0


def test_a_long_list_just_beyond_the_limit_is_the_documented_stand_still_without_a_fault(env):
    hub = Hub()
    world(hub)
    _long_list(hub, 9)
    for day in range(3):
        proposal, result, _ = run(env, hub, now=NOW + timedelta(days=30 + day), package_rows=2, package_budget=2 + 16)
        assert result.exit_code == 0 and not (new_checks(proposal, FIRST) or [])  # no check, no fault, no crash: the same each day
    assert FIRST in env.state()["package_attempts"]


def test_a_list_the_share_cut_is_continued_in_the_second_round_not_read_again_from_its_first_page(env):
    hub = Hub()
    world(hub)
    repos = [f"s{i}/Atlas-1-GGUF" for i in range(3)]
    pages = [K.list_url(FIRST), K.list_url(FIRST) + "&cursor=2", K.list_url(FIRST) + "&cursor=3"]
    for i, url in enumerate(pages):
        F.put(hub, url, [F.entry(repos[i], FIRST, F.SHA_A)], headers={"Link": f'<{pages[i + 1]}>; rel="next"'} if i + 1 < len(pages) else None)
        F.tree(hub, repos[i], F.SHA_A, [("Atlas-1-9B-Q8_0.gguf", 900)])
        F.header(hub, repos[i], F.SHA_A, "Atlas-1-9B-Q8_0.gguf", F.KNOWN, redirect=False)
    proposal, _, transport = run(env, hub, package_rows=2, package_budget=2 + 9)  # share 4 each: the list is cut after its second page
    assert [asked(transport, url) for url in pages] == [1, 1, 1]
