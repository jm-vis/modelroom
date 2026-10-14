"""P4, part 2: stage 1 of the catalog run end to end -- which repositories are asked, what a package gets as a check,
and what a run that fails changes (AP9-KO, V1, V5, V6). The Hub, the source and the file heads are recorded answers of a
`FixtureTransport`; nothing opens a socket. Accounts, repositories and architectures are invented, the llama.cpp source
is the recorded original."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone

import pytest

import fixture_ko_p4 as F
import catalog_scan_packages as K
from test_catalog_scan import NOW, TODAY, Env, Hub, scanner

from modelroom.catalog import CatalogModel
from modelroom.http import Response

ROW = "acme/Atlas-2-9B"
OTHER = "stranger/Atlas-2-9B-GGUF"  # an account that is neither the publisher nor a listed packager
GGUF_BASE = "Atlas-2-9B"
V3 = "catalog-v3"


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    document = json.loads(e.catalog.read_bytes())
    document["schema_version"] = 3
    e.catalog.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return e


def run(env, hub, *, now=NOW, **kw):
    transport = F.Recording(hub.m)
    kw.setdefault("packages", True)
    result = scanner.scan(env.catalog, env.rejected, env.out, transport=transport, now=now, budget=kw.pop("budget", 300), sub_budget=30, **kw)
    day = now.astimezone(timezone.utc).date().isoformat()
    proposal = json.loads((env.out / day / "proposal.json").read_bytes()) if (env.out / day / "proposal.json").exists() else None
    return proposal, result, transport


def checks_change(proposal, repo=ROW):
    return next((c for c in proposal["changes"] if c["change"] == "field" and c["field"] == "gguf_checks" and c["hf_repo"] == repo), None)


def new_checks(proposal, repo=ROW):
    change = checks_change(proposal, repo)
    return None if change is None else change["fields"]["gguf_checks"]["new"]


def world(hub):
    F.quiet_world(hub)


def one_repo(hub, *, files=None, hub_arch=None, total=None, heads=None, sha=F.SHA_A, repo=OTHER):
    """A repository of an unknown account that quantizes ROW, with its tree and its heads."""
    files = files or [("Atlas-2-9B-Q8_0.gguf", 900), ("Atlas-2-9B-Q4_K_M.gguf", 500)]
    F.listing(hub, ROW, [F.entry(repo, ROW, sha, architecture=hub_arch, total=total)])
    F.tree(hub, repo, sha, files)
    for name, arch in (heads or {}).items():
        F.header(hub, repo, sha, name, arch)


def stored(env, checks):
    document = json.loads(env.catalog.read_bytes())
    for row in document["models"]:
        if row["hf_repo"] == ROW:
            row["gguf_checks"] = checks
    env.catalog.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def check_dict(**over):
    base = {
        "repo": OTHER, "revision": F.SHA_A, "identity": "Atlas-2-9B-Q8_0.gguf", "file": "Atlas-2-9B-Q8_0.gguf", "quantization": "Q8_0",
        "architecture": F.UNKNOWN, "architecture_source": "file_header", "result": "absent", "llamacpp_commit": "c" * 40,
        "source_url": "https://github.com/ggml-org/llama.cpp/blob/" + "c" * 40 + "/src/llama-arch.cpp", "checked": "2026-09-01", "same_quantization": 1,
    }
    base.update(over)
    return base


# --- the input: every row, whoever owns the repository ---------------------------------------------------------------------


def test_the_gguf_repositories_of_a_row_are_found_by_the_relation_whoever_owns_them_and_the_account_stays_unlisted(env):
    hub = Hub()
    world(hub)
    one_repo(hub, hub_arch=F.UNKNOWN, total=900, heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    proposal, result, transport = run(env, hub)
    checks = new_checks(proposal)
    assert [(c["file"], c["result"], c["architecture_source"]) for c in checks] == [
        ("Atlas-2-9B-Q4_K_M.gguf", "present", "file_header"), ("Atlas-2-9B-Q8_0.gguf", "absent", "hub_field")]
    assert not [c for c in proposal["changes"] if c.get("field") == "gguf_repos"] and not [c for c in proposal["candidates"]]
    assert "stranger" not in json.dumps(proposal["candidates"])
    assert json.loads(env.catalog.read_bytes())["packagers"] == ["packone", "packtwo"]


def test_a_row_that_is_not_latest_is_asked_too_and_a_row_of_another_kind_is_not(env):
    hub = Hub()
    world(hub)
    _, _, transport = run(env, hub)
    asked = [url for _, url in transport.calls if "filter=gguf" in url]
    assert sorted(asked) == sorted(K.list_url(row) for row in F.ROWS) and "acme/Atlas-1-9B" in " ".join(asked).replace("%2F", "/")
    assert not any("Comet-Embed" in url or "Comet-Echo" in url for _, url in transport.calls)


def test_a_check_is_bound_to_repository_revision_file_source_and_day(env):
    hub = Hub()
    world(hub)
    one_repo(hub, hub_arch=F.UNKNOWN, total=900, heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    proposal, _, _ = run(env, hub)
    q8 = next(c for c in new_checks(proposal) if c["quantization"] == "Q8_0")
    assert q8 == check_dict(checked=TODAY, llamacpp_commit=F.LLAMA_COMMIT, source_url=f"https://github.com/ggml-org/llama.cpp/blob/{F.LLAMA_COMMIT}/src/llama-arch.cpp",
                            architecture_source="hub_field", same_quantization=1)
    change = checks_change(proposal)
    assert (change["stage"], change["field"], change["fields"]["gguf_checks"]["old"]) == ("A9", "gguf_checks", [])
    assert change["evidence"] and all(e.startswith("https://") for e in change["evidence"])


def test_a_proposal_with_checks_still_validates_as_a_schema_3_catalog(env):
    from modelroom.catalog_schema import build_catalog

    hub = Hub()
    world(hub)
    one_repo(hub, hub_arch=F.UNKNOWN, total=900, heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    proposal, _, _ = run(env, hub)
    rules = importlib.import_module("catalog_scan_rules")
    built = build_catalog(json.dumps(rules.apply_changes(json.loads(env.catalog.read_bytes()), proposal["changes"])), env.catalog)
    row = next(m for f in built.families for m in f.models if m.hf_repo == ROW)
    assert len(row.gguf_checks) == 2


# --- the package: the Hub's field for one file, the head for every other (V1) -----------------------------------------------


def test_the_hub_field_proves_only_the_file_of_its_size_and_every_other_package_is_read_by_its_head(env):
    hub = Hub()
    world(hub)
    one_repo(hub, hub_arch=F.UNKNOWN, total=900, heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    _, _, transport = run(env, hub)
    heads = [url for _, url in transport.calls if "/resolve/" in url]
    assert [u for u in heads if "README" not in u] == [K.header_url(OTHER, F.SHA_A, "Atlas-2-9B-Q4_K_M.gguf")]


def test_the_head_is_a_range_request_on_the_hub_and_follows_one_redirect_to_the_hub_cdn(env):
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q4_K_M.gguf", 500)], heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    _, _, transport = run(env, hub)
    index = [i for i, (_, url) in enumerate(transport.calls) if url == K.header_url(OTHER, F.SHA_A, "Atlas-2-9B-Q4_K_M.gguf")]
    assert len(index) == 1
    assert transport.headers[index[0]] == {"Range": "bytes=0-65535"} and transport.headers[index[0] + 1] == {"Range": "bytes=0-65535"}
    assert transport.calls[index[0] + 1][1].startswith("https://us.aws.cdn.hf.co/")
    assert transport.limits[index[0]] == 65536


def test_a_file_the_hub_field_does_not_describe_gets_no_check_from_the_field(env):
    hub = Hub()
    world(hub)
    # the field states a size no file has: the scope cannot be shown, so both packages are read by their heads
    one_repo(hub, hub_arch=F.UNKNOWN, total=12345, heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN, "Atlas-2-9B-Q8_0.gguf": F.KNOWN})
    proposal, _, _ = run(env, hub)
    assert {c["architecture_source"] for c in new_checks(proposal)} == {"file_header"} and {c["result"] for c in new_checks(proposal)} == {"present"}


def test_a_shard_set_is_read_at_its_first_shard_only(env):
    hub = Hub()
    world(hub)
    files = [("Q8_0/Atlas-2-9B-Q8_0-00001-of-00002.gguf", 900), ("Q8_0/Atlas-2-9B-Q8_0-00002-of-00002.gguf", 800)]
    one_repo(hub, files=files, heads={"Q8_0/Atlas-2-9B-Q8_0-00001-of-00002.gguf": F.UNKNOWN})
    proposal, _, transport = run(env, hub)
    [check] = new_checks(proposal)
    assert (check["file"], check["identity"], check["result"]) == ("Q8_0/Atlas-2-9B-Q8_0-00001-of-00002.gguf", "Q8_0/Atlas-2-9B-Q8_0.gguf", "absent")
    assert not any("-00002-of-" in url for _, url in transport.calls)


def test_an_importance_matrix_next_to_the_models_is_no_package_and_cannot_stand_for_one(env):
    hub = Hub()
    world(hub)
    files = [("Atlas-2-9B-Q8_0.gguf", 900), ("imatrix-atlas.gguf", 300)]
    # the field describes the imatrix file (size 300): it proves no package, so the model file is read
    one_repo(hub, files=files, hub_arch="imatrix", total=300, heads={"Atlas-2-9B-Q8_0.gguf": F.KNOWN})
    proposal, _, _ = run(env, hub)
    [check] = new_checks(proposal)
    assert (check["file"], check["architecture"], check["architecture_source"]) == ("Atlas-2-9B-Q8_0.gguf", F.KNOWN, "file_header")


def test_two_packages_of_one_identity_get_no_check_because_the_identity_cannot_tell_them_apart(env):
    hub = Hub()
    world(hub)
    files = [("Atlas-2-9B-Q8_0-00001-of-00002.gguf", 900), ("Atlas-2-9B-Q8_0-00001-of-00003.gguf", 800), ("Atlas-2-9B-Q4_K_M.gguf", 500)]
    one_repo(hub, files=files, heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    proposal, _, _ = run(env, hub)
    assert [c["file"] for c in new_checks(proposal)] == ["Atlas-2-9B-Q4_K_M.gguf"]


def test_same_quantization_counts_the_packages_of_the_repository_at_that_revision(env):
    hub = Hub()
    world(hub)
    files = [("Atlas-2-9B-Q8_0.gguf", 900), ("Atlas-2-9B.Q8_0.gguf", 899)]
    one_repo(hub, files=files, heads={"Atlas-2-9B-Q8_0.gguf": F.KNOWN, "Atlas-2-9B.Q8_0.gguf": F.KNOWN})
    proposal, _, _ = run(env, hub)
    assert [c["same_quantization"] for c in new_checks(proposal)] == [2, 2]


def test_a_repository_that_quantizes_two_models_or_none_of_the_row_is_not_taken(env):
    hub = Hub()
    world(hub)
    wrong = [
        F.entry("x/Merge-GGUF", ROW, F.SHA_A, tags=F.tags_of(ROW, "acme/Atlas-1-9B")),
        F.entry("x/Other-GGUF", ROW, F.SHA_A, tags=F.tags_of("acme/Atlas-1-9B")),
        F.entry("x/NoGguf", ROW, F.SHA_A, tags=F.tags_of(ROW, gguf=False)),
        F.entry("rejectedaccount/Atlas-GGUF", ROW, F.SHA_A),
        F.entry("x/BadSha", ROW, "not-a-sha"),
    ]
    F.listing(hub, ROW, wrong)
    rules = json.loads(env.rejected.read_bytes())
    rules.append({"account": "rejectedaccount"})
    env.rejected.write_bytes(json.dumps(rules).encode("utf-8"))
    proposal, _, transport = run(env, hub)
    assert checks_change(proposal) is None and not any("/tree/" in url for _, url in transport.calls)


def test_a_repo_is_compared_by_the_hub_id_without_case(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict(repo=OTHER.upper().replace("STRANGER", "Stranger"), architecture=F.UNKNOWN)])
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], heads={"Atlas-2-9B-Q8_0.gguf": F.UNKNOWN})
    proposal, _, _ = run(env, hub)
    assert checks_change(proposal) is None or len(new_checks(proposal)) == 1


# --- the failure rules (V5) --------------------------------------------------------------------------------------------------


def broken_source(hub, how):
    if how == "unreachable":
        F.put(hub, F.GITHUB_COMMIT_URL, b"", status=503)
    elif how == "redirect":
        F.put(hub, F.GITHUB_COMMIT_URL, b"", status=301, headers={"Location": "https://api.github.com/elsewhere"})
    elif how == "not a commit":
        F.put(hub, F.GITHUB_COMMIT_URL, b"<html>rate limited</html>")
    elif how == "too long":
        F.put(hub, F.GITHUB_COMMIT_URL, b"a" * 300)
    elif how == "block missing":
        F.put(hub, F.GITHUB_BLOCK_URL, b"int main() {}\n")
    elif how == "block cut":
        F.put(hub, F.GITHUB_BLOCK_URL, F.LLAMA_TEXT[: len(F.LLAMA_TEXT) // 2])
    elif how == "lookup changed":
        F.put(hub, F.GITHUB_BLOCK_URL, F.LLAMA_TEXT.replace(b"kv.second == name", b"kv.first == name"))
    elif how == "no utf8":
        F.put(hub, F.GITHUB_BLOCK_URL, b"\xff\xfe" + F.LLAMA_TEXT)
    elif how == "block redirected":
        F.put(hub, F.GITHUB_BLOCK_URL, b"", status=302, headers={"Location": "https://api.github.com/raw"})
    elif how == "block too long":
        F.put(hub, F.GITHUB_BLOCK_URL, b"x" * (512 * 1024 + 1))
    elif how == "block 404":
        F.put(hub, F.GITHUB_BLOCK_URL, b"{}", status=404)


@pytest.mark.parametrize("how", ["unreachable", "redirect", "not a commit", "too long", "block missing", "block cut", "lookup changed", "no utf8", "block redirected", "block too long", "block 404"])
def test_a_source_that_cannot_be_read_whole_changes_nothing_adds_no_absent_and_removes_no_check(env, how):
    hub = Hub()
    world(hub)
    stored(env, [check_dict(), check_dict(identity="Atlas-2-9B-Q4_K_M.gguf", file="Atlas-2-9B-Q4_K_M.gguf", quantization="Q4_K_M", result="present", architecture=F.KNOWN)])
    one_repo(hub, hub_arch=F.UNKNOWN, total=900, sha=F.SHA_B, heads={"Atlas-2-9B-Q4_K_M.gguf": F.UNKNOWN})
    broken_source(hub, how)
    before = env.catalog.read_bytes()
    proposal, result, transport = run(env, hub)
    assert result.exit_code == 1 and checks_change(proposal) is None
    assert env.catalog.read_bytes() == before
    assert not any("/tree/" in url or "/resolve/" in url and "README" not in url for _, url in transport.calls)
    assert any("llama.cpp" in note and "no check was added, changed or removed" in note for note in proposal["notes"])
    assert "package_attempts" not in env.state() or env.state()["package_attempts"] == {}


def test_a_source_fault_is_a_fault_of_the_run_in_the_exit_code_but_the_rest_of_the_proposal_is_written(env):
    hub = Hub()
    world(hub)
    broken_source(hub, "unreachable")
    proposal, result, _ = run(env, hub)
    assert result.exit_code == 1 and proposal["changes"] == []


def test_the_source_is_asked_twice_with_its_media_types_and_no_redirect_is_followed(env):
    hub = Hub()
    world(hub)
    _, _, transport = run(env, hub)
    github = [(i, url) for i, (_, url) in enumerate(transport.calls) if url.startswith("https://api.github.com/")]
    assert [url for _, url in github] == [F.GITHUB_COMMIT_URL, F.GITHUB_BLOCK_URL]
    assert [transport.headers[i] for i, _ in github] == [{"Accept": "application/vnd.github.sha"}, {"Accept": "application/vnd.github.raw+json"}]
    assert [transport.limits[i] for i, _ in github] == [256, 512 * 1024]


def test_a_head_that_fails_leaves_the_package_without_a_check_and_the_others_keep_theirs(env):
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900), ("Atlas-2-9B-Q4_K_M.gguf", 500), ("Atlas-2-9B-Q6_K.gguf", 700)],
             heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    F.put(hub, K.header_url(OTHER, F.SHA_A, "Atlas-2-9B-Q8_0.gguf"), b"", status=500)
    F.header(hub, OTHER, F.SHA_A, "Atlas-2-9B-Q6_K.gguf", None, raw=b"GGUF-but-no-header")
    proposal, _, _ = run(env, hub)
    assert [c["file"] for c in new_checks(proposal)] == ["Atlas-2-9B-Q4_K_M.gguf"]


def test_a_head_redirected_off_the_hub_is_refused_and_the_package_stays_unknown(env):
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)])
    F.put(hub, K.header_url(OTHER, F.SHA_A, "Atlas-2-9B-Q8_0.gguf"), b"", status=302, headers={"Location": "https://evil.example.org/x.gguf"})
    F.put(hub, "https://evil.example.org/x.gguf", F.gguf_head(F.UNKNOWN), status=206)
    proposal, _, transport = run(env, hub)
    assert checks_change(proposal) is None and ("GET", "https://evil.example.org/x.gguf") not in transport.calls


def test_a_head_answered_whole_by_a_server_that_ignores_the_range_is_refused(env):
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)])
    F.put(hub, K.header_url(OTHER, F.SHA_A, "Atlas-2-9B-Q8_0.gguf"), F.gguf_head(F.UNKNOWN) + b"\0" * 70000)
    proposal, _, _ = run(env, hub)
    assert checks_change(proposal) is None


def test_a_file_list_that_fails_or_goes_on_over_a_second_page_gives_no_check_for_that_repository(env):
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], heads={"Atlas-2-9B-Q8_0.gguf": F.UNKNOWN})
    F.put(hub, K.tree_url(OTHER, F.SHA_A), [{"type": "file", "path": "Atlas-2-9B-Q8_0.gguf", "size": 900}], headers={"Link": '<https://huggingface.co/api/models/x/tree/y?cursor=2>; rel="next"'})
    proposal, _, _ = run(env, hub)
    assert checks_change(proposal) is None
    F.put(hub, K.tree_url(OTHER, F.SHA_A), b"", status=500)
    proposal, _, _ = run(env, hub, now=NOW + timedelta(days=1))
    assert checks_change(proposal) is None


# --- set, keep, replace, remove --------------------------------------------------------------------------------------------


def test_a_check_that_the_hub_and_the_source_still_agree_with_is_kept_unchanged_and_its_file_is_not_read_again(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict()])
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)])
    proposal, _, transport = run(env, hub)
    assert checks_change(proposal) is None
    assert not any("Atlas-2-9B-Q8_0.gguf" in url for _, url in transport.calls)


def test_a_stored_absent_becomes_present_when_a_newer_source_knows_the_architecture_and_the_revision_is_the_stored_one(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict()])
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)])
    new_commit = "d" * 40
    F.source(hub, F.llama_with(F.UNKNOWN), new_commit)
    proposal, _, transport = run(env, hub)
    [check] = new_checks(proposal)
    assert (check["result"], check["llamacpp_commit"], check["checked"], check["revision"]) == ("present", new_commit, TODAY, F.SHA_A)
    assert checks_change(proposal)["fields"]["gguf_checks"]["old"][0]["result"] == "absent"
    assert not any("/resolve/" in url and "README" not in url for _, url in transport.calls)


def test_a_package_that_has_no_check_yet_is_read_although_another_package_of_the_repository_has_one(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict()])
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900), ("Atlas-2-9B-Q4_K_M.gguf", 500)], heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    proposal, _, transport = run(env, hub)
    assert sorted(c["file"] for c in new_checks(proposal)) == ["Atlas-2-9B-Q4_K_M.gguf", "Atlas-2-9B-Q8_0.gguf"]
    assert not any("Atlas-2-9B-Q8_0.gguf" in url for _, url in transport.calls)


def test_a_moved_revision_is_read_again_and_replaces_the_old_check_of_the_package(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict()])
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], sha=F.SHA_B, hub_arch=F.KNOWN, total=900)
    proposal, _, _ = run(env, hub)
    [check] = new_checks(proposal)
    assert (check["revision"], check["result"], check["architecture"], check["checked"]) == (F.SHA_B, "present", F.KNOWN, TODAY)


def test_a_package_that_the_repository_no_longer_holds_at_the_new_revision_is_removed_after_a_complete_read(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict(), check_dict(identity="Atlas-2-9B-Q4_K_M.gguf", file="Atlas-2-9B-Q4_K_M.gguf", quantization="Q4_K_M")])
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], sha=F.SHA_B, hub_arch=F.UNKNOWN, total=900)
    proposal, _, _ = run(env, hub)
    assert [c["identity"] for c in new_checks(proposal)] == ["Atlas-2-9B-Q8_0.gguf"]


def test_nothing_is_removed_when_the_file_list_could_not_be_read(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict()])
    one_repo(hub, files=[], sha=F.SHA_B)
    F.put(hub, K.tree_url(OTHER, F.SHA_B), b"", status=500)
    proposal, _, _ = run(env, hub)
    assert checks_change(proposal) is None


def test_a_check_of_an_earlier_day_never_replaces_a_later_one(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict(checked="2026-10-20", revision=F.SHA_B, result="present", architecture=F.KNOWN)])
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], sha=F.SHA_A, hub_arch=F.UNKNOWN, total=900)
    proposal, _, _ = run(env, hub)
    assert checks_change(proposal) is None


def test_an_open_proposal_of_an_earlier_day_is_the_base_of_the_next_one_so_nothing_proposed_is_lost(env):
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], hub_arch=F.UNKNOWN, total=900)
    first, _, _ = run(env, hub)
    assert len(new_checks(first)) == 1
    hub2 = Hub()
    world(hub2)
    one_repo(hub2, repo="other2/Atlas-2-9B-GGUF", files=[("Atlas-2-9B-Q6_K.gguf", 700)], hub_arch=F.KNOWN, total=700)
    second, _, _ = run(env, hub2, now=NOW + timedelta(days=1))
    assert sorted(c["repo"] for c in new_checks(second)) == ["other2/Atlas-2-9B-GGUF", OTHER]


def test_the_working_catalog_in_schema_2_gets_no_stage_and_no_request(env):
    document = json.loads(env.catalog.read_bytes())
    document["schema_version"] = 2
    env.catalog.write_bytes(json.dumps(document).encode("utf-8"))
    hub = Hub()
    world(hub)
    proposal, result, transport = run(env, hub)
    assert not any("api.github.com" in url for _, url in transport.calls)
    assert any("schema 2" in note and "package" in note for note in proposal["notes"]) and result.exit_code == 0


def test_without_the_switch_the_stage_asks_nothing(env):
    hub = Hub()
    world(hub)
    _, _, transport = run(env, hub, packages=False)
    assert not any("api.github.com" in url or "filter=gguf" in url for _, url in transport.calls)


# --- budget and continuation (V6) ------------------------------------------------------------------------------------------


def test_every_try_is_recorded_for_its_row_whether_it_found_a_repository_a_check_or_an_error(env):
    hub = Hub()
    world(hub)
    F.put(hub, K.list_url(F.ROWS[0]), b"", status=500)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], hub_arch=F.UNKNOWN, total=900)
    _, _, _ = run(env, hub)
    attempts = env.state()["package_attempts"]
    assert attempts == {row: TODAY for row in F.ROWS}


def test_a_run_takes_at_most_the_rows_it_may_and_the_next_one_takes_those_it_did_not_reach(env):
    hub = Hub()
    world(hub)
    _, _, first = run(env, hub, package_rows=2)
    asked = [url for _, url in first.calls if "filter=gguf" in url]
    assert len(asked) == 2 and len(env.state()["package_attempts"]) == 2
    _, _, second = run(env, hub, now=NOW + timedelta(days=1), package_rows=2)
    again = [url for _, url in second.calls if "filter=gguf" in url]
    assert len(again) == 2 and not set(asked) & set(again)
    assert len(env.state()["package_attempts"]) == 4


def test_a_row_tried_today_is_not_tried_again_today(env):
    hub = Hub()
    world(hub)
    run(env, hub)
    _, _, second = run(env, hub)
    assert not any("filter=gguf" in url for _, url in second.calls)


def test_a_budget_that_ends_leaves_the_rows_it_did_not_reach_untried_and_the_run_reports_it(env):
    hub = Hub()
    world(hub)
    proposal, result, transport = run(env, hub, package_budget=4)
    reached = env.state().get("package_attempts", {})
    assert len(reached) < 4 and result.exit_code == 1
    assert any("budget ended" in note and "package" in note for note in proposal["notes"])
    assert result.sub_used <= 30


def test_the_requests_of_the_stage_are_booked_in_the_main_budget_too(env):
    hub = Hub()
    world(hub)
    proposal, result, transport = run(env, hub, budget=3)
    assert result.main_used <= 3 and result.exit_code == 1


def test_the_state_keeps_the_attempts_of_a_run_that_does_not_run_the_stage(env):
    hub = Hub()
    world(hub)
    run(env, hub)
    kept = env.state()["package_attempts"]
    _, _, _ = run(env, hub, now=NOW + timedelta(days=1), packages=False)
    assert env.state()["package_attempts"] == kept


def test_the_old_state_without_attempts_reads_as_never_tried(env):
    env.put_state({})
    hub = Hub()
    world(hub)
    _, result, _ = run(env, hub)
    assert result.exit_code == 0 and len(env.state()["package_attempts"]) == 4


def test_no_cap_of_its_own_keeps_a_package_unknown_the_budget_alone_limits_the_run(env):
    hub = Hub()
    world(hub)
    quants = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "IQ4_XS", "Q4_0", "Q4_1", "Q5_K_S"]
    files = [(f"Atlas-2-9B-{q}.gguf", 100 + i) for i, q in enumerate(quants)]
    one_repo(hub, files=files, heads={name: F.KNOWN for name, _ in files})
    proposal, _, _ = run(env, hub)
    assert len(new_checks(proposal)) == 10


def _more_repos(hub, count, row=ROW):
    repos = [f"s{i}/Atlas-2-9B-GGUF" for i in range(count)]
    for repo in repos:
        F.tree(hub, repo, F.SHA_A, [("Atlas-2-9B-Q8_0.gguf", 900)])
        F.header(hub, repo, F.SHA_A, "Atlas-2-9B-Q8_0.gguf", F.KNOWN)
    return repos


def test_a_list_of_more_than_one_page_is_followed_to_its_end(env):
    hub = Hub()
    world(hub)
    repos = _more_repos(hub, 12)
    second = K.list_url(ROW) + "&cursor=2"
    F.put(hub, K.list_url(ROW), [F.entry(r, ROW, F.SHA_A) for r in repos[:6]], headers={"Link": f'<{second}>; rel="next"'})
    F.put(hub, second, [F.entry(r, ROW, F.SHA_A) for r in repos[6:]])
    proposal, _, _ = run(env, hub)
    assert len(new_checks(proposal)) == 12


def test_a_next_page_that_leaves_the_hub_api_is_a_fault_of_the_row(env):
    hub = Hub()
    world(hub)
    repos = _more_repos(hub, 1)
    F.put(hub, K.list_url(ROW), [F.entry(repos[0], ROW, F.SHA_A)], headers={"Link": '<https://evil.example.org/api/models?x=1>; rel="next"'})
    proposal, _, transport = run(env, hub)
    assert ("GET", "https://evil.example.org/api/models?x=1") not in transport.calls
    assert any(n.startswith(f"{ROW}: package checks:") and "leaves" not in n for n in proposal["notes"])


def test_repositories_without_any_check_are_read_before_those_that_have_one(env):
    hub = Hub()
    world(hub)
    stored(env, [check_dict(repo="s0/Atlas-2-9B-GGUF")])
    repos = _more_repos(hub, 2)
    F.listing(hub, ROW, [F.entry(r, ROW, F.SHA_A) for r in repos])
    _, _, transport = run(env, hub)
    trees = [url for _, url in transport.calls if "/tree/" in url]
    assert trees[0] == K.tree_url("s1/Atlas-2-9B-GGUF", F.SHA_A)


def test_a_row_whose_budget_ended_after_some_progress_is_tried_and_keeps_what_it_read(env):
    hub = Hub()
    world(hub)
    repos = _more_repos(hub, 2)
    first = F.ROWS[0]
    F.listing(hub, first, [F.entry(r, first, F.SHA_A) for r in repos])
    for repo in repos:
        F.tree(hub, repo, F.SHA_A, [("Atlas-1-9B-Q8_0.gguf", 900)])
        F.header(hub, repo, F.SHA_A, "Atlas-1-9B-Q8_0.gguf", F.KNOWN)
    proposal, result, _ = run(env, hub, package_rows=1, package_budget=9)
    assert result.exit_code == 0 and env.state()["package_attempts"] == {first: TODAY}  # the row's share ended: no fault (NB)
    assert len(new_checks(proposal, first)) == 1


# --- the second review round ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [{"type": "file", "path": "Atlas-2-9B-Q8_0.gguf", "size": None}, {"type": "file", "path": "Atlas-2-9B-Q8_0.gguf", "size": "9"},
                                 {"type": "file", "path": "Atlas-2-9B-Q8_0.gguf", "size": True}, {"type": "file", "size": 9}, {"type": "blob", "path": "x", "size": 1}, "text"])
def test_a_file_list_with_an_entry_nobody_understands_is_a_fault_and_deletes_nothing(env, bad):
    hub = Hub()
    world(hub)
    stored(env, [check_dict()])
    one_repo(hub, files=[("Atlas-2-9B-Q4_K_M.gguf", 500)], sha=F.SHA_B, heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    F.put(hub, K.tree_url(OTHER, F.SHA_B), [{"type": "file", "path": "Atlas-2-9B-Q4_K_M.gguf", "size": 500}, bad])
    proposal, result, _ = run(env, hub)
    assert checks_change(proposal) is None and result.exit_code == 1


def test_a_directory_entry_in_the_file_list_is_no_fault(env):
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], hub_arch=F.UNKNOWN, total=900)
    F.put(hub, K.tree_url(OTHER, F.SHA_A), [{"type": "directory", "path": "docs"}, {"type": "file", "path": "Atlas-2-9B-Q8_0.gguf", "size": 900}])
    proposal, _, _ = run(env, hub)
    assert len(new_checks(proposal)) == 1


def test_two_runs_on_one_day_keep_the_proposals_of_both(env):
    hub = Hub()
    world(hub)
    nimbus = "nova/Nimbus-1-30B"
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], hub_arch=F.UNKNOWN, total=900)
    F.listing(hub, nimbus, [F.entry("s9/Nimbus-GGUF", nimbus, F.SHA_B, architecture=F.UNKNOWN, total=800)])
    F.tree(hub, "s9/Nimbus-GGUF", F.SHA_B, [("Nimbus-Q8_0.gguf", 800)])
    first, _, _ = run(env, hub, package_rows=2)  # the two rows of the account `acme`
    assert checks_change(first) is not None and checks_change(first, nimbus) is None
    second, _, _ = run(env, hub, package_rows=2)
    assert checks_change(second) is not None and checks_change(second, nimbus) is not None


def test_a_relation_tag_in_another_case_still_names_the_row(env):
    hub = Hub()
    world(hub)
    tags = ["gguf", f"base_model:{ROW}", f"base_model:quantized:{ROW.upper()}"]
    F.listing(hub, ROW, [F.entry(OTHER, ROW, F.SHA_A, tags=tags)])
    F.tree(hub, OTHER, F.SHA_A, [("Atlas-2-9B-Q8_0.gguf", 900)])
    F.header(hub, OTHER, F.SHA_A, "Atlas-2-9B-Q8_0.gguf", F.UNKNOWN)
    proposal, _, _ = run(env, hub)
    assert len(new_checks(proposal)) == 1


def test_a_rejected_account_is_rejected_in_any_case(env):
    hub = Hub()
    world(hub)
    rejected = json.loads(env.rejected.read_bytes()) + [{"account": "RejectedAccount"}]
    env.rejected.write_bytes(json.dumps(rejected).encode("utf-8"))
    F.listing(hub, ROW, [F.entry("rejectedaccount/Atlas-GGUF", ROW, F.SHA_A)])
    proposal, _, transport = run(env, hub)
    assert checks_change(proposal) is None and not any("/tree/" in url for _, url in transport.calls)


def test_the_attempts_of_one_row_in_two_spellings_are_one_entry_with_the_younger_day():
    from datetime import date

    found = scanner._attempts_of({"package_attempts": {"acme/atlas-2-9b": "2026-10-01", "acme/Atlas-2-9B": "2026-10-03"}})
    assert found == {"acme/Atlas-2-9B": date(2026, 10, 3)}


def test_a_head_from_another_revision_than_the_one_asked_for_gives_no_check(env):
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)])
    F.put(hub, K.header_url(OTHER, F.SHA_A, "Atlas-2-9B-Q8_0.gguf"), F.gguf_head(F.UNKNOWN), status=206, headers={"X-Repo-Commit": F.SHA_B})
    proposal, _, _ = run(env, hub)
    assert checks_change(proposal) is None
    F.put(hub, K.header_url(OTHER, F.SHA_A, "Atlas-2-9B-Q8_0.gguf"), F.gguf_head(F.UNKNOWN), status=206, headers={"X-Repo-Commit": F.SHA_A})
    proposal, _, _ = run(env, hub, now=NOW + timedelta(days=1))
    assert len(new_checks(proposal)) == 1


def test_a_catalog_row_cannot_hold_more_checks_than_the_schema_allows(env):
    hub = Hub()
    world(hub)
    many = [check_dict(identity=f"Atlas-2-9B-X{i}-Q8_0.gguf", file=f"Atlas-2-9B-X{i}-Q8_0.gguf", repo=f"o{i}/Atlas-GGUF") for i in range(200)]
    stored(env, many)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], hub_arch=F.UNKNOWN, total=900)
    proposal, result, _ = run(env, hub)
    assert checks_change(proposal) is None or len(new_checks(proposal)) <= 200


# --- the third review round: every unit is reached, a carried proposal sets nothing back, ids in any spelling ---------------------


def _merge_state(env, **keys):
    state = env.state()
    state.update(keys)
    (env.out / "scan-state.json").write_bytes(json.dumps(state).encode("utf-8"))


def _only_first_row_due(env, day: str) -> None:
    """Every row but the first was tried on `day`, so a run of that day that takes one row takes the first again."""
    attempts = env.state().get("package_attempts", {})
    attempts.update({row: day for row in F.ROWS[1:]})
    _merge_state(env, package_attempts=attempts)


def test_a_repository_that_fails_every_day_does_not_hold_back_the_others_of_its_row(env):
    hub = Hub()
    world(hub)
    first, repos = F.ROWS[0], ["s0/Atlas-1-GGUF", "s1/Atlas-1-GGUF"]
    F.listing(hub, first, [F.entry(r, first, F.SHA_A) for r in repos])
    F.put(hub, K.tree_url(repos[0], F.SHA_A), b"", status=500)
    F.tree(hub, repos[1], F.SHA_A, [("Atlas-1-9B-Q8_0.gguf", 900)])
    F.header(hub, repos[1], F.SHA_A, "Atlas-1-9B-Q8_0.gguf", F.KNOWN, redirect=False)
    budget = 2 + 2 + 1 + 1 + 1  # the source, the card, the list, one file list, one head
    for day in range(3):
        moment = NOW + timedelta(days=day)
        if day:
            _only_first_row_due(env, moment.date().isoformat())
        proposal, _, _ = run(env, hub, now=moment, package_rows=1, package_budget=budget)
    assert [c["repo"] for c in new_checks(proposal, first)] == [repos[1]]


def test_a_package_whose_head_fails_every_day_does_not_hold_back_the_others_of_its_repository(env):
    hub = Hub()
    world(hub)
    first, repo = F.ROWS[0], "s0/Atlas-1-GGUF"
    files = [("Atlas-1-9B-Q2_K.gguf", 100), ("Atlas-1-9B-Q8_0.gguf", 900)]
    F.listing(hub, first, [F.entry(repo, first, F.SHA_A)])
    F.tree(hub, repo, F.SHA_A, files)
    F.put(hub, K.header_url(repo, F.SHA_A, files[0][0]), b"", status=500)
    F.header(hub, repo, F.SHA_A, files[1][0], F.KNOWN, redirect=False)
    budget = 2 + 2 + 1 + 1 + 1  # the source, the card, the list, the file list, one head
    for day in range(3):
        moment = NOW + timedelta(days=day)
        if day:
            _only_first_row_due(env, moment.date().isoformat())
        proposal, _, _ = run(env, hub, now=moment, package_rows=1, package_budget=budget)
    assert [c["file"] for c in new_checks(proposal, first)] == [files[1][0]]


def test_a_row_whose_list_is_longer_than_the_budget_does_not_hold_back_the_rows_behind_it(env):
    hub = Hub()
    world(hub)
    first = F.ROWS[0]
    pages = [K.list_url(first)] + [K.list_url(first) + f"&cursor={i}" for i in range(1, 4)]
    for i, url in enumerate(pages):
        F.put(hub, url, [F.entry(f"s{i}/Atlas-1-GGUF", first, F.SHA_A)], headers={"Link": f'<{pages[i + 1]}>; rel="next"'} if i + 1 < len(pages) else None)
    budget = 2 + 2 + 2  # the source, the card, two of the four pages
    run(env, hub, package_rows=1, package_budget=budget)
    assert env.state()["package_attempts"] == {first: TODAY}
    _, _, second = run(env, hub, now=NOW + timedelta(days=1), package_rows=1, package_budget=budget)
    assert ("GET", K.list_url(F.ROWS[1])) in second.calls


def test_a_row_whose_budget_ended_in_its_list_behind_another_row_stays_untried_and_is_read_whole_next(env):
    hub = Hub()
    world(hub)
    repos = _more_repos(hub, 2)
    following = K.list_url(ROW) + "&cursor=2"
    F.put(hub, K.list_url(ROW), [F.entry(repos[0], ROW, F.SHA_A)], headers={"Link": f'<{following}>; rel="next"'})
    F.put(hub, following, [F.entry(repos[1], ROW, F.SHA_A)])
    run(env, hub, package_rows=2, package_budget=2 + 3 + 2 + 1)  # the source, the first row whole, the card and one page of ROW
    assert env.state()["package_attempts"] == {F.ROWS[0]: TODAY}
    proposal, _, _ = run(env, hub, now=NOW + timedelta(days=1), package_rows=1)
    assert len(new_checks(proposal)) == 2


def _absent_proposed(env, hub):
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900)], hub_arch=F.UNKNOWN, total=900)
    first, _, _ = run(env, hub)
    assert [c["result"] for c in new_checks(first)] == ["absent"]
    return first


@pytest.mark.parametrize("same_day", [False, True])
def test_a_carried_proposal_does_not_set_back_a_newer_check_of_the_catalog(env, same_day):
    hub = Hub()
    world(hub)
    _absent_proposed(env, hub)
    later = NOW + timedelta(days=2)
    if same_day:  # the older absent is in the proposal of the day before the catalog changes
        run(env, hub, now=later, package_rows=0)
    stored(env, [check_dict(checked=(NOW + timedelta(days=1)).date().isoformat(), result="present", architecture=F.KNOWN)])
    proposal, _, _ = run(env, hub, now=later, package_rows=0)
    assert checks_change(proposal) is None


def test_a_carried_removal_does_not_remove_a_check_the_catalog_holds_newer(env):
    stored(env, [check_dict()])
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q4_K_M.gguf", 500)], sha=F.SHA_B, heads={"Atlas-2-9B-Q4_K_M.gguf": F.KNOWN})
    first, _, _ = run(env, hub)
    assert [c["file"] for c in new_checks(first)] == ["Atlas-2-9B-Q4_K_M.gguf"]
    newer = check_dict(checked=(NOW + timedelta(days=1)).date().isoformat(), result="present", architecture=F.KNOWN)
    stored(env, [newer])
    proposal, _, _ = run(env, hub, now=NOW + timedelta(days=2), package_rows=0)
    assert newer in new_checks(proposal) and "Atlas-2-9B-Q4_K_M.gguf" in [c["file"] for c in new_checks(proposal)]


@pytest.mark.parametrize("same_day", [False, True])
@pytest.mark.parametrize("spelling", [ROW, ROW.upper()])
def test_a_model_rejected_between_two_runs_loses_its_package_proposals(env, same_day, spelling):
    hub = Hub()
    world(hub)
    _absent_proposed(env, hub)
    rejected = json.loads(env.rejected.read_bytes()) + [{"date": TODAY, "hf_repo": spelling}]
    env.rejected.write_bytes(json.dumps(rejected).encode("utf-8"))
    proposal, _, _ = run(env, hub, now=NOW + timedelta(days=0 if same_day else 1))
    assert checks_change(proposal) is None


@pytest.mark.parametrize("same_day", [False, True])
def test_a_row_whose_spelling_changed_keeps_its_open_package_proposals(env, same_day):
    hub = Hub()
    world(hub)
    first = _absent_proposed(env, hub)
    document = json.loads(env.catalog.read_bytes())
    for row in document["models"]:
        if row["hf_repo"] == ROW:
            row["hf_repo"] = ROW.lower()
    env.catalog.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    hub.put(scanner.R.b2_url(ROW.lower()), [])  # the row's own B2 page in its new spelling
    proposal, _, _ = run(env, hub, now=NOW + timedelta(days=0 if same_day else 1), package_rows=0)
    change = checks_change(proposal, ROW.lower())
    assert change is not None and change["fields"]["gguf_checks"]["new"] == new_checks(first)


# --- the fourth review round ---------------------------------------------------------------------------------------------------------


def test_two_packages_whose_names_differ_only_in_case_are_two_units(env):
    hub = Hub()
    world(hub)
    first, repo = F.ROWS[0], "s0/Atlas-1-GGUF"
    files = [("Atlas-1-9B-Q8_0.gguf", 900), ("atlas-1-9B-Q8_0.gguf", 901)]
    F.listing(hub, first, [F.entry(repo, first, F.SHA_A)])
    F.tree(hub, repo, F.SHA_A, files)
    F.put(hub, K.header_url(repo, F.SHA_A, files[0][0]), b"", status=500)
    F.header(hub, repo, F.SHA_A, files[1][0], F.KNOWN, redirect=False)
    budget = 2 + 2 + 1 + 1 + 1  # the source, the card, the list, the file list, one head
    for day in range(2):
        moment = NOW + timedelta(days=day)
        if day:
            _only_first_row_due(env, moment.date().isoformat())
        proposal, _, _ = run(env, hub, now=moment, package_rows=1, package_budget=budget)
    assert [c["file"] for c in new_checks(proposal, first)] == [files[1][0]]
    assert f"{repo}:{files[0][0]}" in env.state()["package_unit_attempts"]


@pytest.mark.parametrize("same_day", [False, True])
def test_the_checks_of_a_repository_whose_account_was_rejected_since_in_another_case_are_not_carried(env, same_day):
    hub = Hub()
    world(hub)
    _absent_proposed(env, hub)
    rejected = json.loads(env.rejected.read_bytes()) + [{"account": OTHER.split("/")[0].upper()}]
    env.rejected.write_bytes(json.dumps(rejected).encode("utf-8"))
    proposal, _, _ = run(env, hub, now=NOW + timedelta(days=0 if same_day else 1))
    assert checks_change(proposal) is None


def test_a_row_for_which_no_request_was_sent_is_not_tried(env):
    hub = Hub()
    world(hub)
    proposal, result, _ = run(env, hub, package_budget=2)  # the source takes the whole budget
    assert "package_attempts" not in env.state() and result.exit_code == 1


@pytest.mark.parametrize("location", ["https://huggingface.co:bad/x", "https://[::1/x"])
def test_a_malformed_redirect_address_fails_the_package_only_and_keeps_the_stored_check(env, location):
    stored(env, [check_dict()])
    hub = Hub()
    world(hub)
    one_repo(hub, files=[("Atlas-2-9B-Q8_0.gguf", 900), ("Atlas-2-9B-Q4_K_M.gguf", 500)])
    F.put(hub, K.header_url(OTHER, F.SHA_A, "Atlas-2-9B-Q4_K_M.gguf"), b"", status=302, headers={"Location": location})
    proposal, result, transport = run(env, hub)
    assert result.exit_code == 1 and checks_change(proposal) is None
    assert not any(":bad" in url or "[::1" in url for _, url in transport.calls)
    assert env.state()["package_attempts"]  # the run ended normally, with its state


def test_a_malformed_next_page_address_is_a_fault_of_the_row_and_not_of_the_run(env):
    hub = Hub()
    world(hub)
    repos = _more_repos(hub, 1)
    F.put(hub, K.list_url(ROW), [F.entry(repos[0], ROW, F.SHA_A)], headers={"Link": '<https://[::1/x>; rel="next"'})
    proposal, result, _ = run(env, hub)
    assert result.exit_code == 1 and checks_change(proposal) is None
    assert any(n.startswith(f"{ROW}: package checks:") for n in proposal["notes"]) and len(env.state()["package_attempts"]) == 4


# --- the command line ---------------------------------------------------------------------------------------------------------------


def _main(env, monkeypatch, *flags):
    hub = Hub()
    world(hub)
    made = []

    def fake(**kwargs):
        made.append(kwargs)
        return F.Recording(hub.m)

    monkeypatch.setattr(scanner, "UrllibTransport", fake)
    code = scanner.main(["--catalog", str(env.catalog), "--rejected", str(env.rejected), "--out", str(env.out), "--today", TODAY, *flags])
    return code, made


def test_the_switch_turns_the_stage_on_and_hands_the_scanner_transport_its_own_hosts(env, monkeypatch):
    code, made = _main(env, monkeypatch, "--packages")
    assert code == 0 and len(made) == 1 and isinstance(made[0]["allowed_https_hosts"], K.ScannerHosts)
    assert "package_attempts" in env.state()


def test_without_the_switch_the_transport_has_the_runtime_hosts_and_the_stage_does_not_run(env, monkeypatch):
    code, made = _main(env, monkeypatch)
    assert code == 0 and made == [{}] and "package_attempts" not in env.state()
