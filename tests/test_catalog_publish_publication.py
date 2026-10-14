"""Tests for the publication half of `scripts/catalog_publish.py`: `--publish-pending`, `--verify` and `--upload --probe` (AP9-KA-P).

The helpers, the recording git and upload doubles and the `work` fixture are those of `test_catalog_publish.py`; nothing
here opens a socket, runs git or a process, or reads a token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_catalog_publish import (  # noqa: F401  (`work` is a fixture)
    APPLY, CATALOG_BYTES, COMMIT, FIX, P, PUBLISHED, R, REPO, TODAY, TOKEN_TEXT, Crashing, Interrupted, hub_answers, publish, removed, upload_env, work,
)


# --- --publish-pending ------------------------------------------------------------------------------------------------


def test_publish_pending_pushes_the_recorded_commit_uploads_its_bytes_and_checks_the_published_file(work, tmp_path):
    work.committed()
    env = upload_env(tmp_path)
    work.git.calls.clear()
    work.catalog.write_bytes(R.dump_json({**json.loads(work.catalog.read_bytes()), "published": "2026-10-09"}))  # the working tree moves on: no effect
    work.transport = hub_answers(work.decision()["result_catalog"].encode())
    result = work.call("--publish-pending", work.folder, env=env)
    assert result.code == 0 and work.state() == "published" and result.out.splitlines()[-1] == "published"
    assert work.git.calls == [["show", f"{COMMIT}:modelroom/catalog.json"], ["push", "forgejo", f"{COMMIT}:refs/heads/main"]]  # the bytes are read and checked before the push
    (args, child), = work.runner.calls
    assert args[:6] + args[7:] == ["uvx", "--from", f"huggingface_hub=={P.HF_HUB_VERSION}", "hf", "upload", "jm-vis/modelroom-catalog", ".",
                                   "--repo-type", "dataset", "--commit-message", COMMIT]  # both channel files, one folder, one commit (AP9-KO)
    assert work.runner.uploaded == [work.decision()["result_catalog"].encode()] and P.sha256_of(work.runner.uploaded[0]) == work.decision()["result_sha256"]
    assert str(work.catalog) not in args and not Path(args[6]).exists() and child["HF_TOKEN_PATH"] == env["HF_TOKEN_PATH"] and "HF_TOKEN" not in child
    assert TOKEN_TEXT not in result.out + result.err and [url for _, url in work.transport.calls][0].endswith("/revision/main")


def test_publish_pending_names_another_remote_only_as_a_plain_name(work, tmp_path):
    work.committed()
    assert work.call("--publish-pending", work.folder, "--remote", "a/b", env=upload_env(tmp_path)).code == 2 and not [c for c in work.git.calls if c[0] == "push"]
    work.transport = hub_answers(work.decision()["result_catalog"].encode())
    assert work.call("--publish-pending", work.folder, "--remote", "origin2", env=upload_env(tmp_path)).code == 0
    assert [c for c in work.git.calls if c[0] == "push"] == [["push", "origin2", f"{COMMIT}:refs/heads/main"]]


def test_a_failed_push_leaves_committed_and_the_next_call_pushes_the_same_commit(work, tmp_path):
    work.committed()
    work.git.push_fail = True
    assert work.call("--publish-pending", work.folder, env=upload_env(tmp_path)).code == 1 and work.state() == "committed" and work.runner.calls == []
    work.git.push_fail, work.transport = False, hub_answers(work.decision()["result_catalog"].encode())
    assert work.call("--publish-pending", work.folder, env=upload_env(tmp_path)).code == 0 and work.state() == "published"
    assert [c for c in work.git.calls if c[0] == "push"] == [["push", "forgejo", f"{COMMIT}:refs/heads/main"]] * 2


@pytest.mark.parametrize("fail_at, left, pushes, uploads", [(1, "committed", 2, 1), (2, "pushed", 1, 2), (3, "uploaded", 1, 1)])
def test_an_interrupted_publication_leaves_the_state_before_the_write_the_scanner_names_it_and_the_next_call_resumes_there(work, tmp_path, fail_at, left, pushes, uploads):
    work.committed()
    work.transport = hub_answers(work.decision()["result_catalog"].encode())
    with pytest.raises(Interrupted):
        work.call("--publish-pending", work.folder, files=Crashing(fail_at), env=upload_env(tmp_path))  # the step is done, its state is not written
    assert work.state() == left and work.decision()["commit"] == COMMIT
    nxt, result = work.scan_next()
    assert nxt["summary"]["pending_publication"] == {"date": TODAY, "state": left} and result.exit_code == 0
    assert {c["hf_repo"] for c in nxt["changes"]} == {"orbit/Comet-3-8B", "orbit/Comet-Echo-1"}
    work.transport = hub_answers(work.decision()["result_catalog"].encode())
    assert work.call("--publish-pending", work.folder, env=upload_env(tmp_path)).code == 0 and work.state() == "published"
    assert len([c for c in work.git.calls if c[0] == "push"]) == pushes and len(work.runner.calls) == uploads
    assert all(data == work.decision()["result_catalog"].encode() for data in work.runner.uploaded)


def test_an_interrupted_record_commit_leaves_applied_without_a_commit_and_the_same_call_records_it(work):
    assert work.apply(*APPLY).code == 0
    work.hold()
    with pytest.raises(Interrupted):
        work.call("--record-commit", work.folder, COMMIT, files=Crashing(1))
    assert (work.state(), work.decision()["commit"]) == ("applied", None) and work.scan_next()[0]["summary"]["pending_publication"]["state"] == "applied"
    assert work.call("--record-commit", work.folder, COMMIT).code == 0 and work.state() == "committed"


def test_a_failed_upload_process_leaves_pushed_with_exit_1_and_the_next_call_uploads_the_same_bytes(work, tmp_path):
    work.committed()
    work.runner.code = 1
    result = work.call("--publish-pending", work.folder, env=upload_env(tmp_path))
    assert result.code == 1 and work.state() == "pushed" and TOKEN_TEXT not in result.out + result.err
    work.runner.code, work.transport = 0, hub_answers(work.decision()["result_catalog"].encode())
    assert work.call("--publish-pending", work.folder, env=upload_env(tmp_path)).code == 0 and work.state() == "published"
    assert work.runner.uploaded[0] == work.runner.uploaded[1]


def test_a_failure_of_the_upload_file_after_the_push_is_exit_1_and_leaves_pushed(work, tmp_path, monkeypatch):
    work.committed()
    checks, real = [], publish.Tool._hygiene

    def third_check_fails(self, data, name="catalog-v2.json"):  # the first check of a temporary file: `_ready` checked the three channel files before the push
        checks.append(data)
        if len(checks) == 4:
            raise publish.PublishError("the file fails the hygiene check")
        real(self, data, name)

    monkeypatch.setattr(publish.Tool, "_hygiene", third_check_fails)
    result = work.call("--publish-pending", work.folder, env=upload_env(tmp_path))
    assert (result.code, work.state(), len(checks), work.runner.calls) == (1, "pushed", 4, []) and "state stays" in result.err


def test_a_failed_check_after_the_upload_leaves_uploaded_and_the_next_call_checks_without_uploading_again(work, tmp_path):
    work.committed()
    work.transport = hub_answers(b"something else")
    result = work.call("--publish-pending", work.folder, env=upload_env(tmp_path))
    assert result.code == 1 and work.state() == "uploaded" and "ALARM" in result.out and len(work.runner.calls) == 1
    work.transport = hub_answers(work.decision()["result_catalog"].encode())
    assert work.call("--publish-pending", work.folder, env=upload_env(tmp_path)).code == 0 and work.state() == "published" and len(work.runner.calls) == 1


@pytest.mark.parametrize("state", ["approved", "applied"])
def test_publish_pending_refuses_a_decision_without_a_commit(work, state):
    assert work.apply(*APPLY).code == 0
    if state == "approved":
        work.set_state("approved")
    before = work.tree()
    assert work.call("--publish-pending", work.folder).code == 2 and work.tree() == before and work.git.calls == [] and work.runner.calls == []


def test_publish_pending_on_a_published_decision_does_nothing(work):
    work.committed()
    work.set_state("published")
    work.git.calls.clear()
    assert work.call("--publish-pending", work.folder).code == 0 and work.runner.calls == [] and work.git.calls == []


@pytest.mark.parametrize("start", ["committed", "pushed"])
@pytest.mark.parametrize("fault", ["hygiene", "no-words-variable", "no-words-file", "no-token-variable", "no-token-file", "words-inside-repo", "other-bytes"])
def test_nothing_is_uploaded_or_pushed_when_a_check_before_the_process_fails_and_the_state_stays(work, tmp_path, fault, start):
    work.committed()
    work.set_state(start)
    env = upload_env(tmp_path, words="atlas\n" if fault == "hygiene" else "zzprivate-word\n")
    env = {"no-words-variable": removed(env, "MODELROOM_RELEASE_PRIVATE_WORDS"), "no-token-variable": removed(env, "HF_TOKEN_PATH"),
           "no-words-file": {**env, "MODELROOM_RELEASE_PRIVATE_WORDS": str(tmp_path / "missing.txt")},
           "no-token-file": {**env, "HF_TOKEN_PATH": str(tmp_path / "missing-token")},
           "words-inside-repo": {**env, "MODELROOM_RELEASE_PRIVATE_WORDS": str(REPO / "README.md")}}.get(fault, env)
    if fault == "other-bytes":
        work.git.commits[COMMIT] = CATALOG_BYTES
    result = work.call("--publish-pending", work.folder, env=env)
    assert result.code == 2 and work.runner.calls == [] and work.state() == start and "zzprivate-word" not in result.err and TOKEN_TEXT not in result.err
    assert not [c for c in work.git.calls if c[0] == "push"]  # a refusal is exit 2: nothing was pushed, nothing written


# --- --verify ---------------------------------------------------------------------------------------------------------


def test_verify_compares_the_published_file_with_the_working_tree_without_a_token_path(work):
    work.transport = hub_answers(CATALOG_BYTES)
    result = work.call("--verify", env={})
    assert (result.code, result.out) == (0, "catalog: published files equal working tree\n")
    assert [url for _, url in work.transport.calls] == ["https://huggingface.co/api/datasets/jm-vis/modelroom-catalog/revision/main",
                                                        f"https://huggingface.co/datasets/jm-vis/modelroom-catalog/resolve/{PUBLISHED}/catalog-v4.json",
                                                        f"https://huggingface.co/datasets/jm-vis/modelroom-catalog/resolve/{PUBLISHED}/catalog-v3.json",
                                                        f"https://huggingface.co/datasets/jm-vis/modelroom-catalog/resolve/{PUBLISHED}/catalog-v2.json"]
    assert work.git.calls == [] and work.runner.calls == []


def test_verify_against_a_commit_compares_the_bytes_of_that_commit_and_not_the_working_tree(work):
    of_the_commit = R.dump_json({**json.loads(CATALOG_BYTES), "published": "2026-10-01"})
    work.git.commits[COMMIT] = of_the_commit
    work.transport = hub_answers(of_the_commit)
    assert work.call("--verify", "--commit", COMMIT).out == f"catalog: published files equal {COMMIT}\n"
    work.transport = hub_answers(CATALOG_BYTES)
    result = work.call("--verify", "--commit", COMMIT)
    assert result.code == 1 and result.out == "".join(f"ALARM: the published catalog-v{n}.json differs from {COMMIT}\n" for n in (4, 3, 2))


def test_verify_gives_an_alarm_line_and_exit_1_when_the_published_file_differs(work):
    work.transport = hub_answers(CATALOG_BYTES + b" ")
    result = work.call("--verify")
    assert (result.code, result.out) == (1, "ALARM: the published catalog-v2.json differs from working tree\n")  # the new channel's file is made from it and still equal


@pytest.mark.parametrize("status", [404, 500])
def test_verify_gives_a_line_of_its_own_and_exit_1_when_the_dataset_cannot_be_reached(work, status):
    work.transport = hub_answers(CATALOG_BYTES, status=status)
    result = work.call("--verify")
    assert result.code == 1 and result.out == f"catalog: the published file could not be read (HTTP {status})\n"


def test_verify_gives_that_line_when_the_commit_is_unknown_and_never_reads_the_token_variable(work):
    assert work.call("--verify", "--commit", COMMIT, env={"HF_TOKEN_PATH": str(work.root / "no-such-token")}).code == 1
    assert work.call("--verify", "--commit", "--force").code == 2 and work.transport.calls == []


def test_verify_with_a_probe_reads_the_probe_dataset_and_compares_with_the_named_fixture(work):
    fixture = FIX / "catalog_v2_valid.json"
    work.transport = hub_answers(fixture.read_bytes(), dataset=publish.PROBE_DATASET)
    assert work.call("--verify", "--probe", fixture).out == "catalog: published file equals the probe file\n"
    assert all("modelroom-catalog-probe" in url for _, url in work.transport.calls)


# --- --upload --probe -------------------------------------------------------------------------------------------------


def test_the_probe_upload_goes_to_the_probe_dataset_only_with_the_bytes_of_an_unchanged_fixture(work, tmp_path):
    fixture = FIX / "catalog_v2_valid.json"
    env = upload_env(tmp_path)
    result = work.call("--upload", "--probe", fixture, env=env)
    (args, child), = work.runner.calls
    literal = ":(literal)tests/fixtures/catalog_v2_valid.json"  # never a pathspec pattern: `[ab].json` must not pass as `a.json`
    assert result.code == 0 and work.git.calls == [["ls-files", "--", literal], ["status", "--porcelain", "--", literal]]
    assert args[:6] + args[7:] == ["uvx", "--from", f"huggingface_hub=={P.HF_HUB_VERSION}", "hf", "upload", "jm-vis/modelroom-catalog-probe", "catalog-v2.json",
                                   "--repo-type", "dataset", "--commit-message", "probe"]
    assert "jm-vis/modelroom-catalog" not in args and work.runner.uploaded == [fixture.read_bytes()] and "HF_TOKEN" not in child and TOKEN_TEXT not in result.out
    assert work.call("--upload", "--probe", fixture, env=removed(env, "HF_TOKEN_PATH")).code == 2 and len(work.runner.calls) == 1  # no token path: no process


@pytest.mark.parametrize("where, status, tracked", [("outside", b"", True), ("changed", b" M tests/fixtures/catalog_v2_valid.json\n", True),
                                                    ("untracked", b"?? tests/fixtures/catalog_v2_valid.json\n", False), ("ignored", b"", False),
                                                    ("missing", b"", True), ("folder", b"", True)])
def test_the_probe_upload_refuses_a_file_outside_the_fixtures_a_changed_an_untracked_an_ignored_and_a_missing_one(work, tmp_path, where, status, tracked):
    outside = tmp_path / "other.json"
    outside.write_bytes(CATALOG_BYTES)
    path = {"outside": outside, "missing": FIX / "no-such-fixture.json", "folder": FIX}.get(where, FIX / "catalog_v2_valid.json")
    work.git.status, work.git.tracked = status, work.git.tracked if tracked else b""  # an ignored file: no status line and not in the index
    assert work.call("--upload", "--probe", path, env=upload_env(tmp_path)).code == 2 and work.runner.calls == []


def test_the_probe_upload_stops_on_a_hygiene_finding_before_the_process(work, tmp_path):
    result = work.call("--upload", "--probe", FIX / "catalog_v2_valid.json", env=upload_env(tmp_path, words="acme\n"))
    assert result.code == 2 and "hygiene" in result.err and work.runner.calls == []


def test_a_failed_probe_upload_is_exit_1(work, tmp_path):
    work.runner.code = 3
    assert work.call("--upload", "--probe", FIX / "catalog_v2_valid.json", env=upload_env(tmp_path)).code == 1
