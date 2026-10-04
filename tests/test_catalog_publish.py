"""Tests for `scripts/catalog_publish.py`: taking a proposal into the table after the maintainer's word, and publishing it (AP9-KA-P).

Nothing here opens a socket, runs git or an upload process, reads a token or touches the real catalog. The Hub's answers come
from a `FixtureTransport`, git is a recording function, the upload process is a recording function that reads the file it is
given, and the disk is a subclass of the scanner's `DiskFiles` that stops after the k-th write. The proposal is
`tests/fixtures/publish_proposal.json` against `tests/fixtures/publish_catalog.json` (accounts and models invented): six
changes, one of each kind, and three candidates (K1 a publisher, K2 a packager, K3 a repo of open kind).
"""

from __future__ import annotations

import builtins
import importlib
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the publisher is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))
publish = importlib.import_module("catalog_publish")
P = importlib.import_module("catalog_publish_rules")
R = importlib.import_module("catalog_scan_rules")

from modelroom.catalog_schema import build_catalog_v2  # noqa: E402
from modelroom.http import FixtureTransport, Response  # noqa: E402
from test_catalog_scan import NOW as SCAN_NOW, Hub, scanner  # noqa: E402

FIX = REPO / "tests" / "fixtures"
CATALOG_BYTES, REJECTED_BYTES, PROPOSAL_BYTES = (FIX / "publish_catalog.json").read_bytes(), (FIX / "scan_rejected.json").read_bytes(), (FIX / "publish_proposal.json").read_bytes()
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
TODAY, COMMIT, PUBLISHED = "2026-10-04", "a" * 40, "b" * 40
TOKEN_TEXT = "placeholder-token-content"
APPLY = ("--approve", "1,2", "--reject", "4")  # two rows in, one rejected: both files and the rejection list change


class Result(NamedTuple):
    code: int
    out: str
    err: str


class Interrupted(Exception):
    pass


class Crashing(publish.DiskFiles):
    """A disk that stops when the k-th write is about to happen; `leave_temp` leaves the temporary file of that write behind."""

    def __init__(self, fail_at: int, leave_temp: bool = False) -> None:
        self.fail_at, self.leave_temp, self.writes = fail_at, leave_temp, 0

    def write(self, path, data: bytes) -> None:
        self.writes += 1
        if self.writes == self.fail_at:
            if self.leave_temp:
                (Path(path).parent / f"{Path(path).name}.stray.tmp").write_bytes(data[:5])
            raise Interrupted()
        super().write(path, data)


class Counting(publish.DiskFiles):
    """Records every read and write; `after_read` runs once the proposal has been read (a file replaced in between)."""

    def __init__(self, after_read=None) -> None:
        self.reads, self.writes, self.after_read = [], [], after_read

    def read(self, path) -> bytes:
        self.reads.append(Path(path).name)
        data = super().read(path)
        if self.after_read and Path(path).name == "proposal.json":
            self.after_read()
        return data

    def write(self, path, data: bytes) -> None:
        self.writes.append(Path(path).name)
        super().write(path, data)


class Git:
    """Records every call; `commits` maps a hash to the bytes of its `modelroom/catalog.json`."""

    def __init__(self, commits=None, push_fail: bool = False, status: bytes = b"") -> None:
        self.commits, self.push_fail, self.status, self.calls = dict(commits or {}), push_fail, status, []

    def __call__(self, args):
        self.calls.append(list(args))
        if args[0] == "show":
            commit, _, path = args[1].partition(":")
            if commit not in self.commits or path != "modelroom/catalog.json":
                raise publish.GitError("git show failed (exit 128)")
            return self.commits[commit]
        if args[0] == "push":
            if self.push_fail:
                raise publish.GitError("git push failed (exit 1)")
            return b""
        if args[0] == "status":
            return self.status
        raise AssertionError(args)


class Runner:
    """The upload process: records the argument list and the environment and reads the file it was given while it exists."""

    def __init__(self, code: int = 0) -> None:
        self.code, self.calls, self.uploaded = code, [], []

    def __call__(self, args, env) -> int:
        self.calls.append((list(args), dict(env)))
        self.uploaded.append(Path(args[6]).read_bytes())
        return self.code


def hub_answers(body: bytes, dataset: str = publish.DATASET, status: int = 200) -> FixtureTransport:
    revision = f"https://huggingface.co/api/datasets/{dataset}/revision/main"
    file = f"https://huggingface.co/datasets/{dataset}/resolve/{PUBLISHED}/catalog-v2.json"
    return FixtureTransport({
        ("GET", revision): Response(status, {}, json.dumps({"sha": PUBLISHED}).encode()),
        ("GET", file): Response(200, {"X-Repo-Commit": PUBLISHED}, body),
    })


class Work:
    def __init__(self, root: Path) -> None:
        self.root, self.out = root, root / "out"
        self.folder = self.out / TODAY
        self.folder.mkdir(parents=True)
        self.catalog, self.rejected, self.proposal = root / "catalog.json", root / "rejected.json", self.folder / "proposal.json"
        self.catalog.write_bytes(CATALOG_BYTES)
        self.rejected.write_bytes(REJECTED_BYTES)
        self.proposal.write_bytes(PROPOSAL_BYTES)
        self.sha = P.sha256_of(PROPOSAL_BYTES)
        self.git, self.runner, self.transport = Git(), Runner(), FixtureTransport({})

    def tree(self) -> dict:
        return {str(p.relative_to(self.root)): p.read_bytes() for p in sorted(self.root.rglob("*")) if p.is_file()}

    def call(self, *argv, files=None, env=None, now=NOW) -> Result:
        out, err = io.StringIO(), io.StringIO()
        code = publish.main(["--catalog", str(self.catalog), "--rejected", str(self.rejected), *map(str, argv)], files=files, run_git=self.git,
                            run=self.runner, transport=self.transport, env={} if env is None else env, now=now, out=out, err=err)
        return Result(code, out.getvalue(), err.getvalue())

    def apply(self, *lists, **kw) -> Result:
        return self.call("--apply", self.proposal, "--proposal-sha", self.sha, *lists, **kw)

    def decision(self, folder: Path | None = None) -> dict:
        return json.loads(((folder or self.folder) / "decision.json").read_bytes())

    def state(self) -> str:
        return self.decision()["state"]

    def set_state(self, state: str, **more) -> None:
        (self.folder / "decision.json").write_bytes(R.dump_json({**self.decision(), "state": state, **more}))

    def committed(self) -> None:
        assert self.apply(*APPLY).code == 0
        self.git.commits[COMMIT] = self.decision()["result_catalog"].encode()
        assert self.call("--record-commit", self.folder, COMMIT).code == 0

    def scan_next(self):
        """The scanner over the same folder on the next day, with an empty Hub: the contract of the two tools."""
        result = scanner.scan(self.catalog, self.rejected, self.out, transport=FixtureTransport(Hub().m), now=SCAN_NOW + timedelta(days=1))
        return json.loads((self.out / "2026-10-05" / "proposal.json").read_bytes()), result


@pytest.fixture
def work(tmp_path: Path) -> Work:
    return Work(tmp_path / "work")


def upload_env(tmp: Path, words: str = "zzprivate-word\n") -> dict:
    folder = tmp / "secrets"
    folder.mkdir(exist_ok=True)
    (folder / "token.txt").write_text(TOKEN_TEXT)
    (folder / "words.txt").write_text(words)
    return {"HF_TOKEN_PATH": str(folder / "token.txt"), "HF_TOKEN": "must-not-pass", "MODELROOM_RELEASE_PRIVATE_WORDS": str(folder / "words.txt")}


def removed(env: dict, *names: str) -> dict:
    return {k: v for k, v in env.items() if k not in names}


# --- --show ----------------------------------------------------------------------------------------------------------


def test_show_prints_the_scanner_rendering_of_the_bytes_it_read_with_their_checksum_and_touches_nothing(work):
    before = work.tree()
    result = work.call("--show", work.out)
    text = R.render_proposal(json.loads(PROPOSAL_BYTES), work.sha)
    assert result.code == 0 and result.out.startswith(text) and f"proposal.json sha256: {work.sha}\n" in result.out
    assert result.out[len(text):].splitlines()[0] == "base: current" and "no catalog proposal" not in result.out
    assert work.tree() == before and work.transport.calls == [] and work.git.calls == []
    assert work.call("--show", work.out).out == result.out


PAIRS = [(c["n"], name) for c in json.loads(PROPOSAL_BYTES)["changes"] for name in sorted(c["fields"])]


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return value + ["zz/changed"] if isinstance(value, list) else "changed" if value is None else value + "X"


def _differing(work, mutate) -> tuple[list[str], list[str], list[int]]:
    before = work.call("--show", work.out).out.splitlines()
    data = json.loads(PROPOSAL_BYTES)
    mutate(data)
    work.proposal.write_bytes(R.dump_json(data))
    after = work.call("--show", work.out).out.splitlines()
    assert len(before) == len(after)
    return before, after, [i for i, (a, b) in enumerate(zip(before, after)) if a != b and not a.startswith("to decide")]


@pytest.mark.parametrize("n, name", PAIRS)
def test_show_changes_exactly_that_field_and_the_checksum_when_a_new_value_of_the_proposal_changes(work, n, name):
    def mutate(data):
        pair = next(c for c in data["changes"] if c["n"] == n)["fields"][name]
        pair["new"] = work.new = _changed(pair["new"])

    before, after, at = _differing(work, mutate)
    assert len(at) == 2 and before[at[0]].startswith(f"  {name}: ") and "proposal.json sha256" in before[at[1]]
    assert after[at[0]].endswith(" -> " + json.dumps(work.new)) and after[at[1]].endswith(P.sha256_of(work.proposal.read_bytes()))


@pytest.mark.parametrize("k, key", [("K1", "likes"), ("K1", "downloads"), ("K2", "bases"), ("K2", "rows"), ("K3", "tags"), ("K3", "parameters_b")])
def test_show_changes_exactly_that_field_and_the_checksum_when_a_candidate_value_changes(work, k, key):
    def mutate(data):
        cand = next(c for c in data["candidates"] if c["k"] == k)
        cand[key] = work.new = _changed(cand[key])

    before, after, at = _differing(work, mutate)
    assert len(at) == 2 and before[at[0]].startswith(f"  {key}: ") and after[at[0]] == f"  {key}: {json.dumps(work.new, sort_keys=True)}"


def test_show_does_not_shorten_a_long_list_or_a_long_value(work):
    repos = [f"packone/Very-Long-Model-Name-{i:02d}-With-A-Long-Tail-GGUF" for i in range(30)]
    data = json.loads(PROPOSAL_BYTES)
    data["changes"][1]["fields"]["gguf_repos"]["new"] = repos
    data["changes"][0]["fields"]["note"]["new"] = "n" * 3000
    work.proposal.write_bytes(R.dump_json(data))
    out = work.call("--show", work.out).out
    assert all(repo in out for repo in repos) and "n" * 3000 in out and "..." not in out.split("## Notes")[0]


def test_show_prints_the_checksum_of_the_bytes_it_read_even_when_the_file_is_replaced_before_it_prints(work):
    files = Counting(after_read=lambda: work.proposal.write_bytes(PROPOSAL_BYTES.replace(b"acme", b"acmf")))
    result = work.call("--show", work.out, files=files)
    assert files.reads.count("proposal.json") == 1 and f"proposal.json sha256: {work.sha}" in result.out
    assert result.out.startswith(R.render_proposal(json.loads(PROPOSAL_BYTES), work.sha)) and files.writes == []


def test_show_says_exactly_no_open_catalog_proposal_when_nothing_is_open_and_nothing_is_pending(work):
    work.proposal.unlink()
    assert work.call("--show", work.out).out == "no open catalog proposal\n"
    assert work.call("--show", work.root / "none").out == "no open catalog proposal\n"
    work.proposal.write_bytes(PROPOSAL_BYTES)
    (work.folder / "decision.json").write_bytes(R.dump_json({"state": "published"}))
    assert work.call("--show", work.out).out == "no open catalog proposal\n"


@pytest.mark.parametrize("state", ["approved", "applied", "committed", "pushed", "uploaded"])
def test_show_names_a_pending_publication_before_the_open_proposal_and_alone_when_nothing_else_is_open(work, state):
    earlier = work.out / "2026-10-03"
    earlier.mkdir()
    (earlier / "decision.json").write_bytes(R.dump_json({"state": state, "commit": COMMIT if state in ("pushed", "uploaded") else None}))
    line = f"publication pending: 2026-10-03, state {state}, commit {COMMIT if state in ('pushed', 'uploaded') else 'none'}"
    out = work.call("--show", work.out).out
    assert out.splitlines()[0] == line and out.splitlines()[1].startswith("# Catalog proposal") and "approval waits" in out and "to decide" not in out
    work.proposal.unlink()
    assert work.call("--show", work.out).out == line + "\n"


def test_show_marks_a_stale_base_and_gives_no_hint_to_approve(work):
    assert "to decide: --apply" in work.call("--show", work.out).out
    document = json.loads(CATALOG_BYTES)
    document["published"] = "2026-10-01"
    work.catalog.write_bytes(R.dump_json(document))
    out = work.call("--show", work.out).out
    assert "base: stale, the proposal was made against another catalog; run the scanner again" in out and "to decide" not in out and "base: current" not in out


# --- --apply: only the named rows -------------------------------------------------------------------------------------


def table(work) -> dict:
    return {m.hf_repo: m for f in build_catalog_v2(work.catalog.read_text(), Path("c.json")).families for m in f.models}


def test_apply_takes_exactly_the_named_numbers_and_leaves_the_others_open_for_the_next_proposal(work, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *a: pytest.fail("a prompt was asked"))
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    result = work.apply("--approve", "1,2,5")
    changes = json.loads(PROPOSAL_BYTES)["changes"]
    expected = R.apply_changes(json.loads(CATALOG_BYTES), [changes[0], changes[1], changes[4]])
    expected["published"] = TODAY
    assert result.code == 0 and work.catalog.read_bytes() == R.dump_json(expected) and work.rejected.read_bytes() == REJECTED_BYTES
    rows = table(work)
    assert "acme/Atlas-3-9B" in rows and rows["nova/Nimbus-1-30B"].gguf_repos == ["packone/Nimbus-1-30B-GGUF"] and rows["acme/Atlas-1-9B"].successor is None
    assert rows["orbit/Comet-Echo-1"].ollama_base is None and rows["orbit/Comet-3-8B"].latest_source.endswith("comet-1a2b3c")
    decision = work.decision()
    assert (decision["state"], decision["approved"], decision["rejected"], decision["admitted"], decision["commit"]) == ("applied", [1, 2, 5], [], [], None)
    assert decision["result_sha256"] == P.sha256_of(work.catalog.read_bytes()) and decision["proposal_sha"] == work.sha and decision["date"] == TODAY
    assert work.transport.calls == [] and work.git.calls == []
    nxt, _ = work.scan_next()
    assert {(c["hf_repo"], c.get("field")) for c in nxt["changes"]} == {("acme/Atlas-1-9B", "successor"), ("orbit/Comet-Echo-1", "ollama")}


def test_apply_with_only_a_rejection_adds_the_repo_to_the_list_and_takes_no_row_in(work):
    result = work.apply("--reject", "4")
    assert result.code == 0 and work.catalog.read_bytes() == CATALOG_BYTES
    assert json.loads(work.rejected.read_bytes())[-1] == {"hf_repo": "acme/Atlas-1-9B", "date": TODAY}
    assert (work.state(), work.decision()["approved"], work.decision()["rejected"]) == ("applied", [], [4])
    nxt, _ = work.scan_next()
    assert {c["n"] for c in nxt["changes"]} == {1, 2, 3, 4}  # 1 to 3 and 5 stay open (3 as it is a removal, which the scanner does not carry), 4 is rejected
    assert "acme/Atlas-1-9B" not in {c["hf_repo"] for c in nxt["changes"]}


def test_apply_with_only_an_admission_adds_the_accounts_and_takes_no_row_in(work):
    assert work.apply("--admit", "K1").code == 0
    document = json.loads(work.catalog.read_bytes())
    assert document["publishers"][-1] == {"account": "newlab", "name": "newlab", "region": None} and document["models"] == json.loads(CATALOG_BYTES)["models"]
    assert document["packagers"] == ["packone", "packtwo"] and (work.decision()["admitted"], work.decision()["state"]) == (["K1"], "applied")


def test_apply_admits_a_packager_as_a_plain_name_and_validates_the_result(work):
    assert work.apply("--admit", "K2", "--reject", "K1,K3").code == 0
    assert build_catalog_v2(work.catalog.read_text(), Path("c.json")).packagers[-1] == "packthree"
    assert R.parse_rejected(work.rejected.read_bytes()) == ({"acme/Rejected-1-7B", "acme/Odd-1-5B"}, {"spamlabs", "newlab"})


def test_apply_does_not_list_a_rejected_name_twice(work):
    work.rejected.write_bytes(R.dump_json([{"account": "newlab", "date": "2026-09-01"}]))
    assert work.apply("--reject", "K1").code == 0
    assert json.loads(work.rejected.read_bytes()) == [{"account": "newlab", "date": "2026-09-01"}]


def test_apply_never_asks_for_the_token_path_and_makes_no_request(work):
    files = Counting()
    result = work.apply(*APPLY, files=files, env={"HF_TOKEN_PATH": str(work.root / "no-such-token")})
    assert result.code == 0 and all(name in ("catalog.json", "rejected.json", "proposal.json", "decision.json") for name in files.reads + files.writes)
    assert work.transport.calls == [] and work.runner.calls == []


def _refusal(work, case: str) -> tuple[list, str]:
    base = ["--apply", work.proposal, "--proposal-sha", work.sha, "--approve", "1"]
    if case == "no-sha":
        return ["--apply", work.proposal, "--approve", "1"], "--proposal-sha"
    if case == "short-sha":
        return ["--apply", work.proposal, "--proposal-sha", "abc", "--approve", "1"], "--proposal-sha"
    if case == "other-bytes":
        work.proposal.write_bytes(PROPOSAL_BYTES.replace(b"acme", b"acmf", 1))
        return base, "has changed"
    if case == "other-base":
        document = json.loads(CATALOG_BYTES)
        document["published"] = "2026-10-01"
        work.catalog.write_bytes(R.dump_json(document))
        return base, "another catalog"
    cases = {"unknown-number": (["--approve", "9"], "not in the proposal"), "two-lists": (["--approve", "1", "--reject", "1"], "named twice"),
             "candidate-approved": (["--approve", "K1"], "is a candidate"), "b3-admitted": (["--admit", "K3"], "only --complete"),
             "nothing-named": ([], "nothing is named")}
    if case in cases:
        return ["--apply", work.proposal, "--proposal-sha", work.sha, *cases[case][0]], cases[case][1]
    if case == "decision-exists":
        assert work.apply("--reject", "4").code == 0
        return base, "a decision exists"
    earlier = work.out / "2026-10-03"
    earlier.mkdir()
    (earlier / "decision.json").write_bytes(R.dump_json({"state": "committed", "commit": COMMIT}))
    return base, "another publication is pending"


@pytest.mark.parametrize("case", ["no-sha", "short-sha", "other-bytes", "other-base", "unknown-number", "two-lists", "candidate-approved", "b3-admitted",
                                  "nothing-named", "decision-exists", "other-pending"])
def test_apply_is_refused_with_exit_2_before_any_write_and_nothing_changes(work, case):
    argv, fragment = _refusal(work, case)
    before = work.tree()
    result = work.call(*argv)
    assert result.code == 2 and fragment in result.err and work.tree() == before
    assert work.transport.calls == [] and work.git.calls == [] and work.runner.calls == []


@pytest.mark.parametrize("flag", ["--yes", "--accept", "--all", "--without", "--approve-all", "--appr"])
def test_the_parser_knows_no_yes_no_accept_all_and_no_without_and_takes_no_abbreviation(work, flag):
    before = work.tree()
    assert work.call("--apply", work.proposal, "--proposal-sha", work.sha, flag, "1").code == 2 and work.tree() == before


@pytest.mark.parametrize("argv", [["--show", "x", "--approve", "1"], ["--materialize", "x", "--proposal-sha", "a"], ["--verify", "--approve", "1"],
                                  ["--apply", "x", "--commit", "a"], ["--upload"], ["--upload", "--probe"], ["--show", "x", "--apply", "y"], []])
def test_an_option_of_another_mode_a_missing_mode_and_an_upload_without_a_probe_are_refused(work, argv):
    assert work.call(*argv).code == 2 and work.transport.calls == [] and work.runner.calls == []


# --- the two steps of --apply and the interruption --------------------------------------------------------------------


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_an_interrupted_apply_leaves_a_state_that_the_scanner_reads_and_materialize_completes(work, fail_at):
    with pytest.raises(Interrupted):
        work.apply(*APPLY, files=Crashing(fail_at))
    catalog_done, rejected_done = fail_at > 2, fail_at > 3
    assert (work.folder / "decision.json").exists() is (fail_at > 1)
    assert (work.catalog.read_bytes() == CATALOG_BYTES) is (not catalog_done) and (work.rejected.read_bytes() == REJECTED_BYTES) is (not rejected_done)
    if fail_at == 1:  # nothing happened: the same word works again
        assert work.apply(*APPLY).code == 0 and work.state() == "applied"
        return
    decision = work.decision()
    assert decision["state"] == "approved" and decision["commit"] is None
    nxt, result = work.scan_next()
    assert nxt["summary"]["pending_publication"] == {"date": TODAY, "state": "approved"} and result.exit_code == 0
    assert {c["hf_repo"] for c in nxt["changes"]} == {"orbit/Comet-3-8B", "orbit/Comet-Echo-1"}  # what was taken or rejected is not proposed again
    before = work.tree()
    assert work.apply(*APPLY).code == 2 and work.tree() == before  # the decision exists
    assert work.call("--materialize", work.folder).code == 0
    assert work.catalog.read_text() == decision["result_catalog"] and work.rejected.read_text() == decision["result_rejected"] and work.state() == "applied"
    after = work.tree()
    assert work.call("--materialize", work.folder).code == 2 and work.tree() == after  # applied is not approved: nothing changes


def test_a_temporary_file_left_by_an_interrupted_replace_disturbs_neither_the_scanner_nor_materialize(work):
    with pytest.raises(Interrupted):
        work.apply(*APPLY, files=Crashing(2, leave_temp=True))
    assert [p.name for p in work.root.glob("*.stray.tmp")] == ["catalog.json.stray.tmp"]
    assert work.scan_next()[0]["summary"]["pending_publication"]["state"] == "approved"
    assert work.call("--materialize", work.folder).code == 0 and work.state() == "applied"
    with pytest.raises(Interrupted):
        Work(work.root.parent / "second").apply(*APPLY, files=Crashing(4, leave_temp=True))


def test_materialize_writes_a_file_with_the_target_checksum_never_again_and_a_file_with_the_base_checksum_once(work):
    with pytest.raises(Interrupted):
        work.apply(*APPLY, files=Crashing(4))
    files = Counting()
    assert work.call("--materialize", work.folder, files=files).code == 0
    assert files.writes == ["decision.json"]  # both files already hold the result


@pytest.mark.parametrize("which", ["catalog", "rejected"])
def test_materialize_refuses_a_file_that_is_neither_the_base_nor_the_result_and_writes_nothing(work, which):
    with pytest.raises(Interrupted):
        work.apply(*APPLY, files=Crashing(2))
    path = work.catalog if which == "catalog" else work.rejected
    path.write_bytes(path.read_bytes() + b"\n")
    before = work.tree()
    result = work.call("--materialize", work.folder)
    assert result.code == 2 and "neither the base nor the result" in result.err and work.tree() == before and work.state() == "approved"


@pytest.mark.parametrize("state", ["applied", "committed", "pushed", "uploaded", "published"])
def test_materialize_works_on_approved_only(work, state):
    assert work.apply(*APPLY).code == 0
    work.set_state(state)
    before = work.tree()
    assert work.call("--materialize", work.folder).code == 2 and work.tree() == before


def test_materialize_without_a_decision_or_with_a_damaged_one_is_an_error(work):
    assert work.call("--materialize", work.folder).code == 2
    (work.folder / "decision.json").write_bytes(b'{"state": "approved"}')
    assert work.call("--materialize", work.folder).code == 2 and "damaged" in work.call("--materialize", work.folder).err


# --- --complete -------------------------------------------------------------------------------------------------------


def test_complete_makes_a_candidate_of_open_kind_a_row_and_the_old_checksum_no_longer_fits(work):
    result = work.call("--complete", work.proposal, "--proposal-sha", work.sha, "--admit", "K3", "--kind", "vision")
    data = json.loads(work.proposal.read_bytes())
    new_sha = P.sha256_of(work.proposal.read_bytes())
    assert result.code == 0 and f"proposal.json sha256: {new_sha}" in result.out and "row 2 made from K3" in result.out
    assert work.proposal.read_bytes() == R.dump_json(data) and (work.folder / "proposal.md").read_text() == R.render_proposal(data)
    assert not (work.folder / "decision.json").exists() and work.catalog.read_bytes() == CATALOG_BYTES and work.rejected.read_bytes() == REJECTED_BYTES
    row = {name: pair["new"] for name, pair in data["changes"][1]["fields"].items()}
    assert (row["hf_repo"], row["kind"], row["kind_source"], row["note"]) == ("acme/Odd-1-5B", "vision", "decision", f"Kind vision set by the maintainer on {TODAY}.")
    assert [c["k"] for c in data["candidates"]] == ["K1", "K2"] and work.transport.calls == [] and work.git.calls == []
    shown = work.call("--show", work.out).out
    assert f"proposal.json sha256: {new_sha}" in shown and all(f"  {name}: null -> {json.dumps(pair['new'])}" in shown for name, pair in data["changes"][1]["fields"].items())
    before = work.tree()
    assert work.apply("--approve", "2").code == 2 and work.tree() == before  # the checksum of the old file
    assert work.call("--apply", work.proposal, "--proposal-sha", new_sha, "--approve", "2").code == 0
    rows = table(work)
    assert rows["acme/Odd-1-5B"].kind == "vision" and rows["acme/Odd-1-5B"].kind_source == "decision" and "acme/Atlas-3-9B" not in rows


@pytest.mark.parametrize("extra", [[], ["--kind", "classification"], ["--kind", "Text"], ["--admit", "K3,K2", "--kind", "vision"],
                                   ["--admit", "K1", "--kind", "vision"], ["--admit", "K2", "--kind", "vision"], ["--admit", "K9", "--kind", "vision"]])
def test_complete_is_refused_with_exit_2_and_changes_nothing(work, extra):
    argv = ["--complete", work.proposal, "--proposal-sha", work.sha, "--admit", "K3", "--kind", "vision"]
    for flag, value in zip(extra[::2], extra[1::2]):
        argv[argv.index(flag) + 1] = value
    if "--kind" not in extra and "--admit" not in extra:
        del argv[-2:]
    before = work.tree()
    assert work.call(*argv).code == 2 and work.tree() == before


def test_complete_with_the_old_checksum_after_a_first_completion_is_refused(work):
    argv = ["--complete", work.proposal, "--proposal-sha", work.sha, "--admit", "K3", "--kind", "vision"]
    assert work.call(*argv).code == 0
    before = work.tree()
    assert work.call(*argv).code == 2 and work.tree() == before


def test_complete_is_refused_while_a_publication_is_pending_or_a_decision_exists(work):
    assert work.apply("--reject", "4").code == 0
    before = work.tree()
    assert work.call("--complete", work.proposal, "--proposal-sha", work.sha, "--admit", "K3", "--kind", "vision").code == 2 and work.tree() == before


# --- --record-commit --------------------------------------------------------------------------------------------------


def test_record_commit_stores_the_commit_when_its_catalog_is_the_result_of_the_decision(work):
    assert work.apply(*APPLY).code == 0
    work.git.commits[COMMIT] = work.decision()["result_catalog"].encode()
    assert work.call("--record-commit", work.folder, COMMIT).code == 0
    assert (work.state(), work.decision()["commit"]) == ("committed", COMMIT) and work.git.calls == [["show", f"{COMMIT}:modelroom/catalog.json"]]


@pytest.mark.parametrize("commit, bytes_", [(COMMIT, CATALOG_BYTES), ("c" * 40, None), ("--force", None), ("main", None)])
def test_record_commit_refuses_a_commit_that_is_not_the_result_and_the_state_stays_applied(work, commit, bytes_):
    assert work.apply(*APPLY).code == 0
    if bytes_ is not None:
        work.git.commits[commit] = bytes_
    before = work.tree()
    assert work.call("--record-commit", work.folder, commit).code == 2 and work.tree() == before and work.state() == "applied"


@pytest.mark.parametrize("state", ["approved", "committed", "pushed", "published"])
def test_record_commit_works_on_applied_only(work, state):
    assert work.apply(*APPLY).code == 0
    work.git.commits[COMMIT] = work.decision()["result_catalog"].encode()
    work.set_state(state)
    assert work.call("--record-commit", work.folder, COMMIT).code == 2 and work.state() == state


# --- --publish-pending ------------------------------------------------------------------------------------------------


def test_publish_pending_pushes_the_recorded_commit_uploads_its_bytes_and_checks_the_published_file(work, tmp_path):
    work.committed()
    env = upload_env(tmp_path)
    work.git.calls.clear()
    work.catalog.write_bytes(R.dump_json({**json.loads(work.catalog.read_bytes()), "published": "2026-10-09"}))  # the working tree moves on: no effect
    work.transport = hub_answers(work.decision()["result_catalog"].encode())
    result = work.call("--publish-pending", work.folder, env=env)
    assert result.code == 0 and work.state() == "published" and result.out.splitlines()[-1] == "published"
    assert work.git.calls[0] == ["push", "forgejo", f"{COMMIT}:refs/heads/main"] and "--force" not in work.git.calls[0]
    assert all(call[0] == "show" for call in work.git.calls[1:])
    (args, child), = work.runner.calls
    assert args[:6] + args[7:] == ["uvx", "--from", f"huggingface_hub=={P.HF_HUB_VERSION}", "hf", "upload", "jm-vis/modelroom-catalog", "catalog-v2.json",
                                   "--repo-type", "dataset", "--commit-message", COMMIT]
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


def test_an_interruption_after_the_push_leaves_pushed_and_the_next_call_uploads_the_commit_bytes(work, tmp_path):
    work.committed()
    with pytest.raises(Interrupted):
        work.call("--publish-pending", work.folder, files=Crashing(2), env=upload_env(tmp_path))  # the push is written, the upload is made, its state is not
    assert work.state() == "pushed"
    work.transport = hub_answers(work.decision()["result_catalog"].encode())
    assert work.call("--publish-pending", work.folder, env=upload_env(tmp_path)).code == 0 and work.state() == "published"
    assert work.runner.uploaded[0] == work.runner.uploaded[1] == work.decision()["result_catalog"].encode()


def test_a_failed_upload_process_leaves_pushed_with_exit_1_and_the_next_call_uploads_the_same_bytes(work, tmp_path):
    work.committed()
    work.runner.code = 1
    result = work.call("--publish-pending", work.folder, env=upload_env(tmp_path))
    assert result.code == 1 and work.state() == "pushed" and TOKEN_TEXT not in result.out + result.err
    work.runner.code, work.transport = 0, hub_answers(work.decision()["result_catalog"].encode())
    assert work.call("--publish-pending", work.folder, env=upload_env(tmp_path)).code == 0 and work.state() == "published"
    assert work.runner.uploaded[0] == work.runner.uploaded[1]


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
    assert work.call("--publish-pending", work.folder).code == 0 and work.runner.calls == [] and work.git.calls == work.git.calls[:1]


@pytest.mark.parametrize("fault", ["hygiene", "no-words-variable", "no-words-file", "no-token-variable", "no-token-file", "words-inside-repo", "other-bytes"])
def test_nothing_is_uploaded_when_a_check_before_the_process_fails_and_the_state_stays_pushed(work, tmp_path, fault):
    work.committed()
    work.set_state("pushed")
    env = upload_env(tmp_path, words="atlas\n" if fault == "hygiene" else "zzprivate-word\n")
    env = {"no-words-variable": removed(env, "MODELROOM_RELEASE_PRIVATE_WORDS"), "no-token-variable": removed(env, "HF_TOKEN_PATH"),
           "no-words-file": {**env, "MODELROOM_RELEASE_PRIVATE_WORDS": str(tmp_path / "missing.txt")},
           "no-token-file": {**env, "HF_TOKEN_PATH": str(tmp_path / "missing-token")},
           "words-inside-repo": {**env, "MODELROOM_RELEASE_PRIVATE_WORDS": str(REPO / "README.md")}}.get(fault, env)
    if fault == "other-bytes":
        work.git.commits[COMMIT] = CATALOG_BYTES
    result = work.call("--publish-pending", work.folder, env=env)
    assert result.code == 2 and work.runner.calls == [] and work.state() == "pushed" and "zzprivate-word" not in result.err and TOKEN_TEXT not in result.err


# --- --verify ---------------------------------------------------------------------------------------------------------


def test_verify_compares_the_published_file_with_the_working_tree_without_a_token_path(work):
    work.transport = hub_answers(CATALOG_BYTES)
    result = work.call("--verify", env={})
    assert (result.code, result.out) == (0, "catalog: published file equals working tree\n")
    assert [url for _, url in work.transport.calls] == ["https://huggingface.co/api/datasets/jm-vis/modelroom-catalog/revision/main",
                                                        f"https://huggingface.co/datasets/jm-vis/modelroom-catalog/resolve/{PUBLISHED}/catalog-v2.json"]
    assert work.git.calls == [] and work.runner.calls == []


def test_verify_against_a_commit_compares_the_bytes_of_that_commit_and_not_the_working_tree(work):
    work.git.commits[COMMIT] = b"bytes of the commit"
    work.transport = hub_answers(b"bytes of the commit")
    assert work.call("--verify", "--commit", COMMIT).out == f"catalog: published file equals {COMMIT}\n"
    work.transport = hub_answers(CATALOG_BYTES)
    result = work.call("--verify", "--commit", COMMIT)
    assert result.code == 1 and result.out == f"ALARM: the published catalog differs from {COMMIT}\n"


def test_verify_gives_an_alarm_line_and_exit_1_when_the_published_file_differs(work):
    work.transport = hub_answers(CATALOG_BYTES + b" ")
    result = work.call("--verify")
    assert (result.code, result.out) == (1, "ALARM: the published catalog differs from working tree\n")


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
    assert result.code == 0 and work.git.calls == [["status", "--porcelain", "--", "tests/fixtures/catalog_v2_valid.json"]]
    assert args[:6] + args[7:] == ["uvx", "--from", f"huggingface_hub=={P.HF_HUB_VERSION}", "hf", "upload", "jm-vis/modelroom-catalog-probe", "catalog-v2.json",
                                   "--repo-type", "dataset", "--commit-message", "probe"]
    assert "jm-vis/modelroom-catalog" not in args and work.runner.uploaded == [fixture.read_bytes()] and "HF_TOKEN" not in child and TOKEN_TEXT not in result.out
    assert work.call("--upload", "--probe", fixture, env=removed(env, "HF_TOKEN_PATH")).code == 2 and len(work.runner.calls) == 1  # no token path: no process


@pytest.mark.parametrize("where, status", [("outside", b""), ("changed", b" M tests/fixtures/catalog_v2_valid.json\n"), ("untracked", b"?? tests/fixtures/catalog_v2_valid.json\n"),
                                           ("missing", b""), ("folder", b"")])
def test_the_probe_upload_refuses_a_file_outside_the_fixtures_a_changed_one_and_a_missing_one(work, tmp_path, where, status):
    outside = tmp_path / "other.json"
    outside.write_bytes(CATALOG_BYTES)
    path = {"outside": outside, "missing": FIX / "no-such-fixture.json", "folder": FIX}.get(where, FIX / "catalog_v2_valid.json")
    work.git.status = status
    assert work.call("--upload", "--probe", path, env=upload_env(tmp_path)).code == 2 and work.runner.calls == []


def test_the_probe_upload_stops_on_a_hygiene_finding_before_the_process(work, tmp_path):
    result = work.call("--upload", "--probe", FIX / "catalog_v2_valid.json", env=upload_env(tmp_path, words="acme\n"))
    assert result.code == 2 and "hygiene" in result.err and work.runner.calls == []


def test_a_failed_probe_upload_is_exit_1(work, tmp_path):
    work.runner.code = 3
    assert work.call("--upload", "--probe", FIX / "catalog_v2_valid.json", env=upload_env(tmp_path)).code == 1
