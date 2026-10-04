"""The catalog publisher: takes a proposal into the table after the maintainer's word and carries it to the dataset (AP9-KA-P).

    python scripts/catalog_publish.py --show <out>
    python scripts/catalog_publish.py --apply <proposal.json> --proposal-sha <sha256> [--approve n,..] [--reject n|Kk,..] [--admit Kk,..]
    python scripts/catalog_publish.py --complete <proposal.json> --proposal-sha <sha256> --admit Kk --kind <kind>
    python scripts/catalog_publish.py --materialize <folder>
    python scripts/catalog_publish.py --record-commit <folder> <commit>
    python scripts/catalog_publish.py --publish-pending <folder> [--remote forgejo]
    python scripts/catalog_publish.py --verify [--commit <commit>] [--probe [<fixture>]]
    python scripts/catalog_publish.py --upload --probe <fixture>

One mode per call; a maintainer tool, not in the wheel. `--apply` takes exactly the numbers named, in two steps: first
`decision.json` with the target bytes of both files (state `approved`), then the files from those bytes (`--materialize`,
state `applied`). The state then runs on, each step written atomically: `committed` (`--record-commit`), `pushed`,
`uploaded`, `published` (`--publish-pending`; the upload carries the bytes of the recorded commit, never the working tree).
Exit 0 on success; exit 1 when a step of a process or of the network failed (the state stays); exit 2 when the call or an
input is wrong (nothing is written). There is no prompt, no `--yes` and no way to take "all". The rules are in
`catalog_publish_rules.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the other modules of the scripts sit beside this one
import catalog_publish_rules as P  # noqa: E402
import catalog_scan_rules as R  # noqa: E402
from catalog_scan import DiskFiles  # noqa: E402  (the scanner's one way to touch a disk: atomic writes, a test hands in a subclass)

from modelroom.catalog import CatalogError  # noqa: E402
from modelroom.catalog_remote import _FAULTS, _SHA_RE, DATASET, FILE_MAX_BYTES, REMOTE_FILE_NAME, REVISION_MAX_BYTES, _clean, _Fault, _get  # noqa: E402
from modelroom.catalog_schema import build_catalog_v2  # noqa: E402
from modelroom.contracts import SchemaVersionError  # noqa: E402
from modelroom.http import RequestBudget, Transport, UrllibTransport  # noqa: E402

PROBE_DATASET = "jm-vis/modelroom-catalog-probe"
PRIVATE_WORDS_ENV, TOKEN_PATH_ENV = "MODELROOM_RELEASE_PRIVATE_WORDS", "HF_TOKEN_PATH"
TOKEN_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")  # never handed to the upload; only the path is
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
PublishError = P.PublishError


class GitError(Exception):
    """A git call did not succeed; its text names the command and the exit code, never git's own output."""


class StepFailed(Exception):
    """A step of a process or of the network failed: exit 1, the state stays where it was."""


def default_git(root: Path):
    def run_git(args: list[str]) -> bytes:
        try:
            done = subprocess.run(["git", *args], cwd=root, capture_output=True)
        except OSError as exc:
            raise GitError("git cannot be started") from exc
        if done.returncode != 0:
            raise GitError(f"git {args[0]} failed (exit {done.returncode})")
        return done.stdout

    return run_git


def default_run(args: list[str], env: dict[str, str]) -> int:
    """The upload process: a fixed argument list, never a shell; its exit code."""
    return subprocess.run(args, env=env).returncode


class Tool:
    """One call. Every outside is a parameter: the disk (`files`), git (`run_git`), the one upload process (`run`), the
    network (`transport`), the environment (`env`, read only on the way to an upload) and the clock (`now`)."""

    def __init__(self, *, root: Path, catalog: Path, rejected: Path, files, run_git, run, transport: Transport, env,
                 now: datetime, out, err) -> None:
        self.root, self.catalog_path, self.rejected_path = root, catalog, rejected
        self.files, self.run_git, self.run, self.transport, self.env, self.out, self.err = files, run_git, run, transport, env, out, err
        self.today = now.astimezone(timezone.utc).date().isoformat()

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    # --- reading ----------------------------------------------------------------------------------------------

    def _read(self, path: Path, what: str) -> bytes:
        try:
            return self.files.read(path)
        except OSError as exc:
            raise PublishError(f"{path}: cannot read the {what}: {_clean(exc)}") from exc

    def _document(self, data: bytes) -> dict:
        """The catalog read as a table, validated; the bytes were compared before this is called."""
        try:
            document = json.loads(data.decode("utf-8"))
            build_catalog_v2(data.decode("utf-8"), self.catalog_path)
        except (ValueError, CatalogError, SchemaVersionError) as exc:
            raise PublishError(f"{self.catalog_path}: not a valid catalog: {_clean(exc)}") from exc
        return document

    def _proposal_bytes(self, path: Path) -> bytes:
        if path.name != "proposal.json":
            raise PublishError(f"{path}: the proposal is a file named proposal.json")
        return self._read(path, "proposal")

    def _parsed(self, path: Path, data: bytes) -> dict:
        try:
            proposal = json.loads(data.decode("utf-8"))
        except ValueError as exc:
            raise PublishError(f"{path}: the proposal is no JSON") from exc
        if not isinstance(proposal, dict) or not isinstance(proposal.get("base_catalog_sha256"), str):
            raise PublishError(f"{path}: not a proposal of the scanner")
        return proposal

    def _days(self, out: Path) -> list[str]:
        return sorted(name for name in self.files.folders(out) if _DAY.fullmatch(name))

    def _pending(self, out: Path, skip: str | None = None) -> list[tuple[str, dict]]:
        """Every day with a decision that is not `published`, read as the scanner reads it: a dict whose `state` is one
        of the six; anything else is no decision."""
        found = []
        for day in self._days(out):
            path = out / day / "decision.json"
            if day == skip or not self.files.exists(path):
                continue
            try:
                decision = json.loads(self.files.read(path).decode("utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(decision, dict) and decision.get("state") in P.STATES and decision["state"] != "published":
                found.append((day, decision))
        return found

    def _decision(self, folder: Path) -> dict:
        return P.check_decision(self._read(folder / "decision.json", "decision"))

    def _state(self, folder: Path, decision: dict, state: str, **more) -> dict:
        """The decision in a new state, written atomically; the one place a state changes."""
        decision = {**decision, "state": state, **more}
        self.files.write(folder / "decision.json", R.dump_json(decision))
        return decision

    def _commit_bytes(self, commit: str, expected_sha: str, what: str = P.CATALOG_IN_REPO) -> bytes:
        try:
            data = self.run_git(["show", f"{commit}:{what}"])
        except GitError as exc:
            raise StepFailed(str(exc)) from exc
        if P.sha256_of(data) != expected_sha:
            raise PublishError(f"{what} of this commit is not the result of the decision")
        return data

    # --- --show -----------------------------------------------------------------------------------------------

    def show(self, out: Path) -> int:
        pending = self._pending(out)
        newest = next((d for d in reversed(self._days(out)) if self.files.exists(out / d / "proposal.json")), None)
        open_one = newest if newest is not None and not self.files.exists(out / newest / "decision.json") else None
        for day, decision in pending:
            self.say(P.pending_line(day, decision))
        if open_one is None:
            if not pending:
                self.say("no open catalog proposal")
            return 0
        data = self._read(out / open_one / "proposal.json", "proposal")  # once: the checksum and the text are made from these bytes
        self.out.write(P.render_open(data))
        self.say(P.base_line(data, P.sha256_of(self._read(self.catalog_path, "catalog")))[1])
        return 0

    # --- --apply, --complete, --materialize -----------------------------------------------------------------------

    def _checked(self, proposal_path: Path, given: str | None) -> tuple[bytes, dict, str, dict, bytes, bytes]:
        """The common refusals of `--apply` and `--complete`, in the brief's order: the checksum's form, the bytes that
        were shown, the base the proposal was made against; only then is anything parsed or validated."""
        data = self._proposal_bytes(proposal_path)
        sha = P.check_sha(given, data)
        proposal = self._parsed(proposal_path, data)
        catalog_bytes = self._read(self.catalog_path, "catalog")
        if proposal["base_catalog_sha256"] != P.sha256_of(catalog_bytes):
            raise PublishError("the proposal was made against another catalog; run the scanner again")
        return data, proposal, sha, self._document(catalog_bytes), catalog_bytes, self._read(self.rejected_path, "rejection list")

    def _free(self, folder: Path) -> None:
        if self.files.exists(folder / "decision.json"):
            raise PublishError(f"{folder}: a decision exists for this proposal")
        other = self._pending(folder.parent, skip=folder.name)
        if other:
            raise PublishError(f"another publication is pending ({other[0][0]}, state {other[0][1]['state']}); one at a time")

    def apply(self, proposal_path: Path, given: str | None, approve, reject, admit) -> int:
        _, proposal, sha, document, catalog_bytes, rejected_bytes = self._checked(proposal_path, given)
        request = P.parse_request(approve, reject, admit)
        P.check_request(proposal, request)
        folder = proposal_path.parent
        self._free(folder)
        plan = P.plan_apply(document, catalog_bytes, rejected_bytes, proposal, request, self.today)
        decision = P.decision_document(plan, sha, self.today, P.sha256_of(catalog_bytes), P.sha256_of(rejected_bytes))
        self.files.write(folder / "decision.json", R.dump_json(decision))  # step 1: from here the word is safe
        self._materialize(folder, decision)  # step 2
        self.say(f"applied: approved {plan.approved}, rejected {plan.rejected}, admitted {plan.admitted}; result sha256 {decision['result_sha256']}")
        return 0

    def complete(self, proposal_path: Path, given: str | None, admit: str | None, kind: str | None) -> int:
        _, proposal, _, document, _, _ = self._checked(proposal_path, given)
        request = P.parse_request(None, None, admit)
        folder = proposal_path.parent
        self._free(folder)
        data, number = P.complete_proposal(document, proposal, request, kind, self.today)
        raw = R.dump_json(data)
        text = R.render_proposal(data).encode("utf-8")
        self.files.write(folder / "proposal.json", raw)
        self.files.write(folder / "proposal.md", text)
        self.say(f"row {number} made from {request.admit[0]}")
        self.say(f"proposal.json sha256: {P.sha256_of(raw)}")
        return 0

    def materialize(self, folder: Path) -> int:
        decision = self._decision(folder)
        if decision["state"] != "approved":
            raise PublishError(f"the state is {decision['state']}: --materialize works on approved only")
        self._materialize(folder, decision)
        self.say("applied: both files hold the result of the decision")
        return 0

    def _materialize(self, folder: Path, decision: dict) -> None:
        """Step 2 from the bytes of the decision: a file with the target checksum stays, one with the base checksum gets the
        target, any other is somebody else's change and nothing is written."""
        targets = [
            (self.catalog_path, "catalog", decision["base_catalog_sha256"], decision["result_sha256"], decision["result_catalog"]),
            (self.rejected_path, "rejection list", decision["base_rejected_sha256"], decision["result_rejected_sha256"], decision["result_rejected"]),
        ]
        todo = []
        for path, what, base, target, text in targets:
            have = P.sha256_of(self._read(path, what))
            if have == target:
                continue
            if have != base:
                raise PublishError(f"{path}: the {what} is neither the base nor the result of the decision; nothing is written, the state stays approved")
            todo.append((path, text.encode("utf-8")))
        for path, data in todo:
            self.files.write(path, data)
        self._state(folder, decision, "applied")

    # --- --record-commit, --publish-pending -----------------------------------------------------------------------

    def record_commit(self, folder: Path, commit: str) -> int:
        """The commit that holds the result of the decision, both files; a commit recorded already is confirmed again."""
        P.check_commit(commit)
        decision = self._decision(folder)
        if decision["state"] != "applied":
            if decision["state"] != "approved" and decision.get("commit") == commit:
                self.say(f"committed: {commit}")
                return 0
            raise PublishError(f"the state is {decision['state']}: --record-commit works on applied only")
        try:
            if self.run_git(["cat-file", "-t", commit]).strip() != b"commit":  # a tree with the right files is no commit to push
                raise PublishError(f"{commit} is not a commit")
            for what, digest in ((P.CATALOG_IN_REPO, "result_sha256"), (P.REJECTED_IN_REPO, "result_rejected_sha256")):
                self._commit_bytes(commit, decision[digest], what)
        except (GitError, StepFailed) as exc:
            raise PublishError(str(exc)) from exc
        self._state(folder, decision, "committed", commit=commit)
        self.say(f"committed: {commit}")
        return 0

    def publish_pending(self, folder: Path, remote: str) -> int:
        decision = self._decision(folder)
        state = decision["state"]
        if state in ("approved", "applied"):
            raise PublishError(f"the state is {state}: there is no commit to publish yet")
        if state == "published":
            self.say("published: nothing to do")
            return 0
        commit, remote = P.check_commit(decision.get("commit") or ""), P.check_remote(remote)
        data = self._commit_bytes(commit, decision["result_sha256"])
        token = self._ready(data) if state in ("committed", "pushed") else None  # every refusal of the upload comes before the push
        if state == "committed":
            try:
                self.run_git(["push", remote, f"{commit}:refs/heads/main"])
            except GitError as exc:
                raise StepFailed(f"the push failed, the state stays committed: {exc}") from exc
            decision, state = self._state(folder, decision, "pushed"), "pushed"
            self.say("pushed")
        if state == "pushed":
            self._upload(DATASET, data, commit, token)
            decision, state = self._state(folder, decision, "uploaded"), "uploaded"
            self.say("uploaded")
        if self._verified(DATASET, data, commit) != 0:
            raise StepFailed("the check of the published file failed, the state stays uploaded")
        self._state(folder, decision, "published")
        self.say("published")
        return 0

    # --- the upload and the check of what is published ---------------------------------------------------------------

    def _words(self) -> list[str]:
        """The maintainer's private words, from the file the variable names: no file, no upload."""
        location = self.env.get(PRIVATE_WORDS_ENV, "").strip()
        if not location:
            raise PublishError(f"{PRIVATE_WORDS_ENV} is not set: it names the file of private words")
        try:
            path = Path(location)
            if path.resolve().is_relative_to(self.root.resolve()):
                raise PublishError(f"the file {PRIVATE_WORDS_ENV} names lies inside the repository; keep it outside")
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise PublishError(f"the file {PRIVATE_WORDS_ENV} names cannot be read") from exc
        return sorted({line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}, key=len, reverse=True)

    def _hygiene(self, data: bytes) -> None:
        """The release gate's own reading of a file (`scripts/release-check.py`, loaded as it is) with the private words."""
        words = self._words()
        try:
            spec = importlib.util.spec_from_file_location("release_check", self.root / "scripts" / "release-check.py")
            gate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gate)
            findings = gate.scan_data(REMOTE_FILE_NAME, REMOTE_FILE_NAME, data, gate.with_private_words(gate.load_negative_list(), words))
        except (OSError, AttributeError, ImportError, SyntaxError, TypeError) as exc:
            raise PublishError(f"the hygiene check cannot run: {type(exc).__name__}") from exc
        if findings:
            raise PublishError("the file fails the hygiene check, nothing is uploaded: " + "; ".join(findings[:5]))

    def _ready(self, data: bytes) -> str:
        """What an upload needs, checked before any step is taken: the hygiene gate over these bytes (with the private
        words) and the token path (never its content). Returns the token path."""
        self._hygiene(data)
        token = self.env.get(TOKEN_PATH_ENV, "")
        if not token or not os.path.isfile(token) or not os.access(token, os.R_OK):
            raise PublishError(f"{TOKEN_PATH_ENV} is not set or names no readable file")
        return token

    def _upload(self, dataset: str, data: bytes, message: str, token: str) -> None:
        """The one upload process, for these bytes: a file of its own in a folder of its own, read back and checked by
        the hygiene gate again, then the fixed argument list with the token path in the environment. `_ready` came
        before: what fails here is a step, exit 1, the state stays."""
        with tempfile.TemporaryDirectory(prefix="catalog-upload-") as folder:
            target = Path(folder) / REMOTE_FILE_NAME
            try:
                target.write_bytes(data)
                again = target.read_bytes()
                if P.sha256_of(again) != P.sha256_of(data):
                    raise StepFailed("the upload file does not hold the bytes it was made from, the state stays")
                self._hygiene(again)
            except (OSError, PublishError) as exc:
                raise StepFailed(f"the upload file could not be made and checked, the state stays: {_clean(exc)}") from exc
            child = {k: v for k, v in self.env.items() if k not in TOKEN_VARIABLES}
            child[TOKEN_PATH_ENV] = token
            try:
                code = self.run(P.upload_args(dataset, target, message), child)
            except OSError as exc:
                raise StepFailed(f"the upload process cannot be started: {type(exc).__name__}") from exc
            if code != 0:
                raise StepFailed(f"the upload failed (exit {code}), the state stays")

    def _published(self, dataset: str) -> bytes:
        """The published file as modelroom reads it, without a token: the revision of the main line, then the file at it."""
        budget = RequestBudget(3)  # the three requests of an attempt, booked
        base = "https://huggingface.co"
        sha = _get(self.transport, budget, 1, f"{base}/api/datasets/{dataset}/revision/main", REVISION_MAX_BYTES).json()
        sha = sha.get("sha") if isinstance(sha, dict) else None
        if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
            raise _Fault("the revision names no commit")
        answer = _get(self.transport, budget, 2, f"{base}/datasets/{dataset}/resolve/{sha}/{REMOTE_FILE_NAME}", FILE_MAX_BYTES)
        if answer.header("X-Repo-Commit") != sha:
            raise _Fault("the file is not from the commit that was asked for")
        return answer.body

    def _verified(self, dataset: str, expected: bytes, label: str) -> int:
        try:
            published = self._published(dataset)
        except _FAULTS as exc:
            self.say(f"catalog: the published file could not be read ({_clean(exc)})")
            return 1
        if published == expected:
            self.say(f"catalog: published file equals {label}")
            return 0
        self.say(f"ALARM: the published catalog differs from {label}")
        return 1

    def verify(self, commit: str | None, probe: str | None) -> int:
        dataset = DATASET if probe is None else PROBE_DATASET
        if commit is not None:
            P.check_commit(commit)
            if probe:
                raise PublishError("--commit and a probe file name two things to compare with")
            try:
                expected = self.run_git(["show", f"{commit}:{P.CATALOG_IN_REPO}"])
            except GitError as exc:
                self.say(f"catalog: the commit cannot be read ({exc})")
                return 1
            return self._verified(dataset, expected, commit)
        path = Path(probe) if probe else self.catalog_path
        return self._verified(dataset, self._read(path, "file to compare with"), "the probe file" if probe else "working tree")

    def upload_probe(self, fixture: str) -> int:
        """The setup test: a fixture of the repository, unchanged, to the probe dataset only."""
        fixtures, path = (self.root / "tests" / "fixtures").resolve(), Path(fixture).resolve()
        if not (path.is_file() and path.is_relative_to(fixtures)):
            raise PublishError("the probe file must be a file under tests/fixtures")
        inside = ":(literal)" + path.relative_to(self.root.resolve()).as_posix()  # a name, never a pathspec pattern
        try:
            tracked = self.run_git(["ls-files", "--", inside])  # an ignored file has no status line: it must be in the index
            status = self.run_git(["status", "--porcelain", "--", inside])
        except GitError as exc:
            raise PublishError(f"git cannot tell whether the file is unchanged: {exc}") from exc
        if not tracked.strip() or status.strip():
            raise PublishError("the probe file is not a tracked, unchanged file of the repository")
        data = self._read(path, "probe file")
        self._upload(PROBE_DATASET, data, "probe", self._ready(data))
        self.say(f"uploaded the probe file to {PROBE_DATASET}")
        return 0


# --- the call -----------------------------------------------------------------------------------------------------------

MODES = ("show", "apply", "complete", "materialize", "record_commit", "publish_pending", "verify", "upload")
OPTIONS = ("proposal_sha", "approve", "reject", "admit", "kind", "remote", "commit", "probe", "catalog", "rejected")
ALLOWED = {
    "show": set(), "apply": {"proposal_sha", "approve", "reject", "admit"}, "complete": {"proposal_sha", "admit", "kind"},
    "materialize": set(), "record_commit": set(), "publish_pending": {"remote"}, "verify": {"commit", "probe"}, "upload": {"probe"},
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catalog_publish.py", allow_abbrev=False,
                                     description="Take a proposal into the catalog after the maintainer's word, and publish it.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--show", metavar="OUT")
    group.add_argument("--apply", metavar="PROPOSAL")
    group.add_argument("--complete", metavar="PROPOSAL")
    group.add_argument("--materialize", metavar="FOLDER")
    group.add_argument("--record-commit", nargs=2, metavar=("FOLDER", "COMMIT"))
    group.add_argument("--publish-pending", metavar="FOLDER")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--upload", action="store_true")
    for name in ("proposal-sha", "approve", "reject", "admit", "kind", "remote", "commit"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--probe", nargs="?", const="")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--rejected", type=Path)
    return parser


def main(argv: list[str] | None = None, *, root: Path | None = None, files=None, run_git=None, run=None, transport: Transport | None = None,
         env=None, now: datetime | None = None, out=None, err=None) -> int:
    root = (root or Path(__file__).resolve().parent.parent).resolve()
    out, err = out or sys.stdout, err or sys.stderr
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    mode = next(name for name in MODES if getattr(args, name) not in (None, False))
    stray = sorted(name for name in OPTIONS if getattr(args, name) is not None and name not in ALLOWED[mode] | {"catalog", "rejected"})
    if stray or (mode == "upload" and not args.probe):
        print("error: " + (f"options of another mode: --{', --'.join(n.replace('_', '-') for n in stray)}" if stray else "--upload needs --probe <fixture>"), file=err)
        return 2
    tool = Tool(
        root=root, catalog=args.catalog or root / "modelroom" / "catalog.json", rejected=args.rejected or root / "modelroom" / "catalog-rejected.json",
        files=files or DiskFiles(), run_git=run_git or default_git(root), run=run or default_run, transport=transport or UrllibTransport(),
        env=os.environ if env is None else env, now=now or datetime.now(timezone.utc), out=out, err=err,  # not copied: a token-free mode never reads it
    )
    try:
        if mode == "show":
            return tool.show(Path(args.show))
        if mode == "apply":
            return tool.apply(Path(args.apply), args.proposal_sha, args.approve, args.reject, args.admit)
        if mode == "complete":
            return tool.complete(Path(args.complete), args.proposal_sha, args.admit, args.kind)
        if mode == "materialize":
            return tool.materialize(Path(args.materialize))
        if mode == "record_commit":
            return tool.record_commit(Path(args.record_commit[0]), args.record_commit[1])
        if mode == "publish_pending":
            return tool.publish_pending(Path(args.publish_pending), args.remote or "forgejo")
        if mode == "verify":
            return tool.verify(args.commit, args.probe)
        return tool.upload_probe(args.probe)
    except PublishError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except StepFailed as exc:
        print(f"failed: {exc}", file=err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
