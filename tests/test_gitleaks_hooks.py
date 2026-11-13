"""Tests of the secret scan through the real git hooks in `.githooks/`, run in temporary repositories.

Local only (never in CI): the hooks need `uv` and the project environment. Skipped with a reason unless
`GITLEAKS_HOOK_TESTS=1` and `GITLEAKS_TEST_BINARY` are set; with `GITLEAKS_REQUIRED=1` a missing
setting fails instead. Every temporary repository is built from the index of this repository
(`git checkout-index`), has an empty global git configuration, and sets `core.hooksPath` and
`hooks.gitleaks` only in its own local configuration. The remote is always a bare repository in the
temporary folder. Every bait is made at run time.

Failure messages carry only fixed texts, counts and truth values: every check reduces the output it reads
to a truth value first, and an unexpected exception in a test step is replaced by its type name and line
number (`SafeCase`). A refusal is told apart: a finding names its rule and the gate's summary line, an
error names its class, and its return value is exactly 1 (a hook refuses with 1; git commit and git push
end with 1 when a hook refuses). Every run that loads this file as a module reports a failure, an error of a
test, of a subtest or of the module setup only as its test id, its fixed text and line numbers of this file:
no traceback, so no file path (`load_tests`).
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import io
import os
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
BINARY = os.environ.get("GITLEAKS_TEST_BINARY", "")
REQUIRED = os.environ.get("GITLEAKS_REQUIRED") == "1"
HOOK_TESTS = os.environ.get("GITLEAKS_HOOK_TESTS") == "1"
UV_ENVIRONMENT = os.environ.get("GITLEAKS_HOOK_UV_ENV") or str(ROOT / ".venv")
R = random.SystemRandom()
B62 = string.ascii_letters + string.digits
WORK: Path | None = None  # the temporary folder of a run, made by `setUpModule`
EMPTY_CONFIG: Path | None = None
REFUSED = 1  # the return value of a refusal: of a hook, and of git commit and git push when a hook refuses
SYNTHETIC_FILES = (
    ".githooks/pre-commit", ".githooks/commit-msg", ".githooks/pre-push", "scripts/gitleaks_gate.py",
    "scripts/gitleaks_findings.tmpl", ".gitleaks.toml", ".gitleaks-version", ".gitleaksignore",
    "tests/test_public_hygiene.py", "pyproject.toml", "uv.lock",
)
LINE_FORM = re.compile(r"[A-Za-z0-9._-]+ (?:file:[0-9a-f]{12}:[0-9]+|tag:[0-9a-f]{12}:[0-9]+|message:[0-9]+) "
                       r"(?:[0-9a-f]{8}|-|staged)")
RAW_NAME = b"n\xff\xfeam\xc3\xa4e\xe2\x82.txt"  # not UTF-8: two bad bytes, a valid letter, a cut multi-byte character


class FixedError(RuntimeError):
    """A failure of the test setup with a fixed text: never a path, a value or an output."""


def setUpModule():
    global WORK, EMPTY_CONFIG
    try:
        WORK = Path(tempfile.mkdtemp(prefix="glhook_"))
        EMPTY_CONFIG = WORK / "empty.gitconfig"
        EMPTY_CONFIG.write_text("", encoding="utf-8")
    except OSError:
        raise FixedError("the temporary folder of the tests cannot be made") from None


def tearDownModule():
    if WORK is None:
        return

    def writable(function, path, _info):
        os.chmod(path, stat.S_IWRITE)
        function(path)

    try:
        shutil.rmtree(WORK, onerror=writable)
    except OSError:
        raise FixedError("the temporary folder of the tests cannot be removed") from None


# ---------------------------------------------------------------------------------------------
# failure reports without paths
# ---------------------------------------------------------------------------------------------

HERE = os.path.basename(__file__)


def lines_here(tb) -> list[int]:
    """The line numbers of this file in a traceback, outermost first, without the frames that only wrap a step."""
    found = []
    while tb is not None:
        code = tb.tb_frame.f_code
        if os.path.basename(code.co_filename) == HERE and code.co_name not in ("wrapper", "case"):
            found.append(tb.tb_lineno)
        tb = tb.tb_next
    return found


def report_of(err: tuple) -> tuple:
    """An exception as the runner gets to print it: type, fixed text and line numbers, no traceback (no path).

    Only the text of a failed check or of a setup failure of this file is kept, both are fixed; any other
    exception is named by its type alone.
    """
    kind, value, tb = err
    lines = lines_here(tb)
    where = f" (line {' > '.join(str(number) for number in lines)})" if lines else ""
    if isinstance(value, AssertionError):
        return AssertionError, AssertionError(f"{value}{where}"), None
    text = str(value) if isinstance(value, FixedError) else f"unexpected {kind.__name__}"
    return FixedError, FixedError(f"{text}{where}"), None


def quiet(result) -> None:
    """Makes `result` print every failure and error of this file through `report_of`; other files keep theirs."""
    marks = result.__dict__.setdefault("quiet_modules", set())
    if __name__ in marks:
        return
    marks.add(__name__)
    own = re.compile(rf"(?:^|\(){re.escape(__name__)}[.)]")

    def wrap(original):
        def report(test, err):
            return original(test, report_of(err) if own.search(test.id()) else err)
        return report

    for name in ("addError", "addFailure", "addExpectedFailure"):
        if hasattr(result, name):
            setattr(result, name, wrap(getattr(result, name)))
    if hasattr(result, "addSubTest"):
        original_subtest = result.addSubTest

        def report_subtest(test, subtest, err):
            return original_subtest(test, subtest, report_of(err) if err is not None and own.search(test.id()) else err)
        result.addSubTest = report_subtest


class QuietSuite(unittest.TestSuite):
    def run(self, result, debug=False):
        quiet(result)
        return super().run(result, debug)


def load_tests(loader, tests, pattern):
    """Discovery and `unittest.main` load this file through here: its suite reports through `report_of`."""
    return QuietSuite([tests])


# ---------------------------------------------------------------------------------------------
# failure messages without content
# ---------------------------------------------------------------------------------------------

def unexpected(error: BaseException) -> str:
    line = 0
    for frame in traceback.extract_tb(error.__traceback__):
        if os.path.basename(frame.filename) == os.path.basename(__file__):
            line = frame.lineno or 0
    return f"unexpected {type(error).__name__} in a test step (line {line})"


def sanitized(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except (AssertionError, unittest.SkipTest):
            raise
        except Exception as error:  # its text may name a path, an argument or an output
            raise AssertionError(unexpected(error)) from None
    return wrapper


class SafeCase(unittest.TestCase):
    """Fixed failure texts only: `longMessage` is off, tests and `setUp` are wrapped, `case` wraps a subtest."""

    longMessage = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if callable(value) and (name.startswith("test") or name in ("setUp", "tearDown")):
                setattr(cls, name, sanitized(value))

    @contextlib.contextmanager
    def case(self, **params):
        with self.subTest(**params):
            try:
                yield
            except (AssertionError, unittest.SkipTest):
                raise
            except Exception as error:
                raise AssertionError(unexpected(error)) from None

    def assert_refused_by_finding(self, output: str, mode: str, what: str):
        """A real finding: the gate's summary line of `mode` is there and no error line of the gate."""
        self.assertTrue(is_finding(output, mode), f"{what}: the refusal is not a finding of the gate")

    def assert_refused(self, returncode: int, what: str):
        """A refusal: exactly 1, from the hook itself or from git commit and git push after a hook refused."""
        self.assertEqual(returncode, REFUSED, f"{what}: the return value is not 1 (refused)")

    def assert_no_leak(self, output: str, values: list[str], names: list[str], what: str):
        self.assertEqual(leaked(output, values), 0, f"{what}: a piece of a bait is in the output")
        self.assertEqual(sum(1 for name in names for piece in windows_of(name) | {name} if piece in output), 0,
                         f"{what}: a piece of a name is in the output")


def is_finding(output: str, mode: str) -> bool:
    lines = output.splitlines()
    summary = any(re.fullmatch(rf"gitleaks_gate: {re.escape(mode)}: [0-9]+ finding\(s\), refused", line) for line in lines)
    errors = any(re.match(r"gitleaks_gate: [a-z-]+: error ", line) for line in lines)
    return summary and not errors


def has_line(output: str, pattern: str) -> bool:
    return any(re.fullmatch(pattern, line) for line in output.splitlines())


def error_only(out: str, err: str, label: str, kind: str) -> bool:
    lines = err.splitlines()
    return out == "" and len(lines) == 1 and re.fullmatch(
        re.escape(f"gitleaks_gate: {label}: error {kind}") + r"(?: rc=-?[0-9]+)?", lines[0]) is not None


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def rnd(n: int, alphabet: str = B62) -> str:
    return "".join(R.choice(alphabet) for _ in range(n))


UPPER = string.ascii_uppercase
LOWER = string.ascii_lowercase + string.digits  # folder names: never an upper-case letter
RARE = "QXZJ"


def spread(count: int, alphabet: str) -> list[str]:
    """`count` symbols of `alphabet` in random order, each one floor(count/len) or ceil(count/len) times."""
    full, rest = divmod(count, len(alphabet))
    items = list(alphabet) * full + R.sample(alphabet, rest)
    R.shuffle(items)
    return items


def token(n: int, letters: str = UPPER) -> str:
    """A random value of n characters that alternates letter and digit, starting with a letter.

    Baits and names use this form: it meets its rule on every run, and upper-case letters next to digits
    occur in no output of git, of the hooks or of the gate by chance (object ids, temporary names and the
    folder names of these tests are lower case), so a check that no piece of a bait or a name shows in
    the output cannot fail at random.
    """
    head = spread((n + 1) // 2, letters)
    tail = spread(n // 2, string.digits)
    return "".join(head[i // 2] if i % 2 == 0 else tail[i // 2] for i in range(n))


def with_marks(value: str, marks: str = "!#%+") -> str:
    """`value` with each mark put in at a random place."""
    chars = list(value)
    for mark in marks:
        chars.insert(R.randrange(len(chars) + 1), mark)
    return "".join(chars)


def env_for(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    for key in ("GITLEAKS_CONFIG", "GITLEAKS_CONFIG_TOML", "GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        env.pop(key, None)
    env.update(GIT_CONFIG_GLOBAL=str(EMPTY_CONFIG), GIT_CONFIG_NOSYSTEM="1", GIT_ALLOW_PROTOCOL="file",
               GIT_AUTHOR_NAME="probe", GIT_COMMITTER_NAME="probe",
               GIT_AUTHOR_EMAIL="probe" + "@" + "example.invalid", GIT_COMMITTER_EMAIL="probe" + "@" + "example.invalid",
               UV_PROJECT_ENVIRONMENT=UV_ENVIRONMENT, UV_NO_SYNC="1", UV_OFFLINE="1")
    env.update(extra or {})
    return env


def run_ok(command: list[str], what: str):
    done = subprocess.run(command, capture_output=True, env=env_for())
    if done.returncode != 0:
        raise AssertionError(f"setup step failed: {what} (rc {done.returncode})")
    return done


def git(repo: Path, *args: str, data: bytes | None = None, env: dict | None = None, check: bool = True):
    done = subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                          input=data, capture_output=True, env=env or env_for())
    if check and done.returncode != 0:
        raise AssertionError(f"git setup step failed ({args[0]}, rc {done.returncode})")
    return done


def text_of(done) -> str:
    return done.stdout.decode("utf-8", "replace") + done.stderr.decode("utf-8", "replace")


def write(repo: Path, rel: str, text: str | bytes) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").stdout.decode().strip()


def count(repo: Path) -> int:
    return int(git(repo, "rev-list", "--count", "HEAD").stdout.decode().strip())


def name_id(name: str | bytes) -> str:
    return hashlib.sha256(name.encode("utf-8") if isinstance(name, str) else name).hexdigest()[:12]


def windows_of(value: str) -> set[str]:
    out = set()
    for part in value.split("\n"):
        out |= {part[i:i + 8] for i in range(len(part) - 7)}
    return out


def leaked(output: str, values: list[str]) -> int:
    return sum(1 for value in values for piece in windows_of(value) if piece in output)


def github_token() -> str:
    return "gh" + "p_" + token(36)


def baits() -> dict[str, str]:
    # The word forms use rare upper-case letters: every 8-character piece holds four of them in a row next
    # to a space, which no output of git, the hooks or the gate contains.
    return {
        "github_pat": github_token(), "passwort_quotiert": with_marks(token(20)),
        "phrase": " ".join(rnd(5, RARE) for _ in range(4)),
        "app_passwort": " ".join(rnd(4, RARE) for _ in range(4)),
        "commit_nachricht": github_token(),  # a form every run detects, unlike a random word
    }


def export_index(dest: Path) -> None:
    """The index of this repository, written to `dest` (the full tracked state, nothing else)."""
    prefix = str(dest).replace("\\", "/").rstrip("/") + "/"
    run_ok(["git", "-C", str(ROOT), "checkout-index", "-a", f"--prefix={prefix}"], "checkout-index")


def init_repo(repo: Path) -> None:
    run_ok(["git", "init", "-q", "-b", "main", str(repo)], "git init")


def init_bare(remote: Path) -> None:
    run_ok(["git", "init", "-q", "--bare", "-b", "main", str(remote)], "git init --bare")


def activate(repo: Path, binary: str | None = BINARY) -> None:
    git(repo, "config", "core.hooksPath", ".githooks")
    if binary:
        git(repo, "config", "hooks.gitleaks", binary)


def build_normal(name: str | None = None) -> tuple[Path, Path]:
    """Aufbau B: full tracked state, base commit pushed to a bare remote, hooks then active."""
    repo, remote = build_history(name)
    activate(repo)
    return repo, remote


def build_history(name: str | None = None) -> tuple[Path, Path]:
    """Aufbau C: like B up to the pushed base, but the hooks stay inactive until `activate`."""
    repo = WORK / (name or "h" + rnd(8, LOWER))
    export_index(repo)
    init_repo(repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    remote = WORK / ("remote_" + rnd(8, LOWER) + ".git")
    init_bare(remote)
    git(repo, "remote", "add", "origin", str(remote).replace("\\", "/"))
    git(repo, "push", "-q", "origin", "main")
    return repo, remote


def build_unborn(extra: dict[str, str] | None = None) -> Path:
    """Aufbau A: a clean synthetic index, hooks active, no commit yet. Copies keep the mode of the hooks."""
    repo = WORK / ("u" + rnd(8, LOWER))
    for rel in SYNTHETIC_FILES:
        if (ROOT / rel).exists():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, repo / rel)
    if os.name != "nt":
        for name in ("pre-commit", "commit-msg", "pre-push"):
            path = repo / ".githooks" / name
            if not os.access(path, os.X_OK):
                raise AssertionError("setup step failed: a copied hook is not executable")
    write(repo, "notes.txt", "plain notes\n")
    for rel, text in (extra or {}).items():
        write(repo, rel, text)
    init_repo(repo)
    activate(repo)
    git(repo, "add", "-A")
    return repo


def commit_hooked(repo: Path, message: str | bytes = "work", env: dict | None = None):
    data = message.encode("utf-8") if isinstance(message, str) else message
    return git(repo, "commit", "-q", "-F", "-", data=data, env=env or env_for(), check=False)


def stage_commit_hooked(repo: Path, message: str | bytes = "work"):
    git(repo, "add", "-A")
    return commit_hooked(repo, message)


def commit_plain(repo: Path, message: str | bytes = "work") -> str:
    git(repo, "add", "-A")
    data = message.encode("utf-8") if isinstance(message, str) else message
    git(repo, "commit", "-q", "-F", "-", data=data)
    return head(repo)


def add_raw_name(repo: Path, raw: bytes, text: str) -> None:
    """Puts a file with a raw byte name into the index, without the file system (works on every platform)."""
    blob = git(repo, "hash-object", "-w", "--stdin", data=text.encode("utf-8")).stdout.decode().strip()
    git(repo, "update-index", "--add", "--index-info", data=b"100644 " + blob.encode("ascii") + b"\t" + raw + b"\n")


def gate(repo: Path, *args: str, stdin: bytes = b""):
    done = subprocess.run([sys.executable, str(repo / "scripts" / "gitleaks_gate.py"), *args], input=stdin,
                          capture_output=True, env=env_for(), cwd=str(repo))
    return done.returncode, done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace")


def lookup_in_process(repo: Path, identifier: str, words: tuple[str, ...], change=None):
    """The lookup of `repo` run inside this process; every answer of the git command `words` goes through `change`.

    Returns the result and the original return value of each answer of `words`, the precondition of a change.
    """
    spec = importlib.util.spec_from_file_location("gate_" + rnd(8, LOWER), repo / "scripts" / "gitleaks_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_run, originals = subprocess.run, []

    def run(command, *a, **kw):
        done = real_run(command, *a, **kw)
        if list(command[:1]) == ["git"] and list(command[3:3 + len(words)]) == list(words):
            originals.append(done.returncode)
            if change is not None:
                return subprocess.CompletedProcess(command, *change(done.returncode, done.stdout), b"")
        return done

    out, err = io.TextIOWrapper(io.BytesIO(), encoding="utf-8"), io.StringIO()
    with mock.patch.dict(os.environ, env_for(), clear=True), mock.patch.object(subprocess, "run", run), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = module.main(["lo" + "cate", identifier])
    out.flush()
    return (rc, out.buffer.getvalue().decode("utf-8", "replace"), err.getvalue()), originals


def shell() -> Path:
    """The real POSIX shell of git: on Windows the MSYS `sh.exe` itself, which keeps the given PATH."""
    if os.name == "nt":
        return Path(shutil.which("git")).resolve().parent.parent / "usr" / "bin" / "sh.exe"
    return Path(shutil.which("sh"))


def run_pre_push_directly(repo: Path, remote: Path, line: bytes, tmp: Path, fake_cat: bool = False,
                          uname: str | None = None):
    """The tracked `pre-push` hook run by the real shell; first on PATH, if asked, a `cat` that fails and a
    `uname` that prints the given system name."""
    sh = shell()
    dirs = [str(sh.parent), os.environ.get("PATH", "")]
    fakes = {}
    if fake_cat:
        fakes["cat"] = b"#!/bin/sh\nexit 1\n"
    if uname is not None:
        fakes["uname"] = b"#!/bin/sh\necho " + uname.encode("ascii") + b"\n"
    if fakes:
        fake = WORK / ("fakebin" + rnd(6, LOWER))
        fake.mkdir()
        for name, text in fakes.items():
            (fake / name).write_bytes(text)
            (fake / name).chmod(0o755)
        dirs.insert(0, str(fake))
    env = env_for({"PATH": os.pathsep.join(dirs), "TMPDIR": str(tmp).replace("\\", "/")})
    return subprocess.run([str(sh), ".githooks/pre-push", "origin", str(remote).replace("\\", "/")], input=line,
                          capture_output=True, env=env, cwd=str(repo))


class HookCase(SafeCase):
    def setUp(self):
        missing = []
        if not (BINARY and os.path.isfile(BINARY)):
            missing.append("GITLEAKS_TEST_BINARY")
        if not HOOK_TESTS:
            missing.append("GITLEAKS_HOOK_TESTS=1")
        if missing:
            if REQUIRED:
                self.fail("not set: " + ", ".join(missing))
            self.skipTest("not set: " + ", ".join(missing))


class P2CommitMessage(HookCase):
    def test_p2_a_bait_in_the_message_refuses_the_commit(self):
        repo, _ = build_normal()
        values = baits()
        before = count(repo)
        write(repo, "clean.txt", "clean\n")
        subject = "msg" + token(10)
        done = stage_commit_hooked(repo, subject + "\n\ntoken=" + values["commit_nachricht"] + "\n")
        output = text_of(done)
        self.assert_refused(done.returncode, "commit with a bait in the message")
        self.assertEqual(count(repo), before, "a commit was made")
        self.assert_refused_by_finding(output, "commit-msg", "bait in the message")
        self.assertTrue(has_line(output, r"github-pat message:3 staged"), "the finding line of the message is missing")
        self.assert_no_leak(output, list(values.values()), [subject], "bait in the message")

    def test_p2_crlf_and_non_utf8_messages_do_not_crash_and_a_clean_one_passes(self):
        repo, _ = build_normal()
        before = count(repo)
        write(repo, "clean.txt", "clean\n")
        done = stage_commit_hooked(repo, b"subject line\r\n\r\nbody with \xe9\xff bytes\r\n")
        self.assertEqual(done.returncode, 0, "a clean crlf message was refused")
        self.assertTrue(has_line(text_of(done), r"gitleaks_gate: commit-msg: clean"), "the message hook did not run")
        self.assertFalse("Traceback" in text_of(done), "a traceback is in the output")
        self.assertEqual(count(repo), before + 1, "no commit was made")
        write(repo, "second.txt", "clean too\n")
        self.assertEqual(stage_commit_hooked(repo, "an ordinary message").returncode, 0, "a clean message was refused")

    def test_p2_a_clean_commit_in_a_linked_worktree_passes_the_message_hook(self):
        # git names the message file of a linked worktree by an absolute path; under a Windows shell it
        # starts with a drive.
        repo, _ = build_normal()
        tree = WORK / ("w" + rnd(8, LOWER))
        git(repo, "worktree", "add", "-q", "-b", "side", str(tree))
        self.assertTrue(os.path.isabs(git(tree, "rev-parse", "--git-dir").stdout.decode().strip()),
                        "precondition: the git folder of the linked worktree is absolute")
        before = count(tree)
        write(tree, "clean.txt", "clean\n")
        done = stage_commit_hooked(tree, "an ordinary message")
        self.assertEqual(done.returncode, 0, "a clean commit in a linked worktree was refused")
        self.assertTrue(has_line(text_of(done), r"gitleaks_gate: commit-msg: clean"),
                        "the message hook did not pass in a linked worktree")
        self.assertEqual(count(tree), before + 1, "no commit was made in the linked worktree")

    def test_p2_a_bait_in_a_crlf_message_is_found(self):
        repo, _ = build_normal()
        values = baits()
        write(repo, "clean.txt", "clean\n")
        before = count(repo)
        done = stage_commit_hooked(repo, ("subject\r\n\r\ntoken=" + values["commit_nachricht"] + "\r\n").encode())
        output = text_of(done)
        self.assert_refused(done.returncode, "commit with a bait in a crlf message")
        self.assertEqual(count(repo), before, "a commit was made")
        self.assert_refused_by_finding(output, "commit-msg", "bait in a crlf message")
        self.assertTrue(has_line(output, r"github-pat message:3 staged"), "the finding line names no rule")
        self.assert_no_leak(output, list(values.values()), [], "bait in a crlf message")


class P3PreCommit(HookCase):
    def test_p3_a_staged_bait_refuses_the_commit_and_an_unstaged_one_does_not(self):
        repo, _ = build_normal()
        values = baits()
        before = count(repo)
        write(repo, "leak.py", f'GITHUB_TOKEN = "{values["github_pat"]}"\n')
        done = stage_commit_hooked(repo)
        output = text_of(done)
        self.assert_refused(done.returncode, "commit with a staged bait")
        self.assertEqual(count(repo), before, "a commit was made")
        self.assert_refused_by_finding(output, "pre-commit", "staged bait")
        self.assertTrue(has_line(output, rf"github-pat file:{name_id('leak.py')}:1 staged"), "the finding line is missing")
        self.assert_no_leak(output, list(values.values()), ["leak.py"], "staged bait")
        git(repo, "reset", "-q")
        write(repo, "clean.txt", "clean\n")
        git(repo, "add", "clean.txt")
        self.assertEqual(commit_hooked(repo).returncode, 0, "an unstaged bait blocked the commit")
        self.assertEqual(count(repo), before + 1, "no commit was made")

    def test_p3_the_first_commit_of_a_clean_index_passes_through_both_hooks(self):
        repo = build_unborn()
        done = commit_hooked(repo, "first")
        output = text_of(done)
        self.assertEqual(done.returncode, 0, "the first clean commit was refused")
        self.assertEqual(count(repo), 1, "no commit was made")
        self.assertTrue(has_line(output, r"gitleaks_gate: pre-commit: clean"), "the pre-commit hook did not run")
        self.assertTrue(has_line(output, r"gitleaks_gate: commit-msg: clean"), "the commit-msg hook did not run")

    def test_p3_the_first_commit_with_a_bait_is_refused(self):
        values = baits()
        repo = build_unborn({"leak.py": f'GITHUB_TOKEN = "{values["github_pat"]}"\n'})
        done = commit_hooked(repo, "first")
        output = text_of(done)
        self.assert_refused(done.returncode, "first commit with a bait")
        self.assert_refused_by_finding(output, "pre-commit", "first commit with a bait")
        self.assertTrue(has_line(output, rf"github-pat file:{name_id('leak.py')}:1 staged"), "the finding line is missing")
        self.assertEqual(git(repo, "rev-parse", "--verify", "-q", "HEAD", check=False).returncode, 1, "a commit was made")
        self.assert_no_leak(output, list(values.values()), [], "first commit with a bait")


class P4PrePush(HookCase):
    def push(self, repo: Path, *args: str):
        return git(repo, "push", *args, env=env_for(), check=False)

    def remote_head(self, remote: Path) -> str:
        return git(remote, "rev-parse", "main").stdout.decode().strip()

    def test_p4_a_bait_in_the_content_of_a_new_commit_is_refused(self):
        repo, remote = build_history()
        base = head(repo)
        values = baits()
        write(repo, "leak.py", f'GITHUB_TOKEN = "{values["github_pat"]}"\n')
        leak = commit_plain(repo, "leak")
        activate(repo)
        write(repo, "clean.txt", "clean\n")
        git(repo, "add", "-A")
        self.assertEqual(commit_hooked(repo, "control").returncode, 0, "the clean control commit was refused")
        done = self.push(repo, "origin", "main")
        output = text_of(done)
        self.assert_refused(done.returncode, "push of a bait in content")
        self.assertTrue(self.remote_head(remote) == base, "the remote moved")
        self.assert_refused_by_finding(output, "pre-push", "bait in content")
        self.assertTrue(has_line(output, rf"github-pat file:{name_id('leak.py')}:1 {leak[:8]}"), "the finding line is missing")
        self.assert_no_leak(output, list(values.values()), [], "bait in content")

    def test_p4_a_bait_only_in_the_message_of_the_second_of_three_commits_names_that_commit(self):
        repo, remote = build_history()
        base = head(repo)
        values = baits()
        write(repo, "one.txt", "one\n")
        commit_plain(repo, b"first\r\nwith crlf lines\r\nand a lone cr\rinside\r\n")
        write(repo, "two.txt", "two\n")
        second = commit_plain(repo, "second\n\ntoken=" + values["commit_nachricht"] + "\n")
        write(repo, "three.txt", "three\n")
        commit_plain(repo, "third")
        activate(repo)
        done = self.push(repo, "origin", "main")
        output = text_of(done)
        self.assert_refused(done.returncode, "push of a bait in the second message")
        self.assertTrue(self.remote_head(remote) == base, "the remote moved")
        self.assert_refused_by_finding(output, "pre-push", "bait in the second message")
        # Only the lines of the gate are read for commits: git's own messages carry object ids and temporary
        # paths. Every line of the gate names the second commit, so neither the first nor the third.
        lines = [line for line in output.splitlines() if LINE_FORM.fullmatch(line)]
        self.assertTrue(any(re.fullmatch(r"github-pat message:[0-9]+ " + second[:8], line) for line in lines),
                        "the message line of the second commit is missing")
        self.assertTrue({line.split(" ")[2] for line in lines} == {second[:8]}, "a line names another commit")
        self.assert_no_leak(output, list(values.values()), [], "bait in the second message")

    def test_p4_a_bait_in_the_text_of_an_annotated_tag_is_refused(self):
        repo, remote = build_history()
        values = baits()
        tag_name = "t" + token(18)  # a short name like `v1` shows up in temporary paths by chance
        git(repo, "tag", "-a", tag_name, "-m", "note token=" + values["commit_nachricht"])
        activate(repo)
        done = self.push(repo, "origin", tag_name)
        output = text_of(done)
        self.assert_refused(done.returncode, "push of a bait in a tag text")
        self.assert_refused_by_finding(output, "pre-push", "bait in a tag text")
        self.assertTrue(has_line(output, r"github-pat tag:" + name_id(tag_name) + r":[0-9]+ -"), "the tag line is missing")
        self.assert_no_leak(output, list(values.values()), [tag_name], "bait in a tag text")
        self.assertEqual(git(remote, "tag", "-l").stdout, b"", "the remote has a tag")

    def test_p4_a_tag_pushed_by_its_object_id_is_checked(self):
        repo, remote = build_history()
        values = baits()
        source, target = "s" + token(18), "t" + token(18)
        git(repo, "tag", "-a", source, "-m", "note token=" + values["commit_nachricht"])
        tag_object = git(repo, "rev-parse", "refs/tags/" + source).stdout.decode().strip()
        git(repo, "tag", "-d", source)  # no local name is left: the push names only the object id
        activate(repo)
        done = self.push(repo, "origin", f"{tag_object}:refs/tags/{target}")
        output = text_of(done)
        self.assert_refused(done.returncode, "a tag pushed by its object id")
        self.assert_refused_by_finding(output, "pre-push", "tag pushed by object id")
        # Reported under the name the tag object carries: the one name of it this clone still knows.
        self.assertTrue(has_line(output, r"github-pat tag:" + name_id(source) + r":[0-9]+ -"), "the tag line is missing")
        self.assert_no_leak(output, list(values.values()), [source, target], "tag pushed by object id")
        self.assertEqual(git(remote, "tag", "-l").stdout, b"", "the remote has a tag")

    def resolves(self, repo: Path, identifier: str) -> bool:
        """True if the lookup gives exactly one name for `identifier` and that name has this identifier."""
        rc, resolved, _err = gate(repo, "lo" + "cate", identifier)
        names = resolved.splitlines()
        return rc == 0 and len(names) == 1 and name_id(names[0]) == identifier

    def test_p4_the_identifier_of_a_tag_pushed_by_its_object_id_resolves_locally(self):
        repo, _ = build_history()
        values = baits()
        source, target = "s" + token(18), "t" + token(18)
        git(repo, "tag", "-a", source, "-m", "note token=" + values["commit_nachricht"])
        tag_object = git(repo, "rev-parse", "refs/tags/" + source).stdout.decode().strip()
        git(repo, "tag", "-d", source)  # no ref names the tag object any more
        activate(repo)
        done = self.push(repo, "origin", f"{tag_object}:refs/tags/{target}")
        output = text_of(done)
        self.assert_refused(done.returncode, "a tag pushed by its object id")
        self.assert_refused_by_finding(output, "pre-push", "tag pushed by object id")
        reported = {found.group(1) for found in map(re.compile(r"github-pat tag:([0-9a-f]{12}):[0-9]+ -").fullmatch,
                                                     output.splitlines()) if found}
        self.assertEqual(len(reported), 1, "not exactly one tag identifier is reported with its rule")
        self.assertTrue(self.resolves(repo, reported.pop()), "the lookup does not resolve the reported tag identifier")

    def test_p4_the_files_of_a_commit_pushed_by_its_object_id_resolve_locally(self):
        repo, _ = build_history()
        name = "f" + token(12) + ".py"
        git(repo, "checkout", "-q", "-b", "side")
        write(repo, name, f'GITHUB_TOKEN = "{github_token()}"\n')
        leak = commit_plain(repo, "leak")
        git(repo, "checkout", "-q", "main")
        git(repo, "branch", "-q", "-D", "side")  # no ref reaches the commit any more
        activate(repo)
        done = self.push(repo, "origin", f"{leak}:refs/heads/side")
        output = text_of(done)
        self.assert_refused(done.returncode, "a commit pushed by its object id")
        self.assert_refused_by_finding(output, "pre-push", "a commit pushed by its object id")
        self.assertTrue(has_line(output, rf"github-pat file:{name_id(name)}:1 {leak[:8]}"), "the finding line is missing")
        self.assert_no_leak(output, [], [name], "a commit pushed by its object id")
        self.assertTrue(self.resolves(repo, name_id(name)), "the lookup does not resolve the reported file identifier")

    def test_p4_deleting_a_branch_and_pushing_what_the_remote_has_check_nothing(self):
        repo, remote = build_history()
        git(repo, "push", "-q", "origin", "main:topic")
        activate(repo)
        self.assertEqual(self.push(repo, "origin", ":topic").returncode, 0, "the deletion was refused")
        self.assertEqual(git(remote, "branch", "--list", "topic").stdout, b"", "the branch is still there")
        rc, out, err = gate(repo, "pre-push", "origin", "origin",
                            stdin=f"refs/heads/main {head(repo)} refs/heads/main {head(repo)}\n".encode())
        self.assertTrue(rc == 0 and out == "" and err.splitlines() == ["gitleaks_gate: pre-push: clean"],
                        "pushing what the remote has is not clean")

    def test_p4_a_push_to_a_url_checks_everything_from_the_local_id(self):
        repo, remote = build_history()
        other = WORK / ("remote_" + rnd(8, LOWER) + ".git")
        init_bare(other)
        values = baits()
        write(repo, "leak.py", f'GITHUB_TOKEN = "{values["github_pat"]}"\n')
        leak = commit_plain(repo, "leak")
        git(repo, "update-ref", "refs/remotes/origin/main", leak)  # as if the remote already had it; the hooks are still off
        activate(repo)
        write(repo, "clean.txt", "clean\n")
        git(repo, "add", "-A")
        self.assertEqual(commit_hooked(repo, "control").returncode, 0, "the clean control commit was refused")
        by_url = self.push(repo, str(other).replace("\\", "/"), "main")
        output = text_of(by_url)
        self.assert_refused(by_url.returncode, "push to a url")
        self.assert_refused_by_finding(output, "pre-push", "push to a url")
        self.assertTrue(has_line(output, rf"github-pat file:{name_id('leak.py')}:1 {leak[:8]}"), "the finding line is missing")
        by_name = self.push(repo, "origin", "main")  # the same push by name excludes what the remote is known to have
        self.assertEqual(by_name.returncode, 0, "the push by name was refused")

    def test_p4_every_form_of_a_faulty_push_message_is_refused(self):
        repo, _ = build_history()
        activate(repo)
        sha = head(repo)
        zero = "0" * 40
        ref = "refs/heads/M" + token(12)  # a mark that must never show
        good = f"{ref} {sha} refs/heads/main {zero}"
        forms = {
            "not utf8": f"{ref} \xff\n".encode("latin-1"), "cr": (good + "\r\n").encode(),
            "three fields": f"{ref} {sha} refs/heads/main\n".encode(), "five fields": (good + " x\n").encode(),
            "double space": f"{ref}  {sha} refs/heads/main {zero}\n".encode(),
            "upper case id": f"{ref} {sha.upper()} refs/heads/main {zero}\n".encode(),
            "short id": f"{ref} {sha[:39]} refs/heads/main {zero}\n".encode(),
            "upper case remote id": f"{ref} {sha} refs/heads/main {sha.upper()}\n".encode(),
            "short remote id": f"{ref} {sha} refs/heads/main {zero[:39]}\n".encode(),
            "control in ref": f"{ref}\x07 {sha} refs/heads/main {zero}\n".encode(),
            "control in remote ref": f"{ref} {sha} refs/heads/ma\x07in {zero}\n".encode(),
            "c1 control in ref": f"{ref}\u0085 {sha} refs/heads/main {zero}\n".encode(),
            "empty line before another": ("\n" + good + "\n").encode(), "duplicate": (good + "\n" + good + "\n").encode(),
            "no final lf": good.encode(), "two final lf": (good + "\n\n").encode(),
        }
        for label, data in forms.items():
            with self.case(form=label):
                rc, out, err = gate(repo, "pre-push", "origin", "origin", stdin=data)
                self.assertEqual(rc, 2, f"{label}: the return value is not 2 (error)")
                self.assertTrue(error_only(out, err, "pre-push", "push-input"),
                                f"{label}: the output is not exactly the fixed message of class push-input")
        rc, out, err = gate(repo, "pre-push", "origin", "origin", stdin=b"")
        self.assertTrue(rc == 0 and out == "" and err.splitlines() == ["gitleaks_gate: pre-push: clean"],
                        "no push line is not clean")

    def test_p4_remote_names_that_are_no_plain_name_never_exclude_and_never_show(self):
        repo, _ = build_history()
        write(repo, "leak.py", f'GITHUB_TOKEN = "{github_token()}"\n')
        leak = commit_plain(repo, "leak")
        git(repo, "update-ref", "refs/remotes/origin/main", leak)  # the remote is known to have the leak
        write(repo, "clean.txt", "clean\n")
        control = commit_plain(repo, "control")
        activate(repo)
        line = f"refs/heads/main {control} refs/heads/main {'0' * 40}\n".encode()
        rc, out, err = gate(repo, "pre-push", "origin", "x", stdin=line)
        self.assertTrue(rc == 0 and out == "" and err.splitlines() == ["gitleaks_gate: pre-push: clean"],
                        "precondition: by its plain name the remote excludes what it has")
        mark = "M" + token(12)
        for number, remote in enumerate(["origin " + mark, "tab\t" + mark, "--all", "a:" + mark, "origin/" + mark,
                                         "file:///" + mark]):
            with self.case(remote=number):
                rc, out, err = gate(repo, "pre-push", remote, "x", stdin=line)
                self.assertEqual(rc, 1, "the return value is not 1 (finding): the remote name excluded commits")
                self.assertTrue(f"github-pat file:{name_id('leak.py')}:1 {leak[:8]}" in out.splitlines(),
                                "the commit the remote name would exclude is not checked")
                self.assertTrue(is_finding(err, "pre-push"), "the refusal is not a finding of the gate")
                self.assertFalse(mark in out + err, "the remote name is in the output")

    def test_p4_the_push_is_refused_when_the_push_lines_cannot_be_stored(self):
        repo, remote = build_normal()
        write(repo, "clean.txt", "clean\n")
        git(repo, "add", "-A")
        self.assertEqual(commit_hooked(repo, "control").returncode, 0, "the clean control commit was refused")
        base = git(remote, "rev-parse", "main").stdout.decode().strip()
        line = f"refs/heads/main {head(repo)} refs/heads/main {base}\n".encode()
        tmp, other_tmp = WORK / ("tmp" + rnd(6, LOWER)), WORK / ("tmp" + rnd(6, LOWER))
        tmp.mkdir()
        other_tmp.mkdir()
        control = run_pre_push_directly(repo, remote, line, other_tmp, fake_cat=False)
        self.assertTrue(control.returncode == 0 and has_line(text_of(control), r"gitleaks_gate: pre-push: clean"),
                        "precondition: the hook run by the shell passes the same clean push")
        done = run_pre_push_directly(repo, remote, line, tmp, fake_cat=True)
        output = text_of(done)
        self.assert_refused(done.returncode, "the hook with push lines that could not be stored")
        self.assertTrue("pre-push: the push lines cannot be stored, push refused" in output.splitlines(),
                        "the fixed message of the failed store is missing")
        self.assertFalse(any(line.startswith("gitleaks_gate:") for line in output.splitlines()),
                         "the gate ran on push lines that were not stored")
        self.assertEqual(sum(1 for _ in tmp.iterdir()), 0, "a temporary file of the push lines is left over")


class P5ProgramMissing(HookCase):
    def test_p5_without_the_program_the_commit_is_refused_with_the_message_of_the_gate(self):
        repo, _ = build_normal()
        git(repo, "config", "--unset", "hooks.gitleaks")
        dirs = []
        for tool in ("git", "python", "python3", "uv", "sh", "mktemp"):
            found = shutil.which(tool)
            if found and os.path.dirname(found) not in dirs:
                dirs.append(os.path.dirname(found))
        self.assertTrue(all(shutil.which("gitleaks", path=folder) is None for folder in dirs),
                        "precondition: no gitleaks in the folders of the tools")
        write(repo, "clean.txt", "clean\n")
        git(repo, "add", "-A")
        before = count(repo)
        done = commit_hooked(repo, env=env_for({"PATH": os.pathsep.join(dirs)}))
        output = text_of(done)
        self.assert_refused(done.returncode, "commit without the program")
        self.assertEqual(count(repo), before, "a commit was made")
        self.assertTrue(has_line(output, r"gitleaks_gate: pre-commit: error binary-path"), "the error line of the gate is missing")
        self.assertFalse("public hygiene check failed" in output, "the hygiene check failed first")


class P16Merge(HookCase):
    def merge_with_bait(self, repo: Path, value: str) -> None:
        write(repo, "a.txt", "one\n")
        commit_plain(repo, "base two")
        git(repo, "checkout", "-q", "-b", "side")
        write(repo, "b.txt", "side\n")
        commit_plain(repo, "side")
        git(repo, "checkout", "-q", "main")
        write(repo, "c.txt", "main\n")
        commit_plain(repo, "main")
        git(repo, "merge", "-q", "--no-ff", "--no-commit", "side")
        write(repo, "evil.txt", f'GITHUB_TOKEN = "{value}"\n')

    def test_p16_pre_commit_refuses_the_conclusion_of_a_merge_with_a_bait(self):
        repo, _ = build_history()
        value = github_token()
        self.merge_with_bait(repo, value)
        activate(repo)
        git(repo, "add", "-A")
        before = count(repo)
        done = commit_hooked(repo, "merge")
        output = text_of(done)
        self.assert_refused(done.returncode, "merge commit with a bait")
        self.assertEqual(count(repo), before, "a commit was made")
        self.assert_refused_by_finding(output, "pre-commit", "merge with a bait")
        self.assertTrue(has_line(output, rf"github-pat file:{name_id('evil.txt')}:1 staged"), "the finding line is missing")

    def test_p16_pre_push_refuses_a_merge_that_carries_a_bait(self):
        repo, remote = build_history()
        base = git(remote, "rev-parse", "main").stdout.decode().strip()
        value = github_token()
        self.merge_with_bait(repo, value)
        merge = commit_plain(repo, "merge")
        activate(repo)
        done = git(repo, "push", "origin", "main", env=env_for(), check=False)
        output = text_of(done)
        self.assert_refused(done.returncode, "push of a merge with a bait")
        self.assertTrue(git(remote, "rev-parse", "main").stdout.decode().strip() == base, "the remote moved")
        self.assert_refused_by_finding(output, "pre-push", "pushed merge with a bait")
        self.assertTrue(has_line(output, rf"github-pat file:{name_id('evil.txt')}:1 {merge[:8]}"), "the finding line is missing")
        self.assert_no_leak(output, [value], [], "pushed merge with a bait")


class P18Names(HookCase):
    @staticmethod
    def build_names(repo: Path):
        values = baits()
        random_name = token(20)
        at_name = "at" + "@" + token(6)
        tag_name = "t" + token(18)
        files = {random_name: values["passwort_quotiert"], at_name: values["phrase"], "harmless-notes.txt": values["app_passwort"]}
        for name, value in files.items():
            write(repo, name, "pass" + "word" + ' = "' + value + '"\n')
        return values, files, tag_name

    def test_p18a_the_hook_names_no_file_and_no_bait(self):
        repo, _ = build_normal()
        values, files, _tag = self.build_names(repo)
        before = count(repo)
        done = stage_commit_hooked(repo)
        output = text_of(done)
        self.assert_refused(done.returncode, "commit of names")
        self.assertEqual(count(repo), before, "a commit was made")
        self.assert_refused_by_finding(output, "pre-commit", "names")
        self.assert_no_leak(output, list(values.values()), list(files), "names")
        shown = [line for line in output.splitlines() if " staged" in line and not line.startswith("gitleaks_gate")]
        self.assertEqual(len(shown), 3, "not three finding lines")
        self.assertEqual(sum(1 for line in shown if not LINE_FORM.fullmatch(line)), 0, "a finding line has another form")

    def test_p18a_a_pushed_tag_names_only_an_identifier(self):
        repo, _ = build_history()
        values = baits()
        tag_name = "t" + token(18)
        git(repo, "tag", "-a", tag_name, "-m", "note token=" + values["commit_nachricht"])
        activate(repo)
        done = git(repo, "push", "origin", tag_name, env=env_for(), check=False)
        output = text_of(done)
        self.assert_refused(done.returncode, "push of a tag")
        self.assert_refused_by_finding(output, "pre-push", "pushed tag")
        self.assertTrue(has_line(output, r"github-pat tag:" + name_id(tag_name) + r":[0-9]+ -"), "the tag line is missing")
        self.assert_no_leak(output, list(values.values()), [tag_name], "pushed tag")

    def test_p18b_the_lookup_resolves_every_reported_identifier_and_refuses_bad_forms(self):
        repo = build_unborn()  # a small history of its own: only the synthetic files and the names below
        git(repo, "config", "--unset", "core.hooksPath")
        values, files, tag_name = self.build_names(repo)
        commit_plain(repo, "files")
        git(repo, "tag", "-a", tag_name, "-m", "note token=" + values["commit_nachricht"])
        add_raw_name(repo, RAW_NAME, "pass" + "word" + ' = "' + with_marks(token(20)) + '"\n')
        git(repo, "commit", "-q", "-m", "raw name")  # from the index; the name never touches the file system
        rc, out, _err = gate(repo, "full")
        self.assertEqual(rc, 1, "the return value is not 1 (finding)")
        found = [line.split(" ")[1] for line in out.splitlines() if " " in line]
        identifiers = sorted({loc.split(":")[1] for loc in found if loc.startswith(("file:", "tag:"))})
        self.assertEqual(len(identifiers), 5, "not five reported identifiers")
        self.assertTrue(name_id(RAW_NAME) in identifiers, "the identifier of the raw name is not built from its bytes")
        for number, identifier in enumerate(identifiers):
            with self.case(identifier=number):
                rc, resolved, _ = gate(repo, "lo" + "cate", identifier)
                names = resolved.splitlines()
                self.assertTrue(rc == 0 and len(names) == 1, "the lookup does not give exactly one name")
                if identifier == name_id(RAW_NAME):
                    self.assertTrue(names[0] == RAW_NAME.decode("utf-8", "backslashreplace"),
                                    "the lookup does not show the raw name")
                else:
                    self.assertTrue(name_id(names[0]) == identifier, "the name of the lookup has another identifier")
        for bad in ([], ["abc"], ["ABCDEF123456"], ["0123456789abc"], ["0123456789a"], ["zzzzzzzzzzzz"],
                    ["0123456789ab", "0123456789ab"]):
            with self.case(form=len(bad[0]) if bad else 0, arguments=len(bad)):
                rc, out, err = gate(repo, "lo" + "cate", *bad)
                self.assertEqual(rc, 2, "the return value is not 2 (error)")
                self.assertTrue(error_only(out, err, "lo" + "cate", "usage"), "the output is not exactly the usage error")

    def assert_resolved(self, result, identifier: str, what: str):
        rc, out, err = result
        names = out.splitlines()
        self.assertTrue(rc == 0 and len(names) == 1 and name_id(names[0]) == identifier,
                        f"{what}: the lookup does not give exactly the one name")

    def assert_lookup_error(self, result, what: str):
        rc, out, err = result
        self.assertEqual(rc, 2, f"{what}: the return value is not 2 (error)")
        self.assertTrue(error_only(out, err, "lo" + "cate", "git-error"), f"{what}: the output is not exactly the git error")

    def test_p18b_in_a_clone_without_a_commit_the_lookup_reads_no_history(self):
        repo = build_unborn()
        result, calls = lookup_in_process(repo, name_id("notes.txt"), ("log",))
        self.assertEqual(len(calls), 0, "the history is read although the clone has no commit")
        self.assert_resolved(result, name_id("notes.txt"), "a staged name in a clone without a commit")

    def test_p18b_a_failed_history_call_is_an_error_even_for_a_name_in_the_index(self):
        repo = build_unborn()
        git(repo, "config", "--unset", "core.hooksPath")
        commit_plain(repo, "first")
        self.assert_resolved(gate(repo, "lo" + "cate", name_id("notes.txt")), name_id("notes.txt"),
                             "precondition: the name in the index")
        # A commit whose tree is missing: git accepts the object, its history call fails with partial output.
        broken = ("tree " + "1" * 40 + "\nauthor probe <probe" + "@" + "example.invalid> 1700000000 +0000\ncommitter probe"
                  " <probe" + "@" + "example.invalid> 1700000000 +0000\n\nbroken\n").encode()
        git(repo, "hash-object", "-t", "commit", "-w", "--stdin", data=broken)
        self.assert_lookup_error(gate(repo, "lo" + "cate", name_id("notes.txt")), "a failed history call")

    def test_p18b_a_faulty_inventory_line_is_an_error_even_for_a_name_in_the_index(self):
        repo = build_unborn()
        git(repo, "config", "--unset", "core.hooksPath")
        commit_plain(repo, "first")
        words = ("cat-file", "--batch-all-objects")
        forms = {"unknown type": b"bogus " + b"a" * 40 + b"\n", "three fields": b"blob " + b"a" * 40 + b" x\n",
                 "id not hex": b"tree " + b"z" * 40 + b"\n"}
        result, calls = lookup_in_process(repo, name_id("notes.txt"), words)
        self.assertEqual(calls, [0], "precondition: one sound inventory call")
        self.assert_resolved(result, name_id("notes.txt"), "precondition: the name in the index")
        for label, extra in forms.items():
            with self.case(form=label):
                result, calls = lookup_in_process(repo, name_id("notes.txt"), words, lambda rc, out: (rc, out + extra))
                self.assertEqual(calls, [0], f"precondition: {label}: one sound inventory call before the change")
                self.assert_lookup_error(result, label)


if __name__ == "__main__":
    unittest.main()
