"""Tests of the secret scan gate (`scripts/gitleaks_gate.py`), without the git hooks.

Runs locally and in CI; needs only Python, git and the pinned gitleaks program. Tests that use the real
program need `GITLEAKS_TEST_BINARY`; without it they are skipped with a reason, unless
`GITLEAKS_REQUIRED=1` is set, then they fail. Every bait is made at run time from pieces and random
characters; no source line holds a key prefix, a PEM header or an address.

Failure messages are public in CI logs. Every check therefore reduces what it reads to a truth value or a
count first and fails with a fixed text; no assertion compares an output, a name or a path, and an
unexpected exception in a test step is replaced by its type name and line number (`SafeCase`). Every run
that loads this file as a module (discovery, as in CI, or `unittest.main`) reports a failure, an error of a
test, of a subtest or of the module setup only as its test id, its fixed text and line numbers of this file:
no traceback, so no file path (`load_tests`).
"""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import importlib.util
import io
import json
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
SOURCE = Path(os.environ.get("GITLEAKS_GATE_SOURCE") or ROOT)
BINARY = os.environ.get("GITLEAKS_TEST_BINARY", "")
REQUIRED = os.environ.get("GITLEAKS_REQUIRED") == "1"
GATE_FILES = (
    "scripts/gitleaks_gate.py", "scripts/gitleaks_findings.tmpl", ".gitleaks.toml", ".gitleaks-version",
    ".gitleaksignore",
)
R = random.SystemRandom()
B62 = string.ascii_letters + string.digits
WORK: Path | None = None  # the temporary folder of a run, made by `setUpModule`
EMPTY_CONFIG: Path | None = None
LINE_FORM = re.compile(r"[A-Za-z0-9._-]+ (?:file:[0-9a-f]{12}:[0-9]+|tag:[0-9a-f]{12}:[0-9]+|message:[0-9]+) "
                       r"(?:[0-9a-f]{8}|-|staged)")
REMEDY_LINE = ("gitleaks_gate: install the gitleaks release named in .gitleaks-version and point the repository "
               "at it: git config hooks.gitleaks <absolute path to the program>")


class FixedError(RuntimeError):
    """A failure of the test setup with a fixed text: never a path, a value or an output."""


def setUpModule():
    global WORK, EMPTY_CONFIG
    try:
        WORK = Path(tempfile.mkdtemp(prefix="glgate_"))
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

    def assert_error(self, result, label: str, kind: str, what: str):
        rc, out, err = result
        self.assertEqual(rc, 2, f"{what}: the return value is not 2 (error)")
        self.assertTrue(error_only(out, err, label, kind),
                        f"{what}: the output is not exactly the fixed message of class {kind}")

    def assert_clean(self, result, label: str, what: str):
        rc, out, err = result
        self.assertEqual(rc, 0, f"{what}: the return value is not 0 (clean)")
        self.assertTrue(out == "" and err.splitlines() == [f"gitleaks_gate: {label}: clean"],
                        f"{what}: the output is not exactly the clean message")

    def assert_findings(self, result, label: str, expected: set, what: str):
        """A real finding: return value 1, exactly the expected lines, the fixed summary and nothing else."""
        rc, out, err = result
        self.assertEqual(rc, 1, f"{what}: the return value is not 1 (finding)")
        lines = parse_lines(out)
        self.assertTrue(len(lines) == len(expected) and set(lines) == expected,
                        f"{what}: the finding lines are not exactly the expected ones")
        self.assertTrue(err.splitlines() == [f"gitleaks_gate: {label}: {len(expected)} finding(s), refused"],
                        f"{what}: stderr is not exactly the summary of the findings")

    def assert_found(self, result, label: str, required: set, what: str):
        """A real finding that contains at least the required lines; every line has the fixed form."""
        rc, out, err = result
        self.assertEqual(rc, 1, f"{what}: the return value is not 1 (finding)")
        lines = parse_lines(out)
        self.assertTrue(required <= set(lines), f"{what}: a required finding line is missing")
        self.assertTrue(all(LINE_FORM.fullmatch(line) for line in out.splitlines()),
                        f"{what}: a line of stdout is not of the finding form")
        self.assertTrue(err.splitlines() == [f"gitleaks_gate: {label}: {len(lines)} finding(s), refused"],
                        f"{what}: stderr is not exactly the summary of the findings")


def error_only(out: str, err: str, label: str, kind: str) -> bool:
    """True if stdout is empty and stderr is exactly the fixed error line (and the fixed remedy line)."""
    lines = err.splitlines()
    if out != "" or not lines:
        return False
    if not re.fullmatch(re.escape(f"gitleaks_gate: {label}: error {kind}") + r"(?: rc=-?[0-9]+)?", lines[0]):
        return False
    return lines[1:] == ([REMEDY_LINE] if kind in ("binary-path", "binary-version") else [])


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def rnd(n: int, alphabet: str = B62) -> str:
    return "".join(R.choice(alphabet) for _ in range(n))


UPPER = string.ascii_uppercase
NO_L = UPPER.replace("L", "")
RARE = "QXZJ"


def spread(count: int, alphabet: str) -> list[str]:
    """`count` symbols of `alphabet` in random order, each one floor(count/len) or ceil(count/len) times."""
    full, rest = divmod(count, len(alphabet))
    items = list(alphabet) * full + R.sample(alphabet, rest)
    R.shuffle(items)
    return items


def token(n: int, letters: str = UPPER) -> str:
    """A random value of n characters that alternates letter and digit, starting with a letter.

    The form makes every bait meet its rule on every run, not on most runs: no two letters and no two digits
    stand side by side, so no stop word of the default rules and no word of their allow lists fits in (each
    has two adjacent letters or digits); there is always a digit and the first character is a letter, as
    `generic-api-key` needs; the symbol counts are fixed, so the entropy is fixed too. Upper-case letters
    keep every 8-character piece out of lower-case output (object ids, temporary names), so a check for
    leaked pieces cannot fail by chance.
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


def git_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    for key in ("GITLEAKS_CONFIG", "GITLEAKS_CONFIG_TOML", "GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        env.pop(key, None)
    env.update(GIT_CONFIG_GLOBAL=str(EMPTY_CONFIG), GIT_CONFIG_NOSYSTEM="1", GIT_ALLOW_PROTOCOL="file",
               GIT_AUTHOR_NAME="probe", GIT_COMMITTER_NAME="probe",
               GIT_AUTHOR_EMAIL="probe" + "@" + "example.invalid", GIT_COMMITTER_EMAIL="probe" + "@" + "example.invalid")
    env.update(extra or {})
    return env


def run_ok(command: list[str], what: str, env: dict | None = None):
    done = subprocess.run(command, capture_output=True, env=env or git_env())
    if done.returncode != 0:
        raise AssertionError(f"setup step failed: {what} (rc {done.returncode})")
    return done


def git(repo: Path, *args: str, data: bytes | None = None, env: dict | None = None, check: bool = True):
    done = subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                          input=data, capture_output=True, env=env or git_env())
    if check and done.returncode != 0:
        raise AssertionError(f"git setup step failed ({args[0]}, rc {done.returncode})")
    return done


def new_repo(name: str | None = None) -> Path:
    repo = WORK / (name or "r" + rnd(8, string.ascii_lowercase))
    repo.mkdir(parents=True)
    run_ok(["git", "init", "-q", "-b", "main", str(repo)], "git init")
    for rel in GATE_FILES:
        source = SOURCE / rel
        if source.exists():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, repo / rel)
    return repo


def write(repo: Path, rel: str, text: str | bytes) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)


def commit(repo: Path, message: str | bytes = "work", add: str = "-A") -> str:
    git(repo, "add", add)
    data = message.encode("utf-8") if isinstance(message, str) else message
    git(repo, "commit", "-q", "-F", "-", data=data)
    return rev(repo, "HEAD")


def rev(repo: Path, name: str) -> str:
    return git(repo, "rev-parse", name).stdout.decode().strip()


def run_gate(repo: Path, *args: str, stdin: bytes = b"", env: dict | None = None, script: Path | None = None):
    done = subprocess.run([sys.executable, str(script or repo / "scripts" / "gitleaks_gate.py"), *args], input=stdin,
                          capture_output=True, env=env or git_env(), cwd=str(repo))
    return done.returncode, done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace")


def load_gate(repo: Path):
    """The gate of a test repository as a module, for runs of its `main` inside this process."""
    spec = importlib.util.spec_from_file_location("gate_" + rnd(8, string.ascii_lowercase),
                                                  repo / "scripts" / "gitleaks_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_main(module, *args: str, stdin: bytes = b""):
    """`main` of a loaded gate with the isolated environment, the given stdin and captured output."""
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, git_env(), clear=True), \
            mock.patch.object(sys, "stdin", io.TextIOWrapper(io.BytesIO(stdin))), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = module.main(list(args))
    return rc, out.getvalue(), err.getvalue()


def name_id(name: str | bytes) -> str:
    return hashlib.sha256(name.encode("utf-8") if isinstance(name, str) else name).hexdigest()[:12]


def windows_of(value: str) -> set[str]:
    out = set()
    for part in value.split("\n"):
        out |= {part[i:i + 8] for i in range(len(part) - 7)}
    return out


def leaked(output: str, values: list[str]) -> int:
    """How many 8-character pieces of the given values stand in the output."""
    return sum(1 for value in values for piece in windows_of(value) if piece in output)


def baits() -> dict[str, str]:
    # Every form meets its rule on every run (see `token`). Forms without digits use letters that cannot
    # spell a word of the global allow list; the key id draws 16 distinct symbols, which keeps its entropy
    # above the rule's floor.
    return {
        "github_pat": "gh" + "p_" + token(36),
        "typisiert": token(32),
        "passwort_quotiert": with_marks(token(20)),
        "kopfzeile": token(32),
        "ohne_stichwort": token(20),
        "anthropic": "sk-" + "ant-" + "api03-" + token(93) + "AA",
        "huggingface": "hf" + "_" + "".join(spread(34, NO_L)),
        "kurzes_passwort": "Sommer" + rnd(2, string.digits) + "!" + rnd(2, string.ascii_lowercase),
        "typ_mit_stichwort": token(24),
        "aws_kennung": "AK" + "IA" + "".join(R.sample(NO_L + "234567", 16)),
        "aws_geheim": token(40),
        "phrase": " ".join(rnd(5, RARE) for _ in range(4)),
        "openrouter": "sk-" + "or-" + "v1-" + token(64, "ABCDEF"),
        "app_passwort": " ".join(rnd(4, RARE) for _ in range(4)),
        "commit_nachricht": token(32),
        "pem": "\n".join(token(64) for _ in range(12)),
    }


def github_line() -> str:
    return 'GITHUB_TOKEN = "' + "gh" + "p_" + token(36) + '"\n'


EXPECTED_RULE = {
    "github_pat": "github-pat", "typisiert": "house-typed-assignment", "passwort_quotiert": "house-password-quoted",
    "kopfzeile": "generic-api-key", "anthropic": "anthropic-api-key", "huggingface": "huggingface-access-token",
    "kurzes_passwort": "house-password-bare", "typ_mit_stichwort": "generic-api-key",
    "aws_kennung": "aws-access-token", "aws_geheim": "generic-api-key", "phrase": "house-password-quoted",
    "openrouter": "generic-api-key", "app_passwort": "house-password-quoted", "pem": "private-key",
    "commit_nachricht": "generic-api-key",
}
BAIT_FILES = {
    "config.py": [
        ("github_pat", 'GITHUB_TOKEN = "{w}"'), ("typisiert", 'api_key: str = "{w}"'),
        ("passwort_quotiert", 'password = "{w}"'), ("kopfzeile", 'headers = {{"X-API-Key": "{w}"}}'),
        ("ohne_stichwort", 'client.login("admin", "{w}")'), ("anthropic", 'ANTHROPIC = "{w}"'),
        ("huggingface", 'HF = "{w}"'),
    ],
    "deploy.env": [
        ("kurzes_passwort", "SMTP_PASSWORD={w}"), ("typ_mit_stichwort", "token: Token = {w}"),
        ("aws_kennung", "AWS_ACCESS_KEY_ID={w}"), ("aws_geheim", "AWS_SECRET_ACCESS_KEY={w}"),
        ("phrase", 'DB_PASSWORD="{w}"'), ("openrouter", "OPENROUTER_API_KEY={w}"),
        ("app_passwort", 'MAIL_APP_PASSWORD="{w}"'),
    ],
}
PEM_HEAD = "-----BEGIN " + "RSA PRIVATE" + " KEY-----"
PEM_FOOT = "-----END " + "RSA PRIVATE" + " KEY-----"


def write_baits(repo: Path, values: dict[str, str]) -> dict[str, tuple[str, int | None]]:
    places: dict[str, tuple[str, int | None]] = {}
    for name, lines in BAIT_FILES.items():
        text = ""
        for number, (key, template) in enumerate(lines, 1):
            text += template.format(w=values[key]) + "\n"
            places[key] = (name, number)
        write(repo, name, text)
    write(repo, "id_rsa.pem", PEM_HEAD + "\n" + values["pem"] + "\n" + PEM_FOOT + "\n")
    places["pem"] = ("id_rsa.pem", None)
    return places


def parse_lines(out: str) -> list[tuple[str, ...]]:
    return [tuple(line.split(" ")) for line in out.splitlines() if line.strip()]


# Fake programs: small scripts with a fixed version, output and return value, for forms the real program
# cannot or must not produce. Each call is noted in `calls.txt`; what a `stdin` scan receives is kept in
# `stdin.bin`.
FAKE_SOURCE = '''import base64, json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
spec = json.load(open(os.path.join(here, "spec.json")))
with open(os.path.join(here, "env.txt"), "w") as note:
    note.write("%d %d" % ("GITLEAKS_CONFIG" in os.environ, "GITLEAKS_CONFIG_TOML" in os.environ))
with open(os.path.join(here, "calls.txt"), "a") as calls:
    calls.write((sys.argv[1] if len(sys.argv) > 1 else "") + "\\n")
if sys.argv[1:2] == ["version"]:
    sys.stdout.buffer.write(base64.b64decode(spec["version_out"]))
    sys.exit(spec["version_rc"])
if spec["sleep"]:
    time.sleep(spec["sleep"])
if sys.argv[1:2] == ["stdin"]:
    data = sys.stdin.buffer.read()
    with open(os.path.join(here, "stdin.bin"), "wb") as kept:
        kept.write(data)
sys.stderr.buffer.write(base64.b64decode(spec["scan_err"]))
sys.stdout.buffer.write(base64.b64decode(spec["scan_out"]))
sys.exit(spec["scan_rc"])
'''
VERSION_BRANCH = 'if sys.argv[1:2] == ["version"]:'


def make_fake(folder: Path, version_out: bytes = b"8.30.1\n", version_rc: int = 0, scan_out: bytes = b"",
              scan_rc: int = 0, scan_err: bytes = b"", sleep: int = 0) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    spec = {"version_out": base64.b64encode(version_out).decode(), "version_rc": version_rc,
            "scan_out": base64.b64encode(scan_out).decode(), "scan_rc": scan_rc,
            "scan_err": base64.b64encode(scan_err).decode(), "sleep": sleep}
    (folder / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (folder / "fake_impl.py").write_text(FAKE_SOURCE, encoding="utf-8")
    if os.name == "nt":
        wrapper = folder / "fake.cmd"
        wrapper.write_bytes(f'@"{sys.executable}" "{folder / "fake_impl.py"}" %*\r\n'.encode("utf-8"))
    else:
        wrapper = folder / "fake"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{folder / "fake_impl.py"}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
    return str(wrapper)


def fake_calls(fake: str) -> list[str]:
    path = Path(fake).parent / "calls.txt"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def report_line(rule="github-pat", file="a.txt", line=1, commit="a" * 40) -> bytes:
    return (json.dumps({"rule": rule, "file": file, "line": line, "commit": commit}) + "\n").encode("utf-8")


def real_binary_present() -> bool:
    return bool(BINARY) and os.path.isfile(BINARY)


def add_raw_name(repo: Path, raw: bytes, text: str) -> None:
    """Puts a file with a raw byte name into the index, without the file system (works on every platform)."""
    blob = git(repo, "hash-object", "-w", "--stdin", data=text.encode("utf-8")).stdout.decode().strip()
    git(repo, "update-index", "--add", "--index-info", data=b"100644 " + blob.encode("ascii") + b"\t" + raw + b"\n")


class NeedsBinary(SafeCase):
    def setUp(self):
        if not real_binary_present():
            if REQUIRED:
                self.fail("GITLEAKS_TEST_BINARY is not set to an existing file")
            self.skipTest("GITLEAKS_TEST_BINARY is not set")


class BinaryCase(NeedsBinary):
    def gate(self, repo: Path, *args: str, **kw):
        git(repo, "config", "hooks.gitleaks", BINARY)
        return run_gate(repo, *args, **kw)


# ---------------------------------------------------------------------------------------------
# P1
# ---------------------------------------------------------------------------------------------

class P1BaitMeasurement(BinaryCase):
    def measure(self) -> dict:
        repo = new_repo()
        values = baits()
        places = write_baits(repo, values)
        subject = "msg" + token(10)
        commit(repo, subject + "\n\ntoken=" + values["commit_nachricht"] + "\n")
        git(repo, "config", "hooks.gitleaks", BINARY)
        rc, out, err = run_gate(repo, "full")
        found = parse_lines(out)
        missing = []
        for key, rule in EXPECTED_RULE.items():
            if key == "commit_nachricht":
                hit = any(loc.startswith("message:") and r == rule for r, loc, _ in found)
            elif places[key][1] is None:
                hit = any(loc.startswith(f"file:{name_id(places[key][0])}:") and r == rule for r, loc, _ in found)
            else:
                want = f"file:{name_id(places[key][0])}:{places[key][1]}"
                hit = any(loc == want and r == rule for r, loc, _ in found)
            if not hit:
                missing.append(key)
        gap = places["ohne_stichwort"]
        return {"rc": rc, "missing": missing, "out": out, "err": err, "values": values, "subject": subject,
                "gap_hit": any(loc == f"file:{name_id(gap[0])}:{gap[1]}" for _, loc, _ in found)}

    def test_p1_fifteen_of_sixteen_baits_found_and_nothing_leaks(self):
        # One run, no second try: the bait forms meet their rules on every run (see `token` and `baits`).
        result = self.measure()
        self.assertEqual(result["rc"], 1, "the return value is not 1 (finding)")
        self.assertEqual(len(result["missing"]), 0, f"{len(result['missing'])} bait(s) not reported with their rule")
        self.assertFalse(result["gap_hit"], "the known gap (a value without a keyword) is closed: update Z1 and this test")
        everything = result["out"] + result["err"]
        self.assertEqual(leaked(everything, list(result["values"].values())), 0, "a piece of a bait is in the output")
        self.assertFalse(("probe" + "@" + "example.invalid") in everything, "the author address is in the output")
        self.assertFalse(result["subject"] in everything, "the commit subject is in the output")
        self.assertFalse("config.py" in everything, "a file name is in the output")
        self.assertEqual(sum(1 for line in result["out"].splitlines() if not LINE_FORM.fullmatch(line)), 0,
                         "a line of stdout is not of the finding form")


# ---------------------------------------------------------------------------------------------
# P5: arguments, version file, program, files
# ---------------------------------------------------------------------------------------------

class P5Usage(SafeCase):
    FORMS = [
        ([], "-"), (["bogus"], "-"), (["pre-commit", "extra"], "pre-commit"), (["commit-msg"], "commit-msg"),
        (["commit-msg", "a", "b"], "commit-msg"), (["commit-msg", "--binary"], "commit-msg"),
        (["commit-msg", "--binary", "x"], "commit-msg"), (["pre-push"], "pre-push"), (["pre-push", "a"], "pre-push"),
        (["pre-push", "a", "b", "c"], "pre-push"), (["pre-push", "--binary", "x"], "pre-push"),
        (["pre-push", "a", "--binary"], "pre-push"), (["full", "extra"], "full"), (["full", "--binary"], "full"),
        (["full", "--binary", "a", "b"], "full"), (["pre-commit", "--binary", "x"], "pre-commit"),
        (["full", "--other", "x"], "full"),
    ]

    def test_p5_every_form_of_the_arguments_is_refused(self):
        repo = new_repo()
        commit(repo)
        for number, (form, label) in enumerate(self.FORMS):
            with self.case(form=number):
                self.assert_error(run_gate(repo, *form), label, "usage", "argument form")

    def test_p5_the_arguments_are_checked_before_any_preparation(self):
        # Without the version file every preparation fails; the class must still be `usage`.
        repo = new_repo()
        commit(repo)
        os.remove(repo / ".gitleaks-version")
        for number, (form, label) in enumerate(self.FORMS):
            with self.case(form=number):
                self.assert_error(run_gate(repo, *form), label, "usage", "argument form without a version file")

    def test_p5_an_unknown_mode_is_never_named(self):
        repo = new_repo()
        commit(repo)
        for number, mode in enumerate(["MODE" + token(12), "x\t" + token(12) + "/a\\b", str(WORK / ("P" + token(12)))]):
            with self.case(form=number):
                self.assert_error(run_gate(repo, mode, "x"), "-", "usage", "unknown mode")


class P5VersionFile(SafeCase):
    def test_p5_every_form_of_the_version_file_is_refused(self):
        forms = {
            "empty": b"", "bom": b"\xef\xbb\xbf8.30.1\n", "cr": b"8.30.1\r\n", "two lines": b"8.30.1\n8.30.1\n",
            "space after": b"8.30.1 \n", "space before": b" 8.30.1\n", "short": b"8.30\n", "prefix": b"v8.30.1\n",
            "not utf8": b"\xff\xfe\n", "too long": b"8.30." + b"1" * 70 + b"\n", "no lf": b"8.30.1",
            "two lf": b"8.30.1\n\n", "letters": b"8.30.x\n",
        }
        fake = make_fake(WORK / ("fv" + rnd(6, string.ascii_lowercase)))
        for label, content in forms.items():
            with self.case(form=label):
                repo = new_repo()
                commit(repo)
                write(repo, ".gitleaks-version", content)
                self.assert_error(run_gate(repo, "full", "--binary", fake), "full", "version-file", label)
        for label in ("missing", "folder"):
            with self.case(form=label):
                repo = new_repo()
                commit(repo)
                os.remove(repo / ".gitleaks-version")
                if label == "folder":
                    (repo / ".gitleaks-version").mkdir()
                self.assert_error(run_gate(repo, "full", "--binary", fake), "full", "version-file", label)
        self.assertEqual(fake_calls(fake), [], "the program was called although the version file is refused")


class P5BinaryPath(SafeCase):
    def setUp(self):
        self.repo = new_repo()
        commit(self.repo)
        self.fake = make_fake(WORK / ("fk" + rnd(6, string.ascii_lowercase)))

    def refused(self, *args, env=None, kind="binary-path", what="form"):
        self.assert_error(run_gate(self.repo, *args, env=env), args[0], kind, what)

    def test_p5_forms_of_the_binary_option(self):
        folder = WORK / "afolder"
        folder.mkdir(exist_ok=True)
        forms = {
            "empty": "", "control": "a\tb", "carriage return": "a\rb", "relative": "gitleaks", "folder": str(folder),
            "missing": str(WORK / ("P" + token(12))), "valid plus lf": self.fake + "\n",
            "valid plus cr": self.fake + "\r", "valid plus tab": self.fake + "\t",
        }
        for label, value in forms.items():
            with self.case(form=label):
                self.refused("full", "--binary", value, what=label)
        self.assertEqual(fake_calls(self.fake), [], "the program was called for a refused path")

    def test_p5_forms_of_the_configuration_key(self):
        config = self.repo / ".git" / "config"
        base = config.read_bytes()
        posix = Path(self.fake).as_posix().encode("utf-8")
        forms = {
            "empty": b'[hooks]\n\tgitleaks = ""\n',
            "twice": 2 * (b'[hooks]\n\tgitleaks = "' + posix + b'"\n'),
            "extra empty value": b'[hooks]\n\tgitleaks = "' + posix + b'"\n\tgitleaks = ""\n',
            "tab": b'[hooks]\n\tgitleaks = "a\\tb"\n', "carriage return": b'[hooks]\n\tgitleaks = "a\rb"\n',
            "not utf8": b'[hooks]\n\tgitleaks = "\xff\xfe"\n', "relative": b"[hooks]\n\tgitleaks = gitleaks\n",
            "missing": b'[hooks]\n\tgitleaks = "' + (WORK / ("P" + token(12))).as_posix().encode() + b'"\n',
            "folder": b'[hooks]\n\tgitleaks = "' + WORK.as_posix().encode() + b'"\n',
            "valid plus lf": b'[hooks]\n\tgitleaks = "' + posix + b'\\n"\n',
            "valid plus cr": b'[hooks]\n\tgitleaks = "' + posix + b'\r"\n',
            "valid plus tab": b'[hooks]\n\tgitleaks = "' + posix + b'\\t"\n',
        }
        for label, section in forms.items():
            with self.case(form=label):
                config.write_bytes(base + section)
                self.refused("full", what=label)
        self.assertEqual(fake_calls(self.fake), [], "the program was called for a refused key")
        config.write_bytes(base + b'[hooks]\n\tgitleaks = "' + posix + b'"\n')
        self.assert_clean(run_gate(self.repo, "full"), "full", "the same key with a clean value")
        config.write_bytes(base)

    @staticmethod
    def tool_dirs(*tools: str) -> list[str]:
        dirs = []
        for tool in tools:
            found = shutil.which(tool)
            if found and os.path.dirname(found) not in dirs:
                dirs.append(os.path.dirname(found))
        return dirs

    def test_p5_program_missing_from_path(self):
        dirs = self.tool_dirs("git", "python", "python3", "uv")
        self.assertTrue(all(shutil.which("gitleaks", path=folder) is None for folder in dirs),
                        "precondition: no gitleaks in the folders of the tools")
        self.refused("pre-commit", env=git_env({"PATH": os.pathsep.join(dirs)}), what="program missing from PATH")

    def test_p5_the_program_is_found_over_path(self):
        folder = Path(self.fake).parent
        named = folder / ("gitleaks.cmd" if os.name == "nt" else "gitleaks")
        shutil.copyfile(self.fake, named)
        named.chmod(0o755)
        self.assertEqual(git(self.repo, "config", "--get-all", "hooks.gitleaks", check=False).returncode, 1,
                         "precondition: the key hooks.gitleaks is not set")
        env = git_env({"PATH": os.pathsep.join([str(folder)] + self.tool_dirs("git"))})
        self.assert_clean(run_gate(self.repo, "full", env=env), "full", "program over PATH")
        self.assertTrue(fake_calls(self.fake)[:2] == ["version", "git"], "the program on PATH was not the one called")

    def test_p5_forms_of_the_version_output(self):
        forms = {
            "other version": (b"9.9.9\n", 0), "empty": (b"", 0), "several lines": (b"8.30.1\n8.30.1\n", 0),
            "bom": (b"\xef\xbb\xbf8.30.1\n", 0), "crlf": (b"8.30.1\r\n", 0), "control": (b"8.30.1\x07\n", 0),
            "not utf8": (b"8.30.1\xff\n", 0), "return value": (b"8.30.1\n", 1), "no lf": (b"8.30.1", 0),
            "two lf": (b"8.30.1\n\n", 0), "space": (b"8.30.1 \n", 0),
        }
        for label, (out, rc) in forms.items():
            with self.case(form=label):
                fake = make_fake(WORK / ("fv" + rnd(6, string.ascii_lowercase)), version_out=out, version_rc=rc)
                self.refused("full", "--binary", fake, kind="binary-version", what=label)
                self.assertEqual(fake_calls(fake), ["version"], f"{label}: the scan ran after a refused version")

    def test_p5_only_the_exact_version_output_passes(self):
        self.assertEqual((self.repo / ".gitleaks-version").read_bytes(), b"8.30.1\n", "precondition: version file")
        self.assert_clean(run_gate(self.repo, "full", "--binary", self.fake), "full", "exact version output")

    def test_p5_a_version_call_over_the_time_limit_is_an_error(self):
        folder = Path(self.fake).parent
        source = (folder / "fake_impl.py").read_text(encoding="utf-8")
        self.assertEqual(source.count(VERSION_BRANCH), 1, "precondition: the version branch of the fake occurs once")
        self.assert_clean(run_gate(self.repo, "full", "--binary", self.fake), "full",
                          "precondition: the fake answers before it is made slow")
        slow = source.replace(VERSION_BRANCH, "time.sleep(5)\n" + VERSION_BRANCH)
        self.assertTrue(slow != source and slow.count("time.sleep(5)\n" + VERSION_BRANCH) == 1,
                        "precondition: the version branch of the fake is made slow exactly once")
        (folder / "fake_impl.py").write_text(slow, encoding="utf-8")
        module = load_gate(self.repo)
        module.SHORT_TIMEOUT = 1
        module.LONG_TIMEOUT = 1
        self.assert_error(call_main(module, "full", "--binary", self.fake), "full", "binary-version",
                          "version call over the time limit")


class P5Files(SafeCase):
    def setUp(self):
        self.repo = new_repo()
        commit(self.repo)
        self.fake = make_fake(WORK / ("ff" + rnd(6, string.ascii_lowercase)))

    def swap(self, rel: str, kind: str):
        path = self.repo / rel
        if path.exists():
            path.unlink()
        if kind == "folder":
            path.mkdir()

    def test_p5_missing_or_folder_config_and_template(self):
        for label, rel in (("config", ".gitleaks.toml"), ("template", "scripts/gitleaks_findings.tmpl")):
            for kind in ("missing", "folder"):
                with self.case(file=label, form=kind):
                    self.setUp()
                    self.swap(rel, kind)
                    self.assert_error(run_gate(self.repo, "full", "--binary", self.fake), "full", "config-file", kind)
                    self.assertEqual(fake_calls(self.fake), [], "the program was called before the file check")

    def test_p5_forms_of_the_ignore_file_are_refused_before_the_first_call(self):
        sha = "a" * 40
        forms = {
            "bom": b"\xef\xbb\xbf# c\n", "cr": b"# c\r\n", "not utf8": b"# \xff\n",
            "free text": ("this is not a line " + token(12) + "\n").encode(),
            "no commit": b"a.txt:github-pat:4\n", "short commit": ("a" * 39 + ":a.txt:github-pat:4\n").encode(),
            "upper case commit": ("A" * 40 + ":a.txt:github-pat:4\n").encode(), "rule with space": (sha + ":a.txt:a b:4\n").encode(),
            "line zero": (sha + ":a.txt:github-pat:0\n").encode(), "line letters": (sha + ":a.txt:github-pat:x\n").encode(),
            "negative line": (sha + ":a.txt:github-pat:-1\n").encode(), "empty rule": (sha + ":a.txt::1\n").encode(),
            "empty path": (sha + "::github-pat:1\n").encode(), "long rule": (sha + ":a.txt:" + "r" * 101 + ":1\n").encode(),
            "duplicate": (sha + ":a.txt:github-pat:4\n" + sha + ":a.txt:github-pat:4\n").encode(),
            "indented comment": b" # c\n",
        }
        for label, content in forms.items():
            with self.case(form=label):
                self.setUp()
                write(self.repo, ".gitleaksignore", content)
                self.assert_error(run_gate(self.repo, "full", "--binary", self.fake), "full", "ignore-file", label)
                self.assertEqual(fake_calls(self.fake), [], f"{label}: the program was called before the ignore file check")
        for label in ("missing", "folder"):
            with self.case(form=label):
                self.setUp()
                self.swap(".gitleaksignore", label)
                self.assert_error(run_gate(self.repo, "full", "--binary", self.fake), "full", "ignore-file", label)
                self.assertEqual(fake_calls(self.fake), [], f"{label}: the program was called before the ignore file check")

    def test_p5_valid_ignore_files_pass(self):
        sha = "b" * 40
        text = ("# reason one\n\n# reason two\n" + sha + ":dir/a:b.txt:github-pat:3\n" + "c" * 64 + ":x.txt:rule.name_1:12\n")
        write(self.repo, ".gitleaksignore", text)
        self.assert_clean(run_gate(self.repo, "full", "--binary", self.fake), "full", "valid ignore file")


class P5InvalidFiles(BinaryCase):
    def test_p5_an_invalid_configuration_is_a_scan_error(self):
        repo = new_repo()
        commit(repo)
        write(repo, ".gitleaks.toml", "this is [not valid toml\n")
        self.assert_error(self.gate(repo, "full"), "full", "scan-error", "invalid configuration")

    def test_p5_an_invalid_template_is_a_scan_error(self):
        repo = new_repo()
        commit(repo)
        write(repo, "scripts/gitleaks_findings.tmpl", "{{ range .x }\n")
        self.assert_error(self.gate(repo, "full"), "full", "scan-error", "invalid template")


# ---------------------------------------------------------------------------------------------
# P6, P7: environment and exceptions
# ---------------------------------------------------------------------------------------------

class P6EnvironmentEffect(BinaryCase):
    def test_p6_foreign_configuration_in_the_environment_has_no_effect(self):
        empty = WORK / "norules.toml"
        empty.write_text('title = "none"\n', encoding="utf-8")
        repo = new_repo()
        write(repo, "a.py", github_line())
        head = commit(repo)
        env = git_env({"GITLEAKS_CONFIG": str(empty), "GITLEAKS_CONFIG_TOML": empty.read_text()})
        self.assert_findings(self.gate(repo, "full", env=env), "full", {("github-pat", f"file:{name_id('a.py')}:1", head[:8])},
                             "foreign configuration in the environment")


class P7Exceptions(BinaryCase):
    def test_p7_allow_comment_has_no_effect(self):
        repo = new_repo()
        write(repo, "a.py", github_line()[:-1] + "  # gitleaks:allow\n")
        head = commit(repo)
        self.assert_findings(self.gate(repo, "full"), "full", {("github-pat", f"file:{name_id('a.py')}:1", head[:8])},
                             "allow comment")

    def test_p7_a_commit_bound_fingerprint_hides_exactly_that_finding(self):
        repo = new_repo()
        line = github_line()
        write(repo, "x.txt", line)
        first = commit(repo, "first")
        write(repo, "y.txt", line)
        second = commit(repo, "second")
        both = {("github-pat", f"file:{name_id('x.txt')}:1", first[:8]), ("github-pat", f"file:{name_id('y.txt')}:1", second[:8])}
        self.assert_findings(self.gate(repo, "full"), "full", both, "without a fingerprint")
        write(repo, ".gitleaksignore", f"# reason of the test\n{first}:x.txt:github-pat:1\n")
        self.assert_findings(self.gate(repo, "full"), "full", {("github-pat", f"file:{name_id('y.txt')}:1", second[:8])},
                             "with the fingerprint of the first commit")


# ---------------------------------------------------------------------------------------------
# P8: output of gitleaks
# ---------------------------------------------------------------------------------------------

class P8ReportForms(SafeCase):
    MARK = "RAWMARK" + token(6)

    def setUp(self):
        self.repo = new_repo()
        commit(self.repo)

    def fake(self, out: bytes, rc: int) -> str:
        return make_fake(WORK / ("fr" + rnd(6, string.ascii_lowercase)), scan_out=out, scan_rc=rc,
                         scan_err=(self.MARK + " stderr\n").encode())

    def test_p8_every_form_of_the_report_is_an_error_of_its_class_and_the_raw_line_never_shows(self):
        mark = self.MARK
        good = {"rule": "github-pat", "file": "a.txt", "line": 1, "commit": "a" * 40}

        def line(**changes):
            item = dict(good, **changes)
            return (json.dumps(item) + "\n").encode("utf-8")

        scan_error = {"return 1": (1, b""), "return 126": (126, b""), "return 3 without line": (3, b""),
                      "return 0 with line": (0, line(rule=mark))}
        report_format = {
            "not utf8": (3, b'{"rule": "' + mark.encode() + b'\xff"}\n'),
            "not json": (3, (mark + "\n").encode()), "json array": (3, b'["' + mark.encode() + b'"]\n'),
            "duplicate key": (3, ('{"rule": "' + mark + '", "rule": "x", "file": "a", "line": 1, "commit": ""}\n').encode()),
            "missing key": (3, ('{"rule": "' + mark + '", "file": "a", "line": 1}\n').encode()),
            "extra key": (3, ('{"rule": "' + mark + '", "file": "a", "line": 1, "commit": "", "more": 1}\n').encode()),
            "rule number": (3, line(rule=5)), "rule empty": (3, line(rule="")), "rule charset": (3, line(rule=mark + " x")),
            "rule long": (3, line(rule="r" * 101)), "file number": (3, line(file=7)), "file empty": (3, line(file="")),
            "file lone surrogate": (3, line(file=mark + "\ud800")),
            "line text": (3, line(line=mark)), "line bool": (3, line(line=True)), "line zero": (3, line(line=0)),
            "line huge": (3, line(line=10**9 + 1)), "line float": (3, line(line=1.5)),
            "commit text": (3, line(commit=mark)), "commit short": (3, line(commit="a" * 8)),
            "commit empty in history": (3, line(commit="")), "commit upper": (3, line(commit="A" * 40)),
            "no final lf": (3, line()[:-1]), "empty line inside": (3, line() + b"\n" + line()),
            "nan": (3, b'{"rule": "x", "file": "a", "line": NaN, "commit": ""}\n'),
        }
        for kind, forms in (("scan-error", scan_error), ("report-format", report_format)):
            for label, (rc, out) in forms.items():
                with self.case(form=label):
                    result = run_gate(self.repo, "full", "--binary", self.fake(out, rc))
                    self.assert_error(result, "full", kind, label)

    def test_p8_stderr_of_gitleaks_never_shows(self):
        self.assert_clean(run_gate(self.repo, "full", "--binary", self.fake(b"", 0)), "full", "clean run with stderr")
        result = run_gate(self.repo, "full", "--binary", self.fake(report_line(), 3))
        # the fake answers every scan alike, so the message scans add their own lines
        self.assert_found(result, "full", {("github-pat", f"file:{name_id('a.txt')}:1", "a" * 8)}, "finding with stderr")

    def test_p8_a_valid_line_is_a_finding_and_prints_only_its_parts(self):
        name = "N" + token(12) + ".txt"
        fake = make_fake(WORK / ("fg" + rnd(6, string.ascii_lowercase)), scan_out=report_line(file=name, line=3), scan_rc=3)
        result = run_gate(self.repo, "full", "--binary", fake)
        self.assert_found(result, "full", {("github-pat", f"file:{name_id(name)}:3", "a" * 8)}, "valid line")
        self.assertFalse(name[:8] in result[1] + result[2], "a piece of the name is in the output")

    def test_p8_empty_commit_and_file_where_the_table_allows_them(self):
        git(self.repo, "config", "hooks.gitleaks",
            make_fake(WORK / ("fc" + rnd(6, string.ascii_lowercase)), scan_out=report_line(commit=""), scan_rc=3))
        self.assert_findings(run_gate(self.repo, "pre-commit"), "pre-commit",
                             {("github-pat", f"file:{name_id('a.txt')}:1", "staged")}, "empty commit at pre-commit")
        git(self.repo, "config", "hooks.gitleaks",
            make_fake(WORK / ("fd" + rnd(6, string.ascii_lowercase)), scan_out=report_line(file="", commit=""), scan_rc=3))
        message = WORK / ("m" + rnd(8, string.ascii_lowercase))
        message.write_bytes(b"subject\n")
        self.assert_findings(run_gate(self.repo, "commit-msg", str(message)), "commit-msg",
                             {("github-pat", "message:1", "staged")}, "empty file and commit at stdin")


# ---------------------------------------------------------------------------------------------
# message file (table row "Nachrichtendatei")
# ---------------------------------------------------------------------------------------------

class MessageFileForms(SafeCase):
    def setUp(self):
        self.repo = new_repo()
        commit(self.repo)
        self.fake = make_fake(WORK / ("fm" + rnd(6, string.ascii_lowercase)))
        git(self.repo, "config", "hooks.gitleaks", self.fake)

    def test_a_missing_message_file_or_a_folder_is_an_error(self):
        missing = WORK / ("M" + token(12))
        self.assert_error(run_gate(self.repo, "commit-msg", str(missing)), "commit-msg", "message-file", "missing")
        folder = WORK / ("d" + rnd(8, string.ascii_lowercase))
        folder.mkdir()
        self.assert_error(run_gate(self.repo, "commit-msg", str(folder)), "commit-msg", "message-file", "folder")

    def test_the_bytes_of_the_message_go_to_gitleaks_unchanged(self):
        kept = Path(self.fake).parent / "stdin.bin"
        forms = {"empty": b"", "crlf": b"subject\r\n\r\nbody\r\n", "not utf8": b"subject \xe9\xff\n",
                 "nul": b"sub\x00ject\n", "lone cr": b"subject\rmore\n"}
        for label, data in forms.items():
            with self.case(form=label):
                if kept.exists():
                    kept.unlink()
                path = WORK / ("m" + rnd(8, string.ascii_lowercase))
                path.write_bytes(data)
                self.assert_clean(run_gate(self.repo, "commit-msg", str(path)), "commit-msg", label)
                if data:
                    self.assertTrue(kept.exists() and kept.read_bytes() == data, f"{label}: gitleaks got other bytes")
                else:
                    self.assertFalse(kept.exists(), "an empty message was scanned")


class MessageFileBait(BinaryCase):
    def test_a_bait_in_a_crlf_message_is_a_finding(self):
        repo = new_repo()
        commit(repo)
        path = WORK / ("m" + rnd(8, string.ascii_lowercase))
        path.write_bytes(("subject\r\n\r\n" + github_line()[:-1] + "\r\n").encode())
        self.assert_found(self.gate(repo, "commit-msg", str(path)), "commit-msg", {("github-pat", "message:3", "staged")},
                          "bait in a crlf message")


# ---------------------------------------------------------------------------------------------
# P9, P11: static checks
# ---------------------------------------------------------------------------------------------

class P9Template(SafeCase):
    def test_p9_the_template_names_only_four_fields(self):
        text = (ROOT / "scripts" / "gitleaks_findings.tmpl").read_text(encoding="utf-8")
        self.assertTrue(set(re.findall(r"\.[A-Z][A-Za-z]*", text)) == {".RuleID", ".File", ".StartLine", ".Commit"},
                        "the template reads other fields than the four")
        self.assertTrue(set(re.findall(r'"([a-z]+)"', text)) - {"n"} == {"rule", "file", "line", "commit"},
                        "the template writes other keys than the four")


class P11Static(SafeCase):
    WORKFLOWS = (".github/workflows/secret-scan.yml", ".for" + "gejo/workflows/secret-scan.yml")
    LOOKUP_WORD = "loc" + "ate"

    def texts(self):
        found = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in self.WORKFLOWS if (ROOT / rel).exists()}
        self.assertTrue(self.WORKFLOWS[0] in found, "the GitHub workflow is missing")
        return found

    def test_p11_workflows_state_the_checks(self):
        digests = set()
        for number, text in enumerate(self.texts().values()):
            with self.case(workflow=number):
                for needle in (".gitleaks-version", "scripts/gitleaks_gate.py full", "test_gitleaks_gate.py",
                               "GITLEAKS_REQUIRED=1", "-s tests"):
                    self.assertTrue(needle in text, "a required step text is missing")
                self.assertTrue(re.search(r"permissions:\n  contents: read\n", text) is not None, "permissions")
                self.assertTrue(0 <= text.find("sha256sum -c") < text.find("tar -xzf"), "checksum before unpacking")
                for needle in ("upload-artifact", "--verbose", self.LOOKUP_WORD):
                    self.assertFalse(needle in text, "a forbidden text is present")
                self.assertTrue(set(re.findall(r"uses: (\S+)", text)) == {"actions/checkout@v4"}, "foreign action")
                found = re.findall(r"echo \"([0-9a-f]{64})  ", text)
                self.assertEqual(len(found), 1, "not exactly one checksum literal")
                digests.add(found[0])
        self.assertEqual(len(digests), 1, "the workflows carry different checksums")

    def test_p11_hooks_call_the_gate_and_never_the_lookup(self):
        hooks = ROOT / ".githooks"
        for name in ("pre-commit", "commit-msg", "pre-push"):
            text = (hooks / name).read_text(encoding="utf-8")
            with self.case(hook=name):
                self.assertTrue("scripts/gitleaks_gate.py" in text, "the hook does not call the gate")
                self.assertFalse(self.LOOKUP_WORD in text, "the hook names the lookup")
                self.assertFalse("\r" in text, "the hook has a carriage return")
                self.assertTrue("top=$(git rev-parse --show-toplevel) || exit 1" in text, "the top folder is not checked")

    def test_p11_the_pre_push_hook_refuses_when_the_push_lines_cannot_be_stored(self):
        text = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
        self.assertEqual(text.count('cat > "$lines"'), 1, "not exactly one store of the push lines")
        self.assertTrue('cat > "$lines" || {' in text, "the store of the push lines is not checked")

    def test_p11_this_file_never_runs_the_lookup(self):
        self.assertFalse(self.LOOKUP_WORD in Path(__file__).read_text(encoding="utf-8"), "this file names the lookup")


# ---------------------------------------------------------------------------------------------
# P13, P16, P18a, P19
# ---------------------------------------------------------------------------------------------

class P13Counterexamples(BinaryCase):
    def test_p13_lines_without_a_value_are_no_finding(self):
        repo = new_repo()
        write(repo, "a.py", "\n".join([
            'password = "<your password>"', 'password = "${DB_PASSWORD}"', 'password = os.environ["DB_PASSWORD"]',
            'password = os.environ.get("DB_PASSWORD")', "password: str", 'api_key: str = os.environ["API_KEY"]',
            "token: str", 'password = "changeme"', 'pwd = "xxxxxxxx"', 'password: "<placeholder>"', ""]))
        write(repo, "b.sql", "ALTER ROLE app_user WITH LOGIN PASSWORD :'app_pw';\nCREATE ROLE reader PASSWORD :'reader_pw';\n")
        write(repo, "c.env", "DB_PASSWORD=${DB_PASSWORD}\nMAIL_PASSWORD=changeme\n")
        commit(repo)
        self.assert_clean(self.gate(repo, "full"), "full", "lines without a value")


class P16MergeFull(BinaryCase):
    def test_p16_a_value_only_a_merge_adds_is_found_once(self):
        repo = new_repo()
        write(repo, "a.txt", "one\n")
        commit(repo, "base")
        git(repo, "checkout", "-q", "-b", "side")
        write(repo, "b.txt", "side\n")
        commit(repo, "side")
        git(repo, "checkout", "-q", "main")
        write(repo, "c.txt", "main\n")
        commit(repo, "main")
        git(repo, "merge", "-q", "--no-ff", "--no-commit", "side")
        write(repo, "evil.txt", github_line())
        merge = commit(repo, "merge")
        self.assert_findings(self.gate(repo, "full"), "full", {("github-pat", f"file:{name_id('evil.txt')}:1", merge[:8])},
                             "value only in the merge")


class P18aNamesAsIdentifiers(BinaryCase):
    @staticmethod
    def build(repo: Path):
        values = baits()
        random_name = token(20)
        at_name = "at" + "@" + token(6)
        tag_name = "t" + token(18)
        files = {random_name: values["passwort_quotiert"], at_name: values["phrase"], "harmless-notes.txt": values["app_passwort"]}
        for name, value in files.items():
            write(repo, name, "pass" + "word" + ' = "' + value + '"\n')
        commit(repo, "files")
        git(repo, "tag", "-a", tag_name, "-m", "note token=" + values["github_pat"] + " " + rnd(6))
        return values, files, tag_name

    def test_p18a_no_name_and_no_bait_in_the_output(self):
        repo = new_repo()
        values, files, tag_name = self.build(repo)
        result = self.gate(repo, "full")
        self.assertEqual(result[0], 1, "the return value is not 1 (finding)")
        everything = result[1] + result[2]
        for name in list(files) + [tag_name]:
            self.assertEqual(sum(1 for piece in windows_of(name) | {name} if piece in everything), 0, "a name is in the output")
        self.assertEqual(leaked(everything, list(values.values())), 0, "a piece of a bait is in the output")
        self.assertEqual(sum(1 for line in result[1].splitlines() if not LINE_FORM.fullmatch(line)), 0,
                         "a line of stdout is not of the finding form")
        locations = [loc for _, loc, _ in parse_lines(result[1])]
        for name in files:
            self.assertTrue(any(loc.startswith(f"file:{name_id(name)}:") for loc in locations), "a file is not reported")
        self.assertTrue(any(loc.startswith(f"tag:{name_id(tag_name)}:") for loc in locations), "the tag is not reported")


class P19ChildEnvironment(SafeCase):
    def test_p19_the_child_process_sees_neither_variable(self):
        repo = new_repo()
        commit(repo)
        folder = WORK / ("fe" + rnd(6, string.ascii_lowercase))
        fake = make_fake(folder)
        env = git_env({"GITLEAKS_CONFIG": str(WORK / "x.toml"), "GITLEAKS_CONFIG_TOML": "title = 'x'"})
        self.assert_clean(run_gate(repo, "full", "--binary", fake, env=env), "full", "variables set")
        self.assertTrue((folder / "env.txt").read_text() == "0 0", "a variable reached the child process")


# ---------------------------------------------------------------------------------------------
# annotated tags, names that are not UTF-8, outputs of git, time limits
# ---------------------------------------------------------------------------------------------

class NestedTags(BinaryCase):
    def test_the_text_of_a_tag_inside_a_tag_is_checked(self):
        repo = new_repo()
        commit(repo)
        inner, outer = "i" + token(18), "o" + token(18)
        git(repo, "tag", "-a", inner, "-m", "note token=" + baits()["github_pat"])
        git(repo, "tag", "-a", outer, "refs/tags/" + inner, "-m", "plain note")
        git(repo, "tag", "-d", inner)  # only the outer tag has a ref; the inner text is reachable through it
        result = self.gate(repo, "full")
        self.assertEqual(result[0], 1, "the return value is not 1 (finding)")
        self.assertTrue(any(rule == "github-pat" and loc.startswith(f"tag:{name_id(outer)}:") and ref == "-"
                            for rule, loc, ref in parse_lines(result[1])), "the text of the inner tag is not reported")


class TreeAndBlobEnds(BinaryCase):
    """F4 "content of all refs": the history walk passes over a tree or blob that a ref or the end of a tag chain
    names, so gitleaks never reads it; `full` refuses each such ref with `git-error`, as `pre-push` does."""

    def test_a_ref_or_a_tag_chain_that_ends_at_a_tree_or_a_blob_is_a_git_error_in_full(self):
        for label in ("ref to a blob", "ref to a tree", "tag on a blob", "nested tag on a blob"):
            with self.case(form=label):
                repo = new_repo()
                commit(repo)  # a clean commit next to the bait
                target = git(repo, "hash-object", "-w", "--stdin", data=github_line().encode()).stdout.decode().strip()
                if label == "ref to a tree":
                    target = git(repo, "mktree", data=f"100644 blob {target}\tkey.txt\n".encode()).stdout.decode().strip()
                self.assert_clean(self.gate(repo, "full"), "full", f"precondition: {label}: clean without the ref")
                name = "t" + token(10)
                if label.startswith("ref"):
                    git(repo, "update-ref", "refs/tags/" + name, target)
                else:
                    git(repo, "tag", "-a", name, target, "-m", "plain note")
                if label.startswith("nested"):
                    git(repo, "tag", "-a", "o" + token(10), "refs/tags/" + name, "-m", "plain note")
                    git(repo, "tag", "-d", name)  # only the outer tag has a ref
                self.assert_error(self.gate(repo, "full"), "full", "git-error", label)


class NonUtf8FileNames(BinaryCase):
    RAW = b"n\xff\xfeam\xc3\xa4e\xe2\x82.txt"  # two bad bytes, a valid letter and a cut multi-byte character

    def test_the_identifier_of_a_name_that_is_not_utf8_is_built_from_its_raw_bytes_in_history(self):
        repo = new_repo()
        commit(repo)
        add_raw_name(repo, self.RAW, github_line())
        git(repo, "commit", "-q", "-m", "raw name")  # from the index; the name never touches the file system
        head = rev(repo, "HEAD")
        self.assert_findings(self.gate(repo, "full"), "full", {("github-pat", f"file:{name_id(self.RAW)}:1", head[:8])},
                             "identifier of a raw name in history")

    def test_the_identifier_of_a_name_that_is_not_utf8_is_built_from_its_raw_bytes_when_staged(self):
        repo = new_repo()
        commit(repo)
        add_raw_name(repo, self.RAW, github_line())
        self.assert_findings(self.gate(repo, "pre-commit"), "pre-commit",
                             {("github-pat", f"file:{name_id(self.RAW)}:1", "staged")}, "identifier of a staged raw name")

    def test_two_names_with_the_same_reported_form_are_both_named(self):
        repo = new_repo()
        commit(repo)
        first = b"a\xff" + token(8).encode() + b".txt"
        second = first.replace(b"\xff", b"\xfe")
        self.assertTrue(first.count(b"\xff") == 1 and second.count(b"\xfe") == 1 and second != first,
                        "precondition: the second name differs from the first in exactly one bad byte")
        add_raw_name(repo, first, github_line())
        add_raw_name(repo, second, github_line())
        git(repo, "commit", "-q", "-m", "raw names")
        head = rev(repo, "HEAD")
        expected = {("github-pat", f"file:{name_id(first)}:1", head[:8]), ("github-pat", f"file:{name_id(second)}:1", head[:8])}
        self.assert_findings(self.gate(repo, "full"), "full", expected, "two raw names with one reported form")


class GitAnswers(SafeCase):
    """Runs of `main` with answers of git changed on the way; every change counts what it altered."""

    def setUp(self):
        self.repo = new_repo()
        write(self.repo, "a.txt", "one\n")
        self.base = commit(self.repo, "base")
        write(self.repo, "b.txt", "two\n")
        self.head = commit(self.repo, "second")
        git(self.repo, "tag", "-a", "t" + token(10), "-m", "plain note")
        self.fake = make_fake(WORK / ("fo" + rnd(6, string.ascii_lowercase)))
        git(self.repo, "config", "hooks.gitleaks", self.fake)
        self.module = load_gate(self.repo)
        self.altered = self.misses = self.unfit = 0

    def changed(self, mutate, *args: str, stdin: bytes = b""):
        """`main` with `mutate` applied to every answer of git; counts the answers it altered.

        Every answer the change targets must first be the one of a sound call: return value 0 and an output that is
        empty or ends with LF. Any other answer is unfit, so a failure of git itself never passes as the change.
        """
        real_run = subprocess.run
        self.altered = self.misses = self.unfit = 0

        def run(command, *a, **kw):
            done = real_run(command, *a, **kw)
            changed = mutate(list(command), done)
            if changed is None:
                return done
            if done.returncode != 0 or not (done.stdout == b"" or done.stdout.endswith(b"\n")):
                self.unfit += 1
            if (changed.returncode, changed.stdout) != (done.returncode, done.stdout):
                self.altered += 1
            return changed

        with mock.patch.object(subprocess, "run", run):
            return call_main(self.module, *args, stdin=stdin)

    def assert_changed_once(self, what: str):
        """The answers the change targets were sound, it met its target as intended and altered exactly one answer."""
        self.assertEqual(self.unfit, 0, f"{what}: precondition: an answer of git was not sound before the change")
        self.assertEqual(self.misses, 0, f"{what}: the target of the change is not there as intended")
        self.assertEqual(self.altered, 1, f"{what}: not exactly one answer of git was changed")

    @staticmethod
    def on(sub: str, change, first: str | None = None):
        def mutate(command, done):
            if command[:1] == ["git"] and command[3:4] == [sub] and (first is None or command[4:5] == [first]):
                rc, out = change(done.returncode, done.stdout)
                return subprocess.CompletedProcess(command, rc, out, b"")
            return None
        return mutate

    def replaced(self, old: bytes, new: bytes):
        """Replaces `old` in an answer that holds it; an answer that holds it more than once is a miss."""
        def change(rc, out):
            found = out.count(old)
            if found == 0:
                return rc, out
            if found != 1:
                self.misses += 1
            return rc, out.replace(old, new, 1)
        return change

    def first_line_field(self, index: int, field, count: int):
        """Changes one field of the first line; a first line without `count` fields is a miss."""
        def change(rc, out):
            lines = out.split(b"\n")
            parts = lines[0].split(b" ")
            if len(parts) != count:
                self.misses += 1
                return rc, out
            parts[index] = field(parts[index])
            lines[0] = b" ".join(parts)
            return rc, b"\n".join(lines)
        return change

    def first_commit(self, content_change):
        """Changes the content of the first object of an answer, if it is a commit, and writes its new size."""
        def change(rc, out):
            end = out.find(b"\n")
            head = out[:end].split(b" ") if end > 0 else []
            if len(head) != 3 or head[1] != b"commit":
                return rc, out
            size = int(head[2])
            content = content_change(out[end + 1:end + 1 + size])
            return rc, b" ".join(head[:2] + [str(len(content)).encode()]) + b"\n" + content + out[end + 1 + size:]
        return change

    def once(self, change):
        """`change` for the first answer it alters only; every later answer passes unchanged."""
        def first_only(rc, out):
            return (rc, out) if self.altered else change(rc, out)
        return first_only

    def unterminated(self, content: bytes) -> bytes:
        """The content without its blank line and without its final LF: a header that never ends."""
        if content.count(b"\n\n") != 1:
            self.misses += 1
        return content.replace(b"\n\n", b"\n ", 1)[:-1]

    def without_message(self, content: bytes) -> bytes:
        """The content without its blank line, the last line still ending with LF: a valid object without message."""
        if content.count(b"\n\n") != 1:
            self.misses += 1
        return content.replace(b"\n\n", b"\n ", 1)


class GitOutputForms(GitAnswers):
    """Table row "Ausgaben von Git": every faulty form ends with `git-error`, checked at `main`."""

    def test_faulty_outputs_of_git_in_full_are_git_errors(self):
        self.assert_clean(call_main(self.module, "full"), "full", "precondition: the unchanged run is clean")
        forms = {
            "rev-list fails": ("rev-list", None, lambda rc, out: (128, b"")),
            "rev-list not an id": ("rev-list", None, lambda rc, out: (rc, out + b"zz\n")),
            "rev-list upper case": ("rev-list", None, lambda rc, out: (rc, out.upper())),
            "rev-list empty line": ("rev-list", None, lambda rc, out: (rc, b"\n" + out)),
            "rev-list no final lf": ("rev-list", None, lambda rc, out: (rc, out[:-1])),
            "refs fail": ("for-each-ref", None, lambda rc, out: (1, b"")),
            "refs field count": ("for-each-ref", None, lambda rc, out: (rc, out + b"tag x\n")),
            "refs unknown type": ("for-each-ref", None, self.replaced(b"commit ", b"commix ")),
            "refs id not hex": ("for-each-ref", None, self.first_line_field(1, lambda field: b"z" * 40, 3)),
            "refs name not utf8": ("for-each-ref", None, lambda rc, out: (rc, out + b"commit " + b"a" * 40 + b" refs/heads/\xff\n")),
            "objects fail": ("cat-file", "--batch", lambda rc, out: (128, b"")),
            "object truncated": ("cat-file", "--batch", lambda rc, out: (rc, out[:-2])),
            "object no final lf": ("cat-file", "--batch", lambda rc, out: (rc, out[:-1])),
            "object extra bytes": ("cat-file", "--batch", lambda rc, out: (rc, out + b"x")),
            "object missing": ("cat-file", "--batch", lambda rc, out: (rc, out.split(b" ")[0] + b" missing\n")),
            "object other id": ("cat-file", "--batch", self.first_line_field(0, lambda field: b"f" * 40, 3)),
            "object unknown type": ("cat-file", "--batch", self.first_line_field(1, lambda field: b"bogus", 3)),
            "object size not digits": ("cat-file", "--batch", self.first_line_field(2, lambda field: field + b"x", 3)),
            "object without header end": ("cat-file", "--batch", self.first_commit(self.unterminated)),
            "tag target not an id": ("cat-file", "--batch", self.replaced(b"\nobject ", b"\nobjekt ")),
        }
        for label, (sub, first, change) in forms.items():
            with self.case(form=label):
                result = self.changed(self.on(sub, change, first), "full")
                self.assert_changed_once(label)
                self.assert_error(result, "full", "git-error", label)

    def test_faulty_outputs_and_objects_in_pre_push_are_git_errors(self):
        line = f"refs/heads/main {self.head} refs/heads/main {self.base}\n".encode()
        self.assert_clean(call_main(self.module, "pre-push", "origin", "x", stdin=line), "pre-push",
                          "precondition: the unchanged run is clean")
        forms = {
            "remote refs not an id": ("for-each-ref", None, lambda rc, out: (rc, b"zz\n")),
            "existence check fails": ("cat-file", "-e", lambda rc, out: (128, b"")),
            "objects truncated": ("cat-file", "--batch", lambda rc, out: (rc, out[:-2])),
        }
        for label, (sub, first, change) in forms.items():
            with self.case(form=label):
                result = self.changed(self.on(sub, change, first), "pre-push", "origin", "x", stdin=line)
                self.assert_changed_once(label)
                self.assert_error(result, "pre-push", "git-error", label)
        blob = git(self.repo, "hash-object", "-w", "--stdin", data=b"plain\n").stdout.decode().strip()
        for label, local in (("pushed object missing", "1" * 40), ("pushed blob", blob)):
            with self.case(form=label):
                data = f"refs/tags/x{token(6)} {local} refs/tags/y {'0' * 40}\n".encode()
                self.assert_error(call_main(self.module, "pre-push", "origin", "x", stdin=data), "pre-push", "git-error", label)


class ValidObjectForms(GitAnswers):
    """Objects git's own check accepts are read, not refused (git `fsck.c`, `verify_headers` and the tag check).

    A commit or tag without a blank line has no message when its last header line ends with LF; without that LF
    its header is unterminated and stays an error, and so does a damaged frame of the batch answer around it.
    A tag may name its target id in upper case.
    """

    ZERO = "0" * 40

    @staticmethod
    def ident() -> bytes:
        return b"probe <probe" + b"@" + b"example.invalid> 1700000000 +0000"

    def store(self, kind: str, text: bytes, checked: bool = True) -> str:
        """Writes an object; `checked` lets git run its own object check first (`hash-object` without `--literally`)."""
        extra = [] if checked else ["--literally"]
        return git(self.repo, "hash-object", "-t", kind, "-w", "--stdin", *extra, data=text).stdout.decode().strip()

    def bare_commit(self, final: bytes = b"\n") -> str:
        tree = rev(self.repo, "HEAD^{tree}").encode()
        text = (b"tree " + tree + b"\nparent " + self.head.encode() + b"\nauthor " + self.ident()
                + b"\ncommitter " + self.ident() + final)
        return self.store("commit", text, checked=final == b"\n")

    def bare_tag(self, name: str, final: bytes = b"\n") -> str:
        text = b"object " + self.head.encode() + b"\ntype commit\ntag " + name.encode() + b"\ntagger " + self.ident() + final
        return self.store("tag", text, checked=final == b"\n")

    def assert_clean_in_full_and_pre_push(self, ref: str, sha: str, what: str):
        self.assert_clean(call_main(self.module, "full"), "full", f"{what} in full")
        line = f"{ref} {sha} {ref} {self.ZERO}\n".encode()
        self.assert_clean(call_main(self.module, "pre-push", "origin", "x", stdin=line), "pre-push", f"{what} in pre-push")

    def test_a_commit_without_a_message_is_read_in_full_and_pre_push(self):
        sha = self.bare_commit()
        git(self.repo, "update-ref", "refs/heads/bare", sha)
        self.assert_clean_in_full_and_pre_push("refs/heads/bare", sha, "commit without a message")

    def test_a_tag_without_a_message_is_read_in_full_and_pre_push(self):
        name = "n" + token(10)
        sha = self.bare_tag(name)
        git(self.repo, "update-ref", "refs/tags/" + name, sha)
        self.assert_clean_in_full_and_pre_push("refs/tags/" + name, sha, "tag without a message")

    def test_a_commit_answer_without_its_blank_line_is_read(self):
        change = self.once(self.first_commit(self.without_message))
        result = self.changed(self.on("cat-file", change, "--batch"), "full")
        self.assert_changed_once("commit answer without its blank line")
        self.assert_clean(result, "full", "commit answer without its blank line")

    def test_an_unterminated_header_stays_an_error(self):
        for label in ("commit", "tag"):
            with self.case(form=label):
                name = "x" + token(10)
                ref = "refs/heads/" + name if label == "commit" else "refs/tags/" + name
                sha = self.bare_commit(final=b"") if label == "commit" else self.bare_tag(name, final=b"")
                git(self.repo, "update-ref", ref, sha)
                self.assert_error(call_main(self.module, "full"), "full", "git-error", f"unterminated {label} in full")
                line = f"{ref} {sha} {ref} {self.ZERO}\n".encode()
                self.assert_error(call_main(self.module, "pre-push", "origin", "x", stdin=line), "pre-push", "git-error",
                                  f"unterminated {label} in pre-push")
                git(self.repo, "update-ref", "-d", ref)

    def frame(self, sha: str, delta: int | None):
        """Damages the frame of the answer for `sha`: its size by `delta`, or (None) the LF after its content."""
        def change(rc, out):
            end = out.find(b"\n")
            head = out[:end].split(b" ") if end > 0 else []
            if len(head) != 3 or head[0] != sha.encode("ascii"):
                return rc, out
            size = int(head[2])
            if delta is not None:
                return rc, b" ".join(head[:2] + [str(size + delta).encode()]) + out[end:]
            stop = end + 1 + size
            if out[stop:stop + 1] != b"\n":
                self.misses += 1
                return rc, out
            return rc, out[:stop] + out[stop + 1:]
        return change

    def test_a_damaged_frame_around_an_object_without_a_message_is_an_error(self):
        sha = self.bare_commit()
        line = f"refs/heads/bare {sha} refs/heads/bare {self.ZERO}\n".encode()
        self.assert_clean(call_main(self.module, "pre-push", "origin", "x", stdin=line), "pre-push",
                          "precondition: the object without a message is read")
        for label, delta in (("size one more", 1), ("size one less", -1), ("no lf after the content", None)):
            with self.case(form=label):
                result = self.changed(self.on("cat-file", self.frame(sha, delta), "--batch"), "pre-push", "origin", "x",
                                      stdin=line)
                self.assert_changed_once(label)
                self.assert_error(result, "pre-push", "git-error", label)

    def test_a_tag_whose_target_id_is_in_upper_case_is_followed(self):
        name = "u" + token(10)
        text = (b"object " + self.head.upper().encode() + b"\ntype commit\ntag " + name.encode() + b"\ntagger "
                + self.ident() + b"\n\nplain note\n")
        sha = self.store("tag", text)
        git(self.repo, "update-ref", "refs/tags/" + name, sha)
        self.assertTrue(rev(self.repo, f"refs/tags/{name}^{{commit}}") == self.head, "precondition: git follows the tag")
        self.assert_clean_in_full_and_pre_push("refs/tags/" + name, sha, "upper-case target")


class NonUtf8RefName(SafeCase):
    def test_a_tag_name_that_is_not_utf8_is_a_git_error(self):
        repo = new_repo()
        commit(repo)
        name = "t" + token(10)
        git(repo, "tag", "-a", name, "-m", "plain note")
        fake = make_fake(WORK / ("fn" + rnd(6, string.ascii_lowercase)))
        self.assert_clean(run_gate(repo, "full", "--binary", fake), "full", "precondition: clean before the bad name")
        tag_object = rev(repo, "refs/tags/" + name)
        git(repo, "update-ref", "-d", "refs/tags/" + name)
        packed = repo / ".git" / "packed-refs"
        head = packed.read_bytes() if packed.exists() else b"# pack-refs with: peeled fully-peeled sorted \n"
        packed.write_bytes(head + tag_object.encode() + b" refs/tags/t\xff\xfe" + token(6).encode() + b"\n")
        refs = git(repo, "for-each-ref", "--format=%(refname)").stdout
        self.assertEqual(refs.count(b"refs/tags/t\xff\xfe"), 1, "precondition: git lists the name that is not utf8 once")
        self.assert_error(run_gate(repo, "full", "--binary", fake), "full", "git-error", "tag name not utf8")


class TimeLimits(SafeCase):
    """F4: every child process runs with the time limit of its mode; a timeout is an error of its class."""

    SHORT, LONG = 111, 777

    def setUp(self):
        self.repo = new_repo()
        write(self.repo, "a.txt", "one\n")
        self.base = commit(self.repo, "base")
        write(self.repo, "b.txt", "two\n")
        self.head = commit(self.repo, "second message")
        git(self.repo, "tag", "-a", "t" + token(10), "-m", "plain note")
        self.fake = make_fake(WORK / ("ft" + rnd(6, string.ascii_lowercase)))
        git(self.repo, "config", "hooks.gitleaks", self.fake)
        self.message = WORK / ("m" + rnd(8, string.ascii_lowercase))
        self.message.write_bytes(b"subject\n")
        self.push_line = f"refs/heads/main {self.head} refs/heads/main {self.base}\n".encode()

    def calls(self, mode: str):
        return {"pre-commit": (["pre-commit"], b""), "commit-msg": (["commit-msg", str(self.message)], b""),
                "pre-push": (["pre-push", "origin", "x"], self.push_line), "full": (["full"], b"")}[mode]

    def run_with(self, mode: str, hook):
        module = load_gate(self.repo)
        module.SHORT_TIMEOUT, module.LONG_TIMEOUT = self.SHORT, self.LONG
        real_run = subprocess.run

        def run(command, *a, **kw):
            hook(list(command), kw)
            return real_run(command, *a, **kw)

        args, stdin = self.calls(mode)
        with mock.patch.object(subprocess, "run", run):
            return call_main(module, *args, stdin=stdin)

    def test_every_child_process_has_the_time_limit_of_its_mode(self):
        for mode, limit in (("pre-commit", self.SHORT), ("commit-msg", self.SHORT), ("pre-push", self.LONG), ("full", self.LONG)):
            with self.case(mode=mode):
                seen: list = []
                result = self.run_with(mode, lambda command, kw: seen.append(kw.get("timeout")))
                self.assert_clean(result, mode, "precondition: the run is clean")
                self.assertTrue(len(seen) >= 3, "precondition: the run started its child processes")
                self.assertEqual(sum(1 for value in seen if value != limit), 0,
                                 f"a child process of {mode} runs without the time limit of its mode")

    def test_a_timeout_of_a_child_process_is_an_error_of_its_class(self):
        def is_git(sub, first=None):
            return lambda c: c[:1] == ["git"] and c[3:4] == [sub] and (first is None or c[4:5] == [first])

        def is_scan(head):
            return lambda c: c[:1] == [self.fake] and c[1:2] == [head]

        forms = {
            "pre-commit key": ("pre-commit", is_git("config"), "git-error"),
            "pre-commit version": ("pre-commit", is_scan("version"), "binary-version"),
            "pre-commit scan": ("pre-commit", is_scan("git"), "scan-error"),
            "commit-msg scan": ("commit-msg", is_scan("stdin"), "scan-error"),
            "pre-push existence check": ("pre-push", is_git("cat-file", "-e"), "git-error"),
            "pre-push objects": ("pre-push", is_git("cat-file", "--batch"), "git-error"),
            "pre-push remote refs": ("pre-push", is_git("for-each-ref"), "git-error"),
            "pre-push commits": ("pre-push", is_git("rev-list"), "git-error"),
            "pre-push message scan": ("pre-push", is_scan("stdin"), "scan-error"),
            "full commits": ("full", is_git("rev-list"), "git-error"),
            "full refs": ("full", is_git("for-each-ref"), "git-error"),
            "full scan": ("full", is_scan("git"), "scan-error"),
        }
        for label, (mode, matches, kind) in forms.items():
            with self.case(form=label):
                hits: list = []

                def hook(command, kw, matches=matches, hits=hits):
                    if matches(command):
                        hits.append(1)
                        raise subprocess.TimeoutExpired(command, kw.get("timeout"))
                result = self.run_with(mode, hook)
                self.assertEqual(len(hits), 1, f"{label}: the timeout was not put into exactly one child process")
                self.assert_error(result, mode, kind, label)


if __name__ == "__main__":
    unittest.main()
