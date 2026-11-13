#!/usr/bin/env python3
"""Secret scan gate: the one place that calls gitleaks.

Hooks, the CI workflow and the release steps call this file and nothing else. It checks every input
before it uses it, refuses on a finding and on every error, and prints no value, no commit message,
no author and no file, tag or ref name: only a rule, an identifier of a name, a line number and the
first eight characters of a commit.

Modes:
  pre-commit                       staged content (hook `pre-commit`)
  commit-msg <file>                the message file, unchanged (hook `commit-msg`)
  pre-push <remote> <url>          what a push brings, push lines on stdin (hook `pre-push`)
  full [--binary <path>]           every ref, every commit message, every annotated tag text
  locate <id>                      by hand only: resolves a name identifier (never run by a hook or CI)

Exit codes: 0 clean, 1 findings, 2 error (the message names mode, class and return value only).
Every child process runs with the time limit of its mode. Standard library only, Python 3.11 or newer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ".gitleaks-version"
CONFIG_FILE = ".gitleaks.toml"
IGNORE_FILE = ".gitleaksignore"
TEMPLATE_FILE = "scripts/gitleaks_findings.tmpl"

SHORT_TIMEOUT = 120
LONG_TIMEOUT = 900
SHORT_MODES = ("pre-commit", "commit-msg")
LABELS = ("pre-commit", "commit-msg", "pre-push", "full", "locate")
REPORT_KEYS = {"rule", "file", "line", "commit"}
OBJECT_TYPES = (b"commit", b"tag", b"tree", b"blob")
HEX_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
RULE_ID = re.compile(r"[A-Za-z0-9._-]{1,100}")
REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
VERSION_TEXT = re.compile(rb"[0-9]+\.[0-9]+\.[0-9]+\n")
NAME_ID = re.compile(r"[0-9a-f]{12}")
MAX_LINE = 10**9
ENV_DROPPED = ("GITLEAKS_CONFIG", "GITLEAKS_CONFIG_TOML")
REMEDY = (
    "install the gitleaks release named in .gitleaks-version and point the repository at it: "
    "git config hooks.gitleaks <absolute path to the program>"
)


class GateError(Exception):
    """A failure with a class name and, where there is one, a return value. Never carries an input."""

    def __init__(self, kind: str, rc: int | None = None):
        super().__init__(kind)
        self.kind = kind
        self.rc = rc


def has_control(text: str) -> bool:
    return any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in text)


def name_id(name: bytes) -> str:
    return hashlib.sha256(name).hexdigest()[:12]


def is_zero(sha: str) -> bool:
    return set(sha) == {"0"}


def clean_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in ENV_DROPPED}


def limit_of(mode: str) -> int:
    """The time limit of every child process of a mode, in seconds (read at call time)."""
    return SHORT_TIMEOUT if mode in SHORT_MODES else LONG_TIMEOUT


def run(command: list[str], kind: str, timeout: int, data: bytes | None = None, cwd: str | None = None,
        env: dict | None = None) -> subprocess.CompletedProcess:
    """The one way to start a child process: always with a time limit; a failure to start or a timeout is `kind`."""
    if not isinstance(timeout, int) or timeout <= 0:
        raise GateError("internal")
    try:
        return subprocess.run(
            command, input=data, capture_output=True, timeout=timeout, cwd=cwd, env=env,
            stdin=None if data is not None else subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        raise GateError(kind) from None


# ---------------------------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------------------------

def git_call(root: Path, args: list[str], timeout: int, data: bytes | None = None,
             ok: tuple[int, ...] = (0,)) -> tuple[int, bytes]:
    done = run(["git", "-C", str(root), *args], "git-error", timeout, data=data)
    if done.returncode not in ok:
        raise GateError("git-error", done.returncode)
    return done.returncode, done.stdout


def git(root: Path, *args: str, timeout: int, data: bytes | None = None, ok: tuple[int, ...] = (0,)) -> bytes:
    return git_call(root, list(args), timeout, data=data, ok=ok)[1]


def lines_of(out: bytes) -> list[bytes]:
    """Lines of a git output: empty output is no line; otherwise every line ends with LF and none is empty."""
    if out == b"":
        return []
    if not out.endswith(b"\n"):
        raise GateError("git-error")
    lines = out[:-1].split(b"\n")
    if any(not line for line in lines):
        raise GateError("git-error")
    return lines


def checked_id(raw: bytes) -> str:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise GateError("git-error") from None
    if not HEX_ID.fullmatch(text):
        raise GateError("git-error")
    return text


def id_lines(out: bytes) -> list[str]:
    """Lines that must each be an object identifier."""
    return [checked_id(raw) for raw in lines_of(out)]


def nul_names(out: bytes) -> list[bytes]:
    """Names of a `-z` output: each ends with NUL."""
    if out == b"":
        return []
    if not out.endswith(b"\0"):
        raise GateError("git-error")
    return out[:-1].split(b"\0")


def read_objects(root: Path, ids: list[str], timeout: int) -> dict[str, tuple[str, bytes]]:
    """Object type and content per identifier, in one `cat-file --batch` run; every answer is checked."""
    wanted = list(dict.fromkeys(ids))
    if not wanted:
        return {}
    out = git(root, "cat-file", "--batch", data=("\n".join(wanted) + "\n").encode("ascii"), timeout=timeout)
    result: dict[str, tuple[str, bytes]] = {}
    pos = 0
    for sha in wanted:
        end = out.find(b"\n", pos)
        if end < 0:
            raise GateError("git-error")
        head = out[pos:end].split(b" ")
        if len(head) != 3 or head[0] != sha.encode("ascii") or head[1] not in OBJECT_TYPES:
            raise GateError("git-error")
        if not (head[2].isdigit() and (head[2] == b"0" or not head[2].startswith(b"0"))):
            raise GateError("git-error")
        start = end + 1
        stop = start + int(head[2])
        if stop >= len(out) or out[stop:stop + 1] != b"\n":
            raise GateError("git-error")
        result[sha] = (head[1].decode("ascii"), out[start:stop])
        pos = stop + 1
    if pos != len(out):
        raise GateError("git-error")
    return result


def message_of(content: bytes) -> bytes:
    """The message of a commit or tag object, read as git's own object check reads it (`verify_headers`).

    The header ends at the first blank line and the message follows it. Without a blank line the object has no
    message if its last header line ends with LF; without that LF the header is unterminated, an error.
    """
    cut = content.find(b"\n\n")
    if cut >= 0:
        return content[cut + 2:]
    if content.endswith(b"\n"):
        return b""
    raise GateError("git-error")


def header_of(content: bytes) -> list[bytes]:
    """The header lines of a commit or tag object (everything before the message)."""
    cut = content.find(b"\n\n")
    return (content if cut < 0 else content[:cut]).split(b"\n")


def tag_target(content: bytes) -> str:
    """The object a tag points at. git reads its id in either case, so the id is taken in lower case."""
    first = content.split(b"\n", 1)[0]
    if not first.startswith(b"object "):
        raise GateError("git-error")
    return checked_id(first[len(b"object "):].lower())


def carried_name(content: bytes) -> bytes | None:
    """The name a tag object carries in its `tag` header line, if it has one."""
    for line in header_of(content):
        if line.startswith(b"tag ") and len(line) > len(b"tag "):
            return line[len(b"tag "):]
    return None


def walk_tags(root: Path, starts: list[tuple[bytes, str]], timeout: int) -> tuple[list[tuple[bytes, str, str]], dict[str, str]]:
    """Follows each (name, object id) through annotated tags, nested ones included.

    Returns the text of every tag object met as a scan unit, named after its start, and per start object the type
    of the object at the end of the chain. A tag object is known by its type, never by the name it was pushed as.
    """
    units: list[tuple[bytes, str, str]] = []
    ends: dict[str, str] = {}
    level = [(name, sha, sha) for name, sha in starts]
    seen: set[tuple[bytes, str]] = set()
    while level:
        objects = read_objects(root, [current for _, _, current in level], timeout)
        following = []
        for name, origin, current in level:
            kind, content = objects[current]
            if kind != "tag":
                ends[origin] = kind
                continue
            if (name, current) in seen:
                continue
            seen.add((name, current))
            units.append((message_of(content), f"tag:{name_id(name)}", "-"))
            following.append((name, origin, tag_target(content)))
        level = following
    return units, ends


def commit_units(root: Path, shas: list[str], timeout: int) -> list[tuple[bytes, str, str]]:
    objects = read_objects(root, shas, timeout)
    units = []
    for sha in shas:
        kind, content = objects[sha]
        if kind != "commit":
            raise GateError("git-error")
        units.append((message_of(content), "", sha[:8]))
    return units


# ---------------------------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------------------------

def read_version(root: Path) -> bytes:
    """The bytes of `.gitleaks-version`: exactly `<a>.<b>.<c>` with digits and one LF."""
    path = root / VERSION_FILE
    if not path.is_file():
        raise GateError("version-file")
    try:
        raw = path.read_bytes()
    except OSError:
        raise GateError("version-file") from None
    if len(raw) > 64 or not VERSION_TEXT.fullmatch(raw):
        raise GateError("version-file")
    return raw


def check_ignore_file(root: Path) -> None:
    """gitleaks only warns about a bad line of its ignore file, so the gate reads it first."""
    path = root / IGNORE_FILE
    if not path.is_file():
        raise GateError("ignore-file")
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise GateError("ignore-file") from None
    if text.startswith("﻿") or "\r" in text:
        raise GateError("ignore-file")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    seen = set()
    for line in lines:
        if line == "" or line.startswith("#"):
            continue
        if line in seen:
            raise GateError("ignore-file")
        seen.add(line)
        head, sep1, line_no = line.rpartition(":")
        head, sep2, rule = head.rpartition(":")
        commit, sep3, file_part = head.partition(":")
        if not (sep1 and sep2 and sep3):
            raise GateError("ignore-file")
        if not (HEX_ID.fullmatch(commit) and file_part and RULE_ID.fullmatch(rule)):
            raise GateError("ignore-file")
        if not (line_no.isascii() and line_no.isdigit() and int(line_no) >= 1):
            raise GateError("ignore-file")


def check_files(root: Path) -> None:
    if not (root / CONFIG_FILE).is_file() or not (root / TEMPLATE_FILE).is_file():
        raise GateError("config-file")
    check_ignore_file(root)


def checked_path(raw: bytes) -> str:
    """The value of a path setting: exactly one value, no control character, absolute, an existing file."""
    if not raw.endswith(b"\n"):
        raise GateError("binary-path")
    value = raw[:-1]
    if b"\n" in value:
        raise GateError("binary-path")
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        raise GateError("binary-path") from None
    if not text or has_control(text) or not os.path.isabs(text) or not os.path.isfile(text):
        raise GateError("binary-path")
    return text


def find_binary(root: Path, cli_binary: str | None, timeout: int) -> str:
    if cli_binary is not None:
        if "\n" in cli_binary:
            raise GateError("binary-path")
        return checked_path(cli_binary.encode("utf-8", "surrogateescape") + b"\n")
    rc, out = git_call(root, ["config", "--get-all", "hooks.gitleaks"], timeout, ok=(0, 1))
    if rc == 0:
        return checked_path(out)
    found = shutil.which("gitleaks")
    if found is None:
        raise GateError("binary-path")
    return checked_path(os.fsencode(os.path.abspath(found)) + b"\n")


def check_binary(binary: str, version: bytes, timeout: int) -> None:
    """Found is only a program whose `gitleaks version` output is exactly the content of `.gitleaks-version`."""
    done = run([binary, "version"], "binary-version", timeout, env=clean_env())
    if done.returncode != 0 or done.stdout != version:
        raise GateError("binary-version", done.returncode)


def prepare(root: Path, timeout: int, cli_binary: str | None = None) -> str:
    version = read_version(root)
    check_files(root)
    binary = find_binary(root, cli_binary, timeout)
    check_binary(binary, version, timeout)
    return binary


# ---------------------------------------------------------------------------------------------
# gitleaks
# ---------------------------------------------------------------------------------------------

def _pairs(pairs):
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise ValueError
        out[key] = value
    return out


def _constant(_name):
    raise ValueError


def parse_report(out: bytes, allow_empty_commit: bool, stdin_mode: bool) -> list[dict]:
    try:
        text = out.decode("utf-8")
    except UnicodeDecodeError:
        raise GateError("report-format") from None
    if not text.endswith("\n"):
        raise GateError("report-format")
    findings = []
    for line in text[:-1].split("\n"):
        try:
            item = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
        except ValueError:
            raise GateError("report-format") from None
        if not isinstance(item, dict) or set(item) != REPORT_KEYS:
            raise GateError("report-format")
        rule, file, number, commit = item["rule"], item["file"], item["line"], item["commit"]
        if not isinstance(rule, str) or not RULE_ID.fullmatch(rule):
            raise GateError("report-format")
        if not isinstance(file, str) or (file == "" and not stdin_mode):
            raise GateError("report-format")
        try:
            file.encode("utf-8")
        except UnicodeEncodeError:  # a lone surrogate: no name git can hold
            raise GateError("report-format") from None
        if type(number) is not int or number < 1 or number > MAX_LINE:
            raise GateError("report-format")
        if not isinstance(commit, str) or not (commit == "" or HEX_ID.fullmatch(commit)):
            raise GateError("report-format")
        if commit == "" and not (allow_empty_commit or stdin_mode):
            raise GateError("report-format")
        findings.append(item)
    return findings


def run_gitleaks(binary: str, root: Path, head: list[str], data: bytes | None, timeout: int,
                 allow_empty_commit: bool = False) -> list[dict]:
    stdin_mode = data is not None
    command = [
        binary, *head,
        "--config", str(root / CONFIG_FILE),
        "--gitleaks-ignore-path", str(root / IGNORE_FILE),
        "--ignore-gitleaks-allow", "--redact", "--no-banner", "--no-color",
        "--exit-code", "3", "--max-decode-depth", "5",
        "--report-format", "template", "--report-template", str(root / TEMPLATE_FILE),
        "--report-path", "-",
    ]
    done = run(command, "scan-error", timeout, data=data, cwd=str(root), env=clean_env())
    if done.returncode == 0 and done.stdout == b"":
        return []
    if done.returncode not in (0, 3):
        raise GateError("scan-error", done.returncode)
    if done.returncode == 3 and done.stdout == b"":
        raise GateError("scan-error", done.returncode)
    if done.returncode == 0:
        raise GateError("scan-error", done.returncode)
    return parse_report(done.stdout, allow_empty_commit, stdin_mode)


Finding = tuple[str, str, str]


def lossy(raw: bytes) -> str:
    """A name as gitleaks reports it: every byte that is not part of a valid UTF-8 sequence becomes U+FFFD."""
    out = []
    pos = 0
    while pos < len(raw):
        for size in (1, 2, 3, 4):
            try:
                out.append(raw[pos:pos + size].decode("utf-8"))
            except UnicodeDecodeError:
                continue
            pos += size
            break
        else:
            out.append("�")
            pos += 1
    return "".join(out)


def raw_names(root: Path, item: dict, timeout: int, cache: dict[str, list[bytes]]) -> list[bytes]:
    """The raw bytes of a reported file name.

    gitleaks reports a name that is not UTF-8 with U+FFFD in place of each bad byte, so the bytes come back from git:
    the names of the commit's tree (history) or of the index (staged), whose reported form is the same.
    """
    name = item["file"]
    if "�" not in name:
        return [name.encode("utf-8")]
    commit = item["commit"]
    if commit not in cache:
        if commit:
            out = git(root, "ls-tree", "-r", "-z", "--name-only", "--full-tree", commit, timeout=timeout)
        else:
            out = git(root, "ls-files", "-z", timeout=timeout)
        cache[commit] = nul_names(out)
    hits = sorted({raw for raw in cache[commit] if lossy(raw) == name})
    if not hits:
        raise GateError("git-error")
    return hits


def file_findings(root: Path, items: list[dict], timeout: int) -> set[Finding]:
    out = set()
    cache: dict[str, list[bytes]] = {}
    for item in items:
        commit = item["commit"][:8] if item["commit"] else "staged"
        for raw in raw_names(root, item, timeout, cache):
            out.add((item["rule"], f"file:{name_id(raw)}:{item['line']}", commit))
    return out


def scan_messages(binary: str, root: Path, units: list[tuple[bytes, str, str]], timeout: int) -> set[Finding]:
    """One gitleaks run per message; `units` are (bytes, location prefix, reference)."""
    work = [unit for unit in units if unit[0]]

    def one(unit: tuple[bytes, str, str]) -> set[Finding]:
        data, prefix, ref = unit
        items = run_gitleaks(binary, root, ["stdin"], data, timeout)
        return {(item["rule"], f"{prefix}:{item['line']}" if prefix else f"message:{item['line']}", ref) for item in items}

    found: set[Finding] = set()
    if not work:
        return found
    with ThreadPoolExecutor(max_workers=min(8, len(work))) as pool:
        for result in pool.map(one, work):
            found |= result
    return found


# ---------------------------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------------------------

def mode_pre_commit(root: Path, args: list[str], timeout: int) -> set[Finding]:
    if args:
        raise GateError("usage")
    binary = prepare(root, timeout)
    items = run_gitleaks(binary, root, ["git", "--pre-commit", "--staged", str(root)], None, timeout,
                         allow_empty_commit=True)
    return file_findings(root, items, timeout)


def mode_commit_msg(root: Path, args: list[str], timeout: int) -> set[Finding]:
    if len(args) != 1 or args[0] == "--binary":
        raise GateError("usage")
    binary = prepare(root, timeout)
    if not os.path.isfile(args[0]):
        raise GateError("message-file")
    try:
        data = Path(args[0]).read_bytes()
    except OSError:
        raise GateError("message-file") from None
    return scan_messages(binary, root, [(data, "", "staged")], timeout)


def parse_push(raw: bytes) -> list[tuple[str, str, str, str]]:
    if raw == b"":
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise GateError("push-input") from None
    if "\r" in text or not text.endswith("\n"):
        raise GateError("push-input")
    entries = []
    seen = set()
    for line in text[:-1].split("\n"):
        fields = line.split(" ")
        if len(fields) != 4 or any(not f for f in fields) or has_control(line):
            raise GateError("push-input")
        local_ref, local_sha, remote_ref, remote_sha = fields
        if not (HEX_ID.fullmatch(local_sha) and HEX_ID.fullmatch(remote_sha)):
            raise GateError("push-input")
        if line in seen:
            raise GateError("push-input")
        seen.add(line)
        entries.append((local_ref, local_sha, remote_ref, remote_sha))
    return entries


def short_ref(ref: str) -> bytes:
    """A ref name as the gate reports it: a tag by its short name, every other ref by its full name."""
    if ref.startswith("refs/tags/") and len(ref) > len("refs/tags/"):
        return ref[len("refs/tags/"):].encode("utf-8")
    return ref.encode("utf-8")


def push_name(local_ref: str, remote_ref: str, carried: bytes | None) -> bytes:
    """The name a pushed tag text is reported under, a name `locate` finds in this clone.

    The local ref, if the push names one; else, for an object pushed by its id, the name its tag object carries;
    only a tag object without that header line falls back to the target ref.
    """
    if local_ref.startswith("refs/"):
        return short_ref(local_ref)
    if carried is not None:
        return carried
    return short_ref(remote_ref)


def mode_pre_push(root: Path, args: list[str], timeout: int) -> set[Finding]:
    if len(args) != 2 or "--binary" in args:
        raise GateError("usage")
    remote = args[0]
    try:
        raw = sys.stdin.buffer.read()
    except (AttributeError, OSError, ValueError):
        raise GateError("push-input") from None
    entries = parse_push(raw)
    binary = prepare(root, timeout)
    by_id = [sha for local_ref, sha, _, _ in entries if not is_zero(sha) and not local_ref.startswith("refs/")]
    carried = {sha: carried_name(content) if kind == "tag" else None
               for sha, (kind, content) in read_objects(root, by_id, timeout).items()}
    positives: list[str] = []
    starts: list[tuple[bytes, str]] = []
    for local_ref, local_sha, remote_ref, _ in entries:
        if is_zero(local_sha):
            continue
        if local_sha not in positives:
            positives.append(local_sha)
        starts.append((push_name(local_ref, remote_ref, carried.get(local_sha)), local_sha))
    if not positives:
        return set()
    tag_texts, ends = walk_tags(root, starts, timeout)
    if any(kind != "commit" for kind in ends.values()):
        raise GateError("git-error")  # a tree or blob pushed on its own: its content is not in any commit
    excludes: list[str] = []
    if REMOTE_NAME.fullmatch(remote):
        refs = git(root, "for-each-ref", "--format=%(objectname)", f"refs/remotes/{remote}/", timeout=timeout)
        excludes += [sha for sha in id_lines(refs) if sha not in excludes]
    for _, _, _, remote_sha in entries:
        if not is_zero(remote_sha) and remote_sha not in excludes:
            rc, _ = git_call(root, ["cat-file", "-e", remote_sha], timeout, ok=(0, 1))
            if rc == 0:
                excludes.append(remote_sha)
    options = ["--diff-merges=separate", *positives, *[f"^{sha}" for sha in excludes]]
    items = run_gitleaks(binary, root, ["git", "--log-opts=" + " ".join(options), str(root)], None, timeout)
    found = file_findings(root, items, timeout)
    shas = id_lines(git(root, "rev-list", *positives, *[f"^{sha}" for sha in excludes], timeout=timeout))
    units = commit_units(root, shas, timeout) + tag_texts
    return found | scan_messages(binary, root, units, timeout)


def ref_tags(root: Path, timeout: int) -> list[tuple[bytes, str]]:
    """Every ref that points at an annotated tag object, as (name, object id); every line of git is checked."""
    out = git(root, "for-each-ref", "--format=%(objecttype) %(objectname) %(refname)", timeout=timeout)
    starts = []
    for raw in lines_of(out):
        parts = raw.split(b" ")
        if len(parts) != 3 or parts[0] not in OBJECT_TYPES or not parts[2].startswith(b"refs/"):
            raise GateError("git-error")
        sha = checked_id(parts[1])
        try:
            parts[2].decode("utf-8")
        except UnicodeDecodeError:
            raise GateError("git-error") from None
        if parts[0] in (b"tree", b"blob"):
            raise GateError("git-error")  # a tree or blob a ref names on its own: its content is not in any commit
        if parts[0] != b"tag":  # a ref to a commit has no text of its own
            continue
        ref = parts[2]
        starts.append((ref[len(b"refs/tags/"):] if ref.startswith(b"refs/tags/") else ref, sha))
    return starts


def mode_full(root: Path, args: list[str], timeout: int) -> set[Finding]:
    cli_binary = None
    if args:
        if len(args) != 2 or args[0] != "--binary":
            raise GateError("usage")
        cli_binary = args[1]
    binary = prepare(root, timeout, cli_binary)
    items = run_gitleaks(binary, root, ["git", "--log-opts=--all --diff-merges=separate", str(root)], None, timeout)
    found = file_findings(root, items, timeout)
    shas = id_lines(git(root, "rev-list", "--all", timeout=timeout))
    units = commit_units(root, shas, timeout)
    tag_texts, ends = walk_tags(root, ref_tags(root, timeout), timeout)
    if any(kind != "commit" for kind in ends.values()):
        raise GateError("git-error")  # a tag chain that ends at a tree or blob: its content is not in any commit
    return found | scan_messages(binary, root, units + tag_texts, timeout)


def mode_locate(root: Path, args: list[str], timeout: int) -> int:
    if len(args) != 1 or not NAME_ID.fullmatch(args[0]):
        raise GateError("usage")
    wanted = args[0]
    names: set[bytes] = set(nul_names(git(root, "ls-files", "-z", timeout=timeout)))
    # Every commit and tag object of this clone, also those no ref reaches any more (a push by object id).
    listed = git(root, "cat-file", "--batch-all-objects", "--batch-check=%(objecttype) %(objectname)", timeout=timeout)
    commits: list[str] = []
    tags: list[str] = []
    for raw in lines_of(listed):
        parts = raw.split(b" ")
        if len(parts) != 2 or parts[0] not in OBJECT_TYPES:
            raise GateError("git-error")
        sha = checked_id(parts[1])
        if parts[0] == b"commit":
            commits.append(sha)
        elif parts[0] == b"tag":
            tags.append(sha)
    if commits:  # a clone without a commit has no history to read
        log = git(root, "log", "--all", "--stdin", "-m", "--no-renames", "--name-only", "--format=", "-z",
                  data="".join(sha + "\n" for sha in commits).encode("ascii"), timeout=timeout)
        names.update(chunk for chunk in log.split(b"\0") if chunk)
    for kind, content in read_objects(root, tags, timeout).values():
        name = carried_name(content)
        if name is not None:
            names.add(name)
    for raw in lines_of(git(root, "for-each-ref", "--format=%(refname)", timeout=timeout)):
        names.add(raw[len(b"refs/tags/"):] if raw.startswith(b"refs/tags/") else raw)
    hits = sorted(name for name in names if name and name_id(name) == wanted)
    for name in hits:
        sys.stdout.buffer.write(name.decode("utf-8", "backslashreplace").encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    return 0 if hits else 1


MODES = {
    "pre-commit": mode_pre_commit,
    "commit-msg": mode_commit_msg,
    "pre-push": mode_pre_push,
    "full": mode_full,
}


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else ""
    label = mode if mode in LABELS else "-"  # never an unknown argument: it may be a path or a value
    try:
        timeout = limit_of(mode)
        if mode == "locate":
            return mode_locate(ROOT, argv[1:], timeout)
        if mode not in MODES:
            raise GateError("usage")
        findings = MODES[mode](ROOT, argv[1:], timeout)
    except GateError as error:
        detail = f" rc={error.rc}" if error.rc is not None else ""
        print(f"gitleaks_gate: {label}: error {error.kind}{detail}", file=sys.stderr)
        if error.kind in ("binary-path", "binary-version"):
            print(f"gitleaks_gate: {REMEDY}", file=sys.stderr)
        return 2
    except Exception:  # no message of the exception: it may name a path or a value
        print(f"gitleaks_gate: {label}: error internal", file=sys.stderr)
        return 2
    for rule, location, reference in sorted(findings):
        print(f"{rule} {location} {reference}")
    if findings:
        print(f"gitleaks_gate: {label}: {len(findings)} finding(s), refused", file=sys.stderr)
        return 1
    print(f"gitleaks_gate: {label}: clean", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
