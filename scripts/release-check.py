"""Release gate: is this state of the repository fit to be published?

A pure checker: it writes nothing into the repository and builds offline from the uv cache. It
reports every finding, never stops at the first, and names each finding by place and class
only, never by the text it found (a leak must not spread into logs): a file name, tag name or
version that itself matches the negative list is withheld, and no error text from git or the
build is passed through. Exit codes: 0 free to release, 1 findings, 2 usage or environment
error (findings collected before the error are still printed).

    uv run --frozen python scripts/release-check.py [--tag vX.Y.Z] [--repo PATH]

Checks, in this order:
1. the working tree is clean, `pyproject.toml` carries a valid PEP 440 version without a local
   part, `CHANGELOG.md` has a section for exactly that version; with `--tag`, the tag exists, is
   annotated, points at HEAD and is named `v` + version;
2. the negative list of `tests/test_public_hygiene.py` (the same module, not a copy) over every
   file of HEAD, over every line ever added in the history of every ref (binary files as text,
   renames as additions, merges against each parent), over every file name any commit held and
   over the raw commit and tag objects (every header, signatures included, every tag in a
   chain); replace refs are ignored and a shallow clone is refused. The private words of the
   file `MODELROOM_RELEASE_PRIVATE_WORDS` names (outside the repository, one per line, any case,
   blank lines and `#` lines left out) are the class `private word`; they, the device subsystem
   id and the processor model are checked in HEAD, the distributions and the metadata, not in the
   earlier file contents of the history (known and accepted, decided 2026-09-28), and so is a
   global IPv4 address. Without that file the gate ends with a usage error after every other check
   has reported, and every name that comes from the data is withheld. Every file is read as UTF-8,
   as UTF-16 where it says so and with its JSON escapes resolved; a binary file (a skipped suffix) in
   its printable runs, in every class. The metadata are read in their declared `encoding` (an
   unknown one is a finding); the allowed addresses count for the class `e-mail address` only;
3. `uv build --offline` of HEAD (exported with `git archive`, so the working tree cannot leak
   in); the wheel and the sdist are listed and checked against the negative list and against a
   short list of file classes that must never ship, every entry name and the archives' own
   metadata (comments, extra fields, link targets, owners, pax headers) included. The wheel must
   carry `modelroom/` and the `modelroom` console entry. An error of any kind is reported by its
   class name only.
"""

from __future__ import annotations

import argparse
import codecs
import configparser
import importlib.util
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
HYGIENE_MODULE = REPO / "tests" / "test_public_hygiene.py"
# PEP 440, appendix B, split into the public part and the local part (the package index refuses
# a local part, so a release must not carry one).
_PUBLIC_VERSION = (
    r"v?(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*(?:[-_.]?(?:a|b|c|rc|alpha|beta|pre|preview)[-_.]?[0-9]*)?"
    r"(?:-[0-9]+|[-_.]?(?:post|rev|r)[-_.]?[0-9]*)?(?:[-_.]?dev[-_.]?[0-9]*)?"
)
VERSION = re.compile(rf"^(?P<public>{_PUBLIC_VERSION})(?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?$", re.IGNORECASE)
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
PERSON = re.compile(r"^(.*) <(.*)> (-?\d+) ([+-]\d{4})$")
PERSON_HEADERS = {"author", "committer", "tagger"}
STATE_FILES = {"modelroom.lock", "run-status.json", "modelroom.json"}
WITHHELD = "<withheld>"
# Commit and tag metadata only: addresses that identify no one, plus the maintainer's own address
# (see `maintainer_email`). Each is removed before the negative list runs, and only as a whole
# address (a longer domain is not covered), so anything else on the same line is still checked.
# The maintainer's address is never written into this repository.
ALLOWED_META = r"[\w.+-]+@users\.noreply\.github\.com|noreply@anthropic\.com"
MAINTAINER_ENV = "MODELROOM_RELEASE_MAINTAINER_EMAIL"
# A text file outside the repository, one word per line: the maintainer's own words that must never
# be published (machine names and the like). The words are never written into this repository and
# never printed; a finding names place and class only. Without the file the gate cannot pass.
PRIVATE_WORDS_ENV = "MODELROOM_RELEASE_PRIVATE_WORDS"
PRIVATE_WORD = "private word"
# Classes checked in the current state only -- HEAD, the distributions and the metadata -- and not
# in the file contents of the history, whose earlier contents are known and accepted (decided
# 2026-09-28: they are cleaned in the current state, the history is not rewritten).
CURRENT_STATE_CLASSES = {PRIVATE_WORD, "hardware subsystem id", "processor model", "global IPv4 address"}
EMAIL_CLASS = "e-mail address"
# Single metadata findings that were read and accepted, each bound to (full commit hash, field,
# line, class): any other line, field, commit or class stays a finding.
ALLOWED_META_FINDINGS = {
    # accepted: names the internal workspace only, no path or data; decided 2026-09-23
    ("e87a208f07a01632d918fe0672f136a41da3a06b", "message", 18, "internal workspace name"),
}


class UsageError(Exception):
    """The check cannot run at all: not a repository, git or uv missing, the build failed."""


class NegativeList(NamedTuple):
    """The checks of `tests/test_public_hygiene.py`: of one line (`forbidden`, patterns and the global
    address) and of a whole document (`documents`, the processor). `redact_names`: every name that
    comes from the data is withheld, because a check (the private words) could not be loaded."""

    forbidden: dict[str, Callable[[str], object]]
    allowed: dict[tuple[str, str], str]
    skip_suffixes: set[str]
    documents: dict[str, Callable[[str], list[int]]] = {}
    redact_names: bool = False
    # The private words as byte sequences in UTF-8, UTF-16 LE/BE and latin-1 (Codex r4-1): found in
    # any file as they are stored, short and non-ASCII words included, without a text reading.
    word_bytes: re.Pattern[bytes] | None = None


def load_negative_list() -> NegativeList:
    """The negative list lives once, in tests/test_public_hygiene.py; this imports that module."""
    spec = importlib.util.spec_from_file_location("public_hygiene", HYGIENE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return NegativeList(module.LINE_CHECKS, module.ALLOWED, module.SKIP_SUFFIXES, module.DOCUMENT_CHECKS)


def private_words_pattern(repo: Path = REPO) -> re.Pattern[str]:
    """The words of `private_words` as one pattern, any case."""
    words = private_words(repo)
    if not words:
        return re.compile(r"(?!x)x")  # no word: a pattern that finds nothing
    return re.compile("|".join(re.escape(word) for word in words), re.IGNORECASE)


WORD_ENCODINGS = ("utf-8", "utf-16-le", "utf-16-be", "latin-1")


def private_word_bytes(words: list[str]) -> re.Pattern[bytes] | None:
    """Every word as the bytes it is stored as in each of `WORD_ENCODINGS`, in its cases."""
    forms = set()
    for word in words:
        for variant in {word, word.lower(), word.upper(), word.capitalize()}:
            for encoding in WORD_ENCODINGS:
                try:
                    forms.add(variant.encode(encoding))
                except UnicodeEncodeError:  # a word latin-1 cannot hold is not stored as latin-1 either
                    continue
    if not forms:
        return None
    return re.compile(b"|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True)), re.IGNORECASE)


def private_words(repo: Path = REPO) -> list[str]:
    """The words of the file `MODELROOM_RELEASE_PRIVATE_WORDS` names, the longest first.

    Blank lines and lines starting with `#` are left out; a byte order mark is not part of the first
    word (`utf-8-sig`). An unset variable, a file that cannot be read, or a file inside `repo` (it
    would publish the words it guards) is a usage error: the gate would otherwise pass without the
    check. The message names neither the file nor a word.
    """
    location = os.environ.get(PRIVATE_WORDS_ENV, "").strip()
    if not location:
        raise UsageError(f"{PRIVATE_WORDS_ENV} is not set: it names the file of private words")
    path = Path(location)
    try:
        if path.resolve().is_relative_to(repo.resolve()):
            raise UsageError(f"the file {PRIVATE_WORDS_ENV} names lies inside the repository; keep it outside")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise UsageError(f"the file {PRIVATE_WORDS_ENV} names cannot be read") from exc
    return sorted({line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}, key=len, reverse=True)


def with_private_words(lists: NegativeList, words: re.Pattern[str] | list[str] | None) -> NegativeList:
    """The list with the private words -- as a pattern for the text readings and, given as a list,
    as byte sequences for every file; without them (`None`) every name from the data is withheld."""
    if words is None:
        return lists._replace(redact_names=True)
    if isinstance(words, re.Pattern):
        return lists._replace(forbidden={**lists.forbidden, PRIVATE_WORD: words.search})
    pattern = re.compile("|".join(re.escape(word) for word in words), re.IGNORECASE) if words else re.compile(r"(?!x)x")
    return lists._replace(forbidden={**lists.forbidden, PRIVATE_WORD: pattern.search}, word_bytes=private_word_bytes(words))


def current_state_only(lists: NegativeList) -> NegativeList:
    """The list for the file contents of the history: without the classes of the current state."""
    return lists._replace(
        forbidden={name: check for name, check in lists.forbidden.items() if name not in CURRENT_STATE_CLASSES},
        documents={name: check for name, check in lists.documents.items() if name not in CURRENT_STATE_CLASSES},
    )


def git(repo: Path, *args: str, check: bool = True, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run git; a failure is reported by command and exit code only, never by git's own text."""
    try:
        # --no-replace-objects: a local replace ref must not hide what the published history holds.
        result = subprocess.run(["git", "--no-replace-objects", "-c", "core.quotePath=false", *args], cwd=repo,
                                input=stdin, capture_output=True)
    except FileNotFoundError as exc:
        raise UsageError("git is not on PATH") from exc
    if check and result.returncode != 0:
        raise UsageError(f"git {args[0]} failed (exit {result.returncode})")
    return result


def git_text(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.decode("utf-8", "replace")


def ensure_repository(repo: Path) -> None:
    """The top level of a full clone: a shallow clone hides the history this check must read."""
    result = git(repo, "rev-parse", "--show-toplevel", check=False)
    toplevel = result.stdout.decode("utf-8", "replace").strip()
    if result.returncode != 0 or Path(toplevel).resolve() != repo.resolve():
        raise UsageError("--repo is not the top level of a git repository")
    if git_text(repo, "rev-parse", "--is-shallow-repository").strip() != "false":
        raise UsageError("shallow clone: the full history is needed (git fetch --unshallow)")


EmailView = Callable[[str], str] | None


def _line_classes(line: str, lists: NegativeList, path: str, email_view: EmailView = None) -> list[str]:
    """The classes of one line. `email_view` (metadata only) removes the allowed addresses -- for the
    class `e-mail address` alone; every other class reads the line as it is (Codex r3-8)."""
    found = []
    for name, check in lists.forbidden.items():
        if (path, name) in lists.allowed:
            continue
        seen = email_view(line) if email_view is not None and name == EMAIL_CLASS else line
        if check(seen):
            found.append(name)
    return found


def _document_hits(text: str, lists: NegativeList, path: str) -> list[tuple[int, str]]:
    return [(line_no, name) for name, lines_of in lists.documents.items()
            if (path, name) not in lists.allowed for line_no in lines_of(text)]


def leak_classes(text: str, lists: NegativeList, path: str = "", email_view: EmailView = None) -> list[str]:
    """Every class `text` holds, line checks and document checks alike."""
    names = _line_classes(text, lists, path, email_view)
    return names + [name for _line, name in _document_hits(text, lists, path) if name not in names]


def scan_text(place: str, path: str, text: str, lists: NegativeList, email_view: EmailView = None) -> list[str]:
    """Every line of `text` against every class; `place` prefixes the finding, never the text."""
    findings: dict[str, None] = {}
    for line_no, line in enumerate(text.split("\n"), start=1):
        findings.update(dict.fromkeys(f"{place}:{line_no}: {name}" for name in _line_classes(line, lists, path, email_view)))
    findings.update(dict.fromkeys(f"{place}:{line_no}: {name}" for line_no, name in _document_hits(text, lists, path)))
    return list(findings)


_JSON_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def unescaped(text: str) -> list[str]:
    """`text`, and where it holds JSON `\\uXXXX` escapes, the text with them resolved."""
    return [text, _JSON_ESCAPE.sub(lambda m: chr(int(m[1], 16)), text)] if _JSON_ESCAPE.search(text) else [text]


_PRINTABLE_RUN = re.compile(r"[\x20-\x7e]{6,}")


def printable_runs(text: str) -> str:
    """The runs of six printable characters or more, one per line -- what `strings` reads out of a
    binary file. Random bytes between them are no text and would only make noise (an `@` and a `.`)."""
    return "\n".join(_PRINTABLE_RUN.findall(text))


def text_variants(data: bytes, binary: bool = False) -> list[str]:
    """Every reading of `data` a leak could hide in: UTF-8; UTF-16 where a byte order mark or a share
    of NUL bytes says so; each with its JSON escapes resolved. A binary file (a skipped suffix) is read
    in its printable runs, as UTF-8, byte for byte (latin-1) and as UTF-16 both ways."""
    if binary:
        readings = [data.decode(encoding, "replace") for encoding in ("utf-8", "latin-1", "utf-16-le", "utf-16-be")]
        return [reading for text in readings for reading in unescaped(printable_runs(text))]
    variants = [data.decode("utf-8", "replace")]
    if any(byte >= 0x80 for byte in data):  # a latin-1 text reads otherwise than UTF-8 (Codex r4-1)
        variants.append(data.decode("latin-1"))
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        variants.append(data.decode("utf-16", "replace"))
    elif len(data) >= 4 and data.count(b"\x00") * 4 >= len(data):
        variants += [data.decode("utf-16-le", "replace"), data.decode("utf-16-be", "replace")]
    return [reading for text in variants for reading in unescaped(text)]


def word_byte_findings(place: str, data: bytes, lists: NegativeList) -> list[str]:
    """The private words as the bytes they are stored as (`NegativeList.word_bytes`), at their line."""
    if PRIVATE_WORD not in lists.forbidden:
        return []
    lines = set()
    if lists.word_bytes is not None:
        lines |= {data.count(b"\n", 0, match.start()) + 1 for match in lists.word_bytes.finditer(data)}
    # And as text, the whole decoded reading in each encoding (UTF-16 also from the odd byte), with a
    # Unicode case-insensitive match: `Müller-SÜD` for the word `müller-süd` (Codex r5-1).
    found = lists.forbidden[PRIVATE_WORD]
    for encoding, start in (("utf-8", 0), ("latin-1", 0), ("utf-16-le", 0), ("utf-16-be", 0), ("utf-16-le", 1), ("utf-16-be", 1)):
        text = data[start:].decode(encoding, "replace")
        lines |= {line_no for line_no, line in enumerate(text.split("\n"), start=1) if found(line)}
    return [f"{place}:{line}: {PRIVATE_WORD}" for line in sorted(lines)]


def scan_data(place: str, path: str, data: bytes, lists: NegativeList) -> list[str]:
    """`scan_text` over every reading of a file (`text_variants`), every class, binary files included
    (a skipped suffix is read byte for byte too, Codex r3-6), and the private words as byte sequences
    in every file, whatever its reading (Codex r4-1)."""
    found: dict[str, None] = {}
    for text in text_variants(data, binary=skipped(path, lists)):
        found.update(dict.fromkeys(scan_text(place, path, text, lists)))
    found.update(dict.fromkeys(word_byte_findings(place, data, lists)))
    return list(found)


def shown(name: str, lists: NegativeList) -> str:
    """A file, tag, field or ref name as it may be printed: withheld when it matches the negative list,
    and always where the private words could not be loaded (Codex r3-14)."""
    return WITHHELD if lists.redact_names or any(leak_classes(text, lists) for text in unescaped(name)) else name


def name_findings(prefix: str, name: str, lists: NegativeList, kind: str = "file") -> list[str]:
    classes = dict.fromkeys(cls for text in unescaped(name) for cls in leak_classes(text, lists))
    return [f"{prefix}:{WITHHELD}: {cls} in {kind} name" for cls in classes]


def skipped(path: str, lists: NegativeList) -> bool:
    return PurePosixPath(path).suffix in lists.skip_suffixes


# -- 1: working tree, version, changelog, tag ---------------------------------------------------


def worktree_findings(repo: Path) -> list[str]:
    entries = [line for line in git_text(repo, "status", "--porcelain", "--untracked-files=all").splitlines() if line]
    return [f"worktree: not clean ({len(entries)} entries)"] if entries else []


def ref_version(repo: Path, ref: str) -> str:
    result = git(repo, "show", f"{ref}:pyproject.toml", check=False)
    if result.returncode != 0:
        raise UsageError(f"{ref} has no pyproject.toml")
    try:
        # PEP 440: leading and trailing whitespace is ignored.
        return str(tomllib.loads(result.stdout.decode("utf-8"))["project"]["version"]).strip()
    except (tomllib.TOMLDecodeError, KeyError, UnicodeDecodeError) as exc:
        raise UsageError(f"{ref}:pyproject.toml has no readable [project] version") from exc


def printable_version(version: str, lists: NegativeList | None = None) -> str:
    match = VERSION.match(version)
    if lists is not None and (lists.redact_names or leak_classes(version, lists)):
        return WITHHELD
    return version if match and not match["local"] else WITHHELD


def version_findings(repo: Path, ref: str, lists: NegativeList | None = None) -> list[str]:
    version = ref_version(repo, ref)
    match = VERSION.match(version)
    findings = []
    if not match:
        findings.append("ref:pyproject.toml: version is not a valid PEP 440 version")
    elif match["local"]:
        findings.append("ref:pyproject.toml: local version (+...) is not accepted by the package index")
    changelog = git(repo, "show", f"{ref}:CHANGELOG.md", check=False).stdout.decode("utf-8", "replace")
    heading = re.compile(rf"^## \[?{re.escape(version)}\]?(\s|$)", re.MULTILINE)
    if not heading.search(changelog):
        findings.append(f"ref:CHANGELOG.md: no section for version {printable_version(version, lists)}")
    return findings


def tag_findings(repo: Path, tag: str, lists: NegativeList) -> list[str]:
    version, label = ref_version(repo, "HEAD"), f"tag:{shown(tag, lists)}"
    kind = git(repo, "cat-file", "-t", f"refs/tags/{tag}", check=False)
    if kind.returncode != 0:
        return [f"{label}: does not exist"]
    findings = []
    if tag != f"v{version}":
        findings.append(f"{label}: name is not v{printable_version(version, lists)}")
    if kind.stdout.decode().strip() != "tag":
        findings.append(f"{label}: not an annotated tag")
    target = git(repo, "rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}", check=False)
    if target.returncode != 0:
        findings.append(f"{label}: does not point at a commit")
    elif target.stdout.decode().strip() != git_text(repo, "rev-parse", "HEAD").strip():
        findings.append(f"{label}: does not point at HEAD")
    return findings


# -- 2: negative list over the ref, the history and the metadata --------------------------------


def ref_findings(repo: Path, ref: str, lists: NegativeList) -> list[str]:
    names = git(repo, "ls-tree", "-r", "-z", "--name-only", ref).stdout.decode("utf-8", "replace")
    findings = []
    for path in filter(None, names.split("\0")):
        findings += name_findings("ref", path, lists)
        findings += scan_data(f"ref:{shown(path, lists)}", path, git(repo, "show", f"{ref}:{path}").stdout, lists)
    return findings


def patch_path(field: str) -> str:
    """The new-side path of a `+++ ` line: git C-quotes unusual names and ends a name containing a
    space with a tab; `/dev/null` means none."""
    field = field.removesuffix("\t")
    if len(field) >= 2 and field.startswith('"') and field.endswith('"'):
        field = codecs.escape_decode(field[1:-1].encode("utf-8"))[0].decode("utf-8", "replace")
    return field[2:] if field.startswith("b/") else ""


class _PatchScanner:
    """Walks `git log -p` output and checks every added line at its line number in the new file."""

    def __init__(self, lists: NegativeList, names: NegativeList | None = None) -> None:
        self.lists = lists
        # The list a printed path is withheld by: the whole one, current-state classes included, so
        # a path of the history that holds a private word is never printed either.
        self.names = names or lists
        self.commit = self.path = ""
        self.old_left = self.new_left = self.line_no = 0
        self.findings: dict[str, None] = {}

    def feed(self, raw: bytes | str) -> None:
        """One line of the log, as bytes: an added line keeps its own bytes for every reading."""
        raw_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        if self.old_left > 0 or self.new_left > 0:
            self._hunk_line(raw_bytes)
            return
        text = raw_bytes.decode("utf-8", "replace")
        if text.startswith("commit "):
            self.commit, self.path = text[7:19], ""
        elif text.startswith("+++ "):
            self.path = patch_path(text[4:])
        elif match := HUNK_HEADER.match(text):
            self.old_left = int(match[1] if match[1] is not None else 1)
            self.new_left = int(match[3] if match[3] is not None else 1)
            self.line_no = int(match[2])

    def _hunk_line(self, raw: bytes) -> None:
        marker, content = raw[:1], raw[1:]
        if marker == b"+":
            if self.path:  # every reading HEAD gets, a skipped suffix in its printable runs (Codex r3-6, r4-2)
                place = f"history:{self.commit}:{shown(self.path, self.names)}:{self.line_no}"
                for text in text_variants(content, binary=skipped(self.path, self.lists)):
                    for name in leak_classes(text, self.lists, self.path):
                        self.findings[f"{place}: {name}"] = None
            self.new_left -= 1
            self.line_no += 1
        elif marker == b"-":
            self.old_left -= 1
        elif marker == b" " or raw == b"":
            self.old_left, self.new_left, self.line_no = self.old_left - 1, self.new_left - 1, self.line_no + 1


def history_findings(repo: Path, lists: NegativeList) -> list[str]:
    names, lists = lists, current_state_only(lists)
    # --diff-merges=separate: a merge against each parent, whatever log.diffMerges says; --text:
    # binary files as text; --no-renames: a renamed file is a full addition under its new name;
    # explicit prefixes: a repository's diff.noprefix/mnemonicPrefix cannot change them.
    log = git(repo, "log", "--all", "--diff-merges=separate", "-p", "--text", "--no-renames", "--no-color", "--no-ext-diff",
              "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/", "--format=commit %H").stdout
    scanner = _PatchScanner(lists, names)
    for raw in log.split(b"\n"):
        scanner.feed(raw.removesuffix(b"\r"))
    # The names of the history against the whole list: the exception is for earlier contents only (Codex r4-3).
    return list(scanner.findings) + history_name_findings(repo, names)


def history_name_findings(repo: Path, lists: NegativeList) -> list[str]:
    """Every path any commit ever held, also an empty file that no patch hunk shows."""
    seen: set[str] = set()
    findings = []
    for sha in git_text(repo, "rev-list", "--all").split():
        names = git(repo, "ls-tree", "-r", "-z", "--name-only", sha).stdout.decode("utf-8", "replace")
        for path in set(filter(None, names.split("\0"))) - seen:
            seen.add(path)
            findings += name_findings(f"history:{sha[:12]}", path, lists)
    return findings


def maintainer_email(repo: Path) -> str | None:
    """The maintainer identity: `MODELROOM_RELEASE_MAINTAINER_EMAIL`, else `git config user.email`.

    On the maintainer's machine the history's own address is therefore accepted; anywhere else
    it stays a finding, which is the honest answer for an address that is not the runner's.
    """
    override = os.environ.get(MAINTAINER_ENV, "").strip()
    if override:
        return override
    configured = git(repo, "config", "--get", "user.email", check=False).stdout.decode("utf-8", "replace").strip()
    return configured or None


def meta_allow_pattern(repo: Path) -> re.Pattern[str]:
    maintainer = maintainer_email(repo)
    alternatives = ALLOWED_META + (f"|{re.escape(maintainer)}" if maintainer else "")
    return re.compile(rf"(?<![\w.+-])(?:{alternatives})(?![\w.-])", re.IGNORECASE)


def read_objects(repo: Path, names: list[str]) -> list[tuple[str, str, bytes]]:
    """(hash, type, raw content) for each object, read length-framed through `git cat-file --batch`."""
    if not names:
        return []
    data = git(repo, "cat-file", "--batch", stdin=("\n".join(names) + "\n").encode()).stdout
    objects, pos = [], 0
    for _ in names:
        end = data.index(b"\n", pos)
        header = data[pos:end].decode().split()
        if len(header) != 3:
            raise UsageError("git cat-file could not read an object")
        size = int(header[2])
        objects.append((header[0], header[1], data[end + 1:end + 1 + size]))
        pos = end + 1 + size + 1
    return objects


UNSUPPORTED_ENCODING = "unsupported encoding"
_ENCODING_HEADER = re.compile(rb"^encoding (\S+)$", re.MULTILINE)


def decoded_object(body: bytes) -> tuple[str, str | None]:
    """A raw commit or tag object as text, in the encoding its `encoding` header declares (git writes
    the message and the names in it); an encoding Python does not know is refused as a finding."""
    declared = _ENCODING_HEADER.search(body.partition(b"\n\n")[0])
    if declared is None:
        return body.decode("utf-8", "replace"), None
    try:
        return body.decode(codecs.lookup(declared[1].decode("ascii", "replace")).name, "replace"), None
    except LookupError:
        return body.decode("utf-8", "replace"), UNSUPPORTED_ENCODING


def object_fields(body: bytes) -> tuple[list[tuple[str, str]], str]:
    """The header fields of a raw commit or tag object (people split into name and email) and its message."""
    head, _, message = decoded_object(body)[0].partition("\n\n")
    fields, key = [], ""
    for line in head.split("\n"):
        if line.startswith(" "):
            fields.append((key, line[1:]))
            continue
        key, _, value = line.partition(" ")
        person = PERSON.match(value) if key in PERSON_HEADERS else None
        if person:
            fields += [(f"{key} name", person[1]), (f"{key} email", person[2])]
        else:
            fields.append((key, value))
    return fields, message


def object_findings(place: str, body: bytes, lists: NegativeList, allowed: re.Pattern[str]) -> list[str]:
    """Every field and the message, each also with its JSON escapes resolved. The allowed addresses
    are taken out for the class `e-mail address` only; every other class reads the original."""
    problem = decoded_object(body)[1]
    fields, message = object_fields(body)
    view = lambda text: allowed.sub("noreply", text)  # noqa: E731
    findings = [f"{place}: {problem}"] if problem else []
    for field, value in fields:
        findings += name_findings(place, field, lists, kind="field")
        label = f"{place}:{shown(field, lists)}"
        classes = dict.fromkeys(name for text in unescaped(value) for name in leak_classes(text, lists, email_view=view))
        findings += [f"{label}: {name}" for name in classes]
    for text in unescaped(message):
        findings += scan_text(f"{place}:message", "", text, lists, email_view=view)
    return list(dict.fromkeys(findings))


def tag_objects(repo: Path, lists: NegativeList) -> list[tuple[str, bytes]]:
    """Every annotated tag object any ref reaches, following tag-to-tag chains to the end."""
    refs = git_text(repo, "for-each-ref", "--format=%(objectname) %(objecttype) %(refname)").splitlines()
    queue = [(sha, shown(ref.removeprefix("refs/tags/"), lists))
             for sha, kind, ref in (line.split(" ", 2) for line in refs) if kind == "tag"]
    seen, found = set(), []
    while queue:
        sha, name = queue.pop()
        if sha in seen:
            continue
        seen.add(sha)
        body = read_objects(repo, [sha])[0][2]
        found.append((name, body))
        inner = re.search(rb"^object ([0-9a-f]+)\ntype tag$", body, re.MULTILINE)
        if inner:
            queue.append((inner[1].decode(), inner[1].decode()[:12]))
    return found


def meta_findings(repo: Path, lists: NegativeList,
                  accepted: set[tuple[str, str, int, str]] = ALLOWED_META_FINDINGS) -> list[str]:
    allowed = meta_allow_pattern(repo)
    findings = []
    for sha, _, body in read_objects(repo, git_text(repo, "rev-list", "--all").split()):
        skip = {f"meta:{sha[:12]}:{field}:{line}: {name}" for full, field, line, name in accepted if full == sha}
        findings += [f for f in object_findings(f"meta:{sha[:12]}", body, lists, allowed) if f not in skip]
    for name, body in tag_objects(repo, lists):
        findings += object_findings(f"meta:tag {name}", body, lists, allowed)
    return list(dict.fromkeys(findings))


def source_checks(repo: Path, ref: str, tag: str | None, lists: NegativeList) -> list[Callable[[], list[str]]]:
    """Checks 1 and 2, everything that needs no build, in reporting order."""
    checks = [lambda: worktree_findings(repo), lambda: version_findings(repo, ref, lists)]
    if tag:
        checks.append(lambda: tag_findings(repo, tag, lists))
    return checks + [lambda: ref_findings(repo, ref, lists), lambda: history_findings(repo, lists),
                     lambda: meta_findings(repo, lists)]


def source_findings(repo: Path, ref: str, tag: str | None, lists: NegativeList) -> list[str]:
    return [finding for check in source_checks(repo, ref, tag, lists) for finding in check()]


# -- 3: build and inspect the distributions ------------------------------------------------------


def forbidden_file_class(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    base = parts[-1]
    if (base == ".env" or base.startswith(".env.")) and base != ".env.example":
        return "environment file"
    if base in STATE_FILES:
        return "runtime state file"
    if base == "modelroom.toml":
        return "operator configuration"
    if len(parts) >= 2 and parts[-2] == "hardware" and base.endswith(".json"):
        return "hardware profile"
    if ".trash" in parts:
        return "trash folder"
    if parts[0] == "scripts":
        return "release script"
    return None


class ArchiveContent(NamedTuple):
    """A wheel or an sdist as the check reads it: its regular files, every entry name (folders and
    links included), and its own metadata as `(kind, text)` -- comments, link targets, owners, pax headers."""

    members: list[tuple[str, bytes]]
    names: list[str]
    metadata: list[tuple[str, bytes]]


def archive_members(archive: Path) -> Iterator[tuple[str, bytes]]:
    """(name relative to the package root, content) for every regular file of a wheel or sdist."""
    yield from read_archive(archive).members


def read_archive(archive: Path) -> ArchiveContent:
    """The whole archive; one that does not read is a usage error that names nothing of its content."""
    try:
        return _read_wheel(archive) if archive.suffix == ".whl" else _read_sdist(archive)
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError, ValueError, KeyError) as exc:
        raise UsageError("a built distribution could not be read") from exc


def _read_wheel(archive: Path) -> ArchiveContent:
    content = ArchiveContent([], [], [])
    with zipfile.ZipFile(archive) as wheel:
        content.metadata.append(("archive comment", wheel.comment))
        for info in wheel.infolist():
            content.names.append(info.filename)
            content.metadata.extend([("entry comment", info.comment), ("entry extra field", info.extra)])
            if not info.is_dir():
                content.members.append((info.filename, wheel.read(info)))
    return content


def _read_sdist(archive: Path) -> ArchiveContent:
    content = ArchiveContent([], [], [])
    with tarfile.open(archive, "r:gz") as sdist:
        content.metadata.extend([("pax header", f"{key}={value}".encode("utf-8", "replace")) for key, value in sdist.pax_headers.items()])
        for info in sdist.getmembers():
            content.names.append(info.name)
            content.metadata.extend([("link target", info.linkname.encode("utf-8", "replace")),
                                 ("owner", f"{info.uname}\n{info.gname}".encode("utf-8", "replace"))])
            content.metadata.extend([("pax header", f"{key}={value}".encode("utf-8", "replace")) for key, value in info.pax_headers.items()])
            if info.isfile():
                data = sdist.extractfile(info).read()
                content.members.append((PurePosixPath(*PurePosixPath(info.name).parts[1:]).as_posix(), data))
    return content


def wheel_shape_findings(label: str, members: list[tuple[str, bytes]]) -> list[str]:
    findings = []
    if not any(name.startswith("modelroom/") for name, _ in members):
        findings.append(f"{label}: no modelroom/ package")
    entry_files = [data for name, data in members if name.endswith(".dist-info/entry_points.txt")]
    parser = configparser.ConfigParser()
    try:
        if entry_files:
            parser.read_string(entry_files[0].decode("utf-8", "replace"))
    except configparser.Error:  # its message quotes the file: never passed on
        return findings + [f"{label}: entry_points.txt does not read"]
    if not parser.has_section("console_scripts") or parser["console_scripts"].get("modelroom") != "modelroom.cli:main":
        findings.append(f"{label}: no modelroom console entry (modelroom.cli:main)")
    return findings


def distribution_findings(archive: Path, lists: NegativeList) -> list[str]:
    """Every member on its own, duplicates included: a later member of the same name must not hide
    an earlier one."""
    content = read_archive(archive)
    members = content.members
    label = f"dist:{shown(archive.name, lists)}"
    findings = name_findings("dist", archive.name, lists)
    names = [name for name, _ in members]
    for name in dict.fromkeys(n for n in names if names.count(n) > 1):
        findings.append(f"{label}:{shown(name, lists)}: duplicate member")
    for entry in dict.fromkeys(content.names):  # every entry, folders and links included
        findings += name_findings(label, entry, lists, kind="entry")
    for kind, data in content.metadata:
        if data:
            findings += [finding for finding in scan_data(f"{label}:<{kind}>", "", data, lists) if finding not in findings]
    for name, data in members:
        place = f"{label}:{shown(name, lists)}"
        file_class = forbidden_file_class(name)
        if file_class:
            findings.append(f"{place}: forbidden file ({file_class})")
        findings += scan_data(place, name, data, lists)
    if archive.suffix == ".whl":
        findings += wheel_shape_findings(label, members)
    return findings


def build_distributions(repo: Path, ref: str, workdir: Path) -> list[Path]:
    """`git archive <ref>` into `workdir`, then `uv build --offline` there; returns wheel and sdist."""
    source, dist = workdir / "source", workdir / "dist"
    source.mkdir()
    export = workdir / "export.tar"
    export.write_bytes(git(repo, "archive", "--format=tar", ref).stdout)
    with tarfile.open(export) as archive:
        archive.extractall(source, filter="data")
    try:
        result = subprocess.run(["uv", "build", "--offline", "--out-dir", str(dist), str(source)], capture_output=True)
    except FileNotFoundError as exc:
        raise UsageError("uv is not on PATH") from exc
    if result.returncode != 0:
        raise UsageError(f"uv build --offline failed (exit {result.returncode}); run it in a clean clone to see why")
    built = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
    if len(built) != 2:
        raise UsageError(f"uv build produced {len(built)} distributions, expected a wheel and an sdist")
    return built


def dist_findings(repo: Path, ref: str, lists: NegativeList) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="modelroom-release-check-") as workdir:
        findings = []
        for archive in build_distributions(repo, ref, Path(workdir)):
            findings += distribution_findings(archive, lists)
        return findings


def summary_line(findings: list[str], version: str) -> str:
    return f"RELEASE-CHECK: {len(findings)} findings" if findings else f"RELEASE-CHECK: OK {version}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release-check", description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="the release tag that must point at HEAD, e.g. v0.1.0")
    parser.add_argument("--repo", type=Path, default=REPO, help="repository to check (default: this one)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    findings: list[str] = []
    words_error: UsageError | None = None
    try:
        ensure_repository(args.repo)
        try:
            words = private_words(args.repo)
        except UsageError as exc:  # every other check still runs and reports; the gate does not pass
            words, words_error = None, exc
        lists = with_private_words(load_negative_list(), words)
        for check in source_checks(args.repo, "HEAD", args.tag, lists) + [lambda: dist_findings(args.repo, "HEAD", lists)]:
            findings += check()
        version = printable_version(ref_version(args.repo, "HEAD"), lists)
        if words_error is not None:
            raise words_error
    except UsageError as exc:
        errors = [exc] + ([words_error] if words_error is not None and words_error is not exc else [])
        print("\n".join(findings + [f"RELEASE-CHECK: error: {error}" for error in errors]))
        return 2
    except Exception as exc:  # noqa: BLE001 -- a parser's message may quote the data: only the class
        print("\n".join(findings + [f"RELEASE-CHECK: error: unexpected {type(exc).__name__}; nothing of its text is shown"]))
        return 2
    print("\n".join(findings + [summary_line(findings, version)]))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
