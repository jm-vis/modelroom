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
   chain); replace refs are ignored and a shallow clone is refused;
3. `uv build --offline` of HEAD (exported with `git archive`, so the working tree cannot leak
   in); the wheel and the sdist are listed and checked against the negative list and against a
   short list of file classes that must never ship. The wheel must carry `modelroom/` and the
   `modelroom` console entry.
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
# Single metadata findings that were read and accepted, each bound to (full commit hash, field,
# line, class): any other line, field, commit or class stays a finding.
ALLOWED_META_FINDINGS = {
    # accepted: names the internal workspace only, no path or data; decided 2026-09-23
    ("e87a208f07a01632d918fe0672f136a41da3a06b", "message", 18, "internal workspace name"),
}


class UsageError(Exception):
    """The check cannot run at all: not a repository, git or uv missing, the build failed."""


class NegativeList(NamedTuple):
    forbidden: dict[str, re.Pattern[str]]
    allowed: dict[tuple[str, str], str]
    skip_suffixes: set[str]


def load_negative_list() -> NegativeList:
    """The negative list lives once, in tests/test_public_hygiene.py; this imports that module."""
    spec = importlib.util.spec_from_file_location("public_hygiene", HYGIENE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return NegativeList(module.FORBIDDEN, module.ALLOWED, module.SKIP_SUFFIXES)


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


def leak_classes(text: str, lists: NegativeList, path: str = "") -> list[str]:
    return [name for name, pattern in lists.forbidden.items()
            if (path, name) not in lists.allowed and pattern.search(text)]


def scan_text(place: str, path: str, text: str, lists: NegativeList) -> list[str]:
    """Every line of `text` against every class; `place` prefixes the finding, never the text."""
    findings = []
    for line_no, line in enumerate(text.split("\n"), start=1):
        findings += [f"{place}:{line_no}: {name}" for name in leak_classes(line, lists, path)]
    return findings


def shown(name: str, lists: NegativeList) -> str:
    """A file, tag or ref name as it may be printed: withheld when it matches the negative list."""
    return WITHHELD if leak_classes(name, lists) else name


def name_findings(prefix: str, name: str, lists: NegativeList, kind: str = "file") -> list[str]:
    return [f"{prefix}:{WITHHELD}: {cls} in {kind} name" for cls in leak_classes(name, lists)]


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


def printable_version(version: str) -> str:
    match = VERSION.match(version)
    return version if match and not match["local"] else WITHHELD


def version_findings(repo: Path, ref: str) -> list[str]:
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
        findings.append(f"ref:CHANGELOG.md: no section for version {printable_version(version)}")
    return findings


def tag_findings(repo: Path, tag: str, lists: NegativeList) -> list[str]:
    version, label = ref_version(repo, "HEAD"), f"tag:{shown(tag, lists)}"
    kind = git(repo, "cat-file", "-t", f"refs/tags/{tag}", check=False)
    if kind.returncode != 0:
        return [f"{label}: does not exist"]
    findings = []
    if tag != f"v{version}":
        findings.append(f"{label}: name is not v{printable_version(version)}")
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
        if skipped(path, lists):
            continue
        text = git(repo, "show", f"{ref}:{path}").stdout.decode("utf-8", "replace")
        findings += scan_text(f"ref:{shown(path, lists)}", path, text, lists)
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

    def __init__(self, lists: NegativeList) -> None:
        self.lists = lists
        self.commit = self.path = ""
        self.old_left = self.new_left = self.line_no = 0
        self.findings: dict[str, None] = {}

    def feed(self, raw: str) -> None:
        if self.old_left > 0 or self.new_left > 0:
            self._hunk_line(raw)
        elif raw.startswith("commit "):
            self.commit, self.path = raw[7:19], ""
        elif raw.startswith("+++ "):
            self.path = patch_path(raw[4:])
        elif match := HUNK_HEADER.match(raw):
            self.old_left = int(match[1] if match[1] is not None else 1)
            self.new_left = int(match[3] if match[3] is not None else 1)
            self.line_no = int(match[2])

    def _hunk_line(self, raw: str) -> None:
        marker, content = raw[:1], raw[1:]
        if marker == "+":
            if self.path and not skipped(self.path, self.lists):
                place = f"history:{self.commit}:{shown(self.path, self.lists)}:{self.line_no}"
                for name in leak_classes(content, self.lists, self.path):
                    self.findings[f"{place}: {name}"] = None
            self.new_left -= 1
            self.line_no += 1
        elif marker == "-":
            self.old_left -= 1
        elif marker == " " or raw == "":
            self.old_left, self.new_left, self.line_no = self.old_left - 1, self.new_left - 1, self.line_no + 1


def history_findings(repo: Path, lists: NegativeList) -> list[str]:
    # --diff-merges=separate: a merge against each parent, whatever log.diffMerges says; --text:
    # binary files as text; --no-renames: a renamed file is a full addition under its new name;
    # explicit prefixes: a repository's diff.noprefix/mnemonicPrefix cannot change them.
    log = git_text(repo, "log", "--all", "--diff-merges=separate", "-p", "--text", "--no-renames", "--no-color", "--no-ext-diff",
                   "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/", "--format=commit %H")
    scanner = _PatchScanner(lists)
    for raw in log.split("\n"):
        scanner.feed(raw.rstrip("\r"))
    return list(scanner.findings) + history_name_findings(repo, lists)


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


def object_fields(body: bytes) -> tuple[list[tuple[str, str]], str]:
    """The header fields of a raw commit or tag object (people split into name and email) and its message."""
    head, _, message = body.decode("utf-8", "replace").partition("\n\n")
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
    fields, message = object_fields(body)
    findings = []
    for field, value in fields:
        findings += name_findings(place, field, lists, kind="field")
        label = f"{place}:{shown(field, lists)}"
        findings += [f"{label}: {name}" for name in leak_classes(allowed.sub("noreply", value), lists)]
    return findings + scan_text(f"{place}:message", "", allowed.sub("noreply", message), lists)


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
    checks = [lambda: worktree_findings(repo), lambda: version_findings(repo, ref)]
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


def archive_members(archive: Path) -> Iterator[tuple[str, bytes]]:
    """(name relative to the package root, content) for every regular file of a wheel or sdist."""
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as wheel:
            for info in wheel.infolist():
                if not info.is_dir():
                    yield info.filename, wheel.read(info)
        return
    with tarfile.open(archive, "r:gz") as sdist:
        for info in sdist.getmembers():
            if info.isfile():
                yield PurePosixPath(*PurePosixPath(info.name).parts[1:]).as_posix(), sdist.extractfile(info).read()


def wheel_shape_findings(label: str, members: list[tuple[str, bytes]]) -> list[str]:
    findings = []
    if not any(name.startswith("modelroom/") for name, _ in members):
        findings.append(f"{label}: no modelroom/ package")
    entry_files = [data for name, data in members if name.endswith(".dist-info/entry_points.txt")]
    parser = configparser.ConfigParser()
    if entry_files:
        parser.read_string(entry_files[0].decode("utf-8", "replace"))
    if not parser.has_section("console_scripts") or parser["console_scripts"].get("modelroom") != "modelroom.cli:main":
        findings.append(f"{label}: no modelroom console entry (modelroom.cli:main)")
    return findings


def distribution_findings(archive: Path, lists: NegativeList) -> list[str]:
    """Every member on its own, duplicates included: a later member of the same name must not hide
    an earlier one."""
    members = list(archive_members(archive))
    label = f"dist:{shown(archive.name, lists)}"
    findings = name_findings("dist", archive.name, lists)
    names = [name for name, _ in members]
    for name in dict.fromkeys(n for n in names if names.count(n) > 1):
        findings.append(f"{label}:{shown(name, lists)}: duplicate member")
    for name, data in members:
        findings += name_findings(label, name, lists)
        place = f"{label}:{shown(name, lists)}"
        file_class = forbidden_file_class(name)
        if file_class:
            findings.append(f"{place}: forbidden file ({file_class})")
        if not skipped(name, lists):
            findings += scan_text(place, name, data.decode("utf-8", "replace"), lists)
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
    try:
        ensure_repository(args.repo)
        lists = load_negative_list()
        for check in source_checks(args.repo, "HEAD", args.tag, lists) + [lambda: dist_findings(args.repo, "HEAD", lists)]:
            findings += check()
        version = printable_version(ref_version(args.repo, "HEAD"))
    except UsageError as exc:
        print("\n".join(findings + [f"RELEASE-CHECK: error: {exc}"]))
        return 2
    print("\n".join(findings + [summary_line(findings, version)]))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
