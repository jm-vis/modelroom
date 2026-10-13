"""What the catalog can hold beside its model rows, and the fixed sources and words around it (AP9-KO).

Two findings, both part of a catalog row of schema 3 (`catalog.CatalogModel`):

- `PackageCheck` (V1, V2): whether the GGUF architecture a package states is among the architectures
  the llama.cpp source knows. It is bound to one package -- repository, revision and weight file or
  first shard -- and to the day and the commit of the source it was compared with. It says nothing
  about any other package, revision or repository, and `present` is no promise that a runtime loads
  the file. A package without a check is `unknown`; so is one whose revision differs.
- `SetupNote` (V7): whether the model card of the publisher's repository holds setup instructions
  for vLLM, at one revision of that card. It is a pointer to the card, never a copy of an
  instruction, and never a statement about a runtime.

The rest is the contract around them: where the llama.cpp block is read (`llamacpp_*`, `parse_arch_names`:
read whole or not at all), which file of a repository the Hub's `gguf` field describes
(`hub_metadata_file`), how a failed run keeps what it had (`merge_*`), and the words the decision E3
fixed (`LOAD_BLOCKED` and its kin). CONTRACTS.md, "Package findings and the load check".
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import hf_id_key, validate_hf_repo
from .quantization import identity_stem, parse_hf_quant

# --- the source of the architectures (V5) ---------------------------------------------------------------------------

LLAMACPP_REPO = "ggml-org/llama.cpp"
LLAMACPP_HOST = "api.github.com"  # the host of the scanner only; the runtime of modelroom never opens it (`http.py`)
LLAMACPP_ARCH_FILE = "src/llama-arch.cpp"
LLAMACPP_REF = "HEAD"  # the default branch; resolved to a commit first, then only the commit is read
LLAMACPP_COMMIT_ACCEPT = "application/vnd.github.sha"
LLAMACPP_FILE_ACCEPT = "application/vnd.github.raw+json"
LLAMACPP_COMMIT_MAX_BYTES = 256
LLAMACPP_FILE_MAX_BYTES = 512 * 1024
LLAMACPP_MAX_REDIRECTS = 0  # an answer that redirects is a fault, not a way to another host
ARCH_MIN_ENTRIES = 100  # a plausibility floor: the block held 157 on 2026-10-13; fewer is a block not understood

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_BLOCK_START = "static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {"
_BLOCK_END = "};"
_ENTRY_RE = re.compile(r'[ \t]*\{ (LLM_ARCH_[A-Z0-9_]+),[ \t]+"([^"\\]+)"[ \t]*\},[ \t]*(?://.*)?')
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_SENTINEL = ("LLM_ARCH_UNKNOWN", "(unknown)")
_LOOKUP_NAME = "llm_arch_from_string"
# The whole known lookup, from the start of a line to its closing brace: after the exactly checked head ` {` of
# the loop, that line may carry an ordinary `//` comment (`// NOLINT(...)` and the like; a backslash at its end
# and the exotic forms are refused for the whole file); any added logic, other return or cut-off function is no match.
_LOOKUP_RE = re.compile(
    r"^llm_arch llm_arch_from_string\(const std::string & name\) \{\s*"
    r"for \(const auto & kv : LLM_ARCH_NAMES\) \{(?:[ \t]*//[^\n]*)?[ \t]*\r?\n\s*if \(kv\.second == name\) \{\s*"
    r"return kv\.first;\s*\}\s*\}\s*return LLM_ARCH_UNKNOWN;\s*\}",
    re.ASCII | re.MULTILINE,
)
# What makes the compiler read the file other than line by line, anywhere in it: a block comment can hide the
# block or the lookup, a backslash at the end of a line splices the next line into a `//` comment, a carriage
# return without a line feed starts a line, and a conditional or a macro can replace what is read. The file
# held none of them at the recorded commit; any of them is a form not understood. Exotic forms are refused
# whole, never parsed (decision of 2026-10-13, CONTRACTS.md V5 "Limit of the check"): a form feed or vertical
# tab (whitespace inside a directive), the digraph `%:` (a `#`) and a raw string `R"` (any prefix), which can
# hold a copy of the block as text.
_EXOTIC = ("\r", "/*", "*/", "\f", "\v", "%:", 'R"')
_SPLICE_RE = re.compile(r"\\[ \t\f\v]*$")
_PREPROCESSOR_RE = re.compile(r"[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif|elifdef|elifndef|else|endif|define|undef|line)\b")


class ArchBlockError(ValueError):
    """The source of the architectures could not be read whole; no result follows from it."""


def _commit(value: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"not a commit id (40 lowercase hex characters): {value!r}")
    return value


def llamacpp_commit_url() -> str:
    """Where the commit of the default branch is asked (`LLAMACPP_COMMIT_ACCEPT`: the answer is the id alone)."""
    return f"https://{LLAMACPP_HOST}/repos/{LLAMACPP_REPO}/commits/{LLAMACPP_REF}"


def llamacpp_block_url(commit: str) -> str:
    """Where the file with the block is read, bound to `commit` (`LLAMACPP_FILE_ACCEPT`: the file itself)."""
    return f"https://{LLAMACPP_HOST}/repos/{LLAMACPP_REPO}/contents/{LLAMACPP_ARCH_FILE}?ref={_commit(commit)}"


def llamacpp_page_url(commit: str) -> str:
    """The page of that file at `commit`, the `source_url` a check records."""
    return f"https://github.com/{LLAMACPP_REPO}/blob/{_commit(commit)}/{LLAMACPP_ARCH_FILE}"


def read_commit(answer: str) -> str:
    """The commit id of the answer of `llamacpp_commit_url`; anything but the id alone is a fault."""
    if not isinstance(answer, str) or _COMMIT_RE.fullmatch(answer) is None:
        raise ArchBlockError("the answer is not a commit id")
    return answer


def parse_arch_names(text: str) -> frozenset[str]:
    """The GGUF identifiers of the block `LLM_ARCH_NAMES` of `text`, or `ArchBlockError`.

    Read whole or not at all: the block must start once and end, every line in it must be an entry of the
    one known form (a blank line or a `//` line is nothing), no enum and no identifier may repeat, the
    sentinel `LLM_ARCH_UNKNOWN` is no architecture, at least `ARCH_MIN_ENTRIES` must remain, and the
    function that looks a name up must still compare the second member of an entry with the name it is
    given -- otherwise the block is no longer what the loader reads. A partial result is never returned.
    Nowhere in the file may the compiler read lines other than as they are split here (`_SPLICE_RE`,
    `_PREPROCESSOR_RE`, a block comment, a lone carriage return), the sentinel must be the last entry, no
    entry may follow the end, and the lookup must be defined once.
    """
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    for number, line in enumerate(lines):
        if any(form in line for form in _EXOTIC):
            raise ArchBlockError(f"line {number + 1} holds a form not understood (a carriage return, a block comment, a form feed or vertical tab, a digraph or a raw string)")
        if _SPLICE_RE.search(line) or _PREPROCESSOR_RE.match(line):
            raise ArchBlockError(f"line {number + 1} continues into the next line or is a conditional or a macro")
    starts = [number for number, line in enumerate(lines) if line.rstrip(" \t") == _BLOCK_START]
    if len(starts) != 1:
        raise ArchBlockError("the block LLM_ARCH_NAMES is " + ("missing" if not starts else "there more than once"))
    ends = [number for number in range(starts[0] + 1, len(lines)) if lines[number].rstrip(" \t") == _BLOCK_END]
    if not ends:
        raise ArchBlockError("the block LLM_ARCH_NAMES does not end")
    if any(_ENTRY_RE.fullmatch(line) for line in lines[ends[0] + 1 :]):
        raise ArchBlockError("an entry follows the end of the block")
    names: list[str] = []
    enums: list[str] = []
    for number in range(starts[0] + 1, ends[0]):
        line = lines[number]
        if not line.strip() or line.strip().startswith("//"):
            continue
        entry = _ENTRY_RE.fullmatch(line)
        if entry is None:
            raise ArchBlockError(f"line {number + 1} of the block is not an entry of the known form")
        enum, name = entry.groups()
        if enum == _SENTINEL[0]:
            if name != _SENTINEL[1]:
                raise ArchBlockError("the sentinel entry is not what it was")
        elif _NAME_RE.fullmatch(name) is None:
            raise ArchBlockError(f"line {number + 1} of the block names an identifier of an unknown form")
        else:
            names.append(name)
        enums.append(enum)
    if len(enums) != len(set(enums)) or len(names) != len(set(names)):
        raise ArchBlockError("an entry of the block repeats")
    if not enums or enums[-1] != _SENTINEL[0]:
        raise ArchBlockError("the sentinel is not the last entry of the block")
    if len(names) < ARCH_MIN_ENTRIES:
        raise ArchBlockError(f"the block holds {len(names)} entries, fewer than the {ARCH_MIN_ENTRIES} a whole block has")
    if text.count(_LOOKUP_NAME) != 1:
        raise ArchBlockError("the lookup of an identifier is not named exactly once")
    if _LOOKUP_RE.search(text) is None:
        raise ArchBlockError("the lookup of an identifier no longer compares it with the second member of an entry")
    return frozenset(names)


def compare_architecture(architecture: str, names: frozenset[str]) -> Literal["present", "absent"]:
    """`present` when the identifier is one of `names` as a whole string, else `absent`."""
    return "present" if architecture in names else "absent"


# --- which file the Hub's `gguf` field describes (V1) -----------------------------------------------------------------


def hub_metadata_file(files: Sequence[tuple[str, int]], total_file_size: object) -> str | None:
    """The one file of `files` (`(path, size)`) that the Hub's `gguf` field describes, or `None`.

    The Hub reads the metadata of one file of a repository and states that file's size as `totalFileSize`
    (recorded for seven repositories, `docs/ap9-ko/feasibility/`; the choice of the file is the Hub's own,
    by a rule it does not publish). The field is therefore a statement about the file of that size and
    nothing else; when no file or more than one has the size, the scope cannot be shown.
    """
    if isinstance(total_file_size, bool) or not isinstance(total_file_size, int):
        return None
    found = [path for path, size in files if size == total_file_size]
    return found[0] if len(found) == 1 else None


# --- the findings ----------------------------------------------------------------------------------------------------

PACKAGE_RESULTS = ("present", "absent")  # `unknown` is the absence of a check, never a stored value
ARCHITECTURE_SOURCES = ("hub_field", "file_header")
SETUP_STATUSES = ("instructions", "none_recognized", "not_read")
# V6: how much of the Hub one run of the scanner spends on the checks, and how many rows it takes.
PACKAGE_ROWS_PER_RUN = 40
PACKAGE_STAGE_BUDGET = 200
CARD_MAX_BYTES = 128 * 1024  # the window of a model card read for a setup note; of a longer card only the start, positive findings only

_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_ARCHITECTURE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_FILE_RE = re.compile(r"[A-Za-z0-9._()+=,@~-]+(?:/[A-Za-z0-9._()+=,@~-]+)*")
_SHARD_INDEX_RE = re.compile(r"-(?:split-)?(\d{5})-of-\d{5}\.gguf$", re.IGNORECASE)
_PACKAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_HUB = "https://huggingface.co"


def _revision(value: str, name: str) -> str:
    if _REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 40-hex commit id: {value!r}")
    return value


class PackageCheck(BaseModel):
    """The architecture of one GGUF package compared with a commit of llama.cpp (V1, V2).

    `identity` is the third part of the package identity (`quantization.package_identity_key`: the stems of
    the weight files, shard suffix removed) and `file` the weight file, or the first shard of the set, whose
    architecture was read -- by its header, or by the Hub's field where `hub_metadata_file` shows that field
    to describe that very file. `revision` is the revision of the repository that was read.
    """

    model_config = ConfigDict(extra="forbid")

    repo: str
    revision: str
    identity: str = Field(max_length=300)
    file: str = Field(max_length=300)
    quantization: str | None = None
    architecture: str
    architecture_source: Literal["hub_field", "file_header"]
    result: Literal["present", "absent"]
    llamacpp_commit: str
    source_url: str
    checked: date
    # How many weight packages of this quantization the repository held at `revision`, as the scanner counted
    # them from the file list. A pull name (`hf.co/<repo>:<quant>`) is assigned to this check only where it is 1:
    # without the count, a second package of the quantization that was never checked would be hidden.
    same_quantization: int | None = Field(default=None, ge=1)

    @field_validator("repo")
    @classmethod
    def _check_repo(cls, value: str) -> str:
        return validate_hf_repo(value)

    @field_validator("revision")
    @classmethod
    def _check_revision(cls, value: str) -> str:
        return _revision(value, "revision")

    @field_validator("llamacpp_commit")
    @classmethod
    def _check_commit(cls, value: str) -> str:
        return _revision(value, "llamacpp_commit")

    @field_validator("architecture")
    @classmethod
    def _check_architecture(cls, value: str) -> str:
        if _ARCHITECTURE_RE.fullmatch(value) is None:
            raise ValueError(f"architecture is not a GGUF identifier: {value!r}")
        return value

    @field_validator("file")
    @classmethod
    def _check_file(cls, value: str) -> str:
        if _FILE_RE.fullmatch(value) is None or ".." in value.split("/") or not value.lower().endswith(".gguf"):
            raise ValueError(f"file is not a relative path of a GGUF file: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_binding(self) -> "PackageCheck":
        if self.identity != identity_stem(self.file):
            raise ValueError(f"identity {self.identity!r} is not the stem of file {self.file!r}")
        shard = _SHARD_INDEX_RE.search(self.file)
        if shard is not None and int(shard.group(1)) != 1:
            raise ValueError(f"file must be the first shard of its set: {self.file!r}")
        if self.quantization != parse_hf_quant(self.file):
            raise ValueError(f"quantization {self.quantization!r} is not what the name of {self.file!r} says")
        if self.source_url != llamacpp_page_url(self.llamacpp_commit):
            raise ValueError("source_url is not the page of the block at llamacpp_commit")
        return self


class SetupNote(BaseModel):
    """What the model card of the publisher's repository says about setting up vLLM (V7).

    Valid for `model` and for the card at `card_revision` only. `package` is the name of a package the card
    states as required for inference, set only where the rule of the scanner found exactly that form;
    `none_recognized` says the rule found nothing, never that nothing is required. `not_read` is the form of
    a run that could not read the card: it is reported, and never stored in a catalog.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    status: Literal["instructions", "none_recognized", "not_read"]
    card_revision: str | None = None
    source_url: str
    checked: date
    package: str | None = None
    reason: str | None = Field(default=None, max_length=97)

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        return validate_hf_repo(value)

    @field_validator("package")
    @classmethod
    def _check_package(cls, value: str | None) -> str | None:
        if value is not None and _PACKAGE_RE.fullmatch(value) is None:
            raise ValueError(f"package is not the name of a package: {value!r}")
        return value

    @field_validator("reason")
    @classmethod
    def _check_reason(cls, value: str | None) -> str | None:
        if value is not None and (not value.isascii() or not value.isprintable()):
            raise ValueError(f"reason must be one printable ASCII line: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_status(self) -> "SetupNote":
        unread = self.status == "not_read"
        if unread != (self.card_revision is None):
            raise ValueError("card_revision is set exactly when the card was read")
        if unread != (self.reason is not None):
            raise ValueError("reason is set exactly when the card was not read")
        if self.card_revision is not None:
            _revision(self.card_revision, "card_revision")
        if self.package is not None and self.status != "instructions":
            raise ValueError("package belongs to a note that found instructions")
        expected = f"{_HUB}/{self.model}" + ("" if unread else f"/blob/{self.card_revision}/README.md")
        if self.source_url != expected:
            raise ValueError(f"source_url is not the address of the card read: {expected}")
        return self


# --- keeping what a run could not read again --------------------------------------------------------------------------


def merge_package_checks(existing: Sequence[PackageCheck], fresh: Sequence[PackageCheck]) -> list[PackageCheck]:
    """`existing` with each package that `fresh` checked again replaced by its fresh check.

    A package is the pair (`repo`, `identity`), the repository compared by `hf_id_key` (case-insensitive). A check made on an older day never replaces a newer one, an
    existing check without a fresh one stays with its own day, and a new package is appended -- so a run that
    reads nothing (a fault of the source, a budget that ended) passes `fresh=[]` and removes nothing.
    """
    merged = list(existing)
    for check in fresh:
        key = (hf_id_key(check.repo), check.identity)
        position = next((index for index, old in enumerate(merged) if (hf_id_key(old.repo), old.identity) == key), None)
        if position is None:
            merged.append(check)
        elif check.checked >= merged[position].checked:
            merged[position] = check
    return merged


def package_rows_for_run(rows: Sequence[str], attempts: Mapping[str, date], today: date, limit: int = PACKAGE_ROWS_PER_RUN) -> list[str]:
    """The catalog rows (`hf_repo`) one run of the scanner takes, at most `limit` (V6).

    The state is `attempts`: the day each row was last tried, **whether or not the try found a repository or
    a check** -- an empty or failed row counts. Rows never tried come first, then the one tried longest ago, then
    by name; a row already tried `today` is not taken again. So the run moves on through any number of rows
    that yield nothing, and no stored check is read or changed by it.
    """
    days = {hf_id_key(row): day for row, day in record_attempts(attempts, [], today).items()}
    due = [row for row in rows if days.get(hf_id_key(row)) != today]
    return sorted(due, key=lambda row: (days.get(hf_id_key(row), date.min), row))[:limit]


def record_attempts(attempts: Mapping[str, date], taken: Sequence[str], today: date) -> dict[str, date]:
    """`attempts` with every row of `taken` tried `today`.

    Rows are compared by `hf_id_key` (ids are case-insensitive): spellings of one row are merged into one entry,
    the youngest day wins and keeps its spelling, and a row of `taken` keeps the spelling it was given."""
    merged: dict[str, tuple[str, date]] = {}
    for row, day in [*attempts.items(), *((row, today) for row in taken)]:
        key = hf_id_key(row)
        if key not in merged or day >= merged[key][1]:
            merged[key] = (row, day)
    return dict(merged.values())


def merge_setup_note(existing: SetupNote | None, fresh: SetupNote) -> SetupNote | None:
    """The note a catalog holds after a read: a card that was read replaces the note, one that was not changes nothing."""
    return existing if fresh.status == "not_read" else fresh


# --- the words of the decision E3 ---------------------------------------------------------------------------------------

LOAD_BLOCKED = (
    "{runtime} installation not offered for {model}: architecture {arch} is absent from the checked llama.cpp source "
    "(checked {date})"
)
COMMUNITY_CONVERSION = "{packager} is a community conversion, not checked; not offered as the automatic choice"
KNOWLEDGE_HINT = "{model} ({publisher}) is in the catalog; no GGUF from its publisher or a listed packager is known here"
KNOWLEDGE_HINT_TENSOR = "tensor packages are shown when a machine with vLLM is selected; check the model card for runtime requirements"
SETUP_NOTE = "{model} has setup instructions in its model card: {url}"
SETUP_NOTE_PACKAGE = "it names the package {package} as required; the commands below are templates without it"
# Not fixed by E3: the line for a list in which every shown row is blocked or not recommendable (V4).
NO_RECOMMENDATION = "no automatic recommendation: every row shown is blocked or not checked; choose a row by name"
KNOWLEDGE_HINT_MAX = 3  # lines of the knowledge hint one search shows; the count of the rest goes to `search.json`


def load_blocked_text(runtime: str, model: str, check: PackageCheck) -> str:
    """The red line of a blocked installation (L1); only a check that says `absent` blocks."""
    if check.result != "absent":
        raise ValueError("only a check that says absent blocks an installation")
    return LOAD_BLOCKED.format(runtime=runtime, model=model, arch=check.architecture, date=check.checked.isoformat())


def community_conversion_text(packager: str) -> str:
    return COMMUNITY_CONVERSION.format(packager=packager)


def knowledge_hint_text(model: str, publisher: str, *, tensor_offer: bool) -> str:
    """The gray line of L3; the sentence on tensor packages follows after `; ` only on a positive proof."""
    text = KNOWLEDGE_HINT.format(model=model, publisher=publisher)
    return f"{text}; {KNOWLEDGE_HINT_TENSOR}" if tensor_offer else text


def setup_note_lines(note: SetupNote) -> tuple[str, ...]:
    """The lines over the two vLLM commands: none unless instructions were found, a second line for a named package."""
    if note.status != "instructions":
        return ()
    lines = (SETUP_NOTE.format(model=note.model, url=note.source_url),)
    return lines if note.package is None else (*lines, SETUP_NOTE_PACKAGE.format(package=note.package))
