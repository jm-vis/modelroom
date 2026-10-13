"""The setup note of a model card: what the rule of recognition says (AP9-KO, V7, decision E4-B). Pure: no request, no file.

`detect(markdown, model)` reads the Markdown text of the card of the publisher's repository `model` and answers with a
`Detection`. The rule is finite and written down here, with positive and negative fixtures in `tests/test_ko_p4_card.py`.
Nothing of the card is run, followed or copied; the answer is a status and at most the name of one package.

**Status `instructions`.** The card has a fenced code block (a shell language, or none) with a logical line (backslash
continuations joined, a leading `$ ` allowed) that is `vllm serve <model>` followed by anything, where `<model>` is the
first argument after `serve` and equals the id of this model as a whole, without case. A block for another model (or for
a variant such as `<model>-FP8`, or a path) does not count; neither does a mention in running text. Nothing found: `none_recognized`,
which says only that this rule found nothing -- never that nothing is required.

**A package name** is given only together with `instructions`, and only for a sentence of the running text (outside
code blocks and table rows, front matter excluded) of exactly this form, case ignored: an optional lead-in clause
`To run|serve|use|deploy|start ..., ` that names nothing but the subject, then the subject, `also` optionally, `requires`
or `needs`, `the`, the package name (letters, digits, `.`, `_`, `-`, starting with a letter or digit), optionally
`python`, and the word `package` -- and the rest of the sentence names vLLM or inference or serving.

The subject must mean this model and nothing else. The repository name of the model (`Nova-7B` of `acme/Nova-7B`) means
it anywhere. A pronoun -- `this model`, `the model`, `it` -- and the leading word of the name (`Nova`, which a family
shares) mean it only in a **section without another model**: the section is the text from one heading to the next, read
with every heading above it, and it has another model when its headings or any of its paragraphs, list items or table
rows hold a model reference that is not this model -- a word with a letter and a digit joined by `-`, `.`, `_` or `/`
(`Other-7B`, `other/Another-1`, `Nova-7B-FP8`), or a size (`70B`, `8x7B`). Web addresses and the package named in a
sentence of this form are no reference. In doubt the rule gives no name (V7: a missed name is allowed, a wrong one is
not); a version such as `v0.6.3` or a card name such as `A100-80GB` near the sentence costs the name, never adds one.

A sentence with a negation or an `optional`, one that speaks of training, tuning, quantizing, converting or building, and
any sentence under a heading that speaks of those, give no name; so does a card that names two different packages.
Instructions for another model, an install line alone and every other wording give no name.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import date

from modelroom.catalog_findings import SetupNote
from modelroom.catalog_remote import _clean
from modelroom.contracts import hf_id_key

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^\s`]*)")
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t#]*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SHELL_LANGUAGES = frozenset({"", "bash", "sh", "shell", "console", "zsh", "shell-session", "shellsession"})
_PACKAGE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
_NEGATION_RE = re.compile(r"\b(?:not|no|never|without|cannot|optional|optionally|neither|nor|unless)\b|n't", re.IGNORECASE)
_TRAINING_RE = re.compile(r"\b(?:train\w*|fine[\s-]?tun\w*|tun(?:e|ed|es|ing)|rl|rollouts?|quantiz\w*|convert\w*|conversion|build\w*|compil\w*)\b", re.IGNORECASE)
_CONTEXT_RE = re.compile(r"\b(?:vllm|inference|serving|serve|serves)\b", re.IGNORECASE)
_LEAD_IN = r"(?:to\s+(?:run|serve|use|deploy|start)\b(?P<lead>[^,.;]{0,60}),\s*)?"
_TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]*[A-Za-z0-9])?")
_JOINED_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d).*[-._/].*")  # a letter, a digit and a joint
_SIZE_RE = re.compile(r"\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)?[bm]", re.IGNORECASE)
_VERSION_RE = re.compile(r"v?\d+(?:\.\d+)+", re.IGNORECASE)
_ADDRESS_RE = re.compile(r"\S*://\S*")
_NAMED_PACKAGE_RE = re.compile(r"\bthe\s+" + _PACKAGE + r"\s+(?:python\s+)?package\b", re.IGNORECASE)


@dataclass(frozen=True)
class Detection:
    status: str  # "instructions" or "none_recognized"
    package: str | None = None


def _front_matter_removed(text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].strip() == "---":
        for number in range(1, len(lines)):
            if lines[number].strip() == "---":
                return "\n".join(lines[number + 1 :])
    return text


def _split(text: str) -> tuple[list[tuple[str, list[str]]], list[tuple[int, str, str, bool]]]:
    """The fenced blocks `(language, lines)` and the paragraphs `(section, heading, text, prose)`: `section` counts the
    headings, `heading` is every heading above the text, `prose` is false for a table row."""
    blocks: list[tuple[str, list[str]]] = []
    paragraphs: list[tuple[int, str, str, bool]] = []
    stack: list[tuple[int, str]] = []  # the headings above the text, outermost first: (level, title)
    current, fence = [], None  # fence: (character, length, language, lines)
    heading, section = "", 0

    def flush() -> None:
        if current:
            paragraphs.append((section, heading, " ".join(current), True))
            current.clear()

    for line in text.split("\n"):
        if fence is not None:
            closing = re.match(r"^ {0,3}(" + re.escape(fence[0]) + "{" + str(fence[1]) + r",})[ \t]*$", line)
            if closing:
                blocks.append((fence[2], fence[3]))
                fence = None
            else:
                fence[3].append(line)
            continue
        opening = _FENCE_RE.match(line)
        if opening:
            flush()
            fence = (opening.group(1)[0], len(opening.group(1)), opening.group(2).lower(), [])
            continue
        title = _HEADING_RE.match(line)
        if title:
            flush()
            level = len(title.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title.group(2)))
            heading = " / ".join(text for _, text in stack)  # a section is under every heading above it
            section += 1
        elif not line.strip():
            flush()
        elif line.lstrip().startswith("|"):  # a table row: no sentence of running text, but what it names counts
            flush()
            paragraphs.append((section, heading, line.strip(), False))
        elif _LIST_RE.match(line):
            flush()
            current.append(_LIST_RE.sub("", line, count=1).strip())
        else:
            current.append(line.strip().lstrip(">").strip())
    flush()
    if fence is not None:  # an unclosed fence is a block to its end; the text in it is not prose
        blocks.append((fence[2], fence[3]))
    return blocks, paragraphs


def _logical_lines(lines: list[str]) -> list[str]:
    joined, carry = [], ""
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            carry += stripped[:-1] + " "
            continue
        joined.append((carry + stripped).strip())
        carry = ""
    if carry.strip():
        joined.append(carry.strip())
    return joined


def _serves(block: tuple[str, list[str]], model: str) -> bool:
    language, lines = block
    if language not in _SHELL_LANGUAGES:
        return False
    for line in _logical_lines(lines):
        if line.startswith("$ "):
            line = line[2:]
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if len(tokens) >= 3 and tokens[0] == "vllm" and tokens[1] == "serve" and hf_id_key(tokens[2]) == hf_id_key(model):
            return True
    return False


def _clean_prose(text: str) -> str:
    return " ".join(text.replace("**", "").replace("__", "").replace("`", "").split())


def _other_model(text: str, model: str) -> bool:
    """Whether `text` holds a model reference that is not `model` (see the module text)."""
    own = {hf_id_key(model), hf_id_key(model.split("/", 1)[1])}
    text = _NAMED_PACKAGE_RE.sub(" ", _ADDRESS_RE.sub(" ", _clean_prose(text)))
    for token in _TOKEN_RE.findall(text):
        if hf_id_key(token) in own or _VERSION_RE.fullmatch(token):
            continue
        if _SIZE_RE.fullmatch(token) or _JOINED_RE.fullmatch(token):
            return True
    return False


def _packages(paragraphs: list[tuple[int, str, str, bool]], model: str) -> set[str]:
    name = model.split("/", 1)[1]
    word = re.match(r"[A-Za-z]+", name)
    subjects = ["this model", "the model", "it", re.escape(name)] + ([re.escape(word.group(0))] if word else [])
    rule = re.compile(
        r"^" + _LEAD_IN + r"(?P<subject>" + "|".join(subjects) + r")\s+(?:also\s+)?(?:requires?|needs?)\s+the\s+(?P<pkg>" + _PACKAGE + r")\s+(?:python\s+)?package\b",
        re.IGNORECASE,
    )
    found: dict[str, str] = {}
    refers = re.compile(r"^(?:(?:" + "|".join(subjects) + r")(?:\s+(?:with|in|using|on)\s+vllm)?)?$", re.IGNORECASE)  # what a lead-in may name
    shared = {section for section, heading, text, _ in paragraphs if _other_model(heading, model) or _other_model(text, model)}  # sections with another model
    for section, heading, paragraph, prose in paragraphs:
        if not prose or _TRAINING_RE.search(heading):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", _clean_prose(paragraph)):
            match = rule.match(sentence)
            if match is not None and not refers.fullmatch((match.group("lead") or "").strip()):
                continue  # the lead-in names another model or something else: not a statement about this one
            if match is None or _NEGATION_RE.search(sentence) or _TRAINING_RE.search(sentence):
                continue
            if section in shared and hf_id_key(match.group("subject")) != hf_id_key(name):
                continue  # a pronoun or the leading word where another model is near: in doubt, no name
            if _CONTEXT_RE.search(sentence[: match.start("pkg")] + sentence[match.end("pkg") :]):
                found.setdefault(match.group("pkg").casefold(), match.group("pkg"))
    return set(found.values())


def detect(markdown: str, model: str) -> Detection:
    """What the rule finds in a card: see the module text."""
    blocks, paragraphs = _split(_front_matter_removed(markdown.replace("\r\n", "\n").replace("\r", "\n")))
    if not any(_serves(block, model) for block in blocks):
        return Detection("none_recognized")
    names = _packages(paragraphs, model)
    return Detection("instructions", next(iter(names)) if len(names) == 1 else None)


def note_of(model: str, found: Detection, revision: str, today: str | date) -> SetupNote:
    """The note a card read at `revision` makes."""
    return SetupNote(
        model=model, status=found.status, card_revision=revision, source_url=f"https://huggingface.co/{model}/blob/{revision}/README.md",
        checked=today, package=found.package,
    )


def not_read(model: str, reason: str, today: str | date) -> SetupNote:
    """The report of a card that could not be read: never stored, it changes no stored note."""
    return SetupNote(model=model, status="not_read", source_url=f"https://huggingface.co/{model}", checked=today, reason=_clean(reason))
