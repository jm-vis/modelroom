"""The setup note of a model card: what the rule of recognition says (AP9-KO, V7, decision E4-B). Pure: no request, no file.

`detect(markdown, model)` reads the Markdown text of the card of the publisher's repository `model` and answers with a
`Detection`. The rule is finite and written down here, with positive and negative fixtures in `tests/test_ko_p4_card.py`.
Nothing of the card is run or followed; the answer is a status, at most the name of one package and at most four switches
of a closed list (copied as checked tokens, never as text of the card).

**Status `instructions`.** The card has a fenced code block (a shell language, or none) with a logical line (backslash
continuations joined, a leading `$ ` allowed) that is `vllm serve <model>` followed by anything, where `<model>` is the
first argument after `serve` and equals the id of this model as a whole, without case. A block for another model (or for
a variant such as `<model>-FP8`, or a path) does not count; neither does a mention in running text. Nothing found: `none_recognized`,
which says only that this rule found nothing -- never that nothing is required.

**The switches** (`Detection.vllm_args`, since schema 4) are read from those very lines, the ones for this model and no
other line of the block: of the closed list `--reasoning-parser <name>`, `--tool-call-parser <name>`,
`--enable-auto-tool-choice` and `--kv-cache-dtype <value>`, in the forms `--x value` and `--x=value`, of a line that holds no shell operator, bracket,
substitution, comment, quote, non-ASCII character or ASCII control character (`\x00-\x08`, `\x0a-\x1f`, `\x7f`; only space and
tab are plain white space; a carriage return never reaches a line, `detect` turns it into a line break first), anywhere in a start
line of this model, also at its very end: no switch at all. A value is `[A-Za-z0-9][A-Za-z0-9_.-]{0,39}`; a switch with another value, with
none, or with two different values in the lines of the card is left out, and so is every switch not on the list (a path,
`--trust-remote-code`, a memory or parallelism switch, an environment variable). The note stands without a switch that is
left out; the switches are the card's own words for this model and nothing of them is run.

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

**Limit of the rule** (decision of the management, the measure of V5): it guards against realistic forms of model cards,
not against a card written to deceive it. Of a card longer than the window the caller hands over the start alone and
takes a positive finding only (`catalog_scan_packages`).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import date

from modelroom.catalog_findings import VLLM_FLAG_SWITCHES, VLLM_VALUE_RE, VLLM_VALUE_SWITCHES, SetupNote
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


_UNSAFE_RE = re.compile("[;&|<>()$`#'\"\\\\]|[^\\x00-\\x7f]|[\\x00-\\x08\\x0a-\\x1f\\x7f]")  # anything a shell reads as more than words; see `_switches`
_SWITCH_ORDER = ("--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice", "--kv-cache-dtype")


@dataclass(frozen=True)
class Detection:
    status: str  # "instructions" or "none_recognized"
    package: str | None = None
    vllm_args: tuple[str, ...] = ()  # the switches of the list (`catalog_findings.VLLM_*_SWITCHES`) that the start lines of this model name


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


def whole_paragraphs(text: str) -> str:
    """The start of a cut card up to its last whole paragraph: what follows the last blank line, heading or code block
    that ended -- a paragraph or a code block still open at the cut -- goes, since it may go on with what limits it."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")[:-1]  # the text ends at a line end: the last is empty
    boundary, fence = 0, None
    for number, line in enumerate(lines):
        if fence is not None:
            if re.match(r"^ {0,3}(" + re.escape(fence[0]) + "{" + str(fence[1]) + r",})[ \t]*$", line):
                fence, boundary = None, number + 1
            continue
        opening = _FENCE_RE.match(line)
        if opening:
            fence, boundary = (opening.group(1)[0], len(opening.group(1))), number
        elif not line.strip():
            boundary = number
        elif _HEADING_RE.match(line):
            boundary = number + 1
    return "".join(line + "\n" for line in lines[:boundary])


def _logical_lines(lines: list[str]) -> list[str]:
    joined, carry = [], ""
    for line in lines:
        stripped = line.rstrip(" \t")  # only space and tab: a control character at an edge must stay for the check of `_switches`
        if stripped.endswith("\\"):
            carry += stripped[:-1] + " "
            continue
        joined.append((carry + stripped).strip(" \t"))
        carry = ""
    if carry.strip(" \t"):
        joined.append(carry.strip(" \t"))
    return joined


def _start_lines(block: tuple[str, list[str]], model: str) -> list[str]:
    """Every logical line of `block` (continuations joined, a leading `$ ` removed) that is `vllm serve <model>` for exactly this model."""
    language, lines = block
    if language not in _SHELL_LANGUAGES:
        return []
    found = []
    for line in _logical_lines(lines):
        if line.startswith("$ "):
            line = line[2:]
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if len(tokens) >= 3 and tokens[0] == "vllm" and tokens[1] == "serve" and hf_id_key(tokens[2]) == hf_id_key(model):
            found.append(line)
    return found


def _named(line: str) -> tuple[dict[str, str], set[str]]:
    """What one start line says of the listed switches: `{switch: value}` (`""` for the flag) of those with a sound value,
    and the switches named with a broken one. The line is cut at white space and each word is taken whole: `--x value`, or
    `--x=value` split at the first `=`; a value that starts with `-` is no value, and one off the rule drops its switch,
    it is never cut to a valid part."""
    words = line.split()
    sound: dict[str, str] = {}
    broken: set[str] = set()
    index = 3
    while index < len(words):
        name, equals, inline = words[index].partition("=")
        index += 1
        if name in VLLM_FLAG_SWITCHES:
            if equals or (index < len(words) and not words[index].startswith("-")):
                broken.add(name)  # a flag takes no value: `=x` or a word that is no switch behind it drops it
            else:
                sound[name] = ""
        elif name in VLLM_VALUE_SWITCHES:
            if equals:
                value = inline
            elif index < len(words) and not words[index].startswith("-"):
                value, index = words[index], index + 1
            else:
                value = ""
            if VLLM_VALUE_RE.fullmatch(value) is None:
                broken.add(name)
            elif sound.setdefault(name, value) != value:
                broken.add(name)  # the same switch with two values in one line
    return sound, broken


def _switches(starts: list[str]) -> tuple[str, ...]:
    """The switches of the list that the start lines of this model name, in the fixed order of `_SWITCH_ORDER`.

    One strict rule, in doubt none: a line with a shell operator, bracket, substitution, comment, quote, non-ASCII or ASCII
    control character (only space and tab are white space; `\\r` is a line break before this point) anywhere, ends included, gives no switch to the note at all (the note stays `instructions`); a switch named with a broken value, or with
    two values in the lines of the card, is left out."""
    if any(_UNSAFE_RE.search(line) for line in starts):
        return ()
    values: dict[str, set[str]] = {}
    broken: set[str] = set()
    for line in starts:
        sound, bad = _named(line)
        broken |= bad
        for name, value in sound.items():
            values.setdefault(name, set()).add(value)
    kept = [name for name in _SWITCH_ORDER if name in values and name not in broken and len(values[name]) == 1]
    return tuple(name if name in VLLM_FLAG_SWITCHES else f"{name} {next(iter(values[name]))}" for name in kept)


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
    starts = [tokens for block in blocks for tokens in _start_lines(block, model)]
    if not starts:
        return Detection("none_recognized")
    names = _packages(paragraphs, model)
    return Detection("instructions", next(iter(names)) if len(names) == 1 else None, _switches(starts))


def note_of(model: str, found: Detection, revision: str, today: str | date) -> SetupNote:
    """The note a card read at `revision` makes."""
    return SetupNote(
        model=model, status=found.status, card_revision=revision, source_url=f"https://huggingface.co/{model}/blob/{revision}/README.md",
        checked=today, package=found.package, vllm_args=list(found.vllm_args),
    )


def not_read(model: str, reason: str, today: str | date) -> SetupNote:
    """The report of a card that could not be read: never stored, it changes no stored note."""
    return SetupNote(model=model, status="not_read", source_url=f"https://huggingface.co/{model}", checked=today, reason=_clean(reason))
