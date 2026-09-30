"""Every `modelroom ...` command the README shows as an instruction to type has to be one
`build_parser().parse_args` actually accepts -- otherwise a reader's first copy-paste fails at the
argument check. Codex r1 finding 3 (AP9-FR FR-C, round 2): the "Another computer" section showed
`modelroom hardware` without the `--config` `hardware_parser` requires, and `export-profile`
without it too at one of its two mentions.

Scope: the README's own "Another computer" section, the one Codex reviewed. A placeholder such as
`<name>` or `<share>/<name>.json` stands for a value none of these subcommands judges by content --
`_safe_argv` gives it a plain word, or a number where an argument's own `type=int` needs one
(`--weights-bytes`).

What counts as a command (Codex r2 finding 2, round 3): every `modelroom` followed by more text
inside code -- an inline code span, even one wrapped across two source lines, or any line of a
fenced block, with or without a language tag. It is read when it stands where a shell reads a
program name: at the start, after a prompt or a separator (`$`, `&&`, `;`, `|`), after a path
(`~/.local/bin/modelroom`), quoted (`"modelroom"`), after a runner (`uvx`), or run on another
computer after `ssh <alias>`. Where it is the package's name -- `uv tool install modelroom --force`,
`pip install`, `pipx install`, the value of `--from` -- it is no command. A `modelroom` in code
anywhere else, an unclosed fence, and a `modelroom` followed by a subcommand or an option in plain
prose are all reported as unreadable, with their line, and fail the test instead of being skipped.
Plain prose that only names modelroom ("modelroom keeps whatever you type") is not a command.
"""

from __future__ import annotations

import argparse
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest

from modelroom.cli import build_parser

REPO = Path(__file__).resolve().parent.parent
SECTION_START = "#### Another computer"
SECTION_END = "### Step 2, Packages"

_PLACEHOLDER = re.compile(r"<[^>]+>")
_WEIGHTS_BYTES = re.compile(r"(--weights-bytes)\s+<[^>]+>")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_BACKTICK_RUN = re.compile(r"`+")
# The program word: not part of a file name (`modelroom.toml`) or a longer word, and followed by
# more text on its own line or span -- a bare `uv tool install modelroom` passes nothing to it. A
# shell drops quotes around a program name, so `"modelroom" hardware` is the same call.
_PROGRAM = re.compile(r"(?<![\w.\-])[\"']?modelroom[\"']?(?=[ \t]+\S)")
_SEPARATOR = re.compile(r"&&|\|\||[;|]")
_PROMPT = re.compile(r"^(?:\$|>|PS>)\s*")
_SSH = re.compile(r"^ssh(?:\s+\S+)+$")
# A runner that starts the program itself: `uvx modelroom ...`, `uv tool run --from modelroom modelroom ...`.
_RUNNER = re.compile(r"^(?:uvx|uv\s+tool\s+run)(?:\s+--from\s+\S+)?$")
# Where the word is the package's name, not the program: an install or removal, or the value of
# `--from`, whatever options follow it.
_PACKAGE = re.compile(
    r"^(?:uv\s+tool\s+(?:install|upgrade|uninstall)|uv\s+(?:add|remove)|uv\s+pip\s+(?:install|uninstall)"
    r"|(?:python3?\s+-m\s+)?pip3?\s+(?:install|uninstall)|pipx\s+(?:install|upgrade|uninstall|reinstall))\b.*$"
    r"|^(?:uvx|uv\s+tool\s+run)\b.*\s--from$"
)


def _subcommands() -> frozenset[str]:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return frozenset(action.choices)
    return frozenset()


SUBCOMMANDS = _subcommands()
_PROSE_COMMAND = re.compile(
    r"(?<![\w.\-])modelroom\s+(?:(?:" + "|".join(re.escape(name) for name in sorted(SUBCOMMANDS)) + r")\b|--?[A-Za-z])"
)  # an option, not the README's own ` -- ` dash


@dataclass(frozen=True)
class Invocation:
    line: int
    command: str  # from `modelroom` on, whitespace collapsed


@dataclass(frozen=True)
class Unreadable:
    line: int
    text: str
    reason: str


def _read_code(code: str, line: int, invocations: list, unreadable: list) -> None:
    """One line of a fenced block, or one whole inline span with its whitespace collapsed."""
    for match in _PROGRAM.finditer(code):
        before = code[: match.start()]
        segment = _SEPARATOR.split(before)[-1]
        segment = re.sub(r"\S*[/\\]$", "", segment).strip()  # `~/.local/bin/` before the word
        segment = _PROMPT.sub("", segment)
        if _PACKAGE.match(segment):
            continue
        if segment and not _SSH.match(segment) and not _RUNNER.match(segment):
            unreadable.append(Unreadable(line, code, f"modelroom after {segment!r} is no command position"))
            continue
        rest = code[match.end() :]
        rest = _SEPARATOR.split(rest)[0]
        rest = re.split(r"\s#", rest)[0]
        invocations.append(Invocation(line, " ".join(("modelroom" + rest).split())))


def _read_prose(text: str, first_line: int, invocations: list, unreadable: list) -> None:
    """Text outside fences: inline code spans (CommonMark: a backtick run closed by a run of the
    same length) are read as code; what is left is prose."""
    prose = []
    pos = 0
    while True:
        opening = _BACKTICK_RUN.search(text, pos)
        if opening is None:
            prose.append(text[pos:])
            break
        closing = next(
            (run for run in _BACKTICK_RUN.finditer(text, opening.end()) if len(run.group()) == len(opening.group())),
            None,
        )
        if closing is None:  # a lone run is a literal backtick, not a span
            prose.append(text[pos : opening.end()])
            pos = opening.end()
            continue
        prose.append(text[pos : opening.start()])
        prose.append(" " * (closing.end() - opening.start()))  # keep offsets for the prose scan
        line = first_line + text.count("\n", 0, opening.start())
        _read_code(" ".join(text[opening.end() : closing.start()].split()), line, invocations, unreadable)
        pos = closing.end()
    plain = "".join(prose)
    for match in _PROSE_COMMAND.finditer(plain):
        line = first_line + plain.count("\n", 0, match.start())
        unreadable.append(Unreadable(line, match.group(), "a command outside code"))


def extract(markdown: str, first_line: int = 1) -> tuple[list[Invocation], list[Unreadable]]:
    """Every `modelroom` invocation in `markdown`, and every place that shows one this reader cannot
    check. `first_line` is the source line number of the text's first line."""
    invocations: list[Invocation] = []
    unreadable: list[Unreadable] = []
    lines = markdown.split("\n")
    prose: list[str] = []
    prose_start = first_line
    index = 0
    while index < len(lines):
        fence = _FENCE.match(lines[index])
        if fence is None:
            if not prose:
                prose_start = first_line + index
            prose.append(lines[index])
            index += 1
            continue
        _read_prose("\n".join(prose), prose_start, invocations, unreadable)
        prose = []
        marker = fence.group(1)
        closing = re.compile(r"^[ \t]*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*$")
        end = next((i for i in range(index + 1, len(lines)) if closing.match(lines[i])), None)
        if end is None:
            unreadable.append(Unreadable(first_line + index, lines[index].strip(), "a fence that is never closed"))
            return invocations, unreadable
        for offset in range(index + 1, end):
            _read_code(lines[offset], first_line + offset, invocations, unreadable)
        index = end + 1
    _read_prose("\n".join(prose), prose_start, invocations, unreadable)
    return invocations, unreadable


def _safe_argv(command: str) -> list[str]:
    command = _WEIGHTS_BYTES.sub(r"\1 123", command)
    command = _PLACEHOLDER.sub("x", command)
    return shlex.split(command)


def problem_with(command: str) -> str | None:
    """Why `command` would fail at the argument check, or None when the parser accepts it."""
    argv = _safe_argv(command)
    words = [word for word in argv[1:] if not word.startswith("-")]
    if words and argv[1] == words[0] and words[0] not in SUBCOMMANDS:
        return f"unknown subcommand {words[0]!r}"
    try:
        build_parser().parse_args(argv[1:])
    except SystemExit as exc:
        return f"rejected by the parser (exit {exc.code})"
    return None


def _section() -> tuple[str, int]:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start = text.index(SECTION_START)
    end = text.index(SECTION_END, start)
    return text[start:end], text.count("\n", 0, start) + 1


_INVOCATIONS, _UNREADABLE = extract(*_section())


def test_the_section_shows_no_command_this_test_cannot_read():
    assert not _UNREADABLE, "\n".join(f"README.md:{u.line}: {u.reason}: {u.text}" for u in _UNREADABLE)


def test_the_section_still_shows_a_command_of_each_kind_this_test_covers():
    """A future edit that removed every command from the section would make the test below
    vacuously green; this fails first instead."""
    kinds = {invocation.command.split()[1] for invocation in _INVOCATIONS}
    assert kinds >= {"hardware", "export-profile", "pull"}


@pytest.mark.parametrize("invocation", _INVOCATIONS, ids=[f"{i.line}:{i.command}" for i in _INVOCATIONS])
def test_a_readme_command_of_another_computer_is_accepted_by_the_parser(invocation):
    problem = problem_with(invocation.command)
    assert problem is None, f"README.md:{invocation.line}: {invocation.command}: {problem}"


# --- the reader itself, on small examples ---


def _commands(markdown: str) -> list[str]:
    invocations, unreadable = extract(markdown)
    assert not unreadable, unreadable
    return [invocation.command for invocation in invocations]


def test_an_inline_span_wrapped_across_two_lines_is_one_command():
    assert _commands("Run `modelroom hardware\n  --config modelroom.toml` there.") == [
        "modelroom hardware --config modelroom.toml"
    ]


def test_a_fenced_block_without_a_language_tag_is_read():
    assert _commands("Text.\n\n```\nmodelroom render --config c.toml\n```\n") == ["modelroom render --config c.toml"]


def test_a_fenced_block_with_a_language_tag_is_read():
    markdown = "```sh\n$ modelroom fetch --config c.toml --machine m\n```\n"
    assert _commands(markdown) == ["modelroom fetch --config c.toml --machine m"]


def test_an_indented_tilde_fence_in_a_list_is_read():
    markdown = "- Step:\n\n  ~~~bash\n  modelroom hardware --config c.toml\n  ~~~\n"
    assert _commands(markdown) == ["modelroom hardware --config c.toml"]


def test_a_command_run_over_ssh_is_read_inline_and_in_a_block():
    markdown = "Try `ssh gpu1 modelroom hardware --config c.toml`.\n\n```sh\nssh -p 2222 gpu1 modelroom pull x\n```\n"
    assert _commands(markdown) == ["modelroom hardware --config c.toml", "modelroom pull x"]


def test_a_command_behind_a_path_or_a_separator_is_read():
    markdown = "`~/.local/bin/modelroom hardware --config c.toml`\n\n```\ncd there && modelroom render --config c.toml\n```\n"
    assert _commands(markdown) == ["modelroom hardware --config c.toml", "modelroom render --config c.toml"]


def test_the_line_of_each_command_is_its_source_line():
    invocations, _ = extract("a\n\n```sh\nmodelroom render --config c\n```\nb `modelroom pull x`\n", first_line=10)
    assert [invocation.line for invocation in invocations] == [13, 15]


def test_an_unknown_subcommand_is_read_and_then_rejected():
    assert _commands("`modelroom hardwre --config c.toml`") == ["modelroom hardwre --config c.toml"]
    assert problem_with("modelroom hardwre --config c.toml") == "unknown subcommand 'hardwre'"


def test_a_missing_required_option_is_rejected():
    assert problem_with("modelroom hardware") is not None
    assert problem_with("modelroom hardware --config modelroom.toml") is None


def test_modelroom_in_code_at_no_command_position_is_reported_with_its_line():
    invocations, unreadable = extract("x\n`echo modelroom --help`\n")
    assert invocations == []
    assert [(u.line, u.reason) for u in unreadable] == [(2, "modelroom after 'echo' is no command position")]


def test_a_quoted_program_name_is_read():
    markdown = "```sh\n\"modelroom\" hardware\n'modelroom' hardware --config c.toml\n```\n"
    assert _commands(markdown) == ["modelroom hardware", "modelroom hardware --config c.toml"]
    assert problem_with("modelroom hardware") is not None


def test_the_package_name_after_an_install_command_is_no_command():
    markdown = (
        "```sh\n"
        "uv tool install modelroom --force\n"
        "uv tool upgrade modelroom --reinstall\n"
        "pip install modelroom --upgrade\n"
        "python -m pip install modelroom -U\n"
        "pipx install modelroom --force\n"
        "uvx --from modelroom modelroom hardware --config c.toml\n"
        "```\n"
    )
    assert _commands(markdown) == ["modelroom hardware --config c.toml"]


def test_an_unclosed_fence_is_reported_with_its_line():
    _, unreadable = extract("x\n```sh\nmodelroom hardware --config c\n")
    assert [(u.line, u.reason) for u in unreadable] == [(2, "a fence that is never closed")]


def test_a_command_in_plain_prose_is_reported_with_its_line():
    _, unreadable = extract("One line.\nThen run modelroom hardware on it.\n")
    assert [(u.line, u.reason) for u in unreadable] == [(2, "a command outside code")]


def test_prose_and_code_that_only_name_modelroom_are_not_commands():
    markdown = (
        "modelroom keeps whatever you type -- modelroom -- and modelroom 0.2.0 or newer is needed.\n"
        "`modelroom.toml`, `~/.local/bin/modelroom`, `uv tool install modelroom`\n"
        "```toml\n[modelroom]\n```\n"
    )
    assert extract(markdown) == ([], [])
