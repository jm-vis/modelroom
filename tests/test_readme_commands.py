"""Every `modelroom ...` command the README shows as an instruction to type has to be one
`build_parser().parse_args` actually accepts -- otherwise a reader's first copy-paste fails at the
argument check. Codex r1 finding 3 (AP9-FR FR-C, round 2): the "Another computer" section showed
`modelroom hardware` without the `--config` `hardware_parser` requires, and `export-profile`
without it too at one of its two mentions.

Scope: the README's own "Another computer" section (Step 1's five ways to reach another
computer), the one Codex reviewed. A placeholder such as `<name>` or `<share>/<name>.json` stands
for a value none of these subcommands judges by content -- `_safe_argv` gives it a plain word, or
a number where an argument's own `type=int` needs one (`--weights-bytes`).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from modelroom.cli import build_parser

REPO = Path(__file__).resolve().parent.parent
SECTION_START = "#### Another computer"
SECTION_END = "### Step 2, Packages"

_PLACEHOLDER = re.compile(r"<[^>]+>")
_WEIGHTS_BYTES = re.compile(r"(--weights-bytes)\s+<[^>]+>")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_SUBCOMMANDS = ("fetch", "hardware", "render", "migrate", "export-profile", "import-profile", "pull", "remote")
_COMMAND_START = re.compile(r"^modelroom\s+(?:" + "|".join(_SUBCOMMANDS) + r")\b")


def _section_text() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start = text.index(SECTION_START)
    end = text.index(SECTION_END, start)
    return text[start:end]


def commands_in_section() -> list[str]:
    """Every backtick span of the section whose whitespace-collapsed text is a full `modelroom
    <subcommand> ...` invocation -- not a bare mention (`modelroom.toml`, `` `~/.ssh/config` ``)
    and not a top-level flag with no subcommand (`modelroom --answers <file>` is outside this
    section). A span itself may wrap across two source lines of the README; the inline newline and
    its leading spaces collapse to one space first, the way a reader's eye already reads it."""
    commands = []
    for span in _CODE_SPAN.findall(_section_text()):
        collapsed = " ".join(span.split())
        if _COMMAND_START.match(collapsed):
            commands.append(collapsed)
    return commands


def _safe_argv(command: str) -> list[str]:
    command = _WEIGHTS_BYTES.sub(r"\1 123", command)
    command = _PLACEHOLDER.sub("x", command)
    return shlex.split(command)


_COMMANDS = commands_in_section()


def test_the_section_still_shows_a_command_of_each_kind_this_test_covers():
    """A future edit that removed every command from the section would make the test below
    vacuously green; this fails first instead."""
    kinds = {command.split()[1] for command in _COMMANDS}
    assert kinds >= {"hardware", "export-profile", "pull"}


@pytest.mark.parametrize("command", _COMMANDS, ids=_COMMANDS)
def test_a_readme_command_of_another_computer_is_accepted_by_the_parser(command):
    argv = _safe_argv(command)
    assert argv[0] == "modelroom"
    build_parser().parse_args(argv[1:])  # raises SystemExit (caught by pytest as a failure) on any argparse error
