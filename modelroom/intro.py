"""The start screen of the guided mode, its step heads, and the two decisions every other
module of the dialog reads from here: which glyphs this terminal can print, and whether the
output may carry color.

Both decisions are made in this one module and nowhere else, so the scale, the selection lists
and the result view cannot disagree about what a terminal understands. Color comes from
`prompt_toolkit` (`print_formatted_text`, `Style`), which the dialog library already brings;
nothing else is imported for it. Without a terminal, with `NO_COLOR` set, and under
`modelroom --answers <file>`, the very same lines are printed through the run's own `out`
without any color at all.

The start screen shows the mark, the version, and the three facts that are known before the
first question: the results folder (when a `--config` or the pointer file already names one),
the local Ollama daemon, and this machine. See CONTRACTS.md, "Guided mode", "The start screen".
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import modelroom

from .binding import PointerFileError, read_pointer
from .config import ConfigError, load_config
from .contracts import SchemaVersionError
from .daemon import TAGS_PATH, VERSION_PATH, Daemon, DaemonError
from .importer import scan_profiles
from .measure import Probes
from .profile import HardwareProfile

# The mark's colors, from `docs/assets/banner.svg`: white and a muted blue for the pictogram and
# the name, green for the pointer and for what is done, amber for what is tight, cyan for a path,
# gray for a side note. They are foreground colors only -- the background belongs to the terminal.
WHITE = "#ffffff"
MUTED = "#8cacc9"
MUTED_DARK = "#5a7c99"
GREEN = "#3fa58e"
AMBER = "#d97706"
CYAN = "#0891b2"
GRAY = "#808080"

TAGLINE = "Which local model packages fit your machine."
STEP_NAMES = ("Configuration", "Packages", "Context", "Measurement", "Results")
STEP_COUNT = len(STEP_NAMES)
# `Esc` is bound for every selection list (`dialog.TerminalAsker`); the two text questions end on
# Ctrl-C like every other command, so the line says both instead of promising one for everything.
# And the first sentence promises nothing about *when* files are written, because the run cannot
# keep such a promise: step 1 writes a configuration as soon as the folder is known, a schema-1
# configuration is migrated on the way in, and the pointer file that remembers the folder is
# written without a line of its own. What a reader needs to know is that this is not all-or-nothing
# at the end -- a run left in the middle has written what the steps before it wrote.
CLOSING_LINES = (
    "Five steps: " + ", ".join(name.lower() for name in STEP_NAMES) + ".",
    "Files are written as the run goes on. Esc leaves a list, Ctrl-C leaves at any point.",
)

# The pictogram of the mark, three rows of four cells: `f` a filled cell, `m` a muted one, `x` an
# empty one. Same arrangement as the twelve rectangles of `docs/assets/banner.svg`.
PICTOGRAM = (("f", "f", "m", "f"), ("f", "f", "f", "x"), ("m", "f", "x", "x"))

_LABEL_WIDTH = 8
_VALUE_WIDTH = 22
# Two spaces stand between the columns of a fact row, so a value wider than its column pushes the
# note along instead of growing into it. A results folder of 120 characters is an ordinary value.
_GAP = "  "


@dataclass(frozen=True)
class Glyphs:
    """The characters the dialog draws with, in a set this terminal can encode."""

    full: str
    muted: str
    empty: str
    check: str
    pointer: str
    move: str
    dot: str
    rule: str


UNICODE_GLYPHS = Glyphs(full="██", muted="▓▓", empty="░░", check="✓",
                        pointer="❯", move="↑↓", dot="·", rule="─")
ASCII_GLYPHS = Glyphs(full="##", muted="::", empty="..", check="ok", pointer=">", move="^v", dot="-", rule="-")


def glyphs(stream=None) -> Glyphs:
    """The glyph set for `stream` (stdout by default): the drawn one, or ASCII.

    A Windows console under `cp1252` encodes none of the block or arrow characters, and a
    `print` of them would end the run with a `UnicodeEncodeError` instead of a dialog. So the
    question is asked of the stream itself, once, and the answer is the same for every module.
    """
    target = sys.stdout if stream is None else stream
    encoding = getattr(target, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        return ASCII_GLYPHS
    try:
        "".join(vars(UNICODE_GLYPHS).values()).encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return ASCII_GLYPHS
    return UNICODE_GLYPHS


def use_color(stream=None) -> bool:
    """Whether output to `stream` may carry color: a terminal, and no `NO_COLOR` in the environment."""
    if os.environ.get("NO_COLOR"):
        return False
    target = sys.stdout if stream is None else stream
    try:
        return bool(target.isatty())
    except (AttributeError, ValueError):  # a closed or replaced stream
        return False


def style_rules() -> list[tuple[str, str]]:
    """The one style of the guided mode, as `(class name, style)` pairs.

    `dialog.py` hands these to `questionary` for every question and this module to
    `prompt_toolkit` for the start screen, so a pointer is the same green in both.
    """
    return [
        ("mark", f"fg:{WHITE} bold"),
        ("mark-muted", f"fg:{MUTED}"),
        ("mark-empty", f"fg:{MUTED_DARK}"),
        ("tagline", f"fg:{MUTED}"),
        ("label", f"fg:{MUTED}"),
        ("value", f"fg:{WHITE}"),
        ("note", f"fg:{GRAY}"),
        ("path", f"fg:{CYAN}"),
        ("done", f"fg:{GREEN}"),
        ("tight", f"fg:{AMBER}"),
    ]


@dataclass(frozen=True)
class Fact:
    """One row of the start screen: what it is about, what it says, and the note next to it."""

    label: str
    value: str
    note: str


@dataclass(frozen=True)
class Intro:
    """Everything the start screen prints, already read: nothing is looked up while printing."""

    version: str
    source: str | None
    facts: tuple[Fact, ...]


def step_head(number: int) -> str:
    """The head every step begins with: `Step 3 of 5  Context`."""
    return f"Step {number} of {STEP_COUNT}  {STEP_NAMES[number - 1]}"


def done_line(number: int, summary: str, stream=None) -> str:
    """The one line a finished step leaves behind: a check mark, its name and its answer."""
    return f"{glyphs(stream).check} {STEP_NAMES[number - 1].ljust(_LABEL_WIDTH + 3)} {summary}"


def intro_lines(intro: Intro, stream=None) -> list[list[tuple[str, str]]]:
    """The start screen as lines of `(class name, text)` fragments -- one list per line."""
    marks = glyphs(stream)
    version = f"modelroom {intro.version}"
    right = [
        [("class:mark", "ModelRoom")],
        [("class:tagline", TAGLINE)],
        [("class:note", version if intro.source is None else f"{version} {marks.dot} {intro.source}")],
    ]
    lines = [[*_pictogram_row(row, marks), ("", "   ")] + fragments for row, fragments in zip(PICTOGRAM, right)]
    lines.append([])
    lines += [_fact_line(fact) for fact in intro.facts]
    lines.append([])
    lines += [[("class:note", text)] for text in CLOSING_LINES]
    return lines


def print_intro(intro: Intro, out: Callable[[str], None], stream=None, colored: bool | None = None) -> None:
    """Print the start screen: in color at a terminal, else the same lines through `out`.

    `colored` is the caller's own answer to that question -- `modelroom --answers <file>` passes
    `False`, because a run that answers from a file is read from a log, not from a screen, and its
    output has to be the same wherever it runs. `None` asks the stream (`use_color`).
    """
    lines = intro_lines(intro, stream)
    if colored is None:
        colored = use_color(stream)
    if colored and _print_in_color(lines):
        return
    for line in lines:
        out(" " + "".join(text for _class, text in line) if line else "")


def _print_in_color(lines: list[list[tuple[str, str]]]) -> bool:
    """Print the lines through `prompt_toolkit`; `False` when it cannot drive this console.

    Measured case: a Windows Python under a terminal that says `xterm-256color` (a Git Bash or
    MSYS window), where `prompt_toolkit` looks for a Windows console and raises
    `NoConsoleScreenBufferError`. The start screen is the first thing a run prints, and a
    traceback in its place would be the last thing a user of the guided mode needs -- so
    whatever the library cannot do here, the caller prints as plain lines instead. Nothing is
    swallowed: either these lines are on screen in color, or the same lines are without it.
    """
    from prompt_toolkit import print_formatted_text
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.styles import Style

    style = Style(style_rules())
    try:
        for line in lines:
            print_formatted_text(FormattedText([("", " "), *line]), style=style)
    except Exception:  # noqa: BLE001 -- the library's own failures are not one class
        return False
    return True


def _pictogram_row(row: Sequence[str], marks: Glyphs) -> list[tuple[str, str]]:
    cells = {"f": ("class:mark", marks.full), "m": ("class:mark-muted", marks.muted), "x": ("class:mark-empty", marks.empty)}
    fragments: list[tuple[str, str]] = []
    for index, cell in enumerate(row):
        if index:
            fragments.append(("", " "))
        fragments.append(cells[cell])
    return fragments


def _fact_line(fact: Fact) -> list[tuple[str, str]]:
    # The folder row carries a path, and a path is cyan wherever this dialog prints one.
    style = "class:path" if fact.label == "folder" else "class:value"
    return [
        ("class:label", fact.label.ljust(_LABEL_WIDTH)),
        ("", _GAP),
        (style, fact.value.ljust(_VALUE_WIDTH)),
        ("", _GAP),
        ("class:note", fact.note),
    ]


# --- reading the three facts ---------------------------------------------------------------------


def collect_intro(daemon: Daemon, probes: Probes, pointer_path: Path, config_file: Path | None) -> Intro:
    """Read what the start screen shows. Every fact that cannot be read says so; nothing raises.

    `config_file` is the configuration this run already knows about -- from `--config` or from
    the pointer file -- or `None`, in which case the first question is still to come and the
    folder row says so.
    """
    return Intro(
        version=modelroom.__version__,
        source=source_url(),
        facts=(_folder_fact(config_file), _daemon_fact(daemon), _machine_fact(probes, pointer_path, config_file)),
    )


def source_url() -> str | None:
    """The repository this package names in its metadata (`project.urls`), without the scheme."""
    try:
        entries = importlib.metadata.metadata("modelroom").get_all("Project-URL") or []
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - the package is installed
        return None
    for entry in entries:
        name, _comma, url = str(entry).partition(",")
        if name.strip().lower() == "source":
            return url.strip().removeprefix("https://").removeprefix("http://")
    return None


def hardware_words(profile: HardwareProfile) -> str:
    """One machine's hardware in plain words: the graphics card, and the system memory."""
    if profile.gpu_state == "measured" and profile.vram_gib is not None:
        card = f"one graphics card, {profile.vram_gib:.0f} GB"
    elif profile.gpu_state == "none":
        card = "no graphics card"
    else:
        card = "the graphics memory was not measured"
    memory = "system memory unknown" if profile.ram_physical_gib is None else f"{profile.ram_physical_gib:.0f} GB memory"
    return f"{card}, {memory}"


def _folder_fact(config_file: Path | None) -> Fact:
    if config_file is None:
        return Fact("folder", "not chosen yet", "the first question asks where results live")
    folder = config_file.parent
    written = [name for name in ("state", "docs") if (folder / name).is_dir()]
    note = "a configuration, and results from an earlier run" if written else "a configuration, nothing rendered yet"
    return Fact("folder", str(folder), note)


def _daemon_fact(daemon: Daemon) -> Fact:
    """The local Ollama daemon: its version, and how many models it has.

    Read here and not through `loadtest.py`: the start screen must never end a run, and the load
    test's reader raises for a daemon that answers something unexpected. Two short calls, both
    from the five this package ever makes (`daemon.ALLOWED_CALLS`).
    """
    version = _daemon_value(daemon, VERSION_PATH, "version")
    if version is None:
        return Fact("daemon", "Ollama", "not reachable")
    models = _daemon_value(daemon, TAGS_PATH, "models")
    if not isinstance(models, list):
        return Fact("daemon", f"Ollama {version}", "reachable, its model list did not arrive")
    return Fact("daemon", f"Ollama {version}", f"reachable, {len(models)} models installed")


def _daemon_value(daemon: Daemon, path: str, field: str) -> object | None:
    try:
        answer = daemon("GET", path)
        if answer.status != 200:
            return None
        payload = answer.json()
    except (DaemonError, ValueError, RecursionError):
        return None
    return payload.get(field) if isinstance(payload, dict) else None


def _machine_fact(probes: Probes, pointer_path: Path, config_file: Path | None) -> Fact:
    profile = _bound_profile(pointer_path, config_file)
    note = "not measured in this results folder yet" if profile is None else hardware_words(profile)
    return Fact("machine", probes.hostname(), note)


def _bound_profile(pointer_path: Path, config_file: Path | None) -> HardwareProfile | None:
    """The profile this machine is bound to for that results folder, if the folder holds it."""
    if config_file is None:
        return None
    try:
        bound = read_pointer(pointer_path).binding_for(config_file.parent)
        if bound is None:
            return None
        return scan_profiles(load_config(config_file).paths.hardware_dir).profiles.get(bound)
    except (ConfigError, SchemaVersionError, PointerFileError, OSError):
        return None


__all__ = [
    "ASCII_GLYPHS",
    "STEP_COUNT",
    "STEP_NAMES",
    "UNICODE_GLYPHS",
    "Fact",
    "Glyphs",
    "Intro",
    "collect_intro",
    "done_line",
    "glyphs",
    "hardware_words",
    "intro_lines",
    "print_intro",
    "step_head",
    "style_rules",
    "use_color",
]
