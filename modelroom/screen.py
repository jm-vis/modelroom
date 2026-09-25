"""The screen of the guided mode: every line it draws, and the words each line is made of.

One object draws the run (`Screen`). It knows each line as a list of `(class, text)` fragments
and prints it either with color, through `prompt_toolkit.print_formatted_text` and the one style
of the dialog (`intro.style_rules` plus the question's own classes), or without color as exactly
the same plain text through the run's `out`. That is the decision `intro.print_intro` makes for
the start screen, made once more for the rest of the run, with the same fallback where the
library cannot drive the console at all.

The plain text of a line is its fragments' text with one leading space, as the start screen has
it -- so `modelroom --answers <file>` and a terminal run read as the same screen, one in color
and one not (CONTRACTS.md, "Guided mode", "The screen").

Everything here is a pure function of values a step already read; nothing in this module opens a
file, asks a question or computes a fit. The words of a step's own facts live with that step
(`guided_context.scale_words`, `guided_loadtest`), the words of the result table with the view
that draws it (`modelroom/views.py`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Sequence

from .dialog import QUESTION_RULES
from .intro import STEP_NAMES, Fact, Glyphs, fact_line, glyphs, hardware_words, style_rules, use_color
from .profile import HardwareProfile

Fragment = tuple[str, str]
Line = list[Fragment]

# The head of a step is filled to 72 characters: wide enough to carry the longest step name and
# narrow enough to stand inside the 100 characters the result table takes (decided 2026-09-24).
HEAD_WIDTH = 72
# The step's name in a summary line, padded so that every summary starts its balance in the same
# column (`Measurement` is the longest at 11).
NAME_WIDTH = 14
# A note is a side remark about the answer above it, indented by two.
NOTE_INDENT = "  "
# What `joined` wraps at: the width of the result table, so the notes under it end where it ends.
JOIN_WIDTH = 100
# A fit against system memory says so behind its class, in the selection list of step 2 and in the
# result table alike: the same word in both places means the same thing (decided 2026-09-24).
RAM_MODES = ("cpu_gpu", "cpu")
RAM_SUFFIX = " (RAM)"
# Counts a sentence says in words rather than in digits; beyond ten the digits read better.
_NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


# --- the lines ------------------------------------------------------------------------------------


def plain(line: Line) -> str:
    """One line as a log reads it: the fragments' text, with the one leading space of the screen."""
    if not line:
        return ""
    return (" " + "".join(text for _class, text in line)).rstrip()


def head_line(number: int, marks: Glyphs) -> Line:
    """The head a step begins with: `-- Step 1 of 5  Configuration ------...`, 72 characters wide."""
    opening = f"{marks.rule * 2} Step {number} of {len(STEP_NAMES)}  {STEP_NAMES[number - 1]} "
    return [("class:label", opening + marks.rule * max(HEAD_WIDTH - len(opening), 0))]


def answer_line(question: str, words: str, *, path: bool = False) -> Line:
    """The line the run writes once a question is answered: the question, and the answer in words.

    The run writes it, not the dialog library: `questionary` prints its own idea of the answer
    (`done (2 selections)`, the whole line of the scale it marked) and that reading was the
    "one thing chained to the next" of the test round of 2026-09-24. `TerminalAsker` erases the
    question instead, and this line takes its place -- so the terminal and an answer file leave
    the same line behind, one in color and one not.
    """
    return [
        ("class:qmark", "?"),
        ("", " "),
        ("class:question", question),
        ("", "  "),
        ("class:path" if path else "class:answer", words),
    ]


def note_line(text: str) -> Line:
    """A side note under an answer: what the run made of it, indented and gray."""
    return [("", NOTE_INDENT), ("class:note", text)]


def summary_line(number: int, summary: str, marks: Glyphs, *, done: bool = True) -> Line:
    """The balance of a step: a check mark and its name, or a dash where it did nothing.

    A dash is not a failure -- a step that measured nothing because nobody asked it to did
    exactly what it was told. It is the difference between "done" and "there was nothing to do",
    which a check mark cannot say.
    """
    glyph = ("class:done", marks.check) if done else ("class:tight", marks.skip)
    return [glyph, ("", " "), ("class:value", STEP_NAMES[number - 1].ljust(NAME_WIDTH)), ("", "  "), ("", summary)]


def card_lines(facts: Sequence[Fact]) -> list[Line]:
    """The card of step 5, in the rows of the start screen: label, value, note."""
    return [fact_line(fact) for fact in facts]


def joined(pieces: Sequence[str], *, dot: str = "·", width: int = JOIN_WIDTH, indent: int = 1) -> list[str]:
    """`pieces` as one line, joined with ` <dot> `, wrapped between pieces and never inside one.

    A piece is a whole statement (`1 package too tight (unsloth IQ4_NL)`), so a break inside one
    would split a number from what it counts. A piece longer than `width` therefore stands on a
    line of its own and is left as long as it is.
    """
    separator = f" {dot} "
    lines: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        candidate = piece if not current else current + separator + piece
        if current and len(candidate) + indent > width:
            lines.append(" " * indent + current)
            current = piece
        else:
            current = candidate
    if current:
        lines.append(" " * indent + current)
    return lines


class Screen:
    """The one drawer of the run: every line in color, or the same line as plain text.

    `colored` is the caller's own answer to that question, as `intro.print_intro` takes it --
    `modelroom --answers <file>` passes `False`, because such a run is read from a log and its
    output must not depend on the window it was started in. `None` asks the stream.
    """

    def __init__(self, out: Callable[[str], None], stream=None, colored: bool | None = None) -> None:
        self._out = out
        self._glyphs = glyphs(stream)
        self._colored = use_color(stream) if colored is None else colored
        self._style = None

    @property
    def glyphs(self) -> Glyphs:
        return self._glyphs

    def blank(self) -> None:
        self.write([])

    def head(self, number: int) -> None:
        self.write(head_line(number, self._glyphs))

    def answer(self, question: str, words: str, *, path: bool = False) -> None:
        self.write(answer_line(question, words, path=path))

    def note(self, text: str) -> None:
        self.write(note_line(text))

    def done(self, number: int, summary: str) -> None:
        self.write(summary_line(number, summary, self._glyphs, done=True))

    def skipped(self, number: int, summary: str) -> None:
        self.write(summary_line(number, summary, self._glyphs, done=False))

    def summary(self, number: int, summary: str, done: bool) -> None:
        """The balance of a step: a check mark where it did something, a dash where it did not."""
        self.write(summary_line(number, summary, self._glyphs, done=done))

    def card(self, facts: Sequence[Fact]) -> None:
        for line in card_lines(facts):
            self.write(line)

    def write(self, line: Line) -> None:
        """One line: in color where that works, else the same text through `out`."""
        if self._colored and self._in_color(line):
            return
        self._out(plain(line))

    def _in_color(self, line: Line) -> bool:
        """Print through `prompt_toolkit`; `False` (once and for all) when it cannot drive this console.

        The same measured case `intro._print_in_color` carries: a Windows Python under a terminal
        that says `xterm-256color`, where the library looks for a Windows console and raises. The
        run then prints plain lines for the rest of its life rather than trying again per line.
        """
        from prompt_toolkit import print_formatted_text
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.styles import Style

        if self._style is None:
            self._style = Style(style_rules() + QUESTION_RULES)
        try:
            print_formatted_text(FormattedText([("", " "), *line]), style=self._style)
        except Exception:  # noqa: BLE001 -- the library's own failures are not one class
            self._colored = False
            return False
        return True


# --- the words of an answer -----------------------------------------------------------------------

THIS_FOLDER = "this folder"
THIS_MACHINE = "this machine"
A_PROFILE_FILE = "a profile file"
NO_MACHINE = "none"
NOTHING_CHOSEN = "nothing"
SAME_MACHINE = "the same machine"
A_CLONE = "a clone"
CLONE_QUESTION = "The profile of this folder is gone. Is this the same machine or a clone?"
_MACHINE_WORDS = {"this-machine": THIS_MACHINE, "import": A_PROFILE_FILE}


def yes_no(answer: bool) -> str:
    """A switch as the list of the question shows it."""
    return "Yes" if answer else "No"


def machines_words(picked: Sequence[str]) -> str:
    """What was marked in the machine list, in the words of its own entries."""
    return ", ".join(_MACHINE_WORDS.get(value, value) for value in picked) or NO_MACHINE


def clone_words(answer: str) -> str:
    """The answer to the clone question, in its own words."""
    return A_CLONE if answer == "clone" else SAME_MACHINE


def names_words(names: Sequence[str], empty: str = NOTHING_CHOSEN) -> str:
    """A list of names as one answer: `Qwen3.5-9B, Qwen3-0.6B`, or `empty` when nothing was marked."""
    return ", ".join(names) or empty


def context_tokens_text(context: int) -> str:
    """A context as a number a reader keeps: `32k` where it is whole thousands, else the number."""
    return f"{context // 1024}k" if context % 1024 == 0 else f"{context} tokens"


# --- the notes and the balances of the steps -------------------------------------------------------


def number_word(count: int) -> str:
    return _NUMBER_WORDS[count] if count < len(_NUMBER_WORDS) else str(count)


def measured_note(profile: HardwareProfile, *, again: bool) -> str:
    """The one note step 1 leaves behind about a measurement: the hardware, and who confirmed it.

    Without a cross-check the sentence ends after the hardware: `llmfit` said nothing, so nothing
    is claimed about it.
    """
    checks = profile.llmfit_crosscheck
    confirmed = checks.ram_physical.status == "confirmed" and checks.vram.status == "confirmed"
    opening = "measured again" if again else "measured"
    return f"{opening}: {hardware_words(profile)}" + (", confirmed by llmfit" if confirmed else "")


def imported_note(display_name: str, profile: HardwareProfile) -> str:
    """The one note an import leaves behind: whose machine came in, and what it is."""
    return f"imported {display_name}: {hardware_words(profile)}"


def import_notes(arrived: Sequence[HardwareProfile], report: Sequence[str]) -> list[str]:
    """What an import leaves on the screen: one note per machine that came in, else its own report.

    An import that brought no new profile -- the stored one was newer, or only measurements came in
    -- has nothing to name, and then its report is the only thing that says what happened
    (second-model round, 2026-09-24).
    """
    if arrived:
        return [imported_note(profile.display_name, profile) for profile in arrived]
    return list(report)


def machines_summary(count: int) -> str:
    return f"{count} machine" if count == 1 else f"{count} machines"


def search_notes(log: dict, repositories: int, models: int, dot: str = "·") -> list[str]:
    """The two notes step 2 keeps on the screen, read from the log of that very search.

    One reading of the search for the file and for the screen (`search.SearchOutcome.search_log`):
    a line and a file may not say different numbers about the same thing.
    """
    accounts = log.get("accounts", [])
    answered = [entry["account"] for entry in accounts if entry["class"] == "publisher" and entry["hits"]]
    return [
        searched_note(
            answered,
            sum(1 for entry in accounts if entry["class"] == "packager"),
            open_lists=any(entry["class"] == "open" for entry in accounts),
        ),
        repositories_note(
            repositories, models, page_full=any(entry["page_full"] for entry in accounts), dot=dot
        ),
    ]


def searched_note(publishers: Sequence[str], packagers: int, *, open_lists: bool) -> str:
    """Where the search asked, in one sentence instead of one line per account.

    The seven account lines of the test round of 2026-09-24 said "was my account asked" seven
    times over; a reader of the screen needs the answer once. Which account answered with what
    is in `search.json` (CONTRACTS.md, "Search log").
    """
    # "answered", not "was asked": a publisher of this catalog whose page came back with nothing was
    # asked all the same, and `search.json` names it (second-model round, 2026-09-24).
    if not publishers:
        at = "no publisher of this catalog that answered"
    elif len(publishers) == 1:
        at = f"the publisher {publishers[0]}"
    else:
        at = "the publishers " + ", ".join(publishers[:-1]) + f" and {publishers[-1]}"
    open_part = ", and the two open lists" if open_lists else ""
    return f"searched Hugging Face at {at} and the {number_word(packagers)} listed packagers{open_part}"


def repositories_note(repositories: int, models: int, *, page_full: bool, dot: str = "·") -> str:
    """How much the search answered with, and how much of it is a choice: the two numbers that matter."""
    counted = f"{repositories} repository" if repositories == 1 else f"{repositories} repositories"
    if not models:
        text = f"{counted}, none of them a model you can pick from"
    elif models == 1:
        text = f"{counted}, 1 of them a model you can pick from"
    else:
        text = f"{counted}, {models} of them models you can pick from"
    return f"{text} {dot} a more specific word shortens the list" if page_full else text


def packages_summary(models: int, packages: int, repositories: int) -> str:
    """The balance of step 2: what was chosen, and what the fetch brought in for it."""
    return (
        f"{models} model{'' if models == 1 else 's'}, "
        f"{packages} package{'' if packages == 1 else 's'} from "
        f"{repositories} {'repository' if repositories == 1 else 'repositories'}"
    )


NOTHING_CHOSEN_SUMMARY = "nothing chosen, the folder stays as it is"


# --- the card of step 5 ----------------------------------------------------------------------------

SPEED_NOT_MEASURED = "not measured"
MEASURE_HINT = "say Yes in step 4 to measure an installed package"


def short_hardware(profile: HardwareProfile) -> str:
    """One machine's hardware in the short form the card has room for: `12 GB graphics, 127 GB memory`."""
    if profile.gpu_state == "measured" and profile.vram_gib is not None:
        card = f"{profile.vram_gib:.0f} GB graphics"
    elif profile.gpu_state == "none":
        card = "no graphics card"
    else:
        card = "graphics memory not measured"
    memory = "system memory unknown" if profile.ram_physical_gib is None else f"{profile.ram_physical_gib:.0f} GB memory"
    return f"{card}, {memory}"


def measured_when(profile: HardwareProfile, now: datetime) -> str:
    """When this profile was measured, as a reader says it: `measured today`, else the date."""
    recorded = profile.recorded_at.date()
    return "measured today" if recorded == now.date() else f"measured {recorded.isoformat()}"


def machine_fact(label: str, profile: HardwareProfile | None, now: datetime, reason: str) -> Fact:
    """One machine's row of the card: what it is called, and what it is."""
    if profile is None:
        return Fact("machine", label, reason)
    return Fact("machine", label, f"{short_hardware(profile)}, {measured_when(profile, now)}")


def named_accounts(accounts: Sequence[str], named: int = 3) -> str:
    """Up to `named` accounts by name, the rest as a count: `Qwen, unsloth and Ollama`."""
    if not accounts:
        return "no account"
    if len(accounts) <= named:
        return accounts[0] if len(accounts) == 1 else ", ".join(accounts[:-1]) + f" and {accounts[-1]}"
    return ", ".join(accounts[:named]) + f" and {len(accounts) - named} more"


def models_fact(names: Sequence[str], packages: int, accounts: Sequence[str]) -> Fact:
    """The models row of the card: which models, and how many packages came in for them."""
    counted = f"{packages} package" if packages == 1 else f"{packages} packages"
    return Fact("models", names_words(names, "none"), f"{counted} from {named_accounts(accounts)}")


def speed_fact(measured: int, fastest: tuple[str, float] | None) -> Fact:
    """The speed row of the card: how many packages were measured here, and which was fastest."""
    if not measured or fastest is None:
        return Fact("speed", SPEED_NOT_MEASURED, MEASURE_HINT)
    name, speed = fastest
    return Fact("speed", f"{measured} measured", f"fastest {name} at {speed:.1f} tok/s")


def result_fact(path: str, ranked: int, set_aside: Sequence[tuple[str, int]], dot: str = "·") -> Fact:
    """The result row of the card: where the document is, and what it holds."""
    counted = f"{ranked} package" if ranked == 1 else f"{ranked} packages"
    pieces = [f"{counted} ranked"]
    tight = next((count for word, count in set_aside if word == "too tight"), 0)
    pieces[0] += f", {tight} too tight" if tight else ", none too tight"
    pieces += [f"{count} {word}" for word, count in set_aside if word != "too tight"]
    return Fact("result", path, f" {dot} ".join(pieces))


# --- the line that installs a package (the install itself is a work package of its own) ------------


def install_name(identity: Sequence[str], quantization: str, package_format: str) -> str | None:
    """The local name Ollama would give this package, or `None` where it would give none.

    `hf.co/<repo>:<quantization>` for a Hugging Face build and `<name>:<tag>` for one of the
    Ollama registry -- the two shapes the load test matches an installed model by
    (`loadtest.installed_candidates`). A package that is no GGUF build, or whose quantization the
    registry never stated, has no such name, and then there is no line rather than a guess.
    """
    source, repo_or_name, _rest = identity
    if source == "ollama":
        return repo_or_name or None
    if package_format != "gguf" or not repo_or_name or quantization == "unknown":
        return None
    return f"hf.co/{repo_or_name}:{quantization}"


def install_pull_line(rank: int, name: str) -> str:
    """The line a reader copies to get the first package of their own machine."""
    return f"to install #{rank}: ollama pull {name}"


# --- rank ranges ------------------------------------------------------------------------------------


def ranks_text(ranks: Iterable[int], dash: str = "–") -> str:
    """A set of ranks as a reader reads them: `#1-2` for a run, `#1, #4-5` for two, `#3` for one."""
    ordered = sorted(set(ranks))
    if not ordered:
        return ""
    runs: list[list[int]] = [[ordered[0]]]
    for rank in ordered[1:]:
        if rank == runs[-1][-1] + 1:
            runs[-1].append(rank)
        else:
            runs.append([rank])
    return ", ".join(f"#{run[0]}" if len(run) == 1 else f"#{run[0]}{dash}{run[-1]}" for run in runs)


def fit_word(fit_class: str, mode: str | None) -> str:
    """The fit of a ranked package in a word: its class, and `(RAM)` where the pool is system memory."""
    return f"{fit_class}{RAM_SUFFIX}" if mode in RAM_MODES else fit_class


__all__ = [
    "CLONE_QUESTION",
    "HEAD_WIDTH",
    "JOIN_WIDTH",
    "MEASURE_HINT",
    "NAME_WIDTH",
    "NOTHING_CHOSEN_SUMMARY",
    "RAM_MODES",
    "RAM_SUFFIX",
    "SPEED_NOT_MEASURED",
    "THIS_FOLDER",
    "Screen",
    "answer_line",
    "card_lines",
    "clone_words",
    "context_tokens_text",
    "fit_word",
    "head_line",
    "import_notes",
    "imported_note",
    "install_name",
    "install_pull_line",
    "joined",
    "machine_fact",
    "machines_summary",
    "machines_words",
    "measured_note",
    "measured_when",
    "models_fact",
    "named_accounts",
    "names_words",
    "note_line",
    "number_word",
    "packages_summary",
    "plain",
    "ranks_text",
    "repositories_note",
    "result_fact",
    "search_notes",
    "searched_note",
    "short_hardware",
    "speed_fact",
    "summary_line",
    "yes_no",
]
