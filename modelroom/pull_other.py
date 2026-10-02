"""Step 5's second question: a package for another chosen machine, not this one (A3, AP9-FR).

`pull_step` (`guided_install.py`) already offers this machine's rows; `pull_other_step` follows it
with the rows of every *other* chosen machine, one list, headed by machine (`dialog.Choice(heading=True)`,
the shape U6's own list uses). A row of a machine the run cannot reach over SSH has no daemon here to
check or pull for: the choice only ever hands back a **command to paste elsewhere**, never a POST --
`modelroom pull` (A4) is what runs it, on the machine it names. A row of the run's **active** SSH
target (`run.ssh_active`, B4) is pulled there over SSH instead, whichever rank was chosen
(`guided_remote.pull_remote_row`, B5); every other machine with a stored SSH alias, not connected
this run, is handed over exactly like one without SSH at all.

The question is skipped outright where no other chosen machine has a single named row (A2): the
list would only ever offer `none`. It is asked whether or not this machine itself has a row here --
an import-only run (no row of its own, `this-machine` not chosen) still asks it, so long as another
chosen machine has a package of its own (Codex r1, finding 1). `pull_other` is optional like
`pull_package` -- an answer file without it chooses nothing, silently -- and one target machine at
most, the same as the guided run's own pull (GF question 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .contracts import Package
from .dialog import AnswerMissingError, Choice
from .document import RenderDocument
from .install_vllm import install_block_lines, vllm_install
from .intro import label_line
from .pull_choice import NONE_VALUE, PullRow, has_named_row, package_of, pull_rows, weights_bytes_of
from .runtimes import VLLM
from .views_install import INSTALL_LABEL

if TYPE_CHECKING:  # pragma: no cover - the run object is passed in, never constructed here
    from .guided import GuidedRun
    from .guided_install import PullContext

KEY = "pull_other"
QUESTION = "Which package should go onto another computer?"
NONE_WORDS = "none, do not pull"
PASTE_NOTE = "paste it there: it checks Ollama and the disk and asks before it pulls"


@dataclass(frozen=True)
class OtherRow:
    """One row of the list: the machine it is about, and the row of its own table."""

    machine: str
    row: PullRow

    @property
    def value(self) -> str:
        """The answer file's value: `<machine> <pull name>`, or `<machine> #<rank> locked` for a
        row that cannot be chosen (answer files are flat, `answers.py:26-33`; GF question 6)."""
        return f"{self.machine} {self.row.value}"


def other_rows(
    document: RenderDocument, machine: str, ambiguous: frozenset[str] = frozenset(), dot: str = "·", packages: Sequence[Package] = ()
) -> list[OtherRow]:
    """Every shown row of every chosen machine but `machine`, in document order, each machine's own
    rows in table order. `packages` are the run's: a vLLM row is built from its package (revision),
    so without them its row is locked `no safe command`."""
    rows: list[OtherRow] = []
    for block in document.machines:
        if block.machine != machine:
            rows += [OtherRow(block.machine, row) for row in pull_rows(document, block.machine, ambiguous, dot, packages=packages)]
    return rows


def eligible_other_machines(document: RenderDocument, machine: str, ambiguous: frozenset[str] = frozenset()) -> frozenset[str]:
    """The other machines whose table gets the "or choose a package for it below" note, and that
    `pull_other_step`'s list will show a row for -- both read this, under the one condition A3
    names: the other machine has a shown row with a name. An import-only run (this machine has no
    row of its own in the document at all) is asked all the same -- there is no "also" here, but
    another chosen machine's own package still needs somewhere to go (Codex r1, finding 1)."""
    return frozenset(block.machine for block in document.machines if block.machine != machine and has_named_row(block))


def pull_other_choices(rows: Sequence[OtherRow]) -> list[Choice]:
    """One heading per machine that has a row, its rows under it (locked exactly as they are in its
    own table), then `none`, the pointer starting there -- never on a row (A3)."""
    choices: list[Choice] = []
    seen: set[str] = set()
    for entry in rows:
        if entry.machine not in seen:
            choices.append(Choice(f"__heading_{entry.machine}__", entry.machine, heading=True))
            seen.add(entry.machine)
        choices.append(Choice(entry.value, _row_words(entry), disabled=entry.row.locked, warn=entry.row.over_disk))
    return choices + [Choice(NONE_VALUE, NONE_WORDS, checked=True)]


def _row_words(entry: OtherRow) -> str:
    """`#1  unsloth · Q4_K_S  5.0 GB  hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_S`, as U6's own list writes
    a row (`guided_install._row_words`), without the "local already" mark: no daemon of that
    machine was asked here."""
    row = entry.row
    return f"#{row.rank}  {row.cell}  {row.weights_gib:.1f} GB  {row.locked if row.over_disk else row.name or ''}".rstrip()


def pull_other_step(run: "GuidedRun", context: "PullContext | None") -> None:
    """Ask, then pull the chosen row over SSH where it belongs to the active target (B5), else hand
    over its command (A3); nothing where nothing is eligible or chosen."""
    if context is None:
        return
    if not eligible_other_machines(context.document, context.machine, context.ambiguous):
        return
    rows = other_rows(context.document, context.machine, context.ambiguous, run.screen.glyphs.dot, context.packages)
    choices = pull_other_choices(rows)
    try:
        value = run.asker.select(KEY, QUESTION, choices)
    except AnswerMissingError:
        return
    chosen = next((entry for entry in rows if entry.row.locked is None and entry.value == value), None)
    run.screen.answer(QUESTION, NONE_WORDS if chosen is None else f"{chosen.machine}  #{chosen.row.rank}  {chosen.row.name}")
    if chosen is None:
        return
    if chosen.row.runtime == VLLM.key:
        # modelroom starts no vLLM, here or over SSH: the row is handed over like one for a computer it
        # cannot reach, with both commands and no `modelroom pull` (that command is Ollama's alone).
        for line in vllm_handover_lines(context, chosen.machine, chosen.row):
            run.screen.write(line)
        return
    if chosen.machine == run.ssh_active:
        from .guided_remote import pull_remote_row  # imported here: only an SSH row needs the SSH layer

        pull_remote_row(run, chosen.machine, chosen.row, context)
        return
    package = package_of(chosen.row, context.packages)
    if package is None:
        return
    for line in handover_lines(chosen.machine, chosen.row, weights_bytes_of(package)):
        run.screen.write(line)


def vllm_handover_lines(context: "PullContext", machine: str, row: PullRow) -> list:
    """The handover of a vLLM row (A3, AP9-SV2-I): both install lines with the machine named, what to do
    there, and no `modelroom pull` line (`install_vllm.install_block_lines`)."""
    block = next(block for block in context.document.machines if block.machine == machine)
    entry = next(entry for entry in block.ranked if entry.rank == row.rank)
    install = vllm_install(package_of(row, context.packages), entry.fit, block.profile)
    return [[], *install_block_lines(INSTALL_LABEL, row.rank, install, repo=row.name, where=machine)]


def handover_lines(machine: str, row: PullRow, weights_bytes: float) -> list:
    """The three lines of the handover (A3): the command to paste on `machine`, what it does there,
    and the bare `ollama pull` for a computer without modelroom -- no path, no machine name beyond
    the neutral one the document already carries."""
    command = f"modelroom pull {row.name} --weights-bytes {round(weights_bytes)}"
    return [
        [],
        label_line(INSTALL_LABEL, [("class:command", f"#{row.rank} on {machine}  {command}")]),
        label_line("", [("class:note", PASTE_NOTE)]),
        label_line("", [("class:note", f"without modelroom there: ollama pull {row.name}")]),
    ]


__all__ = [
    "KEY",
    "NONE_WORDS",
    "PASTE_NOTE",
    "QUESTION",
    "OtherRow",
    "eligible_other_machines",
    "handover_lines",
    "other_rows",
    "pull_other_choices",
    "pull_other_step",
    "vllm_handover_lines",
]
