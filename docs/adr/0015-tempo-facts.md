# 0015 -- Tempo facts are shown, never a sort key; quality is a link

- **Status:** proposed, until AP9-T-K is merged (T-G, T-M and T-K add their parts to this record)
- **Supersedes:** nothing. **ADR 0009 stays unchanged:** the ranking rule and its thresholds.

## Context

The result names the fit class, the memory, the speed and a market rating, but nothing about how
good a model is at the task a reader has in mind. A reader who asks "which of these is any good for
my work" has to leave the tool with no pointer. The same result must stay one truth in three files
(Markdown, JSON, terminal): a sentence that sounds like a verdict on speed or quality, or a fact
that quietly moves a row, would make the ranking say more than it measured.

## Decision

1. **Tempo facts are display, never a sort key.** What AP9-T adds to the result -- the share of a
   model on the graphics card, the mixture-of-experts and multi-token-prediction marks, the saved
   memory type of a context -- is shown and stored beside the rows; the ranking rule of ADR 0009
   (fit class first, the largest model, the least compression, the measurement as a tie-break) does
   not read it. A test per package compares the ranking of one input with and without the new
   fields.
2. **Quality is a link, nothing more** (GF 2026-09-30, decision 4 (a)). One line,
   `Quality of these models for your task: https://artificialanalysis.ai`
   (`modelroom/tempo_text.py::QUALITY_LINK_LINE`), stands behind the last machine block of step 5
   and after `modelroom render` on success. modelroom never opens the address; the README says so
   under "Network access". Stars or scores per task inside the tool are 0.3.0 at the earliest.
3. **The line is terminal only.** It is not in the Markdown file and not in the JSON file, so a
   document stays the same data it was and a reader of the files meets no advertisement.
4. **`modelroom render` prints on success** -- after its notes on unreadable profiles, the Ollama
   setting of a saving context choice (T-K), then the link. Only the CLI handler prints;
   `render_with_config` stays silent because the guided mode calls it and draws step 5 itself, so the
   line never stands twice. This changes a promise of 0.1.0 ("silent on success"); CONTRACTS.md,
   README and CHANGELOG say so.
5. **All sentences of the tempo lines live in `modelroom/tempo_text.py`**, one constant each, so a
   wording change is a change to one file and its tests.

## Rejected

A quality score inside modelroom (a second source of truth, a table to maintain, a claim modelroom
cannot check); fetching the site to show a number (a new network target and a failure mode in a
read-only command); the link in the Markdown and JSON files (the data would carry a pointer that is
no data); printing the line from `render_with_config` (twice in the guided run); no line from
`render` (a reader of the terminal after `render` would be the one without the pointer).

## Consequences

A script that expects an empty output after a successful `modelroom render` sees one line. The
parts of T-G (graphics share), T-M (design marks) and T-K (saved context memory) append their
decisions here before this record becomes `accepted`.
