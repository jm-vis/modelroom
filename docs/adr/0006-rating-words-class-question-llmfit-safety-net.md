# 0006 -- Rating words, the class question, llmfit as a safety net

- **Status:** accepted, decided 2026-09-27
- **Supersedes:** single sentences of ADR 0004 and ADR 0005, named below; both stay in force for
  everything else, and so does ADR 0001

## Context

The fit class carried llmfit's words: `perfect`, `good`, `marginal`, `too_tight`. Three of them
are judgements, not facts: the class is arithmetic -- the share of a memory pool a package needs
-- and says nothing about whether a result is good. The selection list of step 2 renamed the
classes on top of that (`perfect` and `good` both read `good`, `too_tight` read `too tight`), so
one class had two texts. No table that shows a class, a GPU state or a machine class said what
its words mean.

The machine class was asked only where the chassis named none, so a reader never saw the class
the chassis reported and could not correct a wrong one. And a deviation between the own reading
and llmfit's blocked the fit: one wrong llmfit reading made every package of a machine `unknown`,
although the own reading is the authority. For cards of different sizes llmfit's sum is not even
comparable: llmfit multiplies one card's size by the count.

## Decision

1. **Rating words.** A rating word says only what was computed or measured, never a judgement;
   one word per level; the code value is the text a view shows; one hint line under every table
   that carries such a column, short enough for a reader without the contract.
   - `Fit.fit_class` is `ample | fits | tight | no_fit | unknown`, one to one for
     `perfect | good | marginal | too_tight | unknown`. The thresholds are unchanged (0.60, 0.85,
     0.98 of the pool, inclusive), and so is the cap: off the graphics card and on the size basis
     the best class is `fits`.
   - A cell prints the value verbatim, `no_fit` included; prose and headings say the words with a
     space (`no fit`, the heading `No fit`). No other mapping exists anywhere.
   - Hint lines, one constant each: the fit hint (`screen.FIT_HINT`) under the Markdown ranking,
     under the terminal table of step 5 and, with its words for the list, in the instruction line
     of step 2; `Speed is measured, see its column.` under the two tables with a speed column; the
     GPU state hint and the class hint under the Markdown Machines table.
   - This is a change of a literal value of a persisted form, which the language standard
     otherwise never allows (`CONTRACTS.md`, "Language standard"). It is the one exception: the
     values were judgements, not names. No other literal follows them.
2. **The class question, always.** Every measurement of this machine in the guided mode asks
   `machine_class`. Where the chassis names a class, the list is the three classes with that one
   marked; an answer equal to it keeps the source `chassis`, any other answer is that class with
   the source `entered`. Where the chassis names none, the list is the four entries as before.
   An answer file that measures this machine therefore carries `machine_class`.
   `hardware --machine-class` is unchanged.
3. **llmfit as a safety net, never a block.** `fit_block_reason` no longer reads the cross-check:
   the fit computes on the own reading. A `deviation` is shown -- behind the measured note of step
   1, in the machine's headline of the Markdown view and in the head of its terminal block, as
   `; llmfit reads RAM X GiB, this measurement Y GiB` per deviating quantity -- and there is no
   column for it. `CrossCheck.status` gains `not_comparable`: `own_gib` required, `llmfit_gib`
   optional, no 5 % rule. `multi_gpu` with cards of different sizes is `not_comparable`, never
   `absent`; cards of one size keep the sum check.

## Consequences

- **No migration (A7).** The fit class is persisted in render document schema 1 of 0.1.0 (the
  ranked entries carry a `Fit`), but the package has no read-back path for render documents: every
  render rebuilds the document from the snapshot, the profiles and the measurements, and
  `RenderDocument.model_validate` accepts only the current schema. Snapshot 1, export 1 and
  measurement 2 carry no fit class. A tool of its own that reads `docs/models.json` has to learn
  the new words.
- **Identifiers stay (A6).** `MachineRanking.too_tight` and `ranking.Ranking.too_tight` (the JSON
  key `too_tight`), `Requirement.perfect_reachable` and the note codes `too_tight` and
  `cpu_caps_at_good` are identifiers of the data contract, not words a reader is shown. They are
  not renamed here; a rename is a schema decision of its own.
- `not_comparable` rides in hardware profile schema 4 without a version step: schema 4 is not
  released yet (0.1.0 writes schema 2). A file that says schema 2 or 3 and carries it is refused,
  directly and inside an export file, like `multi_gpu`.
- The self-test still requires `confirmed` for RAM and VRAM; a deviation now fails the self-test
  although the fit is computed, which is what a self-test is for.
- Every answer file that measures this machine gains one answer line on the screen,
  `? What kind of machine is this?  <class>`.

## Superseded sentences

- ADR 0004, Decision, the machine class: "`hardware --machine-class` and the guided questions
  write `entered`." The guided question writes `entered` only where the answer differs from the
  class the chassis reads (decision 2 above).
- ADR 0004, Decision, the last point: "llmfit's cross-check of several cards compares the sum
  with the sum of llmfit's own `gpus[]` entries of backend `CUDA` (`vram_gb x count`); without such
  a list the check is `absent`." This holds for cards of one size; cards of different sizes are
  `not_comparable` (decision 3).
- ADR 0005, Decision, the pool rule: "then the system memory as before, capped at `good`" and "The
  size basis keeps its cap at `good`." The cap is the same and is called `fits` (decision 1); the
  same word stands in the rejected alternative "Cap a spread fit at `good`".
- ADR 0001, Decision: its class word for a pool of 0 or less is `no_fit` now (decision 1).

## Alternatives and why not

- **Keep the words and add the hint line only.** Rejected: a hint that has to explain that
  `perfect` is no verdict contradicts its own word.
- **Keep the llmfit words and map them in every view.** Rejected: a mapping per view is how the
  list of step 2 came to show one class under two texts.
- **Rename the field names and note codes in the same step.** Rejected for now: they are read by
  tools, not by people, and renaming them is a schema decision of its own.
- **A column for the llmfit readings.** Rejected: llmfit is a safety net against a wrong own
  reading, and the package keeps moving away from it; the readings stay a clause in prose.
