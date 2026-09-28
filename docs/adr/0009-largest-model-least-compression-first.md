# 0009 -- The largest model that fits, the least compression of it, and only the choice of the run

- **Status:** accepted, decided 2026-09-28
- **Supersedes:** parts of 0008 (decisions 1 and 7); ADR 0001 to 0007 stay in force

## Context

A hand test on 2026-09-28 ran a guided run on a rented machine with four L4 cards. Every package of
the chosen model was `ample` there, so the rule of the day -- fit class, then a measurement and its
speed, then the quantization in `QUANT_ORDER` ascending -- put the most compressed build first: an
`IQ1` at #1 on a machine that holds the 16-bit build with room to spare. The same run showed the
measurement of step 4 in no row of the result (the rule of the top ten had pushed it out), rankings
of machines and models the run had not chosen, profiles of the folder that could not be picked, a
selection list whose column head scrolled away, and a step 4 that asked twice and said `vLLM
measured` about a number that counted for nothing.

The owner decided the same day: one row per model where several are shown; `latest` stands only
before `legacy`, between `latest` and an unknown release the size decides; graphics memory before
system memory; the selection of a run is not persisted; `F16`, `BF16` and `F32` are one step, the
smaller weights first.

## Decision

**Among the packages that fit, modelroom ranks graphics memory before system memory, every model
that is not legacy before a legacy one, then the largest model first and, within one model, the
least compression that fits; a measurement only breaks a tie between two builds of the same
compression, a measured package the rule does not show is appended with its place, and no model
family is rated against another.**

1. **Seven steps** (`ranking.rank_packages(..., *, models)`, `ranking.ModelFacts(parameters_b,
   legacy)`): the pool; `legacy` last (`latest` and an unknown release alike, `legacy*` is `legacy`);
   the parameter count, larger first, an unknown one last; the compression, `QUANT_ORDER` reversed, an
   unknown quantization last, the 16- and 32-bit builds one step with the smaller weights first;
   measured before unmeasured; larger weights; the package identity. `ample`, `fits` and `tight` rank
   alike and keep their word; the speed of a measurement is no part of the key.
2. **One row per model** where the (narrowed) snapshot holds several base models: the first package of
   each model under the whole key, the rows in the order of that key (`ranking.best_per_model`). With
   one model the first ten, as before. The render decides from the snapshot, so `modelroom render`
   decides alike.
3. **The measurement of the machine stays visible**: every measured package the rule leaves out is
   appended behind the rule's rows with its place (`ranking.shown_rows`); no row of the rule moves.
   `ranked` of the render document therefore carries one row per model in that case and the appended
   rows in both; `rank` is the number of the row shown; `ranked_total` is every ranked package. The
   row note keeps its form and gains one sentence with the fact it names
   (`ranking.best_per_model`, `ranking.shown_for_measurement`). Document schema 3 is unchanged.
   "Measured" is any measurement of exactly this package on this machine: one that counts, one that
   fails on `requests` alone, or one that counts nowhere here (another context, not comparable, not
   valid) -- the last kind stays without a speed, and its row says why.
4. **The result covers the choice of the run** (`selection.RunSelection`, in memory only): the
   machines step 1 chose -- measured, imported, entered, or a profile of the folder marked as it is --
   and the models step 2 chose, with the release and the size the list showed. Steps 3 and 4, the
   balance of step 2, the card and `docs/models.md/json` read the same selection. An empty choice is
   a choice: a run that chose no model ranks none of the folder's, and one that chose no machine
   names none -- no machine the run did not choose is added, the bound one included -- and says so. Step 4 measures this
   machine only where the run chose it. Nothing of it is written; a later `modelroom render` shows the
   whole folder, one row per model.
5. **Two inputs for the parameter count, one function** (`parameters.parameter_count`): the list of
   step 2 takes `SearchHit.parameters_b`, else the name, and hands that very number to the result;
   `modelroom render` takes `BaseModelSpec.parameters_b`, else the name. Two models of one size by
   their names may therefore stand otherwise in a later `render` where the Hub names two numbers.
   A render without a run knows no release: no model is `legacy` there.
6. **The window of a model**: a package whose model declares a window smaller than the context of the
   run is `unknown` (`context beyond the window of this model (32k)`), so the size scale and the
   ranking count alike; the scale grays out a level only beyond the window of every model of the run.
7. **One rule for every list**: nothing marked in advance, the space bar marks and colors the row,
   `Enter` takes the mark or the pointer row; the machine list offers every profile of the folder, the
   one this machine already has first; the column head of the model list stands outside its window.
8. **Step 4 from one mould**: every server's notes, then one question `load_test`, then one list
   `load_test_packages` with the Ollama candidates and the row `vllm`. A vLLM number counts where the
   served model is a model of the run and the server runs the run's context; both servers pass
   one check before they offer anything -- this machine is the machine of its bound profile,
   fingerprint included, and the run chose it -- so the vLLM server is offered, measured and counted
   only where the daemon would measure too; the reason it will not
   count stands before the question; the balance never says `vLLM measured` where nothing counts.

## Superseded sentences

- ADR 0008, decision 1: `LocalServer(key, name, origin, transport(run), start_fact, step,
  card_facts)` -- now `LocalServer(key, name, origin, transport(run), start_fact, offer, measure,
  card_facts)`.
- ADR 0008, decision 7: "the third optional answer key `load_test_vllm`" -- the key is gone; the
  vLLM server is the row `vllm` of `load_test_packages`, and `load_test` and `pull` are the two
  optional answer keys.

## Rejected

- **Benchmarks or a rating of families**: "best" is the same model with less compression, nothing
  more; modelroom states facts and rates no family against another.
- **The speed as a rank**: a speed says how fast one build ran here, not which build a reader
  should take; it stays information in its row.
- **`latest` before an unknown release**: the owner decided that `latest` stands only before
  `legacy` -- a 7B `latest` is not a better answer than a 27B nobody dated.
- **Persisting the selection** (`[guided].models`, `[guided].machines`): the owner decided against
  it; the configuration of the folder stays the folder's.
- **A new document schema**: `ranked` changes what rows it carries, not its shape.

## Limits

- A package measured on this machine that the fit sets aside -- `no_fit` (above 98 % of its pool)
  or `not covered` (for instance beyond the window of its model) -- is no ranked package, so no row
  is appended for it: it stands in its set-aside list, without its speed. The rows appended for
  their measurement are the measured packages the rule ranks and leaves out, nothing more.

## Consequences

- The first row, and so the install line and the pull, is a larger download: a `Q8_0` or a 16-bit
  build where an `IQ1` stood before.
- Where step 2 names Qwen3.5-9B `legacy*` -- the shipped catalog does -- a smaller model that is not
  overtaken stands before it in a run of both.
- A run that chose one model shows its first ten packages and its measured ones; a `render` of a
  folder of several shows one row per model.
