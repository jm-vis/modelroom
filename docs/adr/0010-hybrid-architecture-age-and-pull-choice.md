# 0010 -- A hybrid architecture read, the age of a line, the publisher's repository, the package chosen

- **Status:** accepted, decided 2026-09-28
- **Supersedes:** nothing; ADR 0001 to 0009 stay in force, and the ranking rule of 0009 is unchanged

## Context

A hand test on the owner's laptop on 2026-09-28, after AP9-UX, found nine things a reader could not
follow or that were computed wrong (U1 to U9), and a second hand run on a rented server found that a
vLLM server serving `Qwen/Qwen3.5-4B` had no package in its folder (SV1-F):

- Qwen3.5 was judged from the size of its packages: its `config.json` holds 24 layers of linear
  attention between 8 of full attention, `architecture_from_hf_config` read that as `unknown`, and
  the size basis overstated the KV cache by 4.5 times. A package between 98 and 100 % of the card was
  `no fit`, although the system memory held it.
- `Qwen3.5-0.8B-Base` stood `latest` beside Qwen 3.8: a variation was a group of its own, and two
  candidates of equal standing gave no successor at all.
- The folder of the last run was used without a word, and a run started in the home folder wrote
  `modelroom.toml`, `docs` and `state` into it.
- Only `#1` could be pulled, nobody saw the free memory or disk, two rows looked alike, and an
  imported server looked like this computer.
- A guided run wrote the packager's repository as the only target of a base model, so the fetch never
  read the publisher's own `tensor` package.

The owner decided F1 to F8 the same day (the build brief of AP9-UX2, the owner's decisions).

## Decision

1. **`hybrid_linear`** is a kind of `Architecture` of its own, read under exact invariants (the three
   numeric fields; `layer_types` of `num_hidden_layers` entries, only `full_attention` and
   `linear_attention`, both present; five `linear_*` fields; no MoE field) and checked by the
   contract. `linear_state_bytes` comes from `mamba_ssm_dtype` -- `float32` 4, `bfloat16`/`float16` 2,
   **4 as a documented assumption where the config names none**. Two sizes follow
   (`modelroom/architecture.py`): the KV cache of one token over the full-attention layers only, and
   the state of the linear layers, once per request whatever the context. Fit v1 and the vLLM fit
   compute a hybrid model on its architecture, not capped at `fits`. Ollama's 16-bit KV cache stays.
2. **Snapshot schema 1 is kept** while `Architecture` gains the kind and six optional fields, by the
   precedent of ADR 0007: an older snapshot reads on with the fields `None`, the next `fetch` reads
   `config.json` again, and the 0.1.0 reader does not read a rewritten snapshot in any case.
3. **A pool is chosen only up to `TIGHT_RATIO`** (98 %) of it, after its reserve; the last pool is
   classified as it is. `no fit` means that no pool holds the package with a margin of 2 %.
4. **The age of a model follows the version of its line (F3).** The group is `(publisher, family,
   line)`: the family is the first word of the name; a line is a size step with its own version series
   from the documented vocabulary `small`, `medium`, `large` and the counter `r` (`R1`); every other
   word is a variation of its version. Unresolved repositories of a listed packager whose family is
   one publisher's prove a higher version too. The successor is always one where there is a
   candidate (size, then the same variation, then the fewest, then the name). A stated `legacy` stays;
   a catalog `latest` gives way to a higher version; a fault's `unknown` is computed and says its fault
   as a note; a cycle chooses only the successor. New line words come only with a GF decision.
5. **The publisher's repository is always a target (F8)**: `apply_hits` adds each resolved base
   model's own repository to its `repos` once; the fetch records its `tensor` package, and
   `match_served` finds what a vLLM server serves.
6. **The user chooses the package to pull (F4)**: a list of the rows shown, the command of the choice,
   then Yes or No; an answer file without `pull_package` takes `#1` as before. A pull name more than
   one package identity answers to, over all active complete GGUF and Ollama packages of the run's
   models before the eligibility filter, belongs to no row. Rows that look alike name their
   repository under the table (F7); nothing is merged.
7. **The room is read live (F5, F6)**: the free memory of each card (`nvidia-smi`) and the free disk
   where Ollama keeps its models -- an approximation (`OLLAMA_MODELS`, else the defaults; both
   defaults existing is unclear) -- stand under the table head, `unknown` where they cannot be read;
   the fit's pool is called its budget. The disk is measured again after the choice: too little asks
   nothing and pulls nothing; unknown asks with a note.
8. **Where a machine came from** (`this`, `imported`, `entered`, `folder`) is said on the card and in
   the table head, and another computer gets its command to copy with no pull.
9. **The folder of the last run is a question (F1)** where it is not this folder, with the pointer on
   it; a home folder or the top of a drive is a question before anything is written (F2), No offering
   `<home>/modelroom`.

## Consequences

- Qwen3.5 packages fit the laptop card at L 32k (`Q4_K_M`: `fits`, not `no fit`), and every Qwen3.5
  model is `legacy` beside Qwen 3.8; its specialised variations too, which ADR 0009 then ranks behind
  every model that is not `legacy`.
- A guided run costs one model info and the pages of one tree more per base model whose repository
  was no target yet.
- `modelroom render` draws no `free` line and no command: it cannot know which machine it runs on.
- The Markdown view and the document keep schema 3; origins, room and choice live in memory only.
- Versions the search does not see (an Ollama-only release) cannot make a model `legacy`.
