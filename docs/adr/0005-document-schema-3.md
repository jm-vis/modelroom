# 0005 -- Render document schema 3: a fit spread over every graphics card

- **Status:** accepted, decided 2026-09-26
- **Supersedes:** nothing (ADR 0003 stays in force for schema 2)

## Context

With several graphics cards in a profile (ADR 0004) the fit has a third place a package can
live: not on one card, not in system memory, but spread over all cards. Ollama loads a model on
one card when any single card holds it, and spreads it over all cards otherwise. `Fit.mode` is a
closed set (`gpu | cpu_gpu | cpu`) in a persisted form, `docs/models.json`; a new value there is a
new schema version, by the same rule ADR 0001 and ADR 0003 follow.

## Decision

- `DOCUMENT_SCHEMA_VERSION = 3`. `Fit.mode` gains `gpu_split`; nothing else in the document is
  new. The number of cards stands in the embedded profile (`gpus`), not in `MachineRanking`.
- The pool rule, in order: unified memory as before; one card first -- the largest card minus
  `reserve_vram_gib`, mode `gpu`; then, with two cards or more, all cards together -- the sum of
  `card - reserve_vram_gib`, mode `gpu_split`, `reserve_gib` the reserve of all cards, **no cap**
  (the package lies in graphics memory whole; what the spread costs is speed, and speed is
  measured, not computed); then the system memory as before, capped at `good`. With one card no
  value of a fit changes. The size basis keeps its cap at `good`.
- `reserve_vram_gib` is the graphics memory left unused **on each graphics card**.
- A `gpu_split` row's note says so: `Fits into graphics memory spread over 2 cards: it needs
  about N GiB of the M GiB left after the reserve of R GiB on each card.` The list bundles such
  rows as `fit into graphics memory, spread over the cards`.

## Consequences

- A tool that checks `schema_version == 2` of `docs/models.json` has to learn schema 3.
- `OLLAMA_SCHED_SPREAD=1` makes the daemon spread a model although one card holds it; the fit then
  says `gpu` and the daemon spreads all the same. The load test measures what that costs.
- Tensor parallelism (weights divided by the cards) is not part of this rule; a serving stack that
  does it needs a rule of its own.

## Alternatives and why not

- **Cap a spread fit at `good`.** Rejected: the class is a statement about memory, and the package
  does lie in graphics memory; capping would hide a machine that holds a model comfortably.
- **Sum the cards and treat them as one pool.** Rejected: a package that one card holds would be
  judged against the sum, and a reserve per card would vanish.
