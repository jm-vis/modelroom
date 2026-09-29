# 0012 -- The memory reserve after the machine's size, the table head names the budget

- **Status:** accepted, decided 2026-09-29
- **Supersedes in part:** the fixed default `reserve_ram_gib = 8.0` (CONTRACTS.md, "MachineConfig",
  "DefaultsConfig") and the `free` line under a table's head of ADR 0010 (AP9-UX2 U8). **ADR 0009
  stays unchanged:** the ranking rule and the thresholds `AMPLE_RATIO`, `FITS_RATIO`, `TIGHT_RATIO`.

## Context

Every machine kept a fixed 8 GiB of its system memory back from the fit, written into each new
`[machines.<name>]` entry and copied from `[defaults]` for an imported one. On a laptop of 128 GB
that leaves too little for the operating system, a browser and an editor (a package needing
112.7 GB ranked `tight` and first); on a tester's machine of 4 GB the pool came out below zero and
the note said `-4.2 GiB`. The reserve was also invisible: the head under each table named the free
graphics memory of the moment, a number the fit never uses, and the rows in system memory named no
budget at all.

## Decision

1. **The reserve is a share of the memory** (`modelroom/reserve.py`, GF 2026-09-29, question 1 (a)):
   a quarter of the physical RAM, at least 2 and at most 16 GiB. Unified memory takes a quarter
   **without** the upper bound, plus the card reserve (question 4 (a)): 128 GiB -> 32 + 1, pool 95.
   `fit._choose_pool` reads it through `reserve.ram_reserve` and nowhere else, so every entry point
   -- the fit of a profile, from the size, from a parameter count, the size scale of the guided mode
   -- computes with the same number. A size fit without a known RAM tries the cards first and counts
   the system memory as 0.
2. **A number entered wins** and is never overwritten, `0.0` included (origin `entered`). Without one
   the reserve is computed at every run (`computed`); with neither a number nor a known RAM it is
   `0.0` (`unknown`).
3. **Configuration schema 3 stays; the key becomes optional** (question 2 (a)). A new folder writes
   no `reserve_ram_gib`; a new entry copies a number from `[defaults]` as before and leaves the key
   out where `[defaults]` has none; an existing entry keeps its reserve through every write, a
   missing one included. `None` is left out by the TOML writer only.
4. **Existing folders are not changed** (question 3 (a)): a machine that carries the `8.0` of 0.1.0
   while the computed reserve differs gets one hint under its table,
   `memory reserve 8.0 GB is set in modelroom.toml; computed from this machine it would be <x> GB`.
5. **The render document stays schema 3, additive** (precedent ADR 0006, 0011): `MachineRanking`
   carries the reserve the fit took and `reserve_ram_origin`, default `entered`.
6. **The head names the budget** (U10): the line `room` says `graphics <budget> GB for models (<card>
   GB card minus reserve)`, `memory <budget> GB for models (<RAM> GB minus reserve)` (or one piece
   `shared memory …` on unified memory) and the free disk where Ollama keeps its models. What the
   daemon holds right now stands on a line `loaded` of its own, only when it holds a model, and
   changes no budget. Live values live in `room.LiveRoom` per machine name, in memory only; the
   budgets never come from them. A pool of 0 or less is said in words, never as a memory below zero.
7. **Each table ranks for its own machine** (question 5 (a)): with two ranked machines or more,
   `shown` says `ranked for this machine; another machine may show other rows`.
8. **A download larger than the free disk is marked, not ranked away** (finding U11 of the hand
   test, GF 2026-09-29: "(a), jetzt in RM", mockup approved). With 26.4 GB free the table, `install`
   and the pull list recommended packages of 57 to 94 GB, and the no came only at the pull. Rank and
   class stay (ADR 0009, the fit is memory only). Where the free disk of the machine was read, such a
   row lies on the house red `#C32F27` at 32 % over the terminal's black (`#471715`, white text) and
   carries ` ×` behind its download, also without color and in the Markdown file; a line `disk` in
   the lighter red `#F07A72` names the rows; `install` takes the best row that also fits the disk or
   says in red that none does; the pull list shows such rows in red with what they need, they cannot
   be chosen, and the pointer starts on the first that can. Another computer and an unknown disk
   change nothing. **The band is the first background color of modelroom** -- until now the colors
   were foreground only and the background the terminal's (decided 2026-09-24, `intro.py`); the
   exception holds for this band alone (the pointer row of a list keeps its own, `dialog.py`). The
   rule is the one of the check before a pull, the download plus the margin of 1 GiB, read from one
   function (`room.pull_need_bytes`; decided 2026-09-29 by the coordination, reported to the GF), so no
   row is offered that the check then refuses; the texts say `need more than` and `needs 94.1 GB plus
   1 GB room`. The check stays as the second safety. The `Download` column became 10 wide for the mark.

## Sources (retrieved 2026-09-29)

- llmfit, `docs/how-it-works.md` (https://github.com/AlexsJones/llmfit/blob/main/docs/how-it-works.md):
  the thresholds 60 / 85 / 98 %; no fixed OS reserve documented.
- llmfit issue #1086 and PR #1089 (https://github.com/AlexsJones/llmfit/issues/1086,
  https://github.com/AlexsJones/llmfit/pull/1089): a share instead of a fixed number, so that "some
  capacity remains available for the OS, display, desktop applications".
- Ollama issues #10920 and #11497 (https://github.com/ollama/ollama/issues/10920,
  https://github.com/ollama/ollama/issues/11497): Ollama compares with the memory free at load time;
  hence the budget in the head and the moment only as a hint.
- Ollama API, `GET /api/ps` (https://github.com/ollama/ollama/blob/main/docs/api.md): `name`, `size`,
  `size_vram` per loaded model.
- The Metal limit on Apple silicon (https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap):
  macOS gives the GPU about 75 % of the memory without a change; 8-16 GB should stay when it is raised.
- Windows 11 idle memory (https://www.digitaltrends.com/computing/how-much-ram-does-windows-11-use/):
  3-4 GB on a 16-GB machine, from user reports, no vendor figure.

## Consequences

- The laptop of 127.7 GB computes with a pool of 111.7 instead of 119.7; the package of 112.7 GB is
  `no_fit`, the largest that still fits needs at most 109.5 GB. A machine of 32 GB is unchanged; the
  tester's machine of 3.8 GB has a pool of 1.8 instead of -4.2.
- **Compatibility.** The new version reads every file 0.1.0 wrote (each carries the number, read as
  `entered`). The released 0.1.0 reads configuration schemas `[1, 3)` only; every file this version
  writes is schema 3, so 0.1.0 refuses it cleanly with exit `3` (checked by hand on 2026-09-29 with
  `uvx --from modelroom==0.1.0 modelroom render --config <folder>/modelroom.toml`). Only a file that
  says schema 1 or 2 and has the key removed by hand makes 0.1.0 stop with a field error (exit `2`).
  A foreign reader of `docs/models.json` meets the new field `reserve_ram_origin`.
- Not here: crediting the card's share in `cpu_gpu`, reading `iogpu.wired_limit_mb`, the free system
  memory as an input, remote machines (AP9-FR, ADR 0013), `OLLAMA_GPU_OVERHEAD`, the vLLM fit.
