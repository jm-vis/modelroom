# 0001 -- Hardware profile schema 3: a machine entered by hand, unified memory computed

- **Status:** accepted, decided 2026-09-25
- **Supersedes:** nothing (first ADR of this repository)

## Context

Version 0.1.0 writes hardware profile schema 2 (`modelroom/profile.py`). Its value sets are
closed: `ram_physical_source` is `os | llmfit | unknown`, `vram_source` is
`nvidia-smi | llmfit | none | unknown`, `gpu_state` has seven values, and `unified_memory` is one
of the states the fit refuses. `origin = "entered"` exists, but only `hardware --cpu-only` writes
it, and its memory is still measured by the operating system.

Two coming work packages need more: a machine entered by hand (its memory, and either its own
graphics card, none, or unified memory) and a fit for unified memory (Apple M, AMD Strix Halo,
Intel Core Ultra, DGX Spark). Both are new **values** in existing fields. Schema versions are
half-open ranges (CONTRACTS.md, "Schema versions"): a 0.1.0 reader accepts profile schema 1 and
2. Written as schema 2, a profile with the value `entered` passes that version check and then
fails field validation -- `migrate` stops with `MigrationError`, `import-profile` and `render`
list the file as unreadable, `hardware` can fail on a foreign profile in a shared folder. The
reader cannot say why.

## Decision

- `PROFILE_SCHEMA_VERSION = 3`, readers accept `[1, 4)`. Same fields as schema 2; new values:
  `ram_physical_source` and `vram_source` gain `entered`, `gpu_state` gains `entered`, and
  `unified_memory` becomes a state the fit computes for (`FIT_GPU_STATES` = `none`, `measured`,
  `entered`, `unified_memory`).
- Coupled rules: an entered memory or graphics memory requires `origin = "entered"`;
  `gpu_state = "entered"` and `vram_source = "entered"` only come together, with `vram_gib > 0`
  and an entered memory; a machine whose memory was entered is exactly one of three shapes --
  its own graphics card of N GiB (`entered`/`entered`/N), no graphics card (`none`/`none`/0) or
  unified memory (`unified_memory`/`none`/0) -- with no llmfit cross-check. A `--cpu-only`
  profile (`origin = "entered"`, memory from the OS) stays valid and keeps its identity.
- Unified memory is one pool: RAM minus both reserves, mode `gpu`, no fallback onto the same
  RAM; a pool of 0 or less is `too_tight`. A profile migrated from schema 1 never computes, its
  `unified_memory` included: it carries llmfit readings only.
- A schema-2 file reads as schema 3 on every path: `read_profile_document`, the profile inside
  an export file (`load_export`; the export itself stays schema 1) and the probe profile
  `render` builds at import time. A file that says schema 2 but carries a schema-3 value is
  refused. Only `modelroom migrate` writes a schema-2 profile back, in place under the same name,
  keeping `<id>.json.v2.bak`.
- A profile entered by hand (`origin` and `ram_physical_source` both `entered`) is never the
  profile a measurement adopts: `binding.KnownProfile` carries both marks, and the takeover rule
  gives such a bound or configured profile the action `new` before it compares fingerprints, also
  for `--same-machine`.

## Consequences

- A 0.1.0 reader meets a schema-3 profile with a clean refusal: `schema_version 3 is outside the
  accepted range (>= 1 and < 3)`, exit `3`, instead of a validation error about one value.
- **Shared results folders need updated readers.** Once one machine writes schema 3, every
  machine that reads the folder needs this version or later. This is the honest limit, stated in
  the changelog. An export file written now stays schema 1 but carries a schema-3 profile inside:
  a 0.1.0 `import-profile` fails on that profile with a validation error (`schema_version must be
  2, got 3`), exit `3` as well, not with the version message -- the export's own version is not
  raised for a change inside it.
- Every fit on unified memory is `computed`; no such machine has been measured yet. The rules are
  tested with synthetic profiles; a real profile of such a machine is still to come.

## Alternatives and why not

- **Stay on schema 2, add the values.** Rejected: 0.1.0 readers would fail on field validation
  with a message about a value, not about a version -- the failure the version exists to prevent.
- **Feed unified memory in as graphics memory** (`vram_gib = ram`). Rejected: the GPU branch
  would subtract only the graphics reserve and compute with 31 GiB of a 32 GiB machine; the
  system reserve would vanish.
- **A migration layer that converts on every read and write.** Rejected: one lossless step
  (a version number) at four fixed places is smaller and easier to test than a layer.
