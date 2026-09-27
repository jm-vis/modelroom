# 0004 -- Hardware profile schema 4: every graphics card, the machine class

- **Status:** accepted, decided 2026-09-26
- **Supersedes:** nothing (ADR 0001 stays in force for schema 3; this adds the next step)

## Context

Profile schema 3 knows one VRAM value and no machine class. `measure.py` reads the first line of
`nvidia-smi` only: two lines or more end as `multi_gpu_not_covered`, with no VRAM, no name and no
fit. A server with several cards is therefore a machine the tool cannot rank, and nothing in the
profile says whether a machine is a laptop, a workstation or a server -- a serving stack needs
both facts to decide how to run a model at all.

The card list and the class are new **fields** of a persisted form with `extra="forbid"`: a 0.1.0
reader would fail on them with a validation error, not with the version message. Schema versions
are half-open ranges (CONTRACTS.md, "Schema versions"), so the step is a new version.

## Decision

- `PROFILE_SCHEMA_VERSION = 4`, readers accept `[1, 5)`.
- `gpus: list[GpuAdapter]`, each card `{index, name, vram_gib, vram_source}`. The list is not
  empty exactly for `gpu_state` `measured`, `entered` and the new `multi_gpu`: one card for the
  first two, with the profile's own `vram_gib` and `gpu_name`; two or more for `multi_gpu`,
  indexes `0 ... n-1`, `vram_gib` their sum. `name` has no length rule, like `gpu_name`, and is
  `None` for a card entered by hand. An empty list says nothing about `vram_gib`: a profile
  migrated from schema 1 keeps llmfit's VRAM next to an empty list.
- `multi_gpu` is a state the fit computes for, measured (`vram_source` `nvidia-smi`) or entered
  by hand (`vram_source` `entered`, an entered memory, no llmfit check, cards of one size).
  `multi_gpu_not_covered` stays in the value set so that files of 0.1.0 still read, and no path
  writes it any more.
- `machine_class: laptop | workstation | server | unknown` with `machine_class_source: chassis |
  entered | unknown`; `unknown` exactly with source `unknown`. `chassis` is the SMBIOS chassis
  type (Windows `Win32_SystemEnclosure`, Linux `/sys/class/dmi/id/chassis_type`), mapped by a
  positive list per class; it requires a memory not entered by hand, so `hardware --cpu-only`
  keeps it. `hardware --machine-class` and the guided questions write `entered`. The class
  changes no number of the fit.
- Reading goes in two steps with a fixed target each: schema 2 to 3 (`normalize_profile_v2`),
  schema 3 to 4 (`normalize_profile_v3`: one card from the state and `gpu_name`, class `unknown`).
  A file that says schema 2 or 3 and carries `gpus`, `machine_class`, `machine_class_source` or
  `multi_gpu` is refused. Both readers of a stored profile take the same chain
  (`read_profile_document`, and the profile inside an export file); `import-profile` delegates.
- Only `modelroom migrate` writes back: in place, the backup named after the stored version
  (`.v2.bak`, `.v3.bak`); a run of an earlier version that stopped halfway is recognized by
  reading the target it left, whatever its schema.
- llmfit's cross-check of several cards compares the sum with the sum of llmfit's own `gpus[]`
  entries of backend `CUDA` (`vram_gb x count`); without such a list the check is `absent`.

## Consequences

- A 0.1.0 reader meets a schema-4 profile with the version message and exit `3`.
- **Shared results folders need updated readers**: once one machine writes schema 4, every
  machine that reads the folder needs this version or later. An export file stays schema 1 and
  carries a schema-4 profile; a 0.1.0 `import-profile` refuses it on the profile's validation.
- The two-card and three-card fixtures are synthetic until a rented GPU machine supplies a
  recorded one; the README's "Tested on" table names every machine a release was really run on.
- A card of another vendor next to an NVIDIA card is not in the list (`nvidia-smi` lists its own),
  and the list shows what `nvidia-smi` shows, whatever `CUDA_VISIBLE_DEVICES` hides from a daemon.

## Alternatives and why not

- **Keep one VRAM value and add a card count.** Rejected: mixed cards and the rule "one card
  first" need each card's own size.
- **Guess the class from the host name or the RAM.** Rejected: a guess is not a measurement; the
  chassis type is, and where it says nothing the user says it.
