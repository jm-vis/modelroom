# 0007 -- The identifiers follow the words

- **Status:** accepted, decided 2026-09-27
- **Supersedes:** two sentences of ADR 0006, named below; ADR 0006 stays in force for everything
  else

## Context

ADR 0006 renamed the fit classes to words that say what was computed and left four identifiers
with the old words: the provenance value `metadata_ok`, the field `too_tight` of
`MachineRanking` and `ranking.Ranking` (the JSON key of the render document) together with the
note code of the same name, the field `Requirement.perfect_reachable` and the note code
`cpu_caps_at_good`. It called a rename "a schema decision of its own". That decision was taken
the same day, the other way: the identifiers follow the words.

Two of the four are persisted. `metadata_ok` stands in snapshot schema 1, which 0.1.0 wrote and
which every command reads before it writes (`fetch` reads the old file first, the F13 gate). The
field and note codes of the render document stand in render document schema 3, which is not
released. `perfect_reachable` is not persisted.

## Decision

1. **Four renames, nothing else.** `metadata_ok` becomes `from_metadata`, `too_tight` becomes
   `no_fit` (field and note code), `perfect_reachable` becomes `ample_reachable`,
   `cpu_caps_at_good` becomes `cpu_caps_at_fits`. No threshold, no cap and no order of the fit
   changes.
2. **Snapshot schema 1 stays, with one read exception.** `contracts.load_snapshot`, the one
   reader of every command, reads a package whose `provenance` is `metadata_ok` as
   `from_metadata`, after the schema-version check and before the model validates. It returns a
   new dict and leaves its input unchanged; no other field is touched and `schema_version` stays
   `1`. `write_snapshot` validates through the same reader, so the next `fetch` writes
   `from_metadata` back. The model itself accepts only the new value.
3. **Render document schema stays 3.** It is not released (0.1.0 wrote 1), the same rule ADR 0006
   applied to profile schema 4, and the package never reads a render document back.
4. **No migration by hand.** The note code and the fit class now share the spelling
   `no_fit`, as `not_covered` is already both a block of `MachineRanking` and a note code.

## Consequences

- The old spelling `metadata_ok` is read once and never written again.
- Snapshot schema 1 accepts `metadata_ok` as `from_metadata` on read and writes only
  `from_metadata` (ADR 0007). Nothing to migrate by hand. The released 0.1.0 reader cannot read
  a rewritten snapshot: every machine that shares the results folder updates before the next
  `fetch` writes.
- A tool of its own that reads `docs/models.json` or the snapshot learns four names.
- llmfit's own class word `too_tight` is llmfit's name, not an identifier of this package, and
  stays where the contract describes llmfit.

## Superseded sentences

- ADR 0006, Consequences, "Identifiers stay (A6)": "`MachineRanking.too_tight` and
  `ranking.Ranking.too_tight` (the JSON key `too_tight`), `Requirement.perfect_reachable` and the
  note codes `too_tight` and `cpu_caps_at_good` are identifiers of the data contract, not words a
  reader is shown. They are not renamed here; a rename is a schema decision of its own." They are
  renamed (decision 1).
- ADR 0006, Alternatives, "Rename the field names and note codes in the same step. Rejected for
  now: they are read by tools, not by people, and renaming them is a schema decision of its own."
  That schema decision is this record.

## Alternatives and why not

- **Snapshot schema 2 with a migration.** Rejected: the value is the only change, the read
  exception is lossless, and a version step would make every 0.1.0 results folder a migration
  case for one word.
- **Keep the identifiers.** Rejected by the decision of 2026-09-27: an identifier that carries a
  word the prose bans keeps the old judgement alive for every tool that reads it.
