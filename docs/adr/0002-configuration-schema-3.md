# 0002 -- Configuration schema 3: the assumed parallel requests in `[guided]`

- **Status:** accepted, decided 2026-09-25
- **Supersedes:** nothing

## Context

A server that answers several people needs more KV cache: Ollama loads the weights once and
keeps one KV cache per parallel slot (`OLLAMA_NUM_PARALLEL`). The ranking computed for one
request only; `Scenario.requests` above 1 made every fit `unknown`.

The number of requests has to live somewhere a later `modelroom render` reads it. `Scenario` is
embedded in the measurement file (schema 2), the export file (schema 1) and the render document;
a field there would move three forms at once. The configuration's `[guided]` table already keeps
the context a guided run chose for the same reason.

Configuration schema versions are half-open ranges too: a 0.1.0 reader accepts schema 1 and 2.
A new key in `[guided]` (`extra="forbid"`) in a file that still says schema 2 would make such a
reader fail on field validation instead of on the version.

## Decision

- `CONFIG_SCHEMA_VERSION = 3`, readers accept `[1, 4)`. `[guided]` gains three keys:
  `users` (`1..10240` or absent -- the head count a user named), `requests` (`1..1024`, default
  1 -- the parallel slots the ranking assumes) and `requests_origin` (`default | entered |
  from_users`, default `default`).
- Rules: requests named directly are `entered` (a head count may stay for the display); a head
  count alone is `from_users` with `requests = min(1024, max(1, ceil(users x 0.10)))`, the share
  `ACTIVE_SHARE = 0.10` being a rule of thumb, not measured; neither is `default` with one
  request and no head count. `requests` and `requests_origin` are written together or not at all;
  a file that breaks a rule is invalid (exit `2`).
- `Scenario` stays unchanged; `render_cmd.scenario_from_config` reads `[guided].requests`
  whether a context is kept or not.
- Schema 2 reads as schema 3 in memory (`load_config`, `Configuration.from_dict`): every value
  stays, a present `[guided]` table gets one request with origin `default`, a missing one stays
  missing. A schema-2 file with a key of schema 3 is refused. Reading never writes the file.
- Three paths write the file back as schema 3: `modelroom migrate` (backup
  `modelroom.toml.v2.bak`, next to an earlier `.v1.bak`), its trigger at the start of a guided run
  (same backup, one note on the screen), and `import-profile`, which rewrites a schema-2 file
  even when it adds no machine and keeps `modelroom.toml.v2.bak` next to its own `.bak` (that one
  is never overwritten and may hold an earlier state). A second run writes nothing.

## Consequences

- A 0.1.0 reader meets a schema-3 configuration with a clean refusal and exit `3`.
- Shared results folders need updated readers once one machine has written schema 3 -- the same
  limit as for profile schema 3 (ADR 0001).
- The dialog is unchanged in this step: a guided run assumes one request by default. Asking for
  the head count is a later work package that writes these keys.

## Alternatives and why not

- **A field in `Scenario`.** Rejected: measurement, export and render document would all move a
  schema for one number, and a measurement always runs one request anyway.
- **Keep schema 2 and add optional keys.** Rejected for the reason above: a 0.1.0 reader fails on
  validation, not on the version.
- **Ask "how many at the same time".** Rejected by the product decision of 2026-09-25: people
  know how many use a server on a typical day, not how many press enter in the same second. The
  derivation stays visible and marked as a rule of thumb.
