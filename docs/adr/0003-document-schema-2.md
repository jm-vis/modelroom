# 0003 -- Render document schema 2: the requests of every fit and where they came from

- **Status:** accepted, decided 2026-09-25
- **Supersedes:** nothing

## Context

`modelroom render` writes `docs/models.json`, the render document, next to the Markdown view.
No reader in this package reads it back, but it is a persisted form that other tools read, and
the rule for every persisted form holds for it: a changed shape is a new schema version.

With parallel requests (ADR 0002) a fit computes the KV cache once per request, so a fit is only
readable together with the number it was computed for. And the Markdown view is built from the
document alone -- it never sees the configuration -- so the head count and the origin of the
requests have to be in the document to be shown at all.

## Decision

- `DOCUMENT_SCHEMA_VERSION = 2`.
- Every `Fit` carries `requests` (`1..1024`, default 1: a fit nobody could compute says 1).
  Every ranked and too-tight fit of a document was computed for `scenario.requests`.
- The document carries `users` and `requests_origin`, copied from the configuration's
  `[guided]` when the rendered scenario is that configuration's -- when no scenario was passed,
  and also when one was passed with the same requests (the guided mode always passes one). An
  explicitly passed scenario of other requests leaves both `None`, so the document never names
  a wrong origin. The same rules as in the configuration hold between the three values.
- A ranked row whose measurement did not count only because of `requests` (a measurement runs
  one request) carries that reason as its note: `measured with 1 request, ranking assumes N`.
  The measurement file never changes.

## Consequences

- A tool that checks `schema_version == 1` of `docs/models.json` has to learn schema 2.
- The views (terminal, Markdown) do not show the new fields in this step; a later work package
  adds the head count to the scenario line and the reason next to the speed.

## Alternatives and why not

- **Keep schema 1 and add optional fields.** Rejected: a reader of schema 1 with
  `extra="forbid"`, as this package's own model is, would fail on validation.
- **Leave the origin out of the document.** Rejected: the Markdown view could then not say
  `3 requests (from 25 users)`, and a JSON reader could not tell an entered number from a derived
  one.
