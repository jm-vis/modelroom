---
typ: moc
---

# Architecture decision records

One line per decision; an ADR is immutable, a changed situation gets a new one that supersedes it.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-profile-schema-3.md) | Hardware profile schema 3: a machine entered by hand, unified memory computed | accepted, 2026-09-25 |
| [0002](0002-configuration-schema-3.md) | Configuration schema 3: the assumed parallel requests in `[guided]` | accepted, 2026-09-25 |
| [0003](0003-document-schema-2.md) | Render document schema 2: the requests of every fit and where they came from | accepted, 2026-09-25 |
| [0004](0004-profile-schema-4.md) | Hardware profile schema 4: every graphics card, the machine class | accepted, 2026-09-26; partly superseded by 0011 |
| [0005](0005-document-schema-3.md) | Render document schema 3: a fit spread over every graphics card | accepted, 2026-09-26 |
| [0006](0006-rating-words-class-question-llmfit-safety-net.md) | Rating words, the class question on every measurement, llmfit as a safety net | accepted, 2026-09-27; partly superseded by 0011 |
| [0007](0007-identifiers-follow-the-words.md) | The identifiers follow the words: `from_metadata`, `no_fit`, `ample_reachable`, `cpu_caps_at_fits` | accepted, 2026-09-27 |
| [0008](0008-local-servers-vllm-read-per-run.md) | Local servers as a list, vLLM read per run: start row, load test at once, fit on one card, nothing persisted | accepted, 2026-09-27; partly superseded by 0009 |
| [0009](0009-largest-model-least-compression-first.md) | The largest model that fits, the least compression of it, the measurement appended, only the choice of the run | accepted, 2026-09-28 |
| [0010](0010-hybrid-architecture-age-and-pull-choice.md) | A hybrid architecture read, the age of a line, the publisher's repository a target, the package chosen, the room read live | accepted, 2026-09-28 |
| [0011](0011-graphics-adapters-llmfit-fallback.md) | Graphics adapters of every vendor, llmfit as fallback, schema 4 extended without a step | accepted, 2026-09-28 |
