---
typ: moc
---

# Architecture decision records

One line per decision; an ADR is immutable, a changed situation gets a new one that supersedes it, except where a build brief orders the final state of a still-open series, dated in the ADR.

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
| [0010](0010-hybrid-architecture-age-and-pull-choice.md) | A hybrid architecture read, the age of a line, the publisher's repository a target, the package chosen, the room read live | accepted, 2026-09-28; partly superseded by 0012 |
| [0011](0011-graphics-adapters-llmfit-fallback.md) | Graphics adapters of every vendor, llmfit as fallback, schema 4 extended without a step | accepted, 2026-09-28 |
| [0012](0012-memory-reserve-and-table-head.md) | The memory reserve after the machine's size, the table head names the budget | accepted, 2026-09-29 |
| [0013](0013-another-computer-three-ways-neutral-names.md) | Another computer: three ways in (copy the command, a shared folder, over SSH), a neutral name at every machine boundary | accepted, 2026-09-29 |
| [0014](0014-runtime-entries-and-vllm-packages.md) | A local-server runtime is one table entry, never a plug-in; snapshot schema steps to 2 for vLLM's tensor packages and the new `QUANT_ORDER` labels; `HardwareProfile.runtimes` extended without a step; the smallest tensor-parallel size that fits (2026-10-03) | accepted, 2026-09-29; brought to its final state on 2026-10-03 by the build brief's own instruction (AP9-SV2-D, D4), the one exception to the index rule; later changes get a new ADR |
| [0015](0015-tempo-facts.md) | Tempo facts are shown, never a sort key; quality is a link in step 5 and after `render`, not in Markdown or JSON | proposed, 2026-10-03 (until AP9-T-K) |
