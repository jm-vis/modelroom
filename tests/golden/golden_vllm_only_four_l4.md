<!-- modelroom render: snapshot_run_at=2026-09-30T08:00:00+00:00 rendered_at=2026-09-30T08:05:00+00:00 -->

# Model packages

Snapshot run at: 2026-09-30T08:00:00+00:00
Rendered at: 2026-09-30T08:05:00+00:00
Base models / packages: 1 / 1
Scenario: context 32k (entered), KV cache f16 (assumed), 10 requests
Load: beyond 8 requests at once this fit says nothing about throughput: measure under load, or look at a serving stack (rule of thumb, not measured)
Ranking rule: graphics memory before system memory, then every model that is not legacy before a legacy one, then the larger model (parameter count), then the least compression, then measured before unmeasured (a valid comparable measurement), then larger weights, then package identity; the first 10 rows, then every measured package the rule leaves out

## Areas

| Source | Base model | Packager | Status | Last success | Error |
|---|---|---|---|---|---|
| huggingface | Qwen/Qwen3-8B | RedHatAI | complete | 2026-09-30T08:00:00+00:00 |  |

## Machines

| Machine | Profile | Origin | RAM GiB | RAM source | VRAM GiB | VRAM source | GPU state | GPUs | Class | Reserve RAM GiB | Reserve VRAM GiB | Recorded at | Profile age (days) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| server | server (5d8f0374c7e19a2b) | measured | 94.29 | os | 89.96 | nvidia-smi | multi_gpu | 0 NVIDIA L4 22.49; 1 NVIDIA L4 22.49; 2 NVIDIA L4 22.49; 3 NVIDIA L4 22.49 | unknown (unknown) | 8.00 | 1.00 | 2026-09-28T09:14:03+00:00 | 1 |

GPU state: measured = a counted card, read by modelroom or llmfit; entered = typed in by hand; multi_gpu = several counted cards, added up; present_unmeasured = a card may be there, nothing about it was read; none = no counted card (none found, none usable, display adapter only, or --cpu-only); unified_memory = system and card share one memory; unsupported_platform = not read on this platform; multi_gpu_not_covered = several cards, measured before the fit covered them; legacy_unknown = from a schema-1 profile

Class: laptop, workstation or server, as the chassis reports it (chassis) or as the user gave it (entered); unknown where neither says. It changes no number.

## Ranking: server

server -- ranked

Showing 1 of 1 ranked packages.

| # | Base model | Stars | Packager | Quant | Format | Weights GiB (computed) | Fit (computed) | Need GiB (computed) | Pool GiB (computed) | Package context | Measured group | Speed tok/s (measured) | Provenance | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen/Qwen3-8B | – | RedHatAI | AWQ | tensor | 5.68 | fits (gpu_split) | 56.68 | 82.76 | unknown | 1 | – | from_metadata | vLLM holds 15 requests at once on 4 cards of 22.5 GiB; the 10 requests need about 56.7 GiB of the 82.8 GiB it takes (0.92 of each card). |

Fit is memory only: how many requests of this context the cards hold at once · ample: twice the requests this ranking assumes or more · fits: that many, below twice them · tight: at least one, fewer than them · no fit: not one · the pool is 0.92 of each card vLLM uses, 1.50 GB of it its runtime · gpu: one card · gpu_split: 2, 4 or 8 equal cards of the largest size, as the head count allows · for a fit from size the highest class is fits · best: the least compression of the same model that fits · modelroom rates no model family against another

Speed is measured, see its column.
