<!-- modelroom render: snapshot_run_at=2026-09-30T08:00:00+00:00 rendered_at=2026-09-30T08:05:00+00:00 -->

# Model packages

Snapshot run at: 2026-09-30T08:00:00+00:00
Rendered at: 2026-09-30T08:05:00+00:00
Base models / packages: 1 / 2
Scenario: context 8k (entered), KV cache f16 (assumed), 1 request
Ranking rule: graphics memory before system memory, then every model that is not legacy before a legacy one, then the larger model (parameter count), then the least compression, then measured before unmeasured (a valid comparable measurement), then larger weights, then package identity; the first 10 rows, then every measured package the rule leaves out

## Areas

| Source | Base model | Packager | Status | Last success | Error |
|---|---|---|---|---|---|
| huggingface | Qwen/Qwen3-8B | unsloth | complete | 2026-09-30T08:00:00+00:00 |  |

## Machines

| Machine | Profile | Origin | RAM GiB | RAM source | VRAM GiB | VRAM source | GPU state | GPUs | Class | Reserve RAM GiB | Reserve VRAM GiB | Recorded at | Profile age (days) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| server | laptop (4c7e19a2b5d8f038) | measured | 31.62 | os | 8.00 | nvidia-smi | measured | 0 NVIDIA GeForce RTX 4060 Laptop GPU 8.00 | laptop (entered) | 8.00 | 1.00 | 2026-09-28T09:14:03+00:00 | 1 |

GPU state: measured = a counted card, read by modelroom or llmfit; entered = typed in by hand; multi_gpu = several counted cards, added up; present_unmeasured = a card may be there, nothing about it was read; none = no counted card (none found, none usable, display adapter only, or --cpu-only); unified_memory = system and card share one memory; unsupported_platform = not read on this platform; multi_gpu_not_covered = several cards, measured before the fit covered them; legacy_unknown = from a schema-1 profile

Class: laptop, workstation or server, as the chassis reports it (chassis) or as the user gave it (entered); unknown where neither says. It changes no number.

## Ranking: server

laptop -- ranked

Memory reserve 8.0 GB is set in modelroom.toml; computed from this machine it would be 7.9 GB.

Showing 2 of 2 ranked packages.

| # | Base model | Stars | Packager | Quant | Format | Weights GiB (computed) | Fit (computed) | Need GiB (computed) | Pool GiB (computed) | Package context | Measured group | Speed tok/s (measured) | Provenance | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen/Qwen3-8B | – | unsloth | Q4_K_M | gguf | 4.68 | tight (gpu) | 6.78 | 7.00 | unknown | 1 | – | from_metadata | Fits into graphics memory: it needs about 6.8 GiB of the 7.0 GiB left after the reserve. |
| 2 | Qwen/Qwen3-8B | – | unsloth | Q8_0 | gguf | 8.11 | fits (cpu_gpu) | 10.55 | 23.62 | unknown | 1 | – | from_metadata | Too large for graphics memory alone, so it runs in system memory with the graphics card helping; the best rating it can reach here is 'fits'. |

Fit is memory only: the package's need against a pool · ample: up to 60 % of the pool · fits: up to 85 % · tight: up to 98 % · no fit: more · the pool is what the mode names, minus its reserve (per card) · gpu: one card · gpu_split: all cards together · cpu_gpu, cpu (RAM): the system memory · system memory keeps a quarter as reserve, 2 to 16 GB, unless modelroom.toml sets it · one memory shared by system and card counts as gpu, a quarter kept, no upper limit · off the card, and for a fit from size the highest class is fits · best: the least compression of the same model that fits · modelroom rates no model family against another

Speed is measured, see its column.
