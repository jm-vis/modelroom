<!-- modelroom render: snapshot_run_at=2026-09-22T09:00:00+00:00 rendered_at=2026-09-22T10:00:00+00:00 -->

# Model packages

Snapshot run at: 2026-09-22T09:00:00+00:00
Rendered at: 2026-09-22T10:00:00+00:00
Base models / packages: 1 / 2
Scenario: context 8192 (default), KV cache f16 (assumed), 1 request
Ranking rule: fit class (perfect, good, marginal), then measured group (valid comparable measurement first), then measured speed (faster first), then quantization, then larger weights, then package identity

## Areas

| Source | Base model | Packager | Status | Last success | Error |
|---|---|---|---|---|---|
| huggingface | acme/Nova-8B | packager | complete | 2026-09-22T09:00:00+00:00 |  |

## Machines

| Machine | Profile | Origin | RAM GiB | RAM source | VRAM GiB | VRAM source | GPU state | Reserve RAM GiB | Reserve VRAM GiB | Recorded at | Profile age (days) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| workstation | workstation (3f9a0c21d4e6b870) | measured | 127.46 | os | 11.94 | nvidia-smi | measured | 16.00 | 1.00 | 2026-09-22T09:00:00+00:00 | 0 |

## Ranking: workstation

workstation -- ranked

Showing 1 of 1 ranked packages.

| # | Base model | Stars | Packager | Quant | Format | Weights GiB (computed) | Fit (computed) | Need GiB (computed) | Pool GiB (computed) | Package context | Measured group | Speed tok/s (measured) | Provenance | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | acme/Nova-8B | – | packager | Q4_K_M | gguf | 5.00 | good (gpu) | 7.12 | 10.94 | unknown | 1 | – | metadata_ok | Fits into graphics memory: it needs about 7.1 GiB of the 10.9 GiB left after the reserve. |

## Too tight: workstation

| Base model | Stars | Packager | Quant | Format | Weights GiB (computed) | Package context | Reason | Note |
|---|---|---|---|---|---|---|---|---|
| acme/Nova-8B | – | packager | Q4_K_M | gguf | 400.00 | unknown | does not fit this machine | Fit contract v1 does not count this as a fit: it needs about 441.6 GiB of the 111.5 GiB left after the reserve. |
