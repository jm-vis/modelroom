# Contracts

How shared data shapes are defined, versioned and tested in this repository. The models
themselves land in `modelroom/contracts.py` with the first feature work package; this file
fixes the rules they follow so that fetcher, hardware probe, renderer and any external
consumer build against the same expectations.

## Source of truth

- `modelroom/contracts.py` holds Pydantic models with `extra="forbid"`, explicit value ranges
  and validators for coupled fields. It is the only definition. `Fit` and `Rating` are defined in
  `modelroom/contracts_fit.py` (AP9-SV2-0, a file split for the size guard) and re-exported from
  `contracts.py`; both count as the definition, and every import site keeps using
  `from .contracts import Fit`/`Rating`.
- This file describes each model for readers and repeats its examples verbatim. A contract
  test loads every example from the module and validates it; another test checks that every
  model named here exists in the module and vice versa.
- Producers validate before they write. A file on disk that does not validate is a bug in the
  producer, not something readers repair.

## Language standard

Every field value, message and paragraph in this file is US English and uses the word list in
`AGENTS.md`, section "Language standard": `latest`, `legacy` (with its `successor`) and
`unknown` for where a model stands in its family; `measured`, `entered` and `computed` for
where a value came from; `stated` for what a local server says about itself (the model it serves),
which is never measured, never ranked and never provenance (AP9-SV1); `publisher`, `listed packager` and `other` for who owns a repository,
which is a class and never a verdict; `unknown` for a fact with no evidence behind it;
`not comparable` for measurements taken under different scenarios. Schema names, field names
and literal values are contract: they are never changed for a language reason, so a value like
`unified_memory` keeps its spelling even where the prose around it is rewritten. The
exceptions are the fit classes of 2026-09-27 (`perfect`, `good`, `marginal`, `too_tight` became
`ample`, `fits`, `tight`, `no_fit`), decided as a contract change by ADR 0006: those values were
judgements, not names; and the four identifiers that carried the old words, renamed by ADR 0007
the same day (`metadata_ok` became `from_metadata`, `too_tight` became `no_fit`,
`perfect_reachable` became `ample_reachable`, `cpu_caps_at_good` became `cpu_caps_at_fits`).
No other literal follows them.
`tests/test_language_standard.py` holds the rule in executable form.

## The three persisted forms

| Form | File | Written by | Read by |
|---|---|---|---|
| Configuration | `modelroom.toml` (or a `Configuration` object) | the user | every command |
| Snapshot | `<state>/modelroom.json` | `fetch` | `render` (and any operator-side freshness check) |
| Hardware profile | `<state>/hardware/<machine>.json` | `hardware`, on that machine | `render` |

Schema 2 (see "Schema 2: profiles, measurements, guided mode" at the end) adds four forms and
renames the profile file:

| Form | File | Written by | Read by |
|---|---|---|---|
| Hardware profile v2 | `<state>/hardware/<profile_id>.json` | `modelroom migrate`; the measuring step on that machine (later work package) | fit, ranking |
| Measurement | `<state>/measurements/<profile_id>/<measurement_id>.json` | `modelroom migrate`; the load test (later work package) | ranking |
| Export | any file the user moves between machines | export (later work package) | import (later work package) |
| Pointer file | `~/.modelroom/guided.json` (per user, outside the state) | the guided mode | the guided mode |
| Catalog | `modelroom/catalog.toml`, shipped in the package | this repository | the search |

Runtime-only files in the state directory (`modelroom.lock`, `run-status.json`, `*.tmp`) are
not contracts; they are documented with the command that owns them.

## Schema versions

Every persisted form carries an integer `schema_version`. Each reader declares the **half-open**
range `[low, high)` it accepts (`version >= low and version < high`), never described as
"inclusive" -- the upper bound itself is already refused. A file outside that range is refused
with exit code `3` and a message that names the file, its version and the accepted range
(stated as `>= low and < high`), before the command writes anything. A version bump is a
documented decision (`docs/adr/`), never a side effect.

| Form | Writes | Reads | Earlier versions | Decision |
|---|---|---|---|---|
| Snapshot | 2 | `[1, 3)` | 1 reads as 2, unchanged; the value `metadata_ok` of 0.1.0 reads as `from_metadata` | ADR 0007, 2026-09-27; ADR 0014, 2026-09-29 (schema step) |
| Configuration (`modelroom.toml`) | 3 | `[1, 4)` | 1 and 2 read as 3 in memory; `migrate` and the guided mode write 1 and 2 back, `import-profile` writes 2 back (1 needs `migrate`) | ADR 0002, 2026-09-25 |
| Hardware profile | 4 | `[1, 5)` | 1 read as `HardwareSnapshot`; 2 and 3 read as 4 everywhere (2 through 3), `migrate` writes them back | ADR 0001, 2026-09-25; ADR 0004, 2026-09-26; ADR 0006, 2026-09-27; ADR 0011, 2026-09-28 (`adapters`, no step); ADR 0014, 2026-09-29 (`runtimes`, no step) |
| Render document (`docs/models.json`) | 3 | not read back by this package | -- | ADR 0003, 2026-09-25; ADR 0005, 2026-09-26; ADR 0006, 2026-09-27 (the fit words) |
| Export file | 1 | `[1, 2)` | its profile is read like a profile file | -- |
| Measurement file | 2 | 2 | -- | -- |
| Remote machines (`~/.modelroom/remote.json`) | 1 | `[1, 2)` | -- (new with AP9-FR) | AP9-FR, 2026-09-29 |

AP9-SV1 changes no persisted form (ADR 0008): a vLLM server's reading and its load test live in
memory for one guided run and are written nowhere. AP9-UX changes no persisted form (ADR 0009): the
selection of a run (`RunSelection`) lives in memory, and the render document keeps schema 3 -- its
`ranked` carries other rows, and `rank` is the number of the row shown. AP9-UX2 (ADR 0010) keeps
snapshot schema 1 while `Architecture` gains the kind `hybrid_linear` and six `linear_*` fields,
by the precedent of ADR 0007 (see "Architecture"); the origins of a run's machines, the free memory
and disk and the pull choice live in memory; `search.json` keeps schema 3. AP9-GPU (ADR 0011) extends
hardware profile schema 4 without a step, by the precedent of ADR 0006: the list `adapters` and the
graphics memory sources `sysfs` and `sysctl`. An empty list is not written, so every file of an
earlier shape stays as it was; a file that says schema 2 or 3 and carries either is refused.
AP9-RM (ADR 0012) keeps configuration schema 3 and render document schema 3: `reserve_ram_gib` of a
machine and of `[defaults]` becomes optional (left out, it is computed from the machine at every run),
and a `MachineRanking` gains `reserve_ram_origin`, additive. **The limit of this choice, stated
plainly:** the new version reads every file 0.1.0 wrote (each carries the number, which reads as
`entered`). The released 0.1.0 reads configuration schemas `[1, 3)` only, and every file this
version writes is schema 3, so 0.1.0 refuses such a folder cleanly with exit `3` (checked by hand
against the published 0.1.0 on 2026-09-29). Only a file that says schema 1 or 2 and has the key
removed by hand makes 0.1.0 stop with a field error (exit `2`), since the schema number does not
change; the rule below needs a later schema number and does not cover that case. The render document
is not read back by this package; the new field concerns other readers only.
AP9-FR changes no shared persisted form: configuration 3, profile 4, document 3, export 1 and
measurement 2 stay. What changes is which values they may hold ("Neutral names"): from 0.2.0 on an
export, a profile and a configuration carry a neutral `display_name` and neutral machine keys, and a
record the forms of its fields the sink allows. The file of the user account `remote.json` is new
("Remote machines").
AP9-LS changes no schema version: `lmstudio` is a new value of `runtimes` and an allowed value of
`Fit.runtime`, which no render writes in 0.2.0 ("Local servers and the vLLM load test", LM Studio).
Every output of a machine without LM Studio stays byte for byte what it was.
AP9-SV2-0 (ADR 0014, 2026-09-29, E8) steps snapshot schema to 2: a Hugging Face `tensor` package
may now be `from_metadata`/`approved` (Ollama's own `tensor` packages, an MLX tag included, stay
locked to `unresolved`/`format`, see "Package" below), `Package` gains `quantization_source`
(additive, default `None`), and `quantization.QUANT_ORDER` gains six vLLM labels (`W4A16`, `AWQ`,
`GPTQ-4`, `GPTQ-8`, `W8A8`, `FP8`, see "Package" below for their placement). This is a real
version step, not an additive one without a step (the precedent of ADR 0006/0010/0011): a 0.1.0
reader that met either new shape under schema 1 would fail with an opaque field error instead of
the named, exit-`3` schema refusal this step gives it -- the choice the GF made at E8. A schema-1
file still reads as schema 2 unchanged (`load_snapshot`'s own normalization; every new field has a
default, so nothing else needs converting). Hardware profile schema 4 is extended again without a
step (the ADR 0011 precedent): `runtimes: list[RuntimeKey]`, additive, excluded when empty (F6's
GF decision; see "Hardware profile v2" / `HardwareProfile` below). `RuntimeKey` and the planned
`RUNTIMES` table (one local-server runtime as one entry, "Laufzeit als Eintrag") are AP9-SV2-L's
subject; this step only reserves the field so later sub-packages do not need a schema change of
their own to fill it.

A reader of an earlier version meets a file of a later schema with a clean refusal and exit
`3`. **A results folder shared between machines therefore needs the same or a later version on
every machine once one of them has written a later schema** (decided 2026-09-25).

Snapshot schema 1 accepts `metadata_ok` as `from_metadata` on read and writes only
`from_metadata` (ADR 0007). Nothing to migrate by hand. The released 0.1.0 reader cannot read
a rewritten snapshot: every machine that shares the results folder updates before the next
`fetch` writes.

`modelroom.contracts.load_snapshot(data)` is the entry point for reading a persisted snapshot:
it checks `schema_version` against `SNAPSHOT_SCHEMA_RANGE` with `check_schema_version` *before*
handing the payload to `Snapshot.model_validate`, so a missing, non-integer or out-of-range
version always raises `SchemaVersionError` -- never a pydantic `ValidationError` mixed in with
unrelated field problems. `Snapshot` itself still carries its own `schema_version` check as a
second line of defense for callers that build one another way.

## Identity

Package identity is the repository identifier as the registry spells it (`hf_repo` for
Hugging Face, `ollama_base` plus tag for Ollama). The package never normalizes, canonicalizes
or merges model names on its own; whoever integrates it maps identities on their side. Every
recorded fact is bound to the revision it was observed at: the commit SHA on Hugging Face,
the manifest digest on Ollama.

The internal `(source, repo|ollama_name, filename-or-tag)` triple that `Snapshot` uses to reject
duplicate packages (see the `Snapshot` model below) never depends on file order: for a Hugging
Face package, its third element is the sorted, de-duplicated set of the identity stems of
every `weights`/`weights_shard` file (shard suffix removed, file extension kept), joined with
`|`: a sharded package's shards collapse to one identity, while `Nova-7B-BF16.gguf` and
`Nova-7B-BF16.safetensors` in the same repo stay two distinct packages. For an Ollama package
it is the tag part of `ollama_name` after the colon.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success: every fetch area is `complete` (`render`: the document was written, even when its rating source failed) |
| `1` | at least one fetch area ended `incomplete` (complete areas were still published); or `fetch`/`render` stopped at another process's lock; or `fetch`'s `run_at` is not newer than the stored snapshot's; or `render` has no snapshot to read; or `render`'s existing document was rendered from a newer snapshot than the one being rendered -- every case but the first leaves the snapshot, the hardware profiles, `run-status.json` and the rendered document unchanged; the lock file may be created or updated, because the lock is taken before the staleness checks (`cli.py`) |
| `2` | the configuration is missing or invalid; the named `--machine` is not a `writer` in this configuration (`fetch`) or not configured at all (`hardware`); `hardware`'s bound profile belongs to another machine (see "Hardware measurement") and neither `--same-machine` nor `--new-identity` says which it is; `hardware --same-machine` and `hardware --new-identity` were passed together; or an external tool a command *requires* is missing or below the minimum version -- since the measurement work package `hardware` no longer requires `llmfit`, which is a cross-check there |
| `3` | an input file (configuration, snapshot, or a hardware profile `render` reads) has an unsupported `schema_version`, is not valid JSON, or does not match its model (fix-round 5, P2-3: `cli.py`'s own `_read_snapshot`/`_read_hardware_snapshot` catch `json.JSONDecodeError`/pydantic `ValidationError` at every load site and name the file in the message, the same exit code as an unsupported `schema_version`) |

Since AP9-FR every writing command sweeps its folder of machine names first ("Neutral names"): a
sweep that cannot finish is exit `1` with nothing else written, and a broken
`~/.modelroom/remote.json` is exit `3`.

`modelroom pull <name>` (AP9-FR A4, no results folder and no configuration, so none of the codes
above apply to it) has its own table under "`modelroom pull`", below "Guided mode".

`modelroom remote <op>` (AP9-FR, FR-B1, "The remote protocol v1") is not interactive and not read
by a human directly: it always exits `0` once it managed to write a line at all -- a fault of the
call itself is an `error` line on stdout, never this command's own exit code. The one exception is
a crash before any line could be written, which is not expected in ordinary operation.

## Models

Every example below is `EXAMPLES["<ModelName>"]` from `modelroom/examples.py`, verbatim; a
contract test parses this file for `### <ModelName>` headings and their `json` blocks and
checks both directions: every model in the module is documented here, and every heading here
names a real model with a matching example.

### Architecture

A base model's transformer shape, read from the *publisher* repo's `config.json` (packager
GGUF repos never carry one; the reader is `modelroom/architecture.py`). `kind` is
`"dense_classic"` only when the three numeric fields are all present, `layer_types` is either
absent or entirely `"full_attention"`, and none of the MoE fields below signal a
mixture-of-experts model. It is `"hybrid_linear"` (AP9-UX2, ADR 0010) under exactly these
invariants and no MoE field: the three numeric fields; `layer_types` of `num_hidden_layers`
entries, each `"full_attention"` or `"linear_attention"`, both present; and the five
`linear_*` fields of the config -- Qwen3.5 is the case. `linear_state_bytes` is read from
`mamba_ssm_dtype` (`float32` 4, `bfloat16`/`float16` 2, a missing field 4 as a documented
assumption; another dtype is `"unknown"`). Any other config, MoE included, or a missing file
resolves to `"unknown"` with the numeric fields left at `None`. The MoE check looks at
both the top level of `config.json` and, when present, its `text_config`: any of
`num_local_experts`, `num_experts`, `n_routed_experts`, `num_experts_per_tok` or
`moe_intermediate_size` present with a value other than `null` rules out `dense_classic` even
when the three numeric fields all look plausible. Presence is the signal, the value is not
interpreted: a dense decoder does not carry these fields at all.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `source_repo` | `str` | -- | the publisher repo the config was read from |
| `source_revision` | `str \| None` | 40-hex commit sha, or `None` | the commit the config was read at |
| `kind` | `"dense_classic" \| "hybrid_linear" \| "unknown"` | see above | a plain decoder-only transformer, one with linear-attention layers between its full-attention ones, or neither |
| `num_hidden_layers` | `int \| None` | `0 < n <= 2**31 - 1`; required unless `kind == "unknown"` | transformer block count |
| `num_key_value_heads` | `int \| None` | `0 < n <= 2**31 - 1`; required unless `kind == "unknown"` | KV heads (for GQA/MQA KV-cache sizing) |
| `head_dim` | `int \| None` | `0 < n <= 2**31 - 1`; required unless `kind == "unknown"` | per-head dimension |
| `layer_types` | `list[str] \| None` | `None` or all `"full_attention"` when `dense_classic`; `hybrid_linear`: `num_hidden_layers` entries of `"full_attention"` and `"linear_attention"`, both present | per-layer attention kind, when the config states one |
| `num_attention_heads` | `int \| None` | `0 < n <= 2**31 - 1` | the attention head count, for vLLM's tensor-parallel rule (AP9-SV2-P/F) |
| `max_context` | `int \| None` | `0 < n <= 2**31 - 1` | the model's trained maximum context length |
| `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` | `int \| None` | `0 < n <= 2**31 - 1`; required when `hybrid_linear`, `None` otherwise | the state of a linear-attention layer (AP9-UX2) |
| `linear_state_bytes` | `int \| None` | `0 < n <= 2**31 - 1`; required when `hybrid_linear`, `None` otherwise | bytes of one state element, from `mamba_ssm_dtype` (4 when the config names none) |

**Persisted form (ADR 0010).** The snapshot keeps schema 1 with the new `kind` value and the six
fields, by the precedent of ADR 0007: a snapshot the 0.1.0 reader wrote reads on (the fields are
`None`), and the next `fetch` reads `config.json` again; the 0.1.0 reader does not read a snapshot
rewritten by a later version in any case.

**`num_attention_heads` (AP9-SV2-P, P4, ADR 0014).** Read independently of `kind` -- kept even
for `"unknown"` the same way `max_context` already is, never required for `dense_classic` or
`hybrid_linear`: a tensor-parallel package may need the head count when nothing else about the
shape could be read. `architecture_from_hf_config` reads it from the same place the three
required numeric fields come from (the top level, or `text_config` when a multimodal config
nests the language-model fields there, Qwen3.5's own shape). SV2-F's tensor-parallel rule treats
a missing head count as `t = 1`, with the reason said on the row.

**Upper bound `2**31 - 1` (R7-10, fix-round 6).** `> 0` alone never bounded the top end: a
publisher `config.json` value like `10**400` (which `json.loads` parses without complaint)
validated fine and then overflowed `fit.py::compute_fit`'s arithmetic (`OverflowError: integer
division result too large for a float`), aborting the whole render instead of ending that one
package's fit `"unknown"`. `compute_fit` also catches `OverflowError` directly, as a second line
of defense against a combination of otherwise in-bound values (or an unrelated, unbounded field
such as `Package.default_context`) that still overflows the float conversion together; that path
returns `Fit(fit_class="unknown", reason="architecture values out of range")`.

```json
{
  "source_repo": "acme/Nova-7B",
  "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
  "kind": "dense_classic",
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "layer_types": null,
  "max_context": 131072
}
```

### BaseModelSpec

One publisher base model that packagers build GGUF/tensor packages from.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `hf_repo` | `str` | `owner/name`, each half starting with a letter or digit (`contracts.validate_hf_repo`, the one rule for every `owner/name` in this file) | the publisher's exact Hugging Face repo, the base model's identity |
| `repo_aliases` | `list[str]` | repo *names*, non-empty, never `owner/name` | further exact packager repo names that count as this base model |
| `ollama_base` | `str \| None` | both set or both `None` with `ollama_tag` | the Ollama library model name, when one exists |
| `ollama_tag` | `str \| None` | both set or both `None` with `ollama_base` | the Ollama library tag naming this base model's size |
| `publisher` | `str` | -- | the organization that trained the model |
| `parameters_b` | `float \| None` | `None`, or `> 0` | parameter count, in billions; `None` means not measured this run and no previous reading exists (fix-round 1, F11) |
| `architecture` | `Architecture` | -- | the transformer shape, from the publisher's `config.json` |

```json
{
  "hf_repo": "acme/Nova-7B",
  "repo_aliases": ["Nova-7B-Instruct-GGUF"],
  "ollama_base": "nova",
  "ollama_tag": "7b",
  "publisher": "acme",
  "parameters_b": 7.0,
  "architecture": {
    "source_repo": "acme/Nova-7B",
    "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
    "kind": "dense_classic",
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "layer_types": null,
    "max_context": 131072
  }
}
```

### Family

A search term and the base models it resolves to.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `name` | `str` | -- | the search term a user configures (e.g. `qwen3.5`) |
| `base_models` | `list[BaseModelSpec]` | at least one | the base models this family names |

```json
{
  "name": "nova",
  "base_models": [
    {
      "hf_repo": "acme/Nova-7B",
      "repo_aliases": ["Nova-7B-Instruct-GGUF"],
      "ollama_base": "nova",
      "ollama_tag": "7b",
      "publisher": "acme",
      "parameters_b": 7.0,
      "architecture": {
        "source_repo": "acme/Nova-7B",
        "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
        "kind": "dense_classic",
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "layer_types": null,
        "max_context": 131072
      }
    }
  ]
}
```

### PackageFile

One file inside a package: a weight (or weight shard), an mmproj projector, or something else.
mmproj files are extras attached to a package; they are never counted as a model on their own
and never take part in quantization parsing or shard-completeness.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `name` | `str` | -- | the file name as the registry spells it |
| `role` | `"weights" \| "weights_shard" \| "mmproj" \| "other"` | -- | what the file is |
| `size_bytes` | `int` | `>= 0` | file size |
| `digest` | `str \| None` | -- | a content digest, when the registry provides one |

```json
{
  "name": "Nova-7B-Q4_K_M.gguf",
  "role": "weights",
  "size_bytes": 4500000000,
  "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd"
}
```

### Approval

A human decision that vouches for a specific package content, despite what its metadata
says. An approval is bound to the exact content it was given for; once that content moves on
(a new commit, a new manifest digest) the approval is void and `decide_provenance` falls back
to the metadata rules, never a blanket `from_metadata`.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `date` | `date` | -- | when the approval was given |
| `content` | `str` | 40-hex sha, or `sha256:` + 64 hex | the exact content approved: an HF revision sha, or an Ollama `manifest_digest` |
| `by` | `str` | -- | who approved it (a role or team, never a personal name) |

```json
{
  "date": "2026-09-01",
  "content": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
  "by": "acme-ai-team"
}
```

### Package

One packaged build of a base model, as observed on Hugging Face or in the Ollama registry.
Exactly the fields of its `source` are set. Only `format == "gguf"` takes part in fit,
provenance and rendering; a `tensor`-format package stays in the snapshot with
`provenance == "unresolved"` and `unresolved_reason == "format"` -- enforced as a model
invariant, not just a convention `decide_provenance` happens to follow. **The one exception**
(AP9-SV2-0, E8, ADR 0014): a `tensor`-format package whose `source` is `"huggingface"` may
instead be `"from_metadata"` or `"approved"`, for SV2-P's tensor packages (safetensors, read by
vLLM). A `tensor`-format package of `source` `"ollama"` (an MLX tag, noted 2026-09-28) stays
locked to `unresolved`/`format` -- it is never a vLLM package.

`quantization_source` says where `quantization` was read from: `"filename"` for a GGUF package
(the packager's own naming convention), `"config"` for a Hugging Face tensor package (its
`config.json`), `None` for an Ollama package (a tag, never a config) or wherever `quantization`
itself is still unknown. Additive, default `None` (AP9-SV2-0).

`quantization.QUANT_ORDER` gained six labels for vLLM's tensor formats (AP9-SV2-0, E8), read
from a tensor repo's `config.json` by SV2-P's reader, never from a filename: `W4A16` (behind
`Q4_K_M`, before `UD-Q4_K_XL`), `AWQ` (behind `W4A16`), `GPTQ-4` (behind `AWQ`, before
`UD-Q4_K_XL`), `GPTQ-8` (behind `Q8_0`), `W8A8` (behind `GPTQ-8`), `FP8` (behind `W8A8`, before
`UD-Q8_K_XL`) -- placed by bits per weight, the same yardstick the existing GGUF labels already
sort by. `BF16`/`F16` need no new label: a tensor repo with no `quantization_config` reads its
`torch_dtype` (top level) or `dtype`, else `text_config.dtype` (Qwen3.5's own shape), in that
search order.

**Provenance for a Hugging Face tensor package (AP9-SV2-P, P3, `modelroom/provenance.py`).** Once
the format lock's one exception (above) lets a tensor package reach a verdict at all, the rule is
close to the GGUF one but not the same. `decide_provenance` itself has no notion of "which
machine" -- the rule below is the same one wherever it runs; `hf.py`'s `build_tensor`
("Request budget") is what decides whether a tensor group is even offered to it at all, never
this function's own business:

1. The repository must still be one of the base model's package targets, or `("unresolved",
   "repo_name")` -- unchanged.
2. **The base model's own repository** (`package.repo == base_model.hf_repo`, the target
   AP9-UX2 P3 always adds) skips **the relation check alone**: that repository *is* the base
   model, so its own tags necessarily name its own predecessor (`Qwen/Qwen3-8B` carries
   `base_model:finetune:Qwen/Qwen3-8B-Base`), which `relation.check_relation` would otherwise
   read as `derivative`, never `quantized`. Rules 1 and 4 still apply to it.
3. Every other tensor repository still needs `relation.check_relation`'s `quantized` verdict,
   exactly like a GGUF package (the same `base_model_tag`/`relation_unknown`/`derivative`/
   `metadata_conflict` reasons).
4. **No file-stem check.** A packager's naming convention does not apply to a Transformers
   repository: its weight file names are fixed by the library, never by a packager, so each
   **whole** weight file name must be one of exactly three forms -- `model.safetensors`,
   `model-NNNNN-of-MMMMM.safetensors` or `model.safetensors-NNNNN-of-MMMMM.safetensors`, with
   `1 <= NNNNN <= MMMMM` -- else `("unresolved", "file_stem")`, the same reason the GGUF rule uses
   for an unrelated file dropped in the same repo. A name that only normalizes to the stem
   `model` (`model.safetensors.safetensors`, `model-split-00001-of-00001.safetensors`) does not pass.
5. Approvals are checked first, same as GGUF (see "Hugging Face" rule above): an approval whose
   `content` matches the package's current `revision` resolves `("approved", None)` before any
   of the rules above run.

`complete` is never accepted as a bare caller-supplied fact: a model validator derives it from
`files` with `shards_complete` and rejects a `Package` whose `complete` contradicts that
derived value. It stays a field in the serialized form -- readers of a snapshot see it without
recomputing it -- but its value is always the one `shards_complete(files)` produces.

**One package per manifest, not per tag** (decided 2026-09-25). The Ollama registry gives the very
same manifest, byte for byte and therefore under the same `manifest_digest`, to several tags:
`qwen3.5:9b` and `qwen3.5:9b-q4_K_M` are two names of one build, and the ranking showed that build
twice (test round, 2026-09-25). `ollama.fetch_ollama_area` therefore builds **one** package per
manifest digest, named by the **shortest** of its tags (ties alphabetically, so the result never
depends on the order the tags page lists them in), and carries the rest in `aliases`. The
quantization is still inherited across every kept tag, so a name that carries none takes it from a
digest sibling whose name does, even when that sibling is now an alias. `aliases` is optional and
empty by default, so a snapshot written before it reads unchanged and `schema_version` stays at 1.
An **approval carries forward from an absorbed tag as well**: the previous snapshot is asked under
this package's own identity and under every alias it now holds, because an approval given to
`qwen3.5:9b-q4_K_M` is about the very same manifest under the very same digest ("Approval
carry-forward across fetch runs"; the binding to the base model is unchanged).

Three more invariants tie `provenance`, `unresolved_reason`, `format` and `approval` together:
a non-`"gguf"` `format` requires `provenance == "unresolved"` and `unresolved_reason ==
"format"`, **unless the package is a Hugging Face `tensor` package** (the one exception above,
AP9-SV2-0); `provenance == "unresolved"` requires a non-empty `unresolved_reason` (never `None`,
the reverse of the "not unresolved -> reason must be `None`" rule below it already stated); and
`provenance == "approved"` requires an `approval` whose `content` equals the package's
*current* `revision` (Hugging Face) or `manifest_digest` (Ollama) -- an approval bound to
different content does not validate, it has to be re-approved or left to the metadata rules.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `source` | `"huggingface" \| "ollama"` | -- | which registry this package came from |
| `repo` | `str \| None` | `owner/name`; required iff `source == "huggingface"` | the packager repo |
| `revision` | `str \| None` | 40-hex commit sha; required iff `source == "huggingface"` | the commit this package was observed at |
| `ollama_name` | `str \| None` | `^[a-z0-9][a-z0-9._-]*:[A-Za-z0-9][A-Za-z0-9._-]*$`; required iff `source == "ollama"` | the Ollama image name |
| `manifest_digest` | `str \| None` | `sha256:` + 64 hex; required iff `source == "ollama"` | the manifest digest observed |
| `aliases` | `list[str]` | each a tag (`^[A-Za-z0-9][A-Za-z0-9._-]*$`), each once, never this package's own tag; empty for `source == "huggingface"` | the other tags of the very same manifest |
| `base_model_hf_repo` | `str` | must exist in the snapshot's `base_models` | which `BaseModelSpec` this package belongs to |
| `format` | `"gguf" \| "tensor" \| "unknown"` | -- | the weights format |
| `files` | `list[PackageFile]` | -- | the files that make up this package |
| `complete` | `bool` | derived from `files` by `shards_complete`, see above | whether all weight shards are present |
| `quantization` | `str \| None` | a `QUANT_ORDER` member, or `None` | `None` means unknown and sorts last |
| `quantization_source` | `"filename" \| "config" \| None` | -- | where `quantization` was read from: a GGUF filename, a tensor repo's `config.json`, or unknown/not applicable (AP9-SV2-0) |
| `default_context` | `int \| None` | `> 0` | Ollama's `num_ctx`; `None` for Hugging Face |
| `provenance` | `"from_metadata" \| "approved" \| "unresolved"` | -- | see `modelroom/provenance.py` |
| `unresolved_reason` | `str \| None` | non-empty iff `provenance == "unresolved"`, see above | why provenance could not be resolved |
| `approval` | `Approval \| None` | required when `provenance == "approved"`, content must match current revision/digest | the approval that made this package trusted |
| `observed_at` | `datetime` | -- | when this run first recorded this package |
| `last_seen` | `datetime` | -- | when this package was last confirmed to still exist |
| `active` | `bool` | -- | whether the package still exists upstream |

```json
{
  "source": "huggingface",
  "repo": "packager/Nova-7B-GGUF",
  "revision": "9f8e7d6c5b4a3928170615243342516078960a1b",
  "base_model_hf_repo": "acme/Nova-7B",
  "format": "gguf",
  "files": [
    {
      "name": "Nova-7B-Q4_K_M.gguf",
      "role": "weights",
      "size_bytes": 4500000000,
      "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd"
    }
  ],
  "complete": true,
  "quantization": "Q4_K_M",
  "quantization_source": "filename",
  "default_context": null,
  "provenance": "from_metadata",
  "unresolved_reason": null,
  "approval": null,
  "observed_at": "2026-09-22T09:00:00Z",
  "last_seen": "2026-09-22T09:00:00Z",
  "active": true
}
```

### Area

One fetch unit: a base model on one source, optionally scoped to one Hugging Face packager.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `source` | `"huggingface" \| "ollama"` | -- | which registry this fetch covered |
| `base_model_hf_repo` | `str` | -- | which base model this fetch covered |
| `packager` | `str \| None` | Hugging Face only | which packager's repo this fetch covered |
| `status` | `"complete" \| "incomplete"` | -- | whether the fetch finished |
| `last_success` | `datetime \| None` | -- | the last time this area completed successfully |
| `error` | `str \| None` | required when `status == "incomplete"` | why the fetch did not complete |

```json
{
  "source": "huggingface",
  "base_model_hf_repo": "acme/Nova-7B",
  "packager": "packager",
  "status": "complete",
  "last_success": "2026-09-22T09:00:00Z",
  "error": null
}
```

### Snapshot

One `fetch` run: the areas it covered, the base models it knows, the packages it found. Every
package's `base_model_hf_repo` must exist in `base_models`, and `(source, repo|ollama_name,
filename-or-tag)` must be unique across `packages`.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `schema_version` | `int` | must equal `SNAPSHOT_SCHEMA_VERSION` (currently `2`) | the snapshot's schema version; schema 1 written by 0.1.0 spells the provenance `metadata_ok`, the reader takes it as `from_metadata` and `fetch` writes it back (ADR 0007); `load_snapshot` reads a stored schema 1 as 2, unchanged (ADR 0014) |
| `run_at` | `datetime` | -- | when this `fetch` run started |
| `areas` | `list[Area]` | -- | the fetch units this run covered |
| `base_models` | `list[BaseModelSpec]` | -- | the base models this run knows |
| `packages` | `list[Package]` | see above | the packages this run found |

```json
{
  "schema_version": 2,
  "run_at": "2026-09-22T09:00:00Z",
  "areas": [
    {
      "source": "huggingface",
      "base_model_hf_repo": "acme/Nova-7B",
      "packager": "packager",
      "status": "complete",
      "last_success": "2026-09-22T09:00:00Z",
      "error": null
    }
  ],
  "base_models": [
    {
      "hf_repo": "acme/Nova-7B",
      "repo_aliases": ["Nova-7B-Instruct-GGUF"],
      "ollama_base": "nova",
      "ollama_tag": "7b",
      "publisher": "acme",
      "parameters_b": 7.0,
      "architecture": {
        "source_repo": "acme/Nova-7B",
        "source_revision": "1a2b3c4d5e6f7890abcdef1234567890abcdef12",
        "kind": "dense_classic",
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "layer_types": null,
        "max_context": 131072
      }
    }
  ],
  "packages": [
    {
      "source": "huggingface",
      "repo": "packager/Nova-7B-GGUF",
      "revision": "9f8e7d6c5b4a3928170615243342516078960a1b",
      "base_model_hf_repo": "acme/Nova-7B",
      "format": "gguf",
      "files": [
        {
          "name": "Nova-7B-Q4_K_M.gguf",
          "role": "weights",
          "size_bytes": 4500000000,
          "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd"
        }
      ],
      "complete": true,
      "quantization": "Q4_K_M",
      "quantization_source": "filename",
      "default_context": null,
      "provenance": "from_metadata",
      "unresolved_reason": null,
      "approval": null,
      "observed_at": "2026-09-22T09:00:00Z",
      "last_seen": "2026-09-22T09:00:00Z",
      "active": true
    }
  ]
}
```

### InstalledModel

One model the local Ollama daemon reports as installed, as `modelroom/ollama_local.py` read it
from `GET http://127.0.0.1:11434/api/tags`. That endpoint returns the digest as bare hex, with
no `sha256:` prefix; the fetcher normalizes it before building this model (CONTRACTS.md,
"Local Ollama inventory" below).

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `name` | `str` | -- | the model name as the Ollama daemon lists it |
| `digest` | `str` | `sha256:` + 64 hex, same validator as `Package.manifest_digest` | the layer digest the daemon reported |
| `size_bytes` | `int` | `>= 0` | the size the daemon reported |
| `observed_at` | `datetime` | aware UTC | when this `hardware` run queried the daemon |

```json
{
  "name": "nova:7b",
  "digest": "sha256:abababababababababababababababababababababababababababababababab",
  "size_bytes": 4500000000,
  "observed_at": "2026-09-22T09:00:00Z"
}
```

### Measurement

One recorded `bench` result, bound to the exact package content and hardware profile it
measured. AP4 only defines and validates this model -- there is no `bench` command yet; it is
a later work package. `content_source` discriminates which of the two field groups below is
populated, exactly the pattern `Package.source` already uses for `repo`/`revision` vs.
`ollama_name`/`manifest_digest`: a `content_source == "ollama"` measurement carries
`ollama_manifest_digest` and no `hf_*` field; a `content_source == "huggingface"` measurement
carries all three `hf_*` fields and no `ollama_manifest_digest`. The brief's other option -- one
polymorphic `content` field holding either a bare digest string or a `{repo, revision,
file_digest}` object -- was not chosen, so this model stays the same shape as every other
coupled-field model in this file.

`profile_measured_at` is meant to equal the `HardwareSnapshot.measured_at` it was benched
against, but that equality is **not** a validator on this model: `Measurement` validates in
isolation (it is never attached to a specific `HardwareSnapshot` instance at validation time,
only stored in its `measurements` list). The renderer (a later work package) reads both and
refuses to render a measurement whose `profile_measured_at` does not match the snapshot it is
being read alongside -- see "Fit contract v1" below for the exact wording it must use.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `content_source` | `"ollama" \| "huggingface"` | -- | which field group below is populated |
| `ollama_manifest_digest` | `str \| None` | `sha256:` + 64 hex; required iff `content_source == "ollama"` | the Ollama manifest digest measured |
| `hf_repo` | `str \| None` | `owner/name`; required iff `content_source == "huggingface"` | the packager repo measured |
| `hf_revision` | `str \| None` | 40-hex commit sha; required iff `content_source == "huggingface"` | the commit measured |
| `hf_file_digest` | `str \| None` | `sha256:` + 64 hex; required iff `content_source == "huggingface"` | the exact file measured |
| `context` | `int` | `> 0` | the context length the benchmark ran at |
| `runtime` | `str` | -- | the inference runtime and version, e.g. `ollama 0.12.3` |
| `profile_measured_at` | `datetime` | aware UTC | the hardware profile this measurement was benched against (see above) |
| `measured_at` | `datetime` | aware UTC | when the benchmark itself ran |
| `tps_mean` | `float` | `> 0` | mean tokens/second observed |
| `tps_range` | `tuple[float, float]` | low `<=` high | the observed tokens/second range |

```json
{
  "content_source": "ollama",
  "ollama_manifest_digest": "sha256:abababababababababababababababababababababababababababababababab",
  "hf_repo": null,
  "hf_revision": null,
  "hf_file_digest": null,
  "context": 8192,
  "runtime": "ollama 0.12.3",
  "profile_measured_at": "2026-09-22T09:00:00Z",
  "measured_at": "2026-09-22T09:05:00Z",
  "tps_mean": 42.5,
  "tps_range": [40.0, 45.0]
}
```

### HardwareSnapshot

One machine's measured hardware, written by `hardware` to `<state>/hardware/<machine>.json`
(see the persisted-forms table above). Reserved headroom (`reserve_ram_gib`/`reserve_vram_gib`)
is deliberately **not** a field here: it is operator policy chosen once per machine, not a
measured fact, and it can change without a new hardware measurement. It lives in the
configuration's `machines.<name>` (`config.MachineConfig`) instead; `modelroom/fit.py::compute_fit`
takes a `HardwareSnapshot` and a `MachineConfig` as two separate parameters rather than merging
them into one persisted shape.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `schema_version` | `int` | must equal `HARDWARE_SCHEMA_VERSION` (currently `1`) | the hardware profile's schema version |
| `machine` | `str` | `^[a-z0-9][a-z0-9-]*$`, same rule as a `Configuration.machines` key | which machine this profile describes |
| `measured_at` | `datetime` | aware UTC | when `hardware` took this measurement |
| `llmfit_version` | `str` | -- | the installed `llmfit` version that produced `system` below |
| `vram_gib` | `float` | `>= 0` | GPU VRAM, from `llmfit system --json`'s `system.gpu_vram_gb` unchanged (already GiB, see below) |
| `ram_gib` | `float` | `> 0` | total system RAM, from `system.total_ram_gb` unchanged |
| `free_ram_gib_at_measurement` | `float \| None` | `>= 0` | free RAM at measurement time, from `system.available_ram_gb` |
| `gpu_name` | `str \| None` | -- | from `system.gpu_name` |
| `backend` | `str \| None` | -- | from `system.backend` |
| `unified_memory` | `bool` | -- | from `system.unified_memory` |
| `installed` | `list[InstalledModel] \| None` | exactly one of `installed`/`installed_unavailable_reason` is set, see below | the local Ollama daemon's inventory, or `None` if it could not be read |
| `installed_unavailable_reason` | `str \| None` | non-empty iff `installed is None` | why the Ollama daemon could not be reached, when `installed` is `None` |
| `measurements` | `list[Measurement]` | -- | bench results carried over from the previous hardware profile for this machine (AP4 never adds to this list, only preserves it) |

```json
{
  "schema_version": 1,
  "machine": "workstation",
  "measured_at": "2026-09-22T09:00:00Z",
  "llmfit_version": "1.1.16",
  "vram_gib": 11.94,
  "ram_gib": 127.46,
  "free_ram_gib_at_measurement": 76.64,
  "gpu_name": "Nova GPU",
  "backend": "CUDA",
  "unified_memory": false,
  "installed": [
    {
      "name": "nova:7b",
      "digest": "sha256:abababababababababababababababababababababababababababababababab",
      "size_bytes": 4500000000,
      "observed_at": "2026-09-22T09:00:00Z"
    }
  ],
  "installed_unavailable_reason": null,
  "measurements": []
}
```

`modelroom.contracts.load_hardware_snapshot(data)` is the entry point for reading a persisted
hardware profile, checking `schema_version` against `HARDWARE_SCHEMA_RANGE` before field
validation, exactly like `load_snapshot`.

### Fit

The result of judging whether one GGUF package fits one measured machine, computed by
`modelroom/fit.py::compute_fit`. **Not persisted by AP4** -- a later renderer calls it fresh
every time against the current snapshot and hardware profile, rather than storing a stale
verdict. See "Fit contract v1" below for the formula, thresholds and the exact wording a
renderer must use.

The field names `MachineRanking.no_fit` and `Requirement.ample_reachable` and the note codes
`no_fit` and `cpu_caps_at_fits` follow the words of the fit classes (decided 2026-09-27, ADR 0007).

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `fit_class` | `"ample" \| "fits" \| "tight" \| "no_fit" \| "unknown"` | -- | what was computed: the share of the pool the package needs (decided 2026-09-27, ADR 0006) |
| `mode` | `"gpu" \| "gpu_split" \| "cpu_gpu" \| "cpu" \| None` | `None` only when `fit_class == "unknown"`; `gpu_split` since document schema 3 (2026-09-26) | where the package would run: one graphics card, spread over every card, system memory with a card helping, system memory only |
| `need_gib` | `float` | `>= 0` | the memory needed, computed |
| `weights_gib` | `float` | `>= 0` | the package's weight files, summed |
| `kv_gib` | `float` | `>= 0` | the KV-cache size at `context`, computed |
| `pool_gib` | `float` | -- | the memory pool judged against (can be negative if reserves exceed capacity) |
| `reserve_gib` | `float` | `>= 0` | the reserve actually applied (VRAM or RAM, matching `mode`) |
| `context` | `int` | `>= 0` | the context length used for `kv_gib` |
| `context_assumed` | `bool` | -- | `true` when `context` is the 8192 fallback, not the package's own `default_context` |
| `basis` | `"architecture" \| "size"` | default `"architecture"` | what the KV cache was computed from: the base model's architecture, or the size alone ("Fit from size") |
| `reason` | `str \| None` | non-`None` typically iff `fit_class == "unknown"` | why the fit could not be computed, when it could not |
| `requests` | `int` | `1..1024`, default `1` | the parallel requests the KV cache was computed for, once per request (decided 2026-09-25; see "Fit contract v1", "Parallel requests") |
| `runtime` | `str` | one value of `RuntimeKey` (`"ollama"`, `"vllm"`), default `"ollama"`, not written at the default | the entry of `runtimes.RUNTIMES` that judged this row; "Fit for vLLM" below |
| `at_once` | `int` | `>= 0`, default `0`, not written at `0` | how many requests of `context` the runtime holds at once on the cards of this row (vLLM only). `0` has two meanings, told apart by `fit_class`: "not computed" (every Ollama fit, and a vLLM `unknown`) and the computed capacity of a vLLM `no_fit`, where the budget after weights and runtime is below one request or negative (`at_once = max(0, ...)`, never below 0) |
| `cards` | `int` | `>= 1`, default `1`, not written at `1` | how many cards the weights are spread over (vLLM's tensor-parallel size `t`; Ollama rows stay at one) |
| `larger_size` | `bool` | default `false`, not written at `false` | vLLM only: an allowed tensor-parallel size above `cards` exists for this model on this machine (`fit_vllm.allowed_sizes`: within the equal cards, dividing the head count); `false` for Ollama, for an unknown head count and where `cards` is already the largest allowed. The install note "larger --tensor-parallel-size ..." follows it (AP9-SV2-D, Codex d1-3) |

```json
{
  "fit_class": "fits",
  "mode": "gpu",
  "need_gib": 7.125,
  "weights_gib": 5.0,
  "kv_gib": 1.125,
  "pool_gib": 10.94,
  "reserve_gib": 1.0,
  "context": 8192,
  "context_assumed": true,
  "basis": "architecture",
  "reason": null,
  "requests": 1
}
```

### Fit for vLLM

`fit_vllm.vllm_fit_as_fit(profile, package, base_model, scenario, machine_config)` is the `fit` of the
`vllm` entry of `runtimes.RUNTIMES` (AP9-SV2-F): it judges one `tensor` package on one machine and returns the
`Fit` above with `runtime == "vllm"`. `render` calls it; `guided_vllm._fit` and the table read that very `Fit`
(one calculation, one truth). It is computed, never measured (ADR 0008), and nothing here is persisted.

**Guards, in this order, each giving `unknown` with a fixed `reason` and a `Fit` that still says
`runtime == "vllm"`** (the words of `fit.py`, so a reader cannot tell a vLLM guard from a GGUF one): a profile the
fit may not compute for (`profile.fit_block_reason`); a package that is not `complete` (`only a complete package
is judged`); a weight file of size 0 (`a weight file has no size`); a known context window smaller than the
run's (`context beyond the window of this model (<n>)`); unified memory (`vLLM does not run on shared memory`);
no graphics card (`vLLM needs a graphics card`). The last two put the package in `not_covered`, never `no_fit`.

**The cards.** The group is the cards within 1 % of the largest (`EQUAL_CARD_TOLERANCE`), in the order of their
`index`; smaller cards stay out and the table head says so. The sizes `t` may take are those of `1, 2, 4, 8` that
are at most the group's size and divide `Architecture.num_attention_heads` (`fit_vllm.allowed_sizes`; the
limit to `1, 2, 4, 8` is modelroom's own, not vLLM's).

**The rule for `t` (b)** (GF 2026-10-03, replacing "the largest size", F1 a of the build brief): `t` is the
**smallest** allowed size at which the package fits. *Fits* is `fit_vllm.vllm_fit_class` over the first `t`
cards of the group (by `index`) giving `fits` or `ample`, so neither `no_fit` nor `tight`; `unknown` does not
occur here, the guards ran before. Where it fits at no allowed size, `t` is the largest allowed one, and the row
shows that best try. Where the head count is unknown, `t = 1` and `reason` is `tensor parallel needs the model's
head count; computed on one card`. The reason is the GF's: a smaller `t` answers faster, and the other cards stay
free. `fit_vllm.largest_parallel_size` stays as it was: the pool in the table head is what the machine offers at
most (the largest size at most the group), whatever the model; three equal cards give two.

```
GIB       = 2**30 B                                          one GiB in bytes
card      = smallest of the t chosen cards                   GiB
budget    = 0.92 x card                                      GiB, per card
weights   = sum of the safetensors in GiB / t                GiB, per card, no 1.10 overhead
runtime   = 1.50                                             GiB, per card
kv        = budget - weights - runtime                       GiB, per card; may be negative
kv_bytes  = floor(max(kv, 0) x GIB)                          B, per card; a negative budget is 0 B
per_token = kv_token_bytes(architecture) / min(t, num_key_value_heads)   B, per card; size basis: 147,456 B, not divided
state     = state_bytes(architecture) / t                    B, per card, per request; hybrid models, else 0
at_once   = kv_bytes // (per_token x context + state)        whole requests, >= 0: max(0, ...) is the max(kv, 0) above
fit_class = vllm_fit_class(at_once, requests, basis)
            no_fit below 1, tight below the requests, ample at twice them or more (architecture basis only), else fits
```

| `Fit` field | Where it comes from, for `runtime == "vllm"` |
|---|---|
| `mode` | `gpu` where `t = 1`, else `gpu_split`; `None` where `fit_class == "unknown"` |
| `cards` | `t` (above) |
| `at_once` | the formula above, over one card's own share; `0` for an `unknown` fit |
| `basis` | `architecture` where the base model's architecture gives the KV bytes per token, else `size` (the class then never reaches `ample`) |
| `reason` | `None`, except the head-count reason (`t = 1`, the fit still computed) or the guard that made the fit `unknown` |
| `weights_gib` | the whole package: the sum of its safetensors files in GiB |
| `kv_gib` | the KV cache of `requests` requests of `context` tokens over **all** `t` cards, in GiB: `(per_token x context + state) x requests x t / GIB` (`per_token` and `state` are bytes, above), what the ranking assumes |
| `need_gib` | `weights_gib + t x 1.50 + kv_gib` |
| `pool_gib` | `t x 0.92 x card`: what vLLM may take of the chosen cards |
| `reserve_gib` | `t x (card - 0.92 x card)`: what vLLM leaves free of them |
| `context_assumed` | always `false`: the context is the run's own, never the 8192 fallback |

`fit_class`, `context` and `requests` are the run's, as for any `Fit`. **Constants:** `0.92`
(`VLLM_GPU_MEMORY_UTILIZATION`) is vLLM 0.30.0's default of `--gpu-memory-utilization`
(docs.vllm.ai/en/v0.30.0/configuration/engine_args). `1.50` GiB (`VLLM_RUNTIME_OVERHEAD_GIB`) is the runtime of the
recording on one A100 at utilization 0.5 (`0.5 x 80 - 0.920 - 37.58`, decided 2026-09-27); the recording on one L4
gave `1.31`. **Assumptions, until a rented server confirms them:** A1, the 1.50 GiB applies to every card under
tensor parallel (documented for one card); A2, with fewer KV heads than cards vLLM replicates the heads, so a card's
`per_token` is `kv_token_bytes(architecture) / min(t, num_key_value_heads)` (the formula above, divided by the heads
it can split, never by `t` alone), and a hybrid model's state divides by `t`. A test of this formula
proves no behavior of vLLM. Worked example (test): four L4 of 22.49 GiB, `Qwen/Qwen3-8B-AWQ` (5.68 GiB, 36 layers,
8 KV heads, `head_dim` 128), context 32,768, 10 requests: `t = 1` holds 3 and `t = 2` holds 7 (both `tight`),
`t = 4` holds 15 (`fits`), so `t = 4`; with 7 requests `t = 2` is the smallest that fits.

### Rating

A market-index star rating for one base model, as a `render.RatingSource` (see "Render (AP5)"
below) returns it to `modelroom render`. Not persisted anywhere -- a `RatingSource` is a live
callable a caller passes to `cli.render_with_config`, never a file this module reads or writes.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `stars` | `float` | `0.5 <= stars <= 5.0`, a multiple of `0.5` | the star rating |
| `source` | `str` | non-empty | a short label naming where the rating came from, e.g. `"market index"` |

```json
{
  "stars": 3.5,
  "source": "market index"
}
```

## Configuration

The models below make up `Configuration`, the parsed form of `modelroom.toml`. They live in
`modelroom/config.py`, not `modelroom/contracts.py` -- `Configuration` is what the user brings
in, `Snapshot` is what `fetch` produces, and the two are never merged into one module. The
three field validators every field below relies on (`hf_repo` shape, `repo_aliases` shape, the
`ollama_base`/`ollama_tag` pair) are the same functions `BaseModelSpec` uses
(`validate_hf_repo`, `validate_repo_aliases`, `validate_ollama_pair` in `modelroom/contracts.py`),
so a base model's identity rules read the same in the config and in the snapshot.

Every example below is `EXAMPLES["<ModelName>"]` from `modelroom/config.py`, verbatim, under a
`#### <ModelName>` heading -- one level deeper than the `### <ModelName>` headings above, so
the doc-sync test for the snapshot's models (which scans the whole file for `### ` headings)
does not also pick these up. `tests/test_config.py` checks both directions the same way
`tests/test_contracts.py` does for the models above: every model in `modelroom/config.py` is
documented here, and every `#### ` heading here names a real model with a matching example.

#### BaseModelConfig

One base model a family resolves to, as the user configures it. Deliberately narrower than
`BaseModelSpec`: `publisher`, `parameters_b` and `architecture` are read from Hugging Face at
fetch time and live only in the snapshot, never here.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `hf_repo` | `str` | `owner/name`, same rule as `BaseModelSpec.hf_repo` | the publisher's exact Hugging Face repo |
| `repo_aliases` | `list[str]` | same rule as `BaseModelSpec.repo_aliases` | further exact packager repo names that count as this base model |
| `repos` | `list[str]` | schema 2; each `owner/name`, unique; default `[]` | owner-bound packager repos (e.g. found by the guided search under an owner that is not in `packagers`); each entry names exactly one repo |
| `ollama_base` | `str \| None` | both set or both `None` with `ollama_tag` | the Ollama library model name, when one exists |
| `ollama_tag` | `str \| None` | both set or both `None` with `ollama_base` | the Ollama library tag naming this base model's size |

**Package targets (schema 2).** `package_targets(config, base_model)` returns every packager repo
that belongs to a base model, each once, in a stable order: (`packagers` plus the base model's
own owner) x (`<name>-GGUF` plus `repo_aliases`), then `repos`. The collision check below, the
fetch, the owner grouping and provenance all use exactly this set, so none of them can disagree
about which repos a base model claims (see "Target set and areas"). For the example below the
targets are `packager/Nova-7B-GGUF`, `packager/Nova-7B-Instruct-GGUF`, `acme/Nova-7B-GGUF`,
`acme/Nova-7B-Instruct-GGUF`, `community/Nova-7B-GGUF`.

```json
{
  "hf_repo": "acme/Nova-7B",
  "repo_aliases": ["Nova-7B-Instruct-GGUF"],
  "repos": ["community/Nova-7B-GGUF"],
  "ollama_base": "nova",
  "ollama_tag": "7b"
}
```

#### FamilyConfig

A family name and the base models it resolves to, as the user configures it.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `name` | `str` | non-empty, `^[a-z0-9][a-z0-9.-]*$` | the family's configured name (e.g. `qwen3.5`) |
| `base_models` | `list[BaseModelConfig]` | at least one | the base models this family names |

```json
{
  "name": "nova",
  "base_models": [
    {
      "hf_repo": "acme/Nova-7B",
      "repo_aliases": ["Nova-7B-Instruct-GGUF"],
      "repos": ["community/Nova-7B-GGUF"],
      "ollama_base": "nova",
      "ollama_tag": "7b"
    }
  ]
}
```

#### MachineConfig

One machine's reserved headroom and whether it runs `fetch` and writes the shared state. The
machine's name is not a field here: it is the key under `Configuration.machines`, validated
there (`^[a-z0-9][a-z0-9-]*$`) against every key at once.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `reserve_ram_gib` | `float \| None` | `>= 0`; default `None` (key left out) | system RAM to leave unused when judging fit; left out, it is computed from the machine at every run (`reserve.py`, ADR 0012); a number always wins |
| `reserve_vram_gib` | `float` | `>= 0` | GPU VRAM to leave unused on each graphics card when judging fit |
| `writer` | `bool` | -- | whether this machine runs `fetch` and writes the shared state |
| `profile` | `str \| None` | schema 2; 16 lowercase hex characters; default `None` | the `profile_id` of this machine's hardware profile (`<state>/hardware/<profile_id>.json`), set by `modelroom migrate` or a guided run |

**The two reserves add up on unified memory** (decided 2026-09-25). On a machine of `gpu_state`
`unified_memory` the operating system and the model share one memory, so the fit takes both
reserves from it: pool = physical RAM - `reserve_ram_gib` - `reserve_vram_gib` (see "Fit contract
v1", "Unified memory"). On every other machine each reserve belongs to its own memory, as before.

**The memory reserve is computed where none is entered** (AP9-RM, decided 2026-09-29, ADR 0012):
a quarter of the physical RAM, at least 2 and at most 16 GiB; on unified memory a quarter with no
upper bound (128 GiB: 32 + the card reserve 1, pool 95). A number in `modelroom.toml` wins, `0.0`
included, and is never overwritten; a machine that still carries the `8.0` of 0.1.0 gets a hint
under its table when the computed reserve differs, nothing more. Without a number and without a
known RAM the reserve is `0.0` with origin `unknown`. A new folder writes no `reserve_ram_gib`;
a new entry takes a number from `[defaults]` as before and leaves the key out where `[defaults]` has
none; an existing entry keeps its own through every write.

```json
{
  "reserve_ram_gib": 8.0,
  "reserve_vram_gib": 1.0,
  "writer": true,
  "profile": "3f9a0c21d4e6b870"
}
```

#### DefaultsConfig

Schema 2. Reserves for a machine that joins without its own `[machines.<name>]` table (an
imported profile); such a machine is never a writer. Since AP9-RM `reserve_ram_gib` has no
default number: left out, the new entry leaves it out too and computes it (ADR 0012).

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `reserve_ram_gib` | `float \| None` | `>= 0`, default `None` | system RAM to leave unused; `None`: computed from the machine |
| `reserve_vram_gib` | `float` | `>= 0`, default `1.0` | GPU VRAM to leave unused on each graphics card |

```json
{
  "reserve_ram_gib": 8.0,
  "reserve_vram_gib": 1.0
}
```

#### UpdatesConfig

Schema 2. Whether the guided mode may look up a newer modelroom release; `check = false` turns
the lookup off.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `check` | `bool` | default `true` | look up a newer release at the start of a guided run |

```json
{
  "check": true
}
```

#### GuidedConfig

Schema 2. What the guided mode recorded when it wrote this configuration: the results folder, and
the context its ranking was computed for. `load_config` resolves a relative `results` against the
configuration file's own directory, exactly like `paths.state`; it is not confined.
`Configuration.from_dict` requires it absolute, like `paths`. `results = None` (the table absent)
means no guided run wrote this file.

`context` is the one context of the whole ranking, kept so that a later `modelroom render` of this
folder shows the same ranking as the guided run that chose it (see "Render", "The scenario"). It
takes exactly the values `Scenario.context_requested` takes (`> 0`, `<= 2**31 - 1`); `None` means
no guided run has chosen a context yet, and a configuration written before this field existed
reads and renders unchanged. The size scale of the guided mode starts on this value -- on its level,
or on a `custom` line of its own when it is no level -- and writes the answer back whenever it
differs from what the file holds by then (see "Guided mode", step 3) -- the first time as well, and
for 8192 as well: a kept context is what the user chose, not the value the question started with.
The field always holds the **number of tokens**, never a level name. Both fields are optional; they
came with schema 2 and no migration step reads or writes them.

**Parallel requests (schema 3, decided 2026-09-25).** `requests` are the parallel slots the
ranking assumes (the daemon's `OLLAMA_NUM_PARALLEL`, see "Fit contract v1", "Parallel requests"),
never "users at once". `requests_origin` says where the number came from, `users` is the head
count a user named:

- requests named directly: `entered`; a head count may stay for the display;
- only a head count: `from_users`, `requests = min(1024, max(1, ceil(users x ACTIVE_SHARE)))`
  with `ACTIVE_SHARE = 0.10` -- rule of thumb, not measured (1 -> 1, 10 -> 1, 11 -> 2,
  25 -> 3, 10240 -> 1024);
- neither: `default`, one request and no head count.

`requests` and `requests_origin` are written together or not at all; a table that breaks one of
the rules is invalid (`ConfigError`, exit `2`). A `[guided]` table without the keys reads as one
request of origin `default`. `render_cmd.scenario_from_config` reads `requests` whether a context
is kept or not. Step 3 of the guided mode asks how many people work at the same time and writes the three keys
together (see "Guided mode", step 3; decided 2026-09-26, question changed 2026-10-10).

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `results` | `Path \| None` | default `None` | the folder the guided mode keeps its configuration and results in |
| `context` | `int \| None` | `> 0`, `<= 2**31 - 1`; default `None` | the context the last guided run's ranking was computed for |
| `users` | `int \| None` | schema 3; `1..10240`; default `None` | the head count a user named (display and derivation) |
| `requests` | `int` | schema 3; `1..1024`; default `1` | the parallel requests the ranking assumes |
| `requests_origin` | `"default" \| "entered" \| "from_users"` | schema 3; default `"default"`; written together with `requests` | where `requests` came from |

```json
{
  "results": "//models/modelroom",
  "context": 4096,
  "users": 25,
  "requests": 3,
  "requests_origin": "from_users"
}
```

#### PathsConfig

Where modelroom keeps its state and its rendered Markdown output. Both fields must be
absolute once this model validates: `load_config` resolves a relative path against the config
file's own directory before validation; a `Configuration` built programmatically
(`Configuration.from_dict`) has to pass paths that are already absolute. This package never
resolves a path against the process' working directory. `snapshot_file`, `lock_file`,
`run_status_file` and `hardware_dir` are derived read-only properties, not fields. "Absolute"
means absolute for the platform the command runs on (`C:\...` on Windows, `/...` on POSIX);
the examples below use a `//host/share/...` form only because it is absolute on both.

**`paths.state` confinement (fix-round 1, F7).** `load_config` additionally requires
`paths.state` to resolve (symlinks followed, `Path.resolve()`) to somewhere inside the config
file's own directory tree; a `state` that escapes it (`../outside`, or a symlink pointing
outside) is a `ConfigError` naming both paths, raised before anything is written. This
confinement is enforced only by `load_config`, the file reader -- a programmatic caller
(`Configuration.from_dict`) is trusted to already have confined its own paths and is not
checked. **`paths.markdown` is deliberately not confined** -- it may live elsewhere by design
(e.g. a shared docs tree outside the state directory). The config file itself must not be one of
the state files (`modelroom.json`, `modelroom.lock`, `run-status.json` in `paths.state`): the lock
alone would overwrite and then empty it. `load_config` raises `ConfigError` before anything is
written.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `state` | `Path` | absolute | the state folder: snapshot, lock, run-status, `hardware/<machine>.json` |
| `markdown` | `Path` | absolute | the rendered Markdown output file |

```json
{
  "state": "//models/modelroom/state",
  "markdown": "//models/modelroom/docs/models.md"
}
```

#### LlmfitConfig

The minimum `llmfit` version this configuration requires. Only the requirement is stored
here; checking the installed tool against it is a later work package.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `min_version` | `str` | default `"1.1.16"`, `^\d+\.\d+\.\d+$` | the minimum accepted `llmfit` version |

```json
{
  "min_version": "1.1.16"
}
```

#### Configuration

The whole `modelroom.toml`: families, allow-listed owners, machines, paths, tool gate. Family
names are unique, `hf_repo` is unique across every family's base models, and the owner part of
every `hf_repo` must appear in `publishers` (the error names the missing owner). `packagers`
and `publishers` are each non-empty owner names, unique within their own list. `allowed_owners()`
returns `frozenset(packagers) | frozenset(publishers)` -- the positive list the *generated*
package targets stay inside. An owner-bound `repos` entry may name any account: it is the user
writing down one exact repository, and only that repository is ever fetched under that account
(see "Target set and areas"). No two base models may claim the same package target (see
`BaseModelConfig`, "Package targets"); the error names the repo and both base models.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `schema_version` | `int` | the model accepts only `CONFIG_SCHEMA_VERSION` (`3`); readers also accept `1` and `2`, see below | the configuration's schema version |
| `families` | `list[FamilyConfig]` | may be empty (schema 2), names unique, `hf_repo` unique across all | the families this configuration knows |
| `packagers` | `list[str]` | non-empty, no `/`, unique | Hugging Face owners that publish packager (GGUF) repos |
| `publishers` | `list[str]` | non-empty, no `/`, unique | Hugging Face owners that publish base models |
| `machines` | `dict[str, MachineConfig]` | keys `^[a-z0-9][a-z0-9-]*$`; may be empty | this deployment's machines, keyed by name |
| `paths` | `PathsConfig` | -- | where state and rendered output live |
| `llmfit` | `LlmfitConfig` | default `LlmfitConfig()` | the minimum required `llmfit` version |
| `defaults` | `DefaultsConfig` | schema 2, default `DefaultsConfig()` | reserves for an imported machine without its own table |
| `updates` | `UpdatesConfig` | schema 2, default `UpdatesConfig()` | the release lookup switch |
| `guided` | `GuidedConfig` | schema 2, default `GuidedConfig()` | the guided mode's results folder, the context it chose and, since schema 3, the parallel requests |

**Reading schema 1 and 2.** `CONFIG_SCHEMA_RANGE` is `(1, 4)`: `load_config` and
`Configuration.from_dict` accept schema 1, 2 and 3 and refuse 4 and above with
`SchemaVersionError` before any field is looked at. `normalize_config` takes a dict of an earlier schema to the
current schema in memory before validation, one step at a time:

- **1 -> 2** (`normalize_config_v1`): every field keeps its value, `[defaults]` and `[updates]`
  are added with their defaults, `[guided]` stays absent, no machine gets a `profile`. Schema 1's
  own rule "at least one family" is checked there (a `ConfigError`), because schema 2 drops it.
  No writer rule is added.
- **2 -> 3** (`normalize_config_v2`, decided 2026-09-25, ADR 0002): every value stays; a present
  `[guided]` table gets `requests = 1` and `requests_origin = "default"`, a missing one stays
  missing. A file that says schema 2 but carries `users`, `requests` or `requests_origin` is
  refused (`ConfigError`).

Reading never changes the file on disk. Three paths write a file of an earlier schema back as schema 3:
`modelroom migrate` (see "Migration"), its trigger at the start of a guided run, and
`import-profile` (see "Export and import").

```json
{
  "schema_version": 3,
  "families": [
    {
      "name": "nova",
      "base_models": [
        {
          "hf_repo": "acme/Nova-7B",
          "repo_aliases": ["Nova-7B-Instruct-GGUF"],
          "repos": ["community/Nova-7B-GGUF"],
          "ollama_base": "nova",
          "ollama_tag": "7b"
        }
      ]
    }
  ],
  "packagers": ["packager"],
  "publishers": ["acme"],
  "machines": {
    "workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": true, "profile": "3f9a0c21d4e6b870"}
  },
  "paths": {
    "state": "//models/modelroom/state",
    "markdown": "//models/modelroom/docs/models.md"
  },
  "llmfit": {"min_version": "1.1.16"},
  "defaults": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0},
  "updates": {"check": true},
  "guided": {
    "results": "//models/modelroom",
    "context": 4096,
    "users": 25,
    "requests": 3,
    "requests_origin": "from_users"
  }
}
```

### Path contract

`paths.state` and `paths.markdown` are always absolute in a validated `Configuration`. A raw
`modelroom.toml` may write them relative; `load_config` resolves a relative path against that
file's own parent directory, never against the current working directory. A caller that builds
a `Configuration` programmatically (`Configuration.from_dict`) has no config file to resolve
against, so it must already pass absolute paths -- a relative one is a validation error naming
the field and explaining why.

**A relative `--config`/`path` argument (fix-round 1, F8).** `load_config` resolves `path`
itself to an absolute path first (`path = path.resolve()`), before reading the file or
resolving `paths.state`/`paths.markdown` against its directory -- a relative `path.parent`
(e.g. `Path(".")` when the process' current directory holds the config file) would otherwise
stay relative forever and never make `paths.state`/`paths.markdown` absolute, failing
`PathsConfig`'s own "must be absolute" check for a reason that has nothing to do with the TOML
file's own content.

### Error mapping

| Error | Raised by | Meaning | Exit code |
|---|---|---|---|
| `SchemaVersionError` | `load_config`, `Configuration.from_dict` | `schema_version` is missing, not an integer, or outside `CONFIG_SCHEMA_RANGE` | `3` |
| `ConfigError` | `load_config`, `Configuration.from_dict` | the config file is missing, is not valid TOML, or fails field/cross-field validation; the message names the config file path (for `load_config`) and the underlying cause | `2` |

`SchemaVersionError` is checked, and raised, before `ConfigError` ever gets a chance to wrap a
field problem -- exactly like `load_snapshot`, so a wrong `schema_version` is never reported as
an ordinary validation error.

## Fetch runtime state (AP3)

`modelroom/state.py` owns three files under `config.paths.state` besides the snapshot itself.
None of the three is a contract in the sense above (no `schema_version`, no `EXAMPLES` entry,
no cross-tool compatibility promise) -- they exist only for the lifetime of one deployment's
own `fetch` runs, exactly as `AGENTS.md` already scoped them.

### Lock file (`modelroom.lock`)

**Fix-round 3: a kernel lock on a stable file, never a rename-based takeover.** The file is
created once, never renamed and never deleted, and `modelroom/state.py::acquire_lock` takes an
exclusive, non-blocking *kernel* lock on it (`msvcrt.locking` on Windows, `fcntl.flock`
elsewhere) rather than arbitrating exclusivity through this module's own bookkeeping. This
replaced fix-round 2's rename-based stale-lock takeover entirely: that design could not be made
exclusive against a third process, because `os.replace` is not bound to the specific file
generation a process actually read. Concretely, process A could see a stale lock, claim it,
create a fresh one and return, and process B could then rename *A's own fresh lock* away (it has
no way to tell A's brand-new lock from another stale one) and create its own -- neither
`os.replace` call ever fails, so both A and B end up believing they hold the lock. Giving a
foreign lock back on release had the same hole: it could hand the file back under a third
process's now-current lock instead of the one it actually took it from. A kernel lock has no
such window: the OS, not this module, decides who holds it. It also needs no "how old is too
old" rule any more -- a crashed holder's lock is released by the kernel the instant its process
exits, and a live holder keeps the lock no matter how long it has held it, so `LockHeldError` no
longer carries an age threshold in its message.

`acquire_lock(path, command, now) -> LockHandle` opens `path` with `os.open(path, O_RDWR |
O_CREAT)` -- never `O_EXCL`, since the file is meant to persist across runs and across crashes --
and locks exactly one byte at offset 0 (`_LOCK_OFFSET`). On success it truncates the file to one
byte past that (`_CONTENT_OFFSET`) and writes `{"pid": <int>, "command": <str>, "started_at":
<ISO 8601 UTC>}` from that offset onward (no `token` any more -- there is nothing left to
arbitrate). If writing that content fails after the lock was won, the fd is closed (one call,
which drops the kernel lock with it) before the error propagates, so no lock is ever held without
a handle to release it (fix-round 4, Codex P2). `LockHandle` is a small dataclass (`path`, `fd`, `released`); the fd stays open for
as long as the caller holds the lock, since the lock lives exactly as long as that fd does.
`release_lock(handle)` truncates the file back to empty, releases the kernel lock, and closes the
fd in a `finally`, so a failing truncate or unlock still closes it and the failure propagates.
Idempotent through the handle's own `released` flag, never through the descriptor: once the fd is
closed its number is reused by the next `os.open` in the process, so a second release must not
touch the descriptor at all (fix-round 4, Codex P1; tested by opening another file right after
the first release and releasing the stale handle again). **The file itself is never deleted**,
by either function; a fresh `acquire_lock` reuses it. On failure (`OSError`, including
`BlockingIOError`) `LockHeldError` is raised, naming the current holder's pid/command/started_at
when a read of the content succeeds and parses (it may still be empty or unparseable while the
holder is mid-write, in which case the message says only that another process holds it).

**Why the lock byte is not part of the content.** The lock is one reserved byte at offset 0,
never part of the JSON, which starts one byte later at `_CONTENT_OFFSET`. Windows locking is
*mandatory*, not advisory: a read that overlaps a locked byte raises `OSError`/`PermissionError`
for any handle but the one holding the lock, so if the JSON started at offset 0, nobody --
including this module's own holder-naming read -- could read it while the lock was held. Nothing
is written or truncated before the lock call: until the lock is held, this process is not the
only one touching the file. (`fcntl.flock` locks the whole file and is advisory, so the offset
only matters on Windows; the layout is the same on both so the file format is one thing.)

**Exclusivity is tested, not retried.**
`tests/test_state.py::test_acquire_lock_is_exclusive_across_three_real_processes` starts three
real processes that race `acquire_lock` against the same fresh file; the winner holds the lock
until the test releases it, and every round must show exactly one winner and two
`LockHeldError`s, with no retry. A first draft of that test let the winner exit right after
acquiring -- which releases the lock (the kernel drops a lock with its process) and lets a slower
sibling acquire legitimately. That looked like "two winners in one round in five" and was
misread during the build as a lock-placement problem; it was a test flaw, and the retry that
had been added to paper over it is gone.

**Ordering (fix-round 1, F13):** `fetch_with_config` checks the existing snapshot's
`schema_version` *before* calling `acquire_lock` at all -- an unsupported version exits `3`
with nothing written, not even a lock file, and never disturbs one already there. The snapshot
is read a second time once the lock is held (it may have changed between the two reads), and
that second check is what a concurrent writer's own stale-schema snapshot would be caught by.

**`hardware` takes no lock.** Every other command that writes shared state under
`config.paths.state` (today, only `fetch`) goes through `modelroom.lock` first, because they
all write to the *same* file (`modelroom.json`) that a second concurrent run could corrupt.
`hardware` writes only `<state>/hardware/<machine>.json`, a file scoped to the one machine it
ran on -- two different machines running `hardware` at the same time write two different
files, and two `hardware` runs on the *same* machine (the only case that could race) are
already serialized by the operator invoking them, not by this tool. See "Hardware profile
(AP4)" below.

### Atomic writes: atomicity, not durability (P3-9, fix-round 5, decided)

`state.py::atomic_write_json`/`atomic_write_text` (the `.pid.tmp` + `os.replace` convention every
persisted file in this package uses) guarantee **atomicity** -- a reader of `modelroom.json`,
`run-status.json` or a hardware profile always sees either the complete old content or the
complete new content, never a torn write from a crash mid-write -- but not **durability**: neither
function calls `os.fsync` on the temporary file's descriptor before `os.replace`, nor on the
containing directory afterward (the second fsync a fully crash-safe rename needs on POSIX, so a
directory-entry update is not itself lost on power failure). A power loss in the narrow window
between `os.replace` returning and the OS actually flushing the new file's pages to disk could
still lose the just-written content. Decided, not fixed: every file this convention writes is a
locally reconstructible cache of what `huggingface.co`/`registry.ollama.ai`/the local Ollama
daemon already say -- `run-status.json` is explicitly "runtime-only, not a contract" (this file's
own "Source of truth" section), and a lost or torn `modelroom.json`/hardware profile is recovered
by simply running `fetch`/`hardware` again, never by restoring from a backup. `fsync` on every
write would cost real latency on every run for a durability guarantee this tool's actual failure
mode (re-run the command) does not need.

**One exception to `os.replace`** (decided 2026-09-25): a configuration that does not exist yet is
created by `state.publish_new_text`, which never replaces a file -- two runs creating it at once
must not have the second one's text take the place of the first one's ("Search over the Hugging
Face API", `write_configuration`). Atomic in the same sense, not durable either.

### Run status (`run-status.json`)

Written by `modelroom/state.py::write_run_status` at the end of every `fetch` run that got past
the lock and the stale-run check -- including one that then ends exit `1` because at least one
area came back incomplete, since `write_run_status` runs before that exit code is even decided
(`cli.py::_run_locked`). **Not** written when the lock is held by another process, or when this
run is not newer than the stored snapshot (`state.check_run_is_newer` raising `StaleRunError`):
both of those also exit `1`, but before `run_fetch`/`write_run_status` are ever reached, so the
previous `run-status.json` (if any) is left completely untouched (corrected R7-13, fix-round 6;
the fix-round 5 CHANGELOG entry for this correction is still accurate -- it was this doc section
that had drifted, not the code). It is still the one file a caller can always read to see what a
run that actually executed did, whether or not the snapshot changed.

```json
{
  "run_at": "2026-09-22T09:00:00+00:00",
  "request_budget": {"used": 87, "total": 400},
  "areas": [
    {
      "source": "huggingface",
      "base_model_hf_repo": "acme/Nova-7B",
      "packager": "packager",
      "status": "complete",
      "error": null,
      "last_success": "2026-09-22T09:00:00+00:00"
    }
  ],
  "candidates": []
}
```

`areas` always lists every area this run covered -- one entry per (source, base model,
packager) triple, mirroring `Snapshot.areas` after the merge (see below). `candidates` is
always `[]` in this work package: ranking under-covered base models or suggesting new
packagers to configure is a later feature, not something AP3 computes.

### Search log (`search.json`)

Written by `modelroom/state.py::write_search_log` under `paths.state`, once per search of the guided
mode, atomically, after the search and before the list is shown -- and replaced by the next search:
it is about the search that just ran. Its data is one reading of that search
(`search.SearchOutcome.search_log`), the same the two notes on the screen come from, so a line and
the file can never say different numbers.

```json
{
  "schema_version": 3,
  "word": "qwen",
  "mode": "word",
  "filter_owners": true,
  "run_at": "2026-09-25T08:00:00+00:00",
  "accounts": [
    {"account": "Qwen", "class": "publisher", "hits": 2, "page_full": false},
    {"account": "unsloth", "class": "packager", "hits": 20, "page_full": true},
    {"account": "most downloaded", "class": "open", "hits": 20, "page_full": false}
  ],
  "requests": 14,
  "budget": {"used": 16, "limit": 150},
  "resolved": 3,
  "unresolved": [{"reason": "the repository does not say it packages a base model", "count": 1}],
  "models": [
    {"base_model": "Qwen/Qwen3.5-9B", "age": "legacy", "successor": "Qwen/Qwen3.8-27B", "release_basis": "computed"},
    {"base_model": "Qwen/Qwen3.8-27B", "age": "latest", "successor": null, "release_basis": "stated"}
  ]
}
```

`schema_version` is **3** since 2026-09-25, for `models`: one entry per resolved base model, in
the order of the hits, with its `age`, its `successor` and its `release_basis` (`stated`,
`computed`, or `null` for `unknown`; "Computed release") -- the list shows a star and no successor,
so the file is where the successor of a computed `legacy` can be read. Schema 2 added `mode`: `word`,
`id` or `catalog` (`SearchOutcome.mode`), so a reader can tell a word search from a typed repository
id and from the catalog pages of a search with no word at all. `word` is what was typed, the empty
string included. `scripts/selftest.py` reads the file with a schema gate and refuses any other
version; a log of a previous schema is not migrated -- a log is about the search that just ran.
`accounts` is one entry per asked **group**, in the order they were asked: `class` is what the search
asked it by -- `publisher` for a publisher account of a matching catalog family, `packager` for the
positive list, `open` for one of the two open lists, `typed` for the one page of a typed repository
id and `catalog` for a page of a search with no word. The last two are their own classes and not
`open`: the note on the screen is built from these classes, and calling them `open` made it say that
the two open lists were asked for a run that asked one request and no open list (live probe,
2026-09-25).
`hits` is what that group's pages answered with together, the number its group line shows, and
`page_full` is set for an **account** group only ("Search over the Hugging Face API"). `requests` is
the number of **pages**, two per account since 2026-09-25, so it is no longer the length of
`accounts`. `unresolved` is the reasons the
repositories of this search are no model of the selection list, each with the number behind it
(`guided_models.unusable_counts`). Why the file: those seven lines, the request count and the budget
answer "why did this account answer nothing", which is a question about the search and not about the
line a reader is looking at -- on a screen that has a choice to offer they said the same thing seven
times over (test round, 2026-09-24). The three automation commands write no such file; `fetch` still
prints `SearchOutcome.group_lines`/`summary_line` unchanged.

**Additive keys of AP9-KO (V8, V9; schema 3 stays).** `direct`: one `DirectLogEntry` per publisher model the
search asked the Hub for as a direct find, in the order they were asked, present only where at least one was.
`knowledge_hint`: the number of catalog models the hint of a search without a vLLM machine could not show for
its limit, present only where it was cut. Both are left out otherwise, so a search without either writes the
file it always wrote. "Package findings and the load check".

**Additive key of AP9-KO P5 (sweep p5-b1; schema 3 stays).** `failed_pages`: one
`{"account": <group label>, "reason": <the page's error>}` per search page that did not answer, in the order
asked (`{"account": "deepseek-ai", "reason": "unexpected status 503"}`), present only where at least one did.
The page still counts in `requests` and its group keeps its entry in `accounts`. "Search over the Hugging
Face API".

**Written without the lock, and deliberately so.** Unlike the configuration, a profile, a
measurement and the snapshot, this one file is not taken under `modelroom.lock`: it is a record of
one run's own search that no other writer of the state touches and no reader of the state depends
on, and the write is atomic, so a reader sees either the old file or the new one. Taking the lock in
the middle of step 2 would let a lock another process holds end a run that has a choice to offer
(second-model round, 2026-09-24). A write that fails is a `run.failed` sentence and the run goes on
with its list -- the search itself went through.

### Area semantics

An "area" (the `Area` model, already defined above) is one Hugging Face packager-owner's
candidate repos for one base model, or one base model's whole Ollama library entry:

- **Hugging Face**: one area per `(base_model.hf_repo, owner)`, where `owner` ranges over
  `modelroom/hf.py::target_owners(package_targets(config, base_model))` -- the owners of the
  base model's package target set, in the order that set produces them: every configured
  `packagers` entry plus the base model's own owner (a publisher counts as a packager of its
  own models), and any further owner an owner-bound `repos` entry names. Every target of that
  owner is fetched within the same area; a target that does not exist does not end the area,
  only a genuine failure does (see below). `Area.packager` is that owner. See "Target set and
  areas" for why a whole owner is one area rather than one area per target.
- **Ollama**: one area per base model that configures both `ollama_base` and `ollama_tag`;
  `Area.packager` is always `None`. **P3-2 (fix-round 5, clarified):** `modelroom/fetch.py::
  run_fetch` only ever calls `fetch_ollama_area` when a base model configures *both*
  `ollama_base` and `ollama_tag` (`if base_model.ollama_base and base_model.ollama_tag:`) -- a
  base model with neither configured gets **no Ollama area at all** in a `Snapshot`, not an
  empty `complete` one; `fetch_ollama_area`'s own early return for that case (`status="complete"`,
  `packages=[]`) is a defensive default for a *direct* caller of the function (covered by its own
  unit test), never something this section's "one area per..." rule describes as appearing in a
  merged `Snapshot`.

An area ends `incomplete` only for a genuine failure: any HTTP status other than `200`
(Hugging Face model-info additionally treats `401`/`404` as "not found", see below, not a
failure), a network error, a body that fails to parse, a response missing a field the fetcher
needs (Hugging Face: `sha`; Ollama: zero tags parsed from the tags page), a tree entry or
manifest layer in a shape the registry never actually sends -- not a dict, a tree entry whose
`type` is anything but `"file"` or `"directory"` (missing, empty, or an unknown value; R7-9/R8-1),
`path`/`mediaType`/`digest` missing or not a string, or a weight (`weights`/`weights_shard`, or an Ollama `.image.model`/
`.image.tensor` layer) with no non-negative integer `size` (fix-round 1, F3/F4: package
assembly for a candidate runs inside the same error handling as its tree/manifest fetch, so a
shape problem in one candidate ends only this area, never the whole run; a non-weight file
missing a size still defaults to `0`) -- or the run's request budget running out mid-area
(`error` is exactly `"budget exhausted"`, or, for an area this run never even started because
the budget was already exhausted before it, `"budget exhausted before this area was started"`,
see "Request budget" below). A candidate repo that simply does not exist, or a base model with
no Ollama configuration, is not a failure.

**Deviation from the original assumption, measured 2026-09-22** (see
`tests/fixtures/README.md`): an anonymous request for a Hugging Face repo that does not exist
returns HTTP `401` (`{"error": "Invalid username or password."}`), not `404` -- Hugging Face
does not let an unauthenticated caller distinguish "private" from "does not exist" any more.
`modelroom/hf.py` treats `401` exactly like `404` on the model-info call, for both a packager
candidate ("no package here") and a base model's own repo (architecture resolves `"unknown"`).

### Request budget

Every `fetch` run wraps its transport in `modelroom/http.py::BudgetedTransport`, default
budget `400` requests for the whole run (shared across every area, not per-area). Once
exhausted, the transport raises `BudgetExhaustedError("budget exhausted")` instead of making
the request; the fetcher in progress catches it and ends its *current* area `incomplete` with
`error` set to **exactly** `"budget exhausted"` -- never a repo- or tag-prefixed variant a
genuine shape problem's error message carries (P3-12, fix-round 5: `modelroom/hf.py`'s
tree-fetch catch clause distinguishes `BudgetExhaustedError` from every other exception for
exactly this reason, since the budget is a whole-run resource limit, not a fact about the
particular repo being fetched when it ran out). Areas already completed before the budget ran
out keep their results.

**Checked before each base model and before each area (fix-round 2, R4).**
`modelroom/fetch.py::run_fetch` reads `budgeted.remaining` before starting `fetch_base_model_meta`
for the next base model *and* before starting each of that base model's own areas -- not only
after a base model finishes. Before R4, the budget was only re-checked once a base model had
already finished all its areas, so a budget already at zero at the very start of the run (or one
that ran out partway through a base model's own areas) still let `fetch_base_model_meta` or a
fetcher be called at least once more: `BudgetedTransport` would then raise
`BudgetExhaustedError` for that call, which `fetch_base_model_meta` (never raises by contract,
see below) or the fetcher in progress would catch and silently resolve into a fresh, budget-
starved "unknown"/empty result -- overwriting a real previous reading, or recording a bare
`"budget exhausted"` transport error instead of the `"budget exhausted before this area was
started"` message a caller expects for an area that was never actually attempted. With the
pre-check in place, a base model or area the budget has already run out for never calls into the
transport at all, so it always gets exactly the `"budget exhausted before this area was
started"` outcome (below), whether the budget ran out before this run even reached that base
model or between two of that same base model's own areas.

**A base model this run never even started reading** (fix-round 1, F9: the budget ran out
before its turn came up in the family/base-model loop) is different from one that was
*attempted* and ran out mid-area: `modelroom/fetch.py::run_fetch` stops calling
`fetch_base_model_meta`/`fetch_hf_area`/`fetch_ollama_area` at all once the budget hits zero,
every area that base model would have needed gets `AreaOutcome(status="incomplete",
error="budget exhausted before this area was started")`, and the base model itself keeps its
**previous `BaseModelSpec`** unchanged when the old snapshot has one (an unknown placeholder
otherwise) -- never a fresh "unknown" reading that would silently overwrite a real one. "Base
models are always replaced by this run's results" (below) means *whenever this run attempted to
read them*; a base model the run never reached is the one exception, and keeps its previous
spec. A base model that *was* attempted and merely ran out of budget mid-area still gets a
fresh spec from whatever `fetch_base_model_meta` managed to read, exactly as before.

**Redirect chains are booked hop by hop (fix-round 2, R5, superseding F10).** F10 had
`UrllibTransport` follow a redirect chain internally and report how many requests it cost via
`Response.requests_made`, with `BudgetedTransport` charging the extra hops *after* the call
already returned -- so a chain could overshoot the budget by up to its own length, and a chain
that failed partway through was never booked for the hops it did make. R5 inverts the
composition instead: `UrllibTransport` makes exactly one request per call and returns a 3xx
exactly like any other status; `modelroom/http.py::RedirectingTransport(inner)` is what follows
a chain, calling `inner` once per hop (up to `MAX_REDIRECTS = 5` hops, the first request already
counting as hop 1; more hops raise `RedirectLimitError`, a `RuntimeError`, before the hop that would exceed the limit is
ever made). `run_fetch` composes `RedirectingTransport(BudgetedTransport(transport, budget))` --
`BudgetedTransport` as the *inner* layer -- so every hop of a chain arrives at the budget check
as its own separate call, checked and booked (or raising `BudgetExhaustedError`) before it is
made, failure paths included, never only after the fact.

**A tensor repository's quantization costs at most one more request per repo and revision, and
only when the fetch may build tensor packages at all (AP9-SV2-P, P2).**
`hf.py::fetch_hf_area` takes `build_tensor` (`False` by default, matching every fetch before this
step): with it `False`, a tensor weight group never reads `config.json` and never runs the
safe-path check at all. What it then becomes depends on what this project already knows about
that exact repository and revision, and two kinds of knowledge are kept apart
(`hf.py::_carried_tensor_facts`, the one place this is decided):

- **Content facts** -- `quantization` and `quantization_source` -- depend on the repository and
  its revision alone. When the previous snapshot holds the same package identity at exactly this
  revision, built past the format lock by an earlier run (its reason is not `format`), they are
  kept as they are, whatever that package's verdict was: an `unresolved` draft model
  (`draft.safetensors`, `file_stem`) keeps its `BF16`/`config` too. Another revision may be other
  content, so nothing is kept then.
- **Verdicts** -- `provenance` and `unresolved_reason` -- also depend on the base model the
  package is assigned to now. They are never copied from the previous snapshot: with content
  facts at hand, `decide_provenance` decides again against the current assignment, the current
  target set and the approvals, which needs no request. The same repository moved to another base
  model at the same revision therefore gets the verdict a fetch with `build_tensor` would give it,
  never the one of its former assignment.
- With no content facts at all (never built, or another revision), the group stays locked
  `unresolved`/`format`, exactly as AP9-SV1 always left it.

Whether such a package is shown anywhere is not the fetch's business: the render decides that per
machine ("Render (schema 2)", tensor visibility). With it `True`, `_assemble_packages` reads the repo's `config.json`
at the already-known revision once -- cached for the rest of that repo's groups, so a repository
with two tensor stems (a main build and a draft/MTP model dropped in the same repo) still costs
one request, not two. A repository with no tensor group costs nothing extra either
way; the request is charged through the same shared transport as every other call in the area, so
it can also end the area `incomplete (budget exhausted)` like any other. Every other failure of
this one request -- a status other than `200`, a body over `tensor_quant.MAX_CONFIG_JSON_BYTES`
(64 KiB), invalid JSON -- resolves `quantization` to `None` instead, never ends the area: the
label is a documented extra on top of the package, not a requirement for it to exist (P2).

**`build_tensor` itself** is the guided mode's own decision, reached in its fetch step
(`guided.py::_fetch_step`) through the one gate the tensor search below reads as well,
`guided_context.vllm_among_chosen`: `True` when **at least one machine this run chose** has vLLM
(`runtimes.machine_has_vllm` per machine) -- by the `runtimes` of its stored profile
(`[machines.<name>].profile`), or, for the machine the run is on, by a vLLM server detected serving
*in this very run's own step 1* (`GuidedRun.servers.found["vllm"]`, an actual `ServedModel`
answer only -- a server asking for a key does not count as detected, E5). Another chosen machine
has no live detection here, so its stored profile alone speaks for it. A plain `modelroom
fetch` (`cli.fetch_with_config`/`run_fetch` called directly, with no pointer or profile of their
own) never sets it, and reads/writes exactly as AP9-SV1 always did. This is also the base model's
own repository's whole story: AP9-UX2 P3 made it a `repos` target on *every* machine, for SV1's
live server match, well before SV2-P existed -- `build_tensor` is what decides whether that one
repository's tensor group may resolve past the format lock, never whether it is fetched at all.

### Transport security and proxying

`modelroom/http.py::UrllibTransport` opens exactly the URLs AGENTS.md's attack-surface promise
names: HTTPS to `huggingface.co`/`ollama.com`/`registry.ollama.ai`, HTTP to the local Ollama
daemon. `_check_allowed(url, allowed_https_hosts, allowed_http_origins)` runs before anything is
opened, against every URL this package ever hands a `Transport` -- including a redirect's
`Location` header and a paginated Hugging Face tree's `Link` header, both of which loop back
through this same check on the next hop (`RedirectingTransport`).

**The local allow-list is an origin, `(host, port)`, not a bare host (R7-4, fix-round 6).**
`_ALLOWED_HTTP_ORIGINS` is exactly `{("127.0.0.1", 11434)}`, the real Ollama daemon
(`modelroom/ollama_local.py::DEFAULT_BASE_URL`); a URL with no explicit port, or a loopback port
other than `11434`, is refused -- a poisoned `Location`/`Link` header naming a different local
port a caller happens to have something listening on is not "the local Ollama daemon" and must
not be treated as one.

**A proxy is honored, but never widens what this transport will reach (R7-5, fix-round 6).**
`_build_opener` registers `urllib.request.ProxyHandler()` (its stdlib default: reads
`HTTP_PROXY`/`HTTPS_PROXY` from the environment, honors `NO_PROXY`) alongside the handlers that
enforce the allow-list. `_check_allowed` in `UrllibTransport.__call__` runs against the request's
own TARGET url *before* the opener ever consults a proxy -- a proxy can only change how an
already-allowed request reaches its target, never add a target that was not already on the
allow-list. An operator who wants the local Ollama daemon reached directly even when a proxy is
configured for everything else sets `NO_PROXY=127.0.0.1` in the environment; that is a deployment
decision, never something this package decides on its own.

**A caller may cap an answer (AP9-KA-L).** `Transport.__call__` takes the keyword `max_bytes: int | None =
None`. `RedirectingTransport` and `BudgetedTransport` pass it unchanged to every hop (and only when it is
set, so a transport that knows no such keyword keeps working under every caller that sets none);
`UrllibTransport` reads at most `max_bytes + 1` bytes of a success **and** of an error answer and raises
`ResponseTooLargeError` before anything is parsed; `FixtureTransport` holds its recorded body against
the same limit and keeps each call's limit in `limits`. A caller that passes none gets the answer whole,
as before. `RedirectingTransport(inner, max_hops=MAX_REDIRECTS)` takes a narrower hop limit for one
caller (the first request counts as a hop); a redirect past it raises `RedirectLimitError` (a `RuntimeError`, like any other chain
that does not resolve); the catalog loader turns only that one into the redirect reason on the screen.

**Another computer's own daemon is never reached over HTTP (AP9-FR, FR-B1).** `SshDaemon`
(`modelroom/remote_client.py`) is a separate transport again, `daemon.py`'s protocol but not its
implementation: it never opens a socket itself, it hands a fixed request shape to the system `ssh`
process (`modelroom/ssh.py`, "The remote protocol v1", above) with a fixed argument vector, never a
shell string, and the alias checked against `ssh.ALIAS_RE` first. `UrllibTransport`'s allow-list is
unchanged by this: the HTTP origin it reaches stays `127.0.0.1:11434` only, on whichever machine
the ssh session's own `remote_server.py` runs.

### Merge rule (building this run's `Snapshot`)

`modelroom/state.py::merge_snapshot(old, run_at, area_outcomes, base_models)` builds the
`Snapshot` a run writes:

- **Base models** are always replaced wholesale by this run's results whenever this run
  *attempted* to read them -- an unknown architecture, or a base model this run tried and could
  not reach at all, is still "this run's result", never silently carried over from `old`. A
  base model the run never reached because the budget was exhausted before its turn is the one
  exception (see "Request budget" above, F9): it keeps its previous spec.
- **Packages of an area no longer configured this run are dropped, not kept forever**
  (fix-round 1, F2): `merge_snapshot` only ever carries an old package into the result when its
  area (`_area_key_of_package`) is among this run's `area_outcomes` -- a base model or an area
  removed from the configuration leaves no orphan packages behind, and
  `Snapshot._check_packages_reference_known_base_models` never has anything to reject. An
  `incomplete` area's old packages are still kept, since that area *is* among this run's
  outcomes, just not confirmed this run (see below).
- **A `complete` area** replaces its packages: every package this run found is written with
  `active=True`; a package that existed in `old` under this exact area (same source, base
  model and -- for Hugging Face -- the same packager owner as the repo it came from) but was
  not found this run is kept with `active=False` rather than deleted, so its history survives.
  A package's `observed_at` is carried over from `old` when the same package identity
  (`quantization.package_identity_key`) already existed; `last_seen` is always bumped to
  `run_at`. The identity key carries no base model, so a repository that moved from base
  model A to base model B between two runs is "not found" by A's area and "found" by B's: the
  stale step of an area never touches an identity that another area of this run has already
  published, so the result is the same whichever area is processed first.
- **An `incomplete` area** leaves every package that belonged to it in `old` completely
  untouched (not even `last_seen` moves) and records `status="incomplete"`, `error`, and the
  *old* `Area.last_success` (never bumped, since nothing was actually confirmed this run).
- An area this run never touches at all (not configured any more) simply does not appear in
  the result, and neither do its packages (see F2 above) -- `merge_snapshot` only ever iterates
  `area_outcomes` and the `base_models` list it is given, it does not scan `old` for leftovers
  to carry forward on its own.

### Version rule

`modelroom/state.py::check_run_is_newer(old, run_at)` refuses a run whose `run_at` is not
*strictly* newer than `old.run_at` (`run_at <= old.run_at`, so the same second counts as
stale, not just an earlier one), raising `StaleRunError` before anything is fetched or
written. `run_at` itself is the run's start time, UTC, truncated to whole seconds
(`datetime.now(timezone.utc).replace(microsecond=0)` in `modelroom/cli.py`, or an injected
value in tests) -- second precision is deliberate: two `fetch` runs started in the same wall
second are indistinguishable and the second one must not silently win.

### Approval carry-forward across fetch runs (fix-round 1, F6)

Before F6, both fetchers called `decide_provenance(stub, base_model, tags, [])` -- always an
empty approvals list -- so a package's `Approval` was lost the moment its area was fetched
again, even when the content it was approved for had not changed. `modelroom/fetch.py::
run_fetch` now builds `previous_by_key = {package_identity_key(p): p for p in
old.packages}` once per run and passes it into `fetch_hf_area`/`fetch_ollama_area` as a new
`previous_by_key` keyword parameter (default `None`, so a direct unit test may omit it). Each
fetcher looks up the freshly assembled package's own identity key in `previous_by_key` before
deciding provenance: when a previous package of the same identity carried an `Approval`, that
approval is attached to the new package and passed to `decide_provenance` as `[approval]`
instead of `[]` -- the existing content rule (`Package` model, "Three more invariants" above)
still decides whether it actually resolves `"approved"` (the content still has to match the
*current* revision/digest) or falls through to the metadata rules. The approval stays on the
`Package` regardless of which way that goes, so its history is never lost even once it is no
longer active. `merge_snapshot` needed no change for this: the freshly built package already
carries the right `approval`/`provenance` by the time it reaches the merge.

**Same base model required (fix-round 2, R3).** `package_identity_key` is only
`(source, repo|ollama_name, filename-or-tag)` -- it says nothing about which base model the
package belongs to. The carry-forward is therefore only performed when
`previous.base_model_hf_repo == stub.base_model_hf_repo` in addition to the identity match:
the same packager repo (or the same Ollama tag) reassembled under a *different* base model --
a family reconfigured to a new `hf_repo`, or two base models that happen to share a packager
repo -- must never inherit an approval that was never given for it. When the base model
differs, the approval is not carried forward at all, not even attached to the new package for
history, since it never belonged to that package's lineage in the first place.

### `parameters_b` when Hugging Face has no answer this run

**Rewritten in fix-round 1 (F11).** `BaseModelSpec.parameters_b` is `float | None`: `None`
means "not measured this run, and no previous reading exists", never a fabricated number that
could be mistaken for a real one; when it *is* a float, it must still be `> 0`. A base model
whose publisher repo could not be reached, or whose model-info response carries no
`safetensors.total`, therefore cannot simply report `0.0` (which would fail the `> 0`
constraint) and no longer reports a placeholder either. `modelroom/fetch.py::run_fetch` falls
back, in order: the previous snapshot's `parameters_b` for the same `hf_repo`, when one exists
(a real earlier reading); otherwise `None`. A base model this run never even attempted to read
because the request budget was exhausted first keeps its previous spec wholesale (see "Request
budget" above, F9) rather than going through this fallback at all. Every consumer of
`parameters_b` has to handle `None`: `modelroom/provenance.py::_decide_ollama` resolves
`("unresolved", "parameters_unknown")` when it is `None`, since the size-token check a few
sections below cannot run without a real number to compare against. Reading the real count
from a packager's GGUF metadata (`gguf.total` in a Hugging Face model-info response) instead of
only the publisher's `safetensors.total` is left for a later work package.

**Fix-round 2 (R6), no migration.** A snapshot written before this change may still carry the
placeholder `1.0` for a base model whose parameter count Hugging Face does not expose; the
carry-over rule would keep it. Rebuild such a snapshot once (delete `modelroom.json`, run
`fetch`) -- there is no automatic migration. No migration code was written: the package is
unreleased and the only existing snapshot is the operator's own, which will be rebuilt.

### Ollama packages: files and `default_context`

An Ollama manifest's weight layer(s) become one `PackageFile` per `application/vnd.ollama.
image.model` layer (role `weights`, the registry's own digest, `format="gguf"`). A
tensor-only manifest (only `application/vnd.ollama.image.tensor` layers, no `.image.model`
layer -- an MLX-style build) can carry hundreds of per-tensor layers; rather than modeling
one `PackageFile` per tensor, `modelroom/ollama.py` aggregates them into a single synthetic
file (summed size, no digest, `format="tensor"`), since a tensor package is always
`("unresolved", "format")` regardless of its individual tensor layout, which is not part of
this catalog's contract. A weight layer with no non-negative integer `size` is a shape error
that ends the area `incomplete`, not a silent `0` (fix-round 1, F4 -- see "Area semantics"
above).

`Package.default_context` (Ollama's `num_ctx`) is always `None`, and stays so until a decision
about this package's attack surface is taken. Reading it means fetching the manifest's `params`
layer by digest, and the registry does not serve that blob itself: measured 2026-09-23,
`GET registry.ollama.ai/v2/library/qwen3.5/blobs/sha256:9371364b...` answers `307` with a signed
`Location` on an object-storage host whose name is not fixed. `modelroom/http.py` opens exactly
three HTTPS hosts (AGENTS.md, "Security, definition of done"); following that redirect means
widening the allow-list to a host that cannot be named up front. Nothing else depends on the
field: a ranking's context is `scenario.context_requested`, never a package's own declared one,
which is shown ("package declares 4096") and never computed with.

**`manifest_digest` (fix-round 1, F5, superseding the original `HEAD`-request design):**
`modelroom/ollama.py::_fetch_manifest` computes the digest as `sha256:` plus the sha256 of the
manifest `GET` response's raw body, and makes no other request for it. Measured 2026-09-22
against the real registry (`registry.ollama.ai/v2/library/qwen3.5/manifests/9b`, 709 bytes):
hashing the `GET` body gives exactly the value the registry's `ollama-content-digest` `HEAD`
response header used to state, so the extra `HEAD` request this project used to make for every
tag bought nothing and is gone.

### Ollama size-token tolerance (AP3 acceptance fix)

`modelroom/provenance.py::_decide_ollama` reads the Ollama tag's size token (the first
`-`-separated token, e.g. `9b` in `9b-q4_K_M`) and compares it against the base model's
measured `parameters_b`. A live snapshot (13 base models, 79 areas, 358 packages, 2026-09-22)
showed the original exact-equality rule (`first_token == f"{parameters_b:g}b"`) can never match
real data: `safetensors.total` counts embeddings, so `Qwen/Qwen3.5-9B` measures
`parameters_b = 9.653104368`, while its library tag is plainly `9b`.

The rule is now a tolerance, not an equality: the first token must fully match a size-token
pattern (`<number>b` for billions or `<number>m` for millions, case-insensitive -- `7banana`
never matches at all, it is not a size token), and the declared size must be within 15 %
**of the declared value** (`abs(measured - declared) <= 0.15 * declared`) -- not of the
measured `parameters_b`. 15 % is a heuristic chosen so that real registry tags pass (`9b` for a
measured `9.653104368`) while a genuinely wrong tag (`70b` for a measured `7.0`) still fails, not
a measured optimum (fix-round 1, F11, rewording only -- the formula itself is unchanged).
Examples: `9b` for a measured `9.653104368` passes (0.65 vs. a 1.35 allowance); `30b` for a
measured `30.5` passes; `1.5b` for a measured `1.54` passes; `70b` for a measured `7.0` still
fails (63 vs. a 10.5 allowance) -- a genuinely wrong tag is still rejected, only the packager's
rounding is tolerated. **`parameters_b` must have been measured at all** (fix-round 1, F11): a
`None` value (see "`parameters_b` when Hugging Face has no answer this run" above) resolves
`("unresolved", "parameters_unknown")` before this tolerance check ever runs, never a
size-token mismatch. The full tag still has to equal `base_model.ollama_tag` or start with
`ollama_tag + "-"` (unchanged token-boundary rule) once the size token itself passes.

### Package file-stem naming conventions (AP3 acceptance fixes)

`modelroom/provenance.py::_decide_huggingface`'s file-stem check
(`quantization.match_base_prefix`) and `modelroom/quantization.py::parse_hf_quant` were
extended against the same live snapshot to match every allow-listed packager's real naming
convention, not only the most common one:

- **Per-quant subfolders** (unsloth): a weight file's basename is matched against
  `<base_name><sep><QUANT>`, never the full path -- `BF16/GLM-5.2-BF16-00001-of-00033.gguf`
  matches base `zai-org/GLM-5.2` on its basename `GLM-5.2-BF16` alone, and two files with the
  same basename in different subfolders (e.g. two `00001-of-00002.gguf` shards under different
  quant folders) still keep distinct package identities (the folder segment is never stripped
  from `quantization.identity_stem`).
- **Dot separator** (mradermacher): `<sep>` is `-` (most packagers) or `.`
  (`Qwen3.5-9B.IQ4_XS.gguf`, `Qwen3.5-9B.mmproj-f16.gguf`).
- **Qwen's own shard suffix**: `-NNNNN-of-NNNNN` may be preceded by an extra `-split` marker
  (`Qwen3VL-235B-A22B-Instruct-F16-split-00001-of-00010.gguf`); such files are role
  `weights_shard` like any other shard.
- **Non-weight markers**: a basename *containing* `mmproj` (role `mmproj`) or `imatrix` (role
  `other`) is never a weight file and never carries a quantization, even mid-name
  (`Qwen3-VL-235B-A22B-Instruct.mmproj-Q8_0.gguf`, `MiniMax-M3-imatrix.gguf`); previously only a
  basename *starting with* `mmproj` was recognized.
- **`QUANT_ORDER`** gained the ternary/1-bit family (`TQ1_0`, `TQ2_0`, their `UD-` dynamic
  variants, `UD-Q1_0`), the missing `IQ1`/`IQ2`/`IQ3`/`Q2_K`/`Q3_K` members, `MXFP4`/
  `MXFP4_MOE`, and case-insensitive matching so mradermacher's lowercase `f16`/`bf16` resolve
  to the canonical uppercase member.

### Package fetch fields (already part of `Package`, not new)

AP1 already gave `Package` the `active: bool`, `observed_at: datetime` and `last_seen:
datetime` fields `fetch` needs to record history (see the `Package` model above); AP3 uses
them exactly as documented there and did not need to extend the contract.

## Hardware profile (AP4)

**Schema 1, read-only since the measurement work package.** This is how `hardware` wrote a
profile before it measured the machine itself: `<state>/hardware/<machine>.json`
(`HardwareSnapshot`, models documented above), filled from `llmfit` and the local Ollama
daemon. `hardware` now writes schema 2 (see "Hardware measurement" below); `write_hardware_snapshot`
and `load_existing_hardware_snapshot` stay for `render`, which reads these files until the
renderer switches, and for `modelroom migrate`, which converts them. The exit codes below
describe that old command; the current ones are in "Hardware measurement".

Unlike `fetch`, the named machine only has to be *configured* (present in
`Configuration.machines`), not a `writer` -- hardware is measured on every machine, including
inference-only ones. Exit codes reused the table above unchanged: `2` for an unconfigured
machine or a missing/too-old `llmfit`, `3` for an existing hardware file with an unsupported
`schema_version`, `0` otherwise -- including when the local Ollama daemon could not be reached
(`installed` is then `None` with a reason, never a command failure).

### Local llmfit binding

`read_llmfit_reference(runner, min_version)` is what the schema-2 measurement uses: it runs the
two calls below and never raises. A missing or too old `llmfit` (`LlmfitUnusableError` -- it was
never asked) is `absent`, a call that failed is `error`, and a good run is `available` with
`ram_gib`/`vram_gib` for the cross-check. The raising functions below stay for schema-1 callers.

`modelroom/llmfit.py` calls `llmfit --version` first (`check_llmfit_version`), comparing it
against `config.llmfit.min_version` as an integer `(major, minor, patch)` tuple -- a version
below the minimum, or a missing `llmfit` binary (`FileNotFoundError` from the runner), both
raise `LlmfitError` naming the minimum version and an install hint; the CLI maps this to exit
code 2. `llmfit system --json` is then called and its `system` block mapped to
`HardwareSnapshot`'s hardware fields:

| `llmfit system --json` field | `HardwareSnapshot` field |
|---|---|
| `system.gpu_vram_gb` | `vram_gib` |
| `system.total_ram_gb` | `ram_gib` |
| `system.available_ram_gb` | `free_ram_gib_at_measurement` |
| `system.gpu_name` | `gpu_name` |
| `system.backend` | `backend` |
| `system.unified_memory` | `unified_memory` |

llmfit's own `_gb` fields are already GiB -- its source divides by `1024**3`, not `1000**3` --
so they are taken unchanged (measured 2026-09-22 on the reference laptop: `total_ram_gb`
`127.46` for a 128 GB machine, `tests/fixtures/llmfit_system_laptop.json`). The `providers`
block and `gpus[]` (per-GPU detail, for a multi-GPU machine) are both ignored in v1. A
GPU-less machine's `gpu_vram_gb` defaults to `0.0` and `gpu_name`/`backend` to `None` rather
than raising; only a response with no `system.total_ram_gb` at all is a hard `LlmfitError`,
since every other field can legitimately be absent.

### Local Ollama inventory

`modelroom/ollama_local.py::fetch_installed_models` reads `GET
http://127.0.0.1:11434/api/tags` (the base URL is a parameter; the daemon on the machine
running `hardware`, never the public Ollama registry `modelroom/ollama.py` fetches from). Each
entry becomes one `InstalledModel(name, digest, size_bytes, observed_at)`. The daemon reports
`digest` as bare hex, with **no** `sha256:` prefix (measured 2026-09-22,
`tests/fixtures/ollama_tags_local.json`, trimmed from a real response to three entries); the
fetcher prepends `sha256:` before building `InstalledModel`. The daemon being unreachable
(a transport error, a non-`200` status, an unparseable body, or a response missing the
`models` field) is **never** a failure of the `hardware` command -- it comes back as
`(None, reason)`, which the CLI writes as `installed=None` /
`installed_unavailable_reason=reason`.

### `hardware` write

`_cmd_hardware`/`hardware_with_config` (`modelroom/cli.py`) run the llmfit gate, `llmfit
system --json`, and the Ollama inventory read, then build a `HardwareSnapshot` with
`measured_at` = now (UTC, no microseconds, same convention as `fetch`'s `run_at`) and
`measurements` carried over unchanged from the machine's existing hardware file, when one
exists (`state.py::load_existing_hardware_snapshot`, `load_hardware_snapshot` under the hood --
an unsupported `schema_version` there is exit `3`, exactly like `fetch`'s snapshot check). The
resulting dict is validated and written with `state.py::write_hardware_snapshot`
(`load_hardware_snapshot` then `atomic_write_json`), never partially. `hardware` prints one
summary line to stdout: the machine name, `vram_gib`/`ram_gib`, and either `installed: <n>` or
`installed: unknown (<reason>)`.

## Fit contract v1

`modelroom/fit.py::compute_fit(package, base_model, hardware, machine_config) -> Fit` is a
pure function (no I/O) computing whether one `Package` fits one machine, given its
`HardwareSnapshot` and the `MachineConfig` naming its reserved headroom. **A renderer must
always label this result "fit (computed, v1)", never "runs"** -- it is computed from a
memory-sizing formula, not a measurement of the package actually loading and generating
tokens (that is what `Measurement`/`bench`, a later work package, are for).

**Coverage.** A `base_model.architecture.kind` of `"dense_classic"` or, since AP9-UX2 (ADR 0010),
`"hybrid_linear"` (e.g. `Qwen/Qwen3.5-9B`) is judged by this formula; anything else (an MoE
architecture, a hybrid one that does not meet the invariants, resolved to `"unknown"`) comes back
`fit_class="unknown"`,
`reason="architecture not covered by v1"` from `compute_fit`, and is computed from its size by
`compute_fit_v2` instead ("Fit from size (basis `size`)" below, decided 2026-09-24). Only a `complete`
`format == "gguf"` package is judged; an incomplete package or a non-GGUF (`tensor`) package
also comes back `"unknown"` with a reason. A `Fit` for an unknown case carries zeroed
numeric fields (`need_gib`, `weights_gib`, `kv_gib`, `pool_gib`, `reserve_gib` all `0.0`,
`context` `0`, `mode` `None`) -- these are placeholders, not measurements, and a renderer must
never print them for an `"unknown"` fit.

**Formula.**

- `weights_gib` = the sum of `size_bytes` over every `PackageFile` with role `weights` or
  `weights_shard`, divided by `1024**3`.
- `context` = `package.default_context` when the package states one, else `8192`
  (`context_assumed=True` in that case). The assumption is deliberately visible in the result,
  never silently baked into `need_gib` alone.
- `kv_gib = (kv_token_bytes * context + state_bytes) * requests / 1024**3`
  (`modelroom/architecture.py`): `kv_token_bytes = 2 * L * KVH * D * 2 bytes`, where `L =
  architecture.num_hidden_layers` for `dense_classic` and the number of `"full_attention"` layers
  for `hybrid_linear`, `KVH = architecture.num_key_value_heads`, `D = architecture.head_dim` --
  the standard KV-cache size for a GQA/MQA transformer at 16-bit precision (`2 bytes` per element,
  Ollama's default, the leading `2` for key and value each). `state_bytes` is `0` for
  `dense_classic`; for `hybrid_linear` it is the state each request keeps in its
  `"linear_attention"` layers whatever the context: `n_linear x (value_heads x key_dim x
  value_dim + (conv_kernel - 1) x (2 x key_heads x key_dim + value_heads x value_dim)) x
  linear_state_bytes`. Qwen3.5-9B and -4B: `kv_token_bytes` 32,768 (8 of 32 layers), `state_bytes`
  52,690,944 -- the size basis below overstated their KV cache by 4.5 times (L 32k: 4.50 GiB
  instead of 1.00 GiB).
- `need_gib = weights_gib * 1.10 + kv_gib + 0.50` -- a 10 % overhead on the weights themselves
  (allocator/runtime overhead) plus a flat 0.50 GiB fixed cost (activations and other small
  buffers), on top of the KV cache.

**Pool and mode.** `available_vram = hardware.vram_gib - machine_config.reserve_vram_gib`.
When `need_gib <= 0.98 x available_vram` (`TIGHT_RATIO`, AP9-UX2 P2), the package fits the GPU: `mode="gpu"`, `pool_gib =
available_vram`, `reserve_gib = reserve_vram_gib`. Otherwise it falls back to the RAM pool:
`pool_gib = hardware.ram_gib - reserve_ram`, `reserve_gib = reserve_ram`, `mode="cpu_gpu"` when
`hardware.vram_gib > 0` (some GPU exists, just not enough) else `mode="cpu"`. `reserve_ram` is
`reserve.ram_reserve(machine_config, ram, unified)`, read in `fit._choose_pool` and nowhere else
(AP9-RM, ADR 0012): `machine_config.reserve_ram_gib` where it is a number, else a quarter of the
RAM between 2 and 16 GiB (no upper bound on unified memory). A fit from the size or the parameter
count without a known RAM tries the cards first; the system memory then counts as 0 (with no number,
reserve 0), so it is `no_fit` there.

**Several graphics cards (decided 2026-09-26).** A profile's memory is a pool of cards
(`fit.GpuPool`: each card's `vram_gib` from `gpus`, and whether it is unified memory); schema 1's
one `vram_gib` is one card above 0, else none. The rule, in order:

1. Unified memory, as below.
2. One card first, the way Ollama loads a model, and only with at least one card:
   `available_one = max(cards) - reserve_vram_gib`; when `need_gib <= 0.98 x available_one`, `mode="gpu"`,
   `pool_gib = available_one`, `reserve_gib = reserve_vram_gib`, not capped. With one card this
   is the rule above, and no value of a fit changes.
3. All cards, with two or more: `available_all = sum(card - reserve_vram_gib)` -- each card keeps
   its own reserve; when `need_gib <= 0.98 x available_all`, `mode="gpu_split"`, `pool_gib =
   available_all`, `reserve_gib = n x reserve_vram_gib` (the reserve of all cards), **not capped**:
   the package lies in graphics memory whole, and what the spread costs is speed, which the load
   test measures.
4. The system memory as above: `mode="cpu_gpu"` with cards, `"cpu"` without, capped at `fits`.

**A pool is chosen only where the package fits it (AP9-UX2 P2, ADR 0010):** "fits" means `need_gib
<= TIGHT_RATIO x pool` with the pool after its reserve, the same line the classification draws.
Until then a package between 98 and 100 % of the card was chosen for the card and classified
`no_fit` there, although the next pool held it (`Q4_K_M` of Qwen3.5-9B at L 32k on the laptop
of the fixtures: 10.82 GiB on 10.99 GiB). The last pool -- the system memory, or unified memory,
which has no fallback -- is classified as it is: above 98 % it stays `no_fit`. `no_fit` therefore
means that no pool holds the package with a margin of 2 %.

The size basis keeps its cap at `fits` on every pool. `OLLAMA_SCHED_SPREAD=1` makes the daemon
spread a model although one card holds it: the fit then says `gpu` and the daemon spreads all the
same.

**Parallel requests (decided 2026-09-25).** `requests` are the parallel slots the Ollama daemon
is assumed to keep (`OLLAMA_NUM_PARALLEL`), not the number of HTTP requests waiting; the context
holds per slot. The weights are loaded once, the KV cache once per slot, so on both bases
`kv_gib = context x kv_per_token x requests` and `need_gib` grows with it. Every `Fit` carries
the `requests` it was computed for (default 1: `compute_fit` of schema 1 and a fit nobody could
compute say 1). The number assumes that the daemon really keeps that many slots and that
`num_ctx` takes effect (the load test sets it, a model default can differ); a memory fit proves
nothing about whether an architecture serves requests in parallel, and the scheduler limits some
architectures to one slot. `fit.OLLAMA_REQUEST_HINT = 8` is a hint, never a limit -- rule of
thumb, not measured; the release that measures it moves the number -- with the text
`beyond 8 requests at once this fit says nothing about throughput: measure under load, or look at
a serving stack (rule of thumb, not measured)` for a view to show above it.

**Unified memory (decided 2026-09-25).** A profile of `gpu_state` `unified_memory` has no
graphics memory of its own: the system and the model share one memory. Its pool is
`ram_physical_gib - reserve_ram_gib - reserve_vram_gib`, `reserve_gib` is the sum of both
reserves, `mode="gpu"`, no fallback onto the same RAM, and the class is not capped on the
architecture basis (the size basis stays capped at `fits`); a pool of 0 or less is `no_fit`.
Example: 32 GiB, reserves 8 + 1 -> pool 23 GiB; 128 GiB with the computed reserve (a quarter,
no upper bound, ADR 0012), 32 + 1 -> pool 95 GiB. Feeding the RAM in as graphics memory would let
the GPU branch compute with 31 GiB and lose the system reserve -- hence the branch of its own.

**Classification.** `ratio = need_gib / pool_gib` (a `pool_gib <= 0` is always `no_fit`,
never a division): `ratio <= 0.60` is `ample`, `<= 0.85` is `fits`, `<= 0.98` is `tight`,
anything above is `no_fit`. **Off the GPU (`mode` is `cpu_gpu` or `cpu`), the class is
capped at `fits`** -- a ratio that would otherwise read `ample` still comes back `fits`,
because `ample` is reserved for a package with room to spare in VRAM. Each word says only what
was computed, the share of the pool, and the value is the text a view shows (decided
2026-09-27, ADR 0006). llmfit's `perfect/good/marginal/too_tight` are `ample/fits/tight/no_fit`
here; the thresholds are its.

**Worked examples** (also `tests/test_fit.py`, values restated in each test's own comments so
they can be recomputed by hand): a dense 8B Q4_K_M package (~5.0 GiB weights, `L=36, KVH=8,
D=128`, context assumed 8192) has `kv_gib = 1.125` GiB and `need_gib = 7.125` GiB. On the
reference laptop (`vram_gib=11.94`, `reserve_vram_gib=1.0`) that is `mode="gpu"`, ratio
`0.6513` -> `fits`. On a small server (`vram_gib=0`, `ram_gib=7.56`, `reserve_ram_gib=3.0`) it
falls to `mode="cpu"`, `pool_gib=4.56`, ratio `1.5625` -> `no_fit`. A dense 9B Q4_K_M
package (~5.6 GiB, `L=42, KVH=8, D=128`) has `need_gib = 7.9725` GiB: `fits` on the laptop
(ratio `0.7288`), `no_fit` on the server (ratio `1.748`).

**`Measurement.profile_measured_at` cross-check.** `Measurement` itself does not check its
`profile_measured_at` against any particular `HardwareSnapshot.measured_at` (see the
`Measurement` model above) -- a renderer reading `hardware.measurements` alongside
`hardware` refuses to show a measurement whose `profile_measured_at != hardware.measured_at`
as if it were current for that profile, since the hardware may have changed since that
measurement was taken. This check belongs to the renderer (a later work package), not to
`compute_fit` or to the `Measurement`/`HardwareSnapshot` models themselves.

### Fit from size (basis `size`)

Decided 2026-09-24, after a hand test in which 25 of 26 packages stood under "architecture not
covered by v1" and one package was ranked: an architecture nobody can read is a reason to compute
more coarsely, not a reason to say nothing. A `Fit` says which of the two it is in `basis`, and
the provenance of both stays `computed` -- the word for a number this package worked out rather
than measured. `size` is never shown as a measurement and never as a promise.

**Two entry points**, both in `modelroom/fit.py`, both fit v1's formula and classes with one term
replaced:

- `fit_from_size(package, vram_gib, ram_gib, machine_config, context, *, requests=1,
  shared_memory=False)` -- after the fetch: `weights_gib` is the package's own weight files as
  always, and `kv_gib = context x KV_PER_TOKEN_SIZE_BASIS x requests / 1024**3` with
  `KV_PER_TOKEN_SIZE_BASIS = 147_456`
  bytes (`2 x 36 layers x 8 KV heads x 128 wide x 2 bytes`, the numbers of a dense 9B model at
  16-bit precision). `compute_fit_v2` calls it for every package whose architecture is
  `unknown` (since AP9-UX2 a `hybrid_linear` one is computed on its architecture).
- `fit_from_parameters(parameters_b, vram_gib, ram_gib, machine_config, context, *, requests=1,
  shared_memory=False)` -- before the fetch, for the selection list of the guided mode's step 2
  (`guided_models.model_fit` hands it `[guided].requests` and the memory the ranking would read,
  `fit.profile_memory`: a graphics card measured or entered by hand, unified memory as its pool):
  `weights_gib = parameters_b x 1e9 x BYTES_PER_PARAMETER_Q4 / 1024**3` with
  `BYTES_PER_PARAMETER_Q4 = 0.6`, the size of a `Q4_K_M` build (about 4.85 bits per weight plus
  the tables). Measured against two real packages of the fixtures: 5,680,522,464 bytes for
  9,653,104,368 parameters (0.588) and 5,027,785,216 bytes for 8,190,735,360 parameters (0.614).

`need_gib`, the pool, the mode and the thresholds are fit v1's, unchanged. **The class is capped
at `fits`**: `ample` stays reserved for a package whose architecture was read. Multiplying the
KV cache by `requests` does not make this assumption any more exact.

**The KV constant is one fixed assumption, not a bound in either direction** (corrected in the
second-model round of 2026-09-24, which measured the case). A hybrid or mixture architecture keeps
fewer layers of full attention and needs less than it; a dense model with more layers needs more.
Measured on the 5.29 GiB `Q4_K_M` package of the fixtures at 131072 tokens: 24.32 GiB on this
basis, against 22.32 GiB for 32 layers, 27.32 for 42 and 30.32 for 48. At 8192 tokens the same
comparison is 7.44 against 7.32 / 7.63 / 7.82 GiB. So a package this basis calls `tight` can be
`no_fit` in truth, and one it calls `no_fit` can fit: the cap at `fits` limits the claim, it
does not make the number exact. Every view therefore says `from size` next to the class, and an
exact computation for these architectures is a work package of its own.

**What stays `unknown`**, size or no size, because there is nothing to compute with: a package
that is not a complete `gguf` build, a weight file with no size, and the GPU states the gate
refuses; a llmfit deviation is shown, never a reason (decided 2026-09-27). More than one request is computed since 2026-09-25 (see "Parallel
requests" above). For the list of step 2 there are two more of its own: `machine not measured`
and `parameter count unknown`.

**Where the basis is visible.** In the JSON view as `fit.basis`, in the terminal view as the
`(from size):` line with the ranks it is about (since 2026-09-24; the `Fit` cell says the memory pool
instead), and in the note of the row -- the note of a ranked package keeps
the sentence of its memory pool in front (`Fits into graphics memory: …`, `Too large for graphics
memory alone, …`; without it a row in system memory read as the equal of one in graphics memory,
acceptance of 2026-09-24) and ends with `From the size of the package, not its architecture.`,
with `fit.basis` added to its facts; a `no_fit` one ends with the same sentence. A row whose
speed was measured carries the measurement's note instead, so in
the Markdown view the basis of such a row is in the JSON view only; the Markdown table's own `Fit
(computed)` cell is unchanged.

**Sections per runtime (AP9-SV2-F, F5, F6).** A machine whose rows belong to both Ollama and vLLM is shown
in one section per runtime, `with Ollama` and `with vLLM`, joined by a sentence that they are
alternatives on the same cards. In the terminal view every section title has exactly one blank line
before it and carries the style class `class:label` (the muted blue of `screen.head_line`); without
color (a file, `NO_COLOR`, an answers run) the text is the same, with the blank line. The Markdown
view already has its blank line and is unchanged. A machine of one runtime has no title and no
blank line before its table.

## Render (AP5)

> **Superseded in part by "Render (schema 2)" below (AP9-C).** What still holds, unchanged: the
> rating source, the lock and the ordering, the header line and the newer-document refusal, and
> the eligibility rule. What no longer holds: the selection rule (one variant per packager), the
> no-recommendation row, the Installed and Speed cells, the Machines table's columns and the
> reading of schema-1 hardware profiles as a fit input -- the render computes a full ranking per
> machine from a schema-2 profile instead. This section is kept because the parts above are
> written out here and nowhere else.

`modelroom render --config <toml>` (`cli.render_with_config(config, rating=None, now=None)` is
the programmatic entry point, same pattern as `fetch_with_config`/`hardware_with_config`) is a
**pure reader** of the current snapshot and every configured machine's hardware profile. It
writes exactly one file, `config.paths.markdown`, and never touches `config.paths.state` beyond
the lock and the reads `load_existing_snapshot`/`load_existing_hardware_snapshot` already do.
`modelroom/render.py::build_document` is the pure function that turns an already-loaded
`Configuration`, `Snapshot` and `{machine: HardwareSnapshot | None}` mapping into the Markdown
text; `cli.py` owns the lock, the schema-version gates and the atomic write.

**Rating source.** `render.RatingSource = Callable[[str], Rating | None]`, called with a base
model's full `hf_repo`. `render.RatingUnavailableError` is for a source that cannot answer at
all; **any exception the source raises is treated the same way**: the whole Stars column
renders `–` and the document carries a note near the top, `Market rating unavailable:
<message>` (`<message>` is `str(exception)`), and the command still exits `0` -- a rating
failure never blocks the package view. `None` from the source for one base model means "no
rating for that model" (`–` for its rows only, no note). No source given (`rating=None`, the
CLI's own default -- there is no `--rating` flag) renders every Stars cell `–` with no note.
The source is called once per base model that has at least one row in the Package table, never
once per package.

**Lock and ordering.** `render` acquires the same lock `fetch` uses
(`acquire_lock(config.paths.lock_file, "render", now)`) and releases it in `finally` -- lock held
elsewhere exits `1`. Same ordering as `fetch` (F13): the snapshot's `schema_version` is checked
*before* the lock is taken; unsupported exits `3` with nothing written, not even a lock file. A
hardware file with an unsupported `schema_version` also exits `3` (checked once the lock is held,
before anything is written). **No snapshot at all exits `1` with "nothing to render, run fetch
first" *before* the lock is taken (R7-12, fix-round 6)** -- `acquire_lock` creates the state
directory and the lock file as a side effect of opening it, so a render with nothing to do must
never call it: the same pre-read that decides the schema-version gate is reused for this check,
and the snapshot is read again once the lock is held (a concurrent `fetch` could have written or
changed it between the two reads), so nothing is written and the state directory is never even
created when there was never anything to render.

**Header and the newer-document refusal.** Every rendered document starts with a fixed header
line:

```
<!-- modelroom render: snapshot_run_at=<ISO UTC> rendered_at=<ISO UTC> -->
```

`render.format_header_line`/`render.parse_header_line` write and read it. Before writing,
`cli.py` reads the existing `config.paths.markdown` (if any) and parses its first line; when the
existing document's `snapshot_run_at` is **strictly newer** than the snapshot about to be
rendered, the write is refused (exit `1`, the message names both timestamps, the file is left
untouched). Equal or older is replaced. A file with no such first line (missing, empty, or from
before `render` ever wrote one) is always replaced. **Fix-round 5, F4:** a header whose
`snapshot_run_at`/`rendered_at` is not an aware UTC datetime (`parse_header_line` -- naive, or
aware at a non-UTC offset, same rule as `contracts.check_aware_utc`) is *also* treated as no
header, not compared at all -- `format_header_line` never writes anything else, so only a
hand-edited or otherwise corrupted document could carry one, and comparing it against the new,
always-aware snapshot's `run_at` would otherwise raise `TypeError` instead of refusing cleanly.
An existing document that is not valid UTF-8 counts the same way (no header, always replaced)
rather than raising `UnicodeDecodeError` out of `render`. The write itself is atomic
(`state.atomic_write_text`, the same `.pid.tmp` + `os.replace` convention as
`atomic_write_json`) and, like every file this package writes, uses LF line endings on every
platform. `run-status.json` is never touched by `render` -- it belongs to `fetch`.

**Two configurations sharing a `paths.markdown` (R7-6, fix-round 6, decided, no guard).** One
state directory owns exactly one markdown document; the newer-document refusal above protects
against an older snapshot under the *same* state directory's lock only. Two configurations with
different `paths.state` but the same `paths.markdown` are unsupported by this contract and not
guarded by a second lock -- a lock file next to the document would land in the documentation tree
of the consuming repo, not in either configuration's own state directory. Nothing else changes;
`paths.markdown` colliding with `paths.state` itself is a configuration error (R7-1, `config.py`,
`PathsConfig`), a different case from this one.

**What is rendered.** A fixed header block (title, snapshot run time, rendered time, the number
of base models / packages -- `fetch`'s own request budget is never shown here, it belongs to
`run-status.json`), an Area status table (one row per `snapshot.areas`, always rendered even
after a partial `fetch` failure -- the document always shows the last valid snapshot), a
Machines table (one row per `config.machines`: name, VRAM/RAM GiB, the configured reserves,
backend/GPU name, profile measured-at, profile age in whole days relative to `rendered_at`, and
the installed-model count or `unknown (<installed_unavailable_reason>)`; a machine with no
hardware profile yet renders `no hardware profile yet` for every measured column), and the
Package table.

**Eligibility.** Only packages with `provenance in ("from_metadata", "approved")`, `complete ==
True` and `active == True` are ever shown.

**Grouping and ordering.** Eligible packages group by (base model, packager), packager being the
Hugging Face owner or the literal `"ollama"`. Groups are ordered by base model in configuration
order (families, then base models within a family), then packager alphabetically with `"ollama"`
always last.

**Selection rule**, per group *and per configured machine*: the package with the largest
quantization (`QUANT_ORDER` index, unknown sorting last -- same convention as
`quantization.sort_key`) whose `fit.py::compute_fit` class on that machine is `"fits"` or
`"ample"`; when none qualifies, the smallest *judged* package (`quantization.sort_key`, a real
`fit_class` other than `"unknown"`) with whatever class it gets. Ties among qualifying packages
at the same quant index break by `sort_key`, first wins. **Fix-round 5, F5:** a machine makes
**no pick at all** when it has no hardware profile, or when `compute_fit`'s class is `"unknown"`
for *every* package of the group (typically the whole architecture is not covered by fit v1 --
the real Qwen3.5 case, measured 2026-09-22: every package came back unknown, and the previous
"else the smallest" fallback still picked an arbitrary smallest-quant package such as
`UD-IQ2_XXS` with no basis at all) -- the "smallest" fallback above only ever considers *judged*
packages, never one fit v1 could not score. Since the pick can differ per machine, the Package
table renders one row per *distinct* package selected by at least one machine (a package
selected by two machines is one row) -- every row still shows a `fit (computed, v1): <machine>`
cell for *every* configured machine, computed fresh for that exact package, whether or not it
was that machine's own pick. `Variants` on a row is the group's eligible-package count minus the
number of distinct rows the group produced.

**No-recommendation row (fix-round 5, F5).** When *no* configured machine picks anything for a
group (every machine's `_select_for_machine` returned no pick), the group renders exactly **one**
row instead of the ordinary per-package rows: Base model, Stars and Packager as usual; Quant,
Format, Size GiB, Context, Installed, Speed, Provenance and Observed all render `–`; each
`fit (computed, v1): <machine>` cell renders `no recommendation: <reason>` (`compute_fit`'s own
`reason` -- the same text for every package when the cause is architecture-level, computed
against the group's `sort_key`-smallest package as a deterministic representative when it is
package-level instead -- or the literal `no profile` when that machine has no hardware profile at
all); `Variants` renders `<N> variants, none judged`, `N` being the group's full eligible-package
count. A configuration with no machines at all is unaffected (unchanged from before F5: such a
group renders no row at all, since there is nothing to have "no recommendation" *for*).

**Fit cell.** `<class> (<mode>, need <need_gib:.1f> / pool <pool_gib:.1f> GiB)` for a judged fit;
`fit_class == "unknown"` on a row that *did* get selected (a different machine picked its package;
see above) renders only `unknown: <reason>`, **never** the zeroed placeholder numbers
`compute_fit` returns for that case (CONTRACTS.md, "Fit contract v1", already forbids printing
them). A machine with no hardware profile renders `no profile` instead of calling `compute_fit`
at all -- both of those still apply to an ordinary row; the no-recommendation row above always
renders `no recommendation: <reason>`/`no recommendation: no profile` instead, never bare
`unknown: <reason>`/`no profile`.

**Installed cell**, per package per machine (measured 2026-09-22: the local Ollama daemon's
`/api/tags` digest equals the registry manifest digest for the same tag, and sibling tags such
as `9b` and `9b-q4_K_M` share one digest): for an Ollama package, `yes` when any
`hardware.installed[].digest == package.manifest_digest`, else `no`; `unknown` when that
machine's `installed` is `None` or it has no hardware profile at all; a Hugging Face package is
always `–` (a GGUF file on disk is not observed by AP4). The Package table's Installed column
joins every configured machine's cell as `<machine>: <cell>; <machine>: <cell>; ...`.

**Speed cell** (one column, not per machine): the renderer searches every configured machine's
hardware profile for a `Measurement` that is both *current* (`profile_measured_at ==
hardware.measured_at` for that machine's own profile) and *content-matching* the package
(`ollama_manifest_digest == package.manifest_digest`, or `hf_repo`/`hf_revision` equal to the
package's and `hf_file_digest` equal to one of its weight files' digests). The first match found
(machines checked in `config.machines` order) renders `<tps_mean:.1f> tps @<context>
(<machine>)`; a stale or non-matching measurement is silently not shown -- `–` when none
matches anywhere.

**Exit codes**, per this section: `0` rendered (also when the rating source failed), `1` the
lock is held, there is no snapshot to render, or the existing document was rendered from a
newer snapshot (all three write nothing), `2` the configuration is missing or invalid, `3` the
snapshot's or a hardware file's `schema_version` is unsupported.

## Schema 2: profiles, measurements, guided mode

The contracts below are what the guided mode builds on. This work package defines them, their
readers and `modelroom migrate`; nothing in `fetch`, provenance or `render` uses them yet (those
switch over in later work packages). Each subject has its own module -- `profile.py`,
`measurements.py`, `binding.py`, `catalog.py`, `relation.py`, `ranking.py`,
`guided_contracts.py` -- instead of growing `contracts.py`; the rules are the same: Pydantic v2,
`extra="forbid"`, validators for coupled fields, examples in `modelroom/examples.py` repeated
here verbatim under `### <ModelName>` and checked by `tests/test_contracts.py`.

**Vocabulary.** User-facing words are fixed: a model's age is `latest`, `legacy` or `unknown`;
a value's origin is `measured`, `entered` or `computed`; a repo owner is a `publisher`, a
`listed packager` or `other`; a value nobody could establish is `unknown`; two measurements that
must not be compared are `not comparable`.

### Hardware profile v2

One machine's hardware, `<state>/hardware/<profile_id>.json`, schema 2, since 2026-09-25
schema 3 (the same fields, the value `entered` added; see below and ADR 0001), and since
2026-09-26 schema 4 (every graphics card and the machine class; see below and ADR 0004). `profile_id` is 16
random lowercase hex characters (`new_profile_id`), never derived from the machine: it stays
the same when the machine is renamed, and two machines can never collide on it. `display_name`
is what the user reads; the machine name in the configuration points to the profile through
`[machines.<name>].profile`.

`os_fingerprint` is the first 16 hex characters of SHA-256 over the raw OS identifier (Windows
`MachineGuid`, Linux `/etc/machine-id`, macOS platform UUID) followed by the salt `modelroom`
(`os_fingerprint`); the raw identifier never leaves the machine. It tells a clone from the same
machine (see "Profile binding"). `os_fingerprint_source` names where it came from; `none` (not
readable, or an `entered` profile) and `legacy` (migrated) store the fingerprint `none`.

Every memory value says where it came from. `ram_physical_gib` is physical RAM (`os`,
`llmfit`, or `entered` by hand); `ram_limit_gib` with `ram_limit_scope` is a process or
container limit (cgroup, job object), a note only and never a fit input; `vram_gib` is
`nvidia-smi`, `sysfs` (an AMD card on Linux), `sysctl` (the unified memory of Apple Silicon),
`llmfit`, `entered` by hand, `none` (no counted GPU: `0`) or `unknown` (`None`). A graphics
adapter's own reported memory (`AdapterRAM`) is never taken as VRAM.

`gpu_state` decides whether the fit computes at all:

| `gpu_state` | Meaning | Fit |
|---|---|---|
| `none` | no counted GPU (none, none usable, or `--cpu-only`); `vram_source` `none`, `vram_gib` `0` | computes, CPU mode |
| `measured` | one GPU, VRAM measured (`> 0`) | computes |
| `entered` | schema 3: one GPU whose memory was entered by hand (`> 0`) | computes, like `measured` |
| `multi_gpu` | schema 4: two graphics cards or more, measured or entered by hand; `vram_gib` their sum | computes: one card first, then all cards ("Several graphics cards") |
| `present_unmeasured` | an adapter is present, its memory was not measured | `unknown` |
| `multi_gpu_not_covered` | more than one GPU, measured before schema 4 covered it; no path writes it any more, a file of 0.1.0 still reads | `unknown` until measured again |
| `unified_memory` | CPU and GPU share memory | computes since schema 3: one pool, RAM minus both reserves, mode `gpu` ("Fit contract v1", "Unified memory"); a profile migrated from schema 1 stays `unknown` until measured again |
| `unsupported_platform` | a platform other than Windows, Linux and macOS | `unknown` |
| `legacy_unknown` | migrated from schema 1, GPU layout not recorded | `unknown` until measured again |

**A machine entered by hand (schema 3, decided 2026-09-25, ADR 0001).** A profile whose memory
was entered (`ram_physical_source` `entered`) has `origin` `entered`, no fingerprint (source
`none`), both cross-checks `absent`, and exactly one of four shapes (the fourth since schema 4):

| Entered as | `gpu_state` | `vram_source` | `vram_gib` | `gpus` | `ram_physical_*` |
|---|---|---|---|---|---|
| its own graphics card of N GiB | `entered` | `entered` | N > 0 | one card of N, `name` `None` | `entered`, > 0 |
| several graphics cards of N GiB each | `multi_gpu` | `entered` | n x N | n cards of N, `name` `None` | `entered`, > 0 |
| no graphics card | `none` | `none` | 0 | empty | `entered`, > 0 |
| unified memory | `unified_memory` | `none` | 0 | empty | `entered`, > 0 |

The coupled rules behind it: an `entered` memory or graphics memory requires `origin` `entered`;
`vram_source` `entered` comes exactly with `gpu_state` `entered` or with `multi_gpu` of origin
`entered`, so a measured GPU never carries an entered size; the cards of a machine entered by hand
have one size (the dialog asks for one), measured cards may differ. `gpu_name` is free or `None`.
A `hardware --cpu-only` profile
(`origin` `entered`, `ram_physical_source` `os`) is no hand-entered machine and stays valid as it
is. No fit on such a machine is ever `measured`: every number there is `computed`. Such a profile
is entered by the guided mode, step 1 (`enter a machine by hand`, "Guided mode").

**The card list and the machine class (schema 4, decided 2026-09-26, ADR 0004).**

| Field | Type | Rule |
|---|---|---|
| `gpus` | `list[GpuAdapter]` | not empty exactly for `gpu_state` `measured`, `entered` and `multi_gpu`; `measured`/`entered`: one card, index 0, with the profile's `vram_gib` and `gpu_name`; `multi_gpu`: two or more, indexes `0 ... n-1`, their sizes adding up to `vram_gib` (within 0.01); every card has the profile's `vram_source`. An empty list says nothing about `vram_gib`: a profile migrated from schema 1 keeps llmfit's VRAM next to an empty list |
| `GpuAdapter.index` | `int` | `>= 0`, the place of the card among the counted cards, `0 ... n-1` in order (ADR 0011); it is the card's index in `nvidia-smi`'s list only while no card of that list is shut out -- where one is, the free graphics memory of the card of step 5 is `unknown` |
| `GpuAdapter.name` | `str \| None` | no length rule, like `gpu_name`; `None` for a card entered by hand and for a card read from schema 3 without a name; a view says "graphics card" for `None` or an empty name |
| `GpuAdapter.vram_gib` | `float` | `> 0`, MiB / 1024, two decimals |
| `GpuAdapter.vram_source` | `nvidia-smi \| sysfs \| sysctl \| llmfit \| entered` | the profile's `vram_source`: the source that counted the card (ADR 0011), `entered` by hand |
| `machine_class` | `laptop \| workstation \| server \| unknown` | `unknown` exactly when `machine_class_source` is `unknown`; changes no number of the fit |
| `machine_class_source` | `chassis \| entered \| unknown` | `chassis` requires a memory not entered by hand (measured, or `--cpu-only`); `entered` from `hardware --machine-class`, or from the guided question where the answer differs from the class the chassis reads (an answer equal to it keeps `chassis`, decided 2026-09-27) |

A measured `multi_gpu` has the `vram_source` of the source that counted its cards (`nvidia-smi`,
`sysfs` or, in the fallback, `llmfit`; ADR 0011); one entered by hand has `entered`, an
entered memory and no llmfit check, like `entered`. `binding.KnownProfile` still tells a machine
entered by hand by `origin` and `ram_physical_source`, which schema 4 does not change.

**What the card list does not know.** It lists the counted cards, from one source (ADR 0011). A
unit that is not counted -- integrated graphics, a card of another driver, a card only Vulkan can
use -- is in `adapters`, not here. `CUDA_VISIBLE_DEVICES` can hide cards from a daemon the list
still shows; MIG instances are not cards of their own here. Each of these is a stated limit, not a
measurement.

**Cross-check with llmfit.** Only like with like: physical RAM against llmfit `total_ram_gb`, the
graphics memory by the first row of this table that applies (ADR 0011, `gpu_llmfit.vram_crosscheck`):

| # | Case | Status |
|---|---|---|
| 1 | `--cpu-only` | `absent` |
| 2 | the own `vram_gib` is `None` (unmeasured, a Mac whose RAM did not read) | `absent`, or `error` when llmfit failed |
| 3 | the counted cards are llmfit's own (the fallback) | `not_comparable`, `llmfit_gib` the same value |
| 4 | counted cards of different sizes | `not_comparable`, llmfit's CUDA sum kept where it said one |
| 4a | one card or cards of one size, llmfit `absent`/`error` | `absent`/`error` |
| 5 | llmfit's entries of the counted backend hold exactly as many cards (`sum(count)`) as are counted, and no own unit of that vendor is left uncounted | the own sum against `sum(vram_gb x count)` of those entries |
| 6 | any other counted case | `not_comparable`, `llmfit_gib` = `system.gpu_vram_gb` |
| 7 | `unified_memory` with a known RAM | `vram_gib` against `system.gpu_vram_gb` |
| 8 | `none` with adapters (`vram_gib` 0) | `not_comparable`, `own_gib` 0 |
| 9 | `none` without adapters | `0` against llmfit's |

`crosscheck(own, llmfit)` is `confirmed` when
`|own - llmfit| / max(own, llmfit) <= 0.05` (both zero is `confirmed`), `deviation` otherwise,
`absent` when llmfit gave no reading, `error` when llmfit failed. Cards of different sizes are
`not_comparable` (schema 4, decided 2026-09-27): llmfit multiplies one card's size by the count,
so its sum says nothing about them; the check keeps the own sum and llmfit's where it said one,
never `absent`. Cards of one size keep the sum check. **The own reading is the authority, and
llmfit is a safety net, never a block** (decided 2026-09-27): a `deviation` is shown -- behind the
measured note of step 1 (`; llmfit reads RAM X GiB, this measurement Y GiB`, per deviating
quantity), in the machine's headline of the Markdown view and in the head of its terminal block
-- and the fit computes on the own reading. A confirmed or absent check leaves both heads as they
are; there is no column for it. `fit_block_reason(profile)` returns the reason the fit must not
compute (GPU state, a profile migrated from schema 1, unknown RAM), or `None`.

**Reading schema 1, 2 and 3.** `read_profile_document` validates schema 1 as `HardwareSnapshot`
and schema 2 to 4 as `HardwareProfile`; 5 and above raise `SchemaVersionError` (exit 3) before
any field is looked at. Two steps with a fixed target each: a schema-2 dict is read as schema 3
by `normalize_profile_v2` (the same fields, every value kept, `schema_version` 3), a schema-3
dict as schema 4 by `normalize_profile_v3` (`gpus` from the state -- one card of the profile's
`vram_gib`, `gpu_name` and `vram_source` for `measured` and `entered`, none otherwise -- and the
class `unknown`/`unknown`). A file that says schema 2 but carries a value of schema 3 (`entered`)
is refused, and so is a file that says schema 2 or 3 but carries `gpus`, `machine_class`,
`machine_class_source`, `gpu_state` `multi_gpu` or a cross-check of status `not_comparable`. The caller's dict is never changed. The same
chain (`normalize_stored_profile`) runs for the profile inside an export file
(`measurements.load_export`); the probe profile `render` builds and `normalize_profile_v1` set the
new fields themselves. Reading never changes a file; `modelroom migrate` rewrites a schema-2 or
schema-3 profile in place (see
"Migration"). `normalize_profile_v1(legacy, profile_id)` converts without guessing:
`display_name` is the old machine key, fingerprint source `legacy`, RAM and VRAM keep their
llmfit values with source `llmfit`, `gpu_state` is `unified_memory` when the old file said so and
`legacy_unknown` otherwise (a positive VRAM did not prove a single GPU, a zero did not prove
none), both cross-checks `absent`. The schema-1 `installed` list is not carried over: it is a
point-in-time observation the next measurement records again.

### Hardware measurement

`modelroom hardware --config <toml> [--machine <name>] [--cpu-only] [--machine-class
laptop|workstation|server] [--new-identity]` measures the machine it runs on and writes one
profile of the current schema (4),
`<state>/hardware/<profile_id>.json`
(`modelroom/measure.py`, `modelroom/cli.py`). modelroom measures the machine itself; `llmfit` is
a cross-check of two quantities and, where modelroom counts no card of its own, the fallback whose
cards may count with the source `llmfit` (ADR 0011) -- never a requirement: a machine without
`llmfit` is measured and written all the same. `--machine` is optional and only names the
configuration entry whose `profile` the takeover rule may adopt. The schema-1 writer
(`write_hardware_snapshot`, `<machine>.json`) stays readable for `render` until the renderer
switches; `hardware` no longer writes it, and a schema-1 file in the folder is skipped, never
adopted and never overwritten. A new `profile_id` is none of the folder's file names and none of
the `profile_id`s its files carry, whatever they are called, in any letter case -- the rule of a
machine entered by hand, one function for both (`guided_entered.fresh_profile_id`, decided
2026-09-26). The command reads no Ollama daemon: the installed packages are a
point-in-time observation the load test records (CONTRACTS.md, "MeasurementRecord"), not part of
a profile.

**Sources.** Every source is a fixed argument list or a fixed file path, run through the
injected command layer (`Runner`, 10 s timeout) or file layer -- no user-controlled string ever
reaches a command line. `AdapterRAM` and the registry's 32-bit `MemorySize` are never read: both
stop at 4 GiB. The registry's 64-bit `HardwareInformation.qwMemorySize` is a source that is shown,
never counted (ADR 0011).

| Quantity | Platform | Source | Unit | Stored as |
|---|---|---|---|---|
| physical RAM | Windows | `GlobalMemoryStatusEx().ullTotalPhys` (ctypes, `kernel32`) | bytes | `ram_physical_gib`, source `os` |
| physical RAM | Linux | `/proc/meminfo`, `MemTotal` | kB | `ram_physical_gib`, source `os` |
| physical RAM | macOS | `sysctl -n hw.memsize` | bytes | `ram_physical_gib`, source `os`; on Apple Silicon also the unified memory (`vram_gib`, source `sysctl`) |
| VRAM | Windows, Linux | `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits`, every line | MiB | the NVIDIA adapters; counted: `gpus`, `vram_gib` (their sum), source `nvidia-smi`, `gpu_name` |
| compute capability, PCI address | Windows, Linux | `nvidia-smi --query-gpu=index,compute_cap,pci.bus_id --format=csv,noheader,nounits` | version, address | below 5.0 the card is not usable, unread shuts nothing out; the address ties the card to its `lspci` line |
| AMD memory | Linux | `/sys/class/drm/card*/device/mem_info_vram_total` (and `_used` for the free memory), the card found by `PCI_SLOT_NAME` in `card*/device/uevent` | bytes | the adapter's memory, source `sysfs`; counted only with exactly one match |
| ROCm compute driver | Linux | `/sys/class/kfd/kfd/topology/nodes/*/properties`, `domain` and `location_id` (bus * 256 + device * 8 + function) | address | `usable` `yes (ROCm)` for the card at that address |
| Apple Silicon | macOS | `sysctl -n hw.optional.arm64` = `1`, else an llmfit entry `Metal` with unified memory; the name from `sysctl -n machdep.cpu.brand_string` | -- | `gpu_state` `unified_memory`, one adapter `unified` |
| graphics units | macOS (Intel) | `system_profiler SPDisplaysDataType -json` (`sppci_model`, `spdisplays_vram`) | -- | adapters, none usable |
| chassis type | Windows | `powershell -NoProfile -NonInteractive -Command "(Get-CimInstance Win32_SystemEnclosure).ChassisTypes"`, the first number | SMBIOS type | `machine_class`, source `chassis` |
| chassis type | Linux | `/sys/class/dmi/id/chassis_type` | SMBIOS type | `machine_class`, source `chassis` |
| display adapters | Linux | `lspci -nnD`, PCI class `03xx`: address, vendor and device id, name | -- | adapters; an NVIDIA line that is certainly a card of `nvidia-smi` is left to it (below) |
| WSL | Linux | `/proc/sys/kernel/osrelease` contains `microsoft` | -- | the Hyper-V id `1414` is the stand-in of the Windows card, no display adapter |
| display adapters | Windows | `Get-CimInstance Win32_VideoController \| Select-Object Name, PNPDeviceID \| ConvertTo-Json -Compress`, read with `powershell -NoProfile -NonInteractive -Command` | -- | adapters; an NVIDIA entry that is certainly a card of `nvidia-smi` is left to it (below) |
| graphics memory shown | Windows | `HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*`, `HardwareInformation.qwMemorySize` (a number, or exactly 8 bytes little-endian, at least 64 MiB), matched by `VEN_xxxx&DEV_yyyy` | bytes | the adapter's memory, source `registry`, never counted; several different values for one id are unknown |
| memory limit | Linux | cgroup v2: `/proc/self/cgroup`, then `memory.max` from this cgroup up to the root; the effective limit is the smallest numeric value | bytes | `ram_limit_gib` + `ram_limit_scope` `cgroup` -- a note, never a fit input |
| machine identifier | Windows | `reg query HKLM\SOFTWARE\Microsoft\Cryptography /v MachineGuid` | string | `os_fingerprint` (salted digest only) |
| machine identifier | Linux | `/etc/machine-id` | string | `os_fingerprint` |
| machine identifier | macOS | `ioreg -rd1 -c IOPlatformExpertDevice`, `IOPlatformUUID` | string | `os_fingerprint` |

**Failure of a source.** Every source has a defined failure: the program is not installed, it
cannot be run, it does not answer within 10 s, it exits non-zero, or its output cannot be read.
None of them raises: the reading becomes "no value" with the reason, the profile records the
source as `unknown`, and the reason is printed as a note. One unreadable source never ends the
command. A byte count that is not a real positive number (`0` from a call that reported success, a
negative value, an infinity, or a value that rounds to `0.00` GiB) is such a failure too, because
the contract's `ram_physical_gib` is `> 0`; so is a number no process can convert (a digit string
above Python's int/str conversion limit, from `/proc/meminfo`, `nvidia-smi` or `memory.max`); and
so is a blank machine identifier, because the fingerprint of an empty identifier is refused. The cgroup files are the one exception to the note
rule above: absent or unreadable means "no limit", not a failure to report -- a machine without
cgroups is the normal case, and the limit is a hint that never enters the fit anyway.

**`gpu_state`** (ADR 0011). Every unit a source names becomes an adapter (see "Graphics
adapters"). A card counts where it is usable (`yes`), a card of its own (`discrete`), and its memory
comes from the one countable source of its backend: CUDA from `nvidia-smi`, ROCm from sysfs with
one unique match. Where no card of modelroom's own counts, llmfit's `CUDA`/`ROCm` cards may count
(the fallback, llmfit alone). Of the counted backends the one with the largest sum counts, a tie
goes to CUDA; the others say `another driver than the counted cards`. The first row that applies:

| # | Observation | `gpu_state` | VRAM / note |
|---|---|---|---|
| 1 | a platform other than Windows, Linux and macOS | `unsupported_platform` | `unknown` |
| 2 | `--cpu-only` | `none` | `0`, source `none`, `adapters` empty, profile `origin` `entered`; no graphics source is asked |
| 3 | one counted card / two or more | `measured` / `multi_gpu` | the sum, the source of the counted cards; a note lists two or more (`nvidia-smi lists 2 adapters: 2 x NVIDIA A100-SXM4-40GB`); counted by llmfit, a note says so |
| 4 | Apple Silicon | `unified_memory` | the RAM, source `sysctl`; unknown RAM: `unknown`; note `macOS keeps part of the memory for the system` |
| 5 | WSL: the own list is only the Hyper-V stand-in, no counted card, no llmfit adapter | `present_unmeasured` | `unknown`, note `WSL shows the Windows card only through its driver; nvidia-smi did not answer` |
| 6 | at least one recognized adapter (own or llmfit), none counted | `none` | `0`, source `none`, a note per adapter (`no counted graphics card: ...`), with `install the NVIDIA driver tools (nvidia-smi)` where they did not answer |
| 7 | display adapters only | `none` | `0`, note `display adapter only` |
| 8 | beside the display adapters at least one unit, none recognized | `present_unmeasured` | `unknown`, note `a graphics adapter is present but nothing about it could be read` |
| 9 | no list could be read and llmfit names no adapter | `present_unmeasured` | `unknown`, note `not recognized: ` and every source that failed |
| 10 | no PCI class-03 device | `none` | `0`, source `none` |

A unit is recognized when it is no display adapter and its vendor or its name is known. An Intel
Mac is a CPU machine (`none`, every unit `not usable`). `nvidia-smi` lines that do not read, a card
without memory, or indexes other than `0 ... n-1` make `nvidia-smi` a failed source: its reason stays in
the note, and every other source and llmfit are still asked (a card of another vendor may count).
An NVIDIA entry of the platform's list is left out only where it is certainly a card `nvidia-smi`
lists: the same PCI address where the entry carries one (`lspci`; an entry at another address is
another card, whatever the counts); for an entry without an address (`Win32_VideoController`) the
same name, one entry per card that no address matched. Every other NVIDIA entry stays, not counted (`nvidia-smi does not list this card`),
and keeps the cross-check `not_comparable`. On macOS a valid `hw.optional.arm64` of `0` is an Intel
Mac whatever llmfit says; llmfit's `Metal` entry proves Apple Silicon only where that reading failed.

**The machine class (since 2026-09-26).** The SMBIOS chassis type, mapped by a positive list per
class; every other number is `unknown`:

| Class | Chassis types |
|---|---|
| `laptop` | 8 Portable, 9 Laptop, 10 Notebook, 11 Hand Held, 14 Sub Notebook, 30 Tablet, 31 Convertible, 32 Detachable |
| `workstation` | 3 Desktop, 4 Low Profile Desktop, 5 Pizza Box, 6 Mini Tower, 7 Tower, 13 All in One, 15 Space-saving, 16 Lunch Box, 24 Sealed-case PC, 34 Embedded PC, 35 Mini PC, 36 Stick PC |
| `server` | 17 Main Server Chassis, 23 Rack Mount Chassis, 25 Multi-system Chassis, 28 Blade, 29 Blade Enclosure |
| `unknown` | 1 Other, 2 Unknown, any other number, no answer, no DMI (WSL2, a guest machine without a chassis), macOS |

The chassis fails like every other source: no answer, a non-zero exit, a file that cannot be read,
or text without a number is `unknown` with a note naming why (`the machine class is unknown:
...`); macOS has no source (`the machine class is not read on darwin`). It is no GPU source, so
`--cpu-only` reads it too. `--machine-class laptop|workstation|server` is the user's own word for
the class: source `entered`, and the chassis is not read. The class changes no number of the fit.

**The display-only vendor list.** A positive list, so an unknown vendor is never declared
harmless: QEMU/Bochs `1234`, virtio `1af4`, VMware `15ad`, Red Hat `1b36`, VirtualBox `80ee`,
Microsoft Hyper-V `1414` (except on WSL, where it stands in for the Windows card), Xen `5853`,
Amazon `1d0f`; on Windows also a `PNPDeviceID` with `VEN_1414` and the `Microsoft Basic Display
Adapter`. Such a unit is recognized with or without a name and is `not usable: display adapter
only`, before any unknown unit is judged.

**Graphics adapters (ADR 0011).** `HardwareProfile.adapters`, one `GraphicsAdapter` per unit, in
the order the sources read them. Ollama's documentation of 2026-09-28 (`docs.ollama.com/gpu`) is
the yardstick of `usable`: NVIDIA from compute capability 5.0 through CUDA, AMD on Linux through
ROCm, Apple Silicon through Metal, Vulkan on Windows and Linux on by default. Exclusions come first:
a display adapter, an Intel Mac and an NVIDIA card below compute capability 5.0 are `no`; AMD on
Windows is `unconfirmed` (`ROCm on Windows needs a HIP7 driver; not checked`) whatever the source.
Then the proofs: `nvidia-smi` lists the card (`yes`, CUDA); a kfd node of this card's address
(`yes`, ROCm; without one `unconfirmed`, Vulkan); Intel is `unconfirmed` (`Ollama may use it
through Vulkan`); Apple Silicon `yes` (Metal); llmfit's `CUDA`/`ROCm` in the fallback. An own
exclusion of a vendor shuts llmfit's units of that vendor out as well, its Vulkan and SYCL entries
included. The build is read from positive marks only, else `unknown`, which never counts:

| Vendor | `discrete` | `integrated` |
|---|---|---|
| NVIDIA | always | -- |
| AMD | the name holds `RX `, `PRO `, `Radeon W`, `Instinct` or `Radeon VII` | `Radeon <3 digits>M`, or `Radeon Graphics`/`Radeon(TM) Graphics` without a mark of the first column |
| Intel | Windows and llmfit: `Arc(TM) [AB]<3 digits>` or `Arc Pro [AB]<2 digits>` | Linux: the address `0000:00:02.0`; elsewhere `UHD Graphics`, `Iris`, `HD Graphics`, exactly `Intel(R) Graphics`, or ending in `Arc(TM) Graphics` |
| Apple | -- | -- (`unified`) |

Integrated graphics never counts; no Intel memory is read on Linux (no kernel file for it could be
shown in a tagged kernel). llmfit's entries become adapters only in the fallback: `count` units each;
`Vulkan` and `SYCL` never count; on Windows a raw `vram_gb` of 4.0 or less never counts (llmfit reads
`AdapterRAM` first); a `vram_gb` of 0 is unknown memory; `Metal` only proves Apple Silicon; CPU and
NPU entries make none.

| Field | Type | Rule |
|---|---|---|
| `vendor` | `nvidia \| amd \| intel \| apple \| other` | PCI `10de`, `1002`, `8086`, `106b`; llmfit by the first word of the name |
| `name` | `str \| None` | the source's name; a view says the vendor where there is none, never `unknown` |
| `pci_id` | `str \| None` | `vvvv:dddd`, lowercase |
| `pci_slot` | `str \| None` | Linux: `dddd:bb:dd.f` |
| `memory_gib` | `float \| None` | `> 0`; `None` for integrated graphics and where nothing was read |
| `memory_kind` | `dedicated \| shared \| unified \| unknown` | -- |
| `form` | `discrete \| integrated \| unified \| unknown` | -- |
| `source` | `nvidia-smi \| sysfs \| registry \| sysctl \| system_profiler \| lspci \| windows \| llmfit` | where the memory came from, else the name |
| `evidence` | `nvidia-smi \| kfd \| sysctl \| llmfit \| None` | where the proof of the backend came from |
| `backend` | `cuda \| rocm \| metal \| vulkan \| none \| unknown` | -- |
| `usable` | `yes \| no \| unconfirmed` | -- |
| `counted` | `bool` | a counted unit is usable and has its memory |
| `reason` | `str \| None` | set exactly when `usable` is not `yes`, or the unit is not counted and not `unified` |

The list is checked only when it is not empty -- empty is valid for every profile (schema 1, 2
and 3, a schema-4 file without the field, a machine entered by hand, `--cpu-only`, an export). Not
empty: `origin` `measured` and a fingerprint source other than `legacy`; the counted units are the
cards of `gpus` in number, order, name and size, all with the profile's `vram_source`; units count
only for `measured` and `multi_gpu`; `unified_memory` is exactly one `unified` unit, with
`vram_gib` = `ram_physical_gib` and `vram_source` `sysctl`, or both unknown.

**The free graphics memory** of the card of step 5 follows the counting source: `nvidia-smi` as
before; sysfs per counted card by its address (`mem_info_vram_total - mem_info_vram_used`, all or
nothing); a card llmfit counted is `graphics unknown`.

**The cross-check and `--cpu-only`.** An entered `0` VRAM is not a reading, so it is not
compared with llmfit's: the VRAM check of a `--cpu-only` profile is `absent`, whatever llmfit
did -- `absent` means "never asked", while `error` would claim the comparison was attempted.
Comparing them on a machine that does have a card was a `deviation`, which blocked the CPU fit
`--cpu-only` exists to produce until 2026-09-27 and would be shown as one since; llmfit's number
is about hardware the user has just said not to use. The physical RAM is still measured there,
and still cross-checked.

**Notes.** Facts only, printed after the summary line and one `graphics  ` line per adapter
(`<name> · <memory> · <source> · <usable>`): the reason a reading is unknown, the GPU
note, the reason the machine identifier could not be read (with its consequence: a clone of this
machine cannot be told apart), and -- when a cgroup limit is *smaller* than the physical RAM --
that the fit computes with the physical RAM and can be too optimistic here. The llmfit state and
the reason the fit is blocked are printed the same way. Two absences are deliberately *not*
notes: a machine with no cgroup files at all (the normal case outside a container) is simply a
machine without a limit, and `--cpu-only` reports the skipped GPU and identifier once, in its own
note, instead of as three failures.

**The profile that is written** is exactly the shape of the `HardwareProfile` example below, and
a Windows machine with one NVIDIA adapter produces exactly its fields: `origin` `measured`,
`ram_physical_source` `os`, `vram_source` `nvidia-smi`, `gpu_state` `measured`, both cross-checks
`confirmed`. The raw machine identifier never reaches the file -- only `os_fingerprint`'s digest.
`display_name` of a new profile is its neutral name, `<class>-<first 8 hex of the profile_id>`
(`computer` where the class is unknown), never the host name; an existing profile keeps its own
through the sink ("Neutral names": a neutral name or the SSH alias stays, anything else becomes the
neutral name). `--cpu-only` writes an `entered` profile, which
by contract carries no fingerprint (`os_fingerprint_source` `none`).

**Which profile is written** follows the takeover rule (`resolve_profile_target`, "Profile
binding"): the home binding, else the configured `[machines.<name>].profile`, else a new
profile; `--new-identity` always writes a new one. The binding key is the folder the
configuration file sits in -- the results folder, the same key `export-profile` and
`import-profile` read the binding under; a configuration handed in as an object without a file
binds on `paths.state`. A bound or configured profile
that is missing or carries another `os_fingerprint` stops the run with exit `2` naming
`--new-identity`; nothing is written. Reading the profiles, reading the pointer file, writing the
new profile and binding it all happen under `modelroom.lock`; the measurement itself runs before
the lock (it reads the machine, not the state).

**The pointer file is read late and written as a merge.** The read happens inside the lock and as
late as the takeover rule allows: two runs that both read "no binding yet" before either wrote one
would each create a profile for the same machine. And because the pointer file is one file per
user for *every* results folder while each folder has its own lock, the write is a merge onto the
newest content, never a write-back of the copy this run read -- a binding another folder's run
added meanwhile is kept. The window between that read and that write is not covered by any lock:
the same trade-off as "atomicity, not durability", and its cost is one extra profile on the next
run, never a lost measurement. A new `profile_id` also avoids every *file name* in the folder, not
only every `profile_id`: a schema-1 file carries no `profile_id`, and a machine name may look
exactly like one.

The pointer file is read before the profile is written, so a corrupt or unreadable pointer file
stops the run at exit `3` with nothing written. The *binding*, however, happens after the profile
is written, and a pointer file that cannot be written or re-read then (a folder in the way, no
permission, a file corrupted meanwhile) does not throw the measurement away: the profile stays,
the message says that the next run may write a second profile, and the run ends `1`. A profile
that cannot be written at all -- a regular file where the `hardware` folder belongs, a full disk,
no permission -- is also exit `1`, with nothing written and nothing bound.

| Exit | `hardware` |
|---|---|
| `0` | measured, written and bound, with or without `llmfit` |
| `1` | another process holds the lock (nothing written); the profile could not be written (nothing written); or the profile was written but the binding could not be recorded |
| `2` | the configuration is missing or invalid, `--machine` names a machine that is not configured, the bound profile belongs to another machine (`--new-identity`), or the measurement did not validate as a profile (nothing written) |
| `3` | a stored profile or the pointer file has an unsupported `schema_version` or cannot be read |

### CrossCheck

One cross-check of one quantity. `confirmed`/`deviation` carry both values and must agree with
the 5 % rule for them (a stored `confirmed` with a larger gap is refused; the bound is inclusive
and compared with a float tolerance, so exactly 5 % such as 8.0 against 7.6 is `confirmed`); `absent`/`error`
carry no llmfit value; `not_comparable` (schema 4, decided 2026-09-27) requires `own_gib`, may carry
`llmfit_gib` and is not held against the 5 % rule. `own_gib` must equal the profile's own value for
that quantity.

```json
{
  "status": "confirmed",
  "own_gib": 31.7,
  "llmfit_gib": 31.9
}
```

### LlmfitCrosscheck

The two cross-checked quantities of one profile.

```json
{
  "ram_physical": {
    "status": "confirmed",
    "own_gib": 31.7,
    "llmfit_gib": 31.9
  },
  "vram": {
    "status": "confirmed",
    "own_gib": 8.0,
    "llmfit_gib": 8.0
  }
}
```

### HardwareProfile

| Field | Type | Constraint |
|---|---|---|
| `schema_version` | `int` | `4` (a schema-2 or schema-3 file is read as 4) |
| `profile_id` | `str` | 16 lowercase hex |
| `display_name` | `str` | 1 to 128 characters |
| `os_fingerprint` | `str` | 16 lowercase hex, or `none` exactly when the source is `none` or `legacy` |
| `os_fingerprint_source` | `windows_machineguid \| linux_machine_id \| macos_platform_uuid \| none \| legacy` | `entered` profiles: `none` |
| `origin` | `measured \| entered` | -- |
| `recorded_at` | UTC timestamp | timezone-aware |
| `ram_physical_gib` | `float \| None` | `> 0`; `None` exactly when the source is `unknown` |
| `ram_physical_source` | `os \| llmfit \| entered \| unknown` | `entered` requires `origin` `entered` and one of the three shapes above |
| `ram_limit_gib` | `float \| None` | `> 0`; `None` exactly when `ram_limit_scope` is `none` |
| `ram_limit_scope` | `str` | e.g. `none`, `cgroup`, `job_object` |
| `vram_gib` | `float \| None` | `>= 0`; `None` exactly when the source is `unknown`; `0` for source `none` |
| `vram_source` | `nvidia-smi \| sysfs \| sysctl \| llmfit \| entered \| none \| unknown` | `gpu_state` `none` requires `none`; `entered` exactly with `gpu_state` `entered`; `sysctl` the unified memory of Apple Silicon |
| `gpu_state` | see the table above | `measured` requires a measured VRAM above 0; `entered` an entered VRAM above 0 and an entered memory; `legacy_unknown` only with source `legacy` |
| `gpu_name` | `str \| None` | display only |
| `llmfit_crosscheck` | `LlmfitCrosscheck` | -- |
| `llmfit_version` | `str \| None` | -- |
| `adapters` | `list[GraphicsAdapter]` | see "Graphics adapters"; not written when empty |
| `runtimes` | `list["ollama" \| "vllm"]` | each once; not written when empty (AP9-SV2-0) |

```json
{
  "schema_version": 4,
  "profile_id": "3f9a0c21d4e6b870",
  "display_name": "workstation",
  "os_fingerprint": "9d2f4b6a8c0e1357",
  "os_fingerprint_source": "windows_machineguid",
  "origin": "measured",
  "recorded_at": "2026-09-23T08:00:00Z",
  "ram_physical_gib": 31.7,
  "ram_physical_source": "os",
  "ram_limit_gib": null,
  "ram_limit_scope": "none",
  "vram_gib": 8.0,
  "vram_source": "nvidia-smi",
  "gpu_state": "measured",
  "gpu_name": "Nova GPU",
  "gpus": [
    {
      "index": 0,
      "name": "Nova GPU",
      "vram_gib": 8.0,
      "vram_source": "nvidia-smi"
    }
  ],
  "machine_class": "workstation",
  "machine_class_source": "chassis",
  "llmfit_crosscheck": {
    "ram_physical": {
      "status": "confirmed",
      "own_gib": 31.7,
      "llmfit_gib": 31.9
    },
    "vram": {
      "status": "confirmed",
      "own_gib": 8.0,
      "llmfit_gib": 8.0
    }
  },
  "llmfit_version": "1.1.16"
}
```

The example above is the shape every earlier version wrote, without `adapters` (an empty list is
not written). The same machine measured by this version carries its adapters as well -- here the
counted card and, beside it, integrated graphics that is shown and not counted:

```json
{
  "adapters": [
    {
      "vendor": "nvidia",
      "name": "Nova GPU",
      "pci_id": null,
      "pci_slot": null,
      "memory_gib": 8.0,
      "memory_kind": "dedicated",
      "form": "discrete",
      "source": "nvidia-smi",
      "evidence": "nvidia-smi",
      "backend": "cuda",
      "usable": "yes",
      "counted": true,
      "reason": null
    },
    {
      "vendor": "intel",
      "name": "Intel(R) Graphics",
      "pci_id": "8086:7d67",
      "pci_slot": null,
      "memory_gib": null,
      "memory_kind": "shared",
      "form": "integrated",
      "source": "windows",
      "evidence": null,
      "backend": "vulkan",
      "usable": "unconfirmed",
      "counted": false,
      "reason": "integrated graphics shares the system memory"
    }
  ]
}
```

**`runtimes`** (AP9-SV2-0, F6's GF decision) names the local-server runtimes this profile's
machine detected serving or installed at the last measurement, same rule as `adapters`: an empty
list is not written, so every earlier file stays as it was, byte for byte. A machine with vLLM
detected carries:

```json
{
  "runtimes": ["ollama", "vllm"]
}
```

Every local measurement writes it (AP9-SV2-I, Codex r1 R1-1): `modelroom hardware` and the guided run's
measurement call `runtimes.local_runtime_keys`, which reads each row of the runtime table in table order
and keeps the ones whose state is `serving` or `installed` (`absent` and `asks_key` do not count; Ollama
`serving` where the daemon answers, `installed` where `ollama` is in the `PATH`; vLLM through `detect_vllm`:
Linux only). The probes are the run's own: its daemon, its server transports, the `room`'s platform, home,
`PATH` and file test -- never a Docker call. An empty list is not written; a machine measured before this
reads as "Ollama only" as ever. A machine reached over SSH writes the field too (AP9-SV2-R, R1): the
`profile` operation of `modelroom remote` reads the runtimes **at the server** (Ollama where its daemon
answers or `ollama` is in the `PATH`, vLLM through `detect_vllm` on the server's own platform, so a Windows or
macOS server never lists it) and puts them in the profile it sends, literals only.

`RuntimeKey` (`modelroom/profile.py`) names `"ollama"`, `"vllm"` and, since AP9-LS-E, `"lmstudio"`: the
third runtime is a new value in the same list, additive again -- never a plug-in, "Laufzeit als Eintrag"
(AP9-SV2-L builds the `RUNTIMES` table this field feeds).

### Scenario

What one ranking assumes, for every package in it: `context_requested` (the context the user
chose; `context_origin` `default` for 8192, `entered`, or `legacy` from a migrated measurement),
the KV cache type and the number of concurrent requests. The daemon's KV cache type is not
readable over its API, so `kv_type` is `f16` (Ollama's default) and `kv_type_assumed` is always
`true`; `context_origin` `default` always means 8192.
A measurement always runs one request. `requests` above 1 appears in the reverse calculation
(`Requirement`) and, since 2026-09-25, in a ranking: `render_cmd.scenario_from_config` takes it
from `[guided].requests` (see "GuidedConfig"); `Scenario` itself did not change.
`default_scenario()` is 8192, `f16` assumed, one request.

```json
{
  "context_requested": 8192,
  "context_origin": "default",
  "kv_type": "f16",
  "kv_type_assumed": true,
  "requests": 1
}
```

### PackageRef

The exact package content a measurement ran, same field rules as the schema-1 `Measurement`:
`ollama` needs `ollama_manifest_digest` and no `hf_*` field; `huggingface` needs `hf_repo`,
a 40-hex `hf_revision` and `hf_file_digest`, and no manifest digest.

```json
{
  "content_source": "huggingface",
  "ollama_manifest_digest": null,
  "hf_repo": "packager/Nova-7B-GGUF",
  "hf_revision": "0123456789abcdef0123456789abcdef01234567",
  "hf_file_digest": "sha256:cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"
}
```

### RunCounters

The raw counters of one measured `/api/generate` run, in nanoseconds as the daemon returns
them. Tokens per second is `eval_count / (eval_duration / 1e9)`.

```json
{
  "done_reason": "length",
  "eval_count": 128,
  "eval_duration": 2560000000,
  "prompt_eval_count": 42,
  "prompt_eval_duration": 90000000,
  "load_duration": 12000000
}
```

### LoadState

The machine's load read just before the measured runs. Load during the runs is not measurable
and is never claimed.

```json
{
  "gpu_utilization_percent": 3.0,
  "cpu_load": 0.4
}
```

### Measurement protocol v1

Shipped as `modelroom/protocol_v1.toml` and read by `protocol_for("v1")`:
one warm-up run, then three measured runs, each `POST /api/generate` with `raw = true` (no
template, no system prompt stored in the model), `stream = false`, `num_predict = 128`,
`temperature = 0.0`, `seed = 42`, a neutral English prompt of a few sentences, and `num_ctx`
equal to the scenario's `context_requested`. A measured run is valid when `done_reason` is
`length`, `eval_count` equals `num_predict`, `eval_duration` is above zero and
`prompt_eval_count + num_predict <= num_ctx` (`run_invalid_reason`); a measurement is valid when
it has exactly `measured_runs` runs and every one is valid (`measurement_invalid_reason`). An
invalid measurement is stored with its reason and never ranked. A change to the protocol is a
new protocol version, never an edit of version 1.

**Time limits** (`modelroom/daemon.py`): every short call to the daemon has 10 s, one
`POST /api/generate` has 600 s of its own -- a CPU machine needs minutes for 128 tokens. A limit
that is reached ends that measurement as a fault with its reason; it is never a `done_reason`
and never a stored measurement.

### Measurement protocol v2

Shipped as `modelroom/protocol_v2.toml` (`protocol_version = 2`) and read by `protocol_for("v2")`.
It differs from version 1 in one request field: `raw = false`. The daemon applies the template of
the model's Modelfile, and the prompt is the user turn. A system text stored in the Modelfile
goes into the prompt as well -- the package is measured the way a user runs it, so the share of the
prompt differs per package (template and system tokens) while the 128 decoded tokens are the same.
Everything else is version 1: `stream = false`, `num_predict = 128`, `temperature = 0.0`,
`seed = 42`, the same prompt text, `num_ctx` equal to the scenario's `context_requested`, one
warm-up run and three measured runs, and the same validity rule (`done_reason` `length`,
`eval_count` equal to `num_predict`, `eval_duration` above zero, the context reserve). There is no
`think` field: a model that cannot think refuses it, thinking tokens are decoded tokens and sit in
`eval_count`, and `thinking` in the answer is not read.

`Protocol.protocol_version` is `1` or `2`, and `options.raw` is `true` exactly for version 1 (a
file that says otherwise is refused with `protocol version <k> ships raw = <v>`).
`protocol_for(version)` returns the shipped protocol of `"v1"` or `"v2"` (each loaded once; any
other word is a `ValueError`, `no measurement protocol <version>`); `load_protocol(path)` has no
default path. `shipped_protocol()` is `protocol_for("v2")` and stays the name the writer calls: a
new protocol version is a new writer, there is no second function name and no choice for the
caller. Version 1 stays in the package for the records it wrote.

### MeasurementRecord

One speed measurement of one package on one profile, schema 2, never changed once written.

- `measurement_id` is `<measured_at as YYYYMMDDTHHMMSSZ>-<8 hex>` and must agree with
  `measured_at`; `new_measurement_id` draws the 8 hex characters at random.
- `protocol` is `v1` or `v2` for a load test after that protocol version, `none` for a measurement
  migrated from a schema-1 profile. `none` requires `validity` `unchecked`, no runs and
  `comparable = false`. `v1` and `v2` are `valid` or `invalid`; `valid` is checked against the
  protocol rule of the file of its own version (`protocol_for(record.protocol)`) and requires the
  speeds. The rules for `v1` and `v2` are the same; a stored `v1` record is never rewritten.
- `tps_mean`, `tps_min`, `tps_max` are all set or all `None`, `tps_mean > 0`,
  `tps_min <= tps_max`. With protocol `v1` or `v2`, `0 < tps_min <= tps_mean <= tps_max` and they are the
  mean, minimum and maximum over `runs`, so a `v1` or `v2` record without runs carries none. A migrated schema-1 range is carried over unchanged
  (schema 1 only required `low <= high`, so `0.0` or a range that misses the mean stays as it
  was).
- `comparable` is decided by the load test (daemon digest and `context_length` equal to the
  package and the scenario at every observation). `validity_reason` is set exactly when the
  measurement is not `valid`, `comparable_reason` exactly when it is not comparable.
- `scenario.requests` is always `1`.

```json
{
  "schema_version": 2,
  "measurement_id": "20260923T083000Z-5c1e9a07",
  "profile_id": "3f9a0c21d4e6b870",
  "protocol": "v1",
  "measured_at": "2026-09-23T08:30:00Z",
  "package": {
    "content_source": "ollama",
    "ollama_manifest_digest": "sha256:abababababababababababababababababababababababababababababababab",
    "hf_repo": null,
    "hf_revision": null,
    "hf_file_digest": null
  },
  "ollama_name": "nova:7b",
  "daemon_version": "0.12.3",
  "scenario": {
    "context_requested": 8192,
    "context_origin": "default",
    "kv_type": "f16",
    "kv_type_assumed": true,
    "requests": 1
  },
  "runs": [
    {
      "done_reason": "length",
      "eval_count": 128,
      "eval_duration": 2560000000,
      "prompt_eval_count": 42,
      "prompt_eval_duration": 90000000,
      "load_duration": 12000000
    },
    {
      "done_reason": "length",
      "eval_count": 128,
      "eval_duration": 2500000000,
      "prompt_eval_count": 42,
      "prompt_eval_duration": 90000000,
      "load_duration": 12000000
    },
    {
      "done_reason": "length",
      "eval_count": 128,
      "eval_duration": 2560000000,
      "prompt_eval_count": 42,
      "prompt_eval_duration": 90000000,
      "load_duration": 12000000
    }
  ],
  "tps_mean": 50.4,
  "tps_min": 50.0,
  "tps_max": 51.2,
  "validity": "valid",
  "validity_reason": null,
  "comparable": true,
  "comparable_reason": null,
  "load_state": {
    "gpu_utilization_percent": 3.0,
    "cpu_load": 0.4
  }
}
```

### Measurement files and import rule

A measurement is its own file, `<state>/measurements/<profile_id>/<measurement_id>.json`.
`write_measurement(state_dir, record, lock)` requires the held `modelroom.lock` of that state
folder, checks under the lock that no file with this id exists (`MeasurementExistsError`
otherwise) and writes with `atomic_write_json`; a measurement is never overwritten.
Every record reaches its file through the sink ("Neutral names"): `ollama_name`,
`comparable_reason`, `validity_reason`, `daemon_version` and the `done_reason` of its runs only in
the forms it allows. The one change to a written file is the sweep's: a measurement file whose
fields the sink changes is written anew, whole and atomically, under the same `measurement_id`.
`read_measurements` returns every readable record of one profile, oldest id first, and lists a
file that is not valid JSON, fails validation, or whose name or folder disagree with its own
`measurement_id`/`profile_id` as unreadable instead of loading it; the other files are still
read (a run counter too large to compute a speed from is a validation failure, too, and so is
an integer literal too long for Python to convert).

Import rule (`classify_measurement_import`, `plan_measurement_import`): an unknown id is `new`;
the same id with the same content is `idempotent` (nothing to do); the same id with other
content is a `conflict`, listed and never imported.

### ExportObject

What one machine hands to another: its profile and all of its measurements, schema 1.
`measurement_id` is unique within the export and every measurement belongs to the exported
profile. `load_export` checks `schema_version` first (`SchemaVersionError`, exit 3). The export
stays schema 1 when the profile schema moves: the profile inside it is read like a profile file,
schema 2 and 3 as schema 4 and 5 and above refused (decided 2026-09-25 and 2026-09-26), so an
export file an earlier version wrote still imports.

```json
{
  "schema_version": 1,
  "profile": {
    "schema_version": 4,
    "profile_id": "3f9a0c21d4e6b870",
    "display_name": "workstation",
    "os_fingerprint": "9d2f4b6a8c0e1357",
    "os_fingerprint_source": "windows_machineguid",
    "origin": "measured",
    "recorded_at": "2026-09-23T08:00:00Z",
    "ram_physical_gib": 31.7,
    "ram_physical_source": "os",
    "ram_limit_gib": null,
    "ram_limit_scope": "none",
    "vram_gib": 8.0,
    "vram_source": "nvidia-smi",
    "gpu_state": "measured",
    "gpu_name": "Nova GPU",
    "gpus": [
      {
        "index": 0,
        "name": "Nova GPU",
        "vram_gib": 8.0,
        "vram_source": "nvidia-smi"
      }
    ],
    "machine_class": "workstation",
    "machine_class_source": "chassis",
    "llmfit_crosscheck": {
      "ram_physical": {
        "status": "confirmed",
        "own_gib": 31.7,
        "llmfit_gib": 31.9
      },
      "vram": {
        "status": "confirmed",
        "own_gib": 8.0,
        "llmfit_gib": 8.0
      }
    },
    "llmfit_version": "1.1.16"
  },
  "measurements": [
    {
      "schema_version": 2,
      "measurement_id": "20260923T083000Z-5c1e9a07",
      "profile_id": "3f9a0c21d4e6b870",
      "protocol": "v1",
      "measured_at": "2026-09-23T08:30:00Z",
      "package": {
        "content_source": "ollama",
        "ollama_manifest_digest": "sha256:abababababababababababababababababababababababababababababababab",
        "hf_repo": null,
        "hf_revision": null,
        "hf_file_digest": null
      },
      "ollama_name": "nova:7b",
      "daemon_version": "0.12.3",
      "scenario": {
        "context_requested": 8192,
        "context_origin": "default",
        "kv_type": "f16",
        "kv_type_assumed": true,
        "requests": 1
      },
      "runs": [
        {
          "done_reason": "length",
          "eval_count": 128,
          "eval_duration": 2560000000,
          "prompt_eval_count": 42,
          "prompt_eval_duration": 90000000,
          "load_duration": 12000000
        },
        {
          "done_reason": "length",
          "eval_count": 128,
          "eval_duration": 2500000000,
          "prompt_eval_count": 42,
          "prompt_eval_duration": 90000000,
          "load_duration": 12000000
        },
        {
          "done_reason": "length",
          "eval_count": 128,
          "eval_duration": 2560000000,
          "prompt_eval_count": 42,
          "prompt_eval_duration": 90000000,
          "load_duration": 12000000
        }
      ],
      "tps_mean": 50.4,
      "tps_min": 50.0,
      "tps_max": 51.2,
      "validity": "valid",
      "validity_reason": null,
      "comparable": true,
      "comparable_reason": null,
      "load_state": {
        "gpu_utilization_percent": 3.0,
        "cpu_load": 0.4
      }
    }
  ]
}
```

### Profile binding

Which profile is "this machine" in a results folder is recorded outside that folder, in the
per-user pointer file `~/.modelroom/guided.json` (`default_pointer_path`): `current` is the
results folder the guided mode used last, `bindings` maps each results folder to this machine's
`profile_id` there. A missing pointer file reads as empty; a broken one raises
`PointerFileError`, an unsupported version `SchemaVersionError`.

`resolve_profile_target(home_binding, config_profile, profiles, local_fingerprint,
fresh_profile_id, new_identity=False, same_machine=False)` is the takeover rule, without I/O:

1. `new_identity` (`hardware --new-identity`, or "a clone" in the guided mode): `new`.
2. The home binding for this results folder: `bound`.
3. Else the selected configuration machine's `profile`: `adopt_config` (it becomes the binding).
4. Else: `new` (a new profile, bound and written to the configuration).

A bound or configured profile that is missing from the results folder, or whose
`os_fingerprint` differs from this machine's, is `ask_clone`: the guided mode asks "same machine
or a clone?", automation stops and names `--same-machine` and `--new-identity`. A fingerprint
`none` on either side never differs (a migrated profile, or a machine whose identifier is
unreadable).

**`same_machine` is the way out of that question** (`hardware --same-machine`, or "the same
machine" in the guided mode; decided 2026-09-24, after a hand test in which the honest answer
"the same machine" left the machine unmeasured and every step after it empty). Where the rule
would ask, it answers `rewrite` with the **id the binding already names** and the reason it was
reached (`... is missing in the results folder; measured again under the same id`, or `... has
another os_fingerprint; measured again under the same id, the answer was that it is the same
machine`). `hardware` then writes the profile under that `profile_id` -- `display_name` the
existing profile's through the sink, else its neutral name -- binds it (the binding does not change, it
is the same id) and the guided mode writes `[machines.<name>].profile` as it does after `new`.
The measurements under `measurements/<profile_id>/` stay what they are: a measurement belongs to
an id, and whether it still counts for a ranking is decided, as ever, by the comparison of
protocol and scenario. Where the rule does not ask, `same_machine` changes nothing. The two
switches are two answers to one question: together they are exit `2` with a sentence naming
both, and `resolve_profile_target` raises `ValueError` for them.

**A machine entered by hand is never adopted** (decided 2026-09-25, ADR 0001). The rule sees a
stored profile as `KnownProfile(profile_id, os_fingerprint, origin, ram_physical_source)`; all
three places that build one (`cli.py`, `guided.py`, `guided_loadtest.py`) pass all four. A bound
or configured profile with `origin` **and** `ram_physical_source` `entered` has the fingerprint
`none`, which agrees with every machine -- so the rule asks for it first, before the fingerprint
and also with `same_machine`: the answer is `new` with a fresh id and the reason `configured
profile was entered by hand; this machine is measured as its own profile` (`bound profile ...`
for the home binding). The entered file stays as it is; a machine measured later is a profile of
its own, never merged into the entered one. A `hardware --cpu-only` profile (`origin` `entered`,
`ram_physical_source` `os`) keeps its identity, as before.

### GuidedPointer

Keys of `bindings` and `current` are absolute folder paths; values are `profile_id`s.

```json
{
  "schema_version": 1,
  "current": "//models/modelroom",
  "bindings": {
    "//models/modelroom": "3f9a0c21d4e6b870"
  }
}
```

### Remote machines (`~/.modelroom/remote.json`)

A file of the user account next to the pointer file (`remote_store.remote_path_for`), new with
AP9-FR, schema 1, read range `[1, 2)`. It never lies in a results folder and no command exports,
renders or copies it. `this_machine` is this computer's key in `[machines.<name>]`:
`computer-<8 random hex>` (`secrets.token_hex`), made on first need and kept, the same in every
results folder this user runs on this computer, and not derivable from the host name. `ssh` maps
a `profile_id` to the host alias it was read over SSH under (written by the SSH way, FR-B1); an
alias belongs to one `profile_id` at most. `profile_share` is the shared folder of "Shared folder"
below (FR-S, GF question 7 (c)): stored exactly as it was typed, after `expanduser`, never
`resolve`d -- a UNC path stays a UNC path, so a Windows drive letter (which only exists for the
signed-in session) is never substituted for it. Every write reads the file, changes one field and
writes it back atomically. The first `this_machine` is made under a lock of the user account
(`remote.lock` next to the file, the kernel lock of "Lock file"), the file read again under it: of
two processes that start at once the second waits (10 s at most) and takes the first one's key. A
missing file reads as empty; a broken one (not JSON, a field of its own, a `this_machine` of another
shape, an alias outside the alias rule, a `profile_share` outside its own form, schema 2) and one
that cannot be written are `RemoteStoreError`, exit `3` for every command, said in a fixed sentence
that names neither a path of the user account nor a value of the file.

```json
{
  "schema_version": 1,
  "this_machine": "computer-5c1e9a07",
  "ssh": {"3f9a0c21d4e6b870": "gpu1"},
  "profile_share": "Z:\\profiles"
}
```

### Neutral names

**No file of modelroom carries a machine name** (AP9-FR, FR-N: nothing about the machines of whoever
runs it in what is passed on). The invariant covers everything that can be passed on:
the results folder (`modelroom.toml` and its backups, `docs/`, `state/` with profiles, records,
snapshot, search log, run status), the export file and the shared folder. `~/.modelroom` is the
user's own state and never leaves the computer: `guided.json` keeps the folders the user chose, as
they are, and nothing of it reaches a file of a results folder, an export or a render; from
`remote.json` only `this_machine` and the aliases reach a results folder, as keys and names.

**One sink.** `modelroom/neutral.py` alone decides what of a name and of foreign text may reach a
file; every place that writes or exports hands its object through it first. The rules, complete --
everything else is dropped:

| Field | Allowed | Otherwise |
|---|---|---|
| machine key `[machines.<k>]` | `NEUTRAL_RE` (`^(laptop\|workstation\|server\|computer)-[0-9a-f]{8}$`), or a host alias at the row of the profile `remote.json` gives it to | this computer's row (the one of the profile this folder binds it to, else its 0.1.0 key formed from the live host name): `this_machine`, and `this_machine` at another row yields to it; with a profile: `neutral_name(profile)`; without one, or where that name is taken: `computer-<8 random hex>` |
| `display_name` (profile, export, the document's `label`) | `NEUTRAL_RE`, or the host alias of that profile | `neutral_name(profile)` = `<class>-<first 8 hex of the profile_id>`, `computer` for the class `unknown` |
| `MeasurementRecord.ollama_name` | the name built from the known package (the load test's; else the one of the folder's own snapshot the record's digest names); without one only `hf.co/<hf_repo>:<tag>` of the record's own repository | `null` |
| `comparable_reason` | `another model was loaded during the measurement`, `no observation of the measured model`, `no observation of /api/ps was made`, `the digest of the measured model changed during the measurement`, `the daemon loaded context <n>, not the ranking's <m>`, or the legacy reason | mapped by the beginning of the local text; else `the measurement is not comparable (reason not kept)` |
| `validity_reason` | the sentences `measurement_invalid_reason` forms (numbers, and Ollama's own done reasons), or the legacy reason | `the measurement is not valid (reason not kept)` |
| `daemon_version` | exactly `X.Y.Z` | `null`: any addition (`-rc1`, `+<address>`) drops the whole value |
| `runs[].done_reason` | `stop`, `length`, `load`, `unload`, `unknown` | `unknown` |

The writers: `hardware` and the guided measurement (the profile), a machine entered by hand, the
import (profile, records, configuration entry), `migrate` (profile and records of schema 1),
`store_measurement` (the load test, with its package), `export-profile` (profile and records), and
every write of the configuration. A load test shows its candidates, its progress and its result by
the name built from the package as well; the name the daemon holds it under is used for the calls
to the daemon alone. `render` builds its document from the files the sweep has left neutral.

**The whole configuration** (`modelroom/neutral_config.py`). Besides its keys, a configuration may
name its folder: `paths.state`, `paths.markdown` and `[guided].results` are written relative to the
configuration's own folder where they lie inside it (`state`, `docs/models.md`, `.`), since a
folder's path can carry a user's or a machine's name; a path outside the folder but below the user's
home folder is written as `~/<the rest>` (`load_config` expands `~`). **Limits, by decision
(2026-09-29):** a path outside both is the user's own choice and stays as written; the rest of a path
below the home folder stays as well; the pointer file `~/.modelroom/guided.json` keeps the full path
of each results folder -- it lives in the user's own account and is no file of the folder (FR-N scope,
the GF decides). The guided mode puts every configuration it writes -- the one read last, with what
another run wrote meanwhile -- through the key rule immediately before the write.

**The sweep.** `neutral_sweep.sweep` runs the same rules over the files a folder already holds:
the profiles (only `display_name` changes, in the schema they are stored in); the configuration --
each renamed key replaced where it is the key of a machine row (a table head, a dotted key, an inline
table; bare, quoted or escaped, also a key spelled like a field of the file) and where it stands in a
comment (the same word, in any case, the comment otherwise kept), and each path in its neutral form,
so every other byte stays; which place is which, TOML decides: the place is replaced by a probe word
and the file read again. Only a file that cannot be brought to the expected table that way is written
anew by the TOML writer. A host name that was no key of the file, or a comment that spells a key
otherwise (split, in other characters), is not found; every measurement record, field by field and whether or
not the rest of it validates (written anew under its own `measurement_id`); modelroom's own backups
(`modelroom.toml.bak`, `modelroom.toml.vN.bak`, `<profile_id>.json.vN.bak`: their keys, the key of a
row the configuration is renamed from taking the same new name, and `display_name`); a schema-1
profile backup `<old key>.json.v1.bak` of an earlier migration takes the name of the profile it became
(`<profile_id>.json.v1.bak`, found by content), else a name of its content (`legacy-<8 hex>.json.v1.bak`;
only that form counts as neutral), and a neutral key, its runtime text and the names of installed
models dropped. Every target is reserved in the plan before anything is written: a target that holds
(or will hold) the same cleaned content already -- a second copy, or a run that stopped between
writing the target and removing the source -- takes the source's place, which is removed; one of
other content sends the source on to its content name, and where that is taken too, the source
stays with a note. And the result documents `docs/models.md` and its JSON view: made from the files
as they were, both are removed where the sweep changes anything else, or where either of them names a
machine by anything but a neutral name or an alias of this user -- the JSON view by its machine
blocks, the Markdown view by its machine headings (`## <what>: <machine>`) and the first two columns
of its Machines table (`the result documents were made from names this sweep changed;
they were removed, the next render writes them`). The sweep makes no backup of its own. A backup
that cannot be read is a note naming the file (`a backup of modelroom could not be cleaned of machine
names: <file>; delete it by hand`). Each renamed machine is a note with its new name alone (`machine
renamed to server-3f9a0c21 (no host names in files)`). A schema-1 configuration and schema-1 profiles
are `migrate`'s: their keys name the profile files, and `migrate` plans the new configuration, the
profiles, the records, both backups and the removal of the result documents through the sink before
it writes any of them ("Migration to schema 2"). Key, profile name and records are checked independently, so a run that stopped halfway
converges, and a second sweep changes nothing.

**The first step of every writing command:** the guided mode (once the configuration is found or
made), `hardware`, `fetch`, `render`, `migrate`, `import-profile` (under its own lock) and
`export-profile` (the last two under their own lock, for the whole command). Otherwise the folder is
read without the lock first; only when something is to change is `modelroom.lock` taken, the folder
read again, checked to be the folder of that lock (a `[paths]` that moved meanwhile is refused), and
the files written. A `--machine <old>` of the same call is led to the new key (`note: --machine now
names <new>`); a later call with the old name meets the unknown-machine message of that command. A
profile or a record that does not read as a JSON object of UTF-8 (it may hold a name as well as any
other file), a record whose package reference is of another shape (it cannot be judged), a profile
that carries a name but no id to form its neutral one, a write that fails and
a lock another process holds end the command with exit `1` and `the files of this folder could not
be cleaned of machine names; nothing else was written`, before it writes anything else. For a
broken profile or record this is a deliberate change of 0.1.0, which listed it as broken and went on
(decided 2026-09-29): such a file may hold a name, and a command that went on would leave it there.
Only a backup that cannot be cleaned is a note instead. This computer's key of 0.1.0 is formed from the live
host name exactly as 0.1.0 formed it (every character that is not alphanumeric one `-`, `node  01`
is `node--01`); a host name that gave 0.1.0 no key gives none here.

**The screen.** This computer's host name stands only in the start line (`machine <host name>`, read
live). In guided-run screen output, a line that names a machine uses the `display_name` of its profile,
or `neutral_name(profile_id, machine_class)` where the profile has none (AP9-FR-N2, extended to every
line by AP9-FR-N3, the instruction line of the context scale included); a machine whose profile cannot
be resolved keeps its key. Contextual references (`this computer`) and the class-only answer of the
machine list (`this machine (laptop)`) are unchanged, and the answer line of the name typed for a
machine entered by hand says `not kept: the machine gets a neutral name`. Outside guided-run screen
output, keys and names follow the files: this changes no file output, and the Markdown ranking headings
(`## Ranking: <key>`) retain their machine keys.

### The remote protocol v1 (AP9-FR, FR-B1)

The wire between a laptop and `modelroom remote <op>` on another computer, carried over the system
`ssh` (`modelroom/ssh.py`), never a shell string: a fixed argument vector (`-T -o BatchMode=yes -o
StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4
-o ForwardAgent=no -o ForwardX11=no -o ClearAllForwardings=yes -o PermitLocalCommand=no -o
RemoteCommand=none -o RequestTTY=no -o LogLevel=ERROR -- <alias> <program> remote <op>`), the
program tried as `modelroom` first and, on exit 127 with no protocol line, once more as
`~/.local/bin/modelroom`. All request data travels as **one JSON object on stdin**, then stdin is
closed (EOF); nothing about the request ever reaches argv. The counterpart (`modelroom/
remote_server.py`) reads at most `MAX_INPUT_BYTES`, 1 MiB, of it -- the same bound for every op this
module answers (`hello`, `profile`, `room`, `daemon`, and `candidates`/`measure`, FR-B2, "Load test
on the server" below; a `candidates`/`measure` request carries the run's own packages, so a very
large snapshot can reach this bound before it reaches Ollama's own limits; `vllm` and `vllm-measure`,
AP9-SV2-R, read up to 16 MiB, `remote_server.input_limit`, for their requests carry packages too, though
only as short references) -- writes JSON lines to
stdout and nothing to the disk of the machine it runs on (Ollama's own pull writes and the load
test's own generate calls excepted), and exits `0` whether or not the call itself succeeded -- a
fault is an `error` line, never a process exit code; exit codes 127/2/255 are read by the caller
only for the pre-checks (program missing, too old, connection/host-key/login) before any line is
trusted.

**The envelope.** Every line is one JSON object, `{"protocol": 1, <exactly one of "progress",
"result", "error">}` -- exactly these two keys, `protocol` the JSON integer `1` (not `true`, not
`1.0`, not `"1"`), no key twice. `progress`: zero or more, before the end. `result`: the last line
of a successful call. `error`: the last line of a failed one. Exactly one of `result`/`error` ends
the answer. The laptop reads stdout **to EOF** within the time limits below and checks every line
the moment it arrives (`ssh._CallState`): a line of another shape, a second terminal line, or any
line after the terminal one (a `progress`, a `result`, an `error`, or text) is a form fault
(`ssh.SshError`, `<alias> answered in a form this modelroom does not read`); a `progress` line is
handed on at once, never collected. On `hello` alone, an envelope whose `protocol` is an integer
other than `1` gets its own sentence naming the number (`modelroom on <alias> speaks protocol <n>;
this one speaks 1; update the older one`).

**Error codes.** An `error` line carries one of eight codes (the last three are AP9-SV2-R's), never
free text of the server; any other value is a form fault. Each has one fixed sentence at the laptop
(`ssh.ERROR_SENTENCES`):

| code | the server sends it when | sentence at the laptop |
|---|---|---|
| `bad_request` | the request is no JSON object, or fails the op's strict request model (`remote_protocol.REQUEST_MODELS`), or the op is unknown | `modelroom on <alias> could not read the request of this one; update the older one` |
| `call_refused` | a `daemon` call outside `daemon.ALLOWED_CALLS`, or a pull name outside `pull_choice.PULL_NAME_RE` | `modelroom on <alias> refused the call to its Ollama` |
| `ollama_unreachable` | the server's own Ollama does not answer a `daemon`, `candidates` or `measure` call | `Ollama does not answer on <alias>` |
| `too_large` | the request is over `MAX_INPUT_BYTES` (1 MiB) | `the request to <alias> was larger than modelroom there reads` |
| `internal` | any other fault while the server answers (its text is dropped) | `modelroom on <alias> failed on its own side before it could answer` |
| `vllm_unreachable` | `vllm-measure`: the server's own vLLM does not answer on `127.0.0.1:8000` | `vLLM does not answer on <alias>` |
| `vllm_asks_key` | `vllm-measure`: the server's vLLM asks for an API key (a `401`) | `the vLLM server on <alias> asks for an API key; modelroom reads an open server only` |
| `other_model` | `vllm-measure`: the server serves another model than the package asked about, or no single model | `vLLM on <alias> serves another model now` |

**How a call ends without a terminal line.** Once stdout has ended, the laptop waits for the
process to exit within what is left of the time limit and reads stderr to its end, so the exit
code and stderr are the real ones: `127` with no protocol line tries `~/.local/bin/modelroom` once,
and then `modelroom is not installed on <alias>, or not on the PATH of an SSH command (see
README)`; `2` with no protocol line `modelroom on <alias> is older than 0.2.0; update it there`;
`255` whose stderr names an unknown or changed host key or a refused login, its own sentence
(stderr classified by pattern, then dropped). These clear start faults keep their sentence for a
pull too. Every other end of a pull without its final line -- with or without a `progress` line
before it, since Ollama may have taken the pull on before its first line -- is `the connection to
<alias> ended; Ollama there keeps what arrived, the same pull resumes`. That sentence is the *reason*
the unchanged `_pull` of the guided mode puts behind its own `the pull of <name> did not finish: `, so
the whole line on the screen says "did not finish" once: `the pull of qwen3.8:27b-mtp-q8_0 did not
finish: the connection to gpu1 ended; Ollama there keeps what arrived, the same pull resumes`
(AP9-FR-N2, d; the other reasons, `Ollama on <alias> answered with status <n>` and the start faults,
carry no "did not finish" of their own and stay). Outside a
pull, any other `255` is `<alias> could not be reached over SSH; try ssh <alias> yourself to see
why`, anything else `the connection to <alias> ended before the answer`.

**Ops (FR-B1: `hello`, `profile`, `room`, `daemon`; FR-B2: `candidates`, `measure`, "Load test on the
server" below; AP9-SV2-R: `vllm`, `vllm-measure`, "vLLM at a server over SSH" below):**

| op | request | `result` |
|---|---|---|
| `hello` | `{}` | `{"version": "X.Y.Z"\|null, "system": "linux"\|"win32"\|"darwin", "ollama": "reachable"\|"unreachable", "agent_forwarded": bool, "os_fingerprint": "<salted>"\|"none", "vllm": "serving"\|"installed"\|"absent"\|"asks_key"}` -- `vllm` optional, `absent` where an older server sends none; `version` is the modelroom version of the server (`modelroom.__version__`, the one version source), read by no rule of the laptop: `protocol` alone decides whether the laptop goes on, so a server of another version with protocol 1 is used as any other (AP9-FR-N2, b) |
| `profile` | `{"profile_id": "<16 hex>", "machine_class": null\|"laptop"\|"workstation"\|"server"}` | `{"export": <ExportObject v1>}`, `display_name` the placeholder `"remote"`, `measurements` empty, `Probes.hostname` never called |
| `room` | `{"cards": [[index, name], ...]}` | `{"disk": {"known": bool, "free_bytes": int\|null, "unclear": bool}, "graphics_free_mib": [..]\|null, "model_disk": {"known": bool, "free_bytes": int\|null, "same_as_ollama": bool, "relative": bool}}` -- no drive, no path; `model_disk` (vLLM's place) optional, not there on an older server |
| `daemon` | `{"method", "path", "body"}` | short calls: `{"status": int, "body": <JSON>\|null}`; `POST /api/pull` also emits `progress` lines and ends with `{"status": int, "done": bool}` |
| `candidates` | `{"packages": [<Package>, ...]}` | `{"candidates": [{"package_identity": [source, repo\|name, stem\|tag]}, ...], "set_aside": int}` |
| `measure` | `{"package_identity": [...], "packages": [<Package>, ...], "scenario": <Scenario>, "profile_id": "<16 hex>"}` | `progress` `{"stage": "warmup"\|"run", "number": int, "of": int}`, `result` `{"record": <MeasurementRecord>}` |
| `vllm` | `{"packages": [<PackageRef>, ...]}` | `{"state": "serving"\|"installed"\|"absent"\|"asks_key", "served": null\|{"package": <index>\|null, "context": int}, "missing_bytes": [int\|null, ...]}` |
| `vllm-measure` | `{"packages": [<PackageRef>, ...], "package": <index>, "scenario": {"context_requested": int, "requests": int}}` | `progress` `{"stage": "warmup"\|"batch", "number": int, "of": int}`, `result` `{"measurement": {"requests", "ranking_requests", "context", "total_tps", "total_min", "total_max", "request_tps", "request_min", "request_max", "problem": <code>\|null}}` |

`daemon` is checked against `daemon.ALLOWED_CALLS` on the server, a pull name additionally against
`pull_choice.PULL_NAME_RE` (`call_refused` for either); a pull's Ollama status line is kept only where it is one of
Ollama's own stages (`pulling manifest`, `pulling <64 hex>`, `verifying sha256 digest`, `writing
manifest`, `success`), else `working`, and an Ollama `error` line becomes `{"error": true}` --
Ollama's own text never crosses the wire. `hello.agent_forwarded` (`SSH_AUTH_SOCK` set in the
server's own session) stops the run at the laptop before any other op runs for that alias.

**`SshDaemon`** (`modelroom/remote_client.py`) makes a computer read over SSH answer `daemon.py`'s
`Daemon` protocol unchanged: a short call's `result` becomes a `Response`; a pull's `progress`
lines become the same shape `guided_install._check_line` already reads (`{"error": true}` becomes
the fixed string `"Ollama on <alias> reported an error"`); `done: true` becomes a `{"status":
"success"}` body, otherwise an empty one (`_end_reason` then reads "before success", as it does
for `LocalDaemon`). A pull whose `result` status is not `200` ends with `Ollama on <alias> answered
with status <n>`, a pull that carried an error line and did not finish with `Ollama on <alias>
reported an error; run ollama pull <name> there to see it` (`<name>` the laptop's own checked
name) -- both raised as `DaemonError`, the reason the unchanged `_pull` names. Nothing here filters `/api/tags`: the unchanged digest rule of `loadtest.py`
decides what is shown, exactly as it does against `LocalDaemon`.

**The process** (`modelroom/ssh_process.py`): one reader thread per output pipe, one writer thread
for stdin, the caller's thread the only consumer; the stdout reader hands bounded lines over through
a small bounded queue whose hand-over gives up at the stop signal. The `ssh` process and every
process it starts form one tree (`modelroom/process_tree.py`). On Windows `ssh` is created suspended,
put into a job object of its own and only then resumed, so it belongs to the job before it runs its
first instruction and every helper it starts belongs to it too; where the job cannot be created or
assigned, or the resume fails, the suspended process is ended before it ran and the call ends fail
closed: `modelroom could not start ssh for <alias> under its own control on this computer; nothing
was sent to <alias>`; an exception or Ctrl-C during that setup likewise ends and reaps the
suspended process, closes its pipes and releases the job before it propagates. The pipes are
closed whatever the kill or the reap raise, and a fault of that cleanup (a second Ctrl-C included)
never takes the place of the original exception; where the reap does not confirm the end within
5 s, the call ends all the same, the process's end unconfirmed (recorded as a note on the exception,
within the limit below). On POSIX `ssh` starts
a session and process group of its own, and its leader is **never reaped before the group was
signaled** (an unreaped leader keeps its PID, and so the group id, from passing to another
process). Its exit is seen without reaping: `os.waitid(..., WNOWAIT)`; where Python has no
`os.waitid` (macOS with Python 3.11 or 3.12), a kqueue `NOTE_EXIT` watch; where neither exists, the
end of stdout plus a grace of 0.5 s stands for it (an `ssh` still running then is ended with its
group, and its exit code is the signal's). Every way a call ends -- an answer, a form fault, a limit,
a timeout, Ctrl-C, any exception -- sets the stop signal and ends the whole tree once: the job
terminated, or on POSIX one `SIGKILL` to the group, sent while the leader is still unreaped and
never a second time -- after an answer too, so a helper such as a `ProxyJump` hop cannot hold stderr
open. Then it reaps `ssh`, joins the reader threads and closes each pipe whose thread has ended,
before `SshError` or the interrupt reaches the caller. **Limit:** a descendant that explicitly
breaks away from the job (Windows) or starts a session of its own (POSIX) is outside the tree, and
so are the helpers of a child whose PID was reaped behind modelroom's back (`ChildProcessError`: no
group signal is sent to a PID no longer known to be its own). A read such a descendant still blocks
is canceled on Windows (`CancelSynchronousIo`); on POSIX it cannot be, and its thread (a daemon
thread) and pipe are then left behind rather than closed under a blocked read.

**Time limits** (seconds, from the ssh process's own start; a limit reached kills the process):
`hello`/`room` 30, `profile` 180, `daemon` short calls 30, `POST /api/generate` 660, `POST
/api/pull` no total limit but 60 s **per line**, `candidates` 60 (one list plus one `/api/show` per
candidate), `vllm` 30 (AP9-SV2-R: one `/v1/models` and the `stat` calls of the cache), `vllm-measure` no
total limit but 660 s **per line** (a warm-up or a batch between two progress lines), `measure` no total limit but 660 s **per line** (between two progress lines lies a whole
`generate`, B1's own table; a pull or a measurement that keeps sending lines is not caught by this, a
truly silent connection is). **Answer limits:** every line at most 1 MiB (pull lines 64 KiB); the
whole answer at most 64 KiB (`hello`/`room`/`vllm`), 4 MiB (`profile`), 16 MiB (`daemon` short calls and
`/api/generate`), 4 MiB (`candidates`) -- `measure` and `vllm-measure` have no total limit, only their 1 MiB per-line one
(a `MeasurementRecord` fits well inside it), like a pull's own per-line rule. Either limit reached:
`<alias> answered with more than modelroom reads`, the process killed.

**What is not checked**: whether the alias truly names the intended computer (only the identity
check of "Another computer" below judges that), whether the alias itself is a neutral name (a
user's own choice, README), the host-key and password rules of a `ProxyJump` hop (the user's own
`ssh_config`: `BatchMode` and `StrictHostKeyChecking` given for the target do not reach the hop),
where Ollama keeps its models on the server (the same approximation "The pull" already states for
this computer), and whose daemon actually answers on the server's `127.0.0.1:11434`.

### Another computer (AP9-FR, FR-B1, "Guided mode" step 1 and step 5)

**Adding a machine over SSH (step 1, row `ssh`, after `import a profile file`):** a host alias
(`ssh.ALIAS_RE`, `^[a-z0-9][a-z0-9-]{0,30}$`, no dot, no colon, no `@`, not all digits, and not in
the shape of a neutral name so an alias and a neutral key can never collide), `hello`, the
`profile_id` to measure under (the one `~/.modelroom/remote.json` already has for that alias when
its profile is still in the folder and its fingerprint has not changed, else a fresh one), the
class question, `profile`. **Write order:** (0) the answer is validated in full -- its
`profile_id` exactly the one asked for, no measurements -- before anything is written; (1)
`remote_store.set_alias(profile_id, alias)`, atomic; (2) `import_profile(..., name=alias,
expected_profile_id=profile_id)` from a temporary file, deleted after. A fault before (1) writes
nothing; a fault between (1) and (2) leaves
`remote.json` pointing at a profile not (yet) in the folder -- read as absent until the same alias
converges it on a later run, never a crash, never a rollback. The profile's `[machines.<name>]` key
is the alias itself (`neutral.key_for`); origin `ssh` (`selection.Origin`), ranked after `this`,
before `imported`.

**A changed fingerprint behind a known alias** (a rented machine replaced): the question `<alias>
now answers with another machine than before. Is it the same machine or a new one?` (default `new`,
GF decision). `same` keeps the id. `new`: a fresh id is measured; once its answer is validated,
(1) moves the alias to the fresh id, and (2) renames and imports under one and the same import
lock: the import's own sweep, its first step under the lock, finds the row `[machines.<alias>]`
still naming the *old* profile -- no longer the alias of its own profile -- and renames it to the
old profile's neutral name (or a free `computer-<8 hex>` where that name is taken; its reserves and
`writer` kept), then the plan reads the rows as renamed and the new profile gets exactly
`[machines.<alias>]`. A rename that cannot be written aborts this import: nothing of the new profile
is written, and `remote.json` names the fresh id, which the same alias converges on a later run.
When a stored profile picked directly turns out to be another machine, the table of *this* run
still counts the old profile (a note says so): no pull and no measurement for `alias` in this run.

**A fingerprint that is `none` (AP9-FR-N2, e; GF decision 2026-10-01).** The comparison needs a stored
profile to compare with; without one nothing is asked and nothing is said. Where one is stored: both
fingerprints a value, different -- the question above, as before; **exactly one of the two `none`**
(the server had an identity and has none now, or the other way round) -- the same question, because a
computer that just lost or gained its identity is not provably the same one; **both `none`** -- no
question (there is nothing to tell them apart by), and once per run the note `<alias> gives no
machine identity; modelroom cannot tell whether it is still the same machine`; both equal -- nothing.
The fingerprint of a Linux server is its `/etc/machine-id`; a server without that file has none. A
second identity for such a server is a package of its own. The binding of this computer
(`binding.py`) is unchanged. A `new` after an import that failed is read as before: `remote.json`
names an id with no profile in the folder, and the next run with the same alias takes a fresh id
without asking (no history of earlier profiles is kept).

**Every already-stored SSH profile is checked once before it is used actively** (room, pull,
measurement) even when it was picked directly from the machine list, not just a freshly typed
alias: the same `hello`/fingerprint comparison and the same question. Under `--answers` with no
answer for it, `AnswerMissingError` (exit `2`) -- and, once the change was detected, nothing more
of this alias is touched: no further `hello`/`profile`/`room`/pull/measurement, no further write.

**One active SSH target per run** (GF decision): a freshly typed alias is it outright; with exactly
one stored SSH profile chosen it is that one; with several, `Which computer over SSH should this
run connect to?` (`ssh_active`, optional -- unanswered or skipped means none). Every other chosen
SSH profile is ranked from its stored profile alone, with no connection at all (`free`-line `disk
unknown (<alias> not connected in this run)`); in `pull_other`'s list its rows are handed over like
those of a computer without SSH (the command to paste there).

**Step 5, the active target:** its table's `room`/`free` line is its own live reading (`room` op),
named by the alias, never a path. Its rows stand in `pull_other`'s list like every other chosen
machine's; the row chosen there -- whichever rank (`pull_other = "<alias> <name>"`) -- is pulled
over SSH (`guided_remote.pull_remote_row`) through the unchanged `PullTarget`/`pull_chosen` chain:
the install line `#<rank> on <alias>  ollama pull <name>`, the digest proof (over `SshDaemon`,
`#<rank> is local already on <alias>`), a fresh `room` reading for the disk check (`not enough disk
on <alias> for #<rank>: needs <n> GB, <m> GB free`), the question `Pull #<rank> onto <alias> now?
(<size>)` (`pull_remote`, default No, optional) and the success line (`✓ Pulled <name> · <size> · on
<alias>`). No row chosen, no pull over SSH -- the first row is never taken on its own.
`where=<alias>`, bare; the "on"/"onto" wording is the screen's own, not part of the field.

**What the active target already holds (AP9-FR-N2, a; GF decision 2026-10-01).** Step 5 asks the
counterpart **once** which packages of the run Ollama there holds (`candidates`, the digest rule of
"Load test on the server": identities only, no name of the server's own), in `guided_remote.active_room`
before the `room` op, and puts the answer into the `held` of the target's `Disk` -- the one set the
table's mark, the `disk` line and the list read, so a held package is never "too large" for that
disk either (U11c, below). A failed call is no problem of its own: no `held`, no line, nothing more
said -- the digest proof after the choice stays the binding check. A failed `room` read keeps the held
set: the `disk` line says `gpu1 not reached`, the `local` line and `local already` stay. Where a table is that of a computer
whose Ollama was asked in this run (this computer with its daemon reached, or the active target), a
line `local` stands behind its `speed` line when at least one shown row is held: `local     #1, #4
already in Ollama on this computer`, `local     #8 already in Ollama on gpu1` (ranks as everywhere,
`#2–5`; no column, no effect on rank or order). In `pull_other`'s list a row of the **active** target
that it holds ends `  local already` like the rows of this computer's own list; the rows of another
stored SSH profile, not connected in this run, do not (no daemon was asked, "One active SSH target
per run").

**What `remote.json`/a file never carries** (B6): `HostName`, address, user name, port, key path,
jump host, ssh's own stderr, the server's own text (error, Ollama's), its paths and drive, its
local model names and its `/api/ps` names, the shared folder's file names, a host name out of an
export, and this computer's own name (live in the start line only, FR-N). The alias is the one
thing kept, as `display_name`, as the machine key, and in `remote.json`. **Limit:** if the alias
*is* the server's real short name, that name stands in the folder -- `ALIAS_RE` blocks an address
or a domain, not a user's own naming choice; README and the question text ask for a neutral one.

#### Shared folder (FR-S, `modelroom/guided_share.py`, GF question 7 (c))

The other way of reading another computer, for a laptop and a server too far apart for a direct
`ssh` connection: a folder of export files a computer without SSH drops profiles into and reads
them back from. Where it is, read only by `guided_share.share_folder(environ, store)`, in this
order: the environment variable `MODELROOM_PROFILE_SHARE` when it is set and not empty (an empty
variable counts as unset everywhere), else `profile_share` of `remote.json`, else none. Nothing
this module writes carries the path further than `remote.json` and this step's own screen ("Remote
machines").

**Row 5 of the machine list** ("Guided mode", step 1) is always there, whatever the folder holds:
`set up a shared folder for profiles` with no folder, `profiles from the shared folder` with one
(stored or from the environment). It never reads the folder -- an unreachable network drive must
not hold up a run that never opens this step.

**The step, "Shared folder"** (value `share`, a `Choice` of the machine list's own checkbox --
shown once, there, never repeated as an interactive note beside it). Its head names the stand --
`none set yet`, the stored path, or `set by environment (MODELROOM_PROFILE_SHARE)` without a path
-- then one choice (`share_setup`): `use` (only where a folder is there, stored or from the
environment) takes profiles from it; `choose`/`change the folder` asks a path (grayed out, with the
reason, while the environment wins) and, once typed (`share_folder`): checked for form (after
`expanduser`, absolute, not empty, no NUL, at most 1024 characters, no Windows device or pipe
prefix `\\.\`/`\\?\` -- a write there could open a pipe instead of a file), then for writing (a
probe file, created exclusively; a write or close that fails after the create still removes it, a
create that fails outright never deletes anything that was already there; a leftover probe file
that could not be removed is a note, the folder counts all the same), stored as typed (never
`resolve`d) on success; `none` **while the environment is set skips this run only, and leaves a
stored folder exactly as it was** -- the environment always wins, so clearing storage under it
would be a promise the run cannot keep; without the environment it clears a stored folder. A form
or write problem asks the question again at a terminal, ends the step on an empty answer, and is
`AnswerInvalidError` (exit `2`) from an answer file, the same shape as `_allowed_folder` above.
`store.profile_share` itself is read in exactly one place, `share_folder`; everything else that
needs to know the stored value alone calls `share_folder({}, store)` (no variable can win against
an empty environment), never the field.

**Reading the folder**, on `use` or a successful `choose`: every `*.json` directly in it (never a
subfolder), at most 200 files -- a file beyond that count is left alone, not even opened. Each is
read as an export of at most `max_bytes + 1` bytes (16 MiB the limit; never the whole file, however
large), and one over it counts as broken, exactly like a file that does not parse or does not
validate (`ExportTooLargeError`, a `StoredFileError`). An unreadable folder is `the shared folder
cannot be read now; nothing was imported from it`, the run goes on; files that do not read as an
export are one note with their count; one row per `profile_id` (`share_files`, a checkbox),
described from the file with the newest `recorded_at`, never from a file name, and never from a
`display_name` that is not itself neutral (`neutral.display_name_for`, the FR-N sink's own fallback
-- a foreign export whose source forgot to neutralize is shown, answered and named in a failure
sentence under its neutral name, never the one the file actually carries).

**Importing a chosen `profile_id`**: every readable export of it, oldest `recorded_at` first (ties
by file name), each through the unchanged `import_profile` with `expected_profile_id` and
`max_bytes` (16 MiB). Both are checked on the object it actually reads, before it is planned, so a
file that changed after it was listed -- a different profile now, or grown past the limit -- is
`StaleExportError` (an `ImportConflictError`) either way. The two checks do not depend on each
other: `read_export_file(..., max_bytes=...)` alone (no `expected_profile_id`, the plain listing's
own use) still raises `ExportTooLargeError` for an oversized file, a `StoredFileError`; it is
`import_profile(..., max_bytes=...)` that turns that `ExportTooLargeError` into `StaleExportError`,
on `max_bytes` alone -- `expected_profile_id` plays no part in that particular translation, it only
adds its own, separate `StaleExportError` for a `profile_id` that no longer matches. Every exception
is one of three fixed sentences, none of them naming the file: `a file in the shared folder changed
after it was listed` (either check above), `a file of <name> in the shared folder contradicts one
imported before` (any other conflict -- a contradicting measurement, most often), `a file of <name>
in the shared folder could not be imported` (everything else); `<name>` is the neutral name as well.
One file's failure never stops the others of the same `profile_id`, or the next `profile_id`. Origin `imported`, the
same as `import a profile file`.

### Load test on the server (AP9-FR, FR-B2)

Step 4's second question round, only for the run's one active SSH target (B4) and only after the
local servers' own round (`local_servers.load_tests`) -- no active target, no question at all.

**Fremde Namen (foreign names, Codex r2-3).** A registry package can be installed at the server
under any local name (proven only by the digest, `loadtest.installed_candidates`), and `/api/ps` can
name other models loaded there. Neither ever reaches the laptop: **every** package name the laptop
shows or stores it builds itself, from the package identity the request already carries
(`screen.install_name`, "Neutral names" above); what is stored is whatever the sink (FR-N) allows.

**`candidates`** (`modelroom/remote_client.py:candidates`, `remote_server.candidates_result`):
request `{"packages": [<Package>, ...]}` -- the run's own packages (`pull_choice.run_packages`).
The server runs stage 1's own rule against its own `/api/tags` (no download, installed only) and
answers `{"candidates": [{"package_identity": [source, repo\|name, stem\|tag]}, ...], "set_aside":
int}`: identity only, never an installed name; cloud and unmatched entries counted together as one
number, never named or broken out by reason. The laptop builds the row of each candidate (name,
base model, quantization, weight size) from its own copy of the package the identity names --
`guided_remote_measure._remote_rows`, the same four columns and widths as the local list.

**`measure`** (`remote_client.measure`, `remote_server.measure_result`): request
`{"package_identity": [...], "packages": [<Package>, ...], "scenario": <Scenario>, "profile_id":
"<16 hex>"}`. The server looks its candidate up again itself (the same digest rule, since its own
state could have changed between the two calls -- no match is `internal`, this server's own fault)
and uses its local name only internally, for the calls to its own Ollama; `loadtest.run_load_test`
runs **at the server**: its own GPU utilization (`nvidia-smi`) and its own one-minute load average
are read there, not on the laptop, because the process that reads them is the server's own.
Progress carries only `{"stage": "warmup"\|"run", "number": int, "of": int}` (B1's own protocol
counts: 1 warm-up run, 3 measured), never a package name. Before the `result` line, the record goes
through the neutral sink (`neutral.neutral_record`, FR-N) with the package the server just matched,
so `ollama_name` is the name built from that package, never the name the server's own Ollama holds
it under, and every other field (`daemon_version`, `comparable_reason`, `validity_reason`,
`runs[].done_reason`) is in its sink-allowed form already at the server.

**The double sink.** The laptop validates the answer against the wire's own strict shape
(`remote_protocol.MeasureResult`), then runs the record through the very same sink **again**, with
its own copy of the matched package, before it is ever offered to `store_measurement`. Where the
result differs from what the server sent -- the server did not hold the rules, or its
`MeasurementRecord.profile_id` is not the alias's own -- the whole record is discarded, never
stored, with the one fixed sentence `<alias> answered in a form this modelroom does not read`
(`remote_client.FORM_ERROR`, the same sentence a malformed envelope gets). A local measurement goes
through the identical single sink pass it always has (`loadtest.store_measurement`); there is no
special case for the remote path (Codex r3-2).

**Stored under the alias's own `profile_id`** (B3/B4): the id the laptop already resolved for this
alias (`remote_store.aliases`), sent in the `measure` request and never chosen by the server. Time
limit as B1's own table: no total, 660 s between two lines of one call (a whole `generate` between
two progress lines).

**Step 4's question, gated on the active target alone (B4, Codex r1-6, r4-2, GF question 6):**
`Measure the speed on <alias> now?` (`load_test_remote`, yes/no, default no, optional; missing under
`--answers` is no), then, only on yes, `Which of these should be measured on <alias>?`
(`load_test_remote_packages`, the candidate list, the local list's own row form). No candidate at
all: neither question is asked. Several stored SSH targets: only the one active target's own
candidates are ever fetched or measured (`guided_remote.resolve_active`, unchanged) -- the other
stored profiles are ranked without a connection, as B4 already has it.

### vLLM at a server over SSH (AP9-SV2-R)

`modelroom remote` reads and measures the **server's own** vLLM: against `127.0.0.1:8000` there, with the same
three calls as a local run (`vllm_server.py`: `GET /v1/models`, `GET /metrics`, `POST /v1/chat/completions`) and
nothing else. No port is forwarded, no tunnel is built, nothing connects from the laptop to that port, and nothing
is started or loaded. It is FR unchanged -- the fixed `ssh` call, the envelope `{"protocol": 1, ...}`, one active
SSH computer per run, the sink of "Neutral names" -- plus three optional fields and two operations
(`modelroom/remote_vllm.py` at the laptop, `remote_server.vllm_result`/`vllm_measure_result` at the server).

**The three fields (R1), all optional, a missing one reads as "not there".** `hello.vllm` is the vLLM entry's own
`runtimes.detect_vllm` at the server (A3), on the server's own platform: `serving`, `installed`, `absent` or
`asks_key`; a Windows or macOS server is `absent` whatever else is on it. `profile.export.profile.runtimes` is
what that server has (see "Hardware profile"). `room.model_disk` is where vLLM keeps its models there
(`hf_cache.hf_cache_candidates`): `{"known", "free_bytes", "same_as_ollama", "relative"}` -- numbers and flags, **no
drive and no path**; `relative` says that the effective cache variable is a relative path, and `same_as_ollama` that
both places are on one drive (`Disk.same_place`: the table then says `disk 812.4 GB free on gpu1 (Ollama and vLLM
models)`, else two pieces). An older server sends none of them (`absent`, nothing, `disk unknown on gpu1` for vLLM's
place).

**`PackageRef` (R2, Codex r1-1, r1-2).** Over SSH goes **never a whole `Package`** (its `approval.by` is a person's
name) and never a whole `Scenario`: `remote_vllm.package_ref(package)` is `{"repo", "revision", "files": [{"name",
"size_bytes"}, ...]}` with exactly the start files (`tensor_quant.start_files`), built from checked values -- the
repo against the Hub's name rule (and ASCII), the revision 40 lower-case hex digits, every file name against
`SAFE_REPO_PATH_RE`; a package that does not pass is not sent. The packages of a request are the active, complete
Hugging Face tensor packages of the run (`remote_vllm.sendable`), and every index an answer names is a position in
that list. `extra="forbid"` on both sides, and the server checks every field **again before it touches the cache**:
a violation is `bad_request`, with no `stat` and no `realpath` made.

**`vllm`.** `state` as `hello` says it; `served` only for a server that answers with one model, `{"package": <index of
the one package of the request whose repo is the model's `root`>|null, "context": <max_model_len>}` -- `null` for none,
and for a repository named twice (one repo at two revisions: the server states no revision). **The model's own name
-- `id`, `root`, a local path, a free `--served-model-name` -- never leaves the server**: only that index does, and this
computer shows only names it builds from its own package. `missing_bytes` holds one entry per package, in the request's
order: what `hf_cache.missing_bytes(ref, cache)` says -- the **one** function both ends read, so the rules of I3 stay
in one place (`hf_cache._inspect`). It is `0` **only** where every start file is there with its declared size, the sum
of the missing ones otherwise, and `null` where nobody can say -- an unsafe name, a link that leaves the repo's folder
and the cache's blob store, an unreadable cache, or a missing start file that declares size 0 (which would otherwise add nothing to the sum). It
is a number or `null`, never a file name or a path. The laptop checks the list against what it sent
(`remote_vllm._checked_state`): as many entries as packages, each `null` or between `0` and the sum of that package's
start files, `served` only where `state` is `serving`, an index only of a package sent -- else the form fault
`<alias> answered in a form this modelroom does not read`. `0` is `on disk already` and needs no disk; `n > 0` goes into
the disk rule (U11) as `n` plus the margin; `null` as the whole download.

**`vllm-measure`.** First a look at what the server serves now: no answer is `vllm_unreachable`, a `401`
`vllm_asks_key`, any model but `packages[package]` (or no single model) `other_model`. Then one warm-up request and three
batches of `min(requests, 256)` at once (`loadtest_vllm.run_vllm_load_test`, `MAX_REQUESTS` = 256 since AP9-PM-1, SV1's protocol otherwise unchanged), with a `progress`
line after the warm-up (`{"stage": "warmup", "number": 1, "of": 1}`) and after each batch (`{"stage": "batch", "number":
n, "of": 3}`) -- numbers only. The result is numbers and **one code** of the fixed set `invalid_status`, `invalid_finish`,
`invalid_tokens`, `other_context`, `other_requests`, `foreign_load`, `foreign_tokens`, `server_gone`: the laptop builds the
words itself (`loadtest_vllm.PROBLEM_WORDS`, SV1's sentences with the numbers it knows -- the context and the requests it
asked for and the ones the server reports). An unknown code is a form fault; speeds come all together and only without an
`invalid_*` code or `server_gone`. The laptop requires the requests the server states to be the ones asked for, and
checks an answer in this order (AP9-PM-1): `server_gone` first, as before; a `ranking_requests` other than the scenario's
is a form fault; fewer requests than `min(requests, 256)` are `modelroom on <alias> reported <got> requests per batch
where this one asks for <asked>; update the older one` (an older modelroom at the server, which measures at most 64);
every other deviation, more than asked included, is a form fault. In all of them no number is taken. A modelroom at the
laptop older than this change refuses a server answer of more than 64 requests as a form fault (`le=64`): that is
the old code and stays so; no pairing of old and new produces a wrong number. `remote_protocol` states the limit
(`le=256`) without importing `loadtest_vllm`; a test keeps the two equal. The measurement stays in memory (E3): the step's note, the card (`identity stated by the server; not ranked, nothing written`),
no file on either side.

**What the server never does.** It writes nothing (the cache is only `stat`ed, never read), stores no measurement,
starts nothing, and sends no text of its own. Its answers carry no name, no path and no host name.

**The guided run (R3).** `hello.vllm` of the active computer goes into the runtimes it has *live* in this run
(`guided_remote_vllm.live_map`, F3): `serving` and `installed` count, `asks_key` and `absent` do not (E5). The `runtime`
row of its table reads `runtime  Ollama · vLLM serving Qwen3-8B-AWQ · on gpu1`, the model name from this run's own
package (`vLLM serving another model` where it serves none of them). Step 4 asks `vllm` for the run's tensor packages
and offers `vLLM server on gpu1  context 32k  10 requests at once  <package name>` in the same list as that computer's
Ollama candidates (`load_test_remote_packages`, value `vllm`); a model that is no package of the run is the one note
`the vLLM server on gpu1 serves a model that is no package of this run`, a server asking for a key
`the vLLM server on gpu1 asks for an API key; modelroom reads an open server only`. Step 5 turns `room.model_disk` and
`missing_bytes` into the disk rule of that computer's vLLM rows (I2): its table marks rows over the free disk, the
`install` line takes the best row that also fits it, and the list of `pull_other` locks the others. The disk rule over
SSH is for the **vLLM rows** of that computer alone: its Ollama rows read no disk there, as in FR (no mark, no lock, no
skipped rank, the same command). A server that sends no `room.model_disk` still has what `vllm` counted taken over, on a
place of unknown free space (`disk unknown on gpu1`, also beside Ollama's disk piece in a mixed table). The numbers of a
`vllm-measure` answer must agree with its code: `requests` below `ranking_requests` is `other_requests` (an `invalid_*`
code may hide it), a `context` other than the requested one has a code, and each of the two codes needs its number, else a
form fault. A server that asks for a key is said so once even where the run has no tensor package to send. Choosing a vLLM row
there prints the two commands of I1, built from the profile the server sent, with `#<rank> on gpu1` in the label and the
cache only as the shell expression (a relative cache variable at the server leaves the Docker line out with its note); it
starts nothing over SSH, asks no yes/no question and pulls nothing (GF question F1 a).

### Migration to schema 2

`modelroom migrate --config <file>` turns a schema-1 deployment into schema 2, in one run under
`modelroom.lock`:

- every schema-1 profile `<state>/hardware/<name>.json` becomes `<profile_id>.json` with a new
  random `profile_id` (`normalize_profile_v1`); its embedded measurements become measurement
  files with `protocol: none` (`legacy_measurement_record`, whose 8 hex characters come from the
  SHA-256 of the old measurement, so the same input always gives the same id); the old file is
  kept as `<profile_id>.json.v1.bak`, through the sink (a neutral key, no runtime text, no names of
  installed models; AP9-FR), and removed last, once the configuration that names its new profile is
  written: a run that stopped before finds the new profile by it again, whatever ids it draws;
- a schema-1 `modelroom.toml` becomes schema 2 (`normalize_config_v1`) with
  `[machines.<name>].profile` set for every migrated profile of that name; the rewritten text is
  validated like `load_config` would before it is written, the original bytes are kept as
  `modelroom.toml.v1.bak`. The rewritten file carries no comments; the backup keeps them.

**Schema 2 to 3 (decided 2026-09-25, ADR 0001 and 0002).** The same run takes a schema-2 folder
on to the current schemas, in place:

- every schema-2 profile `<state>/hardware/<profile_id>.json` is rewritten as schema 3 under the
  same name and id (`normalize_profile_v2`: every value kept), the original bytes kept as
  `<profile_id>.json.v2.bak`;
- a schema-2 `modelroom.toml` is rewritten as schema 3 (`normalize_config_v2`), the original
  bytes kept as `modelroom.toml.v2.bak`; the machines keep their `profile`.

**Profile schema 3 to 4 (decided 2026-09-26, ADR 0004).** The same run takes every schema-3
profile on to schema 4 in place (`normalize_profile_v3`: one card from the state, the class
`unknown`), the original bytes kept as `<profile_id>.json.v3.bak`; a schema-2 profile goes through
both steps in the same run and keeps `.v2.bak` only. The configuration stays schema 3.

Every backup is named after the version the file left, so a folder that went from schema 1 to 2,
to 3 and now to 4 keeps its `.v1.bak` and `.v2.bak` files next to the new `.v3.bak` ones; a
schema-1 folder goes to the current schemas in one run and gets `.v1.bak` only. The guided mode starts the
same run when the configuration it finds is below schema 3 and says so in one note
(`... backup kept: modelroom.toml.v2.bak`). Reading never writes: `render`, `hardware` and the
guided mode read a schema-2 configuration as schema 3, and a schema-2 or schema-3 profile as
schema 4, and leave the files as they are. Profiles are written back by
`modelroom migrate` alone (directly, or through the guided mode's trigger, which looks at the
configuration's version): a folder whose configuration `import-profile` already took to schema 3
keeps its schema-2 profiles, readable, until `modelroom migrate` runs. A file the run cannot read
stops it before anything is written; the guided mode then names that file and says nothing was
changed.

The whole run is planned and checked before the lock is taken, and again under the lock, before
anything is written: every file is read and version-checked, every profile and measurement is
converted, and every file the run would write -- the new profile, each measurement file, each
backup -- must be absent or identical, and no existing part of its folder path may be a file. The new `profile_id`s are drawn once, before the first
check, so both checks look at the paths the run writes. Under the lock the configuration is read
again; if its `[paths]` changed meanwhile, the run stops (exit 3) without writing anything but
the lock file. A file of a schema above the current one (profile 5 or later, configuration 4 or
later) is exit 3; a conversion that fails, a target or backup that exists with other content or cannot be
read (a folder at a backup path, for instance), or two schema-1 files for the same
machine is exit 3 as well; in every case nothing is written and no lock file is created. A run
that finds everything at the current schema prints `nothing to do` and exits 0. A run that
stopped halfway converges on the next run: a converted profile with fingerprint source `legacy`
whose content is the same but for its `profile_id` and `display_name` keeps its `profile_id` (since
AP9-FR: the sweep before the run has given it a neutral name, so the name no longer finds it; a
profile of other content is another machine's, and the run takes a new id), and files that are already there are left as
they are or rewritten in place: a measurement file with the same JSON content, a profile the same
after reading -- whatever schema an earlier version left it in, compared through
`read_profile_document` (decided 2026-09-26). A configuration that is already schema 2 or 3 never gets
`[machines.<name>].profile` from this run: when it sits next to schema-1 profiles, the profiles
are migrated and the entries are left to the guided mode.

| Exit | Meaning |
|---|---|
| `0` | migrated, or `nothing to do` |
| `1` | another process holds the lock |
| `2` | the configuration is missing or invalid |
| `3` | a file has an unsupported `schema_version`, a hardware profile does not read, validate or convert, two schema-1 files name the same machine, a target, measurement file or backup with other content or that cannot be read is in the way, or `[paths]` changed while waiting for the lock |

After migration the old `hardware` and `render` commands no longer find the per-machine
`<name>.json` files; they switch to schema 2 in the work packages that rewrite them.

### Export and import

`modelroom export-profile --config <file> [--profile <profile_id>] --out <file>` writes one
`ExportObject` (above): a profile and every measurement of that profile, read from the results
folder the configuration file sits in (the results folder *is* the configuration's own folder).
Without `--profile` the exported profile is this machine's binding for that folder (the pointer
file, "Profile binding" above); `--profile` names any profile in the folder and wins over the
binding, so a profile can also be handed on from a shared folder by a machine that has no
binding of its own. The file is written with `atomic_write_json`, so a reader never sees half of
it. The whole export holds `modelroom.lock` (a lock another process holds is exit `1`): the sweep of
machine names comes first under it ("Neutral names"), then its profile and every record go through
the sink before the file is written, so it carries a neutral `display_name` (or the SSH alias of that
profile) and no foreign text. Two kinds of broken measurement differ. One the sweep cannot clean --
a profile or measurement file that does not read as a JSON object, or a record whose package
reference is of another shape (not an object, or `content_source`, `hf_repo`, `hf_file_digest`,
`ollama_manifest_digest` that is no text, such as `hf_file_digest: []`) -- stops the export with exit
`1` before it writes (decided 2026-09-29, "Neutral names"). One the sweep has cleaned but that is
still no valid record (a value out of range) is reported and left out; it does not stop the export. `--profile` has to be a `profile_id` (16
lowercase hex characters), and `--out` must be none of the files this package writes itself:
nothing inside `paths.state` (the same rule `PathsConfig` holds for `paths.markdown`), not the
configuration file, not the pointer file -- otherwise an export object would replace a profile,
the snapshot, the lock file, the configuration or the binding, outside the lock.

`modelroom import-profile <file> --config <file>` reads such a file into another results folder,
under `modelroom.lock`. Under the lock the folder is swept first ("Neutral names"), and the export
goes through the sink before the plan, so the freshness and the conflict rule compare neutral with
neutral: the same record from an export of 0.1.0 and one of 0.2.0 is `unchanged`, not a conflict.
`import_profile(..., name=<alias>)` is the SSH way's alone (AP9-FR, FR-B1): the machine is then
`[machines.<alias>]` and its profile carries the alias as its `display_name`. A refused import
leaves what the sweep cleaned cleaned. The whole run is planned under the lock and only then written, so the
plan *is* the freshness check -- every writer of a profile, a measurement or the configuration
holds the same lock. A conflict anywhere in the plan leaves the run without a single write. The
configuration is read once before the lock (a missing, invalid or schema-1 file is refused
without even creating a lock file) and once under it; the second read is the one the plan uses,
because another import may have added its own `[machines.<name>]` entry in between. If `[paths]`
changed meanwhile, the run stops with exit `1` and asks to be repeated. A schema-2 configuration
is written back as schema 3 by every import (decided 2026-09-25) -- also when the imported
machine already has its entry and nothing else about the file changes. Besides the one `.bak`
below, which is never overwritten, it keeps the schema-2 bytes as `modelroom.toml.v2.bak`, the
name `modelroom migrate` uses, and says so in one line (`note: the configuration is written as
schema 3; schema 2 kept as modelroom.toml.v2.bak`); a `.v2.bak` of other content is a conflict,
exit `1`, nothing written. A schema-2 profile inside the export file is read as schema 3.

The profile is keyed by `profile_id`:

| Case | Decision |
|---|---|
| no profile of that id in the folder | written (`new`) |
| the imported `recorded_at` is younger | written (`updated`) |
| both `recorded_at` are equal | nothing, however the content differs (`unchanged`) |
| the stored `recorded_at` is younger | the stored profile stays (`kept`) |
| a file of that name that does not read as that profile, or one that is still schema 1 | conflict, exit `1`, nothing written |
| a file where a folder of the write path belongs (`hardware`, or the profile's measurement folder) | conflict, exit `1`, nothing written |

The measurements are keyed by `measurement_id` and decided independently of the profile
(`plan_measurement_import`, "Measurement files and import rule" above):

| Case | Decision |
|---|---|
| unknown id | written (`new`) |
| known id, same content | nothing (`unchanged`) |
| known id, other content | conflict, exit `1`, nothing written |
| a file of that id that does not read as a measurement | conflict, exit `1`, nothing written |

Two machines are never merged and never renamed:

- the same `os_fingerprint` under another `profile_id` is a note -- `same hardware id (cloned
  image?)` -- and both are kept;
- the same `display_name` under another `profile_id` is a note as well; both are kept, and a
  name that more than one profile carries is shown with the short id (the first 8 hex of the
  `profile_id`).

The imported machine gets a `[machines.<name>]` entry: `<name>` is the key the sink gives a new
row (`neutral.key_for`): the SSH alias where one is passed, else the profile's neutral name, else,
where that name is taken, a random neutral key `computer-<8 hex>`; the reserves come
from `[defaults]`, `writer` is `false` and `profile` is the imported `profile_id`. A machine
that already carries this `profile` keeps its entry, and the file is not rewritten at all; a
machine whose profile is in the folder but in no `[machines.<name>]` entry gets one even when the
profile itself is not written (`kept`, `unchanged`), so an imported machine never stays
unconfigured. The
rewrite goes through `toml_writer.dump_toml` and is validated exactly as `load_config` would
before anything is written, so a configuration that would not load is never written; it keeps
every value of the file it replaces, but not its comments. Therefore the file as the user wrote
it is kept once, next to it, as `<config>.bak`: written immediately before the first rewrite, and
only when no backup is there yet -- an existing `.bak` holds an older state and is never
overwritten. Either way the report names the backup path, and a run that rewrites nothing writes
no backup.

An import never changes the local binding; it reads the pointer file for the sweep only. Broken
files in the target folder (a profile that does not read or validate, one of an unsupported
`schema_version`, a schema-1 profile, an unreadable measurement file) are listed in the report
and left alone; they never stop the import of another profile -- except a profile or a record that
does not read as a JSON object at all, which the sweep before the import cannot clean and which ends
the command with exit `1` ("Neutral names"). The report carries one line per
profile and per measurement, and a run that changed nothing ends with `nothing to do`.

| Exit | `export-profile` | `import-profile` |
|---|---|---|
| `0` | written | written, or nothing to do |
| `1` | -- | another process holds the lock, or a stored file contradicts the import |
| `2` | the configuration is missing or invalid, `--profile` is not a `profile_id`, `--out` is one of the files the package writes itself or cannot be written, no profile is bound and none was named, the named profile is not in the folder, or it is still schema 1 | the configuration is missing, invalid, still schema 1, or holds a value the TOML writer cannot write back (`inf`), or a file could not be written (the message names it; every file already written is complete, and repeating the run finishes the import) |
| `3` | a stored profile, or the pointer file, does not read | the export file does not read, or its `schema_version` is outside the accepted range (`EXPORT_SCHEMA_RANGE`: schema 1 today) |

```console
$ modelroom export-profile --config /srv/models/modelroom.toml --out /srv/exchange/workstation-3f9a0c21.json
exported workstation-3f9a0c21 (3f9a0c21d4e6b870) with 2 measurement(s) to /srv/exchange/workstation-3f9a0c21.json

$ modelroom import-profile /srv/exchange/workstation-3f9a0c21.json --config /srv/laptop/modelroom.toml
profile workstation-3f9a0c21: new
measurement 20260922T202000Z-4da3b585: new
measurement 20260923T083000Z-5c1e9a07: new
note: machine "workstation-3f9a0c21" added with the reserves from [defaults] and writer = false
```

### Catalog rules

The catalog `modelroom/catalog.toml` is maintained in this repository and shipped with the
package: per family the publisher account, its base models, which of them are the publisher's
current models (`latest = true`, a positive statement about that one model), successor links, the Ollama name, and
for every model row the page that proves it (`source`, a `https://huggingface.co/` page). A
missing statement is never evidence: a model with neither `latest` nor `successor` has age
`unknown`. Every `latest = true` carries its own evidence as two fields: `latest_source`, an
https page with a host, of the publisher, that names the model as current (on the Hub -- host
`huggingface.co`, `www.huggingface.co` or `hf.co` in any case and with any port -- only the
publisher's own page `https://huggingface.co/<publisher>` or one of its collections
`https://huggingface.co/collections/<publisher>/...`, query and fragment ignored; a model's own
page is no evidence), and
`latest_checked`, the date the maintainer confirmed it. Both are set exactly when `latest` is
true. A **stated** `latest` or `successor` is never derived from a number, a line or a family: a
higher version number in the same line does not make an earlier model `legacy` in the catalog, only
an explicit `successor` does. The list's **computed** status is derived from the version numbers of
a family, and says so with `*` ("Latest and legacy evidence", "Computed release"; decided
2026-09-25); the catalog itself computes nothing. `Catalog.age_of(hf_repo)` returns `latest`, `legacy` with its successor, or
`unknown`. A model with a successor is never `latest`; successors stay under the family's
publisher and never form a cycle, within a family or across families; family names and
`hf_repo` are unique across the catalog.

`hub_pipeline_tag` carries the task Hugging Face gives the model (`pipeline_tag`, verbatim, or null); `kind` is the view derived from it, and `kind_source` names the evidence where it is not.

**How a successor is proven in the shipped catalog** (decided 2026-09-25). A `successor` names one
concrete model of the same publisher that is itself a `latest` row of the catalog, and the comment
above the row names the publisher's page that says so -- the successor's model card naming this
very model as the one it replaces, next to the collection that carries the successor as current --
with the day it was checked. A line the publisher carries as current is on its own no such proof: it
does not say that every model of the line before it was replaced. Neither is a model of the same
size or a benchmark table: the card of `Qwen3.8-27B` follows "the Qwen3.5 and Qwen3.6 series" and
compares itself with `Qwen3.6-27B`, but names no model as replaced, so both 27B rows keep age
`unknown`. `Magistral-Small-2509` has a successor: the card of `Mistral-Small-4-119B-2603` names it
as one of the previous Magistral models of the family it now carries. The one successor that is no
`latest` row is `EuroLLM-9B-Instruct`'s: the card of `EuroLLM-9B-Instruct-2512` calls itself an
improved version of it, but the publisher's collection lists both lines side by side.
`tests/test_catalog.py` holds the shipped file to both rules.

`load_catalog` raises `SchemaVersionError` for an unsupported version and `CatalogError` for a
file that does not read or validate.

### Published catalog (AP9-KA-L)

`modelroom/catalog_remote.py::current_catalog(transport, now, cache_dir, budget)` returns a
`CatalogChoice(catalog, origin, source)` and never raises: every fault ends in the choice between
the shipped catalog and the one in the cache. Only the guided run calls it, with `cache_dir` =
`<folder of the pointer file>/catalog` (`~/.modelroom/catalog`) and the run's shared request budget
("Shared request budget"). `MODELROOM_CATALOG=shipped` selects the shipped catalog: no request, no
cache file written.

**The cache is one file, one envelope** per channel: `catalog-v3.cache.json` for the default channel since 1.1.0
(`catalog-v2.cache.json` is the cache of release 1.0.0 and of the old channel; the schema major version is in the
name, so releases of different schemas never disturb each other) holds
`{"cache_version": 1, "checked_at": <UTC, whole seconds, "...Z">, "sha": <40 hex | null>,
"last_error": <text | null>, "catalog": <document of the channel's schema | null>}` and nothing else. The whole file is
written each time, a temporary file in the same folder and one `os.replace` ("Atomic writes"), so
content and `sha` never disagree; there is no lock, and when two runs write at once the last one
leaves a consistent envelope. An envelope that does not read, has another `cache_version`, an unknown
or missing key, a stamp without a zone, a `sha` that is not 40 lowercase hex characters, a `sha`
without a catalog (or the reverse), or a catalog that does not validate counts as absent.

**Freshness**: `age = now - checked_at`. `0 <= age < 24 h` asks nothing, even when the last attempt
failed. Exactly 24 h, more, or a `checked_at` in the future asks once more. An attempt is
`GET https://huggingface.co/api/datasets/jm-vis/modelroom-catalog/revision/main` (one request, `sha` of 40
hex characters), and, only when the `sha` differs from the envelope's, `GET
https://huggingface.co/datasets/jm-vis/modelroom-catalog/resolve/<sha>/<file of the channel>`
(`catalog-v3.json` for the default channel, `catalog-v2.json` for the old one), which answers a
`307` to the same host and then `200` (three requests in all). `X-Repo-Commit` has to equal `sha`; the
body has to validate as the channel's schema with all its limits. The same `sha` only renews `checked_at`. **Every
fault is the day's attempt**: the envelope is written with the new `checked_at`, the old `sha` and
`catalog`, and the reason in `last_error`. The one exception is an attempt that fails at the budget
before its first request: it is no attempt and writes nothing. When the cache folder cannot be written,
no stamp is kept, so every run makes its one attempt (at most three requests, booked against the shared budget)
instead of one a day; the choice stays the same.

**Hop limit and answer size**: the revision request allows no redirect
(`RedirectingTransport(..., max_hops=1)`), the file request exactly one (`max_hops=2`), so an attempt
is never more than three requests whatever `MAX_REDIRECTS` says. A hop that does not stay on `https://huggingface.co` is
refused before it is sent (it is booked, not sent). The revision answer may be 64 KiB, the file 1 MiB.

**Choice**: the candidates are the fresh catalog (when the attempt succeeded), the one in the
envelope and the shipped one; the latest `published` day wins, a missing day is the oldest, and a tie
goes to fresh, then envelope, then shipped. The same rule applies offline. `origin` is
`catalog: published 2026-10-02` or `catalog: shipped 2026-09-29`, followed, after a fault, by `(the
published one did not read: <reason>)`, or `... did not read on <date>: <reason>)` for a reason an earlier
run stored. A reason is one line of printable ASCII cut to 97 characters (`search_release._note`).
Step 2 shows the line as the first note of the search.

**Integrity** is the `sha` in the URL and in `X-Repo-Commit`; it does not protect against someone who
controls the dataset. The protection against that is validation, the limits, the maintainer's daily
comparison of the published file with the Hub, never fetching a URL a downloaded file names, and the
switch above.

**Maintenance guarantees**: the background run has no publication path (it scans and verifies, nothing
else). An approval binds to the SHA-256 of the proposal that was shown and to the numbers named. The
upload sends the bytes of the recorded commit, verified against the dataset afterwards.

**Two channels (AP9-KO, V3).** The dataset holds one file per schema: `catalog-v2.json` for schema 2, which
release 1.0.0 reads and which stays exactly as it was, and `catalog-v3.json` for schema 3, the catalog with
findings ("Package findings and the load check"). `catalog_remote.Channel` names a channel's file, its cache file
(`catalog-v2.cache.json`, `catalog-v3.cache.json`: the two never share one), its schema, its size limit (1 MiB,
4 MiB) and its reader; `current_catalog(..., channel=DEFAULT_CHANNEL)` asks the channel it is given, with the
same two or three requests, the same freshness and the same choice. A file of another schema than its channel's is a
fault like any other (`... schema_version 2 ...` on the new channel, `... schema_version 3 ...` on the old one).
`DEFAULT_CHANNEL` is the new channel since release 1.1.0 (P5; it was one line). `catalog_remote.offline_catalog(cache_dir,
channel)` is the same choice without a request, for the commands that ask no one (`render`, `pull`); without a channel
it reads the cache of the new one. A cache of release 1.0.0 (`catalog-v2.cache.json`) is neither read nor changed by
release 1.1.0. Release 1.0.0 asks `catalog-v2.json` only and never meets `catalog-v3.json`
(`tests/test_ko_p5_channel.py` runs the code of 1.0.0 against a dataset that holds both files).

**Order of a release (P5).** The dataset's `catalog-v3.json` is published **first** (`scripts/catalog_publish.py`
makes and verifies both files of the dataset in one upload, "A publication" in `docs/runbooks/catalog-publishing.md`),
**then** release 1.1.0. A 1.1.0 that meets a dataset without that file, or with a file that does not validate as schema
3, is not broken: the attempt is the day's attempt and the run says `(the published one did not read: <reason>)`. The
choice is then between the shipped catalog and a valid cache of the new channel (`catalog-v3.cache.json`; a newer valid one
is kept). There is **no fallback to `catalog-v2.json`**, and a cache of release 1.0.0 is not read, so until the file is
published such a run loses the findings and also the ordinary updates of the published table (models, successors,
packages) that are newer than the shipped catalog. Publishing the release before the file costs that, not the run. The shipped
`modelroom/catalog.json` stays schema 2 and unchanged until the first approved finding makes it schema 3 (the decision of
P5, below the table "The forms and their versions"); the working catalog of a scan with `--packages` is schema 3.

### CatalogModel

```json
{
  "hf_repo": "acme/Nova-7B",
  "latest": false,
  "latest_source": null,
  "latest_checked": null,
  "successor": "acme/Nova-7B-2512",
  "ollama_base": "nova",
  "ollama_tag": "7b",
  "source": "https://huggingface.co/acme/Nova-7B-2512",
  "kind": "text"
}
```

Schema 3 adds two fields to a row (AP9-KO), both left out of the example because their defaults are the
fact `unknown`: `gguf_checks` (a list of `PackageCheck`, at most 200, default `[]`) and `setup_note` (a
`SetupNote` or `null`, default `null`). A schema-2 document carries neither key, not even at its default.

### CatalogFamily

```json
{
  "name": "nova",
  "publisher": "acme",
  "models": [
    {
      "hf_repo": "acme/Nova-7B",
      "latest": false,
      "latest_source": null,
      "latest_checked": null,
      "successor": "acme/Nova-7B-2512",
      "ollama_base": "nova",
      "ollama_tag": "7b",
      "source": "https://huggingface.co/acme/Nova-7B-2512",
      "kind": "text"
    },
    {
      "hf_repo": "acme/Nova-7B-2512",
      "latest": true,
      "latest_source": "https://huggingface.co/collections/acme/nova-2512-0123abcd",
      "latest_checked": "2026-09-23",
      "successor": null,
      "ollama_base": null,
      "ollama_tag": null,
      "source": "https://huggingface.co/acme/Nova-7B-2512",
      "kind": "text"
    }
  ]
}
```

### Catalog

```json
{
  "schema_version": 1,
  "families": [
    {
      "name": "nova",
      "publisher": "acme",
      "models": [
        {
          "hf_repo": "acme/Nova-7B",
          "latest": false,
          "latest_source": null,
          "latest_checked": null,
          "successor": "acme/Nova-7B-2512",
          "ollama_base": "nova",
          "ollama_tag": "7b",
          "source": "https://huggingface.co/acme/Nova-7B-2512",
          "kind": "text"
        },
        {
          "hf_repo": "acme/Nova-7B-2512",
          "latest": true,
          "latest_source": "https://huggingface.co/collections/acme/nova-2512-0123abcd",
          "latest_checked": "2026-09-23",
          "successor": null,
          "ollama_base": null,
          "ollama_tag": null,
          "source": "https://huggingface.co/acme/Nova-7B-2512",
          "kind": "text"
        }
      ]
    }
  ]
}
```

### Relation check

`check_relation(tags, card_data, hf_repo)` decides whether a Hugging Face repo declares itself a
quantization of exactly the configured base model. It reads the repo's `tags`
(`base_model:<repo>`, `base_model:<relation>:<repo>`) and `cardData` (`base_model`,
`base_model_relation`), in this order:

1. `metadata_conflict`: a field has a shape the Hub does not publish (`tags` that is not a list,
   `cardData` that is not an object, a tag that is not a string, `base_model` that is not a string or a list of strings, a relation that is not a
   string), tags and card name different bases, a relation tag names a repo that is not a
   declared base, or tags and card state different relations;
2. `base_model_tag`: the declared bases are not exactly `{hf_repo}`;
3. `relation_unknown`: no relation declared, or one this check does not know;
4. `derivative`: `finetune`, `adapter` or `merge`;
5. `quantized`: the only pass.

A pass allows at most `from_metadata` together with the other provenance rules; only a human
`Approval` makes a package `approved`. Search and fetch use the same function.

### Fit with profile v2

`compute_fit_v2(profile, package, base_model, scenario, machine_config)` is fit contract v1's
formula and classes behind a gate: `fit_block_reason(profile)` first (GPU state, a profile
migrated from schema 1, unknown RAM; a llmfit deviation is no reason since 2026-09-27), then the package rules (a complete
`gguf` build, every weight file with a size). An architecture the formula cannot read is the
one case that is **not** `unknown` here: the package is computed from its size instead,
`basis: "size"` ("Fit from size" above, decided 2026-09-24). Since 2026-09-25 the KV cache counts
once per `scenario.requests` on both bases, and more than one request is no longer `unknown`
("Fit contract v1", "Parallel requests"); every fit carries its `requests`. VRAM is `0` unless
`gpu_state` is `measured` or `entered` (`fit.profile_memory`); `unified_memory` computes as one
pool of its own ("Unified memory"). The RAM
pool is `ram_physical_gib` (a limit is a note, never an input). The context is
`scenario.context_requested` for every package, never the package's own `default_context`, so
`context_assumed` is `false`. `base_model` (its architecture) and `machine_config` (its
reserves) are inputs because the formula needs them.

The window of the model (AP9-UX, ADR 0009): after the gate and the package rules, a package whose
base model declares `architecture.max_context` smaller than `scenario.context_requested` is
`unknown` with the reason `context beyond the window of this model (32k)` and is set aside as not
covered; a window nobody knows (`None`) checks nothing. So the size scale (`count_fitting`) and the
ranking count the same packages beyond a window.

### Ranking rule

`rank_packages(entries, measurements, scenario, *, models=None)` orders one device's packages as a
total order that does not depend on input order (ADR 0009, 2026-09-28; it replaces the rule of
fit class, speed and `QUANT_ORDER` ascending). `models` maps a `base_model_hf_repo` to
`ranking.ModelFacts(parameters_b, legacy)`; a model it does not name -- or `models is None` -- has
an unknown parameter count and is not `legacy`. Seven steps:

1. **pool**: graphics memory (`Fit.mode` `gpu`, `gpu_split`) before system memory (`cpu_gpu`, `cpu`);
2. **`legacy` last**: every model that is not `legacy` before every `legacy` one; `latest` and an
   unknown age rank alike (GF 2026-09-28);
3. **parameters**, larger first; an unknown count behind every known one;
4. **compression**: `QUANT_ORDER` reversed, the least compression first; an unknown quantization
   behind every known one; `F16`, `BF16` and `F32` are one step, the smaller weights first;
5. **measured before unmeasured** (group 0 before group 1) -- a tie-break between two builds of one
   compression step only; the speed is no part of the key;
6. larger weights first;
7. package identity.

`ample`, `fits` and `tight` rank alike and keep their word in the row; `unknown` and `no_fit` are set
apart as before. "Best" means the same model, less compressed: modelroom rates no model family
against another. A measurement counts (`group_zero_measurement`) when it is protocol `v1` or `v2`, `valid`,
comparable, for the same `context_requested` and the same `requests` as the ranking and for
exactly this package content (manifest digest, or repo, revision and a weight file's digest); the
newest one wins. A measurement always runs one request, so a ranking of more requests has no
group 0; the measurement stays `comparable` and counts again in a ranking of one request.
Where a measurement would count but for `requests` alone, `RankedPackage.measurement_note` says
so -- `measured with 1 request, ranking assumes N` -- and the render hands it on as the note of
that row (code `measured_with_one_request`); it is never written into the measurement file
(decided 2026-09-25). `unknown` fits are set aside as not covered, with their reason; `no_fit`
fits are listed apart (field `no_fit`) and never ranked; a computed fit for another context or another number of
requests than the ranking's is a caller error. `Ranking.top` is the first 10. The rule states
facts in order; it recommends nothing.

**The rows of a result** (`ranking.shown_rows(ranking, per_model)`). One base model among the
eligible packages of the (narrowed) snapshot is **case 1**: the first `TOP_LIMIT` (10) packages under
the whole key. Several are **case 2**: one row per model, its best package being its first under the
whole key (`ranking.best_per_model`; the pool first, so a `Q4_K_M` on the card beats a `Q8_0` in
system memory of the same model), the rows in the order of that key. `render.build_render_document`
decides the case from the snapshot, never from the selection, so `modelroom render` decides alike.
Behind the rule's rows every **measured** package of this machine the rule leaves out is appended, in
the order of the key: any measurement of exactly this package content on this machine -- a counting
one (group 0), one that fails on `requests` alone, or one that counts nowhere here for another
reason (another context, not comparable, not valid; `RankedPackage.measured_other`, Codex F3). Such a
last row stays in group 1 with no speed, and its note says why the number does not count (code
`measured_not_comparable`: `measured at context 8k, this ranking's is 16k.`, `measured, not comparable
(<reason>).`, `measured, not valid (<reason>).`); the `speed` line under the table names it the same
way instead of the advice to measure. A measured package the fit sets aside (`no_fit`, `not
covered`) is no ranked package and gets no row (ADR 0009, "Limits"). No row of the rule moves for a
measurement. `ShownRow(ranked, why, place, of)`: `why` is `rule` or
`measured`; `place` of `of` is the place among all ranked packages (case 1) or among the packages of
that model (case 2).

The row note keeps its form and code and gains one sentence (`ranking.row_sentence`), with the fact
it names: case 2, a rule row, `ranking.best_per_model` -- `Best of 12 packages of this model that
fit.`, or `The only package of this model that fits.`; an appended row,
`ranking.shown_for_measurement` -- `Shown for its measurement; the rule puts it at place 14.` (case
1) or `... at place 2 of this model's 2.` (case 2). `RankedEntry.rank` is the number of the row
shown (1, 2, 3 ...); `ranked_total` stays the number of every ranked package. `ranking_rule` is
`RANKING_RULE` (case 1) or `RANKING_RULE_PER_MODEL` (case 2); both name the appended measurements.
The card counts measurements from `ranked` as before, so an appended row counts; the install line
follows `ranked[0]`, never an appended row.

The key under a table (`ranking_words.key_pieces`) carries the fit words, `best: the least
compression of the same model that fits` and `modelroom rates no model family against another`, and
in case 2 `one row per model, its least compression that fits` and `a run with one model shows its
first 10 packages, then the measured ones` -- one wording for the guided mode and `modelroom render`. The `shown` line names the
appended rows (`#3 is shown for its measurement, not by the rule`, `#3–4 are shown for their
measurements, not by the rule`); the Markdown file says `Showing 3 rows of 24 ranked packages: one
row per model, then 1 measured.` (`ranking_words.showing_line`).

The rule reads the pool and not the fit's `basis`, so a package judged from its architecture and
one judged from its size stand mixed. That is deliberate: the basis is how a fit was reached, and a
reader sees it in the row itself ("Fit from size").

### Run selection

`selection.RunSelection(machines, models, release, parameters)`, frozen, in memory only (field
`GuidedRun.selection`): the `[machines.<name>]` step 1 of this run chose -- measured, imported,
entered, or a profile of the folder marked as it is -- and the base models step 2 chose, with the release (`SearchHit.age`) and the
parameter count the list showed. `selection.narrow(config, snapshot, selection)` builds a filtered
`Configuration` and `Snapshot` through their constructors; an empty `machines` or `models` is an
empty choice, not "no filter" (Codex F1): a run that chose no model ranks none of the folder's models
and says so -- `this run chose no model, so the result ranks none` -- and writes that empty result.
A run that chose no machine -- nothing marked in an answer file, or only an `import` that failed --
names none, says `this run chose no machine, so the result ranks none` and measures nothing in step
4; no machine it did not choose is added, the one the pointer binds this folder to included (Codex
r2; this drops the fallback to the bound machine the AP9-UX brief had, a decision of the steering).
At a terminal `Enter` alone takes the pointer row, so a terminal run always chooses a machine. Only
`modelroom render`, which passes no selection, shows the whole folder.
`selection.model_facts(selection)` hands the
rule `ModelFacts(parameters, release == "legacy")`. `render_with_config(..., selection=None)` narrows
first, then ranks with those facts; the card, the tables and `docs/models.md/json` name only the
selection. Steps 3 and 4, the balance of step 2 and the card of an empty step 5 read the same
selection. Nothing of it is written: `modelroom.toml` keeps every family and machine, the fetch
fetches them all, and a later `modelroom render` shows the whole folder again (GF 2026-09-28).

**One rule for the parameter count** (`parameters.parameter_count(stated, name)`): the number the
Hub states, else the one the name states (`parameters.parameters_from_name`), else `None`. The list
of step 2 calls it with `SearchHit.parameters_b`, and `RunSelection.parameters` carries exactly that
number to the result, so the list and the result of a run never order differently; `modelroom render`
calls it with `BaseModelSpec.parameters_b`. Two models of one size by their names may therefore stand
otherwise in a later `render`, where the Hub names two numbers -- one function, two inputs. A render
without a run knows no release: no model is `legacy` there.

### SearchHit

One Hugging Face repository from the search, as the selection list shows it. Every field but
`repo` may be unknown (`None`, or the literal `unknown`). `publisher_status` is `publisher`,
`listed packager`, `other` or `unknown`. A hit is `resolved` only when exactly one publisher base
model is proven (relation `quantized`, publisher per catalog), so a resolved hit has
`base_model` equal to `[resolved_base_model]` and `base_model_relation` `quantized`; an
unresolved hit carries its
`unresolved_reason` (a relation status or `publisher_unknown`) and gets neither a fit nor an
Ollama name. `successor` is set exactly when `age` is `legacy`. `release_basis` says where `age`
comes from (since 2026-09-25): `stated` by the publisher's repository or the catalog
(`decide_age`), `computed` from the version numbers of the family ("Computed release"), or `None`
for an `unknown` age and for a hit built without the field, which reads as `stated`. The coupling
runs one way: `computed` needs an `age` other than `unknown`. `repo_created_at` is shown as
"repo created", never as the model's release date. `downloads` is how often the Hub says the
repository was downloaded (`ge=0`, `None` when the answer carries no count): an indication of what
many people take, **never a rank** -- the ranking rule alone orders the result.

`found_as` (AP9-KO, V8) is `package` -- every hit until now, the example below -- or `direct`: the
publisher's own model, found by its catalog row and read from its own Hub answer. A direct find is
`resolved` as itself (`resolved_base_model` is its `repo`), its owner is a `publisher`, and it declares **no**
base model and no relation (`base_model` `None`, `base_model_relation` `unknown`): a quantization relation
is never invented for it. `age`, `successor` and `ollama` hold as for any hit.

```json
{
  "repo": "packager/Nova-7B-GGUF",
  "publisher_status": "listed packager",
  "base_model": [
    "acme/Nova-7B"
  ],
  "base_model_relation": "quantized",
  "resolved": true,
  "unresolved_reason": null,
  "resolved_base_model": "acme/Nova-7B",
  "repo_created_at": "2026-03-02T10:00:00Z",
  "downloads": 68445,
  "parameters_b": 7.6,
  "license": "apache-2.0",
  "age": "legacy",
  "successor": "acme/Nova-7B-2512",
  "release_basis": "stated",
  "ollama": "nova:7b"
}
```

### ModelChoice

One line of the selection list of step 2: one resolved base model, with every repository of the
search that packages it (`modelroom/guided_models.py`, decided 2026-09-24). `repos` are those
repositories as the search answered them, in the order its groups were asked, so the two the
fetch works through can be picked from it; `packagers` are their accounts in the same order, for
the column that shows them. `downloads` is the sum over the repositories that state one (`None`
when none does), `parameters_b` what a repository states, else what the name says, else `None`,
and `fit` is the fit from that size ("Fit from size"), or `unknown` with `machine not measured`
or `parameter count unknown`. `name` and `publisher` are the two halves of `base_model`. `ollama`
is the first Ollama name a repository names, else the pair the configuration holds for the base
model (`model_choices(..., configured_ollama=...)`, since 2026-09-25), else `None`. `age` and
`release_basis` come from one and the same repository -- the first with an `age` other than
`unknown` -- so a row never shows one hit's age with another hit's star; `computed` needs a known
`age` here as well.

```json
{
  "base_model": "acme/Nova-7B",
  "name": "Nova-7B",
  "publisher": "acme",
  "repos": [
    {
      "repo": "packager/Nova-7B-GGUF",
      "publisher_status": "listed packager",
      "base_model": [
        "acme/Nova-7B"
      ],
      "base_model_relation": "quantized",
      "resolved": true,
      "unresolved_reason": null,
      "resolved_base_model": "acme/Nova-7B",
      "repo_created_at": "2026-03-02T10:00:00Z",
      "downloads": 68445,
      "parameters_b": 7.6,
      "license": "apache-2.0",
      "age": "legacy",
      "successor": "acme/Nova-7B-2512",
      "ollama": "nova:7b"
    }
  ],
  "packagers": [
    "packager"
  ],
  "downloads": 68445,
  "ollama": "nova:7b",
  "age": "legacy",
  "parameters_b": 7.6,
  "fit": {
    "fit_class": "fits",
    "mode": "gpu",
    "need_gib": 6.3,
    "weights_gib": 4.25,
    "kv_gib": 1.125,
    "pool_gib": 10.94,
    "reserve_gib": 1.0,
    "context": 8192,
    "context_assumed": false,
    "basis": "size",
    "reason": null,
    "requests": 1
  }
}
```

### Requirement

What one package needs for a scenario, `computed`, never measured: the reverse calculation.
`kv_gib_total` is `kv_gib_per_request x scenario.requests`; `need_gib` is fit v1's formula with
that total (`weights_gib x 1.10 + kv_gib_total + 0.50`); `ample_reachable` is
`false` in `cpu` mode, where fit v1 caps at `fits`. It carries no speed statement. `package_identity` is the
package identity triple.

```json
{
  "origin": "computed",
  "package_identity": [
    "huggingface",
    "packager/Nova-7B-GGUF",
    "Nova-7B-Q4_K_M.gguf"
  ],
  "scenario": {
    "context_requested": 8192,
    "context_origin": "default",
    "kv_type": "f16",
    "kv_type_assumed": true,
    "requests": 4
  },
  "mode": "gpu",
  "weights_gib": 4.5,
  "kv_gib_per_request": 1.0,
  "kv_gib_total": 4.0,
  "reserve_gib": 1.0,
  "need_gib": 9.45,
  "ample_reachable": true
}
```

### Note

One plain-language note next to a package, a machine, a measurement or a ranking, built only
from facts: `facts` names the fields it was derived from, `origin` is their provenance, `text`
is at most 240 characters.

```json
{
  "code": "cpu_caps_at_fits",
  "subject": "package",
  "origin": "computed",
  "text": "Runs in system memory only, so the best possible rating on this machine is 'fits'.",
  "facts": [
    "fit.mode",
    "fit.fit_class"
  ]
}
```

### Shared request budget

`run_fetch(..., budget=...)` takes either a request count or a `RequestBudget` object. A
`BudgetedTransport` built on the same `RequestBudget` books against the same count, so a caller
that has already spent requests (the search, the resolution) passes what is left into the
fetch; `FetchResult.request_used` and `request_budget` report that shared budget's totals. `BudgetExhaustedError`
and the area outcome `budget exhausted` are unchanged.

`modelroom/search.py::DEFAULT_GUIDED_BUDGET` is `150`: one guided run -- search, resolution,
successor lookups, tree pages and the fetch -- shares that many requests unless the caller
passes its own `RequestBudget`. It was `60` until the search asked one request per account
(decided 2026-09-24): measured live, `qwen` with the owner filter on spends 47 requests before the
selection list is shown -- 7 account pages and one age lookup per distinct resolved base model
(40 of them for 80 resolved hits) -- and 60 left the fetch of two chosen repositories `incomplete
(budget exhausted)`. `run_fetch`'s own default stays `400` for a plain `modelroom fetch`, which
has no search in front of it.

**The vLLM tensor search of step 2 (AP9-SV2-P, P1, `modelroom/tensor_search.py`) only spends from
this same shared budget when at least one chosen machine has vLLM** (`guided_context.vllm_among_chosen`,
the same gate as `build_tensor` above): a run without one costs no extra request at all, so an
ordinary Ollama run and its goldens are unchanged. When it runs, every base model this
very search resolved (`SearchOutcome.hits`, not only the ones later picked -- the search step has
no later hook to act on a choice from) is asked once per account of `tensor_search.TENSOR_PACKAGERS`
plus the base model's own owner (`tensor_search.tensor_packagers`): one request each,
`author=<account>&search=<base name>&filter=safetensors&sort=downloads&direction=-1&limit=20`. A
hit is kept only when `relation.check_relation` proves it `quantized` from this exact base model;
at most `tensor_search.MAX_TENSOR_REPOS_PER_MODEL` (4) are kept per model, by downloads, the rest
named as a count (`"and N more vLLM builds not fetched"`), never silently dropped and never
fetched. `BudgetExhaustedError` ends the search where it stands -- the accounts already asked keep
their candidates. A repository found is written into that base model's `repos` only once the
person chooses that base model, in the same write as the chosen GGUF hits
(`search_apply.apply_tensor_repos`, the same write-back shape `apply_hits` uses for a GGUF hit,
including registering the base model's own owner under `publishers` when neither existed yet), so
the next fetch of this folder reads it through the ordinary `package_targets` mechanism and pays
for it there (the paragraph above, "Request budget") -- no new persisted form.

### What was typed, read as one of three things

**The input is parsed before anything is asked** (`modelroom/search_word.py`, decided 2026-09-25).
Until then every answer went to the Hub as `search=<text>` word for word: a repository id
(`unsloth/Qwen3.5-9B-GGUF`) found nothing, a name with blanks (`qwen 3.5 9b`) found nothing, and an
empty answer was the error `no model name to search for`. `parse_search(text)` returns a
`SearchWord` of one of three **forms**:

| Form | When | What the search does |
|---|---|---|
| `id` | exactly one `/`, both halves Hub-shaped (`[A-Za-z0-9][A-Za-z0-9._-]*`) | one request for that repository ("A typed repository id") |
| `words` | anything else that leaves a word | two pages per account, plus the open lists without the filter |
| `empty` | nothing typed, or nothing but filler | one page per current model of the catalog ("No search word") |

The words are the text cut at blanks, `-`, `_` and `:`, lower case, without the **filler words**
`gguf`, `model`, `models` and `the`. `text` is what was typed with the surrounding blanks stripped
-- what `search.json` and the screen call the word. For an `id` the typed spelling is kept
(`repo_id`, a repository id is case sensitive) and `words` holds the words of its **name** half,
which is what the search falls back to when that repository holds no GGUF file.

**One word goes to the Hub, all of them are held against the answer.** The Hub's `search=` takes one
string and a longer one finds less, so `SearchWord.hub_word` is sent: one word alone as it is (which
is what every recorded answer of the test suite was taken with), and for more than one word the
longest **letter run** -- `qwen 3.5 9b` and `Qwen3.5-9B` both ask `qwen`. The other words then filter
the answer locally: a repository is kept only when every word is part of its id with the separators
removed (`qwen3.59b` in `unslothqwen3.59bgguf`). A **one-word** search is never filtered
(`SearchWord.filters_locally`): that is the search this package has always made, and its recorded
answers carry repositories whose id holds the word only in the owner half
(`mistralai/Ministral-3-14B-Instruct-2512-GGUF` under `mistral`). What a group dropped that way is
counted and said: `unsloth 27, 1 of them already listed, 25 of them other models`.

**A word the catalog does not know leads to candidates.** When no family of the catalog matches the
input -- not by name, not by publisher account, not by one of its model ids -- `difflib.get_close_matches`
at cutoff `0.75` runs over the family names, the publisher accounts, the model names and the
**letter runs** of those names (the runs are what makes `qwn` reach `qwen`, from `Qwen3.5-9B`). At
most five candidates, each once, in the order the words were typed. `SearchOutcome.candidates`
carries them and the guided mode offers them as a list (see "Guided mode", step 2). A word the
catalog *does* know gets none: a correct word with no answer today is a fact about the Hub, not a
spelling mistake. An empty answer gets none either, and neither does a typed id the Hub answered
for -- but an id that **fell back** to the word search is a word search from there on, so the words
of its name half can lead to candidates like any others (second-model round, 2026-09-25).

### Search over the Hugging Face API

**Two pages per account, and open lists only without the filter** (decided 2026-09-24, the second
page 2026-09-25). Until 2026-09-24 the search made one request for the 50 most recently created GGUF
repositories matching the word, and the positive list of packager accounts only *classified*
whatever that answer happened to contain. Measured in an empty folder with `qwen` and with
`deepseek`: the answer was 50 third-party uploads of the last few days and not one repository of
`unsloth`, `bartowski` or `Qwen` -- those accounts are older and fall out of the window, so nothing
was selectable, twice. The account steers the request now.

**One sort order per account was not enough either.** Measured live 2026-09-25:
`author=unsloth&search=qwen&sort=createdAt&limit=20` answers with twenty repositories created after
2026-05, and `unsloth/Qwen3.5-9B-GGUF` (created 2026-02-28) is not among them at all; the same
account's `sort=downloads` page carries it in fourth place. So an account is asked **twice**, its
newest twenty and its twenty most downloaded, and the two answers are **one group** under that
account's label -- a reader asks "was my account asked" once, and the sort orders are this package's
business. `SearchGroup.pages` says how many requests that label cost.

`modelroom/search_pages.py` builds the six forms, over the transport this package already has --
there is no Hugging Face SDK dependency:

```
account, newest: GET https://huggingface.co/api/models?author=<account>&search=<name>&filter=gguf
                     &sort=createdAt&direction=-1&limit=20
                     &expand=cardData&expand=createdAt&expand=downloads&expand=safetensors&expand=tags
account, most
downloaded:      GET https://huggingface.co/api/models?author=<account>&search=<name>&filter=gguf
                     &sort=downloads&direction=-1&limit=20&expand=... (the same five)
most downloaded: GET https://huggingface.co/api/models?search=<name>&filter=gguf&sort=downloads
                     &direction=-1&limit=20&expand=... (the same five)
newest:          GET https://huggingface.co/api/models?search=<name>&filter=gguf&sort=createdAt
                     &direction=-1&limit=10&expand=... (the same five)
catalog page:    GET https://huggingface.co/api/models?search=<model name>&filter=gguf&sort=downloads
                     &direction=-1&limit=10&expand=... (the same five)
typed id:        GET https://huggingface.co/api/models/<owner>/<name>?expand=... (the same five)
```

`expand` is repeated once per field, not sent as a list, and `expand=downloads` is required as soon
as `expand` is set at all -- without it the field is simply absent (measured 2026-09-24, both forms
`HTTP 200`). `HF_ACCOUNT_LIMIT` is `20`, `HF_MOST_DOWNLOADED_LIMIT` is `20`, `HF_OPEN_LIMIT` is
`10`, `HF_CATALOG_LIMIT` is `10`. One request answers everything the
resolution needs of that page, so no hit costs a request of its own. Measured against the live API
2026-09-23 and 2026-09-24: an entry carries `_id`, `id`, `createdAt` and `tags`, most of them a
`cardData`, few a `safetensors` (a GGUF repo usually has no tensor index), and none an `author`,
`sha` or `license` of its own -- so `parameters_b` and `license` are commonly `unknown`, and
`expand=tags` is what makes the relation readable at all.

**Which accounts are asked, with the filter on and off alike** (`search_pages.search_accounts`):

1. every **publisher account of a matching catalog family** first -- a family matches when the word
   is part of its name, of its publisher account or of one of its model ids, compared without
   regard to case and with the word stripped of surrounding blanks. So `qwen` asks `Qwen`,
   `deepseek` asks `deepseek-ai`, and `r1` asks the account of the family that carries an
   `...R1...` model. Each account once, in catalog order;
2. then the **positive list** in its order (`config.packagers`, else `DEFAULT_PACKAGERS`). An
   account on both lists is asked once.

Age plays no part in any of it: each account answers with its own repositories, however old they
are. When no catalog family matches the word, the summary says what *was* asked instead -- two
shapes, because that differs with the filter: `no publisher in the catalog matches 'foo'; only the
listed packagers were asked` with the filter on, and `... the listed packagers and the two open
lists were asked` with it off.

**The two open lists exist only with the filter off** (`run_search(..., open_pages=...)`, the guided
mode passes `not filtered`): the twenty most downloaded and the ten newest GGUF repositories for that
word, asked after the account pages, so a fresh fine-tune under an account nobody listed is visible
at all. Twenty for the downloads page since 2026-09-25: with the filter off, what many people take is
the only place a model of an account nobody listed shows up, and ten of them are mostly one family's
siblings. Switching the filter off **extends** the list, it never replaces it -- the account groups
stay. A repository more than one page answers with is listed **once**, in the first group that had
it; the open list still says how many its page held: `most downloaded 10, 3 of them already
listed`.

**A typed repository id** (`search_pages.typed_plan`, decided 2026-09-25) is **one** request for that
repository, shown under the label `typed`, and it is the whole search: the person named the
repository, so no account page could add to it. Two things about it differ from a word search:

- the **publisher gate does not apply** (`run_search(..., publisher_required=...)`, `False` for the
  typed page whatever the owner filter says). `publisher_unknown` is a preference for the catalog's
  publishers, and someone who types a repository has already said which account they want. Its owner
  class stays what it is (`other` where neither list holds the account), and the owner filter of the
  selection list does not hide it either (`guided_models.hit_reason(..., typed=...)`);
- **without a GGUF file** the search says `<id> holds no GGUF file` and runs the normal word search
  over the id's **name** half. The GGUF file is read off the repository's own `gguf` tag -- the same
  fact `filter=gguf` selects by -- so this stays one request. An answer the Hub does not give, and an
  answer that names another repository (the transport follows redirects, and a moved repository is
  answered with the one it moved to), leave `<id> is no repository this search could read` and the
  same fallback. The input is never simply refused.

**No search word** (`search_pages.catalog_plan`, decided 2026-09-25) is a valid answer and means
"show me what fits this machine": one page per model the catalog calls `latest`
(`search_word.latest_model_names`, 22 of them as the catalog stands), each its ten most downloaded
GGUF repositories, under the model's own name as the group label. The note is `no search word: the
catalog's current models, one page each`. This is the one plan whose length the input does not bound
and the fetch of a chosen model shares the same budget, so it **stops asking** once
`CATALOG_BUDGET_RESERVE` (20) requests or fewer are left and says `the request budget ended after <n> of <m>
catalog pages` instead of ending a run that has a choice to offer with `SearchError`. It is a stop
rule before a page and not a fence around those 20: the last page it is allowed to ask resolves its
own repositories, and each distinct base model of it costs an age lookup from the reserve as well, so
a run can come out of the search with fewer than 20 requests left (second-model round, 2026-09-25).
A fetch that then runs out says so and the run ends with exit `1`; nothing is written wrong.

**Every page costs one request of the shared budget.** A page that does not answer -- a request that
did not go through (`search_pages.TRANSPORT_ERRORS`: a timeout, a reset, a cut-off answer), a status
other than `200` (`429` and `5xx` included), a body that is no JSON, a body that is no list -- costs
**that page only** (AP9-KO P5, sweep p5-b1; measured live 2026-10-14, when single pages answered
`500`/`504` and two guided runs ended at one of them with exit `2`). The page spent its request and adds
no entry; the other pages are asked as always, and their hits are listed. It is never quietly left out,
because "unsloth has nothing for this word" and "unsloth did not answer" are different answers: the
screen shows one note per page in the words of the page's own error
(`search for 'qwen' (mradermacher): unexpected status 503; the search went on without this page`),
`search.json` lists it under `failed_pages` ("Search log"), its group line says `1 of 2 pages not
answered`, and a publisher with no hit and a page that did not answer is not said to have no GGUF
repository. Only a search in which **no** page answered ends with `SearchError` -- the first page's,
word for word as before -- since a search with no answer at all is no result; the guided mode then
ends with exit `2`. An entry with no repository id or one that is not a repository id still ends the
search with `SearchError` **naming the page**: that is no network or answer error of the Hub but a
broken answer. A budget that ends before the last page ends the search too and says how many accounts
have no answer (`... not made: budget exhausted; 6 accounts were not asked: deepseek-ai, unsloth, ...`).
A bug (any other exception) goes up.

**A repository is held against what earlier pages answered before it is resolved**, entry by entry,
so a page that carries the same repository twice lists it once and the duplicate costs no age
lookup. The answer is registry-controlled text, and nothing promises it holds each id once.

**`SearchOutcome` carries the groups and the flat list.** `groups` is a `list[SearchGroup]`
(`label`, `hits`, `page_size`, `page_full`, `already_listed`, `pages`, `filtered_out`) in the order
the pages were asked;
`hits` is the same repositories flat in exactly that order, so `apply_hits`, the selection list and
the answer file's `select` key are unchanged. It also carries `mode` (`word`, `id` or `catalog`),
`typed` (the repository id that was typed, else `None`) and `candidates` (see "What was typed").
`SearchOutcome.requests` is the number of **pages** asked, which is two per account since
2026-09-25, not the number of groups. `SearchOutcome.group_lines()` is the notes plus one
line per group, above the summary -- an account that answered with nothing keeps its line, because
"was my account asked at all" is the question this change exists to answer:

```
Qwen 1
deepseek-ai 0
unsloth 20 (page full: a more specific word shortens the list)
bartowski 0
most downloaded 10, 3 of them already listed
newest 10
```

A full **account** page is never a silent truncation; the note is an invitation to use a more
specific word. `page_full` is set for an account page only: an open list is ten by definition, so
the note would stand on every search and say nothing about the word. The summary line under it stays
one line: `"N repositories, M resolved, K unresolved, R requests, budget used/limit"`
(`SearchOutcome.summary_line`).

A hit becomes a `SearchHit` (above). It is `resolved` only when both hold:

1. `relation.check_relation` returns `quantized` for the one base the repository declares -- the
   same check the fetch uses, so the *relation* verdict is the same in both places. That is not a
   promise that this repository's packages will resolve later: provenance additionally requires
   the repository to be in the target set and every weight file's stem to match the base name, so
   a resolved hit can still yield `unresolved (file_stem)` packages. A hit that declares no base,
   more than one, or one that is not a repository id gets `base_model_tag`; the other statuses
   pass through as they are (`derivative`, `relation_unknown`, `metadata_conflict`);
2. **with the owner filter on**, the catalog knows that base model's account as a publisher
   (`Catalog.is_publisher`), otherwise `publisher_unknown`. With the filter **off** that gate is
   gone (`publisher_required=False`, decided 2026-09-25): a model of an account the catalog does not
   list resolves, its age is `unknown` for want of evidence and its owner class says `other` -- the
   catalog is a preference, never a verdict, and `publisher_unknown` then appears nowhere. The typed
   repository id is never gated at all (above).

An unresolved hit is shown with its reason and gets no fit, no age and no Ollama name -- it is
never silently dropped, and never treated as a package of the family. `search.py::owner_class`
labels an account `publisher`, `listed packager` or `other`; the same three labels are used for
a fetch target's owner. They place an account, they never judge it. `DEFAULT_PACKAGERS` is the
shipped positive list of packager accounts, the same one `modelroom.example.toml` carries.

An Ollama name comes from the catalog or from a line the user types
(`parse_ollama_entry("ollama: <name:tag>")`, which wins over the catalog); a resolved model with
neither is shown as `none known` (`ollama_label`). The search never infers one from a name.

`search.py::apply_hits(config, hits, catalog=...)` is the pure step
`configuration + resolution -> configuration`: each resolved hit's base model becomes (or joins)
a family, its account joins `publishers`, and the hit's own repository is added to that base
model's `repos` -- an owner-bound target, because the search finds repositories under accounts
the packager list does not name. Repeating the same hits changes nothing.
`write_configuration(path, config, now=..., expected_text=...)` renders that configuration with
`toml_writer.dump_toml`, proves it reads back unchanged *before* touching the disk, and writes
it under `modelroom.lock`, the same lock every other writer takes. `expected_text` is the text of
the read `config` was computed from, `None` when there was no file. Under the lock the file is
compared with it; a file another run wrote, created or removed since that read is never written
over but raises `ConfigChangedError` (a `ConfigError`: the file and `changed by another run since
it was read`), and the caller reads again ("Guided mode", "Every write reads the file again
first"). The lock is the one of the configuration that was read, so two runs that read the same
text take the same lock even when one of them moves `paths.state`. A new file has nothing read and
takes the lock of the configuration it writes, so two callers naming different state folders take
different locks -- and the new file is therefore **created exclusively** (decided 2026-09-25):
`state.publish_new_text` writes the text into a temporary file of its own (process id plus a random
part, opened with `O_EXCL`, mode `0666` before the umask like any other written file, LF line
endings) and publishes it with one system call that never replaces an existing file -- `os.rename`
on Windows, `os.link` and then removing the temporary name elsewhere. Of two runs creating the file
at once, the second gets `FileExistsError`, which `write_configuration` turns into the same
`ConfigChangedError`; the temporary file goes on every early end, Ctrl-C included. The comparison
before it stays and catches the ordinary case; for a file that was not there when the run read,
a read Windows refuses because another run is renaming the file into place that very moment
(`PermissionError`, measured 2026-09-25) counts as the same conflict. Outside the guarantee: a file system without hard
links, and a network drive that does not keep `link`/`rename` atomic. No backup is kept: the guided
mode has the user confirm the change first.

### Target set and areas

`config.package_targets(config, base_model)` is the one set of packager repositories a base
model claims: (`packagers` plus the base model's own owner) x (`<name>-GGUF` plus
`repo_aliases`), then the owner-bound `repos`, each entry once. Four places use exactly this
set, so none of them can disagree with another:

- the collision validator, which rejects a target two base models would claim;
- `fetch.run_fetch`, which groups the set by owner (`hf.py::target_owners`);
- `hf.fetch_hf_area`, which fetches the targets of its own owner;
- `provenance.decide_provenance`, which accepts a package only when its repository is in the
  set (`unresolved_reason` `repo_name` otherwise).

An owner that only a `repos` entry names is fetched too. That is the user naming one
repository, not an open door: the generated `<name>-GGUF`/alias names are never probed under
such an owner, only the exact `owner/name` written down.

**One area per (base model, owner), never per target.** `state.merge_snapshot` identifies an
area by `(source, base_model, owner)` and, when the area is `complete`, deactivates every
package of that area this run did not report. Two targets of the same owner published as two
areas would therefore deactivate each other's packages. So all of an owner's targets are worked
through one after the other inside one area, and the area is

- `complete` only when every one of them was worked through -- a target that does not exist
  (`401`/`404`) counts as worked through -- in which case its packages replace the area's
  previous stock;
- `incomplete` with the **first** genuine failure otherwise, in which case the owner's whole
  previous stock stays untouched, including the targets that were already read this run.

### Latest and legacy evidence

`search.py::decide_age(transport, catalog, hf_repo)` says whether a publisher model is the
current one, a superseded one, or neither. Positive evidence only -- a missing statement is
never evidence:

1. the publisher repository's own card carries a `new_version` (one request) whose target is a
   well-formed repository of the **same account**, is not the repository itself, and really
   exists (one request, and the answer has to **name that repository** in its `id`/`modelId` --
   the transport follows redirects and the Hub answers a moved repository's path with the one it
   moved to, so without that check a target under another account, or the repository itself, could
   satisfy the request and defeat both the account rule and the cycle check): `legacy`, with that
   target as `successor`. At most
   `MAX_SUCCESSOR_EDGES = 2` edges are followed, and the second one only to name a younger
   successor: the verdict stays `legacy` when that second edge is missing, invalid, a cycle or
   out of budget;
2. no `new_version` (absent, or not a string): the catalog decides -- `latest` when the catalog
   states it for this one model with its evidence, `legacy` with the catalog's `successor`,
   `unknown` when it states neither or does not list the model. A publisher repository that
   cannot be read at all is treated the same way: it said nothing;
3. a `new_version` whose target is malformed, under another account, the repository itself, or
   unreachable -- and a budget that ends before the first edge can be checked: `unknown`. A
   broken pointer is not evidence for the catalog's statement either, so it never falls back to
   it.

`age` is a statement about the **base model**, so a packager's hit shows the age of the
publisher model it resolved to; an unresolved hit is always `unknown`. `decide_age` is called
once per distinct resolved base model in a search, not once per hit. `repo_created_at` is
labeled "repo created" and is never read as a release date or as an age. Every age this section
decides is `stated` (`SearchHit.release_basis`).

**Computed release** (`modelroom/search_release.py`, decided 2026-09-25, reworked by AP9-UX2 U5
after GF F3, ADR 0010). The list computes a status from the version numbers of one **line** of one
publisher's family instead of keeping a maintained list of successors. `run_search` does it once
every age lookup is done, without a request of its own, and rebuilds each such hit through
`SearchHit.model_validate`, so the coupling checks run. It reaches the list (`latest*`, `legacy*`)
and `search.json`, never the catalog, the configuration or the document.

**Since AP9-UX2 (these rules replace the ones below where they differ).**

- **Family, line, variation.** The **family** is the first word of the name (`qwen3.5` -> `qwen`); a
  name that starts with a size, a version or a date has no reading. A **line** is a size step with
  a name and a version series of its own, from the documented vocabulary `LINE_WORDS = {small,
  medium, large}` and `LINE_COUNTERS = {r}` (a word of that letter and a number, `r1`, is line `r`
  with the version `(1,)` where the name carries no other); the first such word of the name. Every
  other word is a **variation** (`base`, `instruct`, `mini`, `flash`, `vl`, `coder`, `next` ...;
  `instruct`, `it`, `chat` say nothing). New line words come only with a GF decision and a test.
- **Group** = `(publisher, family, line)`: a variation belongs to its version, so a higher version,
  whatever its publisher's variation, makes every variation of the lower one `legacy`; a line is
  not superseded by another line (`Mistral-Small-3.2` stays `latest` beside `Mistral-Large-3`).
- **Population** adds the unresolved repositories of a listed packager (`publisher_status ==
  "listed packager"`), read without `-GGUF`, when their family is the family of **exactly one**
  publisher of the population; they count under that publisher, prove a higher version and can be
  the successor (the packager's repository then). Ollama names without a Hub hit are not seen.
- **Successor, always one** where there is a candidate: of the maximal members, the same size, else
  the next larger, else the next smaller, a candidate without a size last (for a row of unknown
  size the largest known); then the same variation before another; then the fewest variations;
  then the name.
- **Stated against computed.** A stated `legacy` stays, always. A **catalog** `latest` gives way to a
  computed `legacy` only where a member carries a **strictly higher version number** (the catalog
  ages); a later date of the same version does not overrule it, while a computed status still reads
  the date. A silent catalog's `unknown`
  is computed. An `unknown` from a **fault** (a broken `new_version`, a budget) is computed too --
  `legacy` where a member stands over it, `unknown` otherwise -- and the fault is a note of step 2 of
  its own: `<name>: its own statement did not read (<fault>)`, cut to 97 characters.
- **Cycles choose only the successor.** Of the candidates in the order above, the first whose edge
  closes no cycle with the **stated** edges; where every one does, the first all the same and the
  note `<name>: the successor statements around it contradict each other`. No status falls for a
  cycle.

**Since AP9-KA-G (decided 2026-10-08; `modelroom/search_release.py`).** `line_key` reads 19 more catalog names. The
contract has three levels:

- **Token patterns**, exactly these and each only for a token no earlier class took (a reading that stood stays):
  `\d+k` (`200k`, `32K`, `4K`, `8K`) is read as a syntactic `k` attribute and ignored, with no reading as a context length
  or a resolution; `\d{3,4}p` (`720p`, `1080p`) is a resolution and ignored; `\d+bit` (`16bit`) joins the precisions and is
  ignored; `\d+p\d+[bmt]` (`1p3b`) is a size token and so a boundary, while `parameters_from_name` stays as it was (it does
  not read `1p3b`, and `OpenELM-1_1B` stays 1); a word of two letters or more with a dotted version,
  `[a-z]{2,}\.\d+(\.\d+)*` as a whole token (`flux.1`), is a word with a version: at the boundary it gives the family and
  the version, behind it the whole token is a variation (like `qwen3`); a number too long for `int` is no reading and no
  crash. `1M` and `7B` stay sizes; `1080i`, a three-digit number, `\d+d` and a dotted word of one letter (`r.1`) stay
  without a class; `r1` remains a plain word token and is read as a line counter.
- **The underscore between digits** (limit N): `A_B` is read as `A.B` only if all four hold: `B` is one or two digits, no
  unit; `A` is one or two digits or a word of two letters or more with digits, and no precision; nothing in front of `A` is a
  size, a date, a version or a word with a version; `B` does not start a month-year date with the part behind it. So
  `llama-3_1-8b` is (3, 1), `InternVL2_5` is (2, 5), `GOT-OCR2_0` is (2, 0); and `Nova-08_2025`, `Nova-BF16_8B`,
  `Nova-7B-Qwen3_5`, `OpenELM-1_1B`, `Nova-2_08-2025` and `Nova-08-2025-Qwen3_5` read as before.
- **Name composition**: a name with such a token gets a key only if the existing rules hold (a family word first, at most one
  version, at most one date). There is no promise that every name with a new token can be read. More than one version token
  stays rejected, including a trailing `v` revision (`sarvam-1-v0.5`, `sarashina2.2-...-v0.1`).
- **Promised cases**: the keys of the 19 names in `KEYS` (block `AP9-KA-G`) and the cases of the tests of that block. No
  promise for search hits outside the catalog. The rules of AP9-UX2 above are unchanged.

Not read in AP9-KA-G, by decision: a bare number (`apple/DFN5B-CLIP-ViT-H-14-378`), a dimension
(`mixedbread-ai/mxbai-embed-2d-large-v1`) and the trailing `v` revision; these five names stay in
`SHIPPED_WITHOUT_READING`. New token classes, like new line words, come only with a decision and a test. The readings
give a status computed under assumptions, not a statement of the publisher: a successor may change the task (SkyReels-V2
to V3-A2V, Wan2.1-I2V to TI2V-5B or T2V-A14B depending on the size of the hit).

**Since AP9-KA-H (decided 2026-10-09; `modelroom/search_release.py`).** Two more token forms are read, each only for
tokens no earlier class took:

- `\d+s` (`30s`, seconds of audio: `MERT-v2-30s`) is a syntactic attribute like `\d+k`: ignored, with no ordering effect, and
  as many of them as the name holds.
- Three tokens year `20\d\d`, month `01` to `12`, day `01` to `31` are **one** date of the form `iso` (`moondream-2b-2025-04-14`),
  tried before the single tokens, so `2001` to `2012` inside an ISO date never read as YYMM. `LineDate.form` has the new value
  `iso`; the order is unchanged: the version first, then with equal versions a date over no date, and two dates only when they
  have the same form (`iso` against `iso`; `iso` against `yymm` is not comparable).

A three-digit number (`kyutai/ovie-512`) stays without a class, so `378` and `735` stay without a reading. A scanner proposal
marks a new row whose name has no reading with `reading: false` (a mark for the reader, never a field of the row).
`unread_reason(name, size)` gives the reason a name has no reading.

**Until AP9-UX2**, as decided 2026-09-25 (kept for the record):

- **A name as a key** (`line_key`). The name half of the repository id is lowercased and split at
  `-` and `_`; each token takes the first class that matches, in this order: precision (`bf16`,
  `fp8`), number of experts (`16e`) and active size of a mixture (`a3b`, `a0.6b`), all three
  ignored; size (`27b`, `360m`, `2.4t`, `e4b`); a date as `YYMM` (`2512`, `YY` 20 to 39) or `MMDD`
  (`0528`), or a month and a year as two tokens (`08-2025`); a version (`3.5`, `v0.6`, a plain number
  only with one or two digits); a word with a version (`qwen3.5`, `smollm3`); a word (`coder`,
  `r1`). A token of no class -- `735`, `2025` alone -- makes the name **not interpretable**. The
  first token of the size, date, version or word-with-version classes is the **boundary**: the
  family is the publisher plus the words in front of it, without `instruct`, `it` and `chat`
  (`Mistral-Small-Instruct-2409` is a Mistral Small), and the word of a word-with-version
  boundary (`Qwen3.5` is family `qwen`, version `(3, 5)`); the version is the boundary's number, else
  the first version token after it; the date is the first date token. A second version or a second
  date makes the name not interpretable. The **variant** is the set of words after the boundary
  (a word with a version counts whole there: `qwen3` in `DeepSeek-R1-0528-Qwen3-8B`), without
  `instruct`, `it` and `chat` -- the instruction-tuned model is the ordinary one. The size is
  `parameters.parameters_from_name`, the package's one size rule, else the parameter count of a
  hit that is that very repository.
- **The order.** A name stands over another of its group when its version is higher (no version
  is lower than every version), or, with equal versions, when it carries a date and the other none
  (the undated name is the first release, the dated one its revision), or a later date of the same
  form. Dates of different forms are not comparable. The size plays no part.
- **Population and groups.** The resolved base models of this search, every model of the catalog
  and every successor the catalog names -- nothing else, so no request. A group is publisher, family
  and variant. Its **maximal** members -- the ones no member stands over -- are `latest`, all of them
  and whatever their size; every other member is `legacy`. A group of one is `latest`.
- **Successor.** The candidates are the group's maximal members, whatever status they keep (a
  stated `legacy` such as `Magistral-Small-2509` is one). Of them, the one of the same size, else of
  the next larger size, else of the next smaller one; a candidate without a size drops out as soon
  as one with a size fits. Exactly one candidate left: that is the successor. More than one, or an
  unknown size with more than one candidate: the row stays `unknown` -- the rule picks none at random.
- **Cycles.** Every computed edge is drawn first; then each one that lies on a cycle of the graph of
  all edges -- the stated ones (the catalog's and this search's) and every computed one -- falls,
  all at once, and its row stays `unknown`. A stated edge never falls. A chain is allowed: a
  computed successor may itself be `legacy`.
- **What stays as it is.** A stated `latest` or `legacy` keeps its status; it still counts as a
  member and can be a candidate. An `unknown` from a fault -- an invalid or unreachable
  `new_version`, a budget that ended -- is no silence and is not computed. A name that is not
  interpretable stays `unknown` and takes no part. `repo_created_at` is not read.
- **The limit.** The search sees two pages of twenty repositories per account, filtered by the
  search word. What it does not see and the catalog does not know is not compared -- that is why the
  status says `computed` and never stands in for a statement.

## Package findings and the load check (AP9-KO)

What a catalog can say beyond its model rows, and how that reaches the views. The forms here are
the contract of work package P0 of AP9-KO; what the views and the commands do with them (blocking an
installation, the recommendation, the search) is the behavior of the packages after it and is stated here
only as far as the contract needs it. Three principles hold throughout:

- **Rules, not cases.** No model, publisher or packager is named in code or in a test as a special case;
  a rule reads a property (an architecture, an owner, a revision), a datum belongs to the catalog row.
- **Only what was checked is said.** A finding names what was compared, with what and when. `present` is
  no promise that a runtime loads the file; `absent` is a statement about one package against one commit
  of one source, never about a runtime version. Everything else is `unknown`, and `unknown` is the
  absence of a finding, never a stored value.
- **A failed run changes nothing it did not read.** No new `absent`, no deletion, no new date.

### The forms and their versions (V3)

| Form | Written as | Read as | Defaults |
|---|---|---|---|
| catalog file, shipped (`modelroom/catalog.json`) | schema 2; schema 3 once it holds a finding (decided in P5, below) | 1 (TOML), 2, 3 | a row without `gguf_checks` and `setup_note` has `[]` and `None`: `unknown` |
| old channel `catalog-v2.json` | schema 2, made by `catalog_schema.project_to_schema2` only | schema 2 | -- |
| new channel `catalog-v3.json` | schema 3 | schema 3 | as above |
| render document (`models.json`) | schema 3, unchanged | schema 3 | `load_check` and `setup_notes` are left out where they hold their default |
| `search.json` | schema 3, unchanged | -- | `direct` and `knowledge_hint` are left out where there is nothing to say |
| snapshot, configuration, profile, measurement, pointer | unchanged | unchanged | -- |

**The shipped catalog (decision of P5, 2026-10-14).** `modelroom/catalog.json` becomes schema 3 as soon as an approved
finding is in it, and not before: until the maintainer approves findings it stays the schema-2 document it was, byte for
byte. The step is one line (`"schema_version": 2` to `3`, the working catalog of the scanner with `--packages` being
that very file) followed by the approval of the proposal that was made against it; `catalog_publish.py` then makes both
channel files of the dataset from it (the old one by projection without the findings, the new one as it is). The shipped
file is read by `load_catalog` as either schema, so a release that ships a schema-3 file reads it offline too.

A schema-2 document is also **written** as schema 2 (`CatalogDocumentV2` leaves the two keys out when it
serializes), so reading and writing an old file again stays what 1.0.0 reads. A schema-2 catalog is exactly the
shape release 1.0.0 reads. Release 1.0.0 refuses an unknown key in a
row, so a schema-2 document carries neither new key, not even at its default (`CatalogDocumentV2`); a
schema-3 document is read by 1.0.0 as an unsupported version and 1.0.0 falls back to its shipped catalog.
The new channel is therefore a file of its own in the same dataset, with a cache file of its own
("Published catalog"): 1.0.0 never meets it. `catalog_schema.project_to_schema2(document)` is the one way from
a schema-3 document to the old file: it drops the two keys where they hold their default and raises
`ValueError` where a row holds a finding. A scanner that proposes a new row for a schema-2 document leaves the
two keys out at their default (`apply_changes`), and a proposed finding is left out of a schema-2 document with
the reason `schema 2 has no gguf_checks` until the working document is schema 3. The size limit of a
schema-1 or schema-2 file stays 1 MiB; a schema-3 file may hold 4 MiB (`CATALOG_V3_MAX_BYTES`), a row at most 200
checks. The document rules of a row with findings: a `setup_note` is the note of the model of its row and is
never `not_read`; a package (`repo`, `identity`) has one check per row.

### What a check is a statement about (V1, V2)

A `PackageCheck` ("PackageCheck" below) is bound to one package: the repository, the **revision** of the
repository that was read, the **identity** (the stems of the weight files, shard suffix removed, as
`quantization.package_identity_key` builds it) and the weight file, or first shard, whose architecture was
read. A package of the snapshot gets the check where repository, identity **and revision** are equal
(`Catalog.package_check`); the same file in another revision of the repository, the same name in another
repository or folder, and another package of the same repository get nothing: `unknown`. Two rows that
contradict each other on a package leave it `unknown` too. Only a Hugging Face GGUF package has a check; an
Ollama package and a tensor package have none.

**Ids are compared without case** (Codex b4). Hugging Face account and repository ids are case-insensitive, so
every comparison, index and key of the findings on such an id uses `contracts.hf_id_key` (casefold): the
repository of `package_check` and `pull_check`, the package key of `merge_package_checks` and of the
duplicate check of a row, the model of a setup note against its row and against its key in `setup_notes`,
`setup_note_for`, the direct find's `resolved_base_model == repo`, and the owner of `origin.package_origin`.
An id is stored and shown as it was written; file names, identities and revisions are never folded.

**Which file the Hub's `gguf` field describes** (feasibility proof, `docs/ap9-ko/feasibility/`, 2026-10-13,
seven repositories). The field `gguf.architecture` of a repository answer is the architecture of **one file** of
the repository, not of the repository: `gguf.totalFileSize` equals the size of exactly that file, and the choice of
the file is the Hub's own by a rule it does not publish (the first file by name in six of the seven recorded
repositories, the BF16 file in the seventh). One recorded repository holds a file of
another architecture next to its model files (an importance-matrix file). So the field proves a package only
where `catalog_findings.hub_metadata_file(files, totalFileSize)` returns that package's file: the one file of that
size, else `None`. For every other package of the repository the architecture is `unknown` until it is read from
the file's own header (`architecture_source` `file_header`: the first bytes of the first shard, which the
transport reads with a range request on `huggingface.co/<repo>/resolve/<revision>/<file>`); a repository-wide
reading is never made. Where the scope cannot be shown, no check is written.

**Three states.** A missing field, missing Hub metadata (`gguf` is optional in the Hub answer), a revision that was
not checked and a file that cannot be assigned are `unknown`. Only `absent` blocks. The one exception is the
direct pull ("Direct pull", below).

### The source of the architectures (V5)

Canonical source: `github.com/ggml-org/llama.cpp`, the file `src/llama-arch.cpp`, the block
`LLM_ARCH_NAMES` -- the one place where llama.cpp maps its architectures to the identifiers a GGUF file states
(`general.architecture`), and the function `llm_arch_from_string` that compares a given name with the second
member of each entry. The scanner (never the runtime of modelroom) reads it in two steps, and only the commit:

1. `GET https://api.github.com/repos/ggml-org/llama.cpp/commits/HEAD` with `Accept: application/vnd.github.sha`:
   the answer is the commit id alone, 40 lowercase hex characters (`read_commit`; anything else is a fault);
2. `GET https://api.github.com/repos/ggml-org/llama.cpp/contents/src/llama-arch.cpp?ref=<commit>` with
   `Accept: application/vnd.github.raw+json`: the file at that commit.

The answers are limited to 256 bytes and 512 KiB, no redirect is followed (`LLAMACPP_MAX_REDIRECTS` is 0),
and the host `api.github.com` is allowed to the scanner only: `http._ALLOWED_HTTPS_HOSTS` is unchanged. Two
requests per run; the page recorded in a check is `https://github.com/ggml-org/llama.cpp/blob/<commit>/src/llama-arch.cpp`.

`catalog_findings.parse_arch_names(text)` reads the block **whole or not at all** (`ArchBlockError`): it
starts exactly once at `static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {` and ends at the first
line `};`; every line in it is a blank line, a `//` line or an entry `{ LLM_ARCH_X, "name" },` (trailing spaces,
tabs and a trailing `//` comment allowed); the sentinel `LLM_ARCH_UNKNOWN` with the name `(unknown)` is no architecture and is the last
entry, and no entry follows the end; names match
`[a-z0-9][a-z0-9_.-]*`, no enum and no name repeats; at least 100 entries remain (157 at the recorded commit);
and the lookup function `llm_arch_from_string` is named once in the file and is **whole as known** -- the loop over
the block (after its exact head `... : LLM_ARCH_NAMES) {` the line may carry an ordinary `//` comment such as
`// NOLINT(performance-for-range-copy)`, nothing else), the comparison with
the second member, `return kv.first;`, `return LLM_ARCH_UNKNOWN;` and its closing brace, with no other statement
(added logic, another return or a cut-off function is a fault). Nowhere in the file may a form appear that the
compiler reads other than the parser splits it (the list is under "Limit of the check" below). A block that is empty,
unfinished, in another form, only partly understood, there twice or with a changed lookup is a fault, never a
smaller result. `compare_architecture(name, names)` is `present` exactly when the whole identifier is among the
names. The recorded originals (commit `45451c366d4256f94fbc8845645e31a6b9155054`, read 2026-10-13): the whole file
`docs/ap9-ko/feasibility/llama-arch.cpp.45451c366d4256f94fbc8845645e31a6b9155054.txt` and its first, cut recording
`v5-llamacpp-arch-block.txt`; both hold 157 names, are read whole by the test suite, and hold none of the forms
refused below. The whole file maps a GGUF name to an architecture in no other place than the block and
`llm_arch_from_string` (no alias, no second map).

**Limit of the check** (decision of the management, 2026-10-13). The parser guards against **realistic changes
of form by the developers of llama.cpp** -- a new entry, trailing spaces, Windows line ends, a comment, a moved or
rewritten block or lookup -- and against nothing more: it is no C++ parser, and **deliberate deception by the
source** (code written to look like the known form and compile to something else) is outside what it checks.
Exotic forms are not interpreted but refused whole, anywhere in the file, as a fault: a block comment (`/*`, `*/`);
a backslash at the end of a line (it splices the next line into a `//` comment); a carriage return without a line
feed; a form feed or a vertical tab; a conditional or macro line (`#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`,
`#endif`, `#define`, `#undef`, `#line`); the digraph `%:`; a raw string `R"` with or without a prefix (`u8R`,
`uR`, `UR`, `LR`). **The consequence of an error that remains** is bounded: a check only decides whether an
installation is blocked (`absent`, the line `LOAD_BLOCKED`) or may be offered as the automatic choice; it is
bound to one package, one revision and one llama.cpp commit, shows its day, and never states that a runtime loads
a file.

**On a fault** (network, budget, parser, redirect, a changed form): the scanner writes **no new `absent`** and
**no deletion**; every existing check keeps its day. A source read whole may replace an existing check by a new
one (`merge_package_checks`: by package, never an earlier day over a later one, nothing fresh changes nothing);
setting and removing a check goes through the existing approval of the catalog proposals. A gated package is
carried as `not_covered` with its reason, not as `no_fit`.

### Which repositories are checked (V6)

The scanner asks for the GGUF repositories **of each catalog row** through their quantization relation
(`filter=base_model:quantized:<row>`), for every row of `kind` `text` or `vision`, **not only `latest`
ones** and **independent of who owns the repository**: an account that is not a listed packager is checked, and
is not made a listed packager by it. The list queries request `gguf` and `sha` (`expand`), so a revision and the
Hub's architecture arrive with the list; the files of a repository need one tree request, the header of a file
one range request. Per run at most `PACKAGE_ROWS_PER_RUN` (40) rows are taken and at most
`PACKAGE_STAGE_BUDGET` (200) requests are spent on them. The **continuation state** is small and independent of
any finding: `package_attempts` in the scanner's state file, the UTC day each row was last tried, **recorded
whether or not the try found a repository, a check or an error** (`record_attempts`). A run takes the rows by
`package_rows_for_run`: rows never tried first, then the one tried longest ago, then by name, and never a row already
tried that day; so any number of rows that yield nothing cannot hold the run in place, and a run that stops at its
budget leaves the rows it did not reach untried. The state moves no check. The header of every package of a
repository that the Hub's `gguf` field does not describe is read too (decided 2026-10-13): the first bytes of its
first shard by a range request on `huggingface.co`, inside the same budget; where that read fails the package stays
`unknown`. The revision binding stays strict: a new revision of a repository is `unknown` until it was checked
again. A stored check is compared again with every source that was read whole,
without a Hub request, as long as the repository's revision in the list is the one stored; a revision that
moved is read again.

**Hosts of the scanner** (decision of the management, 2026-10-14). With `--packages` the scanner reaches
`huggingface.co`, `api.github.com` (the llama.cpp commit and its file, V5) and the Hub's own CDN `*.hf.co`, to which the
read of a file's first bytes is redirected (at most two hops, kept on the two Hub hosts); nothing else.
`catalog_scan_packages.ScannerHosts` is that list and only the scanner builds its transport with it
(`UrllibTransport(allowed_https_hosts=ScannerHosts())`); a run without `--packages` keeps the transport it had, and
`http._ALLOWED_HTTPS_HOSTS` -- the list of the runtime of modelroom -- does not change. A daily run of the package
stage is `python scripts/catalog_scan.py --catalog ... --rejected ... --out ... --packages --budget 700`: the budget of
the run has to cover the scan of accounts (some 100 to 150 requests) **and** the 200 requests of the package stage,
or the stage ends early and says `budget ended: package checks stopped after n of 40 rows`.

Inside a row the same rule holds one level down (P4): `package_unit_attempts` in the same state file holds the UTC
day a GGUF repository was last read to its end (with a check or with a fault) and a package whose header read
failed (`<repo>:<identity>`; the repository compared by `hf_id_key`, the identity as written; entries older than 60
days are dropped). Repositories and packages are read never tried first, then the one tried longest ago, so a unit
that fails every day goes behind the others. The budget is no fault of a unit: the one it cuts is not recorded.
A row the budget cuts stays untried, except the first row of a run for which a request was sent; a row for which no
request was sent is never tried. **Bound** (decision of the management, 2026-10-14): the list of a row and the file
list of a repository are not continued over runs page by page -- a row whose list alone takes the whole budget
(some ten thousand repositories) and a file list longer than one page (some thousand entries) are taken as
unrealistic; the second is a fault of that repository with a note, and deletes nothing.

### Setup notes of model cards (V7)

A `SetupNote` is a pointer to the model card of the **publisher's** repository, valid for that model and that card
revision. The scanner reads the card's Markdown text at a revision it fixed first (`resolve/<revision>/README.md`),
by a range request for its first `CARD_MAX_BYTES` (128 KiB, the window; an unreadable card is `not_read`). **A card
longer than the window** (decision of the management, 2026-10-14; measured: publisher cards 7-28 KB, one card of
236 KB bloated by HTML tables) is read at its start. What a `206` is, its `Content-Range` alone proves: `bytes
0-<n-1>/<n>` for a body of `n` bytes is the whole card; `bytes 0-<window-1>/<total>` with a body that fills the window
and `total` above it is a start; anything else -- the header missing, another range, a total that fits neither -- is a
fault (`not_read`). Of a start the last line (cut, inside a character too) and then the last paragraph or code block
still open at the cut are dropped (a statement may go on with what limits it), and only a **positive** finding counts
(`instructions`, with `package` where the rule names one); a start without one is `not_read`, never
`none_recognized`, and the stored note stays. A `200` up to the window is the whole card, read as before; a `200`
longer than the window and an `X-Repo-Commit` of another revision are faults (`not_read`). The rule is finite and
documented with positive and negative fixtures (P4). `instructions`: an unambiguous `vllm serve` block of
this model; `package` only for an explicit statement that a named package is required for inference (negation,
training requirements and instructions for other models give none). `none_recognized` says only that the rule
found nothing. `not_read` is the scanner's report of a card it could not read: it is never stored, a fresh
`not_read` changes no stored note (`merge_setup_note`), and a note of another card revision is read again, not
carried. Nothing of a card is executed or copied.

**Limit of the rule** (decision of the management, 2026-10-14, the measure of V5): the rule guards against
**realistic forms of model cards** -- a start block, a sentence that names a package, sections and lists of several
models, negation, training -- and not against a card written to deceive it; in doubt it gives no package name (a
missed name is allowed, a wrong one is not). **Faults of the run** (network, answer, form): the measure is realistic
faults of the network and of answers, and one fault fails only its unit (a card, a package, a repository, a row),
never the run.

**The recognition rule E4-B** (`scripts/catalog_scan_card.py`, `detect(markdown, model)`; the fixtures are
`tests/test_ko_p4_card.py`). `instructions` needs a fenced block (a shell language or none) with a logical line
`vllm serve <model>` whose first argument is exactly this model's id, case ignored: a block for another model, for a
variant such as `<model>-FP8` or for a path does not count, nor does a mention in running text. A **package name** is
given only with `instructions` and only for one sentence of running text (not a code block, not a table row) of the form
`[To run|serve|use|deploy|start ..., ] <subject> [also] requires|needs the <name> [python] package` whose rest names vLLM,
inference or serving. The subject has to mean this model and nothing else: the repository name anywhere; a pronoun or
the leading word of the name only in a section without a reference to another model (a word that joins a letter and a
digit with `-`, `.`, `_` or `/`, or a size such as `70B`). A negation, `optional`, training, tuning, quantizing,
converting or building, a heading that speaks of those, and a card that names two different packages give no name only: a
card with a matching `vllm serve` block stays `instructions` with `package` `null` (`catalog_scan_card.detect`). A missing
package recognition omits the name and nothing else. `none_recognized` arises only where no matching start block was found,
and says only that the rule found nothing.

### Origin of a package (V4)

`origin.package_origin(owner, base_model, packagers)`: `publisher` is the owner of **this** base model (an
account name has no case on the Hub); `listed packager` is an account of `listed_packagers_of(catalog)`, the
catalog's own `packagers` list (the shipped default list where a catalog has none; a list a folder's
configuration adds changes what is searched, not what is recommended); anything else is `other`. The publisher
stands over the listed packager. `search.owner_class` is not used: it counts every publisher account of the
catalog, so an account that publishes other models would be released for a foreign conversion.

The comparison with the list ignores case on both sides. The origin reaches the render document (V4, additive,
like `load_check`): a row carries `origin` `other` where its GGUF package is of an account that is neither the owner of
the row's base model nor in `listed_packagers_of` the catalog the render was given; the key is **not written** for a
publisher, a listed packager, an Ollama package or a render without a catalog, so documents without such a row
are byte for byte those of 1.0.0, and two catalogs that differ in their list give two documents. The standalone
`render` and the Markdown view therefore decide V4 from the document alone.

The origin steers **only** the automatic recommendation, not the fit, the rating or the ranking. Rules for the
list of step 5 and for `offered()` (behavior of P1): a row is **recommendable** when its package is an Ollama
library package, or a GGUF package whose origin is `publisher` or `listed packager` (`auto_recommendable`), and
its check is not `absent`. The recommendation looks at **the rows shown** only -- hidden alternatives are neither
shown nor searched. A row that is not recommendable stays selectable by name (a filter that is off, a direct
id), except one whose check is `absent`: that stays blocked even for an explicit choice. `other` is not
recommended **whatever its check says and whatever approval or measurement it has**. When the first shown row
is skipped for the recommendation, the line `COMMUNITY_CONVERSION` says why; when no shown row is left,
`NO_RECOMMENDATION` is the line and nothing is pulled implicitly. An answer file without `pull_package` takes row
#1 or nothing -- it does not move to another rank.

### Direct pull (V10)

`modelroom pull <name>` reads the catalog offline (`catalog_remote.offline_catalog`, the same file the guided mode
stored) after the name was validated and **before** the daemon is touched. `Catalog.pull_check(name)` assigns
`hf.co/<repo>:<quant>` to the one stored check of that repository and quantization (`canonical_quant`, case
ignored), **only where the check itself states `same_quantization` 1**: the repository held exactly one package of
that quantization when it was checked, so a second package nobody checked cannot hide behind the name. A name without
a quantization, with two packages of that quantization or without that statement, in any other form -- an Ollama
name included, which inherits **no** finding of the Hub through its base model -- is unchecked and takes the
old way. The name carries no revision, so a stored `absent` check **blocks the pull of a movable name until a
released new check replaces it** (decided 2026-10-13, variant A): a decision on the content last checked, not a
statement about what the name serves now; the reason is `LOAD_BLOCKED` with the day of the check. A known block
ends the command with exit `2`, and `--yes` does not pass it. The two machines of a handover may hold different
catalogs; the one that pulls decides.

### Direct find and knowledge hint (V8, V9)

Applies only with a vLLM machine among the chosen ones (`guided_context.vllm_among_chosen`); without one no
tensor query is made and the knowledge hint stands in. The **direct find** puts the publisher's own model, found
by its catalog row, in the list beside the packages found by GGUF, before the list and before the question after
a typo: for a typed repository id the row with that id, for words the rows whose name the words match (the
matching of `search_word`), for a search with no word none. At most `DIRECT_FIND_MAX` (10) models are asked,
each with one request for its own repository answer (`GET /api/models/<repo>`, which must name that repository),
booked in the shared budget. A model whose answer is unreadable, not found or cut off by the budget is **named**
in `search.json` (`DirectLogEntry`), never left out. The owner filter does not hide a publisher's own model,
filter on or off. A direct find is a `SearchHit` with `found_as` `direct`: resolved as itself, no base model, no
relation, deduplicated against the packages by the base model's id. Only chosen models are written to the
configuration, through the existing change (`apply_hits`, `apply_tensor_repos`), and the original repository
is carried explicitly; no choice, no change. `search.json` gets the list `direct` (one `DirectLogEntry` each) where
at least one model was asked, and `knowledge_hint` (the count of models the hint could not show) where it was
cut; both keys are left out otherwise.

The **knowledge hint** shows without a vLLM machine, for catalog rows the words match whose base model no GGUF
hit of this search resolved to and whose row lists no GGUF repository (`gguf_repos`): `KNOWLEDGE_HINT`, and after
`; ` the sentence `KNOWLEDGE_HINT_TENSOR` only where the snapshot of the folder holds an active complete tensor
package of that model (a positive proof; the catalog holds none). No request is made. Order: `latest` rows first,
then the catalog order; at most `KNOWLEDGE_HINT_MAX` (3) lines.

### How the evidence reaches the views (V11)

The evidence is the **catalog**. `render.build_render_document(..., catalog=None)` takes it and writes it into the
document: the check of a package on its row (`load_check`, on `RankedEntry` and `SetAsideEntry`) and the setup
note of each base model that has a row (`setup_notes`, by repository id, only notes that found instructions).
Everything downstream reads the document: the Markdown, the JSON, the terminal, the pull list and the
install lines of step 5 and of the handover. The guided run passes the catalog it used (`GuidedRun.catalog`);
`modelroom render` passes `catalog_remote.offline_catalog(<folder of the pointer file>/catalog)`, the shipped or
the stored published one, whichever is later, and `MODELROOM_CATALOG=shipped` leaves the shipped one. A
catalog that holds nothing for a package leaves the row without `load_check` and the document without
`setup_notes`: both keys are then absent from the JSON. The document is the document of release 1.0.0 byte for
byte only for `None`, or for a catalog that holds nothing for these rows **and** changes the origin of no row:
the origin rule can set `origin: "other"` on a GGUF row of an unlisted packager without any check or note. The document schema stays 3; the two keys are additive, and a reader of schema 3 that refuses an
unknown key is told so here. What a row with a `load_check` shows or locks is P1's.

### What the views do with the evidence (P1, recorded by P5)

The behavior of the views and the lists on the fields of the document (V4, V5, V11); the words are the next section.

- **A blocked row** (`load_check.result` `absent`, for a GGUF package over Ollama or LM Studio, never a vLLM row): it
  keeps its place, class and rank. The terminal table names it in a line that starts with the label `blocked` and
  holds `#2 <red line>` (`packager · quantization` instead of `#n` for a package that has no rank, the repository
  and the file added where two labels would read alike), the Markdown file in a paragraph `blocked: #2 <red line>`; the
  line is said once for the runtimes the machine has (`Ollama and LM Studio`). A package the fit would set aside as
  `no_fit` is `unknown` with the red line as its reason and goes to `not_covered` with the note code `load_blocked`
  (`render._blocked_or`); a package the fit cannot judge for another reason keeps its own reason and shows the line.
- **The pull list** (`pull_choice.PullRow`): `blocked` holds the red line and the row is `locked` with it, so neither
  an explicit choice, a name in an answer file nor `--yes` passes it; `community` holds the packager of a GGUF row of
  origin `other`. `recommendable` is true for a row with neither. `modelroom pull` ends with exit `2` for a name
  `Catalog.pull_check` assigns to an `absent` check.
- **The recommendation** (`pull_choice.offered`, `Offered.passed`): only the shown rows are looked at; a rank none of
  whose fitting rows is recommendable goes to `passed` and the search goes on to the next rank, except for `#1` alone
  (an answer file without `pull_package`), which ends there and never moves to another rank.
  `recommendation_lines(offered)` is what a skipped recommendation says: `COMMUNITY_CONVERSION` once per packager,
  then `NO_RECOMMENDATION` where nothing is left; a blocked row says why under its table, not here.
- **The setup note** (`views_install.setup_note_of(notes, repo)`, by `hf_id_key`): the note of a base model with
  status `instructions`, over both vLLM commands in the card of step 5 and in the handover views.
- **The scale of step 3** (`guided_context.fit_texts(..., catalog)`): a package the catalog holds an `absent` check for
  stays in the total and does not count as fitting, so the column agrees with step 5; without a catalog, or with
  a `present` check or one of another revision, the column is that of 1.0.0.

### The words (decision E3)

English, with placeholders; a change is made for length or layout only and is reported. `LOAD_BLOCKED`:
`{runtime} installation not offered for {model}: architecture {arch} is absent from the checked llama.cpp source
(checked {date})` (red, only for an `absent` check; `load_blocked_text`). `COMMUNITY_CONVERSION`: `{packager} is a
community conversion, not checked; not offered as the automatic choice` (red, only where a recommendation is
passed over). `KNOWLEDGE_HINT`: `{model} ({publisher}) is in the catalog; no GGUF from its publisher or a listed
packager is known here` (gray; `{publisher}` is the account), with `KNOWLEDGE_HINT_TENSOR` after `; ` on a positive
proof. `SETUP_NOTE`: `{model} has setup instructions in its model card: {url}` over both vLLM commands (red), and
`SETUP_NOTE_PACKAGE` as a second line only where the note names a package. `NO_RECOMMENDATION` (not in the
decision): `no automatic recommendation: every row shown is blocked or not checked; choose a row by name`.

### PackageCheck

The architecture of one GGUF package compared with a commit of llama.cpp, on the row of the base model
(`gguf_checks`). `result` is `present` or `absent`; `architecture_source` is `hub_field` (the Hub's field,
valid for the file `hub_metadata_file` shows) or `file_header`.

```json
{
  "repo": "othercorp/Nova-7B-GGUF",
  "revision": "9f8e7d6c5b4a3928170615243342516078960a1b",
  "identity": "Nova-7B-Q8_0.gguf",
  "file": "Nova-7B-Q8_0.gguf",
  "quantization": "Q8_0",
  "architecture": "nova7",
  "architecture_source": "file_header",
  "result": "absent",
  "llamacpp_commit": "45451c366d4256f94fbc8845645e31a6b9155054",
  "source_url": "https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/src/llama-arch.cpp",
  "checked": "2026-10-13",
  "same_quantization": 1
}
```

`identity` is `identity_stem(file)`, a shard set is read at its first shard, `quantization` is what the name of
`file` says, `source_url` is the page of the block at `llamacpp_commit`, and `checked` is a UTC day.
`same_quantization` (optional, `>= 1`) is the number of weight packages of this quantization the repository held at
`revision`, counted by the scanner from the file list; `null` is not recorded. A package is found by repository,
identity, revision **and** `file` (the file read must be a file of that very package: a set of two shards and a
set of three are two packages), and two checks that differ in result, architecture or file leave it `unknown`.

### SetupNote

What the model card of the publisher's repository says about setting up vLLM, at one revision of the card.
`status` is `instructions`, `none_recognized` or `not_read` (never stored); `package` only with `instructions`;
`reason` exactly with `not_read`, one printable ASCII line of at most 97 characters; `source_url` is the card at
`card_revision` (`https://huggingface.co/<model>/blob/<revision>/README.md`), the model's page for `not_read`.

```json
{
  "model": "acme/Nova-7B",
  "status": "instructions",
  "card_revision": "9f8e7d6c5b4a3928170615243342516078960a1b",
  "source_url": "https://huggingface.co/acme/Nova-7B/blob/9f8e7d6c5b4a3928170615243342516078960a1b/README.md",
  "checked": "2026-10-13",
  "package": "nova-inference",
  "reason": null
}
```

### DirectLogEntry

One line of the list `direct` in `search.json`: a publisher model the search asked the Hub for.

```json
{
  "repo": "acme/Nova-7B",
  "source": "word",
  "status": "found",
  "requests": 1,
  "reason": null
}
```

`source` is `id` or `word` (a search with no word makes no direct find); `status` is `found` (a reason is
`null`), `not_read` (asked, no usable answer, `reason` says why) or `budget` (the shared budget ended first:
`requests` is 0). A found model or a failed read costs its request.

## Render (schema 2)

`modelroom render --config <toml>` (`cli.render_with_config(config, rating=None, now=None,
scenario=None, echo=None)` is the programmatic entry point) builds **one**
`document.RenderDocument` and hands it to three writers in `modelroom/render.py`:
`document_markdown`, `document_json` and `document_terminal`. Nothing is computed twice, so no
output can state a number another one does not -- a value that is not in the document is in no
view either. This section replaces the selection rule of "Render (AP5)" above; that section
stays for the header line, the newer-document refusal and the lock, which are unchanged.

**What it reads.** The snapshot, and for every `[machines.<name>]` the schema-2 hardware profile
its `profile` names (`<state>/hardware/<profile_id>.json`) plus that profile's measurement files
(`read_measurements`). The old per-machine `<name>.json` is no longer read as a fit input.

**Which profile a machine is rendered from** (`render.machine_profile`, pure, given
`importer.scan_profiles` of the hardware folder):

| Case | `status` | What is shown |
|---|---|---|
| `profile` names a readable schema-2 file | `ranked` | the ranking, computed from that profile |
| no `profile`, but a schema-1 file `<name>.json` is there | `legacy` | the file name and the fit rule's own reason (`fit_block_reason` for `gpu_state: legacy_unknown`); nothing is written or migrated |
| no `profile` and no such file | `no_profile` | `render.NO_PROFILE_REASON` |
| `profile` names a file that is missing, does not read, or is still schema 1 | `no_profile` | the reason, naming the `profile_id` |

A machine that is not `ranked` carries no entries at all -- there is no fit to show -- and its
readings are deliberately not printed either: a schema-1 file's numbers came from llmfit under
the old schema and say nothing about what schema 2 measures. One unreadable profile file is a
`note:` line on stdout and never keeps another machine's ranking out of the document; only the
snapshot's own `schema_version` still ends the run with exit `3`.

**Eligibility** is unchanged: `provenance in ("from_metadata", "approved")`, `complete`, `active`.
There is no grouping and no selection any more -- **every** eligible package is ranked, and
`_select_for_machine` (one variant per packager) and the reading of measurements embedded in a
schema-1 profile (`_speed_cell`) are gone.

**The scenario** is one context for the whole document: `scenario.context_requested` for every
package, never a package's own `default_context` (that is shown in its own column, `unknown` when
the package declares none). The guided mode passes the context the user just chose; a caller that
passes none gets `render_cmd.scenario_from_config(config)` -- `[guided].context` when the folder
kept one (origin `entered`, 8192 included: a kept context is one a guided run chose, and the
dialog names it so), else `measurements.default_scenario()` (8192, origin `default`). So
`modelroom render --config <toml>` of a folder a guided run wrote shows the ranking of that run
with the same origin, and a measurement taken at that context stays in measured group 0. The
scenario and the rule (`ranking.RANKING_RULE`, or `RANKING_RULE_PER_MODEL` for one row per model)
are printed in every view.

**The ranking** per machine is `ranking.rank_packages` ("Ranking rule" above): `ranked` holds the
rows of `ranking.shown_rows` -- the first `ranking.TOP_LIMIT` entries, or one per model where the
snapshot holds several, then the measured packages the rule leaves out -- and `ranked_total` says how
many packages were ranked in all, `not_covered`
holds the `unknown` fits with their reason, `no_fit` the packages whose fit class is `no_fit`. A measurement
counts only in measured group 0, and only then are `measurement_id` and `speed_tps` set.

**Provenance per value.** Every computed number carries `(computed)` in its column name
(`Weights GiB`, `Fit`, `Need GiB`, `Pool GiB`), the measured one carries `(measured)`
(`Speed tok/s`), and a profile's own readings carry their source in the Machines table
(`Origin`, `RAM source`, `VRAM source`; since 2026-09-26 also `GPUs` -- `index name vram_gib` per
card, joined by `; ` -- and `Class`, `laptop (chassis)`). Nothing computed is ever labeled as a
measurement.

**One note per package**, built from named facts only (`Note`, "Note" above): a measured entry
gets `measured_here` with `origin: measured`; an entry without a measurement gets the note for
its fit mode (`fits_in_graphics_memory`, `spread_over_graphics_cards` -- "Fits into graphics
memory spread over 2 cards: it needs about N GiB of the M GiB left after the reserve of R GiB on
each card.", the card count from the profile's `gpus`, since document schema 3 --,
`shared_between_memories`, `cpu_caps_at_fits`); a
set-aside package gets `not_covered` or `no_fit`. `facts` names the fields the text was
derived from, so a reader can check it.

**The two files.** The Markdown view goes to `config.paths.markdown` (same fixed header line,
same newer-document refusal, same lock as "Render (AP5)"), the JSON view next to it under the
same stem and the suffix `.json` -- `models.md` and `models.json`. The JSON view is written
first, because the Markdown header is what a later run compares against and must only claim a
render that produced both files. A `paths.markdown` that is itself a `.json` file is exit `2`.
**Each file is replaced atomically; the two together are not** -- no file system this package
targets replaces two files as one. Between the two writes a reader can see the new JSON view next
to the old Markdown one; a failure in between is exit `1` with a message that names which of the
two is already new, and the next render replaces both. A consumer that needs the pair consistent
compares the Markdown header's **`rendered_at`** with the JSON view's `rendered_at`: those two are
equal only when both files come from the same render. `snapshot_run_at` is not enough -- the same
snapshot can be rendered twice with different contexts, and then both files agree on
`snapshot_run_at` while their scenario and every fit differ. `echo`, when a caller passes it,
receives the terminal view; `on_document` receives the document itself **while the lock is still
held**, from which the guided mode draws the card of its step 5 -- it reads the snapshot inside that
callback, so the card and the tables rest on one snapshot (second-model round, 2026-09-24) -- and
then prints the blocks on its own; `modelroom render` passes neither, so `render_with_config` stays
silent on success -- the CLI handler `cli._cmd_render` prints after it (AP9-T-L): on success (exit 0)
**after its notes on unreadable profiles** (`note: skipped ...`, printed by `render_cmd._render_locked`
before it returns) **the Ollama setting of a saving context choice (T-K)** and then **one line with
the quality link** (`tempo_text.QUALITY_LINK_LINE`); on any other exit code nothing of this. The order
on stdout is fixed: notes, setting, link (T-K fills the setting between notes and link; until then
the notes are followed by the link alone). The line is printed whether or not the snapshot held a
ranked row, so a script can rely on it. A script that expected empty output after a successful `render`
now sees one line.

**Sections of the Markdown view**, in this order: the header line, the summary block (title,
snapshot run time, rendered time, base model and package counts, `Scenario:`, `Load:` beyond
`OLLAMA_REQUEST_HINT` requests, `Ranking rule:`, and `Market rating unavailable: <message>` when the
rating source failed), `## Areas`,
`## Machines`, and then per machine `## Ranking: <machine>`, `## Not covered: <machine>` and
`## No fit: <machine>` -- the last two only when they have rows. Every machine section is a
`##` heading, so a reader can cut the document at headings.

**One hint line under every table with words of its own** (decided 2026-09-27), one constant each
and in plain words: under the Machines table `views.GPU_STATE_HINT` (`GPU state: measured = read
from nvidia-smi; ...`) and `views.CLASS_HINT` (`Class: laptop, workstation or server, ...`), each
a paragraph of its own; under every ranked table `screen.FIT_HINT`, its pieces joined with ` · `
(`Fit is memory only: the package's need against a pool · ample: up to 60 % of the pool · fits: up
to 85 % · tight: up to 98 % · no fit: more · ...`, the thresholds read from `fit.py`), and then
`views.SPEED_HINT`, `Speed is measured, see its column.` A machine without a ranked table carries
neither of the last two. The hints are prose, so they say `no fit`; the cell says `no_fit`.

**The terminal view** is what step 5 of the guided mode shows, and `document_terminal` carries that
step's head; `modelroom render` writes the two files and prints, besides its notes, only the quality
link line (above). **The quality link** (AP9-T-L): one line, `Quality of these models for your task:
https://artificialanalysis.ai` (`tempo_text.QUALITY_LINK_LINE`, indented like the other lines of the
screen), stands once behind the last machine block -- behind the last runtime section of the last
block -- and before the closing line of the step, only when at least one block has a ranked row;
`show_nothing` (nothing to rank) does not show it. It is in `terminal_blocks`, so `document_terminal`
and step 5 carry the same line; it is **not** in the Markdown view and **not** in the JSON view (GF
decision 4 (a)), and modelroom never opens the address (README, "Network access").
The guided mode draws the head and the card itself and then prints `views.terminal_lines`, which is
the blocks alone. It shows fewer columns than the Markdown table (a terminal is narrow) and the ones
a reader recognizes a package by: `#` (3), `Model` (22, the base model's name without its account),
`Package` (21, `packager · quant`, cut with `…` where it does not fit), `Fit` (11, the longest word
`tight (RAM)`), `Speed` (12, up to `9999.9 tok/s`), `Memory` (8, `need_gib` as `GB` with one
decimal) and, since AP9-UX2 (U8), `Download` (10, `weights_gib` as `GB`: the weight files a pull
brings; since AP9-RM U11 with ` ×` behind it where it does not fit the free disk, which is why it
took one column each from `Package` and `Fit`). With the one space every line is indented by, that
is 100 characters, and no line of this view is wider but the repository lines and the commands below.

**A download larger than the free disk** (AP9-RM U11, GF 2026-09-29, ADR 0012; `modelroom/disk_fit.py`,
`views_install.py`). Where the free disk of the machine a block is about was read (its
`LiveRoom.disk`, known), every shown row whose download (`weights_gib`) plus the margin of the check
before a pull (1 GiB, `room.pull_need_bytes` -- one rule for both, decided 2026-09-29) needs more
than that free space is marked -- its rank and its class stay (ADR 0009: the fit is memory only). A
download 0.5 GiB under the free disk is therefore marked. The row lies on
the house red as a whole (`#C32F27` at 32 % over a terminal's black, mixed: `#471715`; the text
white), the one background color of the table (GF 2026-09-29; until then modelroom colored the text
only, `intro.py`), and carries ` ×` behind its download, which says the same without color
(`NO_COLOR`, the answer file's log; ` x` on a console that cannot encode `×`). Under `memory` a line
`disk` in the lighter red `#F07A72`: `× #1–10 need more than the 26.4 GB free on C: -- free space
first, or pick a smaller model` (`#1 needs` for one row), the tail only where no shown row fits, broken before `--` where the
line would pass the width, the ranks bundled as in `memory`. The `install` line takes the best row
that also fits the disk and says under it in gray `the best row that also fits the disk; #1–4 need
more space`; where none fits it says in red `install   no package fits the 26.4 GB free on C:; free
space first` and offers no command. Another computer, whose disk nobody read, and a disk of this
machine that could not be read change nothing. No row is offered that the check before a pull then
refuses; the check stays all the same, as the second safety, since it reads the disk again. The Markdown file of a
guided run carries ` ×` behind the download (`6.00 ×`) and the sentence of the `disk` line under the
machine's table, no color (`render_with_config(..., disks=)`); `modelroom render` reads no disk and
marks nothing.

**Under the head, the budget** (AP9-RM U10, GF 2026-09-29, ADR 0012; until then AP9-UX2 U8's `free`
line with the free graphics memory of the moment): a line `room`, built by `room_head.head_pieces`
from the profile and the reserves of the block, never from a live value. Its pieces, joined with ` · `
and wrapped between pieces: the graphics budget, `graphics 11.0 GB for models (12.0 GB card minus
reserve)`, for several cards the sum of `card - reserve` (the `gpu_split` pool) with `4 x 22.5 GB
cards` or `22.5 + 22.5 + 11.9 GB cards`, and `<n> cards, <sum> GB in all` where the list would pass
the line; `graphics none counted` without a counted card, `graphics unknown` for a state without a
number (never `none`); the memory budget, `memory 111.7 GB for models (127.7 GB minus reserve)`, or
`memory unknown`; on unified memory both as one piece, `shared memory 95.0 GB for models (128.0 GB
minus both reserves)`; a budget of 0 or less says `nothing left for models`. Last the free disk where
Ollama keeps its models, read live (`disk 41.2 GB free on C: (Ollama models)`, see "The pull"), for
the machine it was read on; for a machine nobody read, `disk unknown` (this computer, with its reason
where there is one), `disk unknown on another computer`, or, in `modelroom render`, which reads no
machine live, `disk unknown (modelroom render reads no machine live)`. What was read live lives in a
`room.LiveRoom` per machine name (`RunView.live`), only in memory; a guided run fills this machine's.

**What Ollama holds now** stands on a line `loaded` under the `room` line of that machine, and only
where the daemon holds at least one model (`GET /api/ps`, `room.loaded_models`): `Ollama holds
qwen3.5:9b (7.4 GB, 6.9 GB of it on the card) · graphics 0.5 GB free now` -- one piece per model, a
name cut with `…` to fit, the free graphics memory of now behind it where `nvidia-smi` or sysfs read
it. Every answer that cannot be read leaves the line out. It is a hint and changes no budget. The choice of a pool and the class compare the very
same ratio `need / pool` with `TIGHT_RATIO`, so they never disagree at a floating-point edge. A machine of the run that is not this
computer says so in its head too (U2): `server (a1b2c3d4) · imported · context L 32k`, or `· another
computer` for an entered machine or another profile of the folder.

Above each machine's table stands `<label> · context L 32k`, and `<label> · <status>` with its
reason for a machine that is not `ranked` (decided 2026-09-24: `Ranking: laptop-1
(Laptop_1)` said the machine's key, its name and a word nobody asked about, and the scenario
stood on a line of its own); where llmfit read the profile otherwise, the deviation clause follows
(`; llmfit reads RAM X GiB, this measurement Y GiB`, "Hardware profile v2"). **Fit** is the class
with ` (RAM)` behind it where the pool is system
memory -- the same word the selection list of step 2 uses -- and never ` (from size)`, which is a
note now. **Speed** is `–` where nothing was measured, not `unknown`.

Under the table, **every line carries the label of what it says** (decided 2026-09-25), in the
card's own column (`intro.label_line`, `LABEL_COLUMN`) so the report of step 5 reads as one block
from its `folder` row down to the install command. Running text buried the command a reader came for
(test round, 2026-09-25). The label is drawn in the color of the card's labels, the text gray, and a
line that continues the one above it carries no label of its own. The pieces of one label are joined
with ` · ` and wrapped between pieces at what the label leaves of the 100 characters
(`screen.joined`):

1. `shown` -- `<n> of <m> packages` (one row per model: `<n> rows of <m> packages, one row per model`),
   the rows appended for their measurement (`#3 is shown for its measurement, not by the rule`, AP9-UX
   A; `ranking_words.shown_pieces`), and one piece per set-aside group: `<k> no fit (<up to three
   names> and <r> more)`, `<k> not covered (…)`, with the reason named where the list has more than
   one of them. The word `package` is the count's own and is not repeated per piece. The names are the
   evidence behind the count and are what gives way where the line has no room for them -- two, then
   one, then none; the count and the reason never do, and no name is cut in the middle (a piece of 137
   characters was measured in the second-model round of 2026-09-24).
2. `memory` -- the fit of the ranked rows **bundled by memory pool**, with their rank ranges and
   **one pool per line**: since AP9-UX2 (U8, F5) the pool is the fit's **budget**, labeled as such
   next to the free memory of the head: `#1–2 fit into graphics memory, budget 10.9 GB (card minus
   reserve, from the profile)`, `#3 fits into graphics memory, spread over the cards, budget 44.0 GB`
   with `(cards minus the reserve on each, from the profile)` on a line of its own where the
   statement is longer than the room, `#4–5 need
   system memory, the graphics card helps, budget 111.7 GB (memory minus reserve, from the profile)`,
   `#6 needs system memory, no graphics card, budget …` -- since AP9-RM the rows in system memory name
   their budget too, so the reserve is not invisible. Two
   statements about two memories joined by a `·` read as one. The numbers come from the fields of
   `Fit`, never from parsing a note's own sentence back apart. A machine that carries the `8.0` of
   0.1.0 while the computed reserve differs gets a row more: `memory reserve 8.0 GB is set in
   modelroom.toml;` / `computed from this machine it would be 16.0 GB` (GF question 3 (a), ADR 0012);
   the Markdown file says it as a sentence under the machine's headline. A package set aside because
   no pool is left says so in words, never as a memory below zero: `it needs about 1.5 GiB and nothing
   is left after the reserve of 8.0 GiB (3.8 GiB in all)`.
   Where a result ranks two machines or more, `shown` ends with `ranked for this machine; another
   machine may show other rows`, and the Markdown file says it once, behind the first table's count
   (GF question 5 (a)).
3. `basis` -- `#2–5 computed from the package size, not its architecture`, where there are such rows.
4. `speed` -- `#3 measured 41.1 tok/s` per measured row, then the rows whose measurement counts but
   for its requests alone, with the reason their note carries: `#3 measured with 1 request, ranking
   assumes 3` (the row note `measured_with_one_request`, "Ranking rule"). Only where **neither** is
   there does it say `nothing measured in the rows shown · say Yes in step 4 to measure an installed
   package`: a row that names why its measurement does not count was measured, and to be told to
   measure it would be wrong (decided 2026-09-26). Under a machine entered by hand the advice gives
   way to `nothing measured in the rows shown · a machine entered by hand is never measured`: it is
   not this machine, and measuring this machine never writes into its profile. **Of the rows shown,
   not of the machine**: since ADR 0009 every measured package the rule leaves out is appended to the
   rows, so the card and this line count the same measurements; a measured package the fit sets
   aside (`no_fit`, `not covered`) is in no `RankedEntry` and in neither count (second-model round,
   2026-09-24 and 2026-09-25; ADR 0009, "Limits").
   Since AP9-FR-N2 the rows whose measurement counts nowhere here are named whatever the reason --
   `#8 measured, not valid`, `#2 measured, not comparable` -- by the card as well as by this line
   (the card's `speed` row below follows this rule and no other).
   `local` -- directly behind `speed`, only in a guided run, only under the table of a computer whose
   Ollama was asked in that run (this computer with its daemon reached; the active target over SSH),
   and only where at least one shown Ollama row is one that daemon provably holds: `local     #1, #4
   already in Ollama on this computer`, `local     #8 already in Ollama on gpu1`. The ranks are
   the rows' own, joined like every rank list (`#2–5`); the label is 5 columns, so it fits the
   label width of 8. No line where nothing is held: the screens of such a run stay as they were.
   It adds no column and changes no rank, order or mark.
5. `key` -- only under a ranked table: the fit hint (`screen.FIT_HINT`, the same pieces as in the
   Markdown view), wrapped between its pieces, and `Speed is measured, see its column.` on a row of
   its own (decided 2026-09-27).
6. `install` -- `#1  ollama pull <local name>`, behind a blank line, with the rank gray and the
   command drawn as a command, for the first row of the machine **the run passes in**
   (`views.show_result(..., install_for)`; the guided mode passes its own machine key) -- since
   AP9-RM U11 the first row that also fits the free disk of that machine, see "A download larger
   than the free disk" --
   `hf.co/<repo>:<quant>` for a Hugging Face build, `<name>:<tag>` for one of the Ollama registry,
   the two shapes the load test matches an installed model by, and (AP9-FR A2, "Neutral names")
   `pull_choice.PULL_NAME_RE`: `screen.install_name` builds the name and checks it before it is ever
   returned, so a Hub or registry answer outside the pattern never reaches a table row, an install
   or handover line, an answer file's value, or a `POST` -- it is no name at all, the same as a
   package with none of its own (below). `modelroom render` draws no such line at all:
   a configuration may name several writers, and which of them the command runs on it cannot know
   (second-model round, 2026-09-24). It is a command to copy, so it is never cut, and it is the one
   line of this view that may pass 100 characters. Installing is a work package of its own. Since
   AP9-UX2 (U3) every other machine of a guided run with an Ollama name in `#1` gets `install   #1 on
   server (a1b2c3d4)  ollama pull <name>` and under it, since AP9-FR A3, `run it on that computer, or
   choose a package for it below` where that machine's table has at least one row with a pull name
   (`pull_other.eligible_other_machines`, the one place both this note and the `pull_other` list
   below read from, so the two never disagree) -- whether or not this machine has a row of its own
   here (an import-only run asks `pull_other` too, Codex r1-1); else the note stays `run it on that
   computer; this run pulls only onto this one` -- no pull, no question, no list. A pull name that
   is not unique has the note `#1 has no command: its pull name is not unique` instead (U4). Rows
   that look alike carry a `repo` line each directly under the table (U4, "The pull").
7. `pull_other` -- AP9-FR A3, `modelroom/pull_other.py`. Where the note above promises a list, one
   follows step 5's own pull question (`guided_install.pull_step`): one heading per other chosen
   machine that has a shown row at all, its rows under it exactly as locked as in its own table (a
   package too large for **this** machine's disk is never checked -- no disk of the other machine is
   read here, FR reads no foreign disk without SSH), then `none, do not pull`, the pointer on it. A
   machine this user already read over SSH (a stored alias in `~/.modelroom/remote.json`) is left out
   entirely: FR-B1 offers it its own way. Choosing a row prints its **handover**, behind a blank
   line: `install   #<rank> on <machine>  modelroom pull <name> --weights-bytes <n>`, then
   `paste it there: it checks Ollama and the disk and asks before it pulls`, then
   `without modelroom there: ollama pull <name>` -- no path, no machine name beyond the neutral one
   (or SSH alias) the document already carries; nothing is called, nothing is sent anywhere. The
   command is what `modelroom pull` (below) reads.

Ten rows carried ten notes of the same two sentences and 41 lines of `not covered` said the same
thing 41 times in the test rounds of 2026-09-24; a reader needs each statement once, with the ranks
it is about. Every value comes from the same document, and the **Markdown view is unchanged to the
byte** (`tests/test_render.py` compares it with a fixture written by the code before this change).

**Exit codes**: `0` rendered (also when the rating source failed, and also when a profile file
was skipped), `1` the lock is held, there is no snapshot, the existing document was rendered
from a newer snapshot, or a view could not be written, `2` the configuration is missing or
invalid or `paths.markdown` is a `.json` file, `3` the snapshot's `schema_version` is
unsupported or it does not read.

### RankedEntry

One row of one machine's ranking. `package_identity` is the package identity triple;
`quantization` is `unknown` when the package declares none, `package_context` is `null` then.
`measurement_group` 0 carries `measurement_id` and `speed_tps`, group 1 carries neither. A
group 0 row also carries `speed_protocol` (`v1` or `v2`), the protocol of the counted measurement,
and a group 1 row never does (`null` in the JSON, like `measurement_id`). The rule runs one way
only: a group 0 row without `speed_protocol` stays valid -- a document written before 1.0.0 has
none, and the card then names no version, it never fills in `v1`. The example below shows that
older form; a row of a measured entry written now ends `"measurement_group": 0, "measurement_id":
"20260923T083000Z-5c1e9a07", "speed_protocol": "v1", "speed_tps": 50.4` (`v2` for a measurement
of protocol v2). The schema version of the document does not change. The
`fit` is always `ample`, `fits` or `tight` -- the other two classes are set aside.

```json
{
  "package_identity": [
    "huggingface",
    "packager/Nova-7B-GGUF",
    "Nova-7B-Q4_K_M.gguf"
  ],
  "base_model_hf_repo": "acme/Nova-7B",
  "packager": "packager",
  "quantization": "Q4_K_M",
  "format": "gguf",
  "weights_gib": 5.0,
  "package_context": null,
  "provenance": "from_metadata",
  "note": {
    "code": "measured_here",
    "subject": "package",
    "origin": "measured",
    "text": "Measured on this machine: 50.4 tokens per second at a context of 8192.",
    "facts": [
      "measurement.tps_mean",
      "measurement.scenario.context_requested"
    ]
  },
  "rank": 1,
  "fit": {
    "fit_class": "fits",
    "mode": "gpu",
    "need_gib": 7.125,
    "weights_gib": 5.0,
    "kv_gib": 1.125,
    "pool_gib": 10.94,
    "reserve_gib": 1.0,
    "context": 8192,
    "context_assumed": false,
    "basis": "architecture",
    "reason": null,
    "requests": 1
  },
  "measurement_group": 0,
  "measurement_id": "20260923T083000Z-5c1e9a07",
  "speed_tps": 50.4
}
```

### SetAsideEntry

One package outside the ranking: an `unknown` fit (not covered) or a `no_fit` one. Same
package fields as `RankedEntry`, plus the `reason` shown next to it.

```json
{
  "package_identity": [
    "huggingface",
    "packager/Nova-9B-GGUF",
    "Nova-9B-Q4_K_M.gguf"
  ],
  "base_model_hf_repo": "acme/Nova-9B",
  "packager": "packager",
  "quantization": "Q4_K_M",
  "format": "gguf",
  "weights_gib": 6.4,
  "package_context": 4096,
  "provenance": "from_metadata",
  "note": {
    "code": "not_covered",
    "subject": "package",
    "origin": "computed",
    "text": "Fit contract v1 cannot judge this package here: a weight file has no size.",
    "facts": [
      "fit.fit_class",
      "fit.reason"
    ]
  },
  "fit": {
    "fit_class": "unknown",
    "mode": null,
    "need_gib": 0.0,
    "weights_gib": 0.0,
    "kv_gib": 0.0,
    "pool_gib": 0.0,
    "reserve_gib": 0.0,
    "context": 0,
    "context_assumed": false,
    "basis": "architecture",
    "reason": "a weight file has no size",
    "requests": 1
  },
  "reason": "a weight file has no size"
}
```

### MachineRanking

One machine's block. `label` is the profile's `display_name`, the schema-1 file's name or the
machine name, depending on `status`; `ranked_total` is how many packages the rule ranked in all.
`reserve_ram_gib` is the memory reserve the fit took, `reserve_ram_origin` where it came from
(AP9-RM, ADR 0012): `entered` (the number in `modelroom.toml`; the default, so a document of 0.1.0
reads so), `computed` (from the profile's RAM), or `unknown` (no number, no profile: `0.0`). A block
without a profile keeps an entered number. The Markdown column `Reserve RAM GiB` shows `16.00
(computed)`, `1.50` for an entered number, `–` for `unknown`.

```json
{
  "machine": "workstation",
  "status": "ranked",
  "label": "workstation",
  "profile": {
    "schema_version": 4,
    "profile_id": "3f9a0c21d4e6b870",
    "display_name": "workstation",
    "os_fingerprint": "9d2f4b6a8c0e1357",
    "os_fingerprint_source": "windows_machineguid",
    "origin": "measured",
    "recorded_at": "2026-09-23T08:00:00Z",
    "ram_physical_gib": 31.7,
    "ram_physical_source": "os",
    "ram_limit_gib": null,
    "ram_limit_scope": "none",
    "vram_gib": 8.0,
    "vram_source": "nvidia-smi",
    "gpu_state": "measured",
    "gpu_name": "Nova GPU",
    "gpus": [
      {
        "index": 0,
        "name": "Nova GPU",
        "vram_gib": 8.0,
        "vram_source": "nvidia-smi"
      }
    ],
    "machine_class": "workstation",
    "machine_class_source": "chassis",
    "llmfit_crosscheck": {
      "ram_physical": {
        "status": "confirmed",
        "own_gib": 31.7,
        "llmfit_gib": 31.9
      },
      "vram": {
        "status": "confirmed",
        "own_gib": 8.0,
        "llmfit_gib": 8.0
      }
    },
    "llmfit_version": "1.1.16"
  },
  "reserve_ram_gib": 8.0,
  "reserve_vram_gib": 1.0,
  "reserve_ram_origin": "entered",
  "reason": null,
  "ranked": [
    {
      "package_identity": [
        "huggingface",
        "packager/Nova-7B-GGUF",
        "Nova-7B-Q4_K_M.gguf"
      ],
      "base_model_hf_repo": "acme/Nova-7B",
      "packager": "packager",
      "quantization": "Q4_K_M",
      "format": "gguf",
      "weights_gib": 5.0,
      "package_context": null,
      "provenance": "from_metadata",
      "note": {
        "code": "measured_here",
        "subject": "package",
        "origin": "measured",
        "text": "Measured on this machine: 50.4 tokens per second at a context of 8192.",
        "facts": [
          "measurement.tps_mean",
          "measurement.scenario.context_requested"
        ]
      },
      "rank": 1,
      "fit": {
        "fit_class": "fits",
        "mode": "gpu",
        "need_gib": 7.125,
        "weights_gib": 5.0,
        "kv_gib": 1.125,
        "pool_gib": 10.94,
        "reserve_gib": 1.0,
        "context": 8192,
        "context_assumed": false,
        "basis": "architecture",
        "reason": null,
        "requests": 1
      },
      "measurement_group": 0,
      "measurement_id": "20260923T083000Z-5c1e9a07",
      "speed_tps": 50.4
    }
  ],
  "ranked_total": 1,
  "not_covered": [
    {
      "package_identity": [
        "huggingface",
        "packager/Nova-9B-GGUF",
        "Nova-9B-Q4_K_M.gguf"
      ],
      "base_model_hf_repo": "acme/Nova-9B",
      "packager": "packager",
      "quantization": "Q4_K_M",
      "format": "gguf",
      "weights_gib": 6.4,
      "package_context": 4096,
      "provenance": "from_metadata",
      "note": {
        "code": "not_covered",
        "subject": "package",
        "origin": "computed",
        "text": "Fit contract v1 cannot judge this package here: a weight file has no size.",
        "facts": [
          "fit.fit_class",
          "fit.reason"
        ]
      },
      "fit": {
        "fit_class": "unknown",
        "mode": null,
        "need_gib": 0.0,
        "weights_gib": 0.0,
        "kv_gib": 0.0,
        "pool_gib": 0.0,
        "reserve_gib": 0.0,
        "context": 0,
        "context_assumed": false,
        "basis": "architecture",
        "reason": "a weight file has no size",
        "requests": 1
      },
      "reason": "a weight file has no size"
    }
  ],
  "no_fit": []
}
```

### RenderDocument

Everything one render says. `ratings` holds the rating of every base model the `RatingSource`
answered for; a base model that is not a key has no rating. A failed rating source sets
`rating_unavailable` to its message and leaves `ratings` empty, and the run still ends `0`.

**Schema 2 (decided 2026-09-25, ADR 0003).** Every `Fit` carries `requests`, and every ranked
and too-tight fit of a document was computed for `scenario.requests`. `users` and
`requests_origin` say where `scenario.requests` came from: `render_cmd` copies them from the
configuration's `[guided]` when the rendered scenario is that configuration's -- no scenario
passed, or one passed with the same requests (the guided mode always passes one). A scenario
passed with other requests leaves both `None`, so the document never names a wrong origin. The
rules of `GuidedConfig` hold between the three (`default`: one request and no head count,
`from_users`: the head count and its derived number); `users` alone, without an origin, is
invalid. The views say them in the scenario sentence (`scenario_text.scenario_line`, which `views` uses; decided 2026-09-26):
`context 32k (entered), KV cache f16 (assumed), 3 requests (from 25 users)`, `12 requests
(entered)`, and `1 request` with no bracket for `default` or a scenario that is not the
configuration's -- the Markdown head's `Scenario:` line and the `context` row of the guided mode's
card say the same sentence. A context that is a level of the scale is said as the scale says it,
without the level's letter (`8k`, `32k`, `128k`), any other as its number (`context 12000
(entered)`); `docs/models.json` keeps the number (`scenario_text.context_words`, decided 2026-09-26). Beyond `OLLAMA_REQUEST_HINT` (8) requests both carry the hint
`beyond 8 requests at once this fit says nothing about throughput: measure under load, or look at a
serving stack (rule of thumb, not measured)` -- `Load:` in the Markdown head, a `load` row in the
card; a hint, never a limit.

| Field | Type | Constraint |
|---|---|---|
| `users` | `int \| None` | schema 2; `1..10240`; set only together with `requests_origin` |
| `requests_origin` | `"default" \| "entered" \| "from_users" \| None` | schema 2; `None` for a scenario that is not the configuration's |
| `setup_notes` | `dict[repo, SetupNote]` | AP9-KO, additive; the key is each note's `model`; only notes that found instructions; **left out of the JSON when empty** |

**Evidence (AP9-KO, V11; schema 3, additive).** `build_render_document(..., catalog=None)` writes the check the
catalog holds for exactly a row's package into that row (`load_check`, a `PackageCheck`, on `RankedEntry` and
`SetAsideEntry`) and the setup note of each base model with a row into `setup_notes`. `load_check` `None` is
`unknown`, and a `None` or an empty one is **left out of the JSON**: with no evidence and no changed origin
the document is the document of release 1.0.0 byte for byte (the limits are V11's, "How the evidence reaches
the views"), and the schema version does not move. A row keeps its place and its
class; what a check changes in a view is not decided by the document. "Package findings and the load check".

**Package visibility (AP9-SV2-P, F3).** Which packages an output may show or count is read from
the runtime table, never from a package's format (`runtimes.visible_packages`, the one filter):
a row's `shown_only_with` names the packages that are that runtime's alone to serve, and such a
package is visible only where at least one machine has one of the runtimes claiming it
(`runtimes.present_runtimes`: the stored profile's `runtimes`, plus live detection for the
machine a guided run is on). Ollama claims nothing, so every Ollama package -- an Ollama-library
tensor (MLX) build included -- and every GGUF are shown on every machine, as before; vLLM claims
the Hugging Face tensor package. `shown_only_with` is not `accepts`: `accepts` pairs a package with
the runtimes that load it, and a package two runtimes load stays visible wherever either is.
Each machine block reads the packages visible for that machine alone. Every document-wide number
-- `package_count`, the ranking rule's one-row-per-model choice -- and the guided mode's balance
and card read the packages visible for at least one machine the document shows. A document none
of whose machines has vLLM is therefore byte for byte the one built without any Hugging Face
tensor package, whatever the snapshot holds; the snapshot itself keeps every package and its raw
facts. `base_model_count` counts the snapshot's base models, which no package changes.

```json
{
  "schema_version": 3,
  "snapshot_run_at": "2026-09-22T09:00:00Z",
  "rendered_at": "2026-09-23T09:00:00Z",
  "base_model_count": 2,
  "package_count": 2,
  "scenario": {
    "context_requested": 8192,
    "context_origin": "default",
    "kv_type": "f16",
    "kv_type_assumed": true,
    "requests": 1
  },
  "ranking_rule": "graphics memory before system memory, then every model that is not legacy before a legacy one, then the larger model (parameter count), then the least compression, then measured before unmeasured (a valid comparable measurement), then larger weights, then package identity; the first 10 rows, then every measured package the rule leaves out",
  "rating_unavailable": null,
  "ratings": {
    "acme/Nova-7B": {
      "stars": 3.5,
      "source": "market index"
    }
  },
  "areas": [
    {
      "source": "huggingface",
      "base_model_hf_repo": "acme/Nova-7B",
      "packager": "packager",
      "status": "complete",
      "last_success": "2026-09-22T09:00:00Z",
      "error": null
    }
  ],
  "machines": [
    {
      "machine": "workstation",
      "status": "ranked",
      "label": "workstation",
      "profile": {
        "schema_version": 4,
        "profile_id": "3f9a0c21d4e6b870",
        "display_name": "workstation",
        "os_fingerprint": "9d2f4b6a8c0e1357",
        "os_fingerprint_source": "windows_machineguid",
        "origin": "measured",
        "recorded_at": "2026-09-23T08:00:00Z",
        "ram_physical_gib": 31.7,
        "ram_physical_source": "os",
        "ram_limit_gib": null,
        "ram_limit_scope": "none",
        "vram_gib": 8.0,
        "vram_source": "nvidia-smi",
        "gpu_state": "measured",
        "gpu_name": "Nova GPU",
        "gpus": [
          {
            "index": 0,
            "name": "Nova GPU",
            "vram_gib": 8.0,
            "vram_source": "nvidia-smi"
          }
        ],
        "machine_class": "workstation",
        "machine_class_source": "chassis",
        "llmfit_crosscheck": {
          "ram_physical": {
            "status": "confirmed",
            "own_gib": 31.7,
            "llmfit_gib": 31.9
          },
          "vram": {
            "status": "confirmed",
            "own_gib": 8.0,
            "llmfit_gib": 8.0
          }
        },
        "llmfit_version": "1.1.16"
      },
      "reserve_ram_gib": 8.0,
      "reserve_vram_gib": 1.0,
      "reserve_ram_origin": "entered",
      "reason": null,
      "ranked": [
        {
          "package_identity": [
            "huggingface",
            "packager/Nova-7B-GGUF",
            "Nova-7B-Q4_K_M.gguf"
          ],
          "base_model_hf_repo": "acme/Nova-7B",
          "packager": "packager",
          "quantization": "Q4_K_M",
          "format": "gguf",
          "weights_gib": 5.0,
          "package_context": null,
          "provenance": "from_metadata",
          "note": {
            "code": "measured_here",
            "subject": "package",
            "origin": "measured",
            "text": "Measured on this machine: 50.4 tokens per second at a context of 8192.",
            "facts": [
              "measurement.tps_mean",
              "measurement.scenario.context_requested"
            ]
          },
          "rank": 1,
          "fit": {
            "fit_class": "fits",
            "mode": "gpu",
            "need_gib": 7.125,
            "weights_gib": 5.0,
            "kv_gib": 1.125,
            "pool_gib": 10.94,
            "reserve_gib": 1.0,
            "context": 8192,
            "context_assumed": false,
            "basis": "architecture",
            "reason": null,
            "requests": 1
          },
          "measurement_group": 0,
          "measurement_id": "20260923T083000Z-5c1e9a07",
          "speed_tps": 50.4
        }
      ],
      "ranked_total": 1,
      "not_covered": [
        {
          "package_identity": [
            "huggingface",
            "packager/Nova-9B-GGUF",
            "Nova-9B-Q4_K_M.gguf"
          ],
          "base_model_hf_repo": "acme/Nova-9B",
          "packager": "packager",
          "quantization": "Q4_K_M",
          "format": "gguf",
          "weights_gib": 6.4,
          "package_context": 4096,
          "provenance": "from_metadata",
          "note": {
            "code": "not_covered",
            "subject": "package",
            "origin": "computed",
            "text": "Fit contract v1 cannot judge this package here: a weight file has no size.",
            "facts": [
              "fit.fit_class",
              "fit.reason"
            ]
          },
          "fit": {
            "fit_class": "unknown",
            "mode": null,
            "need_gib": 0.0,
            "weights_gib": 0.0,
            "kv_gib": 0.0,
            "pool_gib": 0.0,
            "reserve_gib": 0.0,
            "context": 0,
            "context_assumed": false,
            "basis": "architecture",
            "reason": "a weight file has no size",
            "requests": 1
          },
          "reason": "a weight file has no size"
        }
      ],
      "no_fit": []
    }
  ],
  "users": null,
  "requests_origin": "default"
}
```

## Load test (stage 1)

The load test measures how fast a package really runs on this machine. **Stage 1 measures
packages the local Ollama daemon already has**: it downloads nothing, removes nothing and says
nothing about disk space. `modelroom/loadtest.py` is the work of it, `modelroom/daemon.py` the
transport it talks through, and `modelroom/guided_loadtest.py` the guided mode's step 4 around
both.

**The transport** (`daemon.py`). `Daemon` is `(method, path, body) -> Response`, the second and
much smaller transport of this package -- `http.py::Transport` is `(method, url, headers)` and
cannot carry a JSON body, which `POST /api/show` and `POST /api/generate` need. `LocalDaemon`
reaches exactly one origin, `http://127.0.0.1:11434`
(`ollama_local.py::DEFAULT_BASE_URL`, the same origin `http.py::_ALLOWED_HTTP_ORIGINS` names);
a base URL with another scheme, host, port or a path is refused in the constructor
(`DaemonError`). Its opener carries `HTTPHandler` and the error handlers and **nothing else**:
no `ProxyHandler` (a proxy would move this machine's own daemon traffic off the machine), no
`HTTPSHandler`, no file/ftp/data handler, and a 3xx is returned as it came instead of being
followed. `ALLOWED_CALLS` is the whole surface -- `GET /api/version`, `GET /api/tags`,
`GET /api/ps`, `POST /api/show`, `POST /api/generate`, and since 2026-09-25 `POST /api/pull` --
checked before a connection is opened; anything else, a pull by any other method included, is a
`DaemonError`. The sixth call is the pull question of guided step 5 ("The pull", under "Guided
mode"), decided 2026-09-25 because the README promised the install from step 5; `daemon.py` is
the one module that names its path and `guided_install.py` the one that uses it. There is no call
of `/api/delete` anywhere in `modelroom/`, and a test reads every module to prove both. **Time
limits:** 10 s for every short call, 600 s for one `POST /api/generate` (a CPU machine needs
minutes for 128 tokens), and for a pull **60 s per read on the socket and no limit for the whole
pull**: `urllib` limits each blocking read, so 60 s without a byte is a fault, and a slow pull that
keeps sending lines may take as long as it takes. A daemon that keeps sending status lines while
no byte of the package arrives is **not** recognized by this, and nothing here claims it is. A
pull's answer is one JSON line per step of it; it is read line by line and never collected, each
line goes to a callback, and reading stops at the first line that ends the pull -- `success`, or a
line with an `error` -- which is the `Response`: nothing after an error can turn it into a success,
and a daemon that keeps the answer open after `success` is not waited for. Only a `200` is read
that way: any other status, a `2xx` included (`urllib` raises for a `4xx` or `5xx` only), is one
bounded line, and the rest of an answer kept open is not waited for (third review round,
2026-09-25). A line that is no UTF-8, no JSON, no JSON object or longer than 64 KiB is a
`DaemonError`. A limit that is reached, a refused connection
or a body that is not JSON is a fault with a reason, never a traceback. **Documented limit:**
whatever answers on that port is taken to be this machine's daemon -- nothing in the answer proves
it, and nothing here pretends to check.

**The inventory** (`installed_models`, `installed_candidates`). The daemon's own `/api/tags` is
read through `ollama_local.py::fetch_installed_models`, the same reader `hardware` uses, so one
place parses that answer. An entry whose size is `0`, or whose tag is `cloud` or ends in
`-cloud`, is a model the daemon runs elsewhere and never a package on this machine (read from a
real daemon on 2026-09-24: `kimi-k3:cloud` and `gpt-oss:20b-cloud` are listed with a stub of a
few hundred bytes); those are listed by name with that reason and never measured.

**The assignment rule is a digest, never a name.** An Ollama package is the entry whose
`/api/tags` digest equals its `manifest_digest`. A Hugging Face package is an entry named
`hf.co/<owner>/<repo>:<quantization>` whose `<owner>/<repo>` is the package's `repo` **and**
whose weight digest, read with `POST /api/show`, is the digest of one of that package's own
weight files (`weights`/`weights_shard`). `/api/show` answers with a Modelfile whose `FROM`
line names the blob on disk as `…/blobs/sha256-<hex>` (the path may contain spaces, so only the
digest at the end of that line is read); for a package pulled from Hugging Face that hex is the
`sha256` of the GGUF file itself, which is exactly what the fetch recorded as that file's digest
(measured 2026-09-24: `sha256-a86349…4ec1` for
`hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M`, equal to the `Q4_K_M` file's LFS `oid` in
the repository's tree). A name that matches without such a digest is listed as unmatched with its
reason and **never measured**: a package may not claim what the daemon does not show. Only
`active` packages are considered. The `PackageRef` a candidate carries is built so that
`ranking.py::_matches` finds it -- Ollama: the manifest digest; Hugging Face: the package's `repo`
and `revision` and the digest the daemon showed.

**One blob proves one file.** A Hugging Face package whose weights are **several** files therefore
stays unmatched in stage 1, with that as its reason: the daemon holds one blob, and crediting the
whole package with a measurement of one of its files would state a speed for content that never
ran. A package with exactly one weight file is the only one that can be measured here.

**Known limit of the reader side.** `ranking.py::_matches` accepts a measurement whose
`hf_file_digest` is **any** of a package's weight digests. So a repository revision that held both
a single-file package and a several-file package sharing one of those digests would let the
single-file package's measurement count for the several-file one as well. The writer is the
conservative side of that pair and never produces a measurement *of* a several-file package; the
reader's rule is unchanged here and is an open question for the ranking, not for this writer. Two
single-file packages of the same repository, revision and weight digest are the same content under
two names and share a measurement on purpose.

**The protocol is not a parameter.** `run_load_test` always runs `shipped_protocol()`, because the
record it writes says `protocol: v2` and `MeasurementRecord` checks a valid one against the
shipped definition of its own version. A run after another prompt, seed or warm-up count would be stored under a name
it does not deserve, and no validator could tell afterwards. A new protocol version is a new
writer.

**The run** (`run_load_test`). In this order: `GET /api/version` (stored as `daemon_version`,
never a validity criterion), the machine's load once (`LoadState`), the warm-up run, then the
measured runs of protocol v2 (four readings, none before the warm-up). **`/api/ps` is read after every run, the warm-up included**, so
every measured run stands between two observations. Each `POST /api/generate` carries the local
name, the protocol's prompt, `raw`, `stream` and its options plus `num_ctx` = the ranking's
`context_requested`. The warm-up is not one of `runs`.

**`/api/ps` decides `comparable`, not validity.** Every observation has to show the measured
model under its own local name, with the manifest digest `/api/tags` showed for it, and with
`context_length` equal to the scenario's `context_requested`. The model is looked up in the
answer **by its local name**, not taken from the first entry: a daemon can hold several models at
once, and another one of them standing first is no statement about this measurement. A deviation,
or an observation that does not show the measured model at all (then the reason names what was
loaded instead), makes the **whole** measurement `comparable = false` with that reason. It is stored all the same, with its own validity, and is never ranked (it lands in
measurement group 1). Validity is decided only by `measurement_invalid_reason`; the three
speeds are set only for a valid measurement. `tps_mean` is the mean of the measured speeds, held
inside their own minimum and maximum: three runs of the same duration sum to a float one unit in
the last place below that duration's speed, and the record's `tps_min <= tps_mean <= tps_max` rule
would refuse a perfectly good measurement over it.

**The machine's load** is read once, just before the runs: GPU utilization from `nvidia-smi
--query-gpu=utilization.gpu --format=csv,noheader,nounits` through `Probes.runner` with the
10 s limit, and the one-minute load average where the platform has one (`os.getloadavg`;
Windows has none). Every source may be missing -- then there is no value and no failure. Load
*during* a run is not measurable and is never claimed; the guided mode says so in a line of its
own.

**Faults.** A daemon that cannot be reached, a status other than 200, an answer without counters,
an answer nested deeper than the JSON parser goes, counters the record's own rules refuse (a
duration too large to compute a speed from, for instance), a limit that was reached, a model that
is gone: each is a `LoadTestError` with its reason, and a run
that raises it has written **nothing** -- no half file. An interrupted run (Ctrl-C) leaves no
measurement behind for the same reason: the file is written after the last run.

**The file.** `store_measurement(state_dir, record, now)` takes `modelroom.lock` of that state
folder (command `load-test`), writes with `write_measurement` and releases it. Another process
holding the lock reaches the caller as `LockHeldError`, which is exit `1` for every command
here. Nothing outside `<state>/measurements/` and that lock file is written.

**What never happens in stage 1:** no pull, no `delete`, no clean-up, no disk space statement,
and no second way of computing a speed. The load test measures what is there; the one pull of
this package is a question of step 5 of its own ("The pull", under "Guided mode"), and a package
it pulls is measured by the load test of the **next** run.

## Local servers and the vLLM load test

A machine may serve models through more than one local server. **The origins are a list**:
`LOCAL_SERVERS` in `modelroom/local_servers.py` is the one place a server is registered, as
`LocalServer(key, name, origin, transport(run), start_fact, offer, measure, card_facts)` (AP9-UX E:
`offer` says a server's notes and offers its rows of the one list of step 4, `measure` measures what
was picked; they replace `step`, ADR 0009). `ollama` wraps what existed before (its daemon row,
`guided_loadtest.ollama_offer` and `measure_ollama`, the run's `daemon`); `vllm` is the second
(AP9-SV1, decided 2026-09-27). The start screen, step 4 and both
card paths of step 5 loop over the list, so an entry added to it reaches all three. **Nothing a
server states or measures is persisted**: no measurement file, no profile, no snapshot, no
configuration, no export and no render document changes, and a machine without a vLLM server writes
byte for byte what it wrote before (ADR 0008).

**The transport** (`modelroom/vllm_server.py`). `LocalVllmServer` reaches exactly one origin,
`http://127.0.0.1:8000`; a base URL with another scheme, host or port, or with a path, a query, a
fragment or user info, is refused in the constructor (`VllmServerError`). Its opener is
`daemon.py`'s: plain HTTP, no proxy handler, no redirect following. `ALLOWED_CALLS` is the whole
surface -- `GET /v1/models`, `GET /metrics`, `POST /v1/chat/completions` -- checked before a
connection; `/version`, `/health`, `/v1/completions` and everything else are refused. Time limits:
600 s for a chat, 10 s for every other call; an answer above 4 MiB is a fault. A `401` is the
reason `the vLLM server asks for an API key; modelroom reads an open local server only`: no key is
ever sent. `FixtureVllmServer` is the test double, like `FixtureDaemon`. **Documented limit:**
whatever answers on that port with a model list owned by `vllm` is taken as this machine's server.

**The served model** (`read_vllm_models`). `GET /v1/models`, the one entry whose `owned_by` is
`vllm`: `ServedModel(id, root, max_model_len)`, **stated** by the server. Several entries are a
reason (`serves 2 models, modelroom reads one`) and no model. The server states no revision.

**The package it serves** (`guided_vllm.match_served`). Exactly one active complete `tensor`
package of the snapshot whose `repo` is the stated `root` -- the package the fetch records today for
a safetensors repository (`unresolved`/`format`, weights the sum of its safetensors files, no
quantization word; `hf.py`, `contracts.py` and `provenance.py` unchanged). None is the line `no
package of this folder comes from <root>`; several (one repository at several revisions) is `the
server names none of the <k> tensor packages of <root>`, and there is no fit.

**The publisher's repository is always a target** (AP9-UX2 P3, GF F8, ADR 0010). Until then a
guided run wrote only the repository of the hit that led to a base model -- a packager's GGUF
repository, as a rule -- as its `repos` target, so the fetch never read the publisher's `tensor`
package and a server serving the base model found none (second hand run, 2026-09-28).
`search_apply.apply_hits` now adds every resolved base model's **own** repository as a `repos`
target where it is not one yet, once; the fetch reads it in the publisher's area and records its
`tensor` package (`unresolved`/`format`, outside ranking, balance and card). `match_served` is
unchanged. The cost per base model without such a target before: one model info besides the meta
request, and the pages of its tree (usually one). `Qwen/Qwen3.5-4B` has one such package: two
safetensors shards of 5,329,398,688 and 3,990,429,408 bytes (public Hub tree), which
`normalize_file_stem` groups under one stem. Configurations written by hand and `package_targets`
are unchanged. The vLLM fit of a `hybrid_linear` model counts only its full-attention layers per
token and its linear-layer state once per request: `at_once = floor(kv_bytes / (per_token x context
+ state_bytes))` (P1).

**The load test** (`modelroom/loadtest_vllm.py`). One warm-up request, then three batches of `n =
min(scenario.requests, 256)` requests at once (`MAX_REQUESTS`; it was 64 until AP9-PM-1); the scenario
itself never changes. Each request is
the one recorded against vLLM 0.30.0: a chat of 128 tokens, temperature 0.0, seed 42, not streamed,
`Describe how a public library shelves a new book. Request <i>.` (`0` the warm-up, `1..n` in each
batch). Per batch: `GET /metrics`, n threads behind one `threading.Barrier`, `perf_counter_ns` per
request and for the batch, `GET /metrics` -- `1 + 3 x n` chats and 6 metrics calls in all, the
warm-up in no number.

- **The time of a batch** (`modelroom/loadtest_batch.py`, one helper for vLLM and LM Studio, AP9-PM-1): the start
  is read in the barrier's own action, before any request is released; each request takes its own end the moment its
  answer is in; the batch's wall time is the latest end minus that start. Until AP9-PM-1 the start was read in the
  main thread after the barrier, and a late start could make the total throughput of a batch read too high. A wall
  time of 0 or less is the existing invalid path of vLLM, `invalid (batch <k>: no measurable wall time)` with the code
  `invalid_tokens` (the rule of `loadtest_vllm.invalid_reason`; a request without a time of its own is named first).
- **Measuring condition (decision of 2026-10-10):** the measurement is made under the cache setting the server has.
  Requests that differ in their prompt do not exclude cache effects.
- **Known limit of time:** a batch has a nominal HTTP budget of 10 + 600 + 10 s, plus the local run time (threads,
  synchronisation); the effective limit stays 660 s between two lines of the protocol (a progress line goes out after
  each batch). A single request that runs into its 600 s ends the run as before, without a partial result. vLLM puts
  requests beyond its limit of concurrent sequences (`--max-num-seqs`) into its queue, and the waiting counts toward
  the 600 s.

- **Valid**: every request answered 200, `finish_reason` `length`, 128 completion tokens, and
  `prompt_tokens + 128 <= max_model_len`. The first rule that fails is the reason: `invalid (batch 3:
  finish_reason stop)`, and then there are no speeds.
- **Comparable**: `n` equals the ranking's requests, no request running or waiting before and after
  each batch, `vllm:generation_tokens_total` grew by `n x 128` in each batch, and `max_model_len`
  is the ranking's context. The first rule that fails is the reason: `not comparable (256 requests
  measured, the ranking assumes 257)`, `not comparable (context 4096, the ranking's 32768)`.
- **Speeds**: the total of a batch is its completion tokens over its wall time; the total speed is
  the mean of the three, its range their minimum and maximum. The speed per request is the mean over
  all `3 x n` requests, with their range. Each mean is held inside its range (`loadtest._mean_within`).
  A wall time includes the prefill and the queue of the server, so it is never mixed with an Ollama
  measurement of protocol v1. The result is `VllmMeasurement`, in memory. A server that goes away is
  a `LoadTestError`.

**The fit on one card** (`modelroom/fit_vllm.py`, computed, no `Fit` object; `fit.py` unchanged).

```
budget  = 0.92 x card_gib            0.92: vLLM 0.30.0 default, docs.vllm.ai/en/v0.30.0/configuration/engine_args
kv_gib  = budget - weights - 1.50    weights: the safetensors sum, no 1.10 overhead
at_once = floor(kv_gib / per_token / context)
```

`per_token` is the base model's architecture where fit v1 can read it, else the size basis of
`fit.py` (147,456 bytes), and then the class is capped at `fits`. The class: `no_fit` below one
request, `tight` below the ranking's requests, `ample` at twice them or more on the architecture
basis, else `fits`. The card is the **largest card** of the profile this machine is bound to (the
profile `guided_context.machine_checked` reads); the table of a machine with several equal cards
uses the per-card rule with `t` cards of "Fit for vLLM", not this paragraph.
**1.50 GiB** is the runtime of the recording on one A100-SXM4-80GB at utilization 0.5 (`0.5 x 80 -
0.920 - 37.58`, decided 2026-09-27); the recording on one L4 gave **1.31** (`0.5 x 22.494 - 0.920 -
9.02`), the second measured value. **Limits:** one 0.5B model at utilization 0.5 is the evidence; a
server started below 0.92 has less KV cache than computed; CUDA's first card need not be the largest;
the prefix cache is not modeled; `render` never shows any of it.

**The install of a vLLM row (AP9-SV2-I, I1 to I4; `modelroom/install_vllm.py`, `modelroom/hf_cache.py`).**
modelroom starts no vLLM and pulls nothing for it: a vLLM row prints **two commands** built by one
function (`install_vllm.vllm_install(package, fit, profile, relative_cache=False)`) from the same
checked values, and the reader pastes one into a terminal that stays open (GF 2026-09-29, F1 a, F2 b).
Both are plain POSIX shell text for the measured Linux machine only; **nothing here runs a process or
looks for `docker`**.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<index>,... VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 vllm serve <repo> --revision <sha> --max-model-len <context> --tensor-parallel-size <t> --gpu-memory-utilization 0.92 --host 127.0.0.1 --port 8000
mkdir -p "<expr>" && docker --host unix:///var/run/docker.sock run --rm --ipc=host --gpus '"device=<index>,..."' -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 -p 127.0.0.1:8000:8000 --mount "type=bind,src=<expr>,dst=/root/.cache/huggingface/hub" vllm/vllm-openai:v0.30.0 <repo> --revision <sha> --max-model-len <context> --tensor-parallel-size <t> --gpu-memory-utilization 0.92 --host 0.0.0.0 --port 8000
```

`<expr>` is the constant `hf_cache.HF_CACHE_EXPRESSION`,
`${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}`, in the `mkdir` and in the
mount, character for character: the shell of the machine resolves it, so the download lands where the
disk was checked and no host path stands in a line. The mount is `--mount type=bind`, never `-v`;
`--host unix:///var/run/docker.sock` (`DOCKER_LOCAL_HOST`) outranks `DOCKER_HOST` and `DOCKER_CONTEXT`
(rootless Docker uses another socket); the image is the constant `VLLM_IMAGE` (`vllm/vllm-openai:v0.30.0`,
the version of the SV1 recordings and of the calculation, assumption A8; a change is that one line and a
hand test); in the container `--host 0.0.0.0`, outside only `127.0.0.1:8000`, `--ipc=host` as the vLLM
documentation asks. The Docker line appears exactly where the `vllm serve` line does: vLLM is detected on
the measured machine (`runtimes.detect_vllm`: Linux only, a server that answers or `vllm` in the `PATH` or
at `~/.local/bin/vllm`, never a Docker call) and the row's cards are nameable. A relative
`HF_HUB_CACHE`, `HF_HOME` or `XDG_CACHE_HOME` (the effective one, `hf_cache.cache_is_relative`,
`Disk.relative`) leaves the Docker line out and says `docker: left out, the Hugging Face cache is set as a
relative path; set it absolute to get this line`; the `vllm serve` line stays.

Every value is checked before it reaches a line: the repo against the Hub's name rule and ASCII, the
revision against 40 lower-case hex digits (`fullmatch`), the context, the card count and the card
numbers as whole numbers (never text, never `bool`) of the fit; the image, socket, expression and
container path are constants. Anything else gives no line and the lock `no safe command`. **The cards are
bound by number, always** -- the `index` values of exactly the cards the fit chose
(`fit_vllm.chosen_cards`), also when that is all of them, so an inherited `CUDA_VISIBLE_DEVICES` cannot
shrink the line; a chosen card whose memory is not `nvidia-smi`'s (`vram_source` `sysfs`, `sysctl`,
`llmfit`, `entered`) has no CUDA number modelroom can name: no line, the lock `computed only:
modelroom cannot name these cards to vLLM`, the row stays in the ranking with its fit. Under the lines
the table says what to do: `run one of the two in a terminal that stays open; vLLM downloads the model
at its first start`, `then run modelroom again: step 4 finds the server and measures it`; a package on
disk already replaces the first with `#<n> is on disk already (size checked)`; another model served on
port 8000 adds `vLLM serves another model on port 8000 now; stop it first (one model per server)`. Where the
model allows a larger `--tensor-parallel-size` than the one named, on this machine's equal cards (rule (b) takes the
smallest `t` that fits; `Fit.larger_size`, so 14 heads on four cards stop at `t = 2` without it, and an unknown head
count never has it),
`larger --tensor-parallel-size spreads the load over more cards, if the server is yours alone` stands first among
these notes (`VllmInstall.larger`, one line, `LARGER_HINT`). A row
locked `no safe command` (or `computed only`, or `pull name not unique`) where no other row offers a line
says so under the table as `#<rank> <lock>`.

**The list of step 5** (I4) takes the vLLM rows: value `vllm <repo>` (Ollama's values are unchanged), a
locked row says `no safe command`, `computed only: ...` or `pull name not unique` (two groups of one repo).
The question is `Which package should be installed?` only where a vLLM row stands in the list, else
`Which package should be pulled into Ollama?`; a vLLM row is offered whether or not Ollama's daemon
answered. Choosing one prints its lines: **no yes/no question, no pull**; an unknown `pull_package` value is
an error (exit `2`) as before. For another computer (`pull_other`) a vLLM row is handed over with both
commands and the machine named, never `modelroom pull`, also for the active SSH machine.

**Where vLLM keeps its models, and "on disk"** (`hf_cache`). The place is `HF_HUB_CACHE`, else
`$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub` (an empty variable is
unset, as in the shell), measured at the nearest existing parent like Ollama's. **The disk rule of AP9-RM U11
is unchanged and reads each row at its own runtime's place** (`disk_fit.place`): download plus the 1 GiB
margin against the free disk -- Ollama's rows at Ollama's, vLLM's at the cache, nothing marked where that place
was not read. The download of a vLLM row is the sum of its start files (`tensor_quant.start_files`, one
definition); `Disk.needs` carries, per package, the bytes still to fetch. The `room` line says
`disk 41.2 GB free on C: (Ollama and vLLM models)` where both places are one drive, two pieces otherwise,
and is as before for a machine without vLLM. A package is **on disk** when every start file lies in
`<cache>/models--<owner>--<name>/snapshots/<sha>/` with the size the package declares (`stat` only, never
a digest or a read). Missing start files make it not on disk and the rule counts the bytes still missing;
a missing start file declaring size 0 (or a present one of another size) leaves `missing_bytes` `None` and
the rule counts the whole download. Unsafe names (`SAFE_REPO_PATH_RE`), repo or revision are `None`
**before any file is touched**; a path whose real location lies outside `models--<owner>--<name>` **and**
outside the cache's own blob store `<cache>/blobs/` (a link to another repo, `/etc`, a drive, a share) is
`None`, as is an unreadable cache. The legitimate `snapshots/<sha>/<file>` -> `../../blobs/<hash>` links
count, and so does the layout of huggingface_hub 1.x (Xet), where that link leads on to
`<cache>/blobs/<xx>/<sha256>`, one store shared by every repo of the cache (seen at a rented server,
2026-10-03: without it every fetched weight read as unknown). **The repo folder itself is anchored**
(Codex r1, R1-2): `realpath(<cache>/models--<owner>--<name>)` must equal `realpath(<cache>)` joined with that
folder name, else the answer is `None` and nothing below is touched, so a link that moves the whole repo folder
elsewhere is no boundary. **One `stat` per start file, at the resolved path** that passed the boundary check
(R1-3), and only a regular file counts (`S_ISREG`, R1-4): a folder, a FIFO or a socket where a file belongs
is a file that is not there. **Limit:** Docker started with another volume than the printed line has another
place; the check does not apply. A link swapped between the `realpath` and the `stat` is not caught (no
directory handle or `O_NOFOLLOW`: not portable to Windows); that window reads metadata only, never content,
in a cache that belongs to the user.

**LM Studio (AP9-LS, GGUF only).** The third runtime of the table, one entry and no new package class. The
entry `lmstudio` is a new value of `RuntimeKey` and of `Fit.runtime`; the value is effective since the
entry is registered (AP9-LS-E), and no render writes it in 0.2.0, because a package has one fit, the one of the
first runtime that accepts it (Ollama's). The fields: name `LM Studio`; origin `http://127.0.0.1:1234`, fixed, no setting;
`hello_field` `lmstudio` (a computer reached over SSH reads `absent`, no remote operation exists for it);
`accepts` and `shown_only_with` the Hugging Face GGUF packages, never an Ollama or a tensor package; `fit`
the fit rule of Ollama (both run llama.cpp); `install_name` `<owner>/<repo>@<quant>` where the command is
safe; `command` `lms get --gguf https://huggingface.co/<owner>/<repo>@<quant>` (the URL, because
`lms get <owner>/<name>` names an artifact of the LM Studio Hub), **printed, and run once after a yes**
(AP9-LS-G, below); `model_place` the
standard model folder under the home folder; `on_disk` a file of the declared size in it (`stat` only);
words without a prefix or a key hint, the question `Which package should be installed?`.

**The transport and the reader** (`modelroom/lmstudio_server.py`). `LocalLmStudioServer` reaches exactly one
origin, `http://127.0.0.1:1234`, judged by the very function the vLLM transport uses
(`vllm_server.check_origin`): another scheme, host or port, a path, a query, a fragment or user info is
refused in the constructor. The opener is `daemon.py`'s: no proxy handler, no redirect following. The whole
surface is **two calls**, checked before a connection: `GET /api/v1/models` with a time limit of 10 s and
`POST /v1/chat/completions` with 600 s, the one request of the measurement (see "Measurement" below);
`/api/v1/models/load`, `/unload`, `/download`, `/api/v1/chat`, `/v1/models`, `/api/v0/*` and `/metrics` are
refused like any other call. The limit of time holds for the whole call, at the connection, at the body and at
the body of an error status, and is told apart as `LmStudioTimeoutError` wherever it is reached; an answer above
4 MiB is a fault. At the list a `401` or a `403` is the reason `the LM Studio server asks for an API token;
modelroom reads an open local server only`, decided from the status alone (its body is not read): no
`Authorization` header is ever sent. At the chat the same statuses are an answer like any other (row 6 of the
table below). A chat answer whose body cannot be read, or is above 4 MiB, is handed on with its status, an empty
body and the header `x-modelroom-body: unread` or `oversized`, so that the measurement can name its row; only a
limit of time is raised. `read_lmstudio_models`
reads the list in three steps: (1) `keys`, the `key` of every entry that is an object, before any filter;
(2) the structure, which is the version gate (no version number is read): an entry is valid when it is an
object with a `key` string, a `type` string, a `size_bytes` whole number and a `loaded_instances` list; `models`
that is no list, or a non-empty list without one valid entry, is no LM Studio list and the server is not
`serving`, an empty list or one of embedding and MLX entries only is `serving` with nothing to measure;
(3) the measurable instances: only `type` `llm` with `format` `gguf` and names matching
`^[A-Za-z0-9][A-Za-z0-9._@/:-]{0,127}$`, as `LoadedModel(instance_id, key, context_length, parallel)`. The
reader gives no package identity: REST names no repository and no file, and `lms` is never called.

**Detection** (`detect_lmstudio`). `serving` when the list reads, `asks_key` on a `401` or `403`, `installed`
when `lms` is in the `PATH` or at `~/.lmstudio/bin/lms` (`lms.exe` on Windows), else `absent`; only the
`PATH` search and a file test, **never an `lms` call**. `lms_call` says `path`, `home` or `None` for the form
of a later command and starts nothing.

**Measurement: one loaded model, one request after the other (several at once where the instance allows it,
see below), on the wall (AP9-LS-M, protocol v2 contract
section 4; GF decisions of 2026-10-09: E2 the prompt of the protocol, E3 thinking tokens counted, E4 measure
although the server may load on its own).** The start screen names the server as before: `server  LM Studio
<k> loaded: <id>, context <n>`, `<k> models loaded`, or `127.0.0.1:1234  asks for an API token`. Step 4 offers a
row `LM Studio server  <instance id>, context <n>` for **exactly one loaded GGUF instance**; with none or several
it says `LM Studio: no model loaded, nothing to measure` or `LM Studio: <k> models loaded, measure with one
loaded model` and offers no row, and a server that asks for a token has neither row nor note. With a context
other than the run's it offers the row, says `the number will not count: LM Studio runs context <m>, this run
asks for <n>` before the question, and a yes sends no chat.

The request is `POST /v1/chat/completions` with `model` the **instance id** of `loaded_instances[]` (not the
`key`), `max_tokens 128`, `temperature 0.0`, `seed 42`, `stream false` and one user message, the prompt of the
shipped protocol (`shipped_protocol().prompt`, the Ollama prompt). One warm-up and three runs, one request at a
time (this is the course for n = 1, see "Several at once" below), and the list is read five times: before the
warm-up, after the warm-up, and after each of the three runs.
Every read has to show the same instance with the same `context_length`, else the measurement ends. A request is
valid with status `200`, `choices[0].finish_reason` `length`, `usage.completion_tokens` 128 (thinking tokens are
part of them; `reasoning_tokens`, `content` and the draft counters of a speculative decoder are not read, an
empty `content` is fine), `usage.prompt_tokens` a whole number `>= 0` (`bool`, a float or a string are not one)
with `prompt_tokens + 128 <= context_length`, and a wall time above 0. The wall time is the injected clock read
right before the call and right after the whole body is in; the reads and the warm-up lie outside it. The
number is the mean of the three `128 / seconds`, held inside its minimum and maximum
(`loadtest._mean_within`). An error ends the measurement with no number, no part of one and no repeat, and step 4
goes on with the other servers. The first row of this table that applies is the one (the order is the check
order), every line has the prefix `LM Studio: `:

| Nr. | Condition | Line |
|---|---|---|
| 1 | a read fails (transport, limit of 10 s, no model list) | `stopped answering during the measurement` |
| 2 | a read does not show the instance, shows another id or another `context_length` | `the loaded model changed during the measurement` |
| 3 | a chat without an answer (connection gone) | `stopped answering during the measurement` |
| 4 | a chat reaches 600 s: limit of time, or a wall time above 600 s | `a request did not answer within 600 s` |
| 5 | status 400 and the body names `Failed to load model` | `failed to load the model for the request; nothing measured` |
| 6 | any other status than 200 (also 400 without the pattern, 401, 403, 429, 5xx) | `a request did not answer with status 200` |
| 7 | the body is not readable, or `finish_reason`, `prompt_tokens`, `completion_tokens` are missing or not a string / whole number, or a counter is negative | `a request carried no readable finish_reason and usage` |
| 8 | `finish_reason` is not `length` | `a request did not end with finish_reason length` |
| 9 | `completion_tokens` is not 128 | `a request did not carry 128 tokens` |
| 10 | `prompt_tokens + 128 > context_length` | `a request did not fit the context <n>` |
| 11 | wall time `<= 0` | `a request took no measurable time` |

**Several at once (AP9-PM-1).** `n = min(scenario.requests, instance.parallel, MAX_REQUESTS)` with `MAX_REQUESTS` of
`loadtest_vllm` (256). Where `parallel` is not stated or below 2, or the scenario names one request, `n = 1` and
everything above and below holds **byte for byte**: the same request bodies, the same order and number of calls, the
same lines. modelroom never changes `parallel` in LM Studio. With `n >= 2` the course is: read the list, a warm-up of
one request, read, then three batches of n requests each released together (the helper of the time of a batch, see
"The load test" above), and a read after every batch: five reads and `1 + 3n` chats, `calls = 6 + 3n` in the
success (five reads stay five; a measurement that ends early counts the calls actually made, as it does at n = 1).
Every request has its own prompt, `<prompt> Request <k>.`, `k` running on from the warm-up (0), batch 1 `1..n`,
batch 2 `n+1..2n`, batch 3 `2n+1..3n`; no number repeats in a run. That does not exclude cache effects: the
measurement is made under the cache setting the server has.

*A failing batch.* All threads of the batch end first. Then the answers are judged by request index, ascending; for
each request rows 3 to 11 apply in the order of the table, and the first request that failed gives the row; the
measurement ends without a further read. Only if no request failed, the read after the batch follows (rows 1 and
2); only if that read passed, a batch whose own wall time is 0 or less is row 11; only after all of these has
passed are the speeds computed (no division comes earlier). The order is therefore: the requests by index, then the
read after the batch, then the batch's wall time, then the numbers. A request failure goes before a read failure of
the same batch, and a read failure before a wall time of 0 (decision of 2026-10-10). Failures of the warm-up
and of the reads before stay as they were. In every case all threads have ended, no speed exists and `calls` is
the number of calls actually made. A failure of the batch itself (the clock raising, a broken barrier) is no row
of the table: it ends the measurement as an error with no number, and a thread that cannot be started ends the
started ones before the error is raised.

*The numbers.* The total of a batch is `n x 128 / wall time`, the wall time running from the release to the last
answer; `total_tps` is the mean of the three, held inside its range (`_mean_within`), with `total_min` and `total_max`.
The speed of one request is `128 / its seconds` over all `3n` requests (`request_tps`, `request_min`,
`request_max`). `LmStudioMeasurement` carries `requests` (n), `ranking_requests` (the scenario's), `parallel` and
the `total_*` fields (`None` at n = 1). With `n >= 2`, step 4 says (every number with one decimal): `<n> of <N>
requests measured` (only where `n < N`), `measured LM Studio <id>: <t> tok/s total (<min>-<max>), <n> at once, wall
time, context <c>`, `per request: <r> tok/s (<min>-<max>)`, where `parallel` was the limit (`n == parallel < N`) `LM
Studio allows <parallel> at once for this model (Max Concurrent Predictions in LM Studio)`, and the not-ranked line;
the card note reads `<t> tok/s total, <n> at once, wall time` (at most 52 characters: with whole nanoseconds, a
positive wall time and at most `256 x 128` tokens, `t` cannot exceed `32768000000000.0`; the column holds 65) and
the three further rows stay. `1 of 1 measured` is unchanged.

*What the reads still say.* The assurance below, "before and after every request", holds for n = 1. For `n >= 2` it
holds for the **borders of the batches**: a load that LM Studio does inside a batch can lie unseen in its wall time.
The short test does not show that every request that runs at the same time had the full reported context available.

Row 5 is a failed attempt to load, which is no observed change of state (row 2). The result is an
`LmStudioMeasurement` in memory: **nothing is written, nothing is ranked**, no record, no profile, no snapshot,
and nothing from `measurements.py` is imported but the prompt. Step 4 says `measured LM Studio <id>: <x> tok/s
(<min>-<max>), one request, wall time, context <n>` and `LM Studio states no load, so a busy server reads slower;
not ranked, nothing written` and counts `1 of 1 measured`; the card of step 5 carries four rows under `server`
(`LM Studio server  <x> tok/s, one request, wall time`, `package unknown; not ranked, nothing written`, `LM Studio
states no load, so a busy server reads slower`, `never asks to load; an unseen server load can affect the
time`, each note at most 65 characters), and only where no row of the table ended the measurement. The number is
a throughput of this runtime configuration (the speculative decoder may stay on), one request on the wall, and
is never mixed with an Ollama number. **The assurance, public in this text, in the README and on the card in its
short form:** modelroom never asks LM Studio to load, unload or download a model. It reads the model list before
and after every request (at n of 2 or more: before and after every batch) and drops the measurement when the loaded model changed; a load the server does on its
own inside a request can stay unseen and then sits in the measured time. A chat to the id of a model that was
unloaded a moment ago may make LM Studio load it again (just-in-time loading is on by default), which is why the
reads stand before and after every request; unloading and loading the same id with the same context inside one
request cannot be seen. **Limits:** port 1234 only; the standard model folder only; a server with the version 1
list of `GET /api/v1/models`, no minimum version; LM Studio on another computer is not read.

**One table row, two list entries (AP9-LS-E, F7 a).** A Hugging Face GGUF package has one fit and so one row
of the table, shown once with Ollama's fit in one table without a section, a prefix or a legend of its own. On a
machine whose profile names Ollama and LM Studio the pull list gives it two entries of the same rank, Ollama's and
`LM Studio · <packager> · <quant>`, with the list values `hf.co/<repo>:<quant>` and `lmstudio <owner>/<repo>@<quant>`;
a machine with LM Studio alone sees the GGUF packages and the one entry. An Ollama-library package and a tensor
package have no LM Studio entry. A section `with LM Studio` exists only where a machine has LM Studio and vLLM and
no Ollama (its GGUF rows would otherwise be titled `with Ollama`), and the ranking rule then names LM Studio in
place of Ollama; every machine with Ollama keeps its words.

**A choice of an LM Studio entry asks, then runs `lms get` after a yes (G7, S1, AP9-LS-G).** Choosing an LM
Studio entry prints `install  #<n>  lms get --gguf https://huggingface.co/<owner>/<repo>@<quant>` and never calls
`/api/pull` (the daemon pull is Ollama's alone; any entry of a third runtime is printed, nothing else). The
question exists only where all four hold: the call form is `path` or `home` (`lms_form`), the package has a
command (`lms_reason` is `None`), it is not `local already`, and the disk allows it. The order: (1) the
`install` line; (2) the disk check of step 5 (U8) over the `PullTarget` of the run with
`disk=lambda: lmstudio_disk_at(room, packages)` and the weights of the package plus the margin of 1 GB, once;
too little disk ends the branch after the existing line `not enough disk for #<n>: needs ..., ... free on ...`,
with no hint and no question; (3) the hint `lms starts LM Studio's service in the background; a vision model
also fetches its projector file`; (4) the question `Download #<n> with lms now? (<size> GB)` with the size of
the weight file, key `pull`, default No, optional (an answer file without `pull` does not download); (5) the
answer line. **No:** the lines as before (`lms is not on the PATH yet; ...` for the home form, `in LM Studio: load
it with context length <n>, then start its server ...`, `then run modelroom again: ...`). **Yes:** (6)
`download  <command> --yes` (the printed command, in the home form the shell expression, with ` --yes` appended;
the command part of the `install` line is a prefix of the command part of this line); (7) the call of
`lmstudio_get.run_lms`; (8) `package_on_disk` read again. Exit 0 and the file in place with the catalog size:
`✓ Downloaded #<n>  <file> (<size> GB) · in LM Studio: load it with context length <n>, then start its server`
as one line and `then run modelroom again: it finds the server and measures the loaded model` under it.
`Model already downloaded` (exit 0, the file in place) is the same success. Exit other than 0:
`#<n>: lms get ended with exit <code>; a canceled download resumes with the same line`. Exit 0 and the file not
in place: `#<n>: lms get ended, but <file> is not in ~/.lmstudio/models/<owner>/<repo> with the catalog size`.
A program that cannot be named or started (`which` gives nothing now, `OSError` at the start):
`#<n>: could not start lms get: <cause on one line>`. Each fault is one line and the run ends with exit 1.
`KeyboardInterrupt` writes `a canceled download resumes with the same line` once and the central handler
ends the run with 130. Where the form is `None` (no `lms` found), the package is `local already`, the entry is
locked or the disk is too small, the output is the one of AP9-LS-F: no hint about the service and no question.
The notes
break between words to what a labeled row has left of the 100 characters; a command to copy is never broken. Where `lms` is only in `~/.lmstudio/bin` the line is the fixed shell expression
(`& "$env:USERPROFILE\.lmstudio\bin\lms.exe" ...` in PowerShell, `"$HOME/.lmstudio/bin/lms" ...` elsewhere) with a
hint line; where the program is not found, the `PATH` form with the hint that it lives in `.lmstudio/bin`. Only a
package with exactly one weight file in the root of its repository and a stage of the GGUF stages without `UD-`
has a command; a split package, a file in a subfolder or another stage is locked `computed only: lms get for ...
is not checked`, an unsafe name or stage `no safe command` (the same texts under the table). A package that lies
in `~/.lmstudio/models/<owner>/<repo>/<file>` with the size of the catalog revision (`stat` only; a link that
leaves the folder makes it unknown) is `local already`; the choice and the line under the table then say
`#<n> is on disk already (size checked)` in place of the command. A package chosen for another computer is handed
over as the `PATH` line with the machine named, never run.

For a vision model, `lms` also downloads the projector (`mmproj-*.gguf`), so the download is larger than the
single weight file checked by modelroom.

**The disk (F6 a, S3, S7).** An LM Studio entry is checked against the free disk of LM Studio's model folder, read
now at the nearest existing folder above `~/.lmstudio/models` (`Disk.lmstudio`, after vLLM's `Disk.vllm`; this
computer only, never over SSH): the download of the catalog revision plus the margin of 1 GB, as for the other
two. Each entry of the list reads the place of its own runtime. The Markdown file has the disk alone, so a
Hugging Face GGUF row of a machine that names LM Studio (`Disk.runtimes`, the runtimes of its profile) is marked
there, as in the terminal, only where it is too big at the place of every one of Ollama and LM Studio that the
machine has; any other row is judged at the place of its own runtime. The `room` line names the places:
`disk 41.2 GB free on C: (Ollama and LM Studio models)` on one drive, `(Ollama, LM Studio and vLLM models)` for
three, one piece for each drive otherwise; a machine with LM Studio and no Ollama names LM Studio's place. A
machine with LM Studio keeps Ollama's budgets and its table form (the `speed` line and the reserve hint) and gets
a `runtime` row naming its runtimes in table order: `LM Studio (from its profile)` where Ollama is not among them,
`Ollama · LM Studio (from its profile)` with Ollama, and more names with vLLM (`Ollama · vLLM · LM Studio`). The
second list entry of a package (the LM Studio one) exists only where Ollama is there at the same time. An entry of
the list reads the place of exactly the runtime it names (`disk_fit.entry_place`); a runtime of the table with no
place of its own is judged at none, and falls back to the row's own `Fit.runtime` only where no runtime is named.

**`modelroom` runs `lms` at one place only.** Every `lms` call starts the application in the background and
opens the port, `lms ls` included (recording of 2026-10-08), so none is a read: detection, the call form, the
measurement and the load stay without a process. The one place is `modelroom/lmstudio_get.py`, after the yes of
the question above and never before it: the argument list `[<program>, "get", "--gguf", <url>, "--yes"]` with
the program resolved from `room.which("lms")` (form `path`) or the file in `~/.lmstudio/bin` (form `home`),
never `shell`, never a string, never the printed shell expression; standard input, output and error go through
to `lms`, the working directory and the environment are the process's own. It runs for this machine only, never
in `modelroom pull` and never for another computer (that one is handed the `PATH` line). A static test reads
`lmstudio_get.py` by its syntax tree and allows exactly the ordinary `import subprocess` and one call
`subprocess.run(argv, check=False)` inside `run_lms` (one positional argument, the name `argv`, one keyword
argument, `check=False`; `shell=`, `*args`, `**kwargs` and any second call are findings); every other LM Studio
module is read for any process call -- the ordinary import, an alias, a star import and a dynamic import with a
constant module name. In the acceptance test (`tests/test_lmstudio_runs_lms_once.py`) the no cases (no answer for
`pull`, `pull: false`) run under a trap on `subprocess.Popen` and `os.system` that fails the test at any call; the
yes case runs the unchanged `run_lms`, the trap lets through only the call whose first argument is the expected
list and fails any other, and the stand-in `lms` appends its arguments to a file, so the test demands one
process call and one entry. `os.exec*` and `os.spawn*` are guarded by the static test within those import and
call forms.

## Guided mode

`modelroom` with **no subcommand** is the guided mode (`modelroom/guided.py`): a configuration
generator over the three commands. It asks, writes `modelroom.toml`, and then calls exactly the
functions `hardware`, `fetch` and `render` call -- there is no second way of computing anything.
`hardware`, `fetch`, `render`, `migrate`, `export-profile` and `import-profile` never ask a
question.

**The terminal rule.** Without an interactive terminal on both ends (`dialog.is_interactive`)
the guided mode prints the help to stderr and exits `2` -- it never falls back to a default
answer. The one exception is `modelroom --answers <file>`: then the answers come from that file
and everything else is unchanged, the same steps in the same order with the same output.
`--answers` together with a subcommand is exit `2`.

**Ending the dialog.** Ctrl-C is exit `130`, an end of input exit `2`, and so is `Esc` in a
selection list (`dialog.TerminalAsker` binds it after `questionary` has built the question, which
takes no key bindings of its own; a text question keeps prompt_toolkit's own `Esc`, which is the
prefix of its editing keys). None of them leaves a half-written file: every write is atomic
(`atomic_write_json`/`atomic_write_text`), every write of the configuration, a profile, a
measurement or the snapshot happens under `modelroom.lock`, and no write is started in the middle
of a question.

**How every question looks** (`modelroom/dialog.py`, `modelroom/intro.py`). One style and one set
of glyphs for the whole dialog, decided once per run in `intro.py` and read from there by every
other module:

- **Color** only when `sys.stdout` is a terminal and `NO_COLOR` is not set in the environment.
  The colors are foreground colors from `docs/assets/banner.svg` -- white and a muted blue for
  the mark, green for the pointer, the chosen answer and what is done, amber for what is tight,
  cyan for a path, gray for a side note -- and the background belongs to the terminal.
- **Glyphs** (`██ ▓▓ ░░ ▄▄▄ ███ ✓ – ❯ ↑↓ · ─`) only when `sys.stdout.encoding` can encode them, else
  ASCII (`## :: .. ... ### ok - > ^v - -`). A Windows console under `cp1252` encodes none of them,
  and printing one would end a run with a `UnicodeEncodeError` instead of a dialog.
- **Every question is a list** with the arrow keys, a green `❯` on the line the keyboard is on, the
  grayed-out entries with their reason, and the instruction lines under it: `↑↓ move   Space marks
  Enter takes the marked row, or this one   Esc leave` for a list that takes one row, `↑↓ move
  Space marks   Enter takes the marked rows, or this one   Esc leave` for a list that marks. **Yes
  or no is a list of `Yes` and `No`** with the default under the pointer; there is no `(Y/n)`
  anywhere. An answer file still answers it with `true`/`false`.
- **One rule for every list** (AP9-UX C, 2026-09-28): nothing is marked in advance; the arrows move;
  the space bar marks and colors the whole row (`selected`: green, reversed; the labels of one list
  are padded to one width) and, on a marked row, unmarks it -- in both kinds of list alike (Codex
  r2); `Enter` takes what is marked, and the row under the pointer where nothing
  is marked. The pointer starts on the first row, unless a list that takes one row has a remembered
  answer: then it starts there, unmarked. `questionary` put a `select`'s default among the marked
  options and drew it green while the pointer moved on (finding 3); `dialog._bind_single_select` empties
  the marks after the build, marks the pointer row alone on the space bar and takes the mark or the
  pointer row on `Enter`. `dialog._bind_space_marks_row` (a `checkbox` only) swaps the gray title of a
  `dim` row for its plain text while it is marked, and back when it is unmarked. With something
  marked the pointer's row is a row a reader moved past, not a choice (decided 2026-09-25). Every
  binding is added after `questionary` has built the question, like `Esc`; the library itself is
  unchanged, and `FileAsker` answers as before (it takes `header` and ignores it) -- `select = []` in an answer file still means nothing.
- **The row under the pointer is a cursor line** across the whole width (`dialog._mark_pointer_line`,
  style `cursor-line`), a gray row included: it was bold only, and a `legacy` row not even that.
- **The keys stand behind the question, what the list says about itself in a gray line under the
  list** (decided 2026-09-25). Two places, because the two lines are not worth the same: the library
  gives the list window the rows the terminal has left and scrolls it around the pointer, so a line
  at the end of the list is out of sight while the pointer is at its start (measured 2026-09-25
  against `prompt_toolkit`'s renderer, 30 models in 24 rows). The keys are what a reader needs at
  the moment they cannot go on, so they stay next to the question, which has a window of its own and
  wraps. The glossary of a column -- what a column means (`last column: how many fetched packages
  fit <machine>`), what a word of a column stands for, where the fit of a row comes from -- is a
  line a reader looks up once, and it stands under the list. Both go away with the question: a
  sentence that explains a list has nothing to say once the list is gone (decided 2026-09-24).
  `Asker.select`, `Asker.checkbox` and `Asker.select_or_text` take it as `instruction`; an answer
  file ignores it. **A line of a list wraps instead of being cut** (`dialog._wrap_list_lines`): the
  library's own window does not wrap, and a 153-character line lost its last piece in a window of
  100 columns (measured 2026-09-25).
- **A list may carry a column head and gray rows.** `Asker.checkbox(..., header=None)` takes the column
  head of a table-shaped list as text (AP9-UX D): the terminal draws it as a second line under the
  keys, indented by the pointer and the marker (`dialog.HEAD_INDENT`, `guided_models.POINTER_WIDTH`),
  outside the window the list scrolls in, so it stays where it is while the pointer moves; an answer
  file ignores it, and every test double takes the parameter. `dialog.Choice(heading=True)` is still a
  line of a list that is no entry; the pointer never rests on it, and `dialog.selectable` leaves it
  out, so no answer can name it. `dialog.Choice(dim=True)` draws
  a whole row gray, for a row a reader should read as a side note rather than a recommendation (a
  `legacy` model). Without color both are the same characters they always were.
- **Every question erases itself once it is answered** (`application.erase_when_done`, a list and a
  typed answer alike), and the run writes
  one line in its place (`screen.answer_line`, "The screen"). Without it `questionary` writes its
  own idea of the answer -- `done (2 selections)`, `[this machine (measure now)]`, the whole marked
  line of the scale -- under a question that is still on screen, which is the "one thing chained to
  the next" of the test round of 2026-09-24. A list longer than the window leaves the part that
  scrolled out of it behind; that is the terminal's own doing and is accepted.
- **The labels of a list are aligned in columns** (`dialog.columns`), so the scale, the search
  hits and the load test read as tables. A cell wider than its column pushes its own row and is
  never cut.

**The screen** (`modelroom/screen.py`, decided 2026-09-24). One object draws the run. It knows each
line as `(class, text)` fragments and prints it either **with color**, through
`prompt_toolkit.print_formatted_text` and the one style of the dialog (`intro.style_rules` plus the
question's own classes), or **without color** as exactly the same plain text through the run's
`out` -- the same decision `intro.print_intro` makes for the start screen, with the same fallback
where the library cannot drive the console at all. The plain text of a line is its fragments' text
with **one leading space**, as the start screen has it, so `modelroom --answers <file>` and a
terminal run read as the same screen. `tests/golden/guided-screen*.txt` is that screen as a
fixture: a change to it is a change to those files, made on purpose.

| Pattern | What it draws |
|---|---|
| `head(n)` | `── Step 1 of 5  Configuration ────…`, filled to **72** characters (`label`; ASCII `--`), behind a blank line |
| `answer(question, words)` | `? Where should results live?  this folder` -- `?` (`qmark`), the question (`question`), two spaces, the answer (`path` for a path, else `answer`) |
| `note(text)` | two spaces of indent, gray (`note`) -- what the run made of the answer above it |
| `done(n, summary)` | `✓ Configuration   <folder>, 1 machine` -- the check mark (`done`), the name padded to 14 (`value`), the balance |
| `skipped(n, summary)` | the same with `–` (`tight`; ASCII `-`) for a step that did nothing: measured nothing because nobody asked it to, chose nothing |
| `card(facts)` | the `label value note` rows of the start screen (`intro.fact_line`), for the card of step 5 |
| `intro.label_line(label, …)` | a `<label>  <text>` row, the label padded to 8 and the text at column 10 -- the one place that decides where a labeled row's text begins, so the card and the notes under its table stand in the same column |
| `joined(pieces)` | pieces joined with ` · `, wrapped **between** pieces at 100 characters and never inside one |

**The answer line is the run's, not the library's.** Every question leaves exactly one line behind,
with the answer in the words a reader gave it (`screen.answer_words` and the step's own):

| Key | Words |
|---|---|
| `results` | `this folder`, or the path that was entered |
| `machines` | `this machine`, `a profile file`, joined with `, `; nothing marked: `none` |
| `clone` | `the same machine` / `a clone`, under the short question `The profile of this folder is gone. Is this the same machine or a clone?` (the profile id belongs to the question that was asked) |
| `search` | the word as it was typed; nothing typed: `anything that fits this machine` |
| `did_you_mean` | the candidate that was picked, or `none of these` |
| `filter_owners`, `load_test`, `pull` | `Yes` / `No` |
| `select` | the model names joined with `, ` (`Qwen3.5-9B, Qwen3-0.6B`); nothing marked: `nothing`. A value an answer file names as a repository keeps its own spelling |
| `context` | the line of the scale without its fit column, columns two spaces apart (`L  32k  24,000 words  a report or a long contract`); a number of its own: `40,000 tokens` |
| `load_test_packages` | the local names joined with `, ` |

**The start screen** (`modelroom/intro.py`), once per run, before the first question, with
`--answers` as well (then without color): the mark's pictogram from `docs/assets/banner.svg` as
three rows of four cells, next to it the name, the one-line description, and the version with the
repository from the package metadata (`importlib.metadata`, `project.urls`; without that URL only
the version).

**The mark, with color and without** (mark E, decided 2026-09-24). **With color** every cell is a
square of 24 pixels with a gap of 8 in both directions: a row of lower half blocks (`▄▄▄`) carries
the gap above and a row of full blocks (`███`) the square, so one row of cells is two terminal
lines, six lines in all, and the rows do not touch. The color of the cell tells the three kinds
apart (`mark` white, `mark-muted`, `mark-empty`) -- `▓▓` and `░░` are drawn as a coarse dot raster
by the console this package is used at most (measured 2026-09-24, Windows Terminal), which the mark
of a brand is not. The name stands on the second line, the description on the third, the version on
the fourth; the first, fifth and sixth carry no text. **Without color** the mark stays the three
rows of `██ ▓▓ ░░` it always was: a log is no place for a brand surface, and a half block cannot be
shaded. Eight pixels is the smallest gap a terminal has in either direction -- a space sideways, a
half line up and down -- so equal gaps exist only at eight, and a square is then a multiple of it.

Then the three facts that are known before anything is asked, each as `label value note`:

- `folder`: the results folder `--config` or the pointer file already names, with what is in it;
  `not chosen yet` when the first question is still to come.
- `daemon`: `Ollama <version>` from `GET /api/version` with the number of models from
  `GET /api/tags`; `not reachable` when it does not answer. Read in `intro.py` and not through the
  load test: the start screen must never end a run. **The count says how many of those models are on
  this machine** where the daemon lists cloud models of its own: `reachable, 19 models, 7 of them
  local`, and `reachable, 7 models` where there is no cloud entry at all (decided 2026-09-25). A
  cloud entry is one whose tag is `cloud` or ends in `-cloud` (`intro.is_cloud_name`, the rule
  `loadtest.is_cloud` applies, asked of the name alone because nothing on the start screen may
  raise). `19 models installed` counted twelve entries the Ollama app offers as packages of this
  machine (test round, 2026-09-25).
- `server`, one row under the daemon's for every other registered server that answers
  (`local_servers.LOCAL_SERVERS`, AP9-SV1): a vLLM server's row is `server  vLLM  context 4096,
  serves (stated) <id>`, the id last; a server that asks for a key is `server  127.0.0.1:8000  asks
  for an API key`; a server that does not answer has **no row**, so the start screen of a machine
  without one is the one it always was.
- `machine`: this machine's host name, and the hardware of the profile this folder binds it to in
  plain words (`one graphics card, 12 GB, 128 GB memory`); `not measured in this results folder
  yet` when there is none.

Under that, the five steps by name and one line about how this run behaves: **`Files are written as
the run goes on. Esc leaves a list, Ctrl-C leaves at any point.`** It deliberately does not promise
that nothing is written before the end, and it does not promise a line for every write either --
step 1 writes a configuration as soon as the folder is known, a configuration of schema 1 is
migrated on the way in before any question of a second run, and the pointer file that remembers the
folder is written without a line of its own. What a reader needs is the true shape of it: this is
not all-or-nothing at the end, and a run left in the middle has written what the steps before it
wrote.

There is no "press Enter": the first question follows. Under `modelroom --answers <file>` the start
screen is printed **without color**, whatever terminal the run was started in: such a run is read
from a log, and its output must not depend on the window (`cli._cmd_guided` passes `colored=False`,
`intro.print_intro` takes it). Where `prompt_toolkit` cannot drive the console at all (measured: a
Windows Python under a `xterm-256color` terminal), the same lines are printed without color -- the
first thing a run prints is never a traceback.

**Which folder.** `--config <path>` wins over the pointer file's `current`. A `--config` that
names no file, and a remembered folder that no longer holds a `modelroom.toml`, are said out
loud and the folder question is asked again -- neither is silently replaced by a new folder.
Since AP9-UX2 (U1, GF F1; `modelroom/guided_folder.py`) a remembered folder that is not the one the
run was started in is a question too, with `the folder of the last run (<path>)` first and the
pointer on it, then `this folder (<here>)`, then `another path`; the start screen says `the folder
of the last run, not this one`. Every comparison is on resolved paths; the same folder asks
nothing. Before anything is written or migrated, a chosen folder that is the home folder or the
top of a drive is a question (U7, GF F2): a note (`this is your home folder: modelroom.toml, docs
and state would land in it`, or `this is the top of a drive: ...`) and `Write the results into
this folder anyway?`, No by default; No asks for a path with `<home>/modelroom` offered, and at a
terminal the question comes again while the answer is such a folder.

**Five steps.** Each one is a block: a head of 72 characters (`── Step <n> of 5  <name> ───…`), one
answer line per question, at most two notes, and one balance line -- a check mark, the step's name
and what the step came to (`✓ Packages        2 models, 26 packages from 4 repositories`), or a dash
where it did nothing (`– Measurement     none in this run`). Step 5 closes with the card of the
whole run over the result table and `✓ Results         docs\models.md`, the document's path relative
to the results folder. The five are **1 Configuration** (the results folder, the configuration in
it, the machines), **2 Packages** (search, choice, fetch), **3 Context** (how many people work at the same time, then the
size scale),
**4 Measurement** (the load test), **5 Results** (the render).

The balances, one per step: `<folder>, <n> machine(s)` · `<k> models, <p> packages from <r>
repositories` (nothing chosen, and nothing to fetch: `nothing chosen, the folder stays as it is`) ·
the context as `L 32k` · `<n> of <m> measured` or `none in this run` · the Markdown path. All three
numbers of step 2 are about **the snapshot this folder now holds, narrowed to this run's models**
(`selection.narrow_snapshot`, AP9-UX B): `apply_hits` keeps the models that were already there, and
since AP9-UX the balance, like the result, counts only what this run chose -- one frame, never the
marked models next to the packages of the whole folder (second-model round, 2026-09-24).

The **fetch belongs to step 2**, before the context question and not after it: the scale of step 3
counts how many of the packages the fetch just recorded still fit this machine, and the fetch
itself does not depend on the context. An answer file is a table of keys and its order does not
matter, so the move is invisible to it.

**The questions.** Every question has a key, which is also its key in the answer file:

| # | Key | Question | Answer |
|---|---|---|---|
| 1 | `results` | Where should results live? | `here` or `path`; `last` too where the pointer file remembers a folder with a configuration that is not this one (AP9-UX2 U1, the pointer starts on it) |
| 1 | `results_path` | Path to the results folder | text (only after `path`, or after `write_home` false: then `<home>/modelroom` is offered) |
| 1 | `write_home` | Write the results into this folder anyway? | true/false, default false (only when the chosen folder is the home folder or the top of a drive, AP9-UX2 U7; an answer file whose `results_path` names one again ends with exit 2) |
| 1 | `write_config` | Write a configuration into this folder? | true/false (only when the folder already holds results but no `modelroom.toml`) |
| 1 | `machines` | Which machines should the result cover? | a list out of `this-machine`, `import`, `share`, `enter` and `profile:<profile_id>` for a profile of the folder a machine names (AP9-UX C1; `share` FR-S) |
| 1 | `import_file` | Path to the profile file to import | text (only after `import`); in the terminal an empty path or a path with no file asks again, an answer file goes on as before |
| 1 | `share_setup` | What should modelroom do with the shared folder? | `use`, `choose`, `none` (only after `share`, FR-S) |
| 1 | `share_folder` | Folder for the profile exports: | text, the form "Shared folder" checks (only after `choose`) |
| 1 | `share_files` | Which profiles should be imported? | a list of `profile_id`s the shared folder holds (only once that list is shown) |
| 1 | `entered_name` | What is the machine called? | text, 1 to 128 characters after trimming (only after `enter`) |
| 1 | `entered_class` | What kind of machine is it? | `laptop`, `workstation`, `server` or `unknown` ("not sure") (only after `enter`, always; since 2026-09-26) |
| 1 | `entered_ram` | How much memory does it have, in GiB? | a number above 0: `16` … `256` from the list, or any other, decimals allowed as text (`"31.5"`); a TOML float is no answer (only after `enter`) |
| 1 | `entered_gpu` | What runs the model? | `card`, `cards` (several graphics cards of one size, since 2026-09-26), `none` or `unified` -- the value, not the words of the list (only after `enter`) |
| 1 | `entered_cards` | How many graphics cards? | a whole number of 2 to 1024: `2`, `4`, `8` from the list, or any other (only after `cards`) |
| 1 | `entered_vram` | How much graphics memory, in GiB? | a number above 0, as `entered_ram` (only after `card`; after `cards` the question is "How much graphics memory per card, in GiB?") |
| 1 | `clone` | Is this the same machine or a clone? | `same` or `clone` (only on `ask_clone`; both measure -- `same` under the bound profile id, `clone` under a new one) |
| 1 | `machine_class` | What kind of machine is this? | `laptop`, `workstation`, `server` or `unknown` ("not sure"): the class of this machine, asked before every measurement (since 2026-09-27; the class the chassis reads is marked and offered without `unknown`, and choosing it keeps the source `chassis`); `unknown` leaves the class `unknown` |
| 2 | `search` | What are you looking for? | text: a word, a name with blanks, a repository id, or the empty string ("show me what fits this machine") |
| 2 | `filter_owners` | Show only repositories of a publisher or a listed packager? | true/false |
| 2 | `did_you_mean` | Nothing was found. Did you mean one of these? | one of the candidates, or `keep` (only when the search found no model and the catalog knows a word close to the one that was typed) |
| 2 | `select` | Which of these models should the result cover? | a list of base model ids (`Qwen/Qwen3.5-9B`); an answer file may name repository ids instead, as before |
| 3 | `at_once` | How many people work with it at the same time? | a whole number of people at the same time (`1..1024`, origin `entered`), `"N people"` for a daily head count (`1..10240`, origin `from_users`), `"12 requests"` as before, or `"kept"` for the line a folder kept; asked before `context` |
| 3 | `users` | (the same question) | the key of 1.0.0 and earlier, read only from an answer file and only when `at_once` is absent: a whole number of people of a day, or `"12 requests"`; a file holding both keys ends with exit `2` |
| 3 | `context` | How much text should a model handle at once? | a level of the scale (`"XS"` … `"XXL"`), or a whole number of tokens; the pointer starts on `[guided].context` when this folder kept one, else on `L` |
| 4 | `load_test` | Measure the speed on this machine now? | true/false, default false; one question for every local server (AP9-UX E); **one of the two optional answers** -- an answer file that does not mention it does not measure |
| 4 | `load_test_packages` | Which of these should be measured? | a list of local Ollama names, `vllm` for a vLLM server that states a model and `lmstudio` for an LM Studio server with exactly one loaded model (only after `load_test` true, where a server offers a row) |
| 5 | `pull_package` | Which package should be pulled into Ollama? | the Ollama name of a row of the list, or `none` (AP9-UX2 U6); optional: without it `#1` is taken where it has a pull name of its own, else nothing; a name the list does not offer, or a locked row, is exit 2 |
| 5 | `pull` | One key for the one choice of a run, two question texts, each naming the row chosen and adding its weight: Ollama's `Pull #3 into Ollama now? (5.0 GB)` and LM Studio's `Download #7 with lms now? (2.9 GB)` (AP9-LS-G) | true/false, default false; **the other optional answer** -- an answer file that does not mention it does not pull and does not download. Common conditions: a written document, a chosen row that is not local already, and enough disk or disk unknown. For an Ollama row also an Ollama name of its own and a daemon the start screen reached. For an LM Studio row also a safe command (`lms_reason` is `None`) and the call form `path` or `home`, with no Ollama condition at all (a machine without Ollama asks it for LM Studio); a yes runs `lms get ... --yes` once (see "`modelroom` runs `lms` at one place only") |
| 5 | `pull_other` | Which package should go onto another computer? (AP9-FR A3, `pull_other.py`) | `<machine> <pull name>` (a value flat like every other, `answers.py:26-33`) out of the shown rows of every other **chosen** machine, or `none` -- optional: without it nothing is chosen, silently (one target machine at most, GF question 6); a value the list does not offer, or a locked row's, is exit 2; asked whenever at least one other chosen machine has a shown row with a pull name (A2), whether or not this machine has a row of its own -- an import-only run is asked too (Codex r1-1); the chosen row's command (`modelroom pull <name> --weights-bytes <n>`) is printed, never sent anywhere -- `modelroom pull` (A4, below) is what runs it, pasted on the machine it names |

**Every write reads the file again first.** The dialog takes as long as the user takes, so the
configuration is read again immediately before it is changed and written -- an entry another
process added in between (an `import-profile` on a shared folder) is otherwise thrown away by the
copy this run loaded at the start.

**And nothing written in between is lost** (decided 2026-09-25; until then the window between that
read and the write stood open). Every write of this mode goes through `guided_write.update_config`:
the step hands in its **change** (a function from the configuration it reads to the one it writes,
or nothing to write), the file is read once and the text of that very read kept, the change is
applied, and `write_configuration` compares the file with that text under the lock. When another
run wrote in between, the step reads again, applies **its own change** to the new state, writes
once more and goes on with what it really wrote; a second conflict in a row ends the run with exit
`2`, like any other step that cannot go on (a held lock stays `1`, an unsupported schema `3`). No
merge of two configurations is attempted: the six changes are the machine entries for the profiles
the folder holds, this machine as the writer, the profile a measurement recorded, the entry of a
machine entered by hand, the models the search added and the context the scale kept. Two of them
have a rule of their own:

- **The first write of a new file** is no change that could be applied again -- it is a whole initial
  state with empty families. When another run created the file meanwhile, this run takes that file
  over as it is and writes nothing: the initial values never replace existing ones, and this
  machine's entry is added by the writer step, as in any folder someone else set up
  (`guided_write.create_config`).
- **The machine entries for the profiles the folder holds** are worked out again as a whole on the
  new state -- the scan, which profiles are configured, the names, the reserves -- so a name another
  run took meanwhile is not written over and a profile it entered is not entered twice; with nothing
  left to add, nothing is written.

The checks of the profile record (a `[paths]` that moved, a machine entry that is gone) hold again for
the read after a conflict. `migrate` and `import-profile` read the file under the lock already and are
unchanged. The guarantee covers writers that take `modelroom.lock`: an editor that saves between the
comparison and the replace is outside it.

**Profiles the folder already holds get a machine entry.** The render computes one ranking per
configured machine, so a profile that no `[machines.<name>]` names would be left out of the
result. Whenever the guided mode has a configuration in hand, every profile in the folder without
an entry gets one, by the same rule `import-profile` uses (name from the `display_name`, reserves
from `[defaults]`, `writer = false`).

**Step 1, the results folder.** A new `modelroom.toml` is written with `schema_version = 2`,
families empty, `packagers` empty (the search writes every resolved repository as an
owner-bound `repos` target, so a speculative `<packager>/<name>-GGUF` probe under five accounts
per base model would only spend the shared request budget), this device as
`[machines.<this_machine>]` (its neutral key from `~/.modelroom/remote.json`, "Neutral names") with `writer = true`, `paths.state = "state"`, `paths.markdown = "docs/models.md"` and
`[guided].results = "."` -- relative to the file's own folder, which is the results folder, so they
read as `<folder>/state`, `<folder>/docs/models.md` and `<folder>` ("Neutral names"). The folder is
remembered in the pointer file with its full path. A
`modelroom.toml` of schema 1 in the folder is migrated first (`modelroom migrate`, under the
lock, backup kept, idempotent), before anything is written, and the notes it leaves have one
sentence in front of them that says what happened in plain words: `this folder holds a
configuration from an earlier version; it was updated, backup kept: modelroom.toml.v1.bak`. A
folder that already holds results but no `modelroom.toml` is a question, never an assumption.

**What step 1 leaves on the screen** (decided 2026-09-24): the answers, and **one note about the
measurement** -- `measured: <the hardware in plain words>, confirmed by llmfit`, or `measured again:
…` where this machine was already measured in this folder (and for a `the same machine` answer);
where llmfit read a quantity otherwise, the note ends with `; llmfit reads RAM X GiB, this
measurement Y GiB` per deviating quantity instead, and nothing about llmfit where its check is
absent, failed or not comparable (decided 2026-09-27);
a binding to a profile entered by hand is no measurement of this machine, and its first one says
`measured:`. The
sentence ends after the hardware where `llmfit` confirmed nothing. The readings themselves, the
profile id, `wrote <toml> with <name> as the writer of this results folder`, `<name> is now a writer`
and `<name> is measured as profile <id>` are the configuration's and the profile file's business and
are no longer printed (`cli.hardware_with_config(..., summary=False)`; `modelroom hardware` prints
its summary line unchanged). The notes of the measurement itself -- what `nvidia-smi` lists, why the
llmfit cross-check did not run, why no fit is computed for the profile -- go through the screen as
notes, indented like their neighbors and without `note: ` in front (`hardware_with_config(...,
note=run.screen.note)`, AP9-UX G); `modelroom hardware` still prints them as `note: <text>`. An import leaves `imported <display_name>: <the hardware in plain
words>`; an import that does not go through stays a `run.failed` sentence. A machine entered by
hand leaves `entered <name>: 64 GiB memory, graphics card 24 GiB` (or `…, no graphics card`, `…,
shared memory`), the sizes as they were entered; `<name>` is the profile's neutral name, and the
answer line of the name typed in says `not kept: the machine gets a neutral name` ("Neutral names").

**Step 1, the machines** (AP9-UX C and C1, `modelroom/guided_machines.py`). A note stands in front
of the list: `another computer: pick its profile below, or import the file made there`. The list,
from the top, nothing marked:

1. the profile this folder binds this machine to, where it was measured here (value
   `profile:<profile_id>`, `this machine (server-5c1e9a07, as measured 2026-09-28)`);
2. `this machine (server-5c1e9a07, measure again)` with row 1, else `this machine (measure now)`
   (value `this-machine`); the name is the `display_name` of the profile -- the one name the result
   shows for this machine (FR-N2), never its host name ("Neutral names") and not this computer's key
   `this_machine`, which stays the run's internal token and is not printed; where no profile was
   measured here -- none yet, or the binding is to a profile entered by hand -- there is no name and the
   row says `this machine (measure now)`. Row 1 adds the class of the profile only where the name does
   not carry it (`laptop-5c1e9a07` says `laptop`; an alias, or another class in the prefix, keeps it:
   `this machine (gpu-box, server, as measured ...)`);
3. every other profile of the folder a `[machines.<name>]` names, one row each (value
   `profile:<profile_id>`, `<label>  <hardware>`: the label as the import forms it, with the short
   id where two carry one name; the hardware as `2 x NVIDIA L4 44.98 GiB VRAM / 94.29 GiB RAM`, a
   machine entered by hand as `shared memory 32.00 GiB (entered)`), by hardware, then label;
4. `import a profile file`; 5. the shared folder (value `share`, FR-S: `set up a shared folder for
   profiles` with none set, else `profiles from the shared folder` -- always there, and never a
   look at the folder itself, "Shared folder" below); 6. `enter a machine by hand`;
7. grayed out: a profile no machine of the configuration names (`no machine of the configuration
   names this profile` -- each run gives every profile of the folder an entry at its start, so this
   happens only when one was removed by hand afterwards), then the schema-1 files (`run modelroom
   migrate`), then the files that do not read (with their reason).

So `Enter` alone takes the profile this machine already has, where there is one, and measures only
where there is none -- which is what a first run is for. A binding to a profile entered by hand
(which comes about only by hand) is no measurement of this machine: there is no row 1, the profile
stands in row 3, and row 2 says `measure now` (decided 2026-09-26). A marked profile puts the
machine(s) naming it into the selection of this run (`RunSelection.machines`); nothing is measured
or written for it. The marked entries run in this order: measure, import, enter. The key and the
answers `this-machine`, `import`, `enter` stay the same, so an answer file is unaffected; the answer
line names this machine by its class (`this machine (server)`, without a known class `this machine`;
its profile's name stands in the list only) and a marked profile by its label.

`this machine (measure now)` (or `this machine (<name>, measure again)`) makes sure `[machines.<this_machine>]` exists and is a writer -- this
machine is the one that fetches -- and then measures through `cli.hardware_with_config` with the
results folder as the binding key. The takeover rule decides which profile is written
(`resolve_profile_target`, "Profile binding"); its `ask_clone` case is the `clone` question, and
the local fingerprint for it is read on its own (`measure.read_os_identity`), so the question
comes before the measurement. **Both answers measure** (decided 2026-09-24): `a clone` measures
as a new identity, `the same machine` measures again under the profile id this folder already
binds this machine to (`--same-machine`, "Profile binding"). A folder someone emptied is a folder
to measure into; the run that answered "the same machine" and then stood in front of an empty
step 3, an empty step 4 and "nothing to render" was the hand test this rule comes from. After a
measurement the guided mode writes `[machines.<name>].profile` -- the one configuration write
`hardware` leaves to it. Before each measurement the chassis is read on its own
(`measure.read_machine_class`) and the run asks `machine_class` every time (decided 2026-09-27;
until then only where the chassis named none). Where the chassis names a class, the list is the
three classes with that one marked and the pointer on it; an answer equal to it keeps the source
`chassis`, any other is handed on as `hardware_with_config(..., machine_class=)` (source
`entered`). Where it names none, the list is the four entries with nothing marked; "not sure"
leaves `unknown`, with the chassis note. So an answer file that measures this machine carries
`machine_class`; an earlier answer is never taken over, the class is read again on every
measurement.
`import a profile file` runs `import-profile`, which never changes the
binding; a file that does not import is reported and the run goes on. **The shared folder**
(FR-S) is "Shared folder" below.

**`enter a machine by hand`** (decided 2026-09-26, `modelroom/guided_entered.py`) sizes a machine
from its data sheet -- one that is not in the house yet, or one no measurement here covers (the
automatic measurement covers NVIDIA only). The questions, in this order: `entered_name`,
`entered_class`, `entered_ram`, `entered_gpu`, only for `cards` `entered_cards`, and for `card`
or `cards` `entered_vram` (keys in the table above). The sizes are GiB above 0, decimals allowed;
an empty name, a name over 128 characters, a size that is no number, `0` or below, or a card count
that is no whole number of 2 to 1024, or cards whose sizes add up to no number, is asked again in the terminal, the note naming the key; from an answer file it ends the run with exit `2` and names the key, before anything is written. An answer file written before 2026-09-26 has no `entered_class` and ends with exit `2`
naming it. The answers make one of the four shapes of a machine entered by hand ("Hardware
profile v2", the table under "A machine entered by hand"): `card` → `gpu_state` `entered` with
its graphics memory, `cards` → `multi_gpu` with n cards of the size per card and their sum,
`none` → no graphics card, `unified` → `unified_memory`; the class from `entered_class` with
source `entered` (`unknown`/`unknown` for "not sure"); always `origin` and `ram_physical_source`
`entered`, no fingerprint (`none`), no RAM limit, both cross-checks `absent`, no `gpu_name`, no
card names, no `llmfit_version`, `recorded_at` the time of the run. Every view names the class
first and leaves it out when it is `unknown`: `server, 2 graphics cards 24 GB each, 128 GB
memory, entered`. A value the profile refuses is exit `2` with
the profile's own reason, never a traceback. Then, in this order:

- **The file.** `<state>/hardware/<profile_id>.json`, written with the writer `hardware` uses and
  under `modelroom.lock`, like every writer of a profile; a lock another process holds ends the run
  with exit `1` before anything is written. Under the lock, immediately before the write, the new
  `profile_id` is checked against every file name in that folder, in any letter case -- a schema-1
  file and one that does not read carry no `profile_id` a scan could see, and neither is ever
  replaced. A file that cannot be written is a `run.failed` sentence; the run goes on.
- **The entry.** `[machines.<name>]` with `profile = <id>`, `writer = false` and the reserves of
  `[defaults]`, the name from the `display_name` by the rule of `import-profile` (a name that is
  taken gets the short id). It is a change like every other write of this mode, so it is worked out
  again after a conflict; a name that cannot be found, or a `[paths]` whose state folder no longer
  holds the profile, is a `run.failed` sentence, and the profile stays where the next run enters it
  ("Profiles the folder already holds get a machine entry").
- **No binding.** The pointer file stays as it is: a machine entered by hand is never this machine.
  The load test measures bound profiles only, so it never measures one ("Load test (stage 1)").
- **Entered again is new.** Entering the same machine again makes a new profile; the one before
  stays, as a measured one would. There is no editing and no deleting of it in the dialog.
- **Measuring this machine never writes into it.** When a binding or `[machines.<name>].profile`
  of this machine points at a profile entered by hand -- neither comes about by itself, both can be
  written by hand -- the takeover rule answers `new` before the fingerprint ("Profile binding", "A
  machine entered by hand is never adopted"): no clone question, a fresh profile for this machine,
  the entered file unchanged byte for byte. Through the home binding, the entered machine keeps its
  own entry and the result has both blocks. Through `[machines.<this_machine>].profile`, the measurement
  binds that entry to the new profile (the profile record of step 1), so the entered profile is
  without an entry for the rest of this run -- one block -- and the next run enters it again (two
  blocks).

Where it stands, it says so: the block head and the Markdown heading read `<name> (entered)`, the
Machines table `<name> (entered) (<profile_id>)`, and the card of step 5 `graphics card 24 GB, 64 GB
memory, entered` (or `no graphics card, …`, `shared memory 32 GB, entered`) -- no date and never
`measured`. Unified memory computes as one pool, the memory minus both reserves, and its rows say
`shared memory` ("Fit contract v1", "Unified memory").

**Where a machine of the run came from** (AP9-UX2 U2; `RunSelection.origins`, in memory only). Step 1
keeps imported and entered machines apart; each chosen machine is `this` (measured in this run, or
its profile is the one bound to this folder and measured here, `profile:<id>`), else `imported`
(its profile id came from this run's import, even one the folder held already), else `entered`,
else `folder`. On the card, under the machine row of every machine that is not `this`, a line
without a label: `imported profile · not this computer`, `profile from this folder · not this
computer`; an entered machine keeps its own words. The card of an empty step 5 does the same. The
table head says `· imported` or `· another computer` (see "The terminal view"); the Markdown view is
unchanged, since the document does not know the origin.

**Step 2, search, choice and fetch.** The two search questions, the search itself and its notes live
in `modelroom/guided_search.py` (`search_step`, since 2026-09-25); the choice and the write-back into
the configuration stay in `guided.py`, which owns the file. `run_search` gets the run's shared request
budget (`DEFAULT_GUIDED_BUDGET`, 150, shared with the fetch), `open_pages=not filtered` and
`publisher_required=filtered` -- so the owner
filter is a question about the **request** and about the **resolution**, not only about the list: with
it on, only the accounts
are asked and only a base model of a catalog publisher resolves; switching it off adds the two open
lists on top of the account groups and lets any account's base model resolve (see "Search over
the Hugging Face API"). A typed repository id is never hidden by it.

**A search that found no model, for a word the catalog does not know, ends in one more question**
(decided 2026-09-25): the candidates as a list to pick from, `none of these` last and the closest one
under the pointer. Picking one searches again with that word; `none of these` keeps the answer as it
was. The **first** search leaves no line behind in that case -- the notes and `search.json` are
written once, for the search that was really used, so the step still carries one reading.

Then **two notes** (`guided_search.search_notes`, decided 2026-09-24): where it
asked (`searched Hugging Face at the publisher Qwen and the five listed packagers`, the publishers
whose page answered with something, by name; with the filter off `, and the two open lists`) and how
much of the answer is a choice (`120 repositories, 40
of them models you can pick from`; none: `none of them a model you can pick from`; one repository
without `of them`, decided 2026-09-25: `1 repository, a model you can pick from` or `1 repository,
not a model you can pick from`; and with a full
account page ` · a more specific word shortens the list`). Where a publisher of the catalog **was**
asked and every one of them answered with nothing, the first note names them instead of saying "at no
publisher of this catalog that answered" (decided 2026-09-25): `searched Hugging Face at the five
listed packagers; the publisher deepseek-ai has no GGUF repository for this search` -- "for this
search", because an account page is asked with the word and an empty answer says nothing about
everything that account holds (second-model round, 2026-09-25). The two searches that ask no
account page have a first note of their own -- `searched Hugging Face for the repository
unsloth/Qwen3.5-9B-GGUF` and `searched Hugging Face for the 22 current models of this catalog` --
because the account sentence claimed "the no listed packagers, and the two open lists" about them
(live probe, 2026-09-25). Above those two stand the
search's own sentences about the *request* wherever it has one -- a typed id with no GGUF file, a word
no catalog family matches, a run with no word at all, a catalog plan that stopped at the budget.
Everything else the search knows goes
into `search.json` ("Search log"): the seven account lines, the request count, the budget and every
reason a repository is no model of the list said the same thing seven times over on a screen that
had a choice to offer. Then **the list of models** (`modelroom/guided_models.py`,
decided 2026-09-24, in place of the list of repositories: 120 lines of over 200 characters, size
and age `unknown` throughout and 40 grayed-out rows whose appended reason broke the columns --
"I cannot tell what to pick, or what I am risking" was the hand test's verdict).

**A column head, then one line per model** (`ModelChoice`), and only the models that can be picked.
The head is the list's `header`, no row of it (`guided_models.list_header` returns the text, AP9-UX
D): `Model  Fit  Size  Release  Packagers  Downl.  Ollama`, drawn under the keys outside the window
the list scrolls in and indented by `POINTER_WIDTH` so its columns stand over the cells. As the first
row of the list it scrolled out of sight once the pointer went down (71 rows, pointer on row 13).
Without a head a reader had to guess what a cell meant and `latest` was invisible between two columns
of numbers (test round, 2026-09-25).

Seven columns, and the **whole** line is at most 100 characters -- ` ❯ ○ ` included, which is the
one space every line of this screen carries plus the pointer and the marker (`POINTER_WIDTH`, 5;
a label of 100 made a line of 105): the model's name (the part after the `/`, 24), the fit in a word
(14: the class itself, `fits`, `tight`, `no_fit`, `unknown` -- and `fits (RAM)` or `tight (RAM)` where the
graphics memory is too small and the fit is against system memory: fit v1 caps that pool at `fits`,
and live on 2026-09-24 a 122B model stood as `fits` above a 9B `tight` that fits the graphics
card, with nothing to tell the two apart), its size (6: `9B`, `0.6B`, `2.4T`, `unknown`), where it
stands in its family (`Release`, 7: `latest`, `legacy`, `–`, and `latest*` or `legacy*` where the
status is **computed** from the version numbers of the family rather than stated -- "Computed
release"; the successor of a `legacy*` row is in `search.json`, not in the row), the packager accounts that have it
(12: all of them where they fit, else the first one and how many more, `unsloth +3`, else cut), the
downloads of all of them together (6; an indication, not a rank) and the Ollama name (the rest of the
line, 14). `Release` is one constant (`guided_models.RELEASE_COLUMN`), so the head, this section and
the tests cannot name it differently. The packagers had 16 characters and the Ollama name 10 until
2026-09-25, which cut 24 of the 32 Ollama names of the shipped catalog; with 12 and 14, 27 of them
stand whole (decided 2026-09-25).

**The Ollama name of a line** is the one a hit of the model names (the catalog's, or one the user
typed), else the `ollama_base:ollama_tag` pair the configuration holds for that base model -- the
fetch uses that pair, so the list shows it rather than a `–` (decided 2026-09-25). Which name wins
when the configuration **writes** the choice back is unchanged: a name a hit carries replaces a
configured pair, and a hit without one leaves the pair as it is (`apply_hits`).

**A `–`, not a word, where a cell has nothing to say**: `none known` and `unknown` read as facts
about the model in a column of their own (test round, 2026-09-25), so the Ollama column and the
`Release` column carry `–` (AGENTS.md, the language standard, carries that rule for every column of
a list or a table). `none known` stays the word of the search log (`search.NO_OLLAMA_LABEL`), which
has room for it. **Every cell is cut to its column** with `…`:
`dialog.columns` pads and never cuts, and a name, an account, a download count and an Ollama name
all come from a registry. A registry name longer than the Ollama column (`qwen3-embedding:0.6b`) is
cut here; the install line of step 5 carries the whole one.

The order is the rule of the result as far as a list of models can apply it (ADR 0009, AP9-UX): the
memory pool (a fit in graphics memory before one in system memory, whatever its class), then every
model that is not `legacy` before a `legacy` one (`latest` and a release nobody knows alike), then
the larger model (`parameters.parameter_count`; an unknown size last), then the downloads, then the
name, with `no_fit` and `unknown` last; a computed status sorts like a stated one of the same word.
With an empty search word the largest model that fits and is not overtaken stands first. A **`legacy` row
is drawn gray as a whole** (`dialog.Choice(dim=True)`), stated or computed: the family has moved on,
and that belongs in the reading of the row rather than in one cell of it.

How many models and how many repositories the search answered with is the second note above the
list, and the reasons of the repositories that are no model of it are in `search.json` ("Search
log"; `guided_models.UNRESOLVED_REASONS`, and with `filter_owners` on `not a publisher or a listed
packager` for an owner class of `other` -- except for the one repository the person typed, which
that filter never hides) -- the repositories themselves are no lines of the list either. What the
list has to say about itself is its **instruction line** (`guided_models.hint_line`), under the
list and behind the line of the keys: `latest: the publisher's current release of its family ·
legacy: a successor is named · *: computed from the version numbers of the family, not stated by
the publisher · Fit is memory only: the package's need against a pool · ample: up to 60 % of the
pool · fits: up to 85 % · tight: up to 98 % · no fit: more · (RAM): the pool is the system memory`
(a line longer than the window wraps, it is never cut; the fit words since 2026-09-27,
`screen.FIT_SCALE_HINT`). Where the fit comes from stands **on top of the list** instead
(`guided_models.fit_line`, passed as the `notice` of the list): a line of its own between the line of
the keys and the column head, indented like the head, so it is in sight whatever the length of the
list -- `fit from size at 32k context and 1 at once (up to fits); exact after the fetch`, always with
the context and the requests the fit was really computed for. -- With no measured machine in this
folder the legend holds the release words alone, and the line on top is `fit unknown until this
machine is measured`. What nothing
marked would do is no longer said there: `Enter` takes the row under the pointer now.

**The fit of the list** is `fit.fit_from_parameters` ("Fit from size", basis `size`) against the
profile this folder binds this machine to and the reserves of its `[machines.<name>]` -- the same
machine the size scale of step 3 counts against (`guided_context.machine_checked`). Its context is
`[guided].context` when the folder kept one, else **the context of the level the scale will start on**
(`guided_context.DEFAULT_CONTEXT`, 32768, since 2026-09-24 -- and one constant, not two): the list
said `fit at 8k context` while the very next question started on `L 32k`, and two numbers for one run
are one too many (test round, 2026-09-24). A fit stands on a number either way, never on a level.
The parameter count is what a
repository states (`SearchHit.parameters_b`), else what the model's name says
(`parameters.parameters_from_name`: every `<number>B`/`<number>M`/`<number>T` that does not stand
behind a letter -- with an `E` in front allowed, an effective size -- the largest of them, `2.4T` as
2400 and shown as `2.4T` again, `360M` as 0.36, `E4B` as 4 (`M` and `E` since 2026-09-25, when the
computed release took its sizes from this same rule) -- so `Qwen3.5-35B-A3B-MTP` is 35, `Nova-A17B` and
`Qwen-Image-2.1` are nothing), else nothing, and then the fit is `unknown` with `parameter count
unknown` rather than a guess. `machine not measured` is the other such reason.

**A chosen model becomes two repositories**: the publisher's own, when the search resolved one,
and the first listed packager's in the order of the positive list. Of an account's builds the
plain `<name>-GGUF` repository is the one, where the account has it, else the account's first in
the order of the search (`unsloth` answered `-MTP-GGUF`, `-GGUF` and `-UD-GGUF` for one model,
newest first; decided 2026-09-24). A model that only accounts of
neither list hold is fetched from the one with the most downloads -- otherwise picking it would
add nothing at all. Two, because every repository costs the fetch about ten requests of the shared
budget (measured: two repositories spend 20 of 150) and the ranking shows the packages of both;
choosing the packager per model is stage 2.

`select` takes base model ids, and an answer file may name repository ids as it always could --
such a value picks exactly that repository and not the pair, so the answer files written before
this list run unchanged. A value that is **both** a base model of this search and a repository of
it is read as the **base model**: that is what the list offers, so that is what a person can have
marked, and nothing forbids the collision (a repository may package one model and be the base
model of another). A value that is neither ends the run with the answer file's own message
(`not a list out of ...`). When
nothing can be picked the list is shown all the same, `Enter` goes on, and the run says `no
repository of a publisher or a listed packager was resolved; nothing added`. The chosen hits go
through the unchanged `apply_hits` into the configuration, which is written with
`write_configuration`. A hit
whose base model the shipped catalog has no family for gets the family name
`search.family_name_for` derives from the repository name; the guided mode does not ask for one.
Then `fetch_with_config` runs with the same budget object (nothing configured yet means nothing to
fetch, said out loud).

**Step 3, at the same time** (`modelroom/guided_context.py`, decided 2026-09-26, question changed
2026-10-10), asked **before** the scale: the last column of the scale and the fit of the ranking
compute with the same requests. `How many people work with it at the same time?` is a list of `1`,
`5`, `10`, `50`, `100`, each with what it is (`one person`, `a small team`, `a department; consider
a serving stack such as vLLM`, then `a serving stack such as vLLM`), and `enter a number`. For this
question, modelroom assumes one request at a time per person working with the model. The
instruction line under the list: `one request at a time per person; a daily head count is about
10 times as many (rule of thumb, not measured)` (`config.ACTIVE_SHARE`). The pointer starts on
`[guided].requests` when its origin is not `default`, else on `1`; a kept number that is no line of
the list, and every kept `from_users` answer, gets a line of its own, `kept in this folder` (with
`(<users> people)` for `from_users`), and the pointer starts there.

- **A whole number** is the requests: origin `entered`, `users` stays what the file holds when the
  answer is written, or none; only the display uses it. `N requests` (also `1 request`) is the same.
- **`N people`** (also `1 person`) is a daily head count: `requests = requests_from_users(users)`,
  origin `from_users`.
- **`kept`** keeps the stored `users`, `requests` and `requests_origin` unchanged.
- **Anything else**, `0`, `-1`, `abc`, more than 1024 at once, a head count above 10240, is asked again in the terminal, the note naming it; from an answer file it ends the run with exit `2`: `'<answer>' is not a valid answer for step 3: answer a whole number of people
  at the same time (1 to 1024) or `N people` for a daily head count (1 to 10240)`.
- **The answer line**: `? How many people work with it at the same time?  5 at once`, or
  `25 people (3 at once)` for a head count.
- **An answer file** may answer `at_once` or, as in 1.0.0 and earlier, `users` (a whole number is then a
  head count of a day); both in one file end with exit `2`, `the answer file holds both users and
  at_once; answer only one of them`; the answer line is the same in both cases.
- **The write**: `[guided].users`, `requests` and `requests_origin` together
  (`guided_write.with_requests`), right after the answer, compared with the file as it is then and
  not with this run's copy -- the rule of the context below. The configuration that comes back is
  the one the run goes on with: the scale is computed for the machine and the reserves **it** holds,
  not the ones read before the question. After the scale the answer is applied once more,
  together with the context (`guided_write.combined`), so an answer another run wrote while the
  scale was asked cannot leave this run's scenario, its document and the file apart (second-model
  round, 2026-09-26). A second run in the same folder offers the kept answer.

**Step 3, the context as a size scale** (`modelroom/guided_context.py`). Six levels, in this
order, each a line of the list: `XS` 4096, `S` 8192, `M` 16384, `L` 32768, `XL` 65536, `XXL`
131072, with the shown context (`4k` … `128k`), roughly how many words that is (three quarters of
a token each, so `4k` is `3,000 words`), an example of what it is for, and the last column below.
Which machine that column was computed against stands in the **instruction line** of the question
(`last column: how many fetched packages fit <machine>`), and the answered question is the whole
screen of this step: `checking <machine>`, the sentence about the last column and `kept context <n>
in <toml>` all went (test round, 2026-09-24). A machine a fit cannot be computed for is still a note
(`Checked.reason`).

- **The pointer** starts on the level whose context this folder kept (`[guided].context`), else on
  **`L`** -- the default of a folder that has chosen nothing yet.
- **A kept context that is no level** gets a seventh line of its own, `custom <n> tokens kept in
  this folder`, and the pointer starts there. An eighth line, `enter a number`, leads to the text
  question for a number of tokens; it starts empty, because what this folder kept is already the
  line under the pointer.
- **The last column, "how many fit"**: for every level, fit contract v1's own formula
  (`fit.count_fitting` over `compute_fit_v2`, the chosen number of requests, 16-bit KV cache per
  request) over the packages of
  this folder's snapshot that a fit may be computed for (`render._eligible_packages`), against the
  profile **the pointer file binds this machine to for this folder** (the same source step 4
  measures into, and never `[machines.<name>].profile`), with the reserves of that machine's
  `[machines.<name>]`. "Fits" is the ranking rule's own predicate -- a package
  `ranking.rank_packages` ranks -- so for the same packages, the same profile and the same reserves,
  the scale and the result table of that run say the same thing. It is one computation, not two. Text: `all N packages fit`, `k of N packages fit`, `none of N packages fits`. The
  count does **not** have to fall as the scale goes up: fit v1 judges a package against the
  graphics memory while it fits there and against system memory once it does not, so a bigger
  context can move a package into a roomier pool and make it fit again. It is computed **when the
  question is asked**, from the configuration and the snapshot as they are then; like every other
  list of this dialog it is what was true at that moment, and the write that follows reads the file
  again on its own. Without a bound profile there is no column and one line instead: `no measured
  machine in this folder yet, so the scale shows no fit`; with a profile the fit must not compute
  for (`profile.fit_block_reason`), that reason; when no `[machines.<name>]` names that profile,
  that reason as well -- another machine's reserves under another machine's name would be a made-up
  number; without a snapshot, every level says `no packages yet`.
- **The cap** (AP9-UX H): only the base models of this run count. When every one of them declares an
  `Architecture.max_context`, the levels above the **largest** of those are grayed out with `beyond
  the window of every model of this run` and cannot be answered, in an answer file either (exit `2`,
  naming the reason). If one of them declares none, nothing is grayed out -- a window nobody knows is
  no reason to forbid a level. One model with a 32k window no longer locks XL and XXL for one with
  256k: a package beyond its own model's window is `not covered` in the fit itself ("Fit with profile
  v2"), so the count of the scale and the ranking agree.
- **The answer** is a level name or a number of tokens; the level is translated into its tokens
  before anything is written. An answer out of a file is stripped of surrounding blanks **before**
  it is held against the grayed-out entries, so `" XXL "` is refused exactly as `"XXL"` is.
  Anything else is asked again in the terminal, the note naming what was answered; from an answer file it ends the run with exit `2`.
  `context_origin` is **`entered`** for everything this dialog writes, 8192 included: `default` is
  what the three automation commands assume when nobody chose a context
  (`measurements.DEFAULT_CONTEXT_REQUESTED`, `measurements.default_scenario`), and choosing `S` is
  not that.

**The context is kept.** The file is read again ("Every write reads the file again first"), and the
answer is written to `[guided].context` whenever it differs from what **that** read holds -- the
first time as well, and for 8192 as well, because a kept context is a decision and not a default.
An answer that the file already holds writes nothing. The comparison is deliberately against the
fresh read and not against this run's own copy: a run that skipped its write because its copy
already said so would leave a context another process wrote in the file, and the next
`modelroom render` would compute a ranking this run never showed. The write goes through the same
`write_configuration` every other step uses. What it buys: a `modelroom render` of this folder
afterwards computes the same ranking (see "Render (schema 2)", "The scenario"), instead of falling
back to 8192 and putting the run's own measurement into group 1.

**Step 5, the render and the card.** `render_with_config` with the chosen scenario and the selection
of this run (`selection=run.selection`, AP9-UX B: the card, the tables and `docs/models.md/json` name
only the machines and the models this run chose; with no model chosen the step says `this run chose
no model, so the result ranks none`), which writes both
views -- so a measurement the load test just wrote is in the ranking of that same run -- and hands
the document itself back (`on_document`). From it the step draws **the card of the whole run**
(`result_card.result_card`, re-exported by `views`, the `label value note` rows of the start screen, decided 2026-09-24), then one
table per machine (`views.terminal_lines`), then `✓ Results         docs\models.md`. The card is the
report a reader looks at once the run is over:

| Row | Value | Note |
|---|---|---|
| `folder` | the results folder | -- |
| `machine` | one row per machine of the result, its `display_name` | `12 GB graphics, 127 GB memory, measured today` (else `measured <YYYY-MM-DD>` from the profile's `recorded_at`); a machine without a profile carries its own reason |
| `models` | the base models of the snapshot, by name | `<p> packages from <up to three packager accounts, else "and n more">` |
| `context` | the scenario sentence of the Markdown head after its first word, which is the row's own label: `32k (entered), KV cache f16 (assumed), 3 requests (from 25 users)` (decided 2026-09-26) | -- |
| `load` | only beyond `OLLAMA_REQUEST_HINT` (8) requests: the hint of the Markdown head's `Load:` line, wrapped on rows of its own so no line is wider than 100 columns | -- |
| `speed` | `not measured`, `none counted` or `<n> measured` | `fastest <name> at 41.1 tok/s, protocol v2` (the protocol of the counted measurement, `v1` for a stored record of version 1; a document without `speed_protocol` gives `fastest <name> at 41.1 tok/s`, no version is guessed); where no row carries a speed, the rows whose measurement counts nowhere here, each with its reason as the `speed` note under the table says it (`#3 measured with 1 request, ranking assumes 3`, `#8 measured, not valid`, `#2 measured, not comparable`, `#4 measured at context 32k, this ranking's is 8k`; the kinds of both notes in one list, in the order the ranking first meets them, joined by ` · ` where that fits 65 columns, else the first; by count, `6 rows measured with ...` or `6 rows measured, not counted here`, only where the ranks of one kind alone would make the row wider than 100 columns), the value then `none counted`; only without such a row the value is `not measured` and the note `say Yes in step 4 to measure an installed package` (for a machine entered by hand `a machine entered by hand is never measured`) -- the rule of the `speed` note under the table, which the card takes over and no other (AP9-FR-N2, c2) |
| `server` | only for a registered server step 4 read a served model of (AP9-SV1): `vLLM` | whether its number counts for this run (AP9-UX E): `48.2 tok/s total, 20 requests at once, counts for this run`, `..., not counted` (another model or another context), `..., counts, no fit` (no package of this folder); `not measured in this run`; a measurement fault its reason (`invalid (<reason>)`, `not comparable (<reason>)`), cut at 65 columns. On a row of its own without a label `identity stated · fit: 32 at once, fits` (`no fit computed` without a fit, `no package of this folder` without a package). The same rows on the card of a run with nothing to rank (`views.show_nothing`) |
| `result` | the Markdown path **relative to the results folder** | `<ranked_total> packages ranked, <k> no fit` (`0 no fit` where there is none), and every other set-aside reason with its count behind a `·` |

The `speed` and `result` rows count **one** machine's ranking: the machine of the run where it has a
ranking, else the first machine that has one, and only where none has one the machine of the run
(decided 2026-09-26: a run on a machine without a profile next to a machine entered by hand counted
`0 packages ranked` above a table of 25). A package that fits two machines is one package and two rows, and a sum over the
machines would count it twice while the `models` row of the same card counts the packages of the
snapshot once (second-model round, 2026-09-24). The tables below the card carry every machine.

`Written to <markdown>   and   <state>` is gone from the screen: the card's `result` row and the
balance of the step both say where the document is, and the run said the same path three times
(test round, 2026-09-24). `modelroom render` still prints it.

**A step 5 with nothing to rank** -- the folder holds **no snapshot** -- says so and draws the card
all the same (`views.show_nothing`, decided 2026-09-25): the note `nothing to rank: no package in
this folder yet`, the card with the rows it still knows -- the folder, the machines of this run's
selection (`selection.narrow_config`, AP9-UX B) with the profiles this folder holds (`render.machine_profile`, the same decision the document
would have rested on), `models none`, the context that was chosen, no measurement (with the reason
`a machine entered by hand is never measured` where every profile the folder holds was entered by
hand, `scenario_text.never_measured_of`) and `result –` -- and the balance `– Results         nothing
to rank`. Before that the step was a head with nothing
under it and the reason stood on stderr, where no reader of the run is (test round, 2026-09-25). The
exit code is the render's, unchanged (`1`, "nothing to render").

**Every other render that writes no document draws nothing here at all**: a lock another process
holds, a snapshot of an unsupported schema, a view that could not be written. None of those is a
folder without packages, so none of them may be reported as one; the command's own sentence and its
own exit code stand (second-model round, 2026-09-25). A snapshot that holds packages but none a fit
may be computed for is **not** this case either -- the render writes its document, and the table
says `no package of the configured base models is ranked here` with `shown     0 of 0 packages`.

**The pull** (`modelroom/guided_install.py`, decided 2026-09-25: the README promised the install
from step 5, and Ollama's own word for it is `pull`). After `✓ Results` the step asks one more
question, behind a blank line, **only** when all three hold: the render wrote a document; this
machine's first row has an Ollama name -- the very row and name of the install line
(`views.install_target`, one choice for both); and the start screen read a version from the daemon
(`intro.Intro.daemon_reachable`, carried along from the same call, never read out of the words of
the note). The question is the Yes/No list of the dialog, default `No`:
`? Pull #1 into Ollama now? (6.1 GB)  No` -- the size is the row's `weights_gib`, not the memory
column of the table. Its answer file key is `pull`, optional like `load_test`. **The install line
stays in every case**, for anyone who prefers to paste it.

**Since AP9-UX2 the user chooses the package** (U6, GF F4; `modelroom/pull_choice.py`; ADR 0009 and
its rule stay unchanged). Where a shown row of this machine has an Ollama name, the step first asks
`Which package should be pulled into Ollama?` (key `pull_package`): one row per row of the table,
regular and appended measured ones, in table order -- `#1  unsloth · Q4_K_S  5.0 GB
hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_S`, the name last and whole, `local already` behind a row the
daemon provably holds -- then `none, do not pull`. A row without an Ollama name is locked (`no safe
pull name`, AP9-FR A2 -- the same lock a name `PULL_NAME_RE` rejects gets, below), so is every row
whose name is not unique (below). Since AP9-RM U11 a row whose download is
larger than the free disk of this machine, as step 5 read it, stands in light red with what it needs
in place of its name -- `#1  unsloth · Q6_K  94.1 GB  needs 94.1 GB plus 1 GB room, 26.4 GB free` -- and is locked
too (an answer file that names it is exit 2; one without `pull_package` then pulls nothing). A
package the daemon provably holds already needs no disk: it is never marked as too large -- not in
the table, the `disk` line, the install line, `docs/models.md` or this list -- and stays a `local
already` row that can be chosen, with its note as before (U11c; `room.pull_need_bytes` is 0 for it).
It is marked as held, though: the table of that computer says so on its `local` line ("The screen",
`local`), and so does the list of the active SSH target in `pull_other` (AP9-FR-N2, a). The
pointer starts on the first row that can be chosen, on `none, do not pull` where none can; nothing
is marked. After the choice the pull part names it, `install   #3  ollama
pull <name>`, then checks the disk, then asks `Pull #3 into Ollama now? (5.0 GB)`; a local row is
the note instead and no question; `none` is nothing. An answer file without `pull_package` takes
`#1` where it has a name of its own, else nothing -- the behavior before.

**Every name here already passed `PULL_NAME_RE`** (AP9-FR A2, "Guided mode", `install`, above): a
row's name is `pull_choice.entry_pull_name`, and `pull_other`'s (below) is the same function --
both go through `screen.install_name`, the one choke point that checks the pattern before any name
is ever returned. A row this pattern rejects is indistinguishable from one with no name at all: `no
safe pull name`, locked, no install line, no `POST`.

**`PullTarget`** (AP9-FR, the rebuild of A4) carries what the choice, the disk check and the pull
itself need -- `daemon`, `disk` (a callable), `now`, `screen`, `asker`, `failed`, `where` -- so
`modelroom pull` (below) reuses the same disk check and the same pull, error and success handling
with no results folder and no configuration of its own. Behavior here is unchanged; see
"`modelroom pull`" for what it adds.

**The disk, checked again before the pull** (U8, GF F6). Measured fresh after the choice, never the
number of the table head: the place Ollama keeps its models is approximated (`room.py`:
`OLLAMA_MODELS`, else the user default `~/.ollama/models` and on Linux the service default
`/usr/share/ollama/.ollama/models`, the one that exists; none: the user default; both: unclear),
measured at the nearest existing folder, on its drive (`C:`, a UNC path is `network share`, a POSIX
path its mount point). Need = the weight files of the package + `PULL_DISK_MARGIN_BYTES` (1 GiB).
Free at least the need: the question. Less: no question, no POST, `install   not enough disk for
#3: needs 37.9 GB, 21.4 GB free on C:` -- no `run.failed`, the command above stays to copy.
Unknown or unclear: the question, with the note `disk space where Ollama keeps its models is not
known` before it. A pull that fails on a full disk all the same is a fault as below.

**Pull names that are not unique** (U4, GF F7). The name of a pull (`hf.co/<repo>:<quant>`, or an
Ollama name) does not tell file forms apart. `pull_choice.pull_collisions` collects every name more
than one package identity answers to, over every active, complete GGUF or Ollama package of the
run's base models **before** the eligibility filter and whatever its provenance -- an approved row
can have an `unresolved` counterpart the render hides. Such a name belongs to no row: every row with
it says `#2 has no unique pull name: 2 files answer to it` in its `shown` line, is locked in the
list, has no install line (`#1 has no command: its pull name is not unique`) and no disk check --
also where the counterpart is not shown. Rows that look alike (the same `packager · quant` among
the rows shown) name their repository under the table, `repo      #2  unsloth/Qwen3.5-9B-GGUF`,
with the file form where one repository has both (`BF16/ (folder)`, `(single file)`); the Markdown
view says the same under its table. Nothing is merged, no measurement moves.

- **Local already.** Before asking, the step reads `/api/tags` and applies the load test's own rule
  (`loadtest.installed_candidates`): a registry package by its manifest digest, a `hf.co/…` name by
  its name **and** its weight digest through `/api/show`. A row proven that way is no question but a
  note: `install   #1 is local already · say Yes in step 4 to measure it`. A name the daemon lists
  **without** that proof -- old content, several weight files -- is asked like a missing one: the
  pull brings the registry's build of that name, which for Ollama is an **update** (a new manifest,
  the blobs it already has reused). This package deletes nothing; that Ollama may remove layers no
  manifest uses any more is Ollama's own clean-up, which this package does not prevent.
- **The pull.** `POST /api/pull` with `{"model": "<name>", "stream": true}` through the daemon
  transport ("Load test (stage 1)", 60 s per read, no limit for the whole). The screen says
  `pulling   <name>`; at a terminal one more line is written again in place with the current
  layer's share (`pulling   2.1 of 6.1 GB` -- Ollama's `total` and `completed` belong to one layer,
  `completed` may be missing, and nothing is summed over the lines) and erased when the pull ends;
  under `--answers` there are only the first line and the last. The pull ends at `{"status":
  "success"}`, and then `/api/tags` is read once: the name has to be there, and for a registry
  package with the manifest digest the fetch recorded (the manifest digest of a `hf.co/…` name is
  Ollama's own, not the GGUF's, so the name is what is checked). Then:
  `✓ Pulled    <name> · 6.1 GB · say Yes in step 4 of the next run to measure it`. Nothing is
  measured in this run.
- **Faults**, each one line and never a traceback, and each a step that did not finish
  (`run.failed`, exit `1`, the document is written all the same): `the pull of <name> did not
  finish: <reason>` for a daemon that does not answer, a read limit that was reached, a status other
  than 200, an `error` line, an answer that ends before `success`, a line that is no UTF-8 or no
  JSON, and a line of an unexpected shape (a status that is no text, a count that is no whole
  number from 0 below 2^63); a daemon's own text is printed with every control character, line
  breaks included, as a space, and the live line is cut to the terminal's width, so a fault stays
  one line and the live line can be erased; `<name>: pulled, but the daemon does not list it` when
  the list after it does not name it. The connection is closed however the pull ends. **Ctrl-C**
  during a pull adds `a canceled pull resumes with the same command` and goes up to the central
  handler, which says the run was stopped and ends with `130` -- both lines on purpose: Ollama keeps
  what arrived, and the next pull of the same name goes on from there. A `No` is no fault.
- **What is not checked:** where the daemon really keeps its models (it cannot be read from outside
  the daemon; since AP9-UX2 the disk is checked at the approximated place above), whose daemon
  answers on the port (the documented limit of the
  transport), and another process pulling the same name at the same time -- Ollama shares the
  progress of one pull between its callers, and this step reads whatever the daemon reports.
- **What is not written:** no configuration, no state file, no document. The pull changes the
  daemon and nothing else.

**Step 4, the load test** (stage 1, "Load test (stage 1)" above; `modelroom/local_servers.py`
with `modelroom/guided_loadtest.py` and `modelroom/guided_vllm.py`, which `guided.py` calls between
the context and the render). **One mould for every server** (AP9-UX E, 2026-09-28): first every
registered server says what it has to say, in the order of `LOCAL_SERVERS`, the reason a number will
not count among it; then **one** question, `load_test` (`Measure the speed on this machine now?`,
default **no**); on a yes **one** list, `load_test_packages` (`Which of these should be measured?`):
the Ollama candidates of this run's models, where the start screen read a served model the row
`vllm` (`vLLM server  context 4k  20 requests at once  <id>`) and, since AP9-LS-M, where LM Studio has exactly one loaded model the row `lmstudio` (`LM Studio server  <instance id>, context <n>`), none of them marked; `Enter` with nothing marked takes the
pointer row, one candidate, never all (it measured all three in the hand test of 2026-09-24). The key `load_test_vllm` is gone (0.2.0 was never released). Each server has `offer(run,
config, scenario, results_dir) -> Offer` (its notes, its rows, why it could not be read) and
`measure(...) -> str | None` (`local_servers.LocalServer`, replacing `step`).

The daemon's part. A measurement runs one request (protocol v2): the measurement gets a copy of the
run's scenario with `requests = 1` and the same context, the render of step 5 the whole scenario. When
the ranking assumes more than one request, the step's first line says so: `measurements run one
request; this ranking assumes 3` (with another server on the start screen `Ollama measurements ...`)
-- it explains the reason a measured row then carries in step 5 (decided 2026-09-26). It measures
into the profile **the pointer file binds this machine to for this results folder**, and only when the
takeover rule (`resolve_profile_target`, "Profile binding") says `bound` for it -- the profile file is
in the folder and its `os_fingerprint` agrees with this machine's. Never into
`[machines.<host>].profile`: a configuration is shared, and a stale or foreign entry there would give
this machine's speed to another device. Without a binding, without that agreement, or without a
snapshot to take packages from, the daemon says so in one line and offers nothing. Without a binding
the line is `no load test on this computer: it has no profile in this results folder`; "this
computer" because a second round for the active SSH target may follow in the same step, and the line
is about the first (AP9-FR-N2, c1). **Nor where this
machine is not part of the run's result** (Codex F2): where no machine this run chose names the
profile this machine is bound to, its measurement would land in a profile the result does not show,
so the step says `no load test on this computer: it is not part of this run's result` and offers no
Ollama row. Otherwise its
inventory is read and matched against the active packages of **this run's models**
(`selection.narrow_snapshot`, AP9-UX B). What was left out is named first, **one note per reason**
with the number of installed models behind it and **no names** (`12 installed models -- <reason>`);
the **cloud models** only where the daemon has no candidate; an installed model whose name matches a
configured package but whose digest the daemon did not show is named either way. With candidates the
note is `Ollama: 2 installed packages of this run's models, one request each`; without, `no ranked
package is installed on this machine; the load test measures installed packages only (stage 1: no
download)` where no other server offers a row, else `Ollama: no package of this run's models is
installed here`. Before the runs the step notes that the load is read once and that the model behind
the name must not change, and the progress of each run is a note of its own. Each measurement is
written as its own file and reported in one note: `measured <name>: 41.1 tok/s (39.8–42.3), context
32k, protocol v2` (the protocol of the record shown); a record that is not valid or not comparable carries **what it is missing** instead (`<name>:
not comparable (<reason>)`) and the run still ends `0`.

**A run that ends early (AP9-FR-N2, f; GF decision 2026-10-01, option A).** Where a stored record is
`invalid` (protocol `v1` or `v2`) and its first invalid run did not end with `done_reason 'length'`,
a second note follows the fault line, with the neutral name the fault line has (`record.ollama_name`)
and only what was observed and the condition it breaks: `<name>: run 1 ended after 2 of 128 tokens
(done_reason 'stop'); protocol v<k> counts a run only when it reaches 128 tokens (done_reason
'length')`, `k` the version of the record, read from `protocol_for(record.protocol)`. Only for a
stored `v1` record the sentence ends with `; protocol v2 sends the prompt through the model's
template, say Yes to measure again`; a `v2` record gets the sentence without it. The
numbers are `record.runs[i].eval_count`, its `done_reason` and `protocol.options.num_predict`, read
from the record **after** the sink (so `done_reason` is a known word or `unknown`, `eval_count` an
integer); no cause is named on the screen. It is display only: the stored reason and the record stay
as the protocol of their own version forms them, a speed run of such a model counts nowhere under
that version, and a protocol that fits it is a new protocol version, not an edit of v1 ("Measurement
protocol v1"). The same note stands for the measurement of the active SSH target
(`guided_remote_measure`, through `report_problem`).

**Stored records, named before the question (AP9-FR-M, F6).** Before the question `Measure the speed on
this machine now?` the daemon's part names, for each candidate it offers, the newest stored record of
the bound profile that belongs to that package (the ranking's own rule, `ranking._matches`), has the run's
`context_requested` and one request, and is no measurement (`_record_problem`): the fault line and, for
a run that ended early, the second line of `early_end_line` (for a stored `v1` record with the hint).
A valid, comparable record, a record of another context and a file that cannot be read say nothing
here. The newest record decides: where it is a migrated record (`protocol` `none`) the candidate says
nothing, and no older record of the package is shown in its place.

The vLLM part (AP9-SV1, "Local servers and the vLLM load test"). The server runs on this machine, so
it passes the daemon's own check, one function for both (`guided_loadtest.local_measure_target`,
Codex r3 and r4): first the identity -- the profile this machine is bound to is in the folder and its
`os_fingerprint` agrees with this machine's (the takeover rule's `bound`) -- then the choice -- a
machine this run chose names that profile. Where either fails the step says the daemon's note with
`vLLM: ` in front (`vLLM: no load test: profile from home binding has another os_fingerprint (<id>)`,
`vLLM: no load test on this computer: it is not part of this run's result`) and offers no row -- nothing is
measured, and neither the balance nor the card counts a number. Otherwise, where the start screen read a served
model: `a vLLM server answers here, context 4096`, `it states it serves <id>`, `revision unknown (not
stated)`, `one warm-up request, then three batches of 20 requests, 128 tokens each` (`1 request` for
one: `scenario_text.requests_words`, the one place a number of requests is said). Then the first
reason that holds, in this order, why its number will not count (`<name>` is the part behind `/`):

1. another model: `the number will not count: the server serves <name>, this run chose <name>`;
2. another context: `the number will not count: the server runs context 4k, this run asks for 32k`;
3. no package of this folder: `the number counts, no fit: no package of this folder comes from
   <root>`.

The number **counts** where the served `root` is a base model of this run (the ones step 2 chose;
none where it chose none, Codex F1) **and** the server runs this run's context; vLLM is never ranked. A
line wider than 100 columns breaks before the name, indented like a note; no name is cut. A
measurement is `measured vLLM: <t> tok/s total (<a>–<b>), context 4k` and `per request: <r> tok/s
(<c>–<e>)`; an invalid or not comparable one is `vLLM: <reason>` in their place, a capped one says `256
of 257 requests measured` first; then `identity stated by the server; not ranked, nothing written`.
The fit follows for the package the server serves (`vLLM fit, computed: ...`, `32 requests of context
4k at once, from size: fits`); without a profile the fit may compute for, `no vLLM fit: <reason>`;
without the package, the reason of `match_served`. A server that asks for a key says so and offers no
row. Nothing of it is written.

**The states** (`local_servers.load_tests`; "vLLM" means the start screen read a served model):

| Ollama | vLLM | Question | List | Balance | Exit |
|---|---|---|---|---|---|
| not bound, bound to another machine's profile, or not chosen | any (the same check stops vLLM) | none | -- | `none in this run` | 0 |
| no snapshot | none, a key, or a fault | none | -- | `none in this run` | 0 |
| no snapshot | yes | yes | `vllm` only | yes: `vLLM measured` or `vLLM ran, not counted`; no: `none in this run` | 0 |
| no daemon | none, a key, or a fault | yes | -- | no: note `no load test: <reason>`; yes: `nothing measured: <reason>` (`run.failed`) | 0 / 1 |
| no daemon | yes | yes, the note `Ollama: <reason>` before it | `vllm` only | as row 2 | 0 |
| daemon, no candidate | none, a key, or a fault | none, the note `no ranked package is installed ...` | -- | `none in this run` | 0 |
| daemon, no candidate | yes | yes | `vllm` only | as row 2 | 0 |
| candidates | none, a key, or a fault | yes | the candidates | `<k> of <n> measured` | 0 |
| candidates | yes | yes | the candidates and `vllm` | `<k> of <n> measured · vLLM measured` (or `· vLLM ran, not counted`) | 0 |
| any | -- | -- | -- | a measurement a server lets down: `nothing measured for <name>: <reason>` (`run.failed`) | 1 |

`vLLM measured` is said only where the number counts **and** the measurement holds (valid and
comparable); where it does not count, or the measurement is invalid or not comparable, the balance
says `vLLM ran, not counted` -- the card's server row says why. The document is written in every
state.

**The answer file** (`modelroom/answers.py`) is TOML with `schema_version = 1` and one key per
question; an answer may be text, a whole number, `true`/`false` or a list of texts. A question
with no answer ends the run with exit `2` and names the question -- an answer file written for 0.1.0
has neither `at_once` nor `users` and stops at step 3 until one is added (`users = 1` keeps its old ranking); an answer that is not one of
the offered choices, or of the wrong shape, does the same. An unsupported `schema_version` is
exit `3`. **The three exceptions are `load_test`, `pull` and `pull_other`:** a file that does not
mention `load_test` measures nothing -- on no server -- one that does not mention `pull` does not
pull, and one that does not mention `pull_other` chooses nothing for another computer (AP9-FR A3),
because whether any of the three questions is reached at all depends on the machine the file is run
on -- its daemon, its servers, what they already hold, and (for `pull_other`) whether another
chosen machine has a package of its own -- not on the file. The key `load_test_vllm` of
AP9-SV1 is gone (AP9-UX E): the vLLM server is the row `vllm` of `load_test_packages`. An answer of the wrong shape (`pull = "yes"`) is still
an error. A `load_test = true` without `load_test_packages` is a missing answer like any other.
Since 2026-09-27 `machine_class` is asked on every measurement of this machine, so a file that
measures it carries the key.

```toml
# modelroom guided answers
schema_version = 1
results = "here"
machines = ["this-machine"]
machine_class = "laptop"
search = "qwen"
filter_owners = true
select = ["unsloth/Qwen3.5-9B-GGUF"]
# A level of the scale, or a number of tokens: `context = 8192` and `context = "8192"` are the same
# answer as `context = "S"`.
context = "L"
load_test = true
load_test_packages = ["hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M"]
pull = false
```

**A step that did not do what it was asked does not end the run.** A measurement that failed, a
`fetch` that did not complete, an import that did not go through, a `the same machine` answer with
no profile to measure into: each is said out loud and the run goes on with what is there -- one
machine's trouble must not cost the ranking of the others. **The exit code still says so:** a run
that wrote its document but had such a step ends with `1`, not `0`, and prints how many steps did
not finish. Only a run in which every step did what it was asked ends with `0`.

**Exit codes**: `0` a document was written and every step finished, `1` another process holds the
lock, there was nothing to render, or a step did not finish, `2` no terminal and no `--answers`,
an end of input, a missing or unusable answer, or a step that cannot go on at all (the message
says which), `3` a stored file's `schema_version` is unsupported or a stored file does not read,
`130` Ctrl-C. A failure of a stored file keeps the exit code the command that owns it would give:
the guided mode does not flatten `SchemaVersionError`, `MigrationError`, `PointerFileError` or a
held lock into its own `2`.

## `modelroom pull`

`modelroom pull <name> [--weights-bytes N] [--yes]` (AP9-FR A4, `modelroom/cli_pull.py`) pulls one
package straight into this computer's Ollama daemon -- no results folder, no configuration, no
`--machine`. It is the command a guided run's `pull_other` handover line names (above), run *on*
the machine that should get the package, whether or not modelroom ever measured that machine.

**Steps, in order, each ending the command at the first one that fails:**

1. `<name>` against `pull_choice.PULL_NAME_RE` (A2, "Guided mode", `install` above); outside it,
   the command never looks at Ollama or the disk at all: exit `2`.
2. `GET /api/version` against the local daemon (`daemon.LocalDaemon`, `127.0.0.1:11434` only, the
   same origin every other command is held to); not reachable: `Ollama does not answer on this
   computer at 127.0.0.1:11434`, exit `1`.
3. `GET /api/tags`: the name already listed there gets a note, `Ollama lists <name> already;
   pulling it again updates it`, and the run goes on -- without a package there is no digest to
   prove it holds the *same* build, unlike the guided run's own check ("The pull", below), so this
   is deliberately weaker and never skips the question.
4. The free disk (`room.free_disk`, the same reading "A download larger than the free disk" uses)
   against `--weights-bytes` plus the disk margin: too little says `not enough disk: needs <need>,
   <free> free on <drive>` and pulls nothing, exit `1` (no `#<rank>` in the message: there is no
   table row here, just this one name). Disk unknown, or `--weights-bytes` not given at all: a
   note, and the run goes on regardless.
5. `Pull <name> into Ollama now? (<size>)` (the size left out where `--weights-bytes` was not
   given), default No: a terminal asks it, `--yes` skips it outright and answers Yes, and without
   either: `no terminal to ask; add --yes`, exit `2`. **No** pulls nothing, exit `0`.
6. **Yes** pulls it, with the same progress, success and error handling as the guided run's own
   pull (`_send`, `_end_reason`, `pulled_problem`, `screen.pulled_line` -- without a package known
   here, `pulled_problem` checks only that the daemon lists the name afterwards, never a manifest
   digest: "ohne Digestteil"). A fault is one line and exit `1`; success prints the same line the
   guided run's own pull does, size included where one was known: `✓ Pulled    <name> · <size> ·
   say Yes in step 4 of the next run to measure it` -- the sentence names the guided run because
   this command reuses its success line unchanged (`cli_pull.py` passes no `hint` of its own).

**The catalog check (AP9-KO, V10; behavior of P1, data of P0).** After step 1 and before step 2 the name is
looked up in the catalog read offline (`Catalog.pull_check`, "Package findings and the load check", "Direct
pull"). A name the catalog assigns to a package with an `absent` check ends the command with exit `2` and the
reason of the display, with the day of the check; `--yes` does not pass it. A name nothing can be assigned to --
no quantization, two packages, an Ollama name -- takes the way described here unchanged.

**`PullTarget`** (`guided_install.py`) is what steps 4 and 6 draw on either way: seven fields --
`daemon`, `disk` (a callable, read fresh at the check, not a value), `now`, `screen`, `asker`,
`failed`, `where` (empty here and for the guided run's own row; `" on <alias>"` is FR-B1's). The
guided run builds one from itself; `modelroom pull` builds one from the local daemon and this
machine's room alone. `pull_chosen` (the guided run's own last question) takes either a
`PullTarget` or a `GuidedRun` and converts the latter; every helper under it (`_disk_allows`,
`_is_local`, `_holds`, `_pull`, `_send`) takes only a `PullTarget`.

**Exit codes**: `0` pulled, or the user (or the answer given) said No, `1` the daemon did not
answer, the disk does not hold the download, or the pull itself faulted, `2` `<name>` is not a
name `PULL_NAME_RE` allows, or there was no terminal and no `--yes`, `130` Ctrl-C (nothing left
half pulled: the daemon's own pull is not resumed by this command, only Ollama's own retry of the
same name does that).
