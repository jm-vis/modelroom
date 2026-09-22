# Contracts

How shared data shapes are defined, versioned and tested in this repository. The models
themselves land in `modelroom/contracts.py` with the first feature work package; this file
fixes the rules they follow so that fetcher, hardware probe, renderer and any external
consumer build against the same expectations.

## Source of truth

- `modelroom/contracts.py` holds Pydantic models with `extra="forbid"`, explicit value ranges
  and validators for coupled fields. It is the only definition.
- This file describes each model for readers and repeats its examples verbatim. A contract
  test loads every example from the module and validates it; another test checks that every
  model named here exists in the module and vice versa.
- Producers validate before they write. A file on disk that does not validate is a bug in the
  producer, not something readers repair.

## The three persisted forms

| Form | File | Written by | Read by |
|---|---|---|---|
| Configuration | `modelroom.toml` (or a `Configuration` object) | the user | every command |
| Snapshot | `<state>/modelroom.json` | `fetch` | `render`, `check` |
| Hardware profile | `<state>/hardware/<machine>.json` | `hardware`, on that machine | `render` |

Runtime-only files in the state directory (`modelroom.lock`, `run-status.json`, `*.tmp`) are
not contracts; they are documented with the command that owns them.

## Schema versions

Every persisted form carries an integer `schema_version`. Each reader declares the **half-open**
range `[low, high)` it accepts (`version >= low and version < high`), never described as
"inclusive" -- the upper bound itself is already refused. A file outside that range is refused
with exit code `3` and a message that names the file, its version and the accepted range
(stated as `>= low and < high`), before the command writes anything. A version bump is a
documented decision (`docs/adr/`), never a side effect.

`modelroom.contracts.load_snapshot(data)` is the entry point for reading a persisted snapshot:
it checks `schema_version` against `SNAPSHOT_SCHEMA_RANGE` with `check_schema_version` *before*
handing the payload to `Snapshot.model_validate`, so a missing, non-integer or out-of-range
version always raises `SchemaVersionError` -- never a pydantic `ValidationError` mixed in with
unrelated field problems. `Snapshot` itself still carries its own `schema_version` check as a
second line of defense for callers that build one another way.

## Identity

Package identity is the repository identifier as the registry spells it (`hf_repo` for
Hugging Face, `ollama_base` plus tag for Ollama). The package never normalises, canonicalises
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
| `0` | success: every fetch area is `complete` |
| `1` | at least one fetch area ended `incomplete` (complete areas were still published); or `fetch` stopped at another process's lock; or this run's `run_at` is not newer than the stored snapshot's -- the latter two write nothing |
| `2` | the configuration is missing or invalid; the named `--machine` is not a `writer` in this configuration; or a required external tool (`llmfit`) is missing or below the minimum version |
| `3` | an input file (configuration or snapshot) has an unsupported `schema_version` |

## Models

Every example below is `EXAMPLES["<ModelName>"]` from `modelroom/examples.py`, verbatim; a
contract test parses this file for `### <ModelName>` headings and their `json` blocks and
checks both directions: every model in the module is documented here, and every heading here
names a real model with a matching example.

### Architecture

A base model's transformer shape, read from the *publisher* repo's `config.json` (packager
GGUF repos never carry one). `kind` is `"dense_classic"` only when the three numeric fields
are all present, `layer_types` is either absent or entirely `"full_attention"`, and none of
the MoE fields below signal a mixture-of-experts model; a hybrid or MoE config, or a missing
file, resolves to `"unknown"` with the numeric fields left at `None`. The MoE check looks at
both the top level of `config.json` and, when present, its `text_config`: any of
`num_local_experts`, `num_experts`, `n_routed_experts`, `num_experts_per_tok` or
`moe_intermediate_size` present with a value other than `null` rules out `dense_classic` even
when the three numeric fields all look plausible. Presence is the signal, the value is not
interpreted: a dense decoder does not carry these fields at all.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `source_repo` | `str` | -- | the publisher repo the config was read from |
| `source_revision` | `str \| None` | 40-hex commit sha, or `None` | the commit the config was read at |
| `kind` | `"dense_classic" \| "unknown"` | see above | whether the shape is a plain decoder-only transformer |
| `num_hidden_layers` | `int \| None` | `> 0`; required when `kind == "dense_classic"` | transformer block count |
| `num_key_value_heads` | `int \| None` | `> 0`; required when `kind == "dense_classic"` | KV heads (for GQA/MQA KV-cache sizing) |
| `head_dim` | `int \| None` | `> 0`; required when `kind == "dense_classic"` | per-head dimension |
| `layer_types` | `list[str] \| None` | `None` or all `"full_attention"` when `kind == "dense_classic"` | per-layer attention kind, when the config states one |
| `max_context` | `int \| None` | `> 0` | the model's trained maximum context length |

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
| `hf_repo` | `str` | `owner/name` | the publisher's exact Hugging Face repo, the base model's identity |
| `repo_aliases` | `list[str]` | repo *names*, non-empty, never `owner/name` | further exact packager repo names that count as this base model |
| `ollama_base` | `str \| None` | both set or both `None` with `ollama_tag` | the Ollama library model name, when one exists |
| `ollama_tag` | `str \| None` | both set or both `None` with `ollama_base` | the Ollama library tag naming this base model's size |
| `publisher` | `str` | -- | the organisation that trained the model |
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

A human decision that a specific package content is trustworthy, despite what its metadata
says. An approval is bound to the exact content it was given for; once that content moves on
(a new commit, a new manifest digest) the approval is void and `decide_provenance` falls back
to the metadata rules, never a blanket `metadata_ok`.

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
invariant, not just a convention `decide_provenance` happens to follow.

`complete` is never accepted as a bare caller-supplied fact: a model validator derives it from
`files` with `shards_complete` and rejects a `Package` whose `complete` contradicts that
derived value. It stays a field in the serialized form -- readers of a snapshot see it without
recomputing it -- but its value is always the one `shards_complete(files)` produces.

Three more invariants tie `provenance`, `unresolved_reason`, `format` and `approval` together:
a non-`"gguf"` `format` requires `provenance == "unresolved"` and `unresolved_reason ==
"format"`; `provenance == "unresolved"` requires a non-empty `unresolved_reason` (never `None`,
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
| `base_model_hf_repo` | `str` | must exist in the snapshot's `base_models` | which `BaseModelSpec` this package belongs to |
| `format` | `"gguf" \| "tensor" \| "unknown"` | -- | the weights format |
| `files` | `list[PackageFile]` | -- | the files that make up this package |
| `complete` | `bool` | derived from `files` by `shards_complete`, see above | whether all weight shards are present |
| `quantization` | `str \| None` | a `QUANT_ORDER` member, or `None` | `None` means unknown and sorts last |
| `default_context` | `int \| None` | `> 0` | Ollama's `num_ctx`; `None` for Hugging Face |
| `provenance` | `"metadata_ok" \| "approved" \| "unresolved"` | -- | see `modelroom/provenance.py` |
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
  "default_context": null,
  "provenance": "metadata_ok",
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
| `schema_version` | `int` | must equal `SNAPSHOT_SCHEMA_VERSION` (currently `1`) | the snapshot's schema version |
| `run_at` | `datetime` | -- | when this `fetch` run started |
| `areas` | `list[Area]` | -- | the fetch units this run covered |
| `base_models` | `list[BaseModelSpec]` | -- | the base models this run knows |
| `packages` | `list[Package]` | see above | the packages this run found |

```json
{
  "schema_version": 1,
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
      "default_context": null,
      "provenance": "metadata_ok",
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

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `fit_class` | `"perfect" \| "good" \| "marginal" \| "too_tight" \| "unknown"` | -- | the fit verdict |
| `mode` | `"gpu" \| "cpu_gpu" \| "cpu" \| None` | `None` only when `fit_class == "unknown"` | where the package would run |
| `need_gib` | `float` | `>= 0` | estimated memory needed |
| `weights_gib` | `float` | `>= 0` | the package's weight files, summed |
| `kv_gib` | `float` | `>= 0` | estimated KV-cache size at `context` |
| `pool_gib` | `float` | -- | the memory pool judged against (can be negative if reserves exceed capacity) |
| `reserve_gib` | `float` | `>= 0` | the reserve actually applied (VRAM or RAM, matching `mode`) |
| `context` | `int` | `>= 0` | the context length used for `kv_gib` |
| `context_assumed` | `bool` | -- | `true` when `context` is the 8192 fallback, not the package's own `default_context` |
| `reason` | `str \| None` | non-`None` typically iff `fit_class == "unknown"` | why the fit could not be computed, when it could not |

```json
{
  "fit_class": "good",
  "mode": "gpu",
  "need_gib": 7.125,
  "weights_gib": 5.0,
  "kv_gib": 1.125,
  "pool_gib": 10.94,
  "reserve_gib": 1.0,
  "context": 8192,
  "context_assumed": true,
  "reason": null
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
| `ollama_base` | `str \| None` | both set or both `None` with `ollama_tag` | the Ollama library model name, when one exists |
| `ollama_tag` | `str \| None` | both set or both `None` with `ollama_base` | the Ollama library tag naming this base model's size |

```json
{
  "hf_repo": "acme/Nova-7B",
  "repo_aliases": ["Nova-7B-Instruct-GGUF"],
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
| `reserve_ram_gib` | `float` | `>= 0` | system RAM to leave unused when judging fit |
| `reserve_vram_gib` | `float` | `>= 0` | GPU VRAM to leave unused when judging fit |
| `writer` | `bool` | -- | whether this machine runs `fetch` and writes the shared state |

```json
{
  "reserve_ram_gib": 8.0,
  "reserve_vram_gib": 1.0,
  "writer": true
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
(e.g. a shared docs tree outside the state directory).

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
returns `frozenset(packagers) | frozenset(publishers)` -- the positive list a package's repo
owner must belong to.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `schema_version` | `int` | must equal `CONFIG_SCHEMA_VERSION` (currently `1`) | the configuration's schema version |
| `families` | `list[FamilyConfig]` | at least one, names unique, `hf_repo` unique across all | the families this configuration knows |
| `packagers` | `list[str]` | non-empty, no `/`, unique | Hugging Face owners that publish packager (GGUF) repos |
| `publishers` | `list[str]` | non-empty, no `/`, unique | Hugging Face owners that publish base models |
| `machines` | `dict[str, MachineConfig]` | keys `^[a-z0-9][a-z0-9-]*$`; may be empty | this deployment's machines, keyed by name |
| `paths` | `PathsConfig` | -- | where state and rendered output live |
| `llmfit` | `LlmfitConfig` | default `LlmfitConfig()` | the minimum required `llmfit` version |

```json
{
  "schema_version": 1,
  "families": [
    {
      "name": "nova",
      "base_models": [
        {
          "hf_repo": "acme/Nova-7B",
          "repo_aliases": ["Nova-7B-Instruct-GGUF"],
          "ollama_base": "nova",
          "ollama_tag": "7b"
        }
      ]
    }
  ],
  "packagers": ["packager"],
  "publishers": ["acme"],
  "machines": {
    "workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": true}
  },
  "paths": {
    "state": "//models/modelroom/state",
    "markdown": "//models/modelroom/docs/models.md"
  },
  "llmfit": {"min_version": "1.1.16"}
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

A plain JSON object, `{"pid": <int>, "command": <str>, "started_at": <ISO 8601 UTC>, "token":
<str>}`, written by `modelroom/state.py::acquire_lock` before any package command (today, only
`fetch`) does anything else, and removed by `release_lock` in a `finally` block regardless of
how the command ends. `acquire_lock` creates the file with `os.open(path, O_CREAT | O_EXCL |
O_WRONLY)` -- one atomic syscall, never a `path.exists()` check followed by a separate write,
which would leave a window two concurrent processes could both pass through (fix-round 1,
F1). A lock younger than two hours blocks the new run immediately with `LockHeldError` -- this
project never waits for a lock, it fails fast with a message naming the holder's pid, command
and start time. A lock at or beyond two hours, or one whose content cannot be parsed, is
treated as abandoned by a crashed process and taken over: the taking-over process writes its
own content (including a fresh random `token`) and re-reads the file after `os.replace`, only
proceeding when that content is still its own -- otherwise another process won the same
takeover race and this one raises `LockHeldError` instead. `acquire_lock` returns the `token`
it wrote; `release_lock(path, token)` removes the file only when its current content still
carries that exact `token`, so a caller can never delete a lock it no longer holds.

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

### Run status (`run-status.json`)

Written by `modelroom/state.py::write_run_status` at the end of every `fetch` run, including
one that ends exit `1` -- it is the one file a caller can always read to see what happened,
whether or not the snapshot changed.

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

### Area semantics

An "area" (the `Area` model, already defined above) is one Hugging Face packager-owner's
candidate repos for one base model, or one base model's whole Ollama library entry:

- **Hugging Face**: one area per `(base_model.hf_repo, owner)`, where `owner` ranges over
  `modelroom/hf.py::candidate_owners` -- every configured `packagers` entry plus the base
  model's own owner (a publisher counts as a packager of its own models), filtered to
  `config.allowed_owners()`. Every candidate repo name under that owner (`<base_name>-GGUF`
  plus every `repo_aliases` entry, exact names, de-duplicated) is probed within the same area;
  a candidate that does not exist does not end the area, only a genuine failure does (see
  below). `Area.packager` is that owner.
- **Ollama**: one area per base model that configures both `ollama_base` and `ollama_tag`;
  `Area.packager` is always `None`. A base model with neither configured is a trivially
  `complete`, empty area (there is nothing to look up).

An area ends `incomplete` only for a genuine failure: any HTTP status other than `200`
(Hugging Face model-info additionally treats `401`/`404` as "not found", see below, not a
failure), a network error, a body that fails to parse, a response missing a field the fetcher
needs (Hugging Face: `sha`; Ollama: zero tags parsed from the tags page), a tree entry or
manifest layer in a shape the registry never actually sends -- not a dict, `path`/`mediaType`/
`digest` not a string, or a weight (`weights`/`weights_shard`, or an Ollama `.image.model`/
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
the request; the fetcher in progress catches it exactly like any other failure and ends its
*current* area `incomplete` with that message. Areas already completed before the budget ran
out keep their results.

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

A redirect chain `modelroom/http.py::UrllibTransport` follows (F10, see "Redirects" in the
`http.py` module docs) costs one request per hop; `BudgetedTransport` charges every hop, not
just the one call made into it, so a chain can overshoot the budget by at most its own length --
documented, not eliminated, since the hop count is only known after the chain is followed.

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
  `run_at`.
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

### Ollama packages: files and `default_context`

An Ollama manifest's weight layer(s) become one `PackageFile` per `application/vnd.ollama.
image.model` layer (role `weights`, the registry's own digest, `format="gguf"`). A
tensor-only manifest (only `application/vnd.ollama.image.tensor` layers, no `.image.model`
layer -- an MLX-style build) can carry hundreds of per-tensor layers; rather than modelling
one `PackageFile` per tensor, `modelroom/ollama.py` aggregates them into a single synthetic
file (summed size, no digest, `format="tensor"`), since a tensor package is always
`("unresolved", "format")` regardless of its individual tensor layout, which is not part of
this catalog's contract. A weight layer with no non-negative integer `size` is a shape error
that ends the area `incomplete`, not a silent `0` (fix-round 1, F4 -- see "Area semantics"
above). `Package.default_context` (Ollama's `num_ctx`) is always `None` in this work package --
reading it means downloading the manifest's `params` layer by digest, one more request per
tag, left for a later work package.

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

`modelroom hardware --config <toml> --machine <name>` measures the machine it runs on and
writes `<state>/hardware/<machine>.json` (`HardwareSnapshot`, models documented above). Unlike
`fetch`, the named machine only has to be *configured* (present in `Configuration.machines`),
not a `writer` -- hardware is measured on every machine, including inference-only ones. Exit
codes reuse the table above unchanged: `2` for an unconfigured machine or a missing/too-old
`llmfit`, `3` for an existing hardware file with an unsupported `schema_version`, `0`
otherwise -- including when the local Ollama daemon could not be reached (`installed` is then
`None` with a reason, never a command failure).

### Local llmfit binding

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
always label this result "fit (computed, v1)", never "runs"** -- it is an estimate from a
memory-sizing formula, not a measurement of the package actually loading and generating
tokens (that is what `Measurement`/`bench`, a later work package, are for).

**Coverage.** Only `base_model.architecture.kind == "dense_classic"` is judged; anything else
(including a hybrid or MoE architecture resolved to `"unknown"`, e.g. `Qwen/Qwen3.5-9B`) comes
back `fit_class="unknown"`, `reason="architecture not covered by v1"`. Only a `complete`
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
- `kv_gib = 2 * L * KVH * D * 2 bytes * context / 1024**3`, where `L =
  architecture.num_hidden_layers`, `KVH = architecture.num_key_value_heads`, `D =
  architecture.head_dim` -- the standard KV-cache size for a GQA/MQA transformer at 16-bit
  precision (`2 bytes` per element, the leading `2` for key and value each).
- `need_gib = weights_gib * 1.10 + kv_gib + 0.50` -- a 10 % overhead on the weights themselves
  (allocator/runtime overhead) plus a flat 0.50 GiB fixed cost (activations and other small
  buffers), on top of the KV cache.

**Pool and mode.** `available_vram = hardware.vram_gib - machine_config.reserve_vram_gib`.
When `need_gib <= available_vram`, the package fits the GPU: `mode="gpu"`, `pool_gib =
available_vram`, `reserve_gib = reserve_vram_gib`. Otherwise it falls back to the RAM pool:
`pool_gib = hardware.ram_gib - machine_config.reserve_ram_gib`, `reserve_gib =
reserve_ram_gib`, `mode="cpu_gpu"` when `hardware.vram_gib > 0` (some GPU exists, just not
enough) else `mode="cpu"`.

**Classification.** `ratio = need_gib / pool_gib` (a `pool_gib <= 0` is always `too_tight`,
never a division): `ratio <= 0.60` is `perfect`, `<= 0.85` is `good`, `<= 0.98` is `marginal`,
anything above is `too_tight`. **Off the GPU (`mode` is `cpu_gpu` or `cpu`), the class is
capped at `good`** -- a ratio that would otherwise read `perfect` still comes back `good`,
because "perfect" is reserved for a package that comfortably fits in VRAM.

**Worked examples** (also `tests/test_fit.py`, values restated in each test's own comments so
they can be recomputed by hand): a dense 8B Q4_K_M package (~5.0 GiB weights, `L=36, KVH=8,
D=128`, context assumed 8192) has `kv_gib = 1.125` GiB and `need_gib = 7.125` GiB. On the
reference laptop (`vram_gib=11.94`, `reserve_vram_gib=1.0`) that is `mode="gpu"`, ratio
`0.6513` -> `good`. On a small server (`vram_gib=0`, `ram_gib=7.56`, `reserve_ram_gib=3.0`) it
falls to `mode="cpu"`, `pool_gib=4.56`, ratio `1.5625` -> `too_tight`. A dense 9B Q4_K_M
package (~5.6 GiB, `L=42, KVH=8, D=128`) has `need_gib = 7.9725` GiB: `good` on the laptop
(ratio `0.7288`), `too_tight` on the server (ratio `1.748`).

**`Measurement.profile_measured_at` cross-check.** `Measurement` itself does not check its
`profile_measured_at` against any particular `HardwareSnapshot.measured_at` (see the
`Measurement` model above) -- a renderer reading `hardware.measurements` alongside
`hardware` refuses to show a measurement whose `profile_measured_at != hardware.measured_at`
as if it were current for that profile, since the hardware may have changed since that
measurement was taken. This check belongs to the renderer (a later work package), not to
`compute_fit` or to the `Measurement`/`HardwareSnapshot` models themselves.
