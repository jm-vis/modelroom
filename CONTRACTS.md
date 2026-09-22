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

Every example below is `EXAMPLES["<ModelName>"]` from `modelroom/contracts.py`, verbatim; a
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
| `parameters_b` | `float` | `> 0` | parameter count, in billions |
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

A plain JSON object, `{"pid": <int>, "command": <str>, "started_at": <ISO 8601 UTC>}`, written
by `modelroom/state.py::acquire_lock` before any package command (today, only `fetch`) does
anything else, and removed by `release_lock` in a `finally` block regardless of how the
command ends. A lock younger than two hours blocks the new run immediately with
`LockHeldError` -- this project never waits for a lock, it fails fast with a message naming
the holder's pid, command and start time. A lock at or beyond two hours, or one whose content
cannot be parsed, is treated as abandoned by a crashed process and silently overwritten.

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
needs (Hugging Face: `sha`; Ollama: zero tags parsed from the tags page), or the run's request
budget running out mid-area (`error` is exactly `"budget exhausted"`). A candidate repo that
simply does not exist, or a base model with no Ollama configuration, is not a failure.

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
out keep their results; areas not yet started are simply never attempted, and their old state
(if any) is left untouched by the merge rule below.

### Merge rule (building this run's `Snapshot`)

`modelroom/state.py::merge_snapshot(old, run_at, area_outcomes, base_models)` builds the
`Snapshot` a run writes:

- **Base models** are always replaced wholesale by this run's results -- an unknown
  architecture, or a base model this run could not reach at all, is still "this run's result",
  never silently carried over from `old`.
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
- A base model or area this run never touches at all (not configured any more) simply does
  not appear in the result -- `merge_snapshot` only ever iterates `area_outcomes` and the
  `base_models` list it is given, it does not scan `old` for leftovers to carry forward on its
  own.

### Version rule

`modelroom/state.py::check_run_is_newer(old, run_at)` refuses a run whose `run_at` is not
*strictly* newer than `old.run_at` (`run_at <= old.run_at`, so the same second counts as
stale, not just an earlier one), raising `StaleRunError` before anything is fetched or
written. `run_at` itself is the run's start time, UTC, truncated to whole seconds
(`datetime.now(timezone.utc).replace(microsecond=0)` in `modelroom/cli.py`, or an injected
value in tests) -- second precision is deliberate: two `fetch` runs started in the same wall
second are indistinguishable and the second one must not silently win.

### `parameters_b` when Hugging Face has no answer this run

`BaseModelSpec.parameters_b` must be `> 0`; a base model whose publisher repo could not be
reached, or whose model-info response carries no `safetensors.total`, therefore cannot simply
report `0.0`. `modelroom/fetch.py::run_fetch` falls back, in order: the previous snapshot's
`parameters_b` for the same `hf_repo`, when one exists; otherwise the literal placeholder
`1.0` (billion) -- an intentionally obvious, wrong-looking number rather than a fabricated
"real" one, easy to spot in a rendered snapshot as "not actually measured yet". Reading the
real count from a packager's GGUF metadata (`gguf.total` in a Hugging Face model-info
response) instead of only the publisher's `safetensors.total` is left for a later work
package.

### Ollama packages: files and `default_context`

An Ollama manifest's weight layer(s) become one `PackageFile` per `application/vnd.ollama.
image.model` layer (role `weights`, the registry's own digest, `format="gguf"`). A
tensor-only manifest (only `application/vnd.ollama.image.tensor` layers, no `.image.model`
layer -- an MLX-style build) can carry hundreds of per-tensor layers; rather than modelling
one `PackageFile` per tensor, `modelroom/ollama.py` aggregates them into a single synthetic
file (summed size, no digest, `format="tensor"`), since a tensor package is always
`("unresolved", "format")` regardless of its individual tensor layout, which is not part of
this catalog's contract. `Package.default_context` (Ollama's `num_ctx`) is always `None` in
this work package -- reading it means downloading the manifest's `params` layer by digest,
one more request per tag, left for a later work package.

### Ollama size-token tolerance (AP3 acceptance fix)

`modelroom/provenance.py::_decide_ollama` reads the Ollama tag's size token (the first
`-`-separated token, e.g. `9b` in `9b-q4_K_M`) and compares it against the base model's
measured `parameters_b`. A live snapshot (13 base models, 79 areas, 358 packages, 2026-09-22)
showed the original exact-equality rule (`first_token == f"{parameters_b:g}b"`) can never match
real data: `safetensors.total` counts embeddings, so `Qwen/Qwen3.5-9B` measures
`parameters_b = 9.653104368`, while its library tag is plainly `9b`.

The rule is now a tolerance, not an equality: the first token must fully match a size-token
pattern (`<number>b` for billions or `<number>m` for millions, case-insensitive -- `7banana`
never matches at all, it is not a size token), and the declared size must be within 15 % of
`parameters_b` (`abs(parameters_b - declared) <= 0.15 * declared`). Examples: `9b` for a
measured `9.653104368` passes (0.65 vs. a 1.35 allowance); `30b` for a measured `30.5` passes;
`1.5b` for a measured `1.54` passes; `70b` for a measured `7.0` still fails (63 vs. a 10.5
allowance) -- a genuinely wrong tag is still rejected, only the packager's rounding is
tolerated. The full tag still has to equal `base_model.ollama_tag` or start with
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
