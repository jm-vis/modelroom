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
Face package, its third element is the sorted, de-duplicated set of the normalized stems of
every `weights`/`weights_shard` file, joined with `|` (so a sharded package's shards, which all
normalize to the same stem, still collapse to one identity). For an Ollama package it is the
tag part of `ollama_name` after the colon.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | at least one fetch area ended incomplete; complete areas were published |
| `2` | required external tool (`llmfit`) missing or below the minimum version |
| `3` | an input file has an unsupported `schema_version` |

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
`num_local_experts`, `num_experts`, `n_routed_experts` or `num_experts_per_tok` present with a
value `> 1`, or `moe_intermediate_size` present with any truthy value, rules out
`dense_classic` even when the three numeric fields all look plausible.

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
