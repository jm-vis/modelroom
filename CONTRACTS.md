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

## Language standard

Every field value, message and paragraph in this file is US English and uses the word list in
`AGENTS.md`, section "Language standard": `latest`, `legacy` (with its `successor`) and
`unknown` for where a model stands in its family; `measured`, `entered` and `computed` for
where a value came from; `publisher`, `listed packager` and `other` for who owns a repository,
which is a class and never a verdict; `unknown` for a fact with no evidence behind it;
`not comparable` for measurements taken under different scenarios. Schema names, field names
and literal values are contract: they are never changed for a language reason, so a value like
`unified_memory` keeps its spelling even where the prose around it is rewritten.
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
| `num_hidden_layers` | `int \| None` | `0 < n <= 2**31 - 1`; required when `kind == "dense_classic"` | transformer block count |
| `num_key_value_heads` | `int \| None` | `0 < n <= 2**31 - 1`; required when `kind == "dense_classic"` | KV heads (for GQA/MQA KV-cache sizing) |
| `head_dim` | `int \| None` | `0 < n <= 2**31 - 1`; required when `kind == "dense_classic"` | per-head dimension |
| `layer_types` | `list[str] \| None` | `None` or all `"full_attention"` when `kind == "dense_classic"` | per-layer attention kind, when the config states one |
| `max_context` | `int \| None` | `0 < n <= 2**31 - 1` | the model's trained maximum context length |

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
| `aliases` | `list[str]` | each a tag (`^[A-Za-z0-9][A-Za-z0-9._-]*$`), each once, never this package's own tag; empty for `source == "huggingface"` | the other tags of the very same manifest |
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
| `need_gib` | `float` | `>= 0` | the memory needed, computed |
| `weights_gib` | `float` | `>= 0` | the package's weight files, summed |
| `kv_gib` | `float` | `>= 0` | the KV-cache size at `context`, computed |
| `pool_gib` | `float` | -- | the memory pool judged against (can be negative if reserves exceed capacity) |
| `reserve_gib` | `float` | `>= 0` | the reserve actually applied (VRAM or RAM, matching `mode`) |
| `context` | `int` | `>= 0` | the context length used for `kv_gib` |
| `context_assumed` | `bool` | -- | `true` when `context` is the 8192 fallback, not the package's own `default_context` |
| `basis` | `"architecture" \| "size"` | default `"architecture"` | what the KV cache was computed from: the base model's architecture, or the size alone ("Fit from size") |
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
  "basis": "architecture",
  "reason": null
}
```

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
| `reserve_ram_gib` | `float` | `>= 0` | system RAM to leave unused when judging fit |
| `reserve_vram_gib` | `float` | `>= 0` | GPU VRAM to leave unused when judging fit |
| `writer` | `bool` | -- | whether this machine runs `fetch` and writes the shared state |
| `profile` | `str \| None` | schema 2; 16 lowercase hex characters; default `None` | the `profile_id` of this machine's hardware profile (`<state>/hardware/<profile_id>.json`), set by `modelroom migrate` or a guided run |

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
imported profile); such a machine is never a writer. The defaults are the values the shipped
example uses for a workstation.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `reserve_ram_gib` | `float` | `>= 0`, default `8.0` | system RAM to leave unused |
| `reserve_vram_gib` | `float` | `>= 0`, default `1.0` | GPU VRAM to leave unused |

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
The field always holds the **number of tokens**, never a level name. Both fields are optional, so the
configuration's `schema_version` stays `2` and no migration step reads or writes them.

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `results` | `Path \| None` | default `None` | the folder the guided mode keeps its configuration and results in |
| `context` | `int \| None` | `> 0`, `<= 2**31 - 1`; default `None` | the context the last guided run's ranking was computed for |

```json
{
  "results": "//models/modelroom",
  "context": 4096
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
| `schema_version` | `int` | the model accepts only `CONFIG_SCHEMA_VERSION` (`2`); readers also accept `1`, see below | the configuration's schema version |
| `families` | `list[FamilyConfig]` | may be empty (schema 2), names unique, `hf_repo` unique across all | the families this configuration knows |
| `packagers` | `list[str]` | non-empty, no `/`, unique | Hugging Face owners that publish packager (GGUF) repos |
| `publishers` | `list[str]` | non-empty, no `/`, unique | Hugging Face owners that publish base models |
| `machines` | `dict[str, MachineConfig]` | keys `^[a-z0-9][a-z0-9-]*$`; may be empty | this deployment's machines, keyed by name |
| `paths` | `PathsConfig` | -- | where state and rendered output live |
| `llmfit` | `LlmfitConfig` | default `LlmfitConfig()` | the minimum required `llmfit` version |
| `defaults` | `DefaultsConfig` | schema 2, default `DefaultsConfig()` | reserves for an imported machine without its own table |
| `updates` | `UpdatesConfig` | schema 2, default `UpdatesConfig()` | the release lookup switch |
| `guided` | `GuidedConfig` | schema 2, default `GuidedConfig()` | the guided mode's results folder and the context it chose |

**Reading schema 1.** `CONFIG_SCHEMA_RANGE` is `(1, 3)`: `load_config` and
`Configuration.from_dict` accept schema 1 and 2 and refuse 3 and above with `SchemaVersionError`
before any field is looked at. A schema-1 dict is turned into the equivalent schema-2 dict in
memory by `normalize_config_v1` before validation: every field keeps its value, `schema_version`
becomes `2`, `[defaults]` and `[updates]` are added with their defaults, `[guided]` stays absent,
no machine gets a `profile`. Schema 1's own rule "at least one family" is checked there (a
`ConfigError`), because schema 2 drops it. No writer rule is added. The file on disk is changed
only by `modelroom migrate` (see "Migration").

```json
{
  "schema_version": 2,
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
  "guided": {"results": "//models/modelroom", "context": 4096}
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
  "schema_version": 2,
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
  "unresolved": [{"reason": "the repository does not say it packages a base model", "count": 1}]
}
```

`schema_version` is **2** since 2026-09-25, for the one new field: `mode` is `word`, `id` or
`catalog` (`SearchOutcome.mode`), so a reader can tell a word search from a typed repository id and
from the catalog pages of a search with no word at all. `word` is what was typed, the empty string
included.
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
counting as hop 1; more hops raise `RuntimeError` before the hop that would exceed the limit is
ever made). `run_fetch` composes `RedirectingTransport(BudgetedTransport(transport, budget))` --
`BudgetedTransport` as the *inner* layer -- so every hop of a chain arrives at the budget check
as its own separate call, checked and booked (or raising `BudgetExhaustedError`) before it is
made, failure paths included, never only after the fact.

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

**Coverage.** Only `base_model.architecture.kind == "dense_classic"` is judged by this formula;
anything else (including a hybrid or MoE architecture resolved to `"unknown"`, e.g.
`Qwen/Qwen3.5-9B`) comes back `fit_class="unknown"`,
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

### Fit from size (basis `size`)

Decided 2026-09-24, after a hand test in which 25 of 26 packages stood under "architecture not
covered by v1" and one package was ranked: an architecture nobody can read is a reason to compute
more coarsely, not a reason to say nothing. A `Fit` says which of the two it is in `basis`, and
the provenance of both stays `computed` -- the word for a number this package worked out rather
than measured. `size` is never shown as a measurement and never as a promise.

**Two entry points**, both in `modelroom/fit.py`, both fit v1's formula and classes with one term
replaced:

- `fit_from_size(package, vram_gib, ram_gib, machine_config, context)` -- after the fetch:
  `weights_gib` is the package's own weight files as always, and
  `kv_gib = context x KV_PER_TOKEN_SIZE_BASIS / 1024**3` with `KV_PER_TOKEN_SIZE_BASIS = 147_456`
  bytes (`2 x 36 layers x 8 KV heads x 128 wide x 2 bytes`, the numbers of a dense 9B model at
  16-bit precision). `compute_fit_v2` calls it for every package whose architecture is not
  `dense_classic`.
- `fit_from_parameters(parameters_b, vram_gib, ram_gib, machine_config, context)` -- before the
  fetch, for the selection list of the guided mode's step 2:
  `weights_gib = parameters_b x 1e9 x BYTES_PER_PARAMETER_Q4 / 1024**3` with
  `BYTES_PER_PARAMETER_Q4 = 0.6`, the size of a `Q4_K_M` build (about 4.85 bits per weight plus
  the tables). Measured against two real packages of the fixtures: 5,680,522,464 bytes for
  9,653,104,368 parameters (0.588) and 5,027,785,216 bytes for 8,190,735,360 parameters (0.614).

`need_gib`, the pool, the mode and the thresholds are fit v1's, unchanged. **The class is capped
at `good`**: `perfect` stays reserved for a package whose architecture was read.

**The KV constant is one fixed assumption, not a bound in either direction** (corrected in the
second-model round of 2026-09-24, which measured the case). A hybrid or mixture architecture keeps
fewer layers of full attention and needs less than it; a dense model with more layers needs more.
Measured on the 5.29 GiB `Q4_K_M` package of the fixtures at 131072 tokens: 24.32 GiB on this
basis, against 22.32 GiB for 32 layers, 27.32 for 42 and 30.32 for 48. At 8192 tokens the same
comparison is 7.44 against 7.32 / 7.63 / 7.82 GiB. So a package this basis calls `marginal` can be
`too_tight` in truth, and one it calls `too_tight` can fit: the cap at `good` limits the claim, it
does not make the number exact. Every view therefore says `from size` next to the class, and an
exact computation for these architectures is a work package of its own.

**What stays `unknown`**, size or no size, because there is nothing to compute with: a package
that is not a complete `gguf` build, a weight file with no size, the GPU states the schema-2 gate
refuses, a llmfit deviation, and more than one request. For the list of step 2 there are two more
of its own: `machine not measured` and `parameter count unknown`.

**Where the basis is visible.** In the JSON view as `fit.basis`, in the terminal view as the
`(from size):` line with the ranks it is about (since 2026-09-24; the `Fit` cell says the memory pool
instead), and in the note of the row -- the note of a ranked package keeps
the sentence of its memory pool in front (`Fits into graphics memory: …`, `Too large for graphics
memory alone, …`; without it a row in system memory read as the equal of one in graphics memory,
acceptance of 2026-09-24) and ends with `From the size of the package, not its architecture.`,
with `fit.basis` added to its facts; a `too_tight` one ends with the same sentence. A row whose
speed was measured carries the measurement's note instead, so in
the Markdown view the basis of such a row is in the JSON view only; the Markdown table's own `Fit
(computed)` cell is unchanged.

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

**Eligibility.** Only packages with `provenance in ("metadata_ok", "approved")`, `complete ==
True` and `active == True` are ever shown.

**Grouping and ordering.** Eligible packages group by (base model, packager), packager being the
Hugging Face owner or the literal `"ollama"`. Groups are ordered by base model in configuration
order (families, then base models within a family), then packager alphabetically with `"ollama"`
always last.

**Selection rule**, per group *and per configured machine*: the package with the largest
quantization (`QUANT_ORDER` index, unknown sorting last -- same convention as
`quantization.sort_key`) whose `fit.py::compute_fit` class on that machine is `"good"` or
`"perfect"`; when none qualifies, the smallest *judged* package (`quantization.sort_key`, a real
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

One machine's hardware, `<state>/hardware/<profile_id>.json`, schema 2. `profile_id` is 16
random lowercase hex characters (`new_profile_id`), never derived from the machine: it stays
the same when the machine is renamed, and two machines can never collide on it. `display_name`
is what the user reads; the machine name in the configuration points to the profile through
`[machines.<name>].profile`.

`os_fingerprint` is the first 16 hex characters of SHA-256 over the raw OS identifier (Windows
`MachineGuid`, Linux `/etc/machine-id`, macOS platform UUID) followed by the salt `modelroom`
(`os_fingerprint`); the raw identifier never leaves the machine. It tells a clone from the same
machine (see "Profile binding"). `os_fingerprint_source` names where it came from; `none` (not
readable, or an `entered` profile) and `legacy` (migrated) store the fingerprint `none`.

Every memory value says where it came from. `ram_physical_gib` is physical RAM (`os` or
`llmfit`); `ram_limit_gib` with `ram_limit_scope` is a process or container limit (cgroup, job
object), a note only and never a fit input; `vram_gib` is `nvidia-smi`, `llmfit`, `none` (no
GPU: `0`) or `unknown` (`None`). A graphics adapter's own reported memory is never taken as VRAM.

`gpu_state` decides whether the fit computes at all:

| `gpu_state` | Meaning | Fit |
|---|---|---|
| `none` | no GPU; `vram_source` `none`, `vram_gib` `0` | computes, CPU mode |
| `measured` | one GPU, VRAM measured (`> 0`) | computes |
| `present_unmeasured` | an adapter is present, its memory was not measured | `unknown` |
| `multi_gpu_not_covered` | more than one GPU | `unknown` |
| `unified_memory` | CPU and GPU share memory | `unknown` |
| `unsupported_platform` | this platform is not measured yet | `unknown` |
| `legacy_unknown` | migrated from schema 1, GPU layout not recorded | `unknown` until measured again |

**Cross-check with llmfit.** Only like with like: physical RAM against llmfit `total_ram_gb`,
VRAM against llmfit `gpu_vram_gb`. `crosscheck(own, llmfit)` is `confirmed` when
`|own - llmfit| / max(own, llmfit) <= 0.05` (both zero is `confirmed`), `deviation` otherwise,
`absent` when llmfit gave no reading, `error` when llmfit failed. The own reading is the
authority; a `deviation` blocks the fit until the user measures again or enters the value.
`fit_block_reason(profile)` returns the reason the fit must not compute (GPU state, unknown RAM,
a deviation), or `None`.

**Reading schema 1.** `read_profile_document` validates schema 1 as `HardwareSnapshot` and
schema 2 as `HardwareProfile`; 3 and above raise `SchemaVersionError` (exit 3) before any
field is looked at. `normalize_profile_v1(legacy, profile_id)` converts without guessing:
`display_name` is the old machine key, fingerprint source `legacy`, RAM and VRAM keep their
llmfit values with source `llmfit`, `gpu_state` is `unified_memory` when the old file said so and
`legacy_unknown` otherwise (a positive VRAM did not prove a single GPU, a zero did not prove
none), both cross-checks `absent`. The schema-1 `installed` list is not carried over: it is a
point-in-time observation the next measurement records again.

### Hardware measurement

`modelroom hardware --config <toml> [--machine <name>] [--cpu-only] [--new-identity]` measures
the machine it runs on and writes one schema-2 profile, `<state>/hardware/<profile_id>.json`
(`modelroom/measure.py`, `modelroom/cli.py`). modelroom measures the machine itself; `llmfit` is
a cross-check of two quantities, never the source and never a requirement -- a machine without
`llmfit` is measured and written all the same. `--machine` is optional and only names the
configuration entry whose `profile` the takeover rule may adopt. The schema-1 writer
(`write_hardware_snapshot`, `<machine>.json`) stays readable for `render` until the renderer
switches; `hardware` no longer writes it, and a schema-1 file in the folder is skipped, never
adopted and never overwritten. The command reads no Ollama daemon: the installed packages are a
point-in-time observation the load test records (CONTRACTS.md, "MeasurementRecord"), not part of
a profile.

**Sources.** Every source is a fixed argument list or a fixed file path, run through the
injected command layer (`Runner`, 10 s timeout) or file layer -- no user-controlled string ever
reaches a command line, and `AdapterRAM`/`adapter memory` is never read as VRAM.

| Quantity | Platform | Source | Unit | Stored as |
|---|---|---|---|---|
| physical RAM | Windows | `GlobalMemoryStatusEx().ullTotalPhys` (ctypes, `kernel32`) | bytes | `ram_physical_gib`, source `os` |
| physical RAM | Linux | `/proc/meminfo`, `MemTotal` | kB | `ram_physical_gib`, source `os` |
| physical RAM | macOS | `sysctl -n hw.memsize` | bytes | `ram_physical_gib`, source `os` (display only: macOS has no fit) |
| VRAM | Windows, Linux | `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits` | MiB | `vram_gib`, source `nvidia-smi`, `gpu_name` |
| display adapters | Linux | `lspci -nn`, PCI class `03xx` | vendor id | `gpu_state` (adapter rule below) |
| display adapters | Windows | `Win32_VideoController.PNPDeviceID` (`VEN_xxxx`), read with `powershell -NoProfile -NonInteractive -Command` | vendor id | `gpu_state` (adapter rule below) |
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

**`gpu_state`.** `nvidia-smi` decides first; when it is not installed or cannot answer, the
adapter rule does.

| Observation | `gpu_state` | VRAM |
|---|---|---|
| `nvidia-smi` lists exactly one adapter at index 0 with memory above 0 | `measured` | MiB / 1024, source `nvidia-smi` |
| `nvidia-smi` lists more than one adapter | `multi_gpu_not_covered` | `unknown` |
| `nvidia-smi` lists one adapter that is not index 0, reports no memory, or output that cannot be read | `present_unmeasured` | `unknown` |
| no `nvidia-smi`, and no PCI class-03 device at all | `none` | `0`, source `none` |
| no `nvidia-smi`, and every class-03 device is from the display-only list below | `none` | `0`, source `none`, note `display adapter only` |
| no `nvidia-smi`, and at least one class-03 device is from any other vendor | `present_unmeasured` | `unknown` |
| no `nvidia-smi`, and the adapters could not be listed at all | `present_unmeasured` | `unknown` (a GPU cannot be ruled out) |
| macOS, or any platform that is not Windows or Linux | `unsupported_platform` | `unknown` (no unified-memory statement is invented) |
| `--cpu-only` on Windows or Linux | `none` | `0`, source `none`, profile `origin` `entered` |
| `--cpu-only` on any other platform | `unsupported_platform` | `unknown`: the flag speaks about the GPU, not about the platform, and the RAM there is display only |

**The display-only vendor list.** A positive list, so an unknown vendor is never declared
harmless: QEMU/Bochs `1234`, virtio `1af4`, VMware `15ad`, Red Hat `1b36`, VirtualBox `80ee`,
Microsoft Hyper-V `1414`, Xen `5853`, Amazon `1d0f`. Every other vendor -- NVIDIA `10de`, AMD
`1002`, Intel `8086`, or none readable at all -- is `present_unmeasured` with a note naming the
vendors. When the *only* measurable vendor is Intel, the note adds "Ollama uses the CPU on Intel
graphics; run `hardware --cpu-only` for the CPU fit"; on a machine that also carries a discrete
card that advice would be wrong, so it is not given there.

**The cross-check and `--cpu-only`.** An entered `0` VRAM is not a reading, so it is not
compared with llmfit's: the VRAM check of a `--cpu-only` profile is `absent`, whatever llmfit
did -- `absent` means "never asked", while `error` would claim the comparison was attempted.
Comparing them on a machine that does have a card was a `deviation`, and a deviation blocks the
fit -- exactly the CPU fit `--cpu-only` exists to produce. The physical RAM is still measured
there, and still cross-checked.

**Notes.** Facts only, printed after the summary line: the reason a reading is unknown, the GPU
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
`display_name` is this machine's host name for a new profile and is kept as it is when the
profile already exists (the user may rename it). `--cpu-only` writes an `entered` profile, which
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
carry no llmfit value. `own_gib` must equal the profile's own value for that quantity.

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
| `schema_version` | `int` | `2` |
| `profile_id` | `str` | 16 lowercase hex |
| `display_name` | `str` | 1 to 128 characters |
| `os_fingerprint` | `str` | 16 lowercase hex, or `none` exactly when the source is `none` or `legacy` |
| `os_fingerprint_source` | `windows_machineguid \| linux_machine_id \| macos_platform_uuid \| none \| legacy` | `entered` profiles: `none` |
| `origin` | `measured \| entered` | -- |
| `recorded_at` | UTC timestamp | timezone-aware |
| `ram_physical_gib` | `float \| None` | `> 0`; `None` exactly when the source is `unknown` |
| `ram_physical_source` | `os \| llmfit \| unknown` | -- |
| `ram_limit_gib` | `float \| None` | `> 0`; `None` exactly when `ram_limit_scope` is `none` |
| `ram_limit_scope` | `str` | e.g. `none`, `cgroup`, `job_object` |
| `vram_gib` | `float \| None` | `>= 0`; `None` exactly when the source is `unknown`; `0` for source `none` |
| `vram_source` | `nvidia-smi \| llmfit \| none \| unknown` | `gpu_state` `none` requires `none` |
| `gpu_state` | see the table above | `measured` requires a measured VRAM above 0; `legacy_unknown` only with source `legacy` |
| `gpu_name` | `str \| None` | display only |
| `llmfit_crosscheck` | `LlmfitCrosscheck` | -- |
| `llmfit_version` | `str \| None` | -- |

```json
{
  "schema_version": 2,
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

### Scenario

What one ranking assumes, for every package in it: `context_requested` (the context the user
chose; `context_origin` `default` for 8192, `entered`, or `legacy` from a migrated measurement),
the KV cache type and the number of concurrent requests. The daemon's KV cache type is not
readable over its API, so `kv_type` is `f16` (Ollama's default) and `kv_type_assumed` is always
`true`; `context_origin` `default` always means 8192.
A measurement always runs one request; `requests` above 1 appears only in the reverse
calculation (`Requirement`). `default_scenario()` is 8192, `f16` assumed, one request.

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

Shipped as `modelroom/protocol_v1.toml` and read by `load_protocol()`/`shipped_protocol()`:
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

### MeasurementRecord

One speed measurement of one package on one profile, schema 2, never changed once written.

- `measurement_id` is `<measured_at as YYYYMMDDTHHMMSSZ>-<8 hex>` and must agree with
  `measured_at`; `new_measurement_id` draws the 8 hex characters at random.
- `protocol` is `v1` for a load test after the protocol above, `none` for a measurement migrated
  from a schema-1 profile. `none` requires `validity` `unchecked`, no runs and
  `comparable = false`. `v1` is `valid` or `invalid`; `valid` is checked against the protocol
  rule and requires the speeds.
- `tps_mean`, `tps_min`, `tps_max` are all set or all `None`, `tps_mean > 0`,
  `tps_min <= tps_max`. With protocol `v1`, `0 < tps_min <= tps_mean <= tps_max` and they are the
  mean, minimum and maximum over `runs`, so a `v1` record without runs carries none. A migrated schema-1 range is carried over unchanged
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
profile. `load_export` checks `schema_version` first (`SchemaVersionError`, exit 3).

```json
{
  "schema_version": 1,
  "profile": {
    "schema_version": 2,
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
existing profile's, else this machine's host name -- binds it (the binding does not change, it
is the same id) and the guided mode writes `[machines.<name>].profile` as it does after `new`.
The measurements under `measurements/<profile_id>/` stay what they are: a measurement belongs to
an id, and whether it still counts for a ranking is decided, as ever, by the comparison of
protocol and scenario. Where the rule does not ask, `same_machine` changes nothing. The two
switches are two answers to one question: together they are exit `2` with a sentence naming
both, and `resolve_profile_target` raises `ValueError` for them.

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

### Migration to schema 2

`modelroom migrate --config <file>` turns a schema-1 deployment into schema 2, in one run under
`modelroom.lock`:

- every schema-1 profile `<state>/hardware/<name>.json` becomes `<profile_id>.json` with a new
  random `profile_id` (`normalize_profile_v1`); its embedded measurements become measurement
  files with `protocol: none` (`legacy_measurement_record`, whose 8 hex characters come from the
  SHA-256 of the old measurement, so the same input always gives the same id); the old file is
  moved to `<name>.json.v1.bak`;
- a schema-1 `modelroom.toml` becomes schema 2 (`normalize_config_v1`) with
  `[machines.<name>].profile` set for every migrated profile of that name; the rewritten text is
  validated like `load_config` would before it is written, the original bytes are kept as
  `modelroom.toml.v1.bak`. The rewritten file carries no comments; the backup keeps them.

The whole run is planned and checked before the lock is taken, and again under the lock, before
anything is written: every file is read and version-checked, every profile and measurement is
converted, and every file the run would write -- the new profile, each measurement file, each
backup -- must be absent or identical, and no existing part of its folder path may be a file. The new `profile_id`s are drawn once, before the first
check, so both checks look at the paths the run writes. Under the lock the configuration is read
again; if its `[paths]` changed meanwhile, the run stops (exit 3) without writing anything but
the lock file. A file of schema 3 or later is exit 3; a conversion that
fails, a target or backup that exists with other content or cannot be read (a folder at a backup
path, for instance), or two schema-1 files for the same
machine is exit 3 as well; in every case nothing is written and no lock file is created. A run
that finds everything at schema 2 prints `nothing to do` and exits 0. A run that stopped halfway
converges on the next run: a schema-2 profile with fingerprint source `legacy` and the same
`display_name` keeps its `profile_id`, and files that are already there (identical, as checked)
are left as they are. A schema-2 configuration is not rewritten: when it sits next to schema-1
profiles, the profiles are migrated and `[machines.<name>].profile` is left to the guided mode.

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
it. No lock is taken: the export writes nothing inside the state folder, and a measurement file
is published whole and never changed afterwards, so the only effect of a concurrent write is
that a measurement published during the export may not be in it. An unreadable measurement file
is reported and left out; it does not stop the export. `--profile` has to be a `profile_id` (16
lowercase hex characters), and `--out` must be none of the files this package writes itself:
nothing inside `paths.state` (the same rule `PathsConfig` holds for `paths.markdown`), not the
configuration file, not the pointer file -- otherwise an export object would replace a profile,
the snapshot, the lock file, the configuration or the binding, outside the lock.

`modelroom import-profile <file> --config <file>` reads such a file into another results folder,
under `modelroom.lock`. The whole run is planned under the lock and only then written, so the
plan *is* the freshness check -- every writer of a profile, a measurement or the configuration
holds the same lock. A conflict anywhere in the plan leaves the run without a single write. The
configuration is read once before the lock (a missing, invalid or schema-1 file is refused
without even creating a lock file) and once under it; the second read is the one the plan uses,
because another import may have added its own `[machines.<name>]` entry in between. If `[paths]`
changed meanwhile, the run stops with exit `1` and asks to be repeated.

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

The imported machine gets a `[machines.<name>]` entry: `<name>` is its `display_name`
normalized (everything outside `a-z0-9` becomes a single `-`, the ends are trimmed), made
unique with the short id and, as a last resort, with the whole `profile_id`; the reserves come
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

An import never changes the local binding -- it does not even read the pointer file. Broken
files in the target folder (a profile that does not read or validate, one of an unsupported
`schema_version`, a schema-1 profile, an unreadable measurement file) are listed in the report
and left alone; they never stop the import of another profile. The report carries one line per
profile and per measurement, and a run that changed nothing ends with `nothing to do`.

| Exit | `export-profile` | `import-profile` |
|---|---|---|
| `0` | written | written, or nothing to do |
| `1` | -- | another process holds the lock, or a stored file contradicts the import |
| `2` | the configuration is missing or invalid, `--profile` is not a `profile_id`, `--out` is one of the files the package writes itself or cannot be written, no profile is bound and none was named, the named profile is not in the folder, or it is still schema 1 | the configuration is missing, invalid, still schema 1, or holds a value the TOML writer cannot write back (`inf`), or a file could not be written (the message names it; every file already written is complete, and repeating the run finishes the import) |
| `3` | a stored profile, or the pointer file, does not read | the export file does not read, or its `schema_version` is outside the accepted range (`EXPORT_SCHEMA_RANGE`: schema 1 today) |

```console
$ modelroom export-profile --config /srv/models/modelroom.toml --out /srv/exchange/workstation.json
exported workstation (3f9a0c21d4e6b870) with 2 measurement(s) to /srv/exchange/workstation.json

$ modelroom import-profile /srv/exchange/workstation.json --config /srv/laptop/modelroom.toml
profile workstation: new
measurement 20260922T202000Z-4da3b585: new
measurement 20260923T083000Z-5c1e9a07: new
note: machine "workstation" added with the reserves from [defaults] and writer = false
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
true. `latest` is never derived from a number, a line or a family: a newer version number in
the same line does not make an earlier model `legacy`, only an explicit `successor` does. `Catalog.age_of(hf_repo)` returns `latest`, `legacy` with its successor, or
`unknown`. A model with a successor is never `latest`; successors stay under the family's
publisher and never form a cycle, within a family or across families; family names and
`hf_repo` are unique across the catalog.

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
  "source": "https://huggingface.co/acme/Nova-7B-2512"
}
```

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
      "source": "https://huggingface.co/acme/Nova-7B-2512"
    },
    {
      "hf_repo": "acme/Nova-7B-2512",
      "latest": true,
      "latest_source": "https://huggingface.co/collections/acme/nova-2512-0123abcd",
      "latest_checked": "2026-09-23",
      "successor": null,
      "ollama_base": null,
      "ollama_tag": null,
      "source": "https://huggingface.co/acme/Nova-7B-2512"
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
          "source": "https://huggingface.co/acme/Nova-7B-2512"
        },
        {
          "hf_repo": "acme/Nova-7B-2512",
          "latest": true,
          "latest_source": "https://huggingface.co/collections/acme/nova-2512-0123abcd",
          "latest_checked": "2026-09-23",
          "successor": null,
          "ollama_base": null,
          "ollama_tag": null,
          "source": "https://huggingface.co/acme/Nova-7B-2512"
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

A pass allows at most `metadata_ok` together with the other provenance rules; only a human
`Approval` makes a package `approved`. Search and fetch use the same function.

### Fit with profile v2

`compute_fit_v2(profile, package, base_model, scenario, machine_config)` is fit contract v1's
formula and classes, unchanged, behind a gate: `fit_block_reason(profile)` first (GPU state,
unknown RAM, a llmfit deviation), then `scenario.requests == 1` (more requests are the reverse
calculation), then the package rules (a complete `gguf` build, every weight file with a size).
An architecture the formula cannot read is the one case that is **not** `unknown` here: the
package is computed from its size instead, `basis: "size"` ("Fit from size" above, decided
2026-09-24). VRAM is `0` unless `gpu_state` is `measured`; the RAM
pool is `ram_physical_gib` (a limit is a note, never an input). The context is
`scenario.context_requested` for every package, never the package's own `default_context`, so
`context_assumed` is `false`. `base_model` (its architecture) and `machine_config` (its
reserves) are inputs because the formula needs them.

### Ranking rule

`rank_packages(entries, measurements, scenario)` orders one device's packages, best first, as a
total order that does not depend on input order:

1. fit class: `perfect`, `good`, `marginal`;
2. measured group: group 0 has a counting measurement, group 1 has none;
3. measured speed `tps_mean`, faster first (group 0 only);
4. quantization, in `QUANT_ORDER`;
5. larger weights first;
6. package identity.

Fit comes before speed: a package that fits comfortably ranks above a faster one that fits
only barely. A measurement counts (`group_zero_measurement`) when it is protocol `v1`, `valid`,
comparable, for the same `context_requested` as the ranking and for exactly this package
content (manifest digest, or repo, revision and a weight file's digest); the newest one wins.
`unknown` fits are set aside as not covered, with their reason; `too_tight` fits are listed
apart and never ranked; a computed fit for another context than the ranking's is a caller
error. `Ranking.top` is the first 10. The rule states facts in order; it recommends nothing.

The rule reads the fit class and not its `basis`, so inside one class a package judged from its
architecture and one judged from its size stand mixed. That is deliberate: the class is the
statement, the basis is how it was reached, and a reader sees the basis in the row itself
("Fit from size").

### SearchHit

One Hugging Face repository from the search, as the selection list shows it. Every field but
`repo` may be unknown (`None`, or the literal `unknown`). `publisher_status` is `publisher`,
`listed packager`, `other` or `unknown`. A hit is `resolved` only when exactly one publisher base
model is proven (relation `quantized`, publisher per catalog), so a resolved hit has
`base_model` equal to `[resolved_base_model]` and `base_model_relation` `quantized`; an
unresolved hit carries its
`unresolved_reason` (a relation status or `publisher_unknown`) and gets neither a fit nor an
Ollama name. `successor` is set exactly when `age` is `legacy`. `repo_created_at` is shown as
"repo created", never as the model's release date. `downloads` is how often the Hub says the
repository was downloaded (`ge=0`, `None` when the answer carries no count): an indication of what
many people take, **never a rank** -- the ranking rule alone orders the result.

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
model (`model_choices(..., configured_ollama=...)`, since 2026-09-25), else `None`.

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
    "fit_class": "good",
    "mode": "gpu",
    "need_gib": 6.3,
    "weights_gib": 4.25,
    "kv_gib": 1.125,
    "pool_gib": 10.94,
    "reserve_gib": 1.0,
    "context": 8192,
    "context_assumed": false,
    "basis": "size",
    "reason": null
  }
}
```

### Requirement

What one package needs for a scenario, `computed`, never measured: the reverse calculation.
`kv_gib_total` is `kv_gib_per_request x scenario.requests`; `need_gib` is fit v1's formula with
that total (`weights_gib x 1.10 + kv_gib_total + 0.50`); `perfect_reachable` is `false` in `cpu`
mode, where fit v1 caps at `good`. It carries no speed statement. `package_identity` is the
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
  "perfect_reachable": true
}
```

### Note

One plain-language note next to a package, a machine, a measurement or a ranking, built only
from facts: `facts` names the fields it was derived from, `origin` is their provenance, `text`
is at most 240 characters.

```json
{
  "code": "cpu_caps_at_good",
  "subject": "package",
  "origin": "computed",
  "text": "Runs in system memory only, so the best possible rating on this machine is 'good'.",
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

**Every page costs one request of the shared budget.** A page that fails -- a status other than
`200`, a body that is no JSON, a body that is no list, an entry with no repository id or one that is
not a repository id -- ends the search with `SearchError` **naming the page**
(`search for 'qwen' (mradermacher): unexpected status 503`); an account is never quietly
left out, because "unsloth has nothing for this word" and "unsloth was not asked" are different
answers. A budget that ends before the last page does the same and says how many accounts have no
answer (`... not made: budget exhausted; 6 accounts were not asked: deepseek-ai, unsloth, ...`).

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
takes the lock of the configuration it writes: two callers creating the same file at once are kept
apart when they name the same `paths.state` -- every guided run does (`<folder>/state`) -- and not
when a library caller names another one. No backup is kept: the guided mode has the user confirm
the change first.

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
labeled "repo created" and is never read as a release date or as an age.

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

**Eligibility** is unchanged: `provenance in ("metadata_ok", "approved")`, `complete`, `active`.
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
scenario and `ranking.RANKING_RULE` are printed in every view.

**The ranking** per machine is `ranking.rank_packages` ("Ranking rule" above): `ranked` holds the
first `ranking.TOP_LIMIT` entries and `ranked_total` says how many there were, `not_covered`
holds the `unknown` fits with their reason, `too_tight` the ones that do not fit. A measurement
counts only in measured group 0, and only then are `measurement_id` and `speed_tps` set.

**Provenance per value.** Every computed number carries `(computed)` in its column name
(`Weights GiB`, `Fit`, `Need GiB`, `Pool GiB`), the measured one carries `(measured)`
(`Speed tok/s`), and a profile's own readings carry their source in the Machines table
(`Origin`, `RAM source`, `VRAM source`). Nothing computed is ever labeled as a measurement.

**One note per package**, built from named facts only (`Note`, "Note" above): a measured entry
gets `measured_here` with `origin: measured`; an entry without a measurement gets the note for
its fit mode (`fits_in_graphics_memory`, `shared_between_memories`, `cpu_caps_at_good`); a
set-aside package gets `not_covered` or `too_tight`. `facts` names the fields the text was
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
then prints the blocks on its own; `modelroom render` passes neither and stays silent on success.

**Sections of the Markdown view**, in this order: the header line, the summary block (title,
snapshot run time, rendered time, base model and package counts, `Scenario:`, `Ranking rule:`,
and `Market rating unavailable: <message>` when the rating source failed), `## Areas`,
`## Machines`, and then per machine `## Ranking: <machine>`, `## Not covered: <machine>` and
`## Too tight: <machine>` -- the last two only when they have rows. Every machine section is a
`##` heading, so a reader can cut the document at headings.

**The terminal view** is what step 5 of the guided mode shows, and `document_terminal` carries that
step's head; `modelroom render` writes the two files and stays silent unless a caller asks for it.
The guided mode draws the head and the card itself and then prints `views.terminal_lines`, which is
the blocks alone. It shows fewer columns than the Markdown table (a terminal is narrow) and the ones
a reader recognizes a package by: `#` (3), `Model` (24, the base model's name without its account),
`Package` (22, `packager · quant`, cut with `…` where it does not fit), `Fit` (20), `Speed` (12) and
`Memory` (8, `need_gib` as `GB` with one decimal). With the one space every line is indented by,
that is 100 characters, and no line of this view is wider.

Above each machine's table stands `<label> · context L 32k`, and `<label> · <status>` with its
reason for a machine that is not `ranked` (decided 2026-09-24: `Ranking: dellpromax-jv
(DellProMax_JV)` said the machine's key, its name and a word nobody asked about, and the scenario
stood on a line of its own). **Fit** is the class with ` (RAM)` behind it where the pool is system
memory -- the same word the selection list of step 2 uses -- and never ` (from size)`, which is a
note now. **Speed** is `–` where nothing was measured, not `unknown`.

Under the table, **every line carries the label of what it says** (decided 2026-09-25), in the
card's own column (`intro.label_line`, `LABEL_COLUMN`) so the report of step 5 reads as one block
from its `folder` row down to the install command. Running text buried the command a reader came for
(test round, 2026-09-25). The label is drawn in the color of the card's labels, the text gray, and a
line that continues the one above it carries no label of its own. The pieces of one label are joined
with ` · ` and wrapped between pieces at what the label leaves of the 100 characters
(`screen.joined`):

1. `shown` -- `<n> of <m> packages`, and one piece per set-aside group: `<k> too tight (<up to three
   names> and <r> more)`, `<k> not covered (…)`, with the reason named where the list has more than
   one of them. The word `package` is the count's own and is not repeated per piece. The names are the
   evidence behind the count and are what gives way where the line has no room for them -- two, then
   one, then none; the count and the reason never do, and no name is cut in the middle (a piece of 137
   characters was measured in the second-model round of 2026-09-24).
2. `memory` -- the fit of the ranked rows **bundled by memory pool**, with their rank ranges and
   **one pool per line**: `#1–2 fit into graphics memory, 10.9 GB free after the reserve`, `#3–5 need
   system memory, the graphics card helps`, `#6 needs system memory, no graphics card`. Two
   statements about two memories joined by a `·` read as one. The numbers come from the fields of
   `Fit`, never from parsing a note's own sentence back apart.
3. `basis` -- `#2–5 computed from the package size, not its architecture`, where there are such rows.
4. `speed` -- `#3 measured 41.1 tok/s` per measured row, or `nothing measured in the rows shown · say
   Yes in step 4 to measure an installed package` where no row carries a speed. **Of the rows shown,
   not of the machine**: the ranking rule sorts by fit class first, so a measured package can stand
   behind eleven unmeasured ones and be in no `RankedEntry` of the document at all -- the card of the
   same screen counts that measurement, and the two may not contradict each other (second-model
   round, 2026-09-24 and 2026-09-25).
5. `install` -- `#1  ollama pull <local name>`, behind a blank line, with the rank gray and the
   command drawn as a command, for the first row of the machine **the run passes in**
   (`views.show_result(..., install_for)`; the guided mode passes its own machine key) --
   `hf.co/<repo>:<quant>` for a Hugging Face build, `<name>:<tag>` for one of the Ollama registry,
   the two shapes the load test matches an installed model by. Nothing is called and nothing is
   looked up; a package with no such name has no line. `modelroom render` draws no such line at all:
   a configuration may name several writers, and which of them the command runs on it cannot know
   (second-model round, 2026-09-24). It is a command to copy, so it is never cut, and it is the one
   line of this view that may pass 100 characters. Installing is a work package of its own.

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
`measurement_group` 0 carries `measurement_id` and `speed_tps`, group 1 carries neither. The
`fit` is always `perfect`, `good` or `marginal` -- the other two classes are set aside.

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
  "provenance": "metadata_ok",
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
    "fit_class": "good",
    "mode": "gpu",
    "need_gib": 7.125,
    "weights_gib": 5.0,
    "kv_gib": 1.125,
    "pool_gib": 10.94,
    "reserve_gib": 1.0,
    "context": 8192,
    "context_assumed": false,
    "basis": "architecture",
    "reason": null
  },
  "measurement_group": 0,
  "measurement_id": "20260923T083000Z-5c1e9a07",
  "speed_tps": 50.4
}
```

### SetAsideEntry

One package outside the ranking: an `unknown` fit (not covered) or a `too_tight` one. Same
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
  "provenance": "metadata_ok",
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
    "reason": "a weight file has no size"
  },
  "reason": "a weight file has no size"
}
```

### MachineRanking

One machine's block. `label` is the profile's `display_name`, the schema-1 file's name or the
machine name, depending on `status`; `ranked_total` is how many packages the rule ranked in all.

```json
{
  "machine": "workstation",
  "status": "ranked",
  "label": "workstation",
  "profile": {
    "schema_version": 2,
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
      "provenance": "metadata_ok",
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
        "fit_class": "good",
        "mode": "gpu",
        "need_gib": 7.125,
        "weights_gib": 5.0,
        "kv_gib": 1.125,
        "pool_gib": 10.94,
        "reserve_gib": 1.0,
        "context": 8192,
        "context_assumed": false,
        "basis": "architecture",
        "reason": null
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
      "provenance": "metadata_ok",
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
        "reason": "a weight file has no size"
      },
      "reason": "a weight file has no size"
    }
  ],
  "too_tight": []
}
```

### RenderDocument

Everything one render says. `ratings` holds the rating of every base model the `RatingSource`
answered for; a base model that is not a key has no rating. A failed rating source sets
`rating_unavailable` to its message and leaves `ratings` empty, and the run still ends `0`.

```json
{
  "schema_version": 1,
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
  "ranking_rule": "fit class (perfect, good, marginal), then measured group (valid comparable measurement first), then measured speed (faster first), then quantization, then larger weights, then package identity",
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
        "schema_version": 2,
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
          "provenance": "metadata_ok",
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
            "fit_class": "good",
            "mode": "gpu",
            "need_gib": 7.125,
            "weights_gib": 5.0,
            "kv_gib": 1.125,
            "pool_gib": 10.94,
            "reserve_gib": 1.0,
            "context": 8192,
            "context_assumed": false,
            "basis": "architecture",
            "reason": null
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
          "provenance": "metadata_ok",
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
            "reason": "a weight file has no size"
          },
          "reason": "a weight file has no size"
        }
      ],
      "too_tight": []
    }
  ]
}
```

## Load test (stage 1)

The load test measures how fast a package really runs on this machine. **Stage 1 measures
packages the local Ollama daemon already has**: it downloads nothing, removes nothing and says
nothing about disk space. `modelroom/loadtest.py` is the work of it, `modelroom/daemon.py` the
transport it talks through, and `modelroom/guided_loadtest.py` the guided mode's step 5 around
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
`GET /api/ps`, `POST /api/show`, `POST /api/generate` -- checked before a connection is opened;
anything else is a `DaemonError`. There is no call of `/api/pull` or `/api/delete` anywhere in
`modelroom/`, and a test reads every module to prove it. **Time limits:** 10 s for every short
call, 600 s for one `POST /api/generate` (a CPU machine needs minutes for 128 tokens). A limit
that is reached, a refused connection or a body that is not JSON is a fault with a reason, never
a traceback. **Documented limit:** whatever answers on that port is taken to be this machine's
daemon -- nothing in the answer proves it, and nothing here pretends to check.

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
record it writes says `protocol: v1` and `MeasurementRecord` checks a valid one against that same
shipped definition. A run after another prompt, seed or warm-up count would be stored under a name
it does not deserve, and no validator could tell afterwards. A new protocol version is a new
writer.

**The run** (`run_load_test`). In this order: `GET /api/version` (stored as `daemon_version`,
never a validity criterion), the machine's load once (`LoadState`), the warm-up run, then the
measured runs of protocol v1. **`/api/ps` is read after every run, the warm-up included**, so
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

**What never happens in stage 1:** no download, no `pull`, no `delete`, no clean-up, no disk
space statement, and no second way of computing a speed. Stage 2 (download with a consent
screen) is a work package of its own.

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
  grayed-out entries with their reason, and the instruction lines under it: `↑↓ move   Enter
  select   Esc leave`, or `↑↓ move   Space marks   Enter takes the marked rows, or this one   Esc
  leave` for a list that marks. **Yes or no is a list of `Yes` and `No`** with the default under the
  pointer; there is no `(Y/n)` anywhere. An answer file still answers it with `true`/`false`.
- **A list that marks answers with what is marked, and with the row under the pointer where nothing
  is marked** (`dialog.TerminalAsker.checkbox`, decided 2026-09-25). `Enter` alone used to take
  nothing at all, and a reader read the pointer as the selection ("I thought the bar was the
  selection", test round 2026-09-25). With something marked the pointer's row is a row a reader
  moved past, not a choice, so it is not added. The rule is one key binding added after
  `questionary` has built the question, like `Esc`; the library itself is unchanged, and `FileAsker`
  is untouched -- `select = []` in an answer file still means nothing.
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
- **A list may carry a column head and gray rows.** `dialog.Choice(heading=True)` is a line of the
  list that is no entry -- the column head of a table-shaped list; the pointer never rests on it,
  and `dialog.selectable` leaves it out, so no answer can name it. `dialog.Choice(dim=True)` draws
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
| `filter_owners`, `load_test` | `Yes` / `No` |
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

**Five steps.** Each one is a block: a head of 72 characters (`── Step <n> of 5  <name> ───…`), one
answer line per question, at most two notes, and one balance line -- a check mark, the step's name
and what the step came to (`✓ Packages        2 models, 26 packages from 4 repositories`), or a dash
where it did nothing (`– Measurement     none in this run`). Step 5 closes with the card of the
whole run over the result table and `✓ Results         docs\models.md`, the document's path relative
to the results folder. The five are **1 Configuration** (the results folder, the configuration in
it, the machines), **2 Packages** (search, choice, fetch), **3 Context** (the size scale),
**4 Measurement** (the load test), **5 Results** (the render).

The balances, one per step: `<folder>, <n> machine(s)` · `<k> models, <p> packages from <r>
repositories` (nothing chosen, and nothing to fetch: `nothing chosen, the folder stays as it is`) ·
the context as `L 32k` · `<n> of <m> measured` or `none in this run` · the Markdown path. All three
numbers of step 2 are about **the snapshot this folder now holds**, not about what this run marked:
`apply_hits` keeps the models that were already there, so counting the marked models and the packages
of the whole folder in one sentence would mix two frames (second-model round, 2026-09-24).

The **fetch belongs to step 2**, before the context question and not after it: the scale of step 3
counts how many of the packages the fetch just recorded still fit this machine, and the fetch
itself does not depend on the context. An answer file is a table of keys and its order does not
matter, so the move is invisible to it.

**The questions.** Every question has a key, which is also its key in the answer file:

| # | Key | Question | Answer |
|---|---|---|---|
| 1 | `results` | Where should results live? | `here` or `path` |
| 1 | `results_path` | Path to the results folder | text (only after `path`) |
| 1 | `write_config` | Write a configuration into this folder? | true/false (only when the folder already holds results but no `modelroom.toml`) |
| 1 | `machines` | Which machines should the result cover? | a list out of `this-machine`, `import` |
| 1 | `import_file` | Path to the profile file to import | text (only after `import`) |
| 1 | `clone` | Is this the same machine or a clone? | `same` or `clone` (only on `ask_clone`; both measure -- `same` under the bound profile id, `clone` under a new one) |
| 2 | `search` | What are you looking for? | text: a word, a name with blanks, a repository id, or the empty string ("show me what fits this machine") |
| 2 | `filter_owners` | Show only repositories of a publisher or a listed packager? | true/false |
| 2 | `did_you_mean` | Nothing was found. Did you mean one of these? | one of the candidates, or `keep` (only when the search found no model and the catalog knows a word close to the one that was typed) |
| 2 | `select` | Which of these models should the result cover? | a list of base model ids (`Qwen/Qwen3.5-9B`); an answer file may name repository ids instead, as before |
| 3 | `context` | How much text should a model handle at once? | a level of the scale (`"XS"` … `"XXL"`), or a whole number of tokens; the pointer starts on `[guided].context` when this folder kept one, else on `L` |
| 4 | `load_test` | Measure the speed of the checked models that are already installed here? | true/false, default false; **the one optional answer** -- an answer file that does not mention it does not measure |
| 4 | `load_test_packages` | Which of these installed models should be measured? | a list of local Ollama names (only after `load_test` true) |

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
merge of two configurations is attempted: the five changes are the machine entries for the profiles
the folder holds, this machine as the writer, the profile a measurement recorded, the models the
search added and the context the scale kept. Two of them have a rule of their own:

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
per base model would only spend the shared request budget), this device as `[machines.<host>]`
with `writer = true`, `paths.state = <folder>/state`, `paths.markdown = <folder>/docs/models.md`
and `[guided].results = <folder>`. The folder is remembered in the pointer file. A
`modelroom.toml` of schema 1 in the folder is migrated first (`modelroom migrate`, under the
lock, backup kept, idempotent), before anything is written, and the notes it leaves have one
sentence in front of them that says what happened in plain words: `this folder holds a
configuration from an earlier version; it was updated, backup kept: modelroom.toml.v1.bak`. A
folder that already holds results but no `modelroom.toml` is a question, never an assumption.

**What step 1 leaves on the screen** (decided 2026-09-24): the answers, and **one note about the
measurement** -- `measured: <the hardware in plain words>, confirmed by llmfit`, or `measured again:
…` where this machine was already measured in this folder (and for a `the same machine` answer). The
sentence ends after the hardware where `llmfit` confirmed nothing. The readings themselves, the
profile id, `wrote <toml> with <name> as the writer of this results folder`, `<name> is now a writer`
and `<name> is measured as profile <id>` are the configuration's and the profile file's business and
are no longer printed (`cli.hardware_with_config(..., summary=False)`; `modelroom hardware` prints
its summary line unchanged). An import leaves `imported <display_name>: <the hardware in plain
words>`; an import that does not go through stays a `run.failed` sentence.

**Step 1, the machines.** The list is built from every profile file in the results folder
(`scan_profiles`), grouped by hardware class (GPU, VRAM, RAM) with the count and the names,
alphabetical. The group is a matter of operation only -- the ranking is computed per device.
Those entries are shown and cannot be picked: they are already part of the result. So are the
schema-1 files (`run modelroom migrate`), the files that do not read (with their reason), and
`enter a machine by hand` (`stage 2`). The two entries that can be picked are `this machine` and
`import a profile file`.

**This machine is marked only when it has not been measured here.** With no profile bound to this
machine for this folder the entry is `this machine (measure now)` and starts marked -- that is
what a first run is for. Once there is one, it is `this machine (measure again, last measured
<date>)` and starts **unmarked**: measuring again is a decision, and `Enter` alone must not make
it. The key and the answer stay the same (`machines`, `this-machine`), so an answer file is
unaffected.

`this machine (measure now)` makes sure `[machines.<host>]` exists and is a writer -- this
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
`hardware` leaves to it. `import a profile file` runs `import-profile`, which never changes the
binding; a file that does not import is reported and the run goes on.

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
The head is the first line of the list and no entry of it (`guided_models.list_header`,
`dialog.Choice(heading=True)`): `Model  Fit  Size  Release  Packagers  Downl.  Ollama`, indented by
two so its columns stand over the cells -- a heading is drawn without the marker of a row. Without
it a reader had to guess what a cell meant and `latest` was invisible between two columns of numbers
(test round, 2026-09-25).

Seven columns, and the **whole** line is at most 100 characters -- ` ❯ ○ ` included, which is the
one space every line of this screen carries plus the pointer and the marker (`POINTER_WIDTH`, 5;
a label of 100 made a line of 105): the model's name (the part after the `/`, 24), the fit in a word
(14: `good`, `marginal`, `too tight`, `unknown` -- and `good (RAM)` or `marginal (RAM)` where the
graphics memory is too small and the fit is against system memory: fit v1 caps that pool at `good`,
and live on 2026-09-24 a 122B model stood as `good` above a 9B `marginal` that fits the graphics
card, with nothing to tell the two apart), its size (6: `9B`, `0.6B`, `2.4T`, `unknown`), where it
stands in its family (`Release`, 7: `latest`, `legacy`, `–`), the packager accounts that have it
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

The order is the memory pool (a fit in graphics memory before one in system memory, whatever its
class), then the fit class, then the release (`latest`, then a release nobody knows, then `legacy`),
then the downloads, then the name, with `too tight` and `unknown` last (decided 2026-09-24,
release 2026-09-25). A **`legacy` row is drawn gray as a whole** (`dialog.Choice(dim=True)`): the
family has moved on, and that belongs in the reading of the row rather than in one cell of it.

How many models and how many repositories the search answered with is the second note above the
list, and the reasons of the repositories that are no model of it are in `search.json` ("Search
log"; `guided_models.UNRESOLVED_REASONS`, and with `filter_owners` on `not a publisher or a listed
packager` for an owner class of `other` -- except for the one repository the person typed, which
that filter never hides) -- the repositories themselves are no lines of the list either. What the
list has to say about itself is its **instruction line** (`guided_models.hint_line`), under the
list and behind the line of the keys: `latest: the publisher's current release of its family ·
legacy: the publisher named a successor · fit from the size at 32k context, exact after the fetch`
-- with `fit unknown until this machine is measured` in place of the last piece when this folder
holds no measured machine, and always the context the fit was really computed for. What nothing
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
(`guided_models.parameters_from_name`: every `<number>B`/`<number>T` that does not stand behind a
letter, the largest of them, `2.4T` as 2400 and shown as `2.4T` again -- so `Qwen3.5-35B-A3B-MTP` is 35, `Nova-A17B` and
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
  (`fit.count_fitting` over `compute_fit_v2`, one request, 16-bit KV cache) over the packages of
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
- **The cap**: when every base model of the snapshot declares an `Architecture.max_context`, the
  levels above the smallest of those are grayed out with `beyond the window of <base model>` and
  cannot be answered, in an answer file either (exit `2`, naming the reason). If one base model
  declares none, nothing is grayed out -- a window nobody knows is no reason to forbid a level.
- **The answer** is a level name or a number of tokens; the level is translated into its tokens
  before anything is written. An answer out of a file is stripped of surrounding blanks **before**
  it is held against the grayed-out entries, so `" XXL "` is refused exactly as `"XXL"` is.
  Anything else ends the run with exit `2`, naming what was answered.
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

**Step 5, the render and the card.** `render_with_config` with the chosen scenario, which writes both
views -- so a measurement the load test just wrote is in the ranking of that same run -- and hands
the document itself back (`on_document`). From it the step draws **the card of the whole run**
(`views.result_card`, the `label value note` rows of the start screen, decided 2026-09-24), then one
table per machine (`views.terminal_lines`), then `✓ Results         docs\models.md`. The card is the
report a reader looks at once the run is over:

| Row | Value | Note |
|---|---|---|
| `folder` | the results folder | -- |
| `machine` | one row per machine of the result, its `display_name` | `12 GB graphics, 127 GB memory, measured today` (else `measured <YYYY-MM-DD>` from the profile's `recorded_at`); a machine without a profile carries its own reason |
| `models` | the base models of the snapshot, by name | `<p> packages from <up to three packager accounts, else "and n more">` |
| `context` | `L 32k` | the words and the example of that level; a context of its own: `about 30,000 words` |
| `speed` | `not measured` or `<n> measured` | `say Yes in step 4 to measure an installed package`, or `fastest <name> at 41.1 tok/s` |
| `result` | the Markdown path **relative to the results folder** | `<ranked_total> packages ranked, <k> too tight` (else `, none too tight`), and every other set-aside reason with its count behind a `·` |

The `speed` and `result` rows count **one** machine's ranking: the machine of the run, else the first
that has one. A package that fits two machines is one package and two rows, and a sum over the
machines would count it twice while the `models` row of the same card counts the packages of the
snapshot once (second-model round, 2026-09-24). The tables below the card carry every machine.

`Written to <markdown>   and   <state>` is gone from the screen: the card's `result` row and the
balance of the step both say where the document is, and the run said the same path three times
(test round, 2026-09-24). `modelroom render` still prints it.

**A step 5 with nothing to rank** -- the folder holds **no snapshot** -- says so and draws the card
all the same (`views.show_nothing`, decided 2026-09-25): the note `nothing to rank: no package in
this folder yet`, the card with the rows it still knows -- the folder, the machines the configuration
names with the profiles this folder holds (`render.machine_profile`, the same decision the document
would have rested on), `models none`, the context that was chosen, no measurement and `result –` --
and the balance `– Results         nothing to rank`. Before that the step was a head with nothing
under it and the reason stood on stderr, where no reader of the run is (test round, 2026-09-25). The
exit code is the render's, unchanged (`1`, "nothing to render").

**Every other render that writes no document draws nothing here at all**: a lock another process
holds, a snapshot of an unsupported schema, a view that could not be written. None of those is a
folder without packages, so none of them may be reported as one; the command's own sentence and its
own exit code stand (second-model round, 2026-09-25). A snapshot that holds packages but none a fit
may be computed for is **not** this case either -- the render writes its document, and the table
says `no package of the configured base models is ranked here` with `shown     0 of 0 packages`.

**Step 4, the load test** (stage 1, "Load test (stage 1)" above; `modelroom/guided_loadtest.py`,
which `guided.py` calls between the context and the render). It measures into the profile **the
pointer file binds this machine to for this results folder**, and only when the takeover rule
(`resolve_profile_target`, "Profile binding") says `bound` for it -- the profile file is in the
folder and its `os_fingerprint` agrees with this machine's. Never into `[machines.<host>].profile`:
a configuration is shared, and a stale or foreign entry there would give this machine's speed to
another device -- the very case the machine step declines with "put that profile file back".
Without a binding, without that agreement, or without a snapshot to take packages from, the step
says so in one line and the run stays `0` -- there is nothing a load test could be about. Otherwise
the daemon's inventory is read and matched against the snapshot's active packages.

- **The daemon could not be read.** The `load_test` question is asked first. A **no** (which is
  also what an answer file without the key means) is one line, `no load test: <reason>`, and exit
  `0`: the daemon is needed for this step and for nothing else, so a machine without one still
  finishes cleanly. A **yes** that cannot be honored is a step that did not finish -- `nothing
  measured: <reason>` and exit `1`, with the document still written.
- **No installed package matched.** One note, `no ranked package is installed on this machine;
  the load test measures installed packages only (stage 1: no download)`, and no question. What
  was left out is named first, **one note per reason** with the number of installed models behind
  it and **no names** (`12 installed models -- <reason>`): which model exactly is `ollama list`'s
  answer, and twelve names pushed the question off the screen (test round, 2026-09-24). The **cloud
  models** are named only here -- with a candidate on the list, that a machine also holds cloud
  models is beside the point. An installed model whose name matches a configured package but whose
  digest the daemon did not show is named **either way**: that is a fault about a package a reader
  may have wanted measured, and another candidate must not silence it (second-model round,
  2026-09-24).
- **At least one candidate.** `load_test` (default **no**), then `load_test_packages`: every
  candidate, **none of them marked**, one line each with the local name, the base model, the
  quantization and the weight size, aligned in columns. Nothing is marked because `Enter` measured
  all three in the hand test of 2026-09-24 where the user had picked one; which models are measured
  is a decision. A single candidate is a list with one entry as well -- one code path, and an
  answer file reads the same on every machine. Before the runs the step notes that the load is
  read once and that the model behind the name must not change, and the progress of each run is a
  note of its own. Each measurement is written as its own file and reported in one note:
  `measured <name>: 41.1 tok/s (39.8–42.3), context 32k`. That it is valid and comparable is what
  makes it a measurement at all, so it is not said; a record that is neither carries **what it is
  missing** instead (`<name>: not comparable (<reason>)`) and the run still ends `0` -- the record is
  written and the ranking says where such a measurement stands. A daemon that lets a run down here
  is a step that did not finish: the line names the model and the reason, the exit code is `1`, and
  the document is still written.

**The answer file** (`modelroom/answers.py`) is TOML with `schema_version = 1` and one key per
question; an answer may be text, a whole number, `true`/`false` or a list of texts. A question
with no answer ends the run with exit `2` and names the question; an answer that is not one of
the offered choices, or of the wrong shape, does the same. An unsupported `schema_version` is
exit `3`. **The one exception is `load_test`:** a file that does not mention it does not measure,
because whether that question is reached at all depends on the machine the file is run on, not on
the file. A `load_test = true` without `load_test_packages` is a missing answer like any other.

```toml
# modelroom guided answers
schema_version = 1
results = "here"
machines = ["this-machine"]
search = "qwen"
filter_owners = true
select = ["unsloth/Qwen3.5-9B-GGUF"]
# A level of the scale, or a number of tokens: `context = 8192` and `context = "8192"` are the same
# answer as `context = "S"`.
context = "L"
load_test = true
load_test_packages = ["hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M"]
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
