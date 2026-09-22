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

Every persisted form carries an integer `schema_version`. Each reader declares the inclusive
range it accepts. A file outside that range is refused with exit code `3` and a message that
names the file, its version and the accepted range, before the command writes anything. A
version bump is a documented decision (`docs/adr/`), never a side effect.

## Identity

Package identity is the repository identifier as the registry spells it (`hf_repo` for
Hugging Face, `ollama_base` plus tag for Ollama). The package never normalises, canonicalises
or merges model names on its own; whoever integrates it maps identities on their side. Every
recorded fact is bound to the revision it was observed at: the commit SHA on Hugging Face,
the manifest digest on Ollama.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | at least one fetch area ended incomplete; complete areas were published |
| `2` | required external tool (`llmfit`) missing or below the minimum version |
| `3` | an input file has an unsupported `schema_version` |
