# ModelRoom

Does this model have room on your machine?

ModelRoom (package and command: `modelroom`) answers two questions for a list of model
families you care about:

1. **Which local packages exist?** It reads the Hugging Face API and the Ollama registry for
   the base models and packagers you allow, and records every package with its size, format,
   quantization and the exact revision it was seen at.
2. **Which of them fit a given machine?** It measures a machine with
   [llmfit](https://github.com/AlexsJones/llmfit) and computes, per package, whether weights
   plus KV cache plus a configured reserve fit into the memory that machine has.

The result is a JSON snapshot, one hardware profile per machine and a rendered Markdown table
with one fit column per configured machine. Nothing in the tool is tied to a specific
operator: you bring a configuration file with your families, packagers and machines, run it
on the machine you want to size, and keep the files wherever you keep your notes.

## Status

Alpha, version 0.1.0, the first release (see `CHANGELOG.md`). The three commands `fetch`, `hardware` and `render` work end to end; the data
shapes are versioned contracts (see `CONTRACTS.md`). Expect changes to the rendered layout and to the fit rules before a stable
release. `CHANGELOG.md` lists what has landed.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- `llmfit` 1.1.16 or newer on `PATH` is optional: `hardware` measures the machine itself and
  uses `llmfit` only to cross-check the physical RAM and the VRAM. Without it the profile
  records both checks as `absent` and the command still succeeds. The minimum version is set
  in the configuration (`[llmfit] min_version`).
- `nvidia-smi` on `PATH` for a machine with an NVIDIA card; without it the profile records
  that an adapter is present but unmeasured, and no fit is computed for it.
- Internet access for `fetch` only.

## Installation

As a command-line tool from the Python Package Index, with either installer:

```bash
uv tool install modelroom
pipx install modelroom
```

Both put `modelroom` on `PATH` in an environment of its own. Then copy
`modelroom.example.toml` from this repository as your configuration and follow the quick start
below with `modelroom` in place of `uv run --frozen modelroom`.

## Quick start

From a clone of this repository, after `uv sync --frozen`:

```bash
cp modelroom.example.toml modelroom.toml
uv run --frozen modelroom hardware --config modelroom.toml
uv run --frozen modelroom fetch --config modelroom.toml --machine workstation
uv run --frozen modelroom render --config modelroom.toml
cat docs/models.md
```

1. The example configuration names three families, the packagers and publishers it allows,
   two machines (`workstation`, a writer, and `inference-server`) and two paths. Relative
   paths resolve against the configuration file's own directory, never the current working
   directory. Edit the copy for your own models and machines.
2. `hardware` measures the machine it runs on and writes `state/hardware/<profile_id>.json`.
   Run it once on every machine you want to size; the machine is remembered in a small pointer
   file in your home folder, so a second run updates the same profile. `--machine <name>` is
   optional and adopts that configuration entry's profile; `--cpu-only` judges the machine as a
   CPU machine, `--new-identity` measures a clone as a machine of its own.
3. `fetch` queries the registries and writes the snapshot `state/modelroom.json`. The named
   machine has to be marked `writer = true`.
4. `render` reads the snapshot and every machine's hardware profile and writes the Markdown
   table to `paths.markdown` (`docs/models.md` in the example). It takes no `--machine`.
5. Read the table. Only active, complete packages with `metadata_ok` or `approved`
   provenance are shown. For each base model and packager, every machine picks the largest
   quantization that fits it `good` or better (else the smallest package it could judge).
   The table shows one row per picked package and one `fit (computed, v1): <machine>`
   column per configured machine.

`--config` is required by every command; there is no default path.

## Files

Under `paths.state`:

| File | Written by | Kind |
|---|---|---|
| `modelroom.json` | `fetch` | snapshot, versioned contract |
| `hardware/<profile_id>.json` | `hardware`, on that machine | hardware profile, versioned contract |
| `modelroom.lock` | `fetch`, `render` | runtime only, not a contract |
| `run-status.json` | `fetch` | runtime only, not a contract |

The snapshot and the hardware profiles carry a `schema_version` and can be kept under version
control. The lock file and `run-status.json` are operating data of one deployment. The lock
file is never deleted: the lock is a kernel lock on it, which the operating system releases
when its holder exits, and a second `fetch` or `render` against the same state directory
stops at it instead of waiting.

## What the fit class means

For a judged fit, the cell reads `<class> (<mode>, need <n> / pool <n> GiB)`. The class is
computed from a memory formula: weights plus 10 %, plus a 16-bit KV cache for the package's stated context
(8192 when the package states none), plus 0.5 GiB. The result is compared with the machine's
VRAM minus its reserve, or with its RAM minus its reserve when it does not fit the GPU.
`perfect` needs at most 60 % of that pool, `good` 85 %, `marginal` 98 %; anything above is
`too_tight`. Off the GPU the class is capped at `good`. The full rule is in `CONTRACTS.md`,
"Fit contract v1".

What it does not say: the class is arithmetic, not a measurement. It makes no statement about
tokens per second, output quality or whether a runtime actually loads the package. It is
labeled "fit (computed, v1)" in the output for that reason.

## Limits of fit v1

- Only dense transformer architectures are judged. A hybrid or mixture-of-experts
  architecture comes back `unknown` with the reason "architecture not covered by v1". When no
  machine can judge any package of a group, the table shows one row with
  `no recommendation: <reason>` instead of guessing a package.
- Only complete GGUF packages are judged; other formats are `unknown`.
- A machine without a hardware profile shows `no profile` (or `no recommendation: no profile`).
- Unified memory is not modeled. The profile records llmfit's `unified_memory` flag, but the
  fit treats the reported VRAM and RAM as two separate pools. A result for such a machine has
  not been validated.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success: every fetch area complete; `hardware` measured and wrote the profile (also when `llmfit` was not there to cross-check it); `render` wrote the document (also when its rating source failed) |
| `1` | at least one fetch area incomplete (complete areas are still written); or another process holds the lock; or this `fetch` is not newer than the stored snapshot; or `render` has no snapshot; or the existing document was rendered from a newer snapshot; or `hardware` wrote the profile but could not record the binding in the pointer file. Every case but the first and the last leaves the snapshot, hardware profiles, `run-status.json` and the rendered document unchanged; the lock file may be created or updated |
| `2` | the configuration is missing or invalid; the machine is not a writer (`fetch`) or not configured (`hardware`); `hardware`'s bound profile belongs to another machine (run it with `--new-identity`); or a tool a command requires is missing, older than the minimum, fails, or returns invalid output |
| `3` | an unsupported `schema_version` in the configuration, the snapshot, a hardware profile or the pointer file; or one of them is not valid JSON or does not match its model |

Source: `CONTRACTS.md`, "Exit codes", and `AGENTS.md`.

## Network access

Requests go only to `huggingface.co`, `ollama.com` and `registry.ollama.ai` over HTTPS and to
the local Ollama daemon at `127.0.0.1:11434` over HTTP; every redirect and pagination link is
checked against the same list before it is followed. `HTTP_PROXY`, `HTTPS_PROXY` and
`NO_PROXY` from the environment are honored, but a proxy only changes how an allowed request
travels, never which targets are allowed. Details: `CONTRACTS.md`, "Transport security and
proxying".

## Language choice

Python, because the catalog logic and Pydantic contracts it integrates with are Python, and the
run has to be platform neutral (Windows, Linux, macOS). Everything in this repository, from
identifiers to documentation, is English.

## Release

For maintainers, from a clean clone at the commit to be released:

```bash
uv run --frozen python scripts/release-check.py
uv run --frozen python scripts/release-smoke.py
```

The first is the release gate (clean tree, version and changelog, public hygiene over the files,
the history and the commit metadata, content of wheel and sdist); with `--tag vX.Y.Z` it also
checks the release tag. The second installs the built wheel into a fresh environment and runs
`hardware`, an offline `fetch` and `render` against it. Both must end with exit `0`; the full
procedure is in `AGENTS.md`, "Versioning and releases".

## Contributing and conventions

`AGENTS.md` is the tool-neutral rule set for anyone (human or agent) working on this
repository. `CONTRACTS.md` describes the data shapes and their versioning. Issues and pull
requests are welcome.

## About

ModelRoom is built and maintained by [VISCONSULT](https://vis-consult.eu), a consultancy in
Germany that runs AI agents for its own work and for clients under EU data-protection rules.
We use it to decide which models run on our own laptops and servers, and with clients to size
on-premise deployments before anyone buys hardware.

## License

MIT, see [LICENSE](LICENSE).
