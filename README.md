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

Alpha, version 0.1.0, the first release (see `CHANGELOG.md`). The guided mode and the six
subcommands work end to end; the data shapes are versioned contracts (see `CONTRACTS.md`). Expect changes to the rendered layout and to the fit rules before a stable
release. `CHANGELOG.md` lists what has landed.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- `llmfit` 1.1.16 or newer on `PATH` is optional: `hardware` measures the machine itself and
  uses `llmfit` only to cross-check the physical RAM and the VRAM. Without it the profile
  records both checks as `absent` and the command still succeeds. The minimum version is set
  in the configuration (`[llmfit] min_version`).
- `nvidia-smi` on `PATH` for a machine with an NVIDIA card; without it the profile records
  that an adapter is present but unmeasured, and no fit is computed for it.
- An Ollama daemon on `http://127.0.0.1:11434` for the load test only. Without one the guided
  mode says so in one line and goes on; nothing else in the tool needs it.
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

Type the command and answer the questions:

```bash
modelroom
```

It opens with what it already knows and then walks five numbered steps:

```
 ██ ██ ▓▓ ██   ModelRoom
 ██ ██ ██ ░░   Which local model packages fit your machine.
 ▓▓ ██ ░░ ░░   modelroom 0.1.0 · github.com/jm-vis/modelroom

 folder    not chosen yet          the first question asks where results live
 daemon    Ollama 0.34.2           reachable, 19 models installed
 machine   workstation             one graphics card, 12 GB, 128 GB memory

 Five steps: configuration, packages, context, measurement, results.
 Files are written as the run goes on. Esc leaves a list, Ctrl-C leaves at any point.
```

1 **Configuration**: where the results should live, and which machines the result covers --
this one is measured here. 2 **Packages**: a Hugging Face search for a model name, your
selection out of every hit (the ones that cannot be picked are shown with the reason), and the
fetch. 3 **Context**: how much text a model should handle at once, as a scale from XS to XXL
with the words that are, an example, and how many of the packages just fetched still fit this
machine -- that last column is the fit of the ranking itself, not a second calculation.
4 **Measurement**: the speed of the packages your local Ollama daemon already has; nothing is
marked, and it never downloads anything. 5 **Results**: the ranking per machine, written to
`docs/models.md` and `docs/models.json`.

Every question is a list with the arrow keys, including yes and no; `Esc` leaves a list and Ctrl-C
leaves at any point, and each step writes as it goes. The next run is the same command:
it remembers the folder, the machine it measured and the context you chose. Without an
interactive terminal it prints the help and stops; `modelroom --answers <file>` takes the
answers from a TOML file instead (for a self-test or CI) -- `context` there is a level (`"L"`)
or a number of tokens -- and `modelroom --config <file>` works on that configuration rather
than the remembered folder. See `CONTRACTS.md`, "Guided mode", for every question and its key.

The six subcommands never ask anything, and they are what the guided mode calls. From a
clone of this repository, after `uv sync --frozen`:

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
4. `render` reads the snapshot and every machine's hardware profile and writes two views: the
   Markdown one to `paths.markdown` (`docs/models.md` in the example) and the JSON one next
   to it under the same stem. It takes no `--machine`. A machine is ranked from the profile
   its `[machines.<name>].profile` names -- the guided mode writes that entry after it has
   measured the machine. It assumes the context the last guided run of that folder chose
   (`[guided].context`), so the ranking it writes is the one that run showed; a configuration no
   guided run has chosen a context in assumes 8192.
5. Read the table. Only active, complete packages with `metadata_ok` or `approved`
   provenance are shown, and every one of them is ranked per machine: best fit first, a
   measured package ahead of an unmeasured one of the same fit class, with the rule and the
   scenario printed in the header. Packages fit v1 cannot judge stand under "not covered"
   with the reason, ones that do not fit under "too tight". Every computed number says
   `(computed)`, the measured speed says `(measured)`.

`--config` is required by every command; there is no default path.

## Files

Under `paths.state`:

| File | Written by | Kind |
|---|---|---|
| `modelroom.json` | `fetch` | snapshot, versioned contract |
| `hardware/<profile_id>.json` | `hardware`, on that machine | hardware profile, versioned contract |
| `measurements/<profile_id>/<id>.json` | the load test | speed measurement, versioned contract |
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
| `3` | an unsupported `schema_version` in the configuration, the snapshot, a hardware profile, the answer file or the pointer file; or one of them is not valid JSON or does not match its model |
| `130` | the guided mode was stopped with Ctrl-C; nothing is left half written |

The guided mode ends with `1` when it wrote its document but a step did not finish -- a
measurement that failed, a `fetch` that did not complete, an import that did not go through.
Those lines are printed as they happen; the run goes on with the machines that are there.

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
