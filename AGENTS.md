# AGENTS.md

Rules for every tool and every person working on this repository. Tool-specific files (if any
ever appear) import this one and stay thin; the truth lives here.

## Purpose

`modelroom` finds the local packages (GGUF and tensor builds) that exist for an allow-list of
model families on Hugging Face and in the Ollama registry, and computes whether each package
fits a measured machine. It is a command-line tool and a small library, published under MIT.
Status: alpha, first release 0.1.0; the data shapes are versioned contracts (`CONTRACTS.md`),
layout and fit rules may still change.

The commands:

| Command | What it does |
|---|---|
| `modelroom` | the guided mode: asks, writes `modelroom.toml`, then runs the three commands below (`CONTRACTS.md`, "Guided mode") |
| `modelroom --answers <file>` | the same run with the dialog's answers from a TOML file, for a self-test or CI |
| `modelroom --config <file>` | the guided mode on that configuration, rather than the folder it last used |
| `modelroom hardware --config <file> [--machine <name>] [--cpu-only] [--new-identity]` | measure this machine and write its schema-2 profile |
| `modelroom fetch --config <file> --machine <name>` | fetch package metadata for every configured base model |
| `modelroom render --config <file>` | write the ranking per machine: Markdown, and the JSON view next to it |
| `modelroom migrate --config <file>` | move schema-1 profiles and configuration to schema 2 |
| `modelroom export-profile --config <file> [--profile <id>] --out <file>` | write one machine's profile and measurements to a file |
| `modelroom import-profile <file> --config <file>` | read such a file into this results folder |

Only the guided mode ever asks a question; every subcommand runs without a terminal.

## Stack and language choice

Python (3.11 or newer), because the catalog logic and Pydantic contracts it integrates with are
Python and the run has to be platform neutral. Dependencies are kept to the standard library,
`pydantic` and `questionary`; `llmfit` is an external tool called as a subprocess, never
vendored. Everything in this repository is English; the next section says which English, and
which words.

### Terminal dialog library

The guided mode's selection lists with check marks are `questionary` (MIT) on `prompt_toolkit`
(BSD), imported by `modelroom/dialog.py` and by nothing else -- every other module and all six
subcommands run without it. No dialog layer is written by hand.

The pin follows a gate, never convenience: the candidate has to render and read real keys in
every terminal a user of this package may sit at. Measured 2026-09-24 with `questionary` 2.1.1
on `prompt_toolkit` 3.0.53, PowerShell 5.1, key events written into the console input buffer
(`WriteConsoleInputW` on `CONIN$`, the same buffer a keyboard fills) and read back through
prompt_toolkit's own Win32 input: Windows console host (`conhost`) **pass**, Windows Terminal
**pass**. A Linux SSH session with a PTY is not part of that measurement and is open. Without a
passed gate there is no pin.

Package manager is `uv`. `uv.lock` is committed and is the truth for dependency versions;
install with `uv sync --frozen`. To raise a dependency: `uv lock --upgrade-package <name>`,
run the tests, commit the lock.

## Language standard

**US English**, everywhere: identifiers, messages, docstrings, comments, commit messages,
documentation. A British spelling is a finding, not a style choice -- write `color`,
`behavior`, `initialize`, `optimize`, `normalize`, `center`, `catalog`, `analyze`, `favor`,
`license` (noun and verb alike), `modeling`, `labeled`. No German ever reaches a user of the
package, so none of it is written here either.

One word per idea, so that a message, a field value and a paragraph of documentation say the
same thing in the same way:

| Write | For | Never |
|---|---|---|
| `latest`, `legacy` (with its `successor`), `unknown` | where a model stands in its family | `older`, `newer`, `outdated` |
| `measured`, `entered`, `computed` | where a value came from | "detected", "estimated", "real" |
| `publisher`, `listed packager`, `other` | who owns a repository | `untrusted`, `trustworthy`, or any other verdict on a packager; only the class is stated |
| `unknown` | a fact with no evidence behind it | an empty cell, `n/a`, or a plausible guess |
| `not comparable` | two measurements from different scenarios | "slower", "worse" |
| `display adapter only` | a GPU that is present but cannot carry a fit | "no GPU" |
| `none known` | no Ollama name is mapped to this package | "missing", "not available" |

An age statement is always positive evidence about one model (a successor named at the
publisher's repository, or the shipped catalog), never a comparison of two version numbers.

`tests/test_language_standard.py` is this section in executable form: it reads every module
under `modelroom/` and the prose files, and names file, line, word and replacement. Its
exception list carries a reason per entry and has to shrink, never grow quietly.

## Working rules

- Test first (red, green, refactor). Real dependencies or simple in-memory fakes; no mocking
  frameworks. Network-facing code is tested against recorded fixtures, never against the live
  API in the default test run.
- Conventional Commits in English: `feat(scope): ...`, `fix(scope): ...`, `docs: ...`,
  `chore: ...`, `test: ...`, `refactor: ...`.
- YAGNI. No placeholder code, no unimplemented TODOs, no dead code. A function does one thing.
- Explicit error handling. Every failure that a user can act on becomes a message and an exit
  code; nothing is swallowed.
- A cross-process lock is a kernel file lock held on a stable file (`msvcrt.locking` on Windows,
  `fcntl.flock` elsewhere, non-blocking), never a create-rename-delete choreography with an age
  rule: the kernel releases a crashed holder's lock, a live holder keeps it regardless of age, and
  the lock file is never renamed or deleted. Three review rounds showed that no file-name
  choreography closes every race; see the "Lock file" section of `CONTRACTS.md`.
- Exit codes are part of the contract: `0` success (`hardware`: measured and written, even when
  `llmfit` was not there to cross-check the readings), `1` at least one `fetch` area incomplete
  (or a command stopped at another process's lock, or this run is not newer than the stored
  snapshot -- both leave the state directory untouched -- or `hardware` wrote the profile but
  could not bind this machine to it), `2` the configuration is missing or
  invalid, the named machine is not a writer (`fetch`) or not configured at all (`hardware`),
  `hardware`'s bound profile belongs to another machine, or an external tool a command requires
  is missing or too old, `3` an input file has an unsupported schema version. The guided mode
  adds two of its own: `2` as well when there is no terminal and no `--answers`, when the input
  ends, or when an answer is missing or unusable, and `130` when the user presses Ctrl-C.
  Details: `CONTRACTS.md`.

## Hard boundaries

This repository is public. Nothing operator-specific belongs in it:

- No secrets, ever. Not in code, not in tests, not in fixtures, not in the history.
- No personal names, no e-mail addresses, no absolute paths from anyone's machine. The only
  organization named is the maintainer, in `README.md`, `LICENSE` and `pyproject.toml`.
- No references to an internal workspace, its folder layout, its personas or its chronicle.
  `tests/test_public_hygiene.py` enforces a negative list over every tracked file and runs
  from the git hooks in `.githooks/`.
- Configuration is data the user brings (`modelroom.toml` or a `Configuration` object). The
  package never guesses paths relative to the current working directory: paths in a
  configuration file resolve relative to that file, programmatic callers pass absolute paths.

## Contracts

Shared data shapes are defined once, as Pydantic models in `modelroom/contracts.py`
(`extra="forbid"`, value ranges, coupled fields validated), and described for readers in
`CONTRACTS.md` with the same examples. Contract tests keep the two in step. Every persisted
file (configuration, snapshot, hardware profile) carries a `schema_version`; readers declare
the range they accept and refuse anything else before they write. See `CONTRACTS.md`.

## Documentation chain

`docs/` holds the repository's own record, one folder per genre, created with its first real
entry and never as an empty shell:

```
idea -> decision -> execution -> measurement -> operation -> incident
docs/rfc/  docs/adr/  docs/plans/  docs/slo/  docs/runbooks/  docs/postmortems/
                                               docs/playbooks/
```

An ADR is immutable; a changed situation gets a new ADR that supersedes the old one. Each
genre folder carries a `README.md` index with one line per entry. `docs/README.md` is the map.

## Security, definition of done

Attack surface of this tool: outbound HTTPS to `huggingface.co`, `ollama.com` and `registry.ollama.ai`,
HTTP to a local Ollama daemon, a subprocess call to `llmfit`, and file writes under the
configured state directory. No user-facing web surface, no uploads, no HTML rendering of
whatever a registry returns. The tests therefore have to prove, with a deliberately broken
input each:

- a configuration that points the state directory outside its own folder tree is rejected
  before anything is written;
- responses from the two registries are validated against the contracts and an unexpected
  shape ends the area as incomplete instead of corrupting the snapshot;
- a snapshot or hardware file with an unsupported `schema_version` is refused before any
  write (exit `3`);
- a second command against the same state directory stops at the lock instead of waiting;
- the `llmfit` subprocess is called with a fixed argument list, never with user-controlled
  strings, and a missing or too old `llmfit` ends with exit `2` and an install hint.

## Versioning and releases

One version source: `[project] version` in `pyproject.toml`, Semver, starting at `0.1.0`.
`modelroom.__version__` reads it from package metadata; a test fails if a second version
literal appears anywhere. Releases are annotated tags `vX.Y.Z` on `main`. Release procedure:

1. Gate: `uv run --frozen python scripts/release-check.py` ends with `RELEASE-CHECK: OK <version>`.
2. Smoke: `uv run --frozen python scripts/release-smoke.py` ends with `RELEASE-SMOKE: OK`.
3. Tag: `git tag -a vX.Y.Z -m "modelroom X.Y.Z"`, then the gate again with `--tag vX.Y.Z`.
4. Review: an independent second-model review of the release diff, findings closed or answered.
5. Upload: `uv build` into an empty folder and `uv publish` by the maintainer, then push the tag.
