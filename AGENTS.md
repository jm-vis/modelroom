# AGENTS.md

Rules for every tool and every person working on this repository. Tool-specific files (if any
ever appear) import this one and stay thin; the truth lives here.

## Purpose

`modelroom` finds the local packages (GGUF and tensor builds) that exist for an allow-list of
model families on Hugging Face and in the Ollama registry, and computes whether each package
fits a measured machine. It is a command-line tool and a small library, published under MIT.
Status: alpha, first release 0.1.0; `fetch`, `hardware` and `render` work end to end, the data shapes
are versioned contracts (`CONTRACTS.md`), layout and fit rules may still change.

## Stack and language choice

Python (3.11 or newer), because the catalog logic and Pydantic contracts it integrates with are
Python and the run has to be platform neutral. Dependencies are kept to the standard library
plus `pydantic`; `llmfit` is an external tool called as a subprocess, never vendored. Everything
in this repository is English: identifiers, comments, commit messages, documentation.

Package manager is `uv`. `uv.lock` is committed and is the truth for dependency versions;
install with `uv sync --frozen`. To raise a dependency: `uv lock --upgrade-package <name>`,
run the tests, commit the lock.

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
- Exit codes are part of the contract: `0` success (`hardware`: measured and written, even
  when the local Ollama daemon could not be reached), `1` at least one `fetch` area incomplete
  (or `fetch` stopped at another process's lock, or this run is not newer than the stored
  snapshot -- both leave the state directory untouched), `2` the configuration is missing or
  invalid, the named machine is not a writer (`fetch`) or not configured at all (`hardware`),
  or a required external tool (`llmfit`) is missing or too old, `3` an input file has an
  unsupported schema version. Details: `CONTRACTS.md`.

## Hard boundaries

This repository is public. Nothing operator-specific belongs in it:

- No secrets, ever. Not in code, not in tests, not in fixtures, not in the history.
- No personal names, no e-mail addresses, no absolute paths from anyone's machine. The only
  organisation named is the maintainer, in `README.md`, `LICENSE` and `pyproject.toml`.
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
untrusted input. The tests therefore have to prove, with a deliberately broken input each:

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
