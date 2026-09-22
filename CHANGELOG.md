# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow Semver.

## [Unreleased]

### Added

- Project scaffold: package metadata, MIT license, tool-neutral rules (`AGENTS.md`), contract
  discipline (`CONTRACTS.md`), documentation map, public hygiene test and git hooks that run it,
  single-source version test.
- Contracts (AP1): the Pydantic models for family/base-model/package/snapshot identity
  (`modelroom/contracts.py`), the provenance decision rules (`modelroom/provenance.py`), and
  quantization parsing and package ordering (`modelroom/quantization.py`), each documented in
  `CONTRACTS.md` with a validated example and covered by fixture-backed tests.
- Configuration (AP2): the `Configuration` model tree and the `modelroom.toml` reader
  (`modelroom/config.py`) -- `BaseModelConfig`, `FamilyConfig`, `MachineConfig`, `PathsConfig`,
  `LlmfitConfig`, cross-checked family/owner rules, and `load_config`/`Configuration.from_dict`
  with `schema_version` checked before field validation, same convention as `load_snapshot`.
  Shares its `hf_repo`/`repo_aliases`/`ollama_base`+`ollama_tag` validators with
  `BaseModelSpec` via three functions extracted from `modelroom/contracts.py`. Ships
  `modelroom.example.toml` at the repo root; documented in `CONTRACTS.md` under a new
  "Configuration" section and covered by `tests/test_config.py`.
- Fetch (AP3): the `modelroom fetch --config ... --machine ...` command
  (`modelroom/cli.py`, registered as the `modelroom` console script), a minimal stdlib HTTP
  transport with a per-run request budget (`modelroom/http.py`), Hugging Face and Ollama
  fetchers (`modelroom/hf.py`, `modelroom/ollama.py`) that assemble `Package`s from a
  packager's file tree or an Ollama library's manifests, and the run state that ties a fetch
  run together (`modelroom/state.py`): a lock file, the snapshot merge/version rules, atomic
  writes, and `run-status.json`. Every fetcher takes its `Transport` as a parameter --
  dependency injection throughout, fixture-backed and fully offline in tests (`tests/test_hf.py`,
  `tests/test_ollama.py`, `tests/test_state.py`, `tests/test_fetch.py`, `tests/test_cli.py`,
  `tests/test_http.py`). New fixtures and one measured deviation (Hugging Face returns `401`,
  not `404`, for a repo an anonymous caller cannot see) are documented in
  `tests/fixtures/README.md`; the runtime state files, area/merge/version semantics and the
  `parameters_b` fallback are documented in `CONTRACTS.md` under "Fetch runtime state (AP3)".

### Fixed

- AP3 acceptance fixes against a live snapshot (13 base models, 79 areas, 358 packages,
  2026-09-22): the packager naming conventions in `modelroom/provenance.py` and
  `modelroom/quantization.py` now recognize per-quant subfolders (unsloth), the mradermacher
  dot separator (`<base>.<QUANT>.gguf`, including its own lowercase `f16`), Qwen's
  `-split-NNNNN-of-NNNNN` shard suffix, and a non-weight marker (`mmproj`/`imatrix`) anywhere
  in a basename rather than only as a prefix; `QUANT_ORDER` gained 22 previously-unparsed
  tokens (the ternary/1-bit family, the missing `IQ`/`Q2_K`/`Q3_K` members, unsloth's
  `UD-IQ4_XS`/`UD-IQ4_NL`, `MXFP4`); the
  Ollama size-token provenance rule is now a 15 % tolerance around measured `parameters_b`
  instead of exact equality (`safetensors.total` counts embeddings, so it can never equal a
  packager's rounded tag); `hf.py`'s duplicate shard-suffix regex was removed in favor of
  `quantization.py`'s single definition; and `modelroom.cli.fetch_with_config` is now a public
  function `_cmd_fetch` delegates to, for a programmatic caller that already holds a
  `Configuration`. Documented in `CONTRACTS.md` under "Ollama size-token tolerance" and
  "Package file-stem naming conventions".
