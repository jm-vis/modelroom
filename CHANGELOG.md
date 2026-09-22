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
