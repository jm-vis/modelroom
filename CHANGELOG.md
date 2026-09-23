# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow Semver.

## [Unreleased]

### Added

- Schema 2 contracts for the guided mode (CONTRACTS.md, "Schema 2: profiles, measurements,
  guided mode"): hardware profile v2 keyed by a random `profile_id` with a source for every
  memory value, a GPU state and an llmfit cross-check; scenario, measurement records as their
  own files and measurement protocol v1 (`protocol_v1.toml`); export object; pointer file and
  profile takeover rule; relation check; shipped catalog (`catalog.toml`); fit on profile v2;
  ranking rule; search hit, requirement and note shapes.
- `modelroom migrate --config <file>`: moves schema-1 hardware profiles and configuration to
  schema 2 under the lock, keeps `*.v1.bak` backups, and says `nothing to do` on a second run.
- `run_fetch` accepts a `RequestBudget` shared with the caller's own requests.

### Changed

- Configuration schema 2: `repos`, `[machines.<name>].profile`, `[defaults]`, `[updates]`,
  `[guided]`; `families` may be empty. Schema 1 still reads; the shipped example is schema 2.

## [0.1.0] - 2026-09-23

First release.

### Release

- `scripts/release-check.py`: the release gate. Checks a clean working tree, a valid PEP 440
  version without a local part with its own section in this file and, with `--tag`, an
  annotated tag `v<version>` on HEAD; runs the negative list of `tests/test_public_hygiene.py`
  (the same module, not a copy) over every file of HEAD, every line ever added in the history
  (binary files as text, renames as additions) and the raw commit and tag objects; builds wheel
  and sdist offline from HEAD and checks their content and file classes. Findings name place
  and class, never the text found. Commit metadata may carry noreply addresses and the
  maintainer's own address, taken from `git config user.email` or
  `MODELROOM_RELEASE_MAINTAINER_EMAIL` at run time and never written into the repository. Exit
  `0` free, `1` findings, `2` cannot check.
- `scripts/release-smoke.py`: the delivery smoke test. Installs the built wheel into a fresh venv
  outside the repository, derives a customer configuration from `modelroom.example.toml`, runs
  `hardware`, `fetch` with the network blocked through an unreachable proxy (every area
  incomplete, exit `1`, packages kept, lock free) and `render` of a snapshot built from the
  recorded fixtures.
- README: installation from the package index and the two release commands; `AGENTS.md`: the
  release procedure. Neither script ships in the wheel or the sdist.

### Documentation

- README brought to the current state: status alpha, requirements, a five-command quick start
  with the real flags (the planned `check` verb is gone), state files, the meaning and limits of
  the fit class, exit codes and network access, each taken from `CONTRACTS.md`/`AGENTS.md`.

### Added

- Render (AP5): the `modelroom render --config ...` command (`modelroom/cli.py`,
  `cli.render_with_config`, same pattern as `fetch_with_config`/`hardware_with_config`) and the
  pure Markdown document builder it wraps (`modelroom/render.py::build_document`). A pure
  reader of the current snapshot and every configured machine's hardware profile: eligibility
  (`metadata_ok`/`approved` provenance, complete, active), the per-group-per-machine selection
  rule (largest good-or-better quant, else the smallest, one row per distinct package picked by
  any machine), the fit/installed/speed cell rules and a new minimal `Rating` contract
  (`modelroom/contracts.py`) for an optional market-index Stars column -- any exception from a
  `RatingSource` (including the new `render.RatingUnavailableError`) is treated the same way: a
  `Market rating unavailable: <message>` note near the top and every Stars cell `–`, still exit
  `0`. Runs under the same kernel lock as `fetch` and the same F13 schema-version-before-lock
  ordering; refuses to overwrite a rendered document that is already newer than the snapshot
  being rendered (exit `1`, file untouched). Writes atomically via the new
  `state.py::atomic_write_text` (`.pid.tmp` + `os.replace`, alongside `atomic_write_json`);
  both writers now pin LF line endings on every platform (measured 2026-09-22: the snapshot
  and the rendered document came out CRLF on Windows).
  Documented in `CONTRACTS.md` ("Render (AP5)", the `Rating` model, and the updated Exit codes
  table). Covered by `tests/test_render.py` (the pure builder: selection, fit/installed/speed
  cells, stars/rating, header round-trip) and `tests/test_cli.py` (the `render` verb end to
  end: the lock, every schema-version/no-snapshot/newer-document exit path, and
  `render_with_config`).
- Hardware (AP4): the `modelroom hardware --config ... --machine ...` command
  (`modelroom/cli.py`), the persisted per-machine hardware profile
  (`<state>/hardware/<machine>.json`, `HardwareSnapshot`/`InstalledModel`/`Measurement` in
  `modelroom/contracts.py`), a dependency-injected `llmfit` subprocess binding with a version
  gate (`modelroom/llmfit.py`), the local Ollama daemon inventory
  (`modelroom/ollama_local.py::fetch_installed_models`, `GET /api/tags`, never a command
  failure when the daemon is unreachable), and the pure fit computation contract v1
  (`modelroom/fit.py::compute_fit` -- weights/KV-cache sizing, GPU/RAM pool selection, the
  perfect/good/marginal/too_tight thresholds, and the CPU cap). `hardware` only requires the
  named machine to be configured, not a `writer`, and takes no lock (each machine writes only
  its own file). New fixtures (`llmfit_system_laptop.json`, `llmfit_version.txt`,
  `ollama_tags_local.json`, all recorded 2026-09-22) are documented in
  `tests/fixtures/README.md`; the models, the llmfit/Ollama field mappings and "Fit contract
  v1" (formula, thresholds, the exact "fit (computed, v1)" wording a renderer must use) are
  documented in `CONTRACTS.md`. Covered by `tests/test_contracts.py` (new models),
  `tests/test_llmfit.py`, `tests/test_ollama_local.py`, `tests/test_fit.py`,
  `tests/test_state.py` (hardware snapshot path/load/write) and `tests/test_cli.py` (the
  `hardware` verb end to end, including every exit-2/exit-3 path).
- The validated `EXAMPLES` of every contract model moved from `modelroom/contracts.py` to
  `modelroom/examples.py` (data next to the tests that use it; the contract module keeps the
  models and rules, and stays under the repository's file-size guard).

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

- Fix-round 1, thirteen confirmed review findings against AP3+AP4:
  - **F1** `modelroom/state.py::acquire_lock` creates the lock file with
    `os.open(O_CREAT | O_EXCL)` instead of a `path.exists()`-then-`write_text` race, verifies a
    stale-lock takeover against a random `token` it re-reads after writing, and returns that
    `token`; `release_lock(path, token)` deletes only when the file still carries it.
    `modelroom/cli.py` adapted to pass the token through.
  - **F2** `modelroom/state.py::merge_snapshot` drops a package whose area is no longer among
    this run's `area_outcomes`, instead of keeping every old package forever regardless of
    whether its base model or area is still configured.
  - **F3** `modelroom/hf.py::fetch_hf_area` and `modelroom/ollama.py::fetch_ollama_area` run
    package assembly inside the same error handling as their tree/manifest fetch and validate
    every entry/layer's shape, so a malformed registry response (e.g. a tree entry or manifest
    layer that is not an object) ends only that area `incomplete`, never raises out of
    `run_fetch`.
  - **F4** a `weights`/`weights_shard` file/layer with no non-negative integer size is now a
    shape error (F3) rather than a silent `0`; `modelroom/fit.py::compute_fit` additionally
    returns `fit_class="unknown"`, `reason="a weight file has no size"` for any `Package` that
    still reaches it with a 0-byte weight file.
  - **F5** `modelroom/ollama.py::_fetch_manifest` computes the manifest digest as `sha256:` +
    sha256 of the `GET` body and makes no `HEAD` request at all any more (measured 2026-09-22:
    the body hash equals the registry's old `HEAD` header exactly for the recorded `9b`
    fixture).
  - **F6** `modelroom/fetch.py::run_fetch` builds a `previous_by_key` map from the old
    snapshot's packages and passes it into both fetchers, so an `Approval` bound to still-current
    content survives a fetch instead of being silently discarded every run.
  - **F7** `modelroom/config.py::load_config` rejects a `paths.state` that resolves (symlinks
    followed) outside the config file's own directory tree; `paths.markdown` is not confined.
  - **F8** `load_config` resolves its own `path` argument to an absolute path first, so a
    relative `--config` no longer fails to make `paths.state`/`paths.markdown` absolute.
  - **F9** `modelroom/fetch.py::run_fetch` stops fetching entirely once the request budget is
    exhausted; a base model this run never even started reading keeps its previous
    `BaseModelSpec` (or an unknown one with no previous snapshot) instead of being overwritten
    with a fresh, budget-starved "unknown" reading, and every area it would have needed is
    recorded incomplete with `"budget exhausted before this area was started"`.
  - **F10** `modelroom/http.py::Response` gained `requests_made` (default `1`);
    `UrllibTransport` disables `urllib`'s own uncounted redirect following, follows up to 5
    `GET`/`HEAD` hops itself, and reports the hop count; `BudgetedTransport` charges every hop
    against the run's budget.
  - **F11** `BaseModelSpec.parameters_b` is now `float | None` (`None` = not measured this run
    and no previous reading exists); the `1.0` placeholder is gone.
    `modelroom/provenance.py::_decide_ollama` resolves `("unresolved", "parameters_unknown")`
    when it is `None`, before the size-token tolerance check ever runs. A snapshot written
    before this change may still carry the placeholder `1.0` for a base model whose parameter
    count Hugging Face does not expose; the carry-over rule would keep it. Rebuild such a
    snapshot once (delete `modelroom.json`, run `fetch`) -- there is no automatic migration.
  - **F12** `modelroom/llmfit.py::check_llmfit_version`/`fetch_llmfit_system` catch
    `subprocess.TimeoutExpired`/`OSError` from the runner and a non-zero `--version` exit code
    as `LlmfitError`; `hardware_fields_from_llmfit_system` now validates that
    `system.total_ram_gb` is a positive number and `gpu_vram_gb`/`available_ram_gb` are
    non-negative numbers when present, instead of trusting the shape and crashing downstream.
    `FixtureRunner` may map an `args` tuple to an exception instance to raise.
  - **F13** `modelroom/cli.py::fetch_with_config` checks the existing snapshot's
    `schema_version` before calling `acquire_lock`, so an unsupported schema version exits `3`
    without ever creating (or disturbing) a lock file.

  Documented in `CONTRACTS.md` under "Lock file", "Merge rule", "Area semantics", "Request
  budget", "Approval carry-forward across fetch runs", `PathsConfig`, "Path contract",
  "`parameters_b` when Hugging Face has no answer this run" and "Ollama size-token tolerance".
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
- Fix-round 2, seven confirmed review findings against AP3+AP4:
  - **R1** `modelroom/state.py::acquire_lock`'s stale-lock takeover claims the existing file
    first with an atomic `os.replace(path, <path>.stale.<token>)` rename (only one of two racing
    processes can ever win it) instead of write-then-reread, which had a real window letting two
    processes each see their own token (measured directly on Windows as an intermittent
    `PermissionError`); capped at three attempts before raising `LockHeldError`. `release_lock`
    is symmetrically a claim-then-decide (`os.replace(path, <path>.release.<token>)`) instead of
    a check-then-unlink, which had the same kind of window.
  - **R2** `modelroom/hf.py::fetch_base_model_meta` now accepts a model-info `sha` only when it
    is a real 40-hex commit sha and `safetensors.total` only when it is a real number greater
    than zero, and wraps the architecture build in `try`/`except` -- a malformed `sha` or a
    `safetensors.total` of `0` could previously raise a pydantic `ValidationError` straight out
    of `run_fetch`, contradicting this function's own "never raises" docstring promise.
  - **R3** `modelroom/hf.py`/`modelroom/ollama.py::_carry_forward_approval` now also require
    `previous.base_model_hf_repo == stub.base_model_hf_repo` -- `package_identity_key` alone
    says nothing about which base model a package belongs to, so an approval could otherwise
    follow the bare `(repo, filename)`/`ollama_name` identity to an unrelated base model that
    happens to share a packager repo or tag.
  - **R4** `modelroom/fetch.py::run_fetch` checks `budgeted.remaining <= 0` before each base
    model *and* before each of its areas, not only after a base model finishes -- a budget
    already exhausted at the start of the run (or exhausted between two areas of the same base
    model) previously still let `fetch_base_model_meta`/a fetcher be called once more, silently
    resolving into a fresh budget-starved reading or a bare `"budget exhausted"` area error
    instead of `"budget exhausted before this area was started"`.
  - **R5** Inverts F10's redirect-following composition: `modelroom/http.py::UrllibTransport`
    makes exactly one request per call and returns a 3xx like any other status;
    `RedirectingTransport(inner)` now follows a chain by calling `inner` once per hop (up to
    `MAX_REDIRECTS = 5`), and `run_fetch` composes `RedirectingTransport(BudgetedTransport(...))`
    so every hop is checked and booked against the run's request budget *before* it is made,
    failure paths included, rather than `BudgetedTransport` charging the extra hops only after a
    chain already resolved. `Response.requests_made` and `BudgetedTransport`'s post-call
    accounting are removed.
  - **R6** No code change: a snapshot written before F11 landed may still carry the `1.0`
    placeholder for a base model Hugging Face reports no parameter count for; documented as a
    one-time manual rebuild (delete `modelroom.json`, run `fetch`), not an automatic migration,
    since the package is unreleased and the only existing snapshot is the operator's own.
  - **R7** `modelroom/llmfit.py::hardware_fields_from_llmfit_system` now rejects a non-finite
    number (`math.isfinite`, catching `NaN`/infinity that a naive `<= 0` check lets slip
    through), a `gpu_name`/`backend` that is not a string or `None`, and a `unified_memory` that
    is not a real `bool` (previously silently coerced by `bool(...)`, e.g. `bool("no")` is
    `True`). Second line of defense: `modelroom/cli.py::hardware_with_config` catches a pydantic
    `ValidationError` from `write_hardware_snapshot` and maps it to exit `2`, so an llmfit output
    shape neither validator has anticipated still exits cleanly instead of crashing.

  Documented in `CONTRACTS.md` under "Lock file", "Approval carry-forward across fetch runs",
  "Request budget" and "`parameters_b` when Hugging Face has no answer this run".
- Fix-round 3, four confirmed review findings against AP3+AP4:
  - **1+2** `modelroom/state.py`'s lock is now a kernel lock on a stable file, not R1's
    rename-based stale-lock takeover: that design could not be made exclusive against a third
    process (`os.replace` is not bound to the file generation a process actually read, so process
    A claiming a stale lock, creating a fresh one and returning could be followed by process B
    renaming *A's* fresh lock away and creating its own, with neither `os.replace` call ever
    failing), and giving a foreign lock back on release could hand the file back under a third
    process's now-current lock instead of the one it was actually taken from. `acquire_lock`
    now opens the file with `O_RDWR | O_CREAT` (never `O_EXCL`, never renamed, never deleted) and
    takes an exclusive non-blocking kernel lock (`msvcrt.locking`/`fcntl.flock`) on one reserved
    byte; the JSON content (no `token` field any more) starts one byte later, so it stays
    readable by another process the whole time the lock is held. `LockHeldError` no longer
    carries an age threshold -- a crashed holder's lock is released by the kernel on process
    exit, a live holder keeps it regardless of age -- so `LOCK_MAX_AGE`/`MAX_TAKEOVER_ATTEMPTS`
    and the stale-takeover machinery are gone. `acquire_lock` returns a `LockHandle`
    (`path`, `fd`) instead of a token string; `release_lock(handle)` empties the file (never
    deletes it) and is idempotent. `modelroom/cli.py::fetch_with_config` adapted to the new
    signatures. The three-process race test holds the winner's lock until the test releases it
    and demands exactly one winner per round, no retry (a first draft let the winner exit right
    after acquiring, which releases the lock and let a slower sibling win legitimately -- a test
    flaw that briefly looked like a lock-placement problem).
  - **3** `modelroom/hf.py::_valid_parameters_b` now converts `safetensors.total` inside a
    `try`/`except (OverflowError, ValueError)` and requires `math.isfinite(result) and result >
    0` on the converted value, not just `total > 0` before dividing: a subnormal `total` like
    `1e-320` passed the old check but underflowed to `0.0` after `/ 1e9` (which
    `BaseModelSpec.parameters_b`'s `gt=0` would then reject), and a JSON integer far outside
    `float` range (a raw `10**400`, which `json.loads` parses without complaint) overflowed the
    division itself.
  - **4** `modelroom/llmfit.py::_is_finite_number` now catches `OverflowError` from
    `math.isfinite` itself, which raises for an `int` too large to convert to `float` (e.g. the
    same `10**400` shape, this time in `total_ram_gb`) -- treated as "not finite", the same
    verdict as `NaN`/infinity, instead of crashing `hardware_fields_from_llmfit_system`.

  Documented in `CONTRACTS.md` under "Lock file" (rewritten) and `_valid_parameters_b`'s own
  docstring in `modelroom/hf.py`.
- Fix-round 4, two handle-management findings against the new lock:
  - **P1** `release_lock` is idempotent through `LockHandle.released`, not by catching `EBADF`:
    a closed descriptor's number is reused by the next `os.open`, so a second release of a stale
    handle used to truncate, unlock and close whichever file had inherited that number.
  - **P2** `acquire_lock` unlocks and closes the fd when writing the holder content fails after
    the lock was won; `release_lock` closes the fd in `finally`. Previously such a failure left
    the kernel lock held with no handle to release it until the process exited.
- Fix-round 5, nine confirmed findings (two review passes) plus fourteen P3 items verified
  one by one against `e87a208`:
  - **P2-1** `modelroom/http.py::UrllibTransport` refuses to open any URL whose scheme is not
    `https` or whose host is not on an explicit allow-list (production default: exactly
    `huggingface.co`/`ollama.com`/`registry.ollama.ai` over HTTPS, `127.0.0.1` over HTTP for the
    local Ollama daemon), raising the new `TransportSecurityError` before ever opening a
    connection. Probe: `UrllibTransport()("GET", "file:///…/pyvenv.cfg")` previously returned
    178 bytes of a local file -- `build_opener(_NoAutoRedirect)` still carried urllib's default
    `FileHandler`/`FTPHandler`/`DataHandler` alongside it, since `build_opener` auto-fills every
    default handler class not represented (directly or by subclass) among its arguments. The
    opener is now built by hand (`OpenerDirector` + `add_handler`, no auto-fill) from exactly
    `HTTPHandler`/`HTTPSHandler`/`_NoAutoRedirect`/`HTTPErrorProcessor`/`HTTPDefaultErrorHandler`,
    so `file://`/`ftp://`/`data:` are structurally impossible even if the allow-list check were
    ever bypassed. Every hop of a redirect chain re-enters the same check (both
    `RedirectingTransport` and `hf.py::_fetch_tree` call back into the same transport instance
    per hop), so a poisoned `Location`/`Link: rel="next"` target is refused on the hop that would
    have followed it, with no second request ever made.
  - **P2-2** A real loopback `ThreadingHTTPServer` in `tests/test_http.py` proves
    `_NoAutoRedirect` actually stops `urllib.request`'s own redirect following (`/a` answers 302,
    `UrllibTransport` returns it raw with exactly one request logged), and doubles as the test
    bed for P2-1's redirect-refusal cases -- nothing in `test_http.py` exercised the real
    transport against an actual HTTP response before this.
  - **P2-3** Every persisted datetime is now checked aware-UTC (`contracts.check_aware_utc`,
    already used by AP4's own fields): `Package.observed_at`/`last_seen`, `Area.last_success`
    and `Snapshot.run_at` gained the same validator `InstalledModel.observed_at`/
    `Measurement.*`/`HardwareSnapshot.measured_at` already had. Probe: a naive `Snapshot.run_at`
    validated without complaint, then `state.check_run_is_newer` raised `TypeError: can't compare
    offset-naive and offset-aware datetimes` out of `cli.main`. `cli.py` also gained
    `_read_snapshot`/`_read_hardware_snapshot`, wrapping every load site (`fetch`/`hardware`/
    `render`) so `json.JSONDecodeError` and pydantic `ValidationError` map to exit `3` with the
    file name in the message, the same as an unsupported `schema_version` -- a truncated file or
    one missing required fields previously crashed `main()` uncaught.
  - **F4** `render.parse_header_line` returns `None` (treated as "no header", the document is
    replaced) for a header whose `snapshot_run_at`/`rendered_at` is not an aware UTC datetime,
    not just for one that fails to parse at all -- a naive header timestamp used to compare
    against the new, always-aware snapshot's `run_at` and raise `TypeError`.
    `cli.py::_refusal_against_existing_document` also now treats an existing document that is
    not valid UTF-8 as no header (was: `UnicodeDecodeError` straight out of `render`).
  - **F5** (addendum, same rules as F1-F4) `render._select_for_machine` no longer falls back to
    "the smallest package" when fit v1 cannot judge *any* package of a group on a machine (every
    `fit_class` is `"unknown"` there, or the machine has no hardware profile) -- that machine
    makes no pick at all. Real case (Qwen3.5, measured 2026-09-22): the whole architecture is not
    covered by v1, so every package came back unknown, and the old fallback still picked an
    arbitrary smallest-quant package (e.g. `UD-IQ2_XXS`) with no basis at all. When *no* machine
    picks anything for a group, it renders one row instead of the ordinary per-package rows:
    every package cell `–` except each machine's fit cell (`no recommendation: <reason>`) and
    Variants (`<N> variants, none judged`). The "else the smallest" fallback still applies, but
    only across packages fit v1 *did* judge, when at least one was judged and none reached
    good/perfect.
  - **F6** (second addendum) `modelroom/ollama.py::_validated_layers` now raises when a manifest
    has no `'layers'` key at all (or an explicit `null`), instead of silently treating it as `[]`
    -- that used to build an empty/unresolved stub package under the *same* identity key as a
    previously valid package for the tag, which `merge_snapshot` would then silently overwrite
    the good package with.
  - **F7** `modelroom/hf.py::_files_from_tree`'s `if not name: continue` used to drop a
    `type: "file"` entry with a missing, empty or non-string `path` *before* the `isinstance`
    shape check ever ran; a tree entirely made of such entries looked like a genuinely empty,
    `complete` area, and the merge rule deactivated every one of the old area's packages. Now a
    shape error like any other (F3/F4), ending the area `incomplete` with the old packages left
    untouched.
  - **F8** `hf.py::_tree_entry_size`/`ollama.py::_weight_layer_size` reject a `'size'` at or
    above `2**63` -- `json.loads` parses a JSON integer literal of any width (e.g. `10**400`)
    without complaint, and `fit.py::compute_fit`'s `sum(...) / GIB` later raised `OverflowError`
    out of `render`/`fit` instead of the fetcher ending the area `incomplete`.
  - **F9** `modelroom/llmfit.py::SubprocessRunner`: (a) decodes with a fixed
    `encoding="utf-8", errors="replace"` instead of `text=True` alone, which decodes with the
    Windows system codepage, not UTF-8 -- a non-ASCII `gpu_name` in real `llmfit` output could
    raise `UnicodeDecodeError` uncaught by anything in this module; (b) resolves the bare name
    `"llmfit"` to an absolute path via `which` (`shutil.which` by default, injectable) *before*
    calling `subprocess.run`, rather than passing the bare name straight through -- an
    unqualified name's search order can check the current working directory before `PATH`.
    `check_llmfit_version`/`fetch_llmfit_system` are unchanged (still build `["llmfit", ...]`);
    only `SubprocessRunner`, the one runner never faked in a test, resolves it, so every existing
    `FixtureRunner`-based test is unaffected.
  - **P3-1** CONTRACTS.md's "Run status" section said `run-status.json` is written "at the end of
    every fetch run, including one that ends exit 1" without qualification; the exit-`1` cases
    that stop at the lock or the staleness check write nothing at all (`state.py::
    check_run_is_newer`'s own docstring: "before anything is fetched or written"). Reworded to
    say so explicitly -- no code change, the Exit-codes table (line ~66) was already accurate.
  - **P3-2** CONTRACTS.md's "Area semantics" described a base model with neither `ollama_base`
    nor `ollama_tag` configured as "a trivially complete, empty area" without qualification;
    `run_fetch` only ever calls `fetch_ollama_area` when *both* are set, so that area never
    actually appears in a merged `Snapshot` -- the early return is `fetch_ollama_area`'s own
    defensive default for a direct caller (already unit-tested), not something this section's
    "one area per..." rule describes as `run_fetch`'s output. Reworded; no code change.
  - **P3-3** `hf.py::_fetch_tree`'s `Link: rel="next"` pagination is capped at
    `_MAX_TREE_PAGES = 20` hops; a chain that has not terminated by then raises instead of
    fetching forever, and the page that would exceed the cap is never requested.
  - **P3-4** A packager repo's model-info `sha` is validated as a real 40-hex commit sha
    *before* it is used to build the tree-fetch URL (previously only validated once assembled
    into a `Package.revision`, after the unchecked URL had already been requested); an Ollama
    tag parsed from the (network-controlled) tags page is validated against the same charset
    `Package.ollama_name`'s tag half requires before it is used to build the manifest URL.
  - **P3-5** Verified, not changed: `llmfit.py::check_llmfit_version` already catches
    `FileNotFoundError` from a missing `llmfit` binary and raises `LlmfitError` with an install
    hint (`cli.py` maps it to exit `2`), already covered by `tests/test_llmfit.py`. F9's new
    `SubprocessRunner` tests add end-to-end coverage of the same path through the real runner.
  - **P3-6** (independently raised as P2 by the second review pass, treated as mandatory)
    `config.py::Configuration` gained `_check_repo_names_do_not_collide_across_base_models`: two
    base models sharing a packager repo name (the default `<name>-GGUF` or a `repo_aliases`
    entry) is now a configuration error at load time. Package identity
    (`quantization.package_identity_key`) never carries `base_model_hf_repo`
    (CONTRACTS.md, "Identity"), so two base models fetched into the same repo name would
    otherwise write into the *same* `state.merge_snapshot` dict entry, one silently overwriting
    the other with no error.
  - **P3-7** Two tests that did not test their names: `test_atomic_write_json_leaves_no_tmp_file_
    on_a_serialization_failure` renamed (`json.dumps` fails before any file is ever touched, so
    it never actually exercised the except-clause's cleanup) and a new
    `test_atomic_write_json_leaves_no_tmp_file_when_replace_fails` added that does (a real
    `os.replace` `PermissionError` when the destination is a directory);
    `test_incomplete_package_is_unknown_fit` now asserts the exact `fit.reason` instead of only
    `is not None`.
  - **P3-8** Verified, not removed: `cli.py::hardware_with_config`'s R7 `except ValidationError`
    around `write_hardware_snapshot` is still reachable after P2-3 -- a naive `now` (a legitimate
    input for a programmatic caller, even though the real CLI always passes `now=None`) makes
    `measured_at` naive, which `HardwareSnapshot.measured_at`'s aware-UTC check then rejects.
    Newly tested; was previously reachable but untested too.
  - **P3-9** Decided, not fixed: `state.py`'s atomic writers guarantee atomicity (a reader never
    sees a torn write) but not durability (no `os.fsync` before `os.replace`) -- every file this
    writes is a reconstructible cache of the registries/local daemon, recovered by re-running the
    command, never by restoring a backup. Documented in CONTRACTS.md, "Atomic writes: atomicity,
    not durability".
  - **P3-10** Tightened, not rejected: the three-real-process lock race test now has each child
    signal readiness *before* polling for "go", and the parent waits for all three signals before
    writing it -- closes the gap where a still-starting process might not even be in its polling
    loop yet when "go" appeared.
  - **P3-11** The stale-run CLI test now also asserts the lock file is left empty (`release_lock`
    always empties it, per CONTRACTS.md, "Lock file") -- was previously left unchecked.
  - **P3-12** `hf.py`'s tree-fetch catch clause now distinguishes `BudgetExhaustedError` from
    every other exception, passing its message through bare -- it previously prefixed *every*
    exception there with `f"{repo}: "`, contradicting CONTRACTS.md's "Request budget" promise
    that the area's error is exactly `"budget exhausted"`. The three existing tests asserting
    this by substring were tightened to exact equality.
  - **P3-13** `test_fetch_with_config_defaults_to_the_real_transport_and_clock` renamed to
    `..._accepts_omitted_transport_and_now_without_raising` -- its own comment already conceded
    a non-writer machine short-circuits before either default is ever constructed, so it never
    proved what its old name claimed (and, per AGENTS.md, never safely could: this suite must
    never let a real `UrllibTransport()` touch the live network).
  - **P3-14** `fetch_types.py` gained `error_text(exc)`, a `str(exc)` wrapper that falls back to
    naming the exception's type when that string is empty -- a message-less exception from a
    caller-supplied `Transport` (`raise SomeError()` with no args) used to become an empty
    `AreaOutcome.error`, which `Area`'s own validator rejects (`error` required when `status`
    is `"incomplete"`), raising a pydantic `ValidationError` out of `state.merge_snapshot`
    instead of the ordinary `incomplete` result every other failure produces. Applied at every
    `str(exc)` capture in `hf.py`/`ollama.py` that becomes an `Area.error`.

  Documented in `CONTRACTS.md` under "Exit codes", `check_aware_utc`'s own docstring, "Render
  (AP5)" (header/no-recommendation-row/fit-cell paragraphs), "Area semantics", "Request budget"
  and a new "Atomic writes: atomicity, not durability" subsection.
- Fix-round 6, thirteen findings against `0920f5a` (Codex round 7, verified one by one):
  - **R7-1** `config.py::PathsConfig` gained a model validator: `paths.markdown` must not equal
    and must not lie inside `paths.state`'s resolved directory tree -- probe: `paths.markdown`
    set to the state directory's own snapshot or lock file validated fine, which would have let a
    render silently corrupt state a `fetch`/`hardware` run depends on. `load_config` additionally
    rejects `paths.markdown` equal to the config file itself.
  - **R7-2** `llmfit.py::SubprocessRunner`'s default `which` is now `_default_which`, an explicit
    `PATH` walk, replacing `shutil.which` -- on Python 3.11 on Windows, `shutil.which` prepends
    the current working directory to the search list regardless of the `path` argument
    (`_win_path_needs_curdir` was only added in 3.12), reopening exactly the same-named-CWD-shim
    hole F9b (fix-round 5) was written to close. Only absolute `PATH` directories are searched (a
    relative entry is dropped, not resolved against the current directory); Windows tries each
    `PATHEXT` extension plus the bare name per directory.
  - **R7-3** A stored snapshot or hardware file whose JSON root is not an object (`[]`, `null`, a
    bare string) used to raise an uncaught `AttributeError` out of `contracts.load_snapshot`/
    `load_hardware_snapshot` (`data.get("schema_version")` assumes a `dict`); `state.py` gained
    `StateFileShapeError(ValueError)`, raised by `load_existing_snapshot`/
    `load_existing_hardware_snapshot` before either loader runs. `cli.py`'s
    `_read_snapshot`/`_read_hardware_snapshot` now also catch it and `UnicodeDecodeError` (a file
    that is not valid UTF-8 at all), mapping both to exit `3` with the file path, exactly like
    `JSONDecodeError`/`ValidationError` already did.
  - **R7-4** `http.py`'s local allow-list is now an origin, `(host, port)`
    (`_ALLOWED_HTTP_ORIGINS = {("127.0.0.1", 11434)}`), not a bare host -- probe:
    `http://127.0.0.1:59999/x` was accepted with the production defaults, so a poisoned
    `Location`/`Link` header could reach any local port, not only the real Ollama daemon. A URL
    with no explicit port is refused rather than matching "any port".
    `UrllibTransport`'s constructor parameter is renamed `allowed_http_origins` to match.
  - **R7-5** `http.py::_build_opener` registers `urllib.request.ProxyHandler()` again -- the
    explicit handler list `_build_opener` replaced `build_opener(_NoAutoRedirect)` with (P2-1,
    fix-round 5) dropped the `ProxyHandler` that `build_opener`'s auto-fill used to add for free,
    silently losing `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` support. The allow-list check still runs
    against the request's own target URL before the opener ever consults a proxy, so a proxy can
    change how an already-allowed request is reached but never widen what is reachable.
  - **R7-6** (decided, no code) Two configurations with different `paths.state` but the same
    `paths.markdown` are unsupported by contract and not guarded by a second lock -- documented in
    CONTRACTS.md, "Render (AP5)": one state directory owns exactly one markdown document, the
    newer-document refusal protects only against an older snapshot under the *same* state
    directory's lock.
  - **R7-7** `config.py::Configuration` gained a model validator: two base models sharing an
    `ollama_base` may not claim equal or prefix-overlapping `ollama_tag`s -- probe:
    `ollama_base="nova"` with `ollama_tag="7b"` on two different base models validated fine, but
    `ollama._keep_relevant_tags` keeps `tag == ollama_tag or tag.startswith(ollama_tag + "-")`, so
    both would resolve packages under the identity `nova:7b`, and the later-processed area would
    silently overwrite the other's package (the same P3-6 hazard, for Ollama instead of Hugging
    Face packager names).
  - **R7-8** `config.py::_check_repo_names_do_not_collide_across_base_models` (P3-6, fix-round 5)
    now compares the full `owner/name` candidate, not the bare repo name -- probe: `acme/Nova` and
    `other/Nova` with `packagers = []` were wrongly rejected (`'Nova-GGUF' is claimed by both`),
    even though `hf.py::fetch_hf_area` would probe `acme/Nova-GGUF` and `other/Nova-GGUF`, never
    the same repo. A collision sharing a configured packager still fails, unchanged.
  - **R7-9** `hf.py::_files_from_tree`: a tree entry with no `type` field at all (`{}`) was
    silently treated as "not a file" (a directory) and skipped, exactly the "genuinely empty,
    complete area" hazard F7 (fix-round 5) closed for a bad `path` -- `type` must now be a
    non-empty string before it is compared to `"file"`. `ollama.py::_validated_layer`: a manifest
    layer with no `mediaType`/`digest` at all (`{}`) passed through unchanged, and `_build_package`
    built an empty `format="unknown", complete=False` stub under the tag's real identity, silently
    replacing a previously valid package -- both fields are now required, non-empty strings.
  - **R7-10** `contracts.py::Architecture`'s `num_hidden_layers`/`num_key_value_heads`/`head_dim`/
    `max_context` gained an upper bound, `le=2**31 - 1` -- probe: `num_hidden_layers = 10**400`
    validated fine, then `fit.py::compute_fit` raised `OverflowError: integer division result too
    large for a float`, aborting the whole render. `compute_fit` also now catches `OverflowError`
    directly around its arithmetic, returning `fit_class="unknown",
    reason="architecture values out of range"` -- a second line of defense against an unrelated,
    unbounded field (`Package.default_context`) combined with otherwise in-bound values.
  - **R7-11** `render.py` gained a shared `_cell(value)` helper -- escapes `|` and collapses any
    `\r\n`/`\n`/`\r` to a single space -- applied to every cell carrying external text: area
    error, base model repo, packager, quantization/format labels, hardware `gpu_name`/`backend`/
    `installed_unavailable_reason`, fit/no-recommendation reasons, and a `RatingSource`'s error
    message. Probe: an `Area.error` of `"boom | extra\nsecond line"` produced 4 physical lines and
    an extra column in the areas table instead of the one row it should have been.
  - **R7-12** `cli.py::render_with_config`'s pre-lock snapshot read now also decides "nothing to
    render, run fetch first" (exit `1`) *before* `acquire_lock` is ever called -- previously the
    pre-read result was discarded and the check only ran again inside `_render_locked`, after the
    lock (and therefore the state directory and the lock file, both `acquire_lock` side effects)
    had already been created for a render that had nothing to do.
  - **R7-13** CONTRACTS.md's "Run status" section still said `run-status.json` is written "at the
    end of every fetch run, including one that ends exit 1" -- P3-1 (fix-round 5) had already
    concluded this needed rewording, but the correction was never actually applied to this
    section. Reworded to say explicitly what the code (`cli.py::fetch_with_config`/`_run_locked`)
    already did: written at the end of every run that got past the lock and the stale-run check;
    not written when the lock is held or a newer run is detected. No code change.
- Fix-round 7, three remaining findings against `db1c729` (Codex round 8, verified one by one):
  - **R8-1** `hf.py::_files_from_tree` now accepts exactly the two `type` values the tree endpoint
    sends, `"file"` and `"directory"`; any other value is a shape error that ends the area
    `incomplete` -- probe: `[{"type": "garbage"}]` still returned `[]` after R7-9, so a tree of
    unknown types was the same genuinely-empty, complete area that deactivates every old package.
  - **R8-2** `tests/test_http.py` gained an autouse fixture that removes every `*_proxy`
    environment variable (any spelling) before each test -- `urllib` reads them
    case-insensitively and on Unix a lowercase `http_proxy`/`no_proxy` wins over the uppercase
    value a test sets, so a machine with a proxy configured would have routed the loopback tests
    through it or made the proxy tests pass for the wrong reason. Each proxy test sets exactly the
    variables it needs on the clean slate. Test-only.
  - **R8-3** The two HF repo-name collision tests in `tests/test_config.py` copied the first base
    model's `ollama_base`/`ollama_tag` onto the second, so the R7-7 Ollama collision validator
    would have raised the expected `ValidationError` even with the HF check broken; the copy now
    carries no Ollama mapping and the tests match the HF message (`packager repo`). Test-only.
- Fix-round 8, one remaining finding against `1a7d5b4` (Codex round 9, verified with a probe):
  - **R9-1** The `tests/test_http.py` autouse fixture now also sets `NO_PROXY="*"` and removes
    `REQUEST_METHOD` after clearing every `*_proxy` variable -- probe: with no proxy variable at
    all `urllib.request.getproxies()` is empty and falls back to the Windows registry / macOS
    system proxy on those platforms (`getproxies_environment() or getproxies_registry()`), while
    `NO_PROXY="*"` alone keeps the environment dict non-empty (`{'no': '*'}`) and bypasses every
    host; `REQUEST_METHOD` present makes urllib drop `HTTP_PROXY` (`{}` from `{'http': …}`),
    which would have turned the positive proxy test into a direct connection. The two proxy
    tests override or delete `NO_PROXY` themselves. A regression test pins the clean-slate
    mechanism. Test-only.
