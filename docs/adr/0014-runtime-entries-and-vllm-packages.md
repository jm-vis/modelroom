# 0014 -- A local-server runtime is one entry; vLLM's packages get a real schema step

- **Status:** accepted, decided 2026-09-29 (GF, AP9-SV2 build brief, the "bound prior decisions"
  and "questions for the GF" sections)
- **Supersedes:** nothing. Extends ADR 0008 (`local_servers.LOCAL_SERVERS`, AP9-SV1) and the
  precedent of ADR 0006/ADR 0010/ADR 0011 (a schema step versus an additive extension without
  one).

## Context

The GF decided (2026-09-29) that vLLM ships complete in 0.2.0: modelroom detects vLLM and its
safetensors packages, locally and on a server the laptop reaches over SSH, the same way it
already detects Ollama (AP9-SV1). The GF's own framing for the whole effort: standardize as much
as possible -- a runtime is a row in a fixed table, not a special case wired through the
Ollama-specific code paths a second time.

This sub-package (AP9-SV2-0, "Contract, Schemata, ADR") is the first of seven that build vLLM
support; it lays the contract every later one reads, and decides two things the GF was asked
about directly:

1. **Where the vLLM-specific facts belong in the shared models**, and whether that needs a new
   schema version or fits the additive-without-a-step pattern this project has used since ADR
   0006 (the fit-words rename) and ADR 0010/0011 (hybrid architectures, graphics adapters).
2. **What one local-server runtime *is*, as a shape**, so that Ollama (today hard-wired),
   vLLM (this effort) and LM Studio (AP9-LS, decided the same day to follow directly after) are
   three values of one idea rather than three copies of the same logic.

## Decision

### 1. A local-server runtime is one table entry, never a plug-in

`RUNTIMES` (AP9-SV2-L builds the table itself; this ADR fixes the shape it will have) is a
fixed Python tuple of `Runtime` entries, each one a dataclass bundling: its `key`
(`RuntimeKey`, a `Literal`) and display `name`; the `LocalServer` ADR 0008 already defines
(origin, transport, start fact, step-4 offer/measure, card facts); `detect` (serving / installed
/ absent / needs a key); `accepts` (which packages this runtime loads); `fit` (its own fit rule);
`command` (the line or reason shown under a row); `model_place` (where its models live on disk);
`on_disk` (its own "already there" check); and its own words (table prefix, key hint, install
question). **A new runtime is a new tuple entry and a new `RuntimeKey` value -- never a
registration call, a config flag or a class discovered at runtime.** `local_servers.py` becomes
a thin view over this table; `LocalServer` stays exactly what ADR 0008 defined, now one field of
`Runtime` rather than the whole of it.

This ADR only *reserves the field* the table will be read into (`HardwareProfile.runtimes`,
decision 3 below); building `RUNTIMES`, moving Ollama onto it unchanged and adding vLLM's entry
is AP9-SV2-L's and AP9-SV2-F's work, staged after this one (`wartet_auf` in the build brief).
Rejecting a dynamic mechanism here, before any code exists that could tempt one, is deliberate:
see "Alternatives and why not".

### 2. Snapshot schema steps to 2 -- a real version, not an additive extension

Three earlier decisions (ADR 0006, 0010, 0011) extended a schema *without* a version step,
because every new field had a default and an old reader's only failure mode was a validation
error on a field it had never seen -- acceptable there because the new shapes were optional
extras (an architecture kind, a graphics adapter list) an old reader could simply never produce
itself. vLLM's packages are different: a **Hugging Face `tensor` package may now be
`from_metadata`/`approved`** instead of permanently `unresolved`/`format` (SV2-P builds the
provenance rule that does this; this ADR only opens the possibility in the contract), and
`quantization.QUANT_ORDER` gains six labels vLLM's formats need (`W4A16`, `AWQ`, `GPTQ-4`,
`GPTQ-8`, `W8A8`, `FP8`). A 0.1.0 reader that met either new shape under schema 1 would fail
with an opaque field or enum error, not the named, `exit 3` schema refusal this project
otherwise guarantees (CONTRACTS.md, "Exit codes"). The GF's own framing at E8 was exactly this:
a clean refusal, not a silent surprise. So: `SNAPSHOT_SCHEMA_VERSION = 2`,
`SNAPSHOT_SCHEMA_RANGE = (1, 3)`. A stored schema-1 file still reads as 2, unchanged --
`load_snapshot`'s own normalization (the one entry point that does this; `Snapshot.model_validate`
alone still enforces the current version only, as documented since the first version of this
function). Nothing else about a schema-1 file's content needs converting: every new field
(`Package.quantization_source`) defaults to `None`, and the new `QUANT_ORDER` members are purely
additional entries a package either does or does not reference.

### 3. `Package.quantization_source`, the provenance exception, the new `QUANT_ORDER` labels

- `Package.quantization_source: Literal["filename", "config"] | None = None` records where
  `quantization` came from: `"filename"` for GGUF (a packager's own naming convention, unchanged
  logic), `"config"` for a Hugging Face tensor package's `config.json` (SV2-P's reader), `None`
  for Ollama (a tag, never a config) or wherever `quantization` itself is still unknown.
- The provenance invariant (`Package._check_provenance_invariants`) gains one exception: a
  `tensor`-format package whose `source` is `"huggingface"` may be `from_metadata`/`approved`.
  Every other non-`gguf` package -- `unknown` format, or a `tensor` package from `source ==
  "ollama"` (Ollama's own MLX tags, noted 2026-09-28) -- stays locked to
  `unresolved`/`format`, exactly as before this step. The lock stays a model invariant, not a
  convention a caller could forget.
- `QUANT_ORDER` gains `W4A16`, `AWQ`, `GPTQ-4` (between `Q4_K_M` and `UD-Q4_K_XL`) and `GPTQ-8`,
  `W8A8`, `FP8` (between `Q8_0` and `UD-Q8_K_XL`), placed by bits per weight -- the same
  yardstick the existing GGUF labels already sort by (external sources: `Qwen/Qwen3-8B-AWQ`,
  `Qwen/Qwen3-8B-FP8`, `RedHatAI/Qwen3-8B-FP8-dynamic`, `RedHatAI/Qwen3-8B-quantized.w4a16`,
  `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`, read from the Hugging Face API 2026-09-29, revisions
  named in the build brief). `BF16`/`F16` need no new label: a tensor repo with no
  `quantization_config` reads its `torch_dtype` (top level) or `dtype`, else `text_config.dtype`
  (Qwen3.5's own shape -- its `dtype` sits only under `text_config`, fixture
  `tests/fixtures/hf_qwen_qwen35_4b_config.json`), in that search order. `ranking._FULL_STEP`
  reads `QUANT_ORDER.index("F16")` rather than a literal position, so the six insertions ahead
  of it do not move it.

### 4. `HardwareProfile.runtimes`, additive, no schema step (the ADR 0011 precedent)

`runtimes: list[RuntimeKey]` (default `[]`, excluded from the serialized form when empty, the
exact rule ADR 0011 set for `adapters`) names the local-server runtimes a machine's last
measurement detected serving or installed. Unlike the snapshot's step, this needs no version
bump: every value is a member of a closed, additive `Literal`, the field has a safe empty
default, and an old reader meeting it would fail the same way `adapters` would have -- a field
error on a genuinely new shape a 0.1.0 profile could never itself have produced. This answers
GF question F6 (**a**): modelroom remembers a machine's runtimes in its profile, so an imported
profile or one read from the results folder's sharing point still shows vLLM packages even
without a live connection to that machine. `RuntimeKey = Literal["ollama", "vllm"]` today; a
third runtime (LM Studio, AP9-LS) is a fourth value of the same `Literal`, additive again.

## Consequences

- A 0.1.0 reader meets a schema-2 snapshot with the named refusal and exit `3`, not a field
  error -- the GF's own ask at E8.
- **Shared results folders need updated readers** once one machine writes schema 2, the same
  rule this project has stated for every schema step since ADR 0004.
- A 0.1.0 reader meets a hardware profile with `runtimes` (or `adapters`) the same way it always
  has: a field error, not a schema refusal -- unchanged behavior, restated here because the two
  additive fields now sit side by side.
- Every later AP9-SV2 sub-package (L, P, F, I, R) builds inside this contract rather than
  touching it again: SV2-L fills `RUNTIMES` and moves Ollama onto it; SV2-P's `tensor_quant.py`
  reads `config.json` per the `QUANT_ORDER` rule above and sets `quantization_source`;
  SV2-F computes vLLM's `Fit` (the tensor-parallel rule, `num_attention_heads` on `Architecture`,
  out of this ADR's scope -- it is SV2-P's/SV2-F's field to add, not SV2-0's: `Architecture`
  lives in `contracts.py`, which SV2-P does not own in the build brief's file table, a gap the
  next sub-package's brief should close explicitly); SV2-I builds the install/disk rule; SV2-R
  carries vLLM over SSH.
- `contracts.py` stays under the 799-line size guard as this and later schema growth land,
  because `Fit`/`Rating` moved out to `contracts_fit.py` first (decision 5).

### 5. `Fit`/`Rating` move to `contracts_fit.py`, unchanged, re-exported

Both classes move verbatim (no field, validator or docstring changed) into a new
`modelroom/contracts_fit.py`; `contracts.py` imports and re-exports both names, so every
existing `from .contracts import Fit`/`Rating` (six modules, several tests) keeps working
without a change, and `contracts.Fit is contracts_fit.Fit` (the same class object, not a copy).
This is purely a size-guard move (`contracts.py` was already at 788 of 799 lines before this
schema step), done first so the schema additions above have room; it carries no contract change
of its own, and the render-document additions the build brief sketches for `Fit`
(`Fit.runtime`, `Fit.at_once`, `Fit.cards`, for AP9-SV2-L/F) are **not** part of this move --
adding them here would have made the "unchanged" claim false. They land with the sub-package
that first has a value to put in them.

## Alternatives and why not

- **Keep snapshot schema 1 and make the new shapes additive without a step**, the ADR 0006/
  0010/0011 pattern. Rejected: the GF asked at E8 for exit `3` specifically, because the tensor
  provenance exception and the new `QUANT_ORDER` labels are not "an old reader ignores a field
  it has never seen" -- they change what a *value already in range* (`provenance`, a
  `quantization` string) is allowed to mean, which an additive-only step cannot signal cleanly.
- **A runtime as a plug-in (a registry, a discovered class, a config-driven list).** Rejected by
  the GF's own framing (standardize as much as possible) and by this project's existing rule
  against dynamic loading: a fixed tuple in one module is inspectable, diffable in review,
  and needs no discovery mechanism to test. A third runtime (LM Studio) is one more tuple entry,
  not a new kind of extension point.
- **Define `RuntimeKey` in a new `runtimes.py` now, ahead of the table it will belong to.**
  Rejected: this sub-package does not own `runtimes.py` (AP9-SV2-L does, per the build brief's
  file table) and has nothing to put in it yet besides the one type alias the profile field
  needs today. `RuntimeKey` is defined in `modelroom/profile.py`, the one module that uses it in
  this sub-package; AP9-SV2-L imports it from there when it builds `runtimes.py`, rather than
  this sub-package guessing at a module's shape before its owner exists.
- **Let `Architecture.num_attention_heads` land in this ADR too**, since the build brief's
  schema-step table mentions it alongside the other snapshot-2 additions. Rejected here: the
  build brief's own "where the brief and the code disagree" table assigns it to P4/F1
  (AP9-SV2-P/F), and adding a field no producer sets yet would be undocumented scope creep this
  sub-package's own size limit (one field for `contracts.py`) does not leave room for regardless.
