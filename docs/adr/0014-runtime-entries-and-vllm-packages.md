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

### 6. How many cards a vLLM package uses (F4), and the divisibility rule it rests on

**Recorded here as the GF's decision; SV2-F computes it, SV2-I writes it into the command.**
vLLM only splits a model across several cards when the model can be cut into that many equal
pieces: the number of its attention heads must be divisible by the tensor-parallel size `t`
(external, documented by vLLM's own error message -- vLLM issues #4232 and #2652). Beyond that,
modelroom itself proposes using only 1, 2, 4 or 8 cards -- the build brief is explicit that this
is modelroom's own simplification, not a proven limit of vLLM, chosen to keep the arithmetic and
the paid-GPU-hour check small.

The GF answered build-brief question F4 with option **(a)**, "as many equal-sized cards as
possible": the largest of 1, 2, 4, 8 that is at most the count of equal-sized cards and still
divides the model's head count. Four cards usually give `t = 4`; three cards give `t = 2` (the
third stays free); a model with 14 heads on four cards gives `t = 2`. Consequence (the GF's own
framing): the highest number of concurrent requests, one simple rule, and unused cards stay free
while the command names the ones actually used. Two further cases the build brief's table of
edge cases states for this rule: the model's attention-head count is unknown -> `t = 1`, with
the reason named (divisibility cannot be checked, so no split is proposed); unequal cards -> the group of the largest equal-sized cards is used, noted at the head of the
table. `Architecture.num_attention_heads`, the field this rule needs to read, is not part of
this sub-package (see "Alternatives and why not").

### 7. vLLM is started by pasting a command, never by modelroom itself (F1)

**Recorded here as the GF's decision; SV2-I writes the command line, SV2-R carries the same rule
to a server over SSH.** A vLLM server is a program that keeps running until someone stops it.
Build-brief question F1 asked who starts it; the GF answered option **(a)**: modelroom shows the
finished start command, the operator pastes it at the machine and leaves that window open --
modelroom never starts a program itself. The GF's own stated consequence: one extra manual step,
against the alternative of modelroom starting vLLM itself in the background (a real background
process, a log file, a stop command, one more thing to get wrong per operating system) -- kept
out for exactly that reason (see "Rejected simplifications" below). The same rule holds at a
server reached over SSH: `modelroom remote vllm`/`vllm-measure` only read there, matching the
Security DoD line this sub-package adds to `AGENTS.md` ("no start, no loading").

### 8. Assumptions A1-A3: unconfirmed until the paid GPU hour, named as assumptions in the code

The build brief names eight assumptions behind the vLLM work (A1-A8); three of them touch
contracts and fit arithmetic this sub-package's later siblings compute with, and are recorded
here so the assumption -- and the paid-GPU-hour check that will confirm or refute it -- is not
lost between sub-packages:

- **A1**: the 1.50 GiB runtime reserve (`VLLM_RUNTIME_OVERHEAD_GIB`, E9) applies *per card* even
  under tensor-parallel; it is documented for one card only. The rented-GPU check **H6** compares
  it against `Available KV cache memory` in vLLM's own start log.
- **A2**: with fewer KV heads than cards, vLLM replicates KV heads, so each card counts
  `per_token / min(t, num_key_value_heads)`; a hybrid model's state divides by `t`. A later
  Codex round on this build brief (r1-9) noted that a test of the formula alone proves nothing
  about vLLM's own behavior: A2 counts as confirmed only once the rented-GPU check **H6b**
  matches the token count from vLLM's start log, both for a model with fewer KV heads than cards
  and for a hybrid model; until then, the code's `memory` line for such a row is labeled
  `(assumed)`.
- **A3**: a non-interactive SSH command finds `vllm` only on the `PATH` or under `~/.local/bin`
  (where `uv tool install vllm` places it) -- the same assumption AP9-FR's build brief (FR B1)
  already makes for finding `modelroom` itself over SSH. The rented-GPU check has three steps,
  each read from `'{}' | ssh gpu1 modelroom remote hello`: **H1a** before vLLM is installed, the
  answer carries `"vllm": "absent"`; **H1b** after `uv tool install vllm` (binary under
  `~/.local/bin`, which a non-interactive SSH shell does not put on its `PATH`), it carries
  `"vllm": "installed"`; **H1c** with a link to that binary in a directory on the non-interactive
  `PATH` (for example `/usr/local/bin`) and the `~/.local/bin` copy renamed, it still carries
  `"vllm": "installed"`. Any other answer in one of the three steps refutes A3.

None of A1-A3 is confirmed by this sub-package; it only carries the assumption and its named
rented-GPU check forward so the sub-package that computes with it (SV2-F for A1/A2, SV2-R for
A3) does not have to rediscover or silently drop the caveat.

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
  SV2-F computes vLLM's `Fit` (the tensor-parallel rule of decision 6, `num_attention_heads` on
  `Architecture`, out of this ADR's scope -- it is SV2-P's/SV2-F's field to add, not SV2-0's:
  `Architecture` lives in `contracts.py`, which SV2-P does not own in the build brief's file
  table, a gap the next sub-package's brief should close explicitly); SV2-I writes the start
  command of decision 7 and the install/disk rule; SV2-R carries both over SSH.
- `contracts.py` stays under the 799-line size guard as this and later schema growth land,
  because `Fit`/`Rating` moved out to `contracts_fit.py` first (decision 5).
- A1-A3 (decision 8) stay unconfirmed, and every value the later sub-packages compute from them
  is labeled as such, until the rented-GPU checks (H6, H6b) report back.

## Rejected simplifications (from the build brief's "Not here" table)

Entries the build brief itself states as deliberately left out, where they touch a decision
above:

- **modelroom starts vLLM itself, locally or over SSH, in the background.** Rejected: process
  management across operating systems and over SSH (detaching, logs, stopping) is a new attack
  surface for little gain; the GF can paste the command herself (decision 7, F1).
- **Optimizing `t` beyond the power-of-two rule, tensor-parallel over unequal cards, or
  pipeline-parallel.** Rejected: one rule is enough for the command the build brief needs;
  unequal cards are excluded via `CUDA_VISIBLE_DEVICES`, never mixed into one group (decision 6).
- **A `latest` Docker tag, modelroom's own image choice, or Docker Compose.** Rejected: a fixed
  tag (A8) and one output line are enough (build-brief question F2, answered **b**: a Docker line
  is shown beside the plain `vllm serve` command, not a replacement for it).
- **Saving a vLLM measurement, a measurement schema 3, or an export schema 2.** Rejected: E3
  already says a vLLM measurement never ranks, so there is nothing new to persist.
- **Port or GPU utilization as configuration values.** Rejected: both are fixed already (E2 the
  origin `127.0.0.1:8000`, E4 `gpu_memory_utilization = 0.92`), so a setting would only invite a
  value vLLM itself was never asked to honor.

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
