# 0011 -- Graphics adapters of every vendor, llmfit as fallback

- **Status:** accepted, decided 2026-09-28
- **Supersedes in part:** ADR 0004 (a counted measurement from `nvidia-smi` only; the CUDA sum
  check of several cards; "a card of another vendor is not in the list"), ADR 0006 decision 3 as
  far as llmfit is only a safety net, and the sentences of CONTRACTS.md that said llmfit is never
  the source. Everything else of both stays in force.

## Context

modelroom read the graphics memory from `nvidia-smi` alone. Every other card ended as
`present_unmeasured` (the fit blocked) or, on macOS, as `unsupported_platform`. Whoever runs the
tool on an AMD card, a laptop with integrated graphics, a Mac or a machine whose NVIDIA tools are
missing learned nothing about the card and got no fit at all.

## Decision

1. **Own reading first, llmfit as fallback, never mixed.** Mode A: at least one card of
   modelroom's own reading counts -- then only the own source counts and llmfit is the
   cross-check. Mode B: none counts -- then llmfit's `CUDA`/`ROCm` cards may count, llmfit alone,
   with the source `llmfit`. llmfit's cards are never merged with the own ones.
2. **Every unit is an adapter.** `HardwareProfile.adapters` lists every unit a source names, with
   vendor, model, memory, build, source of the memory, source of the proof, backend, `usable`
   (`yes`, `no`, `unconfirmed`), `counted` and the reason. The yardstick of `usable` is Ollama's
   documentation of 2026-09-28 (`docs.ollama.com/gpu`); vLLM stays NVIDIA as before.
3. **One countable source per backend.** CUDA from `nvidia-smi`, ROCm from sysfs with one unique
   match of the card's PCI address; the registry is shown, never counted. Of the counted backends
   the one with the largest sum counts, a tie goes to CUDA.
4. **In doubt a card is not counted.** Exclusions come before proofs, and an own exclusion of a
   vendor shuts llmfit's units of that vendor out too (its Vulkan and SYCL entries included). A
   build is read from positive marks only; integrated graphics, a Vulkan card, AMD on Windows and
   a card without a countable memory never count. A machine with recognized units and none
   counted is `none`: the CPU fit, with a note per unit.
5. **Apple Silicon is unified memory from its own reading.** `sysctl hw.optional.arm64` (or an
   llmfit entry `Metal` with unified memory) proves it; the memory is always the RAM `sysctl
   hw.memsize` read, never llmfit's. An Intel Mac is a CPU machine.
6. **`AdapterRAM` and the 32-bit `MemorySize` are never read**; the registry's 64-bit
   `qwMemorySize` is shown next to a card. `amd-smi` and `rocm-smi` are not asked: sysfs holds the
   memory. No Intel memory is read on Linux: neither the xe tile file nor an i915 `lmem` file is in
   a tagged kernel (checked 2026-09-28).
7. **Schema 4 is extended, no step** (the precedent of ADR 0006): the list `adapters` and the
   memory sources `sysfs` and `sysctl`. The list's rules hold only for a list that is not empty;
   an empty list is valid for every profile and is not written, so every earlier file stays as it
   was. A file that says schema 2 or 3 and carries either is refused.
8. **The cross-check** of the graphics memory follows the first row of the table in CONTRACTS.md
   ("Cross-check with llmfit") that applies: a missing own value is `absent`/`error` before
   anything else; llmfit's own count is `not_comparable`; a comparison happens only where llmfit's
   entries of the counted backend are exactly the counted cards.

## Consequences

- A machine with integrated graphics only, an NVIDIA machine without its driver tools, an Intel
  Mac: until now no fit, from now on the CPU fit with a note that says why no card counts.
- The free graphics memory of the card of step 5: `nvidia-smi` as before, sysfs for an AMD card,
  `unknown` for a card llmfit counted.
- In mode B a card can stand in two rows, one of the own list and one from llmfit; it is counted
  once, and the view says so under llmfit's rows.
- The Ollama backends are bound to the documentation of 2026-09-28 (`OLLAMA_GPU_DOC_DATE`).

## Alternatives and why not

- **Merge llmfit's cards into the own list.** Rejected: names differ between sources, and a
  merge counts a card twice or leaves one out; two modes that never mix have neither problem.
- **Read `amd-smi` for AMD.** Rejected: one more tool to install and match by order; sysfs has
  the memory and ties it to the card's address.
- **A schema 5.** Rejected: the extension adds a list that is empty for every earlier shape, and
  schema 4 is not released yet.
