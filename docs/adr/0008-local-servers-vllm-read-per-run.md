# 0008 -- Local servers as a list, vLLM read per run

- **Status:** accepted, decided 2026-09-27
- **Supersedes:** nothing; ADR 0001 to 0007 stay in force

## Context

A machine that serves models through vLLM answered the guided mode the way a machine without any
server did: the start screen named the Ollama daemon and nothing else, step 4 measured what the
daemon had installed, and the fit assumed Ollama's way of loading a model. vLLM keeps many
requests at once in one batch, takes a fixed share of the card for itself and gives the rest of it
to the KV cache, so neither the one-request speed of protocol v1 nor fit v1 says what such a server
does. On 2026-09-27 a vLLM 0.30.0 server was recorded on two rented cards (one A100-SXM4-80GB, one
L4, `Qwen/Qwen2.5-0.5B-Instruct`, utilization 0.5): its model list, one chat, twenty chats at
once, its metrics before and after, and its start log.

The owner decided the same day: the origins are read per run and nothing is persisted; the server's
identity is `stated`, a line only; the share of the card is 0.92, fixed; no API key; the recordings
of the recording day are the fixtures and the hand test runs on a rented card; the origins are a
list, with no LM Studio code. The safetensors package class is deferred (P1), and the runtime is 1.50 GiB from the
A100 recording, with 1.31 from the L4 recording as the second measured value (P2).

## Decision

1. **A list of origins.** `LOCAL_SERVERS` in `modelroom/local_servers.py` is the one place a local
   server is registered: `LocalServer(key, name, origin, transport(run), start_fact, step,
   card_facts)`. `ollama` wraps what existed, unchanged in behavior; `vllm` is the second. The start
   screen, step 4 and both card paths of step 5 loop over it.
2. **A. The transport** (`modelroom/vllm_server.py`): only `http://127.0.0.1:8000`, a path, query,
   fragment or user info refused before connecting; exactly `GET /v1/models`, `GET /metrics` and
   `POST /v1/chat/completions`; no proxy, no redirect (the opener of `daemon.py`); a `401` is the
   reason that the server asks for a key, which is never sent; answers above 4 MiB refused; 600 s
   for a chat, 10 s otherwise. `read_vllm_models` returns the one model owned by `vllm`, several are
   a reason.
3. **B. The start screen**: one `server` row under the daemon's for a server that answers; none for
   one that does not, so the screen of a machine without vLLM is byte for byte the one before.
4. **C. The load test** (`modelroom/loadtest_vllm.py`): one warm-up, then three batches of
   `min(requests, 64)` chats at once behind a barrier, the recorded request, `/metrics` before and
   after each batch. Valid, comparable, the total and per-request speeds as `CONTRACTS.md` defines
   them, a `VllmMeasurement` in memory.
5. **D. The served package**: exactly one active complete `tensor` package of the snapshot whose
   repository is the stated `root`; the fetch records such a package today (`unresolved`/`format`),
   so `hf.py`, `contracts.py` and `provenance.py` do not change.
6. **E. The fit on one card** (`modelroom/fit_vllm.py`): `0.92 x card - weights - 1.50` GiB is the
   KV cache; how many requests of the ranking's context it holds is the class (`no_fit`, `tight`,
   `fits`, `ample` on the architecture basis only). The largest card; `fit.py` unchanged.
7. **F. Step 4 and the card** (`modelroom/guided_vllm.py`): the lines and rows `CONTRACTS.md`
   lists; the third optional answer key `load_test_vllm`.
8. **Nothing is persisted.** No schema changes: measurement file 2, export 1, snapshot 1, render
   document 3, configuration 3 and profile 4 stay as they are.

## Rejected

- **Measurement schema 3** with a vLLM record: a speed of n requests at once, prefill and queue
  included, is not the speed protocol v1 measures, and the owner decided that nothing is persisted.
- **Export schema 2** carrying such measurements: nothing to carry.
- **Snapshot schema 2** with a safetensors package class of its own: deferred to SV2 (P1).
- **Render document schema 4** with a vLLM section: the render never shows a vLLM reading.

## Limits

- The evidence is one 0.5B model at utilization 0.5 on two kinds of card; 1.50 GiB is one
  recording, not a bound.
- A server started below 0.92 has less KV cache than computed.
- CUDA's first card is the one vLLM uses by default; the fit computes the largest.
- The prefix cache is not modeled; twenty equal prompts share a prefix.
- `render` never shows any of it: the reading lives for one guided run.

## Consequences

- A guided run with a reachable vLLM server that is measured and matched to a package of the
  folder writes the same bytes to the results folder and the home folder as the same run without a
  vLLM server; a test proves it by running both with the same clock and answers.
- A new local server is one entry of `LOCAL_SERVERS` and a module of its own.
- Deferred to SV2 or later: tensor parallel, the package class, a persisted measurement, document
  schema 4, the load hint of `fit.py`, a start command, `/version` and `/health`, port or
  utilization in the configuration, an API key, LM Studio.
