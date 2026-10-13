# AP9-KO P0: the two feasibility proofs

Read on 2026-10-13 (UTC), before the contract was written down. Real answers, no synthetic stand-ins. The
contract that follows from them is in `CONTRACTS.md`, "Package findings and the load check"; the test suite
reads the recorded originals (`tests/test_ko_findings.py`).

## Proof 1: which package a Hub architecture belongs to (V1)

Files: `v1-package-scope-hf-responses.json`.

Requests, for seven GGUF repositories of three cases (a model with converters of other accounts only, a model
with several converters whose architecture identifiers differ, and a model converted by a listed packager):

- `GET https://huggingface.co/api/models/<repo>?expand[]=gguf&expand[]=sha` (the `gguf` field and the revision),
  fetched 2026-10-13T16:59Z to T17:02Z; the fields `chat_template` and `quantize_imatrix_file` of `gguf` are cut
  (a long text, and a path on the machine of the repository's owner);
- `GET https://huggingface.co/api/models/<repo>/tree/main?recursive=true` (the files and their sizes);
- `GET https://huggingface.co/<repo>/resolve/<sha>/<file>` with `Range: bytes=0-65535` for every `.gguf` file
  (its header, `general.architecture`).

Result:

- `gguf.architecture` is **not repository-wide.** `gguf.totalFileSize` equals the size of exactly one file of the
  repository (seven of seven), and the header of that file states the architecture the field states. The Hub chooses
  the file itself, by a rule it does not publish: in six repositories it is the first file by name, in the seventh
  the BF16 file, which is not the first. It cannot be relied on to be any particular file.
- One repository holds files of different architectures: a file `imatrix-...gguf` states `imatrix`, the model files
  state the model's architecture. A repository-wide reading would be wrong for it.
- A shard set states the architecture in its first shard only; the later shards have no `general.architecture` in
  their first bytes.
- Converters of one model state different identifiers (`sarvam_moe` and `bailingmoe2` for the same base model, in
  three repositories), so an identifier is a fact about one conversion, never about the base model.

Rule (CONTRACTS.md): the field proves the architecture of the one file `hub_metadata_file` finds by its size; the
architecture of every other package of the repository is `unknown` until its own header is read; a finding is bound
to repository, revision, weight identity and first shard. Scope not shown: `unknown`.

## Proof 2: the source of the architectures is readable (V5)

Files: `llama-arch.cpp.45451c366d4256f94fbc8845645e31a6b9155054.txt` (the whole answer of request 2, 1,265 lines,
recorded afterwards from the same commit; Git blob `0dc4cc961b4ce567fa51c584458ab28a85f64d34`, the blob id the
request stated), `v5-llamacpp-arch-block.txt` (the original answer, cut to the block and to the lookup function)
and `v5-block-parse-result.txt` (the result of reading it). The tests read the whole file; it holds none of the
forms CONTRACTS.md V5 refuses ("Limit of the check") and no other place that maps a name to an architecture.

Requests (2026-10-13T17:01Z and T17:02Z), the host is `api.github.com`:

1. `GET https://api.github.com/repos/ggml-org/llama.cpp/commits/HEAD`, `Accept: application/vnd.github.sha`:
   `45451c366d4256f94fbc8845645e31a6b9155054` (40 bytes, status 200, no redirect).
2. `GET https://api.github.com/repos/ggml-org/llama.cpp/contents/src/llama-arch.cpp?ref=45451c366d4256f94fbc8845645e31a6b9155054`,
   `Accept: application/vnd.github.raw+json`: status 200, 90,345 bytes, blob id `0dc4cc961b4ce567fa51c584458ab28a85f64d34`
   (the Git hash of the content, which was verified), SHA-256 of the bytes
   `7c558709ae432728cb73bef958f25c584dd0cf47cf429122b63c021216c4a65a`.

Result:

- The block is `LLM_ARCH_NAMES`, lines 8 to 167 of the file: 158 lines of the form `{ LLM_ARCH_X, "name" },`,
  one of them the sentinel `LLM_ARCH_UNKNOWN` named `(unknown)`, so **157 architecture identifiers**, no
  enum and no identifier twice. Every line was understood by one pattern; nothing remained.
- `llm_arch_from_string` (lines 1129 to 1137) compares the given name with the second member of each entry as a
  whole string. That is the comparison a loader makes, so the block is the right thing to compare with.
- At this commit `bailingmoe2` and `glm4moe` are present; `sarvam_moe` and `kolibri1` are absent.
- A change of form would be seen, not guessed: a missing or doubled start line, a missing end, a line that is not an
  entry, a repeated enum or identifier, fewer than 100 entries, or a lookup that no longer compares the second member
  is a fault (`parse_arch_names`); the tests change the recorded original in each of these ways.
- The host `api.github.com` is needed. `modelroom/http.py` allows only `huggingface.co`, `ollama.com` and
  `registry.ollama.ai`, and that stays so: the runtime of modelroom reads no source; only the scanner gets the one
  extra host, with 256 bytes and 512 KiB limits, no redirect and two requests per run. Unauthenticated, the API
  allows 60 requests an hour per address; two per day is far below that.
