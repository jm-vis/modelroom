# Fixtures

Recorded on 2026-09-22, offline afterwards. Tests never call the network; they load these
files. If a base model's real files, tags or manifests change upstream, these fixtures do not
change with them until someone re-records them on purpose.

## Hugging Face

- `hf_unsloth_qwen35_9b_gguf_model.json` -- `GET https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF`.
  A packager repo: `sha` is its current revision (`3885219b6810b007914f3a7950a8d1b469d598a5`),
  `tags` include `base_model:Qwen/Qwen3.5-9B`, `siblings` lists the 26 GGUF/mmproj/imatrix
  file names (22 quantized weight files including `Q4_1`, 3 `mmproj-*.gguf` projectors, 1
  `imatrix_unsloth.gguf_file`).
- `hf_unsloth_qwen35_9b_gguf_tree.json` -- `GET https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF/tree/main`.
  Same repo, with a `size` (and LFS `oid`) per file; the model-info endpoint above does not
  carry sizes, so this fills them in for size-based ordering tests.
- `hf_qwen_qwen35_9b_model.json` -- `GET https://huggingface.co/api/models/Qwen/Qwen3.5-9B`.
  The publisher repo; `sha` is its current revision (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`).
- `hf_qwen_qwen35_9b_config.json` -- `GET https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json`
  (redirects to the resolve-cache; fetched with `curl -L`). The language-model fields sit
  under `text_config`, and `text_config.layer_types` mixes `"linear_attention"` and
  `"full_attention"` -- a hybrid config, so `architecture_from_hf_config` must resolve this to
  `kind="unknown"`, never `dense_classic`.

## Ollama

`GET https://registry.ollama.ai/v2/library/qwen3.5/manifests/<tag>`, saved as
`ollama_qwen35_9b.json`, `ollama_qwen35_9b-q4_K_M.json`, `ollama_qwen35_9b-mlx-bf16.json`.
The registry sends the manifest digest only as a response header to a `HEAD` request
(`ollama-content-digest`, measured 2026-09-22: `HEAD .../manifests/9b` returns
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`); a `GET` carries no
digest header, which is why a fetcher has to issue the `HEAD` (or hash the manifest body it
received). The header is not part of these body fixtures. The manifest bodies carry the facts
the tests actually need:

- `9b` and `9b-q4_K_M` are byte-identical manifests: both list a single
  `application/vnd.ollama.image.model` weights layer with digest
  `sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` -- the digest-sibling
  case for `inherit_from_siblings`.
- `9b-mlx-bf16` lists only `application/vnd.ollama.image.tensor` layers (no `.image.model`
  layer at all) -- the tensor-format case that `decide_provenance` resolves to
  `("unresolved", "format")` regardless of its tag.

## Synthetic fixtures

`hf_synthetic_dense_config.json` is hand-written, not fetched: a plausible dense decoder
config (`num_hidden_layers`, `num_key_value_heads`, `head_dim` all present, no `layer_types`
key) so `architecture_from_hf_config` has a real `kind="dense_classic"` case to resolve
alongside the two real, non-dense fixtures above.

## Fetch (AP3)

A handful of these fixtures carry more than a raw response body -- an HTTP status and/or
response headers matter to the code under test. Those are saved as a small envelope,
`{"status": <int>, "headers": {...}, "body": <object-or-string>}`, loaded by
`tests/fixture_support.py::envelope_response`; everything else keeps the AP1 convention of a
raw body file assumed to be a `200` with no headers of interest
(`tests/fixture_support.py::json_response` / `html_response`).

- `ollama_tags_qwen35.html` -- real, recorded 2026-09-22, `GET
  https://ollama.com/library/qwen3.5/tags`. The tag names live in `<a href="/library/qwen3.5:
  <tag>" ...>` anchors; `modelroom/ollama.py` extracts only the `href` targets, nothing else
  from the page (no login, no anti-bot circumvention, one page per configured base model, per
  the `datenextraktion` rules this project follows).
- `ollama_manifest_head_9b.json` -- envelope, real, recorded 2026-09-22: the headers of `HEAD
  https://registry.ollama.ai/v2/library/qwen3.5/manifests/9b`. `ollama-content-digest` is
  `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`; a `GET` to the same URL
  carries no such header (see the Ollama section above), which is why the manifest digest is
  read from a `HEAD` request in `modelroom/ollama.py`, not from hashing the `GET` body, when the
  header is present.
- `hf_unsloth_does_not_exist_model.json` -- envelope, real, recorded 2026-09-22: `GET
  https://huggingface.co/api/models/unsloth/Does-Not-Exist-GGUF`. **Deviation from the original
  assumption:** an anonymous request against a repo that does not exist gets HTTP `401`
  (`{"error": "Invalid username or password."}`), never a `404` -- Hugging Face does not let an
  unauthenticated caller distinguish "private" from "does not exist". `modelroom/hf.py`
  therefore treats `401` exactly like `404` on the model-info call: "no package here" for a
  packager candidate, `kind="unknown"` for a base model's architecture. A genuine transport
  error or any other status still ends the area `incomplete`.
- `hf_synthetic_paginated_tree_page1.json` / `hf_synthetic_paginated_tree_page2.json` --
  synthetic, hand-written. No repo with more than 1000 files (the point at which Hugging
  Face's tree endpoint paginates) was at hand to record; these two envelopes stand in for that
  case, page 1 carrying a `Link: <...>; rel="next"` header to page 2 and page 2 carrying none,
  so `modelroom/hf.py`'s tree fetch is tested against the real pagination mechanism even
  though the specific file list is invented.

## Hardware (AP4)

Both real commands here were run on the same reference laptop, 2026-09-22.

- `llmfit_system_laptop.json` -- real, recorded 2026-09-22, `llmfit system --json` (llmfit
  1.1.16). `system.gpu_vram_gb` `11.94`, `system.total_ram_gb` `127.46` for a physical 128 GB
  machine (llmfit's own `_gb` fields are already GiB, division by `1024**3` in its source --
  taken unchanged into `HardwareSnapshot.vram_gib`/`ram_gib`, see CONTRACTS.md "Local llmfit
  binding"). Used whole, untrimmed, by `tests/test_llmfit.py` and `tests/test_cli.py`.
- `llmfit_version.txt` -- real, recorded 2026-09-22, `llmfit --version` (`llmfit 1.1.16\n`).
  Small enough to keep as the literal recorded stdout rather than only a hardcoded string in
  test code.
- `ollama_tags_local.json` -- real, recorded 2026-09-22, `GET
  http://127.0.0.1:11434/api/tags` against the local Ollama daemon, **trimmed** from 19
  installed models down to 3 (two `unsloth`/`hf.co` GGUF pulls and one library model,
  `granite4.2:8b`) to keep the fixture small; the trimmed-out entries were plain duplicates of
  the same shape, nothing behaviourally distinct was cut. One field was also redacted: the
  `granite4.2:8b` entry's real `details.parent_model` pointed at a build machine's local
  filesystem path and was blanked to `""` -- `parent_model` is not read by
  `modelroom/ollama_local.py` at all, only `name`/`digest`/`size` are, so the redaction
  changes nothing the fetcher or its tests depend on. Every entry's `digest` is the daemon's
  **bare hex, with no `sha256:` prefix** (measured 2026-09-22) -- `fetch_installed_models`
  normalizes it before building `InstalledModel` (CONTRACTS.md, "Local Ollama inventory").
