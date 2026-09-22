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
