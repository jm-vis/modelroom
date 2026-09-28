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
- `hf_search_qwen_*.json` -- the pinned answers for the search's pages. Since 2026-09-24 a search
  asks **one page per account** and, only with the owner filter off, two open pages on top
  (`CONTRACTS.md`, "Search over the Hugging Face API"); `fixture_support.search_transport_mapping`
  binds each account of a `qwen` search to one of these files. **Real shape, curated entries.**
  The request forms were measured live on 2026-09-23 and again on 2026-09-24, both `HTTP 200`:
  an account page (`author=unsloth&search=qwen3.5&...`) answers with that account's own
  `Qwen3.5-*-GGUF` repositories whatever their age, each with a `downloads` count (3 551 to
  68 445 in that answer) and its `base_model:quantized:` tags; the open `sort=downloads` page
  answers with ten repositories between 1.6 M and 12.0 M downloads, among them several
  `uncensored`/`abliterated` derivatives of accounts nobody listed. `expand=downloads` is
  required as soon as `expand` is set at all -- without it the field is simply absent. Those live
  answers themselves are **not** committed: they are third-party accounts that say nothing about
  this project's own cases and would go stale within the hour. The entries here are built from
  them instead, and are what a pinned run needs to be repeatable:
  - `hf_search_qwen_publisher.json` -- the `Qwen` page: the publisher's own `Qwen/Qwen3.5-9B-GGUF`
    (resolved, `publisher`, with a real `safetensors.total` and 12 031 627 downloads, the `12.0M`
    case of the downloads column).
  - `hf_search_qwen_unsloth.json` -- the `unsloth` page, newest first: an unresolved
    `relation_unknown` repository (3 551 downloads, the plain-number case), the resolved
    `unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF` whose base model the catalog lists *without* an
    Ollama assignment (68 445 downloads, the `68k` case and the `none known` case), and the real
    `unsloth/Qwen3.5-9B-GGUF` (resolved, `listed packager`, 1 626 475 downloads). One unresolved
    repository of a listed account is what keeps the grouped-reason line testable with the owner
    filter **on**.
  - `hf_search_none.json` -- the empty list every other asked account answers with. An account
    that answers with nothing still gets its own line in the summary.
  - `hf_search_qwen_most_downloaded.json` -- the open `sort=downloads` page: a `derivative` of a
    foreign account, `unsloth/Qwen3.5-9B-GGUF` **again** (the duplicate that has to appear once,
    in its account group, and be counted as `already listed`), and a `publisher_unknown` entry.
  - `hf_search_qwen_newest.json` -- the open `sort=createdAt` page: the `base_model_tag` entry
    (the one entry with no `downloads` field at all, the `unknown` case), a `metadata_conflict`
    entry and a `relation_unknown` entry with `downloads: 0`.
  - `hf_search_qwen_page_full.json` -- exactly 20 generated entries, the account page limit, for
    the one test about a full page; the same file minus its last entry is the 19-entry case that
    must **not** be called full.
- `hf_search_mistral_publisher.json` / `hf_search_mistral_unsloth.json` -- the two answered pages of
  a search for `mistral`, **cut from the live answers of 2026-09-25**, both `HTTP 200`:
  `GET https://huggingface.co/api/models?author=mistralai&search=mistral&filter=gguf&sort=createdAt&direction=-1&limit=20&expand=...`
  (11 entries) and the same request with `author=unsloth` (7 entries). Two entries were kept from
  each, with the long `language` lists dropped and nothing else changed:
  - the publisher page keeps `mistralai/Ministral-3-14B-Instruct-2512-GGUF` and
    `mistralai/Magistral-Small-2509-GGUF` -- the publisher packaging its own models, the case that
    resolves to `publisher_status: publisher`;
  - the `unsloth` page keeps `unsloth/Mistral-Small-4-119B-2603-GGUF` and
    `unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF`, each with its
    `base_model:quantized:mistralai/...` tag.
  `fixture_support.mistral_search_mapping` binds them, answers every other asked account with
  `hf_search_none.json` and pins one age lookup per resolved base model. The pages exist because
  `mistralai` became a publisher of the shipped catalog on 2026-09-25: before that, the same
  repositories were all `publisher_unknown`.
- **The three forms of input** (decided 2026-09-25, all four files cut from live answers of that
  day, every entry verbatim; only a long `language` list inside `cardData` was dropped, the same
  curation the `mistral` pages above document):
  - `hf_typed_unsloth_qwen35_9b_gguf.json` -- `GET
    https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF?expand=cardData&expand=createdAt&expand=downloads&expand=safetensors&expand=tags`,
    the whole object unchanged. A typed repository id that **holds** GGUF files: its `tags` carry
    `gguf` and `base_model:quantized:Qwen/Qwen3.5-9B`, so one request resolves it.
  - `hf_typed_qwen_qwen35_9b.json` -- the same request for `Qwen/Qwen3.5-9B`, the base model
    itself. Its `tags` carry `safetensors` and **no** `gguf`, which is the "holds no GGUF file"
    case: the search says so and falls back to the words of the name half.
  - `hf_search_qwen_unsloth_downloads.json` -- two entries of `GET
    ...?author=unsloth&search=qwen&filter=gguf&sort=downloads&direction=-1&limit=20`:
    `unsloth/Qwen3.5-9B-GGUF` and `unsloth/Qwen3.5-4B-GGUF`. **The measurement this page exists
    for:** the same account's `sort=createdAt` page answered with twenty repositories created after
    2026-05 and `unsloth/Qwen3.5-9B-GGUF` (2026-02-28) was not among them at all, so one sort order
    alone cannot find the plain build of a listed model. Since 2026-09-25 an account is therefore
    asked twice.
  - `hf_search_qwen38_27b_catalog_page.json` -- three entries of `GET
    ...?search=Qwen3.8-27B&filter=gguf&sort=downloads&direction=-1&limit=10`, which is one page of
    a search with **no word at all** (one per current model of the catalog, and `Qwen3.8-27B` is
    the catalog's current Qwen model). The three are the cases the owner filter is about:
    `unsloth/Qwen3.8-27B-GGUF` (resolved, `listed packager`), a `DavidAU/...` build whose declared
    base model belongs to an account the catalog does not name as a publisher
    (`publisher_unknown` with the filter on, resolved with `other` when it is off), and
    `cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF`, a publisher's model packaged by an account of
    neither list.

  `fixture_support.typed_transport_mapping` and `catalog_transport_mapping` bind them; the second
  reads the current models from the shipped catalog rather than listing them here, so a model added
  later is bound too instead of ending a run at a URL no fixture answers. Both sort orders of an
  account are bound to that account's **one** curated page in `search_transport_mapping`: the
  curation is about which repositories an account has, not about the order they come back in, and a
  page of duplicates is what an account with few repositories really answers.
- `hf_qwen_qwen35_9b_config.json` -- `GET https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json`
  (redirects to the resolve-cache; fetched with `curl -L`). The language-model fields sit
  under `text_config`, and `text_config.layer_types` mixes `"linear_attention"` and
  `"full_attention"` -- a hybrid config, so `architecture_from_hf_config` must resolve this to
  `kind="unknown"`, never `dense_classic`.

## Ollama

`GET https://registry.ollama.ai/v2/library/qwen3.5/manifests/<tag>`, saved as
`ollama_qwen35_9b.json`, `ollama_qwen35_9b-q4_K_M.json`, `ollama_qwen35_9b-mlx-bf16.json`.
**F5 (fix-round 1, superseding the assumption below):** measured again 2026-09-22 --
`sha256` of the `9b` manifest's raw `GET` body (709 bytes) is
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`, byte-for-byte the same
value a `HEAD .../manifests/9b` request's `ollama-content-digest` header used to state.
`modelroom/ollama.py` now computes the digest this way and makes **no `HEAD` request at
all**; the original assumption below (that the digest could only be read from the `HEAD`
header) was true of the header's *content*, never of the *necessity* of the extra request.
The manifest bodies carry the facts the tests actually need:

- `9b` and `9b-q4_K_M` are byte-identical manifests: both list a single
  `application/vnd.ollama.image.model` weights layer with digest
  `sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` -- the digest-sibling
  case for `inherit_from_siblings`, and since 2026-09-25 the case for **one package per manifest**:
  the two tags have the same manifest digest, so they are one package named `qwen3.5:9b` with
  `aliases = ["9b-q4_K_M"]` (CONTRACTS.md, "Package").
- `9b-mlx-bf16` lists only `application/vnd.ollama.image.tensor` layers (no `.image.model`
  layer at all) -- the tensor-format case that `decide_provenance` resolves to
  `("unresolved", "format")` regardless of its tag.

There is no fixture for a manifest's `params` layer, and no test that reads one: measured
2026-09-23, `GET registry.ollama.ai/v2/library/qwen3.5/blobs/sha256:9371364b...` (the `9b`
manifest's `params` layer) answers `307` with a signed `Location` on an object-storage host
that `modelroom/http.py` is not allowed to open. `Package.default_context` therefore stays
`None` -- see `modelroom/ollama.py`'s module docstring and CONTRACTS.md, "Ollama packages:
files and `default_context`".

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
- `ollama_manifest_head_9b.json` -- envelope, real, recorded 2026-09-22: the headers of a
  `HEAD https://registry.ollama.ai/v2/library/qwen3.5/manifests/9b` request. **Unused since
  fix-round 1 (F5):** `modelroom/ollama.py` no longer makes a `HEAD` request at all (see the
  Ollama section above), so nothing reads this file any more; kept as the historical record of
  the measurement that F5's fix relies on (its `ollama-content-digest` value is exactly the
  `sha256` of the `9b` manifest's `GET` body, `tests/test_ollama.py` asserts the two are equal).
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

## Hardware measurement (AP9-A)

Hand-written, synthetic outputs in the exact shape of the real commands; no host names, no
paths, no serial numbers. Written 2026-09-23. Each is one machine class from the plan.

- `nvidia_smi_one_gpu.csv` -- `nvidia-smi --query-gpu=index,name,memory.total
  --format=csv,noheader,nounits` on a laptop with one NVIDIA adapter: index `0`, `12282` MiB
  (11.99 GiB). The adapter name is the one `llmfit_system_laptop.json` also reports, so the two
  fixtures describe the same machine class.
- `nvidia_smi_two_gpus.csv`, `llmfit_system_two_gpus.json` -- synthetic until 2026-09-27 (2 x
  `NVIDIA A100-SXM4-40GB`, 40 GiB each), **recorded** since; see the next section.
- The chassis type has no fixture file: `Win32_SystemEnclosure.ChassisTypes` and
  `/sys/class/dmi/id/chassis_type` answer one number each, written in the tests that use them --
  `10` (Notebook, the laptop of `fixture_support.windows_runner`) on Windows and `3` (Desktop) in
  every Linux file layer; the recorded `1` of the two rented machines is in the next section.
- `profiles_v2/workstation.json`, `profiles_v3/workstation.json` -- one measured profile exactly as
  version 0.1.0 wrote it, as schema 2 and as schema 3 (no card list, no class), for the readers
  and `modelroom migrate`; `export_v1_profile_v3.json` is `export_v1.json` with that profile as
  schema 3.
- `lspci_nn_cpu_server.txt` -- `lspci -nn` on a CPU-only virtual server: the only class-`0300`
  device is the QEMU/Bochs display adapter `[1234:1111]`, so this is a CPU machine with the note
  `display adapter only`.
- `lspci_nn_intel_laptop.txt` -- `lspci -nn` on a laptop with Intel graphics and no
  `nvidia-smi`: `present_unmeasured` with the CPU hint.
- `lspci_nn_nvidia_laptop.txt` -- Intel graphics *and* a discrete NVIDIA `[10de:2bb4]` adapter,
  again without `nvidia-smi`: `present_unmeasured` naming both vendors, and deliberately without
  the Intel-only hint.
- `lspci_nn_headless.txt` -- a server with no class-03 device at all: `gpu_state` `none`.
- `proc_meminfo_linux.txt` -- the first lines of `/proc/meminfo`, `MemTotal` `32612348` kB
  (31.10 GiB).

The Windows sources have no fixture files: `Win32_VideoController.PNPDeviceID` values, the
`reg query MachineGuid` output, `sysctl hw.memsize` and `ioreg`'s `IOPlatformUUID` are one line
each and are written in the tests that use them (`tests/test_measure.py`), with a fictitious
GUID and UUID.

## Two rented GPU machines (AP9-F)

Recorded on **2026-09-27** by the maintainer on two rented Linux instances with two NVIDIA cards
each, with the exact argument lists of `modelroom/measure.py` and `modelroom/llmfit.py` (llmfit
1.1.16). Whole files, nothing shortened; no host name, address or machine identifier is in them.
The two instances: one at Verda (Helsinki), one at Scaleway (Paris).

- `nvidia_smi_two_gpus.csv` -- recorded 2026-09-27 on Verda, 2 x NVIDIA A100-SXM4-80GB: two lines
  of `81920` MiB, `gpu_state` `multi_gpu` with 160.00 GiB. Replaces the synthetic two-card list of
  2026-09-23.
- `llmfit_system_two_gpus.json` -- recorded 2026-09-27 on Verda, 2 x NVIDIA A100-SXM4-80GB:
  `llmfit system --json`, one `gpus[]` entry of backend `CUDA` with `vram_gb` 80.0 and `count` 2
  (160.00 GiB counted), `total_ram_gb` 235.92. Replaces the synthetic file of 2026-09-26. It proves
  that the cross-check of several cards counts `vram_gb x count`; the laptop recording cannot, both
  of its entries have `count` 1 (and its `SYCL` Intel graphics is not counted).
- `nvidia_smi_two_l4.csv` -- recorded 2026-09-27 on Scaleway, 2 x NVIDIA L4: two lines of `23034`
  MiB (22.49 GiB each), `multi_gpu` with 44.98 GiB.
- `llmfit_system_two_l4.json` -- recorded 2026-09-27 on Scaleway, 2 x NVIDIA L4: `vram_gb` 22.49,
  `count` 2, `total_ram_gb` 94.29. `tests/test_measure_cards.py` holds it against
  `nvidia_smi_two_l4.csv` and that machine's `MemTotal` (98870364 kB, written in the test): both
  checks `confirmed`.
- `lspci_nn_two_a100_server.txt` -- recorded 2026-09-27 on Verda, 2 x NVIDIA A100-SXM4-80GB: the
  two cards as `3D controller [0302]` `[10de:20b2]`, two NVSwitch bridges of class `0680`, and the
  QEMU/Bochs display controller `[1234:1111]` of class `0380`, a vendor of the positive list in
  `measure.DISPLAY_ONLY_VENDORS`. Without `nvidia-smi` this machine is `none` since AP9-GPU: both
  cards named, neither counted, the NVIDIA driver tools named (until then `present_unmeasured`).
- `lspci_nn_two_l4_server.txt` -- recorded 2026-09-27 on Scaleway, 2 x NVIDIA L4: the two cards as
  `3D controller [0302]` `[10de:27b8]`, and no other class-03 device.
- `profiles_v4/two_a100_server.json`, `profiles_v4/two_l4_server.json` -- recorded 2026-09-27 on
  Verda (2 x NVIDIA A100-SXM4-80GB) and on Scaleway (2 x NVIDIA L4): the schema-4 profile
  `hardware` wrote there, unchanged but for two values. `display_name` is `server` (the recorded
  one was the host name), and `os_fingerprint` is a fixed literal of the same length and shape (the
  recorded one is a digest of the machine id and does not travel). Both carry `machine_class`
  `unknown` (chassis type `1`) and two `confirmed` cross-checks, and end without a newline, as
  `state.atomic_write_json` writes every profile.
- Chassis type `1` -- recorded 2026-09-27 on both instances in `/sys/class/dmi/id/chassis_type`.
  Like every chassis type it has no fixture file and is written in the test that reads it
  (`tests/test_measure_cards.py`, Linux file layer), where it names no class. The evidence of what
  stands behind it: `sys_vendor` `QEMU` and `product_name` `Ubuntu 24.04 PC (Q35 + ICH9, 2009)` on
  Verda, `sys_vendor` `Scaleway` and `product_name` `SCW-L4-2-24G` on Scaleway.

Still synthetic: `nvidia_smi_one_gpu.csv` and the other `lspci_nn_*` and `proc_meminfo_linux.txt`
files of the section above, and every list of three cards or more or of mixed sizes, which the
tests write inline (`tests/test_measure_cards.py`, `tests/test_profile_v4.py`). Also synthetic:
`profiles_v4/four_l4_server.json` (AP9-UX, 2026-09-28) -- **derived, not measured**: a copy of
`two_l4_server.json` with these fields changed and nothing else: `gpus` four entries (index 0-3,
each `NVIDIA L4`, 22.49, `nvidia-smi`), `vram_gib` and `llmfit_crosscheck.vram.own_gib`/`llmfit_gib`
89.96, `profile_id` `4c7e19a2b5d8f036`, `os_fingerprint` `c3d4e5f607182939` (a fixed literal of the
same shape; the source stays `linux_machine_id`), `recorded_at` `2026-09-28T09:14:03Z`. Unchanged:
`display_name` `server`, `machine_class` `unknown`, the RAM, the cross-check status, `llmfit_version`,
no newline at the end. All three server profiles are called `server`, so a test finds a block by its
`profile_id`, never by the label (the import puts the short id behind a shared name).

## Graphics adapters of every vendor (AP9-GPU)

**Recorded on the maintainer's laptop, 2026-09-28** (Windows, NVIDIA + Intel graphics), with the
argument lists of `modelroom/gpu_windows.py` and `modelroom/gpu_nvidia.py`, shortened to the fields
the package reads; every `PNPDeviceID` is cut after its second `\` (replaced by `X`), no serial,
host or user name is in them:

- `win_video_controllers_laptop.json.txt` -- `Win32_VideoController` `Name` and `PNPDeviceID`:
  `Intel(R) Graphics` (`VEN_8086&DEV_7D67`) and the NVIDIA RTX PRO 3000 Blackwell Generation Laptop
  GPU (`VEN_10DE&DEV_2F38`).
- `win_registry_display_laptop.json.txt` -- the display class of the registry, `DriverDesc`,
  `MatchingDeviceId` and `HardwareInformation.qwMemorySize`: `null` for the Intel graphics,
  `12820938752` bytes (11.94 GiB, above the 4-GiB cap of `AdapterRAM`) for the NVIDIA card, whose
  id the driver writes in lower case.
- `nvidia_smi_compute_cap_laptop.csv` -- `nvidia-smi --query-gpu=index,compute_cap`: `0, 12.0`.

**Derived from the recordings above:** the `lspci_nnD_*.txt` files are the `lspci_nn_*.txt`
recordings and syntheses of the same name with the PCI domain `0000:` in front of every address
line, as `lspci -nnD` prints it, and nothing else changed.

**Synthetic after the documentation** (the format of the source, values chosen; fetched
2026-09-28):

- `lspci_nnD_two_amd.txt` -- two Navi 31 cards and a Raphael unit, in the `lspci -nnD` line format
  (`pciutils`, `lspci.c`, `-D`, `-nn`).
- `lspci_nnD_wsl.txt` -- the Hyper-V stand-in `[1414:008e]` of WSL (`Microsoft Corporation Basic
  Render Driver`).
- `amdgpu_mem_info_vram_total.txt`, `amdgpu_mem_info_vram_used.txt` -- one byte count each, `%llu\n`
  (`drivers/gpu/drm/amd/amdgpu/amdgpu_vram_mgr.c`, the DOC comments `mem_info_vram_total` and
  `mem_info_vram_used`, documented in `Documentation/gpu/amdgpu/driver-misc.rst`; kernel tag
  v6.16).
- `kfd_node_gpu_properties.txt`, `kfd_node_cpu_properties.txt` -- the `properties` file of a GPU
  node (`location_id` 768 = bus 3, device 0, function 0; `domain` 0) and of a CPU node
  (`location_id` 0): `sysfs_show_32bit_prop(..., "location_id", ...)` and `"domain"`, set from
  `pci_dev_id()` and `pci_domain_nr()` (`drivers/gpu/drm/amd/amdkfd/kfd_topology.c`, tag v6.16).
- `macos_sysctl_arm64.txt`, `macos_sysctl_brand_string.txt` -- `sysctl -n hw.optional.arm64` and
  `sysctl -n machdep.cpu.brand_string` of an Apple Silicon Mac (`1`, `Apple M3 Max`).
- `system_profiler_intel_mac.json.txt` -- `system_profiler SPDisplaysDataType -json` of an Intel
  Mac with integrated and AMD graphics, the keys `sppci_model` and `spdisplays_vram`.
- `llmfit_system_amd.json`, `llmfit_system_apple.json`, `llmfit_system_intel_arc.json`,
  `llmfit_system_zero_memory.json` -- `llmfit_system_laptop.json` with the `system` block changed:
  one `ROCm` card; Apple Silicon (`Metal`, `unified_memory` true); one `Vulkan` Intel Arc A770; a
  `CUDA` and a `SYCL` entry with `vram_gb` 0, which llmfit allows. The backends are llmfit's own
  names (`llmfit-core/src/hardware.rs`, read 2026-09-28).

Not read and not in any fixture: an Intel memory on Linux. Neither the xe file
`tile*/physical_vram_size_bytes` nor an i915 `lmem_total_bytes` is in a tagged kernel (checked
against `drivers/gpu/drm/xe/xe_tile_sysfs.c`, `xe_vram_freq.c` and
`drivers/gpu/drm/i915/i915_sysfs.c` on 2026-09-28), so the package reads neither.

## Local servers (AP9-SV1)

Recorded on **2026-09-27** by the maintainer on the same two rented instances as the section above,
against vLLM 0.30.0 in its container (`vllm serve`, `Qwen/Qwen2.5-0.5B-Instruct`, `--max-model-len
4096`, `--gpu-memory-utilization 0.5`, one card of the two). The client made the three calls of
`modelroom/vllm_server.py` and nothing else. The answers are whole files, verbatim; the two start
logs are excerpts of the lines named below, verbatim, checked for addresses (none is in them). No
host name, address or machine identifier is in any file.

- `vllm_v1_models_l4.json` -- recorded 2026-09-27 on Scaleway, one NVIDIA L4: `GET /v1/models`,
  one entry, `owned_by` `vllm`, `id` equal to `root` (`Qwen/Qwen2.5-0.5B-Instruct`), `max_model_len`
  4096.
- `vllm_chat_one_l4.json` -- the same server: the warm-up request (`Request 0.`) and its answer,
  as the recording client wrote them (`request`, `status`, `response`, wall time): `finish_reason`
  `length`, usage 43 prompt + 128 completion tokens.
- `vllm_chat_concurrent_l4.json` -- the same server: twenty requests at once (`Request 1.` to
  `Request 20.`), all `200`, `length`, 128 tokens; 2560 tokens in 0.7203 s is 3554 tok/s in all, one
  request 179.4 to 185.2 tok/s. The load test tests replay the answers with the recorded wall times.
- `vllm_metrics_before_l4.txt`, `vllm_metrics_after_l4.txt` -- the same server: `GET /metrics`
  before and after the twenty. Running and waiting 0; `vllm:generation_tokens_total` 128 -> 2688,
  twenty times 128. The tests set these three numbers per batch and keep the rest of the page.
- `vllm_server_log_a100.txt` -- recorded 2026-09-27 on Verda, one NVIDIA A100-SXM4-80GB: lines 8,
  13, 21, 28, 41, 43 and 50 of the server's start log -- the arguments (`max_model_len` 4096,
  `gpu_memory_utilization` 0.5), the engine configuration (`dtype=torch.bfloat16`),
  `Checkpoint size: 0.92 GiB`, the loaded model, `Available KV cache memory: 37.58 GiB`, `GPU KV cache size: 3,283,360
  tokens` and the memory summary. With `nvidia_smi_two_gpus.csv` (80 GiB) it is the source of the
  runtime of 1.50 GiB: `0.5 x 80 - 0.920 - 37.58`.
- `vllm_server_log_l4.txt` -- recorded 2026-09-27 on Scaleway, one NVIDIA L4: lines 8, 13, 21, 28,
  42, 44 and 51 of its start log, the same lines: `Available KV cache memory: 9.02 GiB`, `GPU KV
  cache size: 787,840 tokens`. With `nvidia_smi_two_l4.csv` (22.494 GiB) it is the second measured
  value, 1.31 GiB.
- `hf_qwen25_05b_instruct_model.json`, `hf_qwen25_05b_instruct_config.json`,
  `hf_qwen25_05b_instruct_tree.json` -- recorded 2026-09-27 from the Hub, one anonymous `GET` each,
  at the URLs `modelroom/hf.py` builds: `/api/models/Qwen/Qwen2.5-0.5B-Instruct`, `config.json`
  at sha `7ae557604adf67be50417f59c2c2f167def9a775` and the recursive tree at the same sha. One
  `model.safetensors` of 988,097,824 bytes; the configuration has no `head_dim` (896 hidden, 14 heads, 2 KV heads, 24
  layers), so the architecture is `unknown` and the fit is on the size basis.
- `hf_qwen3_8b_awq_model.json`, `hf_qwen3_8b_awq_tree.json` -- recorded 2026-09-27 from the Hub the
  same way: `Qwen/Qwen3-8B-AWQ` at sha `4da05a8edb55c6046cce958586c33b61da07bb79`, two safetensors
  shards (4,853,922,024 and 1,244,659,840 bytes): one complete `tensor` package whose weights are
  the sum.

Synthetic, in the tests: the metrics numbers of each batch, the three batch times of the load test,
a `401`, a list of two served models, a `finish_reason` `stop`, and a server that goes away.

## The publisher's repository as a target (AP9-UX2)

Recorded on **2026-09-28** by the builder of AP9-UX2 from the public Hub, one anonymous `GET` each at
the URLs `modelroom/hf.py` builds, and nothing else. Whole answers, verbatim; no host name, address
or secret is in them. Since AP9-UX2 a guided run fetches every resolved base model's own repository
(P3), so the fixture chain needs the publisher trees; `Qwen/Qwen3.5-4B` is the model the vLLM
server of the second hand run served.

- `hf_qwen_qwen35_9b_tree.json` -- the recursive tree of `Qwen/Qwen3.5-9B` at sha
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (the sha of `hf_qwen_qwen35_9b_model.json`): four
  safetensors shards `model.safetensors-0000N-of-00004.safetensors` and their index.
- `hf_deepseek_r1_qwen3_8b_tree.json` -- the recursive tree of `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
  at sha `6e8885a6ff5c1dc5201574c8fd700323f23c25fa` (the sha of `hf_deepseek_r1_qwen3_8b_model.json`):
  two shards `model-0000N-of-000002.safetensors` and their index.
- `hf_qwen_qwen35_4b_model.json`, `hf_qwen_qwen35_4b_config.json`, `hf_qwen_qwen35_4b_tree.json` --
  `/api/models/Qwen/Qwen3.5-4B` (sha `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), its `config.json` at
  that sha (the text layers of the 9B model: 32 layers, 24 `linear_attention` and 8
  `full_attention`, `mamba_ssm_dtype` `float32`) and the recursive tree at that sha: two shards of
  5,329,398,688 and 3,990,429,408 bytes, one `tensor` package.

Synthetic, in the tests (AP9-UX2): the packages of `acme/Nova-7B` behind the pull list and the
repository lines (`tests/fixture_rows.py`), the fixed room of the laptop (`tests/fixture_room.py`:
10,444 MiB graphics memory free, 44,236,955,648 bytes of disk free on `C:`), and the answers of
`nvidia-smi --query-gpu=index,name,memory.free` in `tests/test_room.py`.

## Schema 2 and migration (AP9-K)

All hand-written, fictitious machines and models (`acme/Nova-*`, `packager/*`); no real host
names, paths or people. Written 2026-09-23.

- `profiles_v1/` -- four schema-1 hardware profiles as `hardware` wrote them, one per case the
  migration has to handle without guessing: `windows-nvidia-laptop.json` (one GPU with VRAM,
  two embedded measurements), `linux-cpu-server.json` (no GPU, VRAM 0, Ollama not reachable),
  `unified-memory.json` (`unified_memory = true`) and `vram-zero.json` (an integrated adapter,
  VRAM 0). The laptop's second measurement has `tps_mean` 21.0 below its `tps_range` of
  21.5..23.0 on purpose: schema 1 never checked the range against the mean, and the migration
  carries such a value over unchanged.
- `config_v1/config.toml` -- a schema-1 configuration with two machines (`laptop` writer,
  `server`), paths relative to its own folder; `tests/test_migrate.py` copies it next to
  `profiles_v1/` into a temporary folder as `modelroom.toml`. The fixture is not named
  `modelroom.toml` itself: the release gate treats a file of that name as an operator's
  configuration, which never ships. `config_v2/config.toml` is the schema-2 counterpart; it stays
  as the evidence of the step to schema 3 (`tests/test_migrate_v3.py`, `tests/test_config_v3.py`).
  `config_v3/config.toml` holds the same values as schema 3 for `tests/test_importer.py` (written
  2026-09-25).
- `catalog_excerpt.toml` -- a catalog in the shipped format with one `latest`, one `legacy`
  (with successor) and one `unknown` model.
- `export_v1.json` -- an export object: the `HardwareProfile` example plus two measurements,
  one protocol `v1` and one migrated (`protocol: none`) from the laptop fixture. Its profile is
  schema 2, as version 0.1.0 wrote it, and stays so: it is the evidence that an export of an
  earlier version still imports (the profile is read as schema 3).
- `measurement_v2_invalid.json` -- a protocol-v1 measurement stored as `invalid` (run 2 ended
  with `done_reason: stop`), with its reason.

The export and the invalid measurement were generated from `modelroom/examples.py` and
validated before they were written; the relation check's positive case uses the recorded
`hf_unsloth_qwen35_9b_gguf_model.json` above (relation only in `tags`).

## The load test, stage 1 (AP9-E)

Recorded from a real Ollama daemon (0.34.2) and the live Hugging Face API on **2026-09-24**. The
example package throughout is `hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M`: a GGUF build
of a dense base model, so fit v1 can judge it, and the one the acceptance run measures for real.
Machine paths and user names are blacked out (`tests/test_public_hygiene.py`).

- `ollama_version_local.json` -- `GET /api/version`.
- `ollama_tags_local_loadtest.json` -- `GET /api/tags`: the three entries of
  `ollama_tags_local.json` above, plus the example package (its manifest digest
  `ecc092d5…3aec`) and one cloud entry (`glm-5.3-flash:cloud`) whose `size` was set to `0` so
  both halves of the cloud rule have a case.
- `ollama_ps_deepseek_loaded.json` -- `GET /api/ps` while that package is loaded: the local name,
  the same `digest` as in `/api/tags` (bare hex, no `sha256:` prefix) and `context_length` 8192,
  the three facts `comparable` is decided by.
- `ollama_show_hf_gguf.json` -- `POST /api/show` for the same package, trimmed to the four fields
  the load test reads. Its `FROM` line is the evidence for the assignment rule: the blob
  `sha256-a86349…4ec1` is byte for byte the `Q4_K_M` file's LFS `oid` in
  `hf_unsloth_deepseek_r1_qwen3_8b_gguf_tree.json`. The real line names a folder under the user's
  home directory; the fixture says `/models/blobs/` instead.
- `ollama_generate_valid.json` -- one `POST /api/generate` after protocol v1 (`num_ctx` 8192):
  `done_reason: length`, `eval_count` 128 and the four other counters. The generated text is cut
  to its first 120 characters -- nothing in the package reads it. `tests/fixture_support.py`
  derives the other runs from this one by overriding `eval_duration` or `done_reason`.
- `hf_deepseek_r1_qwen3_8b_model.json` / `hf_deepseek_r1_qwen3_8b_config.json` -- the base model's
  card (`sha` `6e8885a6…25fa`) and its `config.json`: 36 layers, 8 key/value heads, head
  dimension 128 and no mixture-of-experts field, so the architecture is `dense_classic`.
- `hf_unsloth_deepseek_r1_qwen3_8b_gguf_model.json` /
  `hf_unsloth_deepseek_r1_qwen3_8b_gguf_tree.json` -- the GGUF repository's card (`sha`
  `eb48357c…f261`, relation `quantized` in both `cardData` and `tags`) and its file tree, with
  every `.gguf` file but the `Q4_K_M` one dropped; the remaining LFS `oid` is the digest the
  daemon shows. The cards keep only the fields the fetch reads (`siblings`, `spaces` and
  `widgetData` removed).
