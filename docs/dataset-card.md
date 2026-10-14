---
license: cc-by-4.0
pretty_name: Open Model Catalog
language:
  - en
size_categories:
  - n<1K
task_categories:
  - text-generation
  - image-text-to-text
  - feature-extraction
  - automatic-speech-recognition
  - text-to-image
  - text-to-video
tags:
  - open-weights
  - model-catalog
  - publishers
  - gguf
  - ollama
  - llm
  - vision-language
  - embeddings
configs:
  - config_name: default
    data_files:
      - catalog-v2.json
      - catalog-v3.json
      - catalog-v4.json
---

![Open Model Catalog](catalog-banner.png)

# Open Model Catalog

A small table of the models their publishers call current: text, vision, embedding, speech
recognition, image and video generation. Not every build on the Hub. One row per model, with the
publisher page that says it is current and the date that was checked.

**252 models · 57 publisher accounts watched · one JSON file (243 KB) · schema 2, and schemas 3 and 4 with findings · CC BY 4.0**

## Three files, three schemas

The dataset holds the same table three times, one file per schema, so that each release of the reader finds the file it knows:

| File | Schema | Read by |
|---|---|---|
| `catalog-v2.json` | 2 | release 1.0.0; unchanged in form, it carries no findings |
| `catalog-v3.json` | 3 | release 1.1.0; the rows of schema 2, plus `gguf_checks` and `setup_note` (a note without `vllm_args`) |
| `catalog-v4.json` | 4 | releases that read the findings and the switches below; schema 3, plus `vllm_args` in a `setup_note` |

The files are made from one working table and published together; the published files are compared byte for byte with
the repository every day. A row of schema 3 or 4 without findings reads as "not checked".

## What a row carries

| Field | Meaning |
|---|---|
| `hf_repo`, `publisher`, `family`, `kind` | the repository, its account, the model family, and one of eight kinds: `text`, `vision`, `embedding`, `speech-recognition`, `speech-synthesis`, `image-generation`, `video-generation`, `audio` |
| `parameters_b`, `license` | parameter count in billions, the license the card states |
| `latest`, `latest_source`, `latest_checked` | whether the publisher calls it current, the page that says so, and the date that was checked |
| `successor`, `successor_source`, `successor_checked` | the model that replaced it, when the publisher named one |
| `ollama_base`, `ollama_tag` | the Ollama library entry, where one exists |
| `gguf_repos` | GGUF builds of five packagers: Unsloth, bartowski, mradermacher, LM Studio, ggml |
| `hub_pipeline_tag`, `kind_source` | the task the Hub states, and where the kind came from (the tag, the base model, the config, or a decision) |
| `source`, `added`, `note` | the page the row was read from, the day it was added, a short note where a decision needed one |

Schema 3 (`catalog-v3.json`) adds two findings to a row, and schema 4 (`catalog-v4.json`) one field to the second. Each says what
was compared and when, and nothing else.

| Field | Meaning |
|---|---|
| `gguf_checks` | one entry per GGUF package, bound to its repository, revision and weight file (or first shard): the architecture the file states, compared with the `LLM_ARCH_NAMES` block of the `src/llama-arch.cpp` file of one llama.cpp commit. `present` or `absent`, the commit, the page, the day, and where the architecture was read (the Hub's field for the one file it describes, else the first bytes of the file) |
| `setup_note` | a pointer to the model card of the publisher's repository at one revision: whether the card holds a `vllm serve` start block for the model, and the name of a package the card states as required for inference, where its wording is explicit |
| `setup_note.vllm_args` | schema 4 only: the switches that the start block names for this model, of a closed list (`--reasoning-parser`, `--tool-call-parser`, `--enable-auto-tool-choice`, `--kv-cache-dtype`), each with a value of letters, digits, `.`, `_` and `-`; nothing else of a start block (no path, no `--trust-remote-code`, no memory or GPU switch) is kept |

What these findings do not say: `present` is no promise that a runtime loads the file, and `absent` is a statement about one
package against one commit of one source, never about a runtime version. A package or a card without an entry was not
checked or could not be read; a failed run adds no `absent` and removes no entry. A `setup_note` does not say the card
lists everything a model needs, and `none_recognized` says only that the rule of the scan found nothing. The scan reads
the cards and file headers; it runs nothing from them.

The file also lists the `publishers` (account, display name, region such as `EU`, `CN`, `US`, `KR`,
`JP`) and the `packagers` whose GGUF builds are recorded.

## How it stays current

Every morning a scan reads the 57 publisher accounts and the trending page of each task and
writes a proposal; a part of the catalog rows per day also gets its GGUF packages and model card checked (the findings above). Nothing goes live on its own: the maintainer approves every row by number,
rejected accounts stay rejected, and the published file is compared byte for byte with the
repository every day.

The limit is the maintainer. The scan finds what the Hub shows; whether a row is right is one
person's judgement, with the page that supports it in the row.

## Use it

```python
from datasets import load_dataset

catalog = load_dataset("jm-vis/modelroom-catalog")
```

Or read the file directly:

```python
import json, urllib.request

url = "https://huggingface.co/datasets/jm-vis/modelroom-catalog/resolve/main/catalog-v2.json"
catalog = json.load(urllib.request.urlopen(url))
current = [m for m in catalog["models"] if m["latest"]]
```

The files with the findings are `catalog-v3.json` and `catalog-v4.json`, at the same address.

The first consumer is [ModelRoom](https://github.com/jm-vis/modelroom), which reads the catalog at
the start of each run to answer "which model fits this machine".

## Something missing?

Open an issue in the [ModelRoom repository](https://github.com/jm-vis/modelroom/issues) naming the
account and the page that names its current model. A wrong row: the page that says otherwise.

## License and attribution

The data is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use it,
copy it, change it, build on it, commercially too; name the source:

> modelroom catalog, jm-vis, CC BY 4.0, https://huggingface.co/datasets/jm-vis/modelroom-catalog

The models listed keep their own licenses; the `license` field states what each model card says.
