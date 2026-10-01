"""Shared helpers for building fixture-backed `Response`/`FixtureTransport` values in tests.

Not a test module itself (no `test_` prefix, pytest never collects it). Two fixture shapes
are in use, both documented in `tests/fixtures/README.md`:

- a raw body file (the AP1 convention): the whole file is the response body, status `200`,
  no headers of interest -- `json_response`/`html_response`.
- an envelope file, `{"status": <int>, "headers": {...}, "body": <object-or-string>}`, for a
  fixture where the status or a header (a digest, a `Link` header) is itself the fact under
  test -- `envelope_response`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from modelroom.config import Configuration
from modelroom.http import Response
from modelroom.profile import PROFILE_SCHEMA_VERSION, CrossCheck, GpuAdapter, HardwareProfile, LlmfitCrosscheck
from modelroom.state import atomic_write_json
from modelroom.tensor_search import tensor_packagers, tensor_search_url

# Moved to their own modules (AP9-UX2, size guard) and exported back, so every test's import stays.
from fixture_http import FIXTURES, build_transport, envelope_response, html_response, json_response  # noqa: F401
from fixture_neutral import THIS_MACHINE  # noqa: F401  (this computer's key in every test, FR-N)
from fixture_servers import (  # noqa: F401
    DAEMON_OBSERVED_AT,
    DEEPSEEK_BASE,
    DEEPSEEK_BASE_SHA,
    DEEPSEEK_MANIFEST_DIGEST,
    DEEPSEEK_OLLAMA_NAME,
    DEEPSEEK_REPO,
    DEEPSEEK_SHA,
    DEEPSEEK_WEIGHTS_BYTES,
    DEEPSEEK_WEIGHTS_DIGEST,
    GRANITE_MANIFEST_DIGEST,
    MEASURED_DURATIONS,
    PULL_STREAM,
    PULL_STREAM_LINES,
    QWEN25_REPO,
    QWEN25_SHA,
    QWEN25_WEIGHTS_BYTES,
    QWEN25_WEIGHTS_DIGEST,
    VLLM_UNREACHABLE,
    default_generates,
    deepseek_package,
    empty_ps_answer,
    generate_answer,
    granite_ollama_package,
    loadtest_daemon,
    offline_daemon,
    offline_servers,
    ps_answer,
    pull_answer,
    pull_daemon,
    qwen25_tensor_package,
    qwen25_transport_mapping,
    tags_with,
    vllm_chat_answer,
    vllm_metrics,
    vllm_metrics_pages,
    vllm_servers,
)

PROFILE_ID = "3f9a0c21d4e6b870"
PROFILE_RECORDED_AT = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)

def v2_profile(
    profile_id: str = PROFILE_ID,
    *,
    display_name: str = "workstation",
    vram_gib: float = 11.94,
    ram_gib: float = 127.46,
    recorded_at: datetime = PROFILE_RECORDED_AT,
) -> HardwareProfile:
    """A hardware profile of the current schema the fit computes for (`gpu_state: measured`).

    Built in memory; the name keeps the profile model's family (v2 and later), not its schema.
    """
    absent = CrossCheck(status="absent")
    return HardwareProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        display_name=display_name,
        os_fingerprint="9d2f4b6a8c0e1357",
        os_fingerprint_source="windows_machineguid",
        origin="measured",
        recorded_at=recorded_at,
        ram_physical_gib=ram_gib,
        ram_physical_source="os",
        ram_limit_gib=None,
        ram_limit_scope="none",
        vram_gib=vram_gib,
        vram_source="nvidia-smi" if vram_gib else "none",
        gpu_state="measured" if vram_gib else "none",
        gpu_name="Nova GPU" if vram_gib else None,
        gpus=[GpuAdapter(index=0, name="Nova GPU", vram_gib=vram_gib, vram_source="nvidia-smi")] if vram_gib else [],
        machine_class="unknown",
        machine_class_source="unknown",
        llmfit_crosscheck=LlmfitCrosscheck(ram_physical=absent, vram=absent),
        llmfit_version=None,
    )


def v3_profile(schema_version: int = 3, **changes) -> dict:
    """A profile file exactly as version 0.1.0 wrote it: schema 3 (or 2), no card list, no class.

    Read from `tests/fixtures/profiles_v3/workstation.json`, a measured machine with one card; the
    changes are applied on top, the version last.
    """
    data = json.loads((FIXTURES / "profiles_v3" / "workstation.json").read_text(encoding="utf-8"))
    return {**data, **changes, "schema_version": schema_version}


def place_v2_profile(hardware_dir: Path, **kwargs) -> HardwareProfile:
    """Write `v2_profile(**kwargs)` into `hardware_dir` under its own `profile_id`."""
    profile = v2_profile(**kwargs)
    atomic_write_json(hardware_dir / f"{profile.profile_id}.json", profile.model_dump(mode="json"))
    return profile


MACHINE_GUID = "4f2c1a68-9b03-4d5e-8a77-0c1de2f34567"
LAPTOP_RAM_BYTES = 137_112_547_328  # 127.70 GiB
# The SMBIOS chassis type `Win32_SystemEnclosure` reports for the fixture laptop: 10, Notebook.
NOTEBOOK_CHASSIS = 10


def windows_gpu_responses() -> dict:
    """The graphics sources AP9-GPU added, answered from the laptop recording (`tests/fixtures/README.md`):
    the compute capability, the video controllers and the registry's memory sizes."""
    import subprocess

    from modelroom.measure import NVIDIA_CC_ARGS, WINDOWS_ADAPTER_ARGS, WINDOWS_REGISTRY_ARGS

    files = {
        NVIDIA_CC_ARGS: "nvidia_smi_compute_cap_laptop.csv",
        WINDOWS_ADAPTER_ARGS: "win_video_controllers_laptop.json.txt",
        WINDOWS_REGISTRY_ARGS: "win_registry_display_laptop.json.txt",
    }
    return {
        args: subprocess.CompletedProcess(list(args), 0, stdout=(FIXTURES / name).read_text(encoding="utf-8"), stderr="")
        for args, name in files.items()
    }


def windows_runner(overrides: dict | None = None):
    """Every command a Windows/NVIDIA measurement makes, answered from the recorded fixtures."""
    import subprocess

    from modelroom.llmfit import FixtureRunner
    from modelroom.loadtest import GPU_UTILIZATION_ARGS
    from modelroom.measure import NVIDIA_SMI_ARGS, WINDOWS_CHASSIS_ARGS, WINDOWS_MACHINE_GUID_ARGS

    def completed(args, stdout=""):
        return subprocess.CompletedProcess(list(args), 0, stdout=stdout, stderr="")

    guid = f"\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\n    MachineGuid    REG_SZ    {MACHINE_GUID}\n"
    responses = {
        **windows_gpu_responses(),
        NVIDIA_SMI_ARGS: completed(NVIDIA_SMI_ARGS, (FIXTURES / "nvidia_smi_one_gpu.csv").read_text(encoding="utf-8")),
        GPU_UTILIZATION_ARGS: completed(GPU_UTILIZATION_ARGS, "3\n"),
        WINDOWS_MACHINE_GUID_ARGS: completed(WINDOWS_MACHINE_GUID_ARGS, guid),
        WINDOWS_CHASSIS_ARGS: completed(WINDOWS_CHASSIS_ARGS, f"{NOTEBOOK_CHASSIS}\n"),
        ("llmfit", "--version"): completed(("llmfit", "--version"), "llmfit 1.1.16\n"),
        ("llmfit", "system", "--json"): completed(
            ("llmfit", "system", "--json"), (FIXTURES / "llmfit_system_laptop.json").read_text(encoding="utf-8")
        ),
    }
    responses.update(overrides or {})
    return FixtureRunner(responses)


def windows_probes(host: str = "workstation", ids: list[str] | None = None, runner=None):
    """`Probes` for a Windows/NVIDIA machine, every source a fixture -- no process is spawned.

    `ids` pins the profile ids a run draws, in order; without it the ids count up
    (`0000000000000001`, ...) so a caller never has to know how many a step draws.
    """
    from itertools import count

    from modelroom.measure import FixtureFiles, Probes

    remaining = list(ids or [])
    counter = count(1)
    return Probes(
        platform="win32",
        runner=runner if runner is not None else windows_runner(),
        read_text=FixtureFiles({}),
        memory_bytes=lambda: LAPTOP_RAM_BYTES,
        hostname=lambda: host,
        new_id=lambda: remaining.pop(0) if remaining else f"{next(counter):016x}",
    )


def with_profile(config: Configuration, machine: str, profile_id: str) -> Configuration:
    """The same configuration with `[machines.<machine>].profile` set -- what a guided run writes."""
    data = config.model_dump(mode="json")
    data["machines"][machine]["profile"] = profile_id
    return Configuration.from_dict(data)


QWEN_SHA = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
UNSLOTH_SHA = "3885219b6810b007914f3a7950a8d1b469d598a5"

_SMALL_QWEN35_TAGS_HTML = (
    b'<html><body>'
    b'<a href="/library/qwen3.5:9b">9b</a>'
    b'<a href="/library/qwen3.5:9b-q4_K_M">9b-q4_K_M</a>'
    b'<a href="/library/qwen3.5:9b-mlx-bf16">9b-mlx-bf16</a>'
    b'</body></html>'
)


def qwen35_transport_mapping() -> dict[tuple[str, str], Response]:
    """Every request a full `fetch` run needs for the example `qwen3.5` family to complete.

    Covers: the Qwen/Qwen3.5-9B publisher repo (meta), the unsloth GGUF packager repo, the
    base model's own owner probed as a packager of its own models (not found), and the three
    Ollama qwen3.5 manifests. Real fixtures throughout except the tags page, which is a small
    hand-written HTML snippet (see tests/test_ollama.py for why: the real page lists many
    sizes this test has no manifest fixtures for).
    """

    def manifest_url(tag: str) -> str:
        return f"https://registry.ollama.ai/v2/library/qwen3.5/manifests/{tag}"

    return {
        ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"): json_response("hf_qwen_qwen35_9b_model.json"),
        (
            "GET",
            f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{QWEN_SHA}/config.json",
        ): json_response("hf_qwen_qwen35_9b_config.json"),
        ("GET", "https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF"): json_response(
            "hf_unsloth_qwen35_9b_gguf_model.json"
        ),
        (
            "GET",
            f"https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF/tree/{UNSLOTH_SHA}?recursive=true",
        ): json_response("hf_unsloth_qwen35_9b_gguf_tree.json"),
        ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B-GGUF"): envelope_response(
            "hf_unsloth_does_not_exist_model.json"
        ),
        ("GET", "https://ollama.com/library/qwen3.5/tags"): Response(
            status=200, headers={}, body=_SMALL_QWEN35_TAGS_HTML
        ),
        ("GET", manifest_url("9b")): json_response("ollama_qwen35_9b.json"),
        ("GET", manifest_url("9b-q4_K_M")): json_response("ollama_qwen35_9b-q4_K_M.json"),
        ("GET", manifest_url("9b-mlx-bf16")): json_response("ollama_qwen35_9b-mlx-bf16.json"),
    }


# Which account page answers which fixture for the word `qwen` (decided 2026-09-24: one request
# per account, and two open pages only when the owner filter is off). Every account the search
# asks has an entry here, so a page is never silently missing from a fixture run.
SEARCH_ACCOUNT_PAGES: dict[str, str] = {
    "Qwen": "hf_search_qwen_publisher.json",
    "deepseek-ai": "hf_search_none.json",
    "unsloth": "hf_search_qwen_unsloth.json",
    "bartowski": "hf_search_none.json",
    "mradermacher": "hf_search_none.json",
    "lmstudio-community": "hf_search_none.json",
    "ggml-org": "hf_search_none.json",
}
SEARCH_OPEN_PAGES: dict[str, str] = {
    "most downloaded": "hf_search_qwen_most_downloaded.json",
    "newest": "hf_search_qwen_newest.json",
}

# The same for the word `mistral` (decided 2026-09-25, when `mistralai` became a publisher of the
# shipped catalog): the publisher's page and the `unsloth` page answer with their own pinned
# entries, every other asked account with the empty list.
SEARCH_MISTRAL_PAGES: dict[str, str] = {
    "mistralai": "hf_search_mistral_publisher.json",
    "unsloth": "hf_search_mistral_unsloth.json",
}
# The base models those two pages resolve to, and therefore the age lookups such a run makes.
MISTRAL_BASE_MODELS: tuple[str, ...] = (
    "mistralai/Ministral-3-14B-Instruct-2512",
    "mistralai/Magistral-Small-2509",
    "mistralai/Mistral-Small-4-119B-2603",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
)


def search_transport_mapping(word: str = "qwen") -> dict[tuple[str, str], Response]:
    """Every page one search for `word` asks for, plus the two age lookups its resolution makes.

    This is the "fixed search answer" the acceptance run uses: criteria about resolved and
    unresolved hits must not depend on whichever repositories the live Hub answers with today
    (`tests/fixtures/README.md` records how the fixtures were taken). Both open pages are bound as
    well, so one mapping serves a run with the owner filter on and one with it off.

    Since 2026-09-25 an account is asked twice, newest and most downloaded, and **both** sort
    orders are bound to that account's one curated answer: the curation is about which
    repositories an account has, not about the order they come back in, and a duplicate is exactly
    what the second page of an account usually is. What one sort order finds and the other does not
    is a case of its own (`tests/test_search.py`, with the recorded two-entry downloads page).
    """
    from modelroom.search_pages import account_downloads_url, account_search_url, most_downloaded_url, newest_url

    deepseek = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    mapping: dict[tuple[str, str], Response] = {}
    for account, name in SEARCH_ACCOUNT_PAGES.items():
        mapping[("GET", account_search_url(word, account))] = json_response(name)
        mapping[("GET", account_downloads_url(word, account))] = json_response(name)
    mapping[("GET", most_downloaded_url(word))] = json_response(SEARCH_OPEN_PAGES["most downloaded"])
    mapping[("GET", newest_url(word))] = json_response(SEARCH_OPEN_PAGES["newest"])
    mapping[("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B")] = json_response(
        "hf_qwen_qwen35_9b_model.json"
    )
    mapping[("GET", f"https://huggingface.co/api/models/{deepseek}")] = Response(
        status=200,
        headers={},
        body=json.dumps({"id": deepseek, "sha": "b" * 40, "cardData": {"license": "mit"}}).encode("utf-8"),
    )
    return mapping


# The three cases of the input forms of 2026-09-25, all recorded live that day.
TYPED_GGUF_REPO = "unsloth/Qwen3.5-9B-GGUF"  # holds GGUF files: the `gguf` tag is on it
TYPED_NO_GGUF_REPO = "Qwen/Qwen3.5-9B"  # the base model itself: tensors, no `gguf` tag
# The one current catalog model whose page answers with entries. `Qwen3.8-27B` and not
# `Qwen3.5-9B`: a page per **current** model is what a run with no word asks, and the catalog's
# current Qwen model is that one.
CATALOG_PAGE_MODEL = "Qwen3.8-27B"
CATALOG_PAGE_REPO = "unsloth/Qwen3.8-27B-GGUF"
# A repository of that page whose base model belongs to an account the catalog does not name as a
# publisher: `publisher_unknown` with the owner filter on, resolved with `other` when it is off.
CATALOG_PAGE_FOREIGN_REPO = (
    "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
)
# The base models the recorded catalog page declares, and therefore the age lookups it costs.
CATALOG_PAGE_BASE_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3.8-27B",
    "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU",
)


def _card_answer(repo: str, license_name: str = "apache-2.0") -> Response:
    """The answer one age lookup reads: the repository names itself and says no `new_version`."""
    body = json.dumps({"id": repo, "sha": "d" * 40, "cardData": {"license": license_name}})
    return Response(status=200, headers={}, body=body.encode("utf-8"))


def typed_transport_mapping() -> dict[tuple[str, str], Response]:
    """The one page a typed repository id asks for, both cases, plus the age lookup it costs."""
    from modelroom.search_pages import model_url

    return {
        ("GET", model_url(TYPED_GGUF_REPO)): json_response("hf_typed_unsloth_qwen35_9b_gguf.json"),
        ("GET", model_url(TYPED_NO_GGUF_REPO)): json_response("hf_typed_qwen_qwen35_9b.json"),
        ("GET", f"https://huggingface.co/api/models/{TYPED_NO_GGUF_REPO}"): json_response(
            "hf_qwen_qwen35_9b_model.json"
        ),
    }


def catalog_transport_mapping() -> dict[tuple[str, str], Response]:
    """One page per current model of the shipped catalog: one answers, every other with nothing.

    Read from the catalog rather than written out here, so a current model added later is bound too
    instead of ending the run at a URL no fixture answers.
    """
    from modelroom.catalog import load_catalog
    from modelroom.search_pages import catalog_page_url
    from modelroom.search_word import latest_model_names

    mapping: dict[tuple[str, str], Response] = {}
    for name in latest_model_names(load_catalog()):
        page = (
            json_response("hf_search_qwen38_27b_catalog_page.json")
            if name == CATALOG_PAGE_MODEL
            else json_response("hf_search_none.json")
        )
        mapping[("GET", catalog_page_url(name))] = page
    for base in CATALOG_PAGE_BASE_MODELS:
        mapping[("GET", f"https://huggingface.co/api/models/{base}")] = _card_answer(base)
    return mapping


def mistral_search_mapping(word: str = "mistral") -> dict[tuple[str, str], Response]:
    """Every page one search for `mistral` asks for, plus one age lookup per resolved base model.

    The asked accounts are read from the shipped catalog rather than written out here, so a
    publisher family added later whose name, account or model ids carry the word is bound too,
    instead of ending the run at a URL no fixture answers.
    """
    from modelroom.catalog import load_catalog
    from modelroom.search import DEFAULT_PACKAGERS
    from modelroom.search_pages import account_downloads_url, account_search_url, search_accounts

    accounts = search_accounts(load_catalog(), word, DEFAULT_PACKAGERS)
    mapping: dict[tuple[str, str], Response] = {}
    for account in accounts:
        page = json_response(SEARCH_MISTRAL_PAGES.get(account, "hf_search_none.json"))
        mapping[("GET", account_search_url(word, account))] = page
        mapping[("GET", account_downloads_url(word, account))] = page
    for base in MISTRAL_BASE_MODELS:
        mapping[("GET", f"https://huggingface.co/api/models/{base}")] = Response(
            status=200,
            headers={},
            body=json.dumps({"id": base, "sha": "c" * 40, "cardData": {"license": "apache-2.0"}}).encode("utf-8"),
        )
    return mapping




def deepseek_transport_mapping() -> dict[tuple[str, str], Response]:
    """Every request the fetch of the load test's example package makes.

    The package is the one the acceptance run measures against the real local daemon, so its
    Hugging Face answers are pinned from the live Hub (2026-09-24) rather than hand-written:
    the base model's card and `config.json` (a dense architecture fit v1 can judge), the GGUF
    repository's card and file tree with the `Q4_K_M` file's `sha256`, and the publisher's own
    `-GGUF` repository, which does not exist.
    """
    return {
        ("GET", f"https://huggingface.co/api/models/{DEEPSEEK_BASE}"): json_response(
            "hf_deepseek_r1_qwen3_8b_model.json"
        ),
        (
            "GET",
            f"https://huggingface.co/{DEEPSEEK_BASE}/resolve/{DEEPSEEK_BASE_SHA}/config.json",
        ): json_response("hf_deepseek_r1_qwen3_8b_config.json"),
        ("GET", f"https://huggingface.co/api/models/{DEEPSEEK_REPO}"): json_response(
            "hf_unsloth_deepseek_r1_qwen3_8b_gguf_model.json"
        ),
        (
            "GET",
            f"https://huggingface.co/api/models/{DEEPSEEK_REPO}/tree/{DEEPSEEK_SHA}?recursive=true",
        ): json_response("hf_unsloth_deepseek_r1_qwen3_8b_gguf_tree.json"),
        ("GET", f"https://huggingface.co/api/models/{DEEPSEEK_BASE}-GGUF"): envelope_response(
            "hf_unsloth_does_not_exist_model.json"
        ),
    }


def publisher_tree_mapping() -> dict[tuple[str, str], Response]:
    """The file trees of the two publisher repositories a guided run now fetches (AP9-UX2 P3, F8).

    Every resolved base model becomes a `repos` target of its own, so the fetch reads the tree of
    `Qwen/Qwen3.5-9B` and of `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` at the shas their recorded model
    infos name (public Hub recordings, `tests/fixtures/README.md`).
    """
    api = "https://huggingface.co/api/models"
    return {
        ("GET", f"{api}/Qwen/Qwen3.5-9B/tree/{QWEN_SHA}?recursive=true"): json_response("hf_qwen_qwen35_9b_tree.json"),
        ("GET", f"{api}/{DEEPSEEK_BASE}/tree/{DEEPSEEK_BASE_SHA}?recursive=true"): json_response(
            "hf_deepseek_r1_qwen3_8b_tree.json"
        ),
    }


QWEN35_4B_REPO = "Qwen/Qwen3.5-4B"
QWEN35_4B_SHA = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def qwen35_4b_transport_mapping() -> dict[tuple[str, str], Response]:
    """The Hub answers a fetch of `Qwen/Qwen3.5-4B` as its own `repos` target needs (AP9-UX2 P3)."""
    api = "https://huggingface.co/api/models"
    return {
        ("GET", f"{api}/{QWEN35_4B_REPO}"): json_response("hf_qwen_qwen35_4b_model.json"),
        ("GET", f"https://huggingface.co/{QWEN35_4B_REPO}/resolve/{QWEN35_4B_SHA}/config.json"): json_response(
            "hf_qwen_qwen35_4b_config.json"
        ),
        ("GET", f"{api}/{QWEN35_4B_REPO}/tree/{QWEN35_4B_SHA}?recursive=true"): json_response("hf_qwen_qwen35_4b_tree.json"),
    }


def no_tensor_hits_mapping() -> dict[tuple[str, str], Response]:
    """Every tensor-search page (P1, `tensor_search.search_tensor_repos`) a run
    with Qwen3.5-9B and/or the DeepSeek base model resolved could ask, answered with an empty
    list. the tensor search asks whenever a chosen machine has vLLM -- by its profile's `runtimes`, *or*
    a vLLM server detected live in the very same run: a guided run with a live-serving vLLM fixture (`fixture_servers.vllm_servers`)
    reaches this search even where its profile carries no stored `runtimes` at all, so every such
    run needs these two accounts answered, not only the ones that set `runtimes` explicitly.
    Empty answers keep every existing assertion about what a run writes unchanged.
    """
    empty = Response(status=200, headers={}, body=b"[]")
    bases = {
        "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
        DEEPSEEK_BASE: "DeepSeek-R1-0528-Qwen3-8B",
        QWEN25_REPO: "Qwen2.5-0.5B-Instruct",
    }
    return {
        ("GET", tensor_search_url(name, account)): empty
        for base, name in bases.items()
        for account in tensor_packagers(base)
    }


def guided_transport_mapping() -> dict[tuple[str, str], Response]:
    """Every request one whole guided run makes: the search, its age lookups and the fetch."""
    return {
        **search_transport_mapping(),
        **qwen35_transport_mapping(),
        **deepseek_transport_mapping(),
        **publisher_tree_mapping(),
        **no_tensor_hits_mapping(),
    }


def packager_probes(config: Configuration) -> dict[tuple[str, str], Response]:
    """An answer for every packager repository a fetch of `config` asks about: "does not exist".

    For every base model of `config` and every target of `package_targets` one entry, answered
    with the recorded 401 envelope (`hf_unsloth_does_not_exist_model.json`, public and in the repo),
    which the fetch reads as "not there": the owner's area is then `complete` with no package
    (AP9-UX, Review r3-H1). Put it **first** when mappings are merged, so a bound answer wins:
    `{**packager_probes(config), **guided_transport_mapping(), **<the test's own>}`.
    """
    from modelroom.config import package_targets

    return {
        ("GET", f"https://huggingface.co/api/models/{target}"): envelope_response("hf_unsloth_does_not_exist_model.json")
        for family in config.families
        for base_model in family.base_models
        for target in package_targets(config, base_model)
    }


def example_without_eurollm() -> dict:
    """`modelroom.example.toml` as a dict, with the family `eurollm` taken out and nothing else changed.

    No recording of EuroLLM is in the repo, and a `FixtureTransport` raises at any URL it does not
    bind; everything else stays: two families, five packagers, the machines `workstation` (the
    writer) and `inference-server` (no profile). A test writes it as `modelroom.toml` with
    `write_example_without_eurollm` (AP9-UX, "package repo and fixtures").
    """
    import tomllib

    data = tomllib.loads((Path(__file__).parents[1] / "modelroom.example.toml").read_text(encoding="utf-8"))
    data["families"] = [family for family in data["families"] if family["name"] != "eurollm"]
    return data


EXAMPLE_WITHOUT_EUROLLM = example_without_eurollm


def expected_hf_areas(results: Path) -> set[tuple[str, str]]:
    """Every (base model, owner) the fetch of the written configuration asks the Hub about, computed."""
    from modelroom.config import load_config, package_targets
    from modelroom.hf import target_owners

    config = load_config(results / "modelroom.toml")
    bases = [base for family in config.families for base in family.base_models]
    return {(base.hf_repo, owner) for base in bases for owner in target_owners(package_targets(config, base))}


def area_problems(results: Path) -> list[str]:
    """Why the snapshot of `results` is not complete in every area; empty where it is (AP9-UX).

    Every area `complete` without an error, the Hugging Face areas exactly (base model, owner) for
    every base model of the written configuration and every owner of its targets, the Ollama areas
    exactly the base models with `ollama_base` and `ollama_tag`. A fetch cut short by an unbound
    address or a spent budget shows up here, before any other check of a test.
    """
    from modelroom.config import load_config
    from modelroom.state import read_snapshot

    config = load_config(results / "modelroom.toml")
    snapshot = read_snapshot(config)
    if snapshot is None:
        return ["no snapshot"]
    problems = [
        f"{area.source} {area.base_model_hf_repo} {area.packager}: {area.status} ({area.error})"
        for area in snapshot.areas
        if area.status != "complete" or area.error is not None
    ]
    found = {(area.base_model_hf_repo, area.packager) for area in snapshot.areas if area.source == "huggingface"}
    if found != expected_hf_areas(results):
        problems.append(f"huggingface areas {sorted(found)} are not {sorted(expected_hf_areas(results))}")
    bases = [base for family in config.families for base in family.base_models]
    wanted = {base.hf_repo for base in bases if base.ollama_base and base.ollama_tag}
    ollama = {area.base_model_hf_repo for area in snapshot.areas if area.source == "ollama"}
    if ollama != wanted:
        problems.append(f"ollama areas {sorted(ollama)} are not {sorted(wanted)}")
    return problems


def write_example_without_eurollm(folder: Path) -> Path:
    """Write `example_without_eurollm()` as `<folder>/modelroom.toml`, as bytes; returns the path."""
    from modelroom.toml_writer import dump_toml

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "modelroom.toml"
    path.write_bytes(dump_toml(example_without_eurollm()).encode("utf-8"))
    return path


# The three recorded or derived server profiles (`tests/fixtures/profiles_v4`), by `profile_id`.
TWO_A100_ID = "0291d4f706c0003c"
TWO_L4_ID = "6a2f8beef0d71f93"
FOUR_L4_ID = "4c7e19a2b5d8f036"
_EXPORT_CONFIG = """schema_version = 3

packagers = []
publishers = []

[paths]
state = "state"
markdown = "docs/models.md"
"""


def export_of(name: str, folder: Path) -> Path:
    """An export file of the profile fixture `profiles_v4/<name>`, made with the package's own export.

    The import takes an export file (`importer.import_profile`), so the fixture is placed in a
    scratch results folder under `folder` and exported from there by `importer.export_profile`.
    """
    import shutil

    from modelroom.importer import export_profile

    source = FIXTURES / "profiles_v4" / name
    profile_id = json.loads(source.read_text(encoding="utf-8"))["profile_id"]
    home = folder / f"export-{profile_id}"
    (home / "state" / "hardware").mkdir(parents=True)
    shutil.copyfile(source, home / "state" / "hardware" / f"{profile_id}.json")
    (home / "modelroom.toml").write_bytes(_EXPORT_CONFIG.encode("utf-8"))
    out = folder / f"{profile_id}.export.json"
    export_profile(home / "modelroom.toml", out, profile_id, home / "pointer.json")
    return out


def qwen35_example_config_dict(state_dir: str, markdown_path: str) -> dict:
    """A minimal Configuration payload for the `qwen3.5` family, matching modelroom.example.toml."""
    return {
        "schema_version": 1,
        "families": [
            {
                "name": "qwen3.5",
                "base_models": [
                    {
                        "hf_repo": "Qwen/Qwen3.5-9B",
                        "repo_aliases": ["Qwen3.5-9B-GGUF"],
                        "ollama_base": "qwen3.5",
                        "ollama_tag": "9b",
                    }
                ],
            }
        ],
        "packagers": ["unsloth"],
        "publishers": ["Qwen"],
        "machines": {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True}},
        "paths": {"state": state_dir, "markdown": markdown_path},
    }
