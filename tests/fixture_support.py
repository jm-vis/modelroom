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
from modelroom.http import FixtureTransport, Response
from modelroom.profile import CrossCheck, HardwareProfile, LlmfitCrosscheck
from modelroom.state import atomic_write_json

FIXTURES = Path(__file__).parent / "fixtures"
PROFILE_ID = "3f9a0c21d4e6b870"
PROFILE_RECORDED_AT = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)


def json_response(name: str, status: int = 200, headers: dict[str, str] | None = None) -> Response:
    """A response whose body is exactly the named fixture file's bytes."""
    return Response(status=status, headers=headers or {}, body=(FIXTURES / name).read_bytes())


def html_response(name: str, status: int = 200, headers: dict[str, str] | None = None) -> Response:
    """A response whose body is exactly the named HTML fixture file's bytes."""
    return Response(status=status, headers=headers or {}, body=(FIXTURES / name).read_bytes())


def envelope_response(name: str) -> Response:
    """A response built from an envelope fixture: `{"status", "headers", "body"}`.

    `body` may be a JSON object/array (re-encoded as JSON bytes) or a plain string (encoded as
    UTF-8 text), or left out entirely for a fixture that only records headers (a `HEAD`).
    """
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    body = data.get("body", "")
    body_bytes = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
    return Response(status=data["status"], headers=data.get("headers", {}), body=body_bytes)


def build_transport(mapping: dict[tuple[str, str], Response]) -> FixtureTransport:
    return FixtureTransport(mapping)


def v2_profile(
    profile_id: str = PROFILE_ID,
    *,
    display_name: str = "workstation",
    vram_gib: float = 11.94,
    ram_gib: float = 127.46,
    recorded_at: datetime = PROFILE_RECORDED_AT,
) -> HardwareProfile:
    """A schema-2 hardware profile the fit computes for (`gpu_state: measured`), built in memory."""
    absent = CrossCheck(status="absent")
    return HardwareProfile(
        schema_version=2,
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
        llmfit_crosscheck=LlmfitCrosscheck(ram_physical=absent, vram=absent),
        llmfit_version=None,
    )


def place_v2_profile(hardware_dir: Path, **kwargs) -> HardwareProfile:
    """Write `v2_profile(**kwargs)` into `hardware_dir` under its own `profile_id`."""
    profile = v2_profile(**kwargs)
    atomic_write_json(hardware_dir / f"{profile.profile_id}.json", profile.model_dump(mode="json"))
    return profile


MACHINE_GUID = "4f2c1a68-9b03-4d5e-8a77-0c1de2f34567"
LAPTOP_RAM_BYTES = 137_112_547_328  # 127.70 GiB


def windows_runner(overrides: dict | None = None):
    """Every command a Windows/NVIDIA measurement makes, answered from the recorded fixtures."""
    import subprocess

    from modelroom.llmfit import FixtureRunner
    from modelroom.loadtest import GPU_UTILIZATION_ARGS
    from modelroom.measure import NVIDIA_SMI_ARGS, WINDOWS_MACHINE_GUID_ARGS

    def completed(args, stdout=""):
        return subprocess.CompletedProcess(list(args), 0, stdout=stdout, stderr="")

    guid = f"\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\n    MachineGuid    REG_SZ    {MACHINE_GUID}\n"
    responses = {
        NVIDIA_SMI_ARGS: completed(NVIDIA_SMI_ARGS, (FIXTURES / "nvidia_smi_one_gpu.csv").read_text(encoding="utf-8")),
        GPU_UTILIZATION_ARGS: completed(GPU_UTILIZATION_ARGS, "3\n"),
        WINDOWS_MACHINE_GUID_ARGS: completed(WINDOWS_MACHINE_GUID_ARGS, guid),
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


def search_transport_mapping(word: str = "qwen") -> dict[tuple[str, str], Response]:
    """Every page one search for `word` asks for, plus the two age lookups its resolution makes.

    This is the "fixed search answer" the acceptance run uses: criteria about resolved and
    unresolved hits must not depend on whichever repositories the live Hub answers with today
    (`tests/fixtures/README.md` records how the fixtures were taken). Both open pages are bound as
    well, so one mapping serves a run with the owner filter on and one with it off.
    """
    from modelroom.search_pages import account_search_url, most_downloaded_url, newest_url

    deepseek = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    mapping: dict[tuple[str, str], Response] = {
        ("GET", account_search_url(word, account)): json_response(name)
        for account, name in SEARCH_ACCOUNT_PAGES.items()
    }
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


DEEPSEEK_BASE = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
DEEPSEEK_REPO = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
DEEPSEEK_BASE_SHA = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
DEEPSEEK_SHA = "eb48357c179d34dbf515983f798dfb8752a0f261"
# The local name the Ollama daemon gives that package, its manifest digest in `/api/tags`, and
# the digest of the one GGUF blob it was built from -- all three read from a real daemon
# (2026-09-24) and pinned in `tests/fixtures/ollama_*`.
DEEPSEEK_OLLAMA_NAME = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
DEEPSEEK_MANIFEST_DIGEST = "sha256:ecc092d5e10a34f7277b5748cbb9e940c0eb2bbea563f48517ff804939ce3aec"
DEEPSEEK_WEIGHTS_DIGEST = "sha256:a86349a4180c4e6bb43f874c29c404fa2be3f90b15509bd6d86f697dba724ec1"
DEEPSEEK_WEIGHTS_BYTES = 5027785216
GRANITE_MANIFEST_DIGEST = "sha256:f586c02fdecdf151b656207c339aa003997345774a41768bac1fd6d2fb85913b"


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


def guided_transport_mapping() -> dict[tuple[str, str], Response]:
    """Every request one whole guided run makes: the search, its age lookups and the fetch."""
    return {**search_transport_mapping(), **qwen35_transport_mapping(), **deepseek_transport_mapping()}


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


# --- the Ollama daemon of this machine (AP9-E) --------------------------------------------------

DAEMON_OBSERVED_AT = datetime(2026, 9, 24, 8, 43, 0, tzinfo=timezone.utc)
# One warm-up run and three measured runs, each a little faster or slower than the next, so a
# test can tell mean, minimum and maximum apart.
MEASURED_DURATIONS = (2_000_000_000, 1_900_000_000, 2_100_000_000)


def generate_answer(eval_duration: int | None = None, done_reason: str | None = None) -> Response:
    """The pinned `/api/generate` answer, with its duration or its end reason overridden."""
    data = json.loads((FIXTURES / "ollama_generate_valid.json").read_text(encoding="utf-8"))
    if eval_duration is not None:
        data["eval_duration"] = eval_duration
    if done_reason is not None:
        data["done_reason"] = done_reason
    return Response(status=200, body=json.dumps(data).encode("utf-8"))


def ps_answer(digest: str | None = None, context_length: int | None = None, name: str | None = None) -> Response:
    """The pinned `/api/ps` answer for the loaded example package, with fields overridden."""
    data = json.loads((FIXTURES / "ollama_ps_deepseek_loaded.json").read_text(encoding="utf-8"))
    entry = data["models"][0]
    if digest is not None:
        entry["digest"] = digest
    if context_length is not None:
        entry["context_length"] = context_length
    if name is not None:
        entry["name"] = name
        entry["model"] = name
    return Response(status=200, body=json.dumps(data).encode("utf-8"))


def empty_ps_answer() -> Response:
    """What `/api/ps` answers while no model is loaded (read from a real daemon, 2026-09-24)."""
    return Response(status=200, body=b'{"models": []}')


def default_generates() -> list[Response]:
    """The four `/api/generate` answers of one protocol-v1 run: a warm-up and three measured."""
    return [generate_answer(), *(generate_answer(duration) for duration in MEASURED_DURATIONS)]


def loadtest_daemon(generates=None, observations=None, tags: str = "ollama_tags_local_loadtest.json"):
    """A `FixtureDaemon` that answers every call one whole load test makes.

    `generates` is the answer sequence of `/api/generate`, `observations` that of `/api/ps` --
    one observation after every run, so four by default.
    """
    from modelroom.daemon import GENERATE_PATH, PS_PATH, SHOW_PATH, TAGS_PATH, VERSION_PATH, FixtureDaemon

    return FixtureDaemon(
        {
            ("GET", VERSION_PATH): json_response("ollama_version_local.json"),
            ("GET", TAGS_PATH): json_response(tags),
            ("POST", SHOW_PATH): json_response("ollama_show_hf_gguf.json"),
            ("GET", PS_PATH): list(observations if observations is not None else [ps_answer()] * 4),
            ("POST", GENERATE_PATH): default_generates() if generates is None else list(generates),
        }
    )


def offline_daemon():
    """A machine with no Ollama daemon: every call fails with the reason a user would read."""
    from modelroom.daemon import FixtureDaemon
    from modelroom.ollama_local import DEFAULT_BASE_URL

    return FixtureDaemon({}, unreachable=f"{DEFAULT_BASE_URL}: the Ollama daemon did not answer (refused)")


def deepseek_package(**overrides):
    """The `Q4_K_M` package of the load test's example repository, as the fetch records it."""
    from modelroom.contracts import Package, PackageFile

    fields = {
        "source": "huggingface",
        "repo": DEEPSEEK_REPO,
        "revision": DEEPSEEK_SHA,
        "base_model_hf_repo": DEEPSEEK_BASE,
        "format": "gguf",
        "files": [
            PackageFile(
                name="DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
                role="weights",
                size_bytes=DEEPSEEK_WEIGHTS_BYTES,
                digest=DEEPSEEK_WEIGHTS_DIGEST,
            )
        ],
        "complete": True,
        "quantization": "Q4_K_M",
        "provenance": "metadata_ok",
        "observed_at": DAEMON_OBSERVED_AT,
        "last_seen": DAEMON_OBSERVED_AT,
        "active": True,
    }
    return Package(**{**fields, **overrides})


def granite_ollama_package(**overrides):
    """An Ollama package whose manifest digest is the one `/api/tags` shows for `granite4.2:8b`."""
    from modelroom.contracts import Package, PackageFile

    fields = {
        "source": "ollama",
        "ollama_name": "granite4.2:8b",
        "manifest_digest": GRANITE_MANIFEST_DIGEST,
        "base_model_hf_repo": DEEPSEEK_BASE,
        "format": "gguf",
        "files": [PackageFile(name="model", role="weights", size_bytes=5_347_929_757, digest=None)],
        "complete": True,
        "quantization": "Q4_K_M",
        "provenance": "metadata_ok",
        "observed_at": DAEMON_OBSERVED_AT,
        "last_seen": DAEMON_OBSERVED_AT,
        "active": True,
    }
    return Package(**{**fields, **overrides})
