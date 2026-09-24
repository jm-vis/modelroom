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
    from modelroom.measure import NVIDIA_SMI_ARGS, WINDOWS_MACHINE_GUID_ARGS

    def completed(args, stdout=""):
        return subprocess.CompletedProcess(list(args), 0, stdout=stdout, stderr="")

    guid = f"\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\n    MachineGuid    REG_SZ    {MACHINE_GUID}\n"
    responses = {
        NVIDIA_SMI_ARGS: completed(NVIDIA_SMI_ARGS, (FIXTURES / "nvidia_smi_one_gpu.csv").read_text(encoding="utf-8")),
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


def search_transport_mapping() -> dict[tuple[str, str], Response]:
    """The pinned search answer for `qwen`, plus the two age lookups its resolution makes.

    This is the "fixed search answer" the acceptance run uses: criteria about resolved and
    unresolved hits must not depend on whichever 50 repositories the live Hub answers with today
    (`tests/fixtures/README.md` records how the fixture was taken).
    """
    from modelroom.search import search_url

    deepseek = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    return {
        ("GET", search_url("qwen")): json_response("hf_search_qwen_gguf.json"),
        ("GET", "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"): json_response("hf_qwen_qwen35_9b_model.json"),
        ("GET", f"https://huggingface.co/api/models/{deepseek}"): Response(
            status=200,
            headers={},
            body=json.dumps({"id": deepseek, "sha": "b" * 40, "cardData": {"license": "mit"}}).encode("utf-8"),
        ),
    }


def guided_transport_mapping() -> dict[tuple[str, str], Response]:
    """Every request one whole guided run makes: the search, its age lookups and the fetch."""
    return {**search_transport_mapping(), **qwen35_transport_mapping()}


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
