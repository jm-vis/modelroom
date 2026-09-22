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
from pathlib import Path

from modelroom.http import FixtureTransport, Response

FIXTURES = Path(__file__).parent / "fixtures"


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
