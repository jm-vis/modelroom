"""Shared helpers of the P4 tests (AP9-KO, catalog run): a GGUF header, the recorded llama.cpp source, a Hub that answers
the requests of stage 1 and of the setup notes, and a transport that remembers the headers it was given. Not a test
module (no `test_` prefix). Everything here is synthetic after the real pattern; accounts and models are invented."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the scanner is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))

from modelroom.http import FixtureTransport, Response  # noqa: E402

LLAMA_COMMIT = "45451c366d4256f94fbc8845645e31a6b9155054"
LLAMA_TEXT = (REPO / "docs" / "ap9-ko" / "feasibility" / f"llama-arch.cpp.{LLAMA_COMMIT}.txt").read_bytes()
GITHUB_COMMIT_URL = "https://api.github.com/repos/ggml-org/llama.cpp/commits/HEAD"
GITHUB_BLOCK_URL = f"https://api.github.com/repos/ggml-org/llama.cpp/contents/src/llama-arch.cpp?ref={LLAMA_COMMIT}"
CDN = "https://us.aws.cdn.hf.co/xet-bridge-us/abc"
KNOWN = "llama"  # an identifier the recorded block holds
UNKNOWN = "nova7"  # one it does not hold
ROWS = ("acme/Atlas-1-9B", "acme/Atlas-2-9B", "nova/Nimbus-1-30B", "orbit/Comet-3-8B")  # the text and vision rows of scan_catalog.json
SHA_A, SHA_B = "a" * 40, "b" * 40


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def gguf_head(architecture: str, before: tuple[bytes, ...] = (), version: int = 3) -> bytes:
    """The start of a GGUF file: magic, version, counts, the key-value pairs `before`, then `general.architecture`."""
    arch = _string("general.architecture") + struct.pack("<I", 8) + _string(architecture)
    pairs = (*before, arch)
    return b"GGUF" + struct.pack("<IQQ", version, 0, len(pairs)) + b"".join(pairs)


def kv_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_string_array(key: str, items: list[str]) -> bytes:
    return _string(key) + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(items)) + b"".join(_string(i) for i in items)


def llama_with(*names: str) -> bytes:
    """The recorded source with more architectures in the block (before the sentinel), as a new commit would have them."""
    text = LLAMA_TEXT.decode("utf-8")
    marker = '    { LLM_ARCH_UNKNOWN,'
    assert marker in text
    extra = "".join(f'    {{ LLM_ARCH_X_{n.upper().replace("-", "_")}, "{n}" }},\n' for n in names)
    return text.replace(marker, extra + marker, 1).encode("utf-8")


class Recording(FixtureTransport):
    """A fixture transport that keeps the headers of every call as well."""

    def __init__(self, responses) -> None:
        super().__init__(responses)
        self.headers: list[dict | None] = []

    def __call__(self, method, url, headers=None, *, max_bytes=None):
        self.headers.append(headers)
        return super().__call__(method, url, headers, max_bytes=max_bytes)


def put(hub, url: str, body, status: int = 200, headers: dict | None = None) -> None:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    hub.m[("GET", url)] = Response(status=status, headers=headers or {}, body=raw)


def source(hub, text: bytes = LLAMA_TEXT, commit: str = LLAMA_COMMIT) -> None:
    put(hub, GITHUB_COMMIT_URL, commit.encode("ascii"))
    put(hub, f"https://api.github.com/repos/ggml-org/llama.cpp/contents/src/llama-arch.cpp?ref={commit}", text)


def listing(hub, row: str, entries: list) -> None:
    import catalog_scan_packages as K

    put(hub, K.list_url(row), entries)


def tags_of(*bases: str, gguf: bool = True) -> list[str]:
    """The tags the Hub gives a quantization: the base model in both forms of the relation."""
    return [*(["gguf"] if gguf else []), *(f"base_model:{b}" for b in bases), *(f"base_model:quantized:{b}" for b in bases), "region:us"]


def entry(repo: str, base: str, sha: str, *, architecture: str | None = None, total: int | None = None, tags: list | None = None) -> dict:
    gguf = {} if architecture is None else {"architecture": architecture, "totalFileSize": total}
    return {"id": repo, "sha": sha, "tags": tags if tags is not None else tags_of(base), "gguf": gguf,
            "createdAt": "2026-10-01T00:00:00.000Z", "likes": 1, "downloads": 10}


def tree(hub, repo: str, sha: str, files: list[tuple[str, int]]) -> None:
    import catalog_scan_packages as K

    put(hub, K.tree_url(repo, sha), [{"type": "file", "path": path, "size": size} for path, size in files])


def header(hub, repo: str, sha: str, file: str, architecture: str | None, *, redirect: bool = True, raw: bytes | None = None) -> None:
    """The ranged read of a file's first bytes: a 302 to the CDN and a 206 there, as the Hub answers it."""
    import catalog_scan_packages as K

    url = K.header_url(repo, sha, file)
    body = raw if raw is not None else gguf_head(architecture)
    if redirect:
        target = f"{CDN}/{file.rsplit('/', 1)[-1]}?sig=1"
        put(hub, url, b"", status=302, headers={"Location": target})
        put(hub, target, body, status=206)
    else:
        put(hub, url, body, status=206)


def card(hub, row: str, text: str | None, sha: str = SHA_A, *, status: int = 200) -> None:
    """The model answer (the revision) and the card at it; `text` None leaves the card unrecorded for a test that sets it."""
    import catalog_scan_packages as K

    put(hub, K.model_url(row), {"id": row, "sha": sha})
    if text is not None:
        put(hub, K.card_url(row, sha), text.encode("utf-8"), status=status, headers={"X-Repo-Commit": sha})


def quiet_world(hub) -> None:
    """Every request of the new stage answered with nothing to find: the source, empty lists, cards without a statement."""
    source(hub)
    for row in ROWS:
        listing(hub, row, [])
        card(hub, row, "# A card\n\nNothing to set up here.\n")
