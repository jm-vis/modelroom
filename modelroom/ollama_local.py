"""The local Ollama daemon's own inventory: `GET /api/tags` -> `InstalledModel` list.

Unlike `modelroom/ollama.py` (the public Ollama *registry*, fetched for `fetch`), this talks to
the Ollama daemon running on the machine `hardware` is being run on. The daemon being
unreachable, or answering with something unexpected, is never a failure of the `hardware`
command -- it is recorded as `installed=None` with a reason, exactly like CONTRACTS.md
documents for `HardwareSnapshot.installed_unavailable_reason`.
"""

from __future__ import annotations

from datetime import datetime

from .contracts import InstalledModel
from .http import Transport

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


def fetch_installed_models(
    transport: Transport, now: datetime, base_url: str = DEFAULT_BASE_URL
) -> tuple[list[InstalledModel] | None, str | None]:
    """The daemon's installed models, or `(None, reason)` if they could not be read.

    The daemon's `/api/tags` reports each digest as bare hex, with no `sha256:` prefix
    (measured 2026-09-22); this normalizes it before building `InstalledModel`.
    """
    url = f"{base_url}/api/tags"
    try:
        response = transport("GET", url)
    except Exception as exc:
        return None, f"cannot reach the Ollama daemon at {base_url}: {exc}"

    if response.status != 200:
        return None, f"unexpected status {response.status} from the Ollama daemon at {base_url}"

    try:
        data = response.json()
    except Exception as exc:
        return None, f"cannot parse the Ollama daemon's /api/tags response: {exc}"

    if not isinstance(data, dict) or "models" not in data:
        return None, "the Ollama daemon's /api/tags response has no 'models' field"

    entries = data["models"]
    if not isinstance(entries, list):
        return None, "the Ollama daemon's /api/tags 'models' field is not a list"

    try:
        installed = [_installed_model(entry, now) for entry in entries]
    except Exception as exc:
        return None, f"cannot parse an Ollama daemon model entry: {exc}"

    return installed, None


def _installed_model(entry: dict, now: datetime) -> InstalledModel:
    return InstalledModel(
        name=entry["name"],
        digest=f"sha256:{entry['digest']}",
        size_bytes=entry.get("size", 0),
        observed_at=now,
    )
