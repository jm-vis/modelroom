"""The response helpers every fixture module shares: a fixture file as the body of a `Response`.

Moved out of `tests/fixture_support.py` unchanged (AP9-UX2, size guard), so that
`tests/fixture_servers.py` and `tests/fixture_support.py` can both use them without importing
each other. Not a test module (no `test_` prefix).
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

