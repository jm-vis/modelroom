"""Strict request/result shapes of the remote protocol v1 (AP9-FR, FR-B1, Codex r1 finding 4),
read by both ends: `modelroom/ssh.py`/`remote_client.py` validate an answer against these before
anything reads a field of it; `modelroom/remote_server.py` validates a request the same way before
it ever touches the machine. Every model is `extra="forbid"`: an added field on either side is a
form fault, never a silently accepted extra. Nothing here does I/O.

`RESULT_MODELS`/`PROGRESS_MODELS` map a transport budget key (`ssh.OP_BUDGETS`) to the model its
`result`/`progress` field must validate as; `daemon_generate` shares `daemon_short`'s shape (both
are the same short-call envelope, only the time budget differs).
"""

from __future__ import annotations

import re
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from .contracts import PROFILE_ID_RE
from .measurements import ExportObject
from .profile import MACHINE_CLASSES

SYSTEMS = ("linux", "win32", "darwin")
_DIGEST_STAGE_RE = re.compile(r"^pulling [0-9a-f]{64}$")
PULL_STAGES = ("pulling manifest", "verifying sha256 digest", "writing manifest", "success", "working")
# The `Literal` above is spelled out (a variable cannot feed `Literal[...]` portably); this keeps
# the two in step -- a drift here is a loud `AssertionError` at import time, not a silent mismatch.
assert MACHINE_CLASSES == ("laptop", "workstation", "server"), MACHINE_CLASSES


class _Strict(BaseModel):
    """`extra="forbid"` (Codex r1 finding 4): no unlisted field, on either side. Every field that
    could be confused for another JSON type -- a bool for a status string, a bool for an int, a
    string for a boolean flag -- is spelled `StrictBool`/`StrictInt`/`StrictStr` explicitly, so
    `"done": "false"` (Codex r1 finding 4's own example) fails validation instead of reading as
    truthy later. Structural fields JSON has no exact match for (`cards`, a list of pairs standing
    in for a tuple) stay in pydantic's ordinary, non-strict shape coercion -- there is no second,
    looser way to spell a JSON array as a Python tuple."""

    model_config = ConfigDict(extra="forbid")


# --- requests (validated by the gegenstelle, `remote_server.py`) ----------------------------------


class HelloRequest(_Strict):
    pass


class ProfileRequest(_Strict):
    profile_id: StrictStr = Field(pattern=PROFILE_ID_RE.pattern)
    machine_class: Literal["laptop", "workstation", "server"] | None = None


class RoomRequest(_Strict):
    # The wire carries `[[index, name], ...]`, a JSON array of arrays -- there is no JSON tuple, so
    # accepting a length-2 list as this field's tuple is not a coercion this module means to forbid.
    cards: list[tuple[StrictInt, StrictStr | None]] = Field(default_factory=list)


class DaemonRequest(_Strict):
    method: Literal["GET", "POST"]
    path: StrictStr
    body: dict | None = None


REQUEST_MODELS: dict[str, type[_Strict]] = {
    "hello": HelloRequest,
    "profile": ProfileRequest,
    "room": RoomRequest,
    "daemon": DaemonRequest,
}


# --- results (validated by the laptop, `ssh.py`) --------------------------------------------------


class HelloResult(_Strict):
    version: StrictStr | None
    system: Literal["linux", "win32", "darwin"]
    ollama: Literal["reachable", "unreachable"]
    agent_forwarded: StrictBool
    os_fingerprint: StrictStr


class ProfileResult(_Strict):
    export: ExportObject


class DiskResult(_Strict):
    known: StrictBool
    free_bytes: StrictInt | None = Field(default=None, ge=0)
    unclear: StrictBool


class RoomResult(_Strict):
    disk: DiskResult
    graphics_free_mib: list[float] | None = None


class DaemonShortResult(_Strict):
    status: StrictInt = Field(ge=100, le=599)
    body: object = None


class DaemonPullResult(_Strict):
    status: StrictInt = Field(ge=100, le=599)
    done: StrictBool


class PullProgress(_Strict):
    status: StrictStr
    total: StrictInt | None = Field(default=None, ge=0)
    completed: StrictInt | None = Field(default=None, ge=0)

    @staticmethod
    def allowed_status(value: str) -> bool:
        return value in PULL_STAGES or bool(_DIGEST_STAGE_RE.fullmatch(value))


class PullErrorProgress(_Strict):
    error: Literal[True]


RESULT_MODELS: dict[str, type[_Strict]] = {
    "hello": HelloResult,
    "profile": ProfileResult,
    "room": RoomResult,
    "daemon_short": DaemonShortResult,
    "daemon_generate": DaemonShortResult,
    "daemon_pull": DaemonPullResult,
}
# `daemon_pull` is the one op with a `progress` line at all (in this package's own scope, FR-B1);
# every other budget key never carries one, so a `progress` field there is already a form fault.
PROGRESS_MODELS: dict[str, tuple[type[_Strict], ...]] = {
    "daemon_pull": (PullProgress, PullErrorProgress),
}


def validate_progress(budget_key: str, data: object) -> dict:
    """`data` through the first `PROGRESS_MODELS[budget_key]` model it fits; `ValueError` otherwise
    (including when `budget_key` allows no `progress` line at all).

    `exclude_none=True` (unlike `validate_result`/`validate_request`): `PullProgress.total`/
    `.completed` are read by `guided_install._check_line`'s presence check (`"total" in line`),
    the same rule a real Ollama pull line follows -- most of its lines (`pulling manifest`,
    `verifying sha256 digest`, `writing manifest`) carry neither key at all. A bare `model_dump`
    turns "absent" into `"total": null`, which is a *present*, non-count value and would fail
    every one of those lines (caught live: an SSH pull never got past its first line)."""
    models = PROGRESS_MODELS.get(budget_key, ())
    for model in models:
        try:
            validated = model.model_validate(data)
        except Exception:  # noqa: BLE001 - pydantic's ValidationError, tried in order
            continue
        if isinstance(validated, PullProgress) and not PullProgress.allowed_status(validated.status):
            continue
        return validated.model_dump(mode="json", exclude_none=True)
    raise ValueError(f"no progress shape of {budget_key!r} matches")


def validate_result(budget_key: str, data: object) -> dict:
    """`data` through `RESULT_MODELS[budget_key]`; `ValueError` on any mismatch (missing field,
    extra field, wrong type) -- never a bare `KeyError` later, reading a field that was never there."""
    model = RESULT_MODELS.get(budget_key)
    if model is None:
        raise ValueError(f"no result shape for {budget_key!r}")
    try:
        return model.model_validate(data).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - pydantic's ValidationError
        raise ValueError(str(exc)) from exc


def validate_request(op: str, data: object) -> dict:
    """`data` through `REQUEST_MODELS[op]`; `ValueError` on any mismatch."""
    model = REQUEST_MODELS.get(op)
    if model is None:
        raise ValueError(f"no request shape for {op!r}")
    try:
        return model.model_validate(data).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - pydantic's ValidationError
        raise ValueError(str(exc)) from exc


__all__ = [
    "PULL_STAGES",
    "PROGRESS_MODELS",
    "REQUEST_MODELS",
    "RESULT_MODELS",
    "SYSTEMS",
    "DaemonPullResult",
    "DaemonRequest",
    "DaemonShortResult",
    "DiskResult",
    "HelloRequest",
    "HelloResult",
    "ProfileRequest",
    "ProfileResult",
    "PullErrorProgress",
    "PullProgress",
    "RoomRequest",
    "RoomResult",
    "validate_progress",
    "validate_request",
    "validate_result",
]
