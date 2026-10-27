"""The published catalog: one attempt a day to load it, one envelope to keep it, one rule to choose it.

`current_catalog` is the only way the package reads the catalog from the network, and only the
guided mode calls it (`CONTRACTS.md`, "Published catalog"). It never raises and never stops a run:
every fault ends in the choice between the shipped catalog and the one in the envelope, and the
line it returns says which one applied.

An attempt is at most three requests -- the dataset's revision, then the file at that commit with
one redirect -- booked against the run's shared budget, each answer capped (`max_bytes`) and each
redirect target kept on `huggingface.co`. Nothing a downloaded file names is ever fetched.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.parse
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal, NamedTuple

from .catalog import CATALOG_MAX_BYTES, CATALOG_V3_MAX_BYTES, Catalog, CatalogError, load_catalog
from .catalog_schema import build_catalog_v2, build_catalog_v3, build_catalog_v4
from .contracts import SchemaVersionError
from .http import (
    BudgetedTransport,
    BudgetExhaustedError,
    RedirectingTransport,
    RedirectLimitError,
    RequestBudget,
    Response,
    ResponseTooLargeError,
    Transport,
    TransportSecurityError,
)
from .search_release import _note as _clip  # a reason, cut to 97 characters like every note of step 2
from .state import atomic_write_json

DATASET = "jm-vis/modelroom-catalog"


@dataclass(frozen=True)
class Channel:
    """One published form of the catalog: its file in the dataset, its cache file, its schema, its size, its reader.

    One channel per schema, so a release reads exactly the file whose schema it knows: release 1.0.0 reads the
    channel of schema 2 and never meets the file of schema 3 (CONTRACTS.md, "Published catalog").
    """

    file_name: str
    cache_name: str
    schema: int
    max_bytes: int
    build: Callable[[str, Path], Catalog]

    @property
    def envelope_max_bytes(self) -> int:
        return 2 * self.max_bytes + 4096  # the catalog plus its envelope


CHANNEL_V2 = Channel("catalog-v2.json", "catalog-v2.cache.json", 2, CATALOG_MAX_BYTES, build_catalog_v2)
CHANNEL_V3 = Channel("catalog-v3.json", "catalog-v3.cache.json", 3, CATALOG_V3_MAX_BYTES, build_catalog_v3)
CHANNEL_V4 = Channel("catalog-v4.json", "catalog-v4.cache.json", 4, CATALOG_V3_MAX_BYTES, build_catalog_v4)
# The channel a guided run asks: the newest one, `catalog-v4.json` since release 1.2.0 (it was `catalog-v3.json` in 1.1.0,
# AP9-KO P5). The dataset's file is published before the release is, so a release that meets no such file falls back to
# its shipped catalog like on any other fault; releases 1.0.0 and 1.1.0 ask their own files and do not know this line.
DEFAULT_CHANNEL = CHANNEL_V4
REMOTE_FILE_NAME = CHANNEL_V2.file_name  # the old channel's file, which release 1.0.0 reads and the publication tool probes
CACHE_FILE_NAME = CHANNEL_V2.cache_name
CACHE_VERSION = 1
SWITCH = "MODELROOM_CATALOG"  # `shipped` turns the network off
FRESH_FOR = timedelta(hours=24)
REVISION_MAX_BYTES = 64 * 1024
FILE_MAX_BYTES = CATALOG_MAX_BYTES  # 1 MiB, the same bound the document reader holds for schema 2
_ENVELOPE_KEYS = frozenset({"cache_version", "checked_at", "sha", "last_error", "catalog"})
_HOST = "huggingface.co"
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_STAMP = "%Y-%m-%dT%H:%M:%SZ"

Source = Literal["shipped", "cached", "published"]


class CatalogChoice(NamedTuple):
    """The catalog that applies, the line step 2 shows about it, and where it came from."""

    catalog: Catalog
    origin: str
    source: Source


class _Fault(Exception):
    """An answer this module refuses; its text is the reason on the screen."""


# Everything an attempt can end in. `KeyError` and its kin are not on the list: a bug is not a fault.
_FAULTS = (
    OSError, http.client.HTTPException, ValueError, RuntimeError, BudgetExhaustedError,
    ResponseTooLargeError, TransportSecurityError, CatalogError, SchemaVersionError, _Fault,
)


@dataclass(frozen=True)
class _Envelope:
    checked_at: datetime
    sha: str | None
    last_error: str | None
    document: dict | None
    catalog: Catalog | None


class _SameHost:
    """Refuses a hop that leaves `https://huggingface.co`; it sits below the budget, so a refused hop is booked, never sent."""

    def __init__(self, inner: Transport) -> None:
        self._inner = inner

    def __call__(self, method: str, url: str, headers: dict[str, str] | None = None, *, max_bytes: int | None = None) -> Response:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" or parts.hostname != _HOST or parts.port not in (None, 443):
            raise _Fault("the answer leads to another host")
        return self._inner(method, url, headers, max_bytes=max_bytes)


def current_catalog(
    transport: Transport,
    now: datetime,
    cache_dir: Path,
    budget: RequestBudget,
    channel: Channel = DEFAULT_CHANNEL,
    shipped: Catalog | None = None,
) -> CatalogChoice:
    """The catalog this run uses, and the line that says which (rules 1-9 of the KA brief).
    `shipped` is the candidate "shipped"; `None` reads `modelroom/catalog.json` as ever (tests pass their own)."""
    shipped = load_catalog() if shipped is None else shipped
    if os.environ.get(SWITCH) == "shipped":
        return _choose(shipped, None, None, None)
    stamp = _utc(now)
    path = cache_dir / channel.cache_name
    envelope = _read_envelope(path, channel)
    if envelope is not None and timedelta(0) <= stamp - envelope.checked_at < FRESH_FOR:
        return _choose(shipped, envelope, None, None)  # fresh, even if its last attempt failed
    spent = budget.used
    try:
        sha, document, catalog = _load(transport, budget, envelope, channel)
    except _FAULTS as exc:
        reason = _clean(exc)
        if isinstance(exc, BudgetExhaustedError) and budget.used == spent:
            return _choose(shipped, envelope, None, reason)  # no request left the machine: not the day's attempt
        _store(path, stamp, envelope.sha if envelope else None, envelope.document if envelope else None, reason)
        return _choose(shipped, envelope, None, reason)
    if document is None:  # the same commit: the catalog of the envelope stands, only its stamp moves
        _store(path, stamp, sha, envelope.document, None)
        return _choose(shipped, replace(envelope, checked_at=stamp, last_error=None), None, None)
    _store(path, stamp, sha, document, None)
    # the envelope stays a candidate (rule 5); its old fault is hidden, because this attempt worked
    return _choose(shipped, replace(envelope, last_error=None) if envelope else None, catalog, None)


def offline_catalog(cache_dir: Path, channel: Channel = DEFAULT_CHANNEL, shipped: Catalog | None = None) -> Catalog:
    """The catalog a command that asks no one reads (`render`, `pull`): the shipped one, or the stored published
    one of `channel` when it is the later of the two. Nothing is requested and nothing is written; a missing
    or broken cache, or `MODELROOM_CATALOG=shipped`, leaves the shipped catalog. `shipped=None` reads
    `modelroom/catalog.json` as ever."""
    shipped = load_catalog() if shipped is None else shipped
    if os.environ.get(SWITCH) == "shipped":
        return shipped
    return _choose(shipped, _read_envelope(cache_dir / channel.cache_name, channel), None, None).catalog


def _choose(shipped: Catalog, cached: _Envelope | None, fresh: Catalog | None, fault: str | None) -> CatalogChoice:
    """The candidate with the latest `published`; a tie goes to fresh, then envelope, then shipped (rule 5)."""
    candidates: list[tuple[Source, Catalog | None]] = [
        ("published", fresh), ("cached", cached.catalog if cached else None), ("shipped", shipped),
    ]
    source, catalog = max(
        ((name, found) for name, found in candidates if found is not None),
        key=lambda item: item[1].published or date.min,  # no or unreadable day: the oldest
    )
    day = catalog.published.isoformat() if catalog.published else "undated"
    line = f"catalog: {'shipped' if source == 'shipped' else 'published'} {day}"
    if fault is not None:
        line += f" (the published one did not read: {fault})"
    elif cached is not None and cached.last_error:
        line += f" (the published one did not read on {cached.checked_at.date()}: {cached.last_error})"
    return CatalogChoice(catalog, line, source)


def _load(
    transport: Transport, budget: RequestBudget, known: _Envelope | None, channel: Channel
) -> tuple[str, dict | None, Catalog | None]:
    """The attempt: the revision, and the file only when the commit is not the envelope's. At most three requests."""
    revision = _get(transport, budget, 1, f"https://{_HOST}/api/datasets/{DATASET}/revision/main", REVISION_MAX_BYTES).json()
    sha = revision.get("sha") if isinstance(revision, dict) else None
    if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
        raise _Fault("the revision names no commit")
    if known is not None and known.sha == sha:
        return sha, None, None
    url = f"https://{_HOST}/datasets/{DATASET}/resolve/{sha}/{channel.file_name}"
    answer = _get(transport, budget, 2, url, channel.max_bytes)
    if answer.header("X-Repo-Commit") != sha:
        raise _Fault("the file is not from the commit that was asked for")
    text = answer.body.decode("utf-8")
    return sha, json.loads(text), channel.build(text, Path(channel.file_name))


def _get(transport: Transport, budget: RequestBudget, hops: int, url: str, limit: int) -> Response:
    """One answer, `hops` requests at most (the first counts): booked, kept on the host, capped, and a `200`."""
    layers = RedirectingTransport(BudgetedTransport(_SameHost(transport), budget), max_hops=hops)
    try:
        response = layers("GET", url, None, max_bytes=limit)
    except RedirectLimitError as exc:  # any other `RuntimeError` keeps its own text in `_FAULTS`
        raise _Fault("the answer redirects more often than allowed") from exc
    if response.status != 200:
        raise _Fault(f"HTTP {response.status}")
    announced = response.header("Content-Length")
    if announced is not None and announced.isdigit() and int(announced) != len(response.body):
        raise _Fault("the answer ended early")
    return response


def _read_envelope(path: Path, channel: Channel = DEFAULT_CHANNEL) -> _Envelope | None:
    """The envelope, or `None` when it is missing, does not read or does not validate (rule 2)."""
    limit = channel.envelope_max_bytes
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:  # a bound over the whole file, not over a prefix of it
            return None
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != _ENVELOPE_KEYS or type(data["cache_version"]) is not int:
            return None
        sha, error, document = data["sha"], data["last_error"], data["catalog"]
        checked_at = datetime.fromisoformat(data["checked_at"])
        stamp = checked_at.astimezone(timezone.utc) if checked_at.tzinfo is not None else None  # may overflow
        sound = (
            data["cache_version"] == CACHE_VERSION and stamp is not None
            and (sha is None or (isinstance(sha, str) and _SHA_RE.fullmatch(sha)))
            and (error is None or isinstance(error, str)) and (document is None) == (sha is None)
            and (document is None or isinstance(document, dict))
        )
        catalog = None if not sound or document is None else channel.build(json.dumps(document), path)
    except (OSError, ValueError, TypeError, OverflowError, RuntimeError, CatalogError, SchemaVersionError):
        return None
    if not sound:
        return None
    return _Envelope(stamp, sha, _clean(error) if error else None, document, catalog)


def _store(path: Path, stamp: datetime, sha: str | None, document: dict | None, last_error: str | None) -> None:
    """Write the whole envelope in one atomic replace; a cache folder that cannot be written costs the next run's attempt, not the next day's."""
    envelope = {
        "cache_version": CACHE_VERSION, "checked_at": stamp.strftime(_STAMP), "sha": sha,
        "last_error": last_error, "catalog": document,
    }
    try:
        atomic_write_json(path, envelope)
    except OSError:
        pass  # the choice of this run does not depend on it; every run asks once (the budget caps it) until the folder can be written


def _clean(text: object) -> str:
    """One printable ASCII line cut to 97 characters: a reason can carry what a stranger's file said."""
    return _clip(re.sub(r"[^\x20-\x7e]", "?", " ".join(str(text).split())))


def _utc(now: datetime) -> datetime:
    zoned = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    return zoned.replace(microsecond=0)


__all__ = [
    "CACHE_FILE_NAME",
    "CHANNEL_V2",
    "CHANNEL_V3",
    "CHANNEL_V4",
    "DATASET",
    "DEFAULT_CHANNEL",
    "REMOTE_FILE_NAME",
    "CatalogChoice",
    "Channel",
    "current_catalog",
    "offline_catalog",
]
