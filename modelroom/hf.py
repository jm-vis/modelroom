"""Hugging Face fetching: a base model's architecture, and a packager's packages.

Two independent operations, matching CONTRACTS.md's area semantics:

- `fetch_base_model_meta` reads one base model's publisher repo (model info + `config.json`)
  and always returns a result -- a repo that cannot be reached resolves to an unknown
  architecture and no parameter count, it never raises and never affects package fetching.
- `fetch_hf_area` covers one (base model, packager-owner) area: it probes every candidate
  repo name for that owner, assembles a `Package` per quantization group found in each
  candidate's file tree, and only ever ends `incomplete` on a genuine failure (a status other
  than 200/404/401, a network error, an unparseable response, a tree entry or weight-file size
  in an unexpected shape, or the run's request budget running out) -- a candidate repo that
  simply does not exist is not an error. Package assembly runs inside the same error handling
  as the tree fetch itself, so a shape problem in one candidate ends only this area, never the
  whole run (F3/F4).

Everything here is bound to the revision it was observed at, per CONTRACTS.md's identity
rules: a repo's `sha` from its model-info response is threaded through to the tree fetch and
into every `Package.revision` this area produces.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .config import Configuration
from .contracts import (
    Approval,
    Architecture,
    BaseModelSpec,
    Package,
    PackageFile,
    architecture_from_hf_config,
    shards_complete,
)
from .fetch_types import AreaOutcome, error_text
from .http import BudgetExhaustedError, Transport
from .provenance import decide_provenance
from .quantization import has_shard_suffix, identity_stem, non_weight_role, package_identity_key, parse_hf_quant

HF_API = "https://huggingface.co/api"

# Hugging Face does not let an anonymous caller distinguish "private" from "does not exist":
# a request for a repo that is not there comes back 401 ("Invalid username or password."),
# never 404 (measured 2026-09-22, see tests/fixtures/README.md). Both are treated as "no
# package/repo here", never as a fetch failure.
_NOT_FOUND_STATUSES = (404, 401)

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
# R2: a model-info response's 'sha' is only ever trusted as a real commit sha when it matches
# this shape -- the same 40-hex rule Architecture.source_revision itself enforces, checked here
# first so a malformed sha never reaches that validator and raises out of this module.
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")

# P3-3 (fix-round 5): a hard cap on `_fetch_tree`'s `Link: rel="next"` pagination -- a real
# repo's tree fits in a handful of pages; a chain that has not terminated within this many hops
# is either a broken/hostile response or a page loop, never a legitimate tree, so it raises
# instead of fetching forever. The hop that would exceed the cap is never made, the same
# convention `http.py::MAX_REDIRECTS` uses for a redirect chain.
_MAX_TREE_PAGES = 20


@dataclass(frozen=True)
class BaseModelMeta:
    """What `fetch_base_model_meta` learned about one base model, independent of any package."""

    publisher: str
    parameters_b: float | None
    architecture: Architecture


def candidate_owners(config: Configuration, base_model: BaseModelSpec) -> list[str]:
    """Every owner to probe for `base_model`'s packages: configured packagers, plus its own.

    A base model's own owner counts as a packager of its own models (CONTRACTS.md/AGENTS.md).
    The result is filtered to `config.allowed_owners()` and de-duplicated, order preserved --
    an owner outside that positive list is never returned, and therefore never fetched.
    """
    own_owner = base_model.hf_repo.split("/", 1)[0]
    allowed = config.allowed_owners()
    seen: set[str] = set()
    owners: list[str] = []
    for owner in (*config.packagers, own_owner):
        if owner in allowed and owner not in seen:
            seen.add(owner)
            owners.append(owner)
    return owners


def fetch_base_model_meta(transport: Transport, hf_repo: str) -> BaseModelMeta:
    """Read the publisher repo `hf_repo`. Never raises; a failure resolves to "unknown".

    Takes the repo id directly, not a `BaseModelSpec`: this is what *produces* the
    `publisher`/`parameters_b`/`architecture` fields a `BaseModelSpec` needs, so the caller
    cannot have a full spec yet when it calls this.

    R2: a `sha` is only used when it is a real 40-hex commit sha (never passed on to
    `architecture_from_hf_config`/`Architecture.source_revision` otherwise, and never used to
    build a `config.json` URL); `safetensors.total` is only used when it is a real number
    greater than zero (a `0` would otherwise become `parameters_b=0.0`, which
    `BaseModelSpec.parameters_b`'s own `gt=0` constraint rejects downstream in `fetch.py`). The
    architecture build itself is wrapped in `try`/`except`: any exception it raises resolves to
    an unknown `Architecture` instead of escaping, so a registry response this function has not
    anticipated can never turn into an uncaught exception in `run_fetch`.
    """
    publisher = hf_repo.split("/", 1)[0]
    sha: str | None = None
    parameters_b: float | None = None

    try:
        response = transport("GET", f"{HF_API}/models/{hf_repo}")
    except Exception:
        response = None
    if response is not None and response.status == 200:
        try:
            info = response.json()
        except Exception:
            info = None
        if isinstance(info, dict):
            sha = _valid_sha(info.get("sha"))
            parameters_b = _valid_parameters_b(info.get("safetensors"))

    config_data: dict | None = None
    if sha:
        try:
            config_response = transport("GET", f"https://huggingface.co/{hf_repo}/resolve/{sha}/config.json")
        except Exception:
            config_response = None
        if config_response is not None and config_response.status == 200:
            try:
                parsed = config_response.json()
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                config_data = parsed

    try:
        architecture = architecture_from_hf_config(hf_repo, sha, config_data)
    except Exception:
        architecture = Architecture(source_repo=hf_repo, source_revision=None, kind="unknown")
    return BaseModelMeta(publisher=publisher, parameters_b=parameters_b, architecture=architecture)


def _valid_sha(value: object) -> str | None:
    """`value` as a commit sha, only when it is a real 40-hex string."""
    if isinstance(value, str) and _SHA1_RE.fullmatch(value):
        return value
    return None


def _valid_parameters_b(safetensors: object) -> float | None:
    """`safetensors.total` in billions, only when the conversion produces a real, finite,
    positive `float`.

    Fix-round 3: `total > 0` alone is not enough -- a subnormal value like `1e-320` passes it but
    underflows to `0.0` once divided by `1e9` (which `BaseModelSpec.parameters_b`'s own `gt=0`
    constraint would then reject downstream), and a JSON integer far outside `float` range (a raw
    `10**400`, which `json.loads` parses without complaint) overflows the division itself
    (`OverflowError`). Both resolve to `None` here, exactly like any other "not a real reading".
    """
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if not isinstance(total, (int, float)) or isinstance(total, bool) or total <= 0:
        return None
    try:
        result = total / 1e9
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _candidate_repo_names(base_model: BaseModelSpec) -> list[str]:
    base_name = base_model.hf_repo.split("/", 1)[1]
    names = [f"{base_name}-GGUF", *base_model.repo_aliases]
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _incomplete(base_model: BaseModelSpec, owner: str, error: str) -> AreaOutcome:
    return AreaOutcome(
        source="huggingface",
        base_model_hf_repo=base_model.hf_repo,
        packager=owner,
        status="incomplete",
        error=error,
        packages=[],
    )


def fetch_hf_area(
    transport: Transport,
    base_model: BaseModelSpec,
    owner: str,
    run_at: datetime,
    previous_by_key: dict[tuple[str, str, str], Package] | None = None,
) -> AreaOutcome:
    """Fetch every candidate repo under `owner` for `base_model` and assemble their packages.

    `previous_by_key` (F6, default `None`) maps a package identity key to that package as it
    stood in the previous snapshot; when a freshly assembled package shares its identity with
    one that carried an `Approval`, that approval is carried forward into `decide_provenance` so
    an approval bound to still-current content survives this fetch.
    """
    packages: list[Package] = []
    for name in _candidate_repo_names(base_model):
        repo = f"{owner}/{name}"
        try:
            info_response = transport("GET", f"{HF_API}/models/{repo}")
        except Exception as exc:
            return _incomplete(base_model, owner, error_text(exc))
        if info_response.status in _NOT_FOUND_STATUSES:
            continue
        if info_response.status != 200:
            return _incomplete(base_model, owner, f"{repo}: unexpected status {info_response.status}")
        try:
            info = info_response.json()
        except Exception as exc:
            return _incomplete(base_model, owner, f"{repo}: invalid JSON: {exc}")
        if not isinstance(info, dict) or not isinstance(info.get("sha"), str):
            return _incomplete(base_model, owner, f"{repo}: model info missing 'sha'")
        sha = info["sha"]
        # P3-4 (fix-round 5): `sha` is registry-controlled data, about to become a URL path
        # segment (`_fetch_tree`) and, on assembly, a `Package.revision` -- validated against
        # the same 40-hex shape `Package.revision`/`Architecture.source_revision` already
        # require, *before* it is used to build that URL, rather than only failing once
        # `_fetch_tree` has already made the request and `Package`'s own validator rejects it
        # downstream.
        if not _SHA1_RE.fullmatch(sha):
            return _incomplete(base_model, owner, f"{repo}: model info 'sha' is not a 40-hex commit sha: {sha!r}")
        tags = info.get("tags") if isinstance(info.get("tags"), list) else []

        try:
            entries = _fetch_tree(transport, repo, sha)
            packages.extend(
                _assemble_packages(repo, sha, base_model, tags, entries, run_at, previous_by_key or {})
            )
        except BudgetExhaustedError as exc:
            # P3-12 (fix-round 5): CONTRACTS.md, "Request budget", promises the area's error is
            # exactly BudgetExhaustedError's own message -- never repo-prefixed like a genuine
            # shape problem below, since the budget is a whole-run resource, not a fact about
            # this particular repo. (Always non-empty in practice -- BudgetExhaustedError's
            # message is a fixed literal -- but error_text keeps every str(exc) capture uniform.)
            return _incomplete(base_model, owner, error_text(exc))
        except Exception as exc:
            # F3: a malformed tree page (wrong shape, a weight file with no size, ...) ends
            # this area incomplete like any other genuine failure, never raises out of here.
            return _incomplete(base_model, owner, f"{repo}: {exc}")

    return AreaOutcome(
        source="huggingface",
        base_model_hf_repo=base_model.hf_repo,
        packager=owner,
        status="complete",
        error=None,
        packages=packages,
    )


def _fetch_tree(transport: Transport, repo: str, sha: str) -> list[dict]:
    """The repo's full tree, following `Link: rel="next"` pagination.

    P3-3: capped at `_MAX_TREE_PAGES` hops -- a chain that has not terminated by then raises
    instead of continuing forever, and the page that would exceed the cap is never requested.
    """
    url = f"{HF_API}/models/{repo}/tree/{sha}?recursive=true"
    entries: list[dict] = []
    for _ in range(_MAX_TREE_PAGES):
        response = transport("GET", url)
        if response.status != 200:
            raise RuntimeError(f"{repo}: unexpected status {response.status} fetching tree")
        try:
            page = response.json()
        except Exception as exc:
            raise RuntimeError(f"{repo}: invalid JSON in tree response: {exc}") from exc
        if not isinstance(page, list):
            raise RuntimeError(f"{repo}: tree response is not a list")
        entries.extend(page)
        link = response.header("Link")
        match = _LINK_NEXT_RE.search(link) if link else None
        url = match.group(1) if match else None
        if url is None:
            return entries
    raise RuntimeError(f"{repo}: tree pagination did not finish within {_MAX_TREE_PAGES} pages")


def _classify_role(name: str) -> Literal["weights", "weights_shard", "mmproj", "other"]:
    """Classify one tree entry's path into a `PackageFile.role`.

    `name` is the full tree path (folder segments included, e.g. unsloth's per-quant
    subfolders); only the basename is inspected for the non-weight markers and the extension,
    but the shard check runs against the full path so a shard suffix inside a folder name never
    gets mistaken for one (`_classify_role` and `quantization.parse_hf_quant` share the same
    non-weight-marker and shard-suffix definitions, see `quantization.py`).
    """
    basename = re.split(r"[\\/]", name)[-1].lower()
    role = non_weight_role(basename)
    if role is not None:
        return role  # type: ignore[return-value]
    for extension in (".gguf", ".safetensors"):
        if basename.endswith(extension):
            stem = name[: -len(extension)]
            return "weights_shard" if has_shard_suffix(stem) else "weights"
    return "other"


def _files_from_tree(entries: list[dict]) -> list[PackageFile]:
    """Build one `PackageFile` per file entry, or raise on a shape the registry never sends.

    F3/F4: every entry must be an object; `path` must be a string; a `weights`/`weights_shard`
    file must carry a non-negative integer `size` (a missing or malformed size on a weight file
    is a shape error, since a silently-assumed `0` would later make `fit` believe an empty
    package fits everywhere) -- a non-weight file (`mmproj`/`other`) may still default to `0`.

    F7 (fix-round 5): a `type: "file"` entry's `path` missing, empty or not a string is *also* a
    shape error, never silently skipped -- `if not name: continue` used to drop it before the
    `isinstance` check below ever ran (a real directory entry never reaches here at all, since
    it is filtered out by `type != "file"` above; a genuine file entry always carries a real
    path). Silently dropping every entry this way made a tree that is entirely malformed look
    like a genuinely empty, `complete` area, which `state.merge_snapshot` then reads as "this
    area really has zero files now" and deactivates every one of its old packages.

    R7-9 (fix-round 6): `entry.get("type") != "file"` was `True` for an entry with no `type`
    field at all (`{}`), so it was silently skipped exactly like a real directory entry --
    `_files_from_tree([{}])` returned `[]`, the same "genuinely empty, complete area" hazard F7
    closed for a bad `path`. `type` must now be a non-empty string before it is even compared to
    `"file"`; only an entry that genuinely *names* a non-file type (a real directory) is skipped.
    """
    files: list[PackageFile] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"tree entry is not an object: {entry!r}")
        entry_type = entry.get("type")
        if not isinstance(entry_type, str) or not entry_type:
            raise ValueError(f"tree entry 'type' is missing, empty or not a string: {entry_type!r}")
        if entry_type != "file":
            continue
        name = entry.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tree entry 'path' is missing, empty or not a string: {name!r}")
        role = _classify_role(name)
        size_bytes = _tree_entry_size(name, role, entry.get("size"))
        digest = _tree_entry_digest(entry.get("lfs"))
        files.append(PackageFile(name=name, role=role, size_bytes=size_bytes, digest=digest))
    return files


# F8 (fix-round 5): an upper bound on a tree entry's 'size' -- `json.loads` parses a JSON
# integer literal of any width without complaint, and a value near or beyond this bound would
# later raise `OverflowError` out of `fit.py::compute_fit`'s `sum(...) / GIB` (Python's own
# `int.__truediv__` cannot convert an arbitrarily large int to `float`) instead of this module
# ending the area `incomplete` like any other shape problem. `2**63` is already far past any
# real file size (an exabyte); chosen as a round, generous bound rather than tuned to a
# specific failure threshold.
_MAX_FILE_SIZE_BYTES = 2**63


def _tree_entry_size(name: str, role: str, raw_size: object) -> int:
    if raw_size is None:
        if role in ("weights", "weights_shard"):
            raise ValueError(f"weight file {name!r} has no 'size'")
        return 0
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
        raise ValueError(f"tree entry 'size' for {name!r} is not a non-negative integer: {raw_size!r}")
    if raw_size >= _MAX_FILE_SIZE_BYTES:
        raise ValueError(f"tree entry 'size' for {name!r} is implausibly large: {raw_size!r}")
    return raw_size


def _tree_entry_digest(lfs: object) -> str | None:
    if not isinstance(lfs, dict):
        return None
    oid = lfs.get("oid")
    if oid is None:
        return None
    if not isinstance(oid, str):
        raise ValueError(f"tree entry lfs.oid is not a string: {oid!r}")
    return f"sha256:{oid}"


def _package_format(filename: str) -> Literal["gguf", "tensor", "unknown"]:
    lower = filename.lower()
    if lower.endswith(".gguf"):
        return "gguf"
    if lower.endswith(".safetensors"):
        return "tensor"
    return "unknown"


def _finalize(stub: Package, provenance: str, unresolved_reason: str | None) -> Package:
    """Rebuild `stub` with the real provenance verdict, re-validating every invariant."""
    data = stub.model_dump(mode="python")
    data["provenance"] = provenance
    data["unresolved_reason"] = unresolved_reason
    return Package(**data)


def _assemble_packages(
    repo: str,
    sha: str,
    base_model: BaseModelSpec,
    tags: list[str],
    entries: list[dict],
    run_at: datetime,
    previous_by_key: dict[tuple[str, str, str], Package],
) -> list[Package]:
    files = _files_from_tree(entries)
    weight_files = [f for f in files if f.role in ("weights", "weights_shard")]
    extra_files = [f for f in files if f.role in ("mmproj", "other")]

    groups: dict[str, list[PackageFile]] = {}
    for file in weight_files:
        groups.setdefault(identity_stem(file.name), []).append(file)

    packages: list[Package] = []
    for group_files in groups.values():
        representative = group_files[0].name
        stub = Package(
            source="huggingface",
            repo=repo,
            revision=sha,
            base_model_hf_repo=base_model.hf_repo,
            format=_package_format(representative),
            files=[*group_files, *extra_files],
            complete=shards_complete(group_files),
            quantization=parse_hf_quant(representative),
            default_context=None,
            provenance="unresolved",
            unresolved_reason="format",
            approval=None,
            observed_at=run_at,
            last_seen=run_at,
            active=True,
        )
        stub, approvals = _carry_forward_approval(stub, previous_by_key)
        provenance, unresolved_reason = decide_provenance(stub, base_model, tags, approvals)
        packages.append(_finalize(stub, provenance, unresolved_reason))
    return packages


def _carry_forward_approval(
    stub: Package, previous_by_key: dict[tuple[str, str, str], Package]
) -> tuple[Package, list[Approval]]:
    """F6/R3: attach a previous package's `Approval` to `stub` when their identity keys match
    *and* they belong to the same base model.

    `decide_provenance` gets `[approval]` instead of `[]`; its own content check (the approval
    still has to name the *current* revision) decides whether it actually resolves `approved`.
    The approval stays on the returned `Package` regardless -- CONTRACTS.md: "the approval
    object stays on the package for history". R3: `package_identity_key` is only
    `(source, repo|ollama_name, filename-or-tag)` -- it says nothing about which base model the
    package belongs to, so a bare identity match is not enough: the same packager repo
    reassembled under a *different* base model (a family reconfigured to a new `hf_repo`, or two
    base models that happen to share a packager repo) must never inherit an approval that was
    never given for it. When the base model differs, the approval is not carried at all -- not
    even attached for history, since it never belonged to this package's lineage in the first
    place.
    """
    previous = previous_by_key.get(package_identity_key(stub))
    if previous is None or previous.approval is None:
        return stub, []
    if previous.base_model_hf_repo != stub.base_model_hf_repo:
        return stub, []
    return stub.model_copy(update={"approval": previous.approval}), [previous.approval]
