"""Hugging Face fetching: a base model's architecture, and a packager's packages.

Two independent operations, matching CONTRACTS.md's area semantics:

- `fetch_base_model_meta` reads one base model's publisher repo (model info + `config.json`)
  and always returns a result -- a repo that cannot be reached resolves to an unknown
  architecture and no parameter count, it never raises and never affects package fetching.
- `fetch_hf_area` covers one (base model, packager-owner) area: it probes every candidate
  repo name for that owner, assembles a `Package` per quantization group found in each
  candidate's file tree, and only ever ends `incomplete` on a genuine failure (a status other
  than 200/404/401, a network error, an unparseable response, or the run's request budget
  running out) -- a candidate repo that simply does not exist is not an error.

Everything here is bound to the revision it was observed at, per CONTRACTS.md's identity
rules: a repo's `sha` from its model-info response is threaded through to the tree fetch and
into every `Package.revision` this area produces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .config import Configuration
from .contracts import (
    Architecture,
    BaseModelSpec,
    Package,
    PackageFile,
    architecture_from_hf_config,
    shards_complete,
)
from .fetch_types import AreaOutcome
from .http import Transport
from .provenance import decide_provenance
from .quantization import has_shard_suffix, identity_stem, non_weight_role, parse_hf_quant

HF_API = "https://huggingface.co/api"

# Hugging Face does not let an anonymous caller distinguish "private" from "does not exist":
# a request for a repo that is not there comes back 401 ("Invalid username or password."),
# never 404 (measured 2026-09-22, see tests/fixtures/README.md). Both are treated as "no
# package/repo here", never as a fetch failure.
_NOT_FOUND_STATUSES = (404, 401)

_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


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
            sha = info.get("sha") if isinstance(info.get("sha"), str) else None
            safetensors = info.get("safetensors")
            if isinstance(safetensors, dict) and isinstance(safetensors.get("total"), (int, float)):
                parameters_b = safetensors["total"] / 1e9

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

    architecture = architecture_from_hf_config(hf_repo, sha, config_data)
    return BaseModelMeta(publisher=publisher, parameters_b=parameters_b, architecture=architecture)


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


def fetch_hf_area(transport: Transport, base_model: BaseModelSpec, owner: str, run_at: datetime) -> AreaOutcome:
    """Fetch every candidate repo under `owner` for `base_model` and assemble their packages."""
    packages: list[Package] = []
    for name in _candidate_repo_names(base_model):
        repo = f"{owner}/{name}"
        try:
            info_response = transport("GET", f"{HF_API}/models/{repo}")
        except Exception as exc:
            return _incomplete(base_model, owner, str(exc))
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
        tags = info.get("tags") if isinstance(info.get("tags"), list) else []

        try:
            entries = _fetch_tree(transport, repo, sha)
        except Exception as exc:
            return _incomplete(base_model, owner, str(exc))
        packages.extend(_assemble_packages(repo, sha, base_model, tags, entries, run_at))

    return AreaOutcome(
        source="huggingface",
        base_model_hf_repo=base_model.hf_repo,
        packager=owner,
        status="complete",
        error=None,
        packages=packages,
    )


def _fetch_tree(transport: Transport, repo: str, sha: str) -> list[dict]:
    url = f"{HF_API}/models/{repo}/tree/{sha}?recursive=true"
    entries: list[dict] = []
    while url:
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
    return entries


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
    files: list[PackageFile] = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        name = entry.get("path")
        if not name:
            continue
        digest = None
        lfs = entry.get("lfs")
        if isinstance(lfs, dict) and lfs.get("oid"):
            digest = f"sha256:{lfs['oid']}"
        files.append(
            PackageFile(name=name, role=_classify_role(name), size_bytes=entry.get("size") or 0, digest=digest)
        )
    return files


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
    repo: str, sha: str, base_model: BaseModelSpec, tags: list[str], entries: list[dict], run_at: datetime
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
        provenance, unresolved_reason = decide_provenance(stub, base_model, tags, [])
        packages.append(_finalize(stub, provenance, unresolved_reason))
    return packages
