"""Hugging Face fetching: a base model's architecture, and a packager's packages.

Two independent operations, matching CONTRACTS.md's area semantics:

- `fetch_base_model_meta` reads one base model's publisher repo (model info + `config.json`)
  and always returns a result -- a repo that cannot be reached resolves to an unknown
  architecture and no parameter count, it never raises and never affects package fetching.
- `fetch_hf_area` covers one (base model, packager-owner) area: it works through every package
  target of that owner (`config.package_targets`, grouped by owner with `target_owners`),
  assembles a `Package` per quantization group found in each target's file tree, and only ever
  ends `incomplete` on a genuine failure (a status other than 200/404/401, a network error, an
  unparseable response, a tree entry or weight-file size in an unexpected shape, or the run's
  request budget running out) -- a target repo that simply does not exist is not an error.
  Package assembly runs inside the same error handling as the tree fetch itself, so a shape
  problem in one target ends only this area, never the whole run (F3/F4).

Everything here is bound to the revision it was observed at, per CONTRACTS.md's identity
rules: a repo's `sha` from its model-info response is threaded through to the tree fetch and
into every `Package.revision` this area produces.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

from .architecture import architecture_from_hf_config
from .contracts import (
    Approval,
    Architecture,
    BaseModelSpec,
    Package,
    PackageFile,
    shards_complete,
)
from .fetch_types import AreaOutcome, error_text
from .http import BudgetExhaustedError, Transport
from .provenance import decide_provenance
from .quantization import has_shard_suffix, identity_stem, non_weight_role, package_identity_key, parse_hf_quant
from . import tensor_quant

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
# R8-1: the only `type` values the tree endpoint sends; anything else is a shape error.
_TREE_ENTRY_TYPES: frozenset[str] = frozenset({"file", "directory"})


@dataclass(frozen=True)
class BaseModelMeta:
    """What `fetch_base_model_meta` learned about one base model, independent of any package."""

    publisher: str
    parameters_b: float | None
    architecture: Architecture


def target_owners(targets: Sequence[str]) -> list[str]:
    """The owners a base model's package targets name, each once, in the targets' own order.

    One area per owner, so this is the list `fetch.run_fetch` iterates. The targets come from
    `config.package_targets`, the one set the collision validator, the fetch and provenance
    share: the configured `packagers` and the base model's own owner crossed with the
    `<name>-GGUF`/`repo_aliases` names, plus the owner-bound `repos` entries. An owner that
    appears only through a `repos` entry is therefore fetched too -- a full `owner/name` in the
    configuration is the user naming that one repository, not an open door for the owner's other
    repositories, which the generated names never reach.
    """
    return list(dict.fromkeys(target.split("/", 1)[0] for target in targets))


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
    targets: Sequence[str],
    run_at: datetime,
    previous_by_key: dict[tuple[str, str, str], Package] | None = None,
    build_tensor: bool = False,
) -> AreaOutcome:
    """Fetch every target of `base_model` that belongs to `owner` and assemble their packages.

    `targets` is the base model's whole package target set (`config.package_targets`); this
    area covers exactly those of them under `owner`, one after the other. The area is `complete`
    only when every one of them was worked through -- a target that simply does not exist counts
    as worked through -- and `incomplete` with the *first* genuine failure otherwise, which
    leaves the owner's previous stock untouched (`state.merge_snapshot`). That is why a whole
    owner is one area rather than one target: the merge identifies an area by
    `(source, base_model, owner)` and, on `complete`, deactivates every package of that area it
    does not contain, so two targets of one owner reported separately would deactivate each
    other. The full set is also what `decide_provenance` checks a repository against.

    `previous_by_key` (F6, default `None`) maps a package identity key to that package as it
    stood in the previous snapshot; when a freshly assembled package shares its identity with
    one that carried an `Approval`, that approval is carried forward into `decide_provenance` so
    an approval bound to still-current content survives this fetch.

    `build_tensor` (AP9-SV2-P, fix round 0b) gates a tensor (safetensors) weight group past the
    plain `unresolved`/`format` shape every non-GGUF package otherwise carries: `False` (the
    default, every caller before this round) leaves it exactly as AP9-SV1 always did -- no
    `config.json` request, no safe-path check, no `decide_provenance` call beyond the format
    lock. `True` is reached only from the guided mode's fetch step, only for a machine with vLLM
    (`runtimes.machine_runtime_keys`, P1's own gate): it resolves the base model's own repository
    (always a target since AP9-UX2 P3) alongside whatever P1's tensor search wrote into `repos`,
    both through the one ordinary `package_targets` mechanism.
    """
    packages: list[Package] = []
    for repo in [target for target in targets if target.split("/", 1)[0] == owner]:
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
        # The relation the provenance check needs is declared in `tags` or in `cardData`; both
        # arrive in this same model-info answer, so reading it costs no further request. They are
        # passed on **exactly as the registry sent them**, never cleaned up: `check_relation` is
        # what judges a shape the Hub does not publish, and turning a malformed `tags` into `[]`
        # or a malformed `cardData` into `None` here would hide that judgement -- a repository the
        # search shows as `metadata_conflict` would reach `from_metadata` in the fetch.
        tags = info.get("tags")
        card_data = info.get("cardData")

        try:
            entries = _fetch_tree(transport, repo, sha)
            packages.extend(
                _assemble_packages(
                    repo,
                    sha,
                    base_model,
                    tags,
                    card_data,
                    targets,
                    entries,
                    run_at,
                    previous_by_key or {},
                    transport,
                    build_tensor,
                )
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

    R8-1 (fix-round 7): "names a type" was still too loose -- `{"type": "garbage"}` also fell
    through the `!= "file"` skip, so a tree of unknown types was the same empty, complete area.
    The registry sends exactly `"file"` and `"directory"`; anything else is a shape error.
    """
    files: list[PackageFile] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"tree entry is not an object: {entry!r}")
        entry_type = entry.get("type")
        if entry_type not in _TREE_ENTRY_TYPES:
            raise ValueError(
                f"tree entry 'type' is not one of {sorted(_TREE_ENTRY_TYPES)}: {entry_type!r}"
            )
        if entry_type == "directory":
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


def _group_quantization(
    package_format: str,
    representative: str,
    group_files_all: list[PackageFile],
    repo: str,
    sha: str,
    transport: Transport,
    tensor_label_cache: dict[str, tuple[str | None, Literal["config"] | None]],
    build_tensor: bool,
) -> tuple[str | None, Literal["filename", "config"] | None]:
    """`(quantization, quantization_source)` for one weight group: a GGUF filename, a tensor
    repo's `config.json` (cached in `tensor_label_cache`, fetched at most once per repo), or
    `(None, None)` for anything else (`_assemble_packages`, P2).

    `build_tensor` (fix round 0b): a tensor group costs no `config.json` request and no safe-path
    check at all when it is `False` -- exactly AP9-SV1's own behavior, on a machine without vLLM
    among its `runtimes`.
    """
    if package_format == "gguf":
        quantization = parse_hf_quant(representative)
        return quantization, ("filename" if quantization is not None else None)
    if package_format != "tensor" or not build_tensor:
        return None, None
    _check_safe_tensor_paths(group_files_all)
    if "label" not in tensor_label_cache:
        tensor_label_cache["label"] = _fetch_tensor_quant(transport, repo, sha)
    return tensor_label_cache["label"]


def _assemble_packages(
    repo: str,
    sha: str,
    base_model: BaseModelSpec,
    # `tags` and `card_data` are the registry's raw values, of whatever shape it sent; the
    # relation check is what decides whether a shape is one the Hub actually publishes.
    tags: object,
    card_data: object,
    targets: Sequence[str],
    entries: list[dict],
    run_at: datetime,
    previous_by_key: dict[tuple[str, str, str], Package],
    transport: Transport,
    build_tensor: bool,
) -> list[Package]:
    files = _files_from_tree(entries)
    groups, extra_files = _weight_groups(files)
    # P2: one `config.json` request serves every tensor group of this repo (two stems of one
    # repo read it once, Codex r1-7); fetched on first use, never when the repo has no tensor
    # group at all (an all-GGUF repo costs nothing extra). A one-entry cache, not a plain local:
    # `_group_quantization` has to tell "not fetched yet" apart from "fetched, no label".
    tensor_label_cache: dict[str, tuple[str | None, Literal["config"] | None]] = {}

    packages: list[Package] = []
    for group_files in groups.values():
        stub = _group_stub(
            repo, sha, base_model, group_files, extra_files, run_at, transport, tensor_label_cache, build_tensor
        )
        stub, approvals = _carry_forward_approval(stub, previous_by_key)
        if stub.format == "tensor" and not build_tensor:
            # Fix round 0b/1 (Codex finding 3): without `build_tensor` this run asks nothing new
            # for a tensor group -- but a fact this project already has at this exact revision is
            # not re-locked just because this run cannot afford to re-check it. `_carried_tensor_facts`
            # is the one place that tells "nothing known yet" apart from "known, at this revision".
            carried = _carried_tensor_facts(stub, previous_by_key)
            if carried is None:
                provenance, unresolved_reason = "unresolved", "format"
            else:
                stub = stub.model_copy(
                    update={"quantization": carried.quantization, "quantization_source": carried.quantization_source}
                )
                provenance, unresolved_reason = carried.provenance, carried.unresolved_reason
        else:
            provenance, unresolved_reason = decide_provenance(
                stub, base_model, tags, approvals, hf_card_data=card_data, hf_targets=targets
            )
        packages.append(_finalize(stub, provenance, unresolved_reason))
    return packages


def _weight_groups(files: list[PackageFile]) -> tuple[dict[str, list[PackageFile]], list[PackageFile]]:
    """`files` split into weight groups (by `identity_stem`) and the non-weight extras every
    group carries alongside its own weights."""
    weight_files = [f for f in files if f.role in ("weights", "weights_shard")]
    extra_files = [f for f in files if f.role in ("mmproj", "other")]
    groups: dict[str, list[PackageFile]] = {}
    for file in weight_files:
        groups.setdefault(identity_stem(file.name), []).append(file)
    return groups, extra_files


def _group_stub(
    repo: str,
    sha: str,
    base_model: BaseModelSpec,
    group_files: list[PackageFile],
    extra_files: list[PackageFile],
    run_at: datetime,
    transport: Transport,
    tensor_label_cache: dict[str, tuple[str | None, Literal["config"] | None]],
    build_tensor: bool,
) -> Package:
    """One weight group's `Package`, `unresolved`/`format` until `decide_provenance` judges it."""
    representative = group_files[0].name
    package_format = _package_format(representative)
    group_files_all = [*group_files, *extra_files]
    quantization, quantization_source = _group_quantization(
        package_format, representative, group_files_all, repo, sha, transport, tensor_label_cache, build_tensor
    )
    return Package(
        source="huggingface",
        repo=repo,
        revision=sha,
        base_model_hf_repo=base_model.hf_repo,
        format=package_format,
        files=group_files_all,
        complete=shards_complete(group_files),
        quantization=quantization,
        quantization_source=quantization_source,
        default_context=None,
        provenance="unresolved",
        unresolved_reason="format",
        approval=None,
        observed_at=run_at,
        last_seen=run_at,
        active=True,
    )


def _check_safe_tensor_paths(files: list[PackageFile]) -> None:
    """P2: every start file of a tensor group (weights and the small extras alike) must be a
    safe, cache-relative path. The message never carries the unsafe path itself; the caller
    (`fetch_hf_area`'s own `except Exception` clause, the same one any other shape problem goes
    through) prefixes it with the repo, so the final text reads exactly `<repo>: unsafe file name`.
    """
    for file in tensor_quant.start_files(files):
        if not tensor_quant.is_safe_repo_path(file.name):
            raise ValueError("unsafe file name")


def _fetch_tensor_quant(transport: Transport, repo: str, sha: str) -> tuple[str | None, Literal["config"] | None]:
    """One tensor repo's quantization label, read from its `config.json` at `sha` (P2).

    Any failure but a budget exhaustion -- a transport error, a status other than `200`, a body
    over `tensor_quant.MAX_CONFIG_JSON_BYTES`, or JSON that is not an object -- resolves to
    `(None, None)`: the label is a documented extra on top of the package (P2), never a reason
    to end the area. `BudgetExhaustedError` is not caught here and ends the area like any other
    exhausted budget (CONTRACTS.md, "Request budget").
    """
    try:
        response = transport("GET", f"https://huggingface.co/{repo}/resolve/{sha}/config.json")
    except BudgetExhaustedError:
        raise
    except Exception:
        return None, None
    if response.status != 200 or len(response.body) > tensor_quant.MAX_CONFIG_JSON_BYTES:
        return None, None
    try:
        config = response.json()
    except Exception:
        return None, None
    if not isinstance(config, dict):
        return None, None
    return tensor_quant.tensor_quant_label(config)


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


def _carried_tensor_facts(stub: Package, previous_by_key: dict[tuple[str, str, str], Package]) -> Package | None:
    """The previous package this tensor group's identity and **revision** match, if that earlier
    run ever resolved it past the format lock (fix round 1, Codex finding 3).

    Revision-bound, unlike the approval carry-forward above (which only matches identity and the
    base model): `package_identity_key` alone says nothing about the file *content* a label was
    read from, and a new commit may be a different model entirely. Only an exact `revision` match
    -- the same commit this run's tree fetch just read -- makes a previous quantization label or
    provenance verdict still a fact about *this* package; a previous package of a different
    revision is simply not found here, same as one this project has never seen at all. The base
    model is not checked separately: `base_model_hf_repo` is not part of either package's repo
    identity, and the revision match already pins the exact content this verdict is about.
    """
    previous = previous_by_key.get(package_identity_key(stub))
    if previous is None or previous.revision != stub.revision:
        return None
    if previous.provenance == "unresolved":
        return None
    return previous
