"""Synthetic rows for the tests of the pull list and the repository lines (AP9-UX2 U4, U6).

Not a test module. Everything is built in memory from `EXAMPLES` (the shape of
`tests/test_ranking.py`): packages of the fictitious `acme/Nova-7B` and render documents whose rows
name them. No fixture file is read.
"""

from __future__ import annotations

import copy

from modelroom.contracts import Package
from modelroom.document import RenderDocument
from modelroom.examples import EXAMPLES
from modelroom.quantization import package_identity_key

GIB = 1024**3
BASE = "acme/Nova-7B"


def package(repo: str, file: str, quant: str | None, size: int = 5 * GIB, *, provenance: str = "from_metadata", **changes) -> Package:
    """A GGUF package of `acme/Nova-7B` from `repo` with one weight file `file`."""
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload.update(repo=repo, quantization=quant, provenance=provenance)
    payload["files"] = [{"name": file, "role": "weights", "size_bytes": size, "digest": None}]
    if provenance == "unresolved":
        payload["unresolved_reason"] = "format"
    payload.update(changes)
    return Package.model_validate(payload)


def fillers(count: int, repo: str = "unsloth/Nova-7B-GGUF", avoid: tuple[str, ...] = ()) -> list[Package]:
    """`count` packages of `repo`, each of another quantization of `QUANT_ORDER`, none of `avoid`."""
    from modelroom.quantization import QUANT_ORDER

    quants = [quant for quant in QUANT_ORDER if quant not in avoid][:count]
    assert len(quants) == count
    return [package(repo, f"Nova-7B-{quant}.gguf", quant) for quant in quants]


def ollama_package(name: str, quant: str = "Q4_K_M", size: int = 5 * GIB) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload.update(
        source="ollama", repo=None, revision=None, ollama_name=name, manifest_digest="sha256:" + "a" * 64, quantization=quant
    )
    payload["files"] = [{"name": "model", "role": "weights", "size_bytes": size, "digest": None}]
    return Package.model_validate(payload)


def entry(pkg: Package, rank: int) -> dict:
    """A ranked row of `pkg` at `rank`, unmeasured."""
    row = copy.deepcopy(EXAMPLES["RankedEntry"])
    identity = package_identity_key(pkg)
    row.update(
        package_identity=list(identity),
        packager="ollama" if pkg.source == "ollama" else (pkg.repo or "").split("/", 1)[0],
        quantization=pkg.quantization or "unknown",
        format=pkg.format,
        weights_gib=sum(file.size_bytes for file in pkg.files) / GIB,
        rank=rank,
        measurement_group=1,
        measurement_id=None,
        speed_tps=None,
    )
    return row


def document(packages: list[Package], machine: str = "workstation", *, total: int | None = None) -> RenderDocument:
    """A document whose one machine shows `packages` in this order, ranks 1, 2, ..."""
    return document_multi({machine: packages}, total=total)


def document_multi(per_machine: dict[str, list[Package]], *, total: int | None = None) -> RenderDocument:
    """A document with one block per key of `per_machine`, each showing that machine's packages in
    this order, ranks 1, 2, ... (`pull_other`, A3: a run with more than one chosen machine)."""
    payload = copy.deepcopy(EXAMPLES["RenderDocument"])
    template = payload["machines"][0]
    blocks = []
    for machine, packages in per_machine.items():
        block = copy.deepcopy(template)
        block["machine"] = machine
        block["label"] = machine
        block["ranked"] = [entry(pkg, rank) for rank, pkg in enumerate(packages, start=1)]
        block["ranked_total"] = total if total is not None else len(packages)
        block["not_covered"], block["no_fit"] = [], []
        blocks.append(block)
    payload["machines"] = blocks
    return RenderDocument.model_validate(payload)
