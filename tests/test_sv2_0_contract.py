"""Tests for AP9-SV2-0: the snapshot schema step, the new `QUANT_ORDER` labels, the `Fit`/
`Rating` move to `contracts_fit.py`, and `HardwareProfile.runtimes`.

Behavioral red proof (SOP-0003): every test below that is not marked "control test" in its
docstring ran red against the unchanged code of `a1ca1a2` through the real entry points
(`load_snapshot`, `Snapshot.model_validate`, `Package.model_validate`, `QUANT_ORDER`,
`HardwareProfile.model_validate`) before this sub-package's code changes; the JUnit result of
that run is kept outside the repository (`red-sv2-0.xml`), not committed. A "control test" was
already green at `a1ca1a2` -- it pins behavior this sub-package must not change, which a
collection error could never prove, so it is not counted as red proof.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from modelroom.contracts import (
    SNAPSHOT_SCHEMA_VERSION,
    Fit,
    Package,
    Rating,
    SchemaVersionError,
    Snapshot,
    load_snapshot,
)
from modelroom.examples import EXAMPLES
from modelroom.profile import HardwareProfile
from modelroom.quantization import QUANT_ORDER


def _snapshot_payload(schema_version: int) -> dict:
    payload = json.loads(json.dumps(EXAMPLES["Snapshot"]))
    payload["schema_version"] = schema_version
    return payload


# --- snapshot schema step: 1 reads as 2, 3 is refused ---------------------------------------


def test_snapshot_1_reads_as_2():
    """`load_snapshot`, the real reader `fetch`/`render` use, accepts a stored schema-1 file and
    reads it as 2 with no other change to its content (E8; the precedent of ADR 0007's "old
    spelling reads as the new one", but as a real version step this time)."""
    payload = _snapshot_payload(1)
    snapshot = load_snapshot(payload)
    assert snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION == 2
    assert snapshot.packages[0].base_model_hf_repo == payload["packages"][0]["base_model_hf_repo"]
    assert snapshot.packages[0].quantization == payload["packages"][0]["quantization"]


def test_a_snapshot_3_is_refused_with_exit_3():
    """Control test: version 3 was already outside the accepted range at `a1ca1a2` (`[1, 2)`)
    and stays outside it after this sub-package widens the range to `[1, 3)` -- the boundary
    case most likely to slip on an off-by-one range edit. `SchemaVersionError` is `cli.py`'s
    signal to exit `3` (CONTRACTS.md, "Exit codes")."""
    payload = _snapshot_payload(3)
    with pytest.raises(SchemaVersionError) as excinfo:
        load_snapshot(payload)
    message = str(excinfo.value)
    assert "3" in message
    assert "snapshot" in message


def test_snapshot_direct_validate_still_enforces_the_current_version():
    """`Snapshot.model_validate` alone (no normalization) accepts only the version `fetch`
    writes; reading an old file needs `load_snapshot`, the "second line of defense" its
    docstring describes. Not counted as red proof on its own (it would already have raised at
    `a1ca1a2`, just for a different reason -- version 1 was already the only accepted one
    there); it is here to pin the *current* boundary the step above moves."""
    payload = _snapshot_payload(1)
    with pytest.raises(ValidationError):
        Snapshot.model_validate(payload)


# --- Package: a Hugging Face tensor package may be from_metadata, an Ollama one never --------


def _tensor_package(*, source: str, provenance: str, unresolved_reason: str | None) -> dict:
    base: dict = {
        "source": source,
        "base_model_hf_repo": "acme/Nova-7B",
        "format": "tensor",
        "files": [
            {
                "name": "model-00001-of-00002.safetensors",
                "role": "weights_shard",
                "size_bytes": 4500000000,
                "digest": None,
            },
            {
                "name": "model-00002-of-00002.safetensors",
                "role": "weights_shard",
                "size_bytes": 4400000000,
                "digest": None,
            },
        ],
        "complete": True,
        "quantization": None,
        "default_context": None,
        "provenance": provenance,
        "unresolved_reason": unresolved_reason,
        "approval": None,
        "observed_at": "2026-09-22T09:00:00Z",
        "last_seen": "2026-09-22T09:00:00Z",
        "active": True,
    }
    if source == "huggingface":
        base.update(
            repo="packager/Nova-7B-AWQ",
            revision="9f8e7d6c5b4a3928170615243342516078960a1b",
        )
    else:
        base.update(
            ollama_name="nova:7b-mlx",
            manifest_digest="sha256:" + "ab" * 32,
            aliases=[],
        )
    return base


def test_an_ollama_tensor_package_stays_unresolved_format():
    """Control test (both halves already held at `a1ca1a2` and must still hold): every non-
    `gguf` package, Ollama included, was already forced `unresolved`/`format` there -- the old
    invariant did not look at `source` at all. This pins that an Ollama `tensor` package (an MLX
    tag, noted 2026-09-28) still cannot become `from_metadata` once the Hugging Face exception
    is added; it keeps no vLLM row (P3)."""
    valid = _tensor_package(source="ollama", provenance="unresolved", unresolved_reason="format")
    Package.model_validate(valid)

    with pytest.raises(ValidationError):
        Package.model_validate({**valid, "provenance": "from_metadata", "unresolved_reason": None})


def test_a_hf_tensor_package_may_be_from_metadata():
    """Red at `a1ca1a2`: the old invariant forced every non-`gguf` package `unresolved`/`format`
    regardless of `source`, so this construction raised `ValidationError` there. After this
    sub-package's change, a Hugging Face `tensor` package may carry `from_metadata` (the
    producer that sets it is SV2-P's `provenance.py`, out of scope here; this test proves only
    that the contract now allows the shape)."""
    payload = _tensor_package(source="huggingface", provenance="from_metadata", unresolved_reason=None)
    package = Package.model_validate(payload)
    assert package.provenance == "from_metadata"
    assert package.unresolved_reason is None


def test_package_quantization_source_is_additive_and_defaults_to_none():
    """Red at `a1ca1a2`: `quantization_source` did not exist, so `Package.model_validate` on a
    payload that carries no such key already worked there -- what is red is reading the field
    back off the built model, which raised `AttributeError` at `a1ca1a2` since the attribute did
    not exist. `EXAMPLES["Package"]` itself sets `quantization_source` explicitly (it documents
    the GGUF case), so this test drops the key instead of relying on that example."""
    payload = {k: v for k, v in EXAMPLES["Package"].items() if k != "quantization_source"}
    package = Package.model_validate(payload)
    assert package.quantization_source is None


# --- QUANT_ORDER: the new vLLM labels, every existing GGUF step keeps its place --------------


_NEW_QUANT_LABELS = ("W4A16", "AWQ", "GPTQ-4", "GPTQ-8", "W8A8", "FP8")


def test_quant_order_keeps_every_gguf_step_in_its_order():
    """Red at `a1ca1a2`: none of the new labels existed in `QUANT_ORDER` yet."""
    for label in _NEW_QUANT_LABELS:
        assert label in QUANT_ORDER

    existing_in_order = (
        "Q2_K",
        "Q3_K_M",
        "Q4_K_S",
        "Q4_K_M",
        "UD-Q4_K_XL",
        "MXFP4",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
        "UD-Q8_K_XL",
        "F16",
        "BF16",
        "F32",
    )
    indices = [QUANT_ORDER.index(label) for label in existing_in_order]
    assert indices == sorted(indices), "an existing GGUF step moved relative to another one"


def test_quant_order_places_the_new_vllm_labels_between_the_documented_gguf_steps():
    """Red at `a1ca1a2`: the new labels were absent, so `QUANT_ORDER.index` raised `ValueError`.
    Placement per the brief's table: `W4A16`/`AWQ`/`GPTQ-4` between `Q4_K_M` and `UD-Q4_K_XL`;
    `GPTQ-8`/`W8A8`/`FP8` between `Q8_0` and `UD-Q8_K_XL`."""
    assert (
        QUANT_ORDER.index("Q4_K_M")
        < QUANT_ORDER.index("W4A16")
        < QUANT_ORDER.index("AWQ")
        < QUANT_ORDER.index("GPTQ-4")
        < QUANT_ORDER.index("UD-Q4_K_XL")
    )
    assert (
        QUANT_ORDER.index("Q8_0")
        < QUANT_ORDER.index("GPTQ-8")
        < QUANT_ORDER.index("W8A8")
        < QUANT_ORDER.index("FP8")
        < QUANT_ORDER.index("UD-Q8_K_XL")
    )


def test_quant_order_full_precision_step_is_unaffected_by_the_insertions():
    """Control test: `ranking._FULL_STEP` reads `QUANT_ORDER.index("F16")` at import time rather
    than a literal number, so it already tracked `F16`'s position correctly at `a1ca1a2` and
    keeps doing so once six labels are inserted ahead of it."""
    from modelroom.ranking import _FULL_STEP

    assert _FULL_STEP == QUANT_ORDER.index("F16")


# --- Fit/Rating: the move to contracts_fit.py must not change the re-export ------------------


def test_contracts_reexports_fit_and_rating():
    """Control test: `Fit`/`Rating` were already importable from `modelroom.contracts` and
    already validated their examples at `a1ca1a2`; the extraction to `contracts_fit.py` must
    keep every import site (`document.py`, `fit.py`, `guided_models.py`, `ranking.py`,
    `render.py`, `views.py`, the tests) working unchanged."""
    Fit.model_validate(EXAMPLES["Fit"])
    Rating.model_validate(EXAMPLES["Rating"])


def test_fit_and_rating_are_defined_in_contracts_fit_and_reexported_identically():
    """Not counted as red proof: `modelroom.contracts_fit` does not exist at `a1ca1a2`, so this
    test only collection-errors there, which the task's red-proof rule excludes. It still proves
    the important property once the module exists: `contracts.Fit` and `contracts_fit.Fit` are
    the very same class object, not a copy, so isinstance checks and Pydantic's own model
    identity stay correct everywhere."""
    from modelroom import contracts_fit

    assert Fit is contracts_fit.Fit
    assert Rating is contracts_fit.Rating


# --- HardwareProfile.runtimes: additive, empty list excluded (ADR 0011 precedent) ------------


def _profile_payload(**overrides) -> dict:
    payload = json.loads(json.dumps(EXAMPLES["HardwareProfile"]))
    payload.update(overrides)
    return payload


def test_a_profile_without_runtimes_writes_as_before():
    """Red at `a1ca1a2`, but for a trivial reason: `.runtimes` did not exist yet, so reading it
    back raised `AttributeError` there. What the test actually pins is not trivial: once the
    field exists, a profile built without it still dumps to exactly the same keys as before
    (an empty `runtimes` is excluded, the `adapters` precedent of ADR 0011) -- byte-identical
    serialization for the unchanged case is F6's GF decision."""
    profile = HardwareProfile.model_validate(_profile_payload())
    assert profile.runtimes == []
    dumped = profile.model_dump(mode="json")
    assert "runtimes" not in dumped


def test_a_profile_with_runtimes_round_trips():
    """Red at `a1ca1a2`: `runtimes` was an unknown field under `extra="forbid"`, so this
    construction raised `ValidationError` there."""
    profile = HardwareProfile.model_validate(_profile_payload(runtimes=["vllm"]))
    assert profile.runtimes == ["vllm"]
    dumped = profile.model_dump(mode="json")
    assert dumped["runtimes"] == ["vllm"]


def test_an_unknown_runtime_key_is_rejected():
    """Control test, for a reason worth stating: `runtimes` itself was an unknown field at
    `a1ca1a2` (`extra="forbid"`), so this already raised `ValidationError` there -- the right
    outcome for the wrong reason. After this sub-package it raises because `"lmstudio"` is
    outside `RuntimeKey` (`"ollama" | "vllm"`), which is the rule this test exists to check."""
    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(_profile_payload(runtimes=["lmstudio"]))
