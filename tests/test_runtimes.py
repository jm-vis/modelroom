"""Tests for modelroom.runtimes: the table "Laufzeit als Eintrag" (AP9-SV2-L).

AP9-SV2-L moves Ollama onto `RUNTIMES` without a behavior change -- these tests hold the table's
own shape (every field set, `accepts`/`fit` reachable and equal to today's direct calls) and the
dispatch a later sub-package (AP9-LS) will use for a real second row. The full byte-identical
render/pull/install path stays proven by `tests/test_guided_screen.py`'s golden files and the
rest of `tests/test_pull_choice.py`, `tests/test_guided_install.py`, `tests/test_render.py` --
none of them touched here on purpose (Kernkriterium: a pure refactor).
"""

from __future__ import annotations

import copy
from dataclasses import fields

import pytest

from modelroom import local_servers, runtimes
from modelroom.config import MachineConfig
from modelroom.contracts import BaseModelSpec, Package
from modelroom.examples import EXAMPLES
from modelroom.fit import compute_fit_v2
from modelroom.measurements import Scenario
from modelroom.profile import HardwareProfile

MACHINE = MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)


def _base_model() -> BaseModelSpec:
    return BaseModelSpec.model_validate(EXAMPLES["BaseModelSpec"])


def _package(**changes) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload.update(changes)
    return Package.model_validate(payload)


def _ollama_package(**changes) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload.update(
        source="ollama", repo=None, revision=None, ollama_name="nova:7b", manifest_digest="sha256:" + "a" * 64
    )
    payload.update(changes)
    return Package.model_validate(payload)


def _profile(**changes) -> HardwareProfile:
    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    payload.update(changes)
    return HardwareProfile.model_validate(payload)


def _scenario(context: int = 8192) -> Scenario:
    return Scenario(context_requested=context, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1)


# --- the table's own shape ------------------------------------------------------------------------


def test_every_runtime_has_every_field():
    """Every row of `RUNTIMES` sets every field of `Runtime`, structurally -- not that a vLLM one
    runs (AP9-SV2-L point 2: `VLLM` carries only `server` and a real `accepts` here)."""
    for runtime in runtimes.RUNTIMES:
        for field in fields(runtimes.Runtime):
            assert getattr(runtime, field.name) is not None, f"{runtime.key}.{field.name} is unset"


def test_runtime_keys_are_ollama_and_vllm_in_that_order():
    assert [runtime.key for runtime in runtimes.RUNTIMES] == ["ollama", "vllm"]


def test_each_runtimes_server_is_local_servers_own_entry():
    """`RUNTIMES.server` is the very `LocalServer` object `LOCAL_SERVERS` already holds (SV1), not
    a copy: the start screen, step 4 and the card go on reading `LOCAL_SERVERS` unchanged, and a
    later runtime reaches all three the moment its `server` is one more entry there too."""
    by_key = {server.key: server for server in local_servers.LOCAL_SERVERS}
    assert runtimes.OLLAMA.server is by_key["ollama"]
    assert runtimes.VLLM.server is by_key["vllm"]


@pytest.mark.parametrize("field", ["detect", "fit", "command", "model_place", "on_disk"])
def test_every_vllm_field_not_built_yet_fails_loudly_if_called(field):
    """`VLLM.accepts` refuses every package (below), so none of these run in AP9-SV2-L; a call
    still fails loudly, naming the sub-package that builds it, rather than guessing quietly."""
    with pytest.raises(NotImplementedError, match="AP9-SV2-"):
        getattr(runtimes.VLLM, field)()


# --- accepts: which packages a runtime claims ------------------------------------------------------


def test_ollama_accepts_a_gguf_package_from_hugging_face():
    assert runtimes.OLLAMA.accepts(_package(format="gguf")) is True


def test_ollama_accepts_a_package_the_ollama_library_packed():
    assert runtimes.OLLAMA.accepts(_ollama_package()) is True


def test_ollama_refuses_a_tensor_package_from_hugging_face():
    assert runtimes.OLLAMA.accepts(_package(format="tensor")) is False


def test_vllm_accepts_nothing_yet():
    """AP9-SV2-L point 2: no Tensor package claims a row before AP9-SV2-P builds the Tensor
    package class and AP9-SV2-F builds vLLM's fit -- `accepts` is `lambda package: False`, on
    purpose, whatever the package."""
    assert runtimes.VLLM.accepts(_package(format="tensor")) is False
    assert runtimes.VLLM.accepts(_package(format="gguf")) is False


def test_accepting_runtimes_names_ollama_alone_for_a_gguf_package():
    assert [runtime.key for runtime in runtimes.accepting_runtimes(_package(format="gguf"))] == ["ollama"]


def test_accepting_runtimes_is_empty_for_a_tensor_package():
    assert runtimes.accepting_runtimes(_package(format="tensor")) == ()


def test_accepting_runtimes_grows_with_a_second_runtime_monkeypatched_in(monkeypatch):
    """The probe AP9-LS needs before it gives a second real runtime a `gguf`-accepting `accepts`
    (AP9-SV2-L, "Laufzeit als Eintrag", the Paar-Regel): two runtimes both claiming one package."""
    third = runtimes.OLLAMA.__class__(**{**runtimes.OLLAMA.__dict__, "key": "third", "name": "Third"})
    monkeypatch.setattr(runtimes, "RUNTIMES", (runtimes.OLLAMA, third))

    assert [runtime.key for runtime in runtimes.accepting_runtimes(_package(format="gguf"))] == ["ollama", "third"]


# --- fit: the dispatch is compute_fit_v2, unconditionally, exactly as before ----------------------


def test_fit_for_a_gguf_package_matches_compute_fit_v2_directly():
    package, profile, base_model, scenario = _package(format="gguf", default_context=4096), _profile(), _base_model(), _scenario()

    assert runtimes.fit_for(profile, package, base_model, scenario, MACHINE) == compute_fit_v2(
        profile, package, base_model, scenario, MACHINE
    )


def test_fit_for_a_package_no_runtime_accepts_falls_back_to_compute_fit_v2():
    """Render judged every package before AP9-SV2-L too -- `compute_fit_v2`'s own `unknown` result
    for a non-gguf package (`fit._package_reason`) -- so the fallback here must say the very same,
    byte for byte, not a new `not_covered` reason of its own."""
    package, profile, base_model, scenario = _package(format="tensor"), _profile(), _base_model(), _scenario()

    assert runtimes.fit_for(profile, package, base_model, scenario, MACHINE) == compute_fit_v2(
        profile, package, base_model, scenario, MACHINE
    )
