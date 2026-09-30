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


# --- machine scoping: a runtime pairs only where the machine has it (Codex r1, fix 3) -------------


def test_machine_runtime_keys_defaults_to_ollama_alone_when_the_profile_names_none():
    """Every profile before AP9-SV2-0, and every machine whose last measurement recorded nothing in
    `runtimes`, reads as Ollama's own default -- today's only shape, and the one every existing
    fixture and golden file already assumes."""
    assert runtimes.machine_runtime_keys(_profile(runtimes=[])) == frozenset({"ollama"})
    assert runtimes.machine_runtime_keys(None) == frozenset({"ollama"})


def test_machine_runtime_keys_reads_the_profiles_own_list_where_it_names_any():
    assert runtimes.machine_runtime_keys(_profile(runtimes=["vllm"])) == frozenset({"vllm"})


def test_accepting_on_machine_narrows_to_the_machines_own_runtimes(monkeypatch):
    import types

    third = runtimes.OLLAMA.__class__(
        **{**runtimes.OLLAMA.__dict__, "key": "third", "name": "Third", "accepts": lambda package: package.format == "gguf"}
    )
    monkeypatch.setattr(runtimes, "RUNTIMES", (runtimes.OLLAMA, third))
    package = _package(format="gguf")
    # A duck-typed stand-in, not a real `HardwareProfile`: its `runtimes` field is locked to the
    # schema's own two keys (`RuntimeKey`), so a fixture third key could never validate there --
    # `accepting_on_machine` itself only ever reads `.runtimes` off whatever it is handed.
    with_it = types.SimpleNamespace(runtimes=["ollama", "third"])
    without_it = types.SimpleNamespace(runtimes=[])

    assert [runtime.key for runtime in runtimes.accepting_on_machine(package, with_it)] == ["ollama", "third"]
    assert [runtime.key for runtime in runtimes.accepting_on_machine(package, without_it)] == ["ollama"]


# --- reaches every place a registered runtime should (AP9-SV2-L brief, Codex r1 fix 1) -------------


def test_a_third_in_memory_runtime_reaches_start_step4_card_render_and_install(monkeypatch):
    """A runtime that exists only as one more row of `RUNTIMES` reaches the start screen, step 4,
    the card (`local_servers.py`, derived from the table -- Codex r1, fix 1), the render's fit
    dispatch, and the install/pull list (Codex r1, fixes 2-4) -- without a second place to add it
    to. No golden file changes: `RUNTIMES` itself is monkeypatched back after the test, and every
    other test of the suite (the golden-screen ones included) never sees a third entry.
    """
    from modelroom.dialog import Choice
    from modelroom.intro import Fact
    from modelroom.pull_choice import pull_rows
    from modelroom.views_install import install_target

    from fixture_rows import document as fixture_document

    third = runtimes.Runtime(
        key="third",
        name="Third",
        server=local_servers.LocalServer(
            key="third",
            name="Third",
            origin="http://127.0.0.1:9999",
            transport=lambda run: None,
            start_fact=lambda origins: Fact("server", "Third", "stated by nobody"),
            offer=lambda run, config, scenario, results_dir: local_servers.Offer(
                rows=(Choice("third", "the third server was asked"),)
            ),
            measure=lambda run, config, scenario, results_dir, offer, picked: "Third measured",
            card_facts=lambda run: (Fact("server", "Third", "on the card"),),
        ),
        detect=lambda *args, **kwargs: None,
        hello_field="third",
        accepts=lambda package: package.format == "gguf",
        fit=runtimes.OLLAMA.fit,
        command=lambda package: "third-pull-name",
        model_place=lambda *args, **kwargs: [],
        on_disk=lambda *args, **kwargs: False,
        words=runtimes.RuntimeWords(package_prefix="third · ", key_text=None, question="Which package should be installed?"),
    )
    monkeypatch.setattr(runtimes, "RUNTIMES", (runtimes.OLLAMA, third))
    monkeypatch.setattr(local_servers, "LOCAL_SERVERS", tuple(entry.server for entry in runtimes.RUNTIMES))

    # 1) start screen: `local_servers.start_rows` loops `LOCAL_SERVERS`, derived from `RUNTIMES`.
    from modelroom.daemon import FixtureDaemon

    offline = FixtureDaemon({}, unreachable="no daemon here")
    start, _reachable = local_servers.start_rows(offline, local_servers.ServerRun({}))
    assert any(fact.value == "Third" and fact.note == "stated by nobody" for fact in start)

    # 2) step 4: every registered server's `offer` is read (`load_tests` asks the one question and
    # builds the one list over exactly this loop; the row itself is what reaches it here).
    assert third.server.offer(None, None, None, None).rows == (Choice("third", "the third server was asked"),)

    # 3) the card: `local_servers.card_facts` loops the same derived list.
    assert local_servers.card_facts(run=None) == (Fact("server", "Third", "on the card"),)

    # 4) render: the fit dispatch reads the table in order; Ollama (still first) wins for a
    # package both accept, exactly as `fit_for`'s own tests hold (no behavior change there).
    package = _package(format="gguf", default_context=4096)
    assert runtimes.accepting_runtimes(package)[0].key == "ollama"

    # 5) install: the pull list gives the package a row for each runtime the machine has
    # (`<runtime.key> <name>`, unprefixed for Ollama -- Codex r1, fixes 2-4), and the install line
    # dispatches through the first one that accepts it there; reordering the table changes which
    # one wins, proving the choice comes from `RUNTIMES`, not a name hardcoded anywhere here.
    doc = fixture_document([package])
    doc = doc.model_copy(
        update={
            "machines": [
                block.model_copy(update={"profile": block.profile.model_copy(update={"runtimes": ["ollama", "third"]})})
                for block in doc.machines
            ]
        }
    )
    rows = pull_rows(doc, "workstation", packages=[package])
    by_runtime = {row.runtime: row.name for row in rows}
    assert set(by_runtime) == {"ollama", "third"}
    assert by_runtime["third"] == "third third-pull-name"
    assert by_runtime["ollama"] is not None and by_runtime["ollama"] != by_runtime["third"]

    ollama_wins = install_target(doc.machines[0], "workstation", packages=[package])
    assert ollama_wins is not None and ollama_wins[1] == by_runtime["ollama"]

    monkeypatch.setattr(runtimes, "RUNTIMES", (third, runtimes.OLLAMA))  # reorder: Third now first
    third_wins = install_target(doc.machines[0], "workstation", packages=[package])
    assert third_wins is not None and third_wins[1] == "third-pull-name"  # Third's own command, unprefixed
