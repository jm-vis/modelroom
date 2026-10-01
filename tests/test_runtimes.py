"""Tests for modelroom.runtimes: the one table of local-server runtimes.

Ollama sits on `RUNTIMES` without a behavior change -- these tests hold the table's own shape
(every field set, `accepts`/`fit` reachable and equal to the direct calls), the pairing of a
package with every runtime of a machine that accepts it, and a third runtime that exists only as
one more row of the table reaching every place a runtime is read. The byte-identical
render/pull/install path of Ollama stays proven by the golden files of
`tests/test_guided_screen.py` and the rest of the suite.
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
    runs (`VLLM` carries only `server` and a real `accepts` so far)."""
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


@pytest.mark.parametrize("field", ["detect", "fit", "install_name", "command", "model_place", "on_disk"])
def test_every_vllm_field_not_built_yet_fails_loudly_if_called(field):
    """`VLLM.accepts` refuses every package (below), so none of these runs; a call still fails
    loudly, naming the package that builds it, rather than guessing quietly."""
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
    """No Tensor package claims a row before the Tensor package class and vLLM's fit exist --
    `accepts` is `lambda package: False`, on purpose, whatever the package."""
    assert runtimes.VLLM.accepts(_package(format="tensor")) is False
    assert runtimes.VLLM.accepts(_package(format="gguf")) is False


def test_accepting_runtimes_names_ollama_alone_for_a_gguf_package():
    assert [runtime.key for runtime in runtimes.accepting_runtimes(_package(format="gguf"))] == ["ollama"]


def test_accepting_runtimes_is_empty_for_a_tensor_package():
    assert runtimes.accepting_runtimes(_package(format="tensor")) == ()


def test_accepting_runtimes_grows_with_a_second_runtime_monkeypatched_in(monkeypatch):
    """A second runtime with a `gguf`-accepting `accepts`: two runtimes both claiming one package."""
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
    """Render judged every package before the table too -- `compute_fit_v2`'s own `unknown` result
    for a non-gguf package (`fit._package_reason`) -- so the fallback here must say the very same,
    byte for byte, not a new `not_covered` reason of its own."""
    package, profile, base_model, scenario = _package(format="tensor"), _profile(), _base_model(), _scenario()

    assert runtimes.fit_for(profile, package, base_model, scenario, MACHINE) == compute_fit_v2(
        profile, package, base_model, scenario, MACHINE
    )


# --- machine scoping: a runtime pairs only where the machine has it --------------------------------


def test_machine_runtime_keys_defaults_to_ollama_alone_when_the_profile_names_none():
    """Every profile before AP9-SV2-0, and every machine whose last measurement recorded nothing in
    `runtimes`, reads as Ollama's own default -- today's only shape, and the one every existing
    fixture and golden file already assumes."""
    assert runtimes.machine_runtime_keys(_profile(runtimes=[])) == frozenset({"ollama"})
    assert runtimes.machine_runtime_keys(None) == frozenset({"ollama"})


def test_machine_runtime_keys_reads_the_profiles_own_list_where_it_names_any():
    assert runtimes.machine_runtime_keys(_profile(runtimes=["vllm"])) == frozenset({"vllm"})


# --- machine_has_vllm: the one decision both the tensor search and the fetch step read (fix
# round 0c) -- "runtimes, or detected live" (P1), never tested separately at either call site ---


def _served() -> "ServedModel":
    from modelroom.vllm_server import ServedModel

    return ServedModel(id="acme/Nova-7B", root="acme/Nova-7B", max_model_len=4096)


@pytest.mark.parametrize(
    ("live", "expected"),
    [
        (None, False),
        ("asks for an API key", False),  # E5: a key prompt does not count as detected
        (_served(), True),
    ],
    ids=["not-live", "key-prompt", "live-serving"],
)
def test_machine_has_vllm_without_a_profile_at_all(live, expected):
    assert runtimes.machine_has_vllm(None, live) is expected


@pytest.mark.parametrize(
    ("runtimes_field", "live", "expected"),
    [
        ([], None, False),
        ([], "asks for an API key", False),
        ([], _served(), True),
        (["vllm"], None, True),
        (["vllm"], "asks for an API key", True),
        (["vllm"], _served(), True),
        (["ollama"], None, False),
        (["ollama"], _served(), True),
    ],
    ids=[
        "no-runtimes-not-live",
        "no-runtimes-key-prompt",
        "no-runtimes-live-serving",
        "stored-vllm-not-live",
        "stored-vllm-key-prompt",
        "stored-vllm-live-serving",
        "stored-ollama-only-not-live",
        "stored-ollama-only-live-serving",
    ],
)
def test_machine_has_vllm_every_combination_of_profile_and_live_detection(runtimes_field, live, expected):
    """Profile yes/no (stored `runtimes` naming `vllm` or not) crossed with live yes/no/key-only:
    `machine_has_vllm` is `True` exactly when either source says so, never when neither does."""
    assert runtimes.machine_has_vllm(_profile(runtimes=runtimes_field), live) is expected


def test_both_call_sites_read_the_one_function_not_their_own_logic():
    """Fix round 0c (the SV2-L lesson against two separately-tested paths, "Codex-Tropfen"):
    `guided._fetch_step` and `guided_search._tensor_search_step` each compute their gate with
    exactly one call to `runtimes.machine_has_vllm`, patched here to prove neither one re-derives
    the answer on its own -- a stub that always disagrees with the real function makes both sites
    behave as the stub says, not as the real inputs would."""
    import inspect

    import modelroom.guided as guided_module
    import modelroom.guided_search as guided_search_module

    assert "machine_has_vllm(" in inspect.getsource(guided_module._fetch_step)
    assert "machine_has_vllm(" in inspect.getsource(guided_search_module._tensor_search_step)
    # Neither source reads `machine_runtime_keys` or `ServedModel` on its own any more -- the one
    # function in `runtimes.py` is the only place either fact is read.
    assert "machine_runtime_keys" not in inspect.getsource(guided_module)
    assert "machine_runtime_keys" not in inspect.getsource(guided_search_module)


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


# --- a third runtime, only one more row of `RUNTIMES`, reaches every place ------------------------


def _third(**changes) -> runtimes.Runtime:
    """A runtime that exists only as one more row of the table: its own server, name and command."""
    from modelroom.intro import Fact

    server = local_servers.LocalServer(
        key="third",
        name="Third",
        origin="http://127.0.0.1:9999",
        transport=lambda run: None,
        start_fact=lambda origins: Fact("server", "Third", "stated by nobody"),
        offer=lambda run, config, scenario, results_dir: run.screen.note("the third server was asked") or local_servers.Offer(),
        measure=lambda run, config, scenario, results_dir, offer, picked: "Third measured",
        card_facts=lambda run: (Fact("server", "Third", "on the card"),),
    )
    fields_ = dict(
        key="third",
        name="Third",
        server=server,
        detect=lambda *args, **kwargs: None,
        hello_field="third",
        accepts=lambda package: package.format == "gguf",
        fit=runtimes.OLLAMA.fit,
        install_name=lambda package: f"acme/nova:{package.quantization}",
        command=lambda name: f"third get {name}",
        model_place=lambda *args, **kwargs: [],
        on_disk=lambda target, packages: frozenset(),
        words=runtimes.RuntimeWords(package_prefix="third · ", key_text=None, question="Which package should be installed?"),
    )
    fields_.update(changes)
    return runtimes.Runtime(**fields_)


def _on_machine(doc, keys: list[str]):
    """`doc` with its one machine's profile naming exactly `keys` as its runtimes."""
    blocks = [block.model_copy(update={"profile": block.profile.model_copy(update={"runtimes": keys})}) for block in doc.machines]
    return doc.model_copy(update={"machines": blocks})


def _pull_step(tmp_path, doc, packages, answers: dict) -> tuple[list[str], dict]:
    """The pull of step 5 over `doc`, as the guided run calls it: its lines, and the lists asked."""
    from modelroom.guided_install import PullContext, pull_step
    from modelroom.pull_choice import ambiguous_names

    from test_pull_choice import _Recording, _run

    asker = _Recording(answers)
    run, lines = _run(tmp_path, asker)
    pull_step(run, PullContext(doc, "workstation", packages, ambiguous_names(packages)), True)
    return [line.strip() for line in lines], asker.lists


def test_a_third_in_memory_runtime_reaches_start_step4_card_render_and_install(tmp_path, monkeypatch):
    """Only `RUNTIMES` is patched. A whole guided run shows the third server on the start screen,
    asks it in step 4 and puts its row on the card, and -- first in the table -- the render judges
    every GGUF package with its `fit`, while the machine's pull and install stay Ollama's, since
    the machine's profile does not name it. On a machine whose profile names it, the install line
    under the table, the pull list, the answer taken from it and the collisions all read its entry:
    its install name, its command, its key in the list value."""
    from modelroom.fit import compute_fit_v2
    from modelroom.pull_choice import NOT_UNIQUE, pull_collisions, pull_rows
    from modelroom.screen import plain
    from modelroom.views import RunView, terminal_blocks
    from modelroom.views_install import NOT_UNIQUE_INSTALL

    from fixture_rows import document, package
    from fixture_support import offline_daemon
    from test_guided_install import ANSWERS, FIRST_NAME, _run
    from test_guided_vllm import _card, _step

    judged = []

    def fit(profile, package_, *args):
        judged.append(package_.format)
        return compute_fit_v2(profile, package_, *args)

    first = _third(fit=fit)
    monkeypatch.setattr(runtimes, "RUNTIMES", (first, *runtimes.RUNTIMES))
    (tmp_path / "run" / "results").mkdir(parents=True)
    (tmp_path / "run" / "home").mkdir()

    code, lines, _asker = _run(tmp_path / "run", {**ANSWERS, "pull": False}, offline_daemon())

    assert code == 0
    assert "server    Third                   stated by nobody" in lines  # start screen
    assert "the third server was asked" in _step(lines, 4)  # step 4: its offer
    assert _step(lines, 4)[-1] == "✓ Measurement     Third measured"  # step 4: its measurement
    assert "server    Third                   on the card" in [line.strip() for line in _card(lines)]  # the card
    assert judged and set(judged) == {"gguf"}  # the render: the fit of the first runtime accepting it
    assert f"install   #1  ollama pull {FIRST_NAME}" in lines  # this machine has Ollama alone

    third = _third()
    monkeypatch.setattr(runtimes, "RUNTIMES", (*runtimes.RUNTIMES[1:], third))
    q4 = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
    doc = _on_machine(document([q4]), ["third"])

    view = RunView(install_for="workstation", packages=[q4])
    shown = [plain(line).strip() for line in terminal_blocks(doc, "workstation", view=view)]
    assert "install   #1  third get acme/nova:Q4_K_M" in shown  # the install line: its command
    assert not any("ollama pull" in line for line in shown)

    said, lists = _pull_step(tmp_path / "pull", doc, [q4], {"pull_package": "third acme/nova:Q4_K_M", "pull": False})
    assert [choice.value for choice in lists["pull_package"]] == ["third acme/nova:Q4_K_M", "none"]
    assert "install   #1  third get acme/nova:Q4_K_M" in said  # the answer taken, and its command

    twin = package("unsloth/Nova-7B-GGUF", "sub/Nova-7B-Q4_K_M.gguf", "Q4_K_M")
    collisions = pull_collisions([q4, twin])
    assert len(collisions["third acme/nova:Q4_K_M"]) == 2  # its key in the collision
    ambiguous = frozenset(collisions)
    assert [row.locked for row in pull_rows(doc, "workstation", ambiguous, packages=[q4, twin])] == [NOT_UNIQUE]
    view = RunView(install_for="workstation", ambiguous=ambiguous, packages=[q4, twin])
    shown = [plain(line).strip() for line in terminal_blocks(doc, "workstation", view=view)]
    assert f"install   #1 {NOT_UNIQUE_INSTALL}" in shown


def test_a_third_in_memory_runtime_accepting_gguf_gives_a_second_row_per_package(tmp_path, monkeypatch):
    """A second GGUF runtime on the same machine: every package gets two rows, in the order of the
    table rows (the same ranking rule, the same ranks), each list value its own -- Ollama's exactly
    as before, the other `<key> <name>` -- and the same package under two runtimes is no collision.
    The answer picks the row of the runtime it names, and the command shown is that runtime's."""
    from modelroom.pull_choice import pull_collisions, pull_rows

    from fixture_rows import document, package

    third = _third(install_name=runtimes.OLLAMA.install_name)
    monkeypatch.setattr(runtimes, "RUNTIMES", (*runtimes.RUNTIMES, third))
    name = "hf.co/unsloth/Nova-7B-GGUF"
    q8 = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q8_0.gguf", "Q8_0")
    q4 = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
    doc = _on_machine(document([q8, q4]), ["ollama", "third"])

    rows = pull_rows(doc, "workstation", packages=[q8, q4])

    assert [(row.rank, row.runtime, row.value) for row in rows] == [
        (1, "ollama", f"{name}:Q8_0"),
        (1, "third", f"third {name}:Q8_0"),
        (2, "ollama", f"{name}:Q4_K_M"),
        (2, "third", f"third {name}:Q4_K_M"),
    ]
    assert pull_collisions([q8, q4]) == {}

    said, lists = _pull_step(tmp_path, doc, [q8, q4], {"pull_package": f"third {name}:Q4_K_M", "pull": False})
    assert [choice.value for choice in lists["pull_package"]][:4] == [row.value for row in rows]
    assert f"install   #2  third get {name}:Q4_K_M" in said
    assert not any(line.startswith("install") and "ollama pull" in line for line in said)


# --- what the Ollama daemon holds frees Ollama's rows alone ----------------------------------------


def _held_by_ollama(q4):
    """A free disk too small for `q4`'s download, on which the Ollama daemon holds `q4` already."""
    from modelroom.quantization import package_identity_key
    from modelroom.room import GIB, Disk

    return Disk(int(1 * GIB), "C:", held=frozenset({package_identity_key(q4)}))


def test_what_ollama_holds_frees_no_row_of_another_runtime_in_the_pull_list(monkeypatch):
    """`pull_rows` with a disk and no map of holdings: `disk.held` is the Ollama daemon's, so the
    Ollama row needs no disk, while the third runtime's row of the same package, which holds
    nothing itself, is locked for its download."""
    from modelroom.pull_choice import pull_rows

    from fixture_rows import document, package

    monkeypatch.setattr(runtimes, "RUNTIMES", (*runtimes.RUNTIMES, _third(install_name=runtimes.OLLAMA.install_name)))
    q4 = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
    doc = _on_machine(document([q4]), ["ollama", "third"])

    rows = pull_rows(doc, "workstation", disk=_held_by_ollama(q4), packages=[q4])

    assert [(row.runtime, row.over_disk, row.locked is None) for row in rows] == [("ollama", False, True), ("third", True, False)]
    assert [(row.runtime, row.over_disk) for row in pull_rows(doc, "workstation", disk=_held_by_ollama(q4))] == [("ollama", False)]


def test_the_install_line_checks_the_disk_against_its_own_runtimes_holdings(monkeypatch):
    """The install line under the table decides the row's runtime before the disk: on a machine
    with the third runtime alone, what the Ollama daemon holds does not make its download fit, so
    no command is offered -- the line says no package fits. With Ollama on the machine, the same
    holding frees the row and the line is Ollama's, as before."""
    from modelroom.room import LiveRoom
    from modelroom.screen import plain
    from modelroom.views import RunView, terminal_blocks

    from fixture_rows import document, package

    monkeypatch.setattr(runtimes, "RUNTIMES", (*runtimes.RUNTIMES, _third()))
    q4 = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")

    def install(keys):
        view = RunView(install_for="workstation", live={"workstation": LiveRoom(disk=_held_by_ollama(q4))}, packages=[q4])
        lines = [plain(line).strip() for line in terminal_blocks(_on_machine(document([q4]), keys), "workstation", view=view)]
        return [line for line in lines if line.startswith("install")]

    assert len(install(["third"])) == 1 and install(["third"])[0].startswith("install   no package fits")
    assert install(["ollama"]) == ["install   #1  ollama pull hf.co/unsloth/Nova-7B-GGUF:Q4_K_M"]


def test_what_another_runtime_holds_itself_frees_its_row_in_the_install_line_as_in_the_pull_list(tmp_path, monkeypatch):
    """The third runtime's own `on_disk` finds the package, the Ollama daemon holds nothing, the
    disk is too small for the download. The pull list marks the row "local already" and leaves it
    a choice; the install line under the table, given the same holdings
    (`guided_install.other_runtimes_held`, as the guided run passes them), offers its command
    instead of saying no package fits."""
    from modelroom.guided_install import PullContext, other_runtimes_held, pull_step
    from modelroom.pull_choice import ambiguous_names
    from modelroom.quantization import package_identity_key
    from modelroom.room import GIB, Disk, LiveRoom
    from modelroom.screen import plain
    from modelroom.views import RunView, terminal_blocks

    from fixture_rows import document, package
    from test_pull_choice import _Recording, _run

    q4 = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
    holds = frozenset({package_identity_key(q4)})
    monkeypatch.setattr(runtimes, "RUNTIMES", (*runtimes.RUNTIMES, _third(on_disk=lambda target, packages: holds)))
    doc = _on_machine(document([q4]), ["third"])
    disk = Disk(int(1 * GIB), "C:")
    asker = _Recording({"pull_package": "none"})
    run, _lines = _run(tmp_path, asker)
    context = PullContext(doc, "workstation", [q4], ambiguous_names([q4]), disk)

    held = other_runtimes_held(run, context)
    view = RunView(install_for="workstation", live={"workstation": LiveRoom(disk=disk)}, packages=[q4], held=held)
    shown = [plain(line).strip() for line in terminal_blocks(doc, "workstation", view=view)]

    assert held == {"third": holds}
    assert [line for line in shown if line.startswith("install")] == ["install   #1  third get acme/nova:Q4_K_M"]
    pull_step(run, context, True)
    first = asker.lists["pull_package"][0]
    assert (first.value, first.disabled, first.label.endswith("local already")) == ("third acme/nova:Q4_K_M", None, True)


# --- the install line and the pull of step 5 agree, whatever runtimes, holdings, disk and names -----


MACHINES = {"ollama": ["ollama"], "third": ["third"], "ollama+third": ["ollama", "third"], "third+ollama": ["third", "ollama"]}
NAMES = ("unique", "shared", "missing")


def _agreement_cases():
    """Machine x holder x disk x the name of each runtime the machine has -- the name of a runtime
    the machine does not have changes nothing, so it stays `unique` there (pruned)."""
    cases = []
    for label, machine in MACHINES.items():
        for holder in (None, "ollama", "third"):
            for disk_gib in (100, 1):
                for ollama_name in NAMES if "ollama" in machine else ("unique",):
                    for third_name in NAMES if "third" in machine else ("unique",):
                        case_id = f"{label}-held-by-{holder or 'none'}-{'big' if disk_gib == 100 else 'small'}-disk"
                        case_id += f"-ollama-{ollama_name}-third-{third_name}"
                        cases.append(pytest.param(machine, holder, disk_gib, ollama_name, third_name, id=case_id))
    return cases


def _expected(machine, holder, disk_gib, names):
    """The rule, written out: per runtime of the machine, in table order (Ollama before the third,
    whatever order the profile names them in), the download fits where the disk is big or the
    runtime holds the package itself. None fits: no package fits. Else the first that fits and has
    a name of its own is offered; where every one that fits has none, the first of them says why."""
    keys = [key for key in ("ollama", "third") if key in machine]
    fitting = [key for key in keys if disk_gib == 100 or holder == key]
    if not fitting:
        return "no package fits", None
    free = next((key for key in fitting if names[key] == "unique"), None)
    return ("offer", free) if free is not None else ("stop", names[fitting[0]])


@pytest.mark.parametrize("machine, holder, disk_gib, ollama_name, third_name", _agreement_cases())
def test_the_install_line_and_the_pull_step_offer_the_same_row(tmp_path, monkeypatch, machine, holder, disk_gib, ollama_name, third_name):
    """One package; the machine's runtimes, who holds the package, how much disk is free, and
    whether each runtime's install name is its own, shared with another package (a collision) or
    missing. The install line under the table (`terminal_blocks`) and the pull of step 5 with an
    answer file without `pull_package` (`pull_step`) offer the same row -- the same command, or
    "local already" in the pull -- or the same nothing: `no package fits`, `has no command` for a
    shared name, no line at all for a missing one, and no pull."""
    import dataclasses

    from modelroom.guided_install import PullContext, other_runtimes_held, pull_step
    from modelroom.quantization import package_identity_key
    from modelroom.room import GIB, Disk, LiveRoom
    from modelroom.screen import plain
    from modelroom.views import RunView, terminal_blocks
    from modelroom.views_install import NOT_UNIQUE_INSTALL

    from fixture_rows import document, package
    from test_pull_choice import _Recording, _run

    q4 = package("unsloth/Nova-7B-GGUF", "Nova-7B-Q4_K_M.gguf", "Q4_K_M")
    name = "hf.co/unsloth/Nova-7B-GGUF:Q4_K_M"
    names = {"ollama": ollama_name, "third": third_name}
    holds = frozenset({package_identity_key(q4)})
    own = {key: holds if holder == key else frozenset() for key in ("ollama", "third")}

    def install_name(key):
        return (lambda package_: None) if names[key] == "missing" else runtimes.OLLAMA.install_name

    ollama = dataclasses.replace(runtimes.OLLAMA, install_name=install_name("ollama"), on_disk=lambda target, packages: own["ollama"])
    third = _third(install_name=install_name("third"), on_disk=lambda target, packages: own["third"])
    monkeypatch.setattr(runtimes, "RUNTIMES", (ollama, runtimes.VLLM, third))
    doc = _on_machine(document([q4]), machine)
    disk = Disk(int(disk_gib * GIB), "C:", held=own["ollama"])  # the disk read before the render: Ollama's holding
    # A name shared with another package: the collision `pull_collisions` would report for it.
    ambiguous = frozenset({name} if ollama_name == "shared" else set()) | frozenset({f"third {name}"} if third_name == "shared" else set())

    run, lines = _run(tmp_path, _Recording({"pull": False}))
    context = PullContext(doc, "workstation", [q4], ambiguous, disk)
    held = other_runtimes_held(run, context)
    view = RunView(install_for="workstation", ambiguous=ambiguous, live={"workstation": LiveRoom(disk=disk)}, packages=[q4], held=held)
    installs = [plain(line).strip() for line in terminal_blocks(doc, "workstation", view=view)]
    installs = [line for line in installs if line.startswith("install")]
    pull_step(run, context, True)
    pulled = [line.strip() for line in lines if line.strip().startswith("install")]

    kind, what = _expected(machine, holder, disk_gib, names)
    command = {"ollama": f"ollama pull {name}", "third": f"third get {name}"}
    if kind == "offer":
        assert installs == [f"install   #1  {command[what]}"]
        local = "install   #1 is local already · say Yes in step 4 to measure it"
        assert pulled == [local if holder == what else f"install   #1  {command[what]}"]
    else:
        assert pulled == []
        if kind == "no package fits":
            assert len(installs) == 1 and installs[0].startswith("install   no package fits")
        elif what == "shared":
            assert installs == [f"install   #1 {NOT_UNIQUE_INSTALL}"]
        else:
            assert installs == []
