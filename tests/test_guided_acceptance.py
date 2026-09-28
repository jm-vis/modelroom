"""Acceptance of AP9-UX over the real path: a guided run, its snapshot, its document (ADR 0009).

Every run starts in a folder that holds nothing but the example configuration without EuroLLM
(`fixture_support.example_without_eurollm`): two families, five packagers, the machines
`workstation` (the writer, this machine) and `inference-server` (no profile). The run writes the
configuration and the snapshot itself; the fixture transport answers the search, the fetch and --
through `packager_probes` -- every packager repository the fetch asks about. Blocks are found by
`profile_id`, never by the label: all three server profiles are called `server`.

Every test checks its preconditions first -- every area of the fetch complete, the release of both
models as its catalog states it -- and every "no ..." stands next to a check that the thing left out
is in the folder. Nothing outside `tmp_path` is written, and nothing comes from a rented machine:
the snapshot is the run's own, the profiles are the recorded or derived fixtures of the repository.
"""

from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from modelroom.catalog import Catalog, load_catalog
from modelroom.config import Configuration, MachineConfig, load_config
from modelroom.contracts import Package
from modelroom.dialog import FileAsker
from modelroom.fit import compute_fit_v2
from modelroom.guided import run_guided
from modelroom.measurements import Scenario
from modelroom.profile import HardwareProfile
from modelroom.quantization import QUANT_ORDER
from modelroom.ranking import ModelFacts
from modelroom.render import MachineProfile, build_render_document
from modelroom.render_cmd import render_with_config
from modelroom.state import read_snapshot

from fixture_support import (
    DEEPSEEK_OLLAMA_NAME,
    FIXTURES,
    FOUR_L4_ID,
    TWO_A100_ID,
    TWO_L4_ID,
    area_problems,
    build_transport,
    export_of,
    guided_transport_mapping,
    json_response,
    loadtest_daemon,
    offline_daemon,
    offline_servers,
    packager_probes,
    windows_probes,
    write_example_without_eurollm,
)
from modelroom.http import Response

RUN1 = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)
RUN2 = datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc)
THIS_ID = "7d3e0a91c4b25f68"
QWEN = "Qwen/Qwen3.5-9B"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
UNSLOTH = "unsloth/Qwen3.5-9B-GGUF"
QWEN_GGUF = "Qwen/Qwen3.5-9B-GGUF"
DEEPSEEK_REPO = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
DEEPSEEK_SHA = "eb48357c179d34dbf515983f798dfb8752a0f261"
CHECKED = date(2026, 9, 28)
SCENARIO = Scenario(context_requested=8192, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1)
_GRAPHICS = ("gpu", "gpu_split")
_FULL = ("F16", "BF16", "F32")


# --- the two test catalogs (Review r3-M2) -------------------------------------------------------------


def _catalog(qwen_legacy: bool) -> Catalog:
    """The shipped catalog with two models changed, validated anew (`model_dump` -> `model_validate`)."""
    data = load_catalog().model_dump()
    for family in data["families"]:
        for model in family["models"]:
            if model["hf_repo"] == QWEN and qwen_legacy:
                model.update(latest=False, latest_source=None, latest_checked=None, successor="Qwen/Qwen3.8-27B")
            elif model["hf_repo"] == QWEN:
                model.update(latest=True, latest_source="https://huggingface.co/Qwen", latest_checked=CHECKED, successor=None)
            elif model["hf_repo"] == DEEPSEEK:
                model.update(
                    latest=True, latest_source="https://huggingface.co/deepseek-ai", latest_checked=CHECKED, successor=None
                )
    return Catalog.model_validate(data)


CATALOG_NOT_LEGACY = _catalog(qwen_legacy=False)
CATALOG_QWEN_LEGACY = _catalog(qwen_legacy=True)
AGES = {
    "not legacy": {QWEN: ("latest", None), DEEPSEEK: ("latest", None)},
    "qwen legacy": {QWEN: ("legacy", "Qwen/Qwen3.8-27B"), DEEPSEEK: ("latest", None)},
}


# --- one run of the real path, and its preconditions ----------------------------------------------------


def _run(tmp_path: Path, answers: dict, catalog: Catalog, *, daemon=None, extra: dict | None = None) -> tuple[int, list[str], Path]:
    here = tmp_path / "results"
    config = load_config(write_example_without_eurollm(here))
    (tmp_path / "home").mkdir(exist_ok=True)
    mapping = {**packager_probes(config), **guided_transport_mapping(), **(extra or {})}
    lines: list[str] = []
    code = run_guided(
        FileAsker(answers),
        here=here,
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport(mapping),
        # Twice: step 1 draws one id for the takeover question before the measurement draws the one it writes.
        probes=windows_probes(ids=[THIS_ID, THIS_ID]),
        daemon=daemon if daemon is not None else offline_daemon(),
        servers=offline_servers(),
        now=RUN1,
        out=lines.append,
        catalog=catalog,
        colored=False,
    )
    return code, lines, here


def _preconditions(code: int, here: Path, ages: str) -> None:
    """Every area complete (Review r3-H1), exit 0, and the release of both models as its catalog says."""
    assert area_problems(here) == []
    assert code == 0
    log = json.loads((here / "state" / "search.json").read_text(encoding="utf-8"))
    found = {entry["base_model"]: entry for entry in log["models"]}
    for model, (age, successor) in AGES[ages].items():
        assert (found[model]["age"], found[model]["successor"]) == (age, successor), model
        assert found[model]["release_basis"] != "computed", model
    config = load_config(here / "modelroom.toml")
    assert "deepseek-r1" in [family.name for family in config.families]
    assert "inference-server" in config.machines
    snapshot = read_snapshot(config)
    assert any(package.base_model_hf_repo == DEEPSEEK for package in snapshot.packages)


def _payload(here: Path) -> dict:
    return json.loads((here / "docs" / "models.json").read_text(encoding="utf-8"))


def _block(payload: dict, profile_id: str) -> dict:
    return next(block for block in payload["machines"] if (block.get("profile") or {}).get("profile_id") == profile_id)


# --- the oracle: the least compression of one model that fits, graphics memory first ----------------------


def _step(package: Package) -> tuple[int, int]:
    """The compression step, larger is less compressed; 16/32 bit one step, the smaller weights first."""
    weights = sum(f.size_bytes for f in package.files if f.role in ("weights", "weights_shard"))
    if package.quantization in _FULL:
        return QUANT_ORDER.index("F16"), -weights
    return (QUANT_ORDER.index(package.quantization), 0) if package.quantization in QUANT_ORDER else (-1, 0)


def _oracle(here: Path, model: str, profile: HardwareProfile, reserve_ram: float, reserve_vram: float) -> Package:
    """The package the rule must put first for `model` on `profile`: short and on its own."""
    config = load_config(here / "modelroom.toml")
    snapshot = read_snapshot(config)
    spec = next(spec for spec in snapshot.base_models if spec.hf_repo == model)
    machine = MachineConfig(reserve_ram_gib=reserve_ram, reserve_vram_gib=reserve_vram, writer=False)
    fitting = []
    for package in snapshot.packages:
        if package.base_model_hf_repo != model or not (package.active and package.complete):
            continue
        if package.provenance not in ("from_metadata", "approved"):
            continue
        fit = compute_fit_v2(profile, package, spec, SCENARIO, machine)
        if fit.fit_class in ("ample", "fits", "tight"):
            fitting.append((package, fit))
    on_card = [package for package, fit in fitting if fit.mode in _GRAPHICS] or [package for package, _fit in fitting]
    return max(on_card, key=_step)


def _first_is_the_oracle(here: Path, block: dict, model: str) -> None:
    profile = HardwareProfile.model_validate(block["profile"])
    expected = _oracle(here, model, profile, block["reserve_ram_gib"], block["reserve_vram_gib"])
    first = block["ranked"][0]
    assert first["base_model_hf_repo"] == model
    assert (first["quantization"], first["fit"]["mode"] in _GRAPHICS) == (expected.quantization, True)


# --- A1 -----------------------------------------------------------------------------------------------------


def test_a_guided_run_on_four_l4_ranks_the_least_compression_of_the_chosen_model_first(tmp_path: Path):
    answers = {
        "results": "here",
        "machines": ["this-machine", "import"],
        "machine_class": "laptop",
        "import_file": str(export_of("four_l4_server.json", tmp_path / "exports")),
        "search": "qwen",
        "filter_owners": True,
        "select": [UNSLOTH, QWEN_GGUF],
        "users": 1,
        "context": "8192",
        "load_test": False,
    }
    code, _lines, here = _run(tmp_path, answers, CATALOG_NOT_LEGACY)

    _preconditions(code, here, "not legacy")
    payload = _payload(here)
    block = _block(payload, FOUR_L4_ID)
    _first_is_the_oracle(here, block, QWEN)
    assert {row["base_model_hf_repo"] for row in block["ranked"]} == {QWEN}  # one model: all its packages
    text = json.dumps(payload) + (here / "docs" / "models.md").read_text(encoding="utf-8")
    assert "DeepSeek" not in text
    assert "inference-server" not in [block["machine"] for block in payload["machines"]]
    assert "no_profile" not in [block["status"] for block in payload["machines"]]
    assert sorted(block["profile"]["profile_id"] for block in payload["machines"]) == sorted([THIS_ID, FOUR_L4_ID])


# --- A2 -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ages", ["not legacy", "qwen legacy"])
def test_the_release_of_step_two_reaches_the_ranking(tmp_path: Path, ages):
    answers = {
        "results": "here",
        "machines": ["this-machine", "import"],
        "machine_class": "laptop",
        "import_file": str(export_of("four_l4_server.json", tmp_path / "exports")),
        "search": "qwen",
        "filter_owners": True,
        "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK_REPO],
        "users": 1,
        "context": "8192",
        "load_test": False,
    }
    catalog = CATALOG_QWEN_LEGACY if ages == "qwen legacy" else CATALOG_NOT_LEGACY
    code, _lines, here = _run(tmp_path, answers, catalog)

    _preconditions(code, here, ages)
    rows = _block(_payload(here), FOUR_L4_ID)["ranked"]
    if ages == "not legacy":
        assert [row["base_model_hf_repo"] for row in rows[:2]] == [QWEN, DEEPSEEK]
        return
    assert [row["base_model_hf_repo"] for row in rows[:2]] == [DEEPSEEK, QWEN]
    assert rows[0]["quantization"] == "Q4_K_M"
    assert rows[0]["note"]["text"].endswith("The only package of this model that fits.")
    # `modelroom render` of the same folder knows no release: no model is legacy, the size decides.
    assert render_with_config(load_config(here / "modelroom.toml"), now=RUN2) == 0
    assert _block(_payload(here), FOUR_L4_ID)["ranked"][0]["base_model_hf_repo"] == QWEN


# --- A3 -----------------------------------------------------------------------------------------------------

# SYNTHETIC: a second DeepSeek build, `Q6_K`, added to the recorded tree of the same repository. Its
# size is the one the public Hub tree of that repository states (6 725 900 800 bytes); the object ids
# and the xet hash are fixed synthetic literals of the right shape, not recorded values.
Q6_K_ENTRY = {
    "type": "file",
    "oid": "5b9e0c1d2a3f4b5c6d7e8f90a1b2c3d4e5f60718",
    "size": 6725900800,
    "lfs": {
        "oid": "6c1f2e3d4c5b6a79880716253443526170819a0b1c2d3e4f5a6b7c8d9e0f1a2b",
        "size": 6725900800,
        "pointerSize": 135,
    },
    "xetHash": "7d2e3f40516273849505a6b7c8d9e0f1021324354657687980a1b2c3d4e5f607",
    "path": "DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf",
}


def _deepseek_tree_with_q6_k() -> dict:
    tree = json.loads((FIXTURES / "hf_unsloth_deepseek_r1_qwen3_8b_gguf_tree.json").read_text(encoding="utf-8"))
    position = next(index for index, entry in enumerate(tree) if entry["path"].endswith("Q4_K_M.gguf")) + 1
    tree.insert(position, copy.deepcopy(Q6_K_ENTRY))
    url = f"https://huggingface.co/api/models/{DEEPSEEK_REPO}/tree/{DEEPSEEK_SHA}?recursive=true"
    return {("GET", url): Response(status=200, headers={}, body=json.dumps(tree).encode("utf-8"))}


def test_the_measured_package_of_this_run_stays_in_the_result(tmp_path: Path):
    answers = {
        "results": "here",
        "machines": ["this-machine"],
        "machine_class": "laptop",
        "search": "qwen",
        "filter_owners": True,
        "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK_REPO],
        "users": 1,
        "context": "8192",
        "load_test": True,
        "load_test_packages": [DEEPSEEK_OLLAMA_NAME],
    }
    code, lines, here = _run(tmp_path, answers, CATALOG_NOT_LEGACY, daemon=loadtest_daemon(), extra=_deepseek_tree_with_q6_k())

    _preconditions(code, here, "not legacy")
    snapshot = read_snapshot(load_config(here / "modelroom.toml"))
    deepseek = {p.quantization: p for p in snapshot.packages if p.base_model_hf_repo == DEEPSEEK and p.source == "huggingface"}
    assert {"Q6_K", "Q4_K_M"} <= set(deepseek)
    block = _block(_payload(here), THIS_ID)
    laptop = HardwareProfile.model_validate(block["profile"])
    spec = next(spec for spec in snapshot.base_models if spec.hf_repo == DEEPSEEK)
    q6 = compute_fit_v2(laptop, deepseek["Q6_K"], spec, SCENARIO, MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True))
    assert (q6.fit_class in ("ample", "fits", "tight"), q6.mode) == (True, "gpu")

    rows = block["ranked"]
    assert [(row["base_model_hf_repo"], row["quantization"]) for row in rows] == [
        (QWEN, _oracle(here, QWEN, laptop, 8.0, 1.0).quantization),
        (DEEPSEEK, "Q6_K"),
        (DEEPSEEK, "Q4_K_M"),
    ]
    qwen_total = sum(1 for p in snapshot.packages if p.base_model_hf_repo == QWEN and p.active)  # an upper bound only
    assert rows[0]["note"]["text"].endswith(" packages of this model that fit.") and "Best of " in rows[0]["note"]["text"]
    assert int(rows[0]["note"]["text"].rsplit("Best of ", 1)[1].split()[0]) <= qwen_total
    assert (rows[1]["speed_tps"], rows[1]["note"]["text"].endswith("Best of 2 packages of this model that fit.")) == (None, True)
    appended = rows[2]
    assert (appended["measurement_group"], appended["speed_tps"] is not None) == (0, True)
    assert "ranking.shown_for_measurement" in appended["note"]["facts"]
    assert appended["note"]["text"].endswith("Shown for its measurement; the rule puts it at place 2 of this model's 2.")

    stripped = [line.strip() for line in lines]
    # The `shown` statement of this block, wrapped between its pieces where it is wider than a line.
    shown = next(index for index, line in enumerate(stripped) if line.startswith("shown ") and "one row per model" in line)
    assert "#3 is shown for its measurement, not by the rule" in " ".join(stripped[shown : shown + 2])
    total = block["ranked_total"]
    markdown = (here / "docs" / "models.md").read_text(encoding="utf-8")
    assert f"Showing 3 rows of {total} ranked packages: one row per model, then 1 measured." in markdown
    speed = next(line for line in stripped if line.startswith("speed ") and "fastest" in line)
    assert speed.split()[1:3] == ["1", "measured"]
    assert "fastest DeepSeek-R1-0528-Qwen3-8B at " in speed
    install = next(line for line in stripped if line.startswith("install "))
    assert install.endswith(f"ollama pull hf.co/{rows[0]['package_identity'][1]}:{rows[0]['quantization']}")


# --- A4 -----------------------------------------------------------------------------------------------------

# Fixed facts of the two models (the rule's own input), with and without Qwen3.5-9B `legacy`.
FACTS = {
    "not legacy": {QWEN: ModelFacts(9.65, False), DEEPSEEK: ModelFacts(8.19, False)},
    "qwen legacy": {QWEN: ModelFacts(9.65, True), DEEPSEEK: ModelFacts(8.19, False)},
}
# The expected first row per machine: the model and its pool are fixed, so a broken rule turns red and
# not only a wrong oracle; the compression is the oracle's, on 2 x A100 the 16-bit step as well.
EXPECTED = {
    "laptop": {"not legacy": QWEN, "qwen legacy": DEEPSEEK},
    "two-a100": {"not legacy": QWEN, "qwen legacy": DEEPSEEK},
    "two-l4": {"not legacy": QWEN, "qwen legacy": DEEPSEEK},
    "four-l4": {"not legacy": QWEN, "qwen legacy": DEEPSEEK},
}


def _server(name: str) -> HardwareProfile:
    return HardwareProfile.model_validate_json((FIXTURES / "profiles_v4" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("ages", ["not legacy", "qwen legacy"])
def test_one_rule_for_every_hardware(tmp_path: Path, ages):
    answers = {
        "results": "here",
        "machines": ["this-machine"],
        "machine_class": "laptop",
        "search": "qwen",
        "filter_owners": True,
        "select": [UNSLOTH, QWEN_GGUF, DEEPSEEK_REPO],
        "users": 1,
        "context": "8192",
        "load_test": False,
    }
    code, _lines, here = _run(tmp_path, answers, CATALOG_NOT_LEGACY)
    _preconditions(code, here, "not legacy")
    config = load_config(here / "modelroom.toml")
    laptop = HardwareProfile.model_validate_json((config.paths.hardware_dir / f"{THIS_ID}.json").read_text(encoding="utf-8"))
    profiles = {
        "laptop": laptop,
        "two-a100": _server("two_a100_server.json"),
        "two-l4": _server("two_l4_server.json"),
        "four-l4": _server("four_l4_server.json"),
    }
    assert [profiles[name].profile_id for name in ("two-a100", "two-l4", "four-l4")] == [TWO_A100_ID, TWO_L4_ID, FOUR_L4_ID]
    # The reserves written out: 8 GiB of RAM and 1 GiB per card, `[defaults]` of the example (Review r3-N5).
    machines = {name: {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": False, "profile": p.profile_id} for name, p in profiles.items()}
    four = Configuration.model_validate({**config.model_dump(), "machines": machines})
    found = {name: MachineProfile("ranked", profile.display_name, profile, None) for name, profile in profiles.items()}

    document = build_render_document(
        four, read_snapshot(config), found, {}, SCENARIO, RUN2, models=FACTS[ages]
    )

    for block in document.machines:
        first = block.ranked[0]
        model = EXPECTED[block.machine][ages]
        assert (first.base_model_hf_repo, first.fit.mode in _GRAPHICS) == (model, True), block.machine
        expected = _oracle(here, model, block.profile, 8.0, 1.0)
        assert first.quantization == expected.quantization, block.machine
        if block.machine == "two-a100" and model == QWEN:
            assert first.quantization in _FULL
