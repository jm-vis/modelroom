"""Fit with every graphics card, and render document schema 3 (decided 2026-09-26).

The pool rule: unified memory as before; one card first -- the largest one, the way Ollama loads a
model; then all cards together (`gpu_split`: each card keeps its reserve, the reserve reported is
that of all cards, nothing capped); then the system memory. With one card no number of a fit
changes. The example architecture needs 1 GiB of KV cache at 8192, so need = weights x 1.10 + 1.5.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.cli import render_with_config
from modelroom.config import MachineConfig
from modelroom.contracts import BaseModelSpec, Fit, Package
from modelroom.document import DOCUMENT_SCHEMA_VERSION, RenderDocument
from modelroom.examples import EXAMPLES
from modelroom.fit import GpuPool, _choose_pool, compute_fit_v2, count_fitting, fit_from_parameters, fit_from_size, profile_memory
from modelroom.guided_context import Checked
from modelroom.guided_models import model_fit
from modelroom.measurements import default_scenario
from modelroom.profile import HardwareProfile
from modelroom.render import _computed_note, _set_aside_note
from modelroom.ranking import SetAside
from modelroom.state import atomic_write_json
from modelroom.views import _pool_pieces

from fixture_support import with_profile
from test_cli import RUN2, _rendered

GIB = 1024**3
MACHINE = MachineConfig(reserve_ram_gib=8.0, reserve_vram_gib=1.0, writer=True)


def _package(weights_gib: float) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["files"][0]["size_bytes"] = int(weights_gib * GIB)
    return Package.model_validate(payload)


def _cards(*sizes: float, ram: float = 128.0) -> HardwareProfile:
    payload = copy.deepcopy(EXAMPLES["HardwareProfile"])
    total = round(sum(sizes), 2)
    payload.update(
        ram_physical_gib=ram,
        vram_gib=total,
        gpu_state="multi_gpu" if len(sizes) > 1 else "measured",
        gpu_name="Nova GPU",
        gpus=[{"index": i, "name": "Nova GPU", "vram_gib": size, "vram_source": "nvidia-smi"} for i, size in enumerate(sizes)],
        machine_class="server",
        llmfit_crosscheck={"ram_physical": {"status": "absent", "own_gib": ram}, "vram": {"status": "absent", "own_gib": total}},
    )
    return HardwareProfile.model_validate(payload)


def _fit(profile: HardwareProfile, weights_gib: float) -> Fit:
    base = BaseModelSpec.model_validate(EXAMPLES["BaseModelSpec"])
    return compute_fit_v2(profile, _package(weights_gib), base, default_scenario(), MACHINE)


# --- the pool rule -------------------------------------------------------------------------------


def test_the_card_pool_of_a_profile_lists_each_card():
    assert profile_memory(_cards(24.0, 24.0)) == GpuPool((24.0, 24.0), False)
    assert profile_memory(_cards(8.0)) == GpuPool((8.0,), False)


def test_a_package_one_card_holds_stays_on_one_card():
    fit = _fit(_cards(24.0, 24.0), 10.0)
    assert (fit.mode, fit.pool_gib, fit.reserve_gib) == ("gpu", 23.0, 1.0)


def test_one_card_first_means_the_largest_card():
    fit = _fit(_cards(16.0, 48.0), 30.0)  # need 34.5: only the 48 GiB card holds it
    assert (fit.mode, fit.pool_gib) == ("gpu", 47.0)


def test_a_package_only_all_cards_hold_is_spread_with_the_reserve_on_each_card():
    fit = _fit(_cards(24.0, 24.0), 30.0)  # need 34.5 > 23, <= 46
    assert (fit.mode, fit.pool_gib, fit.reserve_gib, fit.fit_class) == ("gpu_split", 46.0, 2.0, "good")


def test_a_spread_package_is_not_capped_and_may_be_perfect():
    fit = _fit(_cards(24.0, 24.0), 20.0)  # need 23.5 > 23; 23.5 / 46 = 0.51
    assert (fit.mode, fit.fit_class) == ("gpu_split", "perfect")


def test_a_package_no_card_holds_falls_back_to_the_system_memory_capped_at_good():
    fit = _fit(_cards(24.0, 24.0), 50.0)  # need 56.5 > 46
    assert (fit.mode, fit.pool_gib, fit.reserve_gib) == ("cpu_gpu", 120.0, 8.0)
    assert fit.fit_class == "good"


def test_no_card_at_all_is_the_cpu_pool_and_one_card_never_splits():
    assert _choose_pool(12.0, GpuPool(()), 32.0, MACHINE) == ("cpu", 24.0, 8.0, True)
    assert _choose_pool(12.0, GpuPool((8.0,)), 32.0, MACHINE) == ("cpu_gpu", 24.0, 8.0, True)


def test_unified_memory_stays_its_own_pool():
    assert _choose_pool(12.0, GpuPool((), True), 32.0, MACHINE) == ("gpu", 23.0, 9.0, False)


@pytest.mark.parametrize(("sizes", "need"), [((8.0,), 5.0), ((8.0,), 7.5), ((0.5,), 3.0), ((), 3.0), ((12.0,), 11.0)])
def test_every_value_of_a_one_card_fit_is_the_same_as_before(sizes, need):
    """The rule before 2026-09-26, written out: one VRAM minus its reserve, else the RAM."""
    vram = sizes[0] if sizes else 0.0
    before = ("gpu", vram - 1.0, 1.0, False) if need <= vram - 1.0 else ("cpu_gpu" if vram > 0 else "cpu", 24.0, 8.0, True)
    assert _choose_pool(need, GpuPool(sizes), 32.0, MACHINE) == before


def test_the_size_basis_keeps_its_cap_when_spread():
    spread = GpuPool((24.0, 24.0))
    fit = fit_from_size(_package(20.0), spread, 128.0, MACHINE, 8192)
    assert fit.mode == "gpu_split" and fit.fit_class == "good" and fit.basis == "size"
    assert fit_from_parameters(40.0, spread, 128.0, MACHINE, 8192).mode == "gpu_split"  # 22.4 GiB of weights


def test_the_size_entry_points_keep_their_keyword_vram_gib():
    by_keyword = fit_from_parameters(9.0, vram_gib=8.0, ram_gib=31.7, machine_config=MACHINE, context=8192)
    assert by_keyword == fit_from_parameters(9.0, 8.0, 31.7, MACHINE, 8192)
    assert fit_from_size(_package(4.0), vram_gib=8.0, ram_gib=31.7, machine_config=MACHINE, context=8192).mode == "gpu"


def test_a_number_still_reads_as_one_card():
    assert fit_from_parameters(9.0, 8.0, 31.7, MACHINE, 8192) == fit_from_parameters(9.0, GpuPool((8.0,)), 31.7, MACHINE, 8192)
    assert fit_from_parameters(9.0, 0.0, 31.7, MACHINE, 8192).mode == "cpu"


def test_the_preview_of_the_guided_list_spreads_too():
    checked = Checked(profile=_cards(24.0, 24.0), machine_config=MACHINE, reason=None)
    assert model_fit(40.0, checked, 8192).mode == "gpu_split"


def test_the_size_scale_counts_a_spread_package_as_fitting():
    base = BaseModelSpec.model_validate(EXAMPLES["BaseModelSpec"])
    packages = [_package(30.0)]
    assert count_fitting(_cards(24.0, 24.0), MACHINE, packages, {base.hf_repo: base}, 8192) == 1
    assert count_fitting(_cards(24.0), MACHINE, packages, {base.hf_repo: base}, 8192) == 1  # in system memory


# --- document schema 3 and its words ---------------------------------------------------------------


def test_the_document_is_schema_3():
    assert DOCUMENT_SCHEMA_VERSION == 3
    assert EXAMPLES["RenderDocument"]["schema_version"] == 3


def test_a_spread_fit_round_trips_through_the_document_and_schema_2_is_refused():
    payload = copy.deepcopy(EXAMPLES["RenderDocument"])
    payload["machines"][0]["ranked"][0]["fit"]["mode"] = "gpu_split"
    document = RenderDocument.model_validate(payload)
    assert RenderDocument.model_validate(json.loads(json.dumps(document.model_dump(mode="json")))) == document
    with pytest.raises(ValidationError):
        RenderDocument.model_validate({**payload, "schema_version": 2})


def test_the_note_of_a_spread_row_names_the_cards_and_the_reserve_on_each_card():
    fit = _fit(_cards(24.0, 24.0), 30.0)
    note = _computed_note(fit, cards=2)
    assert note.code == "spread_over_graphics_cards"
    assert note.text == (
        "Fits into graphics memory spread over 2 cards: it needs about 34.5 GiB of the 46.0 GiB left after "
        "the reserve of 1.0 GiB on each card."
    )


def test_a_too_tight_row_names_no_card_count():
    fit = _fit(_cards(24.0, 24.0), 200.0)
    text = _set_aside_note(SetAside(package=_package(200.0), fit=fit, reason="too tight"), covered=True).text
    assert "card" not in text and "after the reserve" in text


def _spread_folder(tmp_path: Path):
    config = _rendered(tmp_path)
    profile = _cards(5.0, 5.0).model_copy(update={"profile_id": "5eed00000000000a"})
    atomic_write_json(config.paths.hardware_dir / f"{profile.profile_id}.json", profile.model_dump(mode="json"))
    return with_profile(config, "workstation", profile.profile_id)


def test_a_rendered_machine_with_two_small_cards_ranks_a_spread_row(tmp_path: Path):
    """Two cards of 5 GiB: a package over 4 GiB but under 8 GiB is spread over both."""
    assert render_with_config(_spread_folder(tmp_path), now=RUN2) == 0
    payload = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    rows = [entry for block in payload["machines"] for entry in block["ranked"] if entry["fit"]["mode"] == "gpu_split"]
    assert rows, "no row spread over the two cards"
    assert all("spread over 2 cards" in row["note"]["text"] and "1.0 GiB on each card" in row["note"]["text"] for row in rows)
    block = RenderDocument.model_validate(payload).machines[0]
    assert any("spread over the cards" in piece and "the reserve on each card" in piece for piece in _pool_pieces(block, "-"))
    markdown = (tmp_path / "models.md").read_text(encoding="utf-8")
    assert "| multi_gpu | 0 Nova GPU 5.00; 1 Nova GPU 5.00 | server (chassis) |" in markdown
