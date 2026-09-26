"""The takeover rule and a machine entered by hand: it is never the profile a measurement adopts.

A profile entered by hand (`origin == "entered"` and `ram_physical_source == "entered"`) has no
OS fingerprint, and `none` agrees with every fingerprint. Without its own rule a measurement of
any machine would adopt it. So the rule asks first, before any fingerprint, also for
`--same-machine`, and measures this machine as its own profile. A `hardware --cpu-only` profile
(`origin == "entered"`, RAM measured by the OS) keeps its identity (decided 2026-09-25).
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

import modelroom.guided as guided_module
import modelroom.guided_loadtest as loadtest_module
from modelroom.binding import KnownProfile, resolve_profile_target
from modelroom.examples import EXAMPLES
from modelroom.profile import HardwareProfile
from modelroom.state import atomic_write_json

from test_guided import FULL_ANSWERS
from test_guided import _run as guided_run
from test_hardware_cmd import ID_ONE, ID_TWO, _config, _profiles
from test_hardware_cmd import _run as hardware_run

HAND = "1111111111111111"
FRESH = "ffffffffffffffff"
HERE = "9d2f4b6a8c0e1357"
HAND_REASON = "configured profile was entered by hand; this machine is measured as its own profile"


def _known(origin: str, ram_source: str, fingerprint: str = "none") -> dict[str, KnownProfile]:
    return {HAND: KnownProfile(HAND, fingerprint, origin, ram_source)}


def test_known_profile_carries_origin_and_the_ram_source():
    fields = [field.name for field in dataclasses.fields(KnownProfile)]
    assert fields == ["profile_id", "os_fingerprint", "origin", "ram_physical_source"]
    with pytest.raises(TypeError):
        KnownProfile(HAND, "none")  # type: ignore[call-arg]


@pytest.mark.parametrize("same_machine", [False, True])
def test_a_configured_hand_profile_gives_a_new_profile_before_the_fingerprint_is_asked(same_machine):
    target = resolve_profile_target(None, HAND, _known("entered", "entered"), HERE, FRESH, same_machine=same_machine)
    assert (target.action, target.profile_id, target.reason) == ("new", FRESH, HAND_REASON)


@pytest.mark.parametrize("same_machine", [False, True])
def test_a_bound_hand_profile_gives_a_new_profile_as_well(same_machine):
    target = resolve_profile_target(HAND, None, _known("entered", "entered"), HERE, FRESH, same_machine=same_machine)
    assert (target.action, target.profile_id) == ("new", FRESH)


def test_a_cpu_only_profile_keeps_its_identity():
    target = resolve_profile_target(None, HAND, _known("entered", "os"), HERE, FRESH)
    assert (target.action, target.profile_id) == ("adopt_config", HAND)


def test_a_measured_profile_is_adopted_as_before():
    target = resolve_profile_target(None, HAND, _known("measured", "os", HERE), HERE, FRESH)
    assert (target.action, target.profile_id) == ("adopt_config", HAND)


# --- the three places that build `KnownProfile` -----------------------------------------------------


def _hand_profile(profile_id: str) -> dict:
    data = copy.deepcopy(EXAMPLES["HardwareProfile"])
    data.update(
        schema_version=3,
        profile_id=profile_id,
        display_name="office box",
        os_fingerprint="none",
        os_fingerprint_source="none",
        origin="entered",
        ram_physical_gib=32.0,
        ram_physical_source="entered",
        vram_gib=0.0,
        vram_source="none",
        gpu_state="unified_memory",
        gpu_name=None,
        llmfit_crosscheck={"ram_physical": {"status": "absent"}, "vram": {"status": "absent"}},
        llmfit_version=None,
    )
    return HardwareProfile.model_validate(data).model_dump(mode="json")


@pytest.mark.parametrize("same_machine", [False, True])
def test_hardware_measures_a_new_profile_next_to_a_configured_hand_profile(tmp_path, same_machine):
    machines = {"workstation": {"reserve_ram_gib": 8.0, "reserve_vram_gib": 1.0, "writer": True, "profile": ID_TWO}}
    config = _config(tmp_path, machines=machines)
    hand_file = config.paths.hardware_dir / f"{ID_TWO}.json"
    atomic_write_json(hand_file, _hand_profile(ID_TWO))
    before = hand_file.read_bytes()

    assert hardware_run(config, tmp_path, machine="workstation", same_machine=same_machine) == 0

    assert [path.name for path in _profiles(config)] == [f"{ID_ONE}.json", f"{ID_TWO}.json"]
    assert hand_file.read_bytes() == before


def test_the_guided_mode_builds_known_profiles_with_both_marks(tmp_path, monkeypatch):
    """`guided._clone_mode` and `guided_loadtest._bound_profile_id` hand the rule both marks.

    A hand-entered profile lies in the folder before the run, so each of the two sites has to
    read `entered`/`entered` from the file -- a site that hard-coded `measured`/`os` would fail
    here (review round, 2026-09-26). Both sites are recorded on their own.
    """
    seen: dict[str, list[KnownProfile]] = {"guided": [], "loadtest": []}
    real = resolve_profile_target

    def spy_for(site: str):
        def spy(home_binding, config_profile, profiles, *args, **kwargs):
            seen[site].extend(profiles.values())
            return real(home_binding, config_profile, profiles, *args, **kwargs)

        return spy

    monkeypatch.setattr(guided_module, "resolve_profile_target", spy_for("guided"))
    monkeypatch.setattr(loadtest_module, "resolve_profile_target", spy_for("loadtest"))
    (tmp_path / "results").mkdir()
    (tmp_path / "home").mkdir()
    hardware_dir = tmp_path / "results" / "state" / "hardware"
    hardware_dir.mkdir(parents=True)
    atomic_write_json(hardware_dir / f"{HAND}.json", _hand_profile(HAND))

    # The state folder is there already, so the run asks whether to write a configuration into it.
    assert guided_run(tmp_path, {**FULL_ANSWERS, "write_config": True})[0] == 0

    marks = {
        site: {(profile.profile_id, profile.origin, profile.ram_physical_source) for profile in known}
        for site, known in seen.items()
    }
    for site in ("guided", "loadtest"):
        assert (HAND, "entered", "entered") in marks[site], f"{site} did not hand the rule the entered profile"
    # The machine step asks before it measures, so only the load test also sees this machine's own profile.
    assert any(mark[1:] == ("measured", "os") for mark in marks["loadtest"] if mark[0] != HAND)
