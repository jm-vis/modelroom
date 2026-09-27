"""Step 1 with several graphics cards and the machine class (decided 2026-09-26).

A machine entered by hand names its class (`entered_class`, always asked) and may have several
graphics cards of one size (`cards`, then `entered_cards`, then the size per card). This machine,
measured, is asked for its class only when its chassis says nothing (`machine_class`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from modelroom.binding import KnownProfile
from modelroom.guided import GuidedError, _hardware_class
from modelroom.guided_entered import entered_class, entered_profile
from modelroom.intro import hardware_words
from modelroom.measure import WINDOWS_CHASSIS_ARGS
from modelroom.profile import HardwareProfile
from modelroom.screen import entered_hardware, entered_note, short_hardware

from fixture_support import v2_profile, windows_probes, windows_runner
from test_guided import (
    ENTERED_CARD,
    FULL_ANSWERS,
    RUN1,
    RUN2,
    _answers_without_folder,
    _entered_files,
    _profiles,
    _run,
    _watched,
)

ENTERED_CARDS = {**ENTERED_CARD, "entered_name": "rack", "entered_class": "server", "entered_ram": 128,
                 "entered_gpu": "cards", "entered_cards": 2, "entered_vram": 24}


def _silent_chassis():
    """A Windows machine whose chassis answers nothing a class can be read from."""
    empty = subprocess.CompletedProcess(list(WINDOWS_CHASSIS_ARGS), 0, stdout="", stderr="")
    return windows_probes(runner=windows_runner({WINDOWS_CHASSIS_ARGS: empty}))


def _measured(tmp_path: Path) -> HardwareProfile:
    profiles = [HardwareProfile.model_validate_json(path.read_text(encoding="utf-8")) for path in _profiles(tmp_path)]
    (profile,) = [profile for profile in profiles if profile.ram_physical_source != "entered"]
    return profile


# --- several graphics cards entered by hand -----------------------------------------------------------


def test_several_cards_entered_by_hand_are_multi_gpu_with_their_sum_and_no_names(tmp_path: Path):
    code, lines = _run(tmp_path, ENTERED_CARDS)

    assert code == 0
    (profile,) = _entered_files(tmp_path)
    assert (profile.gpu_state, profile.vram_gib, profile.vram_source) == ("multi_gpu", 48.0, "entered")
    assert [(gpu.index, gpu.name, gpu.vram_gib) for gpu in profile.gpus] == [(0, None, 24.0), (1, None, 24.0)]
    assert (profile.machine_class, profile.machine_class_source) == ("server", "entered")
    answers = [line.strip() for line in lines if line.strip().startswith("? ")]
    assert "? What kind of machine is it?  server" in answers
    assert "? What runs the model?  several graphics cards of one size" in answers
    assert "? How many graphics cards?  2" in answers
    assert "? How much graphics memory per card, in GiB?  24 GiB" in answers


def test_the_card_count_is_asked_only_after_several_cards(tmp_path: Path):
    asker = _watched(tmp_path, ENTERED_CARDS)
    assert [key for key in asker.asked if key.startswith("entered_")] == [
        "entered_name", "entered_class", "entered_ram", "entered_gpu", "entered_cards", "entered_vram",
    ]


@pytest.mark.parametrize(
    "count", [1, 0, "many", "-2", "2.5", "٣", "1" + "0" * 400, 1025],
    ids=["one", "zero", "text", "negative", "decimal", "other-digit", "401-digits", "above-1024"],
)
def test_a_card_count_below_two_or_no_whole_number_ends_the_run_and_writes_no_profile(tmp_path: Path, count):
    with pytest.raises(GuidedError, match="entered_cards"):
        _run(tmp_path, {**ENTERED_CARDS, "entered_cards": count})
    assert _profiles(tmp_path) == []


def test_a_size_per_card_whose_sum_is_no_number_ends_the_run_and_writes_no_profile(tmp_path: Path):
    with pytest.raises(GuidedError, match="entered_vram"):
        _run(tmp_path, {**ENTERED_CARDS, "entered_cards": 1024, "entered_vram": "1e306"})
    assert _profiles(tmp_path) == []


def test_not_sure_leaves_the_class_unknown(tmp_path: Path):
    code, _lines = _run(tmp_path, {**ENTERED_CARDS, "entered_class": "unknown"})
    assert code == 0
    (profile,) = _entered_files(tmp_path)
    assert (profile.machine_class, profile.machine_class_source) == ("unknown", "unknown")


def test_several_cards_entered_by_hand_are_still_recognized_as_entered_by_hand(tmp_path: Path):
    """`binding.entered_by_hand` reads origin and memory source; neither changes for cards."""
    _run(tmp_path, ENTERED_CARDS)
    (profile,) = _entered_files(tmp_path)
    known = KnownProfile(profile.profile_id, profile.os_fingerprint, profile.origin, profile.ram_physical_source)
    assert known.entered_by_hand


# --- the words of a machine with a class and cards -------------------------------------------------------


def _hand(shape: str, machine_class: str = "server") -> HardwareProfile:
    return entered_profile("box", 128.0, shape, 24.0, "0123456789abcdef", RUN1, cards=2, machine_class=machine_class)


def test_the_words_of_several_cards_entered_by_hand_name_the_class_first():
    profile = _hand("cards")
    assert entered_note(profile) == "entered box: server, 128 GiB memory, 2 graphics cards 24 GiB each"
    assert entered_hardware(profile) == "server, 2 graphics cards 24 GB each, 128 GB memory, entered"
    assert entered_class(profile) == "server, 2 graphics cards 24.00 GiB each (entered) / 128.00 GiB RAM"


def test_a_machine_of_unknown_class_says_no_class_word():
    profile = _hand("card", machine_class="unknown")
    assert entered_note(profile) == "entered box: 128 GiB memory, graphics card 24 GiB"
    assert entered_hardware(profile) == "graphics card 24 GB, 128 GB memory, entered"
    assert entered_class(profile) == "graphics card 24.00 GiB (entered) / 128.00 GiB RAM"


def _measured_cards(*cards: tuple[str | None, float], machine_class: str = "server") -> HardwareProfile:
    payload = v2_profile().model_dump(mode="json")
    total = round(sum(size for _name, size in cards), 2)
    names = {name for name, _size in cards}
    payload.update(
        vram_gib=total,
        gpu_state="multi_gpu" if len(cards) > 1 else "measured",
        gpu_name=next(iter(names)) if len(names) == 1 else None,
        gpus=[{"index": i, "name": name, "vram_gib": size, "vram_source": "nvidia-smi"} for i, (name, size) in enumerate(cards)],
        machine_class=machine_class,
        machine_class_source="unknown" if machine_class == "unknown" else "chassis",
        ram_physical_gib=512.0,
    )
    return HardwareProfile.model_validate(payload)


A100 = "NVIDIA A100-SXM4-40GB"


@pytest.mark.parametrize(
    ("cards", "machine_class", "words", "short", "group"),
    [
        (((A100, 40.0),), "laptop", "laptop, one graphics card, 40 GB, 512 GB memory", "laptop, 40 GB graphics, 512 GB memory",
         f"laptop, {A100} 40.00 GiB VRAM / 512.00 GiB RAM"),
        (((A100, 40.0), (A100, 40.0)), "server", "server, 2 graphics cards, 40 + 40 GB, 512 GB memory",
         "server, 2 x 40 GB graphics, 512 GB memory", f"server, 2 x {A100} 80.00 GiB VRAM / 512.00 GiB RAM"),
        ((("A6000", 48.0), ("A4000", 16.0)), "workstation", "workstation, 2 graphics cards, 48 + 16 GB, 512 GB memory",
         "workstation, 48 + 16 GB graphics, 512 GB memory", "workstation, A6000, A4000 64.00 GiB VRAM / 512.00 GiB RAM"),
        (((A100, 80.0),) * 5, "server", "server, 5 graphics cards, 80 GB each, 512 GB memory",
         "server, 5 x 80 GB graphics, 512 GB memory", f"server, 5 x {A100} 400.00 GiB VRAM / 512.00 GiB RAM"),
        ((("A", 80.0),) * 4 + (("B", 16.0),), "server", "server, 5 graphics cards, 336 GB in all, 512 GB memory",
         "server, 80 + 80 + 80 + 80 + 16 GB graphics, 512 GB memory", "server, A, A, A, A, B 336.00 GiB VRAM / 512.00 GiB RAM"),
        (((None, 24.0), ("", 24.0)), "unknown", "2 graphics cards, 24 + 24 GB, 512 GB memory",
         "2 x 24 GB graphics, 512 GB memory", "2 x graphics card 48.00 GiB VRAM / 512.00 GiB RAM"),
    ],
    ids=["one", "two", "mixed", "five", "five-mixed", "no-names-no-class"],
)
def test_every_view_names_the_class_first_and_counts_the_cards(cards, machine_class, words, short, group):
    profile = _measured_cards(*cards, machine_class=machine_class)
    assert hardware_words(profile) == words
    assert short_hardware(profile) == short
    assert _hardware_class(profile) == group


# --- this machine: the class question only when the chassis says nothing -----------------------------------


def test_a_chassis_that_names_the_class_asks_nothing_and_needs_no_answer(tmp_path: Path):
    asker = _watched(tmp_path, FULL_ANSWERS)
    assert "machine_class" not in asker.asked
    assert (_measured(tmp_path).machine_class, _measured(tmp_path).machine_class_source) == ("laptop", "chassis")


def test_a_silent_chassis_asks_and_the_answer_is_entered(tmp_path: Path):
    code, lines = _run(tmp_path, {**FULL_ANSWERS, "machine_class": "server"}, probes=_silent_chassis())
    assert code == 0
    assert (_measured(tmp_path).machine_class, _measured(tmp_path).machine_class_source) == ("server", "entered")
    assert "? What kind of machine is this?  server" in [line.strip() for line in lines]


def test_not_sure_about_this_machine_leaves_the_class_unknown_with_the_note(tmp_path: Path, capsys):
    """The notes of a measurement are printed in the guided run too (`summary=False` keeps them)."""
    code, _lines = _run(tmp_path, {**FULL_ANSWERS, "machine_class": "unknown"}, probes=_silent_chassis())
    assert code == 0
    assert (_measured(tmp_path).machine_class, _measured(tmp_path).machine_class_source) == ("unknown", "unknown")
    assert "note: the machine class is unknown: the chassis type is not a number" in capsys.readouterr().out


def test_a_silent_chassis_without_an_answer_ends_the_run_naming_the_key(tmp_path: Path):
    with pytest.raises(Exception, match="machine_class"):
        _run(tmp_path, FULL_ANSWERS, probes=_silent_chassis())


def test_measuring_again_reads_the_class_again(tmp_path: Path):
    """An earlier answer is never taken over: the second run asks again and keeps its own answer."""
    assert _run(tmp_path, {**FULL_ANSWERS, "machine_class": "server"}, probes=_silent_chassis())[0] == 0
    answers = {**_answers_without_folder(FULL_ANSWERS), "machine_class": "workstation"}
    code, _lines = _run(tmp_path, answers, probes=_silent_chassis(), now=RUN2)
    assert code == 0
    assert _measured(tmp_path).machine_class == "workstation"
