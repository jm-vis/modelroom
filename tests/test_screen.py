"""Tests for modelroom.screen: every line the guided mode draws, with color and without.

Each pattern is checked twice -- as the `(class, text)` fragments a terminal gets and as the plain
text a log gets -- and once more in ASCII, which is what a Windows console under `cp1252` can
encode. See CONTRACTS.md, "Guided mode", "The screen".
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest

from modelroom.intro import ASCII_GLYPHS, UNICODE_GLYPHS, Fact
from modelroom.screen import (
    HEAD_WIDTH,
    JOIN_WIDTH,
    NAME_WIDTH,
    Screen,
    answer_line,
    card_lines,
    clone_words,
    context_tokens_text,
    fit_word,
    head_line,
    imported_note,
    install_name,
    install_pull_line,
    joined,
    machine_fact,
    machines_summary,
    machines_words,
    measured_note,
    measured_when,
    models_fact,
    named_accounts,
    names_words,
    note_line,
    number_word,
    packages_summary,
    plain,
    ranks_text,
    repositories_note,
    result_fact,
    searched_note,
    short_hardware,
    speed_fact,
    summary_line,
    yes_no,
)

from fixture_support import v2_profile

NOW = datetime(2026, 9, 25, 8, 0, 0, tzinfo=timezone.utc)


def _stream(encoding: str, *, tty: bool = False):
    """A real text stream with an encoding of its own, as a console would have."""

    class _Buffer(io.BytesIO):
        def isatty(self) -> bool:
            return tty

    return io.TextIOWrapper(_Buffer(), encoding=encoding, errors="strict")


def _screen(encoding: str = "utf-8") -> tuple[Screen, list[str]]:
    lines: list[str] = []
    return Screen(lines.append, _stream(encoding), colored=False), lines


# --- the head of a step ----------------------------------------------------------------------------


def test_the_head_of_a_step_is_filled_to_seventy_two_characters():
    line = head_line(1, UNICODE_GLYPHS)

    assert plain(line) == " " + "── Step 1 of 5  Configuration ".ljust(HEAD_WIDTH, "─")
    assert len(plain(line)) == HEAD_WIDTH + 1
    assert [name for name, _text in line] == ["class:label"]


def test_the_head_of_a_step_is_ascii_where_the_rule_cannot_be_printed():
    line = head_line(5, ASCII_GLYPHS)

    assert plain(line).strip().startswith("-- Step 5 of 5  Results -")
    assert len(plain(line).strip()) == HEAD_WIDTH


# --- the answer line -------------------------------------------------------------------------------


def test_the_answer_line_is_the_question_and_the_answer_in_words():
    line = answer_line("Where should results live?", "this folder")

    assert plain(line) == " ? Where should results live?  this folder"
    assert [name for name, _text in line if name] == ["class:qmark", "class:question", "class:answer"]


def test_the_answer_of_a_path_is_cyan_and_never_green():
    line = answer_line("Path to the results folder", "C:/results", path=True)

    assert [name for name, _text in line if name] == ["class:qmark", "class:question", "class:path"]


# --- notes and balances ----------------------------------------------------------------------------


def test_a_note_is_indented_by_two_under_the_answer_it_belongs_to():
    assert plain(note_line("measured again: one graphics card")) == "   measured again: one graphics card"
    assert [name for name, _text in note_line("x") if name] == ["class:note"]


def test_a_finished_step_carries_a_check_mark_and_its_name_in_one_column():
    line = summary_line(1, "C:/results, 1 machine", UNICODE_GLYPHS)

    assert plain(line) == " ✓ Configuration   C:/results, 1 machine"
    assert [name for name, _text in line if name] == ["class:done", "class:value"]


def test_a_step_that_did_nothing_carries_a_dash_in_amber():
    line = summary_line(4, "none in this run", UNICODE_GLYPHS, done=False)

    assert plain(line) == " – Measurement     none in this run"
    assert [name for name, _text in line if name] == ["class:tight", "class:value"]


def test_the_name_of_every_step_ends_in_the_same_column():
    lines = [plain(summary_line(number, "x", UNICODE_GLYPHS)) for number in range(1, 6)]

    assert {line.rindex("x") for line in lines} == {3 + NAME_WIDTH + 2}


def test_a_balance_says_ok_and_a_hyphen_where_the_glyphs_cannot_be_printed():
    assert plain(summary_line(1, "x", ASCII_GLYPHS)).startswith(" ok Configuration")
    assert plain(summary_line(4, "x", ASCII_GLYPHS, done=False)).startswith(" - Measurement")


# --- the card --------------------------------------------------------------------------------------


def test_the_card_draws_the_rows_of_the_start_screen():
    facts = [Fact("folder", "C:/results", ""), Fact("context", "L 32k", "24,000 words")]

    lines = card_lines(facts)

    assert plain(lines[0]) == " folder    C:/results"
    assert plain(lines[1]) == " context   L 32k                   24,000 words"
    assert [name for name, _text in lines[0] if name] == ["class:label", "class:path"]


def test_a_value_wider_than_its_column_pushes_the_note_along():
    long_path = "C:/" + "folder/" * 20 + "results"

    line = plain(card_lines([Fact("result", long_path, "25 packages ranked")])[0])

    assert f"{long_path}  25 packages ranked" in line


# --- joined --------------------------------------------------------------------------------------


def test_joined_puts_the_pieces_of_one_statement_on_one_line():
    assert joined(["showing 5 of 25", "1 package too tight"]) == [" showing 5 of 25 · 1 package too tight"]


def test_joined_breaks_between_pieces_and_never_inside_one():
    pieces = ["a" * 60, "b" * 60, "c" * 20]

    lines = joined(pieces, width=JOIN_WIDTH)

    assert lines == [" " + "a" * 60, " " + "b" * 60 + " · " + "c" * 20]
    assert all(len(line) <= JOIN_WIDTH or " · " not in line for line in lines)


def test_a_piece_longer_than_the_width_stands_on_a_line_of_its_own_and_is_not_cut():
    long_piece = "x" * 120

    assert joined(["short", long_piece]) == [" short", " " + long_piece]


def test_joined_keeps_the_same_indent_on_every_line():
    lines = joined(["a" * 80, "b" * 80], indent=3)

    assert all(line.startswith("   ") for line in lines)


def test_joined_leaves_out_empty_pieces():
    assert joined(["a", "", "b"]) == [" a · b"]
    assert joined([]) == []


# --- the words of an answer ------------------------------------------------------------------------


def test_the_machine_list_is_read_back_in_the_words_of_its_entries():
    assert machines_words(["this-machine"]) == "this machine"
    assert machines_words(["this-machine", "import"]) == "this machine, a profile file"
    assert machines_words([]) == "none"


@pytest.mark.parametrize("answer, words", [("same", "the same machine"), ("clone", "a clone")])
def test_the_clone_answer_is_read_back_in_its_own_words(answer, words):
    assert clone_words(answer) == words


def test_a_switch_is_read_back_as_the_list_showed_it():
    assert (yes_no(True), yes_no(False)) == ("Yes", "No")


def test_a_list_of_names_is_one_answer_and_nothing_is_a_word_of_its_own():
    assert names_words(["Qwen3.5-9B", "Qwen3-0.6B"]) == "Qwen3.5-9B, Qwen3-0.6B"
    assert names_words([]) == "nothing"
    assert names_words([], "none") == "none"


def test_a_context_is_named_in_thousands_where_it_is_whole_thousands():
    assert context_tokens_text(32768) == "32k"
    assert context_tokens_text(8192) == "8k"
    assert context_tokens_text(5000) == "5000 tokens"


# --- the notes of the steps ------------------------------------------------------------------------


def test_a_measurement_note_names_the_hardware_and_who_confirmed_it():
    profile = v2_profile()

    assert measured_note(profile, again=False) == "measured: one graphics card, 12 GB, 127 GB memory"
    assert measured_note(profile, again=True).startswith("measured again: ")


def test_a_cross_check_that_confirmed_both_readings_is_named():
    from modelroom.profile import CrossCheck, LlmfitCrosscheck

    confirmed = CrossCheck(status="confirmed", own_gib=127.46, llmfit_gib=127.9)
    profile = v2_profile().model_copy(
        update={"llmfit_crosscheck": LlmfitCrosscheck(ram_physical=confirmed, vram=confirmed)}
    )

    assert measured_note(profile, again=False).endswith(", confirmed by llmfit")


def test_an_import_names_the_machine_that_came_in():
    assert imported_note("server", v2_profile()).startswith("imported server: one graphics card")


@pytest.mark.parametrize("count, words", [(1, "1 machine"), (2, "2 machines")])
def test_the_balance_of_step_one_counts_machines(count, words):
    assert machines_summary(count) == words


def test_the_search_note_names_the_publishers_that_answered_and_the_packagers_as_a_word():
    assert searched_note(["Qwen"], 5, open_lists=False) == (
        "searched Hugging Face at the publisher Qwen and the five listed packagers"
    )
    assert searched_note(["Qwen", "deepseek-ai"], 5, open_lists=False).startswith(
        "searched Hugging Face at the publishers Qwen and deepseek-ai"
    )
    assert searched_note([], 5, open_lists=False).startswith("searched Hugging Face at no publisher of this catalog")
    assert searched_note(["Qwen"], 5, open_lists=True).endswith("listed packagers, and the two open lists")


@pytest.mark.parametrize("count, word", [(0, "no"), (1, "one"), (5, "five"), (10, "ten"), (11, "11")])
def test_a_count_of_ten_or_less_is_said_in_words(count, word):
    assert number_word(count) == word


def test_the_second_note_says_how_much_of_the_answer_is_a_choice():
    assert repositories_note(120, 40, page_full=False) == "120 repositories, 40 of them models you can pick from"
    assert repositories_note(1, 1, page_full=False) == "1 repository, 1 of them a model you can pick from"
    assert repositories_note(59, 0, page_full=False) == "59 repositories, none of them a model you can pick from"


def test_a_full_page_invites_a_more_specific_word():
    assert repositories_note(120, 40, page_full=True).endswith("· a more specific word shortens the list")


def test_the_balance_of_step_two_names_models_packages_and_repositories():
    assert packages_summary(2, 26, 4) == "2 models, 26 packages from 4 repositories"
    assert packages_summary(1, 1, 1) == "1 model, 1 package from 1 repository"


# --- the card of step 5 ----------------------------------------------------------------------------


def test_the_hardware_of_the_card_is_the_short_form():
    assert short_hardware(v2_profile()) == "12 GB graphics, 127 GB memory"
    assert short_hardware(v2_profile(vram_gib=0.0)) == "no graphics card, 127 GB memory"


def test_a_profile_measured_today_says_today_and_any_other_says_its_date():
    profile = v2_profile(recorded_at=NOW)

    assert measured_when(profile, NOW) == "measured today"
    assert measured_when(profile, NOW.replace(day=26)) == "measured 2026-09-25"


def test_a_machine_without_a_profile_carries_its_reason_instead():
    fact = machine_fact("workstation", None, NOW, "no hardware profile yet")

    assert (fact.label, fact.value, fact.note) == ("machine", "workstation", "no hardware profile yet")


def test_up_to_three_accounts_are_named_and_the_rest_is_a_count():
    assert named_accounts(["Qwen"]) == "Qwen"
    assert named_accounts(["Qwen", "unsloth", "Ollama"]) == "Qwen, unsloth and Ollama"
    assert named_accounts(["a", "b", "c", "d"]) == "a, b, c and 1 more"


def test_the_models_row_names_the_models_and_where_their_packages_came_from():
    fact = models_fact(["Qwen3.5-9B"], 26, ["Qwen", "unsloth"])

    assert (fact.value, fact.note) == ("Qwen3.5-9B", "26 packages from Qwen and unsloth")


def test_the_speed_row_says_what_to_do_where_nothing_was_measured():
    nothing = speed_fact(0, None)
    measured = speed_fact(2, ("Qwen3-0.6B", 41.06))

    assert (nothing.value, nothing.note) == ("not measured", "say Yes in step 4 to measure an installed package")
    assert (measured.value, measured.note) == ("2 measured", "fastest Qwen3-0.6B at 41.1 tok/s")


def test_the_result_row_counts_what_was_ranked_and_what_was_set_aside():
    assert result_fact("docs/models.md", 25, [("too tight", 1)]).note == "25 packages ranked, 1 too tight"
    assert result_fact("docs/models.md", 25, []).note == "25 packages ranked, none too tight"
    assert result_fact("docs/models.md", 25, [("too tight", 1), ("not covered", 3)]).note == (
        "25 packages ranked, 1 too tight · 3 not covered"
    )


# --- the fit word, the rank ranges and the install line ----------------------------------------------


def test_a_fit_against_system_memory_says_ram_behind_its_class():
    assert fit_word("good", "gpu") == "good"
    assert fit_word("good", "cpu_gpu") == "good (RAM)"
    assert fit_word("marginal", "cpu") == "marginal (RAM)"


@pytest.mark.parametrize(
    "ranks, text",
    [([1], "#1"), ([1, 2], "#1–2"), ([1, 4, 5], "#1, #4–5"), ([3], "#3"), ([2, 3, 4, 7], "#2–4, #7")],
)
def test_a_set_of_ranks_reads_as_ranges(ranks, text):
    assert ranks_text(ranks) == text


def test_a_rank_range_is_ascii_where_the_dash_cannot_be_printed():
    assert ranks_text([1, 2], ASCII_GLYPHS.skip) == "#1-2"


def test_the_local_name_of_a_hugging_face_package_is_the_hf_co_name():
    identity = ("huggingface", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf")

    assert install_name(identity, "Q4_K_M", "gguf") == "hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M"


def test_the_local_name_of_an_ollama_package_is_its_own_name():
    assert install_name(("ollama", "qwen3.5:9b-q4_K_M", "9b-q4_K_M"), "Q4_K_M", "gguf") == "qwen3.5:9b-q4_K_M"


def test_a_package_with_no_local_name_gets_no_line():
    identity = ("huggingface", "unsloth/Qwen3.5-9B", "model.safetensors")

    assert install_name(identity, "Q4_K_M", "tensor") is None
    assert install_name(identity, "unknown", "gguf") is None


def test_the_install_line_is_a_line_to_copy():
    """The rank is a side remark, the command is the thing: `to install #1: ollama pull …` read as
    one sentence and the command went under in it (test round, 2026-09-25)."""
    line = install_pull_line(1, "hf.co/a/b:Q8_0")

    assert plain(line) == " #1  ollama pull hf.co/a/b:Q8_0"
    assert [name for name, _text in line] == ["class:note", "", "class:command"]


# --- the screen itself -----------------------------------------------------------------------------


def test_without_color_every_line_goes_through_out_as_plain_text():
    screen, lines = _screen()

    screen.head(1)
    screen.answer("Where should results live?", "this folder")
    screen.note("measured: one graphics card")
    screen.blank()
    screen.done(1, "C:/results, 1 machine")
    screen.skipped(4, "none in this run")
    screen.card([Fact("folder", "C:/results", "")])

    assert lines[0].strip().startswith("── Step 1 of 5  Configuration")
    assert lines[1] == " ? Where should results live?  this folder"
    assert lines[2] == "   measured: one graphics card"
    assert lines[3] == ""
    assert lines[4] == " ✓ Configuration   C:/results, 1 machine"
    assert lines[5] == " – Measurement     none in this run"
    assert lines[6] == " folder    C:/results"


def test_a_console_that_cannot_encode_the_glyphs_gets_ascii():
    screen, lines = _screen("cp1252")

    screen.head(1)
    screen.done(1, "x")
    screen.skipped(4, "y")

    assert all(line.isascii() for line in lines)
    assert screen.glyphs is ASCII_GLYPHS


def test_with_color_the_lines_go_through_the_library_and_not_through_out(capsys):
    """Either the line is on screen in color, or the same line is printed without it."""
    lines: list[str] = []
    screen = Screen(lines.append, _stream("utf-8", tty=True), colored=True)

    screen.head(1)
    screen.done(1, "C:/results, 1 machine")

    printed = capsys.readouterr().out + "\n".join(lines)
    assert "Step 1 of 5  Configuration" in printed
    assert "Configuration   C:/results, 1 machine" in printed
