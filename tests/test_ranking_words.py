"""Tests for modelroom.ranking_words: what a result says about the rows it shows."""

from __future__ import annotations

from types import SimpleNamespace

from modelroom.ranking import BEST_FACT, SHOWN_FACT
from modelroom.ranking_words import PER_MODEL_PIECES, key_pieces, showing_line, shown_pieces
from modelroom.screen import FIT_HINT


def _block(facts: list[list[str]], total: int = 24) -> SimpleNamespace:
    rows = [SimpleNamespace(rank=number, note=SimpleNamespace(facts=row)) for number, row in enumerate(facts, start=1)]
    return SimpleNamespace(ranked=rows, ranked_total=total)


def test_one_appended_row_is_singular_and_several_are_plural():
    one = _block([["fit.mode", BEST_FACT], ["fit.mode", BEST_FACT], ["measurement.tps_mean", SHOWN_FACT]])
    several = _block([["fit.mode"]] * 10 + [["measurement.tps_mean", SHOWN_FACT]] * 2)

    assert "#3 is shown for its measurement, not by the rule" in shown_pieces(one, "–")
    assert "#11–12 are shown for their measurements, not by the rule" in shown_pieces(several, "–")
    assert showing_line(one) == "Showing 3 rows of 24 ranked packages: one row per model, then 1 measured."
    assert showing_line(several) == "Showing 12 of 24 ranked packages: the first 10 by the rule, then 2 measured."


def test_one_row_per_model_is_said_in_the_key_and_one_model_says_nothing_more():
    per_model = _block([["fit.mode", BEST_FACT], ["fit.mode", BEST_FACT]])
    one_model = _block([["fit.mode"]] * 3)

    assert key_pieces(per_model) == [*FIT_HINT, *PER_MODEL_PIECES]
    assert key_pieces(one_model) == list(FIT_HINT)
    assert shown_pieces(per_model, "–") == ["2 rows of 24 packages, one row per model"]
    assert shown_pieces(one_model, "–") == ["3 of 24 packages"]
    assert showing_line(one_model) == "Showing 3 of 24 ranked packages."


def test_the_key_says_what_best_means_and_rates_no_family():
    assert "best: the least compression of the same model that fits" in FIT_HINT
    assert "modelroom rates no model family against another" in FIT_HINT
