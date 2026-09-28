"""Tests for modelroom.parameters: one rule for a model's parameter count, for the list and the result."""

from __future__ import annotations

import pytest

from modelroom.guided_context import Checked
from modelroom.guided_contracts import SearchHit
from modelroom.guided_models import model_choices
from modelroom.parameters import parameter_count, parameters_from_name


def _hit(repo: str, base: str, parameters_b: float | None) -> SearchHit:
    return SearchHit(
        repo=repo,
        publisher_status="listed packager",
        base_model=[base],
        base_model_relation="quantized",
        resolved=True,
        resolved_base_model=base,
        parameters_b=parameters_b,
    )


@pytest.mark.parametrize(
    "stated, name, expected",
    [(9.65, "Qwen3.5-9B", 9.65), (None, "Qwen3.5-9B", 9.0), (None, "Nova-A3B", None), (None, "Nova-2.4T", 2400.0)],
)
def test_the_stated_number_wins_over_the_name(stated, name, expected):
    assert parameter_count(stated, name) == expected


def test_the_list_and_the_result_count_the_same_parameters():
    """The selection list of step 2 takes its size from this one function, and so does the result."""
    stated = _hit("packager/Nova-9B-GGUF", "nova-ai/Nova-9B", 9.65)
    named = _hit("packager/Vega-7B-GGUF", "vega-ai/Vega-7B", None)

    models = model_choices([stated, named], filtered=False, checked=Checked(), context=8192)

    by_base = {model.base_model: model.parameters_b for model in models}
    assert by_base == {
        "nova-ai/Nova-9B": parameter_count(9.65, "Nova-9B"),
        "vega-ai/Vega-7B": parameter_count(None, "Vega-7B"),
    }
    assert parameters_from_name("Vega-7B") == 7.0
