"""Tests for `search_release`: a release computed from the version numbers of one family.

The key table covers every `hf_repo` of the shipped catalog plus the names of the recorded answers
and the cases the grammar was written for; a name the table does not name is a name the grammar
was not checked against (CONTRACTS.md, "Latest and legacy evidence", "Computed release").
"""

from __future__ import annotations

import itertools
from datetime import date

import pytest
from pydantic import ValidationError

from modelroom.catalog import Catalog, load_catalog
from modelroom.guided_contracts import SearchHit
from modelroom.search_age import AgeVerdict
from modelroom.search_release import LineDate, Release, compute_release, line_key, with_release

MISTRAL_SMALL_3 = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
MISTRAL_SMALL_4 = "mistralai/Mistral-Small-4-119B-2603"


def _yymm(value: str) -> LineDate:
    return LineDate("yymm", int(value[:2]), int(value[2:]), None)


def _mmdd(value: str) -> LineDate:
    return LineDate("mmdd", None, int(value[:2]), int(value[2:]))


# (hf_repo, family, variant, size, version, date) -- the family without its publisher.
KEYS = [
    # --- the shipped catalog, all 62 rows ---
    ("mistralai/Ministral-3-14B-Instruct-2512", "ministral", set(), 14, (3,), _yymm("2512")),
    ("mistralai/Ministral-3-8B-Instruct-2512", "ministral", set(), 8, (3,), _yymm("2512")),
    ("mistralai/Ministral-3-3B-Instruct-2512", "ministral", set(), 3, (3,), _yymm("2512")),
    (MISTRAL_SMALL_4, "mistral-small", set(), 119, (4,), _yymm("2603")),
    ("mistralai/Devstral-Small-2-24B-Instruct-2512", "devstral-small", set(), 24, (2,), _yymm("2512")),
    ("mistralai/Magistral-Small-2509", "magistral-small", set(), None, (), _yymm("2509")),
    (MISTRAL_SMALL_3, "mistral-small", set(), 24, (3, 2), _yymm("2506")),
    ("utter-project/EuroLLM-9B-Instruct", "eurollm", set(), 9, (), None),
    ("utter-project/EuroLLM-9B-Instruct-2512", "eurollm", set(), 9, (), _yymm("2512")),
    ("utter-project/EuroLLM-22B-Instruct-2512", "eurollm", set(), 22, (), _yymm("2512")),
    ("utter-project/EuroMoE-2.6B-A0.6B-Instruct-2512", "euromoe", set(), 2.6, (), _yymm("2512")),
    ("openGPT-X/Teuken-7B-instruct-v0.6", "teuken", set(), 7, (0, 6), None),
    ("Qwen/Qwen3.5-9B", "qwen", set(), 9, (3, 5), None),
    ("Qwen/Qwen3.5-4B", "qwen", set(), 4, (3, 5), None),
    ("Qwen/Qwen3.5-27B", "qwen", set(), 27, (3, 5), None),
    ("Qwen/Qwen3.5-35B-A3B", "qwen", set(), 35, (3, 5), None),
    ("Qwen/Qwen3.5-122B-A10B", "qwen", set(), 122, (3, 5), None),
    ("Qwen/Qwen3.6-27B", "qwen", set(), 27, (3, 6), None),
    ("Qwen/Qwen3.6-35B-A3B", "qwen", set(), 35, (3, 6), None),
    ("Qwen/Qwen3.8-27B", "qwen", set(), 27, (3, 8), None),
    ("Qwen/Qwen3-Coder-Next", "qwen", {"coder", "next"}, None, (3,), None),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen", {"coder"}, 30, (3,), None),
    ("Qwen/Qwen3-Embedding-8B", "qwen", {"embedding"}, 8, (3,), None),
    ("Qwen/Qwen3-Embedding-0.6B", "qwen", {"embedding"}, 0.6, (3,), None),
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "deepseek-r1", {"qwen3"}, 8, (), _mmdd("0528")),
    ("deepseek-ai/DeepSeek-V3.2", "deepseek", set(), None, (3, 2), None),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek", {"flash"}, None, (4,), _mmdd("0731")),
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama", {"scout"}, 17, (4,), None),
    ("meta-llama/Llama-3.3-70B-Instruct", "llama", set(), 70, (3, 3), None),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama", set(), 8, (3, 1), None),
    ("meta-llama/Llama-3.2-3B-Instruct", "llama", set(), 3, (3, 2), None),
    ("google/gemma-4-31B-it", "gemma", set(), 31, (4,), None),
    ("google/gemma-4-26B-A4B-it", "gemma", set(), 26, (4,), None),
    ("google/gemma-4-12B-it", "gemma", set(), 12, (4,), None),
    ("google/gemma-4-E4B-it", "gemma", set(), 4, (4,), None),
    ("microsoft/phi-4", "phi", set(), None, (4,), None),
    ("microsoft/Phi-4-mini-instruct", "phi", {"mini"}, None, (4,), None),
    ("openai/gpt-oss-120b", "gpt-oss", set(), 120, (), None),
    ("openai/gpt-oss-20b", "gpt-oss", set(), 20, (), None),
    ("zai-org/GLM-5.3-Flash", "glm", {"flash"}, None, (5, 3), None),
    ("zai-org/GLM-4.7-Flash", "glm", {"flash"}, None, (4, 7), None),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "kimi-linear", set(), 48, (), None),
    ("ibm-granite/granite-4.2-30b", "granite", set(), 30, (4, 2), None),
    ("ibm-granite/granite-4.2-8b", "granite", set(), 8, (4, 2), None),
    ("ibm-granite/granite-4.2-3b", "granite", set(), 3, (4, 2), None),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nvidia-nemotron", {"lightning"}, 30, (3, 5), None),
    ("allenai/Olmo-3.1-32B-Instruct", "olmo", set(), 32, (3, 1), None),
    ("allenai/Olmo-3-7B-Instruct", "olmo", set(), 7, (3,), None),
    ("tiiuae/Falcon-H1-34B-Instruct", "falcon-h1", set(), 34, (), None),
    ("tiiuae/Falcon-H1R-7B", "falcon-h1r", set(), 7, (), None),
    ("tiiuae/Falcon-H1-7B-Instruct", "falcon-h1", set(), 7, (), None),
    ("CohereLabs/command-a-reasoning-08-2025", "command-a-reasoning", set(), None, (), LineDate("mm-yyyy", 2025, 8, None)),
    ("CohereLabs/North-Mini-Code-1.0", "north-mini-code", set(), None, (1, 0), None),
    ("CohereLabs/aya-expanse-8b", "aya-expanse", set(), 8, (), None),
    ("CohereLabs/tiny-aya-global", "tiny-aya-global", set(), None, (), None),
    ("LiquidAI/LFM2.5-8B-A1B", "lfm", set(), 8, (2, 5), None),
    ("LiquidAI/LFM2.5-2.6B", "lfm", set(), 2.6, (2, 5), None),
    ("HuggingFaceTB/SmolLM3-3B", "smollm", set(), 3, (3,), None),
    ("openbmb/MiniCPM5-2B", "minicpm", set(), 2, (5,), None),
    ("tencent/Hy-MT2-30B-A3B", "hy-mt", set(), 30, (2,), None),
    ("baidu/ERNIE-4.5-21B-A3B-Thinking", "ernie", {"thinking"}, 21, (4, 5), None),
    ("MiniMaxAI/MiniMax-H3", "minimax-h3", set(), None, (), None),
    # --- the recorded answers and the cases the grammar was written for ---
    ("Qwen/Qwen3.8-2.4T-A95B", "qwen", set(), 2400, (3, 8), None),
    ("Qwen/Qwen3-VL-4B-Instruct", "qwen", {"vl"}, 4, (3,), None),
    ("Qwen/Qwen3.5-9B-Base", "qwen", {"base"}, 9, (3, 5), None),
    ("Qwen/Qwen3-235B-A22B", "qwen", set(), 235, (3,), None),
    ("other-org/Nebula-9B", "nebula", set(), 9, (), None),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "smollm", set(), 0.36, (2,), None),
    ("acme/Nova-7B", "nova", set(), 7, (), None),
    ("acme/Nova-7B-2512", "nova", set(), 7, (), _yymm("2512")),
    ("acme/Nova-1B", "nova", set(), 1, (), None),
    # `a0.6b` is the active part of a mixture and no size; nothing else marks a boundary.
    ("acme/Nova-A0.6B", "nova", set(), None, (), None),
    # `16e` and `bf16` are read and ignored.
    ("acme/Nova-2-7B-16E-BF16", "nova", set(), 7, (2,), None),
]

NOT_INTERPRETABLE = [
    "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU",  # `735` has no class
    "acme/Nova-2025",  # a year alone: neither YYMM nor MMDD, and too long for a version
    "acme/Nova-12345",
    "acme/Nova-735",
    "acme/Nova-2-7B-3",  # a second version token
    "acme/Nova-2506-0731",  # a second date
    "acme/Nova-13-2025",  # month 13: no MM-YYYY date, and `2025` alone has no class
    "acme/Nova--7B",  # an empty token
    "acme/3-7B",  # no family word in front of the boundary
]


@pytest.mark.parametrize("repo, family, variant, size, version, dated", KEYS, ids=[row[0] for row in KEYS])
def test_the_key_of_a_name(repo, family, variant, size, version, dated):
    key = line_key(repo, None)

    assert key is not None
    assert (key.family, set(key.variant), key.version, key.date) == (family, variant, version, dated)
    assert key.size == (pytest.approx(size) if size is not None else None)
    assert key.publisher == repo.split("/", 1)[0].lower()


def test_the_table_covers_every_row_of_the_shipped_catalog():
    catalog_repos = {model.hf_repo for family in load_catalog().families for model in family.models}

    assert len(catalog_repos) == 62
    assert catalog_repos <= {row[0] for row in KEYS}


@pytest.mark.parametrize("repo", NOT_INTERPRETABLE)
def test_a_name_the_grammar_has_no_reading_for_is_none(repo):
    assert line_key(repo, None) is None


@pytest.mark.parametrize(
    "repo", ["acme/Nova-v" + "9" * 5000, "acme/Nova" + "9" * 5000 + "-7B"], ids=["version", "word-with-version"]
)
def test_a_version_too_long_for_a_number_is_no_reading_and_no_crash(repo):
    """A repository id has no length limit here, and Python refuses `int` of 5000 digits."""
    assert line_key(repo, None) is None


def test_the_size_comes_from_the_name_first_and_else_from_the_parameter_count():
    assert line_key("acme/Nova-7B", 7.6).size == 7
    assert line_key("zai-org/GLM-5.3-Flash", 321.0).size == pytest.approx(321.0)
    assert line_key("zai-org/GLM-5.3-Flash", None).size is None


def test_a_word_with_a_number_behind_the_boundary_is_a_word_of_the_variant():
    key = line_key("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", None)

    assert key.version == ()
    assert "qwen3" in key.variant


def test_a_month_and_a_year_are_one_date_and_a_month_alone_is_a_version():
    assert line_key("acme/Nova-08-2025", None).date == LineDate("mm-yyyy", 2025, 8, None)
    assert line_key("acme/Nova-08", None).version == (8,)


def test_instruct_it_and_chat_are_no_variant():
    assert line_key("acme/Nova-2-7B-Instruct", None).variant == frozenset()
    assert line_key("acme/Nova-2-7B-it", None).variant == frozenset()
    assert line_key("acme/Nova-2-7B-Chat", None).variant == frozenset()


def test_instruct_before_the_boundary_is_no_word_of_the_family_either():
    """`Mistral-Small-Instruct-2409` is a Mistral Small, not a family of its own (author decision,
    2026-09-25): the instruction-tuned model is the plain one before the boundary as behind it."""
    key = line_key("mistralai/Mistral-Small-Instruct-2409", None)

    assert (key.family, key.variant, key.version, key.date) == ("mistral-small", frozenset(), (), _yymm("2409"))
    assert _compute(["mistralai/Mistral-Small-Instruct-2409", MISTRAL_SMALL_4]) == {
        "mistralai/Mistral-Small-Instruct-2409": Release("legacy", MISTRAL_SMALL_4),
        MISTRAL_SMALL_4: Release("latest", None),
    }


# --- the order ------------------------------------------------------------------------------------


def _over(upper: str, lower: str) -> bool:
    return line_key(upper, None).stands_over(line_key(lower, None))


def test_a_higher_version_stands_over_a_lower_one_and_no_version_is_the_lowest():
    assert _over("acme/Nova-3-7B", "acme/Nova-2.9-7B")
    assert _over("acme/Nova-2-7B", "acme/Nova-7B")
    assert not _over("acme/Nova-7B", "acme/Nova-2-7B")


def test_with_the_same_version_a_dated_name_stands_over_an_undated_one():
    assert _over("acme/Nova-7B-2512", "acme/Nova-7B")
    assert not _over("acme/Nova-7B", "acme/Nova-7B-2512")
    assert _over("acme/Nova-7B-2509", "acme/Nova-7B-2506")


def test_dates_of_two_forms_are_not_comparable():
    assert not _over("acme/Nova-7B-2506", "acme/Nova-7B-0731")
    assert not _over("acme/Nova-7B-0731", "acme/Nova-7B-2506")


def test_the_order_is_irreflexive_and_transitive_over_the_table():
    keys = [line_key(row[0], None) for row in KEYS]
    groups: dict = {}
    for key in keys:
        groups.setdefault(key.group, []).append(key)
    for members in groups.values():
        for a in members:
            assert not a.stands_over(a)
            for b, c in itertools.product(members, members):
                if a.stands_over(b) and b.stands_over(c):
                    assert a.stands_over(c)


# --- compute_release ------------------------------------------------------------------------------


def _hit(base: str, *, age: str = "unknown", successor: str | None = None, parameters_b: float | None = None,
         repo: str | None = None) -> SearchHit:
    return SearchHit(
        repo=repo or f"packager/{base.split('/', 1)[1]}-GGUF",
        publisher_status="listed packager",
        base_model=[base],
        base_model_relation="quantized",
        resolved=True,
        resolved_base_model=base,
        age=age,
        successor=successor,
        parameters_b=parameters_b,
        release_basis="stated" if age != "unknown" else None,
    )


def _silent(*bases: str) -> dict[str, AgeVerdict]:
    """What `decide_age` says about a model neither the repository nor the catalog decides."""
    return {base: AgeVerdict("unknown", None, "catalog") for base in bases}


def _empty() -> Catalog:
    return Catalog.model_validate({"schema_version": 1, "families": []})


def _catalog(*models: dict, publisher: str = "acme") -> Catalog:
    return Catalog.model_validate(
        {"schema_version": 1, "families": [{"name": "nova", "publisher": publisher, "models": list(models)}]}
    )


def _row(hf_repo: str, **fields) -> dict:
    return {"hf_repo": hf_repo, "source": f"https://huggingface.co/{hf_repo}", **fields}


def _latest_row(hf_repo: str) -> dict:
    owner = hf_repo.split("/", 1)[0]
    return _row(hf_repo, latest=True, latest_source=f"https://huggingface.co/{owner}",
                latest_checked=date(2026, 9, 25).isoformat())


def _compute(bases: list[str], catalog: Catalog | None = None, ages: dict | None = None, hits=None):
    hits = hits if hits is not None else [_hit(base) for base in bases]
    return compute_release(hits, catalog if catalog is not None else _empty(), ages if ages is not None else _silent(*bases))


def test_a_lower_version_of_a_group_without_sizes_is_legacy_with_the_highest_as_successor():
    """Decided 2026-09-25: the size is no rank, so Mistral Small 3.2 (24B) is `legacy` to Small 4."""
    hits = [_hit(MISTRAL_SMALL_3), _hit(MISTRAL_SMALL_4, age="latest")]
    ages = {**_silent(MISTRAL_SMALL_3), MISTRAL_SMALL_4: AgeVerdict("latest", None, "catalog")}

    result = compute_release(hits, load_catalog(), ages)

    assert result == {MISTRAL_SMALL_3: Release("legacy", MISTRAL_SMALL_4)}


@pytest.mark.parametrize(
    "lower, upper",
    [
        ("acme/Nova-2-VL-7B", "acme/Nova-3-7B"),
        ("meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        ("zai-org/GLM-4.7-Flash", "zai-org/GLM-5.3-Pro"),
        ("acme/Nova-2-Mini", "acme/Nova-3"),
        ("acme/Nova-2-9B-Base", "acme/Nova-3-9B"),
    ],
)
def test_a_variant_is_a_group_of_its_own(lower, upper):
    assert _compute([lower, upper]) == {lower: Release("latest", None), upper: Release("latest", None)}


def test_instruct_is_the_plain_model_and_no_group_of_its_own():
    result = _compute(["acme/Nova-2-7B-Instruct", "acme/Nova-3-7B"])

    assert result["acme/Nova-2-7B-Instruct"] == Release("legacy", "acme/Nova-3-7B")


def test_every_maximal_member_is_latest_whatever_its_size():
    result = _compute(["acme/Nova-3-7B", "acme/Nova-3-70B", "acme/Nova-2-7B"])

    assert result["acme/Nova-3-7B"] == result["acme/Nova-3-70B"] == Release("latest", None)
    assert result["acme/Nova-2-7B"] == Release("legacy", "acme/Nova-3-7B")


def test_the_result_does_not_depend_on_the_order_of_the_hits():
    bases = ["acme/Nova-7B-2506", "acme/Nova-7B", "acme/Nova-7B-2509"]
    expected = {
        "acme/Nova-7B-2509": Release("latest", None),
        "acme/Nova-7B-2506": Release("legacy", "acme/Nova-7B-2509"),
        "acme/Nova-7B": Release("legacy", "acme/Nova-7B-2509"),
    }

    for order in itertools.permutations(bases):
        assert _compute(list(order)) == expected


def test_two_dates_of_different_forms_are_both_latest():
    result = _compute(["acme/Nova-7B-2506", "acme/Nova-7B-0731"])

    assert set(result.values()) == {Release("latest", None)}


def test_a_group_of_one_is_latest():
    assert _compute(["other-org/Nebula-9B"]) == {"other-org/Nebula-9B": Release("latest", None)}


def test_a_name_without_a_version_is_under_one_with_a_version():
    assert _compute(["acme/Nova-7B", "acme/Nova-2-7B"])["acme/Nova-7B"] == Release("legacy", "acme/Nova-2-7B")


@pytest.mark.parametrize(
    "legacy, uppers, expected",
    [
        # the same size
        ("acme/Nova-2-7B", ["acme/Nova-3-3B", "acme/Nova-3-7B", "acme/Nova-3-70B"], "acme/Nova-3-7B"),
        # else the next larger one
        ("acme/Nova-2-9B", ["acme/Nova-3-3B", "acme/Nova-3-27B", "acme/Nova-3-70B"], "acme/Nova-3-27B"),
        # else the next smaller one
        ("acme/Nova-2-400B", ["acme/Nova-3-3B", "acme/Nova-3-27B"], "acme/Nova-3-27B"),
        # a candidate without a size drops out once one with a size fits
        ("acme/Nova-2-7B", ["acme/Nova-3", "acme/Nova-3-9B"], "acme/Nova-3-9B"),
    ],
)
def test_the_successor_is_the_candidate_of_the_nearest_size(legacy, uppers, expected):
    result = _compute([legacy, *uppers])

    assert result[legacy] == Release("legacy", expected)


def test_the_same_size_means_the_same_number_and_not_a_near_one():
    """Two sizes that differ are two sizes, however close: the same one wins, not both."""
    result = _compute(["acme/Nova-1-7B", "acme/Nova-2-7B", "acme/Nova-2-7.000000001B"])

    assert result["acme/Nova-1-7B"] == Release("legacy", "acme/Nova-2-7B")


def test_the_next_larger_size_wins_over_a_slightly_smaller_one():
    result = _compute(["acme/Nova-1-7B", "acme/Nova-2-6.999999999B", "acme/Nova-2-9B"])

    assert result["acme/Nova-1-7B"] == Release("legacy", "acme/Nova-2-9B")


def test_a_name_too_long_to_read_takes_no_part_and_ends_nothing():
    long_one = "acme/Nova-v" + "9" * 5000

    assert _compute([long_one, "acme/Nova-2-7B"]) == {"acme/Nova-2-7B": Release("latest", None)}


def test_one_candidate_is_the_successor_whatever_the_sizes_say():
    assert _compute(["acme/Nova-2", "acme/Nova-3-7B"])["acme/Nova-2"] == Release("legacy", "acme/Nova-3-7B")


def test_an_unknown_size_and_more_than_one_candidate_stays_unknown():
    result = _compute(["acme/Nova-2", "acme/Nova-3-7B", "acme/Nova-3-9B"])

    assert "acme/Nova-2" not in result


def test_two_candidates_that_fit_equally_well_leave_the_row_unknown():
    result = _compute(["acme/Nova-1-7B", "acme/Nova-2-7B-Instruct", "acme/Nova-2-7B-Chat"])

    assert "acme/Nova-1-7B" not in result
    assert result["acme/Nova-2-7B-Instruct"] == result["acme/Nova-2-7B-Chat"] == Release("latest", None)


def test_a_maximal_member_with_a_stated_legacy_is_still_the_candidate():
    """Magistral 2509 is `legacy` by the catalog and the highest of its line all the same."""
    old = "mistralai/Magistral-Small-2506"
    stated = "mistralai/Magistral-Small-2509"
    hits = [_hit(old), _hit(stated, age="legacy", successor=MISTRAL_SMALL_4)]
    ages = {**_silent(old), stated: AgeVerdict("legacy", MISTRAL_SMALL_4, "catalog")}

    result = compute_release(hits, load_catalog(), ages)

    assert result == {old: Release("legacy", stated)}


def test_a_stated_release_stays_as_it_is_and_counts_as_a_member():
    catalog = _catalog(_latest_row("acme/Nova-3-7B"), _row("acme/Nova-2-7B"))
    hits = [_hit("acme/Nova-3-7B", age="latest"), _hit("acme/Nova-2-7B")]
    ages = {"acme/Nova-3-7B": AgeVerdict("latest", None, "catalog"), **_silent("acme/Nova-2-7B")}

    result = compute_release(hits, catalog, ages)

    assert result == {"acme/Nova-2-7B": Release("legacy", "acme/Nova-3-7B")}


def test_a_catalog_row_without_a_hit_is_a_member_and_gets_no_status():
    catalog = _catalog(_row("acme/Nova-3-7B"))

    assert _compute(["acme/Nova-2-7B"], catalog) == {"acme/Nova-2-7B": Release("legacy", "acme/Nova-3-7B")}


@pytest.mark.parametrize(
    "verdict",
    [
        AgeVerdict("unknown", None, "budget exhausted before the first edge"),
        AgeVerdict("unknown", None, "new_version 'acme/Gone' is not a reachable repo of acme"),
    ],
)
def test_an_unknown_that_a_fault_decided_is_not_computed(verdict):
    """The group has a maximum, and still: a broken pointer or a budget is no silence."""
    ages = {"acme/Nova-2-7B": verdict, **_silent("acme/Nova-3-7B")}

    result = _compute(["acme/Nova-2-7B", "acme/Nova-3-7B"], ages=ages)

    assert "acme/Nova-2-7B" not in result
    assert result["acme/Nova-3-7B"] == Release("latest", None)


def test_a_name_the_grammar_cannot_read_stays_unknown():
    davidau = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"

    assert _compute([davidau]) == {}


def test_computed_edges_on_a_cycle_with_the_stated_ones_all_fall():
    """Stated Nova-2 -> Luna-1 and Luna-2 -> Nova-1; computed Nova-1 -> Nova-2 and Luna-1 -> Luna-2."""
    rows = [_row("acme/Nova-2", successor="acme/Luna-1"), _row("acme/Luna-2", successor="acme/Nova-1")]
    bases = ["acme/Nova-1", "acme/Luna-1"]

    for row_order in itertools.permutations(rows):
        for base_order in itertools.permutations(bases):
            assert _compute(list(base_order), _catalog(*row_order)) == {}


def test_a_stated_edge_of_a_repository_closes_a_cycle_as_well():
    hits = [_hit("acme/Nova-1"), _hit("acme/Nova-2", age="legacy", successor="acme/Nova-1-Pro"),
            _hit("acme/Nova-1-Pro"), _hit("acme/Nova-2-Pro", age="legacy", successor="acme/Nova-1")]
    ages = {
        **_silent("acme/Nova-1", "acme/Nova-1-Pro"),
        "acme/Nova-2": AgeVerdict("legacy", "acme/Nova-1-Pro", "new_version at acme/Nova-2"),
        "acme/Nova-2-Pro": AgeVerdict("legacy", "acme/Nova-1", "new_version at acme/Nova-2-Pro"),
    }

    assert compute_release(hits, _empty(), ages) == {}


def test_a_chain_is_allowed():
    catalog = _catalog(_row("acme/Nova-3-7B", successor="acme/Luna-1-7B"), _latest_row("acme/Luna-1-7B"))

    assert _compute(["acme/Nova-2-7B"], catalog) == {"acme/Nova-2-7B": Release("legacy", "acme/Nova-3-7B")}


# --- the hits, rebuilt ----------------------------------------------------------------------------


def test_with_release_rebuilds_a_hit_it_computed_and_leaves_every_other_one():
    stated = _hit(MISTRAL_SMALL_4, age="latest")
    unresolved = SearchHit(repo="someone/Nova-GGUF", resolved=False, unresolved_reason="relation_unknown")
    hits = [_hit(MISTRAL_SMALL_3), stated, unresolved]
    ages = {**_silent(MISTRAL_SMALL_3), MISTRAL_SMALL_4: AgeVerdict("latest", None, "catalog")}

    rebuilt = with_release(hits, load_catalog(), ages)

    computed = rebuilt[0]
    assert (computed.age, computed.successor, computed.release_basis) == ("legacy", MISTRAL_SMALL_4, "computed")
    assert SearchHit.model_validate(computed.model_dump()) == computed
    assert rebuilt[1:] == [stated, unresolved]


def test_a_computed_basis_needs_a_known_age():
    with pytest.raises(ValidationError, match="computed"):
        SearchHit(repo="packager/Nova-GGUF", resolved=False, unresolved_reason="relation_unknown",
                  release_basis="computed")


def test_a_stated_basis_or_none_goes_with_any_age():
    assert _hit("acme/Nova-7B").release_basis is None
    assert _hit("acme/Nova-7B", age="latest").release_basis == "stated"
