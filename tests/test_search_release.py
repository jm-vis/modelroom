"""Tests for `search_release`: a release computed from the version numbers of one family.

The key table covers every reference row of the shipped catalog (`catalog_v1_reference.toml`) plus the
names of the recorded answers, hand samples and the cases the grammar was written for; any other shipped
row must give a `LineKey` unless it is listed with a reason in `SHIPPED_WITHOUT_READING`; a name the table
does not name is a name the grammar was not checked against (CONTRACTS.md, "Latest and legacy evidence", "Computed release").
"""

from __future__ import annotations

import itertools
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelroom.catalog import Catalog, load_catalog
from modelroom.guided_contracts import SearchHit
from modelroom.search_age import AgeVerdict
from modelroom.search_release import LineDate, LineKey, Release, compute_release, line_key, unread_reason, with_release

MISTRAL_SMALL_3 = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
MISTRAL_SMALL_4 = "mistralai/Mistral-Small-4-119B-2603"


def _yymm(value: str) -> LineDate:
    return LineDate("yymm", int(value[:2]), int(value[2:]), None)


def _mmdd(value: str) -> LineDate:
    return LineDate("mmdd", None, int(value[:2]), int(value[2:]))


# (hf_repo, family, line, variant, size, version, date) -- the family without its publisher: its first
# word (GF F3, AP9-UX2 U5); the line a word of `LINE_WORDS`/`LINE_COUNTERS`; every other word a variant.
KEYS = [
    # --- the shipped catalog, the 62 rows of the first publication ---
    ("mistralai/Ministral-3-14B-Instruct-2512", "ministral", "", set(), 14, (3,), _yymm("2512")),
    ("mistralai/Ministral-3-8B-Instruct-2512", "ministral", "", set(), 8, (3,), _yymm("2512")),
    ("mistralai/Ministral-3-3B-Instruct-2512", "ministral", "", set(), 3, (3,), _yymm("2512")),
    (MISTRAL_SMALL_4, "mistral", "small", set(), 119, (4,), _yymm("2603")),
    ("mistralai/Devstral-Small-2-24B-Instruct-2512", "devstral", "small", set(), 24, (2,), _yymm("2512")),
    ("mistralai/Magistral-Small-2509", "magistral", "small", set(), None, (), _yymm("2509")),
    (MISTRAL_SMALL_3, "mistral", "small", set(), 24, (3, 2), _yymm("2506")),
    ("utter-project/EuroLLM-9B-Instruct", "eurollm", "", set(), 9, (), None),
    ("utter-project/EuroLLM-9B-Instruct-2512", "eurollm", "", set(), 9, (), _yymm("2512")),
    ("utter-project/EuroLLM-22B-Instruct-2512", "eurollm", "", set(), 22, (), _yymm("2512")),
    ("utter-project/EuroMoE-2.6B-A0.6B-Instruct-2512", "euromoe", "", set(), 2.6, (), _yymm("2512")),
    ("openGPT-X/Teuken-7B-instruct-v0.6", "teuken", "", set(), 7, (0, 6), None),
    ("Qwen/Qwen3.5-9B", "qwen", "", set(), 9, (3, 5), None),
    ("Qwen/Qwen3.5-4B", "qwen", "", set(), 4, (3, 5), None),
    ("Qwen/Qwen3.5-27B", "qwen", "", set(), 27, (3, 5), None),
    ("Qwen/Qwen3.5-35B-A3B", "qwen", "", set(), 35, (3, 5), None),
    ("Qwen/Qwen3.5-122B-A10B", "qwen", "", set(), 122, (3, 5), None),
    ("Qwen/Qwen3.6-27B", "qwen", "", set(), 27, (3, 6), None),
    ("Qwen/Qwen3.6-35B-A3B", "qwen", "", set(), 35, (3, 6), None),
    ("Qwen/Qwen3.8-27B", "qwen", "", set(), 27, (3, 8), None),
    ("Qwen/Qwen3-Coder-Next", "qwen", "", {"coder", "next"}, None, (3,), None),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen", "", {"coder"}, 30, (3,), None),
    ("Qwen/Qwen3-Embedding-8B", "qwen", "", {"embedding"}, 8, (3,), None),
    ("Qwen/Qwen3-Embedding-0.6B", "qwen", "", {"embedding"}, 0.6, (3,), None),
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "deepseek", "r", {"qwen3"}, 8, (1,), _mmdd("0528")),
    ("deepseek-ai/DeepSeek-V3.2", "deepseek", "", set(), None, (3, 2), None),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek", "", {"flash"}, None, (4,), _mmdd("0731")),
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama", "", {"scout"}, 17, (4,), None),
    ("meta-llama/Llama-3.3-70B-Instruct", "llama", "", set(), 70, (3, 3), None),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama", "", set(), 8, (3, 1), None),
    ("meta-llama/Llama-3.2-3B-Instruct", "llama", "", set(), 3, (3, 2), None),
    ("google/gemma-4-31B-it", "gemma", "", set(), 31, (4,), None),
    ("google/gemma-4-26B-A4B-it", "gemma", "", set(), 26, (4,), None),
    ("google/gemma-4-12B-it", "gemma", "", set(), 12, (4,), None),
    ("google/gemma-4-E4B-it", "gemma", "", set(), 4, (4,), None),
    ("microsoft/phi-4", "phi", "", set(), None, (4,), None),
    ("microsoft/Phi-4-mini-instruct", "phi", "", {"mini"}, None, (4,), None),
    ("openai/gpt-oss-120b", "gpt", "", {"oss"}, 120, (), None),
    ("openai/gpt-oss-20b", "gpt", "", {"oss"}, 20, (), None),
    ("zai-org/GLM-5.3-Flash", "glm", "", {"flash"}, None, (5, 3), None),
    ("zai-org/GLM-4.7-Flash", "glm", "", {"flash"}, None, (4, 7), None),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "kimi", "", {"linear"}, 48, (), None),
    ("ibm-granite/granite-4.2-30b", "granite", "", set(), 30, (4, 2), None),
    ("ibm-granite/granite-4.2-8b", "granite", "", set(), 8, (4, 2), None),
    ("ibm-granite/granite-4.2-3b", "granite", "", set(), 3, (4, 2), None),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nvidia", "", {"lightning", "nemotron"}, 30, (3, 5), None),
    ("allenai/Olmo-3.1-32B-Instruct", "olmo", "", set(), 32, (3, 1), None),
    ("allenai/Olmo-3-7B-Instruct", "olmo", "", set(), 7, (3,), None),
    ("tiiuae/Falcon-H1-34B-Instruct", "falcon", "", {"h1"}, 34, (), None),
    ("tiiuae/Falcon-H1R-7B", "falcon", "", {"h1r"}, 7, (), None),
    ("tiiuae/Falcon-H1-7B-Instruct", "falcon", "", {"h1"}, 7, (), None),
    ("CohereLabs/command-a-reasoning-08-2025", "command", "", {"a", "reasoning"}, None, (), LineDate("mm-yyyy", 2025, 8, None)),
    ("CohereLabs/North-Mini-Code-1.0", "north", "", {"code", "mini"}, None, (1, 0), None),
    ("CohereLabs/aya-expanse-8b", "aya", "", {"expanse"}, 8, (), None),
    ("CohereLabs/tiny-aya-global", "tiny", "", {"aya", "global"}, None, (), None),
    ("LiquidAI/LFM2.5-8B-A1B", "lfm", "", set(), 8, (2, 5), None),
    ("LiquidAI/LFM2.5-2.6B", "lfm", "", set(), 2.6, (2, 5), None),
    ("HuggingFaceTB/SmolLM3-3B", "smollm", "", set(), 3, (3,), None),
    ("openbmb/MiniCPM5-2B", "minicpm", "", set(), 2, (5,), None),
    ("tencent/Hy-MT2-30B-A3B", "hy", "", {"mt"}, 30, (2,), None),
    ("baidu/ERNIE-4.5-21B-A3B-Thinking", "ernie", "", {"thinking"}, 21, (4, 5), None),
    ("MiniMaxAI/MiniMax-H3", "minimax", "", {"h3"}, None, (), None),
    # --- rows added by the catalog publication of 2026-10-07 ---
    ("CohereLabs/tiny-aya-en-thinker", "tiny", "", {"aya", "en", "thinker"}, None, (), None),
    ("CohereLabs/tiny-aya-l2-thinker", "tiny", "", {"aya", "l2", "thinker"}, None, (), None),
    ("LiquidAI/LFM2.5-VL-3B-DSpark", "lfm", "", {"vl", "dspark"}, 3, (2, 5), None),
    ("Qwen/Qwen-Image-2.1", "qwen", "", {"image"}, None, (2, 1), None),
    ("Qwen/Qwen-Image-2.1-PE-I2I", "qwen", "", {"image", "pe", "i2i"}, None, (2, 1), None),
    ("Qwen/Qwen-Image-2.1-PE-T2I", "qwen", "", {"image", "pe", "t2i"}, None, (2, 1), None),
    ("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "deepseek", "", {"flash", "vision", "exp"}, None, (4,), None),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek", "", {"flash"}, None, (4, 1), None),
    ("google/embeddinggemma-2", "embeddinggemma", "", set(), None, (2,), None),
    ("microsoft/FrogNano-4B-2609", "frognano", "", set(), 4, (), _yymm("2609")),
    ("microsoft/VibeVoice-ASR-Streaming-1.5B", "vibevoice", "", {"asr", "streaming"}, 1.5, (), None),
    ("microsoft/VibeVoice-ASR-Streaming-7B", "vibevoice", "", {"asr", "streaming"}, 7, (), None),
    ("openbmb/MiniCPM5-2B-DSpark", "minicpm", "", {"dspark"}, 2, (5,), None),
    ("openbmb/MiniCPM5-2B-Midtrain", "minicpm", "", {"midtrain"}, 2, (5,), None),
    ("tencent/EVIE-4.5B", "evie", "", set(), 4.5, (), None),
    ("tencent/EVIE-8B", "evie", "", set(), 8, (), None),
    ("tencent/WeVisDoc-2B", "wevisdoc", "", set(), 2, (), None),
    ("tencent/WeVisDoc-4B", "wevisdoc", "", set(), 4, (), None),
    # --- the recorded answers and the cases the grammar was written for ---
    ("Qwen/Qwen3.8-2.4T-A95B", "qwen", "", set(), 2400, (3, 8), None),
    ("Qwen/Qwen3-VL-4B-Instruct", "qwen", "", {"vl"}, 4, (3,), None),
    ("Qwen/Qwen3.5-9B-Base", "qwen", "", {"base"}, 9, (3, 5), None),
    ("Qwen/Qwen3-235B-A22B", "qwen", "", set(), 235, (3,), None),
    ("other-org/Nebula-9B", "nebula", "", set(), 9, (), None),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "smollm", "", set(), 0.36, (2,), None),
    ("acme/Nova-7B", "nova", "", set(), 7, (), None),
    ("acme/Nova-7B-2512", "nova", "", set(), 7, (), _yymm("2512")),
    ("acme/Nova-1B", "nova", "", set(), 1, (), None),
    # `a0.6b` is the active part of a mixture and no size; nothing else marks a boundary.
    ("acme/Nova-A0.6B", "nova", "", set(), None, (), None),
    # `16e` and `bf16` are read and ignored.
    ("acme/Nova-2-7B-16E-BF16", "nova", "", set(), 7, (2,), None),
    # --- AP9-KA-G: names the extended grammar reads ---
    ("01-ai/Yi-6B-200K", "yi", "", set(), 6, (), None),
    ("CohereLabs/tiny-aya-base-32K", "tiny", "", {"aya", "base"}, None, (), None),
    ("Skywork/SkyReels-V2-DF-14B-720P", "skyreels", "", {"df"}, 14, (2,), None),
    ("Skywork/SkyReels-V2-I2V-14B-720P", "skyreels", "", {"i2v"}, 14, (2,), None),
    ("Skywork/SkyReels-V2-T2V-14B-720P", "skyreels", "", {"t2v"}, 14, (2,), None),
    ("Wan-AI/Wan2.1-I2V-14B-480P", "wan", "", {"i2v"}, 14, (2, 1), None),
    ("black-forest-labs/FLUX.1-Kontext-dev", "flux", "", {"kontext", "dev"}, None, (1,), None),
    ("black-forest-labs/FLUX.1-dev", "flux", "", {"dev"}, None, (1,), None),
    ("black-forest-labs/FLUX.1-schnell", "flux", "", {"schnell"}, None, (1,), None),
    ("black-forest-labs/FLUX.2-dev", "flux", "", {"dev"}, None, (2,), None),
    ("black-forest-labs/FLUX.2-klein-4B", "flux", "", {"klein"}, 4, (2,), None),
    ("inception42/jais-family-1p3b-chat", "jais", "", {"family"}, None, (), None),
    ("inception42/jais-family-2p7b-chat", "jais", "", {"family"}, None, (), None),
    ("upstage/SOLAR-0-70b-16bit", "solar", "", set(), 70, (0,), None),
    ("Aleph-Alpha/llama-3_1-8b-tfree-hat-dpo", "llama", "", {"tfree", "hat", "dpo"}, 8, (3, 1), None),
    ("Aleph-Alpha/llama-3_1-8b-tfree-hat-sft", "llama", "", {"tfree", "hat", "sft"}, 8, (3, 1), None),
    ("OpenGVLab/InternVL2_5-4B-MPO", "internvl", "", {"mpo"}, 4, (2, 5), None),
    ("internlm/internlm2_5-7b-chat", "internlm", "", set(), 7, (2, 5), None),
    ("stepfun-ai/GOT-OCR2_0", "got", "", {"ocr"}, None, (2, 0), None),
    # --- AP9-KA-H K1: a `\d+s` attribute and a date in ISO form ---
    ("m-a-p/MERT-v2-30s", "mert", "", set(), None, (2,), None),
    ("moondream/moondream-2b-2025-04-14", "moondream", "", set(), 2, (), LineDate("iso", 2025, 4, 14)),
]

# `KEYS` = the reference rows plus hand samples. This list: shipped names that deliberately have no reading (`line_key`
# is `None`), each with a reason; every other shipped row outside the reference must give a `LineKey` (coverage test).
# Empty as of the publication of 2026-10-07; the publication of 2026-10-08 (231 rows) added 24. AP9-KA-G (decided 2026-10-08)
# reads 19 of them (see the last block of `KEYS`); these stay out of the line comparison, each a decision of the
# owner and not a grammar fault: reading them would extend the grammar, which is a product contract for the computed
# release. Each entry is checked to stay unread (test below).
SHIPPED_WITHOUT_READING: list[str] = [
    # a bare number (`378`): the patch size of ViT-H/14, no version (class 3, rejected in KA-G)
    "apple/DFN5B-CLIP-ViT-H-14-378",
    # a dimension (`2d`): reading it as a variation gives a wrong successor (class 4, rejected in KA-G)
    "mixedbread-ai/mxbai-embed-2d-large-v1",
    # a version and a trailing `v` revision: two version tokens, no evidence which one counts (class 9, rejected in KA-G)
    "sarvamai/sarvam-1-v0.5",
    "sbintuitions/sarashina2.2-0.5b-instruct-v0.1",
    "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    # three digits (`512`) have no class; shipped without a reading by decision, the grammar stays as it is
    "kyutai/ovie-512",
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
    # AP9-KA-G: near misses of the new token patterns and the cases it rejects
    "acme/Nova-7B-378",  # a bare number
    "acme/Nova-2d-7B",  # a dimension
    "acme/Nova-2-v0.5",  # a version and a `v` revision
    "acme/Nova-r.1-7B",  # a dotted word of one letter
    "acme/Nova-1080i",  # not a resolution in `p`
    "acme/Nova-7B-16bits",  # not a `bit` precision
    "acme/Nova-1p3-7B",  # a `p` size without a unit
    "acme/Nova-2.5_1",  # the underscore joins no part with a dot
    # AP9-KA-H: near misses of the ISO date, and the three-digit number that stays without a class
    "acme/Nova-2025-13-01",  # month 13: no ISO date, and `2025` alone has no class
    "acme/Nova-2025-04-32",  # day 32
    "acme/Nova-1999-04-14",  # not a year of `20xx`
    "acme/Nova-2025-04-14-2025-05-01",  # two ISO dates
    "acme/Nova-2025-04-14-2506",  # an ISO date and a YYMM date
]


@pytest.mark.parametrize("repo, family, line, variant, size, version, dated", KEYS, ids=[row[0] for row in KEYS])
def test_the_key_of_a_name(repo, family, line, variant, size, version, dated):
    key = line_key(repo, None)

    assert key is not None
    assert (key.family, key.line, set(key.variant), key.version, key.date) == (family, line, variant, version, dated)
    assert key.size == (pytest.approx(size) if size is not None else None)
    assert key.publisher == repo.split("/", 1)[0].lower()


def test_the_table_covers_every_row_of_the_shipped_catalog():
    """The rows of the first publication (`catalog_v1_reference.toml`) are in `KEYS`; every other shipped row gives a
    `LineKey` too, unless it stands with a reason in `SHIPPED_WITHOUT_READING`: a lost reading is a decision, never a
    side effect."""
    reference = load_catalog(Path(__file__).parent / "fixtures" / "catalog_v1_reference.toml")
    reference_repos = {model.hf_repo for family in reference.families for model in family.models}
    assert reference_repos <= {row[0] for row in KEYS} | set(SHIPPED_WITHOUT_READING)
    assert unread_rows(load_catalog(), reference_repos, line_key) == []


def unread_rows(catalog, reference_repos, reading) -> list[str]:
    """The shipped rows outside the reference that `reading` gives no `LineKey` and that are not listed, with a reason, in
    `SHIPPED_WITHOUT_READING`: a reading lost is a decision, never a side effect (AP9-KA-T3, round 5)."""
    lost = []
    for family in catalog.families:
        for model in family.models:
            if model.hf_repo in reference_repos or model.hf_repo in SHIPPED_WITHOUT_READING:
                continue
            if not isinstance(reading(model.hf_repo, model.parameters_b), LineKey):
                lost.append(model.hf_repo)
    return lost


def test_a_shipped_row_that_loses_its_reading_is_named():
    catalog = load_catalog()
    reference = load_catalog(Path(__file__).parent / "fixtures" / "catalog_v1_reference.toml")
    reference_repos = {model.hf_repo for family in reference.families for model in family.models}
    extra = next(m.hf_repo for f in catalog.families for m in f.models if m.hf_repo not in reference_repos)
    assert unread_rows(catalog, reference_repos, lambda repo, size: None if repo == extra else line_key(repo, size)) == [extra]


@pytest.mark.parametrize("repo", NOT_INTERPRETABLE + SHIPPED_WITHOUT_READING)
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

    assert key.version == (1,)  # from the line counter `r1`, not from `qwen3` (AP9-UX2 U5)
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

    assert (key.family, key.line, key.variant, key.version, key.date) == ("mistral", "small", frozenset(), (), _yymm("2409"))
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


@pytest.mark.parametrize("year", ["2001", "2012"])
def test_a_year_from_2001_to_2012_in_an_iso_date_is_one_iso_date_and_never_a_yymm_date(year):
    """`2001` to `2012` alone read as YYMM (year 20, month 01 to 12); inside `yyyy-mm-dd` the three tokens are one date."""
    key = line_key(f"acme/Nova-{year}-05-07", None)

    assert key.date == LineDate("iso", int(year), 5, 7) and key.version == ()
    assert line_key(f"acme/Nova-{year}", None).date == LineDate("yymm", 20, int(year[2:]), None)  # regression: alone it stays YYMM


def test_the_iso_date_needs_all_three_tokens_and_leaves_a_year_with_a_month_as_before():
    assert line_key("acme/Nova-2001-05", None).date == LineDate("yymm", 20, 1, None)  # regression: `05` is a version
    assert line_key("acme/Nova-2001-05", None).version == (5,)


def test_an_iso_date_stands_over_an_older_one_and_over_no_date_but_the_version_comes_first():
    assert _over("acme/Nova-2025-05-01", "acme/Nova-2025-04-30")
    assert _over("acme/Nova-2026-01-01", "acme/Nova-2025-12-31")
    assert not _over("acme/Nova-2025-04-30", "acme/Nova-2025-05-01")
    assert _over("acme/Nova-2025-04-14", "acme/Nova")
    assert not _over("acme/Nova", "acme/Nova-2025-04-14")
    assert _over("acme/Nova-3", "acme/Nova-2-2025-12-31")  # a higher version beats any date, whatever its form
    assert not _over("acme/Nova-2025-12-31", "acme/Nova-2-2025-01-01")  # equal versions only (here () against (2,))


def test_the_higher_version_wins_whatever_the_forms_of_the_dates_are():
    """Codex kah-1 finding 10: the version comes first even when the two names carry dates of different forms."""
    assert _over("acme/Nova-3-2506", "acme/Nova-2-2025-12-31")  # YYMM against ISO
    assert not _over("acme/Nova-2-2025-12-31", "acme/Nova-3-2506")
    assert _over("acme/Nova-3-2025-01-01", "acme/Nova-2-2512")  # ISO against YYMM, the other way round
    assert not _over("acme/Nova-2-2512", "acme/Nova-3-2025-01-01")


def test_an_iso_date_and_a_date_of_another_form_are_not_comparable():
    assert not _over("acme/Nova-2025-05-01", "acme/Nova-2506")
    assert not _over("acme/Nova-2506", "acme/Nova-2025-05-01")


def test_a_seconds_attribute_is_read_and_ignored_and_it_does_not_order_names():
    """`MERT-v2-30s`: seconds of audio, an attribute like a `k` attribute (AP9-KA-H); two of them are as good as one."""
    assert line_key("acme/Nova-30s-45s", None) is not None
    plain, thirty, forty_five = (line_key(f"acme/Nova-2{tail}", None) for tail in ("", "-30s", "-45s"))
    assert (plain.version, thirty.version, forty_five.version) == ((2,), (2,), (2,))
    assert (plain.size, plain.date, plain.variant) == (thirty.size, thirty.date, thirty.variant) == (forty_five.size, forty_five.date, forty_five.variant)
    assert not thirty.stands_over(forty_five) and not forty_five.stands_over(thirty) and not thirty.stands_over(plain)


@pytest.mark.parametrize("repo, reason", [
    ("kyutai/ovie-512", "token '512' has no class"),
    ("acme/Nova-2025-13-01", "token '2025' has no class"),
    ("acme/Nova-2-7B-3", "two versions"),
    ("acme/Nova-2506-0731", "two dates"),
    ("acme/Nova-2025-04-14-2025-05-01", "two dates"),
    ("acme/3-7B", "starts with version"),
    ("apple/DFN5B-CLIP-ViT-H-14-378", "token '378' has no class"),
])
def test_the_reason_a_name_has_no_reading_is_named(repo, reason):
    assert unread_reason(repo, None) == reason


def test_a_name_with_a_reading_has_no_reason_to_give():
    assert unread_reason("m-a-p/MERT-v2-30s", None) is None and unread_reason("acme/Nova-7B", 7.0) is None


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
def test_every_variant_of_a_lower_version_is_legacy(lower, upper):
    """Until AP9-UX2 a variant was a group of its own; since GF F3 it belongs to its version."""
    assert _compute([lower, upper]) == {lower: Release("legacy", upper), upper: Release("latest", None)}


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


def test_an_unknown_size_takes_the_largest_candidate():
    result = _compute(["acme/Nova-2", "acme/Nova-3-7B", "acme/Nova-3-9B"])

    assert result["acme/Nova-2"] == Release("legacy", "acme/Nova-3-9B")


def test_variants_of_equal_standing_still_give_one_successor():
    """Same size, same variation (`instruct` and `chat` make none): the name decides (U5 rule 3)."""
    result = _compute(["acme/Nova-1-7B", "acme/Nova-2-7B-Instruct", "acme/Nova-2-7B-Chat"])

    assert result["acme/Nova-1-7B"] == Release("legacy", "acme/Nova-2-7B-Chat")
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
    """U5 rule 4: a catalog `latest` with nothing above it stays; it is a member all the same."""
    catalog = _catalog(_latest_row("acme/Nova-3-7B"), _row("acme/Nova-2-7B"))
    hits = [_hit("acme/Nova-3-7B", age="latest"), _hit("acme/Nova-2-7B")]
    ages = {"acme/Nova-3-7B": AgeVerdict("latest", None, "catalog"), **_silent("acme/Nova-2-7B")}

    result = compute_release(hits, catalog, ages)

    assert result == {"acme/Nova-2-7B": Release("legacy", "acme/Nova-3-7B")}


def test_a_catalog_row_without_a_hit_is_a_member_and_gets_no_status():
    catalog = _catalog(_row("acme/Nova-3-7B"))

    assert _compute(["acme/Nova-2-7B"], catalog) == {"acme/Nova-2-7B": Release("legacy", "acme/Nova-3-7B")}


def _cut(text: str) -> str:
    """A note as step 2 shows it: at most 97 characters, a cut one ending in `...`."""
    return text if len(text) <= 97 else text[:94] + "..."


FAULTS = [
    AgeVerdict("unknown", None, "budget exhausted before the first edge"),
    AgeVerdict("unknown", None, "new_version 'acme/Gone' is not a reachable repo of acme"),
]


@pytest.mark.parametrize("verdict", FAULTS)
def test_a_fault_is_a_note_and_a_higher_version_still_makes_legacy(verdict):
    """GF F3 (Codex r2-2): a higher version decides the age; the fault stays a note of its own."""
    from modelroom.search_release import release_notes

    ages = {"acme/Nova-2-7B": verdict, **_silent("acme/Nova-3-7B")}
    hits = [_hit("acme/Nova-2-7B"), _hit("acme/Nova-3-7B")]

    result = compute_release(hits, _empty(), ages)

    assert result["acme/Nova-2-7B"] == Release("legacy", "acme/Nova-3-7B")
    assert result["acme/Nova-3-7B"] == Release("latest", None)
    assert release_notes(hits, _empty(), ages) == [_cut(f"Nova-2-7B: its own statement did not read ({verdict.evidence})")]


@pytest.mark.parametrize("verdict", FAULTS)
def test_a_fault_without_a_higher_version_stays_unknown_with_its_note(verdict):
    from modelroom.search_release import release_notes

    ages = {"acme/Nova-3-7B": verdict, **_silent("acme/Nova-2-7B")}
    hits = [_hit("acme/Nova-2-7B"), _hit("acme/Nova-3-7B")]

    assert "acme/Nova-3-7B" not in compute_release(hits, _empty(), ages)
    assert release_notes(hits, _empty(), ages) == [_cut(f"Nova-3-7B: its own statement did not read ({verdict.evidence})")]


def test_a_long_note_is_cut_to_97_characters():
    from modelroom.search_release import release_notes

    long_fault = AgeVerdict("unknown", None, "new_version '" + "acme/" + "x" * 120 + "' is not a reachable repo of acme")
    notes = release_notes([_hit("acme/Nova-2-7B")], _empty(), {"acme/Nova-2-7B": long_fault})

    assert len(notes[0]) == 97 and notes[0].startswith("Nova-2-7B: its own statement did not read (")


def test_a_name_the_grammar_cannot_read_stays_unknown():
    davidau = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"

    assert _compute([davidau]) == {}


def test_a_successor_that_closes_a_cycle_is_skipped():
    """Stated Nova-2-7B -> Luna-1-7B -> Nova-1-7B: the edge Nova-1-7B -> Nova-2-7B would close a
    cycle, so the next candidate by rule 3 is taken -- whatever the order of rows and hits."""
    rows = [
        _row("acme/Nova-2-7B", successor="acme/Luna-1-7B"),
        _row("acme/Luna-1-7B", successor="acme/Nova-1-7B"),
        _row("acme/Nova-2-9B"),
    ]
    bases = ["acme/Nova-1-7B", "acme/Luna-1-3B"]

    for row_order in itertools.permutations(rows):
        for base_order in itertools.permutations(bases):
            result = _compute(list(base_order), _catalog(*row_order))
            assert result["acme/Nova-1-7B"] == Release("legacy", "acme/Nova-2-9B")


def test_when_every_successor_closes_a_cycle_the_model_is_still_legacy_with_a_note():
    from modelroom.search_release import release_notes

    hits = [_hit("acme/Nova-1"), _hit("acme/Nova-2", age="legacy", successor="acme/Nova-1")]
    ages = {**_silent("acme/Nova-1"), "acme/Nova-2": AgeVerdict("legacy", "acme/Nova-1", "new_version at acme/Nova-2")}

    for order in itertools.permutations(hits):
        assert compute_release(list(order), _empty(), ages) == {"acme/Nova-1": Release("legacy", "acme/Nova-2")}
        assert release_notes(list(order), _empty(), ages) == [
            "Nova-1: the successor statements around it contradict each other"
        ]


def test_a_chain_is_allowed():
    catalog = _catalog(_row("acme/Nova-3-7B", successor="acme/Luna-1-7B"), _latest_row("acme/Luna-1-7B"))

    assert _compute(["acme/Nova-2-7B"], catalog) == {"acme/Nova-2-7B": Release("legacy", "acme/Nova-3-7B")}


# --- lines and variations (GF F3, AP9-UX2 U5) -----------------------------------------------------


def test_the_group_has_no_variant():
    assert line_key("Qwen/Qwen3.5-9B-Base", None).group == ("qwen", "qwen", "")
    assert line_key("Qwen/Qwen3.5-9B-Base", None).variant == frozenset({"base"})
    assert line_key("Qwen/Qwen3-Coder-Next", None).group == line_key("Qwen/Qwen3.8-27B", None).group


def test_a_variant_before_the_version_belongs_to_the_family():
    key = line_key("acme/Nova-Mini-2-7B", None)
    assert (key.family, key.line, key.variant, key.version) == ("nova", "", frozenset({"mini"}), (2,))
    assert _compute(["acme/Nova-Mini-2-7B", "acme/Nova-3-7B"])["acme/Nova-Mini-2-7B"] == Release("legacy", "acme/Nova-3-7B")


def test_a_variant_after_the_version_belongs_to_the_family():
    assert _compute(["acme/Nova-2-7B-Flash", "acme/Nova-3-7B"])["acme/Nova-2-7B-Flash"] == Release("legacy", "acme/Nova-3-7B")


def test_an_older_release_of_a_line_is_legacy():
    old = "mistralai/Mistral-Small-Instruct-2409"
    assert _compute([old, MISTRAL_SMALL_3])[old] == Release("legacy", MISTRAL_SMALL_3)


def test_a_line_is_not_superseded_by_another_line():
    large = "mistralai/Mistral-Large-3"
    assert line_key(large, None).group == ("mistralai", "mistral", "large")
    assert _compute([MISTRAL_SMALL_3, large]) == {MISTRAL_SMALL_3: Release("latest", None), large: Release("latest", None)}


def test_a_line_counter_is_a_line_of_its_own():
    r1, others = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", ["deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V4-Flash-0731"]
    assert line_key(r1, None).group == ("deepseek-ai", "deepseek", "r")
    assert line_key("deepseek-ai/DeepSeek-V3.2", None).line == ""  # `v` is the prefix of a version
    assert _compute([r1, *others])[r1] == Release("latest", None)
    assert _compute([r1, *others, "deepseek-ai/DeepSeek-R2"])[r1] == Release("legacy", "deepseek-ai/DeepSeek-R2")


def test_line_words_are_a_documented_vocabulary():
    """New line words only with a GF decision and a test (ADR 0010)."""
    from modelroom.search_release import LINE_COUNTERS, LINE_WORDS

    assert LINE_WORDS == frozenset({"small", "medium", "large"})
    assert LINE_COUNTERS == frozenset({"r"})


def _packager(repo: str, parameters_b: float | None = None) -> SearchHit:
    return SearchHit(
        repo=repo, publisher_status="listed packager", resolved=False, unresolved_reason="relation_unknown",
        parameters_b=parameters_b,
    )


def test_a_listed_packager_repo_proves_a_higher_version():
    hits = [_hit("Qwen/Qwen3.5-9B"), _packager("unsloth/Qwen3.8-27B-GGUF")]

    result = compute_release(hits, _empty(), _silent("Qwen/Qwen3.5-9B"))

    assert result == {"Qwen/Qwen3.5-9B": Release("legacy", "unsloth/Qwen3.8-27B-GGUF")}


def test_a_packager_family_matching_two_publishers_counts_for_none():
    hits = [_hit("acme/Nova-2-7B"), _hit("other-org/Nova-1-7B"), _packager("unsloth/Nova-3-7B-GGUF")]

    result = compute_release(hits, _empty(), _silent("acme/Nova-2-7B", "other-org/Nova-1-7B"))

    assert result == {"acme/Nova-2-7B": Release("latest", None), "other-org/Nova-1-7B": Release("latest", None)}


def test_a_catalog_latest_gives_way_to_a_higher_version():
    catalog = _catalog(_latest_row("acme/Nova-2-7B"), _row("acme/Nova-3-7B"))
    hits = [_hit("acme/Nova-2-7B", age="latest")]
    ages = {"acme/Nova-2-7B": AgeVerdict("latest", None, "catalog")}

    assert compute_release(hits, catalog, ages) == {"acme/Nova-2-7B": Release("legacy", "acme/Nova-3-7B")}
    rebuilt = with_release(hits, catalog, ages)[0]
    assert (rebuilt.age, rebuilt.successor, rebuilt.release_basis) == ("legacy", "acme/Nova-3-7B", "computed")


def test_a_catalog_latest_gives_way_only_to_a_strictly_higher_version():
    """Codex r1 M3 (U5 rule 4): a later date of the same version does not overrule the catalog."""
    catalog = _catalog(_latest_row("acme/Nova-2-7B-2409"), _row("acme/Nova-2-7B-2506"))
    hits = [_hit("acme/Nova-2-7B-2409", age="latest")]
    ages = {"acme/Nova-2-7B-2409": AgeVerdict("latest", None, "catalog")}

    assert compute_release(hits, catalog, ages) == {}
    higher = _catalog(_latest_row("acme/Nova-2-7B-2409"), _row("acme/Nova-2-7B-2506"), _row("acme/Nova-3-7B"))
    assert compute_release(hits, higher, ages) == {"acme/Nova-2-7B-2409": Release("legacy", "acme/Nova-3-7B")}
    # A computed status still reads the date: a silent catalog's 2409 is `legacy` to 2506.
    assert _compute(["acme/Nova-2-7B-2409", "acme/Nova-2-7B-2506"])["acme/Nova-2-7B-2409"] == Release("legacy", "acme/Nova-2-7B-2506")


def test_a_stated_legacy_is_never_recomputed():
    hits = [_hit("acme/Nova-2-7B", age="legacy", successor="acme/Nova-2.5-7B"), _hit("acme/Nova-3-7B")]
    ages = {
        "acme/Nova-2-7B": AgeVerdict("legacy", "acme/Nova-2.5-7B", "new_version at acme/Nova-2-7B"),
        **_silent("acme/Nova-3-7B"),
    }

    assert "acme/Nova-2-7B" not in compute_release(hits, _empty(), ages)


def test_a_faulty_unknown_stays_unknown():
    """A fault's `unknown` never becomes a computed `latest`, not even as the only member."""
    ages = {"acme/Nova-3-7B": FAULTS[0]}

    assert compute_release([_hit("acme/Nova-3-7B")], _empty(), ages) == {}


def test_qwen35_0_8b_base_is_legacy():
    bases = ["Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-9B"]
    hits = [*(_hit(base) for base in bases), _hit("Qwen/Qwen3.8-27B", age="latest")]
    ages = {**_silent(*bases), "Qwen/Qwen3.8-27B": AgeVerdict("latest", None, "catalog")}

    result = compute_release(hits, _empty(), ages)

    assert result == {base: Release("legacy", "Qwen/Qwen3.8-27B") for base in bases}


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


# --- AP9-KA-G: the grammar reads more catalog names -------------------------------------------------


@pytest.mark.parametrize(
    "repo, size, version, variant",
    [
        ("acme/Nova-7B-8K", 7, (), set()),  # `8K` is a k attribute, no size and no boundary
        ("acme/Nova-1080p", None, (), set()),  # a resolution is ignored
        ("acme/Nova-1p3b-Qwen2", None, (), {"qwen2"}),  # the `p` size sets the boundary: `qwen2` stays a variation
        ("acme/Nova_2_5-7B", 7, (2, 5), set()),
        ("acme/Nova-7B-1M", 7, (), set()),  # `1M` stays a size
        ("acme/Nova-R1", None, (1,), set()),  # `r1` stays a line counter
    ],
)
def test_ka_g_edge_cases(repo, size, version, variant):
    key = line_key(repo, None)

    assert key is not None
    assert (key.version, set(key.variant)) == (version, variant)
    assert key.size == (pytest.approx(size) if size is not None else None)


def test_ka_g_a_one_m_stays_a_size_and_a_resolution_is_not_a_size():
    assert line_key("acme/Nova-7B-1M", None).line == ""
    assert line_key("acme/Nova-R1", None).line == "r"
    assert line_key("acme/Nova-1080p", None).family == "nova"


@pytest.mark.parametrize(
    "repo, family, version, variant",
    [
        ("acme/Nova.2-7B", "nova", (2,), set()),  # at the boundary the dotted word gives its word and its version
        ("acme/Nova-7B-Qwen.3", "nova", (), {"qwen.3"}),  # behind the boundary it is a variation, whole
    ],
)
def test_ka_g_a_dotted_word(repo, family, version, variant):
    key = line_key(repo, None)

    assert key is not None
    assert (key.family, key.version, set(key.variant)) == (family, version, variant)


def test_ka_g_a_dotted_word_with_a_number_too_long_is_no_reading_and_no_crash():
    assert line_key("acme/Nova." + "9" * 5000, None) is None


@pytest.mark.parametrize(
    "repo, version, dated, variant",
    [
        ("apple/OpenELM-1_1B-Instruct", (1,), None, set()),
        ("acme/Nova-08_2025", (), LineDate("mm-yyyy", 2025, 8, None), set()),
        ("acme/Nova-BF16_8B", (), None, set()),
        ("acme/Nova-7B-Qwen3_5", (5,), None, {"qwen3"}),
        ("acme/Nova-2_08-2025", (2,), LineDate("mm-yyyy", 2025, 8, None), set()),
        ("acme/Nova-08-2025-Qwen3_5", (5,), LineDate("mm-yyyy", 2025, 8, None), {"qwen3"}),
    ],
)
def test_a_reading_that_stood_stays(repo, version, dated, variant):
    """The underscore between digits changes no reading that stood before (AP9-KA-G, rule N)."""
    key = line_key(repo, None)

    assert key is not None
    assert (key.version, key.date, set(key.variant)) == (version, dated, variant)


def test_a_reading_that_stood_stays_with_its_size():
    assert line_key("apple/OpenELM-1_1B-Instruct", None).size == 1
    assert line_key("acme/Nova-BF16_8B", None).size == 8


# --- the comparison over the real catalog and over fixtures (compute_release) ------------------------


def _publisher_hit(repo: str, parameters_b: float | None = None) -> SearchHit:
    return SearchHit(
        repo=repo,
        publisher_status="publisher",
        base_model=[repo],
        base_model_relation="quantized",
        resolved=True,
        resolved_base_model=repo,
        age="unknown",
        parameters_b=parameters_b,
    )


def _compare(catalog: Catalog, repos: list[str], sizes: dict[str, float] | None = None):
    hits = [_publisher_hit(repo, (sizes or {}).get(repo)) for repo in repos]
    ages = {repo: AgeVerdict(*catalog.age_of(repo), "catalog") for repo in repos}
    return compute_release(hits, catalog, ages)


WAN = {
    "Wan-AI/Wan2.1-I2V-14B-480P": 16.394878784,
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": 1.4189968,
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": 14.288901184,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": 14.288491584,
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": 4.999787712,
}
FLUX = {
    "black-forest-labs/FLUX.1-Kontext-dev": 11.90140832,
    "black-forest-labs/FLUX.1-dev": 11.90140832,
    "black-forest-labs/FLUX.1-schnell": 11.89117856,
    "black-forest-labs/FLUX.2-dev": 32.223281152,
    "black-forest-labs/FLUX.2-klein-4B": 3.875544576,
}


def _fixture_catalog(rows: dict[str, float], publisher: str) -> Catalog:
    return _catalog(*(_row(repo) for repo in rows), publisher=publisher)


def test_ka_g_flux_one_is_legacy_to_a_flux_two_by_the_size_of_the_hit():
    catalog = _fixture_catalog(FLUX, "black-forest-labs")
    repos = list(FLUX)

    by_name = _compare(catalog, repos)
    by_hit = _compare(catalog, repos, FLUX)

    for result, successor in ((by_name, "black-forest-labs/FLUX.2-klein-4B"), (by_hit, "black-forest-labs/FLUX.2-dev")):
        assert result["black-forest-labs/FLUX.1-dev"] == Release("legacy", successor)
        assert result["black-forest-labs/FLUX.1-Kontext-dev"].age == "legacy"
        assert result["black-forest-labs/FLUX.1-schnell"].age == "legacy"
        assert result["black-forest-labs/FLUX.2-dev"] == Release("latest", None)
        assert result["black-forest-labs/FLUX.2-klein-4B"] == Release("latest", None)


def test_ka_g_wan_two_one_is_legacy_to_a_wan_two_two_by_the_size_of_the_hit():
    catalog = _fixture_catalog(WAN, "Wan-AI")
    repos = list(WAN)

    by_name = _compare(catalog, repos)
    by_hit = _compare(catalog, repos, WAN)

    assert by_name["Wan-AI/Wan2.1-I2V-14B-480P"] == Release("legacy", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert by_hit["Wan-AI/Wan2.1-I2V-14B-480P"] == Release("legacy", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")


def test_ka_g_internvl_two_five_is_legacy_to_internvl_three_and_no_three_is_legacy():
    catalog = load_catalog()
    repos = [m.hf_repo for f in catalog.families for m in f.models if m.hf_repo.lower().startswith("opengvlab/")]
    assert "OpenGVLab/InternVL2_5-4B-MPO" in repos

    result = _compare(catalog, repos)

    assert result["OpenGVLab/InternVL2_5-4B-MPO"] == Release("legacy", "OpenGVLab/InternVL3-8B")
    assert [r for r in repos if "InternVL3" in r and result.get(r, Release("latest", None)).age == "legacy"] == []


def test_ka_g_yi_200k_is_legacy_to_yi_one_five():
    catalog = load_catalog()
    repos = [m.hf_repo for f in catalog.families for m in f.models if m.hf_repo.lower().startswith("01-ai/")]
    assert "01-ai/Yi-6B-200K" in repos

    result = _compare(catalog, repos)

    assert result["01-ai/Yi-6B-200K"] == Release("legacy", "01-ai/Yi-1.5-9B-Chat")
