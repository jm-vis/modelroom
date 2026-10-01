"""Tests for modelroom.ranking: the ranking rule is a total order, independent of input order."""

from __future__ import annotations

import copy
import itertools
import random
from datetime import datetime, timedelta, timezone

import pytest

from modelroom.contracts import Fit, Package
from modelroom.examples import EXAMPLES
from modelroom.measurements import MeasurementRecord, default_scenario, new_measurement_id
from modelroom.quantization import QUANT_ORDER
from modelroom.ranking import (
    TOP_LIMIT,
    ModelFacts,
    best_per_model,
    group_zero_measurement,
    rank_packages,
    row_sentence,
    shown_rows,
)

REVISION = "9f8e7d6c5b4a3928170615243342516078960a1b"
BASE_TIME = datetime(2026, 9, 23, 8, 30, 0, tzinfo=timezone.utc)


def _digest(n: int) -> str:
    return "sha256:" + f"{n:02x}" * 32


def _package(
    name: str, quant: str | None, size: int, n: int, repo: str = "packager/Nova-7B-GGUF", base: str = "acme/Nova-7B"
) -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["repo"] = repo
    payload["base_model_hf_repo"] = base
    payload["files"] = [{"name": f"{repo.split('/')[1]}-{name}.gguf", "role": "weights", "size_bytes": size, "digest": _digest(n)}]
    payload["quantization"] = quant
    return Package.model_validate(payload)


def _fit(fit_class: str, context: int = 8192, mode: str = "gpu") -> Fit:
    payload = copy.deepcopy(EXAMPLES["Fit"])
    payload.update(fit_class=fit_class, context=context, context_assumed=False, mode=mode)
    if fit_class == "unknown":
        payload.update(mode=None, need_gib=0.0, weights_gib=0.0, kv_gib=0.0, pool_gib=0.0, reserve_gib=0.0, context=0, reason="architecture not covered by v1")
    return Fit.model_validate(payload)


def _measurement(package: Package, tps: float, minutes: int = 0, **changes) -> MeasurementRecord:
    measured_at = BASE_TIME + timedelta(minutes=minutes)
    run = {**EXAMPLES["RunCounters"], "eval_duration": int(128 / tps * 1e9)}
    runs = [run, dict(run), dict(run)]
    speed = 128 / (run["eval_duration"] / 1e9)
    payload = copy.deepcopy(EXAMPLES["MeasurementRecord"])
    payload.update(
        measurement_id=new_measurement_id(measured_at, f"{minutes:08x}"),
        measured_at=measured_at.isoformat().replace("+00:00", "Z"),
        package={
            "content_source": "huggingface",
            "ollama_manifest_digest": None,
            "hf_repo": package.repo,
            "hf_revision": package.revision,
            "hf_file_digest": package.files[0].digest,
        },
        ollama_name=None,
        runs=runs,
        tps_mean=speed,
        tps_min=speed,
        tps_max=speed,
    )
    payload.update(changes)
    return MeasurementRecord.model_validate(payload)


Q4 = _package("Q4_K_M", "Q4_K_M", 4_500_000_000, 1)
Q5 = _package("Q5_K_M", "Q5_K_M", 5_200_000_000, 2)
Q6 = _package("Q6_K", "Q6_K", 6_000_000_000, 3)
Q8 = _package("Q8_0", "Q8_0", 7_700_000_000, 4)
Q4_OTHER = _package("Q4_K_M", "Q4_K_M", 4_500_000_000, 5, repo="other/Nova-7B-GGUF")
F16 = _package("F16", "F16", 14_000_000_000, 6)


def _entries() -> list[tuple[Package, Fit]]:
    return [
        (Q4, _fit("ample")),
        (Q5, _fit("ample")),
        (Q6, _fit("fits")),
        (Q8, _fit("tight")),
        (Q4_OTHER, _fit("ample")),
        (F16, _fit("no_fit")),
    ]


def _measurements() -> list[MeasurementRecord]:
    return [_measurement(Q5, 40.0), _measurement(Q6, 90.0, minutes=1)]


def _order(ranking) -> list[tuple[str, str, str]]:
    return [(r.package.repo, r.package.files[0].name, r.fit.fit_class) for r in ranking.ranked]


def test_the_least_compression_ranks_first_whatever_the_fit_class_and_the_speed():
    """ADR 0009: `ample`, `fits` and `tight` rank alike, and a measurement only breaks a tie."""
    ranking = rank_packages(_entries(), _measurements(), default_scenario())
    names = [(r.package.repo.split("/")[0], r.package.quantization, r.measurement_group) for r in ranking.ranked]
    assert names == [
        ("packager", "Q8_0", 1),  # tight, and still the least compression that fits
        ("packager", "Q6_K", 0),
        ("packager", "Q5_K_M", 0),
        ("other", "Q4_K_M", 1),  # same quant as the next, identity decides
        ("packager", "Q4_K_M", 1),
    ]
    assert [r.rank for r in ranking.ranked] == [1, 2, 3, 4, 5]
    assert [s.package.quantization for s in ranking.no_fit] == ["F16"]


def test_ranking_is_the_same_for_every_input_order():
    expected = _order(rank_packages(_entries(), _measurements(), default_scenario()))
    for permutation in itertools.permutations(_entries()):
        for measurements in (_measurements(), list(reversed(_measurements()))):
            assert _order(rank_packages(list(permutation), measurements, default_scenario())) == expected


def test_a_faster_measurement_does_not_lift_a_more_compressed_package():
    entries = [(Q4, _fit("ample")), (Q5, _fit("ample"))]
    ranking = rank_packages(entries, [_measurement(Q4, 60.0), _measurement(Q5, 40.0, minutes=1)], default_scenario())
    assert [r.package.quantization for r in ranking.ranked] == ["Q5_K_M", "Q4_K_M"]


def test_larger_weights_break_a_tie_on_quantization():
    small = _package("Q4_K_M", "Q4_K_M", 4_000_000_000, 7, repo="aaa/Nova-7B-GGUF")
    large = _package("Q4_K_M", "Q4_K_M", 4_600_000_000, 8, repo="zzz/Nova-7B-GGUF")
    ranking = rank_packages([(small, _fit("fits")), (large, _fit("fits"))], [], default_scenario())
    assert [r.package.repo for r in ranking.ranked] == ["zzz/Nova-7B-GGUF", "aaa/Nova-7B-GGUF"]


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol": "none", "runs": [], "validity": "unchecked", "validity_reason": "old", "comparable": False, "comparable_reason": "old"},
        {"comparable": False, "comparable_reason": "daemon digest changed"},
        {"scenario": {**EXAMPLES["Scenario"], "context_origin": "entered", "context_requested": 4096}},
    ],
)
def test_only_valid_comparable_same_context_measurements_count(changes):
    record = _measurement(Q5, 40.0, **changes)
    assert group_zero_measurement(Q5, [record], default_scenario()) is None


def test_invalid_measurement_does_not_count():
    run = {**EXAMPLES["RunCounters"], "done_reason": "stop", "eval_duration": 3_200_000_000}  # 40 tps
    record = _measurement(Q5, 40.0, runs=[run, run, run], validity="invalid", validity_reason="run 1: done_reason 'stop'")
    assert group_zero_measurement(Q5, [record], default_scenario()) is None


def test_measurement_of_other_content_does_not_count():
    assert group_zero_measurement(Q4_OTHER, [_measurement(Q4, 50.0)], default_scenario()) is None


def test_newest_counting_measurement_wins():
    older, newer = _measurement(Q5, 40.0), _measurement(Q5, 30.0, minutes=5)
    assert group_zero_measurement(Q5, [newer, older], default_scenario()) == newer


def test_the_chosen_measurement_is_the_same_for_every_input_order():
    tied_stamp = new_measurement_id(BASE_TIME + timedelta(minutes=5), "ffffffff")
    records = [
        _measurement(Q5, 40.0),
        _measurement(Q5, 30.0, minutes=5),  # measurement_id token 00000005
        _measurement(Q5, 20.0, minutes=5, measurement_id=tied_stamp),  # same measured_at, larger id
        _measurement(Q5, 50.0, minutes=2),
    ]
    for permutation in itertools.permutations(records):
        chosen = group_zero_measurement(Q5, list(permutation), default_scenario())
        assert chosen.measurement_id == tied_stamp


def test_ollama_measurement_matches_by_manifest_digest():
    ollama = Package.model_validate(
        {
            **copy.deepcopy(EXAMPLES["Package"]),
            "source": "ollama",
            "repo": None,
            "revision": None,
            "ollama_name": "nova:7b",
            "manifest_digest": EXAMPLES["MeasurementRecord"]["package"]["ollama_manifest_digest"],
        }
    )
    record = MeasurementRecord.model_validate(EXAMPLES["MeasurementRecord"])
    assert group_zero_measurement(ollama, [record], default_scenario()) == record


def test_unknown_fits_are_set_aside_with_their_reason():
    ranking = rank_packages([(Q4, _fit("unknown")), (Q5, _fit("fits"))], [], default_scenario())
    assert [r.package.quantization for r in ranking.ranked] == ["Q5_K_M"]
    assert ranking.not_covered[0].reason == "architecture not covered by v1"


def test_a_fit_for_another_context_is_a_caller_error():
    with pytest.raises(ValueError, match="context"):
        rank_packages([(Q4, _fit("fits", context=4096))], [], default_scenario())


def test_top_is_the_first_ten():
    packages = [_package(f"Q4_K_M-{i}", "Q4_K_M", 4_000_000_000 + i, 10 + i) for i in range(12)]
    ranking = rank_packages([(p, _fit("fits")) for p in packages], [], default_scenario())
    assert len(ranking.ranked) == 12
    assert ranking.top == ranking.ranked[:TOP_LIMIT] and len(ranking.top) == 10


def test_rank_packages_takes_the_facts_of_the_models_by_keyword():
    ranking = rank_packages([(Q4, _fit("fits"))], [], default_scenario(), models={"acme/Nova-7B": ModelFacts(7.0, False)})
    assert [r.package for r in ranking.ranked] == [Q4]


# --- the rule of ADR 0009: the largest model that fits, the least compression of it ---------------

NOVA_BIG = "acme/Nova-9B"  # 9.65 B
NOVA_MID = "acme/Nova-9B-Flash"  # 9.15 B
NOVA_SMALL = "acme/Nova-8B"  # 8.19 B
NOVA_TINY = "acme/Nova-4B"  # 4.66 B
FACTS = {
    NOVA_BIG: ModelFacts(9.65, False),
    NOVA_MID: ModelFacts(9.15, False),
    NOVA_SMALL: ModelFacts(8.19, False),
    NOVA_TINY: ModelFacts(4.66, False),
}


def _of(base: str, quant: str | None, n: int, size: int = 5_000_000_000, packager: str = "packager") -> Package:
    return _package(quant or "tensor", quant, size, n, repo=f"{packager}/{base.split('/')[1]}-GGUF", base=base)


def _bases(ranking) -> list[str]:
    return [r.package.base_model_hf_repo for r in ranking.ranked]


def _quants(rows) -> list[str | None]:
    return [row.package.quantization for row in rows]


def test_the_largest_model_that_fits_ranks_first():
    big, small = _of(NOVA_BIG, "Q4_K_M", 101), _of(NOVA_SMALL, "Q4_K_M", 102)
    ranking = rank_packages([(small, _fit("ample")), (big, _fit("tight"))], [], default_scenario(), models=FACTS)
    assert _bases(ranking) == [NOVA_BIG, NOVA_SMALL]


def test_within_one_model_the_least_compression_that_fits_ranks_first():
    packages = [_of(NOVA_BIG, quant, 110 + i) for i, quant in enumerate(("IQ1_S", "Q8_0", "Q4_K_M"))]
    ranking = rank_packages([(p, _fit("ample")) for p in packages], [], default_scenario(), models=FACTS)
    assert _quants(ranking.ranked) == ["Q8_0", "Q4_K_M", "IQ1_S"]


def test_the_16_and_32_bit_builds_are_one_step_with_the_smaller_first():
    packages = [_of(NOVA_BIG, "F32", 120, 30_000_000_000), _of(NOVA_BIG, "BF16", 121, 15_000_000_000), _of(NOVA_BIG, "Q8_0", 122)]
    ranking = rank_packages([(p, _fit("fits")) for p in packages], [], default_scenario(), models=FACTS)
    assert _quants(ranking.ranked) == ["BF16", "F32", "Q8_0"]


def test_an_unknown_quantization_ranks_after_every_known_one_of_its_model():
    packages = [_of(NOVA_BIG, None, 130, 20_000_000_000), _of(NOVA_BIG, "IQ1_S", 131)]
    ranking = rank_packages([(p, _fit("fits")) for p in packages], [], default_scenario(), models=FACTS)
    assert _quants(ranking.ranked) == ["IQ1_S", None]


def test_an_unknown_parameter_count_ranks_after_every_known_one():
    unknown, tiny = _of("acme/Nova-Mystery", "Q8_0", 140), _of(NOVA_TINY, "IQ1_S", 141)
    ranking = rank_packages([(unknown, _fit("ample")), (tiny, _fit("ample"))], [], default_scenario(), models=FACTS)
    assert _bases(ranking) == [NOVA_TINY, "acme/Nova-Mystery"]


def test_speed_no_longer_reorders_packages_of_one_quantization_step():
    """Two `Q4_K_M` builds of one model: a measurement puts its package first, its speed does not."""
    first = _of(NOVA_BIG, "Q4_K_M", 150, packager="aaa")
    second = _of(NOVA_BIG, "Q4_K_M", 151, packager="zzz")
    both = [_measurement(first, 20.0), _measurement(second, 90.0, minutes=1)]
    ranking = rank_packages([(second, _fit("fits")), (first, _fit("fits"))], both, default_scenario(), models=FACTS)
    assert [r.package.repo.split("/")[0] for r in ranking.ranked] == ["aaa", "zzz"]
    only_second = rank_packages([(first, _fit("fits")), (second, _fit("fits"))], both[1:], default_scenario(), models=FACTS)
    assert [r.package.repo.split("/")[0] for r in only_second.ranked] == ["zzz", "aaa"]


def test_graphics_memory_ranks_before_system_memory():
    on_card = _of(NOVA_TINY, "Q4_K_M", 160)
    in_ram_big = _of(NOVA_BIG, "Q8_0", 161)
    in_ram_same = _of(NOVA_TINY, "Q8_0", 162)
    entries = [(in_ram_big, _fit("fits", mode="cpu_gpu")), (in_ram_same, _fit("fits", mode="cpu")), (on_card, _fit("tight"))]
    ranking = rank_packages(entries, [], default_scenario(), models=FACTS)
    assert [(r.package.base_model_hf_repo, r.package.quantization) for r in ranking.ranked] == [
        (NOVA_TINY, "Q4_K_M"),  # on the card: first, although smaller and more compressed
        (NOVA_BIG, "Q8_0"),
        (NOVA_TINY, "Q8_0"),
    ]


def test_a_smaller_model_ranks_before_a_larger_legacy_one():
    big, small = _of(NOVA_BIG, "Q8_0", 170), _of(NOVA_SMALL, "Q4_K_M", 171)
    facts = {**FACTS, NOVA_BIG: ModelFacts(9.65, True)}
    ranking = rank_packages([(big, _fit("ample")), (small, _fit("ample"))], [], default_scenario(), models=facts)
    assert _bases(ranking) == [NOVA_SMALL, NOVA_BIG]


def test_latest_and_unknown_age_rank_by_size_alone():
    """`latest` stands only before `legacy` (GF 2026-09-28): a 7B `latest` behind a 9B `–`."""
    latest_7b, unknown_9b = _of("acme/Nova-7B", "Q8_0", 180), _of(NOVA_BIG, "Q4_K_M", 181)
    facts = {"acme/Nova-7B": ModelFacts(7.0, False), NOVA_BIG: ModelFacts(9.65, False)}
    ranking = rank_packages([(latest_7b, _fit("ample")), (unknown_9b, _fit("ample"))], [], default_scenario(), models=facts)
    assert _bases(ranking) == [NOVA_BIG, "acme/Nova-7B"]


def _vllm_package(quant: str, n: int, *, base: str = NOVA_BIG, size: int = 5_000_000_000) -> Package:
    """A Hugging Face tensor package (AP9-SV2-F): `_package` builds GGUF; this is vLLM's own shape."""
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload.update(
        repo=f"RedHatAI/{base.split('/')[1]}-{quant}",
        base_model_hf_repo=base,
        format="tensor",
        files=[{"name": "model.safetensors", "role": "weights", "size_bytes": size, "digest": _digest(n)}],
        quantization=quant,
        quantization_source="config",
    )
    return Package.model_validate(payload)


def _vllm_fit(fit_class: str) -> Fit:
    payload = copy.deepcopy(EXAMPLES["Fit"])
    payload.update(fit_class=fit_class, context=8192, context_assumed=False, mode="gpu_split", runtime="vllm", cards=4, at_once=20)
    return Fit.model_validate(payload)


def test_vllm_rows_rank_by_adr_0009_between_gguf_rows():
    """AP9-SV2-F: a vLLM row's `Fit.runtime` plays no part in ADR 0009's rule -- graphics memory,
    size, then the least compression that fits orders a Hugging Face tensor row exactly like a
    GGUF one of the same model: FP8 (least compressed of the three) before AWQ before Q4_K_M."""
    fp8 = _vllm_package("FP8", 500)
    awq = _vllm_package("AWQ", 501)
    q4 = _of(NOVA_BIG, "Q4_K_M", 502)  # a GGUF row of the very same base model

    ranking = rank_packages(
        [(fp8, _vllm_fit("ample")), (awq, _vllm_fit("ample")), (q4, _fit("ample"))], [], default_scenario(), models=FACTS
    )

    assert [(r.package.format, r.package.quantization, r.fit.runtime) for r in ranking.ranked] == [
        ("tensor", "FP8", "vllm"),
        ("tensor", "AWQ", "vllm"),
        ("gguf", "Q4_K_M", "ollama"),
    ]


def test_the_runtime_only_breaks_a_tie_of_an_otherwise_identical_package_identity():
    """`ranking.py`'s last key, extended for AP9-SV2-F: two rows of the very same package identity
    differ only by which runtime's row they are -- the runtime key alone decides their order, a
    case today's one-runtime-per-format world never produces outside this direct test."""
    from modelroom.ranking import _sort_key

    package = q4 = _of(NOVA_BIG, "Q4_K_M", 510)
    facts = FACTS[NOVA_BIG]
    ollama_key = _sort_key(package, _fit("ample"), None, facts)
    vllm_key = _sort_key(package, _vllm_fit("ample"), None, facts)

    assert ollama_key[:-1] == vllm_key[:-1]  # every earlier step of the rule agrees
    assert ollama_key[-1] < vllm_key[-1]  # "ollama" sorts before "vllm", alphabetically, as a tie-break only


def test_the_best_package_of_a_model_is_the_first_under_the_whole_key():
    on_card, in_ram = _of(NOVA_BIG, "Q4_K_M", 190), _of(NOVA_BIG, "Q8_0", 191)
    other = _of(NOVA_SMALL, "Q8_0", 192)
    entries = [(in_ram, _fit("fits", mode="cpu_gpu")), (on_card, _fit("tight")), (other, _fit("ample"))]
    best = best_per_model(rank_packages(entries, [], default_scenario(), models=FACTS))
    assert [(r.package.base_model_hf_repo, r.package.quantization) for r in best] == [
        (NOVA_BIG, "Q4_K_M"),
        (NOVA_SMALL, "Q8_0"),
    ]


def _four_models() -> list[tuple[Package, Fit]]:
    entries = []
    for number, base in enumerate((NOVA_TINY, NOVA_SMALL, NOVA_MID, NOVA_BIG)):
        for step, quant in enumerate(("Q4_K_M", "Q6_K", "Q8_0")):
            entries.append((_of(base, quant, 200 + number * 10 + step), _fit("fits")))
    return entries


def test_several_models_give_one_row_each_with_its_best_package():
    ranking = rank_packages(_four_models(), [], default_scenario(), models=FACTS)
    rows = shown_rows(ranking, per_model=True)
    assert [(row.ranked.package.base_model_hf_repo, row.ranked.package.quantization, row.why) for row in rows] == [
        (NOVA_BIG, "Q8_0", "rule"),
        (NOVA_MID, "Q8_0", "rule"),
        (NOVA_SMALL, "Q8_0", "rule"),
        (NOVA_TINY, "Q8_0", "rule"),
    ]
    assert [(row.place, row.of) for row in rows] == [(1, 3)] * 4
    legacy = rank_packages(_four_models(), [], default_scenario(), models={**FACTS, NOVA_BIG: ModelFacts(9.65, True)})
    assert [row.ranked.package.base_model_hf_repo for row in shown_rows(legacy, per_model=True)] == [
        NOVA_MID,
        NOVA_SMALL,
        NOVA_TINY,
        NOVA_BIG,
    ]


def _fifteen_packages() -> list[Package]:
    return [_of(NOVA_BIG, quant, 60 + i, size=4_000_000_000 + i) for i, quant in enumerate(QUANT_ORDER[5:20])]


def test_a_measured_package_outside_the_rule_is_appended_with_its_place():
    packages = _fifteen_packages()
    ranking = rank_packages([(p, _fit("fits")) for p in packages], [], default_scenario(), models=FACTS)
    fourteenth = ranking.ranked[13].package
    measured = rank_packages(
        [(p, _fit("fits")) for p in packages], [_measurement(fourteenth, 42.0)], default_scenario(), models=FACTS
    )
    rows = shown_rows(measured, per_model=False)
    assert [row.why for row in rows] == ["rule"] * TOP_LIMIT + ["measured"]
    assert [row.ranked.package for row in rows[:TOP_LIMIT]] == [r.package for r in ranking.ranked[:TOP_LIMIT]]
    assert (rows[-1].ranked.package, rows[-1].place, rows[-1].of) == (fourteenth, 14, 15)
    assert row_sentence(rows[-1], per_model=False) == (
        "Shown for its measurement; the rule puts it at place 14.",
        "ranking.shown_for_measurement",
    )


def test_a_package_measured_at_another_context_stays_visible_without_a_comparable_speed():
    """Codex F3: a measurement of the package that does not count -- here another context -- still
    keeps the package in sight; group 1, no speed, and the row says why the number does not count."""
    from modelroom.render import _ranked_entries

    packages = _fifteen_packages()
    ranking = rank_packages([(p, _fit("fits")) for p in packages], [], default_scenario(), models=FACTS)
    fourteenth = ranking.ranked[13].package
    other_context = {"scenario": {**EXAMPLES["Scenario"], "context_origin": "entered", "context_requested": 4096}}
    measured = rank_packages(
        [(p, _fit("fits")) for p in packages], [_measurement(fourteenth, 42.0, **other_context)], default_scenario(), models=FACTS
    )

    rows = shown_rows(measured, per_model=False)
    assert [row.why for row in rows] == ["rule"] * TOP_LIMIT + ["measured"]
    appended = rows[-1].ranked
    assert (appended.package, appended.measurement_group, appended.measurement) == (fourteenth, 1, None)
    assert appended.measured_other == "measured at context 4k, this ranking's is 8k"
    entry = _ranked_entries(measured)[-1]
    assert (entry.speed_tps, entry.note.code) == (None, "measured_not_comparable")
    assert entry.note.text.startswith("measured at context 4k, this ranking's is 8k.")
    assert "Shown for its measurement; the rule puts it at place 14." in entry.note.text


def test_a_measured_package_of_a_model_stays_visible_in_one_row_per_model():
    entries = _four_models()
    measured_q4 = next(p for p, _fit_ in entries if p.base_model_hf_repo == NOVA_SMALL and p.quantization == "Q4_K_M")
    ranking = rank_packages(entries, [_measurement(measured_q4, 33.0)], default_scenario(), models=FACTS)
    rows = shown_rows(ranking, per_model=True)
    assert [row.why for row in rows] == ["rule"] * 4 + ["measured"]
    appended = rows[-1]
    assert (appended.ranked.package, appended.ranked.measurement_group, appended.place, appended.of) == (measured_q4, 0, 3, 3)
    assert row_sentence(appended, per_model=True)[0] == "Shown for its measurement; the rule puts it at place 3 of this model's 3."
    assert row_sentence(rows[0], per_model=True) == ("Best of 3 packages of this model that fit.", "ranking.best_per_model")


def test_the_only_package_of_a_model_says_so():
    ranking = rank_packages([(_of(NOVA_BIG, "Q4_K_M", 90), _fit("fits")), (_of(NOVA_TINY, "Q4_K_M", 91), _fit("fits"))], [], default_scenario(), models=FACTS)
    rows = shown_rows(ranking, per_model=True)
    assert [row_sentence(row, per_model=True)[0] for row in rows] == ["The only package of this model that fits."] * 2
    assert row_sentence(shown_rows(ranking, per_model=False)[0], per_model=False) is None


def test_rank_counts_the_rows_shown():
    """`rank` is the number of the row on display, whatever place the rule gives the package."""
    from modelroom.render import _ranked_entries

    entries = _four_models()
    measured_q4 = next(p for p, _fit_ in entries if p.base_model_hf_repo == NOVA_TINY and p.quantization == "Q4_K_M")
    ranking = rank_packages(entries, [_measurement(measured_q4, 33.0)], default_scenario(), models=FACTS)
    shown = _ranked_entries(ranking, per_model=True)
    assert [entry.rank for entry in shown] == [1, 2, 3, 4, 5]
    assert shown[-1].quantization == "Q4_K_M" and shown[-1].measurement_group == 0
    assert "ranking.shown_for_measurement" in shown[-1].note.facts


def test_shuffled_large_input_gives_one_order():
    packages = [_package(f"Q4_K_M-{i}", "Q4_K_M", 4_000_000_000 + (i % 3), 30 + i) for i in range(20)]
    entries = [(p, _fit(("ample", "fits", "tight")[i % 3])) for i, p in enumerate(packages)]
    expected = _order(rank_packages(entries, [], default_scenario()))
    rng = random.Random(7)
    for _ in range(50):
        rng.shuffle(entries)
        assert _order(rank_packages(entries, [], default_scenario())) == expected
