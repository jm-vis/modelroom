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
from modelroom.ranking import TOP_LIMIT, group_zero_measurement, rank_packages

REVISION = "9f8e7d6c5b4a3928170615243342516078960a1b"
BASE_TIME = datetime(2026, 9, 23, 8, 30, 0, tzinfo=timezone.utc)


def _digest(n: int) -> str:
    return "sha256:" + f"{n:02x}" * 32


def _package(name: str, quant: str, size: int, n: int, repo: str = "packager/Nova-7B-GGUF") -> Package:
    payload = copy.deepcopy(EXAMPLES["Package"])
    payload["repo"] = repo
    payload["files"] = [{"name": f"Nova-7B-{name}.gguf", "role": "weights", "size_bytes": size, "digest": _digest(n)}]
    payload["quantization"] = quant
    return Package.model_validate(payload)


def _fit(fit_class: str, context: int = 8192) -> Fit:
    payload = copy.deepcopy(EXAMPLES["Fit"])
    payload.update(fit_class=fit_class, context=context, context_assumed=False)
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
        (Q4, _fit("perfect")),
        (Q5, _fit("perfect")),
        (Q6, _fit("good")),
        (Q8, _fit("marginal")),
        (Q4_OTHER, _fit("perfect")),
        (F16, _fit("too_tight")),
    ]


def _measurements() -> list[MeasurementRecord]:
    return [_measurement(Q5, 40.0), _measurement(Q6, 90.0, minutes=1)]


def _order(ranking) -> list[tuple[str, str, str]]:
    return [(r.package.repo, r.package.files[0].name, r.fit.fit_class) for r in ranking.ranked]


def test_fit_class_before_speed_before_quantization():
    ranking = rank_packages(_entries(), _measurements(), default_scenario())
    names = [(r.package.repo.split("/")[0], r.package.quantization, r.measurement_group) for r in ranking.ranked]
    assert names == [
        ("packager", "Q5_K_M", 0),  # perfect, measured
        ("other", "Q4_K_M", 1),  # perfect, unmeasured: same quant as the next, identity decides
        ("packager", "Q4_K_M", 1),
        ("packager", "Q6_K", 0),  # good, although it is the fastest measured package
        ("packager", "Q8_0", 1),  # marginal
    ]
    assert [r.rank for r in ranking.ranked] == [1, 2, 3, 4, 5]
    assert [s.package.quantization for s in ranking.too_tight] == ["F16"]


def test_ranking_is_the_same_for_every_input_order():
    expected = _order(rank_packages(_entries(), _measurements(), default_scenario()))
    for permutation in itertools.permutations(_entries()):
        for measurements in (_measurements(), list(reversed(_measurements()))):
            assert _order(rank_packages(list(permutation), measurements, default_scenario())) == expected


def test_faster_measured_package_ranks_first_within_its_class_and_group():
    entries = [(Q4, _fit("perfect")), (Q5, _fit("perfect"))]
    ranking = rank_packages(entries, [_measurement(Q4, 60.0), _measurement(Q5, 40.0, minutes=1)], default_scenario())
    assert [r.package.quantization for r in ranking.ranked] == ["Q4_K_M", "Q5_K_M"]


def test_larger_weights_break_a_tie_on_quantization():
    small = _package("Q4_K_M", "Q4_K_M", 4_000_000_000, 7, repo="aaa/Nova-7B-GGUF")
    large = _package("Q4_K_M", "Q4_K_M", 4_600_000_000, 8, repo="zzz/Nova-7B-GGUF")
    ranking = rank_packages([(small, _fit("good")), (large, _fit("good"))], [], default_scenario())
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
    ranking = rank_packages([(Q4, _fit("unknown")), (Q5, _fit("good"))], [], default_scenario())
    assert [r.package.quantization for r in ranking.ranked] == ["Q5_K_M"]
    assert ranking.not_covered[0].reason == "architecture not covered by v1"


def test_a_fit_for_another_context_is_a_caller_error():
    with pytest.raises(ValueError, match="context"):
        rank_packages([(Q4, _fit("good", context=4096))], [], default_scenario())


def test_top_is_the_first_ten():
    packages = [_package(f"Q4_K_M-{i}", "Q4_K_M", 4_000_000_000 + i, 10 + i) for i in range(12)]
    ranking = rank_packages([(p, _fit("good")) for p in packages], [], default_scenario())
    assert len(ranking.ranked) == 12
    assert ranking.top == ranking.ranked[:TOP_LIMIT] and len(ranking.top) == 10


def test_shuffled_large_input_gives_one_order():
    packages = [_package(f"Q4_K_M-{i}", "Q4_K_M", 4_000_000_000 + (i % 3), 30 + i) for i in range(20)]
    entries = [(p, _fit(("perfect", "good", "marginal")[i % 3])) for i, p in enumerate(packages)]
    expected = _order(rank_packages(entries, [], default_scenario()))
    rng = random.Random(7)
    for _ in range(50):
        rng.shuffle(entries)
        assert _order(rank_packages(entries, [], default_scenario())) == expected
