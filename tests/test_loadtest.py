"""Tests for modelroom.loadtest: the inventory, the measured run and the stored file.

Every run here goes against a `FixtureDaemon` built from answers a real Ollama daemon gave on
2026-09-24 (`tests/fixtures/ollama_*`). No process is spawned and no socket is opened; the
machine load comes from the same fixture `Probes` the hardware tests use.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.contracts import InstalledModel
from modelroom.daemon import GENERATE_PATH, PS_PATH, SHOW_PATH, DaemonError, FixtureDaemon
from modelroom.http import Response
from modelroom.loadtest import (
    GPU_UTILIZATION_ARGS,
    LoadTestError,
    installed_candidates,
    installed_models,
    run_load_test,
    store_measurement,
    weight_digest,
)
from modelroom.measurements import (
    Scenario,
    default_scenario,
    measurement_path,
    read_measurements,
    shipped_protocol,
)
from modelroom.ranking import group_zero_measurement
from modelroom.state import acquire_lock, release_lock

from fixture_support import (
    DEEPSEEK_MANIFEST_DIGEST,
    DEEPSEEK_OLLAMA_NAME,
    DEEPSEEK_WEIGHTS_DIGEST,
    GRANITE_MANIFEST_DIGEST,
    deepseek_package,
    default_generates,
    empty_ps_answer,
    generate_answer,
    granite_ollama_package,
    json_response,
    loadtest_daemon,
    offline_daemon,
    ps_answer,
    windows_probes,
    windows_runner,
)

NOW = datetime(2026, 9, 24, 8, 43, 0, tzinfo=timezone.utc)
PROFILE_ID = "3f9a0c21d4e6b870"
CLOUD_NAME = "glm-5.3-flash:cloud"


def _installed(name: str, digest: str, size_bytes: int = 5_027_786_589) -> InstalledModel:
    return InstalledModel(name=name, digest=digest, size_bytes=size_bytes, observed_at=NOW)


def _finished(returncode: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(list(GPU_UTILIZATION_ARGS), returncode, stdout=stdout, stderr="")


def _shows(digest: str | None, reason: str | None = "no weight digest in the /api/show answer"):
    return lambda _name: (digest, None if digest else reason)


def _candidate(daemon=None):
    inventory = installed_candidates(
        [deepseek_package()],
        [_installed(DEEPSEEK_OLLAMA_NAME, DEEPSEEK_MANIFEST_DIGEST)],
        _shows(DEEPSEEK_WEIGHTS_DIGEST),
    )
    assert inventory.candidates, "the example package has to match its own installed model"
    return inventory.candidates[0]


def _measure(daemon, scenario: Scenario | None = None, probes=None):
    return run_load_test(
        daemon,
        _candidate(),
        scenario or default_scenario(),
        PROFILE_ID,
        probes=probes or windows_probes(),
        measured_at=NOW,
    )


# --- the inventory ------------------------------------------------------------------------


def test_the_inventory_reads_the_daemons_installed_models():
    models, reason = installed_models(loadtest_daemon(), NOW)

    assert reason is None
    assert DEEPSEEK_OLLAMA_NAME in [model.name for model in models]


def test_an_unreachable_daemon_is_a_reason_not_a_failure():
    models, reason = installed_models(offline_daemon(), NOW)

    assert models is None
    assert "did not answer" in reason


def test_an_ollama_package_matches_on_its_manifest_digest():
    inventory = installed_candidates(
        [granite_ollama_package()],
        [_installed("granite4.2:8b", GRANITE_MANIFEST_DIGEST)],
        _shows(None),
    )

    candidate = inventory.candidates[0]
    assert candidate.ollama_name == "granite4.2:8b"
    assert candidate.package_ref.content_source == "ollama"
    assert candidate.package_ref.ollama_manifest_digest == GRANITE_MANIFEST_DIGEST


def test_a_hugging_face_package_matches_on_repo_and_a_shown_weight_digest():
    candidate = _candidate()

    ref = candidate.package_ref
    assert ref.content_source == "huggingface"
    assert ref.hf_repo == deepseek_package().repo
    assert ref.hf_revision == deepseek_package().revision
    assert ref.hf_file_digest == DEEPSEEK_WEIGHTS_DIGEST


def test_the_measurement_of_a_candidate_is_found_by_the_ranking_rule():
    """What the load test writes has to be what `ranking._matches` looks for, or it stays unseen."""
    record = _measure(loadtest_daemon())

    assert group_zero_measurement(deepseek_package(), [record], default_scenario()) is record


def test_a_name_hit_without_a_shown_digest_is_listed_and_never_measured():
    inventory = installed_candidates(
        [deepseek_package()],
        [_installed(DEEPSEEK_OLLAMA_NAME, DEEPSEEK_MANIFEST_DIGEST)],
        _shows(None),
    )

    assert inventory.candidates == []
    assert [entry.name for entry in inventory.unmatched] == [DEEPSEEK_OLLAMA_NAME]
    assert "no weight digest" in inventory.unmatched[0].reason


def test_a_name_hit_whose_digest_belongs_to_another_file_is_listed_and_never_measured():
    inventory = installed_candidates(
        [deepseek_package()],
        [_installed(DEEPSEEK_OLLAMA_NAME, DEEPSEEK_MANIFEST_DIGEST)],
        _shows("sha256:" + "c" * 64),
    )

    assert inventory.candidates == []
    assert "no weight file of" in inventory.unmatched[0].reason


def test_a_package_whose_weights_are_several_files_stays_unmatched():
    """One blob proves one file; a sharded package must not be credited with measuring all of it."""
    from modelroom.contracts import PackageFile

    shards = [
        PackageFile(name="model-00001-of-00002.gguf", role="weights_shard", size_bytes=4_000_000_000,
                    digest=DEEPSEEK_WEIGHTS_DIGEST),
        PackageFile(name="model-00002-of-00002.gguf", role="weights_shard", size_bytes=4_000_000_000,
                    digest="sha256:" + "b" * 64),
    ]

    inventory = installed_candidates(
        [deepseek_package(files=shards)],
        [_installed(DEEPSEEK_OLLAMA_NAME, DEEPSEEK_MANIFEST_DIGEST)],
        _shows(DEEPSEEK_WEIGHTS_DIGEST),
    )

    assert inventory.candidates == []
    assert "several files" in inventory.unmatched[0].reason


def test_a_blob_path_with_a_space_in_it_still_proves_the_digest():
    from modelroom.daemon import SHOW_PATH as _SHOW

    modelfile = f"# Modelfile\nFROM C:\\Model Cache\\blobs\\{DEEPSEEK_WEIGHTS_DIGEST.replace(':', '-')}\n"
    daemon = FixtureDaemon(
        {("POST", _SHOW): Response(status=200, body=json.dumps({"modelfile": modelfile}).encode("utf-8"))}
    )

    assert weight_digest(daemon, DEEPSEEK_OLLAMA_NAME) == (DEEPSEEK_WEIGHTS_DIGEST, None)


def test_a_cloud_model_is_never_a_candidate_whatever_its_size():
    inventory = installed_candidates(
        [deepseek_package()],
        [
            _installed(CLOUD_NAME, "sha256:" + "3" * 64, size_bytes=0),
            _installed("gpt-oss:20b-cloud", "sha256:" + "8" * 64, size_bytes=381),
        ],
        _shows(DEEPSEEK_WEIGHTS_DIGEST),
    )

    assert inventory.candidates == []
    assert sorted(inventory.cloud) == ["glm-5.3-flash:cloud", "gpt-oss:20b-cloud"]


def test_an_entry_of_size_zero_is_never_a_candidate():
    inventory = installed_candidates(
        [deepseek_package()],
        [_installed(DEEPSEEK_OLLAMA_NAME, DEEPSEEK_MANIFEST_DIGEST, size_bytes=0)],
        _shows(DEEPSEEK_WEIGHTS_DIGEST),
    )

    assert inventory.candidates == []


def test_a_package_that_is_no_longer_active_is_never_a_candidate():
    inventory = installed_candidates(
        [deepseek_package(active=False)],
        [_installed(DEEPSEEK_OLLAMA_NAME, DEEPSEEK_MANIFEST_DIGEST)],
        _shows(DEEPSEEK_WEIGHTS_DIGEST),
    )

    assert inventory.candidates == []
    assert inventory.unmatched == []


def test_an_installed_model_no_package_claims_is_passed_over_in_silence():
    inventory = installed_candidates(
        [deepseek_package()],
        [_installed("nomic-embed-text:latest", "sha256:" + "0" * 64, size_bytes=274_302_450)],
        _shows(DEEPSEEK_WEIGHTS_DIGEST),
    )

    assert (inventory.candidates, inventory.unmatched, inventory.cloud) == ([], [], [])


# --- the weight digest the daemon shows -----------------------------------------------------


def test_the_weight_digest_comes_from_the_shown_modelfile():
    daemon = FixtureDaemon({("POST", SHOW_PATH): json_response("ollama_show_hf_gguf.json")})

    digest, reason = weight_digest(daemon, DEEPSEEK_OLLAMA_NAME)

    assert (digest, reason) == (DEEPSEEK_WEIGHTS_DIGEST, None)
    assert daemon.calls == [("POST", SHOW_PATH, {"model": DEEPSEEK_OLLAMA_NAME})]


def test_a_shown_model_without_a_blob_line_has_no_digest():
    daemon = FixtureDaemon({("POST", SHOW_PATH): Response(status=200, body=b'{"modelfile": "FROM nova:7b"}')})

    digest, reason = weight_digest(daemon, "nova:7b")

    assert digest is None
    assert "no weight digest" in reason


def test_a_daemon_that_cannot_show_a_model_gives_a_reason():
    daemon = FixtureDaemon({("POST", SHOW_PATH): Response(status=404, body=b'{"error": "not found"}')})

    digest, reason = weight_digest(daemon, "nova:7b")

    assert digest is None
    assert "404" in reason


# --- the measured run -----------------------------------------------------------------------


def test_a_clean_run_is_valid_comparable_and_carries_the_three_speeds():
    record = _measure(loadtest_daemon())

    protocol = shipped_protocol()
    assert len(record.runs) == protocol.measured_runs
    assert (record.validity, record.comparable) == ("valid", True)
    assert record.protocol == "v1"
    assert record.daemon_version == "0.34.2"
    assert record.ollama_name == DEEPSEEK_OLLAMA_NAME
    assert record.measured_at == NOW
    assert record.scenario == default_scenario()
    assert round(record.tps_min, 2) == 60.95
    assert round(record.tps_max, 2) == 67.37
    assert record.tps_min < record.tps_mean < record.tps_max


def test_the_warm_up_is_run_but_is_not_one_of_the_measured_runs():
    daemon = loadtest_daemon()

    record = _measure(daemon)

    protocol = shipped_protocol()
    generates = [call for call in daemon.calls if call[1] == GENERATE_PATH]
    assert len(generates) == protocol.warmup_runs + protocol.measured_runs
    assert len(record.runs) == protocol.measured_runs


def test_every_generate_carries_the_protocol_and_the_ranking_context():
    daemon = loadtest_daemon()
    _measure(daemon, Scenario(context_requested=4096, context_origin="entered", kv_type="f16", kv_type_assumed=True, requests=1))

    protocol = shipped_protocol()
    for _method, path, body in [call for call in daemon.calls if call[1] == GENERATE_PATH]:
        assert body["model"] == DEEPSEEK_OLLAMA_NAME
        assert body["prompt"] == protocol.prompt
        assert (body["raw"], body["stream"]) == (True, False)
        assert body["options"] == {
            "num_ctx": 4096,
            "num_predict": protocol.options.num_predict,
            "temperature": protocol.options.temperature,
            "seed": protocol.options.seed,
        }


def test_the_machine_load_is_read_once_before_the_runs():
    runner = windows_runner({GPU_UTILIZATION_ARGS: _finished(0, "17\n")})

    record = _measure(loadtest_daemon(), probes=windows_probes(runner=runner))

    assert record.load_state.gpu_utilization_percent == 17.0
    assert runner.calls.count(GPU_UTILIZATION_ARGS) == 1


@pytest.mark.parametrize(
    "finished",
    [_finished(1, ""), _finished(0, ""), _finished(0, "N/A\n"), _finished(0, "220\n")],
    ids=["failed", "silent", "not a number", "out of range"],
)
def test_a_load_source_that_answers_nothing_usable_is_no_value_and_no_failure(finished):
    record = _measure(
        loadtest_daemon(), probes=windows_probes(runner=windows_runner({GPU_UTILIZATION_ARGS: finished}))
    )

    assert record.load_state.gpu_utilization_percent is None
    assert record.validity == "valid"


def test_a_machine_without_a_load_average_claims_none():
    record = run_load_test(
        loadtest_daemon(),
        _candidate(),
        default_scenario(),
        PROFILE_ID,
        probes=windows_probes(),
        measured_at=NOW,
        cpu_load=lambda: None,
    )

    assert record.load_state.cpu_load is None


def test_a_run_that_did_not_reach_the_token_limit_is_invalid_with_its_reason():
    generates = default_generates()
    generates[2] = generate_answer(2_000_000_000, done_reason="stop")

    record = _measure(loadtest_daemon(generates=generates))

    assert record.validity == "invalid"
    assert "done_reason 'stop'" in record.validity_reason
    assert record.tps_mean is None
    assert record.comparable is True


def test_a_digest_that_changes_between_two_observations_is_not_comparable():
    observations = [ps_answer(), ps_answer(), ps_answer(digest="d" * 64), ps_answer()]

    record = _measure(loadtest_daemon(observations=observations))

    assert record.comparable is False
    assert "digest" in record.comparable_reason
    assert record.validity == "valid"


def test_a_context_the_daemon_did_not_load_is_not_comparable():
    observations = [ps_answer(context_length=4096)] + [ps_answer()] * 3

    record = _measure(loadtest_daemon(observations=observations))

    assert record.comparable is False
    assert "4096" in record.comparable_reason


def test_an_observation_that_shows_another_model_instead_is_not_comparable():
    observations = [ps_answer()] * 3 + [ps_answer(name="granite4.2:8b")]

    record = _measure(loadtest_daemon(observations=observations))

    assert record.comparable is False
    assert "granite4.2:8b" in record.comparable_reason


def test_a_second_model_loaded_beside_the_measured_one_is_still_comparable():
    """A daemon may hold several models; another one standing first says nothing about this run."""
    import json as _json

    from modelroom.http import Response as _Response

    loaded = _json.loads(ps_answer().body.decode("utf-8"))
    other = dict(loaded["models"][0], name="granite4.2:8b", model="granite4.2:8b", digest="e" * 64)
    beside = _Response(
        status=200, body=_json.dumps({"models": [other, loaded["models"][0]]}).encode("utf-8")
    )

    record = _measure(loadtest_daemon(observations=[beside] * 4))

    assert (record.comparable, record.comparable_reason) == (True, None)


def test_an_empty_observation_is_not_comparable_and_says_so():
    observations = [ps_answer(), empty_ps_answer(), ps_answer(), ps_answer()]

    record = _measure(loadtest_daemon(observations=observations))

    assert record.comparable is False
    assert "no observation" in record.comparable_reason


def test_a_daemon_that_goes_away_mid_run_is_a_fault_with_a_reason():
    generates = default_generates()
    generates[1] = DaemonError("the Ollama daemon did not answer (refused)")

    with pytest.raises(LoadTestError) as caught:
        _measure(loadtest_daemon(generates=generates))

    assert "did not answer" in str(caught.value)


def test_a_run_that_reached_the_time_limit_is_a_fault_with_a_reason():
    generates = default_generates()
    generates[0] = DaemonError("no answer within 600.0 s")

    with pytest.raises(LoadTestError) as caught:
        _measure(loadtest_daemon(generates=generates))

    assert "600.0 s" in str(caught.value)


def test_a_model_the_daemon_no_longer_has_is_a_fault_with_a_reason():
    generates = default_generates()
    generates[0] = Response(status=404, body=b'{"error": "model not found"}')

    with pytest.raises(LoadTestError) as caught:
        _measure(loadtest_daemon(generates=generates))

    assert "404" in str(caught.value)


@pytest.mark.parametrize("duration", [2_100_000_000, 2_000_000_000, 1_900_000_000, 3_000_000_000])
def test_three_runs_of_the_same_duration_are_one_speed_and_a_valid_measurement(duration: int):
    """A float sum can land one unit below its own minimum; that must not cost the measurement."""
    generates = [generate_answer(), *(generate_answer(duration) for _ in range(3))]

    record = _measure(loadtest_daemon(generates=generates))

    assert record.validity == "valid"
    assert record.tps_min == record.tps_max
    assert record.tps_min <= record.tps_mean <= record.tps_max
    assert record.tps_mean == pytest.approx(128 / (duration / 1e9))


def test_a_counter_too_large_to_compute_a_speed_from_is_a_fault_with_a_reason():
    """`RunCounters` has no upper bound, so a nonsense duration passes it and overflows the speed."""
    generates = default_generates()
    generates[1] = Response(
        status=200,
        body=json.dumps(
            {
                "done_reason": "length",
                "eval_count": 128,
                "eval_duration": 10**400,
                "prompt_eval_count": 52,
                "prompt_eval_duration": 17000000,
                "load_duration": 8000000,
            }
        ).encode("utf-8"),
    )

    with pytest.raises(LoadTestError) as caught:
        _measure(loadtest_daemon(generates=generates))

    assert "not a measurement this protocol can record" in str(caught.value)


def test_an_answer_that_is_not_a_measurement_is_a_fault_with_a_reason():
    generates = default_generates()
    generates[1] = Response(status=200, body=b'{"done": true}')

    with pytest.raises(LoadTestError) as caught:
        _measure(loadtest_daemon(generates=generates))

    assert "done_reason" in str(caught.value)


@pytest.mark.parametrize("path", [GENERATE_PATH, PS_PATH], ids=["generate", "ps"])
def test_an_answer_nested_deeper_than_the_parser_goes_is_a_fault_with_a_reason(path):
    """`json.loads` raises `RecursionError` there, which is no `ValueError` -- and no traceback."""
    deep = Response(status=200, body=(b"[" * 200_000) + (b"]" * 200_000))
    daemon = loadtest_daemon()
    daemon._answers[("POST" if path == GENERATE_PATH else "GET", path)] = deep

    with pytest.raises(LoadTestError) as caught:
        _measure(daemon)

    assert "cannot read" in str(caught.value)


def test_a_daemon_with_no_version_is_a_fault_before_any_run():
    daemon = loadtest_daemon()
    daemon._answers[("GET", "/api/version")] = Response(status=500, body=b"")

    with pytest.raises(LoadTestError):
        _measure(daemon)

    assert [call for call in daemon.calls if call[1] == GENERATE_PATH] == []


# --- the file the run leaves behind -----------------------------------------------------------


def test_the_measurement_is_stored_under_the_profile_and_reads_back(tmp_path: Path):
    record = _measure(loadtest_daemon())

    path = store_measurement(tmp_path / "state", record, NOW)

    assert path == measurement_path(tmp_path / "state", PROFILE_ID, record.measurement_id)
    read = read_measurements(tmp_path / "state", PROFILE_ID)
    assert read.unreadable == []
    assert [stored.measurement_id for stored in read.records] == [record.measurement_id]
    assert group_zero_measurement(deepseek_package(), read.records, default_scenario()) is not None


def test_a_held_lock_stops_the_write_and_leaves_no_file(tmp_path: Path):
    from modelroom.state import LockHeldError

    record = _measure(loadtest_daemon())
    state = tmp_path / "state"
    handle = acquire_lock(state / "modelroom.lock", "fetch", NOW)
    try:
        with pytest.raises(LockHeldError):
            store_measurement(state, record, NOW)
    finally:
        release_lock(handle)

    assert read_measurements(state, PROFILE_ID).records == []


def test_a_failed_run_leaves_no_file_at_all(tmp_path: Path):
    generates = default_generates()
    generates[2] = DaemonError("the Ollama daemon did not answer (refused)")

    with pytest.raises(LoadTestError):
        _measure(loadtest_daemon(generates=generates))

    assert read_measurements(tmp_path / "state", PROFILE_ID).records == []


def test_an_invalid_measurement_is_stored_too_and_stays_out_of_group_zero(tmp_path: Path):
    generates = default_generates()
    generates[1] = generate_answer(2_000_000_000, done_reason="stop")
    record = _measure(loadtest_daemon(generates=generates))

    store_measurement(tmp_path / "state", record, NOW)

    stored = read_measurements(tmp_path / "state", PROFILE_ID).records
    assert len(stored) == 1
    assert json.loads((tmp_path / "state" / "measurements" / PROFILE_ID / f"{record.measurement_id}.json").read_text(encoding="utf-8"))["validity"] == "invalid"
    assert group_zero_measurement(deepseek_package(), stored, default_scenario()) is None
