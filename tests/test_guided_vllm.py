"""Tests for the vLLM server in the guided mode (AP9-SV1): step 4, the card, and nothing persisted.

A results folder is laid out by hand as a guided run on a rented two-L4 server leaves it: a
configuration with the served model's repository in `repos`, the recorded profile
`profiles_v4/two_l4_server.json` bound by the pointer file. The fetch of the run records the
`tensor` package from the recorded Hub answers, the vLLM server answers from the recordings of
vLLM 0.30.0, and the Ollama daemon is not there. Nothing outside `tmp_path` is written.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelroom.binding import POINTER_SCHEMA_VERSION, GuidedPointer, write_pointer
from modelroom.dialog import FileAsker
from modelroom.guided import run_guided
from modelroom.guided_vllm import match_served
from modelroom.http import Response
from modelroom.vllm_server import CHAT_PATH, METRICS_PATH, MODELS_PATH, FixtureVllmServer, ServedModel, VllmServerError

from fixture_support import (
    FIXTURES,
    json_response,
    QWEN25_REPO,
    build_transport,
    guided_transport_mapping,
    offline_daemon,
    offline_servers,
    qwen25_tensor_package,
    qwen25_transport_mapping,
    vllm_servers,
    windows_probes,
)

RUN_AT = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
L4_PROFILE = "6a2f8beef0d71f93"
SERVED = ServedModel(id=QWEN25_REPO, root=QWEN25_REPO, max_model_len=4096)
CONFIG = f"""schema_version = 3

packagers = []
publishers = ["Qwen"]

[[families]]
name = "qwen2.5"

  [[families.base_models]]
  hf_repo = "{QWEN25_REPO}"
  repos = ["{QWEN25_REPO}"]

[machines.workstation]
reserve_ram_gib = 8.0
reserve_vram_gib = 1.0
writer = true
profile = "{L4_PROFILE}"

[paths]
state = "state"
markdown = "docs/models.md"
"""
ANSWERS = {
    "machines": [],
    "search": "qwen",
    "filter_owners": True,
    "select": [],
    "users": "20 requests",
    "context": "4096",
    "load_test_vllm": True,
}


def _folder(tmp_path: Path, bound: bool = True) -> Path:
    """The results folder and the pointer file of a machine that measured itself into it before."""
    results = (tmp_path / "results").resolve()
    hardware = results / "state" / "hardware"
    hardware.mkdir(parents=True)
    (results / "modelroom.toml").write_bytes(CONFIG.encode("utf-8"))
    shutil.copyfile(FIXTURES / "profiles_v4" / "two_l4_server.json", hardware / f"{L4_PROFILE}.json")
    bindings = {str(results): L4_PROFILE} if bound else {}
    pointer = GuidedPointer(schema_version=POINTER_SCHEMA_VERSION, current=str(results), bindings=bindings)
    write_pointer(tmp_path / "home" / ".modelroom" / "guided.json", pointer)
    return results


def _run(tmp_path: Path, answers: dict | None = None, *, servers=None, bound: bool = True) -> tuple[int, list[str]]:
    results = _folder(tmp_path, bound)
    lines: list[str] = []
    code = run_guided(
        FileAsker(ANSWERS if answers is None else answers),
        here=results,
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport({**guided_transport_mapping(), **qwen25_transport_mapping()}),
        probes=windows_probes(host="workstation"),
        daemon=offline_daemon(),
        servers=servers if servers is not None else vllm_servers(),
        now=RUN_AT,
        out=lines.append,
        colored=False,
    )
    return code, lines


def _step(lines: list[str], number: int) -> list[str]:
    """The lines of one step, from its head to its balance, stripped and with speeds masked."""
    start = next(index for index, line in enumerate(lines) if f"Step {number} of 5" in line)
    end = next(index for index in range(start + 1, len(lines)) if lines[index].strip()[:1] in ("✓", "–"))
    return [_masked(line.strip()) for line in lines[start + 1 : end + 1]]


def _masked(line: str) -> str:
    import re

    return re.sub(r"\d+\.\d(?= tok/s|–)|(?<=–)\d+\.\d", "<n>", line)


def _card(lines: list[str]) -> list[str]:
    start = next(index for index, line in enumerate(lines) if "Step 5 of 5" in line)
    return [_masked(line.rstrip()) for line in lines[start:] if line.strip()]


# --- which package the server serves (D) ----------------------------------------------------------


def test_the_one_active_complete_tensor_package_of_the_root_is_the_served_one():
    package = qwen25_tensor_package()

    assert match_served([package], SERVED) is package


@pytest.mark.parametrize(
    "packages",
    [
        [],
        [qwen25_tensor_package(active=False)],
        [qwen25_tensor_package(repo="Qwen/Qwen2.5-7B-Instruct")],
    ],
    ids=["none", "inactive", "another-repository"],
)
def test_no_such_package_is_a_reason_with_the_repository_last(packages):
    assert match_served(packages, SERVED) == f"no package of this folder comes from {QWEN25_REPO}"


def test_two_revisions_of_the_repository_are_no_answer_and_no_fit():
    """The server states no revision, so it names none of them (`hf.py:434-448` keeps both)."""
    other = qwen25_tensor_package(revision="0" * 40)

    assert match_served([qwen25_tensor_package(), other], SERVED) == (
        f"the server names none of the 2 tensor packages of {QWEN25_REPO}"
    )


# --- step 4 and the card (F) -------------------------------------------------------------------------


def test_step_4_reads_the_lines_of_the_brief_at_load_test_vllm_true(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    assert _step(lines, 4) == [
        "Ollama measurements run one request; this ranking assumes 20",
        f"no load test: profile from home binding has another os_fingerprint ({L4_PROFILE})",
        "a vLLM server answers here, context 4096",
        f"it states it serves {QWEN25_REPO}",
        "revision unknown (not stated)",
        "one warm-up request, then three batches of 20 requests, 128 tokens each",
        "? Measure the vLLM server with 20 requests at once?  Yes",
        "measured vLLM: <n> tok/s total (<n>–<n>), context 4k",
        "per request: <n> tok/s (<n>–<n>)",
        "identity stated by the server; not ranked, nothing written",
        "vLLM fit, computed: one card of 22.49 GiB, 0.92 of it for vLLM",
        "weights 0.92 GiB, runtime 1.50 GiB (one recording), KV cache 18.27 GiB",
        "32 requests of context 4k at once, from size: fits",
        "✓ Measurement     vLLM measured",
    ]


def test_the_start_screen_names_the_served_model_and_the_card_its_speed(tmp_path: Path):
    _code, lines = _run(tmp_path)

    assert f"server    vLLM                    context 4096, serves (stated) {QWEN25_REPO}" in [line.strip() for line in lines]
    card = _card(lines)
    assert " server    vLLM                    <n> tok/s measured (vLLM, 20 requests, total)" in card
    assert "                                   identity stated · fit: 32 at once, fits" in card
    labels = [line.split()[0] for line in card if line[1:2].strip() and not line.startswith(" ──")]
    assert labels[labels.index("speed") : labels.index("result") + 1] == ["speed", "server", "result"]


def test_sixty_five_requests_measure_sixty_four_and_are_not_comparable(tmp_path: Path):
    code, lines = _run(tmp_path, {**ANSWERS, "users": "65 requests"}, servers=vllm_servers(requests=65))

    step = _step(lines, 4)
    assert code == 0
    assert "64 of 65 requests measured" in step
    assert "vLLM: not comparable (64 requests measured, the ranking assumes 65)" in step
    assert not any(line.startswith(("measured vLLM", "per request")) for line in step)
    assert " server    vLLM                    not comparable (64 requests measured, the ranking assumes 65)" in _card(lines)


def test_an_invalid_measurement_says_its_reason_instead_of_the_speeds(tmp_path: Path):
    code, lines = _run(tmp_path, servers=vllm_servers(finish="stop"))

    step = _step(lines, 4)
    assert code == 0
    assert "vLLM: invalid (batch 1: finish_reason stop)" in step
    assert not any(line.startswith(("measured vLLM", "per request")) for line in step)
    assert " server    vLLM                    invalid (batch 1: finish_reason stop)" in _card(lines)


def test_a_server_that_lets_the_load_test_down_is_a_step_that_did_not_finish(tmp_path: Path):
    """The warm-up chat fails: `nothing measured for vLLM: <reason>`, exit `1`, the document still written."""
    reason = "http://127.0.0.1:8000/v1/chat/completions: the vLLM server did not answer (reset)"
    going = FixtureVllmServer(
        {
            ("GET", MODELS_PATH): json_response("vllm_v1_models_l4.json"),
            ("GET", METRICS_PATH): json_response("vllm_metrics_before_l4.txt"),
            ("POST", CHAT_PATH): VllmServerError(reason),
        }
    )

    code, lines = _run(tmp_path, servers={"vllm": going})

    assert code == 1
    assert f"nothing measured for vLLM: {reason}" in lines
    assert "1 step(s) did not finish; see the lines above" in lines
    assert _step(lines, 4)[-1] == "– Measurement     none in this run"
    assert " server    vLLM                    not measured in this run" in _card(lines)
    assert (tmp_path / "results" / "docs" / "models.md").is_file()


def test_a_server_that_asks_for_a_key_measures_nothing_and_ends_0(tmp_path: Path):
    asking = {"vllm": FixtureVllmServer({("GET", MODELS_PATH): Response(status=401, body=b"{}")})}

    code, lines = _run(tmp_path, servers=asking)

    assert code == 0
    assert "server    127.0.0.1:8000          asks for an API key" in [line.strip() for line in lines]
    step = _step(lines, 4)
    assert "the vLLM server asks for an API key; modelroom reads an open local server only" in step
    assert step[-1] == "– Measurement     none in this run"
    assert [call[1] for call in asking["vllm"].calls] == [MODELS_PATH]
    assert not any("server    vLLM" in line for line in _card(lines))


def test_an_answer_file_without_the_vllm_answer_measures_nothing(tmp_path: Path):
    """`load_test_vllm` is the third optional answer key: missing, it is `no`."""
    answers = {key: value for key, value in ANSWERS.items() if key != "load_test_vllm"}
    servers = vllm_servers()

    code, lines = _run(tmp_path, answers, servers=servers)

    assert code == 0
    assert "? Measure the vLLM server with 20 requests at once?  No" in _step(lines, 4)
    assert [call[1] for call in servers["vllm"].calls] == [MODELS_PATH]
    card = _card(lines)
    assert " server    vLLM                    not measured in this run" in card
    assert "                                   identity stated · fit: 32 at once, fits" in card


def test_without_a_bound_profile_the_fit_line_gives_the_reason(tmp_path: Path):
    _code, lines = _run(tmp_path, bound=False)

    step = _step(lines, 4)
    assert "no vLLM fit: this machine has no profile in this folder a fit may compute for" in step
    assert not any(line.startswith("vLLM fit, computed") for line in step)
    assert "                                   identity stated · no fit computed" in _card(lines)


def test_a_folder_with_nothing_to_rank_shows_the_row_without_a_speed(tmp_path: Path):
    """`show_nothing`: no snapshot, so no package matches, and the row says so (`views.py:751`)."""
    config = CONFIG.split("[[families]]")[0] + CONFIG.split('repos = ["' + QWEN25_REPO + '"]\n')[1]
    results = _folder(tmp_path / "empty")
    (results / "modelroom.toml").write_bytes(config.encode("utf-8"))
    lines: list[str] = []

    run_guided(
        FileAsker(ANSWERS),
        here=results,
        pointer_path=tmp_path / "empty" / "home" / ".modelroom" / "guided.json",
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(host="workstation"),
        daemon=offline_daemon(),
        servers=vllm_servers(),
        now=RUN_AT,
        out=lines.append,
        colored=False,
    )

    assert f"no package of this folder comes from {QWEN25_REPO}" in _step(lines, 4)
    card = _card(lines)
    assert " server    vLLM                    no package of this folder" in card
    assert card[-1].strip().startswith("–")


# --- the list of servers (the owner's condition) ----------------------------------------------------


def test_a_third_server_in_the_list_reaches_the_start_screen_step_4_and_both_cards(tmp_path: Path, monkeypatch):
    """No line of the guided mode changes: the entry is added to `LOCAL_SERVERS` and read everywhere."""
    from modelroom import local_servers
    from modelroom.intro import Fact

    third = local_servers.LocalServer(
        key="third",
        name="Third",
        origin="http://127.0.0.1:9999",
        transport=lambda run: None,
        start_fact=lambda origins: Fact("server", "Third", "stated by nobody"),
        step=lambda run, config, scenario, results_dir: run.screen.note("the third server was asked") or "Third measured",
        card_facts=lambda run: (Fact("server", "Third", "on the card"),),
    )
    monkeypatch.setattr(local_servers, "LOCAL_SERVERS", (*local_servers.LOCAL_SERVERS, third))

    _code, lines = _run(tmp_path, servers=offline_servers())

    stripped = [line.strip() for line in lines]
    assert "server    Third                   stated by nobody" in stripped
    assert "the third server was asked" in _step(lines, 4)
    assert _step(lines, 4)[-1] == "✓ Measurement     Third measured"
    assert " server    Third                   on the card" in _card(lines)


def test_the_balance_puts_the_daemons_own_first_and_joins_with_a_dot(tmp_path: Path, monkeypatch):
    from modelroom import local_servers

    first = local_servers.LOCAL_SERVERS[0]
    measured = local_servers.LocalServer(
        **{**first.__dict__, "step": lambda run, *args: run.screen.head(4) or "1 of 1 measured"}
    )
    monkeypatch.setattr(local_servers, "LOCAL_SERVERS", (measured, *local_servers.LOCAL_SERVERS[1:]))

    _code, lines = _run(tmp_path)

    assert _step(lines, 4)[-1] == "✓ Measurement     1 of 1 measured · vLLM measured"


# --- nothing persisted, proven -----------------------------------------------------------------------


def _written(tmp_path: Path) -> dict[str, bytes]:
    """Every file the run left in the results folder and in the home folder, by relative path."""
    return {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file() and path.name != "modelroom.lock"
    }


def test_a_run_with_a_vllm_server_writes_the_same_bytes_as_one_without(tmp_path: Path):
    """Two runs, same clock and answers, Ollama offline in both; one with the vLLM fixtures.

    The snapshot is the fetch's own, from the recorded Hub answers, and holds the served package
    (`qwen25_tensor_package`); the profile is `profiles_v4/two_l4_server.json`, bound by the pointer.
    """
    without = tmp_path / "run"
    code, _lines = _run(without, servers=offline_servers())
    first = _written(without)
    shutil.rmtree(without)

    code_with, lines = _run(without, servers=vllm_servers())
    second = _written(without)

    assert (code, code_with) == (0, 0)
    assert "✓ Measurement     vLLM measured" in [line.strip() for line in lines]
    assert set(first) == set(second)
    assert {"results/state/modelroom.json", "results/docs/models.md", "results/docs/models.json"} <= set(first)
    assert [name for name in first if first[name] != second[name]] == []
    snapshot = json.loads(second["results/state/modelroom.json"])
    fetched = [package for package in snapshot["packages"] if package["repo"] == QWEN25_REPO]
    expected = json.loads(qwen25_tensor_package().model_dump_json())
    assert len(fetched) == 1
    for key in ("source", "repo", "revision", "base_model_hf_repo", "format", "complete", "quantization", "provenance"):
        assert fetched[0][key] == expected[key], key
    assert [f for f in fetched[0]["files"] if f["role"] == "weights"] == expected["files"]
    assert not (without / "results" / "state" / "measurements").exists()
