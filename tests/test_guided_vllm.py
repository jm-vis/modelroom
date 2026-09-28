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
    # The run chooses this machine -- its bound profile, row 1 of the list: vLLM measures and counts
    # only for a machine the run chose (Codex r3).
    "machines": [f"profile:{L4_PROFILE}"],
    "search": "qwen",
    "filter_owners": True,
    # The run chooses the served model (Codex F1: a run that chooses none ranks none, and a vLLM
    # number then counts for no model of the run). It is choosable through `_choosable_qwen25`.
    "select": [QWEN25_REPO],
    "users": "20 requests",
    "context": "4096",
    # One question and one list for every server since AP9-UX E; `load_test_vllm` is gone.
    "load_test": True,
    "load_test_packages": ["vllm"],
}


def _own_fingerprint() -> str:
    """The fingerprint of the machine `windows_probes()` stands for, as the measurement computes it."""
    from modelroom.measure import read_os_identity
    from modelroom.profile import os_fingerprint

    probes = windows_probes(host="workstation")
    return os_fingerprint(read_os_identity(probes.platform, probes.runner, probes.read_text).raw_id)


def _folder(tmp_path: Path, bound: bool = True, own: bool = True) -> Path:
    """The results folder and the pointer file of a machine that measured itself into it before.

    `own`: the bound profile carries this machine's fingerprint (it is this machine); else the
    recorded one of the rented server, so the binding points at another machine (Codex r4).
    """
    results = (tmp_path / "results").resolve()
    hardware = results / "state" / "hardware"
    hardware.mkdir(parents=True)
    (results / "modelroom.toml").write_bytes(CONFIG.encode("utf-8"))
    profile = json.loads((FIXTURES / "profiles_v4" / "two_l4_server.json").read_text(encoding="utf-8"))
    if own:
        profile["os_fingerprint"] = _own_fingerprint()
    (hardware / f"{L4_PROFILE}.json").write_bytes(json.dumps(profile, indent=2).encode("utf-8"))
    bindings = {str(results): L4_PROFILE} if bound else {}
    pointer = GuidedPointer(schema_version=POINTER_SCHEMA_VERSION, current=str(results), bindings=bindings)
    write_pointer(tmp_path / "home" / ".modelroom" / "guided.json", pointer)
    return results


def _choosable_qwen25() -> dict:
    """The `Qwen` account page of the search for `qwen`, with one more entry: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`.

    SYNTHETIC, built here from the recorded page of the same account: a quantized repository of the
    served model, so step 2 can offer that model and a run can choose it. The fetch then asks for the
    repository and gets the recorded "does not exist" (`qwen25_transport_mapping`), so no package of it
    enters the snapshot -- the served `tensor` package stays the only one.
    """
    from modelroom.search_pages import account_downloads_url, account_search_url

    page = json.loads((FIXTURES / "hf_search_qwen_publisher.json").read_text(encoding="utf-8"))
    page.append(
        {
            "_id": "0000000000000000000000aa",
            "id": f"{QWEN25_REPO}-GGUF",
            "cardData": {"license": "apache-2.0", "base_model": QWEN25_REPO, "base_model_relation": "quantized"},
            "tags": ["gguf", f"base_model:{QWEN25_REPO}", f"base_model:quantized:{QWEN25_REPO}", "license:apache-2.0"],
            "createdAt": "2024-09-19T00:00:00.000Z",
            "downloads": 1000,
        }
    )
    body = Response(status=200, headers={}, body=json.dumps(page).encode("utf-8"))
    return {("GET", account_search_url("qwen", "Qwen")): body, ("GET", account_downloads_url("qwen", "Qwen")): body}


def _run(
    tmp_path: Path,
    answers: dict | None = None,
    *,
    servers=None,
    bound: bool = True,
    keep: bool = False,
    own: bool = True,
    daemon=None,
) -> tuple[int, list[str]]:
    """`keep`: the folder was laid out by the test already (e.g. with an imported profile)."""
    results = (tmp_path / "results").resolve() if keep else _folder(tmp_path, bound, own)
    lines: list[str] = []
    code = run_guided(
        FileAsker(ANSWERS if answers is None else answers),
        here=results,
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport({**guided_transport_mapping(), **qwen25_transport_mapping(), **_choosable_qwen25()}),
        probes=windows_probes(host="workstation"),
        daemon=daemon if daemon is not None else offline_daemon(),
        servers=servers if servers is not None else vllm_servers(),
        now=RUN_AT,
        out=lines.append,
        colored=False,
    )
    return code, lines


def _answering_daemon():
    """The load test's daemon: it answers, and has no package of the served model installed."""
    from fixture_support import loadtest_daemon

    return loadtest_daemon()


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


def test_step_4_reads_the_lines_of_the_brief_with_the_vllm_row_picked(tmp_path: Path):
    code, lines = _run(tmp_path)

    assert code == 0
    assert _step(lines, 4) == [
        "Ollama measurements run one request; this ranking assumes 20",
        "a vLLM server answers here, context 4096",
        f"it states it serves {QWEN25_REPO}",
        "revision unknown (not stated)",
        "one warm-up request, then three batches of 20 requests, 128 tokens each",
        # This machine is bound to its own profile now (Codex r4), so the daemon's reason stands too.
        "Ollama: cannot reach the Ollama daemon at http://127.0.0.1:11434: "
        "http://127.0.0.1:11434: the Ollama daemon did not answer (refused)",
        "? Measure the speed on this machine now?  Yes",
        "? Which of these should be measured?  vLLM server",
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
    assert " server    vLLM                    <n> tok/s total, 20 requests at once, counts for this run" in card
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

    # A daemon that answers, with nothing of this run installed: Ollama offers no row either.
    code, lines = _run(tmp_path, servers=asking, daemon=_answering_daemon())

    assert code == 0
    assert "server    127.0.0.1:8000          asks for an API key" in [line.strip() for line in lines]
    step = _step(lines, 4)
    assert "the vLLM server asks for an API key; modelroom reads an open local server only" in step
    assert step[-1] == "– Measurement     none in this run"
    assert [call[1] for call in asking["vllm"].calls] == [MODELS_PATH]
    assert not any("server    vLLM" in line for line in _card(lines))


def test_an_answer_file_without_the_load_test_answer_measures_nothing(tmp_path: Path):
    """`load_test` is the one optional answer key: missing, it is `no`, and no list is asked."""
    answers = {key: value for key, value in ANSWERS.items() if key not in ("load_test", "load_test_packages")}
    servers = vllm_servers()

    code, lines = _run(tmp_path, answers, servers=servers)

    assert code == 0
    assert "? Measure the speed on this machine now?  No" in _step(lines, 4)
    assert not any(line.startswith("? Which of these") for line in _step(lines, 4))
    assert [call[1] for call in servers["vllm"].calls] == [MODELS_PATH]
    card = _card(lines)
    assert " server    vLLM                    not measured in this run" in card
    assert "                                   identity stated · fit: 32 at once, fits" in card


def test_without_a_bound_profile_the_server_is_not_offered(tmp_path: Path):
    """Without a binding this machine is no machine of the run (Codex r3): the server is neither
    measured nor counted, even where the run picked the folder's profile."""
    _code, lines = _run(tmp_path, bound=False)

    step = _step(lines, 4)
    assert "vLLM: no load test: this machine is not bound to a profile in this results folder" in step
    assert not any(line.startswith(("vLLM fit, computed", "measured vLLM")) for line in step)
    assert not any("server    vLLM" in line for line in _card(lines))


def test_a_folder_with_nothing_to_rank_shows_the_row_without_a_speed(tmp_path: Path):
    """`show_nothing`: no snapshot, so no package matches, and the row says so (`views.py:751`)."""
    config = CONFIG.split("[[families]]")[0] + CONFIG.split('repos = ["' + QWEN25_REPO + '"]\n')[1]
    results = _folder(tmp_path / "empty")
    (results / "modelroom.toml").write_bytes(config.encode("utf-8"))
    lines: list[str] = []

    run_guided(
        FileAsker({**ANSWERS, "select": []}),
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

    step = _step(lines, 4)
    # No model of this run is the served one: said before the question, the package after the measurement.
    reason = "the number will not count: the server serves Qwen2.5-0.5B-Instruct, this run chose nothing"
    assert step.index(reason) < step.index("? Measure the speed on this machine now?  Yes")
    assert f"no package of this folder comes from {QWEN25_REPO}" in step
    assert step[-1] == "✓ Measurement     vLLM ran, not counted"
    card = _card(lines)
    assert " server    vLLM                    <n> tok/s total, 20 requests at once, not counted" in card
    assert "                                   identity stated · no package of this folder" in card
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
        offer=lambda run, config, scenario, results_dir: run.screen.note("the third server was asked") or local_servers.Offer(),
        measure=lambda run, config, scenario, results_dir, offer, picked: "Third measured",
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
    measured = local_servers.LocalServer(**{**first.__dict__, "measure": lambda run, *args: "1 of 1 measured"})
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
    """Two runs, same clock and answers, the same daemon with nothing of the run installed; one with
    the vLLM fixtures.

    The snapshot is the fetch's own, from the recorded Hub answers, and holds the served package
    (`qwen25_tensor_package`); the profile is `profiles_v4/two_l4_server.json` with this machine's
    fingerprint, bound by the pointer.
    """
    without = tmp_path / "run"
    code, _lines = _run(without, servers=offline_servers(), daemon=_answering_daemon())
    first = _written(without)
    shutil.rmtree(without)

    code_with, lines = _run(without, servers=vllm_servers(), daemon=_answering_daemon())
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


# --- step 4 from one mould (AP9-UX E) --------------------------------------------------------------------

UNSLOTH_QWEN = "unsloth/Qwen3.5-9B-GGUF"
DEEPSEEK_REPO = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
FRESH = {
    "results": "here",
    "machines": ["this-machine"],
    "machine_class": "laptop",
    "search": "qwen",
    "filter_owners": True,
    "select": [UNSLOTH_QWEN],
    "users": 1,
    "context": "8192",
}
WITH_DEEPSEEK = {**FRESH, "select": [UNSLOTH_QWEN, DEEPSEEK_REPO]}


class _Counting(FileAsker):
    """A file asker that keeps every list it was asked, by key."""

    def __init__(self, answers):
        super().__init__(answers)
        self.lists: dict[str, list] = {}

    def checkbox(self, key, question, choices, instruction=None, header=None):
        self.lists[key] = list(choices)
        return super().checkbox(key, question, choices, instruction)


def _fresh(tmp_path: Path, answers: dict, *, daemon=None, servers=None) -> tuple[int, list[str], _Counting]:
    """A first guided run in an empty folder: this machine measures itself, so Ollama is bound here."""
    from fixture_support import loadtest_daemon  # noqa: F401 -- the callers pass it

    results = tmp_path / "results"
    results.mkdir(parents=True)
    (tmp_path / "home").mkdir()
    lines: list[str] = []
    asker = _Counting(answers)
    code = run_guided(
        asker,
        here=results,
        pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
        transport=build_transport(guided_transport_mapping()),
        probes=windows_probes(),
        daemon=daemon if daemon is not None else offline_daemon(),
        servers=servers if servers is not None else offline_servers(),
        now=RUN_AT,
        out=lines.append,
        colored=False,
    )
    return code, lines, asker


def test_step_four_asks_one_question_for_every_server(tmp_path: Path):
    from fixture_support import DEEPSEEK_OLLAMA_NAME, loadtest_daemon

    answers = {**WITH_DEEPSEEK, "load_test": True, "load_test_packages": [DEEPSEEK_OLLAMA_NAME, "vllm"]}
    code, lines, asker = _fresh(tmp_path, answers, daemon=loadtest_daemon(), servers=vllm_servers(requests=1))

    assert code == 0
    assert asker.asked.count("load_test") == 1 and asker.asked.count("load_test_packages") == 1
    rows = asker.lists["load_test_packages"]
    assert [row.value for row in rows] == [DEEPSEEK_OLLAMA_NAME, "vllm"]
    assert not any(row.checked for row in rows)
    step = _step(lines, 4)
    assert "Ollama: 1 installed package of this run's models, one request each" in step
    assert step[-1] == "✓ Measurement     1 of 1 measured · vLLM ran, not counted"


def test_the_reason_a_number_will_not_count_stands_before_the_question():
    from modelroom.guided_vllm import not_counting_reason, wrapped

    served = ServedModel(id=QWEN25_REPO, root=QWEN25_REPO, max_model_len=4096)
    package = qwen25_tensor_package()
    other = ("Qwen/Qwen3.5-9B",)

    # The first reason that holds, in the order 1, 2, 3.
    assert not_counting_reason(served, other, 32768, package) == (
        "the number will not count: the server serves Qwen2.5-0.5B-Instruct, this run chose Qwen3.5-9B"
    )
    assert not_counting_reason(served, (QWEN25_REPO,), 32768, package) == (
        "the number will not count: the server runs context 4k, this run asks for 32k"
    )
    assert not_counting_reason(served, (QWEN25_REPO,), 4096, "no package") == (
        f"the number counts, no fit: no package of this folder comes from {QWEN25_REPO}"
    )
    assert not_counting_reason(served, (QWEN25_REPO,), 4096, package) is None
    # Beyond 100 columns the line breaks before the name, and no name is cut.
    long_name = "acme/" + "Nova-" * 16 + "9B"
    lines = wrapped(not_counting_reason(ServedModel(id=long_name, root=long_name, max_model_len=4096), other, 4096, package))
    assert lines[0] == "the number will not count: the server serves"
    assert lines[1].startswith("Nova-Nova-") and lines[1].endswith(", this run chose Qwen3.5-9B")


def test_the_balance_never_says_measured_when_nothing_counts(tmp_path: Path):
    code, lines = _run(tmp_path, {**ANSWERS, "context": "8192"})

    step = _step(lines, 4)
    assert code == 0
    assert "the number will not count: the server runs context 4k, this run asks for 8k" in step
    assert step[-1] == "✓ Measurement     vLLM ran, not counted"
    assert " server    vLLM                    <n> tok/s total, 20 requests at once, not counted" in _card(lines)


@pytest.mark.parametrize(
    "answers, servers",
    [
        (ANSWERS, lambda: vllm_servers(finish="stop")),  # invalid: the balance reads the card's problem
        ({**ANSWERS, "users": "65 requests"}, lambda: vllm_servers(requests=65)),  # capped: not comparable
    ],
    ids=["invalid", "not comparable"],
)
def test_a_measurement_with_a_problem_is_never_measured_in_the_balance(tmp_path: Path, answers, servers):
    """Model and context of the run match (the number would count), but the measurement itself does
    not hold: the balance says what the card says, never `vLLM measured` (Diff-Review M1)."""
    code, lines = _run(tmp_path, answers, servers=servers())

    step = _step(lines, 4)
    assert code == 0
    assert not any(line.startswith("the number will not count") for line in step)  # it would count
    assert step[-1] == "✓ Measurement     vLLM ran, not counted"


def test_one_request_is_singular(tmp_path: Path):
    from modelroom.fit_vllm import VllmFit
    from modelroom.guided_vllm import fit_lines, vllm_rows, VllmOffer
    from modelroom.scenario_text import requests_words

    answers = {**FRESH, "context": "4096", "load_test": True, "load_test_packages": ["vllm"]}
    code, lines, _asker = _fresh(tmp_path, answers, servers=vllm_servers(requests=1))

    assert code == 0
    assert "one warm-up request, then three batches of 1 request, 128 tokens each" in _step(lines, 4)
    assert " server    vLLM                    <n> tok/s total, 1 request at once, not counted" in _card(lines)
    row = vllm_rows(VllmOffer(SERVED, 1, "no package", False))[0]
    assert row.label == f"vLLM server  context 4k  1 request at once  {QWEN25_REPO}"
    assert (requests_words(1), requests_words(2)) == ("1 request", "2 requests")
    from modelroom import guided_context

    assert requests_words is guided_context.requests_words  # one definition (Diff-Review N5)
    fit = VllmFit(
        card_gib=22.49, weights_gib=1.0, runtime_gib=1.5, kv_gib=10.0, context=4096, requests=1, at_once=1,
        fit_class="tight", basis="size", budget_gib=20.7, kv_tokens=4096,
    )
    assert fit_lines(fit)[2].startswith("1 request of context 4k at once")


@pytest.mark.parametrize(
    "case, said, identity",
    [
        ("counts", "counts for this run", "identity stated · fit: 32 at once, fits"),
        ("other context", "not counted", "identity stated · fit: 16 at once, tight"),
        ("no package", "counts, no fit", "identity stated · no package of this folder"),
    ],
)
def test_the_server_row_of_the_card_says_whether_the_number_counts(tmp_path: Path, case, said, identity):
    """The three states of the server row (AP9-UX E), with the recordings of `vllm_servers()`."""
    answers = {**ANSWERS, "context": "8192"} if case == "other context" else ANSWERS
    if case == "no package":
        results = _folder(tmp_path)
        # The model of the folder is the served one, but no repository of it is configured: no package.
        (results / "modelroom.toml").write_bytes(CONFIG.replace(f'  repos = ["{QWEN25_REPO}"]\n', "").encode("utf-8"))
        lines: list[str] = []
        code = run_guided(
            FileAsker(answers),
            here=results,
            pointer_path=tmp_path / "home" / ".modelroom" / "guided.json",
            transport=build_transport({**guided_transport_mapping(), **qwen25_transport_mapping(), **_choosable_qwen25()}),
            probes=windows_probes(host="workstation"),
            daemon=offline_daemon(),
            servers=vllm_servers(),
            now=RUN_AT,
            out=lines.append,
            colored=False,
        )
    else:
        code, lines = _run(tmp_path, answers)

    card = _card(lines)
    assert code == 0
    assert f" server    vLLM                    <n> tok/s total, 20 requests at once, {said}" in card
    assert f"                                   {identity}" in card


@pytest.mark.parametrize(
    "case", ["nothing marked", "an import that failed", "another machine only", "another fingerprint"]
)
def test_the_vllm_server_is_measured_only_where_the_run_chose_this_machine(tmp_path: Path, case):
    """Codex r3 and r4: the daemon's own check -- this machine is the machine of its bound profile
    (fingerprint included), and the run chose it; else no question, no measurement, no number."""
    from modelroom.importer import import_profile

    from fixture_support import FOUR_L4_ID, export_of

    machines = {
        "nothing marked": [],
        "an import that failed": ["import"],
        "another machine only": [f"profile:{FOUR_L4_ID}"],
        "another fingerprint": [f"profile:{L4_PROFILE}"],
    }
    answers = {**ANSWERS, "machines": machines[case], "import_file": "missing.export.json"}
    if case == "another machine only":
        results = _folder(tmp_path)
        import_profile(results / "modelroom.toml", export_of("four_l4_server.json", tmp_path / "exports"), RUN_AT)
    servers = vllm_servers()

    code, lines = _run(
        tmp_path, answers, servers=servers, keep=case == "another machine only", own=case != "another fingerprint"
    )

    step = _step(lines, 4)
    expected = (
        f"vLLM: no load test: profile from home binding has another os_fingerprint ({L4_PROFILE})"
        if case == "another fingerprint"
        else "vLLM: no load test: this machine is not part of this run's result"
    )
    assert expected in step
    assert not any(line.startswith(("? Measure the speed", "measured vLLM")) for line in step)
    assert step[-1] == "– Measurement     none in this run"
    assert [call[1] for call in servers["vllm"].calls] == [MODELS_PATH]  # the start screen read it, nothing more
    assert not any("tok/s" in line or "server    vLLM" in line for line in _card(lines))
    assert code == (1 if case == "an import that failed" else 0)


STATES = [
    # (folder, daemon, servers, answers, question asked, list, balance, exit)
    ("unbound", "none", "offline", {}, False, None, "– Measurement     none in this run", 0),
    # Not bound: this machine is no machine of the run, so the vLLM server is not offered either (r3).
    ("unbound", "none", "vllm", {}, False, None, "– Measurement     none in this run", 0),
    # Bound to a profile of another fingerprint: this is not the machine of that profile, so neither
    # server offers anything (Codex r4) -- the same check for both.
    ("foreign", "none", "vllm", {}, False, None, "– Measurement     none in this run", 0),
    ("foreign", "none", "vllm", {"load_test": False}, False, None, "– Measurement     none in this run", 0),
    # Bound to its own profile and chosen: the vLLM server is offered, measured and counted.
    ("own", "none", "vllm", {}, True, ["vllm"], "✓ Measurement     vLLM measured", 0),
    ("fresh", "offline", "offline", {"load_test": False}, True, None, "– Measurement     none in this run", 0),
    ("fresh", "offline", "offline", {"load_test": True}, True, None, "– Measurement     none in this run", 1),
    ("fresh", "offline", "vllm", {}, True, ["vllm"], "✓ Measurement     vLLM ran, not counted", 0),
    ("fresh", "loadtest", "offline", {}, False, None, "– Measurement     none in this run", 0),
    ("fresh", "loadtest", "vllm", {}, True, ["vllm"], "✓ Measurement     vLLM ran, not counted", 0),
    ("deepseek", "loadtest", "offline", {}, True, ["ollama"], "✓ Measurement     1 of 1 measured", 0),
    ("deepseek", "loadtest", "vllm", {}, True, ["ollama", "vllm"], "✓ Measurement     1 of 1 measured · vLLM ran, not counted", 0),
]


@pytest.mark.parametrize("folder, daemon, servers, extra, asked, rows, balance, exit_code", STATES)
def test_every_state_of_step_four(tmp_path: Path, folder, daemon, servers, extra, asked, rows, balance, exit_code):
    """The state table of AP9-UX E, one row each: the question, the list, the balance, the exit code."""
    from fixture_support import DEEPSEEK_OLLAMA_NAME, loadtest_daemon

    values = {"ollama": DEEPSEEK_OLLAMA_NAME, "vllm": "vllm"}
    picked = [values[row] for row in rows or []]
    answers = {"load_test": True, "load_test_packages": picked, **extra}
    the_servers = vllm_servers() if servers == "vllm" else offline_servers()
    if folder in ("unbound", "foreign", "own"):
        code, lines = _run(
            tmp_path, {**ANSWERS, **answers}, servers=the_servers, bound=folder != "unbound", own=folder == "own"
        )
        asker_rows = None
    else:
        base = WITH_DEEPSEEK if folder == "deepseek" else FRESH
        the_daemon = loadtest_daemon() if daemon == "loadtest" else offline_daemon()
        code, lines, asker = _fresh(tmp_path, {**base, **answers}, daemon=the_daemon, servers=the_servers)
        asker_rows = [row.value for row in asker.lists.get("load_test_packages", [])] or None

    step = _step(lines, 4)
    assert any(line.startswith("? Measure the speed on this machine now?") for line in step) is asked
    if folder not in ("unbound", "foreign", "own"):
        assert asker_rows == (picked or None if rows else None)
    assert step[-1] == balance
    assert code == exit_code
