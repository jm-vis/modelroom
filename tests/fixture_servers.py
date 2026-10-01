"""Fixture helpers for the local servers: the Ollama daemon, the pull, and the vLLM server.

Moved out of `tests/fixture_support.py` unchanged (AP9-UX2, size guard); `fixture_support`
exports every name back, so the tests' imports stay as they are. This module never imports
`fixture_support`. Not a test module (no `test_` prefix).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fixture_http import FIXTURES, envelope_response, json_response
from modelroom.http import Response

DEEPSEEK_BASE = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
DEEPSEEK_REPO = "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
DEEPSEEK_BASE_SHA = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
DEEPSEEK_SHA = "eb48357c179d34dbf515983f798dfb8752a0f261"
# The local name the Ollama daemon gives that package, its manifest digest in `/api/tags`, and
# the digest of the one GGUF blob it was built from -- all three read from a real daemon
# (2026-09-24) and pinned in `tests/fixtures/ollama_*`.
DEEPSEEK_OLLAMA_NAME = "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M"
DEEPSEEK_MANIFEST_DIGEST = "sha256:ecc092d5e10a34f7277b5748cbb9e940c0eb2bbea563f48517ff804939ce3aec"
DEEPSEEK_WEIGHTS_DIGEST = "sha256:a86349a4180c4e6bb43f874c29c404fa2be3f90b15509bd6d86f697dba724ec1"
DEEPSEEK_WEIGHTS_BYTES = 5027785216
GRANITE_MANIFEST_DIGEST = "sha256:f586c02fdecdf151b656207c339aa003997345774a41768bac1fd6d2fb85913b"


# --- the Ollama daemon of this machine (AP9-E) --------------------------------------------------

DAEMON_OBSERVED_AT = datetime(2026, 9, 24, 8, 43, 0, tzinfo=timezone.utc)
# One warm-up run and three measured runs, each a little faster or slower than the next, so a
# test can tell mean, minimum and maximum apart.
MEASURED_DURATIONS = (2_000_000_000, 1_900_000_000, 2_100_000_000)


def generate_answer(eval_duration: int | None = None, done_reason: str | None = None) -> Response:
    """The pinned `/api/generate` answer, with its duration or its end reason overridden."""
    data = json.loads((FIXTURES / "ollama_generate_valid.json").read_text(encoding="utf-8"))
    if eval_duration is not None:
        data["eval_duration"] = eval_duration
    if done_reason is not None:
        data["done_reason"] = done_reason
    return Response(status=200, body=json.dumps(data).encode("utf-8"))


def ps_answer(digest: str | None = None, context_length: int | None = None, name: str | None = None) -> Response:
    """The pinned `/api/ps` answer for the loaded example package, with fields overridden."""
    data = json.loads((FIXTURES / "ollama_ps_deepseek_loaded.json").read_text(encoding="utf-8"))
    entry = data["models"][0]
    if digest is not None:
        entry["digest"] = digest
    if context_length is not None:
        entry["context_length"] = context_length
    if name is not None:
        entry["name"] = name
        entry["model"] = name
    return Response(status=200, body=json.dumps(data).encode("utf-8"))


def empty_ps_answer() -> Response:
    """What `/api/ps` answers while no model is loaded (read from a real daemon, 2026-09-24)."""
    return Response(status=200, body=b'{"models": []}')


def default_generates() -> list[Response]:
    """The four `/api/generate` answers of one protocol-v1 run: a warm-up and three measured."""
    return [generate_answer(), *(generate_answer(duration) for duration in MEASURED_DURATIONS)]


def loadtest_daemon(generates=None, observations=None, tags: str = "ollama_tags_local_loadtest.json"):
    """A `FixtureDaemon` that answers every call one whole load test makes.

    `generates` is the answer sequence of `/api/generate`, `observations` that of `/api/ps` --
    one observation after every run, so four by default. One answer more stands behind them: the
    read of step 5 for its `loaded` line (AP9-RM U10), so a sequence the load test used up does not
    end in an `IndexError` that the hint, which says nothing on any failure, would swallow.
    """
    from modelroom.daemon import FixtureDaemon

    return FixtureDaemon(_loadtest_answers(generates, observations, tags))


def _loadtest_answers(generates=None, observations=None, tags: str = "ollama_tags_local_loadtest.json") -> dict:
    from modelroom.daemon import GENERATE_PATH, PS_PATH, SHOW_PATH, TAGS_PATH, VERSION_PATH

    return {
        ("GET", VERSION_PATH): json_response("ollama_version_local.json"),
        ("GET", TAGS_PATH): json_response(tags),
        ("POST", SHOW_PATH): json_response("ollama_show_hf_gguf.json"),
        ("GET", PS_PATH): [*(observations if observations is not None else [ps_answer()] * 4), ps_answer()],
        ("POST", GENERATE_PATH): default_generates() if generates is None else list(generates),
    }


# --- the pull of step 5 -------------------------------------------------------------------------

# The answer of one real `POST /api/pull` for `smollm2:135m` against the local daemon, recorded byte
# for byte on 2026-09-25: 384 lines, `pulling manifest`, one `pulling <digest>` run per layer with
# `total` and (not always) `completed`, `verifying sha256 digest`, `writing manifest`, `success`.
PULL_STREAM = "ollama_pull_smollm2_stream.ndjson"
PULL_STREAM_LINES = 384


def pull_answer() -> Response:
    """The recorded pull, as the pinned body a `FixtureDaemon` plays back line by line."""
    return Response(status=200, body=(FIXTURES / PULL_STREAM).read_bytes())


def tags_with(name: str, digest: str = "d" * 64, size: int = 1_234_567_890) -> Response:
    """The load test's `/api/tags` answer with one more local entry, as it reads after a pull."""
    data = json.loads((FIXTURES / "ollama_tags_local_loadtest.json").read_text(encoding="utf-8"))
    data["models"].append({"name": name, "model": name, "size": size, "digest": digest, "details": {}})
    return Response(status=200, body=json.dumps(data).encode("utf-8"))


def pull_daemon(pull=None, after_pull: Response | None = None, tags: Response | None = None):
    """The load test's daemon, plus the pull: `/api/tags` answers `after_pull` once a pull was made.

    `pull` is what `POST /api/pull` answers (the recorded stream by default, or a planted
    `DaemonError`); `tags` the inventory before it (the load test's own list by default);
    `after_pull` the inventory the daemon shows from then on, the one from before when it is
    `None` -- a daemon that does not list what it just pulled.
    """
    from modelroom.daemon import PULL_PATH, TAGS_PATH, FixtureDaemon

    answers = _loadtest_answers()
    answers[("POST", PULL_PATH)] = pull if pull is not None else pull_answer()
    if tags is not None:
        answers[("GET", TAGS_PATH)] = tags
    after = after_pull if after_pull is not None else answers[("GET", TAGS_PATH)]

    class _Pulling(FixtureDaemon):
        def __call__(self, method, path, body=None, progress=None):
            pulled = any(call[1] == PULL_PATH for call in self.calls)
            if (method, path) == ("GET", TAGS_PATH) and pulled:
                self.calls.append((method, path, body))
                return after
            return super().__call__(method, path, body, progress)

    return _Pulling(answers)


def offline_daemon():
    """A machine with no Ollama daemon: every call fails with the reason a user would read."""
    from modelroom.daemon import FixtureDaemon
    from modelroom.ollama_local import DEFAULT_BASE_URL

    return FixtureDaemon({}, unreachable=f"{DEFAULT_BASE_URL}: the Ollama daemon did not answer (refused)")


def deepseek_package(**overrides):
    """The `Q4_K_M` package of the load test's example repository, as the fetch records it."""
    from modelroom.contracts import Package, PackageFile

    fields = {
        "source": "huggingface",
        "repo": DEEPSEEK_REPO,
        "revision": DEEPSEEK_SHA,
        "base_model_hf_repo": DEEPSEEK_BASE,
        "format": "gguf",
        "files": [
            PackageFile(
                name="DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
                role="weights",
                size_bytes=DEEPSEEK_WEIGHTS_BYTES,
                digest=DEEPSEEK_WEIGHTS_DIGEST,
            )
        ],
        "complete": True,
        "quantization": "Q4_K_M",
        "provenance": "from_metadata",
        "observed_at": DAEMON_OBSERVED_AT,
        "last_seen": DAEMON_OBSERVED_AT,
        "active": True,
    }
    return Package(**{**fields, **overrides})


def granite_ollama_package(**overrides):
    """An Ollama package whose manifest digest is the one `/api/tags` shows for `granite4.2:8b`."""
    from modelroom.contracts import Package, PackageFile

    fields = {
        "source": "ollama",
        "ollama_name": "granite4.2:8b",
        "manifest_digest": GRANITE_MANIFEST_DIGEST,
        "base_model_hf_repo": DEEPSEEK_BASE,
        "format": "gguf",
        "files": [PackageFile(name="model", role="weights", size_bytes=5_347_929_757, digest=None)],
        "complete": True,
        "quantization": "Q4_K_M",
        "provenance": "from_metadata",
        "observed_at": DAEMON_OBSERVED_AT,
        "last_seen": DAEMON_OBSERVED_AT,
        "active": True,
    }
    return Package(**{**fields, **overrides})


# --- local servers (AP9-SV1): a vLLM server, recorded from vLLM 0.30.0 on one L4 ----------------

QWEN25_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN25_SHA = "7ae557604adf67be50417f59c2c2f167def9a775"
QWEN25_WEIGHTS_BYTES = 988_097_824
QWEN25_WEIGHTS_DIGEST = "sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
VLLM_UNREACHABLE = "http://127.0.0.1:8000/v1/models: the vLLM server did not answer (refused)"


def offline_servers() -> dict:
    """A machine with no vLLM server: every call fails, so the start screen has no row for one."""
    from modelroom.vllm_server import FixtureVllmServer

    return {"vllm": FixtureVllmServer({}, unreachable=VLLM_UNREACHABLE)}


def qwen25_tensor_package(**overrides):
    """The `tensor` package the recorded vLLM server serves, as the fetch records it from the Hub.

    `repo == base_model_hf_repo`: this is the base model's own repository (AP9-UX2 P3), which
    AP9-SV2-P's provenance rule resolves `from_metadata` without a relation check at all. Its
    `config.json` names no `quantization_config` and `torch_dtype: "bfloat16"`
    (`tests/fixtures/hf_qwen25_05b_instruct_config.json`), so `tensor_quant.tensor_quant_label`
    reads `BF16` wherever a test's transport maps that repo's `config.json` -- the two tests that
    do not (`test_guided_vllm_match.py`'s `match_served` cases) pass `quantization=None` of their
    own accord through `overrides` where that still matters to them.
    """
    from modelroom.contracts import Package, PackageFile

    fields = {
        "source": "huggingface",
        "repo": QWEN25_REPO,
        "revision": QWEN25_SHA,
        "base_model_hf_repo": QWEN25_REPO,
        "format": "tensor",
        "files": [
            PackageFile(
                name="model.safetensors", role="weights", size_bytes=QWEN25_WEIGHTS_BYTES, digest=QWEN25_WEIGHTS_DIGEST
            )
        ],
        "complete": True,
        "quantization": "BF16",
        "quantization_source": "config",
        "provenance": "from_metadata",
        "unresolved_reason": None,
        "observed_at": DAEMON_OBSERVED_AT,
        "last_seen": DAEMON_OBSERVED_AT,
        "active": True,
    }
    return Package(**{**fields, **overrides})


def vllm_metrics(generated: float, running: float = 0.0, waiting: float = 0.0) -> Response:
    """The recorded metrics page with the three numbers the load test reads set to these."""
    import re

    text = (FIXTURES / "vllm_metrics_before_l4.txt").read_text(encoding="utf-8")
    for name, value in (("generation_tokens_total", generated), ("num_requests_running", running), ("num_requests_waiting", waiting)):
        text = re.sub(rf"^(vllm:{name}\{{[^}}]*\}}) \S+$", rf"\g<1> {value}", text, flags=re.M)
    return Response(status=200, body=text.encode("utf-8"))


def vllm_metrics_pages(n: int) -> list[Response]:
    """Six metrics pages, before and after each of three batches, each adding n x 128 tokens."""
    pages = []
    for batch in range(3):
        before = 128.0 + batch * n * 128
        pages += [vllm_metrics(before), vllm_metrics(before + n * 128)]
    return pages


def vllm_chat_answer(finish: str | None = None):
    """A chat answer function: the recorded answer for `Request <i>.` (the warm-up for `0`)."""
    import copy
    import re

    warm_up = json.loads((FIXTURES / "vllm_chat_one_l4.json").read_text(encoding="utf-8"))
    recorded = json.loads((FIXTURES / "vllm_chat_concurrent_l4.json").read_text(encoding="utf-8"))["requests"]
    by_index = {record["index"]: record for record in recorded}

    def answer(body: dict) -> Response:
        index = int(re.search(r"Request (\d+)\.", body["messages"][0]["content"]).group(1))
        response = copy.deepcopy((warm_up if index == 0 else by_index[(index - 1) % 20 + 1])["response"])
        if finish is not None and index != 0:
            response["choices"][0]["finish_reason"] = finish
        return Response(status=200, body=json.dumps(response).encode("utf-8"))

    return answer


def vllm_servers(requests: int = 20, models: str = "vllm_v1_models_l4.json", finish: str | None = None) -> dict:
    """A machine with a vLLM server that answers every call of the start screen and one load test."""
    from modelroom.vllm_server import CHAT_PATH, METRICS_PATH, MODELS_PATH, FixtureVllmServer

    answers = {
        ("GET", MODELS_PATH): json_response(models),
        ("GET", METRICS_PATH): vllm_metrics_pages(min(requests, 64)),
        ("POST", CHAT_PATH): vllm_chat_answer(finish),
    }
    return {"vllm": FixtureVllmServer(answers)}


def qwen25_transport_mapping() -> dict[tuple[str, str], Response]:
    """The Hub answers a fetch of `Qwen/Qwen2.5-0.5B-Instruct` with that repository in `repos` needs."""
    api = "https://huggingface.co/api/models"
    return {
        ("GET", f"{api}/{QWEN25_REPO}"): json_response("hf_qwen25_05b_instruct_model.json"),
        ("GET", f"https://huggingface.co/{QWEN25_REPO}/resolve/{QWEN25_SHA}/config.json"): json_response(
            "hf_qwen25_05b_instruct_config.json"
        ),
        ("GET", f"{api}/{QWEN25_REPO}/tree/{QWEN25_SHA}?recursive=true"): json_response(
            "hf_qwen25_05b_instruct_tree.json"
        ),
        ("GET", f"{api}/{QWEN25_REPO}-GGUF"): envelope_response("hf_unsloth_does_not_exist_model.json"),
    }
