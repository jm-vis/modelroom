"""Tests for `scripts/catalog_scan_rules.py`: the pure rules of the catalog scanner (AP9-KA-T).

No network, no file: every test hands a rule a value and reads the value it gives back. The two
lists the brief fixes -- the suffixes of a quantization and the Hub tasks that are never a model of
this table -- are written out here word for word, so a change to either shows up as a failing test.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if not (REPO / ".git").exists():
    pytest.skip("scripts/ is not shipped in the sdist; the scanner is tested from a clone", allow_module_level=True)
sys.path.insert(0, str(REPO / "scripts"))
rules = importlib.import_module("catalog_scan_rules")

SUFFIXES = (
    "-gguf", "-awq", "-gptq", "-mlx", "-exl2", "-onnx", "-fp8", "-fp4", "-nvfp4", "-mxfp4", "-int4", "-int8",
    "-w4a16", "-w8a8", "-4bit", "-8bit", "-bnb-4bit",
)
TAGS = ("gguf", "mlx", "awq", "gptq", "exl2", "onnx")
EXCLUDED = (
    "image-classification", "zero-shot-image-classification", "object-detection", "zero-shot-object-detection",
    "image-segmentation", "mask-generation", "keypoint-detection", "depth-estimation", "video-classification",
    "audio-classification", "voice-activity-detection", "text-classification", "token-classification",
    "zero-shot-classification", "fill-mask", "question-answering", "table-question-answering",
    "tabular-classification", "tabular-regression", "time-series-forecasting", "reinforcement-learning",
    "robotics", "graph-ml", "text-to-3d", "image-to-3d",
)
SOME_OBJECT = {"hidden_size": 1}
NOT_AN_OBJECT = (False, True, None, 42, "x", {}, [], [1])


def test_the_suffix_list_is_the_one_of_the_brief():
    assert rules.QUANTIZATION_SUFFIXES == SUFFIXES
    assert rules.QUANTIZATION_TAGS == TAGS


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_a_name_ending_on_a_suffix_is_a_quantization_in_any_case(suffix):
    assert rules.is_quantization([], f"acme/Model-7B{suffix}")
    assert rules.is_quantization([], f"acme/Model-7B{suffix.upper()}")


@pytest.mark.parametrize("name", ["acme/Foo-gguf-Instruct", "acme/int4-thing", "acme/Foo-fp8x", "acme/gptqish-7B", "acme/Foo-7B"])
def test_a_name_that_only_contains_a_suffix_is_no_quantization(name):
    assert not rules.is_quantization([], name)


@pytest.mark.parametrize("tag", TAGS)
def test_each_listed_tag_marks_a_quantization(tag):
    assert rules.is_quantization(["transformers", tag], "acme/Foo-7B")


def test_a_quantized_base_model_tag_marks_a_quantization_and_a_plain_base_tag_does_not():
    assert rules.is_quantization(["base_model:quantized:acme/Foo-7B"], "acme/Foo-7B-x")
    assert not rules.is_quantization(["base_model:acme/Foo-7B", "base_model:finetune:acme/Foo-7B"], "acme/Foo-7B-x")


def test_the_exclusion_list_is_the_one_of_the_brief():
    assert rules.EXCLUDED_PIPELINE_TAGS == EXCLUDED
    assert len(set(EXCLUDED)) == 25


def test_the_two_embedding_tasks_are_not_excluded_and_every_excluded_task_is_outside_the_kind_table():
    from modelroom.catalog import PIPELINE_TAG_TO_KIND

    assert "feature-extraction" not in rules.EXCLUDED_PIPELINE_TAGS and "sentence-similarity" not in rules.EXCLUDED_PIPELINE_TAGS
    assert not set(rules.EXCLUDED_PIPELINE_TAGS) & set(PIPELINE_TAG_TO_KIND)


def test_the_config_constants_are_the_ones_of_the_brief():
    assert rules.VISION_CONFIG_KEYS == ("vision_config", "vision_encoder")
    assert rules.AUDIO_CONFIG_KEYS == ("audio_config",)
    assert rules.TEXT_ARCHITECTURE_SUFFIX == "ForCausalLM"


@pytest.mark.parametrize("key", ["vision_config", "vision_encoder"])
def test_a_vision_key_with_an_object_decides_vision(key):
    assert rules.kind_from_config({key: SOME_OBJECT}) == ("vision", key)


def test_an_audio_key_with_an_object_decides_audio():
    assert rules.kind_from_config({"audio_config": SOME_OBJECT}) == ("audio", "audio_config")


def test_a_vision_and_an_audio_key_together_decide_nothing():
    assert rules.kind_from_config({"vision_config": SOME_OBJECT, "audio_config": SOME_OBJECT}) is None


@pytest.mark.parametrize("key", ["vision_config", "vision_encoder", "audio_config"])
@pytest.mark.parametrize("value", NOT_AN_OBJECT, ids=repr)
def test_a_modality_key_without_an_object_decides_nothing(key, value):
    assert rules.kind_from_config({key: value, "architectures": ["XForConditionalGeneration"]}) is None


def test_an_invalid_modality_key_next_to_a_valid_proof_stays_undecided():
    assert rules.kind_from_config({"vision_config": False, "audio_config": SOME_OBJECT}) is None
    assert rules.kind_from_config({"vision_config": 42, "architectures": ["XForCausalLM"]}) is None


def test_causal_architectures_decide_text_and_a_mixed_list_does_not():
    assert rules.kind_from_config({"architectures": ["XForCausalLM"]}) == ("text", "architectures")
    assert rules.kind_from_config({"architectures": ["XForCausalLM", "YForCausalLM"]}) == ("text", "architectures")
    assert rules.kind_from_config({"architectures": ["XForCausalLM", "XForConditionalGeneration"]}) is None


@pytest.mark.parametrize("config", [
    {"architectures": "XForCausalLM"}, {"architectures": ["XForCausalLM", 3]}, {"architectures": []}, {},
    {"architectures": None}, [], "x", None, 3,
])
def test_a_config_that_is_not_a_usable_object_decides_nothing(config):
    assert rules.kind_from_config(config) is None


def test_a_nested_text_config_is_not_read():
    assert rules.kind_from_config({"text_config": {"architectures": ["XForCausalLM"]}}) is None


def test_the_request_forms_are_pinned_literals():
    assert rules.list_url("acme") == (
        "https://huggingface.co/api/models?author=acme&sort=createdAt&direction=-1&limit=100"
        "&expand=cardData&expand=createdAt&expand=downloads&expand=likes&expand=pipeline_tag&expand=safetensors&expand=tags"
    )
    assert rules.b1_url("text-generation").startswith(
        "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&direction=-1&limit=100&expand=createdAt"
    )
    assert rules.b2_url("acme/Atlas-2-9B").startswith(
        "https://huggingface.co/api/models?filter=base_model%3Aquantized%3Aacme%2FAtlas-2-9B&sort=downloads&direction=-1&limit=10"
    )
    assert rules.base_url("acme/Base-9B") == "https://huggingface.co/api/models/acme/Base-9B"
    assert rules.config_url("acme/Base-9B") == "https://huggingface.co/acme/Base-9B/resolve/main/config.json"
    assert rules.collections_url("acme") == "https://huggingface.co/api/collections?owner=acme"


def test_a_next_url_is_kept_only_on_the_hub_api_page_of_the_same_account():
    good = "https://huggingface.co/api/models?author=acme&cursor=abc&limit=100"
    assert rules.next_url_ok(good, "acme")
    for bad in (
        "https://evil.example/api/models?author=acme", "http://huggingface.co/api/models?author=acme",
        "https://huggingface.co/api/datasets?author=acme", "https://huggingface.co/api/models?author=other",
        "https://huggingface.co/api/models", "https://huggingface.co:8443/api/models?author=acme", "",
    ):
        assert not rules.next_url_ok(bad, "acme"), bad


def test_the_rendering_shows_every_field_whole_and_ends_on_the_checksum_of_the_json_bytes():
    repos = [f"packone/Long-Model-Name-{i:02d}-GGUF" for i in range(30)]
    data = {
        "schema": 1, "date": "2026-10-04", "base_catalog_sha256": "b" * 64, "notes": ["packone: page error (HTTP 500), not advanced"],
        "summary": {"new": 1, "changed": 1, "candidates": 1, "incomplete": 0, "pending_publication": None},
        "changes": [
            {"n": 1, "change": "new_row", "stage": "A1", "hf_repo": "acme/Atlas-3-9B", "kind": "text", "open_since": "2026-10-04",
             "evidence": ["https://huggingface.co/acme/Atlas-3-9B"], "kind_evidence": {"stage": "pipeline_tag", "value": "text-generation"},
             "fields": {"family": {"old": None, "new": "atlas"}, "latest": {"old": None, "new": True}}},
            {"n": 2, "change": "field", "field": "gguf_repos", "stage": "A2", "hf_repo": "acme/Atlas-2-9B", "kind": "text",
             "open_since": "2026-10-01", "evidence": [], "fields": {"gguf_repos": {"old": [], "new": repos}}},
        ],
        "candidates": [{"k": "K1", "candidate": "B1", "account": "newlab", "likes": 120, "open_since": "2026-10-04", "evidence": []}],
    }
    text = rules.render_proposal(data, "c" * 64)
    assert "1 | A1 | acme/Atlas-3-9B | kind text | evidence https://huggingface.co/acme/Atlas-3-9B | open since 2026-10-04" in text
    assert "  family: null -> \"atlas\"\n" in text and "  latest: null -> true\n" in text
    assert f"  gguf_repos: [] -> {json.dumps(repos)}\n" in text
    assert "K1 | B1 | newlab | kind - | evidence  | open since 2026-10-04" in text
    assert "  likes: 120\n" in text and "packone: page error (HTTP 500), not advanced" in text
    assert text.endswith("proposal.json sha256: " + "c" * 64 + "\n")
    assert rules.render_proposal(data) == rules.render_proposal(data, None)
    assert "sha256: c" not in rules.render_proposal(data)


def test_a_change_of_one_new_value_changes_the_rendering_at_that_place_only():
    data = {
        "schema": 1, "date": "2026-10-04", "base_catalog_sha256": "b" * 64, "notes": [], "candidates": [],
        "summary": {"new": 0, "changed": 1, "candidates": 0, "incomplete": 0, "pending_publication": None},
        "changes": [{"n": 1, "change": "field", "field": "successor", "stage": "A3", "hf_repo": "acme/Atlas-2-9B", "kind": "text",
                     "open_since": "2026-10-04", "evidence": [], "fields": {"successor": {"old": None, "new": "acme/Atlas-3-9B"}}}],
    }
    before = rules.render_proposal(data).splitlines()
    data["changes"][0]["fields"]["successor"]["new"] = "acme/Atlas-4-9B"
    after = rules.render_proposal(data).splitlines()
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [before.index("  successor: null -> \"acme/Atlas-3-9B\"")]

# --- the kind of a repo, decided by a whole run -----------------------------------------------------------------------

from modelroom.catalog import PIPELINE_TAG_TO_KIND  # noqa: E402
from test_catalog_scan import (  # noqa: E402,F401  (the recorded Hub, the run and the readers of a proposal)
    BASELINE_CALLS, LATEST_TEXT_VISION, b1_entry, FIX, MISSING, NOW, TABLE, TODAY, YESTERDAY, Hub, applied, asked, b3, cands, ent, env, go, new_of, new_rows,
    stage_calls, stamp,
)
from datetime import timedelta  # noqa: E402

@pytest.mark.parametrize("task, kind", sorted(TABLE.items()))
def test_each_of_the_fifteen_tasks_gives_a_row_of_the_right_kind_with_the_task_verbatim(env, task, kind):
    hub = Hub()
    hub.page("nova", [ent("nova/Fresh-Thing-7B", stamp(1), pipeline_tag=task, tags=["safetensors", "region:us"])])
    p, _, transport = go(env, hub)
    row = new_rows(p)["nova/Fresh-Thing-7B"]
    new = new_of(row)
    assert (row["kind"], new["kind"], new["hub_pipeline_tag"], new["kind_source"]) == (kind, kind, task, None)
    assert row["kind_evidence"] == {"stage": "pipeline_tag", "value": task} and len(transport.calls) == BASELINE_CALLS
    assert applied(p).model_for("nova/Fresh-Thing-7B").kind == kind


def test_the_kind_table_of_this_suite_is_the_one_of_the_catalog():
    assert TABLE == PIPELINE_TAG_TO_KIND


# --- the kind in stages ---------------------------------------------------------------------------------------------


def test_a_task_outside_the_table_with_one_table_tag_is_decided_by_the_tags_without_a_request(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag="any-to-any", tags=["safetensors", "image-text-to-text"])])
    p, _, transport = go(env, hub)
    row = new_rows(p)["acme/Zeta-1-3B"]
    assert (row["kind"], new_of(row)["kind_source"], new_of(row)["hub_pipeline_tag"]) == ("vision", "tags", "any-to-any")
    assert row["kind_evidence"]["stage"] == "tags" and len(transport.calls) == BASELINE_CALLS and applied(p)


def test_one_base_model_with_a_table_task_decides_the_kind_with_one_request(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag=None, tags=["base_model:acme/Atlas-Base-9B"])])
    hub.put(rules.base_url("acme/Atlas-Base-9B"), (FIX / "scan_model_answer.json").read_bytes())
    p, result, transport = go(env, hub)
    row = new_rows(p)["acme/Zeta-1-3B"]
    assert (row["kind"], new_of(row)["kind_source"], new_of(row)["hub_pipeline_tag"]) == ("vision", "base_model", None)
    assert row["kind_evidence"] == {"stage": "base_model", "value": "acme/Atlas-Base-9B"}
    assert asked(transport, "/resolve/") == [] and result.sub_used == 1 and applied(p)


@pytest.mark.parametrize("redirect, calls", [(False, 1), (True, 2)])
def test_a_config_with_a_vision_object_decides_the_kind_with_one_request_and_a_redirect_costs_two(env, redirect, calls):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag=None)])
    hub.config("acme/Zeta-1-3B", (FIX / "scan_config_vision.json").read_bytes(), redirect=redirect)
    p, result, transport = go(env, hub)
    row = new_rows(p)["acme/Zeta-1-3B"]
    assert (row["kind"], new_of(row)["kind_source"]) == ("vision", "config") and row["kind_evidence"]["stage"] == "config"
    assert (result.sub_used, stage_calls(transport)) == (calls, calls) and result.main_used == len(transport.calls) and applied(p)


def test_a_causal_architecture_decides_text(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag="translation")])
    hub.config("acme/Zeta-1-3B", (FIX / "scan_config_text.json").read_bytes())
    row = new_rows(go(env, hub)[0])["acme/Zeta-1-3B"]
    assert (row["kind"], new_of(row)["kind_source"], new_of(row)["hub_pipeline_tag"]) == ("text", "config", "translation")


def test_a_task_of_the_exclusion_list_is_never_a_row_and_never_a_candidate_even_with_a_table_tag(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag="image-classification", tags=["image-text-to-text"])])
    p, _, transport = go(env, hub)
    assert p["changes"] == [] and p["candidates"] == [] and len(transport.calls) == BASELINE_CALLS


@pytest.mark.parametrize("task", [
    "image-classification", "zero-shot-image-classification", "object-detection", "zero-shot-object-detection",
    "image-segmentation", "mask-generation", "keypoint-detection", "depth-estimation", "video-classification",
    "audio-classification", "voice-activity-detection", "text-classification", "token-classification",
    "zero-shot-classification", "fill-mask", "question-answering", "table-question-answering",
    "tabular-classification", "tabular-regression", "time-series-forecasting", "reinforcement-learning",
    "robotics", "graph-ml", "text-to-3d", "image-to-3d",
])
def test_each_excluded_task_keeps_a_repo_out_without_a_request_of_the_stages(env, task):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag=task, tags=["base_model:acme/Atlas-Base-9B"])])
    p, _, transport = go(env, hub)
    assert p["changes"] == [] and p["candidates"] == [] and len(transport.calls) == BASELINE_CALLS


@pytest.mark.parametrize("task, tags", [
    (None, []), (MISSING, []), ("foo-bar", []), (None, ["text-generation", "image-text-to-text"]), ("any-to-any", ["safetensors"]),
])
def test_a_repo_without_a_decisive_proof_becomes_a_candidate_of_open_kind_and_is_not_excluded(env, task, tags):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag=task, tags=tags)])
    hub.put(rules.config_url("acme/Zeta-1-3B"), b"not found", status=404)
    p, result, _ = go(env, hub)
    (candidate,) = cands(p, "B3")
    assert candidate["hf_repo"] == "acme/Zeta-1-3B" and candidate["tags"] == tags and candidate["open_since"] == TODAY
    assert candidate["hub_pipeline_tag"] == (task if isinstance(task, str) else None) and new_rows(p) == {}
    assert candidate["kind_evidence"]["stage"] != "budget" and result.exit_code == 0


def test_two_different_bases_a_base_that_is_the_repo_itself_and_an_invalid_base_are_undecided_without_a_request(env):
    hub = Hub()
    cases = {
        "acme/Zeta-1-3B": ["base_model:acme/One-7B", "base_model:acme/Two-7B"],
        "acme/Zeta-2-3B": ["base_model:acme/Zeta-2-3B"],
        "acme/Zeta-3-3B": ["base_model:acme/no way"],
    }
    hub.page("acme", [ent(repo, stamp(1), pipeline_tag=None, tags=tags) for repo, tags in cases.items()])
    for repo in cases:
        hub.put(rules.config_url(repo), b"", status=404)
    p, _, transport = go(env, hub)
    assert [c["hf_repo"] for c in cands(p, "B3")] == sorted(cases) and asked(transport, "/api/models/") == []


def test_a_base_without_a_table_task_is_undecided_after_one_request_and_the_base_of_the_base_is_never_asked(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag=None, tags=["base_model:acme/Mid-7B"])])
    hub.put(rules.base_url("acme/Mid-7B"), ent("acme/Mid-7B", pipeline_tag=None, tags=["base_model:acme/Root-7B"]))
    hub.put(rules.config_url("acme/Zeta-1-3B"), b"", status=404)
    p, _, transport = go(env, hub)
    assert asked(transport, "/api/models/") == [rules.base_url("acme/Mid-7B")] and len(cands(p, "B3")) == 1


@pytest.mark.parametrize("answer", ["not_found", "second_redirect", "too_large"])
def test_a_config_that_cannot_be_used_leaves_the_repo_a_candidate_and_every_try_is_booked_in_both_budgets(env, answer):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag=None)])
    url = rules.config_url("acme/Zeta-1-3B")
    if answer == "not_found":
        hub.put(url, b"", status=404)
    elif answer == "second_redirect":
        hub.put(url, b"", status=307, headers={"Location": "https://huggingface.co/api/resolve-cache/a"})
        hub.put("https://huggingface.co/api/resolve-cache/a", b"", status=307, headers={"Location": "https://huggingface.co/api/resolve-cache/b"})
    else:
        hub.put(url, b"x" * (64 * 1024 + 1))
    p, result, transport = go(env, hub)
    assert len(cands(p, "B3")) == 1 and env.state()["accounts"]["acme"]["pending"] is None
    assert stage_calls(transport) == result.sub_used >= 1 and result.main_used == len(transport.calls)
    assert env.state()["accounts"]["acme"]["covered_until"] != "2026-08-30T00:00:00.000Z"  # the page was read, the account moved on


def test_the_end_of_the_run_budget_in_a_stage_makes_a_budget_candidate_and_the_read_page_still_moves_the_mark(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag=None)])
    p, result, transport = go(env, hub, budget=1)
    (candidate,) = cands(p, "B3")
    assert candidate["kind_evidence"]["stage"] == "budget" and result.exit_code == 1 and len(transport.calls) == 1
    assert (result.main_used, result.sub_used) == (1, 0)  # the refused try is booked nowhere
    assert env.state()["accounts"]["acme"]["covered_until"] == stamp(1)


def test_the_end_of_the_stage_budget_stops_the_stages_only_and_both_budgets_count_each_try(env):
    hub = Hub()
    hub.page("acme", [ent("acme/Zeta-1-3B", stamp(1), pipeline_tag=None, tags=["base_model:acme/Mid-7B"])])
    hub.put(rules.base_url("acme/Mid-7B"), ent("acme/Mid-7B", pipeline_tag=None))
    hub.config("acme/Zeta-1-3B", (FIX / "scan_config_vision.json").read_bytes(), redirect=True)
    p, result, transport = go(env, hub, sub=2)
    (candidate,) = cands(p, "B3")  # one request for the base, the first hop of the config, then the second hop is refused
    assert candidate["kind_evidence"]["stage"] == "budget" and result.sub_used == 2 and result.main_used == len(transport.calls)
    assert len(asked(transport, "resolve-cache")) == 0 and result.exit_code == 1


def test_eleven_repos_of_open_kind_on_one_page_and_an_older_open_one_all_stand_in_the_proposal_and_come_back(env):
    hub = Hub()
    repos = [f"acme/Mystery-{i:02d}-7B" for i in range(11)]
    hub.page("acme", [ent(repo, stamp(1 + i / 100), pipeline_tag=None) for i, repo in enumerate(repos)])
    for repo in repos:
        hub.put(rules.config_url(repo), b"", status=404)
    env.prev("2026-09-20", candidates=[b3("acme/Older-7B", "2026-09-20")])
    p, _, _ = go(env, hub)
    assert sorted(c["hf_repo"] for c in cands(p, "B3")) == sorted(repos + ["acme/Older-7B"]) and len(cands(p)) == 12
    assert env.state()["accounts"]["acme"]["covered_until"] == stamp(1)
    quiet = Hub()
    nxt, _, _ = go(env, quiet, now=NOW + timedelta(days=1))
    assert {c["hf_repo"]: c["open_since"] for c in cands(nxt, "B3")} == {**{r: TODAY for r in repos}, "acme/Older-7B": "2026-09-20"}


def test_a_budget_candidate_is_checked_again_before_new_repos_and_becomes_a_row_with_its_stage_and_its_old_date(env):
    hub = Hub()
    env.prev(YESTERDAY, candidates=[b3("acme/Old-7B", "2026-10-01", budget=True, tags=["base_model:acme/Atlas-Base-9B"])])
    hub.put(rules.base_url("acme/Atlas-Base-9B"), (FIX / "scan_model_answer.json").read_bytes())
    hub.page("acme", [ent("acme/Fresh-1-7B", stamp(1), pipeline_tag=None)])
    hub.put(rules.config_url("acme/Fresh-1-7B"), b"", status=404)
    p, _, transport = go(env, hub)
    row = new_rows(p)["acme/Old-7B"]
    assert (row["kind"], new_of(row)["kind_source"], row["open_since"]) == ("vision", "base_model", "2026-10-01")
    assert [b["hf_repo"] for b in cands(p, "B3")] == ["acme/Fresh-1-7B"]
    urls = [u for _, u in transport.calls]
    assert urls.index(rules.base_url("acme/Atlas-Base-9B")) < urls.index(rules.config_url("acme/Fresh-1-7B"))


def test_budget_candidates_are_checked_oldest_first_and_the_young_ones_wait_for_the_next_run(env):
    hub = Hub()
    old = [f"acme/Old{i:02d}-7B" for i in range(12)]
    since = {repo: f"2026-09-{i + 1:02d}" for i, repo in enumerate(old)}
    env.prev(YESTERDAY, candidates=[b3(r, since[r], budget=True, tags=[f"base_model:acme/Base{r[-7:-4]}"]) for r in reversed(old)])
    hub.page("acme", [ent(f"acme/New{i:02d}-3B", stamp(1), pipeline_tag=None, tags=[f"base_model:acme/NB{i:02d}"]) for i in range(10)])
    for repo in old:
        base = f"acme/Base{repo[-7:-4]}"
        hub.put(rules.base_url(base), ent(base, pipeline_tag=None))
        hub.config(repo, (FIX / "scan_config_vision.json").read_bytes(), redirect=True)
    p, _, transport = go(env, hub)
    assert asked(transport, "/api/models/") == [rules.base_url(f"acme/Base{r[-7:-4]}") for r in old[:10]]
    assert sorted(new_rows(p)) == sorted(old[:10])
    waiting = {c["hf_repo"] for c in cands(p, "B3")}
    assert waiting == set(old[10:]) | {f"acme/New{i:02d}-3B" for i in range(10)}
    assert all(c["kind_evidence"]["stage"] == "budget" for c in cands(p, "B3"))
    hub2 = Hub()  # the next run: the two that waited are checked first, then the young ones, by date and name
    for i in range(10):
        hub2.put(rules.base_url(f"acme/NB{i:02d}"), ent("acme/NB", pipeline_tag=None))
        hub2.config(f"acme/New{i:02d}-3B", (FIX / "scan_config_vision.json").read_bytes(), redirect=True)
    for repo in old[10:]:
        hub2.put(rules.base_url(f"acme/Base{repo[-7:-4]}"), ent("acme/B", pipeline_tag=None))
        hub2.config(repo, (FIX / "scan_config_vision.json").read_bytes(), redirect=True)
    _, _, second = go(env, hub2, now=NOW + timedelta(days=1))
    order = asked(second, "/api/models/")
    assert order[:2] == [rules.base_url(f"acme/Base{r[-7:-4]}") for r in old[10:]]
    assert order[2:] == [rules.base_url(f"acme/NB{i:02d}") for i in range(8)]


# --- the open pages: new publishers and new packagers -------------------------------------------------------------------

@pytest.mark.parametrize("name, entry, expected", [
    ("at the thresholds", b1_entry("newlab", 100, 1000, timedelta(days=30)), True),
    ("one like short", b1_entry("newlab", 99), False),
    ("one download short", b1_entry("newlab", downloads=999), False),
    ("thirty days and a second", b1_entry("newlab", age=timedelta(days=30, seconds=1)), False),
    ("a base model tag", b1_entry("newlab", tags=["base_model:finetune:other/Thing-7B"]), False),
    ("a quantization name", ent("newlab/Rising-1-7B-GGUF", stamp(3), likes=500, downloads=5000), False),
    ("a quantization tag", b1_entry("newlab", tags=["gguf"]), False),
    ("a listed publisher", b1_entry("acme"), False),
    ("a listed packager", b1_entry("packone"), False),
    ("a rejected account", b1_entry("spamlabs"), False),
])
def test_a_new_publisher_is_a_candidate_only_over_every_threshold_of_the_rule(env, name, entry, expected):
    hub = Hub()
    hub.put(rules.b1_url("text-generation"), [entry])
    p, _, _ = go(env, hub)
    assert [c["account"] for c in cands(p, "B1")] == (["newlab"] if expected else []), name
    assert new_rows(p) == {}


def test_a_b1_candidate_carries_its_evidence_and_never_becomes_a_row(env):
    hub = Hub()
    hub.put(rules.b1_url("text-generation"), [b1_entry("newlab", 150, 2000)])
    candidate = cands(go(env, hub)[0], "B1")[0]
    assert (candidate["k"], candidate["likes"], candidate["downloads"], candidate["open_since"]) == ("K1", 150, 2000, TODAY)
    assert candidate["repos"][0]["hf_repo"] == "newlab/Rising-1-7B" and candidate["created_at"]


def b2_hub(counts: dict[str, int], owner: str = "quantco") -> Hub:
    hub = Hub()
    for index, repo in enumerate(LATEST_TEXT_VISION):
        if counts.get(repo):
            hub.put(rules.b2_url(repo), [ent(f"{owner}/{repo.split('/')[1]}-GGUF", stamp(1), tags=["gguf", f"base_model:quantized:{repo}"], likes=40 + index)])
    return hub


def test_a_new_packager_is_a_candidate_at_three_rows_and_not_at_two_or_when_listed_or_not_gguf(env):
    full = {r: 1 for r in LATEST_TEXT_VISION}
    assert [c["account"] for c in cands(go(env, b2_hub(full))[0], "B2")] == ["quantco"]
    assert cands(go(env, b2_hub({r: 1 for r in LATEST_TEXT_VISION[:2]}))[0], "B2") == []
    assert cands(go(env, b2_hub(full, "packone"))[0], "B2") == []
    hub = b2_hub(full)
    for repo in LATEST_TEXT_VISION:
        hub.put(rules.b2_url(repo), [ent("quantco/X", stamp(1), tags=["safetensors", f"base_model:quantized:{repo}"])])
    assert cands(go(env, hub)[0], "B2") == []


def test_at_most_ten_publisher_and_packager_candidates_stand_in_a_run_but_a_candidate_not_decided_comes_back(env):
    hub = Hub()
    hub.put(rules.b1_url("text-generation"), [b1_entry(f"lab{i:02d}", likes=100 + i) for i in range(12)])
    p, _, _ = go(env, hub)
    listed = cands(p, "B1")
    assert [c["account"] for c in listed] == [f"lab{i:02d}" for i in range(11, 1, -1)] and any("over the limit of 10" in n for n in p["notes"])
    again, _, _ = go(env, Hub(), now=NOW + timedelta(days=1))
    assert {c["account"]: c["open_since"] for c in cands(again, "B1")} == {c["account"]: TODAY for c in listed}


def test_a_fine_tune_of_an_unlisted_account_appears_nowhere(env):
    hub = Hub()
    hub.put(rules.b1_url("text-generation"), [ent("other/Atlas-2-9B-Uncensored", stamp(1), likes=900, downloads=90_000, tags=["base_model:finetune:acme/Atlas-2-9B"])])
    hub.page("packone", [ent("other/Atlas-2-9B-Merge", stamp(1), tags=["base_model:merge:acme/Atlas-2-9B"])])
    p, _, _ = go(env, hub)
    assert p["changes"] == [] and p["candidates"] == []
