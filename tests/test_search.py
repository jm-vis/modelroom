"""Tests for modelroom.search: the Hugging Face search, resolution, age and the written config.

Network-facing code is tested against recorded fixtures only, never the live API. The pinned
search answer is `tests/fixtures/hf_search_qwen_gguf.json` (see `tests/fixtures/README.md` for
how it was recorded and why its entries are curated).
"""

from __future__ import annotations

import json
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from modelroom.catalog import Catalog, load_catalog
from modelroom.config import ConfigError, Configuration, load_config, package_targets
from modelroom.http import BudgetedTransport, RequestBudget, Response
from modelroom.search import (
    DEFAULT_PACKAGERS,
    DEFAULT_GUIDED_BUDGET,
    HF_SEARCH_LIMIT,
    NO_OLLAMA_LABEL,
    SearchError,
    apply_hits,
    decide_age,
    ollama_label,
    owner_class,
    parse_ollama_entry,
    run_search,
    search_url,
    write_configuration,
)
from modelroom.state import LockHeldError, acquire_lock, release_lock

from fixture_support import build_transport, json_response, qwen35_example_config_dict

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).resolve().parent.parent

QWEN_INFO = "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"
DEEPSEEK_REPO = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
DEEPSEEK_INFO = f"https://huggingface.co/api/models/{DEEPSEEK_REPO}"


def _catalog() -> Catalog:
    return load_catalog()


def _search_mapping(extra: dict | None = None) -> dict:
    mapping = {
        ("GET", search_url("qwen")): json_response("hf_search_qwen_gguf.json"),
        ("GET", QWEN_INFO): json_response("hf_qwen_qwen35_9b_model.json"),
        # The second resolved base model of the pinned answer: listed in the catalog, but with no
        # Ollama assignment there, and its card names no `new_version`.
        ("GET", DEEPSEEK_INFO): Response(
            status=200,
            body=json.dumps({"id": DEEPSEEK_REPO, "sha": "b" * 40, "cardData": {"license": "mit"}}).encode(
                "utf-8"
            ),
        ),
    }
    mapping.update(extra or {})
    return mapping


def _hit(outcome, repo: str):
    return next(hit for hit in outcome.hits if hit.repo == repo)


def _base_model_of(config: Configuration, hf_repo: str):
    """The configured base model with this `hf_repo`, whichever family holds it."""
    return next(
        base_model
        for family in config.families
        for base_model in family.base_models
        if base_model.hf_repo == hf_repo
    )


# --- the request itself --------------------------------------------------------------------


def test_search_url_is_the_pinned_request_with_repeated_expand():
    url = search_url("qwen")

    # The literal request, not one interpolated from the module's own constants: this is the
    # pinned request the fixture was recorded with.
    assert url == (
        "https://huggingface.co/api/models?search=qwen&filter=gguf&sort=createdAt&direction=-1"
        "&limit=50"
        "&expand=cardData&expand=createdAt&expand=safetensors&expand=tags"
    )
    assert HF_SEARCH_LIMIT == 50


def test_search_url_quotes_the_query():
    assert "search=qwen+3.5&filter=gguf" in search_url("qwen 3.5")
    assert "search=a%26b&filter=gguf" in search_url("a&b")


def test_run_search_makes_exactly_one_request_for_the_search_itself():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert transport.calls[0] == ("GET", search_url("qwen"))
    # one search, plus one age lookup per *distinct* resolved base model (two of them for
    # three resolved hits): a hit itself costs no request
    assert outcome.budget_used == 3
    assert outcome.budget_limit == 60


def test_run_search_defaults_to_the_shared_guided_budget():
    transport = build_transport(_search_mapping())

    outcome = run_search(transport, "qwen", catalog=_catalog())

    assert outcome.budget_limit == DEFAULT_GUIDED_BUDGET == 60


def test_run_search_reports_a_failing_status_instead_of_guessing():
    transport = build_transport({("GET", search_url("qwen")): Response(status=503, body=b"")})

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert "503" in str(exc.value)


def test_run_search_reports_a_body_that_is_not_a_list():
    transport = build_transport({("GET", search_url("qwen")): Response(status=200, body=b"{}")})

    with pytest.raises(SearchError):
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))


def test_run_search_reports_an_exhausted_budget_instead_of_an_empty_result():
    transport = build_transport(_search_mapping())

    with pytest.raises(SearchError) as exc:
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(0))

    assert "budget" in str(exc.value)


# --- resolution ------------------------------------------------------------------------------


def test_the_unsloth_hit_resolves_to_its_publisher_base_model():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, "unsloth/Qwen3.5-9B-GGUF")
    assert hit.resolved is True
    assert hit.resolved_base_model == "Qwen/Qwen3.5-9B"
    assert hit.base_model == ["Qwen/Qwen3.5-9B"]
    assert hit.base_model_relation == "quantized"
    assert hit.unresolved_reason is None
    assert hit.publisher_status == "listed packager"
    assert hit.license == "apache-2.0"
    assert hit.repo_created_at == datetime(2026, 2, 28, 14, 4, 29, tzinfo=timezone.utc)
    assert hit.parameters_b is None


def test_the_publishers_own_gguf_repo_is_marked_publisher_and_carries_its_parameter_count():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, "Qwen/Qwen3.5-9B-GGUF")
    assert hit.resolved is True
    assert hit.publisher_status == "publisher"
    assert hit.parameters_b == pytest.approx(9.653104368)


def test_a_finetune_hit_is_unresolved_as_a_derivative():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, "community-user/Qwen3.5-9B-Roleplay-GGUF")
    assert hit.resolved is False
    assert hit.unresolved_reason == "derivative"
    assert hit.publisher_status == "other"
    assert hit.ollama is None


@pytest.mark.parametrize(
    "repo, reason",
    [
        ("community-user/Qwen3.5-9B-Mystery-GGUF", "base_model_tag"),
        ("community-user/Qwen3.5-9B-Plain-GGUF", "relation_unknown"),
        ("community-user/Nebula-9B-GGUF", "publisher_unknown"),
        ("community-user/Qwen3.5-9B-Conflict-GGUF", "metadata_conflict"),
    ],
)
def test_every_unresolved_reason_is_reported_from_the_pinned_answer(repo, reason):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, repo)
    assert hit.resolved is False
    assert hit.unresolved_reason == reason


def test_an_unresolved_hit_inherits_neither_an_ollama_name_nor_an_age():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    for hit in outcome.hits:
        if not hit.resolved:
            assert hit.ollama is None
            assert hit.age == "unknown"
            assert hit.successor is None


def test_the_summary_line_counts_repositories_resolved_unresolved_and_the_budget():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    assert outcome.summary_line() == "8 repositories, 3 resolved, 5 unresolved, budget 3/60"


@pytest.mark.parametrize(
    "declared",
    [
        "not-a-repo-id",
        "(no single declared base)",
        "a/b/c",
        # A path segment, not a repository: the shared `hf_repo` pattern still lets these
        # through, and the declared base becomes a URL path segment of the age lookup.
        "Qwen/..",
        "Qwen/.",
        "../x",
        "Qwen/-leading-hyphen",
    ],
)
def test_a_declared_base_that_is_not_a_repo_id_is_unresolved_not_a_crash(declared):
    # The declared base comes from the registry; a value that is not a repository id (including
    # one that happens to equal the module's own "no single base" marker) must end as a reason,
    # never as a `SearchHit` validation error out of `run_search`.
    body = json.dumps(
        [
            {
                "id": "community-user/Odd-GGUF",
                "tags": [f"base_model:{declared}", f"base_model:quantized:{declared}"],
                "cardData": {},
            }
        ]
    ).encode("utf-8")
    transport = build_transport({("GET", search_url("qwen")): Response(status=200, body=body)})

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    hit = _hit(outcome, "community-user/Odd-GGUF")
    assert hit.resolved is False
    assert hit.unresolved_reason == "base_model_tag"


@pytest.mark.parametrize(
    "created", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-05:00", "not a date", 17, None]
)
def test_a_created_at_the_conversion_cannot_carry_is_unknown_not_a_crash(created):
    # `createdAt` is registry text; a value at the edge of the calendar makes the conversion to
    # UTC overflow, which must end as "unknown", never as an exception out of the search.
    body = json.dumps(
        [{"id": "community-user/Odd-GGUF", "tags": [], "cardData": {}, "createdAt": created}]
    ).encode("utf-8")
    transport = build_transport({("GET", search_url("qwen")): Response(status=200, body=body)})

    outcome = run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))

    assert _hit(outcome, "community-user/Odd-GGUF").repo_created_at is None


def test_a_hit_whose_id_is_not_a_repo_id_ends_the_search_explicitly():
    body = json.dumps([{"id": "no-owner", "tags": [], "cardData": {}}]).encode("utf-8")
    transport = build_transport({("GET", search_url("qwen")): Response(status=200, body=body)})

    with pytest.raises(SearchError):
        run_search(transport, "qwen", catalog=_catalog(), budget=RequestBudget(60))


def test_the_shipped_packager_allow_list_matches_the_example_configuration():
    example = tomllib.loads((REPO_ROOT / "modelroom.example.toml").read_text(encoding="utf-8"))

    assert list(DEFAULT_PACKAGERS) == example["packagers"]


def test_owner_class_labels_a_fetch_target_the_same_way_as_a_search_hit():
    catalog = _catalog()

    assert owner_class("Qwen", catalog=catalog) == "publisher"
    assert owner_class("unsloth", catalog=catalog) == "listed packager"
    assert owner_class("community-user", catalog=catalog) == "other"
    assert owner_class("unsloth", catalog=catalog, listed_packagers=()) == "other"


def test_an_owner_outside_both_lists_is_other_not_a_packager():
    outcome = run_search(
        build_transport(_search_mapping()),
        "qwen",
        catalog=_catalog(),
        listed_packagers=(),
        budget=RequestBudget(60),
    )

    assert _hit(outcome, "unsloth/Qwen3.5-9B-GGUF").publisher_status == "other"


# --- Ollama assignment -----------------------------------------------------------------------


def test_a_resolved_hit_takes_its_ollama_name_from_the_catalog():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    assert _hit(outcome, "unsloth/Qwen3.5-9B-GGUF").ollama == "qwen3.5:9b"


def test_a_user_entry_overrides_the_catalogs_ollama_name():
    outcome = run_search(
        build_transport(_search_mapping()),
        "qwen",
        catalog=_catalog(),
        budget=RequestBudget(60),
        ollama_entries={"unsloth/Qwen3.5-9B-GGUF": "qwen3.5:9b-q4_K_M"},
    )

    assert _hit(outcome, "unsloth/Qwen3.5-9B-GGUF").ollama == "qwen3.5:9b-q4_K_M"


def test_parse_ollama_entry_reads_the_typed_line():
    assert parse_ollama_entry("ollama: qwen3.5:9b") == "qwen3.5:9b"
    assert parse_ollama_entry("  OLLAMA:qwen3.5:9b  ") == "qwen3.5:9b"


@pytest.mark.parametrize("line", ["qwen3.5:9b", "ollama: qwen3.5", "ollama:", "ollama: a b:c"])
def test_parse_ollama_entry_rejects_anything_else(line):
    with pytest.raises(ValueError):
        parse_ollama_entry(line)


def test_a_resolved_model_the_catalog_gives_no_ollama_name_is_labeled_none_known():
    # The decisive case: the hit *resolved*, its publisher is known, and the catalog simply has no
    # Ollama assignment for that model -- nothing may be invented from the name.
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    hit = _hit(outcome, "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF")
    assert hit.resolved is True
    assert hit.resolved_base_model == DEEPSEEK_REPO
    assert hit.ollama is None
    assert ollama_label(hit) == NO_OLLAMA_LABEL


def test_a_model_without_an_assignment_is_labeled_none_known():
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )

    assert ollama_label(_hit(outcome, "unsloth/Qwen3.5-9B-GGUF")) == "qwen3.5:9b"
    assert ollama_label(_hit(outcome, "community-user/Nebula-9B-GGUF")) == NO_OLLAMA_LABEL
    assert NO_OLLAMA_LABEL == "none known"


# --- latest / legacy from positive evidence ---------------------------------------------------


def _catalog_with(models: list[dict]) -> Catalog:
    return Catalog.model_validate({"schema_version": 1, "families": [{"name": "nova", "publisher": "acme", "models": models}]})


def _plain(hf_repo: str) -> dict:
    return {"hf_repo": hf_repo, "source": f"https://huggingface.co/{hf_repo}"}


def _catalog_calling_nova_latest() -> Catalog:
    """A catalog that states `latest` for `acme/Nova-7B`, with its evidence.

    Used wherever the rule is "this is `unknown` and must *not* fall back to the catalog": with a
    silent catalog those tests would pass even if the fallback happened.
    """
    return _catalog_with(
        [
            {
                "hf_repo": "acme/Nova-7B",
                "latest": True,
                "latest_source": "https://huggingface.co/acme",
                "latest_checked": date(2026, 9, 23).isoformat(),
                "source": "https://huggingface.co/acme/Nova-7B",
            }
        ]
    )


def _info(card: dict | None, repo: str = "acme/Nova-7B") -> Response:
    """A model-info answer for `repo`, in the shape the Hub sends it (`id` included)."""
    payload: dict = {"id": repo, "sha": "a" * 40}
    if card is not None:
        payload["cardData"] = card
    return Response(status=200, body=json.dumps(payload).encode("utf-8"))


def _url(repo: str) -> str:
    return f"https://huggingface.co/api/models/{repo}"


def test_a_valid_new_version_makes_the_model_legacy_with_its_successor():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({}, "acme/Nova-7B-2512"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")


def test_a_second_valid_edge_names_the_youngest_successor_and_stays_legacy():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({"new_version": "acme/Nova-7B-2606"}, "acme/Nova-7B-2512"),
            ("GET", _url("acme/Nova-7B-2606")): _info({"new_version": "acme/Nova-7B-2701"}, "acme/Nova-7B-2606"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    # at most two edges are followed; the third jump is never made
    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2606")
    assert ("GET", _url("acme/Nova-7B-2701")) not in transport.calls


def test_a_failing_second_edge_leaves_the_model_legacy_with_the_first_successor():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({"new_version": "acme/Nova-7B-gone"}, "acme/Nova-7B-2512"),
            ("GET", _url("acme/Nova-7B-gone")): Response(status=401, body=b"{}"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")


def test_a_new_version_target_that_answers_for_another_repository_is_unknown():
    # The transport follows redirects, so a request for `acme/Nova-7B-moved` can be answered by
    # `other/Nova-7B` -- which would slip past the same-account rule. The answer has to name the
    # repository that was asked for before it counts as that repository existing.
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-moved"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-moved")): _info({}, "other/Nova-7B"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_target_that_redirects_back_to_the_model_itself_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-moved"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-moved")): _info({}, "acme/Nova-7B"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_target_whose_answer_names_no_repository_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): Response(status=200, body=b"{}"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_target_that_does_not_exist_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-gone"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-gone")): Response(status=401, body=b"{}"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_under_another_account_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({"new_version": "other/Nova-7B-2512"}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert verdict.age == "unknown"
    assert ("GET", _url("other/Nova-7B-2512")) not in transport.calls


def test_a_self_referencing_new_version_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B"}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_cycle_on_the_second_edge_keeps_the_first_successor():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({"new_version": "acme/Nova-7B"}, "acme/Nova-7B-2512"),
        }
    )

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")


def test_a_budget_that_ends_before_the_first_edge_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    budget = RequestBudget(1)
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info({}, "acme/Nova-7B-2512"),
        }
    )

    verdict = decide_age(BudgetedTransport(transport, budget), catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)
    assert budget.remaining == 0


def test_a_budget_that_ends_after_the_first_valid_edge_stays_legacy():
    # The second edge only names a younger successor; running out of budget for it must not undo
    # the `legacy` the first edge already proved.
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    budget = RequestBudget(2)
    transport = build_transport(
        {
            ("GET", _url("acme/Nova-7B")): _info({"new_version": "acme/Nova-7B-2512"}, "acme/Nova-7B"),
            ("GET", _url("acme/Nova-7B-2512")): _info(
                {"new_version": "acme/Nova-7B-2606"}, "acme/Nova-7B-2512"
            ),
            ("GET", _url("acme/Nova-7B-2606")): _info({}, "acme/Nova-7B-2606"),
        }
    )

    verdict = decide_age(BudgetedTransport(transport, budget), catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")
    assert budget.remaining == 0
    assert ("GET", _url("acme/Nova-7B-2606")) not in transport.calls


def test_a_budget_that_ends_before_the_repo_is_read_at_all_is_unknown():
    catalog = _catalog_calling_nova_latest()  # a fallback would say 'latest'
    transport = build_transport({})

    verdict = decide_age(BudgetedTransport(transport, RequestBudget(0)), catalog, "acme/Nova-7B")

    assert verdict.age == "unknown"
    assert transport.calls == []


def test_without_a_new_version_the_catalogs_latest_statement_decides():
    catalog = _catalog_with(
        [
            {
                "hf_repo": "acme/Nova-7B",
                "latest": True,
                "latest_source": "https://huggingface.co/acme",
                "latest_checked": date(2026, 9, 23).isoformat(),
                "source": "https://huggingface.co/acme/Nova-7B",
            }
        ]
    )
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("latest", None)


def test_without_a_new_version_a_catalog_successor_makes_the_model_legacy():
    catalog = _catalog_with(
        [
            {
                "hf_repo": "acme/Nova-7B",
                "successor": "acme/Nova-7B-2512",
                "source": "https://huggingface.co/acme/Nova-7B-2512",
            },
            _plain("acme/Nova-7B-2512"),
        ]
    )
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("legacy", "acme/Nova-7B-2512")


def test_a_model_the_catalog_says_nothing_about_is_unknown():
    catalog = _catalog_with([_plain("acme/Nova-7B")])
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert (verdict.age, verdict.successor) == ("unknown", None)


def test_a_new_version_that_is_not_a_string_is_no_evidence_and_the_catalog_decides():
    catalog = _catalog_with(
        [
            {
                "hf_repo": "acme/Nova-7B",
                "latest": True,
                "latest_source": "https://huggingface.co/acme",
                "latest_checked": date(2026, 9, 23).isoformat(),
                "source": "https://huggingface.co/acme/Nova-7B",
            }
        ]
    )
    transport = build_transport({("GET", _url("acme/Nova-7B")): _info({"new_version": ["acme/Nova-7B-2512"]}, "acme/Nova-7B")})

    verdict = decide_age(transport, catalog, "acme/Nova-7B")

    assert verdict.age == "latest"


# --- the resolution written back into the configuration ---------------------------------------


def _config(tmp_path) -> Configuration:
    data = qwen35_example_config_dict(str(tmp_path / "state"), str(tmp_path / "models.md"))
    data["schema_version"] = 2
    data["families"] = []
    data["publishers"] = []
    return Configuration.from_dict(data)


def _resolved_hits(tmp_path=None):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )
    return [hit for hit in outcome.hits if hit.resolved]


def test_apply_hits_writes_the_base_model_as_a_family_and_the_repo_as_an_owner_bound_target(tmp_path):
    config = _config(tmp_path)

    updated = apply_hits(config, _resolved_hits(), catalog=_catalog())

    # one family per resolved base model, named after the catalog's own family
    assert [family.name for family in updated.families] == ["qwen3.5", "deepseek-r1"]
    base_model = _base_model_of(updated, "Qwen/Qwen3.5-9B")
    assert base_model.hf_repo == "Qwen/Qwen3.5-9B"
    assert base_model.repos == ["unsloth/Qwen3.5-9B-GGUF", "Qwen/Qwen3.5-9B-GGUF"]
    assert updated.publishers == ["Qwen", "deepseek-ai"]
    assert (base_model.ollama_base, base_model.ollama_tag) == ("qwen3.5", "9b")
    assert "unsloth/Qwen3.5-9B-GGUF" in package_targets(updated, base_model)


def _hits_with_entry(entry: str | None):
    outcome = run_search(
        build_transport(_search_mapping()),
        "qwen",
        catalog=_catalog(),
        budget=RequestBudget(60),
        ollama_entries={"Qwen/Qwen3.5-9B-GGUF": entry} if entry else None,
    )
    return [hit for hit in outcome.hits if hit.resolved]


def test_a_typed_ollama_name_reaches_the_configuration_whatever_the_hit_order(tmp_path):
    # The user's own assignment belongs to the base model, so it must not depend on which hit of
    # that base model happens to be processed first, nor on the base model already being there.
    hits = _hits_with_entry("qwen3.5:9b-q4_K_M")

    forwards = apply_hits(_config(tmp_path), hits, catalog=_catalog())
    backwards = apply_hits(_config(tmp_path), list(reversed(hits)), catalog=_catalog())

    for updated in (forwards, backwards):
        base_model = _base_model_of(updated, "Qwen/Qwen3.5-9B")
        assert (base_model.ollama_base, base_model.ollama_tag) == ("qwen3.5", "9b-q4_K_M")


def test_a_typed_ollama_name_also_replaces_one_the_configuration_already_carries(tmp_path):
    first = apply_hits(_config(tmp_path), _hits_with_entry(None), catalog=_catalog())
    assert _base_model_of(first, "Qwen/Qwen3.5-9B").ollama_tag == "9b"

    second = apply_hits(first, _hits_with_entry("qwen3.5:9b-q4_K_M"), catalog=_catalog())

    assert _base_model_of(second, "Qwen/Qwen3.5-9B").ollama_tag == "9b-q4_K_M"


def test_two_typed_ollama_names_for_one_base_model_end_the_search():
    with pytest.raises(SearchError) as exc:
        run_search(
            build_transport(_search_mapping()),
            "qwen",
            catalog=_catalog(),
            budget=RequestBudget(60),
            ollama_entries={
                "unsloth/Qwen3.5-9B-GGUF": "qwen3.5:9b",
                "Qwen/Qwen3.5-9B-GGUF": "qwen3.5:9b-q4_K_M",
            },
        )

    assert "Qwen/Qwen3.5-9B" in str(exc.value)


def test_apply_hits_refuses_hits_of_one_base_model_that_name_different_ollama_packages(tmp_path):
    # `run_search` already refuses contradictory input; this is the same rule at the writing end,
    # for a caller that assembles hits itself.
    hits = _hits_with_entry(None)
    contradictory = [hits[0], hits[1].model_copy(update={"ollama": "qwen3.5:9b-q4_K_M"})]

    with pytest.raises(ConfigError) as exc:
        apply_hits(_config(tmp_path), contradictory, catalog=_catalog())

    assert "Qwen/Qwen3.5-9B" in str(exc.value)


def test_apply_hits_leaves_the_original_configuration_untouched(tmp_path):
    config = _config(tmp_path)

    apply_hits(config, _resolved_hits(), catalog=_catalog())

    assert config.families == []
    assert config.publishers == []


def test_apply_hits_is_idempotent(tmp_path):
    config = _config(tmp_path)
    hits = _resolved_hits()

    once = apply_hits(config, hits, catalog=_catalog())
    twice = apply_hits(once, hits, catalog=_catalog())

    assert twice.model_dump(mode="json") == once.model_dump(mode="json")


def test_apply_hits_ignores_unresolved_hits(tmp_path):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )
    unresolved = [hit for hit in outcome.hits if not hit.resolved]

    updated = apply_hits(_config(tmp_path), unresolved, catalog=_catalog())

    assert updated.families == []


def test_apply_hits_names_a_family_after_the_base_model_when_the_catalog_has_none(tmp_path):
    outcome = run_search(
        build_transport(_search_mapping()), "qwen", catalog=_catalog(), budget=RequestBudget(60)
    )
    hit = _hit(outcome, "unsloth/Qwen3.5-9B-GGUF")
    empty_catalog = Catalog.model_validate({"schema_version": 1, "families": []})

    updated = apply_hits(_config(tmp_path), [hit], catalog=empty_catalog)

    assert [family.name for family in updated.families] == ["qwen3.5-9b"]
    assert _base_model_of(updated, "Qwen/Qwen3.5-9B").ollama_base == "qwen3.5"


def test_write_configuration_writes_a_file_that_reads_back_as_the_same_configuration(tmp_path):
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW)

    reread = load_config(path)
    assert reread.model_dump(mode="json") == config.model_dump(mode="json")
    assert tomllib.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2


def test_write_configuration_writes_where_it_checked_even_through_a_symlink(tmp_path):
    # The check resolves the path (it has to, to know which directory confines `paths.state`), so
    # the write has to use the same resolved path: writing to the link itself would replace the
    # link with a file next to it, and the next `load_config` of the real file would then find a
    # state directory outside its own tree.
    real_dir = tmp_path / "real"
    work_dir = tmp_path / "work"
    real_dir.mkdir()
    work_dir.mkdir()
    real_path = real_dir / "modelroom.toml"
    link_path = work_dir / "modelroom.toml"
    try:
        link_path.symlink_to(Path("..") / "real" / "modelroom.toml")
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"this platform does not allow creating a symlink here: {exc}")
    real_path.write_text("", encoding="utf-8")
    data = qwen35_example_config_dict(str(real_dir / "state"), str(real_dir / "models.md"))
    data["schema_version"] = 2
    data["families"] = []
    data["publishers"] = []
    config = Configuration.from_dict(data)

    write_configuration(link_path, config, now=NOW)

    assert link_path.is_symlink(), "the link must still be a link, not a file of its own"
    assert load_config(real_path).model_dump(mode="json") == config.model_dump(mode="json")


def test_write_configuration_releases_the_lock_when_the_write_fails(tmp_path):
    # A guard that is only ever seen succeeding proves nothing about the `finally`.
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    path.mkdir()  # a directory where the file should go: os.replace cannot overwrite it

    with pytest.raises(OSError):
        write_configuration(path, config, now=NOW)

    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    release_lock(handle)


def test_write_configuration_stops_at_another_processs_lock(tmp_path):
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()
    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    try:
        with pytest.raises(LockHeldError):
            write_configuration(path, config, now=NOW)
    finally:
        release_lock(handle)

    assert not path.exists()


def test_write_configuration_refuses_a_state_directory_the_reader_would_reject(tmp_path):
    # `load_config` confines `paths.state` to the config file's own directory (F7); the writer
    # has to fail on the same rule rather than leave a file no command can read.
    data = qwen35_example_config_dict(str(tmp_path / "outside" / "state"), str(tmp_path / "models.md"))
    data["schema_version"] = 2
    data["families"] = []
    data["publishers"] = []
    config = Configuration.from_dict(data)
    path = tmp_path / "here" / "modelroom.toml"
    path.parent.mkdir()

    with pytest.raises(ConfigError):
        write_configuration(path, config, now=NOW)

    assert not path.exists()


def test_write_configuration_releases_the_lock_afterwards(tmp_path):
    config = apply_hits(_config(tmp_path), _resolved_hits(), catalog=_catalog())
    path = tmp_path / "modelroom.toml"
    (tmp_path / "state").mkdir()

    write_configuration(path, config, now=NOW)

    handle = acquire_lock(config.paths.lock_file, "fetch", NOW)
    release_lock(handle)
